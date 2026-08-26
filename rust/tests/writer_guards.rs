//! Guards on the record writer. RERUN_PLAN.md §2.7, verification gate 8.
//!
//! The records in the mmap files are a packed byte stream with no delimiters and
//! no per-record length. That makes one thing fatal: a record that comes out
//! shorter than the reader expects. Every molecule after it is then decoded from
//! the wrong offset, silently, and the Python reader's bare `except: continue`
//! used to swallow the wreckage.
//!
//! The writer used to be able to produce exactly that. Inside the ECFP4 block it
//! had two `continue` statements — one for a fingerprint of the wrong width, one
//! for a molecule RDKit could not parse — and both fired *after* the rest of the
//! record was already on disk, leaving it 256 bytes short.
//!
//! These tests run the real binary over real fixtures and fail the build if that
//! comes back.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const BIN: &str = env!("CARGO_BIN_EXE_rust_processor");

/// Bytes per record, ECFP4 only: iso len + iso + canon len + canon + y_clean +
/// y_written + fingerprint. The two SMILES are length-prefixed, so this is a
/// function of the molecule.
fn expected_record_len(smiles: &str) -> usize {
    let n = smiles.len();
    4 + n + 4 + n + 4 + 4 + 256
}

fn write_split(dir: &Path, name: &str, file_no: usize, rows: &[(String, f32)]) {
    let path = dir.join(format!("{}_{}.mmap", name, file_no));
    let mut out: Vec<u8> = Vec::new();
    for (smiles, y) in rows {
        let b = smiles.as_bytes();
        out.extend_from_slice(&(b.len() as u32).to_le_bytes());
        out.extend_from_slice(b);
        out.extend_from_slice(&(b.len() as u32).to_le_bytes());
        out.extend_from_slice(b);
        out.extend_from_slice(&y.to_le_bytes());
    }
    fs::write(path, out).unwrap();
}

struct Fixture {
    dir: PathBuf,
    file_no: usize,
    train: Vec<(String, f32)>,
    held: Vec<(String, f32)>,
}

/// `train` is the training column; the held-out splits reuse the first molecule
/// so the fixture stays small and the arithmetic below stays readable.
fn fixture(tag: &str, train: Vec<(String, f32)>) -> Fixture {
    let dir = std::env::temp_dir().join(format!("writer_guards_{}_{}", tag, std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let file_no = 11;

    let held: Vec<(String, f32)> = vec![("CCCCO".to_string(), 5.0), ("CCCCC".to_string(), 6.0)];

    write_split(&dir, "train", file_no, &train);
    write_split(&dir, "val", file_no, &held);
    write_split(&dir, "test", file_no, &held);

    let config = serde_json::json!({
        "sample_size": train.len() + held.len() * 2,
        "noise": true,
        "train_count": train.len(),
        "test_count": held.len(),
        "val_count": held.len(),
        "max_vocab": 30,
        "file_no": file_no,
        "molecular_representations": ["ecfp4"],
        "k_domains": 1,
        "logging": false,
        "regression": true,
        "normalize": true,
        "uncertainty": false,
    });
    fs::write(
        dir.join(format!("config_{}.json", file_no)),
        serde_json::to_string(&config).unwrap(),
    )
    .unwrap();

    Fixture { dir, file_no, train, held }
}

impl Fixture {
    /// The injector rewrites the mmap files in place, so a second run would
    /// otherwise read its predecessor's output.
    fn reset(&self) {
        write_split(&self.dir, "train", self.file_no, &self.train);
        write_split(&self.dir, "val", self.file_no, &self.held);
        write_split(&self.dir, "test", self.file_no, &self.held);
        let _ = fs::remove_file(
            self.dir
                .join(format!("featurisation_failures_{}.csv", self.file_no)),
        );
    }

    fn run(&self, extra: &[&str]) -> std::process::Output {
        self.reset();
        let mut cmd = Command::new(BIN);
        cmd.current_dir(&self.dir)
            .arg("--seed")
            .arg("42")
            .arg("--config")
            .arg(format!("config_{}.json", self.file_no))
            .arg("--model")
            .arg("rf")
            .arg("--noise-level")
            .arg("0.3")
            .arg("--noise-shape")
            .arg("gaussian")
            .arg("--noise-targeting")
            .arg("uniform");
        for a in extra {
            cmd.arg(a);
        }
        cmd.output().unwrap()
    }

    fn train_bytes(&self) -> Vec<u8> {
        fs::read(self.dir.join(format!("train_{}.mmap", self.file_no))).unwrap()
    }

    fn failures_csv(&self) -> Option<String> {
        fs::read_to_string(
            self.dir
                .join(format!("featurisation_failures_{}.csv", self.file_no)),
        )
        .ok()
    }
}

/// Ten molecules RDKit parses, so nothing is exceptional about the fixture
/// itself; the length arithmetic below is the reference every other test uses.
fn good_molecules() -> Vec<(String, f32)> {
    // Five characters minimum: read_smiles_data rejects anything shorter, and a
    // rejected molecule never reaches the writer at all.
    let smiles = [
        "CCCCO", "CCCCC", "CCCCN", "CCCCCl", "CCCCBr", "COCCO", "CCCCCO", "CCCCCC",
        "CC(C)CO", "c1ccccc1",
    ];
    smiles
        .iter()
        .enumerate()
        .map(|(i, s)| (s.to_string(), 4.0 + i as f32 * 0.37))
        .collect()
}

#[test]
fn every_record_is_the_length_the_reader_expects() {
    let train = good_molecules();
    let f = fixture("good", train.clone());
    let out = f.run(&[]);
    assert!(
        out.status.success(),
        "binary failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let expected: usize = train.iter().map(|(s, _)| expected_record_len(s)).sum();
    assert_eq!(
        f.train_bytes().len(),
        expected,
        "the training file is not the sum of its records' expected lengths"
    );
    assert!(
        f.failures_csv().is_none(),
        "molecules that all parse must not produce a failure list"
    );
}

/// The regression test for the two deleted `continue` statements. One molecule in
/// the middle cannot be parsed by RDKit. Before the fix its record was written
/// 256 bytes short and every molecule after it read at the wrong offset.
#[test]
fn an_unfingerprintable_molecule_does_not_shorten_its_record() {
    let mut train = good_molecules();
    // Not a molecule. RDKit returns an error rather than a mol.
    train[4] = ("this-is-not-a-smiles".to_string(), 5.5);  // 20 chars: length-legal, RDKit-illegal

    let f = fixture("bad", train.clone());

    // Default behaviour: the run refuses to finish, because a zero fingerprint
    // would otherwise train as if it were real features.
    let out = f.run(&[]);
    assert!(
        !out.status.success(),
        "a molecule that cannot be fingerprinted must stop the run by default"
    );

    // ...and the file it wrote is still correctly aligned, which is the half that
    // matters if the failure is ever knowingly accepted.
    let out = f.run(&["--allow-featurisation-failures"]);
    assert!(
        out.status.success(),
        "--allow-featurisation-failures should let it finish: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let expected: usize = train.iter().map(|(s, _)| expected_record_len(s)).sum();
    assert_eq!(
        f.train_bytes().len(),
        expected,
        "the failed molecule shortened its record — the two `continue` statements are back"
    );

    let csv = f.failures_csv().expect("the failure must be listed");
    assert!(csv.starts_with("split,record_index,canonical_smiles,reason"));
    let rows: Vec<&str> = csv.lines().skip(1).filter(|l| !l.is_empty()).collect();
    assert_eq!(rows.len(), 1, "exactly one molecule failed: {:?}", rows);
    assert!(
        rows[0].starts_with("train,4,"),
        "the failure must name the split and the record index, got {}",
        rows[0]
    );
}

/// A zero fingerprint is a plausible-looking feature vector. The run must not be
/// able to end quietly with one in it.
#[test]
fn a_zero_fingerprint_never_passes_silently() {
    let mut train = good_molecules();
    train[7] = ("%%%%%%".to_string(), 5.5);
    let f = fixture("silent", train);

    let out = f.run(&[]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success());
    assert!(
        stderr.contains("could not be fingerprinted"),
        "the failure must say what happened, got: {}",
        stderr
    );
}

/// The configuration file has no default name. A caller that does not say which
/// one to read must be refused rather than falling back to a shared `config.json`
/// that a concurrent task may have just overwritten (RERUN_PLAN.md §2.8a).
#[test]
fn the_configuration_path_has_no_default() {
    let f = fixture("noconfig", good_molecules());
    fs::write(
        f.dir.join("config.json"),
        fs::read(f.dir.join(format!("config_{}.json", f.file_no))).unwrap(),
    )
    .unwrap();

    let out = Command::new(BIN)
        .current_dir(&f.dir)
        .args(["--seed", "42", "--model", "rf", "--noise-level", "0.3"])
        .args(["--noise-shape", "gaussian", "--noise-targeting", "uniform"])
        .output()
        .unwrap();

    assert!(
        !out.status.success(),
        "the binary must not fall back to a shared config.json"
    );
}
