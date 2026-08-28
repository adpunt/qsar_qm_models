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
//! Since 2026-08-27 the fingerprint is NOT computed here. 'ECFP4' means Morgan
//! radius 2, and the only binding this side has is `rdk_fingerprint_mol`, which
//! is RDKit's PATH fingerprint — a different fingerprint, which agreed with
//! Morgan on 0 of the first 1,500 QM9 molecules and returned all zeros for
//! methane, ammonia and water (RERUN_PLAN.md §2.13). Python computes it and this
//! side carries the bytes through, so the fixtures below write the block into the
//! input record and an all-zero block is what a failed featurisation looks like.
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

/// A stand-in for the 256-byte block Python writes. Not all zeros, and different
/// per molecule, so a misaligned read cannot look correct by accident. `zero`
/// asks for the all-zero block, which is what a failed featurisation looks like
/// on this side now.
fn ecfp4_block(smiles: &str, zero: bool) -> [u8; 256] {
    let mut block = [0u8; 256];
    if zero {
        return block;
    }
    let seed = smiles.bytes().fold(7u8, |a, b| a.wrapping_mul(31).wrapping_add(b));
    for (i, byte) in block.iter_mut().enumerate() {
        *byte = seed.wrapping_add(i as u8).wrapping_mul(13) | 1;
    }
    block
}

fn write_split(dir: &Path, name: &str, file_no: usize, rows: &[(String, f32)]) {
    write_split_with(dir, name, file_no, rows, &[])
}

/// `zero_rows` names the record indices whose ECFP4 block is written all-zero.
fn write_split_with(dir: &Path, name: &str, file_no: usize, rows: &[(String, f32)],
                    zero_rows: &[usize]) {
    let path = dir.join(format!("{}_{}.mmap", name, file_no));
    let mut out: Vec<u8> = Vec::new();
    for (i, (smiles, y)) in rows.iter().enumerate() {
        let b = smiles.as_bytes();
        out.extend_from_slice(&(b.len() as u32).to_le_bytes());
        out.extend_from_slice(b);
        out.extend_from_slice(&(b.len() as u32).to_le_bytes());
        out.extend_from_slice(b);
        out.extend_from_slice(&y.to_le_bytes());
        out.extend_from_slice(&ecfp4_block(smiles, zero_rows.contains(&i)));
    }
    fs::write(path, out).unwrap();
}

struct Fixture {
    dir: PathBuf,
    file_no: usize,
    train: Vec<(String, f32)>,
    held: Vec<(String, f32)>,
    /// Training record indices whose ECFP4 block is written all-zero. `reset`
    /// re-writes the splits before every run, so it has to know about them --
    /// without this it silently un-did the zeroing and the guard had nothing to
    /// catch.
    zero_rows: Vec<usize>,
}

/// `train` is the training column; the held-out splits reuse the first molecule
/// so the fixture stays small and the arithmetic below stays readable.
fn fixture(tag: &str, train: Vec<(String, f32)>) -> Fixture {
    fixture_zeroing(tag, train, &[])
}

/// As `fixture`, with the named training records' ECFP4 block written all-zero.
fn fixture_zeroing(tag: &str, train: Vec<(String, f32)>, zero_rows: &[usize]) -> Fixture {
    let dir = std::env::temp_dir().join(format!("writer_guards_{}_{}", tag, std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let file_no = 11;

    let held: Vec<(String, f32)> = vec![("CCCCO".to_string(), 5.0), ("CCCCC".to_string(), 6.0)];

    write_split_with(&dir, "train", file_no, &train, zero_rows);
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

    Fixture { dir, file_no, train, held, zero_rows: zero_rows.to_vec() }
}

impl Fixture {
    /// The injector rewrites the mmap files in place, so a second run would
    /// otherwise read its predecessor's output.
    fn reset(&self) {
        write_split_with(&self.dir, "train", self.file_no, &self.train, &self.zero_rows);
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
            .arg("--selection-seed")
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
/// the middle has no fingerprint. Before the fix its record was written 256 bytes
/// short and every molecule after it read at the wrong offset.
///
/// A failed featurisation now arrives as an ALL-ZERO block from the Python
/// writer, which is what this side can see. (Python refuses to write one at all;
/// this is the guard for a block that reaches here anyway.)
#[test]
fn an_unfingerprintable_molecule_does_not_shorten_its_record() {
    let train = good_molecules();
    let f = fixture_zeroing("bad", train.clone(), &[4]);

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

/// CONTROL for the test above: with no zero block anywhere, nothing is reported.
#[test]
fn a_full_fingerprint_is_not_reported_as_a_failure() {
    let f = fixture("control", good_molecules());
    let out = f.run(&[]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(f.failures_csv().is_none());
}

/// A zero fingerprint is a plausible-looking feature vector. The run must not be
/// able to end quietly with one in it.
#[test]
fn a_zero_fingerprint_never_passes_silently() {
    let f = fixture_zeroing("silent", good_molecules(), &[7]);

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

    // Every other required flag IS supplied, so the refusal can only be about the
    // configuration path. Leaving one out would make this pass on the wrong flag.
    let out = Command::new(BIN)
        .current_dir(&f.dir)
        .args(["--seed", "42", "--selection-seed", "42", "--model", "rf"])
        .args(["--noise-level", "0.3"])
        .args(["--noise-shape", "gaussian", "--noise-targeting", "uniform"])
        .output()
        .unwrap();

    let said = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(
        !out.status.success(),
        "the binary must not fall back to a shared config.json"
    );
    assert!(
        said.contains("--config"),
        "it stopped, but not for the missing configuration path. It said:\n{}",
        said
    );
}

/// The randomized-SMILES field, in the two states the record format allows.
///
/// Nothing separates one molecule from the next in this file. The reader consumes
/// four bytes of length whenever `randomized_smiles` is among the representations,
/// and copes with a length of zero; the writer used to emit those four bytes only
/// when the molecule actually had one, so the two sides disagreed about the shape
/// of a record (RERUN_PLAN.md 2.19). The writer's condition now mirrors the
/// reader's exactly.
///
/// A molecule with no randomized SMILES cannot be one-hot encoded against the
/// vocabulary either. That used to be an `unwrap()`, which panicked with no
/// message and no molecule name; it now says which molecule and why. Writing an
/// all-zero row instead would put a molecule with no features into the training
/// column under its own name, which is the failure this refuses.
///
/// Nothing in the study takes either path -- the representation is refused by name
/// in `process_and_train.py`, and QM9 drops molecules with no randomized SMILES
/// before writing -- but the record stream is the one thing that cannot survive
/// misalignment.
#[test]
fn the_randomized_smiles_field_is_written_the_way_the_reader_reads_it() {
    fn write(dir: &Path, name: &str, file_no: usize, rows: &[(&str, f32, Option<&str>)]) {
        let mut out: Vec<u8> = Vec::new();
        for (smiles, y, randomized) in rows {
            let b = smiles.as_bytes();
            out.extend_from_slice(&(b.len() as u32).to_le_bytes());
            out.extend_from_slice(b);
            out.extend_from_slice(&(b.len() as u32).to_le_bytes());
            out.extend_from_slice(b);
            out.extend_from_slice(&y.to_le_bytes());
            let r = randomized.map(|s| s.as_bytes()).unwrap_or(&[]);
            out.extend_from_slice(&(r.len() as u32).to_le_bytes());
            out.extend_from_slice(r);
            out.extend_from_slice(&ecfp4_block(smiles, false));
        }
        fs::write(dir.join(format!("{}_{}.mmap", name, file_no)), out).unwrap();
    }

    /// What the reader expects a record to be, with this representation on. The
    /// one-hot block is appended after these bytes, so this is a floor, not the
    /// whole record -- a missing length prefix takes the total BELOW it.
    fn floor(rows: &[(&str, f32, Option<&str>)]) -> usize {
        rows.iter()
            .map(|(s, _, r)| {
                let n = s.len();
                4 + n + 4 + n + 4 + 4 + r.map(|x| x.len()).unwrap_or(0) + 256
            })
            .sum()
    }

    fn run(tag: &str, file_no: usize, rows: &[(&str, f32, Option<&str>)])
        -> (PathBuf, std::process::Output)
    {
        let dir = std::env::temp_dir()
            .join(format!("writer_guards_rand_{}_{}", tag, std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let held: Vec<(&str, f32, Option<&str>)> = vec![("CCCCO", 4.0, Some("OCCCC"))];
        write(&dir, "train", file_no, rows);
        write(&dir, "val", file_no, &held);
        write(&dir, "test", file_no, &held);
        let config = serde_json::json!({
            "sample_size": rows.len() + held.len() * 2,
            "noise": true,
            "train_count": rows.len(),
            "test_count": held.len(),
            "val_count": held.len(),
            "max_vocab": 30,
            "file_no": file_no,
            "molecular_representations": ["ecfp4", "randomized_smiles"],
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
        let out = Command::new(BIN)
            .current_dir(&dir)
            .args(["--seed", "42", "--selection-seed", "42"])
            .args(["--config", &format!("config_{}.json", file_no)])
            .args(["--model", "rf", "--noise-level", "0.3"])
            .args(["--noise-shape", "gaussian", "--noise-targeting", "uniform"])
            .output()
            .unwrap();
        (dir, out)
    }

    // Every molecule has one: the run completes and no record is short.
    let present: Vec<(&str, f32, Option<&str>)> = vec![
        ("CCCCO", 4.0, Some("OCCCC")),
        ("CCCCN", 4.5, Some("NCCCC")),
        ("CCCCC", 5.0, Some("CCCCC")),
        ("COCCO", 5.5, Some("OCCOC")),
    ];
    let (dir, out) = run("present", 23, &present);
    assert!(
        out.status.success(),
        "the run failed with every randomized SMILES present:\n{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let written = fs::read(dir.join("train_23.mmap")).unwrap().len();
    assert!(
        written >= floor(&present),
        "the training column is {} bytes, below the {} the reader expects before \
         the one-hot block -- a record is short and every molecule after it \
         decodes from the wrong offset",
        written,
        floor(&present)
    );
    let _ = fs::remove_dir_all(&dir);

    // One molecule has none: the run stops, and says which molecule.
    let missing: Vec<(&str, f32, Option<&str>)> = vec![
        ("CCCCO", 4.0, Some("OCCCC")),
        ("CCCCN", 4.5, Some("NCCCC")),
        ("CCCCC", 5.0, None),
        ("COCCO", 5.5, Some("OCCOC")),
    ];
    let (dir, out) = run("missing", 24, &missing);
    assert!(
        !out.status.success(),
        "a molecule with no randomized SMILES was written anyway; an all-zero \
         one-hot row would be a molecule with no features under its own name"
    );
    let said = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        said.contains("CCCCC") && said.contains("randomized SMILES"),
        "the refusal does not name the molecule or say what is missing:\n{}",
        said
    );
    let _ = fs::remove_dir_all(&dir);
}
