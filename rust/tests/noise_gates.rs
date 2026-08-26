//! End-to-end gates for the redesigned noise injection.
//!
//! These run the real binary over real mmap files, and each one FAILS THE BUILD if
//! the fix it guards is removed. RERUN_PLAN.md §8 gates 3, 4, 5 and 9; the
//! standardisation order from §2.4; and the molecule-identity guard that stops the
//! class of mistake the original held-out bug belonged to.
//!
//! The cross-type gate — every noise type delivers the same dose — is gate 1, and it
//! lives in the binary's own `--self-test` mode because it needs the real label
//! column rather than a toy one. `cargo test` runs it here on a synthetic column;
//! the preflight runs it on QM9.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

const BIN: &str = env!("CARGO_BIN_EXE_rust_processor");

const N_TRAIN: usize = 400;
const N_VAL: usize = 50;
const N_TEST: usize = 50;

/// A deterministic pseudo-random stream, so the fixture is fixed without pulling a
/// dev-dependency in.
struct Lcg(u64);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

fn smiles_for(split: &str, i: usize) -> String {
    format!("C1CC{}C{}{}", split, i, "O")
}

fn write_split(dir: &Path, name: &str, file_no: usize, labels: &[(String, f32)]) {
    let path = dir.join(format!("{}_{}.mmap", name, file_no));
    let mut out: Vec<u8> = Vec::new();
    for (smiles, y) in labels {
        let b = smiles.as_bytes();
        out.extend_from_slice(&(b.len() as u32).to_le_bytes());
        out.extend_from_slice(b);
        out.extend_from_slice(&(b.len() as u32).to_le_bytes());
        out.extend_from_slice(b);
        out.extend_from_slice(&y.to_le_bytes());
    }
    fs::write(path, out).unwrap();
}

/// Read back what the injector wrote: (canonical, y_clean_raw, y_written).
fn read_written(path: &Path) -> Vec<(String, f32, f32)> {
    let bytes = fs::read(path).unwrap();
    let mut rows = Vec::new();
    let mut p = 0usize;
    while p + 4 <= bytes.len() {
        let mut take_string = |p: &mut usize| -> String {
            let len = u32::from_le_bytes(bytes[*p..*p + 4].try_into().unwrap()) as usize;
            *p += 4;
            let s = String::from_utf8(bytes[*p..*p + len].to_vec()).unwrap();
            *p += len;
            s
        };
        let _iso = take_string(&mut p);
        let canon = take_string(&mut p);
        let y_clean = f32::from_le_bytes(bytes[p..p + 4].try_into().unwrap());
        p += 4;
        let y_written = f32::from_le_bytes(bytes[p..p + 4].try_into().unwrap());
        p += 4;
        rows.push((canon, y_clean, y_written));
    }
    rows
}

struct Fixture {
    dir: PathBuf,
    file_no: usize,
    train: Vec<(String, f32)>,
}

fn fixture(tag: &str) -> Fixture {
    let dir = std::env::temp_dir().join(format!("noise_gates_{}_{}", tag, std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let file_no = 7;

    let mut rng = Lcg(20260826);
    let mut make = |split: &str, n: usize| -> Vec<(String, f32)> {
        (0..n)
            .map(|i| {
                // A label column with a real spread, so a dose is meaningful.
                let y = 5.0 + 2.5 * (rng.next_f64() as f32 - 0.5) * 2.0;
                (smiles_for(split, i), y)
            })
            .collect()
    };
    let train = make("T", N_TRAIN);
    let val = make("V", N_VAL);
    let test = make("S", N_TEST);

    write_split(&dir, "train", file_no, &train);
    write_split(&dir, "val", file_no, &val);
    write_split(&dir, "test", file_no, &test);

    let config = serde_json::json!({
        "sample_size": N_TRAIN + N_VAL + N_TEST,
        "noise": true,
        "train_count": N_TRAIN,
        "test_count": N_TEST,
        "val_count": N_VAL,
        "max_vocab": 30,
        "file_no": file_no,
        "molecular_representations": [],
        "k_domains": 1,
        "logging": false,
        "regression": true,
        "normalize": true,
        "uncertainty": false,
    });
    fs::write(dir.join("config.json"), serde_json::to_string(&config).unwrap()).unwrap();

    // Scaffold groups, keyed by canonical SMILES: twenty groups over the training
    // molecules, deliberately uneven so the affected-molecule fraction cannot be
    // assumed equal to the affected-group fraction.
    let mut groups = serde_json::Map::new();
    for (i, (smiles, _)) in train.iter().enumerate() {
        let g = if i < 200 { i % 3 } else { 3 + (i % 17) };
        groups.insert(smiles.clone(), serde_json::json!(g));
    }
    fs::write(
        dir.join(format!("scaffold_groups_{}.json", file_no)),
        serde_json::to_string(&serde_json::Value::Object(groups)).unwrap(),
    )
    .unwrap();

    Fixture { dir, file_no, train }
}

impl Fixture {
    /// Restore the pristine mmap files, since the injector rewrites them in place.
    fn reset(&self) {
        let mut rng = Lcg(20260826);
        let mut make = |split: &str, n: usize| -> Vec<(String, f32)> {
            (0..n)
                .map(|i| {
                    let y = 5.0 + 2.5 * (rng.next_f64() as f32 - 0.5) * 2.0;
                    (smiles_for(split, i), y)
                })
                .collect()
        };
        let train = make("T", N_TRAIN);
        let val = make("V", N_VAL);
        let test = make("S", N_TEST);
        write_split(&self.dir, "train", self.file_no, &train);
        write_split(&self.dir, "val", self.file_no, &val);
        write_split(&self.dir, "test", self.file_no, &test);
    }

    fn run(&self, args: &[&str]) -> std::process::Output {
        self.reset();
        let mut cmd = Command::new(BIN);
        cmd.current_dir(&self.dir)
            .arg("--seed")
            .arg("42")
            .arg("--model")
            .arg("rf");
        for a in args {
            cmd.arg(a);
        }
        cmd.output().unwrap()
    }

    fn provenance(&self) -> Vec<ProvRow> {
        let text = fs::read_to_string(
            self.dir.join(format!("noise_provenance_{}.csv", self.file_no)),
        )
        .unwrap();
        let mut rows = Vec::new();
        for (n, line) in text.lines().enumerate() {
            if n == 0 {
                assert_eq!(
                    line,
                    "split,record_index,canonical_smiles,y_clean_raw,epsilon_raw,y_noisy_raw,y_written",
                    "gate 9: the provenance header names every recorded column"
                );
                continue;
            }
            let f: Vec<&str> = line.split(',').collect();
            assert_eq!(f.len(), 7, "gate 9: every provenance column is populated");
            rows.push(ProvRow {
                split: f[0].to_string(),
                index: f[1].parse().unwrap(),
                canonical: f[2].to_string(),
                y_clean: f[3].parse().unwrap(),
                epsilon: f[4].parse().unwrap(),
                y_noisy: f[5].parse().unwrap(),
                y_written: f[6].parse().unwrap(),
            });
        }
        rows
    }

    fn manifest(&self) -> serde_json::Value {
        serde_json::from_str(
            &fs::read_to_string(self.dir.join(format!("noise_manifest_{}.json", self.file_no)))
                .unwrap(),
        )
        .unwrap()
    }
}

#[derive(Debug, Clone)]
struct ProvRow {
    split: String,
    index: usize,
    canonical: String,
    y_clean: f32,
    epsilon: f32,
    y_noisy: f32,
    y_written: f32,
}

fn ok(out: &std::process::Output) {
    assert!(
        out.status.success(),
        "injector failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

const TYPES: &[(&str, &[&str])] = &[
    ("uniform gaussian", &["--noise-targeting", "uniform", "--noise-shape", "gaussian"]),
    ("uniform student_t", &["--noise-targeting", "uniform", "--noise-shape", "student_t", "--nu", "5"]),
    ("uniform laplace", &["--noise-targeting", "uniform", "--noise-shape", "laplace"]),
    ("grouped_wide", &["--noise-targeting", "grouped_wide", "--noise-shape", "gaussian"]),
    ("grouped_shift", &["--noise-targeting", "grouped_shift", "--noise-shape", "gaussian"]),
    ("outlier", &["--noise-targeting", "outlier", "--noise-shape", "gaussian"]),
    ("censoring", &["--noise-targeting", "censoring", "--noise-shape", "gaussian"]),
];

/// Gate 4 — the recorded noise reconstructs the label exactly, every type, every
/// level. The noise is recorded where it is drawn; nothing downstream has to guess
/// it back out of a regression.
#[test]
fn recorded_noise_reconstructs_the_label() {
    let f = fixture("reconstruct");
    for (name, args) in TYPES {
        for level in ["0.2", "0.5"] {
            let mut a = args.to_vec();
            a.extend_from_slice(&["--noise-level", level]);
            ok(&f.run(&a));
            for r in f.provenance() {
                assert_eq!(
                    r.y_noisy,
                    r.y_clean + r.epsilon,
                    "gate 4: {} at {} — row {} of {} does not reconstruct",
                    name,
                    level,
                    r.index,
                    r.split
                );
            }
        }
    }
}

/// Gate 5 — zero noise records EXACTLY zero, not something small. The old pipeline
/// recovered the injected noise by fitting a line, so its zero-noise control was
/// floating-point rounding whose size grew with the label — and the control showed a
/// stronger signal than the real levels.
#[test]
fn zero_level_records_exactly_zero() {
    let f = fixture("zero");
    for (name, args) in TYPES {
        let mut a = args.to_vec();
        a.extend_from_slice(&["--noise-level", "0.0"]);
        ok(&f.run(&a));
        for r in f.provenance() {
            assert_eq!(
                r.epsilon, 0.0,
                "gate 5: {} at level 0 recorded {} for row {} of {}, not exactly zero",
                name, r.epsilon, r.index, r.split
            );
        }
    }
}

/// Gate 3 — held-out labels are untouched at every level. This is the check that
/// caught the original bug: the validation and test labels must be bit-identical
/// across the whole level grid.
#[test]
fn held_out_labels_are_bit_identical_across_levels() {
    let f = fixture("heldout");
    let mut baseline: Option<HashMap<(String, usize), u32>> = None;

    for (name, args) in TYPES {
        // Censoring's level is a censored fraction, so it runs its own grid.
        let levels: &[&str] = if *name == "censoring" {
            &["0.0", "0.1", "0.25", "0.4"]
        } else {
            &["0.0", "0.2", "0.5", "1.0"]
        };
        for level in levels {
            let mut a = args.to_vec();
            a.extend_from_slice(&["--noise-level", level]);
            ok(&f.run(&a));

            let mut held: HashMap<(String, usize), u32> = HashMap::new();
            for r in f.provenance() {
                if r.split == "train" {
                    continue;
                }
                assert_eq!(
                    r.epsilon, 0.0,
                    "gate 3: {} at {} put noise on a held-out molecule",
                    name, level
                );
                held.insert((r.split.clone(), r.index), r.y_written.to_bits());
            }
            assert!(!held.is_empty(), "gate 3: no held-out rows were recorded");

            match &baseline {
                None => baseline = Some(held),
                Some(b) => assert_eq!(
                    b, &held,
                    "gate 3: {} at {} changed a held-out label — they must be bit-identical \
                     across every noise type and every level",
                    name, level
                ),
            }
        }
    }
}

/// The standardisation constants come from the CLEAN training labels. If they came
/// from the noisy ones (the old behaviour) they would drift with the level, and the
/// same nominal amount of noise would pose a different learning problem at each one.
#[test]
fn standardisation_uses_the_clean_training_spread() {
    let f = fixture("standardise");
    let mut seen: Option<(f64, f64)> = None;
    for level in ["0.0", "0.2", "0.5", "1.0", "1.5"] {
        ok(&f.run(&[
            "--noise-targeting",
            "uniform",
            "--noise-shape",
            "gaussian",
            "--noise-level",
            level,
        ]));
        let m = f.manifest();
        let mean = m["standardisation_mean"].as_f64().unwrap();
        let sd = m["standardisation_sd"].as_f64().unwrap();
        match seen {
            None => seen = Some((mean, sd)),
            Some((m0, s0)) => {
                assert!(
                    (mean - m0).abs() < 1e-9 && (sd - s0).abs() < 1e-9,
                    "the standardisation constants moved with the noise level: \
                     ({}, {}) at level {} against ({}, {}) at level 0",
                    mean,
                    sd,
                    level,
                    m0,
                    s0
                );
            }
        }

        // and the written value really is the noisy label standardised by them
        for r in f.provenance() {
            let expected = (r.y_noisy - mean as f32) / sd as f32;
            assert!(
                (r.y_written - expected).abs() < 1e-4,
                "row {} of {} was written as {} but the clean constants give {}",
                r.index,
                r.split,
                r.y_written,
                expected
            );
        }
    }
}

/// What the provenance file says was written is what the mmap actually holds, and
/// the molecule it is attributed to is the molecule it landed on.
#[test]
fn provenance_matches_the_mmap_it_describes() {
    let f = fixture("mmap");
    ok(&f.run(&[
        "--noise-targeting",
        "uniform",
        "--noise-shape",
        "gaussian",
        "--noise-level",
        "0.5",
    ]));
    let prov = f.provenance();
    for split in ["train", "val", "test"] {
        let written = read_written(&f.dir.join(format!("{}_{}.mmap", split, f.file_no)));
        let rows: Vec<&ProvRow> = prov.iter().filter(|r| r.split == split).collect();
        assert_eq!(
            written.len(),
            rows.len(),
            "{}: the provenance describes {} rows but the file holds {}",
            split,
            rows.len(),
            written.len()
        );
        for (r, (canon, y_clean, y_written)) in rows.iter().zip(written.iter()) {
            assert_eq!(&r.canonical, canon, "{}: molecule mismatch at row {}", split, r.index);
            assert_eq!(r.y_clean, *y_clean, "{}: clean label mismatch at row {}", split, r.index);
            assert_eq!(r.y_written, *y_written, "{}: written label mismatch at row {}", split, r.index);
        }
    }
}

/// Every dose-matched type runs, records what it delivered, and lands inside the
/// band its own effective sample size allows.
///
/// Four hundred molecules — and, for the shifted grouped type, twenty scaffold groups
/// — cannot pin a dose tightly, so this fixture can only catch gross errors. The
/// tight form of gate 1 is `gate_one_dose_is_flat_across_types` below, which runs the
/// injector's own self-test over a hundred thousand labels.
#[test]
fn every_dose_matched_type_records_what_it_delivered() {
    let f = fixture("dose");
    let level = 0.5f64;
    let mut covered = 0usize;
    for (name, args) in TYPES {
        if *name == "censoring" {
            continue;
        }
        let mut a = args.to_vec();
        a.extend_from_slice(&["--noise-level", "0.5"]);
        ok(&f.run(&a));
        let m = f.manifest();
        let sd = m["clean_label_sd"].as_f64().unwrap();
        let got = m["delivered_dose_in_label_units"].as_f64().unwrap();
        let n_eff = m["effective_n"].as_f64().unwrap();
        let target = level * sd;
        // four standard errors at Gaussian kurtosis; the injector applies the exact
        // band using the sample kurtosis and dies if it is missed
        let band = (4.0 * (2.0 / (4.0 * n_eff)).sqrt()).max(0.01);
        assert!(
            ((got / target) - 1.0).abs() < band,
            "gate 1: {} delivered {:.6} against a target of {:.6} with {:.0} effective observations",
            name,
            got,
            target,
            n_eff
        );
        assert!(
            m["affected_molecule_fraction"].as_f64().unwrap() > 0.0,
            "{} affected no molecules",
            name
        );
        covered += 1;
    }
    assert!(covered >= 6, "the gate must cover every dose-matched type");
}

/// Gate 1, in the form that actually proves the confound is gone: at one target, on
/// a label column large enough to pin the dose, every noise type must deliver the
/// same amount. If this fails, the whole re-run is confounded and worthless.
///
/// The injector runs this itself and exits non-zero on any failure, so the same
/// command is the preflight gate before any cluster time is spent.
#[test]
fn gate_one_dose_is_flat_across_types() {
    let dir = std::env::temp_dir().join(format!("noise_gate_one_{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();

    // A hundred thousand labels with a realistic spread, plus scaffold groups with
    // the lopsided sizes real Murcko groups have.
    let labels_path = dir.join("labels.csv");
    let groups_path = dir.join("groups.json");
    let n = 100_000usize;
    let mut rng = Lcg(31337);
    let mut labels = fs::File::create(&labels_path).unwrap();
    let mut groups = serde_json::Map::new();
    for i in 0..n {
        // Box-Muller, so the column is normal rather than uniform
        let u1 = rng.next_f64().max(1e-12);
        let u2 = rng.next_f64();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let y = 6.8 + 1.29 * z;
        let smiles = format!("C1CC{}CO", i);
        writeln!(labels, "{},{:.6}", smiles, y).unwrap();
        // 3,000 groups, heavily skewed: a tenth of them hold half the molecules
        let g = if i % 2 == 0 { i % 300 } else { 300 + (i % 2700) };
        groups.insert(smiles, serde_json::json!(g));
    }
    labels.flush().unwrap();
    fs::write(&groups_path, serde_json::to_string(&serde_json::Value::Object(groups)).unwrap())
        .unwrap();

    let out = Command::new(BIN)
        .arg("--self-test")
        .arg(&labels_path)
        .arg("--scaffold-file")
        .arg(&groups_path)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "gate 1 failed — the noise types do not deliver the same dose:\n{}\n{}",
        stdout,
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(stdout.contains("grouped_wide"), "the grouped types were not covered:\n{}", stdout);
    assert!(stdout.contains("grouped_shift"), "the shifted grouped type was not covered:\n{}", stdout);
    let _ = fs::remove_dir_all(&dir);
}

/// The one parameter the design says must be measured rather than assumed: with
/// real scaffold groups, which are very unevenly sized, the fraction of MOLECULES in
/// a fifth of the GROUPS is not a fifth of the molecules.
#[test]
fn grouped_wide_measures_the_affected_molecule_fraction() {
    let f = fixture("grouped");
    ok(&f.run(&[
        "--noise-targeting",
        "grouped_wide",
        "--noise-shape",
        "gaussian",
        "--noise-level",
        "0.5",
        "--group-fraction",
        "0.2",
    ]));
    let m = f.manifest();
    let affected = m["affected_molecule_fraction"].as_f64().unwrap();
    let requested = m["parameters"]["requested_group_fraction"].as_f64().unwrap();
    assert!(affected > 0.0 && affected < 1.0, "affected fraction {} is degenerate", affected);
    assert!(
        (affected - requested).abs() > 1e-6,
        "the fixture's groups are uneven, so the affected MOLECULE fraction ({}) must not \
         equal the requested GROUP fraction ({}) — if these are equal the code is assuming \
         rather than measuring",
        affected,
        requested
    );
}

/// Censoring is one-directional. It is the only type that biases labels rather than
/// scattering them, so its mean shift must be non-zero and must point the right way.
#[test]
fn censoring_is_one_directional() {
    let f = fixture("censor");
    ok(&f.run(&[
        "--noise-targeting",
        "censoring",
        "--noise-level",
        "0.25",
        "--censor-side",
        "upper",
    ]));
    let m = f.manifest();
    assert!(
        m["mean_epsilon"].as_f64().unwrap() < 0.0,
        "upper censoring must push labels down"
    );
    let clipped = m["affected_molecule_fraction"].as_f64().unwrap();
    assert!((clipped - 0.25).abs() < 0.02, "clipped {:.3}, asked for 0.25", clipped);

    ok(&f.run(&[
        "--noise-targeting",
        "censoring",
        "--noise-level",
        "0.25",
        "--censor-side",
        "lower",
    ]));
    assert!(
        f.manifest()["mean_epsilon"].as_f64().unwrap() > 0.0,
        "lower censoring must push labels up"
    );
}

/// A Student-t with two or fewer degrees of freedom has undefined variance, so its
/// dose cannot be matched and the run would be silently meaningless. It is refused.
#[test]
fn student_t_refuses_undefined_variance() {
    let f = fixture("nu");
    for nu in ["2", "1.5", "0.5"] {
        let out = f.run(&[
            "--noise-targeting",
            "uniform",
            "--noise-shape",
            "student_t",
            "--nu",
            nu,
            "--noise-level",
            "0.5",
        ]);
        assert!(!out.status.success(), "nu = {} should have been refused", nu);
    }
}

/// A scaffold file that does not describe this split is refused rather than quietly
/// treated as a set of singleton groups, which would turn grouped noise into uniform
/// noise while still calling itself grouped.
#[test]
fn a_mismatched_scaffold_file_is_refused() {
    let f = fixture("badscaffold");
    let mut wrong = serde_json::Map::new();
    for i in 0..10 {
        wrong.insert(format!("NOTAMOLECULE{}", i), serde_json::json!(i));
    }
    let path = f.dir.join("wrong_groups.json");
    fs::write(&path, serde_json::to_string(&serde_json::Value::Object(wrong)).unwrap()).unwrap();

    let out = f.run(&[
        "--noise-targeting",
        "grouped_wide",
        "--noise-level",
        "0.5",
        "--scaffold-file",
        path.to_str().unwrap(),
    ]);
    assert!(!out.status.success(), "a mismatched scaffold file must stop the run");
}

/// A record stream that goes short is caught, not trained on.
///
/// `read_smiles_data` rejects a malformed SMILES *after* consuming part of the
/// record, so one bad record desynchronises everything after it. The noise plan is
/// built in one pass over the training file and applied in a second, and the two
/// passes must see the same molecules in the same order — so a short read has to
/// stop the run rather than silently shift every molecule's noise by one.
///
/// The molecule-identity assertion in the write path is the second half of this
/// guard: it compares the SMILES the noise was drawn for against the SMILES it is
/// about to land on, row by row.
#[test]
fn a_short_record_stream_is_caught() {
    let f = fixture("shortstream");
    f.reset();

    // Corrupt one training record in the middle: a SMILES too short to be valid.
    let path = f.dir.join(format!("train_{}.mmap", f.file_no));
    let bytes = fs::read(&path).unwrap();
    let mut out: Vec<u8> = Vec::new();
    let mut p = 0usize;
    let mut record = 0usize;
    while p < bytes.len() {
        let start = p;
        let iso_len = u32::from_le_bytes(bytes[p..p + 4].try_into().unwrap()) as usize;
        p += 4 + iso_len;
        let can_len = u32::from_le_bytes(bytes[p..p + 4].try_into().unwrap()) as usize;
        p += 4 + can_len;
        p += 4;
        if record == N_TRAIN / 2 {
            let bad = b"CC";
            out.extend_from_slice(&(bad.len() as u32).to_le_bytes());
            out.extend_from_slice(bad);
            out.extend_from_slice(&bytes[start + 4 + iso_len..p]);
        } else {
            out.extend_from_slice(&bytes[start..p]);
        }
        record += 1;
    }
    fs::write(&path, out).unwrap();

    let out = Command::new(BIN)
        .current_dir(&f.dir)
        .args(["--seed", "42", "--model", "rf"])
        .args(["--noise-targeting", "uniform", "--noise-level", "0.5"])
        .output()
        .unwrap();
    assert!(
        !out.status.success(),
        "a training stream that goes short must stop the run, not shift every molecule's noise"
    );
}

/// The run-level manifest carries every provenance column the results rows need
/// (RERUN_PLAN.md §5.2). A missing column here means a figure that cannot be traced
/// back to the amount of noise that produced it.
#[test]
fn the_manifest_carries_every_provenance_column() {
    let f = fixture("manifest");
    ok(&f.run(&[
        "--noise-targeting",
        "outlier",
        "--noise-shape",
        "gaussian",
        "--noise-level",
        "0.5",
    ]));
    let m = f.manifest();
    for key in [
        "noise_type",
        "noise_shape",
        "noise_targeting",
        "noise_level",
        "unit_dose",
        "solved_scale",
        "target_dose_in_label_units",
        "delivered_dose_in_label_units",
        "delivered_dose_as_fraction_of_label_spread",
        "mean_epsilon",
        "affected_molecule_fraction",
        "standardisation_mean",
        "standardisation_sd",
        "clean_label_mean",
        "clean_label_sd",
        "seed",
        "n_train",
        "parameters",
    ] {
        assert!(!m[key].is_null(), "the manifest is missing '{}'", key);
    }
}

/// The same level, the same seed, twice: identical noise. Replicates are only
/// exchangeable if they are reproducible.
#[test]
fn the_same_seed_draws_the_same_noise() {
    let f = fixture("repro");
    let args = [
        "--noise-targeting",
        "grouped_shift",
        "--noise-shape",
        "gaussian",
        "--noise-level",
        "0.5",
    ];
    ok(&f.run(&args));
    let first: Vec<f32> = f.provenance().iter().map(|r| r.epsilon).collect();
    ok(&f.run(&args));
    let second: Vec<f32> = f.provenance().iter().map(|r| r.epsilon).collect();
    assert_eq!(first, second, "the same seed must draw the same noise");
    assert!(first.iter().any(|e| *e != 0.0), "the fixture drew no noise at all");
}
