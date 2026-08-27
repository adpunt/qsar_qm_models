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

use std::collections::{HashMap, HashSet};
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
    /// How much wider the VALIDATION column's own spread is than the training
    /// column's. 1.0 for every test that does not care; the anchoring test sets it
    /// to something far from 1 so a dose taken from the wrong column cannot be
    /// mistaken for sampling noise.
    val_scale: f32,
}

/// The three splits, regenerated identically every time — the injector rewrites the
/// mmap files in place, so every run needs the pristine column back.
fn make_splits(val_scale: f32) -> (Vec<(String, f32)>, Vec<(String, f32)>, Vec<(String, f32)>) {
    let mut rng = Lcg(20260826);
    let mut make = |split: &str, n: usize, scale: f32| -> Vec<(String, f32)> {
        (0..n)
            .map(|i| {
                // A label column with a real spread, so a dose is meaningful.
                let y = 5.0 + scale * 2.5 * (rng.next_f64() as f32 - 0.5) * 2.0;
                (smiles_for(split, i), y)
            })
            .collect()
    };
    let train = make("T", N_TRAIN, 1.0);
    let val = make("V", N_VAL, val_scale);
    let test = make("S", N_TEST, 1.0);
    (train, val, test)
}

fn fixture(tag: &str) -> Fixture {
    fixture_with_val_scale(tag, 1.0)
}

/// Every experiment in this project uses a SCAFFOLD split, which holds whole
/// scaffold groups out — so a held-out molecule is in a group no training molecule
/// is in. The ordinary fixture puts held-out molecules in the same twenty groups as
/// training, which is a random split, and that is why a grouped condition passed
/// every test here and then died on the first real run.
fn fixture_scaffold_split(tag: &str) -> Fixture {
    let f = fixture_with_val_scale(tag, 1.0);
    let (train, val, test) = make_splits(1.0);
    let mut groups = serde_json::Map::new();
    for (i, (smiles, _)) in train.iter().enumerate() {
        let g = if i < 200 { i % 3 } else { 3 + (i % 17) };
        groups.insert(smiles.clone(), serde_json::json!(g));
    }
    // Group ids starting at 1000: disjoint from every training group by construction.
    for (i, (smiles, _)) in val.iter().chain(test.iter()).enumerate() {
        groups.insert(smiles.clone(), serde_json::json!(1000 + (i % 20)));
    }
    fs::write(
        f.dir.join(format!("scaffold_groups_{}.json", f.file_no)),
        serde_json::to_string(&serde_json::Value::Object(groups)).unwrap(),
    )
    .unwrap();
    f
}

fn fixture_with_val_scale(tag: &str, val_scale: f32) -> Fixture {
    let dir = std::env::temp_dir().join(format!("noise_gates_{}_{}", tag, std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let file_no = 7;

    let (train, val, test) = make_splits(val_scale);

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
    // Named per task, not "config.json". The binary has no default for --config:
    // a shared fixed name is what let concurrent array tasks read each other's
    // configuration and rewrite each other's training data (RERUN_PLAN.md §2.8a).
    fs::write(
        dir.join(format!("config_{}.json", file_no)),
        serde_json::to_string(&config).unwrap(),
    )
    .unwrap();

    // Scaffold groups, keyed by canonical SMILES: twenty groups over the training
    // molecules, deliberately uneven so the affected-molecule fraction cannot be
    // assumed equal to the affected-group fraction.
    // The held-out splits are in the file too. They have to be: a held-out molecule's
    // `noise_pattern_raw` is the shape its scaffold group receives, and there is
    // nothing to look up if the group assignment stops at the training set.
    let mut groups = serde_json::Map::new();
    for (i, (smiles, _)) in train.iter().enumerate() {
        let g = if i < 200 { i % 3 } else { 3 + (i % 17) };
        groups.insert(smiles.clone(), serde_json::json!(g));
    }
    for (i, (smiles, _)) in val.iter().chain(test.iter()).enumerate() {
        // Held-out molecules land in the SAME twenty groups, so some of them fall in
        // an affected group and some do not.
        groups.insert(smiles.clone(), serde_json::json!(i % 20));
    }
    fs::write(
        dir.join(format!("scaffold_groups_{}.json", file_no)),
        serde_json::to_string(&serde_json::Value::Object(groups)).unwrap(),
    )
    .unwrap();

    Fixture { dir, file_no, train, val_scale }
}

impl Fixture {
    /// Restore the pristine mmap files, since the injector rewrites them in place.
    fn reset(&self) {
        let (train, val, test) = make_splits(self.val_scale);
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
            .arg("--config")
            .arg(format!("config_{}.json", self.file_no))
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
                    "split,record_index,canonical_smiles,y_clean_raw,epsilon_raw,noise_scale_raw,\
noise_pattern_raw,y_noisy_raw,y_written",
                    "gate 9: the provenance header names every recorded column"
                );
                continue;
            }
            let f: Vec<&str> = line.split(',').collect();
            assert_eq!(f.len(), 9, "gate 9: every provenance column is populated");
            rows.push(ProvRow {
                split: f[0].to_string(),
                index: f[1].parse().unwrap(),
                canonical: f[2].to_string(),
                y_clean: f[3].parse().unwrap(),
                epsilon: f[4].parse().unwrap(),
                noise_scale: f[5].parse().unwrap(),
                noise_pattern: f[6].parse().unwrap(),
                y_noisy: f[7].parse().unwrap(),
                y_written: f[8].parse().unwrap(),
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
    noise_scale: f32,
    noise_pattern: f32,
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

/// Gate 3 — the TEST labels are untouched at every level. This is the check that
/// caught the original bug: the test labels must be bit-identical across the whole
/// level grid, and the test split is never noised on any flag.
///
/// Validation is deliberately NOT in this check any more. Decision 3, settled
/// 2026-08-26: training noisy, validation noisy from a separate draw, test clean.
/// `validation_carries_its_own_independent_noise` below is validation's gate, and
/// `clean_validation_restores_untouched_validation` covers the flag that turns it off.
#[test]
fn test_labels_are_bit_identical_across_levels() {
    let f = fixture("heldout");
    let mut baseline: Option<HashMap<usize, u32>> = None;

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

            let mut held: HashMap<usize, u32> = HashMap::new();
            for r in f.provenance() {
                if r.split != "test" {
                    continue;
                }
                assert_eq!(
                    r.epsilon, 0.0,
                    "gate 3: {} at {} put noise on a test molecule",
                    name, level
                );
                assert_eq!(
                    r.noise_scale, 0.0,
                    "gate 3: {} at {} recorded a non-zero applied amount on a test molecule",
                    name, level
                );
                held.insert(r.index, r.y_written.to_bits());
            }
            assert!(!held.is_empty(), "gate 3: no test rows were recorded");

            match &baseline {
                None => baseline = Some(held),
                Some(b) => assert_eq!(
                    b, &held,
                    "gate 3: {} at {} changed a test label — they must be bit-identical \
                     across every noise type and every level",
                    name, level
                ),
            }
        }
    }
}

/// Decision 3 (2026-08-26): the validation labels carry their own noise, from a
/// separate draw. The test split stays clean.
///
/// Four things have to hold at once, and each one fails differently:
///   * validation labels actually move, on every noise type;
///   * they move by a DIFFERENT amount than the training molecule at the same row —
///     an independent draw, not a copy of the training plan indexed by position,
///     which is the shape of the original held-out bug;
///   * the test labels do not move at all;
///   * validation and training receive the same AMOUNT, in absolute label units.
#[test]
fn validation_carries_its_own_independent_noise() {
    let f = fixture("valnoise");
    for (name, args) in TYPES {
        let level = if *name == "censoring" { "0.25" } else { "0.5" };
        let mut a = args.to_vec();
        a.extend_from_slice(&["--noise-level", level]);
        ok(&f.run(&a));

        let prov = f.provenance();
        let val: Vec<&ProvRow> = prov.iter().filter(|r| r.split == "val").collect();
        let train: Vec<&ProvRow> = prov.iter().filter(|r| r.split == "train").collect();
        let test: Vec<&ProvRow> = prov.iter().filter(|r| r.split == "test").collect();
        assert_eq!(val.len(), N_VAL);
        assert_eq!(test.len(), N_TEST);

        assert!(
            val.iter().any(|r| r.epsilon != 0.0),
            "{}: no validation label was noised — validation carries its own noise now",
            name
        );
        assert!(
            val.iter().all(|r| r.y_noisy == r.y_clean + r.epsilon),
            "{}: a validation row does not reconstruct from its recorded noise",
            name
        );
        assert!(
            test.iter().all(|r| r.epsilon == 0.0 && r.noise_scale == 0.0),
            "{}: the test split was noised — it never is",
            name
        );

        // An independent draw, not the training plan read at the same row position.
        let shared: usize = val
            .iter()
            .zip(train.iter())
            .filter(|(v, t)| v.epsilon == t.epsilon && v.epsilon != 0.0)
            .count();
        assert_eq!(
            shared, 0,
            "{}: {} validation rows carry exactly the training row's noise at the same \
             position — the validation draw is not independent",
            name, shared
        );

        // and the same amount, in absolute label units
        if *name != "censoring" {
            let rms = |rows: &[&ProvRow]| -> f64 {
                (rows.iter().map(|r| (r.epsilon as f64).powi(2)).sum::<f64>()
                    / rows.len() as f64)
                    .sqrt()
            };
            let (dv, dt) = (rms(&val), rms(&train));
            // A loose band on purpose: fifty validation molecules cannot pin a dose,
            // and the tight, tolerance-derived form of this check is the binary's own
            // gate, which refuses the run before it gets here.
            assert!(
                (dv / dt - 1.0).abs() < 0.6,
                "{}: validation received {:.6} label units against training's {:.6}",
                name,
                dv,
                dt
            );
        }
    }
}

/// The validation dose is anchored on the CLEAN TRAINING spread, not on the
/// validation column's own.
///
/// The fixture's validation labels have five times the training spread. Dosing them
/// against their own spread — the obvious way to write it — would deliver five times
/// as much noise to validation while both splits still called it "level 0.5", and no
/// amount of sampling noise can look like that.
#[test]
fn the_validation_dose_is_anchored_on_the_training_spread() {
    let f = fixture_with_val_scale("valanchor", 5.0);
    ok(&f.run(&[
        "--noise-targeting",
        "uniform",
        "--noise-shape",
        "gaussian",
        "--noise-level",
        "0.5",
    ]));
    let prov = f.provenance();
    let sd = |rows: &[&ProvRow]| -> f64 {
        let m = rows.iter().map(|r| r.y_clean as f64).sum::<f64>() / rows.len() as f64;
        (rows.iter().map(|r| (r.y_clean as f64 - m).powi(2)).sum::<f64>() / rows.len() as f64)
            .sqrt()
    };
    let rms = |rows: &[&ProvRow]| -> f64 {
        (rows.iter().map(|r| (r.epsilon as f64).powi(2)).sum::<f64>() / rows.len() as f64).sqrt()
    };
    let val: Vec<&ProvRow> = prov.iter().filter(|r| r.split == "val").collect();
    let train: Vec<&ProvRow> = prov.iter().filter(|r| r.split == "train").collect();

    let train_sd = sd(&train);
    let val_sd = sd(&val);
    assert!(
        val_sd / train_sd > 3.0,
        "the fixture is meant to give validation a much wider column: {:.4} against {:.4}",
        val_sd,
        train_sd
    );

    let delivered = rms(&val);
    let anchored = 0.5 * train_sd; // what the training spread asks for
    let unanchored = 0.5 * val_sd; // what validation's own spread would have given
    assert!(
        (delivered / anchored - 1.0).abs() < 0.35,
        "validation received {:.6}; the clean training spread asks for {:.6} and its own \
         spread would have given {:.6}",
        delivered,
        anchored,
        unanchored
    );
    assert!(
        (delivered / unanchored - 1.0).abs() > 0.5,
        "validation received {:.6}, which is what its OWN spread would give ({:.6}) — the \
         dose is not anchored on the training column",
        delivered,
        unanchored
    );
}

/// The flag that puts the old behaviour back, for anyone who needs to reproduce a
/// pre-2026-08-26 run.
#[test]
fn clean_validation_restores_untouched_validation() {
    let f = fixture("cleanval");
    ok(&f.run(&[
        "--noise-targeting",
        "outlier",
        "--noise-shape",
        "gaussian",
        "--noise-level",
        "0.5",
        "--clean-validation",
    ]));
    let prov = f.provenance();
    let val: Vec<&ProvRow> = prov.iter().filter(|r| r.split == "val").collect();
    assert_eq!(val.len(), N_VAL);
    assert!(
        val.iter().all(|r| r.epsilon == 0.0 && r.noise_scale == 0.0),
        "--clean-validation must leave the validation labels untouched"
    );
    // ...and the shape column survives, because the analysis needs it on every split
    assert!(
        val.iter().all(|r| r.noise_pattern.is_finite()),
        "the level-free shape must still be recorded on a clean split"
    );
    assert!(
        prov.iter().filter(|r| r.split == "train").any(|r| r.epsilon != 0.0),
        "the flag must only touch validation"
    );
    assert!(
        f.manifest()["validation_noised"].as_bool() == Some(false),
        "the manifest must record that validation was left clean"
    );
}

/// The level-free shape is what makes the zero-level subtraction possible, so it has
/// to be the SAME COLUMN at every level — bit for bit, zero included.
///
/// If it moved with the level, the zero-level correlation being subtracted off would
/// be a correlation against a different quantity, and the confound would not cancel.
#[test]
fn the_noise_shape_is_bit_identical_across_levels_including_zero() {
    let f = fixture("shape");
    for (name, args) in TYPES {
        let levels: &[&str] = if *name == "censoring" {
            &["0.0", "0.1", "0.25", "0.4"]
        } else {
            &["0.0", "0.2", "0.5", "1.0"]
        };
        let mut baseline: Option<HashMap<(String, usize), u32>> = None;
        for level in levels {
            let mut a = args.to_vec();
            a.extend_from_slice(&["--noise-level", level]);
            ok(&f.run(&a));
            let shape: HashMap<(String, usize), u32> = f
                .provenance()
                .iter()
                .map(|r| ((r.split.clone(), r.index), r.noise_pattern.to_bits()))
                .collect();
            assert!(!shape.is_empty());
            match &baseline {
                None => baseline = Some(shape),
                Some(b) => assert_eq!(
                    b, &shape,
                    "{}: the level-free shape changed between level {} and the first level \
                     — it must not depend on the level at all",
                    name, level
                ),
            }
        }
    }
}

/// At level zero nothing is applied, to the bit, on any split — and the shape column
/// still carries the condition's structure, which is the whole reason the zero level
/// is worth running.
#[test]
fn the_zero_level_applies_nothing_but_still_records_the_shape() {
    let f = fixture("zeroshape");
    for (name, args, expect_varying_shape) in [
        ("grouped_wide", &["--noise-targeting", "grouped_wide"][..], true),
        ("outlier", &["--noise-targeting", "outlier"][..], true),
        ("censoring", &["--noise-targeting", "censoring"][..], true),
        ("uniform", &["--noise-targeting", "uniform"][..], false),
    ] {
        let mut a = args.to_vec();
        a.extend_from_slice(&["--noise-level", "0.0"]);
        ok(&f.run(&a));
        let prov = f.provenance();
        assert!(
            prov.iter().all(|r| r.epsilon == 0.0 && r.noise_scale == 0.0),
            "{}: level zero must apply exactly nothing, on every split",
            name
        );
        let train: Vec<f32> = prov
            .iter()
            .filter(|r| r.split == "train")
            .map(|r| r.noise_pattern)
            .collect();
        let varies = train.iter().any(|p| *p != train[0]);
        assert_eq!(
            varies, expect_varying_shape,
            "{}: at level zero the recorded shape {} between molecules",
            name,
            if varies { "varies" } else { "is flat" }
        );
    }
}

/// The applied amount is the level times the level-free shape, exactly. Downstream
/// treats the two as interchangeable on any ranking statistic, which is only true if
/// this identity holds.
#[test]
fn the_applied_amount_is_the_level_times_the_shape() {
    let f = fixture("scaleshape");
    for (name, args) in TYPES {
        if *name == "censoring" {
            continue; // censoring has no dose axis; its amount is the shift itself
        }
        for level in [0.2f32, 0.5, 1.0] {
            let mut a = args.to_vec();
            let level_s = format!("{}", level);
            a.extend_from_slice(&["--noise-level", &level_s]);
            ok(&f.run(&a));
            for r in f.provenance() {
                if r.split != "train" && r.split != "val" {
                    continue;
                }
                let expected = level * r.noise_pattern;
                assert!(
                    (r.noise_scale - expected).abs() <= expected.abs() * 1e-5,
                    "{} at {}: row {} of {} records a scale of {} but level x shape is {}",
                    name,
                    level,
                    r.index,
                    r.split,
                    r.noise_scale,
                    expected
                );
            }
        }
    }
}

/// A held-out molecule carries the shape ITS REGION receives, looked up from the
/// training split's decision — not a fresh draw and not a blank.
///
/// Under the grouped-wider condition a scaffold group either has the wider error or
/// it does not, and that is decided on the training labels. A held-out molecule in an
/// affected group must carry the affected multiplier: without it, nothing downstream
/// can say what a held-out molecule's region was exposed to.
#[test]
fn held_out_rows_carry_the_shape_their_region_receives() {
    let f = fixture("region");
    ok(&f.run(&[
        "--noise-targeting",
        "grouped_wide",
        "--noise-shape",
        "gaussian",
        "--noise-level",
        "0.5",
        "--lambda",
        "3.0",
    ]));
    let prov = f.provenance();
    for split in ["train", "val", "test"] {
        let shapes: Vec<f32> = prov
            .iter()
            .filter(|r| r.split == split)
            .map(|r| r.noise_pattern)
            .collect();
        assert!(!shapes.is_empty(), "{}: no rows", split);
        let lo = shapes.iter().cloned().fold(f32::INFINITY, f32::min);
        let hi = shapes.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            (hi / lo - 3.0).abs() < 1e-3,
            "{}: the widest shape is {:.6} and the narrowest {:.6}, a ratio of {:.4}; \
             lambda is 3, so an affected region must be exactly three times the rest",
            split,
            hi,
            lo,
            hi / lo
        );
    }
    // and it is the TRAINING split's selection, not a fresh draw per split. A held-out
    // molecule is on the wide value if and only if its scaffold group is one of the
    // groups training chose — anything else is a different condition wearing the same
    // name, and it makes the held-out shape column unusable.
    let groups: HashMap<String, u32> = serde_json::from_str::<serde_json::Value>(
        &fs::read_to_string(f.dir.join(format!("scaffold_groups_{}.json", f.file_no))).unwrap(),
    )
    .unwrap()
    .as_object()
    .unwrap()
    .iter()
    .map(|(k, v)| (k.clone(), v.as_u64().unwrap() as u32))
    .collect();

    let narrow = prov
        .iter()
        .map(|r| r.noise_pattern)
        .fold(f32::INFINITY, f32::min);
    let mut train_affected: HashSet<u32> = HashSet::new();
    for r in prov.iter().filter(|r| r.split == "train") {
        if r.noise_pattern > narrow * 1.5 {
            train_affected.insert(groups[&r.canonical]);
        }
    }
    assert!(!train_affected.is_empty(), "training selected no scaffold groups");

    for split in ["val", "test"] {
        let mut disagreements = 0usize;
        let mut on_wide = 0usize;
        for r in prov.iter().filter(|r| r.split == split) {
            let is_wide = r.noise_pattern > narrow * 1.5;
            if is_wide {
                on_wide += 1;
            }
            if is_wide != train_affected.contains(&groups[&r.canonical]) {
                disagreements += 1;
            }
        }
        assert!(on_wide > 0, "{}: no held-out molecule fell in an affected group", split);
        assert_eq!(
            disagreements, 0,
            "{}: {} molecules disagree with the TRAINING split's choice of affected \
             scaffold groups — the selection was drawn again for this split instead of \
             being inherited",
            split, disagreements
        );
    }

    // and the test split, which receives no noise at all, still records it
    assert!(
        prov.iter()
            .filter(|r| r.split == "test")
            .all(|r| r.epsilon == 0.0 && r.noise_scale == 0.0),
        "the test split must record the shape without receiving any noise"
    );
}

/// The assay limit is a property of the ASSAY, so it is read off the training labels
/// and applied unchanged to validation.
///
/// The fixture's validation column is five times wider than training's. Reading the
/// limit off validation's own quantile would censor exactly the fraction asked for on
/// each split at a different value; reading it off training censors whatever share of
/// the wider validation column happens to sit past the training limit — which here is
/// far more than the fraction asked for.
#[test]
fn censoring_takes_its_limit_from_the_training_labels() {
    let f = fixture_with_val_scale("censorlimit", 5.0);
    ok(&f.run(&[
        "--noise-targeting",
        "censoring",
        "--noise-level",
        "0.25",
        "--censor-side",
        "upper",
    ]));
    let prov = f.provenance();
    let limit = f.manifest()["parameters"]["censor_limit"].as_f64().unwrap() as f32;

    let train: Vec<&ProvRow> = prov.iter().filter(|r| r.split == "train").collect();
    let val: Vec<&ProvRow> = prov.iter().filter(|r| r.split == "val").collect();

    let clipped_train =
        train.iter().filter(|r| r.epsilon != 0.0).count() as f32 / train.len() as f32;
    let clipped_val = val.iter().filter(|r| r.epsilon != 0.0).count() as f32 / val.len() as f32;
    assert!(
        (clipped_train - 0.25).abs() < 0.02,
        "training clipped {:.3}, asked for 0.25",
        clipped_train
    );
    assert!(
        clipped_val > 0.35,
        "validation clipped {:.3} of a column five times wider than training's. A limit \
         taken from the TRAINING labels catches far more than a quarter of it; {:.3} means \
         the limit came from validation's own quantile",
        clipped_val,
        clipped_val
    );
    // every clipped validation label lands exactly on the training limit
    for r in val.iter().filter(|r| r.epsilon != 0.0) {
        assert!(
            (r.y_noisy - limit).abs() < 1e-3,
            "a censored validation label was written as {} against a training limit of {}",
            r.y_noisy,
            limit
        );
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

/// The shifted grouped condition's precision is set by the EFFECTIVE number of
/// scaffold groups, not the raw count.
///
/// The group-level offset is averaged over molecules, so a few large groups dominate
/// it: the effective count is (sum n_g)^2 / sum n_g^2. On the real QM9 assignment that
/// is 189 against a raw count of 30,313 — a factor of 160.
///
/// Getting this wrong does not put a wrong number in a results row. It makes
/// `dose_tolerance` demand a precision the condition cannot deliver, so the flat-dose
/// gate fails runs that were never defective — and it fails them intermittently, by
/// seed, which is the worst way for a gate to be wrong. Found by chat B's cross-check
/// (RERUN_PLAN.md §2.3a).
///
/// The fixture's groups are deliberately lopsided: 3 groups hold half the molecules
/// and 17 hold the other half, so the raw count (20) and the effective count (~10.2)
/// are far apart and the two formulas cannot be confused for one another.
#[test]
fn grouped_shift_precision_uses_the_effective_group_count() {
    let f = fixture("effgroups");
    ok(&f.run(&[
        "--noise-targeting",
        "grouped_shift",
        "--noise-shape",
        "gaussian",
        "--noise-level",
        "0.5",
    ]));
    let m = f.manifest();
    let got = m["effective_n"].as_f64().unwrap();

    // sizes: 3 groups of ~67, 17 of ~12, over 400 molecules
    let n = 400.0f64;
    let mut sizes = vec![0.0f64; 20];
    for i in 0..400usize {
        let g = if i < 200 { i % 3 } else { 3 + (i % 17) };
        sizes[g] += 1.0;
    }
    let sum_sq: f64 = sizes.iter().map(|c| c * c).sum();
    let eff_groups = n * n / sum_sq;
    let rho = 0.62f64;
    let expected = 1.0 / (rho * rho / eff_groups + (1.0 - rho) * (1.0 - rho) / n);
    let raw_count_answer = 1.0 / (rho * rho / 20.0 + (1.0 - rho) * (1.0 - rho) / n);

    assert!(
        (got / expected - 1.0).abs() < 0.02,
        "effective_n is {:.2}; the effective group count ({:.2} groups) gives {:.2}",
        got,
        eff_groups,
        expected
    );
    assert!(
        (got / raw_count_answer - 1.0).abs() > 0.2,
        "effective_n is {:.2}, which is what the RAW group count gives ({:.2}) — the \
         group term is averaged over molecules, so a few large groups dominate it",
        got,
        raw_count_answer
    );

    // and the tolerance that comes out of it must actually admit the condition's spread
    let mut delivered = Vec::new();
    for seed in ["11", "22", "33", "44", "55", "66", "77", "88"] {
        let mut cmd = Command::new(BIN);
        f.reset();
        let out = cmd
            .current_dir(&f.dir)
            .args(["--seed", seed, "--config", &format!("config_{}.json", f.file_no)])
            .args(["--model", "rf", "--noise-targeting", "grouped_shift"])
            .args(["--noise-level", "0.5"])
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "the flat-dose gate rejected seed {} — the tolerance does not admit this \
             condition's own sampling spread:\n{}",
            seed,
            String::from_utf8_lossy(&out.stderr)
        );
        delivered.push(f.manifest()["delivered_dose_in_label_units"].as_f64().unwrap());
    }
    assert_eq!(delivered.len(), 8);
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

/// A scaffold split holds whole scaffold groups out, so no held-out molecule is in a
/// group the training selection ever marked. Two things have to be true then, and
/// they pull in opposite directions:
///
///   * VALIDATION receives noise (decision 3, 2026-08-26), so it must still carry the
///     condition's structure. Looking the training groups up finds nothing, so it
///     draws its own selection at the same molecule fraction. Without that it would
///     carry plain Gaussian noise under a name that says otherwise.
///   * TEST receives none. Its recorded shape is what the molecule's region WOULD
///     have got, and for a scaffold-keyed condition on an unseen scaffold the honest
///     answer is that the condition never reached it — flat. Drawing a selection here
///     would record an injection that did not happen.
///
/// Before this was handled the run aborted on the gate, on the first real QM9 run of
/// a grouped condition, having passed every test in this file — because the ordinary
/// fixture models a random split.
#[test]
fn a_scaffold_split_still_gives_validation_the_condition_and_leaves_test_flat() {
    let f = fixture_scaffold_split("scafsplit");
    let out = f.run(&[
        "--noise-targeting", "grouped_wide",
        "--noise-shape", "gaussian",
        "--noise-level", "0.5",
    ]);
    ok(&out);

    let prov = f.provenance();
    let val: Vec<&ProvRow> = prov.iter().filter(|r| r.split == "val").collect();
    let test: Vec<&ProvRow> = prov.iter().filter(|r| r.split == "test").collect();
    assert_eq!(val.len(), N_VAL);
    assert_eq!(test.len(), N_TEST);

    assert!(
        val.iter().any(|r| r.epsilon != 0.0),
        "validation received no noise under a scaffold split — the training selection \
         reaches none of its molecules, so it must draw its own"
    );
    let val_shape_varies = val
        .iter()
        .any(|r| (r.noise_pattern - val[0].noise_pattern).abs() > 1e-12);
    assert!(
        val_shape_varies,
        "validation's level-free shape is flat, so it is carrying uniform noise under \
         the name of a grouped condition"
    );

    assert!(
        test.iter().all(|r| r.epsilon == 0.0 && r.noise_scale == 0.0),
        "the test split was noised — it never is"
    );
    let test_shape_varies = test
        .iter()
        .any(|r| (r.noise_pattern - test[0].noise_pattern).abs() > 1e-12);
    assert!(
        !test_shape_varies,
        "the test split's level-free shape varies, which means a selection was drawn \
         for scaffolds the condition never reached — an injection that did not happen"
    );
}

// ---------------------------------------------------------------------------
// The settled condition set. `noise_conditions.json` at the repository root says
// what the study runs; these tests make it binding on this side.
//
// The point is not documentation. `scripts/noise_strategy_params.json` was a
// settings file that nothing ever read -- it was never passed to the binary, so
// for the life of the project it silently meant nothing while everyone believed
// it was in force. A file that describes the run and a run that ignores it is
// worse than no file. These tests fail if the two stop agreeing.

fn conditions_file() -> serde_json::Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the crate sits inside the repository")
        .join("noise_conditions.json");
    let text = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", path.display(), e));
    serde_json::from_str(&text).expect("noise_conditions.json is not valid JSON")
}

fn names_in(v: &serde_json::Value, key: &str) -> Vec<String> {
    v[key]
        .as_array()
        .unwrap_or_else(|| panic!("noise_conditions.json has no array at {}", key))
        .iter()
        .map(|e| e["name"].as_str().expect("every entry needs a name").to_string())
        .collect()
}

#[test]
fn the_settled_conditions_are_the_ones_this_injector_can_build() {
    let spec = conditions_file();
    let mut wanted = names_in(&spec, "stage_1_full_grid");
    wanted.extend(names_in(&spec, "stage_2_depth_only"));

    // Every condition the study runs must be one the self-test actually exercises,
    // or it ships unverified.
    let self_tested: HashSet<&str> = [
        "gaussian",
        "student_t_nu10",
        "student_t_nu5",
        "student_t_nu3",
        "laplace",
        "outlier_p01",
        "outlier_p05",
        "outlier_p10",
        "grouped_wider",
        "grouped_shifted",
        "censoring",
    ]
    .into_iter()
    .collect();

    for name in &wanted {
        assert!(
            self_tested.contains(name.as_str()),
            "the study runs '{}' but `--self-test` never exercises it, so it would ship \
             unverified. Add it to the `types` list in `self_test`, or take it out of \
             noise_conditions.json",
            name
        );
    }
}

#[test]
fn the_dropped_conditions_stay_dropped() {
    let spec = conditions_file();
    let stage_1 = names_in(&spec, "stage_1_full_grid");
    let stage_2 = names_in(&spec, "stage_2_depth_only");
    let dropped = names_in(&spec, "not_run");

    for name in &dropped {
        assert!(
            !stage_1.contains(name) && !stage_2.contains(name),
            "'{}' is listed as not run AND as something that runs. The evidence for \
             dropping it is in RERUN_PLAN.md §13.9; if that has been revisited, move the \
             entry rather than listing it twice",
            name
        );
    }

    // The four settings the twelve-replicate screen found redundant. Naming them
    // explicitly means putting one back is a deliberate edit to a test, with the
    // measurement in front of whoever does it, rather than a quiet line in a job script.
    for name in ["student_t_nu10", "student_t_nu3", "outlier_p01", "outlier_p05"] {
        assert!(
            dropped.contains(&name.to_string()),
            "'{}' was dropped on 2026-08-27: twelve replicates on real QM9 put every \
             Student-t and Outlier setting within 0.006 R2 of Gaussian at the reporting \
             level, against a test that could have detected 0.006 to 0.021. Putting it \
             back costs 4,680 training runs (9.1% of the old design) and needs a reason \
             that measurement does not already answer",
            name
        );
    }

    // The skewed draw is not in either injector and is not to be built.
    let skewed = spec["not_run"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["name"] == "skewed_draw")
        .expect("skewed_draw belongs in not_run");
    assert_eq!(
        skewed["never_implemented"], serde_json::json!(true),
        "the skewed draw was tested and rejected; it exists only in the local screen"
    );
}

#[test]
fn the_cli_defaults_are_the_settled_settings() {
    // The single-setting decisions are only real if the defaults follow them. A grid
    // that says "one Student-t setting" while the binary defaults to a different one
    // is two sources of truth, and the job scripts would decide which wins.
    let spec = conditions_file();
    let settings = &spec["settings_that_follow"];
    let f = fixture("cli_defaults");

    // Ask for Student-t WITHOUT naming the tail weight. The shape's own name carries
    // it, so the manifest says which one was actually used.
    let out = f.run(&["--noise-shape", "student_t", "--noise-level", "0.5"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let nu = settings["nu"].as_f64().unwrap();
    let want = format!("student_t_nu{}", nu);
    let got = f.manifest()["noise_shape"].as_str().unwrap_or_default().to_string();
    assert_eq!(
        got, want,
        "the binary defaults Student-t to '{}', but noise_conditions.json settles it at \
         nu = {}. RERUN_PLAN.md §13.9: the three tail weights are within 0.006 R2 of each \
         other, so exactly one runs, and the default is how that decision takes effect",
        got, nu
    );

    // Same for the contamination fraction, which the manifest records by name.
    let out = f.run(&["--noise-targeting", "outlier", "--noise-level", "0.5"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let want_p = settings["outlier_p"].as_f64().unwrap();
    let got_p = f.manifest()["parameters"]["outlier_p"].as_f64().unwrap_or(-1.0);
    assert!(
        (got_p - want_p).abs() < 1e-6,
        "the binary defaults the contaminated fraction to {}, but noise_conditions.json \
         settles it at {}. One setting runs, not three",
        got_p,
        want_p
    );
}
