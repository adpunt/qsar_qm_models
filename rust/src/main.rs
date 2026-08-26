use std::collections::HashMap;
use std::hash::Hash;
use regex::Regex;
use std::io::{self, BufReader, BufRead, Read, Seek, SeekFrom};
use ndarray::Array2;
use std::fs::File;
use serde::{Deserialize, Serialize};
use clap::{Arg, Command, ArgAction};
use num_traits::{Float, FromPrimitive};
use std::iter::Sum;
use rand_distr::{ChiSquared, Distribution, StandardNormal};
use std::io::Write;
use std::cmp::Reverse;
use std::io::BufWriter;
use std::fs::{OpenOptions, remove_file, rename};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

extern crate rdkit_sys;

use rdkit_sys::ro_mol_ffi::{smiles_to_mol};
use rdkit_sys::fingerprint_ffi::{rdk_fingerprint_mol, explicit_bit_vect_to_u64_vec}; // Assuming fingerprint generation is related to this type.
use cxx::let_cxx_string;
use cxx::UniquePtr;
use cxx::CxxVector;

const DELIMITER: u8 = 0x1F;  // ASCII 31 (Unit Separator)

struct SmilesTokenizer {
    regex: Regex,
}

impl SmilesTokenizer {
    fn new() -> Self {
        let regex_pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>>?|\*|\$|%[0-9]{2}|[0-9])";
        SmilesTokenizer {
            regex: Regex::new(regex_pattern).unwrap(),
        }
    }

    fn tokenize(&self, smiles: &str) -> Vec<String> {
        self.regex.find_iter(smiles).map(|mat| mat.as_str().to_owned()).collect()
    }
}

#[derive(Deserialize, Debug)]
struct Config {
    sample_size: usize,
    noise: bool,
    train_count: usize, 
    test_count: usize,
    val_count: usize,
    max_vocab: usize,
    file_no: usize,
    molecular_representations: Vec<String>,
    k_domains: usize, 
    logging: bool,
    regression: bool,
    normalize: bool,
    uncertainty: bool,
}

#[derive(Debug)]
struct SmilesData {
    isomeric_smiles: String,
    canonical_smiles: String,
    randomized_smiles: Option<String>,
    target_value: f32,
    sns_buf: [u8; 128],
    pdv_buf: [u8; 25],
    continuous_pdv_buf: [u8; 800],
    // The learned embeddings are 32-bit floats, four bytes a dimension: mol2vec
    // 300 dims, chemberta 768, mhggnn 1024. They used to be one byte a dimension,
    // min-max rescaled per molecule, which destroyed comparability between
    // molecules (RERUN_PLAN.md 2.8c). These widths and the Python writer's must
    // move together or every record after the first is read at the wrong offset.
    mol2vec_buf: [u8; 1200],
    chemberta_buf: [u8; 3072],
    mhggnn_buf: [u8; 4096],
    morgan_buf: [u8; 256],
    avalon_buf: [u8; 256],
}

#[derive(Serialize, Clone)]
struct PlotPoint<T> {
    x: T,
    y: T,
}

// ============================================================================
// NOISE INJECTION — the redesigned scheme.
//
// Specification: NOISE_DESIGN.md §1 (the control quantity), §2 (the types and
// their algebra), §6.2 (what to build). Working reference that this must agree
// with: rust/reference/noise_arms.rs.
//
// The one rule the whole redesign exists to enforce:
//
//     the noise LEVEL is the amount of noise actually delivered.
//
// The old code set a single knob per type and let each type deliver whatever it
// happened to deliver — between 0.49x and 2.00x the Gaussian amount on QM9. Every
// apparent difference between "noise types" was therefore a difference in amount.
// Here each type computes its own unit dose G from the CLEAN training labels and
// solves its internal scale as `target / G`, so all of them land on the same
// delivered dose and a comparison between them is a comparison of shape.
//
// Two things are deliberately separated, following the structure that was already
// in this file:
//
//     NoiseShape     — the shape of each individual draw
//     NoiseTargeting — who gets hit, and how hard (the per-molecule scale map)
//
// so that shape and targeting are independently selectable.
// ============================================================================

/// The shape of an individual draw. Gaussian is the reference case; Student-t is
/// the heavy-tailed family that nests Gaussian at nu -> infinity; Laplace is the
/// shape actually fitted to real bioactivity disagreements.
#[derive(Debug, Clone, Copy, PartialEq)]
enum NoiseShape {
    Gaussian,
    StudentT { nu: f32 },
    Laplace,
}

impl NoiseShape {
    /// Standard deviation of the shape at unit scale parameter. This is the factor
    /// that turns a scale parameter into a delivered amount, and it is half of the
    /// unit dose G.
    fn unit_sd(&self) -> f32 {
        match self {
            // A standard normal has variance 1.
            NoiseShape::Gaussian => 1.0,
            // A standard t with nu degrees of freedom has variance nu/(nu-2).
            NoiseShape::StudentT { nu } => (nu / (nu - 2.0)).sqrt(),
            // A Laplace with scale 1 has variance 2.
            NoiseShape::Laplace => 2f32.sqrt(),
        }
    }

    /// One draw at unit scale parameter (NOT unit variance — divide by `unit_sd`
    /// for that).
    fn draw(&self, rng: &mut StdRng) -> f32 {
        match self {
            NoiseShape::Gaussian => StandardNormal.sample(rng),
            NoiseShape::StudentT { nu } => {
                // t_nu = Z / sqrt(V/nu),  V ~ chi-squared(nu)
                let z: f32 = StandardNormal.sample(rng);
                let chi = ChiSquared::new(*nu).expect("chi-squared requires nu > 0");
                let v: f32 = chi.sample(rng);
                z / (v / nu).sqrt()
            }
            NoiseShape::Laplace => {
                // inverse CDF: -sgn(u) * ln(1 - 2|u|),  u ~ U(-0.5, 0.5)
                let u: f32 = rng.random::<f32>() - 0.5;
                -u.signum() * (1.0 - 2.0 * u.abs()).ln()
            }
        }
    }

    /// One draw rescaled to unit variance. Used by the two-component types, where
    /// the shape's own spread has to be divided out before the components are
    /// combined.
    fn draw_unit_variance(&self, rng: &mut StdRng) -> f32 {
        self.draw(rng) / self.unit_sd()
    }

    fn name(&self) -> String {
        match self {
            NoiseShape::Gaussian => "gaussian".to_string(),
            NoiseShape::StudentT { nu } => format!("student_t_nu{}", nu),
            NoiseShape::Laplace => "laplace".to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum CensorSide {
    Upper,
    Lower,
}

/// Who gets hit and how hard.
#[derive(Debug, Clone, Copy, PartialEq)]
enum NoiseTargeting {
    /// Every molecule, same expected amount.
    Uniform,
    /// Whole scaffold groups get a `lambda`x WIDER error, still centred on the
    /// true value. Evidence: within-laboratory error must be multiplied by about
    /// three to reach between-laboratory error (Avdeef 2019).
    GroupedWide { lambda: f32, group_fraction: f32 },
    /// Whole scaffold groups have their labels SHIFTED by a constant offset — a
    /// group-level term plus a within-molecule term, the two variances summing to
    /// the target. Evidence: 62% of measurement variance is between laboratories
    /// (Bentz et al. 2013, Table 7), and a laboratory differing from another is an
    /// offset, not a widening.
    GroupedShift { group_variance_share: f32 },
    /// A RANDOM fraction `p` of labels get a `lambda`x wider error. Huber's
    /// contamination model. Selection is random, not by label value — the premise
    /// that error tracks the measured value was tested and disproved
    /// (NOISE_DESIGN.md §3.2).
    Outlier { p: f32, lambda: f32 },
    /// Values past an assay limit recorded as the limit. Not zero-mean, and it has
    /// no variance parameter, so it does NOT go through the dose solver — its
    /// level is the fraction of labels clipped.
    Censoring { side: CensorSide },
}

impl NoiseTargeting {
    fn is_dose_matched(&self) -> bool {
        !matches!(self, NoiseTargeting::Censoring { .. })
    }

    fn needs_groups(&self) -> bool {
        matches!(
            self,
            NoiseTargeting::GroupedWide { .. } | NoiseTargeting::GroupedShift { .. }
        )
    }

    fn name(&self) -> String {
        match self {
            NoiseTargeting::Uniform => "uniform".to_string(),
            NoiseTargeting::GroupedWide { .. } => "grouped_wider".to_string(),
            NoiseTargeting::GroupedShift { .. } => "grouped_shifted".to_string(),
            NoiseTargeting::Outlier { .. } => "outlier".to_string(),
            NoiseTargeting::Censoring { side } => match side {
                CensorSide::Upper => "censoring_upper".to_string(),
                CensorSide::Lower => "censoring_lower".to_string(),
            },
        }
    }
}

/// How `level` is read. `Spread` means a fraction of the clean training label
/// standard deviation (the only honest axis on QM9, which has no assay error to
/// anchor to). `Label` means the dose directly in the label's own units (log units
/// on the experimental datasets, where assay error IS quotable).
#[derive(Debug, Clone, Copy, PartialEq)]
enum DoseUnits {
    Spread,
    Label,
}

#[derive(Debug, Clone)]
struct NoiseSpec {
    shape: NoiseShape,
    targeting: NoiseTargeting,
    /// The dose for the dose-matched types; the censored fraction for censoring.
    level: f32,
    units: DoseUnits,
    seed: u64,
}

/// Everything the run injected, plus everything needed to prove what it injected.
/// Nothing here is reconstructed after the fact — it is recorded where it is drawn.
#[derive(Debug, Clone)]
struct NoisePlan {
    /// Per training record, in RAW label units, in training-record order.
    epsilon: Vec<f32>,
    /// The canonical SMILES each epsilon belongs to. Carried so the write path can
    /// assert it is applying a molecule's own noise rather than a row position's.
    canonical: Vec<String>,

    noise_type: String,
    shape_name: String,
    targeting_name: String,
    unit_dose_g: f32,
    solved_scale: f32,
    target_dose_label_units: f32,
    realised_dose_label_units: f32,
    realised_dose_fraction_of_spread: f32,
    mean_epsilon: f32,
    affected_molecule_fraction: f32,
    /// How many independent contributions the delivered dose is averaged over. Not
    /// the molecule count: a scale map that concentrates the noise on a few
    /// molecules, or a group-level term drawn once per scaffold group, pins the dose
    /// far less precisely than the raw count suggests. See `dose_tolerance`.
    effective_n: f32,
    clean_label_mean: f32,
    clean_label_sd: f32,
    seed: u64,
    n_train: usize,
    params: serde_json::Value,
}

// Every statistic accumulates in f64. The draws themselves stay f32, matching the
// pipeline, but summing 133,885 f32 squares loses enough precision to look like a
// real disagreement with the Python injector when it is only accumulation error.

/// The registry name for a condition. Kept in step with `noiseInject.CONDITIONS`
/// and with `roster()` in `rust/reference/noise_arms.rs`; the cross-check joins the
/// two implementations on it.
///
/// The plain shapes carry no targeting prefix, because "uniform gaussian" is just
/// the Gaussian condition. A shape other than the default on a targeted condition
/// appends the shape, so the combination cannot be silently mistaken for the default.
fn condition_name(spec: &NoiseSpec) -> String {
    let shape = spec.shape.name();
    match spec.targeting {
        NoiseTargeting::Uniform => shape,
        NoiseTargeting::GroupedWide { .. } | NoiseTargeting::GroupedShift { .. } => {
            let base = spec.targeting.name();
            if spec.shape == NoiseShape::Gaussian {
                base
            } else {
                format!("{}_{}", base, shape)
            }
        }
        NoiseTargeting::Outlier { p, .. } => {
            let base = format!("outlier_p{:02}", (p * 100.0).round() as u32);
            if spec.shape == NoiseShape::Gaussian {
                base
            } else {
                format!("{}_{}", base, shape)
            }
        }
        NoiseTargeting::Censoring { side } => {
            let pct = (spec.level * 100.0).round() as u32;
            match side {
                CensorSide::Upper => format!("censoring_{}", pct),
                CensorSide::Lower => format!("censoring_lower_{}", pct),
            }
        }
    }
}

fn population_mean(v: &[f32]) -> f32 {
    (v.iter().map(|x| *x as f64).sum::<f64>() / v.len() as f64) as f32
}

fn population_sd(v: &[f32]) -> f32 {
    let m = v.iter().map(|x| *x as f64).sum::<f64>() / v.len() as f64;
    ((v.iter().map(|x| (*x as f64 - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()) as f32
}

fn rms(v: &[f32]) -> f32 {
    ((v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / v.len() as f64).sqrt()) as f32
}

/// Quantile by linear interpolation between order statistics — numpy's default, so
/// the two injectors put the censoring limit in the same place.
fn quantile(sorted: &[f32], q: f32) -> f32 {
    if sorted.is_empty() {
        return f32::NAN;
    }
    let q = q.clamp(0.0, 1.0) as f64;
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let w = (pos - lo as f64) as f32;
        sorted[lo] * (1.0 - w) + sorted[hi] * w
    }
}

/// Scaffold group per molecule, keyed by CANONICAL SMILES rather than by row
/// position. Keying by position is what let the original held-out bug attach one
/// molecule's noise to another; keying by molecule cannot.
fn load_scaffold_groups(file_path: &str) -> io::Result<HashMap<String, u32>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let data: HashMap<String, u32> = serde_json::from_reader(reader)?;
    Ok(data)
}

/// Resolve each training molecule to a group id. A molecule with no entry in the
/// assignment file gets its own singleton group, and the miss rate is returned so
/// the caller can refuse to run on a stale or mismatched file.
fn resolve_groups(canonical: &[String], groups: &HashMap<String, u32>) -> (Vec<u32>, f32) {
    let mut next_singleton: u32 = groups.values().copied().max().unwrap_or(0) + 1;
    let mut misses = 0usize;
    let mut out = Vec::with_capacity(canonical.len());
    for smiles in canonical {
        match groups.get(smiles) {
            Some(g) => out.push(*g),
            None => {
                misses += 1;
                out.push(next_singleton);
                next_singleton += 1;
            }
        }
    }
    (out, misses as f32 / canonical.len().max(1) as f32)
}

/// The per-molecule scale multipliers this targeting rule applies at unit scale,
/// and the fraction of molecules it actually affects.
///
/// The affected fraction is MEASURED, never assumed. For the grouped types in
/// particular, real Murcko scaffold groups are very unevenly sized, so the
/// fraction of molecules in a fifth of the groups is not a fifth of the molecules
/// (NOISE_DESIGN.md §2, implementation point 2).
fn scale_map(
    targeting: &NoiseTargeting,
    n: usize,
    groups: Option<&[u32]>,
    rng: &mut StdRng,
) -> (Vec<f32>, f32) {
    match targeting {
        NoiseTargeting::Uniform => (vec![1.0; n], 1.0),

        NoiseTargeting::GroupedShift { .. } => (vec![1.0; n], 1.0),

        NoiseTargeting::GroupedWide {
            lambda,
            group_fraction,
        } => {
            // Rule 1 of NOISE_DESIGN.md §2a: select groups until a MOLECULE fraction
            // is reached, never by counting groups.
            //
            // Real Murcko scaffolds are wildly uneven — on the first 10,000 QM9
            // molecules, 855 distinct scaffolds of which 523 are singletons. Taking a
            // fifth of the GROUPS puts anywhere between 6.7% and 55.1% of the
            // MOLECULES in the affected set, so the condition's defining parameter
            // would swing eightfold from replicate to replicate. Dose matching
            // survives either way, because the solver uses whatever fraction actually
            // resulted — but who gets hit is the condition, not noise in it.
            let g = groups.expect("grouped noise requires scaffold group assignments");
            let mut sizes: HashMap<u32, usize> = HashMap::new();
            for gi in g.iter() {
                *sizes.entry(*gi).or_insert(0) += 1;
            }
            let mut uniq: Vec<u32> = sizes.keys().copied().collect();
            uniq.sort_unstable();

            // Fisher-Yates over the group ids, so the selection is a fresh draw each
            // replicate rather than a function of the scaffold ordering.
            for i in (1..uniq.len()).rev() {
                let j = rng.random_range(0..=i);
                uniq.swap(i, j);
            }

            // Add groups one at a time; skip any that would take the running
            // molecule fraction further from the target than stopping would, and stop
            // once the target is reached. Matches `select_groups_by_molecule_fraction`
            // in `rust/reference/noise_arms.rs`, which is the fixed point both
            // implementations are checked against.
            let target_fraction = *group_fraction;
            let mut cumulative = 0usize;
            let mut bad: std::collections::HashSet<u32> = std::collections::HashSet::new();
            for gid in uniq.iter() {
                let size = sizes[gid];
                let here = (cumulative as f32 / n as f32 - target_fraction).abs();
                let there = ((cumulative + size) as f32 / n as f32 - target_fraction).abs();
                if cumulative > 0 && there > here {
                    continue;
                }
                bad.insert(*gid);
                cumulative += size;
                if cumulative as f32 / n as f32 >= target_fraction {
                    break;
                }
            }

            let scales: Vec<f32> = g
                .iter()
                .map(|gi| if bad.contains(gi) { *lambda } else { 1.0 })
                .collect();
            let affected =
                scales.iter().filter(|s| **s != 1.0).count() as f32 / n.max(1) as f32;
            (scales, affected)
        }

        NoiseTargeting::Outlier { p, lambda } => {
            let scales: Vec<f32> = (0..n)
                .map(|_| if rng.random::<f32>() < *p { *lambda } else { 1.0 })
                .collect();
            let affected =
                scales.iter().filter(|s| **s != 1.0).count() as f32 / n.max(1) as f32;
            (scales, affected)
        }

        NoiseTargeting::Censoring { .. } => (vec![1.0; n], 0.0),
    }
}

/// Unit dose G: the root-mean-square of the per-molecule scale map, times the
/// shape's own unit standard deviation. Solving `scale = target / G` makes the
/// delivered root-mean-square noise equal `target`.
///
/// This is the whole fix. NOISE_DESIGN.md §1.
fn unit_dose(shape: &NoiseShape, scales: &[f32]) -> f32 {
    let ms: f64 = scales.iter().map(|s| (*s as f64) * (*s as f64)).sum::<f64>()
        / scales.len().max(1) as f64;
    (ms.sqrt() as f32) * shape.unit_sd()
}

/// Build the noise for one run: the per-molecule values, and the provenance that
/// proves what they are.
fn build_noise_plan(
    labels: &[f32],
    canonical: &[String],
    spec: &NoiseSpec,
    groups: Option<&HashMap<String, u32>>,
) -> io::Result<NoisePlan> {
    assert_eq!(
        labels.len(),
        canonical.len(),
        "label and SMILES columns must be the same length"
    );
    let n = labels.len();
    if n == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "no training labels were read — cannot build a noise plan",
        ));
    }

    let clean_mean = population_mean(labels);
    let clean_sd = population_sd(labels);

    let mut params = serde_json::Map::new();
    params.insert("shape".to_string(), serde_json::json!(spec.shape.name()));
    params.insert(
        "targeting".to_string(),
        serde_json::json!(spec.targeting.name()),
    );
    params.insert("level".to_string(), serde_json::json!(spec.level));
    params.insert(
        "dose_units".to_string(),
        serde_json::json!(match spec.units {
            DoseUnits::Spread => "fraction_of_label_spread",
            DoseUnits::Label => "label_units",
        }),
    );

    // The condition's name, exactly as it appears in `noiseInject.CONDITIONS` and in
    // `roster()` in `rust/reference/noise_arms.rs`. One name per condition — a job
    // script, a results row and a figure label have to agree, and the paper has
    // already had one quantity carrying two names on facing pages.
    let noise_type = condition_name(spec);

    // ---- The zero level. Exactly zero, not something small. -----------------
    // Failure mode 2: the old pipeline reconstructed the injected noise by fitting
    // a line, so at zero noise the "noise" was floating-point rounding whose size
    // grows with the label. The negative control showed a stronger signal than the
    // real levels. Recorded ground truth cannot do that.
    if spec.level <= 0.0 {
        return Ok(NoisePlan {
            epsilon: vec![0.0; n],
            canonical: canonical.to_vec(),
            noise_type,
            shape_name: spec.shape.name(),
            targeting_name: spec.targeting.name(),
            unit_dose_g: 1.0,
            solved_scale: 0.0,
            target_dose_label_units: 0.0,
            realised_dose_label_units: 0.0,
            realised_dose_fraction_of_spread: 0.0,
            mean_epsilon: 0.0,
            affected_molecule_fraction: 0.0,
            effective_n: n as f32,
            clean_label_mean: clean_mean,
            clean_label_sd: clean_sd,
            seed: spec.seed,
            n_train: n,
            params: serde_json::Value::Object(params),
        });
    }

    let mut rng = StdRng::seed_from_u64(spec.seed);

    // ---- Censoring. Its own axis: no dose to solve for. ---------------------
    if let NoiseTargeting::Censoring { side } = spec.targeting {
        let fraction = spec.level;
        if !(0.0..1.0).contains(&fraction) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("censored fraction must be in [0, 1), got {}", fraction),
            ));
        }
        let mut sorted = labels.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let cut = match side {
            CensorSide::Upper => quantile(&sorted, 1.0 - fraction),
            CensorSide::Lower => quantile(&sorted, fraction),
        };
        let epsilon: Vec<f32> = labels
            .iter()
            .map(|y| match side {
                CensorSide::Upper => {
                    if *y > cut {
                        cut - y
                    } else {
                        0.0
                    }
                }
                CensorSide::Lower => {
                    if *y < cut {
                        cut - y
                    } else {
                        0.0
                    }
                }
            })
            .collect();
        let affected = epsilon.iter().filter(|e| **e != 0.0).count() as f32 / n as f32;
        let realised = rms(&epsilon);
        let mean_eps = population_mean(&epsilon);
        params.insert("censor_limit".to_string(), serde_json::json!(cut));
        params.insert(
            "requested_censored_fraction".to_string(),
            serde_json::json!(fraction),
        );
        return Ok(NoisePlan {
            epsilon,
            canonical: canonical.to_vec(),
            noise_type,
            shape_name: spec.shape.name(),
            targeting_name: spec.targeting.name(),
            unit_dose_g: f32::NAN, // censoring does not go through the dose solver
            solved_scale: f32::NAN,
            target_dose_label_units: f32::NAN,
            realised_dose_label_units: realised,
            realised_dose_fraction_of_spread: realised / clean_sd,
            mean_epsilon: mean_eps,
            affected_molecule_fraction: affected,
            effective_n: n as f32,
            clean_label_mean: clean_mean,
            clean_label_sd: clean_sd,
            seed: spec.seed,
            n_train: n,
            params: serde_json::Value::Object(params),
        });
    }

    // ---- The dose-matched types. -------------------------------------------
    let target = match spec.units {
        DoseUnits::Spread => spec.level * clean_sd,
        DoseUnits::Label => spec.level,
    };

    let group_ids: Option<Vec<u32>> = if spec.targeting.needs_groups() {
        let map = groups.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "{} requires a scaffold group assignment file",
                    spec.targeting.name()
                ),
            )
        })?;
        let (ids, miss_rate) = resolve_groups(canonical, map);
        if miss_rate > 0.01 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "{:.2}% of training molecules are missing from the scaffold group file — \
                     it does not match this split. Refusing to run.",
                    miss_rate * 100.0
                ),
            ));
        }
        let mut sizes: HashMap<u32, usize> = HashMap::new();
        for gi in ids.iter() {
            *sizes.entry(*gi).or_insert(0) += 1;
        }
        let largest = sizes.values().copied().max().unwrap_or(0);
        params.insert("n_scaffold_groups".to_string(), serde_json::json!(sizes.len()));
        // Rule 2 of NOISE_DESIGN.md §2a: if one group holds a large share of the
        // molecules, a single offset draw moves that share at once and the delivered
        // dose swings. Record it so the condition can be read rather than guessed at.
        params.insert(
            "largest_group_share_of_molecules".to_string(),
            serde_json::json!(largest as f32 / ids.len().max(1) as f32),
        );
        params.insert(
            "scaffold_lookup_miss_fraction".to_string(),
            serde_json::json!(miss_rate),
        );
        Some(ids)
    } else {
        None
    };

    let (scales, mut affected) = scale_map(
        &spec.targeting,
        n,
        group_ids.as_deref(),
        &mut rng,
    );
    let g = unit_dose(&spec.shape, &scales);
    let solved = target / g;

    let epsilon: Vec<f32> = match spec.targeting {
        // Two components: a group-level offset carrying `rho` of the variance and a
        // within-molecule term carrying the rest, so the two sum to the target.
        //   eps_i = solved * ( sqrt(rho) * z_g(i) + sqrt(1-rho) * w_i )
        // with z_g and w_i both unit-variance, hence G = 1 by construction.
        NoiseTargeting::GroupedShift {
            group_variance_share,
        } => {
            let rho = group_variance_share;
            let ids = group_ids.as_ref().expect("grouped shift requires groups");
            // Both components are unit draws FROM THE SHAPE (NOISE_DESIGN.md §2a),
            // so a heavy-tailed shifted condition has heavy tails in the group
            // offsets as well as in the within-molecule term.
            let mut offsets: HashMap<u32, f32> = HashMap::new();
            for gid in ids.iter() {
                if !offsets.contains_key(gid) {
                    let b = spec.shape.draw_unit_variance(&mut rng);
                    offsets.insert(*gid, b);
                }
            }
            params.insert(
                "group_variance_share".to_string(),
                serde_json::json!(rho),
            );
            ids.iter()
                .map(|gid| {
                    let b = offsets[gid];
                    let w = spec.shape.draw_unit_variance(&mut rng);
                    solved * (rho.sqrt() * b + (1.0 - rho).sqrt() * w)
                })
                .collect()
        }
        _ => scales
            .iter()
            .map(|s| spec.shape.draw(&mut rng) * solved * s)
            .collect(),
    };

    if let NoiseTargeting::GroupedShift { .. } = spec.targeting {
        affected = 1.0;
    }

    match spec.targeting {
        NoiseTargeting::GroupedWide {
            lambda,
            group_fraction,
        } => {
            params.insert("lambda".to_string(), serde_json::json!(lambda));
            params.insert(
                "requested_group_fraction".to_string(),
                serde_json::json!(group_fraction),
            );
        }
        NoiseTargeting::Outlier { p, lambda } => {
            params.insert("outlier_p".to_string(), serde_json::json!(p));
            params.insert("lambda".to_string(), serde_json::json!(lambda));
        }
        _ => {}
    }

    // How many independent contributions the delivered dose is really averaged over.
    //
    // For a plain scale map this is the standard effective sample size for a
    // weighted second moment, (sum s^2)^2 / sum s^4: it collapses towards the number
    // of heavily-weighted molecules when the map concentrates the noise on a few of
    // them. For the shifted grouped type the group-level term is drawn once per
    // SCAFFOLD GROUP, so its component is averaged over the group count, not the
    // molecule count, and the two components combine as a variance-weighted sum.
    let effective_n = match spec.targeting {
        NoiseTargeting::GroupedShift {
            group_variance_share,
        } => {
            let rho = group_variance_share;
            let ids = group_ids.as_ref().expect("grouped shift requires groups");
            let mut uniq = ids.clone();
            uniq.sort_unstable();
            uniq.dedup();
            let n_groups = uniq.len().max(1) as f32;
            1.0 / (rho * rho / n_groups + (1.0 - rho) * (1.0 - rho) / n as f32)
        }
        _ => {
            let s2: f32 = scales.iter().map(|s| s * s).sum();
            let s4: f32 = scales.iter().map(|s| s * s * s * s).sum();
            if s4 > 0.0 {
                s2 * s2 / s4
            } else {
                n as f32
            }
        }
    };

    let realised = rms(&epsilon);
    let mean_eps = population_mean(&epsilon);
    Ok(NoisePlan {
        epsilon,
        canonical: canonical.to_vec(),
        noise_type,
        shape_name: spec.shape.name(),
        targeting_name: spec.targeting.name(),
        unit_dose_g: g,
        solved_scale: solved,
        target_dose_label_units: target,
        realised_dose_label_units: realised,
        realised_dose_fraction_of_spread: realised / clean_sd,
        mean_epsilon: mean_eps,
        affected_molecule_fraction: affected,
        effective_n,
        clean_label_mean: clean_mean,
        clean_label_sd: clean_sd,
        seed: spec.seed,
        n_train: n,
        params: serde_json::Value::Object(params),
    })
}

/// How far the SAMPLE dose may sit from the POPULATION dose before the run is
/// refused.
///
/// The population dose is exact by construction — the solver sets the scale so that
/// E[eps^2] equals the target. What varies run to run is the sample root-mean-square,
/// and how much it varies is not a matter of taste: for a quantity averaged over
/// `n_eff` independent contributions with kurtosis `k`, the relative standard error
/// of the second moment is sqrt((k - 1) / (4 * n_eff)), and the root halves it.
///
/// That formula reproduces the measured spreads in NOISE_DESIGN.md §5.1b: at
/// n = 133,885 Gaussian draws it gives 0.19%, which is exactly the standard
/// deviation recorded there across 40 seeds. A flat half-percent band is only right
/// at that sample size; at four hundred molecules the same construction wobbles by
/// several percent and a flat band would fail correct code.
///
/// Both inputs are taken from what was actually drawn: the kurtosis from the sample,
/// and `n_eff` from the scale map and the group count (a map that concentrates the
/// noise on a few molecules, or a term drawn once per scaffold group, pins the dose
/// far less tightly than the molecule count suggests).
///
/// At nu <= 4 the fourth moment of a Student-t is infinite, so the sample kurtosis is
/// meaningless and the band is set by fiat instead. Across 40 seeds at nu = 3 the
/// error ranged from -3.5% to +6.8% while remaining unbiased.
fn dose_tolerance(shape: &NoiseShape, epsilon: &[f32], effective_n: f32) -> f32 {
    if let NoiseShape::StudentT { nu } = shape {
        if *nu <= 4.0 {
            return 0.15;
        }
    }
    let n_eff = effective_n.max(1.0);
    let m2: f32 = epsilon.iter().map(|e| e * e).sum::<f32>() / epsilon.len() as f32;
    let m4: f32 = epsilon.iter().map(|e| e.powi(4)).sum::<f32>() / epsilon.len() as f32;
    let kurtosis = if m2 > 0.0 {
        (m4 / (m2 * m2)).clamp(3.0, 60.0)
    } else {
        3.0
    };
    // three standard errors, floored at the half a percent quoted in the design for
    // the full QM9 column
    let se = ((kurtosis - 1.0) / (4.0 * n_eff)).sqrt();
    (3.0 * se).max(0.005)
}

/// The gates that must hold for the noise this run injected. These fail the run.
/// RERUN_PLAN.md §8 gates 4 and 5; NOISE_DESIGN.md §6.3.
fn assert_noise_plan_gates(plan: &NoisePlan, spec: &NoiseSpec, labels: &[f32]) {
    assert_eq!(
        plan.epsilon.len(),
        labels.len(),
        "gate: one recorded noise value per training molecule"
    );

    // Gate 5 — zero noise records EXACTLY zero, not something small.
    if spec.level <= 0.0 {
        assert!(
            plan.epsilon.iter().all(|e| *e == 0.0),
            "gate: at level 0 every recorded noise value must be exactly zero"
        );
        return;
    }

    assert!(
        plan.epsilon.iter().all(|e| e.is_finite()),
        "gate: every recorded noise value must be finite"
    );

    // Gate 1, first half (NOISE_DESIGN.md §2a rule 3) — the CONSTRUCTION is right:
    // unit dose times solved scale is the target, exactly. This is the part that is
    // true by algebra rather than by luck of the draw, so it gets an exact check.
    //
    // The shifted grouped condition has no solver step — its two variances sum to the
    // target by construction — so its unit dose is 1 and the identity holds trivially.
    if spec.targeting.is_dose_matched() {
        let constructed = plan.unit_dose_g * plan.solved_scale;
        let slack = plan.target_dose_label_units.abs() * 1e-5;
        assert!(
            (constructed - plan.target_dose_label_units).abs() <= slack,
            "gate: {} constructs a dose of {:.9} against a target of {:.9} — the solver is wrong",
            plan.noise_type,
            constructed,
            plan.target_dose_label_units
        );
    }

    // Gate 1, second half — what one realisation actually delivered. The population
    // dose is exact; the sample root-mean-square wobbles, by an amount the band below
    // works out from the draw itself. This catches breakage, not sampling.
    if spec.targeting.is_dose_matched() {
        let tol = dose_tolerance(&spec.shape, &plan.epsilon, plan.effective_n);
        let err = (plan.realised_dose_label_units / plan.target_dose_label_units - 1.0).abs();
        assert!(
            err <= tol,
            "gate: {} delivered {:.6} against a target of {:.6} ({:+.2}%), outside the {:.2}% \
             band that {:.0} effective observations allow",
            plan.noise_type,
            plan.realised_dose_label_units,
            plan.target_dose_label_units,
            (plan.realised_dose_label_units / plan.target_dose_label_units - 1.0) * 100.0,
            tol * 100.0,
            plan.effective_n
        );
    }

    // Failure mode 6 — a cut-point in the wrong units caught 99.99925% of molecules
    // and nobody noticed, because the affected fraction was never recorded.
    assert!(
        plan.affected_molecule_fraction > 0.0,
        "gate: {} affected no molecules at level {} — the condition is degenerate",
        plan.noise_type,
        spec.level
    );
}

/// Run the cross-type gate that no single training run can run: at ONE target, on
/// the real clean labels, every dose-matched noise type must deliver the same
/// amount. This is the single check that proves the confound is gone
/// (RERUN_PLAN.md §8 gate 1), plus gates 5 and 7.
///
/// Returns the number of failures.
fn self_test(labels: &[f32], groups: Option<&HashMap<String, u32>>, canonical: &[String]) -> usize {
    let mut failures = 0usize;
    let clean_sd = population_sd(labels);
    println!(
        "self-test on {} clean labels — mean {:.6}, SD {:.6}\n",
        labels.len(),
        population_mean(labels),
        clean_sd
    );

    let mut types: Vec<(NoiseShape, NoiseTargeting)> = vec![
        (NoiseShape::Gaussian, NoiseTargeting::Uniform),
        (NoiseShape::StudentT { nu: 10.0 }, NoiseTargeting::Uniform),
        (NoiseShape::StudentT { nu: 5.0 }, NoiseTargeting::Uniform),
        (NoiseShape::StudentT { nu: 3.0 }, NoiseTargeting::Uniform),
        (NoiseShape::Laplace, NoiseTargeting::Uniform),
        (
            NoiseShape::Gaussian,
            NoiseTargeting::Outlier { p: 0.01, lambda: 3.0 },
        ),
        (
            NoiseShape::Gaussian,
            NoiseTargeting::Outlier { p: 0.05, lambda: 3.0 },
        ),
        (
            NoiseShape::Gaussian,
            NoiseTargeting::Outlier { p: 0.10, lambda: 3.0 },
        ),
    ];
    if groups.is_some() {
        types.push((
            NoiseShape::Gaussian,
            NoiseTargeting::GroupedWide {
                lambda: 3.0,
                group_fraction: 0.2,
            },
        ));
        types.push((
            NoiseShape::Gaussian,
            NoiseTargeting::GroupedShift {
                group_variance_share: 0.62,
            },
        ));
    } else {
        println!("(no scaffold group file given — the two grouped types are not covered)\n");
    }

    for level in [0.25f32, 0.5, 1.0] {
        let target = level * clean_sd;
        println!("=== level {} of the label spread -> target dose {:.6} ===", level, target);
        for (shape, targeting) in &types {
            let spec = NoiseSpec {
                shape: *shape,
                targeting: *targeting,
                level,
                units: DoseUnits::Spread,
                seed: 42,
            };
            match build_noise_plan(labels, canonical, &spec, groups) {
                Ok(plan) => {
                    let err = (plan.realised_dose_label_units / target - 1.0) * 100.0;
                    let tol = dose_tolerance(shape, &plan.epsilon, plan.effective_n) * 100.0;
                    let ok = err.abs() <= tol;
                    if !ok {
                        failures += 1;
                    }
                    let beyond_3x = plan
                        .epsilon
                        .iter()
                        .filter(|e| e.abs() > 3.0 * target)
                        .count() as f32
                        / plan.epsilon.len() as f32;
                    println!(
                        "  {:<34} G={:.4} scale={:.4} delivered={:.6} err={:+.2}% (tol {:.1}%) affected={:.1}% >3x={:.2}% {}",
                        plan.noise_type,
                        plan.unit_dose_g,
                        plan.solved_scale,
                        plan.realised_dose_label_units,
                        err,
                        tol,
                        plan.affected_molecule_fraction * 100.0,
                        beyond_3x * 100.0,
                        if ok { "ok" } else { "FAIL" }
                    );
                }
                Err(e) => {
                    failures += 1;
                    println!("  {:<34} FAIL: {}", targeting.name(), e);
                }
            }
        }
        println!();
    }

    // Gate 1, the form NOISE_DESIGN.md §2a rule 3 specifies: the MEAN delivered dose
    // over at least 20 seeds must sit on the target, and on every other condition's
    // mean. One realisation wobbles — grouped-shifted by about 5% on QM9, Student-t
    // at nu=3 by more — and that spread is a number the Methods states rather than a
    // check that fails. The mean is where the confound would show.
    println!("gate: the mean delivered dose over 20 seeds is flat across noise types");
    let seeds: Vec<u64> = (0..20u64).map(|i| 1000 + i * 7919).collect();
    let level = 0.5f32;
    let target = level * clean_sd;
    let mut means: Vec<(String, f32, f32)> = Vec::new();
    for (shape, targeting) in &types {
        if !targeting.is_dose_matched() {
            continue;
        }
        let mut delivered: Vec<f32> = Vec::new();
        let mut label = String::new();
        for seed in &seeds {
            let spec = NoiseSpec {
                shape: *shape,
                targeting: *targeting,
                level,
                units: DoseUnits::Spread,
                seed: *seed,
            };
            let plan = build_noise_plan(labels, canonical, &spec, groups).unwrap();
            // the construction identity holds on every single draw
            let constructed = plan.unit_dose_g * plan.solved_scale;
            if (constructed - target).abs() > target.abs() * 1e-5 {
                println!(
                    "  {} FAIL: constructs {:.9} against a target of {:.9}",
                    plan.noise_type, constructed, target
                );
                failures += 1;
            }
            label = plan.noise_type.clone();
            delivered.push(plan.realised_dose_label_units);
        }
        let mean = population_mean(&delivered);
        let sd = population_sd(&delivered);
        // the standard error of the mean over 20 seeds
        let se_of_mean = sd / (seeds.len() as f32).sqrt();
        let err = mean / target - 1.0;
        // three standard errors of the mean, floored so a tiny spread cannot make the
        // gate unreasonably strict
        let band = (3.0 * se_of_mean / target).max(0.005);
        let ok = err.abs() <= band;
        if !ok {
            failures += 1;
        }
        println!(
            "  {:<34} mean={:.6} ({:+.2}%)  per-run SD={:.2}%  band={:.2}%  {}",
            label,
            mean,
            err * 100.0,
            100.0 * sd / target,
            band * 100.0,
            if ok { "ok" } else { "FAIL" }
        );
        means.push((label, mean, sd));
    }
    // and the means agree with each other, which is the confound gate proper
    if means.len() > 1 {
        let lo = means.iter().map(|m| m.1).fold(f32::INFINITY, f32::min);
        let hi = means.iter().map(|m| m.1).fold(f32::NEG_INFINITY, f32::max);
        let spread = hi / lo - 1.0;
        let ok = spread <= 0.03;
        if !ok {
            failures += 1;
        }
        println!(
            "  spread between the noise types' mean delivered dose: {:.2}%  {}",
            spread * 100.0,
            if ok { "ok" } else { "FAIL — the noise types are NOT dose-matched" }
        );
    }
    println!();

    // Gate 5 — exactly zero at level zero, for every type.
    print!("gate: zero level records exactly zero ... ");
    let mut zero_ok = true;
    for (shape, targeting) in &types {
        let spec = NoiseSpec {
            shape: *shape,
            targeting: *targeting,
            level: 0.0,
            units: DoseUnits::Spread,
            seed: 42,
        };
        let plan = build_noise_plan(labels, canonical, &spec, groups).unwrap();
        if !plan.epsilon.iter().all(|e| *e == 0.0) {
            zero_ok = false;
        }
    }
    if zero_ok {
        println!("ok");
    } else {
        println!("FAIL");
        failures += 1;
    }

    // Gate 7 — Student-t reduces to Gaussian in the limit.
    print!("gate: Student-t at nu=200 matches Gaussian ... ");
    let gauss = build_noise_plan(
        labels,
        canonical,
        &NoiseSpec {
            shape: NoiseShape::Gaussian,
            targeting: NoiseTargeting::Uniform,
            level: 0.5,
            units: DoseUnits::Spread,
            seed: 7,
        },
        groups,
    )
    .unwrap();
    let t200 = build_noise_plan(
        labels,
        canonical,
        &NoiseSpec {
            shape: NoiseShape::StudentT { nu: 200.0 },
            targeting: NoiseTargeting::Uniform,
            level: 0.5,
            units: DoseUnits::Spread,
            seed: 7,
        },
        groups,
    )
    .unwrap();
    let tail_g = gauss
        .epsilon
        .iter()
        .filter(|e| e.abs() > 3.0 * gauss.target_dose_label_units)
        .count() as f32
        / gauss.epsilon.len() as f32;
    let tail_t = t200
        .epsilon
        .iter()
        .filter(|e| e.abs() > 3.0 * t200.target_dose_label_units)
        .count() as f32
        / t200.epsilon.len() as f32;
    let dose_gap = (t200.realised_dose_label_units / gauss.realised_dose_label_units - 1.0).abs();
    if dose_gap <= 0.02 && (tail_t - tail_g).abs() <= 0.005 {
        println!(
            "ok (dose gap {:+.2}%, tail beyond 3x: gaussian {:.2}% vs t {:.2}%)",
            dose_gap * 100.0,
            tail_g * 100.0,
            tail_t * 100.0
        );
    } else {
        println!(
            "FAIL (dose gap {:+.2}%, tail beyond 3x: gaussian {:.2}% vs t {:.2}%)",
            dose_gap * 100.0,
            tail_g * 100.0,
            tail_t * 100.0
        );
        failures += 1;
    }

    // Censoring is not dose-matched; report what it delivers so the level grid can
    // be read off, and check it clips the fraction it was asked to clip.
    println!("\ncensoring — not dose-matched, reported on its own axis:");
    for fraction in [0.10f32, 0.20, 0.25, 0.30, 0.40, 0.50] {
        let plan = build_noise_plan(
            labels,
            canonical,
            &NoiseSpec {
                shape: NoiseShape::Gaussian,
                targeting: NoiseTargeting::Censoring {
                    side: CensorSide::Upper,
                },
                level: fraction,
                units: DoseUnits::Spread,
                seed: 42,
            },
            groups,
        )
        .unwrap();
        let ok = (plan.affected_molecule_fraction - fraction).abs() <= 0.01;
        if !ok {
            failures += 1;
        }
        println!(
            "  censored {:>4.0}%  clipped={:.1}%  delivered={:.6} ({:.3} of spread)  mean shift={:+.6}  {}",
            fraction * 100.0,
            plan.affected_molecule_fraction * 100.0,
            plan.realised_dose_label_units,
            plan.realised_dose_fraction_of_spread,
            plan.mean_epsilon,
            if ok { "ok" } else { "FAIL" }
        );
    }

    failures
}

fn read_smiles_data(
    reader: &mut BufReader<File>,
    molecular_representations: Vec<String>,
    _k_domains: usize,
) -> Option<SmilesData> {
    // Helper: read a 4-byte length-prefixed UTF-8 string
    fn read_len_prefixed_string(reader: &mut BufReader<File>) -> Option<String> {
        let mut len_buf = [0u8; 4];
        reader.read_exact(&mut len_buf).ok()?;
        let str_len = u32::from_le_bytes(len_buf) as usize;
        let mut buf = vec![0u8; str_len];
        reader.read_exact(&mut buf).ok()?;
        String::from_utf8(buf).ok()
    }

    // Read isomeric_smiles and check validity
    let isomeric_smiles = read_len_prefixed_string(reader)?;
    if isomeric_smiles.len() < 5 || isomeric_smiles.len() > 300 || isomeric_smiles.contains(['\u{FFFD}', '\0', '\'', '�']) {
        eprintln!("Skipping malformed isomeric_smiles: {:?}", isomeric_smiles);
        return None;
    }

    // Read canonical_smiles
    let canonical_smiles = read_len_prefixed_string(reader)?;

    // Read property_value (float)
    let mut prop_buf = [0u8; 4];
    reader.read_exact(&mut prop_buf).ok()?;
    let target_value = f32::from_le_bytes(prop_buf);

    // Read randomized_smiles if applicable
    let mut randomized_smiles = None;
    if molecular_representations.contains(&"randomized_smiles".to_string()) {
        let mut len_buf = [0u8; 4];
        reader.read_exact(&mut len_buf).ok()?;
        let rand_len = u32::from_le_bytes(len_buf) as usize;
        if rand_len > 0 {
            let mut rand_buf = vec![0u8; rand_len];
            reader.read_exact(&mut rand_buf).ok()?;
            randomized_smiles = String::from_utf8(rand_buf).ok();
        }
    }

    // Read sns_fp if applicable
    let mut sns_buf = [0u8; 128];
    if molecular_representations.contains(&"sns".to_string()) {
        reader.read_exact(&mut sns_buf).ok()?;
    }

    // Read pdv (optional, 800 bytes)
    let mut pdv_buf = [0u8; 25];
    if molecular_representations.contains(&"pdv".to_string()) {
        reader.read_exact(&mut pdv_buf).ok()?; 
    }

    let mut continuous_pdv_buf = [0u8; 800];
    if molecular_representations.contains(&"continuous_pdv".to_string()) {
        reader.read_exact(&mut continuous_pdv_buf).ok()?;
    }

    let mut mol2vec_buf = [0u8; 1200];
    if molecular_representations.contains(&"mol2vec".to_string()) {
        reader.read_exact(&mut mol2vec_buf).ok()?; 
    }

    let mut chemberta_buf = [0u8; 3072];
    if molecular_representations.contains(&"chemberta".to_string()) {
        reader.read_exact(&mut chemberta_buf).ok()?;
    }

    let mut mhggnn_buf = [0u8; 4096];
    if molecular_representations.contains(&"mhggnn".to_string()) {
        reader.read_exact(&mut mhggnn_buf).ok()?;
    }

    let mut morgan_buf = [0u8; 256];
    if molecular_representations.contains(&"morgan".to_string()) {
        reader.read_exact(&mut morgan_buf).ok()?;
    }

    // Avalon: 2048 bits packed to 256 bytes in Python, passed straight through,
    // exactly like morgan. Binary, so nothing here rescales or standardises it.
    let mut avalon_buf = [0u8; 256];
    if molecular_representations.contains(&"avalon".to_string()) {
        reader.read_exact(&mut avalon_buf).ok()?;
    }

    // Store parsed data
    Some(SmilesData {
        isomeric_smiles,
        canonical_smiles,
        randomized_smiles,
        target_value,
        sns_buf,
        pdv_buf,
        continuous_pdv_buf,
        mol2vec_buf,
        chemberta_buf,
        mhggnn_buf,
        morgan_buf,
        avalon_buf,
    })
}

/// Read the clean training labels, with the canonical SMILES each one belongs to.
///
/// The SMILES column is not decoration. The noise is built in this order and
/// applied in a second pass over the same file, so the write path can assert that
/// row `i` on the way out is the same molecule row `i` was on the way in. The
/// original held-out bug was exactly this going unchecked.
fn read_train_labels(config: &Config) -> io::Result<(Vec<String>, Vec<f32>)> {
    let train_file = File::open(format!("train_{}.mmap", config.file_no))?;
    let mut reader = BufReader::new(train_file);
    reader.seek(SeekFrom::Start(0))?;

    let mut canonical = Vec::with_capacity(config.train_count);
    let mut labels = Vec::with_capacity(config.train_count);

    for index in 0..config.train_count {
        match read_smiles_data(
            &mut reader,
            config.molecular_representations.clone(),
            config.k_domains,
        ) {
            Some(d) => {
                canonical.push(d.canonical_smiles);
                labels.push(d.target_value);
            }
            // Was `break`, which silently returned a short label vector: the
            // noise plan would then be built over fewer molecules than the file
            // holds, and nothing downstream would say so (RERUN_PLAN.md §2.7).
            // The write side indexes the plan by record position, so a short
            // plan is a wrong plan, not a smaller one.
            None => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "training record {} could not be read but the configuration says there \
                         are {}. The record stream is truncated or misaligned; refusing to build \
                         a noise plan over a partial training set.",
                        index, config.train_count
                    ),
                ));
            }
        }
    }

    Ok((canonical, labels))
}

/// One molecule that could not be fingerprinted, and why.
#[derive(Debug, Clone)]
struct FeaturisationFailure {
    split: String,
    index: usize,
    canonical_smiles: String,
    reason: String,
}

/// Build the 256-byte ECFP4 block for one molecule.
///
/// Returns the block and, when it could not be computed, the reason. On failure
/// the block is 256 zero bytes: the record must stay the same length whatever
/// happens, because a short record shifts the read offset of every molecule
/// after it and the whole file is silently misparsed from that point on
/// (RERUN_PLAN.md §2.7). The failure is not swallowed — it is recorded, and the
/// run refuses to finish with any recorded failure unless explicitly permitted,
/// so a zero fingerprint can never reach a model as if it were real features.
fn prepare_ecfp4(isomeric_smiles: &str) -> (Vec<u8>, Option<String>) {
    let_cxx_string!(smiles_cxx = isomeric_smiles.to_string());
    match smiles_to_mol(&smiles_cxx) {
        Ok(mol) => {
            let fingerprint = rdk_fingerprint_mol(&mol);
            let cxx_vec_ptr: UniquePtr<CxxVector<u64>> = explicit_bit_vect_to_u64_vec(&fingerprint);
            let cxx_vec_ref: &CxxVector<u64> = &*cxx_vec_ptr;
            let u64_vec: Vec<u64> = cxx_vec_ref.iter().copied().collect();

            if u64_vec.len() != 32 {
                return (
                    vec![0u8; 256],
                    Some(format!(
                        "fingerprint is not 2048 bits ({} chunks of 64, expected 32)",
                        u64_vec.len()
                    )),
                );
            }

            let mut packed_fingerprint = vec![0u8; 256];
            for (i, chunk) in u64_vec.iter().enumerate() {
                packed_fingerprint[i * 8..(i + 1) * 8].copy_from_slice(&chunk.to_le_bytes());
            }
            (packed_fingerprint, None)
        }
        Err(_) => (
            vec![0u8; 256],
            Some("RDKit could not parse the SMILES".to_string()),
        ),
    }
}

fn write_data(
    reader: &mut BufReader<File>,
    writer: &mut BufWriter<File>,
    config: &Config,
    mean: f32,
    std_dev: f32,
    plan: &NoisePlan,
    tokenizer: &SmilesTokenizer,
    vocab: &HashMap<String, usize>,
    vocab_size: usize,
    max_sequence_length: usize,
    data_count: usize,
    log_writes: bool,
    apply_noise: bool,
    split_name: &str,
    provenance: &mut BufWriter<File>,
    failures: &mut Vec<FeaturisationFailure>,
) -> io::Result<()> {
    let wants_ecfp4 = config
        .molecular_representations
        .contains(&"ecfp4".to_string());

    for index in 0..data_count {
        if let Some(smiles_data) = read_smiles_data(
            reader,
            config.molecular_representations.clone(),
            config.k_domains,
        ) {

            if log_writes {
                println!("Writing data for index: {}", index);
            }

            // Decided BEFORE a single byte of this record is written. The old
            // code computed the fingerprint in place at the end of the record
            // and used `continue` on failure, which left the record short by
            // 256 bytes with everything before it already on disk.
            let ecfp4_block = if wants_ecfp4 {
                let (block, failure) = prepare_ecfp4(&smiles_data.isomeric_smiles);
                if let Some(reason) = failure {
                    failures.push(FeaturisationFailure {
                        split: split_name.to_string(),
                        index,
                        canonical_smiles: smiles_data.canonical_smiles.clone(),
                        reason,
                    });
                }
                Some(block)
            } else {
                None
            };

            // Write isomeric_smiles with length prefix
            let iso_bytes = smiles_data.isomeric_smiles.as_bytes();
            let iso_len_bytes = (iso_bytes.len() as u32).to_le_bytes();
            writer.write_all(&iso_len_bytes)?;
            writer.write_all(iso_bytes)?;
            if log_writes {
                println!("isomeric_smiles: {}", smiles_data.isomeric_smiles);
                println!("isomeric_smiles_len bytes: {:02X?}", iso_len_bytes);
                println!("isomeric_smiles bytes: {:02X?}", iso_bytes);
            }

            // Write canonical_smiles with length prefix
            let canon_bytes = smiles_data.canonical_smiles.as_bytes();
            let canon_len_bytes = (canon_bytes.len() as u32).to_le_bytes();
            writer.write_all(&canon_len_bytes)?;
            writer.write_all(canon_bytes)?;
            if log_writes {
                println!("canonical_smiles: {}", smiles_data.canonical_smiles);
                println!("canonical_smiles_len bytes: {:02X?}", canon_len_bytes);
                println!("canonical_smiles bytes: {:02X?}", canon_bytes);
            }

            // Write target value
            let target_bytes = smiles_data.target_value.to_le_bytes();
            writer.write_all(&target_bytes)?;
            if log_writes {
                println!("property_value: {}", smiles_data.target_value);
                println!("property_value bytes: {:02X?}", target_bytes);
            }

            // Write randomized_smiles if exists
            if let Some(randomized) = &smiles_data.randomized_smiles {
                let bytes = randomized.as_bytes();
                let len_bytes = (bytes.len() as u32).to_le_bytes();
                writer.write_all(&len_bytes)?;
                writer.write_all(bytes)?;
                if log_writes {
                    println!("randomized_smiles: {}", randomized);
                    println!("randomized_smiles_len bytes: {:02X?}", len_bytes);
                    println!("randomized_smiles bytes: {:02X?}", bytes);
                }
            }

            // Write sns_fp
            if config.molecular_representations.contains(&"sns".to_string()) {
                let sns_fp = smiles_data.sns_buf;
                writer.write_all(&sns_fp)?;
                if log_writes {
                    println!("sns_fp: {:?}", sns_fp);
                }
            }

            // Write pdv (800 bytes)
            if config.molecular_representations.contains(&"pdv".to_string()) {
                let pdv = smiles_data.pdv_buf;
                writer.write_all(&pdv)?;
                if log_writes {
                    println!("pdv: {:?}", pdv);
                }
            }

            // continuous_pdv (800 bytes, float32)
            if config.molecular_representations.contains(&"continuous_pdv".to_string()) {
                let continuous_pdv = smiles_data.continuous_pdv_buf;
                writer.write_all(&continuous_pdv)?;
                if log_writes {
                    println!("continuous_pdv: {:?}", continuous_pdv);
                }
            }

            // mol2vec (1200 bytes, float32)
            if config.molecular_representations.contains(&"mol2vec".to_string()) {
                let mol2vec = smiles_data.mol2vec_buf;
                writer.write_all(&mol2vec)?;
                if log_writes {
                    println!("mol2vec: {:?}", mol2vec);
                }
            }

            // chemberta (3072 bytes, float32)
            if config.molecular_representations.contains(&"chemberta".to_string()) {
                let chemberta = smiles_data.chemberta_buf;
                writer.write_all(&chemberta)?;
                if log_writes {
                    println!("chemberta: {:?}", chemberta);
                }
            }

            // mhggnn (4096 bytes, float32)
            if config.molecular_representations.contains(&"mhggnn".to_string()) {
                let mhggnn = smiles_data.mhggnn_buf;
                writer.write_all(&mhggnn)?;
                if log_writes {
                    println!("mhggnn: {:?}", mhggnn);
                }
            }

            // morgan (256 bytes, ECFP4 radius=2 computed in Python)
            if config.molecular_representations.contains(&"morgan".to_string()) {
                let morgan = smiles_data.morgan_buf;
                writer.write_all(&morgan)?;
                if log_writes {
                    println!("morgan: {:?}", morgan);
                }
            }

            // avalon (256 bytes, packed bits computed in Python)
            if config.molecular_representations.contains(&"avalon".to_string()) {
                let avalon = smiles_data.avalon_buf;
                writer.write_all(&avalon)?;
                if log_writes {
                    println!("avalon: {:?}", avalon);
                }
            }

            // Add noise to the label.
            //
            // `apply_noise` is true ONLY for the training split. The plan is built
            // in training-record order, and this loop restarts its counter at 0 for
            // every split — so applying it to val or test would hand each held-out
            // molecule the noise drawn for the training molecule at the same
            // position. That is the original bug (RERUN_PLAN.md §2.1): it corrupted
            // the held-out labels, contrary to the Methods, and attached the
            // corruption to the wrong molecules.
            //
            // The identity check below is what stops that class of mistake rather
            // than this one instance of it. Row `index` on the way out must be the
            // same molecule as row `index` on the way in, or the run dies here.
            let y_clean_raw = smiles_data.target_value;
            let epsilon_raw = if apply_noise {
                assert!(
                    index < plan.epsilon.len(),
                    "guard: training record {} has no entry in the noise plan (plan holds {})",
                    index,
                    plan.epsilon.len()
                );
                assert_eq!(
                    plan.canonical[index], smiles_data.canonical_smiles,
                    "guard: training record {} is molecule {} on the way out but was {} \
                     when the noise was drawn — the record stream has drifted",
                    index, smiles_data.canonical_smiles, plan.canonical[index]
                );
                plan.epsilon[index]
            } else {
                // Held-out labels are never touched. Recorded as exactly zero, so the
                // provenance file itself is the evidence for that.
                0.0
            };

            let y_noisy_raw = y_clean_raw + epsilon_raw;

            // Standardise with the CLEAN training mean and spread. Standardising with
            // the noisy spread (the old behaviour) moved the target scale with the
            // noise level, so "the same amount of noise" meant something different at
            // every level (RERUN_PLAN.md §2.4).
            let mut property_value = y_noisy_raw;
            if config.regression && config.normalize {
                property_value = (property_value - mean) / std_dev;
            }

            writeln!(
                provenance,
                "{},{},{},{:.9e},{:.9e},{:.9e},{:.9e}",
                split_name,
                index,
                smiles_data.canonical_smiles,
                y_clean_raw,
                epsilon_raw,
                y_noisy_raw,
                property_value
            )?;

            let processed_bytes = property_value.to_le_bytes();
            writer.write_all(&processed_bytes)?;
            if log_writes {
                println!("noisy y: {}", property_value);
                println!("noisy y bytes: {:02X?}", processed_bytes);
            }

            // Write domain label if applicable
            if config.k_domains > 1 {
                writer.write_all(&[0u8])?;
                if log_writes {
                    println!("domain_flag: 0");
                    println!("domain_flag bytes: 00");
                }
            }

            // Write smiles or randomized_smiles OHE if used
            for smiles_type in ["smiles", "randomized_smiles"] {
                if config.molecular_representations.contains(&smiles_type.to_string()) {
                    let smiles_string = if smiles_type == "smiles" {
                        &smiles_data.canonical_smiles
                    } else {
                        smiles_data.randomized_smiles.as_ref().unwrap()
                    };

                    let smiles_ohe = smiles_to_ohe(
                        smiles_string,
                        tokenizer,
                        vocab,
                        vocab_size,
                        max_sequence_length,
                    );

                    let bit_packed_len = (smiles_ohe.len() + 7) / 8;
                    let mut bit_packed_data = vec![0u8; bit_packed_len];
                    for (i, &bit) in smiles_ohe.iter().enumerate() {
                        let byte_index = i / 8;
                        let bit_offset = i % 8;
                        if bit > 0.0 {
                            bit_packed_data[byte_index] |= 1 << bit_offset;
                        }
                    }

                    let len_bytes = (bit_packed_data.len() as u32).to_le_bytes();
                    writer.write_all(&len_bytes)?;
                    writer.write_all(&bit_packed_data)?;
                    if log_writes {
                        println!("{}: {:?}", smiles_type, bit_packed_data);
                        println!("{}_ohe_len bytes: {:02X?}", smiles_type, len_bytes);
                        println!("{}_ohe bytes: {:02X?}", smiles_type, bit_packed_data);
                    }
                }
            }

            // Write ECFP4 fingerprint. Always 256 bytes, always written — the
            // block was prepared at the top of this record and a failure has
            // already been recorded against `failures`.
            if let Some(packed_fingerprint) = &ecfp4_block {
                writer.write_all(packed_fingerprint)?;
                if log_writes {
                    println!("ecfp4_fingerprint: {:?}", packed_fingerprint);
                }
            }

            if log_writes {
                println!("Finished writing entry {}\n", index);
            }
        }
    }

    writer.flush()?;

    Ok(())
}



fn tanimoto_distance(fp1: &Vec<u64>, fp2: &Vec<u64>) -> f32 {
    let intersection = fp1.iter().zip(fp2.iter()).map(|(&a, &b)| (a & b).count_ones()).sum::<u32>();
    let union = fp1.iter().zip(fp2.iter()).map(|(&a, &b)| (a | b).count_ones()).sum::<u32>();
    1.0 - (intersection as f32 / union as f32)
}

fn smiles_to_ohe(smiles: &str, tokenizer: &SmilesTokenizer, vocab: &HashMap<String, usize>, vocab_size: usize, max_length: usize) -> Array2<f32> {
    let tokens = tokenizer.tokenize(smiles);
    let mut ohe = Array2::<f32>::zeros((max_length, vocab_size));
    for (i, token) in tokens.iter().enumerate().take(max_length) {
        if let Some(&index) = vocab.get(token) {
            if i < max_length {
                ohe[(i, index)] = 1.0;
            }
        }
    }
    ohe
}

fn mean_absolute_error<T: Float + FromPrimitive + Sum<T>>(y_true: &[T], y_pred: &[T]) -> T {
    y_true.iter().zip(y_pred.iter())
          .map(|(true_val, pred_val)| (*true_val - *pred_val).abs())
          .sum::<T>() / T::from_usize(y_true.len()).unwrap()
}

fn mean_squared_error<T: Float + FromPrimitive + Sum<T>>(y_true: &[T], y_pred: &[T]) -> T {
    y_true.iter().zip(y_pred.iter())
          .map(|(true_val, pred_val)| (*true_val - *pred_val).powi(2))
          .sum::<T>() / T::from_usize(y_true.len()).unwrap()
}

fn root_mean_squared_error<T: Float + FromPrimitive + Sum<T>>(y_true: &[T], y_pred: &[T]) -> T {
    mean_squared_error(y_true, y_pred).sqrt()
}

fn r2_score<T: Float + FromPrimitive + Sum<T>>(y_true: &[T], y_pred: &[T]) -> T {
    let mean_true = y_true.iter().map(|&x| x).sum::<T>() / T::from_usize(y_true.len()).unwrap();
    let ss_tot = y_true.iter().map(|&x| (x - mean_true).powi(2)).sum::<T>();
    let ss_res = y_true.iter().zip(y_pred.iter())
                       .map(|(&true_val, &pred_val)| (true_val - pred_val).powi(2))
                       .sum::<T>();
    T::one() - (ss_res / ss_tot)
}

fn count_token_frequencies(smiles_list: &[String], tokenizer: &SmilesTokenizer) -> HashMap<String, usize> {
    let mut token_counts: HashMap<String, usize> = HashMap::new();
    for smiles in smiles_list {
        let tokens = tokenizer.tokenize(smiles);
        for token in tokens {
            *token_counts.entry(token).or_insert(0) += 1;
        }
    }
    token_counts
}

fn trim_vocab<T: Eq + Hash + Ord + Clone>(token_counts: HashMap<T, usize>, max_vocab_size: usize) -> HashMap<T, usize> {
    let mut token_counts_vec: Vec<(T, usize)> = token_counts.into_iter().collect();

    // Sort tokens by count, descending
    token_counts_vec.sort_by_key(|&(_, count)| Reverse(count));

    // Truncate to keep only the top max_vocab_size tokens
    token_counts_vec.truncate(max_vocab_size);

    let trimmed_vocab: HashMap<T, usize> = token_counts_vec.into_iter()
        .enumerate()
        .map(|(idx, (token, _))| (token, idx))
        .collect();

    trimmed_vocab
}

/// Vocabulary, sequence length, and the standardisation constants.
///
/// The constants come from the CLEAN training labels. They used to come from the
/// noisy ones (RERUN_PLAN.md §2.4), which made the standardised target scale a
/// function of the noise level: the same nominal amount of noise produced a
/// different learning problem at every level, and the paper's claim that noise was
/// added "on normalized data" was false besides. Nothing here sees the noise now.
fn generate_aggregate_stats(
    config: &Config,
) -> io::Result<(f32, f32, usize, HashMap<String, usize>, usize)> {
    let tokenizer = SmilesTokenizer::new();
    let mut smiles_list: Vec<String> = Vec::new();
    let mut y_values: Vec<f32> = Vec::new();
    let mut max_sequence_length = 0usize;

    let train_file = File::open(format!("train_{}.mmap", config.file_no))?;
    let mut reader = BufReader::new(train_file);
    reader.seek(SeekFrom::Start(0))?;

    for _index in 0..config.train_count {
        if let Some(smiles_data) = read_smiles_data(&mut reader, config.molecular_representations.clone(), config.k_domains) {
            if ["smiles", "randomized_smiles"].iter().any(|r| config.molecular_representations.contains(&r.to_string())) {
                smiles_list.push(smiles_data.canonical_smiles.clone());
                let tokens = tokenizer.tokenize(&smiles_data.canonical_smiles);
                max_sequence_length = std::cmp::max(max_sequence_length, tokens.len());
            }

            y_values.push(smiles_data.target_value);
        }
    }

    let token_counts = count_token_frequencies(&smiles_list, &tokenizer);
    let trimmed_vocab = trim_vocab(token_counts, config.max_vocab);
    let vocab_size = trimmed_vocab.len();

    let mean: f32 = y_values.iter().sum::<f32>() / y_values.len() as f32;
    let variance: f32 = y_values.iter().map(|value| {
        let diff = mean - *value;
        diff * diff
    }).sum::<f32>() / y_values.len() as f32;
    let std_deviation: f32 = variance.sqrt();

    Ok((mean, std_deviation, vocab_size, trimmed_vocab, max_sequence_length))
}

fn preprocess_data(
    config: &Config,
    mean: f32,
    std_dev: f32,
    vocab_size: usize,
    vocab: &HashMap<String, usize>,
    plan: &NoisePlan,
    max_sequence_length: usize,
    provenance_path: &str,
    failures_path: &str,
    allow_featurisation_failures: bool,
) -> io::Result<()> {
    let tokenizer = SmilesTokenizer::new();

    // Molecules whose fingerprint could not be computed. Collected across all
    // three splits and dealt with at the end of this function.
    let mut failures: Vec<FeaturisationFailure> = Vec::new();

    // Per-molecule provenance, every split. The injected value is RECORDED here,
    // where it is applied — never reconstructed downstream by fitting a line
    // (RERUN_PLAN.md §0.6, failure mode 2). `record_index` plus `canonical_smiles`
    // is what makes a row linkable to a molecule and matchable across replicates.
    let mut provenance = BufWriter::new(
        OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(provenance_path)?,
    );
    writeln!(
        provenance,
        "split,record_index,canonical_smiles,y_clean_raw,epsilon_raw,y_noisy_raw,y_written"
    )?;

    let train_file_path = format!("train_{}.mmap", config.file_no);
    let train_file_new_path = format!("train_{}_new.mmap", config.file_no);
    let test_file_path = format!("test_{}.mmap", config.file_no);
    let test_file_new_path = format!("test_{}_new.mmap", config.file_no);
    let val_file_path = format!("val_{}.mmap", config.file_no);
    let val_file_new_path = format!("val_{}_new.mmap", config.file_no);

    let train_file = File::open(&train_file_path)?;
    let test_file = File::open(&test_file_path)?;
    let val_file = File::open(&val_file_path)?;

    let mut train_reader = BufReader::new(train_file);
    let mut train_writer = BufWriter::new(
        OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&train_file_new_path)?
    );

    train_reader.seek(SeekFrom::Start(0))?;
    write_data(
        &mut train_reader,
        &mut train_writer,
        config,
        mean,
        std_dev,
        plan,
        &tokenizer,
        vocab,
        vocab_size,
        max_sequence_length,
        config.train_count,
        config.logging,
        true,   // apply_noise
        "train",
        &mut provenance,
        &mut failures,
    )?;
    remove_file(&train_file_path)?;
    rename(&train_file_new_path, &train_file_path)?;

    let mut val_reader = BufReader::new(val_file);
    let mut val_writer = BufWriter::new(
        OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&val_file_new_path)?
    );
    val_reader.seek(SeekFrom::Start(0))?;
    write_data(
        &mut val_reader,
        &mut val_writer,
        config,
        mean,
        std_dev,
        plan,
        &tokenizer,
        vocab,
        vocab_size,
        max_sequence_length,
        config.val_count,
        config.logging,
        false,   // apply_noise
        "val",
        &mut provenance,
        &mut failures,
    )?;
    remove_file(&val_file_path)?;
    rename(&val_file_new_path, &val_file_path)?;

    let mut test_reader = BufReader::new(test_file);
    let mut test_writer = BufWriter::new(
        OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&test_file_new_path)?
    );
    test_reader.seek(SeekFrom::Start(0))?;
    write_data(
        &mut test_reader,
        &mut test_writer,
        config,
        mean,
        std_dev,
        plan,
        &tokenizer,
        vocab,
        vocab_size,
        max_sequence_length,
        config.test_count,
        config.logging,
        false,   // apply_noise
        "test",
        &mut provenance,
        &mut failures,
    )?;
    remove_file(&test_file_path)?;
    rename(&test_file_new_path, &test_file_path)?;

    provenance.flush()?;

    // A molecule that could not be fingerprinted was written with a 256-byte
    // zero block so the record stayed full length and the reader stayed
    // aligned. That keeps the FILE correct; it does not make the FEATURES
    // correct. A zero fingerprint carries a real label into training as if it
    // were a real molecule, so the run stops here rather than handing a model
    // blank inputs (RERUN_PLAN.md §2.7, gate 8).
    if !failures.is_empty() {
        let mut f = BufWriter::new(
            OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(failures_path)?,
        );
        writeln!(f, "split,record_index,canonical_smiles,reason")?;
        for fail in &failures {
            writeln!(
                f,
                "{},{},{},{}",
                fail.split, fail.index, fail.canonical_smiles, fail.reason
            )?;
        }
        f.flush()?;

        if !allow_featurisation_failures {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "{} molecule(s) could not be fingerprinted and were written as an \
                     all-zero ECFP4 block. They are listed in {}. Refusing to finish: a zero \
                     fingerprint would train as if it were real features. Pass \
                     --allow-featurisation-failures to accept a partial set knowingly.",
                    failures.len(),
                    failures_path
                ),
            ));
        }
        eprintln!(
            "WARNING: {} molecule(s) could not be fingerprinted; written as all-zero ECFP4 \
             blocks and listed in {}. Accepted because --allow-featurisation-failures was given.",
            failures.len(),
            failures_path
        );
    }

    Ok(())
}

/// Labels for the self-test: one per line, either `y` or `canonical_smiles,y`.
/// A header line naming the columns is tolerated and skipped.
fn read_self_test_labels(path: &str) -> io::Result<(Vec<String>, Vec<f32>)> {
    let file = File::open(path)?;
    let mut canonical = Vec::new();
    let mut labels = Vec::new();
    for line in BufReader::new(file).lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let (smiles, value) = match line.rsplit_once(',') {
            Some((left, right)) => (left.to_string(), right.trim().parse::<f32>()),
            None => (String::new(), line.parse::<f32>()),
        };
        match value {
            Ok(v) if v.is_finite() => {
                canonical.push(if smiles.is_empty() {
                    format!("mol_{}", labels.len())
                } else {
                    smiles
                });
                labels.push(v);
            }
            // header line, or an unparseable row
            _ => continue,
        }
    }
    Ok((canonical, labels))
}

fn write_noise_manifest(path: &str, plan: &NoisePlan, spec: &NoiseSpec) -> io::Result<()> {
    let manifest = serde_json::json!({
        "noise_type": plan.noise_type,
        "noise_shape": plan.shape_name,
        "noise_targeting": plan.targeting_name,
        "noise_level": spec.level,
        "unit_dose": plan.unit_dose_g,
        "solved_scale": plan.solved_scale,
        "target_dose_in_label_units": plan.target_dose_label_units,
        "delivered_dose_in_label_units": plan.realised_dose_label_units,
        "delivered_dose_as_fraction_of_label_spread": plan.realised_dose_fraction_of_spread,
        "mean_epsilon": plan.mean_epsilon,
        "affected_molecule_fraction": plan.affected_molecule_fraction,
        "effective_n": plan.effective_n,
        "standardisation_mean": plan.clean_label_mean,
        "standardisation_sd": plan.clean_label_sd,
        "clean_label_mean": plan.clean_label_mean,
        "clean_label_sd": plan.clean_label_sd,
        "seed": plan.seed,
        "n_train": plan.n_train,
        "parameters": plan.params,
    });
    let mut f = BufWriter::new(
        OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?,
    );
    serde_json::to_writer_pretty(&mut f, &manifest)?;
    writeln!(f)?;
    f.flush()?;
    Ok(())
}

fn main() -> io::Result<()> {
    let app = Command::new("My Rust Processor")
        .arg(Arg::new("seed")
             .long("seed")
             .action(ArgAction::Set)
             .help("Random seed for the process"))
        .arg(Arg::new("model")
             .long("model")
             .action(ArgAction::Set)
             .help("Model to use for prediction"))
        .arg(Arg::new("noise_targeting")
             .long("noise-targeting")
             .action(ArgAction::Set)
             .help("Who gets hit: uniform, grouped_wide, grouped_shift, outlier, censoring"))
        .arg(Arg::new("noise_shape")
             .long("noise-shape")
             .action(ArgAction::Set)
             .help("Shape of each draw: gaussian, student_t, laplace"))
        .arg(Arg::new("noise_level")
             .long("noise-level")
             .action(ArgAction::Set)
             .help("The dose to deliver; for censoring, the fraction of labels clipped"))
        .arg(Arg::new("dose_units")
             .long("dose-units")
             .action(ArgAction::Set)
             .help("How --noise-level is read: 'spread' (fraction of the clean label SD) or 'label' (the label's own units)"))
        .arg(Arg::new("nu")
             .long("nu")
             .action(ArgAction::Set)
             .help("Degrees of freedom for Student-t. Must be > 2"))
        .arg(Arg::new("lambda")
             .long("lambda")
             .action(ArgAction::Set)
             .help("How many times wider the affected molecules' error is (grouped_wide, outlier)"))
        .arg(Arg::new("group_fraction")
             .long("group-fraction")
             .action(ArgAction::Set)
             .help("Fraction of scaffold GROUPS affected (grouped_wide)"))
        .arg(Arg::new("group_variance_share")
             .long("group-variance-share")
             .action(ArgAction::Set)
             .help("Share of the total variance carried by the group-level offset (grouped_shift)"))
        .arg(Arg::new("outlier_p")
             .long("outlier-p")
             .action(ArgAction::Set)
             .help("Fraction of labels contaminated (outlier)"))
        .arg(Arg::new("censor_side")
             .long("censor-side")
             .action(ArgAction::Set)
             .help("Which end of the label range the assay limit sits at: upper or lower"))
        .arg(Arg::new("config")
             .long("config")
             .action(ArgAction::Set)
             .required_unless_present("self_test")
             .help("Path to this task's configuration JSON. REQUIRED — there is deliberately \
                    no default: a fixed 'config.json' is shared by every concurrent array \
                    task and one task then opens and rewrites another's training data \
                    (RERUN_PLAN.md §2.8a)"))
        .arg(Arg::new("scaffold_file")
             .long("scaffold-file")
             .action(ArgAction::Set)
             .help("JSON mapping canonical SMILES to scaffold group id (required by the grouped types)"))
        .arg(Arg::new("noise_manifest")
             .long("noise-manifest")
             .action(ArgAction::Set)
             .help("Where to write the run-level noise provenance"))
        .arg(Arg::new("noise_provenance")
             .long("noise-provenance")
             .action(ArgAction::Set)
             .help("Where to write the per-molecule noise provenance"))
        .arg(Arg::new("featurisation_failures")
             .long("featurisation-failures")
             .action(ArgAction::Set)
             .help("Where to list molecules whose fingerprint could not be computed"))
        .arg(Arg::new("allow_featurisation_failures")
             .long("allow-featurisation-failures")
             .action(ArgAction::SetTrue)
             .help("Finish even if some molecules were written with an all-zero ECFP4 block. \
                    Off by default: a zero fingerprint trains as if it were real features"))
        .arg(Arg::new("self_test")
             .long("self-test")
             .action(ArgAction::Set)
             .help("Run the noise gates against a labels file and exit. One label per line, or 'canonical_smiles,y'"));

    let matches = app.get_matches();

    // ---- Shape ------------------------------------------------------------
    let nu: f32 = matches
        .get_one::<String>("nu")
        .map(|v| v.parse().expect("--nu must be a valid float"))
        .unwrap_or(5.0);
    let shape = match matches
        .get_one::<String>("noise_shape")
        .map(|s| s.as_str())
        .unwrap_or("gaussian")
    {
        "gaussian" => NoiseShape::Gaussian,
        "student_t" => {
            // Below nu = 2 the variance of a t is undefined, so "the same amount of
            // noise" stops meaning anything and the run would be silently pointless.
            // Refuse at parse time rather than produce numbers (NOISE_DESIGN.md §6.2).
            assert!(
                nu > 2.0,
                "--nu must be greater than 2: the variance of a Student-t is undefined at or below 2, \
                 so the dose cannot be matched"
            );
            NoiseShape::StudentT { nu }
        }
        "laplace" => NoiseShape::Laplace,
        other => panic!(
            "unknown --noise-shape '{}'. Valid: gaussian, student_t, laplace",
            other
        ),
    };

    // ---- Targeting --------------------------------------------------------
    let lambda: f32 = matches
        .get_one::<String>("lambda")
        .map(|v| v.parse().expect("--lambda must be a valid float"))
        .unwrap_or(3.0);
    let group_fraction: f32 = matches
        .get_one::<String>("group_fraction")
        .map(|v| v.parse().expect("--group-fraction must be a valid float"))
        .unwrap_or(0.2);
    let group_variance_share: f32 = matches
        .get_one::<String>("group_variance_share")
        .map(|v| v.parse().expect("--group-variance-share must be a valid float"))
        .unwrap_or(0.62);
    let outlier_p: f32 = matches
        .get_one::<String>("outlier_p")
        .map(|v| v.parse().expect("--outlier-p must be a valid float"))
        .unwrap_or(0.05);
    let censor_side = match matches
        .get_one::<String>("censor_side")
        .map(|s| s.as_str())
        .unwrap_or("upper")
    {
        "upper" => CensorSide::Upper,
        "lower" => CensorSide::Lower,
        other => panic!("unknown --censor-side '{}'. Valid: upper, lower", other),
    };
    let targeting = match matches
        .get_one::<String>("noise_targeting")
        .map(|s| s.as_str())
        .unwrap_or("uniform")
    {
        "uniform" => NoiseTargeting::Uniform,
        "grouped_wide" => {
            assert!(
                group_variance_share.is_finite(),
                "--group-variance-share is not used by grouped_wide"
            );
            NoiseTargeting::GroupedWide {
                lambda,
                group_fraction,
            }
        }
        "grouped_shift" => {
            assert!(
                (0.0..=1.0).contains(&group_variance_share),
                "--group-variance-share must be in [0, 1]"
            );
            NoiseTargeting::GroupedShift {
                group_variance_share,
            }
        }
        "outlier" => {
            assert!(
                (0.0..1.0).contains(&outlier_p),
                "--outlier-p must be in [0, 1)"
            );
            NoiseTargeting::Outlier {
                p: outlier_p,
                lambda,
            }
        }
        "censoring" => NoiseTargeting::Censoring { side: censor_side },
        other => panic!(
            "unknown --noise-targeting '{}'. Valid: uniform, grouped_wide, grouped_shift, outlier, censoring",
            other
        ),
    };

    let noise_level: f32 = matches
        .get_one::<String>("noise_level")
        .map(|v| v.parse().expect("--noise-level must be a valid float"))
        .unwrap_or(0.0);
    let dose_units = match matches
        .get_one::<String>("dose_units")
        .map(|s| s.as_str())
        .unwrap_or("spread")
    {
        "spread" => DoseUnits::Spread,
        "label" => DoseUnits::Label,
        other => panic!("unknown --dose-units '{}'. Valid: spread, label", other),
    };

    // ---- Self-test mode: the gates, on real labels, with no pipeline. ------
    if let Some(labels_path) = matches.get_one::<String>("self_test") {
        let (canonical, labels) = read_self_test_labels(labels_path)?;
        let groups = match matches.get_one::<String>("scaffold_file") {
            Some(path) => Some(load_scaffold_groups(path)?),
            None => None,
        };
        let failures = self_test(&labels, groups.as_ref(), &canonical);
        if failures > 0 {
            eprintln!("\n{} noise gate(s) FAILED", failures);
            std::process::exit(1);
        }
        println!("\nall noise gates passed");
        return Ok(());
    }

    let seed: u64 = matches
        .get_one::<String>("seed")
        .expect("--seed is required")
        .parse()
        .expect("Seed must be a valid integer");
    let _model = matches.get_one::<String>("model");

    let spec = NoiseSpec {
        shape,
        targeting,
        level: noise_level,
        units: dose_units,
        seed,
    };

    // Reading the configuration file. The path comes from the caller and has no
    // default — see the --config help text.
    let config_path = matches
        .get_one::<String>("config")
        .expect("--config is required");
    let config_file = File::open(config_path).map_err(|e| {
        io::Error::new(
            e.kind(),
            format!("cannot open the configuration file {}: {}", config_path, e),
        )
    })?;
    let reader = BufReader::new(config_file);
    let config: Config = serde_json::from_reader(reader)
                          .expect("JSON was not well-formatted or did not match the expected structure");

    let scaffold_path = matches
        .get_one::<String>("scaffold_file")
        .cloned()
        .unwrap_or_else(|| format!("scaffold_groups_{}.json", config.file_no));
    let manifest_path = matches
        .get_one::<String>("noise_manifest")
        .cloned()
        .unwrap_or_else(|| format!("noise_manifest_{}.json", config.file_no));
    let provenance_path = matches
        .get_one::<String>("noise_provenance")
        .cloned()
        .unwrap_or_else(|| format!("noise_provenance_{}.csv", config.file_no));
    let failures_path = matches
        .get_one::<String>("featurisation_failures")
        .cloned()
        .unwrap_or_else(|| format!("featurisation_failures_{}.csv", config.file_no));
    let allow_featurisation_failures = matches.get_flag("allow_featurisation_failures");

    let (canonical, clean_labels) = read_train_labels(&config)?;
    assert_eq!(
        clean_labels.len(),
        config.train_count,
        "guard: read {} training records but the configuration says {} — the record stream is short",
        clean_labels.len(),
        config.train_count
    );

    let groups = if spec.targeting.needs_groups() {
        Some(load_scaffold_groups(&scaffold_path)?)
    } else {
        None
    };

    let plan = build_noise_plan(&clean_labels, &canonical, &spec, groups.as_ref())?;
    assert_noise_plan_gates(&plan, &spec, &clean_labels);

    println!(
        "noise: {} at level {} ({}) -> unit dose {:.4}, scale {:.6}, delivered {:.6} ({:.4} of the label spread), {:.1}% of molecules affected",
        plan.noise_type,
        spec.level,
        match spec.units {
            DoseUnits::Spread => "fraction of the clean label spread",
            DoseUnits::Label => "label units",
        },
        plan.unit_dose_g,
        plan.solved_scale,
        plan.realised_dose_label_units,
        plan.realised_dose_fraction_of_spread,
        plan.affected_molecule_fraction * 100.0
    );

    write_noise_manifest(&manifest_path, &plan, &spec)?;

    // Standardisation constants come from the CLEAN training labels only.
    let (mean, std_dev, vocab_size, vocab, max_sequence_length) =
        generate_aggregate_stats(&config)?;

    preprocess_data(
        &config,
        mean,
        std_dev,
        vocab_size,
        &vocab,
        &plan,
        max_sequence_length,
        &provenance_path,
        &failures_path,
        allow_featurisation_failures,
    )?;

    Ok(())
}
