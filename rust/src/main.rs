use std::collections::{HashMap, HashSet};
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
    // Sort & Slice: 1024 substructure COUNTS as u16, not one presence bit per
    // substructure. The Python writer flattened the counts to bits and cast to
    // u8 before packing, so a count that was a multiple of 256 wrapped to zero
    // and the substructure recorded as absent (RERUN_PLAN.md 3.4.1). This side
    // only carries the bytes through, but the WIDTH has to match the writer's or
    // every field after it decodes at the wrong offset.
    sns_buf: [u8; 2048],
    pdv_buf: [u8; 25],
    continuous_pdv_buf: [u8; 800],
    // The learned embeddings are 32-bit floats, four bytes a dimension:
    // chemberta 384 dims, mhggnn 1024. They used to be one byte a dimension,
    // min-max rescaled per molecule, which destroyed comparability between
    // molecules (RERUN_PLAN.md 2.8c). These widths and the Python writer's must
    // move together or every record after the first is read at the wrong offset.
    //
    // chemberta was 768 until 2026-08-27, when both pipelines were settled on
    // DeepChem/ChemBERTa-77M-MTR (384 wide). It must match CHEMBERTA_DIMS in
    // scripts/process_and_train.py.
    chemberta_buf: [u8; 1536],
    mhggnn_buf: [u8; 4096],
    avalon_buf: [u8; 256],
    // ECFP4: Morgan radius 2, 2,048 bits, computed in Python and carried
    // through. It used to be computed HERE with `rdk_fingerprint_mol`, which is
    // RDKit's PATH fingerprint (RDKFingerprintMol) -- a different fingerprint
    // from ECFP4 entirely. Measured on the first 1,500 QM9 molecules: the two
    // agreed on ZERO of them, and methane, ammonia and water came back all-zero
    // because a molecule with one heavy atom has no bond paths. The rdkit-sys
    // binding offers only `morgan_fingerprint_mol`, hardcoded to radius 3, so
    // there was no route to radius 2 on this side (RERUN_PLAN.md 2.13).
    ecfp4_buf: [u8; 256],
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

/// The region-level targeting decisions. Drawn ONCE, on the training labels, and
/// then reused by every other split.
///
/// A scaffold group whose measurements are three times worse in training has three
/// times worse measurements in validation as well: "this scaffold is hard to
/// measure" is a property of the scaffold, not of which split a molecule landed in.
/// Redrawing the selection per split would make the validation split a different
/// condition wearing the training condition's name.
///
/// It is also what makes a HELD-OUT molecule's `noise_pattern_raw` answerable at
/// all. That column is "the shape this molecule's region receives", and whether its
/// region is one of the affected ones is a training-side decision — there is
/// nothing to look up without this.
#[derive(Debug, Clone, Default)]
struct TargetingState {
    /// GroupedWide: the scaffold groups carrying the wider error.
    affected_groups: HashSet<u32>,
    /// GroupedShift: the offset drawn once per scaffold group, at unit variance.
    group_offsets: HashMap<u32, f32>,
}

/// Which split is being planned, and against what.
///
/// Everything that defines the CONDITION — the amount of noise, the assay limit,
/// which scaffold groups are hit — is a property of the clean TRAINING labels, and
/// is measured against them whichever split is being noised. Only the draws
/// themselves are per split.
///
/// Anchoring on the validation split's own spread instead is the mistake this type
/// exists to prevent: validation is a tenth the size and, after a scaffold split,
/// need not have the training column's spread at all, so "the same noise level"
/// would mean a different number of label units on each split.
struct PlanContext<'a> {
    /// The clean TRAINING labels. The dose and every cut-point are measured here.
    reference_labels: &'a [f32],
    /// The training split's region-level decisions. `None` when this IS the
    /// training split and they are being made.
    shared: Option<&'a TargetingState>,
    /// Whether this split's labels actually receive the noise. A split that does
    /// not still gets its `noise_pattern_raw` — the shape its region would
    /// receive — but every `epsilon_raw` and `noise_scale_raw` is exactly zero.
    apply: bool,
    /// For gate messages.
    split_name: &'a str,
}

impl<'a> PlanContext<'a> {
    /// The training split: it defines the condition, so it is its own reference.
    fn training(labels: &'a [f32]) -> Self {
        PlanContext {
            reference_labels: labels,
            shared: None,
            apply: true,
            split_name: "train",
        }
    }
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
    /// Per molecule, RAW label units: the amount of noise this molecule receives at
    /// THIS level. For the dose-matched types it is the standard deviation of that
    /// molecule's own draw, which is `level * noise_pattern` exactly. For censoring
    /// — which has no dose axis — it is the size of the shift actually applied.
    /// Exactly zero on a split that receives no noise, and at level zero.
    noise_scale: Vec<f32>,
    /// Per molecule, RAW label units: the same quantity at a FIXED reference level
    /// of 1.0. Identical at every level INCLUDING zero, by construction, because
    /// nothing in it depends on the level.
    ///
    /// This is what makes the zero-level subtraction possible: the model trained at
    /// level zero saw the same labels and no corruption, so its correlation with
    /// this column is exactly the label-magnitude confound and can be subtracted off.
    /// Without it there is no negative control on QM9 at all.
    noise_pattern: Vec<f32>,
    /// The region-level decisions this plan made (training) or inherited.
    targeting_state: TargetingState,
    /// The spread the dose was measured against: the CLEAN TRAINING label spread,
    /// on every split. Equals `clean_label_sd` on the training plan itself.
    dose_reference_sd: f32,

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
///
/// `shared` carries the training split's region-level decisions. When it is given,
/// the affected scaffold groups are LOOKED UP rather than drawn again, so a
/// held-out molecule in an affected group carries that group's multiplier.
///
/// `apply` says whether this split's labels actually receive noise. It only matters
/// for the outlier type, whose selection is per molecule and has no region to look
/// up: a split that receives noise draws its own contamination (a held-out
/// measurement is as likely to be a bad one as a training measurement), and a split
/// that does not receives the multiplier of an uncontaminated molecule rather than a
/// fabricated coin flip.
fn scale_map(
    targeting: &NoiseTargeting,
    n: usize,
    groups: Option<&[u32]>,
    shared: Option<&TargetingState>,
    apply: bool,
    rng: &mut StdRng,
) -> (Vec<f32>, f32, HashSet<u32>) {
    match targeting {
        NoiseTargeting::Uniform => (vec![1.0; n], 1.0, HashSet::new()),

        NoiseTargeting::GroupedShift { .. } => (vec![1.0; n], 1.0, HashSet::new()),

        NoiseTargeting::GroupedWide {
            lambda,
            group_fraction,
        } => {
            let g = groups.expect("grouped noise requires scaffold group assignments");

            // A split that inherits the training split's selection does not draw one.
            //
            // EXCEPT when the inherited selection reaches none of this split's
            // molecules, which under a scaffold split is the ORDINARY case, not an
            // edge case: the splitter holds whole scaffold groups out, so validation
            // and test share no scaffold with training by construction. Looking the
            // training groups up then multiplies every molecule by 1.0 and the shape
            // comes out flat — no molecule is in an affected region, so the condition
            // has no structure on this split at all.
            //
            // A flat shape is not a smaller effect. It is an absent one: the
            // "does the model become less certain where the data is unreliable"
            // question has nothing to correlate against, and validation would carry
            // plain Gaussian noise under a name that says otherwise.
            //
            // So this split draws its OWN selection, under the identical rule and at
            // the same molecule fraction, from its own seed. Decision 3 (2026-08-26)
            // is that validation carries the same kind and amount of noise as
            // training, drawn independently; for a scaffold-keyed condition, drawn
            // independently is the only thing it can mean.
            if let Some(state) = shared {
                let scales: Vec<f32> = g
                    .iter()
                    .map(|gi| if state.affected_groups.contains(gi) { *lambda } else { 1.0 })
                    .collect();
                let hits = scales.iter().filter(|s| **s != 1.0).count();
                // A split that RECEIVES no noise keeps the lookup even when it finds
                // nothing. Its recorded shape is "what this molecule's region would
                // have got", and for a scaffold-keyed condition the honest answer on
                // an unseen scaffold is that the condition never reached it. Drawing
                // a selection here instead would record an injection that did not
                // happen, and question B would be scored against it.
                //
                // A consequence to state rather than paper over: for the grouped
                // conditions under a scaffold split, the held-out shape is FLAT, so
                // "does the model become less certain where the data is unreliable"
                // is undefined on held-out molecules for those conditions. It is
                // answerable on the out-of-fold training rows, where the regions are
                // ones the model was actually exposed to.
                if hits > 0 || !apply {
                    let affected = hits as f32 / n.max(1) as f32;
                    return (scales, affected, state.affected_groups.clone());
                }
                println!(
                    "  grouped_wide: none of this split's {} molecules is in a scaffold \
                     group the training split marked, and this split DOES receive noise, \
                     so it draws its own selection at the same molecule fraction ({:.2}). \
                     Expected under a scaffold split, which holds whole groups out.",
                    n, group_fraction
                );
            }

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
            (scales, affected, bad)
        }

        NoiseTargeting::Outlier { p, lambda } => {
            // No region to inherit: contamination is drawn per measurement. A split
            // that receives noise draws its own; a split that does not gets the
            // uncontaminated multiplier rather than a coin flip nothing acts on.
            if !apply {
                return (vec![1.0; n], 0.0, HashSet::new());
            }
            let scales: Vec<f32> = (0..n)
                .map(|_| if rng.random::<f32>() < *p { *lambda } else { 1.0 })
                .collect();
            let affected =
                scales.iter().filter(|s| **s != 1.0).count() as f32 / n.max(1) as f32;
            (scales, affected, HashSet::new())
        }

        NoiseTargeting::Censoring { .. } => (vec![1.0; n], 0.0, HashSet::new()),
    }
}

/// Tags for the per-split seeds. Fixed constants, so a split's noise is a
/// reproducible function of the run seed and nothing else.
const VALIDATION_SEED_TAG: u64 = 1;
const TEST_SEED_TAG: u64 = 2;

/// A split's seed, derived from the run seed.
///
/// The validation draw has to be INDEPENDENT of the training draw — the same run
/// seed would hand the first validation molecule the first training molecule's
/// number, which is the same class of mistake as indexing one plan by two splits'
/// row positions. It also has to be REPRODUCIBLE from the run seed alone, so that a
/// resubmitted or gap-filling job injects the same noise as the original.
///
/// SplitMix64's finalising mix: every input bit affects every output bit, so
/// neighbouring run seeds (which is what the replicate loop hands out) do not give
/// neighbouring validation seeds.
fn derive_split_seed(base: u64, tag: u64) -> u64 {
    let mut z = base
        .wrapping_add(tag.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
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

/// Build the noise for one split: the per-molecule values, and the provenance that
/// proves what they are.
///
/// `ctx` says which split this is. The training split defines the condition and is
/// its own reference; every other split measures its dose and its cut-points against
/// the clean TRAINING labels and inherits the training split's region-level
/// decisions, so "the same noise level" means the same number of label units on all
/// of them.
fn build_noise_plan(
    labels: &[f32],
    canonical: &[String],
    spec: &NoiseSpec,
    groups: Option<&HashMap<String, u32>>,
    ctx: &PlanContext,
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
            format!(
                "no {} labels were read — cannot build a noise plan",
                ctx.split_name
            ),
        ));
    }
    if ctx.reference_labels.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "the clean training labels are empty — there is nothing to anchor the dose to",
        ));
    }

    let clean_mean = population_mean(labels);
    let clean_sd = population_sd(labels);

    // THE ANCHOR. The dose is a number of label units, and that number is fixed by
    // the clean TRAINING spread on every split. Using this split's own spread is the
    // bug this parameter exists to prevent: validation is a tenth the size and, after
    // a scaffold split, need not share the training column's spread, so the same
    // `--noise-level` would deliver a different amount of noise to each split and the
    // two would no longer be comparable.
    let reference_sd = population_sd(ctx.reference_labels);

    // The dose delivered at a reference level of 1.0. `noise_pattern_raw` is measured
    // here and `noise_scale_raw` is `level * noise_pattern_raw` exactly, so the shape
    // column carries no dependence on the level whatsoever.
    let reference_dose = match spec.units {
        DoseUnits::Spread => reference_sd,
        DoseUnits::Label => 1.0,
    };

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
    params.insert("split".to_string(), serde_json::json!(ctx.split_name));
    params.insert("noise_applied".to_string(), serde_json::json!(ctx.apply));

    // The condition's name, exactly as it appears in `noiseInject.CONDITIONS` and in
    // `roster()` in `rust/reference/noise_arms.rs`. One name per condition — a job
    // script, a results row and a figure label have to agree, and the paper has
    // already had one quantity carrying two names on facing pages.
    let noise_type = condition_name(spec);

    // The random stream is opened at EVERY level, including zero.
    //
    // The scale map is its first consumer, so opening it unconditionally is what
    // makes `noise_pattern_raw` bit-identical from level zero upwards: the same seed
    // draws the same scale map whatever the level, and the pattern is a function of
    // the scale map alone. Returning early at level zero — the old shape of this
    // function — would leave the negative control with no shape column at all, and
    // the zero-level subtraction is the only thing that removes the label-magnitude
    // confound.
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
        // The assay limit is a property of the ASSAY, so it is read off the training
        // distribution and applied unchanged to every split. Reading it off the
        // validation labels' own quantile — which is what happens if this uses
        // `labels` — censors a fixed fraction of each split at a different value, and
        // that is a bug the experimental pipeline already had and fixed.
        let mut sorted = ctx.reference_labels.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let cut = match side {
            CensorSide::Upper => quantile(&sorted, 1.0 - fraction),
            CensorSide::Lower => quantile(&sorted, fraction),
        };
        // The level-free shape. Censoring is not on a dose axis, so "the same shape at
        // level 1.0" means the limit pushed all the way to the end of the training
        // range: how far a molecule sits past the far end of the training
        // distribution, which is exactly what decides whether and how hard censoring
        // hits it. It uses a fixed quantile, so it does not move with the level.
        let reference_cut = match side {
            CensorSide::Upper => quantile(&sorted, 0.0),
            CensorSide::Lower => quantile(&sorted, 1.0),
        };
        let clip = |y: f32, limit: f32| -> f32 {
            match side {
                CensorSide::Upper => {
                    if y > limit {
                        limit - y
                    } else {
                        0.0
                    }
                }
                CensorSide::Lower => {
                    if y < limit {
                        limit - y
                    } else {
                        0.0
                    }
                }
            }
        };
        let noise_pattern: Vec<f32> = labels.iter().map(|y| clip(*y, reference_cut)).collect();
        let epsilon: Vec<f32> = if ctx.apply && fraction > 0.0 {
            labels.iter().map(|y| clip(*y, cut)).collect()
        } else {
            vec![0.0; n]
        };
        // For censoring the amount applied is the shift itself — there is no scale
        // parameter behind it.
        let noise_scale: Vec<f32> = epsilon.iter().map(|e| e.abs()).collect();
        let affected = epsilon.iter().filter(|e| **e != 0.0).count() as f32 / n as f32;
        let realised = rms(&epsilon);
        let mean_eps = population_mean(&epsilon);
        params.insert("censor_limit".to_string(), serde_json::json!(cut));
        params.insert(
            "censor_reference_limit".to_string(),
            serde_json::json!(reference_cut),
        );
        params.insert(
            "requested_censored_fraction".to_string(),
            serde_json::json!(fraction),
        );
        return Ok(NoisePlan {
            epsilon,
            canonical: canonical.to_vec(),
            noise_scale,
            noise_pattern,
            targeting_state: TargetingState::default(),
            dose_reference_sd: reference_sd,
            noise_type,
            shape_name: spec.shape.name(),
            targeting_name: spec.targeting.name(),
            unit_dose_g: f32::NAN, // censoring does not go through the dose solver
            solved_scale: f32::NAN,
            target_dose_label_units: f32::NAN,
            realised_dose_label_units: realised,
            realised_dose_fraction_of_spread: realised / reference_sd,
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
    let target = spec.level * reference_dose;

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
                    "{:.2}% of {} molecules are missing from the scaffold group file — \
                     it does not match this split. Refusing to run.",
                    miss_rate * 100.0,
                    ctx.split_name
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

    let (scales, mut affected, affected_groups) = scale_map(
        &spec.targeting,
        n,
        group_ids.as_deref(),
        ctx.shared,
        ctx.apply,
        &mut rng,
    );
    let g = unit_dose(&spec.shape, &scales);
    let solved = target / g;

    // The per-molecule amount, before the level is applied.
    //
    //   sd(eps_i) = solved * s_i * unit_sd = (target / (rms(s) * unit_sd)) * s_i * unit_sd
    //             = target * s_i / rms(s)
    //             = level * reference_dose * s_i / rms(s)
    //
    // so the shape divides out of the level exactly, whatever the draw's shape is,
    // and `noise_scale_raw == level * noise_pattern_raw` holds to the bit.
    //
    // The shifted grouped condition has no scale map — every molecule receives the
    // same amount, and what the condition changes is how the noise is CORRELATED, not
    // where it is concentrated — so its shape is flat at `reference_dose`, which is
    // what it is meant to deliver. Note that the delivered amount equals that only
    // for a shape whose unit standard deviation is 1, which is every shifted grouped
    // condition in the roster (they are all Gaussian). Paired with a heavy-tailed
    // shape the existing solver delivers `target / unit_sd` instead — the delivered
    // dose gate above catches it and refuses the run, so it cannot pass quietly, but
    // the combination does not work today. Flagged, not fixed here: the solver
    // belongs to the dose-matching work, not to this change.
    let scale_rms = {
        let ms: f64 = scales.iter().map(|s| (*s as f64) * (*s as f64)).sum::<f64>()
            / scales.len().max(1) as f64;
        ms.sqrt() as f32
    };
    let noise_pattern: Vec<f32> = scales
        .iter()
        .map(|s| reference_dose * s / scale_rms)
        .collect();
    let noise_scale: Vec<f32> = if ctx.apply {
        noise_pattern.iter().map(|p| spec.level * p).collect()
    } else {
        vec![0.0; n]
    };

    let mut group_offsets: HashMap<u32, f32> = HashMap::new();
    let mut inherited_offsets = 0usize;
    let epsilon: Vec<f32> = match spec.targeting {
        // Two components: a group-level offset carrying `rho` of the variance and a
        // within-molecule term carrying the rest, so the two sum to the target.
        //   eps_i = solved * ( sqrt(rho) * z_g(i) + sqrt(1-rho) * w_i )
        // z_g and w_i are draws at the SHAPE'S OWN SPREAD, exactly as every other
        // targeting draws them, and `solved` is what the dose solver returned — so
        // the two variances sum to the target the way they do everywhere else.
        //
        // They used to be rescaled to unit variance here while the solver still put
        // the shape's spread into G, so this type alone delivered the target divided
        // by that spread: 0.71x under Laplace and 0.77x under Student-t at nu=5.
        // Gaussian has unit spread, and the condition roster is Gaussian at every
        // entry, which is why no gate ever saw it (RERUN_PLAN.md §2.14).
        NoiseTargeting::GroupedShift {
            group_variance_share,
        } => {
            let rho = group_variance_share;
            let ids = group_ids.as_ref().expect("grouped shift requires groups");
            // Both components are unit draws FROM THE SHAPE (NOISE_DESIGN.md §2a),
            // so a heavy-tailed shifted condition has heavy tails in the group
            // offsets as well as in the within-molecule term.
            //
            // The offset is a property of the GROUP — a laboratory reading high reads
            // high for everything it measured — so a split that inherits the training
            // split's state reuses its offsets rather than drawing new ones. A group
            // that appears only outside training has no offset to inherit and draws
            // its own; the count is recorded.
            for gid in ids.iter() {
                if group_offsets.contains_key(gid) {
                    continue;
                }
                if let Some(state) = ctx.shared {
                    if let Some(b) = state.group_offsets.get(gid) {
                        group_offsets.insert(*gid, *b);
                        inherited_offsets += 1;
                        continue;
                    }
                }
                let b = spec.shape.draw(&mut rng);
                group_offsets.insert(*gid, b);
            }
            params.insert(
                "group_variance_share".to_string(),
                serde_json::json!(rho),
            );
            params.insert(
                "group_offsets_inherited_from_training".to_string(),
                serde_json::json!(inherited_offsets),
            );
            ids.iter()
                .map(|gid| {
                    let b = group_offsets[gid];
                    let w = spec.shape.draw(&mut rng);
                    solved * (rho.sqrt() * b + (1.0 - rho).sqrt() * w)
                })
                .collect()
        }
        _ => scales
            .iter()
            .map(|s| spec.shape.draw(&mut rng) * solved * s)
            .collect(),
    };

    // Exactly zero, not something small, and not a signed zero from multiplying a
    // draw by a zero scale.
    //
    // Failure mode 2: the old pipeline reconstructed the injected noise by fitting a
    // line, so at zero noise the "noise" was floating-point rounding whose size grew
    // with the label, and the negative control showed a stronger signal than the real
    // levels. Recorded ground truth cannot do that — as long as the record really is
    // zero.
    let epsilon: Vec<f32> = if ctx.apply && spec.level > 0.0 {
        epsilon
    } else {
        vec![0.0; n]
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
            // NOT the group count. The group-level term is averaged over MOLECULES,
            // so a few large scaffold groups dominate it: the effective number of
            // independent group contributions is (Σ n_g)² / Σ n_g². On the real QM9
            // scaffold assignment that is 189 against a group count of 30,313 — a
            // factor of 160.
            //
            // Using the raw count overstates the precision of the delivered dose by
            // that factor, which does not put a wrong number in a results row: it
            // makes `dose_tolerance` demand 0.79% where the true sampling spread is
            // 9.6%, so the flat-dose gate fails runs that were never defective.
            // Found by chat B's cross-check (RERUN_PLAN.md §2.3a); the same formula
            // is in rust/reference/noise_arms.rs and in noiseInject.
            let rho = group_variance_share as f64;
            let ids = group_ids.as_ref().expect("grouped shift requires groups");
            let mut sizes: HashMap<u32, f64> = HashMap::new();
            for gi in ids.iter() {
                *sizes.entry(*gi).or_insert(0.0) += 1.0;
            }
            let total: f64 = sizes.values().sum();
            let sum_sq: f64 = sizes.values().map(|c| c * c).sum();
            let eff_groups = if sum_sq > 0.0 { total * total / sum_sq } else { 1.0 };
            (1.0 / (rho * rho / eff_groups + (1.0 - rho) * (1.0 - rho) / n as f64)) as f32
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
        noise_scale,
        noise_pattern,
        targeting_state: TargetingState {
            affected_groups,
            group_offsets,
        },
        dose_reference_sd: reference_sd,
        noise_type,
        shape_name: spec.shape.name(),
        targeting_name: spec.targeting.name(),
        unit_dose_g: g,
        solved_scale: solved,
        target_dose_label_units: target,
        realised_dose_label_units: realised,
        realised_dose_fraction_of_spread: realised / reference_sd,
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
///
/// `ctx.split_name` names the split in every message, and `ctx.apply` says whether
/// this split's labels were meant to receive noise at all. The dose gates apply only
/// to a split that receives noise; the zero and shape gates apply to all of them.
fn assert_noise_plan_gates(
    plan: &NoisePlan,
    spec: &NoiseSpec,
    labels: &[f32],
    ctx: &PlanContext,
) {
    let split = ctx.split_name;
    assert_eq!(
        plan.epsilon.len(),
        labels.len(),
        "gate ({split}): one recorded noise value per molecule"
    );
    assert_eq!(
        plan.noise_scale.len(),
        labels.len(),
        "gate ({split}): one recorded noise scale per molecule"
    );
    assert_eq!(
        plan.noise_pattern.len(),
        labels.len(),
        "gate ({split}): one recorded noise shape per molecule"
    );
    assert!(
        plan.noise_pattern.iter().all(|p| p.is_finite()),
        "gate ({split}): every recorded noise shape must be finite"
    );

    // The shape column is what the zero-level subtraction is built on, so it has to
    // carry the condition's structure at EVERY level — including zero, where there is
    // no noise to describe. A targeted condition whose shape is flat has lost that
    // structure and the negative control silently becomes uninformative.
    let shape_varies = plan
        .noise_pattern
        .iter()
        .any(|p| (*p - plan.noise_pattern[0]).abs() > 1e-12);
    match spec.targeting {
        NoiseTargeting::Censoring { .. } => {
            // Censoring is keyed to the LABEL, so its shape is defined on any split,
            // held-out included: the cut-points come from the training distribution
            // and a held-out label either clears them or does not.
            assert!(
                shape_varies,
                "gate ({split}): {} is keyed to the label, so its level-free shape must \
                 vary between molecules at every level — it is flat",
                plan.noise_type
            );
        }
        NoiseTargeting::GroupedWide { .. } => {
            // Keyed to the SCAFFOLD GROUP, and the splitter holds whole groups out.
            // So a held-out molecule is in a group the selection never saw, and its
            // level-free shape is flat — truthfully, because the condition never
            // reached that region. Asserting otherwise would demand a structure that
            // cannot exist, and drawing one would record an injection that did not
            // happen.
            //
            // Consequence, and it belongs in the Methods rather than in a comment
            // alone: for the grouped conditions under a scaffold split, "does the
            // model become less certain where the data is unreliable" is answerable
            // on the out-of-fold TRAINING rows and undefined on held-out molecules.
            if ctx.apply {
                assert!(
                    shape_varies,
                    "gate ({split}): {} is keyed to the scaffold group and this split \
                     RECEIVES noise, so its level-free shape must vary between \
                     molecules — it is flat",
                    plan.noise_type
                );
            }
        }
        NoiseTargeting::Outlier { .. } => {
            // Only the splits that actually receive contamination have a per-molecule
            // outlier draw; a clean split carries the uncontaminated multiplier,
            // which is flat on purpose.
            if ctx.apply {
                assert!(
                    shape_varies,
                    "gate ({split}): {} contaminates a random fraction, so its level-free \
                     shape must vary between molecules — it is flat",
                    plan.noise_type
                );
            }
        }
        NoiseTargeting::Uniform | NoiseTargeting::GroupedShift { .. } => {
            // These two hit every molecule equally hard, so a shape that varies means
            // a scale map crept in where there should be none.
            assert!(
                !shape_varies,
                "gate ({split}): {} delivers the same amount to every molecule, so its \
                 level-free shape must be flat — it varies",
                plan.noise_type
            );
        }
    }

    // Gate 5 — zero noise records EXACTLY zero, not something small. On EVERY split,
    // and for the applied amount as well as the draw: a `noise_scale_raw` that is
    // merely small at level zero would make the negative control look like a weak
    // dose rather than none.
    if spec.level <= 0.0 || !ctx.apply {
        let why = if spec.level <= 0.0 {
            "at level 0"
        } else {
            "on a split that receives no noise"
        };
        assert!(
            plan.epsilon.iter().all(|e| *e == 0.0),
            "gate ({split}): {why} every recorded noise value must be exactly zero"
        );
        assert!(
            plan.noise_scale.iter().all(|s| *s == 0.0),
            "gate ({split}): {why} every recorded noise scale must be exactly zero"
        );
        return;
    }

    assert!(
        plan.epsilon.iter().all(|e| e.is_finite()),
        "gate ({split}): every recorded noise value must be finite"
    );

    // `noise_scale_raw` is the level times `noise_pattern_raw`, exactly. Downstream
    // treats the two as interchangeable on any ranking statistic, and that is only
    // true if this identity holds.
    if spec.targeting.is_dose_matched() {
        for (i, (s, p)) in plan.noise_scale.iter().zip(plan.noise_pattern.iter()).enumerate() {
            let expected = spec.level * p;
            assert!(
                (s - expected).abs() <= expected.abs() * 1e-6,
                "gate ({split}): molecule {} records a scale of {} but the level times its \
                 shape is {}",
                i,
                s,
                expected
            );
        }
    }

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
            "gate ({split}): {} constructs a dose of {:.9} against a target of {:.9} — the \
             solver is wrong",
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
            "gate ({split}): {} delivered {:.6} against a target of {:.6} ({:+.2}%), outside \
             the {:.2}% band that {:.0} effective observations allow",
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
    //
    // This is a statement about the CONDITION, which is defined on the training
    // split. A held-out split of a few hundred molecules can legitimately contain
    // none past an assay limit set on the training column, and that is a fact about
    // the sample rather than a degenerate condition.
    if ctx.shared.is_none() {
        assert!(
            plan.affected_molecule_fraction > 0.0,
            "gate ({split}): {} affected no molecules at level {} — the condition is degenerate",
            plan.noise_type,
            spec.level
        );
    }
}

/// B3's cross-split gate: the validation split must receive the SAME AMOUNT of noise
/// as the training split, in absolute label units.
///
/// This is the check that fails if the validation plan is ever anchored on the
/// validation labels' own spread. Validation is a tenth the size and, after a
/// scaffold split, need not share the training column's spread at all — so the
/// amounts would silently differ while both splits still called it "the same level".
///
/// The band is the two splits' own dose tolerances combined in quadrature, because
/// each realised amount is a sample root-mean-square with its own sampling spread and
/// the smaller split's dominates. It is not a fixed percentage: at a few hundred
/// validation molecules a flat band would fail correct code.
fn assert_validation_matches_training(
    train: &NoisePlan,
    validation: &NoisePlan,
    spec: &NoiseSpec,
) {
    if spec.level <= 0.0 || !spec.targeting.is_dose_matched() {
        return;
    }
    let t_train = dose_tolerance(&spec.shape, &train.epsilon, train.effective_n);
    let t_val = dose_tolerance(&spec.shape, &validation.epsilon, validation.effective_n);
    let band = (t_train * t_train + t_val * t_val).sqrt();
    let ratio = validation.realised_dose_label_units / train.realised_dose_label_units;
    assert!(
        (ratio - 1.0).abs() <= band,
        "gate: validation received {:.6} label units of {} noise against training's {:.6} \
         ({:+.2}%), outside the {:.2}% band the two splits' sizes allow. The two splits must \
         receive the same AMOUNT — check the validation dose is anchored on the clean \
         TRAINING spread ({:.6}) and not on validation's own ({:.6}).",
        validation.realised_dose_label_units,
        train.noise_type,
        train.realised_dose_label_units,
        (ratio - 1.0) * 100.0,
        band * 100.0,
        train.clean_label_sd,
        validation.clean_label_sd
    );
}

/// Run the cross-type gate that no single training run can run: at ONE target, on
/// the real clean labels, every dose-matched noise type must deliver the same
/// amount. This is the single check that proves the confound is gone
/// (RERUN_PLAN.md §8 gate 1), plus gates 5 and 7.
///
/// Returns the number of failures.
/// Median absolute noise — a first-moment shape diagnostic, so it separates
/// conditions that deliver the same second moment.
fn median_abs(v: &[f32]) -> f64 {
    let mut a: Vec<f64> = v.iter().map(|x| x.abs() as f64).collect();
    a.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let n = a.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        a[n / 2]
    } else {
        0.5 * (a[n / 2 - 1] + a[n / 2])
    }
}

/// The worst-hit 5%'s share of the total noise energy: how concentrated the damage
/// is, at a fixed total amount. This is what actually differs between the conditions
/// once the dose is matched.
fn top5_energy_share(v: &[f32]) -> f64 {
    let mut sq: Vec<f64> = v.iter().map(|x| (*x as f64) * (*x as f64)).collect();
    sq.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let total: f64 = sq.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    let k = ((sq.len() as f64) * 0.05).round() as usize;
    sq[..k.max(1)].iter().sum::<f64>() / total
}

/// The full condition roster, named exactly as `roster()` in
/// `rust/reference/noise_arms.rs` and as `noiseInject.CONDITIONS`.
fn condition_roster(with_groups: bool) -> Vec<(NoiseShape, NoiseTargeting, f32)> {
    let mut v: Vec<(NoiseShape, NoiseTargeting, f32)> = vec![
        (NoiseShape::Gaussian, NoiseTargeting::Uniform, f32::NAN),
        (NoiseShape::StudentT { nu: 10.0 }, NoiseTargeting::Uniform, f32::NAN),
        (NoiseShape::StudentT { nu: 5.0 }, NoiseTargeting::Uniform, f32::NAN),
        (NoiseShape::StudentT { nu: 3.0 }, NoiseTargeting::Uniform, f32::NAN),
        (NoiseShape::Laplace, NoiseTargeting::Uniform, f32::NAN),
    ];
    if with_groups {
        v.push((
            NoiseShape::Gaussian,
            NoiseTargeting::GroupedWide { lambda: 3.0, group_fraction: 0.2 },
            f32::NAN,
        ));
        v.push((
            NoiseShape::Gaussian,
            NoiseTargeting::GroupedShift { group_variance_share: 0.62 },
            f32::NAN,
        ));
    }
    for p in [0.01f32, 0.05, 0.10] {
        v.push((
            NoiseShape::Gaussian,
            NoiseTargeting::Outlier { p, lambda: 3.0 },
            f32::NAN,
        ));
    }
    // The grid of NOISE_DESIGN.md §6.4, zero control included. Censoring at 0% is the
    // negative control for the one condition that is not zero-mean, and leaving it out
    // is how the roster silently disagreed with the reference.
    for pct in [0u32, 10, 20, 25, 30, 40, 50] {
        v.push((
            NoiseShape::Gaussian,
            NoiseTargeting::Censoring { side: CensorSide::Upper },
            pct as f32 / 100.0,
        ));
    }
    v
}

/// The self-test's statistics, in the same JSON shape `rust/reference/noise_arms.rs`
/// emits, so the pipeline can be compared against the reference row by row.
///
/// Chat B's `scripts/crosscheck_injectors.py` ties the reference to the Python
/// injector. This ties the reference to the PIPELINE — without it the pipeline could
/// drift from both and nothing would notice, which is the exact failure the gate
/// exists to prevent.
/// `only` runs ONE named shape-and-targeting pair instead of the roster. The roster
/// is Gaussian at every entry, which is right -- it is what the study runs -- but it
/// leaves the pipeline's grouped-shift under a heavy tail unreachable, so nothing
/// could compare it against the Python injector. This is the way in, and nothing but
/// a test uses it: both `--noise-shape` and `--noise-targeting` must be given, and
/// no caller of the roster path passes either.
fn self_test_json(
    labels: &[f32],
    groups: Option<&HashMap<String, u32>>,
    canonical: &[String],
    k: f32,
    seeds: u64,
    only: Option<(NoiseShape, NoiseTargeting)>,
) {
    let clean_sd = population_sd(labels);
    let target = k * clean_sd;
    let mut rows: Vec<String> = Vec::new();

    let roster = match only {
        Some((shape, targeting)) => vec![(shape, targeting, f32::NAN)],
        None => condition_roster(groups.is_some()),
    };
    for (shape, targeting, censor_level) in roster {
        let level = if censor_level.is_nan() { k } else { censor_level };
        let mut delivered: Vec<f32> = Vec::new();
        let mut last: Option<NoisePlan> = None;
        let mut spec_used = None;
        for i in 0..seeds.max(1) {
            let spec = NoiseSpec {
                shape,
                targeting,
                level,
                units: DoseUnits::Spread,
                seed: 1000 + i * 7919,
            };
            let plan = match build_noise_plan(labels, canonical, &spec, groups, &PlanContext::training(labels)) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("{}", e);
                    std::process::exit(1);
                }
            };
            delivered.push(plan.realised_dose_label_units);
            spec_used = Some(spec.clone());
            last = Some(plan);
        }
        let plan = last.unwrap();
        let spec = spec_used.unwrap();
        let dose_matched = targeting.is_dose_matched();
        let mean_d = population_mean(&delivered) as f64;
        let sd_d = population_sd(&delivered) as f64;
        let thr = if dose_matched {
            3.0 * target as f64
        } else {
            3.0 * plan.realised_dose_label_units as f64
        };
        let beyond = plan.epsilon.iter().filter(|e| (e.abs() as f64) > thr).count() as f64
            / plan.epsilon.len() as f64;
        let censor_limit = plan.params.get("censor_limit").and_then(|v| v.as_f64());
        rows.push(format!(
            r#"{{"condition":"{}","k":{},"dose_matched":{},"target_dose":{},"unit_dose":{},"solved_scale":{},"censoring_limit":{},"delivered_dose":{},"delivered_dose_sd":{},"mean_shift":{},"frac_beyond_3":{},"median_abs":{},"top5_energy_share":{},"affected_molecule_fraction":{},"effective_n":{},"dose_tolerance":{},"seeds":{}}}"#,
            plan.noise_type,
            k,
            dose_matched,
            if dose_matched { format!("{}", target) } else { "null".to_string() },
            if plan.unit_dose_g.is_nan() { "null".to_string() } else { format!("{}", plan.unit_dose_g) },
            if plan.solved_scale.is_nan() { "null".to_string() } else { format!("{}", plan.solved_scale) },
            censor_limit.map(|v| format!("{}", v)).unwrap_or_else(|| "null".to_string()),
            mean_d,
            sd_d,
            plan.mean_epsilon,
            beyond,
            median_abs(&plan.epsilon),
            top5_energy_share(&plan.epsilon),
            plan.affected_molecule_fraction,
            plan.effective_n,
            dose_tolerance(&spec.shape, &plan.epsilon, plan.effective_n),
            seeds.max(1),
        ));
    }
    println!(
        r#"{{"n":{},"label_sd":{},"rows":[{}]}}"#,
        labels.len(),
        clean_sd,
        rows.join(",")
    );
}

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
            match build_noise_plan(labels, canonical, &spec, groups, &PlanContext::training(labels)) {
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
    // 200 seeds, not 20. Rule 3 of NOISE_DESIGN.md §2a gates on the MEAN delivered
    // dose, so the gate's power is set by the standard error of that mean — and two
    // conditions have per-run dose spreads far too wide for twenty draws to pin it
    // down. Measured on 3,200 real QM9 training labels with real scaffold groups
    // (chat G, 2026-08-26): per-run SD is 1.3% for Gaussian but 3.9% for
    // grouped-shifted and 6.9% for Student-t ν=3, so over twenty seeds their means
    // wander by ±0.9% and ±1.5% respectively and the 3% spread criterion below is
    // breached by sampling noise alone — this gate failed at 3.39% on labels where
    // 400 seeds put grouped-shifted at +0.03% ± 0.19%, i.e. exactly on target. A
    // launch gate that fails at random is worse than no gate, because the next
    // person turns it off. Two hundred seeds put every condition's standard error
    // under 0.5%.
    const GATE_SEEDS: u64 = 200;
    println!(
        "gate: the mean delivered dose over {} seeds is flat across noise types",
        GATE_SEEDS
    );
    let seeds: Vec<u64> = (0..GATE_SEEDS).map(|i| 1000 + i * 7919).collect();
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
            let plan = build_noise_plan(labels, canonical, &spec, groups, &PlanContext::training(labels)).unwrap();
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
        let plan = build_noise_plan(labels, canonical, &spec, groups, &PlanContext::training(labels)).unwrap();
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
    //
    // Averaged over 50 seeds, not compared on one. Two independent draws of a few
    // thousand labels each carry a per-run dose spread of about 1.3%, so the ratio of
    // one against the other wobbles by roughly 1.8% and a 2% threshold on a single
    // seed fails by chance about a quarter of the time — measured on real QM9 labels
    // (chat G, 2026-08-26), where this gate failed at +2.21% while the nesting itself
    // is exact. The claim being tested is about the distributions, so the statistic
    // has to be one too.
    const NESTING_SEEDS: u64 = 50;
    print!(
        "gate: Student-t at nu=200 matches Gaussian, averaged over {} seeds ... ",
        NESTING_SEEDS
    );
    let mut dose_ratios: Vec<f32> = Vec::new();
    let mut tail_gaps: Vec<f32> = Vec::new();
    let mut tail_g_mean = 0.0f32;
    let mut tail_t_mean = 0.0f32;
    for seed in 0..NESTING_SEEDS {
        let mk = |shape: NoiseShape| {
            build_noise_plan(
                labels,
                canonical,
                &NoiseSpec {
                    shape,
                    targeting: NoiseTargeting::Uniform,
                    level: 0.5,
                    units: DoseUnits::Spread,
                    seed: 7 + seed * 104_729,
                },
                groups,
                &PlanContext::training(labels),
            )
            .unwrap()
        };
        let gauss = mk(NoiseShape::Gaussian);
        let t200 = mk(NoiseShape::StudentT { nu: 200.0 });
        let tail = |p: &NoisePlan| {
            p.epsilon
                .iter()
                .filter(|e| e.abs() > 3.0 * p.target_dose_label_units)
                .count() as f32
                / p.epsilon.len() as f32
        };
        let tg = tail(&gauss);
        let tt = tail(&t200);
        tail_g_mean += tg / NESTING_SEEDS as f32;
        tail_t_mean += tt / NESTING_SEEDS as f32;
        tail_gaps.push(tt - tg);
        dose_ratios
            .push(t200.realised_dose_label_units / gauss.realised_dose_label_units - 1.0);
    }
    let dose_gap = population_mean(&dose_ratios).abs();
    let tail_gap = population_mean(&tail_gaps).abs();
    if dose_gap <= 0.02 && tail_gap <= 0.005 {
        println!(
            "ok (mean dose gap {:+.2}%, mean tail beyond 3x: gaussian {:.2}% vs t {:.2}%)",
            dose_gap * 100.0,
            tail_g_mean * 100.0,
            tail_t_mean * 100.0
        );
    } else {
        println!(
            "FAIL (mean dose gap {:+.2}%, mean tail beyond 3x: gaussian {:.2}% vs t {:.2}%)",
            dose_gap * 100.0,
            tail_g_mean * 100.0,
            tail_t_mean * 100.0
        );
        failures += 1;
    }

    // Gate 5b — the level-free shape is bit-identical at EVERY level, zero included.
    //
    // This column is the only negative control QM9 has. The zero-level model saw the
    // same labels and no corruption, so its correlation with the shape is exactly the
    // label-magnitude confound and can be subtracted off — but only if the shape it is
    // correlated against is the SAME column at every level. A shape that moves with
    // the level subtracts a different thing at each one and the control is worthless.
    print!("gate: the level-free noise shape is identical at every level, zero included ... ");
    let mut shape_ok = true;
    let mut shape_varies_somewhere = false;
    for (shape, targeting) in &types {
        let mut baseline: Option<Vec<u32>> = None;
        for level in [0.0f32, 0.25, 0.5, 1.0] {
            let spec = NoiseSpec {
                shape: *shape,
                targeting: *targeting,
                level,
                units: DoseUnits::Spread,
                seed: 42,
            };
            let plan =
                build_noise_plan(labels, canonical, &spec, groups, &PlanContext::training(labels))
                    .unwrap();
            let bits: Vec<u32> = plan.noise_pattern.iter().map(|p| p.to_bits()).collect();
            if plan.noise_pattern.iter().any(|p| *p != plan.noise_pattern[0]) {
                shape_varies_somewhere = true;
            }
            // and at level zero nothing is applied, to the bit
            if level == 0.0 {
                if !plan.epsilon.iter().all(|e| *e == 0.0)
                    || !plan.noise_scale.iter().all(|v| *v == 0.0)
                {
                    shape_ok = false;
                }
            }
            match &baseline {
                None => baseline = Some(bits),
                Some(b) => {
                    if *b != bits {
                        shape_ok = false;
                    }
                }
            }
        }
    }
    if shape_ok && shape_varies_somewhere {
        println!("ok");
    } else {
        println!(
            "FAIL (identical across levels: {}, varies between molecules somewhere: {})",
            shape_ok, shape_varies_somewhere
        );
        failures += 1;
    }

    // Gate 11 — the validation split receives the SAME AMOUNT as training, in absolute
    // label units, even when its own spread is nothing like training's.
    //
    // Validation is a tenth the size and split by scaffold, so its spread is not the
    // training spread. Dosing it against its own spread — the obvious way to write it —
    // would deliver a different number of label units to each split while both still
    // called it "level 0.5". The column below is rigged: the validation labels' spread
    // is three times training's, so an unanchored dose would be out by a factor of three
    // and could not be mistaken for sampling noise.
    //
    // Averaged over seeds, not compared on one. This is the third launch gate to have
    // needed that fix and the only one that exits 1, so it is the one that stops the
    // run (chat G, 2026-08-27). On a 4,000-molecule column Student-t ν=3 delivered
    // +36.54% against a 21.21% band and failed on every attempt, while the same
    // construction averaged over 100 seeds is flat. The validation part is a fifth of
    // an already small column, so its per-draw dose spread is the largest anywhere in
    // the self-test, and the ratio of two single heavy-tailed draws is not a statistic
    // about the anchoring rule at all. It passed on the full 133,885-label column only
    // because that split is large enough to hide the problem — which is the worst way
    // for a gate to be right, since it makes the gate's verdict depend on how much data
    // it happens to be handed. The defect being tested is a factor of three; a mean
    // over 100 seeds catches that with several orders of magnitude to spare.
    const VALIDATION_SEEDS: u64 = 100;
    println!(
        "\ngate: validation is dosed against the CLEAN TRAINING spread, not its own \
         (mean over {} seeds)",
        VALIDATION_SEEDS
    );
    let cut = (labels.len() * 4) / 5;
    let train_part: Vec<f32> = labels[..cut].to_vec();
    let train_names: Vec<String> = canonical[..cut].to_vec();
    let tail_mean = population_mean(&labels[cut..]);
    let val_part: Vec<f32> = labels[cut..]
        .iter()
        .map(|y| tail_mean + 3.0 * (y - tail_mean))
        .collect();
    let val_names: Vec<String> = canonical[cut..].to_vec();
    let train_sd_here = population_sd(&train_part);
    let val_sd_here = population_sd(&val_part);
    println!(
        "  training spread {:.6}, validation spread {:.6} ({:.2}x)",
        train_sd_here,
        val_sd_here,
        val_sd_here / train_sd_here
    );
    let val_seeds: Vec<u64> = (0..VALIDATION_SEEDS).map(|i| 2000 + i * 6151).collect();
    for (shape, targeting) in &types {
        if !targeting.is_dose_matched() {
            continue;
        }
        let mut train_doses: Vec<f32> = Vec::new();
        let mut val_doses: Vec<f32> = Vec::new();
        let mut label = String::new();
        let mut errored = false;
        for seed in &val_seeds {
            let spec = NoiseSpec {
                shape: *shape,
                targeting: *targeting,
                level: 0.5,
                units: DoseUnits::Spread,
                seed: *seed,
            };
            let t_ctx = PlanContext::training(&train_part);
            let t_plan =
                match build_noise_plan(&train_part, &train_names, &spec, groups, &t_ctx) {
                    Ok(p) => p,
                    Err(e) => {
                        println!("  {:<34} FAIL: {}", targeting.name(), e);
                        failures += 1;
                        errored = true;
                        break;
                    }
                };
            let v_spec = NoiseSpec {
                seed: derive_split_seed(spec.seed, VALIDATION_SEED_TAG),
                ..spec.clone()
            };
            let v_ctx = PlanContext {
                reference_labels: &train_part,
                shared: Some(&t_plan.targeting_state),
                apply: true,
                split_name: "val",
            };
            let v_plan = match build_noise_plan(&val_part, &val_names, &v_spec, groups, &v_ctx) {
                Ok(p) => p,
                Err(e) => {
                    println!("  {:<34} FAIL: {}", targeting.name(), e);
                    failures += 1;
                    errored = true;
                    break;
                }
            };
            label = t_plan.noise_type.clone();
            train_doses.push(t_plan.realised_dose_label_units);
            val_doses.push(v_plan.realised_dose_label_units);
        }
        if errored {
            continue;
        }
        // The ratio of the two MEANS, not the mean of the per-seed ratios.
        //
        // Both were tried. The mean of ratios is wrong here for two reasons and the
        // second is the one that matters. First, it does not reproduce the two numbers
        // printed beside it — on the 4,000-molecule column it read +0.39% for Student-t
        // ν = 3 while the train and val means printed next to it give −0.99%, the
        // opposite sign, which is exactly the contradiction guard 4 exists to stop.
        // Second, the two draws are independent, so E[V/T] ≈ (E[V]/E[T])(1 + CV_T²) and
        // ν = 3's per-draw spread of ~15% biases the statistic upward by about 2% — a
        // bias that grows with the tail weight of the very shape the gate is weakest on.
        // The ratio of means has neither problem and is what the printed line shows.
        let m_train = population_mean(&train_doses);
        let m_val = population_mean(&val_doses);
        let n = train_doses.len() as f32;
        let se_train = population_sd(&train_doses) / n.sqrt();
        let se_val = population_sd(&val_doses) / n.sqrt();
        let err = m_val / m_train - 1.0;
        // standard error of the ratio, to first order, from both sides' own spread
        let se_ratio = ((se_val * se_val + (m_val / m_train).powi(2) * se_train * se_train)
            .sqrt())
            / m_train;
        // three of them, floored so a tiny spread cannot make the gate unreasonably
        // strict — the same rule the flat-dose gate above uses
        let band = (3.0 * se_ratio).max(0.005);
        let ok = err.abs() <= band;
        if !ok {
            failures += 1;
        }
        // guard 4: the ratio never appears without the two doses it is a ratio of, and
        // those two doses now reproduce it exactly
        let unanchored = 0.5 * val_sd_here;
        println!(
            "  {:<34} train={:.6} val={:.6} ({:+.2}%, per-run SD {:.2}%/{:.2}%, band {:.2}%)  \
             [val's own spread would give {:.6}]  {}",
            label,
            m_train,
            m_val,
            err * 100.0,
            100.0 * population_sd(&train_doses) / m_train,
            100.0 * population_sd(&val_doses) / m_val,
            band * 100.0,
            unanchored,
            if ok { "ok" } else { "FAIL" }
        );
    }
    println!();

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
            &PlanContext::training(labels),
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

    // Read isomeric_smiles and check validity.
    //
    // A malformed record used to `return None` HERE, in the middle of a record.
    // Every caller reads `if let Some(data) = read_smiles_data(...)` with no
    // else, so a None is indistinguishable from end-of-data -- but the rest of
    // THIS record was still unread in the buffer, so the next call started
    // mid-record and every molecule after it decoded from the wrong offset. That
    // is the one failure this file cannot survive (RERUN_PLAN.md 2.7, 2.13).
    let isomeric_smiles = read_len_prefixed_string(reader)?;
    if isomeric_smiles.len() < 5 || isomeric_smiles.len() > 300 || isomeric_smiles.contains(['\u{FFFD}', '\0', '\'', '�']) {
        panic!(
            "malformed isomeric SMILES in the record stream: {:?} ({} bytes). \
             Skipping it would leave the rest of its record unread, and every \
             molecule after it would decode from the wrong offset.",
            isomeric_smiles,
            isomeric_smiles.len()
        );
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
    let mut sns_buf = [0u8; 2048];
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

    let mut chemberta_buf = [0u8; 1536];
    if molecular_representations.contains(&"chemberta".to_string()) {
        reader.read_exact(&mut chemberta_buf).ok()?;
    }

    let mut mhggnn_buf = [0u8; 4096];
    if molecular_representations.contains(&"mhggnn".to_string()) {
        reader.read_exact(&mut mhggnn_buf).ok()?;
    }

    // Avalon: 2048 bits packed to 256 bytes in Python and passed straight
    // through. Binary, so nothing here rescales or standardises it.
    let mut avalon_buf = [0u8; 256];
    if molecular_representations.contains(&"avalon".to_string()) {
        reader.read_exact(&mut avalon_buf).ok()?;
    }

    // ECFP4, LAST in the record. The Python writer appends it last and the
    // output record below puts it last too; the two orderings have to agree.
    let mut ecfp4_buf = [0u8; 256];
    if molecular_representations.contains(&"ecfp4".to_string()) {
        reader.read_exact(&mut ecfp4_buf).ok()?;
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
        chemberta_buf,
        mhggnn_buf,
        avalon_buf,
        ecfp4_buf,
    })
}

/// Read one split's clean labels, with the canonical SMILES each one belongs to.
///
/// The SMILES column is not decoration. The noise is built in this order and
/// applied in a second pass over the same file, so the write path can assert that
/// row `i` on the way out is the same molecule row `i` was on the way in. The
/// original held-out bug was exactly this going unchecked.
///
/// Every split is read this way now, not just training: validation carries its own
/// independently drawn noise, and the held-out splits carry the level-free shape
/// their region would receive, so all three need a plan of their own.
fn read_split_labels(
    config: &Config,
    split_name: &str,
    count: usize,
) -> io::Result<(Vec<String>, Vec<f32>)> {
    let train_file = File::open(format!("{}_{}.mmap", split_name, config.file_no))?;
    let mut reader = BufReader::new(train_file);
    reader.seek(SeekFrom::Start(0))?;

    let mut canonical = Vec::with_capacity(count);
    let mut labels = Vec::with_capacity(count);

    for index in 0..count {
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
                        "{} record {} could not be read but the configuration says there \
                         are {}. The record stream is truncated or misaligned; refusing to build \
                         a noise plan over a partial split.",
                        split_name, index, count
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

/// Write one memory-mapped split: every molecule's features, label and record.
///
/// Records are ALL-OR-NOTHING and every record is the same length. That is the
/// property the Python reader depends on: a record short by even one field
/// shifts the read offset of every molecule after it, and the file is then
/// silently misparsed from that point on with no error anywhere
/// (RERUN_PLAN.md §2.7).
///
/// The ECFP4 block is where that used to break. It is no longer computed here —
/// this side had no route to Morgan radius 2, so the Python writer computes it
/// and it is carried through as a fixed `[u8; 256]` (see `SmilesData`), which
/// cannot be short. What is checked here is the other half: an all-zero block
/// means the Python writer's own refusal was bypassed, and a row of zeros would
/// train as if it were real features. Such a molecule is recorded in `failures`
/// and `main` refuses to finish with any recorded failure unless
/// `--allow-featurisation-failures` is passed.
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

            // Carried through from the Python writer, which computes Morgan
            // radius 2 and REFUSES an all-zero row. Nothing is computed here any
            // more: this side had no route to radius 2 (see SmilesData). An
            // all-zero block reaching this point means the writer's guard was
            // bypassed, so it is recorded as a failure rather than written.
            let ecfp4_block = if wants_ecfp4 {
                let block = smiles_data.ecfp4_buf.to_vec();
                if block.iter().all(|b| *b == 0) {
                    failures.push(FeaturisationFailure {
                        split: split_name.to_string(),
                        index,
                        canonical_smiles: smiles_data.canonical_smiles.clone(),
                        reason: "ECFP4 is all zeros; a row with no features would \
                                 train as if it were featurised"
                            .to_string(),
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

            // Write randomized_smiles. The LENGTH GOES OUT EITHER WAY.
            //
            // There is nothing between one molecule and the next in this file, so
            // a field that is sometimes four bytes and sometimes nothing is not a
            // missing field -- it moves every molecule after it. The reader
            // consumes four bytes whenever this representation was asked for
            // (read_smiles_data), and the Python writer has always emitted a zero
            // for a molecule that has none (process_and_train.py). This used to
            // write nothing at all, so one molecule without a randomized SMILES
            // put the rest of the file out of step, silently.
            //
            // No molecule in the study takes this path -- QM9 drops molecules with
            // no randomized SMILES before writing, and the representation is
            // refused by name anyway -- so nothing that has been run changes.
            // The condition MIRRORS the reader's exactly -- the config, not
            // whether this molecule happens to have one. Writing the field when
            // the reader will not read it moves the file the other way.
            if config
                .molecular_representations
                .contains(&"randomized_smiles".to_string())
            {
                let bytes = smiles_data
                    .randomized_smiles
                    .as_ref()
                    .map(|s| s.as_bytes())
                    .unwrap_or(&[]);
                let len_bytes = (bytes.len() as u32).to_le_bytes();
                writer.write_all(&len_bytes)?;
                writer.write_all(bytes)?;
                if log_writes {
                    println!(
                        "randomized_smiles: {}",
                        smiles_data.randomized_smiles.as_deref().unwrap_or("(none)")
                    );
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

            // chemberta (1536 bytes = 384 float32)
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
            // Every split now arrives with ITS OWN plan, built over its own records
            // in its own order. That is what the original bug was the absence of: one
            // plan, built in training order, indexed by a counter that restarts at 0
            // for each split, so every held-out molecule got the noise drawn for the
            // training molecule at the same position (RERUN_PLAN.md §2.1).
            //
            // The identity check below is what stops that class of mistake rather
            // than this one instance of it, and it now runs on all three splits
            // whether or not the split receives noise. Row `index` on the way out
            // must be the same molecule as row `index` on the way in, or the run
            // dies here.
            //
            // `apply_noise` is false for the test split, and for validation under
            // --clean-validation. Those rows still carry `noise_pattern_raw` — the
            // shape their region would receive — but `epsilon_raw` and
            // `noise_scale_raw` are exactly zero.
            let y_clean_raw = smiles_data.target_value;
            assert!(
                index < plan.epsilon.len(),
                "guard: {} record {} has no entry in the noise plan (plan holds {})",
                split_name,
                index,
                plan.epsilon.len()
            );
            assert_eq!(
                plan.canonical[index], smiles_data.canonical_smiles,
                "guard: {} record {} is molecule {} on the way out but was {} \
                 when the noise was drawn — the record stream has drifted",
                split_name, index, smiles_data.canonical_smiles, plan.canonical[index]
            );
            let epsilon_raw = if apply_noise { plan.epsilon[index] } else { 0.0 };
            let noise_scale_raw = if apply_noise { plan.noise_scale[index] } else { 0.0 };
            let noise_pattern_raw = plan.noise_pattern[index];

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
                "{},{},{},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e}",
                split_name,
                index,
                smiles_data.canonical_smiles,
                y_clean_raw,
                epsilon_raw,
                noise_scale_raw,
                noise_pattern_raw,
                y_noisy_raw,
                property_value
            )?;

            let processed_bytes = property_value.to_le_bytes();
            writer.write_all(&processed_bytes)?;
            if log_writes {
                println!("noisy y: {}", property_value);
                println!("noisy y bytes: {:02X?}", processed_bytes);
            }

            // Write domain label if applicable.
            //
            // It is a LITERAL ZERO for every molecule, and the Python reader
            // consumes the byte and does nothing with it. So `--k_domains > 1`
            // does not carry a domain assignment through this file: whatever
            // clustering produced it is thrown away here, and anything reading
            // the byte sees one constant (RERUN_PLAN.md 2.13). Kept as a byte so
            // the record layout does not change; refused above instead.
            if config.k_domains > 1 {
                writer.write_all(&[0u8])?;
                if log_writes {
                    println!("domain_flag: 0 (placeholder -- see the note above)");
                }
            }

            // Write smiles or randomized_smiles OHE if used
            for smiles_type in ["smiles", "randomized_smiles"] {
                if config.molecular_representations.contains(&smiles_type.to_string()) {
                    let smiles_string = if smiles_type == "smiles" {
                        &smiles_data.canonical_smiles
                    } else {
                        // A molecule with no randomized SMILES cannot be encoded
                        // against the vocabulary, and an all-zero row would be a
                        // silent lie in a column a model then trains on. The
                        // record format allows the field to be empty -- the
                        // Python writer emits a zero length for it and the reader
                        // yields nothing -- so this says which molecule and stops,
                        // rather than unwrapping and panicking with no name in the
                        // message (RERUN_PLAN.md 2.19).
                        match smiles_data.randomized_smiles.as_ref() {
                            Some(s) => s,
                            None => panic!(
                                "molecule {:?} has no randomized SMILES, but \
                                 `randomized_smiles` is among the representations. \
                                 It cannot be one-hot encoded, and writing an \
                                 all-zero row would put a molecule with no features \
                                 into the training column under its own name.",
                                smiles_data.canonical_smiles
                            ),
                        }
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
    train_plan: &NoisePlan,
    val_plan: &NoisePlan,
    test_plan: &NoisePlan,
    noise_validation: bool,
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
        "split,record_index,canonical_smiles,y_clean_raw,epsilon_raw,noise_scale_raw,\
noise_pattern_raw,y_noisy_raw,y_written"
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
        train_plan,
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
        val_plan,
        &tokenizer,
        vocab,
        vocab_size,
        max_sequence_length,
        config.val_count,
        config.logging,
        // Decision 3, settled 2026-08-26: validation labels carry their own noise,
        // drawn independently, anchored on the clean TRAINING spread. Training noisy,
        // validation noisy from a separate draw, test clean. --clean-validation
        // restores the old behaviour.
        noise_validation,
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
        test_plan,
        &tokenizer,
        vocab,
        vocab_size,
        max_sequence_length,
        config.test_count,
        config.logging,
        // The test split is NEVER noised. This is the original bug's fix and it is
        // not on a flag.
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

fn write_noise_manifest(
    path: &str,
    plan: &NoisePlan,
    spec: &NoiseSpec,
    validation: &NoisePlan,
    noise_validation: bool,
) -> io::Result<()> {
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
        // The spread the dose was measured against. On the training plan this is the
        // training spread; every other split is anchored on the SAME number, which is
        // the whole point of recording it.
        "dose_reference_sd": plan.dose_reference_sd,
        // Decision 3 (2026-08-26): validation carries its own noise, drawn from an
        // independent seed and dosed against the clean TRAINING spread. Recorded here
        // so a results row can be traced to what validation actually received.
        "validation_noised": noise_validation,
        "validation_seed": validation.seed,
        "validation_n": validation.n_train,
        "validation_label_sd": validation.clean_label_sd,
        "validation_target_dose_in_label_units": validation.target_dose_label_units,
        "validation_delivered_dose_in_label_units": validation.realised_dose_label_units,
        "validation_affected_molecule_fraction": validation.affected_molecule_fraction,
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
             .help("Fraction of labels contaminated (outlier). Default 0.10, the one \
                    setting the study runs (noise_conditions.json)"))
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
        .arg(Arg::new("clean_validation")
             .long("clean-validation")
             .action(ArgAction::SetTrue)
             .help("Leave the validation labels clean, the pre-2026-08-26 behaviour. \
                    OFF by default: validation carries its own noise, drawn from an \
                    independent seed and dosed against the CLEAN TRAINING spread so both \
                    splits receive the same amount. The test split is never noised either way"))
        .arg(Arg::new("allow_featurisation_failures")
             .long("allow-featurisation-failures")
             .action(ArgAction::SetTrue)
             .help("Finish even if some molecules were written with an all-zero ECFP4 block. \
                    Off by default: a zero fingerprint trains as if it were real features"))
        .arg(Arg::new("json")
             .long("json")
             .action(ArgAction::SetTrue)
             .help("With --self-test: emit the statistics as JSON, in the same shape \
                    rust/reference/noise_arms.rs emits, so the two can be compared"))
        .arg(Arg::new("seeds")
             .long("seeds")
             .action(ArgAction::Set)
             .help("With --self-test --json: how many seeds to average the delivered dose over"))
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
    // 0.10, not 0.05. Settled 2026-08-27 (`noise_conditions.json`, RERUN_PLAN.md
    // §13.9): twelve replicates on real QM9 put 1%, 5% and 10% contamination within
    // 0.005 R² of each other and of Gaussian, so one setting runs rather than three,
    // and it is the top of Hampel's published range -- the strongest contamination,
    // where anything that is ever going to show will show.
    let outlier_p: f32 = matches
        .get_one::<String>("outlier_p")
        .map(|v| v.parse().expect("--outlier-p must be a valid float"))
        .unwrap_or(0.10);
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
        // Both spellings. The keyword was `grouped_wide` while every results row,
        // every manifest and every figure says `grouped_wider`, so typing the name
        // read off a row killed the run with "unknown --noise-targeting". Same for
        // the shifted pair. The EMITTED name is unchanged -- `NoiseTargeting::name`
        // above still returns one string per condition, and that is what rows join on.
        "grouped_wide" | "grouped_wider" => {
            assert!(
                group_variance_share.is_finite(),
                "--group-variance-share is not used by grouped_wide"
            );
            NoiseTargeting::GroupedWide {
                lambda,
                group_fraction,
            }
        }
        "grouped_shift" | "grouped_shifted" => {
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
            "unknown --noise-targeting '{}'. Valid: uniform, grouped_wide (or grouped_wider), \
             grouped_shift (or grouped_shifted), outlier, censoring",
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
        if matches.get_flag("json") {
            let seeds: u64 = matches
                .get_one::<String>("seeds")
                .map(|v| v.parse().expect("--seeds must be an integer"))
                .unwrap_or(20);
            let k: f32 = matches
                .get_one::<String>("noise_level")
                .map(|v| v.parse().expect("--noise-level must be a valid float"))
                .unwrap_or(0.5);
            let only = if matches.get_one::<String>("noise_targeting").is_some()
                && matches.get_one::<String>("noise_shape").is_some()
            {
                Some((shape, targeting))
            } else {
                None
            };
            self_test_json(&labels, groups.as_ref(), &canonical, k, seeds, only);
            return Ok(());
        }
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

    let noise_validation = !matches.get_flag("clean_validation");

    let (canonical, clean_labels) = read_split_labels(&config, "train", config.train_count)?;
    assert_eq!(
        clean_labels.len(),
        config.train_count,
        "guard: read {} training records but the configuration says {} — the record stream is short",
        clean_labels.len(),
        config.train_count
    );
    let (val_canonical, val_labels) = read_split_labels(&config, "val", config.val_count)?;
    let (test_canonical, test_labels) = read_split_labels(&config, "test", config.test_count)?;

    let groups = if spec.targeting.needs_groups() {
        Some(load_scaffold_groups(&scaffold_path)?)
    } else {
        None
    };

    // The training plan first: it DEFINES the condition. The dose, the assay limit
    // and the affected scaffold groups are all read off the clean training labels,
    // and every other split then measures itself against them.
    let train_ctx = PlanContext::training(&clean_labels);
    let plan = build_noise_plan(&clean_labels, &canonical, &spec, groups.as_ref(), &train_ctx)?;
    assert_noise_plan_gates(&plan, &spec, &clean_labels, &train_ctx);

    // Validation. An independent draw — a separate seed, so validation's noise is not
    // a copy or a continuation of training's — but the SAME condition: the same dose
    // in absolute label units, the same assay limit, the same affected scaffold
    // groups. Decision 3, settled 2026-08-26.
    let val_spec = NoiseSpec {
        seed: derive_split_seed(spec.seed, VALIDATION_SEED_TAG),
        ..spec.clone()
    };
    let val_ctx = PlanContext {
        reference_labels: &clean_labels,
        shared: Some(&plan.targeting_state),
        apply: noise_validation,
        split_name: "val",
    };
    let val_plan = build_noise_plan(
        &val_labels,
        &val_canonical,
        &val_spec,
        groups.as_ref(),
        &val_ctx,
    )?;
    assert_noise_plan_gates(&val_plan, &val_spec, &val_labels, &val_ctx);
    if noise_validation {
        // The two splits must receive the same AMOUNT, not the same level applied to
        // two different spreads. Nothing else checks it at run time: the unit tests
        // check the property on synthetic labels, and a real run has neither their
        // labels nor their tolerances.
        assert_validation_matches_training(&plan, &val_plan, &spec);
    }

    // Test. Never noised — that is the original bug's fix and it is not on a flag.
    // The plan exists so held-out rows still carry `noise_pattern_raw`, the level-free
    // shape their region would receive; every applied amount on it is exactly zero.
    let test_spec = NoiseSpec {
        seed: derive_split_seed(spec.seed, TEST_SEED_TAG),
        ..spec.clone()
    };
    let test_ctx = PlanContext {
        reference_labels: &clean_labels,
        shared: Some(&plan.targeting_state),
        apply: false,
        split_name: "test",
    };
    let test_plan = build_noise_plan(
        &test_labels,
        &test_canonical,
        &test_spec,
        groups.as_ref(),
        &test_ctx,
    )?;
    assert_noise_plan_gates(&test_plan, &test_spec, &test_labels, &test_ctx);

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
    if noise_validation {
        println!(
            "      validation: seed {} (derived from {}), delivered {:.6} against training's \
{:.6} — both dosed against the clean TRAINING spread {:.6}, not validation's own {:.6}",
            val_plan.seed,
            spec.seed,
            val_plan.realised_dose_label_units,
            plan.realised_dose_label_units,
            plan.clean_label_sd,
            val_plan.clean_label_sd
        );
    } else {
        println!("      validation: CLEAN (--clean-validation was given)");
    }
    println!("      test: clean, always");

    write_noise_manifest(&manifest_path, &plan, &spec, &val_plan, noise_validation)?;

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
        &val_plan,
        &test_plan,
        noise_validation,
        max_sequence_length,
        &provenance_path,
        &failures_path,
        allow_featurisation_failures,
    )?;

    Ok(())
}
