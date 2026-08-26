//! Reference implementation of the redesigned noise conditions, with dose matching.
//!
//! Reference for `rust/src/main.rs`. Everything here is self-contained: no RDKit,
//! no memmap, no pipeline. It exists to prove the conditions deliver the dose they
//! promise before anything in the real pipeline is touched, and to be the fixed
//! point the Python injector is checked against.
//!
//! The contract, for every dose-matched condition:
//!   dose = k * SD(clean training labels)
//!   the condition's internal scale is SOLVED so the realised RMS noise equals that.
//!
//! Censoring is neither zero-mean nor dose-matched, so it takes a censored
//! fraction instead of a dose and reports its delivered dose as a diagnostic.
//!
//! Build and run:
//!   cd rust/reference && cargo run --release -- <labels.txt> [--groups <g.txt>] [--json] [--seeds N]
//!
//! The condition names printed here are exactly the names in
//! `noiseInject.CONDITIONS`, so the two implementations can be joined on them.
//! Specification: `NOISE_DESIGN.md` sections 1, 2, 2a and 6.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal, ChiSquared};
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(Debug, Clone)]
pub enum NoiseArm {
    /// Every label perturbed by the same typical amount.
    Gaussian,
    /// Heavy-tailed. `nu` must be > 2 or the variance is undefined.
    StudentT { nu: f32 },
    /// Heavy-tailed, the shape fitted to real bioactivity data.
    Laplace,
    /// Whole groups (e.g. scaffold clusters) get a `lambda`x wider error,
    /// still centred on the true value.
    GroupedWider { lambda: f32, group_fraction: f32 },
    /// Whole groups have their labels pushed in one direction by a constant.
    /// `rho` is the share of the total variance carried by the group term:
    /// 0.62, from Bentz et al. (2013) Table 7. The offsets are deliberately
    /// NOT centred -- the asymmetry is the mechanism under test.
    GroupedShifted { rho: f32 },
    /// A random fraction `p` of labels get a `lambda`x wider error.
    /// Selection is RANDOM, not by label value: a mistyped unit is a property
    /// of the record, not of the value.
    Outlier { p: f32, lambda: f32 },
    /// Values past an assay limit are recorded as the limit. Not zero-mean,
    /// not dose-matched.
    Censoring { fraction: f32 },
}

impl NoiseArm {
    fn is_dose_matched(&self) -> bool {
        !matches!(self, NoiseArm::Censoring { .. })
    }
    fn needs_groups(&self) -> bool {
        matches!(self, NoiseArm::GroupedWider { .. } | NoiseArm::GroupedShifted { .. })
    }
}

/// The per-molecule scale multipliers this condition would apply at unit scale,
/// and the fraction of MOLECULES that end up affected.
///
/// For conditions whose scale does not depend on the record, this is all ones.
fn scale_map(arm: &NoiseArm, n: usize, groups: Option<&[usize]>, rng: &mut StdRng) -> (Vec<f32>, f64) {
    match arm {
        // 1.0, not 0.0: every molecule IS affected. "No targeting applies" is a
        // defensible reading of the name but cannot share the column with the
        // selection rules' fractions. Settled 2026-08-26 across all three
        // implementations -- see NOISE_DESIGN.md section 5.1c.
        NoiseArm::Gaussian | NoiseArm::StudentT { .. } | NoiseArm::Laplace => (vec![1.0; n], 1.0),

        // The group term enters additively, not as a multiplier -- see `generate`.
        NoiseArm::GroupedShifted { .. } => (vec![1.0; n], 1.0),

        NoiseArm::GroupedWider { lambda, group_fraction } => {
            let g = groups.expect("grouped noise requires group assignments");
            let affected = select_groups_by_molecule_fraction(g, *group_fraction, rng);
            let frac = affected.iter().filter(|a| **a).count() as f64 / n as f64;
            let scales = affected.iter().map(|a| if *a { *lambda } else { 1.0 }).collect();
            (scales, frac)
        }

        NoiseArm::Outlier { p, lambda } => {
            let hit: Vec<bool> = (0..n).map(|_| rng.random::<f32>() < *p).collect();
            let frac = hit.iter().filter(|h| **h).count() as f64 / n as f64;
            (hit.iter().map(|h| if *h { *lambda } else { 1.0 }).collect(), frac)
        }

        NoiseArm::Censoring { fraction } => (vec![1.0; n], *fraction as f64),
    }
}

/// Choose whole groups until the affected MOLECULE fraction is closest to `f`.
///
/// Selecting a fraction of *groups* does not control who gets hit: real Murcko
/// scaffolds are very unevenly sized. Measured on 10,000 QM9 molecules, where
/// 32% share one empty (acyclic) scaffold, a nominal group fraction of 0.2
/// delivered an affected molecule fraction anywhere between 0.067 and 0.551.
/// See NOISE_DESIGN.md section 2a rule 1.
fn select_groups_by_molecule_fraction(g: &[usize], f: f32, rng: &mut StdRng) -> Vec<bool> {
    let n = g.len();
    let mut uniq: Vec<usize> = g.to_vec();
    uniq.sort_unstable();
    uniq.dedup();

    let mut sizes = std::collections::HashMap::new();
    for gi in g {
        *sizes.entry(*gi).or_insert(0usize) += 1;
    }

    // Shuffle the group order (Fisher-Yates, so the same group is never tried twice).
    let mut order: Vec<usize> = (0..uniq.len()).collect();
    for i in (1..order.len()).rev() {
        order.swap(i, rng.random_range(0..=i));
    }

    let mut chosen: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut cum = 0usize;
    for idx in order {
        let gid = uniq[idx];
        let size = sizes[&gid];
        let here = (cum as f32 / n as f32 - f).abs();
        let there = ((cum + size) as f32 / n as f32 - f).abs();
        // Skip a group that would take us further from the target than stopping.
        if cum > 0 && there > here {
            continue;
        }
        chosen.insert(gid);
        cum += size;
        if cum as f32 / n as f32 >= f {
            break;
        }
    }
    g.iter().map(|gi| chosen.contains(gi)).collect()
}

/// Unit dose G: the RMS of the per-molecule scale map, times the shape's own
/// unit standard deviation. Solving `scale = target / G` makes the realised
/// RMS noise equal `target` in expectation.
fn unit_dose(arm: &NoiseArm, scales: &[f32]) -> f32 {
    // Accumulate in f64. The draws stay f32, matching the pipeline, but summing
    // 133,885 f32 squares loses enough precision to look like a real
    // disagreement with the Python side when it is only accumulation error.
    let ms: f64 = scales.iter().map(|s| (*s as f64) * (*s as f64)).sum::<f64>() / scales.len() as f64;
    let shape_sd = match arm {
        // A standard t with nu d.f. has variance nu/(nu-2).
        NoiseArm::StudentT { nu } => (nu / (nu - 2.0)).sqrt(),
        // A Laplace with scale 1 has variance 2.
        NoiseArm::Laplace => 2f32.sqrt(),
        _ => 1.0,
    };
    (ms.sqrt() as f32) * shape_sd
}

/// Draw one standardised deviate from the condition's shape (unit scale parameter).
fn draw_shape(arm: &NoiseArm, rng: &mut StdRng) -> f32 {
    match arm {
        NoiseArm::StudentT { nu } => {
            // t_nu = Z / sqrt(V/nu),  V ~ chi-squared(nu)
            let z: f32 = StandardNormal.sample(rng);
            let chi = ChiSquared::new(*nu).unwrap();
            let v: f32 = chi.sample(rng);
            z / (v / nu).sqrt()
        }
        NoiseArm::Laplace => {
            // inverse-CDF: -sgn(u) * ln(1-2|u|), u ~ U(-0.5, 0.5)
            let u: f32 = rng.random::<f32>() - 0.5;
            -u.signum() * (1.0 - 2.0 * u.abs()).ln()
        }
        _ => StandardNormal.sample(rng),
    }
}

/// What one injection produced, and everything needed to trace it.
pub struct Injection {
    pub epsilon: Vec<f32>,
    pub unit_dose: f32,
    pub solved_scale: f32,
    pub affected_fraction: f64,
    pub censoring_limit: f32,
    /// How many independent contributions the delivered dose is averaged over.
    /// NOT the record count: a scale map that concentrates the noise on a few
    /// records, or a group-level term drawn once per group, pins the dose far
    /// less precisely than the raw count suggests. Matches `effective_n` in
    /// `rust/src/main.rs` and `_effective_n` in noiseInject.
    pub effective_n: f64,
}

/// Generate the noise vector.
/// `k` is the noise-to-label-spread ratio; `label_sd` is the SD of the CLEAN
/// training labels. Censoring ignores both and clips instead.
pub fn generate(
    arm: &NoiseArm,
    y: &[f32],
    k: f32,
    label_sd: f32,
    groups: Option<&[usize]>,
    seed: u64,
) -> Injection {
    let n = y.len();
    let mut rng = StdRng::seed_from_u64(seed);
    let target = k * label_sd;

    if let NoiseArm::Censoring { fraction } = arm {
        if *fraction <= 0.0 {
            // A censored fraction of zero is the CLEAN baseline: nothing is
            // clipped, and there is no assay limit to report. Reporting the
            // largest label as "the limit" would put a number where there is
            // no quantity -- and the Python injector reports none, so the two
            // would disagree on a row that is otherwise identical.
            return Injection {
                epsilon: vec![0.0; n],
                unit_dose: f32::NAN,
                solved_scale: f32::NAN,
                affected_fraction: 0.0,
                censoring_limit: f32::NAN,
                effective_n: n as f64,
            };
        }
        let limit = upper_quantile(y, 1.0 - *fraction);
        let epsilon: Vec<f32> = y.iter().map(|v| if *v > limit { limit - *v } else { 0.0 }).collect();
        let affected = epsilon.iter().filter(|e| **e != 0.0).count() as f64 / n as f64;
        return Injection {
            epsilon,
            unit_dose: f32::NAN,
            solved_scale: f32::NAN,
            affected_fraction: affected,
            censoring_limit: limit,
            effective_n: n as f64,
        };
    }

    let (scales, affected) = scale_map(arm, n, groups, &mut rng);
    let g = unit_dose(arm, &scales);
    let solved = target / g;

    let epsilon: Vec<f32> = match arm {
        NoiseArm::GroupedShifted { rho } => {
            // eps_i = sqrt(rho)*tau*b_g(i) + sqrt(1-rho)*tau*e_i.
            // The two variances sum to tau^2 by construction, so this is
            // dose-matched without a solver step. NOT centred.
            let g_ids = groups.expect("grouped noise requires group assignments");
            let mut uniq: Vec<usize> = g_ids.to_vec();
            uniq.sort_unstable();
            uniq.dedup();
            let index: std::collections::HashMap<usize, usize> =
                uniq.iter().enumerate().map(|(i, u)| (*u, i)).collect();
            let offsets: Vec<f32> = (0..uniq.len()).map(|_| draw_shape(arm, &mut rng)).collect();
            let a = rho.sqrt();
            let b = (1.0 - rho).sqrt();
            g_ids
                .iter()
                .map(|gi| target * (a * offsets[index[gi]] + b * draw_shape(arm, &mut rng)))
                .collect()
        }
        _ => scales.iter().map(|s| draw_shape(arm, &mut rng) * solved * s).collect(),
    };

    let effective_n = match arm {
        NoiseArm::GroupedShifted { rho } => {
            // NOT the group count. The group term is averaged over MOLECULES, so
            // a few large groups dominate: the effective number of independent
            // group contributions is (sum n_g)^2 / sum n_g^2, which on real QM9
            // scaffolds is 189 against a group count of 30,313 -- a factor of
            // 160. Using the count overstates the precision of the delivered
            // dose by the same factor, and makes the flat-dose gate fail on
            // sampling variability that was never a defect.
            let ids = groups.expect("grouped noise requires group assignments");
            let mut sizes = std::collections::HashMap::new();
            for gi in ids {
                *sizes.entry(*gi).or_insert(0f64) += 1.0;
            }
            let total: f64 = sizes.values().sum();
            let sum_sq: f64 = sizes.values().map(|c| c * c).sum();
            let eff_groups = if sum_sq > 0.0 { total * total / sum_sq } else { 1.0 };
            let r = *rho as f64;
            1.0 / (r * r / eff_groups + (1.0 - r) * (1.0 - r) / n as f64)
        }
        _ => {
            let s2: f64 = scales.iter().map(|s| (*s as f64) * (*s as f64)).sum();
            let s4: f64 = scales.iter().map(|s| (*s as f64).powi(4)).sum();
            if s4 > 0.0 { s2 * s2 / s4 } else { n as f64 }
        }
    };

    Injection { epsilon, unit_dose: g, solved_scale: solved, affected_fraction: affected,
                censoring_limit: f32::NAN, effective_n }
}

fn upper_quantile(y: &[f32], q: f32) -> f32 {
    // Linear interpolation between order statistics, matching numpy's default.
    let mut s: Vec<f32> = y.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pos = q as f64 * (s.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        s[lo]
    } else {
        let w = (pos - lo as f64) as f32;
        s[lo] * (1.0 - w) + s[hi] * w
    }
}

/// How close the realised dose can be expected to land, for THIS condition.
/// Three standard errors of an RMS estimate, floored at the half a percent the
/// design quotes for the full QM9 label column. Student-t at nu <= 4 gets a flat
/// 15%: its fourth moment is infinite, so the sample kurtosis this is computed
/// from is itself meaningless. Matches `dose_tolerance` in `rust/src/main.rs`
/// and `noiseInject.dose_tolerance`.
fn dose_tolerance(arm: &NoiseArm, epsilon: &[f32], effective_n: f64) -> f64 {
    if let NoiseArm::StudentT { nu } = arm {
        if *nu <= 4.0 {
            return 0.15;
        }
    }
    let n_eff = effective_n.max(1.0);
    let m2: f64 = epsilon.iter().map(|e| (*e as f64).powi(2)).sum::<f64>() / epsilon.len() as f64;
    let m4: f64 = epsilon.iter().map(|e| (*e as f64).powi(4)).sum::<f64>() / epsilon.len() as f64;
    let kurtosis = if m2 > 0.0 { (m4 / (m2 * m2)).clamp(3.0, 60.0) } else { 3.0 };
    (3.0 * ((kurtosis - 1.0) / (4.0 * n_eff)).sqrt()).max(0.005)
}

// --- statistics, all accumulated in f64 -------------------------------------

fn rms(v: &[f32]) -> f64 {
    (v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / v.len() as f64).sqrt()
}

fn mean(v: &[f32]) -> f64 {
    v.iter().map(|x| *x as f64).sum::<f64>() / v.len() as f64
}

fn median_abs(v: &[f32]) -> f64 {
    let mut a: Vec<f64> = v.iter().map(|x| x.abs() as f64).collect();
    a.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let n = a.len();
    if n % 2 == 1 { a[n / 2] } else { 0.5 * (a[n / 2 - 1] + a[n / 2]) }
}

/// The worst-hit 5%'s share of the total noise energy: how concentrated the
/// damage is, at a fixed total amount.
fn top5_energy_share(v: &[f32]) -> f64 {
    let mut sq: Vec<f64> = v.iter().map(|x| (*x as f64) * (*x as f64)).collect();
    sq.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let total: f64 = sq.iter().sum();
    if total <= 0.0 { return 0.0; }
    let k = ((sq.len() as f64) * 0.05).round() as usize;
    sq[..k.max(1)].iter().sum::<f64>() / total
}

fn frac_beyond(v: &[f32], threshold: f64) -> f64 {
    v.iter().filter(|e| (e.abs() as f64) > threshold).count() as f64 / v.len() as f64
}

// --- the condition roster, named exactly as noiseInject.CONDITIONS ----------

fn roster() -> Vec<(String, NoiseArm)> {
    let mut v: Vec<(String, NoiseArm)> = vec![
        ("gaussian".into(), NoiseArm::Gaussian),
        ("student_t_nu10".into(), NoiseArm::StudentT { nu: 10.0 }),
        ("student_t_nu5".into(), NoiseArm::StudentT { nu: 5.0 }),
        ("student_t_nu3".into(), NoiseArm::StudentT { nu: 3.0 }),
        ("laplace".into(), NoiseArm::Laplace),
        ("grouped_wider".into(), NoiseArm::GroupedWider { lambda: 3.0, group_fraction: 0.2 }),
        ("grouped_shifted".into(), NoiseArm::GroupedShifted { rho: 0.62 }),
        ("outlier_p01".into(), NoiseArm::Outlier { p: 0.01, lambda: 3.0 }),
        ("outlier_p05".into(), NoiseArm::Outlier { p: 0.05, lambda: 3.0 }),
        ("outlier_p10".into(), NoiseArm::Outlier { p: 0.10, lambda: 3.0 }),
    ];
    // Censoring is swept on its own axis: the level IS the fraction clipped, so
    // the name is derived from it rather than stored beside it. One number, one
    // place. Matches `condition_name` in rust/src/main.rs and the derived name
    // in noiseInject.
    for pct in CENSORING_PERCENTS {
        v.push((format!("censoring_{}", pct), NoiseArm::Censoring { fraction: pct as f32 / 100.0 }));
    }
    v
}

/// The censoring sweep. 0 is the clean baseline and must clip nothing --
/// previously the fraction was baked into the condition, so a "level 0" run
/// still clipped and the clean baseline was not clean.
const CENSORING_PERCENTS: [u32; 7] = [0, 10, 20, 25, 30, 40, 50];

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut path = "labels.txt".to_string();
    let mut groups_path: Option<String> = None;
    let mut as_json = false;
    let mut seeds: u64 = 1;
    let mut ks: Vec<f32> = vec![0.25, 0.5, 1.0];

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--groups" => { groups_path = Some(args[i + 1].clone()); i += 2; }
            "--json" => { as_json = true; i += 1; }
            "--seeds" => { seeds = args[i + 1].parse().expect("--seeds takes an integer"); i += 2; }
            "--k" => {
                ks = args[i + 1].split(',').map(|s| s.trim().parse().expect("--k takes a comma-separated list")).collect();
                i += 2;
            }
            other => { path = other.to_string(); i += 1; }
        }
    }

    let y = read_f32_lines(&path);
    let n = y.len();
    let mu = mean(&y);
    let label_sd = (y.iter().map(|v| (*v as f64 - mu).powi(2)).sum::<f64>() / n as f64).sqrt() as f32;

    // Groups: read them if given, so both implementations use the SAME
    // assignment. Otherwise fall back to 2,000 synthetic clusters.
    let groups: Vec<usize> = match &groups_path {
        Some(p) => read_usize_lines(p),
        None => {
            let mut grng = StdRng::seed_from_u64(7);
            (0..n).map(|_| grng.random_range(0..2000)).collect()
        }
    };
    assert_eq!(groups.len(), n, "groups file has a different length from the labels");

    if !as_json {
        println!("labels: n={}  mean={:.4}  SD={:.4}  groups={}\n", n, mu, label_sd,
                 { let mut u = groups.clone(); u.sort_unstable(); u.dedup(); u.len() });
    }

    let mut rows: Vec<String> = Vec::new();

    for k in &ks {
        let target = (*k * label_sd) as f64;
        if !as_json {
            println!("=== k = {}  ->  target dose = {:.4} ===", k, target);
        }
        for (name, arm) in roster() {
            // Censoring does not depend on k, so report it once.
            if !arm.is_dose_matched() && *k != ks[0] {
                continue;
            }
            let g_opt = if arm.needs_groups() { Some(&groups[..]) } else { None };

            let mut realised = Vec::new();
            let mut shift = Vec::new();
            let mut f3 = Vec::new();
            let mut med = Vec::new();
            let mut top5 = Vec::new();
            let mut affected = Vec::new();
            let mut g_val = f32::NAN;
            let mut solved = f32::NAN;
            let mut limit = f32::NAN;
            let mut eff_n = 0.0f64;
            let mut tol = 0.0f64;

            for seed in 0..seeds {
                let inj = generate(&arm, &y, *k, label_sd, g_opt, 42 + seed);
                realised.push(rms(&inj.epsilon));
                shift.push(mean(&inj.epsilon));
                // For censoring the dose is a diagnostic, so scale the
                // "badly wrong" threshold to what was actually delivered.
                let thr = if arm.is_dose_matched() { 3.0 * target } else { 3.0 * rms(&inj.epsilon) };
                f3.push(frac_beyond(&inj.epsilon, thr));
                med.push(median_abs(&inj.epsilon));
                top5.push(top5_energy_share(&inj.epsilon));
                affected.push(inj.affected_fraction);
                g_val = inj.unit_dose;
                solved = inj.solved_scale;
                limit = inj.censoring_limit;
                eff_n = inj.effective_n;
                tol = dose_tolerance(&arm, &inj.epsilon, inj.effective_n);
            }
            let avg = |v: &Vec<f64>| v.iter().sum::<f64>() / v.len() as f64;
            let sd_of = |v: &Vec<f64>| {
                let m = avg(v);
                (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
            };

            if as_json {
                rows.push(format!(
                    "{{\"condition\":\"{}\",\"k\":{},\"dose_matched\":{},\"target_dose\":{},\
                     \"unit_dose\":{},\"solved_scale\":{},\"censoring_limit\":{},\
                     \"delivered_dose\":{},\"delivered_dose_sd\":{},\"mean_shift\":{},\
                     \"frac_beyond_3\":{},\"median_abs\":{},\"top5_energy_share\":{},\
                     \"affected_molecule_fraction\":{},\"effective_n\":{},\
                     \"dose_tolerance\":{},\"seeds\":{}}}",
                    name, k, arm.is_dose_matched(), json_f64(target),
                    json_f64(g_val as f64), json_f64(solved as f64), json_f64(limit as f64),
                    json_f64(avg(&realised)), json_f64(sd_of(&realised)), json_f64(avg(&shift)),
                    json_f64(avg(&f3)), json_f64(avg(&med)), json_f64(avg(&top5)),
                    json_f64(avg(&affected)), json_f64(eff_n), json_f64(tol), seeds));
            } else {
                let err = 100.0 * (avg(&realised) / target - 1.0);
                println!(
                    "  {:18} G={:.4} scale={:.4}  realised={:.4}  err={:+.2}%  >3x dose={:.2}%  affected={:.3}",
                    name, g_val, solved, avg(&realised), err, avg(&f3) * 100.0, avg(&affected));
            }
        }
        if !as_json { println!(); }
    }

    if as_json {
        println!("{{\"n\":{},\"label_sd\":{},\"rows\":[{}]}}", n, json_f64(label_sd as f64), rows.join(","));
    }
}

/// NaN and infinity are not valid JSON. Emit null instead of writing a file the
/// consumer cannot parse.
fn json_f64(v: f64) -> String {
    if v.is_finite() { format!("{}", v) } else { "null".to_string() }
}

fn read_f32_lines(path: &str) -> Vec<f32> {
    let f = File::open(path).unwrap_or_else(|_| panic!("cannot open {}", path));
    BufReader::new(f)
        .lines()
        .filter_map(|l| l.ok())
        .filter_map(|l| l.trim().parse::<f32>().ok())
        .collect()
}

fn read_usize_lines(path: &str) -> Vec<usize> {
    let f = File::open(path).unwrap_or_else(|_| panic!("cannot open {}", path));
    BufReader::new(f)
        .lines()
        .filter_map(|l| l.ok())
        .filter_map(|l| l.trim().parse::<usize>().ok())
        .collect()
}
