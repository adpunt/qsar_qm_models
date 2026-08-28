//! Reference implementation of the redesigned noise arms, with dose matching.
//!
//! Drop-in for rust/src/main.rs. Everything here is self-contained: no RDKit,
//! no memmap, no pipeline. It exists to prove the arms deliver the dose they
//! promise before anything in the real pipeline is touched.
//!
//! The contract, for every arm:
//!   dose = k * SD(clean training labels)
//!   the arm's internal scale is SOLVED so the realised RMS noise equals that.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal, StandardNormal, ChiSquared};
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(Debug, Clone)]
pub enum NoiseArm {
    /// A. Every label perturbed by the same typical amount.
    Gaussian,
    /// B. Heavy-tailed. `nu` must be > 2 or the variance is undefined.
    StudentT { nu: f32 },
    /// C. Heavy-tailed, the shape fitted to real bioactivity data.
    Laplace,
    /// D. Whole groups (e.g. scaffold clusters) get a `lambda`x wider error.
    Grouped { lambda: f32, group_fraction: f32 },
    /// E. A random fraction `p` of labels get a `lambda`x wider error.
    /// This is the Outlier arm: selection is RANDOM, not by label value.
    Outlier { p: f32, lambda: f32 },
}

/// The per-molecule scale multipliers this arm would apply at unit scale.
/// For arms whose shape does not depend on the labels, this is all ones.
fn scale_map(arm: &NoiseArm, n: usize, groups: Option<&[usize]>, rng: &mut StdRng) -> Vec<f32> {
    match arm {
        NoiseArm::Gaussian | NoiseArm::StudentT { .. } | NoiseArm::Laplace => vec![1.0; n],

        NoiseArm::Grouped { lambda, group_fraction } => {
            let g = groups.expect("Grouped arm requires group assignments");
            let mut uniq: Vec<usize> = g.to_vec();
            uniq.sort_unstable();
            uniq.dedup();
            let n_bad = ((group_fraction * uniq.len() as f32).round() as usize).max(1);
            // choose n_bad distinct groups uniformly
            let mut chosen = vec![false; uniq.len()];
            let mut picked = 0;
            while picked < n_bad {
                let i = rng.random_range(0..uniq.len());
                if !chosen[i] { chosen[i] = true; picked += 1; }
            }
            let bad: std::collections::HashSet<usize> =
                uniq.iter().zip(chosen.iter()).filter(|(_, c)| **c).map(|(u, _)| *u).collect();
            g.iter().map(|gi| if bad.contains(gi) { *lambda } else { 1.0 }).collect()
        }

        NoiseArm::Outlier { p, lambda } => {
            (0..n).map(|_| if rng.random::<f32>() < *p { *lambda } else { 1.0 }).collect()
        }
    }
}

/// Unit dose G: the RMS of the per-molecule scale map, times the shape's own
/// unit standard deviation. Solving `scale = target / G` makes the realised
/// RMS noise equal `target` exactly.
fn unit_dose(arm: &NoiseArm, scales: &[f32]) -> f32 {
    let ms: f32 = scales.iter().map(|s| s * s).sum::<f32>() / scales.len() as f32;
    let shape_sd = match arm {
        // A standard t with nu d.f. has variance nu/(nu-2).
        NoiseArm::StudentT { nu } => (nu / (nu - 2.0)).sqrt(),
        // A Laplace with scale 1 has variance 2.
        NoiseArm::Laplace => 2f32.sqrt(),
        _ => 1.0,
    };
    ms.sqrt() * shape_sd
}

/// Draw one standardised deviate from the arm's shape (unit scale parameter).
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

/// Generate the dose-matched noise vector.
/// `k` is the noise-to-label-spread ratio; `label_sd` is SD of the CLEAN training labels.
pub fn generate(
    arm: &NoiseArm,
    n: usize,
    k: f32,
    label_sd: f32,
    groups: Option<&[usize]>,
    seed: u64,
) -> (Vec<f32>, f32, f32) {
    let mut rng = StdRng::seed_from_u64(seed);
    let target = k * label_sd;
    let scales = scale_map(arm, n, groups, &mut rng);
    let g = unit_dose(arm, &scales);
    let solved = target / g;
    let noise: Vec<f32> = scales
        .iter()
        .map(|s| draw_shape(arm, &mut rng) * solved * s)
        .collect();
    (noise, g, solved)
}

fn rms(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = args.get(1).cloned().unwrap_or_else(|| "labels.txt".to_string());

    // one label per line
    let f = File::open(&path).expect("cannot open labels file");
    let y: Vec<f32> = BufReader::new(f)
        .lines()
        .filter_map(|l| l.ok())
        .filter_map(|l| l.trim().parse::<f32>().ok())
        .collect();
    let n = y.len();
    let mean = y.iter().sum::<f32>() / n as f32;
    let label_sd = (y.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32).sqrt();
    println!("labels: n={}  mean={:.4}  SD={:.4}\n", n, mean, label_sd);

    // fake scaffold groups: 2000 clusters, for the Grouped arm
    let mut grng = StdRng::seed_from_u64(7);
    let groups: Vec<usize> = (0..n).map(|_| grng.random_range(0..2000)).collect();

    let arms: Vec<(&str, NoiseArm)> = vec![
        ("A Gaussian", NoiseArm::Gaussian),
        ("B Student-t nu=10", NoiseArm::StudentT { nu: 10.0 }),
        ("B Student-t nu=5", NoiseArm::StudentT { nu: 5.0 }),
        ("B Student-t nu=3", NoiseArm::StudentT { nu: 3.0 }),
        ("C Laplace", NoiseArm::Laplace),
        ("D Grouped lam=3 f=0.2", NoiseArm::Grouped { lambda: 3.0, group_fraction: 0.2 }),
        ("E Outlier p=0.01 lam=3", NoiseArm::Outlier { p: 0.01, lambda: 3.0 }),
        ("E Outlier p=0.05 lam=3", NoiseArm::Outlier { p: 0.05, lambda: 3.0 }),
        ("E Outlier p=0.10 lam=3", NoiseArm::Outlier { p: 0.10, lambda: 3.0 }),
    ];

    for k in [0.25f32, 0.5, 1.0] {
        let target = k * label_sd;
        println!("=== k = {}  ->  target dose = {:.4} ===", k, target);
        for (name, arm) in &arms {
            let g_opt = match arm { NoiseArm::Grouped { .. } => Some(&groups[..]), _ => None };
            let (noise, g, solved) = generate(arm, n, k, label_sd, g_opt, 42);
            let realised = rms(&noise);
            let err = 100.0 * (realised / target - 1.0);
            // fraction beyond 3x the dose
            let f3 = noise.iter().filter(|e| e.abs() > 3.0 * target).count() as f32 / n as f32;
            println!(
                "  {:26} G={:.4} scale={:.4}  realised={:.4}  err={:+.2}%  >3x dose={:.2}%",
                name, g, solved, realised, err, f3 * 100.0
            );
        }
        println!();
    }
}
