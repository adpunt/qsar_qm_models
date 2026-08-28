"""Test the five proposed noise arms: does dose-matching actually deliver?

Run against the real QM9 HOMO-LUMO gap labels and the implied LogD distribution.
Everything here is a check, not an assumption: for each arm we compute the
predicted scale from the algebra, draw the noise, and measure what we got.

The tables come first. The last section is an assertion rather than a table, and
it is the only place the two REAL injectors -- the Rust one that noises QM9 and
the Python one that noises the three experimental sets -- are compared on the
shifted grouped condition. Exits non-zero if they disagree.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(20260824)

# ---------------------------------------------------------------- labels
qm9 = pd.read_csv(os.path.join(REPO, 'data/QM9/raw/gdb9.sdf.csv'))
y_qm9 = (qm9['gap'] * 27.211386).values          # eV
SD_qm9 = y_qm9.std()

print(f"QM9 gap: n={len(y_qm9)}  mean={y_qm9.mean():.4f} eV  SD={SD_qm9:.4f} eV")

# ---------------------------------------------------------------- the arms
# Each arm returns (noise_array, name). Each is dose-matched to `target`.

def arm_gaussian(y, target, rng):
    return rng.normal(0.0, target, len(y))

def arm_student_t(y, target, nu, rng):
    s = target * np.sqrt((nu - 2.0) / nu)         # Var = s^2 * nu/(nu-2)
    return rng.standard_t(nu, len(y)) * s

def arm_grouped(y, target, lam, f, groups, rng):
    """Whole groups get a wider error. `groups` = integer group id per molecule."""
    gids = np.unique(groups)
    n_bad = max(1, int(round(f * len(gids))))
    bad = set(rng.choice(gids, size=n_bad, replace=False).tolist())
    is_bad = np.array([g in bad for g in groups])
    f_real = is_bad.mean()                        # fraction of MOLECULES, not groups
    s_low = target / np.sqrt(1 - f_real + f_real * lam**2)
    scale = np.where(is_bad, lam * s_low, s_low)
    return rng.normal(0.0, 1.0, len(y)) * scale, f_real

def arm_contaminated(y, target, p, lam, rng):
    s0 = target / np.sqrt(1 + p * (lam**2 - 1))
    hit = rng.random(len(y)) < p
    scale = np.where(hit, lam * s0, s0)
    return rng.normal(0.0, 1.0, len(y)) * scale, hit

def arm_contaminated_replace(y, target, p, rng):
    """Contaminated labels replaced by a draw from the observed label range.
    Dose is NOT free here - it is set by p and the label spread. We report it."""
    hit = rng.random(len(y)) < p
    lo, hi = y.min(), y.max()
    repl = rng.uniform(lo, hi, len(y))
    noise = np.where(hit, repl - y, 0.0)
    return noise, hit

def arm_censor(y, p_hi):
    """Top p_hi fraction clipped to the threshold. Not zero-mean, not dose-matched."""
    c = np.quantile(y, 1 - p_hi)
    ynew = np.minimum(y, c)
    return ynew - y

# ---------------------------------------------------------------- diagnostics
def describe(noise, target, label, extra=""):
    d = np.sqrt(np.mean(noise**2))                # realised dose
    sd = noise.std()
    # share of total noise energy carried by the worst-hit 5% of molecules
    e = np.sort(noise**2)[::-1]
    share5 = e[:max(1, int(.05*len(e)))].sum() / e.sum()
    # fraction beyond 3x the dose
    frac3 = np.mean(np.abs(noise) > 3*target)
    try:
        kurt = stats.kurtosis(noise, fisher=True)
    except Exception:
        kurt = np.nan
    hit = "" if not extra else f"  {extra}"
    print(f"  {label:34s} dose={d:.4f} (target {target:.4f}, err {100*(d/target-1):+5.1f}%) "
          f"SD={sd:.4f}  >3x dose={frac3*100:5.2f}%  top5%share={share5*100:5.1f}%  exkurt={kurt:8.2f}{hit}")
    return d

# ---------------------------------------------------------------- run
for k in [0.2, 0.5]:
    target = k * SD_qm9
    print(f"\n{'='*118}\nQM9, target dose = {k} x label SD = {target:.4f} eV\n{'='*118}")

    describe(arm_gaussian(y_qm9, target, rng), target, "A. Gaussian")

    for nu in [30, 10, 5, 3]:
        describe(arm_student_t(y_qm9, target, nu, rng), target, f"B. Student-t  nu={nu}")

    # grouped: no scaffold clusters locally, so simulate ~2000 clusters of varying size
    ngroups = 2000
    sizes = rng.dirichlet(np.ones(ngroups)*0.6)
    groups = rng.choice(ngroups, size=len(y_qm9), p=sizes)
    for lam, f in [(3.0, 0.2)]:
        n, f_real = arm_grouped(y_qm9, target, lam, f, groups, rng)
        describe(n, target, f"C. Grouped  lam={lam} f_groups={f}", f"(f_molecules={f_real:.3f})")

    for p in [0.01, 0.05, 0.10]:
        n, hit = arm_contaminated(y_qm9, target, p, 3.0, rng)
        describe(n, target, f"D. Contaminated p={p} lam=3", f"(hit {hit.mean()*100:.1f}%)")

    for p in [0.01, 0.05]:
        n, hit = arm_contaminated_replace(y_qm9, target, p, rng)
        describe(n, target, f"D'. Replace-with-random p={p}", "(dose NOT matchable)")

print(f"\n{'='*118}\nE. Censoring - not zero-mean, cannot be dose-matched\n{'='*118}")
for p in [0.10, 0.25, 0.40]:
    n = arm_censor(y_qm9, p)
    print(f"  censor top {p*100:4.1f}%:  mean shift={n.mean():+.4f} eV  "
          f"rms={np.sqrt(np.mean(n**2)):.4f} eV  = {np.sqrt(np.mean(n**2))/SD_qm9:.3f} x label SD  "
          f"(nonzero for {np.mean(n!=0)*100:.1f}% of molecules)")

# ---------------------------------------------------------------- can the arms be told apart?
print(f"\n{'='*118}\nAre the arms actually DIFFERENT at matched dose? (QM9, k=0.5)\n{'='*118}")
target = 0.5 * SD_qm9
rows = []
rows.append(("A. Gaussian", arm_gaussian(y_qm9, target, rng)))
for nu in [10, 5, 3]:
    rows.append((f"B. Student-t nu={nu}", arm_student_t(y_qm9, target, nu, rng)))
n, _ = arm_grouped(y_qm9, target, 3.0, 0.2, groups, rng); rows.append(("C. Grouped lam=3 f=0.2", n))
for p in [0.01, 0.05, 0.10]:
    n, _ = arm_contaminated(y_qm9, target, p, 3.0, rng); rows.append((f"D. Contaminated p={p}", n))

print(f"  {'arm':26s}{'>1x dose':>10s}{'>2x':>8s}{'>3x':>8s}{'>5x':>8s}{'median|e|':>11s}{'top5% share':>13s}")
for lab, n in rows:
    a = np.abs(n)
    print(f"  {lab:26s}{np.mean(a>target)*100:9.2f}%{np.mean(a>2*target)*100:7.2f}%"
          f"{np.mean(a>3*target)*100:7.2f}%{np.mean(a>5*target)*100:7.2f}%"
          f"{np.median(a):11.4f}{(np.sort(n**2)[::-1][:int(.05*len(n))].sum()/np.sum(n**2))*100:12.1f}%")


# ---------------------------------------------------------------- the two injectors
#
# THE SHIFTED GROUPED CONDITION, ACROSS SHAPES. Nothing compared the two
# implementations here before, and both were wrong in opposite directions:
#
#   Python multiplied the two components by the TARGET while drawing them at the
#   shape's own spread, so it delivered the target TIMES that spread -- 1.51x
#   under Laplace, 1.19x under Student-t at nu=5.
#   Rust divided the components down to unit variance while the solver still put
#   the shape's spread into G, so it delivered the target DIVIDED by it -- 0.71x
#   and 0.77x.
#
# A Gaussian draw has spread 1, and the condition roster is Gaussian at every
# entry, so the two agreed exactly everywhere anything looked. The shape has to
# be the second axis or this is invisible (RERUN_PLAN.md section 2.14).
#
# The reference implementation cannot take part: `rust/reference/noise_arms.rs`
# fuses the shape and the targeting into one enum, so its shifted grouped arm has
# no shape to set. This compares the pipeline injector against the Python one.

sys.path.insert(0, os.path.expanduser('~/repos/NoiseInject'))
from noiseInject import NoiseInjectorRegression, dose_tolerance   # noqa: E402

PIPELINE_BIN = os.path.join(REPO, 'rust', 'target', 'release', 'rust_processor')
CROSS_N = 4000
CROSS_GROUPS = 200
CROSS_K = 0.5
CROSS_SEEDS = 20
RHO = 0.62
SHAPES = [('gaussian', None), ('laplace', None), ('student_t', 5.0)]


def _cross_labels():
    """A skewed, strictly positive column with evenly sized groups.

    Even groups on purpose: uneven ones are the realistic case and are covered
    by the QM9 gates, but they widen the band this check has to allow, and the
    thing being measured here is a factor of 1.41, not a sampling wobble.
    """
    r = np.random.RandomState(20260827)
    y = np.abs(r.normal(6.8, 1.29, CROSS_N)) + 0.5
    g = np.repeat(np.arange(CROSS_GROUPS), CROSS_N // CROSS_GROUPS)
    return y, g


def _rust_delivered(tmp, shape, nu, y, g):
    """The pipeline injector's delivered dose, averaged over the same seed count.

    `--noise-shape` with `--noise-targeting` runs that ONE pair instead of the
    roster, which is Gaussian throughout and cannot express this.
    """
    labels = os.path.join(tmp, 'labels.csv')
    groups = os.path.join(tmp, 'groups.json')
    smiles = [f'M{i}' for i in range(len(y))]
    with open(labels, 'w') as f:
        for smi, v in zip(smiles, y):
            f.write(f'{smi},{v}\n')
    json.dump({smi: int(gg) for smi, gg in zip(smiles, g)}, open(groups, 'w'))

    cmd = [PIPELINE_BIN, '--self-test', labels, '--json',
           '--seeds', str(CROSS_SEEDS), '--noise-level', str(CROSS_K),
           '--scaffold-file', groups,
           '--noise-shape', shape, '--noise-targeting', 'grouped_shift',
           '--group-variance-share', str(RHO)]
    if nu is not None:
        cmd += ['--nu', str(nu)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f'the pipeline injector failed:\n{out.stderr}')
    return json.loads(out.stdout)['rows'][0]


def _python_delivered(shape, nu, y, g, target):
    kw = {'nu': nu} if nu is not None else {}
    runs = []
    for seed in range(CROSS_SEEDS):
        inj = NoiseInjectorRegression(strategy='grouped_shifted', distribution=shape,
                                      random_state=42 + seed, rho=RHO, **kw)
        runs.append(inj.inject_verbose(y, target, groups=g))
    return (float(np.mean([r.realised_dose_label_units for r in runs])),
            dose_tolerance(runs[0].epsilon, runs[0].effective_n, nu=nu),
            runs[0])


def cross_check_grouped_shifted():
    build = subprocess.run(['cargo', 'build', '--release'],
                           cwd=os.path.join(REPO, 'rust'),
                           capture_output=True, text=True)
    if build.returncode != 0:
        sys.exit('cargo build failed:\n' + build.stderr)

    y, g = _cross_labels()
    target = CROSS_K * y.std()
    failures = []
    print(f"\n{'='*118}")
    print('The two injectors on grouped_shifted, one row per shape '
          f'(n={CROSS_N}, {CROSS_GROUPS} groups, target {target:.4f}, '
          f'{CROSS_SEEDS} seeds each)')
    print('='*118)
    print(f"  {'shape':16s}{'target':>10s}{'rust':>10s}{'python':>10s}"
          f"{'rust err':>11s}{'py err':>10s}{'band':>9s}")

    with tempfile.TemporaryDirectory() as tmp:
        for shape, nu in SHAPES:
            rust = _rust_delivered(tmp, shape, nu, y, g)
            py_dose, py_tol, py_run = _python_delivered(shape, nu, y, g, target)

            # The band each side's own tolerance function allows, shrunk by the
            # number of seeds averaged over. Derived per condition from the draw,
            # exactly as the QM9 gates derive it -- not a hand-kept number.
            band = max(rust['dose_tolerance'], py_tol) / np.sqrt(CROSS_SEEDS)
            rust_err = rust['delivered_dose'] / target - 1.0
            py_err = py_dose / target - 1.0
            name = shape if nu is None else f'{shape} nu={nu:g}'
            print(f"  {name:16s}{target:10.4f}{rust['delivered_dose']:10.4f}"
                  f"{py_dose:10.4f}{100*rust_err:+10.2f}%{100*py_err:+9.2f}%"
                  f"{100*band:8.2f}%")

            # Each side delivers what was asked ...
            if abs(rust_err) > band:
                failures.append(f'{name}: the Rust injector delivered '
                                f'{rust["delivered_dose"]:.4f} against a target of '
                                f'{target:.4f} ({100*rust_err:+.1f}%)')
            if abs(py_err) > band:
                failures.append(f'{name}: the Python injector delivered '
                                f'{py_dose:.4f} against a target of {target:.4f} '
                                f'({100*py_err:+.1f}%)')
            # ... and they agree with each other, which is the point: a shared
            # error would pass the two checks above and still be an error.
            if abs(rust['delivered_dose'] - py_dose) > 2 * band * target:
                failures.append(f'{name}: Rust delivered {rust["delivered_dose"]:.4f}, '
                                f'Python {py_dose:.4f}')
            # The construction, which is arithmetic on both sides rather than a
            # draw, so it must agree far more tightly than the delivered amount.
            if abs(rust['unit_dose'] - py_run.unit_dose_g) > 1e-4:
                failures.append(f'{name}: unit dose G is {rust["unit_dose"]:.6f} in '
                                f'Rust and {py_run.unit_dose_g:.6f} in Python')

    if failures:
        print('\nFAILED:')
        for f in failures:
            print('  ' + f)
        return 1
    print('\n  both injectors deliver the requested amount under all three shapes')
    return 0


sys.exit(cross_check_grouped_shifted())
