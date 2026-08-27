"""Test the five proposed noise arms: does dose-matching actually deliver?

Run against the real QM9 HOMO-LUMO gap labels and the implied LogD distribution.
Everything here is a check, not an assumption: for each arm we compute the
predicted scale from the algebra, draw the noise, and measure what we got.
"""
import numpy as np
import pandas as pd
from scipy import stats

rng = np.random.default_rng(20260824)

# ---------------------------------------------------------------- labels
qm9 = pd.read_csv('data/QM9/raw/gdb9.sdf.csv')
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
