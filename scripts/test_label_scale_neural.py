#!/usr/bin/env python3
"""Do the neural models fail on QM9 because of the model, or because of the label scale?

RERUN_PLAN.md 5.5i records that all three neural families fail to FIT on QM9
descriptors -- R2 from -0.03 to -4.8 against 0.84 for every tree model on
identical data -- and concludes that the case the literature holds up (a network
with a variance head, Kendall & Gal eq. 6) is untested here rather than
disproven.

That conclusion came from `scripts/decomposition_controls.py`, whose `_standardise`
scales X and NEVER TOUCHES y. The QM9 label is the HOMO-LUMO gap in Hartree:
mean 0.254, spread 0.047. The production pipeline standardises the label
(`--normalize True`; every results row carries standardisation_mean and
standardisation_sd).

Why that could matter enormously, and only for the neural models:

  * A random forest is invariant to any monotone rescaling of y, so it cannot
    care. NGBoost fits a distribution whose scale is a free parameter, so it
    adapts. Those are the models that scored 0.84.
  * A network trained by gradient descent at a fixed learning rate is NOT
    invariant. Worse, these networks use the heteroscedastic likelihood,
    0.5*(log_var + (y - mean)^2 / var). The residuals here are of order 0.01, so
    the log-variance that minimises it is near log(1e-4) = -9.2, and the head
    starts near 0, i.e. a variance 10,000x too large. That is a punishing place
    to start.

So the measured failure is consistent with two very different stories: the
models cannot do it, or the harness handed them an unscaled target. This tells
them apart by changing ONE thing.

Fit identical networks on identical molecules, with the label standardised and
not, and score both in the SAME units by converting back.
"""
import os
import sys
import numpy as np

REPO = '/Users/apunt/repos/qsar_qm_models'
sys.path.insert(0, os.path.join(REPO, 'scripts'))
sys.path.insert(0, os.path.join(REPO, 'models'))

import decomposition_controls as dc                      # noqa: E402
from sklearn.metrics import r2_score                     # noqa: E402

N_MOL = int(os.environ.get('N_MOL', 3000))
N_FOLDS = 3
SEED = 0
LEVELS = (0.0, 1.0)


def fit_scaled(x_fit, y_fit, x_score, seed=0, arch='dnn'):
    """`fit_bnn_mve`, with the label standardised on the FIT rows only.

    Everything else is the harness's own function, called unchanged. The mean
    comes back in the label's own units and both variance terms are multiplied
    by the square of the spread, which is the conversion the shared module's
    `to_raw_variance` performs -- a variance scales by the SQUARE of the spread,
    not by the spread.
    """
    mu = float(np.mean(y_fit))
    sd = float(np.std(y_fit))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError('degenerate label spread')
    mean_s, alea_s, epis_s = dc.fit_bnn_mve(
        x_fit, (np.asarray(y_fit, dtype=float) - mu) / sd, x_score,
        arch=arch, seed=seed)
    mean = np.asarray(mean_s, dtype=float) * sd + mu
    alea = None if alea_s is None else np.asarray(alea_s, dtype=float) * sd ** 2
    epis = None if epis_s is None else np.asarray(epis_s, dtype=float) * sd ** 2
    return mean, alea, epis


def main():
    x_desc, y, groups = dc.load_qm9(N_MOL, seed=SEED)
    smiles = dc._selected_smiles(N_MOL, seed=SEED)
    spread = float(np.std(y))
    x = np.asarray(dc.build_representations(smiles, ['PDV'])['PDV'], dtype=float)
    print(f"{len(y)} real QM9 molecules, label '{dc.LABEL}': "
          f"mean {y.mean():.4f}, spread {spread:.4f}")
    print(f"PDV, {x.shape[1]} descriptors, {N_FOLDS}-fold out of fold on scaffold groups")
    print("every number below is in the LABEL'S OWN UNITS\n")

    print(f"{'network':6}{'label':16}{'level':>7}{'R2 oof':>10}"
          f"{'aleatoric':>13}{'epistemic':>13}")
    print('-' * 65)
    out = {}
    for arch in ('dnn', 'mlp'):
        for label, fn in (('as measured', lambda a, b, c: dc.fit_bnn_mve(a, b, c, arch=arch, seed=SEED)),
                          ('standardised', lambda a, b, c: fit_scaled(a, b, c, seed=SEED, arch=arch))):
            for lvl in LEVELS:
                y_noisy = dc.injector_noise('gaussian', y, lvl, groups, spread, SEED)
                try:
                    mean, alea, epis, _f = dc._oof(fn, x, y_noisy, groups, N_FOLDS)
                except Exception as exc:
                    print(f"{arch:6}{label:16}{lvl:>7.1f}  BLOCKED {type(exc).__name__}: {exc}")
                    continue
                ok = np.isfinite(mean)
                r2 = float(r2_score(y[ok], mean[ok]))
                ma = float(np.nanmean(alea[ok])) if alea is not None else float('nan')
                me = float(np.nanmean(epis[ok])) if epis is not None else float('nan')
                out[(arch, label, lvl)] = (r2, ma, me)
                print(f"{arch:6}{label:16}{lvl:>7.1f}{r2:>10.4f}{ma:>13.4g}{me:>13.4g}")
            k0, k1 = (arch, label, LEVELS[0]), (arch, label, LEVELS[1])
            if k0 in out and k1 in out:
                a0, a1 = out[k0][1], out[k1][1]
                e0, e1 = out[k0][2], out[k1][2]
                print(f"{'':22}{'RATIO':>7}{'':>10}{a1 / a0:>12.1f}x{e1 / e0:>12.1f}x")
            print()

    print("\nHOW TO READ IT")
    print("  If 'standardised' turns a negative R2 into a positive one, the recorded")
    print("  neural failure is the harness handing the network an unscaled target,")
    print("  not the models being unable to do it -- and the decomposition verdict")
    print("  for those models has to be re-taken, not carried forward.")
    print("  If R2 stays negative, the models genuinely do not fit these features")
    print("  and the recorded conclusion stands as written.")


if __name__ == '__main__':
    main()
