#!/usr/bin/env python3
"""Does NGBoost's seed ensemble give a real epistemic term, and does it behave?

Same protocol as the measurement that produced the level-response table: real
QM9, PDV, out-of-fold on Murcko scaffold groups, plain gaussian label noise at
level 0 and level 1. The ONLY thing varied here is `minibatch_frac` -- the
fraction of rows each boosting stage sees.

Why that parameter and no other. NGBoost's `random_state` feeds exactly two
things: which rows a stage sees (`minibatch_frac`) and which columns it sees
(`col_sample`). The shared spec sets both to 1.0, so every seed sees every row
and every column, every fit is the same fit, and the variance across their means
-- which IS the epistemic term for this model -- is floating-point dust.

Nothing else about the model changes. Member 0 is fitted the same way at every
setting, so its aleatoric term and its accuracy are directly comparable down the
column.

The test a real epistemic term has to pass is the one the study already uses:
the ALEATORIC term must rise with the noise level, because there is more
observation noise; the EPISTEMIC term must NOT, because neither the amount of
data nor the model class has changed. A term that is dust passes that test
vacuously, which is the whole problem.
"""
import os
import sys
import numpy as np

REPO = '/Users/apunt/repos/qsar_qm_models'
sys.path.insert(0, os.path.join(REPO, 'scripts'))
sys.path.insert(0, os.path.join(REPO, 'models'))

import decomposition_controls as dc              # noqa: E402
from uncertainty_decomposition import (          # noqa: E402
    decompose_seed_ensemble, assert_matches_support, DecompositionError)
from model_defaults import SKLEARN_DEFAULTS      # noqa: E402
from sklearn.metrics import r2_score             # noqa: E402
from sklearn.tree import DecisionTreeRegressor   # noqa: E402
from ngboost import NGBRegressor                 # noqa: E402
from ngboost.distns import Normal                # noqa: E402
from ngboost.scores import MLE                   # noqa: E402

N_MOL = int(os.environ.get('N_MOL', 3000))
N_FOLDS = 3
SEED = 0
LEVELS = (0.0, 1.0)
N_SEEDS = int(SKLEARN_DEFAULTS.get('ngboost_ensemble_seeds', 3))
P = dict(SKLEARN_DEFAULTS['ngboost'])


def make_ensemble_fitter(minibatch_frac, col_sample):
    """The production ensemble, with the two sampling knobs exposed.

    Every member gets the SAME base learner the shared spec names -- depth 3,
    friedman_mse -- which the production code does not do today: it passes the
    spec's tree to member 0 and lets members 1 and 2 fall through to ngboost's
    own default. That difference is fixed here so the comparison is about
    sampling and nothing else.
    """
    def fit(x_fit, y_fit, x_score, seed=0):
        means, variances = [], []
        for k in range(N_SEEDS):
            m = NGBRegressor(
                Dist=Normal, Score=MLE,
                Base=DecisionTreeRegressor(
                    max_depth=P['base_max_depth'],
                    criterion=P['base_criterion'],
                    random_state=seed * 100 + k),
                natural_gradient=P['natural_gradient'],
                n_estimators=P['n_estimators'],
                learning_rate=P['learning_rate'],
                minibatch_frac=minibatch_frac,
                col_sample=col_sample,
                verbose=False,
                random_state=seed * 100 + k)
            m.fit(x_fit, y_fit)
            d = m.pred_dist(x_score)
            means.append(np.asarray(d.loc, dtype=float))
            variances.append(np.asarray(d.scale, dtype=float) ** 2)
        mean_by_seed = np.stack(means, axis=0)
        var_by_seed = np.stack(variances, axis=0)
        alea, epis, _ = decompose_seed_ensemble(mean_by_seed, var_by_seed)
        return mean_by_seed.mean(axis=0), alea, epis
    return fit


def main():
    x_desc, y, groups = dc.load_qm9(N_MOL, seed=SEED)
    smiles = dc._selected_smiles(N_MOL, seed=SEED)
    spread = float(np.std(y))
    print(f"{len(y)} real QM9 molecules, label '{dc.LABEL}', spread {spread:.4f}")
    x = np.asarray(dc.build_representations(smiles, ['PDV'])['PDV'], dtype=float)
    print(f"PDV: {x.shape[1]} descriptors, {len(np.unique(groups))} scaffold groups")
    print(f"{N_SEEDS} seeds, {N_FOLDS}-fold out-of-fold on scaffold groups, "
          f"{P['n_estimators']} stages\n")

    settings = [
        ('as shipped', 1.0, 1.0),
        ('rows 0.5', 0.5, 1.0),
        ('rows 0.8', 0.8, 1.0),
        ('cols 0.8', 1.0, 0.8),
    ]
    print(f"{'setting':14}{'level':>7}{'R2 oof':>9}{'aleatoric':>13}{'epistemic':>13}"
          f"{'alea/epis':>12}{'guard':>9}")
    print('-' * 78)
    out = {}
    for label, mb, cs in settings:
        fit = make_ensemble_fitter(mb, cs)
        got = {}
        for lvl in LEVELS:
            y_noisy = dc.injector_noise('gaussian', y, lvl, groups, spread, SEED)
            mean, alea, epis, _fold = dc._oof(
                lambda a, b, c: fit(a, b, c, SEED), x, y_noisy, groups, N_FOLDS)
            ok = np.isfinite(mean)
            r2 = float(r2_score(y[ok], mean[ok]))
            ma = float(np.nanmean(alea[ok]))
            me = float(np.nanmean(epis[ok]))
            # The guard the pipeline actually applies before writing a row.
            try:
                assert_matches_support('ngboost', alea[ok], epis[ok],
                                       n_molecules=int(ok.sum()))
                guard = 'writes'
            except DecompositionError:
                guard = 'REFUSES'
            got[lvl] = (r2, ma, me)
            print(f"{label:14}{lvl:>7.1f}{r2:>9.4f}{ma:>13.4g}{me:>13.4g}"
                  f"{ma / me if me else float('inf'):>12.4g}{guard:>9}")
        out[label] = got
        a0, a1 = got[LEVELS[0]][1], got[LEVELS[1]][1]
        e0, e1 = got[LEVELS[0]][2], got[LEVELS[1]][2]
        print(f"{'':14}{'RATIO':>7}{'':>9}{a1 / a0:>12.1f}x{e1 / e0:>12.1f}x")
        print()

    print("\nWHAT TO READ")
    print("  aleatoric must RISE with the level. epistemic must NOT.")
    print("  alea/epis is how many times bigger the aleatoric term is. A ratio in")
    print("  the thousands means the epistemic term is not a quantity, it is dust.")


if __name__ == '__main__':
    main()
