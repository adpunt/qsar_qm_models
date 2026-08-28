#!/usr/bin/env python3
"""The noise-predicting Gaussian process must learn from FITTED residuals, not prior ones.

WHY THIS EXISTS
---------------
This model is the only one measured so far that separates a prediction's
uncertainty into "the label here is noisy" and "the model does not know here".
Both forest halves report the same signal -- +0.84 and +0.81 against the true
corruption -- so the forests cannot do it. This one can, and the paper's
aleatoric/epistemic split rests on it.

It works by fitting a small network alongside the process that predicts each
molecule's noise from its own features, trained on the squared residuals of the
process. The residual is where the whole thing lives, and there are two of them:

  * in gpytorch's TRAIN mode a model call returns the PRIOR at the training
    inputs, so `y - mean` is `y - a constant` -- the spread of the labels
    themselves, dominated by real structure;
  * in EVAL mode it returns the fitted posterior, so `y - mean` is what the
    process could not explain, which is the corruption.

Measured on 300 molecules whose loud half carries 100x the noise variance of
the quiet half:

    prior residuals    loud 6.44  quiet 5.92   ratio  1.09
    fitted residuals   loud 0.88  quiet 0.033  ratio 27.08

Trained on the prior residuals the network's output is FLAT. Its rank
correlation with the true corruption sat between -0.12 and +0.09 at every
learning rate from 0.001 to 0.05 and every length from 60 to 2,000 epochs, so it
is not a convergence problem and no amount of training fixes it. Trained on the
fitted residuals the same model reaches +0.53 to +0.60 with its accuracy
unchanged.

Both pipelines wrote it the first way. This check goes red if either goes back:
it MEASURES the laboratory model, which can be built here, and READS the QM9 one,
which needs the whole QM9 stack to run and would otherwise go unchecked on this
laptop. The two carry the identical model and must move together.

    python scripts/test_hetero_gp_learns_the_noise.py
"""
import os
import sys

import numpy as np

KIRBY = os.environ.get('KIRBY_DIR') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'KIRBy')
RUNNER = os.path.join(KIRBY, 'tests')
if not os.path.isdir(RUNNER):
    sys.exit(f"no KIRBy checkout at {KIRBY}. Set KIRBY_DIR.")
sys.path.insert(0, RUNNER)

# A loud half and a quiet half, told apart by a feature the model can see. The
# ratio is deliberately extreme: a model that cannot separate 100x apart will
# not separate anything a real assay does.
LOUD_SD, QUIET_SD = 1.0, 0.1
N, DIM = 600, 8
EPOCHS = 100

# Thresholds. The measured values are ~10x and ~0.60; these sit well below, so
# the check fails on the defect and not on a seed.
MIN_RATIO = 3.0
MIN_RHO = 0.30


def data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(N, DIM))
    y_clean = x[:, 0] * 2.0 + x[:, 1] - 0.5 * x[:, 2]
    loud = x[:, 3] > 0
    y = y_clean + rng.normal(scale=np.where(loud, LOUD_SD, QUIET_SD))
    train = np.arange(N) < N // 2
    return x, y, y_clean, loud, train, ~train


def main():
    import alternative_data_noise_robustness as runner
    if not getattr(runner, 'HAS_GP', False):
        sys.exit("gpytorch is not importable here, so this model cannot be built. "
                 "This check is not optional -- run it where the environment is whole.")

    from scipy.stats import spearmanr
    x, y, y_clean, loud, train, test = data()

    model = runner.HeteroscedasticGaussianProcessGauche(kernel='rbf', epochs=EPOCHS)
    model.fit(x[train], y[train])
    mean, std = model.predict(x[test], return_std=True)
    data_noise = model.aleatoric_var_
    model_noise = model.epistemic_var_

    failures = []

    # 1. It has to find which molecules are corrupted.
    ratio = data_noise[loud[test]].mean() / data_noise[~loud[test]].mean()
    rho = spearmanr(data_noise, np.abs(y[test] - y_clean[test])).statistic
    print(f"  data-noise, loud vs quiet   {ratio:6.2f}x   (needs >= {MIN_RATIO})")
    print(f"  rank correlation with truth {rho:6.3f}    (needs >= {MIN_RHO})")
    if not ratio >= MIN_RATIO:
        failures.append(
            f"the network gives loud and quiet molecules the same noise ({ratio:.2f}x). "
            f"It is almost certainly training on PRIOR residuals: call the process in "
            f"eval mode when computing them.")
    if not rho >= MIN_RHO:
        failures.append(
            f"the predicted noise does not rank the corrupted molecules (rho={rho:.3f}). "
            f"Same cause as above.")

    # 2. The two halves must be two different things, or the model buys nothing
    #    over the forests -- which report one signal twice.
    together = spearmanr(data_noise, model_noise).statistic
    print(f"  the two halves against each other {together:6.3f}  (must not be ~1)")
    if together > 0.95:
        failures.append(
            f"the data half and the model half move together (rho={together:.3f}), so "
            f"this model separates no more than the forests do.")

    # 3. Everything is a variance until the last step.
    if not np.allclose(std ** 2, data_noise + model_noise):
        failures.append(
            "the reported spread is not the square root of the two variances added. "
            "Two spreads were probably added instead.")

    # 4. It must not have bought this by predicting worse.
    r2 = 1 - ((y[test] - mean) ** 2).sum() / ((y[test] - y[test].mean()) ** 2).sum()
    print(f"  R2 on held-out molecules    {r2:6.3f}    (needs >= 0.5)")
    if r2 < 0.5:
        failures.append(f"accuracy collapsed (R2={r2:.3f}); the split is not worth having.")

    # 5. The QM9 twin must not have drifted back. It is the same model; running
    #    it here needs the whole QM9 stack, so this reads it instead of guessing.
    qm9 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'models', 'models.py')
    src = open(qm9).read()
    if 'def train_heteroscedastic_gp' not in src:
        failures.append(f"{qm9} has no train_heteroscedastic_gp to check")
    else:
        body = src[src.index('def train_heteroscedastic_gp'):]
        body = body[:body.index('\ndef ', 1)] if '\ndef ' in body[1:] else body
        if 'gp_pred = gp(xf).mean' in body and 'gp.eval()' not in body:
            failures.append(
                "the QM9 model computes its residuals with the process in train mode, so "
                "it is learning from PRIOR residuals and its predicted noise will be flat. "
                "Same one-line fix as the laboratory side: gp.eval() around the residual.")
        else:
            print("  the QM9 twin uses fitted residuals too")

    if failures:
        print(f"\nFAIL — {len(failures)} problem(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nOK: the noise-predicting Gaussian process finds which labels are corrupted, "
          "and its two halves are different quantities")
    return 0


if __name__ == '__main__':
    sys.exit(main())
