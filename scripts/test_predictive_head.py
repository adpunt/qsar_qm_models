#!/usr/bin/env python
"""The network's own predicted variance reaches the results, and is the fitted one.

Two losses make the network output how uncertain it is about each molecule
alongside the prediction. Every prediction site sliced that second output off and
kept the first, so both models reported the spread over their stochastic passes
-- what an ordinary network reports -- and the per-molecule variance they exist
to produce reached no file (RERUN_PLAN.md 2.17).

Two things have to hold, and only the second of them is about plumbing:

  1. The variance read back is the one that was FITTED. The transforms live in
     `scripts/loss_functions.py`; this asserts against those classes rather than
     against a copy of their algebra, so changing the loss and not the reader
     fails here.
  2. A network with such a head produces an aleatoric term that DIFFERS between
     molecules. Restore the slice and it is None, which is what this catches.

    python3 scripts/test_predictive_head.py
"""
import os
import sys
import traceback

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'models'))

from utils import (split_predictive_head,                       # noqa: E402
                   decompose_uncertainty_sampling,
                   decompose_uncertainty_sampling_heteroscedastic)
from loss_functions import HeteroscedasticLoss, EvidentialLoss  # noqa: E402


def the_variance_is_the_one_the_loss_fitted():
    """Read the variance back through the loss's own definition, not a copy."""
    rng = np.random.default_rng(0)

    # Heteroscedastic. The loss is 0.5 * (log_var + (y - mean)^2 / var), so at a
    # fixed mean it is smallest when var equals the squared error. Build outputs
    # whose second column IS that minimiser and check the reader returns it.
    n = 64
    y = rng.normal(0.0, 1.0, (n, 1))
    mean = y + rng.normal(0.0, 0.5, (n, 1))
    true_var = np.maximum((y - mean) ** 2, 1e-6)
    out = np.hstack([mean, np.log(true_var)])
    got_mean, got_var = split_predictive_head(out, 'heteroscedastic')
    assert np.allclose(got_mean, mean), "the prediction is not the first output"
    assert np.allclose(got_var, true_var, rtol=1e-6), (
        "the variance read back is not exp(log_var), which is what the loss fitted")

    # And the loss really is minimised there: perturbing the fitted variance in
    # either direction must not lower it.
    t_out = torch.tensor(out, dtype=torch.float64)
    t_y = torch.tensor(y, dtype=torch.float64)
    loss = HeteroscedasticLoss()
    at_fit = loss(t_out, t_y).item()
    for shift in (-0.5, 0.5):
        moved = t_out.clone()
        moved[:, 1] += shift
        assert loss(moved, t_y).item() > at_fit, (
            f"the heteroscedastic loss is not minimised at the variance this "
            f"reader returns (shift {shift})")

    # Evidential. Four outputs of a Normal-Inverse-Gamma, softplus on the last
    # three, +1 on v and alpha; its observation variance is beta / (alpha - 1).
    raw = rng.normal(0.0, 1.0, (n, 4))
    got_mean, got_var = split_predictive_head(raw, 'evidential')
    assert np.allclose(got_mean, raw[:, 0:1]), "the prediction is not gamma"
    sp = torch.nn.functional.softplus
    alpha = (sp(torch.tensor(raw[:, 2:3])) + 1.0).numpy()
    beta = sp(torch.tensor(raw[:, 3:4])).numpy()
    assert np.allclose(got_var, beta / (alpha - 1.0), rtol=1e-6), (
        "the evidential variance is not beta / (alpha - 1) under the loss's own "
        "softplus transforms")
    # The loss must accept the same four columns it was handed.
    EvidentialLoss()(torch.tensor(raw, dtype=torch.float64),
                     torch.tensor(y, dtype=torch.float64))

    # Every other loss is passed through untouched, one column in, one out.
    plain = rng.normal(0.0, 1.0, (n, 1))
    got, var = split_predictive_head(plain, 'mse')
    assert var is None and np.allclose(got, plain), (
        "a loss with no variance head must come back unchanged, with no variance")

    print(f"    {n} molecules, both heads read back through their own loss")


def the_aleatoric_term_differs_between_molecules():
    """The whole point: a per-molecule variance, not one number for the run.

    This is what the slice destroyed. With the second output dropped, the
    stochastic passes are all the decomposition has, aleatoric comes back None,
    and the total is the model term alone.
    """
    rng = np.random.default_rng(1)
    passes, n = 20, 200

    # A network that is confident about some molecules and not others: the
    # predicted variance spans two orders of magnitude across the test set.
    per_molecule_var = np.exp(rng.normal(0.0, 1.0, n))
    means = rng.normal(0.0, 0.1, (passes, n))
    variances = per_molecule_var[None, :] * np.exp(rng.normal(0.0, 0.05, (passes, n)))

    epi, ale, total = decompose_uncertainty_sampling_heteroscedastic(means, variances)
    assert ale is not None, "the head predicted a variance and it was not reported"
    assert ale.shape == (n,), f"aleatoric is not per molecule: shape {ale.shape}"
    spread = float(np.std(ale) / np.mean(ale))
    assert spread > 0.1, (
        f"the aleatoric term is the same for every molecule (relative spread "
        f"{spread:.4f}) -- it is not carrying what the head predicted")
    assert np.all(total >= epi - 1e-12), (
        "the total predictive spread is smaller than its model part")
    assert np.allclose(total, np.sqrt(epi ** 2 + ale ** 2)), (
        "the total is not the two terms added in quadrature")

    # And the contrast: dropping the variance, which is what the code did, gives
    # no aleatoric term at all and a total equal to the model part.
    epi_only, ale_only, total_only = decompose_uncertainty_sampling(means, passes)
    assert ale_only is None and np.allclose(total_only, epi_only), (
        "the sampling-only decomposition is meant to report no observation term")
    assert not np.allclose(total_only, total), (
        "keeping the predicted variance changed nothing, so the two paths are "
        "reporting the same quantity and the fix is not doing anything")

    print(f"    {n} molecules, aleatoric spread {spread:.2f} of its own mean")


def the_wiring_carries_it_to_the_file():
    """Train the real network and read the aleatoric column back off disk.

    The two checks above are about the arithmetic. This one is about the
    plumbing, and it is the one that goes red if the slice comes back: with the
    second output dropped, `aleatoric_uncertainty` is written blank.

    Small on purpose -- 120 synthetic molecules, six features, a handful of
    epochs. Nothing here is a result; it is the same code path a QM9 job takes.
    """
    import csv
    import tempfile

    sys.path.insert(0, os.path.join(HERE, '..', 'models'))
    import models as M

    class Args:
        """Everything the driver sets. Anything not named here is None, which is
        the off state for every one of them."""
        def __init__(self, **kw):
            self.__dict__.update(kw)

        def __getattr__(self, name):
            return None

    rng = np.random.default_rng(0)
    n, n_features = 120, 6
    x = rng.normal(size=(n, n_features)).astype(np.float32)
    y = (x[:, 0] * 2.0 + rng.normal(scale=0.3, size=n)).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        results = os.path.join(tmp, 'r.csv')
        args = Args(loss='heteroscedastic', tuning=False, k_domains=1,
                    bayesian_transformation='full', uncertainty=True,
                    epochs=3, patience=2, sample_size=n,
                    use_best_params=False, oof_folds=0, filepath=results)
        torch.manual_seed(0)
        M.train_dnn_model(x[:60], y[:60], x[60:90], y[60:90], x[90:], y[90:],
                          args, 0.0, 'ecfp4', 0, 42, 1, y[60:90])

        written = results.replace('.csv', '_uncertainty_values.csv')
        assert os.path.exists(written), (
            f"the run wrote no per-molecule uncertainty at all ({written})")
        with open(written) as f:
            rows = list(csv.DictReader(f))

    assert rows, "the uncertainty file is empty"
    assert 'aleatoric_uncertainty' in rows[0], (
        f"no aleatoric column was written; got {sorted(rows[0])}")
    values = [r['aleatoric_uncertainty'] for r in rows]
    blank = [v for v in values if v in ('', 'None')]
    assert not blank, (
        f"{len(blank)} of {len(values)} molecules have no aleatoric term. The "
        f"head predicted one and it did not reach the file -- the wide output is "
        f"being sliced before the decomposition sees it.")
    numbers = np.array([float(v) for v in values])
    assert np.std(numbers) > 0, (
        "every molecule got the same aleatoric term, so it is not the "
        "per-molecule variance the head predicted")
    print(f"    {len(numbers)} molecules, aleatoric written for every one, "
          f"spread {np.std(numbers) / np.mean(numbers):.2f} of its mean")


def check(name, fn):
    print(f"  {name}")
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        print(f"    FAIL: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return False
    print("    ok")
    return True


def main():
    print("the predicted variance reaches the results (RERUN_PLAN.md 2.17)")
    results = [
        check("the variance read back is the one the loss fitted",
              the_variance_is_the_one_the_loss_fitted),
        check("the aleatoric term differs between molecules",
              the_aleatoric_term_differs_between_molecules),
        check("a real run writes it for every molecule",
              the_wiring_carries_it_to_the_file),
    ]
    if not all(results):
        print("\nFAIL: the network's predicted variance is not being reported")
        return 1
    print("\nOK: the head's predicted variance is read back as fitted, per molecule")
    return 0


if __name__ == "__main__":
    sys.exit(main())
