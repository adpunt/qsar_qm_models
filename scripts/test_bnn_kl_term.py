#!/usr/bin/env python
"""BNN-alpha and BNN-beta are fitted on the ELBO, not on plain MSE.

Until 2026-08-27 there was no KL, ELBO or BKLLoss anywhere in either pipeline
(RERUN_PLAN.md §2.12). torchbnn samples weights through the reparameterisation
trick in train AND eval mode, so `weight_log_sigma` received MSE gradients --
and MSE is minimised by making the sampling deterministic. Nothing pulled the
posterior toward the prior; `prior_sigma` was only an initialisation. What was
reported as epistemic uncertainty was the residual weight noise the fit happened
to leave, and it was compared against the VBLL variants, which carried a KL term
all along.

Three properties, each run rather than read:

  1. the criterion a BNN is fitted with is strictly larger than the same loss
     without the KL term, and by the weight the shared spec says;
  2. training on plain MSE collapses the posterior width; training on the ELBO
     does not -- measured on the same data with the same seed;
  3. both pipelines take the weight from the same spec function.

Run it directly:  python scripts/test_bnn_kl_term.py
"""

import os
import sys
import traceback

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'models'))

import torchbnn as bnn  # noqa: E402

from model_defaults import BAYESIAN_DEFAULTS, bnn_kl_weight  # noqa: E402
from models import (apply_bayesian_transformation,  # noqa: E402
                    bnn_elbo_criterion)


def small_net(width=8):
    return nn.Sequential(nn.Linear(4, width), nn.ReLU(), nn.Linear(width, 1))


def posterior_width(model):
    """Mean sampling standard deviation over every Bayesian weight."""
    sigmas = [torch.exp(m.weight_log_sigma).mean().item()
              for m in model.modules() if hasattr(m, 'weight_log_sigma')]
    assert sigmas, "the transformed model has no Bayesian layers"
    return float(np.mean(sigmas))


def the_criterion_carries_the_kl_term():
    torch.manual_seed(0)
    model = apply_bayesian_transformation(small_net())
    n_train = 500
    base = nn.MSELoss()
    wrapped = bnn_elbo_criterion(base, model, n_train)

    x = torch.randn(16, 4)
    y = torch.randn(16, 1)
    torch.manual_seed(1)
    out = model(x)
    plain = base(out, y).item()
    total = wrapped(out, y).item()

    kl = bnn.BKLLoss(reduction='mean', last_layer_only=False)(model).item()
    expected = plain + bnn_kl_weight(n_train) * kl
    # float32: compare to single precision, not to double.
    assert abs(total - expected) <= 1e-6 * max(1.0, abs(expected)), (total, expected)
    assert total > plain, (total, plain)
    print(f"    MSE {plain:.6f} -> ELBO {total:.6f} "
          f"(KL {kl:.4f} x weight {bnn_kl_weight(n_train):.3e})")


def plain_mse_collapses_the_posterior_and_the_elbo_does_not():
    """The CONTROL is the objective this replaced."""
    rng = np.random.default_rng(0)
    X = torch.tensor(rng.normal(size=(400, 4)), dtype=torch.float32)
    Y = torch.tensor(rng.normal(size=(400, 1)), dtype=torch.float32)

    widths = {}
    for label in ('mse', 'elbo'):
        torch.manual_seed(7)
        model = apply_bayesian_transformation(small_net())
        start = posterior_width(model)
        criterion = (nn.MSELoss() if label == 'mse'
                     else bnn_elbo_criterion(nn.MSELoss(), model, len(X)))
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(300):
            opt.zero_grad()
            criterion(model(X), Y).backward()
            opt.step()
        widths[label] = (start, posterior_width(model))

    (s0, mse_end) = widths['mse']
    (_, elbo_end) = widths['elbo']
    print(f"    posterior width: start {s0:.5f}; after 300 steps on MSE "
          f"{mse_end:.5f}, on the ELBO {elbo_end:.5f}")
    assert mse_end < s0, (
        f"plain MSE did not shrink the posterior here ({s0:.5f} -> "
        f"{mse_end:.5f}), so this check cannot tell the two objectives apart")
    assert elbo_end > mse_end, (
        f"the ELBO left the posterior at {elbo_end:.5f}, no wider than plain "
        f"MSE's {mse_end:.5f} -- the KL term is not reaching the optimiser")


def both_pipelines_take_the_weight_from_the_same_spec():
    assert BAYESIAN_DEFAULTS['bnn_kl_weight'] == 'elbo', \
        BAYESIAN_DEFAULTS['bnn_kl_weight']
    assert abs(bnn_kl_weight(8000) - 1 / 8000) < 1e-12

    runner = os.path.join(ROOT, '..', 'KIRBy', 'tests',
                          'alternative_data_noise_robustness.py')
    if not os.path.exists(runner):
        print("    (the experimental runner is not checked out beside this one)")
        return
    import importlib.util
    spec = importlib.util.spec_from_file_location('kirby_runner', runner)
    module = importlib.util.module_from_spec(spec)
    saved = sys.argv
    sys.argv = ['alternative_data_noise_robustness.py']
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv = saved
    torch.manual_seed(0)
    model = apply_bayesian_transformation(small_net())
    theirs = module.bnn_elbo_criterion(nn.MSELoss(), model, 500)
    mine = bnn_elbo_criterion(nn.MSELoss(), model, 500)
    assert theirs.kl_weight == mine.kl_weight, (theirs.kl_weight, mine.kl_weight)
    print(f"    both pipelines use KL weight {mine.kl_weight:.3e} at n=500")


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
    print("the BNN objective (RERUN_PLAN.md §2.12)")
    results = [
        check("the criterion carries the KL term",
              the_criterion_carries_the_kl_term),
        check("plain MSE collapses the posterior and the ELBO does not",
              plain_mse_collapses_the_posterior_and_the_elbo_does_not),
        check("both pipelines take the weight from the same spec",
              both_pipelines_take_the_weight_from_the_same_spec),
    ]
    if not all(results):
        print("\nFAIL: the Bayesian networks are not fitted on the ELBO")
        return 1
    print("\nOK: BNN-alpha and BNN-beta are fitted on the ELBO")
    return 0


if __name__ == "__main__":
    sys.exit(main())
