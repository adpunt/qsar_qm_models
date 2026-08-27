#!/usr/bin/env python
"""Changing a number in the shared spec must change what the pipeline builds.

Every results row carries `spec_version` and `spec_hash`, asserting that the row
was produced by the parameters in `models/model_defaults.py`. models.py restated
many of those numbers as literals that merely HAPPENED to equal their spec
counterpart, so editing the spec would have changed the hash on the row without
changing the model behind it (RERUN_PLAN.md §2.13). `audit_pipeline_parity.py`
cannot catch that: it builds models FROM the spec, so it never sees that models.py
builds from literals instead.

This does the opposite. It changes a spec value at run time and asserts the thing
the pipeline constructs changes with it.

Run it directly:  python scripts/test_spec_is_live.py
"""

import os
import sys
import traceback

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'models'))

import model_defaults  # noqa: E402
from model_defaults import BAYESIAN_DEFAULTS  # noqa: E402
from models import (VBLLLayer, apply_bayesian_transformation,  # noqa: E402
                    apply_bayesian_transformation_last_layer)


def net():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))


def bayes_priors(model):
    return sorted({(float(m.prior_mu), float(m.prior_sigma))
                   for m in model.modules() if hasattr(m, 'prior_sigma')})


def spec_change_reaches(key, new_value, build, read):
    old = BAYESIAN_DEFAULTS[key]
    before = read(build())
    BAYESIAN_DEFAULTS[key] = new_value
    try:
        after = read(build())
    finally:
        BAYESIAN_DEFAULTS[key] = old
    assert after != before, (
        f"changing {key} from {old!r} to {new_value!r} did not change what the "
        f"pipeline builds ({before!r} both times) -- models.py is using a "
        f"literal, and the spec_hash on every row would certify a run that did "
        f"not use the spec")
    return before, after


def the_bnn_prior_comes_from_the_spec():
    before, after = spec_change_reaches(
        'bnn_prior_sigma', 0.77,
        lambda: apply_bayesian_transformation(net()), bayes_priors)
    print(f"    full BNN priors {before} -> {after}")

    before, after = spec_change_reaches(
        'bnn_prior_sigma', 0.55,
        lambda: apply_bayesian_transformation_last_layer(net()), bayes_priors)
    print(f"    last-layer BNN priors {before} -> {after}")


def the_vbll_initialisation_comes_from_the_spec():
    before, after = spec_change_reaches(
        'vbll_init_log_sigma', -1.25,
        lambda: VBLLLayer(4, 1),
        lambda layer: round(float(layer.weight_log_sigma.flatten()[0]), 6))
    print(f"    VBLL initial log sigma {before} -> {after}")

    before, after = spec_change_reaches(
        'vbll_prior_sigma', 2.5,
        lambda: VBLLLayer(4, 1),
        lambda layer: float(layer.prior_sigma))
    print(f"    VBLL prior sigma {before} -> {after}")

    before, after = spec_change_reaches(
        'vbll_init_log_noise_var', -0.5,
        lambda: VBLLLayer(4, 1),
        lambda layer: round(float(layer.log_noise_var), 6))
    print(f"    VBLL initial log noise variance {before} -> {after}")


def the_kl_weight_comes_from_the_spec():
    old = BAYESIAN_DEFAULTS['bnn_kl_weight']
    try:
        BAYESIAN_DEFAULTS['bnn_kl_weight'] = 0.25
        assert model_defaults.bnn_kl_weight(1000) == 0.25
        BAYESIAN_DEFAULTS['bnn_kl_weight'] = 'elbo'
        assert model_defaults.bnn_kl_weight(1000) == 0.001
    finally:
        BAYESIAN_DEFAULTS['bnn_kl_weight'] = old
    print("    a fixed KL weight and the ELBO scaling both come through")


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
    print("the shared spec is live, not restated (RERUN_PLAN.md §2.13)")
    torch.manual_seed(0)
    results = [
        check("the BNN prior comes from the spec",
              the_bnn_prior_comes_from_the_spec),
        check("the VBLL initialisation comes from the spec",
              the_vbll_initialisation_comes_from_the_spec),
        check("the KL weight comes from the spec",
              the_kl_weight_comes_from_the_spec),
    ]
    if not all(results):
        print("\nFAIL: a spec value is restated as a literal in models.py")
        return 1
    print("\nOK: changing the spec changes what the pipeline builds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
