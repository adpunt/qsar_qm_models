"""Gates on the variational layer that predicts noise per molecule.

The layer as built has ONE learned observation noise for the whole fit, so its
aleatoric term is identical for every molecule and cannot correlate with
per-molecule injected noise however good the model is (`RERUN_PLAN.md` 5.5).
The reference library saved at `research_archive/f692d614/vbll_regression.py`
carries a heteroscedastic variant, `HetRegression` at :277-370, whose noise is a
function of the input. That variant is now built here, and these are its gates.

Run:  python scripts/test_heteroscedastic_vbll.py

Nothing here needs the cluster or a GPU. The last gate trains a small network
and takes a few seconds.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'models'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import (VBLLLayer, VBLLLoss,  # noqa: E402
                    apply_bayesian_transformation_full_variational,
                    apply_bayesian_transformation_last_layer_variational)

FAILURES = []


def check(name, fn):
    try:
        fn()
        print(f"  ok    {name}")
    except AssertionError as e:
        print(f"  FAIL  {name}: {e}")
        FAILURES.append(name)
    except Exception as e:
        print(f"  ERROR {name}: {type(e).__name__}: {e}")
        FAILURES.append(name)


def test_default_layer_is_unchanged():
    """The homoscedastic layer is what every existing result was produced with.
    Adding the option must not alter it."""
    layer = VBLLLayer(4, 1)
    assert layer.heteroscedastic is False
    assert not hasattr(layer, 'noise_mu'), (
        "the homoscedastic layer grew a noise head it should not have")
    x = torch.randn(7, 4)
    noise = layer.noise_var_for(x)
    assert float(noise.max() - noise.min()) < 1e-12, (
        "the default layer's noise is no longer one number for the whole fit")


def test_heteroscedastic_noise_varies_per_molecule():
    torch.manual_seed(0)
    layer = VBLLLayer(4, 1, heteroscedastic=True)
    # Start it away from the initialisation, which deliberately predicts nothing.
    with torch.no_grad():
        layer.noise_mu.normal_(0.0, 1.0)
    x = torch.randn(64, 4)
    noise = layer.noise_var_for(x)
    assert noise.shape == (64,), f"expected one value per molecule, got {noise.shape}"
    assert float(noise.std()) > 1e-6, (
        "the heteroscedastic layer produced the same noise for every molecule")
    assert torch.all(noise > 0), "a variance came out non-positive"


def test_noise_head_starts_at_the_homoscedastic_value():
    """It must earn any departure from the single-noise model rather than
    starting somewhere arbitrary."""
    torch.manual_seed(0)
    het = VBLLLayer(6, 1, heteroscedastic=True)
    hom = VBLLLayer(6, 1)
    x = torch.zeros(5, 6)  # at zero input only the bias speaks
    with torch.no_grad():
        het.noise_bias_log_sigma.fill_(-20.0)  # silence the sampling noise
        got = het.noise_var_for(x)
        want = hom.noise_var
    assert torch.allclose(got, want.expand_as(got), rtol=1e-4), (
        f"noise head starts at {float(got[0]):.4g}, the single-noise layer is "
        f"at {float(want):.4g}")


def test_noise_head_is_regularised():
    """The head is variational in the reference implementation, so it carries a
    KL term. Without one it is a free function fitting residuals, and the
    aleatoric term stops being a posterior quantity."""
    torch.manual_seed(0)
    het = VBLLLayer(5, 1, heteroscedastic=True)
    hom = VBLLLayer(5, 1)
    with torch.no_grad():
        for p_het, p_hom in ((het.weight_mu, hom.weight_mu),
                             (het.weight_log_sigma, hom.weight_log_sigma),
                             (het.bias_mu, hom.bias_mu),
                             (het.bias_log_sigma, hom.bias_log_sigma)):
            p_het.copy_(p_hom)
        het.noise_mu.normal_(0.0, 1.0)
    assert float(het.kl_divergence()) > float(hom.kl_divergence()) + 1e-6, (
        "the noise head contributes nothing to the KL term, so it is "
        "unregularised")


def test_loss_uses_the_per_molecule_noise():
    """If the loss reads the single noise value while the head predicts per
    molecule, the head never trains and the column it writes means nothing."""
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.ReLU(),
                              VBLLLayer(8, 1, heteroscedastic=True))
    with torch.no_grad():
        net[2].noise_mu.normal_(0.0, 2.0)
    loss_fn = VBLLLoss(net, n_data=100)

    x = torch.randn(32, 3)
    pred = net(x)
    cached = net[2]._last_noise_var
    assert cached is not None and cached.shape == (32,), (
        "the layer did not record a per-molecule noise on its forward pass")
    assert float(cached.std()) > 1e-6, "the recorded noise does not vary"

    loss = loss_fn(pred, torch.randn(32, 1))
    loss.backward()
    assert net[2].noise_mu.grad is not None, (
        "no gradient reached the noise head")
    assert float(net[2].noise_mu.grad.abs().sum()) > 0, (
        "the gradient reaching the noise head is exactly zero, so the loss is "
        "not reading what the head predicted")


def test_it_learns_where_the_noise_is():
    """The point of the whole thing: train on labels whose noise is three times
    larger for one half of the molecules, and check the layer's aleatoric term
    is larger for that half on molecules it was not trained on."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    n, d = 900, 4
    x = rng.normal(size=(n, d)).astype(np.float32)
    signal = x[:, 0] * 2.0
    hot = x[:, 1] > 0.0                       # which molecules get noisier labels
    scale = np.where(hot, 0.9, 0.3).astype(np.float32)
    y = (signal + rng.normal(scale=scale)).astype(np.float32)

    xt = torch.from_numpy(x[:700])
    yt = torch.from_numpy(y[:700]).unsqueeze(1)
    xs = torch.from_numpy(x[700:])
    hot_test = hot[700:]

    net = torch.nn.Sequential(torch.nn.Linear(d, 32), torch.nn.ReLU(),
                              VBLLLayer(32, 1, heteroscedastic=True))
    loss_fn = VBLLLoss(net, n_data=len(xt))
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    for _ in range(400):
        opt.zero_grad()
        loss_fn(net(xt), yt).backward()
        opt.step()

    net.eval()
    with torch.no_grad():
        passes = np.array([net[2].noise_var_for(
            torch.nn.functional.relu(net[0](xs))).numpy() for _ in range(50)])
    alea = passes.mean(axis=0)

    noisy, quiet = alea[hot_test].mean(), alea[~hot_test].mean()
    assert noisy > quiet, (
        f"the layer put MORE noise on the quiet molecules: noisy {noisy:.4g} "
        f"vs quiet {quiet:.4g}")
    assert noisy > quiet * 1.5, (
        f"the separation is too small to be worth reporting: noisy {noisy:.4g} "
        f"vs quiet {quiet:.4g} (the true spreads differ by 3x)")


def test_the_switch_reaches_the_layer():
    """It was built and nothing could reach it: no command-line flag, no roster
    entry, and the transformation that builds the layers took no argument. This
    is the wiring, from the flag through to the layer (RERUN_PLAN.md 5.5f)."""
    def _net():
        return torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.ReLU(),
                                   torch.nn.Linear(6, 1))

    plain = apply_bayesian_transformation_full_variational(_net())
    layers = [m for m in plain.modules() if isinstance(m, VBLLLayer)]
    assert layers and not any(l.heteroscedastic for l in layers), (
        "the default must be untouched")

    het = apply_bayesian_transformation_full_variational(_net(),
                                                         heteroscedastic=True)
    layers = [m for m in het.modules() if isinstance(m, VBLLLayer)]
    assert layers[-1].heteroscedastic, "the OUTPUT layer must carry the noise head"
    assert not any(l.heteroscedastic for l in layers[:-1]), (
        "a hidden layer has no observation noise to make heteroscedastic")

    last = apply_bayesian_transformation_last_layer_variational(
        _net(), heteroscedastic=True)
    assert [m for m in last.modules()
            if isinstance(m, VBLLLayer)][-1].heteroscedastic

    # The flag itself. Read from the source rather than by importing
    # process_and_train.py, which pulls in the whole training stack.
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, 'scripts', 'process_and_train.py')).read()
    assert '--heteroscedastic-vbll' in src, "the flag is not declared"
    assert 'heteroscedastic_vbll' in src and 'parser.error' in src, (
        "the flag is declared but nothing refuses it where it means nothing")
    assert 'heteroscedastic_vbll' in open(
        os.path.join(root, 'models', 'models.py')).read(), (
        "no trainer reads the flag, so it would be accepted and ignored")

    gen = open(os.path.join(root, 'slurm_scripts_qm9_rerun',
                            'generate_scripts.py')).read()
    for label in ('dnn_bnn_full_variational_hetero',
                  'mlp_bnn_full_variational_hetero',
                  'heteroscedastic_gp'):
        assert label in gen, f"{label} is in no job script, so it never runs"


def test_the_row_name_separates_the_two_models():
    """A per-molecule column and a broadcast constant under one model name are
    indistinguishable once they are in a file."""
    from uncertainty_decomposition import SUPPORT, PER_MOLECULE, CONSTANT
    assert SUPPORT['dnn_bnn_full_variational'][0] == CONSTANT
    assert SUPPORT['dnn_bnn_full_variational_hetero'][0] == PER_MOLECULE
    assert SUPPORT['mlp_bnn_full_variational'][0] == CONSTANT
    assert SUPPORT['mlp_bnn_full_variational_hetero'][0] == PER_MOLECULE


if __name__ == '__main__':
    print("Heteroscedastic variational layer -- gates\n")
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith('test_') and callable(v)]
    for name, fn in tests:
        check(name, fn)
    print(f"\n{len(tests) - len(FAILURES)}/{len(tests)} passed")
    if FAILURES:
        print("FAILED: " + ", ".join(FAILURES))
        sys.exit(1)
    sys.exit(0)
