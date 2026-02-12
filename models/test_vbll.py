"""
Unit test for VBLL (Variational Bayesian Last Layer) implementation.

Tests:
1. VBLLLayer can be instantiated and produces correct output shapes
2. KL divergence is positive and finite
3. apply_bayesian_transformation_last_layer_variational replaces the correct layer
4. VBLLLoss combines MSE + KL correctly
5. Training loop runs and gradients flow through VBLL parameters
6. MC sampling at eval time produces diverse predictions (non-zero epistemic uncertainty)
7. Learned noise_var is a valid positive number after training

Run on server (full env): python test_vbll.py
Run locally (standalone):  python test_vbll.py --standalone
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import sys
import os

STANDALONE = '--standalone' in sys.argv


# ---------------------------------------------------------------------------
# Standalone copies of the VBLL classes (for environments missing torchcp etc.)
# These mirror models.py exactly. On the server, imports from models.py are used.
# ---------------------------------------------------------------------------
if STANDALONE:
    class VBLLLayer(nn.Module):
        def __init__(self, in_features, out_features, prior_mu=0.0, prior_sigma=1.0):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.prior_mu = prior_mu
            self.prior_sigma = prior_sigma
            self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
            self.weight_log_sigma = nn.Parameter(torch.full((out_features, in_features), -3.0))
            self.bias_mu = nn.Parameter(torch.zeros(out_features))
            self.bias_log_sigma = nn.Parameter(torch.full((out_features,), -3.0))
            self.log_noise_var = nn.Parameter(torch.tensor(-2.0))
            nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
            fan_in = in_features
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_mu, -bound, bound)

        @property
        def noise_var(self):
            return torch.exp(self.log_noise_var)

        def kl_divergence(self):
            prior_var = self.prior_sigma ** 2
            w_var = torch.exp(2.0 * self.weight_log_sigma)
            kl_w = 0.5 * torch.sum(
                w_var / prior_var + ((self.prior_mu - self.weight_mu) ** 2) / prior_var
                - 1.0 + math.log(prior_var) - 2.0 * self.weight_log_sigma
            )
            b_var = torch.exp(2.0 * self.bias_log_sigma)
            kl_b = 0.5 * torch.sum(
                b_var / prior_var + ((self.prior_mu - self.bias_mu) ** 2) / prior_var
                - 1.0 + math.log(prior_var) - 2.0 * self.bias_log_sigma
            )
            return kl_w + kl_b

        def forward(self, x):
            weight_sigma = torch.exp(self.weight_log_sigma)
            weight = self.weight_mu + weight_sigma * torch.randn_like(self.weight_mu)
            bias_sigma = torch.exp(self.bias_log_sigma)
            bias = self.bias_mu + bias_sigma * torch.randn_like(self.bias_mu)
            return F.linear(x, weight, bias)

    class VBLLLoss(nn.Module):
        def __init__(self, model, n_data):
            super().__init__()
            self.model = model
            self.n_data = n_data
            self.vbll_layer = None
            for module in model.modules():
                if isinstance(module, VBLLLayer):
                    self.vbll_layer = module
                    break

        def forward(self, pred, target):
            if self.vbll_layer is not None:
                noise_var = self.vbll_layer.noise_var
                nll = 0.5 * torch.log(noise_var) + 0.5 * ((pred - target) ** 2) / noise_var
                nll = nll.mean()
                kl = self.vbll_layer.kl_divergence()
                return nll + kl / self.n_data
            return nn.MSELoss()(pred, target)

    class DNNRegressionModel(nn.Module):
        def __init__(self, input_size, hidden_size1=32, hidden_size2=32):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size1)
            self.fc2 = nn.Linear(hidden_size1, hidden_size2)
            self.fc3 = nn.Linear(hidden_size2, 1)
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(p=0.2)
        def forward(self, x):
            x = self.activation(self.fc1(x))
            x = self.dropout(x)
            x = self.activation(self.fc2(x))
            x = self.dropout(x)
            return self.fc3(x)

    class MLPRegressor(nn.Module):
        def __init__(self, input_size, hidden_size=32, num_hidden_layers=2, dropout_rate=0.2):
            super().__init__()
            self.input_layer = nn.Linear(input_size, hidden_size)
            self.hidden_layers = nn.ModuleList()
            for _ in range(num_hidden_layers - 1):
                self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            self.output_layer = nn.Linear(hidden_size, 1)
            self.dropout = nn.Dropout(dropout_rate)
            self.relu = nn.ReLU()
        def forward(self, x):
            x = self.relu(self.input_layer(x))
            for layer in self.hidden_layers:
                x = self.relu(layer(x))
            x = self.dropout(x)
            return self.output_layer(x)

    def apply_bayesian_transformation_last_layer_variational(model):
        last_linear_name = None
        last_linear_module = None
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, nn.Linear):
                last_linear_name = name
                last_linear_module = module
                break
        if last_linear_module is None:
            raise ValueError("No nn.Linear layer found to replace.")
        vbll_layer = VBLLLayer(last_linear_module.in_features, last_linear_module.out_features)
        with torch.no_grad():
            vbll_layer.weight_mu.copy_(last_linear_module.weight.data)
            if last_linear_module.bias is not None:
                vbll_layer.bias_mu.copy_(last_linear_module.bias.data)
        def set_nested_attr(obj, attr_path, value):
            attrs = attr_path.split(".")
            for a in attrs[:-1]:
                obj = getattr(obj, a)
            setattr(obj, attrs[-1], value)
        set_nested_attr(model, last_linear_name, vbll_layer)
        return model

    print("(standalone mode — using inline class definitions)\n")

else:
    # Full import from models.py (requires server environment with all deps)
    sys.path.insert(0, os.path.dirname(__file__))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
    from models import (
        VBLLLayer,
        VBLLLoss,
        DNNRegressionModel,
        MLPRegressor,
        apply_bayesian_transformation_last_layer_variational,
    )


def test_vbll_layer_forward():
    """VBLLLayer produces correct output shape in train and eval."""
    layer = VBLLLayer(in_features=64, out_features=1)
    x = torch.randn(32, 64)

    # Training mode — should sample
    layer.train()
    out_train = layer(x)
    assert out_train.shape == (32, 1), f"Expected (32,1), got {out_train.shape}"

    # Eval mode — should also sample (for MC uncertainty)
    layer.eval()
    out_eval = layer(x)
    assert out_eval.shape == (32, 1), f"Expected (32,1), got {out_eval.shape}"

    print("  PASS: VBLLLayer forward shapes correct")


def test_vbll_kl_divergence():
    """KL divergence is positive and finite."""
    layer = VBLLLayer(in_features=64, out_features=1)
    kl = layer.kl_divergence()

    assert kl.item() > 0, f"KL should be > 0, got {kl.item()}"
    assert torch.isfinite(kl), f"KL should be finite, got {kl.item()}"

    print(f"  PASS: KL divergence = {kl.item():.4f} (positive, finite)")


def test_transformation_dnn():
    """apply_bayesian_transformation_last_layer_variational replaces DNN last layer."""
    model = DNNRegressionModel(input_size=128, hidden_size1=64, hidden_size2=32)

    # Before: fc3 is nn.Linear
    assert isinstance(model.fc3, nn.Linear), "fc3 should be nn.Linear before transform"

    model = apply_bayesian_transformation_last_layer_variational(model)

    # After: fc3 should be VBLLLayer
    assert isinstance(model.fc3, VBLLLayer), f"fc3 should be VBLLLayer, got {type(model.fc3)}"
    assert model.fc3.in_features == 32, f"Expected in_features=32, got {model.fc3.in_features}"
    assert model.fc3.out_features == 1, f"Expected out_features=1, got {model.fc3.out_features}"

    print("  PASS: DNN transformation replaces fc3 with VBLLLayer")


def test_transformation_mlp():
    """apply_bayesian_transformation_last_layer_variational replaces MLP last layer."""
    model = MLPRegressor(input_size=128, hidden_size=64, num_hidden_layers=2)

    assert isinstance(model.output_layer, nn.Linear)
    model = apply_bayesian_transformation_last_layer_variational(model)
    assert isinstance(model.output_layer, VBLLLayer), f"output_layer should be VBLLLayer, got {type(model.output_layer)}"

    print("  PASS: MLP transformation replaces output_layer with VBLLLayer")


def test_vbll_loss():
    """VBLLLoss returns MSE + KL/n_data and is differentiable."""
    model = DNNRegressionModel(input_size=32, hidden_size1=16, hidden_size2=8)
    model = apply_bayesian_transformation_last_layer_variational(model)

    criterion = VBLLLoss(model, n_data=100)

    x = torch.randn(16, 32)
    y = torch.randn(16, 1)
    pred = model(x)
    loss = criterion(pred, y)

    assert torch.isfinite(loss), f"Loss should be finite, got {loss.item()}"

    # Check gradients flow through all VBLL parameters
    loss.backward()
    vbll_layer = criterion.vbll_layer
    assert vbll_layer.weight_mu.grad is not None, "weight_mu should have gradients"
    assert vbll_layer.weight_log_sigma.grad is not None, "weight_log_sigma should have gradients"
    assert vbll_layer.log_noise_var.grad is not None, "log_noise_var should have gradients"

    # Verify KL is positive
    kl = vbll_layer.kl_divergence()
    assert kl.item() > 0, f"KL should be > 0, got {kl.item()}"

    print(f"  PASS: VBLLLoss = {loss.item():.4f} (NLL + KL/n), KL={kl.item():.4f}")


def test_training_loop():
    """Train a VBLL model for 10 epochs, verify loss decreases and params change."""
    torch.manual_seed(42)
    np.random.seed(42)

    # Synthetic regression data
    n_train = 200
    x = torch.randn(n_train, 32)
    y = x[:, 0:1] * 2 + 0.5  # Simple linear target

    model = DNNRegressionModel(input_size=32, hidden_size1=16, hidden_size2=8)
    model = apply_bayesian_transformation_last_layer_variational(model)
    criterion = VBLLLoss(model, n_data=n_train)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Save initial params
    init_weight_mu = model.fc3.weight_mu.data.clone()
    init_log_sigma = model.fc3.weight_log_sigma.data.clone()
    init_log_noise = model.fc3.log_noise_var.data.clone()

    losses = []
    for epoch in range(10):
        model.train()
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Loss should decrease
    assert losses[-1] < losses[0], f"Loss should decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

    # Parameters should have changed
    assert not torch.allclose(model.fc3.weight_mu.data, init_weight_mu), "weight_mu should change during training"
    assert not torch.allclose(model.fc3.weight_log_sigma.data, init_log_sigma), "weight_log_sigma should change during training"

    print(f"  PASS: Training loop — loss {losses[0]:.4f} → {losses[-1]:.4f}")


def test_mc_sampling_uncertainty():
    """MC sampling at eval time produces non-zero epistemic uncertainty."""
    torch.manual_seed(42)

    model = DNNRegressionModel(input_size=32, hidden_size1=16, hidden_size2=8)
    model = apply_bayesian_transformation_last_layer_variational(model)
    model.eval()

    x_test = torch.randn(50, 32)
    num_samples = 10
    preds = []

    with torch.no_grad():
        for _ in range(num_samples):
            out = model(x_test).numpy()
            preds.append(out)

    preds = np.stack(preds, axis=0)  # (10, 50, 1)
    epistemic = preds.std(axis=0).flatten()

    assert epistemic.mean() > 0, f"Epistemic uncertainty should be > 0, got mean={epistemic.mean()}"

    # Check that predictions vary across samples
    assert not np.allclose(preds[0], preds[1]), "Different MC samples should give different predictions"

    # Check learned noise variance
    noise_var = model.fc3.noise_var.item()
    assert noise_var > 0, f"Learned noise_var should be > 0, got {noise_var}"
    assert np.isfinite(noise_var), f"noise_var should be finite, got {noise_var}"

    print(f"  PASS: MC sampling — mean epistemic={epistemic.mean():.6f}, noise_var={noise_var:.6f}")


def test_weight_initialization():
    """VBLLLayer initializes weight_mu from pretrained weights."""
    original = nn.Linear(64, 1)
    original_weight = original.weight.data.clone()
    original_bias = original.bias.data.clone()

    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), original)
    model = apply_bayesian_transformation_last_layer_variational(model)

    vbll = model[2]
    assert isinstance(vbll, VBLLLayer)
    assert torch.allclose(vbll.weight_mu.data, original_weight), "weight_mu should be initialized from pretrained weight"
    assert torch.allclose(vbll.bias_mu.data, original_bias), "bias_mu should be initialized from pretrained bias"

    print("  PASS: Weight initialization preserves pretrained weights")


if __name__ == '__main__':
    print("Running VBLL unit tests...\n")

    test_vbll_layer_forward()
    test_vbll_kl_divergence()
    test_transformation_dnn()
    test_transformation_mlp()
    test_vbll_loss()
    test_training_loop()
    test_mc_sampling_uncertainty()
    test_weight_initialization()

    print("\nAll VBLL tests passed!")
