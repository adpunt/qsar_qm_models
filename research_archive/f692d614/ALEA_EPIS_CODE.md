# Aleatoric / epistemic decomposition — actual implementation code

Every block below was read from the source file named above it. Nothing here is
paraphrased from a paper abstract. Fetched 2026-08-21.

---

## 1. The decomposition, in code

### 1a. Ryu, Kwon & Kim 2019 — the canonical five lines

**Source read:** `https://raw.githubusercontent.com/SeongokRyu/uq_molecule/master/train_cep.py`
(repo is `SeongokRyu/uq_molecule`, **not** `seongokryu/uq-molecule` — that name 404s).
Lines 119–133, verbatim:

```python
        # MC-sampling
        P_mean = []
        P_logvar = []
        for n in range(num_sampling):
            Y_mean, Y_logvar, loss = model.test(A_batch, X_batch, Y_batch)
            P_mean.append(Y_mean.flatten())
            P_logvar.append(Y_logvar.flatten())

        P_mean = np.asarray(P_mean)
        P_logvar = np.exp(np.asarray(P_logvar))

        mean = np.mean(P_mean, axis=0)
        ale_unc = np.mean(P_logvar, axis=0)
        epi_unc = np.var(P_mean, axis=0)
        tot_unc = ale_unc + epi_unc
```

Shapes: `P_mean` and `P_logvar` are `(num_sampling, batch)`. Note `P_logvar` is
overwritten with `exp(...)` on the line before use — so `ale_unc` is a mean of
**variances**, not of log-variances. All three outputs are in variance units;
they save them straight to disk:

```python
    np.save('./statistics/'+model_name+'_mc_epi_unc.npy', epi_unc_total)
    np.save('./statistics/'+model_name+'_mc_ale_unc.npy', ale_unc_total)
    np.save('./statistics/'+model_name+'_mc_tot_unc.npy', tot_unc_total)
```

### 1b. Chemprop v1 — same thing, written as a running sum

**Source read:** `https://raw.githubusercontent.com/chemprop/chemprop/v1.7.1/chemprop/uncertainty/uncertainty_predictor.py`
`class MVEPredictor`, lines 581–584, verbatim:

```python
            uncal_preds = sum_preds / self.num_models
            uncal_vars = (sum_vars + sum_squared) / self.num_models - np.square(
                sum_preds / self.num_models
            )
```

where, earlier in the same loop over ensemble members (lines 512–538):

```python
                sum_preds = np.array(preds)
                sum_squared = np.square(preds)
                sum_vars = np.array(var)
...
                sum_preds += np.array(preds)
                sum_squared += np.square(preds)
                sum_vars += np.array(var)
```

This is algebraically identical to Ryu. I checked it numerically:

```python
# (sum_vars + sum_squared)/M - (sum_preds/M)**2
#   ==  vars_.mean(0) + mus.var(0)      -> True, exact float match
```

so Chemprop's single `uncal_vars` **is** `aleatoric + epistemic` fused. If you
want them separately from Chemprop v1 you must split that expression yourself:

```python
aleatoric = sum_vars / M                                  # mean of predicted variances
epistemic = sum_squared / M - (sum_preds / M) ** 2         # variance of the means
```

The pure-ensemble predictor (no variance head) keeps only the second term.
`class EnsemblePredictor`, lines 1155–1158, verbatim:

```python
            uncal_vars = (
                sum_squared / self.num_models
                - np.square(sum_preds) / self.num_models**2
            )
```

### 1c. Chemprop v2 — the same two halves, now in separate estimator classes

**Source read:** `https://raw.githubusercontent.com/chemprop/chemprop/main/chemprop/uncertainty/estimator.py`

`MVEEstimator.__call__` (lines 167–180) just unbinds the head and does **not**
pool across models:

```python
        means = []
        vars = []
...
                mean, var = mves.unbind(dim=-1)
                means.append(mean)
                vars.append(var)
```

`EnsembleEstimator.__call__` (line 241) supplies the epistemic half:

```python
                vars = torch.var(stacked_preds, dim=0, correction=0).unsqueeze(0)
```

`DropoutEstimator.__call__` (lines 540–541):

```python
                means = torch.mean(stacked_preds, dim=0)
                vars = torch.var(stacked_preds, dim=0, correction=0)
```

`correction=0` is the population variance (divide by N, not N−1) — matches
`np.var` default. **In v2 you have to add the two halves yourself**; v1 did it
for you.

### 1d. Scalia et al. 2020 — code found, and it is the same recipe

**Source read:** `https://raw.githubusercontent.com/gscalia/chemprop/uncertainty/chemprop/train/predict.py`
lines 106–117, verbatim (the `aleatoric and mc_dropout` branch):

```python
        elif aleatoric and mc_dropout:
            with torch.no_grad():
                P_mean = []
                P_logvar = []

                for ss in range(sampling_size):
                    batch_preds, batch_logvar = model(batch, features_batch)
                    P_mean.append(batch_preds)
                    P_logvar.append(torch.exp(batch_logvar))

                batch_preds = torch.mean(torch.stack(P_mean), 0)
                batch_ale_unc = torch.mean(torch.stack(P_logvar), 0)
                batch_epi_unc = torch.var(torch.stack(P_mean), 0)
```

Note the one thing everybody forgets — **rescale the variances, not the
standard deviations**, when you undo target standardisation (lines 124–127):

```python
            if scaler is not None:
                batch_preds = scaler.inverse_transform(batch_preds)
                batch_ale_unc = scaler.inverse_transform_variance(batch_ale_unc)
                batch_epi_unc = scaler.inverse_transform_variance(batch_epi_unc)
```

---

## 2. How to add a variance head to a BNN

### 2a. Two output layers off the same penultimate features

**Source read:** `https://raw.githubusercontent.com/gscalia/chemprop/uncertainty/chemprop/models/model.py`
lines 96–117, verbatim:

```python
        if self.aleatoric:
            self.output_layer = nn.Linear(last_linear_dim, args.output_size)
            self.logvar_layer = nn.Linear(last_linear_dim, args.output_size)
        else:
            self.output_layer = nn.Linear(last_linear_dim, args.output_size)

    def forward(self, *input):
        _output = self._ffn(self.encoder(*input))

        if self.aleatoric:
            output = self.output_layer(_output)
            logvar = self.logvar_layer(_output)

            # Gaussian uncertainty only for regression, directly returning in this case
            return output, logvar
```

Ryu's TensorFlow version is the same shape.
**Source read:** `https://raw.githubusercontent.com/SeongokRyu/uq_molecule/master/blocks.py`
lines 102–113, verbatim:

```python
    _Y = tf.nn.relu(dense_with_concrete_dropout(Z, latent_dim, wd, dd))
    Y_mean = tf.keras.layers.Dense(units=1,
                                   use_bias=True,
                                   activation=None,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                   bias_initializer=tf.contrib.layers.xavier_initializer())(_Y)
    Y_logvar = tf.keras.layers.Dense(units=1,
                                     use_bias=True,
                                     activation=None,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.contrib.layers.xavier_initializer())(_Y)
    return Z, Y_mean, Y_logvar
```

Chemprop v2 instead emits one 2-wide tensor and softpluses the second column.
**Source read:** `https://raw.githubusercontent.com/chemprop/chemprop/main/chemprop/nn/predictors.py`
`class MveFFN`, lines 173–188, verbatim:

```python
class MveFFN(RegressionFFN):
    n_targets = 2
    _T_default_criterion = MVELoss

    def forward(self, Z: Tensor) -> Tensor:
        Y = self.ffn(Z)
        mean, var = torch.chunk(Y, self.n_targets, 1)
        var = F.softplus(var)

        mean = self.output_transform(mean)
        if not isinstance(self.output_transform, nn.Identity):
            var = self.output_transform.transform_variance(var)

        return torch.stack((mean, var), dim=2)
```

Two designs, one decision: predict `log_var` unconstrained (Ryu, Scalia), or
predict `var` and force positivity with `softplus` (Chemprop v2). `log_var` is
numerically better behaved early in training.

### 2b. The Gaussian NLL loss — three real versions

**Scalia fork**, `https://raw.githubusercontent.com/gscalia/chemprop/uncertainty/chemprop/utils.py`
lines 155–166, verbatim:

```python
def heteroscedastic_loss(true, mean, log_var):
    """
    Compute the heteroscedastic loss for regression.
    """
    precision = torch.exp(-log_var)
    loss = precision * (true - mean)**2 + log_var
    return loss
```

**Ryu**, `https://raw.githubusercontent.com/SeongokRyu/uq_molecule/master/mc_dropout.py`
line 52, verbatim (note the `0.5` factors that Scalia drops — same minimum,
different scale):

```python
        pred_loss = tf.reduce_mean(0.5*tf.exp(-P_logvar)*(P_truth-P_mean)**2 + P_logvar*0.5)
```

**Chemprop v1**, `chemprop/train/loss_functions.py` lines 240–243, verbatim
(parameterised in variance, includes the `2*pi` constant):

```python
    # Unpack combined prediction values
    pred_means, pred_var = torch.split(pred_values, pred_values.shape[1] // 2, dim=1)

    return torch.log(2 * np.pi * pred_var) / 2 + (pred_means - targets) ** 2 / (2 * pred_var)
```

**Chemprop v2**, `chemprop/nn/metrics.py` `class MVELoss` lines 211–217, verbatim:

```python
    def _calc_unreduced_loss(self, preds: Tensor, targets: Tensor, *args) -> Tensor:
        mean, var = torch.unbind(preds, dim=-1)

        L_sos = (mean - targets) ** 2 / (2 * var)
        L_kl = (2 * torch.pi * var).log() / 2

        return L_sos + L_kl
```

A drop-in for your own PyTorch BNN, log-var parameterised (this is the Ryu form
with the constant added so the number is a real NLL):

```python
import math, torch

def gaussian_nll(y, mean, log_var):
    """Per-element heteroscedastic Gaussian NLL. mean/log_var are the two heads."""
    return 0.5 * (math.log(2 * math.pi) + log_var + torch.exp(-log_var) * (y - mean) ** 2)
```

Torch also ships `torch.nn.functional.gaussian_nll_loss(input, target, var)`,
which takes **var**, not log-var, and clamps it at `eps=1e-6`.

---

## 3. QRF aleatoric / epistemic without retraining

The mechanism to copy is how `zillow/quantile-forest` recovers the training
labels sitting in each leaf.
**Source read:** `https://raw.githubusercontent.com/zillow/quantile-forest/main/quantile_forest/_quantile_forest.py`
lines 356–378, verbatim:

```python
            X_leaves = self.apply(X)

        if self.bootstrap:
            args = (n_samples, self.max_samples)
...
            n_samples_bootstrap = _get_n_samples_bootstrap(*args)
...
        for i, estimator in enumerate(self.estimators_):
            # Get bootstrap indices.
            if self.bootstrap:
                args = (estimator.random_state, n_samples, n_samples_bootstrap)
...
                bootstrap_indices[:, i] = _generate_sample_indices(*args)
            else:
                bootstrap_indices[:, i] = np.arange(n_samples)

            # Get leaf node indices of bootstrap training samples.
            X_leaves_bootstrap[:, i] = X_leaves[bootstrap_indices[:, i], i]
```

Attribute check (run against the installed sklearn, **1.3.2**, in this repo's env):
`forest.estimators_` ✅, `forest.apply(X)` → shape `(n_samples, n_trees)` ✅,
`tree_.value` → shape `(n_nodes, 1, 1)` for single-output regression ✅,
`tree_.impurity` → shape `(n_nodes,)` ✅, `criterion == "squared_error"` so
`impurity` at a node **is** the within-node MSE ✅.
`_generate_sample_indices` and `_get_n_samples_bootstrap` import fine from
`sklearn.ensemble._forest` ✅.

### Working function (tested, see below)

```python
import numpy as np
from sklearn.ensemble._forest import _generate_sample_indices, _get_n_samples_bootstrap


def forest_aleatoric_epistemic(forest, X_train, y_train, X):
    """Split a fitted sklearn RandomForestRegressor's predictive variance into
    aleatoric (mean within-leaf variance) and epistemic (variance of per-tree means).

    Law of total variance over the mixture of per-tree leaf distributions:
        Var[y|x] = E_t[ Var(y | leaf_t(x)) ]  +  Var_t[ E(y | leaf_t(x)) ]
                   \_______ aleatoric _______/   \______ epistemic ______/

    Returns (mean, aleatoric, epistemic), each shape (n_samples,), in y units**2.
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train, dtype=float).ravel()
    n_train = X_train.shape[0]

    # Which leaf every training point and every query point lands in, per tree.
    train_leaves = forest.apply(X_train)          # (n_train, n_trees)
    test_leaves = forest.apply(X)                 # (n_query, n_trees)

    if forest.bootstrap:
        n_boot = _get_n_samples_bootstrap(n_train, forest.max_samples)

    tree_means = np.empty((test_leaves.shape[0], forest.n_estimators))
    tree_vars = np.empty_like(tree_means)

    for i, est in enumerate(forest.estimators_):
        # Reproduce exactly the rows this tree was grown on (in-bag, with repeats).
        if forest.bootstrap:
            idx = _generate_sample_indices(est.random_state, n_train, n_boot)
        else:
            idx = np.arange(n_train)

        leaf_of = train_leaves[idx, i]
        y_of = y_train[idx]

        n_nodes = est.tree_.node_count
        cnt = np.bincount(leaf_of, minlength=n_nodes).astype(float)
        s1 = np.bincount(leaf_of, weights=y_of, minlength=n_nodes)
        s2 = np.bincount(leaf_of, weights=y_of ** 2, minlength=n_nodes)

        safe = np.where(cnt > 0, cnt, 1.0)
        leaf_mean = s1 / safe
        leaf_var = np.maximum(s2 / safe - leaf_mean ** 2, 0.0)

        tree_means[:, i] = leaf_mean[test_leaves[:, i]]
        tree_vars[:, i] = leaf_var[test_leaves[:, i]]

    mean = tree_means.mean(axis=1)
    aleatoric = tree_vars.mean(axis=1)                    # E_t[Var within leaf]
    epistemic = tree_means.var(axis=1)                    # Var_t[leaf mean]
    return mean, aleatoric, epistemic
```

Shortcut if you do not want to recompute leaf statistics: `leaf_mean` is already
stored as `est.tree_.value[leaf, 0, 0]` and `leaf_var` as
`est.tree_.impurity[leaf]` (because the criterion is `squared_error`). The
version above reproduces both from the in-bag rows so that it stays correct if
you ever pass `sample_weight` or a non-default criterion.

### What the test actually showed — read this before you use it

Test: n=2000, 5 features, `y = 3*X0 + N(0, 0.1 + 0.9*X1)` (aleatoric noise
deliberately made a function of X1), `RandomForestRegressor(n_estimators=200,
min_samples_leaf=20, random_state=0)`.

```
mean matches forest.predict:  True         # exact, so the plumbing is right
alea in low-noise region (X1<0.2):  0.183   true sigma^2 ~ 0.036
alea in high-noise region (X1>0.8): 0.484   true sigma^2 ~ 0.828
epistemic mean:                     0.056
```

Two honest caveats:

1. **Aleatoric is biased.** Within-leaf variance also contains whatever signal
   variation survives inside the leaf, so it is inflated at low noise and
   compressed at high noise. It *tracks* heteroscedasticity (0.18 → 0.48 in the
   right direction) but it is not an unbiased estimate of sigma^2(x). Shrinking
   `min_samples_leaf` reduces the bias and raises the variance of the estimate.
2. **Epistemic is a disagreement measure, not a posterior variance.** Repeating
   the test at n=200 gave epistemic 0.049 vs 0.056 at n=2000 — it did **not**
   shrink with more data the way a real posterior would, because tree-to-tree
   spread is driven by bootstrap and feature subsampling, not by sample size
   alone. Treat it as an ordinal signal and calibrate it before reporting
   coverage.

For calibration, Chemprop's z-scaling is the simple thing to copy.
**Source read:** `https://raw.githubusercontent.com/chemprop/chemprop/main/chemprop/uncertainty/calibrator.py`
`class ZScalingCalibrator`, lines 96–111, verbatim:

```python
            errors = preds_j - targets_j

            def objective(scaler_value: float):
                scaled_vars = uncs_j * scaler_value**2
                nll = np.log(2 * np.pi * scaled_vars) / 2 + errors**2 / (2 * scaled_vars)
                return nll.sum()

            zscore = errors / np.sqrt(uncs_j)
            initial_guess = np.std(zscore)
            scalings[j] = fmin(objective, x0=initial_guess, disp=False).item()

        self.scalings = torch.tensor(scalings)
        return self

    def apply(self, uncs: Tensor) -> Tensor:
        return uncs * self.scalings**2
```

Fit `scalings` on a held-out validation split, apply to test. `fmin` is
`scipy.optimize.fmin`.

---

## 4. Calibration metrics

**Source read:** `https://raw.githubusercontent.com/uncertainty-toolbox/uncertainty-toolbox/main/uncertainty_toolbox/metrics_calibration.py`

### Sharpness — lines 19–37, verbatim body

```python
def sharpness(y_std: np.ndarray) -> float:
    """Return sharpness (a single measure of the overall confidence)."""
    # Check that input arrays are flat
    assert_is_flat_same_shape(y_std)
    # Check that input std is positive
    assert_is_positive(y_std)

    # Compute sharpness
    sharp_metric = np.sqrt(np.mean(y_std**2))

    return sharp_metric
```

Root-mean-**square** of the standard deviations, i.e. `sqrt(mean(variance))` —
not the mean of the sigmas. Easy thing to get wrong.

### Mean absolute calibration error (== ECE) — lines 124–137, verbatim body

```python
    # Get lists of expected and observed proportions for a range of quantiles
    if vectorized:
        (exp_proportions, obs_proportions) = get_proportion_lists_vectorized(
            y_pred, y_std, y_true, num_bins, recal_model, prop_type
        )
    else:
        (exp_proportions, obs_proportions) = get_proportion_lists(
            y_pred, y_std, y_true, num_bins, recal_model, prop_type
        )

    abs_diff_proportions = np.abs(exp_proportions - obs_proportions)
    mace = np.mean(abs_diff_proportions)
```

with `get_proportion_lists_vectorized` (lines 368–392, verbatim):

```python
    # Compute proportions
    exp_proportions = np.linspace(0, 1, num_bins)
    # If we are recalibrating, input proportions are recalibrated proportions
    if recal_model is not None:
        in_exp_proportions = recal_model.predict(exp_proportions)
    else:
        in_exp_proportions = exp_proportions

    residuals = y_pred - y_true
    normalized_residuals = (residuals.flatten() / y_std.flatten()).reshape(-1, 1)
    norm = stats.norm(loc=0, scale=1)
    if prop_type == "interval":
        gaussian_lower_bound = norm.ppf(0.5 - in_exp_proportions / 2.0)
        gaussian_upper_bound = norm.ppf(0.5 + in_exp_proportions / 2.0)

        above_lower = normalized_residuals >= gaussian_lower_bound
        below_upper = normalized_residuals <= gaussian_upper_bound

        within_quantile = above_lower * below_upper
        obs_proportions = np.sum(within_quantile, axis=0).flatten() / len(residuals)
    elif prop_type == "quantile":
        gaussian_quantile_bound = norm.ppf(in_exp_proportions)
        below_quantile = normalized_residuals <= gaussian_quantile_bound
        obs_proportions = np.sum(below_quantile, axis=0).flatten() / len(residuals)
```

### Miscalibration area — lines 267–301, verbatim body

```python
    # Compute the expected proportions and the residuals.
    exp_proportions = np.linspace(0, 1, num_bins)
    if recal_model is not None:
        in_exp_proportions = recal_model.predict(exp_proportions)
    else:
        in_exp_proportions = exp_proportions
    residuals = y_pred - y_true

    # Get the inverse of the CDF at each of these depending on the prop_type.
    if prop_type == "interval":
        expected_sd_multiples = stats.norm(0, 1).ppf(0.5 + in_exp_proportions / 2.0)
        sd_multiples = np.abs(residuals) / y_std
    elif prop_type == "quantile":
        expected_sd_multiples = stats.norm(0, 1).ppf(in_exp_proportions)
        sd_multiples = residuals / y_std
    else:
        raise ValueError(f"Unknown prop_type {prop_type}")

    # For each bin edge, see how many of our data points deviate less than the
    # corresponding sd multiple.
    if vectorized:
        obs_proportions = (sd_multiples.reshape(-1, 1) <= expected_sd_multiples).mean(0)
    else:
        obs_proportions = np.array(
            [
                np.mean(sd_multiples <= expected_sd_multiples[i])
                for i in range(len(expected_sd_multiples))
            ]
        )

    # Calculate and return the area between these and the line y=x.
    miscal_area = miscalibration_area_from_proportions(exp_proportions, obs_proportions)
```

and the integration step (lines 320–330, verbatim):

```python
    areas = trapezoid_area(
        exp_proportions[:-1],
        exp_proportions[:-1],
        obs_proportions[:-1],
        exp_proportions[1:],
        exp_proportions[1:],
        obs_proportions[1:],
        absolute=True,
    )
    return areas.sum()
```

`trapezoid_area` lives in
`https://raw.githubusercontent.com/uncertainty-toolbox/uncertainty-toolbox/main/uncertainty_toolbox/utils.py`,
lines 77–104, verbatim:

```python
    # Differences
    dl = bl - al
    dr = br - ar

    # The ordering is the same for both iff they do not cross.
    cross = dl * dr < 0

    # Treat the degenerate case as a trapezoid
    cross = cross * (1 - ((dl == 0) * (dr == 0)))

    # trapezoid for non-crossing lines
    area_trapezoid = (xr - xl) * 0.5 * ((bl - al) + (br - ar))
    if absolute:
        area_trapezoid = np.abs(area_trapezoid)

    # Hourglass for crossing lines.
    with np.errstate(divide="ignore", invalid="ignore"):
        x_intersect = intersection((xl, bl), (xr, br), (xl, al), (xr, ar))[0]
    tl_area = 0.5 * (bl - al) * (x_intersect - xl)
    tr_area = 0.5 * (br - ar) * (xr - x_intersect)
    if absolute:
        area_hourglass = np.abs(tl_area) + np.abs(tr_area)
    else:
        area_hourglass = tl_area + tr_area

    # The nan_to_num function allows us to do 0 * nan = 0
    return (1 - cross) * area_trapezoid + cross * np.nan_to_num(area_hourglass)
```

The only reason that hourglass branch exists is to handle bins where the
observed curve crosses the y=x diagonal; if you approximate with
`np.trapz(np.abs(obs - exp), exp)` you get a very slightly different number.

### Self-contained reimplementation (no uncertainty-toolbox dependency)

```python
import numpy as np
from scipy import stats


def sharpness(y_std):
    return float(np.sqrt(np.mean(np.asarray(y_std) ** 2)))


def calibration_curve(y_pred, y_std, y_true, num_bins=100):
    """Expected vs observed coverage of centred prediction intervals."""
    exp_p = np.linspace(0, 1, num_bins)
    sd_mult = np.abs(np.asarray(y_pred) - np.asarray(y_true)) / np.asarray(y_std)
    expected_sd_mult = stats.norm(0, 1).ppf(0.5 + exp_p / 2.0)
    obs_p = (sd_mult.reshape(-1, 1) <= expected_sd_mult).mean(0)
    return exp_p, obs_p


def mean_absolute_calibration_error(y_pred, y_std, y_true, num_bins=100):
    exp_p, obs_p = calibration_curve(y_pred, y_std, y_true, num_bins)
    return float(np.mean(np.abs(exp_p - obs_p)))


def miscalibration_area(y_pred, y_std, y_true, num_bins=100):
    exp_p, obs_p = calibration_curve(y_pred, y_std, y_true, num_bins)
    return float(np.trapz(np.abs(obs_p - exp_p), exp_p))
```

Sanity check I ran on this reimplementation, n=3000, `y_true ~ N(0,1)`,
`y_pred = 0`:

```
sigma = 1.5 (over-cautious):  miscalibration_area = 0.1233
sigma = 1.0 (calibrated):     miscalibration_area = 0.0056
```

Note `exp_p[0] == 0` gives an interval of zero width, so `obs_p[0] == 0` — the
curve is anchored at the origin as it should be.

---

## 5. VBLL — which term is which

**Source read:** `https://raw.githubusercontent.com/VectorInstitute/vbll/main/vbll/layers/regression.py`
`class Regression`, lines 100–122, verbatim:

```python
    def W(self):
        cov_diag = torch.exp(self.W_logdiag)
        if self.W_dist == Normal:
            cov = self.W_dist(self.W_mean, cov_diag)
        elif (self.W_dist == DenseNormal) or (self.W_dist == DenseNormalPrec):
            tril = torch.tril(self.W_offdiag, diagonal=-1) + torch.diag_embed(cov_diag)
            cov = self.W_dist(self.W_mean, tril)
        elif self.W_dist == LowRankNormal:
            cov = self.W_dist(self.W_mean, self.W_offdiag, cov_diag)

        return cov

    def noise(self):
        return Normal(self.noise_mean, torch.exp(self.noise_logdiag))
...
    def predictive(self, x):
        return (self.W() @ x[..., None]).squeeze(-1) + self.noise()
```

That one `predictive` line is the entire decomposition. Reading the two
operators it uses, from
`https://raw.githubusercontent.com/VectorInstitute/vbll/main/vbll/utils/distributions.py`:

`DenseNormal.__matmul__`, lines 217–221, verbatim — **this is the epistemic
term**, the feature-map quadratic form `x^T Sigma_W x`:

```python
    def __matmul__(self, inp):
        assert inp.shape[-2] == self.loc.shape[-1]
        assert inp.shape[-1] == 1
        new_cov = self.covariance_weighted_inner_prod(inp.unsqueeze(-3), reduce_dim = False)
        return Normal(self.loc @ inp, torch.sqrt(torch.clip(new_cov, min = 1e-12)))
```

with (lines 207–210):

```python
    def covariance_weighted_inner_prod(self, b, reduce_dim=True):
        assert b.shape[-1] == 1
        prod = ((tp(self.scale_tril) @ b)**2).sum(-2)
        return prod.squeeze(-1) if reduce_dim else prod
```

`Normal.__add__`, lines 160–167, verbatim — **this adds the aleatoric term**,
the learned noise covariance:

```python
    def __add__(self, inp):
        if isinstance(inp, Normal):
            new_cov =  self.var + inp.var
            return Normal(self.mean + inp.mean, torch.sqrt(torch.clip(new_cov, min = 1e-12)))
        elif isinstance(inp, torch.Tensor):
            return Normal(self.mean + inp, self.scale)
        else:
            raise NotImplementedError('Distribution addition only implemented for diag covs')
```

So to extract the two halves from a VBLL `Regression` layer, given features `x`:

```python
# epistemic: x^T Sigma_W x        aleatoric: learned homoscedastic noise variance
Wx        = (layer.W() @ x[..., None]).squeeze(-1)     # a vbll Normal
epistemic = Wx.var                                     # == Wx.scale ** 2
aleatoric = layer.noise().var                          # == exp(2 * noise_logdiag)
total     = layer.predictive(x).variance               # == epistemic + aleatoric
```

Two things to know:

- `noise_logdiag` is a plain `nn.Parameter` of shape `(out_features,)`
  (line 82: `self.noise_logdiag = nn.Parameter(torch.randn(out_features) * (np.log(wishart_scale)))`),
  so **VBLL's aleatoric term is homoscedastic** — one number per output
  dimension, not per molecule. It cannot express input-dependent noise. If your
  noise-injection experiment varies sigma per molecule, VBLL will not track it,
  and that alone can explain a coverage anomaly.
- `class tRegression` is different: it infers the noise variance and returns a
  Student-t. Lines 241–246, verbatim:

  ```python
      def predictive(self, x):
          dof = 2 * self.noise.concentration
          Wx = (self.W @ x[..., None]).squeeze(-1)
          mean = Wx.mean
          pred_cov = (Wx.variance + 1) * self.noise.rate / self.noise.concentration
          return torch.distributions.studentT.StudentT(dof, mean, torch.sqrt(pred_cov))
  ```

  Here the split is not a clean sum: `Wx.variance` (epistemic, unitless because
  it is scaled by the noise) and the `+1` (aleatoric) are both multiplied by
  `rate/concentration`, the posterior mean noise variance.

---

## 6. Sources — every URL actually fetched (HTTP 200, read in full or in part)

- `https://raw.githubusercontent.com/SeongokRyu/uq_molecule/master/train_cep.py`
- `https://raw.githubusercontent.com/SeongokRyu/uq_molecule/master/mc_dropout.py`
- `https://raw.githubusercontent.com/SeongokRyu/uq_molecule/master/blocks.py`
- `https://raw.githubusercontent.com/SeongokRyu/uq_molecule/master/train_zinc.py`
- `https://raw.githubusercontent.com/chemprop/chemprop/v1.7.1/chemprop/uncertainty/uncertainty_predictor.py`
- `https://raw.githubusercontent.com/chemprop/chemprop/v1.7.1/chemprop/train/loss_functions.py`
- `https://raw.githubusercontent.com/chemprop/chemprop/main/chemprop/uncertainty/estimator.py`
- `https://raw.githubusercontent.com/chemprop/chemprop/main/chemprop/uncertainty/calibrator.py`
- `https://raw.githubusercontent.com/chemprop/chemprop/main/chemprop/nn/predictors.py`
- `https://raw.githubusercontent.com/chemprop/chemprop/main/chemprop/nn/metrics.py`
- `https://raw.githubusercontent.com/gscalia/chemprop/uncertainty/chemprop/train/predict.py`
- `https://raw.githubusercontent.com/gscalia/chemprop/uncertainty/chemprop/utils.py`
- `https://raw.githubusercontent.com/gscalia/chemprop/uncertainty/chemprop/models/model.py`
- `https://raw.githubusercontent.com/uncertainty-toolbox/uncertainty-toolbox/main/uncertainty_toolbox/metrics_calibration.py`
- `https://raw.githubusercontent.com/uncertainty-toolbox/uncertainty-toolbox/main/uncertainty_toolbox/utils.py`
- `https://raw.githubusercontent.com/VectorInstitute/vbll/main/vbll/layers/regression.py`
- `https://raw.githubusercontent.com/VectorInstitute/vbll/main/vbll/utils/distributions.py`
- `https://raw.githubusercontent.com/zillow/quantile-forest/main/quantile_forest/_quantile_forest.py`
- sklearn `1.3.2` installed locally — `sklearn.ensemble._forest._generate_sample_indices`
  and `_get_n_samples_bootstrap` read via `inspect.getsource`

## 7. Paywalled / could not reach

- **Scalia, Grambow, Pernici, Li & Green 2020, JCIM**,
  `https://pubs.acs.org/doi/10.1021/acs.jcim.9b00975` — **PAYWALLED, HTTP 403.**
  I did not read the paper. The preprint is open at
  `https://arxiv.org/abs/1910.03127` (also not fetched), and their **code is
  available and I did read it**: `github.com/gscalia/chemprop`, branch
  `uncertainty` — see §1d, §2a, §2b.
- **Ryu, Kwon & Kim 2019, Chem. Sci.**,
  `https://pubs.rsc.org/en/content/articlelanding/2019/sc/c9sc01992h` —
  **HTTP 403, could not fetch.** Everything quoted for Ryu comes from their
  GitHub repo, not the paper.
- `github.com/seongokryu/uq-molecule` — **does not exist (404).** The correct
  repo is `github.com/SeongokRyu/uq_molecule` (underscore, different
  capitalisation).
- `chemprop/nn/loss.py` on `main` — **404.** In current Chemprop v2 the loss
  functions live in `chemprop/nn/metrics.py`, which I read instead.
- I did not read the compiled Cython in `quantile-forest`
  (`_quantile_forest_fast.pyx`, `_utils.pyx`); the leaf-mapping logic I quoted
  is from the Python file only.
