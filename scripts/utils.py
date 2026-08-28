from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)
from scipy.stats import pearsonr
import numpy as np
import os
import csv
import shap
import pandas as pd

import torch
from torch_geometric.data import Data
from rdkit import Chem

RESULT_COLUMNS = ["sigma", "iteration", "model", "rep", "sample_size", "mae",
                  "mse", "rmse", "r2", "pearson_corr", "params_source",
                  "loss_function", "spec_version", "spec_hash", "gp_fit_method",
                  "gp_collapsed",
                  # Added 2026-08-27. The condition a row belongs to used to
                  # survive only in the output FILENAME, so the figure script
                  # recovered it by matching the stem against a list of six
                  # retired names -- under which a new-scheme file either matched
                  # nothing (and every condition for a (model, rep, level,
                  # replicate) collapsed to one row on a blank key) or matched
                  # `outlier`, which used to mean something else entirely
                  # (RERUN_PLAN.md 2.11).
                  "noise_type",
                  # Added 2026-08-27. What the number in `sigma` MEASURES. The
                  # column holds three different physical quantities across the
                  # two pipelines -- a fraction of the clean training label
                  # spread on QM9, raw log units on the experimental side
                  # (anchored to published assay error), and a fraction of
                  # labels clipped on the censoring axis, which is not a noise
                  # amount at all. auc_norm is mean retention over each
                  # configuration's own level range, so two auc_norm values on
                  # one axis compare retention over different spans unless the
                  # units agree (RERUN_PLAN.md 2.12).
                  "level_units",
                  # Added 2026-08-27. What the noise actually DELIVERED at this
                  # level, in raw label units, as the injector measured it. This
                  # is what puts two pipelines' levels on one axis: the
                  # predecessor this study follows (Kolmar & Grulke, J Cheminform
                  # 13:92, 2021) doses per dataset as a fraction of that
                  # dataset's endpoint range and then compares datasets with the
                  # noise divided by the noise-free baseline error, sigma/RMSE0,
                  # against RMSE/RMSE0. Without the delivered amount on the row,
                  # that axis cannot be built after the fact.
                  "delivered_dose",
                  # Added 2026-08-27. The noise manifest is keyed on
                  # (iteration, file_no, noise_level) and the results row carried
                  # no file_no, so a results row could not be joined to the
                  # provenance of the noise that produced it -- and file_no is
                  # what separates two replicates that share a level
                  # (RERUN_PLAN.md 2.13).
                  "file_no",
                  # Added 2026-08-28. What the labels were standardised by, so
                  # the error columns beside them can be put back on either
                  # scale. `mae` and `rmse` are now in the label's own units on
                  # both pipelines; these two say what that conversion was, and
                  # blank means none was recorded and the numbers are in
                  # whatever units the labels arrived in (RERUN_PLAN.md 2.18).
                  "standardisation_mean", "standardisation_sd"]

# The three things a level can measure.
LEVEL_UNITS = ('label_sd', 'raw_label', 'fraction_censored')


# What every row written from here belongs to. process_and_train sets both once
# per (noise level, replicate) from the manifest the injector wrote; the
# condition name is never guessed from a filename and never composed here from
# the CLI flags, because a second implementation of the naming is a second thing
# to drift.
_CURRENT_NOISE_TYPE = None
_CURRENT_LEVEL_UNITS = None
_CURRENT_DELIVERED_DOSE = None


_CURRENT_STANDARDISATION = (None, None)
_CURRENT_FILE_NO = None


def set_current_noise_type(name, level_units=None, delivered_dose=None,
                           standardisation=None, file_no=None):
    """Record which condition, in what units, and how much, from now on."""
    global _CURRENT_NOISE_TYPE, _CURRENT_LEVEL_UNITS, _CURRENT_DELIVERED_DOSE
    global _CURRENT_STANDARDISATION, _CURRENT_FILE_NO
    _CURRENT_FILE_NO = file_no
    _CURRENT_NOISE_TYPE = None if name in (None, '') else str(name)
    _CURRENT_DELIVERED_DOSE = delivered_dose
    _CURRENT_STANDARDISATION = tuple(standardisation or (None, None))
    if level_units is not None and level_units not in LEVEL_UNITS:
        raise ValueError(
            f"level_units={level_units!r} is not one of {LEVEL_UNITS}; a row "
            f"whose level units are unnamed cannot be put on an axis with "
            f"another pipeline's.")
    _CURRENT_LEVEL_UNITS = level_units


def current_noise_type():
    return _CURRENT_NOISE_TYPE


def current_level_units():
    return _CURRENT_LEVEL_UNITS


def current_delivered_dose():
    return _CURRENT_DELIVERED_DOSE


def current_standardisation():
    """(mean, spread) the labels were standardised by, or (None, None)."""
    return _CURRENT_STANDARDISATION


def save_results(filepath, s, iteration, model, rep, n, metrics, params_source='default',
                 loss_function='mse', gp_fit_method='', gp_collapsed='',
                 noise_type=None, level_units=None, delivered_dose=None,
                 file_no=None):
    """
    Save results to a CSV file with loss function tracking.

    Three provenance columns were added on 2026-08-26 (Chat E, RERUN_PLAN.md
    section 5.2): the version and content hash of models/model_defaults.py, so a
    row can always be traced to the parameters that produced it, and which
    optimiser actually fitted the Gaussian process -- the Adam fallback is a
    different fit and used to leave no trace on disk.

    Appending to a file written with the OLD header would produce ragged rows
    that read back silently wrong, so that is a hard error rather than a
    surprise three months later.
    """
    if not filepath:
        return

    file_exists = os.path.isfile(filepath)
    if file_exists:
        with open(filepath, newline='') as f:
            header = next(csv.reader(f), [])
        if header and header != RESULT_COLUMNS:
            missing = [c for c in RESULT_COLUMNS if c not in header]
            raise RuntimeError(
                f"{filepath} was written with a different column set and cannot "
                f"be appended to. Missing: {missing}. Move or delete the old "
                f"file -- appending would produce rows that do not line up with "
                f"the header.")

    try:
        from model_defaults import SPEC_VERSION, spec_hash
        spec_version, spec = SPEC_VERSION, spec_hash()
    except Exception:
        spec_version, spec = '', ''

    with open(filepath, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(RESULT_COLUMNS)
        writer.writerow([s, iteration, model, rep, n, metrics[0], metrics[1],
                         metrics[2], metrics[3], metrics[4], params_source,
                         loss_function, spec_version, spec, gp_fit_method,
                         gp_collapsed,
                         noise_type if noise_type is not None
                         else (_CURRENT_NOISE_TYPE or ''),
                         level_units if level_units is not None
                         else (_CURRENT_LEVEL_UNITS or ''),
                         delivered_dose if delivered_dose is not None
                         else ('' if _CURRENT_DELIVERED_DOSE is None
                               else _CURRENT_DELIVERED_DOSE),
                         file_no if file_no is not None
                         else ('' if _CURRENT_FILE_NO is None
                               else _CURRENT_FILE_NO),
                         '' if _CURRENT_STANDARDISATION[0] is None
                         else _CURRENT_STANDARDISATION[0],
                         '' if _CURRENT_STANDARDISATION[1] is None
                         else _CURRENT_STANDARDISATION[1]])

def calculate_regression_metrics(y_test, prediction, logging=False):
    """The five metrics, with the error ones in the LABEL'S OWN UNITS.

    Every QM9 label is standardised in the injector and nothing turned it back,
    so `rmse` and `mae` came out in label standard deviations while the same two
    columns on the experimental side were in log units -- two quantities under
    one name, in two files that are meant to be read together (RERUN_PLAN.md
    2.18).

    The convention is the label's own units. QM9 work reports each target's error
    in eV or meV against chemical accuracy at 0.043 eV, and uses a standardised
    error only when averaging ACROSS the twelve targets, whose units cannot
    otherwise be pooled (Godwin et al., arXiv:2106.07971). This study trains one
    target at a time, so that reason does not apply.

    Multiplying afterwards is exact rather than a shortcut: the labels and the
    predictions were shifted and scaled by the SAME constants, so the offset
    cancels in all five metrics and the spread cancels in r2 and the correlation.
    scripts/test_metric_units.py asserts that against arrays converted directly.

    When no standardisation was recorded the numbers are returned as they are,
    and the two columns beside them on the row are blank -- which is what says so.
    """
    mae = mean_absolute_error(y_test, prediction)
    mse = mean_squared_error(y_test, prediction)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, prediction)
    pearson_corr, _ = pearsonr(y_test, prediction)

    _, sd = current_standardisation()
    if sd is not None and float(sd) > 0:
        sd = float(sd)
        mae, mse, rmse = mae * sd, mse * sd * sd, rmse * sd

    # Optionally log the metrics
    if logging:
        print("Mean Absolute Error:", mae)
        print("Mean Squared Error:", mse)
        print("RMSE:", rmse)
        print("R-squared:", r2)
        print("Pearson Correlation:", pearson_corr)

    return mae, mse, rmse, r2, pearson_corr


# ============================================================================
# UNCERTAINTY DECOMPOSITION -- ONE DEFINITION, IMPORTED
# ============================================================================
#
# There is nothing to compute here. Five local definitions of the split used to
# sit at this point in the file and they disagreed with one another and with the
# laboratory runner, which had none at all -- failure mode 10 in RERUN_PLAN.md
# 0.6. They are deleted (the delete list in RERUN_PLAN.md 5.5) and both
# pipelines now import scripts/uncertainty_decomposition.py.
#
# What each of the five did, and what replaced it:
#
#   decompose_uncertainty_sampling             -> decompose_sampling
#       hard-coded the aleatoric term to None and set the total equal to the
#       model term, so every coverage number computed from that "total" silently
#       omitted observation noise.
#   decompose_uncertainty_sampling_heteroscedastic -> decompose_sampling
#       right in shape, but returned standard deviations, which the callers then
#       added in quadrature by hand at each site.
#   decompose_uncertainty_gp                   -> decompose_gp
#   decompose_uncertainty_vbll                 -> decompose_sampling
#       both broadcast one number onto every molecule without recording that
#       they had, so a column that cannot correlate with anything per-molecule
#       was indistinguishable from one that can.
#   decompose_uncertainty_distributional       -> decompose_single_distribution
#       hard-coded the model term to None. Its premise is false for the quantile
#       forest -- a forest IS an ensemble and the spread across its trees is a
#       legitimate model-uncertainty estimate, which is what the forest split now
#       computes.
#
# EVERY value in and out of that module is a VARIANCE. The single conversion to
# a standard deviation happens in save_uncertainty_values below, at the point of
# writing, and nowhere else.
from uncertainty_decomposition import (  # noqa: E402
    assert_matches_support, support, variance_to_std)


def split_predictive_head(output, loss_name):
    """Split a wide network output into the prediction and the per-molecule variance.

    Two losses make the network predict how uncertain it is about each molecule
    as well as the value, and that second output was sliced off and dropped at
    every prediction site for the life of the project -- so both models reported
    the spread over their stochastic passes, which is what an ordinary network
    reports, and the per-molecule variance they exist to produce never reached a
    file (RERUN_PLAN.md 2.17).

    The transforms here are the ones in `scripts/loss_functions.py`, because that
    is what was FITTED. A different transform would report a variance the network
    never learned:

      heteroscedastic  outputs (mean, log_var), log_var clamped to [-10, 10] in
                       the loss. Variance is exp(log_var).
      evidential       outputs (gamma, v, alpha, beta) of a Normal-Inverse-Gamma,
                       with softplus on the last three and +1 on v and alpha. Its
                       aleatoric variance is beta / (alpha - 1).

    Returns (prediction, variance). `variance` is None for every other loss, and
    for those the output is returned unchanged.
    """
    if loss_name == 'heteroscedastic':
        mean = output[:, 0:1]
        log_var = np.clip(output[:, 1:2], -10.0, 10.0)
        return mean, np.exp(log_var)
    if loss_name == 'evidential':
        gamma = output[:, 0:1]
        alpha = _softplus(output[:, 2:3]) + 1.0
        beta = _softplus(output[:, 3:4])
        return gamma, beta / (alpha - 1.0)
    return output, None


def _softplus(x):
    """log(1 + exp(x)), computed the way that does not overflow."""
    x = np.asarray(x, dtype=float)
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


# ============================================================================
# SAVE UNCERTAINTY WITH DECOMPOSITION
# ============================================================================

UNCERTAINTY_COLUMNS = [
    "model", "representation", "sigma", "iteration", "file_no", "sample_idx",
    "y_pred_mean", "y_pred_std_uncalibrated", "y_true_original", "y_true_noisy",
    "injected_noise", "y_pred_std_calibrated", "temperature",
    "epistemic_uncertainty", "aleatoric_uncertainty",
    # Added 2026-08-28. Whether each component VARIES PER MOLECULE or is one
    # number per fit copied onto every row. Nothing recorded this, and it is the
    # distinction that decides whether a correlation against per-molecule
    # injected noise means anything: a constant term correlates at exactly zero
    # however good the model is, so a null read off such a column is a property
    # of the model and not a finding (RERUN_PLAN.md 5.5, 5.5a point 5). Values
    # are 'per_molecule', 'constant' or 'none', from the SUPPORT table in
    # scripts/uncertainty_decomposition.py, and assert_matches_support has
    # already refused to write the row if the numbers disagree with the label.
    "aleatoric_support", "epistemic_support",
    # Added 2026-08-26 (Chat F, agent C). split/canonical_smiles/noise_scale/
    # noise_pattern/noise_pattern_pred/oof_folds_ok mirror the KIRBy schema so one
    # analysis module can read both producers.
    "split", "canonical_smiles", "noise_scale", "noise_pattern",
    "noise_pattern_pred", "oof_folds_ok",
    # Added 2026-08-27. y_true_original is the RAW label; y_pred_mean,
    # y_pred_std_* and y_true_noisy are on the STANDARDISED scale, because the
    # Rust writer standardises with the clean training mean and spread before the
    # models ever see a label. Subtracting one from the other is wrong by exactly
    # these two numbers, and the figure script carried a workaround for it. They
    # are on the row now, so the file says what its own columns mean
    # (RERUN_PLAN.md 2.13).
    "standardisation_mean", "standardisation_sd",
    # Added 2026-08-27: the condition's registry name, so a row can be conditioned
    # on its noise type rather than the type being guessed from the file name.
    "noise_type",
]

# 'validation' was added 2026-08-28. A validation molecule satisfies the two
# things a scored molecule has to satisfy -- no model fitted it, and the
# injector recorded its noise draw -- so it can answer the same question a
# train_oof row answers, without the extra fits (RERUN_PLAN.md 13 chat O).
VALID_SPLITS = ("test", "train_oof", "validation")

#: Splits whose labels were corrupted, so the recorded draw is mandatory.
CORRUPTED_SPLITS = ("train_oof", "validation")


def _per_row_values(value, n, name):
    """Broadcast a scalar, or validate a per-molecule sequence, to length n.

    Values are passed through WITHOUT dtype coercion so a float64 array reaches the
    CSV bit-for-bit. None yields NaN for every row.
    """
    if value is None:
        return [np.nan] * n
    if np.isscalar(value) or isinstance(value, (str, bytes)):
        return [value] * n
    seq = np.asarray(value)
    if seq.ndim == 0:
        return [seq.item()] * n
    if len(seq) != n:
        raise ValueError(
            f"save_uncertainty_values: '{name}' has length {len(seq)} but there are "
            f"{n} molecules. A per-molecule column that does not line up with the "
            f"predictions would be silently mis-attributed.")
    return list(seq)


def save_uncertainty_values(y_pred_mean, y_pred_std, y_true_original, y_true_noisy,
                           filepath, model_name, rep, sigma_noise, iteration, file_no,
                           y_pred_std_calibrated=None, temperature=None,
                           aleatoric_var=None, epistemic_var=None,
                           support_model=None, loss_name=None,
                           support_blocks=None,
                           split='test', injected_noise=None, canonical_smiles=None,
                           noise_scale=None, noise_pattern=None,
                           noise_pattern_pred=None, oof_folds_ok=None,
                           noise_type=None):
    """
    Save per-molecule uncertainty values with optional epistemic/aleatoric decomposition.

    injected_noise is RECORDED, never reconstructed. Until 2026-08-26 this function
    regressed the noisy label on the clean one and kept the residuals; held-out labels
    are no longer noised, so those residuals were identically zero and the column was
    dead. The regression is gone.

    Args:
        split: 'test', 'train_oof' or 'validation'. A 'train_oof' row is scored by a
            model fitted without that molecule, so y_true_original is the CLEAN label
            and the label the model was trained on is y_true_original + injected_noise.
            A 'validation' row carries corrupted labels the same way, but no model
            fitted it in the first place -- it is held out of every fit and scored
            once, by the outer model, with no inner folds.
        injected_noise: the value the injector RECORDED for each molecule, written
            verbatim. When it is omitted, a test row gets exactly 0.0 and a train_oof
            or validation row raises -- a row whose label was corrupted, without its
            recorded noise, cannot answer anything about whether uncertainty finds the
            corrupted labels.
        canonical_smiles: the molecule identifier. sample_idx is a row position and
            cannot link a molecule across models or noise levels.
        noise_scale: per-molecule amount of noise actually applied.
        noise_pattern: the shape of the noise at a fixed reference level of 1.0,
            identical at every noise level including zero. Subtracting the zero-level
            correlation is what removes the label-magnitude confound.
        noise_pattern_pred: the same shape recomputed from the model's own PREDICTED
            label, as a ceiling on what counts as detection.
        oof_folds_ok: how many inner folds produced a value. Test and validation
            rows get -1: neither was cross-fitted.
        aleatoric_var, epistemic_var: the two components, as VARIANCES, from
            scripts/uncertainty_decomposition.py. This is the ONE place in the
            QM9 pipeline where a variance becomes a standard deviation, which is
            what the two columns have always held; the callers used to convert
            at fourteen different sites and two of them added standard
            deviations in quadrature by hand (RERUN_PLAN.md 5.5, the
            standardisation trap, part 3). None means the model has no such
            component -- which is not the same as zero, and the column says which.
        support_model: the name to look up in the SUPPORT table when the name
            WRITTEN on the row is decorated (het_gp_rbf against the roster's
            heteroscedastic_gp). Defaults to model_name.
        loss_name: the loss the model was fitted with. Two of them give a network
            a per-molecule variance head, which changes what SUPPORT expects.
        support_blocks: which FIT each row came from. Out-of-fold rows come from
            several fits, so a component that is one number per fit takes a
            different value in each fold; without this the guard would read that
            as a term varying per molecule and stop a correct run.
    """
    if split not in VALID_SPLITS:
        raise ValueError(
            f"save_uncertainty_values: split={split!r} is not one of {VALID_SPLITS}.")

    uncertainty_file = filepath.replace('.csv', '_uncertainty_values.csv')

    n = len(y_pred_mean)


    if injected_noise is None:
        if split in CORRUPTED_SPLITS:
            raise ValueError(
                f"save_uncertainty_values: split={split!r} requires injected_noise. "
                "A row whose label WAS corrupted, but which does not carry the noise "
                "the injector recorded for it, cannot answer whether uncertainty "
                "tracks the corruption; it must not be reconstructed from the labels.")
        injected_noise_rows = [0.0] * n
    else:
        injected_noise_rows = _per_row_values(injected_noise, n, 'injected_noise')

    if oof_folds_ok is None:
        # -1 is 'this row was not cross-fitted'. It is the right value for a
        # validation row as well as a test row: both are scored once, by the
        # outer fit, and neither came from an inner fold.
        oof_folds_ok_rows = ([-1] * n if split in ('test', 'validation')
                             else [np.nan] * n)
    else:
        oof_folds_ok_rows = _per_row_values(oof_folds_ok, n, 'oof_folds_ok')

    smiles_rows = _per_row_values(canonical_smiles, n, 'canonical_smiles')
    noise_scale_rows = _per_row_values(noise_scale, n, 'noise_scale')
    noise_pattern_rows = _per_row_values(noise_pattern, n, 'noise_pattern')
    noise_pattern_pred_rows = _per_row_values(
        noise_pattern_pred, n, 'noise_pattern_pred')

    # THE GUARD, before anything reaches disk. A component declared per molecule
    # that is in fact one number copied onto every row -- or the reverse -- is
    # the failure that has cost this project most, and it is invisible in the
    # output file. assert_matches_support raises rather than writing a column
    # nobody can interpret (RERUN_PLAN.md 5.5b).
    _support_key = support_model or model_name
    assert_matches_support(_support_key, aleatoric_var, epistemic_var,
                           n_molecules=n, loss_name=loss_name,
                           blocks=support_blocks)
    _alea_kind, _epis_kind = support(_support_key, loss_name)
    # The single conversion. Variances add; standard deviations do not, so the
    # arithmetic is all done upstream in variance space and this is the last
    # step before the numbers are written.
    aleatoric_uncertainty = variance_to_std(aleatoric_var)
    epistemic_uncertainty = variance_to_std(epistemic_var)

    rows = []
    for i in range(n):
        row = {
            'model': model_name,
            'representation': rep,
            'sigma': sigma_noise,
            'iteration': iteration,
            'file_no': file_no,
            'sample_idx': i,
            'y_pred_mean': y_pred_mean[i],
            'y_pred_std_uncalibrated': y_pred_std[i],
            'y_true_original': y_true_original[i],
            'y_true_noisy': y_true_noisy[i],
            'injected_noise': injected_noise_rows[i],
            # What separates y_true_original (raw) from the standardised
            # columns beside it.
            'standardisation_mean': ('' if _CURRENT_STANDARDISATION[0] is None
                                     else _CURRENT_STANDARDISATION[0]),
            'standardisation_sd': ('' if _CURRENT_STANDARDISATION[1] is None
                                   else _CURRENT_STANDARDISATION[1]),
        }

        # Add calibrated values if provided
        if y_pred_std_calibrated is not None:
            row['y_pred_std_calibrated'] = y_pred_std_calibrated[i]
            row['temperature'] = temperature
        else:
            # No calibration - set calibrated = uncalibrated
            row['y_pred_std_calibrated'] = y_pred_std[i]
            row['temperature'] = 1.0

        # Add decomposition if provided
        if epistemic_uncertainty is not None:
            row['epistemic_uncertainty'] = epistemic_uncertainty[i]
        else:
            row['epistemic_uncertainty'] = np.nan

        if aleatoric_uncertainty is not None:
            row['aleatoric_uncertainty'] = aleatoric_uncertainty[i]
        else:
            row['aleatoric_uncertainty'] = np.nan

        row['aleatoric_support'] = _alea_kind
        row['epistemic_support'] = _epis_kind

        row['split'] = split
        row['canonical_smiles'] = smiles_rows[i]
        row['noise_scale'] = noise_scale_rows[i]
        row['noise_pattern'] = noise_pattern_rows[i]
        row['noise_pattern_pred'] = noise_pattern_pred_rows[i]
        row['oof_folds_ok'] = oof_folds_ok_rows[i]
        # The condition this row was produced under. Without it, two noise types
        # land in one column with nothing to separate them, and every statistic
        # computed over the file pools across a dimension it must condition on.
        row['noise_type'] = noise_type

        rows.append(row)

    df = pd.DataFrame(rows, columns=UNCERTAINTY_COLUMNS)

    if os.path.exists(uncertainty_file):
        with open(uncertainty_file, newline='') as f:
            header = next(csv.reader(f), [])
        if header and header != UNCERTAINTY_COLUMNS:
            missing = [c for c in UNCERTAINTY_COLUMNS if c not in header]
            raise RuntimeError(
                f"{uncertainty_file} was written with a different column set and "
                f"cannot be appended to. Missing: {missing}. Move or delete the old "
                f"file -- appending would produce rows that do not line up with the "
                f"header.")
        df.to_csv(uncertainty_file, mode='a', header=False, index=False)
    else:
        df.to_csv(uncertainty_file, mode='w', header=True, index=False)


def save_conformal_intervals(y_pred, y_lower, y_upper, y_true, filepath, model_name, rep, sigma_noise, iteration, file_no, alpha):
    # Create directory if it doesn't exist
    intervals_dir = os.path.join(os.path.dirname(filepath), "conformal_intervals")
    os.makedirs(intervals_dir, exist_ok=True)
    
    # Create dataframe with results
    results_df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'y_lower': y_lower,
        'y_upper': y_upper,
        'interval_width': y_upper - y_lower,
        'coverage': ((y_true >= y_lower) & (y_true <= y_upper)).astype(int),
        'alpha': alpha,
        'model_name': model_name,
        'rep': rep,
        'sigma_noise': sigma_noise,
        'iteration': iteration,
        'file_no': file_no
    })
    
    # Calculate empirical coverage
    empirical_coverage = results_df['coverage'].mean()
    results_df['empirical_coverage'] = empirical_coverage
    
    # Save to file following your naming convention
    filename = f"conformal_intervals_{model_name}_{rep}_sigma{sigma_noise}_iter{iteration}_file{file_no}.csv"
    filepath_full = os.path.join(intervals_dir, filename)
    results_df.to_csv(filepath_full, index=False)
    
    print(f"Conformal intervals saved. Target coverage: {1-alpha:.1%}, Empirical coverage: {empirical_coverage:.1%}")

def save_per_epoch_metrics(train_losses, val_losses, filepath, model_name, rep, sigma_noise, iteration, file_no):
   """
   Save per-epoch training and validation metrics to a CSV file.
   """
   if filepath:
       # Create per-epoch specific filepath
       base_path = filepath.replace('.csv', '_per_epoch.csv')
       file_exists = os.path.isfile(base_path)
       
       with open(base_path, mode='a', newline='') as f:
           writer = csv.writer(f)
           
           # Write header if the file is new
           if not file_exists:
               writer.writerow(["sigma", "iteration", "model", "rep", "file_no", "epoch", "train_loss", "val_loss"])
           
           # Save metrics for each epoch
           for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
               writer.writerow([sigma_noise, iteration, model_name, rep, file_no, epoch, train_loss, val_loss])

def calibrate_uncertainty_simple(y_pred_mean, y_pred_std, y_true):
    """
    Find optimal temperature T for variance scaling.
    Works for any model that outputs mean and std.
    
    Args:
        y_pred_mean: predicted means (numpy array)
        y_pred_std: predicted stds (numpy array)
        y_true: true values (numpy array)
    
    Returns:
        float: optimal temperature T
    """
    from scipy.optimize import minimize_scalar
    
    def nll(T):
        scaled_std = np.maximum(y_pred_std * T, 1e-6)
        return (0.5 * np.log(2 * np.pi * scaled_std**2) + 
                0.5 * ((y_true - y_pred_mean)**2 / scaled_std**2)).mean()
    
    result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
    return result.x


def save_calibration_metadata(filepath, model_name, rep, sigma_noise, iteration, 
                              n_train, n_cal, n_val, n_test, alpha_list):
    """Save calibration set size and split information"""
    metadata_dir = os.path.join(os.path.dirname(filepath), "conformal_metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    
    metadata_df = pd.DataFrame({
        'model_name': [model_name],
        'rep': [rep],
        'sigma_noise': [sigma_noise],
        'iteration': [iteration],
        'n_train': [n_train],
        'n_calibration': [n_cal],
        'n_validation': [n_val],
        'n_test': [n_test],
        'cal_pct_of_total': [n_cal / (n_train + n_cal + n_val + n_test) * 100],
        'alphas_tested': [str(alpha_list)]
    })
    
    filename = f"calibration_metadata_{model_name}_{rep}_sigma{sigma_noise}_iter{iteration}.csv"
    filepath_full = os.path.join(metadata_dir, filename)
    metadata_df.to_csv(filepath_full, index=False)

"""
Graph Utilities for Molecular Property Prediction

Converts SMILES to PyTorch Geometric Data objects with proper features.
"""

def smiles_to_graph(smiles, y_value=None):
    """
    Convert SMILES to PyTorch Geometric Data object.
    
    Args:
        smiles: SMILES string
        y_value: Optional target value
        
    Returns:
        torch_geometric.data.Data or None if invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    
    # Node features: [atomic_num, degree, formal_charge, is_aromatic, hybridization, num_h]
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetIsAromatic()),
            int(atom.GetHybridization()),
            atom.GetTotalNumHs()
        ])
    
    x = torch.tensor(atom_features, dtype=torch.float)
    
    # Edge indices
    edge_index = []
    edge_attr = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        # Bond features: [bond_type, is_conjugated, is_in_ring]
        bond_features = [
            float(bond.GetBondTypeAsDouble()),
            int(bond.GetIsConjugated()),
            int(bond.IsInRing())
        ]
        
        # Add both directions (undirected graph)
        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append(bond_features)
        edge_attr.append(bond_features)
    
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 3), dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    if y_value is not None:
        data.y = torch.tensor([y_value], dtype=torch.float)
    
    return data


def smiles_list_to_graphs(smiles_list, y_values=None):
    """
    Convert list of SMILES to list of PyG graphs.
    
    Args:
        smiles_list: List of SMILES strings
        y_values: Optional array of target values
        
    Returns:
        List of Data objects (None for invalid SMILES)
    """
    graphs = []
    failed = 0
    
    for i, smiles in enumerate(smiles_list):
        y_val = y_values[i] if y_values is not None else None
        graph = smiles_to_graph(smiles, y_val)
        
        if graph is None:
            failed += 1
        
        graphs.append(graph)
    
    if failed > 0:
        print(f"Warning: {failed}/{len(smiles_list)} molecules failed conversion")
    
    return graphs


def create_graph_loaders(train_graphs, test_graphs, val_graphs, 
                        y_train_noisy, y_test_noisy, y_val_noisy,
                        batch_size=32):
    """
    Create PyG DataLoaders with noisy targets attached.
    
    This properly handles the noisy targets by attaching them to the graphs.
    
    Args:
        train_graphs, test_graphs, val_graphs: Lists of Data objects
        y_train_noisy, y_test_noisy, y_val_noisy: Noisy target arrays
        batch_size: Batch size
        
    Returns:
        train_loader, test_loader, val_loader
    """
    from torch_geometric.loader import DataLoader
    
    # Filter out None graphs and attach noisy targets
    def attach_targets(graphs, y_noisy):
        valid_data = []
        for graph, y_val in zip(graphs, y_noisy):
            if graph is not None:
                graph.y = torch.tensor([y_val], dtype=torch.float)
                valid_data.append(graph)
        return valid_data
    
    train_data = attach_targets(train_graphs, y_train_noisy)
    test_data = attach_targets(test_graphs, y_test_noisy)
    val_data = attach_targets(val_graphs, y_val_noisy)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, val_loader


def extract_graph_features_for_kernel(smiles_list):
    """
    Extract graph structure for kernel computation.
    
    Returns node labels and edge lists suitable for grakel/graph kernels.
    
    Args:
        smiles_list: List of SMILES strings
        
    Returns:
        List of tuples (node_labels_dict, edge_list)
    """
    graphs = []
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumBonds() == 0:
            graphs.append(None)
            continue
        
        # Node labels: atomic symbol
        node_labels = {atom.GetIdx(): atom.GetSymbol() for atom in mol.GetAtoms()}
        
        # Edge list
        edge_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_list.append((i, j))
        
        graphs.append((node_labels, edge_list))
    
    return graphs