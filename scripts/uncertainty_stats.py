"""Statistics for the per-molecule uncertainty runs (QM9 and the three
experimental datasets), as one self-contained module.

Nothing else in any of the three repositories computes a number from these runs:
`merge_results.py` concatenates task files and stops. This module is what the
figure script calls.

--------------------------------------------------------------------------
What is in here
--------------------------------------------------------------------------

``load_uncertainty``            reads both producers' schemas into one frame
``q4_plain_correlation``        the honest near-zero (NOT the answer)
``q4_error_ratio``              the answer to question 4
``confound_controlled_effect``  question B, with the zero-level subtraction
                                and the sham ceiling
``permutation_null``            the null the runbook specifies, and the naive
                                one it warns about, behind one flag
``q5_mean_uncertainty``         a POPULATION statement, labelled as one
``q6_error_ranking``            uncertainty against error from the CLEAN label

Three diagnostics that report on the inputs rather than on the science:
``check_pattern_invariance``, ``check_noise_scale_redundancy`` and
``scale_check_coverage``.

--------------------------------------------------------------------------
Two things a reader must know before using any number from here
--------------------------------------------------------------------------

1. ``noise_scale`` equals the noise level times ``noise_pattern`` exactly, for
   every condition. On any ranking statistic the two are therefore
   interchangeable and ``noise_scale`` adds nothing: within a fixed noise level
   they order the molecules identically (measured Spearman 1.000). This module
   deliberately computes the pattern statistics only. ``noise_scale`` is carried
   through the loader for provenance and is used by no reported statistic.
   ``check_noise_scale_redundancy`` exists to verify that identity on real data;
   it is a diagnostic, not a result.

2. Every correlation here is computed inside one
   (dataset, model, rep, condition, noise level, split) cell, and
   ``assert_single_cell`` is called before each one. Pooling across a dimension
   that should have been conditioned on is what produced the paper's per-molecule
   claim in the first place, so the assertion is not decoration.

--------------------------------------------------------------------------
Canonical column names
--------------------------------------------------------------------------

Neither producer is renamed on disk; both are normalised on read. The canonical
frame uses:

    dataset, model, rep, condition, sigma, fold, split, mol_id, sample_idx,
    y_true_clean, y_pred, uncertainty, injected_noise, noise_scale,
    noise_pattern, noise_pattern_pred, oof_folds_ok, source_file,
    aleatoric_uncertainty, epistemic_uncertainty,
    aleatoric_support, epistemic_support, label_scale

--------------------------------------------------------------------------
The units, which are NOT the same on the two producers
--------------------------------------------------------------------------

Every value column on a row is on the scale that row's model was fitted on, and
that scale is not the same for the two producers. QM9 standardises the labels in
the injector before any model sees one, so its predictions, its uncertainty and
both halves of its uncertainty come back in units of the clean training label
spread. The experimental runner converts everything back before it writes, so
its columns are in the label's own units — log units on all three datasets.

Within one row that is harmless: every rank statistic here is invariant to it,
and so is any ratio of two columns from the same row. It stops being harmless
the moment a MAGNITUDE from one producer is put beside a magnitude from the
other, which `q5_mean_uncertainty` and `q6_error_ranking`'s
`mean_abs_error_clean` are the only statistics here that report. Measured on one
experiment written into both schemas: every QM9 magnitude came out smaller than
the identical laboratory magnitude by exactly 1.293, the QM9 label spread.

``label_scale`` is the number that closes that gap: multiply a canonical SPREAD
on this row -- an uncertainty, either of its two halves, or any of the noise
columns -- and you have the label's own units. It is the standardisation spread
on QM9, and since 2026-08-29 it is the clean training label spread on the
experimental datasets too, because those rows are now divided by it on load. It
is 1.0 only where no conversion happened: a QM9 file written without
standardisation, or an experimental file written before its runner recorded the
spread. Those last are marked ``on_settled_scale = False`` and cannot be pooled
with converted rows.

It does NOT put ``y_true_clean`` or ``y_pred`` back in label units on QM9, which
are CENTRED as well as scaled -- a 6.0 eV label loads as -0.688 and multiplying
by the spread does not undo the mean. ``standardisation_mean`` is not carried on
the canonical row, so that direction is not recoverable here; take it from the
results row. The
two magnitude statistics apply it and say so in a ``*_units`` column; the
settled convention for reported error is the label's own units on both sides
(`RERUN_PLAN.md` 2.18), and this is the same convention for the uncertainty.

``condition`` is the noise type. It is read from whichever of ``condition``,
``noise_type``, ``task_condition`` or the legacy ``strategy``/``task_strategy``
the producer wrote, and from the file name for QM9 files that carry none of
them. The merge step writes ``task_condition``; files merged before 2026-08-27
carry ``task_strategy``, and both are accepted so an old merge still loads.

``y_true_clean`` is the UNCORRUPTED label on both producers: QM9 writes it as
``y_true_original``; on KIRBy ``y_true`` is already the clean label and the
corrupted one is ``y_true + injected_noise``.

``fold`` is the unit within which one model was fitted. On KIRBy that is the
outer scaffold fold. QM9 has no outer fold, so ``fold`` is built from
``file_no`` and ``iteration``; it is a string on both producers because it is
only ever a grouping key.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

__all__ = [
    'UncertaintySchemaError',
    'ConditioningError',
    'CELL_COLS',
    'PERM_GROUP_COLS',
    'load_uncertainty',
    'assert_single_cell',
    'q4_plain_correlation',
    'q4_error_ratio',
    'confound_controlled_effect',
    'permutation_null',
    'q5_mean_uncertainty',
    'q6_error_ranking',
    'check_pattern_invariance',
    'check_noise_scale_redundancy',
    'scale_check_coverage',
    'STATISTICS',
]


class UncertaintySchemaError(ValueError):
    """A file could not be mapped onto either producer's schema."""


class ConditioningError(AssertionError):
    """A statistic was handed a frame spanning more than one cell."""


# The cell every correlation must be computed inside.
#
# `fold` IS PART OF THE CELL. One fold is one fitted model, and the noise is
# drawn once per fold: a molecule that appears in the training half of four
# folds was corrupted four separate times, by four separate draws. A correlation
# computed across folds therefore mixes four different corruptions of the same
# molecule against four different models' opinions of it, and the number it
# produces is not the correlation of any fit that was actually run.
#
# The permutation null already grouped on fold for exactly this reason while the
# correlations it is a null FOR did not, so the two were computed over different
# groupings. Corrected 2026-08-28, on the author's instruction: "the uncertainty
# gets reported on a per-fold basis, and the correlation should be compared on a
# per-fold basis."
#
# On QM9 the loader puts the replicate number in `fold`, which is the same rule:
# one replicate is one fitted model with its own draw.
CELL_COLS = ['dataset', 'model', 'rep', 'condition', 'sigma', 'fold', 'split']

# The permutation null's group. Identical to the cell now; kept as its own name
# because the null's grouping is a statement about the null, and if the two ever
# have to differ again this is where it says so.
PERM_GROUP_COLS = ['dataset', 'model', 'rep', 'condition', 'split', 'fold', 'sigma']

# The columns that identify a cell without the noise level, used for the
# zero-level subtraction. The zero-noise control is subtracted WITHIN a fold, for
# the same reason: it is that fold's own model on that fold's own molecules.
_BASE_COLS = ['dataset', 'model', 'rep', 'condition', 'fold', 'split']

CANONICAL_COLS = [
    'dataset', 'model', 'rep', 'condition', 'sigma', 'fold', 'split',
    'mol_id', 'sample_idx', 'y_true_clean', 'y_pred', 'uncertainty',
    'injected_noise', 'noise_scale', 'noise_pattern', 'noise_pattern_pred',
    'oof_folds_ok', 'source_file',
    # The two halves of the uncertainty and, for each, whether it varies per
    # molecule or is one number per fit. Both producers write all four from
    # 2026-08-28 (RERUN_PLAN.md 5.5g); a file written before that has them
    # blank, which is not the same as a model that has no split -- the support
    # columns are what tell the two apart.
    #
    # NOTHING IN THIS MODULE CORRELATES A COMPONENT WITH ANYTHING YET. They are
    # carried so the split is readable at all; a statistic built on them must
    # first condition on the support column, because a rank correlation against
    # a constant column is undefined rather than zero.
    #
    # They were NOT carried until 2026-08-29. Both normalisers build their
    # output column by column and neither copied any of the four, and the
    # `reindex` at the end of `load_uncertainty` then created all four as
    # all-NaN -- so nothing raised, the comment above was false, and a file
    # that had the split on disk lost it on read. Measured: a QM9 file written
    # by the real writer with all four populated on 20 of 20 rows loaded with
    # 0 of 20. Both are copied through on the row's own scale now, which is the
    # scale that row's `uncertainty` is on, so the identity the writers enforce
    # (the two halves add as variances to the total) survives the read.
    'aleatoric_uncertainty', 'epistemic_uncertainty',
    'aleatoric_support', 'epistemic_support',
    # Multiply a SPREAD on this row -- the uncertainty, either half of it, or
    # any of the noise columns -- by this to get the label's own units. NOT
    # y_true_clean or y_pred on QM9, which are centred as well as scaled. See
    # "The units" in the module docstring.
    'label_scale',
    # Whether this row is on the SETTLED scale -- fractions of the clean
    # training label spread, the author's decision of 2026-08-27. QM9 is unless
    # it was run with --normalize False; a laboratory row is once the loader has
    # divided it by the spread its runner recorded, and is not if the file
    # predates that column. `assert_one_scale` refuses a frame that mixes the
    # two, because mean uncertainty over such a frame adds log units to
    # multiples of a spread.
    'on_settled_scale',
    # The invocation a row came from. QM9 names its memory-mapped files by it,
    # and it changes with every noise level, so it is PROVENANCE and not a fold
    # -- putting it in `fold` is what made the zero-noise baseline unmatchable
    # and every `effect` NaN. Blank on the laboratory side, which has no such
    # identifier.
    'file_no',
]

# The conditions the injector can produce. Read from the injector itself where it
# can be imported, so this list cannot drift from the one that draws the noise --
# it already had: this file was written against the six pre-redesign names and the
# redesign renamed all of them, so a real run's file name matched nothing.
#
# The retired names stay recognised so a file written before the redesign still
# parses; they are not produced any more.
_RETIRED_CONDITIONS = {
    'legacy', 'quantile', 'threshold', 'hetero', 'valprop',
    'heteroscedastic', 'value_proportional', 'outlier',
}
try:  # pragma: no cover - depends on the injector being installed
    from noiseInject import CONDITIONS as _INJECTOR_CONDITIONS
    _CURRENT_CONDITIONS = set(_INJECTOR_CONDITIONS)
except Exception:
    _CURRENT_CONDITIONS = {
        'gaussian', 'laplace', 'censoring', 'grouped_wider', 'grouped_shifted',
        'student_t_nu3', 'student_t_nu5', 'student_t_nu10',
        'outlier_p01', 'outlier_p05', 'outlier_p10',
    }
_VALID_CONDITIONS = _CURRENT_CONDITIONS | _RETIRED_CONDITIONS
_CONDITION_NORMALISE = {
    'heteroscedastic': 'hetero',
    'value_proportional': 'valprop',
}

# Censoring is the one condition whose written name carries its own level: both
# injectors name it `censoring_<percent clipped>` (Rust `condition_name`, and
# noiseInject's provenance row), so a clean run is `censoring_0` and a run at
# 30% is `censoring_30`.
#
# That silently voids the condition. `_subtract_zero_level` pairs a noisy row
# with its clean one on (dataset, model, rep, condition, split), so under two
# different names censoring has no clean partner, the subtraction produces NaN,
# and question B — the one censoring is in the study to answer — comes out empty
# rather than wrong. The two injectors AGREE on this name, so the cross-check
# between them cannot catch it.
#
# The level is already carried by its own column, so the name does not need it.
# Normalising here rather than renaming in the injectors keeps every file ever
# written readable, including the ones already on disk.
_CENSORING_LEVEL_SUFFIX = re.compile(r'^(censoring(?:_lower)?)_\d+$')


def _normalise_condition(name):
    """The condition name a row should be grouped under.

    Strips the level that censoring's name carries, and applies the retired-name
    map. Anything else is returned unchanged.
    """
    if not isinstance(name, str):
        return name
    m = _CENSORING_LEVEL_SUFFIX.match(name)
    if m:
        name = m.group(1)
    return _CONDITION_NORMALISE.get(name, name)

_DEFAULT_MIN_N = 20

# ---------------------------------------------------------------------------
# The model-name correspondence between the two pipelines.
#
# This module POOLS the two into one frame and grouped every statistic by the
# `model` column verbatim, so 'het_gp_rbf' (what QM9 writes) and 'GP-Hetero'
# (what the laboratory runner writes) were two different models, and 'RF' and
# 'rf' were two more. Nothing raised: each dataset's rows are internally
# consistent, so a per-dataset table looks right and only a table putting the
# four datasets side by side comes out wrong.
#
# model_names.json is the one place the correspondence is written down, the same
# treatment condition_names.json gives the noise conditions. A name that is not
# in it is lower-cased and kept -- a legacy file still loads -- and named in
# `unmapped_model_names` on the frame's attrs so a caller can see it.
# ---------------------------------------------------------------------------

def _load_model_names():
    path = Path(__file__).resolve().parent.parent / 'model_names.json'
    try:
        spec = json.loads(path.read_text())
    except Exception:
        return {}, {}
    qm9 = dict(spec.get('qm9', {}))
    val = dict(spec.get('validation', {}))
    # The per-molecule uncertainty FILENAMES strip hyphens, so a legacy file
    # arrives as 'BNNFull'. Accept both spellings.
    for m in (qm9, val):
        for name, target in list(m.items()):
            m.setdefault(name.replace('-', ''), target)
    return qm9, val


_QM9_MODEL_NAMES, _VALIDATION_MODEL_NAMES = _load_model_names()

# Names seen on disk that model_names.json does not know. Module-level, so the
# loader can report the whole set once rather than per file.
_UNMAPPED_MODELS = set()


def _canonical_model(series, mapping):
    """Map a `model` column to canonical names, recording what did not map."""
    raw = series.astype(str)
    _UNMAPPED_MODELS.update(sorted(set(raw[~raw.isin(mapping)])))
    return raw.map(mapping).fillna(raw.str.lower())


def unmapped_model_names():
    """Model names loaded so far that model_names.json does not name."""
    return sorted(_UNMAPPED_MODELS)



# How far the clean label plus the recorded noise may sit from the corrupted
# label before the frame is refused as mis-scaled. The columns are written as
# float32, so a correctly scaled file misses by about 2e-6; a file read on the
# wrong scale misses by the standardisation mean over the spread, which is 5.3
# on QM9. Nothing lands between the two.
#
# What it covers, exactly: the clean label, the recorded noise and the corrupted
# label. NOT the prediction and NOT the uncertainty. A file whose three label
# columns agree but whose `y_pred` was left on the other scale passes this and
# is wrong -- checked by writing one, which loaded without complaint and gave a
# mean absolute error of 6.96 where the truth was 0.24. The check is worth
# having and is narrower than "the file is on one scale".
_SCALE_TOLERANCE = 1e-3


def _check_label_noise_triple(out, noisy, path):
    """Refuse a frame whose clean label, recorded noise and corrupted label do
    not add up. Returns True if the check could be performed at all.

    The corrupted label is what the model actually trained on, so
    ``y_true_clean + injected_noise == y_true_noisy`` is checkable on every row
    without reference to anything outside the file. `noisy` must already be on
    the canonical frame's scale.

    Only QM9 writes the corrupted label. On the experimental files the column is
    absent, so this returns False and the frame is loaded unchecked -- which is
    why `load_uncertainty` records, per file, whether the check ran, rather than
    letting a producer that cannot be checked look the same as one that passed.
    """
    if noisy is None:
        return False
    gap = (out['y_true_clean'] + out['injected_noise'] - noisy).abs()
    if not gap.notna().any():
        return False
    worst = float(gap.max())
    if worst > _SCALE_TOLERANCE:
        raise UncertaintySchemaError(
            f"{path}: the clean label, the injected noise and the corrupted "
            f"label are not on one scale -- max |y_true_clean + "
            f"injected_noise - y_true_noisy| = {worst:.4g}, tolerance "
            f"{_SCALE_TOLERANCE:g}. On an experimental file every column is "
            f"written in the label's own units, so this can only mean one of "
            f"the three was left on the model's scale. QM9 writes y_true_noisy "
            f"standardised and y_true_original and injected_noise in the "
            f"label's own units, and this loader converts with "
            f"standardisation_mean / standardisation_sd. If those columns "
            f"are blank the file predates 2026-08-27, and every error-based "
            f"statistic on it would rank the signed residual by the raw "
            f"label instead of the size of the error; re-run rather than "
            f"loading it.")
    return True


# ---------------------------------------------------------------------------
# small numeric helpers
# ---------------------------------------------------------------------------

def _spearman(a, b):
    """Spearman correlation that returns NaN instead of warning on a constant.

    scipy emits a ConstantInputWarning and returns NaN; a constant input is a
    routine and meaningful outcome here (at noise level zero the injected noise
    is identically zero), so it is handled rather than warned about.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    a, b = a[m], b[m]
    if np.unique(a).size < 2 or np.unique(b).size < 2:
        return np.nan
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    if denom == 0:
        return np.nan
    return float((ra * rb).sum() / denom)


def _auc(score, positive):
    """Rank-based AUC (Mann-Whitney U / n_pos n_neg). Ties handled by ranks."""
    score = np.asarray(score, dtype=np.float64)
    positive = np.asarray(positive, dtype=bool)
    m = np.isfinite(score)
    score, positive = score[m], positive[m]
    n_pos = int(positive.sum())
    n_neg = int(positive.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.nan
    r = rankdata(score)
    return float((r[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _slope(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2 or np.unique(x[m]).size < 2:
        return np.nan
    return float(np.polyfit(x[m], y[m], 1)[0])


def _label_scale_of(cell):
    """The factor that puts this cell's value columns into the label's own units.

    One cell is one fitted model on one file, so the standardisation constants
    are one number; the median is taken rather than the first value only so a
    hand-built frame with a stray NaN still returns something usable. A frame
    with no `label_scale` column at all — every fixture in the test suite, and
    any frame assembled by a caller rather than by the loader — is taken to be
    in label units already, which is what it was before the column existed.
    """
    if 'label_scale' not in cell.columns:
        return 1.0
    s = pd.to_numeric(cell['label_scale'], errors='coerce')
    s = s[np.isfinite(s) & (s > 0)]
    if s.empty:
        return 1.0
    return float(np.median(s.to_numpy(dtype=np.float64)))


def _require_columns(df, cols, what):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise UncertaintySchemaError(
            f"{what} needs column(s) {missing}, which the frame does not have. "
            f"Present: {sorted(df.columns)}")
    for c in cols:
        if df[c].isna().all():
            raise UncertaintySchemaError(
                f"{what} needs column '{c}', which is present but entirely "
                f"missing (all NaN). A file predating the pipeline rewrite does "
                f"not carry it; nothing here reconstructs it.")


# ---------------------------------------------------------------------------
# E6 — the conditioning guard
# ---------------------------------------------------------------------------

def assert_single_cell(df, cols=None):
    """Raise unless `df` is exactly one (dataset, model, rep, condition, noise
    level, split) cell.

    Called before every correlation in this module. Pooling across one of these
    dimensions produces a number that looks like a per-molecule result and is
    not one; this is the only thing that stops that recurring.
    """
    cols = list(CELL_COLS if cols is None else cols)
    present = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ConditioningError(
            f"cannot verify the conditioning: column(s) {missing} are absent, "
            f"so the frame may be pooled across them without it being visible.")
    offenders = {}
    for c in present:
        vals = df[c].dropna().unique()
        if len(vals) > 1:
            offenders[c] = sorted(map(str, vals))[:8]
    if offenders:
        raise ConditioningError(
            "a correlation was handed a frame pooled across "
            + ", ".join(f"{c} ({len(v)}+ values: {v})" for c, v in offenders.items())
            + ". Condition on it first.")
    assert_one_scale(df)
    return True


class ScaleError(UncertaintySchemaError):
    """Rows in one frame are not on one scale."""


def assert_one_scale(df):
    """Refuse a frame that mixes the settled scale with raw label units.

    The settled scale is fractions of the clean training label spread (author,
    2026-08-27). QM9 writes on it directly, because it standardises before
    fitting and never converts back. The laboratory runner writes in the label's
    own units -- log units on all three of its datasets -- and the loader divides
    those rows by the clean training label spread the runner records on each row.

    A row that could not be converted is one whose file predates that column, and
    it is left in raw units and marked. Such a row is fine on its own and cannot
    be averaged with a converted one: mean uncertainty, coverage and calibration
    would all be adding log units to multiples of a spread. There is no way to
    convert it after the fact -- the spread is a property of the fold's training
    split, and the scored rows do not contain it.

    So the choice the author set is taken literally: convert the laboratory rows,
    or refuse to put them in one table.
    """
    if 'on_settled_scale' not in df.columns:
        return True
    flags = df['on_settled_scale'].dropna().unique()
    if len(flags) > 1:
        files = sorted(set(df.loc[~df['on_settled_scale'].astype(bool),
                                  'source_file'].astype(str))) \
            if 'source_file' in df.columns else []
        raise ScaleError(
            "this frame mixes rows on the settled scale (fractions of the clean "
            "training label spread) with rows still in raw label units, so any "
            "average over it adds two different quantities together. The rows "
            "that could not be converted come from file(s) written before the "
            "runner recorded the spread it standardised by"
            + (f": {files[:4]}" if files else "")
            + ". Re-run those, or select one scale before pooling.")
    return True


# ---------------------------------------------------------------------------
# E1 — the loader
# ---------------------------------------------------------------------------

def _condition_from_name(stem):
    stem = stem.replace('_uncertainty_values', '')
    for prefix in ('uncertainty_', 'anova_'):
        if stem.startswith(prefix):
            rest = stem[len(prefix):]
            for s in sorted(_VALID_CONDITIONS, key=len, reverse=True):
                if rest.startswith(s + '_'):
                    return _CONDITION_NORMALISE.get(s, s)
    return None


def _pick(df, names):
    for n in names:
        if n in df.columns:
            return df[n]
    return None


# The two halves of the uncertainty and the two columns that say whether each
# half varies per molecule. Both producers write all four; neither normaliser
# copied any of them until 2026-08-29, and the `reindex` at the end of the
# loader then manufactured them as all-NaN, so the split was silently absent
# from every frame this module produced.
_COMPONENT_COLS = ('aleatoric_uncertainty', 'epistemic_uncertainty')
_SUPPORT_COLS = ('aleatoric_support', 'epistemic_support')


def _copy_components(df, out, div=1.0):
    """Carry the uncertainty split across, on the row's own scale.

    `div` is what the caller divided `uncertainty` by to reach the settled scale.
    The halves MUST be divided by the same number: they come from the same model
    outputs, and the writers enforce an identity between them -- the two halves
    add as variances to the square of the total. Dividing the total and not the
    halves breaks that identity on exactly the rows the conversion touches, and
    nothing downstream would see it, because the shipped fixtures set both halves
    to NaN. Found by the smoke test of 2026-08-29, not by the test suite.

    The two numeric halves are left exactly as `uncertainty` is left: they come
    from the same model outputs, so they are standardised on QM9 and in label
    units on the experimental datasets, and the writers' identity -- the halves
    add as variances to the square of the total -- only holds if all three are
    treated alike. `label_scale` is what puts any of them in label units.

    The support labels are strings ('per_molecule', 'constant', 'none') and are
    the reason a constant half is not read as a null result, so they travel with
    the numbers rather than being reconstructed from them.
    """
    for c in _COMPONENT_COLS:
        out[c] = (pd.to_numeric(df[c], errors='coerce') / div
                  if c in df.columns else np.nan)
    for c in _SUPPORT_COLS:
        out[c] = df[c].astype(object) if c in df.columns else np.nan
    return out


def _normalise_qm9(df, path, strict, uncertainty_column, dataset_name):
    """QM9's writer -> canonical names. No QM9 column is renamed on disk."""
    required_new = ['split', 'canonical_smiles', 'noise_pattern']
    missing_new = [c for c in required_new if c not in df.columns]
    if missing_new and strict:
        raise UncertaintySchemaError(
            f"{path}: this QM9 uncertainty file predates the pipeline rewrite — "
            f"it is missing {missing_new}. Without 'noise_pattern' the "
            f"zero-level subtraction cannot be formed and without 'split' there "
            f"are no out-of-fold training rows. Nothing here reconstructs the "
            f"injected noise by fitting a line to it. Re-run the pipeline, or "
            f"pass strict=False to load it for test-split statistics only.")

    out = pd.DataFrame(index=df.index)
    out['dataset'] = df['dataset'] if 'dataset' in df.columns else dataset_name
    out['model'] = _canonical_model(df['model'], _QM9_MODEL_NAMES)
    rep = _pick(df, ['rep', 'representation'])
    out['rep'] = rep
    cond = _pick(df, ['condition', 'noise_type', 'task_condition',
                      'strategy', 'task_strategy'])
    if cond is None:
        cond = _condition_from_name(Path(path).stem)
        if cond is None:
            if strict:
                raise UncertaintySchemaError(
                    f"{path}: no condition column and the file name does not "
                    f"name a known condition, so rows cannot be conditioned on "
                    f"the noise type. Known: {sorted(_VALID_CONDITIONS)}")
            cond = 'unspecified'
    out['condition'] = cond
    out['condition'] = out['condition'].map(
        _normalise_condition)

    out['sigma'] = pd.to_numeric(df['sigma'], errors='coerce')
    # `fold` IS THE REPLICATE ON QM9, AND NOTHING ELSE.
    #
    # It used to be `file_no + ':' + iteration`, which broke the one statistic
    # the uncertainty question is defined by. `file_no` is a fresh identifier for
    # each invocation of the injector, and the pipeline invokes it once per NOISE
    # LEVEL -- so a level-0 row and a level-0.5 row of the same replicate carried
    # different fold labels.
    #
    # `_subtract_zero_level` matches the zero-noise control within
    # (dataset, model, rep, condition, FOLD, split), deliberately excluding
    # sigma. With the file number in the fold, the level-0 row could never be
    # matched to any other level, so `effect` -- (correlation at the level) minus
    # (correlation at zero) -- came out NaN on EVERY QM9 row. Measured on a real
    # seven-condition run: every effect NaN, including censoring, whose
    # correlation against the shape is -0.21 at level 0 and -0.73 at 0.5, so the
    # answer was sitting in the two columns beside a blank.
    #
    # The laboratory side was unaffected: there `fold` is the outer scaffold fold
    # index, which is the same across levels.
    #
    # This module's own comment above already said "on QM9 the loader puts the
    # replicate number in `fold`". It did not. The fixture in
    # scripts/test_uncertainty_stats.py gave every level ONE file number, so the
    # merge always matched and the gate never saw it (fixed there too).
    #
    # The file number stays on the frame as its own column, because it is real
    # provenance -- it names the memory-mapped files a row came from -- it is
    # just not a fold.
    itr = df['iteration'].astype(str) if 'iteration' in df.columns else None
    out['fold'] = itr if itr is not None else '0'
    out['file_no'] = (df['file_no'].astype(str) if 'file_no' in df.columns
                      else pd.Series('', index=df.index))

    out['split'] = df['split'] if 'split' in df.columns else 'test'
    out['mol_id'] = (df['canonical_smiles'] if 'canonical_smiles' in df.columns
                     else np.nan)
    out['sample_idx'] = df['sample_idx'] if 'sample_idx' in df.columns else np.nan
    # ---- the two scales in one row -----------------------------------------
    # QM9 writes y_pred_mean, y_pred_std_* and y_true_noisy on the STANDARDISED
    # scale the model was fitted on, and y_true_original, injected_noise,
    # noise_scale and noise_pattern in the label's own units. utils.py
    # UNCERTAINTY_COLUMNS says so, and puts standardisation_mean and
    # standardisation_sd on every row for exactly this reason.
    #
    # Reading them as one scale is not a rounding error. On QM9 the clean label
    # averages 6.89 eV, so |y_true_clean - y_pred| becomes that constant plus a
    # small residual, the absolute value never folds, and the statistic ranks the
    # SIGNED residual instead of the size of the error. Measured on a real run
    # before this was fixed: q4_error_ratio returned rho_error = -0.024 at level
    # 1.5, where the corrupted labels are by construction the largest errors.
    # q6_error_ranking was wrong the same way. Everything is put on the model's
    # scale here, because that is the scale y_pred and the uncertainty are on and
    # neither carries the constants needed to go the other way.
    std_mean = (pd.to_numeric(df['standardisation_mean'], errors='coerce')
                if 'standardisation_mean' in df.columns
                else pd.Series(np.nan, index=df.index))
    std_sd = (pd.to_numeric(df['standardisation_sd'], errors='coerce')
              if 'standardisation_sd' in df.columns
              else pd.Series(np.nan, index=df.index))
    # A file written with --normalize False, or by the writer before 2026-08-27,
    # leaves these blank; then every column is already on one scale and nothing
    # is converted. The consistency check below fires either way.
    to_model_scale = std_mean.notna() & std_sd.notna() & (std_sd > 0)
    sd_safe = std_sd.where(to_model_scale, 1.0)
    mean_safe = std_mean.where(to_model_scale, 0.0)

    y_clean_raw = pd.to_numeric(df['y_true_original'], errors='coerce')
    out['y_true_clean'] = (y_clean_raw - mean_safe) / sd_safe
    out['y_pred'] = pd.to_numeric(df['y_pred_mean'], errors='coerce')

    if uncertainty_column == 'calibrated':
        unc = _pick(df, ['y_pred_std_calibrated', 'y_pred_std_uncalibrated'])
    elif uncertainty_column == 'uncalibrated':
        unc = _pick(df, ['y_pred_std_uncalibrated', 'y_pred_std_calibrated'])
    else:
        unc = _pick(df, [uncertainty_column])
        if unc is None:
            raise UncertaintySchemaError(
                f"{path}: no column '{uncertainty_column}'.")
    out['uncertainty'] = pd.to_numeric(unc, errors='coerce')

    # The noise columns are amounts, not positions, so they take the spread and
    # not the mean. noise_scale = level x noise_pattern holds either side of the
    # division, which is what check_noise_scale_redundancy verifies.
    for c in ('injected_noise', 'noise_scale', 'noise_pattern',
              'noise_pattern_pred'):
        out[c] = (pd.to_numeric(df[c], errors='coerce') / sd_safe
                  if c in df.columns else np.nan)
    out['oof_folds_ok'] = (pd.to_numeric(df['oof_folds_ok'], errors='coerce')
                           if 'oof_folds_ok' in df.columns else np.nan)
    out['source_file'] = str(path)

    # The two halves of the uncertainty, on the same scale as `uncertainty`
    # itself. They come out of the same model outputs as y_pred_std_*, so they
    # are standardised exactly as it is and are NOT divided by the spread here:
    # dividing them would break the identity the writer enforces, that the two
    # halves add as variances to the square of the total.
    _copy_components(df, out)

    # What multiplies a value on this row into the label's own units. QM9 is
    # standardised in the injector, so that is the spread; a file written with
    # --normalize False is already in label units, so it is 1.0 -- which is what
    # sd_safe holds in both cases.
    out['label_scale'] = sd_safe
    # Whether this row is on the settled scale -- fractions of the clean
    # training label spread. A QM9 file written with --normalize False is in raw
    # label units and is not, so it cannot be pooled with one that is.
    out['on_settled_scale'] = to_model_scale

    # The check that makes a scale mismatch an error rather than a quiet wrong
    # number.
    noisy = (pd.to_numeric(df['y_true_noisy'], errors='coerce')
             if 'y_true_noisy' in df.columns else None)
    out.attrs['scale_checked'] = _check_label_noise_triple(out, noisy, path)
    return out


def _normalise_kirby(df, path, strict, dataset_name):
    """KIRBy's writer -> canonical names. KIRBy's schema is the reference."""
    out = pd.DataFrame(index=df.index)
    ds = _pick(df, ['dataset', 'task_dataset'])
    out['dataset'] = ds if ds is not None else (dataset_name or Path(path).parent.name)
    model = _pick(df, ['model', 'task_model'])
    rep = _pick(df, ['rep', 'representation', 'task_rep'])
    if model is None or rep is None:
        # The five-column legacy files carry neither; the file name does.
        stem = Path(path).stem.replace('_uncertainty_values', '')
        bits = stem.split('_')
        if len(bits) >= 2:
            model = model if model is not None else bits[0]
            rep = rep if rep is not None else bits[1]
    out['model'] = (_canonical_model(pd.Series(model, index=df.index),
                                     _VALIDATION_MODEL_NAMES)
                    if model is not None else model)
    out['rep'] = rep

    cond = _pick(df, ['condition', 'noise_type', 'task_condition',
                      'strategy', 'task_strategy'])
    missing = []
    if cond is None:
        missing.append('condition (written as noise_type/strategy)')
    for c in ('split', 'injected_noise', 'noise_pattern'):
        if c not in df.columns:
            missing.append(c)
    if missing and strict:
        raise UncertaintySchemaError(
            f"{path}: this experimental uncertainty file predates the runner "
            f"rewrite — it is missing {missing}. Without 'split' it holds test "
            f"molecules only; without 'injected_noise' the corrupted molecules "
            f"cannot be identified; without 'noise_pattern' the zero-level "
            f"subtraction cannot be formed; and without the condition, rows "
            f"from different noise types cannot be told apart. Nothing here "
            f"reconstructs any of them. Re-run, or pass strict=False to load "
            f"it for whatever it can still support.")
    # A file that names no condition gets 'unspecified' rather than NaN, so the
    # conditioning guard and every group key can still SEE the column. An
    # all-NaN column would let a pooled frame slip past `assert_single_cell`.
    out['condition'] = cond if cond is not None else 'unspecified'
    out['condition'] = out['condition'].map(
        _normalise_condition)

    out['sigma'] = pd.to_numeric(df['sigma'], errors='coerce')
    out['fold'] = df['fold'].astype(str) if 'fold' in df.columns else '0'
    out['split'] = df['split'] if 'split' in df.columns else 'test'
    out['mol_id'] = df['mol_idx'] if 'mol_idx' in df.columns else np.nan
    out['sample_idx'] = df['sample_idx'] if 'sample_idx' in df.columns else np.nan
    # ---- ONE SCALE ACROSS THE TWO PIPELINES ---------------------------------
    # This runner converts its predictions and uncertainties back to RAW label
    # units before writing them -- log units on all three laboratory datasets.
    # QM9 leaves them on the standardised scale it fitted on, which is multiples
    # of the clean training label spread, and _normalise_qm9 above keeps them
    # there. So the column called `uncertainty` held two different physical
    # quantities, and any table pooling the four datasets -- mean uncertainty,
    # coverage, calibration -- averaged log units together with multiples of a
    # spread.
    #
    # The settled scale is fractions of the CLEAN TRAINING LABEL SPREAD (author,
    # 2026-08-27, the same decision that fixed the noise level grid). So this
    # side is divided by `label_scale`, which the runner writes on every row: the
    # spread of the clean training labels of that fold, the number it
    # standardised by.
    #
    # A file written before the runner carried that column CANNOT be converted --
    # the spread is a property of the fold's training split and is not
    # recoverable from the scored rows. Those files load, so a single-dataset
    # analysis still works, but the rows are marked and `assert_one_scale` below
    # refuses to pool them. Guessing the spread from the rows present would be a
    # silent wrong answer, which is the failure this whole module exists to stop.
    scale = (pd.to_numeric(df['label_scale'], errors='coerce')
             if 'label_scale' in df.columns
             else pd.Series(np.nan, index=df.index))
    convertible = scale.notna() & (scale > 0)
    div = scale.where(convertible, 1.0)
    out['on_settled_scale'] = convertible

    out['y_true_clean'] = pd.to_numeric(df['y_true'], errors='coerce') / div
    out['y_pred'] = pd.to_numeric(df['y_pred'], errors='coerce') / div
    out['uncertainty'] = pd.to_numeric(df['uncertainty'], errors='coerce') / div
    # The noise columns are in the label's own units on both sides, so they are
    # divided by the same number and stay comparable with the error columns.
    for c in ('injected_noise', 'noise_scale', 'noise_pattern',
              'noise_pattern_pred'):
        out[c] = (pd.to_numeric(df[c], errors='coerce') / div
                  if c in df.columns else np.nan)
    out['oof_folds_ok'] = (pd.to_numeric(df['oof_folds_ok'], errors='coerce')
                           if 'oof_folds_ok' in df.columns else np.nan)
    out['source_file'] = str(path)
    _copy_components(df, out, div=div)

    # What multiplies a value on this row back into the label's own units. The
    # runner writes in label units and the block above divides by the clean
    # training label spread to reach the settled scale, so the factor back is
    # that same spread -- the same meaning this column has on the QM9 side. A row
    # that could not be converted was left alone, and its factor is 1.
    out['label_scale'] = div

    # The same scale check the QM9 files get. It cannot fire today, because the
    # runner writes no corrupted label -- y_true is the clean one and the row
    # carries the noise beside it, but never their sum. That is the whole point
    # of recording whether it ran: an experimental file is not checked, and
    # until 2026-08-29 nothing said so, so a mis-scaled experimental file was
    # indistinguishable from a checked one. Written as a lookup rather than a
    # QM9-only block so that the day the runner adds the column, the check
    # starts biting with no further change here.
    # Divided by the same factor as `y_true_clean` and `injected_noise` above.
    # The check adds the first two and compares against this one, so handing it
    # the raw column while the other two are converted refuses a self-consistent
    # file with a false scale error -- which is what it did until the smoke test
    # of 2026-08-29 built a laboratory file that carried the corrupted label.
    noisy = _pick(df, ['y_true_noisy'])
    if noisy is not None:
        noisy = pd.to_numeric(noisy, errors='coerce') / div
    out.attrs['scale_checked'] = _check_label_noise_triple(out, noisy, path)
    return out


def _detect_schema(df):
    if 'y_pred_mean' in df.columns or 'y_true_original' in df.columns:
        return 'qm9'
    if 'y_pred' in df.columns and 'uncertainty' in df.columns:
        return 'kirby'
    return None


def load_uncertainty(paths, strict=True, uncertainty_column='uncalibrated',
                     dataset_name=None, pattern='*_uncertainty_values.csv',
                     on_error='raise'):
    """Read per-molecule uncertainty files from either producer into one frame.

    Parameters
    ----------
    paths : str | Path | iterable
        Files, directories (searched with `pattern`, recursively), or a mix.
    strict : bool
        True (default) refuses a file that predates the pipeline rewrite,
        naming the columns it lacks, rather than loading something that cannot
        answer the question. False loads it for whatever it can support.
    uncertainty_column : {'uncalibrated', 'calibrated'} or a column name
        QM9 writes both. Temperature scaling multiplies the uncalibrated value
        by one constant per file, so the two give identical RANKS and every
        rank statistic in this module is unaffected by the choice. It matters
        only for `q5_mean_uncertainty`, which is on the raw scale.
    dataset_name : str, optional
        Used for QM9 files, which carry no dataset column. Defaults to 'QM9'.
    on_error : {'raise', 'skip'}
        What to do with a file that cannot be mapped.

    Returns
    -------
    pandas.DataFrame with the canonical columns listed in the module docstring.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    files = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files.extend(sorted(p.rglob(pattern)))
        else:
            files.append(p)
    if not files:
        raise UncertaintySchemaError(f"no uncertainty files found under {paths!r}")

    frames, problems = [], []
    checked, unchecked = [], []
    for f in files:
        try:
            raw = pd.read_csv(f)
            if len(raw) == 0:
                continue
            schema = _detect_schema(raw)
            if schema == 'qm9':
                frames.append(_normalise_qm9(
                    raw, f, strict, uncertainty_column,
                    dataset_name if dataset_name is not None else 'QM9'))
            elif schema == 'kirby':
                frames.append(_normalise_kirby(raw, f, strict, dataset_name))
            else:
                raise UncertaintySchemaError(
                    f"{f}: matches neither schema. A QM9 file has 'y_pred_mean' "
                    f"and 'y_true_original'; a KIRBy file has 'y_pred' and "
                    f"'uncertainty'. Found: {sorted(raw.columns)}")
            (checked if frames[-1].attrs.get('scale_checked')
             else unchecked).append(str(f))
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            if on_error == 'raise':
                raise
            problems.append((str(f), str(exc)))
    if problems:
        warnings.warn(
            "skipped %d uncertainty file(s):\n  " % len(problems)
            + "\n  ".join(f"{p}: {e}" for p, e in problems), stacklevel=2)
    if not frames:
        raise UncertaintySchemaError(
            f"every candidate file failed to load: {problems}")

    out = pd.concat(frames, ignore_index=True)
    out = out.reindex(columns=CANONICAL_COLS)
    out['fold'] = out['fold'].astype(str)
    # Which files had their label/noise/corrupted-label arithmetic verified and
    # which could not be. `pd.concat` drops attrs, so they are set here, after
    # it. A caller that wants to know whether a frame was checked has to be able
    # to ask; before this, "checked and passed" and "no corrupted label on the
    # row, so nothing was checked" looked identical.
    out.attrs['scale_checked_files'] = checked
    out.attrs['scale_unchecked_files'] = unchecked
    return out


# ---------------------------------------------------------------------------
# shared machinery for the per-cell statistics
# ---------------------------------------------------------------------------

def _cell_cols(extra_group_cols=None):
    return CELL_COLS + [c for c in (extra_group_cols or [])
                        if c not in CELL_COLS]


def _cell_iter(df, split=None, min_n=_DEFAULT_MIN_N, extra_group_cols=None):
    """Yield (key_dict, cell_frame) for each conditioned cell.

    `extra_group_cols` conditions on MORE than the contract's cell. The obvious
    use is `['fold']`: one fold is one fitted model on its own molecules, so a
    correlation pooled across folds can be produced by differences between the
    folds rather than by anything inside one. The default is the cell the
    specification names; pass `['fold']` to check that a result survives.
    """
    cols = _cell_cols(extra_group_cols)
    d = df
    if split is not None:
        d = d[d['split'] == split]
    if len(d) == 0:
        return
    for key, cell in d.groupby(cols, dropna=False, sort=True):
        rec = dict(zip(cols, key))
        assert_single_cell(cell)
        rec['n'] = int(len(cell))
        rec['n_sufficient'] = bool(len(cell) >= min_n)
        yield rec, cell


def _base_cols(extra_group_cols=None):
    return _BASE_COLS + [c for c in (extra_group_cols or [])
                         if c not in _BASE_COLS and c != 'sigma']


def _subtract_zero_level(out, value_cols, base_cols=None):
    """Attach the noise-level-zero value of each `value_cols` entry and the
    difference, matched within (dataset, model, rep, condition, split)."""
    base_cols = list(_BASE_COLS if base_cols is None else base_cols)
    if len(out) == 0:
        for c in value_cols:
            out[c + '_at_sigma0'] = []
            out[c + '_baselined'] = []
        return out
    zero = out[np.isclose(out['sigma'].astype(float), 0.0)]
    zero = zero[base_cols + list(value_cols)].rename(
        columns={c: c + '_at_sigma0' for c in value_cols})
    zero = zero.drop_duplicates(subset=base_cols)
    out = out.merge(zero, on=base_cols, how='left')
    for c in value_cols:
        out[c + '_baselined'] = out[c] - out[c + '_at_sigma0']
    return out


# ---------------------------------------------------------------------------
# E2 — the two question-4 statistics
# ---------------------------------------------------------------------------

def q4_plain_correlation(df, split='train_oof', min_n=_DEFAULT_MIN_N,
                         extra_group_cols=None):
    """The PLAIN correlation between predicted uncertainty and the size of the
    injected noise. **This is not the answer to question 4.**

    A near-zero value here is the EXPECTED result and is not a failure of the
    models. Under cross-fitting the model that scores a molecule never saw that
    molecule's noise draw, and under Gaussian noise every molecule receives the
    same noise scale, so there is nothing in the label for the model to have
    learned about the individual draw. The design forbids any other answer. It
    is reported because it is the number a reader will otherwise assume was
    hidden, and because a LARGE value here would be evidence of leakage.

    The answer to question 4 is `q4_error_ratio`.

    The zero-level subtraction, which the confound control needs, is DEGENERATE
    here and is reported as NaN: at noise level zero `injected_noise` is
    identically zero, so its correlation with anything is undefined rather than
    zero. `rho_raw` is the column to read. `rho_baselined` is emitted only for
    symmetry with `confound_controlled_effect` and will be NaN in every ordinary
    run; `baseline_defined` says why.

    Returns one row per (dataset, model, rep, condition, noise level, split).
    """
    _require_columns(df, ['injected_noise', 'uncertainty'], 'q4_plain_correlation')
    rows = []
    for rec, cell in _cell_iter(df, split=split, min_n=min_n,
                                extra_group_cols=extra_group_cols):
        size = np.abs(cell['injected_noise'].to_numpy(dtype=np.float64))
        rec['noise_size_constant'] = bool(np.unique(size[np.isfinite(size)]).size < 2)
        rec['rho_raw'] = (_spearman(cell['uncertainty'], size)
                          if rec['n_sufficient'] else np.nan)
        rows.append(rec)
    out = pd.DataFrame(rows, columns=(
        _cell_cols(extra_group_cols) + ['n', 'n_sufficient', 'noise_size_constant', 'rho_raw']))
    out = _subtract_zero_level(out, ['rho_raw'],
                               base_cols=_base_cols(extra_group_cols))
    out = out.rename(columns={'rho_raw_at_sigma0': 'rho_at_sigma0',
                              'rho_raw_baselined': 'rho_baselined'})
    out['baseline_defined'] = out['rho_at_sigma0'].notna()
    out['statistic'] = 'q4_plain_correlation_NOT_THE_ANSWER'
    return out


def q4_error_ratio(df, split='train_oof', min_n=_DEFAULT_MIN_N, top_frac=0.10,
                   extra_group_cols=None):
    """The answer to question 4: does the predicted uncertainty add anything on
    top of the cross-fitted error?

    The out-of-fold error `|y_true + injected_noise - y_pred|` DOES track the
    injected noise, because the corrupted label is part of it. The question is
    whether dividing that error by the predicted uncertainty ranks corrupted
    labels better than the error alone. Both rankings and their difference are
    reported; `rho_delta` and `auc_delta` are the numbers that answer it, and a
    delta of zero means the uncertainty added nothing.

    `rho_*` is the Spearman correlation against the size of the injected noise.
    `auc_*` is the probability that a molecule in the most-corrupted `top_frac`
    of the cell scores above one outside it.

    At noise level zero every molecule receives exactly zero noise, so the
    target is constant and every column is NaN. That is the negative control
    working, not a gap.
    """
    _require_columns(df, ['injected_noise', 'uncertainty', 'y_true_clean', 'y_pred'],
                     'q4_error_ratio')
    rows = []
    for rec, cell in _cell_iter(df, split=split, min_n=min_n,
                                extra_group_cols=extra_group_cols):
        eps = cell['injected_noise'].to_numpy(dtype=np.float64)
        y = cell['y_true_clean'].to_numpy(dtype=np.float64)
        p = cell['y_pred'].to_numpy(dtype=np.float64)
        u = cell['uncertainty'].to_numpy(dtype=np.float64)
        err = np.abs(y + eps - p)
        usable = np.isfinite(u) & (u > 0)
        rec['n_uncertainty_unusable'] = int((~usable).sum())
        size = np.abs(eps)
        ratio = np.where(usable, err / np.where(usable, u, 1.0), np.nan)
        if not rec['n_sufficient']:
            rec.update({k: np.nan for k in
                        ('rho_error', 'rho_ratio', 'rho_delta',
                         'auc_error', 'auc_ratio', 'auc_delta')})
            rows.append(rec)
            continue
        rec['rho_error'] = _spearman(err, size)
        rec['rho_ratio'] = _spearman(ratio, size)
        rec['rho_delta'] = rec['rho_ratio'] - rec['rho_error']
        finite_size = size[np.isfinite(size)]
        if finite_size.size and np.unique(finite_size).size > 1:
            thr = np.quantile(finite_size, 1.0 - top_frac)
            pos = size >= thr
            if pos.all():
                # A condition that leaves most molecules at exactly zero noise
                # (outlier, threshold) puts the quantile ON the zero mass, so
                # `>=` selects everything. The corrupted set is then the
                # molecules above it.
                pos = size > thr
            if pos.all() or not pos.any():
                rec['auc_error'] = rec['auc_ratio'] = rec['auc_delta'] = np.nan
            else:
                rec['n_corrupted'] = int(pos.sum())
                rec['auc_error'] = _auc(err, pos)
                rec['auc_ratio'] = _auc(np.where(np.isfinite(ratio), ratio, -np.inf), pos)
                rec['auc_delta'] = rec['auc_ratio'] - rec['auc_error']
        else:
            rec['auc_error'] = rec['auc_ratio'] = rec['auc_delta'] = np.nan
        rows.append(rec)
    out = pd.DataFrame(rows, columns=(
        _cell_cols(extra_group_cols)
        + ['n', 'n_sufficient', 'n_uncertainty_unusable', 'n_corrupted',
           'rho_error', 'rho_ratio', 'rho_delta',
           'auc_error', 'auc_ratio', 'auc_delta']))
    out['top_frac'] = top_frac
    out['statistic'] = 'q4_error_ratio_THE_ANSWER'
    return out


# ---------------------------------------------------------------------------
# E3 — the confound control
# ---------------------------------------------------------------------------

def confound_controlled_effect(df, split=None, min_n=_DEFAULT_MIN_N,
                               extra_group_cols=None):
    """Question B: does the model become less certain where the labels are
    unreliable, once the label-magnitude confound is removed?

    Correlate predicted uncertainty against `noise_pattern` — the SHAPE of the
    noise at a fixed reference level of 1.0, identical at every noise level
    including zero — and subtract the same correlation at noise level zero.
    The zero-level model saw the same labels and no corruption at all, so its
    correlation is exactly the confound: whatever of the pattern is a function
    of the label itself. `effect` is what is left.

    The sham ceiling: `noise_pattern_pred` is the same shape recomputed from the
    model's own PREDICTED label, baselined the same way. An effect that is no
    larger against the real shape than against the predicted one is the model
    tracking its own prediction, not the noise. `is_detection` is
    `effect > effect_pred`, and it is a necessary condition, not a sufficient
    one.

    **The ceiling only exists for a condition whose shape depends on the label.**
    Substituting the predicted label changes nothing for a condition that keys
    its shape on the scaffold group or on a seeded draw, so the recomputed shape
    comes back bit-identical to the real one, the two correlations agree to
    every digit, and `effect > effect_pred` is False by construction rather than
    by measurement. Measured on both producers: identical for gaussian, laplace,
    student-t, the two grouped conditions and outlier; different only for
    censoring. `ceiling_is_degenerate` says so, and `is_detection` is left
    undefined rather than False wherever it is — a control that cannot differ is
    undefined, not failed. Two conditions on the current grid are affected in a
    way that matters, grouped_wider and outlier; the rest have a constant shape
    anyway, which `pattern_constant` already reports.

    `noise_scale` is deliberately not used: it equals the noise level times
    `noise_pattern` exactly, so within a level the two rank molecules
    identically and reporting both would be reporting one number twice.
    """
    _require_columns(df, ['noise_pattern', 'uncertainty'],
                     'confound_controlled_effect')
    have_pred = ('noise_pattern_pred' in df.columns
                 and df['noise_pattern_pred'].notna().any())
    rows = []
    for rec, cell in _cell_iter(df, split=split, min_n=min_n,
                                extra_group_cols=extra_group_cols):
        if rec['n_sufficient']:
            rec['rho_pattern'] = _spearman(cell['uncertainty'], cell['noise_pattern'])
            rec['rho_pattern_pred'] = (
                _spearman(cell['uncertainty'], cell['noise_pattern_pred'])
                if have_pred else np.nan)
        else:
            rec['rho_pattern'] = np.nan
            rec['rho_pattern_pred'] = np.nan
        pat_all = cell['noise_pattern'].to_numpy(dtype=np.float64)
        pat = pat_all[np.isfinite(pat_all)]
        rec['pattern_constant'] = bool(np.unique(pat).size < 2)
        # Is the ceiling a copy of the thing it is supposed to be a ceiling on?
        # Compared elementwise within the cell, treating two NaNs in the same
        # position as agreeing, because a molecule missing from one column is
        # missing from both.
        if have_pred:
            pred_all = cell['noise_pattern_pred'].to_numpy(dtype=np.float64)
            same = ((pat_all == pred_all)
                    | (~np.isfinite(pat_all) & ~np.isfinite(pred_all)))
            rec['ceiling_is_degenerate'] = bool(same.all())
        else:
            rec['ceiling_is_degenerate'] = False
        rows.append(rec)
    out = pd.DataFrame(rows, columns=(
        _cell_cols(extra_group_cols) + ['n', 'n_sufficient', 'pattern_constant',
                     'ceiling_is_degenerate',
                     'rho_pattern', 'rho_pattern_pred']))
    out = _subtract_zero_level(out, ['rho_pattern', 'rho_pattern_pred'],
                               base_cols=_base_cols(extra_group_cols))
    out = out.rename(columns={'rho_pattern_baselined': 'effect',
                              'rho_pattern_pred_baselined': 'effect_pred'})
    # Three-valued on purpose: True, False, or undefined. It was a plain bool,
    # so a cell where the comparison could not be made at all -- no ceiling on
    # the file, no zero-level partner to subtract, or a ceiling that is a copy
    # of the real shape -- came out False, which reads as "this model does not
    # detect the noise" and is a claim the data does not support. On the six
    # non-censoring conditions the ceiling is always a copy, so every cell read
    # False and the column measured nothing.
    undefined = (out['effect'].isna() | out['effect_pred'].isna()
                 | out['ceiling_is_degenerate'].fillna(False).astype(bool))
    out['is_detection'] = pd.array(
        np.where(undefined, None, out['effect'] > out['effect_pred']),
        dtype='boolean')
    out['sham_ceiling_available'] = have_pred
    out['statistic'] = 'confound_controlled_effect'
    return out


# ---------------------------------------------------------------------------
# E4 — the permutation null
# ---------------------------------------------------------------------------

def _error_from(frame_arrays, use_cached):
    y, p, eps, cached = frame_arrays
    if use_cached and cached is not None:
        return cached
    return np.abs(y + eps - p)


def _stat_error_noise_spearman(arrays, use_cached):
    y, p, eps, cached = arrays
    err = _error_from(arrays, use_cached)
    return _spearman(err, np.abs(eps))


def _stat_ratio_noise_spearman(arrays, use_cached, unc=None):
    y, p, eps, cached = arrays
    err = _error_from(arrays, use_cached)
    usable = np.isfinite(unc) & (unc > 0)
    ratio = np.where(usable, err / np.where(usable, unc, 1.0), np.nan)
    return _spearman(ratio, np.abs(eps))


def _stat_unc_noise_spearman(arrays, use_cached, unc=None):
    _, _, eps, _ = arrays
    return _spearman(unc, np.abs(eps))


STATISTICS = {
    # the runbook's headline: the cross-fitted error against the injected noise
    'error_noise_spearman': _stat_error_noise_spearman,
    # the same thing after dividing by the predicted uncertainty
    'ratio_noise_spearman': _stat_ratio_noise_spearman,
    # the plain correlation, for completeness
    'uncertainty_noise_spearman': _stat_unc_noise_spearman,
}


def permutation_null(df, statistic='error_noise_spearman', n_permutations=200,
                     recompute_error=True, seed=0, split='train_oof',
                     min_n=_DEFAULT_MIN_N, group_cols=None):
    """The permutation null for the question-4 statistics.

    The noise is permuted within one
    (dataset, model, rep, condition, split, fold, noise level) group AND THE
    ERROR IS RECOMPUTED FROM THE PERMUTED VALUE. Both halves are load-bearing.

    Why `recompute_error=False` is wrong, and why it is still offered: the
    cross-fitted error is `(y_true - y_pred) + injected_noise`, so it CONTAINS
    the value being correlated against. Permuting the noise while leaving the
    error as computed compares an error carrying the real noise with a shuffled
    copy of that noise, and therefore declares a leak on a simulation that has
    none. On clean simulated data the naive null gives a band of about
    [-0.04, +0.04] against an observed +0.62; the correct null gives about
    [+0.58, +0.62] with the observed value inside it. The flag exists so that
    fact is demonstrable rather than asserted, and the test file demonstrates
    it. Leave it at True for any reported number.

    Parameters
    ----------
    statistic : str or callable
        A key of `STATISTICS`, or `f(arrays, use_cached, unc=...) -> float`
        where `arrays` is `(y_true_clean, y_pred, injected_noise, cached_error)`.
    recompute_error : bool
        True (default) is the correct null. False is the naive one.

    Returns
    -------
    One row per permutation group: `observed`, `null_mean`, `null_lo`,
    `null_hi` (the 2.5th and 97.5th percentiles), `observed_inside_null`,
    and a two-sided permutation p-value.
    """
    _require_columns(df, ['injected_noise', 'y_true_clean', 'y_pred', 'uncertainty'],
                     'permutation_null')
    fn = STATISTICS[statistic] if isinstance(statistic, str) else statistic
    name = statistic if isinstance(statistic, str) else getattr(
        statistic, '__name__', 'custom')
    group_cols = list(PERM_GROUP_COLS if group_cols is None else group_cols)

    d = df if split is None else df[df['split'] == split]
    rows = []
    for key, g in d.groupby(group_cols, dropna=False, sort=True):
        rec = dict(zip(group_cols, key))
        assert_single_cell(g)
        n = len(g)
        rec['n'] = int(n)
        y = g['y_true_clean'].to_numpy(dtype=np.float64)
        p = g['y_pred'].to_numpy(dtype=np.float64)
        eps = g['injected_noise'].to_numpy(dtype=np.float64)
        unc = g['uncertainty'].to_numpy(dtype=np.float64)
        cached = np.abs(y + eps - p) if not recompute_error else None

        def call(e, use_cached):
            try:
                return fn((y, p, e, cached), use_cached, unc=unc)
            except TypeError:
                return fn((y, p, e, cached), use_cached)

        if n < min_n:
            rec.update(observed=np.nan, null_mean=np.nan, null_lo=np.nan,
                       null_hi=np.nan, p_value=np.nan,
                       observed_inside_null=False, n_permutations=0)
            rows.append(rec)
            continue

        rec['observed'] = call(eps, use_cached=False)
        rng = np.random.default_rng(
            abs(hash((name, recompute_error, seed, tuple(map(str, key))))) % (2**32))
        draws = np.empty(n_permutations, dtype=np.float64)
        for k in range(n_permutations):
            draws[k] = call(rng.permutation(eps), use_cached=not recompute_error)
        finite = draws[np.isfinite(draws)]
        rec['n_permutations'] = int(finite.size)
        if finite.size == 0 or not np.isfinite(rec['observed']):
            rec.update(null_mean=np.nan, null_lo=np.nan, null_hi=np.nan,
                       p_value=np.nan, observed_inside_null=False)
        else:
            rec['null_mean'] = float(finite.mean())
            rec['null_lo'] = float(np.percentile(finite, 2.5))
            rec['null_hi'] = float(np.percentile(finite, 97.5))
            centre = rec['null_mean']
            extreme = int((np.abs(finite - centre) >= abs(rec['observed'] - centre)).sum())
            rec['p_value'] = (extreme + 1) / (finite.size + 1)
            rec['observed_inside_null'] = bool(
                rec['null_lo'] <= rec['observed'] <= rec['null_hi'])
        rows.append(rec)

    out = pd.DataFrame(rows)
    out['statistic'] = name
    out['null_kind'] = 'correct_recomputed' if recompute_error else 'naive_UNSOUND'
    return out


# ---------------------------------------------------------------------------
# E5 — question 5 and question 6
# ---------------------------------------------------------------------------

def q5_mean_uncertainty(df, split=None, min_n=1, extra_group_cols=None):
    """Question 5: the mean predicted uncertainty against the noise level.

    **This is a statement about the population, not about any molecule.** It
    says that a model trained on noisier labels reports more uncertainty on
    average. It says nothing about whether the molecules whose labels were
    corrupted are the ones that come back uncertain — that is question 4, and
    `q4_error_ratio` is where it is answered. The `level_of_inference` column
    carries the label so it cannot be quoted as a per-molecule result.

    Unlike every other statistic here this one is on the raw uncertainty scale,
    so the `uncertainty_column` chosen at load time does change it.

    **The units are the label's own** — eV on QM9, log units on the three
    experimental datasets — on both producers, which is the convention settled
    for reported error in `RERUN_PLAN.md` 2.18. That took a conversion: QM9's
    models are fitted on standardised labels and its uncertainty column comes
    back in units of the clean training label spread, while the experimental
    runner converts back before writing. Every magnitude here was therefore
    smaller on QM9 than on the experimental datasets by exactly the QM9 label
    spread, for an identical experiment: measured at 1.293 on every level and on
    the slope. The `label_scale` column the loader puts on each row is what
    closes it, and `mean_uncertainty_model_scale` keeps the unconverted number
    so nothing is lost. A frame built by hand with no `label_scale` is taken to
    be in label units already.

    Also reports, broadcast onto every row of a
    (dataset, model, rep, condition, split) series, the slope and the Spearman
    correlation of the mean uncertainty against the noise level. The noise level
    is dimensionless on both producers, so the slope is in label units too.
    """
    _require_columns(df, ['uncertainty'], 'q5_mean_uncertainty')
    rows = []
    for rec, cell in _cell_iter(df, split=split, min_n=min_n,
                                extra_group_cols=extra_group_cols):
        scale = _label_scale_of(cell)
        u = cell['uncertainty'].to_numpy(dtype=np.float64)
        u = u[np.isfinite(u)]
        u_label = u * scale
        rec['mean_uncertainty'] = float(u_label.mean()) if u.size else np.nan
        rec['median_uncertainty'] = float(np.median(u_label)) if u.size else np.nan
        rec['sd_uncertainty'] = (float(u_label.std(ddof=1)) if u.size > 1
                                 else np.nan)
        rec['mean_uncertainty_model_scale'] = float(u.mean()) if u.size else np.nan
        rec['label_scale'] = scale
        rows.append(rec)
    out = pd.DataFrame(rows, columns=(
        _cell_cols(extra_group_cols) + ['n', 'n_sufficient', 'mean_uncertainty',
                     'median_uncertainty', 'sd_uncertainty',
                     'mean_uncertainty_model_scale', 'label_scale']))
    if len(out):
        base = _base_cols(extra_group_cols)
        trend = out.groupby(base, dropna=False).apply(
            lambda g: pd.Series({
                'slope_mean_unc_vs_sigma': _slope(g['sigma'], g['mean_uncertainty']),
                'rho_mean_unc_vs_sigma': _spearman(g['sigma'], g['mean_uncertainty']),
                'n_levels': int(g['sigma'].nunique()),
            }), include_groups=False).reset_index()
        out = out.merge(trend, on=base, how='left')
    out['level_of_inference'] = 'population_not_per_molecule'
    out['uncertainty_units'] = 'label_units'
    out['statistic'] = 'q5_mean_uncertainty'
    return out


def q6_error_ranking(df, split=None, min_n=_DEFAULT_MIN_N,
                     extra_group_cols=None):
    """Question 6: does predicted uncertainty rank the size of the error?

    The error is taken against the CLEAN label, `|y_true_clean - y_pred|`, and
    the correlation is computed WITHIN each noise level. Both departures matter:
    the error against the noisy label mixes in the corruption that was added to
    the label, and pooling across noise levels measures the population trend of
    question 5 wearing question 6's name.

    The clean label is available on both producers: QM9 writes it as
    `y_true_original`, and on KIRBy `y_true` IS the clean label with the
    corrupted one being `y_true + injected_noise`. The loader maps both to
    `y_true_clean`.

    The correlation is a rank statistic and is the answer to the question; it is
    unaffected by which scale the frame is on. `mean_abs_error_clean` sits
    beside it and is NOT, so it is reported in the label's own units on both
    producers, the same convention as every other reported error
    (`RERUN_PLAN.md` 2.18). Before that it was in units of the label spread on
    QM9 and in log units on the experimental datasets under one column name, and
    the two differed by the label spread for an identical experiment.
    """
    _require_columns(df, ['uncertainty', 'y_true_clean', 'y_pred'],
                     'q6_error_ranking')
    rows = []
    for rec, cell in _cell_iter(df, split=split, min_n=min_n,
                                extra_group_cols=extra_group_cols):
        scale = _label_scale_of(cell)
        err = np.abs(cell['y_true_clean'].to_numpy(dtype=np.float64)
                     - cell['y_pred'].to_numpy(dtype=np.float64))
        rec['rho_unc_vs_clean_error'] = (_spearman(cell['uncertainty'], err)
                                         if rec['n_sufficient'] else np.nan)
        rec['mean_abs_error_clean'] = (float(np.nanmean(err)) * scale
                                       if np.isfinite(err).any() else np.nan)
        rec['label_scale'] = scale
        rows.append(rec)
    out = pd.DataFrame(rows, columns=(
        _cell_cols(extra_group_cols) + ['n', 'n_sufficient', 'rho_unc_vs_clean_error',
                     'mean_abs_error_clean', 'label_scale']))
    out['statistic'] = 'q6_error_ranking'
    out['error_reference'] = 'clean_label'
    out['error_units'] = 'label_units'
    return out


# ---------------------------------------------------------------------------
# diagnostics — statements about the inputs, not results
# ---------------------------------------------------------------------------

def check_pattern_invariance(df, tol=1e-9):
    """Verify the premise of the zero-level subtraction: `noise_pattern` must be
    identical at every noise level for a given molecule, including zero.

    A diagnostic, not a result. Returns one row per
    (dataset, model, rep, condition, split, fold, molecule-set) with the largest
    spread of the pattern across noise levels and whether it is within `tol`.
    """
    _require_columns(df, ['noise_pattern', 'mol_id'], 'check_pattern_invariance')
    keys = ['dataset', 'model', 'rep', 'condition', 'split', 'fold', 'mol_id']
    g = df.groupby(keys, dropna=False)['noise_pattern']
    spread = (g.max() - g.min()).reset_index(name='pattern_spread_across_levels')
    lvl = df.groupby(keys, dropna=False)['sigma'].nunique().reset_index(
        name='n_levels')
    out = spread.merge(lvl, on=keys, how='left')
    out['invariant'] = out['pattern_spread_across_levels'].fillna(0.0).abs() <= tol
    return out


def scale_check_coverage(df):
    """Which loaded files had their label arithmetic verified, and which could not.

    A diagnostic, not a result. The check needs the corrupted label the model
    actually trained on. QM9 writes it, so a QM9 file that loads has passed; the
    experimental runner writes the clean label and the noise but never their
    sum, so an experimental file loads unchecked. Before this was recorded the
    two looked the same from the outside, and a mis-scaled experimental file
    would have been as quiet as a correct one -- demonstrated by putting the
    recorded noise on the wrong scale in each producer's schema in turn: the QM9
    file was refused by name, the experimental file loaded without complaint.
    """
    checked = list(df.attrs.get('scale_checked_files', []))
    unchecked = list(df.attrs.get('scale_unchecked_files', []))
    return {'n_checked': len(checked), 'n_unchecked': len(unchecked),
            'checked': checked, 'unchecked': unchecked,
            'all_checked': bool(checked) and not unchecked}


def check_noise_scale_redundancy(df, min_n=_DEFAULT_MIN_N):
    """Verify, on real data, the claim this module relies on: within one noise
    level `noise_scale` equals the level times `noise_pattern`, so the two rank
    molecules identically and only the pattern is reported.

    A diagnostic, not a result. `rho` should be 1.000 wherever the pattern
    varies, and `max_abs_deviation` should be at machine precision.
    """
    _require_columns(df, ['noise_scale', 'noise_pattern'],
                     'check_noise_scale_redundancy')
    rows = []
    for rec, cell in _cell_iter(df, split=None, min_n=min_n):
        sc = cell['noise_scale'].to_numpy(dtype=np.float64)
        pt = cell['noise_pattern'].to_numpy(dtype=np.float64)
        sig = float(rec['sigma'])
        rec['rho_scale_vs_pattern'] = _spearman(sc, pt)
        dev = np.abs(sc - sig * pt)
        rec['max_abs_deviation'] = (float(np.nanmax(dev))
                                    if np.isfinite(dev).any() else np.nan)
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Display names, the reporting level, and the per-cell interval.
#
# MOVED HERE 2026-08-29 from scripts/uncertainty_screen_tables.py, which was
# deleted. Only these four pieces were shared; the rest of that file computed a
# model and representation list, and that list is the author's decision of
# 2026-08-28, not something a script decides.
# ---------------------------------------------------------------------------

#: The noise level the tables report at.
REPORT_LEVEL = 1.5

MODEL_LABELS = {
    'qrf': 'QRF',
    'ngboost': 'NGBoost',
    'gauche': 'GP',
    'gauche_rbf': 'GP',
    # The pipeline writes the DNN Bayesian variants without the 'dnn_' prefix.
    'bnn_full': 'BNN-Full',
    'bnn_full_variational': 'VBLL-Full',
    'dnn_bnn_full': 'BNN-Full',
    'dnn_bnn_full_variational': 'VBLL-Full',
    'mlp_bnn_full': 'MLP-BNN-Full',
    'mlp_bnn_full_variational': 'MLP-VBLL-Full',
}
REP_LABELS = {
    'ecfp4': 'ECFP4',
    'pdv': 'PDV',
    'sns': 'SNS',
    'mhggnn': 'MHG-GNN-pretrained',
    'avalon': 'Avalon',
    'chemberta': 'ChemBERTa',
}


def bootstrap_cis(df, split='train_oof', n_boot=300, seed=20260827):
    """A spread for each cell's two deciding statistics.

    One replicate is run, so there is no replicate spread to quote and the
    cells cannot be separated by eye. Resampling the molecules inside a cell
    gives the interval that says whether one model is really ahead of another,
    which is the difference between a decision and a ranking of noise. It is a
    statement about these molecules, not a replicate spread, and is labelled so.
    """
    rng = np.random.default_rng(seed)
    keys = ['dataset', 'model', 'rep', 'condition', 'sigma']
    sub = df[df['split'] == split] if split else df
    rows = []
    for key, cell in sub.groupby(keys, dropna=False):
        rec = dict(zip(keys, key))
        eps = cell['injected_noise'].to_numpy(dtype=float)
        y = cell['y_true_clean'].to_numpy(dtype=float)
        p = cell['y_pred'].to_numpy(dtype=float)
        u = cell['uncertainty'].to_numpy(dtype=float)
        size = np.abs(eps)
        err = np.abs(y + eps - p)
        clean_err = np.abs(y - p)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(np.isfinite(u) & (u > 0), err / u, np.nan)
        n = len(cell)
        deltas, ranks = [], []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            deltas.append(_spearman(ratio[idx], size[idx])
                          - _spearman(err[idx], size[idx]))
            ranks.append(_spearman(u[idx], clean_err[idx]))
        for name, vals in (('rho_delta', deltas), ('rho_unc_vs_clean_error', ranks)):
            arr = np.asarray(vals, dtype=float)
            if np.isfinite(arr).sum() < 10:
                rec[f'{name}_lo'] = rec[f'{name}_hi'] = np.nan
                continue
            rec[f'{name}_lo'] = float(np.nanpercentile(arr, 2.5))
            rec[f'{name}_hi'] = float(np.nanpercentile(arr, 97.5))
        rec['n_boot'] = n_boot
        rows.append(rec)
    return pd.DataFrame(rows)
