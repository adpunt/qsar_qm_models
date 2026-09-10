#!/usr/bin/env python
"""Every constant the figure and analysis scripts share, and the ONLY place a
display choice is written down.

WHY THIS FILE EXISTS
--------------------
`generate_paper_figures_v2.py` held all of this inline in a 5,378-line module,
which is how four separate things went wrong:

  1. Two name maps lived in two dicts a hundred lines apart and neither was
     updated when four models started being emitted (`model_names.json` exists
     because of that).
  2. Two baseline gates ran under one headline -- 0.6 governed the simple-effects
     table and 0.3 governed everything else (RERUN_PLAN.md 5.4). ONE gate is
     declared here.
  3. Panel orders were hard-coded lists of the six noise types retired on
     2026-08-26, so `create_figure2` renders blank axes on current data. Nothing
     here restates a condition roster; it is read from `noise_conditions.json`.
  4. A reporting level of 0.3 sat as a function's default argument, on a scale
     that no longer exists, chosen by nobody. Nothing here restates one; it is
     read from `models/model_defaults.py`, which RAISES for a dataset whose level
     is unset.

RULES
-----
1. NO roster, level grid, reporting level or condition list is restated here.
   Each is read from the file that owns it, and an unreadable source raises
   rather than falling back to a default.
2. Display choices -- colours, markers, labels, ordering, figure width -- DO live
   here, because nothing else owns them.
3. One threshold per question. If two numbers answer the same question, that is
   the bug this file was written to stop.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'models'))

# ---------------------------------------------------------------------------
# The sources of truth. Each is read, never restated.
# ---------------------------------------------------------------------------

MODEL_NAMES_FILE = ROOT / 'model_names.json'
CONDITION_NAMES_FILE = ROOT / 'condition_names.json'
NOISE_CONDITIONS_FILE = ROOT / 'noise_conditions.json'
QM9_GENERATOR = ROOT / 'slurm_scripts_qm9_rerun' / 'generate_scripts.py'


def _read_json(path, what):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        raise SystemExit(
            f"ERROR: {path.name} is missing and it is the only place {what} is "
            f"written down. Nothing here holds a second copy on purpose "
            f"(RERUN_PLAN.md 0.6, failure mode 10).")


_MODEL_NAMES = _read_json(MODEL_NAMES_FILE, 'the model-name correspondence')
_SETTLED = _read_json(NOISE_CONDITIONS_FILE, 'the settled noise conditions')

#: Every model name a figure may use. From `model_names.json`.
CANONICAL_MODELS = list(_MODEL_NAMES['canonical'])

#: Raw name (as each pipeline writes it) -> canonical. Hyphen-stripped spellings
#: are added because the per-molecule uncertainty FILENAMES strip hyphens.
def _name_map(key):
    raw = dict(_MODEL_NAMES[key])
    for written, canonical in list(raw.items()):
        raw.setdefault(written.replace('-', ''), canonical)
        raw.setdefault(written.replace('-', '').lower(), canonical)
        raw.setdefault(written.lower(), canonical)
    # A name that is ALREADY canonical maps to itself. Without this, anything
    # reading a frame that has been normalised once -- a merged table, a cached
    # parquet, a fixture -- warns that every model is unknown and then lower-
    # cases names that were already right. The canonical list is authoritative,
    # so identity is the correct mapping, not a fallback.
    for canonical in _MODEL_NAMES['canonical']:
        raw.setdefault(canonical, canonical)
        raw.setdefault(canonical.replace('_', '').lower(), canonical)
    return raw


QM9_MODEL_MAP = _name_map('qm9')
VALIDATION_MODEL_MAP = _name_map('validation')

#: The settled conditions, by stage. Read, never restated -- a hard-coded panel
#: order of retired names is what makes the current fig2 render blank.
FULL_GRID_CONDITIONS = [c['name'] for c in _SETTLED['stage_1_full_grid']]
DEPTH_ONLY_CONDITIONS = [c['name'] for c in _SETTLED['stage_2_depth_only']]
SETTLED_CONDITIONS = FULL_GRID_CONDITIONS + DEPTH_ONLY_CONDITIONS
NOT_RUN_CONDITIONS = [c['name'] for c in _SETTLED.get('not_run', [])]

#: Conditions that run on a NAMED SUBSET of pairs rather than the whole grid.
#: `noise_conditions.json` carries the scope block; censoring's says in its own
#: words: "No claim about WHICH model resists censoring best can rest on this
#: run." figlib_guard turns that into an assertion.
PAIR_SUBSET_CONDITIONS = {
    c['name']: c['scope']
    for c in _SETTLED['stage_1_full_grid'] + _SETTLED['stage_2_depth_only']
    if isinstance(c.get('scope'), dict)
    and c['scope'].get('mode') == 'pair_subset'
}


def _levels(name):
    """Lift a level ladder out of the QM9 generator rather than holding a copy.

    NOISE_DESIGN.md 6.4 is the one place the levels live, and the QM9 generator
    holds the operative copy. `slurm_scripts_validation_rerun/generate_scripts.py`
    already reads them this way for the same reason; this is the third consumer,
    not a third copy.
    """
    src = QM9_GENERATOR.read_text()
    m = re.search(r"^%s\s*=\s*'([^']+)'" % name, src, re.M)
    if not m:
        raise SystemExit(
            f"ERROR: cannot find {name} in {QM9_GENERATOR.name}. The ladder "
            f"lives in NOISE_DESIGN.md 6.4 and the generator holds the "
            f"operative copy; this file reads it rather than restating it.")
    return [float(x) for x in m.group(1).split()]


DOSE_LEVELS = _levels('DOSE_LEVELS')
CENSOR_LEVELS = _levels('CENSOR_LEVELS')


def expected_levels(condition):
    """The ladder a condition is run on. Censoring has its own axis (a fraction
    of labels clipped), which is why it can never share a colour scale or an
    x-axis with the others."""
    return CENSOR_LEVELS if condition.startswith('censoring') else DOSE_LEVELS


# `reporting_level(dataset)` RAISES for a dataset whose level is unset, which is
# the whole point -- every previous default silently became the answer.
from model_defaults import (  # noqa: E402
    REPORTING_LEVELS,
    UNCERTAINTY_DEFAULTS,
    gp_fit_collapsed,
    reporting_level,
    spec_hash,
    SPEC_VERSION,
)

#: Which column counts as "the model's uncertainty". Measured, not chosen --
#: see the comment block in `models/model_defaults.py`.
UNCERTAINTY_PRIMARY = UNCERTAINTY_DEFAULTS['primary_column']
COVERAGE_LEVELS = UNCERTAINTY_DEFAULTS['coverage_levels']

# ---------------------------------------------------------------------------
# Thresholds. ONE number per question.
# ---------------------------------------------------------------------------

#: A configuration whose CLEAN accuracy is below this is not asked how much it
#: retains, because a near-zero denominator makes the ratio meaningless.
#:
#: The old script had two of these -- 0.6 driving the simple-effects table and
#: 0.3 driving everything else -- so "we excluded weak configurations" meant two
#: different rosters on facing pages (RERUN_PLAN.md 5.4). 0.3 is the survivor:
#: auc_norm is a retention FRACTION and stays meaningful well below the 0.6 gate
#: that the retired slope metric needed.
BASELINE_THRESHOLD = 0.3

#: Minimum replicates in a cell before it may enter a variance decomposition.
#: RERUN_PLAN.md 0.6 guard 3.
MIN_CELL_ITERS = 5

#: A single replicate whose accuracy falls below this did not train; the whole
#: replicate is dropped and the drop is DECLARED (guard 8), never silent.
CATASTROPHIC_R2_THRESHOLD = -0.5

#: auc_norm above this is reported and counted, never patched. RERUN_PLAN.md
#: 7.3 says it is structural and will reappear; 14.6 row 10 makes it a decision.
AUC_NORM_IMPLAUSIBLE_HIGH = 1.05

# ---------------------------------------------------------------------------
# The probabilistic transformations (Aim 2, table T5).
#
# Each pair is (deterministic base, probabilistic sibling). Read off the notes
# in the QM9 job generator's own MODELS dict.
#
# `gauche` is NOT in this table: a Tanimoto kernel on binary fingerprints is a
# different model from an RBF kernel on any representation, not a probabilistic
# transformation of it (model_names.json, the "GP" note).
# ---------------------------------------------------------------------------

PROBABILISTIC_PAIRS = [
    ('rf', 'qrf'),
    ('dnn', 'dnn_bnn_full'),
    ('dnn_bnn_full', 'dnn_vbll'),
    ('dnn_bnn_full', 'dnn_bnn_full_mve'),
    ('mlp', 'mlp_bnn_full'),
    ('mlp_bnn_full', 'mlp_vbll'),
    ('mlp_bnn_full', 'mlp_bnn_full_mve'),
    ('gauche_rbf', 'het_gp_rbf'),
]

# ---------------------------------------------------------------------------
# Display. Nothing else owns these.
# ---------------------------------------------------------------------------

#: Springer Nature sn-mathphys-num textwidth is ~6.3 in (16 cm). 6.5 in is the
#: canonical width, so \includegraphics[width=\textwidth] renders 1:1 and 10pt
#: matplotlib text stays 10pt on the printed page. Journal of Cheminformatics
#: allows 170 mm full width and 225 mm maximum height.
TEXTWIDTH_IN = 6.5
MAX_HEIGHT_IN = 225 / 25.4

RCPARAMS = {
    'figure.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 11.5,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.linewidth': 0.8,
    'legend.frameon': False,
    'lines.linewidth': 1.5,
}


def apply_style():
    """Set the shared matplotlib style. Called once, by the entry point.

    Deliberately does NOT call warnings.filterwarnings('ignore'), which the old
    script did at module scope -- it hid scipy's constant-input correlation
    warning, which is exactly the warning that fires when a per-level slice is
    degenerate.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('ticks')
    plt.rcParams.update(RCPARAMS)


CLEAN_COLOR = '#0072B2'  # used ONLY for clean (no-noise) data

CONDITION_COLORS = {
    'gaussian': '#E31A1C',
    'grouped_wider': '#33A02C',
    'grouped_shifted': '#6A3D9A',
    'censoring': '#B15928',
    'student_t_nu5': '#FF7F00',
    'outlier_p10': '#E91E63',
    'laplace': '#17BECF',
}

CONDITION_LABELS = {
    'gaussian': 'Gaussian',
    'grouped_wider': 'Grouped, wider',
    'grouped_shifted': 'Grouped, shifted',
    'censoring': 'Censoring',
    'student_t_nu5': 'Student-$t$ ($\\nu$=5)',
    'outlier_p10': 'Outlier (10%)',
    'laplace': 'Laplace',
}

#: FIXED colour anchors, so two panels of the same quantity are comparable.
#: Deriving them from each panel's own data -- which is what I did first -- makes
#: the same colour mean a different number in each panel, and a reader compares
#: panels by colour before reading a single printed value.
#:
#: The ranges are the old script's, and its reasons hold: 0.4-1.0 on QM9 where
#: everything survives the baseline gate; wider on the assay sets, where a model
#: can degrade below useless (negative) or come out slightly above 1, and
#: collapsing either into one end of the scale hides it.
#: ONE range, shared by every AUC_norm panel in the paper, so a colour means
#: the same number in F3, F4 and F8 alike.
#:
#: 0.4 to 1.0 is the old script's QM9 anchor and it is the right one: the
#: baseline gate already removes configurations too weak to have a meaningful
#: retention, so the surviving values sit well inside it and the map spends its
#: whole range where the data is. The old script used -0.5 to 1.1 on the assay
#: sets, against the possibility of a model degrading below useless -- but on
#: the re-run those all land between 0.67 and 0.94, and that wider range paints
#: every one of them the same green.
#:
#: A value outside the range is CLIPPED AND COUNTED, never silently flattened;
#: shapes.grid says how many and by how much.
AUC_RANGE = (0.4, 1.0)
AUC_RANGE_QM9 = AUC_RANGE
AUC_RANGE_ASSAY = AUC_RANGE

#: Grey, not black and not the darkest end of the colour map. A missing cell
#: rendered in the map's own dark end reads as a very low value.
MISSING_CELL_COLOUR = '#9E9E9E'

#: Inches per row of a grid, and the fixed overhead for a title and tick labels.
#: Measured, not guessed: below this, 10pt row labels and the printed cell
#: values overlap. Nineteen models is 5 inches of plot area before anything else.
GRID_ROW_INCHES = 0.26
GRID_CHROME_INCHES = 1.15


def grid_height(n_rows, n_panels=1):
    """How tall a stack of grids has to be to stay readable."""
    per_panel = n_rows * GRID_ROW_INCHES + GRID_CHROME_INCHES
    return per_panel * n_panels


ANOVA_FACTOR_COLORS = {
    'Model': '#6BAED6',
    'Representation': '#FC8D59',
    'Interaction': '#B39DDB',
    'Residual': '#BDBDBD',
}

#: Variants of one family share a colour; MODEL_MARKERS separates them.
MODEL_COLORS = {
    'rf': '#0072B2', 'qrf': '#0072B2',
    'xgboost': '#56B4E9', 'lgb': '#009E73', 'ngboost': '#D55E00',
    'dnn': '#E69F00', 'dnn_bnn_full': '#E69F00', 'dnn_bnn_last': '#E69F00',
    'dnn_vbll': '#E69F00', 'dnn_vbll_hetero': '#E69F00',
    'dnn_bnn_full_mve': '#E69F00',
    'mlp': '#CC79A7', 'mlp_bnn_full': '#CC79A7', 'mlp_bnn_last': '#CC79A7',
    'mlp_vbll': '#CC79A7', 'mlp_vbll_hetero': '#CC79A7',
    'mlp_bnn_full_mve': '#CC79A7',
    'svm': '#999999',
    'gauche': '#882255', 'gauche_rbf': '#882255', 'het_gp_rbf': '#882255',
}

#: For the uncertainty figures, where several variants of one family appear
#: together and family colours would make three pairs of indistinguishable
#: lines. Distinctness, not grouping -- which is why it does not follow
#: MODEL_COLORS.
UNCERTAINTY_COLORS = {
    'qrf': '#E69F00', 'ngboost': '#D55E00',
    'gauche': '#882255', 'gauche_rbf': '#117733', 'het_gp_rbf': '#661100',
    'dnn_bnn_full': '#0072B2', 'dnn_vbll': '#009E73',
    'dnn_vbll_hetero': '#88CCEE', 'dnn_bnn_full_mve': '#44AA99',
    'mlp_bnn_full': '#CC79A7', 'mlp_vbll': '#332288',
    'mlp_vbll_hetero': '#DDCC77', 'mlp_bnn_full_mve': '#999933',
}

#: The two halves of the uncertainty, and they keep these colours in every
#: panel of F6 so the reader learns them once (RERUN_PLAN.md 14.5 F6).
COMPONENT_COLORS = {'aleatoric': '#D55E00', 'epistemic': '#0072B2',
                    'total': '#666666'}
COMPONENT_LABELS = {'aleatoric': 'Aleatoric (data)',
                    'epistemic': 'Epistemic (model)',
                    'total': 'Total'}

#: The three lines on each F7 panel. The two references are grey on purpose:
#: they are what the model's own line is read against, not results themselves.
CURVE_COLORS = {'uncertainty': '#0072B2', 'ratio': '#0072B2',
                'error': '#666666', 'oracle': '#009E73', 'random': '#999999'}
CURVE_LABELS = {'uncertainty': 'Ordered by predicted uncertainty',
                'ratio': 'Ordered by error / uncertainty',
                'error': 'Ordered by out-of-fold error alone',
                'oracle': 'Ordered by true error (the best possible)',
                'random': 'No ordering'}
CURVE_STYLES = {'uncertainty': '-', 'ratio': '-', 'error': '--',
                'oracle': ':', 'random': '--'}


def component_label(name):
    return COMPONENT_LABELS.get(str(name), str(name))


def curve_label(name):
    return CURVE_LABELS.get(str(name), str(name))


#: Base model = circle, full Bayesian = square, VBLL = diamond, a per-molecule
#: noise head = plus, a variance head = star.
MODEL_MARKERS = {
    'rf': 'o', 'qrf': 'D', 'xgboost': 'o', 'lgb': 'o', 'ngboost': 'o',
    'svm': 'o', 'gauche': 'o', 'gauche_rbf': 's', 'het_gp_rbf': 'P',
    'dnn': 'o', 'dnn_bnn_full': 's', 'dnn_bnn_last': '^', 'dnn_vbll': 'D',
    'dnn_vbll_hetero': 'P', 'dnn_bnn_full_mve': '*',
    'mlp': 'o', 'mlp_bnn_full': 's', 'mlp_bnn_last': '^', 'mlp_vbll': 'D',
    'mlp_vbll_hetero': 'P', 'mlp_bnn_full_mve': '*',
}

MODEL_ORDER = [
    'rf', 'qrf',
    'xgboost', 'lgb', 'ngboost',
    'svm', 'gauche', 'gauche_rbf', 'het_gp_rbf',
    'dnn', 'dnn_bnn_full', 'dnn_bnn_full_mve', 'dnn_vbll', 'dnn_vbll_hetero',
    'mlp', 'mlp_bnn_full', 'mlp_bnn_full_mve', 'mlp_vbll', 'mlp_vbll_hetero',
]

MODEL_LABELS = {
    'rf': 'RF', 'xgboost': 'XGBoost', 'lgb': 'LightGBM', 'qrf': 'QRF',
    'ngboost': 'NGBoost', 'svm': 'SVM',
    'dnn': 'NN-α', 'mlp': 'NN-β',
    'gauche': 'GP', 'gauche_rbf': 'GP (RBF)',
    'dnn_bnn_full': 'BNN-α', 'dnn_vbll': 'VBLL-α',
    'mlp_bnn_full': 'BNN-β', 'mlp_vbll': 'VBLL-β',
    # The suffix is load-bearing and stays in the label: these report a
    # different KIND of data-noise term from the model they are a variant of
    # (RERUN_PLAN.md 5.5f), and folding it away makes a per-molecule column and
    # a broadcast constant indistinguishable.
    'het_gp_rbf': 'GP (RBF, het.)',
    'dnn_bnn_full_mve': 'BNN-α (var. head)',
    'mlp_bnn_full_mve': 'BNN-β (var. head)',
    'dnn_vbll_hetero': 'VBLL-α (het.)',
    'mlp_vbll_hetero': 'VBLL-β (het.)',
}

REP_LABELS = {
    'ecfp4': 'ECFP4',
    'pdv': 'PDV',
    'mhggnn': 'MHG-GNN',
    'avalon': 'Avalon',
    'chemberta': 'ChemBERTa',
    'sns': 'Sort & Slice',
}

#: Every spelling of a representation either pipeline writes, mapped to the
#: canonical one. This lived inline in the old figure script and went stale the
#: same way the model map did. `mhggnn` is the case that matters: the assay
#: runner writes `MHG-GNN-pretrained`, whose lower-cased form joins to nothing.
REP_ALIASES = {
    'ecfp4': 'ecfp4', 'pdv': 'pdv', 'sns': 'sns', 'avalon': 'avalon',
    'chemberta': 'chemberta', 'mhggnn': 'mhggnn',
    'mhg-gnn-pretrained': 'mhggnn', 'mhggnnpretrained': 'mhggnn',
    'mhg_gnn_pretrained': 'mhggnn', 'mhg-gnn': 'mhggnn',
    'sort_and_slice': 'sns', 'sortandslice': 'sns',
    'morgan': 'ecfp4',
}


def canonical_rep(name):
    """The canonical spelling of a representation, whoever wrote it."""
    key = str(name).strip().lower()
    return REP_ALIASES.get(key, REP_ALIASES.get(key.replace('-', ''), key))


def canonical_model(name, pipeline='qm9'):
    """The canonical spelling of a model name, from whichever pipeline wrote it.

    An unknown name comes back lower-cased and unchanged, as it always did, so a
    legacy file still loads -- but callers that care can compare against
    CANONICAL_MODELS to find out.
    """
    mapping = QM9_MODEL_MAP if pipeline == 'qm9' else VALIDATION_MODEL_MAP
    raw = str(name)
    return mapping.get(raw, mapping.get(raw.lower(), raw.lower()))


DATASET_ORDER = ['qm9', 'logd', 'caco2', 'herg']
DATASET_LABELS = {
    'qm9': 'QM9 (HOMO–LUMO gap)',
    'logd': 'logD',
    'caco2': 'Caco-2',
    'herg': 'hERG K$_i$',
}

#: The word for a repeat is not the same on the two sides, and the two must be
#: labelled differently (RERUN_PLAN.md 3.2b). QM9 has ten independent replicates;
#: the assay datasets have five scaffold folds, which are a PARTITION, not
#: repeats, so they carry no error bar.
REPLICATE_COLUMN = {'qm9': 'iteration'}
FOLD_COLUMN = 'fold'


#: The uncertainty loader names QM9 'QM9' and the accuracy side names it 'qm9'.
#: Anything that filters on the dataset silently matches nothing across that
#: line -- which is how F6 came to draw an empty figure without failing.
DATASET_ALIASES = {'qm9': 'qm9', 'QM9': 'qm9', 'openadmet_logd': 'logd',
                   'openadmet_caco2': 'caco2', 'chembl_herg_ki': 'herg',
                   'herg_ki': 'herg', 'logd': 'logd', 'caco2': 'caco2',
                   'herg': 'herg'}


def canonical_dataset(name):
    """One spelling per dataset, whichever producer wrote the row."""
    key = str(name)
    return DATASET_ALIASES.get(key, DATASET_ALIASES.get(key.lower(),
                                                        key.lower()))


def replicate_column(dataset):
    return REPLICATE_COLUMN.get(str(dataset).lower(), FOLD_COLUMN)


def has_true_replicates(dataset):
    return str(dataset).lower() == 'qm9'


# ---------------------------------------------------------------------------
# Metric names. RERUN_PLAN.md 0.6 guard 12: one number, one name.
#
# The submitted paper captions the same values with two different metric names
# on facing pages, and a retired metric survives in the Conclusion as THE
# headline. A caption's metric name is generated from this table, keyed on the
# column the number came from, so the two cannot drift.
# ---------------------------------------------------------------------------

METRICS = {
    'r2': ('R$^2$', 'Variance explained; the accuracy metric'),
    'rmse': ('RMSE', 'Accuracy in the label\'s own units'),
    'mae': ('MAE', 'Accuracy in the label\'s own units'),
    'auc_norm': ('AUC$_{norm}$',
                 'Normalised area under R$^2$($\\sigma$)/R$^2$(0), trapezoidal; '
                 'higher is more robust'),
    'baseline_r2': ('Clean R$^2$', 'Accuracy with no noise added'),
    'eta2_model': ('$\\eta^2$ model', 'Share of variance from model architecture'),
    'eta2_rep': ('$\\eta^2$ representation',
                 'Share of variance from molecular representation'),
    'eta2_interaction': ('$\\eta^2$ interaction',
                         'Share of variance from the model-representation pairing'),
    'eta2_residual': ('$\\eta^2$ residual', 'Share of variance left unexplained'),
    'kendall_w': ("Kendall's $W$", 'Agreement of model rankings across conditions'),
    'icc': ('ICC(1,1)', 'Profile-level redundancy between two models'),
    'coverage_1sig': ('Coverage 1$\\sigma$',
                      'Fraction inside one predicted standard deviation; targets 0.68'),
    'coverage_2sig': ('Coverage 2$\\sigma$',
                      'Fraction inside two predicted standard deviations; targets 0.95'),
    'rho_unc_vs_clean_error': ('$\\rho$(uncertainty, error)',
                               'Rank correlation against error on the CLEAN label, within one level'),
    'auc_delta': ('$\\Delta$AUC (error/uncertainty)',
                  'Gain in finding the most-corrupted labels from dividing the '
                  'error by the uncertainty; zero means the uncertainty added nothing'),
    'rho_delta': ('$\\Delta\\rho$ (error/uncertainty)',
                  'Same comparison as a rank correlation against the injected amount'),
    'slope_mean_unc_vs_sigma': ('Uncertainty slope',
                                'Mean predicted uncertainty against noise level; '
                                'a POPULATION statement, not a per-molecule one'),
    'delivered_dose': ('Delivered noise',
                       'The amount of noise actually applied, measured not requested'),
}


def metric_label(column):
    """The display name for a column. RAISES on an unknown column rather than
    inventing one, because a caption that names a metric nothing computed is
    exactly failure mode 12."""
    if column not in METRICS:
        raise KeyError(
            f"no metric name is registered for the column {column!r}. Add it to "
            f"figlib_config.METRICS -- a caption's metric name is generated from "
            f"that table so one number cannot end up with two names "
            f"(RERUN_PLAN.md 0.6, failure mode 12). Known: {sorted(METRICS)}")
    return METRICS[column][0]


def metric_definition(column):
    if column not in METRICS:
        raise KeyError(f"no metric registered for {column!r}")
    return METRICS[column][1]


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

def model_label(model):
    key = str(model)
    if key in MODEL_LABELS:
        return MODEL_LABELS[key]
    if key.lower() in MODEL_LABELS:
        return MODEL_LABELS[key.lower()]
    return key.replace('_', '-').upper()


def rep_label(rep):
    return REP_LABELS.get(str(rep).lower(), str(rep).upper())


def condition_label(condition):
    return CONDITION_LABELS.get(str(condition), str(condition))


def dataset_label(dataset):
    return DATASET_LABELS.get(str(dataset).lower(), str(dataset))


def model_color(model):
    return MODEL_COLORS.get(str(model), '#333333')


def model_marker(model):
    return MODEL_MARKERS.get(str(model), 'o')


def sort_models(models):
    """Family order, unknown names alphabetically at the end."""
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    return sorted(models, key=lambda m: (order.get(m, len(MODEL_ORDER)), m))


def sort_conditions(conditions):
    """Settled order, unknown names alphabetically at the end."""
    order = {c: i for i, c in enumerate(SETTLED_CONDITIONS)}
    return sorted(conditions,
                  key=lambda c: (order.get(c, len(SETTLED_CONDITIONS)), c))


def provenance():
    """What every output file records, so a number can be traced to the
    parameters that produced it."""
    return {'spec_version': SPEC_VERSION, 'spec_hash': spec_hash()}


if __name__ == '__main__':
    print(f'spec {SPEC_VERSION} / {spec_hash()}')
    print(f'models          : {len(CANONICAL_MODELS)}')
    print(f'full grid       : {FULL_GRID_CONDITIONS}')
    print(f'depth only      : {DEPTH_ONLY_CONDITIONS}')
    print(f'pair subset     : {sorted(PAIR_SUBSET_CONDITIONS)}')
    print(f'dose levels     : {DOSE_LEVELS}')
    print(f'censor levels   : {CENSOR_LEVELS}')
    print(f'reporting levels: {REPORTING_LEVELS}')
    print(f'uncertainty col : {UNCERTAINTY_PRIMARY}')
    print(f'baseline gate   : {BASELINE_THRESHOLD}')
