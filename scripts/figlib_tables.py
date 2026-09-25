#!/usr/bin/env python
"""The paper's tables. Seven slots, CSV and LaTeX from one place.

RERUN_PLAN.md 14.7.

  T1  Metrics summary                        Methods
  T2  Noise conditions                       Methods
  T3  Variance decomposition, with spread    Q1
  T4  AUC_norm by model and condition        Q2/Q3
  T5  Probabilistic transformations          Aim 2
  T6  Uncertainty                            Q4/Q5/Q6
  T7  Rank transfer, QM9 against the assays  new

TWO JOURNAL RULES THAT SHAPE EVERY ONE OF THEM
----------------------------------------------
Journal of Cheminformatics tables carry NO COLOUR AND NO SHADING, so emphasis is
bold text or nothing -- a heat-mapped table is a figure and has to be one.

And a rule of this study rather than the journal's: **no table gets a "Mean"
column across noise conditions**. A mean over conditions is an average over a
factor, and it is the column the submitted paper sorted its ranking table by.
`assert_no_mean_column` refuses one.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figlib_config as C  # noqa: E402
import figlib_guard as G  # noqa: E402

#: Column names that would be an average over a factor.
_BANNED = ('mean', 'average', 'avg', 'overall', 'across')


def assert_no_mean_column(table, where):
    """A mean across noise conditions describes none of them.

    RERUN_PLAN.md 14.7 T4: "NO 'Mean' column -- a mean across noise conditions is
    averaging over a factor". The submitted paper's ranking table is sorted by
    exactly that column.
    """
    bad = [c for c in table.columns
           if any(b in str(c).lower() for b in _BANNED)
           and 'replicate' not in str(c).lower()]
    if bad:
        raise G.GuardError(
            f'{where}: {bad} averages over a factor. A mean across noise '
            f'conditions describes none of them, and it is the column the '
            f'submitted paper sorted its ranking by (RERUN_PLAN.md 14.7).')
    return table


#: Characters these tables carry that pdflatex cannot set, and what to set them
#: as. The table text is authored in real Unicode on purpose -- the CSV, the
#: figure titles and DECISIONS.md all want it that way -- so the swap happens
#: here, in the LaTeX fragment alone.
_LATEX_UNICODE = {
    '\u00b2': '$^2$', '\u00b3': '$^3$', '\u00b1': '$\\pm$', '\u00f7': '$\\div$',
    '\u00d7': '$\\times$', '\u2192': '$\\rightarrow$', '\u2013': '--', '\u2014': '---',
    '\u2212': '$-$', '\u2265': '$\\ge$', '\u2264': '$\\le$', '\u2248': '$\\approx$',
    '\u1d62': '$_i$', '\u03b1': '$\\alpha$', '\u03b2': '$\\beta$', '\u03b7': '$\\eta$',
    '\u03c1': '$\\rho$', '\u0394': '$\\Delta$', '\u03c3': '$\\sigma$', '\u03bd': '$\\nu$',
    '\u03bc': '$\\mu$', '\u03c7': '$\\chi$', '\u03bb': '$\\lambda$', '\u03c4': '$\\tau$',
}

#: The four characters that are LaTeX syntax in ordinary text. `%` is the one
#: that has actually broken a table: it comments out the rest of its line, so
#: "Outlier (10%)" silently ate the column after it AND the row's own `\\`.
_LATEX_ESCAPE = {'&': r'\&', '%': r'\%', '#': r'\#', '_': r'\_'}

#: Splits a cell into math segments (`$...$`) and ordinary text. The table text
#: already contains deliberate LaTeX -- `AUC$_{norm}$`, `Student-$t$ ($\nu$=5)`,
#: `hERG K$_i$` -- and escaping inside those would print the source.
_MATH = re.compile(r'(\$[^$]*\$)')


#: What a cell with no value prints as in the LaTeX fragment. `---` is the
#: em dash in the Springer class, which is what the target journal's tables
#: use. The CSV keeps the real NaN.
MISSING_CELL = '---'


def latex_safe(value):
    """One cell or column name, safe to paste into a tabular.

    Escapes `&`, `%`, `#` and `_` OUTSIDE math and maps the Unicode the study's
    labels use onto math the Springer class can set. Math segments are returned
    untouched.

    This is not cosmetic. Of the sixteen fragments the 17 September harvest
    generated, NINE could not compile at all: "Sort & Slice" added a column to
    its row in T5, T6 and T8, and "Outlier (10%)" commented out the end of its
    line -- including the row's own terminator -- in T2, T3 and all four T4s.
    FOURTEEN of the sixteen also carried a character pdflatex cannot set. Only
    T1 was clean. None of it shows in the CSV, which is what every other reader
    of these tables uses, so it survived until someone pasted one.
    """
    text = str(value)
    out = []
    for piece in _MATH.split(text):
        if piece.startswith('$') and piece.endswith('$') and len(piece) > 1:
            out.append(piece)
            continue
        for character, replacement in _LATEX_ESCAPE.items():
            piece = piece.replace(character, replacement)
        for character, replacement in _LATEX_UNICODE.items():
            piece = piece.replace(character, replacement)
        out.append(piece)
    # `R$^2$$_{norm}$` from two adjacent swaps is legal but ugly; join the pair.
    return ''.join(out).replace('$$', '')


def latex_frame(table):
    """A copy of the frame with every cell and column name made LaTeX-safe."""
    out = table.copy()
    out.columns = [latex_safe(c) for c in out.columns]
    for column in out.columns:
        if out[column].dtype == object:
            out[column] = out[column].map(
                lambda v: v if v is None or (isinstance(v, float) and pd.isna(v))
                else latex_safe(v))
    return out


def write(table, output_dir, name, caption='', where=None, index=False,
          float_format='%.3f'):
    """One table, as CSV and as a LaTeX fragment.

    The LaTeX is booktabs and carries no colour, because the journal's tables
    have none. It is a fragment, not a float: the caption and the label belong
    in the manuscript where they can be edited.
    """
    where = where or name
    assert_no_mean_column(table, where)
    G.with_components(table, where)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / f'{name}.csv', index=index)
    try:
        # A MISSING CELL PRINTS AN EM DASH, NOT `NaN`. The journal's tables use
        # a dash for a cell with no value and say in the footnote what absence
        # means; `NaN` is a pandas repr that reads to a referee as a failed
        # calculation rather than a combination that was never run. The CSV
        # above is written first and keeps the real NaN, so nothing downstream
        # of it has to parse a dash back.
        body = latex_frame(table).to_latex(
            index=index, escape=False, float_format=float_format,
            na_rep=MISSING_CELL,
            column_format='l' + 'c' * (len(table.columns) - 1))
    except Exception:                                          # pragma: no cover
        body = ''
    if body:
        header = (f'% {name}. {latex_safe(caption)}\n'
                  f'% Generated by scripts/run_paper_analysis.py; do not edit.\n'
                  f'% No colour or shading: Journal of Cheminformatics tables '
                  f'have none.\n')
        (output_dir / f'{name}.tex').write_text(header + body)
    print(f'    {name}: {len(table)} row(s)')
    return table


# ---------------------------------------------------------------------------
# T1 -- the metrics
# ---------------------------------------------------------------------------

def t1_metrics(output_dir):
    """Every metric the study reports, from the one place they are registered.

    Generated from figlib_config.METRICS rather than typed, so a metric cannot
    appear in a table under a name nothing computes -- which is failure mode 12,
    and how a retired metric survived in the Conclusion as the headline.
    """
    rows = [{'Metric': label, 'Definition': definition}
            for column, (label, definition) in sorted(C.METRICS.items(),
                                                      key=lambda kv: kv[1][0])]
    table = pd.DataFrame(rows)
    return write(table, output_dir, 'T1_metrics',
                 'Metrics used in this study. Generated from the registry the '
                 'figure code reads, so a caption cannot name a metric nothing '
                 'computes.')


# ---------------------------------------------------------------------------
# T2 -- the noise conditions
# ---------------------------------------------------------------------------

def t2_conditions(output_dir):
    """One row per settled condition: what it does to a label and why it is run.

    Read from noise_conditions.json, the file both injectors' tests read. The
    old Methods figure held its own copy of six conditions, all retired, and
    drew seven panels of no noise at all.
    """
    settled = C._SETTLED
    rows = []
    for stage, label in (('stage_1_full_grid', 'Full grid'),
                         ('stage_2_depth_only', 'Depth only')):
        for entry in settled.get(stage, []):
            scope = entry.get('scope') or {}
            rows.append({
                'Condition': C.condition_label(entry['name']),
                'Run in': label,
                'Level axis': ('fraction of labels clipped'
                               if entry['name'].startswith('censoring')
                               else 'fraction of the clean label spread'),
                'Scope': (f"{scope.get('n_pairs', '')} named pairs"
                          if scope.get('mode') == 'pair_subset'
                          else 'every model and representation'),
                'Why it is in the study': ' '.join(
                    str(entry.get('why', entry.get('rationale', ''))).split())[:400],
            })
    table = pd.DataFrame(rows)
    return write(table, output_dir, 'T2_noise_conditions',
                 'The settled noise conditions, read from noise_conditions.json '
                 '-- the file both injectors are tested against.')


# ---------------------------------------------------------------------------
# T3 -- the variance decomposition
# ---------------------------------------------------------------------------

def t3_variance(anova, output_dir, dataset='qm9', clean=None):
    """One row per condition and outcome, with the replicate spread beside each
    share. No version of this table has ever carried the spread.

    `clean` is the clean-label decomposition; its row for this dataset is put
    at the top, because the whole argument of the subsection is that the two
    choices divide accuracy one way before noise and another way after, and a
    reader cannot make that comparison across two separate tables (the author,
    2026-09-23). It has no noise condition by construction: the clean fit is
    made once per replicate and every condition starts from it.
    """
    if anova is None or not len(anova):
        return None
    frame = anova.copy()
    rows = []
    if clean is not None and len(clean):
        here = clean[clean['dataset'] == dataset]
        for _, r in here.iterrows():
            frame = pd.concat([pd.DataFrame([dict(
                r, condition='__clean__',
                outcome='Accuracy (R$^2$ on clean labels)')]), frame],
                ignore_index=True)
    for _, r in frame.iterrows():
        row = {'Condition': ('None (clean labels)'
                             if r['condition'] == '__clean__'
                             else C.condition_label(r['condition'])),
               'Outcome': r['outcome']}
        for column, label in (('eta2_model', 'Model'),
                              ('eta2_rep', 'Representation'),
                              ('eta2_interaction', 'Interaction'),
                              ('eta2_residual', 'Residual')):
            spread = r.get(f'{column}_spread')
            row[f'{label} η² (%)'] = (
                f"{r[column]:.1f}" if not np.isfinite(spread or np.nan)
                else f"{r[column]:.1f} ± {spread / 2:.1f}")
        row['Models'] = int(r.get('n_models', 0))
        row['Representations'] = int(r.get('n_reps', 0))
        row['Replicates'] = int(r.get('n_replicates', 0))
        rows.append(row)
    table = pd.DataFrame(rows)
    return write(table, output_dir, f'T3_variance_decomposition_{dataset}',
                 'Share of variance explained by model, representation, their '
                 'pairing and the residual, per noise condition. The ± is half '
                 'the leave-one-replicate-out range: the decomposition is '
                 'repeated with each replicate dropped in turn, so every fit '
                 'keeps a real error term. It is NOT a per-replicate spread -- '
                 'decomposing one replicate leaves one observation per cell, '
                 'and the residual is then arithmetically zero. The assay '
                 'datasets have no true replicates and get no band at all for '
                 'the same reason.')


# ---------------------------------------------------------------------------
# T4 -- AUC_norm by model and condition
# ---------------------------------------------------------------------------

def t4_robustness(summary, output_dir, rep, dataset='qm9'):
    """Models down the side, conditions across, clean accuracy at the left.

    NO MEAN COLUMN, and the clean column is not optional: guard 4 refuses a
    ratio printed without its denominator, and this is the ratio the paper's
    headline rests on.
    """
    frame = C.cross_model(
        summary[(summary['dataset'] == dataset) & (summary['rep'] == rep)],
        'T4')
    if not len(frame):
        return None
    G.declare(frame, 'T4', fixed={'dataset': dataset, 'rep': rep},
              varies=('model', 'condition'))
    conditions = C.sort_conditions(frame['condition'].unique())
    wide = frame.pivot_table(index='model', columns='condition',
                             values='auc_norm', aggfunc='median')
    wide = wide.reindex(index=C.sort_models(wide.index),
                        columns=[c for c in conditions if c in wide.columns])
    baseline = frame.groupby('model')['baseline_r2'].median()
    table = pd.DataFrame({'Model': [C.model_label(m) for m in wide.index]})
    table['Clean R²'] = baseline.reindex(wide.index).to_numpy()
    for condition in wide.columns:
        table[C.condition_label(condition)] = wide[condition].to_numpy()
    return write(table, output_dir, f'T4_robustness_{dataset}_{rep}',
                 f'Robustness (AUC_norm) by model and noise condition on '
                 f'{C.dataset_label(dataset)}, {C.rep_label(rep)}. The first '
                 f'column is the clean accuracy each figure is a fraction of. '
                 f'There is deliberately no mean column: a mean across noise '
                 f'conditions describes none of them.')


# ---------------------------------------------------------------------------
# T5 -- the probabilistic transformations
# ---------------------------------------------------------------------------

def t5_probabilistic(pairs, output_dir):
    """One row per pair and condition, one column per representation.

    The submitted paper's version carries a single number per row, which is a
    mean across representations -- the same averaging defect in miniature.
    """
    if pairs is None or not len(pairs):
        return None
    rows = []
    for (base, variant, condition), group in pairs.groupby(
            ['base', 'variant', 'condition']):
        row = {'Comparison': f'{C.model_label(base)} → {C.model_label(variant)}',
               'Condition': C.condition_label(condition)}
        for _, r in group.iterrows():
            mark = '*' if r.get('significant') else ''
            change = r.get('median_change')
            row[C.rep_label(r['rep'])] = (
                f'{change:+.3f}{mark}' if np.isfinite(change or np.nan) else '—')
        row['Pairs'] = int(group['n_pairs'].max())
        rows.append(row)
    table = pd.DataFrame(rows)
    return write(table, output_dir, 'T5_probabilistic_transformations',
                 'Change in robustness from replacing a model with its '
                 'probabilistic counterpart, paired on the replicate. One '
                 'column per representation and no averaging across them. '
                 '* marks a two-sided signed-rank test below 0.05; note that on '
                 'five pairs such a test cannot go below 0.0625 however large '
                 'the effect.')


# ---------------------------------------------------------------------------
# T6 -- uncertainty
# ---------------------------------------------------------------------------

def t6_uncertainty(support, q4, q6, slopes, output_dir,
                   dataset=None, rep=None, name='T6_uncertainty'):
    """One row per model and NOISE CONDITION, with the support flags.

    THE CONDITION IS A COLUMN, NOT A MEDIAN. Every number column here used to
    be a median over the seven noise conditions, on the argument that the
    support flags do not vary by condition -- which is true of the flags and of
    nothing else in the table. Censoring is the one condition under which a
    model becomes MORE certain as its labels are corrupted: the two component
    slopes run -0.43 and -0.31 against +0.63 to +0.70 and +0.08 to +0.13 under
    the other six, and not one cell separates the two components under it
    against 78 of 109 under plain Gaussian noise. A median over seven
    conditions with one of them reversed cancels the reversal out, so the table
    disagreed with the paragraph of the Results that points at it.

    `dataset` and `rep` narrow the table to one of each, which is what the
    paper prints; passing neither writes the whole thing for an additional
    file. Both are matched after `canonical_rep`, so either spelling works.

    A component flagged as one number per fit gets its flag printed and no
    slope, because a slope through a constant is arithmetic about the fit
    rather than a property of the molecules.
    """
    if support is None or not len(support):
        return None
    KEYS = ['dataset', 'model', 'rep', 'condition']

    flags = support[['dataset', 'model', 'rep', 'aleatoric_support',
                     'epistemic_support']].drop_duplicates().copy()

    # The conditions each combination actually ran, taken from the frames that
    # carry one. A cross product would invent rows for the depth-only
    # conditions on models that never ran them.
    # A ROW WITH NO CONDITION IS DROPPED, NOT FOLDED IN. `d7_q6` carries 55
    # rows whose condition is blank, and while every number here was a median
    # over the conditions those rows joined silently into every one of them.
    # What writes them is not yet established; until it is, a row that cannot
    # say which condition it belongs to cannot be in a table whose rows ARE
    # conditions.
    seen = [f.dropna(subset=['condition'])[KEYS].drop_duplicates()
            for f in (q6, q4, slopes)
            if f is not None and len(f) and 'condition' in f.columns]
    if not seen:
        return None
    table = pd.concat(seen, ignore_index=True).drop_duplicates()
    table = table.merge(flags, on=['dataset', 'model', 'rep'], how='inner')

    table['Model'] = table['model'].map(C.model_label)
    # THROUGH `canonical_rep` FIRST. The uncertainty frames carry whatever
    # spelling the producing pipeline wrote, and the laboratory runner writes
    # `MHG-GNN-pretrained` where QM9 writes `mhggnn`. `rep_label` upper-cases
    # anything it does not recognise, so the unmapped spelling became a seventh
    # representation called MHG-GNN-PRETRAINED sitting beside MHG-GNN -- one
    # representation printed as two, with the assay rows under one name and the
    # QM9 rows under the other.
    table['Representation'] = table['rep'].map(
        lambda r: C.rep_label(C.canonical_rep(r)))
    # THE DATASET IS A COLUMN, NOT A HIDDEN DIMENSION. Every merge below joins
    # on `dataset`, so the frame carries one row per dataset -- four rows per
    # model and representation, identical in their first two columns and
    # different in every number, with nothing saying which was which. A reader
    # could not tell a QM9 row from a Caco-2 one, and neither could a sort.
    table['Dataset'] = table['dataset'].map(C.dataset_label)
    table['Condition'] = table['condition'].map(C.condition_label)

    if q6 is not None and len(q6):
        rho = (q6.groupby(KEYS)['rho_unc_vs_clean_error']
               .median().rename('ρ(uncertainty, error)'))
        table = table.merge(rho, on=KEYS, how='left')
    if q4 is not None and len(q4):
        got = q4.groupby(KEYS)[['auc_error', 'auc_ratio', 'auc_delta']] \
            .median().reset_index()
        # A MAJORITY OF FOLDS, ON THE GAIN'S BAND, ON THE UPPER SIDE.
        # `.any()` over `outside_null` printed True for every one of the 82
        # rows that had a value, which is a column that cannot distinguish
        # anything -- and it was reading the band for the error rather than for
        # the uncertainty's contribution (figlib_uncertainty.q4).
        signal = 'adds_signal' if 'adds_signal' in q4.columns else None
        if signal:
            fires = (q4.groupby(KEYS)[signal].mean() >= 0.5).rename(
                'Uncertainty adds signal')
            got = got.merge(fires, on=KEYS, how='left')
        table = table.merge(got, on=KEYS, how='left')
    if slopes is not None and len(slopes):
        want = ['slope_aleatoric', 'slope_epistemic']
        if set(want) <= set(slopes.columns):
            got = slopes.groupby(KEYS)[want].median().reset_index()
            table = table.merge(got, on=KEYS, how='left')

    if dataset is not None:
        table = table[table['dataset'].astype(str).str.lower()
                      == str(dataset).lower()]
    if rep is not None:
        target = C.canonical_rep(rep)
        table = table[table['rep'].map(C.canonical_rep) == target]
    if not len(table):
        return None

    order = {c: i for i, c in enumerate(C.SETTLED_CONDITIONS)}
    table = table.assign(_c=table['condition'].map(
        lambda c: order.get(c, len(order)))).sort_values(
            ['Dataset', 'Model', 'Representation', '_c']).drop(columns='_c')

    ordered = ['Dataset', 'Model', 'Representation', 'Condition',
               'aleatoric_support', 'epistemic_support']
    # The narrowed table has one value in the columns it was narrowed on, and a
    # column of one repeated value is noise in a printed table. It stays in the
    # caption instead.
    ordered = [c for c in ordered
               if not (dataset is not None and c == 'Dataset')
               and not (rep is not None and c == 'Representation')]
    ordered += [c for c in table.columns if c not in ordered
                and c not in ('dataset', 'model', 'rep', 'condition',
                              'Dataset', 'Representation')]
    table = table[ordered].rename(columns={
        'aleatoric_support': 'Aleatoric varies', 'epistemic_support':
        'Epistemic varies', 'auc_error': 'AUC (error alone)',
        'auc_ratio': 'AUC (error ÷ uncertainty)', 'auc_delta': 'Δ AUC',
        'slope_aleatoric': 'Aleatoric slope',
        'slope_epistemic': 'Epistemic slope'})

    held = []
    if dataset is not None:
        held.append(C.dataset_label(dataset))
    if rep is not None:
        held.append(f'the {C.rep_label(C.canonical_rep(rep))} representation')
    where = f' on {" and ".join(held)}' if held else ''
    return write(table, output_dir, name,
                 f'Uncertainty statistics per model and noise condition{where}. '
                 'The two support columns say whether each component varies per '
                 'molecule or is one number per fit; a component that does not '
                 'vary has no slope, because a slope through a constant '
                 'describes the fit and not the molecules. Δ AUC is the gain '
                 'from dividing the error by the uncertainty; zero means the '
                 'uncertainty added nothing. The condition is a column and not '
                 'a median across conditions, because the two slopes reverse '
                 'sign under censoring and a median over the seven would '
                 'cancel that out.')


# ---------------------------------------------------------------------------
# T7 -- rank transfer
# ---------------------------------------------------------------------------

def t8_pairs_across_datasets(pairs, output_dir, top_n=12):
    """The model-and-representation pairs, on every dataset, ranked.

    One row per pair, one pair of columns per dataset: clean R2 and AUC_norm.
    **No score column** -- the ranking mechanism is described in the caption and
    the number itself says nothing to a reader (the author, 2026-09-14).

    Only pairs that ran on every dataset are shown, because a pair that ran on
    one cannot be ranked against one that ran on four.
    """
    if pairs is None or not len(pairs):
        return None
    frame = pairs[pairs.get('comparable', True)].copy()
    frame = frame[~frame['any_auc_norm_above_one'].astype(bool)]
    if not len(frame):
        return None
    frame = frame.sort_values('combined_scaled', ascending=False).head(top_n)
    datasets = [d for d in C.DATASET_ORDER
                if f'{d}_auc_norm' in frame.columns]
    table = pd.DataFrame({
        'Model': [C.model_label(m) for m in frame['model']],
        'Representation': [C.rep_label(r) for r in frame['rep']],
    })
    for dataset in datasets:
        table[f'{C.dataset_label(dataset)} clean R²'] = \
            frame[f'{dataset}_clean_r2'].to_numpy()
        table[f'{C.dataset_label(dataset)} AUC_norm'] = \
            frame[f'{dataset}_auc_norm'].to_numpy()
    return write(table, output_dir, 'T8_pairs_across_datasets',
                 f'The {top_n} best model-and-representation pairings across all '
                 f'{len(datasets)} datasets. For each dataset, clean R² is the '
                 f'accuracy with no noise added and AUC_norm is the share of it '
                 f'retained as noise rises. Rows are ordered by accuracy and '
                 f'robustness together: within each dataset both quantities are '
                 f'rescaled to run from 0 at that dataset\'s worst pairing to 1 '
                 f'at its best, the two are averaged, and the result is averaged '
                 f'over the datasets — so a clean R² of 0.40, which is near the '
                 f'top on Caco-2 and near the bottom on QM9, counts for what it '
                 f'is worth on each. Pairings whose AUC_norm exceeds 1 on any '
                 f'dataset are excluded: that happens when the clean accuracy a '
                 f'pairing is measured against is itself small, and the ratio '
                 f'then reports the small denominator rather than any real '
                 f'tolerance to noise. Only pairings that ran on every dataset '
                 f'appear.')


def t7_rank_transfer(transfer, output_dir, rep=None, condition='gaussian'):
    """Each model's robustness rank on QM9 beside its rank on each assay set.

    ONE TABLE PER REPRESENTATION. The transfer question is exactly where
    averaging over representations would hide the answer.
    """
    if transfer is None or not len(transfer):
        return None
    # ONE table per representation, at the reference condition. Every
    # representation crossed with every condition is eighteen files, which is
    # not a table set, it is a directory. The full long form is already written
    # as d9_rank_transfer.csv for anyone who wants a different condition.
    available = set(transfer['condition'])
    use = condition if condition in available else sorted(available)[0]
    written = []
    for one_rep in sorted(transfer['rep'].unique()):
        if rep is not None and one_rep != rep:
            continue
        frame = transfer[transfer['rep'] == one_rep]
        for condition in [use]:
            sub = frame[frame['condition'] == condition]
            if not len(sub):
                continue
            wide = sub.pivot_table(index='model', columns='dataset',
                                   values='assay_rank')
            qm9 = sub.groupby('model')['qm9_rank'].first()
            table = pd.DataFrame({'Model': [C.model_label(m) for m in wide.index]})
            table['Rank on QM9'] = qm9.reindex(wide.index).to_numpy()
            for dataset in wide.columns:
                table[f'Rank on {C.dataset_label(dataset)}'] = \
                    wide[dataset].to_numpy()
            table['Largest change'] = (
                wide.sub(qm9.reindex(wide.index), axis=0).abs().max(axis=1)
                .to_numpy())
            written.append(write(
                table, output_dir,
                f'T7_rank_transfer_{one_rep}',
                f'Robustness rank on QM9 beside the rank on each assay dataset, '
                f'on {C.rep_label(one_rep)} under '
                f'{C.condition_label(condition)}, the reference condition. '
                f'One table per representation: '
                f'averaging over representations is exactly what would hide a '
                f'ranking that does not transfer.', float_format='%.0f'))
    return written[0] if written else None


# ---------------------------------------------------------------------------
# T3b -- the same decomposition on clean labels, every dataset
# ---------------------------------------------------------------------------

def t3b_variance_clean(anova_clean, output_dir):
    """One row per dataset: how accuracy divides before any noise is added.

    T3 answers the question under noise, on QM9 alone. This is the comparison
    that separates "molecular representation governs accuracy" from
    "molecular representation governs accuracy only once the labels are
    wrong", and without it the clean-label paragraph has a figure and no
    numbers behind it (the author, 2026-09-23).

    The assay datasets get no band. Their five scaffold folds partition one
    dataset rather than repeating an experiment, so dropping one in turn does
    not measure what dropping one of the ten QM9 replicates measures -- the
    same reason T3 gives.
    """
    if anova_clean is None or not len(anova_clean):
        return None
    rows = []
    for _, r in anova_clean.iterrows():
        row = {'Dataset': C.dataset_label(r['dataset'])}
        for column, label in (('eta2_model', 'Model'),
                              ('eta2_rep', 'Representation'),
                              ('eta2_interaction', 'Interaction'),
                              ('eta2_residual', 'Residual')):
            spread = r.get(f'{column}_spread')
            row[f'{label} η² (%)'] = (
                f"{r[column]:.1f}" if not np.isfinite(spread or np.nan)
                else f"{r[column]:.1f} ± {spread / 2:.1f}")
        row['Models'] = int(r.get('n_models', 0))
        row['Representations'] = int(r.get('n_reps', 0))
        row['Replicates'] = int(r.get('n_replicates', 0))
        rows.append(row)
    table = pd.DataFrame(rows)
    return write(table, output_dir, 'T3b_variance_decomposition_clean',
                 'Share of the variance in predictive accuracy on CLEAN '
                 'labels explained by model, representation, their pairing '
                 'and the residual, one row per dataset. Accuracy is R2 on '
                 'held-out molecules with no noise added to the training '
                 'labels. There is no noise-condition column: the clean fit '
                 'is made once per replicate and every noise condition starts '
                 'from it. The ± is half the leave-one-replicate-out range, '
                 'as in the noise decomposition. The assay datasets have no '
                 'true replicates and get no band: their five scaffold folds '
                 'partition one dataset rather than repeating an experiment.')


# ---------------------------------------------------------------------------
# T9 -- mean predicted uncertainty against the noise level
# ---------------------------------------------------------------------------

#: Units of the predicted SD on each dataset. QM9's is in eV once multiplied
#: by `label_scale`, which `q5_mean_uncertainty` has already done.
_UNITS = {'qm9': 'eV', 'logd': 'log units', 'caco2': 'log units',
          'herg': 'log units'}


def t9_uncertainty_rise(rise, output_dir, rep, condition='gaussian',
                        split='test'):
    """Does mean predicted uncertainty rise with label noise, model by model?

    The author, 2026-09-25: the population paragraph of the uncertainty
    subsection had no table behind it. One table per representation; one row
    per model; for each dataset the mean predicted SD with no noise and at the
    top level (median over folds), and the top divided by the clean value in
    its lowest and highest fold. † marks a model that fails to rise in at least
    one fold. Held-out molecules, because they cover the most models. Every
    probabilistic model is kept, variants included: they are the subject here.
    """
    frame = rise[(rise['rep'] == rep) & (rise['condition'] == condition)
                 & (rise['split'] == split)].copy()
    if not len(frame):
        return None
    # The uncertainty side writes the long names; C.canonical_dataset does not
    # know them, and changing it would move every caller.
    long = {'openadmet-logd': 'logd', 'openadmet-caco2_efflux': 'caco2',
            'chembl-herg-ki': 'herg'}
    frame['dataset'] = frame['dataset'].map(
        lambda d: long.get(str(d), C.canonical_dataset(d)))
    G.declare(frame, 'T9', fixed={'rep': rep, 'condition': condition},
              varies=('model', 'dataset'))
    datasets = [d for d in C.DATASET_ORDER if d in set(frame['dataset'])]
    rows = []
    for model in C.sort_models(frame['model'].unique()):
        row = {'Model': C.model_label(model)}
        for d in datasets:
            r = frame[(frame['model'] == model) & (frame['dataset'] == d)]
            name = C.dataset_label(d)
            if not len(r):
                row[f'{name}: no noise'] = row[f'{name}: level {_top(frame)}'] = np.nan
                row[f'{name}: ratio, folds'] = MISSING_CELL
                continue
            r = r.iloc[0]
            row[f'{name}: no noise'] = r['clean_median']
            row[f'{name}: level {_top(frame)}'] = r['top_median']
            mark = '' if r['rises_in_every_fold'] else ' †'
            row[f'{name}: ratio, folds'] = (f'{r["ratio_lowest_fold"]:.2f}–'
                                            f'{r["ratio_highest_fold"]:.2f}{mark}')
        rows.append(row)
    table = pd.DataFrame(rows)
    units = ', '.join(f'{C.dataset_label(d)} in {_UNITS.get(d, "label units")}'
                      for d in datasets)
    return write(table, output_dir, f'T9_uncertainty_rise_{rep}',
                 f'Mean predicted uncertainty (SD) with no added noise and at '
                 f'the highest noise level, on {C.rep_label(rep)}, '
                 f'{C.condition_label(condition)} noise, held-out molecules. '
                 f'Each value is the median over folds ({units}). "Ratio, '
                 f'folds" is the value at the highest level divided by the '
                 f'value with no noise, in the lowest and the highest fold. '
                 f'† marks a model whose uncertainty fails to rise in at least '
                 f'one fold. A dash is a combination that was not run.')


def _top(frame):
    levels = frame['top_level'].dropna().unique()
    return f'{levels.max():g}' if len(levels) else 'top'


# ---------------------------------------------------------------------------
# T10 -- held-out accuracy at one noise level, on every dataset
# ---------------------------------------------------------------------------

def t10_accuracy_at_level(accuracies, output_dir, rep, level=1.0,
                          conditions=('gaussian', 'grouped_wider',
                                      'grouped_shifted')):
    """Held-out R2 with no noise and at one noise level, per model, on all four
    datasets (the author, 2026-09-25: "I need to see held out accuracy at a
    given noise level").

    One row is one base model. For each dataset: clean R2, then R2 at `level`
    under each condition, each the median over replicates (QM9) or scaffold
    folds (assay). The clean column is the Gaussian ladder's level 0; every
    condition starts from the same clean fit.
    """
    frames = [f for f in accuracies if f is not None and len(f)]
    if not frames:
        return None
    acc = pd.concat(frames, ignore_index=True)
    acc = acc.assign(dataset=acc['dataset'].map(C.canonical_dataset))
    acc = C.cross_model(acc[(acc['rep'] == rep)
                            & acc['condition'].isin(conditions)], 'T10')
    if not len(acc):
        return None
    G.declare(acc, 'T10', fixed={'rep': rep},
              varies=('model', 'dataset', 'condition', 'sigma'),
              aggregates=('replicate',))
    med = (acc.groupby(['dataset', 'model', 'condition', 'sigma'])['r2']
           .median())
    datasets = [d for d in C.DATASET_ORDER if d in set(acc['dataset'])]
    rows = []
    for model in C.sort_models(acc['model'].unique()):
        row = {'Model': C.model_label(model)}
        for d in datasets:
            short = C.dataset_label(d).split(' (')[0]
            row[f'{short}: clean'] = med.get((d, model, 'gaussian', 0.0), np.nan)
            for c in conditions:
                row[f'{short}: {C.condition_label(c)}'] = med.get(
                    (d, model, c, level), np.nan)
        rows.append(row)
    table = pd.DataFrame(rows)
    return write(table, output_dir, f'T10_accuracy_at_level_{rep}',
                 f'Held-out R² with no added noise and at a noise level of '
                 f'{level:g} (a fraction of the clean training label spread), '
                 f'on {C.rep_label(rep)}. One row per base model; for each '
                 f'dataset the clean value, then one column per noise '
                 f'condition. Medians over ten replicates on QM9 and over the '
                 f'scaffold folds on the assay datasets. A dash is a '
                 f'combination that was not run.')


# ---------------------------------------------------------------------------
# T11 -- each model against its own counterpart, one dataset per table
# ---------------------------------------------------------------------------

T11_CONDITIONS = ('gaussian', 'grouped_wider', 'grouped_shifted')


def t11_counterparts(changes, output_dir, dataset,
                     conditions=T11_CONDITIONS):
    """One row per pair and condition, one column per representation.

    Each cell is the change in AUC_norm, then the change in clean R2 in
    brackets, counterpart minus base, the median of the paired differences.
    † marks a change in AUC_norm with the same sign in every replicate (QM9) or
    scaffold fold (assay). The Settings column says whether the two members run
    at one setting.
    """
    if changes is None or not len(changes):
        return None
    frame = changes[(changes['dataset'] == dataset)
                    & changes['condition'].isin(conditions)]
    if not len(frame):
        return None
    rows = []
    for (base, variant, condition), group in frame.groupby(
            ['base', 'variant', 'condition'], sort=False):
        row = {'Comparison': f'{C.model_label(base)} → {C.model_label(variant)}',
               'Settings': 'shared' if group['matched'].any() else 'differ',
               'Condition': C.condition_label(condition)}
        for _, r in group.iterrows():
            n = r['n_pairs']
            same = n and (r['higher_in'] == n or r['lower_in'] == n)
            row[C.rep_label(r['rep'])] = (
                f'{r["auc_norm_change"]:+.3f}{"†" if same else ""} '
                f'({r["clean_r2_change"]:+.3f})')
        rows.append(row)
    table = pd.DataFrame(rows)
    reps = [C.rep_label(r) for r in C.REP_LABELS
            if C.rep_label(r) in table.columns]
    table = table[['Comparison', 'Settings', 'Condition'] + reps]
    unit = ('ten replicates' if dataset == 'qm9'
            else 'five scaffold folds')
    return write(table, output_dir, f'T11_counterparts_{dataset}',
                 f'{C.dataset_label(dataset)}: what replacing a model with its '
                 f'own probabilistic counterpart changes. Each cell is the '
                 f'change in AUC_norm, then the change in clean R2 in '
                 f'brackets, counterpart minus base, the median of the paired '
                 f'differences over {unit}. † marks an AUC_norm change with the '
                 f'same sign in every one. One column per representation and '
                 f'no averaging across them.')


# ---------------------------------------------------------------------------
# T12 -- what systematic error costs each model, on every representation
# ---------------------------------------------------------------------------

def t12_condition_cost(summary, output_dir, dataset='qm9',
                       condition='grouped_shifted', reference='gaussian'):
    """One row per model, one column per representation, and the yardstick beside it.

    THE CLAIM THIS TABLE EXISTS TO CARRY. Systematic error is the only
    dose-matched condition that costs anything, and it costs each model family a
    different amount -- the forests least, the plain networks most, the
    variational networks almost nothing. That claim was being made from a median
    across the roster at one representation, which is the averaging the author
    has ruled out: a median describes no model, and one representation cannot
    speak for the other five.

    So every cell is one model on one representation, and the last two columns
    are what decides whether a cell means anything. `Replicate spread` is the
    median over the six representations of that model's own highest-minus-lowest
    AUC_norm over the ten replicates under the reference condition. `Beats it
    on` counts the representations where the change is larger than that model's
    own spread, out of the ones it ran -- because a change smaller than the
    run-to-run variation is smaller than the measurement, and the paper should
    not report it as an effect.

    Negative is a loss. The reference column is the accuracy the ratio is taken
    against, so it travels with the table for the same reason T4's does.
    """
    frame = C.cross_model(summary[summary['dataset'] == dataset], 'T12')
    frame = frame[frame['condition'].isin([condition, reference])]
    if not len(frame):
        return None
    # CONDITION VARIES HERE, and saying otherwise is a lie the guard catches.
    # Every other table holds one condition; this one is a DIFFERENCE between
    # two, so both are in the frame and both have to be declared.
    G.declare(frame, 'T12', fixed={'dataset': dataset},
              varies=('model', 'rep', 'condition'))

    auc = frame.pivot_table(index=['model', 'rep'], columns='condition',
                            values='auc_norm', aggfunc='median')
    if condition not in auc.columns or reference not in auc.columns:
        return None
    change = (auc[condition] - auc[reference]).unstack()
    wobble = (frame[frame['condition'] == reference]
              .pivot_table(index='model', columns='rep',
                           values='auc_norm_spread', aggfunc='median'))
    # REP_LABELS carries the study's own ordering of the representations, and
    # there is no sort_representations() to call -- the label map IS the order.
    reps = [r for r in C.REP_LABELS if r in change.columns]
    change = change.reindex(index=C.sort_models(change.index), columns=reps)
    wobble = wobble.reindex(index=change.index, columns=reps)
    beats = (change.abs() > wobble).sum(axis=1)
    ran = change.notna().sum(axis=1)

    table = pd.DataFrame({'Model': [C.model_label(m) for m in change.index]})
    for rep in reps:
        table[C.rep_label(rep)] = change[rep].to_numpy()
    # THE YARDSTICK AS A RANGE, NOT A MEDIAN. A median over the six
    # representations is an average across representations, which this study
    # does not report, and it would also read as the number the comparison was
    # made against. The comparison in the last column is made cell by cell,
    # each change against the spread of THAT model on THAT representation.
    table['Replicate spread'] = [
        f'{wobble.loc[m].min():.3f}\u2013{wobble.loc[m].max():.3f}'
        for m in change.index]
    table['Beats it on'] = [f'{b} of {n}' for b, n in zip(beats, ran)]
    return write(table, output_dir, f'T12_condition_cost_{dataset}_{condition}',
                 f'Change in AUC_norm on moving from '
                 f'{C.condition_label(reference)} to '
                 f'{C.condition_label(condition)} on {C.dataset_label(dataset)}, '
                 f'one cell per model and representation. Negative is a loss. '
                 f'Replicate spread is the '
                 f'same model\'s highest minus lowest AUC_norm over the ten '
                 f'replicates under {C.condition_label(reference)}, given as its '
                 f'range over the representations, and the last column counts '
                 f'the representations where the change exceeds that model\'s '
                 f'own spread on that same representation. Clean accuracy is '
                 f'not repeated here; it is in T4 and in F3.')
