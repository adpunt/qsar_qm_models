#!/usr/bin/env python
"""The ten analyses that settle the open figure choices, and nothing else.

RERUN_PLAN.md 14.6 lists fourteen contingent figures and 14.9 eight open
choices. Not one can be settled by argument -- each needs a number off the real
runs. This module computes the trigger for each, prints whether it fired and
what it fired on, and writes DECISIONS.md.

It DECIDES NOTHING that is the author's. Where a choice is a preference -- which
representation is held constant -- it lays the evidence out and says so.

Each analysis returns (table, verdict). A verdict carries:
    id, question, fired, says, and the numbers behind it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figlib_config as C  # noqa: E402
import figlib_guard as G  # noqa: E402
import figlib_load as L  # noqa: E402
import figlib_metrics as M  # noqa: E402


def _verdict(ident, question, fired, says, **numbers):
    return dict(id=ident, question=question, fired=bool(fired), says=says,
                **numbers)


# ---------------------------------------------------------------------------
# D0 -- coverage, before any number is quoted
# ---------------------------------------------------------------------------

def d0_coverage(qm9, assay, merged_coverage=None):
    """What landed, what is thin, and what the declared filters remove.

    The screen lands in pieces, so a partial grid is normal. The failure this
    stops is quoting a headline off whichever cells happen to have finished.
    """
    tables, notes = {}, []
    frames = [('qm9', qm9), ('assay', assay)]
    covers = []
    for name, frame in frames:
        if frame is None or not len(frame):
            notes.append(f'{name}: nothing loaded')
            continue
        covers.append(L.coverage(frame, name).assign(source=name))
    cover = pd.concat(covers, ignore_index=True) if covers else pd.DataFrame()
    tables['d0_coverage'] = cover

    if merged_coverage is not None and len(merged_coverage):
        # Already computed by the run's own merge step, which knows what each
        # task was asked to produce. Read, not rebuilt.
        tables['d0_coverage_uncertainty_runs'] = merged_coverage
        notes.append(
            'the assay uncertainty coverage is the merge step\'s own '
            f'coverage.csv: {dict(merged_coverage["status"].value_counts())}'
            if 'status' in merged_coverage.columns else '')

    filter_rows = []
    for name, frame in frames:
        if frame is None or not len(frame):
            continue
        for filt in (L.catastrophic_filter(), L.collapsed_gp_filter()):
            _, dropped = filt.apply(frame)
            keys = [c for c in ('model', 'rep', 'condition') if c in frame.columns]
            filter_rows.append({
                'source': name, 'filter': filt.name, 'reason': filt.reason,
                'rows_dropped': int(len(dropped)),
                'cells_touched': int(dropped[keys].drop_duplicates().shape[0])
                if len(dropped) else 0})
    tables['d0_declared_filters'] = pd.DataFrame(filter_rows)

    total = int(len(cover)) if len(cover) else 0
    complete = int((cover['status'] == 'OK').sum()) if len(cover) else 0
    incomplete = total - complete
    says = (f'{complete} of {total} cells are complete. '
            + (f'{incomplete} are not, so no headline may be quoted across the '
               f'whole grid yet.' if incomplete else
               'The grid is complete.'))
    if len(cover) and cover['gp_collapsed'].sum():
        says += (f' {int((cover["gp_collapsed"] > 0).sum())} cell(s) contain '
                 f'collapsed Gaussian-process fits and are filtered before any '
                 f'number is drawn (contingent row 11).')
    return tables, _verdict(
        'D0', 'What landed, and what do the declared filters remove?',
        fired=incomplete > 0, says=says, cells=total, complete=complete,
        notes='; '.join(n for n in notes if n))


# ---------------------------------------------------------------------------
# D1 -- which representation is held constant
# ---------------------------------------------------------------------------

def d1_representation(summary, qm9=None):
    """Evidence for which representation is held constant; the choice is the
    author's.

    ⚠️ THERE IS NO MEDIAN OVER MODELS HERE, AND THAT IS THE POINT.

    The first version of this table reported `median_clean_r2` and
    `median_auc_norm` per representation. Both were wrong twice over.

    First, the variance decomposition says the choice of MODEL is the dominant
    source of variance. A median over models is therefore a median over the very
    thing that explains most of the spread: it describes which models happened
    to run, not the representation.

    Second, the medians were over DIFFERENT SETS. On the first real run `sns`
    had 16 models and 3 conditions while `ecfp4` had 19 and 7, so the two
    numbers in one column were not comparable at all. That is exactly the defect
    recorded against the paper's own Kendall's W -- a model present on six
    representations ranked against one present on two, on differently
    constituted means.

    What is reported instead:

      COVERAGE      -- how much of the grid each representation actually has.
                       A hard constraint, not a preference.
      RANK AGREEMENT-- computed WITHIN each condition, then summarised as a
                       median and a range ACROSS conditions. A representation
                       that orders the models the way the others do is a neutral
                       place to stand; one that does not, is not.
      A PAIRED TABLE-- one row per (model, condition), one column per
                       representation, nothing collapsed. That is the evidence,
                       and it is what to read before choosing.
    """
    if summary is None or not len(summary):
        return {}, _verdict('D1', 'Which representation is held constant?',
                            False, 'no robustness numbers yet')

    reps = sorted(summary['rep'].dropna().unique())
    tables = {}

    # The common roster: the models and conditions EVERY representation has.
    # Anything compared outside it is comparing different experiments.
    per_rep_models = {r: set(summary[summary['rep'] == r]['model'])
                      for r in reps}
    per_rep_conditions = {r: set(summary[summary['rep'] == r]['condition'])
                          for r in reps}
    common_models = set.intersection(*per_rep_models.values()) if reps else set()
    common_conditions = (set.intersection(*per_rep_conditions.values())
                         if reps else set())

    rows = []
    for rep in reps:
        sub = summary[summary['rep'] == rep]
        record = {
            'rep': rep,
            'n_models': int(sub['model'].nunique()),
            'n_conditions': int(sub['condition'].nunique()),
            'n_cells': int(len(sub)),
            'models_missing_vs_widest': int(
                max(len(m) for m in per_rep_models.values())
                - sub['model'].nunique()),
            'in_common_roster': bool(
                per_rep_models[rep] >= common_models
                and per_rep_conditions[rep] >= common_conditions),
        }
        spread = sub['auc_norm_spread']
        record['replicate_spread_zero_cells'] = int((spread == 0).sum())
        if qm9 is not None and 'gp_collapsed' in qm9.columns:
            record['gp_collapsed_rows'] = int(pd.to_numeric(
                qm9[qm9['rep'] == rep]['gp_collapsed'],
                errors='coerce').fillna(0).sum())
        rows.append(record)
    table = pd.DataFrame(rows)

    # Rank agreement, WITHIN one condition, then summarised across conditions.
    agreement = []
    for condition in sorted(summary['condition'].dropna().unique()):
        inside = summary[summary['condition'] == condition]
        for i, a in enumerate(reps):
            for b in reps[i + 1:]:
                got = M.profile_spearman(
                    inside, 'auc_norm', 'model', 'rep', a, b,
                    where=f'D1 rank agreement, {condition}')
                agreement.append(dict(got, condition=condition))
    agree = pd.DataFrame(agreement)
    if len(agree):
        both = pd.concat([
            agree.rename(columns={'a': 'rep', 'b': 'against'}),
            agree.rename(columns={'b': 'rep', 'a': 'against'}),
        ], ignore_index=True)[['rep', 'against', 'condition', 'rho', 'n']]
        per_rep = both.groupby('rep')['rho'].agg(['median', 'min', 'max'])
        table['rank_agreement_median'] = table['rep'].map(per_rep['median'])
        table['rank_agreement_min'] = table['rep'].map(per_rep['min'])
        table['rank_agreement_max'] = table['rep'].map(per_rep['max'])
        tables['d1_rank_agreement'] = both

    # THE EVIDENCE: one row per (model, condition), one column per
    # representation. Nothing collapsed, so the choice is made by looking at
    # models rather than at an average over them.
    paired = summary[summary['model'].isin(common_models)
                     & summary['condition'].isin(common_conditions)]
    if len(paired):
        wide = paired.pivot_table(index=['model', 'condition'], columns='rep',
                                  values='auc_norm').reset_index()
        wide.insert(2, 'best_rep', wide[list(reps)].idxmax(axis=1)
                    if set(reps) <= set(wide.columns) else '')
        tables['d1_auc_norm_by_model'] = wide
        clean = paired.pivot_table(index=['model', 'condition'], columns='rep',
                                   values='baseline_r2').reset_index()
        tables['d1_clean_r2_by_model'] = clean

        # How often each representation is the best one FOR A MODEL. A count of
        # wins over a fixed roster, which a median over models cannot give.
        wins = wide['best_rep'].value_counts()
        table['times_best_for_a_model'] = table['rep'].map(wins).fillna(0).astype(int)
        table['of_paired_cells'] = int(len(wide))

    tables['d1_representations'] = table
    complete = table[table['n_models'] == table['n_models'].max()]['rep'].tolist()
    widest_conditions = table[
        table['n_conditions'] == table['n_conditions'].max()]['rep'].tolist()
    typical = (table.sort_values('rank_agreement_median', ascending=False)
               ['rep'].tolist() if 'rank_agreement_median' in table else [])

    says = (
        f'The choice is the author\'s, and it is NOT a median over models -- '
        f'the model is the dominant source of variance, so an average over '
        f'models describes the roster and not the representation. '
        f'Widest model coverage: {", ".join(complete)}. '
        f'Widest condition coverage: {", ".join(widest_conditions)}. ')
    if typical:
        says += (f'Orders the models most like the others: {typical[0]}; least '
                 f'like them: {typical[-1]}. ')
    says += (f'The paired evidence is d1_auc_norm_by_model.csv -- '
             f'{len(common_models)} model(s) and {len(common_conditions)} '
             f'condition(s) that every representation has, one row each, '
             f'nothing collapsed.')
    zero = table[table['replicate_spread_zero_cells'] > 0]
    if len(zero):
        says += (' ⚠ ' + ', '.join(zero['rep'])
                 + ' has cells whose replicates do not differ at all, which is '
                   'a run that did not vary rather than a model that is stable.')
    return tables, _verdict(
        'D1', 'Which representation is held constant?', fired=False, says=says)


# ---------------------------------------------------------------------------
# What is missing, said in terms of the figures and tables that want it
# ---------------------------------------------------------------------------

def what_is_missing(tables, verdicts, rep):
    """Every gap, named by the figure or table that cannot be finished without it.

    `d0_coverage` already says which cells are thin. It does NOT say what that
    costs -- which panel comes out empty, which table loses a column, which
    decision cannot fire. This walks the slots and asks each one what it wanted
    and did not get, so the next run answers "what is missing" by itself instead
    of by someone reading four CSVs (the author, 2026-09-13).
    """
    said = {v['id']: v for v in verdicts}
    rows = []

    def gap(slot, wants, why, fix='', dataset='', condition='', models='',
            reps='', runnable=''):
        """`runnable` says whether submitting more tasks would close it.

        The file is handed to whoever submits jobs, so a row that cannot be
        closed by running anything has to say so -- otherwise a deep-run
        condition that covers a named subset BY DESIGN reads as a queue to
        clear (the author, 2026-09-14).
        """
        rows.append({'slot': slot, 'missing': wants, 'consequence': why,
                     'what_would_fill_it': fix, 'dataset': dataset,
                     'condition': condition, 'models': models, 'reps': reps,
                     'runnable': runnable})

    qm9 = tables.get('auc_norm_qm9')
    assay = tables.get('auc_norm_assay')
    coverage = tables.get('d0_coverage')
    excluded = tables.get('excluded_qm9')

    # -- the grid itself -----------------------------------------------------
    if coverage is not None and len(coverage):
        thin = coverage[coverage['status'] != 'OK']
        for dataset, group in thin.groupby('dataset', dropna=False):
            for condition, byc in group.groupby('condition', dropna=False):
                gap(f'every figure on {dataset}',
                    f'{len(byc)} combination(s) incomplete',
                    'no headline may be quoted across the whole grid',
                    'resubmit these model-and-representation pairs',
                    dataset=str(dataset), condition=str(condition),
                    models=';'.join(sorted(set(byc['model'].astype(str)))),
                    reps=';'.join(sorted(set(byc['rep'].astype(str)))),
                    runnable='yes')

    # -- cells that ran and were dropped ------------------------------------
    if excluded is not None and len(excluded) and 'reason' in excluded.columns:
        for reason, group in excluded.groupby('reason', dropna=False):
            conditions = sorted(set(group.get('condition', pd.Series(dtype=str))
                                    .dropna().astype(str)))
            runnable = ('no' if 'clean R2' in str(reason)
                        else 'yes' if 'level' in str(reason) else 'unknown')
            gap('F4c, F8, T4',
                f'{len(group)} cell(s) dropped: {reason}',
                'they print as "excluded" rather than as a number, and they are '
                'why a condition can look empty in one panel and full in another',
                ('adding the missing noise levels would fill these'
                 if runnable == 'yes' else
                 'a clean accuracy under the gate is a property of the fit, '
                 'not a missing task'),
                condition=';'.join(conditions),
                models=';'.join(sorted(set(group.get('model', pd.Series(dtype=str))
                                           .astype(str)))[:12]),
                runnable=runnable)

    # -- the clean level, which is the cheapest gap in the study -------------
    # A cell with no level-0 run has no denominator, so AUC_norm cannot be
    # computed for it at all and the whole cell is dropped. On the 2026-09-13
    # run that is 15 of 21 combinations for each of the three deep-run
    # conditions. It is ONE extra task per cell and it converts an excluded
    # cell into a usable one, which is why it is called out separately from
    # "the deep run covers a subset by design".
    if coverage is not None and len(coverage) and 'has_clean_level' in coverage:
        blind = coverage[~coverage['has_clean_level'].astype(bool)]
        for (dataset, condition), group in blind.groupby(
                ['dataset', 'condition'], dropna=False):
            gap('F4c, F8, T4, R15, R19, D2',
                f'{len(group)} combination(s) have no clean (level 0) run',
                'AUC_norm is a ratio to the clean score, so a cell without one '
                'is dropped entirely rather than drawn -- this is the reason '
                'these conditions look thin, and it is not the deep-run pair '
                'list',
                'ONE extra task per cell: the same model, representation and '
                'condition at noise level 0',
                dataset=str(dataset), condition=str(condition),
                models=';'.join(sorted(set(group['model'].astype(str)))),
                reps=';'.join(sorted(set(group['rep'].astype(str)))),
                runnable='yes')

    # -- conditions too thin for the comparisons ----------------------------
    not_judged = said.get('D2', {}).get('not_judged') or []
    if not_judged:
        gap('F3, D2, R19',
            f'{len(not_judged)} noise condition(s) share too few cells with the '
            f'rest of the grid to compare',
            'they are neither a main-text panel nor an additional file, because '
            'nothing can be said about whether their picture repeats another',
            'widening these conditions beyond the deep-run pair list would '
            'close it; the deep run covers a named subset BY DESIGN, so this '
            'is a decision to widen the study and not a queue to clear',
            condition=';'.join(not_judged), runnable='decision')

    # -- the deep run, condition by condition -------------------------------
    if qm9 is not None and len(qm9):
        widest = qm9.groupby('condition')['model'].nunique()
        full = int(widest.max()) if len(widest) else 0
        for condition, n_models in widest.items():
            if full and n_models < full:
                ran = sorted(set(qm9[qm9['condition'] == condition]['model']
                                 .astype(str)))
                absent = sorted(set(qm9['model'].astype(str)) - set(ran))
                gap('F4c, R15, T4, R19',
                    f'{n_models} of {full} models have a robustness number',
                    'the column cannot rank the roster, so the condition is '
                    'shown in R19 over the pairs that ran rather than in the '
                    'main grid',
                    'submitting the absent models for this condition would '
                    'let it rejoin the main grid',
                    dataset='qm9', condition=str(condition),
                    models=';'.join(absent[:12]), runnable='decision')

    # -- the uncertainty side -----------------------------------------------
    for name, key, why in (
            ('F6, T6', 'unc_q5', 'the decomposition figure and the uncertainty '
                                 'table have nothing to draw'),
            ('T6, D7', 'd7_q4', 'whether uncertainty finds the corrupted labels '
                                'cannot be answered')):
        if tables.get(key) is None or not len(tables.get(key, [])):
            gap(name, f'{key} is empty', why,
                'the uncertainty runs for these conditions')

    # -- the assay side ------------------------------------------------------
    if assay is not None and len(assay):
        for dataset, group in assay.groupby('dataset', dropna=False):
            here = group[group['rep'] == rep] if rep else group
            if not len(here):
                gap('F8, T4, T7', f'{dataset} has nothing at {rep}',
                    'that dataset drops out of every table held at the primary '
                    'representation',
                    f'the {dataset} tasks at {rep}')
    else:
        gap('F8, T4, T7, D9', 'no assay robustness at all',
            'the transfer question cannot be asked',
            'the KIRBy validation runs')

    table = pd.DataFrame(rows, columns=[
        'slot', 'missing', 'consequence', 'what_would_fill_it', 'dataset',
        'condition', 'models', 'reps', 'runnable'])
    # Rows that a submission can close come first; 'no' rows are properties of
    # the fits and 'decision' rows need the author to widen the study.
    order = {'yes': 0, 'decision': 1, 'unknown': 2, 'no': 3, '': 4}
    if len(table):
        table = table.assign(
            _o=table['runnable'].map(lambda v: order.get(str(v), 4))
        ).sort_values('_o').drop(columns='_o').reset_index(drop=True)
    return table


# ---------------------------------------------------------------------------
# Accuracy per model at each noise level, across representations
# ---------------------------------------------------------------------------

def accuracy_across_representations(accuracy, dataset='qm9',
                                    condition='gaussian'):
    """R2 per model at every noise level, averaged over the representations.

    The author asked for this in as many words and got win counts instead.
    It is one row per model and noise level: the mean R2 over the
    representations, with the lowest and the highest so the average is never
    read without its spread, and the count so nobody reads a mean of two as a
    mean of six.

    THE MEAN OVER REPRESENTATIONS IS A DECISION AID, NOT A PAPER NUMBER.
    Representation is a factor and no table in the paper averages over it
    (RERUN_PLAN.md 14.2); this table exists so the author can decide which
    models to carry through the cross-model figures, and it says so in its own
    header row.
    """
    if accuracy is None or not len(accuracy):
        return {}, _verdict('A', 'How accurate is each model, level by level?',
                            False, 'no accuracy rows yet')
    frame = accuracy[(accuracy['dataset'] == dataset)
                     & (accuracy['condition'] == condition)]
    if not len(frame):
        return {}, _verdict('A', 'How accurate is each model, level by level?',
                            False, f'nothing at {dataset} under {condition}')
    # Median over replicates first -- that is the only axis that may be
    # collapsed -- then across representations.
    per_cell = (frame.groupby(['model', 'rep', 'sigma'], dropna=False)['r2']
                .median().reset_index())
    out = (per_cell.groupby(['model', 'sigma'], dropna=False)['r2']
           .agg(mean_r2='mean', lowest_r2='min', highest_r2='max',
                n_representations='size')
           .reset_index())
    out.insert(0, 'dataset', dataset)
    out.insert(1, 'condition', condition)

    levels = sorted(out['sigma'].dropna().unique())
    top = levels[-1] if levels else None
    clean = levels[0] if levels else None
    says = (f'R2 for {out["model"].nunique()} model(s) at '
            f'{len(levels)} noise level(s) on {C.dataset_label(dataset)} under '
            f'{C.condition_label(condition)}, averaged over the '
            f'{int(out["n_representations"].max())} representations that ran. ')
    if clean is not None and top is not None and clean != top:
        at_clean = out[out['sigma'] == clean].set_index('model')['mean_r2']
        at_top = out[out['sigma'] == top].set_index('model')['mean_r2']
        shared = at_clean.index.intersection(at_top.index)
        if len(shared):
            drop = (at_clean.loc[shared] - at_top.loc[shared])
            says += (f'From level {clean:g} to level {top:g} the mean R2 falls '
                     f'by between {drop.min():.3f} and {drop.max():.3f}; '
                     f'least for {C.model_label(drop.idxmin())} and most for '
                     f'{C.model_label(drop.idxmax())}.')
    says += (' The mean over representations is a decision aid only -- '
             'representation is a factor and no paper table averages over it.')
    return ({'model_accuracy_by_level': out},
            _verdict('A', 'How accurate is each model, level by level?',
                     fired=False, says=says))


def standout_pairs(qm9_summary, assay_summary, condition='gaussian',
                   top_n=15):
    """Which model-and-representation pairs are robust on more than one dataset.

    RAW NUMBERS, NOT A DERIVED SCORE. One row per pair, its clean R2 and its
    AUC_norm on each dataset it ran on, and a count of how many datasets it
    reached the top `top_n` on. The author asked for the data and got a
    constructed "surplus" instead (2026-09-14); this is the data.

    A pair that tops a list on a clean R2 near 0.35 is flagged: AUC_norm is a
    ratio and a small denominator inflates it, which D6 already reports.
    """
    frames = [f for f in (qm9_summary, assay_summary)
              if f is not None and len(f)]
    if not frames:
        return {}, _verdict('P', 'Which pairs are robust on more than one '
                            'dataset?', False, 'no robustness numbers yet')
    d = pd.concat(frames, ignore_index=True)
    d = C.cross_model(d, 'the standout pairs')
    d = d[d['condition'] == condition].dropna(subset=['auc_norm'])
    if not len(d):
        return {}, _verdict('P', 'Which pairs are robust on more than one '
                            'dataset?', False, f'nothing under {condition}')
    d = d.copy()
    # ACCURACY AND ROBUSTNESS COUNT EQUALLY, and the two are put on a common
    # footing by scaling each to 0-1 WITHIN its own dataset -- a clean R2 of
    # 0.40 is near the top on Caco-2 and near the bottom on QM9, so a raw
    # average across datasets would just rank the datasets. The author's shape,
    # 2026-09-14. This changes the answer: ranking on AUC_norm alone puts the
    # forests on ChemBERTa first, and once accuracy counts they fall to the
    # middle because ChemBERTa is not accurate on the assay sets.
    for column in ('baseline_r2', 'auc_norm'):
        low = d.groupby('dataset')[column].transform('min')
        high = d.groupby('dataset')[column].transform('max')
        span = (high - low).replace(0, np.nan)
        d[f'{column}_scaled'] = (d[column] - low) / span
    d['combined_scaled'] = (d['baseline_r2_scaled']
                            + d['auc_norm_scaled']) / 2
    d['rank_in_dataset'] = d.groupby('dataset')['auc_norm'].rank(
        ascending=False)
    d['in_top'] = d['rank_in_dataset'] <= top_n
    d['auc_norm_above_one'] = d['auc_norm'] > C.AUC_NORM_IMPLAUSIBLE_HIGH

    rows = []
    for (model, rep), group in d.groupby(['model', 'rep'], dropna=False):
        rec = {'model': model, 'rep': rep, 'condition': condition,
               'combined_scaled': float(group['combined_scaled'].mean()),
               'datasets_run': int(group['dataset'].nunique()),
               'datasets_in_top': int(group['in_top'].sum()),
               'lowest_auc_norm': float(group['auc_norm'].min()),
               'any_auc_norm_above_one': bool(
                   group['auc_norm_above_one'].any())}
        for _, r in group.iterrows():
            rec[f'{r["dataset"]}_clean_r2'] = float(r['baseline_r2'])
            rec[f'{r["dataset"]}_auc_norm'] = float(r['auc_norm'])
        rows.append(rec)
    table = pd.DataFrame(rows)
    # Ranked by the combined score over the datasets a pair ran on ALL of --
    # a pair that ran on one dataset cannot be compared with one that ran on
    # four, so the ranking column is only filled where the coverage is equal.
    widest = int(table['datasets_run'].max()) if len(table) else 0
    table['comparable'] = table['datasets_run'] == widest
    table = table.sort_values(['comparable', 'combined_scaled'],
                              ascending=False).reset_index(drop=True)

    comparable = table[table['comparable'] & (~table['any_auc_norm_above_one'])]
    if len(comparable):
        named = '; '.join(
            f'{C.model_label(r.model)} on {C.rep_label(r.rep)} '
            f'({r.combined_scaled:.3f})'
            for r in comparable.head(4).itertuples())
        says = (f'{len(comparable)} pair(s) ran on all {widest} dataset(s). '
                f'Ranked on accuracy and robustness together, each scaled '
                f'within its own dataset: {named}.')
    else:
        says = 'No pair ran on every dataset.'
    robust_only = table[(table['datasets_in_top'] >= table['datasets_run'])
                        & (table['datasets_run'] >= 2)
                        & (~table['any_auc_norm_above_one'])]
    if len(robust_only):
        says += (' On ROBUSTNESS alone, reaching the top '
                 f'{top_n} on every dataset they ran: '
                 + '; '.join(f'{C.model_label(r.model)} on {C.rep_label(r.rep)}'
                             for r in robust_only.head(4).itertuples()) + '.')
    says += (' Pairs whose AUC_norm exceeds one are flagged rather than ranked: '
             'the metric is a ratio and a clean baseline near 0.35 inflates it.')
    return ({'standout_pairs': table},
            _verdict('P', 'Which pairs are robust on more than one dataset?',
                     fired=False, says=says))


def uncertainty_pairs(q4, statistic='rho_ratio', sigma=None):
    """Which pairs are best at FINDING the corrupted labels. Raw numbers.

    One row per model and representation: the statistic, the permutation band it
    is read against, and whether it cleared the band. No derived score. The
    author asked whether any pair is unusually good at this the way the boosted
    trees are unusually good on Sort & Slice for robustness.
    """
    if q4 is None or not len(q4) or statistic not in q4.columns:
        return {}, _verdict('U', 'Which pairs find the corrupted labels?',
                            False, 'no q4 rows yet')
    frame = q4.dropna(subset=[statistic]).copy()
    if not len(frame):
        return {}, _verdict('U', 'Which pairs find the corrupted labels?',
                            False, f'{statistic} is empty')
    if sigma is not None and 'sigma' in frame.columns:
        frame = frame[frame['sigma'] == sigma]
    keys = [c for c in ('dataset', 'model', 'rep', 'condition', 'sigma')
            if c in frame.columns]
    table = (frame.groupby(keys, dropna=False)
             .agg(statistic_median=(statistic, 'median'),
                  null_lo=('null_lo', 'median'),
                  null_hi=('null_hi', 'median'),
                  folds_outside_null=('outside_null', 'sum'),
                  folds=(statistic, 'size'))
             .reset_index()
             .sort_values('statistic_median', ascending=False))
    table['clears_the_band'] = (table['folds_outside_null']
                                >= 0.5 * table['folds'])
    clear = table[table['clears_the_band']]
    if len(clear):
        named = '; '.join(
            f'{C.model_label(r.model)} on {C.rep_label(r.rep)} '
            f'({statistic} {r.statistic_median:.3f})'
            for r in clear.head(4).itertuples())
        says = (f'{len(clear)} of {len(table)} model-and-representation cells '
                f'clear their permutation band on {statistic}. Highest: '
                f'{named}.')
    else:
        says = (f'No cell clears its permutation band on {statistic}, over '
                f'{len(table)} cells.')
    return ({'uncertainty_pairs': table},
            _verdict('U', 'Which pairs find the corrupted labels?',
                     fired=False, says=says))


# ---------------------------------------------------------------------------
# D2 -- how many noise conditions does F3 show
# ---------------------------------------------------------------------------

#: Two grids "agree" when they order the cells the same way AND differ by less
#: than the run-to-run wobble. Both, because either alone is satisfiable by an
#: uninteresting accident: a perfect rank match on tiny differences, or tiny
#: differences with the order scrambled.
GRID_AGREEMENT_RHO = 0.90

#: How much of the grid a pair of conditions must actually share before their
#: agreement means anything. On the 9 September run the three deep-run
#: conditions overlapped the others on SIX model-and-representation cells out of
#: about 110, and six cells gave grouped_wider vs laplace a rank correlation of
#: exactly 1.000 -- which sent Laplace to an additional file as a "repeat" of a
#: grid it had never been run against. Six cells is not a grid. A pair below
#: this share is recorded, reported, and never clustered.
GRID_MIN_SHARED_SHARE = 0.5
GRID_MIN_SHARED_CELLS = 8


def d2_grid_similarity(summary, per_replicate=None):
    """Which noise conditions give the same model-by-representation grid.

    RERUN_PLAN.md 14.6 row 14. The rule is fixed in advance and the answer comes
    off the data: show the conditions whose grids DIFFER; a condition whose grid
    repeats another's goes to an additional file. If every grid differs, every
    grid is a main-text panel and F3 becomes the largest figure in the paper.
    """
    if summary is None or not len(summary):
        return {}, _verdict('D2', 'How many noise conditions does F3 show?', False,
                            'no robustness numbers yet')
    usable = G.ranking_conditions(sorted(summary['condition'].dropna().unique()))
    wobble = (float(per_replicate.groupby(
        ['model', 'rep', 'condition'])['auc_norm'].std().median())
        if per_replicate is not None and len(per_replicate) else np.nan)

    rows = []
    for i, a in enumerate(usable):
        for b in usable[i + 1:]:
            ga = summary[summary['condition'] == a].set_index(
                ['model', 'rep'])['auc_norm']
            gb = summary[summary['condition'] == b].set_index(
                ['model', 'rep'])['auc_norm']
            shared = ga.index.intersection(gb.index)
            if len(shared) < 3:
                continue
            x, y = ga.loc[shared].astype(float), gb.loc[shared].astype(float)
            rho, p = stats.spearmanr(x, y)
            mad = float(np.abs(x - y).mean())
            rows.append({'a': a, 'b': b, 'rho': float(rho), 'p_value': float(p),
                         'mean_abs_difference': mad,
                         'replicate_wobble': wobble,
                         'n_cells': int(len(shared))})
    table = pd.DataFrame(rows)

    # THE OVERLAP GATE. A pair is only allowed to declare two grids the same if
    # the two grids were actually run over the same ground. The widest overlap
    # in the table is what a full comparison looks like; anything under half of
    # it is a handful of cells and says nothing about the rest of the roster.
    thin = pd.Series(dtype=bool)
    if len(table):
        widest = int(table['n_cells'].max())
        floor = max(GRID_MIN_SHARED_CELLS,
                    int(round(GRID_MIN_SHARED_SHARE * widest)))
        table['enough_overlap'] = table['n_cells'] >= floor
        table['overlap_floor'] = floor
        table['grids_agree'] = (
            table['enough_overlap']
            & (table['rho'] >= GRID_AGREEMENT_RHO)
            & np.isfinite(wobble)
            & (table['mean_abs_difference'] <= wobble))
        thin = ~table['enough_overlap']

    # One representative per cluster of agreeing conditions.
    parent = {c: c for c in usable}

    def root(x):
        while parent[x] != x:
            x = parent[x]
        return x

    for r in table[table['grids_agree']].itertuples() if len(table) else []:
        parent[root(r.b)] = root(r.a)
    clusters = {}
    for c in usable:
        clusters.setdefault(root(c), []).append(c)
    main_text = C.sort_conditions([members[0] for members in clusters.values()])
    supplementary = [c for c in usable if c not in main_text]

    # A condition every one of whose comparisons was too thin is not a repeat of
    # anything and must not be described as one. It is a condition the grid does
    # not yet cover, and that is a different sentence in the paper.
    judged = set()
    if len(table):
        ok = table[table['enough_overlap']]
        judged = set(ok['a']) | set(ok['b'])
    unjudged = [c for c in usable if c not in judged]
    supplementary = [c for c in supplementary if c not in unjudged]
    main_text = [c for c in main_text if c not in unjudged]

    says = (f'F3 shows {len(main_text)} panel(s): '
            + ', '.join(C.condition_label(c) for c in main_text) + '. '
            + (f'Held back as repeats of one of those: '
               f'{", ".join(C.condition_label(c) for c in supplementary)}. '
               if supplementary else
               'Every grid that could be compared differs from every other, '
               'so each is a main-text panel and F3 is the largest figure in '
               'the paper. ')
            + (f'{len(unjudged)} condition(s) could not be judged either way -- '
               f'{", ".join(C.condition_label(c) for c in unjudged)} share too '
               f'few model-and-representation cells with the rest of the grid '
               f'to say whether their picture repeats it '
               f'({int(table.loc[thin, "n_cells"].max()) if thin.any() else 0} '
               f'cells at most, against a floor of '
               f'{int(table["overlap_floor"].iloc[0]) if len(table) else 0}). '
               f'They are neither main text nor an additional file until the '
               f'deep run fills them in.'
               if unjudged else ''))
    return ({'d2_grid_similarity': table},
            _verdict('D2', 'How many noise conditions does F3 show, and which?',
                     fired=bool(supplementary), says=says,
                     main_text=main_text, supplementary=supplementary,
                     not_judged=unjudged, replicate_wobble=wobble))


# ---------------------------------------------------------------------------
# D3 -- do the noise conditions separate the models
# ---------------------------------------------------------------------------

def d3_condition_separation(per_replicate, summary, rep):
    """Paired signed-rank across models, condition against condition.

    RERUN_PLAN.md 14.6 rows 7 and 8, and 14.9 item 7 -- the test that exists
    nowhere and costs no compute. Paired on the MODEL, so n is the roster size
    and the test can reach significance; the assay datasets have five folds and
    a two-sided signed-rank test on five pairs cannot go below p = 0.0625.
    """
    if summary is None or not len(summary):
        return {}, _verdict('D3', 'Do the noise conditions separate the models?',
                            False, 'no robustness numbers yet')
    one = summary[summary['rep'] == rep]
    usable = G.ranking_conditions(sorted(one['condition'].dropna().unique()))
    rows = []
    for i, a in enumerate(usable):
        for b in usable[i + 1:]:
            got = M.wilcoxon_paired(one[one['condition'].isin([a, b])],
                                    'auc_norm', 'condition', a, b,
                                    pair_on='model')
            rows.append(dict(got, rep=rep))
    paired = pd.DataFrame(rows)

    # And the plain question: is the spread across conditions bigger than the
    # spread across replicates of one condition?
    spread_rows = []
    for model, group in one.groupby('model'):
        across = float(group['auc_norm'].max() - group['auc_norm'].min())
        within = float(group['auc_norm_spread'].median())
        spread_rows.append({'model': model, 'rep': rep,
                            'spread_across_conditions': across,
                            'replicate_spread': within,
                            'exceeds_wobble': bool(across > within)})
    spread = pd.DataFrame(spread_rows)

    kendall = M.kendalls_w(summary, rep)
    # Lists do not survive a CSV round trip as lists, and a column holding
    # "['gaussian', 'laplace']" is unreadable in a spreadsheet.
    kendall_row = {k: ('; '.join(str(x) for x in v) if isinstance(v, list) else v)
                   for k, v in kendall.items()}
    n_sig = int(paired['significant'].sum()) if len(paired) else 0
    n_exceed = int(spread['exceeds_wobble'].sum()) if len(spread) else 0
    fired = n_sig > 0 or n_exceed > len(spread) / 2
    says = (f'{n_sig} of {len(paired)} condition pairs differ significantly on '
            f'{C.rep_label(rep)}; {n_exceed} of {len(spread)} models vary more '
            f'across conditions than across replicates. '
            f"Kendall's W = {kendall['kendall_w']:.3f} "
            f"(p = {kendall['p_value']:.2g}, {kendall['n_models']} models, "
            f"{kendall['n_conditions']} condition(s) -- "
            f"{kendall_row.get('conditions', '')} -- at {C.rep_label(rep)}). "
            + (f"{kendall['reason'].capitalize()}. "
               if kendall.get('conditions_dropped') else '')
            + ('The kind of noise matters, so F4 gets its one-line-per-condition '
               'panel (row 7).' if fired else
               'Only the AMOUNT matters, not the kind -- which is a headline '
               'finding in its own right (row 8).'))
    return ({'d3_condition_pairs': paired, 'd3_condition_spread': spread,
             'd3_kendall_w': pd.DataFrame([kendall_row])},
            _verdict('D3', 'Do the noise conditions separate the models?',
                     fired=fired, says=says, n_significant=n_sig,
                     kendall_w=kendall['kendall_w']))


# ---------------------------------------------------------------------------
# D4 -- does the interaction dominate
# ---------------------------------------------------------------------------

def d4_interaction(anova):
    """RERUN_PLAN.md 14.6 row 12: if the pairing beats both main effects, the
    simple-effects table is promoted to a panel on F2."""
    if anova is None or not len(anova):
        return {}, _verdict('D4', 'Does the interaction term dominate?', False,
                            'no decomposition yet')
    table = anova.copy()
    table['interaction_dominates'] = (
        (table['eta2_interaction'] > table['eta2_model'])
        & (table['eta2_interaction'] > table['eta2_rep']))
    n = int(table['interaction_dominates'].sum())
    fired = n > 0
    says = (f'The model-representation pairing is the largest structured source '
            f'of variance in {n} of {len(table)} condition-and-outcome rows. '
            + ('Simple effects are promoted from a supplementary table to an F2 '
               'panel (row 12).' if fired else
               'Main effects can be read directly; no extra panel is needed.'))
    return {'d4_interaction': table}, _verdict(
        'D4', 'Does the interaction term dominate?', fired=fired, says=says,
        n_rows=n)


# ---------------------------------------------------------------------------
# D5 -- is one representation an outlier
# ---------------------------------------------------------------------------

def d5_representation_outlier(summary, condition=None):
    """RERUN_PLAN.md 14.6 row 6. The case the averaging rules exist to catch:
    a model whose AUC_norm at one representation sits outside the range of its
    others, dragging any summary that averages over representation."""
    if summary is None or not len(summary):
        return {}, _verdict('D5', 'Is one representation an outlier?', False,
                            'no robustness numbers yet')
    frame = summary if condition is None else summary[
        summary['condition'] == condition]
    rows = []
    for (model, cond), group in frame.groupby(['model', 'condition']):
        if group['rep'].nunique() < 3:
            continue
        for _, row in group.iterrows():
            others = group[group['rep'] != row['rep']]['auc_norm'].astype(float)
            wobble = float(group['auc_norm_spread'].median())
            distance = float(min(abs(row['auc_norm'] - others.min()),
                                 abs(row['auc_norm'] - others.max())))
            outside = bool(row['auc_norm'] < others.min()
                           or row['auc_norm'] > others.max())
            rows.append({
                'model': model, 'condition': cond, 'rep': row['rep'],
                'auc_norm': float(row['auc_norm']),
                'others_min': float(others.min()),
                'others_max': float(others.max()),
                'distance_outside': distance if outside else 0.0,
                'replicate_spread': wobble,
                'is_outlier': bool(outside and distance > wobble)})
    table = pd.DataFrame(rows)
    hits = table[table['is_outlier']] if len(table) else table
    fired = bool(len(hits))
    if fired:
        worst = hits.sort_values('distance_outside', ascending=False).iloc[0]
        says = (f'{len(hits)} model-and-representation cell(s) sit outside the '
                f'range of that model\'s other representations by more than the '
                f'replicate spread. The largest is '
                f'{C.model_label(worst["model"])} on '
                f'{C.rep_label(worst["rep"])} under '
                f'{C.condition_label(worst["condition"])}, '
                f'{worst["distance_outside"]:.3f} outside. F3 option 3B -- the '
                f'representation-profile lines -- goes in the main text (row 6), '
                f'and no summary may average over representation.')
    else:
        says = ('No model has a representation that sits outside the range of '
                'its others by more than the replicate spread.')
    return {'d5_representation_outlier': table}, _verdict(
        'D5', 'Is one representation an outlier?', fired=fired, says=says,
        n_outliers=int(len(hits)))


# ---------------------------------------------------------------------------
# D6 -- AUC_norm above 1
# ---------------------------------------------------------------------------

def d6_auc_above_one(summaries, per_replicates=None):
    """RERUN_PLAN.md 14.6 row 10. Counted beside its clean baseline, never
    patched: 7.3 says this is structural and will reappear.

    BOTH GRANULARITIES, because they disagree and the disagreement is the
    finding. A cell's MEDIAN can sit under 1.05 while several of its replicates
    are over it -- the first run on real data warned about 67 replicate values
    and then reported "0 of 317 cells", which is two true numbers that read as a
    contradiction. Reporting one without the other is failure mode 12.

    And on every dataset, not just QM9: the assay side had 59 of the 67.
    """
    frames = [f for f in (summaries if isinstance(summaries, (list, tuple))
                          else [summaries]) if f is not None and len(f)]
    if not frames:
        return {}, _verdict('D6', 'Does AUC_norm exceed 1?', False,
                            'no robustness numbers yet')
    summary = pd.concat(frames, ignore_index=True)
    hits = summary[summary['auc_norm'] > C.AUC_NORM_IMPLAUSIBLE_HIGH].copy()

    reps = [f for f in (per_replicates or []) if f is not None and len(f)]
    rep_hits = pd.DataFrame()
    n_rep_total = 0
    if reps:
        allreps = pd.concat(reps, ignore_index=True)
        n_rep_total = int(len(allreps))
        rep_hits = allreps[
            allreps['auc_norm'] > C.AUC_NORM_IMPLAUSIBLE_HIGH].copy()

    fired = bool(len(hits) or len(rep_hits))
    says = (f'{len(hits)} of {len(summary)} cell medians and '
            f'{len(rep_hits)} of {n_rep_total} individual replicates score '
            f'above {C.AUC_NORM_IMPLAUSIBLE_HIGH} -- retaining more than they '
            f'started with. ')
    if len(rep_hits) and not len(hits):
        says += ('The medians are all under the line and some replicates are '
                 'over it, so this is run-to-run spread rather than a model '
                 'that improves under noise. ')
    if fired:
        says += ('Clean baselines are printed beside them; this gets an '
                 'additional file and a Methods sentence, not a patched metric '
                 '(row 10).')
    tables = {'d6_auc_above_one': hits.sort_values('auc_norm', ascending=False)}
    if len(rep_hits):
        tables['d6_auc_above_one_replicates'] = rep_hits.sort_values(
            'auc_norm', ascending=False)
    return tables, _verdict(
        'D6', 'Does AUC_norm exceed 1?', fired=fired, says=says,
        n_cell_medians=int(len(hits)), n_replicates=int(len(rep_hits)))


# ---------------------------------------------------------------------------
# D7 -- which F7 option runs
# ---------------------------------------------------------------------------

def d7_uncertainty_option(q4, q6, support):
    """RERUN_PLAN.md 14.6 rows 1 to 3, which pick between 7A, 7B and 7C."""
    if q4 is None or not len(q4):
        return {}, _verdict('D7', 'Which F7 option runs?', False,
                            'no out-of-fold uncertainty rows yet')
    table = q4.copy()
    outside = table.get('outside_null')
    have_band = outside is not None and outside.notna().any()
    table['fires'] = outside.fillna(False) if have_band else False

    if not have_band:
        # Without the permutation band there is no way to tell "adds nothing"
        # from "was never tested", and the two look identical in the data: every
        # `fires` is False either way. Reporting the first would put a null in
        # the paper that no test produced.
        tables = {'d7_q4': table}
        if support is not None and len(support):
            tables['d7_support'] = support
        return tables, _verdict(
            'D7', 'Which F7 option runs?', fired=False,
            says='No permutation band was computed, so rows 1 to 3 cannot be '
                 'read. Every auc_delta looks the same as every other without '
                 'one, and "the uncertainty adds nothing" is then a statement '
                 'about the analysis rather than about the models. Re-run '
                 'without --permutations 0.',
            option='undecided', band_computed=False)

    censoring = table[table['condition'].astype(str).str.startswith('censoring')]
    row1 = bool(censoring['fires'].any()) if len(censoring) else False

    row2 = False
    q6_summary = pd.DataFrame()
    if q6 is not None and len(q6):
        q6_summary = (q6.groupby(['dataset', 'model', 'rep', 'condition'],
                                 dropna=False)['rho_unc_vs_clean_error']
                      .median().reset_index())
        row2 = bool((q6_summary['rho_unc_vs_clean_error'] > 0.2).any())

    row3 = bool(len(table) and not table['fires'].any())

    if row1:
        option, says = '7B', (
            'Censoring: the uncertainty finds clipped labels better than the '
            'error alone, outside the permutation band. F7 is the enrichment '
            'curve (7B) and it goes in the main text -- it is the strongest '
            'possible answer to the paper\'s own title (row 1).')
    elif row2:
        option, says = '7A', (
            'The uncertainty ranks the error against the clean label well above '
            'zero. F7 is the error-retention curve (7A), which works under '
            'every condition (row 2).')
    elif row3:
        option, says = '7C', (
            'Every auc_delta sits inside its permutation band: the uncertainty '
            'adds nothing anywhere. F7 is the grid across conditions (7C), so '
            'the null is SHOWN rather than asserted. A clean null is a result '
            'and gets a figure, not a sentence (row 3).')
    else:
        option, says = 'undecided', (
            'Some cells fire and some do not, and none of rows 1 to 3 fires '
            'cleanly. Read d7_q4.csv before choosing.')

    tables = {'d7_q4': table}
    if len(q6_summary):
        tables['d7_q6'] = q6_summary
    if support is not None and len(support):
        tables['d7_support'] = support
    return tables, _verdict(
        'D7', 'Which F7 option runs?', fired=option != 'undecided', says=says,
        option=option, censoring_fires=row1, error_ranking_holds=row2,
        clean_null=row3)


# ---------------------------------------------------------------------------
# D8 -- does the decomposition work
# ---------------------------------------------------------------------------

def d8_decomposition(slopes, support):
    """RERUN_PLAN.md 14.6 rows 4, 5 and 13."""
    if slopes is None or not len(slopes):
        return ({'d8_support': support} if support is not None else {},
                _verdict('D8', 'Does the decomposition work?', False,
                         'no component slopes yet'))
    counts = slopes['verdict'].value_counts().to_dict()
    separates = int(slopes.get('separates', pd.Series(dtype=bool)).sum())
    failed = int(slopes['verdict'].str.startswith('BOTH').sum())
    says = ('; '.join(f'{v}: {k}' for k, v in counts.items()) + '. '
            + (f'{separates} pair(s) attribute added label noise to the data '
               f'rather than to themselves, so F6 is the headline as settled '
               f'(row 4).' if separates else '')
            + (f' {failed} pair(s) show BOTH components rising together, which '
               f'is what the failure looks like -- F6 still runs, reframed as '
               f'the negative result, with the support flags printed beside '
               f'each panel (row 5).' if failed else ''))
    tables = {'d8_component_slopes': slopes}
    if support is not None and len(support):
        tables['d8_support'] = support
    return tables, _verdict(
        'D8', 'Does the aleatoric/epistemic split work?',
        fired=bool(separates or failed), says=says,
        n_separates=separates, n_failed=failed)


# ---------------------------------------------------------------------------
# D9 -- rank transfer, QM9 against the assay datasets
# ---------------------------------------------------------------------------

def d9_rank_transfer(qm9_summary, assay_summary):
    """RERUN_PLAN.md 14.6 row 9. One table PER REPRESENTATION -- the transfer
    question is exactly where averaging over representation would hide the
    answer."""
    if (qm9_summary is None or not len(qm9_summary)
            or assay_summary is None or not len(assay_summary)):
        return {}, _verdict('D9', 'Do QM9 and the assay datasets agree?', False,
                            'need both sides')
    rows, correlations = [], []
    for rep in sorted(set(qm9_summary['rep']) & set(assay_summary['rep'])):
        for condition in sorted(set(qm9_summary['condition'])
                                & set(assay_summary['condition'])):
            q = (qm9_summary[(qm9_summary['rep'] == rep)
                             & (qm9_summary['condition'] == condition)]
                 .set_index('model')['auc_norm'])
            if not len(q):
                continue
            q_rank = q.rank(ascending=False)
            for dataset in sorted(assay_summary['dataset'].unique()):
                a = (assay_summary[(assay_summary['rep'] == rep)
                                   & (assay_summary['condition'] == condition)
                                   & (assay_summary['dataset'] == dataset)]
                     .set_index('model')['auc_norm'])
                shared = q.index.intersection(a.index)
                if len(shared) < 3:
                    continue
                a_rank = a.loc[shared].rank(ascending=False)
                rho, p = stats.spearmanr(q_rank.loc[shared], a_rank)
                correlations.append({'rep': rep, 'condition': condition,
                                     'dataset': dataset, 'rho': float(rho),
                                     'p_value': float(p),
                                     'n_models': int(len(shared))})
                for model in shared:
                    rows.append({
                        'rep': rep, 'condition': condition, 'model': model,
                        'dataset': dataset,
                        'qm9_auc_norm': float(q.loc[model]),
                        'qm9_rank': float(q_rank.loc[model]),
                        'assay_auc_norm': float(a.loc[model]),
                        'assay_rank': float(a_rank.loc[model]),
                        'rank_change': float(a_rank.loc[model]
                                             - q_rank.loc[model])})
    transfer = pd.DataFrame(rows)
    agreement = pd.DataFrame(correlations)
    weak = agreement[agreement['rho'] < 0.5] if len(agreement) else agreement
    fired = bool(len(weak))
    says = ((f'Median rank agreement {agreement["rho"].median():.3f} across '
             f'{len(agreement)} representation-condition-dataset combinations. ')
            if len(agreement) else 'no comparable cells. ')
    says += ('T7 is promoted to a figure: the two sides disagree on the model '
             'ranking in ' f'{len(weak)} combination(s) (row 9).' if fired
             else 'The ranking transfers; T7 stays a table.')
    return ({'d9_rank_transfer': transfer, 'd9_rank_agreement': agreement},
            _verdict('D9', 'Does the model ranking transfer to assay data?',
                     fired=fired, says=says,
                     median_rho=(float(agreement['rho'].median())
                                 if len(agreement) else np.nan)))


# ---------------------------------------------------------------------------
# D10 -- the probabilistic transformations
# ---------------------------------------------------------------------------

def d10_probabilistic(per_replicate):
    """Table T5's content: one row per pair, one column per representation.

    No averaging. The current table's single number per row is a mean across
    representations -- averaging defect 1 in miniature.
    """
    if per_replicate is None or not len(per_replicate):
        return {}, _verdict('D10', 'Do probabilistic models resist noise better?',
                            False, 'no robustness numbers yet')
    rows = []
    for base, variant in C.PROBABILISTIC_PAIRS:
        for rep in sorted(per_replicate['rep'].dropna().unique()):
            for condition in sorted(per_replicate['condition'].dropna().unique()):
                sub = per_replicate[
                    (per_replicate['rep'] == rep)
                    & (per_replicate['condition'] == condition)
                    & (per_replicate['model'].isin([base, variant]))]
                if sub['model'].nunique() < 2:
                    continue
                got = M.wilcoxon_paired(sub, 'auc_norm', 'model', base, variant,
                                        pair_on='replicate')
                rows.append(dict(got, base=base, variant=variant, rep=rep,
                                 condition=condition))
    table = pd.DataFrame(rows)
    if not len(table):
        return {}, _verdict('D10', 'Do probabilistic models resist noise better?',
                            False, 'no pair had both halves')
    wins = table[table['significant'] & (table['median_change'] > 0)]
    losses = table[table['significant'] & (table['median_change'] < 0)]
    says = (f'{len(table)} pair-representation-condition comparisons: '
            f'{len(wins)} where the probabilistic version is significantly more '
            f'robust, {len(losses)} where it is significantly less. '
            f'Paired on the replicate, one row per representation, no averaging.')
    return {'d10_probabilistic': table}, _verdict(
        'D10', 'Do probabilistic models resist noise better?',
        fired=bool(len(wins) or len(losses)), says=says,
        n_wins=int(len(wins)), n_losses=int(len(losses)))


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def write_report(verdicts, tables, output_dir, context=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        if name.startswith('_'):
            continue          # carried between steps, not an output table
        if table is not None and len(table):
            table.to_csv(output_dir / f'{name}.csv', index=False)

    lines = [
        '# Decisions off the data',
        '',
        'Every open choice in `RERUN_PLAN.md` §14.6 and §14.9, with the number '
        'that settles it. Generated by `scripts/run_paper_analysis.py '
        '--only decisions`; do not edit by hand.',
        '',
    ]
    if context:
        lines += ['| | |', '|---|---|']
        lines += [f'| {k} | {v} |' for k, v in context.items()]
        lines.append('')
    lines += ['## Summary', '', '| | Question | Fired | Says |',
              '|---|---|---|---|']
    for v in verdicts:
        says = v['says'].replace('\n', ' ').replace('|', '/')
        short = says if len(says) < 150 else says[:147] + '...'
        lines.append(f"| **{v['id']}** | {v['question']} | "
                     f"{'yes' if v['fired'] else 'no'} | {short} |")
    lines.append('')
    for v in verdicts:
        lines += [f"## {v['id']} — {v['question']}", '', v['says'], '']
        extra = {k: val for k, val in v.items()
                 if k not in ('id', 'question', 'fired', 'says') and val != ''}
        if extra:
            lines += ['```', *(f'{k} = {val}' for k, val in extra.items()),
                      '```', '']
    path = output_dir / 'DECISIONS.md'
    path.write_text('\n'.join(lines))
    return path
