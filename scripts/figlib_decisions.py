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
# D2 -- how many noise types does F3 show
# ---------------------------------------------------------------------------

#: Two grids "agree" when they order the cells the same way AND differ by less
#: than the run-to-run wobble. Both, because either alone is satisfiable by an
#: uninteresting accident: a perfect rank match on tiny differences, or tiny
#: differences with the order scrambled.
GRID_AGREEMENT_RHO = 0.90


def d2_grid_similarity(summary, per_replicate=None):
    """Which noise conditions give the same model-by-representation grid.

    RERUN_PLAN.md 14.6 row 14. The rule is fixed in advance and the answer comes
    off the data: show the conditions whose grids DIFFER; a condition whose grid
    repeats another's goes to an additional file. If every grid differs, every
    grid is a main-text panel and F3 becomes the largest figure in the paper.
    """
    if summary is None or not len(summary):
        return {}, _verdict('D2', 'How many noise types does F3 show?', False,
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
            if len(shared) < 4:
                continue
            x, y = ga.loc[shared].astype(float), gb.loc[shared].astype(float)
            rho, p = stats.spearmanr(x, y)
            mad = float(np.abs(x - y).mean())
            same = bool(rho >= GRID_AGREEMENT_RHO
                        and np.isfinite(wobble) and mad <= wobble)
            rows.append({'a': a, 'b': b, 'rho': float(rho), 'p_value': float(p),
                         'mean_abs_difference': mad,
                         'replicate_wobble': wobble,
                         'grids_agree': same, 'n_cells': int(len(shared))})
    table = pd.DataFrame(rows)

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

    says = (f'F3 shows {len(main_text)} panel(s): '
            + ', '.join(C.condition_label(c) for c in main_text) + '. '
            + (f'Held back as repeats of one of those: '
               f'{", ".join(C.condition_label(c) for c in supplementary)}.'
               if supplementary else
               'Every grid differs from every other, so every one is a '
               'main-text panel and F3 is the largest figure in the paper.'))
    return ({'d2_grid_similarity': table},
            _verdict('D2', 'How many noise types does F3 show, and which?',
                     fired=bool(supplementary), says=says,
                     main_text=main_text, supplementary=supplementary,
                     replicate_wobble=wobble))


# ---------------------------------------------------------------------------
# D3 -- do the noise types separate the models
# ---------------------------------------------------------------------------

def d3_condition_separation(per_replicate, summary, rep):
    """Paired signed-rank across models, condition against condition.

    RERUN_PLAN.md 14.6 rows 7 and 8, and 14.9 item 7 -- the test that exists
    nowhere and costs no compute. Paired on the MODEL, so n is the roster size
    and the test can reach significance; the assay datasets have five folds and
    a two-sided signed-rank test on five pairs cannot go below p = 0.0625.
    """
    if summary is None or not len(summary):
        return {}, _verdict('D3', 'Do the noise types separate the models?',
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
    n_sig = int(paired['significant'].sum()) if len(paired) else 0
    n_exceed = int(spread['exceeds_wobble'].sum()) if len(spread) else 0
    fired = n_sig > 0 or n_exceed > len(spread) / 2
    says = (f'{n_sig} of {len(paired)} condition pairs differ significantly on '
            f'{C.rep_label(rep)}; {n_exceed} of {len(spread)} models vary more '
            f'across conditions than across replicates. '
            f"Kendall's W = {kendall['kendall_w']:.3f} "
            f"(p = {kendall['p_value']:.2g}, {kendall['n_models']} models, "
            f"{kendall['n_conditions']} conditions, {C.rep_label(rep)} only). "
            + ('The kind of noise matters, so F4 gets its one-line-per-condition '
               'panel (row 7).' if fired else
               'Only the AMOUNT matters, not the kind -- which is a headline '
               'finding in its own right (row 8).'))
    return ({'d3_condition_pairs': paired, 'd3_condition_spread': spread,
             'd3_kendall_w': pd.DataFrame([kendall])},
            _verdict('D3', 'Do the noise types separate the models?',
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
