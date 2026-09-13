#!/usr/bin/env python3
"""Fail if the tuning scripts and the job generators disagree about which
representations exist.

    python scripts/test_rep_lists_agree.py

WHY THIS EXISTS
---------------
Between 2026-09-01 and 2026-09-12, scripts/write_chosen_settings.py ranked
settings over five representations and scripts/ship_tuned_settings.py wrote the
winner out to six. Avalon was the sixth. So Avalon was handed a setting picked
by a contest it never entered, and on Caco-2 the setting it was handed is one
that loses 1.885 R-squared on Avalon itself.

Nothing failed. Both scripts ran, both wrote valid files, and the two lists were
forty lines apart in different files. This check closes that: the lists are
compared, not trusted.

WHAT MUST MATCH
---------------
1. The ranking list in write_chosen_settings.py, the shipping list in
   ship_tuned_settings.py, and ALL_REPS in the QM9 job generator, as sets.
2. ALL_REPS in the laboratory job generator, which uses display names, once
   mapped through REP_NAME_MAP in models/tuning_rosters.py.
3. REP_NAME_MAP itself, which must cover every representation and invent none.
4. Every entry in both shipped files, which must carry all six. The laboratory
   reader RAISES on a dataset-and-model entry that is missing a representation
   (alternative_data_noise_robustness.py, tuned_neural_params), so shipping five
   would stop every Avalon task rather than quietly untune it.

WHAT IS DELIBERATELY NOT COMPARED
---------------------------------
The uncertainty generator's REPS is ECFP4, PDV and ChemBERTa. That is a chosen
subset of three, not a drift, and it is not checked here.

The QM9 generator's FP_REPS is ECFP4 alone, because the Tanimoto kernel needs
binary vectors. Also not a drift.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, 'models'))

MASTER = os.path.join(_ROOT, 'results', 'master_tuned_hyperparameters.json')
LAB = os.path.join(_ROOT, 'results', 'master_tuned_hyperparameters_lab.json')

# THE SIX, WRITTEN OUT. Checking the lists only against each other would pass
# if a representation were removed from all of them at once, which is exactly
# how mol2vec-shaped deletions happen. CLAUDE.md line 167 names these six:
# "The representations are PDV, MHG-GNN, Avalon, ECFP4, ChemBERTa and Sort &
# Slice." Short names as the QM9 generator spells them.
THE_SIX = {'pdv', 'mhggnn', 'avalon', 'ecfp4', 'chemberta', 'sns'}

_fails = []


def ok(msg):
    print(f'  ok    {msg}')


def fail(msg):
    print(f'  FAIL  {msg}')
    _fails.append(msg)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    os.environ.setdefault('QSAR_QM_MODELS_ROOT', _ROOT)

    chooser = _load('write_chosen_settings',
                    os.path.join(_HERE, 'write_chosen_settings.py'))
    shipper = _load('ship_tuned_settings',
                    os.path.join(_HERE, 'ship_tuned_settings.py'))
    qm9_gen = _load('qm9_job_generator',
                    os.path.join(_ROOT, 'slurm_scripts_qm9_rerun',
                                 'generate_scripts.py'))
    lab_gen = _load('lab_job_generator',
                    os.path.join(_ROOT, 'slurm_scripts_validation_rerun',
                                 'generate_scripts.py'))
    import tuning_rosters as R

    truth = set(qm9_gen.ALL_REPS)
    print(f'the QM9 job generator submits {len(truth)}: '
          f'{", ".join(sorted(truth))}')

    print('\nagainst the six CLAUDE.md names')
    if truth == THE_SIX:
        ok('the QM9 job generator submits the six the study measures')
    else:
        fail(f'the QM9 job generator does not submit the six CLAUDE.md names: '
             f'missing {sorted(THE_SIX - truth) or "none"}, '
             f'extra {sorted(truth - THE_SIX) or "none"}. Representation is a '
             f'measured factor, so dropping one is an author decision, not a '
             f'code edit.')

    # A THIRD COPY OF THE LIST. scripts/final_tuned_list.py is an older, second
    # chooser for the same question, with its own tiebreak. Its FINAL_LIST.json
    # and .md are not on disk and nothing but RERUN_PLAN.md mentions them, so it
    # is not live -- but it holds its own list of six and would drift silently.
    lister = _load('final_tuned_list',
                   os.path.join(_HERE, 'final_tuned_list.py'))

    print('\nthe tuning scripts')
    for label, got in (('write_chosen_settings.py ranks over', set(chooser.REPS)),
                       ('ship_tuned_settings.py writes out to', set(shipper.REPS)),
                       ('final_tuned_list.py ranks over', set(lister.REPS))):
        if got == truth:
            ok(f'{label} the same {len(got)}')
        else:
            fail(f'{label} {len(got)}: '
                 f'missing {sorted(truth - got) or "none"}, '
                 f'extra {sorted(got - truth) or "none"}')
    if len(set(chooser.REPS)) != len(chooser.REPS):
        fail('write_chosen_settings.py REPS repeats a representation')
    if len(set(shipper.REPS)) != len(shipper.REPS):
        fail('ship_tuned_settings.py REPS repeats a representation')

    print('\nthe laboratory job generator, through REP_NAME_MAP')
    display = set(lab_gen.ALL_REPS)
    if set(R.REP_NAME_MAP) != truth:
        fail(f'REP_NAME_MAP keys are not the {len(truth)} the QM9 generator '
             f'submits: missing {sorted(truth - set(R.REP_NAME_MAP))}, '
             f'extra {sorted(set(R.REP_NAME_MAP) - truth)}')
    mapped = {R.REP_NAME_MAP[r] for r in truth & set(R.REP_NAME_MAP)}
    if mapped == display:
        ok(f'{len(display)} display names, and they are the same {len(truth)}')
    else:
        fail(f'the laboratory generator submits {sorted(display)}, the QM9 one '
             f'maps to {sorted(mapped)}')

    print('\nthe two shipped files')
    for path in (MASTER, LAB):
        name = os.path.basename(path)
        if not os.path.exists(path):
            print(f'  skip  {name} does not exist yet — run '
                  f'scripts/ship_tuned_settings.py --write')
            continue
        with open(path) as fh:
            blob = json.load(fh)
        # The QM9 file is model -> rep -> setting. The laboratory file is
        # dataset -> model -> rep -> setting. Walk to the depth that holds
        # representation names either way.
        entries = []
        for k, v in sorted(blob.items()):
            if all(isinstance(x, dict) and set(x) & truth for x in v.values()):
                entries += [(f'{k}/{m}', by_rep) for m, by_rep in sorted(v.items())]
            else:
                entries.append((k, v))
        bad = [(who, sorted(truth - set(by_rep)), sorted(set(by_rep) - truth))
               for who, by_rep in entries if set(by_rep) != truth]
        if bad:
            for who, missing, extra in bad:
                fail(f'{name}: {who} is missing {missing or "none"} and '
                     f'carries {extra or "none"} that nothing submits — the '
                     f'laboratory reader raises on a missing one, it does not '
                     f'fall back')
        else:
            ok(f'{name}: {len(entries)} entries, each carrying all '
               f'{len(truth)}')

    print()
    if _fails:
        print(f'{len(_fails)} check(s) FAILED')
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
