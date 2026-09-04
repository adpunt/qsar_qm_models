#!/usr/bin/env python
"""The selector writes TWO files, and censoring's is not the deep run's.

WHY THIS EXISTS. Until 2026-09-05 `select_deep_run_pairs.py` wrote one file --
`deep_run_pairs.json`, a model list crossed with a representation list -- and the
command it printed pointed censoring at that same file. Six models on three
representations is EIGHTEEN pairs, and censoring is costed at about five
(RERUN_PLAN.md 13.13, `noise_conditions.json`). Nothing caught it because nothing
checked: the QM9 generator's pair-count guard only ran when `--models` was given, and
the laboratory generator's run-time-selection branch was a bare `pass`.

So this test drives the selector over a synthetic screen and asserts the two things
that went wrong: censoring gets its OWN named pairs, and both generators accept the
censoring file and refuse the deep-run one for censoring.

No cluster, no real results -- the frame below is built here.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SELECTOR = ROOT / 'scripts' / 'select_deep_run_pairs.py'
QM9_GEN = ROOT / 'slurm_scripts_qm9_rerun' / 'generate_scripts.py'
LAB_GEN = ROOT / 'slurm_scripts_validation_rerun' / 'generate_scripts.py'

# Enough of the roster to exercise the rule: models that emit a per-molecule
# uncertainty and models that do not, across more than one family.
MODELS = ['rf', 'xgboost', 'svm', 'ngboost', 'qrf', 'gauche_rbf',
          'het_gp_rbf', 'dnn', 'dnn_bnn_full_mve']
REPS = ['ecfp4', 'pdv', 'chemberta']
CONDITIONS = ['gaussian', 'grouped_wider', 'grouped_shifted']
LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]


def synthetic_screen(path: Path):
    """A screen with a deliberate spread of noise tolerance, and no ties.

    r2 falls from a clean baseline at a per-model rate, so auc_norm orders the
    models predictably, and each model peaks on a different representation so the
    "its best representation" step has something to choose.
    """
    rng = np.random.default_rng(0)
    rows = []
    for i, model in enumerate(MODELS):
        decay = 0.10 + 0.05 * i                     # how fast this model loses R2
        for j, rep in enumerate(REPS):
            clean = 0.70 + 0.03 * ((i + j) % 3)     # its best representation varies
            for cond in CONDITIONS:
                for lvl in LEVELS:
                    r2 = clean - decay * lvl + rng.normal(0, 1e-4)
                    rows.append(dict(model=model, rep=rep, noise_type=cond,
                                     sigma=lvl, r2=r2, iteration=0,
                                     rmse=1.0, mae=0.8, dataset='qm9'))
    pd.DataFrame(rows).to_csv(path / 'anova_synthetic.csv', index=False)


def run(cmd, **kw):
    return subprocess.run([sys.executable, *map(str, cmd)],
                          capture_output=True, text=True, **kw)


def main():
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        results = tmp / 'results'
        results.mkdir()
        synthetic_screen(results)

        deep = tmp / 'deep_run_pairs.json'
        proc = run([SELECTOR, '--results-dir', results, '--out', deep])
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr)
            return fail(['the selector exited '
                         f'{proc.returncode} on a synthetic screen'])

        cen = tmp / 'censoring_pairs.json'
        if not cen.exists():
            return fail([f'the selector wrote {deep.name} but no censoring_pairs.json '
                         'beside it -- censoring would be pointed back at the deep '
                         "run's eighteen pairs, which is what this test exists for"])

        spec_deep = json.loads(deep.read_text())
        spec_cen = json.loads(cen.read_text())

        n_deep = len(spec_deep['generator_labels']) * len(spec_deep['representations'])
        pairs = spec_cen['generator_pairs']
        if len(pairs) >= n_deep:
            failures.append(f'censoring names {len(pairs)} pairs and the deep run '
                            f'{n_deep}; censoring is meant to be the smaller subset')
        if not 1 <= len(pairs) <= 10:
            failures.append(f'censoring names {len(pairs)} pairs; the rule is about '
                            f'five and both generators refuse more than ten')
        if len(pairs) != len(spec_cen['validation_pairs']):
            failures.append('generator_pairs and validation_pairs are different '
                            'lengths, so one pipeline would run pairs the other does '
                            'not')
        if spec_cen['uncertainty_models_named'] < spec_cen['uncertainty_models_required']:
            failures.append(
                f"censoring names {spec_cen['uncertainty_models_named']} model(s) that "
                f"emit an uncertainty and the rule asks for "
                f"{spec_cen['uncertainty_models_required']}")
        if len({tuple(p) for p in pairs}) != len(pairs):
            failures.append('censoring names the same pair twice')

        # A hand-edited DEEP RUN file must survive a re-run of the selector too, and
        # for a sharper reason: skipping is one-way, so a rewrite that drops a model
        # drops it for good on every array already in the queue.
        deep.write_text(json.dumps({'generator_labels': ['rf'],
                                    'validation_labels': ['RF'],
                                    'representations': ['ecfp4'],
                                    'mine': True}, indent=2) + '\n')
        run([SELECTOR, '--results-dir', results, '--out', deep])
        if not json.loads(deep.read_text()).get('mine'):
            failures.append('a re-run of the selector overwrote a hand-edited '
                            'deep_run_pairs.json; every queued deep-run task reads '
                            'that file and a dropped model can never be recovered')
        if not (tmp / 'deep_run_pairs.suggested.json').exists():
            failures.append('the selector left the hand-edited deep-run file alone but '
                            'wrote no suggestion beside it, so the reading is lost')
        deep.write_text(json.dumps(spec_deep, indent=2) + '\n')

        # A hand-edited censoring file must survive a re-run of the selector.
        cen.write_text(json.dumps({'generator_pairs': [['rf', 'ecfp4']],
                                   'validation_pairs': [['RF', 'ECFP4']],
                                   'mine': True}, indent=2) + '\n')
        run([SELECTOR, '--results-dir', results, '--out', deep])
        after = json.loads(cen.read_text())
        if not after.get('mine'):
            failures.append('a re-run of the selector overwrote a hand-edited '
                            'censoring_pairs.json -- the whole point of these files '
                            'is that the author edits them')
        if not (tmp / 'censoring_pairs.suggested.json').exists():
            failures.append('the selector left the hand-edited file alone but wrote '
                            'no suggestion beside it, so the newer screen is lost')

        # Both generators must ACCEPT the censoring file and REFUSE the deep-run one.
        cen.write_text(json.dumps(spec_cen, indent=2) + '\n')
        out = tmp / 'gen'
        for label, gen, extra in [
                ('QM9', QM9_GEN, ['--stage', '2', '--conditions', 'censoring',
                                  '--max-hours', '720']),
                ('laboratory', LAB_GEN, ['--conditions', 'censoring'])]:
            good = run([gen, *extra, '--runtime-selection', cen,
                        '--out-dir', out / f'{label}_ok'])
            if good.returncode != 0:
                failures.append(f'{label}: censoring with censoring_pairs.json exited '
                                f'{good.returncode}\n{good.stderr[-600:]}')
            bad = run([gen, *extra, '--runtime-selection', deep,
                       '--out-dir', out / f'{label}_bad'])
            if bad.returncode == 0:
                failures.append(f'{label}: censoring ACCEPTED the deep run\'s '
                                f'{n_deep}-pair file, which is 3.6x its settled cost')

    return fail(failures)


def fail(failures):
    if failures:
        print(f'FAIL — {len(failures)} problem(s):\n')
        for f in failures:
            print(f'  - {f}')
        return 1
    print('PASS — the selector writes censoring its own named pairs, leaves a '
          'hand-edited file alone, and both generators accept that file and refuse '
          "the deep run's.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
