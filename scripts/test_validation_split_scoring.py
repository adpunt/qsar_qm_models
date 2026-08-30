#!/usr/bin/env python
"""Executable checks for the validation route through the uncertainty writer.

WHAT THIS IS FOR
----------------
A molecule can answer "does the predicted uncertainty find the corrupted
labels?" when two things hold: no model fitted it, and the injector recorded
the noise it received. Out-of-fold scoring buys the first at the price of
`--oof-folds` extra fits per model -- five of every six model fits in the
uncertainty runs were that. Since 2026-08-27 a validation molecule has both for
free: no model stacks validation into its training set, and validation carries
its own independently drawn noise, recorded per molecule in the same provenance
file the training rows come from (RERUN_PLAN.md 13 chat O).

Everything here RUNS the code. Nothing greps the source for a matching line --
this project has already shipped a smoke test that did, and it hid a live bug
for two days. The one check that looks at a function rather than calling it
reads the COMPILED code object's names, so a call commented out or renamed
fails it while a call inside an untaken branch still passes.

Run it:

    python scripts/test_validation_split_scoring.py

Exits non-zero on any failure.
"""
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, 'models'))

from utils import save_uncertainty_values, VALID_SPLITS, CORRUPTED_SPLITS  # noqa: E402

FAILURES = []


def check(label, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ''))
    if not ok:
        FAILURES.append(label)


def raises(label, exc_type, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc_type as e:
        check(label, True, f"{type(e).__name__}: {str(e)[:50]}")
    except Exception as e:  # noqa: BLE001
        check(label, False, f"raised {type(e).__name__}, wanted {exc_type.__name__}")
    else:
        check(label, False, "did not raise")


def unc_path(p):
    return p.replace('.csv', '_uncertainty_values.csv')


class Args:
    """The three attributes the scorer reads off the parsed arguments."""

    def __init__(self, filepath, uncertainty=True, score_validation=True):
        self.filepath = filepath
        self.uncertainty = uncertainty
        self.score_validation = score_validation


def write_provenance(tmp, n_train=40, n_val=10, n_test=10, level=1.5):
    """A provenance CSV and manifest in the injector's own format, so the REAL
    TrainingNoiseRecord parses them rather than a stand-in built for the test."""
    rng = np.random.default_rng(20260828)
    rows = []
    smiles = {}
    for split, n in (('train', n_train), ('val', n_val), ('test', n_test)):
        names = [f'{split[0].upper()}{"C" * (i + 1)}O' for i in range(n)]
        smiles[split] = names
        clean = rng.normal(6.9, 1.3, n)
        pattern = np.full(n, 1.0)
        eps = (rng.normal(0.0, level * 1.3, n) if split != 'test'
               else np.zeros(n))
        scale = (np.full(n, level * 1.3) if split != 'test' else np.zeros(n))
        for i in range(n):
            rows.append({
                'split': split, 'record_index': i, 'canonical_smiles': names[i],
                'y_clean_raw': clean[i], 'epsilon_raw': eps[i],
                'noise_scale_raw': scale[i], 'noise_pattern_raw': pattern[i],
                'y_noisy_raw': clean[i] + eps[i],
                'y_written': (clean[i] + eps[i] - 6.9) / 1.3,
            })
    prov = os.path.join(tmp, 'prov.csv')
    pd.DataFrame(rows).to_csv(prov, index=False)
    man = os.path.join(tmp, 'man.json')
    with open(man, 'w') as f:
        json.dump({'standardisation_mean': 6.9, 'standardisation_sd': 1.3,
                   'noise_targeting': 'uniform', 'noise_type': 'gaussian',
                   'parameters': {}}, f)
    groups = {sm: i % 7 for part in smiles.values() for i, sm in enumerate(part)}
    return prov, man, groups, smiles


def main():
    from process_and_train import TrainingNoiseRecord
    import models as M

    print(__doc__.strip().splitlines()[0])
    print()

    tmp = tempfile.mkdtemp(prefix='valsplit_')

    # ---- the writer accepts the split, and demands the recorded draw --------
    print("the writer")
    check("'validation' is a split the writer accepts",
          'validation' in VALID_SPLITS)
    check("'validation' is a split whose labels were corrupted",
          'validation' in CORRUPTED_SPLITS and 'test' not in CORRUPTED_SPLITS)

    n = 12
    eps = np.linspace(-1.5, 1.5, n)
    y_true = np.linspace(5.0, 8.0, n)
    y_pred = y_true + 0.1
    y_std = np.linspace(0.2, 0.9, n)
    f_val = os.path.join(tmp, 'w.csv')
    save_uncertainty_values(
        y_pred_mean=y_pred, y_pred_std=y_std,
        y_true_original=y_true, y_true_noisy=y_true + eps,
        filepath=f_val, model_name='qrf', rep='pdv',
        sigma_noise=1.5, iteration=0, file_no=1,
        split='validation', injected_noise=eps,
        canonical_smiles=[f'C{"C" * i}O' for i in range(n)],
        noise_scale=np.full(n, 1.95), noise_pattern=np.ones(n),
        noise_pattern_pred=np.ones(n),
        aleatoric_var=np.linspace(0.04, 0.25, n),
        epistemic_var=np.linspace(0.01, 0.09, n))
    df = pd.read_csv(unc_path(f_val), float_precision='round_trip')
    check("a validation row reaches the CSV under split='validation'",
          len(df) == n and bool((df['split'] == 'validation').all()))
    check("its injected_noise is the recorded draw, to the last bit",
          df['injected_noise'].to_numpy(dtype=np.float64).tobytes() == eps.tobytes())
    check("oof_folds_ok is -1: a validation row was not cross-fitted",
          bool((df['oof_folds_ok'].to_numpy() == -1).all()))

    raises("a validation row with no recorded noise raises", ValueError,
           save_uncertainty_values,
           y_pred_mean=y_pred, y_pred_std=y_std,
           y_true_original=y_true, y_true_noisy=y_true,
           filepath=os.path.join(tmp, 'bad.csv'), model_name='qrf', rep='pdv',
           sigma_noise=1.5, iteration=0, file_no=1, split='validation')

    # ---- the scorer --------------------------------------------------------
    print("\nthe scorer")
    prov, man, groups, smiles = write_provenance(tmp)
    rec = TrainingNoiseRecord.load(prov, man, groups, level=1.5)
    n_val = len(rec.splits['val']['epsilon_raw'])

    def predict(x):
        # Deterministic, and both components vary per molecule -- which is what
        # SUPPORT says the quantile forest produces, so the writer's own guard
        # has something real to check rather than being stepped around.
        m = np.asarray(x, dtype=float).ravel()
        k = len(m)
        alea = 0.04 + 0.01 * np.arange(k)
        epis = 0.01 + 0.002 * np.arange(k)
        return m, np.sqrt(alea + epis), alea, epis

    f_s = os.path.join(tmp, 's.csv')
    x_val = np.arange(n_val, dtype=float)
    got = M.score_validation_molecules(
        predict, x_val, rec, Args(f_s), 1.5, 'pdv', 0, 1, 'qrf')
    dfs = pd.read_csv(unc_path(f_s), float_precision='round_trip')
    check("the scorer writes one row per validation molecule",
          got == n_val and len(dfs) == n_val, f"{got} rows")
    check("it writes the injector's recorded validation draw, not a reconstruction",
          np.allclose(dfs['injected_noise'].to_numpy(),
                      rec.splits['val']['epsilon_raw']))
    check("it writes the validation molecules, in the provenance's order",
          list(dfs['canonical_smiles']) == list(rec.splits['val']['canonical_smiles']))
    check("the clean label, the draw and the corrupted label are on one scale",
          np.allclose((dfs['y_true_original'] + dfs['injected_noise'] - 6.9) / 1.3,
                      dfs['y_true_noisy'], atol=1e-9))
    check("the condition travels onto the row",
          bool((dfs['noise_type'] == 'gaussian').all()))

    # ---- the refusals ------------------------------------------------------
    print("\nthe refusals")
    f_off = os.path.join(tmp, 'off.csv')
    off = M.score_validation_molecules(
        predict, x_val, rec, Args(f_off, score_validation=False),
        1.5, 'pdv', 0, 1, 'qrf')
    check("without --score-validation it writes nothing",
          off is None and not os.path.exists(unc_path(f_off)))

    f_nou = os.path.join(tmp, 'nou.csv')
    nou = M.score_validation_molecules(
        predict, x_val, rec, Args(f_nou, uncertainty=False),
        1.5, 'pdv', 0, 1, 'qrf')
    check("with uncertainty off it writes nothing",
          nou is None and not os.path.exists(unc_path(f_nou)))

    f_mis = os.path.join(tmp, 'mis.csv')
    raises("scoring a different number of rows than the provenance covers raises",
           RuntimeError, M.score_validation_molecules,
           predict, np.arange(n_val + 3, dtype=float), rec, Args(f_mis),
           1.5, 'pdv', 0, 1, 'qrf')
    check("...and it wrote no file while refusing",
          not os.path.exists(unc_path(f_mis)))

    # ---- the wiring --------------------------------------------------------
    # Read off the COMPILED code object, so a call that is deleted, renamed or
    # commented out fails this while a call in an untaken branch still passes.
    print("\nthe wiring: every trainer that cross-fits also scores validation")
    trainers = [
        ('train_rf_model', 'the two forests'),
        ('train_ngboost_model', 'NGBoost'),
        ('train_gauche_model', 'the Gaussian process'),
        ('train_dnn_model', 'the DNN family'),
        ('train_mlp_variant_model', 'the MLP family'),
        ('train_heteroscedastic_gp', 'the noise-predicting GP'),
    ]
    for name, label in trainers:
        fn = getattr(M, name, None)
        if fn is None:
            check(f"{label} ({name}) exists", False)
            continue
        names = set(fn.__code__.co_names)
        for const in fn.__code__.co_consts:
            if hasattr(const, 'co_names'):
                names |= set(const.co_names)
        cross = 'score_training_molecules_out_of_fold' in names
        val = 'score_validation_molecules' in names
        check(f"{label} scores validation as well as cross-fitting",
              cross and val, f"cross-fit={cross} validation={val}")

    # ---- the QM9 job scripts ask for ONE route, and it is the clean one -----
    # Two things are checked here and they are different claims.
    #
    # (1) A model that emits an uncertainty must ask for SOMETHING. Until
    #     2026-08-28 the QM9 scripts asked for neither route, and a QM9 test
    #     label is never corrupted, so the whole grid answered nothing about
    #     whether uncertainty finds the corrupted labels.
    #
    # (2) Every such model must ask for the SAME thing. Mixing the routes across
    #     models is what this gate exists to stop: the forests would then be
    #     scored on 500 validation molecules by a model fitted on 80% of the
    #     data and the networks on 4,000 training molecules by inner models
    #     fitted on ~53%, so a difference between two models would confound the
    #     model with the route. That is the confound closed on 2026-08-27.
    #
    # The route is cross-fitting, settled by the author 2026-08-28 on the ground
    # that the four neural families and NGBoost early-stop on validation, so
    # scoring them there is optimistic and the comparison has to be leak-free
    # everywhere rather than in places.
    print("\nthe QM9 job scripts ask for one route, the same one, for every uncertainty model")
    import subprocess
    gen = os.path.join(REPO, 'slurm_scripts_qm9_rerun', 'generate_scripts.py')
    if not os.path.exists(gen):
        check("the QM9 job generator exists", False, gen)
    else:
        # BOTH passes, with the arguments the runbook actually submits.
        #
        # This used to invoke the generator bare. The generator's defaults are
        # the main grid at a 47-hour ceiling, and it REFUSES that combination on
        # purpose -- ngboost's tier needs 202 hours, and silently capping the
        # request is what gets a job killed after days in the queue. So this
        # gate went red on a correct refusal, reporting "the QM9 job generator
        # runs" as the failure and saying nothing about the route it exists to
        # check. A gate that cries wolf is worse than no gate.
        for label, extra in (('the screen', ['--stage', '0']),
                             ('the main grid',
                              ['--stage', '1', '--max-hours', '720'])):
            with tempfile.TemporaryDirectory() as gd:
                r = subprocess.run([sys.executable, gen, '--out-dir', gd] + extra,
                                   capture_output=True, text=True)
                if r.returncode != 0:
                    check(f"the QM9 job generator runs ({label})", False,
                          r.stderr[-200:] or r.stdout[-200:])
                    continue
                bodies = {f: open(os.path.join(gd, f)).read()
                          for f in sorted(os.listdir(gd)) if f.endswith('.sh')}
                unc = {f: b for f, b in bodies.items() if '-u True' in b}
                plain = {f: b for f, b in bodies.items() if '-u True' not in b}

                # THREE STATES, NOT TWO. This check used to demand that every
                # script passing `-u True` also ask for a route, and that was
                # true when it was written on 2026-08-28. On 2026-08-29 the
                # author settled which model-and-representation pairs get the
                # out-of-fold pass (uncertainty_pairs.json), and `-u True` was
                # deliberately left unrestricted -- it is free and it writes the
                # test rows either way. So a script can now be in one of three
                # states, and only the third is a defect:
                #
                #   cross-fits           on the settled pairs, --oof-folds
                #   test rows only       -u True, no route: legitimate, but its
                #                        rows carry injected_noise = 0 by
                #                        construction (scripts/utils.py
                #                        CORRUPTED_SPLITS), so they answer
                #                        nothing about corruption
                #   scores validation    --score-validation: the leaky route
                #
                # Demanding two states made this gate fail on the settled design
                # rather than on a defect. What is still worth asserting is
                # sharper: the set that cross-fits is EXACTLY the settled pairs.
                routed = {f: b for f, b in unc.items() if '--oof-folds' in b}
                leaky = {f: b for f, b in unc.items() if '--score-validation' in b}
                unrouted = {f for f in unc if f not in routed and f not in leaky}

                check(f"{label}: nothing takes the leaky route (--score-validation)",
                      not leaky, f"{sorted(leaky)}")
                check(f"{label}: everything routed takes the SAME route, cross-fitting",
                      bool(routed) and not leaky,
                      f"{len(routed)} cross-fit, {len(leaky)} validation")

                pairs = json.load(open(os.path.join(REPO, 'uncertainty_pairs.json')))
                settled = {m['qm9'] for m in pairs['models']}
                got = {f[len('qm9_s%d_' % int(extra[1])):-len('.sh')] for f in routed}
                check(f"{label}: the models that cross-fit are exactly the settled pairs",
                      got == settled,
                      f"extra {sorted(got - settled)}, missing {sorted(settled - got)}")

                check(f"{label}: a model with no per-molecule uncertainty asks for neither",
                      all('--oof-folds' not in b and '--score-validation' not in b
                          for b in plain.values()),
                      f"{len(plain)} scripts")

                # NOT a failure -- a consequence of the settled design, printed
                # so it is on the record rather than discovered at analysis time.
                # A test row's injected noise is exactly zero, so a model here
                # contributes nothing to "does uncertainty find the corrupted
                # labels", however good its uncertainty is.
                if unrouted:
                    print(f"       note ({label}): {len(unrouted)} script(s) emit an "
                          f"uncertainty on TEST ROWS ONLY, whose injected noise is "
                          f"zero by construction: "
                          f"{', '.join(sorted(f[:-3] for f in unrouted))}")

        # And the refusal itself is a property worth holding: the bare defaults
        # must NOT produce scripts, because they cannot be honoured.
        with tempfile.TemporaryDirectory() as gd:
            r = subprocess.run([sys.executable, gen, '--out-dir', gd],
                               capture_output=True, text=True)
            wrote = [f for f in os.listdir(gd) if f.endswith('.sh')]
            check("the generator refuses a wall clock it cannot honour",
                  r.returncode != 0,
                  f"rc={r.returncode}, {len(wrote)} script(s) written")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("all checks passed")
    return 0


if __name__ == '__main__':
    sys.exit(main())
