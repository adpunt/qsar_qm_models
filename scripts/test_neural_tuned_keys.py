#!/usr/bin/env python3
"""The two base networks must be tuned over the same four numbers.

WHY THIS EXISTS
---------------
`train_mlp_variant_model` read hidden_size, num_hidden_layers, dropout_rate and
lr out of its tuned entry. `train_dnn_model` read width, depth and activation
only: its learning rate came straight from the shared spec and its dropout was
the literal 0.2 inside `DNNRegressionModel`. So a sweep could move four numbers
on one base network and two on the other, and a tuned entry carrying `lr` or
`dropout_rate` for a dnn key would have been written and silently never applied
-- the run would then not be at the setting the file claims.

The four Bayesian and variational transformations of each base network read the
same dict, so the asymmetry reached eight models, not two.

This checks the delivery path, not the arithmetic of a fit: a chosen learning
rate has to arrive at the optimiser and a chosen dropout has to arrive at every
dropout layer, on BOTH builders, and the spec's value has to arrive when the
tuned entry carries neither.

It also checks that the two variance-head networks and the two heteroscedastic
ones ASK for their own tuned key. No sweep has scored them yet, so they train at
their defaults today; when one does, the entry it writes has to be picked up
without a further code change.

And it checks the WRITING side, which is the half that matters: a builder that
can read a learning rate is worth nothing if no sweep can search one.
`SEARCH_SPACES['dnn']` in scripts/tune_hyperparameters.py searched three numbers
against `SEARCH_SPACES['mlp']`'s four, and `search_family` did not strip the
suffix '_bnn_full_mve', so the two variance-head networks resolved to no search
space at all and every sweep recorded them 'blocked'.

    python scripts/test_neural_tuned_keys.py
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'models'))
sys.path.insert(0, os.path.join(REPO, 'scripts'))

N, DIM = 80, 12
N_TRAIN = 48

# Values nothing in the spec would produce, so an arrival is not a coincidence.
TUNED_LR = 0.0031415
TUNED_DROPOUT = 0.4127


class Args:
    pass


def base_args(scratch, loss='mse', bayesian=None):
    a = Args()
    a.use_best_params = True
    a.tuning = False
    a.loss = loss
    a.loss_params = None
    a.bayesian_transformation = bayesian
    a.k_domains = 1
    a.uncertainty = False
    a.filepath = scratch
    a.sample_size = N
    a.epochs = 2
    a.save_per_epoch_metrics = False
    return a


def data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(N, DIM)).astype(np.float32)
    y = (x[:, 0] * 2.0 - x[:, 1]).astype(np.float64)
    return (x[:N_TRAIN], y[:N_TRAIN],
            x[N_TRAIN:], y[N_TRAIN:],
            x[N_TRAIN:], y[N_TRAIN:])


class Capture:
    """Records what the builder actually handed the optimiser and the layers."""

    def __init__(self, M, torch, nn, tuned):
        self.M, self.torch, self.nn, self.tuned = M, torch, nn, tuned
        self.lrs = []
        self.dropouts = []
        self.asked = []

    def __enter__(self):
        M, torch, nn = self.M, self.torch, self.nn
        self._adam = torch.optim.Adam
        self._reader = M.load_best_hyperparameters
        self._dropout = nn.Dropout
        outer = self

        def adam(params, lr=1e-3, **kw):
            outer.lrs.append(float(lr))
            return outer._adam(params, lr=lr, **kw)

        def reader(model_type, rep, results_dir='results'):
            outer.asked.append(model_type)
            return dict(outer.tuned) if outer.tuned is not None else None

        # A SUBCLASS, not a function. The builders are written in the old style,
        # `super(DNNRegressionModel, self)`, which looks the name up in the
        # module at call time -- so replacing the class with a factory makes
        # super() raise, and the check would fail on its own instrumentation.
        class RecordingDropout(outer._dropout):
            def __init__(self, p=0.5, inplace=False):
                outer.dropouts.append(float(p))
                super().__init__(p=p, inplace=inplace)

        torch.optim.Adam = adam
        M.load_best_hyperparameters = reader
        nn.Dropout = RecordingDropout
        return self

    def __exit__(self, *exc):
        self.torch.optim.Adam = self._adam
        self.M.load_best_hyperparameters = self._reader
        self.nn.Dropout = self._dropout
        return False


def fit(M, torch, nn, which, tuned, scratch, loss='mse', bayesian=None):
    x_tr, y_tr, x_te, y_te, x_val, y_val = data()
    args = base_args(scratch, loss=loss, bayesian=bayesian)
    with Capture(M, torch, nn, tuned) as cap:
        if which == 'dnn':
            M.train_dnn_model(x_tr, y_tr, x_te, y_te, x_val, y_val,
                              args, 0.0, 'pdv', 0, 0, 0, y_te)
        else:
            M.train_mlp_variant_model(x_tr, y_tr, x_te, y_te, x_val, y_val,
                                      'mlp', args, 0.0, 'pdv', 0, 0, 0, y_te)
    return cap


def main():
    scratch = os.path.join(os.environ.get('TMPDIR', '/tmp'),
                           'test_neural_tuned_keys.csv')
    if os.path.exists(scratch):
        os.remove(scratch)

    import torch
    import torch.nn as nn
    import models as M
    from model_defaults import NEURAL_DEFAULTS
    from tuning_rosters import TUNED_KEY, roster_label

    failures = []
    spec_lr = NEURAL_DEFAULTS['training']['lr']

    tuned = {'hidden_size1': 64, 'hidden_size2': 32, 'activation': 'relu',
             'hidden_size': 64, 'num_hidden_layers': 1,
             'lr': TUNED_LR, 'dropout_rate': TUNED_DROPOUT}

    print(f"a tuned entry carrying lr={TUNED_LR} and dropout_rate={TUNED_DROPOUT}")
    for which in ('dnn', 'mlp'):
        cap = fit(M, torch, nn, which, tuned, scratch)
        lr_ok = cap.lrs and all(np.isclose(v, TUNED_LR) for v in cap.lrs)
        drop_ok = cap.dropouts and all(np.isclose(v, TUNED_DROPOUT)
                                       for v in cap.dropouts)
        print(f"  {which:4s} optimiser saw {sorted(set(cap.lrs))}, "
              f"dropout layers saw {sorted(set(cap.dropouts))}")
        if not lr_ok:
            failures.append(
                f"{which}: the tuned learning rate did not reach the optimiser "
                f"(saw {sorted(set(cap.lrs))}, wanted {TUNED_LR}). The run would not "
                f"be at the setting the tuned file claims.")
        if not drop_ok:
            failures.append(
                f"{which}: the tuned dropout did not reach the layers "
                f"(saw {sorted(set(cap.dropouts))}, wanted {TUNED_DROPOUT}).")

    print("\nno tuned entry at all -- the shared spec is the fallback")
    for which, spec_drop in (('dnn', NEURAL_DEFAULTS['dnn']['dropout_rate']),
                             ('mlp', NEURAL_DEFAULTS['mlp']['dropout_rate'])):
        cap = fit(M, torch, nn, which, None, scratch)
        print(f"  {which:4s} optimiser saw {sorted(set(cap.lrs))}, "
              f"dropout layers saw {sorted(set(cap.dropouts))}")
        if not (cap.lrs and all(np.isclose(v, spec_lr) for v in cap.lrs)):
            failures.append(
                f"{which}: with no tuned entry the learning rate is "
                f"{sorted(set(cap.lrs))}, not the spec's {spec_lr}.")
        if not (cap.dropouts and all(np.isclose(v, spec_drop)
                                     for v in cap.dropouts)):
            failures.append(
                f"{which}: with no tuned entry the dropout is "
                f"{sorted(set(cap.dropouts))}, not the spec's {spec_drop}.")

    print("\na tuned entry that carries the width and nothing else")
    partial = {'hidden_size1': 64, 'hidden_size2': 32, 'activation': 'relu',
               'hidden_size': 64, 'num_hidden_layers': 1}
    for which in ('dnn', 'mlp'):
        try:
            cap = fit(M, torch, nn, which, partial, scratch)
        except Exception as exc:
            failures.append(
                f"{which}: a tuned entry without lr or dropout_rate raised "
                f"{type(exc).__name__}: {exc}. It has to fall back to the spec.")
            continue
        print(f"  {which:4s} optimiser saw {sorted(set(cap.lrs))}")
        if not (cap.lrs and all(np.isclose(v, spec_lr) for v in cap.lrs)):
            failures.append(
                f"{which}: a partial tuned entry did not fall back to the spec's "
                f"learning rate (saw {sorted(set(cap.lrs))}).")

    # The four models a sweep has not scored yet have to ASK for their own key,
    # or its output cannot be picked up without another code change.
    print("\nthe models no sweep has scored yet ask for their own key")
    wanted = {
        ('dnn', 'full', 'heteroscedastic', False): 'dnn_bnn_full_mve',
        ('mlp', 'full', 'heteroscedastic', False): 'mlp_bnn_full_mve',
        ('dnn', 'full_variational', 'mse', True): 'dnn_bnn_full_variational_hetero',
        ('mlp', 'full_variational', 'mse', True): 'mlp_bnn_full_variational_hetero',
    }
    for (base, bt, loss, hetero), label in wanted.items():
        a = base_args(scratch, loss=loss, bayesian=bt)
        a.heteroscedastic_vbll = hetero
        got = roster_label(base, a)
        print(f"  {base} + {bt} + loss={loss}"
              f"{' + noise head' if hetero else ''}  ->  {got}")
        if got != label:
            failures.append(
                f"{base} with {bt}/{loss} resolves to '{got}', not '{label}', so a "
                f"tuned entry written under '{label}' would never be read.")
        if TUNED_KEY.get(label) != label:
            failures.append(
                f"TUNED_KEY['{label}'] is {TUNED_KEY.get(label)!r}; it has to be its "
                f"own key or the entry reaches another model.")

    # THE OTHER HALF OF THE SAME DEFECT. The builders reading the same four keys
    # is worth nothing if no sweep can write them. SEARCH_SPACES['dnn'] held
    # three things and SEARCH_SPACES['mlp'] four, which is why every mlp entry in
    # results/master_tuned_hyperparameters.json carries an lr and a dropout_rate
    # and no dnn entry carries either.
    print("\nthe sweep searches the same two numbers on both base networks")
    import tune_hyperparameters as TH
    for key in ('dnn', 'mlp'):
        space = sorted(TH.SEARCH_SPACES[key])
        print(f"  SEARCH_SPACES[{key!r}] searches {space}")
        for name in ('lr', 'dropout_rate'):
            if name not in TH.SEARCH_SPACES[key]:
                failures.append(
                    f"SEARCH_SPACES[{key!r}] does not search {name!r}, so no sweep "
                    f"can ever write it and the builder's fallback is the only "
                    f"value that model will ever take.")

    # And the two variance-head networks have to resolve to a search space at
    # all. '_bnn_full_mve' was missing from the suffix list in `search_family`,
    # so both returned None and the sweep recorded them 'blocked' -- the reason
    # no sweep has ever scored them.
    print("\nevery model in the sweep resolves to a search space")
    for label, base in (('dnn_bnn_full_mve', 'dnn'),
                        ('mlp_bnn_full_mve', 'mlp'),
                        ('dnn_bnn_full_variational_hetero', 'dnn'),
                        ('mlp_bnn_full_variational_hetero', 'mlp')):
        import tuning_rosters as _rosters
        got = TH.search_family(label, _rosters)
        print(f"  {label:34s} -> {got}")
        if got != base:
            failures.append(
                f"search_family({label!r}) is {got!r}, not {base!r}. The sweep "
                f"records it 'blocked' and never fits it.")

    # The laboratory runner carries the same two networks and the same tuned
    # file, and its TUNED_KEYS_APPLIED promises dropout_rate is applied. Its DNN
    # branch hard-coded 0.2 the same way QM9's did, so the guard that exists to
    # stop a setting arriving silently would have passed one straight through.
    print("\nthe laboratory runner's two networks")
    kirby = os.environ.get('KIRBY_DIR') or os.path.join(
        os.path.dirname(REPO), 'KIRBy')
    runner_dir = os.path.join(kirby, 'tests')
    if not os.path.isdir(runner_dir):
        failures.append(
            f"no KIRBy checkout at {kirby}, so the laboratory half of this check "
            f"did not run. Set KIRBY_DIR.")
    else:
        sys.path.insert(0, runner_dir)
        import alternative_data_noise_robustness as runner
        for cls, kwargs in ((runner.DeterministicRegressor,
                             dict(input_dim=DIM, dropout_rate=TUNED_DROPOUT)),
                            (runner.MLPRegressor,
                             dict(input_dim=DIM, dropout_rate=TUNED_DROPOUT))):
            built = cls(**kwargs)
            ps = sorted({float(m.p) for m in built.modules()
                         if isinstance(m, nn.Dropout)})
            print(f"  {cls.__name__:24s} dropout layers saw {ps}")
            if not ps or not all(np.isclose(v, TUNED_DROPOUT) for v in ps):
                failures.append(
                    f"{cls.__name__} does not apply dropout_rate (saw {ps}), but "
                    f"TUNED_KEYS_APPLIED says the key is applied.")
        if 'dropout_rate' not in runner.TUNED_KEYS_APPLIED:
            failures.append(
                "TUNED_KEYS_APPLIED no longer lists dropout_rate, so a tuned entry "
                "carrying it would be refused rather than applied.")

    if failures:
        print(f"\nFAIL -- {len(failures)} problem(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nOK: both base networks read the same four keys from the same place, "
          "and the four untuned models ask for their own")
    return 0


if __name__ == '__main__':
    sys.exit(main())
