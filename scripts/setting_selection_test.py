"""Which noise settings earn cluster time? (RERUN_PLAN.md chat G)

NOISE_DESIGN.md section 2 lists Student-t at three tail weights (nu = 10, 5, 3) and
Outlier at three contamination fractions (1%, 5%, 10%). Running every setting at full
grid puts nine conditions on the noise axis instead of six -- the largest single
multiplier on the design. This script decides whether the extra settings buy anything.

The question, per setting: at a MATCHED delivered amount of noise, is it distinguishable
from Gaussian in model accuracy, and by how much relative to the replicate-to-replicate
spread? A difference smaller than the run-to-run wobble is not a setting worth running.

Also tested, because RERUN_PLAN.md section 13.3 settled the conditions but nothing has
ever measured them: grouped-shifted (whole scaffold families pushed in one direction)
against grouped-wider (families scattered more widely), and a genuinely skewed draw
against a symmetric one, all at matched amount.

WHAT THIS CHANGES FROM THE EARLIER PILOTS (scripts/pilot_noise_arms.py,
scripts/noise_range_finding.py). Both fixed the molecule subsample, the scaffold split
and the model seeds across replicates, so their only source of variation was the noise
draw. The real pipeline re-derives all of it per replicate: iteration_seed at
scripts/process_and_train.py:1871 seeds a torch.randperm shuffle in split_qm9, so the
sampled molecules and the scaffold split move too. Since the decision rule compares an
effect against the replicate spread, that spread has to be the one the real run will
have. Here every replicate draws a fresh subsample and a fresh split.

Real task: QM9 HOMO-LUMO gap from RDKit physicochemical descriptors (PDV).
Real protocol: scaffold split, noise on TRAIN labels only, scored on CLEAN test labels.

Usage:
    python3 scripts/setting_selection_test.py --self-check   # assertions only, no fitting
    python3 scripts/setting_selection_test.py                # the full run
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
import lightgbm as lgb

RDLogger.DisableLog('rdApp.*')

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SDF = os.path.join(REPO, 'data/QM9/raw/gdb9.sdf')
CSV = os.path.join(REPO, 'data/QM9/raw/gdb9.sdf.csv')
OUT_ROWS = os.path.join(REPO, 'results/setting_selection_test.csv')
OUT_SHAPE = os.path.join(REPO, 'results/setting_selection_shape_diagnostics.csv')
CACHE = os.environ.get(
    'SETTING_SELECTION_CACHE',
    '/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/'
    'f7379a39-dde4-4cd6-b692-5d60f0daebba/scratchpad/qm9_pool.npz')

POOL_N = 20000        # molecules featurised once
SUBSAMPLE_N = 4000    # molecules per replicate, as in the range-finding run
N_REPLICATES = 12
# Two levels, not three. The reporting level, and the top of the QM9 grid -- section
# 5.5 of NOISE_DESIGN.md found the noise types only begin to separate above 1.0, so a
# test that omits the top of the grid cannot find an effect even if one exists, while
# 1.0 sits between two points that already bracket the answer. Dropping it buys a third
# of the runtime on a machine this test has to share.
LEVELS = [0.5, 1.5]        # fraction of the training label spread
REPORTING_LEVEL = 0.5      # RERUN_PLAN.md section 13.1 open item 5 -- the standing suggestion
POOL_SEED = 11
# The roster the cluster runs, minus the ones that cannot be fitted here. Ridge is GONE:
# it is not one of the thirteen models in slurm_scripts_qm9_rerun/generate_scripts.py and
# no conclusion may rest on it. The quantile forest cannot be built in this environment
# (RERUN_PLAN.md 3.4.4d, reconfirmed 2026-08-27: scikit-learn 1.3.2 rejects
# monotonic_cst). The Gaussian process is left out on purpose -- it segfaults in a
# process that has already loaded the boosting libraries (RERUN_PLAN.md 2.8e). The four
# Bayesian neural variants are variants of DNN and MLP and add nothing to a screen.
ALL_SCREEN_MODELS = ['LGBM', 'XGBoost', 'RF', 'SVM', 'NGBoost', 'DNN', 'MLP']
MODELS = list(ALL_SCREEN_MODELS)
# Features standardised before fitting. Trees are invariant to it; the kernel and the
# neural models are not. models/model_defaults.should_standardise() makes the same call
# in the real pipeline.
SCALED_MODELS = {'SVM', 'DNN', 'MLP'}
N_THREADS = int(os.environ.get('SETTING_SELECTION_THREADS', '2'))

# Taken from models/model_defaults.py, the file both training pipelines read, so this
# screen tests the models the cluster will actually run. Copied rather than imported
# because that module pulls in the whole training stack.
RF_DEFAULTS = dict(n_estimators=100, max_depth=None, min_samples_leaf=1,
                   min_samples_split=2, max_features=0.3, bootstrap=True)
LGBM_DEFAULTS = dict(n_estimators=100, learning_rate=0.1, num_leaves=31, max_depth=-1,
                     subsample=1.0, colsample_bytree=1.0, min_child_samples=20,
                     reg_alpha=0.0, reg_lambda=0.0)
XGB_DEFAULTS = dict(n_estimators=100, learning_rate=0.1, max_depth=6, subsample=1.0,
                    colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=1.0)
SVM_DEFAULTS = dict(C=1.0, gamma='scale', kernel='rbf')
NGB_DEFAULTS = dict(n_estimators=500, learning_rate=0.01, natural_gradient=True)
# models/model_defaults.NEURAL_DEFAULTS, copied for the same reason the tree settings are.
NEURAL = dict(
    dnn=dict(hidden_sizes=[128, 64], dropout_rate=0.2),
    mlp=dict(hidden_sizes=[128, 128], dropout_rate=0.2),
    training=dict(lr=0.001, weight_decay=0.0, batch_size=32, epochs=100, patience=10,
                  restore_best_weights=True, standardise_targets=True),
)


class TorchNet:
    """The study's two neural models, built from NEURAL_DEFAULTS.

    Not imported from models/models.py because that module pulls in the whole training
    stack. What matters for a screen is that the architecture, the optimiser, the batch
    size, the early-stopping rule and the target standardisation are the ones the cluster
    will run -- including restore_best_weights, which RERUN_PLAN.md 3.4.4c settled on
    2026-08-26 for both pipelines.

    A tenth of the training split is held out to early-stop against. It carries the
    injected noise like the rest of the training data: nobody gets clean labels when
    deciding when to stop (RERUN_PLAN.md 4, decision 2).
    """

    def __init__(self, kind, seed):
        self.kind, self.seed = kind, seed

    def fit(self, X, y):
        import torch
        import torch.nn as nn
        torch.manual_seed(self.seed)
        torch.set_num_threads(N_THREADS)
        cfg, tr = NEURAL[self.kind], NEURAL['training']
        self.y_mu, self.y_sd = (float(y.mean()), float(y.std())) \
            if tr['standardise_targets'] else (0.0, 1.0)
        yv = (np.asarray(y, dtype=np.float32) - self.y_mu) / (self.y_sd or 1.0)
        Xv = np.asarray(X, dtype=np.float32)

        rng = np.random.default_rng(self.seed)
        val = rng.permutation(len(Xv))[: max(1, len(Xv) // 10)]
        mask = np.ones(len(Xv), bool)
        mask[val] = False
        Xt, yt = torch.from_numpy(Xv[mask]), torch.from_numpy(yv[mask])
        Xva, yva = torch.from_numpy(Xv[~mask]), torch.from_numpy(yv[~mask])

        layers, prev = [], Xv.shape[1]
        for h in cfg['hidden_sizes']:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(cfg['dropout_rate'])]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

        opt = torch.optim.Adam(self.net.parameters(), lr=tr['lr'],
                               weight_decay=tr['weight_decay'])
        lossf = nn.MSELoss()
        best, best_state, bad = float('inf'), None, 0
        bs = tr['batch_size']
        for _ in range(tr['epochs']):
            self.net.train()
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), bs):
                b = perm[i:i + bs]
                opt.zero_grad()
                lossf(self.net(Xt[b]).squeeze(-1), yt[b]).backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                v = float(lossf(self.net(Xva).squeeze(-1), yva))
            if v < best:
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= tr['patience']:
                    break
        if tr['restore_best_weights'] and best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def predict(self, X):
        import torch
        self.net.eval()
        with torch.no_grad():
            p = self.net(torch.from_numpy(np.asarray(X, dtype=np.float32))).squeeze(-1)
        return p.numpy() * (self.y_sd or 1.0) + self.y_mu
# The exact-dose sensitivity pass runs only where the confound it controls for exists:
# the conditions whose per-run delivered dose is wobbliest (NOISE_DESIGN.md 5.1b).
EXACT_DOSE_CONDITIONS = ['Gaussian', 'Student-t nu=5', 'Student-t nu=3', 'Laplace',
                         'Outlier p=0.10', 'Grouped shifted']

# Bentz et al. 2013 Table 7: 62% of measurement variance sits between laboratories.
# This fixes the grouped-shifted split rather than leaving it to judgement.
BETWEEN_GROUP_SHARE = 0.62
GAMMA_SHAPE = 1.0     # skewed draw: skewness = 2/sqrt(a)
LAMBDA = 3.0          # NOISE_DESIGN.md section 2, four independent sources agree
GROUP_FRACTION = 0.2  # affected group fraction, no published number, chosen and declared

# Rule 3 of NOISE_DESIGN.md section 2a: fix the POPULATION dose and gate on the mean
# realised dose over at least 20 seeds, never on one realisation. Student-t nu=3 and
# grouped-shifted cannot meet a per-run tolerance by construction.
N_DOSE_SEEDS = 24
DOSE_CHECK_MIN_OBS = 8    # below this a standard error is not a standard error
DOSE_TOL = 0.02       # on the MEAN over N_DOSE_SEEDS draws


# --------------------------------------------------------------- noise constructions
# These MIRROR the Rust injector as chat A landed it (commit d25bcb0, rust/src/main.rs):
# a SHAPE (how each draw is distributed) crossed with a TARGETING rule (who gets hit),
# and one dose solver over both. Anything that diverges here would be screening a
# scheme the cluster is not going to run.
#
#   scales     the per-molecule scale map          (scale_map,  rust/src/main.rs:382)
#   G          rms(scales) * the shape's unit SD   (unit_dose,  rust/src/main.rs:486)
#   solved     target / G, so delivered rms = target
#   epsilon    raw draw * solved * scale
#
# One shape is NOT in the Rust injector: the skewed draw. It is a chat G proposal and
# is marked as such wherever it appears.

def shape_unit_sd(shape):
    """SD of one raw draw from this shape at scale 1. rust/src/main.rs:127-140."""
    kind = shape[0]
    if kind == 'gaussian':
        return 1.0
    if kind == 'student_t':
        nu = shape[1]
        assert nu > 2, "variance is undefined at nu <= 2, so 'the same amount of noise' stops meaning anything"
        return np.sqrt(nu / (nu - 2.0))
    if kind == 'laplace':
        return np.sqrt(2.0)
    if kind == 'skewed':
        # centred Gamma, a = GAMMA_SHAPE: variance a at scale 1
        return np.sqrt(GAMMA_SHAPE)
    raise ValueError(shape)


def shape_draw(shape, n, rng):
    """One raw draw per molecule. rust/src/main.rs:142-160."""
    kind = shape[0]
    if kind == 'gaussian':
        return rng.standard_normal(n)
    if kind == 'student_t':
        return rng.standard_t(shape[1], n)
    if kind == 'laplace':
        return rng.laplace(0.0, 1.0, n)
    if kind == 'skewed':
        # ❌ TESTED AND REJECTED, 2026-08-27. Lives here and NOWHERE ELSE -- it is not
        # in the Rust injector, it is not in the Python one, and it is not to be built.
        # `noise_conditions.json` records it under `not_run` with `never_implemented`,
        # and a test on each side fails if that changes.
        #
        # It exists so the rejection is a measurement rather than an argument. Errors
        # far more likely in one direction: g ~ Gamma(a, 1) centred on its mean,
        # skewness 2/sqrt(a). RERUN_PLAN.md 13.3 rejected a skewed draw for the three
        # experimental datasets because log-scale potency error is symmetric; this
        # measures what that rejection costs on QM9, which is computed rather than
        # measured and so has no assay to justify any shape. The answer: nothing at the
        # reporting level, and -0.045 R2 for the random forest alone at the top of the
        # grid. The asymmetry story is carried by censoring and by grouped-shifted,
        # which are mechanisms with sources behind them rather than a chosen shape.
        return rng.gamma(GAMMA_SHAPE, 1.0, n) - GAMMA_SHAPE
    raise ValueError(shape)


def shape_draw_unit_variance(shape, n, rng):
    return shape_draw(shape, n, rng) / shape_unit_sd(shape)


def select_groups_by_molecule_fraction(groups, target_fraction, rng):
    """Add scaffold groups until the MOLECULE fraction reaches the target.

    Never by counting groups. Real Murcko groups are wildly uneven -- taking a fifth
    of the GROUPS puts anywhere between 7% and 55% of the MOLECULES in the affected
    set, so the condition's defining parameter would swing eightfold between
    replicates. rust/src/main.rs:408-450, and NOISE_DESIGN.md section 2a rule 1.
    """
    g = np.asarray(groups)
    n = len(g)
    sizes = pd.Series(g).value_counts().to_dict()
    uniq = sorted(sizes)
    rng.shuffle(uniq)
    bad, cumulative = set(), 0
    for gid in uniq:
        size = sizes[gid]
        here = abs(cumulative / n - target_fraction)
        there = abs((cumulative + size) / n - target_fraction)
        if cumulative > 0 and there > here:
            continue
        bad.add(gid)
        cumulative += size
        if cumulative / n >= target_fraction:
            break
    return np.array([x in bad for x in g])


def scale_map(targeting, n, groups, rng):
    """Per-molecule scale map, and the fraction of molecules it actually affects."""
    kind = targeting[0]
    if kind in ('uniform', 'grouped_shift'):
        return np.ones(n), (1.0 if kind == 'grouped_shift' else 0.0)
    if kind == 'grouped_wide':
        lam, group_fraction = targeting[1], targeting[2]
        is_bad = select_groups_by_molecule_fraction(groups, group_fraction, rng)
        return np.where(is_bad, lam, 1.0), float(is_bad.mean())
    if kind == 'outlier':
        p, lam = targeting[1], targeting[2]
        hit = rng.random(n) < p
        return np.where(hit, lam, 1.0), float(hit.mean())
    raise ValueError(targeting)


def build_noise(shape, targeting, n, tau, groups, rng):
    """The whole dose solver. Returns (epsilon, provenance dict).

    Delivered root-mean-square noise equals `tau` by construction, whatever the shape
    and whoever is targeted. This is the fix that makes the noise types comparable at
    all -- NOISE_DESIGN.md section 1.
    """
    scales, affected = scale_map(targeting, n, groups, rng)
    g = float(np.sqrt(np.mean(scales ** 2)) * shape_unit_sd(shape))
    solved = tau / g

    if targeting[0] == 'grouped_shift':
        # Two components: a group-level offset carrying `rho` of the variance and a
        # within-molecule term carrying the rest, so the two sum to the target.
        # Both are unit-variance draws FROM THE SHAPE, so a heavy-tailed shifted
        # condition has heavy tails in the offsets too. rust/src/main.rs:693-716.
        rho = targeting[1]
        gg = np.asarray(groups)
        uniq, inv = np.unique(gg, return_inverse=True)
        offsets = shape_draw_unit_variance(shape, len(uniq), rng)
        within = shape_draw_unit_variance(shape, n, rng)
        eps = solved * (np.sqrt(rho) * offsets[inv] + np.sqrt(1.0 - rho) * within)
    else:
        eps = shape_draw(shape, n, rng) * solved * scales

    return eps, dict(unit_dose_g=g, solved_scale=float(solved),
                     affected_molecule_fraction=affected)


GAUSSIAN = ('gaussian',)
UNIFORM = ('uniform',)

# The settings under question, in the injector's own vocabulary. Each row is one
# condition the cluster would have to run.
CONDITIONS = [
    ('Gaussian',        GAUSSIAN,             UNIFORM,                          'reference'),
    ('Student-t nu=10', ('student_t', 10.0),  UNIFORM,                          'heavy tail'),
    ('Student-t nu=5',  ('student_t', 5.0),   UNIFORM,                          'heavy tail'),
    ('Student-t nu=3',  ('student_t', 3.0),   UNIFORM,                          'heavy tail'),
    ('Laplace',         ('laplace',),         UNIFORM,                          'heavy tail'),
    ('Outlier p=0.01',  GAUSSIAN,             ('outlier', 0.01, LAMBDA),        'contamination'),
    ('Outlier p=0.05',  GAUSSIAN,             ('outlier', 0.05, LAMBDA),        'contamination'),
    ('Outlier p=0.10',  GAUSSIAN,             ('outlier', 0.10, LAMBDA),        'contamination'),
    ('Grouped wider',   GAUSSIAN,             ('grouped_wide', LAMBDA, GROUP_FRACTION), 'structure-keyed'),
    ('Grouped shifted', GAUSSIAN,             ('grouped_shift', BETWEEN_GROUP_SHARE),   'structure-keyed'),
    # Rejected 2026-08-27 -- kept so the rejection stays reproducible, not to be built.
    ('Skewed draw',     ('skewed',),          UNIFORM,                          'asymmetric'),
]
CONDITION_NAMES = [c[0] for c in CONDITIONS]
CONDITION_BY_NAME = {c[0]: c for c in CONDITIONS}

# The ladders, for the consecutive-pair contrasts. The question is whether a ladder
# separates, not only whether its ends differ from Gaussian.
LADDERS = {
    'Student-t': ['Gaussian', 'Student-t nu=10', 'Student-t nu=5', 'Student-t nu=3'],
    'Outlier': ['Gaussian', 'Outlier p=0.01', 'Outlier p=0.05', 'Outlier p=0.10'],
    'Laplace': ['Gaussian', 'Laplace'],
    # Grouped: wider is the symmetric half, shifted the asymmetric one. Both against
    # Gaussian, and shifted against wider -- the two differ ONLY in whether the group's
    # error is centred, so that contrast isolates direction at matched amount.
    'Grouped': ['Gaussian', 'Grouped wider', 'Grouped shifted'],
    'Skew': ['Gaussian', 'Skewed draw'],
}


# --------------------------------------------------------------- data
def load_pool(cache=CACHE):
    """Featurise the pool ONCE. Each replicate then draws its own subsample from it."""
    if cache and os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        X, y, scaffs = d['X'], d['y'], d['scaffs']
        print(f"pool from cache {cache}: {X.shape}, label SD {y.std():.4f} eV", flush=True)
        return X, y, scaffs
    t0 = time.time()
    props = pd.read_csv(CSV)
    gap_by_id = dict(zip(props.mol_id, props.gap * 27.211386))   # Hartree -> eV
    supp = Chem.SDMolSupplier(SDF, removeHs=False, sanitize=True)
    mols, ys, scaffs = [], [], []
    for m in supp:
        if m is None:
            continue
        name = m.GetProp('_Name') if m.HasProp('_Name') else None
        if name not in gap_by_id:
            continue
        try:
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
        except Exception:
            continue
        mols.append(m)
        ys.append(gap_by_id[name])
        scaffs.append(sc)
        if len(mols) >= POOL_N:
            break
    print(f"loaded {len(mols)} molecules in {time.time() - t0:.0f}s", flush=True)

    names = [d[0] for d in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(names)
    X = np.array([calc.CalcDescriptors(m) for m in mols], dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = X[:, X.std(axis=0) > 0]
    y = np.asarray(ys)
    print(f"PDV descriptors: {X.shape}   pool label SD = {y.std():.4f} eV "
          f"({time.time() - t0:.0f}s)", flush=True)
    scaffs = np.asarray(scaffs, dtype=object)
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, X=X, y=y, scaffs=scaffs)
    return X, y, scaffs


def noise_group_ids(scaffs):
    """Group assignment for the two grouped noise conditions.

    Rule 2 of NOISE_DESIGN.md section 2a: the empty Murcko scaffold is NOT a group.
    Acyclic molecules -- 32% of QM9 -- all share the empty scaffold string, so left as
    one group a single offset draw moves a third of the dataset at once and the
    realised dose swings by 11%. Each acyclic molecule becomes its own group.

    The corrected ids are used for BOTH the noise grouping and the scaffold split, at
    every call site, because the real pipeline corrects both. Leaving the split on raw
    scaffolds put the whole acyclic third of the data on one side of it: the clean
    baseline fell to 0.83 for LightGBM and 0.79 for the random forest against roughly
    0.92 for both, which is how the omission was found. (A third model, ridge, went to
    -0.40, but ridge is not in the study and that figure is not evidence for anything.)
    """
    return np.asarray([f"__singleton_{i}" if (sc == '' or sc is None) else sc
                       for i, sc in enumerate(scaffs)], dtype=object)


def group_summary(group_ids):
    sizes = pd.Series(group_ids).value_counts()
    return dict(n_groups=int(len(sizes)),
                largest_group_share=float(sizes.iloc[0] / len(group_ids)))


def scaffold_split(group_ids, n_total, rng, test_frac=0.2):
    """80/20 split by whole scaffold group, on the SAME grouping the noise uses.

    Two things this has to get right, and the earlier pilots got one of them wrong.

    Rule 2 of NOISE_DESIGN.md section 2a applies to the split as much as to the noise:
    a third of QM9 is acyclic and shares the empty Murcko scaffold, so a rule that
    treats it as one group can put that entire chemical class on one side. Measured
    here: it drove the clean R2 to 0.83 / 0.79 / -0.40 against the 0.92 / 0.92 / 0.90
    the range-finding run reports. Splitting on the singleton-corrected grouping keeps
    the largest group under 4% of molecules.

    The group order is shuffled per replicate. The real pipeline's splitter is
    deterministic given its subsample (deepchem's ScaffoldSplitter, largest group
    first, called from split_qm9 at scripts/process_and_train.py:624-631) and gets its
    replicate variation from a fresh torch.randperm subsample instead. Shuffling here
    adds split variation to subsample variation, so the replicate spread this test
    reports is, if anything, WIDER than the real run's -- which is the safe direction
    for a test whose whole purpose is to ask whether an effect clears that spread.
    """
    sizes = pd.Series(group_ids).value_counts().to_dict()
    uniq = sorted(sizes)
    rng.shuffle(uniq)
    test_sc, n_t = set(), 0
    for s in uniq:
        if n_t >= test_frac * n_total:
            break
        test_sc.add(s)
        n_t += sizes[s]
    return np.array([s in test_sc for s in group_ids])


def build_models(seed):
    """The two tree models take the pipeline's OWN shared defaults.

    `models/model_defaults.py` is where both training pipelines now get their forest and
    boosting settings, so a screen that invents its own is screening a different model
    from the one the cluster will run. The forest there is 100 trees at
    `max_features=0.3`; the earlier pilots used 150 trees at scikit-learn's default of
    every feature, which is both unfaithful and, on 178 descriptors, about three times
    the work per split -- this test's single largest cost on a machine it has to share.
    Ridge has no entry there; it is the cheap linear reference the pilots used, kept at
    alpha = 1.0 for continuity with them.

    n_jobs is pinned rather than left at -1. This is a shared machine: at a load average
    of 42 on 8 cores one LightGBM fit with n_jobs=-1 took 410 seconds against the ~2
    seconds it needs unloaded, because every fit spawned eight threads to fight over the
    same cores.
    """
    rf = dict(RF_DEFAULTS)
    lgbm = dict(LGBM_DEFAULTS)
    built = {
        'LGBM': lambda: lgb.LGBMRegressor(random_state=seed, verbose=-1,
                                          n_jobs=N_THREADS, **lgbm),
        'RF': lambda: RandomForestRegressor(random_state=seed, n_jobs=N_THREADS, **rf),
        'XGBoost': lambda: _xgb().XGBRegressor(random_state=seed, n_jobs=N_THREADS,
                                               **XGB_DEFAULTS),
        'SVM': lambda: SVR(**SVM_DEFAULTS),
        'NGBoost': lambda: _ngb()(random_state=seed, verbose=False, **NGB_DEFAULTS),
        'DNN': lambda: TorchNet('dnn', seed),
        'MLP': lambda: TorchNet('mlp', seed),
    }
    return {m: built[m]() for m in MODELS}


def _xgb():
    import xgboost
    return xgboost


def _ngb():
    from ngboost import NGBRegressor
    return NGBRegressor


# --------------------------------------------------------------- shape diagnostics
def shape_diagnostics(noise, tau):
    """NOISE_DESIGN.md section 5.2, recomputed at these settings including p = 1%."""
    a = np.abs(noise)
    order = np.sort(a)[::-1]
    top5 = order[:max(1, int(round(0.05 * len(a))))]
    return dict(
        frac_beyond_3_tau=float((a > 3 * tau).mean()),
        share_on_worst_5pct=float(top5.sum() ** 2 / (a ** 2).sum()) if (a ** 2).sum() > 0 else np.nan,
        share_of_variance_on_worst_5pct=float((top5 ** 2).sum() / (a ** 2).sum()),
        median_abs_error=float(np.median(a)),
        skewness=float(stats.skew(noise)),
        mean_shift=float(noise.mean()),
    )


# --------------------------------------------------------------- self check
def self_check(X, y, scaffs):
    """Dose and construction assertions on the real labels. No model fitting.

    Fails the run rather than warning -- RERUN_PLAN.md section 0.6, guards 1 and 5.
    """
    print("\n=== SELF-CHECK: dose matching and the two new constructions ===", flush=True)
    rng = np.random.default_rng(POOL_SEED)
    idx = rng.choice(len(y), SUBSAMPLE_N, replace=False)
    ys, sc = y[idx], scaffs[idx]
    gids = noise_group_ids(sc)
    is_test = scaffold_split(gids, len(ys), np.random.default_rng(POOL_SEED))
    ytr = ys[~is_test]
    sctr = gids[~is_test]
    sd = ytr.std()
    n = len(ytr)
    gs = group_summary(sctr)
    raw = group_summary(sc[~is_test])
    print(f"train {n} molecules, label SD {sd:.4f} eV", flush=True)
    print(f"noise groups: {gs['n_groups']} (largest holds {gs['largest_group_share']:.1%}); "
          f"raw Murcko would be {raw['n_groups']} (largest {raw['largest_group_share']:.1%}) "
          f"— rule 2 of NOISE_DESIGN.md section 2a", flush=True)

    failures = []
    rows = []
    for level in LEVELS:
        tau = level * sd
        for name, shape, targeting, family in CONDITIONS:
            errs = []
            for rep in range(N_DOSE_SEEDS):
                r = np.random.default_rng(90000 + rep)
                noise, _ = build_noise(shape, targeting, n, tau, sctr, r)
                errs.append(np.sqrt(np.mean(noise ** 2)) / tau - 1.0)
            m = float(np.mean(errs))
            s = float(np.std(errs, ddof=1))
            flag = "" if abs(m) <= DOSE_TOL else "  <-- OUT OF TOLERANCE"
            if abs(m) > DOSE_TOL:
                failures.append(f"{name} at level {level}: realised dose off by {m:+.2%}")
            rows.append(dict(level=level, condition=name, mean_dose_error=m, sd_dose_error=s))
            print(f"  level {level:<4} {name:<18} realised/target − 1 = {m:+.2%} "
                  f"(SD {s:.2%} over {N_DOSE_SEEDS} draws){flag}", flush=True)

    # the two new constructions, checked against what they claim
    tau = REPORTING_LEVEL * sd
    r = np.random.default_rng(4242)
    shifted, _ = build_noise(GAUSSIAN, ('grouped_shift', BETWEEN_GROUP_SHARE), n, tau, sctr, r)
    g = np.asarray(sctr)
    uniq, inv = np.unique(g, return_inverse=True)
    group_means = np.array([shifted[inv == i].mean() for i in range(len(uniq))])
    group_sizes = np.array([(inv == i).sum() for i in range(len(uniq))])
    between = float(np.average(group_means ** 2, weights=group_sizes))
    total = float(np.mean(shifted ** 2))
    print(f"\n  grouped-shifted: between-group share of variance = {between / total:.3f} "
          f"(source says {BETWEEN_GROUP_SHARE}); mean shift {shifted.mean():+.4f} eV", flush=True)
    # NOISE_DESIGN.md section 2a: DO NOT centre the offsets. This condition is not
    # zero-mean in any one run, and that is exactly the mechanism under test -- a
    # chemical family pushed in one direction rather than scattered. What must hold is
    # that it is zero-mean in the POPULATION, over seeds.
    shifts = []
    for rep in range(N_DOSE_SEEDS):
        e, _ = build_noise(GAUSSIAN, ('grouped_shift', BETWEEN_GROUP_SHARE), n, tau,
                           sctr, np.random.default_rng(5100 + rep))
        shifts.append(e.mean())
    shifts = np.asarray(shifts)
    print(f"  grouped-shifted mean label shift: {shifts.mean():+.4f} eV over "
          f"{N_DOSE_SEEDS} seeds, per-run range {shifts.min():+.4f} to {shifts.max():+.4f} "
          f"(tau = {tau:.4f}) — one-directional per run by design, centred over seeds",
          flush=True)
    if abs(shifts.mean()) > 0.25 * tau:
        failures.append(f"grouped-shifted is not zero-mean in the population: "
                        f"{shifts.mean():+.4f} over {N_DOSE_SEEDS} seeds")
    if np.abs(shifts).max() < 0.02 * tau:
        failures.append("grouped-shifted shows no per-run label shift, so the mechanism "
                        "under test is not present")
    # Murcko groups on QM9 are small, so the empirical between-group share sits ABOVE the
    # nominal 0.62 -- a group of one carries its own within-molecule term into its mean.
    # Assert only that the group term dominates, which is the claim being made.
    if not (0.5 < between / total < 0.98):
        failures.append(f"grouped-shifted between-group share {between / total:.3f} is not in (0.5, 0.98)")

    # The Gaussian reference must NOT shift labels -- guard against the comparison being
    # confounded by a direction effect that is meant to be unique to the shifted type.
    gshifts = []
    for rep in range(N_DOSE_SEEDS):
        e, _ = build_noise(GAUSSIAN, UNIFORM, n, tau, sctr, np.random.default_rng(5200 + rep))
        gshifts.append(e.mean())
    print(f"  Gaussian mean label shift for comparison: per-run range "
          f"{min(gshifts):+.4f} to {max(gshifts):+.4f}", flush=True)

    r = np.random.default_rng(4245)
    _, prov = build_noise(GAUSSIAN, ('grouped_wide', LAMBDA, GROUP_FRACTION), n, tau, sctr, r)
    frac = prov['affected_molecule_fraction']
    print(f"  grouped-wider: {frac:.3f} of MOLECULES affected (asked for {GROUP_FRACTION}) — "
          f"selection is by molecule fraction, never by counting groups", flush=True)
    if abs(frac - GROUP_FRACTION) > 0.05:
        failures.append(f"grouped-wider hit {frac:.3f} of molecules, asked for {GROUP_FRACTION}")

    r = np.random.default_rng(4243)
    skew_draw, _ = build_noise(('skewed',), UNIFORM, n, tau, sctr, r)
    sk = float(stats.skew(skew_draw))
    print(f"  skewed draw: sample skewness {sk:+.3f} (target {2 / np.sqrt(GAMMA_SHAPE):+.3f}); "
          f"mean shift {skew_draw.mean():+.4f} eV", flush=True)
    if sk < 1.0:
        failures.append(f"skewed draw is not skewed: sample skewness {sk:.3f}")
    if abs(skew_draw.mean()) > 0.1 * tau:
        failures.append("skewed draw is not centred")

    r = np.random.default_rng(4244)
    gsym, _ = build_noise(GAUSSIAN, UNIFORM, n, tau, sctr, r)
    if abs(float(stats.skew(gsym))) > 0.3:
        failures.append("the Gaussian reference is not symmetric")

    if failures:
        print("\nSELF-CHECK FAILED:", flush=True)
        for f in failures:
            print(f"  - {f}", flush=True)
        sys.exit(1)
    print("\nSELF-CHECK PASSED", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------- the run
def run(X, y, scaffs, rescale_to_exact_dose, conditions=None,
        rows_path=None, shape_path=None):
    """Every replicate is written to disk the moment it finishes.

    This machine is shared and its load average has been over a hundred while this test
    was running. A run that holds everything in memory until the end is a run whose
    results are unreachable if it is interrupted, and this one was interrupted at five
    hours with seven replicates computed and nothing on disk. Appending per replicate
    means the analysis can be run against whatever has landed, and a restart resumes
    from the cache rather than from nothing.
    """
    t0 = time.time()
    rows, shape_rows = [], []

    for rep in range(N_REPLICATES):
        # Mirror what the real pipeline does per replicate: fresh subsample, fresh split,
        # fresh model seeds (scripts/process_and_train.py:1871 -> split_qm9).
        rep_rng = np.random.default_rng(70000 + rep)
        idx = rep_rng.choice(len(y), SUBSAMPLE_N, replace=False)
        Xs, ys, scs = X[idx], y[idx], scaffs[idx]
        gids = noise_group_ids(scs)
        is_test = scaffold_split(gids, len(ys), np.random.default_rng(80000 + rep))
        Xtr, Xte = Xs[~is_test], Xs[is_test]
        ytr, yte = ys[~is_test], ys[is_test]
        sctr = gids[~is_test]
        gsum = group_summary(sctr)
        sd = ytr.std()
        n = len(ytr)

        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

        base = {}
        for mn, mdl in build_models(rep).items():
            Xa, Xb = (Xtr_s, Xte_s) if mn in SCALED_MODELS else (Xtr, Xte)
            mdl.fit(Xa, ytr)
            base[mn] = r2_score(yte, mdl.predict(Xb))
        print(f"[rep {rep}] train {n} test {len(yte)} label SD {sd:.4f}  clean R2 "
              + "  ".join(f"{m}={base[m]:.4f}" for m in MODELS)
              + f"   ({time.time() - t0:.0f}s)", flush=True)

        for level in LEVELS:
            tau = level * sd
            for name, shape, targeting, family in (conditions or CONDITIONS):
                # Common random numbers across conditions within a replicate: the
                # comparison against Gaussian is paired.
                noise_seed = 60000 + rep * 100 + int(level * 10)
                draw_rng = np.random.default_rng(noise_seed)
                noise, prov = build_noise(shape, targeting, n, tau, sctr, draw_rng)
                realised = float(np.sqrt(np.mean(noise ** 2)) / sd)
                if rescale_to_exact_dose:
                    noise = noise * (tau / np.sqrt(np.mean(noise ** 2)))
                    realised_used = level
                else:
                    realised_used = realised
                ynoisy = ytr + noise

                if rep == 0:
                    d = shape_diagnostics(noise, tau)
                    d.update(dict(level=level, condition=name, family=family,
                                  realised_dose=realised, replicate=rep,
                                  unit_dose_g=prov['unit_dose_g'],
                                  affected_molecule_fraction=prov['affected_molecule_fraction'],
                                  dose_mode='exact' if rescale_to_exact_dose else 'algebraic'))
                    shape_rows.append(d)

                for mn, mdl in build_models(rep).items():
                    Xa, Xb = (Xtr_s, Xte_s) if mn in SCALED_MODELS else (Xtr, Xte)
                    mdl.fit(Xa, ynoisy)
                    rows.append(dict(
                        dataset='QM9', representation='PDV', target='homo_lumo_gap',
                        replicate=rep, level=level, condition=name, family=family,
                        model=mn, n_train=n, n_test=len(yte), label_sd=float(sd),
                        target_dose=level, realised_dose=realised,
                        dose_used=realised_used,
                        dose_mode='exact' if rescale_to_exact_dose else 'algebraic',
                        unit_dose_g=prov['unit_dose_g'],
                        affected_molecule_fraction=prov['affected_molecule_fraction'],
                        n_noise_groups=gsum['n_groups'],
                        largest_group_share=gsum['largest_group_share'],
                        noise_seed=noise_seed,
                        r2=r2_score(yte, mdl.predict(Xb)), r2_clean=base[mn],
                    ))
            print(f"  [rep {rep}] level {level} done ({time.time() - t0:.0f}s)", flush=True)

        # Append this replicate before starting the next one.
        if rows_path is not None:
            done = [r for r in rows if r['replicate'] == rep]
            pd.DataFrame(done).to_csv(rows_path, mode='a', index=False,
                                      header=not os.path.exists(rows_path))
        if shape_path is not None and shape_rows:
            pd.DataFrame([r for r in shape_rows if r['replicate'] == rep]).to_csv(
                shape_path, mode='a', index=False, header=not os.path.exists(shape_path))
            shape_rows = [r for r in shape_rows if r['replicate'] != rep]

    return pd.DataFrame(rows), pd.DataFrame(shape_rows)


# --------------------------------------------------------------- analysis
# Guard 8 of RERUN_PLAN.md section 0.6: a filter that is not random with respect to the
# question. Whole replicates below an accuracy floor bias unstable configurations
# upward, and this project's pipelines apply such a filter without declaring it. So:
# declare it, report every verdict WITH and WITHOUT, and say so when the two disagree.
# The floor is on the CLEAN baseline, so it is fixed before any noise is added and
# cannot depend on the condition being judged.
CLEAN_R2_FLOOR = 0.0


def paired_table(df, a, b, level, model):
    """Paired difference a − b within replicate. Returns None if the pairing is broken."""
    sa = df[(df.condition == a) & (df.level == level) & (df.model == model)].set_index('replicate')
    sb = df[(df.condition == b) & (df.level == level) & (df.model == model)].set_index('replicate')
    reps = sorted(set(sa.index) & set(sb.index))
    if len(reps) < 3:
        return None
    d = sa.loc[reps].r2.values - sb.loc[reps].r2.values
    wobble = float(sb.loc[reps].r2.std(ddof=1))
    try:
        p_signed_rank = float(stats.wilcoxon(d).pvalue)
    except ValueError:
        p_signed_rank = np.nan
    # A paired t-test as well, and it is the one the verdict rule uses. The signed-rank
    # test floors at 2/2^n, so at five replicates it CANNOT return p < 0.05 whatever the
    # data -- at five replicates it called a 0.053 R2 difference "not significant"
    # because five points cannot be significant, not because the difference was small.
    # Both are reported; the distribution-free one is the check on the parametric one.
    p_t = float(stats.ttest_rel(sa.loc[reps].r2.values, sb.loc[reps].r2.values).pvalue)
    return dict(
        contrast=f"{a} − {b}", level=level, model=model, n_replicates=len(reps),
        mean_delta_r2=float(d.mean()), sd_delta_r2=float(d.std(ddof=1)),
        reference_mean_r2=float(sb.loc[reps].r2.mean()),
        subject_mean_r2=float(sa.loc[reps].r2.mean()),
        replicate_wobble_sd=wobble,
        ratio_to_wobble=float(abs(d.mean()) / wobble) if wobble > 0 else np.nan,
        wilcoxon_p=p_signed_rank,
        signed_rank_floor=float(2 / 2 ** len(reps)),
        paired_t_p=p_t,
        # What this many replicates could have detected: the smallest true difference
        # the contrast would find at 80% power, two-sided, alpha 0.05. Without it a null
        # cannot be told apart from a test that was never able to see anything -- and
        # the differences at stake here are known to be small.
        min_detectable_delta=float(2.8 * d.std(ddof=1) / np.sqrt(len(reps))),
    )


def build_contrasts(frame):
    """Every ladder contrast: each rung against the reference, and against the previous."""
    rows = []
    for ladder, chain in LADDERS.items():
        for i in range(1, len(chain)):
            for level in sorted(frame.level.unique()):
                for m in MODELS:
                    r = paired_table(frame, chain[i], chain[0], level, m)
                    if r:
                        r['ladder'], r['kind'] = ladder, 'vs reference'
                        rows.append(r)
                    if i > 1:
                        r = paired_table(frame, chain[i], chain[i - 1], level, m)
                        if r:
                            r['ladder'], r['kind'] = ladder, 'vs previous setting'
                            rows.append(r)
    return pd.DataFrame(rows)


def verdict_table(cdf):
    """The recommendation, one row per setting. Recommends; does not decide."""
    out = []
    for name in CONDITION_NAMES:
        if name == 'Gaussian':
            continue
        here = cdf[(cdf.contrast.str.startswith(name + ' ')) & (cdf.kind == 'vs reference')]
        at_report = here[here.level == REPORTING_LEVEL]
        passes = at_report[(at_report.ratio_to_wobble > 1) & (at_report.paired_t_p < 0.05)]
        anywhere = here[(here.ratio_to_wobble > 1) & (here.paired_t_p < 0.05)]
        v = ('EARNS FULL GRID' if len(passes)
             else ('separates only above the reporting level' if len(anywhere)
                   else 'REDUNDANT with its reference'))
        # Guard 8, second half. The accuracy floor drops a (replicate, model) row, so
        # a contrast can rest on fewer replicates than the run has -- and the smallest
        # detectable difference SHRINKS as replicates are lost only if you forget the
        # sqrt(n). It does not; but the row was labelled "what twelve replicates could
        # have detected" while the winning contrast had eleven. The count travels with
        # the number now, so the label cannot drift from the arithmetic again.
        # The effect and the resolution must come from the SAME model, or the pair is
        # two different rows presented as one comparison. Taking the largest effect
        # across models and the smallest threshold across models did exactly that on
        # six of ten conditions, and always in the flattering direction: Student-t
        # ν = 10 read "largest 0.0017, could have seen 0.0064" when the model carrying
        # the 0.0017 could only have seen 0.0119, nearly double. The best-case
        # threshold is still reported, separately and labelled as such, because "the
        # most sensitive model could have seen X" is a real and different claim.
        if len(at_report) and at_report.min_detectable_delta.notna().any():
            lead = at_report.loc[at_report.mean_delta_r2.abs().idxmax()]
            best = at_report.loc[at_report.min_detectable_delta.idxmin()]
            largest = float(abs(lead.mean_delta_r2))
            lead_model, lead_n = str(lead.model), int(lead.n_replicates)
            lead_detectable = float(lead.min_detectable_delta)
            best_detectable, best_model = float(best.min_detectable_delta), str(best.model)
        else:
            largest = lead_detectable = best_detectable = np.nan
            lead_model = best_model = ''
            lead_n = 0
        out.append(dict(
            condition=name, verdict=v,
            largest_abs_delta_at_reporting_level=largest,
            largest_delta_model=lead_model,
            n_replicates_for_that_model=lead_n,
            # what THAT model could have detected — the honest partner to the number above
            detectable_for_that_model=lead_detectable,
            # and the best any model in the run could have done, which is a weaker claim
            best_detectable_any_model=best_detectable,
            best_detectable_model=best_model,
            models_passing_at_reporting_level=len(passes),
            models_passing_at_any_level=len(anywhere)))
    return pd.DataFrame(out)


def models_in(df):
    """The models actually present in the file, not the module's current roster.

    `--analyse-only` reads a CSV that may have been written by an earlier roster. Indexing
    a results frame by the roster constant made the analysis die with KeyError the moment
    the roster changed (it did, on 2026-08-27, when Ridge was removed and five roster
    models were added). Read the file instead. A row for a model that is NOT in the study
    is dropped here and named, so no number in the analysis can rest on one.
    """
    present = [m for m in df.model.unique()]
    outside = [m for m in present if m not in ALL_SCREEN_MODELS]
    if outside:
        print(f"  !! dropping {', '.join(sorted(outside))} — not in the study roster "
              f"(RERUN_PLAN.md 13.10). Every number below excludes them.", flush=True)
    return [m for m in ALL_SCREEN_MODELS if m in present]


def analyse(df, shape):
    print("\n" + "=" * 116)
    print("QM9 / PDV. Noise on training labels only, scored on clean test labels.")
    print(f"{df.replicate.nunique()} replicates, each a fresh subsample and a fresh scaffold split.")
    print("Every condition delivers the SAME amount of noise at a given level.")
    print("=" * 116)

    # -- the declared filter, guard 8 -----------------------------------------
    bad = df[df.r2_clean < CLEAN_R2_FLOOR][['replicate', 'model', 'r2_clean']].drop_duplicates()
    print(f"\n--- declared filter (guard 8): replicate x model cells whose CLEAN baseline R2 "
          f"is below {CLEAN_R2_FLOOR} ---")
    if len(bad):
        for _, r in bad.iterrows():
            print(f"  replicate {int(r.replicate)}, {r.model}: clean R2 = {r.r2_clean:.4f} — a "
                  f"scaffold split this model cannot fit at all, so every contrast in that cell "
                  f"is arithmetic on a broken baseline")
        print(f"  {len(bad)} cell(s) of {df.replicate.nunique() * len(MODELS)} excluded from the "
              f"contrasts. Every verdict is reported with and without them.")
    else:
        print("  none")
    kept = df[df.r2_clean >= CLEAN_R2_FLOOR]

    # -- realised dose, guard 5 -----------------------------------------------
    print("\n--- delivered amount, by condition (guard 5: equal across conditions) ---")
    # Rule 3 of NOISE_DESIGN.md section 2a: judge the MEAN, against its own standard
    # error, never one realisation against a flat tolerance. Student-t nu=3 and
    # grouped-shifted have per-run dose spreads several times the others', and a gate
    # that ignores that fails at random -- which is exactly what the Rust one did.
    for level in sorted(df.level.unique()):
        # One dose per (replicate, condition) -- the three models share a noise draw, so
        # counting model rows would treble the sample size and shrink the standard error
        # by root three for free.
        per_draw = (df[df.level == level]
                    .drop_duplicates(subset=['replicate', 'condition'])
                    [['condition', 'realised_dose']])
        s = per_draw.groupby('condition').realised_dose.agg(['mean', 'std', 'count'])
        s['se'] = s['std'] / np.sqrt(s['count'])
        s['off'] = s['mean'] - level
        s['band'] = np.maximum(3 * s['se'], 0.005 * level)
        n_obs = int(s['count'].max())
        head = (f"  level {level}: realised {s['mean'].min():.4f}–{s['mean'].max():.4f} over "
                f"{n_obs} replicate(s)")
        if n_obs < DOSE_CHECK_MIN_OBS:
            # A standard error from two or three draws is not a standard error. The
            # flat-dose gate proper is the injector's own self-test, which averages 200
            # seeds; this line is informational until enough replicates have landed.
            print(head + f"; too few to judge flatness — the gate is "
                         f"`rust_processor --self-test`, over 200 seeds")
        else:
            bad = s[s['off'].abs() > s['band']]
            print(head + "; " + ("every condition within three standard errors of target"
                                 if not len(bad) else
                                 "OUTSIDE three standard errors: "
                                 + ", ".join(f"{c} ({r['off']:+.4f} vs band {r['band']:.4f})"
                                             for c, r in bad.iterrows())))
        for c in [c for c in s.index if 'nu=3' in c or 'nu=5' in c or 'shifted' in c]:
            print(f"      {c}: per-run dose SD {s.loc[c, 'std']:.4f} — the POPULATION dose is "
                  f"what is fixed, not this (NOISE_DESIGN.md 5.1b, rule 3 of 2a)")

    # -- clean baselines, guard 4 ---------------------------------------------
    # The roster comes from the FILE, and non-roster models are dropped and named.
    MODELS_HERE = models_in(df)
    df = df[df.model.isin(MODELS_HERE)]
    kept = kept[kept.model.isin(MODELS_HERE)]
    print("\n--- clean baseline R2 per replicate (guard 4: the denominator, always shown) ---")
    b = df.groupby('model').r2_clean.agg(['mean', 'std', 'min', 'max'])
    for m in MODELS_HERE:
        print(f"  {m:<6} {b.loc[m, 'mean']:+.4f} +- {b.loc[m, 'std']:.4f}   "
              f"range {b.loc[m, 'min']:+.4f} to {b.loc[m, 'max']:+.4f}")

    # -- accuracy per condition -----------------------------------------------
    print("\n--- R2 per condition, mean +- SD over replicates (never averaged across conditions) ---")
    for level in sorted(kept.level.unique()):
        print(f"\n  level {level} x label spread")
        print("  " + f"{'condition':<20}" + "".join(f"{m:>24}" for m in MODELS_HERE))
        for name in CONDITION_NAMES:
            cells = []
            for m in MODELS_HERE:
                s = kept[(kept.level == level) & (kept.condition == name) & (kept.model == m)].r2
                cells.append(f"{s.mean():.4f}+-{s.std(ddof=1):.4f}" if len(s) else "—")
            print("  " + f"{name:<20}" + "".join(f"{c:>24}" for c in cells))

    # -- the decision statistic -----------------------------------------------
    cdf = build_contrasts(kept)
    cdf_unfiltered = build_contrasts(df)

    print("\n" + "=" * 116)
    print("THE DECISION STATISTIC: paired difference against the replicate-to-replicate wobble.")
    print("A difference smaller than the run-to-run wobble is not a setting worth cluster time.")
    print("'detectable' is the smallest true difference this many replicates could have found.")
    floor = 2 / 2 ** kept.replicate.nunique()
    print(f"'t p' decides; 'rank p' is the distribution-free check on it and cannot fall below "
          f"{floor:.4f} at {kept.replicate.nunique()} replicates.")
    print("=" * 116)
    for kind in ['vs reference', 'vs previous setting']:
        print(f"\n### {kind}")
        sub = cdf[cdf.kind == kind]
        for level in sorted(kept.level.unique()):
            tag = "  (the QM9 reporting level)" if level == REPORTING_LEVEL else ""
            print(f"\n  level {level}{tag}")
            print("  " + f"{'contrast':<42}{'model':<7}{'mean dR2':>11}{'SD':>9}"
                  f"{'wobble':>9}{'|d|/wobble':>12}{'t p':>8}{'rank p':>8}{'detectable':>12}")
            for _, r in sub[sub.level == level].iterrows():
                mark = "  *" if (r.ratio_to_wobble > 1 and r.paired_t_p < 0.05) else ""
                print("  " + f"{r.contrast:<42}{r.model:<7}{r.mean_delta_r2:>+11.4f}"
                      f"{r.sd_delta_r2:>9.4f}{r.replicate_wobble_sd:>9.4f}"
                      f"{r.ratio_to_wobble:>12.2f}{r.paired_t_p:>8.3f}{r.wilcoxon_p:>8.3f}"
                      f"{r.min_detectable_delta:>12.4f}{mark}")

    # -- verdict per setting --------------------------------------------------
    print("\n" + "=" * 116)
    print("VERDICT PER SETTING (proposed rule: earns full grid if |mean dR2| exceeds the")
    print("replicate wobble for at least one model at the reporting level, with paired p < 0.05)")
    print("=" * 116)
    verdicts = verdict_table(cdf)
    other = verdict_table(cdf_unfiltered).set_index('condition')
    for _, r in verdicts.iterrows():
        alt = other.loc[r.condition, 'verdict'] if r.condition in other.index else r.verdict
        flag = "" if alt == r.verdict else f"   <-- WITHOUT the filter: {alt}"
        print(f"  {r.condition:<20} {r.verdict:<40} largest |dR2| at level {REPORTING_LEVEL}: "
              f"{r.largest_abs_delta_at_reporting_level:.4f} on {r.largest_delta_model} "
              f"({r.n_replicates_for_that_model} reps), which could have seen "
              f"{r.detectable_for_that_model:.4f}; best model {r.best_detectable_model} "
              f"could have seen {r.best_detectable_any_model:.4f}{flag}")
    disagree = [r.condition for _, r in verdicts.iterrows()
                if r.condition in other.index and other.loc[r.condition, 'verdict'] != r.verdict]
    print(f"\n  The declared filter changes the verdict for: {', '.join(disagree)}"
          if disagree else "\n  The declared filter changes no verdict. Guard 8 satisfied.")

    # -- label-side shape -----------------------------------------------------
    print("\n--- label-side shape at the reporting level (distinguishable in the LABELS?) ---")
    s = shape[shape.level == REPORTING_LEVEL]
    print("  " + f"{'condition':<20}{'>3x dose':>10}{'var on worst 5%':>18}"
          f"{'median |e|':>12}{'skewness':>10}{'mean shift':>12}")
    for name in CONDITION_NAMES:
        r = s[s.condition == name]
        if not len(r):
            continue
        r = r.iloc[0]
        print("  " + f"{name:<20}{r.frac_beyond_3_tau:>10.2%}"
              f"{r.share_of_variance_on_worst_5pct:>18.1%}{r.median_abs_error:>12.4f}"
              f"{r.skewness:>10.2f}{r.mean_shift:>+12.4f}")

    return cdf, verdicts

def compute_costs(verdicts):
    """Price each option against RERUN_PLAN.md section 13.1."""
    N_MODELS, N_REPS_DIM = 13, 6           # models x representations = 78 combinations
    COMBOS = N_MODELS * N_REPS_DIM
    NONZERO_LEVELS = 6
    STAGE1_REPLICATES = 10
    STAGE2_COMBOS = 4 * 3                  # 4 models x 3 representations
    OLD_DESIGN = 6 * 11 * 10 * COMBOS      # reproduces 51,480 exactly

    per_setting_stage1 = COMBOS * NONZERO_LEVELS * STAGE1_REPLICATES
    per_setting_stage2 = STAGE2_COMBOS * NONZERO_LEVELS * STAGE1_REPLICATES

    print("\n" + "=" * 108)
    print("COMPUTE, priced against RERUN_PLAN.md section 13.1")
    print("=" * 108)
    print(f"  old design (6 types x 11 levels x 10 replicates x {COMBOS} combinations) = "
          f"{OLD_DESIGN:,} training runs")
    print(f"  one extra setting at full grid in STAGE 1 = {per_setting_stage1:,} runs "
          f"({per_setting_stage1 / OLD_DESIGN:.1%} of the old design)")
    print(f"  one extra setting in STAGE 2 only        = {per_setting_stage2:,} runs "
          f"({per_setting_stage2 / OLD_DESIGN:.1%} of the old design)")
    stage1_base = COMBOS * 19 * STAGE1_REPLICATES
    print(f"\n  stage 1 as agreed (3 noise types + clean = 19 level-conditions) = {stage1_base:,}")
    for extra in [0, 1, 2, 4]:
        tot = stage1_base + extra * per_setting_stage1 + 4440
        print(f"    + {extra} extra setting(s) in stage 1: stage 1 = "
              f"{stage1_base + extra * per_setting_stage1:,}, "
              f"staged total = {tot:,} ({tot / OLD_DESIGN:.0%} of the old design)")
    print("\n  NOTE for section 13.1: the stage 0 row is priced at 1,482 runs, which is")
    print(f"  {COMBOS} x 19 x 1 -- the SAME nineteen level-conditions as stage 1. Its description")
    print("  says 'Gaussian and censoring', which is thirteen conditions and 1,014 runs. The")
    print("  pricing is the coherent one, because stage 0 is reused as replicate 0 of stage 1")
    print("  and that reuse only works if it runs the same conditions. The description is wrong.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--self-check', action='store_true',
                    help='dose and construction assertions only, no model fitting')
    ap.add_argument('--analyse-only', action='store_true',
                    help='re-analyse the saved CSVs without re-running')
    ap.add_argument('--replicates', type=int, default=None)
    ap.add_argument('--models', nargs='+', default=None, choices=ALL_SCREEN_MODELS,
                    help='Subset of the screen roster. Default is all of them.')
    ap.add_argument('--levels', nargs='+', type=float, default=None,
                    help='Noise levels to sweep, as a fraction of the training label '
                         'spread. Default is 0.5 and 1.5.')
    ap.add_argument('--conditions', nargs='+', default=None,
                    help='Subset of the noise conditions, by name.')
    ap.add_argument('--out', default=None,
                    help='Write to this CSV instead of the default, so a focused run '
                         'does not overwrite the full screen.')
    args = ap.parse_args()

    global N_REPLICATES, LEVELS, MODELS, OUT_ROWS, OUT_SHAPE
    if args.replicates is not None:
        N_REPLICATES = args.replicates
    if args.models:
        MODELS = list(args.models)
    if args.levels:
        LEVELS = list(args.levels)
    conditions = None
    if args.conditions:
        unknown = [c for c in args.conditions if c not in CONDITION_NAMES]
        assert not unknown, f"unknown conditions {unknown}; known are {CONDITION_NAMES}"
        conditions = [c for c in CONDITIONS if c[0] in args.conditions]
    if args.out:
        OUT_ROWS = args.out
        OUT_SHAPE = args.out.replace('.csv', '_shape.csv')
    # A subset run screens a different grid from the full one, so it must not overwrite
    # the full screen's file and must not be checked against the full grid's row count.
    subset = bool(args.models or args.levels or args.conditions)
    assert not (subset and not args.out), \
        '--models/--levels/--conditions change the grid, so pass --out and keep the ' \
        'full screen\'s results intact.'

    if args.analyse_only:
        df = pd.read_csv(OUT_ROWS)
        shape = pd.read_csv(OUT_SHAPE)
        cdf, verdicts = analyse(df[df.dose_mode == 'algebraic'],
                                shape[shape.dose_mode == 'algebraic'])
        cdf.to_csv(OUT_ROWS.replace('.csv', '_contrasts.csv'), index=False)
        # The SAME declared filter as the main analysis (guard 8). It is keyed on the
        # CLEAN baseline, so it is fixed before any noise is added; leaving it off here
        # while applying it there was making the two blocks disagree about the same
        # model on the same replicate.
        ex = df[(df.dose_mode == 'exact') & (df.r2_clean >= CLEAN_R2_FLOOR)]
        ex = ex[ex.model.isin(ALL_SCREEN_MODELS)]
        MODELS_HERE = [m for m in ALL_SCREEN_MODELS if m in set(ex.model)]
        if len(ex):
            top = ex.level.max()
            print("\n--- sensitivity: the same contrasts with the dose rescaled to exactly "
                  "the target ---")
            print("  Guard 4: the difference is never shown without the two means it is a "
                  "difference of.")
            print("  " + f"{'condition':<20}{'model':<7}{'mean dR2':>11}{'this':>11}"
                  f"{'Gaussian':>11}{'|d|/wobble':>12}{'t p':>8}")
            broken = []
            for name in CONDITION_NAMES:
                if name == 'Gaussian':
                    continue
                for m in MODELS_HERE:
                    r = paired_table(ex, name, 'Gaussian', top, m)
                    if not r:
                        continue
                    print("  " + f"{name:<20}{m:<7}{r['mean_delta_r2']:>+11.4f}"
                          f"{r['subject_mean_r2']:>+11.3f}{r['reference_mean_r2']:>+11.3f}"
                          f"{r['ratio_to_wobble']:>12.2f}{r['paired_t_p']:>8.3f}")
                    if min(r['subject_mean_r2'], r['reference_mean_r2']) < 0:
                        broken.append(f"{name}/{m}")
            if broken:
                print(f"\n  ⚠ Negative mean R2 in: {', '.join(sorted(set(broken)))}. At the top of "
                      f"the level grid the linear model diverges on some replicates, so a "
                      f"difference between two divergences means nothing. Those rows are NOT "
                      f"filtered out — filtering on the outcome is what guard 8 forbids — but they "
                      f"carry no evidence, and the sensitivity check rests on the tree models.")
        compute_costs(verdicts)
        return

    X, y, scaffs = load_pool()
    self_check(X, y, scaffs)
    if args.self_check:
        return

    os.makedirs(os.path.dirname(OUT_ROWS), exist_ok=True)
    for path in (OUT_ROWS, OUT_SHAPE):
        if os.path.exists(path):
            os.remove(path)
    df, shape = run(X, y, scaffs, rescale_to_exact_dose=False, conditions=conditions,
                    rows_path=OUT_ROWS, shape_path=OUT_SHAPE)

    # Guard 9: assert the row count, a condition that produced no rows must fail loudly.
    ran = [c[0] for c in (conditions or CONDITIONS)]
    expected = len(ran) * len(LEVELS) * len(MODELS) * N_REPLICATES
    assert len(df) == expected, f"expected {expected} rows, wrote {len(df)}"
    for name in ran:
        assert (df.condition == name).sum() == len(LEVELS) * len(MODELS) * N_REPLICATES, \
            f"condition {name} is short of rows"

    if subset:
        print(f"\nwrote {OUT_ROWS} ({len(df)} rows). Subset run: the full screen's "
              f"contrast analysis is not applicable, so it is skipped.")
        return

    # Secondary pass at the top level only: rescale each draw to exactly the target RMS,
    # so the accuracy comparison is separated from the nu=3 dose wobble (section 5.1b).
    saved = LEVELS
    LEVELS = [max(saved)]
    df_exact, shape_exact = run(
        X, y, scaffs, rescale_to_exact_dose=True,
        conditions=[c for c in CONDITIONS if c[0] in EXACT_DOSE_CONDITIONS],
        rows_path=OUT_ROWS, shape_path=OUT_SHAPE)
    LEVELS = saved

    df = pd.read_csv(OUT_ROWS)
    shape = pd.read_csv(OUT_SHAPE)
    print(f"\nwrote {OUT_ROWS} ({len(df)} rows) and {OUT_SHAPE}", flush=True)

    primary = df[df.dose_mode == 'algebraic']
    cdf, verdicts = analyse(primary, shape[shape.dose_mode == 'algebraic'])
    cdf.to_csv(OUT_ROWS.replace('.csv', '_contrasts.csv'), index=False)

    print("\n--- sensitivity: the same contrasts with the dose rescaled to exactly the target ---")
    ex = df[df.dose_mode == 'exact']
    if len(ex):
        for name in CONDITION_NAMES:
            if name == 'Gaussian':
                continue
            for m in MODELS_HERE:
                r = paired_table(ex, name, 'Gaussian', max(saved), m)
                if r:
                    print(f"  level {max(saved)} {name:<20}{m:<7}"
                          f"{r['mean_delta_r2']:>+10.4f}  |d|/wobble {r['ratio_to_wobble']:>6.2f}"
                          f"  p={r['wilcoxon_p']:.3f}")

    compute_costs(verdicts)


if __name__ == '__main__':
    main()
