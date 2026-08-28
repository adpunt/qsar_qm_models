"""Do the two halves of a model's uncertainty mean what they are reported to mean?

Three checks, on real QM9, with every model scored OUT OF FOLD on scaffold
groups. `scripts/test_decomposition_controls.py` runs them as gates; this module
is the measurement, and it writes one disaggregated row per (model,
representation, noise condition, level, statistic) so nothing is pooled.

WHY THESE THREE
===============
Every correlation reported for this split before 2026-08-28 came from a single
design: half the molecules got three times the noise of the other half, and the
correlation was taken against the noise SIZE each molecule's region carries.
Two things were never checked, and both of them decide whether the numbers mean
anything (`RERUN_PLAN.md` 5.5e, "Not checked").

  EVEN        Give every molecule the SAME amount of noise. There is nothing to
              find. A model that still scores high is tracking something else
              about the molecule -- its size, its label magnitude -- and its
              positive result under an uneven condition is that same artefact,
              not noise localisation. This is the negative control, and it
              applies to the Gaussian process result as much as to anything else.

  GRADED      Give every molecule a DIFFERENT amount, on a continuous scale.
              With only two noise amounts a correlation of +0.79 means the model
              told two blocks apart; it does not mean the model ranked
              molecules. If the correlation collapses under a graded scale, that
              is the real answer and it gets reported as such.

  ZERO        No noise at all. Uncertainty may already track the pattern for
              reasons that have nothing to do with corruption -- a molecule's
              region may be intrinsically harder. The honest effect is
              (correlation under the condition) minus (correlation at zero), so
              every correlation here is reported beside its zero-noise value and
              never alone.

WHAT THE CORRELATION IS AGAINST
===============================
`noise_scale`: the standard deviation of the noise the injector applied to that
molecule's label. On the out-of-fold rows this is the corruption the molecule
ACTUALLY received, which is the quantity the question is about. Held-out labels
are never corrupted, so a test row could only ever be scored against the amount
its REGION carries -- which is why every model here is scored out of fold.

Spearman, not Pearson: the claim is "the model ranks the corrupted molecules
above the clean ones", not that it recovers the magnitude. The Gaussian process
compresses threefold true spreads into a 1.4-fold reported difference
(`RERUN_PLAN.md` 5.5e), so a magnitude claim would fail where a ranking claim
holds.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'models'))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The label is the HOMO-LUMO gap. `homo` and `lumo` are NOT features: the gap is
# their difference, so a model given either would be reading the answer. The
# four total energies (u0, u298, h298, g298) are excluded for the same reason --
# they are dominated by molecule size and carry the gap's scale with them.
LABEL = 'gap'
FEATURES = ['A', 'B', 'C', 'mu', 'alpha', 'r2', 'zpve', 'cv', 'u0_atom']

# The three designs, plus the zero-noise control every one of them is reported
# beside.
CONDITIONS = ('even', 'two_block', 'graded')


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _smiles_cache_path():
    scratch = os.environ.get('DECOMP_SCRATCH')
    if scratch:
        return os.path.join(scratch, 'qm9_smiles.txt')
    return os.path.join(REPO, '.decomposition_controls_qm9_smiles.txt')


def qm9_smiles(n_needed):
    """Canonical SMILES for the first `n_needed` records of gdb9.sdf, cached.

    Read from the structure file rather than reconstructed, because the scaffold
    grouping has to be the real one: a random grouping puts close analogues of
    every held-out molecule in the fit set, so the out-of-fold uncertainty comes
    from an interpolation regime while the study's own splits are extrapolation.
    """
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')

    cache = _smiles_cache_path()
    if os.path.exists(cache):
        with open(cache) as fh:
            cached = [line.rstrip('\n') for line in fh]
        if len(cached) >= n_needed:
            return cached[:n_needed]

    supplier = Chem.ForwardSDMolSupplier(
        os.path.join(REPO, 'data', 'QM9', 'raw', 'gdb9.sdf'), removeHs=False)
    out = []
    for mol in supplier:
        out.append('' if mol is None else Chem.MolToSmiles(mol))
        if len(out) >= n_needed:
            break
    try:
        with open(cache, 'w') as fh:
            fh.write('\n'.join(out))
    except OSError:
        pass
    return out


def scaffold_groups(smiles_list):
    """Murcko scaffold group ids, with an ACYCLIC molecule in a group of its own.

    Exactly the rule both pipelines use (`assign_scaffold_groups` in the
    laboratory runner). Without it every acyclic molecule collapses into one
    group that GroupKFold must place entirely in one fold, and on QM9 that is a
    large fraction of the dataset.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog('rdApp.*')

    labels = []
    for i, smi in enumerate(smiles_list):
        core = ''
        if smi:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                try:
                    core = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
                except Exception:
                    core = ''
        labels.append(core if core else f'__acyclic__{i}')
    order = {s: i for i, s in enumerate(sorted(set(labels)))}
    return np.array([order[s] for s in labels])


def load_qm9(n_molecules, seed=0):
    """(X, y, groups) for `n_molecules` real QM9 molecules.

    Sampled across the whole file rather than taken from the front: the records
    are ordered by size, so the first few thousand are the smallest molecules in
    the set and almost all acyclic.
    """
    import pandas as pd

    csv = os.path.join(REPO, 'data', 'QM9', 'raw', 'gdb9.sdf.csv')
    frame = pd.read_csv(csv)
    smiles = qm9_smiles(len(frame))
    frame = frame.iloc[:len(smiles)].copy()
    frame['__smiles'] = smiles
    frame = frame[frame['__smiles'] != '']

    rng = np.random.default_rng(seed)
    take = rng.choice(len(frame), size=min(n_molecules, len(frame)),
                      replace=False)
    take.sort()
    frame = frame.iloc[take]

    x = frame[FEATURES].to_numpy(dtype=float)
    y = frame[LABEL].to_numpy(dtype=float)
    groups = scaffold_groups(list(frame['__smiles']))
    return x, y, groups


# ---------------------------------------------------------------------------
# The noise designs
# ---------------------------------------------------------------------------

def noise_scales(condition, x, level, spread, seed):
    """The per-molecule noise standard deviation for one design.

    Every design is dose-matched: the mean of the squared scales equals
    (level x spread)^2, so the three deliver the SAME total amount of noise and a
    difference between them is a difference in WHERE the noise sits and nowhere
    else. Without that, a design that simply added more noise would look like a
    design a model localises better.
    """
    n = len(x)
    target = float(level) * float(spread)
    if target <= 0:
        return np.zeros(n)

    if condition == 'even':
        # Nothing to find. Every molecule the same.
        shape = np.ones(n)
    elif condition == 'two_block':
        # The design every earlier number came from: molecules above the median
        # of one feature get three times the spread of the rest. A model can
        # score well here by telling two blocks apart.
        cut = np.median(x[:, 0])
        shape = np.where(x[:, 0] > cut, 3.0, 1.0)
    elif condition == 'graded':
        # A continuous ladder, drawn from the molecule's own position rather
        # than at random, so it is reproducible and is a property of the
        # molecule. Spans the same 1x-to-3x range as two_block, so the two
        # differ in GRADATION and not in how much noise is on offer.
        rank = np.argsort(np.argsort(x[:, 0])) / max(n - 1, 1)
        shape = 1.0 + 2.0 * rank
    else:
        raise ValueError(f"unknown condition {condition!r}")

    # Dose-match on the SECOND moment, because that is what adds.
    shape = shape / np.sqrt(float(np.mean(shape ** 2)))
    return shape * target


def reference_patterns(condition, x, spread, seed):
    """What a model's data-noise term is correlated AGAINST, for one design.

    Taken at a fixed reference level of 1.0, so it is the SAME column at every
    noise level including zero -- which is what makes the zero-noise
    subtraction meaningful.

    An uneven design is scored against its own shape. The EVEN design has no
    shape of its own -- it is constant, and a rank correlation against a
    constant is undefined, not zero -- so it is scored against the patterns
    from the uneven designs. That IS the control: does a model that scores well
    under two_block or graded still score against those same patterns when the
    noise carries no such structure? If it does, the positive result there was
    never noise localisation.
    """
    if condition == 'even':
        return {c: noise_scales(c, x, 1.0, spread, seed)
                for c in ('two_block', 'graded')}
    return {condition: noise_scales(condition, x, 1.0, spread, seed)}


def apply_noise(y, scales, seed):
    rng = np.random.default_rng(seed)
    return y + rng.normal(0.0, 1.0, size=len(y)) * scales


# ---------------------------------------------------------------------------
# THE PROJECT'S OWN NOISE CONDITIONS, and its own representations
# ---------------------------------------------------------------------------
# The three designs above are written for the three checks: they are the
# cleanest way to ask "is there anything to find, and can the model rank it?".
# They are NOT the study's noise conditions, and a split that survives them has
# not been shown to survive the noise the study actually injects.
#
# Everything below uses the real thing on both axes: the conditions come from
# the installed injector (`noiseInject`), and the features from the laboratory
# runner's own representation builders. Neither is reimplemented here.
#
# WHICH CONDITIONS CAN ANSWER THIS AT ALL. Only some conditions give different
# molecules different amounts of noise. For the rest every molecule gets the
# same amount, so "which molecules were corrupted" is UNDEFINED rather than
# zero, exactly as the even design above is. Those conditions are still worth
# running, as the study's own version of the negative control.
KIRBY_ROOT = os.environ.get(
    'KIRBY_ROOT', os.path.join(os.path.dirname(REPO), 'KIRBy'))


def _kirby_runner():
    """The laboratory runner, imported for its representation builders and its
    scaffold-group rule. Imported, not copied: a second implementation of a
    representation is a second thing to drift."""
    import importlib.util
    path = os.path.join(KIRBY_ROOT, 'tests',
                        'alternative_data_noise_robustness.py')
    if not os.path.isfile(path):
        raise DecompositionControlsError(
            f"cannot find the laboratory runner at {path}. Set KIRBY_ROOT.")
    os.environ.setdefault('QSAR_QM_MODELS_ROOT', REPO)
    spec = importlib.util.spec_from_file_location('kirby_runner', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class DecompositionControlsError(RuntimeError):
    pass


def build_representations(smiles, names):
    """Real representations, from the laboratory runner's own builders."""
    runner = _kirby_runner()
    return runner.generate_representations(list(smiles), rep_filter=list(names))


def injector_patterns(condition, y, groups, spread):
    """The per-molecule noise scale a real condition delivers, at a fixed
    reference level, and whether it varies at all.

    Taken at level 1.0 so it is the same column at every level including zero,
    which is what makes the zero-noise subtraction mean something.
    """
    from noiseInject import CONDITIONS as INJ_CONDITIONS
    from noiseInject import NoiseInjectorRegression
    if condition not in INJ_CONDITIONS:
        raise DecompositionControlsError(
            f"{condition!r} is not a condition the installed injector knows. "
            f"Known: {sorted(INJ_CONDITIONS)}")
    inj = NoiseInjectorRegression.from_condition(condition, random_state=0)
    # Censoring's level is the FRACTION of labels clipped, which is already
    # dimensionless. Multiplying it by a label spread gives a fraction outside
    # [0, 1] and the quantile raises. Same exemption the runners make.
    ref = 1.0 if INJ_CONDITIONS[condition]['strategy'] == 'censoring' else spread
    scale = inj.noise_scale(y, ref, reference=y, groups=groups,
                            reference_groups=groups)
    scale = np.asarray(scale, dtype=float)
    varies = bool(len(np.unique(np.round(scale, 12))) > 1)
    return scale, varies


def injector_noise(condition, y, level, groups, spread, seed):
    """Corrupt the labels with a real condition at a real level."""
    from noiseInject import CONDITIONS as INJ_CONDITIONS
    from noiseInject import NoiseInjectorRegression
    if float(level) == 0.0:
        return np.asarray(y, dtype=float).copy()
    inj = NoiseInjectorRegression.from_condition(condition, random_state=seed)
    ref = 1.0 if INJ_CONDITIONS[condition]['strategy'] == 'censoring' else spread
    return np.asarray(
        inj.inject_verbose(y, float(level) * ref, groups=groups).y_noisy,
        dtype=float)


# ---------------------------------------------------------------------------
# The models, each scored out of fold on scaffold groups
# ---------------------------------------------------------------------------

def _oof(fit_predict, x, y, groups, n_folds):
    """Out-of-fold mean and split. Mirrors both pipelines' own out-of-fold pass."""
    from sklearn.model_selection import GroupKFold

    n = len(y)
    mean = np.full(n, np.nan)
    alea = np.full(n, np.nan)
    epis = np.full(n, np.nan)
    fold = np.full(n, -1, dtype=int)
    got_a = got_e = False

    splitter = GroupKFold(n_splits=n_folds)
    for i, (keep, held) in enumerate(splitter.split(x, y, groups)):
        m, a, e = fit_predict(x[keep], y[keep], x[held])
        mean[held] = np.asarray(m, dtype=float).ravel()
        fold[held] = i
        if a is not None:
            alea[held] = np.asarray(a, dtype=float).ravel()
            got_a = True
        if e is not None:
            epis[held] = np.asarray(e, dtype=float).ravel()
            got_e = True
    return (mean, alea if got_a else None, epis if got_e else None, fold)


def _standardise(x_fit, x_score):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(x_fit)
    return scaler.transform(x_fit), scaler.transform(x_score)


def fit_rf(x_fit, y_fit, x_score, model='rf', seed=0):
    from uncertainty_decomposition import decompose_forest
    from model_defaults import sklearn_params
    if model == 'qrf':
        from quantile_forest import RandomForestQuantileRegressor
        forest = RandomForestQuantileRegressor(random_state=seed,
                                               **sklearn_params('qrf'))
    else:
        from sklearn.ensemble import RandomForestRegressor
        forest = RandomForestRegressor(random_state=seed,
                                       **sklearn_params('rf'))
    forest.fit(x_fit, y_fit)
    mean, alea, epis = decompose_forest(forest, x_fit, y_fit, x_score)
    return mean, alea, epis


def fit_gp(x_fit, y_fit, x_score, heteroscedastic=False, seed=0, epochs=150):
    """The two Gaussian processes, built the way models.py builds them."""
    import gpytorch
    import torch
    from model_defaults import gp_fit_threads
    from uncertainty_decomposition import decompose_gp, decompose_hetero_gp

    torch.manual_seed(seed)
    xf, xs = _standardise(x_fit, x_score)
    xt = torch.tensor(xf, dtype=torch.float32)
    yt = torch.tensor(y_fit, dtype=torch.float32)
    xst = torch.tensor(xs, dtype=torch.float32)

    class _GP(gpytorch.models.ExactGP):
        def __init__(self, tx, ty, lik):
            super().__init__(tx, ty, lik)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel())

        def forward(self, z):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(z), self.covar_module(z))

    lik = gpytorch.likelihoods.GaussianLikelihood()
    gp = _GP(xt, yt, lik)
    # The width starts at the median distance between training molecules. The
    # library default of 0.7 sits far below the real distances, the fit has
    # nothing to learn from and collapses to one flat prediction, and the score
    # then reads as a weak representation rather than a failed fit
    # (RERUN_PLAN.md 2.8f).
    with torch.no_grad():
        d = torch.cdist(xt[:500], xt[:500])
        med = float(d[d > 0].median())
        gp.covar_module.base_kernel.lengthscale = max(med, 1e-3)

    noise_net = None
    params = [{'params': gp.parameters(), 'lr': 0.1}]
    if heteroscedastic:
        noise_net = torch.nn.Sequential(
            torch.nn.Linear(xt.shape[1], 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 1), torch.nn.Softplus())
        params.append({'params': noise_net.parameters(), 'lr': 0.001})

    opt = torch.optim.Adam(params)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, gp)
    gp.train()
    lik.train()
    # SINGLE-THREADED, through the shared spec's own context manager. Not an
    # optimisation: torch's thread pool and the linear-algebra backend under
    # gpytorch fight over the same cores, and the fit does not merely slow down
    # -- it deadlocks, and it segfaults (RERUN_PLAN.md 2.8e, 2.8e-ter). Both
    # pipelines wrap their Gaussian-process fits this way and so does this.
    with gp_fit_threads():
        for _ in range(epochs):
            opt.zero_grad()
            loss = -mll(gp(xt), yt)
            if noise_net is not None:
                with torch.no_grad():
                    resid = (yt - gp(xt).mean) ** 2
                pv = noise_net(xt).squeeze(-1) + 1e-4
                loss = loss + torch.mean(0.5 * torch.log(pv) + resid / (2 * pv))
            loss.backward()
            opt.step()

    gp.eval()
    lik.eval()
    with gp_fit_threads(), torch.no_grad(), gpytorch.settings.fast_pred_var():
        post = gp(xst)
        mean = post.mean.numpy()
        latent = np.clip(post.variance.numpy(), 1e-12, None)
        if noise_net is not None:
            nv = (noise_net(xst).squeeze(-1) + 1e-4).numpy()
            alea, epis, _ = decompose_hetero_gp(latent, nv)
        else:
            alea, epis, _ = decompose_gp(latent, float(lik.noise.item()))
    return mean, alea, epis


def fit_vbll(x_fit, y_fit, x_score, heteroscedastic=False, seed=0, epochs=150,
             passes=50):
    """The variational models, from models.py's own classes and its own routine."""
    import torch
    from models import (VBLLLayer, VBLLLoss,
                        apply_bayesian_transformation_full_variational,
                        sample_network_split)
    from model_defaults import NEURAL_DEFAULTS, gp_fit_threads

    torch.manual_seed(seed)
    xf, xs = _standardise(x_fit, x_score)
    xt = torch.tensor(xf, dtype=torch.float32)
    yt = torch.tensor(y_fit, dtype=torch.float32).view(-1, 1)
    xst = torch.tensor(xs, dtype=torch.float32)

    h1, h2 = NEURAL_DEFAULTS['dnn']['hidden_sizes']
    net = torch.nn.Sequential(
        torch.nn.Linear(xt.shape[1], h1), torch.nn.ReLU(),
        torch.nn.Linear(h1, h2), torch.nn.ReLU(),
        torch.nn.Linear(h2, 1))
    net = apply_bayesian_transformation_full_variational(
        net, heteroscedastic=heteroscedastic)
    loss_fn = VBLLLoss(net, n_data=len(y_fit))
    opt = torch.optim.Adam(net.parameters(),
                           lr=NEURAL_DEFAULTS['training']['lr'])

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xt, yt),
        batch_size=NEURAL_DEFAULTS['training']['batch_size'], shuffle=True)
    net.train()
    # SINGLE-THREADED, for the same reason the Gaussian process is. Importing
    # models.py pulls in gpytorch, gauche and KeOps, and once they are loaded a
    # plain torch training loop on this machine spends most of its wall clock
    # blocked rather than computing -- measured at 10% of one core
    # (RERUN_PLAN.md 2.8e-bis, 2.8i). The context manager is the shared spec's,
    # not a second copy of the workaround.
    with gp_fit_threads():
        for _ in range(epochs):
            for bx, by in loader:
                opt.zero_grad()
                loss_fn(net(bx), by).backward()
                opt.step()

        net.eval()
        # The pipeline's OWN routine, not a copy of it: if the split changes
        # there, this measurement changes with it.
        mean, alea, epis, _total = sample_network_split(net, xst, passes, 'mse')
    return mean, alea, epis


def fit_ngboost(x_fit, y_fit, x_score, seed=0):
    """NGBoost predicts a distribution per molecule from ONE fit, so its noise
    half varies and it has no model half at all -- there is no second fit to
    disagree with. Recorded as absent, not as zero."""
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    from ngboost.scores import MLE
    from model_defaults import SKLEARN_DEFAULTS
    from uncertainty_decomposition import decompose_single_distribution

    p = dict(SKLEARN_DEFAULTS['ngboost'])
    model = NGBRegressor(Dist=Normal, Score=MLE, verbose=False,
                         random_state=seed,
                         natural_gradient=p['natural_gradient'],
                         n_estimators=p['n_estimators'],
                         learning_rate=p['learning_rate'])
    model.fit(x_fit, y_fit)
    dist = model.pred_dist(x_score)
    alea, epis, _ = decompose_single_distribution(
        np.asarray(dist.scale, dtype=float) ** 2)
    return np.asarray(dist.loc, dtype=float), alea, epis


MODELS = {
    'rf':                   lambda a, b, c, s: fit_rf(a, b, c, 'rf', s),
    'ngboost':              lambda a, b, c, s: fit_ngboost(a, b, c, s),
    'qrf':                  lambda a, b, c, s: fit_rf(a, b, c, 'qrf', s),
    'gauche_rbf':           lambda a, b, c, s: fit_gp(a, b, c, False, s),
    'heteroscedastic_gp':   lambda a, b, c, s: fit_gp(a, b, c, True, s),
    'dnn_bnn_full_variational':
        lambda a, b, c, s: fit_vbll(a, b, c, False, s),
    'dnn_bnn_full_variational_hetero':
        lambda a, b, c, s: fit_vbll(a, b, c, True, s),
}


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------

def measure(models=None, conditions=CONDITIONS, level=0.6, n_molecules=2000,
            n_folds=3, seed=0, out_csv=None, verbose=True):
    """One row per (model, representation, condition, level, statistic).

    Nothing is pooled and nothing is averaged across models, conditions or
    levels. The zero-noise value for a model is written as its own rows, at
    level 0.0, so a reader can put every correlation beside it.
    """
    from scipy.stats import spearmanr
    from sklearn.metrics import r2_score
    from uncertainty_decomposition import support

    models = list(MODELS) if models is None else list(models)
    x, y, groups = load_qm9(n_molecules, seed=seed)
    spread = float(np.std(y))
    if verbose:
        print(f"  {len(y)} real QM9 molecules, {x.shape[1]} descriptors, "
              f"label '{LABEL}', spread {spread:.4f}, "
              f"{len(np.unique(groups))} scaffold groups")

    rows = []
    for condition in conditions:
        # WHAT THE CORRELATION IS TAKEN AGAINST.
        #
        # For an uneven design it is that design's own shape: the amount of
        # noise each molecule received, at a fixed reference level so the target
        # is the same column at every level INCLUDING zero. That is what makes
        # the zero-noise subtraction work -- it isolates the pre-existing
        # association between a model's uncertainty and whatever the shape is
        # built from.
        #
        # For the EVEN design there is no shape of its own to correlate against:
        # it is constant, and a rank correlation against a constant is
        # undefined, not zero. The control is therefore scored against THE
        # PATTERNS FROM THE UNEVEN CASES. That is the whole point of it: the
        # question is whether a model that scores well under two_block or graded
        # still scores against those same patterns when the noise carries no
        # such structure -- if it does, its positive result there was never
        # noise localisation.
        references = reference_patterns(condition, x, spread, seed)
        for lvl in (0.0, float(level)):
            scales = noise_scales(condition, x, lvl, spread, seed)
            y_noisy = apply_noise(y, scales, seed + 991)

            for name in models:
                fit = MODELS[name]
                try:
                    mean, alea, epis, fold = _oof(
                        lambda a, b, c: fit(a, b, c, seed), x, y_noisy, groups,
                        n_folds)
                except Exception as exc:
                    # A model this interpreter cannot build is BLOCKED, not a
                    # result. The quantile forest is the standing case: it needs
                    # scikit-learn 1.6.1 with quantile-forest 1.4.1 and this
                    # laptop has 1.3.2 (RERUN_PLAN.md 3.4.4d). A blocked model
                    # is written into the file as such, so a reader cannot
                    # mistake its absence for a null.
                    if verbose:
                        print(f"    {condition:<10} level {lvl:.1f}  {name:<32} "
                              f"BLOCKED: {type(exc).__name__}: "
                              f"{str(exc)[:90]}", flush=True)
                    for term in ('aleatoric', 'epistemic'):
                        for ref_name in references:
                            rows.append({
                                'model': name,
                                'representation': 'qm9_descriptors_9',
                                'noise_condition': condition,
                                'reference_pattern': ref_name,
                                'level': lvl,
                                'term': term,
                                'support': 'blocked',
                                'spearman_vs_pattern': float('nan'),
                                'is_constant': 0,
                                'r2_oof': float('nan'),
                                'n_molecules': 0,
                                'n_folds': n_folds,
                                'seed': seed,
                                'blocked': f"{type(exc).__name__}: {exc}",
                            })
                    continue

                ok = np.isfinite(mean)
                r2 = float(r2_score(y[ok], mean[ok]))
                alea_kind, epis_kind = support(name)

                for term, values, kind in (('aleatoric', alea, alea_kind),
                                           ('epistemic', epis, epis_kind)):
                    if values is None:
                        # ABSENT, not constant: the model produced no such term.
                        v = None
                        constant = False
                    else:
                        v = np.asarray(values, dtype=float)
                        good = ok & np.isfinite(v)
                        # CONSTANT WITHIN EACH INNER FIT, not down the whole
                        # column. These rows come from several fits, so a term
                        # that is one number per fit -- a homoscedastic
                        # likelihood noise, a variational layer's single
                        # observation noise -- takes a DIFFERENT value in each
                        # fold. Judged down the column it looks like a term that
                        # varies, and a correlation against it is then a
                        # correlation with fold membership. That is the same
                        # rule the writer's guard uses (RERUN_PLAN.md 5.5g).
                        constant = all(
                            float(np.nanmax(v[good & (fold == f)])
                                  - np.nanmin(v[good & (fold == f)])) <= 1e-12
                            for f in np.unique(fold[good]))
                    for ref_name, ref in references.items():
                        # A rank correlation against a constant column is
                        # UNDEFINED, not zero. Writing 0.0 would read as "the
                        # model found nothing", which is a claim about the
                        # model; the truth is that the column cannot answer the
                        # question at all.
                        rho = (float('nan') if (v is None or constant)
                               else float(spearmanr(v[good], ref[good])[0]))
                        rows.append({
                            'model': name,
                            'representation': 'qm9_descriptors_9',
                            'noise_condition': condition,
                            'reference_pattern': ref_name,
                            'level': lvl,
                            'term': term,
                            'support': kind,
                            'spearman_vs_pattern': rho,
                            'is_constant': int(constant),
                            'r2_oof': r2,
                            'n_molecules': int(ok.sum()),
                            'n_folds': n_folds,
                            'seed': seed,
                            'blocked': '',
                        })
                if verbose:
                    for ref_name in references:
                        def _fmt(t):
                            r = lookup(rows, name, condition, t, lvl, ref_name)
                            if r['support'] == 'none':
                                return '    (absent)'
                            if r['is_constant']:
                                return '  (constant)'
                            return f"{r['spearman_vs_pattern']:+.4f}"
                        print(f"    {condition:<10} level {lvl:.1f}  "
                              f"vs {ref_name:<10} {name:<32} "
                              f"R2 {r2:+.4f}  aleatoric {_fmt('aleatoric')}  "
                              f"epistemic {_fmt('epistemic')}", flush=True)

    if out_csv:
        import pandas as pd
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        if verbose:
            print(f"  wrote {len(rows)} rows to {out_csv}")
    return rows


def measure_real_conditions(models, reps, conditions, level=0.6,
                            n_molecules=900, n_folds=3, seed=0, out_csv=None,
                            verbose=True):
    """The same three questions, asked with the STUDY'S OWN noise and features.

    One row per (model, representation, condition, level, term). Nothing pooled.

    Two things this adds over the synthetic designs above.

      The noise is the injector's, so a result here is about a condition the
      study actually runs rather than about a shape written for a test.

      The features are the study's own representations, so a split that depends
      on having nine tidy descriptors shows up as a split that stops working on
      a fingerprint.

    The correlation target is the condition's own per-molecule noise scale, at a
    fixed reference level. For a condition that gives every molecule the same
    amount that column is CONSTANT, the question is undefined rather than zero,
    and the row says so instead of carrying a number.
    """
    from scipy.stats import spearmanr
    from sklearn.metrics import r2_score
    from uncertainty_decomposition import support

    x_desc, y, groups = load_qm9(n_molecules, seed=seed)
    smiles = _selected_smiles(n_molecules, seed)
    spread = float(np.std(y))
    if verbose:
        print(f"  {len(y)} real QM9 molecules, label '{LABEL}', spread "
              f"{spread:.4f}, {len(np.unique(groups))} scaffold groups")
        print(f"  building representations: {', '.join(reps)}", flush=True)
    features = build_representations(smiles, reps)

    rows = []
    for condition in conditions:
        pattern, varies = injector_patterns(condition, y, groups, spread)
        if verbose:
            _note = ('VARIES per molecule' if varies else
                     'is the SAME for every molecule, so the question is '
                     'UNDEFINED here rather than zero: this condition is the '
                     "study's own negative control")
            print(f"\n  condition {condition}: noise scale {_note}", flush=True)
        for lvl in (0.0, float(level)):
            y_noisy = injector_noise(condition, y, lvl, groups, spread, seed)
            for rep_name in reps:
                x = np.asarray(features[rep_name], dtype=float)
                for name in models:
                    fit = MODELS[name]
                    try:
                        mean, alea, epis, fold = _oof(
                            lambda a, b, c: fit(a, b, c, seed), x, y_noisy,
                            groups, n_folds)
                    except Exception as exc:
                        if verbose:
                            print(f"    {condition:<14} {lvl:.1f} {rep_name:<7} "
                                  f"{name:<32} BLOCKED: {type(exc).__name__}: "
                                  f"{str(exc)[:70]}", flush=True)
                        for term in ('aleatoric', 'epistemic'):
                            rows.append(_row(name, rep_name, condition, lvl,
                                             term, 'blocked', float('nan'), 0,
                                             float('nan'), 0, n_folds, seed,
                                             varies,
                                             f"{type(exc).__name__}: {exc}"))
                        continue

                    ok = np.isfinite(mean)
                    r2 = float(r2_score(y[ok], mean[ok]))
                    alea_kind, epis_kind = support(name)
                    for term, values, kind in (('aleatoric', alea, alea_kind),
                                               ('epistemic', epis, epis_kind)):
                        if values is None:
                            # ABSENT, not constant. NGBoost has no model half at
                            # all: one fit has no ensemble to disagree with.
                            # Calling that 'constant' would say the model
                            # produced one number, when it produced none.
                            constant, rho = False, float('nan')
                        else:
                            v = np.asarray(values, dtype=float)
                            good = ok & np.isfinite(v)
                            constant = all(
                                float(np.nanmax(v[good & (fold == f)])
                                      - np.nanmin(v[good & (fold == f)])) <= 1e-12
                                for f in np.unique(fold[good]))
                            rho = (float('nan') if (constant or not varies)
                                   else float(spearmanr(v[good],
                                                        pattern[good])[0]))
                        rows.append(_row(name, rep_name, condition, lvl, term,
                                         kind, rho, int(constant), r2,
                                         int(ok.sum()), n_folds, seed, varies,
                                         ''))
                    if verbose:
                        def _f(t):
                            r = [r for r in rows if r['model'] == name
                                 and r['representation'] == rep_name
                                 and r['noise_condition'] == condition
                                 and abs(r['level'] - lvl) < 1e-12
                                 and r['term'] == t][-1]
                            # Three different things, and they must not read the
                            # same: the model has no such term; the term is one
                            # number for the whole fit; the noise itself has no
                            # pattern to find.
                            if r['support'] == 'none':
                                return '    (absent)'
                            if r['is_constant']:
                                return '  (constant)'
                            if not r['pattern_varies']:
                                return ' (undefined)'
                            if np.isnan(r['spearman_vs_pattern']):
                                return ' (undefined)'
                            return f"{r['spearman_vs_pattern']:+.4f}"
                        print(f"    {condition:<14} {lvl:.1f} {rep_name:<7} "
                              f"{name:<32} R2 {r2:+.4f}  aleatoric {_f('aleatoric')}"
                              f"  epistemic {_f('epistemic')}", flush=True)

    if out_csv:
        import pandas as pd
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        if verbose:
            print(f"\n  wrote {len(rows)} rows to {out_csv}")
    return rows


def _row(model, rep, condition, level, term, support_kind, rho, constant, r2,
         n, n_folds, seed, pattern_varies, blocked):
    return {
        'model': model,
        'representation': rep,
        'noise_condition': condition,
        'reference_pattern': condition,
        'level': level,
        'term': term,
        'support': support_kind,
        'spearman_vs_pattern': rho,
        'is_constant': constant,
        'pattern_varies': int(bool(pattern_varies)),
        'r2_oof': r2,
        'n_molecules': n,
        'n_folds': n_folds,
        'seed': seed,
        'blocked': blocked,
    }


def _selected_smiles(n_molecules, seed):
    """The SMILES of exactly the molecules `load_qm9` selects, in the same order."""
    import pandas as pd
    frame = pd.read_csv(os.path.join(REPO, 'data', 'QM9', 'raw', 'gdb9.sdf.csv'))
    smiles = qm9_smiles(len(frame))
    frame = frame.iloc[:len(smiles)].copy()
    frame['__smiles'] = smiles
    frame = frame[frame['__smiles'] != '']
    rng = np.random.default_rng(seed)
    take = rng.choice(len(frame), size=min(n_molecules, len(frame)),
                      replace=False)
    take.sort()
    return list(frame.iloc[take]['__smiles'])


def lookup(rows, model, condition, term, level, reference=None):
    for r in rows:
        if (r['model'] == model and r['noise_condition'] == condition
                and r['term'] == term and abs(r['level'] - level) < 1e-12
                and (reference is None
                     or r['reference_pattern'] == reference)):
            return r
    raise KeyError(f"{model} / {condition} / {term} / {level} / {reference}")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--models', nargs='*', default=None)
    ap.add_argument('--conditions', nargs='*', default=list(CONDITIONS))
    ap.add_argument('--level', type=float, default=0.6)
    ap.add_argument('--n-molecules', type=int, default=2000)
    ap.add_argument('--n-folds', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    ap.add_argument('--real-conditions', action='store_true',
                    help="Use the injector's own noise conditions and the "
                         "study's own representations instead of the three "
                         "designs written for the checks.")
    ap.add_argument('--reps', nargs='*', default=['ECFP4', 'PDV', 'SNS'])
    a = ap.parse_args()
    if a.real_conditions:
        measure_real_conditions(
            models=a.models or list(MODELS), reps=a.reps,
            conditions=a.conditions, level=a.level,
            n_molecules=a.n_molecules, n_folds=a.n_folds, seed=a.seed,
            out_csv=a.out)
    else:
        measure(models=a.models, conditions=a.conditions, level=a.level,
                n_molecules=a.n_molecules, n_folds=a.n_folds, seed=a.seed,
                out_csv=a.out)
