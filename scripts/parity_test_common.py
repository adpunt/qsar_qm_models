"""Shared fixtures for the two Chat E parity tests.

Both tests answer a question that decides a setting used in BOTH pipelines, so
they have to load QM9 the way the production pipeline does, not a convenient
approximation. What is shared:

  * QM9 loading from the raw SDF, with the uncharacterised molecules removed
  * the two representations that behave differently under the forest feature
    setting: the continuous descriptor vector (PDV) and Morgan radius-2
  * a scaffold split that holds out whole Murcko scaffold groups
  * label noise added to TRAINING labels only -- held-out labels stay clean,
    which is the fix in commit 9d7db67 and the thing that made every previous
    QM9 result invalid
  * quantile-forest emulation, because quantile_forest 1.4.1 cannot be fitted
    against scikit-learn 1.3.2 in this environment (see QRF_UNAVAILABLE_REASON)

Nothing here writes to results/. The test scripts do that.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from sklearn.ensemble._forest import _generate_sample_indices, _get_n_samples_bootstrap

RDLogger.DisableLog('rdApp.*')

REPO = Path(__file__).resolve().parent.parent
QM9_SDF = REPO / 'data' / 'QM9' / 'raw' / 'gdb9.sdf'
QM9_CSV = REPO / 'data' / 'QM9' / 'raw' / 'gdb9.sdf.csv'
QM9_UNCHAR = REPO / 'data' / 'QM9' / 'raw' / 'uncharacterized.txt'
CACHE = REPO / 'results' / 'parity_tests' / '_cache'

HARTREE_TO_EV = 27.2114
TARGET = 'gap'          # homo_lumo_gap, the production target


# ---------------------------------------------------------------------------
# quantile_forest availability
# ---------------------------------------------------------------------------

def probe_quantile_forest():
    """Return (available, reason). Actually FITS one, because importing succeeds
    in this environment and fitting does not."""
    try:
        from quantile_forest import RandomForestQuantileRegressor
    except Exception as exc:
        return False, f'import failed: {type(exc).__name__}: {exc}'
    try:
        rng = np.random.RandomState(0)
        X = rng.normal(size=(50, 4))
        y = X[:, 0] + rng.normal(scale=0.1, size=50)
        m = RandomForestQuantileRegressor(n_estimators=5, random_state=0).fit(X, y)
        m.predict(X[:3], quantiles=[0.16, 0.5, 0.84])
    except Exception as exc:
        return False, f'fit failed: {type(exc).__name__}: {exc}'
    return True, 'ok'


HAS_QRF, QRF_UNAVAILABLE_REASON = probe_quantile_forest()


# ---------------------------------------------------------------------------
# QM9
# ---------------------------------------------------------------------------

def _uncharacterized_indices():
    """The 3,054 molecules the QM9 authors flag as having failed the consistency
    check. process_and_train.py drops them; so do we."""
    if not QM9_UNCHAR.exists():
        return set()
    bad = set()
    for line in QM9_UNCHAR.read_text().splitlines():
        parts = line.split()
        if parts and parts[0].isdigit():
            bad.add(int(parts[0]) - 1)      # file is 1-indexed by gdb9 id
    return bad


def load_qm9(n_molecules, cache=True):
    """Return (smiles, y, scaffolds). y is the HOMO-LUMO gap in eV.

    Molecules are taken in file order, which is what the production pipeline
    does with -n; the scaffold split is what makes the held-out set hard, not
    the ordering.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE / f'qm9_{n_molecules}.npz'
    if cache and cache_file.exists():
        d = np.load(cache_file, allow_pickle=True)
        return list(d['smiles']), d['y'], d['scaffolds']

    import pandas as pd
    if not QM9_SDF.exists():
        raise FileNotFoundError(f'QM9 SDF not found at {QM9_SDF}')
    df = pd.read_csv(QM9_CSV)
    skip = _uncharacterized_indices()
    sup = Chem.SDMolSupplier(str(QM9_SDF), removeHs=False)

    smiles, y = [], []
    for i, mol in enumerate(sup):
        if mol is None or i in skip:
            continue
        smiles.append(Chem.MolToSmiles(mol))
        y.append(float(df.iloc[i][TARGET]) * HARTREE_TO_EV)
        if len(smiles) >= n_molecules:
            break
    y = np.asarray(y, dtype=float)

    mols = [Chem.MolFromSmiles(s) for s in smiles]
    scaffolds = np.array([
        MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
        if m is not None else ''
        for m in mols
    ])
    if cache:
        np.savez_compressed(cache_file, smiles=np.array(smiles, dtype=object),
                            y=y, scaffolds=scaffolds)
    return smiles, y, scaffolds


def build_representations(smiles, which=('descriptors', 'morgan'), cache=True):
    """Morgan radius-2 2048 bits, and the continuous descriptor vector (PDV).

    Standardisation of the descriptors is NOT done here -- it is train-only and
    therefore happens after the split, the way both pipelines do it.

    Cached per (representation, molecule count): the 208 RDKit descriptors cost
    about 100 seconds per 2,000 molecules and both tests need them.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    out, todo = {}, []
    for name in which:
        f = CACHE / f'{name}_{len(smiles)}.npy'
        if cache and f.exists():
            out[name] = np.load(f)
        else:
            todo.append(name)
    if not todo:
        return out

    mols = [Chem.MolFromSmiles(s) for s in smiles]
    if 'morgan' in todo:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        out['morgan'] = np.array([gen.GetFingerprintAsNumPy(m) for m in mols],
                                 dtype=np.float32)
    if 'descriptors' in todo:
        names = [d[0] for d in Descriptors._descList]
        calc = MolecularDescriptorCalculator(names)
        raw = np.array([calc.CalcDescriptors(m) for m in mols], dtype=float)
        out['descriptors'] = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    if cache:
        for name in todo:
            np.save(CACHE / f'{name}_{len(smiles)}.npy', out[name])
    return out


def scaffold_split(scaffolds, seed, train_frac=0.8, val_frac=0.0):
    """Hold out whole Murcko scaffold groups.

    val_frac=0 gives the 80/20 that the QM9 tree models effectively run, because
    they stack validation back into training (models.py:1382). Pass val_frac to
    carve one out for models that need it (the calibration test does).
    """
    rng = np.random.RandomState(seed)
    groups = np.unique(scaffolds)
    rng.shuffle(groups)
    n = len(scaffolds)
    train, val, test = [], [], []
    for g in groups:
        idx = np.where(scaffolds == g)[0]
        if len(train) < train_frac * n:
            train.extend(idx)
        elif len(val) < val_frac * n:
            val.extend(idx)
        else:
            test.extend(idx)
    return np.array(train), np.array(val, dtype=int), np.array(test)


def standardise(X_train, *others):
    """Train-only per-feature standardisation. Returns the same number of arrays."""
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd == 0] = 1.0
    return tuple((X - mu) / sd for X in (X_train,) + others)


def inject_gaussian(y_train, sigma, seed):
    """Gaussian noise at `sigma` label spreads, TRAINING LABELS ONLY.

    Scaled by the CLEAN training label spread, so sigma means the same thing at
    every level. (The production Rust injector standardises using the noisy
    spread instead -- RERUN_PLAN.md 2.4. That is a separate defect, owned by
    Chat A, and using it here would make the noise levels non-comparable.)
    """
    if sigma == 0:
        return np.array(y_train, dtype=float), np.zeros(len(y_train))
    rng = np.random.RandomState(90000 + seed)
    eps = rng.normal(0.0, sigma * float(np.std(y_train)), len(y_train))
    return np.asarray(y_train, dtype=float) + eps, eps


# ---------------------------------------------------------------------------
# quantile-forest emulation
# ---------------------------------------------------------------------------

def forest_leaf_stats(forest, X_train, y_train, X_query):
    """Per-tree leaf mean and within-leaf variance for every query point.

    Reproduces exactly the in-bag rows each tree was grown on, so the leaf
    statistics are the ones the tree itself saw. Returns
    (tree_means, tree_vars), each (n_query, n_trees).
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train, dtype=float).ravel()
    n_train = X_train.shape[0]

    train_leaves = forest.apply(X_train)
    query_leaves = forest.apply(X_query)
    n_boot = (_get_n_samples_bootstrap(n_train, forest.max_samples)
              if forest.bootstrap else None)

    tree_means = np.empty((query_leaves.shape[0], len(forest.estimators_)))
    tree_vars = np.empty_like(tree_means)

    for i, est in enumerate(forest.estimators_):
        idx = (_generate_sample_indices(est.random_state, n_train, n_boot)
               if forest.bootstrap else np.arange(n_train))
        leaf_of, y_of = train_leaves[idx, i], y_train[idx]
        n_nodes = est.tree_.node_count
        cnt = np.bincount(leaf_of, minlength=n_nodes).astype(float)
        s1 = np.bincount(leaf_of, weights=y_of, minlength=n_nodes)
        s2 = np.bincount(leaf_of, weights=y_of ** 2, minlength=n_nodes)
        safe = np.where(cnt > 0, cnt, 1.0)
        leaf_mean = s1 / safe
        leaf_var = np.maximum(s2 / safe - leaf_mean ** 2, 0.0)
        tree_means[:, i] = leaf_mean[query_leaves[:, i]]
        tree_vars[:, i] = leaf_var[query_leaves[:, i]]
    return tree_means, tree_vars


def forest_quantiles(forest, X_train, y_train, X_query, quantiles=(0.16, 0.5, 0.84),
                     max_pool=None):
    """Emulate RandomForestQuantileRegressor.predict(quantiles=...).

    quantile_forest pools the in-bag training labels that land in the same leaf
    as the query point, across all trees, and takes empirical quantiles of that
    pooled set. That is what this does. Verified against the real estimator in
    parity_test_forest.py --verify-emulation wherever the library can be fitted.
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train, dtype=float).ravel()
    n_train = X_train.shape[0]

    train_leaves = forest.apply(X_train)
    query_leaves = forest.apply(X_query)
    n_boot = (_get_n_samples_bootstrap(n_train, forest.max_samples)
              if forest.bootstrap else None)

    # leaf -> in-bag label list, per tree
    per_tree = []
    for i, est in enumerate(forest.estimators_):
        idx = (_generate_sample_indices(est.random_state, n_train, n_boot)
               if forest.bootstrap else np.arange(n_train))
        leaf_of, y_of = train_leaves[idx, i], y_train[idx]
        order = np.argsort(leaf_of, kind='stable')
        leaf_sorted, y_sorted = leaf_of[order], y_of[order]
        bounds = np.searchsorted(leaf_sorted, np.arange(est.tree_.node_count + 1))
        per_tree.append((y_sorted, bounds))

    out = np.empty((len(X_query), len(quantiles)))
    rng = np.random.RandomState(0)
    for r in range(len(X_query)):
        pooled = [y_sorted[bounds[query_leaves[r, i]]:bounds[query_leaves[r, i] + 1]]
                  for i, (y_sorted, bounds) in enumerate(per_tree)]
        pool = np.concatenate(pooled) if pooled else np.array([np.nan])
        if max_pool is not None and len(pool) > max_pool:
            pool = pool[rng.choice(len(pool), max_pool, replace=False)]
        out[r] = np.quantile(pool, quantiles)
    return out


def env_fingerprint():
    """Package versions, so a results CSV can be traced to what produced it."""
    import importlib
    import sys
    versions = {'python': sys.version.split()[0]}
    for name in ('sklearn', 'numpy', 'scipy', 'rdkit', 'xgboost', 'lightgbm',
                 'ngboost', 'quantile_forest', 'torch', 'gpytorch', 'gauche',
                 'botorch', 'torchbnn'):
        try:
            versions[name] = getattr(importlib.import_module(name), '__version__', '?')
        except Exception:
            versions[name] = 'MISSING'
    versions['threads'] = os.environ.get('OMP_NUM_THREADS', 'unset')
    return versions


# ---------------------------------------------------------------------------
# hERG Ki -- the experimental dataset the count-scaling test needs
# ---------------------------------------------------------------------------

def load_herg(cache=True):
    """Return (smiles, y, scaffolds) for ChEMBL hERG Ki, pKi as the label.

    Fetched and filtered exactly as KIRBy does it
    (alternative_data_noise_robustness.py fetch_chembl_herg_ki): binding assays
    only, median pChEMBL per compound, compounds whose replicate spread exceeds
    one log unit dropped. Cached under results/parity_tests/_cache so the ChEMBL
    API is hit once.

    hERG is here because the substructure-count question has to be answered on a
    real experimental set as well as on QM9: QM9 molecules are small and
    saturated, so a substructure repeats far less often there than in a
    drug-like series, and how often a substructure repeats is the whole subject.
    """
    import pandas as pd
    CACHE.mkdir(parents=True, exist_ok=True)
    raw = CACHE / 'chembl_herg_ki.csv'
    prepared = CACHE / 'herg_prepared.npz'
    if cache and prepared.exists():
        d = np.load(prepared, allow_pickle=True)
        return list(d['smiles']), d['y'], d['scaffolds']

    if raw.exists():
        df = pd.read_csv(raw)
    else:
        import time
        import requests
        records, offset = [], 0
        while True:
            resp = requests.get(
                'https://www.ebi.ac.uk/chembl/api/data/activity.json',
                params={'target_chembl_id': 'CHEMBL240', 'standard_type': 'Ki',
                        'pchembl_value__isnull': 'false', 'standard_relation': '=',
                        'data_validity_comment__isnull': 'true',
                        'limit': 1000, 'offset': offset, 'format': 'json'},
                timeout=60)
            resp.raise_for_status()
            data = resp.json()
            acts = data.get('activities', [])
            if not acts:
                break
            records += [{'canonical_smiles': a.get('canonical_smiles'),
                         'pchembl_value': a.get('pchembl_value'),
                         'assay_type': a.get('assay_type')} for a in acts]
            if data.get('page_meta', {}).get('next') is None:
                break
            offset += 1000
            time.sleep(0.5)
        if not records:
            raise RuntimeError('no hERG Ki records retrieved from ChEMBL')
        df = pd.DataFrame(records)
        df = df[df['assay_type'] == 'B'].copy()
        df['pchembl_value'] = pd.to_numeric(df['pchembl_value'], errors='coerce')
        df = df.dropna(subset=['pchembl_value', 'canonical_smiles'])
        grouped = df.groupby('canonical_smiles')['pchembl_value']
        merged = (grouped.median().reset_index()
                  .merge(grouped.std().reset_index().rename(
                      columns={'pchembl_value': 'std'}), on='canonical_smiles'))
        merged = merged[(merged['std'].isna()) | (merged['std'] <= 1.0)]
        df = merged[['canonical_smiles', 'pchembl_value']].copy()
        df.columns = ['SMILES', 'pKi']
        df.to_csv(raw, index=False)

    smiles, y = [], []
    for s, label in zip(df['SMILES'], df['pKi']):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        smiles.append(Chem.MolToSmiles(mol))
        y.append(float(label))
    y = np.asarray(y, dtype=float)
    scaffolds = np.array([
        MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s),
                                            includeChirality=False)
        for s in smiles])
    if cache:
        np.savez_compressed(prepared, smiles=np.array(smiles, dtype=object),
                            y=y, scaffolds=scaffolds)
    return smiles, y, scaffolds


# ---------------------------------------------------------------------------
# Sort & Slice
# ---------------------------------------------------------------------------

def sort_and_slice(smiles, train_idx, vec_dimension=1024):
    """Sort & Slice substructure counts, the pooling operator fitted on TRAINING
    molecules only -- which is what makes it a fitted representation rather than
    a hash.

    The featuriser is imported from scripts/process_and_train.py rather than
    reimplemented. Reimplementing it is what this whole class of question keeps
    going wrong on: the representation is identified by its name and then turns
    out to be a different function (RERUN_PLAN.md section 3.4.1). The import is
    slow -- it pulls the whole pipeline -- and that is the price of measuring the
    real thing.

    Returns the raw COUNT matrix. The production storage path packs this to
    presence bits (process_and_train.py:539-541), so `(X > 0)` is what the
    models see today and `X` is what they will see once that is fixed.
    """
    import sys
    sys.path.insert(0, str(REPO / 'scripts'))
    from process_and_train import create_sort_and_slice_ecfp_featuriser

    mols = [Chem.MolFromSmiles(s) for s in smiles]
    mols_train = [mols[i] for i in train_idx if mols[i] is not None]
    featuriser = create_sort_and_slice_ecfp_featuriser(
        mols_train=mols_train, max_radius=2, pharm_atom_invs=False,
        bond_invs=True, chirality=False, sub_counts=True,
        vec_dimension=vec_dimension, print_train_set_info=False)
    rows = []
    for mol in mols:
        vec = featuriser(mol) if mol is not None else np.zeros(vec_dimension)
        vec = np.asarray(vec, dtype=np.float32)
        if vec.shape != (vec_dimension,):       # a molecule with no substructures
            vec = np.zeros(vec_dimension, dtype=np.float32)
        rows.append(vec)
    return np.vstack(rows)
