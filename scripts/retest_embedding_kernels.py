#!/usr/bin/env python3
"""Did the storage defect break the radial-basis kernel? Measure it, don't infer it.

RERUN_PLAN.md 10b.2 harvested ten seed-matched clean replicates from the cluster and
found the two kernels agreeing to within 0.004 wherever the features are binary
(ECFP4 0.8225 Tanimoto against 0.8201 radial basis) and the radial basis collapsing
on the two learned embeddings (MHG-GNN 0.8724 against -0.0158; mol2vec 0.8677
against 0.0087). That collapse was ATTRIBUTED to the per-molecule rescaling in
2.8c. Attribution is not measurement, and 13.6 says so.

This script measures it. For each representation it scores the same molecules, the
same scaffold split and the same seeds twice:

  retired  -- each molecule's own range stretched to fill 0-255, stored as bytes,
              handed to the model unscaled. What the pipeline used to do.
  fixed    -- the model's own float values, standardised per feature with constants
              fitted on the training split. What the pipeline does now.

PDV is the reference row, and its two cells mean different things from the
embeddings'. It was ALWAYS stored as floats and standardised, so its `fixed` cell
is the real pipeline number (0.8890 in the harvest, the best cell in the study)
and its `retired` cell is counterfactual -- what the defect would have done to it
had it been stored the way the embeddings were.

Distance is also reported directly, because the kernel's collapse is a distance
phenomenon and a number that needs no model fitting is worth having beside one
that does: how strongly the distance between two molecules tracks the difference
in their labels. If the storage destroys comparability, that correlation should
fall towards zero whatever the model does with it afterwards.

The molecule count is far below the cluster's 10,000, so the absolute numbers are
NOT comparable to the harvest. The paired difference within a row is the result.

Usage:
    python scripts/retest_embedding_kernels.py --n-molecules 2000 --replicates 3
"""
import argparse
import os
import sys
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SCRIPTS)
for d in ('models', 'preprocessing', 'results', 'scripts'):
    sys.path.insert(0, os.path.join(REPO, d))

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import r2_score

RDLogger.DisableLog('rdApp.*')

import torch
import gpytorch
import gauche
try:                                    # matches models/models.py
    from botorch.fit import fit_gpytorch_mll as fit_gpytorch_model
except ImportError:
    from botorch import fit_gpytorch_model

import process_and_train as P
from models import Gauche

HARTREE_TO_EV = 27.211386

KERNELS = {
    'tanimoto': gauche.kernels.fingerprint_kernels.tanimoto_kernel.TanimotoKernel,
    'rbf': gpytorch.kernels.RBFKernel,
}


CACHE = os.path.join(REPO, 'results', 'embedding_storage_retest', 'qm9_smiles_gap.npz')


def load_qm9():
    """HOMO-LUMO gap in electronvolts, with the SMILES from the structure file.

    The SMILES are NOT in the properties table -- they come from the structure
    file, and the uncharacterised molecules are dropped. investigate_pdv.py
    already does exactly this, so this reuses it rather than repeating it, and
    caches the result because reading the structure file takes a minute.
    """
    if os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        return d['smiles'], d['y']

    from investigate_pdv import load_qm9_data
    smiles, gap_hartree = load_qm9_data(n_samples=10 ** 9, seed=0)
    smiles = np.array(smiles, dtype=object)
    y = (np.asarray(gap_hartree, dtype=np.float64) * HARTREE_TO_EV)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, smiles=smiles, y=y)
    return smiles, y


def scaffold_split(smiles_list, seed, train_frac=0.8):
    """Scaffold split, matching scripts/tune_svm_all_reps.py. Whole scaffolds move together."""
    scaffolds = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            scaffolds['NONE'].append(i)
            continue
        try:
            scaffolds[MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)].append(i)
        except Exception:
            scaffolds['ERROR'].append(i)

    groups = list(scaffolds.values())
    np.random.RandomState(seed).shuffle(groups)

    cutoff = int(train_frac * len(smiles_list))
    train, test = [], []
    for g in groups:
        (train if len(train) < cutoff else test).extend(g)
    return np.array(train), np.array(test)


# ── the representations, as raw float values ───────────────────────────────

def chemberta_batched(smiles_list, batch_size=64):
    """Same embedding as P.chemberta_fingerprint, computed a batch at a time.

    The pipeline embeds one molecule per call, which costs about 0.8 seconds each.
    Batching needs the mean pooling to ignore padding, or a short molecule in a
    batch with a long one gets a different vector than it would alone. The
    attention mask is what makes the two identical -- asserted below.
    """
    tokenizer, model = P.get_chemberta_model()
    out = []
    for i in range(0, len(smiles_list), batch_size):
        chunk = list(smiles_list[i:i + batch_size])
        inputs = tokenizer(chunk, return_tensors="pt", padding=True,
                           truncation=True, max_length=512)
        with torch.no_grad():
            hidden = model(**inputs).last_hidden_state
        mask = inputs['attention_mask'].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        out.append(pooled.numpy().astype(np.float32))
    return np.vstack(out)


def build_raw(rep, smiles_list):
    """One float vector per molecule, exactly as the model produces it."""
    if rep == 'mol2vec':
        model = P.load_mol2vec_model(P.get_mol2vec_model_path(
            argparse.Namespace(mol2vec_model=None)))
        out = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            out.append(P.mol2vec_fingerprint(mol, model, 300))
        return np.vstack(out)
    if rep == 'chemberta':
        batched = chemberta_batched(smiles_list)
        # The batched path must agree with the pipeline's one-at-a-time path.
        solo = P.chemberta_fingerprint(smiles_list[0], 768)
        assert np.allclose(batched[0], solo, atol=1e-4), \
            f"batched ChemBERTa differs from the pipeline's: max {np.abs(batched[0]-solo).max():.2e}"
        return batched
    if rep == 'mhggnn':
        model = P.get_mhg_gnn_model()
        out = []
        for i in range(0, len(smiles_list), 32):
            chunk = list(smiles_list[i:i + 32])
            embs = model.encode(chunk)
            out.append(np.vstack([e.cpu().detach().numpy() for e in embs]).astype(np.float32))
        return np.vstack(out)
    if rep == 'continuous_pdv':
        out = []
        for smi in smiles_list:
            try:
                out.append(P.rdkit_mol_descriptors_from_smiles(smi))
            except Exception:
                out.append(np.zeros(200))
        return np.nan_to_num(np.vstack(out).astype(np.float32), nan=0.0,
                             posinf=0.0, neginf=0.0)
    raise ValueError(rep)


def retired_storage(raw):
    """Each molecule's own smallest and largest value stretched to fill 0-255.

    Reproduced here because the pipeline no longer contains it: this is the thing
    being measured, and it has to come from somewhere.
    """
    out = np.empty(raw.shape, dtype=np.uint8)
    for i, vec in enumerate(raw):
        lo, hi = vec.min(), vec.max()
        out[i] = (((vec - lo) / (hi - lo)) * 255).astype(np.uint8) if hi - lo > 1e-6 \
            else np.zeros(raw.shape[1], dtype=np.uint8)
    return out.astype(np.float64)


def standardise(x_train, x_test):
    """Per feature, fitted on the training split only. The pipeline's own block."""
    mean = np.nanmean(x_train, axis=0)
    std = np.nanstd(x_train, axis=0)
    std[std == 0] = 1.0
    f = lambda x: np.nan_to_num((x - mean) / std, 0.0)
    return f(x_train), f(x_test)


def distance_signal(x, y, rng, n_pairs=20000):
    """How strongly does the distance between two molecules track the gap in their
    labels? Spearman, on random pairs. No model, no fitting -- if the storage has
    destroyed comparability between molecules this falls towards zero on its own.
    """
    from scipy.stats import spearmanr
    n = len(x)
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    d_feat = np.linalg.norm(x[i] - x[j], axis=1)
    d_label = np.abs(y[i] - y[j])
    rho = spearmanr(d_feat, d_label).statistic
    return rho, float(np.median(d_feat))


def fit_gp(x_train, y_train, x_test, kernel_name, iters=150):
    """The pipeline's Gaussian process class, likelihood noise and kernels.

    The fit itself is plain gradient ascent on the marginal likelihood rather than
    the pipeline's botorch call: botorch 0.16 refuses a bare gpytorch model
    ('Gauche' object has no attribute 'transform_inputs'), so the pipeline's own
    call cannot run in this environment either. Both storage schemes are fitted by
    exactly the same routine for the same number of steps, which is what the
    comparison rests on.
    """
    x_tr = torch.from_numpy(np.ascontiguousarray(x_train)).double()
    x_te = torch.from_numpy(np.ascontiguousarray(x_test)).double()
    y_tr = torch.from_numpy(np.ascontiguousarray(y_train)).double()

    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise=1e-3)
    model = Gauche(x_tr, y_tr, likelihood, KERNELS[kernel_name])
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    model.train()
    likelihood.train()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.1)
    for _ in range(iters):
        optimiser.zero_grad()
        loss = -mll(model(x_tr), y_tr)
        loss.backward()
        optimiser.step()

    model.eval()
    likelihood.eval()
    with torch.no_grad():
        return model(x_te).mean.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-molecules', type=int, default=2000)
    ap.add_argument('--replicates', type=int, default=3)
    ap.add_argument('--iters', type=int, default=100,
                    help='Gradient steps on the marginal likelihood. Identical for both '
                         'storage schemes, which is what makes the pair comparable.')
    ap.add_argument('--pool', type=int, default=3000,
                    help='Molecules embedded once and cached; replicates subsample it.')
    ap.add_argument('--reps', nargs='*',
                    default=['mol2vec', 'mhggnn', 'chemberta', 'continuous_pdv'])
    ap.add_argument('--kernels', nargs='*', default=['rbf', 'tanimoto'])
    ap.add_argument('--out', type=str,
                    default=os.path.join(REPO, 'results', 'embedding_storage_retest',
                                         'qm9_kernel_retest.csv'))
    args = ap.parse_args()

    smiles_all, y_all = load_qm9()
    print(f"QM9: {len(smiles_all)} molecules, gap SD {y_all.std():.4f} eV")

    # One pool of molecules, embedded once per representation and cached. Each
    # replicate then draws its own subset and its own scaffold split out of the
    # pool, so the replicates differ in molecules AND split while the expensive
    # part is paid once. Both storage schemes see the identical subset and split.
    pool_size = max(args.n_molecules, args.pool)
    pool_idx = np.random.RandomState(20260826).choice(len(smiles_all), pool_size, replace=False)
    pool_smiles, pool_y = smiles_all[pool_idx], y_all[pool_idx]

    rows = []
    for rep in args.reps:
        cache = os.path.join(os.path.dirname(args.out), f'raw_{rep}_{pool_size}.npy')
        if os.path.exists(cache):
            raw_pool = np.load(cache)
            print(f"\n{rep}: {raw_pool.shape} from cache")
        else:
            t0 = time.time()
            raw_pool = build_raw(rep, pool_smiles)
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            np.save(cache, raw_pool)
            print(f"\n{rep}: built {raw_pool.shape} in {time.time()-t0:.0f}s")

        for replicate in range(args.replicates):
            seed = 20260826 + replicate
            rng = np.random.RandomState(seed)
            sub = rng.choice(pool_size, args.n_molecules, replace=False)
            smiles, y, raw = pool_smiles[sub], pool_y[sub], raw_pool[sub]
            train, test = scaffold_split(smiles, seed)
            print(f"  replicate {replicate}: train {len(train)} test {len(test)}")

            variants = {}
            # What the pipeline used to hand the model.
            old = retired_storage(raw)
            variants['retired'] = (old[train], old[test])
            # What it hands the model now.
            variants['fixed'] = standardise(raw[train].astype(np.float64),
                                            raw[test].astype(np.float64))

            for storage, (x_tr, x_te) in variants.items():
                rho, med_d = distance_signal(x_tr, y[train],
                                             np.random.default_rng(seed))
                print(f"  {storage:8s} distance-vs-label rho = {rho:+.4f}, "
                      f"median distance {med_d:.4g}")
                for kernel in args.kernels:
                    t0 = time.time()
                    try:
                        pred = fit_gp(x_tr, y[train], x_te, kernel, iters=args.iters)
                        r2 = r2_score(y[test], pred)
                    except Exception as e:
                        print(f"  {storage:8s} {kernel:9s} FAILED: {e}")
                        r2 = np.nan
                    rows.append(dict(rep=rep, replicate=replicate, storage=storage,
                                     kernel=kernel, r2=r2,
                                     distance_label_rho=round(rho, 4),
                                     median_distance=round(med_d, 4),
                                     n=args.n_molecules,
                                     n_train=len(train), n_test=len(test),
                                     seconds=round(time.time() - t0, 1)))
                    print(f"  {storage:8s} {kernel:9s} R2 = {r2:+.4f}  ({rows[-1]['seconds']}s)")

            pd.DataFrame(rows).to_csv(args.out, index=False)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    print("\n" + "=" * 72)
    print("Mean R2 over replicates. Every replicate is kept in the CSV; nothing is")
    print("averaged across representations or kernels.")
    print("=" * 72)
    table = df.pivot_table(index='rep', columns=['kernel', 'storage'], values='r2',
                           aggfunc='mean')
    print(table.round(4).to_string())
    print(f"\nrows: {args.out}")


if __name__ == '__main__':
    main()
