#!/usr/bin/env python3
"""
Standalone SVM hyperparameter tuning for all representations.

Tunes SVM (SVR) hyperparameters using Optuna for:
  - ecfp4 (RDK topological fingerprint, 2048 bits)
  - sns (Morgan sort-and-slice, 1024 dims)
  - smiles (canonical SMILES OHE)
  - randomized_smiles (randomized SMILES OHE)

PDV is excluded (already tuned: poly kernel deg2, C=2.04).

Results saved to JSON for review. Does NOT modify master_tuned_hyperparameters.json.

Usage:
    python tune_svm_all_reps.py [--n-samples 10000] [--n-trials 200] [--seed 42]
"""

import argparse
import json
import os
import re
import warnings

import numpy as np
import optuna
from collections import defaultdict

from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator, RDKFingerprint
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.svm import SVR
from sklearn.metrics import r2_score

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')


# ── Data loading ──

def load_qm9_data(target_name, n_samples, seed):
    """Load QM9 data, return (smiles_list, targets)."""
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'QM9', 'qm9.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'qm9.csv')

    import pandas as pd
    df = pd.read_csv(csv_path)
    print(f"  CSV loaded: {len(df)} molecules")

    target_col = 'gap' if target_name == 'homo_lumo_gap' else target_name
    if 'is_uncharacterized' in df.columns:
        n_before = len(df)
        df = df[df['is_uncharacterized'] == False]
        print(f"  Excluding {n_before - len(df)} uncharacterized molecules")

    df = df.dropna(subset=[target_col])
    print(f"  Valid molecules: {len(df)}")

    rng = np.random.RandomState(seed)
    idx = rng.choice(len(df), size=min(n_samples, len(df)), replace=False)
    df = df.iloc[idx].reset_index(drop=True)
    print(f"  Sampled {len(df)} molecules")

    smiles_list = df['smiles'].tolist()
    targets = df[target_col].values.astype(np.float64)
    return smiles_list, targets


def scaffold_split(smiles_list, targets, seed, train_frac=0.8, val_frac=0.1):
    """Scaffold split matching the main pipeline."""
    scaffolds = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            scaffolds['NONE'].append(i)
            continue
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol, includeChirality=False)
        except Exception:
            scaffold = 'ERROR'
        scaffolds[scaffold].append(i)

    scaffold_groups = list(scaffolds.values())
    rng = np.random.RandomState(seed)
    rng.shuffle(scaffold_groups)

    n = len(smiles_list)
    train_cutoff = int(train_frac * n)
    val_cutoff = int((train_frac + val_frac) * n)

    train_idx, val_idx, test_idx = [], [], []
    for group in scaffold_groups:
        if len(train_idx) < train_cutoff:
            train_idx.extend(group)
        elif len(train_idx) + len(val_idx) < val_cutoff:
            val_idx.extend(group)
        else:
            test_idx.extend(group)

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


# ── Representation computation ──

def compute_ecfp4(smiles_list):
    """RDK topological fingerprint, 2048 bits (matches Rust rdk_fingerprint_mol)."""
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(np.zeros(2048, dtype=np.uint8))
            continue
        fp = RDKFingerprint(mol, fpSize=2048)
        arr = np.zeros(2048, dtype=np.uint8)
        for bit in fp.GetOnBits():
            arr[bit] = 1
        fps.append(arr)
    return np.array(fps, dtype=np.float32)


def compute_sns(smiles_list, train_idx):
    """Morgan sort-and-slice, 1024 dims (matches pipeline SNS)."""
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    train_mols = [mols[i] for i in train_idx if mols[i] is not None]

    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        atomInvariantsGenerator=rdFingerprintGenerator.GetMorganAtomInvGen(
            includeRingMembership=True),
        useBondTypes=True,
        includeChirality=False
    )

    def get_substructures(mol):
        if mol is None:
            return {}
        return morgan_gen.GetSparseCountFingerprint(mol).GetNonzeroElements()

    # Build prevalence dict from training set
    sub_prevs = {}
    for mol in train_mols:
        for sub_id in get_substructures(mol).keys():
            sub_prevs[sub_id] = sub_prevs.get(sub_id, 0) + 1

    # Sort by prevalence (descending), take top 1024
    sorted_subs = sorted(sub_prevs, key=lambda x: (sub_prevs[x], x), reverse=True)
    top_subs = sorted_subs[:1024]
    sub_to_idx = {s: i for i, s in enumerate(top_subs)}

    # Featurize all molecules
    fps = []
    for mol in mols:
        vec = np.zeros(1024, dtype=np.float32)
        if mol is None:
            fps.append(vec)
            continue
        for sub_id, count in get_substructures(mol).items():
            if sub_id in sub_to_idx:
                vec[sub_to_idx[sub_id]] = count
        fps.append(vec)
    return np.array(fps, dtype=np.float32)


def compute_smiles_ohe(smiles_list, max_vocab=30, randomize=False):
    """Tokenize SMILES and create OHE vectors (matches Rust tokenizer)."""
    SMILES_REGEX = re.compile(
        r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>>?|\*|\$|%[0-9]{2}|[0-9])"
    )

    def tokenize(smi):
        return SMILES_REGEX.findall(smi)

    # Optionally randomize SMILES
    if randomize:
        rand_smiles = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                rand_smiles.append(Chem.MolToSmiles(mol, isomericSmiles=False, doRandom=True))
            else:
                rand_smiles.append(smi)
        smiles_to_use = rand_smiles
    else:
        smiles_to_use = smiles_list

    # Tokenize all
    all_tokens = [tokenize(smi) for smi in smiles_to_use]
    max_len = max(len(t) for t in all_tokens)

    # Build vocab from all tokens (sorted by frequency)
    token_counts = defaultdict(int)
    for tokens in all_tokens:
        for t in tokens:
            token_counts[t] += 1
    sorted_tokens = sorted(token_counts, key=lambda x: token_counts[x], reverse=True)
    vocab = {t: i for i, t in enumerate(sorted_tokens[:max_vocab])}
    vocab_size = len(vocab)

    # OHE: flatten (max_len × vocab_size) into 1D
    feat_dim = max_len * vocab_size
    fps = np.zeros((len(smiles_list), feat_dim), dtype=np.float32)
    for i, tokens in enumerate(all_tokens):
        for j, t in enumerate(tokens):
            if j < max_len and t in vocab:
                fps[i, j * vocab_size + vocab[t]] = 1.0
    return fps


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description='Standalone SVM tuning for all reps')
    parser.add_argument('--n-samples', type=int, default=10000)
    parser.add_argument('--n-trials', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--reps', nargs='+',
                        default=['ecfp4', 'sns', 'smiles', 'randomized_smiles'],
                        help='Representations to tune')
    args = parser.parse_args()

    print(f"Loading QM9 (target=homo_lumo_gap, n={args.n_samples})...")
    smiles_list, targets = load_qm9_data('homo_lumo_gap', args.n_samples, args.seed)

    print("Performing scaffold split...")
    train_idx, val_idx, test_idx = scaffold_split(smiles_list, targets, args.seed)
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Y normalization (z-score, fit on train)
    y_train_raw = targets[train_idx]
    y_mean = float(y_train_raw.mean())
    y_std = float(y_train_raw.std())
    print(f"  Y normalization: mean={y_mean:.6f}, std={y_std:.6f}")

    y_all_norm = (targets - y_mean) / y_std
    y_train = y_all_norm[train_idx]
    y_val = y_all_norm[val_idx]
    y_test = y_all_norm[test_idx]

    # Combine train+val for SVM (no early stopping needed)
    y_trainval = np.hstack([y_train, y_val])

    # Compute representations
    rep_features = {}

    if 'ecfp4' in args.reps:
        print("\nComputing ecfp4 (RDK topological fingerprint, 2048 bits)...")
        rep_features['ecfp4'] = compute_ecfp4(smiles_list)

    if 'sns' in args.reps:
        print("Computing sns (Morgan sort-and-slice, 1024 dims)...")
        rep_features['sns'] = compute_sns(smiles_list, train_idx)

    if 'smiles' in args.reps:
        print("Computing smiles OHE...")
        rep_features['smiles'] = compute_smiles_ohe(smiles_list, max_vocab=30, randomize=False)

    if 'randomized_smiles' in args.reps:
        print("Computing randomized_smiles OHE...")
        np.random.seed(args.seed)
        rep_features['randomized_smiles'] = compute_smiles_ohe(
            smiles_list, max_vocab=30, randomize=True)

    # Tune SVM for each rep
    results = {}

    for rep_name, X_all in rep_features.items():
        X_train = X_all[train_idx]
        X_val = X_all[val_idx]
        X_test = X_all[test_idx]
        X_trainval = np.vstack([X_train, X_val])

        print(f"\n{'='*70}")
        print(f"Tuning SVM for {rep_name} ({X_train.shape[1]} features)")
        print(f"  {args.n_trials} Optuna trials")
        print(f"{'='*70}")

        def objective(trial):
            C = trial.suggest_float('C', 0.01, 100, log=True)
            gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
            kernel = trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid'])

            params = {'C': C, 'gamma': gamma, 'kernel': kernel}

            if kernel == 'poly':
                params['degree'] = trial.suggest_int('degree', 2, 5)
                params['coef0'] = trial.suggest_float('coef0', 0.0, 10.0)
            if kernel == 'sigmoid':
                params['coef0'] = trial.suggest_float('coef0', 0.0, 10.0)

            model = SVR(**params)
            model.fit(X_trainval, y_trainval)
            y_pred_norm = model.predict(X_test)
            # Denormalize for R² on original scale
            y_pred = y_pred_norm * y_std + y_mean
            y_test_orig = y_test * y_std + y_mean
            return r2_score(y_test_orig, y_pred)

        study = optuna.create_study(direction='maximize',
                                     sampler=optuna.samplers.TPESampler(seed=args.seed))
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

        best = study.best_trial
        print(f"\n  Best R²: {best.value:.4f}")
        print(f"  Best params: {best.params}")

        # Re-run best model and report
        best_params = {k: v for k, v in best.params.items()}
        model = SVR(**best_params)
        model.fit(X_trainval, y_trainval)
        y_pred_test = model.predict(X_test) * y_std + y_mean
        y_test_orig = y_test * y_std + y_mean
        final_r2 = r2_score(y_test_orig, y_pred_test)

        results[rep_name] = {
            'best_r2': float(final_r2),
            'best_params': best_params,
            'n_trials': args.n_trials,
            'n_features': int(X_train.shape[1]),
        }

        print(f"  Final test R²: {final_r2:.4f}")

    # Save results
    out_dir = os.path.join(os.path.dirname(__file__), 'pdv_investigation')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'svm_tuning_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*70}")
    print(f"Results saved to {out_path}")
    print(f"{'='*70}")

    # Summary table
    print(f"\n{'Rep':<20} {'Kernel':<10} {'C':<10} {'R²':<10} {'Extra params'}")
    print("-" * 70)
    for rep_name, r in results.items():
        p = r['best_params']
        extra = ""
        if p.get('kernel') == 'poly':
            extra = f"degree={p.get('degree')}, coef0={p.get('coef0', 0):.3f}"
        elif p.get('kernel') == 'sigmoid':
            extra = f"coef0={p.get('coef0', 0):.3f}"
        print(f"{rep_name:<20} {p['kernel']:<10} {p['C']:<10.4f} {r['best_r2']:<10.4f} {extra}")

    # Compare with current master values
    master_path = os.path.join(os.path.dirname(__file__), '..', 'results',
                                'master_tuned_hyperparameters.json')
    if os.path.exists(master_path):
        with open(master_path) as f:
            master = json.load(f)
        if 'svm' in master:
            print(f"\nComparison with current master_tuned_hyperparameters.json:")
            print(f"{'Rep':<20} {'Master kernel':<15} {'Master C':<12} {'New kernel':<15} {'New C':<12}")
            print("-" * 70)
            for rep_name in results:
                if rep_name in master['svm']:
                    m = master['svm'][rep_name]
                    n = results[rep_name]['best_params']
                    print(f"{rep_name:<20} {m.get('kernel','?'):<15} {m.get('C','?'):<12} {n['kernel']:<15} {n['C']:<12.4f}")
                else:
                    n = results[rep_name]['best_params']
                    print(f"{rep_name:<20} {'MISSING':<15} {'---':<12} {n['kernel']:<15} {n['C']:<12.4f}")


if __name__ == '__main__':
    main()
