#!/usr/bin/env python3
"""Generate LaTeX table content for supplementary tables S2-S5 from CSV data."""

import csv
from collections import defaultdict

DATA_DIR = "/Users/apunt/repos/qsar_qm_models/results/paper_figures/paper_figures"

MODEL_LABELS = {
    'dnn': r'NN-$\alpha$',
    'mlp': r'NN-$\beta$',
    'dnn_bnn_full': r'BNN-$\alpha$',
    'mlp_bnn_full': r'BNN-$\beta$',
    'dnn_vbll': r'VBLL-$\alpha$',
    'mlp_vbll': r'VBLL-$\beta$',
    'lgb': 'LightGBM',
    'xgboost': 'XGBoost',
    'rf': 'RF',
    'qrf': 'QRF',
    'svm': 'SVM',
    'ngboost': 'NGBoost',
    'gauche': 'Gauche',
    'gauche_rbf': 'Gauche (RBF)',
}

REP_LABELS = {
    'ecfp4': 'Topological',
    'continuous_pdv': 'PDV',
    'pdv': 'PDV (binary)',
    'mol2vec': 'Mol2Vec',
    'mhggnn': 'MHGGNN',
    'sns': 'SNS',
    'smiles': 'SMILES',
    'randomized_smiles': 'Rand.~SMILES',
    'morgan': 'Morgan',
}

def label_model(m):
    return MODEL_LABELS.get(m, m)

def label_rep(r):
    return REP_LABELS.get(r, r)


def gen_s2_excluded_configs():
    """S2: Model × Rep cross-tabulation of excluded strategy counts."""
    rows = []
    with open(f"{DATA_DIR}/excluded_configs.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Count strategies per model×rep
    counts = defaultdict(int)
    for row in rows:
        counts[(row['model'], row['rep'])] += 1

    # Get unique models and reps (ordered)
    models = sorted(set(r[0] for r in counts.keys()),
                    key=lambda m: list(MODEL_LABELS.keys()).index(m) if m in MODEL_LABELS else 99)
    reps = sorted(set(r[1] for r in counts.keys()),
                  key=lambda r: list(REP_LABELS.keys()).index(r) if r in REP_LABELS else 99)

    ncols = len(reps) + 1
    header = r"\begin{tabular}{l" + "c" * len(reps) + "}"
    lines = [header, r"\toprule"]
    lines.append(r"\textbf{Model} & " + " & ".join(rf"\textbf{{{label_rep(r)}}}" for r in reps) + r" \\")
    lines.append(r"\midrule")

    for m in models:
        cells = []
        for r in reps:
            c = counts.get((m, r), 0)
            cells.append(str(c) if c > 0 else "---")
        lines.append(f"{label_model(m)} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def gen_s3_model_redundancy():
    """S3: Model pair Spearman correlations (excluded pairs + top retained pairs)."""
    rows = []
    with open(f"{DATA_DIR}/table_supp_model_redundancy.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Filter out gauche_rbf pairs (artifacts with n=6)
    rows = [r for r in rows if r['model_a'] != 'gauche_rbf' and r['model_b'] != 'gauche_rbf']

    # Sort by rho descending
    rows.sort(key=lambda r: float(r['spearman_rho']), reverse=True)

    # Show pairs with rho >= 0.95
    rows = [r for r in rows if float(r['spearman_rho']) >= 0.95]

    lines = [r"\begin{tabular}{llccc}", r"\toprule"]
    lines.append(r"\textbf{Model A} & \textbf{Model B} & \textbf{$n$} & \textbf{$\rho$} & \textbf{Excluded} \\")
    lines.append(r"\midrule")

    for row in rows:
        ma = label_model(row['model_a'])
        mb = label_model(row['model_b'])
        n = row['n_shared_points']
        rho = float(row['spearman_rho'])
        excl = r"\checkmark" if row['excluded_from_anova'] == 'yes' else ""
        lines.append(f"{ma} & {mb} & {n} & {rho:.3f} & {excl}" + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def gen_s4_rep_redundancy():
    """S4: Representation pair Spearman correlations."""
    rows = []
    with open(f"{DATA_DIR}/table_supp_rep_redundancy.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Sort by rho descending
    rows.sort(key=lambda r: float(r['spearman_rho']), reverse=True)

    lines = [r"\begin{tabular}{llccc}", r"\toprule"]
    lines.append(r"\textbf{Rep.\ A} & \textbf{Rep.\ B} & \textbf{$n$} & \textbf{$\rho$} & \textbf{Excluded} \\")
    lines.append(r"\midrule")

    for row in rows:
        ra = label_rep(row['rep_a'])
        rb = label_rep(row['rep_b'])
        n = row['n_shared_points']
        rho = float(row['spearman_rho'])
        excl = r"\checkmark" if row['excluded_from_anova'] == 'yes' else ""
        lines.append(f"{ra} & {rb} & {n} & {rho:.3f} & {excl}" + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def gen_s5_icc():
    """S5: ICC(1,1) for model pairs with ICC >= 0.5."""
    rows = []
    with open(f"{DATA_DIR}/table_supp_icc.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Sort by ICC descending
    rows.sort(key=lambda r: float(r['icc_1_1']), reverse=True)

    # Filter to ICC >= 0.5
    rows = [r for r in rows if float(r['icc_1_1']) >= 0.5]

    lines = [r"\begin{tabular}{llcccc}", r"\toprule"]
    lines.append(r"\textbf{Model A} & \textbf{Model B} & \textbf{$n$} & \textbf{ICC(1,1)} & \textbf{Mean $|\Delta$NDS$|$} & \textbf{Family} \\")
    lines.append(r"\midrule")

    # Determine family groupings
    def get_family(ma, mb):
        nn_alpha = {'dnn', 'dnn_bnn_full', 'dnn_vbll'}
        nn_beta = {'mlp', 'mlp_bnn_full', 'mlp_vbll'}
        trees = {'rf', 'qrf', 'lgb', 'xgboost', 'ngboost'}
        if ma in nn_alpha and mb in nn_alpha:
            return r"$\alpha$"
        if ma in nn_beta and mb in nn_beta:
            return r"$\beta$"
        if ma in trees and mb in trees:
            return "Tree"
        if (ma in nn_alpha and mb in nn_beta) or (ma in nn_beta and mb in nn_alpha):
            return r"$\alpha$/$\beta$"
        return "Cross"

    for row in rows:
        ma_raw, mb_raw = row['model_a'], row['model_b']
        ma = label_model(ma_raw)
        mb = label_model(mb_raw)
        n = row['n_shared_reps']
        icc = float(row['icc_1_1'])
        mad = float(row['mean_abs_nds_diff'])
        fam = get_family(ma_raw, mb_raw)
        lines.append(f"{ma} & {mb} & {n} & {icc:.3f} & {mad:.4f} & {fam}" + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 80)
    print("TABLE S2: Excluded Configurations")
    print("=" * 80)
    print(gen_s2_excluded_configs())
    print()
    print("=" * 80)
    print("TABLE S3: Model Redundancy (rho >= 0.95)")
    print("=" * 80)
    print(gen_s3_model_redundancy())
    print()
    print("=" * 80)
    print("TABLE S4: Representation Redundancy")
    print("=" * 80)
    print(gen_s4_rep_redundancy())
    print()
    print("=" * 80)
    print("TABLE S5: ICC (>= 0.5)")
    print("=" * 80)
    print(gen_s5_icc())
