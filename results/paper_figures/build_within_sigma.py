import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

SCRATCH = "/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/37046d5b-871b-4e27-82fb-a35e1ea2ef1d/scratchpad"
CSV = "/Users/apunt/repos/qsar_qm_models/results/paper_figures/table4_supp_uncertainty_by_strategy_rep.csv"

df = pd.read_csv(CSV)
VAL = "Unc-Noise ρ σ=0.6"  # within-slice Spearman at sigma=0.6

STRATS = ["Gaussian", "Threshold", "Heteroscedastic", "Value-Prop.", "Quantile", "Outlier"]
MODELS = ["GP", "NGBoost", "QRF", "BNN-α", "BNN-β", "VBLL-α", "VBLL-β"]
REPS = ["continuous_pdv", "pdv", "ecfp4", "sns", "morgan", "smiles", "mol2vec", "mhggnn"]
REP_LABELS = {
    "continuous_pdv": "cont. PDV", "pdv": "PDV", "ecfp4": "Topological",
    "sns": "SNS", "morgan": "Morgan", "smiles": "SMILES",
    "mol2vec": "Mol2vec", "mhggnn": "MHG-GNN",
}

def cell(sub):
    # average duplicates if any
    if len(sub) == 0:
        return np.nan
    v = sub[VAL].dropna()
    return v.mean() if len(v) else np.nan

# Panel A: rep=continuous_pdv, rows=models, cols=strategies
A = np.full((len(MODELS), len(STRATS)), np.nan)
for i, m in enumerate(MODELS):
    for j, s in enumerate(STRATS):
        A[i, j] = cell(df[(df.Rep == "continuous_pdv") & (df.Model == m) & (df.Strategy == s)])

# Panel B: strategy=Outlier, rows=models, cols=reps
B = np.full((len(MODELS), len(REPS)), np.nan)
for i, m in enumerate(MODELS):
    for j, r in enumerate(REPS):
        B[i, j] = cell(df[(df.Strategy == "Outlier") & (df.Model == m) & (df.Rep == r)])

# Save CSVs
pd.DataFrame(A, index=MODELS, columns=STRATS).to_csv(os.path.join(SCRATCH, "within_sigma_panelA_continuous_pdv_by_strategy.csv"))
pd.DataFrame(B, index=MODELS, columns=[REP_LABELS[r] for r in REPS]).to_csv(os.path.join(SCRATCH, "within_sigma_panelB_outlier_by_rep.csv"))

# Plot
plt.rcParams.update({"font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [len(STRATS), len(REPS)]})
cmap = plt.get_cmap("RdBu_r").copy()
cmap.set_bad(color="#bdbdbd")
norm = TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=0.5)

def draw(ax, M, cols, title):
    Mm = np.ma.masked_invalid(M)
    im = ax.imshow(Mm, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=40, ha="right")
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                ax.text(j, i, "n/a", ha="center", va="center", color="#555555", fontsize=7)
            else:
                col = "white" if abs(v) > 0.32 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color=col, fontsize=8)
    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(MODELS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    return im

im = draw(axes[0], A, STRATS,
          "A. Noise-type dependence\n(rep = continuous PDV, within-σ ρ at σ=0.6)")
draw(axes[1], B, [REP_LABELS[r] for r in REPS],
     "B. Representation gate\n(strategy = Outlier, within-σ ρ at σ=0.6)")

cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
cbar.set_label("within-σ Spearman ρ (uncertainty vs injected noise)", fontsize=9)
fig.suptitle("Per-sample noise detection is noise-type dependent and representation-gated",
             fontsize=13, fontweight="bold", y=1.02)

out = os.path.join(SCRATCH, "within_sigma_uncertainty.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
print("SAVED:", out)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
print("\n=== Panel A: continuous_pdv, models x strategies (within-sigma rho @ sigma=0.6) ===")
print(pd.DataFrame(A, index=MODELS, columns=STRATS).round(3).to_string())
print("\n=== Panel B: Outlier strategy, models x reps (within-sigma rho @ sigma=0.6) ===")
print(pd.DataFrame(B, index=MODELS, columns=[REP_LABELS[r] for r in REPS]).round(3).to_string())
print("\nRendered without error.")
