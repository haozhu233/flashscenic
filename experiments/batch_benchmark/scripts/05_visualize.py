"""
Step 5: Visualization.

Produces four figures:
  Figure 1: UMAP panels — 2 rows (batch / cell_type) × n_methods columns
  Figure 2: scIB metric heatmap across methods
  Figure 3: ML predictor bar chart (sex AUROC + cell type AUROC)
  Figure 4: Sex AUROC vs iLISI scatter (batch mixing vs bio preservation)

Output: results/figures/*.pdf and *.png

Usage:
    python 05_visualize.py [--dataset immune_human|pancreas|all]
"""

import argparse
import sys
import warnings
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, str(Path(__file__).parent))
from config import DATASETS, METHODS, DATA_DIR, METRICS_DIR, FIGURES_DIR, GLOBAL_SEED

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

# Consistent colors and display names across all figures
METHOD_COLORS = {
    "raw_pca":     "#888888",
    "harmony":     "#4878CF",
    "scvi":        "#E84646",
    "flashscenic": "#27AE60",
}
METHOD_LABELS = {
    "raw_pca":     "Raw PCA",
    "harmony":     "Harmony",
    "scvi":        "scVI",
    "flashscenic": "flashSCENIC",
}
EMBEDDING_KEYS = {
    "raw_pca":     "X_raw_pca",
    "harmony":     "X_harmony",
    "scvi":        "X_scvi",
    "flashscenic": "X_flashscenic",
}


# ---------------------------------------------------------------------------
# UMAP computation
# ---------------------------------------------------------------------------

def compute_umap(adata: ad.AnnData, method: str) -> np.ndarray:
    key = EMBEDDING_KEYS[method]
    umap_key = f"X_umap_{method}"

    if umap_key in adata.obsm:
        return adata.obsm[umap_key]

    tmp = ad.AnnData(X=adata.obsm[key])
    sc.pp.neighbors(tmp, use_rep="X", n_neighbors=15, random_state=GLOBAL_SEED)
    sc.tl.umap(tmp, random_state=GLOBAL_SEED)
    umap = tmp.obsm["X_umap"]
    adata.obsm[umap_key] = umap
    return umap


# ---------------------------------------------------------------------------
# Figure 1: UMAP panels
# ---------------------------------------------------------------------------

def fig_umaps(adata: ad.AnnData, cfg: dict, name: str) -> None:
    available = [m for m in METHODS if EMBEDDING_KEYS[m] in adata.obsm]
    if not available:
        print(f"  [skip] no embeddings found for UMAP")
        return

    batch_key = cfg["batch_key"]
    ct_key = cfg["cell_type_key"]
    n_cols = len(available)

    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    batch_categories = adata.obs[batch_key].astype("category").cat.categories
    ct_categories = adata.obs[ct_key].astype("category").cat.categories

    batch_palette = plt.cm.tab20(np.linspace(0, 1, len(batch_categories)))
    ct_palette = plt.cm.tab20b(np.linspace(0, 1, len(ct_categories)))

    for col, method in enumerate(available):
        print(f"    Computing UMAP for {method} ...", end=" ", flush=True)
        umap = compute_umap(adata, method)
        print("done")

        batch_codes = pd.Categorical(adata.obs[batch_key]).codes
        ct_codes = pd.Categorical(adata.obs[ct_key]).codes

        # Row 0: colored by batch
        ax = axes[0, col]
        for i, b in enumerate(batch_categories):
            mask = adata.obs[batch_key] == b
            ax.scatter(umap[mask, 0], umap[mask, 1],
                       c=[batch_palette[i]], s=1, alpha=0.4, linewidths=0,
                       label=str(b))
        ax.set_title(METHOD_LABELS[method], fontsize=11, fontweight="bold",
                     color=METHOD_COLORS[method])
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("Colored by batch", fontsize=9)

        # Row 1: colored by cell type
        ax = axes[1, col]
        for i, ct in enumerate(ct_categories):
            mask = adata.obs[ct_key] == ct
            ax.scatter(umap[mask, 0], umap[mask, 1],
                       c=[ct_palette[i]], s=1, alpha=0.4, linewidths=0,
                       label=str(ct))
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("Colored by cell type", fontsize=9)

    # Add per-row legends to the right of the last column
    handles_batch = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=batch_palette[i], markersize=6, label=str(b))
        for i, b in enumerate(batch_categories)
    ]
    handles_ct = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=ct_palette[i], markersize=6, label=str(ct))
        for i, ct in enumerate(ct_categories)
    ]

    axes[0, n_cols - 1].legend(
        handles=handles_batch, title="Batch",
        loc="upper left", bbox_to_anchor=(1.01, 1.0),
        fontsize=7, title_fontsize=8, frameon=True, ncol=1,
        borderaxespad=0,
    )
    axes[1, n_cols - 1].legend(
        handles=handles_ct, title="Cell type",
        loc="upper left", bbox_to_anchor=(1.01, 1.0),
        fontsize=7, title_fontsize=8, frameon=True, ncol=1,
        borderaxespad=0,
    )

    fig.suptitle(f"UMAP — {name}", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    for ext in ["pdf", "png"]:
        out = FIGURES_DIR / f"fig1_umap_{name}.{ext}"
        plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"    Saved fig1_umap_{name}")

    # Save updated adata with UMAP coords
    out_path = DATA_DIR / f"preprocessed_{name}.h5ad"
    adata.write_h5ad(out_path)


# ---------------------------------------------------------------------------
# Figure 2: scIB metric heatmap
# ---------------------------------------------------------------------------

def fig_scib_heatmap(name: str) -> None:
    csv_path = METRICS_DIR / f"scib_scores_{name}.csv"
    if not csv_path.exists():
        print(f"  [skip] {csv_path.name} not found")
        return

    df = pd.read_csv(csv_path).set_index("method")
    metric_cols = ["iLISI", "ASW_batch", "cLISI", "ASW_cell_type",
                   "NMI", "ARI", "batch_score", "bio_score", "overall_score"]
    metric_cols = [c for c in metric_cols if c in df.columns]
    df = df[metric_cols]

    # Rename rows to display labels
    df.index = [METHOD_LABELS.get(m, m) for m in df.index]

    fig, ax = plt.subplots(figsize=(len(metric_cols) * 1.2, len(df) * 0.9 + 1))
    im = ax.imshow(df.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(metric_cols)))
    ax.set_xticklabels(metric_cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.index, fontsize=10)

    for i in range(len(df)):
        for j in range(len(metric_cols)):
            val = df.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if 0.3 < val < 0.7 else "white")

    # Vertical separators between batch / bio / composite
    batch_cols = ["iLISI", "ASW_batch"]
    n_batch = len([c for c in metric_cols if c in batch_cols])
    bio_end = n_batch + len([c for c in metric_cols
                              if c in ["cLISI", "ASW_cell_type", "NMI", "ARI"]])
    ax.axvline(n_batch - 0.5, color="white", lw=2)
    ax.axvline(bio_end - 0.5, color="white", lw=2)

    ax.set_title(f"scIB Metrics — {name}", fontsize=12, fontweight="bold", pad=10)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="Score (higher = better)")
    plt.tight_layout()

    for ext in ["pdf", "png"]:
        out = FIGURES_DIR / f"fig2_metrics_{name}.{ext}"
        plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"    Saved fig2_metrics_{name}")


# ---------------------------------------------------------------------------
# Figure 3: ML predictor bar chart
# ---------------------------------------------------------------------------

def fig_ml_bars(name: str) -> None:
    csv_path = METRICS_DIR / f"ml_summary_{name}.csv"
    if not csv_path.exists():
        print(f"  [skip] {csv_path.name} not found")
        return

    df = pd.read_csv(csv_path)
    available_methods = [m for m in METHODS if m in df["method"].values]

    has_sex = "sex_AUROC_mean" in df.columns and df["sex_AUROC_mean"].notna().any()
    has_ct = "celltype_AUROC_mean" in df.columns and df["celltype_AUROC_mean"].notna().any()

    n_panels = int(has_sex) + int(has_ct)
    if n_panels == 0:
        print(f"  [skip] no ML results to plot")
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), sharey=False)
    if n_panels == 1:
        axes = [axes]

    panel = 0

    if has_sex:
        ax = axes[panel]
        panel += 1
        methods = [m for m in available_methods
                   if not pd.isna(df.loc[df.method == m, "sex_AUROC_mean"].values[0])]
        y = [df.loc[df.method == m, "sex_AUROC_mean"].values[0] for m in methods]
        yerr = [df.loc[df.method == m, "sex_AUROC_std"].values[0] for m in methods]
        colors = [METHOD_COLORS[m] for m in methods]
        labels = [METHOD_LABELS[m] for m in methods]

        bars = ax.bar(range(len(methods)), y, yerr=yerr, color=colors,
                      capsize=5, edgecolor="white", linewidth=0.5)
        ax.axhline(0.5, color="gray", linestyle="--", lw=1, label="Random (0.5)")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
        ax.set_ylabel("AUROC (mean ± std across held-out batches)", fontsize=9)
        ax.set_title("Sex Prediction AUROC\n(key test: biological signal preservation)",
                     fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.axhline(1.0, color="black", lw=0.5, alpha=0.3)

        # Annotate flashscenic bar
        if "flashscenic" in methods:
            idx = methods.index("flashscenic")
            ax.text(idx, y[idx] + (yerr[idx] or 0) + 0.02, "★",
                    ha="center", va="bottom", fontsize=14,
                    color=METHOD_COLORS["flashscenic"])

    if has_ct:
        ax = axes[panel]
        methods = [m for m in available_methods
                   if not pd.isna(df.loc[df.method == m, "celltype_AUROC_mean"].values[0])]
        y = [df.loc[df.method == m, "celltype_AUROC_mean"].values[0] for m in methods]
        yerr = [df.loc[df.method == m, "celltype_AUROC_std"].values[0] for m in methods]
        colors = [METHOD_COLORS[m] for m in methods]
        labels = [METHOD_LABELS[m] for m in methods]

        ax.bar(range(len(methods)), y, yerr=yerr, color=colors,
               capsize=5, edgecolor="white", linewidth=0.5)
        ax.axhline(0.5, color="gray", linestyle="--", lw=1, label="Random (0.5)")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
        ax.set_ylabel("Macro AUROC (mean ± std across held-out batches)", fontsize=9)
        ax.set_title("Cell Type Prediction AUROC\n(positive control: all methods should score high)",
                     fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.05)

    fig.suptitle(f"ML Biological Signal Preservation — {name}",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()

    for ext in ["pdf", "png"]:
        out = FIGURES_DIR / f"fig3_ml_bars_{name}.{ext}"
        plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"    Saved fig3_ml_bars_{name}")


# ---------------------------------------------------------------------------
# Figure 4: Sex AUROC vs iLISI scatter
# ---------------------------------------------------------------------------

def fig_scatter(name: str) -> None:
    scib_path = METRICS_DIR / f"scib_scores_{name}.csv"
    ml_path = METRICS_DIR / f"ml_summary_{name}.csv"

    if not scib_path.exists() or not ml_path.exists():
        print(f"  [skip] scatter plot: missing metrics files")
        return

    scib_df = pd.read_csv(scib_path).set_index("method")
    ml_df = pd.read_csv(ml_path).set_index("method")

    if "sex_AUROC_mean" not in ml_df.columns:
        print(f"  [skip] scatter plot: no sex AUROC data")
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    for method in METHODS:
        if method not in scib_df.index or method not in ml_df.index:
            continue
        if "iLISI" not in scib_df.columns:
            continue

        x = scib_df.loc[method, "iLISI"]
        y = ml_df.loc[method, "sex_AUROC_mean"]
        yerr = ml_df.loc[method, "sex_AUROC_std"]

        if pd.isna(x) or pd.isna(y):
            continue

        ax.errorbar(x, y, yerr=yerr if not pd.isna(yerr) else 0,
                    fmt="o", markersize=12, capsize=4,
                    color=METHOD_COLORS[method],
                    label=METHOD_LABELS[method])
        ax.annotate(METHOD_LABELS[method], (x, y),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=9, color=METHOD_COLORS[method])

    # Ideal quadrant annotation
    ax.axhline(0.5, color="gray", linestyle="--", lw=1, alpha=0.6)
    ax.text(ax.get_xlim()[1] * 0.95, 0.97,
            "Ideal:\nhigh batch mixing\n+ high sex AUROC",
            ha="right", va="top", fontsize=8, color="green",
            alpha=0.6, transform=ax.transAxes)

    ax.set_xlabel("iLISI (batch mixing — higher = better)", fontsize=10)
    ax.set_ylabel("Sex Prediction AUROC (biological signal — higher = better)", fontsize=10)
    ax.set_title(f"Batch Mixing vs. Biological Signal Preservation\n{name}",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, frameon=True)
    plt.tight_layout()

    for ext in ["pdf", "png"]:
        out = FIGURES_DIR / f"fig4_scatter_{name}.{ext}"
        plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"    Saved fig4_scatter_{name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all",
                        choices=list(DATASETS.keys()) + ["all"])
    args = parser.parse_args()

    to_process = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    for name in to_process:
        print(f"\n{'='*60}")
        print(f"Visualization: {name}")
        print(f"{'='*60}")

        preprocessed_path = DATA_DIR / f"preprocessed_{name}.h5ad"
        if not preprocessed_path.exists():
            print(f"  [skip] {preprocessed_path.name} not found")
            continue

        adata = ad.read_h5ad(preprocessed_path)

        print(f"\n  Figure 1: UMAP panels")
        fig_umaps(adata, DATASETS[name], name)

        print(f"\n  Figure 2: scIB heatmap")
        fig_scib_heatmap(name)

        print(f"\n  Figure 3: ML predictor bars")
        fig_ml_bars(name)

        print(f"\n  Figure 4: Sex AUROC vs iLISI scatter")
        fig_scatter(name)

    print(f"\nAll figures saved to {FIGURES_DIR}")
    print("Run 06_summarize.py to generate the final report.")


if __name__ == "__main__":
    main()
