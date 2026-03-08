"""
Step 5b: Regulon Specificity Scores (RSS) from binary AUCell activity.

Binarizes flashSCENIC AUCell scores per regulon (GMM threshold, SCENIC protocol),
computes RSS to identify cell-type-specific TF regulons, and plots a heatmap of
the top regulons per cell type.

Outputs:
  results/metrics/{dataset}_rss.csv       — full RSS matrix (cell_types × regulons)
  results/figures/fig5_rss_{dataset}.pdf
  results/figures/fig5_rss_{dataset}.png

Usage:
    python 05b_rss.py [--dataset immune_human|pancreas|ad_neurons|all]
"""

import argparse
import sys
import warnings
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.parent.parent  # flashscenic repo root
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from config import DATASETS, DATA_DIR, EMBEDDINGS_DIR, METRICS_DIR, FIGURES_DIR

from flashscenic.binarize_aucell import binarize_auc_matrix
from flashscenic.rss import regulon_specificity_scores

warnings.filterwarnings("ignore")

N_TOP = 5          # top regulons to show per cell type in the heatmap
RSS_ANNOT_THR = 0.3  # annotate cells with RSS value if >= this threshold


# ---------------------------------------------------------------------------
# Per-dataset RSS computation
# ---------------------------------------------------------------------------

def compute_rss(name: str, cfg: dict) -> dict | None:
    auc_path = EMBEDDINGS_DIR / f"{name}_flashscenic.npy"
    reg_path = EMBEDDINGS_DIR / f"{name}_flashscenic_regulon_names.npy"
    h5ad_path = DATA_DIR / f"preprocessed_{name}.h5ad"

    if not auc_path.exists() or not reg_path.exists():
        print(f"  [skip] flashscenic embeddings not found for {name}")
        return None
    if not h5ad_path.exists():
        print(f"  [skip] preprocessed h5ad not found for {name}")
        return None

    auc_scores = np.load(auc_path)
    regulon_names = np.load(reg_path, allow_pickle=True).tolist()
    print(f"  AUCell scores: {auc_scores.shape}  ({len(regulon_names)} regulons)")

    adata = ad.read_h5ad(h5ad_path)
    ct_key = cfg["cell_type_key"]
    cell_type_labels = adata.obs[ct_key].astype(str).values
    n_types = len(set(cell_type_labels))
    print(f"  Cell types ({ct_key}): {n_types}")

    print(f"  Binarizing AUCell scores (GMM) ...")
    binary = binarize_auc_matrix(auc_scores)
    active_frac = binary.mean()
    print(f"  Active fraction after binarization: {active_frac:.3f}")

    print(f"  Computing RSS ...")
    rss_result = regulon_specificity_scores(
        binary.astype(np.float64),
        cell_type_labels,
        regulon_names=regulon_names,
    )

    return rss_result


# ---------------------------------------------------------------------------
# Figure 5: RSS heatmap — top N regulons per cell type
# ---------------------------------------------------------------------------

def fig_rss_heatmap(rss_result: dict, name: str) -> None:
    rss_matrix = rss_result["rss"]          # (n_cell_types, n_regulons)
    cell_types = rss_result["cell_types"]
    regulon_names = rss_result["regulon_names"]

    n_types, n_regulons = rss_matrix.shape

    # Select top N_TOP regulons per cell type (by RSS), deduplicated
    selected_cols = []
    for ridx in range(n_types):
        row = rss_matrix[ridx, :]
        top_idx = np.argsort(row)[::-1][:N_TOP]
        for idx in top_idx:
            if idx not in selected_cols:
                selected_cols.append(idx)

    selected_cols = list(selected_cols)
    sub_matrix = rss_matrix[:, selected_cols]
    sub_names = [regulon_names[i] for i in selected_cols]

    n_shown = len(selected_cols)
    fig_w = max(10, n_shown * 0.45)
    fig_h = max(5, n_types * 0.38 + 1.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(sub_matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

    ax.set_xticks(range(n_shown))
    ax.set_xticklabels(sub_names, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(n_types))
    ax.set_yticklabels(cell_types, fontsize=8)

    # Annotate cells with RSS value where >= threshold
    for ridx in range(n_types):
        for cidx in range(n_shown):
            val = sub_matrix[ridx, cidx]
            if val >= RSS_ANNOT_THR:
                ax.text(cidx, ridx, f"{val:.2f}",
                        ha="center", va="center", fontsize=5.5,
                        color="black" if val < 0.7 else "white")

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="RSS (0–1)")
    ax.set_title(
        f"{name} — Regulon Specificity Scores\n"
        f"(binary AUCell, top {N_TOP} regulons per cell type)",
        fontsize=11, fontweight="bold", pad=10,
    )

    plt.tight_layout()

    for ext in ["pdf", "png"]:
        out = FIGURES_DIR / f"fig5_rss_{name}.{ext}"
        plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved fig5_rss_{name}  ({n_shown} regulons × {n_types} cell types)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute and plot RSS from binary AUCell scores")
    parser.add_argument("--dataset", default="all",
                        choices=list(DATASETS.keys()) + ["all"])
    args = parser.parse_args()

    to_process = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    for name in to_process:
        print(f"\n{'='*60}")
        print(f"RSS: {name}")
        print(f"{'='*60}")

        rss_result = compute_rss(name, DATASETS[name])
        if rss_result is None:
            continue

        # Save CSV
        csv_path = METRICS_DIR / f"{name}_rss.csv"
        df = pd.DataFrame(
            rss_result["rss"],
            index=rss_result["cell_types"],
            columns=rss_result["regulon_names"],
        )
        df.index.name = "cell_type"
        df.to_csv(csv_path)
        print(f"  Saved RSS matrix → {csv_path.name}  ({df.shape})")

        # Plot
        fig_rss_heatmap(rss_result, name)

    print(f"\nRSS done. Figures in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
