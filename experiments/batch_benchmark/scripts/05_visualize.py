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
from config import DATASETS, METHODS, DATA_DIR, EMBEDDINGS_DIR, METRICS_DIR, FIGURES_DIR, GLOBAL_SEED

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

# Consistent colors and display names across all figures
METHOD_COLORS = {
    "raw_pca":              "#888888",
    "harmony":              "#4878CF",
    "scvi":                 "#E84646",
    "flashscenic":          "#4878CF",
    "flashscenic_metacell": "#1a3a6b",   # darker blue for metacell variant
}
METHOD_LABELS = {
    "raw_pca":              "Raw PCA",
    "harmony":              "Harmony",
    "scvi":                 "scVI",
    "flashscenic":          "TF Activity",
    "flashscenic_metacell": "TF Activity\n(metacell)",
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

def compute_umap(adata: ad.AnnData, method: str, name: str) -> np.ndarray:
    """Compute or load cached UMAP for a method.

    Coordinates are cached as .npy in EMBEDDINGS_DIR so the preprocessed
    h5ad is never overwritten (overwriting large h5ad files risks corruption).
    """
    key = EMBEDDING_KEYS[method]
    cache_path = EMBEDDINGS_DIR / f"{name}_umap_{method}.npy"

    if cache_path.exists():
        return np.load(cache_path)

    tmp = ad.AnnData(X=adata.obsm[key])
    sc.pp.neighbors(tmp, use_rep="X", n_neighbors=15, random_state=GLOBAL_SEED)
    sc.tl.umap(tmp, random_state=GLOBAL_SEED)
    umap = tmp.obsm["X_umap"]
    np.save(cache_path, umap)
    return umap


# ---------------------------------------------------------------------------
# Figure 1: UMAP panels
# ---------------------------------------------------------------------------

def fig_umaps(adata: ad.AnnData, cfg: dict, name: str) -> None:
    available = [m for m in METHODS if EMBEDDING_KEYS[m] in adata.obsm]
    if not available:
        print(f"  [skip] no embeddings found for UMAP")
        return

    batch_key   = cfg["batch_key"]
    ct_key      = cfg["cell_type_key"]
    disease_key = cfg.get("disease_key")
    n_cols = len(available)

    fig, axes = plt.subplots(3, n_cols, figsize=(4 * n_cols, 3.5 * 3))
    if n_cols == 1:
        axes = axes.reshape(3, 1)

    batch_categories = adata.obs[batch_key].astype("category").cat.categories
    ct_categories = adata.obs[ct_key].astype("category").cat.categories

    batch_palette = plt.cm.tab20(np.linspace(0, 1, len(batch_categories)))
    ct_palette = plt.cm.tab20b(np.linspace(0, 1, len(ct_categories)))

    for col, method in enumerate(available):
        print(f"    Computing UMAP for {method} ...", end=" ", flush=True)
        umap = compute_umap(adata, method, name)
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
        umap_label = "Inferred TF Activity" if method == "flashscenic" else METHOD_LABELS[method]
        ax.set_title(umap_label, fontsize=11, fontweight="bold",
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

        # Row 2: colored by disease status
        if disease_key and disease_key in adata.obs.columns:
            ax = axes[2, col]
            disease_labels = adata.obs[disease_key].astype(str).values
            unique_labels = np.unique(disease_labels)
            dis_color_map = {
                lbl: ("#4878CF" if lbl.lower() in ("normal", "healthy", "control")
                      else "#C44E52")
                for lbl in unique_labels
            }
            for lbl in unique_labels:
                mask = disease_labels == lbl
                ax.scatter(umap[mask, 0], umap[mask, 1],
                           c=[dis_color_map[lbl]], s=1, alpha=0.4, linewidths=0,
                           label=lbl)
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel("Colored by disease", fontsize=9)

    # Add per-row legends to the left of the first column
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

    axes[0, 0].legend(
        handles=handles_batch, title="Batch",
        loc="upper right", bbox_to_anchor=(-0.05, 1.0),
        fontsize=7, title_fontsize=8, frameon=True, ncol=1,
        borderaxespad=0,
    )
    axes[1, 0].legend(
        handles=handles_ct, title="Cell type",
        loc="upper right", bbox_to_anchor=(-0.05, 1.0),
        fontsize=7, title_fontsize=8, frameon=True, ncol=1,
        borderaxespad=0,
    )

    if disease_key and disease_key in adata.obs.columns:
        unique_labels = np.unique(adata.obs[disease_key].astype(str).values)
        dis_color_map = {
            lbl: ("#4878CF" if lbl.lower() in ("normal", "healthy", "control")
                  else "#C44E52")
            for lbl in unique_labels
        }
        handles_dis = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=dis_color_map[lbl], markersize=6, label=lbl)
            for lbl in unique_labels
        ]
        axes[2, 0].legend(
            handles=handles_dis, title="Disease",
            loc="upper right", bbox_to_anchor=(-0.05, 1.0),
            fontsize=7, title_fontsize=8, frameon=True, ncol=1,
            borderaxespad=0,
        )

    fig.suptitle("UMAP", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()

    fig_dir = FIGURES_DIR / name
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        out = fig_dir / f"fig1_umap_{name}.{ext}"
        plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"    Saved fig1_umap_{name}")


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

    ax.set_title("scIB Metrics", fontsize=12, fontweight="bold", pad=10)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="Score (higher = better)")
    plt.tight_layout()

    fig_dir = FIGURES_DIR / name
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        out = fig_dir / f"fig2_metrics_{name}.{ext}"
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

    # Load metacell results per task if available (adds flashscenic_metacell bar)
    meta_rows: dict[str, pd.Series] = {}
    for _task in ("sex", "disease"):
        _p = METRICS_DIR / f"ml_{_task}_{name}_metacells.csv"
        if _p.exists():
            _mdf = pd.read_csv(_p)
            _fs = _mdf[_mdf["method"] == "flashscenic"]
            if not _fs.empty:
                meta_rows[_task] = _fs.iloc[0]

    has_sex     = "sex_test_AUROC"     in df.columns and df["sex_test_AUROC"].notna().any()
    has_disease = "disease_test_AUROC" in df.columns and df["disease_test_AUROC"].notna().any()

    panels = []
    if has_sex:
        panels.append(("sex_test_AUROC",     "Sex Prediction AUROC (6-fold CV)"))
    if has_disease:
        panels.append(("disease_test_AUROC", "Disease Prediction AUROC (6-fold CV)"))

    if not panels:
        print(f"  [skip] no ML AUROC results to plot")
        return

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(2 * n_panels, 9), sharey=False)
    if n_panels == 1:
        axes = [axes]

    for ax, (col, title) in zip(axes, panels):
        std_col = col + "_std"
        methods = [m for m in available_methods
                   if m in df["method"].values
                   and not pd.isna(df.loc[df.method == m, col].values[0])]
        y      = [df.loc[df.method == m, col].values[0] for m in methods]
        yerr   = [df.loc[df.method == m, std_col].values[0]
                  if std_col in df.columns else 0.0 for m in methods]

        # Append metacell flashscenic bar if available for this task
        task_key = col.replace("_test_AUROC", "")   # "sex" or "disease"
        meta_row = meta_rows.get(task_key)
        if meta_row is not None:
            methods = methods + ["flashscenic_metacell"]
            y      = y      + [float(meta_row.get("test_AUROC", np.nan))]
            yerr   = yerr   + [float(meta_row.get("test_AUROC_std", 0.0))]

        colors = [METHOD_COLORS.get(m, "grey") for m in methods]
        labels = [METHOD_LABELS.get(m, m)      for m in methods]

        ax.bar(range(len(methods)), y, color=colors, edgecolor="white", linewidth=0.5,
               yerr=yerr, error_kw={"ecolor": "black", "capsize": 4, "lw": 1.2})
        ax.axhline(0.5, color="gray", linestyle="--", lw=1, label="Random (0.5)")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
        ax.set_ylabel("mean AUROC ± std (6 folds)", fontsize=9)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.axhline(1.0, color="black", lw=0.5, alpha=0.3)

        # Donor-level AUROC overlay (diamond markers)
        donor_col = col.replace("test_AUROC", "test_donor_AUROC")
        if donor_col in df.columns:
            donor_legend_added = False
            for xi, m in enumerate(methods):
                if m == "flashscenic_metacell":
                    val = float(meta_row.get("test_donor_AUROC", np.nan)) \
                          if meta_row is not None else np.nan
                    if not np.isnan(val):
                        ax.scatter(xi, val, marker="D", s=55,
                                   color="white", edgecolors=colors[xi],
                                   linewidths=1.5, zorder=5,
                                   label="Donor AUROC (◆)" if not donor_legend_added else "")
                        donor_legend_added = True
                    continue
                row = df.loc[df["method"] == m, donor_col]
                if len(row) and not pd.isna(row.values[0]):
                    ax.scatter(xi, row.values[0], marker="D", s=55,
                               color="white", edgecolors=colors[xi],
                               linewidths=1.5, zorder=5,
                               label="Donor AUROC (◆)" if not donor_legend_added else "")
                    donor_legend_added = True
            if donor_legend_added:
                ax.legend(fontsize=8, loc="lower right")

        if "flashscenic" in methods:
            idx = methods.index("flashscenic")
            ax.text(idx, y[idx] + max(yerr[idx], 0.02) + 0.02, "★",
                    ha="center", va="bottom",
                    fontsize=14, color=METHOD_COLORS["flashscenic"])

    fig.suptitle("ML Biological Signal Preservation",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()

    fig_dir = FIGURES_DIR / name
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        out = fig_dir / f"fig3_ml_bars_{name}.{ext}"
        plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"    Saved fig3_ml_bars_{name}")


# ---------------------------------------------------------------------------
# Figure 5: flashSCENIC coefficient bar charts
# ---------------------------------------------------------------------------

def fig_coef_bars(name: str) -> None:
    """Horizontal bar charts of top 25 positive + top 25 negative ElasticNet
    coefficients for each task where a flashSCENIC coefficient CSV exists."""
    import matplotlib.patches as mpatches

    tasks_found = []
    for task in ("sex", "disease", "age"):
        # Prefer metacell coefficients; fall back to cell-level
        csv_meta = METRICS_DIR / f"ml_coef_{name}_flashscenic_{task}_metacells.csv"
        csv_cell = METRICS_DIR / f"ml_coef_{name}_flashscenic_{task}.csv"
        if csv_meta.exists():
            tasks_found.append((task, csv_meta))
            print(f"  [coef] {task}: using metacell coefficients")
        elif csv_cell.exists():
            tasks_found.append((task, csv_cell))
            print(f"  [coef] {task}: using cell-level coefficients (metacell not found)")

    if not tasks_found:
        print(f"  [skip] no flashSCENIC coefficient CSVs found")
        return

    fig_dir = FIGURES_DIR / name
    fig_dir.mkdir(parents=True, exist_ok=True)

    for task, csv in tasks_found:
        df = pd.read_csv(csv).sort_values("coefficient")
        neg = df[df["coefficient"] < 0].tail(25)
        pos = df[df["coefficient"] > 0].tail(25)
        plot_df = pd.concat([neg, pos]).reset_index(drop=True)

        if plot_df.empty:
            print(f"  [skip] all coefficients are zero for {task}")
            continue

        # Build legend labels: use saved class columns if present (new CSVs),
        # otherwise infer from the preprocessed h5ad obs (old CSVs).
        if "class_0" in df.columns:
            label_neg = f"associated with {df['class_0'].iloc[0]}"
            label_pos = f"associated with {df['class_1'].iloc[0]}"
        elif task in ("sex", "disease"):
            cfg = DATASETS[name]
            key = cfg.get(f"{task}_key")
            try:
                _adata = ad.read_h5ad(DATA_DIR / f"preprocessed_{name}.h5ad", backed="r")
                vals = sorted(_adata.obs[key].dropna().unique())
                _adata.file.close()
                label_neg = f"associated with {vals[0]}"
                label_pos = f"associated with {vals[1]}" if len(vals) > 1 else "associated with class 1"
            except Exception:
                label_neg, label_pos = "associated with class 0", "associated with class 1"
        else:
            label_neg = "associated with lower age"
            label_pos = "associated with higher age"

        n_bars = len(plot_df)
        fig, ax = plt.subplots(figsize=(10, max(6, 0.2 * n_bars + 1)))
        colors = ["#E84646" if c < 0 else "#4878CF" for c in plot_df["coefficient"]]
        ax.barh(plot_df["feature"], plot_df["coefficient"], color=colors)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("ElasticNet coefficient", fontsize=10)
        ax.set_title(f"{task.capitalize()} predictor coefficients",
                     fontsize=12, fontweight="bold")

        ax.legend(handles=[
            mpatches.Patch(color="#4878CF", label=f"Positive ({label_pos})"),
            mpatches.Patch(color="#E84646", label=f"Negative ({label_neg})"),
        ], fontsize=8, loc="lower right")

        plt.tight_layout()
        for ext in ["pdf", "png"]:
            out = fig_dir / f"fig5_coef_{task}_{name}.{ext}"
            plt.savefig(out, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"    Saved fig5_coef_{task}_{name}")


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

    if "sex_test_AUROC" not in ml_df.columns:
        print(f"  [skip] scatter plot: no sex AUROC data")
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    for method in METHODS:
        if method not in scib_df.index or method not in ml_df.index:
            continue
        if "iLISI" not in scib_df.columns:
            continue

        x = scib_df.loc[method, "iLISI"]
        y = ml_df.loc[method, "sex_test_AUROC"]

        if pd.isna(x) or pd.isna(y):
            continue

        ax.plot(x, y, "o", markersize=12,
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
    ax.set_title("Batch Mixing vs. Biological Signal Preservation",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, frameon=True)
    plt.tight_layout()

    fig_dir = FIGURES_DIR / name
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        out = fig_dir / f"fig4_scatter_{name}.{ext}"
        plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"    Saved fig4_scatter_{name}")


# ---------------------------------------------------------------------------
# Figure 6: Venn diagrams — DE TFs vs disease model coefficients
# ---------------------------------------------------------------------------

def fig_venn_tfs(name: str, de_method: str = "mwu") -> None:
    """Two Venn diagram figures comparing DE TFs with flashSCENIC disease model TFs.

    Figure A (fig6a_venn_2set): DE sig TFs (FDR<0.05) vs any model TF (any coef ≠ 0).
    Figure B (fig6b_venn_4set): 1×2 panel — direction-matched pairs:
      left:  DE ↑disease vs model +coef;
      right: DE ↑healthy vs model −coef.

    de_method: which DE method subfolder to load from ('mwu' or 'pseudobulk').
    """
    import importlib.util
    if importlib.util.find_spec("matplotlib_venn") is None:
        print("  [skip venn] matplotlib_venn not installed — "
              "pip install matplotlib-venn to enable")
        return
    from matplotlib_venn import venn2

    de_csv   = METRICS_DIR / de_method / f"de_tfs_{name}_all.csv"
    coef_csv = METRICS_DIR / f"ml_coef_{name}_flashscenic_disease.csv"

    if not de_csv.exists():
        print(f"  [skip venn] {de_csv.name} not found")
        return
    if not coef_csv.exists():
        print(f"  [skip venn] {coef_csv.name} not found")
        return

    de_df   = pd.read_csv(de_csv)
    coef_df = pd.read_csv(coef_csv)

    de_up    = set(de_df[(de_df["qval"] < 0.05) & (de_df["delta_mean"] >  0)]["TF"])
    de_down  = set(de_df[(de_df["qval"] < 0.05) & (de_df["delta_mean"] <  0)]["TF"])
    sig_all  = de_up | de_down
    pos_tfs  = set(coef_df[coef_df["coefficient"] > 0]["feature"])
    neg_tfs  = set(coef_df[coef_df["coefficient"] < 0]["feature"])
    model_all = pos_tfs | neg_tfs

    fig_dir = FIGURES_DIR / name
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- Save intersection CSVs -----------------------------------------
    def _intersect_df(sets: dict) -> pd.DataFrame:
        all_tfs = sorted(set.union(*sets.values())) if sets else []
        rows = []
        for tf in all_tfs:
            membership = {k: (tf in v) for k, v in sets.items()}
            rows.append({"TF": tf, **membership,
                         "in_intersection": sum(membership.values()) > 1})
        return pd.DataFrame(rows)

    de_metrics_sub = METRICS_DIR / de_method
    de_metrics_sub.mkdir(parents=True, exist_ok=True)

    csv_2set = de_metrics_sub / f"venn_2set_{name}.csv"
    _intersect_df({"DE_sig": sig_all, "model_TFs": model_all}).to_csv(
        csv_2set, index=False)
    print(f"    Saved {csv_2set.name}")

    csv_up = de_metrics_sub / f"venn_disease_{name}.csv"
    _intersect_df({"DE_up_disease": de_up, "model_pos_coef": pos_tfs}).to_csv(
        csv_up, index=False)
    print(f"    Saved {csv_up.name}")

    csv_down = de_metrics_sub / f"venn_healthy_{name}.csv"
    _intersect_df({"DE_up_healthy": de_down, "model_neg_coef": neg_tfs}).to_csv(
        csv_down, index=False)
    print(f"    Saved {csv_down.name}")

    # ---- Figure A: 2-set ------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 5))
    v = venn2([sig_all, model_all],
              set_labels=["DE TFs (FDR<0.05)", "Model TFs (any coef)"],
              set_colors=["#888888", "#E84646"],
              alpha=0.5, ax=ax)
    ax.set_title("DE TFs vs disease model", fontsize=11, fontweight="bold")
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        plt.savefig(fig_dir / f"fig6a_venn_2set_{name}.{ext}",
                    bbox_inches="tight", dpi=150)
    plt.close()
    print(f"    Saved fig6a_venn_2set_{name}")

    # ---- Figure B: direction-matched 1×2 panel --------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    venn2([de_up, pos_tfs],
          set_labels=["DE ↑disease (FDR<0.05)", "Model +coef (disease)"],
          set_colors=["#E84646", "#E84646"],
          alpha=0.5, ax=axes[0])
    axes[0].set_title("Disease-associated TFs", fontsize=10, fontweight="bold")

    venn2([de_down, neg_tfs],
          set_labels=["DE ↑healthy (FDR<0.05)", "Model −coef (healthy)"],
          set_colors=["#4878CF", "#4878CF"],
          alpha=0.5, ax=axes[1])
    axes[1].set_title("Healthy-associated TFs", fontsize=10, fontweight="bold")

    fig.suptitle("Direction-consistent TF overlap", fontsize=12, fontweight="bold")
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        plt.savefig(fig_dir / f"fig6b_venn_4set_{name}.{ext}",
                    bbox_inches="tight", dpi=150)
    plt.close()
    print(f"    Saved fig6b_venn_4set_{name}")


# ---------------------------------------------------------------------------
# Figure 7: Per-cell-type disease prediction
# ---------------------------------------------------------------------------

def fig_ml_per_ct(name: str) -> None:
    """Horizontal grouped bar chart of per-cell-type disease AUROC."""
    path = METRICS_DIR / f"ml_disease_per_ct_{name}.csv"
    if not path.exists():
        print(f"  [skip] {path.name} not found")
        return

    df = pd.read_csv(path)
    if "test_AUROC" not in df.columns or df.empty:
        print(f"  [skip] per-CT AUROC: empty or missing test_AUROC column")
        return

    methods = [m for m in METHODS if m in df["method"].unique()]
    cell_types = sorted(df["cell_type"].unique())
    n_ct = len(cell_types)
    n_methods = len(methods)
    bar_height = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(6, max(3, n_ct * 0.5 * n_methods + 1)))

    for mi, m in enumerate(methods):
        sub = df[df["method"] == m].set_index("cell_type")
        offsets = np.arange(n_ct) + (mi - n_methods / 2 + 0.5) * bar_height
        vals = []
        errs = []
        for ct in cell_types:
            if ct in sub.index:
                vals.append(sub.loc[ct, "test_AUROC"])
                errs.append(sub.loc[ct, "test_AUROC_std"] if "test_AUROC_std" in sub.columns else 0.0)
            else:
                vals.append(np.nan)
                errs.append(0.0)
        # Only plot non-nan values
        valid = [i for i, v in enumerate(vals) if not np.isnan(v)]
        if not valid:
            continue
        ax.barh(
            [offsets[i] for i in valid],
            [vals[i] for i in valid],
            height=bar_height * 0.9,
            xerr=[errs[i] for i in valid],
            color=METHOD_COLORS.get(m, "grey"),
            label=METHOD_LABELS.get(m, m),
            capsize=2, error_kw={"lw": 0.8},
        )

    # Overlay metacell AUROC as diamond markers (if available)
    meta_path = METRICS_DIR / f"ml_disease_per_ct_{name}_metacells.csv"
    if meta_path.exists():
        meta_df = pd.read_csv(meta_path)
        meta_legend_added = False
        for mi, m in enumerate(methods):
            sub = meta_df[meta_df["method"] == m].set_index("cell_type")
            offsets = np.arange(n_ct) + (mi - n_methods / 2 + 0.5) * bar_height
            for ci, ct in enumerate(cell_types):
                if ct in sub.index and not pd.isna(sub.loc[ct, "test_AUROC"]):
                    ax.scatter(sub.loc[ct, "test_AUROC"], offsets[ci],
                               marker="D", s=40,
                               color=METHOD_COLORS.get(m, "grey"),
                               zorder=5,
                               label="Metacell (◆)" if not meta_legend_added else "")
                    meta_legend_added = True

    ax.axvline(0.5, color="grey", linestyle="--", lw=1, label="Random (0.5)")
    ax.axvline(1.0, color="black", lw=0.5, alpha=0.3)
    ax.set_yticks(np.arange(n_ct))
    ax.set_yticklabels(cell_types, fontsize=8)
    ax.set_xlabel("mean AUROC ± std (6 folds)", fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_title("Disease Prediction per Cell Type", fontsize=12, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.suptitle("Per-cell-type disease AUROC", fontsize=12, y=1.01)
    plt.tight_layout()

    fig_dir = FIGURES_DIR / name
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        out = fig_dir / f"fig7_ml_per_ct_{name}.{ext}"
        plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"    Saved fig7_ml_per_ct_{name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all",
                        choices=list(DATASETS.keys()) + ["all"])
    parser.add_argument("--de-method", default="mwu",
                        choices=["mwu", "pseudobulk"],
                        help="Which DE method results to use for Venn diagrams (default: pseudobulk)")
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

        print(f"\n  Figure 7: Per-cell-type disease prediction")
        fig_ml_per_ct(name)

        print(f"\n  Figure 4: Sex AUROC vs iLISI scatter")
        fig_scatter(name)

        print(f"\n  Figure 5: flashSCENIC coefficient bar charts")
        fig_coef_bars(name)

        print(f"\n  Figure 6: TF overlap Venn diagrams ({args.de_method})")
        fig_venn_tfs(name, de_method=args.de_method)

    print(f"\nAll figures saved to {FIGURES_DIR}")
    print("Run 06_summarize.py to generate the final report.")


if __name__ == "__main__":
    main()
