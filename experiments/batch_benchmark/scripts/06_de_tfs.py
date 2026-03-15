"""
Step 6: Differential TF activity between disease groups.

Uses flashSCENIC AUCell scores. For each dataset that has a disease_key,
computes donor-level pseudobulk mean AUCell scores, then runs a Mann-Whitney U
test (two-sided) per regulon between disease and control donors. FDR is
controlled with the Benjamini-Hochberg procedure.

Runs for all cells combined AND for each cell type separately.
Cell types with < 3 donors in either group are skipped.

Outputs (per subset):
  results/metrics/de_tfs_{dataset}_{celltype|all}.csv
  results/figures/de_tfs_{dataset}_{celltype|all}.pdf/.png

Usage:
    python 06_de_tfs.py [--dataset als_motor_cortex|als_spinal_cord|all]
                        [--top_n 30] [--min_abs_delta 0]
"""

import argparse
import sys
import warnings
from pathlib import Path

import anndata as ad
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, ttest_ind
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).parent))
from config import DATASETS, DATA_DIR, EMBEDDINGS_DIR, METRICS_DIR, FIGURES_DIR, MIN_CELLS_PER_CT, ML

warnings.filterwarnings("ignore")

# Disease-up color / disease-down color (matches METHOD_COLORS palette)
COLOR_UP   = "#E84646"   # up in disease
COLOR_DOWN = "#4878CF"   # down in disease


# ---------------------------------------------------------------------------
# Metacell helper
# ---------------------------------------------------------------------------

def _make_metacells_de(
    auc_scores: np.ndarray,
    donors: np.ndarray,
    labels: np.ndarray,
    n_metacells: int = 100,
    metacell_size: int = 32,
    seed: int = 42,
    expected_reuse: float | None = 2.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each donor, draw bootstrap samples of metacell_size cells (with replacement)
    and average their AUCell scores. Returns (meta_scores, meta_labels, meta_donors).
    Cells passed in must already be from a single cell type.

    When expected_reuse is set, n_metacells is adaptive per donor:
        n_mc = max(1, round(expected_reuse * n_cells_donor / metacell_size))
    """
    rng = np.random.default_rng(seed)
    meta_scores, meta_labels, meta_donors = [], [], []
    for donor in np.unique(donors):
        mask = donors == donor
        X_d = auc_scores[mask]
        label = labels[mask][0]
        n_cells = len(X_d)
        n_mc = (max(1, round(expected_reuse * n_cells / metacell_size))
                if expected_reuse is not None else n_metacells)
        for _ in range(n_mc):
            idx = rng.integers(0, n_cells, size=min(metacell_size, n_cells))
            meta_scores.append(X_d[idx].mean(axis=0))
            meta_labels.append(label)
            meta_donors.append(donor)
    return np.array(meta_scores), np.array(meta_labels), np.array(meta_donors)


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def _resolve_disease_labels(
    disease_series: pd.Series,
) -> tuple[str, str] | None:
    """Return (disease_label, control_label) or None if groups can't be resolved."""
    values = disease_series.unique()
    if len(values) < 2:
        return None
    control_set = {"normal", "healthy", "control"}
    disease_cands = [v for v in values if str(v).lower() not in control_set]
    control_cands = [v for v in values if str(v).lower() in control_set]
    if not disease_cands or not control_cands:
        return str(values[0]), str(values[1])
    return str(disease_cands[0]), str(control_cands[0])


def run_de(
    auc_scores: np.ndarray,
    obs: pd.DataFrame,
    disease_key: str,
    regulon_names: list[str],
    min_cells: int = 10,
) -> pd.DataFrame | None:
    """
    Cell-level Mann-Whitney U test per regulon between disease and control cells.

    Using cells instead of donors gives far more statistical power, especially
    when the number of donors is small (e.g. 12 ALS donors).
    Caveat: cells from the same donor are not independent — treat results as
    exploratory.

    Returns None if either group has < min_cells cells.
    """
    labels = _resolve_disease_labels(obs[disease_key])
    if labels is None:
        print(f"    [skip] only one disease class found")
        return None
    disease_label, control_label = labels

    dis_mask  = (obs[disease_key] == disease_label).values
    ctrl_mask = (obs[disease_key] == control_label).values

    if dis_mask.sum() < min_cells or ctrl_mask.sum() < min_cells:
        print(f"    [skip] too few cells: "
              f"{dis_mask.sum()} disease, {ctrl_mask.sum()} control "
              f"(need ≥ {min_cells} each)")
        return None

    dis_mat  = auc_scores[dis_mask]   # (n_dis_cells,  n_regulons)
    ctrl_mat = auc_scores[ctrl_mask]  # (n_ctrl_cells, n_regulons)

    n_regs = dis_mat.shape[1]
    pvals    = np.empty(n_regs)
    mean_dis  = dis_mat.mean(axis=0)
    mean_ctrl = ctrl_mat.mean(axis=0)

    for i in range(n_regs):
        _, pvals[i] = mannwhitneyu(dis_mat[:, i], ctrl_mat[:, i], alternative="two-sided")

    delta_means = mean_dis - mean_ctrl
    _, qvals, _, _ = multipletests(pvals, method="fdr_bh")
    neglog10q = -np.log10(np.clip(qvals, 1e-300, 1.0))

    result = pd.DataFrame({
        "TF":             regulon_names,
        "pval":           pvals,
        "qval":           qvals,
        "neglog10q":      neglog10q,
        "mean_disease":   mean_dis,
        "mean_control":   mean_ctrl,
        "delta_mean":     delta_means,
        "abs_delta_mean": np.abs(delta_means),
    }).sort_values("qval")

    print(f"    DE complete (cell-level): {dis_mask.sum():,} {disease_label} vs "
          f"{ctrl_mask.sum():,} {control_label} cells — "
          f"{(qvals < 0.05).sum()} TFs at FDR < 0.05")
    return result


def run_de_pseudobulk(
    auc_scores: np.ndarray,
    obs: pd.DataFrame,
    disease_key: str,
    donor_key: str,
    regulon_names: list[str],
    sex_key: str | None = None,
    min_donors: int = 3,
    fdr_report: float = 0.2,
    min_abs_delta: float = 0.0,
) -> pd.DataFrame | None:
    """
    Pseudobulk OLS per regulon (DESeq2-style for continuous AUCell scores).

    Aggregates AUCell scores to per-donor means, then fits an OLS model:
        auc ~ C(disease) [+ C(sex)]
    Falls back to Welch t-test for TFs where the OLS matrix is singular.

    min_abs_delta: pre-filter — only test TFs where |mean_disease - mean_control|
    exceeds this threshold, reducing multiple testing burden.

    Reports significance at FDR < fdr_report (default 0.2).
    """
    labels = _resolve_disease_labels(obs[disease_key])
    if labels is None:
        print(f"    [skip] only one disease class found")
        return None
    disease_label, control_label = labels

    # Build per-donor metadata + AUCell matrix
    all_donors = np.unique(obs[donor_key].values)
    meta_rows: list[dict] = []
    auc_rows:  list[np.ndarray] = []

    for d in all_donors:
        mask = obs[donor_key].values == d
        row: dict = {
            "donor":   d,
            "disease": obs.loc[mask, disease_key].iloc[0],
        }
        if sex_key:
            row["sex"] = obs.loc[mask, sex_key].iloc[0]
        meta_rows.append(row)
        auc_rows.append(auc_scores[mask].mean(axis=0))

    donor_meta = pd.DataFrame(meta_rows)
    auc_pb = np.vstack(auc_rows)   # (n_donors, n_regs)

    n_dis_donors  = (donor_meta["disease"] == disease_label).sum()
    n_ctrl_donors = (donor_meta["disease"] == control_label).sum()

    if n_dis_donors < min_donors or n_ctrl_donors < min_donors:
        print(f"    [skip pseudobulk] too few donors: "
              f"{n_dis_donors} disease, {n_ctrl_donors} control "
              f"(need ≥ {min_donors} each)")
        return None

    # Pre-compute mean per group for all regulons (used for pre-filter + output)
    mean_dis  = auc_pb[donor_meta["disease"].values == disease_label].mean(axis=0)
    mean_ctrl = auc_pb[donor_meta["disease"].values == control_label].mean(axis=0)
    raw_delta = mean_dis - mean_ctrl

    # Pre-filter: only test TFs with large enough effect
    if min_abs_delta > 0:
        test_mask = np.abs(raw_delta) >= min_abs_delta
        n_skipped = (~test_mask).sum()
        print(f"    Pre-filter |Δ| ≥ {min_abs_delta}: "
              f"{test_mask.sum()} / {len(test_mask)} TFs to test "
              f"({n_skipped} skipped)")
    else:
        test_mask = np.ones(len(regulon_names), dtype=bool)

    # Build OLS formula (sex covariate only)
    covariates = []
    if sex_key and "sex" in donor_meta.columns:
        covariates.append("C(sex)")
    covar_str = (" + " + " + ".join(covariates)) if covariates else ""
    formula = (f"auc ~ C(disease, Treatment(reference='{control_label}'))"
               f"{covar_str}")
    disease_col = (f"C(disease, Treatment(reference='{control_label}'))"
                   f"[T.{disease_label}]")

    test_indices = np.where(test_mask)[0]
    pvals         = np.ones(len(regulon_names))
    disease_coefs = raw_delta.copy()
    n_fallback    = 0

    donor_meta = donor_meta.copy()
    for i in test_indices:
        donor_meta["auc"] = auc_pb[:, i]
        try:
            fit = smf.ols(formula, data=donor_meta).fit(disp=0)
            pvals[i]         = fit.pvalues.get(disease_col, np.nan)
            disease_coefs[i] = fit.params.get(disease_col, np.nan)
        except Exception:
            # Singular matrix → fall back to Welch t-test (coef stays as raw_delta)
            dis_vals  = auc_pb[donor_meta["disease"].values == disease_label, i]
            ctrl_vals = auc_pb[donor_meta["disease"].values == control_label, i]
            _, pvals[i] = ttest_ind(dis_vals, ctrl_vals, equal_var=False)
            n_fallback += 1

    if n_fallback:
        print(f"    [pseudobulk] {n_fallback} / {len(test_indices)} TFs used Welch t-test fallback "
              f"(singular OLS matrix; coef = raw Δ mean)")

    # Replace NaN p-values / coefficients with safe defaults
    pvals = np.where(np.isnan(pvals), 1.0, pvals)
    nan_coef = np.isnan(disease_coefs)
    disease_coefs[nan_coef] = raw_delta[nan_coef]

    # BH correction only over tested TFs (not the pre-filtered ones)
    _, qvals_tested, _, _ = multipletests(pvals[test_mask], method="fdr_bh")
    qvals = np.ones(len(regulon_names))
    qvals[test_mask] = qvals_tested

    neglog10q = -np.log10(np.clip(qvals, 1e-300, 1.0))

    result = pd.DataFrame({
        "TF":             regulon_names,
        "pval":           pvals,
        "qval":           qvals,
        "neglog10q":      neglog10q,
        "mean_disease":   mean_dis,
        "mean_control":   mean_ctrl,
        "delta_mean":     disease_coefs,
        "abs_delta_mean": np.abs(disease_coefs),
    }).sort_values("qval")

    covariates_desc = (f"covariates: {', '.join(covariates)}" if covariates
                       else "no covariates")
    print(f"    DE complete (pseudobulk OLS, {covariates_desc}): "
          f"{n_dis_donors} {disease_label} vs {n_ctrl_donors} {control_label} donors — "
          f"{(qvals[test_mask] < fdr_report).sum()} TFs at FDR < {fdr_report} "
          f"(tested {test_mask.sum()} TFs)")
    return result


def run_de_metacell(
    auc_scores: np.ndarray,
    obs: pd.DataFrame,
    disease_key: str,
    donor_key: str,
    regulon_names: list[str],
    n_metacells: int = 100,
    metacell_size: int = 32,
    seed: int = 42,
    min_donors: int = 3,
    expected_reuse: float | None = 2.5,
    delta_weighting: str = "donor",
) -> pd.DataFrame | None:
    """
    Metacell-level Mann-Whitney U test per regulon.

    For each donor, draws adaptive bootstrap samples (expected_reuse * n_cells /
    metacell_size per donor) and averages their AUCell scores → per-donor metacells.
    Then runs MWU between all disease metacells and all control metacells.

    Intermediate between cell-level MWU (pseudoreplication) and pseudobulk OLS
    (underpowered at n=12 donors).

    Cell-type homogeneity is guaranteed by the calling code: obs is always
    pre-filtered to a single cell type before this function is invoked.

    delta_weighting:
        "donor"    → mean_disease / mean_control are averages of per-donor means
                     (each donor contributes equally regardless of cell count)
        "metacell" → grand mean across all pooled metacells
                     (larger donors dominate proportionally to cell count)
    """
    labels = _resolve_disease_labels(obs[disease_key])
    if labels is None:
        return None
    disease_label, control_label = labels

    donors = obs[donor_key].values
    disease_labels = obs[disease_key].values

    n_dis_donors  = len(np.unique(donors[disease_labels == disease_label]))
    n_ctrl_donors = len(np.unique(donors[disease_labels == control_label]))
    if n_dis_donors < min_donors or n_ctrl_donors < min_donors:
        print(f"    [skip metacell] too few donors: "
              f"{n_dis_donors} disease, {n_ctrl_donors} control "
              f"(need ≥ {min_donors} each)")
        return None

    meta_scores, meta_labels, meta_donors = _make_metacells_de(
        auc_scores, donors, disease_labels,
        n_metacells=n_metacells, metacell_size=metacell_size, seed=seed,
        expected_reuse=expected_reuse,
    )

    dis_mask  = meta_labels == disease_label
    ctrl_mask = meta_labels == control_label
    dis_mat   = meta_scores[dis_mask]
    ctrl_mat  = meta_scores[ctrl_mask]

    n_regs = dis_mat.shape[1]
    pvals = np.empty(n_regs)
    for i in range(n_regs):
        _, pvals[i] = mannwhitneyu(dis_mat[:, i], ctrl_mat[:, i], alternative="two-sided")

    if delta_weighting == "donor":
        # Average metacells within each donor first → equal weight per donor
        dis_donors  = np.unique(meta_donors[dis_mask])
        ctrl_donors = np.unique(meta_donors[ctrl_mask])
        mean_dis  = np.stack([meta_scores[(meta_donors == d) & dis_mask].mean(axis=0)
                              for d in dis_donors]).mean(axis=0)
        mean_ctrl = np.stack([meta_scores[(meta_donors == d) & ctrl_mask].mean(axis=0)
                              for d in ctrl_donors]).mean(axis=0)
    else:
        # Grand mean across all pooled metacells (larger donors dominate)
        mean_dis  = dis_mat.mean(axis=0)
        mean_ctrl = ctrl_mat.mean(axis=0)

    delta_means = mean_dis - mean_ctrl
    _, qvals, _, _ = multipletests(pvals, method="fdr_bh")
    neglog10q = -np.log10(np.clip(qvals, 1e-300, 1.0))

    result = pd.DataFrame({
        "TF":             regulon_names,
        "pval":           pvals,
        "qval":           qvals,
        "neglog10q":      neglog10q,
        "mean_disease":   mean_dis,
        "mean_control":   mean_ctrl,
        "delta_mean":     delta_means,
        "abs_delta_mean": np.abs(delta_means),
    }).sort_values("qval")

    reuse_str = f"k={expected_reuse}" if expected_reuse is not None else f"fixed n={n_metacells}"
    print(f"    DE complete (metacell MWU, {reuse_str}×{metacell_size} cells/donor, "
          f"Δ weighted by {delta_weighting}): "
          f"{len(dis_mat):,} disease vs {len(ctrl_mat):,} control metacells — "
          f"{(qvals < 0.05).sum()} TFs at FDR < 0.05")
    return result


# ---------------------------------------------------------------------------
# Dotplot
# ---------------------------------------------------------------------------

def dotplot_de(
    de_df: pd.DataFrame,
    title: str,
    outfile: Path,
    top_n: int = 30,
    min_abs_delta: float = 0.0,
    fdr_threshold: float = 0.05,
) -> None:
    """Seaborn-styled dotplot: x = -log10(q), y = TF name, color = direction."""
    sns.set_theme(style="whitegrid", font_scale=1.0)

    df = de_df.copy()
    if min_abs_delta > 0:
        df = df[df["abs_delta_mean"] >= min_abs_delta]

    df = df.sort_values("neglog10q", ascending=False).head(top_n).copy()
    if df.empty:
        print(f"    [skip] no TFs pass filters for {title}")
        return

    df = df.sort_values("neglog10q", ascending=True)  # ascending so most sig on top

    sizes = df["abs_delta_mean"].values
    max_size = np.nanmax(sizes) if np.nanmax(sizes) > 0 else 1.0
    size_scaled = 200 * (sizes / max_size)

    colors = [COLOR_UP if d > 0 else COLOR_DOWN for d in df["delta_mean"]]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(df) + 1.5)))
    ax.scatter(df["neglog10q"].values, np.arange(len(df)),
               s=np.abs(size_scaled), c=colors, alpha=0.85, linewidths=0.5,
               edgecolors="white")

    ax.axvline(-np.log10(fdr_threshold), color="gray", linestyle="--", lw=1,
               label=f"FDR = {fdr_threshold}")
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["TF"].values, fontsize=8)
    ax.set_xlabel("−log₁₀(q-value)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")

    legend_elems = [
        mpatches.Patch(color=COLOR_UP,   label="↑ in disease"),
        mpatches.Patch(color=COLOR_DOWN, label="↓ in disease"),
        plt.Line2D([0], [0], color="gray", linestyle="--", lw=1,
                   label=f"FDR = {fdr_threshold}"),
    ]
    for frac in [0.25, 0.5, 1.0]:
        legend_elems.append(
            ax.scatter([], [], s=200 * frac, c="gray", alpha=0.85,
                       label=f"|Δ| = {frac * max_size:.3f}")
        )
    ax.legend(handles=legend_elems, fontsize=8, loc="lower right")

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        plt.savefig(outfile.with_suffix(f".{ext}"), bbox_inches="tight", dpi=150)
    plt.close()
    plt.rcdefaults()  # reset seaborn theme to avoid affecting other plots


def _collect_sig_tfs(
    results: dict[str, pd.DataFrame],
    top_n: int = 5,
    fdr_threshold: float = 0.05,
    min_abs_delta: float = 0.0,
    balanced: bool = True,
) -> tuple[list[str], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """
    For each cell type select significant TFs (FDR < fdr_threshold).

    balanced=True  → top `top_n` up-regulated AND top `top_n` down-regulated
                     (selected by |delta_mean| within each direction group)
    balanced=False → top `top_n` overall by |delta_mean| regardless of direction
    """
    ct_names: list[str] = [k for k in results if k != "all"]
    tf_scores: dict[str, dict[str, float]] = {}
    tf_deltas: dict[str, dict[str, float]] = {}

    for ct in ct_names:
        df = results[ct].copy()
        sig = df[df["qval"] < fdr_threshold]
        if min_abs_delta > 0:
            sig = sig[sig["abs_delta_mean"] >= min_abs_delta]
        if balanced:
            top_up   = sig[sig["delta_mean"] > 0].nlargest(top_n, "delta_mean")
            top_down = sig[sig["delta_mean"] < 0].nsmallest(top_n, "delta_mean")
            top = pd.concat([top_up, top_down])
        else:
            top = sig.nlargest(top_n, "abs_delta_mean")
        for _, row in top.iterrows():
            tf = row["TF"]
            tf_scores.setdefault(tf, {})[ct] = row["neglog10q"]
            tf_deltas.setdefault(tf, {})[ct] = row["delta_mean"]

    return ct_names, tf_scores, tf_deltas


def _hclust_order(tfs: list[str], mat: np.ndarray, tf_idx: dict[str, int]) -> list[str]:
    """Reorder TFs by hierarchical clustering of their row vectors."""
    from scipy.cluster.hierarchy import linkage, leaves_list

    if len(tfs) < 3:
        return tfs
    rows = np.array([mat[tf_idx[t]] for t in tfs])
    rows = np.nan_to_num(rows, nan=0.0)
    Z = linkage(rows, method="average", metric="euclidean")
    return [tfs[i] for i in leaves_list(Z)]


def _add_left_label(ax: plt.Axes, y_frac: float, text: str, color: str) -> None:
    """Add a rotated group label to the left of the y-axis (further left than tick labels)."""
    ax.annotate(
        text,
        xy=(0, y_frac), xycoords="axes fraction",
        xytext=(-0.16, y_frac), textcoords="axes fraction",
        va="center", ha="right", fontsize=8,
        color=color, fontweight="bold", rotation=90,
        annotation_clip=False,
    )


def dotplot_summary(
    results: dict[str, pd.DataFrame],
    title: str,
    outfile: Path,
    top_n_per_ct: int = 5,
    min_abs_delta: float = 0.0,
    fdr_threshold: float = 0.05,
) -> None:
    """
    Multi-column dotplot: rows = TFs, columns = cell types.

    TF selection: top `top_n_per_ct` per cell type by |delta| among FDR < fdr_threshold.
    Within up/down groups: hierarchical clustering.
    Direction labels on the LEFT side (further left than TF tick labels).

    Color = −log10(q-value)   (missing = not significant in that cell type)
    Size  = |delta_mean|
    """
    sns.set_theme(style="whitegrid", font_scale=0.9)

    ct_names, tf_scores, tf_deltas = _collect_sig_tfs(
        results, top_n=top_n_per_ct, min_abs_delta=min_abs_delta,
        fdr_threshold=fdr_threshold
    )
    if not ct_names:
        return
    if not tf_scores:
        print(f"    [skip summary dotplot] no TFs pass FDR < {fdr_threshold}")
        return

    all_tfs = list(tf_scores.keys())

    def _mean_delta(tf: str) -> float:
        vals = list(tf_deltas[tf].values())
        return float(np.nanmean(vals)) if vals else 0.0

    # ---- build matrices (needed for clustering) -------------------------
    n_tfs_all = len(all_tfs)
    n_cts     = len(ct_names)
    neglogq_tmp = np.full((n_tfs_all, n_cts), np.nan)
    tf_idx_tmp  = {tf: i for i, tf in enumerate(all_tfs)}
    ct_idx      = {ct: j for j, ct in enumerate(ct_names)}
    for tf in all_tfs:
        for ct, val in tf_scores[tf].items():
            neglogq_tmp[tf_idx_tmp[tf], ct_idx[ct]] = val

    up_tfs   = [t for t in all_tfs if _mean_delta(t) >= 0]
    down_tfs = [t for t in all_tfs if _mean_delta(t) <  0]
    up_tfs   = _hclust_order(up_tfs,   neglogq_tmp, tf_idx_tmp)
    down_tfs = _hclust_order(down_tfs, neglogq_tmp, tf_idx_tmp)
    ordered_tfs = up_tfs + down_tfs

    # ---- rebuild final matrices -----------------------------------------
    n_tfs = len(ordered_tfs)
    neglogq_mat = np.full((n_tfs, n_cts), np.nan)
    delta_mat   = np.full((n_tfs, n_cts), np.nan)
    tf_idx = {tf: i for i, tf in enumerate(ordered_tfs)}

    for tf in ordered_tfs:
        for ct, val in tf_scores[tf].items():
            neglogq_mat[tf_idx[tf], ct_idx[ct]] = val
            delta_mat[tf_idx[tf], ct_idx[ct]] = tf_deltas[tf].get(ct, np.nan)

    # ---- size scaling ---------------------------------------------------
    max_delta = np.nanmax(np.abs(delta_mat))
    max_delta = max_delta if max_delta > 0 else 1.0
    size_mat = 300 * (np.abs(delta_mat) / max_delta)

    # ---- plot -----------------------------------------------------------
    fig_h = max(5, 0.28 * n_tfs + 2.0)
    fig_w = max(5, 0.7 * n_cts + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    valid_vals = neglogq_mat[~np.isnan(neglogq_mat)]
    cmap = plt.cm.YlOrRd
    vmin = 0.0
    vmax = float(np.nanpercentile(valid_vals, 95)) if len(valid_vals) else 1.0

    xs, ys, colors_vals, sizes_vals = [], [], [], []
    for i in range(n_tfs):
        for j in range(n_cts):
            if np.isnan(neglogq_mat[i, j]):
                continue
            xs.append(j); ys.append(i)
            colors_vals.append(neglogq_mat[i, j])
            sizes_vals.append(size_mat[i, j])

    sc = ax.scatter(xs, ys, c=colors_vals, s=sizes_vals,
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    alpha=0.9, linewidths=0.4, edgecolors="gray")

    cbar = plt.colorbar(sc, ax=ax, shrink=0.45, pad=0.02)
    cbar.ax.set_title("−log₁₀(q)", fontsize=8, pad=4)

    legend_sizes = [0.25, 0.5, 1.0]
    legend_handles = [
        plt.scatter([], [], s=300 * s, color="gray", alpha=0.7,
                    label=f"|Δ| = {s * max_delta:.3f}")
        for s in legend_sizes
    ]
    ax.legend(handles=legend_handles, title="|Δ mean AUCell|", fontsize=7,
              title_fontsize=7, loc="lower left", framealpha=0.8,
              bbox_to_anchor=(0.0, -0.02))

    ax.set_xticks(range(n_cts))
    ax.set_xticklabels(ct_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n_tfs))
    ax.set_yticklabels(ordered_tfs, fontsize=7)
    ax.set_xlim(-0.5, n_cts - 0.5)
    ax.set_ylim(-0.5, n_tfs - 0.5)
    ax.invert_yaxis()

    # ---- group separator + LEFT-side labels ----------------------------
    if up_tfs and down_tfs:
        sep_y = len(up_tfs) - 0.5
        ax.axhline(sep_y, color="black", lw=1.2, linestyle="--")
        up_frac   = (len(up_tfs)   / 2) / n_tfs
        down_frac = (len(up_tfs) + len(down_tfs) / 2) / n_tfs
        _add_left_label(ax, 1 - up_frac,   "↑ disease", COLOR_UP)
        _add_left_label(ax, 1 - down_frac, "↓ disease", COLOR_DOWN)
    elif up_tfs:
        _add_left_label(ax, 0.5, "↑ disease", COLOR_UP)
    else:
        _add_left_label(ax, 0.5, "↓ disease", COLOR_DOWN)

    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        plt.savefig(outfile.with_suffix(f".{ext}"), bbox_inches="tight", dpi=150)
    plt.close()
    plt.rcdefaults()


def dotplot_de_by_delta(
    de_df: pd.DataFrame,
    title: str,
    outfile: Path,
    top_n: int = 10,
    fdr_threshold: float = 0.05,
) -> None:
    """Dotplot ordered by delta_mean: top (top_n/2) up + top (top_n/2) down TFs by |delta|.

    Mirrors the TF selection criterion used in heatmap_summary so individual
    cell-type plots are consistent with the summary heatmap.
    """
    sns.set_theme(style="whitegrid", font_scale=1.0)

    df = de_df.copy()
    sig = df[df["qval"] < fdr_threshold]

    half = top_n // 2
    top_up   = sig[sig["delta_mean"] > 0].nlargest(half,   "delta_mean")
    top_down = sig[sig["delta_mean"] < 0].nsmallest(half,  "delta_mean")
    plot_df  = pd.concat([top_up, top_down])

    if plot_df.empty:
        print(f"    [skip by-Δ] no significant TFs for {title}")
        return

    # Sort ascending so highest delta (most disease-up) is at the top of the plot
    plot_df = plot_df.sort_values("delta_mean", ascending=True)

    neglogq     = plot_df["neglog10q"].values
    max_neglogq = np.nanmax(neglogq) if np.nanmax(neglogq) > 0 else 1.0
    size_scaled = 200 * (neglogq / max_neglogq)
    colors      = [COLOR_UP if d > 0 else COLOR_DOWN for d in plot_df["delta_mean"]]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(plot_df) + 1.5)))
    ax.scatter(plot_df["delta_mean"].values, np.arange(len(plot_df)),
               s=size_scaled, c=colors, alpha=0.85, linewidths=0.5,
               edgecolors="white")

    ax.axvline(0, color="gray", linestyle="--", lw=1)
    ax.set_yticks(np.arange(len(plot_df)))
    ax.set_yticklabels(plot_df["TF"].values, fontsize=8)
    ax.set_xlabel("Δ mean AUCell (disease − control)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")

    legend_elems = [
        mpatches.Patch(color=COLOR_UP,   label="↑ in disease"),
        mpatches.Patch(color=COLOR_DOWN, label="↓ in disease"),
    ]
    for frac in [0.25, 0.5, 1.0]:
        legend_elems.append(
            ax.scatter([], [], s=200 * frac, c="gray", alpha=0.85,
                       label=f"−log₁₀(q) = {frac * max_neglogq:.1f}")
        )
    ax.legend(handles=legend_elems, fontsize=8, loc="lower right")

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        plt.savefig(outfile.with_suffix(f".{ext}"), bbox_inches="tight", dpi=150)
    plt.close()
    plt.rcdefaults()


def _stitch_pngs(
    png_paths: list[Path],
    out_path: Path,
    n_cols: int = 3,
    title: str = "",
    pad: int = 20,
) -> None:
    """Stitch PNG files into a grid and save as PNG + PDF."""
    from PIL import Image, ImageDraw, ImageFont

    if not png_paths:
        return

    imgs   = [Image.open(p) for p in png_paths]
    cell_w = max(im.width  for im in imgs)
    cell_h = max(im.height for im in imgs)
    n_rows = (len(imgs) + n_cols - 1) // n_cols
    title_h = 60 if title else 0

    canvas_w = n_cols * cell_w + (n_cols + 1) * pad
    canvas_h = n_rows * cell_h + (n_rows + 1) * pad + title_h
    canvas   = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))

    if title:
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
        except Exception:
            font = ImageFont.load_default()
        draw.text((canvas_w // 2, pad), title, fill=(30, 30, 30),
                  font=font, anchor="mt")

    for idx, im in enumerate(imgs):
        row = idx // n_cols
        col = idx % n_cols
        x   = pad + col * (cell_w + pad)
        y   = title_h + pad + row * (cell_h + pad)
        canvas.paste(im, (x + (cell_w - im.width) // 2,
                          y + (cell_h - im.height) // 2))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(out_path))
    dpi   = 150
    fig_w = canvas_w / dpi
    fig_h = canvas_h / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(np.array(canvas))
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(str(out_path).replace(".png", ".pdf"), bbox_inches="tight", dpi=dpi)
    plt.close()
    print(f"    Saved {out_path.name} ({len(imgs)} panels, {n_cols} cols)")


def _add_top_label(ax: plt.Axes, x_frac: float, text: str, color: str) -> None:
    """Add a group label above the x-axis at a given axes-fraction x position."""
    ax.annotate(
        text,
        xy=(x_frac, 1), xycoords="axes fraction",
        xytext=(x_frac, 1.02), textcoords="axes fraction",
        va="bottom", ha="center", fontsize=10,
        color=color, fontweight="bold",
        annotation_clip=False,
    )


def heatmap_summary(
    results: dict[str, pd.DataFrame],
    title: str,
    outfile: Path,
    top_n_per_ct: int = 5,
    min_abs_delta: float = 0.0,
    balanced: bool = True,
    fdr_threshold: float = 0.05,
) -> None:
    """
    Heatmap summary: rows = cell types, columns = TFs.

    Color = delta_mean (diverging RdBu_r: red = higher in disease, blue = higher in healthy).
    Dot overlay = size ∝ −log10(q), drawn only where data exists.
    TF order: disease-associated (delta > 0) on left, healthy-associated on right.
    Within groups: hierarchical clustering.
    Direction labels above the x-axis.

    balanced=True  → top_n_per_ct up + top_n_per_ct down per cell type
    balanced=False → top_n_per_ct overall by |delta_mean| per cell type
    """
    sns.set_theme(style="white", font_scale=0.9)

    ct_names, tf_scores, tf_deltas = _collect_sig_tfs(
        results, top_n=top_n_per_ct, min_abs_delta=min_abs_delta,
        balanced=balanced, fdr_threshold=fdr_threshold
    )
    if not ct_names:
        return
    if not tf_scores:
        print(f"    [skip summary heatmap] no TFs pass FDR < {fdr_threshold}")
        return

    all_tfs = list(tf_scores.keys())

    def _mean_delta(tf: str) -> float:
        vals = list(tf_deltas[tf].values())
        return float(np.nanmean(vals)) if vals else 0.0

    # ---- build delta matrix for clustering (shape: n_tfs × n_cts) -------
    n_tfs_all = len(all_tfs)
    n_cts     = len(ct_names)
    delta_tmp = np.full((n_tfs_all, n_cts), np.nan)
    tf_idx_tmp = {tf: i for i, tf in enumerate(all_tfs)}
    ct_idx     = {ct: j for j, ct in enumerate(ct_names)}
    for tf in all_tfs:
        for ct, val in tf_deltas[tf].items():
            delta_tmp[tf_idx_tmp[tf], ct_idx[ct]] = val

    up_tfs   = [t for t in all_tfs if _mean_delta(t) >= 0]
    down_tfs = [t for t in all_tfs if _mean_delta(t) <  0]
    up_tfs   = _hclust_order(up_tfs,   delta_tmp, tf_idx_tmp)
    down_tfs = _hclust_order(down_tfs, delta_tmp, tf_idx_tmp)
    ordered_tfs = up_tfs + down_tfs

    # ---- final matrices (shape: n_tfs × n_cts) --------------------------
    n_tfs = len(ordered_tfs)
    delta_mat   = np.full((n_tfs, n_cts), np.nan)
    neglogq_mat = np.full((n_tfs, n_cts), np.nan)
    tf_idx = {tf: i for i, tf in enumerate(ordered_tfs)}

    for tf in ordered_tfs:
        for ct, d in tf_deltas[tf].items():
            delta_mat[tf_idx[tf], ct_idx[ct]] = d
        for ct, q in tf_scores[tf].items():
            neglogq_mat[tf_idx[tf], ct_idx[ct]] = q

    # ---- transpose so rows=cell types, cols=TFs -------------------------
    # delta_mat_T shape: (n_cts, n_tfs)
    delta_mat_T   = delta_mat.T
    neglogq_mat_T = neglogq_mat.T

    # ---- size scaling for dots (−log10 q) --------------------------------
    max_neglogq = np.nanmax(neglogq_mat)
    max_neglogq = max_neglogq if max_neglogq > 0 else 1.0
    size_mat_T = 300 * (neglogq_mat_T / max_neglogq)

    # ---- plot -----------------------------------------------------------
    fig_w = max(6, 0.5 * n_tfs + 3.0)
    fig_h = max(4, 0.5 * n_cts + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = plt.cm.RdBu_r
    cmap.set_bad("lightgray")
    vmax = float(np.nanmax(np.abs(delta_mat)))
    vmax = vmax if vmax > 0 else 1.0

    masked = np.ma.masked_invalid(delta_mat_T)
    im = ax.pcolormesh(masked, cmap=cmap, vmin=-vmax, vmax=vmax)

    # Overlay dots for −log10(q) significance
    # Rows=cell types (j), cols=TFs (i)
    xs, ys, sizes_vals = [], [], []
    for i in range(n_tfs):   # x = TF index
        for j in range(n_cts):   # y = cell type index
            if not np.isnan(neglogq_mat_T[j, i]):
                xs.append(i + 0.5)
                ys.append(j + 0.5)
                sizes_vals.append(size_mat_T[j, i])
    ax.scatter(xs, ys, s=sizes_vals, color="black", alpha=0.55, linewidths=0)

    # ---- colorbar -------------------------------------------------------
    cbar = plt.colorbar(im, ax=ax, shrink=0.45, pad=0.02)
    cbar.ax.set_title("Δ AUCell\n(disease − ctrl)", fontsize=7, pad=4)

    # ---- size legend (q thresholds) ------------------------------------
    q_levels = [0.05, 0.01, 0.001]
    size_handles = [
        plt.scatter([], [], s=300 * (-np.log10(q) / max_neglogq),
                    color="black", alpha=0.55,
                    label=f"q < {q}")
        for q in q_levels
    ]
    ax.legend(handles=size_handles, title="Adjusted p-value", fontsize=7,
              title_fontsize=7, loc="lower right", framealpha=0.8)

    # ---- axes: x = TFs, y = cell types ----------------------------------
    ax.set_xticks(np.arange(n_tfs) + 0.5)
    ax.set_xticklabels(ordered_tfs, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(n_cts) + 0.5)
    ax.set_yticklabels(ct_names, fontsize=8)
    ax.set_xlim(0, n_tfs)
    ax.set_ylim(0, n_cts)

    # ---- group separator (vertical) + top labels -----------------------
    if up_tfs and down_tfs:
        sep_x = len(up_tfs)
        ax.axvline(sep_x, color="black", lw=1.2, linestyle="--")
        up_frac   = (len(up_tfs)   / 2) / n_tfs
        down_frac = (len(up_tfs) + len(down_tfs) / 2) / n_tfs
        _add_top_label(ax, up_frac,   "↑ disease", COLOR_UP)
        _add_top_label(ax, down_frac, "↓ disease", COLOR_DOWN)
    elif up_tfs:
        _add_top_label(ax, 0.5, "↑ disease", COLOR_UP)
    else:
        _add_top_label(ax, 0.5, "↓ disease", COLOR_DOWN)

    ax.set_title(title, fontsize=11, fontweight="bold", pad=20)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        plt.savefig(outfile.with_suffix(f".{ext}"), bbox_inches="tight", dpi=150)
    plt.close()
    plt.rcdefaults()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all",
                        choices=list(DATASETS.keys()) + ["all"])
    parser.add_argument("--top_n", type=int, default=30,
                        help="Number of top TFs to show in dotplot")
    parser.add_argument("--min_abs_delta", type=float, default=0.0,
                        help="Minimum |delta mean AUCell| to pre-filter TFs before pseudobulk OLS (0 = no filter)")
    parser.add_argument("--no-metacell", dest="metacell", action="store_false", default=True,
                        help="Skip metacell-level DE (default: run it)")
    args = parser.parse_args()

    to_process = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    for name in to_process:
        cfg = DATASETS[name]
        disease_key = cfg.get("disease_key")
        if not disease_key:
            print(f"\n[skip] {name}: no disease_key in config")
            continue

        print(f"\n{'='*60}")
        print(f"DE TF activity: {name}")
        print(f"{'='*60}")

        # --- Load flashSCENIC embeddings ---
        auc_path = EMBEDDINGS_DIR / f"{name}_flashscenic.npy"
        reg_path = EMBEDDINGS_DIR / f"{name}_flashscenic_regulon_names.npy"
        prep_path = DATA_DIR / f"preprocessed_{name}.h5ad"

        for p in (auc_path, reg_path, prep_path):
            if not p.exists():
                print(f"  [skip] {p.name} not found")
                break
        else:
            pass  # all files exist, continue

        if not all(p.exists() for p in (auc_path, reg_path, prep_path)):
            continue

        auc_scores    = np.load(auc_path)                              # (n_cells, n_regs)
        regulon_names = np.load(reg_path, allow_pickle=True).tolist()  # [str, ...]
        adata         = ad.read_h5ad(prep_path, backed="r")
        obs           = adata.obs.copy()
        adata.file.close()

        ct_key    = cfg["cell_type_key"]
        donor_key = cfg.get("batch_key")
        sex_key   = cfg.get("sex_key")

        print(f"  AUCell: {auc_scores.shape[0]:,} cells × {auc_scores.shape[1]:,} regulons")
        print(f"  Disease key: '{disease_key}' — "
              f"{obs[disease_key].value_counts().to_dict()}")

        # --- Subsets to test: all cells + cell types with enough cells ---
        subsets = {"all": np.arange(len(obs))}
        for ct in sorted(obs[ct_key].unique()):
            idx = np.where((obs[ct_key] == ct).values)[0]
            if len(idx) < MIN_CELLS_PER_CT:
                print(f"  [skip CT] {ct!r}: {len(idx):,} cells < {MIN_CELLS_PER_CT} threshold")
                continue
            subsets[ct] = idx

        # --- Run DE methods ---
        # MWU: cell-level, inflated significance (pseudoreplication).
        # Metacell MWU: bootstrap metacells per donor — intermediate between
        #   cell-level (pseudoreplication) and pseudobulk (underpowered).
        # Pseudobulk OLS: donor-level, correct but underpowered at n=12 donors.
        methods_to_run = [
            ("mwu",
             lambda sc, ob: run_de(sc, ob, disease_key, regulon_names),
             0.05,
             True,
             0.0),              # no delta filter for MWU
            ("pseudobulk",
             lambda sc, ob: run_de_pseudobulk(
                 sc, ob, disease_key, donor_key, regulon_names,
                 sex_key=sex_key,
                 min_abs_delta=args.min_abs_delta),
             0.20,
             False,
             args.min_abs_delta),  # delta filter for pseudobulk only
        ]
        if args.metacell:
            methods_to_run.append((
                "metacell",
                lambda sc, ob: run_de_metacell(
                    sc, ob, disease_key, donor_key, regulon_names,
                    n_metacells=ML["n_metacells"],
                    metacell_size=ML["metacell_size"],
                    expected_reuse=ML.get("expected_reuse", 2.5),
                    delta_weighting=ML.get("de_delta_weighting", "donor"),
                ),
                0.05,
                True,
                0.0,
            ))

        for method_name, de_fn, fdr_threshold, make_summary, method_delta in methods_to_run:
            print(f"\n  --- Method: {method_name} ---")

            metrics_sub = METRICS_DIR / method_name
            metrics_sub.mkdir(parents=True, exist_ok=True)
            de_fig_dir = FIGURES_DIR / name / f"de_tfs_{method_name}"
            de_fig_dir.mkdir(parents=True, exist_ok=True)

            ct_results: dict[str, pd.DataFrame] = {}

            for subset_name, idx in subsets.items():
                safe_name = subset_name.replace("/", "_").replace(" ", "_")
                print(f"\n  Subset: {subset_name!r} ({len(idx):,} cells)")

                de_df = de_fn(auc_scores[idx], obs.iloc[idx])
                if de_df is None:
                    continue

                ct_results[subset_name] = de_df

                csv_out = metrics_sub / f"de_tfs_{name}_{safe_name}.csv"
                de_df.to_csv(csv_out, index=False)
                print(f"    Saved {csv_out.name}")

                fig_out = de_fig_dir / f"de_tfs_{name}_{safe_name}.pdf"
                dotplot_de(
                    de_df,
                    title=f"DE TFs — {subset_name}",
                    outfile=fig_out,
                    top_n=args.top_n,
                    min_abs_delta=method_delta,
                    fdr_threshold=fdr_threshold,
                )
                print(f"    Saved de_tfs_{name}_{safe_name}.pdf/.png")

                # Extra: per-cell-type plot ordered by Δ (metacell only — consistent with summary heatmap)
                if subset_name != "all" and method_name == "metacell":
                    fig_out_delta = de_fig_dir / f"de_tfs_{name}_{safe_name}_by_delta.pdf"
                    dotplot_de_by_delta(
                        de_df,
                        title=f"DE TFs — {subset_name} (by Δ)",
                        outfile=fig_out_delta,
                        top_n=10,
                        fdr_threshold=fdr_threshold,
                    )
                    print(f"    Saved de_tfs_{name}_{safe_name}_by_delta.pdf/.png")

            # --- Summary figures (MWU only; pseudobulk too underpowered) ---
            if make_summary and len([k for k in ct_results if k != "all"]) >= 2:
                fig_dir = FIGURES_DIR / name
                fig_dir.mkdir(parents=True, exist_ok=True)

                print(f"\n  Building summary dotplot ({method_name}) …")
                summary_out = fig_dir / f"de_tfs_{name}_{method_name}_summary.pdf"
                dotplot_summary(
                    ct_results,
                    title="DE TFs across cell types",
                    outfile=summary_out,
                    top_n_per_ct=5,
                    min_abs_delta=method_delta,
                    fdr_threshold=fdr_threshold,
                )
                print(f"    Saved {summary_out.name[:-4]}.pdf/.png")

                print(f"\n  Building summary heatmap ({method_name}, top 10 by |Δ|) …")
                heatmap_out = fig_dir / f"de_tfs_{name}_{method_name}_summary_heatmap.pdf"
                heatmap_summary(
                    ct_results,
                    title="DE TF Δ activity (top 10)",
                    outfile=heatmap_out,
                    top_n_per_ct=10,
                    min_abs_delta=method_delta,
                    balanced=False,
                    fdr_threshold=fdr_threshold,
                )
                print(f"    Saved {heatmap_out.name[:-4]}.pdf/.png")

                print(f"\n  Building summary heatmap ({method_name}, 5 up + 5 down) …")
                heatmap_bal_out = fig_dir / f"de_tfs_{name}_{method_name}_summary_heatmap_balanced.pdf"
                heatmap_summary(
                    ct_results,
                    title="DE TF Δ activity (5 up + 5 down)",
                    outfile=heatmap_bal_out,
                    top_n_per_ct=5,
                    min_abs_delta=method_delta,
                    balanced=True,
                    fdr_threshold=fdr_threshold,
                )
                print(f"    Saved {heatmap_bal_out.name[:-4]}.pdf/.png")

                # Combined figure of per-cell-type by-delta plots
                pngs_delta = sorted(
                    p for p in de_fig_dir.glob(f"de_tfs_{name}_*_by_delta.png")
                    if "_combined" not in p.stem
                )
                if pngs_delta:
                    print(f"\n  Building combined by-Δ panel ({method_name}) …")
                    combined_delta_out = de_fig_dir / f"de_tfs_{name}_by_delta_combined.png"
                    _stitch_pngs(
                        pngs_delta, combined_delta_out, n_cols=3,
                        title="Differential TF Activity per Cell Type (by Δ)",
                    )

    print("\nDE TF analysis done.")


if __name__ == "__main__":
    main()
