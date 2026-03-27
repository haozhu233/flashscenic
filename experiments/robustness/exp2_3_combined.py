"""
Experiments 2 & 3 (combined): Regulon set convergence and AUCell stability

Runs the per-k loop once and saves three figures:
  - exp2_jaccard_convergence.pdf  — Jaccard similarity of regulon sets
  - exp3_aucell_stability.pdf     — Mean Spearman r of per-cell AUCell scores
  - exp_regulon_counts.pdf        — Number of regulons detected at each k

Five strategies compared at each k = 2..n_runs:
  - mean-adj:           average k adj matrices, run Steps 2–5 once (no filter)
  - mean-adj+CV=1.0:    same, zero edges with CV >= 1.0
  - mean-adj+CV=0.5:    same, zero edges with CV >= 0.5 (strict)
  - freq-baseline-50%:  pySCENIC-style, keep TF-gene pairs in >= 50% of runs
  - freq-baseline-75%:  pySCENIC-style, keep TF-gene pairs in >= 75% of runs

AUCell scores for mean-adj strategies come directly from run_flashscenic results
computed for the Jaccard metrics — no extra pipeline calls.
AUCell for freq-baseline adds only a get_aucell() call (no cisTarget rerun).

Usage
-----
    python exp2_3_combined.py --h5ad data.h5ad --n_runs 10 \\
        --species human --output_dir ./figures
"""

import sys
sys.path.insert(0, "/orcd/data/omarabu/001/gonzalo/flashscenic_2/flashscenic")

import argparse
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from scipy.stats import spearmanr
from flashscenic.pipeline import run_flashscenic
from flashscenic.multi_run import multi_run_flashscenic
from flashscenic.data import download_data
from flashscenic.aucell import get_aucell
from flashscenic import regulons_to_adjacency


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", required=True, help="Path to .h5ad file")
    parser.add_argument("--output_dir", default="./figures")
    parser.add_argument("--n_runs", type=int, default=10)
    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--species", default="human")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--freq_threshold", type=float, default=0.5,
                        help="Lenient frequency threshold for pySCENIC-style baseline")
    parser.add_argument("--freq_threshold_strict", type=float, default=0.75,
                        help="Strict frequency threshold for pySCENIC-style baseline")
    parser.add_argument("--cv_threshold", type=float, default=1.0,
                        help="Lenient CV threshold for mean-adj+CV strategy")
    parser.add_argument("--cv_threshold_strict", type=float, default=0.5,
                        help="Strict CV threshold for mean-adj+CV strategy")
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


# ── Jaccard helpers ───────────────────────────────────────────────────────────

def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    union = len(set_a | set_b)
    return len(set_a & set_b) / union if union > 0 else 0.0


def regulon_set(result):
    """Set of (TF, gene) pairs from a run_flashscenic result."""
    pairs = set()
    for reg in result['regulons']:
        for gene in reg['genes']:
            pairs.add((reg['tf'], gene))
    return pairs


def freq_baseline_set(full_run_results, freq_threshold):
    """pySCENIC-style TF-gene pair set from k full runs."""
    n = len(full_run_results)
    tf_gene_counts = defaultdict(int)
    tf_counts = defaultdict(int)
    for result in full_run_results:
        for reg in result['regulons']:
            tf = reg['tf']
            tf_counts[tf] += 1
            for gene in reg['genes']:
                tf_gene_counts[(tf, gene)] += 1
    pairs = set()
    for (tf, gene), count in tf_gene_counts.items():
        tf_n = tf_counts[tf]
        if tf_n / n >= freq_threshold and count / tf_n >= freq_threshold:
            pairs.add((tf, gene))
    return pairs


# ── AUCell helpers ────────────────────────────────────────────────────────────

def freq_baseline_regulons(full_run_results, gene_names, freq_threshold):
    """Consensus regulon list + adjacency matrix for AUCell scoring."""
    n = len(full_run_results)
    tf_gene_counts = defaultdict(int)
    tf_counts = defaultdict(int)
    tf_meta = {}
    for result in full_run_results:
        for reg in result['regulons']:
            tf = reg['tf']
            tf_counts[tf] += 1
            if tf not in tf_meta:
                tf_meta[tf] = reg
            for gene in reg['genes']:
                tf_gene_counts[(tf, gene)] += 1
    consensus_regulons = []
    for tf, tf_n in tf_counts.items():
        if tf_n / n < freq_threshold:
            continue
        genes = [gene for (t, gene), cnt in tf_gene_counts.items()
                 if t == tf and cnt / tf_n >= freq_threshold]
        if not genes:
            continue
        meta = tf_meta[tf]
        consensus_regulons.append({
            'name': f"{tf}(+)", 'tf': tf,
            'motif': meta.get('motif', ''), 'n_genes': len(genes),
            'genes': genes, 'nes': meta.get('nes', 0.0),
            'auc': meta.get('auc', 0.0), 'context': meta.get('context', ''),
            'database': meta.get('database', ''),
        })
    if not consensus_regulons:
        return None, []
    return consensus_regulons, regulons_to_adjacency(consensus_regulons, gene_names)


def mean_spearman(auc_prev, auc_curr, names_prev, names_curr):
    """Mean Spearman r over regulons present in both k-1 and k results."""
    common = set(names_prev) & set(names_curr)
    if not common:
        return float("nan")
    prev_idx = {n: i for i, n in enumerate(names_prev)}
    curr_idx = {n: i for i, n in enumerate(names_curr)}
    corrs = []
    for name in common:
        r, _ = spearmanr(auc_prev[:, prev_idx[name]], auc_curr[:, curr_idx[name]])
        if not np.isnan(r):
            corrs.append(r)
    return float(np.mean(corrs)) if corrs else float("nan")


def apply_cv_mask(A_stack_k, cv_threshold):
    """Return CV-filtered mean adjacency matrix from a stack of k matrices."""
    A_mean_k = A_stack_k.mean(axis=0)
    A_std_k  = A_stack_k.std(axis=0)
    A_cv_k   = np.divide(A_std_k, A_mean_k,
                         out=np.zeros_like(A_mean_k), where=A_mean_k > 0)
    return np.where(A_cv_k < cv_threshold, A_mean_k, 0.0).astype(np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.h5ad}...")
    adata = sc.read_h5ad(args.h5ad)
    exp_matrix = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
    gene_names = list(adata.var_names)
    exp_float32 = np.asarray(exp_matrix, dtype=np.float32)
    print(f"  {exp_float32.shape[0]} cells × {len(gene_names)} genes")

    print("Preparing resources...")
    resources = download_data(species=args.species)
    shared_pipeline_kwargs = dict(
        tf_list_path=str(resources.tf_list),
        ranking_db_paths=[str(p) for p in resources.ranking_dbs],
        motif_annotation_path=str(resources.motif_annotation),
        device=args.device,
        verbose=False,
    )

    # ── Phase 1: n_runs RegDiffusion runs ────────────────────────────────────
    print(f"Running multi_run_flashscenic ({args.n_runs} runs)...")
    ref_result = multi_run_flashscenic(
        exp_matrix, gene_names, args.species,
        n_runs=args.n_runs,
        cv_threshold=None,
        return_adj_matrices=True,
        grn_n_steps=args.n_steps,
        **shared_pipeline_kwargs,
    )
    adj_matrices = ref_result['adj_matrices']
    ref_set_mean = regulon_set(ref_result)

    # ── Phase 2: CV-filtered reference sets (reuse adj_mean + adj_cv) ────────
    def _cv_ref(cv_thr):
        A = np.where(ref_result['adj_cv'] < cv_thr,
                     ref_result['adj_mean'], 0.0).astype(np.float32)
        r = run_flashscenic(exp_matrix, gene_names, args.species,
                            adj_matrix=A, **shared_pipeline_kwargs)
        return regulon_set(r), len(r['regulons'])

    ref_set_cv,        _ = _cv_ref(args.cv_threshold)
    ref_set_cv_strict, _ = _cv_ref(args.cv_threshold_strict)

    # ── Phase 3: n_runs full pipeline runs for freq-baseline ─────────────────
    print(f"Running full pipeline {args.n_runs} times (freq-baseline)...")
    full_results = []
    for i in range(args.n_runs):
        print(f"  Full run {i + 1}/{args.n_runs}...")
        full_results.append(run_flashscenic(
            exp_matrix, gene_names, args.species, **shared_pipeline_kwargs,
        ))
    ref_set_freq        = freq_baseline_set(full_results, args.freq_threshold)
    ref_set_freq_strict = freq_baseline_set(full_results, args.freq_threshold_strict)

    print(f"\nReference regulon set sizes at k={args.n_runs}:")
    print(f"  mean-adj:              {len(ref_set_mean)} TF-gene pairs")
    print(f"  mean-adj+CV={args.cv_threshold}:      {len(ref_set_cv)} TF-gene pairs")
    print(f"  mean-adj+CV={args.cv_threshold_strict}:      {len(ref_set_cv_strict)} TF-gene pairs")
    print(f"  freq-baseline-{int(args.freq_threshold*100)}%:    {len(ref_set_freq)} TF-gene pairs")
    print(f"  freq-baseline-{int(args.freq_threshold_strict*100)}%:    {len(ref_set_freq_strict)} TF-gene pairs")

    # ── Phase 4: per-k loop ───────────────────────────────────────────────────
    k_values = list(range(2, args.n_runs + 1))

    # Jaccard trackers
    jf_mean,         jc_mean         = [], []
    jf_cv,           jc_cv           = [], []
    jf_cv_strict,    jc_cv_strict    = [], []
    jf_freq,         jc_freq         = [], []
    jf_freq_strict,  jc_freq_strict  = [], []

    # Spearman trackers
    sp_mean, sp_cv, sp_cv_strict, sp_freq, sp_freq_strict = [], [], [], [], []

    # Regulon count trackers
    n_mean, n_cv, n_cv_strict, n_freq, n_freq_strict = [], [], [], [], []

    prev_set   = [None] * 5
    auc_prev   = [None] * 5
    names_prev = [None] * 5

    for k in k_values:
        print(f"  k={k}...")

        A_stack_k = np.stack(adj_matrices[:k], axis=0)
        A_mean_k  = A_stack_k.mean(axis=0).astype(np.float32)

        # ── mean-adj ──────────────────────────────────────────────────────────
        r0 = run_flashscenic(exp_matrix, gene_names, args.species,
                             adj_matrix=A_mean_k, **shared_pipeline_kwargs)
        s0 = regulon_set(r0)
        jf_mean.append(jaccard(s0, ref_set_mean))
        jc_mean.append(jaccard(s0, prev_set[0]) if prev_set[0] is not None else None)
        if auc_prev[0] is not None:
            sp_mean.append(mean_spearman(auc_prev[0], r0['auc_scores'],
                                         names_prev[0], r0['regulon_names']))
        n_mean.append(len(r0['regulons']))
        prev_set[0], auc_prev[0], names_prev[0] = s0, r0['auc_scores'], r0['regulon_names']

        # ── mean-adj+CV (lenient) ─────────────────────────────────────────────
        A_cv = apply_cv_mask(A_stack_k, args.cv_threshold)
        r1 = run_flashscenic(exp_matrix, gene_names, args.species,
                             adj_matrix=A_cv, **shared_pipeline_kwargs)
        s1 = regulon_set(r1)
        jf_cv.append(jaccard(s1, ref_set_cv))
        jc_cv.append(jaccard(s1, prev_set[1]) if prev_set[1] is not None else None)
        if auc_prev[1] is not None:
            sp_cv.append(mean_spearman(auc_prev[1], r1['auc_scores'],
                                       names_prev[1], r1['regulon_names']))
        n_cv.append(len(r1['regulons']))
        prev_set[1], auc_prev[1], names_prev[1] = s1, r1['auc_scores'], r1['regulon_names']

        # ── mean-adj+CV (strict) ──────────────────────────────────────────────
        A_cv_strict = apply_cv_mask(A_stack_k, args.cv_threshold_strict)
        r2 = run_flashscenic(exp_matrix, gene_names, args.species,
                             adj_matrix=A_cv_strict, **shared_pipeline_kwargs)
        s2 = regulon_set(r2)
        jf_cv_strict.append(jaccard(s2, ref_set_cv_strict))
        jc_cv_strict.append(jaccard(s2, prev_set[2]) if prev_set[2] is not None else None)
        if auc_prev[2] is not None:
            sp_cv_strict.append(mean_spearman(auc_prev[2], r2['auc_scores'],
                                              names_prev[2], r2['regulon_names']))
        n_cv_strict.append(len(r2['regulons']))
        prev_set[2], auc_prev[2], names_prev[2] = s2, r2['auc_scores'], r2['regulon_names']

        # ── freq-baseline (lenient) ───────────────────────────────────────────
        s3 = freq_baseline_set(full_results[:k], args.freq_threshold)
        jf_freq.append(jaccard(s3, ref_set_freq))
        jc_freq.append(jaccard(s3, prev_set[3]) if prev_set[3] is not None else None)
        prev_set[3] = s3

        cr3, radj3 = freq_baseline_regulons(full_results[:k], gene_names, args.freq_threshold)
        if cr3:
            auc3       = get_aucell(exp_float32, radj3, k=50, auc_threshold=0.05,
                                    device=args.device, batch_size=32)
            names3     = [r['name'] for r in cr3]
            n_freq.append(len(cr3))
        else:
            auc3, names3 = np.zeros((exp_float32.shape[0], 0), dtype=np.float32), []
            n_freq.append(0)
        if auc_prev[3] is not None:
            sp_freq.append(mean_spearman(auc_prev[3], auc3, names_prev[3], names3))
        auc_prev[3], names_prev[3] = auc3, names3

        # ── freq-baseline (strict) ────────────────────────────────────────────
        s4 = freq_baseline_set(full_results[:k], args.freq_threshold_strict)
        jf_freq_strict.append(jaccard(s4, ref_set_freq_strict))
        jc_freq_strict.append(jaccard(s4, prev_set[4]) if prev_set[4] is not None else None)
        prev_set[4] = s4

        cr4, radj4 = freq_baseline_regulons(full_results[:k], gene_names,
                                            args.freq_threshold_strict)
        if cr4:
            auc4       = get_aucell(exp_float32, radj4, k=50, auc_threshold=0.05,
                                    device=args.device, batch_size=32)
            names4     = [r['name'] for r in cr4]
            n_freq_strict.append(len(cr4))
        else:
            auc4, names4 = np.zeros((exp_float32.shape[0], 0), dtype=np.float32), []
            n_freq_strict.append(0)
        if auc_prev[4] is not None:
            sp_freq_strict.append(mean_spearman(auc_prev[4], auc4, names_prev[4], names4))
        auc_prev[4], names_prev[4] = auc4, names4

    # ── Plotting helpers ──────────────────────────────────────────────────────
    COLORS = {
        "mean":         "#2196F3",   # blue
        "cv":           "#4CAF50",   # green
        "cv_strict":    "#1B5E20",   # dark green
        "freq":         "#FF7043",   # orange
        "freq_strict":  "#B71C1C",   # dark red
    }
    LABELS = {
        "mean":         "Mean-adj (no CV filter)",
        "cv":           f"Mean-adj + CV<{args.cv_threshold}",
        "cv_strict":    f"Mean-adj + CV<{args.cv_threshold_strict}",
        "freq":         f"Freq-baseline ≥{int(args.freq_threshold*100)}%",
        "freq_strict":  f"Freq-baseline ≥{int(args.freq_threshold_strict*100)}%",
    }
    STYLES = {
        "mean": ("o", "-"), "cv": ("^", "-"), "cv_strict": ("v", "-"),
        "freq": ("s", "--"), "freq_strict": ("D", "--"),
    }

    def _plot_curves(ax, k_vals, data_dict, ref_line=None, ref_label=None):
        for key, vals in data_dict.items():
            m, ls = STYLES[key]
            ax.plot(k_vals, vals, marker=m, linestyle=ls,
                    color=COLORS[key], label=LABELS[key], linewidth=2, markersize=5)
        if ref_line is not None:
            ax.axhline(ref_line, color="gray", linewidth=0.8, linestyle=":", alpha=0.7,
                       label=ref_label)

    # ── Figure 1: Jaccard convergence (exp2) ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    _plot_curves(axes[0], k_values, {
        "mean": jf_mean, "cv": jf_cv, "cv_strict": jf_cv_strict,
        "freq": jf_freq, "freq_strict": jf_freq_strict,
    }, ref_line=0.9, ref_label="J = 0.9")
    # Elbow on full proposed method (lenient CV)
    for k_val, j in zip(k_values, jf_cv):
        if j >= 0.9:
            axes[0].axvspan(k_val - 0.5, k_val + 0.5, alpha=0.12, color="green",
                            label=f"Elbow CV<{args.cv_threshold} (J≥0.9)")
            break
    axes[0].set_xlabel("Number of GRN runs (k)")
    axes[0].set_ylabel("Jaccard similarity to final regulon set")
    axes[0].set_title("Convergence to final regulon set")
    axes[0].legend(fontsize=7)
    axes[0].set_ylim(0, 1.05)

    k_consec = k_values[1:]
    _plot_curves(axes[1], k_consec, {
        "mean": jc_mean[1:], "cv": jc_cv[1:], "cv_strict": jc_cv_strict[1:],
        "freq": jc_freq[1:], "freq_strict": jc_freq_strict[1:],
    }, ref_line=0.95, ref_label="J = 0.95")
    axes[1].set_xlabel("Number of GRN runs (k)")
    axes[1].set_ylabel("Jaccard(S_k, S_{k-1})")
    axes[1].set_title("Incremental stability (consecutive runs)")
    axes[1].legend(fontsize=7)
    axes[1].set_ylim(0, 1.05)

    fig.suptitle("Experiment 2: Regulon set convergence across k runs",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    out2 = os.path.join(args.output_dir, "exp2_jaccard_convergence.pdf")
    fig.savefig(out2, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved {out2}")

    # ── Figure 2: AUCell Spearman stability (exp3) ────────────────────────────
    k_plot = k_values[1:]
    fig, ax = plt.subplots(figsize=(9, 5))
    _plot_curves(ax, k_plot, {
        "mean": sp_mean, "cv": sp_cv, "cv_strict": sp_cv_strict,
        "freq": sp_freq, "freq_strict": sp_freq_strict,
    }, ref_line=0.99, ref_label="r = 0.99 plateau")
    ax.set_xlabel("Number of GRN runs (k)")
    ax.set_ylabel("Mean Spearman r (per-cell AUCell scores, consecutive k)")
    ax.set_title("Experiment 3: AUCell score stability across k runs")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out3 = os.path.join(args.output_dir, "exp3_aucell_stability.pdf")
    fig.savefig(out3, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved {out3}")

    # ── Figure 3: Regulon counts vs k ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    _plot_curves(ax, k_values, {
        "mean": n_mean, "cv": n_cv, "cv_strict": n_cv_strict,
        "freq": n_freq, "freq_strict": n_freq_strict,
    })
    ax.set_xlabel("Number of GRN runs (k)")
    ax.set_ylabel("Number of regulons (TFs passing cisTarget pruning)")
    ax.set_title("Regulon count across k runs")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_rc = os.path.join(args.output_dir, "exp_regulon_counts.pdf")
    fig.savefig(out_rc, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved {out_rc}")


if __name__ == "__main__":
    main()
