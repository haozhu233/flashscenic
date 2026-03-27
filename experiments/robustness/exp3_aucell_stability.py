"""
Experiment 3: AUCell score stability across k runs

Question: Does instability in the regulon gene sets propagate to downstream
biology, and does mean-adjacency aggregation produce more stable AUCell scores
than frequency-based aggregation?

For each k = 2..n_runs and all three strategies, AUCell scores are computed and
the mean Spearman correlation between consecutive k steps is tracked. Convergence
to r ≈ 1.0 indicates stable per-cell activity scores regardless of k.

Three strategies compared:
  - mean-adj:      average of first k GRN adjacency matrices → Steps 2–5 once
  - mean-adj+CV:   same as mean-adj but zero out edges with CV >= cv_threshold
                   before passing to Steps 2–5 (full proposed method)
  - freq-baseline: full pipeline k times, keep TF-gene pairs in ≥50% of runs,
                   then re-score with AUCell

Usage
-----
    python exp3_aucell_stability.py --h5ad data.h5ad --n_runs 10 \\
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
    parser.add_argument("--freq_threshold", type=float, default=0.5)
    parser.add_argument("--cv_threshold", type=float, default=1.0,
                        help="CV threshold for mean-adj+CV strategy")
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def freq_baseline_regulons(full_run_results, gene_names, freq_threshold):
    """
    Build pySCENIC-style consensus regulon list from k full runs, then
    compute AUCell scores.
    """
    n = len(full_run_results)
    tf_gene_counts = defaultdict(int)
    tf_counts = defaultdict(int)
    tf_meta = {}  # store first-seen metadata

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
            'name': f"{tf}(+)",
            'tf': tf,
            'motif': meta.get('motif', ''),
            'n_genes': len(genes),
            'genes': genes,
            'nes': meta.get('nes', 0.0),
            'auc': meta.get('auc', 0.0),
            'context': meta.get('context', ''),
            'database': meta.get('database', ''),
        })

    if not consensus_regulons:
        return None, []

    regulon_adj = regulons_to_adjacency(consensus_regulons, gene_names)
    return consensus_regulons, regulon_adj


def mean_spearman(auc_k_prev, auc_k_curr, regulon_names_prev, regulon_names_curr):
    """
    Compute mean Spearman r between per-cell AUC vectors for regulons
    present in both k-1 and k results.
    """
    common = set(regulon_names_prev) & set(regulon_names_curr)
    if not common:
        return float("nan")

    corrs = []
    prev_idx = {n: i for i, n in enumerate(regulon_names_prev)}
    curr_idx = {n: i for i, n in enumerate(regulon_names_curr)}

    for name in common:
        col_prev = auc_k_prev[:, prev_idx[name]]
        col_curr = auc_k_curr[:, curr_idx[name]]
        r, _ = spearmanr(col_prev, col_curr)
        if not np.isnan(r):
            corrs.append(r)

    return float(np.mean(corrs)) if corrs else float("nan")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print(f"Loading {args.h5ad}...")
    adata = sc.read_h5ad(args.h5ad)
    exp_matrix = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
    gene_names = list(adata.var_names)
    exp_float32 = np.asarray(exp_matrix, dtype=np.float32)
    print(f"  {exp_float32.shape[0]} cells × {len(gene_names)} genes")

    # Download resources once
    print("Preparing resources...")
    resources = download_data(species=args.species)
    tf_list_path = str(resources.tf_list)
    ranking_db_paths = [str(p) for p in resources.ranking_dbs]
    motif_annotation_path = str(resources.motif_annotation)

    shared_pipeline_kwargs = dict(
        tf_list_path=tf_list_path,
        ranking_db_paths=ranking_db_paths,
        motif_annotation_path=motif_annotation_path,
        device=args.device,
        verbose=False,
    )

    # ---- Collect n_runs GRN adjacency matrices via multi_run_flashscenic ----
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

    # ---- Collect n_runs full pipeline results (freq baseline) ----
    print(f"Running full pipeline {args.n_runs} times (freq-baseline)...")
    full_results = []
    for i in range(args.n_runs):
        print(f"  Full run {i + 1}/{args.n_runs}...")
        result = run_flashscenic(
            exp_matrix, gene_names, args.species,
            **shared_pipeline_kwargs,
        )
        full_results.append(result)

    # ---- Compute AUCell results for each k ----
    k_values = list(range(2, args.n_runs + 1))

    auc_mean_prev,    names_mean_prev    = None, None
    auc_mean_cv_prev, names_mean_cv_prev = None, None
    auc_freq_prev,    names_freq_prev    = None, None

    spearman_mean, spearman_mean_cv, spearman_freq = [], [], []

    for k in k_values:
        print(f"  k={k}...")

        A_stack_k = np.stack(adj_matrices[:k], axis=0)
        A_mean_k  = A_stack_k.mean(axis=0).astype(np.float32)

        # Mean-adj (no CV filter)
        res_k = run_flashscenic(
            exp_matrix, gene_names, args.species,
            adj_matrix=A_mean_k,
            **shared_pipeline_kwargs,
        )
        auc_k_mean    = res_k['auc_scores']
        names_k_mean  = res_k['regulon_names']

        if auc_mean_prev is not None:
            spearman_mean.append(mean_spearman(
                auc_mean_prev, auc_k_mean, names_mean_prev, names_k_mean,
            ))
        auc_mean_prev, names_mean_prev = auc_k_mean, names_k_mean

        # Mean-adj + CV filter
        A_std_k = A_stack_k.std(axis=0)
        A_cv_k  = np.divide(A_std_k, A_mean_k,
                            out=np.zeros_like(A_mean_k), where=A_mean_k > 0)
        A_cv_filtered_k = np.where(
            A_cv_k < args.cv_threshold, A_mean_k, 0.0
        ).astype(np.float32)
        res_k_cv = run_flashscenic(
            exp_matrix, gene_names, args.species,
            adj_matrix=A_cv_filtered_k,
            **shared_pipeline_kwargs,
        )
        auc_k_mean_cv   = res_k_cv['auc_scores']
        names_k_mean_cv = res_k_cv['regulon_names']

        if auc_mean_cv_prev is not None:
            spearman_mean_cv.append(mean_spearman(
                auc_mean_cv_prev, auc_k_mean_cv,
                names_mean_cv_prev, names_k_mean_cv,
            ))
        auc_mean_cv_prev, names_mean_cv_prev = auc_k_mean_cv, names_k_mean_cv

        # Freq-baseline strategy
        consensus_regs, regulon_adj_freq = freq_baseline_regulons(
            full_results[:k], gene_names, args.freq_threshold,
        )
        if consensus_regs:
            auc_k_freq = get_aucell(
                exp_float32, regulon_adj_freq,
                k=50, auc_threshold=0.05,
                device=args.device, batch_size=32,
            )
            names_k_freq = [r['name'] for r in consensus_regs]
        else:
            auc_k_freq = np.zeros((exp_float32.shape[0], 0), dtype=np.float32)
            names_k_freq = []

        if auc_freq_prev is not None:
            spearman_freq.append(mean_spearman(
                auc_freq_prev, auc_k_freq, names_freq_prev, names_k_freq,
            ))
        auc_freq_prev, names_freq_prev = auc_k_freq, names_k_freq

    # ---- Plot ----
    k_plot  = k_values[1:]   # consecutive pairs start at k=3
    colors  = {"mean": "#2196F3", "mean_cv": "#4CAF50", "freq": "#FF7043"}
    labels  = {
        "mean":    "Mean-adj (no CV filter)",
        "mean_cv": f"Mean-adj + CV filter (cv<{args.cv_threshold})",
        "freq":    "Freq-baseline (pySCENIC-style)",
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(k_plot, spearman_mean, "o-", color=colors["mean"],
            label=labels["mean"], linewidth=2, markersize=6)
    ax.plot(k_plot, spearman_mean_cv, "^-", color=colors["mean_cv"],
            label=labels["mean_cv"], linewidth=2, markersize=6)
    ax.plot(k_plot, spearman_freq, "s--", color=colors["freq"],
            label=labels["freq"], linewidth=2, markersize=6)
    ax.axhline(0.99, color="gray", linewidth=0.8, linestyle=":",
               alpha=0.7, label="r = 0.99 plateau")

    ax.set_xlabel("Number of GRN runs (k)")
    ax.set_ylabel("Mean Spearman r (per-cell AUCell scores, consecutive k)")
    ax.set_title("Experiment 3: AUCell score stability across k runs")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    fig.tight_layout()

    out_path = os.path.join(args.output_dir, "exp3_aucell_stability.pdf")
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
