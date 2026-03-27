"""
Experiment 2: Regulon set convergence across k runs

Question: How many runs are needed for the consensus regulon set to stabilise,
and does aggregating the adjacency matrix converge faster than pySCENIC-style
post-pruning frequency aggregation?

Three strategies compared:
  - mean-adj:      average the first k GRN adjacency matrices, run Steps 2–5 once
  - mean-adj+CV:   same as mean-adj but zero out edges with CV >= cv_threshold
                   before passing to Steps 2–5 (full proposed method)
  - freq-baseline: run the full pipeline k times independently, keep TF-gene
                   pairs that appear in ≥50% of those runs (pySCENIC-style)

For each k = 2..n_runs, two metrics are computed:
  - Jaccard(S_k, S_{n_runs}): convergence to final answer
  - Jaccard(S_k, S_{k-1}): incremental stability (consecutive)

The elbow in Jaccard(S_k, S_{n_runs}) for the proposed method gives the
recommended default n_runs.

Usage
-----
    python exp2_jaccard_convergence.py --h5ad data.h5ad --n_runs 10 \\
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
from flashscenic.pipeline import run_flashscenic
from flashscenic.data import download_data
from flashscenic.multi_run import multi_run_flashscenic


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", required=True, help="Path to .h5ad file")
    parser.add_argument("--output_dir", default="./figures")
    parser.add_argument("--n_runs", type=int, default=10)
    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--species", default="human")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--freq_threshold", type=float, default=0.5,
                        help="Gene/TF frequency threshold for baseline strategy")
    parser.add_argument("--cv_threshold", type=float, default=1.0,
                        help="CV threshold for mean-adj+CV strategy")
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def regulon_set(result):
    """Return set of (TF, gene) pairs from a run_flashscenic result."""
    pairs = set()
    for reg in result['regulons']:
        tf = reg['tf']
        for gene in reg['genes']:
            pairs.add((tf, gene))
    return pairs


def freq_baseline_set(full_run_results, freq_threshold):
    """
    pySCENIC-style: keep (TF, gene) pairs appearing in >= freq_threshold
    fraction of full runs.
    """
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
        tf_n = tf_counts[tf]  # runs where TF appeared
        if tf_n / n >= freq_threshold and count / tf_n >= freq_threshold:
            pairs.add((tf, gene))
    return pairs


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

    # ---- Run multi_run_flashscenic once to collect adj matrices + raw mean ref ----
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

    # ---- Mean-adj+CV reference: apply CV mask to the full-run consensus ----
    A_mean_full = ref_result['adj_mean']
    A_cv_full = ref_result['adj_cv']
    A_consensus_cv = np.where(
        A_cv_full < args.cv_threshold, A_mean_full, 0.0
    ).astype(np.float32)
    ref_result_cv = run_flashscenic(
        exp_matrix, gene_names, args.species,
        adj_matrix=A_consensus_cv,
        **shared_pipeline_kwargs,
    )
    ref_set_mean_cv = regulon_set(ref_result_cv)

    # ---- Freq-baseline strategy: n_runs independent full pipeline runs ----
    print(f"Running full pipeline {args.n_runs} times (freq-baseline strategy)...")
    full_results = []
    for i in range(args.n_runs):
        print(f"  Full run {i + 1}/{args.n_runs}...")
        result = run_flashscenic(
            exp_matrix, gene_names, args.species,
            **shared_pipeline_kwargs,
        )
        full_results.append(result)
    ref_set_freq = freq_baseline_set(full_results, args.freq_threshold)

    print(f"\nReference regulon set sizes at k={args.n_runs}:")
    print(f"  mean-adj:        {len(ref_set_mean)} TF-gene pairs")
    print(f"  mean-adj+CV:     {len(ref_set_mean_cv)} TF-gene pairs")
    print(f"  freq-baseline:   {len(ref_set_freq)} TF-gene pairs")

    # ---- Compute Jaccard curves for k = 2..n_runs ----
    k_values = list(range(2, args.n_runs + 1))

    jaccard_to_final_mean,    jaccard_consec_mean    = [], []
    jaccard_to_final_mean_cv, jaccard_consec_mean_cv = [], []
    jaccard_to_final_freq,    jaccard_consec_freq    = [], []

    prev_set_mean    = None
    prev_set_mean_cv = None
    prev_set_freq    = None

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
        set_k_mean = regulon_set(res_k)
        jaccard_to_final_mean.append(jaccard(set_k_mean, ref_set_mean))
        jaccard_consec_mean.append(
            jaccard(set_k_mean, prev_set_mean) if prev_set_mean is not None else None
        )
        prev_set_mean = set_k_mean

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
        set_k_mean_cv = regulon_set(res_k_cv)
        jaccard_to_final_mean_cv.append(jaccard(set_k_mean_cv, ref_set_mean_cv))
        jaccard_consec_mean_cv.append(
            jaccard(set_k_mean_cv, prev_set_mean_cv) if prev_set_mean_cv is not None else None
        )
        prev_set_mean_cv = set_k_mean_cv

        # Freq-baseline strategy
        set_k_freq = freq_baseline_set(full_results[:k], args.freq_threshold)
        jaccard_to_final_freq.append(jaccard(set_k_freq, ref_set_freq))
        jaccard_consec_freq.append(
            jaccard(set_k_freq, prev_set_freq) if prev_set_freq is not None else None
        )
        prev_set_freq = set_k_freq

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    colors = {"mean": "#2196F3", "mean_cv": "#4CAF50", "freq": "#FF7043"}
    labels = {
        "mean":    "Mean-adj (no CV filter)",
        "mean_cv": f"Mean-adj + CV filter (cv<{args.cv_threshold})",
        "freq":    "Freq-baseline (pySCENIC-style)",
    }

    # Panel A: convergence to final answer
    ax = axes[0]
    ax.plot(k_values, jaccard_to_final_mean, "o-", color=colors["mean"],
            label=labels["mean"], linewidth=2, markersize=5)
    ax.plot(k_values, jaccard_to_final_mean_cv, "^-", color=colors["mean_cv"],
            label=labels["mean_cv"], linewidth=2, markersize=5)
    ax.plot(k_values, jaccard_to_final_freq, "s--", color=colors["freq"],
            label=labels["freq"], linewidth=2, markersize=5)

    # Shade elbow: first k where the full proposed method (mean+CV) exceeds 0.9
    for k_val, jcv in zip(k_values, jaccard_to_final_mean_cv):
        if jcv >= 0.9:
            ax.axvspan(k_val - 0.5, k_val + 0.5, alpha=0.15, color="green",
                       label="Elbow mean+CV (J ≥ 0.9)")
            break

    ax.set_xlabel("Number of GRN runs (k)")
    ax.set_ylabel("Jaccard similarity to final regulon set")
    ax.set_title("Convergence to final regulon set")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.9, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)

    # Panel B: consecutive stability
    ax = axes[1]
    k_consec = k_values[1:]
    ax.plot(k_consec, [j for j in jaccard_consec_mean[1:]], "o-",
            color=colors["mean"], label=labels["mean"], linewidth=2, markersize=5)
    ax.plot(k_consec, [j for j in jaccard_consec_mean_cv[1:]], "^-",
            color=colors["mean_cv"], label=labels["mean_cv"], linewidth=2, markersize=5)
    ax.plot(k_consec, [j for j in jaccard_consec_freq[1:]], "s--",
            color=colors["freq"], label=labels["freq"], linewidth=2, markersize=5)
    ax.set_xlabel("Number of GRN runs (k)")
    ax.set_ylabel("Jaccard(S_k, S_{k-1})")
    ax.set_title("Incremental stability (consecutive runs)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.95, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)

    fig.suptitle("Experiment 2: Regulon set convergence across k runs",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    out_path = os.path.join(args.output_dir, "exp2_jaccard_convergence.pdf")
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
