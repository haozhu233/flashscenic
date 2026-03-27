"""
Experiment 1: Run-to-consensus deviation (Bland-Altman style)

Question: Where does RegDiffusion's stochasticity live — in strong edges,
weak edges, or uniformly?

Plot 1 (left): Scatter of per-edge deviation from consensus vs consensus
weight, with empirical 5th–95th percentile bands per bin.

Plot 2 (right): Histogram of per-edge coefficient of variation (CV = std/mean).

Usage
-----
    python exp1_bland_altman.py --h5ad data.h5ad --n_runs 10 --output_dir ./figures
"""

import sys
sys.path.insert(0, "/orcd/data/omarabu/001/gonzalo/flashscenic_2/flashscenic")

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import torch
import regdiffusion as rd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", required=True, help="Path to .h5ad file")
    parser.add_argument("--output_dir", default="./figures")
    parser.add_argument("--n_runs", type=int, default=10)
    parser.add_argument("--n_steps", type=int, default=1000,
                        help="RegDiffusion training steps")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_scatter_points", type=int, default=500_000,
                        help="Max edges to plot in scatter (sampled if exceeded)")
    parser.add_argument("--n_bins", type=int, default=50,
                        help="Number of bins for percentile bands")
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def run_grn_k_times(exp_float32, n_runs, n_steps, device):
    adj_matrices = []
    for i in range(n_runs):
        print(f"  GRN run {i + 1}/{n_runs}...")
        trainer = rd.RegDiffusionTrainer(exp_float32, n_steps=n_steps, device=device)
        trainer.train()
        adj_matrices.append(trainer.get_adj())
    return adj_matrices


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print(f"Loading {args.h5ad}...")
    adata = sc.read_h5ad(args.h5ad)
    exp_matrix = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
    exp_float32 = np.asarray(exp_matrix, dtype=np.float32)
    n_genes = exp_float32.shape[1]
    print(f"  {exp_float32.shape[0]} cells × {n_genes} genes")

    # Run RegDiffusion k times
    print(f"Running RegDiffusion {args.n_runs} times...")
    adj_matrices = run_grn_k_times(exp_float32, args.n_runs, args.n_steps, args.device)

    # Aggregate
    A_stack = np.stack(adj_matrices, axis=0)   # (k, n_genes, n_genes)
    A_mean = A_stack.mean(axis=0)              # (n_genes, n_genes)
    A_std = A_stack.std(axis=0)
    A_cv = np.divide(A_std, A_mean, out=np.zeros_like(A_mean), where=A_mean > 0)

    # Flatten: only non-zero mean edges are informative
    mask = A_mean > 0
    mean_flat = A_mean[mask]
    cv_flat = A_cv[mask]

    # Deviations: (k, n_nonzero_edges)
    deviations = []
    for A in adj_matrices:
        deviations.append(A[mask] - mean_flat)
    deviations = np.stack(deviations, axis=0)  # (k, n_edges)

    # Sample if too many edges
    n_edges = mean_flat.shape[0]
    if n_edges * args.n_runs > args.max_scatter_points:
        sample_n = args.max_scatter_points // args.n_runs
        rng = np.random.default_rng(0)
        idx = rng.choice(n_edges, size=sample_n, replace=False)
        mean_scatter = mean_flat[idx]
        dev_scatter = deviations[:, idx]
    else:
        mean_scatter = mean_flat
        dev_scatter = deviations

    # Compute empirical 5th–95th percentile bands per bin
    bins = np.percentile(mean_flat, np.linspace(0, 100, args.n_bins + 1))
    bins = np.unique(bins)
    bin_centers, p05, p95 = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = (mean_flat >= lo) & (mean_flat < hi)
        if in_bin.sum() < 5:
            continue
        dev_bin = deviations[:, in_bin].ravel()
        bin_centers.append((lo + hi) / 2)
        p05.append(np.percentile(dev_bin, 5))
        p95.append(np.percentile(dev_bin, 95))
    bin_centers = np.array(bin_centers)
    p05, p95 = np.array(p05), np.array(p95)

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Bland-Altman scatter
    ax = axes[0]
    x_all = np.tile(mean_scatter, args.n_runs)
    y_all = dev_scatter.ravel()
    ax.scatter(x_all, y_all, s=0.5, alpha=0.2, color="steelblue", rasterized=True)
    ax.fill_between(bin_centers, p05, p95, alpha=0.35, color="tomato",
                    label="5th–95th pct band")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Consensus edge weight (mean across runs)")
    ax.set_ylabel("Deviation from consensus (run − mean)")
    ax.set_title(f"Run-to-consensus deviation\n(n_runs={args.n_runs})")
    ax.legend(fontsize=8)

    # Panel B: CV histogram
    ax = axes[1]
    ax.hist(cv_flat, bins=80, color="steelblue", edgecolor="none", alpha=0.8)
    ax.axvline(1.0, color="tomato", linewidth=1.5, linestyle="--",
               label="CV=1.0 (default threshold)")
    ax.set_xlabel("Coefficient of variation (std / mean)")
    ax.set_ylabel("Number of edges")
    ax.set_title("Edge-weight CV distribution")
    ax.legend(fontsize=8)
    frac_above = (cv_flat >= 1.0).mean()
    ax.text(0.98, 0.95, f"{frac_above:.1%} of edges\nhave CV ≥ 1.0",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    fig.suptitle("Experiment 1: RegDiffusion stochasticity in the adjacency matrix",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    out_path = os.path.join(args.output_dir, "exp1_bland_altman.pdf")
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
