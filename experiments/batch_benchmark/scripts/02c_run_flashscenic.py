"""
Step 2c: flashSCENIC AUCell scoring.

Runs the full flashSCENIC pipeline (GRN → modules → cisTarget → AUCell)
on log-normalized HVG expression from the preprocessed dataset.

No batch information is passed — batch correction is implicit via the
regulon aggregation, not explicit.

Output: obsm["X_flashscenic"] of shape (n_cells, n_regulons)

Usage:
    python 02c_run_flashscenic.py [--dataset immune_human|pancreas|all]
                                  [--subsample N]
                                  [--cache_dir PATH]
"""

import argparse
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent))
from config import DATASETS, FLASHSCENIC as FS_CFG, DATA_DIR, EMBEDDINGS_DIR


def run_flashscenic(name: str, cfg: dict, subsample: int = None,
                    cache_dir: str = None, force: bool = False) -> None:
    preprocessed_path = DATA_DIR / f"preprocessed_{name}.h5ad"
    if not preprocessed_path.exists():
        raise FileNotFoundError(
            f"{preprocessed_path} not found. Run 01_preprocess.py first."
        )

    fs_path = EMBEDDINGS_DIR / f"{name}_flashscenic.npy"
    regulon_names_path = EMBEDDINGS_DIR / f"{name}_flashscenic_regulon_names.npy"

    if fs_path.exists() and regulon_names_path.exists() and not force:
        print(f"  [skip] flashscenic embedding already exists for {name}")
        return

    print(f"\n{'='*60}")
    print(f"Running flashSCENIC: {name}")
    print(f"{'='*60}")

    try:
        import flashscenic as fs
    except ImportError:
        raise ImportError(
            "flashscenic not installed. Install from the repo root: pip install -e ."
        )

    adata = ad.read_h5ad(preprocessed_path)
    print(f"  Loaded: {adata.shape}")

    # Optionally subsample to manage GPU memory / runtime on very large datasets
    if subsample is not None and adata.n_obs > subsample:
        print(f"  Subsampling: {adata.n_obs:,} → {subsample:,} cells")
        rng = np.random.default_rng(FS_CFG["seed"])
        idx = rng.choice(adata.n_obs, size=subsample, replace=False)
        idx.sort()
        adata_sub = adata[idx].copy()
    else:
        adata_sub = adata
        idx = None

    # Use HVG subset
    if "highly_variable" in adata_sub.var.columns:
        hvg_mask = adata_sub.var.highly_variable.values
        gene_names = list(adata_sub.var_names[hvg_mask])
        print(f"  Using {len(gene_names):,} HVGs")
    else:
        hvg_mask = np.ones(adata_sub.n_vars, dtype=bool)
        gene_names = list(adata_sub.var_names)
        print(f"  Using all {len(gene_names):,} genes (no HVG mask found)")

    # Extract log-normalized expression matrix
    log_norm = adata_sub.layers["normalized_log"]
    if sp.issparse(log_norm):
        log_norm = log_norm.toarray()
    exp_matrix = log_norm[:, hvg_mask].astype(np.float32)

    # Remove zero-variance genes (RegDiffusion requires all genes to have variance > 0)
    gene_var = exp_matrix.var(axis=0)
    nonzero_var = gene_var > 0
    n_zero_var = (~nonzero_var).sum()
    if n_zero_var > 0:
        print(f"  Removing {n_zero_var} zero-variance genes before GRN inference")
        exp_matrix = exp_matrix[:, nonzero_var]
        gene_names = [g for g, keep in zip(gene_names, nonzero_var) if keep]

    print(f"  Expression matrix: {exp_matrix.shape} (cells x HVGs)")

    t0 = time.time()
    result = fs.run_flashscenic(
        exp_matrix=exp_matrix,
        gene_names=gene_names,
        species=cfg["species"],
        datasource=FS_CFG["datasource"],
        version=FS_CFG["version"],
        cache_dir=cache_dir,
        grn_n_steps=FS_CFG["grn_n_steps"],
        grn_sparsity_threshold=FS_CFG["grn_sparsity_threshold"],
        module_k=FS_CFG["module_k"],
        module_percentile_thresholds=FS_CFG["module_percentile_thresholds"],
        module_top_n_per_target=FS_CFG["module_top_n_per_target"],
        module_min_targets=FS_CFG["module_min_targets"],
        aucell_batch_size=FS_CFG["aucell_batch_size"],
        device=FS_CFG["device"],
        seed=FS_CFG["seed"],
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"  flashSCENIC complete in {elapsed/60:.1f} min")

    auc_scores = result["auc_scores"]   # (n_cells_sub, n_regulons)
    regulon_names = result["regulon_names"]
    print(f"  AUCell scores: {auc_scores.shape}  ({len(regulon_names)} regulons)")

    # If we subsampled, we need full-dataset scores via a second AUCell pass
    if idx is not None:
        print(f"  Subsampled run — scoring full dataset with learned regulons...")
        from flashscenic.aucell import get_aucell
        from flashscenic import regulons_to_adjacency

        regulon_adj = result["regulon_adj"]  # (n_regulons, n_genes_hvg)

        # Full dataset expression
        full_log_norm = adata.layers["normalized_log"]
        if sp.issparse(full_log_norm):
            full_log_norm = full_log_norm.toarray()
        full_exp = full_log_norm[:, hvg_mask].astype(np.float32)

        auc_scores = get_aucell(
            full_exp,
            regulon_adj,
            k=FS_CFG["module_k"],
            auc_threshold=0.05,
            device=FS_CFG["device"],
            batch_size=FS_CFG["aucell_batch_size"],
            seed=FS_CFG["seed"],
        )
        print(f"  Full-dataset AUCell scores: {auc_scores.shape}")

    # Save
    np.save(fs_path, auc_scores)
    np.save(regulon_names_path, np.array(regulon_names))
    print(f"  Saved embeddings to {fs_path.name}")
    print(f"  Saved regulon names to {regulon_names_path.name}")

    print(f"  Embedding saved as .npy — h5ad will be merged by 02d_merge_embeddings.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all",
                        choices=list(DATASETS.keys()) + ["all"])
    parser.add_argument(
        "--cache_dir", type=str, default=None,
        help="Directory for flashSCENIC resource cache (TF lists, ranking DBs).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rerun even if embedding .npy files already exist.",
    )
    args = parser.parse_args()

    subsample = FS_CFG.get("grn_subsample", None)
    if subsample is not None:
        print(f"GRN subsample: {subsample:,} cells (set in config.py FLASHSCENIC['grn_subsample'])")
    else:
        print("GRN subsample: disabled (full dataset)")

    to_process = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    for name in to_process:
        run_flashscenic(name, DATASETS[name],
                        subsample=subsample,
                        cache_dir=args.cache_dir,
                        force=args.force)

    print("\nflashSCENIC done.")


if __name__ == "__main__":
    main()
