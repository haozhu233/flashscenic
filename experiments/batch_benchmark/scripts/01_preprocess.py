"""
Step 1: Preprocessing.

Applies a standard shared preprocessing pipeline to the raw downloaded data:
  1. Cell QC (min genes, max mito fraction)
  2. Gene QC (min cells)
  3. Normalize + log1p
  4. HVG selection (batch-aware)
  5. PCA (for baseline / Harmony)
  6. Save preprocessed AnnData with raw counts stored in .layers["counts"]

Output: data/preprocessed_{dataset_name}.h5ad

Usage:
    python 01_preprocess.py [--dataset immune_human|pancreas|all]
"""

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc

sys.path.insert(0, str(Path(__file__).parent))
from config import DATASETS, PREPROCESS, DATA_DIR, GLOBAL_SEED


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_counts(adata: ad.AnnData, counts_in: str) -> np.ndarray:
    """Return raw count matrix from wherever it lives."""
    if counts_in == "X":
        import scipy.sparse as sp
        return adata.X.copy()
    elif counts_in == "layers['counts']":
        return adata.layers["counts"].copy()
    elif counts_in == "layers['raw_counts']":
        return adata.layers["raw_counts"].copy()
    elif counts_in == "raw":
        return adata.raw.X.copy()
    else:
        raise ValueError(f"Unknown counts_in value: {counts_in!r}")


def _detect_counts_location(adata: ad.AnnData) -> str:
    """
    Determine where raw counts live.
    Returns one of: "X", "layers['counts']", "layers['raw_counts']", "raw"

    Uses atol=0.5 for the integer check so that float32-stored counts
    (common in scran-normalized datasets) are still detected correctly.
    """
    import scipy.sparse as sp

    def _is_count_like(mat):
        if sp.issparse(mat):
            sample = np.asarray(mat[:min(200, mat.shape[0])].todense()).ravel()
        else:
            sample = np.asarray(mat[:min(200, mat.shape[0])]).ravel()
        sample = sample[sample != 0]  # ignore structural zeros
        if len(sample) == 0:
            return False
        return np.all(sample >= 0) and np.allclose(sample, np.round(sample), atol=0.5)

    if _is_count_like(adata.X):
        return "X"
    for layer in ["counts", "raw_counts"]:
        if layer in adata.layers and _is_count_like(adata.layers[layer]):
            return f"layers['{layer}']"
    if adata.raw is not None:
        return "raw"
    # Last resort: if a named counts layer exists, trust it even if values
    # look non-integer (e.g. scran size-factor-normalized stored as 'counts')
    for layer in ["counts", "raw_counts"]:
        if layer in adata.layers:
            print(f"  WARNING: layers['{layer}'] found but values are not integer-like. "
                  f"Using it anyway — scVI may require truly raw counts.")
            return f"layers['{layer}']"
    raise ValueError(
        "Cannot detect raw counts. adata.X appears normalized, no integer "
        "layers found, and adata.raw is None. Check the dataset manually."
    )


def _mito_gene_pattern(species: str) -> str:
    if species == "human":
        return "MT-"
    elif species == "mouse":
        return "mt-"
    return "MT-"


# ---------------------------------------------------------------------------
# Main preprocessing function
# ---------------------------------------------------------------------------

def preprocess(name: str, cfg: dict, pp: dict) -> ad.AnnData:
    path: Path = cfg["path"]
    species: str = cfg["species"]

    print(f"\n{'='*60}")
    print(f"Preprocessing: {name}")
    print(f"{'='*60}")

    adata = ad.read_h5ad(path)
    print(f"  Loaded: {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")

    # ---- 0. Rename var_names to gene symbols if currently Ensembl IDs -------
    if adata.var_names[0].startswith("ENSG") and "feature_name" in adata.var.columns:
        print(f"  var_names are Ensembl IDs — renaming to gene symbols via var['feature_name']")
        adata.var_names = adata.var["feature_name"].astype(str).values
        # Make unique in case of duplicates (rare but possible)
        adata.var_names_make_unique()
        print(f"  var_names now: {list(adata.var_names[:5])}")

    # ---- 1. Store raw counts ------------------------------------------------
    counts_loc = _detect_counts_location(adata)
    print(f"  Raw counts detected in: {counts_loc}")

    if counts_loc == "raw":
        # Align raw genes to current var_names
        raw_adata = adata.raw.to_adata()
        # Subset raw to same genes as adata (may differ)
        common_genes = adata.var_names.intersection(raw_adata.var_names)
        raw_adata = raw_adata[:, common_genes]
        adata = adata[:, common_genes].copy()
        adata.layers["counts"] = raw_adata.X.copy()
    elif counts_loc == "X":
        adata.layers["counts"] = adata.X.copy()
    else:
        # already in layers, rename to "counts" for consistency
        layer_name = counts_loc.replace("layers['", "").replace("']", "")
        if layer_name != "counts":
            adata.layers["counts"] = adata.layers[layer_name].copy()

    # ---- 2. Cell QC ---------------------------------------------------------
    mito_prefix = _mito_gene_pattern(species)
    adata.var["mito"] = adata.var_names.str.startswith(mito_prefix)
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mito"], percent_top=None, log1p=False, inplace=True
    )

    n_before = adata.n_obs
    adata = adata[adata.obs.n_genes_by_counts >= pp["min_genes"]].copy()
    adata = adata[adata.obs.pct_counts_mito <= pp["max_mito_frac"] * 100].copy()
    print(f"  Cell QC: {n_before:,} → {adata.n_obs:,} cells "
          f"(removed {n_before - adata.n_obs:,})")

    # ---- 3. Gene QC ---------------------------------------------------------
    n_genes_before = adata.n_vars
    if pp["n_hvgs"] is None and pp.get("min_cells_frac"):
        min_cells_abs = max(pp["min_cells"],
                            int(np.ceil(pp["min_cells_frac"] * adata.n_obs)))
        print(f"  Gene filter: min_cells={min_cells_abs} "
              f"({pp['min_cells_frac']*100:.1f}% of {adata.n_obs:,} cells)")
    else:
        min_cells_abs = pp["min_cells"]
    sc.pp.filter_genes(adata, min_cells=min_cells_abs)
    print(f"  Gene QC: {n_genes_before:,} → {adata.n_vars:,} genes "
          f"(removed {n_genes_before - adata.n_vars:,})")

    # ---- 3b. Remove zero-variance genes -------------------------------------
    import scipy.sparse as sp
    X = adata.layers["counts"]
    if sp.issparse(X):
        gene_var = np.asarray(X.power(2).mean(axis=0)) - np.asarray(X.mean(axis=0)) ** 2
        gene_var = gene_var.ravel()
    else:
        gene_var = X.var(axis=0)
    nonzero_var = gene_var > 0
    n_zero = (~nonzero_var).sum()
    if n_zero > 0:
        adata = adata[:, nonzero_var].copy()
        print(f"  Zero-variance filter: removed {n_zero:,} genes → {adata.n_vars:,} remaining")

    # ---- 4. Normalize + log1p -----------------------------------------------
    sc.pp.normalize_total(adata, target_sum=pp["target_sum"])
    sc.pp.log1p(adata)
    # Store log-normalized as a layer (needed by flashSCENIC and ML predictor)
    adata.layers["normalized_log"] = adata.X.copy()
    print(f"  Normalized (target_sum={pp['target_sum']:.0f}) + log1p")

    # ---- 5. HVG selection (batch-aware, optional) ---------------------------
    batch_key = cfg["batch_key"]
    batch_available = batch_key in adata.obs.columns

    if pp["n_hvgs"] is not None:
        # Cap n_hvgs at the number of genes available after QC
        n_hvgs = min(pp["n_hvgs"], adata.n_vars)
        if n_hvgs < pp["n_hvgs"]:
            print(f"  HVG: requested {pp['n_hvgs']:,} but only {adata.n_vars:,} genes available "
                  f"— using all {n_hvgs:,} genes")

        if batch_available:
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=n_hvgs,
                batch_key=batch_key,
                flavor="seurat_v3",
                layer="counts",
            )
            print(f"  HVG: {adata.var.highly_variable.sum():,} genes selected "
                  f"(batch-aware, batch_key='{batch_key}')")
        else:
            sc.pp.highly_variable_genes(
                adata, n_top_genes=n_hvgs, flavor="seurat_v3", layer="counts"
            )
            print(f"  HVG: {adata.var.highly_variable.sum():,} genes selected (no batch correction)")
    else:
        print(f"  HVG selection skipped — using all {adata.n_vars:,} filtered genes")

    # ---- 6. PCA -------------------------------------------------------------
    # Scale only for PCA (zero-mean unit-variance), keep normalized_log intact
    if pp["n_hvgs"] is not None and "highly_variable" in adata.var.columns:
        pca_input = adata[:, adata.var.highly_variable].copy()
        print(f"  PCA on {pca_input.n_vars:,} HVGs")
    else:
        pca_input = adata.copy()
        print(f"  PCA on all {pca_input.n_vars:,} filtered genes")
    sc.pp.scale(pca_input, max_value=10)
    sc.pp.pca(pca_input, n_comps=pp["n_pcs"], random_state=GLOBAL_SEED)
    adata.obsm["X_pca"] = pca_input.obsm["X_pca"]
    print(f"  PCA: {pp['n_pcs']} components")
    del pca_input

    # ---- 7. Print label distributions ---------------------------------------
    for key_name, col in [
        ("batch", cfg["batch_key"]),
        ("cell_type", cfg["cell_type_key"]),
        ("sex", cfg["sex_key"]),
    ]:
        if col and col in adata.obs.columns:
            vc = adata.obs[col].value_counts()
            print(f"\n  {key_name} distribution ({col}):")
            print(vc.to_string(max_rows=15))

    print(f"\n  Final adata: {adata.shape}")
    return adata


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess scIB datasets")
    parser.add_argument(
        "--dataset", default="all",
        choices=list(DATASETS.keys()) + ["all"],
    )
    args = parser.parse_args()

    to_process = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    for name in to_process:
        cfg = DATASETS[name]
        out_path = DATA_DIR / f"preprocessed_{name}.h5ad"

        if out_path.exists():
            print(f"\n[skip] {out_path.name} already exists. Delete to rerun.")
            continue

        adata = preprocess(name, cfg, PREPROCESS)

        print(f"\n  Saving to {out_path} ...")
        adata.write_h5ad(out_path)
        print(f"  Saved ({out_path.stat().st_size / 1e6:.1f} MB)")

    print("\nPreprocessing complete. Ready to run 02a/02b/02c scripts.")


if __name__ == "__main__":
    main()
