"""
Preprocessing for robustness experiments.

Loads the raw ALS spinal cord h5ad, applies QC filtering, normalises, and
log-transforms. Saves two outputs:

    PREPROCESSED_PATH  — full dataset after QC/norm (used for AUCell scoring)
    GRN_DATA_PATH      — random subsample of cells for GRN inference
                         (controlled by FLASHSCENIC["grn_subsample"])

Run once:
    python 01_preprocessing.py

Both output paths are defined in config.py so all experiment scripts
automatically pick up the same files.
"""

import sys
from pathlib import Path

import numpy as np
import scanpy as sc

# ── resolve package so we can be run from any directory ──────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.robustness.config import (
    DATASETS,
    FLASHSCENIC,
    FIGURES_DIR,
    GRN_DATA_PATH,
    PREPROCESS,
    PREPROCESSED_PATH,
    RESULTS_DIR,
)


def main():
    dataset = DATASETS["als_spinal_cord"]
    raw_path = dataset["path"]

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data not found: {raw_path}\n"
            "Download the dataset or update DATASETS['als_spinal_cord']['path'] "
            "in config.py."
        )

    # ── Load ─────────────────────────────────────────────────────────────────
    print(f"Loading {raw_path} ...")
    adata = sc.read_h5ad(raw_path)
    print(f"  Raw: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")
    print(f"  var_names sample: {list(adata.var_names[:5])}")

    # Use raw counts if available
    if adata.raw is not None:
        print("  Using adata.raw as count matrix.")
        adata = adata.raw.to_adata()

    # ── Convert Ensembl IDs → HGNC gene symbols ───────────────────────────────
    # CellxGENE datasets store var_names as ENSG IDs; TF lists use HGNC symbols.
    symbol_col = next(
        (c for c in ["feature_name", "gene_name", "hgnc_symbol", "gene_symbols"]
         if c in adata.var.columns),
        None,
    )
    if symbol_col is not None:
        print(f"  Converting var_names from Ensembl IDs to HGNC symbols "
              f"(using adata.var['{symbol_col}']).")
        adata.var_names = adata.var[symbol_col].astype(str)
        adata.var_names_make_unique()
        adata.var.index.name = "gene_symbols"  # avoid name collision with var column
        print(f"  var_names sample after conversion: {list(adata.var_names[:5])}")
    else:
        print("  WARNING: no gene symbol column found in adata.var — "
              "var_names will be used as-is. TF matching may fail if using "
              "Ensembl IDs.")

    # ── QC ───────────────────────────────────────────────────────────────────
    # Mitochondrial genes (detected by HGNC symbol prefix after conversion)
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )

    # Cell filters
    n_before = adata.n_obs
    adata = adata[adata.obs["n_genes_by_counts"] >= PREPROCESS["min_genes"]].copy()
    adata = adata[
        adata.obs["pct_counts_mt"] <= PREPROCESS["max_mito_frac"] * 100
    ].copy()
    print(
        f"  After cell QC: {adata.n_obs:,} cells "
        f"(removed {n_before - adata.n_obs:,})"
    )

    # Gene filters
    min_cells = max(
        PREPROCESS["min_cells"],
        int(np.ceil(PREPROCESS["min_cells_frac"] * adata.n_obs)),
    )
    sc.pp.filter_genes(adata, min_cells=min_cells)
    print(f"  After gene filter (min_cells={min_cells}): {adata.n_vars:,} genes")

    # ── Normalise & log-transform ─────────────────────────────────────────────
    sc.pp.normalize_total(adata, target_sum=PREPROCESS["target_sum"])
    sc.pp.log1p(adata)
    print(f"  Normalised (target_sum={PREPROCESS['target_sum']}) and log1p-transformed.")

    # ── HVG selection ─────────────────────────────────────────────────────────
    n_hvgs = PREPROCESS["n_hvgs"]
    if n_hvgs is not None:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvgs, subset=True)
        print(f"  HVG selection: kept {adata.n_vars:,} genes (n_hvgs={n_hvgs}).")
    else:
        print(f"  HVG selection: skipped (n_hvgs=None). Using all {adata.n_vars:,} genes.")

    # ── Save full preprocessed dataset ───────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(PREPROCESSED_PATH)
    print(f"  Saved full preprocessed: {PREPROCESSED_PATH}")

    # ── Subsample for GRN inference ───────────────────────────────────────────
    grn_subsample = FLASHSCENIC.get("grn_subsample")
    if grn_subsample is not None and grn_subsample < adata.n_obs:
        rng = np.random.default_rng(FLASHSCENIC.get("seed", 42))
        idx = rng.choice(adata.n_obs, size=grn_subsample, replace=False)
        adata_grn = adata[idx].copy()
        print(
            f"  Subsampled {grn_subsample:,} cells for GRN inference "
            f"(from {adata.n_obs:,})."
        )
    else:
        adata_grn = adata
        print("  No subsampling (grn_subsample >= n_cells or not set).")

    adata_grn.write_h5ad(GRN_DATA_PATH)
    print(f"  Saved GRN subsample: {GRN_DATA_PATH}")

    print("\nPreprocessing complete.")
    print(f"  Full dataset : {PREPROCESSED_PATH}")
    print(f"  GRN subsample: {GRN_DATA_PATH}")


if __name__ == "__main__":
    main()
