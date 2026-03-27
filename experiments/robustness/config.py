"""
Central configuration for the batch integration benchmark experiment.

Adjust paths and hyperparameters here before running any script.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
ROBUSTNESS_DIR = Path(__file__).parent.resolve()   # experiments/robustness/
BASE_DIR = ROBUSTNESS_DIR.parent.parent.resolve()  # project root (flashscenic/)
RESULTS_DIR = ROBUSTNESS_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
DATA_DIR = BASE_DIR / "data"                       # project root data/

# Preprocessing outputs (written by 01_preprocessing.py, read by exp scripts)
PREPROCESSED_PATH = RESULTS_DIR / "als_preprocessed.h5ad"
GRN_DATA_PATH = RESULTS_DIR / "als_grn_subsample.h5ad"  # subsampled for GRN inference

for d in [RESULTS_DIR, FIGURES_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
# scIB figshare article 12420968 — file IDs verified via API
# https://api.figshare.com/v2/articles/12420968/files
DATASETS = {
    "als_spinal_cord": {
        # CellxGENE — ALS spinal cord (6 ALS + 6 healthy donors)
        "url": "https://datasets.cellxgene.cziscience.com/f6f233ba-91ff-4d9e-80e9-eaaf1d66828c.h5ad",
        "path": DATA_DIR / "als_spinal_cord.h5ad",
        "batch_key": "donor_id",
        "cell_type_key": "cell_type",
        "sex_key": "sex",
        "age_key": "development_stage",
        "disease_key": "disease",
        "species": "human",
        "min_cells_per_sex_per_batch": None,
    },
}

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
PREPROCESS = {
    "min_genes": 200,           # min genes per cell
    "max_mito_frac": 0.20,      # max fraction mitochondrial reads
    "min_cells": 10,            # min cells a gene must appear in (absolute floor)
    "min_cells_frac": 0.05,     # min fraction of cells a gene must appear in (used when n_hvgs is None)
    "n_hvgs": None,             # None = no HVG selection; int = top N HVGs (batch-aware)
    "n_pcs": 50,                # PCA components
    "target_sum": 1e4,          # normalization target
}

FLASHSCENIC = {
    "grn_n_steps": 1000,
    "grn_sparsity_threshold": 1.5,
    "module_k": 50,
    "module_percentile_thresholds": (75,),
    "module_top_n_per_target": (5, 10, 50),
    "module_min_targets": 20,
    "aucell_batch_size": 64,
    "device": "cuda",
    "seed": 42,
    "version": "v10",
    "datasource": "scenic",
    # GRN inference subsample: learn regulons on this many cells (None = full dataset).
    # Full 424K cells exceeds RTX 4090 24GB VRAM; 20K is sufficient for GRN learning.
    # Regulons are then scored on ALL cells via AUCell.
    "grn_subsample": 100000,
}

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
