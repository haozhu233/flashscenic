"""
Central configuration for the batch integration benchmark experiment.

Adjust paths and hyperparameters here before running any script.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent.resolve()
RESULTS_DIR = BASE_DIR / "results"
EMBEDDINGS_DIR = RESULTS_DIR / "embeddings"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
DATA_DIR = BASE_DIR / "data"

for d in [RESULTS_DIR, EMBEDDINGS_DIR, METRICS_DIR, FIGURES_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
# scIB figshare article 12420968 — file IDs verified via API
# https://api.figshare.com/v2/articles/12420968/files
DATASETS = {    
#    "ad_neurons": {
        # CellxGENE collection 0d35c0fd — ALL cells, AD resilience study
        # Confirmed: 424,528 cells × 60,305 genes, 46 donors (batches)
        # sex: female=239,615 / male=184,913, perfectly distributed (each donor is one sex)
        # counts: in adata.raw (X is already normalized) — handled by _detect_counts_location
        # https://cellxgene.cziscience.com/collections/0d35c0fd-ef0b-4b70-bce6-645a4660e5fa
#        "url": "https://datasets.cellxgene.cziscience.com/9066d7f5-924e-4022-9c8a-ceaff3d50104.h5ad",
#        "path": DATA_DIR / "ad_neurons.h5ad",
#        "batch_key": "donor_id",         # 46 donors, confirmed
#        "cell_type_key": "Author_Annotation",  # finer subtypes vs. 7-class 'cell_type'
#        "sex_key": "sex",                # confirmed: 'female' / 'male'
#       "age_key": "Age",                # donor age in years
#        "species": "human",
#        "min_cells_per_sex_per_batch": 20,
#    },
#    "ad_inhibitory": {
#        # CellxGENE — inhibitory neuron subset of the same AD resilience study
#        # Same obs columns and donor structure as ad_neurons
#        "url": "https://datasets.cellxgene.cziscience.com/e7be14ca-e499-4dfa-8292-0768896852dd.h5ad",
#        "path": DATA_DIR / "ad_inhibitory.h5ad",
#        "batch_key": "donor_id",
#        "cell_type_key": "Author_Annotation",
#        "sex_key": "sex",
#        "age_key": "Age",
#        "species": "human",
#        "min_cells_per_sex_per_batch": 20,
#    },
    "als_motor_cortex": {
        # CellxGENE — ALS motor cortex (6 ALS + 6 healthy donors)
        "url": "https://datasets.cellxgene.cziscience.com/0ab54d91-066c-4223-a9ea-6a3b0d1adef4.h5ad",
        "path": DATA_DIR / "als_motor_cortex.h5ad",
        "batch_key": "donor_id",
        "cell_type_key": "cell_type",
        "sex_key": "sex",                   # confirmed: "male" / "female"
        "age_key": "development_stage",     # confirmed: "67-year-old stage" format
        "disease_key": "disease",           # confirmed: "amyotrophic lateral sclerosis" / "normal"
        "species": "human",
        "min_cells_per_sex_per_batch": None,
    },
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

# Primary dataset used for sex-prediction experiment
PRIMARY_DATASET = "ad_neurons"

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
PREPROCESS = {
    "min_genes": 200,           # min genes per cell
    "max_mito_frac": 0.20,      # max fraction mitochondrial reads
    "min_cells": 10,            # min cells a gene must appear in (absolute floor)
    "min_cells_frac": 0.01,     # min fraction of cells a gene must appear in (used when n_hvgs is None)
    "n_hvgs": None,             # None = no HVG selection; int = top N HVGs (batch-aware)
    "n_pcs": 50,                # PCA components
    "target_sum": 1e4,          # normalization target
}

# ---------------------------------------------------------------------------
# Analysis filters
# ---------------------------------------------------------------------------
# Minimum number of cells a cell type must have to be included in RSS (05b)
# and DE TF analysis (06). Cell types with fewer cells are skipped.
MIN_CELLS_PER_CT = 500

# ---------------------------------------------------------------------------
# Integration methods
# ---------------------------------------------------------------------------
METHODS = ["raw_pca", "flashscenic"]

HARMONY = {
    "max_iter_harmony": 20,
    "random_state": 42,
}

SCVI = {
    "n_latent": 30,
    "n_layers": 2,
    "n_epochs": 400,
    "batch_size": 128,
    "early_stopping": True,
    "early_stopping_patience": 30,
    "seed": 42,
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
# scIB metrics
# ---------------------------------------------------------------------------
SCIB = {
    "n_neighbors": 90,          # for graph-based metrics
    "batch_weight": 0.4,        # weight for batch removal score in composite
    "bio_weight": 0.6,          # weight for bio conservation score
    # Leiden clustering resolutions to try for NMI/ARI
    "leiden_resolutions": [0.1, 0.3, 0.5, 1.0],
}

# ---------------------------------------------------------------------------
# ML predictor
# ---------------------------------------------------------------------------
ML = {
    "seed": 42,
    "elasticnet": {
        "C": 1.0,
        "l1_ratio": 0.5,        # equal L1+L2 (true ElasticNet)
        "max_iter": 2000,
        "solver": "saga",       # handles elasticnet penalty
        "tol": 1e-3,
    },
    "test_size": 0.25,          # fraction of donors held out for test set (datasets without disease)
    "n_test_per_disease": 2,    # donors per disease group in test set (when disease_key is available)
    # Metacell aggregation: both cell-level and metacell-level are always run.
    # Metacells are applied to flashscenic only (raw_pca always cell-level).
    "metacell_size": 16,        # cells averaged per metacell
    "n_metacells": 100,         # fallback if expected_reuse is None
    # Adaptive metacell count: n_mc = max(1, round(expected_reuse * n_cells / metacell_size))
    # Set to None to use fixed n_metacells instead.
    "expected_reuse": 2.5,      # expected times one cell appears across metacells per donor
    # How to compute delta_mean / mean_disease / mean_control for DE TF output:
    #   "donor"   → average metacells within each donor first, then average donor means
    #               (equal weight per donor regardless of cell count — recommended)
    #   "metacell" → pool all metacells and take the grand mean
    #                (larger donors dominate proportionally to their cell count)
    "de_delta_weighting": "donor",
}

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
GLOBAL_SEED = 42
