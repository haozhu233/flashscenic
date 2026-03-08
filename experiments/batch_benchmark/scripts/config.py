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
    "immune_human": {
        # Immune_ALL_human.h5ad — figshare file ID 25717328 (2064 MB)
        "url": "https://api.figshare.com/v2/file/download/25717328",
        "path": DATA_DIR / "immune_human.h5ad",
        "batch_key": "batch",
        "cell_type_key": "final_annotation",  # confirmed from adata.obs
        "sex_key": None,             # dataset has no sex column
        "species": "human",
        "min_cells_per_sex_per_batch": None,
    },
    "pancreas": {
        # human_pancreas_norm_complexBatch.h5ad — figshare file ID 24539828 (316 MB)
        "url": "https://api.figshare.com/v2/file/download/24539828",
        "path": DATA_DIR / "pancreas.h5ad",
        "batch_key": "tech",
        "cell_type_key": "celltype",
        "sex_key": None,             # pancreas dataset lacks sex annotation
        "species": "human",
        "min_cells_per_sex_per_batch": None,
    },
    "ad_neurons": {
        # CellxGENE collection 0d35c0fd — ALL cells, AD resilience study
        # Confirmed: 424,528 cells × 60,305 genes, 46 donors (batches)
        # sex: female=239,615 / male=184,913, perfectly distributed (each donor is one sex)
        # counts: in adata.raw (X is already normalized) — handled by _detect_counts_location
        # https://cellxgene.cziscience.com/collections/0d35c0fd-ef0b-4b70-bce6-645a4660e5fa
        "url": "https://datasets.cellxgene.cziscience.com/9066d7f5-924e-4022-9c8a-ceaff3d50104.h5ad",
        "path": DATA_DIR / "ad_neurons.h5ad",
        "batch_key": "donor_id",         # 46 donors, confirmed
        "cell_type_key": "Author_Annotation",  # finer subtypes vs. 7-class 'cell_type'
        "sex_key": "sex",                # confirmed: 'female' / 'male'
        "species": "human",
        "min_cells_per_sex_per_batch": 20,
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
    "min_cells": 10,            # min cells a gene must appear in
    "n_hvgs": 12000,            # number of highly variable genes — 10K for 60K-gene atlas
    "n_pcs": 50,                # PCA components
    "target_sum": 1e4,          # normalization target
}

# ---------------------------------------------------------------------------
# Integration methods
# ---------------------------------------------------------------------------
METHODS = ["raw_pca", "harmony", "scvi", "flashscenic"]

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
    "min_cells_per_class": 20,  # skip a class if fewer than this in test fold
    "sex_n_splits": 5,          # K for grouped K-fold when donors are single-sex
}

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
GLOBAL_SEED = 42
