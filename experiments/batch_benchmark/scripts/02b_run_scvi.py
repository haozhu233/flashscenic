"""
Step 2b: scVI integration.

Trains a scVI model on raw counts with batch_key as the batch covariate.
Sex is NEVER passed as a covariate — the model may not destroy its signal
intentionally, so any destruction is due to over-correction.

Saves obsm["X_scvi"] embedding (n_cells, n_latent=30).

Usage:
    python 02b_run_scvi.py [--dataset immune_human|pancreas|all]
"""

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import DATASETS, SCVI as SCVI_CFG, DATA_DIR, EMBEDDINGS_DIR


def run_scvi(name: str, cfg: dict) -> None:
    preprocessed_path = DATA_DIR / f"preprocessed_{name}.h5ad"
    if not preprocessed_path.exists():
        raise FileNotFoundError(
            f"{preprocessed_path} not found. Run 01_preprocess.py first."
        )

    scvi_path = EMBEDDINGS_DIR / f"{name}_scvi.npy"
    if scvi_path.exists():
        print(f"  [skip] scvi embedding already exists for {name}")
        return

    print(f"\n{'='*60}")
    print(f"Running scVI: {name}")
    print(f"{'='*60}")

    try:
        import scvi
    except ImportError:
        raise ImportError("scvi-tools not installed. Run: pip install scvi-tools")

    scvi.settings.seed = SCVI_CFG["seed"]

    adata = ad.read_h5ad(preprocessed_path)
    print(f"  Loaded: {adata.shape}")

    batch_key = cfg["batch_key"]
    if batch_key not in adata.obs.columns:
        raise ValueError(f"batch_key '{batch_key}' not found in adata.obs")

    # scVI requires raw counts — find them
    if "counts" in adata.layers:
        adata_scvi = adata.copy()
        adata_scvi.X = adata_scvi.layers["counts"]
        print(f"  Using raw counts from layers['counts']")
    else:
        raise ValueError(
            "Raw counts not found in adata.layers['counts']. "
            "Ensure 01_preprocess.py was run successfully."
        )

    # Subset to HVGs for faster training (matches preprocessing)
    if "highly_variable" in adata_scvi.var.columns:
        adata_scvi = adata_scvi[:, adata_scvi.var.highly_variable].copy()
        print(f"  Subset to {adata_scvi.n_vars:,} HVGs")

    # Setup scVI
    scvi.model.SCVI.setup_anndata(
        adata_scvi,
        layer=None,   # use .X which we've set to counts
        batch_key=batch_key,
        # sex is intentionally NOT included as a covariate
    )

    model = scvi.model.SCVI(
        adata_scvi,
        n_latent=SCVI_CFG["n_latent"],
        n_layers=SCVI_CFG["n_layers"],
    )

    print(f"  Training scVI ({SCVI_CFG['n_epochs']} max epochs, "
          f"n_latent={SCVI_CFG['n_latent']}, "
          f"batch_key='{batch_key}')...")

    model.train(
        max_epochs=SCVI_CFG["n_epochs"],
        batch_size=SCVI_CFG["batch_size"],
        early_stopping=SCVI_CFG["early_stopping"],
        early_stopping_patience=SCVI_CFG["early_stopping_patience"],
        plan_kwargs={"lr": 1e-3},
    )

    X_scvi = model.get_latent_representation()
    print(f"  scVI latent representation: {X_scvi.shape}")

    np.save(scvi_path, X_scvi)
    print(f"  Saved to {scvi_path.name}")

    print(f"  Embedding saved as .npy — h5ad will be merged by 02d_merge_embeddings.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all",
                        choices=list(DATASETS.keys()) + ["all"])
    args = parser.parse_args()

    to_process = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    for name in to_process:
        run_scvi(name, DATASETS[name])

    print("\nscVI done.")


if __name__ == "__main__":
    main()
