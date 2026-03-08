"""
Step 2a: Raw PCA baseline and Harmony integration.

Reads preprocessed_{dataset}.h5ad, adds two embeddings:
  - obsm["X_raw_pca"]: PCA from 01_preprocess.py, no batch correction
  - obsm["X_harmony"]: Harmony-corrected PCA embedding

Saves embeddings as .npy to results/embeddings/ and writes updated h5ad.

Usage:
    python 02a_run_baselines.py [--dataset immune_human|pancreas|all]
"""

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import DATASETS, HARMONY, DATA_DIR, EMBEDDINGS_DIR, GLOBAL_SEED


def run_baselines(name: str, cfg: dict) -> None:
    preprocessed_path = DATA_DIR / f"preprocessed_{name}.h5ad"
    if not preprocessed_path.exists():
        raise FileNotFoundError(
            f"{preprocessed_path} not found. Run 01_preprocess.py first."
        )

    print(f"\n{'='*60}")
    print(f"Running baselines: {name}")
    print(f"{'='*60}")

    adata = ad.read_h5ad(preprocessed_path)
    print(f"  Loaded: {adata.shape}")

    batch_key = cfg["batch_key"]

    # ---- Raw PCA (no correction) -------------------------------------------
    raw_pca_path = EMBEDDINGS_DIR / f"{name}_raw_pca.npy"
    if raw_pca_path.exists():
        print(f"  [skip] raw_pca embedding already exists")
        X_raw_pca = np.load(raw_pca_path)
    else:
        X_raw_pca = adata.obsm["X_pca"].copy()
        np.save(raw_pca_path, X_raw_pca)
        print(f"  Raw PCA: {X_raw_pca.shape} saved to {raw_pca_path.name}")

    adata.obsm["X_raw_pca"] = X_raw_pca

    # ---- Harmony --------------------------------------------------------------
    harmony_path = EMBEDDINGS_DIR / f"{name}_harmony.npy"
    if harmony_path.exists():
        print(f"  [skip] harmony embedding already exists")
        X_harmony = np.load(harmony_path)
    else:
        if batch_key not in adata.obs.columns:
            raise ValueError(
                f"batch_key '{batch_key}' not found in adata.obs. "
                f"Available: {list(adata.obs.columns)}"
            )

        try:
            import harmonypy
        except ImportError:
            raise ImportError("harmonypy not installed. Run: pip install harmonypy")

        print(f"  Running Harmony (batch_key='{batch_key}', "
              f"max_iter={HARMONY['max_iter_harmony']})...")
        ho = harmonypy.run_harmony(
            X_raw_pca,
            adata.obs,
            batch_key,
            max_iter_harmony=HARMONY["max_iter_harmony"],
            random_state=HARMONY["random_state"],
        )
        # harmonypy Z_corr orientation varies by version:
        # orient to (n_cells, n_pcs) regardless
        Z = ho.Z_corr
        X_harmony = Z if Z.shape[0] == adata.n_obs else Z.T
        np.save(harmony_path, X_harmony)
        print(f"  Harmony: {X_harmony.shape} saved to {harmony_path.name}")

    adata.obsm["X_harmony"] = X_harmony

    print(f"  Embeddings saved as .npy — h5ad will be merged by 02d_merge_embeddings.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all",
                        choices=list(DATASETS.keys()) + ["all"])
    args = parser.parse_args()

    to_process = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    for name in to_process:
        run_baselines(name, DATASETS[name])

    print("\nBaselines done. Run 02b_run_scvi.py and 02c_run_flashscenic.py in parallel.")


if __name__ == "__main__":
    main()
