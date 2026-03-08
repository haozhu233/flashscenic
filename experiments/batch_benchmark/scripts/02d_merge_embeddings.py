"""
Step 02d: Merge all embeddings into the preprocessed h5ad.

Reads each embedding from its .npy file (written by 02a/02b/02c) and
writes them all into a single updated preprocessed_{dataset}.h5ad.
This runs after 02a, 02b, 02c complete, avoiding concurrent h5ad writes.

Output:
    data/preprocessed_{dataset}.h5ad  (with X_raw_pca, X_harmony, X_scvi,
                                        X_flashscenic added to obsm)
"""

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import DATASETS, METHODS, DATA_DIR, EMBEDDINGS_DIR

EMBEDDING_FILES = {
    "raw_pca":     "{name}_raw_pca.npy",
    "harmony":     "{name}_harmony.npy",
    "scvi":        "{name}_scvi.npy",
    "flashscenic": "{name}_flashscenic.npy",
}

OBSM_KEYS = {
    "raw_pca":     "X_raw_pca",
    "harmony":     "X_harmony",
    "scvi":        "X_scvi",
    "flashscenic": "X_flashscenic",
}


def merge(name: str) -> None:
    preprocessed_path = DATA_DIR / f"preprocessed_{name}.h5ad"
    if not preprocessed_path.exists():
        print(f"  [skip] {preprocessed_path.name} not found")
        return

    print(f"  Loading {preprocessed_path.name} ...")
    adata = ad.read_h5ad(preprocessed_path)

    added = []
    missing = []
    for method in METHODS:
        npy_path = EMBEDDINGS_DIR / EMBEDDING_FILES[method].format(name=name)
        if not npy_path.exists():
            missing.append(method)
            print(f"  [skip] {method}: {npy_path.name} not found")
            continue
        emb = np.load(npy_path)
        adata.obsm[OBSM_KEYS[method]] = emb
        added.append(method)
        print(f"  + {OBSM_KEYS[method]}: {emb.shape}")

    if not added:
        print(f"  [abort] no embeddings found — nothing to write")
        return

    adata.write_h5ad(preprocessed_path)
    print(f"  Wrote {preprocessed_path.name} with obsm keys: {added}")
    if missing:
        print(f"  Missing (not included): {missing}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all",
                        choices=list(DATASETS.keys()) + ["all"])
    args = parser.parse_args()

    to_process = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    for name in to_process:
        print(f"\n{'='*60}")
        print(f"Merging embeddings: {name}")
        print(f"{'='*60}")
        merge(name)

    print("\nMerge done. Run 03_metrics.py and 04_ml_predictor.py next.")


if __name__ == "__main__":
    main()
