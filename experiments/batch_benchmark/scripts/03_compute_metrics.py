"""
Step 3: Compute batch integration metrics (scIB-style).

For each method's embedding, computes:
  Batch removal:
    - iLISI  (integration LISI — higher = more batch mixing)
    - ASW_batch  (batch silhouette width — near 0 = good)
    - kBET acceptance rate  (if scib-metrics provides it)

  Biological conservation:
    - cLISI  (cell-type LISI — lower = cell types well separated)
    - ASW_cell_type  (cell-type silhouette width — higher = good)
    - NMI  (Leiden clustering vs true cell type labels)
    - ARI  (Adjusted Rand Index of clustering)

  Composite:
    - overall_score = 0.4 * batch_score + 0.6 * bio_score

Output: results/metrics/scib_scores_{dataset}.csv

Usage:
    python 03_compute_metrics.py [--dataset immune_human|pancreas|all]
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, str(Path(__file__).parent))
from config import DATASETS, METHODS, SCIB, DATA_DIR, METRICS_DIR, GLOBAL_SEED

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Embedding loader
# ---------------------------------------------------------------------------

EMBEDDING_KEYS = {
    "raw_pca":     "X_raw_pca",
    "harmony":     "X_harmony",
    "scvi":        "X_scvi",
    "flashscenic": "X_flashscenic",
}


def load_embedding(adata: ad.AnnData, method: str) -> np.ndarray:
    key = EMBEDDING_KEYS[method]
    if key not in adata.obsm:
        raise KeyError(f"Embedding '{key}' not found in adata.obsm. "
                       f"Run the corresponding 02x script first.")
    return adata.obsm[key]


# ---------------------------------------------------------------------------
# LISI computation (pure numpy, no R dependency)
# ---------------------------------------------------------------------------

def compute_lisi(X: np.ndarray, labels: np.ndarray, n_neighbors: int = 90,
                 perplexity: float = 30.0) -> np.ndarray:
    """
    Compute LISI (Local Inverse Simpson's Index) per cell.

    For each cell, fits a Gaussian kernel over the k-NN neighborhood and
    computes the effective number of classes in that neighborhood.

    Parameters
    ----------
    X : (n_cells, n_dims) embedding
    labels : (n_cells,) integer-encoded labels
    n_neighbors : number of neighbors to consider
    perplexity : target perplexity for kernel bandwidth search

    Returns
    -------
    lisi_per_cell : (n_cells,) float array. Higher = more mixed.
    """
    from sklearn.neighbors import NearestNeighbors

    classes = np.unique(labels)
    n_classes = len(classes)
    n_cells = X.shape[0]

    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean", n_jobs=-1)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    # Exclude self (first neighbor)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    lisi = np.zeros(n_cells)

    for i in range(n_cells):
        d = distances[i]
        idx = indices[i]
        neighbor_labels = labels[idx]

        # Binary search for bandwidth beta s.t. perplexity matches
        beta = 1.0
        lo, hi = 0.0, np.inf
        for _ in range(50):
            P = np.exp(-d * beta)
            P_sum = P.sum()
            if P_sum == 0:
                break
            P /= P_sum
            H = -np.sum(P * np.log(P + 1e-12))
            Hdiff = H - np.log(perplexity)
            if abs(Hdiff) < 1e-5:
                break
            if Hdiff > 0:
                lo = beta
                beta = beta * 2 if hi == np.inf else (beta + hi) / 2
            else:
                hi = beta
                beta = (lo + beta) / 2

        # Compute inverse Simpson's index
        simpson = 0.0
        for c in classes:
            mask = neighbor_labels == c
            p_c = P[mask].sum()
            simpson += p_c ** 2
        lisi[i] = 1.0 / (simpson + 1e-12)

    return lisi


def ilisi(X: np.ndarray, batch_labels: np.ndarray, n_neighbors: int = 90) -> float:
    """Mean iLISI (integration LISI). Higher = better batch mixing."""
    lisi_vals = compute_lisi(X, batch_labels, n_neighbors=n_neighbors)
    n_batches = len(np.unique(batch_labels))
    # Normalize to [0, 1]: (lisi - 1) / (n_batches - 1)
    return float(np.mean((lisi_vals - 1) / max(n_batches - 1, 1)))


def clisi(X: np.ndarray, ct_labels: np.ndarray, n_neighbors: int = 90) -> float:
    """
    Mean cLISI (cell-type LISI). Lower raw LISI = better separation.
    We return 1 - normalized_cLISI so higher = better (consistent sign with other metrics).
    """
    lisi_vals = compute_lisi(X, ct_labels, n_neighbors=n_neighbors)
    n_ct = len(np.unique(ct_labels))
    normalized = (lisi_vals - 1) / max(n_ct - 1, 1)
    return float(1.0 - np.mean(normalized))


# ---------------------------------------------------------------------------
# ASW computation
# ---------------------------------------------------------------------------

def compute_asw(X: np.ndarray, labels: np.ndarray, subsample: int = 10000) -> float:
    """
    Average Silhouette Width.
    Subsamples for speed on large datasets.
    Returns value in [-1, 1]. For batch: want near 0. For cell type: want near 1.
    """
    from sklearn.metrics import silhouette_score

    n = X.shape[0]
    if n > subsample:
        rng = np.random.default_rng(GLOBAL_SEED)
        idx = rng.choice(n, size=subsample, replace=False)
        X_sub = X[idx]
        labels_sub = labels[idx]
    else:
        X_sub = X
        labels_sub = labels

    if len(np.unique(labels_sub)) < 2:
        return np.nan

    return float(silhouette_score(X_sub, labels_sub, metric="euclidean"))


def asw_batch(X: np.ndarray, batch_labels: np.ndarray) -> float:
    """
    ASW for batch labels, normalized to [0, 1].
    Perfect batch mixing = ASW near 0. Returns 1 - |ASW| / 1.
    """
    raw_asw = compute_asw(X, batch_labels)
    # Normalize: 1 - |asw| gives 1 for perfect mixing, 0 for perfect separation
    return float(1.0 - abs(raw_asw))


def asw_cell_type(X: np.ndarray, ct_labels: np.ndarray) -> float:
    """
    ASW for cell-type labels, normalized to [0, 1].
    Perfect cell-type separation = ASW near 1.
    Returns (asw + 1) / 2 to scale [-1,1] → [0,1].
    """
    raw_asw = compute_asw(X, ct_labels)
    return float((raw_asw + 1.0) / 2.0)


# ---------------------------------------------------------------------------
# Clustering-based metrics (NMI, ARI)
# ---------------------------------------------------------------------------

def compute_nmi_ari(X: np.ndarray, ct_labels: np.ndarray,
                    adata_ref: ad.AnnData, method: str,
                    resolutions: list) -> tuple[float, float]:
    """
    Build k-NN graph, run Leiden at multiple resolutions, pick best NMI,
    return corresponding ARI.
    """
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

    # Build a temporary AnnData to use scanpy's graph tools
    tmp = ad.AnnData(X=X)
    sc.pp.neighbors(tmp, use_rep="X", n_neighbors=15, random_state=GLOBAL_SEED)

    best_nmi = -1.0
    best_ari = 0.0

    for res in resolutions:
        sc.tl.leiden(tmp, resolution=res, random_state=GLOBAL_SEED, key_added="leiden")
        pred = tmp.obs["leiden"].astype(int).values
        nmi = normalized_mutual_info_score(ct_labels, pred, average_method="arithmetic")
        ari = adjusted_rand_score(ct_labels, pred)
        if nmi > best_nmi:
            best_nmi = nmi
            best_ari = ari

    return float(best_nmi), float(best_ari)


# ---------------------------------------------------------------------------
# Main metrics computation
# ---------------------------------------------------------------------------

def compute_all_metrics(name: str, cfg: dict) -> pd.DataFrame:
    preprocessed_path = DATA_DIR / f"preprocessed_{name}.h5ad"
    adata = ad.read_h5ad(preprocessed_path)

    batch_key = cfg["batch_key"]
    ct_key = cfg["cell_type_key"]

    # Encode labels as integers
    batch_labels = pd.Categorical(adata.obs[batch_key]).codes
    ct_labels = pd.Categorical(adata.obs[ct_key]).codes

    n_neighbors = SCIB["n_neighbors"]
    resolutions = SCIB["leiden_resolutions"]

    rows = []

    for method in METHODS:
        try:
            X = load_embedding(adata, method)
        except KeyError as e:
            print(f"  [skip] {method}: {e}")
            continue

        print(f"\n  Computing metrics for: {method} (embedding shape: {X.shape})")

        row = {"method": method, "dataset": name}

        # --- Batch removal metrics ---
        print(f"    iLISI ...", end=" ", flush=True)
        row["iLISI"] = ilisi(X, batch_labels, n_neighbors=n_neighbors)
        print(f"{row['iLISI']:.3f}")

        print(f"    ASW_batch ...", end=" ", flush=True)
        row["ASW_batch"] = asw_batch(X, batch_labels)
        print(f"{row['ASW_batch']:.3f}")

        # --- Bio conservation metrics ---
        print(f"    cLISI ...", end=" ", flush=True)
        row["cLISI"] = clisi(X, ct_labels, n_neighbors=n_neighbors)
        print(f"{row['cLISI']:.3f}")

        print(f"    ASW_cell_type ...", end=" ", flush=True)
        row["ASW_cell_type"] = asw_cell_type(X, ct_labels)
        print(f"{row['ASW_cell_type']:.3f}")

        print(f"    NMI / ARI ...", end=" ", flush=True)
        row["NMI"], row["ARI"] = compute_nmi_ari(
            X, ct_labels, adata, method, resolutions
        )
        print(f"NMI={row['NMI']:.3f}, ARI={row['ARI']:.3f}")

        # --- Composite score ---
        batch_score = np.mean([row["iLISI"], row["ASW_batch"]])
        bio_score = np.mean([row["cLISI"], row["ASW_cell_type"], row["NMI"], row["ARI"]])
        row["batch_score"] = float(batch_score)
        row["bio_score"] = float(bio_score)
        row["overall_score"] = (
            SCIB["batch_weight"] * batch_score +
            SCIB["bio_weight"] * bio_score
        )
        print(f"    Composite: batch={batch_score:.3f}, bio={bio_score:.3f}, "
              f"overall={row['overall_score']:.3f}")

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all",
                        choices=list(DATASETS.keys()) + ["all"])
    args = parser.parse_args()

    to_process = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    for name in to_process:
        print(f"\n{'='*60}")
        print(f"scIB metrics: {name}")
        print(f"{'='*60}")

        out_path = METRICS_DIR / f"scib_scores_{name}.csv"
        df = compute_all_metrics(name, DATASETS[name])
        df.to_csv(out_path, index=False)
        print(f"\n  Results saved to {out_path.name}")
        print(df.set_index("method").drop(columns=["dataset"]).round(3).to_string())

    print("\nMetrics done. Run 04_ml_predictor.py next.")


if __name__ == "__main__":
    main()
