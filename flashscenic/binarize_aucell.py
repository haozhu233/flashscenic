"""
Binarization of AUCell scores into per-cell regulon on/off calls.

Reference: Aibar et al., Nature Protocols 2020
https://www.nature.com/articles/s41596-020-0336-2
"""

import numpy as np


def binarize_auc_matrix(
    auc_matrix: np.ndarray,
    method: str = "gmm",
    random_state: int = 42,
) -> np.ndarray:
    """
    Binarize AUCell score matrix into per-cell regulon on/off calls.

    For each regulon, a threshold is derived from the distribution of AUC scores
    across all cells. Cells above the threshold are "on" (1), below are "off" (0).

    The GMM method (default) follows the SCENIC protocol: fit a two-component
    Gaussian mixture model to each regulon's score distribution and place the
    threshold at the density minimum between the two component means.

    Parameters
    ----------
    auc_matrix : np.ndarray
        AUCell scores of shape (n_cells, n_regulons).
    method : {"gmm", "zscore"}, default "gmm"
        Thresholding strategy:
        - "gmm": bimodal GMM threshold (SCENIC standard). Requires scikit-learn.
        - "zscore": threshold = mean + 2 * std per regulon (fast, no fitting).
    random_state : int, default 42
        Random seed passed to GaussianMixture (only used when method="gmm").

    Returns
    -------
    np.ndarray
        Binary matrix of shape (n_cells, n_regulons), dtype uint8.
        1 = regulon active in that cell, 0 = inactive.
    """
    auc_matrix = np.asarray(auc_matrix, dtype=np.float64)
    n_cells, n_regulons = auc_matrix.shape
    binary = np.zeros((n_cells, n_regulons), dtype=np.uint8)

    if method == "gmm":
        thresholds = _gmm_thresholds(auc_matrix, random_state=random_state)
    elif method == "zscore":
        thresholds = _zscore_thresholds(auc_matrix)
    else:
        raise ValueError(f"Unknown method {method!r}. Choose 'gmm' or 'zscore'.")

    for j in range(n_regulons):
        binary[:, j] = (auc_matrix[:, j] > thresholds[j]).astype(np.uint8)

    return binary


# ---------------------------------------------------------------------------
# Internal threshold helpers
# ---------------------------------------------------------------------------

def _gmm_thresholds(auc_matrix: np.ndarray, random_state: int = 42) -> np.ndarray:
    """
    Per-regulon GMM threshold: minimum density between the two component means.

    If the distribution appears unimodal (both component means within 1e-4 of
    each other), the threshold is set to +inf so all cells are called inactive.
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        raise ImportError(
            "scikit-learn is required for GMM binarization: pip install scikit-learn"
        )

    n_regulons = auc_matrix.shape[1]
    thresholds = np.full(n_regulons, np.inf)

    for j in range(n_regulons):
        scores = auc_matrix[:, j]

        # Skip constant columns
        if scores.max() - scores.min() < 1e-10:
            continue

        gmm = GaussianMixture(n_components=2, random_state=random_state)
        gmm.fit(scores.reshape(-1, 1))

        means = gmm.means_.ravel()
        lo_idx, hi_idx = np.argsort(means)
        mu_lo, mu_hi = means[lo_idx], means[hi_idx]

        # Unimodal guard
        if abs(mu_hi - mu_lo) < 1e-4:
            continue  # threshold stays +inf → all cells inactive

        # Find density minimum between the two means on a 512-point grid
        grid = np.linspace(mu_lo, mu_hi, 512).reshape(-1, 1)
        log_probs = gmm.score_samples(grid)
        thresholds[j] = grid[np.argmin(log_probs), 0]

    return thresholds


def _zscore_thresholds(auc_matrix: np.ndarray, n_std: float = 2.0) -> np.ndarray:
    """Per-regulon threshold: mean + n_std * std."""
    means = auc_matrix.mean(axis=0)
    stds = auc_matrix.std(axis=0)
    return means + n_std * stds
