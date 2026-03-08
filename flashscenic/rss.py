import numpy as np
from scipy.spatial.distance import jensenshannon

from .binarize_aucell import binarize_auc_matrix


def regulon_specificity_scores(auc_matrix, cell_type_labels, regulon_names=None, binarize=False):
    """
    Compute Regulon Specificity Scores (RSS) based on Jensen-Shannon divergence.

    RSS quantifies how specific each regulon's activity is to each cell type.
    A score close to 1 means the regulon is exclusively active in that cell type.

    Reference: Suo et al. 2018 (doi: 10.1016/j.celrep.2018.10.045)

    Parameters
    ----------
    auc_matrix : np.ndarray
        AUCell scores of shape (n_cells, n_regulons).
    cell_type_labels : array-like
        Cell type label per cell (length n_cells). Can be a list, numpy array,
        or pandas Series.
    regulon_names : list of str, optional
        Names for each regulon column. If None, integer indices are used.
    binarize : bool, default False
        If True, binarize the AUC matrix into per-cell on/off calls using
        a per-regulon GMM threshold before computing RSS. Follows the SCENIC
        protocol (Aibar et al., Nature Protocols 2020).

    Returns
    -------
    dict
        'rss' : np.ndarray of shape (n_cell_types, n_regulons)
            RSS values. Higher means more specific.
        'cell_types' : list of str
            Sorted unique cell type labels (row labels of rss).
        'regulon_names' : list of str
            Regulon names (column labels of rss).
    """
    if binarize:
        auc_matrix = binarize_auc_matrix(auc_matrix).astype(np.float64)

    labels = np.asarray(cell_type_labels)
    cell_types = sorted(set(labels.tolist()))
    n_types = len(cell_types)
    n_regulons = auc_matrix.shape[1]

    if regulon_names is None:
        regulon_names = [str(i) for i in range(n_regulons)]

    rss_values = np.empty((n_types, n_regulons), dtype=np.float32)

    for cidx in range(n_regulons):
        aucs = auc_matrix[:, cidx].astype(np.float64)
        auc_sum = aucs.sum()
        if auc_sum == 0:
            rss_values[:, cidx] = 0.0
            continue
        auc_norm = aucs / auc_sum

        for ridx, cell_type in enumerate(cell_types):
            indicator = (labels == cell_type).astype(np.float64)
            indicator_sum = indicator.sum()
            if indicator_sum == 0:
                rss_values[ridx, cidx] = 0.0
                continue
            indicator_norm = indicator / indicator_sum
            rss_values[ridx, cidx] = 1.0 - jensenshannon(auc_norm, indicator_norm)

    return {
        'rss': rss_values,
        'cell_types': cell_types,
        'regulon_names': list(regulon_names),
    }
