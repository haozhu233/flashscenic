import numpy as np

from .aucell import get_aucell
from .data import download_data, DownloadedResources, list_available_resources
from .pipeline import run_flashscenic
from .cistarget import (
    CisTargetPruner,
    GPUCisTargetPruner,  # backwards compat alias
    MultiDatabaseCisTargetPruner,  # Multi-database support
    MotifAnnotation,  # Motif annotation filtering
    filter_by_annotations,  # Annotation filtering function
    compute_recovery_aucs,
    compute_nes,
    prune_single_module,
)
from .modules import (
    select_topk_targets,
    select_threshold_targets,
    filter_by_min_targets,
    filter_by_mapped_fraction,
    get_target_indices,
    binarize,
    to_numpy,
)


def regulons_to_adjacency(regulons: list[dict], gene_names: list[str]) -> np.ndarray:
    """
    Convert list of regulon dicts to adjacency matrix for AUCell.

    Parameters
    ----------
    regulons : list of dict
        Output from CisTargetPruner.prune_modules(), each dict has 'genes' key
    gene_names : list of str
        List of gene names matching columns of expression matrix

    Returns
    -------
    np.ndarray
        Binary adjacency matrix of shape (n_regulons, n_genes)
    """
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    n_genes = len(gene_names)

    adj_rows = []
    for reg in regulons:
        row = np.zeros(n_genes, dtype=np.float32)
        for gene in reg['genes']:
            if gene in gene_to_idx:
                row[gene_to_idx[gene]] = 1.0
        adj_rows.append(row)

    return np.stack(adj_rows, axis=0)


__all__ = [
    # Pipeline
    'run_flashscenic',
    # Data download
    'download_data',
    'DownloadedResources',
    'list_available_resources',
    # AUCell
    'get_aucell',
    'regulons_to_adjacency',
    # cisTarget
    'CisTargetPruner',
    'GPUCisTargetPruner',
    'MultiDatabaseCisTargetPruner',
    'MotifAnnotation',
    'filter_by_annotations',
    'compute_recovery_aucs',
    'compute_nes',
    'prune_single_module',
    # Module selection
    'select_topk_targets',
    'select_threshold_targets',
    'filter_by_min_targets',
    'filter_by_mapped_fraction',
    'get_target_indices',
    'binarize',
    'to_numpy',
]
