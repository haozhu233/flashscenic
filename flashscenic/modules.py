"""
Module creation utilities for flashscenic.

All operations are tensor-native for GPU acceleration.
Input: adjacency matrix (n_tfs x n_genes) from GRN inference
Output: filtered adjacency matrix ready for AUCell
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union

ArrayLike = Union[np.ndarray, torch.Tensor]


def select_topk_targets(
    adj: ArrayLike,
    k: int = 50,
    include_tf: bool = True,
    tf_indices: Optional[ArrayLike] = None,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Select top-k targets per TF from adjacency matrix.

    Parameters
    ----------
    adj : array-like
        Adjacency matrix (n_tfs x n_genes) with edge weights
    k : int, default=50
        Number of top targets to select per TF
    include_tf : bool, default=True
        Include TF itself in its module (sets diagonal to 1 if tf_indices provided)
    tf_indices : array-like, optional
        Index of each TF in the gene list. Required if include_tf=True and
        TFs are part of the gene set.
    device : str, default='cuda'
        Device for computation

    Returns
    -------
    torch.Tensor
        Filtered adjacency matrix with only top-k targets per TF
        Shape: (n_tfs, n_genes)

    Example
    -------
    >>> adj = torch.rand(100, 5000)  # 100 TFs, 5000 genes
    >>> filtered = select_topk_targets(adj, k=50)
    >>> # Each row now has at most 50 non-zero values
    """
    # Convert to tensor if needed
    if isinstance(adj, np.ndarray):
        adj = torch.from_numpy(adj)
    adj = adj.to(device=device, dtype=torch.float32)

    n_tfs, n_genes = adj.shape

    # Get top-k values and indices for each TF
    topk_values, topk_indices = torch.topk(adj, min(k, n_genes), dim=1)

    # Create output matrix with zeros
    output = torch.zeros_like(adj)

    # Scatter top-k values back
    output.scatter_(1, topk_indices, topk_values)

    # Include TF in its own module
    if include_tf and tf_indices is not None:
        if isinstance(tf_indices, np.ndarray):
            tf_indices = torch.from_numpy(tf_indices)
        tf_indices = tf_indices.to(device=device, dtype=torch.long)
        # Set diagonal elements (TF -> TF) to 1.0
        for i, tf_idx in enumerate(tf_indices):
            if tf_idx < n_genes:
                output[i, tf_idx] = 1.0

    return output


def select_threshold_targets(
    adj: ArrayLike,
    threshold: float = 0.0,
    percentile: Optional[float] = None,
    include_tf: bool = True,
    tf_indices: Optional[ArrayLike] = None,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Select targets above threshold from adjacency matrix.

    Parameters
    ----------
    adj : array-like
        Adjacency matrix (n_tfs x n_genes) with edge weights
    threshold : float, default=0.0
        Absolute threshold (edges below this become 0)
    percentile : float, optional
        If provided, use this percentile of non-zero weights as threshold
        (overrides threshold parameter). Value between 0-100.
    include_tf : bool, default=True
        Include TF itself in its module
    tf_indices : array-like, optional
        Index of each TF in the gene list
    device : str, default='cuda'
        Device for computation

    Returns
    -------
    torch.Tensor
        Filtered adjacency matrix
    """
    if isinstance(adj, np.ndarray):
        adj = torch.from_numpy(adj)
    adj = adj.to(device=device, dtype=torch.float32)

    # Calculate threshold from percentile if provided
    if percentile is not None:
        nonzero_values = adj[adj > 0]
        if len(nonzero_values) > 0:
            threshold = torch.quantile(nonzero_values, percentile / 100.0).item()

    # Apply threshold
    output = torch.where(adj >= threshold, adj, torch.zeros_like(adj))

    # Include TF in its own module
    if include_tf and tf_indices is not None:
        if isinstance(tf_indices, np.ndarray):
            tf_indices = torch.from_numpy(tf_indices)
        tf_indices = tf_indices.to(device=device, dtype=torch.long)
        n_genes = adj.shape[1]
        for i, tf_idx in enumerate(tf_indices):
            if tf_idx < n_genes:
                output[i, tf_idx] = 1.0

    return output


def filter_by_min_targets(
    adj: ArrayLike,
    min_targets: int = 20,
    min_fraction: Optional[float] = 0.8,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Filter out TFs with fewer than min_targets or below min_fraction of targets.

    This function supports both absolute count filtering (like pySCENIC's min_genes=20)
    and percentage-based filtering (like pySCENIC's 80% mapping requirement).

    Parameters
    ----------
    adj : array-like
        Adjacency matrix (n_tfs x n_genes)
    min_targets : int, default=20
        Minimum number of non-zero targets required.
        Set to 0 to disable absolute count filtering.
    min_fraction : float or None, default=0.8
        Minimum fraction of total genes that must be non-zero targets.
        Value between 0.0 and 1.0. Default 0.8 matches pySCENIC's behavior
        of skipping modules where less than 80% of genes can be mapped.
        Set to None to disable fraction-based filtering.
    device : str, default='cuda'
        Device for computation

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        - Filtered adjacency matrix (n_valid_tfs x n_genes)
        - Boolean mask indicating which TFs were kept

    Examples
    --------
    >>> adj = torch.rand(100, 5000) > 0.5  # Random binary adjacency
    >>> # Filter by absolute count (default pySCENIC behavior)
    >>> filtered, mask = filter_by_min_targets(adj, min_targets=20)
    >>> # Filter by percentage (pySCENIC's 80% rule)
    >>> filtered, mask = filter_by_min_targets(adj, min_targets=0, min_fraction=0.8)
    >>> # Combine both filters
    >>> filtered, mask = filter_by_min_targets(adj, min_targets=20, min_fraction=0.8)
    """
    if isinstance(adj, np.ndarray):
        adj = torch.from_numpy(adj)
    adj = adj.to(device=device, dtype=torch.float32)

    n_tfs, n_genes = adj.shape

    # Count non-zero targets per TF
    target_counts = (adj > 0).sum(dim=1)

    # Create mask for TFs with enough targets (absolute count)
    mask = target_counts >= min_targets

    # Apply fraction-based filtering if specified
    if min_fraction is not None:
        assert 0.0 <= min_fraction <= 1.0, "min_fraction must be between 0.0 and 1.0"
        min_count_from_fraction = int(n_genes * min_fraction)
        fraction_mask = target_counts >= min_count_from_fraction
        mask = mask & fraction_mask

    return adj[mask], mask


def filter_by_mapped_fraction(
    adj: ArrayLike,
    reference_indices: Optional[ArrayLike] = None,
    min_fraction: float = 0.8,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Filter out TFs where less than min_fraction of targets can be mapped to a reference.

    This mimics pySCENIC's behavior of skipping modules where "less than 80% of
    the genes could be mapped to the ranking database" (transform.py:298-307).

    Parameters
    ----------
    adj : array-like
        Adjacency matrix (n_tfs x n_genes)
    reference_indices : array-like, optional
        Indices of genes that exist in the reference database (e.g., ranking DB).
        If None, uses all genes (no filtering based on mapping).
    min_fraction : float, default=0.8
        Minimum fraction of targets that must be mappable to the reference.
        Default 0.8 matches pySCENIC's 80% threshold.
    device : str, default='cuda'
        Device for computation

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        - Filtered adjacency matrix (n_valid_tfs x n_genes)
        - Boolean mask indicating which TFs were kept

    Notes
    -----
    pySCENIC's logic (from transform.py):
        n_missing = len(module) - len(genes)  # genes not in ranking DB
        frac_missing = float(n_missing) / len(module)
        if frac_missing >= 0.20:  # i.e., less than 80% mapped
            skip this module

    Examples
    --------
    >>> adj = torch.rand(100, 5000) > 0.5  # 100 TFs, 5000 genes
    >>> # Assume only genes 0-4000 are in the ranking database
    >>> db_gene_indices = torch.arange(4000)
    >>> filtered, mask = filter_by_mapped_fraction(adj, db_gene_indices, min_fraction=0.8)
    """
    if isinstance(adj, np.ndarray):
        adj = torch.from_numpy(adj)
    adj = adj.to(device=device, dtype=torch.float32)

    n_tfs, n_genes = adj.shape

    # If no reference provided, keep all TFs
    if reference_indices is None:
        mask = torch.ones(n_tfs, dtype=torch.bool, device=device)
        return adj, mask

    if isinstance(reference_indices, np.ndarray):
        reference_indices = torch.from_numpy(reference_indices)
    reference_indices = reference_indices.to(device=device, dtype=torch.long)

    # Create a mask for genes that are in the reference database
    reference_mask = torch.zeros(n_genes, dtype=torch.bool, device=device)
    reference_mask[reference_indices] = True

    # For each TF, count total targets and targets that can be mapped
    targets_per_tf = adj > 0  # (n_tfs, n_genes) boolean
    total_targets = targets_per_tf.sum(dim=1).float()  # (n_tfs,)

    # Count targets that are also in the reference
    mappable_targets = (targets_per_tf & reference_mask.unsqueeze(0)).sum(dim=1).float()

    # Calculate fraction of targets that can be mapped
    # Avoid division by zero for TFs with no targets
    fraction_mapped = torch.where(
        total_targets > 0,
        mappable_targets / total_targets,
        torch.zeros_like(total_targets)
    )

    # Keep TFs where at least min_fraction of targets are mappable
    mask = fraction_mapped >= min_fraction

    return adj[mask], mask


def get_target_indices(
    adj: ArrayLike,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get indices of non-zero targets for each TF.

    Useful for cisTarget pruning which needs gene indices.

    Parameters
    ----------
    adj : array-like
        Adjacency matrix (n_tfs x n_genes)
    device : str, default='cuda'
        Device for computation

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        - Flat tensor of gene indices
        - Tensor of (start, end) positions for each TF's targets
    """
    if isinstance(adj, np.ndarray):
        adj = torch.from_numpy(adj)
    adj = adj.to(device=device, dtype=torch.float32)

    n_tfs = adj.shape[0]

    # Find non-zero indices
    nonzero = adj > 0

    # Get indices and counts per TF
    all_indices = []
    positions = torch.zeros(n_tfs + 1, dtype=torch.long, device=device)

    for i in range(n_tfs):
        indices = torch.where(nonzero[i])[0]
        all_indices.append(indices)
        positions[i + 1] = positions[i] + len(indices)

    if all_indices:
        flat_indices = torch.cat(all_indices)
    else:
        flat_indices = torch.tensor([], dtype=torch.long, device=device)

    return flat_indices, positions


def binarize(adj: ArrayLike, device: str = 'cuda') -> torch.Tensor:
    """
    Convert weighted adjacency matrix to binary (0/1).

    Parameters
    ----------
    adj : array-like
        Adjacency matrix (n_tfs x n_genes)
    device : str, default='cuda'

    Returns
    -------
    torch.Tensor
        Binary adjacency matrix
    """
    if isinstance(adj, np.ndarray):
        adj = torch.from_numpy(adj)
    adj = adj.to(device=device, dtype=torch.float32)

    return (adj > 0).float()


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array."""
    return tensor.detach().cpu().numpy()
