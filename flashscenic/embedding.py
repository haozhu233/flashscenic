import numpy as np
import torch


def gpu_knn(X, n_neighbors, device='cuda', batch_size=1024):
    """
    Exact brute-force k-nearest-neighbors on GPU (Euclidean metric).

    Computes distances in batches of query points against all points, so peak
    memory scales as ~ ``batch_size * n_samples * 4`` bytes. For very large
    datasets, lower ``batch_size``.

    The point itself is returned as its own first neighbor (distance ~0), which
    matches the convention umap-learn expects for a precomputed kNN.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features). For flashscenic this is
        typically the AUCell score matrix (n_cells, n_regulons).
    n_neighbors : int
        Number of neighbors to return per point (including self).
    device : str
        Device, 'cpu' or 'cuda'. Default is 'cuda'.
    batch_size : int
        Number of query points per batch. Default is 1024.

    Returns
    -------
    knn_indices : np.ndarray
        Int64 array of shape (n_samples, n_neighbors).
    knn_dists : np.ndarray
        Float32 array of shape (n_samples, n_neighbors), Euclidean distances.
    """
    X_tensor = torch.as_tensor(X, device=device, dtype=torch.float32)
    n_samples = X_tensor.shape[0]
    k = min(n_neighbors, n_samples)

    knn_indices = np.empty((n_samples, k), dtype=np.int64)
    knn_dists = np.empty((n_samples, k), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            query = X_tensor[i:min(i + batch_size, n_samples), :]
            dists = torch.cdist(query, X_tensor)  # (batch, n_samples)
            batch_dists, batch_idx = torch.topk(dists, k, dim=1, largest=False)
            end = i + query.shape[0]
            knn_indices[i:end] = batch_idx.cpu().numpy()
            knn_dists[i:end] = batch_dists.cpu().numpy()

    return knn_indices, knn_dists


def run_umap(X, n_neighbors=15, n_components=2, min_dist=0.1, metric='euclidean',
             device='cuda', knn_batch_size=1024, random_state=42, **umap_kwargs):
    """
    UMAP embedding with a GPU-computed k-nearest-neighbor graph.

    The kNN graph (the expensive part on large single-cell data) is computed on
    the GPU with :func:`gpu_knn`, then handed to umap-learn as a precomputed kNN
    so the fuzzy-simplicial-set construction and layout optimization run on CPU
    without repeating the neighbor search. This avoids the heavy RAPIDS/cuML
    dependency while still accelerating the bottleneck.

    Requires the optional ``umap-learn`` dependency:
    ``pip install flashscenic[viz]``.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features), e.g. the AUCell score
        matrix (n_cells, n_regulons).
    n_neighbors : int
        UMAP ``n_neighbors`` (also the k used for the kNN graph). Default is 15.
    n_components : int
        Embedding dimensionality. Default is 2.
    min_dist : float
        UMAP ``min_dist``. Default is 0.1.
    metric : str
        Metric for the UMAP layout. The GPU kNN uses Euclidean distances, so
        keep this 'euclidean' (the default) for consistent results.
    device : str
        Device for the kNN computation, 'cpu' or 'cuda'. Default is 'cuda'.
    knn_batch_size : int
        Query batch size for the GPU kNN. Lower it if you hit GPU OOM on large
        datasets. Default is 1024.
    random_state : int
        Random seed passed to UMAP for reproducibility. Default is 42.
    **umap_kwargs
        Extra keyword arguments forwarded to ``umap.UMAP``.

    Returns
    -------
    np.ndarray
        Embedding of shape (n_samples, n_components).
    """
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "run_umap requires umap-learn. Install it with "
            "`pip install flashscenic[viz]` or `pip install umap-learn`."
        ) from exc

    X = np.asarray(X, dtype=np.float32)
    knn_indices, knn_dists = gpu_knn(
        X, n_neighbors, device=device, batch_size=knn_batch_size
    )

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric=metric,
        precomputed_knn=(knn_indices, knn_dists, None),
        random_state=random_state,
        **umap_kwargs,
    )
    return reducer.fit_transform(X)
