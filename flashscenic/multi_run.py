"""
Multi-run flashscenic wrapper with GRN consensus aggregation.

Runs RegDiffusion k times, averages the resulting adjacency matrices into a
consensus, optionally masks high-CV edges, and runs the rest of the pipeline
(module selection, cisTarget pruning, AUCell) once on the stable consensus.

This is architecturally superior to post-pruning frequency aggregation because
it operates on continuous edge weights before any binary thresholding occurs.
"""

import numpy as np
import torch
from typing import Dict, List, Optional


def multi_run_flashscenic(
    exp_matrix: np.ndarray,
    gene_names: List[str],
    species: str = "human",
    *,
    # --- Multi-run / aggregation ---
    n_runs: int = 5,
    cv_threshold: Optional[float] = 1.0,
    return_adj_matrices: bool = False,
    seeds: Optional[List[int]] = None,
    # --- All run_flashscenic kwargs (same defaults) ---
    datasource: str = "scenic",
    version: str = "v10",
    cache_dir: Optional[str] = None,
    tf_list_path: Optional[str] = None,
    ranking_db_paths: Optional[List[str]] = None,
    motif_annotation_path: Optional[str] = None,
    grn_n_steps: int = 1000,
    grn_sparsity_threshold: float = 1.5,
    module_k: int = 50,
    module_percentile_thresholds: tuple = (75,),
    module_top_n_per_target: tuple = (5, 10, 50),
    module_min_targets: int = 20,
    module_min_fraction: float = None,
    module_include_tf: bool = True,
    pruning_rank_threshold: int = 5000,
    pruning_auc_threshold: float = 0.05,
    pruning_nes_threshold: float = 3.0,
    pruning_min_genes: int = 0,
    pruning_merge_strategy: str = "union",
    annotation_motif_similarity_fdr: float = 0.001,
    annotation_orthologous_identity: float = 0.0,
    aucell_k: Optional[int] = None,
    aucell_auc_threshold: float = 0.05,
    aucell_batch_size: int = 32,
    device: str = "cuda",
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Dict:
    """
    Run flashscenic k times, aggregate GRN adjacency matrices, then run the
    downstream pipeline once on the consensus.

    Parameters
    ----------
    exp_matrix : np.ndarray
        Expression matrix of shape (n_cells, n_genes). Should be
        log-transformed and (optionally) subset to highly variable genes.
    gene_names : list of str
        Gene names corresponding to columns of exp_matrix.
    species : str, default='human'
        Species for TF list and ranking databases.

    n_runs : int, default=5
        Number of independent RegDiffusion GRN inference runs.
    cv_threshold : float or None, default=1.0
        Coefficient of variation threshold for edge masking. Edges with
        std/mean >= cv_threshold are zeroed in the consensus. Set to None
        to use the raw mean without masking.

        Recommended values:

        +-----------+--------------------------------------------------+
        | threshold | effect                                           |
        +===========+==================================================+
        | None      | no filtering — use raw mean                      |
        | 2.0       | lenient: only highly unstable edges removed      |
        | 1.0       | balanced: std > mean means noise-dominated       |
        | 0.5       | strict: only edges with std < half the mean kept |
        +-----------+--------------------------------------------------+

    return_adj_matrices : bool, default=False
        If True, include all k raw adjacency matrices in the output under
        ``'adj_matrices'``. Disabled by default to save memory for large
        expression matrices.
    seeds : list of int or None
        Per-run random seeds passed to ``torch.manual_seed`` before each
        RegDiffusion training. Length must equal n_runs. If None, all runs
        are fully stochastic.

    All remaining parameters are forwarded unchanged to
    :func:`run_flashscenic` (Steps 2–5 only).

    Returns
    -------
    dict
        All fields returned by :func:`run_flashscenic`, plus:

        - ``'adj_mean'``: np.ndarray (n_genes, n_genes) — consensus adjacency
        - ``'adj_cv'``: np.ndarray (n_genes, n_genes) — per-edge CV
        - ``'adj_matrices'``: list of np.ndarray or None
        - ``'n_runs'``: int
        - ``'cv_threshold'``: float or None
        - ``'n_edges_cv_filtered'``: int — edges removed by the CV mask

    Raises
    ------
    ValueError
        If seeds is provided but its length does not equal n_runs.
    """
    import regdiffusion as rd

    from .data import download_data
    from .pipeline import run_flashscenic

    if seeds is not None and len(seeds) != n_runs:
        raise ValueError(
            f"seeds length ({len(seeds)}) must equal n_runs ({n_runs})"
        )

    def _log(msg: str):
        if verbose:
            print(f"[flashscenic] {msg}")

    # ---- Phase 0: Pre-download resources once ----
    _log("Preparing resources...")
    if (tf_list_path is None
            or ranking_db_paths is None
            or motif_annotation_path is None):
        resources = download_data(
            species=species,
            version=version,
            datasource=datasource,
            cache_dir=cache_dir,
        )
        if tf_list_path is None:
            tf_list_path = str(resources.tf_list)
        if ranking_db_paths is None:
            ranking_db_paths = [str(p) for p in resources.ranking_dbs]
        if motif_annotation_path is None:
            motif_annotation_path = str(resources.motif_annotation)

    # ---- Phase 1: Run RegDiffusion k times ----
    _log(f"Running RegDiffusion {n_runs} times...")
    exp_float32 = np.asarray(exp_matrix, dtype=np.float32)
    adj_matrices = []

    for i in range(n_runs):
        if seeds is not None:
            torch.manual_seed(seeds[i])
        trainer = rd.RegDiffusionTrainer(
            exp_float32, n_steps=grn_n_steps, device=device,
        )
        trainer.train()
        adj_matrices.append(trainer.get_adj())  # numpy (n_genes, n_genes)
        _log(f"  GRN run {i + 1}/{n_runs} done")

    # ---- Phase 2: Aggregate adjacency matrices ----
    _log("Aggregating adjacency matrices...")
    A_stack = np.stack(adj_matrices, axis=0)   # (k, n_genes, n_genes)
    A_mean = A_stack.mean(axis=0)              # (n_genes, n_genes)
    A_std = A_stack.std(axis=0)
    # CV = 0 where mean == 0 to avoid division by zero
    A_cv = np.divide(A_std, A_mean, out=np.zeros_like(A_mean), where=A_mean > 0)

    if cv_threshold is not None:
        A_consensus = np.where(A_cv < cv_threshold, A_mean, 0.0).astype(np.float32)
        n_filtered = int((A_mean > 0) & (A_cv >= cv_threshold)).sum() if False else int(
            np.sum((A_mean > 0) & (A_cv >= cv_threshold))
        )
        _log(f"  CV filter (threshold={cv_threshold}) removed {n_filtered} edges")
    else:
        A_consensus = A_mean.astype(np.float32)
        n_filtered = 0

    # ---- Phase 3: Run Steps 2–5 once on consensus ----
    _log("Running downstream pipeline on consensus adjacency...")
    result = run_flashscenic(
        exp_matrix,
        gene_names,
        species,
        adj_matrix=A_consensus,
        tf_list_path=tf_list_path,
        ranking_db_paths=ranking_db_paths,
        motif_annotation_path=motif_annotation_path,
        grn_sparsity_threshold=grn_sparsity_threshold,
        module_k=module_k,
        module_percentile_thresholds=module_percentile_thresholds,
        module_top_n_per_target=module_top_n_per_target,
        module_min_targets=module_min_targets,
        module_min_fraction=module_min_fraction,
        module_include_tf=module_include_tf,
        pruning_rank_threshold=pruning_rank_threshold,
        pruning_auc_threshold=pruning_auc_threshold,
        pruning_nes_threshold=pruning_nes_threshold,
        pruning_min_genes=pruning_min_genes,
        pruning_merge_strategy=pruning_merge_strategy,
        annotation_motif_similarity_fdr=annotation_motif_similarity_fdr,
        annotation_orthologous_identity=annotation_orthologous_identity,
        aucell_k=aucell_k,
        aucell_auc_threshold=aucell_auc_threshold,
        aucell_batch_size=aucell_batch_size,
        device=device,
        seed=seed,
        verbose=verbose,
    )

    # ---- Phase 4: Attach multi-run diagnostics ----
    result['adj_mean'] = A_mean
    result['adj_cv'] = A_cv
    result['adj_matrices'] = adj_matrices if return_adj_matrices else None
    result['n_runs'] = n_runs
    result['cv_threshold'] = cv_threshold
    result['n_edges_cv_filtered'] = n_filtered
    result['parameters'].update({
        'n_runs': n_runs,
        'cv_threshold': cv_threshold,
        'seeds': seeds,
    })

    return result
