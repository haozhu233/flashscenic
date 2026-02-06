"""
End-to-end flashscenic pipeline.

Runs GRN inference, module filtering, cisTarget pruning, and AUCell scoring
in a single function call. Takes numpy arrays as input and returns a dict
of results. No AnnData or scanpy dependency.
"""

import numpy as np
import torch
from typing import Dict, List, Optional


def run_flashscenic(
    exp_matrix: np.ndarray,
    gene_names: List[str],
    species: str = "human",
    *,
    # --- Data source / caching ---
    datasource: str = "scenic",
    version: str = "v10",
    cache_dir: Optional[str] = None,
    tf_list_path: Optional[str] = None,
    ranking_db_paths: Optional[List[str]] = None,
    motif_annotation_path: Optional[str] = None,
    # --- GRN inference (RegDiffusion) ---
    grn_n_steps: int = 1000,
    grn_sparsity_threshold: float = 1.5,
    # --- Module filtering ---
    module_k: int = 50,
    module_min_targets: int = 20,
    module_min_fraction: float = 0.8,
    module_include_tf: bool = True,
    # --- cisTarget pruning ---
    pruning_rank_threshold: int = 5000,
    pruning_auc_threshold: float = 0.05,
    pruning_nes_threshold: float = 3.0,
    pruning_min_genes: int = 0,
    pruning_merge_strategy: str = "union",
    # --- Motif annotation filtering ---
    annotation_motif_similarity_fdr: float = 0.001,
    annotation_orthologous_identity: float = 0.0,
    # --- AUCell scoring ---
    aucell_k: Optional[int] = None,
    aucell_auc_threshold: float = 0.05,
    aucell_batch_size: int = 32,
    # --- General ---
    device: str = "cuda",
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Dict:
    """
    Run the complete flashscenic pipeline.

    Performs GRN inference (RegDiffusion), module filtering, cisTarget pruning,
    and AUCell scoring. Stops before dimensionality reduction / visualization.

    Parameters
    ----------
    exp_matrix : np.ndarray
        Expression matrix of shape (n_cells, n_genes). Should be
        log-transformed and (optionally) subset to highly variable genes.
    gene_names : list of str
        Gene names corresponding to columns of exp_matrix. Length must
        equal exp_matrix.shape[1].
    species : str, default='human'
        Species for TF list and ranking databases. One of 'human', 'mouse',
        'drosophila'.

    datasource : str, default='scenic'
        Data source for resource downloads.
    version : str, default='v10'
        Motif collection version.
    cache_dir : str or None
        Cache directory for downloaded resources. Defaults to
        ``./flashscenic_data/``.
    tf_list_path : str or None
        Path to a custom TF list file. Overrides downloaded TF list.
    ranking_db_paths : list of str or None
        Paths to custom ranking database .feather files. Overrides
        downloaded databases.
    motif_annotation_path : str or None
        Path to a custom motif annotation .tbl file. Overrides
        downloaded annotation.

    grn_n_steps : int, default=1000
        Number of training steps for RegDiffusion.
    grn_sparsity_threshold : float, default=1.5
        Edges below this weight are zeroed. Higher = sparser network.

    module_k : int, default=50
        Top target genes per TF for module creation.
    module_min_targets : int, default=20
        Minimum target genes for a TF module to be retained.
    module_min_fraction : float, default=0.8
        Minimum fraction of targets required. Matches pySCENIC's 80% rule.
    module_include_tf : bool, default=True
        Include TF itself in its own module.

    pruning_rank_threshold : int, default=5000
        Maximum rank for cisTarget recovery curve.
    pruning_auc_threshold : float, default=0.05
        Fraction of genome for cisTarget AUC.
    pruning_nes_threshold : float, default=3.0
        NES threshold for motif enrichment.
    pruning_min_genes : int, default=0
        Minimum genes per regulon after pruning.
    pruning_merge_strategy : str, default='union'
        How to merge regulons from multiple databases ('union' or 'best').

    annotation_motif_similarity_fdr : float, default=0.001
        Maximum FDR for motif similarity filtering.
    annotation_orthologous_identity : float, default=0.0
        Minimum orthologous identity threshold.

    aucell_k : int or None
        Top k targets for AUCell scoring. Defaults to module_k if None.
    aucell_auc_threshold : float, default=0.05
        Fraction of genome for AUCell AUC calculation.
    aucell_batch_size : int, default=32
        Batch size for AUCell computation.

    device : str, default='cuda'
        PyTorch device ('cuda' or 'cpu').
    seed : int or None
        Random seed for reproducibility.
    verbose : bool, default=True
        Print progress messages.

    Returns
    -------
    dict
        - ``'auc_scores'``: np.ndarray of shape (n_cells, n_regulons)
        - ``'regulon_names'``: list of regulon name strings
        - ``'regulons'``: list of regulon dicts from cisTarget
        - ``'regulon_adj'``: np.ndarray of shape (n_regulons, n_genes)
        - ``'parameters'``: dict of all parameters used

    Raises
    ------
    ImportError
        If regdiffusion is not installed.
    ValueError
        If no TFs survive filtering or no regulons survive pruning.

    Examples
    --------
    >>> import flashscenic as fs
    >>> result = fs.run_flashscenic(exp_matrix, gene_names, species='human')
    >>> auc_scores = result['auc_scores']  # (n_cells, n_regulons)
    """
    import regdiffusion as rd

    from .data import download_data
    from .aucell import get_aucell
    from .cistarget import CisTargetPruner
    from .modules import select_topk_targets, filter_by_min_targets
    from . import regulons_to_adjacency

    if aucell_k is None:
        aucell_k = module_k

    def _log(msg: str):
        if verbose:
            print(f"[flashscenic] {msg}")

    # Validate inputs
    n_cells, n_genes = exp_matrix.shape
    if len(gene_names) != n_genes:
        raise ValueError(
            f"gene_names length ({len(gene_names)}) != "
            f"exp_matrix columns ({n_genes})"
        )

    # ---- Step 0: Download resources if needed ----
    _log("Step 0/5: Preparing resources...")
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

    # ---- Step 1: GRN Inference ----
    _log(f"Step 1/5: Running RegDiffusion GRN inference "
         f"({n_cells} cells, {n_genes} genes, {grn_n_steps} steps)...")
    exp_float32 = np.asarray(exp_matrix, dtype=np.float32)
    rd_trainer = rd.RegDiffusionTrainer(
        exp_float32, n_steps=grn_n_steps, device=device,
    )
    rd_trainer.train()
    adj_matrix = rd_trainer.get_adj()
    _log(f"  Adjacency matrix: {adj_matrix.shape}")

    # ---- Step 2: TF Filtering ----
    _log("Step 2/5: Filtering to known TFs...")
    with open(tf_list_path, "r") as f:
        known_tfs = set(line.strip() for line in f if line.strip())

    tf_indices = [i for i, g in enumerate(gene_names) if g in known_tfs]
    adj_matrix = adj_matrix[tf_indices, :]
    tf_names = [gene_names[i] for i in tf_indices]

    # Sparsify weak edges
    adj_matrix[adj_matrix < grn_sparsity_threshold] = 0
    _log(f"  {len(tf_names)} TFs found, sparsified at "
         f"threshold={grn_sparsity_threshold}")

    if len(tf_names) == 0:
        raise ValueError(
            "No TFs found in gene_names. Check that the TF list matches "
            "your gene naming convention."
        )

    # ---- Step 3: Module Filtering ----
    _log("Step 3/5: Creating and filtering modules...")
    tf_indices_tensor = torch.tensor(tf_indices, device=device)
    filtered_adj = select_topk_targets(
        adj_matrix,
        k=module_k,
        include_tf=module_include_tf,
        tf_indices=tf_indices_tensor,
        device=device,
    )
    filtered_adj, tf_mask = filter_by_min_targets(
        filtered_adj,
        min_targets=module_min_targets,
        min_fraction=module_min_fraction,
        device=device,
    )

    valid_tf_names = [
        tf_names[i]
        for i, keep in enumerate(tf_mask.cpu().numpy())
        if keep
    ]
    n_valid_tfs = len(valid_tf_names)
    _log(f"  {n_valid_tfs} TFs with >= {module_min_targets} targets "
         f"(min_fraction={module_min_fraction})")

    if n_valid_tfs == 0:
        raise ValueError(
            "No TF modules survived filtering. Consider lowering "
            "module_min_targets, module_min_fraction, or "
            "grn_sparsity_threshold."
        )

    # Build module gene index lists
    modules = []
    for i in range(n_valid_tfs):
        target_mask = filtered_adj[i] > 0
        target_indices = torch.where(target_mask)[0]
        modules.append(target_indices)

    # ---- Step 4: cisTarget Pruning ----
    _log(f"Step 4/5: Running cisTarget pruning "
         f"({len(ranking_db_paths)} databases)...")
    pruner = CisTargetPruner(
        rank_threshold=pruning_rank_threshold,
        auc_threshold=pruning_auc_threshold,
        nes_threshold=pruning_nes_threshold,
        device=device,
        min_genes_per_regulon=pruning_min_genes,
        merge_strategy=pruning_merge_strategy,
    )
    pruner.load_database(ranking_db_paths)
    pruner.load_annotations(
        motif_annotation_path,
        filter_for_annotation=True,
        motif_similarity_fdr=annotation_motif_similarity_fdr,
        orthologous_identity_threshold=annotation_orthologous_identity,
    )

    merged_regulons = pruner.prune_modules(
        modules, valid_tf_names, list(gene_names),
    )
    _log(f"  {len(merged_regulons)} regulons after pruning")

    # Free GPU memory
    pruner.clear_gpu_memory()

    if len(merged_regulons) == 0:
        raise ValueError(
            "No regulons survived cisTarget pruning. Consider lowering "
            "pruning_nes_threshold or grn_sparsity_threshold."
        )

    # ---- Step 5: AUCell Scoring ----
    _log("Step 5/5: Computing AUCell scores...")
    regulon_adj = regulons_to_adjacency(merged_regulons, list(gene_names))

    auc_scores = get_aucell(
        exp_float32,
        regulon_adj,
        k=aucell_k,
        auc_threshold=aucell_auc_threshold,
        device=device,
        batch_size=aucell_batch_size,
        seed=seed,
    )

    regulon_names = [reg["name"] for reg in merged_regulons]
    _log(f"Done! {len(regulon_names)} regulons, "
         f"AUCell scores shape: {auc_scores.shape}")

    return {
        "auc_scores": auc_scores,
        "regulon_names": regulon_names,
        "regulons": merged_regulons,
        "regulon_adj": regulon_adj,
        "parameters": {
            "species": species,
            "datasource": datasource,
            "version": version,
            "grn_n_steps": grn_n_steps,
            "grn_sparsity_threshold": grn_sparsity_threshold,
            "module_k": module_k,
            "module_min_targets": module_min_targets,
            "module_min_fraction": module_min_fraction,
            "module_include_tf": module_include_tf,
            "pruning_rank_threshold": pruning_rank_threshold,
            "pruning_auc_threshold": pruning_auc_threshold,
            "pruning_nes_threshold": pruning_nes_threshold,
            "pruning_min_genes": pruning_min_genes,
            "pruning_merge_strategy": pruning_merge_strategy,
            "annotation_motif_similarity_fdr": annotation_motif_similarity_fdr,
            "annotation_orthologous_identity": annotation_orthologous_identity,
            "aucell_k": aucell_k,
            "aucell_auc_threshold": aucell_auc_threshold,
            "aucell_batch_size": aucell_batch_size,
            "device": device,
            "seed": seed,
            "n_cells": n_cells,
            "n_genes": n_genes,
            "n_tfs": n_valid_tfs,
            "n_regulons": len(merged_regulons),
        },
    }
