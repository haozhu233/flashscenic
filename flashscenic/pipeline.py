"""
End-to-end flashscenic pipeline.

Runs GRN inference, module filtering, cisTarget pruning, and AUCell scoring
in a single function call. Takes numpy arrays as input and returns a dict
of results. No AnnData or scanpy dependency.
"""

import numpy as np
import torch
from typing import Dict, List, Optional
from tqdm.auto import tqdm


def run_flashscenic(
    exp_matrix: np.ndarray,
    gene_names: List[str],
    cell_names: Optional[List[str]] = None,
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
    module_percentile_thresholds: tuple = (75,),
    module_top_n_per_target: tuple = (5, 10, 50),
    module_min_targets: int = 20,
    module_min_fraction: float = None,
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
    # --- Multirun ---
    multirun: int = 1,
    threshold_occurrence: float = 0.8,
    min_gene_per_regulon: int = 5,
    tqdm_multirun: bool = True,
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
    cell_names : list of str or None
        Cell names corresponding to rows of exp_matrix. Length must
        equal exp_matrix.shape[0]. Optional, used for AUCell output.
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
    module_percentile_thresholds : tuple of int, default=(75,)
        Percentile thresholds for percentile-based modules. Each value
        creates a module type keeping targets above that global weight
        percentile. Empty tuple to skip.
    module_top_n_per_target : tuple of int, default=(5, 10, 50)
        N values for top-N-per-target modules. For each N, finds each
        target gene's top N strongest regulators, then regroups by TF.
        Empty tuple to skip.
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

    multirun : int, default=1
        Number of iterations of pipeline runs before AUC computation to
        increase robustness. Default to no multirun.
    threshold_occurrence : float, default=0.8
        Minimum occurrence in all runs of target gene per regulon. Default
        to 80% as pySCENIC multirun default.
    min_gene_per_regulon : int, default=5
        Minimum of target genes in each regulon. Default to 5 as pySCENIC
        multirun default.
    tqdm_multirun : bool, default=True
        Whether to show a tqdm progress bar for multirun iterations
        or for each regdiffusion step. Set to False to log runs instead.

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
        - ``'regulons'``: list of regulon dicts from cisTarget merged for all runs
        - ``'regulon_adj'``: np.ndarray of shape (n_regulons, n_genes)
        - ``'regulon_occurrence'``: np.ndarray of shape (n_regulons, n_genes)
        - ``'regulon_names_occurrence'``: list of regulon name strings for occurrence matrix
        - ``'genes_names'``: list of gene name strings
        - ``'cell_names'``: list of cell name strings
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
    from .modules import (
       select_topk_targets,
       select_threshold_targets,
       select_top_n_per_target,
       filter_by_min_targets,
    )
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
            f"gene_names length ({len(gene_names)}) != exp_matrix columns ({n_genes})"
        )

    # ---- Step 0: Download resources if needed ----
    _log("Step 0/6: Preparing resources...")
    if (
        tf_list_path is None
        or ranking_db_paths is None
        or motif_annotation_path is None
    ):
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

    # Initiate multirun
    multirun_adj = []
    merged_regulon_dict = {}

    for run_id in tqdm(range(multirun), disable=not tqdm_multirun, desc="Multirun iterations"):
        if not tqdm_multirun:
            _log(
                f"Starting run {run_id + 1} on {multirun}"
            )  # Python index at 0, adding one so it is more intuitive
        
        # ---- Step 1: GRN Inference ----
        _log(
            f"Step 1/6: Running RegDiffusion GRN inference "
            f"({n_cells} cells, {n_genes} genes, {grn_n_steps} steps)..."
        )
        exp_float32 = np.asarray(exp_matrix, dtype=np.float32)
        rd_trainer = rd.RegDiffusionTrainer(
            exp_float32,
            n_steps=grn_n_steps,
            device=device,
        )
        rd_trainer.train()
        adj_matrix = rd_trainer.get_adj()
        _log(f"  Adjacency matrix: {adj_matrix.shape}")

        # ---- Step 2: TF Filtering ----
        _log("Step 2/6: Filtering to known TFs...")
        with open(tf_list_path, "r") as f:
            known_tfs = set(line.strip() for line in f if line.strip())

        tf_indices = [i for i, g in enumerate(gene_names) if g in known_tfs]
        adj_matrix = adj_matrix[tf_indices, :]
        tf_names = [gene_names[i] for i in tf_indices]

        # Sparsify weak edges
        adj_matrix[adj_matrix < grn_sparsity_threshold] = 0
        _log(
            f"  {len(tf_names)} TFs found, sparsified at "
            f"threshold={grn_sparsity_threshold}"
        )

        if len(tf_names) == 0:
            raise ValueError(
                "No TFs found in gene_names. Check that the TF list matches "
                "your gene naming convention."
            )

        # ---- Step 3: Module Filtering ----
        _log("Step 3/6: Creating and filtering modules...")
        tf_indices_tensor = torch.tensor(tf_indices, device=device)
        adj_tensor = (
            torch.from_numpy(adj_matrix).to(device=device, dtype=torch.float32)
            if isinstance(adj_matrix, np.ndarray)
            else adj_matrix
        )

        # Generate multiple module types from the same adjacency matrix
        module_type_adjs = []  # list of (filtered_adj, label)

        # Type 1: Top-k targets (always included)
        topk_adj = select_topk_targets(
            adj_tensor,
            k=module_k,
            include_tf=module_include_tf,
            tf_indices=tf_indices_tensor,
            device=device,
        )
        module_type_adjs.append((topk_adj, f"top{module_k}"))

        # Type 2: Percentile threshold modules
        for pct in module_percentile_thresholds:
            pct_adj = select_threshold_targets(
                adj_tensor,
                percentile=pct,
                include_tf=module_include_tf,
                tf_indices=tf_indices_tensor,
                device=device,
            )
            module_type_adjs.append((pct_adj, f"pct{pct}"))

        # Type 3: Top-N-per-target modules
        for n_val in module_top_n_per_target:
            npt_adj = select_top_n_per_target(
                adj_tensor,
                n=n_val,
                include_tf=module_include_tf,
                tf_indices=tf_indices_tensor,
                device=device,
            )
            module_type_adjs.append((npt_adj, f"top{n_val}pertarget"))

        _log(
            f"  {len(module_type_adjs)} module types: "
            f"{[label for _, label in module_type_adjs]}"
        )

        # Filter each module type and collect all modules + TF names
        modules = []
        valid_tf_names = []
        total_modules = 0

        for type_adj, label in module_type_adjs:
            filtered_adj, tf_mask = filter_by_min_targets(
                type_adj,
                min_targets=module_min_targets,
                min_fraction=module_min_fraction,
                device=device,
            )
            mask_np = tf_mask.cpu().numpy()
            type_tf_names = [tf_names[i] for i, keep in enumerate(mask_np) if keep]
            n_type_tfs = len(type_tf_names)
            total_modules += n_type_tfs

            for i in range(n_type_tfs):
                target_mask = filtered_adj[i] > 0
                target_indices = torch.where(target_mask)[0]
                modules.append(target_indices)

            valid_tf_names.extend(type_tf_names)
            _log(f"    {label}: {n_type_tfs} TF modules")

        _log(f"  {total_modules} total modules across all types")

        if total_modules == 0:
            raise ValueError(
                "No TF modules survived filtering. Consider lowering "
                "module_min_targets, module_min_fraction, or "
                "grn_sparsity_threshold."
            )

        # ---- Step 4: cisTarget Pruning ----
        _log(
            f"Step 4/6: Running cisTarget pruning "
            f"({len(ranking_db_paths)} databases)..."
        )
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
            modules,
            valid_tf_names,
            list(gene_names),
        )
        _log(f"  {len(merged_regulons)} regulons after pruning")

        # Free GPU memory
        pruner.clear_gpu_memory()

        if len(merged_regulons) == 0:
            raise ValueError(
                "No regulons survived cisTarget pruning. Consider lowering "
                "pruning_nes_threshold or grn_sparsity_threshold."
            )

        # ---- Step 5: Merge multirun results ----
        # Handle adjacencies from multiruns
        regulon_adj_run = regulons_to_adjacency(merged_regulons, list(gene_names))
        regulon_names_run = [reg["name"] for reg in merged_regulons]
        multirun_adj.append(
            {"matrix": regulon_adj_run, "regulon_names": regulon_names_run}
        )

        # Put all regulons results to a list and remove genes and n_genes for rebuilding with adjacency
        for reg in merged_regulons:
            # remove number of genes and genes to rebuild with adjacency
            reg.pop("genes", None)
            reg.pop("n_genes", None)

            # Convert other non-name/tf keys to single-item lists to be merged on multiple passes
            for key in reg.keys() - {"name", "tf"}:
                reg[key] = [reg[key]]

            # Merge in the dictionnary on "name" key
            reg_name = reg["name"]
            if reg_name in merged_regulon_dict:
                for key in reg.keys() - {"name", "tf"}:
                    merged_regulon_dict[reg_name][key].extend(reg[key])
                    # Remove duplicate in each item
                    merged_regulon_dict[reg_name][key] = list(
                        set(merged_regulon_dict[reg_name][key])
                    )
            else:
                merged_regulon_dict[reg_name] = reg

    # End of multirun for loop
    # Covert back to a list to respect previous type
    _log("Step 5/6: Merging multiruns...")
    multirun_regulon = list(merged_regulon_dict.values())

    # Make occurrence matrix
    # Collect all TFs names across runs
    regulon_names_occurrence = sorted(
        set(tf for run in multirun_adj for tf in run["regulon_names"])
    )

    n_tfs = len(regulon_names_occurrence)
    n_genes = len(gene_names)
    tf_index = {tf: i for i, tf in enumerate(regulon_names_occurrence)}

    # Accumulate counts into a (n_all_tfs, n_genes) matrix
    accumulated = np.zeros((n_tfs, n_genes), dtype=float)

    for run in multirun_adj:
        for local_row, tf in enumerate(run["regulon_names"]):
            global_row = tf_index[tf]
            accumulated[global_row] += run["matrix"][local_row]

    # Divide by number of runs to make the occurence matrix
    regulon_occ_adj = accumulated / len(multirun_adj)

    # Binarize
    regulon_adj = np.where(regulon_occ_adj >= threshold_occurrence, 1, 0)

    # Find TFs that have at least min_gene_per_regulon target genes
    tf_mask = regulon_adj.sum(axis=1) >= min_gene_per_regulon

    # Filter both the matrix and the TF list
    regulon_adj = regulon_adj[tf_mask]
    regulon_names = [
        regulon for regulon, keep in zip(regulon_names_occurrence, tf_mask) if keep
    ]

    # Get the gene names and number per regulon in a dictionnary
    regulons_genes_map = {}
    for i, reg_name in enumerate(regulon_names):
        consensus_gene_mask = regulon_adj[i].astype(bool)
        consensus_genes = [
            g for g, keep in zip(gene_names, consensus_gene_mask) if keep
        ]

        regulons_genes_map[reg_name] = {
            "n_genes": len(consensus_genes),
            "genes": consensus_genes,
        }

    # Filter the regulon on adjacency and add gene names and number back in a dictionnary
    regulons_filtered_adj = []
    for entry in multirun_regulon:
        if entry["name"] in set(regulon_names):
            entry["n_genes"] = regulons_genes_map[entry["name"]]["n_genes"]
            entry["genes"] = regulons_genes_map[entry["name"]]["genes"]
            regulons_filtered_adj.append(entry)

    # ---- Step 6: AUCell Scoring ----
    _log("Step 6/6: Computing AUCell scores...")

    auc_scores = get_aucell(
        exp_float32,
        regulon_adj,
        k=aucell_k,
        auc_threshold=aucell_auc_threshold,
        device=device,
        batch_size=aucell_batch_size,
        seed=seed,
    )

    _log(
        f"Done! {len(regulon_names)} regulons, AUCell scores shape: {auc_scores.shape}"
    )

    return {
        "auc_scores": auc_scores,
        "regulon_names": regulon_names,
        "regulons": regulons_filtered_adj,
        "regulon_adj": regulon_adj,
        "regulon_occurrence": regulon_occ_adj,
        "regulon_names_occurrence": regulon_names_occurrence,
        "gene_names": gene_names,
        "cell_names": cell_names,
        "parameters": {
            "species": species,
            "datasource": datasource,
            "version": version,
            "number_of_runs": multirun,
            "grn_n_steps": grn_n_steps,
            "grn_sparsity_threshold": grn_sparsity_threshold,
            "module_k": module_k,
            "module_percentile_thresholds": module_percentile_thresholds,
            "module_top_n_per_target": module_top_n_per_target,
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
            "min_gene_per_regulon": min_gene_per_regulon,
            "threshold_occurrence": threshold_occurrence,
            "aucell_k": aucell_k,
            "aucell_auc_threshold": aucell_auc_threshold,
            "aucell_batch_size": aucell_batch_size,
            "device": device,
            "seed": seed,
            "n_cells": n_cells,
            "n_genes": n_genes,
            "n_modules": total_modules,
            "n_regulons": len(regulons_filtered_adj),
        },
    }
