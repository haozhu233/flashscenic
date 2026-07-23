"""
Multi-run flashscenic wrapper with GRN consensus aggregation.

Runs RegDiffusion k times, then aggregates across runs using one of two
strategies (``aggregation_method``):

- ``"frequency"`` (default): runs the *full* pipeline (through cisTarget
  pruning) k times independently, then keeps a (TF, gene) edge only if it
  survives pruning in at least ``frequency_threshold`` fraction of runs,
  dropping any TF whose surviving target count falls below
  ``min_genes_filter``. AUCell is then run once on the consensus regulons.
  Validated against an independent TF-target database (CollecTRI),
  "frequency >= 50% + min 5 genes" produced the best precision and recall of
  every strategy tested — see ``experiments/robustness/``.
- ``"mean_adjacency"``: averages the k adjacency matrices into a continuous
  consensus (optionally masking high-CV edges) and runs the rest of the
  pipeline (module selection, cisTarget pruning, AUCell) once on it. Cheaper
  — the downstream steps only run once instead of k times — but empirically
  infers lower-quality regulons than frequency aggregation.
"""

import gc
import warnings
from collections import defaultdict

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple


def _build_frequency_consensus(
    full_results: List[Dict],
    gene_names: List[str],
    frequency_threshold: float,
    min_genes_filter: int,
) -> Tuple[List[Dict], np.ndarray, Dict[str, float]]:
    """
    Aggregate regulons from independent full-pipeline runs by (TF, gene)
    edge frequency.

    For each (TF, gene) pair, counts how many of the runs' regulons contain
    it. Keeps pairs whose count/n_runs >= frequency_threshold, then drops
    any TF whose surviving target-gene count is below
    ``max(1, min_genes_filter)``.

    Returns
    -------
    consensus_regulons : list of dict
        Regulon dicts in the same schema as CisTargetPruner's output
        (``name``, ``tf``, ``motif``, ``n_genes``, ``genes``, ``context``,
        ``nes``, ``auc``, ``database``). Empty list if nothing survives.
    regulon_adj : np.ndarray
        Shape (n_consensus_regulons, n_genes). Shape (0, n_genes) if
        ``consensus_regulons`` is empty.
    tf_frequency_scores : dict
        Per-TF mean pair-frequency among its retained genes.
    """
    from . import regulons_to_adjacency

    n_runs = len(full_results)
    tf_gene_counts = defaultdict(int)
    tf_meta = {}
    for result in full_results:
        for reg in result['regulons']:
            tf = reg['tf']
            if tf not in tf_meta:
                tf_meta[tf] = reg
            for gene in reg['genes']:
                tf_gene_counts[(tf, gene)] += 1

    tf_genes = defaultdict(list)
    tf_freqs = defaultdict(list)
    for (tf, gene), count in tf_gene_counts.items():
        freq = count / n_runs
        if freq >= frequency_threshold:
            tf_genes[tf].append(gene)
            tf_freqs[tf].append(freq)

    min_genes = max(1, min_genes_filter)
    consensus_regulons = []
    tf_frequency_scores = {}
    for tf in sorted(tf_genes):
        genes = tf_genes[tf]
        if len(genes) < min_genes:
            continue
        meta = tf_meta[tf]
        consensus_regulons.append({
            'name': f"{tf}(+)",
            'tf': tf,
            'motif': meta.get('motif', ''),
            'n_genes': len(genes),
            'genes': genes,
            'context': meta.get('context', ''),
            'nes': meta.get('nes', 0.0),
            'auc': meta.get('auc', 0.0),
            'database': meta.get('database', ''),
        })
        tf_frequency_scores[tf] = float(np.mean(tf_freqs[tf]))

    if not consensus_regulons:
        return [], np.zeros((0, len(gene_names)), dtype=np.float32), {}

    regulon_adj = regulons_to_adjacency(consensus_regulons, gene_names)
    return consensus_regulons, regulon_adj, tf_frequency_scores


def multi_run_flashscenic(
    exp_matrix: np.ndarray,
    gene_names: List[str],
    species: str = "human",
    *,
    # --- Multi-run / aggregation ---
    n_runs: int = 15,
    aggregation_method: str = "frequency",
    frequency_threshold: float = 0.5,
    min_genes_filter: int = 5,
    cv_threshold: Optional[float] = None,
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
    Run flashscenic k times and aggregate into a consensus set of regulons.

    Parameters
    ----------
    exp_matrix : np.ndarray
        Expression matrix of shape (n_cells, n_genes). Should be
        log-transformed and (optionally) subset to highly variable genes.
    gene_names : list of str
        Gene names corresponding to columns of exp_matrix.
    species : str, default='human'
        Species for TF list and ranking databases.

    n_runs : int, default=15
        Number of independent RegDiffusion GRN inference runs.
    aggregation_method : str, default='frequency'
        How to combine the k runs into a consensus:

        - ``'frequency'``: runs the full pipeline (module selection through
          cisTarget pruning) k times independently, keeps a (TF, gene) edge
          if it survives pruning in >= ``frequency_threshold`` fraction of
          runs, drops TFs below ``min_genes_filter`` retained targets, then
          runs AUCell once on the consensus regulons. More expensive (k full
          pipeline passes instead of k RegDiffusion-only passes), but
          empirically the better-performing strategy (see below).
        - ``'mean_adjacency'``: averages the k adjacency matrices (optionally
          CV-masked) and runs the downstream pipeline once on the consensus.
          Cheaper, but validated to infer lower-quality regulons.

        Robustness testing against an independent TF-target database
        (CollecTRI) found ``'frequency'`` with the defaults below
        (``frequency_threshold=0.5``, ``min_genes_filter=5``) had the best
        precision and recall of every strategy tested, including every
        ``'mean_adjacency'`` variant. See ``experiments/robustness/``.
    frequency_threshold : float, default=0.5
        Fraction of runs (0–1) a (TF, gene) pair must survive cisTarget
        pruning in to be kept in the consensus. Only used when
        ``aggregation_method='frequency'``.
    min_genes_filter : int, default=5
        Minimum number of target genes a regulon must retain to survive.
        Applies to both aggregation methods:

        - ``'frequency'``: applied *after* frequency aggregation — any TF
          whose consensus target count is below this is dropped.
        - ``'mean_adjacency'``: composed with ``pruning_min_genes`` on the
          single downstream ``run_flashscenic`` call as
          ``max(pruning_min_genes, min_genes_filter)`` — whichever floor is
          stricter wins.
    cv_threshold : float or None, default=None
        Coefficient of variation threshold for edge masking. Edges with
        std/|mean| >= cv_threshold are zeroed in the consensus. Only used
        when ``aggregation_method='mean_adjacency'`` — a warning is raised
        if set together with ``aggregation_method='frequency'``, since it
        would otherwise silently have no effect.

        In practice, thresholds >= 1.0 rarely filter anything: edges that
        survive `grn_sparsity_threshold` (Step 2) already tend to have low
        run-to-run variability, so CV masking only has a visible effect
        around 0.25 or lower. Robustness testing against an independent TF-
        target database (CollecTRI) did not find that strict CV filtering
        improves downstream regulon quality — stricter thresholds performed
        no better, and sometimes worse, than the unfiltered mean. Treat this
        as an available knob for your own experimentation rather than
        something to enable by default.

    return_adj_matrices : bool, default=False
        If True, include all k raw adjacency matrices in the output under
        ``'adj_matrices'``. Disabled by default to save memory for large
        expression matrices.
    seeds : list of int or None
        Per-run random seeds passed to ``torch.manual_seed`` before each
        RegDiffusion training. Length must equal n_runs. If None and
        ``seed`` is also None, all runs are fully stochastic. If None and
        ``seed`` is set, n_runs per-run seeds are deterministically derived
        from ``seed`` (see below) — explicit ``seeds`` always take priority.

    All remaining parameters are forwarded unchanged to
    :func:`run_flashscenic`, with one exception: ``seed`` additionally seeds
    the ensemble itself. If ``seed`` is set and ``seeds`` is not, it is
    expanded into n_runs per-run seeds (via
    ``numpy.random.SeedSequence(seed).spawn(n_runs)``) so the whole
    multi-run — GRN ensemble and the downstream AUCell tie-breaking — is
    reproducible from a single seed. ``seed`` is also forwarded as-is to
    every downstream ``run_flashscenic`` call for AUCell.

    Returns
    -------
    dict
        Same fields as :func:`run_flashscenic` (``auc_scores``,
        ``regulon_names``, ``regulons``, ``regulon_adj``, ``parameters``),
        plus:

        - ``'adj_mean'``: np.ndarray (n_genes, n_genes) — mean adjacency
          across the k runs (the consensus used for Steps 2-5 when
          ``aggregation_method='mean_adjacency'``; a diagnostic only when
          ``aggregation_method='frequency'``)
        - ``'adj_cv'``: np.ndarray (n_genes, n_genes) — per-edge CV
        - ``'adj_matrices'``: list of np.ndarray or None
        - ``'n_runs'``: int
        - ``'aggregation_method'``: str
        - ``'frequency_threshold'``: float
        - ``'min_genes_filter'``: int
        - ``'cv_threshold'``: float or None
        - ``'n_edges_cv_filtered'``: int — edges removed by the CV mask
          (``'mean_adjacency'`` only, always 0 under ``'frequency'``)
        - ``'tf_frequency_scores'``: dict mapping TF name to its mean
          pair-frequency among retained genes, or None under
          ``'mean_adjacency'``

    Raises
    ------
    ValueError
        If seeds is provided but its length does not equal n_runs, if
        ``aggregation_method`` is not one of ``'mean_adjacency'`` /
        ``'frequency'``, if ``frequency_threshold`` is not in (0, 1], or if
        no regulons survive frequency aggregation.
    """
    import regdiffusion as rd

    from .data import download_data
    from .pipeline import run_flashscenic

    if aggregation_method not in ("mean_adjacency", "frequency"):
        raise ValueError(
            "aggregation_method must be 'mean_adjacency' or 'frequency', "
            f"got {aggregation_method!r}"
        )
    if not (0 < frequency_threshold <= 1):
        raise ValueError(
            f"frequency_threshold must be in (0, 1], got {frequency_threshold}"
        )
    if cv_threshold is not None and aggregation_method == "frequency":
        warnings.warn(
            "cv_threshold has no effect when aggregation_method='frequency' "
            "(CV masking only applies to the 'mean_adjacency' consensus).",
            stacklevel=2,
        )

    if seeds is not None and len(seeds) != n_runs:
        raise ValueError(
            f"seeds length ({len(seeds)}) must equal n_runs ({n_runs})"
        )

    if seeds is None and seed is not None:
        # Derive n_runs independent per-run seeds from the single `seed` so
        # the whole ensemble (not just the downstream AUCell step) is
        # reproducible from one value. SeedSequence.spawn avoids the subtle
        # correlations that naive `seed + i` offsets can introduce.
        seeds = [
            int(child.generate_state(1)[0])
            for child in np.random.SeedSequence(seed).spawn(n_runs)
        ]

    n_genes = exp_matrix.shape[1]
    if len(gene_names) != n_genes:
        raise ValueError(
            f"gene_names length ({len(gene_names)}) != "
            f"exp_matrix columns ({n_genes})"
        )

    if n_runs < 1:
        raise ValueError(f"n_runs must be >= 1, got {n_runs}")
    if n_runs < 2:
        warnings.warn(
            f"n_runs={n_runs}: with fewer than 2 runs, per-edge std is always "
            "0, so adj_cv is always 0 and cv_threshold has no filtering effect.",
            stacklevel=2,
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
    if seeds is not None:
        _log(f"Running RegDiffusion {n_runs} times with seeds={seeds}...")
    else:
        _log(f"Running RegDiffusion {n_runs} times (unseeded, stochastic)...")
    exp_float32 = np.asarray(exp_matrix, dtype=np.float32)
    adj_matrices = []

    for i in range(n_runs):
        if seeds is not None:
            torch.manual_seed(seeds[i])
        trainer = rd.RegDiffusionTrainer(
            exp_float32, n_steps=grn_n_steps, device=device,
        )
        trainer.train()
        # get_adj() returns float16; upcast immediately so the mean/std/CV
        # aggregation below isn't computed at float16 precision (numpy does
        # not promote float16 accumulators the way it does int/bool).
        adj_matrices.append(trainer.get_adj().astype(np.float32))  # (n_genes, n_genes)
        # Each iteration's RegDiffusionTrainer holds its own model, optimizer
        # state, and CUDA tensors; without an explicit release, PyTorch's
        # caching allocator keeps all of it live across the loop, and peak
        # memory grows roughly linearly with n_runs. Freeing it here is what
        # lets n_runs=30 on a full transcriptome fit in a single A100 (this
        # was silently dropped at some point -- without it, a long run OOMs
        # partway through rather than at the first iteration, since it's
        # genuine cross-iteration accumulation, not a single-run spike).
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        _log(f"  GRN run {i + 1}/{n_runs} done")

    # ---- Phase 2: Adjacency mean/CV diagnostics (always computed) ----
    _log("Computing adjacency mean/CV diagnostics...")
    A_stack = np.stack(adj_matrices, axis=0)   # (k, n_genes, n_genes)
    A_mean = A_stack.mean(axis=0)              # (n_genes, n_genes)
    A_std = A_stack.std(axis=0)
    # CV = std / |mean|, so repressive (negative-mean) edges are handled
    # symmetrically with activating ones. 0 where mean == 0 to avoid division
    # by zero.
    A_abs_mean = np.abs(A_mean)
    A_cv = np.divide(A_std, A_abs_mean, out=np.zeros_like(A_mean), where=A_abs_mean > 0)

    shared_kwargs = dict(
        datasource=datasource, version=version, cache_dir=cache_dir,
        tf_list_path=tf_list_path, ranking_db_paths=ranking_db_paths,
        motif_annotation_path=motif_annotation_path,
        grn_n_steps=grn_n_steps, grn_sparsity_threshold=grn_sparsity_threshold,
        module_k=module_k, module_percentile_thresholds=module_percentile_thresholds,
        module_top_n_per_target=module_top_n_per_target,
        module_min_targets=module_min_targets, module_min_fraction=module_min_fraction,
        module_include_tf=module_include_tf,
        pruning_rank_threshold=pruning_rank_threshold,
        pruning_auc_threshold=pruning_auc_threshold,
        pruning_nes_threshold=pruning_nes_threshold,
        pruning_merge_strategy=pruning_merge_strategy,
        annotation_motif_similarity_fdr=annotation_motif_similarity_fdr,
        annotation_orthologous_identity=annotation_orthologous_identity,
        aucell_k=aucell_k, aucell_auc_threshold=aucell_auc_threshold,
        aucell_batch_size=aucell_batch_size,
        device=device, seed=seed, verbose=verbose,
    )

    n_filtered = 0
    tf_frequency_scores = None

    if aggregation_method == "mean_adjacency":
        # ---- Phase 3 (mean_adjacency): run Steps 2-5 once on the consensus ----
        if cv_threshold is not None:
            A_consensus = np.where(A_cv < cv_threshold, A_mean, 0.0).astype(np.float32)
            n_filtered = int(np.sum((A_abs_mean > 0) & (A_cv >= cv_threshold)))
            _log(f"  CV filter (threshold={cv_threshold}) removed {n_filtered} edges")
        else:
            A_consensus = A_mean.astype(np.float32)

        effective_min_genes = max(pruning_min_genes, min_genes_filter)
        _log("Running downstream pipeline on consensus adjacency...")
        result = run_flashscenic(
            exp_matrix, gene_names, species,
            adj_matrix=A_consensus,
            pruning_min_genes=effective_min_genes,
            **shared_kwargs,
        )
    else:
        # ---- Phase 3 (frequency): run Steps 2-5 k times, vote, AUCell once ----
        # Reuses Phase 1's adjacency draws (paired design — same n_runs GRN
        # samples as the mean_adjacency path would use), so no extra
        # RegDiffusion runs are required. Each run's own AUCell pass is
        # discarded; only its regulons are kept for frequency voting.
        _log(f"Running full pipeline {n_runs} times for frequency aggregation...")
        full_results = []
        for i in range(n_runs):
            full_results.append(run_flashscenic(
                exp_matrix, gene_names, species,
                adj_matrix=adj_matrices[i],
                pruning_min_genes=pruning_min_genes,
                **shared_kwargs,
            ))
            gc.collect()
            torch.cuda.empty_cache()
            _log(f"  Full pipeline run {i + 1}/{n_runs} done "
                 f"({len(full_results[-1]['regulons'])} regulons)")

        _log(f"Aggregating regulons by frequency "
             f"(threshold={frequency_threshold}, min_genes={min_genes_filter})...")
        consensus_regulons, regulon_adj, tf_frequency_scores = _build_frequency_consensus(
            full_results, gene_names, frequency_threshold, min_genes_filter,
        )
        if not consensus_regulons:
            raise ValueError(
                "No regulons survived frequency aggregation. Consider "
                "lowering frequency_threshold or min_genes_filter."
            )
        _log(f"  {len(consensus_regulons)} consensus regulons")

        from .aucell import get_aucell
        resolved_aucell_k = aucell_k if aucell_k is not None else module_k
        _log("Computing AUCell scores on consensus regulons...")
        auc_scores = get_aucell(
            exp_float32, regulon_adj,
            k=resolved_aucell_k,
            auc_threshold=aucell_auc_threshold,
            device=device, batch_size=aucell_batch_size, seed=seed,
        )

        result = {
            "auc_scores": auc_scores,
            "regulon_names": [reg["name"] for reg in consensus_regulons],
            "regulons": consensus_regulons,
            "regulon_adj": regulon_adj,
            "parameters": {
                "species": species,
                "datasource": datasource,
                "version": version,
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
                "aucell_k": resolved_aucell_k,
                "aucell_auc_threshold": aucell_auc_threshold,
                "aucell_batch_size": aucell_batch_size,
                "device": device,
                "seed": seed,
                "n_cells": exp_matrix.shape[0],
                "n_genes": n_genes,
                "n_modules": None,
                "n_regulons": len(consensus_regulons),
            },
        }

    # ---- Phase 4: Attach multi-run diagnostics ----
    result['adj_mean'] = A_mean
    result['adj_cv'] = A_cv
    result['adj_matrices'] = adj_matrices if return_adj_matrices else None
    result['n_runs'] = n_runs
    result['aggregation_method'] = aggregation_method
    result['frequency_threshold'] = frequency_threshold
    result['min_genes_filter'] = min_genes_filter
    result['cv_threshold'] = cv_threshold
    result['n_edges_cv_filtered'] = n_filtered
    result['tf_frequency_scores'] = tf_frequency_scores
    result['parameters'].update({
        'n_runs': n_runs,
        'aggregation_method': aggregation_method,
        'frequency_threshold': frequency_threshold,
        'min_genes_filter': min_genes_filter,
        'cv_threshold': cv_threshold,
        'seeds': seeds,
    })

    return result
