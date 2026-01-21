"""
GPU-accelerated cisTarget pruning implementation for flashscenic.

This module implements the core cisTarget algorithm used for regulatory 
hypothesis pruning based on motif enrichment analysis.

The algorithm:
1. For each TF module (set of co-expressed genes):
   - Calculate AUC for each motif based on gene rankings
   - Compute Normalized Enrichment Score (NES)
   - Filter enriched motifs (NES >= threshold)
   - Identify leading edge genes (target genes)

Reference: pySCENIC (https://github.com/aertslab/pySCENIC)
"""

import torch
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass


@dataclass
class PruningResult:
    """Container for cisTarget pruning results."""
    tf_name: str
    enriched_motifs: List[str]
    nes_scores: np.ndarray
    auc_scores: np.ndarray
    target_genes: Dict[str, List[Tuple[str, float]]]  # motif -> [(gene, weight), ...]
    rank_at_max: Dict[str, int]  # motif -> rank at max recovery


def compute_recovery_aucs_gpu(
    rankings: torch.Tensor,
    module_gene_mask: torch.Tensor,
    rank_threshold: int,
    auc_threshold: float,
    weights: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute recovery curves and AUCs for all motifs given module genes.
    
    This implements the same algorithm as ctxcore.recovery.recovery() but on GPU.
    
    Args:
        rankings: (n_motifs, n_genes) - rank of each gene for each motif (0-indexed)
        module_gene_mask: (n_genes,) - boolean mask for genes in the module
        rank_threshold: Maximum rank to consider for recovery curve
        auc_threshold: Fraction of genome for AUC calculation
        weights: (n_module_genes,) - optional weights for weighted recovery
    
    Returns:
        rccs: (n_motifs, rank_threshold) - recovery curves
        aucs: (n_motifs,) - AUC values
    """
    device = rankings.device
    n_motifs, total_genes = rankings.shape
    
    # Get rankings for module genes only
    module_rankings = rankings[:, module_gene_mask]  # (n_motifs, n_module_genes)
    n_module_genes = module_rankings.shape[1]
    
    if weights is None:
        weights = torch.ones(n_module_genes, device=device, dtype=torch.float32)
    
    # Derive rank cutoff (same as pySCENIC)
    rank_cutoff = int(round(auc_threshold * total_genes)) - 1
    rank_cutoff = max(1, min(rank_cutoff, rank_threshold - 1))
    
    # Compute recovery curves using bincount approach
    # For each motif, for each rank position r, count cumulative weighted hits
    rccs = torch.zeros((n_motifs, rank_threshold), device=device, dtype=torch.float32)
    
    # Vectorized implementation: for each motif, sort rankings and compute cumsum
    # This is more memory efficient than full bincount for large matrices
    
    # Clamp rankings to rank_threshold (genes with higher rank don't contribute)
    clamped_rankings = module_rankings.clamp(max=total_genes - 1)
    
    # For each motif, compute recovery curve
    # Recovery at position r = sum of weights for genes with rank < r
    
    # Efficient batch computation using scatter_add
    # Create position indices for scatter
    batch_indices = torch.arange(n_motifs, device=device).unsqueeze(1).expand(-1, n_module_genes)
    
    # Compute recovery curves by accumulating weights at each rank position
    for motif_idx in range(n_motifs):
        ranks_for_motif = clamped_rankings[motif_idx]  # (n_module_genes,)
        # Bincount: count weighted occurrences at each rank
        bincount = torch.zeros(rank_threshold, device=device, dtype=torch.float32)
        valid_mask = ranks_for_motif < rank_threshold
        valid_ranks = ranks_for_motif[valid_mask].long()
        valid_weights = weights[valid_mask]
        
        if len(valid_ranks) > 0:
            bincount.scatter_add_(0, valid_ranks, valid_weights)
        
        # Cumsum to get recovery curve
        rccs[motif_idx] = bincount.cumsum(dim=0)
    
    # Compute AUC
    # max_auc = (rank_cutoff + 1) * sum(weights)
    weight_sum = weights.sum()
    max_auc = float((rank_cutoff + 1) * weight_sum.item())
    
    # AUC = sum of recovery curve up to rank_cutoff / max_auc
    aucs = rccs[:, :rank_cutoff].sum(dim=1) / max_auc
    
    return rccs, aucs


def compute_recovery_aucs_gpu_batch(
    rankings: torch.Tensor,
    module_gene_indices: torch.Tensor,
    rank_threshold: int,
    auc_threshold: float,
    weights: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute recovery curves and AUCs using a more memory-efficient batch approach.
    
    This version processes motifs in batches to avoid memory issues with large databases.
    
    Args:
        rankings: (n_motifs, n_genes) - rank of each gene for each motif
        module_gene_indices: (n_module_genes,) - indices of genes in the module
        rank_threshold: Maximum rank for recovery curve
        auc_threshold: Fraction of genome for AUC
        weights: Optional gene weights
    
    Returns:
        rccs: (n_motifs, rank_threshold) - recovery curves
        aucs: (n_motifs,) - AUC values
    """
    device = rankings.device
    n_motifs, total_genes = rankings.shape
    n_module_genes = len(module_gene_indices)
    
    if weights is None:
        weights = torch.ones(n_module_genes, device=device, dtype=torch.float32)
    
    # Derive rank cutoff
    rank_cutoff = int(round(auc_threshold * total_genes)) - 1
    rank_cutoff = max(1, min(rank_cutoff, rank_threshold - 1))
    
    # Get rankings for module genes
    module_rankings = rankings[:, module_gene_indices]  # (n_motifs, n_module_genes)
    
    # Compute AUC using the weighted_auc approach (more memory efficient)
    # Filter rankings < rank_cutoff, sort, and compute area
    
    max_auc = float((rank_cutoff + 1) * weights.sum().item())
    aucs = torch.zeros(n_motifs, device=device, dtype=torch.float32)
    rccs = torch.zeros((n_motifs, rank_threshold), device=device, dtype=torch.float32)
    
    for motif_idx in range(n_motifs):
        ranks = module_rankings[motif_idx]  # (n_module_genes,)
        
        # Filter ranks within cutoff
        mask = ranks < rank_cutoff
        valid_ranks = ranks[mask]
        valid_weights = weights[mask]
        
        if len(valid_ranks) == 0:
            continue
        
        # Sort by rank
        sort_indices = torch.argsort(valid_ranks)
        sorted_ranks = valid_ranks[sort_indices]
        sorted_weights = valid_weights[sort_indices]
        
        # Cumulative weights
        cumsum_weights = sorted_weights.cumsum(dim=0)
        
        # Compute AUC using trapezoidal rule
        # Add rank_cutoff as endpoint
        ranks_with_cutoff = torch.cat([sorted_ranks, torch.tensor([rank_cutoff], device=device, dtype=sorted_ranks.dtype)])
        rank_diffs = ranks_with_cutoff[1:] - ranks_with_cutoff[:-1]
        auc = (rank_diffs.float() * cumsum_weights).sum()
        aucs[motif_idx] = auc / max_auc
        
        # Build recovery curve using bincount
        bincount = torch.zeros(rank_threshold, device=device, dtype=torch.float32)
        valid_all_ranks = ranks[ranks < rank_threshold].long()
        valid_all_weights = weights[ranks < rank_threshold]
        if len(valid_all_ranks) > 0:
            bincount.scatter_add_(0, valid_all_ranks, valid_all_weights)
        rccs[motif_idx] = bincount.cumsum(dim=0)
    
    return rccs, aucs


def compute_nes(aucs: torch.Tensor) -> torch.Tensor:
    """
    Compute Normalized Enrichment Scores (NES) from AUC values.
    
    NES = (AUC - mean(AUC)) / std(AUC)
    
    Uses population std (ddof=0) to match numpy/pySCENIC behavior.
    
    Args:
        aucs: (n_motifs,) - AUC values
    
    Returns:
        nes: (n_motifs,) - NES values
    """
    mean_auc = aucs.mean()
    # Use unbiased=False for population std (ddof=0), matching numpy default
    std_auc = aucs.std(unbiased=False)
    
    if std_auc > 0:
        nes = (aucs - mean_auc) / std_auc
    else:
        nes = torch.zeros_like(aucs)
    
    return nes


def compute_leading_edge(
    rcc: torch.Tensor,
    avg2std_rcc: torch.Tensor,
    rankings: torch.Tensor,
    gene_indices: torch.Tensor,
    weights: torch.Tensor
) -> Tuple[List[Tuple[int, float]], int]:
    """
    Compute the leading edge genes for an enriched motif.
    
    The leading edge is defined as genes with rank <= rank_at_max,
    where rank_at_max is the position where (rcc - avg2std_rcc) is maximum.
    
    Args:
        rcc: (rank_threshold,) - recovery curve for this motif
        avg2std_rcc: (rank_threshold,) - average + 2*std recovery curve
        rankings: (n_module_genes,) - rankings for module genes under this motif
        gene_indices: (n_module_genes,) - original gene indices
        weights: (n_module_genes,) - gene weights/importances
    
    Returns:
        leading_edge: List of (gene_index, weight) tuples
        rank_at_max: The rank at maximum difference
    """
    # Find rank at maximum difference
    diff = rcc - avg2std_rcc
    rank_at_max = int(diff.argmax().item())
    
    # Get genes with rank <= rank_at_max
    mask = rankings <= rank_at_max
    
    leading_gene_indices = gene_indices[mask]
    leading_weights = weights[mask]
    
    # Sort by rank
    leading_ranks = rankings[mask]
    sort_order = torch.argsort(leading_ranks)
    
    leading_edge = [
        (int(leading_gene_indices[idx].item()), float(leading_weights[idx].item()))
        for idx in sort_order
    ]
    
    return leading_edge, rank_at_max


def prune_module_gpu(
    ranking_matrix: np.ndarray,
    module_gene_indices: np.ndarray,
    motif_names: List[str],
    gene_names: List[str],
    rank_threshold: int = 1500,
    auc_threshold: float = 0.05,
    nes_threshold: float = 3.0,
    weights: Optional[np.ndarray] = None,
    device: str = 'cuda'
) -> Dict:
    """
    Perform cisTarget pruning for a single module on GPU.
    
    Args:
        ranking_matrix: (n_motifs, n_genes) - ranking database as numpy array
        module_gene_indices: Indices of genes in this module
        motif_names: List of motif names
        gene_names: List of all gene names
        rank_threshold: Maximum rank for recovery curve
        auc_threshold: Fraction of genome for AUC
        nes_threshold: NES threshold for enrichment
        weights: Optional gene weights
        device: 'cuda' or 'cpu'
    
    Returns:
        Dictionary with enrichment results
    """
    n_motifs, n_genes = ranking_matrix.shape
    n_module_genes = len(module_gene_indices)
    
    # Transfer to GPU
    rankings_tensor = torch.tensor(ranking_matrix, device=device, dtype=torch.int32)
    gene_idx_tensor = torch.tensor(module_gene_indices, device=device, dtype=torch.long)
    
    if weights is not None:
        weights_tensor = torch.tensor(weights, device=device, dtype=torch.float32)
    else:
        weights_tensor = torch.ones(n_module_genes, device=device, dtype=torch.float32)
    
    # Compute AUCs and recovery curves
    rccs, aucs = compute_recovery_aucs_gpu_batch(
        rankings_tensor, gene_idx_tensor, rank_threshold, auc_threshold, weights_tensor
    )
    
    # Compute NES
    nes = compute_nes(aucs)
    
    # Filter enriched motifs
    enriched_mask = nes >= nes_threshold
    enriched_indices = torch.where(enriched_mask)[0]
    
    if len(enriched_indices) == 0:
        return {
            'enriched_motifs': [],
            'nes_scores': np.array([]),
            'auc_scores': np.array([]),
            'target_genes': {},
            'rank_at_max': {}
        }
    
    # Compute average + 2*std recovery curve
    avg_rcc = rccs.mean(dim=0)
    std_rcc = rccs.std(dim=0)
    avg2std_rcc = avg_rcc + 2.0 * std_rcc
    
    # Get module gene rankings for enriched motifs
    module_rankings = rankings_tensor[:, gene_idx_tensor]
    
    # Compute leading edge for enriched motifs
    target_genes = {}
    rank_at_max_dict = {}
    
    enriched_motif_names = [motif_names[i] for i in enriched_indices.cpu().numpy()]
    module_gene_names = [gene_names[i] for i in module_gene_indices]
    
    for idx, motif_idx in enumerate(enriched_indices):
        motif_name = motif_names[int(motif_idx.item())]
        
        leading_edge, rank_at_max = compute_leading_edge(
            rccs[motif_idx],
            avg2std_rcc,
            module_rankings[motif_idx],
            gene_idx_tensor,
            weights_tensor
        )
        
        # Convert gene indices to names with weights
        target_genes[motif_name] = [
            (gene_names[gene_idx], weight) for gene_idx, weight in leading_edge
        ]
        rank_at_max_dict[motif_name] = rank_at_max
    
    return {
        'enriched_motifs': enriched_motif_names,
        'nes_scores': nes[enriched_mask].cpu().numpy(),
        'auc_scores': aucs[enriched_mask].cpu().numpy(),
        'target_genes': target_genes,
        'rank_at_max': rank_at_max_dict
    }


def prune_modules_batch_gpu(
    ranking_matrix: np.ndarray,
    modules: List[Dict],  # List of {'name': str, 'genes': List[str], 'tf': str}
    motif_names: List[str],
    gene_names: List[str],
    rank_threshold: int = 1500,
    auc_threshold: float = 0.05,
    nes_threshold: float = 3.0,
    device: str = 'cuda',
    batch_size: int = 100
) -> List[Dict]:
    """
    Perform cisTarget pruning for multiple modules in batch.
    
    Args:
        ranking_matrix: (n_motifs, n_genes) - full ranking database
        modules: List of module dictionaries with 'name', 'genes', 'tf' keys
        motif_names: List of motif names
        gene_names: List of gene names
        rank_threshold: Maximum rank for recovery curve
        auc_threshold: Fraction of genome for AUC
        nes_threshold: NES threshold
        device: 'cuda' or 'cpu'
        batch_size: Number of modules to process at once
    
    Returns:
        List of pruning results for each module
    """
    # Create gene name to index mapping
    gene_name_to_idx = {name: idx for idx, name in enumerate(gene_names)}
    gene_set = set(gene_names)
    
    # Transfer ranking matrix to GPU once
    rankings_tensor = torch.tensor(ranking_matrix, device=device, dtype=torch.int32)
    
    results = []
    
    for module in modules:
        # Get gene indices for this module (filter out genes not in database)
        module_genes = [g for g in module['genes'] if g in gene_set]
        
        if len(module_genes) < 0.8 * len(module['genes']):
            # Skip if less than 80% of genes are mapped (same as pySCENIC)
            results.append({
                'module_name': module['name'],
                'tf': module.get('tf', ''),
                'skipped': True,
                'reason': 'Less than 80% of genes mapped'
            })
            continue
        
        module_gene_indices = np.array([gene_name_to_idx[g] for g in module_genes])
        
        # Compute weights if available
        weights = None
        if 'weights' in module:
            weights = np.array([module['weights'].get(g, 1.0) for g in module_genes])
        
        # Run pruning on GPU
        prune_result = prune_module_gpu(
            ranking_matrix,  # Re-use the numpy array (rankings_tensor is for reference)
            module_gene_indices,
            motif_names,
            gene_names,
            rank_threshold,
            auc_threshold,
            nes_threshold,
            weights,
            device
        )
        
        prune_result['module_name'] = module['name']
        prune_result['tf'] = module.get('tf', '')
        prune_result['skipped'] = False
        
        results.append(prune_result)
    
    return results


class GPUCisTargetPruner:
    """
    GPU-accelerated cisTarget pruning class.
    
    This class provides a high-level interface for performing cisTarget pruning
    on GPU, compatible with the pySCENIC workflow.
    
    Example usage:
        ```python
        pruner = GPUCisTargetPruner(
            ranking_db_path='path/to/ranking.feather',
            motif_annotations_path='path/to/annotations.tbl'
        )
        
        # Load database to GPU
        pruner.load_database()
        
        # Prune modules
        results = pruner.prune(modules)
        
        # Convert to regulons
        regulons = pruner.to_regulons(results)
        ```
    """
    
    def __init__(
        self,
        ranking_db_path: Optional[str] = None,
        motif_annotations_path: Optional[str] = None,
        rank_threshold: int = 1500,
        auc_threshold: float = 0.05,
        nes_threshold: float = 3.0,
        device: str = 'cuda'
    ):
        """
        Initialize the GPU cisTarget pruner.
        
        Args:
            ranking_db_path: Path to the cisTarget ranking database (.feather)
            motif_annotations_path: Path to motif annotations file
            rank_threshold: Maximum rank for recovery curve calculation
            auc_threshold: Fraction of genome for AUC calculation
            nes_threshold: NES threshold for enrichment filtering
            device: 'cuda' or 'cpu'
        """
        self.ranking_db_path = ranking_db_path
        self.motif_annotations_path = motif_annotations_path
        self.rank_threshold = rank_threshold
        self.auc_threshold = auc_threshold
        self.nes_threshold = nes_threshold
        self.device = device
        
        self.ranking_matrix = None
        self.motif_names = None
        self.gene_names = None
        self.motif_annotations = None
        self._rankings_tensor = None
    
    def load_database(self):
        """Load the ranking database and annotations into memory."""
        if self.ranking_db_path is None:
            raise ValueError("ranking_db_path must be specified")
        
        # Load ranking database using ctxcore
        try:
            from ctxcore.rnkdb import opendb
            db = opendb(self.ranking_db_path, name='db')
            df = db.load_full()
            
            self.ranking_matrix = df.values
            self.motif_names = list(df.index)
            self.gene_names = list(df.columns)
            
        except ImportError:
            # Fallback: direct feather loading
            import pyarrow.feather as pf
            df = pf.read_table(self.ranking_db_path).to_pandas()
            
            # Assume first column is motif names
            index_col = None
            for col in ['motifs', 'tracks', 'regions', 'genes']:
                if col in df.columns:
                    index_col = col
                    break
            
            if index_col:
                df = df.set_index(index_col)
            
            self.ranking_matrix = df.values
            self.motif_names = list(df.index)
            self.gene_names = list(df.columns)
        
        # Load motif annotations if provided
        if self.motif_annotations_path:
            self.motif_annotations = pd.read_csv(
                self.motif_annotations_path, 
                sep='\t',
                index_col=0
            )
        
        # Pre-transfer to GPU
        self._rankings_tensor = torch.tensor(
            self.ranking_matrix, 
            device=self.device, 
            dtype=torch.int32
        )
        
        print(f"Loaded ranking database: {len(self.motif_names)} motifs x {len(self.gene_names)} genes")
    
    def load_from_arrays(
        self,
        ranking_matrix: np.ndarray,
        motif_names: List[str],
        gene_names: List[str],
        motif_annotations: Optional[pd.DataFrame] = None
    ):
        """
        Load database from numpy arrays directly.
        
        Args:
            ranking_matrix: (n_motifs, n_genes) ranking matrix
            motif_names: List of motif names
            gene_names: List of gene names  
            motif_annotations: Optional motif annotations dataframe
        """
        self.ranking_matrix = ranking_matrix
        self.motif_names = motif_names
        self.gene_names = gene_names
        self.motif_annotations = motif_annotations
        
        self._rankings_tensor = torch.tensor(
            self.ranking_matrix,
            device=self.device,
            dtype=torch.int32
        )
    
    def prune(self, modules: List[Dict]) -> List[Dict]:
        """
        Prune modules using cisTarget algorithm on GPU.
        
        Args:
            modules: List of module dictionaries with keys:
                - 'name': Module name
                - 'genes': List of gene names in the module
                - 'tf': (Optional) Transcription factor name
                - 'weights': (Optional) Dict mapping gene names to weights
        
        Returns:
            List of pruning result dictionaries
        """
        if self.ranking_matrix is None:
            raise ValueError("Database not loaded. Call load_database() first.")
        
        return prune_modules_batch_gpu(
            self.ranking_matrix,
            modules,
            self.motif_names,
            self.gene_names,
            self.rank_threshold,
            self.auc_threshold,
            self.nes_threshold,
            self.device
        )
    
    def annotate_results(self, results: List[Dict]) -> List[Dict]:
        """
        Add motif annotations to pruning results.
        
        This matches enriched motifs with TF annotations from the annotation file.
        
        Args:
            results: Pruning results from prune()
        
        Returns:
            Results with added 'annotations' field
        """
        if self.motif_annotations is None:
            return results
        
        for result in results:
            if result.get('skipped', False):
                continue
            
            annotations = {}
            for motif in result.get('enriched_motifs', []):
                if motif in self.motif_annotations.index:
                    annotations[motif] = self.motif_annotations.loc[motif].to_dict()
            
            result['annotations'] = annotations
        
        return results
    
    def clear_gpu_memory(self):
        """Release GPU memory."""
        if self._rankings_tensor is not None:
            del self._rankings_tensor
            self._rankings_tensor = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
