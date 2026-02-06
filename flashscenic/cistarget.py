"""
GPU-accelerated cisTarget pruning for flashscenic.

All operations are tensor-native for GPU acceleration.
"""

import torch
import numpy as np
import csv
from typing import Tuple, List, Dict, Optional, Union
from collections import defaultdict

ArrayLike = Union[np.ndarray, torch.Tensor]

def compute_recovery_aucs(
    rankings: torch.Tensor,
    module_gene_indices: torch.Tensor,
    rank_threshold: int,
    auc_threshold: float,
    weights: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute recovery curves and AUCs for all motifs given module genes.

    Vectorized implementation - processes all motifs in parallel.

    Args:
        rankings: (n_motifs, n_genes) - rank of each gene for each motif (0-indexed)
        module_gene_indices: (n_module_genes,) - indices of genes in the module
        rank_threshold: Maximum rank to consider for recovery curve
        auc_threshold: Fraction of genome for AUC calculation
        weights: (n_module_genes,) - optional weights for weighted recovery

    Returns:
        rccs: (n_motifs, rank_threshold) - recovery curves
        aucs: (n_motifs,) - AUC values
    """
    device = rankings.device
    n_motifs, total_genes = rankings.shape
    n_module_genes = len(module_gene_indices)

    if weights is None:
        weights = torch.ones(n_module_genes, device=device, dtype=torch.float32)

    # Derive rank cutoff (same as pySCENIC)
    rank_cutoff = int(round(auc_threshold * total_genes)) - 1
    rank_cutoff = max(1, min(rank_cutoff, rank_threshold - 1))

    # Get rankings for module genes: (n_motifs, n_module_genes)
    module_rankings = rankings[:, module_gene_indices]

    # Compute recovery curves using scatter
    # For each motif, accumulate weights at each rank position
    rccs = torch.zeros((n_motifs, rank_threshold), device=device, dtype=torch.float32)

    # Clamp to valid range
    valid_mask = module_rankings < rank_threshold  # (n_motifs, n_module_genes)

    # Process in vectorized manner using scatter_add
    # Create batch indices for scatter
    batch_idx = torch.arange(n_motifs, device=device).unsqueeze(1).expand(-1, n_module_genes)

    # Flatten for scatter
    flat_batch = batch_idx[valid_mask]  # (n_valid,)
    flat_ranks = module_rankings[valid_mask].long()  # (n_valid,)
    flat_weights = weights.unsqueeze(0).expand(n_motifs, -1)[valid_mask]  # (n_valid,)

    # Create index for 2D scatter: batch_idx * rank_threshold + rank
    flat_idx = flat_batch * rank_threshold + flat_ranks

    # Scatter add into flattened rccs
    rccs_flat = rccs.view(-1)
    rccs_flat.scatter_add_(0, flat_idx, flat_weights)
    rccs = rccs_flat.view(n_motifs, rank_threshold)

    # Cumsum to get recovery curves
    rccs = rccs.cumsum(dim=1)

    # Compute AUC
    weight_sum = weights.sum()
    max_auc = float((rank_cutoff + 1) * weight_sum.item())
    aucs = rccs[:, :rank_cutoff].sum(dim=1) / max_auc

    return rccs, aucs


def compute_nes(aucs: torch.Tensor) -> torch.Tensor:
    """
    Compute Normalized Enrichment Scores (NES) from AUC values.

    NES = (AUC - mean(AUC)) / std(AUC)
    Uses population std (ddof=0) to match pySCENIC.
    """
    mean_auc = aucs.mean()
    std_auc = aucs.std(unbiased=False)

    if std_auc > 0:
        return (aucs - mean_auc) / std_auc
    return torch.zeros_like(aucs)


class MotifAnnotation:
    """
    Lightweight motif annotation storage without pandas.
    
    Stores motif annotations in a dictionary for fast lookup.
    Matches pyscenic's annotation filtering behavior.
    """
    
    def __init__(self):
        # Core data structure: dict[(tf, motif_id)] -> annotation_info
        self.annotations: Dict[Tuple[str, str], Dict] = {}
        # All motif IDs (for fast lookup)
        self.all_motif_ids: set = set()
        # All TF names (for reference)
        self.all_tf_names: set = set()
    
    @classmethod
    def load_from_file(
        cls,
        fname: str,
        motif_similarity_fdr: float = 0.001,
        orthologous_identity_threshold: float = 0.0,
        column_names: Optional[Tuple[str, ...]] = None
    ) -> 'MotifAnnotation':
        """
        Load motif annotations from a motif2TF snapshot file.
        
        Args:
            fname: Path to TSV annotation file
            motif_similarity_fdr: Maximum FDR threshold (default: 0.001)
            orthologous_identity_threshold: Minimum orthologous identity (default: 0.0)
            column_names: Optional tuple of column names to use. If None, reads from header.
        
        Returns:
            MotifAnnotation instance
        """
        instance = cls()
        
        # Read file and collect all annotations
        annotations_list = []
        
        with open(fname, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            # Read header
            header = next(reader, None)
            if header is None:
                raise ValueError("Empty annotation file")
            
            # Clean header: strip whitespace (keep # prefix for motif_id)
            header_clean = [col.strip() for col in header]
            
            # Find column indices
            try:
                # motif_id may have # prefix - check original header first
                if "#motif_id" in header_clean:
                    motif_id_idx = header_clean.index("#motif_id")
                elif "motif_id" in header_clean:
                    motif_id_idx = header_clean.index("motif_id")
                else:
                    raise ValueError("motif_id column not found")
                
                # Other columns don't have # prefix
                gene_name_idx = header_clean.index("gene_name")
                similarity_qval_idx = header_clean.index("motif_similarity_qvalue")
                ortho_identity_idx = header_clean.index("orthologous_identity")
                description_idx = header_clean.index("description")
            except ValueError as e:
                raise ValueError(f"Required column not found in header: {e}. "
                               f"Header: {header_clean}")
            
            print(f"Reading annotation file: {fname}")
            print(f"Column indices: motif_id={motif_id_idx}, gene_name={gene_name_idx}, "
                  f"similarity_qvalue={similarity_qval_idx}, ortho_identity={ortho_identity_idx}, "
                  f"description={description_idx}")
            
            row_count = 0
            for row in reader:
                row_count += 1
                if len(row) <= max(motif_id_idx, gene_name_idx, similarity_qval_idx, 
                                  ortho_identity_idx, description_idx):
                    continue
                
                try:
                    motif_id = row[motif_id_idx].strip()
                    gene_name = row[gene_name_idx].strip()
                    
                    # Skip empty rows
                    if not motif_id or not gene_name:
                        continue
                    
                    # Parse similarity_qvalue (may be 0 or scientific notation)
                    similarity_qval_str = row[similarity_qval_idx].strip()
                    if not similarity_qval_str or similarity_qval_str.lower() == 'none':
                        similarity_qval = 0.0  # Treat 0 as direct annotation (best case)
                    else:
                        similarity_qval = float(similarity_qval_str)
                    
                    # Parse orthologous_identity (may be empty or 0)
                    ortho_identity_str = row[ortho_identity_idx].strip()
                    if not ortho_identity_str or ortho_identity_str.lower() == 'none':
                        ortho_identity = float('nan')
                    else:
                        ortho_identity = float(ortho_identity_str)
                    
                    description = row[description_idx].strip() if description_idx < len(row) else ""
                    
                    # Apply filters
                    if similarity_qval > motif_similarity_fdr:
                        continue
                    if not np.isnan(ortho_identity) and ortho_identity < orthologous_identity_threshold:
                        continue
                    
                    annotations_list.append({
                        'key': (gene_name, motif_id),
                        'motif_similarity_qvalue': similarity_qval,
                        'orthologous_identity': ortho_identity,
                        'annotation': description,
                    })
                    
                except (ValueError, IndexError) as e:
                    # Skip malformed rows
                    if row_count <= 5:  # Only warn for first few rows
                        print(f"Warning: Skipping malformed row {row_count}: {e}")
                    continue
            
            print(f"Read {row_count} rows, {len(annotations_list)} passed filters")
        
        # Sort and deduplicate: keep best annotation for each (TF, motif_id)
        # Match pyscenic: sort_values([COLUMN_NAME_MOTIF_SIMILARITY_QVALUE, COLUMN_NAME_ORTHOLOGOUS_IDENTITY],
        #                              ascending=[False, True]) then keep last (best)
        # ascending=[False, True] means:
        #   - qvalue DESC: higher qvalue first, lower qvalue last (we want lower, so last is better)
        #   - ortho ASC: lower ortho first, higher ortho last (we want higher, so last is better)
        # Then keep="last" means keep the last one after sorting, which is the best
        annotations_list.sort(
            key=lambda x: (
                -x['motif_similarity_qvalue'],  # DESC (negative for descending)
                x['orthologous_identity'] if not np.isnan(x['orthologous_identity']) else float('inf')  # ASC
            )
        )
        
        # Deduplicate: keep last (best) for each (TF, motif_id) pair
        # pyscenic uses: ~annotated_features.index.duplicated(keep="last")
        # After sorting DESC/ASC, the last entry for each key is the best
        seen_keys = set()
        for ann in reversed(annotations_list):  # Process in reverse to keep first best (which is last after sort)
            key = ann['key']
            if key not in seen_keys:
                seen_keys.add(key)
                instance.annotations[key] = {
                    'motif_similarity_qvalue': ann['motif_similarity_qvalue'],
                    'orthologous_identity': ann['orthologous_identity'],
                    'annotation': ann['annotation'],
                }
                instance.all_motif_ids.add(key[1])  # motif_id
                instance.all_tf_names.add(key[0])   # TF name
        
        print(f"Loaded {len(instance.annotations)} motif annotations "
              f"({len(instance.all_motif_ids)} unique motifs, "
              f"{len(instance.all_tf_names)} unique TFs)")
        
        return instance
    
    def has_annotation(self, motif_id: str, tf_name: Optional[str] = None) -> bool:
        """
        Check if a motif has annotation.
        
        Args:
            motif_id: Motif ID
            tf_name: Optional TF name (if provided, checks (TF, motif) pair)
        
        Returns:
            True if annotation exists
        """
        if tf_name is not None:
            return (tf_name, motif_id) in self.annotations
        return motif_id in self.all_motif_ids
    
    def get_annotation(
        self,
        motif_id: str,
        tf_name: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Get annotation for a motif.
        
        Args:
            motif_id: Motif ID
            tf_name: Optional TF name
        
        Returns:
            Annotation dict or None
        """
        if tf_name is not None:
            return self.annotations.get((tf_name, motif_id))
        # If no TF specified, return first match (or None)
        for key, ann in self.annotations.items():
            if key[1] == motif_id:
                return ann
        return None


def filter_by_annotations(
    result: Dict[str, torch.Tensor],
    motif_names: List[str],
    motif_annotations: Optional[MotifAnnotation],
    filter_for_annotation: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Filter enriched motifs by annotations (CPU implementation).
    
    Matches pyscenic behavior: filters enriched motifs to keep only those
    with annotations when filter_for_annotation=True.
    
    Args:
        result: Pruning result dict with 'enriched_mask', 'nes', 'aucs', etc.
        motif_names: List of motif names (from database)
        motif_annotations: MotifAnnotation object (None = no filtering)
        filter_for_annotation: If True, only keep motifs with annotations
    
    Returns:
        Filtered result dict (all tensors remain on original device)
    """
    if motif_annotations is None or not filter_for_annotation:
        return result
    
    enriched_indices = torch.where(result['enriched_mask'])[0].cpu().numpy()
    
    if len(enriched_indices) == 0:
        return result
    
    # Create mask for motifs with annotations (CPU)
    device = result['enriched_mask'].device
    has_annotation_mask = torch.zeros(len(motif_names), dtype=torch.bool)
    
    for idx in enriched_indices:
        motif_id = motif_names[idx]
        if motif_annotations.has_annotation(motif_id):
            has_annotation_mask[idx] = True
    
    # Apply filter (keep on same device as result)
    has_annotation_mask = has_annotation_mask.to(device)
    new_enriched_mask = result['enriched_mask'] & has_annotation_mask
    
    # If no enriched motifs remain, return early
    n_enriched = new_enriched_mask.sum().item()
    if n_enriched == 0:
        result['enriched_mask'] = new_enriched_mask
        # Update leading_edge_masks and rank_at_max to empty
        n_module_genes = 0
        if 'leading_edge_masks' in result and len(result['leading_edge_masks']) > 0:
            n_module_genes = result['leading_edge_masks'].shape[1]
        result['leading_edge_masks'] = torch.zeros((0, n_module_genes), dtype=torch.bool, device=device)
        result['rank_at_max'] = torch.zeros(0, dtype=torch.long, device=device)
        return result
    
    # Update enriched_mask
    result['enriched_mask'] = new_enriched_mask
    
    # Update leading_edge_masks and rank_at_max to only include filtered motifs
    old_enriched_indices = enriched_indices
    new_enriched_indices = torch.where(new_enriched_mask)[0].cpu().numpy()
    
    # Create mapping from old indices to new indices
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(new_enriched_indices)}
    
    # Filter leading_edge_masks and rank_at_max
    if 'leading_edge_masks' in result and len(result['leading_edge_masks']) > 0:
        filtered_masks = []
        filtered_ranks = []
        for i, old_idx in enumerate(old_enriched_indices):
            if old_idx in old_to_new:
                filtered_masks.append(result['leading_edge_masks'][i])
                filtered_ranks.append(result['rank_at_max'][i])
        
        if filtered_masks:
            result['leading_edge_masks'] = torch.stack(filtered_masks)
            result['rank_at_max'] = torch.stack(filtered_ranks)
        else:
            n_module_genes = result['leading_edge_masks'].shape[1]
            result['leading_edge_masks'] = torch.zeros((0, n_module_genes), dtype=torch.bool, device=device)
            result['rank_at_max'] = torch.zeros(0, dtype=torch.long, device=device)
    
    return result


def compute_leading_edge(
    rcc: torch.Tensor,
    avg2std_rcc: torch.Tensor,
    rankings: torch.Tensor,
    weights: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Compute leading edge genes for an enriched motif.

    Args:
        rcc: (rank_threshold,) - recovery curve for this motif
        avg2std_rcc: (rank_threshold,) - average + 2*std recovery curve
        rankings: (n_module_genes,) - rankings for module genes
        weights: (n_module_genes,) - gene weights

    Returns:
        mask: (n_module_genes,) - boolean mask for leading edge genes
        leading_weights: weights for leading edge genes
        rank_at_max: rank at maximum difference
    """
    # Find rank at maximum difference
    diff = rcc - avg2std_rcc
    rank_at_max = int(diff.argmax().item())

    # Get genes with rank <= rank_at_max
    mask = rankings <= rank_at_max

    return mask, weights[mask], rank_at_max


def prune_single_module(
    rankings: torch.Tensor,
    module_gene_indices: torch.Tensor,
    rank_threshold: int = 5000,
    auc_threshold: float = 0.05,
    nes_threshold: float = 3.0,
    weights: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    """
    Perform cisTarget pruning for a single module.

    All inputs and outputs are tensors on the same device.

    Args:
        rankings: (n_motifs, n_genes) - ranking database tensor
        module_gene_indices: (n_module_genes,) - gene indices for this module
        rank_threshold: Maximum rank for recovery curve
        auc_threshold: Fraction of genome for AUC
        nes_threshold: NES threshold for enrichment
        weights: Optional (n_module_genes,) gene weights

    Returns:
        Dict with keys:
            - enriched_mask: (n_motifs,) bool - which motifs are enriched
            - nes: (n_motifs,) - NES scores
            - aucs: (n_motifs,) - AUC scores
            - rccs: (n_motifs, rank_threshold) - recovery curves
            - leading_edge_masks: (n_enriched, n_module_genes) - leading edge for each enriched motif
            - rank_at_max: (n_enriched,) - rank at max for each enriched motif
    """
    device = rankings.device
    n_motifs = rankings.shape[0]
    n_module_genes = len(module_gene_indices)

    if weights is None:
        weights = torch.ones(n_module_genes, device=device, dtype=torch.float32)

    # Compute AUCs and recovery curves
    rccs, aucs = compute_recovery_aucs(
        rankings, module_gene_indices, rank_threshold, auc_threshold, weights
    )

    # Compute NES
    nes = compute_nes(aucs)

    # Filter enriched motifs
    enriched_mask = nes >= nes_threshold
    enriched_indices = torch.where(enriched_mask)[0]
    n_enriched = len(enriched_indices)

    result = {
        'enriched_mask': enriched_mask,
        'nes': nes,
        'aucs': aucs,
        'rccs': rccs,
    }

    if n_enriched == 0:
        result['leading_edge_masks'] = torch.zeros((0, n_module_genes), dtype=torch.bool, device=device)
        result['rank_at_max'] = torch.zeros(0, dtype=torch.long, device=device)
        return result

    # Compute average + 2*std recovery curve
    avg_rcc = rccs.mean(dim=0)
    std_rcc = rccs.std(dim=0, unbiased=False)
    avg2std_rcc = avg_rcc + 2.0 * std_rcc

    # Get module gene rankings for all motifs
    module_rankings = rankings[:, module_gene_indices]  # (n_motifs, n_module_genes)

    # Compute leading edge for enriched motifs
    leading_edge_masks = torch.zeros((n_enriched, n_module_genes), dtype=torch.bool, device=device)
    rank_at_max = torch.zeros(n_enriched, dtype=torch.long, device=device)

    for i, motif_idx in enumerate(enriched_indices):
        mask, _, r_max = compute_leading_edge(
            rccs[motif_idx], avg2std_rcc, module_rankings[motif_idx], weights
        )
        leading_edge_masks[i] = mask
        rank_at_max[i] = r_max

    result['leading_edge_masks'] = leading_edge_masks
    result['rank_at_max'] = rank_at_max

    return result

class CisTargetPruner:
    """
    GPU-accelerated cisTarget pruning with support for single or multiple databases.
    
    Example (single database):
        ```python
        pruner = CisTargetPruner(device='cuda')
        pruner.load_database('rankings.feather')
        pruner.load_annotations('motifs.tbl', filter_for_annotation=True)
        
        # Prune with tensor input
        result = pruner.prune(module_gene_indices)
        ```
    
    Example (multiple databases):
        ```python
        pruner = CisTargetPruner(device='cuda')
        pruner.load_database(['db_500bp.feather', 'db_10kb.feather'])
        pruner.load_annotations('motifs.tbl')
        
        # Prune modules across all databases
        regulon_info = pruner.prune_modules(modules, tf_names, gene_names)
        ```
    """

    def __init__(
        self,
        rank_threshold: int = 5000,  # Match pyscenic CLI default
        auc_threshold: float = 0.05,
        nes_threshold: float = 3.0,
        device: str = 'cuda',
        min_genes_per_regulon: int = 0,  # Minimum genes per regulon (for multi-db mode)
        merge_strategy: str = 'union'  # 'union' or 'best' - how to merge regulons from multiple DBs
    ):
        self.rank_threshold = rank_threshold
        self.auc_threshold = auc_threshold
        self.nes_threshold = nes_threshold
        self.device = device
        self.min_genes_per_regulon = min_genes_per_regulon
        self.merge_strategy = merge_strategy

        # Single database mode
        self.rankings: Optional[torch.Tensor] = None
        self.n_motifs: int = 0
        self.n_genes: int = 0
        self.database_name: Optional[str] = None
        self.motif_names: Optional[List[str]] = None
        self.gene_names: Optional[List[str]] = None
        self.gene_to_idx: Optional[Dict[str, int]] = None
        
        # Multi-database mode
        self.pruners: List['CisTargetPruner'] = []
        self.database_names: List[str] = []
        self._multi_db_mode: bool = False
        
        # Motif annotations (shared across all databases)
        self.motif_annotations: Optional[MotifAnnotation] = None
        self.filter_for_annotation: bool = True

    def load_database(
        self,
        paths: Union[str, List[str]],
        database_names: Optional[Union[str, List[str]]] = None
    ):
        """
        Load ranking database(s) from feather file(s).

        Args:
            paths: Path to .feather ranking database, or list of paths for multiple databases
            database_names: Optional name(s) for database(s) (defaults to filename(s))
        """
        import pyarrow.feather as pf
        import os

        # Check if single or multiple databases
        if isinstance(paths, str):
            # Single database mode
            self._multi_db_mode = False
            path = paths
            database_name = database_names if isinstance(database_names, str) else None
            
            if database_name is None:
                database_name = os.path.splitext(os.path.basename(path))[0]
            self.database_name = database_name

            table = pf.read_table(path)
            columns = table.column_names

            # First column is usually the index (motif/region names)
            # Remaining columns are genes
            index_col = columns[-1]
            gene_cols = columns[:-1]

            # Extract data
            self.motif_names = table.column(index_col).to_pylist()
            self.gene_names = list(gene_cols)
            self.gene_to_idx = {g: i for i, g in enumerate(self.gene_names)}

            # Load rankings as numpy then convert to tensor
            ranking_data = table.select(gene_cols).to_pandas().values

            self.rankings = torch.tensor(
                ranking_data,
                device=self.device,
                dtype=torch.int32
            )

            self.n_motifs, self.n_genes = self.rankings.shape
            print(f"Loaded database '{database_name}': {self.n_motifs} motifs × {self.n_genes} genes")
            
        else:
            # Multiple databases mode
            self._multi_db_mode = True
            database_paths = paths
            
            if database_names is None:
                database_names = [os.path.splitext(os.path.basename(p))[0] for p in database_paths]
            elif isinstance(database_names, str):
                database_names = [database_names]
            
            self.pruners = []
            self.database_names = []
            
            for path, name in zip(database_paths, database_names):
                pruner = CisTargetPruner(
                    rank_threshold=self.rank_threshold,
                    auc_threshold=self.auc_threshold,
                    nes_threshold=self.nes_threshold,
                    device=self.device
                )
                # For single database, pass database_name as string
                pruner.load_database(path, database_names=name)
                
                # Share motif annotations if loaded
                if self.motif_annotations is not None:
                    pruner.motif_annotations = self.motif_annotations
                    pruner.filter_for_annotation = self.filter_for_annotation
                
                self.pruners.append(pruner)
                self.database_names.append(name)
            
            print(f"Loaded {len(self.pruners)} databases")

    def load_annotations(
        self,
        annotation_file: str,
        filter_for_annotation: bool = True,
        motif_similarity_fdr: float = 0.001,
        orthologous_identity_threshold: float = 0.0
    ):
        """
        Load motif annotations and enable filtering.
        
        Args:
            annotation_file: Path to motif annotation TSV file
            filter_for_annotation: If True, filter enriched motifs to keep only those with annotations
            motif_similarity_fdr: Maximum FDR threshold (default: 0.001)
            orthologous_identity_threshold: Minimum orthologous identity (default: 0.0)
        """
        self.motif_annotations = MotifAnnotation.load_from_file(
            annotation_file,
            motif_similarity_fdr=motif_similarity_fdr,
            orthologous_identity_threshold=orthologous_identity_threshold
        )
        self.filter_for_annotation = filter_for_annotation
        
        # Update sub-pruners if in multi-db mode
        if self._multi_db_mode:
            for pruner in self.pruners:
                pruner.motif_annotations = self.motif_annotations
                pruner.filter_for_annotation = self.filter_for_annotation
        
        print(f"Loaded motif annotations (filter_for_annotation={filter_for_annotation})")

    def load_from_tensor(
        self,
        rankings: ArrayLike,
        motif_names: Optional[List[str]] = None,
        gene_names: Optional[List[str]] = None
    ):
        """
        Load database from tensor/array directly.

        Args:
            rankings: (n_motifs, n_genes) ranking matrix
            motif_names: Optional list of motif names
            gene_names: Optional list of gene names
        """
        if isinstance(rankings, np.ndarray):
            rankings = torch.from_numpy(rankings)

        self.rankings = rankings.to(device=self.device, dtype=torch.int32)
        self.n_motifs, self.n_genes = self.rankings.shape

        self.motif_names = motif_names
        self.gene_names = gene_names
        if gene_names:
            self.gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    def genes_to_indices(self, genes: List[str]) -> torch.Tensor:
        """Convert gene names to indices tensor."""
        if self.gene_to_idx is None:
            raise ValueError("Gene names not loaded. Use load_from_tensor with gene_names.")

        indices = [self.gene_to_idx[g] for g in genes if g in self.gene_to_idx]
        return torch.tensor(indices, device=self.device, dtype=torch.long)

    def prune(
        self,
        module_gene_indices: ArrayLike,
        weights: Optional[ArrayLike] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Prune a single module (single database mode only).

        Args:
            module_gene_indices: (n_module_genes,) indices of genes in module
            weights: Optional (n_module_genes,) gene weights

        Returns:
            Dict with pruning results (all tensors)
        """
        if self._multi_db_mode:
            raise ValueError("prune() is for single database mode. Use prune_modules() for multiple databases.")
        
        if self.rankings is None:
            raise ValueError("Database not loaded")

        # Convert to tensor if needed
        if isinstance(module_gene_indices, np.ndarray):
            module_gene_indices = torch.from_numpy(module_gene_indices)
        module_gene_indices = module_gene_indices.to(device=self.device, dtype=torch.long)

        if weights is not None:
            if isinstance(weights, np.ndarray):
                weights = torch.from_numpy(weights)
            weights = weights.to(device=self.device, dtype=torch.float32)

        result = prune_single_module(
            self.rankings,
            module_gene_indices,
            self.rank_threshold,
            self.auc_threshold,
            self.nes_threshold,
            weights
        )
        
        # Apply annotation filtering if enabled
        if self.motif_annotations is not None and self.filter_for_annotation:
            result = filter_by_annotations(
                result,
                self.motif_names,
                self.motif_annotations,
                filter_for_annotation=self.filter_for_annotation
            )
        
        return result

    def prune_batch(
        self,
        modules: List[torch.Tensor],
        weights_list: Optional[List[torch.Tensor]] = None
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Prune multiple modules.

        Args:
            modules: List of (n_genes_i,) tensors with gene indices
            weights_list: Optional list of weight tensors

        Returns:
            List of pruning result dicts
        """
        results = []
        for i, module_indices in enumerate(modules):
            weights = weights_list[i] if weights_list else None
            results.append(self.prune(module_indices, weights))
        return results

    def get_enriched_motif_names(self, result: Dict[str, torch.Tensor]) -> List[str]:
        """Get names of enriched motifs from pruning result."""
        if self.motif_names is None:
            raise ValueError("Motif names not loaded")

        enriched_indices = torch.where(result['enriched_mask'])[0].cpu().numpy()
        return [self.motif_names[i] for i in enriched_indices]

    def get_leading_edge_genes(
        self,
        result: Dict[str, torch.Tensor],
        module_gene_indices: torch.Tensor
    ) -> List[List[str]]:
        """
        Get leading edge gene names for each enriched motif.

        Args:
            result: Pruning result dict
            module_gene_indices: Original module gene indices

        Returns:
            List of gene name lists, one per enriched motif
        """
        if self.gene_names is None:
            raise ValueError("Gene names not loaded")

        leading_edges = []
        for mask in result['leading_edge_masks']:
            gene_indices = module_gene_indices[mask].cpu().numpy()
            genes = [self.gene_names[i] for i in gene_indices]
            leading_edges.append(genes)

        return leading_edges

    def prune_modules(
        self,
        modules: List[torch.Tensor],
        tf_names: List[str],
        gene_names: List[str],
        weights_list: Optional[List[torch.Tensor]] = None
    ) -> List[Dict]:
        """
        Prune modules across all databases and merge results (multi-database mode only).
        
        Args:
            modules: List of (n_genes_i,) tensors with gene indices for each TF module
            tf_names: List of TF names corresponding to modules
            gene_names: List of all gene names
            weights_list: Optional list of weight tensors for each module
        
        Returns:
            List of regulon dictionaries with keys: name, tf, motif, n_genes, genes, context, nes, auc
        """
        if not self._multi_db_mode:
            raise ValueError("prune_modules() is for multi-database mode. Use prune() for single database.")
        
        all_regulons = []
        
        # Prune modules for each database
        for db_idx, (pruner, db_name) in enumerate(zip(self.pruners, self.database_names)):
            print(f"\nPruning with database {db_idx + 1}/{len(self.pruners)}: {db_name}")
            
            for module_idx, (module_indices, tf_name) in enumerate(zip(modules, tf_names)):
                # Convert gene indices to names, then to database indices
                target_genes = [gene_names[idx] for idx in module_indices.cpu().numpy()]
                db_indices = pruner.genes_to_indices(target_genes)
                
                if len(db_indices) < 20:  # Skip if too few genes mapped
                    continue
                
                # Get weights if provided
                weights = weights_list[module_idx] if weights_list else None
                
                # Prune
                result = pruner.prune(db_indices, weights)
                
                # Check for enriched motifs
                n_enriched = result['enriched_mask'].sum().item()
                if n_enriched > 0:
                    # Get enriched motifs and their info
                    enriched_indices = torch.where(result['enriched_mask'])[0].cpu().numpy()
                    enriched_motif_names = pruner.get_enriched_motif_names(result)
                    leading_edge_genes = pruner.get_leading_edge_genes(result, db_indices)
                    nes_values = result['nes'][enriched_indices].cpu().numpy()
                    auc_values = result['aucs'][enriched_indices].cpu().numpy()
                    
                    # Create regulon for each enriched motif
                    for motif_name, le_genes, nes, auc in zip(
                        enriched_motif_names, leading_edge_genes, nes_values, auc_values
                    ):
                        # Filter by minimum genes if specified
                        if len(le_genes) < self.min_genes_per_regulon:
                            continue
                        
                        regulon_name = f"{tf_name}_{motif_name}"
                        all_regulons.append({
                            'name': regulon_name,
                            'tf': tf_name,
                            'motif': motif_name,
                            'n_genes': len(le_genes),
                            'genes': le_genes,
                            'context': db_name,  # Store database name in context
                            'nes': float(nes),
                            'auc': float(auc),
                            'database': db_name
                        })
        
        # Merge regulons from multiple databases
        print(f"\nTotal regulons before merging: {len(all_regulons)}")
        merged_regulons = self._merge_regulons(all_regulons)
        print(f"Total regulons after merging: {len(merged_regulons)}")
        
        # Merge regulons by TF (matching pyscenic behavior)
        # pyscenic groups by (TF, Type) and merges all motifs for each TF into one regulon
        final_regulons = self._merge_regulons_by_tf(merged_regulons)
        print(f"Total regulons after TF merging: {len(final_regulons)}")
        
        return final_regulons
    
    def _merge_regulons(self, regulons: List[Dict]) -> List[Dict]:
        """
        Merge regulons from multiple databases.
        
        For the same TF+motif combination:
        - If merge_strategy='union': keep all (they may have different genes from different DBs)
        - If merge_strategy='best': keep the one with highest NES
        
        Note: pyscenic uses union strategy - it merges genes from all databases
        for the same TF+motif combination.
        """
        if self.merge_strategy == 'best':
            # Group by TF+motif, keep best NES
            regulon_dict = {}
            for reg in regulons:
                key = (reg['tf'], reg['motif'])
                if key not in regulon_dict or reg['nes'] > regulon_dict[key]['nes']:
                    regulon_dict[key] = reg
            return list(regulon_dict.values())
        
        elif self.merge_strategy == 'union':
            # Group by TF+motif, merge genes (union)
            regulon_groups = defaultdict(list)
            for reg in regulons:
                key = (reg['tf'], reg['motif'])
                regulon_groups[key].append(reg)
            
            merged = []
            for (tf, motif), group in regulon_groups.items():
                # Merge genes (union)
                all_genes = set()
                all_contexts = set()
                best_nes = max(reg['nes'] for reg in group)
                best_auc = max(reg['auc'] for reg in group)
                
                for reg in group:
                    all_genes.update(reg['genes'])
                    all_contexts.add(reg['context'])
                
                merged.append({
                    'name': f"{tf}_{motif}",
                    'tf': tf,
                    'motif': motif,
                    'n_genes': len(all_genes),
                    'genes': list(all_genes),
                    'context': ','.join(sorted(all_contexts)),  # All database names
                    'nes': best_nes,
                    'auc': best_auc,
                    'database': ','.join(sorted(all_contexts))
                })
            
            return merged
        
        else:
            raise ValueError(f"Unknown merge_strategy: {self.merge_strategy}")
    
    def _merge_regulons_by_tf(self, regulons: List[Dict]) -> List[Dict]:
        """
        Merge regulons by TF, matching pyscenic's df2regulons behavior.
        
        pyscenic groups by (TF, Type) and uses Regulon.union to merge all motifs
        for each TF into a single regulon. This function implements the same logic.
        
        Args:
            regulons: List of regulon dictionaries
        
        Returns:
            Merged regulons (one per TF)
        """
        from collections import defaultdict
        
        # Group by TF
        tf_to_regulons = defaultdict(list)
        for reg in regulons:
            tf_to_regulons[reg['tf']].append(reg)
        
        merged = []
        for tf, regs in tf_to_regulons.items():
            # Merge genes (union) - matching Regulon.union behavior
            all_genes = set()
            all_motifs = set()
            best_nes = max(reg['nes'] for reg in regs)
            best_auc = max(reg['auc'] for reg in regs)
            
            # Collect all genes and motifs
            for reg in regs:
                all_genes.update(reg['genes'])
                all_motifs.add(reg['motif'])
            
            # Use TF name as regulon name (matching pyscenic: "{tf}(+)")
            # pyscenic uses the highest NES motif's name in context, but regulon name is just TF
            merged.append({
                'name': f"{tf}(+)",  # pyscenic format: TF name + interaction type
                'tf': tf,
                'motif': ','.join(sorted(all_motifs)),  # All motifs (for reference)
                'n_genes': len(all_genes),
                'genes': list(all_genes),
                'nes': best_nes,  # Keep best NES (pyscenic keeps max combined score)
                'auc': best_auc,
                'context': regs[0].get('context', ''),  # Keep context from first regulon
                'database': regs[0].get('database', '')  # Keep database info
            })
        
        return merged

    def clear_gpu_memory(self):
        """Release GPU memory."""
        if self._multi_db_mode:
            for pruner in self.pruners:
                pruner.clear_gpu_memory()
        else:
            if self.rankings is not None:
                del self.rankings
                self.rankings = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Backwards compatibility aliases
MultiDatabaseCisTargetPruner = CisTargetPruner
GPUCisTargetPruner = CisTargetPruner
