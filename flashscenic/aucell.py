import torch
import numpy as np

def compare_rows_element_presence(A, B):
    """
    Compares each row of B with each row of A, checking for element presence.
    For every row b in B, it compares it with every row a in A.
    If an element in row b is found within row a, it marks 1; otherwise, 0.
    Args:
        A: A torch tensor with shape (m, k).
        B: A torch tensor with shape (n, t).
    Returns:
        A torch tensor C with shape (n, m, t). C[i, j, l] is 1 if B[i, l] 
        is present in A[j], and 0 otherwise.
    """
    A_expanded = A.unsqueeze(0).unsqueeze(3)
    B_expanded = B.unsqueeze(1).unsqueeze(2)
    comparison = (A_expanded == B_expanded)
    C = comparison.any(dim=2)
    return C.int()


def get_aucell_fast(exp_array, adj_array, 
                    k=50, auc_threshold=0.05, 
                    device='cuda', batch_size=32):
    """
    Fast but approximate AUCell Score calculation using GPU.
    
    NOTE: This is a SIMPLIFIED version that assumes uniform ranking intervals.
    Use `get_aucell` for pySCENIC-equivalent results.

    Args:
        exp_array (np.ndarray): 2D numpy array. If used on single-cell RNAseq, 
            the rows are cells and the columns are genes. Data should be log 
            transformed.
        adj_array (np.ndarray): 2D numpy array (gene by gene/feature by feature).
        k (int): Top k target gene for each Transcription factor. Default is 50, 
            same as pyscenic. 
        auc_threshold (float): The fraction of the ranked genome to take into 
            account for the calculation of the Area Under the recovery Curve.
            Default is 0.05, which is the same as pyscenic. 
        device (str): Device, cpu or cuda. Default is cuda
        batch_size (int): Batch size when processing expression data. 
    """
    adj_tensor = torch.tensor(adj_array, device=device)
    exp_tensor = torch.tensor(exp_array, device=device)
    cutoff = max(1, int(adj_tensor.shape[1] * auc_threshold))
    with torch.no_grad():
        topk_exp_values, topk_exp_idx = exp_tensor.topk(cutoff, dim=1)
        topk_adj_values, topk_adj_idx = adj_tensor.topk(k, dim=1)
        n_samples = exp_tensor.shape[0]
        processed_aucs = []
        for i in range(0, n_samples, batch_size):
            batch = topk_exp_idx[i:(i+batch_size), :]
            hit_tensor = compare_rows_element_presence(topk_adj_idx, batch)
            auc = hit_tensor.cumsum(dim=2).sum(dim=2) / k / cutoff
            processed_aucs.append(auc.cpu().detach().numpy())
        full_aucell = np.concatenate(processed_aucs)
    return full_aucell


def get_aucell(exp_array, adj_array, 
               k=50, auc_threshold=0.05, 
               device='cuda', batch_size=32,
               seed=None):
    """
    pySCENIC-equivalent AUCell Score calculation using GPU.
    
    This implementation matches the mathematical formula used by pySCENIC/ctxcore:
    - Creates a ranking matrix (descending order, 0 = highest expression)
    - For each TF's regulon, calculates AUC using the trapezoidal rule
    - AUC = sum(diff(ranks) * cumsum(weights)) / max_auc
    - max_auc = (rank_cutoff + 1) * sum(weights)

    Args:
        exp_array (np.ndarray): 2D numpy array. If used on single-cell RNAseq, 
            the rows are cells and the columns are genes. Data should be log 
            transformed.
        adj_array (np.ndarray): 2D numpy array (gene by gene/feature by feature).
            This represents the adjacency/importance matrix from GRN inference.
        k (int): Top k target gene for each Transcription factor. Default is 50, 
            same as pyscenic. 
        auc_threshold (float): The fraction of the ranked genome to take into 
            account for the calculation of the Area Under the recovery Curve.
            Default is 0.05, which is the same as pyscenic. 
        device (str): Device, cpu or cuda. Default is cuda
        batch_size (int): Batch size when processing expression data. 
        seed (int): Random seed for tie-breaking in ranking. Default is None.
    
    Returns:
        np.ndarray: AUCell scores matrix of shape (n_cells, n_TFs)
    """
    n_cells, n_genes = exp_array.shape
    n_tfs = adj_array.shape[0]
    
    # Calculate rank cutoff (same as pySCENIC: round(threshold * total) - 1)
    rank_cutoff = int(round(auc_threshold * n_genes)) - 1
    rank_cutoff = max(1, rank_cutoff)
    
    # Get top k target genes for each TF from adjacency matrix
    adj_tensor = torch.tensor(adj_array, device=device, dtype=torch.float32)
    _, topk_adj_idx = adj_tensor.topk(k, dim=1)  # (n_tfs, k)
    topk_adj_idx = topk_adj_idx.cpu().numpy()
    
    # Create ranking matrix using numpy (descending order, 0 = highest)
    # Shuffle to handle ties randomly (same as pySCENIC)
    if seed is not None:
        np.random.seed(seed)
    
    # Rank each cell's genes (higher expression = lower rank = 0-indexed)
    # argsort of argsort gives ranks; we need descending so negate expression
    exp_tensor = torch.tensor(exp_array, device=device, dtype=torch.float32)
    
    # Process in batches
    all_aucs = []
    
    with torch.no_grad():
        for i in range(0, n_cells, batch_size):
            batch_exp = exp_tensor[i:min(i+batch_size, n_cells), :]  # (batch, n_genes)
            batch_size_actual = batch_exp.shape[0]
            
            # Create ranking matrix (descending order)
            # Add small noise for tie-breaking (similar to pySCENIC's shuffle approach)
            if seed is not None:
                torch.manual_seed(seed + i)
            noise = torch.rand_like(batch_exp) * 1e-10
            batch_exp_noisy = batch_exp + noise
            
            # Get rankings: argsort(argsort(-values)) gives ranks in descending order
            order = torch.argsort(-batch_exp_noisy, dim=1)
            rankings = torch.argsort(order, dim=1)  # (batch, n_genes)
            
            # For each TF, calculate AUC
            batch_aucs = torch.zeros((batch_size_actual, n_tfs), device=device)
            
            for tf_idx in range(n_tfs):
                target_genes = topk_adj_idx[tf_idx]  # (k,) gene indices
                
                # Get rankings of target genes for this batch
                target_rankings = rankings[:, target_genes]  # (batch, k)
                
                # Calculate AUC using trapezoidal rule (vectorized)
                # Filter rankings < rank_cutoff and compute AUC
                batch_aucs[:, tf_idx] = _compute_auc_batch(
                    target_rankings, rank_cutoff, k, device
                )
            
            all_aucs.append(batch_aucs.cpu().numpy())
    
    return np.concatenate(all_aucs, axis=0)


def _compute_auc_batch(target_rankings, rank_cutoff, k, device):
    """
    Compute AUC for a batch of cells using the pySCENIC formula.
    
    Args:
        target_rankings: (batch_size, k) tensor of rankings for target genes
        rank_cutoff: maximum rank to consider
        k: number of target genes
        device: torch device
    
    Returns:
        (batch_size,) tensor of AUC values
    """
    batch_size = target_rankings.shape[0]
    
    # max_auc = (rank_cutoff + 1) * k (assuming uniform weights of 1)
    max_auc = float((rank_cutoff + 1) * k)
    
    # For each cell, calculate AUC using the recovery curve approach
    # Mask rankings >= rank_cutoff
    mask = target_rankings < rank_cutoff  # (batch, k)
    
    # Set masked rankings to a large value for sorting
    masked_rankings = target_rankings.clone().float()
    masked_rankings[~mask] = float('inf')
    
    # Sort rankings and compute cumulative hits
    sorted_rankings, _ = torch.sort(masked_rankings, dim=1)  # (batch, k)
    
    # Count valid rankings per cell
    valid_counts = mask.sum(dim=1)  # (batch,)
    
    # Compute AUC using trapezoidal rule
    # AUC = sum over i of (rank[i+1] - rank[i]) * i for sorted ranks within cutoff
    # Plus (rank_cutoff - last_rank) * num_hits
    
    aucs = torch.zeros(batch_size, device=device)
    
    for b in range(batch_size):
        n_valid = int(valid_counts[b].item())
        if n_valid == 0:
            continue
        
        valid_ranks = sorted_rankings[b, :n_valid]
        
        # Add rank_cutoff as the end point
        ranks_with_cutoff = torch.cat([valid_ranks, torch.tensor([rank_cutoff], device=device, dtype=torch.float32)])
        
        # Cumulative count at each rank (1, 2, 3, ...)
        cumsum_weights = torch.arange(1, n_valid + 1, device=device, dtype=torch.float32)
        
        # Trapezoidal area: sum of (rank[i+1] - rank[i]) * cumsum[i]
        rank_diffs = ranks_with_cutoff[1:] - ranks_with_cutoff[:-1]
        auc = (rank_diffs * cumsum_weights).sum()
        
        aucs[b] = auc / max_auc
    
    return aucs


def get_aucell_vectorized(exp_array, adj_array, 
                          k=50, auc_threshold=0.05, 
                          device='cuda', batch_size=32,
                          seed=None):
    """
    Fully vectorized pySCENIC-equivalent AUCell calculation.
    
    This version avoids Python loops over TFs and cells for better GPU utilization.
    Trade-off: Uses more memory but is significantly faster for large datasets.

    Args:
        exp_array (np.ndarray): Expression matrix (n_cells x n_genes)
        adj_array (np.ndarray): Adjacency matrix (n_tfs x n_genes)
        k (int): Top k target genes per TF. Default is 50.
        auc_threshold (float): Fraction of genome for AUC calculation. Default is 0.05.
        device (str): Device, 'cpu' or 'cuda'. Default is 'cuda'.
        batch_size (int): Batch size for processing cells. Default is 32.
        seed (int): Random seed for tie-breaking. Default is None.
    
    Returns:
        np.ndarray: AUCell scores matrix of shape (n_cells, n_TFs)
    """
    n_cells, n_genes = exp_array.shape
    n_tfs = adj_array.shape[0]
    
    # Calculate rank cutoff
    rank_cutoff = max(1, int(round(auc_threshold * n_genes)) - 1)
    max_auc = float((rank_cutoff + 1) * k)
    
    # Get top k target genes for each TF
    adj_tensor = torch.tensor(adj_array, device=device, dtype=torch.float32)
    _, topk_adj_idx = adj_tensor.topk(k, dim=1)  # (n_tfs, k)
    
    exp_tensor = torch.tensor(exp_array, device=device, dtype=torch.float32)
    all_aucs = []
    
    with torch.no_grad():
        for i in range(0, n_cells, batch_size):
            batch_exp = exp_tensor[i:min(i+batch_size, n_cells), :]
            batch_size_actual = batch_exp.shape[0]
            
            # Add noise for tie-breaking
            if seed is not None:
                torch.manual_seed(seed + i)
            noise = torch.rand_like(batch_exp) * 1e-10
            batch_exp_noisy = batch_exp + noise
            
            # Get rankings (descending order)
            order = torch.argsort(-batch_exp_noisy, dim=1)
            rankings = torch.argsort(order, dim=1)  # (batch, n_genes)
            
            # Expand rankings to get target gene rankings for all TFs at once
            # rankings: (batch, n_genes), topk_adj_idx: (n_tfs, k)
            # We want: (batch, n_tfs, k)
            
            # Gather target gene rankings
            topk_adj_idx_expanded = topk_adj_idx.unsqueeze(0).expand(batch_size_actual, -1, -1)
            # (batch, n_tfs, k)
            
            target_rankings = torch.gather(
                rankings.unsqueeze(1).expand(-1, n_tfs, -1),  # (batch, n_tfs, n_genes)
                dim=2,
                index=topk_adj_idx_expanded
            )  # (batch, n_tfs, k)
            
            # Compute AUC for all TFs at once
            batch_aucs = _compute_auc_vectorized(target_rankings, rank_cutoff, k, max_auc)
            all_aucs.append(batch_aucs.cpu().numpy())
    
    return np.concatenate(all_aucs, axis=0)


def _compute_auc_vectorized(target_rankings, rank_cutoff, k, max_auc):
    """
    Vectorized AUC computation for a batch of cells across all TFs.
    
    Args:
        target_rankings: (batch_size, n_tfs, k) tensor
        rank_cutoff: int
        k: int
        max_auc: float
    
    Returns:
        (batch_size, n_tfs) tensor of AUC values
    """
    batch_size, n_tfs, k_val = target_rankings.shape
    device = target_rankings.device
    
    # Mask rankings < rank_cutoff (valid rankings)
    mask = target_rankings < rank_cutoff  # (batch, n_tfs, k)
    
    # Count valid entries per (cell, TF)
    valid_counts = mask.sum(dim=2)  # (batch, n_tfs)
    
    # Convert to float for computation
    target_rankings_float = target_rankings.float()
    
    # Replace invalid rankings with a large value (but not inf to avoid NaN)
    # Use rank_cutoff as replacement so they sort to the end
    invalid_replacement = torch.full_like(target_rankings_float, float(rank_cutoff))
    target_rankings_masked = torch.where(mask, target_rankings_float, invalid_replacement)
    
    # Sort rankings (invalid ones will be at the end)
    sorted_rankings, _ = torch.sort(target_rankings_masked, dim=2)  # (batch, n_tfs, k)
    
    # Create position indices (1, 2, 3, ..., k)
    positions = torch.arange(1, k_val + 1, device=device, dtype=torch.float32)
    positions = positions.view(1, 1, k_val).expand(batch_size, n_tfs, k_val)
    
    # Mask positions beyond valid count
    position_mask = positions <= valid_counts.unsqueeze(2).float()  # (batch, n_tfs, k)
    
    # For the trapezoidal rule, we need:
    # AUC = sum_i (rank[i+1] - rank[i]) * i
    # where ranks are sorted and include rank_cutoff at the end
    
    # Append rank_cutoff to sorted_rankings
    cutoff_tensor = torch.full((batch_size, n_tfs, 1), rank_cutoff, device=device, dtype=torch.float32)
    sorted_with_cutoff = torch.cat([sorted_rankings, cutoff_tensor], dim=2)  # (batch, n_tfs, k+1)
    
    # Compute differences: rank[i+1] - rank[i]
    rank_diffs = sorted_with_cutoff[:, :, 1:] - sorted_with_cutoff[:, :, :-1]  # (batch, n_tfs, k)
    
    # Cumulative weights (1, 2, 3, ...) up to valid_count
    cumsum_weights = positions * position_mask  # (batch, n_tfs, k)
    
    # AUC contribution from each step
    auc_contrib = rank_diffs * cumsum_weights
    
    # Zero out contributions from invalid positions (this handles the case where
    # invalid rankings were replaced with rank_cutoff, so rank_diffs will be 0)
    auc_contrib = auc_contrib * position_mask
    
    # Sum and normalize
    aucs = auc_contrib.sum(dim=2) / max_auc  # (batch, n_tfs)
    
    # Handle edge case: if all rankings are invalid, ensure AUC is 0 (not NaN)
    aucs = torch.where(valid_counts > 0, aucs, torch.zeros_like(aucs))
    
    return aucs
