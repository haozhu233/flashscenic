import torch
import numpy as np

def get_aucell(exp_array, adj_array, 
                          k=50, auc_threshold=0.05, 
                          device='cuda', batch_size=32,
                          seed=None):
    """
    Fully vectorized pySCENIC-equivalent AUCell calculation.
    
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
            batch_aucs = _compute_auc(target_rankings, rank_cutoff, k, max_auc)
            all_aucs.append(batch_aucs.cpu().numpy())
    
    return np.concatenate(all_aucs, axis=0)


def _compute_auc(target_rankings, rank_cutoff, k, max_auc):
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
