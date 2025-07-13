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

def get_aucell(exp_array, adj_array, 
               k=50, auc_threshold=0.05, 
               device='cuda', batch_size=32):
    """
    Torch based AUCell Score calculation. 

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
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch = topk_exp_idx[i:(i+batch_size), :]
                hit_tensor = compare_rows_element_presence(topk_adj_idx, batch)
                auc = hit_tensor.cumsum(dim=2).sum(dim=2) / k / cutoff
                processed_aucs.append(auc.cpu().detach().numpy())
        full_aucell = np.concat(processed_aucs)
    return full_aucell