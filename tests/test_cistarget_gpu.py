"""
Test GPU cisTarget pruning implementation against pySCENIC.
"""
import numpy as np
import pandas as pd
import torch
import sys
sys.path.insert(0, '/orcd/data/omarabu/001/hao/flashscenic')


def test_auc_computation():
    """Test that GPU AUC computation matches pySCENIC."""
    from flashscenic.cistarget import compute_recovery_aucs_gpu_batch
    from ctxcore.recovery import aucs as pyscenic_aucs, recovery
    
    np.random.seed(42)
    
    n_motifs = 100
    n_genes = 1000
    n_module_genes = 50
    rank_threshold = 500
    auc_threshold = 0.05
    
    # Create synthetic ranking matrix
    # Rankings are 0-indexed, each row is a permutation of 0 to n_genes-1
    ranking_matrix = np.zeros((n_motifs, n_genes), dtype=np.int32)
    for i in range(n_motifs):
        ranking_matrix[i] = np.random.permutation(n_genes)
    
    # Select random module genes
    module_gene_indices = np.random.choice(n_genes, n_module_genes, replace=False)
    weights = np.ones(n_module_genes, dtype=np.float32)
    
    # GPU computation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rankings_tensor = torch.tensor(ranking_matrix, device=device, dtype=torch.int32)
    gene_idx_tensor = torch.tensor(module_gene_indices, device=device, dtype=torch.long)
    weights_tensor = torch.tensor(weights, device=device, dtype=torch.float32)
    
    rccs_gpu, aucs_gpu = compute_recovery_aucs_gpu_batch(
        rankings_tensor, gene_idx_tensor, rank_threshold, auc_threshold, weights_tensor
    )
    aucs_gpu = aucs_gpu.cpu().numpy()
    
    # pySCENIC computation
    # Create DataFrame for pySCENIC
    gene_names = [f'gene_{i}' for i in range(n_genes)]
    motif_names = [f'motif_{i}' for i in range(n_motifs)]
    
    ranking_df = pd.DataFrame(
        ranking_matrix,
        index=motif_names,
        columns=gene_names
    )
    
    # Subset to module genes
    module_gene_names = [gene_names[i] for i in module_gene_indices]
    module_ranking_df = ranking_df[module_gene_names]
    
    aucs_pyscenic = pyscenic_aucs(module_ranking_df, n_genes, weights, auc_threshold)
    
    # Compare results
    diff = np.abs(aucs_gpu - aucs_pyscenic)
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    print(f"AUC Computation Test:")
    print(f"  Max absolute difference: {max_diff:.6e}")
    print(f"  Mean absolute difference: {mean_diff:.6e}")
    print(f"  Correlation: {np.corrcoef(aucs_gpu, aucs_pyscenic)[0, 1]:.6f}")
    
    # Check if close enough (allowing for small numerical differences)
    success = np.allclose(aucs_gpu, aucs_pyscenic, rtol=1e-4, atol=1e-5)
    print(f"  Results match: {success}")
    
    return success


def test_nes_computation():
    """Test NES computation."""
    from flashscenic.cistarget import compute_nes
    
    # Create test AUC values
    aucs = np.random.rand(100).astype(np.float32)
    aucs_tensor = torch.tensor(aucs)
    
    # GPU NES
    nes_gpu = compute_nes(aucs_tensor).numpy()
    
    # Manual NES (pySCENIC formula) - use ddof=0 for population std
    nes_manual = (aucs - aucs.mean()) / aucs.std(ddof=0)
    
    diff = np.abs(nes_gpu - nes_manual)
    # Use relaxed tolerance for float32 operations
    success = np.allclose(nes_gpu, nes_manual, rtol=1e-4, atol=1e-6)
    
    print(f"\nNES Computation Test:")
    print(f"  Max absolute difference: {diff.max():.6e}")
    print(f"  Results match: {success}")
    
    return success


def test_full_pruning():
    """Test the full pruning pipeline."""
    from flashscenic.cistarget import prune_module_gpu
    
    np.random.seed(42)
    
    n_motifs = 200
    n_genes = 500
    n_module_genes = 30
    
    # Create ranking matrix
    ranking_matrix = np.zeros((n_motifs, n_genes), dtype=np.int32)
    for i in range(n_motifs):
        ranking_matrix[i] = np.random.permutation(n_genes)
    
    module_gene_indices = np.random.choice(n_genes, n_module_genes, replace=False)
    
    gene_names = [f'gene_{i}' for i in range(n_genes)]
    motif_names = [f'motif_{i}' for i in range(n_motifs)]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    result = prune_module_gpu(
        ranking_matrix,
        module_gene_indices,
        motif_names,
        gene_names,
        rank_threshold=250,
        auc_threshold=0.05,
        nes_threshold=2.0,  # Lower threshold for testing
        device=device
    )
    
    print(f"\nFull Pruning Test:")
    print(f"  Device: {device}")
    print(f"  Enriched motifs found: {len(result['enriched_motifs'])}")
    if len(result['enriched_motifs']) > 0:
        print(f"  Top motif: {result['enriched_motifs'][0]}")
        print(f"  Top NES: {result['nes_scores'][0]:.4f}")
        print(f"  Target genes for top motif: {len(result['target_genes'].get(result['enriched_motifs'][0], []))}")
    
    return True


def test_batch_pruning():
    """Test batch processing of multiple modules."""
    from flashscenic.cistarget import prune_modules_batch_gpu
    
    np.random.seed(42)
    
    n_motifs = 150
    n_genes = 400
    n_modules = 5
    genes_per_module = 25
    
    # Create ranking matrix
    ranking_matrix = np.zeros((n_motifs, n_genes), dtype=np.int32)
    for i in range(n_motifs):
        ranking_matrix[i] = np.random.permutation(n_genes)
    
    gene_names = [f'gene_{i}' for i in range(n_genes)]
    motif_names = [f'motif_{i}' for i in range(n_motifs)]
    
    # Create modules
    modules = []
    for i in range(n_modules):
        gene_indices = np.random.choice(n_genes, genes_per_module, replace=False)
        modules.append({
            'name': f'module_{i}',
            'genes': [gene_names[j] for j in gene_indices],
            'tf': f'TF_{i}'
        })
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    results = prune_modules_batch_gpu(
        ranking_matrix,
        modules,
        motif_names,
        gene_names,
        rank_threshold=200,
        auc_threshold=0.05,
        nes_threshold=2.0,
        device=device
    )
    
    print(f"\nBatch Pruning Test:")
    print(f"  Modules processed: {len(results)}")
    for r in results:
        status = "skipped" if r.get('skipped', False) else f"{len(r['enriched_motifs'])} motifs"
        print(f"    {r['module_name']}: {status}")
    
    return True


def benchmark_gpu_vs_cpu():
    """Benchmark GPU vs CPU performance."""
    import time
    from flashscenic.cistarget import prune_modules_batch_gpu
    
    np.random.seed(42)
    
    n_motifs = 500
    n_genes = 2000
    n_modules = 10
    genes_per_module = 50
    
    # Create ranking matrix
    ranking_matrix = np.zeros((n_motifs, n_genes), dtype=np.int32)
    for i in range(n_motifs):
        ranking_matrix[i] = np.random.permutation(n_genes)
    
    gene_names = [f'gene_{i}' for i in range(n_genes)]
    motif_names = [f'motif_{i}' for i in range(n_motifs)]
    
    # Create modules
    modules = []
    for i in range(n_modules):
        gene_indices = np.random.choice(n_genes, genes_per_module, replace=False)
        modules.append({
            'name': f'module_{i}',
            'genes': [gene_names[j] for j in gene_indices],
            'tf': f'TF_{i}'
        })
    
    print(f"\nBenchmark: {n_motifs} motifs x {n_genes} genes, {n_modules} modules")
    
    # CPU benchmark
    start = time.time()
    results_cpu = prune_modules_batch_gpu(
        ranking_matrix, modules, motif_names, gene_names,
        rank_threshold=500, device='cpu'
    )
    cpu_time = time.time() - start
    print(f"  CPU time: {cpu_time:.3f}s")
    
    # GPU benchmark (if available)
    if torch.cuda.is_available():
        # Warm up
        _ = prune_modules_batch_gpu(
            ranking_matrix, modules[:1], motif_names, gene_names,
            rank_threshold=500, device='cuda'
        )
        torch.cuda.synchronize()
        
        start = time.time()
        results_gpu = prune_modules_batch_gpu(
            ranking_matrix, modules, motif_names, gene_names,
            rank_threshold=500, device='cuda'
        )
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        print(f"  GPU time: {gpu_time:.3f}s")
        print(f"  Speedup: {cpu_time/gpu_time:.2f}x")
    else:
        print("  GPU not available for benchmark")


if __name__ == "__main__":
    print("=" * 60)
    print("cisTarget GPU Implementation Tests")
    print("=" * 60)
    
    test1 = test_auc_computation()
    test2 = test_nes_computation()
    test3 = test_full_pruning()
    test4 = test_batch_pruning()
    
    benchmark_gpu_vs_cpu()
    
    print("\n" + "=" * 60)
    all_passed = test1 and test2 and test3 and test4
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("=" * 60)
