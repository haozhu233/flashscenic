"""
Test to verify that flashscenic's AUCell implementation matches pySCENIC.

Usage:
    python test_aucell_pyscenic_equivalence.py [--device cpu|cuda] [--n_cells N] [--n_genes N] [--n_tfs N] [--k N]

Examples:
    # Run on CPU with default sizes
    python test_aucell_pyscenic_equivalence.py --device cpu
    
    # Run on GPU with larger dataset
    python test_aucell_pyscenic_equivalence.py --device cuda --n_cells 5000 --n_genes 2000 --n_tfs 50
"""
import numpy as np
import pandas as pd
import sys
import time
import argparse
sys.path.insert(0, '/orcd/data/omarabu/001/hao/flashscenic')

# Check if CUDA is available
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        CUDA_DEVICE_NAME = torch.cuda.get_device_name(0)
    else:
        CUDA_DEVICE_NAME = "N/A"
except ImportError:
    CUDA_AVAILABLE = False
    CUDA_DEVICE_NAME = "N/A"


def test_aucell_equivalence(
    device: str = 'cpu',
    n_cells: int = 100,
    n_genes: int = 500,
    n_tfs: int = 10,
    k: int = 20,
    auc_threshold: float = 0.05,
    batch_size: int = 32,
    seed: int = 42
):
    """
    Test that get_aucell produces results equivalent to pySCENIC's aucell.
    
    Args:
        device: 'cpu' or 'cuda'
        n_cells: Number of cells/samples
        n_genes: Number of genes/features
        n_tfs: Number of transcription factors
        k: Top k target genes per TF
        auc_threshold: AUC threshold
        batch_size: Batch size for processing
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with test results and timing information
    """
    from flashscenic.aucell import get_aucell, get_aucell_vectorized, get_aucell_fast
    from ctxcore.recovery import aucs as pyscenic_aucs, derive_rank_cutoff
    from pyscenic.aucell import create_rankings
    
    print(f"\n{'='*60}")
    print(f"Test Configuration:")
    print(f"  Device: {device}")
    print(f"  n_cells: {n_cells}")
    print(f"  n_genes: {n_genes}")
    print(f"  n_tfs: {n_tfs}")
    print(f"  k (targets per TF): {k}")
    print(f"  auc_threshold: {auc_threshold}")
    print(f"  batch_size: {batch_size}")
    print(f"{'='*60}\n")
    
    results = {
        'device': device,
        'n_cells': n_cells,
        'n_genes': n_genes,
        'n_tfs': n_tfs,
        'k': k,
        'timings': {}
    }
    
    # Create test data
    np.random.seed(seed)
    
    print("Generating test data...")
    t0 = time.time()
    exp_array = np.random.randn(n_cells, n_genes).astype(np.float32)
    exp_array = np.clip(exp_array, 0, None)  # Non-negative expression
    adj_array = np.random.rand(n_tfs, n_genes).astype(np.float32)
    topk_indices = np.argsort(-adj_array, axis=1)[:, :k]
    results['timings']['data_generation'] = time.time() - t0
    print(f"  Data generation time: {results['timings']['data_generation']:.3f}s")
    
    # Test flashscenic get_aucell
    print(f"\nTesting flashscenic get_aucell on {device}...")
    
    # Warm-up run for GPU
    if device == 'cuda':
        _ = get_aucell(exp_array[:10], adj_array, k=k, auc_threshold=auc_threshold, 
                       device=device, batch_size=batch_size, seed=seed)
        if CUDA_AVAILABLE:
            import torch
            torch.cuda.synchronize()
    
    t0 = time.time()
    flash_aucs = get_aucell(
        exp_array, adj_array, 
        k=k, auc_threshold=auc_threshold, 
        device=device, batch_size=batch_size,
        seed=seed
    )
    if device == 'cuda' and CUDA_AVAILABLE:
        import torch
        torch.cuda.synchronize()
    results['timings']['get_aucell'] = time.time() - t0
    print(f"  Shape: {flash_aucs.shape}")
    print(f"  Time: {results['timings']['get_aucell']:.3f}s")
    
    # Test flashscenic get_aucell_vectorized
    print(f"\nTesting flashscenic get_aucell_vectorized on {device}...")
    
    # Warm-up
    if device == 'cuda':
        _ = get_aucell_vectorized(exp_array[:10], adj_array, k=k, auc_threshold=auc_threshold,
                                   device=device, batch_size=batch_size, seed=seed)
        if CUDA_AVAILABLE:
            import torch
            torch.cuda.synchronize()
    
    t0 = time.time()
    flash_aucs_vec = get_aucell_vectorized(
        exp_array, adj_array,
        k=k, auc_threshold=auc_threshold,
        device=device, batch_size=batch_size,
        seed=seed
    )
    if device == 'cuda' and CUDA_AVAILABLE:
        import torch
        torch.cuda.synchronize()
    results['timings']['get_aucell_vectorized'] = time.time() - t0
    print(f"  Shape: {flash_aucs_vec.shape}")
    print(f"  Time: {results['timings']['get_aucell_vectorized']:.3f}s")
    
    # Compare the two flashscenic implementations
    diff_internal = np.abs(flash_aucs - flash_aucs_vec).max()
    print(f"  Max diff between get_aucell and get_aucell_vectorized: {diff_internal:.6e}")
    
    # Test flashscenic get_aucell_fast (approximate)
    print(f"\nTesting flashscenic get_aucell_fast on {device}...")
    
    # Warm-up
    if device == 'cuda':
        _ = get_aucell_fast(exp_array[:10], adj_array, k=k, auc_threshold=auc_threshold,
                            device=device, batch_size=batch_size)
        if CUDA_AVAILABLE:
            import torch
            torch.cuda.synchronize()
    
    t0 = time.time()
    flash_aucs_fast = get_aucell_fast(
        exp_array, adj_array,
        k=k, auc_threshold=auc_threshold,
        device=device, batch_size=batch_size
    )
    if device == 'cuda' and CUDA_AVAILABLE:
        import torch
        torch.cuda.synchronize()
    results['timings']['get_aucell_fast'] = time.time() - t0
    print(f"  Shape: {flash_aucs_fast.shape}")
    print(f"  Time: {results['timings']['get_aucell_fast']:.3f}s")
    
    # Test pySCENIC (CPU only, for reference)
    print("\nTesting pySCENIC (CPU baseline)...")
    exp_df = pd.DataFrame(exp_array, 
                          index=[f'cell_{i}' for i in range(n_cells)],
                          columns=[f'gene_{i}' for i in range(n_genes)])
    
    t0 = time.time()
    rnk_df = create_rankings(exp_df, seed=seed)
    results['timings']['pyscenic_create_rankings'] = time.time() - t0
    print(f"  Rankings creation time: {results['timings']['pyscenic_create_rankings']:.3f}s")
    
    t0 = time.time()
    pyscenic_results = []
    for tf_idx in range(n_tfs):
        target_genes = [f'gene_{i}' for i in topk_indices[tf_idx]]
        rnk_subset = rnk_df[target_genes]
        weights = np.ones(k)
        aucs = pyscenic_aucs(rnk_subset, n_genes, weights, auc_threshold)
        pyscenic_results.append(aucs)
    pyscenic_aucs_arr = np.stack(pyscenic_results, axis=1)
    results['timings']['pyscenic_aucell'] = time.time() - t0
    results['timings']['pyscenic_total'] = results['timings']['pyscenic_create_rankings'] + results['timings']['pyscenic_aucell']
    print(f"  AUCell computation time: {results['timings']['pyscenic_aucell']:.3f}s")
    print(f"  Total pySCENIC time: {results['timings']['pyscenic_total']:.3f}s")
    
    # Compare results
    print(f"\n{'='*60}")
    print("Accuracy Comparison:")
    print(f"{'='*60}")
    
    # get_aucell vs pySCENIC
    diff = np.abs(flash_aucs - pyscenic_aucs_arr)
    results['get_aucell_max_diff'] = diff.max()
    results['get_aucell_mean_diff'] = diff.mean()
    results['get_aucell_correlation'] = np.corrcoef(flash_aucs.flatten(), pyscenic_aucs_arr.flatten())[0, 1]
    print(f"\nget_aucell vs pySCENIC:")
    print(f"  Max absolute difference: {results['get_aucell_max_diff']:.6e}")
    print(f"  Mean absolute difference: {results['get_aucell_mean_diff']:.6e}")
    print(f"  Correlation: {results['get_aucell_correlation']:.6f}")
    
    # get_aucell_fast vs pySCENIC
    diff_fast = np.abs(flash_aucs_fast - pyscenic_aucs_arr)
    results['get_aucell_fast_max_diff'] = diff_fast.max()
    results['get_aucell_fast_mean_diff'] = diff_fast.mean()
    results['get_aucell_fast_correlation'] = np.corrcoef(flash_aucs_fast.flatten(), pyscenic_aucs_arr.flatten())[0, 1]
    print(f"\nget_aucell_fast vs pySCENIC:")
    print(f"  Max absolute difference: {results['get_aucell_fast_max_diff']:.6e}")
    print(f"  Mean absolute difference: {results['get_aucell_fast_mean_diff']:.6e}")
    print(f"  Correlation: {results['get_aucell_fast_correlation']:.6f}")

    # get_aucell_vectorized vs pySCENIC
    diff_fast = np.abs(flash_aucs_vec - pyscenic_aucs_arr)
    results['get_aucell_vectorized_max_diff'] = diff_fast.max()
    results['get_aucell_vectorized_mean_diff'] = diff_fast.mean()
    results['get_aucell_vectorized_correlation'] = np.corrcoef(flash_aucs_vec.flatten(), pyscenic_aucs_arr.flatten())[0, 1]
    print(f"\nget_aucell_vectorized vs pySCENIC:")
    print(f"  Max absolute difference: {results['get_aucell_vectorized_max_diff']:.6e}")
    print(f"  Mean absolute difference: {results['get_aucell_vectorized_mean_diff']:.6e}")
    print(f"  Correlation: {results['get_aucell_vectorized_correlation']:.6f}")
    
    # Check if results match
    results['all_close'] = np.allclose(flash_aucs, pyscenic_aucs_arr, rtol=1e-4, atol=1e-5)
    print(f"\nResults match within tolerance (rtol=1e-4, atol=1e-5): {results['all_close']}")
    
    # Timing summary
    print(f"\n{'='*60}")
    print("Timing Summary:")
    print(f"{'='*60}")
    print(f"\n{'Method':<30} {'Time (s)':<12} {'Speedup vs pySCENIC':<20}")
    print("-" * 62)
    
    pyscenic_time = results['timings']['pyscenic_total']
    for method in ['get_aucell', 'get_aucell_vectorized', 'get_aucell_fast']:
        t = results['timings'][method]
        speedup = pyscenic_time / t if t > 0 else float('inf')
        print(f"{method:<30} {t:<12.3f} {speedup:<20.2f}x")
    print(f"{'pySCENIC (CPU)':<30} {pyscenic_time:<12.3f} {'1.00':<20}x")
    
    return results


def test_edge_cases(device: str = 'cpu'):
    """Test edge cases like zero expression, sparse data, etc."""
    from flashscenic.aucell import get_aucell
    
    print(f"\n{'='*60}")
    print(f"Testing Edge Cases on {device}")
    print(f"{'='*60}")
    
    # Test with sparse data
    n_cells, n_genes, n_tfs = 50, 100, 5
    k = 10
    
    print("\n1. Sparse data test:")
    exp_array = np.random.randn(n_cells, n_genes).astype(np.float32)
    exp_array[exp_array < 0.5] = 0  # Make it sparse
    adj_array = np.random.rand(n_tfs, n_genes).astype(np.float32)
    
    aucs = get_aucell(exp_array, adj_array, k=k, device=device)
    print(f"   Shape: {aucs.shape}, Range: [{aucs.min():.4f}, {aucs.max():.4f}]")
    
    # Test with single cell
    print("\n2. Single cell test:")
    exp_single = exp_array[:1, :]
    aucs_single = get_aucell(exp_single, adj_array, k=k, device=device)
    print(f"   Shape: {aucs_single.shape}")
    
    # Test with single TF
    print("\n3. Single TF test:")
    adj_single = adj_array[:1, :]
    aucs_single_tf = get_aucell(exp_array, adj_single, k=k, device=device)
    print(f"   Shape: {aucs_single_tf.shape}")
    
    # Test with very small k
    print("\n4. Small k (k=3) test:")
    aucs_small_k = get_aucell(exp_array, adj_array, k=3, device=device)
    print(f"   Shape: {aucs_small_k.shape}")
    
    print("\n✓ All edge case tests passed!")


def benchmark_scaling(device: str = 'cuda', max_cells: int = 10000):
    """Benchmark how performance scales with data size."""
    from flashscenic.aucell import get_aucell, get_aucell_fast
    
    if device == 'cuda' and not CUDA_AVAILABLE:
        print("CUDA not available, skipping GPU scaling benchmark")
        return
    
    print(f"\n{'='*60}")
    print(f"Scaling Benchmark on {device}")
    print(f"{'='*60}")
    
    n_genes = 1000
    n_tfs = 20
    k = 50
    
    cell_sizes = [100, 500, 1000, 2000, 5000]
    cell_sizes = [c for c in cell_sizes if c <= max_cells]
    
    print(f"\nFixed: n_genes={n_genes}, n_tfs={n_tfs}, k={k}")
    print(f"{'n_cells':<10} {'get_aucell (s)':<18} {'get_aucell_fast (s)':<20} {'cells/sec':<15}")
    print("-" * 65)
    
    for n_cells in cell_sizes:
        exp_array = np.random.randn(n_cells, n_genes).astype(np.float32)
        exp_array = np.clip(exp_array, 0, None)
        adj_array = np.random.rand(n_tfs, n_genes).astype(np.float32)
        
        # Warm-up
        if device == 'cuda':
            _ = get_aucell(exp_array[:10], adj_array, k=k, device=device)
            import torch
            torch.cuda.synchronize()
        
        # Benchmark get_aucell
        t0 = time.time()
        _ = get_aucell(exp_array, adj_array, k=k, device=device)
        if device == 'cuda':
            import torch
            torch.cuda.synchronize()
        t_aucell = time.time() - t0
        
        # Benchmark get_aucell_fast
        t0 = time.time()
        _ = get_aucell_fast(exp_array, adj_array, k=k, device=device)
        if device == 'cuda':
            import torch
            torch.cuda.synchronize()
        t_fast = time.time() - t0
        
        cells_per_sec = n_cells / t_fast
        print(f"{n_cells:<10} {t_aucell:<18.4f} {t_fast:<20.4f} {cells_per_sec:<15.0f}")


def main():
    parser = argparse.ArgumentParser(
        description='Test flashscenic AUCell implementation against pySCENIC',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device to run tests on')
    parser.add_argument('--n_cells', type=int, default=100,
                        help='Number of cells/samples')
    parser.add_argument('--n_genes', type=int, default=500,
                        help='Number of genes/features')
    parser.add_argument('--n_tfs', type=int, default=10,
                        help='Number of transcription factors')
    parser.add_argument('--k', type=int, default=20,
                        help='Top k target genes per TF')
    parser.add_argument('--auc_threshold', type=float, default=0.05,
                        help='AUC threshold')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--skip_edge_cases', action='store_true',
                        help='Skip edge case tests')
    parser.add_argument('--benchmark_scaling', action='store_true',
                        help='Run scaling benchmark')
    parser.add_argument('--max_cells', type=int, default=10000,
                        help='Maximum cells for scaling benchmark')
    
    args = parser.parse_args()
    
    # Check device availability
    if args.device == 'cuda' and not CUDA_AVAILABLE:
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        args.device = 'cpu'
    
    print("=" * 60)
    print("AUCell Implementation Equivalence Test")
    print("=" * 60)
    print(f"\nSystem Info:")
    print(f"  CUDA Available: {CUDA_AVAILABLE}")
    if CUDA_AVAILABLE:
        print(f"  CUDA Device: {CUDA_DEVICE_NAME}")
    
    # Run main equivalence test
    results = test_aucell_equivalence(
        device=args.device,
        n_cells=args.n_cells,
        n_genes=args.n_genes,
        n_tfs=args.n_tfs,
        k=args.k,
        auc_threshold=args.auc_threshold,
        batch_size=args.batch_size,
        seed=args.seed
    )
    
    # Run edge case tests
    if not args.skip_edge_cases:
        test_edge_cases(device=args.device)
    
    # Run scaling benchmark
    if args.benchmark_scaling:
        benchmark_scaling(device=args.device, max_cells=args.max_cells)
    
    # Final summary
    print(f"\n{'='*60}")
    if results['all_close']:
        print("✓ All tests passed! Implementation matches pySCENIC.")
    else:
        print("✗ Some tests failed. Results may differ from pySCENIC.")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()
