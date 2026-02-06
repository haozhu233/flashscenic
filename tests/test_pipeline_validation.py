"""
Validation tests for flashscenic implementations against pySCENIC/ctxcore reference.

This test file validates:
1. AUCell scoring equivalence
2. cisTarget pruning (recovery curves, NES, leading edge)
3. Module creation utilities
4. End-to-end pipeline
"""

import numpy as np
import pandas as pd
import torch
import pytest
from typing import List, Dict
import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
# Add local ctxcore (has aucell module, installed version may not)
sys.path.insert(0, os.path.join(parent_dir, 'ctxcore', 'src'))


# ============================================================================
# Test fixtures
# ============================================================================

@pytest.fixture
def small_expression_matrix():
    """Small expression matrix for quick tests."""
    np.random.seed(42)
    n_cells = 100
    n_genes = 500
    return np.random.rand(n_cells, n_genes).astype(np.float32)


@pytest.fixture
def gene_names():
    """Gene names for test data."""
    return [f"gene_{i}" for i in range(500)]


@pytest.fixture
def tf_names():
    """TF names for test data."""
    return [f"TF_{i}" for i in range(50)]


@pytest.fixture
def adjacency_matrix():
    """Adjacency matrix (TFs x genes)."""
    np.random.seed(42)
    n_tfs = 50
    n_genes = 500
    # Sparse adjacency - most values near zero
    adj = np.abs(np.random.randn(n_tfs, n_genes).astype(np.float32))
    # Sparsify
    adj[adj < 1.5] = 0
    return adj


@pytest.fixture
def synthetic_ranking_db():
    """Synthetic ranking database for cisTarget tests."""
    np.random.seed(42)
    n_motifs = 100
    n_genes = 500
    # Create random rankings (each row is a permutation of 0 to n_genes-1)
    rankings = np.zeros((n_motifs, n_genes), dtype=np.int32)
    for i in range(n_motifs):
        rankings[i] = np.random.permutation(n_genes)
    return rankings


# ============================================================================
# AUCell Validation Tests
# ============================================================================

class TestAUCellValidation:
    """Test AUCell implementation against ctxcore reference."""

    def test_ranking_computation_equivalence(self, small_expression_matrix, gene_names):
        """Test that ranking computation produces valid results.

        Note: ctxcore and flashscenic use different tie-breaking methods:
        - ctxcore: shuffles columns before ranking (random column order)
        - flashscenic: adds noise for tie-breaking

        Both are valid approaches but produce different results for ties.
        We verify that both produce valid rankings (0 to n-1).
        """
        from ctxcore.aucell import create_rankings

        # Create expression DataFrame
        exp_df = pd.DataFrame(
            small_expression_matrix,
            columns=gene_names,
            index=[f"cell_{i}" for i in range(small_expression_matrix.shape[0])]
        )

        # ctxcore ranking
        ctxcore_rankings = create_rankings(exp_df, seed=42)

        # flashscenic ranking (using same approach as in get_aucell)
        torch.manual_seed(42)
        exp_tensor = torch.tensor(small_expression_matrix)
        noise = torch.rand_like(exp_tensor) * 1e-10
        exp_noisy = exp_tensor + noise
        order = torch.argsort(-exp_noisy, dim=1)
        flash_rankings = torch.argsort(order, dim=1).numpy()

        n_genes = small_expression_matrix.shape[1]

        # Verify both produce valid rankings
        for cell_idx in range(min(10, small_expression_matrix.shape[0])):
            ctxcore_rank = ctxcore_rankings.iloc[cell_idx].values
            flash_rank = flash_rankings[cell_idx]

            # Check valid range
            assert ctxcore_rank.min() == 0, "ctxcore ranking should start at 0"
            assert ctxcore_rank.max() == n_genes - 1, "ctxcore ranking should end at n-1"
            assert flash_rank.min() == 0, "flashscenic ranking should start at 0"
            assert flash_rank.max() == n_genes - 1, "flashscenic ranking should end at n-1"

            # Check all values are unique (proper ranking)
            assert len(np.unique(ctxcore_rank)) == n_genes, "ctxcore ranking should have unique values"
            assert len(np.unique(flash_rank)) == n_genes, "flashscenic ranking should have unique values"

            # Verify high expression genes get low ranks (0 = highest)
            # For non-tied genes, the relative order should be preserved
            exp_values = small_expression_matrix[cell_idx]
            highest_exp_idx = np.argmax(exp_values)
            highest_exp_gene = gene_names[highest_exp_idx]

            # ctxcore returns DataFrame with shuffled column order - use loc properly
            ctxcore_rank_for_highest = ctxcore_rankings.iloc[cell_idx].loc[highest_exp_gene]
            # flashscenic ranking is in original order
            flash_rank_for_highest = flash_rank[highest_exp_idx]

            # The highest expressed gene should have rank 0 in both
            assert ctxcore_rank_for_highest == 0, f"ctxcore: Highest expression should have rank 0, got {ctxcore_rank_for_highest}"
            assert flash_rank_for_highest == 0, f"flashscenic: Highest expression should have rank 0, got {flash_rank_for_highest}"

    def test_auc_formula_equivalence(self):
        """Test AUC calculation formula matches ctxcore weighted_auc1d."""
        from ctxcore.recovery import weighted_auc1d

        # Create test case
        np.random.seed(42)
        n_genes = 100
        ranking = np.random.permutation(n_genes).astype(np.int64)
        weights = np.ones(n_genes)
        rank_cutoff = 5  # Small cutoff for easier verification
        max_auc = float((rank_cutoff + 1) * weights.sum())

        # ctxcore AUC
        ctxcore_auc = weighted_auc1d(ranking, weights, rank_cutoff, max_auc)

        # Manual calculation (same as flashscenic logic)
        filter_mask = ranking < rank_cutoff
        filtered_ranks = ranking[filter_mask]
        filtered_weights = weights[filter_mask]

        sort_idx = np.argsort(filtered_ranks)
        sorted_ranks = np.concatenate([filtered_ranks[sort_idx], [rank_cutoff]])
        cumsum_weights = filtered_weights[sort_idx].cumsum()

        rank_diffs = np.diff(sorted_ranks)
        manual_auc = (rank_diffs * cumsum_weights).sum() / max_auc

        np.testing.assert_allclose(ctxcore_auc, manual_auc, rtol=1e-5)

    def test_get_aucell_basic(self, small_expression_matrix, adjacency_matrix):
        """Test basic get_aucell functionality."""
        from flashscenic import get_aucell

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Run flashscenic AUCell
        result = get_aucell(
            small_expression_matrix,
            adjacency_matrix,
            k=50,
            auc_threshold=0.05,
            device=device,
            batch_size=32
        )

        # Check output shape
        n_cells, n_genes = small_expression_matrix.shape
        n_tfs = adjacency_matrix.shape[0]
        assert result.shape == (n_cells, n_tfs), f"Expected ({n_cells}, {n_tfs}), got {result.shape}"

        # Check values are in valid range [0, 1]
        assert result.min() >= 0, f"AUC values should be >= 0, got min {result.min()}"
        assert result.max() <= 1, f"AUC values should be <= 1, got max {result.max()}"

        # Check for NaN/Inf
        assert not np.isnan(result).any(), "Result contains NaN"
        assert not np.isinf(result).any(), "Result contains Inf"

    def test_aucell_vs_ctxcore_genesig(self, small_expression_matrix, gene_names):
        """Compare flashscenic AUCell against ctxcore using GeneSignatures.

        Note: Due to different tie-breaking methods (ctxcore shuffles columns,
        flashscenic adds noise), results will differ. We verify:
        1. Both produce valid AUC scores in [0, 1]
        2. Both show similar variance patterns (some cells score high, some low)
        3. The mean scores are in the same ballpark
        """
        from flashscenic import get_aucell
        from ctxcore.aucell import aucell as ctxcore_aucell
        from ctxcore.genesig import GeneSignature

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Create simple modules with known genes
        np.random.seed(42)
        k = 50
        n_tfs = 5

        # Create adjacency matrix where each TF has exactly k non-zero targets
        n_genes = len(gene_names)
        adj_matrix = np.zeros((n_tfs, n_genes), dtype=np.float32)
        gene_signatures = []

        for tf_idx in range(n_tfs):
            # Select k random genes
            target_indices = np.random.choice(n_genes, k, replace=False)
            adj_matrix[tf_idx, target_indices] = 1.0

            # Create GeneSignature with same genes
            target_genes = [gene_names[i] for i in target_indices]
            gene_signatures.append(GeneSignature(
                name=f"TF_{tf_idx}",
                gene2weight=target_genes  # Equal weights
            ))

        # Create expression DataFrame
        exp_df = pd.DataFrame(
            small_expression_matrix,
            columns=gene_names,
            index=[f"cell_{i}" for i in range(small_expression_matrix.shape[0])]
        )

        # flashscenic AUCell
        flash_result = get_aucell(
            small_expression_matrix,
            adj_matrix,
            k=k,
            auc_threshold=0.05,
            device=device,
            seed=42
        )

        # ctxcore AUCell
        ctxcore_result = ctxcore_aucell(
            exp_df,
            gene_signatures,
            auc_threshold=0.05,
            noweights=True,  # Equal weights
            seed=42,
            num_workers=1
        )

        ctxcore_values = ctxcore_result.values

        # Verify both have same shape
        assert flash_result.shape == ctxcore_values.shape, "Shape mismatch"

        # Verify both are in valid range
        assert flash_result.min() >= 0, "flashscenic AUC < 0"
        assert flash_result.max() <= 1, "flashscenic AUC > 1"
        assert ctxcore_values.min() >= 0, "ctxcore AUC < 0"
        assert ctxcore_values.max() <= 1, "ctxcore AUC > 1"

        # Verify mean scores are similar (within 20% of each other)
        for tf_idx in range(n_tfs):
            flash_mean = flash_result[:, tf_idx].mean()
            ctxcore_mean = ctxcore_values[:, tf_idx].mean()
            # Allow for reasonable variation due to tie-breaking
            assert abs(flash_mean - ctxcore_mean) < 0.1, \
                f"TF {tf_idx}: mean difference too large ({flash_mean:.3f} vs {ctxcore_mean:.3f})"

        # Verify variance is non-zero (cells have different scores)
        assert flash_result.var() > 0, "flashscenic variance is 0"
        assert ctxcore_values.var() > 0, "ctxcore variance is 0"


# ============================================================================
# cisTarget Validation Tests
# ============================================================================

class TestCisTargetValidation:
    """Test cisTarget pruning implementation."""

    def test_recovery_curve_computation(self, synthetic_ranking_db, gene_names):
        """Test recovery curve computation against ctxcore."""
        from flashscenic.cistarget import compute_recovery_aucs
        from ctxcore.recovery import recovery

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Select a subset of genes as "module genes"
        np.random.seed(42)
        n_module_genes = 30
        module_gene_indices = np.sort(np.random.choice(len(gene_names), n_module_genes, replace=False))

        weights = np.ones(n_module_genes)
        # rank_threshold must be < total_genes for ctxcore
        rank_threshold = min(1500, len(gene_names) - 1)
        auc_threshold = 0.05

        # flashscenic recovery (new tensor-native API)
        rankings_tensor = torch.tensor(synthetic_ranking_db, device=device, dtype=torch.int32)
        indices_tensor = torch.tensor(module_gene_indices, device=device, dtype=torch.long)
        weights_tensor = torch.tensor(weights, device=device, dtype=torch.float32)

        flash_rccs, flash_aucs = compute_recovery_aucs(
            rankings_tensor,
            indices_tensor,
            rank_threshold,
            auc_threshold,
            weights_tensor
        )
        flash_rccs = flash_rccs.cpu().numpy()
        flash_aucs = flash_aucs.cpu().numpy()

        # ctxcore recovery (for first few motifs to compare)
        n_test_motifs = 5
        for motif_idx in range(n_test_motifs):
            # Create DataFrame for ctxcore
            rnk_df = pd.DataFrame(
                synthetic_ranking_db[motif_idx:motif_idx+1, module_gene_indices],
                index=[f"motif_{motif_idx}"],
                columns=[gene_names[i] for i in module_gene_indices]
            )

            ctx_rccs, ctx_aucs = recovery(
                rnk_df,
                total_genes=len(gene_names),
                weights=weights,
                rank_threshold=rank_threshold,
                auc_threshold=auc_threshold
            )

            # Compare AUCs
            np.testing.assert_allclose(
                flash_aucs[motif_idx],
                ctx_aucs[0],
                rtol=0.01,
                err_msg=f"AUC mismatch for motif {motif_idx}"
            )

    def test_nes_computation(self):
        """Test NES calculation."""
        from flashscenic.cistarget import compute_nes

        # Test case
        aucs = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        nes = compute_nes(aucs)

        # Manual calculation
        mean_auc = aucs.mean()
        std_auc = aucs.std(unbiased=False)
        expected_nes = (aucs - mean_auc) / std_auc

        np.testing.assert_allclose(nes.numpy(), expected_nes.numpy(), rtol=1e-5)

        # Verify NES has mean 0 and std 1
        np.testing.assert_allclose(nes.mean().item(), 0, atol=1e-5)
        np.testing.assert_allclose(nes.std(unbiased=False).item(), 1, atol=1e-5)

    def test_prune_module_basic(self, synthetic_ranking_db, gene_names):
        """Test basic module pruning functionality (tensor-native API)."""
        from flashscenic.cistarget import prune_single_module

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Create module genes
        np.random.seed(42)
        module_gene_indices = np.sort(np.random.choice(len(gene_names), 30, replace=False))

        # Convert to tensors
        rankings_tensor = torch.tensor(synthetic_ranking_db, device=device, dtype=torch.int32)
        indices_tensor = torch.tensor(module_gene_indices, device=device, dtype=torch.long)

        result = prune_single_module(
            rankings_tensor,
            indices_tensor,
            rank_threshold=min(1500, len(gene_names) - 1),
            auc_threshold=0.05,
            nes_threshold=3.0
        )

        # Check result structure (all tensors now)
        assert 'enriched_mask' in result
        assert 'nes' in result
        assert 'aucs' in result
        assert 'rccs' in result
        assert 'leading_edge_masks' in result
        assert 'rank_at_max' in result

        # Check types
        assert isinstance(result['enriched_mask'], torch.Tensor)
        assert isinstance(result['nes'], torch.Tensor)
        assert isinstance(result['aucs'], torch.Tensor)

        # Check shapes
        n_motifs = synthetic_ranking_db.shape[0]
        assert result['enriched_mask'].shape == (n_motifs,)
        assert result['nes'].shape == (n_motifs,)
        assert result['aucs'].shape == (n_motifs,)

        # If there are enriched motifs, check consistency
        n_enriched = result['enriched_mask'].sum().item()
        if n_enriched > 0:
            assert result['leading_edge_masks'].shape[0] == n_enriched
            assert result['rank_at_max'].shape[0] == n_enriched

            # All NES scores for enriched motifs should be >= threshold
            enriched_nes = result['nes'][result['enriched_mask']]
            assert (enriched_nes >= 3.0).all()


# ============================================================================
# Module Creation Tests (Tensor-Native API)
# ============================================================================

class TestModuleCreation:
    """Test module creation utilities (tensor-native)."""

    def test_select_topk_targets(self, adjacency_matrix, gene_names, tf_names):
        """Test top-k target selection."""
        from flashscenic.modules import select_topk_targets

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Make adjacency matrix match tf_names/gene_names dimensions
        adj = adjacency_matrix[:len(tf_names), :len(gene_names)]
        k = 50

        result = select_topk_targets(adj, k=k, device=device)

        # Check output is tensor
        assert isinstance(result, torch.Tensor)
        assert result.shape == adj.shape

        # Check each row has at most k non-zero values
        for i in range(result.shape[0]):
            n_nonzero = (result[i] > 0).sum().item()
            assert n_nonzero <= k, f"Row {i} has {n_nonzero} non-zero values, expected <= {k}"

        # Check that top-k values are preserved
        adj_tensor = torch.tensor(adj, device=device, dtype=torch.float32)
        for i in range(min(5, result.shape[0])):  # Check first 5 rows
            original_row = adj_tensor[i]
            result_row = result[i]
            # Get top-k indices from original
            topk_vals, topk_idx = torch.topk(original_row, min(k, (original_row > 0).sum().item()))
            # Verify those values are in result
            for idx, val in zip(topk_idx, topk_vals):
                if val > 0:
                    assert result_row[idx] == val, f"Value mismatch at row {i}, col {idx}"

    def test_select_threshold_targets(self, adjacency_matrix, gene_names, tf_names):
        """Test threshold-based target selection."""
        from flashscenic.modules import select_threshold_targets

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        adj = adjacency_matrix[:len(tf_names), :len(gene_names)]
        threshold = 1.8

        result = select_threshold_targets(adj, threshold=threshold, device=device)

        # Check output is tensor
        assert isinstance(result, torch.Tensor)
        assert result.shape == adj.shape

        # Check all non-zero values are >= threshold
        nonzero_mask = result > 0
        nonzero_values = result[nonzero_mask]
        assert (nonzero_values >= threshold).all(), "Some values below threshold"

    def test_filter_by_min_targets(self, adjacency_matrix, gene_names, tf_names):
        """Test filtering by minimum targets."""
        from flashscenic.modules import filter_by_min_targets, select_topk_targets

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # First create modules with top-k selection
        adj = adjacency_matrix[:len(tf_names), :len(gene_names)]
        filtered_adj = select_topk_targets(adj, k=50, device=device)

        # Now filter by min targets
        min_targets = 20
        result, mask = filter_by_min_targets(filtered_adj, min_targets=min_targets, device=device)

        # Check all remaining TFs have enough targets
        target_counts = (result > 0).sum(dim=1)
        assert (target_counts >= min_targets).all(), "Some TFs have too few targets"

        # Check mask is consistent
        assert mask.sum().item() == result.shape[0]

    def test_get_target_indices(self, adjacency_matrix, gene_names, tf_names):
        """Test getting target indices."""
        from flashscenic.modules import get_target_indices, select_topk_targets

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        adj = adjacency_matrix[:len(tf_names), :len(gene_names)]
        filtered_adj = select_topk_targets(adj, k=30, device=device)

        flat_indices, positions = get_target_indices(filtered_adj, device=device)

        # Check positions array
        n_tfs = filtered_adj.shape[0]
        assert positions.shape == (n_tfs + 1,)
        assert positions[0] == 0

        # Check indices are valid
        n_genes = filtered_adj.shape[1]
        assert (flat_indices >= 0).all()
        assert (flat_indices < n_genes).all()

        # Verify consistency
        for i in range(n_tfs):
            start, end = positions[i].item(), positions[i + 1].item()
            n_targets = (filtered_adj[i] > 0).sum().item()
            assert end - start == n_targets, f"TF {i}: position range {end-start} != target count {n_targets}"


# ============================================================================
# End-to-End Pipeline Test (Tensor-Native)
# ============================================================================

class TestEndToEndPipeline:
    """Test complete flashscenic pipeline (tensor-native)."""

    def test_full_pipeline_synthetic(self, small_expression_matrix, gene_names, adjacency_matrix):
        """Test full pipeline with synthetic data."""
        from flashscenic.modules import (
            select_topk_targets,
            filter_by_min_targets,
            binarize
        )
        from flashscenic import get_aucell

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Step 1: Select top-k targets from adjacency matrix
        adj = adjacency_matrix[:, :len(gene_names)]
        filtered_adj = select_topk_targets(adj, k=50, device=device)

        # Step 2: Filter TFs with too few targets
        filtered_adj, mask = filter_by_min_targets(filtered_adj, min_targets=20, device=device)
        n_valid_tfs = mask.sum().item()

        print(f"TFs with >= 20 targets: {n_valid_tfs}")

        # Step 3: Skip cisTarget pruning (requires real database)
        # In real usage:
        #   pruner = CisTargetPruner(device=device)
        #   pruner.load_database('rankings.feather')
        #   for tf_idx in range(n_valid_tfs):
        #       indices = get_target_indices(filtered_adj[tf_idx:tf_idx+1])
        #       result = pruner.prune(indices)

        # Step 4: Compute AUCell scores (using filtered adjacency as "regulons")
        if n_valid_tfs > 0:
            # Convert to numpy for get_aucell
            regulon_adj = filtered_adj.cpu().numpy()

            auc_scores = get_aucell(
                small_expression_matrix,
                regulon_adj,
                k=50,
                device=device
            )

            # Verify output
            assert auc_scores.shape == (small_expression_matrix.shape[0], n_valid_tfs)
            assert not np.isnan(auc_scores).any()

            print(f"AUCell scores shape: {auc_scores.shape}")
            print(f"AUCell scores range: [{auc_scores.min():.4f}, {auc_scores.max():.4f}]")


# ============================================================================
# Performance Benchmarks
# ============================================================================

class TestPerformance:
    """Performance benchmarks."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_aucell_gpu_performance(self):
        """Benchmark AUCell GPU vs CPU."""
        from flashscenic import get_aucell
        import time

        np.random.seed(42)
        n_cells = 1000
        n_genes = 5000
        n_tfs = 100

        exp_matrix = np.random.rand(n_cells, n_genes).astype(np.float32)
        adj_matrix = np.random.rand(n_tfs, n_genes).astype(np.float32)

        # GPU timing
        start = time.time()
        gpu_result = get_aucell(exp_matrix, adj_matrix, device='cuda')
        gpu_time = time.time() - start

        # CPU timing
        start = time.time()
        cpu_result = get_aucell(exp_matrix, adj_matrix, device='cpu')
        cpu_time = time.time() - start

        print(f"\nPerformance ({n_cells} cells, {n_genes} genes, {n_tfs} TFs):")
        print(f"  GPU: {gpu_time:.3f}s")
        print(f"  CPU: {cpu_time:.3f}s")
        print(f"  Speedup: {cpu_time/gpu_time:.2f}x")

        # Results should be similar (allow small differences from float precision)
        np.testing.assert_allclose(gpu_result, cpu_result, rtol=1e-3, atol=1e-4)


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
