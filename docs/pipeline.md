# Pipeline Guide

`run_flashscenic()` orchestrates five stages. This guide explains each stage and how to tune its parameters.

## Pipeline Overview

```
Expression Matrix (n_cells x n_genes)
    │
    ▼
Step 1: GRN Inference (RegDiffusion)     ← grn_*
    │
    ▼
Step 2: TF Filtering                     ← tf_list_path, grn_sparsity_threshold
    │
    ▼
Step 3: Module Filtering                  ← module_*
    │
    ▼
Step 4: cisTarget Pruning                 ← pruning_*, annotation_*
    │
    ▼
Step 5: AUCell Scoring                    ← aucell_*
    │
    ▼
Result dict (auc_scores, regulons, ...)
```

## Step 1: GRN Inference

Uses [RegDiffusion](https://github.com/TuftsBCB/RegDiffusion) to infer a gene-gene adjacency matrix from the expression data.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `grn_n_steps` | 1000 | Training iterations. More steps = better convergence. |
| `grn_sparsity_threshold` | 1.5 | Edges below this weight are zeroed. Higher = sparser. |

**Tuning tips:**
- Increase `grn_n_steps` to 2000+ for larger datasets
- Raise `grn_sparsity_threshold` if you get too many TF modules; lower it if too few survive

## Step 2: TF Filtering

Loads a known transcription factor list and subsets the adjacency matrix to rows corresponding to TFs.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tf_list_path` | Auto-downloaded | Path to TF gene list (one per line) |

## Step 3: Module Filtering

Selects top target genes per TF and filters out TFs with too few targets.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `module_k` | 50 | Top k target genes per TF |
| `module_min_targets` | 20 | Minimum absolute target count |
| `module_min_fraction` | 0.8 | Minimum fraction of targets required (pySCENIC 80% rule) |
| `module_include_tf` | True | Include TF in its own module |

**Tuning tips:**
- Increase `module_k` (e.g., 100) for broader modules
- Lower `module_min_targets` if few TFs survive filtering
- Set `module_min_fraction=None` to disable the fraction-based filter

## Step 4: cisTarget Pruning

Validates regulatory hypotheses against motif enrichment using ranking databases.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pruning_rank_threshold` | 5000 | Max rank for recovery curve |
| `pruning_auc_threshold` | 0.05 | Fraction of genome for AUC |
| `pruning_nes_threshold` | 3.0 | NES cutoff for enrichment |
| `pruning_min_genes` | 0 | Min genes per regulon |
| `pruning_merge_strategy` | 'union' | 'union' or 'best' across databases |
| `annotation_motif_similarity_fdr` | 0.001 | FDR threshold for motif annotations |
| `annotation_orthologous_identity` | 0.0 | Min orthologous identity |

**Tuning tips:**
- Lower `pruning_nes_threshold` (e.g., 2.5) if too few regulons survive
- Use `pruning_merge_strategy='best'` to keep only the highest-NES regulon per TF across databases

## Step 5: AUCell Scoring

Computes cell-state-specific regulatory activity scores.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `aucell_k` | None (uses module_k) | Top k targets for scoring |
| `aucell_auc_threshold` | 0.05 | Fraction of genome for AUC |
| `aucell_batch_size` | 32 | Cells per batch (memory vs speed) |

**Tuning tips:**
- Increase `aucell_batch_size` (64, 128) if GPU memory allows for faster processing
- `aucell_k` defaults to `module_k` but can be set independently

## Custom Resource Files

You can provide your own files instead of the auto-downloaded defaults:

```python
result = fs.run_flashscenic(
    exp_matrix, gene_names,
    tf_list_path='my_custom_tfs.txt',
    ranking_db_paths=['my_db1.feather', 'my_db2.feather'],
    motif_annotation_path='my_annotations.tbl',
)
```

When all three path arguments are provided, no download occurs.
