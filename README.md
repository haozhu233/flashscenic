# flashscenic

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**GPU-accelerated SCENIC workflow for gene regulatory network analysis. Seconds instead of hours.**

flashscenic replaces the bottleneck steps in the [SCENIC](https://scenic.aertslab.org/) pipeline with GPU-powered alternatives: [RegDiffusion](https://github.com/TuftsBCB/RegDiffusion) for GRN inference and vectorized PyTorch implementations of AUCell and cisTarget. The result is a complete GRN analysis pipeline that scales to 20,000 genes and millions of cells, running in seconds on a single GPU.

## Installation

```bash
pip install flashscenic
```

For documentation development:
```bash
pip install flashscenic[docs]
```

**Requirements:** Python 3.9+, PyTorch with CUDA support (CPU fallback available).

## Quick Start

```python
import flashscenic as fs

# Run the full pipeline in one call
# exp_matrix: (n_cells, n_genes) log-transformed numpy array
# gene_names: list of gene symbols matching columns
result = fs.run_flashscenic(exp_matrix, gene_names, species='human')

# Results
auc_scores = result['auc_scores']       # (n_cells, n_regulons)
regulon_names = result['regulon_names']  # regulon labels
```

Required resource files (TF lists, ranking databases, motif annotations) are downloaded automatically on first run.

## Downloading Data

flashscenic can automatically download the cistarget resource files needed for motif-based pruning:

```python
import flashscenic as fs

# Download human v10 resources (default)
resources = fs.download_data(species='human', version='v10')
print(resources)

# Download mouse resources
resources = fs.download_data(species='mouse')

# List all available resource sets
for rs in fs.list_available_resources():
    print(f"{rs.datasource}/{rs.species}/{rs.version}")
```

Files are cached in `./flashscenic_data/` by default and skipped on subsequent calls.

### Supported species and versions

| Species | Version | Source |
|---------|---------|--------|
| human | v10 (recommended), v9 | Aertslab |
| mouse | v10, v9 | Aertslab |
| drosophila | v10 | Aertslab |

## Step-by-Step Usage

For more control, you can run each pipeline step individually:

```python
import numpy as np
import torch
import flashscenic as fs

# 1. GRN Inference (using RegDiffusion separately)
import regdiffusion as rd
trainer = rd.RegDiffusionTrainer(exp_matrix)
trainer.train()
adj_matrix = trainer.get_adj()

# 2. Filter to known TFs and sparsify
# (load your TF list, subset adj_matrix rows, zero out weak edges)

# 3. Module filtering
filtered_adj = fs.select_topk_targets(adj_matrix, k=50, device='cuda')
filtered_adj, tf_mask = fs.filter_by_min_targets(
    filtered_adj, min_targets=20, min_fraction=0.8
)

# 4. cisTarget pruning
pruner = fs.CisTargetPruner(device='cuda')
pruner.load_database(['db_500bp.feather', 'db_10kb.feather'])
pruner.load_annotations('motifs.tbl', filter_for_annotation=True)
regulons = pruner.prune_modules(modules, tf_names, gene_names)

# 5. AUCell scoring
regulon_adj = fs.regulons_to_adjacency(regulons, gene_names)
auc_scores = fs.get_aucell(exp_matrix, regulon_adj, k=50, device='cuda')
```

## Pipeline Parameters

`run_flashscenic` exposes all tunable parameters with stage-based prefixes:

| Prefix | Stage | Key Parameters |
|--------|-------|----------------|
| `grn_` | RegDiffusion | `grn_n_steps`, `grn_sparsity_threshold` |
| `module_` | Module filtering | `module_k`, `module_min_targets`, `module_min_fraction` |
| `pruning_` | cisTarget | `pruning_rank_threshold`, `pruning_nes_threshold`, `pruning_merge_strategy` |
| `annotation_` | Motif filtering | `annotation_motif_similarity_fdr`, `annotation_orthologous_identity` |
| `aucell_` | AUCell scoring | `aucell_k`, `aucell_auc_threshold`, `aucell_batch_size` |

Example with custom parameters:

```python
result = fs.run_flashscenic(
    exp_matrix, gene_names,
    species='mouse',
    module_k=100,
    module_min_targets=10,
    module_min_fraction=None,  # disable fraction filter
    pruning_nes_threshold=2.5,
    device='cpu',
)
```

## Core API

| Function / Class | Description |
|-----------------|-------------|
| `run_flashscenic()` | Full pipeline in one call |
| `download_data()` | Download cistarget resource files |
| `get_aucell()` | GPU-accelerated AUCell scoring |
| `CisTargetPruner` | GPU cisTarget motif pruning |
| `select_topk_targets()` | Top-k module filtering |
| `filter_by_min_targets()` | Min-target module filtering |
| `regulons_to_adjacency()` | Convert regulons to adjacency matrix |

## Authors

- Hao Zhu (haozhu233@gmail.com)
- Donna Slonim (donna.slonim@tufts.edu)

## License

MIT License. See [LICENSE](LICENSE) for details.
