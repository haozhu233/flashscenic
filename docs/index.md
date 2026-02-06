# flashscenic

**GPU-accelerated SCENIC workflow for gene regulatory network analysis.**

flashscenic replaces the bottleneck steps in the SCENIC pipeline with GPU-powered alternatives, achieving **seconds instead of hours** for large-scale single-cell analyses.

## Key Features

- **One-call pipeline**: `run_flashscenic()` handles GRN inference through AUCell scoring
- **Automatic data downloads**: TF lists, ranking databases, and motif annotations
- **GPU-accelerated**: All core operations use vectorized PyTorch
- **Scalable**: Handles 20,000 genes and millions of cells
- **Multi-species**: Human, mouse, and drosophila support

## Getting Started

```python
import flashscenic as fs

result = fs.run_flashscenic(exp_matrix, gene_names, species='human')
auc_scores = result['auc_scores']  # (n_cells, n_regulons)
```

```{toctree}
:maxdepth: 2
:caption: User Guide

installation
quickstart
pipeline
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
```
