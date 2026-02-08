# Quick Start

## One-Call Pipeline

The simplest way to use flashscenic is the `run_flashscenic()` function, which runs the entire pipeline from expression matrix to AUCell scores:

```python
import flashscenic as fs
import numpy as np

# Load your expression data (n_cells x n_genes), log-transformed
# For example, from an h5ad file using scanpy:
#   import scanpy as sc
#   adata = sc.read_h5ad('data.h5ad')
#   sc.pp.log1p(adata)
#   sc.pp.highly_variable_genes(adata, n_top_genes=8000)
#   adata = adata[:, adata.var['highly_variable']].copy()
#   exp_matrix = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
#   gene_names = adata.var_names.tolist()

result = fs.run_flashscenic(exp_matrix, gene_names, species='human')
```

The function automatically downloads required resource files (TF lists, ranking databases, motif annotations) on first run.

## Working with Results

```python
# AUCell scores: regulatory activity per cell
auc_scores = result['auc_scores']       # (n_cells, n_regulons)
regulon_names = result['regulon_names']  # list of regulon labels

# Full regulon information
regulons = result['regulons']            # list of dicts with gene members
regulon_adj = result['regulon_adj']      # (n_regulons, n_genes) adjacency

# Parameters used
params = result['parameters']
print(f"Found {params['n_regulons']} regulons from {params['n_tfs']} TFs")
```

## Downstream Analysis

After obtaining AUCell scores, you can use them for dimensionality reduction and visualization:

```python
import scanpy as sc

# Store AUCell scores in AnnData
adata.obsm['X_aucell'] = auc_scores

# Compute neighbors and UMAP on AUCell space
sc.pp.neighbors(adata, use_rep='X_aucell', n_neighbors=15)
sc.tl.umap(adata)
sc.pl.umap(adata, color='cell_type')
```

For a comprehensive guide to downstream analyses—including binary regulon activity, regulon specificity scoring, differential regulon activity, co-regulation modules, GRN visualization, and export to other tools—see the [Recommended Downstream Tasks](downstream.md) page.

## Pre-Downloading Data

To download resource files ahead of time (useful for cluster environments without internet):

```python
resources = fs.download_data(species='human', version='v10')

# Then use the local files
result = fs.run_flashscenic(
    exp_matrix, gene_names,
    tf_list_path=str(resources.tf_list),
    ranking_db_paths=[str(p) for p in resources.ranking_dbs],
    motif_annotation_path=str(resources.motif_annotation),
)
```
