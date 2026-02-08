# Recommended Downstream Tasks

After running `run_flashscenic()`, you have a matrix of **regulon activity scores** (AUCell scores) for each cell. This matrix is a rich representation of regulatory state that enables many downstream analyses. This page walks through the most common and informative ones.

## Setup

All examples below assume you have already run the pipeline and have your results in an [AnnData](https://anndata.readthedocs.io/) object:

```python
import flashscenic as fs
import scanpy as sc
import numpy as np
import pandas as pd

# Run the pipeline
result = fs.run_flashscenic(exp_matrix, gene_names, species='human')

# Store AUCell scores in AnnData
auc_scores = result['auc_scores']        # (n_cells, n_regulons)
regulon_names = result['regulon_names']   # list of regulon labels

adata.obsm['X_aucell'] = auc_scores

# Also create a standalone AUCell AnnData for regulon-centric analyses
auc_adata = sc.AnnData(
    X=auc_scores,
    obs=adata.obs.copy(),
    var=pd.DataFrame(index=regulon_names),
)
```

## 1. Regulon-Based Clustering and Visualization

Clustering cells by their regulon activity—rather than gene expression—often reveals regulatory-state-based groupings that are more interpretable than expression-based clusters.

```python
# Compute neighbors and cluster on AUCell space
sc.pp.neighbors(adata, use_rep='X_aucell', n_neighbors=15)
sc.tl.leiden(adata, key_added='regulon_clusters')
sc.tl.umap(adata)

# Visualize
sc.pl.umap(adata, color='regulon_clusters')
```

You can compare these clusters with your expression-based clusters or known cell type annotations to see how regulatory state maps onto cell identity.

## 2. Binary Regulon Activity (On/Off)

Binarizing AUCell scores into "on" or "off" states per cell simplifies interpretation and is useful for heatmaps and specificity analyses. A common approach is to fit a bimodal distribution to each regulon's AUCell scores and set a threshold at the valley between modes.

```python
from scipy.stats import gaussian_kde

def binarize_aucell(scores, num_points=200):
    """Binarize AUCell scores per regulon using KDE-based thresholding."""
    binary = np.zeros_like(scores, dtype=int)
    for i in range(scores.shape[1]):
        col = scores[:, i]
        if col.std() == 0:
            continue
        kde = gaussian_kde(col)
        x = np.linspace(col.min(), col.max(), num_points)
        density = kde(x)
        # Find the first local minimum as the threshold
        for j in range(1, len(density) - 1):
            if density[j] < density[j - 1] and density[j] < density[j + 1]:
                binary[:, i] = (col > x[j]).astype(int)
                break
        else:
            # Fallback: use the median
            binary[:, i] = (col > np.median(col)).astype(int)
    return binary

binary_activity = binarize_aucell(auc_scores)
adata.obsm['X_aucell_binary'] = binary_activity
```

## 3. Regulon Specificity Score (RSS)

The Regulon Specificity Score measures how specific each regulon's activity is to a given cell type. Higher RSS means the regulon is more exclusively active in that cell type, making it useful for identifying master regulators.

```python
from scipy.stats import entropy

def regulon_specificity_score(auc_scores, cell_labels):
    """Compute RSS (Jensen-Shannon divergence based) for each regulon-cell type pair."""
    unique_labels = np.unique(cell_labels)
    n_regulons = auc_scores.shape[1]
    rss = np.zeros((len(unique_labels), n_regulons))

    for i, label in enumerate(unique_labels):
        mask = cell_labels == label
        # Cell type indicator distribution (uniform over cells of this type)
        cell_type_dist = mask.astype(float)
        cell_type_dist /= cell_type_dist.sum()

        for j in range(n_regulons):
            # Regulon activity distribution across cells
            regulon_dist = auc_scores[:, j].copy()
            regulon_dist = np.maximum(regulon_dist, 0)
            total = regulon_dist.sum()
            if total == 0:
                continue
            regulon_dist /= total

            # Jensen-Shannon divergence
            m = 0.5 * (cell_type_dist + regulon_dist)
            jsd = 0.5 * entropy(cell_type_dist, m) + 0.5 * entropy(regulon_dist, m)
            rss[i, j] = 1.0 - np.sqrt(jsd)

    return pd.DataFrame(rss, index=unique_labels, columns=regulon_names)

rss_df = regulon_specificity_score(auc_scores, adata.obs['cell_type'].values)

# Top 5 most specific regulons per cell type
for ct in rss_df.index:
    top = rss_df.loc[ct].nlargest(5)
    print(f"{ct}: {', '.join(top.index.tolist())}")
```

## 4. Differential Regulon Activity

Compare regulon activity between conditions (e.g., disease vs. control, treated vs. untreated) using standard differential testing on the AUCell score matrix.

```python
# Using scanpy's rank_genes_groups on the AUCell AnnData
sc.tl.rank_genes_groups(auc_adata, groupby='condition', method='wilcoxon')
sc.pl.rank_genes_groups(auc_adata, n_genes=10)

# Extract results as a DataFrame
diff_results = sc.get.rank_genes_groups_df(auc_adata, group='disease')
significant = diff_results[diff_results['pvals_adj'] < 0.05]
print(f"{len(significant)} differentially active regulons")
```

You can also perform this per cell type to find condition-specific regulatory changes within each population.

## 5. Regulon Activity Heatmap

Heatmaps of regulon activity across cell types give a quick overview of the regulatory landscape.

```python
# Mean AUCell score per cell type
mean_activity = auc_adata.to_df().groupby(adata.obs['cell_type'].values).mean()

# Plot heatmap of top variable regulons
variance = mean_activity.var(axis=0)
top_regulons = variance.nlargest(30).index.tolist()

import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(
    mean_activity[top_regulons].T,
    cmap='viridis',
    xticklabels=True,
    yticklabels=True,
    ax=ax,
)
ax.set_title('Regulon Activity by Cell Type')
plt.tight_layout()
plt.savefig('regulon_heatmap.png', dpi=150)
```

## 6. Regulon-Regulon Correlation and Co-Regulation Modules

Identifying groups of co-active regulons can reveal regulatory programs that work together.

```python
# Correlation matrix of regulon activity across cells
corr = np.corrcoef(auc_scores.T)
corr_df = pd.DataFrame(corr, index=regulon_names, columns=regulon_names)

# Connection Specificity Index (CSI)
# CSI normalizes correlations by how specific they are
def connection_specificity_index(corr_matrix):
    """Compute CSI from a correlation matrix."""
    n = corr_matrix.shape[0]
    csi = np.zeros_like(corr_matrix)
    for i in range(n):
        for j in range(n):
            if i == j:
                csi[i, j] = 1.0
                continue
            rho = corr_matrix[i, j]
            # Fraction of regulons with lower correlation to both i and j
            rank_i = (corr_matrix[i, :] < rho).sum() / n
            rank_j = (corr_matrix[j, :] < rho).sum() / n
            csi[i, j] = rank_i * rank_j
    return csi

csi = connection_specificity_index(corr)
csi_df = pd.DataFrame(csi, index=regulon_names, columns=regulon_names)

# Cluster the CSI matrix to find co-regulation modules
sns.clustermap(csi_df, cmap='RdBu_r', vmin=0, vmax=1, figsize=(10, 10))
plt.savefig('regulon_csi_clustermap.png', dpi=150)
```

## 7. GRN Visualization

Visualize the inferred gene regulatory network to explore TF-target relationships.

```python
import networkx as nx

regulons = result['regulons']

# Build a network from regulon data
G = nx.DiGraph()
for reg in regulons:
    tf = reg['tf']
    for gene in reg['genes'][:10]:  # top 10 targets for readability
        G.add_edge(tf, gene)

# Draw
pos = nx.spring_layout(G, k=2, seed=42)
tf_nodes = [reg['tf'] for reg in regulons]
target_nodes = [n for n in G.nodes if n not in tf_nodes]

fig, ax = plt.subplots(figsize=(14, 10))
nx.draw_networkx_nodes(G, pos, nodelist=tf_nodes, node_color='tomato',
                       node_size=300, ax=ax, label='TFs')
nx.draw_networkx_nodes(G, pos, nodelist=target_nodes, node_color='lightblue',
                       node_size=100, ax=ax, label='Targets')
nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=6, ax=ax)
ax.legend(scatterpoints=1)
ax.set_title('Gene Regulatory Network')
plt.tight_layout()
plt.savefig('grn_network.png', dpi=150)
```

## 8. Cell Type Annotation via Regulon Activity

Known master regulators can be used to annotate or validate cell types. For example, PAX6 regulon activity should be high in certain neural progenitors.

```python
# Check activity of known marker regulons across clusters
marker_regulons = ['PAX6(+)', 'SOX2(+)', 'GATA1(+)', 'SPI1(+)']  # adjust to your data
available = [r for r in marker_regulons if r in regulon_names]

for reg in available:
    idx = regulon_names.index(reg)
    adata.obs[reg] = auc_scores[:, idx]

sc.pl.umap(adata, color=available, ncols=2)
```

## 9. Export for Other Tools

### Export to loom format (for SCope or SCENIC+)

```python
import loompy

# Create a loom file with regulon data
row_attrs = {'Gene': gene_names}
col_attrs = {
    'CellID': adata.obs_names.tolist(),
    'cell_type': adata.obs['cell_type'].values.astype(str),
}

# Add regulon AUCell scores as column attributes
for i, name in enumerate(regulon_names):
    col_attrs[f'RegulonsAUC_{name}'] = auc_scores[:, i]

loompy.create('flashscenic_output.loom', exp_matrix.T, row_attrs, col_attrs)
```

### Export regulons to GMT format (for GSEA or Enrichr)

```python
with open('regulons.gmt', 'w') as f:
    for reg in result['regulons']:
        genes = '\t'.join(reg['genes'])
        f.write(f"{reg['tf']}\tflashscenic_regulon\t{genes}\n")
```

## Summary

| Task | Purpose | Key Output |
|------|---------|------------|
| Regulon-based clustering | Group cells by regulatory state | Cluster labels, UMAP |
| Binary activity | Simplify on/off interpretation | Binary matrix |
| Regulon specificity (RSS) | Find cell-type-specific regulons | RSS score table |
| Differential activity | Compare conditions | Ranked regulon list |
| Activity heatmap | Overview of regulatory landscape | Heatmap figure |
| Regulon correlation / CSI | Find co-regulation programs | CSI matrix, modules |
| GRN visualization | Explore TF-target structure | Network plot |
| Cell type annotation | Validate/annotate clusters | Marker overlays |
| Export (loom, GMT) | Interoperate with other tools | Portable files |
