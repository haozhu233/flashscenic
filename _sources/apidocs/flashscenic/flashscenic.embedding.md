# {py:mod}`flashscenic.embedding`

```{py:module} flashscenic.embedding
```

```{autodoc2-docstring} flashscenic.embedding
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`gpu_knn <flashscenic.embedding.gpu_knn>`
  - ```{autodoc2-docstring} flashscenic.embedding.gpu_knn
    :summary:
    ```
* - {py:obj}`run_umap <flashscenic.embedding.run_umap>`
  - ```{autodoc2-docstring} flashscenic.embedding.run_umap
    :summary:
    ```
````

### API

````{py:function} gpu_knn(X, n_neighbors, device='cuda', batch_size=1024)
:canonical: flashscenic.embedding.gpu_knn

```{autodoc2-docstring} flashscenic.embedding.gpu_knn
```
````

````{py:function} run_umap(X, n_neighbors=15, n_components=2, min_dist=0.1, metric='euclidean', device='cuda', knn_batch_size=1024, random_state=42, **umap_kwargs)
:canonical: flashscenic.embedding.run_umap

```{autodoc2-docstring} flashscenic.embedding.run_umap
```
````
