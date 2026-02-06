# {py:mod}`flashscenic.aucell`

```{py:module} flashscenic.aucell
```

```{autodoc2-docstring} flashscenic.aucell
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_aucell <flashscenic.aucell.get_aucell>`
  - ```{autodoc2-docstring} flashscenic.aucell.get_aucell
    :summary:
    ```
* - {py:obj}`_compute_auc <flashscenic.aucell._compute_auc>`
  - ```{autodoc2-docstring} flashscenic.aucell._compute_auc
    :summary:
    ```
````

### API

````{py:function} get_aucell(exp_array, adj_array, k=50, auc_threshold=0.05, device='cuda', batch_size=32, seed=None)
:canonical: flashscenic.aucell.get_aucell

```{autodoc2-docstring} flashscenic.aucell.get_aucell
```
````

````{py:function} _compute_auc(target_rankings, rank_cutoff, k, max_auc)
:canonical: flashscenic.aucell._compute_auc

```{autodoc2-docstring} flashscenic.aucell._compute_auc
```
````
