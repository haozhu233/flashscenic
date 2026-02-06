# {py:mod}`flashscenic.pipeline`

```{py:module} flashscenic.pipeline
```

```{autodoc2-docstring} flashscenic.pipeline
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`run_flashscenic <flashscenic.pipeline.run_flashscenic>`
  - ```{autodoc2-docstring} flashscenic.pipeline.run_flashscenic
    :summary:
    ```
````

### API

````{py:function} run_flashscenic(exp_matrix: numpy.ndarray, gene_names: typing.List[str], species: str = 'human', *, datasource: str = 'scenic', version: str = 'v10', cache_dir: typing.Optional[str] = None, tf_list_path: typing.Optional[str] = None, ranking_db_paths: typing.Optional[typing.List[str]] = None, motif_annotation_path: typing.Optional[str] = None, grn_n_steps: int = 1000, grn_sparsity_threshold: float = 1.5, module_k: int = 50, module_min_targets: int = 20, module_min_fraction: float = 0.8, module_include_tf: bool = True, pruning_rank_threshold: int = 5000, pruning_auc_threshold: float = 0.05, pruning_nes_threshold: float = 3.0, pruning_min_genes: int = 0, pruning_merge_strategy: str = 'union', annotation_motif_similarity_fdr: float = 0.001, annotation_orthologous_identity: float = 0.0, aucell_k: typing.Optional[int] = None, aucell_auc_threshold: float = 0.05, aucell_batch_size: int = 32, device: str = 'cuda', seed: typing.Optional[int] = None, verbose: bool = True) -> typing.Dict
:canonical: flashscenic.pipeline.run_flashscenic

```{autodoc2-docstring} flashscenic.pipeline.run_flashscenic
```
````
