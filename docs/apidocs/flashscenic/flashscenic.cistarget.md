# {py:mod}`flashscenic.cistarget`

```{py:module} flashscenic.cistarget
```

```{autodoc2-docstring} flashscenic.cistarget
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MotifAnnotation <flashscenic.cistarget.MotifAnnotation>`
  - ```{autodoc2-docstring} flashscenic.cistarget.MotifAnnotation
    :summary:
    ```
* - {py:obj}`CisTargetPruner <flashscenic.cistarget.CisTargetPruner>`
  - ```{autodoc2-docstring} flashscenic.cistarget.CisTargetPruner
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`compute_recovery_aucs <flashscenic.cistarget.compute_recovery_aucs>`
  - ```{autodoc2-docstring} flashscenic.cistarget.compute_recovery_aucs
    :summary:
    ```
* - {py:obj}`compute_nes <flashscenic.cistarget.compute_nes>`
  - ```{autodoc2-docstring} flashscenic.cistarget.compute_nes
    :summary:
    ```
* - {py:obj}`filter_by_annotations <flashscenic.cistarget.filter_by_annotations>`
  - ```{autodoc2-docstring} flashscenic.cistarget.filter_by_annotations
    :summary:
    ```
* - {py:obj}`compute_leading_edge <flashscenic.cistarget.compute_leading_edge>`
  - ```{autodoc2-docstring} flashscenic.cistarget.compute_leading_edge
    :summary:
    ```
* - {py:obj}`prune_single_module <flashscenic.cistarget.prune_single_module>`
  - ```{autodoc2-docstring} flashscenic.cistarget.prune_single_module
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ArrayLike <flashscenic.cistarget.ArrayLike>`
  - ```{autodoc2-docstring} flashscenic.cistarget.ArrayLike
    :summary:
    ```
* - {py:obj}`MultiDatabaseCisTargetPruner <flashscenic.cistarget.MultiDatabaseCisTargetPruner>`
  - ```{autodoc2-docstring} flashscenic.cistarget.MultiDatabaseCisTargetPruner
    :summary:
    ```
* - {py:obj}`GPUCisTargetPruner <flashscenic.cistarget.GPUCisTargetPruner>`
  - ```{autodoc2-docstring} flashscenic.cistarget.GPUCisTargetPruner
    :summary:
    ```
````

### API

````{py:data} ArrayLike
:canonical: flashscenic.cistarget.ArrayLike
:value: >
   None

```{autodoc2-docstring} flashscenic.cistarget.ArrayLike
```

````

````{py:function} compute_recovery_aucs(rankings: torch.Tensor, module_gene_indices: torch.Tensor, rank_threshold: int, auc_threshold: float, weights: typing.Optional[torch.Tensor] = None) -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: flashscenic.cistarget.compute_recovery_aucs

```{autodoc2-docstring} flashscenic.cistarget.compute_recovery_aucs
```
````

````{py:function} compute_nes(aucs: torch.Tensor) -> torch.Tensor
:canonical: flashscenic.cistarget.compute_nes

```{autodoc2-docstring} flashscenic.cistarget.compute_nes
```
````

`````{py:class} MotifAnnotation()
:canonical: flashscenic.cistarget.MotifAnnotation

```{autodoc2-docstring} flashscenic.cistarget.MotifAnnotation
```

```{rubric} Initialization
```

```{autodoc2-docstring} flashscenic.cistarget.MotifAnnotation.__init__
```

````{py:method} load_from_file(fname: str, motif_similarity_fdr: float = 0.001, orthologous_identity_threshold: float = 0.0, column_names: typing.Optional[typing.Tuple[str, ...]] = None) -> flashscenic.cistarget.MotifAnnotation
:canonical: flashscenic.cistarget.MotifAnnotation.load_from_file
:classmethod:

```{autodoc2-docstring} flashscenic.cistarget.MotifAnnotation.load_from_file
```

````

````{py:method} has_annotation(motif_id: str, tf_name: typing.Optional[str] = None) -> bool
:canonical: flashscenic.cistarget.MotifAnnotation.has_annotation

```{autodoc2-docstring} flashscenic.cistarget.MotifAnnotation.has_annotation
```

````

````{py:method} get_annotation(motif_id: str, tf_name: typing.Optional[str] = None) -> typing.Optional[typing.Dict]
:canonical: flashscenic.cistarget.MotifAnnotation.get_annotation

```{autodoc2-docstring} flashscenic.cistarget.MotifAnnotation.get_annotation
```

````

`````

````{py:function} filter_by_annotations(result: typing.Dict[str, torch.Tensor], motif_names: typing.List[str], motif_annotations: typing.Optional[flashscenic.cistarget.MotifAnnotation], filter_for_annotation: bool = True) -> typing.Dict[str, torch.Tensor]
:canonical: flashscenic.cistarget.filter_by_annotations

```{autodoc2-docstring} flashscenic.cistarget.filter_by_annotations
```
````

````{py:function} compute_leading_edge(rcc: torch.Tensor, avg2std_rcc: torch.Tensor, rankings: torch.Tensor, weights: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor, int]
:canonical: flashscenic.cistarget.compute_leading_edge

```{autodoc2-docstring} flashscenic.cistarget.compute_leading_edge
```
````

````{py:function} prune_single_module(rankings: torch.Tensor, module_gene_indices: torch.Tensor, rank_threshold: int = 5000, auc_threshold: float = 0.05, nes_threshold: float = 3.0, weights: typing.Optional[torch.Tensor] = None) -> typing.Dict[str, torch.Tensor]
:canonical: flashscenic.cistarget.prune_single_module

```{autodoc2-docstring} flashscenic.cistarget.prune_single_module
```
````

`````{py:class} CisTargetPruner(rank_threshold: int = 5000, auc_threshold: float = 0.05, nes_threshold: float = 3.0, device: str = 'cuda', min_genes_per_regulon: int = 0, merge_strategy: str = 'union')
:canonical: flashscenic.cistarget.CisTargetPruner

```{autodoc2-docstring} flashscenic.cistarget.CisTargetPruner
```

```{rubric} Initialization
```

```{autodoc2-docstring} flashscenic.cistarget.CisTargetPruner.__init__
```

````{py:method} load_database(paths: typing.Union[str, typing.List[str]], database_names: typing.Optional[typing.Union[str, typing.List[str]]] = None)
:canonical: flashscenic.cistarget.CisTargetPruner.load_database

```{autodoc2-docstring} flashscenic.cistarget.CisTargetPruner.load_database
```

````

````{py:method} load_annotations(annotation_file: str, filter_for_annotation: bool = True, motif_similarity_fdr: float = 0.001, orthologous_identity_threshold: float = 0.0)
:canonical: flashscenic.cistarget.CisTargetPruner.load_annotations

```{autodoc2-docstring} flashscenic.cistarget.CisTargetPruner.load_annotations
```

````

````{py:method} load_from_tensor(rankings: flashscenic.cistarget.ArrayLike, motif_names: typing.Optional[typing.List[str]] = None, gene_names: typing.Optional[typing.List[str]] = None)
:canonical: flashscenic.cistarget.CisTargetPruner.load_from_tensor

```{autodoc2-docstring} flashscenic.cistarget.CisTargetPruner.load_from_tensor
```

````

````{py:method} genes_to_indices(genes: typing.List[str]) -> torch.Tensor
:canonical: flashscenic.cistarget.CisTargetPruner.genes_to_indices

```{autodoc2-docstring} flashscenic.cistarget.CisTargetPruner.genes_to_indices
```

````

````{py:method} prune(module_gene_indices: flashscenic.cistarget.ArrayLike, weights: typing.Optional[flashscenic.cistarget.ArrayLike] = None) -> typing.Dict[str, torch.Tensor]
:canonical: flashscenic.cistarget.CisTargetPruner.prune

```{autodoc2-docstring} flashscenic.cistarget.CisTargetPruner.prune
```

````

````{py:method} prune_batch(modules: typing.List[torch.Tensor], weights_list: typing.Optional[typing.List[torch.Tensor]] = None) -> typing.List[typing.Dict[str, torch.Tensor]]
:canonical: flashscenic.cistarget.CisTargetPruner.prune_batch

```{autodoc2-docstring} flashscenic.cistarget.CisTargetPruner.prune_batch
```

````

````{py:method} get_enriched_motif_names(result: typing.Dict[str, torch.Tensor]) -> typing.List[str]
:canonical: flashscenic.cistarget.CisTargetPruner.get_enriched_motif_names

```{autodoc2-docstring} flashscenic.cistarget.CisTargetPruner.get_enriched_motif_names
```

````

````{py:method} get_leading_edge_genes(result: typing.Dict[str, torch.Tensor], module_gene_indices: torch.Tensor) -> typing.List[typing.List[str]]
:canonical: flashscenic.cistarget.CisTargetPruner.get_leading_edge_genes

```{autodoc2-docstring} flashscenic.cistarget.CisTargetPruner.get_leading_edge_genes
```

````

````{py:method} prune_modules(modules: typing.List[torch.Tensor], tf_names: typing.List[str], gene_names: typing.List[str], weights_list: typing.Optional[typing.List[torch.Tensor]] = None) -> typing.List[typing.Dict]
:canonical: flashscenic.cistarget.CisTargetPruner.prune_modules

```{autodoc2-docstring} flashscenic.cistarget.CisTargetPruner.prune_modules
```

````

````{py:method} _merge_regulons(regulons: typing.List[typing.Dict]) -> typing.List[typing.Dict]
:canonical: flashscenic.cistarget.CisTargetPruner._merge_regulons

```{autodoc2-docstring} flashscenic.cistarget.CisTargetPruner._merge_regulons
```

````

````{py:method} _merge_regulons_by_tf(regulons: typing.List[typing.Dict]) -> typing.List[typing.Dict]
:canonical: flashscenic.cistarget.CisTargetPruner._merge_regulons_by_tf

```{autodoc2-docstring} flashscenic.cistarget.CisTargetPruner._merge_regulons_by_tf
```

````

````{py:method} clear_gpu_memory()
:canonical: flashscenic.cistarget.CisTargetPruner.clear_gpu_memory

```{autodoc2-docstring} flashscenic.cistarget.CisTargetPruner.clear_gpu_memory
```

````

`````

````{py:data} MultiDatabaseCisTargetPruner
:canonical: flashscenic.cistarget.MultiDatabaseCisTargetPruner
:value: >
   None

```{autodoc2-docstring} flashscenic.cistarget.MultiDatabaseCisTargetPruner
```

````

````{py:data} GPUCisTargetPruner
:canonical: flashscenic.cistarget.GPUCisTargetPruner
:value: >
   None

```{autodoc2-docstring} flashscenic.cistarget.GPUCisTargetPruner
```

````
