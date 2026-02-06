# {py:mod}`flashscenic.modules`

```{py:module} flashscenic.modules
```

```{autodoc2-docstring} flashscenic.modules
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`select_topk_targets <flashscenic.modules.select_topk_targets>`
  - ```{autodoc2-docstring} flashscenic.modules.select_topk_targets
    :summary:
    ```
* - {py:obj}`select_threshold_targets <flashscenic.modules.select_threshold_targets>`
  - ```{autodoc2-docstring} flashscenic.modules.select_threshold_targets
    :summary:
    ```
* - {py:obj}`filter_by_min_targets <flashscenic.modules.filter_by_min_targets>`
  - ```{autodoc2-docstring} flashscenic.modules.filter_by_min_targets
    :summary:
    ```
* - {py:obj}`filter_by_mapped_fraction <flashscenic.modules.filter_by_mapped_fraction>`
  - ```{autodoc2-docstring} flashscenic.modules.filter_by_mapped_fraction
    :summary:
    ```
* - {py:obj}`get_target_indices <flashscenic.modules.get_target_indices>`
  - ```{autodoc2-docstring} flashscenic.modules.get_target_indices
    :summary:
    ```
* - {py:obj}`binarize <flashscenic.modules.binarize>`
  - ```{autodoc2-docstring} flashscenic.modules.binarize
    :summary:
    ```
* - {py:obj}`to_numpy <flashscenic.modules.to_numpy>`
  - ```{autodoc2-docstring} flashscenic.modules.to_numpy
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ArrayLike <flashscenic.modules.ArrayLike>`
  - ```{autodoc2-docstring} flashscenic.modules.ArrayLike
    :summary:
    ```
````

### API

````{py:data} ArrayLike
:canonical: flashscenic.modules.ArrayLike
:value: >
   None

```{autodoc2-docstring} flashscenic.modules.ArrayLike
```

````

````{py:function} select_topk_targets(adj: flashscenic.modules.ArrayLike, k: int = 50, include_tf: bool = True, tf_indices: typing.Optional[flashscenic.modules.ArrayLike] = None, device: str = 'cuda') -> torch.Tensor
:canonical: flashscenic.modules.select_topk_targets

```{autodoc2-docstring} flashscenic.modules.select_topk_targets
```
````

````{py:function} select_threshold_targets(adj: flashscenic.modules.ArrayLike, threshold: float = 0.0, percentile: typing.Optional[float] = None, include_tf: bool = True, tf_indices: typing.Optional[flashscenic.modules.ArrayLike] = None, device: str = 'cuda') -> torch.Tensor
:canonical: flashscenic.modules.select_threshold_targets

```{autodoc2-docstring} flashscenic.modules.select_threshold_targets
```
````

````{py:function} filter_by_min_targets(adj: flashscenic.modules.ArrayLike, min_targets: int = 20, min_fraction: typing.Optional[float] = 0.8, device: str = 'cuda') -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: flashscenic.modules.filter_by_min_targets

```{autodoc2-docstring} flashscenic.modules.filter_by_min_targets
```
````

````{py:function} filter_by_mapped_fraction(adj: flashscenic.modules.ArrayLike, reference_indices: typing.Optional[flashscenic.modules.ArrayLike] = None, min_fraction: float = 0.8, device: str = 'cuda') -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: flashscenic.modules.filter_by_mapped_fraction

```{autodoc2-docstring} flashscenic.modules.filter_by_mapped_fraction
```
````

````{py:function} get_target_indices(adj: flashscenic.modules.ArrayLike, device: str = 'cuda') -> typing.Tuple[torch.Tensor, torch.Tensor]
:canonical: flashscenic.modules.get_target_indices

```{autodoc2-docstring} flashscenic.modules.get_target_indices
```
````

````{py:function} binarize(adj: flashscenic.modules.ArrayLike, device: str = 'cuda') -> torch.Tensor
:canonical: flashscenic.modules.binarize

```{autodoc2-docstring} flashscenic.modules.binarize
```
````

````{py:function} to_numpy(tensor: torch.Tensor) -> numpy.ndarray
:canonical: flashscenic.modules.to_numpy

```{autodoc2-docstring} flashscenic.modules.to_numpy
```
````
