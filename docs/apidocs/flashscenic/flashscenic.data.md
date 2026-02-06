# {py:mod}`flashscenic.data`

```{py:module} flashscenic.data
```

```{autodoc2-docstring} flashscenic.data
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ResourceFile <flashscenic.data.ResourceFile>`
  - ```{autodoc2-docstring} flashscenic.data.ResourceFile
    :summary:
    ```
* - {py:obj}`ResourceSet <flashscenic.data.ResourceSet>`
  - ```{autodoc2-docstring} flashscenic.data.ResourceSet
    :summary:
    ```
* - {py:obj}`DownloadedResources <flashscenic.data.DownloadedResources>`
  - ```{autodoc2-docstring} flashscenic.data.DownloadedResources
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_register_scenic_resources <flashscenic.data._register_scenic_resources>`
  - ```{autodoc2-docstring} flashscenic.data._register_scenic_resources
    :summary:
    ```
* - {py:obj}`_download_file <flashscenic.data._download_file>`
  - ```{autodoc2-docstring} flashscenic.data._download_file
    :summary:
    ```
* - {py:obj}`download_data <flashscenic.data.download_data>`
  - ```{autodoc2-docstring} flashscenic.data.download_data
    :summary:
    ```
* - {py:obj}`list_available_resources <flashscenic.data.list_available_resources>`
  - ```{autodoc2-docstring} flashscenic.data.list_available_resources
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AERTSLAB_BASE <flashscenic.data.AERTSLAB_BASE>`
  - ```{autodoc2-docstring} flashscenic.data.AERTSLAB_BASE
    :summary:
    ```
* - {py:obj}`_RESOURCE_REGISTRY <flashscenic.data._RESOURCE_REGISTRY>`
  - ```{autodoc2-docstring} flashscenic.data._RESOURCE_REGISTRY
    :summary:
    ```
````

### API

````{py:data} AERTSLAB_BASE
:canonical: flashscenic.data.AERTSLAB_BASE
:value: >
   'https://resources.aertslab.org/cistarget/'

```{autodoc2-docstring} flashscenic.data.AERTSLAB_BASE
```

````

`````{py:class} ResourceFile
:canonical: flashscenic.data.ResourceFile

```{autodoc2-docstring} flashscenic.data.ResourceFile
```

````{py:attribute} url
:canonical: flashscenic.data.ResourceFile.url
:type: str
:value: >
   None

```{autodoc2-docstring} flashscenic.data.ResourceFile.url
```

````

````{py:attribute} filename
:canonical: flashscenic.data.ResourceFile.filename
:type: str
:value: >
   None

```{autodoc2-docstring} flashscenic.data.ResourceFile.filename
```

````

````{py:attribute} category
:canonical: flashscenic.data.ResourceFile.category
:type: str
:value: >
   None

```{autodoc2-docstring} flashscenic.data.ResourceFile.category
```

````

````{py:attribute} description
:canonical: flashscenic.data.ResourceFile.description
:type: str
:value: <Multiline-String>

```{autodoc2-docstring} flashscenic.data.ResourceFile.description
```

````

````{py:attribute} expected_size_mb
:canonical: flashscenic.data.ResourceFile.expected_size_mb
:type: float
:value: >
   0

```{autodoc2-docstring} flashscenic.data.ResourceFile.expected_size_mb
```

````

`````

`````{py:class} ResourceSet
:canonical: flashscenic.data.ResourceSet

```{autodoc2-docstring} flashscenic.data.ResourceSet
```

````{py:attribute} species
:canonical: flashscenic.data.ResourceSet.species
:type: str
:value: >
   None

```{autodoc2-docstring} flashscenic.data.ResourceSet.species
```

````

````{py:attribute} version
:canonical: flashscenic.data.ResourceSet.version
:type: str
:value: >
   None

```{autodoc2-docstring} flashscenic.data.ResourceSet.version
```

````

````{py:attribute} datasource
:canonical: flashscenic.data.ResourceSet.datasource
:type: str
:value: >
   None

```{autodoc2-docstring} flashscenic.data.ResourceSet.datasource
```

````

````{py:attribute} files
:canonical: flashscenic.data.ResourceSet.files
:type: typing.List[flashscenic.data.ResourceFile]
:value: >
   'field(...)'

```{autodoc2-docstring} flashscenic.data.ResourceSet.files
```

````

````{py:property} tf_list
:canonical: flashscenic.data.ResourceSet.tf_list
:type: typing.Optional[flashscenic.data.ResourceFile]

```{autodoc2-docstring} flashscenic.data.ResourceSet.tf_list
```

````

````{py:property} ranking_dbs
:canonical: flashscenic.data.ResourceSet.ranking_dbs
:type: typing.List[flashscenic.data.ResourceFile]

```{autodoc2-docstring} flashscenic.data.ResourceSet.ranking_dbs
```

````

````{py:property} motif_annotation
:canonical: flashscenic.data.ResourceSet.motif_annotation
:type: typing.Optional[flashscenic.data.ResourceFile]

```{autodoc2-docstring} flashscenic.data.ResourceSet.motif_annotation
```

````

`````

`````{py:class} DownloadedResources
:canonical: flashscenic.data.DownloadedResources

```{autodoc2-docstring} flashscenic.data.DownloadedResources
```

````{py:attribute} tf_list
:canonical: flashscenic.data.DownloadedResources.tf_list
:type: typing.Optional[pathlib.Path]
:value: >
   None

```{autodoc2-docstring} flashscenic.data.DownloadedResources.tf_list
```

````

````{py:attribute} ranking_dbs
:canonical: flashscenic.data.DownloadedResources.ranking_dbs
:type: typing.List[pathlib.Path]
:value: >
   'field(...)'

```{autodoc2-docstring} flashscenic.data.DownloadedResources.ranking_dbs
```

````

````{py:attribute} motif_annotation
:canonical: flashscenic.data.DownloadedResources.motif_annotation
:type: typing.Optional[pathlib.Path]
:value: >
   None

```{autodoc2-docstring} flashscenic.data.DownloadedResources.motif_annotation
```

````

````{py:attribute} cache_dir
:canonical: flashscenic.data.DownloadedResources.cache_dir
:type: pathlib.Path
:value: >
   'field(...)'

```{autodoc2-docstring} flashscenic.data.DownloadedResources.cache_dir
```

````

````{py:method} __repr__() -> str
:canonical: flashscenic.data.DownloadedResources.__repr__

````

`````

````{py:data} _RESOURCE_REGISTRY
:canonical: flashscenic.data._RESOURCE_REGISTRY
:type: typing.Dict[typing.Tuple[str, str, str], flashscenic.data.ResourceSet]
:value: >
   None

```{autodoc2-docstring} flashscenic.data._RESOURCE_REGISTRY
```

````

````{py:function} _register_scenic_resources()
:canonical: flashscenic.data._register_scenic_resources

```{autodoc2-docstring} flashscenic.data._register_scenic_resources
```
````

````{py:function} _download_file(url: str, dest_path: pathlib.Path, description: str = '', expected_size_mb: float = 0, chunk_size: int = 1024 * 1024, max_retries: int = 3) -> pathlib.Path
:canonical: flashscenic.data._download_file

```{autodoc2-docstring} flashscenic.data._download_file
```
````

````{py:function} download_data(species: str = 'human', version: str = 'v10', datasource: str = 'scenic', cache_dir: typing.Optional[str] = None, force: bool = False) -> flashscenic.data.DownloadedResources
:canonical: flashscenic.data.download_data

```{autodoc2-docstring} flashscenic.data.download_data
```
````

````{py:function} list_available_resources(datasource: typing.Optional[str] = None, species: typing.Optional[str] = None, version: typing.Optional[str] = None) -> typing.List[flashscenic.data.ResourceSet]
:canonical: flashscenic.data.list_available_resources

```{autodoc2-docstring} flashscenic.data.list_available_resources
```
````
