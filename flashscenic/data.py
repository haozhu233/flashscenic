"""
Download utilities for flashscenic cistarget resource files.

Supports downloading TF lists, motif ranking databases, and motif annotation
files from configurable data sources (default: Aertslab SCENIC resources).
"""

import os
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


AERTSLAB_BASE = "https://resources.aertslab.org/cistarget/"


@dataclass
class ResourceFile:
    """Describes a single downloadable resource file."""
    url: str
    filename: str
    category: str  # 'tf_list', 'ranking_db', 'motif_annotation'
    description: str = ""
    expected_size_mb: float = 0


@dataclass
class ResourceSet:
    """Complete set of resources for a species/version/datasource combination."""
    species: str
    version: str
    datasource: str
    files: List[ResourceFile] = field(default_factory=list)

    @property
    def tf_list(self) -> Optional[ResourceFile]:
        matches = [f for f in self.files if f.category == 'tf_list']
        return matches[0] if matches else None

    @property
    def ranking_dbs(self) -> List[ResourceFile]:
        return [f for f in self.files if f.category == 'ranking_db']

    @property
    def motif_annotation(self) -> Optional[ResourceFile]:
        matches = [f for f in self.files if f.category == 'motif_annotation']
        return matches[0] if matches else None


@dataclass
class DownloadedResources:
    """Paths to downloaded resource files."""
    tf_list: Optional[Path] = None
    ranking_dbs: List[Path] = field(default_factory=list)
    motif_annotation: Optional[Path] = None
    cache_dir: Path = field(default_factory=lambda: Path("."))

    def __repr__(self) -> str:
        dbs = ", ".join(str(p.name) for p in self.ranking_dbs)
        return (
            f"DownloadedResources(\n"
            f"  cache_dir={self.cache_dir},\n"
            f"  tf_list={self.tf_list.name if self.tf_list else None},\n"
            f"  ranking_dbs=[{dbs}],\n"
            f"  motif_annotation="
            f"{self.motif_annotation.name if self.motif_annotation else None}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Resource registry
# ---------------------------------------------------------------------------

_RESOURCE_REGISTRY: Dict[Tuple[str, str, str], ResourceSet] = {}


def _register_scenic_resources():
    """Register all known Aertslab SCENIC resources."""

    # ---- Human v10 ----
    _RESOURCE_REGISTRY[("scenic", "human", "v10")] = ResourceSet(
        species="human", version="v10", datasource="scenic",
        files=[
            ResourceFile(
                url=AERTSLAB_BASE + "tf_lists/allTFs_hg38.txt",
                filename="allTFs_hg38.txt",
                category="tf_list",
                description="Human TF list (hg38)",
                expected_size_mb=0.01,
            ),
            ResourceFile(
                url=(AERTSLAB_BASE
                     + "databases/homo_sapiens/hg38/refseq_r80/mc_v10_clust/"
                     + "gene_based/hg38_500bp_up_100bp_down_full_tx_v10_clust"
                     + ".genes_vs_motifs.rankings.feather"),
                filename="hg38_500bp_up_100bp_down_full_tx_v10_clust"
                         ".genes_vs_motifs.rankings.feather",
                category="ranking_db",
                description="Human 500bp/100bp ranking database (v10)",
                expected_size_mb=298,
            ),
            ResourceFile(
                url=(AERTSLAB_BASE
                     + "databases/homo_sapiens/hg38/refseq_r80/mc_v10_clust/"
                     + "gene_based/hg38_10kbp_up_10kbp_down_full_tx_v10_clust"
                     + ".genes_vs_motifs.rankings.feather"),
                filename="hg38_10kbp_up_10kbp_down_full_tx_v10_clust"
                         ".genes_vs_motifs.rankings.feather",
                category="ranking_db",
                description="Human 10kbp ranking database (v10)",
                expected_size_mb=297,
            ),
            ResourceFile(
                url=AERTSLAB_BASE
                    + "motif2tf/motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl",
                filename="motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl",
                category="motif_annotation",
                description="Human motif annotations (v10)",
                expected_size_mb=94,
            ),
        ],
    )

    # ---- Mouse v10 ----
    _RESOURCE_REGISTRY[("scenic", "mouse", "v10")] = ResourceSet(
        species="mouse", version="v10", datasource="scenic",
        files=[
            ResourceFile(
                url=AERTSLAB_BASE + "tf_lists/allTFs_mm.txt",
                filename="allTFs_mm.txt",
                category="tf_list",
                description="Mouse TF list",
                expected_size_mb=0.01,
            ),
            ResourceFile(
                url=(AERTSLAB_BASE
                     + "databases/mus_musculus/mm10/refseq_r80/mc_v10_clust/"
                     + "gene_based/mm10_500bp_up_100bp_down_full_tx_v10_clust"
                     + ".genes_vs_motifs.rankings.feather"),
                filename="mm10_500bp_up_100bp_down_full_tx_v10_clust"
                         ".genes_vs_motifs.rankings.feather",
                category="ranking_db",
                description="Mouse 500bp/100bp ranking database (v10)",
                expected_size_mb=227,
            ),
            ResourceFile(
                url=(AERTSLAB_BASE
                     + "databases/mus_musculus/mm10/refseq_r80/mc_v10_clust/"
                     + "gene_based/mm10_10kbp_up_10kbp_down_full_tx_v10_clust"
                     + ".genes_vs_motifs.rankings.feather"),
                filename="mm10_10kbp_up_10kbp_down_full_tx_v10_clust"
                         ".genes_vs_motifs.rankings.feather",
                category="ranking_db",
                description="Mouse 10kbp ranking database (v10)",
                expected_size_mb=226,
            ),
            ResourceFile(
                url=AERTSLAB_BASE
                    + "motif2tf/motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl",
                filename="motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl",
                category="motif_annotation",
                description="Mouse motif annotations (v10)",
                expected_size_mb=108,
            ),
        ],
    )

    # ---- Drosophila v10 ----
    _RESOURCE_REGISTRY[("scenic", "drosophila", "v10")] = ResourceSet(
        species="drosophila", version="v10", datasource="scenic",
        files=[
            ResourceFile(
                url=AERTSLAB_BASE + "tf_lists/allTFs_dmel.txt",
                filename="allTFs_dmel.txt",
                category="tf_list",
                description="Drosophila TF list",
                expected_size_mb=0.005,
            ),
            ResourceFile(
                url=(AERTSLAB_BASE
                     + "databases/drosophila_melanogaster/dm6/"
                     + "flybase_r6.02/mc_v10_clust/gene_based/"
                     + "dm6_500bp_up_100bp_down_full_tx_v10_clust"
                     + ".genes_vs_motifs.rankings.feather"),
                filename="dm6_500bp_up_100bp_down_full_tx_v10_clust"
                         ".genes_vs_motifs.rankings.feather",
                category="ranking_db",
                description="Drosophila 500bp/100bp ranking database (v10)",
                expected_size_mb=120,
            ),
            ResourceFile(
                url=(AERTSLAB_BASE
                     + "databases/drosophila_melanogaster/dm6/"
                     + "flybase_r6.02/mc_v10_clust/gene_based/"
                     + "dm6_10kbp_up_10kbp_down_full_tx_v10_clust"
                     + ".genes_vs_motifs.rankings.feather"),
                filename="dm6_10kbp_up_10kbp_down_full_tx_v10_clust"
                         ".genes_vs_motifs.rankings.feather",
                category="ranking_db",
                description="Drosophila 10kbp ranking database (v10)",
                expected_size_mb=120,
            ),
            ResourceFile(
                url=AERTSLAB_BASE
                    + "motif2tf/motifs-v10nr_clust-nr.flybase-m0.001-o0.0.tbl",
                filename="motifs-v10nr_clust-nr.flybase-m0.001-o0.0.tbl",
                category="motif_annotation",
                description="Drosophila motif annotations (v10)",
                expected_size_mb=66,
            ),
        ],
    )

    # ---- Human v9 ----
    _RESOURCE_REGISTRY[("scenic", "human", "v9")] = ResourceSet(
        species="human", version="v9", datasource="scenic",
        files=[
            ResourceFile(
                url=AERTSLAB_BASE + "tf_lists/allTFs_hg38.txt",
                filename="allTFs_hg38.txt",
                category="tf_list",
                description="Human TF list (hg38)",
                expected_size_mb=0.01,
            ),
            ResourceFile(
                url=(AERTSLAB_BASE
                     + "databases/homo_sapiens/hg38/refseq_r80/mc9nr/"
                     + "gene_based/hg38__refseq-r80__500bp_up_and_100bp"
                     + "_down_tss.mc9nr.genes_vs_motifs.rankings.feather"),
                filename="hg38__refseq-r80__500bp_up_and_100bp_down_tss"
                         ".mc9nr.genes_vs_motifs.rankings.feather",
                category="ranking_db",
                description="Human 500bp/100bp ranking database (v9)",
                expected_size_mb=1200,
            ),
            ResourceFile(
                url=(AERTSLAB_BASE
                     + "databases/homo_sapiens/hg38/refseq_r80/mc9nr/"
                     + "gene_based/hg38__refseq-r80__10kb_up_and_down_tss"
                     + ".mc9nr.genes_vs_motifs.rankings.feather"),
                filename="hg38__refseq-r80__10kb_up_and_down_tss"
                         ".mc9nr.genes_vs_motifs.rankings.feather",
                category="ranking_db",
                description="Human 10kb ranking database (v9)",
                expected_size_mb=1200,
            ),
            ResourceFile(
                url=AERTSLAB_BASE
                    + "motif2tf/motifs-v9-nr.hgnc-m0.001-o0.0.tbl",
                filename="motifs-v9-nr.hgnc-m0.001-o0.0.tbl",
                category="motif_annotation",
                description="Human motif annotations (v9)",
                expected_size_mb=99,
            ),
        ],
    )

    # ---- Mouse v9 ----
    _RESOURCE_REGISTRY[("scenic", "mouse", "v9")] = ResourceSet(
        species="mouse", version="v9", datasource="scenic",
        files=[
            ResourceFile(
                url=AERTSLAB_BASE + "tf_lists/allTFs_mm.txt",
                filename="allTFs_mm.txt",
                category="tf_list",
                description="Mouse TF list",
                expected_size_mb=0.01,
            ),
            ResourceFile(
                url=(AERTSLAB_BASE
                     + "databases/mus_musculus/mm10/refseq_r80/mc9nr/"
                     + "gene_based/mm10__refseq-r80__500bp_up_and_100bp"
                     + "_down_tss.mc9nr.genes_vs_motifs.rankings.feather"),
                filename="mm10__refseq-r80__500bp_up_and_100bp_down_tss"
                         ".mc9nr.genes_vs_motifs.rankings.feather",
                category="ranking_db",
                description="Mouse 500bp/100bp ranking database (v9)",
                expected_size_mb=900,
            ),
            ResourceFile(
                url=(AERTSLAB_BASE
                     + "databases/mus_musculus/mm10/refseq_r80/mc9nr/"
                     + "gene_based/mm10__refseq-r80__10kb_up_and_down_tss"
                     + ".mc9nr.genes_vs_motifs.rankings.feather"),
                filename="mm10__refseq-r80__10kb_up_and_down_tss"
                         ".mc9nr.genes_vs_motifs.rankings.feather",
                category="ranking_db",
                description="Mouse 10kb ranking database (v9)",
                expected_size_mb=900,
            ),
            ResourceFile(
                url=AERTSLAB_BASE
                    + "motif2tf/motifs-v9-nr.mgi-m0.001-o0.0.tbl",
                filename="motifs-v9-nr.mgi-m0.001-o0.0.tbl",
                category="motif_annotation",
                description="Mouse motif annotations (v9)",
                expected_size_mb=107,
            ),
        ],
    )


_register_scenic_resources()


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_file(
    url: str,
    dest_path: Path,
    description: str = "",
    expected_size_mb: float = 0,
    chunk_size: int = 1024 * 1024,  # 1 MB chunks
    max_retries: int = 3,
) -> Path:
    """
    Download a single file with progress reporting and retry logic.

    Writes to a .partial temp file first, then renames on success.
    """
    partial_path = dest_path.with_suffix(dest_path.suffix + ".partial")
    label = description or dest_path.name

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as response:
                total_bytes = response.getheader("Content-Length")
                total_mb = float(total_bytes) / (1024 * 1024) if total_bytes else expected_size_mb
                total_bytes = int(total_bytes) if total_bytes else None

                downloaded = 0
                last_report = 0

                with open(partial_path, "wb") as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        downloaded_mb = downloaded / (1024 * 1024)

                        # Report progress every ~10 MB
                        if downloaded_mb - last_report >= 10 or not chunk:
                            if total_bytes:
                                pct = downloaded / total_bytes * 100
                                print(
                                    f"\r  [{label}] {downloaded_mb:.1f} / "
                                    f"{total_mb:.1f} MB ({pct:.0f}%)",
                                    end="", flush=True,
                                )
                            else:
                                print(
                                    f"\r  [{label}] {downloaded_mb:.1f} MB",
                                    end="", flush=True,
                                )
                            last_report = downloaded_mb

                print()  # newline after progress

            # Rename partial to final
            partial_path.rename(dest_path)
            return dest_path

        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            if partial_path.exists():
                partial_path.unlink()
            if attempt < max_retries:
                wait = 4 ** (attempt - 1)  # 1s, 4s, 16s
                print(f"  Retry {attempt}/{max_retries} for {label} "
                      f"(waiting {wait}s): {e}")
                time.sleep(wait)
            else:
                raise ConnectionError(
                    f"Failed to download {url} after {max_retries} attempts: {e}"
                ) from e

    # Should not reach here, but just in case
    raise ConnectionError(f"Failed to download {url}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_data(
    species: str = "human",
    version: str = "v10",
    datasource: str = "scenic",
    cache_dir: Optional[str] = None,
    force: bool = False,
) -> DownloadedResources:
    """
    Download cistarget resource files required for the flashscenic pipeline.

    Parameters
    ----------
    species : str, default='human'
        Species to download resources for. One of 'human', 'mouse',
        'drosophila'.
    version : str, default='v10'
        Motif collection version. One of 'v10' (recommended), 'v9'.
    datasource : str, default='scenic'
        Data source identifier. Currently only 'scenic' (Aertslab) is
        supported. Architecture supports adding alternative sources.
    cache_dir : str or None, default=None
        Local directory to store downloaded files. If None, defaults to
        ``./flashscenic_data/``.
    force : bool, default=False
        If True, re-download files even if they already exist locally.

    Returns
    -------
    DownloadedResources
        Dataclass with Path objects pointing to each downloaded file.

    Raises
    ------
    ValueError
        If the species/version/datasource combination is not recognized.
    ConnectionError
        If download fails after retries.
    """
    key = (datasource, species, version)
    if key not in _RESOURCE_REGISTRY:
        available = sorted(_RESOURCE_REGISTRY.keys())
        raise ValueError(
            f"Unknown resource combination: datasource={datasource!r}, "
            f"species={species!r}, version={version!r}.\n"
            f"Available combinations: {available}"
        )

    resource_set = _RESOURCE_REGISTRY[key]

    # Resolve cache directory
    if cache_dir is None:
        cache_path = Path("./flashscenic_data")
    else:
        cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    result = DownloadedResources(cache_dir=cache_path)

    print(f"Downloading {datasource}/{species}/{version} resources "
          f"to {cache_path}/")

    for rf in resource_set.files:
        dest = cache_path / rf.filename

        if dest.exists() and not force:
            print(f"  [{rf.description}] Already cached: {rf.filename}")
        else:
            _download_file(
                url=rf.url,
                dest_path=dest,
                description=rf.description,
                expected_size_mb=rf.expected_size_mb,
            )

        # Assign to result
        if rf.category == "tf_list":
            result.tf_list = dest
        elif rf.category == "ranking_db":
            result.ranking_dbs.append(dest)
        elif rf.category == "motif_annotation":
            result.motif_annotation = dest

    print("Download complete.")
    return result


def list_available_resources(
    datasource: Optional[str] = None,
    species: Optional[str] = None,
    version: Optional[str] = None,
) -> List[ResourceSet]:
    """
    List available resource sets, optionally filtered.

    Parameters
    ----------
    datasource : str or None
        Filter by data source (e.g., 'scenic').
    species : str or None
        Filter by species (e.g., 'human', 'mouse', 'drosophila').
    version : str or None
        Filter by version (e.g., 'v10', 'v9').

    Returns
    -------
    list of ResourceSet
        Matching resource sets.
    """
    results = []
    for (ds, sp, ver), rs in sorted(_RESOURCE_REGISTRY.items()):
        if datasource is not None and ds != datasource:
            continue
        if species is not None and sp != species:
            continue
        if version is not None and ver != version:
            continue
        results.append(rs)
    return results
