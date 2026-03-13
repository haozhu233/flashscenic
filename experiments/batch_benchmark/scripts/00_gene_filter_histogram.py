"""
Diagnostic: plot "% cells a gene appears in" per dataset to choose min_cells_frac.

Run from experiments/batch_benchmark/:
    python scripts/00_gene_filter_histogram.py

Outputs: results/gene_filter_histograms.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import h5py
import matplotlib.pyplot as plt
import anndata as ad
import numpy as np
from config import DATASETS, DATA_DIR

CANDIDATE_THRESHOLDS = [0.1, 0.5, 1.0, 5.0]

fig, axes = plt.subplots(1, len(DATASETS), figsize=(5 * len(DATASETS), 4), sharey=False)
if len(DATASETS) == 1:
    axes = [axes]

for ax, (name, cfg) in zip(axes, DATASETS.items()):
    path = DATA_DIR / f"{name}.h5ad"
    if not path.exists():
        ax.set_title(f"{name}\n(not found)")
        ax.axis("off")
        continue

    # Read only metadata to get shapes (backed mode is fine for this)
    adata_meta = ad.read_h5ad(path, backed="r")
    n_obs = adata_meta.n_obs
    n_vars = adata_meta.n_vars
    raw_n_vars = adata_meta.raw.n_vars if adata_meta.raw is not None else n_vars
    adata_meta.file.close()

    # Read only the sparse indices array from the HDF5 file — no data values needed.
    # A CSR matrix stores column indices of nonzero entries in "indices";
    # np.bincount over those gives nnz per gene without loading the full matrix.
    with h5py.File(path, "r") as f:
        result = None
        for loc, nv in [
            ("layers/counts",     n_vars),
            ("layers/raw_counts", n_vars),
            ("raw/X",             raw_n_vars),
            ("X",                 n_vars),
        ]:
            g = f.get(loc)
            if g is not None and isinstance(g, h5py.Group) and "indices" in g:
                result = (g["indices"][:], nv)
                break

        if result is None:
            print(f"  [skip] {name}: no sparse count matrix found")
            ax.set_title(f"{name}\n(no sparse data)")
            ax.axis("off")
            continue

    indices, nv = result
    cells_per_gene = np.bincount(indices, minlength=nv)
    pct = cells_per_gene / n_obs * 100

    ax.hist(pct, bins=200, log=True, color="steelblue", alpha=0.7)
    ax.set_xlabel("% cells gene appears in")
    ax.set_ylabel("# genes (log scale)")
    ax.set_title(f"{name}\n(n_cells={n_obs:,}, n_genes={nv:,})")

    ymin, ymax = ax.get_ylim()
    for thresh in CANDIDATE_THRESHOLDS:
        n_kept = (pct >= thresh).sum()
        ax.axvline(thresh, color="red", linestyle="--", linewidth=0.8)
        ax.text(thresh + 0.05, ymax * 0.3,
                f"{thresh}%\n→{n_kept:,}", fontsize=7, color="red")

    print(f"{name} (n_cells={n_obs:,}, n_genes={nv:,}):")
    for thresh in CANDIDATE_THRESHOLDS:
        n_kept = (pct >= thresh).sum()
        print(f"  >= {thresh}% cells: {n_kept:,} genes")
    print()

plt.tight_layout()
out = Path("results/gene_filter_histograms.png")
out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=150)
print(f"Saved {out}")
