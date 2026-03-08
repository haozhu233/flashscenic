"""
Step 0b: Validate downloaded datasets.

Run AFTER downloading with the shell script:
    bash 00_download_data.sh

Checks that required metadata columns (batch, cell_type, sex) exist,
prints distributions and cross-tabulations, and writes
results/data_summary.json.

Usage:
    python 00_validate_data.py [--dataset immune_human|pancreas|ad_neurons|all]
"""

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import DATASETS, PRIMARY_DATASET, RESULTS_DIR


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_dataset(name: str, cfg: dict) -> dict:
    """
    Load the dataset and run validation checks.
    Returns a summary dict for the dataset.
    """
    path: Path = cfg["path"]
    print(f"\n{'='*60}")
    print(f"Validating: {name}")
    print(f"{'='*60}")

    adata = ad.read_h5ad(path)
    print(f"  Shape: {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")
    print(f"  obs columns: {list(adata.obs.columns)}")

    summary = {
        "name": name,
        "path": str(path),
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "obs_columns": list(adata.obs.columns),
        "checks_passed": [],
        "checks_failed": [],
        "warnings": [],
    }

    # --- Check batch column ---
    batch_key = cfg["batch_key"]
    if batch_key in adata.obs.columns:
        batches = adata.obs[batch_key].value_counts()
        n_batches = len(batches)
        print(f"\n  Batch column '{batch_key}': {n_batches} batches")
        print(batches.to_string())
        summary["batches"] = batches.to_dict()
        summary["n_batches"] = n_batches
        if n_batches >= 3:
            summary["checks_passed"].append(f"batch_key '{batch_key}' with {n_batches} batches")
        else:
            summary["checks_failed"].append(
                f"batch_key '{batch_key}' has only {n_batches} batches (need ≥3)"
            )
    else:
        # Try to find an alternative
        candidates = [c for c in adata.obs.columns if "batch" in c.lower()
                      or "tech" in c.lower() or "study" in c.lower()]
        msg = f"batch_key '{batch_key}' not found. Candidates: {candidates}"
        summary["checks_failed"].append(msg)
        print(f"  WARNING: {msg}")

    # --- Check cell type column ---
    ct_key = cfg["cell_type_key"]
    if ct_key in adata.obs.columns:
        cell_types = adata.obs[ct_key].value_counts()
        n_ct = len(cell_types)
        print(f"\n  Cell type column '{ct_key}': {n_ct} types")
        print(cell_types.to_string())
        summary["cell_types"] = cell_types.to_dict()
        summary["n_cell_types"] = n_ct
        summary["checks_passed"].append(f"cell_type_key '{ct_key}' with {n_ct} types")
    else:
        candidates = [c for c in adata.obs.columns if "cell" in c.lower()
                      or "type" in c.lower() or "cluster" in c.lower()]
        msg = f"cell_type_key '{ct_key}' not found. Candidates: {candidates}"
        summary["checks_failed"].append(msg)
        print(f"  WARNING: {msg}")

    # --- Check sex column ---
    sex_key = cfg["sex_key"]
    if sex_key is None:
        summary["warnings"].append("sex_key is None — sex prediction experiment will be skipped")
        print(f"\n  Sex column: not configured (sex test will be skipped)")
    elif sex_key in adata.obs.columns:
        sex_counts = adata.obs[sex_key].value_counts()
        print(f"\n  Sex column '{sex_key}':")
        print(sex_counts.to_string())
        summary["sex_counts"] = sex_counts.to_dict()

        # Cross-tabulate sex × batch to check for confounding
        if batch_key in adata.obs.columns:
            xtab = pd.crosstab(adata.obs[sex_key], adata.obs[batch_key])
            print(f"\n  Sex × batch cross-tabulation:")
            print(xtab.to_string())
            summary["sex_batch_crosstab"] = xtab.to_dict()

            # Gate: each sex must appear in at least 2 batches
            min_thresh = cfg.get("min_cells_per_sex_per_batch", 50)
            sex_batch_ok = (xtab >= min_thresh).sum(axis=1)
            problematic = sex_batch_ok[sex_batch_ok < 2].index.tolist()
            if problematic:
                msg = (f"Sex value(s) {problematic} appear in <2 batches with ≥{min_thresh} "
                       f"cells. Cross-batch sex prediction may be unreliable.")
                summary["warnings"].append(msg)
                print(f"\n  WARNING: {msg}")
            else:
                summary["checks_passed"].append(
                    f"sex_key '{sex_key}' well-distributed across batches"
                )
        summary["sex_key_found"] = True
    else:
        # Try to find alternatives
        candidates = [c for c in adata.obs.columns
                      if any(kw in c.lower() for kw in ["sex", "gender", "male", "female"])]
        msg = f"sex_key '{sex_key}' not found. Candidates: {candidates}"
        if candidates:
            summary["warnings"].append(msg + " — consider updating config.py")
        else:
            summary["checks_failed"].append(msg + " — sex test cannot run")
        print(f"  WARNING: {msg}")
        summary["sex_key_found"] = False

    # --- Check raw counts availability ---
    # scVI and flashSCENIC need counts; check X and layers
    import scipy.sparse as sp
    has_counts = False
    x_sample = adata.X[:100]
    if sp.issparse(x_sample):
        sample = np.asarray(x_sample.todense()).ravel()
    else:
        sample = np.asarray(x_sample).ravel()

    if np.all(sample >= 0) and np.allclose(sample, sample.astype(int)):
        has_counts = True
        summary["counts_in"] = "X"
        summary["checks_passed"].append("raw counts available in adata.X")
    elif "counts" in adata.layers:
        has_counts = True
        summary["counts_in"] = "layers['counts']"
        summary["checks_passed"].append("raw counts available in adata.layers['counts']")
    elif "raw_counts" in adata.layers:
        has_counts = True
        summary["counts_in"] = "layers['raw_counts']"
        summary["checks_passed"].append("raw counts in adata.layers['raw_counts']")
    elif adata.raw is not None:
        has_counts = True
        summary["counts_in"] = "raw"
        summary["checks_passed"].append("raw counts available in adata.raw (X is normalized)")
        print("  Raw counts found in adata.raw — preprocessing will extract them automatically.")
    else:
        summary["warnings"].append(
            "Could not detect raw counts in X, layers, or adata.raw. "
            "scVI and flashSCENIC require integer counts — check the dataset manually."
        )
        print("  WARNING: raw counts not found anywhere — scVI/flashSCENIC will fail")

    # --- Check gene name / TF list compatibility ---
    tf_list_path = Path(__file__).parent / "flashscenic_data" / "allTFs_hg38.txt"
    if tf_list_path.exists():
        with open(tf_list_path) as f:
            tfs = set(line.strip() for line in f if line.strip())

        var_names = set(adata.var_names)
        overlap = var_names & tfs
        print(f"\n  Gene name / TF list check:")
        print(f"    var_names sample: {list(adata.var_names[:5])}")
        print(f"    TF list sample:   {sorted(tfs)[:5]}")
        print(f"    TF list size: {len(tfs)}, var_names size: {len(var_names)}")
        print(f"    Overlap (TFs found in var_names): {len(overlap)}")
        if len(overlap) == 0:
            # Show what the var column names look like to help diagnose
            print(f"    var columns: {list(adata.var.columns)}")
            # Check if a gene_name/symbol column exists in var
            symbol_cols = [c for c in adata.var.columns
                           if any(kw in c.lower() for kw in
                                  ["symbol", "name", "gene_name", "feature_name"])]
            rescued = False
            if symbol_cols:
                print(f"    Possible symbol columns in var: {symbol_cols}")
                overlap_via_col = set(adata.var[symbol_cols[0]]) & tfs
                print(f"    Overlap via '{symbol_cols[0]}': {len(overlap_via_col)}")
                if len(overlap_via_col) > 100:
                    # Preprocessing will rename var_names — not a blocking issue
                    summary["checks_passed"].append(
                        f"Gene symbols in var['{symbol_cols[0]}'] with {len(overlap_via_col)} TF overlaps "
                        f"— 01_preprocess.py will rename var_names automatically."
                    )
                    summary["warnings"].append(
                        f"var_names are Ensembl IDs; gene symbols in var['{symbol_cols[0]}'] "
                        f"will be used by preprocessing."
                    )
                    rescued = True
            if not rescued:
                summary["checks_failed"].append(
                    "ZERO overlap between var_names and TF list — gene names likely use "
                    "Ensembl IDs or a different convention. flashSCENIC will find 0 TFs."
                )
        else:
            summary["checks_passed"].append(
                f"{len(overlap)} TFs found in var_names — gene naming is compatible"
            )
    else:
        summary["warnings"].append(
            "allTFs_hg38.txt not found in flashscenic_data/ — "
            "run 00_prefetch_resources.sh before checking TF compatibility"
        )
        print(f"\n  TF list not cached yet — skipping gene name check")

    # --- Print summary ---
    print(f"\n  Checks passed ({len(summary['checks_passed'])}):")
    for c in summary["checks_passed"]:
        print(f"    ✓ {c}")
    if summary["checks_failed"]:
        print(f"\n  Checks FAILED ({len(summary['checks_failed'])}):")
        for c in summary["checks_failed"]:
            print(f"    ✗ {c}")
    if summary["warnings"]:
        print(f"\n  Warnings ({len(summary['warnings'])}):")
        for w in summary["warnings"]:
            print(f"    ! {w}")

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate downloaded scIB datasets")
    parser.add_argument(
        "--dataset", default="all",
        choices=list(DATASETS.keys()) + ["all"],
        help="Which dataset(s) to validate",
    )
    args = parser.parse_args()

    to_process = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    all_summaries = {}
    overall_ok = True

    for name in to_process:
        cfg = DATASETS[name]
        if not cfg["path"].exists():
            print(f"\n[missing] {cfg['path'].name} — run: bash 00_download_data.sh")
            continue  # skip silently, don't fail overall_ok for missing files

        summary = validate_dataset(name, cfg)
        all_summaries[name] = summary

        if summary["checks_failed"]:
            overall_ok = False

    # Save summary
    out_path = RESULTS_DIR / "data_summary.json"
    with open(out_path, "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print(f"\nSummary saved to {out_path}")

    # Final status
    print("\n" + "="*60)
    if overall_ok:
        print("All validation checks PASSED. Ready to run 01_preprocess.py")
    else:
        print("Some validation checks FAILED. Review data_summary.json before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
