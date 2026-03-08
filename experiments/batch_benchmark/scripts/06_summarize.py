"""
Step 6: Generate final summary report.

Reads all metric CSVs and prints a ranked table for each dataset.
Flags anomalies and writes results/SUMMARY.md.

Usage:
    python 06_summarize.py [--dataset immune_human|pancreas|all]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import DATASETS, METHODS, METRICS_DIR, RESULTS_DIR

METHOD_LABELS = {
    "raw_pca":     "Raw PCA",
    "harmony":     "Harmony",
    "scvi":        "scVI",
    "flashscenic": "flashSCENIC",
}


def load_or_empty(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def summarize_dataset(name: str) -> list[str]:
    lines = []

    def h(text, level=2):
        lines.append("#" * level + " " + text)
        lines.append("")

    def p(text):
        lines.append(text)
        lines.append("")

    h(f"Dataset: {name}", level=2)

    # --- scIB metrics ---
    scib_df = load_or_empty(METRICS_DIR / f"scib_scores_{name}.csv")
    if not scib_df.empty:
        h("scIB Integration Metrics", level=3)
        cols = ["method", "iLISI", "ASW_batch", "cLISI", "ASW_cell_type",
                "NMI", "ARI", "batch_score", "bio_score", "overall_score"]
        cols = [c for c in cols if c in scib_df.columns]
        display = scib_df[cols].copy()
        display["method"] = display["method"].map(lambda m: METHOD_LABELS.get(m, m))
        display = display.sort_values("overall_score", ascending=False)
        lines.append("```")
        lines.append(display.set_index("method").round(3).to_string())
        lines.append("```")
        lines.append("")

        winner = display.iloc[0]["method"]
        p(f"Best overall scIB score: **{winner}**")
    else:
        p("scIB metrics not available (run 03_compute_metrics.py).")

    # --- ML predictor ---
    ml_df = load_or_empty(METRICS_DIR / f"ml_summary_{name}.csv")
    if not ml_df.empty:
        h("ML Biological Signal Preservation", level=3)
        ml_df["method_label"] = ml_df["method"].map(lambda m: METHOD_LABELS.get(m, m))

        sex_cols = [c for c in ml_df.columns if "sex" in c]
        ct_cols = [c for c in ml_df.columns if "celltype" in c]

        if sex_cols:
            h("Sex Prediction (key test)", level=4)
            sex_display = ml_df[["method_label"] + sex_cols].set_index("method_label")
            sex_display = sex_display.sort_values("sex_AUROC_mean", ascending=False)
            lines.append("```")
            lines.append(sex_display.round(3).to_string())
            lines.append("```")
            lines.append("")

            sex_aurocs = ml_df.set_index("method")["sex_AUROC_mean"]
            if "flashscenic" in sex_aurocs.index and "scvi" in sex_aurocs.index:
                fs_auroc = sex_aurocs["flashscenic"]
                scvi_auroc = sex_aurocs["scvi"]
                delta = fs_auroc - scvi_auroc
                if not np.isnan(delta):
                    direction = "higher" if delta > 0 else "lower"
                    p(f"flashSCENIC sex AUROC is **{abs(delta):.3f} {direction}** than scVI "
                      f"({fs_auroc:.3f} vs {scvi_auroc:.3f}).")

            # Anomaly check
            best_sex = sex_aurocs.max()
            for method in METHODS:
                if method in sex_aurocs.index:
                    auroc = sex_aurocs[method]
                    if not np.isnan(auroc) and auroc < 0.55:
                        p(f"⚠️ **Anomaly**: {METHOD_LABELS.get(method, method)} sex AUROC "
                          f"= {auroc:.3f} (near random). Integration may be over-correcting.")

        if ct_cols:
            h("Cell Type Prediction (positive control)", level=4)
            ct_display = ml_df[["method_label"] + ct_cols].set_index("method_label")
            ct_display = ct_display.sort_values("celltype_AUROC_mean", ascending=False)
            lines.append("```")
            lines.append(ct_display.round(3).to_string())
            lines.append("```")
            lines.append("")

            ct_aurocs = ml_df.set_index("method")["celltype_AUROC_mean"]
            for method in METHODS:
                if method in ct_aurocs.index:
                    auroc = ct_aurocs[method]
                    if not np.isnan(auroc) and auroc < 0.70:
                        p(f"⚠️ **Anomaly**: {METHOD_LABELS.get(method, method)} cell type AUROC "
                          f"= {auroc:.3f} (below 0.70). Cell type information may be destroyed.")
    else:
        p("ML predictor results not available (run 04_ml_predictor.py).")

    # --- Interpretation ---
    h("Interpretation", level=3)
    if not scib_df.empty and not ml_df.empty:
        lines.append(
            "The experiment tests whether flashSCENIC (regulon activity space) achieves "
            "soft batch correction: removing technical variation while preserving "
            "biological signal.\n"
        )
        lines.append("Key claims to evaluate:")
        lines.append(
            "1. **Batch mixing** (iLISI, ASW_batch): flashSCENIC should score comparably "
            "to or better than raw PCA, but may not reach scVI's level of explicit correction."
        )
        lines.append(
            "2. **Sex signal preservation** (sex AUROC): flashSCENIC should retain more "
            "cross-batch sex signal than scVI. A drop in sex AUROC after scVI integration "
            "demonstrates over-correction destroying biological variance."
        )
        lines.append(
            "3. **Cell type preservation** (celltype AUROC): All methods should score high. "
            "This is the positive control. Failure here indicates a broken integration."
        )
        lines.append("")

    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all",
                        choices=list(DATASETS.keys()) + ["all"])
    args = parser.parse_args()

    to_process = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    all_lines = ["# Batch Integration Benchmark — Summary Report", ""]

    for name in to_process:
        print(f"\n{'='*60}")
        print(f"Summary: {name}")
        print(f"{'='*60}")
        section = summarize_dataset(name)
        all_lines.extend(section)

        # Also print to stdout
        for line in section:
            print(line)

    out_path = RESULTS_DIR / "SUMMARY.md"
    with open(out_path, "w") as f:
        f.write("\n".join(all_lines))
    print(f"\nFull summary written to {out_path}")


if __name__ == "__main__":
    main()
