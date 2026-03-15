"""
Step 6: Generate summary report for flashSCENIC downstream analysis.

Covers five analysis areas for the ALS spinal cord dataset:
  1. scIB integration metrics (Raw PCA vs TF activity)
  2. Regulon Specificity Scores (top TFs per cell type)
  3. ML sex prediction (control) — cell-level and metacell
  4. ML disease prediction — cell-level and metacell + model coefficients
  5. DE TFs — metacell-level MWU between ALS and controls

Usage:
    python 06_summarize.py [--dataset als_spinal_cord]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _md_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a simple Markdown table (no tabulate dependency)."""
    cols = list(df.columns)
    widths = [max(len(str(c)), max((len(str(v)) for v in df[c]), default=0)) for c in cols]
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    header = "| " + " | ".join(str(c).ljust(w) for c, w in zip(cols, widths)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[c]).ljust(w) for c, w in zip(cols, widths)) + " |")
    return "\n".join([header, sep] + rows)

sys.path.insert(0, str(Path(__file__).parent))
from config import DATASETS, METRICS_DIR, RESULTS_DIR

METHOD_LABELS = {
    "raw_pca":     "Raw PCA",
    "flashscenic": "TF Activity",
}


def load_or_empty(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Individual sections
# ---------------------------------------------------------------------------

def section_scib(name: str) -> list[str]:
    lines = []
    scib_df = load_or_empty(METRICS_DIR / f"scib_scores_{name}.csv")
    if scib_df.empty:
        lines.append("_scIB metrics not available (run 03_compute_metrics.py)._\n")
        return lines

    scib_df = scib_df[scib_df["method"].isin(["raw_pca", "flashscenic"])].copy()
    cols = ["method", "iLISI", "ASW_batch", "cLISI", "ASW_cell_type",
            "NMI", "ARI", "batch_score", "bio_score", "overall_score"]
    cols = [c for c in cols if c in scib_df.columns]
    scib_df["method"] = scib_df["method"].map(lambda m: METHOD_LABELS.get(m, m))
    scib_df = scib_df.sort_values("overall_score", ascending=False)

    lines.append("```")
    lines.append(scib_df[cols].set_index("method").round(3).to_string())
    lines.append("```\n")

    if len(scib_df) == 2:
        row_fs = scib_df[scib_df["method"] == "TF Activity"].iloc[0]
        row_pca = scib_df[scib_df["method"] == "Raw PCA"].iloc[0]
        winner = scib_df.iloc[0]["method"]
        lines.append(f"Best overall: **{winner}** "
                     f"(TF Activity {row_fs['overall_score']:.3f} vs "
                     f"Raw PCA {row_pca['overall_score']:.3f}).")
        if "NMI" in row_fs and "ARI" in row_fs:
            lines.append(f"Cell-type clustering: NMI {row_fs['NMI']:.3f} / ARI {row_fs['ARI']:.3f} "
                         f"(TF Activity) vs NMI {row_pca['NMI']:.3f} / ARI {row_pca['ARI']:.3f} "
                         f"(Raw PCA).")
    lines.append("")
    return lines


def section_rss(name: str) -> list[str]:
    lines = []
    rss_df = load_or_empty(METRICS_DIR / f"{name}_rss.csv")
    if rss_df.empty:
        lines.append("_RSS not available (run 05_visualize.py)._\n")
        return lines

    # Wide format: first column = cell_type, rest = TF RSS scores
    ct_col = rss_df.columns[0]
    tf_cols = [c for c in rss_df.columns if c != ct_col]

    rows = []
    for _, row in rss_df.iterrows():
        ct = row[ct_col]
        scores = row[tf_cols].astype(float)
        top5 = scores.nlargest(5).index.tolist()
        rows.append({"Cell type": ct, "Top 5 regulons (RSS)": ", ".join(top5)})

    table = pd.DataFrame(rows)
    lines.append(_md_table(table))
    lines.append("")
    return lines


def _ml_auroc_row(df: pd.DataFrame, method: str) -> str:
    """Format 'mean ± std' from a ML result DataFrame for one method."""
    sub = df[df["method"] == method]
    if sub.empty:
        return "n/a"
    row = sub.iloc[0]
    auroc = row.get("test_AUROC", np.nan)
    std   = row.get("test_AUROC_std", np.nan)
    if np.isnan(auroc):
        return "n/a"
    if np.isnan(std):
        return f"{auroc:.3f}"
    return f"{auroc:.3f} ± {std:.3f}"


def section_ml_sex(name: str) -> list[str]:
    lines = []
    cell_df = load_or_empty(METRICS_DIR / f"ml_sex_{name}.csv")
    meta_df = load_or_empty(METRICS_DIR / f"ml_sex_{name}_metacells.csv")

    if cell_df.empty:
        lines.append("_Sex prediction results not available (run 04_ml_predictor.py)._\n")
        return lines

    rows = []
    rows.append({"Model": "Raw PCA (cell-level)",
                 "AUROC (mean ± std)": _ml_auroc_row(cell_df, "raw_pca")})
    rows.append({"Model": "TF Activity (cell-level)",
                 "AUROC (mean ± std)": _ml_auroc_row(cell_df, "flashscenic")})
    if not meta_df.empty:
        rows.append({"Model": "TF Activity (metacell)",
                     "AUROC (mean ± std)": _ml_auroc_row(meta_df, "flashscenic")})

    lines.append(_md_table(pd.DataFrame(rows)))
    lines.append("\n_Sex is a sanity-check control: should be robustly predictable from TF activity._\n")
    return lines


def section_ml_disease(name: str) -> list[str]:
    lines = []
    cell_df = load_or_empty(METRICS_DIR / f"ml_disease_{name}.csv")
    meta_df = load_or_empty(METRICS_DIR / f"ml_disease_{name}_metacells.csv")

    if cell_df.empty:
        lines.append("_Disease prediction results not available (run 04_ml_predictor.py)._\n")
        return lines

    rows = []
    rows.append({"Model": "Raw PCA (cell-level)",
                 "AUROC (mean ± std)": _ml_auroc_row(cell_df, "raw_pca")})
    rows.append({"Model": "TF Activity (cell-level)",
                 "AUROC (mean ± std)": _ml_auroc_row(cell_df, "flashscenic")})
    if not meta_df.empty:
        rows.append({"Model": "TF Activity (metacell)",
                     "AUROC (mean ± std)": _ml_auroc_row(meta_df, "flashscenic")})

    lines.append(_md_table(pd.DataFrame(rows)))
    lines.append("")
    return lines


def section_ml_coef(name: str) -> list[str]:
    lines = []
    # Prefer metacell coefficients; fall back to cell-level
    coef_path = METRICS_DIR / f"ml_coef_{name}_flashscenic_disease_metacells.csv"
    label = "metacell"
    if not coef_path.exists():
        coef_path = METRICS_DIR / f"ml_coef_{name}_flashscenic_disease.csv"
        label = "cell-level"

    if not coef_path.exists():
        lines.append("_Disease model coefficients not available._\n")
        return lines

    coef_df = pd.read_csv(coef_path).sort_values("coefficient")
    class_0 = coef_df["class_0"].iloc[0] if "class_0" in coef_df.columns else "control"
    class_1 = coef_df["class_1"].iloc[0] if "class_1" in coef_df.columns else "disease"

    neg = coef_df[coef_df["coefficient"] < 0].tail(10)[["feature", "coefficient"]]
    pos = coef_df[coef_df["coefficient"] > 0].tail(10)[["feature", "coefficient"]]

    lines.append(f"_Source: {label} ElasticNet refit on all donors._\n")

    if not pos.empty:
        lines.append(f"**Disease-associated TFs** (↑ {class_1}):\n")
        lines.append(_md_table(pos[::-1].reset_index(drop=True)))
        lines.append("")

    if not neg.empty:
        lines.append(f"**Control-associated TFs** (↑ {class_0}):\n")
        lines.append(_md_table(neg.reset_index(drop=True)))
        lines.append("")

    return lines


def section_ml_per_ct(name: str) -> list[str]:
    lines = []
    cell_df = load_or_empty(METRICS_DIR / f"ml_disease_per_ct_{name}.csv")
    meta_df = load_or_empty(METRICS_DIR / f"ml_disease_per_ct_{name}_metacells.csv")

    if cell_df.empty:
        lines.append("_Per-cell-type disease prediction not available (run 04_ml_predictor.py)._\n")
        return lines

    cell_types = sorted(cell_df["cell_type"].unique())
    rows = []
    for ct in cell_types:
        row = {"Cell type": ct}
        # Raw PCA cell-level
        sub = cell_df[(cell_df["cell_type"] == ct) & (cell_df["method"] == "raw_pca")]
        row["Raw PCA"] = _ml_auroc_row(sub, "raw_pca") if not sub.empty else "n/a"
        # flashscenic cell-level
        sub = cell_df[(cell_df["cell_type"] == ct) & (cell_df["method"] == "flashscenic")]
        row["TF Activity (cell)"] = _ml_auroc_row(sub, "flashscenic") if not sub.empty else "n/a"
        # flashscenic metacell
        if not meta_df.empty:
            sub = meta_df[(meta_df["cell_type"] == ct) & (meta_df["method"] == "flashscenic")]
            row["TF Activity (metacell)"] = _ml_auroc_row(sub, "flashscenic") if not sub.empty else "n/a"
        rows.append(row)

    lines.append(_md_table(pd.DataFrame(rows)))
    lines.append("")
    return lines


def section_de_tfs(name: str) -> list[str]:
    lines = []
    meta_dir = METRICS_DIR / "metacell"
    all_csv = meta_dir / f"de_tfs_{name}_all.csv"

    if not all_csv.exists():
        lines.append("_DE TF (metacell) results not available (run 06_de_tfs.py)._\n")
        return lines

    all_df = pd.read_csv(all_csv)
    n_sig_all = (all_df["qval"] < 0.05).sum()
    lines.append(f"Overall sig TFs (q < 0.05, pooled across cell types): **{n_sig_all}**\n")

    # Per cell type
    ct_files = sorted(meta_dir.glob(f"de_tfs_{name}_*.csv"))
    ct_files = [f for f in ct_files if f.name != f"de_tfs_{name}_all.csv"]

    if ct_files:
        rows = []
        for f in ct_files:
            ct = f.stem.replace(f"de_tfs_{name}_", "").replace("_", " ")
            df = pd.read_csv(f)
            sig = df[df["qval"] < 0.05]
            n_sig = len(sig)
            if n_sig == 0:
                rows.append({"Cell type": ct, "n_sig": 0, "Top ↑ ALS": "—", "Top ↑ ctrl": "—"})
                continue
            up   = sig[sig["delta_mean"] > 0].nlargest(3, "delta_mean")["TF"].tolist()
            down = sig[sig["delta_mean"] < 0].nsmallest(3, "delta_mean")["TF"].tolist()
            rows.append({
                "Cell type": ct,
                "n_sig (q<0.05)": n_sig,
                "Top ↑ ALS": ", ".join(up) if up else "—",
                "Top ↑ ctrl": ", ".join(down) if down else "—",
            })

        if rows:
            lines.append(_md_table(pd.DataFrame(rows)))
            lines.append("")

    return lines


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def summarize_dataset(name: str) -> list[str]:
    lines = []

    def h(text, level=2):
        lines.append("#" * level + " " + text)
        lines.append("")

    def extend(section_lines):
        lines.extend(section_lines)

    h(f"Dataset: {name}", level=2)

    h("1. scIB Integration Metrics", level=3)
    extend(section_scib(name))

    h("2. Regulon Specificity Scores (top 5 per cell type)", level=3)
    extend(section_rss(name))

    h("3. ML — Sex Prediction (sanity control)", level=3)
    extend(section_ml_sex(name))

    h("4. ML — Disease Prediction (ALS vs control)", level=3)
    extend(section_ml_disease(name))

    h("5. ML — Disease Prediction Per Cell Type", level=3)
    extend(section_ml_per_ct(name))

    h("6. Disease Model Coefficients (TF Activity, top 10 per direction)", level=3)
    extend(section_ml_coef(name))

    h("7. Differential TF Activity — Metacell-level MWU", level=3)
    extend(section_de_tfs(name))

    h("Interpretation", level=3)
    lines.append(
        "TF activity (flashSCENIC) is evaluated as a biologically structured representation "
        "of single-cell gene regulatory networks. Key questions:\n"
    )
    lines.append(
        "1. **Representation quality** (scIB): Does TF activity better separate cell types "
        "while mixing donors compared to raw PCA?"
    )
    lines.append(
        "2. **Biological signal** (ML sex): Is donor sex robustly encoded in TF activity? "
        "High AUROC confirms the representation retains cross-donor biological variance."
    )
    lines.append(
        "3. **Disease signal** (ML disease): Can TF activity discriminate ALS from controls? "
        "Metacell aggregation reduces pseudoreplication and provides donor-level estimates."
    )
    lines.append(
        "4. **DE TFs** (metacell MWU): Cell-level MWU inflates significance (cells are not "
        "independent); pseudobulk is underpowered with 12 donors. Metacells per donor+cell-type "
        "balance power and independence. Significant TFs represent candidate disease-relevant "
        "regulons."
    )
    lines.append("")

    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="als_spinal_cord",
                        choices=list(DATASETS.keys()))
    args = parser.parse_args()

    name = args.dataset

    print(f"\n{'='*60}")
    print(f"Summary: {name}")
    print(f"{'='*60}")

    all_lines = ["# flashSCENIC Downstream Analysis — ALS Spinal Cord", ""]
    section = summarize_dataset(name)
    all_lines.extend(section)

    for line in section:
        print(line)

    out_path = RESULTS_DIR / "SUMMARY.md"
    with open(out_path, "w") as f:
        f.write("\n".join(all_lines))
    print(f"\nSummary written to {out_path}")


if __name__ == "__main__":
    main()
