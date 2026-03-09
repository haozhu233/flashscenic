"""
Step 4: ML biological signal preservation test.

For each method's embedding, trains an ElasticNet classifier to predict:
  1. Sex  (binary — key test: preserved by flashSCENIC, destroyed by over-correctors)
  2. Cell type  (multiclass — positive control: all methods should score high)

Uses leave-one-batch-out cross-validation:
  - Train on cells from all batches except one
  - Test on the held-out batch
  - Repeat for each batch, report mean ± std

This design proves the predictor generalizes across batches (signal is
biological, not technical), and exposes any method that destroys
cross-batch generalizable biological signal.

Output:
  results/metrics/ml_sex_{dataset}.csv
  results/metrics/ml_celltype_{dataset}.csv
  results/metrics/ml_summary_{dataset}.csv

Usage:
    python 04_ml_predictor.py [--dataset immune_human|pancreas|all]
                              [--tasks sex celltype]
"""

import argparse
import sys
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent))
from config import DATASETS, METHODS, ML, DATA_DIR, METRICS_DIR

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Embedding access
# ---------------------------------------------------------------------------

EMBEDDING_KEYS = {
    "raw_pca":     "X_raw_pca",
    "harmony":     "X_harmony",
    "scvi":        "X_scvi",
    "flashscenic": "X_flashscenic",
}


def load_embedding(adata: ad.AnnData, method: str) -> np.ndarray:
    key = EMBEDDING_KEYS[method]
    if key not in adata.obsm:
        raise KeyError(f"Embedding '{key}' not found. Run 02{method[0]}_run_{method}.py first.")
    return np.array(adata.obsm[key])


# ---------------------------------------------------------------------------
# Stratified subsampling
# ---------------------------------------------------------------------------

def subsample_cells(
    obs: pd.DataFrame,
    batch_key: str,
    cell_type_key: str,
    max_cells: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Return positional integer indices for a stratified subsample of max_cells cells.

    Stratifies by (batch × cell_type) so every combination keeps ≥1 cell,
    guaranteeing all donors and all cell types are represented after subsampling.
    """
    rng = np.random.default_rng(seed)
    total = len(obs)
    tmp = obs[[batch_key, cell_type_key]].copy()
    tmp["_pos"] = np.arange(total)
    chosen = []
    for _, grp in tmp.groupby([batch_key, cell_type_key], observed=True)["_pos"]:
        pos = grp.values
        n_target = max(1, round(max_cells * len(pos) / total))
        n_sample = min(len(pos), n_target)
        chosen.append(rng.choice(pos, size=n_sample, replace=False))
    return np.sort(np.concatenate(chosen))


# ---------------------------------------------------------------------------
# ElasticNet predictor (via LogisticRegression with elasticnet penalty)
# ---------------------------------------------------------------------------

def make_model(multiclass: bool = False) -> LogisticRegression:
    cfg = ML["elasticnet"]
    return LogisticRegression(
        penalty="elasticnet",
        C=cfg["C"],
        l1_ratio=cfg["l1_ratio"],
        solver=cfg["solver"],
        max_iter=cfg["max_iter"],
        tol=cfg["tol"],
        multi_class="multinomial" if multiclass else "auto",
        random_state=ML["seed"],
        class_weight="balanced",  # handle imbalanced classes
    )


# ---------------------------------------------------------------------------
# Regression model (age prediction)
# ---------------------------------------------------------------------------

def make_regression_model() -> ElasticNet:
    cfg = ML["elasticnet"]
    return ElasticNet(
        alpha=1.0 / cfg["C"],
        l1_ratio=cfg["l1_ratio"],
        max_iter=cfg["max_iter"],
        tol=cfg["tol"],
        random_state=ML["seed"],
    )


# ---------------------------------------------------------------------------
# Leave-one-batch-out evaluation
# ---------------------------------------------------------------------------

def _is_single_sex_per_batch(y_sex: np.ndarray, batch_labels: np.ndarray) -> bool:
    """Return True if every batch contains only one sex class."""
    for b in np.unique(batch_labels):
        if len(np.unique(y_sex[batch_labels == b])) > 1:
            return False
    return True


def sex_grouped_kfold_eval(
    X: np.ndarray,
    y_sex: np.ndarray,
    batch_labels: np.ndarray,
    method: str,
    dataset: str,
    n_splits: int = 5,
    seed: int = 42,
) -> list[dict]:
    """
    Sex prediction CV for single-sex-per-batch datasets (e.g. one donor = one sex).

    Splits donors by sex, then for each fold holds out ~1/K female donors
    and ~1/K male donors so that both classes always appear in test.
    """
    rng = np.random.default_rng(seed)
    batches = np.unique(batch_labels)

    # Map batch → sex (single-sex per batch, so take first cell's label)
    female_batches = [b for b in batches if np.unique(y_sex[batch_labels == b])[0] == 0]
    male_batches   = [b for b in batches if np.unique(y_sex[batch_labels == b])[0] == 1]

    rng.shuffle(female_batches := np.array(female_batches))
    rng.shuffle(male_batches   := np.array(male_batches))

    k = min(n_splits, len(female_batches), len(male_batches))
    print(f"      [sex-grouped-kfold] {len(female_batches)} F batches, "
          f"{len(male_batches)} M batches → {k} folds")

    rows = []
    for fold in range(k):
        test_f = female_batches[fold::k]
        test_m = male_batches[fold::k]
        test_batches = np.concatenate([test_f, test_m])

        test_mask  = np.isin(batch_labels, test_batches)
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y_sex[train_mask]
        X_test,  y_test  = X[test_mask],  y_sex[test_mask]

        if len(np.unique(y_test)) < 2 or len(np.unique(y_train)) < 2:
            print(f"      [skip] fold {fold}: degenerate class split")
            continue

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        model = make_model(multiclass=False)
        try:
            model.fit(X_train_sc, y_train)
        except Exception as e:
            print(f"      [error] fold {fold}: {e}")
            continue

        y_pred  = model.predict(X_test_sc)
        y_proba = model.predict_proba(X_test_sc)[:, 1]
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        try:
            auroc = roc_auc_score(y_test, y_proba)
        except ValueError:
            auroc = np.nan

        n_test_f = test_mask.sum() - (batch_labels[test_mask] == test_m[0]).sum() \
                   if len(test_m) else test_mask.sum()
        print(f"      Fold {fold}: AUROC={auroc:.3f}, BalAcc={bal_acc:.3f} "
              f"(n_train={train_mask.sum():,}, n_test={test_mask.sum():,}, "
              f"test_donors F={len(test_f)} M={len(test_m)})")

        rows.append({
            "dataset": dataset,
            "method": method,
            "task": "sex",
            "held_out_batch": f"fold_{fold}",
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "AUROC": float(auroc) if not np.isnan(auroc) else None,
            "balanced_accuracy": float(bal_acc),
        })

    return rows


def lobo_eval(
    X: np.ndarray,
    y: np.ndarray,
    batch_labels: np.ndarray,
    task: str,          # "sex" or "celltype"
    method: str,
    dataset: str,
) -> list[dict]:
    """
    Leave-One-Batch-Out cross-validation.

    Returns a list of per-fold result dicts.
    """
    batches = np.unique(batch_labels)
    multiclass = task == "celltype"
    rows = []


    for held_out_batch in batches:
        train_mask = batch_labels != held_out_batch
        test_mask = batch_labels == held_out_batch

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # Skip fold if too few test samples or degenerate classes
        if len(X_test) < ML["min_cells_per_class"]:
            continue
        if len(np.unique(y_test)) < 2:
            print(f"      [skip] held-out batch '{held_out_batch}': "
                  f"only {len(np.unique(y_test))} class(es) in test set")
            continue
        # For binary sex task: require both classes in test
        if not multiclass and len(np.unique(y_test)) < 2:
            continue

        # Scale embedding (fit on train, apply to test)
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        model = make_model(multiclass=multiclass)
        try:
            model.fit(X_train_sc, y_train)
        except Exception as e:
            print(f"      [error] fold '{held_out_batch}': {e}")
            continue

        y_pred = model.predict(X_test_sc)
        bal_acc = balanced_accuracy_score(y_test, y_pred)

        # AUROC
        if multiclass:
            # Need all classes present in train; skip classes absent from train
            classes_in_train = np.unique(y_train)
            if len(classes_in_train) < 2:
                auroc = np.nan
            else:
                y_proba = model.predict_proba(X_test_sc)
                # Only use columns for classes present in train
                # Filter test samples to only classes in train
                test_mask_classes = np.isin(y_test, classes_in_train)
                if test_mask_classes.sum() < 2:
                    auroc = np.nan
                else:
                    # Intersect labels with classes actually present in filtered test
                    y_test_filtered = y_test[test_mask_classes]
                    classes_in_test = np.intersect1d(classes_in_train,
                                                     np.unique(y_test_filtered))
                    if len(classes_in_test) < 2:
                        auroc = np.nan
                    else:
                        idx_in_model = [
                            np.where(model.classes_ == c)[0][0]
                            for c in classes_in_test
                            if c in model.classes_
                        ]
                        keep = np.isin(y_test_filtered, classes_in_test)
                        try:
                            proba_subset = y_proba[test_mask_classes][keep][:, idx_in_model]
                            # Renormalise rows so they sum to 1 (required after
                            # subsetting a k-of-16 class softmax to k columns)
                            row_sums = proba_subset.sum(axis=1, keepdims=True)
                            row_sums = np.where(row_sums == 0, 1, row_sums)
                            proba_subset = proba_subset / row_sums
                            auroc = roc_auc_score(
                                y_test_filtered[keep],
                                proba_subset,
                                multi_class="ovr",
                                average="macro",
                                labels=classes_in_test,
                            )
                        except ValueError as e:
                            print(f"      [auroc-warn] {e}")
                            auroc = np.nan
        else:
            y_proba = model.predict_proba(X_test_sc)[:, 1]
            try:
                auroc = roc_auc_score(y_test, y_proba)
            except ValueError:
                auroc = np.nan

        rows.append({
            "dataset": dataset,
            "method": method,
            "task": task,
            "held_out_batch": str(held_out_batch),
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "AUROC": float(auroc) if not np.isnan(auroc) else None,
            "balanced_accuracy": float(bal_acc),
        })

        print(f"      Batch '{held_out_batch}': "
              f"AUROC={auroc:.3f}, BalAcc={bal_acc:.3f} "
              f"(n_train={train_mask.sum():,}, n_test={test_mask.sum():,})")

    return rows


def lobo_age_eval(
    X: np.ndarray,
    y_age: np.ndarray,
    batch_labels: np.ndarray,
    method: str,
    dataset: str,
) -> list[dict]:
    """
    LOBO regression for donor age prediction.

    Each held-out batch is one donor (all cells share the same age).
    Fits ElasticNet on training donors, averages cell predictions for the
    held-out donor, and collects (true_age, predicted_age) per donor.
    """
    batches = np.unique(batch_labels)
    rows = []

    for held_out in batches:
        train_mask = batch_labels != held_out
        test_mask  = batch_labels == held_out
        if test_mask.sum() < 2:
            continue

        X_train, y_train = X[train_mask], y_age[train_mask]
        X_test = X[test_mask]
        true_age = float(y_age[test_mask][0])   # same value for all cells in donor

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        model = make_regression_model()
        try:
            model.fit(X_train_sc, y_train)
        except Exception as e:
            print(f"      [error] donor '{held_out}': {e}")
            continue

        pred_age = float(model.predict(X_test_sc).mean())
        abs_err  = abs(pred_age - true_age)

        print(f"      Donor '{held_out}': true={true_age:.0f}, "
              f"pred={pred_age:.1f}, |err|={abs_err:.1f}")

        rows.append({
            "dataset": dataset,
            "method": method,
            "task": "age",
            "held_out_batch": str(held_out),
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "true_age": true_age,
            "predicted_age": pred_age,
            "abs_error": abs_err,
        })

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_task(
    name: str,
    cfg: dict,
    task: str,
    adata: ad.AnnData,
    subsample_idx: np.ndarray | None = None,
) -> pd.DataFrame:
    """Run one task (sex or celltype) for all methods on one dataset."""
    batch_key = cfg["batch_key"]
    batch_labels = pd.Categorical(adata.obs[batch_key]).codes
    if subsample_idx is not None:
        batch_labels = batch_labels[subsample_idx]

    if task == "sex":
        sex_key = cfg["sex_key"]
        if sex_key is None or sex_key not in adata.obs.columns:
            print(f"  [skip] sex task: sex_key '{sex_key}' not available for {name}")
            return pd.DataFrame()
        le = LabelEncoder()
        y = le.fit_transform(adata.obs[sex_key])
        if subsample_idx is not None:
            y = y[subsample_idx]
        print(f"  Sex labels: {dict(zip(le.classes_, np.bincount(y)))}")
    elif task == "age":
        age_key = cfg.get("age_key")
        if not age_key or age_key not in adata.obs.columns:
            print(f"  [skip] age task: age_key '{age_key}' not available for {name}")
            return pd.DataFrame()
        y = pd.to_numeric(adata.obs[age_key], errors="coerce").values
        if subsample_idx is not None:
            y = y[subsample_idx]
        print(f"  Age range: {np.nanmin(y):.0f}–{np.nanmax(y):.0f} years "
              f"({np.sum(np.isnan(y))} NaN)")
    else:
        ct_key = cfg["cell_type_key"]
        le = LabelEncoder()
        y = le.fit_transform(adata.obs[ct_key])
        if subsample_idx is not None:
            y = y[subsample_idx]
        print(f"  Cell types: {len(le.classes_)} classes")

    # For sex prediction: detect if every batch is single-sex → use grouped K-fold
    use_grouped_kfold = (
        task == "sex"
        and _is_single_sex_per_batch(y, batch_labels)
    )
    if use_grouped_kfold:
        print("  [sex-grouped-kfold] Every batch is single-sex — switching to "
              "grouped K-fold CV (holds out balanced male+female donors per fold)")

    all_rows = []
    for method in METHODS:
        try:
            X = load_embedding(adata, method)
        except KeyError as e:
            print(f"  [skip] {method}: {e}")
            continue

        if subsample_idx is not None:
            X = X[subsample_idx]
        print(f"\n  Method: {method} ({X.shape})")
        if task == "age":
            rows = lobo_age_eval(X, y, batch_labels, method=method, dataset=name)
        elif use_grouped_kfold:
            rows = sex_grouped_kfold_eval(
                X, y, batch_labels,
                method=method, dataset=name,
                n_splits=ML.get("sex_n_splits", 5),
                seed=ML["seed"],
            )
        else:
            rows = lobo_eval(X, y, batch_labels, task=task, method=method, dataset=name)
        all_rows.extend(rows)

    return pd.DataFrame(all_rows)


def summarize(df_sex: pd.DataFrame, df_ct: pd.DataFrame, df_age: pd.DataFrame,
              name: str) -> pd.DataFrame:
    """Aggregate per-fold results into mean ± std per method."""
    rows = []
    for method in METHODS:
        row = {"dataset": name, "method": method}

        for task_name, df in [("sex", df_sex), ("celltype", df_ct)]:
            if df.empty:
                row[f"{task_name}_AUROC_mean"] = np.nan
                row[f"{task_name}_AUROC_std"] = np.nan
                row[f"{task_name}_BalAcc_mean"] = np.nan
                row[f"{task_name}_BalAcc_std"] = np.nan
                continue

            subset = df[df.method == method]
            if subset.empty:
                row[f"{task_name}_AUROC_mean"] = np.nan
                row[f"{task_name}_AUROC_std"] = np.nan
                row[f"{task_name}_BalAcc_mean"] = np.nan
                row[f"{task_name}_BalAcc_std"] = np.nan
            else:
                row[f"{task_name}_AUROC_mean"] = subset["AUROC"].mean()
                row[f"{task_name}_AUROC_std"] = subset["AUROC"].std()
                row[f"{task_name}_BalAcc_mean"] = subset["balanced_accuracy"].mean()
                row[f"{task_name}_BalAcc_std"] = subset["balanced_accuracy"].std()

        # Age regression: MedAE and Pearson R across donors
        age_rows = df_age[df_age.method == method] if not df_age.empty else pd.DataFrame()
        if not age_rows.empty and len(age_rows) >= 3:
            trues = age_rows["true_age"].values
            preds = age_rows["predicted_age"].values
            row["age_MedAE"] = float(np.median(np.abs(preds - trues)))
            r, _ = pearsonr(trues, preds)
            row["age_R"] = float(r)
        else:
            row["age_MedAE"] = np.nan
            row["age_R"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all",
                        choices=list(DATASETS.keys()) + ["all"])
    parser.add_argument("--tasks", nargs="+", default=["sex", "celltype", "age"],
                        choices=["sex", "celltype", "age"])
    args = parser.parse_args()

    to_process = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    for name in to_process:
        print(f"\n{'='*60}")
        print(f"ML predictor: {name}")
        print(f"{'='*60}")

        preprocessed_path = DATA_DIR / f"preprocessed_{name}.h5ad"
        if not preprocessed_path.exists():
            print(f"  [skip] {preprocessed_path.name} not found.")
            continue

        adata = ad.read_h5ad(preprocessed_path, backed="r")

        # Subsample large datasets while preserving all batches and cell types
        n_cells = adata.n_obs
        max_cells = ML.get("max_cells")
        subsample_idx = None
        if max_cells and n_cells > max_cells:
            cfg_ds = DATASETS[name]
            subsample_idx = subsample_cells(
                adata.obs,
                batch_key=cfg_ds["batch_key"],
                cell_type_key=cfg_ds["cell_type_key"],
                max_cells=max_cells,
                seed=ML["seed"],
            )
            print(f"  Subsampled {n_cells:,} → {len(subsample_idx):,} cells "
                  f"(stratified by batch × cell type, target={max_cells:,})")

        df_sex = pd.DataFrame()
        df_ct = pd.DataFrame()
        df_age = pd.DataFrame()

        if "sex" in args.tasks:
            print(f"\n--- Task: sex prediction ---")
            df_sex = run_task(name, DATASETS[name], "sex", adata, subsample_idx)
            if not df_sex.empty:
                out = METRICS_DIR / f"ml_sex_{name}.csv"
                df_sex.to_csv(out, index=False)
                print(f"\n  Sex results saved to {out.name}")

        if "celltype" in args.tasks:
            print(f"\n--- Task: cell type prediction (positive control) ---")
            df_ct = run_task(name, DATASETS[name], "celltype", adata, subsample_idx)
            if not df_ct.empty:
                out = METRICS_DIR / f"ml_celltype_{name}.csv"
                df_ct.to_csv(out, index=False)
                print(f"\n  Cell type results saved to {out.name}")

        if "age" in args.tasks:
            print(f"\n--- Task: age prediction (donor-level regression) ---")
            df_age = run_task(name, DATASETS[name], "age", adata, subsample_idx)
            if not df_age.empty:
                out = METRICS_DIR / f"ml_age_{name}.csv"
                df_age.to_csv(out, index=False)
                print(f"\n  Age results saved to {out.name}")

        # Aggregate summary
        df_summary = summarize(df_sex, df_ct, df_age, name)
        out = METRICS_DIR / f"ml_summary_{name}.csv"
        df_summary.to_csv(out, index=False)

        print(f"\n  Summary ({name}):")
        pd.set_option("display.float_format", "{:.3f}".format)
        print(df_summary.set_index("method").drop(columns=["dataset"]).to_string())

    print("\nML predictor done. Run 05_visualize.py next.")


if __name__ == "__main__":
    main()
