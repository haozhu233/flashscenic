"""
Step 4: ML biological signal preservation test.

For each method's embedding, trains an ElasticNet model to predict:
  1. Sex     (binary classification)
  2. Age     (regression, evaluated at donor level)
  3. Disease (binary classification — ALS vs healthy, where available)

Uses 6-fold cross-validation with balanced donor-level folds:
  - Disease task: each val fold has exactly 1 disease + 1 healthy donor.
  - Sex task:     each val fold has exactly 1 male + 1 female donor.
  - Age task:     folds balanced by sex (proxy).

Nested CV: for each outer fold, a 5-fold inner GridSearchCV tunes
ElasticNet hyperparameters (C × l1_ratio, 30 combinations).

After outer CV, the best hyperparams (from the highest-val-AUROC fold) are
used to refit a model on ALL donors; those coefficients are saved for flashSCENIC.

Reports mean ± std across 6 folds.

Output:
  results/metrics/ml_sex_{dataset}.csv
  results/metrics/ml_age_{dataset}.csv
  results/metrics/ml_disease_{dataset}.csv
  results/metrics/ml_summary_{dataset}.csv
  results/metrics/ml_coef_{dataset}_flashscenic_{task}.csv  (flashSCENIC only)

Usage:
    python 04_ml_predictor.py [--dataset als_motor_cortex|als_spinal_cord|all]
                              [--tasks sex age disease]
"""

import argparse
import sys
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from config import DATASETS, METHODS, ML, MIN_CELLS_PER_CT, DATA_DIR, METRICS_DIR, EMBEDDINGS_DIR, PREPROCESS

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

EMBEDDING_KEYS = {
    "raw_pca":     "X_raw_pca",
    "harmony":     "X_harmony",
    "scvi":        "X_scvi",
    "flashscenic": "X_flashscenic",
}

# Hyperparameter grid for inner CV (30 combinations)
PARAM_GRID_C        = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]  # 6
PARAM_GRID_L1RATIO  = [0.1, 0.3, 0.5, 0.7, 0.9]             # 5
N_OUTER_FOLDS  = 6
N_INNER_FOLDS  = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_feature_names(name: str, method: str) -> list[str] | None:
    """Return regulon names for flashSCENIC, None for other methods."""
    if method != "flashscenic":
        return None
    reg_path = EMBEDDINGS_DIR / f"{name}_flashscenic_regulon_names.npy"
    if not reg_path.exists():
        return None
    return np.load(reg_path, allow_pickle=True).tolist()


def parse_age(series: pd.Series) -> np.ndarray:
    """Parse age from numeric values or CellxGene 'XX-year-old stage' strings."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.5:
        return numeric.values
    parsed = series.str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(parsed, errors="coerce").values


# ---------------------------------------------------------------------------
# Fold construction
# ---------------------------------------------------------------------------

def make_cv_folds(
    obs: pd.DataFrame,
    batch_key: str,
    stratify_key: str,
    sex_key: str | None = None,
    n_splits: int = N_OUTER_FOLDS,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Donor-stratified k-fold CV.

    Each validation fold contains exactly 1 donor per class in `stratify_key`.
    When sex_key is provided (disease task), folds are additionally balanced by sex:
    pairs (disease_male + control_female) and (disease_female + control_male)
    are alternated across folds, guaranteeing 1M+1F per val fold.

    Returns list of (train_cell_idx, val_cell_idx) as positional integer arrays.
    """
    rng = np.random.default_rng(seed)
    donor_first = obs.groupby(batch_key).first()

    if sex_key and sex_key in obs.columns:
        donor_first = donor_first.copy()
        donor_first["_strat"] = (
            donor_first[stratify_key].astype(str) + "__" +
            donor_first[sex_key].astype(str)
        )
        strat_col = "_strat"
    else:
        strat_col = stratify_key

    unique_labels = sorted(np.unique(donor_first[strat_col].astype(str).values))
    strata: dict[str, list] = {}
    for lbl in unique_labels:
        d = donor_first.index[donor_first[strat_col].astype(str) == lbl].tolist()
        strata[lbl] = rng.permutation(d).tolist()

    # 4-group case (disease × sex): pair complementary groups so each fold has 1M+1F
    if len(unique_labels) == 4:
        half = n_splits // 2
        if all(len(strata[l]) >= half for l in unique_labels):
            sl = sorted(unique_labels)  # alphabetical: [ctrl_F, ctrl_M, dis_F, dis_M]
            pair_a = (sl[3], sl[0])     # dis_M + ctrl_F
            pair_b = (sl[2], sl[1])     # dis_F + ctrl_M
            all_donors = set(donor_first.index)
            folds = []
            for i in range(half):
                for pair in (pair_a, pair_b):
                    val_donors = {strata[pair[0]][i], strata[pair[1]][i]}
                    train_donors = all_donors - val_donors
                    val_idx   = np.where(obs[batch_key].isin(val_donors).values)[0]
                    train_idx = np.where(obs[batch_key].isin(train_donors).values)[0]
                    folds.append((train_idx, val_idx))
            print(f"  [folds] sex-balanced: pairs {sl[3]}+{sl[0]} / {sl[2]}+{sl[1]}")
            return folds

    # Fallback: original single-key stratification
    for lbl in unique_labels:
        if len(strata[lbl]) < n_splits:
            raise ValueError(
                f"Class '{lbl}' has only {len(strata[lbl])} donors but n_splits={n_splits}. "
                "Cannot build balanced folds."
            )
    all_donors = set(donor_first.index)
    folds = []
    for i in range(n_splits):
        val_donors = {strata[lbl][i] for lbl in unique_labels}
        train_donors = all_donors - val_donors
        val_idx   = np.where(obs[batch_key].isin(val_donors).values)[0]
        train_idx = np.where(obs[batch_key].isin(train_donors).values)[0]
        folds.append((train_idx, val_idx))
    return folds


# ---------------------------------------------------------------------------
# Feature extraction (no data leakage for raw_pca)
# ---------------------------------------------------------------------------

def get_features(
    adata: ad.AnnData,
    method: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (X_train, X_test) for the given method.

    raw_pca: fits StandardScaler + PCA on train cells from normalized_log
             to avoid data leakage.
    Others:  slices the pre-computed obsm embedding by train/test index.
    """
    if method == "raw_pca":
        X_log_raw = adata.layers["normalized_log"]
        X_log = X_log_raw.toarray() if sp.issparse(X_log_raw) else np.asarray(X_log_raw)
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_log[train_idx])
        X_te_sc = scaler.transform(X_log[test_idx])
        n_comp = min(PREPROCESS["n_pcs"], X_tr_sc.shape[1], X_tr_sc.shape[0] - 1)
        pca = PCA(n_components=n_comp, random_state=ML["seed"])
        X_train = pca.fit_transform(X_tr_sc)
        X_test  = pca.transform(X_te_sc)
        print(f"    [raw_pca] PCA fit on {len(train_idx):,} train cells "
              f"→ {n_comp} components "
              f"(var explained: {pca.explained_variance_ratio_.sum():.1%})")
    else:
        key = EMBEDDING_KEYS[method]
        if key not in adata.obsm:
            raise KeyError(f"Embedding '{key}' not found in obsm.")
        X_all   = np.array(adata.obsm[key])
        X_train = X_all[train_idx]
        X_test  = X_all[test_idx]

    return X_train, X_test


# ---------------------------------------------------------------------------
# Inner CV hyperparameter search
# ---------------------------------------------------------------------------

def _best_hyperparams(
    X_train: np.ndarray,
    y_train: np.ndarray,
    donor_groups: np.ndarray,
    task: str,
    seed: int,
) -> tuple[float, float]:
    """
    5-fold inner CV; returns (best_C, best_l1_ratio).

    Classification (sex/disease): StratifiedGroupKFold, scored by roc_auc.
    Regression (age):             GroupKFold,            scored by neg_MAE.

    Uses Pipeline(StandardScaler, model) so scaler is fit only on inner train.
    """
    cfg = ML["elasticnet"]

    if task == "age":
        model = ElasticNet(max_iter=cfg["max_iter"], tol=cfg["tol"],
                           random_state=seed)
        param_grid = {
            "clf__alpha":    [1.0 / c for c in PARAM_GRID_C],
            "clf__l1_ratio": PARAM_GRID_L1RATIO,
        }
        inner_cv = GroupKFold(n_splits=N_INNER_FOLDS)
        scoring = "neg_mean_absolute_error"
    else:
        model = LogisticRegression(
            penalty="elasticnet", solver="saga",
            class_weight="balanced",
            max_iter=cfg["max_iter"], tol=cfg["tol"],
            random_state=seed,
        )
        param_grid = {
            "clf__C":        PARAM_GRID_C,
            "clf__l1_ratio": PARAM_GRID_L1RATIO,
        }
        inner_cv = StratifiedGroupKFold(n_splits=N_INNER_FOLDS)
        scoring = "roc_auc"

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
    gs = GridSearchCV(
        pipe, param_grid, cv=inner_cv, scoring=scoring,
        refit=True, n_jobs=1, error_score=np.nan,
    )
    gs.fit(X_train, y_train, groups=donor_groups)

    if task == "age":
        best_alpha = gs.best_params_["clf__alpha"]
        best_C     = 1.0 / best_alpha
    else:
        best_C = gs.best_params_["clf__C"]
    best_l1 = gs.best_params_["clf__l1_ratio"]
    return best_C, best_l1


# ---------------------------------------------------------------------------
# Metacell aggregation
# ---------------------------------------------------------------------------

def make_metacells(
    X: np.ndarray,
    donors: np.ndarray,
    y: np.ndarray,
    n_metacells: int = 100,
    metacell_size: int = 32,
    seed: int = 42,
    expected_reuse: float | None = 2.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each donor, draw bootstrap samples of metacell_size cells (with replacement)
    and average their features to form metacells.
    Cells must all be from the same cell type (guaranteed by per-CT mode).

    When expected_reuse is set (default 2.5), n_metacells is adaptive per donor:
        n_mc = max(1, round(expected_reuse * n_cells_donor / metacell_size))
    This prevents over-reuse in small cell-type pools. n_metacells is only used
    when expected_reuse is None.

    Returns (X_meta, y_meta, donors_meta).
    """
    rng = np.random.default_rng(seed)
    X_meta, y_meta, donors_meta = [], [], []
    for donor in np.unique(donors):
        mask = donors == donor
        X_d = X[mask]
        y_d = y[mask][0]
        n_cells = len(X_d)
        n_mc = (max(1, round(expected_reuse * n_cells / metacell_size))
                if expected_reuse is not None else n_metacells)
        for _ in range(n_mc):
            idx = rng.integers(0, n_cells, size=metacell_size)
            X_meta.append(X_d[idx].mean(axis=0))
            y_meta.append(y_d)
            donors_meta.append(donor)
    return np.array(X_meta), np.array(y_meta), np.array(donors_meta)


# ---------------------------------------------------------------------------
# Per-fold evaluation helpers
# ---------------------------------------------------------------------------

def _fit_classifier(X_train, y_train, C, l1_ratio, seed):
    cfg = ML["elasticnet"]
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    multiclass = len(np.unique(y_train)) > 2
    model = LogisticRegression(
        penalty="elasticnet", C=C, l1_ratio=l1_ratio,
        solver="saga", max_iter=cfg["max_iter"], tol=cfg["tol"],
        multi_class="multinomial" if multiclass else "auto",
        class_weight="balanced", random_state=seed,
    )
    model.fit(X_tr, y_train)
    return scaler, model


def _eval_classifier(scaler, model, X, y, groups=None):
    Xs = scaler.transform(X)
    y_pred = model.predict(Xs)
    bal_acc = balanced_accuracy_score(y, y_pred)
    multiclass = len(np.unique(y)) > 2
    donor_auroc = np.nan
    try:
        if multiclass:
            proba = model.predict_proba(Xs)
            auroc = roc_auc_score(y, proba, multi_class="ovr", average="macro")
        else:
            proba = model.predict_proba(Xs)[:, 1]
            auroc = roc_auc_score(y, proba)
            if groups is not None:
                donor_df = pd.DataFrame({"proba": proba, "y": y, "donor": groups})
                donor_agg = donor_df.groupby("donor").agg({"proba": "mean", "y": "first"})
                try:
                    donor_auroc = roc_auc_score(donor_agg["y"], donor_agg["proba"])
                except ValueError:
                    donor_auroc = np.nan
    except ValueError:
        auroc = np.nan
    return float(auroc), float(bal_acc), float(donor_auroc)


def _fit_regressor(X_train, y_train, C, l1_ratio, seed):
    cfg = ML["elasticnet"]
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    model = ElasticNet(
        alpha=1.0 / C, l1_ratio=l1_ratio,
        max_iter=cfg["max_iter"], tol=cfg["tol"], random_state=seed,
    )
    model.fit(X_tr, y_train)
    return scaler, model


def _eval_regressor_donor(scaler, model, X, y, donors):
    Xs = scaler.transform(X)
    y_pred_cell = model.predict(Xs)
    unique_donors = np.unique(donors)
    trues, preds = [], []
    for d in unique_donors:
        mask = donors == d
        trues.append(float(y[mask][0]))
        preds.append(float(y_pred_cell[mask].mean()))
    trues = np.array(trues)
    preds = np.array(preds)
    mae = mean_absolute_error(trues, preds)
    try:
        r, _ = pearsonr(trues, preds)
    except Exception:
        r = np.nan
    return float(mae), float(r)


# ---------------------------------------------------------------------------
# Per-task CV runner
# ---------------------------------------------------------------------------

def run_task(
    name: str,
    cfg: dict,
    task: str,
    adata: ad.AnnData,
    cell_type: str | None = None,
    use_metacells: bool = False,
    n_metacells: int = 100,
    metacell_size: int = 32,
) -> pd.DataFrame:
    """
    6-fold nested CV for one task (sex, age, disease) across all methods.
    Each outer fold: inner GridSearchCV tunes 30 hyperparameter combos.
    Reports mean ± std AUROC / BalAcc (classification) or MAE / R (regression).
    Saves best-fold coefficients for flashSCENIC.

    If cell_type is given, subsets adata to that cell type before running.
    If use_metacells, aggregates cells into metacells before fitting.
    """
    if cell_type is not None:
        ct_key = cfg.get("cell_type_key", "cell_type")
        adata = adata[adata.obs[ct_key] == cell_type].copy()

    if use_metacells:
        if cell_type is None:
            print("  [warn] --metacells is most meaningful in per-cell-type mode; "
                  "applying across all cell types.")
        print(f"  Mode: METACELL  (metacell_size={metacell_size}, n_metacells={n_metacells} per donor per fold)")
    else:
        print(f"  Mode: single-cell")

    batch_key = cfg["batch_key"]
    seed      = ML["seed"]
    le        = None

    # ---- Encode labels ----
    if task == "sex":
        key = cfg.get("sex_key")
        if not key or key not in adata.obs.columns:
            print(f"  [skip] sex: sex_key '{key}' not available")
            return pd.DataFrame()
        le = LabelEncoder()
        y = le.fit_transform(adata.obs[key].values)
        print(f"  Sex labels: {dict(zip(le.classes_, np.bincount(y)))}")
        stratify_key = key

    elif task == "age":
        key = cfg.get("age_key")
        if not key or key not in adata.obs.columns:
            print(f"  [skip] age: age_key '{key}' not available")
            return pd.DataFrame()
        y = parse_age(adata.obs[key])
        print(f"  Age range: {np.nanmin(y):.0f}–{np.nanmax(y):.0f} years "
              f"({np.isnan(y).sum()} NaN)")
        # Stratify age folds by sex (proxy) when available
        sex_key = cfg.get("sex_key")
        stratify_key = sex_key if sex_key and sex_key in adata.obs.columns else None

    elif task == "disease":
        key = cfg.get("disease_key")
        if not key or key not in adata.obs.columns:
            print(f"  [skip] disease: disease_key not available")
            return pd.DataFrame()
        le = LabelEncoder()
        y = le.fit_transform(adata.obs[key].values)
        print(f"  Disease labels: {dict(zip(le.classes_, np.bincount(y)))}")
        stratify_key = key

    else:
        print(f"  [skip] unknown task: {task}")
        return pd.DataFrame()

    # ---- Build outer CV folds ----
    if stratify_key is None:
        print(f"  [warn] no stratify key available; using GroupKFold by donor")
        from sklearn.model_selection import GroupKFold as _GKF
        donor_first = adata.obs.groupby(batch_key).first()
        all_donors = donor_first.index.tolist()
        rng = np.random.default_rng(seed)
        all_donors = rng.permutation(all_donors).tolist()
        donors_arr = adata.obs[batch_key].values
        cell_idx   = np.arange(len(donors_arr))
        gkf = _GKF(n_splits=N_OUTER_FOLDS)
        folds = list(gkf.split(cell_idx, groups=donors_arr))
    else:
        try:
            sex_key_for_folds = cfg.get("sex_key") if task == "disease" else None
            folds = make_cv_folds(adata.obs, batch_key, stratify_key,
                                  sex_key=sex_key_for_folds,
                                  n_splits=N_OUTER_FOLDS, seed=seed)
        except ValueError as e:
            print(f"  [warn] {e}; falling back to GroupKFold")
            from sklearn.model_selection import GroupKFold as _GKF
            donors_arr = adata.obs[batch_key].values
            cell_idx   = np.arange(len(donors_arr))
            gkf = _GKF(n_splits=N_OUTER_FOLDS)
            folds = list(gkf.split(cell_idx, groups=donors_arr))

    print(f"  Outer folds: {N_OUTER_FOLDS} (inner grid: "
          f"{len(PARAM_GRID_C)}×{len(PARAM_GRID_L1RATIO)}={len(PARAM_GRID_C)*len(PARAM_GRID_L1RATIO)} combos, "
          f"{N_INNER_FOLDS}-fold inner CV)")

    rows = []
    for method in METHODS:
        print(f"\n  Method: {method}")

        fold_metrics: dict[str, list] = {}
        fold_best_C:    list[float] = []
        fold_best_l1:   list[float] = []
        fold_coefs:     list[np.ndarray] = []
        fold_val_score: list[float] = []

        for fold_i, (train_idx, val_idx) in enumerate(folds):
            is_meta_fold = use_metacells and method == "flashscenic"
            print(f"    Fold {fold_i + 1}/{N_OUTER_FOLDS} "
                  f"({len(train_idx):,} train / {len(val_idx):,} val cells"
                  f"{' → metacells' if is_meta_fold else ''})")

            # Features
            try:
                X_train, X_val = get_features(adata, method, train_idx, val_idx)
            except KeyError as e:
                print(f"      [skip] {e}")
                break

            y_train = y[train_idx]
            y_val   = y[val_idx]
            donor_groups_train = adata.obs[batch_key].values[train_idx]

            if task == "age":
                valid_tr = ~np.isnan(y_train)
                valid_va = ~np.isnan(y_val)
                if valid_tr.sum() < 2 or valid_va.sum() < 2:
                    print(f"      [skip fold] insufficient non-NaN ages")
                    continue
                X_train_use = X_train[valid_tr]
                y_train_use = y_train[valid_tr]
                X_val_use   = X_val[valid_va]
                y_val_use   = y_val[valid_va]
                donor_groups_use = donor_groups_train[valid_tr]

                # Inner CV for hyperparam tuning
                best_C, best_l1 = _best_hyperparams(
                    X_train_use, y_train_use, donor_groups_use, task, seed
                )
                print(f"      best C={best_C}, l1={best_l1}")
                fold_best_C.append(best_C)
                fold_best_l1.append(best_l1)

                scaler, model = _fit_regressor(X_train_use, y_train_use, best_C, best_l1, seed)
                donors_train_use = donor_groups_use
                donors_val_use   = adata.obs[batch_key].values[val_idx][valid_va]

                mae_tr, r_tr = _eval_regressor_donor(
                    scaler, model, X_train_use, y_train_use, donors_train_use
                )
                mae_va, r_va = _eval_regressor_donor(
                    scaler, model, X_val_use, y_val_use, donors_val_use
                )
                print(f"      val  MAE={mae_va:.1f} yrs, R={r_va:.3f}")

                fold_metrics.setdefault("val_MAE", []).append(mae_va)
                fold_metrics.setdefault("val_R",   []).append(r_va)
                fold_coefs.append(model.coef_)
                fold_val_score.append(-mae_va)  # higher = better for selection

            else:
                if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
                    print(f"      [skip fold] only one class in train or val")
                    continue

                # Metacell aggregation (flashscenic only, adaptive n_metacells per donor)
                expected_reuse = ML.get("expected_reuse", 2.5)
                if use_metacells and method == "flashscenic":
                    donor_groups_val_orig = adata.obs[batch_key].values[val_idx]
                    X_train, y_train, donor_groups_train = make_metacells(
                        X_train, donor_groups_train, y_train,
                        metacell_size=metacell_size,
                        expected_reuse=expected_reuse,
                        seed=seed + fold_i,
                    )
                    X_val, y_val, donor_groups_val_meta = make_metacells(
                        X_val, donor_groups_val_orig, y_val,
                        metacell_size=metacell_size,
                        expected_reuse=expected_reuse,
                        seed=seed + fold_i + 1000,
                    )
                    print(f"      Metacells: {len(X_train):,} train / {len(X_val):,} val")
                else:
                    donor_groups_val_meta = adata.obs[batch_key].values[val_idx]

                # Inner CV for hyperparam tuning
                best_C, best_l1 = _best_hyperparams(
                    X_train, y_train, donor_groups_train, task, seed
                )
                print(f"      best C={best_C}, l1={best_l1}")
                fold_best_C.append(best_C)
                fold_best_l1.append(best_l1)

                scaler, model = _fit_classifier(X_train, y_train, best_C, best_l1, seed)

                auroc_va, bacc_va, donor_auroc_va = _eval_classifier(
                    scaler, model, X_val, y_val,
                    groups=donor_groups_val_meta)
                print(f"      val  AUROC={auroc_va:.4f}, BalAcc={bacc_va:.4f}, "
                      f"DonorAUROC={donor_auroc_va:.4f}")

                fold_metrics.setdefault("val_AUROC",       []).append(auroc_va)
                fold_metrics.setdefault("val_BalAcc",      []).append(bacc_va)
                fold_metrics.setdefault("val_donor_AUROC", []).append(donor_auroc_va)

                if model.coef_.ndim == 2:
                    coef = model.coef_[0]
                else:
                    coef = model.coef_
                fold_coefs.append(coef)
                fold_val_score.append(auroc_va if not np.isnan(auroc_va) else 0.0)

        if not fold_metrics:
            print(f"    [skip] no valid folds for {method}")
            continue

        # ---- Aggregate across folds ----
        result: dict = {
            "dataset": name, "method": method, "task": task,
            "cell_type": cell_type if cell_type is not None else "all",
        }

        if task == "age":
            maes = fold_metrics["val_MAE"]
            rs   = fold_metrics["val_R"]
            result["test_MAE"]     = round(float(np.mean(maes)), 2)
            result["test_MAE_std"] = round(float(np.std(maes, ddof=1) if len(maes) > 1 else 0.0), 2)
            result["test_R"]       = round(float(np.nanmean(rs)), 4)
            result["test_R_std"]   = round(float(np.nanstd(rs, ddof=1) if len(rs) > 1 else 0.0), 4)
            print(f"    CV mean  MAE={result['test_MAE']:.1f}±{result['test_MAE_std']:.1f}, "
                  f"R={result['test_R']:.3f}±{result['test_R_std']:.3f}")
        else:
            aurocs       = fold_metrics["val_AUROC"]
            baccs        = fold_metrics["val_BalAcc"]
            donor_aurocs = fold_metrics.get("val_donor_AUROC", [])
            result["test_AUROC"]            = round(float(np.nanmean(aurocs)), 4)
            result["test_AUROC_std"]        = round(float(np.nanstd(aurocs, ddof=1) if len(aurocs) > 1 else 0.0), 4)
            result["test_BalAcc"]           = round(float(np.mean(baccs)), 4)
            result["test_BalAcc_std"]       = round(float(np.std(baccs, ddof=1) if len(baccs) > 1 else 0.0), 4)
            result["test_donor_AUROC"]      = round(float(np.nanmean(donor_aurocs)) if donor_aurocs else float("nan"), 4)
            result["test_donor_AUROC_std"]  = round(float(np.nanstd(donor_aurocs, ddof=1)) if len(donor_aurocs) > 1 else 0.0, 4)
            print(f"    CV mean  AUROC={result['test_AUROC']:.4f}±{result['test_AUROC_std']:.4f}, "
                  f"BalAcc={result['test_BalAcc']:.4f}±{result['test_BalAcc_std']:.4f}, "
                  f"DonorAUROC={result['test_donor_AUROC']:.4f}±{result['test_donor_AUROC_std']:.4f}")

        # ---- Save coefficients: refit best-fold params on ALL donors ----
        if method == "flashscenic" and fold_coefs and fold_best_C:
            best_fold_i = int(np.argmax(fold_val_score))
            best_C_all  = fold_best_C[best_fold_i]
            best_l1_all = fold_best_l1[best_fold_i]

            all_idx = np.arange(len(adata.obs))
            try:
                X_all, _ = get_features(adata, method, all_idx, all_idx[:1])
                # For flashscenic, X_all is just the full obsm slice
                X_all_emb = np.array(adata.obsm[EMBEDDING_KEYS[method]])
            except Exception:
                X_all_emb = None

            if X_all_emb is not None:
                y_all = y
                donors_all = adata.obs[batch_key].values
                if use_metacells and method == "flashscenic":
                    X_all_emb, y_all, _ = make_metacells(
                        X_all_emb, donors_all, y_all,
                        metacell_size=metacell_size,
                        expected_reuse=ML.get("expected_reuse", 2.5),
                        seed=seed,
                    )
                if task == "age":
                    valid = ~np.isnan(y_all)
                    scaler_all, model_all = _fit_regressor(
                        X_all_emb[valid], y_all[valid], best_C_all, best_l1_all, seed
                    )
                    coef_all = model_all.coef_
                else:
                    scaler_all, model_all = _fit_classifier(
                        X_all_emb, y_all, best_C_all, best_l1_all, seed
                    )
                    coef_all = model_all.coef_[0] if model_all.coef_.ndim == 2 else model_all.coef_

                feat_names = get_feature_names(name, method)
                if feat_names and len(feat_names) == len(coef_all):
                    meta_suffix = f"_metacells_{cell_type}" if use_metacells and cell_type else (
                        "_metacells" if use_metacells else ""
                    )
                    out = METRICS_DIR / f"ml_coef_{name}_flashscenic_{task}{meta_suffix}.csv"
                    coef_df = pd.DataFrame({"feature": feat_names, "coefficient": coef_all})
                    if le is not None and len(le.classes_) >= 2:
                        coef_df["class_0"] = le.classes_[0]
                        coef_df["class_1"] = le.classes_[1]
                    coef_df.to_csv(out, index=False)
                    print(f"    [coef] Saved {out.name} "
                          f"(best fold {best_fold_i+1}, C={best_C_all}, l1={best_l1_all})")

        rows.append(result)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-cell-type disease prediction
# ---------------------------------------------------------------------------

def run_per_ct_disease(
    name: str,
    cfg: dict,
    adata: ad.AnnData,
    use_metacells: bool = False,
    n_metacells: int = 100,
    metacell_size: int = 32,
) -> None:
    """
    Run disease prediction separately for each cell type.
    Skips cell types with < 2 donors per disease class.
    Saves results to ml_disease_per_ct_{name}[_metacells].csv.
    """
    ct_key      = cfg.get("cell_type_key", "cell_type")
    disease_key = cfg.get("disease_key")
    batch_key   = cfg["batch_key"]

    if not disease_key or disease_key not in adata.obs.columns:
        print("  [skip per-CT] disease_key not available")
        return
    if ct_key not in adata.obs.columns:
        print(f"  [skip per-CT] cell_type_key '{ct_key}' not in obs")
        return

    cell_types = sorted(adata.obs[ct_key].unique())
    suffix = "_metacells" if use_metacells else ""
    print(f"\n--- Per-cell-type disease prediction ({len(cell_types)} cell types)"
          f"{' [metacells]' if use_metacells else ''} ---")

    all_results = []
    for ct in cell_types:
        ct_mask = adata.obs[ct_key] == ct
        ct_adata = adata[ct_mask]
        donors_per_class = (
            ct_adata.obs.groupby(disease_key)[batch_key]
            .nunique()
        )
        if ct_mask.sum() < MIN_CELLS_PER_CT:
            print(f"  [skip] {ct}: {ct_mask.sum():,} cells < {MIN_CELLS_PER_CT} minimum")
            continue
        if donors_per_class.min() < 2:
            print(f"  [skip] {ct}: too few donors per class {donors_per_class.to_dict()}")
            continue
        print(f"\n  Cell type: {ct} ({ct_mask.sum():,} cells)")
        try:
            df_ct = run_task(name, cfg, "disease", adata, cell_type=ct,
                             use_metacells=use_metacells,
                             n_metacells=n_metacells, metacell_size=metacell_size)
            if not df_ct.empty:
                all_results.append(df_ct)
        except Exception as e:
            print(f"  [error] {ct}: {e}")

    if all_results:
        out_df = pd.concat(all_results, ignore_index=True)
        out = METRICS_DIR / f"ml_disease_per_ct_{name}{suffix}.csv"
        out_df.to_csv(out, index=False)
        print(f"\n  Per-CT disease results saved → {out.name}  "
              f"({out_df.cell_type.nunique()} cell types)")
    else:
        print("  No per-CT results produced.")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize(dfs: dict[str, pd.DataFrame], name: str) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        row = {"dataset": name, "method": method}
        for task, df in dfs.items():
            if df.empty:
                continue
            subset = df[df.method == method]
            if subset.empty:
                continue
            if task in ("sex", "disease"):
                for col in ["test_AUROC", "test_AUROC_std", "test_BalAcc", "test_BalAcc_std",
                            "test_donor_AUROC", "test_donor_AUROC_std"]:
                    if col in subset.columns:
                        row[f"{task}_{col}"] = subset[col].iloc[0]
            elif task == "age":
                for col in ["test_MAE", "test_MAE_std", "test_R", "test_R_std"]:
                    if col in subset.columns:
                        row[f"age_{col}"] = subset[col].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all",
                        choices=list(DATASETS.keys()) + ["all"])
    parser.add_argument("--tasks", nargs="+", default=["sex", "disease"],
                        choices=["sex", "age", "disease"])
    parser.add_argument("--metacell-size", type=int, default=ML["metacell_size"],
                        help="Cells per metacell bootstrap sample")
    parser.add_argument("--n-metacells", type=int, default=ML["n_metacells"],
                        help="Max metacells per donor per fold (overridden by expected_reuse)")
    args = parser.parse_args()

    to_process = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    for name in to_process:
        print(f"\n{'='*60}")
        print(f"ML predictor (nested 6-fold CV): {name}")
        print(f"{'='*60}")

        preprocessed_path = DATA_DIR / f"preprocessed_{name}.h5ad"
        if not preprocessed_path.exists():
            print(f"  [skip] {preprocessed_path.name} not found.")
            continue

        adata = ad.read_h5ad(preprocessed_path)

        cfg = DATASETS[name]
        ct_key = cfg.get("cell_type_key", "cell_type")
        if ct_key in adata.obs.columns:
            ct_counts = adata.obs[ct_key].value_counts()
            small_cts = ct_counts[ct_counts < MIN_CELLS_PER_CT].index.tolist()
            if small_cts:
                print(f"  Filtering {len(small_cts)} cell type(s) with < {MIN_CELLS_PER_CT} cells: "
                      f"{small_cts}")
                adata = adata[~adata.obs[ct_key].isin(small_cts)].copy()
                print(f"  Remaining: {adata.n_obs:,} cells across "
                      f"{adata.obs[ct_key].nunique()} cell types")

        dfs = {}
        for task in args.tasks:
            print(f"\n--- Task: {task} (cell-level) ---")
            df = run_task(name, cfg, task, adata)
            dfs[task] = df
            if not df.empty:
                out = METRICS_DIR / f"ml_{task}_{name}.csv"
                df.to_csv(out, index=False)
                print(f"\n  {task} results saved to {out.name}")

        for task in [t for t in ("sex", "disease") if t in args.tasks]:
            print(f"\n--- Task: {task} (metacell, flashscenic only) ---")
            df_meta = run_task(name, cfg, task, adata,
                               use_metacells=True,
                               n_metacells=args.n_metacells,
                               metacell_size=args.metacell_size)
            if not df_meta.empty:
                out = METRICS_DIR / f"ml_{task}_{name}_metacells.csv"
                df_meta.to_csv(out, index=False)
                print(f"\n  {task} metacell results saved to {out.name}")

        if "disease" in args.tasks:
            # Cell-level: raw_pca + flashscenic
            run_per_ct_disease(name, cfg, adata, use_metacells=False,
                               n_metacells=args.n_metacells,
                               metacell_size=args.metacell_size)
            # Metacell: flashscenic only (raw_pca is skipped by the flashscenic gate)
            run_per_ct_disease(name, cfg, adata, use_metacells=True,
                               n_metacells=args.n_metacells,
                               metacell_size=args.metacell_size)

        df_summary = summarize(dfs, name)
        out = METRICS_DIR / f"ml_summary_{name}.csv"
        if out.exists():
            existing = pd.read_csv(out)
            # Merge: update only the columns produced by this run; keep the rest from disk
            new_cols = [c for c in df_summary.columns if c not in ("dataset", "method")]
            merged = existing.copy()
            for col in new_cols:
                merged[col] = merged["method"].map(
                    df_summary.set_index("method")[col]
                )
            df_summary = merged
        df_summary.to_csv(out, index=False)

        print(f"\n  Summary ({name}):")
        pd.set_option("display.float_format", "{:.3f}".format)
        display_cols = [c for c in df_summary.columns if c not in ("dataset", "method")]
        if display_cols:
            print(df_summary.set_index("method")[display_cols].to_string())

    print("\nML predictor done. Run 05_visualize.py next.")


if __name__ == "__main__":
    main()
