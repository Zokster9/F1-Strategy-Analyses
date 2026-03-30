"""
Optimizacija hiperparametara za klasifikacione i klasterske modele.

Koristi Optuna (Bayesian TPE) za klasifikaciju i sistematsko pretrazivanje
za klasterovanje (silhouette, BIC/AIC).

Pokretanje:
    python -m src.optimization [--mode classification|clustering|all]
                               [--n-trials 100]
                               [--data-path data/processed/f1_laps_processed.csv]
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    adjusted_rand_score,
    f1_score,
    silhouette_score,
)
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

import optuna
from imblearn.over_sampling import SMOTE

from src.modeling import (
    FEATURE_COLUMNS,
    _GroupedIQRFilter,
    _ClusteringFeaturePreprocessor,
    _cluster_purity,
    _split_classification_dataset,
    _split_clustering_dataset,
)

XGBOOST_AVAILABLE = False
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBClassifier = None

LOGGER = logging.getLogger(__name__)

# Prosireni skup obelezja (ukljucuje izvedena obelezja iz preprocessing.py).
EXTENDED_FEATURE_COLUMNS = FEATURE_COLUMNS + [
    "SectorRatio_S1",
    "SectorRatio_S2",
    "SectorRatio_S3",
    "SpeedDelta",
    "TyreLifeSquared",
    "LapFraction",
]


def _available_features(data: pd.DataFrame, extended: bool = True) -> list[str]:
    """Vraca listu obelezja koja postoje u podacima."""
    candidates = EXTENDED_FEATURE_COLUMNS if extended else FEATURE_COLUMNS
    return [col for col in candidates if col in data.columns]


def _compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """Racuna balansirane tezine uzoraka: total / (n_classes * class_count)."""
    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    total = len(y)
    weight_map = {cls: total / (n_classes * cnt) for cls, cnt in zip(classes, counts)}
    return np.array([weight_map[label] for label in y], dtype=float)


# ---------------------------------------------------------------------------
# Klasifikacija — Optuna optimizacija
# ---------------------------------------------------------------------------

def _rf_objective(trial: optuna.Trial, x: np.ndarray, y: np.ndarray,
                  cv: StratifiedKFold, use_smote: bool) -> float:
    params = {
        "n_estimators": trial.suggest_categorical("rf_n_estimators", [200, 400, 600, 800]),
        "max_depth": trial.suggest_categorical("rf_max_depth", [15, 20, 30, 50, 0]),
        "min_samples_split": trial.suggest_categorical("rf_min_samples_split", [2, 4, 8, 16, 32]),
        "min_samples_leaf": trial.suggest_categorical("rf_min_samples_leaf", [1, 2, 4, 8]),
        "max_features": trial.suggest_categorical("rf_max_features", ["sqrt", "log2", 0.5, 0.75]),
        "class_weight": trial.suggest_categorical("rf_class_weight", ["balanced", "balanced_subsample"]),
        "criterion": trial.suggest_categorical("rf_criterion", ["gini", "entropy"]),
        "random_state": 42,
        "n_jobs": -1,
    }
    if params["max_depth"] == 0:
        params["max_depth"] = None

    scores = []
    for train_idx, val_idx in cv.split(x, y):
        x_train_fold, x_val_fold = x[train_idx], x[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        if use_smote:
            smote = SMOTE(random_state=42)
            x_train_fold, y_train_fold = smote.fit_resample(x_train_fold, y_train_fold)

        model = RandomForestClassifier(**params)
        model.fit(x_train_fold, y_train_fold)
        y_pred = model.predict(x_val_fold)
        scores.append(f1_score(y_val_fold, y_pred, average="macro"))

    return float(np.mean(scores))


def _xgb_objective(trial: optuna.Trial, x: np.ndarray, y: np.ndarray,
                   cv: StratifiedKFold, use_smote: bool) -> float:
    if not XGBOOST_AVAILABLE:
        raise RuntimeError("XGBoost nije dostupan.")

    params = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "n_estimators": trial.suggest_categorical("xgb_n_estimators", [200, 400, 600, 800, 1000]),
        "max_depth": trial.suggest_categorical("xgb_max_depth", [4, 6, 8, 10, 12]),
        "learning_rate": trial.suggest_categorical("xgb_learning_rate", [0.01, 0.03, 0.05, 0.1, 0.2]),
        "subsample": trial.suggest_categorical("xgb_subsample", [0.6, 0.7, 0.8, 0.9, 1.0]),
        "colsample_bytree": trial.suggest_categorical("xgb_colsample_bytree", [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        "min_child_weight": trial.suggest_categorical("xgb_min_child_weight", [1, 3, 5, 7, 10]),
        "gamma": trial.suggest_categorical("xgb_gamma", [0, 0.1, 0.3, 0.5, 1.0]),
        "reg_alpha": trial.suggest_categorical("xgb_reg_alpha", [0, 0.01, 0.1, 1.0]),
        "reg_lambda": trial.suggest_categorical("xgb_reg_lambda", [0.5, 1.0, 3.0, 5.0]),
        "random_state": 42,
        "n_jobs": -1,
    }

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    scores = []
    for train_idx, val_idx in cv.split(x, y_encoded):
        x_train_fold, x_val_fold = x[train_idx], x[val_idx]
        y_train_fold, y_val_fold = y_encoded[train_idx], y_encoded[val_idx]

        if use_smote:
            smote = SMOTE(random_state=42)
            x_train_fold, y_train_fold = smote.fit_resample(x_train_fold, y_train_fold)

        sample_weights = _compute_sample_weights(y_train_fold)

        model = XGBClassifier(**params)
        model.fit(
            x_train_fold, y_train_fold,
            sample_weight=sample_weights,
            eval_set=[(x_val_fold, y_val_fold)],
            verbose=False,
        )
        y_pred = model.predict(x_val_fold).astype(int)
        y_pred_labels = encoder.inverse_transform(y_pred)
        y_val_labels = encoder.inverse_transform(y_val_fold)
        scores.append(f1_score(y_val_labels, y_pred_labels, average="macro"))

    return float(np.mean(scores))


def optimize_classification(
    data: pd.DataFrame,
    output_dir: Path = Path("results/optimization"),
    n_trials: int = 100,
    use_extended_features: bool = True,
    use_smote: bool = True,
    random_state: int = 42,
) -> dict[str, dict]:
    """
    Pokrece Optuna optimizaciju za RandomForest i XGBoost klasifikatore.
    Vraca dict sa najboljim parametrima za svaki model.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = _available_features(data, extended=use_extended_features)
    LOGGER.info("Optimizacija klasifikacije sa obelezjima: %s", feature_cols)

    subset = data.dropna(subset=feature_cols + ["Compound"]).copy()
    split_data = _split_classification_dataset(subset, random_state=random_state)

    # Koristimo train+validation za CV, test je strogo odvojen.
    train_val = pd.concat([split_data["train"], split_data["validation"]], ignore_index=True)
    test_df = split_data["test"]

    outlier_group_columns = [col for col in ["Year", "EventName", "Driver"] if col in subset.columns]
    iqr_filter = _GroupedIQRFilter(
        value_column="LapTime",
        group_columns=outlier_group_columns,
        iqr_multiplier=1.5,
    ).fit(train_val)
    train_val = iqr_filter.transform(train_val)
    test_df = iqr_filter.transform(test_df)

    x_cv = train_val[feature_cols].to_numpy()
    y_cv = train_val["Compound"].astype(str).to_numpy()
    x_test = test_df[feature_cols].to_numpy()
    y_test = test_df["Compound"].astype(str).to_numpy()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    results: dict[str, dict] = {}

    # --- RandomForest ---
    LOGGER.info("Pokretanje Optuna optimizacije za RandomForest (%d trial-ova)...", n_trials)
    rf_study = optuna.create_study(direction="maximize", study_name="rf_optimization")
    rf_study.optimize(
        lambda trial: _rf_objective(trial, x_cv, y_cv, cv, use_smote),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    rf_best = rf_study.best_params.copy()
    # Konvertuj max_depth=0 nazad u None.
    if rf_best.get("rf_max_depth") == 0:
        rf_best["rf_max_depth"] = None
    rf_best_clean = {k.replace("rf_", ""): v for k, v in rf_best.items()}

    # Evaluacija na test setu sa najboljim parametrima.
    rf_model = RandomForestClassifier(**rf_best_clean, random_state=42, n_jobs=-1)
    x_train_final, y_train_final = x_cv, y_cv
    if use_smote:
        smote = SMOTE(random_state=42)
        x_train_final, y_train_final = smote.fit_resample(x_cv, y_cv)
    rf_model.fit(x_train_final, y_train_final)
    rf_test_pred = rf_model.predict(x_test)
    rf_test_macro_f1 = f1_score(y_test, rf_test_pred, average="macro")
    rf_test_weighted_f1 = f1_score(y_test, rf_test_pred, average="weighted")

    results["RandomForest"] = {
        "best_params": rf_best_clean,
        "best_cv_macro_f1": rf_study.best_value,
        "test_macro_f1": rf_test_macro_f1,
        "test_weighted_f1": rf_test_weighted_f1,
        "n_trials": n_trials,
        "use_smote": use_smote,
        "feature_columns": feature_cols,
    }
    LOGGER.info(
        "RandomForest — najbolji CV macro F1: %.4f, test macro F1: %.4f",
        rf_study.best_value, rf_test_macro_f1,
    )

    # --- XGBoost ---
    if XGBOOST_AVAILABLE:
        LOGGER.info("Pokretanje Optuna optimizacije za XGBoost (%d trial-ova)...", n_trials)
        xgb_study = optuna.create_study(direction="maximize", study_name="xgb_optimization")
        xgb_study.optimize(
            lambda trial: _xgb_objective(trial, x_cv, y_cv, cv, use_smote),
            n_trials=n_trials,
            show_progress_bar=True,
        )

        xgb_best = xgb_study.best_params.copy()
        xgb_best_clean = {k.replace("xgb_", ""): v for k, v in xgb_best.items()}

        encoder = LabelEncoder()
        y_cv_enc = encoder.fit_transform(y_cv)
        y_test_enc = encoder.transform(y_test)

        x_train_final, y_train_final = x_cv, y_cv_enc
        if use_smote:
            smote = SMOTE(random_state=42)
            x_train_final, y_train_final = smote.fit_resample(x_cv, y_cv_enc)

        sample_weights = _compute_sample_weights(y_train_final)
        xgb_model = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            **xgb_best_clean,
            random_state=42,
            n_jobs=-1,
        )
        xgb_model.fit(x_train_final, y_train_final, sample_weight=sample_weights, verbose=False)
        xgb_test_pred = encoder.inverse_transform(xgb_model.predict(x_test).astype(int))
        xgb_test_macro_f1 = f1_score(y_test, xgb_test_pred, average="macro")
        xgb_test_weighted_f1 = f1_score(y_test, xgb_test_pred, average="weighted")

        results["XGBoost"] = {
            "best_params": xgb_best_clean,
            "best_cv_macro_f1": xgb_study.best_value,
            "test_macro_f1": xgb_test_macro_f1,
            "test_weighted_f1": xgb_test_weighted_f1,
            "n_trials": n_trials,
            "use_smote": use_smote,
            "use_sample_weights": True,
            "feature_columns": feature_cols,
        }
        LOGGER.info(
            "XGBoost — najbolji CV macro F1: %.4f, test macro F1: %.4f",
            xgb_study.best_value, xgb_test_macro_f1,
        )
    else:
        LOGGER.warning("XGBoost nije dostupan, preskacemo optimizaciju.")

    # Sacuvaj rezultate.
    with (output_dir / "classification_optimization.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    LOGGER.info("Rezultati klasifikacione optimizacije sacuvani u %s", output_dir)
    return results


# ---------------------------------------------------------------------------
# Klasterovanje — Silhouette, BIC/AIC i parametarsko pretrazivanje
# ---------------------------------------------------------------------------

def optimize_clustering(
    data: pd.DataFrame,
    output_dir: Path = Path("results/optimization"),
    random_state: int = 42,
) -> dict[str, object]:
    """
    Pokrece optimizaciju klasterovanja:
    1. Silhouette analiza za odredjivanje optimalnog broja klastera.
    2. BIC/AIC za GMM selekciju broja komponenti.
    3. Parametarsko pretrazivanje KMeans i GMM.
    4. Test sa i bez PCA redukcije dimenzionalnosti.
    5. Test sa i bez Driver one-hot obelezja.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    team_label_col = "CanonicalTeam" if "CanonicalTeam" in data.columns else "Team"
    feature_cols = _available_features(data, extended=True)
    required_cols = feature_cols + [team_label_col]
    subset = data.dropna(subset=required_cols).copy()

    split_data = _split_clustering_dataset(subset, team_label_col, random_state=random_state)
    raw_train_df = split_data["train"]
    raw_val_df = split_data["validation"]
    raw_test_df = split_data["test"]

    outlier_group_columns = [col for col in ["Year", "EventName", "Driver"] if col in subset.columns]
    iqr_filter = _GroupedIQRFilter(
        value_column="LapTime",
        group_columns=outlier_group_columns,
        iqr_multiplier=1.5,
    ).fit(raw_train_df)
    train_df = iqr_filter.transform(raw_train_df)
    val_df = iqr_filter.transform(raw_val_df)
    test_df = iqr_filter.transform(raw_test_df)

    y_val = val_df[team_label_col].astype(str)
    y_test = test_df[team_label_col].astype(str)

    results: dict[str, object] = {}

    # --- Priprema razlicitih prostora obelezja ---
    feature_spaces: dict[str, dict] = {}

    # Prostor 1: Sa Driver one-hot (originalni pristup).
    preprocessor_with_driver = _ClusteringFeaturePreprocessor(
        [c for c in FEATURE_COLUMNS if c in data.columns]
    ).fit(train_df)
    feature_spaces["with_driver"] = {
        "x_train": preprocessor_with_driver.transform(train_df),
        "x_val": preprocessor_with_driver.transform(val_df),
        "x_test": preprocessor_with_driver.transform(test_df),
        "description": "event_compound_zscore + StandardScaler + Driver one-hot",
    }

    # Prostor 2: Bez Driver one-hot.
    preprocessor_no_driver = _ClusteringFeaturePreprocessor(
        [c for c in FEATURE_COLUMNS if c in data.columns]
    )
    preprocessor_no_driver.include_driver = False
    preprocessor_no_driver.zscore.fit(train_df)
    preprocessor_no_driver.scaler.fit(
        preprocessor_no_driver.zscore.transform(train_df)[
            [c for c in FEATURE_COLUMNS if c in data.columns]
        ]
    )
    feature_spaces["no_driver"] = {
        "x_train": preprocessor_no_driver.transform(train_df),
        "x_val": preprocessor_no_driver.transform(val_df),
        "x_test": preprocessor_no_driver.transform(test_df),
        "description": "event_compound_zscore + StandardScaler, bez Driver one-hot",
    }

    # Prostor 3: Sa Driver one-hot + PCA (95% varijanse).
    x_train_full = feature_spaces["with_driver"]["x_train"]
    pca = PCA(n_components=0.95, random_state=random_state)
    x_train_pca = pca.fit_transform(x_train_full)
    feature_spaces["with_driver_pca"] = {
        "x_train": x_train_pca,
        "x_val": pca.transform(feature_spaces["with_driver"]["x_val"]),
        "x_test": pca.transform(feature_spaces["with_driver"]["x_test"]),
        "description": f"with_driver + PCA ({pca.n_components_} komponenti, 95% varijanse)",
        "pca_n_components": int(pca.n_components_),
    }
    LOGGER.info("PCA redukcija: %d -> %d komponenti (95%% varijanse)",
                x_train_full.shape[1], pca.n_components_)

    # --- 1. Silhouette analiza ---
    LOGGER.info("Silhouette analiza...")
    k_range = list(range(3, 26))
    silhouette_results = []

    # Koristimo no_driver prostor za silhouette (niža dimenzionalnost).
    x_sil = feature_spaces["no_driver"]["x_train"]
    # Poduzorkovanje za efikasnost silhouette izracunavanja.
    if len(x_sil) > 20000:
        rng = np.random.RandomState(random_state)
        sil_idx = rng.choice(len(x_sil), 20000, replace=False)
        x_sil_sample = x_sil[sil_idx]
    else:
        x_sil_sample = x_sil

    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(x_sil_sample)
        sil = silhouette_score(x_sil_sample, labels)
        silhouette_results.append({"k": k, "silhouette_score": sil})
        LOGGER.info("  k=%d: silhouette=%.4f", k, sil)

    sil_df = pd.DataFrame(silhouette_results)
    sil_df.to_csv(output_dir / "silhouette_analysis.csv", index=False)
    best_k_silhouette = int(sil_df.loc[sil_df["silhouette_score"].idxmax(), "k"])
    results["silhouette"] = {
        "best_k": best_k_silhouette,
        "best_score": float(sil_df["silhouette_score"].max()),
        "all_scores": silhouette_results,
    }
    LOGGER.info("Optimalni k po silhouette: %d (score=%.4f)",
                best_k_silhouette, sil_df["silhouette_score"].max())

    # --- 2. BIC/AIC analiza za GMM ---
    LOGGER.info("BIC/AIC analiza za GMM...")
    bic_aic_results = []
    x_bic = feature_spaces["no_driver"]["x_train"]

    for n_comp in k_range:
        for cov_type in ["full", "tied", "diag", "spherical"]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gmm = GaussianMixture(
                    n_components=n_comp,
                    covariance_type=cov_type,
                    reg_covar=1e-4,
                    n_init=3,
                    max_iter=300,
                    random_state=random_state,
                )
                gmm.fit(x_bic)
                bic_aic_results.append({
                    "n_components": n_comp,
                    "covariance_type": cov_type,
                    "bic": gmm.bic(x_bic),
                    "aic": gmm.aic(x_bic),
                })

    bic_df = pd.DataFrame(bic_aic_results)
    bic_df.to_csv(output_dir / "gmm_bic_aic_analysis.csv", index=False)
    best_bic_row = bic_df.loc[bic_df["bic"].idxmin()]
    results["bic_aic"] = {
        "best_n_components_bic": int(best_bic_row["n_components"]),
        "best_covariance_type_bic": str(best_bic_row["covariance_type"]),
        "best_bic": float(best_bic_row["bic"]),
    }
    LOGGER.info("Optimalni GMM po BIC: n_components=%d, cov_type=%s",
                int(best_bic_row["n_components"]), best_bic_row["covariance_type"])

    # --- 3. Parametarsko pretrazivanje ---
    LOGGER.info("Parametarsko pretrazivanje klasterovanja...")
    clustering_results = []

    k_values = [3, 5, 7, best_k_silhouette, 10, 12, 15, 20]
    k_values = sorted(set(k_values))

    for space_name, space_data in feature_spaces.items():
        x_train_s = space_data["x_train"]
        x_val_s = space_data["x_val"]
        x_test_s = space_data["x_test"]

        for k in k_values:
            # KMeans
            for n_init in [10, 20, 40]:
                km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
                km.fit(x_train_s)
                val_pred = km.predict(x_val_s)
                test_pred = km.predict(x_test_s)

                val_ari = adjusted_rand_score(y_val, val_pred)
                val_purity = _cluster_purity(y_val, val_pred)
                test_ari = adjusted_rand_score(y_test, test_pred)
                test_purity = _cluster_purity(y_test, test_pred)

                clustering_results.append({
                    "model": "KMeans",
                    "feature_space": space_name,
                    "k": k,
                    "n_init": n_init,
                    "covariance_type": None,
                    "reg_covar": None,
                    "val_ari": val_ari,
                    "val_purity": val_purity,
                    "val_combined": val_ari + val_purity,
                    "test_ari": test_ari,
                    "test_purity": test_purity,
                    "test_combined": test_ari + test_purity,
                })

            # GMM
            for cov_type in ["full", "tied", "diag", "spherical"]:
                for reg_covar in [1e-6, 1e-5, 1e-4, 1e-3]:
                    for n_init in [3, 5, 10]:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            try:
                                gmm = GaussianMixture(
                                    n_components=k,
                                    covariance_type=cov_type,
                                    reg_covar=reg_covar,
                                    n_init=n_init,
                                    init_params="kmeans",
                                    max_iter=500,
                                    random_state=random_state,
                                )
                                gmm.fit(x_train_s)
                                val_pred = gmm.predict(x_val_s)
                                test_pred = gmm.predict(x_test_s)

                                val_ari = adjusted_rand_score(y_val, val_pred)
                                val_purity = _cluster_purity(y_val, val_pred)
                                test_ari = adjusted_rand_score(y_test, test_pred)
                                test_purity = _cluster_purity(y_test, test_pred)

                                clustering_results.append({
                                    "model": "GMM",
                                    "feature_space": space_name,
                                    "k": k,
                                    "n_init": n_init,
                                    "covariance_type": cov_type,
                                    "reg_covar": reg_covar,
                                    "val_ari": val_ari,
                                    "val_purity": val_purity,
                                    "val_combined": val_ari + val_purity,
                                    "test_ari": test_ari,
                                    "test_purity": test_purity,
                                    "test_combined": test_ari + test_purity,
                                })
                            except Exception as exc:
                                LOGGER.debug(
                                    "GMM konvergencija neuspesna (k=%d, cov=%s, reg=%.1e): %s",
                                    k, cov_type, reg_covar, exc,
                                )

    clust_df = pd.DataFrame(clustering_results)
    clust_df.to_csv(output_dir / "clustering_parameter_search.csv", index=False)

    # Nadji najbolje konfiguracije po validation combined score.
    if not clust_df.empty:
        best_kmeans_idx = clust_df[clust_df["model"] == "KMeans"]["val_combined"].idxmax()
        best_gmm_idx = clust_df[clust_df["model"] == "GMM"]["val_combined"].idxmax()

        best_kmeans = clust_df.loc[best_kmeans_idx].to_dict()
        best_gmm = clust_df.loc[best_gmm_idx].to_dict()

        results["best_kmeans"] = best_kmeans
        results["best_gmm"] = best_gmm

        LOGGER.info(
            "Najbolji KMeans: k=%s, space=%s, n_init=%s — val ARI=%.4f, purity=%.4f",
            best_kmeans["k"], best_kmeans["feature_space"], best_kmeans["n_init"],
            best_kmeans["val_ari"], best_kmeans["val_purity"],
        )
        LOGGER.info(
            "Najbolji GMM: k=%s, space=%s, cov=%s, reg=%.1e — val ARI=%.4f, purity=%.4f",
            best_gmm["k"], best_gmm["feature_space"], best_gmm["covariance_type"],
            best_gmm["reg_covar"], best_gmm["val_ari"], best_gmm["val_purity"],
        )

    # --- 4. Compound-based ARI evaluacija (k=10 prioritet) ---
    LOGGER.info("Compound-based ARI evaluacija...")
    compound_col = "Compound"
    has_compound = compound_col in val_df.columns and compound_col in test_df.columns

    if has_compound:
        y_val_compound = val_df[compound_col].astype(str)
        y_test_compound = test_df[compound_col].astype(str)

        compound_results = []

        # Evaluiraj k=10 konfiguracije za oba modela i sve prostore obelezja.
        for space_name, space_data in feature_spaces.items():
            x_train_s = space_data["x_train"]
            x_val_s = space_data["x_val"]
            x_test_s = space_data["x_test"]

            # KMeans k=10
            km = KMeans(n_clusters=10, n_init=40, random_state=random_state)
            km.fit(x_train_s)
            val_pred = km.predict(x_val_s)
            test_pred = km.predict(x_test_s)

            compound_results.append({
                "model": "KMeans",
                "feature_space": space_name,
                "k": 10,
                "val_ari_team": float(adjusted_rand_score(y_val, val_pred)),
                "val_purity_team": float(_cluster_purity(y_val, val_pred)),
                "val_ari_compound": float(adjusted_rand_score(y_val_compound, val_pred)),
                "val_purity_compound": float(_cluster_purity(y_val_compound, val_pred)),
                "test_ari_team": float(adjusted_rand_score(y_test, test_pred)),
                "test_purity_team": float(_cluster_purity(y_test, test_pred)),
                "test_ari_compound": float(adjusted_rand_score(y_test_compound, test_pred)),
                "test_purity_compound": float(_cluster_purity(y_test_compound, test_pred)),
            })

            # GMM k=10, full covariance (current best baseline)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    gmm = GaussianMixture(
                        n_components=10, covariance_type="full",
                        reg_covar=1e-4, n_init=10, init_params="kmeans",
                        max_iter=500, random_state=random_state,
                    )
                    gmm.fit(x_train_s)
                    val_pred = gmm.predict(x_val_s)
                    test_pred = gmm.predict(x_test_s)

                    compound_results.append({
                        "model": "GMM",
                        "feature_space": space_name,
                        "k": 10,
                        "val_ari_team": float(adjusted_rand_score(y_val, val_pred)),
                        "val_purity_team": float(_cluster_purity(y_val, val_pred)),
                        "val_ari_compound": float(adjusted_rand_score(y_val_compound, val_pred)),
                        "val_purity_compound": float(_cluster_purity(y_val_compound, val_pred)),
                        "test_ari_team": float(adjusted_rand_score(y_test, test_pred)),
                        "test_purity_team": float(_cluster_purity(y_test, test_pred)),
                        "test_ari_compound": float(adjusted_rand_score(y_test_compound, test_pred)),
                        "test_purity_compound": float(_cluster_purity(y_test_compound, test_pred)),
                    })
                except Exception as exc:
                    LOGGER.debug("GMM compound eval neuspesna (space=%s): %s", space_name, exc)

        compound_df = pd.DataFrame(compound_results)
        compound_df.to_csv(output_dir / "compound_vs_team_ari.csv", index=False)
        results["compound_evaluation"] = compound_results

        for row in compound_results:
            LOGGER.info(
                "  %s [%s] k=10 — team ARI=%.4f, compound ARI=%.4f",
                row["model"], row["feature_space"],
                row["val_ari_team"], row["val_ari_compound"],
            )
    else:
        LOGGER.warning("Kolona '%s' nije pronadjena, preskacemo compound evaluaciju.", compound_col)

    # Sacuvaj sve rezultate.
    with (output_dir / "clustering_optimization.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    LOGGER.info("Rezultati klasterske optimizacije sacuvani u %s", output_dir)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimizacija hiperparametara F1 modela")
    parser.add_argument(
        "--mode",
        choices=["classification", "clustering", "all"],
        default="all",
        help="Koji deo optimizacije pokrenuti (default: all)",
    )
    parser.add_argument("--n-trials", type=int, default=100, help="Broj Optuna trial-ova (default: 100)")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/processed/f1_laps_processed.csv"),
        help="Putanja do procesiranih podataka",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/optimization"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-smote", action="store_true", help="Iskljuci SMOTE oversampling")
    parser.add_argument("--no-extended-features", action="store_true", help="Koristi samo bazna obelezja")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    # Smanji Optuna logovanje.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    args = parse_args()

    LOGGER.info("Ucitavanje podataka iz %s...", args.data_path)
    data = pd.read_csv(args.data_path)
    LOGGER.info("Ucitano %d redova.", len(data))

    if args.mode in ("classification", "all"):
        LOGGER.info("=== OPTIMIZACIJA KLASIFIKACIJE ===")
        cls_results = optimize_classification(
            data,
            output_dir=args.output_dir,
            n_trials=args.n_trials,
            use_extended_features=not args.no_extended_features,
            use_smote=not args.no_smote,
            random_state=args.random_state,
        )
        print("\n=== REZULTATI KLASIFIKACIJE ===")
        for model_name, model_results in cls_results.items():
            print(f"\n{model_name}:")
            print(f"  Najbolji CV macro F1:  {model_results['best_cv_macro_f1']:.4f}")
            print(f"  Test macro F1:         {model_results['test_macro_f1']:.4f}")
            print(f"  Test weighted F1:      {model_results['test_weighted_f1']:.4f}")
            print(f"  Najbolji parametri:    {model_results['best_params']}")

    if args.mode in ("clustering", "all"):
        LOGGER.info("=== OPTIMIZACIJA KLASTEROVANJA ===")
        clust_results = optimize_clustering(
            data,
            output_dir=args.output_dir,
            random_state=args.random_state,
        )
        print("\n=== REZULTATI KLASTEROVANJA ===")
        if "silhouette" in clust_results:
            sil = clust_results["silhouette"]
            print(f"\nSilhouette: optimalni k={sil['best_k']}, score={sil['best_score']:.4f}")
        if "bic_aic" in clust_results:
            bic = clust_results["bic_aic"]
            print(f"BIC: optimalni n_components={bic['best_n_components_bic']}, "
                  f"cov_type={bic['best_covariance_type_bic']}")
        if "best_kmeans" in clust_results:
            bk = clust_results["best_kmeans"]
            print(f"\nNajbolji KMeans: k={bk['k']}, space={bk['feature_space']}, "
                  f"val ARI={bk['val_ari']:.4f}, purity={bk['val_purity']:.4f}")
        if "best_gmm" in clust_results:
            bg = clust_results["best_gmm"]
            print(f"Najbolji GMM: k={bg['k']}, space={bg['feature_space']}, "
                  f"cov={bg['covariance_type']}, val ARI={bg['val_ari']:.4f}, purity={bg['val_purity']:.4f}")


if __name__ == "__main__":
    main()
