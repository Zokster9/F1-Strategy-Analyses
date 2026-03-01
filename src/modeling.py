from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    adjusted_rand_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

XGBOOST_IMPORT_ERROR: str | None = None
try:
    from xgboost import XGBClassifier
except Exception as exc:  # pragma: no cover - zavisi od okruzenja
    # Na nekim sistemima xgboost paket postoji, ali ne moze da ucita libxgboost/libomp.
    XGBClassifier = None
    XGBOOST_IMPORT_ERROR = str(exc)

LOGGER = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "LapNumber",
    "TyreLife",
    "LapTime",
    "Sector1Time",
    "Sector2Time",
    "Sector3Time",
    "MaxSpeed",
    "AvgSpeed",
]


def _safe_model_name(model_name: str) -> str:
    return model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")


def _utc_now() -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    run_timestamp = now.isoformat(timespec="seconds").replace("+00:00", "Z")
    run_id = now.strftime("%Y%m%dT%H%M%S%fZ")
    return run_id, run_timestamp


def _append_history_rows(df: pd.DataFrame, history_path: Path) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(history_path, mode="a", index=False, header=not history_path.exists())


def _model_params_json(model: object) -> str:
    if hasattr(model, "get_params"):
        params = model.get_params(deep=False)  # type: ignore[attr-defined]
        return json.dumps(params, ensure_ascii=False, sort_keys=True, default=str)
    return "{}"


def _cluster_purity(y_true: pd.Series, y_pred: np.ndarray) -> float:
    contingency = pd.crosstab(pd.Series(y_pred, name="cluster"), y_true)
    return float(contingency.max(axis=1).sum() / len(y_true))


def _event_compound_zscore(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """
    Stabilizuje skalu između različitih staza i smesa guma.
    """
    normalized = df.copy()

    if all(col in normalized.columns for col in ["Year", "EventName", "Compound"]):
        group_keys = [normalized["Year"], normalized["EventName"], normalized["Compound"].astype(str)]
        for col in feature_columns:
            normalized[col] = normalized.groupby(group_keys)[col].transform(
                lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-6)
            )

    return normalized


def _build_clustering_feature_space(
    subset: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[np.ndarray, str]:
    """
    Formira ulazni prostor za klasterovanje:
    - numeričke karakteristike normalizovane po trci+smesi
    - opciono one-hot kodiran vozač (jači signal za timsku pripadnost)
    """
    normalized = _event_compound_zscore(subset, feature_columns)

    scaler = StandardScaler()
    x_numeric = scaler.fit_transform(normalized[feature_columns])
    x_all = x_numeric
    feature_space = "event_compound_zscore_numeric"

    if "Driver" in normalized.columns:
        driver_series = normalized["Driver"].astype(str).str.strip()
        if driver_series.nunique() > 1:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            x_driver = encoder.fit_transform(driver_series.to_frame(name="Driver"))
            x_all = np.hstack([x_numeric, x_driver])
            feature_space = "event_compound_zscore_numeric_plus_driver"

    return x_all, feature_space


def _split_classification_dataset(
    data: pd.DataFrame,
    random_state: int = 42,
) -> dict[str, pd.DataFrame | pd.Series]:
    features = data[FEATURE_COLUMNS]
    target = data["Compound"].astype(str)

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=random_state,
        stratify=target,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=0.25,
        random_state=random_state,
        stratify=y_train_val,
    )

    return {
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


def evaluate_classification_models(
    data: pd.DataFrame,
    output_dir: Path,
    random_state: int = 42,
    run_id: str | None = None,
    run_timestamp: str | None = None,
) -> pd.DataFrame:
    """
    Trenira i evaluira modele klasifikacije.

    Snima:
    - classification_metrics.csv
    - confusion_matrix_test_<model>.csv
    - classification_report_test_<model>.txt
    - split_summary.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    if run_id is None or run_timestamp is None:
        generated_run_id, generated_run_timestamp = _utc_now()
        run_id = run_id or generated_run_id
        run_timestamp = run_timestamp or generated_run_timestamp

    if data.empty:
        raise ValueError("Ulazni skup za klasifikaciju je prazan.")

    subset = data.dropna(subset=FEATURE_COLUMNS + ["Compound"]).copy()
    if subset.empty:
        raise ValueError("Nema dovoljno podataka za klasifikaciju nakon ciscenja.")

    split_data = _split_classification_dataset(subset, random_state=random_state)
    y_labels = sorted(split_data["y_test"].astype(str).unique())

    models: dict[str, object] = {
        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=4,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
    }

    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_estimators=400,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        if XGBOOST_IMPORT_ERROR:
            LOGGER.warning(
                "xgboost nije dostupan (%s); evaluacija ce biti uradjena samo za RandomForest.",
                XGBOOST_IMPORT_ERROR,
            )
        else:
            LOGGER.warning("xgboost nije instaliran; evaluacija ce biti uradjena samo za RandomForest.")

    metric_rows: list[dict[str, str | float]] = []

    for model_name, model in models.items():
        LOGGER.info("Trening modela: %s", model_name)
        model_params = _model_params_json(model)

        x_train = split_data["x_train"]
        x_val = split_data["x_val"]
        x_test = split_data["x_test"]
        y_train = split_data["y_train"].astype(str)
        y_val = split_data["y_val"].astype(str)
        y_test = split_data["y_test"].astype(str)

        if model_name == "XGBoost":
            encoder = LabelEncoder()
            y_train_enc = encoder.fit_transform(y_train)

            model.fit(x_train, y_train_enc)
            y_val_pred = encoder.inverse_transform(model.predict(x_val).astype(int))
            y_test_pred = encoder.inverse_transform(model.predict(x_test).astype(int))
        else:
            model.fit(x_train, y_train)
            y_val_pred = model.predict(x_val)
            y_test_pred = model.predict(x_test)

        for split_name, y_true, y_pred in [
            ("validation", y_val, y_val_pred),
            ("test", y_test, y_test_pred),
        ]:
            metric_rows.append(
                {
                    "run_id": run_id,
                    "run_timestamp": run_timestamp,
                    "model": model_name,
                    "split": split_name,
                    "macro_f1": f1_score(y_true, y_pred, average="macro"),
                    "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
                    "model_params": model_params,
                }
            )

        cm = confusion_matrix(y_test, y_test_pred, labels=y_labels)
        cm_df = pd.DataFrame(
            cm,
            index=[f"true_{label}" for label in y_labels],
            columns=[f"pred_{label}" for label in y_labels],
        )

        safe_name = _safe_model_name(model_name)
        cm_df.to_csv(output_dir / f"confusion_matrix_test_{safe_name}.csv", index=True)
        cm_df.to_csv(runs_dir / f"confusion_matrix_test_{safe_name}__{run_id}.csv", index=True)

        report = classification_report(y_test, y_test_pred, digits=4)
        with (output_dir / f"classification_report_test_{safe_name}.txt").open("w", encoding="utf-8") as handle:
            handle.write(report)
        with (runs_dir / f"classification_report_test_{safe_name}__{run_id}.txt").open("w", encoding="utf-8") as handle:
            handle.write(report)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(output_dir / "classification_metrics.csv", index=False)
    metrics_df.to_csv(runs_dir / f"classification_metrics_{run_id}.csv", index=False)
    _append_history_rows(metrics_df, output_dir / "classification_metrics_history.csv")

    split_summary = {
        "run_id": run_id,
        "run_timestamp": run_timestamp,
        "train_size": int(len(split_data["x_train"])),
        "validation_size": int(len(split_data["x_val"])),
        "test_size": int(len(split_data["x_test"])),
        "xgboost_available": bool(XGBClassifier is not None),
        "xgboost_import_error": XGBOOST_IMPORT_ERROR,
        "model_configs": {
            model_name: (model.get_params(deep=False) if hasattr(model, "get_params") else {})
            for model_name, model in models.items()
        },
    }
    with (output_dir / "split_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(split_summary, handle, ensure_ascii=False, indent=2)
    with (runs_dir / f"split_summary_{run_id}.json").open("w", encoding="utf-8") as handle:
        json.dump(split_summary, handle, ensure_ascii=False, indent=2)
    with (output_dir / "latest_run.json").open("w", encoding="utf-8") as handle:
        json.dump({"run_id": run_id, "run_timestamp": run_timestamp}, handle, ensure_ascii=False, indent=2)

    return metrics_df


def evaluate_clustering_models(
    data: pd.DataFrame,
    output_dir: Path,
    random_state: int = 42,
    run_id: str | None = None,
    run_timestamp: str | None = None,
) -> pd.DataFrame:
    """
    Pokreće KMeans i GMM klasterovanje i evaluira ARI i purity.

    Snima:
    - clustering_metrics.csv
    - cluster_assignments_<model>.csv
    - cluster_centroids_kmeans.csv
    - cluster_means_gmm.csv
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    if run_id is None or run_timestamp is None:
        generated_run_id, generated_run_timestamp = _utc_now()
        run_id = run_id or generated_run_id
        run_timestamp = run_timestamp or generated_run_timestamp

    if data.empty:
        raise ValueError("Ulazni skup za klasterovanje je prazan.")

    team_label_col = "CanonicalTeam" if "CanonicalTeam" in data.columns else "Team"
    required_cols = FEATURE_COLUMNS + [team_label_col]
    subset = data.dropna(subset=required_cols).copy()
    if subset.empty:
        raise ValueError("Nema dovoljno podataka za klasterovanje nakon ciscenja.")

    y_true = subset[team_label_col].astype(str)
    n_clusters = y_true.nunique()
    if n_clusters < 2:
        raise ValueError("Potrebna su bar dva kanonska tima za evaluaciju klasterovanja.")

    x_clustering, feature_space = _build_clustering_feature_space(subset, FEATURE_COLUMNS)

    models = {
        "KMeans": KMeans(n_clusters=n_clusters, n_init=40, random_state=random_state),
        "GMM": GaussianMixture(
            n_components=n_clusters,
            covariance_type="diag",
            reg_covar=1e-4,
            n_init=5,
            init_params="kmeans",
            max_iter=300,
            random_state=random_state,
        ),
    }

    metric_rows: list[dict[str, str | float]] = []

    for model_name, model in models.items():
        LOGGER.info("Klasterovanje modelom: %s", model_name)
        model_params = _model_params_json(model)

        if model_name == "KMeans" and "Driver" in subset.columns:
            # Dvofazni pristup: klasterujemo profile vozača pa dodeljujemo klaster svakom krugu.
            driver_profiles = (
                pd.DataFrame(x_clustering)
                .assign(Driver=subset["Driver"].astype(str).values)
                .groupby("Driver", as_index=False)
                .mean()
            )
            driver_names = driver_profiles["Driver"].copy()
            driver_feature_matrix = driver_profiles.drop(columns=["Driver"]).to_numpy()
            driver_clusters = model.fit_predict(driver_feature_matrix)
            driver_to_cluster = dict(zip(driver_names, driver_clusters))
            predicted_clusters = subset["Driver"].astype(str).map(driver_to_cluster).to_numpy()
            clustering_strategy = "driver_profile_to_lap_assignment"
        else:
            predicted_clusters = model.fit_predict(x_clustering)
            clustering_strategy = "direct_lap_clustering"

        ari = adjusted_rand_score(y_true, predicted_clusters)
        purity = _cluster_purity(y_true, predicted_clusters)

        metric_rows.append(
            {
                "run_id": run_id,
                "run_timestamp": run_timestamp,
                "model": model_name,
                "label_basis": team_label_col,
                "feature_space": feature_space,
                "clustering_strategy": clustering_strategy,
                "n_clusters": n_clusters,
                "adjusted_rand_index": ari,
                "cluster_purity": purity,
                "model_params": model_params,
            }
        )

        assignment_cols = [col for col in ["Year", "EventName", "Driver", "Team", "CanonicalTeam"] if col in subset.columns]
        assignments = subset[assignment_cols].copy()
        assignments["cluster"] = predicted_clusters
        safe_name = _safe_model_name(model_name)
        assignments.to_csv(output_dir / f"cluster_assignments_{safe_name}.csv", index=False)
        assignments.to_csv(runs_dir / f"cluster_assignments_{safe_name}__{run_id}.csv", index=False)

        if model_name == "KMeans":
            # Profili u originalnoj skali obelezja radi lakse interpretacije.
            centers_df = (
                subset.assign(cluster=predicted_clusters)
                .groupby("cluster", as_index=False)[FEATURE_COLUMNS]
                .mean()
                .sort_values("cluster")
            )
            centers_df.to_csv(output_dir / "cluster_centroids_kmeans.csv", index=False)
            centers_df.to_csv(runs_dir / f"cluster_centroids_kmeans__{run_id}.csv", index=False)

        if model_name == "GMM":
            means_df = (
                subset.assign(cluster=predicted_clusters)
                .groupby("cluster", as_index=False)[FEATURE_COLUMNS]
                .mean()
                .sort_values("cluster")
            )
            means_df.to_csv(output_dir / "cluster_means_gmm.csv", index=False)
            means_df.to_csv(runs_dir / f"cluster_means_gmm__{run_id}.csv", index=False)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(output_dir / "clustering_metrics.csv", index=False)
    metrics_df.to_csv(runs_dir / f"clustering_metrics_{run_id}.csv", index=False)
    _append_history_rows(metrics_df, output_dir / "clustering_metrics_history.csv")
    with (output_dir / "latest_run.json").open("w", encoding="utf-8") as handle:
        json.dump({"run_id": run_id, "run_timestamp": run_timestamp}, handle, ensure_ascii=False, indent=2)

    return metrics_df
