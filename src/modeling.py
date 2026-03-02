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

    def _runs_pattern_for_history(path: Path) -> str | None:
        name = path.name
        if name == "classification_metrics_history.csv":
            return "classification_metrics_*.csv"
        if name == "clustering_metrics_history.csv":
            return "clustering_metrics_*.csv"
        return None

    def _rebuild_history_from_runs(path: Path) -> bool:
        pattern = _runs_pattern_for_history(path)
        if pattern is None:
            return False

        runs_dir = path.parent / "runs"
        run_files = sorted(runs_dir.glob(pattern))
        if not run_files:
            return False

        frames: list[pd.DataFrame] = []
        columns_order: list[str] = []
        for run_file in run_files:
            try:
                run_df = pd.read_csv(run_file)
            except Exception:
                run_df = pd.read_csv(run_file, engine="python", on_bad_lines="skip")
            if run_df.empty:
                continue
            frames.append(run_df)
            for col in run_df.columns:
                if col not in columns_order:
                    columns_order.append(col)

        if not frames:
            return False

        merged = pd.concat([frame.reindex(columns=columns_order) for frame in frames], ignore_index=True)
        if "run_timestamp" in merged.columns:
            merged["run_timestamp"] = pd.to_datetime(merged["run_timestamp"], errors="coerce")
            merged = merged.sort_values("run_timestamp", kind="stable")
            merged["run_timestamp"] = merged["run_timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        merged.to_csv(path, index=False)
        return True

    if not history_path.exists():
        df.to_csv(history_path, mode="a", index=False, header=True)
        return

    try:
        existing_cols = pd.read_csv(history_path, nrows=0).columns.tolist()
    except Exception:
        if _rebuild_history_from_runs(history_path):
            existing_cols = pd.read_csv(history_path, nrows=0).columns.tolist()
        else:
            existing_cols = list(df.columns)

    if existing_cols != list(df.columns):
        # Ako se schema promenila između run-ova, rebuild history iz run-specifičnih fajlova
        # da bismo izbegli ParserError i zadržali sve dostupne kolone.
        if _rebuild_history_from_runs(history_path):
            return

        aligned = df.reindex(columns=existing_cols)
        aligned.to_csv(history_path, mode="a", index=False, header=False)
        return

    df.to_csv(history_path, mode="a", index=False, header=False)


def _model_params_json(model: object) -> str:
    if hasattr(model, "get_params"):
        params = model.get_params(deep=False)  # type: ignore[attr-defined]
        return json.dumps(params, ensure_ascii=False, sort_keys=True, default=str)
    return "{}"


def _cluster_purity(y_true: pd.Series, y_pred: np.ndarray) -> float:
    y_true_aligned = pd.Series(np.asarray(y_true), name="label")
    y_pred_aligned = pd.Series(np.asarray(y_pred), name="cluster")
    contingency = pd.crosstab(y_pred_aligned, y_true_aligned)
    return float(contingency.max(axis=1).sum() / len(y_true))


class _GroupedIQRFilter:
    """
    IQR outlier filter koji se fituje na TRAIN splitu i primenjuje na ostale splitove.
    """

    def __init__(
        self,
        value_column: str,
        group_columns: list[str],
        iqr_multiplier: float = 1.5,
    ) -> None:
        self.value_column = value_column
        self.group_columns = group_columns
        self.iqr_multiplier = iqr_multiplier
        self.thresholds: pd.DataFrame | None = None
        self.global_lower: float | None = None
        self.global_upper: float | None = None

    def fit(self, df: pd.DataFrame) -> "_GroupedIQRFilter":
        if self.value_column not in df.columns:
            raise ValueError(f"Nedostaje kolona '{self.value_column}' za IQR filter.")
        if df.empty:
            raise ValueError("Ne moze se fitovati IQR filter na praznom skupu.")

        values = pd.to_numeric(df[self.value_column], errors="coerce")
        q1_global = float(values.quantile(0.25))
        q3_global = float(values.quantile(0.75))
        iqr_global = q3_global - q1_global
        self.global_lower = q1_global - self.iqr_multiplier * iqr_global
        self.global_upper = q3_global + self.iqr_multiplier * iqr_global

        if self.group_columns:
            grouped = (
                df.groupby(self.group_columns, dropna=False)[self.value_column]
                .agg(q1=lambda s: s.quantile(0.25), q3=lambda s: s.quantile(0.75))
                .reset_index()
            )
            iqr = grouped["q3"] - grouped["q1"]
            grouped["lower"] = grouped["q1"] - self.iqr_multiplier * iqr
            grouped["upper"] = grouped["q3"] + self.iqr_multiplier * iqr
            self.thresholds = grouped[self.group_columns + ["lower", "upper"]]

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.global_lower is None or self.global_upper is None:
            raise ValueError("IQR filter mora biti fitovan pre transformacije.")
        if df.empty:
            return df.copy()

        values = pd.to_numeric(df[self.value_column], errors="coerce")

        if self.group_columns and self.thresholds is not None and set(self.group_columns).issubset(df.columns):
            keys = df[self.group_columns].copy()
            joined = keys.merge(self.thresholds, on=self.group_columns, how="left")
            lower = joined["lower"].fillna(self.global_lower).to_numpy(dtype=float)
            upper = joined["upper"].fillna(self.global_upper).to_numpy(dtype=float)
        else:
            lower = np.full(len(df), self.global_lower, dtype=float)
            upper = np.full(len(df), self.global_upper, dtype=float)

        mask = (values.to_numpy(dtype=float) >= lower) & (values.to_numpy(dtype=float) <= upper)
        return df.loc[mask].copy()


class _EventCompoundZScoreTransformer:
    """
    Fituje statistiku na TRAIN skupu i transformiše VALIDATION/TEST bez leakage-a.
    """

    def __init__(self, feature_columns: list[str]) -> None:
        self.feature_columns = feature_columns
        self.group_means: pd.DataFrame | None = None
        self.group_stds: pd.DataFrame | None = None
        self.global_means: pd.Series | None = None
        self.global_stds: pd.Series | None = None
        self.use_grouping = False

    def fit(self, df: pd.DataFrame) -> "_EventCompoundZScoreTransformer":
        if df.empty:
            raise ValueError("Ne moze se fitovati z-score transformacija na praznom skupu.")

        self.global_means = df[self.feature_columns].mean(numeric_only=True)
        global_stds = df[self.feature_columns].std(ddof=0, numeric_only=True)
        self.global_stds = global_stds.replace(0, np.nan).fillna(1.0)

        has_group_cols = {"Year", "EventName", "Compound"}.issubset(df.columns)
        if has_group_cols:
            train_grouped = df.copy()
            train_grouped["Compound"] = train_grouped["Compound"].astype(str)
            grouped = train_grouped.groupby(["Year", "EventName", "Compound"], dropna=False)[self.feature_columns]

            self.group_means = grouped.mean().reset_index()
            self.group_stds = grouped.std(ddof=0).replace(0, np.nan).reset_index()
            self.use_grouping = True

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.global_means is None or self.global_stds is None:
            raise ValueError("Z-score transformer mora biti fitovan pre transformacije.")

        transformed = df.copy()
        has_group_cols = {"Year", "EventName", "Compound"}.issubset(transformed.columns)

        if self.use_grouping and has_group_cols and self.group_means is not None and self.group_stds is not None:
            keys = transformed.loc[:, ["Year", "EventName"]].copy()
            keys["Compound"] = transformed["Compound"].astype(str)

            means_df = keys.merge(self.group_means, on=["Year", "EventName", "Compound"], how="left")
            stds_df = keys.merge(self.group_stds, on=["Year", "EventName", "Compound"], how="left")

            for col in self.feature_columns:
                means = means_df[col].fillna(self.global_means[col]).to_numpy(dtype=float)
                stds = stds_df[col].fillna(self.global_stds[col]).replace(0, np.nan).fillna(1.0).to_numpy(dtype=float)
                values = transformed[col].to_numpy(dtype=float)
                transformed[col] = (values - means) / (stds + 1e-6)
        else:
            for col in self.feature_columns:
                transformed[col] = (transformed[col] - self.global_means[col]) / (self.global_stds[col] + 1e-6)

        return transformed


class _ClusteringFeaturePreprocessor:
    """
    Train-fitted preprocessing za klasterovanje:
    1) event+compound z-score (fit na train),
    2) global StandardScaler (fit na train),
    3) opcioni Driver one-hot encoder (fit na train).
    """

    def __init__(self, feature_columns: list[str]) -> None:
        self.feature_columns = feature_columns
        self.zscore = _EventCompoundZScoreTransformer(feature_columns)
        self.scaler = StandardScaler()
        self.driver_encoder: OneHotEncoder | None = None
        self.include_driver = False
        self.feature_space = "event_compound_zscore_numeric"

    def fit(self, train_df: pd.DataFrame) -> "_ClusteringFeaturePreprocessor":
        normalized_train = self.zscore.fit(train_df).transform(train_df)
        self.scaler.fit(normalized_train[self.feature_columns])

        if "Driver" in train_df.columns:
            driver_series = train_df["Driver"].astype(str).str.strip()
            if driver_series.nunique() > 1:
                self.driver_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                self.driver_encoder.fit(driver_series.to_frame(name="Driver"))
                self.include_driver = True
                self.feature_space = "event_compound_zscore_numeric_plus_driver"

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        normalized = self.zscore.transform(df)
        x_numeric = self.scaler.transform(normalized[self.feature_columns])

        if self.include_driver and self.driver_encoder is not None and "Driver" in df.columns:
            driver = df["Driver"].astype(str).str.strip().to_frame(name="Driver")
            x_driver = self.driver_encoder.transform(driver)
            return np.hstack([x_numeric, x_driver])
        return x_numeric


def _split_classification_dataset(
    data: pd.DataFrame,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    target = data["Compound"].astype(str)

    train_val, test = train_test_split(
        data,
        test_size=0.2,
        random_state=random_state,
        stratify=target,
    )
    y_train_val = train_val["Compound"].astype(str)
    train, val = train_test_split(
        train_val,
        test_size=0.25,
        random_state=random_state,
        stratify=y_train_val,
    )

    return {
        "train": train.copy(),
        "validation": val.copy(),
        "test": test.copy(),
    }


def _split_clustering_dataset(
    data: pd.DataFrame,
    label_col: str,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    y = data[label_col].astype(str)
    train_val, test = train_test_split(
        data,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )
    y_train_val = train_val[label_col].astype(str)
    train, val = train_test_split(
        train_val,
        test_size=0.25,
        random_state=random_state,
        stratify=y_train_val,
    )
    return {"train": train.copy(), "validation": val.copy(), "test": test.copy()}


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

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Nakon train-fitted IQR filtriranja nema dovoljno podataka za klasifikaciju.")

    y_labels = sorted(subset["Compound"].astype(str).unique())

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

        x_train = train_df[FEATURE_COLUMNS]
        x_val = val_df[FEATURE_COLUMNS]
        x_test = test_df[FEATURE_COLUMNS]
        y_train = train_df["Compound"].astype(str)
        y_val = val_df["Compound"].astype(str)
        y_test = test_df["Compound"].astype(str)

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
        "train_size_before_outlier_filter": int(len(raw_train_df)),
        "validation_size_before_outlier_filter": int(len(raw_val_df)),
        "test_size_before_outlier_filter": int(len(raw_test_df)),
        "train_size": int(len(train_df)),
        "validation_size": int(len(val_df)),
        "test_size": int(len(test_df)),
        "preprocessing": {
            "fit_on": "train_only",
            "iqr_outlier_filter": {
                "value_column": "LapTime",
                "group_columns": outlier_group_columns,
                "iqr_multiplier": 1.5,
                "rows_removed": {
                    "train": int(len(raw_train_df) - len(train_df)),
                    "validation": int(len(raw_val_df) - len(val_df)),
                    "test": int(len(raw_test_df) - len(test_df)),
                },
            },
            "numeric_scaler": None,
            "applied_to": ["train", "validation", "test"],
        },
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

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Nakon train-fitted IQR filtriranja nema dovoljno podataka za klasterovanje.")

    n_clusters = train_df[team_label_col].astype(str).nunique()
    if n_clusters < 2:
        raise ValueError("Potrebna su bar dva kanonska tima za evaluaciju klasterovanja.")

    preprocessor = _ClusteringFeaturePreprocessor(FEATURE_COLUMNS).fit(train_df)
    x_train = preprocessor.transform(train_df)
    x_val = preprocessor.transform(val_df)
    x_test = preprocessor.transform(test_df)
    y_by_split = {
        "validation": val_df[team_label_col].astype(str),
        "test": test_df[team_label_col].astype(str),
    }

    def _score_split(y_true: pd.Series, y_pred: np.ndarray) -> tuple[float, float, float]:
        ari = adjusted_rand_score(y_true, y_pred)
        purity = _cluster_purity(y_true, y_pred)
        return ari + purity, ari, purity

    metric_rows: list[dict[str, str | float]] = []
    assignment_cols = [col for col in ["Year", "EventName", "Driver", "Team", "CanonicalTeam"] if col in subset.columns]

    # ---- KMeans: biramo bolju strategiju po validation (direct vs driver_profile) ----
    LOGGER.info("Klasterovanje modelom: KMeans")
    kmeans_candidates: list[dict[str, object]] = []

    km_direct = KMeans(n_clusters=n_clusters, n_init=40, random_state=random_state)
    km_direct.fit(x_train)
    pred_direct = {
        "train": km_direct.predict(x_train),
        "validation": km_direct.predict(x_val),
        "test": km_direct.predict(x_test),
    }
    direct_score, _, _ = _score_split(y_by_split["validation"], pred_direct["validation"])
    kmeans_candidates.append(
        {
            "model": km_direct,
            "strategy": "direct_lap_clustering_trainfit",
            "predicted_by_split": pred_direct,
            "validation_score": direct_score,
        }
    )

    if "Driver" in train_df.columns:
        km_profile = KMeans(n_clusters=n_clusters, n_init=40, random_state=random_state)
        train_profiles = (
            pd.DataFrame(x_train)
            .assign(Driver=train_df["Driver"].astype(str).values)
            .groupby("Driver", as_index=False)
            .mean()
        )
        km_profile.fit(train_profiles.drop(columns=["Driver"]).to_numpy())

        def _predict_from_driver_profiles(x_split: np.ndarray, split_df: pd.DataFrame) -> np.ndarray:
            profiles = (
                pd.DataFrame(x_split)
                .assign(Driver=split_df["Driver"].astype(str).values)
                .groupby("Driver", as_index=False)
                .mean()
            )
            predicted_driver = km_profile.predict(profiles.drop(columns=["Driver"]).to_numpy())
            driver_to_cluster = dict(zip(profiles["Driver"], predicted_driver))
            return split_df["Driver"].astype(str).map(driver_to_cluster).to_numpy()

        pred_profile = {
            "train": _predict_from_driver_profiles(x_train, train_df),
            "validation": _predict_from_driver_profiles(x_val, val_df),
            "test": _predict_from_driver_profiles(x_test, test_df),
        }
        profile_score, _, _ = _score_split(y_by_split["validation"], pred_profile["validation"])
        kmeans_candidates.append(
            {
                "model": km_profile,
                "strategy": "driver_profile_trainfit_driver_profile_predict",
                "predicted_by_split": pred_profile,
                "validation_score": profile_score,
            }
        )

    best_kmeans = max(kmeans_candidates, key=lambda item: float(item["validation_score"]))
    kmeans_model = best_kmeans["model"]  # type: ignore[assignment]
    kmeans_strategy = str(best_kmeans["strategy"])
    kmeans_pred = best_kmeans["predicted_by_split"]  # type: ignore[assignment]
    kmeans_params = _model_params_json(kmeans_model)

    for split_name in ["validation", "test"]:
        y_true = y_by_split[split_name]
        y_pred = kmeans_pred[split_name]
        _, ari, purity = _score_split(y_true, y_pred)
        metric_rows.append(
            {
                "run_id": run_id,
                "run_timestamp": run_timestamp,
                "model": "KMeans",
                "split": split_name,
                "label_basis": team_label_col,
                "feature_space": preprocessor.feature_space,
                "clustering_strategy": kmeans_strategy,
                "n_clusters": n_clusters,
                "adjusted_rand_index": ari,
                "cluster_purity": purity,
                "model_params": kmeans_params,
            }
        )

    safe_name = _safe_model_name("KMeans")
    for split_name, split_df in {"validation": val_df, "test": test_df}.items():
        assignments = split_df[assignment_cols].copy()
        assignments["cluster"] = kmeans_pred[split_name]
        assignments.to_csv(output_dir / f"cluster_assignments_{safe_name}_{split_name}.csv", index=False)
        assignments.to_csv(runs_dir / f"cluster_assignments_{safe_name}_{split_name}__{run_id}.csv", index=False)

    centers_df = (
        train_df.assign(cluster=kmeans_pred["train"])
        .groupby("cluster", as_index=False)[FEATURE_COLUMNS]
        .mean()
        .sort_values("cluster")
    )
    centers_df.to_csv(output_dir / "cluster_centroids_kmeans.csv", index=False)
    centers_df.to_csv(runs_dir / f"cluster_centroids_kmeans__{run_id}.csv", index=False)

    # ---- GMM: biramo najbolji covariance_type po validation ----
    LOGGER.info("Klasterovanje modelom: GMM")
    gmm_candidates: list[dict[str, object]] = []
    for covariance_type in ["diag", "full", "tied"]:
        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type=covariance_type,
            reg_covar=1e-4,
            n_init=5,
            init_params="kmeans",
            max_iter=300,
            random_state=random_state,
        )
        gmm.fit(x_train)
        pred = {
            "train": gmm.predict(x_train),
            "validation": gmm.predict(x_val),
            "test": gmm.predict(x_test),
        }
        val_score, _, _ = _score_split(y_by_split["validation"], pred["validation"])
        gmm_candidates.append(
            {
                "model": gmm,
                "strategy": f"direct_lap_clustering_trainfit_cov_{covariance_type}",
                "predicted_by_split": pred,
                "validation_score": val_score,
            }
        )

    best_gmm = max(gmm_candidates, key=lambda item: float(item["validation_score"]))
    gmm_model = best_gmm["model"]  # type: ignore[assignment]
    gmm_strategy = str(best_gmm["strategy"])
    gmm_pred = best_gmm["predicted_by_split"]  # type: ignore[assignment]
    gmm_params = _model_params_json(gmm_model)

    for split_name in ["validation", "test"]:
        y_true = y_by_split[split_name]
        y_pred = gmm_pred[split_name]
        _, ari, purity = _score_split(y_true, y_pred)
        metric_rows.append(
            {
                "run_id": run_id,
                "run_timestamp": run_timestamp,
                "model": "GMM",
                "split": split_name,
                "label_basis": team_label_col,
                "feature_space": preprocessor.feature_space,
                "clustering_strategy": gmm_strategy,
                "n_clusters": n_clusters,
                "adjusted_rand_index": ari,
                "cluster_purity": purity,
                "model_params": gmm_params,
            }
        )

    safe_name = _safe_model_name("GMM")
    for split_name, split_df in {"validation": val_df, "test": test_df}.items():
        assignments = split_df[assignment_cols].copy()
        assignments["cluster"] = gmm_pred[split_name]
        assignments.to_csv(output_dir / f"cluster_assignments_{safe_name}_{split_name}.csv", index=False)
        assignments.to_csv(runs_dir / f"cluster_assignments_{safe_name}_{split_name}__{run_id}.csv", index=False)

    means_df = (
        train_df.assign(cluster=gmm_pred["train"])
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

    split_summary = {
        "run_id": run_id,
        "run_timestamp": run_timestamp,
        "train_size_before_outlier_filter": int(len(raw_train_df)),
        "validation_size_before_outlier_filter": int(len(raw_val_df)),
        "test_size_before_outlier_filter": int(len(raw_test_df)),
        "train_size": int(len(train_df)),
        "validation_size": int(len(val_df)),
        "test_size": int(len(test_df)),
        "preprocessing": {
            "fit_on": "train_only",
            "iqr_outlier_filter": {
                "value_column": "LapTime",
                "group_columns": outlier_group_columns,
                "iqr_multiplier": 1.5,
                "rows_removed": {
                    "train": int(len(raw_train_df) - len(train_df)),
                    "validation": int(len(raw_val_df) - len(val_df)),
                    "test": int(len(raw_test_df) - len(test_df)),
                },
            },
            "event_compound_zscore": True,
            "numeric_scaler": "StandardScaler",
            "driver_one_hot": preprocessor.include_driver,
            "applied_to": ["train", "validation", "test"],
        },
        "model_selection": {
            "kmeans_strategy_selected_on": "validation_adjusted_rand_plus_purity",
            "gmm_covariance_selected_on": "validation_adjusted_rand_plus_purity",
        },
        "n_clusters": n_clusters,
        "label_basis": team_label_col,
    }
    with (output_dir / "split_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(split_summary, handle, ensure_ascii=False, indent=2)
    with (runs_dir / f"split_summary_{run_id}.json").open("w", encoding="utf-8") as handle:
        json.dump(split_summary, handle, ensure_ascii=False, indent=2)
    with (output_dir / "latest_run.json").open("w", encoding="utf-8") as handle:
        json.dump({"run_id": run_id, "run_timestamp": run_timestamp}, handle, ensure_ascii=False, indent=2)

    return metrics_df
