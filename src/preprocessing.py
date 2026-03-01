from __future__ import annotations

import logging

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

COMPOUNDS = {"SOFT", "MEDIUM", "HARD"}
SPEED_COLUMNS = ["SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"]


def _normalize_team_text(team: object) -> str:
    text = str(team).upper().strip()
    normalized = "".join(ch for ch in text if ch.isalnum())
    return normalized


def _canonical_team_label(team: object) -> str | float:
    """
    Mapira istorijske nazive na kanonske nazive franšiza.
    """
    if pd.isna(team):
        return np.nan

    normalized = _normalize_team_text(team)

    if any(token in normalized for token in ["ALFAROMEO", "SAUBER", "KICKSAUBER"]):
        return "Sauber"
    if any(
        token in normalized
        for token in [
            "TOROROSSO",
            "ALPHATAURI",
            "VISCASHAPPRB",
            "VCARB",
            "RBF1TEAM",
            "RACINGBULLS",
        ]
    ):
        return "RB"
    if any(token in normalized for token in ["RACINGPOINT", "ASTONMARTIN"]):
        return "Aston Martin"
    if any(token in normalized for token in ["RENAULT", "ALPINE"]):
        return "Alpine"
    if "REDBULL" in normalized:
        return "Red Bull"
    if "MERCEDES" in normalized:
        return "Mercedes"
    if "FERRARI" in normalized:
        return "Ferrari"
    if "MCLAREN" in normalized:
        return "McLaren"
    if "HAAS" in normalized:
        return "Haas"
    if "WILLIAMS" in normalized:
        return "Williams"

    # Fallback za neočekivane vrednosti koje ne možemo sigurno mapirati.
    return str(team).strip()


def _timedelta_to_seconds(series: pd.Series) -> pd.Series:
    return pd.to_timedelta(series, errors="coerce").dt.total_seconds()


def _is_green_like_status(status: object) -> bool:
    """
    Vraća False za statuse koji impliciraju SC/VSC/prekid,
    kako bi se izbacili spori/nenormalni krugovi.
    """
    if pd.isna(status):
        return True

    text = str(status)
    forbidden_codes = {"4", "5", "6", "7"}
    return not any(code in text for code in forbidden_codes)


def _remove_iqr_outliers(
    df: pd.DataFrame,
    value_column: str,
    group_columns: list[str],
    iqr_multiplier: float = 1.5,
) -> pd.DataFrame:
    q1 = df.groupby(group_columns)[value_column].transform(lambda s: s.quantile(0.25))
    q3 = df.groupby(group_columns)[value_column].transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1

    lower = q1 - iqr_multiplier * iqr
    upper = q3 + iqr_multiplier * iqr

    mask = (df[value_column] >= lower) & (df[value_column] <= upper)
    return df.loc[mask].copy()


def preprocess_laps(raw_laps: pd.DataFrame) -> pd.DataFrame:
    """
    Transformiše sirove FastF1 krugove u format spreman za modelovanje.
    """
    if raw_laps.empty:
        raise ValueError("Sirov skup podataka je prazan; nije moguca obrada.")

    df = raw_laps.copy()

    # Standardizacija tipa gume i filtriranje samo na suve smese.
    df["Compound"] = df["Compound"].astype(str).str.upper().str.strip()
    df = df[df["Compound"].isin(COMPOUNDS)]

    # Uklanjanje prvog kruga i krugova sa ulaskom/izlaskom iz boksa.
    df["LapNumber"] = pd.to_numeric(df["LapNumber"], errors="coerce")
    df = df[df["LapNumber"] > 1]

    if "PitOutTime" in df.columns:
        df = df[df["PitOutTime"].isna()]
    if "PitInTime" in df.columns:
        df = df[df["PitInTime"].isna()]

    # Ako je informacija dostupna, zadrzavamo samo precizne krugove.
    if "IsAccurate" in df.columns:
        df = df[df["IsAccurate"].isna() | (df["IsAccurate"] == True)]  # noqa: E712

    # Uklanjanje krugova pod SC/VSC i slicnim uslovima.
    if "TrackStatus" in df.columns:
        df = df[df["TrackStatus"].apply(_is_green_like_status)]

    # Konverzija vremena u sekunde.
    df["LapTime"] = _timedelta_to_seconds(df["LapTime"])
    df["Sector1Time"] = _timedelta_to_seconds(df["Sector1Time"])
    df["Sector2Time"] = _timedelta_to_seconds(df["Sector2Time"])
    df["Sector3Time"] = _timedelta_to_seconds(df["Sector3Time"])

    # Bazicna validacija vremena kruga i sektora.
    time_columns = ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]
    df = df.dropna(subset=time_columns)
    df = df[(df["LapTime"] >= 50.0) & (df["LapTime"] <= 200.0)]

    # Brzine i dodatna obelezja.
    for col in SPEED_COLUMNS + ["TyreLife"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    available_speed_cols = [col for col in SPEED_COLUMNS if col in df.columns]
    if available_speed_cols:
        df["MaxSpeed"] = df[available_speed_cols].max(axis=1)
        df["AvgSpeed"] = df[available_speed_cols].mean(axis=1)
    else:
        df["MaxSpeed"] = np.nan
        df["AvgSpeed"] = np.nan

    model_columns = [
        "LapNumber",
        "TyreLife",
        "LapTime",
        "Sector1Time",
        "Sector2Time",
        "Sector3Time",
        "MaxSpeed",
        "AvgSpeed",
    ]
    df = df.dropna(subset=model_columns)

    if "Team" in df.columns:
        df["CanonicalTeam"] = df["Team"].apply(_canonical_team_label)
    else:
        df["CanonicalTeam"] = np.nan

    # Uklanjanje outlier krugova po vozacu i trci.
    group_columns = [col for col in ["Year", "EventName", "Driver"] if col in df.columns]
    if group_columns:
        df = _remove_iqr_outliers(df, value_column="LapTime", group_columns=group_columns)

    # Standardizacija tipa gume za klasifikaciju.
    df["Compound"] = pd.Categorical(
        df["Compound"], categories=["SOFT", "MEDIUM", "HARD"], ordered=False
    )

    ordered_columns = [
        "Year",
        "RoundNumber",
        "EventName",
        "Country",
        "Location",
        "Driver",
        "Team",
        "CanonicalTeam",
        "LapNumber",
        "TyreLife",
        "Compound",
        "LapTime",
        "Sector1Time",
        "Sector2Time",
        "Sector3Time",
        "MaxSpeed",
        "AvgSpeed",
    ]
    final_columns = [col for col in ordered_columns if col in df.columns]
    df = df[final_columns].copy()

    sort_columns = [col for col in ["Year", "RoundNumber", "Driver", "LapNumber"] if col in df.columns]
    if sort_columns:
        df = df.sort_values(sort_columns).reset_index(drop=True)

    LOGGER.info("Nakon obrade ostalo je %s krugova", len(df))
    return df
