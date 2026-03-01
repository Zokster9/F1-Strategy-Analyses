from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import time

import fastf1
import pandas as pd

LOGGER = logging.getLogger(__name__)

LAP_COLUMNS = [
    "Driver",
    "Team",
    "LapNumber",
    "TyreLife",
    "Compound",
    "LapTime",
    "Sector1Time",
    "Sector2Time",
    "Sector3Time",
    "SpeedI1",
    "SpeedI2",
    "SpeedFL",
    "SpeedST",
    "TrackStatus",
    "PitOutTime",
    "PitInTime",
    "IsAccurate",
]


@dataclass
class CollectionConfig:
    start_year: int = 2019
    end_year: int = 2025
    session_type: str = "R"
    cache_dir: Path = Path("data/cache")
    max_schedule_retries: int = 3
    max_session_retries: int = 3
    max_collection_passes: int = 3
    retry_sleep_seconds: float = 1.5


def configure_fastf1_cache(cache_dir: Path) -> None:
    """Konfiguriše FastF1 cache direktorijum."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))


def _event_key(year: int, round_number: object, event_name: str) -> tuple[int, str, str]:
    if pd.notna(round_number):
        return year, str(int(round_number)), str(event_name)
    return year, "NA", str(event_name)


def _load_schedule_with_retries(config: CollectionConfig, year: int) -> pd.DataFrame | None:
    last_exc: Exception | None = None
    for attempt in range(1, config.max_schedule_retries + 1):
        try:
            return fastf1.get_event_schedule(year, include_testing=False)
        except Exception as exc:  # pragma: no cover - zavisi od mreze/FastF1 API-ja
            last_exc = exc
            LOGGER.warning(
                "Neuspesno citanje rasporeda za %s (pokusaj %s/%s): %s",
                year,
                attempt,
                config.max_schedule_retries,
                exc,
            )
            if attempt < config.max_schedule_retries:
                time.sleep(config.retry_sleep_seconds)

    LOGGER.error("Preskacem sezonu %s: raspored nije dostupan posle retry-ja: %s", year, last_exc)
    return None


def _load_event_laps_with_retries(
    config: CollectionConfig,
    year: int,
    round_number: object,
    event_name: str,
) -> pd.DataFrame | None:
    last_exc: Exception | None = None

    for attempt in range(1, config.max_session_retries + 1):
        try:
            identifier = int(round_number) if pd.notna(round_number) else event_name
            session = fastf1.get_session(year, identifier, config.session_type)
            session.load(laps=True, telemetry=False, weather=False, messages=False)
            return session.laps.copy()
        except Exception as exc:  # pragma: no cover - zavisi od mreze/FastF1 API-ja
            last_exc = exc
            LOGGER.warning(
                "Neuspesno citanje trke %s (%s) sezona %s (pokusaj %s/%s): %s",
                event_name,
                round_number,
                year,
                attempt,
                config.max_session_retries,
                exc,
            )
            if attempt < config.max_session_retries:
                time.sleep(config.retry_sleep_seconds)

    LOGGER.warning(
        "Preskacem trku %s (%s) sezona %s posle retry-ja: %s",
        event_name,
        round_number,
        year,
        last_exc,
    )
    return None


def collect_race_laps(config: CollectionConfig) -> pd.DataFrame:
    """
    Prikuplja krugove sa trka Formule 1 za zadati opseg sezona.

    Povratna vrednost je DataFrame sa metapodacima o sezoni/trci i kolonama
    potrebnim za dalju obradu.
    """
    configure_fastf1_cache(config.cache_dir)

    events_map: dict[tuple[int, str, str], dict[str, object]] = {}
    for year in range(config.start_year, config.end_year + 1):
        LOGGER.info("Ucitavanje rasporeda za sezonu %s", year)
        events = _load_schedule_with_retries(config, year)
        if events is None:
            continue

        for _, event in events.iterrows():
            event_name = event.get("EventName", "UnknownEvent")
            round_number = event.get("RoundNumber")
            key = _event_key(year, round_number, event_name)
            events_map[key] = {
                "Year": year,
                "RoundNumber": round_number,
                "EventName": event_name,
                "Country": event.get("Country"),
                "Location": event.get("Location"),
                "OfficialEventName": event.get("OfficialEventName"),
            }

    if not events_map:
        columns = LAP_COLUMNS + [
            "Year",
            "RoundNumber",
            "EventName",
            "Country",
            "Location",
            "OfficialEventName",
        ]
        return pd.DataFrame(columns=columns)

    loaded_map: dict[tuple[int, str, str], pd.DataFrame] = {}
    empty_events: set[tuple[int, str, str]] = set()

    for pass_idx in range(1, config.max_collection_passes + 1):
        pending_keys = [key for key in events_map if key not in loaded_map and key not in empty_events]
        if not pending_keys:
            break

        LOGGER.info(
            "Pass %s/%s: preostalo trka za ucitavanje %s",
            pass_idx,
            config.max_collection_passes,
            len(pending_keys),
        )

        progress_in_pass = 0
        for key in sorted(pending_keys, key=lambda k: (k[0], k[1], k[2])):
            info = events_map[key]
            year = int(info["Year"])
            round_number = info["RoundNumber"]
            event_name = str(info["EventName"])

            laps = _load_event_laps_with_retries(config, year, round_number, event_name)
            if laps is None:
                continue
            if laps.empty:
                empty_events.add(key)
                LOGGER.warning("Trka %s (%s, %s) vraca prazan laps dataset.", event_name, year, round_number)
                continue

            existing_columns = [col for col in LAP_COLUMNS if col in laps.columns]
            selected = laps[existing_columns].copy()
            selected["Year"] = info["Year"]
            selected["RoundNumber"] = info["RoundNumber"]
            selected["EventName"] = info["EventName"]
            selected["Country"] = info["Country"]
            selected["Location"] = info["Location"]
            selected["OfficialEventName"] = info["OfficialEventName"]

            loaded_map[key] = selected
            progress_in_pass += 1
            LOGGER.info(
                "Ucitano %s krugova za %s %s (%s)",
                len(selected),
                year,
                event_name,
                round_number,
            )

        loaded_by_year: dict[int, int] = {}
        expected_by_year: dict[int, int] = {}
        for key in events_map:
            year = key[0]
            expected_by_year[year] = expected_by_year.get(year, 0) + 1
            if key in loaded_map or key in empty_events:
                loaded_by_year[year] = loaded_by_year.get(year, 0) + 1

        for year in sorted(expected_by_year):
            LOGGER.info(
                "Coverage sezona %s: %s/%s trka (loaded/known-empty).",
                year,
                loaded_by_year.get(year, 0),
                expected_by_year[year],
            )

        if progress_in_pass == 0:
            LOGGER.warning("Nema napretka u pass-u %s; prekidam dalje prolaze.", pass_idx)
            break

    missing_keys = [key for key in events_map if key not in loaded_map and key not in empty_events]
    if missing_keys:
        LOGGER.warning(
            "I dalje nedostaje %s trka nakon svih prolaza. Primer: %s",
            len(missing_keys),
            [f"{k[0]}-{k[2]}(round={k[1]})" for k in missing_keys[:5]],
        )

    frames = list(loaded_map.values())
    if not frames:
        columns = LAP_COLUMNS + [
            "Year",
            "RoundNumber",
            "EventName",
            "Country",
            "Location",
            "OfficialEventName",
        ]
        return pd.DataFrame(columns=columns)

    combined = pd.concat(frames, ignore_index=True)
    LOGGER.info("Ukupno ucitano %s krugova", len(combined))
    return combined
