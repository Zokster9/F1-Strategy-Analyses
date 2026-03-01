from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import logging
import pandas as pd

from src.data_collection import CollectionConfig, collect_race_laps
from src.modeling import evaluate_classification_models, evaluate_clustering_models
from src.preprocessing import preprocess_laps


def run_pipeline(
    start_year: int = 2019,
    end_year: int = 2025,
    raw_output_path: Path = Path("data/raw/f1_laps_raw.csv"),
    processed_output_path: Path = Path("data/processed/f1_laps_processed.csv"),
    results_dir: Path = Path("results"),
    cache_dir: Path = Path("data/cache"),
    random_state: int = 42,
    max_schedule_retries: int = 3,
    max_session_retries: int = 3,
    max_collection_passes: int = 3,
    retry_sleep_seconds: float = 1.5,
) -> dict[str, object]:
    """Pokrece ceo tok: prikupljanje -> obrada -> modeli -> evaluacija."""
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_output_path.parent.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S%fZ")
    run_timestamp = now.isoformat(timespec="seconds").replace("+00:00", "Z")

    collection_config = CollectionConfig(
        start_year=start_year,
        end_year=end_year,
        cache_dir=cache_dir,
        max_schedule_retries=max_schedule_retries,
        max_session_retries=max_session_retries,
        max_collection_passes=max_collection_passes,
        retry_sleep_seconds=retry_sleep_seconds,
    )

    raw_laps = collect_race_laps(collection_config)
    raw_laps.to_csv(raw_output_path, index=False)

    processed_laps = preprocess_laps(raw_laps)
    processed_laps.to_csv(processed_output_path, index=False)

    classification_dir = results_dir / "classification"
    clustering_dir = results_dir / "clustering"

    classification_metrics = evaluate_classification_models(
        processed_laps,
        output_dir=classification_dir,
        random_state=random_state,
        run_id=run_id,
        run_timestamp=run_timestamp,
    )
    clustering_metrics = evaluate_clustering_models(
        processed_laps,
        output_dir=clustering_dir,
        random_state=random_state,
        run_id=run_id,
        run_timestamp=run_timestamp,
    )

    summary = {
        "run_id": run_id,
        "run_timestamp": run_timestamp,
        "raw_rows": int(len(raw_laps)),
        "processed_rows": int(len(processed_laps)),
        "classification_results_file": str(classification_dir / "classification_metrics.csv"),
        "classification_history_file": str(classification_dir / "classification_metrics_history.csv"),
        "clustering_results_file": str(clustering_dir / "clustering_metrics.csv"),
        "clustering_history_file": str(clustering_dir / "clustering_metrics_history.csv"),
        "classification_models": classification_metrics["model"].drop_duplicates().tolist(),
        "clustering_models": clustering_metrics["model"].drop_duplicates().tolist(),
        "collection_config": {
            "max_schedule_retries": max_schedule_retries,
            "max_session_retries": max_session_retries,
            "max_collection_passes": max_collection_passes,
            "retry_sleep_seconds": retry_sleep_seconds,
        },
    }

    with (results_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    run_history_row = pd.DataFrame([summary])
    run_history_path = results_dir / "run_history.csv"
    run_history_row.to_csv(
        run_history_path,
        mode="a",
        index=False,
        header=not run_history_path.exists(),
    )

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="F1 strategija guma: data pipeline + evaluacija")
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--raw-output", type=Path, default=Path("data/raw/f1_laps_raw.csv"))
    parser.add_argument(
        "--processed-output",
        type=Path,
        default=Path("data/processed/f1_laps_processed.csv"),
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--max-schedule-retries", type=int, default=3)
    parser.add_argument("--max-session-retries", type=int, default=3)
    parser.add_argument("--max-collection-passes", type=int, default=3)
    parser.add_argument("--retry-sleep-seconds", type=float, default=1.5)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    args = parse_args()

    summary = run_pipeline(
        start_year=args.start_year,
        end_year=args.end_year,
        raw_output_path=args.raw_output,
        processed_output_path=args.processed_output,
        results_dir=args.results_dir,
        cache_dir=args.cache_dir,
        random_state=args.random_state,
        max_schedule_retries=args.max_schedule_retries,
        max_session_retries=args.max_session_retries,
        max_collection_passes=args.max_collection_passes,
        retry_sleep_seconds=args.retry_sleep_seconds,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
