#!/usr/bin/env python3
"""Inspect IMU timestamp quality and export participant-level CSV reports."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect IMU timestamp columns and export summary CSV reports."
    )
    parser.add_argument(
        "--imu-root",
        default="assets/IMU_data",
        help="Root directory containing group subfolders with left_*_acc.csv files.",
    )
    parser.add_argument(
        "--output-root",
        default="assets/imu_time_inspection",
        help="Directory where summary and participant-level CSV files will be written.",
    )
    return parser.parse_args()


def safe_slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def summarize_segments(df: pd.DataFrame) -> pd.DataFrame:
    labels = df["label"].fillna("None").astype(str)
    times = pd.to_numeric(df["time"], errors="coerce")
    starts_new = labels.ne(labels.shift(1))
    segment_id = starts_new.cumsum()

    segments = (
        df.assign(label_clean=labels, time_num=times, segment_id=segment_id)
        .groupby("segment_id", sort=True)
        .agg(
            label=("label_clean", "first"),
            start_row=("segment_id", lambda s: int(s.index[0])),
            end_row=("segment_id", lambda s: int(s.index[-1])),
            num_rows=("segment_id", "size"),
            start_time=("time_num", "first"),
            end_time=("time_num", "last"),
            unique_timestamps=("time_num", lambda s: int(pd.Series(s).nunique(dropna=True))),
        )
        .reset_index(drop=True)
    )
    segments["duration_seconds_wall"] = segments["end_time"] - segments["start_time"]
    return segments


def summarize_timestamp_counts(df: pd.DataFrame) -> pd.DataFrame:
    labels = df["label"].fillna("None").astype(str)
    times = pd.to_numeric(df["time"], errors="coerce")
    grouped = (
        pd.DataFrame({"time": times, "label": labels})
        .groupby("time", dropna=False, sort=True)
        .agg(
            num_rows=("time", "size"),
            first_label=("label", "first"),
            last_label=("label", "last"),
            unique_labels=("label", lambda s: int(pd.Series(s).nunique(dropna=True))),
        )
        .reset_index()
    )
    grouped["delta_from_prev"] = grouped["time"].diff()
    grouped["is_integer_time"] = np.isclose(grouped["time"] % 1.0, 0.0, atol=1e-9)
    return grouped


def build_file_summary(group: str, participant_id: str, path: Path, df: pd.DataFrame) -> dict:
    times = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=np.float64)
    valid_mask = np.isfinite(times)
    valid_times = times[valid_mask]
    unique_times = np.unique(valid_times)
    diffs = np.diff(valid_times) if len(valid_times) > 1 else np.empty(0, dtype=np.float64)
    unique_diffs = np.diff(unique_times) if len(unique_times) > 1 else np.empty(0, dtype=np.float64)

    zero_diffs = int(np.sum(np.isclose(diffs, 0.0, atol=1e-12)))
    positive_diffs = diffs[diffs > 0.0]
    negative_diffs = int(np.sum(diffs < 0.0))

    labels = df["label"].fillna("None").astype(str)
    label_counts = labels.value_counts(dropna=False).to_dict()
    non_none_labels = sorted(label for label in label_counts if label not in {"None", "nan", ""})

    samples_per_unique = len(valid_times) / len(unique_times) if len(unique_times) else np.nan
    duration = float(unique_times[-1] - unique_times[0]) if len(unique_times) > 1 else 0.0
    estimated_hz = len(valid_times) / duration if duration > 0.0 else np.nan
    all_integer = bool(np.all(np.isclose(valid_times % 1.0, 0.0, atol=1e-9))) if len(valid_times) else False

    return {
        "group": group,
        "participant_id": participant_id,
        "file_path": str(path),
        "num_rows": int(len(df)),
        "num_valid_time_rows": int(len(valid_times)),
        "num_unique_timestamps": int(len(unique_times)),
        "start_time": float(unique_times[0]) if len(unique_times) else np.nan,
        "end_time": float(unique_times[-1]) if len(unique_times) else np.nan,
        "duration_seconds": duration,
        "estimated_sample_rate_hz": float(estimated_hz) if np.isfinite(estimated_hz) else np.nan,
        "avg_rows_per_unique_timestamp": float(samples_per_unique) if np.isfinite(samples_per_unique) else np.nan,
        "all_timestamps_integer": all_integer,
        "num_zero_diffs": zero_diffs,
        "num_negative_diffs": negative_diffs,
        "min_positive_row_diff": float(positive_diffs.min()) if len(positive_diffs) else np.nan,
        "median_positive_row_diff": float(np.median(positive_diffs)) if len(positive_diffs) else np.nan,
        "min_unique_timestamp_diff": float(unique_diffs.min()) if len(unique_diffs) else np.nan,
        "median_unique_timestamp_diff": float(np.median(unique_diffs)) if len(unique_diffs) else np.nan,
        "num_distinct_labels": int(pd.Series(labels).nunique(dropna=True)),
        "observed_labels": ",".join(non_none_labels),
    }


def main() -> None:
    args = parse_args()
    imu_root = Path(args.imu_root).resolve()
    output_root = Path(args.output_root).resolve()
    per_participant_root = output_root / "per_participant"
    per_participant_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    participant_index_rows: list[dict] = []

    for csv_path in sorted(imu_root.rglob("left_*_acc.csv")):
        group = csv_path.parent.name
        participant_id = csv_path.stem[len("left_") : -len("_acc")]
        df = pd.read_csv(csv_path, usecols=["time", "label"])

        timestamp_counts = summarize_timestamp_counts(df)
        task_segments = summarize_segments(df)
        summary_rows.append(build_file_summary(group, participant_id, csv_path, df))

        participant_dir = per_participant_root / safe_slug(group)
        participant_dir.mkdir(parents=True, exist_ok=True)

        counts_path = participant_dir / f"{safe_slug(participant_id)}__timestamp_counts.csv"
        segments_path = participant_dir / f"{safe_slug(participant_id)}__task_segments.csv"
        timestamp_counts.to_csv(counts_path, index=False)
        task_segments.to_csv(segments_path, index=False)

        participant_index_rows.append(
            {
                "group": group,
                "participant_id": participant_id,
                "source_csv": str(csv_path),
                "timestamp_counts_csv": str(counts_path),
                "task_segments_csv": str(segments_path),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["group", "participant_id"]).reset_index(drop=True)
    summary_df.to_csv(output_root / "imu_timestamp_file_summary.csv", index=False)

    pd.DataFrame(participant_index_rows).sort_values(["group", "participant_id"]).to_csv(
        output_root / "imu_timestamp_participant_index.csv",
        index=False,
    )

    print(f"Wrote {len(summary_df)} file summaries to {output_root / 'imu_timestamp_file_summary.csv'}")
    print(f"Wrote participant index to {output_root / 'imu_timestamp_participant_index.csv'}")
    print(f"Wrote participant-level CSVs under {per_participant_root}")


if __name__ == "__main__":
    main()
