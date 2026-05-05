#!/usr/bin/env python3
"""Validate raw 5 Hz RGB/depth Zarr task stores before derived preprocessing."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_ARRAYS = {
    "rgb",
    "depth",
    "target_timestamps",
    "rgb_source_timestamps",
    "depth_source_timestamps",
    "rgb_frame_indices",
    "depth_frame_indices",
}
REQUIRED_ATTRS = {"base_subject_id", "group", "task_id", "view_type", "stress_label"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate raw RGB/depth task Zarr stores.")
    parser.add_argument(
        "--input-root",
        default="processed_rgb_depth_zarr_5hz_raw",
        help="Root containing group/base_subject_id.zarr stores.",
    )
    parser.add_argument(
        "--output-csv",
        default="assets/derived_feature_visual_checks/raw_zarr_validation_report.csv",
        help="CSV report path.",
    )
    parser.add_argument("--expected-sample-rate-hz", type=float, default=5.0)
    parser.add_argument("--max-stores", type=int, default=None)
    parser.add_argument("--fail-on-incomplete", action="store_true")
    return parser.parse_args()


def require_zarr():
    try:
        import zarr
    except ImportError as exc:
        raise SystemExit("Install zarr<3 in the active environment before running validation.") from exc
    major = int(str(getattr(zarr, "__version__", "0")).split(".", maxsplit=1)[0])
    if major >= 3:
        raise SystemExit(f"Unsupported zarr version {zarr.__version__}; this pipeline expects zarr<3.")
    return zarr


def validate_task(task_group, expected_step: float) -> dict:
    keys = set(task_group.keys())
    attrs = dict(task_group.attrs)
    missing_arrays = sorted(REQUIRED_ARRAYS - keys)
    missing_attrs = sorted(REQUIRED_ATTRS - set(attrs))
    row = {
        "task_record_id": attrs.get("task_record_id", ""),
        "group": attrs.get("group", ""),
        "base_subject_id": attrs.get("base_subject_id", ""),
        "task_id": attrs.get("task_id", ""),
        "view_type": attrs.get("view_type", ""),
        "stress_label": attrs.get("stress_label", ""),
        "status": "complete",
        "missing_arrays": ",".join(missing_arrays),
        "missing_attrs": ",".join(missing_attrs),
        "rgb_shape": "",
        "depth_shape": "",
        "rgb_dtype": "",
        "depth_dtype": "",
        "num_samples": np.nan,
        "timestamp_step_median": np.nan,
        "timestamp_step_error": np.nan,
        "error": "",
    }
    if missing_arrays or missing_attrs:
        row["status"] = "incomplete"
        return row

    try:
        rgb = task_group["rgb"]
        depth = task_group["depth"]
        timestamps = np.asarray(task_group["target_timestamps"][:], dtype=np.float64)
        row.update(
            {
                "rgb_shape": tuple(rgb.shape),
                "depth_shape": tuple(depth.shape),
                "rgb_dtype": str(rgb.dtype),
                "depth_dtype": str(depth.dtype),
                "num_samples": int(rgb.shape[0]),
            }
        )
        if rgb.ndim != 4 or rgb.shape[-1] != 3:
            row["status"] = "bad_shape"
        if depth.ndim != 3:
            row["status"] = "bad_shape"
        if rgb.shape[0] != depth.shape[0] or rgb.shape[0] != len(timestamps):
            row["status"] = "length_mismatch"
        if len(timestamps) > 1:
            median_step = float(np.median(np.diff(timestamps)))
            row["timestamp_step_median"] = median_step
            row["timestamp_step_error"] = abs(median_step - expected_step)
            if abs(median_step - expected_step) > 0.05:
                row["status"] = "bad_timestamp_step"
    except Exception as exc:  # noqa: BLE001 - keep validating the rest of the dataset.
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def main() -> None:
    args = parse_args()
    zarr = require_zarr()
    input_root = Path(args.input_root).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    stores = sorted(input_root.glob("*/*.zarr"))
    if args.max_stores is not None:
        stores = stores[: int(args.max_stores)]
    rows = []
    expected_step = 1.0 / float(args.expected_sample_rate_hz)
    for store in stores:
        try:
            root = zarr.open_group(str(store), mode="r")
            tasks = root["tasks"]
            for task_name in sorted(tasks.keys()):
                row = validate_task(tasks[task_name], expected_step)
                row["zarr_path"] = str(store)
                row["zarr_task_group"] = f"tasks/{task_name}"
                rows.append(row)
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "zarr_path": str(store),
                    "zarr_task_group": "",
                    "task_record_id": "",
                    "status": "store_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    report = pd.DataFrame(rows)
    report.to_csv(output_csv, index=False)
    print(f"Wrote validation report to {output_csv}")
    print(f"Stores checked: {len(stores)}")
    print(f"Task groups checked: {len(report)}")
    if len(report):
        print(report["status"].value_counts(dropna=False).to_string())
    bad = report[~report["status"].eq("complete")] if len(report) else report
    if args.fail_on_incomplete and len(bad):
        raise SystemExit(f"Validation failed: {len(bad)} non-complete task groups")


if __name__ == "__main__":
    main()
