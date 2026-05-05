#!/usr/bin/env python3
"""Generate IMU-to-depth/RGB session mapping manifests and missing reports."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd


TASK_COLUMNS = [
    "start_arithmetic",
    "end_arithmetic",
    "start_bad",
    "end_bad",
    "start_baseline",
    "end_baseline",
    "start_count",
    "end_count",
    "start_good",
    "end_good",
    "start_jelly",
    "end_jelly",
    "start_nature_video",
    "end_nature_video",
    "start_song",
    "end_song",
    "start_speech",
    "end_speech",
    "start_stress",
    "end_stress",
    "start_stroop",
    "end_stroop",
]

FILENAME_TS_RE = re.compile(r"(\d{8}_\d{6})")
SKIP_IMU_STEMS = {"8876"}
MANUAL_PARTICIPANT_ID_OVERRIDES = {
    ("Control", "xianfei"): "xianfei",
}
MANUAL_MANIFEST_GROUP_OVERRIDES = {
    ("Control", "xianfei"): "OUD",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map local IMU CSV files to depth/RGB session files using approved participant mappings "
            "and a manifest, while rebuilding full modality paths from runtime directory roots."
        )
    )
    parser.add_argument("--imu-root", default="assets/IMU_data")
    parser.add_argument("--manifest-csv", default="assets/manifest_mapping_clean_updated_sol.csv")
    parser.add_argument("--candidate-csv", default="assets/imu_participant_mapping_candidates.csv")
    parser.add_argument("--depth-root", required=True)
    parser.add_argument("--rgb-root", required=True)
    parser.add_argument("--output-root", default="assets/imu_video_mapping")
    parser.add_argument(
        "--session-match-threshold-seconds",
        type=float,
        default=6 * 3600,
        help=(
            "Maximum allowed absolute difference between the IMU file's first timestamp and the "
            "video session timestamp parsed from the H5 filename."
        ),
    )
    return parser.parse_args()


def safe_sort_key(series: pd.Series) -> pd.Series:
    return series.astype(str)


def collect_imu_files(imu_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for csv_path in sorted(imu_root.rglob("left_*_acc.csv")):
        group = csv_path.parent.name
        imu_stem = csv_path.stem[len("left_") : -len("_acc")]
        if imu_stem in SKIP_IMU_STEMS:
            continue
        time_df = pd.read_csv(csv_path, usecols=["time"])
        time_values = pd.to_numeric(time_df["time"], errors="coerce").dropna()
        rows.append(
            {
                "group": group,
                "imu_stem": imu_stem,
                "imu_csv_path": str(csv_path.resolve()),
                "imu_csv_name": csv_path.name,
                "imu_first_time": float(time_values.iloc[0]) if not time_values.empty else pd.NA,
                "imu_last_time": float(time_values.iloc[-1]) if not time_values.empty else pd.NA,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "group",
            "imu_stem",
            "imu_csv_path",
            "imu_csv_name",
            "imu_first_time",
            "imu_last_time",
        ],
    )


def normalize_candidates(candidate_df: pd.DataFrame) -> pd.DataFrame:
    df = candidate_df.copy()
    df["approved"] = df["approved"].astype(str).str.lower().isin({"true", "1", "yes"})
    df["participant_id"] = df["participant_id"].astype(str)
    df["proposed_imu_stem"] = df["proposed_imu_stem"].astype(str)
    df["group"] = df["group"].astype(str)
    return df


def build_candidate_suggestions(candidate_df: pd.DataFrame, group: str, imu_stem: str) -> str:
    subset = candidate_df[candidate_df["group"] == group].copy()
    if subset.empty:
        return ""

    exact = subset[subset["proposed_imu_stem"] == imu_stem]
    if not exact.empty:
        cols = ["participant_id", "match_type", "confidence", "approved"]
        return exact[cols].to_json(orient="records")

    numeric = subset[subset["participant_id"].astype(str) == imu_stem]
    if not numeric.empty:
        cols = ["participant_id", "proposed_imu_stem", "match_type", "confidence", "approved"]
        return numeric[cols].to_json(orient="records")

    starts = subset[subset["proposed_imu_stem"].str.startswith(imu_stem[:3], na=False)]
    if not starts.empty:
        cols = ["participant_id", "proposed_imu_stem", "match_type", "confidence", "approved"]
        return starts.head(10)[cols].to_json(orient="records")

    return subset.head(10)[["participant_id", "proposed_imu_stem", "match_type", "confidence", "approved"]].to_json(
        orient="records"
    )


def extract_timestamp_from_filename(filename: str) -> float | None:
    match = FILENAME_TS_RE.search(filename)
    if not match:
        return None
    dt = pd.to_datetime(match.group(1), format="%Y%m%d_%H%M%S", errors="coerce")
    if pd.isna(dt):
        return None
    return float(dt.timestamp())


def derive_v2_fallback_rows(manifest_df: pd.DataFrame, imu_stem: str, group: str) -> pd.DataFrame:
    if not imu_stem.endswith("v2"):
        return manifest_df.iloc[0:0].copy()
    base_stem = imu_stem[:-2]
    subset = manifest_df[
        (manifest_df["group"] == group)
        & (
            (manifest_df["participant_id"] == base_stem)
            | (manifest_df["participant_id_norm"] == base_stem)
        )
    ].copy()
    if subset.empty:
        return subset
    filename_lower = subset["depth_filename"].astype(str).str.lower()
    session2_mask = (
        filename_lower.str.contains("_p2_")
        | filename_lower.str.contains("sess2")
        | filename_lower.str.contains("session2")
        | (pd.to_numeric(subset["is_session2"], errors="coerce").fillna(0) > 0)
        | (pd.to_numeric(subset["has_session2"], errors="coerce").fillna(0) > 0)
    )
    preferred = subset[session2_mask].copy()
    return preferred if not preferred.empty else subset


def select_nearest_session_rows_by_view(
    manifest_hits: pd.DataFrame,
    session_match_threshold_seconds: float,
) -> pd.DataFrame:
    """Keep the nearest video session candidate for each view.

    A participant can legitimately have frontal and side recordings. What we do
    not want is multiple frontal or multiple side rows for the same IMU session.
    """
    if manifest_hits.empty:
        return manifest_hits

    selected_rows = []
    total_candidates = int(len(manifest_hits))
    rank_df = manifest_hits.copy()
    rank_df["view_type"] = rank_df["view_type"].fillna("").astype(str)
    rank_df["_rank_diff"] = pd.to_numeric(rank_df["session_time_diff_seconds"], errors="coerce")
    rank_df["_rank_diff"] = rank_df["_rank_diff"].fillna(float("inf"))
    rank_df["_rank_filename"] = rank_df["depth_filename"].astype(str)

    rank_df = rank_df.sort_values(
        by=["view_type", "_rank_diff", "_rank_filename"],
        kind="mergesort",
    ).copy()
    rank_df["session_candidate_count_total"] = total_candidates
    rank_df["session_candidate_count_for_view"] = rank_df.groupby("view_type")["view_type"].transform("size")
    rank_df["session_candidate_rank_for_view"] = rank_df.groupby("view_type").cumcount() + 1
    rank_df["session_dropped_candidate_count_for_view"] = rank_df["session_candidate_count_for_view"] - 1
    rank_df["session_selected_by"] = "nearest_video_session_per_view"
    rank_df["session_within_threshold"] = (
        pd.to_numeric(rank_df["session_time_diff_seconds"], errors="coerce")
        <= float(session_match_threshold_seconds)
    )

    selected_rows.append(rank_df[rank_df["session_candidate_rank_for_view"] == 1].copy())
    selected = pd.concat(selected_rows, ignore_index=False)
    return selected.drop(columns=["_rank_diff", "_rank_filename"])


def main() -> None:
    args = parse_args()
    imu_root = Path(args.imu_root).resolve()
    manifest_csv = Path(args.manifest_csv).resolve()
    candidate_csv = Path(args.candidate_csv).resolve()
    depth_root = Path(args.depth_root).resolve()
    rgb_root = Path(args.rgb_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    imu_df = collect_imu_files(imu_root)
    if imu_df.empty:
        raise FileNotFoundError(
            "No IMU CSV files were found. Expected files matching "
            f"'left_*_acc.csv' under IMU root: {imu_root}. "
            "On HPC, either copy assets/IMU_data into the repo checkout or run with "
            "IMU_ROOT=/path/to/IMU_data bash scripts/hpc_rgb_depth_preprocessing.sh"
        )
    manifest_df = pd.read_csv(manifest_csv)
    candidate_df = normalize_candidates(pd.read_csv(candidate_csv))

    approved_df = candidate_df[candidate_df["approved"]].copy()
    approved_df = approved_df.rename(columns={"proposed_imu_stem": "imu_stem"})

    manifest_df["participant_id"] = manifest_df["participant_id"].astype(str)
    manifest_df["participant_id_norm"] = manifest_df["participant_id_norm"].astype(str)
    manifest_df["group"] = manifest_df["group"].astype(str)
    manifest_df["global_time_id"] = manifest_df["global_time_id"].astype(str)
    manifest_df["depth_filename"] = manifest_df["depth_path"].astype(str).map(lambda value: Path(value).name)
    manifest_df["video_session_ts"] = manifest_df["depth_filename"].map(extract_timestamp_from_filename)

    merged = imu_df.merge(
        approved_df[
            ["group", "imu_stem", "participant_id", "match_type", "confidence", "notes"]
        ].drop_duplicates(),
        on=["group", "imu_stem"],
        how="left",
    )

    no_mapping_mask = merged["participant_id"].isna()
    if no_mapping_mask.any():
        override_mask = merged.apply(
            lambda row: (row["group"], row["imu_stem"]) in MANUAL_PARTICIPANT_ID_OVERRIDES and pd.isna(row["participant_id"]),
            axis=1,
        )
        if override_mask.any():
            merged.loc[override_mask, "participant_id"] = merged.loc[override_mask].apply(
                lambda row: MANUAL_PARTICIPANT_ID_OVERRIDES[(row["group"], row["imu_stem"])],
                axis=1,
            )
            merged.loc[override_mask, "match_type"] = "manual_participant_override"
            merged.loc[override_mask, "confidence"] = "user_confirmed"
            merged.loc[override_mask, "notes"] = "Participant mapping supplied manually by user."

    no_mapping_mask = merged["participant_id"].isna()
    missing_mapping_rows = merged.loc[no_mapping_mask, ["group", "imu_stem", "imu_csv_path", "imu_first_time"]].copy()
    if not missing_mapping_rows.empty:
        missing_mapping_rows["issue"] = "no_approved_participant_mapping"
        missing_mapping_rows["candidate_suggestions"] = [
            build_candidate_suggestions(candidate_df, row.group, row.imu_stem) for row in missing_mapping_rows.itertuples()
        ]

    fallback_rows = []
    for row in merged.loc[no_mapping_mask].itertuples():
        fallback_hits = derive_v2_fallback_rows(manifest_df, row.imu_stem, row.group)
        if fallback_hits.empty:
            continue
        fallback_rows.append(
            {
                "group": row.group,
                "imu_stem": row.imu_stem,
                "imu_csv_path": row.imu_csv_path,
                "imu_csv_name": row.imu_csv_name,
                "imu_first_time": row.imu_first_time,
                "imu_last_time": row.imu_last_time,
                "participant_id": str(fallback_hits["participant_id"].iloc[0]),
                "match_type": "auto_v2_base_manifest_fallback",
                "confidence": "medium",
                "notes": f"Derived from IMU stem {row.imu_stem} -> manifest participant {fallback_hits['participant_id'].iloc[0]} using session2/p2 video rows.",
            }
        )

    if fallback_rows:
        fallback_df = pd.DataFrame(fallback_rows)
        merged = pd.concat([merged.loc[~no_mapping_mask].copy(), fallback_df], ignore_index=True)
        resolved_stems = set(fallback_df["imu_stem"].astype(str))
        missing_mapping_rows = missing_mapping_rows[~missing_mapping_rows["imu_stem"].astype(str).isin(resolved_stems)].copy()

    mapped = merged.loc[merged["participant_id"].notna()].copy()
    session_rows: list[dict] = []
    missing_manifest_rows: list[dict] = []

    for row in mapped.itertuples():
        manifest_group = MANUAL_MANIFEST_GROUP_OVERRIDES.get((row.group, row.imu_stem), row.group)
        manifest_hits = manifest_df[
            (manifest_df["group"] == manifest_group)
            & (
                (manifest_df["participant_id"] == str(row.participant_id))
                | (manifest_df["participant_id_norm"] == str(row.imu_stem))
            )
        ].copy()

        if manifest_hits.empty:
            suggestions = manifest_df[manifest_df["group"] == row.group].copy()
            suggestion_payload = suggestions[
                ["participant_id", "participant_id_norm", "global_time_id", "view_type"]
            ].drop_duplicates().head(15).to_json(orient="records")
            missing_manifest_rows.append(
                {
                    "group": row.group,
                    "imu_stem": row.imu_stem,
                    "imu_csv_path": row.imu_csv_path,
                    "imu_first_time": row.imu_first_time,
                    "participant_id": row.participant_id,
                    "issue": "approved_mapping_but_no_manifest_rows",
                    "candidate_suggestions": suggestion_payload,
                }
            )
            continue

        if pd.notna(row.imu_first_time):
            imu_first_time = float(row.imu_first_time)
            manifest_hits["imu_first_time"] = imu_first_time
            manifest_hits["session_time_diff_seconds"] = (manifest_hits["video_session_ts"] - imu_first_time).abs()
        else:
            manifest_hits["imu_first_time"] = pd.NA
            manifest_hits["session_time_diff_seconds"] = pd.NA
        manifest_hits = select_nearest_session_rows_by_view(
            manifest_hits,
            session_match_threshold_seconds=float(args.session_match_threshold_seconds),
        )

        for hit in manifest_hits.itertuples():
            depth_filename = hit.depth_filename
            rgb_filename = depth_filename
            depth_full_path = depth_root / depth_filename
            rgb_full_path = rgb_root / rgb_filename

            payload = {
                "group": row.group,
                "manifest_group": manifest_group,
                "imu_stem": row.imu_stem,
                "imu_csv_path": row.imu_csv_path,
                "participant_id": hit.participant_id,
                "participant_id_norm": hit.participant_id_norm,
                "global_time_id": hit.global_time_id,
                "view_type": getattr(hit, "view_type", ""),
                "depth_filename": depth_filename,
                "rgb_filename": rgb_filename,
                "depth_path": str(depth_full_path),
                "rgb_path": str(rgb_full_path),
                "depth_exists": depth_full_path.exists(),
                "rgb_exists": rgb_full_path.exists(),
                "audio_start_ts": hit.audio_start_ts,
                "depth_start_ts": hit.depth_start_ts,
                "video_session_ts": getattr(hit, "video_session_ts", pd.NA),
                "imu_first_time": getattr(hit, "imu_first_time", pd.NA),
                "session_time_diff_seconds": getattr(hit, "session_time_diff_seconds", pd.NA),
                "session_candidate_count_total": getattr(hit, "session_candidate_count_total", pd.NA),
                "session_candidate_count_for_view": getattr(hit, "session_candidate_count_for_view", pd.NA),
                "session_candidate_rank_for_view": getattr(hit, "session_candidate_rank_for_view", pd.NA),
                "session_dropped_candidate_count_for_view": getattr(
                    hit, "session_dropped_candidate_count_for_view", pd.NA
                ),
                "session_selected_by": getattr(hit, "session_selected_by", pd.NA),
                "session_within_threshold": getattr(hit, "session_within_threshold", pd.NA),
                "offset_depth": hit.offset_depth,
                "is_session2": hit.is_session2,
                "has_session2": hit.has_session2,
                "mapping_match_type": row.match_type,
                "mapping_confidence": row.confidence,
                "mapping_notes": row.notes,
            }
            for column in TASK_COLUMNS:
                payload[column] = getattr(hit, column)
            session_rows.append(payload)

    session_df = pd.DataFrame(session_rows)
    if not session_df.empty:
        session_df = session_df.sort_values(
            by=["group", "imu_stem", "global_time_id", "view_type", "depth_filename"],
            key=safe_sort_key,
        ).reset_index(drop=True)

    missing_reports = []
    if not missing_mapping_rows.empty:
        missing_reports.append(missing_mapping_rows)
    if missing_manifest_rows:
        missing_reports.append(pd.DataFrame(missing_manifest_rows))

    missing_df = pd.concat(missing_reports, ignore_index=True) if missing_reports else pd.DataFrame(
        columns=["group", "imu_stem", "imu_csv_path", "participant_id", "issue", "candidate_suggestions"]
    )
    if not session_df.empty:
        existence_issues = session_df[(~session_df["depth_exists"]) | (~session_df["rgb_exists"])].copy()
        if not existence_issues.empty:
            existence_issues["issue"] = existence_issues.apply(
                lambda row: "missing_depth_and_rgb"
                if (not row["depth_exists"] and not row["rgb_exists"])
                else ("missing_depth_file" if not row["depth_exists"] else "missing_rgb_file"),
                axis=1,
            )
            existence_cols = [
                "group",
                "imu_stem",
                "imu_csv_path",
                "participant_id",
                "global_time_id",
                "view_type",
                "depth_filename",
                "rgb_filename",
                "depth_path",
                "rgb_path",
                "issue",
            ]
            missing_df = pd.concat([missing_df, existence_issues[existence_cols]], ignore_index=True)

    session_manifest_path = output_root / "imu_to_video_session_manifest.csv"
    missing_report_path = output_root / "imu_to_video_missing_report.csv"
    summary_path = output_root / "imu_to_video_mapping_summary.csv"

    session_df.to_csv(session_manifest_path, index=False)
    missing_df.to_csv(missing_report_path, index=False)

    summary_rows = [
        {"metric": "num_imu_files", "value": int(len(imu_df))},
        {"metric": "num_approved_mapped_imu_files", "value": int(len(mapped))},
        {"metric": "num_missing_approved_mapping", "value": int(len(missing_mapping_rows))},
        {"metric": "num_session_manifest_rows", "value": int(len(session_df))},
        {
            "metric": "num_unique_imu_with_session_rows",
            "value": int(session_df[["group", "imu_stem"]].drop_duplicates().shape[0]) if not session_df.empty else 0,
        },
        {"metric": "num_missing_report_rows", "value": int(len(missing_df))},
        {"metric": "num_depth_missing_rows", "value": int((~session_df["depth_exists"]).sum()) if not session_df.empty else 0},
        {"metric": "num_rgb_missing_rows", "value": int((~session_df["rgb_exists"]).sum()) if not session_df.empty else 0},
        {
            "metric": "num_session_duplicate_candidates_dropped",
            "value": int(session_df["session_dropped_candidate_count_for_view"].sum()) if not session_df.empty else 0,
        },
        {
            "metric": "num_session_rows_outside_threshold",
            "value": int((~session_df["session_within_threshold"].astype(bool)).sum()) if not session_df.empty else 0,
        },
    ]
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print(f"Wrote session manifest to {session_manifest_path}")
    print(f"Wrote missing report to {missing_report_path}")
    print(f"Wrote summary to {summary_path}")

    if not missing_df.empty:
        print("\nMissing or incomplete mappings were found. Please inspect the missing report before generating frame indices.")


if __name__ == "__main__":
    main()
