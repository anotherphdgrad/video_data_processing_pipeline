#!/usr/bin/env bash
# HPC-safe launcher for RGB/depth task-aligned preprocessing.
#
# This script rebuilds all RGB/depth paths from H5 filenames plus the HPC
# /scratch roots, then generates the task/frame manifest and optional Zarr data.
# Launch this from a conda environment that already has the required Python
# dependencies available.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
if [[ ! -d "${PROJECT_ROOT}" ]]; then
  echo "WARNING: PROJECT_ROOT=${PROJECT_ROOT} does not exist; using script-inferred root ${DEFAULT_PROJECT_ROOT}" >&2
  PROJECT_ROOT="${DEFAULT_PROJECT_ROOT}"
fi

DEPTH_ROOT="${DEPTH_ROOT:-/scratch/hsharm62/OUD_Stress_depth/depth_hdf5}"
RGB_ROOT="${RGB_ROOT:-/scratch/hsharm62/OUD_Stress_RGB/rgb_hdf5}"

STAGE="${STAGE:-all}"
SAMPLE_RATE_HZ="${SAMPLE_RATE_HZ:-5.0}"
RAW_SAMPLE_RATE_HZ="${RAW_SAMPLE_RATE_HZ:-32.0}"
NEAREST_TOLERANCE_SECONDS="${NEAREST_TOLERANCE_SECONDS:-0.25}"
COVERAGE_TOLERANCE_SECONDS="${COVERAGE_TOLERANCE_SECONDS:-0.25}"
FRAMES_PER_CHUNK="${FRAMES_PER_CHUNK:-150}"
COMPRESSOR="${COMPRESSOR:-zstd}"
COMPRESSION_LEVEL="${COMPRESSION_LEVEL:-5}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_BY="${SHARD_BY:-subject}"
LOCAL_ZARR_WORKERS="${LOCAL_ZARR_WORKERS:-1}"
CPU_THREADS_PER_WORKER="${CPU_THREADS_PER_WORKER:-1}"

PARTICIPANTS="${PARTICIPANTS:-}"
TASKS="${TASKS:-}"
VIEWS="${VIEWS:-}"
MAX_SESSION_ROWS="${MAX_SESSION_ROWS:-}"
MAX_ZARR_ROWS="${MAX_ZARR_ROWS:-}"
OVERWRITE_ZARR="${OVERWRITE_ZARR:-0}"
REQUIRE_COMPLETE="${REQUIRE_COMPLETE:-0}"

IMU_ROOT="${IMU_ROOT:-${PROJECT_ROOT}/assets/IMU_data}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-${PROJECT_ROOT}/assets/manifest_mapping_clean_updated_sol.csv}"
CANDIDATE_CSV="${CANDIDATE_CSV:-${PROJECT_ROOT}/assets/imu_participant_mapping_candidates.csv}"

MAPPING_OUTPUT_ROOT="${MAPPING_OUTPUT_ROOT:-${PROJECT_ROOT}/assets/imu_video_mapping_hpc}"
SESSION_MANIFEST="${SESSION_MANIFEST:-${MAPPING_OUTPUT_ROOT}/imu_to_video_session_manifest.csv}"
TASK_FRAME_MANIFEST="${TASK_FRAME_MANIFEST:-${MAPPING_OUTPUT_ROOT}/rgb_depth_task_frame_manifest.csv}"
ZARR_OUTPUT_ROOT="${ZARR_OUTPUT_ROOT:-/scratch/hsharm62/OUD_Stress_preprocessed/rgb_depth_zarr_5hz_raw}"

cd "${PROJECT_ROOT}"

mkdir -p "${MAPPING_OUTPUT_ROOT}"
if [[ "${STAGE}" == "all" || "${STAGE}" == "zarr" ]]; then
  mkdir -p "${ZARR_OUTPUT_ROOT}"
fi

echo "python=$(command -v python)"
python --version
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "IMU_ROOT=${IMU_ROOT}"
echo "SOURCE_MANIFEST=${SOURCE_MANIFEST}"
echo "CANDIDATE_CSV=${CANDIDATE_CSV}"
echo "DEPTH_ROOT=${DEPTH_ROOT}"
echo "RGB_ROOT=${RGB_ROOT}"
echo "MAPPING_OUTPUT_ROOT=${MAPPING_OUTPUT_ROOT}"
echo "ZARR_OUTPUT_ROOT=${ZARR_OUTPUT_ROOT}"
echo "STAGE=${STAGE}"
echo "NUM_SHARDS=${NUM_SHARDS}"
echo "SHARD_INDEX=${SHARD_INDEX}"
echo "SHARD_BY=${SHARD_BY}"
echo "LOCAL_ZARR_WORKERS=${LOCAL_ZARR_WORKERS}"
echo "CPU_THREADS_PER_WORKER=${CPU_THREADS_PER_WORKER}"

common_participant_args=()
if [[ -n "${PARTICIPANTS}" ]]; then
  # shellcheck disable=SC2206
  common_participant_args=(--participants ${PARTICIPANTS})
fi

task_args=()
if [[ -n "${TASKS}" ]]; then
  # shellcheck disable=SC2206
  task_args=(--tasks ${TASKS})
fi

view_args=()
if [[ -n "${VIEWS}" ]]; then
  # shellcheck disable=SC2206
  view_args=(--views ${VIEWS})
fi

max_session_args=()
if [[ -n "${MAX_SESSION_ROWS}" ]]; then
  max_session_args=(--max-session-rows "${MAX_SESSION_ROWS}")
fi

max_zarr_args=()
if [[ -n "${MAX_ZARR_ROWS}" ]]; then
  max_zarr_args=(--max-rows "${MAX_ZARR_ROWS}")
fi

overwrite_args=()
if [[ "${OVERWRITE_ZARR}" == "1" ]]; then
  overwrite_args=(--overwrite)
fi

require_complete_args=()
if [[ "${REQUIRE_COMPLETE}" == "1" ]]; then
  require_complete_args=(--require-complete)
fi

run_zarr_shard() {
  local shard_index="$1"
  local num_shards="$2"
  echo
  echo "[$(date)] Writing Zarr shard ${shard_index}/${num_shards}..."
  OMP_NUM_THREADS="${CPU_THREADS_PER_WORKER}" \
  OPENBLAS_NUM_THREADS="${CPU_THREADS_PER_WORKER}" \
  MKL_NUM_THREADS="${CPU_THREADS_PER_WORKER}" \
  NUMEXPR_NUM_THREADS="${CPU_THREADS_PER_WORKER}" \
  python scripts/preprocess_rgb_depth_task_zarr.py \
    --task-frame-manifest "${TASK_FRAME_MANIFEST}" \
    --output-root "${ZARR_OUTPUT_ROOT}" \
    --sample-rate-hz "${SAMPLE_RATE_HZ}" \
    --nearest-tolerance-seconds "${NEAREST_TOLERANCE_SECONDS}" \
    --frames-per-chunk "${FRAMES_PER_CHUNK}" \
    --compressor "${COMPRESSOR}" \
    --compression-level "${COMPRESSION_LEVEL}" \
    --num-shards "${num_shards}" \
    --shard-index "${shard_index}" \
    --shard-by "${SHARD_BY}" \
    "${common_participant_args[@]}" \
    "${task_args[@]}" \
    "${view_args[@]}" \
    "${max_zarr_args[@]}" \
    "${overwrite_args[@]}" \
    "${require_complete_args[@]}"
}

if [[ "${STAGE}" == "all" || "${STAGE}" == "session" || "${STAGE}" == "manifests" ]]; then
  echo
  echo "[$(date)] Generating HPC session manifest..."
  python scripts/generate_imu_video_mapping_manifest.py \
    --imu-root "${IMU_ROOT}" \
    --manifest-csv "${SOURCE_MANIFEST}" \
    --candidate-csv "${CANDIDATE_CSV}" \
    --depth-root "${DEPTH_ROOT}" \
    --rgb-root "${RGB_ROOT}" \
    --output-root "${MAPPING_OUTPUT_ROOT}"
fi

if [[ "${STAGE}" == "all" || "${STAGE}" == "task_frames" || "${STAGE}" == "manifests" ]]; then
  echo
  echo "[$(date)] Generating RGB/depth task-frame manifest..."
  python scripts/generate_rgb_depth_task_frame_manifest.py \
    --session-manifest "${SESSION_MANIFEST}" \
    --output-csv "${TASK_FRAME_MANIFEST}" \
    --raw-sample-rate-hz "${RAW_SAMPLE_RATE_HZ}" \
    --coverage-tolerance-seconds "${COVERAGE_TOLERANCE_SECONDS}" \
    "${common_participant_args[@]}" \
    "${task_args[@]}" \
    "${max_session_args[@]}"
fi

if [[ "${STAGE}" == "all" || "${STAGE}" == "zarr" ]]; then
  echo
  echo "[$(date)] Writing 5 Hz raw RGB/depth Zarr stores..."
  if [[ "${LOCAL_ZARR_WORKERS}" -gt 1 ]]; then
    if [[ "${NUM_SHARDS}" != "1" || "${SHARD_INDEX}" != "0" ]]; then
      echo "ERROR: Use either LOCAL_ZARR_WORKERS or explicit NUM_SHARDS/SHARD_INDEX, not both." >&2
      exit 2
    fi
    if [[ -n "${MAX_ZARR_ROWS}" ]]; then
      echo "ERROR: MAX_ZARR_ROWS is not supported with LOCAL_ZARR_WORKERS because each worker would apply it independently." >&2
      exit 2
    fi
    pids=()
    for shard in $(seq 0 $((LOCAL_ZARR_WORKERS - 1))); do
      run_zarr_shard "${shard}" "${LOCAL_ZARR_WORKERS}" &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do
      wait "${pid}"
    done
  else
    run_zarr_shard "${SHARD_INDEX}" "${NUM_SHARDS}"
  fi
fi

echo
echo "[$(date)] Done."
echo "Session manifest: ${SESSION_MANIFEST}"
echo "Task/frame manifest: ${TASK_FRAME_MANIFEST}"
echo "Zarr output root: ${ZARR_OUTPUT_ROOT}"
