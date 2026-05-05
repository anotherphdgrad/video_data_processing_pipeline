#!/usr/bin/env bash
#SBATCH --job-name=rgb_depth_preprocess
#SBATCH --output=logs/rgb_depth_preprocess_%j.out
#SBATCH --error=logs/rgb_depth_preprocess_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Optional SLURM wrapper. Submit this only from an environment where the needed
# conda env is already active. Edit partition/account/GPU directives here if
# your cluster requires them, then submit with:
#   sbatch scripts/hpc_rgb_depth_preprocessing.sbatch

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
if [[ ! -d "${PROJECT_ROOT}" ]]; then
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
mkdir -p "${PROJECT_ROOT}/logs"

export PROJECT_ROOT
export STAGE="${STAGE:-all}"
export DEPTH_ROOT="${DEPTH_ROOT:-/scratch/hsharm62/OUD_Stress_depth/depth_hdf5}"
export RGB_ROOT="${RGB_ROOT:-/scratch/hsharm62/OUD_Stress_RGB/rgb_hdf5}"
export ZARR_OUTPUT_ROOT="${ZARR_OUTPUT_ROOT:-/scratch/hsharm62/OUD_Stress_preprocessed/rgb_depth_zarr_5hz_raw}"

bash "${PROJECT_ROOT}/scripts/hpc_rgb_depth_preprocessing.sh"
