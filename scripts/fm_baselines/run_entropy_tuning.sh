#!/bin/bash

# Entropy-based frame selection + downstream tuning
# This script orchestrates:
# 1. Extract information-dense frames via entropy (once)
# 2. Run downstream Optuna tuning on entropy-selected frames

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
EMBEDDING_ROOT="${EMBEDDING_ROOT:-outputs_rgb_depth_fm/embeddings_zarr2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs_rgb_depth_fm_downstream_search}"
ENTROPY_OUTPUT_ROOT="${ENTROPY_OUTPUT_ROOT:-outputs_rgb_depth_fm/embeddings_zarr2_entropy_selected}"
TOP_K="${TOP_K:-75}"
ENTROPY_METHOD="${ENTROPY_METHOD:-shannon}"
OPTUNA_TRIALS="${OPTUNA_TRIALS:-30}"
DEVICE="${DEVICE:-cuda}"

echo "=========================================="
echo "Entropy-Based Frame Selection + Tuning"
echo "=========================================="
echo "Stage 1: Extract entropy-selected frames"
echo "  Input: $EMBEDDING_ROOT"
echo "  Output: $ENTROPY_OUTPUT_ROOT"
echo "  Top-K: $TOP_K frames (50% reduction from 150)"
echo "  Method: $ENTROPY_METHOD"
echo ""

# Stage 1: Extract entropy-selected frames
python "$SCRIPT_DIR/entropy_frame_selection.py" \
    --embedding-root "$EMBEDDING_ROOT" \
    --output-root "$ENTROPY_OUTPUT_ROOT" \
    --top-k "$TOP_K" \
    --entropy-method "$ENTROPY_METHOD" \
    --standardize

echo ""
echo "=========================================="
echo "Stage 2: Run Optuna tuning on entropy frames"
echo "  Embedding root: $ENTROPY_OUTPUT_ROOT"
echo "  Optuna trials: $OPTUNA_TRIALS"
echo "  Device: $DEVICE"
echo ""

# Stage 2: Run downstream tuning on entropy-selected frames
python "$SCRIPT_DIR/downstream_search/run_optuna_downstream.py" \
    --embedding-root "$ENTROPY_OUTPUT_ROOT" \
    --output-root "$OUTPUT_ROOT/entropy_selected_top${TOP_K}" \
    --optuna-trials "$OPTUNA_TRIALS" \
    --device "$DEVICE"

echo ""
echo "=========================================="
echo "✓ Complete! Results saved to:"
echo "  $OUTPUT_ROOT/entropy_selected_top${TOP_K}"
echo "=========================================="
