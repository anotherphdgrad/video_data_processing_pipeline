#!/usr/bin/env python3
"""
Entropy-based frame selection for FM embeddings.

Selects information-dense frames from each window by computing embedding entropy.
Creates new zarr stores with selected frames, leaving originals untouched.

Usage:
    python entropy_frame_selection.py --embedding-root outputs_rgb_depth_fm/embeddings_zarr2 \
                                       --output-root outputs_rgb_depth_fm/embeddings_zarr2_entropy_selected \
                                       --top-k 75
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import dataclass
import shutil

import numpy as np
import pandas as pd
import zarr
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class EntropyConfig:
    embedding_root: Path
    output_root: Path
    top_k: int
    entropy_method: str  # "shannon" or "diversity"
    standardize: bool = True


def compute_embedding_entropy(embeddings: np.ndarray, method: str = "shannon") -> np.ndarray:
    """
    Compute entropy for each embedding in a sequence.
    
    Args:
        embeddings: shape (seq_len, embedding_dim)
        method: "shannon" (entropy from probabilities) or "diversity" (norm-based)
    
    Returns:
        entropy scores: shape (seq_len,)
    """
    if method == "shannon":
        # Normalize embeddings to probability-like distribution
        embeddings_normalized = np.abs(embeddings) / (np.linalg.norm(np.abs(embeddings), axis=1, keepdims=True) + 1e-8)
        entropy_scores = -np.sum(embeddings_normalized * np.log(embeddings_normalized + 1e-10), axis=1)
    elif method == "diversity":
        # Use norm as diversity measure (high-magnitude embeddings are more informative)
        entropy_scores = np.linalg.norm(embeddings, axis=1)
    else:
        raise ValueError(f"Unknown entropy method: {method}")
    
    return entropy_scores


def select_top_k_frames(X: np.ndarray, top_k: int, entropy_method: str = "shannon") -> tuple[np.ndarray, np.ndarray]:
    """
    Select top-K frames from each window based on embedding entropy.
    
    Args:
        X: shape (num_windows, seq_len, embedding_dim)
        top_k: number of frames to select from each window
        entropy_method: entropy computation method
    
    Returns:
        selected_X: shape (num_windows, top_k, embedding_dim)
        selected_indices: shape (num_windows, top_k) - indices of selected frames
    """
    num_windows, seq_len, embedding_dim = X.shape
    
    if top_k > seq_len:
        raise ValueError(f"top_k ({top_k}) cannot exceed sequence length ({seq_len})")
    
    selected_X = np.zeros((num_windows, top_k, embedding_dim), dtype=X.dtype)
    selected_indices = np.zeros((num_windows, top_k), dtype=np.int32)
    
    for window_idx in range(num_windows):
        embeddings = X[window_idx]  # (seq_len, embedding_dim)
        entropy_scores = compute_embedding_entropy(embeddings, method=entropy_method)
        
        # Select top-K by entropy, then restore original temporal order so that
        # sequence models (RNN, TCN) receive frames in chronological sequence.
        top_indices = np.sort(np.argsort(entropy_scores)[-top_k:])

        selected_X[window_idx] = embeddings[top_indices]
        selected_indices[window_idx] = top_indices
    
    return selected_X, selected_indices


def process_zarr_store(
    input_store_path: Path,
    output_store_path: Path,
    top_k: int,
    entropy_method: str = "shannon",
    standardize: bool = True,
) -> dict:
    """
    Read a zarr store, select frames by entropy, write to new zarr store.
    
    Args:
        input_store_path: path to input zarr store
        output_store_path: path to output zarr store (will be created)
        top_k: number of frames to select
        entropy_method: entropy computation method
        standardize: whether to standardize frames before entropy computation
    
    Returns:
        dict with processing statistics
    """
    print(f"\nProcessing: {input_store_path.name}")
    
    # Open input store
    input_store = zarr.open(str(input_store_path), mode="r")
    X = input_store["X"][:]  # (num_windows, seq_len, embedding_dim)
    y = input_store["y"][:]

    original_seq_len = X.shape[1]
    print(f"  Input shape: {X.shape}")

    # Standardize only for entropy scoring — keep original embeddings for storage.
    # The downstream search (train_eval.py) re-standardizes fold-locally; writing
    # globally-standardized values here would cause double standardization.
    X_for_scoring = X
    if standardize:
        X_flat = X.reshape(-1, X.shape[-1])
        scaler = StandardScaler()
        X_for_scoring = scaler.fit_transform(X_flat).reshape(X.shape)

    # Select frames by entropy (scored on standardized copy, sliced from originals)
    _X_scored, frame_indices = select_top_k_frames(X_for_scoring, top_k, entropy_method=entropy_method)
    X_selected = np.stack([X[i, frame_indices[i]] for i in range(len(X))], axis=0)
    
    print(f"  Output shape: {X_selected.shape}")
    print(f"  Compression: {original_seq_len} → {top_k} frames ({100*top_k/original_seq_len:.1f}%)")
    
    # Create output zarr store by copying then updating X
    output_store_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy entire zarr directory
    if output_store_path.exists():
        shutil.rmtree(output_store_path)
    shutil.copytree(input_store_path, output_store_path)

    # Copy sibling metadata CSV required by load_embedding_store / discover_embedding_stores
    feature_name = input_store_path.stem
    src_meta = input_store_path.parent / f"{feature_name}_metadata.csv"
    dst_meta = output_store_path.parent / f"{feature_name}_metadata.csv"
    if src_meta.exists():
        shutil.copy2(src_meta, dst_meta)
    
    # Re-open and replace X dataset with selected frames
    output_store = zarr.open_group(str(output_store_path), mode="r+")
    
    # Remove and replace X dataset
    del output_store["X"]
    output_store.create_dataset(
        "X",
        data=X_selected.astype(X.dtype),
        chunks=(64, 32, X.shape[-1]),
    )
    
    # Update metadata in attributes
    output_store.attrs["original_frames_per_window"] = original_seq_len
    output_store.attrs["selected_frames_per_window"] = top_k
    output_store.attrs["entropy_method"] = entropy_method
    output_store.attrs["standardized"] = standardize
    output_store.attrs["frames_per_window"] = top_k
    
    return {
        "encoder": input_store_path.parent.name,
        "feature": input_store_path.stem.replace(".zarr", ""),
        "input_shape": str(X.shape),
        "output_shape": str(X_selected.shape),
        "compression_ratio": f"{100*top_k/original_seq_len:.1f}%",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Select information-dense frames from FM embeddings using entropy."
    )
    parser.add_argument(
        "--embedding-root",
        default="outputs_rgb_depth_fm/embeddings_zarr2",
        help="Root directory containing encoder zarr stores",
    )
    parser.add_argument(
        "--output-root",
        default="outputs_rgb_depth_fm/embeddings_zarr2_entropy_selected",
        help="Output root directory for entropy-selected zarr stores",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=75,
        help="Number of frames to select from each 150-frame window",
    )
    parser.add_argument(
        "--entropy-method",
        choices=["shannon", "diversity"],
        default="shannon",
        help="Method for computing frame entropy/informativeness",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        default=True,
        help="Standardize embeddings before entropy computation (default: True)",
    )
    parser.add_argument(
        "--no-standardize",
        dest="standardize",
        action="store_false",
        help="Skip standardization",
    )
    parser.add_argument(
        "--encoders",
        nargs="*",
        default=["imagebind", "omnivore", "dinov2"],
        help="Encoders to process",
    )
    parser.add_argument(
        "--features",
        nargs="*",
        default=[
            "masked_rgb",
            "masked_depth",
            "motion_prev_rgb",
            "motion_prev_depth",
            "flow_edge_rgb",
            "flow_edge_depth",
        ],
        help="Features to process",
    )
    
    args = parser.parse_args()
    
    embedding_root = Path(args.embedding_root)
    output_root = Path(args.output_root)
    
    print(f"Entropy-Based Frame Selection")
    print(f"=" * 60)
    print(f"Input root: {embedding_root}")
    print(f"Output root: {output_root}")
    print(f"Top-K frames: {args.top_k}")
    print(f"Entropy method: {args.entropy_method}")
    print(f"Standardize: {args.standardize}")
    
    results = []
    
    # Process all encoder/feature combinations
    for encoder in args.encoders:
        enc_dir = embedding_root / encoder
        if not enc_dir.exists():
            print(f"\nSkipping encoder {encoder} (not found)")
            continue
        
        for feature in args.features:
            input_zarr = enc_dir / f"{feature}.zarr"
            if not input_zarr.exists():
                print(f"\nSkipping {encoder}/{feature} (not found)")
                continue
            
            output_zarr = output_root / encoder / f"{feature}.zarr"
            
            try:
                result = process_zarr_store(
                    input_zarr,
                    output_zarr,
                    top_k=args.top_k,
                    entropy_method=args.entropy_method,
                    standardize=args.standardize,
                )
                results.append(result)
            except Exception as e:
                print(f"  ERROR: {e}")
    
    # Save summary
    summary_path = output_root / "entropy_selection_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "method": "entropy-based frame selection",
        "config": {
            "embedding_root": str(embedding_root),
            "top_k": args.top_k,
            "entropy_method": args.entropy_method,
            "standardize": args.standardize,
        },
        "results": results,
    }
    
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"✓ Processed {len(results)} encoder/feature combinations")
    print(f"✓ Output root: {output_root}")
    print(f"✓ Summary: {summary_path}")


if __name__ == "__main__":
    main()
