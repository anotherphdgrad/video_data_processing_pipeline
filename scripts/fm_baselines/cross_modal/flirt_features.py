#!/usr/bin/env python3
"""
Standalone FLIRT ACC feature extractor.

Copied and adapted from IMU-Stress-sensing/imu_stress/features.py.
Only dependency beyond stdlib is the `flirt` package (pip install flirt)
and the local imu_dataset.IMUStressWindowDataset.
"""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from imu_dataset import ACC_COLUMNS, IMUStressWindowDataset


@dataclass
class FlirtACCFeatureExtractor:
    """Cache FLIRT ACC features per dataset window."""

    dataset: IMUStressWindowDataset
    num_cores: int = 1
    use_jelly_baseline_delta: bool = False
    feature_mode: str = "raw"
    use_abs_delta: bool = False

    def __post_init__(self) -> None:
        try:
            import flirt.acc  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "flirt package not installed. Run: pip install flirt"
            ) from exc
        self._feature_cache: Dict[int, np.ndarray] = {}
        self._feature_names: list[str] | None = None
        self._raw_feature_names: list[str] | None = None
        self._baseline_feature_cache: Dict[str, np.ndarray] = {}
        valid_modes = {"raw", "delta", "raw_delta"}
        if self.feature_mode not in valid_modes:
            raise ValueError(f"feature_mode must be one of {sorted(valid_modes)}")

    def feature_matrix(self, indices: Sequence[int]) -> np.ndarray:
        return np.stack([self.window_features(int(i)) for i in indices], axis=0)

    def window_features(self, index: int) -> np.ndarray:
        if index not in self._feature_cache:
            self._feature_cache[index] = self._compute_window_features(index)
        return self._feature_cache[index]

    def feature_names(self) -> list[str]:
        if self._feature_names is None:
            self.window_features(0)
        return self._feature_names

    def _compute_window_features(self, index: int) -> np.ndarray:
        metadata = self.dataset.windows[index]
        raw_features = self._raw_window_feature_vector(metadata)
        if not self.use_jelly_baseline_delta:
            if self._feature_names is None:
                self._feature_names = list(self._raw_feature_names)
            return raw_features

        baseline_features = self._baseline_feature_vector(metadata.base_subject_id)
        delta_features = raw_features - baseline_features
        if self.use_abs_delta:
            delta_features = np.abs(delta_features)
        if self.feature_mode == "delta":
            features = delta_features
            prefix = "abs_delta_" if self.use_abs_delta else "delta_"
            names = [f"{prefix}{n}" for n in self._raw_feature_names]
        elif self.feature_mode == "raw_delta":
            features = np.concatenate([raw_features, delta_features], axis=0)
            prefix = "abs_delta_" if self.use_abs_delta else "delta_"
            names = list(self._raw_feature_names) + [f"{prefix}{n}" for n in self._raw_feature_names]
        else:
            features = raw_features
            names = list(self._raw_feature_names)
        if self._feature_names is None:
            self._feature_names = names
        return features.astype(np.float32, copy=False)

    def _raw_window_feature_vector(self, metadata) -> np.ndarray:
        import flirt.acc
        signal_file = self.dataset.signal_files[metadata.file_id]
        sequence = self.dataset._raw_window_sequence(metadata)
        timestamps = signal_file.timestamps[metadata.start_index:metadata.end_index]
        acc_df = pd.DataFrame(sequence, columns=ACC_COLUMNS)
        acc_df.index = pd.to_datetime(timestamps, unit="s")
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            feature_df = flirt.acc.get_acc_features(
                acc_df,
                window_length=int(round(self.dataset.window_seconds)),
                window_step_size=int(round(self.dataset.window_seconds)),
                data_frequency=int(round(self.dataset.sample_rate_hz)),
                num_cores=self.num_cores,
            )
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if len(feature_df) != 1:
            raise ValueError(
                f"Expected 1 FLIRT row for window {metadata.window_id}, got {len(feature_df)}"
            )
        if self._raw_feature_names is None:
            self._raw_feature_names = list(feature_df.columns)
        return feature_df.iloc[0].to_numpy(dtype=np.float32)

    def _baseline_feature_vector(self, base_subject_id: str) -> np.ndarray:
        if base_subject_id not in self._baseline_feature_cache:
            baseline_windows = self.dataset.reserved_baseline_windows_for_subject(base_subject_id)
            if not baseline_windows:
                raise KeyError(f"Missing reserved jelly baseline windows for subject {base_subject_id}")
            baseline_features = [self._raw_window_feature_vector(w) for w in baseline_windows]
            self._baseline_feature_cache[base_subject_id] = (
                np.mean(np.stack(baseline_features, axis=0), axis=0).astype(np.float32)
            )
        return self._baseline_feature_cache[base_subject_id]
