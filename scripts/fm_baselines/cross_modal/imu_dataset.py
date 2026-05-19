#!/usr/bin/env python3
"""
Standalone IMU windowed dataset.

Copied and adapted from:
  IMU-Stress-sensing/imu_stress/dataset.py
  IMU-Stress-sensing/imu_stress/labels.py
  IMU-Stress-sensing/imu_stress/utils.py

No imports from the IMU pipeline source — self-contained.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

NON_STRESS_TASKS = frozenset({"jelly", "count", "baseline"})
STRESS_TASKS = frozenset({"bad", "stress", "arithmetic", "stroop"})
TARGET_TASKS = NON_STRESS_TASKS | STRESS_TASKS

ACC_COLUMNS = ["acc0", "acc1", "acc2"]
SEQUENCE_MODES = {
    "raw_delta", "raw_absdelta", "delta_only", "absdelta_only",
    "raw_zdelta", "raw_abs_zdelta", "zdelta_only", "abs_zdelta_only",
    "raw_mad_delta", "raw_abs_mad_delta", "mad_delta_only", "abs_mad_delta_only",
}

_VERSION_SUFFIX_RE = re.compile(r"v\d+$")


def canonical_subject_id(subject_id: str) -> str:
    return _VERSION_SUFFIX_RE.sub("", subject_id)


def task_to_label(task_id: str) -> int:
    if task_id in NON_STRESS_TASKS:
        return 0
    if task_id in STRESS_TASKS:
        return 1
    raise ValueError(f"Unsupported task label: {task_id}")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SignalFile:
    file_id: int
    file_path: str
    population: str
    subject_id: str
    base_subject_id: str
    timestamps: np.ndarray
    signal: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class WindowMetadata:
    window_id: int
    file_id: int
    file_path: str
    population: str
    subject_id: str
    base_subject_id: str
    task_id: str
    label: int
    window_start: float
    window_end: float
    start_index: int
    end_index: int
    num_rows: int


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class IMUStressWindowDataset(Dataset):
    """Windowed dataset for left-hand accelerometer stress prediction."""

    def __init__(
        self,
        data_root: Path | str,
        window_seconds: float = 30.0,
        overlap_seconds: float = 15.0,
        sample_rate_hz: float = 32.0,
        raw_sample_rate_hz: float = 32.0,
        excluded_subject_ids: Sequence[str] | None = None,
        min_axis_std: float = 1e-3,
        min_magnitude_std: float = 1e-3,
        reserve_jelly_baseline_windows: bool = False,
        use_jelly_baseline_delta: bool = False,
        jelly_baseline_task: str = "jelly",
        jelly_baseline_windows: int = 3,
        jelly_delta_abs: bool = False,
        jelly_sequence_mode: str = "raw_delta",
        jelly_baseline_scale_eps: float = 1e-6,
    ) -> None:
        if overlap_seconds >= window_seconds:
            raise ValueError("overlap_seconds must be smaller than window_seconds")
        self.data_root = Path(data_root)
        self.window_seconds = float(window_seconds)
        self.overlap_seconds = float(overlap_seconds)
        self.sample_rate_hz = float(sample_rate_hz)
        self.raw_sample_rate_hz = float(raw_sample_rate_hz)
        self.excluded_subject_ids = set(excluded_subject_ids or [])
        self.min_axis_std = float(min_axis_std)
        self.min_magnitude_std = float(min_magnitude_std)
        self.reserve_jelly_baseline_windows = bool(reserve_jelly_baseline_windows or use_jelly_baseline_delta)
        self.use_jelly_baseline_delta = bool(use_jelly_baseline_delta)
        self.jelly_baseline_task = jelly_baseline_task
        self.jelly_baseline_windows = int(jelly_baseline_windows)
        self.jelly_delta_abs = bool(jelly_delta_abs)
        self.jelly_sequence_mode = jelly_sequence_mode
        self.jelly_baseline_scale_eps = float(jelly_baseline_scale_eps)
        self.window_size = int(round(window_seconds * sample_rate_hz))
        self.step_size = int(round((window_seconds - overlap_seconds) * sample_rate_hz))
        if self.window_size <= 0 or self.step_size <= 0:
            raise ValueError("Window and step sizes must be positive")
        if self.jelly_sequence_mode not in SEQUENCE_MODES:
            raise ValueError(f"jelly_sequence_mode must be one of {sorted(SEQUENCE_MODES)}")

        self.signal_files: List[SignalFile] = []
        self.windows: List[WindowMetadata] = []
        self._baseline_window_ids: set[int] = set()
        self._baseline_means_by_subject: Dict[str, np.ndarray] = {}
        self._baseline_mads_by_subject: Dict[str, np.ndarray] = {}
        self._baseline_stds_by_subject: Dict[str, np.ndarray] = {}
        self._reserved_baseline_windows: Dict[str, List[WindowMetadata]] = {}

        self._load_signal_files()
        self._build_windows()
        if self.reserve_jelly_baseline_windows:
            self._reserve_jelly_baseline_windows()

    def _load_signal_files(self) -> None:
        file_paths = sorted(self.data_root.rglob("left_*_acc.csv"))
        for file_id, file_path in enumerate(file_paths):
            population = file_path.parent.name
            subject_id = file_path.stem[len("left_"):-len("_acc")]
            base_subject_id = canonical_subject_id(subject_id)
            if subject_id in self.excluded_subject_ids or base_subject_id in self.excluded_subject_ids:
                continue
            df = pd.read_csv(file_path, usecols=["time", *ACC_COLUMNS, "label"])
            resampled_df = self._resample_acc_dataframe(df)
            labels = resampled_df["label"].fillna("").astype(str).to_numpy()
            self.signal_files.append(SignalFile(
                file_id=len(self.signal_files),
                file_path=str(file_path),
                population=population,
                subject_id=subject_id,
                base_subject_id=base_subject_id,
                timestamps=resampled_df["time"].to_numpy(dtype=np.float64),
                signal=resampled_df[ACC_COLUMNS].to_numpy(dtype=np.float32),
                labels=labels,
            ))

    def _resample_acc_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        start_time = float(df["time"].iloc[0])
        relative_time = np.arange(len(df), dtype=np.float64) / self.raw_sample_rate_hz
        source_index = pd.to_datetime(start_time + relative_time, unit="s")
        signal_df = pd.DataFrame(df[ACC_COLUMNS].to_numpy(dtype=np.float32), index=source_index, columns=ACC_COLUMNS)
        label_series = pd.Series(df["label"].fillna("").astype(str).to_numpy(), index=source_index, name="label")
        target_freq = f"{int(round(1e9 / self.sample_rate_hz))}ns"
        resampled_signal = signal_df.resample(target_freq).interpolate(method="time")
        resampled_labels = label_series.resample(target_freq).ffill().fillna("")
        resampled_df = resampled_signal.copy()
        resampled_df["label"] = resampled_labels.to_numpy()
        resampled_df["time"] = resampled_df.index.view("int64") / 1e9
        return resampled_df[["time", *ACC_COLUMNS, "label"]].reset_index(drop=True)

    def _build_windows(self) -> None:
        window_id = 0
        for signal_file in self.signal_files:
            labels = signal_file.labels
            segment_start = None
            current_task = None
            for index, raw_label in enumerate(labels):
                task_id = raw_label if raw_label in TARGET_TASKS else None
                if task_id is None:
                    if segment_start is not None:
                        window_id = self._add_segment_windows(signal_file, current_task, segment_start, index, window_id)
                        segment_start = None
                        current_task = None
                    continue
                if segment_start is None:
                    segment_start = index
                    current_task = task_id
                    continue
                if task_id != current_task:
                    window_id = self._add_segment_windows(signal_file, current_task, segment_start, index, window_id)
                    segment_start = index
                    current_task = task_id
            if segment_start is not None:
                window_id = self._add_segment_windows(signal_file, current_task, segment_start, len(labels), window_id)

    def _reserve_jelly_baseline_windows(self) -> None:
        windows_by_subject: Dict[str, List[WindowMetadata]] = {}
        for metadata in self.windows:
            windows_by_subject.setdefault(metadata.base_subject_id, []).append(metadata)
        kept_windows: List[WindowMetadata] = []
        for subject_windows in windows_by_subject.values():
            ordered = sorted(subject_windows, key=lambda w: (w.window_start, w.file_id, w.start_index))
            baseline_windows = [w for w in ordered if w.task_id == self.jelly_baseline_task][:self.jelly_baseline_windows]
            if len(baseline_windows) < self.jelly_baseline_windows:
                continue
            subject_id = baseline_windows[0].base_subject_id
            self._reserved_baseline_windows[subject_id] = list(baseline_windows)
            self._baseline_window_ids.update(w.window_id for w in baseline_windows)
            if self.use_jelly_baseline_delta:
                seqs = [self._raw_window_sequence(w) for w in baseline_windows]
                samples = np.concatenate(seqs, axis=0).astype(np.float32)
                mean = samples.mean(axis=0).astype(np.float32)
                self._baseline_means_by_subject[subject_id] = mean
                self._baseline_mads_by_subject[subject_id] = np.mean(np.abs(samples - mean), axis=0).astype(np.float32)
                self._baseline_stds_by_subject[subject_id] = samples.std(axis=0).astype(np.float32)
            kept_windows.extend(w for w in ordered if w.window_id not in self._baseline_window_ids)
        self.windows = sorted(kept_windows, key=lambda w: w.window_id)

    def _add_segment_windows(self, signal_file, task_id, segment_start, segment_end, next_window_id):
        segment_length = segment_end - segment_start
        if segment_length < self.window_size:
            return next_window_id
        label = task_to_label(task_id)
        last_start = segment_end - self.window_size
        for start_index in range(segment_start, last_start + 1, self.step_size):
            end_index = start_index + self.window_size
            sequence = signal_file.signal[start_index:end_index]
            if not self._passes_quality_filter(sequence):
                continue
            self.windows.append(WindowMetadata(
                window_id=next_window_id,
                file_id=signal_file.file_id,
                file_path=str(signal_file.file_path),
                population=signal_file.population,
                subject_id=signal_file.subject_id,
                base_subject_id=signal_file.base_subject_id,
                task_id=task_id,
                label=label,
                window_start=float(signal_file.timestamps[start_index]),
                window_end=float(signal_file.timestamps[end_index - 1]),
                start_index=start_index,
                end_index=end_index,
                num_rows=self.window_size,
            ))
            next_window_id += 1
        return next_window_id

    def _passes_quality_filter(self, sequence: np.ndarray) -> bool:
        if sequence.shape[0] != self.window_size:
            return False
        if not np.isfinite(sequence).all():
            return False
        if float(sequence.std(axis=0).max()) < self.min_axis_std:
            return False
        if float(np.linalg.norm(sequence, axis=1).std()) < self.min_magnitude_std:
            return False
        return True

    def _raw_window_sequence(self, metadata: WindowMetadata) -> np.ndarray:
        return self.signal_files[metadata.file_id].signal[metadata.start_index:metadata.end_index]

    def _window_sequence(self, metadata: WindowMetadata) -> np.ndarray:
        sequence = self._raw_window_sequence(metadata)
        if not self.use_jelly_baseline_delta:
            return sequence.copy()
        baseline_mean = self._baseline_means_by_subject.get(metadata.base_subject_id)
        if baseline_mean is None:
            raise KeyError(f"Missing jelly baseline for subject {metadata.base_subject_id}")
        baseline_mad = self._baseline_mads_by_subject[metadata.base_subject_id]
        baseline_std = self._baseline_stds_by_subject[metadata.base_subject_id]
        eps = self.jelly_baseline_scale_eps
        residual = sequence - baseline_mean[None, :]
        mode = self.jelly_sequence_mode
        if mode == "raw_delta":
            transformed = residual
        elif mode == "raw_absdelta":
            transformed = np.abs(residual)
        elif mode == "delta_only":
            return residual.astype(np.float32)
        elif mode == "absdelta_only":
            return np.abs(residual).astype(np.float32)
        elif mode == "raw_zdelta":
            transformed = residual / (baseline_std[None, :] + eps)
        elif mode == "raw_abs_zdelta":
            transformed = np.abs(residual) / (baseline_std[None, :] + eps)
        elif mode == "zdelta_only":
            return (residual / (baseline_std[None, :] + eps)).astype(np.float32)
        elif mode == "abs_zdelta_only":
            return (np.abs(residual) / (baseline_std[None, :] + eps)).astype(np.float32)
        elif mode == "raw_mad_delta":
            transformed = residual / (baseline_mad[None, :] + eps)
        elif mode == "raw_abs_mad_delta":
            transformed = np.abs(residual) / (baseline_mad[None, :] + eps)
        elif mode == "mad_delta_only":
            return (residual / (baseline_mad[None, :] + eps)).astype(np.float32)
        elif mode == "abs_mad_delta_only":
            return (np.abs(residual) / (baseline_mad[None, :] + eps)).astype(np.float32)
        else:
            raise ValueError(f"Unsupported jelly_sequence_mode: {mode}")
        return np.concatenate([sequence, transformed], axis=1).astype(np.float32)

    def get_sequence_array(self, index: int) -> np.ndarray:
        return self._window_sequence(self.windows[index])

    def get_raw_sequence_array(self, index: int) -> np.ndarray:
        return self._raw_window_sequence(self.windows[index]).copy()

    def labels_array(self) -> np.ndarray:
        return np.asarray([w.label for w in self.windows], dtype=np.int64)

    def metadata_frame(self) -> pd.DataFrame:
        return pd.DataFrame(asdict(w) for w in self.windows)

    def reserved_baseline_windows_for_subject(self, base_subject_id: str) -> List[WindowMetadata]:
        return list(self._reserved_baseline_windows.get(base_subject_id, []))

    @property
    def sequence_feature_dim(self) -> int:
        if not self.use_jelly_baseline_delta:
            return 3
        return 3 if self.jelly_sequence_mode.endswith("_only") else 6

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict:
        metadata = self.windows[index]
        sequence = self._window_sequence(metadata)
        return {
            "sequence": torch.from_numpy(sequence),
            "label": torch.tensor(metadata.label, dtype=torch.long),
            "metadata": asdict(metadata),
        }


def build_imu_dataset(
    data_root: str | Path,
    channel_mode: str = "raw_absdelta",
    window_seconds: float = 30.0,
    overlap_seconds: float = 15.0,
    sample_rate_hz: float = 64.0,
    raw_sample_rate_hz: float = 32.0,
) -> IMUStressWindowDataset:
    """
    Convenience constructor matching the FLIRT-Torch baseline defaults.
    channel_mode controls the sequence transformation applied to ACC windows.
    """
    _mode_kwargs = {
        "raw_only":        dict(reserve_jelly_baseline_windows=False, use_jelly_baseline_delta=False, jelly_sequence_mode="raw_delta"),
        "raw_absdelta":    dict(reserve_jelly_baseline_windows=True,  use_jelly_baseline_delta=True,  jelly_sequence_mode="raw_absdelta"),
        "raw_delta":       dict(reserve_jelly_baseline_windows=True,  use_jelly_baseline_delta=True,  jelly_sequence_mode="raw_delta"),
        "raw_abs_zdelta":  dict(reserve_jelly_baseline_windows=True,  use_jelly_baseline_delta=True,  jelly_sequence_mode="raw_abs_zdelta"),
    }
    if channel_mode not in _mode_kwargs:
        raise ValueError(f"channel_mode must be one of {list(_mode_kwargs)}; got {channel_mode!r}")
    return IMUStressWindowDataset(
        data_root=data_root,
        window_seconds=window_seconds,
        overlap_seconds=overlap_seconds,
        sample_rate_hz=sample_rate_hz,
        raw_sample_rate_hz=raw_sample_rate_hz,
        **_mode_kwargs[channel_mode],
    )
