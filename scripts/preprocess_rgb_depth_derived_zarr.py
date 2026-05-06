#!/usr/bin/env python3
"""Create masked, motion, and RAFT flow-edge RGB/depth Zarr representations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
REQUIRED_RAW_ARRAYS = {
    "rgb",
    "depth",
    "target_timestamps",
    "rgb_source_timestamps",
    "depth_source_timestamps",
    "rgb_frame_indices",
    "depth_frame_indices",
}
TASKS_WITH_STRESS_LABELS = ("jelly", "count", "baseline", "bad", "stress", "arithmetic", "stroop")


@dataclass
class ModelBundle:
    yolo: object | None = None
    sam_predictor: object | None = None
    raft: object | None = None
    raft_weights: object | None = None
    raft_transforms: object | None = None
    torch: object | None = None
    device: str = "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build derived RGB/depth representations from raw 5 Hz task Zarr stores."
    )
    parser.add_argument("--input-root", default="processed_rgb_depth_zarr_5hz_raw")
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--representation",
        required=True,
        choices=["human_masked", "motion_previous", "motion_jelly_mean3", "flow_edge_raft"],
    )
    parser.add_argument("--participants", nargs="*", default=None)
    parser.add_argument("--tasks", nargs="*", default=None, choices=TASKS_WITH_STRESS_LABELS)
    parser.add_argument("--views", nargs="*", default=None, choices=["frontal", "side"])
    parser.add_argument("--max-task-groups", type=int, default=None)
    parser.add_argument("--frames-per-chunk", type=int, default=150)
    parser.add_argument("--compressor", default="zstd", choices=["zstd", "lz4", "blosclz", "zlib"])
    parser.add_argument("--compression-level", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--mask-source-root", default=None, help="Existing human_masked Zarr root for motion/flow stages.")
    parser.add_argument("--write-smoke-png", default=None, help="Optional PNG path for the first processed task.")
    parser.add_argument("--smoke-frame-index", type=int, default=None, help="Local frame index inside the task for PNG.")
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--sam2-config", default=None, help="SAM2 model config path/name.")
    parser.add_argument("--sam2-checkpoint", default=None, help="SAM2 checkpoint path.")
    parser.add_argument("--person-confidence", type=float, default=0.25)
    parser.add_argument("--mask-carry-forward-frames", type=int, default=15)
    parser.add_argument("--mask-min-area-fraction", type=float, default=0.01)
    parser.add_argument("--mask-max-area-fraction", type=float, default=0.70)
    parser.add_argument(
        "--allow-missing-masks",
        action="store_true",
        help="Write zero masks for frames with no valid direct/carry-forward mask instead of skipping the task.",
    )
    parser.add_argument("--mask-max-frames", type=int, default=None, help="Debug limit per task.")
    parser.add_argument("--start-frame", type=int, default=0, help="Local frame offset inside each task for smoke/debug runs.")
    parser.add_argument("--motion-window-seconds", type=float, default=30.0)
    parser.add_argument("--motion-stride-seconds", type=float, default=15.0)
    parser.add_argument("--sample-rate-hz", type=float, default=5.0)
    parser.add_argument("--raft-model", choices=["small", "large"], default="small")
    parser.add_argument("--raft-weights", default="C_T_V2")
    parser.add_argument("--flow-lag", type=int, default=5)
    parser.add_argument("--flow-edge-clamp", type=float, default=10.0)
    parser.add_argument("--flow-edge-gamma", type=float, default=1.0)
    parser.add_argument(
        "--depth-flow-mode",
        choices=["depth_diff_sobel", "raft_depth_pseudo"],
        default="depth_diff_sobel",
        help="Depth flow-edge strategy. Default avoids RAFT on fake RGB depth images.",
    )
    parser.add_argument(
        "--write-flow-edge-sweep-png",
        default=None,
        help="Optional PNG comparing flow-edge clamp/gamma visualization settings for the first processed task.",
    )
    return parser.parse_args()


def require_zarr_modules():
    try:
        import zarr
        from numcodecs import Blosc
    except ImportError as exc:
        raise SystemExit("Install zarr<3 and numcodecs in the active environment.") from exc
    major = int(str(getattr(zarr, "__version__", "0")).split(".", maxsplit=1)[0])
    if major >= 3:
        raise SystemExit(f"Unsupported zarr version {zarr.__version__}; this pipeline expects zarr<3.")
    return zarr, Blosc


def safe_name(value: object) -> str:
    text = SAFE_NAME_RE.sub("_", str(value)).strip("_")
    return text or "unknown"


def iter_task_groups(zarr, input_root: Path, args: argparse.Namespace) -> Iterable[tuple[Path, str, object]]:
    participants = {str(item) for item in args.participants or []}
    tasks = {str(item) for item in args.tasks or []}
    views = {str(item) for item in args.views or []}
    yielded = 0
    for store_path in sorted(input_root.glob("*/*.zarr")):
        root = zarr.open_group(str(store_path), mode="r")
        if "tasks" not in root:
            continue
        for task_name in sorted(root["tasks"].keys()):
            task_group = root["tasks"][task_name]
            attrs = dict(task_group.attrs)
            if participants and str(attrs.get("base_subject_id")) not in participants and str(attrs.get("imu_stem")) not in participants:
                continue
            if tasks and str(attrs.get("task_id")) not in tasks:
                continue
            if views and str(attrs.get("view_type")) not in views:
                continue
            if not REQUIRED_RAW_ARRAYS.issubset(set(task_group.keys())):
                continue
            yield store_path, task_name, task_group
            yielded += 1
            if args.max_task_groups is not None and yielded >= int(args.max_task_groups):
                return


def create_dataset(group, name: str, data: np.ndarray, chunks, compressor) -> None:
    arr = group.create_dataset(
        name,
        shape=data.shape,
        chunks=chunks,
        dtype=data.dtype,
        compressor=compressor,
        overwrite=True,
    )
    arr[:] = data


def copy_provenance_arrays(
    source_group,
    dest_group,
    chunk_n: int,
    compressor,
    length: int | None = None,
    start_frame: int = 0,
) -> None:
    for name in [
        "target_timestamps",
        "rgb_source_timestamps",
        "depth_source_timestamps",
        "rgb_frame_indices",
        "depth_frame_indices",
    ]:
        data = np.asarray(source_group[name][:])
        if start_frame:
            data = data[int(start_frame) :]
        if length is not None:
            data = data[: int(length)]
        create_dataset(dest_group, name, data, chunks=(chunk_n,), compressor=compressor)


def require_yolo_sam2(args: argparse.Namespace) -> ModelBundle:
    if not args.sam2_config or not args.sam2_checkpoint:
        raise SystemExit("--sam2-config and --sam2-checkpoint are required for SAM2 masking.")
    try:
        import torch
        from ultralytics import YOLO
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as exc:
        raise SystemExit(
            "SAM2 masking requires torch, ultralytics, and sam2. Install SAM2 in the active env, "
            "then pass --sam2-config and --sam2-checkpoint."
        ) from exc

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    try:
        yolo = YOLO(args.yolo_model)
    except Exception as exc:  # noqa: BLE001 - usually uncached/default YOLO weights offline.
        raise SystemExit(
            f"Could not load YOLO model {args.yolo_model}. Pass a local YOLO weights path or "
            "pre-cache the requested Ultralytics model before running on an offline machine."
        ) from exc
    sam2_config = resolve_sam2_config_name(args.sam2_config)
    sam_model = build_sam2(sam2_config, args.sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam_model)
    return ModelBundle(yolo=yolo, sam_predictor=predictor, torch=torch, device=device)


def resolve_sam2_config_name(config: str) -> str:
    """Accept either SAM2's package-relative config name or an absolute YAML path."""
    config_path = Path(config)
    if not config_path.is_absolute():
        return config
    try:
        import sam2
    except ImportError:
        return config
    package_root = Path(sam2.__file__).resolve().parent
    try:
        return config_path.resolve().relative_to(package_root).as_posix()
    except ValueError:
        return config


def require_raft(args: argparse.Namespace) -> ModelBundle:
    try:
        import torch
        from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights, raft_large, raft_small
    except ImportError as exc:
        raise SystemExit("RAFT flow-edge extraction requires torch and torchvision.") from exc
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    if args.raft_model == "large":
        weights_enum = Raft_Large_Weights
        model_fn = raft_large
    else:
        weights_enum = Raft_Small_Weights
        model_fn = raft_small
    weights = weights_enum[args.raft_weights]
    try:
        model = model_fn(weights=weights, progress=True).to(device).eval()
    except Exception as exc:  # noqa: BLE001 - typically uncached weights in offline environments.
        raise SystemExit(
            f"Could not load torchvision RAFT weights {args.raft_model}:{args.raft_weights}. "
            "If this machine has no internet, pre-cache the weights in "
            "$TORCH_HOME/hub/checkpoints or run once on a machine with network access."
        ) from exc
    return ModelBundle(raft=model, raft_weights=weights, raft_transforms=weights.transforms(), torch=torch, device=device)


def yolo_person_boxes(yolo, rgb: np.ndarray, confidence: float) -> np.ndarray:
    result = yolo.predict(rgb, classes=[0], conf=confidence, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)


def sam2_mask_for_frame(bundle: ModelBundle, rgb: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros(rgb.shape[:2], dtype=bool)
    predictor = bundle.sam_predictor
    predictor.set_image(rgb)
    masks, scores, _ = predictor.predict(box=boxes, multimask_output=False)
    masks = np.asarray(masks)
    scores = np.asarray(scores).reshape(-1)
    if masks.ndim == 4:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None, ...]
    if masks.size == 0:
        return np.zeros(rgb.shape[:2], dtype=bool)
    valid = scores >= 0 if scores.size == masks.shape[0] else np.ones(masks.shape[0], dtype=bool)
    return np.any(masks[valid].astype(bool), axis=0)


def valid_mask_area(mask: np.ndarray, min_fraction: float, max_fraction: float) -> tuple[bool, float]:
    area_fraction = float(np.count_nonzero(mask)) / float(mask.size)
    return float(min_fraction) <= area_fraction <= float(max_fraction), area_fraction


def masked_rgb_depth(rgb: np.ndarray, depth: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb_out = rgb.copy()
    depth_out = depth.copy()
    rgb_out[~mask] = 0
    depth_out[~mask] = 0
    return rgb_out, depth_out


def build_human_masked_arrays(source_group, bundle: ModelBundle, args: argparse.Namespace) -> dict[str, np.ndarray]:
    rgb = np.asarray(source_group["rgb"][:])
    depth = np.asarray(source_group["depth"][:])
    if args.start_frame:
        rgb = rgb[int(args.start_frame) :]
        depth = depth[int(args.start_frame) :]
    if args.mask_max_frames is not None:
        rgb = rgb[: int(args.mask_max_frames)]
        depth = depth[: int(args.mask_max_frames)]
    masks = np.zeros(rgb.shape[:3], dtype=np.uint8)
    rgb_masked = np.zeros_like(rgb)
    depth_masked = np.zeros_like(depth)
    mask_source = np.zeros(rgb.shape[0], dtype=np.uint8)
    mask_area_fraction = np.zeros(rgb.shape[0], dtype=np.float32)
    mask_age_frames = np.full(rgb.shape[0], -1, dtype=np.int16)
    yolo_box_count = np.zeros(rgb.shape[0], dtype=np.int16)
    last_valid_mask: np.ndarray | None = None
    last_valid_idx: int | None = None
    missing_frames: list[int] = []
    iterator = range(rgb.shape[0])
    if tqdm is not None and not args.no_progress:
        iterator = tqdm(iterator, desc="SAM2/YOLO masks", unit="frame", leave=False)
    for idx in iterator:
        boxes = yolo_person_boxes(bundle.yolo, rgb[idx], args.person_confidence)
        yolo_box_count[idx] = int(len(boxes))
        direct_mask = sam2_mask_for_frame(bundle, rgb[idx], boxes)
        direct_valid, direct_area = valid_mask_area(
            direct_mask,
            args.mask_min_area_fraction,
            args.mask_max_area_fraction,
        )
        if direct_valid:
            mask = direct_mask
            mask_source[idx] = 1
            mask_age_frames[idx] = 0
            last_valid_mask = mask.copy()
            last_valid_idx = idx
        elif (
            last_valid_mask is not None
            and last_valid_idx is not None
            and (idx - last_valid_idx) <= int(args.mask_carry_forward_frames)
        ):
            mask = last_valid_mask
            mask_source[idx] = 2
            mask_age_frames[idx] = int(idx - last_valid_idx)
        else:
            mask = np.zeros(rgb.shape[1:3], dtype=bool)
            missing_frames.append(idx + int(args.start_frame))
            mask_source[idx] = 0
            mask_age_frames[idx] = -1
        rgb_masked[idx], depth_masked[idx] = masked_rgb_depth(rgb[idx], depth[idx], mask)
        masks[idx] = mask.astype(np.uint8)
        mask_area_fraction[idx] = valid_mask_area(mask, 0.0, 1.0)[1] if mask.any() else direct_area
    if missing_frames and not args.allow_missing_masks:
        preview = ",".join(str(item) for item in missing_frames[:20])
        suffix = "" if len(missing_frames) <= 20 else f"...(+{len(missing_frames) - 20} more)"
        raise ValueError(
            "Missing valid human masks after YOLO+SAM2 and carry-forward policy. "
            f"frames={preview}{suffix}. Use --allow-missing-masks to write zero masks for inspection."
        )
    return {
        "rgb_masked": rgb_masked,
        "depth_masked": depth_masked,
        "human_mask": masks,
        "mask_source": mask_source,
        "mask_area_fraction": mask_area_fraction,
        "mask_age_frames": mask_age_frames,
        "yolo_box_count": yolo_box_count,
    }


def load_masked_task(zarr, mask_source_root: Path, attrs: dict, task_name: str):
    store = mask_source_root / safe_name(attrs["group"]) / f"{safe_name(attrs['base_subject_id'])}.zarr"
    root = zarr.open_group(str(store), mode="r")
    return root["tasks"][safe_name(task_name)]


def find_jelly_task(masked_tasks_group, attrs: dict):
    view = str(attrs.get("view_type"))
    for name in sorted(masked_tasks_group.keys()):
        tg = masked_tasks_group[name]
        if str(tg.attrs.get("task_id")) == "jelly" and str(tg.attrs.get("view_type")) == view:
            return tg
    return None


def motion_previous(rgb_masked: np.ndarray, depth_masked: np.ndarray) -> dict[str, np.ndarray]:
    motion_rgb = np.zeros_like(rgb_masked, dtype=np.int16)
    motion_depth = np.zeros(depth_masked.shape, dtype=np.float16)
    if rgb_masked.shape[0] > 1:
        motion_rgb[1:] = rgb_masked[1:].astype(np.int16) - rgb_masked[:-1].astype(np.int16)
        motion_depth[1:] = (depth_masked[1:].astype(np.float32) - depth_masked[:-1].astype(np.float32)).astype(np.float16)
    return {"motion_rgb": motion_rgb, "motion_depth": motion_depth}


def motion_jelly_mean3(zarr, mask_source_root: Path, attrs: dict, task_name: str, args: argparse.Namespace) -> dict[str, np.ndarray]:
    current = load_masked_task(zarr, mask_source_root, attrs, task_name)
    masked_store = mask_source_root / safe_name(attrs["group"]) / f"{safe_name(attrs['base_subject_id'])}.zarr"
    masked_root = zarr.open_group(str(masked_store), mode="r")
    jelly_task = find_jelly_task(masked_root["tasks"], attrs)
    if jelly_task is None:
        raise ValueError(f"No masked jelly task found for {attrs.get('base_subject_id')} {attrs.get('view_type')}")
    window = int(round(float(args.motion_window_seconds) * float(args.sample_rate_hz)))
    stride = int(round(float(args.motion_stride_seconds) * float(args.sample_rate_hz)))
    baseline_end = min(jelly_task["rgb_masked"].shape[0], window + 2 * stride)
    if baseline_end <= 0:
        raise ValueError("Jelly task has no frames for motion baseline")
    rgb_baseline = np.asarray(jelly_task["rgb_masked"][:baseline_end], dtype=np.float32).mean(axis=0)
    depth_baseline = np.asarray(jelly_task["depth_masked"][:baseline_end], dtype=np.float32).mean(axis=0)
    rgb = np.asarray(current["rgb_masked"][:], dtype=np.float32)
    depth = np.asarray(current["depth_masked"][:], dtype=np.float32)
    return {
        "motion_rgb": np.clip(rgb - rgb_baseline, -32768, 32767).astype(np.int16),
        "motion_depth": (depth - depth_baseline).astype(np.float16),
    }


def depth_to_rgb_like(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    finite = depth[np.isfinite(depth) & (depth > 0)]
    if finite.size:
        low, high = np.percentile(finite, [2, 98])
    else:
        low, high = 0.0, 1.0
    if high <= low:
        high = low + 1.0
    scaled = np.clip((depth - low) / (high - low), 0.0, 1.0)
    image = (scaled * 255.0).astype(np.uint8)
    return np.repeat(image[..., None], 3, axis=-1)


def raft_flow(bundle: ModelBundle, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    torch = bundle.torch
    with torch.inference_mode():
        t1 = torch.from_numpy(np.asarray(img1)).permute(2, 0, 1).float()[None]
        t2 = torch.from_numpy(np.asarray(img2)).permute(2, 0, 1).float()[None]
        t1, t2 = bundle.raft_transforms(t1, t2)
        t1 = t1.to(bundle.device)
        t2 = t2.to(bundle.device)
        flow = bundle.raft(t1, t2)[-1][0].detach().cpu().numpy()
    return np.moveaxis(flow, 0, -1)


def flow_edge_map(flow: np.ndarray, clamp_value: float, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for Sobel flow-edge maps.")
    magnitude = np.sqrt(np.sum(np.square(flow.astype(np.float32)), axis=-1))
    sx = cv2.Sobel(magnitude, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(magnitude, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(np.square(sx) + np.square(sy))
    edge = np.clip(edge, 0.0, float(clamp_value))
    if float(gamma) != 1.0:
        normalized = edge / max(float(clamp_value), 1e-6)
        edge = np.power(normalized, float(gamma)) * float(clamp_value)
    edge = edge.astype(np.float16)
    return magnitude.astype(np.float16), edge


def scalar_edge_map(values: np.ndarray, clamp_value: float, gamma: float) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for Sobel flow-edge maps.")
    values = np.asarray(values, dtype=np.float32)
    sx = cv2.Sobel(values, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(values, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(np.square(sx) + np.square(sy))
    edge = np.clip(edge, 0.0, float(clamp_value))
    if float(gamma) != 1.0:
        normalized = edge / max(float(clamp_value), 1e-6)
        edge = np.power(normalized, float(gamma)) * float(clamp_value)
    return edge.astype(np.float16)


def build_flow_edge_arrays(raw_group, masked_group, bundle: ModelBundle, args: argparse.Namespace) -> dict[str, np.ndarray]:
    rgb = np.asarray(raw_group["rgb"][:])
    depth = np.asarray(masked_group["depth_masked"][:])
    masks = np.asarray(masked_group["human_mask"][:]).astype(bool)
    masked_rgb_shape = masked_group["rgb_masked"].shape
    if rgb.shape[0] != masked_rgb_shape[0]:
        start_offset = int(masked_group.attrs.get("source_start_frame_offset", 0) or 0)
        rgb = rgb[start_offset : start_offset + masked_rgb_shape[0]]
    if args.mask_max_frames is not None:
        rgb = rgb[: int(args.mask_max_frames)]
        depth = depth[: int(args.mask_max_frames)]
        masks = masks[: int(args.mask_max_frames)]
    n = rgb.shape[0]
    flow_mag_rgb = np.zeros((n, rgb.shape[1], rgb.shape[2]), dtype=np.float16)
    flow_edge_rgb = np.zeros_like(flow_mag_rgb)
    flow_mag_depth = np.zeros_like(flow_mag_rgb)
    flow_edge_depth = np.zeros_like(flow_mag_rgb)
    iterator = range(int(args.flow_lag), n)
    if tqdm is not None and not args.no_progress:
        iterator = tqdm(iterator, desc="RAFT flow edges", unit="frame", leave=False)
    for idx in iterator:
        flow_rgb = raft_flow(bundle, rgb[idx - int(args.flow_lag)], rgb[idx])
        flow_mag_rgb[idx], flow_edge_rgb[idx] = flow_edge_map(flow_rgb, args.flow_edge_clamp, args.flow_edge_gamma)
        flow_mag_rgb[idx][~masks[idx]] = 0
        flow_edge_rgb[idx][~masks[idx]] = 0
        if args.depth_flow_mode == "raft_depth_pseudo":
            d1 = depth_to_rgb_like(depth[idx - int(args.flow_lag)])
            d2 = depth_to_rgb_like(depth[idx])
            flow_depth = raft_flow(bundle, d1, d2)
            flow_mag_depth[idx], flow_edge_depth[idx] = flow_edge_map(flow_depth, args.flow_edge_clamp, args.flow_edge_gamma)
        else:
            depth_diff = np.abs(depth[idx].astype(np.float32) - depth[idx - int(args.flow_lag)].astype(np.float32))
            flow_mag_depth[idx] = np.clip(depth_diff, 0.0, float(args.flow_edge_clamp)).astype(np.float16)
            flow_edge_depth[idx] = scalar_edge_map(depth_diff, args.flow_edge_clamp, args.flow_edge_gamma)
        flow_mag_depth[idx][~masks[idx]] = 0
        flow_edge_depth[idx][~masks[idx]] = 0
    return {
        "flow_magnitude_rgb": flow_mag_rgb,
        "flow_edge_rgb": flow_edge_rgb,
        "flow_magnitude_depth": flow_mag_depth,
        "flow_edge_depth": flow_edge_depth,
    }


def write_smoke_png(
    path: Path,
    arrays: dict[str, np.ndarray],
    source_group,
    attrs: dict,
    args: argparse.Namespace,
    source_start_frame: int | None = None,
) -> None:
    if path is None:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for --write-smoke-png") from exc
    idx = args.smoke_frame_index
    n = next(iter(arrays.values())).shape[0]
    if idx is None:
        idx = min(max(int(args.flow_lag), 0), max(n - 1, 0))
    idx = min(max(int(idx), 0), max(n - 1, 0))
    if source_start_frame is None:
        source_start_frame = int(args.start_frame)
    source_idx = idx + int(source_start_frame)
    rgb_key = "rgb" if "rgb" in source_group else "rgb_masked"
    depth_key = "depth" if "depth" in source_group else "depth_masked"
    raw_rgb = np.asarray(source_group[rgb_key][source_idx])
    raw_depth = np.asarray(source_group[depth_key][source_idx])
    rgb_title = "Raw RGB" if rgb_key == "rgb" else "Masked RGB Source"
    depth_title = "Raw Depth" if depth_key == "depth" else "Masked Depth Source"
    panels = [(rgb_title, raw_rgb), (depth_title, raw_depth)]
    for key in [
        "human_mask",
        "rgb_masked",
        "depth_masked",
        "motion_rgb",
        "motion_depth",
        "flow_magnitude_rgb",
        "flow_edge_rgb",
        "flow_edge_depth",
    ]:
        if key in arrays:
            value = arrays[key][idx]
            panels.append((key, value))
    cols = 5
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.asarray(axes).reshape(-1)
    for ax, (title, image) in zip(axes, panels):
        ax.set_title(title)
        display_image, cmap, kwargs = prepare_panel_for_display(title, image)
        if display_image.ndim == 2:
            ax.imshow(display_image, cmap=cmap or "magma", **kwargs)
        else:
            ax.imshow(display_image)
        ax.axis("off")
    for ax in axes[len(panels) :]:
        ax.axis("off")
    ts = np.asarray(source_group["target_timestamps"][:])
    text = (
        f"{attrs.get('base_subject_id')} {attrs.get('view_type')} {attrs.get('task_id')} "
        f"derived_frame={idx} raw_frame={source_idx} ts={ts[source_idx] if source_idx < len(ts) else 'NA'}\n"
        f"repr={args.representation} yolo={args.yolo_model} sam2_config={args.sam2_config} "
        f"sam2_ckpt={args.sam2_checkpoint} raft={args.raft_model}:{args.raft_weights} lag={args.flow_lag}"
    )
    fig.suptitle(text, fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def display_edge(edge: np.ndarray, clamp_value: float, gamma: float) -> np.ndarray:
    edge = np.asarray(edge, dtype=np.float32)
    edge = np.clip(edge, 0.0, float(clamp_value)) / max(float(clamp_value), 1e-6)
    if float(gamma) != 1.0:
        edge = np.power(edge, float(gamma))
    return np.clip(edge, 0.0, 1.0)


def prepare_panel_for_display(title: str, image: np.ndarray) -> tuple[np.ndarray, str, dict]:
    image = np.asarray(image)
    lower_title = title.lower()
    kwargs: dict = {}
    if image.ndim == 3 and image.shape[-1] == 3 and image.dtype.kind in {"i", "u"} and "motion" not in lower_title:
        return np.clip(image, 0, 255).astype(np.uint8), "", kwargs
    if image.ndim == 3:
        if "motion" in lower_title:
            image = np.mean(image.astype(np.float32), axis=-1)
        else:
            return np.clip(image, 0, 255).astype(np.uint8), "", kwargs
    image = image.astype(np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return image, "magma", kwargs
    if "motion" in lower_title:
        vmax = max(float(np.percentile(np.abs(finite), 99)), 1e-6)
        kwargs.update({"vmin": -vmax, "vmax": vmax})
        return image, "coolwarm", kwargs
    if "flow_edge" in lower_title or "flow_magnitude" in lower_title:
        vmax = max(float(np.percentile(finite, 99)), 1e-6)
        kwargs.update({"vmin": 0.0, "vmax": vmax})
        return image, "magma", kwargs
    if "depth" in lower_title:
        nonzero = finite[np.abs(finite) > 0]
        if nonzero.size:
            kwargs.update({"vmin": float(np.percentile(nonzero, 2)), "vmax": float(np.percentile(nonzero, 98))})
        return image, "magma", kwargs
    return image, "magma", kwargs


def write_flow_edge_sweep_png(path: Path, arrays: dict[str, np.ndarray], args: argparse.Namespace) -> None:
    if path is None or "flow_edge_rgb" not in arrays or "flow_edge_depth" not in arrays:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for --write-flow-edge-sweep-png") from exc
    n = arrays["flow_edge_rgb"].shape[0]
    idx = args.smoke_frame_index
    if idx is None:
        idx = min(max(int(args.flow_lag), 0), max(n - 1, 0))
    idx = min(max(int(idx), 0), max(n - 1, 0))
    clamp_values = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    gamma_values = [1.0, 0.75, 0.5, 0.33]
    images = {
        "rgb": np.asarray(arrays["flow_edge_rgb"][idx], dtype=np.float32),
        "depth": np.asarray(arrays["flow_edge_depth"][idx], dtype=np.float32),
    }
    rows = len(images) * len(gamma_values)
    cols = len(clamp_values)
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.8 * rows))
    axes = np.asarray(axes).reshape(rows, cols)
    row_idx = 0
    for modality, edge in images.items():
        for gamma in gamma_values:
            for col_idx, clamp in enumerate(clamp_values):
                ax = axes[row_idx, col_idx]
                ax.imshow(display_edge(edge, clamp, gamma), cmap="magma", vmin=0.0, vmax=1.0)
                ax.set_title(f"{modality} clamp={clamp:g} gamma={gamma:g}", fontsize=8)
                ax.axis("off")
            row_idx += 1
    fig.suptitle(f"Flow-edge display sweep, frame={idx}, stored clamp={args.flow_edge_clamp:g}, stored gamma={args.flow_edge_gamma:g}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def add_context_arrays_for_smoke(
    zarr,
    arrays: dict[str, np.ndarray],
    attrs: dict,
    task_name: str,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], object | None]:
    if not args.write_smoke_png or not args.mask_source_root:
        return arrays, None
    if {"human_mask", "rgb_masked", "depth_masked"}.issubset(arrays):
        return arrays, None
    try:
        masked = load_masked_task(zarr, Path(args.mask_source_root), attrs, task_name)
        context = {
            "human_mask": np.asarray(masked["human_mask"][:]),
            "rgb_masked": np.asarray(masked["rgb_masked"][:]),
            "depth_masked": np.asarray(masked["depth_masked"][:]),
        }
        if "motion_rgb" not in arrays and "motion_depth" not in arrays:
            context.update(motion_previous(context["rgb_masked"], context["depth_masked"]))
        context.update(arrays)
        return context, masked
    except Exception:
        return arrays, None


def output_arrays_for_representation(zarr, raw_group, attrs: dict, task_name: str, args: argparse.Namespace, bundle: ModelBundle):
    if args.representation == "human_masked":
        return build_human_masked_arrays(raw_group, bundle, args)
    if args.representation == "motion_previous":
        masked = load_masked_task(zarr, Path(args.mask_source_root), attrs, task_name)
        return motion_previous(np.asarray(masked["rgb_masked"][:]), np.asarray(masked["depth_masked"][:]))
    if args.representation == "motion_jelly_mean3":
        return motion_jelly_mean3(zarr, Path(args.mask_source_root), attrs, task_name, args)
    if args.representation == "flow_edge_raft":
        masked = load_masked_task(zarr, Path(args.mask_source_root), attrs, task_name)
        return build_flow_edge_arrays(raw_group, masked, bundle, args)
    raise ValueError(args.representation)


def provenance_source_group(zarr, raw_group, attrs: dict, task_name: str, args: argparse.Namespace):
    if args.representation == "human_masked":
        return raw_group, int(args.start_frame)
    if args.mask_source_root:
        masked = load_masked_task(zarr, Path(args.mask_source_root), attrs, task_name)
        return masked, 0
    return raw_group, 0


def main() -> None:
    args = parse_args()
    zarr, Blosc = require_zarr_modules()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.representation in {"motion_previous", "motion_jelly_mean3", "flow_edge_raft"} and not args.mask_source_root:
        raise SystemExit(f"--mask-source-root is required for {args.representation}")

    bundle = ModelBundle(device=args.device)
    if args.representation == "human_masked":
        bundle = require_yolo_sam2(args)
    elif args.representation == "flow_edge_raft":
        bundle = require_raft(args)

    compressor = Blosc(cname=args.compressor, clevel=int(args.compression_level), shuffle=Blosc.BITSHUFFLE)
    metadata_rows = []
    skipped_rows = []
    iterator = list(iter_task_groups(zarr, input_root, args))
    if tqdm is not None and not args.no_progress:
        iterator = tqdm(iterator, desc=f"Writing {args.representation}", unit="task")
    wrote_smoke = False

    for store_path, task_name, raw_group in iterator:
        attrs = dict(raw_group.attrs)
        output_store = output_root / safe_name(attrs["group"]) / f"{safe_name(attrs['base_subject_id'])}.zarr"
        task_group_name = safe_name(task_name)
        try:
            out_root = zarr.open_group(str(output_store), mode="a")
            out_root.attrs.update(
                {
                    "group": str(attrs.get("group")),
                    "base_subject_id": str(attrs.get("base_subject_id")),
                    "representation": args.representation,
                    "sample_rate_hz": float(args.sample_rate_hz),
                }
            )
            tasks_group = out_root.require_group("tasks")
            if task_group_name in tasks_group:
                if not args.overwrite:
                    skipped_rows.append({"task_record_id": task_name, "skip_reason": "task_group_exists_use_overwrite"})
                    continue
                del tasks_group[task_group_name]
            arrays = output_arrays_for_representation(zarr, raw_group, attrs, task_name, args, bundle)
            task_group = tasks_group.create_group(task_group_name)
            task_group.attrs.update({key: str(value) for key, value in attrs.items()})
            task_group.attrs.update(
                {
                    "representation": args.representation,
                    "source_start_frame_offset": int(args.start_frame),
                    "yolo_model": args.yolo_model,
                    "sam2_config": args.sam2_config,
                    "sam2_checkpoint": args.sam2_checkpoint,
                    "mask_source_codes": "0=missing,1=direct_yolo_sam2,2=carry_forward",
                    "mask_carry_forward_frames": int(args.mask_carry_forward_frames),
                    "mask_min_area_fraction": float(args.mask_min_area_fraction),
                    "mask_max_area_fraction": float(args.mask_max_area_fraction),
                    "allow_missing_masks": bool(args.allow_missing_masks),
                    "raft_model": args.raft_model,
                    "raft_weights": args.raft_weights,
                    "flow_lag": int(args.flow_lag),
                    "flow_edge_clamp": float(args.flow_edge_clamp),
                    "flow_edge_gamma": float(args.flow_edge_gamma),
                    "depth_flow_mode": args.depth_flow_mode,
                    "motion_window_seconds": float(args.motion_window_seconds),
                    "motion_stride_seconds": float(args.motion_stride_seconds),
                }
            )
            n = next(iter(arrays.values())).shape[0]
            chunk_n = max(1, min(int(args.frames_per_chunk), n))
            for name, data in arrays.items():
                chunks = (chunk_n, *data.shape[1:])
                create_dataset(task_group, name, data, chunks=chunks, compressor=compressor)
            source_group_for_provenance, provenance_start_frame = provenance_source_group(
                zarr,
                raw_group,
                attrs,
                task_name,
                args,
            )
            copy_provenance_arrays(
                source_group_for_provenance,
                task_group,
                chunk_n,
                compressor,
                length=n,
                start_frame=provenance_start_frame,
            )
            metadata_rows.append(
                {
                    "task_record_id": task_name,
                    "zarr_path": str(output_store),
                    "zarr_task_group": f"tasks/{task_group_name}",
                    "representation": args.representation,
                    "group": attrs.get("group"),
                    "base_subject_id": attrs.get("base_subject_id"),
                    "task_id": attrs.get("task_id"),
                    "view_type": attrs.get("view_type"),
                    "stress_label": attrs.get("stress_label"),
                    "num_samples": n,
                    "array_names": ",".join(sorted(arrays)),
                    "mask_missing_frames": int(np.count_nonzero(arrays.get("mask_source", np.ones(n, dtype=np.uint8)) == 0)),
                    "mask_direct_frames": int(np.count_nonzero(arrays.get("mask_source", np.zeros(n, dtype=np.uint8)) == 1)),
                    "mask_carry_forward_frames": int(np.count_nonzero(arrays.get("mask_source", np.zeros(n, dtype=np.uint8)) == 2)),
                }
            )
            if args.write_smoke_png and not wrote_smoke:
                smoke_arrays, smoke_source_group = add_context_arrays_for_smoke(zarr, arrays, attrs, task_name, args)
                plot_source_group = smoke_source_group if smoke_source_group is not None else raw_group
                plot_start_frame = 0 if smoke_source_group is not None else int(args.start_frame)
                write_smoke_png(
                    Path(args.write_smoke_png),
                    smoke_arrays,
                    plot_source_group,
                    attrs,
                    args,
                    source_start_frame=plot_start_frame,
                )
                if args.write_flow_edge_sweep_png:
                    write_flow_edge_sweep_png(Path(args.write_flow_edge_sweep_png), smoke_arrays, args)
                wrote_smoke = True
            elif args.write_flow_edge_sweep_png and not wrote_smoke:
                write_flow_edge_sweep_png(Path(args.write_flow_edge_sweep_png), arrays, args)
                wrote_smoke = True
        except Exception as exc:  # noqa: BLE001
            skipped_rows.append({"task_record_id": task_name, "skip_reason": f"{type(exc).__name__}: {exc}"})

    pd.DataFrame(metadata_rows).to_csv(output_root / f"{args.representation}_metadata.csv", index=False)
    pd.DataFrame(skipped_rows).to_csv(output_root / f"{args.representation}_skipped.csv", index=False)
    print(f"Wrote output root: {output_root}")
    print(f"Processed task groups: {len(metadata_rows)}")
    print(f"Skipped task groups: {len(skipped_rows)}")


if __name__ == "__main__":
    main()
