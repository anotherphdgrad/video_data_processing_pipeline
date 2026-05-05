# RGB/Depth Stress Pipeline Preprocessing Notes

This document captures the current understanding of the RGB/depth assets, their relationship to the IMU stress dataset, and the intended preprocessing pipeline for building modeling-ready temporal clips.

It is based on:

- your description in chat
- the asset CSV files under [`assets/`](/home/harshit/2024/video_data_processing_pipeline/assets)
- a quick structural review of the manifests and path conventions

It is meant to serve as the working contract for the preprocessing stage.

## Goal

Build RGB-only and depth-only temporal pipelines for stress detection, using the same task labels and task boundaries as the IMU dataset.

For preprocessing specifically, the goal is to:

- locate RGB and depth sessions
- align them to task intervals
- downsample them to `5 Hz`
- compute a human mask from RGB frames using a SAM-family model
- apply that mask to the corresponding depth frames
- save background-removed RGB and depth temporal clips at `224 x 224`
- preserve timestamps so later stages can align clips with IMU segments
- store the outputs in compressed Zarr format

## Three Planned Stages

The full multimodal pipeline is currently planned in three stages:

1. Preprocessing
2. Caching
3. Modeling and evaluation

Current focus: preprocessing only.

## Asset Files Reviewed

### `assets/master_mapping_clean.csv`

Purpose:

- canonical task timing table derived from the audio timeline

Observed columns:

- participant identity: `group`, `id`, `global_time_id`
- task boundaries:
  - `start_*`
  - `end_*`

Important detail:

- `global_time_id` is the audio filename and acts as the audio-start reference
- task start/end values are in seconds relative to that audio reference

### `assets/manifest_mapping_clean_updated_sol.csv`

Purpose:

- session-level manifest for remote depth assets, with depth/audio timing alignment and view metadata

Observed columns:

- `participant_id`
- `participant_id_norm`
- `group`
- `global_time_id`
- `depth_path`
- `audio_start_ts`
- `depth_start_ts`
- `offset_depth`
- `is_session2`
- `has_session2`
- all task `start_*` and `end_*` columns
- `view_type`

Important details:

- this is the main table that currently connects task timing to depth sessions
- it already includes `view_type`
- it uses HPC-style depth paths such as `/scratch/hsharm62/OUD_Stress_depth/depth_hdf5/...`

### `assets/manifest_mapping_clean_updated_with_view_type.csv`

Purpose:

- same timing content, but using local mount-style depth paths

Important detail:

- depth paths point to `/home/harshit/mnt/sol_scratch/OUD_Stress_depth/depth_hdf5/...`

### `assets/inspection_depth_modalities_local_manifest.csv`

Purpose:

- another local manifest variant for inspection

Important detail:

- it does not currently include `view_type`

### `assets/imu_participant_mapping_candidates.csv`

Purpose:

- proposed crosswalk from RGB/depth participant IDs to IMU participant stems

Important detail:

- this is likely the right bridge for later RGB/depth-to-IMU alignment

## Current Data Inventory From The Manifest Review

From `manifest_mapping_clean_updated_sol.csv`:

- total rows: `124`
- unique normalized participants: `77`
- group split:
  - `Control`: `71` rows
  - `OUD`: `53` rows
- `view_type` counts:
  - `frontal`: `81`
  - `side`: `43`

Rows per participant:

- `30` participants have one recorded row
- `47` participants have two recorded rows

Interpreting that with the duplicated `(participant_id_norm, global_time_id)` pairs:

- `41` participant-session entries appear to have both views
- `36` participant-session entries appear to have a single view only

This means the preprocessing pipeline should assume:

- some sessions have both `frontal` and `side`
- some sessions have only one view

## Path Conventions

### Depth

Your current mounted local depth root exists at:

- `/home/harshit/mnt/sol_scratch/OUD_Stress_depth/depth_hdf5`

Your HPC root will use:

- `/scratch/hsharm62/OUD_Stress_depth/depth_hdf5`

### RGB

Confirmed current local RGB root:

- `/home/harshit/mnt/sol_scratch/OUD_Stress_RGB/rgb_hdf5`

Confirmed HPC RGB root to use:

- `/scratch/hsharm62/OUD_Stress_RGB/rgb_hdf5`

### RGB path derivation assumption

Based on your description and the filename pattern, the intended RGB session path appears derivable from the depth path by:

- keeping the same filename
- replacing the depth directory root with the RGB directory root

Example:

- depth: `/home/harshit/mnt/sol_scratch/OUD_Stress_depth/depth_hdf5/0001_20230201_114815.h5`
- rgb: `/home/harshit/mnt/sol_scratch/OUD_Stress_RGB/rgb_hdf5/0001_20230201_114815.h5`

This should be treated as a preprocessing rule only after we confirm that the RGB directory naming is stable and all expected files exist.

## Time Alignment Understanding

The current understanding is:

1. Task boundaries were defined using the audio timeline.
2. `global_time_id` identifies the audio start reference.
3. `audio_start_ts` is the audio absolute start timestamp.
4. `depth_start_ts` is the timestamp of the first depth frame in the H5 file.
5. `offset_depth` was computed so audio-based task times can be transferred onto the depth timeline.

The intended alignment logic is therefore:

- audio-relative task times come from the mapping tables
- depth-frame times come from the depth H5 timestamps
- `offset_depth` is used to convert task boundaries from audio reference to depth reference

Equivalent intuition:

- if depth starts earlier or later than audio, the task windows on the depth stream must be shifted accordingly

## Planned Temporal Units

The RGB and depth streams were collected at a higher frame rate than what we want for modeling.

Planned preprocessing output:

- downsample RGB and depth to `5 Hz`
- build temporal windows/clips
- save clips plus timestamps
- save both depth and RGB versions
- save masked variants with background removed

You confirmed that IMU-driven indexing is preferred.

That means the preprocessing pipeline should use IMU timestamps as the reference timeline for fetching RGB and depth frames.

Implications:

- task boundaries still originate from the audio-based mapping tables
- those task boundaries should be converted onto the IMU timeline
- RGB and depth frame indices should then be found from modality timestamps using the IMU-driven target timestamps
- we should explicitly detect missing coverage, where RGB or depth has no data for part or all of an IMU-aligned task interval

Important verification item:

- the `time` column in the IMU CSV should be checked carefully before implementation
- at least the top of `assets/IMU_data/Control/left_0001_acc.csv` shows repeated epoch-second values
- before relying on IMU-driven alignment, we should confirm whether the stored timestamps have sufficient sub-second precision throughout the file or whether effective IMU sample timing is reconstructed elsewhere

## Current Implementation Status

We have already completed two concrete pre-preprocessing inspection steps:

1. IMU timestamp inspection
2. IMU-to-video session manifest generation

### IMU timestamp inspection

Current inspection script:

- [`scripts/inspect_imu_timestamps.py`](/home/harshit/2024/video_data_processing_pipeline/scripts/inspect_imu_timestamps.py)

Generated outputs:

- [`assets/imu_time_inspection/imu_timestamp_file_summary.csv`](/home/harshit/2024/video_data_processing_pipeline/assets/imu_time_inspection/imu_timestamp_file_summary.csv)
- [`assets/imu_time_inspection/imu_timestamp_participant_index.csv`](/home/harshit/2024/video_data_processing_pipeline/assets/imu_time_inspection/imu_timestamp_participant_index.csv)
- participant-level CSVs under [`assets/imu_time_inspection/per_participant`](/home/harshit/2024/video_data_processing_pipeline/assets/imu_time_inspection/per_participant)

What this inspection showed:

- all currently inspected IMU files use integer-valued timestamps
- timestamps advance in `1`-second steps
- each second contains roughly `32` IMU rows
- effective IMU sample rate is therefore about `32 Hz`

Interpretation:

- the `time` column is not a true sub-second per-sample timestamp stream
- instead, it behaves like a coarse per-second wall-clock anchor with about `32` samples packed into each second

Example outputs:

- [`0001__timestamp_counts.csv`](/home/harshit/2024/video_data_processing_pipeline/assets/imu_time_inspection/per_participant/Control/0001__timestamp_counts.csv)
- [`0001__task_segments.csv`](/home/harshit/2024/video_data_processing_pipeline/assets/imu_time_inspection/per_participant/Control/0001__task_segments.csv)

This does not block IMU-driven alignment, but it means frame-level lookup must be done with the understanding that IMU timing is currently second-granularity plus an implicit sample rate, not explicit high-resolution timestamps.

### IMU-to-video session manifest generation

Current manifest generator:

- [`scripts/generate_imu_video_mapping_manifest.py`](/home/harshit/2024/video_data_processing_pipeline/scripts/generate_imu_video_mapping_manifest.py)

Generated outputs:

- [`assets/imu_video_mapping/imu_to_video_session_manifest.csv`](/home/harshit/2024/video_data_processing_pipeline/assets/imu_video_mapping/imu_to_video_session_manifest.csv)
- [`assets/imu_video_mapping/imu_to_video_missing_report.csv`](/home/harshit/2024/video_data_processing_pipeline/assets/imu_video_mapping/imu_to_video_missing_report.csv)
- [`assets/imu_video_mapping/imu_to_video_mapping_summary.csv`](/home/harshit/2024/video_data_processing_pipeline/assets/imu_video_mapping/imu_to_video_mapping_summary.csv)

Current runtime assumptions used by the mapper:

- IMU files are discovered from the local IMU root
- depth and RGB roots are passed as runtime arguments
- modality paths are rebuilt from filenames only, so path-prefix changes across servers do not break the mapping

### Current manifest-generation logic

The current mapper now does all of the following:

- starts from local IMU CSV files under `assets/IMU_data`
- uses approved participant mappings from `imu_participant_mapping_candidates.csv`
- joins to the video manifest using participant identity
- rebuilds `depth_path` and `rgb_path` from runtime roots plus H5 filename
- checks whether the rebuilt depth and RGB files actually exist
- parses the video session timestamp from the H5 filename
- compares that to the first timestamp in the IMU CSV
- keeps only session rows whose video timestamp is consistent with the IMU session timestamp

This timestamp-consistency check was added because participant identity alone was incorrectly pulling in extra session rows for some participants, especially frontal rows from later sessions.

### Session timestamp consistency check

The mapper currently uses:

- the first timestamp in the IMU CSV as the IMU session anchor
- the `YYYYMMDD_HHMMSS` portion of the H5 filename as the video session timestamp

The mapper records:

- `imu_first_time`
- `video_session_ts`
- `session_time_diff_seconds`

This check removed false same-participant matches that came from later session-2 style video files.

Important observation:

- the retained IMU/video matches still show multi-hour offsets between `imu_first_time` and `video_session_ts`
- this suggests a stable clock or timezone offset between the two recording systems

That is acceptable for session matching, but should be kept in mind during frame-level alignment work.

### Manual and special-case mapping decisions now encoded

The current mapper also includes a few explicit decisions we made during inspection:

- skip IMU participant `8876`
- treat `9933v2` as a `v2 -> base participant` case and map it to video participant `9933` using session-2 / `_p2_` rows
- treat `xianfei` as a valid participant mapping even though the IMU-side folder is `Control` and the video manifest rows are labeled `OUD`

For `xianfei`, the session manifest now preserves both:

- `group` from the IMU side
- `manifest_group` from the video manifest side

This makes the override explicit instead of silently erasing the discrepancy.

### Current mapping status

The current session-level mapping status is:

- `78` IMU files considered
- `78` IMU files mapped
- `0` missing approved mappings
- `0` missing depth files
- `0` missing RGB files

So the session-level manifest is now complete enough to proceed to frame-index mapping.

## Preprocessing Plan

### Stage 1: Session manifest normalization

Build one canonical manifest that:

- preserves `participant_id`, `participant_id_norm`, `group`
- preserves `global_time_id`
- preserves `view_type`
- contains both depth and RGB absolute paths
- records whether each modality file actually exists
- records audio/depth start timestamps and offsets
- optionally records IMU participant mapping

Recommended output fields:

- `participant_id`
- `participant_id_norm`
- `group`
- `global_time_id`
- `session_key`
- `view_type`
- `depth_path`
- `rgb_path`
- `audio_start_ts`
- `depth_start_ts`
- `rgb_start_ts` if available
- `offset_depth`
- `offset_rgb` if available
- `imu_stem`
- `is_session2`
- `has_session2`
- task boundary columns

### Stage 2: H5 schema inspection

Before coding the main preprocessor, verify for both RGB and depth H5 files:

- dataset names for frames
- dataset names for timestamps
- whether timestamps are seconds, milliseconds, or nanoseconds
- whether RGB and depth each have their own timestamps
- whether frame counts match timestamp counts
- frame shapes and channel layouts

Also verify for IMU:

- whether the CSV `time` column alone is sufficient for frame-level temporal alignment
- whether IMU sampling timestamps need any reconstruction beyond the stored values

### Stage 2.5: Build IMU-to-video frame mapping tables

Before writing the final clip preprocessor, build a dedicated mapping script that:

- reads IMU task intervals for a participant/session
- converts task windows onto the IMU timeline
- reads RGB and depth timestamps from the H5 files
- finds the corresponding RGB and depth frame indices for those IMU timestamps
- flags missing coverage and timestamp mismatches

Recommended outputs per participant-session-view:

- task name
- stress label
- IMU start timestamp
- IMU end timestamp
- RGB start frame index
- RGB end frame index
- depth start frame index
- depth end frame index
- coverage ratio for RGB
- coverage ratio for depth
- mismatch diagnostics

This mapping artifact should be created first, because it will tell us:

- whether IMU-driven alignment is clean enough
- where RGB/depth coverage is missing
- whether RGB and depth timestamps track each other closely enough for downstream clip building

### Stage 3: Build modality-aligned frame timeline

For each session and view:

- read RGB timestamps
- read depth timestamps
- compute or validate absolute start timestamps
- downsample each modality to `5 Hz`
- use IMU-defined target timestamps as the anchor
- pull nearest RGB and depth frames for each target timestamp
- record any missing or out-of-tolerance matches

### Stage 4: Human masking

For each selected RGB frame:

- run SAM-family segmentation
- extract the human/body mask
- save the RGB mask
- apply the same mask spatially to the corresponding depth frame

Important assumption:

- RGB and depth frames are already pixel-aligned enough for direct mask transfer

This assumption has now been explicitly confirmed by you for the intended preprocessing workflow.

### Stage 5: Clip extraction and Zarr writing

For each task segment:

- convert task boundary times to target timestamps
- slice downsampled RGB and depth into temporal clips
- resize/crop to `224 x 224`
- apply background removal
- write compressed Zarr arrays
- store clip-level timestamps and metadata

Recommended Zarr structure:

- `clips/rgb`
- `clips/depth`
- `clips/mask`
- `clips/timestamps`
- `clips/metadata`

Recommended clip metadata:

- participant ID
- group
- session key
- view type
- task ID
- binary stress label
- start timestamp
- end timestamp
- number of frames
- source depth path
- source RGB path
- IMU stem if available

## Labeling Strategy

Task labels should follow the same binary mapping as the IMU stress pipeline.

Non-stress:

- `jelly`
- `count`
- `baseline`

Stress:

- `bad`
- `stress`
- `arithmetic`
- `stroop`

Ignored unless you say otherwise:

- `good`
- `nature_video`
- `song`
- `speech`

## Risks And Open Technical Points

### 1. H5 timestamp schema still needs verification

I did not fully complete a live H5 schema inspection through the mount.

So the following are still assumptions until confirmed:

- timestamp dataset names
- units
- RGB/depth frame storage layout
- whether RGB has its own absolute timing field
- timestamp units and exact matching tolerances needed for IMU-driven alignment

### 2. IMU timestamp precision still needs verification

You want IMU timestamps to drive frame fetching, which is a good design choice.

Before implementation we still need to verify:

- whether the IMU CSV timestamps have frame-level precision directly
- or whether exact IMU sample times are reconstructed implicitly from sample rate plus coarse timestamp anchors

### 3. Mask transfer assumes RGB-depth spatial correspondence

Applying an RGB-derived mask directly onto depth assumes the two streams are spatially registered.

You confirmed that this assumption is valid for the current setup.

### 4. Offset handling is currently depth-specific

The manifest contains `offset_depth`.

You confirmed that RGB also has its own timestamps.

That means we likely also need:

- `rgb_start_ts`
- `offset_rgb`

### 5. View handling should remain explicit in metadata

You want:

- frontal-only evaluation later
- side-only evaluation later
- both-view evaluation later

So preprocessing should keep views separate and preserve `view_type` as metadata rather than collapsing views now.

## Proposed Preprocessing Output Contract

Suggested per-clip output:

- RGB clip: `T x 224 x 224 x 3`
- depth clip: `T x 224 x 224`
- mask clip: `T x 224 x 224`
- timestamps: `T`
- metadata row with participant, session, view, task, label, and modality source info

Suggested sampling:

- `5 Hz`

Suggested storage:

- compressed Zarr

Suggested naming granularity:

- one row per participant-session-view-task clip

Initial default windowing for later clip extraction:

- use the same clip duration and stride as the IMU torch FLIRT baselines

If needed, we can still keep the preprocessing output rich enough that later caching can regenerate alternative clip durations and strides.

## Resolved Decisions

The following decisions are now fixed for preprocessing:

1. Use `rgb_hdf5` as the RGB directory name locally and on HPC.

2. Treat RGB as having its own timestamps and plan for RGB-side temporal alignment checks.

3. Assume RGB and depth are spatially aligned enough to transfer the RGB human mask onto depth directly.

4. Use IMU timestamps as the reference timeline for fetching RGB and depth frames.

5. Start with the same clip duration and stride used in the IMU torch FLIRT baselines.

6. Keep `frontal` and `side` separate in the data and preserve `view_type` for later evaluation slices.

7. Drop non-target tasks from preprocessing outputs rather than preserving them for now.

## Remaining Verification Items

The main remaining items before implementation are now technical verification tasks rather than design decisions:

- inspect RGB and depth H5 schemas
- verify RGB timestamp semantics
- compute or validate `offset_rgb`
- verify IMU timestamp precision for frame-level alignment
- define acceptable RGB/depth timestamp mismatch thresholds for the mapping script

## Implemented Task-Aligned V1

The current implemented preprocessing path is intentionally raw-first:

- use IMU labels to derive coarse task start/end timestamps
- map those task intervals onto RGB/depth H5 timestamp arrays
- downsample task streams to `5 Hz`
- write participant-wise compressed Zarr stores
- defer SAM masking/background removal until the raw alignment and storage path is validated

This means the first local Zarr outputs are unmasked RGB/depth task streams, not final masked clips.

### Session manifest generation

Script:

- [`scripts/generate_imu_video_mapping_manifest.py`](/home/harshit/2024/video_data_processing_pipeline/scripts/generate_imu_video_mapping_manifest.py)

Important behavior:

- reconstructs RGB/depth paths from H5 filenames plus runtime `--depth-root` and `--rgb-root`
- skips IMU participant `8876`
- preserves `xianfei` as `group=Control` while retaining `manifest_group` provenance
- maps `9933v2` to the `9933` session-2/video `_p2_` files
- if multiple video candidates exist, keeps the nearest video row per `(group, imu_stem, view_type)`
- keeps frontal and side views as separate rows when both are available
- adds audit columns for candidate counts, selected rank, dropped duplicate candidates, and threshold status

Recommended local command:

```bash
source /home/harshit/anaconda3/etc/profile.d/conda.sh
conda activate imagebind
python scripts/generate_imu_video_mapping_manifest.py \
  --imu-root assets/IMU_data \
  --manifest-csv assets/manifest_mapping_clean_updated_sol.csv \
  --candidate-csv assets/imu_participant_mapping_candidates.csv \
  --depth-root /home/harshit/mnt/sol_scratch/OUD_Stress_depth/depth_hdf5 \
  --rgb-root /home/harshit/mnt/sol_scratch/OUD_Stress_RGB/rgb_hdf5 \
  --output-root assets/imu_video_mapping
```

### Task/frame manifest generation

Script:

- [`scripts/generate_rgb_depth_task_frame_manifest.py`](/home/harshit/2024/video_data_processing_pipeline/scripts/generate_rgb_depth_task_frame_manifest.py)

Important behavior:

- reads the session manifest
- reconstructs IMU sample timestamps using the IMU branch convention: first IMU timestamp plus row index at `32 Hz`
- keeps only IMU target tasks: `jelly`, `count`, `baseline`, `bad`, `stress`, `arithmetic`, `stroop`
- maps binary labels using the IMU branch contract:
  - non-stress: `jelly`, `count`, `baseline`
  - stress: `bad`, `stress`, `arithmetic`, `stroop`
- reads only H5 timestamp arrays for mapping
- outputs RGB/depth start/end frame indices and coverage status for every participant/session/view/task

Recommended local command:

```bash
source /home/harshit/anaconda3/etc/profile.d/conda.sh
conda activate imagebind
python scripts/generate_rgb_depth_task_frame_manifest.py \
  --session-manifest assets/imu_video_mapping/imu_to_video_session_manifest.csv \
  --output-csv assets/imu_video_mapping/rgb_depth_task_frame_manifest.csv
```

This shows a `tqdm` progress bar over session rows. Use `--no-progress` if running in a logger that does not handle progress bars well.

Smoke-test command:

```bash
source /home/harshit/anaconda3/etc/profile.d/conda.sh
conda activate imagebind
python scripts/generate_rgb_depth_task_frame_manifest.py \
  --session-manifest assets/imu_video_mapping/imu_to_video_session_manifest.csv \
  --output-csv /tmp/rgb_depth_task_frame_manifest_smoke.csv \
  --participants 0001 \
  --max-session-rows 2
```

Fast interval-only smoke-test command:

```bash
source /home/harshit/anaconda3/etc/profile.d/conda.sh
conda activate imagebind
python scripts/generate_rgb_depth_task_frame_manifest.py \
  --session-manifest assets/imu_video_mapping/imu_to_video_session_manifest.csv \
  --output-csv /tmp/rgb_depth_task_intervals_0001_smoke.csv \
  --participants 0001 \
  --max-session-rows 2 \
  --skip-h5
```

Use `--skip-h5` only to validate IMU-derived task intervals. It does not generate usable RGB/depth frame indices for Zarr extraction.

### Raw 5 Hz Zarr preprocessing

Script:

- [`scripts/preprocess_rgb_depth_task_zarr.py`](/home/harshit/2024/video_data_processing_pipeline/scripts/preprocess_rgb_depth_task_zarr.py)

Important behavior:

- consumes `rgb_depth_task_frame_manifest.csv`
- samples target timestamps at `5 Hz`
- chooses nearest RGB and depth frames within a configurable tolerance
- writes one compressed Zarr store per `group/base_subject_id`
- writes one task group per participant/view/task segment
- stores RGB, depth, target timestamps, source timestamps, source frame indices, task id, stress label, participant id, view type, and source file provenance
- defaults to resizing frames to `224 x 224`
- uses Blosc/Zstandard compression with frame chunks of `150` samples, matching `30s` at `5 Hz`

Recommended local smoke-test command:

```bash
source /home/harshit/anaconda3/etc/profile.d/conda.sh
conda activate imagebind
python scripts/preprocess_rgb_depth_task_zarr.py \
  --task-frame-manifest assets/imu_video_mapping/rgb_depth_task_frame_manifest.csv \
  --output-root /tmp/rgb_depth_zarr_smoke \
  --participants 0001 \
  --tasks jelly \
  --views frontal \
  --max-rows 1 \
  --overwrite
```

This shows a `tqdm` progress bar over task groups and a nested frame progress bar for the current task. Each completed task prints the destination Zarr store path.

### Evaluation alignment

The metadata written by the task/frame manifest and Zarr preprocessing includes `base_subject_id`.

This should be used for future RGB/depth folds so evaluation remains aligned with:

- `/home/harshit/2024/IMU_stress_sensing_src/imu_stress/dataset.py`
- `/home/harshit/2024/IMU_stress_sensing_src/imu_stress/folds.py`

The intended evaluation contract remains:

- participant-disjoint outer folds via `GroupKFold`
- canonical participant grouping via `base_subject_id`
- inner validation split via `GroupShuffleSplit`
- default fold seed `42`, with inner split seed `42 + fold_id`

## HPC Launch

HPC-safe launch files:

- [`scripts/hpc_rgb_depth_preprocessing.sh`](/home/harshit/2024/video_data_processing_pipeline/scripts/hpc_rgb_depth_preprocessing.sh)
- [`scripts/hpc_rgb_depth_preprocessing.sbatch`](/home/harshit/2024/video_data_processing_pipeline/scripts/hpc_rgb_depth_preprocessing.sbatch)

The launcher regenerates the session manifest with HPC roots:

- depth: `/scratch/hsharm62/OUD_Stress_depth/depth_hdf5`
- RGB: `/scratch/hsharm62/OUD_Stress_RGB/rgb_hdf5`

It writes HPC-specific mapping outputs by default to:

- `assets/imu_video_mapping_hpc/imu_to_video_session_manifest.csv`
- `assets/imu_video_mapping_hpc/rgb_depth_task_frame_manifest.csv`

It writes the local compressed Zarr dataset on scratch by default to:

- `/scratch/hsharm62/OUD_Stress_preprocessed/rgb_depth_zarr_5hz_raw`

### Interactive HPC Run

Activate a conda environment that already has the dependencies, then run from the repo root on HPC:

```bash
source /home/harshit/anaconda3/etc/profile.d/conda.sh
conda activate imagebind

cd /scratch/hsharm62/video_data_processing_pipeline

bash scripts/hpc_rgb_depth_preprocessing.sh
```

This runs all stages:

- session manifest with `/scratch/hsharm62` RGB/depth paths
- task/frame manifest
- 5 Hz raw RGB/depth Zarr writing

### SLURM HPC Run

If your cluster uses SLURM:

```bash
source /home/harshit/anaconda3/etc/profile.d/conda.sh
conda activate imagebind

cd /scratch/hsharm62/video_data_processing_pipeline

sbatch scripts/hpc_rgb_depth_preprocessing.sbatch
```

The launcher assumes the correct environment is already active. Edit `scripts/hpc_rgb_depth_preprocessing.sbatch` first if your cluster requires `--partition`, `--account`, or GPU directives.

### Smoke Tests On HPC

Generate manifests only:

```bash
STAGE=manifests bash scripts/hpc_rgb_depth_preprocessing.sh
```

Smoke test one participant through task/frame mapping:

```bash
STAGE=task_frames \
PARTICIPANTS="0001" \
MAX_SESSION_ROWS=2 \
bash scripts/hpc_rgb_depth_preprocessing.sh
```

Smoke test one Zarr task after a task/frame manifest exists:

```bash
STAGE=zarr \
PARTICIPANTS="0001" \
TASKS="jelly" \
VIEWS="frontal" \
MAX_ZARR_ROWS=1 \
OVERWRITE_ZARR=1 \
bash scripts/hpc_rgb_depth_preprocessing.sh
```

### Useful Overrides

The HPC launcher is controlled by environment variables:

- `PROJECT_ROOT`: repo path, default is inferred from the launcher location
- `DEPTH_ROOT`: depth H5 root, default `/scratch/hsharm62/OUD_Stress_depth/depth_hdf5`
- `RGB_ROOT`: RGB H5 root, default `/scratch/hsharm62/OUD_Stress_RGB/rgb_hdf5`
- `MAPPING_OUTPUT_ROOT`: manifest output root, default `assets/imu_video_mapping_hpc`
- `ZARR_OUTPUT_ROOT`: compressed Zarr output root, default `/scratch/hsharm62/OUD_Stress_preprocessed/rgb_depth_zarr_5hz_raw`
- `STAGE`: `all`, `session`, `task_frames`, `manifests`, or `zarr`
- `PARTICIPANTS`, `TASKS`, `VIEWS`: optional space-separated filters
- `MAX_SESSION_ROWS`, `MAX_ZARR_ROWS`: optional smoke-test row limits
- `OVERWRITE_ZARR=1`: replace existing task groups
- `REQUIRE_COMPLETE=1`: skip partial-coverage task rows during Zarr writing

Progress bars are enabled by default in the task/frame and Zarr stages.
