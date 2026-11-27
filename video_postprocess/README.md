# Video Postprocess

This folder contains utilities for GoPro-style video postprocessing. The core functions documented here are:

1. Cut a frame-accurate video segment for every camera directly from the raw per-camera video chunks, either by relative time or by embedded video timecode.
2. Split a requested interval of the raw per-camera video chunks into numbered frame images.
3. Generate a per-frame real-time timestamp CSV for every raw per-camera video chunk.
4. Find the raw frame each keypoint-pipeline session video starts at, so its frames can be given world-clock times.
5. Recalibrate the extrinsics of all top and side cameras of a session from every calibration board in dense, world-synchronized frames ([section 6](#joint-recalibration-with-every-board-camera_calibrationside_calib)).

Segment cutting, frame splitting, and timestamp generation all correct for cameras that record at a true rate (e.g. 59.94 fps) slightly off from the nominal 60 fps clock that timecodes and requested times are expressed against, so multi-file, multi-hour recordings stay frame-accurate.

The scripts use `ffmpeg` / `ffprobe` through Python, so both command-line tools must be available on `PATH`.

## Installation

1. Install [uv](https://docs.astral.sh/uv/)
2. Clone this repo
3. Go into this folder
    ```bash
    cd COSILab/video_postprocess
    ```
4. Create a venv and install the dependencies
    ```bash
    uv sync
    ```

The scripts live in the `video_postprocess` package under `src/video_postprocess/`, so the commands below run them as modules (`uv run python -m video_postprocess.<script>`) from this folder.

## 1. Fix Corrupted Videos

If a camera stopped recording because the GoPro battery died, repair the last chunk before doing any segment cutting or frame splitting.

Use:

```text
fix_gopro_battery_dies_video.py
```

Run:

```bash
uv run python -m video_postprocess.fix_gopro_battery_dies_video \
  --source-directory /path/to/raw_videos \
  --target-directory /path/to/fixed_videos
```

The script checks the last video in each camera folder. If it is truncated, it repairs it with [untrunc](https://github.com/anthwlock/untrunc) and writes the fixed file into the target directory, preserving the camera subfolder layout. `untrunc` is an external tool and must be installed separately.

## 2. Cut Video Segments From a Specific Interval

Use:

```text
extract_segment_from_video.py
```

This script works directly on raw per-camera video chunks. It expects a source directory containing one subdirectory per camera, each holding that camera's raw video chunks:

```text
raw_videos/
  cam01/
    GX010001.MP4
    GX020001.MP4
  cam02/
    GX010002.MP4
    GX020002.MP4
```

Cut a relative interval from every camera:

```bash
uv run python -m video_postprocess.extract_segment_from_video \
  --source-directory /path/to/raw_videos \
  --target-directory /path/to/output_segments \
  --start-time 00:01:00 \
  --end-time 00:02:00
```

With no `--use-timecode`, `--start-time` and `--end-time` are interpreted as offsets from the first frame of the first video in each camera folder.

Cut an absolute timecode interval:

```bash
uv run python -m video_postprocess.extract_segment_from_video \
  --source-directory /path/to/raw_videos \
  --target-directory /path/to/output_segments \
  --start-time 13:45:00 \
  --end-time 13:45:30 \
  --use-timecode
```

With `--use-timecode`, start/end times are matched directly against each camera's embedded video timecodes, spanning as many chunk files as needed.

Times can be given as `HH:MM:SS:FF`, `HH:MM:SS`, `MM:SS` or `SS` (`FF` is a frame number).

For each camera, the script:

1. Reads every chunk's embedded timecode, framerate and frame count with `video_segments.collect_camera_video_infos()`.
2. Resolves the requested start/end time to an exact `(file, frame)` position, correcting for the camera's true framerate against the nominal 60 fps clock the times are expressed against.
3. Builds the output by stream-copying whole frame ranges and re-encoding only the partial GOPs at the cut edges (so the cut is frame-exact without re-encoding footage that doesn't need it).
4. Concatenates the parts into a single output file per camera, embedding the actual clock timecode of the resolved start frame (the source file's own embedded timecode advanced to that frame) as the output's `-timecode`.

The output is one file per camera, named after the camera folder, with its own embedded start timecode matching the cut:

```text
output_segments/
  cam01.mp4
  cam02.mp4
```

### Process One Camera Only

Set `CAMERA_TO_PROCESS` to the camera folder name:

```bash
CAMERA_TO_PROCESS=cam02 uv run python -m video_postprocess.extract_segment_from_video \
  --source-directory /path/to/raw_videos \
  --target-directory /path/to/output_segments \
  --start-time 13:45:00 \
  --end-time 13:45:30 \
  --use-timecode
```

Only `cam02` will be processed.

### Fixed 30-Second Annotation Segments

For the current INGroup GoPro annotation split, `cut_gopro_annotation_segments.py` wraps `extract_segment_from_video.py` and cuts fixed 30-second timecode segments from the raw per-camera video chunks:

- Group cameras `06-10`: `13:45:00` to `14:20:00`
- Group cameras `01-05`: `14:52:00` to `15:27:00`

Run:

```bash
uv run python -m video_postprocess.cut_gopro_annotation_segments \
  --group-06-10-directory /path/to/group_06_10_raw_videos \
  --group-01-05-directory /path/to/group_01_05_raw_videos \
  --target-directory /path/to/annotation_segments
```

It creates folders like:

```text
annotation_segments/
  seg_134500_134530/
  seg_134530_134600/
```

and writes one cut video per camera into each segment folder.

### Batch Extract Predefined Segments

`batch_extract_segments.py` calls `extract_segment_from_video.py` once for each of a hard-coded list of 30-second timecode segments (`TIME_SEGS_1` / `TIME_SEGS_2`, edit the script to change them):

```bash
uv run python -m video_postprocess.batch_extract_segments \
  --source-directory /path/to/raw_videos \
  --target-directory /path/to/output_segments \
  --segment-set both \
  --use-timecode
```

Every segment in a run is written to the same `--target-directory`, and each camera's output file is always named `<camera>.mp4`, so **only the last segment of a run survives** in that directory. Use a separate `--target-directory` per segment (or per small batch) if you need to keep more than one.

## 3. Split Video Into Frame Images

Use:

```text
extract_frames_from_video.py
```

Like segment cutting, this works directly on raw per-camera video chunks:

```text
raw_videos/
  cam01/
    GX010001.MP4
  cam02/
    GX010002.MP4
```

Run:

```bash
uv run python -m video_postprocess.extract_frames_from_video \
  --source-directory /path/to/raw_videos \
  --target-directory /path/to/frames \
  --start-time 13:45:00 \
  --end-time 13:45:30 \
  --use-timecode
```

`--start-time`, `--end-time` and `--use-timecode` behave the same as in `extract_segment_from_video.py`, including the frame-rate correction and support for spanning multiple chunk files.

Frames are written as `%09d.png`, numbered continuously per camera even when the requested interval spans several chunk files:

```text
frames/
  cam01/
    000000000.png
    000000001.png
  cam02/
    000000000.png
    000000001.png
```

To sample instead of extracting every frame, pass `--every-n-frames`:

```bash
uv run python -m video_postprocess.extract_frames_from_video \
  --source-directory /path/to/raw_videos \
  --target-directory /path/to/frames \
  --every-n-frames 10
```

This keeps every 10th frame of the requested interval; the sampling grid stays anchored to the start of the interval even across a chunk-file boundary.

`CAMERA_TO_PROCESS` (see above) also limits this script to a single camera.

## 4. Generate Real-Time Timestamp CSVs

Use:

```text
generate_timestamps_csv.py
```

Like segment cutting and frame splitting, this works directly on raw per-camera video chunks:

```text
raw_videos/
  cam01/
    GX010001.MP4
  cam02/
    GX010002.MP4
```

Run:

```bash
uv run python -m video_postprocess.generate_timestamps_csv \
  --source-directory /path/to/raw_videos \
  --output-directory /path/to/timestamp_csvs
```

For every video file, this writes a CSV into `--output-directory`, recreating the camera subdirectory layout of `--source-directory` rather than writing next to the `.MP4` files:

```text
timestamp_csvs/
  cam01/
    GX010001.csv
  cam02/
    GX010002.csv
```

Each CSV has one row per frame and three columns:

```text
original_timecode, world_timecode,  world_timestamp
HH:MM:SS:FF,        HH:MM:SS:FF,     HH:MM:SS.ssssss
```

- `original_timecode` is the frame's embedded camera timecode, advanced once per frame per row.
- `world_timecode` is the inverse correction: the nominal (60 fps, drift-uncorrected) timecode that would need to be passed as `extract_segment_from_video.py`'s `--start-time --use-timecode` to seek to this same frame, undoing the same cross-file `physical_frames_before` correction `video_segments.py` applies when locating frames.
- `world_timestamp` is the same instant as `world_timecode`, with its `:FF` frame field converted to 6 fractional digits of seconds (1 µs resolution) instead.

Both columns correct for a camera's true framerate (e.g. 59.94 fps) against the nominal 60 fps clock the `:FF` field counts against — the per-frame version of the same correction `extract_segment_from_video.py` and `extract_frames_from_video.py` apply per-segment via `physical_frames_before`, so results stay consistent with those scripts across multi-file recordings.

`CAMERA_TO_PROCESS` (see above) also limits this script to a single camera.

### Segment Timecode CSV for a Cut Section

Use:

```text
generate_segment_timecode_csv.py
```

This is for a section cut with `extract_segment_from_video.py --use-timecode` (e.g. the 35 min sections) and then split into consecutive fixed-length segments. It computes everything from the section's `--start-time` / `--end-time` alone and reads no video.

Run:

```bash
uv run python -m video_postprocess.generate_segment_timecode_csv \
  --start-time 13:46:00 \
  --end-time 14:21:00 \
  --output-csv /path/to/segment_timecodes.csv
```

The times are nominal 60 fps timecodes, the same values passed to `extract_segment_from_video.py`. The footage is really 59.94 fps (60000/1001), so the section holds `round(nominal_frames * 1000/1001)` physical frames: a 35 min section has 125874 frames, not 126000. These are split into segments of `--frames-per-segment` frames (18000 by default), and the last segment takes the remainder. For 35 min that is six segments of 18000 frames and one of 17874.

The CSV has one row per frame:

```text
segment, frame, timecode
1,       0,     13:46:00:00
```

- `segment` is the 1-based segment number.
- `frame` is the 0-based frame index within that segment.
- `timecode` is the nominal 60 fps `HH:MM:SS:FF` of that frame: frame `g` of the section is `round(g * 1001/1000)` nominal frames after `--start-time`. It is the value `--start-time --use-timecode` would need to land on that frame, and one `FF` value is skipped roughly every 1000 frames.

When `extract_segment_from_video.py` locates the section start, it corrects for drift across the camera's earlier raw chunks (`physical_frames_before`). This script doesn't know about those chunks, so its timecodes can be off by ±1 frame from where the cut actually landed. For exact per-frame times of the raw chunks, use `generate_timestamps_csv.py` above.

## 5. Match Session Videos to Raw Frames

Use:

```text
match_session_video_to_raw.py
```

The keypoint pipeline ran on per-camera session videos (`mingling_session_videos/camXX.mp4`), split into 600-frame segments (`camXX_segNNN.mp4`). Each session video is a gap-free run of consecutive raw frames, continuing across chunk boundaries, extracted without the 59.94 fps correction. Its embedded timecode is not reliable. This script measures which raw frame each camera's session video starts at, by matching the first frames of its segments against the raw chunks.

Expected inputs:

```text
top_view_camera-mingle_session_2/   # --raw-directory
  camera_01-mingle_session_2/
    GH010189.MP4
    ...
video_segs_raw/                     # --segments-directory
  cam01/
    cam01_seg001.mp4
    ...
mingling_session_videos/            # --session-videos-directory
  cam01.mp4
```

Run once per session, into the same output CSV:

```bash
uv run python -m video_postprocess.match_session_video_to_raw \
  --raw-directory /path/to/gopro_data/top_view_camera-mingle_session_2 \
  --segments-directory /path/to/keypoints/video_segs_raw \
  --session-videos-directory /path/to/mingling_session_videos \
  --session-start 14:52:00:00 \
  --cameras 01,03,05 \
  --output-csv /path/to/session_video_raw_start.csv
```

`--session-start` is the time label of the first session video frame: `13:45:00:00` for session 1 and `14:52:00:00` for session 2.

For each camera, the script:

1. Reads every chunk's framerate and frame count with `video_segments.collect_camera_video_infos()`.
2. Converts `--session-start` to a first guess of the raw start frame, correcting for the true framerate, and searches `--first-search` frames around it (1200 by default) for the raw frame that best matches the first frame of segment 1. Frames are compared as small grayscale thumbnails by mean absolute difference.
3. Checks `--num-checks` further segments (5 by default): segment `n` must start exactly at raw frame `raw_start_frame + (n - 1) * 600`, searched within `--check-search` frames. It fails if any segment lands elsewhere or if no frame matches.
4. Checks that the whole session video fits within the readable raw chunks.

The output CSV has one row per camera. Rows already in the file for other cameras are kept:

```text
camera, session, raw_start_file, raw_start_local_frame, raw_start_frame, session_frames
```

- `raw_start_frame` counts frames across all readable chunks of the camera in playback order. `raw_start_file` and `raw_start_local_frame` give the same frame within its chunk.
- `session_frames` is the frame count of the session video.

Session video frame `s` is raw frame `raw_start_frame + s`, and its world-clock time is the matching row of the timestamp CSVs from [section 4](#4-generate-real-time-timestamp-csvs). `baselines/conversation_group/ViTPose/demo/add_world_time.py` uses this to add `world_time` to the keypoint dataframes. The measured values for the six keypoint cameras are in `baselines/conversation_group/ViTPose/demo/session_video_raw_start.csv`.

## 6. Camera Calibration Pipeline

The camera calibration utilities in this folder support this workflow:

```text
calibration images grouped by camera
  -> Camera Calibrator project JSON
  -> solve calibration in the external calibrator
  -> convert solved calibrator JSON to IDIAP or EasyMocap files
```

Main files:

```text
generate_calib_project.py
calib-project-template.json
convert_camera_calibration.py
convert_idiap_to_calibrator.py
camera_calibration/
  idiap.py
  easymocap.py
```

### Step 1: Prepare Calibration Images

Calibration frames can be sampled from the raw GoPro chunks with `extract_frames_from_video.py` (see [section 3](#3-split-video-into-frame-images)). `job_scripts/extract_calib.sh` (Linux) and `job_scripts/extract_calib.ps1` (Windows) do this for the side-view and top-view cameras of both mingle sessions, using the calibration timecode intervals and `--every-n-frames` sampling; edit the data paths at the top of the script before running it.

`generate_calib_project.py` expects one folder per camera:

```text
calib_images/
  cam01/
    frame_0001.jpg
    frame_0002.jpg
  cam02/
    frame_0001.jpg
    frame_0002.jpg
```

The script sorts camera folders and image filenames. Images with the same sorted index are treated as the same calibration pose across cameras, so keep filenames/order aligned across cameras.

### Step 2: Generate a Camera Calibrator Project

Run:

```bash
uv run python -m video_postprocess.generate_calib_project /path/to/calib_images \
  --output-filename /path/to/calib_images/calib-project.json
```

The script:

1. Loads `calib-project-template.json`.
2. Adds one camera entry per camera folder.
3. Reads the first image of each camera to set image width/height.
4. Initializes `cx` and `cy` to the image center.
5. Adds every image path to the calibrator `fileInfo` section.

The output `calib-project.json` is intended to be opened in the external Camera Calibrator. After detecting the calibration target and optimizing, export/save the solved calibrator JSON.

### Step 3: Convert Solved Calibration to Runtime Formats

Use:

```text
convert_camera_calibration.py
```

Convert to IDIAP-style per-camera JSON files:

```bash
uv run python -m video_postprocess.convert_camera_calibration \
  --calibrator-file /path/to/solved-calib-project.json \
  --camera-format idiap \
  --output-directory /path/to/camera_params \
  --cameras-names "cam01 cam02 cam03"
```

This writes:

```text
camera_params/
  cam01/
    intrinsic.json
    extrinsic.json
  cam02/
    intrinsic.json
    extrinsic.json
```

`intrinsic.json` contains:

```text
intrinsic              # 3x3 camera matrix
distortion_coefficients # [k1, k2, p1, p2, k3]
```

`extrinsic.json` contains:

```text
rvec
tvec
```

Convert to EasyMocap format instead:

```bash
uv run python -m video_postprocess.convert_camera_calibration \
  --calibrator-file /path/to/solved-calib-project.json \
  --camera-format easymocap \
  --output-directory /path/to/easymocap_camera \
  --cameras-names "cam01 cam02 cam03"
```

This writes:

```text
easymocap_camera/
  intri.yml
  extri.yml
```

If `--cameras-names` is omitted, the converter infers names from directories next to the calibrator file. Passing `--cameras-names` is safer because the names are assigned by sorted camera order.

### Optional: Merge IDIAP Extrinsics With Calibrator Intrinsics

Use:

```text
convert_idiap_to_calibrator.py
```

This script is for the case where intrinsics are already in calibrator JSON format, but extrinsics come from IDIAP-style `camera/extrinsic.json` files.

Expected inputs:

```text
idiap_extrinsics/
  cam01/
    extrinsic.json
  cam02/
    extrinsic.json

calibrator_intrinsics/
  cam01.json
  cam02.json
```

Run:

```bash
uv run python -m video_postprocess.convert_idiap_to_calibrator \
  --idiap-extrinsics-directory /path/to/idiap_extrinsics \
  --calibrator-intrinsics-directory /path/to/calibrator_intrinsics \
  --output-directory /path/to/combined_calibrator \
  --cameras-names "cam01.json cam02.json"
```

The script:

1. Loads each calibrator intrinsic file.
2. Loads the matching IDIAP `extrinsic.json`.
3. Converts translations from centimeters to meters.
4. Treats the first camera as the origin.
5. Transforms all other camera extrinsics into the first camera coordinate system.
6. Writes one combined calibrator JSON per camera.

### Optional: Calibration Inspection Helpers

`camera_calibration/` also contains standalone helpers that are not required by the conversion steps above:

```text
camera_calibration/
  detect_checkerboard.py      OpenCV checkerboard detection, written as calibrator detections JSON (typer CLI)
  show_saved_detections.py    interactive viewer for detections stored in a calibrator JSON (dearpygui, typer CLI)
  show_calib_v2.py            3D plot of camera poses from IDIAP-style <camera>/extrinsic.json folders (fastplotlib)
  relative_cam.py             chains two IDIAP extrinsics so camera 2 is expressed relative to camera 0
```

Run them the same way, e.g.:

```bash
uv run python -m video_postprocess.camera_calibration.show_calib_v2 /path/to/camera_params
```

where `/path/to/camera_params` is an IDIAP output directory from Step 3 (one subfolder per camera, each with `extrinsic.json`). Several such directories can be passed at once.

### Joint Recalibration With Every Board (`camera_calibration/side_calib/`)

The original extrinsics came from Calibrator projects with very few board views. `detect_checkerboard.py` finds only one board per image, while 2–3 boards are usually visible, and frames where different boards would be matched had to be discarded by hand.
- **Session 2:** the 4TU top calibration (every 120th frame) has 8–11 board detections per camera, only 2 of them for cam04. cam04 was about 5° off and cam03 about 3°.
- **Side cameras:** 12–15 were calibrated in separate 3-camera projects. Cams 12/14 were chained through one top camera (cam10 or cam04) and inherited its error.

The recalibration replaces this with one bundle adjustment per session over every board in dense, world-synchronized frames. The world-origin camera stays at its 4TU pose; all other top and side cameras are solved together. Intrinsics are not changed.

| session | fixed (world origin) | solved |
|---|---|---|
| 1 | cam06 | cams 07–10, 12–15 |
| 2 | cam01 | cams 02–04, 12–15 (cam05 is never used: its time sync is bad) |

Pipeline:

1. **Dense frames** (`extract_dense.py`)
   - Takes every 10th world tick (1/6 s) of the calibration window, decoded from the raw GoPro files: 1598 frames per camera for session 1, 998 for session 2.
   - World time comes from the raw MP4 timecode tags with the 4TU rule. It never comes from the `timecode_fixed` videos, which have wrong timecodes.
   - Top cameras use the 4TU `processed/videos/timestamps` CSVs. Side cameras use `timestamps.py`, which applies the same rule. `timestamps.py --check` reproduces all 4TU rows of the top cameras exactly.
   - Every frame is within 1 tick of its grid time.
2. **Board detection** (`detect_multi.py`)
   - Runs the same detector as `detect_checkerboard.py`: `findChessboardCorners(5,3)` followed by `cornerSubPix`.
   - Each image is searched in 4 versions: plain, CLAHE, 2× upscaled, and both. After each hit the board is masked and the search repeated, up to 4 boards per image.
   - The old calibration frames (`calib_frames/`, every 30th/60th frame) and the old Calibrator corners are added too.
3. **Automatic review and joint fit** (`auto_review.py`, `joint_fit.py`, driven by `joint.py`). This replaces the manual rotate/discard pass of `show_saved_detections.py`.
   - **Board poses:** each frame's boards are seeded from the top cameras by PnP and merged across top views.
   - **Matching:** side detections are assigned by Hungarian matching on projected corners, over the 4 corner orders (the "rotate").
   - **Discards:** detections without a consistent partner are dropped (the "discard").
   - **Fit:** a sparse `least_squares` (soft-L1 loss) solves all free camera poses and all board poses, with fisheye reprojection of the 15 corners.
   - **Iterations:** review and fit alternate 4 times, with the gates tightening: side 150 → 25 px, top 60 → 6 px. Views above 30 → 5 px are dropped.
4. **Evaluation** (`joint.py`)
   - The dense frames are split into 5 contiguous blocks, plus the old frames from the same moments. Each block is held out in turn.
   - Every held-out board is posed from its other views and projected into the remaining camera, with the 4TU/stored poses and with the fold's poses.
   - The delivered extrinsics come from one final fit on all data.

`test_synthetic.py` checks the review and the fit on synthetic boards seen by the real cameras, including flips, fake boards and perturbed starting poses. It recovers the poses to within 1 mm and 0.02°.

Run one session end to end (about 3–4 h for session 1 and 2–3 h for session 2 on 32 cores, mostly detection):

```bash
bash job_scripts/side_calib.sh 1
bash job_scripts/side_calib.sh 2
```

The script copies the old calibration frames (top and side, from the side calibration folder's `calib_frames/`), writes the side-camera timestamps, extracts and detects the dense frames, then runs `joint.py SESSION`. All outputs go to `video_postprocess/side_calib_work/` (gitignored); `/tudelft.net` is only read. Run the Python steps from `src/video_postprocess/camera_calibration/side_calib/`, because they import each other as plain modules. `joint.py SESSION --folds 0` runs the final fit only.

Outputs per session, in `side_calib_work/s{N}joint/`:

```text
idiap_params/mingling_N_top/camera?XX/extrinsic.json    new top extrinsics (not the fixed origin camera)
idiap_params/mingling_N_side/camera?XX/extrinsic.json   new side extrinsics (12-15)
review.log      detections per camera, per-iteration fit statistics, pose change per camera
eval.log        held-out evaluation (table below)
reviewed.json   every kept board: frame, pose, and per view the camera, corner order, source and error
```

Delivered copies (2026-11-27). In both calibration folders the previous calibration was moved to `original/`, which the code reads as its starting point and reference:
- **Top cameras:** `/tudelft.net/staff-umbrella/CosiLabExternal/4TU_upload/processed/videos/camera_calibration/idiap_params/`.
- **Side cameras:** `/tudelft.net/staff-umbrella/neon/cosilab_project/processed_data/gopro_data/side_view_camera_calibration/idiap_params/`.
- **Side-camera timestamp CSVs:** `/tudelft.net/staff-umbrella/neon/cosilab_project/raw_data/gopro_data/timestamps/`.

The side extrinsics share a world frame with the new top extrinsics; don't combine them with the original top ones.

The format is that of `idiap_params/` above: world-to-camera `rvec`/`tvec` in metres, in the world frame of the session's origin camera. Folder names follow the existing files: `camera-XX` for session 1 and `camera_XX` for session 2. The intrinsics are unchanged, so the existing `intrinsic.json` files still apply.

Held-out results: median (p90) mean corner error in px, on boards left out of the fit. The final fits reach 0.6–1.8 px median per camera, and the 5 fold poses agree with the final poses within 1.4 cm / 0.23° (session 1) and 5 cm / 0.63° (session 2).

| session 1 | 4TU / stored | new | pose change |
|---|---|---|---|
| cam06 | 2.7 (6.2) | 2.0 (4.6) | fixed |
| cam07 | 2.5 (8.7) | 1.4 (4.6) | 1.8 cm, 0.46° |
| cam08 | 3.8 (12.7) | 1.9 (5.5) | 3.0 cm, 0.65° |
| cam09 | 5.9 (12.9) | 3.0 (6.7) | 4.0 cm, 0.62° |
| cam10 | 9.3 (23.3) | 4.5 (10.4) | 5.6 cm, 1.34° |
| cam12 | 4.6 (17.7) | 2.3 (7.6) | 8.0 cm, 1.12° |
| cam13 | 8.8 (17.8) | 2.8 (7.2) | 5.9 cm, 1.20° |
| cam14 | 8.2 (21.1) | 2.9 (7.2) | 9.3 cm, 1.13° |
| cam15 | 15.9 (30.1) | 2.2 (7.9) | 17.9 cm, 2.50° |

| session 2 | 4TU / stored | new | pose change |
|---|---|---|---|
| cam01 | 3.7 (17.6) | 3.2 (11.6) | fixed |
| cam02 | 8.2 (38.6) | 3.3 (9.5) | 4.4 cm, 0.84° |
| cam03 | 28.4 (57.2) | 5.4 (12.3) | 11.2 cm, 2.87° |
| cam04 | 46.1 (70.6) | 5.9 (14.4) | 16.6 cm, 4.95° |
| cam12 | 7.2 (24.4) | 3.1 (7.8) | 35.0 cm, 4.98° |
| cam13 | 8.3 (39.5) | 4.7 (10.7) | 4.5 cm, 0.73° |
| cam14 | 5.3 (27.8) | 3.1 (10.4) | 52.9 cm, 5.30° |
| cam15 | 8.9 (33.3) | 3.3 (8.1) | 2.7 cm, 0.42° |

The session 2 cams 12/14 move about 5° with cam04: they had been calibrated relative to cam04. With the new poses the session 2 layout is regular:
- cams 01–04 lie on one line within 1 cm, about 1 m apart.
- Three side cameras sit on a rectangle aligned with that line.

## 7. Submit on the Cluster

`job_scripts/` holds the Slurm wrappers that run the segment-cutting steps above inside the `video_postprocess` Apptainer image, so no `uv sync` is needed on the cluster:

```text
job_scripts/
  extract_segments_daic.sh    DAIC, annotation or batch segment cutting
  extract_segments.sh         DelftBlue, batch segment cutting only
```

(`job_scripts/extract_calib.sh` and `extract_calib.ps1` are not Slurm wrappers: they run locally through `uv run --project`, see [Step 1: Prepare Calibration Images](#step-1-prepare-calibration-images).)

Both scripts bind the COSILab checkout into the container and set `PYTHONPATH` to `<repo>/video_postprocess/src`, so the job runs the code from your checkout rather than the copy frozen into the image at build time. Build the image first if it is missing:

```bash
apptainer build /path/to/video_postprocess.sif video_postprocess/apptainer/video_postprocess.def
```

### DAIC: `extract_segments_daic.sh`

Runs on the `insy,general` partitions and defaults to the `cosilab_project` locations under `/tudelft.net/staff-umbrella/neon`:

```text
SIF=/tudelft.net/staff-umbrella/neon/apptainer/video_postprocess.sif
REPO_DIR=/home/nfs/zli33/projects/COSILab
```

Two modes:

- `--mode annotation` (default) runs `cut_gopro_annotation_segments.py` for both camera groups, as described in [Fixed 30-Second Annotation Segments](#fixed-30-second-annotation-segments).
- `--mode batch` runs `batch_extract_segments.py` with the hardcoded `TIME_SEGS_1` / `TIME_SEGS_2` lists.

Both groups, default source and target paths:

```bash
sbatch video_postprocess/job_scripts/extract_segments_daic.sh
```

Only the `06-10` group, one camera, into a scratch target:

```bash
sbatch video_postprocess/job_scripts/extract_segments_daic.sh -g5 none \
  -c camera_06-mingle_session_1 \
  -t /path/to/video_clips_30s_test
```

Batch mode:

```bash
sbatch video_postprocess/job_scripts/extract_segments_daic.sh --mode batch \
  -s /path/to/raw_videos \
  -t /path/to/output_segments \
  --segment-set both --use-timecode
```

Options: `-g6` / `-g5` (annotation-mode source directories, `none` to skip a group), `-s` / `-t`, `--segment-set 1|2|both`, `--use-timecode`, `-c <camera>`, `--sif <path>`, `--repo <dir>`.

The script validates the image and the source layout before submitting work — it fails early if a source directory holds no camera subdirectories or no `.MP4` chunks — then verifies inside the container that `video_postprocess` imports from the checkout and aborts if it resolves to the image's own copy. ffmpeg's intermediate cut parts go to a per-job `/tmp` directory that is removed on exit.

### DelftBlue: `extract_segments.sh`

Runs batch mode only, on the `compute` partition under `--account=research-eemcs-insy`, with `/scratch/zli33` defaults:

```bash
sbatch video_postprocess/job_scripts/extract_segments.sh \
  -s /scratch/zli33/data/cosilab/raw_videos \
  -t /scratch/zli33/data/cosilab/segments \
  --segment-set 1 --use-timecode -c cam01
```

`-s` and `-t` are required. It takes the same `--segment-set`, `--use-timecode`, `-c`, `--sif`, and `--repo` options, plus `--baked-code` to ignore the checkout and use the code inside the image. Unlike the DAIC script it only warns, rather than failing, when the checkout is missing.

## Minimal Files Needed

If you only want to keep the segment-cutting, frame-splitting, and timestamp-CSV functions above, the required runtime files are:

```text
video_postprocess/
  __init__.py
  extract_segment_from_video.py
  extract_frames_from_video.py
  generate_timestamps_csv.py
  video_segments.py
  timecode.py
  utils.py
  video_utils.py
```

Keep these optional wrappers only if you use them:

```text
video_postprocess/
  fix_gopro_battery_dies_video.py    # repair a truncated last chunk with untrunc
  cut_gopro_annotation_segments.py   # fixed 30-second GoPro annotation cuts
  batch_extract_segments.py          # predefined segment lists TIME_SEGS_1 / TIME_SEGS_2
  generate_segment_timecode_csv.py   # segment/frame -> timecode CSV for a cut, split section
  match_session_video_to_raw.py      # raw start frame of the keypoint-pipeline session videos
  py.typed                           # optional typing marker, not needed at runtime
```

Everything else in `video_postprocess/` is unrelated to cutting video intervals, splitting videos into frames, or generating timestamp CSVs, and can be removed for this reduced workflow.

If you also want to keep camera calibration support, keep these additional files:

```text
video_postprocess/
  calib-project-template.json
  generate_calib_project.py
  convert_camera_calibration.py
  convert_idiap_to_calibrator.py
  camera_calibration/
    __init__.py
    idiap.py
    easymocap.py
```

Required Python packages for segment cutting, frame splitting, and timestamp CSV generation:

```text
av
click
numpy
opencv-python
psutil
pydantic
python-ffmpeg
tqdm
winfiletime
```

Additional Python packages for camera calibration utilities:

```text
PyYAML
scipy
```

`pyproject.toml` additionally declares the packages used only by the optional scripts:

```text
dearpygui                camera_calibration/show_saved_detections.py
fastplotlib, glfw        camera_calibration/show_calib_v2.py (glfw is the desktop window backend)
typer                    camera_calibration/detect_checkerboard.py, show_saved_detections.py
matplotlib               sync_by_audio.py
pandas                   extract_audio_from_midge.py
pillow                   add_black_border_to_imgs.py
```

Required system tools:

```text
ffmpeg
ffprobe
untrunc   # only for fix_gopro_battery_dies_video.py
```

Note: the scripts import modules as `video_postprocess.<module>`. Keep the package import path configured the same way as the current project does, or update those imports to match the directory name if you run the folder standalone.
