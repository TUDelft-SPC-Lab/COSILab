# Video Postprocess

This folder contains utilities for GoPro-style video postprocessing. The core functions documented here are:

1. Concatenate raw per-camera video chunks into one continuous video per camera.
2. Cut a frame-accurate video segment for every camera directly from the raw per-camera video chunks, either by relative time or by embedded video timecode.
3. Split a requested interval of the raw per-camera video chunks into numbered frame images.

Segment cutting and frame splitting both correct for cameras that record at a true rate (e.g. 59.94 fps) slightly off from the nominal 60 fps clock that timecodes and requested times are expressed against, so multi-file, multi-hour recordings stay frame-accurate.

The scripts use `ffmpeg` / `ffprobe` through Python, so both command-line tools must be available on `PATH`.

## Installation

1. Install [uv](https://docs.astral.sh/uv/)
2. Clone this repo
3. Go into this folder
    ```bash
    cd COSILab/video_postprocess
    ```
2. Create a venv and install the dependencies
    ```bash
    uv sync
    ```

## 1. Fix Corrupted Videos

If a camera stopped recording because the GoPro battery died, repair the last chunk before doing any concatenation, segment cutting, or frame splitting.

Use:

```text
fix_gopro_battery_dies_video.py
```

Run:

```bash
uv run python fix_gopro_battery_dies_video.py \
  --source-directory /path/to/raw_videos \
  --target-directory /path/to/fixed_videos
```

The script checks the last video in each camera folder. If it is truncated, it repairs it with [untrunc](https://github.com/anthwlock/untrunc) and writes the fixed file into the target directory, preserving the camera subfolder layout.

## 2. Concatenate Raw Video Segments by Timecode

Use:

```text
concat_videos.py
```

This script expects a source directory containing one subdirectory per camera. Each camera subdirectory should contain the raw video chunks for that camera:

```text
raw_videos/
  cam01/
    GX010001.MP4
    GX020001.MP4
  cam02/
    GX010002.MP4
    GX020002.MP4
```

Run:

```bash
uv run python concat_videos.py \
  --source-directory /path/to/raw_videos \
  --target-directory /path/to/concatenated_videos
```

For each camera folder, the script:

1. Finds `*.mp4` and `*.MP4` files.
2. Sorts them by filename.
3. Reads the embedded timecode from the first video with `VideoTimecode.from_video()`.
4. Writes a temporary ffmpeg concat list.
5. Runs ffmpeg concat with stream copy:

```text
-f concat -safe 0 -i <concat_list> -c copy -timecode <first_video_timecode>
```

The output is one file per camera:

```text
concatenated_videos/
  cam01.mp4
  cam02.mp4
```

The resulting concatenated file keeps the first raw segment's embedded timecode as the output timecode.

This step is now independent of segment cutting and frame splitting below — both of those work directly on the raw per-camera chunks, so only run this if you want a single continuous file per camera for other purposes (e.g. playback, archiving).

## 3. Cut Video Segments From a Specific Interval

Use:

```text
extract_segment_from_video.py
```

This script works directly on raw per-camera video chunks — the same source layout as `concat_videos.py` above, **not** its concatenated output:

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
uv run python extract_segment_from_video.py \
  --source-directory /path/to/raw_videos \
  --target-directory /path/to/output_segments \
  --start-time 00:01:00 \
  --end-time 00:02:00
```

With no `--use-timecode`, `--start-time` and `--end-time` are interpreted as offsets from the first frame of the first video in each camera folder.

Cut an absolute timecode interval:

```bash
uv run python extract_segment_from_video.py \
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
4. Concatenates the parts into a single output file per camera.

The output is one file per camera, named after the camera folder:

```text
output_segments/
  cam01.mp4
  cam02.mp4
```

### Process One Camera Only

Set `CAMERA_TO_PROCESS` to the camera folder name:

```bash
CAMERA_TO_PROCESS=cam02 uv run python extract_segment_from_video.py \
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
uv run python cut_gopro_annotation_segments.py \
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
uv run python batch_extract_segments.py \
  --source-directory /path/to/raw_videos \
  --target-directory /path/to/output_segments \
  --segment-set both \
  --use-timecode
```

Every segment in a run is written to the same `--target-directory`, and each camera's output file is always named `<camera>.mp4`, so **only the last segment of a run survives** in that directory. Use a separate `--target-directory` per segment (or per small batch) if you need to keep more than one.

## 4. Split Video Into Frame Images

Use:

```text
split_video_into_frames.py
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
uv run python split_video_into_frames.py \
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
uv run python split_video_into_frames.py \
  --source-directory /path/to/raw_videos \
  --target-directory /path/to/frames \
  --every-n-frames 10
```

This keeps every 10th frame of the requested interval; the sampling grid stays anchored to the start of the interval even across a chunk-file boundary.

`CAMERA_TO_PROCESS` (see above) also limits this script to a single camera.

## 5. Camera Calibration Pipeline

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
uv run python generate_calib_project.py /path/to/calib_images \
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
uv run python convert_camera_calibration.py \
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
uv run python convert_camera_calibration.py \
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
uv run python convert_idiap_to_calibrator.py \
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

## Minimal Files Needed

If you only want to keep the concatenation, segment-cutting, and frame-splitting functions above, the required runtime files are:

```text
video_postprocess/
  __init__.py
  concat_videos.py
  extract_segment_from_video.py
  split_video_into_frames.py
  video_segments.py
  timecode.py
  utils.py
  video_utils.py
```

Keep these optional wrappers only if you use them:

```text
video_postprocess/
  cut_gopro_annotation_segments.py   # fixed 30-second GoPro annotation cuts
  batch_extract_segments.py          # predefined segment lists TIME_SEGS_1 / TIME_SEGS_2
  py.typed                           # optional typing marker, not needed at runtime
```

Everything else in `video_postprocess/` is unrelated to concatenating videos, cutting video intervals, or splitting videos into frames, and can be removed for this reduced workflow.

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

Required uv run python packages for video concatenation, segment cutting, and frame splitting:

```text
av
click
numpy
opencv-python
pydantic
python-ffmpeg
tqdm
winfiletime
```

Additional uv run python packages for camera calibration utilities:

```text
PyYAML
scipy
```

Required system tools:

```text
ffmpeg
ffprobe
```

Note: the scripts import modules as `video_postprocess.<module>`. Keep the package import path configured the same way as the current project does, or update those imports to match the directory name if you run the folder standalone.
