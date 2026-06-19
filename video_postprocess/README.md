# Video Postprocess

This folder contains utilities for GoPro-style video postprocessing. The two core functions documented here are:

1. Concatenate raw per-camera video chunks into one continuous video per camera.
2. Cut video segments from a requested interval, either by relative video time or by embedded video timecode.

The scripts use `ffmpeg` / `ffprobe` through Python, so both command-line tools must be available on `PATH`.

## 1. Fix Corrupted Videos

If a camera stopped recording because the GoPro battery died, repair the last chunk before doing any concatenation or segment cutting.

Use:

```text
fix_gopro_battery_dies_video.py
```

Run:

```bash
python fix_gopro_battery_dies_video.py \
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
python concat_videos.py \
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

The resulting concatenated file keeps the first raw segment's embedded timecode as the output timecode. This is what allows later segment extraction to use absolute clock-style timecode.

## 3. Cut Video Segments From a Specific Interval

Use:

```text
extract_segment_from_video.py
```

This script expects a directory containing concatenated camera videos:

```text
concatenated_videos/
  cam01.mp4
  cam02.mp4
  cam03.mp4
```

Cut a relative interval from every camera video:

```bash
python extract_segment_from_video.py \
  --source-directory /path/to/concatenated_videos \
  --target-directory /path/to/output_segments \
  --start-time 00:01:00 \
  --end-time 00:02:00
```

With no `--use-timecode`, `--start-time` and `--end-time` are interpreted as offsets from the beginning of each video.

Cut an absolute timecode interval:

```bash
python extract_segment_from_video.py \
  --source-directory /path/to/concatenated_videos \
  --target-directory /path/to/output_segments \
  --start-time 13:45:00 \
  --end-time 13:45:30 \
  --use-timecode
```

With `--use-timecode`, the script:

1. Reads each video's embedded start timecode.
2. Converts the requested start/end times to timedeltas.
3. Subtracts the embedded video start timecode from the requested times.
4. Passes the resulting relative interval to ffmpeg.
5. Writes the output with a new embedded timecode matching the cut segment start.

For example, if `cam01.mp4` starts at embedded timecode `13:45:00:00`, then:

```text
--start-time 13:46:00 --end-time 13:46:30 --use-timecode
```

cuts from relative `00:01:00` to `00:01:30`.

The output filename includes the requested start and end labels:

```text
cam01_13460000_13463000.mp4
```

The script uses stream copy (`-c copy`) and disables data streams (`-dn`) to avoid `tmcd` stream errors.

### Process One Camera Only

Set `CAMERA_TO_PROCESS` to the camera filename stem:

```bash
CAMERA_TO_PROCESS=cam02 python extract_segment_from_video.py \
  --source-directory /path/to/concatenated_videos \
  --target-directory /path/to/output_segments \
  --start-time 13:45:00 \
  --end-time 13:45:30 \
  --use-timecode
```

Only `concatenated_videos/cam02.mp4` will be processed.

### Fixed 30-Second Annotation Segments

For the current INGroup GoPro annotation split, `cut_gopro_annotation_segments.py` wraps `extract_segment_from_video.py` and cuts fixed 30-second timecode segments:

- Group cameras `06-10`: `13:45:00` to `14:20:00`
- Group cameras `01-05`: `14:52:00` to `15:27:00`

Run:

```bash
python cut_gopro_annotation_segments.py \
  --group-06-10-directory /path/to/group_06_10_concatenated \
  --group-01-05-directory /path/to/group_01_05_concatenated \
  --target-directory /path/to/annotation_segments
```

It creates folders like:

```text
annotation_segments/
  seg_134500_134530/
  seg_134530_134600/
```

and writes one cut video per camera into each segment folder.

## 4. Camera Calibration Pipeline

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
python generate_calib_project.py /path/to/calib_images \
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
python convert_camera_calibration.py \
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
python convert_camera_calibration.py \
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
python convert_idiap_to_calibrator.py \
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

If you only want to keep the two functions above, the required runtime files are:

```text
video_postprocess/
  __init__.py
  concat_videos.py
  extract_segment_from_video.py
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

Everything else in `video_postprocess/` is unrelated to concatenating videos or cutting video intervals and can be removed for this reduced workflow.

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

Required Python packages for video concatenation and segment cutting:

```text
click
numpy
pydantic
tqdm
python-ffmpeg
winfiletime
```

Additional Python packages for camera calibration utilities:

```text
opencv-python
PyYAML
scipy
```

Required system tools:

```text
ffmpeg
ffprobe
```

Note: the scripts import modules as `video_postprocess.<module>`. Keep the package import path configured the same way as the current project does, or update those imports to match the directory name if you run the folder standalone.
