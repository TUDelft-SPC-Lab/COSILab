#!/bin/bash
# Joint multi-board recalibration of one mingle session from dense world-synced frames (see README section 6,
# "Joint recalibration with every board"). Everything is written under video_postprocess/side_calib_work/
# (gitignored); /tudelft.net is only read.
#
# Usage: bash job_scripts/side_calib.sh SESSION        (SESSION = 1 or 2)
# Runtime on 32 cores: about 3-4 h for session 1, 2-3 h for session 2, mostly board detection.
set -euo pipefail
S=$1
case $S in
  1) TOP="6 7 8 9 10" ;;
  2) TOP="1 2 3 4" ;;   # cam05: bad time sync, never used
  *) echo "session must be 1 or 2" >&2; exit 1 ;;
esac
SIDE="12 13 14 15"
cd "$(dirname "$0")/../src/video_postprocess/camera_calibration/side_calib"
export OMP_NUM_THREADS=1 OPENCV_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
W=../../../../side_calib_work
mkdir -p "$W/frames/s$S"

# 1. Old calibration frames (used next to the dense ones), copied once from the read-only share.
CAL=/tudelft.net/staff-umbrella/neon/cosilab_project/processed_data/gopro_data/side_view_camera_calibration
rsync -a "$CAL/calib_frames/side_view_camera-mingle_session_$S/" "$W/frames/s$S/"
rsync -a "$CAL/calib_frames/top_view_camera-mingle_session_$S/" "$W/frames/s$S/"
[ -f "$W/detections_s$S.npz" ] || python3 detect_multi.py "$S" --workers 30 | tee "$W/detect_s$S.log"

# 2. Side-camera timestamps from the raw timecode tags (the 4TU rule); the check must report 0 differing rows.
python3 timestamps.py "$S" --check $TOP --cams $SIDE | tee "$W/timestamps_s$S.log"

# 3. Dense frames: every 10th world tick of the calibration window, 3 cameras at a time (memory).
printf "%s\n" $TOP $SIDE | xargs -P 3 -I{} sh -c \
  "python3 extract_dense.py --session $S --cams {} --step 10 > $W/extract_s${S}dense_cam{}.log 2>&1"
tail -qn 1 "$W"/extract_s${S}dense_cam*.log

# 4. Board detection on the dense frames (all boards per image).
[ -f "$W/detections_s${S}dense.npz" ] || \
  python3 detect_multi.py "$S" --subdir "s${S}dense" --workers 28 | tee "$W/detect_s${S}dense.log"

# 5. Joint fit (origin camera fixed, all others free) and 5-fold held-out evaluation.
python3 joint.py "$S" | tee "$W/joint_s$S.log"
