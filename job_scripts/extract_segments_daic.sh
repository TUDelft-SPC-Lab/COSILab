#!/bin/bash
#SBATCH --job-name=vp-extract-segments
#SBATCH --partition=insy,general
#SBATCH --qos=medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END
#SBATCH --output=/home/nfs/zli33/slurm_outputs/video_postprocess/extract_segments_%j.out
#SBATCH --error=/home/nfs/zli33/slurm_outputs/video_postprocess/extract_segments_%j.err

# DAIC: cut 30 s clips out of the raw GoPro chunks, inside the video_postprocess
# Apptainer image. CPU only - the work is ffmpeg stream-copy plus a re-encode of
# the partial GOPs at each cut edge.
#
# Two modes:
#   --mode annotation   (default) cut_gopro_annotation_segments.py - walks
#                       13:45:00-14:20:00 for the 06-10 group and
#                       14:52:00-15:27:00 for the 01-05 group in 30 s steps,
#                       always by embedded timecode, one seg_<HHMMSS>_<HHMMSS>/
#                       output directory per clip.
#   --mode batch        batch_extract_segments.py - the hardcoded TIME_SEGS_1 /
#                       TIME_SEGS_2 lists, all clips into one flat directory.
#
# Arguments (annotation mode), all optional - the defaults below are the
# cosilab_project locations, so plain `sbatch job_scripts/extract_segments_daic.sh`
# cuts both groups:
#   -g6 <dir>   source directory for cameras 06-10 (default: mingle_session_1)
#   -g5 <dir>   source directory for cameras 01-05 (default: mingle_session_2)
#               pass `none` to either one to skip that group
#   -t <dir>    target directory
#
# Arguments (batch mode):
#   -s <dir>    source directory
#   -t <dir>    target directory
#   --segment-set <1|2|both>
#   --use-timecode
#
# Common optional arguments:
#   -c <name>   process only this camera subdirectory (CAMERA_TO_PROCESS)
#   --sif <path> / --repo <dir> / --baked-code
#
# Each source directory holds one subdirectory per camera, and each of those the
# raw GH*.MP4 / GX*.MP4 chunks:
#   top_view_camera-mingle_session_1/
#     camera_06-mingle_session_1/GH010123.MP4, GH020123.MP4, ...
#     camera_07-mingle_session_1/...
# Output clips are named after the camera directory, e.g.
#   <target>/seg_134500_134530/camera_06-mingle_session_1.mp4
#
# Examples:
#   # both groups, default source and target paths
#   sbatch job_scripts/extract_segments_daic.sh
#
#   # only the 06-10 group, one camera, into a scratch target
#   sbatch job_scripts/extract_segments_daic.sh -g5 none \
#     -c camera_06-mingle_session_1 \
#     -t .../data_temp/video_clips_30s_test
#
#   # non-default sources
#   sbatch job_scripts/extract_segments_daic.sh \
#     -g6 .../top_view_camera-mingle_session_1 \
#     -g5 .../top_view_camera-mingle_session_2

set -euo pipefail

NEON=/tudelft.net/staff-umbrella/neon
COSILAB=${NEON}/cosilab_project

SIF="${SIF:-${NEON}/apptainer/video_postprocess.sif}"
REPO_DIR="${REPO_DIR:-/home/nfs/zli33/projects/COSILab}"

GOPRO_DATA=${COSILAB}/raw_sensor_data/gopro_data

MODE="annotation"
GROUP_06_10="${GOPRO_DATA}/top_view_camera-mingle_session_1"
GROUP_01_05="${GOPRO_DATA}/top_view_camera-mingle_session_2"
SOURCE_DIR=""
TARGET_DIR="${COSILAB}/data_temp/video_clips_30s_timestamp_fixed"
SEGMENT_SET="both"
USE_TIMECODE=0
CAMERA=""
USE_BAKED_CODE=0

usage() {
  echo "Usage: sbatch job_scripts/extract_segments_daic.sh [--mode annotation] [-g6 <dir>|none] [-g5 <dir>|none] [-t <dir>] [-c <camera>]"
  echo "       sbatch job_scripts/extract_segments_daic.sh --mode batch -s <dir> -t <dir> [--segment-set 1|2|both] [--use-timecode] [-c <camera>]"
  echo "       optional everywhere: --sif <path> --repo <dir> --baked-code"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    -g6|--group-06-10-directory)
      GROUP_06_10="${2:-}"
      shift 2
      ;;
    -g5|--group-01-05-directory)
      GROUP_01_05="${2:-}"
      shift 2
      ;;
    -s|--source|--source-directory)
      SOURCE_DIR="${2:-}"
      shift 2
      ;;
    -t|--target|--target-directory)
      TARGET_DIR="${2:-}"
      shift 2
      ;;
    --segment-set)
      SEGMENT_SET="${2:-}"
      shift 2
      ;;
    --use-timecode)
      USE_TIMECODE=1
      shift
      ;;
    -c|--camera)
      CAMERA="${2:-}"
      shift 2
      ;;
    --sif)
      SIF="${2:-}"
      shift 2
      ;;
    --repo)
      REPO_DIR="${2:-}"
      shift 2
      ;;
    --baked-code)
      USE_BAKED_CODE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

# `-g6 none` / `-g5 none` opts a group out of the defaults above.
if [[ "${GROUP_06_10,,}" == "none" ]]; then
  GROUP_06_10=""
fi
if [[ "${GROUP_01_05,,}" == "none" ]]; then
  GROUP_01_05=""
fi

case "$MODE" in
  annotation|batch) ;;
  *)
    echo "[ERROR] --mode must be 'annotation' or 'batch' (got: $MODE)" >&2
    exit 2
    ;;
esac

if [ ! -f "$SIF" ]; then
  echo "[ERROR] Apptainer image not found: $SIF" >&2
  echo "        Build it from the repo root with:" >&2
  echo "          cd $REPO_DIR && apptainer build $SIF video_postprocess/apptainer/video_postprocess.def" >&2
  exit 2
fi

check_source() {
  local label="$1" dir="$2"
  if [ ! -d "$dir" ]; then
    echo "[ERROR] $label directory not found: $dir" >&2
    exit 2
  fi
  # Fail early on the common layout mistake: pointing at a camera directory
  # instead of the parent that holds one subdirectory per camera.
  local n_cams
  n_cams=$(find "$dir" -mindepth 1 -maxdepth 1 -type d | wc -l)
  if [ "$n_cams" -eq 0 ]; then
    echo "[ERROR] $label has no camera subdirectories: $dir" >&2
    echo "        Expected <dir>/<camera_xx-...>/GH*.MP4" >&2
    exit 2
  fi
  local n_videos
  n_videos=$(find "$dir" -mindepth 2 -maxdepth 2 \( -name '*.MP4' -o -name '*.mp4' \) | wc -l)
  if [ "$n_videos" -eq 0 ]; then
    echo "[ERROR] $label has camera subdirectories but no .MP4/.mp4 chunks inside: $dir" >&2
    exit 2
  fi
  echo "[INFO] $label: $dir ($n_cams camera dirs, $n_videos video chunks)"
}

if [ "$MODE" = "annotation" ]; then
  if [[ -z "$GROUP_06_10" && -z "$GROUP_01_05" ]]; then
    echo "[ERROR] Both groups are disabled; leave -g6 / -g5 unset to use the defaults." >&2
    usage >&2
    exit 2
  fi
  [ -n "$GROUP_06_10" ] && check_source "group 06-10" "$GROUP_06_10"
  [ -n "$GROUP_01_05" ] && check_source "group 01-05" "$GROUP_01_05"
else
  if [ -z "$SOURCE_DIR" ]; then
    echo "[ERROR] batch mode needs -s <source-dir>." >&2
    usage >&2
    exit 2
  fi
  case "$SEGMENT_SET" in
    1|2|both) ;;
    *)
      echo "[ERROR] --segment-set must be one of: 1, 2, both (got: $SEGMENT_SET)" >&2
      exit 2
      ;;
  esac
  check_source "source" "$SOURCE_DIR"
fi

if [ -z "$TARGET_DIR" ]; then
  echo "[ERROR] -t <target-dir> is required." >&2
  exit 2
fi
mkdir -p "$TARGET_DIR"

# ffmpeg writes the intermediate head/mid/tail parts of every cut to TMPDIR.
# Keep that on the node's local disk rather than the umbrella share.
JOB_TMP="/tmp/video_postprocess_${SLURM_JOB_ID:-$$}"
mkdir -p "$JOB_TMP"
trap 'rm -rf "$JOB_TMP"' EXIT

env_args=(--env "CAMERA_TO_PROCESS=$CAMERA" --env "TMPDIR=$JOB_TMP")

# Run the checkout's code by default so edits take effect without a rebuild; the
# image's own copy is the fallback.
if [ "$USE_BAKED_CODE" = "1" ]; then
  echo "[INFO] Using the code baked into $SIF"
elif [ -d "$REPO_DIR/video_postprocess/src" ]; then
  echo "[INFO] Using code from $REPO_DIR/video_postprocess/src"
  env_args+=(--env "PYTHONPATH=$REPO_DIR/video_postprocess/src")
else
  echo "[WARN] $REPO_DIR/video_postprocess/src not found; falling back to the code baked into the image."
fi

if [ "$MODE" = "annotation" ]; then
  cmd=(python -m video_postprocess.cut_gopro_annotation_segments
       --target-directory "$TARGET_DIR")
  [ -n "$GROUP_06_10" ] && cmd+=(--group-06-10-directory "$GROUP_06_10")
  [ -n "$GROUP_01_05" ] && cmd+=(--group-01-05-directory "$GROUP_01_05")
else
  cmd=(python -m video_postprocess.batch_extract_segments
       --source-directory "$SOURCE_DIR"
       --target-directory "$TARGET_DIR"
       --segment-set "$SEGMENT_SET")
  if [ "$USE_TIMECODE" = "1" ]; then
    cmd+=(--use-timecode)
  else
    cmd+=(--no-use-timecode)
  fi
fi

echo "[INFO] mode=$MODE"
echo "[INFO] sif=$SIF"
echo "[INFO] target=$TARGET_DIR"
echo "[INFO] camera=${CAMERA:-<all>}"
echo "[INFO] tmpdir=$JOB_TMP"
echo "[INFO] command: ${cmd[*]}"
echo ""

apptainer exec \
  --containall \
  "${env_args[@]}" \
  -B "$HOME:$HOME" \
  -B /tudelft.net/:/tudelft.net/ \
  -B /tmp:/tmp \
  "$SIF" \
  "${cmd[@]}"

echo ""
echo "[INFO] Done. Clips written to $TARGET_DIR"
