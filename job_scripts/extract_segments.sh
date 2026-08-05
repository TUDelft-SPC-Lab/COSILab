#!/bin/bash
#SBATCH --job-name="vp_extract_segments"
#SBATCH --partition=compute
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4000M
#SBATCH --mail-type=END
#SBATCH --account=research-eemcs-insy
#SBATCH --output=/scratch/zli33/slurm_outputs/video_postprocess/slurm_%j.out
#SBATCH --error=/scratch/zli33/slurm_outputs/video_postprocess/slurm_%j.err

# Batch segment extraction with video_postprocess/batch_extract_segments.py,
# inside the video_postprocess Apptainer image.
#
# Required args:
#   -s <dir>    source directory (one subdirectory per camera, holding the raw chunks)
#   -t <dir>    target directory for the extracted segments
#
# Optional args:
#   --segment-set <1|2|both>   which hardcoded time-segment set to run (default: both)
#   --use-timecode             interpret the times as embedded video timecodes
#   -c <name>                  process only this camera subdirectory (CAMERA_TO_PROCESS)
#   --sif <path>               override the image path
#   --repo <dir>               COSILab checkout to run the code from (default: $HOME/projects/COSILab)
#   --baked-code               ignore --repo and use the code baked into the image
#
# Examples:
#   sbatch job_scripts/extract_segments.sh \
#     -s /scratch/zli33/data/cosilab/concatenated_videos \
#     -t /scratch/zli33/data/cosilab/segments
#
#   sbatch job_scripts/extract_segments.sh \
#     -s /scratch/zli33/data/cosilab/raw_videos \
#     -t /scratch/zli33/data/cosilab/segments \
#     --segment-set 1 --use-timecode -c cam01

set -euo pipefail

home_path=/home/zli33
scratch_path=/scratch/zli33

sif_path=$scratch_path/apptainers/video_postprocess.sif
repo_dir=$home_path/projects/COSILab

SOURCE_DIR=""
TARGET_DIR=""
SEGMENT_SET="both"
USE_TIMECODE=0
CAMERA=""
USE_BAKED_CODE=0

usage() {
  echo "Usage: sbatch job_scripts/extract_segments.sh -s <source-dir> -t <target-dir>" \
       "[--segment-set 1|2|both] [--use-timecode] [-c <camera>] [--sif <path>] [--repo <dir>] [--baked-code]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
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
      sif_path="${2:-}"
      shift 2
      ;;
    --repo)
      repo_dir="${2:-}"
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

if [[ -z "$SOURCE_DIR" || -z "$TARGET_DIR" ]]; then
  echo "[ERROR] Both -s and -t are required." >&2
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

if [ ! -f "$sif_path" ]; then
  echo "[ERROR] Apptainer image not found: $sif_path" >&2
  echo "        Build it with:" >&2
  echo "          cd $repo_dir && apptainer build $sif_path video_postprocess/apptainer/video_postprocess.def" >&2
  exit 2
fi

if [ ! -d "$SOURCE_DIR" ]; then
  echo "[ERROR] Source directory not found: $SOURCE_DIR" >&2
  exit 2
fi

# Resolve to absolute paths: they are passed straight through to the container,
# which binds them at the identical location.
SOURCE_DIR="$(readlink -f "$SOURCE_DIR")"
mkdir -p "$TARGET_DIR"
TARGET_DIR="$(readlink -f "$TARGET_DIR")"

bind_args=(--bind "$SOURCE_DIR:$SOURCE_DIR" --bind "$TARGET_DIR:$TARGET_DIR")
env_args=(--env "CAMERA_TO_PROCESS=$CAMERA")

# Run the checkout's code by default so edits take effect without a rebuild; the
# image's own copy is the fallback.
if [ "$USE_BAKED_CODE" = "1" ]; then
  echo "[INFO] Using the code baked into $sif_path"
elif [ -d "$repo_dir/video_postprocess/src" ]; then
  echo "[INFO] Using code from $repo_dir/video_postprocess/src"
  bind_args+=(--bind "$repo_dir:$repo_dir")
  env_args+=(--env "PYTHONPATH=$repo_dir/video_postprocess/src")
else
  echo "[WARN] $repo_dir/video_postprocess/src not found; falling back to the code baked into the image."
fi

cmd=(python -m video_postprocess.batch_extract_segments
     --source-directory "$SOURCE_DIR"
     --target-directory "$TARGET_DIR"
     --segment-set "$SEGMENT_SET")

if [ "$USE_TIMECODE" = "1" ]; then
  cmd+=(--use-timecode)
else
  cmd+=(--no-use-timecode)
fi

echo "[INFO] sif=$sif_path"
echo "[INFO] source=$SOURCE_DIR"
echo "[INFO] target=$TARGET_DIR"
echo "[INFO] segment-set=$SEGMENT_SET use-timecode=$USE_TIMECODE camera=${CAMERA:-<all>}"
echo "[INFO] command: ${cmd[*]}"
echo ""

apptainer exec "${bind_args[@]}" "${env_args[@]}" "$sif_path" "${cmd[@]}"

echo ""
echo "[INFO] Done. Segments written to $TARGET_DIR"
