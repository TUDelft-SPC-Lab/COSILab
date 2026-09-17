#!/bin/bash

# The data was downloaded from
# sftp.tudelft.nl/staff-umbrella/neon/ingroup_dataset/processed
# and stored in /home/era/data/processed
# If you want to reproduce the results, replace the paths below accordingly

project_dir="$(dirname "$0")/.."

export FOLDER_TO_PROCESS="side_view_camera-mingle_session_1"

uv run --project "$project_dir" python -m video_postprocess.extract_frames_from_video \
    --source-directory "/home/era/data/processed/gopro_data/timecode_fixed/$FOLDER_TO_PROCESS" \
    --target-directory "/home/era/data/processed/gopro_data/calib_frames/$FOLDER_TO_PROCESS" \
    --start-time "13:25:20" \
    --end-time "13:29:46" \
    --use-timecode \
    --every-n-frames "60"

#############
export FOLDER_TO_PROCESS="top_view_camera-mingle_session_1"

uv run --project "$project_dir" python -m video_postprocess.extract_frames_from_video \
    --source-directory "/home/era/data/processed/gopro_data/timecode_fixed/$FOLDER_TO_PROCESS" \
    --target-directory "/home/era/data/processed/gopro_data/calib_frames/$FOLDER_TO_PROCESS" \
    --start-time "13:25:20" \
    --end-time "13:29:46" \
    --use-timecode \
    --every-n-frames "60"

#############
#############
export FOLDER_TO_PROCESS="side_view_camera-mingle_session_2"

uv run --project "$project_dir" python -m video_postprocess.extract_frames_from_video \
    --source-directory "/home/era/data/processed/gopro_data/timecode_fixed/$FOLDER_TO_PROCESS" \
    --target-directory "/home/era/data/processed/gopro_data/calib_frames/$FOLDER_TO_PROCESS" \
    --start-time "14:47:14" \
    --end-time "14:50:00" \
    --use-timecode \
    --every-n-frames "30"

#############
export FOLDER_TO_PROCESS="top_view_camera-mingle_session_2"

uv run --project "$project_dir" python -m video_postprocess.extract_frames_from_video \
    --source-directory "/home/era/data/processed/gopro_data/timecode_fixed/$FOLDER_TO_PROCESS" \
    --target-directory "/home/era/data/processed/gopro_data/calib_frames/$FOLDER_TO_PROCESS" \
    --start-time "14:47:14" \
    --end-time "14:50:00" \
    --use-timecode \
    --every-n-frames "30"
