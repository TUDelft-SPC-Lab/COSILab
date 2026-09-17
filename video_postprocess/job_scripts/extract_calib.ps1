$RawDataDir = "E:\Era\INGroup\raw_sensor_data\gopro_data"
$CalibFramesDir = "E:\Era\INGroup\processed_data\gopro_data\calib_frames"

$ProjectDir = Join-Path $PSScriptRoot ".."

$env:FOLDER_TO_PROCESS = "side_view_camera-mingle_session_1"

uv run --project $ProjectDir python -m video_postprocess.extract_frames_from_video `
    --source-directory "$RawDataDir\$env:FOLDER_TO_PROCESS" `
    --target-directory "$CalibFramesDir\$env:FOLDER_TO_PROCESS" `
    --start-time "13:25:20" `
    --end-time "13:29:46" `
    --use-timecode `
    --every-n-frames "60"

#############
$env:FOLDER_TO_PROCESS = "top_view_camera-mingle_session_1"

uv run --project $ProjectDir python -m video_postprocess.extract_frames_from_video `
    --source-directory "$RawDataDir\$env:FOLDER_TO_PROCESS" `
    --target-directory "$CalibFramesDir\$env:FOLDER_TO_PROCESS" `
    --start-time "13:25:20" `
    --end-time "13:29:46" `
    --use-timecode `
    --every-n-frames "60"

#############
#############
$env:FOLDER_TO_PROCESS = "side_view_camera-mingle_session_2"

uv run --project $ProjectDir python -m video_postprocess.extract_frames_from_video `
    --source-directory "$RawDataDir\$env:FOLDER_TO_PROCESS" `
    --target-directory "$CalibFramesDir\$env:FOLDER_TO_PROCESS" `
    --start-time "14:47:14" `
    --end-time "14:50:00" `
    --use-timecode `
    --every-n-frames "30"

#############
$env:FOLDER_TO_PROCESS = "top_view_camera-mingle_session_2"

uv run --project $ProjectDir python -m video_postprocess.extract_frames_from_video `
    --source-directory "$RawDataDir\$env:FOLDER_TO_PROCESS" `
    --target-directory "$CalibFramesDir\$env:FOLDER_TO_PROCESS" `
    --start-time "14:47:14" `
    --end-time "14:50:00" `
    --use-timecode `
    --every-n-frames "30"
