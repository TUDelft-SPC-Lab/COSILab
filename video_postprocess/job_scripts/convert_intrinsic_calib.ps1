$CalibDir = "E:\Era\INGroup\camera_intrinsic_frames\fisheye lens"
$OutputDir = "E:\Era\INGroup\new_clean\idiap_params\intrinsics"

$ProjectDir = Join-Path $PSScriptRoot ".."

# One parameters-camera-XX.json per camera, each holding a single fisheye camera
# Output is $OutputDir\camera_XX\intrinsic.json
Get-ChildItem -Path $CalibDir -Filter "parameters-camera-*.json" | Sort-Object Name | ForEach-Object {
    $CameraName = $_.BaseName -replace "^parameters-camera-", "camera_"
    Write-Host "Converting $($_.Name) -> $CameraName"

    uv run --project $ProjectDir python -m video_postprocess.convert_camera_calibration `
        --calibrator-file $_.FullName `
        --camera-format "IDIAP" `
        --output-directory $OutputDir `
        --cameras-names $CameraName

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to convert $($_.Name)"
        exit $LASTEXITCODE
    }
}
