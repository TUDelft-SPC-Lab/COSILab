import subprocess
from fractions import Fraction
from pathlib import Path


def get_video_duration_in_seconds(file_path: Path) -> float:
    # fmt: off
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]
    # fmt: on
    output = subprocess.run(
        cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    return float(output.stdout)


def get_video_framerate(video_path: Path) -> Fraction:
    # fmt: off
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v",
        "-of", "default=noprint_wrappers=1:nokey=1",
        "-show_entries", "stream=r_frame_rate",
        video_path,
    ]
    # fmt: on
    output = subprocess.run(
        cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    framerate_str = output.stdout.decode("utf-8").strip()
    numerator, denominator = map(int, framerate_str.split("/"))
    return Fraction(numerator, denominator)
