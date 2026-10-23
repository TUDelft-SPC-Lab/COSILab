"""Find where each per-camera session video starts in that camera's raw GoPro files.

The session videos (`mingling_session_videos/camXX.mp4`) used for the keypoint pipeline are gap-free runs of
consecutive raw GoPro frames, continuing across file boundaries, extracted without any 59.94 fps correction. Their
embedded timecodes are not reliable, so the start is measured by matching pixels: the first frame of segment n (`camXX_segNNN.mp4`, 600 frames
each) must equal raw frame `raw_start_frame + (n - 1) * 600`, where frames are counted globally across the raw files
of the camera in playback order.

The first segment is searched in a wide window around a drift-corrected guess. A spread of other segments is then
checked in a narrow window, which fails if the session video is not a gap-free run of raw frames.
"""

from __future__ import annotations

import csv
import re
import subprocess
from pathlib import Path

import click
import numpy as np
from tqdm import tqdm

from video_postprocess.video_segments import (
    FPS_60,
    VideoFileInfo,
    collect_camera_video_infos,
    correct_for_fps_59_94,
    parse_time_str_to_frames,
)

FRAMES_PER_SEGMENT = 600
# Frames are compared as small grayscale thumbnails, which is enough to tell neighbouring frames apart
THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT = 192, 108
# A real match only differs by re-encoding noise; anything above this is treated as no match
MAX_MATCH_DIFFERENCE = 2.0

CSV_COLUMNS = ["camera", "session", "raw_start_file", "raw_start_local_frame", "raw_start_frame", "session_frames"]


def decode_thumbnails(video_path: Path, start_frame: int, count: int, framerate: float) -> tuple[int, np.ndarray]:
    """Decode `count` frames from `start_frame` on as grayscale thumbnails. Returns the index of the first frame
    actually decoded (from its pts, so seeking inaccuracies cannot go unnoticed) and the frames."""
    # fmt: off
    cmd = [
        "ffmpeg", "-v", "info", "-nostats",
        "-ss", f"{start_frame / framerate:.6f}", "-copyts",
        "-i", str(video_path),
        "-frames:v", str(count),
        "-vf", f"scale={THUMBNAIL_WIDTH}:{THUMBNAIL_HEIGHT},format=gray,showinfo",
        "-f", "rawvideo", "-",
    ]
    # fmt: on
    result = subprocess.run(cmd, capture_output=True, check=True)
    frames = np.frombuffer(result.stdout, np.uint8).reshape(-1, THUMBNAIL_HEIGHT, THUMBNAIL_WIDTH)
    first_pts_time = float(re.search(r"pts_time:([\d.]+)", result.stderr.decode()).group(1))
    return round(first_pts_time * framerate), frames.astype(np.float32)


def decode_raw_window(infos: list[VideoFileInfo], start_frame: int, count: int) -> np.ndarray:
    """Decode global raw frames [start_frame, start_frame + count), spanning file boundaries if needed."""
    pieces = []
    for info in infos:
        file_start = max(start_frame - info.physical_frames_before, 0)
        file_end = min(start_frame + count - info.physical_frames_before, info.frame_count)
        if file_start >= file_end:
            continue
        first, frames = decode_thumbnails(info.path, file_start, file_end - file_start, float(info.framerate))
        if first != file_start or len(frames) != file_end - file_start:
            error_msg = f"Asked for frames {file_start}-{file_end} of {info.path}, got {len(frames)} from {first}"
            raise RuntimeError(error_msg)
        pieces.append(frames)
    return np.concatenate(pieces)


def find_best_match(
    infos: list[VideoFileInfo], segment_path: Path, predicted_frame: int, search: int
) -> tuple[int, np.ndarray]:
    """Find the global raw frame closest to the first frame of `segment_path`, within `search` frames of
    `predicted_frame`. Returns that frame and the mean absolute difference for every frame in the window."""
    total_frames = infos[-1].physical_frames_before + infos[-1].frame_count
    window_start = max(predicted_frame - search, 0)
    window_end = min(predicted_frame + search + 1, total_frames)

    _, segment_frame = decode_thumbnails(segment_path, 0, 1, FPS_60)
    raw_frames = decode_raw_window(infos, window_start, window_end - window_start)
    differences = np.abs(raw_frames - segment_frame[0]).mean(axis=(1, 2))

    best = int(differences.argmin())
    if differences[best] > MAX_MATCH_DIFFERENCE:
        error_msg = (
            f"No raw frame matches {segment_path.name} within {search} frames of {predicted_frame}, "
            f"closest mean difference is {differences[best]:.2f}"
        )
        raise RuntimeError(error_msg)
    return window_start + best, differences


def get_session_video_frame_count(video_path: Path) -> int:
    # fmt: off
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    # fmt: on
    return int(subprocess.run(cmd, capture_output=True, check=True, text=True).stdout)


def match_camera(
    raw_camera_directory: Path,
    segment_paths: list[Path],
    session_start: str,
    first_search: int,
    check_search: int,
    num_checks: int,
) -> tuple[list[VideoFileInfo], int]:
    infos = collect_camera_video_infos(raw_camera_directory)

    # Drift-corrected guess of where the session start label falls in the raw files
    nominal_offset = parse_time_str_to_frames(session_start, FPS_60) - infos[0].nominal_start_frame
    guess = correct_for_fps_59_94(nominal_offset, 0, infos[0].framerate)

    raw_start_frame, differences = find_best_match(infos, segment_paths[0], guess, first_search)
    print(
        f"  {segment_paths[0].name}: raw frame {raw_start_frame} ({raw_start_frame - guess:+d} from guess), "
        f"mean difference {differences.min():.2f}, window median {np.median(differences):.2f}"
    )

    check_indices = np.unique(np.linspace(1, len(segment_paths) - 1, num_checks).round().astype(int))
    for index in tqdm(check_indices, desc="Checking segments", unit="Segment", leave=False):
        predicted = raw_start_frame + index * FRAMES_PER_SEGMENT
        best, differences = find_best_match(infos, segment_paths[index], predicted, check_search)
        print(
            f"  {segment_paths[index].name}: raw frame {best} ({best - predicted:+d} from prediction), "
            f"mean difference {differences.min():.2f}, window median {np.median(differences):.2f}"
        )
        if best != predicted:
            error_msg = (
                f"{segment_paths[index].name} starts at raw frame {best}, expected {predicted}: "
                "the session video is not a gap-free run of raw frames"
            )
            raise RuntimeError(error_msg)

    return infos, raw_start_frame


def write_rows(output_csv: Path, rows: list[dict]) -> None:
    """Write `rows` to `output_csv`, replacing earlier rows for the same cameras and keeping the others."""
    existing = []
    if output_csv.exists():
        with output_csv.open(newline="") as f:
            existing = list(csv.DictReader(f))
    cameras = {row["camera"] for row in rows}
    merged = [row for row in existing if row["camera"] not in cameras] + rows

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(sorted(merged, key=lambda row: row["camera"]))


@click.command()
@click.option(
    "--raw-directory",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Raw GoPro session directory, e.g. raw_sensor_data/gopro_data/top_view_camera-mingle_session_2",
    required=True,
)
@click.option(
    "--segments-directory",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Directory with one camXX/ subdirectory of camXX_segNNN.mp4 segments per camera",
    required=True,
)
@click.option(
    "--session-videos-directory",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Directory with the camXX.mp4 session videos the segments were cut from",
    required=True,
)
@click.option("--session-start", help="Time label of the first session video frame, HH:MM:SS:FF", required=True)
@click.option("--cameras", help="Comma-separated camera numbers, e.g. 01,03,05", required=True)
@click.option(
    "--output-csv",
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help="CSV to write; rows for other cameras already in it are kept",
    required=True,
)
@click.option("--first-search", default=1200, show_default=True, help="Search window in frames for the first segment")
@click.option("--check-search", default=60, show_default=True, help="Search window in frames for the checked segments")
@click.option("--num-checks", default=5, show_default=True, help="Number of further segments to check")
def main(
    raw_directory: Path,
    segments_directory: Path,
    session_videos_directory: Path,
    session_start: str,
    cameras: str,
    output_csv: Path,
    first_search: int,
    check_search: int,
    num_checks: int,
) -> None:
    session = raw_directory.name.split("-", maxsplit=1)[1]  # top_view_camera-mingle_session_2
    rows = []
    for camera in cameras.split(","):
        camera = camera.strip().zfill(2)
        print(f"cam{camera}")
        segment_paths = sorted((segments_directory / f"cam{camera}").glob(f"cam{camera}_seg*.mp4"))
        session_frames = get_session_video_frame_count(session_videos_directory / f"cam{camera}.mp4")
        if len(segment_paths) != -(-session_frames // FRAMES_PER_SEGMENT):
            error_msg = f"cam{camera}: {len(segment_paths)} segments do not cover {session_frames} session frames"
            raise RuntimeError(error_msg)

        infos, raw_start_frame = match_camera(
            raw_directory / f"camera_{camera}-{session}",
            segment_paths,
            session_start,
            first_search,
            check_search,
            num_checks,
        )

        total_raw_frames = infos[-1].physical_frames_before + infos[-1].frame_count
        if raw_start_frame + session_frames > total_raw_frames:
            error_msg = f"cam{camera}: session video runs past the end of the readable raw files"
            raise RuntimeError(error_msg)

        start_info = next(
            info
            for info in infos
            if info.physical_frames_before <= raw_start_frame < info.physical_frames_before + info.frame_count
        )
        rows.append(
            {
                "camera": f"camera_{camera}",
                "session": session,
                "raw_start_file": start_info.path.name,
                "raw_start_local_frame": raw_start_frame - start_info.physical_frames_before,
                "raw_start_frame": raw_start_frame,
                "session_frames": session_frames,
            }
        )

    write_rows(output_csv, rows)
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
