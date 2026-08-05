from __future__ import annotations

import bisect
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import click
from tqdm import tqdm
import cv2
from datetime import timedelta
from video_postprocess.split_video_into_frames import (
    get_video_framerate,
)
from video_postprocess.timecode import VideoTimecode
from video_postprocess.utils import get_camera_to_process

# Nominal (non-drop) frame rate that start-time/end-time/timecodes are expressed against, matching
# split_video_into_frames.py. Actual 59.94 fps footage is corrected against this via correct_for_fps_59_94.
FPS_60 = 60

# Quality target for the re-encoded head/tail edges of a cut. CRF rather than a bit_rate, since these
# segments are only ever a few frames long, too short for bitrate-mode rate control to ramp up.
RE_ENCODE_CRF = "18"

def get_video_total_frame_num(video_path: Path) -> int:
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            error_msg = f"Could not open video '{video_path}'."
            raise OSError(error_msg)
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()

def parse_time_str_to_frames(time_str: str | None, framerate: Fraction | int | float) -> int | None:
    """Convert a time string in HH:MM:SS:FF, HH:MM:SS, MM:SS or SS format into a total frame count.

    Everything is computed directly in whole frames against the nominal (rounded) framerate, so no
    precision is lost by round-tripping through seconds or microseconds.
    """
    if time_str is None:
        return None

    fr = round(framerate)

    parts = time_str.split(":")
    if len(parts) == 4:
        hours, minutes, seconds, frames = map(int, parts)
    else:
        values = list(map(int, parts))
        if len(values) == 2:
            values = [0, *values]
        elif len(values) == 1:
            values = [0, 0, *values]
        elif len(values) != 3:
            error_msg = "Time string must be in HH:MM:SS:FF, HH:MM:SS, MM:SS or SS format."
            raise ValueError(error_msg)
        hours, minutes, seconds = values
        frames = 0

    return ((hours * 3600 + minutes * 60 + seconds) * fr) + frames

@dataclass(frozen=True)
class VideoFileInfo:
    path: Path
    framerate: Fraction | int | float
    frame_count: int
    nominal_start_frame: int
    physical_frames_before: int

    @property
    def nominal_end_frame(self) -> int:
        return self.nominal_start_frame + self.frame_count


@dataclass(frozen=True)
class FramePosition:
    file_index: int
    local_frame: int


def collect_camera_video_infos(camera_directory: Path) -> list[VideoFileInfo]:
    """Gather per-file metadata for every video in a camera directory, in playback order."""
    infos: list[VideoFileInfo] = []
    physical_frames_before = 0
    for video_path in sorted(camera_directory.glob("*.mp4")):
        try:
            framerate = get_video_framerate(video_path)
        except OSError:
            # e.g. a moov atom not found, as happens when a GoPro's battery dies mid-recording.
            continue
        frame_count = get_video_total_frame_num(video_path)
        timecode = VideoTimecode.from_video(video_path)
        infos.append(
            VideoFileInfo(
                path=video_path,
                framerate=framerate,
                frame_count=frame_count,
                nominal_start_frame=timecode.to_total_frames(FPS_60),
                physical_frames_before=physical_frames_before,
            )
        )
        physical_frames_before += frame_count
    return infos


def correct_for_fps_59_94(
    nominal_frame_offset: int,
    physical_frames_before: int,
    framerate: Fraction | int | float,
) -> int:
    """Convert a frame offset counted at the nominal FPS_60 rate into a physical frame index.

    `physical_frames_before` is the number of real frames already elapsed in earlier files of the
    same camera, needed so drift from the true (e.g. 59.94) frame rate is corrected relative to
    the whole recording, not just the current file.
    """
    return round((physical_frames_before + nominal_frame_offset) * (framerate / FPS_60) - physical_frames_before)


def locate_frame_position(infos: list[VideoFileInfo], nominal_target_frame: int, boundary_name: str) -> FramePosition:
    for file_index, info in enumerate(infos):
        if info.nominal_start_frame <= nominal_target_frame < info.nominal_end_frame:
            local_frame = correct_for_fps_59_94(
                nominal_target_frame - info.nominal_start_frame,
                info.physical_frames_before,
                info.framerate,
            )
            return FramePosition(file_index, min(max(local_frame, 0), info.frame_count))

    if nominal_target_frame < infos[0].nominal_start_frame:
        print(f"Warning: {boundary_name}-time is before the first available video, clamping to the start.")
        return FramePosition(0, 0)

    print(f"Warning: {boundary_name}-time is after the last available video, clamping to the end.")
    last_index = len(infos) - 1
    return FramePosition(last_index, infos[last_index].frame_count)


def resolve_start_position(infos: list[VideoFileInfo], start_time: str | None, use_timecode: bool) -> FramePosition:
    if start_time is None:
        return FramePosition(0, 0)

    nominal_frame = parse_time_str_to_frames(start_time, FPS_60)
    assert nominal_frame is not None
    if not use_timecode:
        nominal_frame += infos[0].nominal_start_frame
    return locate_frame_position(infos, nominal_frame, "start")


def resolve_end_position(infos: list[VideoFileInfo], end_time: str | None, use_timecode: bool) -> FramePosition:
    if end_time is None:
        last_index = len(infos) - 1
        return FramePosition(last_index, infos[last_index].frame_count)

    nominal_frame = parse_time_str_to_frames(end_time, FPS_60)
    assert nominal_frame is not None
    if not use_timecode:
        nominal_frame += infos[0].nominal_start_frame
    return locate_frame_position(infos, nominal_frame, "end")


@dataclass(frozen=True)
class FrameTimestamps:
    frame_pts: list[int]
    keyframe_pts: list[int]
    time_base: Fraction


def decode_frame_timestamps(video_path: Path) -> FrameTimestamps:
    """Read the pts of every frame and every keyframe in a video, in playback order.

    Packets come out of demux() in decode order, not display order, so their pts values
    aren't collected in playback sequence; sorting them recovers presentation order without
    the far more expensive alternative of fully decoding every frame's pixels just to have
    the codec hand them back in the right order.
    """
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        time_base = stream.time_base
        frame_pts: list[int] = []
        keyframe_pts: list[int] = []
        for packet in container.demux(stream):
            if packet.pts is None:
                continue
            frame_pts.append(packet.pts)
            if packet.is_keyframe:
                keyframe_pts.append(packet.pts)

    frame_pts.sort()
    keyframe_pts.sort()

    return FrameTimestamps(frame_pts=frame_pts, keyframe_pts=keyframe_pts, time_base=time_base)


def run_ffmpeg_segment(
    video_path: Path,
    output_path: Path,
    start_seconds: Fraction,
    frame_count: int | None,
    codec_args: list[str],
) -> None:
    """`-ss` before `-i` does a fast seek to the nearest keyframe, then trims forward from there -
    no need to decode (or even scan) everything before it ourselves.

    The frame count is capped with `-frames:v` rather than a `-to` end time: ffmpeg's `-to` isn't
    frame-exact for `-c copy` (observed pulling in 2 extra frames around a B-frame boundary in
    testing), whereas we already know the exact frame count from our own pts bookkeeping.
    """
    # fmt: off
    cmd = ["ffmpeg", "-y", "-ss", f"{float(start_seconds):.6f}", "-i", str(video_path), "-map", "0:v:0"]
    if frame_count is not None:
        cmd += ["-frames:v", str(frame_count)]
    cmd += [*codec_args, str(output_path)]
    # fmt: on
    subprocess.run(cmd, check=True)


def encode_segment(video_path: Path, output_path: Path, start_seconds: Fraction, frame_count: int | None) -> None:
    run_ffmpeg_segment(video_path, output_path, start_seconds, frame_count, ["-c:v", "libx264", "-crf", RE_ENCODE_CRF])


def copy_segment(video_path: Path, output_path: Path, start_seconds: Fraction, frame_count: int | None) -> None:
    run_ffmpeg_segment(video_path, output_path, start_seconds, frame_count, ["-c", "copy"])


def build_partial_segment_parts(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    tmp_dir: Path,
    tag: str,
) -> list[Path]:
    """Split [start_frame, end_frame) out of a single file, re-encoding only the partial GOPs
    at the edges and stream-copying everything in between."""
    timestamps = decode_frame_timestamps(video_path)
    frame_pts = timestamps.frame_pts
    time_base = timestamps.time_base

    def seconds(pts: int) -> Fraction:
        return pts * time_base

    start_pts = frame_pts[start_frame]
    end_pts = frame_pts[end_frame] if end_frame < len(frame_pts) else None

    copy_start = next((pts for pts in timestamps.keyframe_pts if pts >= start_pts), None)

    if copy_start is None:
        # No keyframe at or after the start: the whole range must be re-encoded.
        part_path = tmp_dir / f"{tag}_enc.mp4"
        encode_segment(video_path, part_path, seconds(start_pts), end_frame - start_frame)
        return [part_path]

    copy_start_frame = bisect.bisect_left(frame_pts, copy_start)

    parts: list[Path] = []

    if start_frame < copy_start_frame:
        head_path = tmp_dir / f"{tag}_head.mp4"
        encode_segment(video_path, head_path, seconds(start_pts), copy_start_frame - start_frame)
        parts.append(head_path)

    if end_pts is None:
        mid_path = tmp_dir / f"{tag}_mid.mp4"
        copy_segment(video_path, mid_path, seconds(copy_start), None)
        parts.append(mid_path)
        return parts

    copy_end = max((pts for pts in timestamps.keyframe_pts if copy_start <= pts <= end_pts), default=copy_start)
    copy_end_frame = bisect.bisect_left(frame_pts, copy_end)

    if copy_end_frame > copy_start_frame:
        mid_path = tmp_dir / f"{tag}_mid.mp4"
        copy_segment(video_path, mid_path, seconds(copy_start), copy_end_frame - copy_start_frame)
        parts.append(mid_path)

    if copy_end_frame < end_frame:
        tail_path = tmp_dir / f"{tag}_tail.mp4"
        encode_segment(video_path, tail_path, seconds(copy_end), end_frame - copy_end_frame)
        parts.append(tail_path)

    return parts


def build_camera_segment_parts(
    infos: list[VideoFileInfo],
    start: FramePosition,
    end: FramePosition,
    tmp_dir: Path,
) -> list[Path]:
    parts: list[Path] = []
    for file_index in range(start.file_index, end.file_index + 1):
        info = infos[file_index]
        segment_start = start.local_frame if file_index == start.file_index else 0
        segment_end = end.local_frame if file_index == end.file_index else info.frame_count

        if segment_start >= segment_end:
            continue

        if segment_start == 0 and segment_end == info.frame_count:
            # Whole file is included: no need to decode/re-mux it at all.
            parts.append(info.path)
            continue

        parts.extend(
            build_partial_segment_parts(info.path, segment_start, segment_end, tmp_dir, tag=f"file{file_index}")
        )
    return parts


def concat_parts(parts: list[Path], output_path: Path, tmp_dir: Path) -> None:
    concat_list_path = tmp_dir / "concat.txt"
    with concat_list_path.open("w", encoding="utf-8") as concat_list_file:
        for part in parts:
            concat_list_file.write(f"file '{part.resolve()}'\n")

    # fmt: off
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list_path),
            "-c", "copy",
            str(output_path),
        ],
        check=True,
    )
    # fmt: on


def extract_camera_section(
    camera_directory: Path,
    target_directory: Path,
    start_time: str | None,
    end_time: str | None,
    use_timecode: bool,
) -> None:
    infos = collect_camera_video_infos(camera_directory)
    if not infos:
        print(f"No videos found for camera '{camera_directory.name}', skipping.")
        return

    start = resolve_start_position(infos, start_time, use_timecode)
    end = resolve_end_position(infos, end_time, use_timecode)

    if (start.file_index, start.local_frame) >= (end.file_index, end.local_frame):
        print(f"Requested section for camera '{camera_directory.name}' is empty, skipping.")
        return

    output_path = target_directory / f"{camera_directory.name}.mp4"

    with tempfile.TemporaryDirectory(prefix=f"extract_section_{camera_directory.name}_") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        parts = build_camera_segment_parts(infos, start, end, tmp_dir)
        concat_parts(parts, output_path, tmp_dir)

    print(f"Wrote '{output_path}' for camera '{camera_directory.name}'.")


def extract_segment_from_video(
    source_directory: Path,
    target_directory: Path,
    start_time: str | None = None,
    end_time: str | None = None,
    use_timecode: bool = False,
) -> None:
    camera_to_process = get_camera_to_process()

    target_directory.mkdir(parents=True, exist_ok=True)

    camera_directories = [
        camera_dir
        for camera_dir in sorted(source_directory.iterdir())
        if camera_dir.is_dir() and (camera_to_process is None or camera_dir.name == camera_to_process)
    ]

    for camera_directory in tqdm(camera_directories, desc="Extracting sections", unit="Camera"):
        extract_camera_section(
            camera_directory=camera_directory,
            target_directory=target_directory,
            start_time=start_time,
            end_time=end_time,
            use_timecode=use_timecode,
        )


@click.command()
@click.option(
    "--source-directory",
    type=click.Path(file_okay=False, dir_okay=True, writable=False, path_type=Path),
    help="Directory that contains all input data files. This is expected to be a raw_sensor_data/gopro_data/<mingle folder>",
    required=True,
)
@click.option(
    "--target-directory",
    type=click.Path(file_okay=False, dir_okay=True, writable=True, path_type=Path),
    help="Path to output directory",
    required=True,
)
@click.option(
    "--start-time",
    type=str,
    default=None,
    help="Start time in format HH:MM:SS:FF, HH:MM:SS, MM:SS or SS",
)
@click.option(
    "--end-time",
    type=str,
    default=None,
    help="End time in format HH:MM:SS:FF, HH:MM:SS, MM:SS or SS",
)
@click.option(
    "--use-timecode",
    is_flag=True,
    help="Whether the start-time and end-time should be interpreted as timecodes from the video",
)
def main(
    source_directory: Path,
    target_directory: Path,
    start_time: str | None,
    end_time: str | None,
    use_timecode: bool,
) -> None:
    extract_segment_from_video(
        source_directory=source_directory,
        target_directory=target_directory,
        start_time=start_time,
        end_time=end_time,
        use_timecode=use_timecode,
    )


if __name__ == "__main__":
    main()
