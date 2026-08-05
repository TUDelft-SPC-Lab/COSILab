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

from video_postprocess.utils import get_camera_to_process
from video_postprocess.video_segments import (
    FramePosition,
    VideoFileInfo,
    collect_camera_video_infos,
    resolve_end_position,
    resolve_start_position,
    timecode_at_local_frame,
)

# Quality target for the re-encoded head/tail edges of a cut. CRF rather than a bit_rate, since these
# segments are only ever a few frames long, too short for bitrate-mode rate control to ramp up.
RE_ENCODE_CRF = "18"


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


def concat_parts(parts: list[Path], output_path: Path, tmp_dir: Path, timecode: str) -> None:
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
            "-timecode", timecode,
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
    output_timecode = timecode_at_local_frame(infos[start.file_index], start.local_frame).to_ffmpeg_format()

    with tempfile.TemporaryDirectory(prefix=f"extract_section_{camera_directory.name}_") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        parts = build_camera_segment_parts(infos, start, end, tmp_dir)
        concat_parts(parts, output_path, tmp_dir, output_timecode)

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
    help=(
        "Directory that contains all input data files. This is expected to "
        "be a raw_sensor_data/gopro_data/<mingle folder>"
    ),
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
