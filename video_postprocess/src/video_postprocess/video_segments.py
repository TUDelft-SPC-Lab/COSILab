from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import timedelta
from fractions import Fraction
from pathlib import Path

import cv2

from video_postprocess.timecode import VideoTimecode
from video_postprocess.video_utils import get_video_framerate

# Nominal (non-drop) frame rate that start-time/end-time/timecodes are expressed against. Actual
# 59.94 fps footage is corrected against this via correct_for_fps_59_94.
FPS_60 = 60


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
    start_timecode: VideoTimecode

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
    # GoPro videos are named with capital letters, including the extension. Both spellings are
    # collected into a set first: on case-insensitive filesystems each glob returns every file.
    video_paths = sorted({*camera_directory.glob("*.mp4"), *camera_directory.glob("*.MP4")})
    for video_path in video_paths:
        try:
            framerate = get_video_framerate(video_path)
        except (OSError, subprocess.CalledProcessError) as e:
            # moov atom not found, as happens when a GoPro's battery dies mid-recording.
            if "moov atom not found" in e.stdout.decode():
                continue
            else:
                raise
        frame_count = get_video_total_frame_num(video_path)
        timecode = VideoTimecode.from_video(video_path)
        infos.append(
            VideoFileInfo(
                path=video_path,
                framerate=framerate,
                frame_count=frame_count,
                nominal_start_frame=timecode.to_total_frames(FPS_60),
                physical_frames_before=physical_frames_before,
                start_timecode=timecode,
            )
        )
        physical_frames_before += frame_count
    return infos


def timecode_at_local_frame(info: VideoFileInfo, local_frame: int) -> VideoTimecode:
    """The embedded-timecode clock time of `local_frame` within `info`'s file: its own start
    timecode advanced by `local_frame` at the file's actual (physical) framerate."""
    elapsed = timedelta(seconds=float(local_frame / info.framerate))
    return VideoTimecode.from_timedelta(info.start_timecode.to_timedelta() + elapsed, info.framerate)


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


def invert_correct_for_fps_59_94(
    local_frame: int,
    physical_frames_before: int,
    framerate: Fraction | int | float,
) -> int:
    """Inverse of `correct_for_fps_59_94`: recover the nominal (FPS_60) frame offset that a
    `--start-time`/`--use-timecode` value would need to encode, relative to the file's own start
    timecode, for ffmpeg's `-ss` to land on `local_frame`, the physical frame actually reached in
    the file.
    """
    return round(
        (local_frame + physical_frames_before) * (Fraction(FPS_60) / framerate) - physical_frames_before
    )


def world_timecode_at_local_frame(info: VideoFileInfo, local_frame: int) -> VideoTimecode:
    """The nominal (FPS_60, drift-uncorrected) timecode that `--start-time --use-timecode` would
    need to be given to land on `local_frame` of `info`'s file -- the inverse of the correction
    `locate_frame_position` applies via `correct_for_fps_59_94`, including the drift already
    accumulated over the camera's earlier files via `info.physical_frames_before`."""
    nominal_offset = invert_correct_for_fps_59_94(local_frame, info.physical_frames_before, info.framerate)
    return VideoTimecode.from_total_frames(info.nominal_start_frame + nominal_offset, FPS_60)


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
