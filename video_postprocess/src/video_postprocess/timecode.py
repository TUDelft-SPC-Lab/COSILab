from __future__ import annotations

import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path
from typing import Annotated

import numpy as np
import winfiletime  # type:ignore noqa:PGH003
from pydantic import AfterValidator, BaseModel

from video_postprocess.video_utils import get_video_framerate


def _is_hours(value: int) -> bool:
    if 0 <= value < 24:
        return value
    error_msg = (
        f"Invalid hours value: {value}. Must be between 0 and 23 inclusive."
    )
    raise ValueError(error_msg)


def _is_minutes_or_seconds(value: int) -> bool:
    if 0 <= value < 60:
        return value
    error_msg = f"Invalid minutes/seconds value: {value}. Must be between 0 and 59 inclusive."
    raise ValueError(error_msg)


def _is_frames(value: int) -> bool:
    if 0 <= value < 60:
        return value
    error_msg = (
        f"Invalid frames value: {value}. Must be between 0 and 59 inclusive."
    )
    raise ValueError(error_msg)


class VideoTimecode(BaseModel):
    """
    a video timecode contains only the time-of-day and frame number.
    fps is the frames-per-second value.
    """

    hours: Annotated[int, AfterValidator(_is_hours)]
    minutes: Annotated[int, AfterValidator(_is_minutes_or_seconds)]
    seconds: Annotated[int, AfterValidator(_is_minutes_or_seconds)]
    frames: Annotated[int, AfterValidator(_is_frames)]
    fps: Annotated[Fraction, "frames per second"]

    def to_datetime(self) -> datetime:
        """
        Convert to a datetime object.

        Note that the microseconds are rounded to the nearest integer.

        @rtype a datetime object with the contained time, date is set to 0001-01-01
        """
        microseconds = timecode_ff_to_microseconds(self.frames, self.fps)
        microseconds = round(microseconds)
        return datetime(
            year=1,
            month=1,
            day=1,
            hour=self.hours,
            minute=self.minutes,
            second=self.seconds,
            microsecond=microseconds,
        )

    def to_timedelta(self) -> timedelta:
        """
        @rtype a timedelta object with the contained time as a delta-time
        """
        microseconds = timecode_ff_to_microseconds(self.frames, self.fps)
        return timedelta(
            hours=self.hours,
            minutes=self.minutes,
            seconds=self.seconds,
            microseconds=microseconds,
        )

    def to_ffmpeg_format(self) -> str:
        """
        returns contained times as  "HH:MM:SS:FF"
        """
        return f"{self.hours:02}:{self.minutes:02}:{self.seconds:02}:{self.frames:02}"

    @staticmethod
    def from_ffmpeg_format(
        timecode_str: str, frame_rate: Fraction
    ) -> VideoTimecode:
        """
        @param timecode_str: From a str "HH:MM:SS:FF". FF the frame number
        @param frame_rate the framerate, between 0 and 60
        @rtype VideoTimecode containing the given value
        """
        hours, minutes, seconds, frames = map(int, timecode_str.split(":"))
        return VideoTimecode(
            hours=hours,
            minutes=minutes,
            seconds=seconds,
            frames=frames,
            fps=frame_rate,
        )

    @staticmethod
    def from_20char(tc: str, frame_rate: Fraction) -> VideoTimecode:
        """
        @param timecode: string with format is MMDDYYYYHHMMSSFFFFFF
            For example: 04162024103931800001 -> 04/16/2024 10:39:31:800001
            The last 6 digits is the frame number,
            represented as a decimal fraction of the framerate
            This type of timecode is encountered in filenames
            of video frames generated from IOI exporter
        @param framerate the exact framerate, typically 30fps
        @rtype the VideoTimecode matching this timecode. NOTE
            this drops the day information.
        """
        if len(tc) != 20:
            raise ValueError("Timecode is not 20 chars:" + tc)
        timecode = tc[len("MMDDYYYY") :]
        frame_num = timecode_microseconds_to_ff(int(timecode[6:]), frame_rate)
        return VideoTimecode(
            hours=int(timecode[:2]),
            minutes=int(timecode[2:4]),
            seconds=int(timecode[4:6]),
            frames=frame_num,
            fps=frame_rate,
        )

    @staticmethod
    def from_datetime(t: datetime, framerate: Fraction) -> VideoTimecode:
        """
        @param t the datetime. Datetime is assumed localtime.
        @param framerate is the actual framerate of the video
        @rtype the VideoTimecode matching the given datetime,
        the day is actually ignored.
        The milliseconds of the time are used
        to determine the frame number inside the given time.
        """
        return VideoTimecode(
            hours=t.hour,
            minutes=t.minute,
            seconds=t.second,
            frames=round(framerate * t.microsecond / 1000000),
            fps=framerate,
        )

    @staticmethod
    def from_timedelta(t: timedelta, framerate: Fraction) -> VideoTimecode:
        """
        @param t the datetime. Datetime is assumed localtime.
        @param framerate is the actual framerate of the video
        @rtype the VideoTimecode matching the given datetime,
        the day is actually ignored.
        The milliseconds of the time are used
        to determine the frame number inside the given time.
        """
        hours = t.days * 24 + (t.seconds // 3600)
        minutes = (t.seconds % 3600) // 60
        return VideoTimecode(
            hours=hours,
            minutes=minutes,
            seconds=t.seconds % 60,
            frames=round(framerate * t.microseconds / 1000000),
            fps=framerate,
        )

    @staticmethod
    def from_win_timecode(
        start: str, fps: Fraction, utc_offset: timedelta
    ) -> VideoTimecode:
        """
        @param start the 64-bit Microsoft Filetime,
        this is a number like 128930364000000000
        @param fps the frame rate
        @param utc_offset a timedelta indicating
        the time hours offset = localtime-UTC. This is needed
        because VideoTimecode always contains localtime
        and windows timecode is UTC.
        @return a VideoTimecode matching the given time.
        The milliseconds of the time are used to determine
        the frame number inside the given time.
        """
        if len(start) != 18:
            raise ValueError("Timecode is not 18 chars:" + start)
        return VideoTimecode.from_datetime(
            winfiletime.to_datetime(int(start)) + utc_offset, fps
        )

    @staticmethod
    def from_IOI_xml(xmlfile: Path, utc_offset: timedelta) -> VideoTimecode:
        """
        @param xmlfile the filename of an IOI SceneinfoConfig.xml file.
        The xml file contains a SceneInfoConfig node,
        with FPS, Start and Frame numbers.
        The Start value contains a Windows FileTime matching the
        start time of the video.
        We are not aware of official spec, but reverse engineered
        @param utc_offset a timedelta indicating
        the time hours offset = localtime-UTC. This is needed
        because VideoTimecode always contains localtime
        and windows FileTime is UTC.
        @rtype a VideoTimecode according to the file info
        """
        tree = ET.parse(xmlfile)
        root = tree.getroot()
        elt = root.find("FPS")
        fps: Fraction = Fraction(
            int(elt.get("Numerator")), int(elt.get("Denominator"))
        )
        start: str = root.find("Start").get("time")
        return VideoTimecode.from_win_timecode(start, fps, utc_offset)

    @staticmethod
    def from_video(video_path: Path) -> VideoTimecode:
        """
        @param video_path path to a video file
        """
        # fmt: off
        cmd = [
            "ffprobe",
            "-hide_banner",
            "-show_streams",
            "-i", str(video_path),
        ]
        # fmt: on
        res = subprocess.run(cmd, check=True, capture_output=True)

        video_timecode = res.stdout.decode("utf-8").rstrip()

        # Find TAG:timecode=<VAL> in the string, the value is in the format HH:MM:SS:FF
        video_timecode = video_timecode.split("TAG:timecode=")[1].split("\n")[0]

        return VideoTimecode.from_ffmpeg_format(
            video_timecode, get_video_framerate(video_path)
        )


# The timecode in a video is HH:MM:SS:FF FF, hours, minutes, seconds, frame numer
# The frame number, which goes from 00 to [frame rate - 1])
# e.g. if the frame rate is 60, FF goes from [00 to 59].
# Notice that FF never reaches the 1000000 microseconds equivalent, as that would be the next frame.
# We assume that the first frame for the first second of footage is triggered at 0000000, the last frame of the
# first second must be triggered before the second second of footage starts.
#
# Example of a 2 second video at 2 FPS where we have 5 frames:
# Frames        ||----||----||----||----||
# Frame number  00    01    02    03    04
# FF            00    01    00    01    00
# Time (FF)     0    0.5     0   0.5     0
# Time (s)      0    0.5     1   1.5     2
#
# Hence, Time (FF) = FF / framerate, e.g. 1 / 2 = 0.5.
# And FF = Time (FF) * framerate, e.g. Time (s) = 1.5 -> Time (FF) = 0.5, then 0.5 * 2 = 1.
def timecode_microseconds_to_ff(
    timecode_microseconds: float, framerate: Fraction | int
) -> int:
    frame_num = round((timecode_microseconds / 1000000) * framerate)
    return int(np.clip(frame_num, 0, round(framerate - 1)))


def timecode_ff_to_microseconds(
    timecode_ff: int, framerate: float | int
) -> float:
    return float(1000000 * (timecode_ff / framerate))
