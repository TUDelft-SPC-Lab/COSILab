"""Decoding media files for backends that consume pixels and samples.

Local backends need frames, waveforms and images; hosted ones upload the file
untouched. This module holds the decoding side, shared by any backend that needs
it, so no task ever imports PyAV, librosa or Pillow.

Every heavy import is lazy and inside a function: importing this module costs
nothing, which keeps ``models.base`` and ``models.registry`` importable in a
standard-library-only environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Both qwen_omni_utils and bailingmm_utils round a requested frame count to a
# multiple of this and refuse anything below it, so the shared policy produces
# multiples of it and the backends have nothing left to change.
FRAME_FACTOR = 2

__all__ = [
    "FRAME_FACTOR",
    "frames_at_fps",
    "get_video_frame_count",
    "get_video_frame_rate",
    "load_audio_array",
    "load_image",
    "load_video_frames",
    "select_video_num_frames",
]


def get_video_frame_count(video_path: str | Path) -> int | None:
    """Return the video frame count when it can be inferred cheaply."""
    try:
        import av
    except ImportError:
        return None

    try:
        with av.open(str(video_path)) as container:
            video_stream = next(
                (stream for stream in container.streams if stream.type == "video"),
                None,
            )
            if video_stream is None:
                return None
            if video_stream.frames and video_stream.frames > 0:
                return int(video_stream.frames)
            if (
                video_stream.duration is not None
                and video_stream.time_base is not None
                and video_stream.average_rate is not None
            ):
                duration_seconds = float(video_stream.duration * video_stream.time_base)
                estimated_frames = int(duration_seconds * float(video_stream.average_rate))
                if estimated_frames > 0:
                    return estimated_frames

            decoded_frames = 0
            for _ in container.decode(video=0):
                decoded_frames += 1
            return decoded_frames if decoded_frames > 0 else None
    except Exception:
        return None


def get_video_frame_rate(video_path: str | Path) -> float | None:
    """The container's own frame rate, when it can be read cheaply."""
    try:
        import av
    except ImportError:
        return None

    try:
        with av.open(str(video_path)) as container:
            video_stream = next(
                (stream for stream in container.streams if stream.type == "video"),
                None,
            )
            if video_stream is None or video_stream.average_rate is None:
                return None
            rate = float(video_stream.average_rate)
            return rate if rate > 0 else None
    except Exception:
        return None


def select_video_num_frames(video_path: str | Path, max_video_frames: int) -> int:
    if max_video_frames <= 0:
        raise ValueError(f"max_video_frames must be positive, got {max_video_frames}")
    frame_count = get_video_frame_count(video_path)
    if frame_count is None:
        return max_video_frames
    return max(1, min(max_video_frames, frame_count))


def frames_at_fps(
    video_path: str | Path,
    fps: float,
    min_frames: int,
    max_frames: int,
) -> int:
    """How many frames to sample from this clip, under one shared policy.

    The single place the frame count is decided, for every backend. It matters
    that there is only one: frame count is the largest controllable confound in
    a model comparison, and until this existed the three backends resolved it
    differently enough that a 2-second clip gave Qwen 4 frames and Ming 50 --
    a sampling difference that would have read as a model difference.

    The rule, which is Qwen's own upstream rule generalised to the others:

        n = duration * fps, clamped to [min_frames, max_frames],
        never more than the clip actually holds, and rounded down to even.

    Even because both qwen_omni_utils and bailingmm_utils round to a multiple of
    2 anyway (``FRAME_FACTOR``); doing it here means all three backends receive
    a number none of them will change, so they agree exactly rather than
    approximately.

    Falls back to ``min_frames`` when the container will not say how long it is,
    rather than guessing high: an unreadable clip should cost little, and the
    packages raise if asked for more frames than exist.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps!r}")
    if min_frames < FRAME_FACTOR or max_frames < min_frames:
        raise ValueError(
            f"need {FRAME_FACTOR} <= min_frames <= max_frames, "
            f"got min_frames={min_frames}, max_frames={max_frames}"
        )

    frame_count = get_video_frame_count(video_path)
    source_fps = get_video_frame_rate(video_path)
    if frame_count is None or source_fps is None:
        return min_frames

    requested = frame_count / source_fps * fps
    clamped = min(max(requested, min_frames), max_frames, frame_count)
    even = int(clamped) // FRAME_FACTOR * FRAME_FACTOR
    # A clip shorter than FRAME_FACTOR frames cannot be sampled at all; the
    # backends raise on it, so return the floor and let them say so by name.
    return max(FRAME_FACTOR, even)


def load_video_frames(video_path: str | Path, num_frames: int) -> list[Any]:
    """Decode and evenly sample video frames as PIL images."""
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("PyAV is required to pre-decode video inputs") from exc

    decoded_frames: list[Any] = []
    with av.open(str(video_path)) as container:
        for frame in container.decode(video=0):
            decoded_frames.append(frame.to_image().convert("RGB"))

    if not decoded_frames:
        raise ValueError(f"No decodable video frames found: {video_path}")
    if len(decoded_frames) <= num_frames:
        return decoded_frames
    if num_frames == 1:
        return [decoded_frames[len(decoded_frames) // 2]]

    last_index = len(decoded_frames) - 1
    sample_indices = [
        round(index * last_index / (num_frames - 1))
        for index in range(num_frames)
    ]
    return [decoded_frames[index] for index in sample_indices]


def load_audio_array(audio_path: str | Path, sampling_rate: int) -> Any:
    """Decode audio as a mono float array at the processor sampling rate."""
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("librosa is required to pre-decode audio inputs") from exc

    audio, _ = librosa.load(str(audio_path), sr=sampling_rate, mono=True)
    return audio


def load_image(image_path: str | Path) -> Any:
    """Load an image as RGB PIL.Image."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to pre-decode image inputs") from exc

    with Image.open(str(image_path)) as image:
        return image.convert("RGB")
