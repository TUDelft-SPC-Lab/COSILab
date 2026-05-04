"""Extract audio from video files without resampling."""

import subprocess
from datetime import timedelta
from pathlib import Path

import click
from tqdm import tqdm

from capturesystem.timecode import VideoTimecode


def get_audio_sample_rate(video_path: Path) -> int:
    """Get the audio sample rate from a video file using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return int(result.stdout.strip())


def get_audio_channels(video_path: Path) -> int:
    """Get the number of audio channels from a video file using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=channels",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return int(result.stdout.strip())


def str_to_timedelta(time_str: str | None) -> timedelta | None:
    """Convert a time string in HH:MM:SS, MM:SS or SS format to a timedelta object."""
    if time_str is None:
        return None

    parts = list(map(float, time_str.split(":")))
    if len(parts) == 2:
        parts = [0, parts[0], parts[1]]
    elif len(parts) == 1:
        parts = [0, 0, parts[0]]
    elif len(parts) != 3:
        error_msg = "Time string must be in HH:MM:SS, MM:SS or SS format."
        raise ValueError(error_msg)
    return timedelta(hours=parts[0], minutes=parts[1], seconds=parts[2])


def timedelta_to_hhmmsscs(td: timedelta | None) -> str:
    """Convert a timedelta to HHMMSSss format (Hours:Minutes:Seconds:Centiseconds)."""
    if td is None:
        return ""

    total_seconds = td.total_seconds()
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    centiseconds = int((total_seconds % 1) * 100)  # Centiseconds are hundredths of a second

    return f"{hours:02d}{minutes:02d}{seconds:02d}{centiseconds:02d}"


def extract_audio_from_video(
    video_path: Path,
    output_path: Path,
    start_time: timedelta | None = None,
    end_time: timedelta | None = None,
    use_timecode: bool = False,
    mono: bool = False,
    preserve_sample_rate: bool = True,
    target_sample_rate: int | None = None,
) -> int:
    """
    Extract audio from a video file to WAV format.

    Args:
        video_path: Path to the input video file.
        output_path: Path to the output WAV file.
        start_time: Optional start time for extraction.
        end_time: Optional end time for extraction.
        use_timecode: If True, interpret start_time and end_time as absolute
                      timecodes and adjust based on the video's embedded timecode.
        mono: If True, convert to mono (single channel).
        preserve_sample_rate: If True, keep original sample rate.
        target_sample_rate: If specified and preserve_sample_rate is False,
                           resample to this rate.

    Returns:
        The sample rate of the extracted audio.
    """
    # Get original audio properties
    sample_rate = get_audio_sample_rate(video_path)
    channels = get_audio_channels(video_path)
    
    print(f"  Original sample rate: {sample_rate} Hz")
    print(f"  Original channels: {channels}")

    # Handle timecode adjustment
    start_time_adj = start_time
    end_time_adj = end_time
    
    if use_timecode:
        if start_time is None and end_time is None:
            print("  --use-timecode ignored: start-time and end-time were not provided.")
        else:
            timecode = VideoTimecode.from_video(video_path)
            video_start = timecode.to_timedelta()
            print(f"  Video timecode: {timecode.to_ffmpeg_format()}")
            
            if start_time is not None:
                start_time_adj = start_time - video_start
                if start_time_adj.total_seconds() < 0:
                    print(f"  Warning: Start time {start_time} is before video start {video_start}")
            
            if end_time is not None:
                end_time_adj = end_time - video_start
                if end_time_adj.total_seconds() < 0:
                    print(f"  Warning: End time {end_time} is before video start {video_start}")

    # Build ffmpeg command
    cmd = ["ffmpeg", "-y"]
    cmd.extend(["-i", str(video_path)])
    cmd.append("-vn")

    # Calculate sample positions from the VIDEO START (sample 0)
    if start_time_adj is not None:
        start_sample = int(start_time_adj.total_seconds() * sample_rate)
    else:
        start_sample = 0

    if end_time_adj is not None:
        end_sample = int(end_time_adj.total_seconds() * sample_rate)
        trim_filter = f"atrim=start_sample={start_sample}:end_sample={end_sample}"
    else:
        trim_filter = f"atrim=start_sample={start_sample}"

    cmd.extend(["-af", trim_filter])
    
    # Audio channels
    if mono:
        cmd.extend(["-ac", "1"])
    
    # Sample rate
    if not preserve_sample_rate and target_sample_rate is not None:
        cmd.extend(["-ar", str(target_sample_rate)])
        output_sample_rate = target_sample_rate
        print(f"  Resampling to: {target_sample_rate} Hz")
    else:
        output_sample_rate = sample_rate
    
    # Output file
    cmd.append(str(output_path))
    
    # Run ffmpeg
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    
    return output_sample_rate


def extract_audio_from_directory(
    source_directory: Path,
    target_directory: Path,
    start_time: timedelta | None = None,
    end_time: timedelta | None = None,
    use_timecode: bool = False,
    mono: bool = False,
    preserve_sample_rate: bool = True,
    target_sample_rate: int | None = None,
    video_extension: str = "mp4",
) -> None:
    """
    Extract audio from all video files in a directory.

    Args:
        source_directory: Directory containing video files.
        target_directory: Directory to save extracted audio files.
        start_time: Optional start time for extraction.
        end_time: Optional end time for extraction.
        use_timecode: If True, interpret start_time and end_time as absolute
                      timecodes and adjust based on each video's embedded timecode.
        mono: If True, convert to mono (single channel).
        preserve_sample_rate: If True, keep original sample rate.
        target_sample_rate: If specified and preserve_sample_rate is False,
                           resample to this rate.
        video_extension: Extension of video files to process.
    """
    target_directory.mkdir(parents=True, exist_ok=True)
    
    video_paths = sorted(source_directory.glob(f"*.{video_extension}"))
    
    if not video_paths:
        print(f"No .{video_extension} files found in {source_directory}")
        return
    
    print(f"Found {len(video_paths)} video file(s)")
    
    for video_path in tqdm(video_paths, desc="Extracting audio", unit="file"):
        output_path = target_directory / f"{video_path.stem}_audioD_{timedelta_to_hhmmsscs(start_time)}_{timedelta_to_hhmmsscs(end_time)}.wav"
        
        print(f"\nProcessing: {video_path.name}")
        
        extract_audio_from_video(
            video_path=video_path,
            output_path=output_path,
            start_time=start_time,
            end_time=end_time,
            use_timecode=use_timecode,
            mono=mono,
            preserve_sample_rate=preserve_sample_rate,
            target_sample_rate=target_sample_rate,
        )
        
        print(f"  Saved to: {output_path.name}")


@click.command()
@click.option(
    "--source",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to a video file or directory containing video files.",
)
@click.option(
    "--target",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to output WAV file or directory for extracted audio.",
)
@click.option(
    "--start-time",
    type=str,
    default=None,
    help="Start time in format HH:MM:SS, MM:SS, or SS (supports decimals).",
)
@click.option(
    "--end-time",
    type=str,
    default=None,
    help="End time in format HH:MM:SS, MM:SS, or SS (supports decimals).",
)
@click.option(
    "--use-timecode",
    is_flag=True,
    help="Interpret start-time and end-time as absolute timecodes from the video's embedded timecode.",
)
@click.option(
    "--mono",
    is_flag=True,
    help="Convert audio to mono (single channel).",
)
@click.option(
    "--resample",
    type=int,
    default=None,
    help="Resample audio to this sample rate (Hz). If not specified, preserves original.",
)
@click.option(
    "--extension",
    type=str,
    default="mp4",
    help="Video file extension to process when source is a directory (default: mp4).",
)
def main(
    source: Path,
    target: Path,
    start_time: str | None,
    end_time: str | None,
    use_timecode: bool,
    mono: bool,
    resample: int | None,
    extension: str,
) -> None:
    """Extract audio from video files without resampling (by default)."""
    
    start_td = str_to_timedelta(start_time)
    end_td = str_to_timedelta(end_time)
    
    preserve_sample_rate = resample is None
    
    if source.is_file():
        # Single file mode
        if target.suffix.lower() != ".wav":
            target = target / f"{source.stem}.wav"
        target.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing: {source.name}")
        
        sample_rate = extract_audio_from_video(
            video_path=source,
            output_path=target,
            start_time=start_td,
            end_time=end_td,
            use_timecode=use_timecode,
            mono=mono,
            preserve_sample_rate=preserve_sample_rate,
            target_sample_rate=resample,
        )
        
        print(f"  Saved to: {target}")
        print(f"  Output sample rate: {sample_rate} Hz")
    else:
        # Directory mode
        extract_audio_from_directory(
            source_directory=source,
            target_directory=target,
            start_time=start_td,
            end_time=end_td,
            use_timecode=use_timecode,
            mono=mono,
            preserve_sample_rate=preserve_sample_rate,
            target_sample_rate=resample,
            video_extension=extension,
        )


if __name__ == "__main__":
    main()
