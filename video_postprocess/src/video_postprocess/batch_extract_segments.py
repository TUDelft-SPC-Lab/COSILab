"""Batch extract multiple time segments from videos."""

from pathlib import Path

import click

from video_postprocess.extract_segment_from_video import (
    extract_segment_from_video,
    str_to_timedelta,
)

# Time segments to extract
TIME_SEGS_1 = [
    ["13:46:00", "13:46:30"],
    ["13:46:30", "13:47:00"],
    ["13:47:00", "13:47:30"],
    ["13:47:30", "13:48:00"],
    ["13:48:00", "13:48:30"],
    ["13:48:30", "13:49:00"],
    ["13:49:00", "13:49:30"],
    ["13:49:30", "13:50:00"],
    ["13:50:00", "13:50:30"],
    ["13:50:30", "13:51:00"],
    ["13:51:00", "13:51:30"],
    ["13:51:30", "13:52:00"],
    ["13:52:00", "13:52:30"],
    ["13:52:30", "13:53:00"],
    ["13:53:00", "13:53:30"],
    ["13:53:30", "13:54:00"],
    ["13:54:00", "13:54:30"],
    ["13:54:30", "13:55:00"],
    ["13:55:00", "13:55:30"],
    ["13:55:30", "13:56:00"],
]

TIME_SEGS_2 = [
    ["14:54:00", "14:54:30"],
    ["14:54:30", "14:55:00"],
    ["14:55:00", "14:55:30"],
    ["14:55:30", "14:56:00"],
    ["14:56:00", "14:56:30"],
    ["14:56:30", "14:57:00"],
    ["14:57:00", "14:57:30"],
    ["14:57:30", "14:58:00"],
    ["14:58:00", "14:58:30"],
    ["14:58:30", "14:59:00"],
    ["14:59:00", "14:59:30"],
    ["14:59:30", "15:00:00"],
    ["15:00:00", "15:00:30"],
    ["15:00:30", "15:01:00"],
    ["15:01:00", "15:01:30"],
    ["15:01:30", "15:02:00"],
    ["15:02:00", "15:02:30"],
    ["15:02:30", "15:03:00"],
    ["15:03:00", "15:03:30"],
    ["15:03:30", "15:04:00"],
]
@click.command()
@click.option(
    "--source-directory",
    type=click.Path(
        file_okay=False, dir_okay=True, writable=False, path_type=Path
    ),
    help="Directory that contains all input video files",
    required=True,
)
@click.option(
    "--target-directory",
    type=click.Path(
        file_okay=False, dir_okay=True, writable=True, path_type=Path
    ),
    help="Path to output directory for extracted segments",
    required=True,
)
@click.option(
    "--segment-set",
    type=click.Choice(["1", "2", "both"]),
    default="both",
    help="Which set of time segments to process: '1', '2', or 'both'",
)
@click.option(
    "--use-timecode/--no-use-timecode",
    default=False,
    help="Whether the times should be interpreted as timecodes from the video (default: False)",
)
def main(
    source_directory: Path,
    target_directory: Path,
    segment_set: str,
    use_timecode: bool,
) -> None:
    """Extract multiple time segments from videos in batch.
    
    This script processes videos using predefined time segments and
    extracts each segment to a separate output file.
    """
    # Determine which segments to process
    segments_to_process = []
    if segment_set in ["1", "both"]:
        segments_to_process.extend(TIME_SEGS_1)
    if segment_set in ["2", "both"]:
        segments_to_process.extend(TIME_SEGS_2)
    
    print(f"Processing {len(segments_to_process)} time segments...")
    print(f"Source directory: {source_directory}")
    print(f"Target directory: {target_directory}")
    print(f"Use timecode: {use_timecode}")
    print("-" * 60)
    
    # Process each segment
    for idx, (start_tc, end_tc) in enumerate(segments_to_process, 1):
        start_time_str = start_tc
        end_time_str = end_tc
        
        print(f"\n[{idx}/{len(segments_to_process)}] Processing segment: {start_tc} -> {end_tc}")
        
        try:
            extract_segment_from_video(
                source_directory=source_directory,
                target_directory=target_directory,
                start_time=str_to_timedelta(start_time_str),
                end_time=str_to_timedelta(end_time_str),
                use_timecode=use_timecode,
                every_n_frames=None,
            )
            print(f"✓ Segment {idx} completed successfully")
        except Exception as e:
            print(f"✗ Error processing segment {idx}: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("Batch extraction complete!")


if __name__ == "__main__":
    main()
