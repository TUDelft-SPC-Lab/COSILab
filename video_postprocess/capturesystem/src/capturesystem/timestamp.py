import argparse
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

from ffmpeg import FFmpeg

from capturesystem.timecode import VideoTimecode


def timestamp_video(source: Path, target: Path, timecode: VideoTimecode):
    """
    executes ffmpeg command to create
    a target video, as a copy of source,
    with the timecode added.
    :param source the path to the source video (mp4)
    :param target the path th the target video (mp4)
    :param timecode the video timecode
    """
    if not source.suffix == ".mp4":
        raise ValueError("source path must be mp4 file, got " + str(source))
    if not target.suffix == ".mp4":
        raise ValueError("target path must be mp4 file, got " + str(target))
    ffmpeg = (  # extra brackets needed for multiline...
        FFmpeg()
        .option("n")
        .input(str(source))
        .output(
            str(target),
            {"codec:v": "copy", "codec:a": "copy"},
            timecode=timecode.to_ffmpeg_format(),
        )
    )
    ffmpeg.execute()


class TimeStampedVideos:
    """
    Contains copy of videos in sourcedir, with timestamps
    added to all the videos.
    """

    def __init__(
        self,
        sourcedir: Path,
        utc_offset: timedelta,
        targetdir: Path | None = None,
        xmlfile: str | None = None,
    ):
        """
        @param sourcedir the directory Path containing the videos
        @param utc_offset a timedelta indicating
            the time hours offset = localtime-UTC. This is needed
            because VideoTimecode always contains localtime
            and windows timecode is UTC.
        @param xmlfile a string with the filename of the
            xml file containing the windows filetime.
            Defaults to "SceneInfoConfig.xml" if None.
        @param targetdir a directory Path where the timestamped
            videos are copied.
            if None, a temp directory is created
        """
        if xmlfile is None:
            xmlfile = "SceneInfoConfig.xml"
        self.__sourcedir = sourcedir
        self.__targetdir = (
            Path(tempfile.mkdtemp()) if targetdir is None else targetdir
        )
        if not self.__targetdir.is_dir():  # fixme also check writability?
            raise ValueError(
                "targetdir " + str(self.__targetdir) + " is not a directory"
            )
        self.__utc_offset = utc_offset
        self.__xmlfile = self.__sourcedir.joinpath(xmlfile)
        self.__timecode = VideoTimecode.from_IOI_xml(
            self.__xmlfile, self.__utc_offset
        )

        # the actual conversion of the videos. We could also do this lazily
        for sourcevid in sourcedir.glob("**/*.mp4"):
            targetvid: Path = self.__targetdir.joinpath(sourcevid.name)
            timestamp_video(sourcevid, targetvid, self.__timecode)

    def getTargetDir(self):
        """
        @rtype the directory where the timestamped videos are
        """
        return self.__targetdir

    @staticmethod
    def main(cargs) -> str:
        """
        Tool to help using a command line with arguments.
        WARNING this code is mainly here to support testing.
        Use the TimeStampedVideos class directly in python.
        @param cargs the command line args (without the program name itself)
        @rtype the target directory
        """
        parser = argparse.ArgumentParser(
            description="A tool to put the timestamps in an"
            ".xml file into the videos of the same directory"
        )
        parser.add_argument(
            "-s",
            "--source",
            type=str,
            required=True,
            help="source directory. Must contain .mp4 files and an xml file",
        )
        parser.add_argument(
            "-o",
            "--offset",
            type=float,
            required=True,
            help="#hours to add to conver the UTC in the xml file to local time",
        )
        parser.add_argument(
            "-t",
            "--target",
            type=str,
            default="",
            help="target directory. Creates temp folder if left empty",
        )
        parser.add_argument(
            "-x",
            "--xmlfile",
            type=str,
            help="filename of xml file in source dir",
        )
        args = parser.parse_args(cargs)

        target = None if args.target == "" else Path(args.target)
        # we assume the offset is
        res = TimeStampedVideos(
            Path(args.source),
            timedelta(minutes=int(args.offset * 60)),
            target,
            args.xmlfile,
        )
        return res.getTargetDir()


if __name__ == "__main__":
    print(TimeStampedVideos.main(sys.argv[1:]))
