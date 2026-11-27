"""Per-frame world timestamps of raw GoPro files from their MP4 timecode tags (the 4TU timestamps rule).

original timecode = file tag + row (60 per timecode second); world = first file's tag + round-half-even(global
frame * 1.001) ticks of 1/60 s (59.94 fps drift). Port of Video-OnlineHMR lib/utils/cosilab.tag_timestamps.

Usage: python timestamps.py SESSION --check 6 7 8 9 10 --cams 12 13 14 15
  --check rebuilds the CSVs of top cameras and compares them with the 4TU ones row by row;
  --cams writes side_calib_work/timestamps/mingle_session_N/camera_XX-mingle_session_N/*.csv
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np

from common import ROOT, TU4, WORK

TOP_TS = TU4 / "processed/videos/timestamps"
TOP_RAW = TU4 / "raw/video"
SIDE_RAW = ROOT / "raw_data/gopro_data"


def tc_frames(tc):
    h, m, s, f = map(int, tc.split(":"))
    return ((h * 60 + m) * 60 + s) * 60 + f


def fmt_tc(n):
    return f"{n // 216000:02d}:{n // 3600 % 60:02d}:{n // 60 % 60:02d}:{n % 60:02d}"


def fmt_ts(n):
    return f"{n // 216000:02d}:{n // 3600 % 60:02d}:{(n % 3600) / 60:09.6f}"


def tag_timestamps(video_dir):
    names = sorted(n.name for n in Path(video_dir).iterdir() if re.fullmatch(r"GH\d+\.MP4", n.name))
    out, g, prev_end, start = [], 0, None, None
    for name in names:
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                "stream=nb_frames:stream_tags=timecode", "-of", "json", str(Path(video_dir) / name)],
                               capture_output=True, text=True)
        if probe.returncode != 0:
            print(f"[warning] {video_dir}/{name} is unreadable; it and later files get no timestamps", flush=True)
            break
        st = json.loads(probe.stdout)["streams"][0]
        n0, count = tc_frames(st["tags"]["timecode"]), int(st["nb_frames"])
        if prev_end is not None and n0 != prev_end:
            raise RuntimeError(f"{video_dir}/{name}: timecode tag does not continue the previous file")
        start = n0 if start is None else start
        world = start + np.round((g + np.arange(count)) * 1001 / 1000).astype(np.int64)
        out.append((name.replace(".MP4", ".csv"), [(fmt_tc(n0 + i), fmt_tc(int(w)), fmt_ts(int(w)))
                                                   for i, w in enumerate(world)]))
        g, prev_end = g + count, n0 + count
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=int)
    ap.add_argument("--check", type=int, nargs="*", default=[])
    ap.add_argument("--cams", type=int, nargs="*", default=[])
    a = ap.parse_args()
    S = a.session
    for cam in a.check:
        sub = f"camera_{cam:02d}-mingle_session_{S}"
        ref_dir = TOP_TS / f"mingle_session_{S}" / sub
        bad = rows = 0
        for name, made in tag_timestamps(TOP_RAW / f"top_view_camera-mingle_session_{S}" / sub):
            p = ref_dir / name
            if not p.exists():
                print(f"  cam{cam:02d}: {name} not in the 4TU CSVs (skipped)")
                continue
            ref = [tuple(l.rstrip("\n").split(",")) for l in p.read_text().splitlines(True)[1:] if l.strip()]
            bad += sum(x != y for x, y in zip(made, ref)) + abs(len(made) - len(ref))
            rows += len(ref)
        print(f"cam{cam:02d}: {bad} of {rows} rows differ from the 4TU CSVs", flush=True)
    for cam in a.cams:
        sub = f"camera_{cam:02d}-mingle_session_{S}"
        out = WORK / "timestamps" / f"mingle_session_{S}" / sub
        out.mkdir(parents=True, exist_ok=True)
        files = tag_timestamps(SIDE_RAW / f"side_view_camera-mingle_session_{S}" / sub)
        for name, rows in files:
            (out / name).write_text("original_timecode,world_timecode,world_timestamp\n" +
                                    "".join(",".join(r) + "\n" for r in rows))
        print(f"cam{cam:02d}: {len(files)} files, {sum(len(r) for _, r in files)} frames, world "
              f"{files[0][1][0][2]} .. {files[-1][1][-1][2]} -> {out}", flush=True)


if __name__ == "__main__":
    main()
