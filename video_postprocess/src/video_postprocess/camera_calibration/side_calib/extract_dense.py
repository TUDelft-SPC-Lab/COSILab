"""Extract dense calibration frames, synchronized in world time with the raw timecode tag rule.

World time per raw frame comes from the 4TU timestamp CSVs (top cameras) or from timestamps.py (side cameras; same
rule: world = first file's tag + round(frame * 1.001) ticks of 1/60 s), never from the timecode_fixed videos.
The grid covers the calibration window of the session (raw timecode of the anchor camera: session 1 cam06
13:25:20:00-13:29:46:00, session 2 cam04 14:47:14:00-14:50:00:00 minus 18 frames) in steps of STEP world ticks;
each camera contributes the raw frame whose world tick is nearest (all within 1 tick).

Output: side_calib_work/frames/s{N}dense/camera_XX-mingle_session_N/{k:09d}.png (grey) and
grid_cam{XX}.json (k -> world tick, raw file/row and world-tick mismatch).
Usage: python extract_dense.py --session N --cams 6 7 --step 10     (one or a few cameras per process: memory)
"""
import argparse
import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from common import ROOT, TU4, WORK

# Per session: top-camera timestamps are the 4TU CSVs; side-camera ones are made with the same tag rule by
# timestamps.py (side_calib_work/timestamps/; identical to Video-OnlineHMR's cosilab_side_timestamps for session 2).
# Calibration window (raw timecode of the anchor camera) and the anchor of the grid.
SESSION_CFG = {
    1: dict(start=((13 * 60 + 25) * 60 + 20) * 60, end=((13 * 60 + 29) * 60 + 46) * 60, anchor_cam=6, anchor_offset=0,
            side_ts=WORK / "timestamps/mingle_session_1"),
    2: dict(start=((14 * 60 + 47) * 60 + 14) * 60, end=((14 * 60 + 50) * 60 + 0) * 60, anchor_cam=4,
            anchor_offset=-18,  # raw timecode of old calib frame i in cam04 = start + 30 i - 18
            side_ts=WORK / "timestamps/mingle_session_2"),
}
SESSION = 2
FPS = 60000 / 1001


def ts_dir(cam):
    S = SESSION
    base = SESSION_CFG[S]["side_ts"] if cam > 10 else TU4 / f"processed/videos/timestamps/mingle_session_{S}"
    return base / f"camera_{cam:02d}-mingle_session_{S}"


def raw_dir(cam):
    S = SESSION
    base = (ROOT / f"raw_data/gopro_data/side_view_camera-mingle_session_{S}" if cam > 10
            else TU4 / f"raw/video/top_view_camera-mingle_session_{S}")
    return base / f"camera_{cam:02d}-mingle_session_{S}"


def ticks(tc):
    h, m, s, f = map(int, tc.split(":"))
    return ((h * 60 + m) * 60 + s) * 60 + f


def table(cam):
    """rows: (file, row, original tick, world tick) for every raw frame."""
    d = ts_dir(cam)
    out = []
    for f in sorted(d.glob("*.csv")):
        with open(f) as fh:
            r = csv.reader(fh)
            next(r)
            for i, row in enumerate(r):
                out.append((f.stem + ".MP4", i, ticks(row[0]), ticks(row[1])))
    return out


def decode_rows(path, rows):
    """Yield (row, grey full-resolution frame) for the given sorted rows of one file, one at a time."""
    r0, r1 = rows[0], rows[-1]
    w, h = 1920, 1080
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{r0 / FPS:.6f}", "-i", str(path), "-frames:v", str(r1 - r0 + 1),
           "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    want = set(rows)
    for r in range(r0, r1 + 1):
        buf = p.stdout.read(w * h)
        if len(buf) < w * h:
            break
        if r in want:
            yield r, np.frombuffer(buf, np.uint8).reshape(h, w)
    p.stdout.close()
    p.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=int, default=2)
    ap.add_argument("--cams", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--step", type=int, default=10)
    a = ap.parse_args()
    global SESSION
    SESSION = a.session
    cfg = SESSION_CFG[SESSION]
    by_orig = {o: w for _, _, o, w in table(cfg["anchor_cam"])}
    w0 = by_orig[cfg["start"] + cfg["anchor_offset"]]
    wend = by_orig[cfg["end"] + cfg["anchor_offset"]]
    grid = list(range(w0, wend + 1, a.step))
    outdir = WORK / "frames" / f"s{SESSION}dense"
    info = {"step": a.step, "world_ticks": grid, "cams": {}}
    for cam in a.cams:
        t = table(cam)
        world = np.array([x[3] for x in t])
        order = np.argsort(world, kind="stable")
        ws = world[order]
        pos = np.clip(np.searchsorted(ws, grid), 1, len(ws) - 1)
        pick = np.where(np.abs(ws[pos - 1] - grid) <= np.abs(ws[pos] - grid), pos - 1, pos)
        idx = order[pick]
        mism = (world[idx] - np.array(grid)).tolist()
        sel = [(k, t[i][0], t[i][1]) for k, i in enumerate(idx)]
        d = outdir / f"camera_{cam:02d}-mingle_session_{SESSION}"
        d.mkdir(parents=True, exist_ok=True)
        n = 0
        for fname in sorted({s[1] for s in sel}):
            rows = sorted(s[2] for s in sel if s[1] == fname)
            ks = defaultdict(list)
            for k, f, r in sel:
                if f == fname:
                    ks[r].append(k)
            for r, img in decode_rows(raw_dir(cam) / fname, rows):
                for k in ks[r]:
                    cv2.imwrite(str(d / f"{k:09d}.png"), img); n += 1
        info["cams"][cam] = {"file_row": [(f, r) for _, f, r in sel], "tick_mismatch": mism}
        print(f"cam{cam:02d}: {n}/{len(grid)} frames, world-tick mismatch max {max(map(abs, mism))}", flush=True)
    (outdir / f"grid_cam{'_'.join(map(str, a.cams))}.json").write_text(json.dumps(info))


if __name__ == "__main__":
    main()
