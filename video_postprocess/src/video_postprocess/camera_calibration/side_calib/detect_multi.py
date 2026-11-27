"""Detect every 5x3 board in each calibration frame (numbers only, nothing is displayed).

Same detector as detect_checkerboard.py (findChessboardCorners(5,3) + cornerSubPix 3x3), run on four versions of
each image (plain, CLAHE, 2x upscaled, both). After each hit the board's hull, grown by about 1.5 squares, is
filled with the image median and the search is repeated. Boards found by several versions are kept once.

Usage: python detect_multi.py SESSION [--workers 30] [--max-boards 4]
Output: side_calib_work/detections_s{SESSION}.npz  (cam, frame, k, variant, corners[15,2])
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from common import SESSIONS, WORK

CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def mask_board(gray, corners, fill):
    c = corners.reshape(-1, 2)
    g = c.reshape(3, 5, 2)
    sq = np.mean(np.r_[np.linalg.norm(np.diff(g, axis=1), axis=2).ravel(),
                       np.linalg.norm(np.diff(g, axis=0), axis=2).ravel()])
    hull = cv2.convexHull(c.astype(np.float32)).reshape(-1, 2)
    ctr = c.mean(0)
    d = hull - ctr
    n = np.linalg.norm(d, axis=1, keepdims=True)
    grown = ctr + d * (n + 1.5 * sq) / np.maximum(n, 1e-6)
    cv2.fillConvexPoly(gray, np.round(grown).astype(np.int32), int(fill))


def variants(img):
    """Preprocessed copies of the image with their scale. Each finds a different subset of the small boards."""
    cl = cv2.createCLAHE(3.0, (8, 8)).apply(img)
    yield "default", img, 1
    yield "clahe", cl, 1
    yield "up2", cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), 2
    yield "clahe_up2", cv2.resize(cl, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), 2


def same_board(a, b):
    """True if two detections cover the same board (corner sets coincide up to order)."""
    sq = np.median(np.linalg.norm(np.diff(a.reshape(3, 5, 2), axis=1), axis=2))
    return np.median(np.linalg.norm(a[:, None] - b[None], axis=2).min(1)) < 0.3 * sq


def detect_file(args):
    path, max_boards = args
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return path, None
    found = []  # (variant, corners)
    for name, im, s in variants(img):
        fill = int(np.median(im))
        work = im.copy()
        for _ in range(max_boards):
            ok, corners = cv2.findChessboardCorners(work, (5, 3), None)
            if not ok:
                break
            mask_board(work, corners, fill)
            corners = cv2.cornerSubPix(im, corners, (3 * s, 3 * s), (-1, -1), CRIT).reshape(-1, 2) / s
            if not any(same_board(corners, f) for _, f in found):
                found.append((name, corners))
    return path, found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=int)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--max-boards", type=int, default=4)
    ap.add_argument("--cams", type=int, nargs="*")
    ap.add_argument("--limit", type=int, default=None, help="only the first N frames per camera (testing)")
    ap.add_argument("--subdir", default=None, help="frames subfolder (default s{SESSION}), e.g. s2dense")
    a = ap.parse_args()
    cams = a.cams or SESSIONS[a.session]["top"] + SESSIONS[a.session]["side"]
    jobs = []
    for c in cams:
        sub = a.subdir or f"s{a.session}"
        files = sorted((WORK / "frames" / sub / f"camera_{c:02d}-mingle_session_{a.session}").glob("*.png"))[: a.limit]
        jobs += [(c, int(f.stem), f) for f in files]
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        res = ex.map(detect_file, [(f, a.max_boards) for _, _, f in jobs], chunksize=2)
        for (c, i, _), (_, found) in zip(jobs, res):
            if found is None:
                print(f"unreadable: cam{c:02d} frame {i}", flush=True)
                continue
            rows += [(c, i, k, v, x) for k, (v, x) in enumerate(found)]
    cam = np.array([r[0] for r in rows]); fr = np.array([r[1] for r in rows]); k = np.array([r[2] for r in rows])
    variant = np.array([r[3] for r in rows]); corners = np.array([r[4] for r in rows]).reshape(-1, 15, 2)
    name = a.subdir or f"s{a.session}"
    out = WORK / (f"detections_{name}.npz" if a.limit is None else f"detections_{name}_test.npz")
    np.savez(out, cam=cam, frame=fr, k=k, variant=variant, corners=corners)
    print(f"session {a.session}: {len(jobs)} images in {time.time() - t0:.0f} s -> {out}")
    for c in cams:
        m = cam == c
        nimg = len({i for cc, i, _ in jobs if cc == c})
        by_var = {v: int((variant[m] == v).sum()) for v in ("default", "clahe", "up2", "clahe_up2")}
        print(f"  cam{c:02d}: {m.sum():4d} boards in {nimg} images ({len(set(fr[m]))} with any); first found by {by_var}")


if __name__ == "__main__":
    main()
