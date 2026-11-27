"""Joint calibration from dense world-synced frames, per session.

Only the world-origin camera stays at its 4TU pose; every other top and side camera is solved together:
  session 1: cam06 fixed; cams 07-10 and 12-15 solved,
  session 2: cam01 fixed; cams 02-04 and 12-15 solved (cam05 is never used: bad time sync).
All intrinsics are fixed.
Data: dense frames (every 10th tick, extract_dense.py + detect_multi.py --subdir s{N}dense), plus the old
calibration frames (detect_multi.py N) and the old Calibrator corners, with frame ids + 10000.

Evaluation: 5 contiguous blocks of dense frames are held out in turn. A held-out board is posed from all its other
views (each camera in turn left out) and projected into the left-out camera, with (a) the 4TU/stored poses and
(b) the fold's poses. Final poses come from one fit on all data.

Usage: python joint.py SESSION [--folds 5] [--out NAME]     (--folds 0: final fit only, no evaluation)
Output: side_calib_work/{NAME or s{N}joint}/{review.log, eval.log, reviewed.json, idiap_params/mingling_N_{top,side}/}
"""
import argparse
import json
from collections import defaultdict

import numpy as np

import auto_review as ar
from common import SESSIONS, WORK, board_world_points, load_session, pose_diff, project, reorder
from joint_fit import fit_board_pose

SIDE = [12, 13, 14, 15]
GATES = [150, 60, 25, 25]
OLD = 10000
# old calibration frame i happens at dense frame OLD_A + OLD_B * i (measured by pixel difference)
CFG = {
    1: dict(free_top=[7, 8, 9, 10], top_gates=[60, 25, 10, 6], n_dense=1598, old_a=-1.0, old_b=6.0, n_old=266),
    2: dict(free_top=[2, 3, 4], top_gates=[60, 25, 10, 6], n_dense=998, old_a=0.0, old_b=3.003, n_old=332),
}
S = 2


def load_all():
    dets = []
    one = WORK / f"detections_s{S}dense.npz"
    names = [one] if one.exists() else [WORK / f"detections_s{S}dense_top.npz", WORK / f"detections_s{S}dense_side.npz"]
    for name in names:
        z = np.load(name)
        dets += [dict(cam=int(c), frame=int(f), corners=x, src=f"dense_{v}") for c, f, v, x in
                 zip(z["cam"], z["frame"], z["variant"], z["corners"]) if int(c) != 5]
    old, _ = ar.load_detections(S)
    dets += [dict(d, frame=d["frame"] + OLD) for d in old if d["cam"] != 5]
    return dets


def run_fit(dets, log, exclude=()):
    """exclude: dense frame ids; the old frames taken at the same moments (old i ~ dense old_a + old_b i) go too."""
    c = CFG[S]
    if exclude:
        lo, hi = min(exclude), max(exclude)
        exclude = set(exclude) | {OLD + i for i in range(c["n_old"]) if lo - 3 <= c["old_a"] + c["old_b"] * i <= hi + 3}
    cams = load_session(S)
    return ar.run(S, GATES, SIDE, 40, 6, "stored", None, log, exclude_frames=exclude, dets=dets, cams=cams,
                  free_top=c["free_top"], top_gates=c["top_gates"])


def save(cams, boards, dets, err, out):
    out.mkdir(parents=True, exist_ok=True)
    for c in CFG[S]["free_top"] + SIDE:
        kind = "top" if c < 10 else "side"
        d = out / "idiap_params" / f"mingling_{S}_{kind}" / f"camera{SESSIONS[S]['sep']}{c:02d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "extrinsic.json").write_text(json.dumps(
            {"rvec": [[float(v)] for v in cams[c]["rvec"]], "tvec": [[float(v)] for v in cams[c]["tvec"]]}, indent=1))
    rows, vi = [], 0
    for b in boards:
        vs = []
        for di, o in b["views"]:
            dd = dets[di]
            vs.append(dict(cam=dd["cam"], order=o, src=dd["src"], err_px=round(float(err[vi]), 3),
                           corners=np.round(dd["corners"], 3).tolist()))
            vi += 1
        rows.append(dict(frame=b["frame"], pose=np.round(b["pose"], 6).tolist(), views=vs))
    (out / "reviewed.json").write_text(json.dumps(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=int)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default=None, help="output folder name in side_calib_work (default s{SESSION}joint)")
    a = ap.parse_args()
    global S
    S = a.session
    FREE_TOP = CFG[S]["free_top"]
    TOP = [c for c in SESSIONS[S]["top"] if c != 5]
    out = WORK / (a.out or f"s{S}joint")
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "review.log", "w")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n"); logf.flush()

    dets = load_all()
    cams, boards, D, err, _ = run_fit(dets, log)
    save(cams, boards, D, err, out)
    st = load_session(S)

    elog = open(out / "eval.log", "w")

    def ev(m):
        print(m, flush=True)
        elog.write(m + "\n"); elog.flush()

    ev("final vs 4TU/stored pose change")
    for c in FREE_TOP + SIDE:
        ev(f"  cam{c:02d}: %.3f m, %.2f deg" % pose_diff(cams[c]["rvec"], cams[c]["tvec"], st[c]["rvec"], st[c]["tvec"]))
    ev("views per camera in the final fit: " + str({c: sum(1 for b in boards for d, _ in b["views"] if D[d]["cam"] == c)
                                                     for c in TOP + SIDE}))
    if a.folds < 2:
        return

    # held-out evaluation over blocks of dense frames
    final_boards = [dict(frame=b["frame"], views=[(D[d]["cam"], reorder(D[d]["corners"], ar.ORDERS[o]))
                                                  for d, o in b["views"]], pose=b["pose"]) for b in boards]
    n_dense = CFG[S]["n_dense"] or (max(d["frame"] for d in dets if d["frame"] < OLD) + 1)
    dense_frames = np.arange(n_dense)
    held = {k: defaultdict(list) for k in ("4TU", "fold")}
    held2 = {k: defaultdict(list) for k in ("4TU", "fold")}
    for k, blk in enumerate(np.array_split(dense_frames, a.folds)):
        bset = set(blk.tolist())
        fc, _, _, _, _ = run_fit(dets, lambda m: None, exclude=bset)
        spread = {c: pose_diff(fc[c]["rvec"], fc[c]["tvec"], cams[c]["rvec"], cams[c]["tvec"]) for c in FREE_TOP + SIDE}
        ev(f"fold {k + 1}: frames {blk[0]}..{blk[-1]} held out; fold vs final: " +
           ", ".join(f"cam{c:02d} {m:.3f} m/{d:.2f} deg" for c, (m, d) in spread.items()))
        for b in final_boards:
            if b["frame"] not in bset or len(b["views"]) < 2:
                continue
            for i, (c, obs) in enumerate(b["views"]):
                rest = [v for j, v in enumerate(b["views"]) if j != i]
                for name, cc in (("4TU", st), ("fold", fc)):
                    pose, e = fit_board_pose(rest, cc, b["pose"])
                    if e.max() > 10:
                        continue
                    Xw = board_world_points(pose[:3], pose[3:])
                    r = float(np.linalg.norm(project(Xw, cc[c]["rvec"], cc[c]["tvec"], cc[c]["K"], cc[c]["D"]) - obs,
                                             axis=1).mean())
                    held[name][c].append(r)
                    if len(rest) >= 2:
                        held2[name][c].append(r)
    ev("\nheld-out boards: each camera scored against the board posed from its other views")
    ev("  cam  | 4TU/stored: n, median, p90 | new: n, median, p90 | boards with >= 2 other views: 4TU median, new median")
    for c in TOP + SIDE:
        a4, af = np.array(held["4TU"][c]), np.array(held["fold"][c])
        b4, bf = np.array(held2["4TU"][c]), np.array(held2["fold"][c])
        f = lambda v: f"{len(v):4d} {np.median(v):6.1f} {np.percentile(v, 90):6.1f}" if len(v) else "   0      -      -"
        g = lambda v: f"{np.median(v):6.1f}" if len(v) else "     -"
        ev(f"  {c:02d}   | {f(a4)} | {f(af)} | n {len(bf):3d}: {g(b4)} {g(bf)}")


if __name__ == "__main__":
    main()
