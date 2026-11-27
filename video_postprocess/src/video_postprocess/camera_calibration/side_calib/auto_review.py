"""Automatic review of the multi-board detections, alternated with the joint fit.

This is the automatic replacement of the manual rotate/discard pass (show_saved_detections.py). joint.py drives it.

For every frame:
  1. Boards seen by the top cameras (current poses): seed from the largest top detection (PnP), add the other top
     views that project within the top gate, refine the board pose on all its top views, drop views that disagree.
  2. Side cameras: project the boards with the current side poses and assign detections by Hungarian matching on
     the mean corner error, over the 4 corner orders (the 'rotate' of the manual GUI). Unmatched detections or
     errors above the gate are discarded (the 'discard').
  3. Side-only boards: seeded from a side detection with the current pose, kept if a second side camera agrees.
Then the joint fit (free camera poses + board poses) runs, and the review repeats with the new poses and tighter
gates. Top cameras listed in run(free_top=...) are solved too; the others stay fixed.

Standalone use (side cameras only, top cameras fixed, old calibration frames):
  python auto_review.py SESSION [--gates 150,60,25] [--loose-cams 12,14]   -> side_calib_work/s{N}/
"""
import argparse
import json
from collections import defaultdict

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from common import (SESSIONS, SIDE_CAL_ORIG, WORK, board_world_points, compose, corner_orders, invert,
                    load_session, pnp, pose_diff, project, reorder)
from joint_fit import Problem, camera_pose_from_boards, fit_board_pose

ORDERS = corner_orders()
OLD_PROJECTS = {
    1: ["mingling_1_side_Extrinsic Parameters/top_cam_06_side_cam_13_15.json",
        "mingling_1_side_Extrinsic Parameters/top_cam_10_side_cam_12_14.json"],
    2: ["mingling_2_side_Extrinsic Parameters/top_cam_01_side_cam_13_15.json",
        "mingling_2_side_Extrinsic Parameters/top_cam_04_side_cam_12_14.json"],
}


def sq_px(c):
    return float(np.median(np.linalg.norm(np.diff(c.reshape(3, 5, 2), axis=1), axis=2)))


def load_detections(session, use_old=True):
    """list of dict(cam, frame, corners, src). New multi-board detections plus the old Calibrator detections
    (deduplicated against the new ones)."""
    z = np.load(WORK / f"detections_s{session}.npz")
    dets = [dict(cam=int(c), frame=int(f), corners=x, src=str(v)) for c, f, v, x in
            zip(z["cam"], z["frame"], z["variant"], z["corners"])]
    n_old_added = 0
    if use_old:
        idx = defaultdict(list)
        for d in dets:
            idx[(d["cam"], d["frame"])].append(d)
        for proj in OLD_PROJECTS[session]:
            p = json.loads((SIDE_CAL_ORIG / "calibrator_projects" / proj).read_text())
            files = p["fileInfo"]
            cams = [int(x["filePath"].split("/")[-2].split("_")[1].split("-")[0]) for x in files[0]]
            for det in p["detections"]:
                cam = cams[det["cameraId"]]
                frame = int(files[det["poseId"]][det["cameraId"]]["filePath"].split("/")[-1].split(".")[0])
                c = np.array([[q["point"]["x"], q["point"]["y"]] for q in det["featurePoints"]])
                if len(c) != 15:
                    continue
                s = sq_px(c)
                if any(np.median(np.linalg.norm(c[:, None] - e["corners"][None], axis=2).min(1)) < 0.3 * s
                       for e in idx[(cam, frame)]):
                    continue
                d = dict(cam=cam, frame=frame, corners=c, src="old_project")
                dets.append(d); idx[(cam, frame)].append(d); n_old_added += 1
    return dets, n_old_added


def match_error(Xw, cam, det, cams):
    """Best corner order and mean px error of a detection against a world board."""
    c = cams[cam]
    proj = project(Xw, c["rvec"], c["tvec"], c["K"], c["D"])
    best = (None, np.inf)
    for name, perm in ORDERS.items():
        e = np.linalg.norm(reorder(det["corners"], perm) - proj, axis=1).mean()
        if e < best[1]:
            best = (name, e)
    return best


def seed_pose(det, cams):
    """Board world pose (6,) from one detection with a known camera pose (rot0 order)."""
    c = cams[det["cam"]]
    r_bc, t_bc = pnp(det["corners"], c["K"], c["D"])          # board -> cam
    r_cw, t_cw = invert(c["rvec"], c["tvec"])                 # cam -> world
    r, t = compose(r_cw, t_cw, r_bc, t_bc)
    return np.r_[r, t]


def review_frame(fdets, cams, top_cams, side_cams, gates, top_gate):
    """Returns boards: list of dict(pose(6,), views: list of (det_idx, order)) and discard reasons."""
    used = {}
    boards = []
    top = sorted([i for i, d in enumerate(fdets) if d["cam"] in top_cams], key=lambda i: -sq_px(fdets[i]["corners"]))
    for i in top:
        if i in used:
            continue
        pose = seed_pose(fdets[i], cams)
        views = {fdets[i]["cam"]: (i, "rot0")}
        for loose in (True, False):
            Xw = board_world_points(pose[:3], pose[3:])
            for j in top:
                cj = fdets[j]["cam"]
                if j in used or j == i or (cj in views and views[cj][0] != j):
                    continue
                order, e = match_error(Xw, cj, fdets[j], cams)
                gate = 3 * sq_px(fdets[j]["corners"]) + 20 if loose else top_gate
                if e < gate:
                    views[cj] = (j, order)
            vl = [(c, reorder(fdets[j]["corners"], ORDERS[o])) for c, (j, o) in views.items()]
            pose, err = fit_board_pose(vl, cams, pose)
            keep = {c: v for (c, v), e in zip(views.items(), err) if e < top_gate or c == fdets[i]["cam"]}
            if len(keep) < len(views):
                views = keep
                vl = [(c, reorder(fdets[j]["corners"], ORDERS[o])) for c, (j, o) in views.items()]
                pose, err = fit_board_pose(vl, cams, pose)
        for c, (j, o) in views.items():
            used[j] = len(boards)
        boards.append(dict(pose=pose, views=[(j, o) for j, o in views.values()], top_views=len(views)))
    # side cameras: Hungarian per camera
    for c in side_cams:
        cand = [i for i, d in enumerate(fdets) if d["cam"] == c and i not in used]
        if not cand or not boards:
            continue
        cost = np.full((len(boards), len(cand)), 1e6)
        orders = {}
        for b, bd in enumerate(boards):
            Xw = board_world_points(bd["pose"][:3], bd["pose"][3:])
            for k, i in enumerate(cand):
                o, e = match_error(Xw, c, fdets[i], cams)
                orders[(b, k)] = o
                if e < gates[c]:
                    cost[b, k] = e
        rr, cc = linear_sum_assignment(cost)
        for b, k in zip(rr, cc):
            if cost[b, k] < 1e6:
                used[cand[k]] = b
                boards[b]["views"].append((cand[k], orders[(b, k)]))
    # side-only boards
    rest = sorted([i for i, d in enumerate(fdets) if d["cam"] in side_cams and i not in used],
                  key=lambda i: -sq_px(fdets[i]["corners"]))
    for i in rest:
        if i in used:
            continue
        pose = seed_pose(fdets[i], cams)
        Xw = board_world_points(pose[:3], pose[3:])
        views = [(i, "rot0")]
        for c in side_cams:
            if c == fdets[i]["cam"]:
                continue
            best = (None, None, np.inf)
            for j in rest:
                if j in used or j == i or fdets[j]["cam"] != c:
                    continue
                o, e = match_error(Xw, c, fdets[j], cams)
                # the seed's depth comes from a single small view: allow a wider gate for the first match
                if e < best[2]:
                    best = (j, o, e)
            if best[0] is not None and best[2] < 2 * gates[c]:
                views.append((best[0], best[1]))
        if len(views) >= 2:
            for j, _ in views:
                used[j] = len(boards)
            boards.append(dict(pose=pose, views=views, top_views=0))
    return boards, used


def run_review(dets, cams, top_cams, side_cams, gates, top_gate, free_top=()):
    by_frame = defaultdict(list)
    for i, d in enumerate(dets):
        by_frame[d["frame"]].append(i)
    all_boards = []
    for f in sorted(by_frame):
        idx = by_frame[f]
        fd = [dets[i] for i in idx]
        boards, _ = review_frame(fd, cams, top_cams, side_cams, gates, top_gate)
        for b in boards:
            sides = [j for j, _ in b["views"] if fd[j]["cam"] in side_cams]
            free = [j for j, _ in b["views"] if fd[j]["cam"] in free_top]
            if len(b["views"]) < 2 or not (sides or free):
                continue
            all_boards.append(dict(frame=f, pose=b["pose"], top_views=b["top_views"],
                                   views=[(idx[j], o) for j, o in b["views"]]))
    return all_boards


def build_problem(boards, dets, cams, free):
    views = []
    for bi, b in enumerate(boards):
        for di, o in b["views"]:
            views.append((dets[di]["cam"], bi, reorder(dets[di]["corners"], ORDERS[o])))
    return Problem(views, cams, free, np.array([b["pose"] for b in boards]))


def fit(boards, dets, cams, free, drop_px, log):
    """Joint fit; views with mean error > drop_px are removed and the fit is repeated once.
    Updates cams[free] and the board poses in place; returns (boards, view errors of the final fit)."""
    for rnd in range(2):
        p = build_problem(boards, dets, cams, free)
        x = p.solve()
        cp, bp = p.unpack(x)
        for c in free:
            cams[c] = dict(cams[c], rvec=cp[c][:3].copy(), tvec=cp[c][3:].copy())
        for b, pose in zip(boards, bp):
            b["pose"] = pose.copy()
        err = p.view_errors(x)
        if rnd == 1:
            break
        vi, bad = 0, 0
        for b in boards:
            keep = []
            for v in b["views"]:
                if err[vi] <= drop_px:
                    keep.append(v)
                else:
                    bad += 1
                vi += 1
            b["views"] = keep
        boards = [b for b in boards if len(b["views"]) >= 2 and any(dets[d]["cam"] in free for d, _ in b["views"])]
        log(f"    fit: dropped {bad} views above {drop_px:.0f} px")
    return boards, err


def ransac_pose(cam, boards_top, dets, cams, inlier_px=15.0, log=print):
    """Pose of `cam` that ignores its current calibration: each (top board, detection in cam, corner order) of the
    same frame gives a PnP hypothesis; the score is the number of top boards with a detection within inlier_px."""
    c = cams[cam]
    by_frame = defaultdict(list)
    for i, d in enumerate(dets):
        if d["cam"] == cam:
            by_frame[d["frame"]].append(i)
    bl = [b for b in boards_top if by_frame.get(b["frame"])]
    if not bl:
        return None
    Xall = np.concatenate([board_world_points(b["pose"][:3], b["pose"][3:]) for b in bl])
    det_sets = [[dets[i]["corners"] for i in by_frame[b["frame"]]] for b in bl]

    def score(r, t):
        P = project(Xall, r, t, c["K"], c["D"]).reshape(len(bl), 15, 2)
        inl = []
        for k, (proj, ds) in enumerate(zip(P, det_sets)):
            best = min((np.linalg.norm(reorder(dc, perm) - proj, axis=1).mean(), j, o)
                       for j, dc in enumerate(ds) for o, perm in ORDERS.items())
            if best[0] < inlier_px:
                inl.append((k, best[1], best[2]))
        return inl

    best = (-1, None)
    for k, b in enumerate(bl):
        Xw = Xall[15 * k: 15 * k + 15]
        for dc in det_sets[k]:
            for o, perm in ORDERS.items():
                try:
                    r, t = camera_pose_from_boards([(Xw, reorder(dc, perm))], c["K"], c["D"])
                except Exception:
                    continue
                if ((Xw @ cv2.Rodrigues(r)[0].T + t)[:, 2]).min() <= 0:
                    continue
                n = len(score(r, t))
                if n > best[0]:
                    best = (n, (r, t))
    if best[1] is None:
        return None
    r, t = best[1]
    for _ in range(3):  # refine on all inliers
        inl = score(r, t)
        pairs = [(Xall[15 * k: 15 * k + 15], reorder(det_sets[k][j], ORDERS[o])) for k, j, o in inl]
        r, t = camera_pose_from_boards(pairs, c["K"], c["D"], init=(r, t))
    log(f"    ransac cam{cam:02d}: {len(score(r, t))} of {len(bl)} top boards within {inlier_px:.0f} px")
    return r, t


def summarize(boards, dets, err, cams, stored, side_cams, log):
    vi = 0
    per = defaultdict(list)
    for b in boards:
        for d, _ in b["views"]:
            per[dets[d]["cam"]].append((err[vi], b["frame"], b["top_views"]))
            vi += 1
    for c in sorted(per):
        e = np.array([x[0] for x in per[c]])
        line = f"    cam{c:02d}: {len(e):4d} views in {len({x[1] for x in per[c]}):3d} frames, " \
               f"mean corner err median {np.median(e):.2f} / p90 {np.percentile(e, 90):.2f} px"
        if c in side_cams:
            dm, da = pose_diff(cams[c]["rvec"], cams[c]["tvec"], *stored[c])
            line += f"; moved {dm:.3f} m / {da:.2f} deg from stored"
        log(line)


def save(session, boards, dets, cams, side_cams, err, out):
    out.mkdir(parents=True, exist_ok=True)
    rows, vi = [], 0
    for bi, b in enumerate(boards):
        vs = []
        for d, o in b["views"]:
            dd = dets[d]
            vs.append(dict(cam=dd["cam"], order=o, src=dd["src"], err_px=round(float(err[vi]), 3),
                           corners=np.round(dd["corners"], 3).tolist()))
            vi += 1
        rows.append(dict(board=bi, frame=b["frame"], top_views=b["top_views"],
                         pose=np.round(b["pose"], 6).tolist(), views=vs))
    (out / "reviewed.json").write_text(json.dumps(rows))
    for c in side_cams:
        d = out / "idiap_params" / f"mingling_{session}_side" / f"camera{SESSIONS[session]['sep']}{c:02d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "extrinsic.json").write_text(json.dumps(
            {"rvec": [[float(v)] for v in cams[c]["rvec"]], "tvec": [[float(v)] for v in cams[c]["tvec"]]}, indent=1))


def run(session, gates_seq, loose, tight_gate, top_gate, init, out, log, exclude_frames=(), dets=None, cams=None,
        stored=None, free_top=(), top_gates=None):
    """free_top: top cameras solved together with the side cameras (the others stay fixed);
    top_gates: per-iteration top-view gate (px), loose first when free top cameras start far off."""
    cams = load_session(session) if cams is None else cams
    top_cams = [c for c in SESSIONS[session]["top"] if cams[c]["rvec"] is not None]
    side_cams = SESSIONS[session]["side"]
    if stored is None:
        stored = {c: (cams[c]["rvec"].copy(), cams[c]["tvec"].copy()) for c in side_cams + list(free_top)}
    if dets is None:
        dets, n_old = load_detections(session)
    else:
        n_old = 0
    dets = [d for d in dets if d["frame"] not in set(exclude_frames)]
    log(f"session {session}: {len(dets)} detections ({n_old} only in the old projects); top {top_cams}, side {side_cams}")
    for c in top_cams + side_cams:
        log(f"  cam{c:02d}: {sum(d['cam'] == c for d in dets)} detections")
    if init == "ransac":
        # boards seen by >= 2 top cameras (independent of the side calibration)
        bt = []
        by_frame = defaultdict(list)
        for i, d in enumerate(dets):
            by_frame[d["frame"]].append(i)
        for f, idx in by_frame.items():
            fd = [dets[i] for i in idx]
            bs, _ = review_frame(fd, cams, top_cams, [], {}, top_gate)
            bt += [dict(frame=f, pose=b["pose"]) for b in bs if b["top_views"] >= 2]
        log(f"  ransac init for {loose} from {len(bt)} boards with >= 2 top views")
        for c in loose:
            rp = ransac_pose(c, bt, dets, cams, log=log)
            if rp is not None:
                dm, da = pose_diff(*rp, *stored[c])
                log(f"    cam{c:02d} ransac start is {dm:.3f} m / {da:.2f} deg from stored")
                cams[c] = dict(cams[c], rvec=rp[0], tvec=rp[1])
    boards, err = [], None
    for it, g in enumerate(gates_seq):
        gates = {c: (g if c in loose else min(g, tight_gate)) for c in side_cams}
        tg = top_gates[it] if top_gates else top_gate
        boards = run_review(dets, cams, top_cams, side_cams, gates, tg, free_top)
        n_side = sum(1 for b in boards for d, _ in b["views"] if dets[d]["cam"] in side_cams)
        log(f"  iteration {it + 1}, gates {gates}, top gate {tg}: {len(boards)} boards, {n_side} side views")
        if not boards:
            raise SystemExit("no boards with side views")
        boards, err = fit(boards, dets, cams, side_cams + list(free_top), max(g / 5, 4.0), log)
        summarize(boards, dets, err, cams, stored, side_cams + list(free_top), log)
    if out is not None:
        save(session, boards, dets, cams, side_cams, err, out)
    return cams, boards, dets, err, stored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=int)
    ap.add_argument("--gates", default="150,60,25")
    ap.add_argument("--loose-cams", default="")
    ap.add_argument("--tight-gate", type=float, default=40.0)
    ap.add_argument("--top-gate", type=float, default=6.0)
    ap.add_argument("--init", choices=["stored", "ransac"], default="stored")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    loose = [int(c) for c in a.loose_cams.split(",") if c]
    out = WORK / f"s{a.session}{('_' + a.tag) if a.tag else ''}"
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "review.log", "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n"); logf.flush()

    log(f"args: {vars(a)}")
    run(a.session, [float(g) for g in a.gates.split(",")], loose, a.tight_gate, a.top_gate, a.init, out, log)


if __name__ == "__main__":
    main()
