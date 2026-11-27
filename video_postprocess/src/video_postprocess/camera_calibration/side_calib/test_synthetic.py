"""Synthetic check of auto_review + joint_fit with the real session-1 cameras (no images involved)."""
import numpy as np, cv2
from common import load_session, SESSIONS, board_world_points, project, corner_orders, pose_diff, compose
import auto_review as ar

rng = np.random.default_rng(0)
truth = load_session(1)
side = SESSIONS[1]["side"]; top = [c for c in SESSIONS[1]["top"] if truth[c]["rvec"] is not None]
orders = list(corner_orders().values())
# room extent from the top camera centres
C = np.array([-cv2.Rodrigues(truth[c]["rvec"])[0].T @ truth[c]["tvec"] for c in top])
lo, hi = C.min(0) - 0.5, C.max(0) + 0.5
dets, nb = [], 0
for f in range(120):
    for k in range(3):
        # board on a ray of a random top camera, 2-2.7 m away, roughly facing it
        c0 = top[rng.integers(len(top))]; cc0 = truth[c0]
        ray = np.r_[rng.uniform(-0.8, 0.8, 2), 1.0]
        Rc = cv2.Rodrigues(cc0["rvec"])[0]
        centre = Rc.T @ (ray / np.linalg.norm(ray) * rng.uniform(2.0, 2.7) - cc0["tvec"])
        R_face = Rc.T @ cv2.Rodrigues(np.r_[rng.normal(0, 0.5, 2), rng.uniform(-np.pi, np.pi)])[0]
        r_b = cv2.Rodrigues(R_face)[0].ravel(); t_b = centre
        Xw = board_world_points(r_b, t_b)
        for c in top + side:
            cc = truth[c]
            Xc = Xw @ cv2.Rodrigues(cc["rvec"])[0].T + cc["tvec"]
            if Xc[:, 2].min() < 0.3:
                continue
            px = project(Xw, cc["rvec"], cc["tvec"], cc["K"], cc["D"])
            if (px < 0).any() or (px[:, 0] > 1920).any() or (px[:, 1] > 1080).any():
                continue
            # board must face the camera reasonably
            n = cv2.Rodrigues(r_b)[0][:, 2]; v = Xw.mean(0) + cv2.Rodrigues(cc["rvec"])[0].T @ cc["tvec"]
            if abs(n @ (v / np.linalg.norm(v))) < 0.3 or rng.random() < 0.5:
                continue
            perm = orders[rng.integers(4)]
            obs = (px + rng.normal(0, 0.3, px.shape))[perm]  # detected order: corner k = board id perm[k]
            dets.append(dict(cam=c, frame=f, corners=obs, src="syn", truth=(f, k)))
# swap: some side detections replaced by a fake board 40 px away
n_fake = 0
for d in list(dets):
    if d["cam"] in side and rng.random() < 0.05:
        dets.append(dict(cam=d["cam"], frame=d["frame"], corners=d["corners"] + rng.normal(0, 60, 2), src="fake", truth=None)); n_fake += 1
print("synthetic detections", len(dets), "fakes", n_fake, {c: sum(d['cam'] == c for d in dets) for c in top + side})
cams = {c: dict(v) for c, v in truth.items()}
stored = {}
for c in (12, 14):  # perturb: ~0.5 m, ~8 deg
    r, t = compose(rng.normal(0, 0.08, 3), rng.normal(0, 0.3, 3), cams[c]["rvec"], cams[c]["tvec"])
    cams[c] = dict(cams[c], rvec=r, tvec=t)
for c in side:
    stored[c] = (cams[c]["rvec"].copy(), cams[c]["tvec"].copy())
    print(f"start cam{c}: {pose_diff(*stored[c], truth[c]['rvec'], truth[c]['tvec'])}")
for init in ("stored", "ransac"):
    cc = {c: dict(v) for c, v in cams.items()}
    out, boards, D, err, _ = ar.run(1, [150, 60, 25], [12, 14], 40, 6, init, None, print, dets=dets, cams=cc, stored=stored)
    for c in side:
        print(f"[{init}] cam{c} vs truth: %.4f m, %.3f deg" % pose_diff(out[c]["rvec"], out[c]["tvec"], truth[c]["rvec"], truth[c]["tvec"]))
    used_fake = sum(1 for b in boards for d, _ in b["views"] if D[d]["src"] == "fake")
    print(f"[{init}] fakes kept: {used_fake}")
