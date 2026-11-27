"""Shared paths, camera loading and fisheye geometry for the multi-board side-camera calibration."""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/tudelft.net/staff-umbrella/neon/cosilab_project")
TU4 = Path("/tudelft.net/staff-umbrella/CosiLabExternal/4TU_upload")  # moved from ROOT/4TU_upload (2026-11-26)
SIDE_CAL = ROOT / "processed_data/gopro_data/side_view_camera_calibration"
TOP_CAL = TU4 / "processed/videos/camera_calibration"
WORK = Path(__file__).resolve().parents[4] / "side_calib_work"

SESSIONS = {
    1: dict(top=[6, 7, 8, 9, 10], side=[12, 13, 14, 15], sep="-"),
    2: dict(top=[1, 2, 3, 4, 5], side=[12, 13, 14, 15], sep="_"),
}

# 5x3 inner corners, 9 cm squares; ids row-major, as in the Calibrator target.
COLS, ROWS, SQ = 5, 3, 0.09
OBJ = np.array([[c * SQ, r * SQ, 0.0] for r in range(ROWS) for c in range(COLS)])


def frames_dir(session, cam):
    return WORK / "frames" / f"s{session}" / f"camera_{cam:02d}-mingle_session_{session}"


def load_intrinsics(cam):
    base = TOP_CAL if cam < 11 else SIDE_CAL
    d = json.loads((base / "idiap_params/intrinsics" / f"camera_{cam:02d}/intrinsic.json").read_text())
    return np.array(d["intrinsic"], float), np.array(d["distortion_coefficients"], float)


def ext_dirname(session, cam):
    return f"camera{SESSIONS[session]['sep']}{cam:02d}"


def load_extrinsics(session, cam):
    """World->camera (rvec, tvec); None if the camera has no extrinsics (cam05)."""
    if cam < 11:
        p = TOP_CAL / f"idiap_params/mingling_{session}_top" / ext_dirname(session, cam) / "extrinsic.json"
    else:
        p = SIDE_CAL / f"idiap_params/mingling_{session}_side" / ext_dirname(session, cam) / "extrinsic.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return np.array(d["rvec"], float).ravel(), np.array(d["tvec"], float).ravel()


def load_session(session):
    """dict cam -> dict(K, D, rvec, tvec) for every camera with intrinsics; rvec None if unknown."""
    cams = {}
    for c in SESSIONS[session]["top"] + SESSIONS[session]["side"]:
        K, D = load_intrinsics(c)
        e = load_extrinsics(session, c)
        cams[c] = dict(K=K, D=D, rvec=None if e is None else e[0], tvec=None if e is None else e[1])
    return cams


def project(X, rvec, tvec, K, D):
    """Fisheye projection of Nx3 world points."""
    p, _ = cv2.fisheye.projectPoints(np.asarray(X, float).reshape(-1, 1, 3), np.asarray(rvec, float).reshape(3, 1),
                                     np.asarray(tvec, float).reshape(3, 1), K, D)
    return p.reshape(-1, 2)


def undistort(px, K, D):
    """Fisheye-undistort to normalized coordinates (Nx2)."""
    return cv2.fisheye.undistortPoints(np.asarray(px, float).reshape(-1, 1, 2), K, D).reshape(-1, 2)


def corner_orders():
    """Index permutations mapping detected corner order (rows of 5, as findChessboardCorners(5,3) returns
    them) to board ids. The board (6x4 squares) looks the same after a 180 deg turn, and OpenCV may also start
    a row or the rows at either end, so all 4 grid symmetries are candidates. perm[k] = board id of corner k.
    The mirrored ones ('flip_x', 'flip_y') correspond to the board seen from its back; cross-camera
    consistency picks the right one.
    """
    g = np.arange(ROWS * COLS).reshape(ROWS, COLS)
    return {"rot0": g.ravel(), "rot180": np.rot90(g, 2).ravel(),
            "flip_x": g[:, ::-1].ravel(), "flip_y": g[::-1, :].ravel()}


def reorder(corners, perm):
    """Return corners indexed by board id."""
    out = np.empty_like(corners)
    out[perm] = corners
    return out


def pnp(corners_by_id, K, D):
    """Board->camera pose from 15 corners in board-id order."""
    n = undistort(corners_by_id, K, D)
    ok, r, t = cv2.solvePnP(OBJ, n, np.eye(3), None, flags=cv2.SOLVEPNP_IPPE)
    ok, r, t = cv2.solvePnP(OBJ, n, np.eye(3), None, r, t, True, flags=cv2.SOLVEPNP_ITERATIVE)
    return r.ravel(), t.ravel()


def compose(r_a, t_a, r_b, t_b):
    """(a o b): apply b first, then a. Poses as x' = R x + t."""
    Ra, Rb = cv2.Rodrigues(r_a)[0], cv2.Rodrigues(r_b)[0]
    return cv2.Rodrigues(Ra @ Rb)[0].ravel(), Ra @ t_b + t_a


def invert(r, t):
    R = cv2.Rodrigues(np.asarray(r, float))[0]
    return cv2.Rodrigues(R.T)[0].ravel(), -R.T @ t


def board_world_points(r_bw, t_bw):
    R = cv2.Rodrigues(np.asarray(r_bw, float))[0]
    return OBJ @ R.T + t_bw


def cam_center(r, t):
    R = cv2.Rodrigues(np.asarray(r, float))[0]
    return -R.T @ np.asarray(t, float)


def pose_diff(r1, t1, r2, t2):
    """Camera-centre distance (m) and relative rotation angle (deg) between two world->cam poses."""
    R1, R2 = cv2.Rodrigues(np.asarray(r1, float))[0], cv2.Rodrigues(np.asarray(r2, float))[0]
    ang = np.degrees(np.linalg.norm(cv2.Rodrigues(R1 @ R2.T)[0]))
    return float(np.linalg.norm(cam_center(r1, t1) - cam_center(r2, t2))), float(ang)
