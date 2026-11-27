"""Bundle adjustment of side-camera poses and board poses; top cameras and all intrinsics are fixed.

Unknowns: world->camera pose of every free camera, world pose of every board (6 DoF each).
Residuals: fisheye reprojection of the 15 corners of every (board, view), in px.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from common import OBJ


def rodrigues_batch(r):
    """Nx3 rotation vectors -> Nx3x3 matrices."""
    th = np.linalg.norm(r, axis=1)[:, None, None]
    k = r / np.maximum(th[:, :, 0], 1e-12)
    Kx = np.zeros((len(r), 3, 3))
    Kx[:, 0, 1], Kx[:, 0, 2], Kx[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    Kx[:, 1, 0], Kx[:, 2, 0], Kx[:, 2, 1] = k[:, 2], -k[:, 1], k[:, 0]
    I = np.eye(3)[None]
    return I + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def fisheye_px(Xc, K, D):
    """Camera-frame points (...,3) -> px with the OpenCV fisheye model; K, D broadcast over leading dims."""
    z = Xc[..., 2]
    a, b = Xc[..., 0] / z, Xc[..., 1] / z
    r = np.sqrt(a * a + b * b)
    th = np.arctan(r)
    th2 = th * th
    thd = th * (1 + D[..., 0:1] * th2 + D[..., 1:2] * th2 ** 2 + D[..., 2:3] * th2 ** 3 + D[..., 3:4] * th2 ** 4)
    s = np.where(r > 1e-9, thd / np.maximum(r, 1e-12), 1.0)
    u = K[..., 0, 0:1] * s * a + K[..., 0, 1:2] * s * b + K[..., 0, 2:3]
    v = K[..., 1, 1:2] * s * b + K[..., 1, 2:3]
    return np.stack([u, v], -1)


class Problem:
    """views: list of (cam, board_idx, corners_by_id[15,2]).
    cams: dict cam -> dict(K, D, rvec, tvec); free_cams: cams whose pose is optimized.
    boards: (B,6) initial world poses (rvec, tvec) of the boards (board -> world)."""

    def __init__(self, views, cams, free_cams, boards):
        self.cams, self.free = cams, list(free_cams)
        self.cam_ids = sorted({v[0] for v in views})
        self.vc = np.array([v[0] for v in views])
        self.vb = np.array([v[1] for v in views])
        self.obs = np.array([v[2] for v in views])
        self.K = np.array([cams[c]["K"] for c in self.vc])
        self.D = np.array([cams[c]["D"] for c in self.vc])
        self.nb = len(boards)
        self.fidx = {c: i for i, c in enumerate(self.free)}
        self.x0 = np.r_[np.concatenate([np.r_[cams[c]["rvec"], cams[c]["tvec"]] for c in self.free]) if self.free else [],
                        np.asarray(boards, float).ravel()]
        self.fixed_pose = {c: np.r_[cams[c]["rvec"], cams[c]["tvec"]] for c in self.cam_ids if c not in self.fidx}
        self.weights = np.ones(len(views))

    def unpack(self, x):
        nf = 6 * len(self.free)
        cp = {c: x[6 * i: 6 * i + 6] for c, i in self.fidx.items()}
        cp.update(self.fixed_pose)
        return cp, x[nf:].reshape(-1, 6)

    def project_all(self, x):
        cp, bp = self.unpack(x)
        cpose = np.array([cp[c] for c in self.vc])
        b = bp[self.vb]
        Rb = rodrigues_batch(b[:, :3])
        Xw = np.einsum("vij,kj->vki", Rb, OBJ) + b[:, None, 3:]
        Rc = rodrigues_batch(cpose[:, :3])
        Xc = np.einsum("vij,vkj->vki", Rc, Xw) + cpose[:, None, 3:]
        return fisheye_px(Xc, self.K, self.D), Xc[..., 2]

    def residuals(self, x):
        px, z = self.project_all(x)
        r = (px - self.obs) * self.weights[:, None, None]
        r = np.where(z[..., None] > 0, r, 1e3)
        return r.ravel()

    def sparsity(self):
        n_res, n_par = len(self.vc) * 30, len(self.x0)
        A = lil_matrix((n_res, n_par), dtype=int)
        nf = 6 * len(self.free)
        for i, (c, b) in enumerate(zip(self.vc, self.vb)):
            rows = slice(30 * i, 30 * i + 30)
            if c in self.fidx:
                j = 6 * self.fidx[c]
                A[rows, j:j + 6] = 1
            A[rows, nf + 6 * b: nf + 6 * b + 6] = 1
        return A

    def solve(self, f_scale=2.0, max_nfev=200, x0=None):
        res = least_squares(self.residuals, self.x0 if x0 is None else x0, jac_sparsity=self.sparsity(),
                            loss="soft_l1", f_scale=f_scale, x_scale="jac", max_nfev=max_nfev, method="trf")
        return res.x

    def view_errors(self, x):
        """Mean corner error (px) of every view."""
        px, _ = self.project_all(x)
        return np.linalg.norm(px - self.obs, axis=2).mean(1)


def fit_board_pose(view_list, cams, x0):
    """Board world pose from views in fixed cameras. view_list: [(cam, corners_by_id)]; x0: (6,) start."""
    vs = [(c, 0, o) for c, o in view_list]
    p = Problem(vs, cams, [], np.asarray(x0)[None])
    x = p.solve(f_scale=2.0, max_nfev=50)
    return x, p.view_errors(x)


def camera_pose_from_boards(pairs, K, D, init=None):
    """World->camera pose from (world corners[15,3], px[15,2]) pairs via fisheye-undistort + PnP."""
    X = np.concatenate([p[0] for p in pairs])
    n = cv2.fisheye.undistortPoints(np.concatenate([p[1] for p in pairs]).reshape(-1, 1, 2), K, D).reshape(-1, 2)
    if init is None:
        ok, r, t = cv2.solvePnP(X, n, np.eye(3), None, flags=cv2.SOLVEPNP_SQPNP)
    else:
        r, t = np.asarray(init[0], float).reshape(3, 1), np.asarray(init[1], float).reshape(3, 1)
    ok, r, t = cv2.solvePnP(X, n, np.eye(3), None, r, t, True, flags=cv2.SOLVEPNP_ITERATIVE)
    return r.ravel(), t.ravel()
