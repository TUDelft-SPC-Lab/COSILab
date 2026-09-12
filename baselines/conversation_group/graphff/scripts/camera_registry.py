#!/usr/bin/env python3
"""Camera identity for the Mingling benchmark, shared by the matrix scripts.

Deliberately stdlib-only and Python 3.7 compatible: the cross-camera evaluation
scripts import this *inside* the Apptainer image, and DANTE's environment
(``dante_tf1``) has no pandas, so it cannot import ``camera_matrix``.

``camera_matrix`` imports the two tables from here, so the camera list and the
session mapping have one definition rather than several.
"""

# flat camera order: mingling1 first, then mingling2. Cameras are compared
# camera against camera, so a session is a contiguous block of this order rather
# than an aggregation level of its own.
CAMERA_ORDER = ("cam06", "cam08", "cam10", "cam01", "cam03")
CAMERA_SESSION = {
    "cam06": "mingling1",
    "cam08": "mingling1",
    "cam10": "mingling1",
    "cam01": "mingling2",
    "cam03": "mingling2",
}

NUM_FOLDS = 5


def normalise_camera(value):
    """'6' | 'cam6' | 'mingling1/cam06' | 'mingling1_cam08' -> 'cam06'.

    Returns None when the value names no camera, so callers can decide whether
    that is an error. Zero-padding keeps 1 (cam01, mingling2) distinct from 10
    (cam10, mingling1).
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.rsplit("/", 1)[-1]
    if not text.startswith("cam"):
        text = text.rsplit("_", 1)[-1]
    if text.startswith("cam"):
        text = text[3:]
    if not text.isdigit() or len(text) > 2:
        return None
    camera = "cam%02d" % int(text)
    if camera not in CAMERA_SESSION:
        return None
    return camera


def session_of(camera):
    """'cam06' -> 'mingling1'."""
    session = CAMERA_SESSION.get(camera)
    if session is None:
        raise ValueError(
            "Unknown camera %r (expected one of %s)"
            % (camera, ", ".join(CAMERA_ORDER)))
    return session


def dataset_of(camera):
    """'cam06' -> 'mingling1/cam06', the layout both data roots ship."""
    return session_of(camera) + "/" + camera


def dataset_label_of(camera):
    """'cam06' -> 'mingling1_cam06', the token in LSTM artifact filenames."""
    return session_of(camera) + "_" + camera


def parse_camera_selection(value):
    """'all' | '06' | 'cam06,cam10' -> tuple of cameras in CAMERA_ORDER order."""
    if value is None:
        value = "all"
    text = str(value).strip()
    if text == "" or text.lower() == "all":
        return tuple(CAMERA_ORDER)

    selected = []
    for token in text.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        camera = normalise_camera(token)
        if camera is None:
            raise ValueError(
                "Unknown camera %r (expected one of %s, or all)"
                % (token, ", ".join(CAMERA_ORDER)))
        if camera not in selected:
            selected.append(camera)
    if not selected:
        raise ValueError("No cameras selected from %r" % value)
    return tuple(camera for camera in CAMERA_ORDER if camera in selected)


def parse_fold_selection(value, num_folds=NUM_FOLDS):
    """'all' | '2' | '0,2,4' -> sorted tuple of fold indices."""
    if value is None:
        value = "all"
    text = str(value).strip()
    if text == "" or text.lower() == "all":
        return tuple(range(num_folds))

    selected = []
    for token in text.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if not token.isdigit() or not (0 <= int(token) < num_folds):
            raise ValueError(
                "Fold must be 0-%d, got: %s" % (num_folds - 1, token))
        fold = int(token)
        if fold not in selected:
            selected.append(fold)
    if not selected:
        raise ValueError("No folds selected from %r" % value)
    return tuple(sorted(selected))
