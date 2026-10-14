from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

MODELS = ("lr", "tcn", "tcn_deepsets")
MODALITIES = ("kp", "acc", "kp_acc")
RATES = {"kp": 60, "acc": 20}
CHANNELS = {"kp": 18, "acc": 3}

DUAL_BALANCED_PROTOCOL_POST_SECONDS = {
    "B3-v2-Stage2v3-dual-balanced5-pre2-post2": 2.0,
    "B3-v2-Stage2v3-dual-balanced5-pre2-post5": 5.0,
    "B3-v2-Stage2v3-dual-balanced5-pre2-post10": 10.0,
}

SUPPORTED_PROTOCOLS = tuple(DUAL_BALANCED_PROTOCOL_POST_SECONDS)


def resolve(root, path):
    path = Path(path)
    return path if path.is_absolute() else Path(root) / path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    os.replace(f.name, path)


def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def unique_index(rows, key):
    result = {r[key]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate {key}")
    return result


def load_config(path):
    config = json.loads(Path(path).read_text())
    if config["protocol_version"] in DUAL_BALANCED_PROTOCOL_POST_SECONDS:
        expected_bins = ["pre_far", "pre_near", "expression_first", "expression_second",
                         "response_0_2", "response_2_5", "response_5_10"]
        post_seconds = DUAL_BALANCED_PROTOCOL_POST_SECONDS[config["protocol_version"]]
        tasks = {
            "successful_transition": {
                "positive_types": ["success"],
                "negative_types": ["failure", "ordinary"],
            },
            "attempt_detection": {
                "positive_types": ["success", "failure"],
                "negative_types": ["ordinary"],
            },
        }
        task = config.get("task_name")
        if (config.get("window") != {"kind": "expression_to_anchor_end",
                                     "pre_seconds": 2.0, "post_seconds": post_seconds}
                or config.get("phase_bins") != expected_bins or config.get("folds") != 5
                or config.get("ordinary_per_attempt_target") != 2
                or config.get("split_policy") != "session_balanced_participant_set_v1"
                or config.get("cameras") != {"mingling1": ["cam06", "cam08", "cam10"],
                                               "mingling2": ["cam01", "cam03", "cam05"]}
                or task not in tasks
                or any(config.get(k) != v for k, v in tasks.get(task, {}).items())):
            raise ValueError("Invalid B3 V2 dual-task balanced-five-fold protocol")
        return config
    raise ValueError("This folder supports the topic transition detection V2 dual-task configs only")


def modalities(name):
    if name not in MODALITIES:
        raise ValueError(name)
    return ("kp", "acc") if name == "kp_acc" else (name,)


def roles(test_fold):
    if test_fold not in range(5):
        raise ValueError("fold is zero-based, 0..4")
    validation = (test_fold - 1) % 5
    return {f: "test" if f == test_fold else "validation" if f == validation else "train"
            for f in range(5)}
