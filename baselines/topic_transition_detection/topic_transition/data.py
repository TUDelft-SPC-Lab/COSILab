from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .common import (CHANNELS, RATES, SUPPORTED_PROTOCOLS, digest, modalities,
                     object_hash, read_jsonl)
from .splits import audit_balanced_splits


def load_cohort(prepared, config, verify_windows=True):
    if config["protocol_version"] not in SUPPORTED_PROTOCOLS:
        raise ValueError("Unsupported topic transition detection cohort protocol")
    freeze = json.loads((prepared / "freeze.json").read_text())
    for key, name in (("samples_sha256", "samples.jsonl"), ("fold_map_sha256", "fold_map.json"),
                      ("source_inventory_sha256", "source_inventory.json")):
        if digest(prepared / name) != freeze[key]:
            raise ValueError(f"Frozen input changed: {name}")
    if object_hash(config) != freeze["config_sha256"]:
        raise ValueError("Config differs from the preparation freeze; use a new version/output directory")
    rows = read_jsonl(prepared / "samples.jsonl")
    if (freeze.get("protocol_version") != config["protocol_version"]
            or freeze.get("task_name") != config["task_name"]):
        raise ValueError("Wrong B3 dual-task cohort protocol or task")
    for name, key in (("split_membership.csv", "split_membership_sha256"),
                      ("split_audit.json", "split_audit_sha256"),
                      ("split_plan.json", "split_plan_sha256")):
        if digest(prepared / name) != freeze[key]:
            raise ValueError(f"Balanced split artifact changed: {name}")
    expected_y = ({"success": 1, "failure": 0, "ordinary": 0}
                  if config["task_name"] == "successful_transition"
                  else {"success": 1, "failure": 1, "ordinary": 0})
    if any(row["y"] != expected_y[row["sample_type"]] for row in rows):
        raise ValueError("Task labels differ from the frozen target definition")
    if audit_balanced_splits(rows) != freeze["split_audit"]:
        raise ValueError("B3 balanced split changed")
    if verify_windows:
        for row in rows:
            if digest(prepared / row["window_path"]) != row["window_sha256"]:
                raise ValueError(f"Changed sensor window: {row['sample_id']}")
    return rows, freeze


def load_window(prepared, row):
    if row.get("protocol_version") not in SUPPORTED_PROTOCOLS:
        raise ValueError("Unsupported topic transition detection window protocol")
    with np.load(Path(prepared) / row["window_path"], allow_pickle=False) as f:
        keys = ["kp", "kp_mask", "acc", "acc_mask", "participant_ids",
                "person_present", "kp_times", "acc_times"]
        if row.get("protocol_version") in SUPPORTED_PROTOCOLS:
            keys += ["kp_bin_ids", "acc_bin_ids"]
        arrays = {k: f[k] for k in keys}
    people = arrays["participant_ids"].astype(str).tolist()
    expected_people = ([row["focal_participant"], *sorted(
        p for p in row["participant_ids"] if p != row["focal_participant"])]
        if row.get("protocol_version") in SUPPORTED_PROTOCOLS else row["participant_ids"])
    if people != expected_people or people[0] != row["focal_participant"]:
        raise ValueError("Participant order must preserve annotated roster, focal first")
    if arrays["person_present"].shape != (len(people),) or not arrays["person_present"][0]:
        raise ValueError("Invalid person eligibility mask")
    for m in ("kp", "acc"):
        n = (int(np.ceil((row['window_end']-row['window_start'])*RATES[m]-1e-8))
             if row.get('protocol_version') in SUPPORTED_PROTOCOLS else 15*RATES[m])
        shape = (n, len(people), CHANNELS[m])
        if arrays[m].shape != shape or arrays[m + "_mask"].shape != shape:
            raise ValueError(f"Unexpected {m} window shape")
        targets = row["window_start"] + np.arange(shape[0]) / RATES[m]
        if not np.allclose(arrays[m + "_times"], targets, rtol=0, atol=1e-8):
            raise ValueError(f"Invalid {m} time grid")
        mask = arrays[m + "_mask"].astype(bool) & arrays["person_present"][None, :, None]
        if not np.isfinite(arrays[m][mask]).all():
            raise ValueError("Non-finite observed signal")
        arrays[m + "_mask"] = mask
        arrays[m] = np.where(mask, arrays[m], 0).astype(np.float32)
        if row.get("protocol_version") in SUPPORTED_PROTOCOLS:
            bins = arrays[m + "_bin_ids"]
            if bins.shape != (n,) or np.any((bins < 0) | (bins >= 7)):
                raise ValueError(f"Invalid {m} phase-bin indices")
    return arrays


def masked_summary(values, mask):
    count = mask.sum(axis=0)
    mean = np.divide(np.where(mask, values, 0).sum(axis=0), count,
                     out=np.full(values.shape[-1], np.nan), where=count > 0)
    variance = np.divide(np.where(mask, (values - mean) ** 2, 0).sum(axis=0), count,
                         out=np.full_like(mean, np.nan), where=count > 0)
    return mean, np.sqrt(variance)


def lr_features(arrays, modality):
    blocks = []
    for m in modalities(modality):
        x, mask = arrays[m][:, 0], arrays[m + "_mask"][:, 0]
        ids = arrays[m + "_bin_ids"]
        stats = []
        for b in range(7):
            mean, std = masked_summary(x[ids == b], mask[ids == b])
            stats.extend((mean, std))
        blocks.append(np.stack(stats, axis=1).ravel())
    return np.concatenate(blocks)


def fit_normalization(prepared, train_rows, modality, focal_only):
    stats = {}
    for m in modalities(modality):
        count, total, squared = [np.zeros(CHANNELS[m], np.float64) for _ in range(3)]
        for row in train_rows:
            arr = load_window(prepared, row)
            x, mask = arr[m].astype(np.float64), arr[m + "_mask"]
            if focal_only:
                x, mask = x[:, :1], mask[:, :1]
            count += mask.sum(axis=(0, 1))
            total += np.where(mask, x, 0).sum(axis=(0, 1))
            squared += np.where(mask, x * x, 0).sum(axis=(0, 1))
        mean = total / np.maximum(count, 1)
        std = np.sqrt(np.maximum(squared / np.maximum(count, 1) - mean * mean, 0))
        stats[m] = {"mean": mean.tolist(), "std": np.where(std < 1e-6, 1, std).tolist(), "count": count.tolist()}
    return stats


class WindowDataset:
    def __init__(self, prepared, rows, modality, stats, focal_only):
        self.prepared, self.rows, self.modality, self.stats, self.focal_only = prepared, rows, modality, stats, focal_only

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        arr = load_window(self.prepared, self.rows[i])
        item = {"y": self.rows[i]["y"], "person_present": arr["person_present"]}
        for m in modalities(self.modality):
            mask = arr[m + "_mask"]
            x = (arr[m] - np.asarray(self.stats[m]["mean"])) / np.asarray(self.stats[m]["std"])
            item[m] = np.where(mask, x, 0).astype(np.float32)
            item[m + "_mask"] = mask
            if m + "_bin_ids" in arr:
                item[m + "_bin_ids"] = arr[m + "_bin_ids"]
        if self.focal_only:
            item["person_present"] = item["person_present"][:1]
            for m in modalities(self.modality):
                item[m], item[m + "_mask"] = item[m][:, :1], item[m + "_mask"][:, :1]
        return item


def collate(items):
    import torch
    people = max(len(i["person_present"]) for i in items)
    result = {"y": torch.tensor([i["y"] for i in items], dtype=torch.float32),
              "person_present": torch.zeros(len(items), people, dtype=torch.bool)}
    for b, item in enumerate(items):
        result["person_present"][b, :len(item["person_present"])] = torch.as_tensor(item["person_present"])
    for m in ("kp", "acc"):
        if m not in items[0]:
            continue
        steps = max(len(item[m]) for item in items)
        shape = (len(items), steps, people, CHANNELS[m])
        result[m], result[m + "_mask"] = torch.zeros(shape), torch.zeros(shape, dtype=torch.bool)
        if m + "_bin_ids" in items[0]:
            result[m + "_bin_ids"] = torch.full((len(items), steps), -1, dtype=torch.long)
        for b, item in enumerate(items):
            n = len(item["person_present"])
            t = len(item[m])
            result[m][b, :t, :n] = torch.as_tensor(item[m])
            result[m + "_mask"][b, :t, :n] = torch.as_tensor(item[m + "_mask"])
            if m + "_bin_ids" in item:
                result[m + "_bin_ids"][b, :t] = torch.as_tensor(item[m + "_bin_ids"], dtype=torch.long)
    return result
