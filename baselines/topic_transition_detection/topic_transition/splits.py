"""Audits for the V2 dual-task, session-balanced participant-set folds."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib

from .common import roles, unique_index


KIND = "session_balanced_participant_set_v1"


def participant_set_key(row):
    people = tuple(sorted((str(p) for p in row["participant_ids"]), key=lambda x: int(x)))
    return row["session"], people


def participant_set_id(row):
    session, people = participant_set_key(row)
    value = session + "|" + ",".join(people)
    return "ps_" + hashlib.sha256(value.encode()).hexdigest()[:16]


def audit_balanced_splits(rows):
    unique_index(rows, "sample_id")
    folds_by_group = defaultdict(set)
    folds_by_person = defaultdict(set)
    identities = defaultdict(set)
    spans = defaultdict(list)
    for row in rows:
        if row.get("split_protocol") != KIND or row["outer_fold"] not in range(5):
            raise ValueError("Invalid balanced split metadata")
        expected = participant_set_id(row)
        if row["split_group_id"] != expected:
            raise ValueError("Split group is not the frozen session participant set")
        folds_by_group[expected].add(row["outer_fold"])
        keys = ["source:" + row["anchor_source_id"]]
        keys += ["attempt:" + str(x) for x in
                 (row.get("attempt_id"), row.get("retry_of_attempt_id")) if x]
        keys += ["alias:" + str(x) for x in row.get("source_identity_keys", [])]
        for key in keys:
            identities[key].add(row["outer_fold"])
        for person in row["participant_ids"]:
            folds_by_person[row["session"], str(person)].add(row["outer_fold"])
            spans[row["session"], str(person)].append(
                (row["support_start"], row["support_end"], row["outer_fold"], row["sample_id"]))
    if any(len(value) != 1 for value in folds_by_group.values()):
        raise ValueError("One participant set crosses folds")
    if any(len(value) != 1 for value in identities.values()):
        raise ValueError("One source, attempt, or alias crosses folds")
    for values in spans.values():
        active = []
        for left, right, fold, sample_id in sorted(values):
            active = [item for item in active if item[0] > left]
            if any(other_fold != fold for _, other_fold, _ in active):
                raise ValueError(f"Shared participant/time support crosses folds: {sample_id}")
            active.append((right, fold, sample_id))

    fold_counts = {}
    for fold in range(5):
        subset = [row for row in rows if row["outer_fold"] == fold]
        sessions = Counter(row["session"] for row in subset)
        types = Counter(row["sample_type"] for row in subset)
        if set(sessions) != {"mingling1", "mingling2"}:
            raise ValueError(f"Fold {fold} does not contain both sessions")
        if set(types) != {"success", "failure", "ordinary"}:
            raise ValueError(f"Fold {fold} does not contain every sample type")
        if {row["y"] for row in subset} != {0, 1}:
            raise ValueError(f"Fold {fold} lacks one task class")
        fold_counts[str(fold)] = {
            "total": len(subset),
            "sessions": dict(sorted(sessions.items())),
            "sample_types": dict(sorted(types.items())),
            "positive": sum(row["y"] for row in subset),
            "negative": len(subset) - sum(row["y"] for row in subset),
        }
    sizes = [fold_counts[str(fold)]["total"] for fold in range(5)]
    if max(sizes) / min(sizes) > 1.10:
        raise ValueError("Balanced fold size ratio exceeds 1.10")
    for test_fold in range(5):
        allocation = roles(test_fold)
        for role in ("train", "validation", "test"):
            subset = [row for row in rows if allocation[row["outer_fold"]] == role]
            if {row["y"] for row in subset} != {0, 1}:
                raise ValueError(f"Run {test_fold} {role} lacks one task class")
            if {row["session"] for row in subset} != {"mingling1", "mingling2"}:
                raise ValueError(f"Run {test_fold} {role} lacks one session")
    return {
        "passed": True,
        "samples": len(rows),
        "groups": len(folds_by_group),
        "split_protocol": KIND,
        "test_visits_per_sample": 1,
        "participant_set_disjoint": True,
        "individual_participant_disjoint": not any(
            len(folds) > 1 for folds in folds_by_person.values()),
        "shared_participant_sensor_support_disjoint": True,
        "source_and_attempt_disjoint": True,
        "every_fold_has_both_sessions": True,
        "fold_size_max_min_ratio": max(sizes) / min(sizes),
        "fold_counts": fold_counts,
    }
