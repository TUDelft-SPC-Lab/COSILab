from __future__ import annotations

from collections import Counter
import csv
import json

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .common import MODELS, MODALITIES, digest, read_jsonl, unique_index, write_json, write_jsonl
from .data import load_cohort


def metrics(y, scores):
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if len(y) != len(scores) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Invalid prediction probabilities")
    result = {"n": len(y), "positive": int(y.sum()), "prevalence": float(y.mean()) if len(y) else None,
              "auroc": None, "auprc": None, "undefined_reason": None}
    if len(np.unique(y)) != 2:
        result["undefined_reason"] = "requires_both_classes"
    else:
        result.update(auroc=float(roc_auc_score(y, scores)), auprc=float(average_precision_score(y, scores)))
    return result


def report_predictions(rows, scores):
    result = {}
    for session in ("all", "mingling1", "mingling2"):
        pairs = [(r, p) for r, p in zip(rows, scores) if session == "all" or r["session"] == session]
        result[session] = dict(metrics([r["y"] for r, _ in pairs], [p for _, p in pairs]),
                               subtype_counts=dict(Counter(r["sample_type"] for r, _ in pairs)))
    return result


def summarize(prepared, results, config):
    rows, freeze = load_cohort(prepared, config, verify_windows=False)
    expected = unique_index(rows, "sample_id")
    comparison, all_results = [], {}
    common_code_hash = None
    for model in MODELS:
        for modality in MODALITIES:
            predictions, folds = [], []
            for fold in range(5):
                directory = results / model / modality / f"fold_{fold}"
                result = json.loads((directory / "result.json").read_text())
                if (result["freeze"] != freeze or result["model"] != model
                        or result["modality"] != modality or result["test_fold"] != fold):
                    raise ValueError(f"Run provenance mismatch: {directory}")
                if digest(directory / "test_predictions.jsonl") != result["test_predictions_sha256"]:
                    raise ValueError("Predictions changed after evaluation")
                code_hash = result["provenance"]["implementation_sha256"]
                if common_code_hash is None:
                    common_code_hash = code_hash
                if common_code_hash != code_hash:
                    raise ValueError("Mixed implementation versions in result matrix")
                pred = read_jsonl(directory / "test_predictions.jsonl")
                if {p["sample_id"] for p in pred} != {r["sample_id"] for r in rows if r["outer_fold"] == fold}:
                    raise ValueError("Wrong held-out sample set")
                for p in pred:
                    r = expected[p["sample_id"]]
                    if any(p[k] != r[k] for k in ("y", "sample_type", "session", "outer_fold", "split_group_id")):
                        raise ValueError("Prediction metadata changed")
                unique_index(pred, "sample_id")
                predictions.extend(pred)
                folds.append(report_predictions(pred, [p["score"] for p in pred]))
            if set(unique_index(predictions, "sample_id")) != set(expected):
                raise ValueError("Each sample must be held out exactly once")
            condition = f"{model}/{modality}"
            write_jsonl(results / model / modality / "oof_predictions.jsonl", sorted(predictions, key=lambda p: p["sample_id"]))
            pooled = report_predictions(predictions, [p["score"] for p in predictions])
            all_results[condition] = {"folds": folds, "pooled_oof_secondary": pooled}
            for session in ("all", "mingling1", "mingling2"):
                row = {"model": model, "input": modality, "session": session, "n": pooled[session]["n"],
                       "positive_prevalence": pooled[session]["prevalence"]}
                prevalences = [f[session]['prevalence'] for f in folds if f[session]['prevalence'] is not None]
                valid_prevalences = [f[session]['prevalence'] for f in folds if f[session]['auprc'] is not None]
                row['positive_prevalence_fold_mean'] = float(np.mean(prevalences)) if prevalences else None
                row['constant_score_ap_valid_folds_mean'] = float(np.mean(valid_prevalences)) if valid_prevalences else None
                row['single_class_or_empty_folds'] = sum(f[session]['undefined_reason'] is not None for f in folds)
                for metric in ("auroc", "auprc"):
                    values = [f[session][metric] for f in folds if f[session][metric] is not None]
                    row[metric + "_mean"] = float(np.mean(values)) if values else None
                    row[metric + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
                    row[metric + "_valid_folds"] = len(values)
                comparison.append(row)
    write_json(results / "comparison.json", {"primary": comparison, "details": all_results, "freeze": freeze,
                                              "auprc_definition": config["auprc_definition"], "std_ddof": 1})
    with (results / "comparison.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    return comparison
