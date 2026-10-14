from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random

import numpy as np

from .common import (MODELS, MODALITIES, SUPPORTED_PROTOCOLS, digest, object_hash,
                     roles, write_json, write_jsonl)
from .data import load_cohort, load_window, lr_features
from .evaluate import metrics, report_predictions


def provenance():
    package = Path(__file__).resolve().parent
    root = package.parent
    sources = list(package.glob("*.py")) + [root / "run.py"]
    hashes = {str(p.relative_to(root)): digest(p) for p in sorted(sources)}
    versions = {}
    for name in ("numpy", "pandas", "scikit-learn", "torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"implementation_sha256": object_hash(hashes), "source_code_hashes": hashes,
            "python": platform.python_version(), "packages": versions}


def run_lr(prepared, train, val, test, modality, config, output, seed):
    import joblib
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def features(rows):
        return np.stack([lr_features(load_window(prepared, r), modality) for r in rows])

    train_x, val_x = features(train), features(val)
    train_y, val_y = [r["y"] for r in train], [r["y"] for r in val]
    best_score, best, selected, trials = -np.inf, None, None, []
    for c in config["lr"]["C_grid"]:
        pipeline = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                                 LogisticRegression(C=c, class_weight="balanced", max_iter=config["lr"]["max_iter"],
                                                    solver="lbfgs", random_state=seed))
        pipeline.fit(train_x, train_y)
        score = metrics(val_y, pipeline.predict_proba(val_x)[:, 1])
        trials.append({"C": c, "validation": score})
        # Deterministic first-in-grid tie break, never use test or a failure-only score.
        if score["auprc"] > best_score:
            best_score, best, selected = score["auprc"], pipeline, c
    joblib.dump(best, output / "model.joblib")
    write_json(output / "selection.json", {"selected_C": selected, "trials": trials, "selection_metric": "validation_auprc"})
    # Test features are loaded only after selection is complete; no train+val refit.
    return best.predict_proba(features(test))[:, 1]


def run_tcn(prepared, train, val, test, model_name, modality, config, output, seed, device):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from torch.utils.data import DataLoader
    from .data import WindowDataset, collate, fit_normalization
    from .model import TransitionTCN

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu for a local smoke test")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    focal_only = model_name == "tcn"
    stats = fit_normalization(prepared, train, modality, focal_only)
    write_json(output / "normalization.json", stats)
    cfg = config["tcn"]
    generator = torch.Generator().manual_seed(seed)

    def loader(rows, shuffle=False):
        return DataLoader(WindowDataset(prepared, rows, modality, stats, focal_only), batch_size=cfg["batch_size"],
                          shuffle=shuffle, num_workers=0, collate_fn=collate, generator=generator if shuffle else None)

    model_cfg = (dict(cfg, phase_bins=config["phase_bins"])
                 if config["protocol_version"] in SUPPORTED_PROTOCOLS else cfg)
    net = TransitionTCN(modality, model_cfg, group=not focal_only).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    positives = sum(r["y"] for r in train)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(train) - positives) / positives, device=device))

    def predict(batches):
        net.eval()
        scores = []
        with torch.no_grad():
            for batch in batches:
                batch = {k: v.to(device) for k, v in batch.items()}
                scores.extend(net(batch).sigmoid().cpu().tolist())
        return np.asarray(scores)

    train_loader, val_loader = loader(train, True), loader(val)
    best_score, best_epoch, best_state, stale = -np.inf, None, None, 0
    history = []
    for epoch in range(cfg["max_epochs"]):
        net.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(net(batch), batch["y"])
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["gradient_clip"])
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch["y"])
        scores = predict(val_loader)
        report = metrics([r["y"] for r in val], scores)
        history.append({"epoch": epoch + 1, "train_loss": total_loss / len(train), "validation": report})
        print(json.dumps(history[-1]), flush=True)
        if report["auprc"] > best_score:
            best_score, best_epoch = report["auprc"], epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= cfg["patience"]:
            break
    net.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "normalization": stats, "config": config,
                "model": model_name, "modality": modality, "seed": seed, "selected_epoch": best_epoch}, output / "model.pt")
    write_json(output / "selection.json", {"selected_epoch": best_epoch, "history": history,
                                          "parameter_count": sum(p.numel() for p in net.parameters()),
                                          "selection_metric": "validation_auprc"})
    return predict(loader(test))


def run(prepared, results, config, model, modality, fold, device="cpu"):
    if model not in MODELS or modality not in MODALITIES:
        raise ValueError("Unknown experiment condition")
    rows, freeze = load_cohort(prepared, config)
    rotation = roles(fold)
    partitions = {role: [r for r in rows if rotation[r["outer_fold"]] == role] for role in ("train", "validation", "test")}
    output = results / model / modality / f"fold_{fold}"
    output.mkdir(parents=True, exist_ok=False)  # Never silently replace or reuse an old test run.
    seed = config["training_seed"] + fold
    random.seed(seed)
    np.random.seed(seed)
    run_provenance = provenance()
    write_json(output / "run.json", {"config": config, "freeze": freeze, "model": model, "modality": modality,
                                     "test_fold": fold, "seed": seed, "provenance": run_provenance,
                                     "partition_ids": {k: [r["sample_id"] for r in v] for k, v in partitions.items()}})
    args = (prepared, partitions["train"], partitions["validation"], partitions["test"])
    if model == "lr":
        scores = run_lr(*args, modality, config, output, seed)
    else:
        scores = run_tcn(*args, model, modality, config, output, seed, device)
    test = partitions["test"]
    report = report_predictions(test, scores)
    predictions = [dict({k: r[k] for k in ("sample_id", "sample_type", "y", "session", "split_group_id", "outer_fold", "anchor_time")},
                        score=float(p), model=model, modality=modality) for r, p in zip(test, scores)]
    write_jsonl(output / "test_predictions.jsonl", predictions)
    result = {"freeze": freeze, "model": model, "modality": modality, "test_fold": fold, "seed": seed,
              "provenance": run_provenance, "metrics": report,
              "test_predictions_sha256": digest(output / "test_predictions.jsonl")}
    write_json(output / "result.json", result)  # Only complete runs have this marker.
    return result
