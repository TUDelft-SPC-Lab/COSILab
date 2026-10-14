#!/usr/bin/env python3
"""Train and evaluate the topic transition detection baselines."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
CONFIGS = ROOT / "configs"
TASKS = ("successful_transition", "attempt_detection")
MODELS = ("lr", "tcn", "tcn_deepsets")
MODALITIES = ("kp", "acc", "kp_acc")
EXPECTED_COUNTS = {"success": 316, "failure": 148, "ordinary": 928}


def cohort_name(horizon):
    suffix = "" if horizon == 10 else f"_pre2_post{horizon}"
    return "prepared_v2_stage2v3_dual_balanced5" + suffix


def config_path(horizon, task):
    return CONFIGS / f"post{horizon}_{task}.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_plan(args):
    jobs = []
    for horizon in args.horizons:
        for task in args.tasks:
            config = json.loads(config_path(horizon, task).read_text())
            for model in args.models:
                for modality in args.modalities:
                    for fold in args.folds:
                        jobs.append({
                            "horizon": horizon, "task": task, "model": model,
                            "modality": modality, "fold": fold,
                            "validation_fold": (fold - 1) % 5,
                            "training_folds": [i for i in range(5) if i not in (fold, (fold - 1) % 5)],
                            "seed": config["training_seed"] + fold,
                            "device": "cpu" if model == "lr" else args.device,
                            "config": str(config_path(horizon, task)),
                            "prepared": str(args.data_root / cohort_name(horizon) / task),
                            "results": str(args.results_root / f"post{horizon}" / task),
                        })
    return jobs


def load_inputs(data_root, horizon, task, verify_windows):
    from topic_transition.common import load_config
    from topic_transition.data import load_cohort, load_window

    prepared = data_root / cohort_name(horizon) / task
    if not (prepared / "freeze.json").is_file():
        raise FileNotFoundError(f"Missing frozen cohort: {prepared}. See README.md for the external data layout.")
    config = load_config(config_path(horizon, task))
    rows, freeze = load_cohort(prepared, config, verify_windows=False)
    if len(rows) != 1392 or freeze["counts"] != EXPECTED_COUNTS:
        raise ValueError(f"Cohort size or class counts differ from topic transition detection V2: {prepared}")
    for filename, key in (("cohort_summary.json", "cohort_summary_sha256"),
                          ("sampling_report.json", "sampling_report_sha256"),
                          ("validation.json", "validation_sha256")):
        if sha256(prepared / filename) != freeze[key]:
            raise ValueError(f"Frozen metadata changed: {prepared / filename}")
    if verify_windows:
        for row in rows:
            path = prepared / row["window_path"]
            if not path.is_file():
                raise FileNotFoundError(f"Missing sensor array: {path}. The prepared data is supplied separately; see README.md.")
            if sha256(path) != row["window_sha256"]:
                raise ValueError(f"Sensor window changed: {row['sample_id']}")
            load_window(prepared, row)
    return prepared, config, rows, freeze


def verify_completed(output, rows, freeze, job, implementation_hash):
    """Resume only complete runs of this code on these frozen inputs."""
    from topic_transition.common import read_jsonl, unique_index

    for name in ("result.json", "run.json", "selection.json", "test_predictions.jsonl",
                 "model.joblib" if job["model"] == "lr" else "model.pt"):
        if not (output / name).is_file():
            raise ValueError(f"Incomplete run at {output}; choose a new results-root or resolve it explicitly.")
    result = json.loads((output / "result.json").read_text())
    run = json.loads((output / "run.json").read_text())
    for record in (result, run):
        if (record["freeze"] != freeze or record["model"] != job["model"]
                or record["modality"] != job["modality"] or record["test_fold"] != job["fold"]
                or record["seed"] != job["seed"]
                or record["provenance"]["implementation_sha256"] != implementation_hash):
            raise ValueError(f"Existing run has different inputs or code: {output}")
    if sha256(output / "test_predictions.jsonl") != result["test_predictions_sha256"]:
        raise ValueError(f"Existing predictions changed: {output}")
    expected = {r["sample_id"]: r for r in rows if r["outer_fold"] == job["fold"]}
    predictions = unique_index(read_jsonl(output / "test_predictions.jsonl"), "sample_id")
    if set(expected) != set(predictions):
        raise ValueError(f"Wrong held-out samples in existing run: {output}")
    for sample_id, prediction in predictions.items():
        if any(prediction[key] != expected[sample_id][key]
               for key in ("sample_type", "y", "session", "outer_fold", "split_group_id")):
            raise ValueError(f"Wrong prediction metadata: {output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "validate", "run", "summarize"))
    parser.add_argument("--data-root", type=Path,
                        help="External prepared-data directory; required except for plan")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--horizons", nargs="+", type=int, choices=(2, 5, 10), default=[10])
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--modalities", nargs="+", choices=MODALITIES, default=list(MODALITIES))
    parser.add_argument("--folds", nargs="+", type=int, choices=range(5), default=list(range(5)))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="Skip only verified, complete runs of the same code")
    parser.add_argument("--metadata-only", action="store_true", help="Validate frozen labels and splits without sensor arrays")
    parser.add_argument("--plan-file", type=Path, help="Write the complete matrix plan (refuses existing files)")
    args = parser.parse_args(argv)
    for key in ("horizons", "tasks", "models", "modalities", "folds"):
        if len(getattr(args, key)) != len(set(getattr(args, key))):
            parser.error(f"Duplicate values in --{key}")
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.metadata_only and args.command != "validate":
        parser.error("--metadata-only applies to validate")
    if args.resume and args.command != "run":
        parser.error("--resume applies to run")
    if args.data_root is None:
        if args.command != "plan":
            parser.error("--data-root is required; see README.md for the external data layout")
        args.data_root = Path("prepared-data")
    args.data_root = args.data_root.expanduser().resolve()
    args.results_root = args.results_root.expanduser().resolve()
    jobs = build_plan(args)
    if args.command == "plan":
        if args.plan_file:
            with args.plan_file.open("x", encoding="utf-8") as stream:
                json.dump(jobs, stream, indent=2)
                stream.write("\n")
        print(json.dumps({"runs": len(jobs), "horizons": args.horizons, "tasks": args.tasks,
                          "models": args.models, "modalities": args.modalities, "folds": args.folds,
                          "data_root": str(args.data_root), "results_root": str(args.results_root),
                          "plan_file": str(args.plan_file) if args.plan_file else None}, indent=2))
        return
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = str(args.threads)
    loaded = {}
    for horizon in args.horizons:
        for task in args.tasks:
            loaded[horizon, task] = load_inputs(args.data_root, horizon, task,
                                               args.command == "run" or
                                               (args.command == "validate" and not args.metadata_only))
            print(f"Validated post{horizon}/{task}: {len(loaded[horizon, task][2])} samples", flush=True)
    identities = [sorted((r["sample_id"], r["outer_fold"], r["sample_type"]) for r in value[2])
                  for value in loaded.values()]
    if any(value != identities[0] for value in identities):
        raise ValueError("Selected task/horizon views differ in samples or folds")
    if args.command == "validate":
        print("Frozen metadata verified" if args.metadata_only else "Frozen metadata, sensor hashes and array shapes verified")
        return
    if args.command == "run":
        from topic_transition.train import provenance, run

        if any(job["model"] != "lr" for job in jobs):
            import torch
            torch.set_num_threads(args.threads)
            if args.device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable; use --device cpu or install compatible PyTorch")
        implementation_hash = provenance()["implementation_sha256"]
        for index, job in enumerate(jobs, 1):
            prepared, config, rows, freeze = loaded[job["horizon"], job["task"]]
            results = Path(job["results"])
            output = results / job["model"] / job["modality"] / f"fold_{job['fold']}"
            description = f"post{job['horizon']}/{job['task']}/{job['model']}/{job['modality']}/fold_{job['fold']}"
            if output.exists():
                if not args.resume:
                    raise FileExistsError(f"Existing run: {output}. Use --resume to verify and skip complete runs.")
                verify_completed(output, rows, freeze, job, implementation_hash)
                print(f"[{index}/{len(jobs)}] Verified existing {description}", flush=True)
                continue
            log = results / "logs" / job["model"] / job["modality"] / f"fold_{job['fold']}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            print(f"[{index}/{len(jobs)}] Running {description}; log: {log}", flush=True)
            with log.open("x", encoding="utf-8") as stream, redirect_stdout(stream), redirect_stderr(stream):
                run(prepared, results, config, job["model"], job["modality"], job["fold"], job["device"])
        full_matrix = (set(args.models) == set(MODELS) and set(args.modalities) == set(MODALITIES)
                       and set(args.folds) == set(range(5)))
        if not full_matrix:
            print("Selected runs completed. Summarize after completing all 45 runs per task/horizon.")
            return
    from topic_transition.evaluate import summarize
    for (horizon, task), (prepared, config, _, _) in loaded.items():
        results = args.results_root / f"post{horizon}" / task
        summarize(prepared, results, config)
        print(f"Summary: {results / 'comparison.csv'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(2)
