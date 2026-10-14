# Topic Transition Detection

Training and evaluation baselines for topic transition detection on COSILab, using the V2 dual-task, balanced-five-fold protocol. This baseline corresponds to the original Benchmark 3 experiments.

## Folder structure

```text
COSILab/
  baselines/
    topic_transition_detection/
      README.md
      run.py
      requirements.txt
      requirements-tcn.txt
      configs/                # six task/horizon configurations
      topic_transition/       # data loading, split audits, models and evaluation
```

This folder runs the benchmark from prepared windows. Sensor files, sample metadata, reference result tables, and raw-data preparation are supplied separately. No other Python code from the original research repository is required.

## Benchmark settings

The shared cohort has 1,392 windows: 316 SUCCESS, 148 FAILURE, and 928 ordinary controls.

| Task | Positive | Negative |
| --- | --- | --- |
| `attempt_detection` | SUCCESS + FAILURE | ordinary |
| `successful_transition` | SUCCESS | FAILURE + ordinary |

The targets are fixed V2 Stage 1 v3 / Stage 2 v3 model annotations, not independent human-gold labels. Candidate turns were not deduplicated into independent conversational events.

Each window starts two seconds before expression onset and ends 2, 5, or 10 seconds after expression end, subject to the stable-group boundary. Shorter horizons retain the +10s sample selection, labels and folds. Response-period signals are included, so these tasks are not exclusively pre-attempt forecasts.

The three baselines are logistic regression (`lr`), focal-person TCN (`tcn`), and group TCN-DeepSets (`tcn_deepsets`). Inputs are keypoints (`kp`, 18 channels at 60 Hz), accelerometer signals (`acc`, 3 channels at 20 Hz), or both (`kp_acc`). Missingness masks and seven phase bins are retained.

Five folds use participant-set groups within Mingling sessions. Splits are participant-set disjoint and checked for source/attempt overlap and shared-person temporal-support overlap. Individual participants and conversation episodes can occur in multiple folds at nonoverlapping times. Each rotation uses fold k for testing, fold (k-1) modulo 5 for validation, and the other three for training.

## Installation

Use Python 3.10 or newer. Reference runs used Python 3.12.3 and the scientific package versions pinned here.

```bash
cd COSILab/baselines/topic_transition_detection
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For either TCN baseline, also install PyTorch:

```bash
python -m pip install -r requirements-tcn.txt
```

Reference runs used PyTorch 2.10.0+cu128. Select the CPU/CUDA build appropriate for your machine. The runner defaults to CPU; `--device cuda` requests GPU execution. On a managed cluster, use a scheduler allocation.

## External reference data

Set `--data-root` to a directory containing the three prepared cohorts below. The data can be kept anywhere outside the GitHub code folder.

```text
prepared-data/
  prepared_v2_stage2v3_dual_balanced5/
    windows/                             # 1,392 original +10s NPZ files
    attempt_detection/                   # frozen task metadata
    successful_transition/               # frozen task metadata
  prepared_v2_stage2v3_dual_balanced5_pre2_post2/
    windows/                             # 1,392 original +2s NPZ files
    attempt_detection/
    successful_transition/
  prepared_v2_stage2v3_dual_balanced5_pre2_post5/
    windows/                             # 1,392 original +5s NPZ files
    attempt_detection/
    successful_transition/
```

Each task directory needs the original `samples.jsonl`, `freeze.json`, `source_inventory.json`, `fold_map.json`, `split_membership.csv`, `split_audit.json`, `split_plan.json`, `cohort_summary.json`, `sampling_report.json`, and `validation.json`. Retain the supplied `config.json` as well. Sample records address arrays with `../windows/<name>.npz`. Preserve the original files and paths; exact reproduction requires the original prepared data.

The external data retains its original cohort directory names and B3 protocol identifiers for compatibility with the frozen version. The two tasks at each horizon share the same windows directory. The 4,176 NPZ files total 2,238,037,893 bytes before additional compression. Providing only the NPZ files is insufficient: the task labels and splits come from the accompanying metadata. No raw audio/video or annotation-model access is required once the prepared data is available.

Each NPZ contains `participant_ids`, `person_present`, `kp`, `kp_mask`, `acc`, `acc_mask`, `kp_times`, `acc_times`, `kp_bin_ids`, and `acc_bin_ids`, plus the original preparation provenance. The loader checks participant order, array shapes, observation masks, finite observed values, time grids and phase bins.

The configs also retain preparation fields such as `accelerometer_root`, `badge_map`, and camera lists so they match the frozen dataset configuration. These fields document upstream preparation; training does not open those raw-data paths. Raw sensor extraction, transcript annotation and cohort reconstruction are upstream data-preparation steps outside this folder.

## Plan and validate

The run plan can be inspected without data or scientific dependencies:

```bash
python run.py plan --horizons 10 2 5
```

Check supplied metadata without reading sensor arrays, then optionally validate the complete data:

```bash
python run.py validate --data-root /path/to/prepared-data \
  --metadata-only --horizons 10 2 5
python run.py validate --data-root /path/to/prepared-data \
  --horizons 10 2 5
```

## Train

Run one LR condition to check the environment:

```bash
python run.py run --data-root /path/to/prepared-data \
  --results-root results_check --horizons 10 \
  --tasks attempt_detection --models lr --modalities kp --folds 0
```

Run the full +10s experiment (90 fold conditions):

```bash
python run.py run --data-root /path/to/prepared-data \
  --results-root results --horizons 10 --device cuda --threads 4
```

Run the +2s and +5s comparison (180 more conditions):

```bash
python run.py run --data-root /path/to/prepared-data \
  --results-root results --horizons 2 5 --device cuda --threads 4
```

Use `--horizons 10 2 5` for all 270 conditions, or select individual tasks, models, modalities and folds. Use `--device cpu` where needed. Existing run directories are not overwritten. `--resume` verifies and skips complete compatible runs; use a new results directory after changing code or inputs.

## Results and reference-table comparison

Outputs are placed under `results/post{2,5,10}/{task}/{model}/{modality}/fold_{0..4}/`. Each run records its configuration, selected hyperparameters, checkpoint, test probabilities and metrics. Full 45-run task/horizon matrices are summarized automatically. Summaries can also be regenerated:

```bash
python run.py summarize --data-root /path/to/prepared-data \
  --results-root results --horizons 10 2 5
```

`comparison.csv` and `comparison.json` contain five-fold AUROC/AP means and sample standard deviations (`ddof=1`), plus session diagnostics. Each model/modality also gets pooled out-of-fold predictions. AP is `sklearn.metrics.average_precision_score`, not trapezoidal PR area. Validation AP selects LR regularization and TCN checkpoints; seeds are 13 plus the test-fold index. Production TCN settings use at most 80 epochs and patience 10.

When comparing with a reference table, match the target, response horizon, model, modality and five-fold definition. Sensor data, reference tables and generated results belong outside this GitHub folder.
