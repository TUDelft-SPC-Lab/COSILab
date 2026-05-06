# DANTE Dataset Artifacts for Mingling

This directory contains the DANTE-side dataset preparation code for the
Mingling benchmark. Generated camera-level DANTE artifacts are expected under:

```text
DANTE-master/datasets/mingling1/cam06/
DANTE-master/datasets/mingling1/cam08/
DANTE-master/datasets/mingling1/cam10/
DANTE-master/datasets/mingling2/cam01/
DANTE-master/datasets/mingling2/cam03/
```

The generated artifacts are data files and are ignored by Git. They are either
provided by the external data deposit or regenerated from the canonical
LSTM/GraphFF camera CSV files.

## Source Inputs

The preparation script reads the canonical camera-level files from:

```text
data/mingling1/cam06/
data/mingling1/cam08/
data/mingling1/cam10/
data/mingling2/cam01/
data/mingling2/cam03/
```

Each source camera directory must contain:

```text
features.csv
GT.csv
group_names.txt
scene_continuity.csv
dataset_info.json
```

## Generate DANTE DS Utils

Run from the repository root:

```bash
python DANTE-master/datasets/prepare_mingling.py --session mingling1 --overwrite
python DANTE-master/datasets/prepare_mingling.py --session mingling2 --overwrite
```

This writes the DANTE `DS_utils` files and a time mapping file:

```text
DANTE-master/datasets/<session>/<camera>/DS_utils/features.txt
DANTE-master/datasets/<session>/<camera>/DS_utils/group_names.txt
DANTE-master/datasets/<session>/<camera>/time_map.csv
```

## Format Rules

- `features.txt` uses 32 fixed participant slots.
- Slot `k` corresponds to participant `k`.
- Missing participants are padded with `fake fake fake fake fake`.
- DANTE participant IDs follow the original DANTE convention:
  `ID_001`, ..., `ID_0032`.
- Original frame identifiers are remapped to safe DANTE time tokens such as
  `t000000`.
- `time_map.csv` records `dante_time`, `original_time`, and `source_batch`.

## Build Pairwise Files and Folds

Run from `DANTE-master/datasets`:

```bash
python reformat_data.py -d mingling1/cam06 -p 32 -f 5 -a 6
python build_dataset.py -p mingling1/cam06
```

Repeat the two commands for:

```text
mingling1/cam08
mingling1/cam10
mingling2/cam01
mingling2/cam03
```

The resulting camera directory contains:

```text
coordinates.txt
affinities.txt
timechanges.txt
fold_0/train.p
fold_0/val.p
fold_0/test.p
...
fold_4/train.p
fold_4/val.p
fold_4/test.p
```

These files are benchmark data artifacts, not model weights.

## Train DANTE

Run from `DANTE-master/deep_fformation`:

```bash
python run_models.py -d mingling1/cam06 -f 0
```
