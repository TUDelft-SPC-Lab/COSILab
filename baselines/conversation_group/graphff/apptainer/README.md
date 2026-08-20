# Apptainer Environment

This directory contains the Apptainer recipe for the Mingling benchmark runtime.
The recipe is tracked so the LSTM/GraphFF and DANTE environments can be rebuilt
on local machines or HPC systems.

Generated `.sif` images are large binary artifacts and are ignored by Git.

## Image Contents

`deep_fformation.def` builds one image with two Conda environments:

| Environment | Python | Used By | Main Packages |
| --- | --- | --- | --- |
| `/opt/conda/envs/py371` | 3.7.1 | LSTM/GraphFF | PyTorch 1.10.1, pandas, scipy, scikit-learn |
| `/opt/conda/envs/dante_tf1` | 3.7.1 | DANTE | TensorFlow GPU 1.14.0, Keras 2.2.2 |

The default runtime remains the PyTorch environment. DANTE Slurm scripts select
`/opt/conda/envs/dante_tf1/bin/python` through `DANTE_PYTHON_BIN`.

## Image Location

The DANTE job script looks for the image on shared storage:

```text
/tudelft.net/staff-umbrella/neon/apptainer/deep_fformation_dante.sif
```

Override with `APPTAINER_IMAGE=/path/to/image.sif` at submit time. The LSTM/GraphFF
scripts still default to `$PROJECT_ROOT/apptainer/deep_fformation_dante.sif`.

## Build

Run from the repository root, then move the image to the shared location above:

```bash
apptainer build apptainer/deep_fformation_dante.sif apptainer/deep_fformation.def
```

If the local Apptainer setup requires fakeroot:

```bash
apptainer build --fakeroot apptainer/deep_fformation_dante.sif apptainer/deep_fformation.def
```

## Quick Checks

Against the shared image:

```bash
SIF=/tudelft.net/staff-umbrella/neon/apptainer/deep_fformation_dante.sif
```

Check the LSTM/GraphFF environment:

```bash
apptainer exec --nv "$SIF" \
  /opt/conda/envs/py371/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Check the DANTE environment:

```bash
apptainer exec --nv "$SIF" \
  /opt/conda/envs/dante_tf1/bin/python -c "import tensorflow as tf, keras; print(tf.__version__); print(keras.__version__)"
```

## LSTM/GraphFF Slurm Example

Run one five-fold camera job:

```bash
sbatch --export=ALL,DATASET=mingling1/cam06,TRAIN=1,DATASET_MAKE=1,FRAME_STRIDE=20 \
  slurm/run_vitpose_dataframe_5fold.sbatch
```

Run all reported LSTM/GraphFF cameras:

```bash
bash slurm/submit_lstm_mingling_all.sh
```

## DANTE Slurm Example

Run one five-fold camera job. The camera number selects the Mingling session
(06, 08, 10 are mingling1; 01, 03 are mingling2):

```bash
bash slurm/submit_dante.sh --cam=06
```

Run all reported DANTE cameras:

```bash
bash slurm/submit_dante.sh --cam=all
```

DANTE runs on CPU by default. Request the GPU profile with:

```bash
USE_GPU=1 bash slurm/submit_dante.sh --cam=06
```

If TensorFlow 1.14 is incompatible with the available GPU runtime, stay on the
CPU default.

## Git Tracking Policy

Tracked:

```text
apptainer/deep_fformation.def
apptainer/README.md
```

Ignored:

```text
*.sif
```

The expected workflow is:

1. Commit recipe changes when dependencies change.
2. Build the `.sif` image locally or on a suitable build machine.
3. Place the image at
   `/tudelft.net/staff-umbrella/neon/apptainer/deep_fformation_dante.sif` for the
   DANTE jobs, or at `apptainer/deep_fformation_dante.sif` for the LSTM/GraphFF
   jobs. Either default can be overridden with `APPTAINER_IMAGE=/path/to/image.sif`
   when submitting.
