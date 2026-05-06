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

## Build

Run from the repository root:

```bash
apptainer build apptainer/deep_fformation_dante.sif apptainer/deep_fformation.def
```

If the local Apptainer setup requires fakeroot:

```bash
apptainer build --fakeroot apptainer/deep_fformation_dante.sif apptainer/deep_fformation.def
```

## Quick Checks

Check the LSTM/GraphFF environment:

```bash
apptainer exec --nv apptainer/deep_fformation_dante.sif \
  /opt/conda/envs/py371/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Check the DANTE environment:

```bash
apptainer exec --nv apptainer/deep_fformation_dante.sif \
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

Run one five-fold camera job:

```bash
sbatch --export=ALL,DATASET=mingling1/cam06 slurm/run_dante_mingling_cpu_5fold.sbatch
```

Run all reported DANTE cameras:

```bash
bash slurm/submit_dante_mingling_all.sh
```

The GPU DANTE wrapper is also available:

```bash
sbatch --export=ALL,DATASET=mingling1/cam06 slurm/run_dante_mingling_5fold.sbatch
```

If TensorFlow 1.14 is incompatible with the available GPU runtime, use the CPU
wrapper.

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
3. Place the image at `apptainer/deep_fformation_dante.sif`, or set
   `APPTAINER_IMAGE=/path/to/image.sif` when submitting jobs.
