# SAM3 / SAM-Body4D for INGroup

This directory contains a local adaptation of the original SAM-Body4D repository:

https://github.com/gaomingqi/sam-body4d

Refer to the upstream repository for the original installation guide, model details, citation, and general SAM-Body4D usage. The notes below only document the INGroup mask-generation workflow added in this project.

## Environment and Config

Dependencies are declared in two places, as upstream: `pyproject.toml` (the `sam-body4d` project, which packages `models/`) and `models/sam3/pyproject.toml` (the vendored SAM-3 package: `timm`, `iopath`, `ftfy`, `huggingface_hub`, ...). PyTorch/torchvision are not listed in either and must be installed separately for your CUDA version. On the cluster everything comes from the Apptainer image, and the job puts `models/sam3` and the repository root on `PYTHONPATH` instead of installing them.

`configs/body4d.yaml` is not tracked. Generate it and download the checkpoints with:

```bash
python scripts/setup.py --ckpt-root /path/to/checkpoints
```

This downloads `configs/config.template.yaml` from the upstream `gaomingqi/sam-body4d` release (overwriting the local copy), fills in `${CKPT_ROOT}`, and writes `configs/body4d.yaml`. The SAM-3 and SAM-3D-Body checkpoints are gated on Hugging Face, so request access first. Alternatively, copy `configs/config.template.yaml` to `configs/body4d.yaml` and replace `${CKPT_ROOT}` by hand.

The job scripts below pass `--config configs/body4d.yaml`; `run_sam3_masklets_batch.py` resolves it against the working directory and then its own directory, so the generated file must exist in the checkout the job runs from.

## Generate Masks for INGroup

The INGroup mask workflow runs SAM-3 masklet extraction over 10-second video segments, using the first-frame person bounding-box annotations for each segment.

Main entry points:

- Slurm batch script: `job_scripts/masklets_batch_ingroup.sh`
- Python runner called by the job: `run_sam3_masklets_batch.py`
- SAM-Body4D config: `configs/body4d.yaml` (generated, see above)

`job_scripts/masklets_batch_delftblue.sh` is the equivalent wrapper for the Conflab layout (`/mnt/data/conflab/video_segs_20s/<NUM>`, invoked as `sbatch job_scripts/masklets_batch_delftblue.sh 428` or `-all 4`); see its header for its defaults.

The Slurm script expects this cluster layout:

```text
/scratch/zli33/data/ingroup/
  video_segs_10s/
    cam02_batch01/
      *_seg*.mp4
  video_frame_annotations/
    cam02_batch01*/
      *_seg*.json
  bbox_kp/
    cam02_batch01/
```

Annotation JSON files are matched to segment videos by segment basename. Each annotation file should contain a `shapes` list. Each shape is treated as one person and should include:

- `label`: the real person ID
- `points`: two rectangle corners, `[[x1, y1], [x2, y2]]`

### Submit a Job

From the SAM3 directory on the cluster:

```bash
cd /home/zli33/projects/sam-body4d
sbatch job_scripts/masklets_batch_ingroup.sh -cam 2 -b 1
```

Run multiple batches in one job:

```bash
sbatch job_scripts/masklets_batch_ingroup.sh -cam 2 -b 1,3-5
```

Use a different annotation-to-video coordinate scale:

```bash
sbatch job_scripts/masklets_batch_ingroup.sh -cam 2 -b 1,3-5 -cam_res_scale 0.5
```

### Arguments

- `-cam <num>`: camera number. The script formats it as `cam%02d`.
- `-b <spec>`: batch number specification in the range `1` to `7`.
  Supported forms are `1`, `1,3,5`, `3-5`, and `1,3-5`.
- `-cam_res_scale <float>`: scale applied to annotation bounding boxes before converting them to SAM coordinates. Default is `0.5`.

For each selected batch, the script constructs:

```text
cam%02d_batch%02d
```

For example, `-cam 2 -b 1` processes `cam02_batch01`.

### Runtime Behavior

Before running inference, the Slurm script attempts to create a code snapshot under:

```text
/scratch/zli33/data/ingroup/bbox_kp/_snapshots/
```

If the snapshot succeeds, the container runs from that snapshot. If it fails, the job falls back to the live repository under `/home/zli33/projects/sam-body4d`.

The job binds the checkpoint, INGroup data, SAM4D data, and home directories into the Apptainer image:

```text
/scratch/zli33/models/sam4d_checkpoints -> /mnt/sam4d_checkpoints
/scratch/zli33/data/ingroup          -> /mnt/data/ingroup
/scratch/zli33/data/sam4d            -> /mnt/data/sam4d_body
/home/zli33                          -> /mnt/home/zli33
```

### Outputs

For each sequence, outputs are written to:

```text
/scratch/zli33/data/ingroup/bbox_kp/<sequence>/
```

Typical output files include:

- `masks.zip`: palette mask PNGs with frame-level object IDs
- `images.zip`: extracted RGB frames
- `video_mask.mp4`: visualization video with masks overlaid
- `mask_bbox.json`: bounding boxes extracted from generated masks
- `id_mapping.json`: per-segment mapping from consecutive SAM IDs to real person IDs
- `masklets_meta.json`: run metadata
- `gpu_mem_stage1_batch.json`: GPU memory summary

Slurm logs are written to:

```text
/scratch/zli33/slurm_outputs/sam_4d_body/slurm_<job_id>.out
/scratch/zli33/slurm_outputs/sam_4d_body/slurm_<job_id>.err
```
