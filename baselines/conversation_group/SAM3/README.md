# SAM3 / SAM-Body4D for INGroup

This directory contains a local adaptation of the original SAM-Body4D repository:

https://github.com/gaomingqi/sam-body4d

Refer to the upstream repository for the original installation guide, model details, citation, and general SAM-Body4D usage. The notes below only document the INGroup mask-generation workflow added in this project.

## Generate Masks for INGroup

The INGroup mask workflow runs SAM-3 masklet extraction over 10-second video segments, using the first-frame person bounding-box annotations for each segment.

Main entry points:

- Slurm batch script: `job_scripts/masklets_batch_ingroup.sh`
- Python runner called by the job: `run_sam3_masklets_batch.py`
- SAM-Body4D config: `configs/body4d.yaml`

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
