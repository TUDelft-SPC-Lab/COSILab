# ViTPose for INGroup

This directory contains a local adaptation of the original ViTPose repository:

https://github.com/ViTAE-Transformer/ViTPose

Refer to the upstream repository for the original model documentation, installation guide, training details, and citation. The notes below only document the INGroup-specific pose-estimation workflow added in this project.

## Pipeline Overview

The INGroup pose pipeline is:

```text
SAM3 mask output
  -> mask_bbox.json + images.zip
  -> ViTPose keypoint inference
  -> vitpose_keypoints.json
  -> dataframe conversion
  -> vitpose_dataframe.pkl
```

Main project-specific entry points:

- Finetuning job: `slurm/submit.sh`
- Finetuning config: `configs/ViTPose_coco_plus_conflab_w_bg_256x192.py`
- ViTPose inference job: `slurm/ingroup_daic_zonghuan.sh`
- ViTPose inference script: `demo/infer_mask_bbox.py`
- Dataframe conversion job: `slurm/submit_vitpose_dataframe_daic.sh`
- Dataframe conversion script: `demo/vitpose_to_dataframe.py`

## 0. Finetuning the ViTPose Model

The finetuned model used by the INGroup inference jobs is trained with:

```text
slurm/submit.sh
```

This job launches `tools/train.py` inside the ViTPose Apptainer image:

```bash
python /workspace/tools/train.py \
  configs/ViTPose_coco_plus_conflab_w_bg_256x192.py \
  --work-dir work_dirs/ViTPose_coco_plus_conflab_w_bg_256x192_filtered \
  --cfg-options model.pretrained=data/conflab/models/vitpose_base_coco_aic_mpii.pth \
  --seed 0
```

Submit it from the ViTPose repository on DAIC:

```bash
cd /home/nfs/zli33/projects/ViTPose
sbatch --job-name vitpose-conflab \
  --account ewi-insy-prb \
  --partition insy,general \
  slurm/submit.sh
```

The Slurm wrapper mounts:

```text
current repo                           -> /workspace
/tudelft.net/staff-umbrella/neon/experiments/VIT003/ -> /workspace/data/conflab
/tmp                                  -> /tmp
```

The finetuning config uses the Conflab dataset definition from:

```text
configs/_base_/datasets/conflab.py
```

and reads training and validation data through the container path:

```text
/workspace/data/conflab/
  keypoints_and_bboxes_train_filtered.json
  keypoints_and_bboxes_test.json
  images_train/
  images_test/
  models/vitpose_base_coco_aic_mpii.pth
```

Important config settings:

- Base checkpoint: `data/conflab/models/vitpose_base_coco_aic_mpii.pth`
- Output channels: `17` Conflab keypoints
- Image size: `192 x 256`
- Optimizer: `AdamW`, learning rate `1e-4`
- Training length: `5` epochs
- Evaluation: every epoch with `mAP`, saving the best `AP`
- Work directory: `work_dirs/ViTPose_coco_plus_conflab_w_bg_256x192_filtered`

The best checkpoint is expected at a path such as:

```text
work_dirs/ViTPose_coco_plus_conflab_w_bg_256x192_filtered/best_AP_epoch_1.pth
```

or, on NEON when used by the INGroup inference wrapper:

```text
/tudelft.net/staff-umbrella/neon/code/ViTPose/work_dirs/ViTPose_coco_plus_conflab_w_bg_256x192_filtered/best_AP_epoch_1.pth
```

Use that checkpoint as `POSE_CKPT` when running inference:

```bash
POSE_CKPT=/path/to/best_AP_epoch_1.pth \
BATCH=cam06_batch02 \
sbatch slurm/ingroup_daic_zonghuan.sh
```

## 1. Reading SAM3 Mask Outputs

ViTPose does not read `masks.zip` directly during inference. SAM3 first writes mask images, then `run_sam3_masklets_batch.py` extracts per-person bounding boxes from those masks into `mask_bbox.json`. The ViTPose INGroup runner consumes:

```text
<sam3_output>/<batch>/
  mask_bbox.json
  images.zip
```

or, alternatively:

```text
<sam3_output>/<batch>/
  mask_bbox.json
  images/
    00000000.jpg
    00000001.jpg
```

The expected `mask_bbox.json` schema is:

```json
{
  "annotations": {
    "1": {
      "bbox": {
        "1": [x, y, w, h],
        "3": [x, y, w, h]
      }
    }
  }
}
```

The top-level keys under `annotations` are frame IDs. The keys under `bbox` are person or track IDs inherited from the SAM3 mask-ID mapping. Each bounding box is passed to ViTPose in `xywh` format.

The Slurm wrapper looks for SAM3 output here by default:

```text
/tudelft.net/staff-umbrella/neon/ingroup_dataset/B2_pipeline/sam3_output/<BATCH>/
```

## 2. Running INGroup ViTPose

Use `slurm/ingroup_daic_zonghuan.sh` to run ViTPose on one SAM3 batch output directory.

Default paths in the job:

```text
NEON=/tudelft.net/staff-umbrella/neon
VITPOSE_DIR=/home/nfs/zli33/projects/ViTPose
SIF=${NEON}/apptainer/vitpose-0.0.5.sif
DATA_ROOT=${NEON}/ingroup_dataset/B2_pipeline/sam3_output/${BATCH}
OUT_DIR=${NEON}/ingroup_dataset/B2_pipeline/vitpose_results/${BATCH}
```

The job implementation does the following:

1. Loads CUDA and checks that `DATA_ROOT` exists.
2. Creates `OUT_DIR`.
3. Extracts `DATA_ROOT/images.zip` into `/tmp/ingroup_images/${BATCH}` when the zip exists.
4. Falls back to `DATA_ROOT/images` if an unzipped image directory already exists.
5. Runs `demo/infer_mask_bbox.py` inside the Apptainer image.
6. Writes `vitpose_keypoints.json` and, with `--save-video`, `vitpose_kp.mp4`.
7. Removes the temporary extracted images from `/tmp`.

Submit one batch:

```bash
cd /home/nfs/zli33/projects/ViTPose
BATCH=cam06_batch02 sbatch slurm/ingroup_daic_zonghuan.sh
```

Override the checkpoint:

```bash
POSE_CKPT=/path/to/best_AP_epoch_5.pth \
BATCH=cam06_batch02 \
sbatch slurm/ingroup_daic_zonghuan.sh
```

Override the SAM3 input root:

```bash
DATA_ROOT=/path/to/sam3_output/cam06_batch02 \
BATCH=cam06_batch02 \
sbatch slurm/ingroup_daic_zonghuan.sh
```

Run several batches:

```bash
for b in cam06_batch01 cam06_batch02 cam08_batch01 cam10_batch01; do
  BATCH=$b sbatch slurm/ingroup_daic_zonghuan.sh
done
```

The inference command inside the Slurm job is:

```bash
python demo/infer_mask_bbox.py \
  configs/ViTPose_coco_plus_conflab_w_bg_256x192.py \
  "${POSE_CKPT}" \
  --data-root "${DATA_ROOT}" \
  --img-dir "${IMG_DIR}" \
  --out-dir "${OUT_DIR}" \
  --save-video
```

The resulting output layout is:

```text
/tudelft.net/staff-umbrella/neon/ingroup_dataset/B2_pipeline/vitpose_results/
  cam06_batch02/
    vitpose_keypoints.json
    vitpose_kp.mp4
```

`vitpose_keypoints.json` keeps the SAM3 person IDs and adds a 17-keypoint Conflab pose for each person:

```json
{
  "annotations": {
    "1": {
      "bbox": {
        "1": [x, y, w, h]
      },
      "keypoints": {
        "1": [[x, y, score]]
      }
    }
  }
}
```

## 3. Converting ViTPose JSON to Dataframe PKL

Use `slurm/submit_vitpose_dataframe_daic.sh` after `vitpose_keypoints.json` files have been generated.

Default paths in the conversion job:

```text
RESULTS_ROOT=${NEON}/ingroup_dataset/B2_pipeline/vitpose_results
OUTPUT_ROOT=${NEON}/ingroup_dataset/B2_pipeline/vitpose_dataframe
CAMERA_PARAMS_ROOT=${NEON}/ingroup_dataset/processed_data/gopro_data/camera_calibration/camera_params
GT_GROUPS_ROOT=${NEON}/ingroup_dataset/B2_pipeline/cgroup_annotation
PLOT_DIR=${NEON}/ingroup_dataset/B2_pipeline/person_plotting
FRAMES_ROOT=${NEON}/ingroup_dataset/B2_pipeline/video_segs_raw
```

Submit conversion for all cameras found under `RESULTS_ROOT`:

```bash
cd /home/nfs/zli33/projects/ViTPose
sbatch slurm/submit_vitpose_dataframe_daic.sh
```

Convert only selected cameras:

```bash
sbatch slurm/submit_vitpose_dataframe_daic.sh --cam=06,08,10
```

Override input and output roots:

```bash
RESULTS_ROOT=/path/to/vitpose_results \
OUTPUT_ROOT=/path/to/vitpose_dataframe \
sbatch slurm/submit_vitpose_dataframe_daic.sh
```

The conversion job runs:

```bash
python demo/vitpose_to_dataframe.py \
  "${RESULTS_ROOT}" \
  --output_dir "${OUTPUT_ROOT}" \
  --output_name vitpose_dataframe.pkl \
  --camera_params_root "${CAMERA_PARAMS_ROOT}" \
  --gt_groups_root "${GT_GROUPS_ROOT}" \
  --plot_dir "${PLOT_DIR}" \
  --frames_root "${FRAMES_ROOT}"
```

The converter scans:

```text
<RESULTS_ROOT>/cam*/vitpose_keypoints.json
```

For every camera or camera-batch folder, it writes:

```text
<OUTPUT_ROOT>/<cam_or_batch_name>/vitpose_dataframe.pkl
```

Each dataframe row represents one frame. Important columns are:

- `timestamp`: frame ID from `vitpose_keypoints.json`
- `time`: wall-clock time derived from camera number, batch number, and 60 fps timing assumptions
- `spaceFeat`: dict with `head`, `shoulder`, `hip`, and `foot`
- `groups`: ground-truth conversational groups loaded from `mingle_1_groups.csv` or `mingle_2_groups.csv` when available
- `group_ids`: empty placeholder list

Each `spaceFeat` segment is an `(n_people, 4)` object array:

```text
[person_id, x, y, orientation]
```

The `x` and `y` coordinates are world-floor coordinates. The converter back-projects 2D ViTPose keypoints using:

```text
<CAMERA_PARAMS_ROOT>/camera_XX/intrinsic.json
<CAMERA_PARAMS_ROOT>/camera_XX/extrinsic.json
```

Diagnostic plots are written under:

```text
<PLOT_DIR>/camXX/
```

when `--plot_dir` is provided by the Slurm wrapper.

Slurm logs for both ViTPose inference and dataframe conversion are written to:

```text
/home/nfs/zli33/slurm_outputs/vitpose-ingroup/infer_slurm_<job_id>.out
/home/nfs/zli33/slurm_outputs/vitpose-ingroup/infer_slurm_<job_id>.err
```
