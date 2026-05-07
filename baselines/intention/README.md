# Intention Baseline: Gemma Inference and Annotation Analysis

This folder contains the INGroup intention-recognition baseline that prompts a multimodal Gemma model on 30-second clips, plus analysis code for survey responses and model-human annotation comparison.

Main files:

- `model_inference.py`: prepares each manifest record, builds Gemma multimodal inputs, runs inference, and writes model responses.
- `prompt_ingroup.json`: prompt configuration used to ask the model for participant intentions.
- `gemma_ingroup_daic.sh`: Slurm submission wrapper for running Gemma inference on DAIC.
- `process_data.ipynb`: parses raw Gemma responses, flattens structured intention annotations, optionally labels annotation dimensions with an LLM helper, and prepares CSV tables for downstream analysis.
- `semantic_similarity.ipynb`: compares model and human intention annotations with sentence-embedding similarity and UMAP projections.
- `LLM_as_judge.ipynb`: runs a paired model-vs-human annotation comparison where an LLM judge predicts which annotation was human-written.
- `presurvey-processor.R`: processes the pre-annotation survey using `pre_annotation_survey_codebook.xlsx`, scores survey measures, recodes demographics, and writes a cleaned survey CSV.
- `annotator-analysis.R`: joins cleaned survey measures with annotation outputs and fits mixed-effects models for annotator behavior.
- `pre_annotation_survey_codebook.xlsx`: codebook used to map and score pre-annotation survey columns.

## How `model_inference.py` Uses Gemma

`model_inference.py` expects an input JSON manifest containing one record per inference item. The top-level JSON can be a list, or an object containing one of:

```text
records
entries
items
data
```

Each retained record is converted into one Gemma chat turn. The script prepares:

- an indicated participant image from `participant_<id>.png`
- the video clip from the record's `video` field
- the participant's own audio from `audios[participant - 1]`
- aggregated conversation-floor audio from the IDs in `conversation_floor`
- a rendered user prompt from `prompt_ingroup.json`

The prompt config provides:

- `system_prompt`: the Gemma system instruction
- `user_prompt_template`: the actual intention-recognition prompt

The prompt asks Gemma to identify the indicated participant's intentions, including timestamps, confidence, reasoning, intensity, and counterfactual explanations. If no clear intention is visible, the model is instructed to return the no-intention format.

Inference uses Hugging Face Transformers:

```python
processor = AutoProcessor.from_pretrained(args.model)
model = AutoModelForMultimodalLM.from_pretrained(
    args.model,
    dtype="auto",
    device_map="auto",
)
```

For every record, the model input contains:

```text
image: participant reference image
audio: participant audio
audio: mixed conversation-floor audio
video: sampled video frames
text: system prompt + rendered user prompt
```

The output JSON contains:

- `__summary__`: run metadata and skip counts
- `__skipped__`: records skipped before inference and why
- `results`: one item per processed record, including media paths, rendered prompt, and Gemma response in `assistant`

Note: `model_inference.py` imports shared helper functions from `batch_infer_context.py`. Make sure that helper module is available in the same Python environment when running locally or on DAIC.

## Run Locally

Example direct command:

```bash
python model_inference.py \
  --model /path/to/GemmaE4B \
  --input-json /path/to/annotation_clips.json \
  --output /path/to/model_responses/annotation_clips.json \
  --prompt-config prompt_ingroup.json \
  --video-media-path-prefix "https://example/video_segs" \
  --video-local-path-prefix /path/to/annotation_video \
  --audio-media-path-prefix "https://example/audio_segs_normalized" \
  --audio-local-path-prefix /path/to/annotation_audio \
  --max-video-frames 32 \
  --max-new-tokens 512
```

Useful options:

- `--no-audio`: omit audio inputs and run video-only inference.
- `--limit N`: process only the first `N` retained records.
- `--start-index X --end-index Y`: process a zero-based manifest index range.
- `--enable-thinking`: enable Gemma thinking mode when supported.
- `--do-sample`: use sampling instead of deterministic generation.

## Submit on DAIC

Use:

```text
gemma_ingroup_daic.sh
```

Default DAIC paths in the script:

```text
PROJECT_ROOT=/home/zli33/linuxhome/projects/vlm_social
SIF_PATH=/tudelft.net/staff-umbrella/neon/apptainer/gemma.sif
DATA_ROOT=/tudelft.net/staff-umbrella/neon/ingroup_dataset
MODEL_PATH=/tudelft.net/staff-umbrella/neon/zonghuan/models/GemmaE4B
input_json=/tudelft.net/staff-umbrella/neon/B1_pipeline/annotation_clips.json
output_dir=/tudelft.net/staff-umbrella/neon/B1_pipeline/model_responses
```

Submit with defaults:

```bash
sbatch baselines/intention/gemma_ingroup_daic.sh
```

Process a specific manifest index range:

```bash
sbatch baselines/intention/gemma_ingroup_daic.sh --index-range 0-99
```

Run video-only inference:

```bash
sbatch baselines/intention/gemma_ingroup_daic.sh --no-audio
```

Override model, input, and output:

```bash
sbatch baselines/intention/gemma_ingroup_daic.sh \
  --model-path /path/to/GemmaE4B \
  --input-json /path/to/annotation_clips.json \
  --output /path/to/model_responses/run.json
```

The wrapper:

1. Validates the model, SIF, prompt config, and input manifest.
2. Maps remote media URL prefixes in the manifest to local DAIC filesystem paths.
3. Builds the `python ... model_inference.py` command with prompt, model, media-prefix, and range options.
4. Runs it with `srun apptainer exec --nv`.
5. Writes logs under:

```text
/home/zli33/linuxhome/slurm_outputs/gemma/
```

## Annotation and Survey Analysis

The analysis notebooks and R scripts are intended for post-inference evaluation and annotator analysis. They assume local CSV/JSON outputs from the annotation workflow and are not self-contained data bundles.

Typical sequence:

1. Run Gemma inference with `model_inference.py` or `gemma_ingroup_daic.sh`.
2. Use `process_data.ipynb` to load model response JSON files, parse the assistant text into structured rows, and write tables such as `df_model.csv` and LLM-labeled annotation CSVs.
3. Use `semantic_similarity.ipynb` to compare human annotations and model annotations by embedding intention descriptions or explanations, computing pairwise similarity, and visualizing sources such as annotator A, annotator B, final human annotations, and model outputs.
4. Use `LLM_as_judge.ipynb` for a complementary comparison where an LLM receives paired model/human annotations for the same item and predicts which one was human-written.
5. Use `presurvey-processor.R` to convert raw pre-annotation survey exports into scored participant-level measures.
6. Use `annotator-analysis.R` to merge survey measures with annotation outputs and test relationships between annotator characteristics and annotation content.

These files may contain local path placeholders from the original analysis environment. Update input paths and filenames before running them locally.
