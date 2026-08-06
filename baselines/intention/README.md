# Intention Baseline: Gemma Inference and Annotation Analysis

This folder contains the INGroup intention-recognition baseline that prompts a multimodal Gemma model on 30-second clips, plus analysis code for survey responses and model-human annotation comparison.

The inference code is packaged as an installable Python project (`intention_inference`) built with [uv](https://docs.astral.sh/uv/). The annotation and survey analysis code is kept separately as standalone notebooks and R scripts.

## Layout

```text
pyproject.toml                      project metadata and dependencies
uv.lock                             locked dependency versions
src/intention_inference/
  cli.py                            argument parsing and the inference run loop
  manifest.py                       manifest loading and dotted-key lookup
  media.py                          media path resolution and prefix rewriting
  audio.py                          speaker selection and conversation-floor mixing
  prompt.py                         prompt config loading and template rendering
  gemma.py                          Gemma chat template, media decoding, generation
  prompt_ingroup.json               default prompt configuration
  __main__.py                       entry point for `python -m intention_inference`
job_scripts/
  gemma_cosilab_daic.sh             Slurm submission wrapper for DAIC
```

## Installation

1. Install [uv](https://docs.astral.sh/uv/)
2. Clone this repo
3. Go into this folder
    ```bash
    cd COSILab/baselines/intention
    ```
4. Create a venv and install the dependencies
    ```bash
    uv sync
    ```

The project requires Python >= 3.11 and depends on `torch`, `transformers`, `accelerate`, `numpy`, `librosa`, `soundfile`, `Pillow`, and `av`. Installing exposes a console script named `intention-inference`; the same entry point is reachable as `python -m intention_inference`.

## How the Inference Works

The CLI expects an input JSON manifest containing one record per inference item. The top-level JSON can be a list, or an object containing one of:

```text
records
entries
items
data
```

Each retained record is converted into one Gemma chat turn. For each record the pipeline prepares:

- an indicated participant image from `participant_<id>.png` under `--participant-image-root`, selected by the record's `participant` field
- the video clip from the record's `video` field
- the participant's own audio from `audios[participant - 1]`
- aggregated conversation-floor audio mixed from the speaker IDs in `conversation_floor` (the participant's own ID is excluded and duplicates are dropped)
- a rendered user prompt from the prompt config

Records are skipped before inference, with the reason recorded, when the video or participant image is missing or not on disk, when the audio list or speaker selection is malformed, when participant or conversation-floor audio cannot be resolved, when audio aggregation fails, or when a resolved path matches an `--exclude-video-substring` / `--exclude-audio-substring` filter.

The prompt config provides:

- `system_prompt`: the Gemma system instruction
- `user_prompt_template`: the actual intention-recognition prompt

If `user_prompt_template` (or `prompt`) is absent, the loader assembles the template from structured `intro`, `questions`, and `examples` fields. Templates are rendered with the record's flattened fields, so `{field}` and `{nested.field}` placeholders are filled from the record, `{record_json}` expands to the whole record, and unknown placeholders are left untouched.

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

Images, audio, and video frames are decoded ahead of the processor call with Pillow, librosa, and PyAV. Video frames are sampled evenly up to `--max-video-frames`, capped at the clip's real frame count. If the processor ships no chat template, a built-in Gemma 4 fallback template is used.

For every record, the model input contains:

```text
image: participant reference image
audio: participant audio
audio: mixed conversation-floor audio
video: sampled video frames
text: system prompt + rendered user prompt
```

The aggregated conversation-floor mixes are written next to the output file, under:

```text
<output_dir>/_audio_mixes/<output_stem>/<index>_<record_id>_conversation_floor.wav
```

Floor tracks whose length differs from the participant track by more than the tolerance are discarded with a warning; if nothing remains, silence is used. The mix is peak-normalized when it clips.

The output JSON contains:

- `__summary__`: run metadata, selected/retained/processed counts, and skip reasons
- `__skipped__`: records skipped before inference and why
- `results`: one item per processed record, including source/rewritten/resolved media paths, speaker IDs, audio warnings, the rendered prompt, and the Gemma response in `assistant`

Generation errors do not abort the run: the failing record's `assistant` field is set to `[ERROR] ...` and counted in `__summary__.error_count`.

## Run Locally

```bash
uv run intention-inference \
  --model /path/to/GemmaE4B \
  --input-json /path/to/annotation_clips.json \
  --output /path/to/model_responses/annotation_clips.json \
  --participant-image-root /path/to/participant_imgs \
  --video-media-path-prefix "https://example/video_segs" \
  --video-local-path-prefix /path/to/annotation_video \
  --audio-media-path-prefix "https://example/audio_segs_normalized" \
  --audio-local-path-prefix /path/to/annotation_audio \
  --max-video-frames 32 \
  --max-new-tokens 512
```

Only `--input-json` is required. `--model` defaults to the `MODEL_PATH` environment variable (falling back to `/scratch/zli33/models/GemmaE4B`), `--output` to `ingroup_results.json`, and `--prompt-config` to the packaged `prompt_ingroup.json`.

Media path options:

- `--media-path-prefix` / `--local-path-prefix`: rewrite a shared source prefix (for example the covfee URL stored in the manifest) to a local filesystem prefix. Providing a media prefix without a local prefix is an error.
- `--video-media-path-prefix` / `--video-local-path-prefix`: video-specific overrides of the shared pair.
- `--audio-media-path-prefix` / `--audio-local-path-prefix`: audio-specific overrides of the shared pair.
- `--media-root`: root used to resolve relative media paths that do not exist next to the manifest.
- `--participant-image-root`: folder containing `participant_<n>.png`.

Selection and filtering:

- `--limit N`: process only the first `N` retained records.
- `--start-index X --end-index Y`: process a zero-based manifest index range, inclusive.
- `--exclude-video-substring S` / `--exclude-audio-substring S`: skip records whose resolved media path contains `S`. Both are repeatable.
- `--id-key`: record key used as the stable result id, dotted paths supported. Defaults to `id`.

Prompting and generation:

- `--system-prompt`: override the system prompt from the prompt config.
- `--no-audio`: omit audio inputs and run video-only inference (no conversation-floor mixing).
- `--enable-thinking`: enable Gemma thinking mode when the chat template supports it.
- `--do-sample`: use sampling instead of deterministic generation; `--temperature`, `--top-p`, and `--top-k` apply only in this mode.
- `--max-new-tokens`, `--max-video-frames`: generation length and video frame budget.

## Submit on DAIC

Use:

```text
job_scripts/gemma_cosilab_daic.sh
```

Default DAIC paths in the script:

```text
PROJECT_ROOT=/home/zli33/linuxhome/projects/COSILab
SIF_PATH=/tudelft.net/staff-umbrella/neon/apptainer/gemma.sif
DATA_ROOT=/tudelft.net/staff-umbrella/neon/ingroup_dataset
MODEL_PATH=/tudelft.net/staff-umbrella/neon/zonghuan/models/GemmaE4B
input_json=/tudelft.net/staff-umbrella/neon/B1_pipeline/annotation_clips.json
output_dir=/tudelft.net/staff-umbrella/neon/B1_pipeline/model_responses
```

`PROJECT_ROOT` is the COSILab checkout, overridable with `--project-root` or the `PROJECT_ROOT` environment variable. It is bound to `/workspace` in the container, and `PYTHONPATH` is set to `/workspace/baselines/intention/src`, so the job always runs the checkout's code rather than a copy baked into the image. `--prompt-config` defaults to the `prompt_ingroup.json` shipped inside the package under that checkout.

Submit with defaults:

```bash
sbatch baselines/intention/job_scripts/gemma_cosilab_daic.sh
```

Process a specific manifest index range:

```bash
sbatch baselines/intention/job_scripts/gemma_cosilab_daic.sh --index-range 0-99
```

Run video-only inference:

```bash
sbatch baselines/intention/job_scripts/gemma_cosilab_daic.sh --no-audio
```

Override model, input, and output:

```bash
sbatch baselines/intention/job_scripts/gemma_cosilab_daic.sh \
  --model-path /path/to/GemmaE4B \
  --input-json /path/to/annotation_clips.json \
  --output /path/to/model_responses/run.json
```

The wrapper:

1. Validates the model, SIF, checkout, prompt config, and input manifest.
2. Maps remote media URL prefixes in the manifest to local DAIC filesystem paths.
3. Builds the `python -m intention_inference` command with prompt, model, media-prefix, and range options.
4. Checks inside the container that `intention_inference` resolves to the checkout, then runs the command with `srun apptainer exec --nv`.
5. Writes logs under:

```text
/home/zli33/linuxhome/slurm_outputs/gemma/
```

## Annotation and Survey Analysis

These notebooks and R scripts are intended for post-inference evaluation and annotator analysis. They are not part of the `intention_inference` package and have their own dependencies; they assume local CSV/JSON outputs from the annotation workflow and are not self-contained data bundles.

Notebooks:

- `process_data.ipynb`: parses raw Gemma responses, flattens structured intention annotations, optionally labels annotation dimensions with an LLM helper, and prepares CSV tables for downstream analysis.
- `semantic_similarity.ipynb`: compares model and human intention annotations with sentence-embedding similarity and UMAP projections.
- `LLM_as_judge.ipynb`: runs a paired model-vs-human annotation comparison where an LLM judge predicts which annotation was human-written.

R scripts and supporting files:

- `presurvey-processor.R`: processes the pre-annotation survey using `pre_annotation_survey_codebook.xlsx`, scores survey measures, recodes demographics, and writes a cleaned survey CSV.
- `annotator-analysis.R`: joins cleaned survey measures with annotation outputs and fits mixed-effects models for annotator behavior.
- `pre_annotation_survey_codebook.xlsx`: codebook used to map and score pre-annotation survey columns.

Results:

- `benchmark_sim.md`: recorded model-human answer similarity tables and persona-level similarity comparisons.

Typical sequence:

1. Run Gemma inference with `intention-inference` or `job_scripts/gemma_cosilab_daic.sh`.
2. Use `process_data.ipynb` to load model response JSON files, parse the assistant text into structured rows, and write tables such as `df_model.csv` and LLM-labeled annotation CSVs.
3. Use `semantic_similarity.ipynb` to compare human annotations and model annotations by embedding intention descriptions or explanations, computing pairwise similarity, and visualizing sources such as annotator A, annotator B, final human annotations, and model outputs.
4. Use `LLM_as_judge.ipynb` for a complementary comparison where an LLM receives paired model/human annotations for the same item and predicts which one was human-written.
5. Use `presurvey-processor.R` to convert raw pre-annotation survey exports into scored participant-level measures.
6. Use `annotator-analysis.R` to merge survey measures with annotation outputs and test relationships between annotator characteristics and annotation content.

These files may contain local path placeholders from the original analysis environment. Update input paths and filenames before running them locally.
