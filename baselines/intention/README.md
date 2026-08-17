# Intention Baseline: Multimodal Inference and Annotation Analysis

This folder contains the INGroup intention-recognition baseline that prompts a multimodal model on 30-second clips, plus analysis code for survey responses and model-human annotation comparison.

Two things vary independently, and neither is a property of the code. Which model runs is a `--backend` choice -- `gemma` and `qwen7b` today -- and the task itself imports no torch and knows nothing about either. What is asked of it, and what grounds the question, is a `--mode` choice: `participant_image` today, `participant_ids` reserved as a placeholder. Any mode runs on any backend.

The inference code is packaged as an installable Python project (`intention_inference`) built with [uv](https://docs.astral.sh/uv/). The annotation and survey analysis code is kept separately as standalone notebooks and R scripts.

## Layout

```text
pyproject.toml                      project metadata and dependencies
uv.lock                             locked dependency versions
src/intention_inference/            the task: what to ask, of which clips
  cli.py                            argument parsing and the order of operations
  engine.py                         the inference pass, skip accounting, result writing
  manifest.py                       manifest loading and dotted-key lookup
  media.py                          media path resolution and prefix rewriting
  audio.py                          speaker selection and conversation-floor mixing
  records.py                        one record -> the clip and audio on this filesystem
  prompt.py                         prompt config loading and template rendering
  model_config.json                 weights, decoding parameters, frame policy
  modes/                            the task modes: what is asked, and what grounds it
    base.py                         the interface every mode implements
    registry.py                     mode name -> class, imported lazily
    participant_image.py            grounding by a reference photo (the default today)
    participant_ids.py              grounding by participant id (PLACEHOLDER)
    prompt_participant_image.json   the participant_image mode's prompt
    prompt_participant_ids.json     stub, unwritten
  __main__.py                       entry point for `python -m intention_inference`
src/models/                         the model backends the task runs on
  base.py                           the interface every backend implements
  registry.py                       backend name -> class, imported lazily
  config.py                         model_config.json -> a resolved run config
  media_io.py                       video/audio/image decoding, frame policy
  gemma/                            Gemma 4 via transformers
  qwen/                             Qwen2.5-Omni via transformers
job_scripts/
  intention/cosilab_daic.sh         Slurm stub: header + three host paths
  lib/intention_job.sh              all job logic and options
  lib/intention_paths.sh            data, manifest and output paths
  lib/model_backends.sh             backend -> container image
```

No model code and no weights live in this repo. `models/` holds thin wrappers
around `transformers`; `qwen_omni_utils` comes from the container image (or from
`QWEN_OMNI_UTILS_PATH`), and the checkpoints are named only in
`model_config.json`.

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

The project requires Python >= 3.11 and depends on `torch`, `transformers`, `accelerate`, `numpy`, `librosa`, `soundfile`, `Pillow`, `av`, and `qwen-omni-utils`. Installing exposes a console script named `intention-inference`; the same entry point is reachable as `python -m intention_inference`.

## How the Inference Works

The CLI expects an input JSON manifest containing one record per inference item. The top-level JSON can be a list, or an object containing one of:

```text
records
entries
items
data
```

Each retained record is converted into one chat turn. Every mode gets the same clip and the same audio, prepared by `records.py`:

- the video clip from the record's `video` field
- the participant's own audio from `audios[participant - 1]`
- aggregated conversation-floor audio mixed from the speaker IDs in `conversation_floor` (the participant's own ID is excluded and duplicates are dropped)

What the mode adds is the grounding -- which reference media says *who* is being asked about -- plus the rendered prompt and the order of the parts. For `participant_image` the grounding is an indicated participant image from `participant_<id>.png` under `--participant-image-root`, selected by the record's `participant` field.

Records are skipped before inference, with the reason recorded, when the video or the mode's reference media is missing or not on disk, when the audio list or speaker selection is malformed, when participant or conversation-floor audio cannot be resolved, when audio aggregation fails, or when a resolved path matches an `--exclude-video-substring` / `--exclude-audio-substring` filter.

Each mode ships its own prompt config, and `--prompt-config` overrides it. A prompt config provides:

- `system_prompt`: the system instruction, folded into the user text
- `user_prompt_template`: the actual intention-recognition prompt

If `user_prompt_template` (or `prompt`) is absent, the loader assembles the template from structured `intro`, `questions`, and `examples` fields. Templates are rendered with the record's flattened fields, so `{field}` and `{nested.field}` placeholders are filled from the record, `{record_json}` expands to the whole record, and unknown placeholders are left untouched.

The `participant_image` prompt asks the model to identify the indicated participant's intentions, including timestamps, confidence, reasoning, intensity, and counterfactual explanations. If no clear intention is visible, the model is instructed to return the no-intention format. It never names the participant by number: the grounding is entirely facial, and the conversation-floor speakers stay anonymous because their tracks arrive summed into one soundtrack.

The model is reached only through `models/`: `--backend` names it, `models/registry.py` resolves the name to a class, and `model_config.json` says which weights it loads with what decoding parameters. Task code never imports torch.

The question is reached the same way through `modes/`: `--mode` names it, `modes/registry.py` resolves the name to a class, and the mode owns its reference media, its prompt config and its turn layout. Add a mode by implementing `modes.base.BaseTaskMode` and adding one line to `MODE_MODULES`. `BaseTaskMode.prepare_record` is the default composition -- clip, reference media, audio, prompt -- and a mode that does not fit it overrides it and composes `records.resolve_video_media` / `records.resolve_audio_bundle` itself.

`participant_ids` is registered but unimplemented: selecting it raises at startup, before the model config or the manifest is read, and says what remains to be written.

Frame sampling is one policy for every backend, set in `model_config.json` under `defaults`:

```text
video_fps         target rate (2.0)
min_video_frames  floor (4)
max_video_frames  ceiling (32)
```

`models/media_io.py` `frames_at_fps` resolves them once -- `n = duration * fps`, clamped, rounded down to even -- and every backend is handed that exact count, so a difference between two models is the model rather than how much video each one saw. A 3-second clip gives 6 frames; anything under 2 seconds lands on the floor, anything over 16 seconds on the ceiling.

Images, audio, and video frames are decoded ahead of the processor call with Pillow, librosa, and PyAV. If a Gemma processor ships no chat template, a built-in Gemma 4 fallback template is used.

The turn layout belongs to the mode, because part order is the order the model meets the placeholders in. For `participant_image`, every record's input contains:

```text
image: participant reference image
audio: participant audio
audio: mixed conversation-floor audio
video: sampled video frames
text: system prompt + rendered user prompt
```

Note that the `<image>`, `<audio1>` and `<audio2>` markers in that prompt's text are literal characters, not placeholders: the real media tokens are emitted at the parts' positions, all of them ahead of the text block. The binding between the prose and the media therefore rests on part order alone. Worth not reproducing in a new mode -- interleave text between the parts instead, or describe them strictly by order.

The aggregated conversation-floor mixes are written next to the output file, under:

```text
<output_dir>/_audio_mixes/<output_stem>/<index>_<record_id>_conversation_floor.wav
```

Floor tracks whose length differs from the participant track by more than the tolerance are discarded with a warning; if nothing remains, silence is used. The mix is peak-normalized when it clips.

The output JSON contains:

- `__summary__`: run metadata, selected/retained/processed counts, and skip reasons
- `__skipped__`: records skipped before inference and why
- `results`: one item per processed record, including source/rewritten/resolved media paths, speaker IDs, audio warnings, the rendered prompt, and the model response in `assistant`

`__summary__` records which backend ran and in which mode, the resolved weights path, the decoding parameters and the frame policy, so a result file answers "what produced this" without anyone having to find the command that made it. Beyond the shared keys, each result row carries whatever the mode's `result_fields` contributes -- for `participant_image`, the reference image that was shown.

Generation errors do not abort the run: the failing record's `assistant` field is set to `[ERROR] ...` and counted in `__summary__.error_count`.

## Run Locally

```bash
uv run intention-inference \
  --backend gemma \
  --mode participant_image \
  --input-json /path/to/annotation_clips.json \
  --participant-image-root /path/to/participant_imgs \
  --video-media-path-prefix "https://example/video_segs" \
  --video-local-path-prefix /path/to/annotation_video \
  --audio-media-path-prefix "https://example/audio_segs_normalized" \
  --audio-local-path-prefix /path/to/annotation_audio
```

`--backend`, `--mode` and `--input-json` are required. Neither `--backend` nor `--mode` is defaulted: between them they decide which weights ran and what was asked, so a run should never be attributed to a model or a question chosen by omission.

`--output` defaults to `ingroup_results/<backend>/<mode>/<input stem>.json`, with the index range appended when one is given. Both are directory components because two runs differing only in the model, or only in the question, otherwise produce the same file name and the second silently overwrites the first.

`--model-config` defaults to the `model_config.json` shipped inside the package. `--prompt-config` defaults to whichever config the chosen mode ships with, under `modes/`.

Media path options:

- `--media-path-prefix` / `--local-path-prefix`: rewrite a shared source prefix (for example the covfee URL stored in the manifest) to a local filesystem prefix. Providing a media prefix without a local prefix is an error.
- `--video-media-path-prefix` / `--video-local-path-prefix`: video-specific overrides of the shared pair.
- `--audio-media-path-prefix` / `--audio-local-path-prefix`: audio-specific overrides of the shared pair.
- `--media-root`: root used to resolve relative media paths that do not exist next to the manifest.
- `--participant-image-root`: folder containing `participant_<n>.png`. Used by the `participant_image` mode.

Selection and filtering:

- `--limit N`: process only the first `N` retained records.
- `--start-index X --end-index Y`: process a zero-based manifest index range, inclusive.
- `--exclude-video-substring S` / `--exclude-audio-substring S`: skip records whose resolved media path contains `S`. Both are repeatable.
- `--id-key`: record key used as the stable result id, dotted paths supported. Defaults to `id`.

Prompting:

- `--system-prompt`: override the system prompt from the prompt config.
- `--no-audio`: omit audio inputs and run video-only inference (no conversation-floor mixing).

Model and generation live in `model_config.json`, not on the command line:

```text
backends.<name>.model_id     the weights (the only place they are named)
backends.<name>.load         constructor keywords for that backend
backends.<name>.generation   decoding overrides on top of `defaults`
backends.<name>.extra        backend-only knobs, e.g. gemma's enable_thinking
defaults                     max_new_tokens, sampling, and the frame policy
```

There are no `--model` / `--max-new-tokens` / `--temperature` / `--top-p` / `--top-k` / `--do-sample` / `--enable-thinking` / `--max-video-frames` flags any more. Change a run by editing that file and resubmitting; the settings are then something that can be read, diffed and committed, and they are recorded into every result file. Setting a knob a backend does not support (`enable_thinking` on `qwen7b`) is refused at startup rather than dropped.

The two `model_id` values ship as `/REPLACE_ME/...` placeholders and will fail the startup check by design. Replace them with absolute paths: a relative value is treated as a Hugging Face hub id and skips that check.

## Submit on DAIC

The job is split in two, so the environment and the logic are separate files:

```text
job_scripts/intention/cosilab_daic.sh   the SLURM header and three host paths
job_scripts/lib/intention_job.sh        everything else, sourced by the stub
```

The stub carries only what `sbatch` must read literally plus `PROJECT_ROOT`, `HOME_ROOT` and `SLURM_OUTPUT_ROOT`. A second environment is a second stub, not a second copy of the logic. Run `bash job_scripts/intention/cosilab_daic.sh --help` for the full option list.

Default DAIC paths (`lib/intention_paths.sh` and `lib/model_backends.sh`):

```text
PROJECT_ROOT   /home/zli33/linuxhome/projects/COSILab
DATA_ROOT      /tudelft.net/staff-umbrella/neon/ingroup_dataset
input_json     /tudelft.net/staff-umbrella/neon/B1_pipeline/annotation_clips.json
output_dir     /tudelft.net/staff-umbrella/neon/B1_pipeline/model_responses/<backend>/<mode>
gemma  SIF     /tudelft.net/staff-umbrella/neon/apptainer/gemma.sif
qwen7b SIF     /tudelft.net/staff-umbrella/neon/apptainer/qwen2.5-omni-inference.sif
```

`PROJECT_ROOT` is bound to `/workspace` in the container and `PYTHONPATH` is set to `/workspace/baselines/intention/src`, so the job always runs the checkout's code rather than a copy baked into the image.

Submit:

```bash
sbatch baselines/intention/job_scripts/intention/cosilab_daic.sh --backend gemma --mode participant_image
sbatch baselines/intention/job_scripts/intention/cosilab_daic.sh --backend qwen7b --mode participant_image --index-range 0-99
sbatch baselines/intention/job_scripts/intention/cosilab_daic.sh --backend gemma --mode participant_image --no-audio
```

Override input and output:

```bash
sbatch baselines/intention/job_scripts/intention/cosilab_daic.sh \
  --backend qwen7b \
  --mode participant_image \
  --input-json /path/to/annotation_clips.json \
  --output /path/to/model_responses/run.json
```

The weights are not an option here: set `backends.<name>.model_id` in `model_config.json`.

The job body:

1. Validates the backend, the mode being present, SIF, checkout, the model config and the input manifest. The mode *name* is not validated here: the job script picks no image from it, so the list of valid names lives only in `modes/registry.py` and `cli.py` rejects a bad one.
2. Maps remote media URL prefixes in the manifest to local DAIC filesystem paths.
3. Creates node-local numba and Triton cache directories. Without a writable `NUMBA_CACHE_DIR` the librosa mix fails inside the read-only image and *every* record is skipped as `audio_aggregation_failed`.
4. Builds the `python -m intention_inference` command.
5. Checks inside the container that both `intention_inference` and `models` resolve to the checkout, then runs with `srun apptainer exec --nv`.
6. Writes logs under `/home/zli33/linuxhome/slurm_outputs/intention/`.

Results are filed per backend and then per mode, two levels deeper than the original flat tree. Runs made before the backend split still sit directly in `model_responses/`, and runs made between the two splits sit one level in; nothing was moved, but a notebook globbing `model_responses/*.json` will not see new runs until its pattern is updated.

## Annotation and Survey Analysis

These notebooks and R scripts are intended for post-inference evaluation and annotator analysis. They are not part of the `intention_inference` package and have their own dependencies; they assume local CSV/JSON outputs from the annotation workflow and are not self-contained data bundles.

Notebooks:

- `process_data.ipynb`: parses raw model responses, flattens structured intention annotations, optionally labels annotation dimensions with an LLM helper, and prepares CSV tables for downstream analysis.
- `semantic_similarity.ipynb`: compares model and human intention annotations with sentence-embedding similarity and UMAP projections.
- `LLM_as_judge.ipynb`: runs a paired model-vs-human annotation comparison where an LLM judge predicts which annotation was human-written.

R scripts and supporting files:

- `presurvey-processor.R`: processes the pre-annotation survey using `pre_annotation_survey_codebook.xlsx`, scores survey measures, recodes demographics, and writes a cleaned survey CSV.
- `annotator-analysis.R`: joins cleaned survey measures with annotation outputs and fits mixed-effects models for annotator behavior.
- `pre_annotation_survey_codebook.xlsx`: codebook used to map and score pre-annotation survey columns.

Results:

- `benchmark_sim.md`: recorded model-human answer similarity tables and persona-level similarity comparisons.

Typical sequence:

1. Run inference with `intention-inference` or `job_scripts/intention/cosilab_daic.sh`.
2. Use `process_data.ipynb` to load model response JSON files, parse the assistant text into structured rows, and write tables such as `df_model.csv` and LLM-labeled annotation CSVs.
3. Use `semantic_similarity.ipynb` to compare human annotations and model annotations by embedding intention descriptions or explanations, computing pairwise similarity, and visualizing sources such as annotator A, annotator B, final human annotations, and model outputs.
4. Use `LLM_as_judge.ipynb` for a complementary comparison where an LLM receives paired model/human annotations for the same item and predicts which one was human-written.
5. Use `presurvey-processor.R` to convert raw pre-annotation survey exports into scored participant-level measures.
6. Use `annotator-analysis.R` to merge survey measures with annotation outputs and test relationships between annotator characteristics and annotation content.

These files may contain local path placeholders from the original analysis environment. Update input paths and filenames before running them locally.
