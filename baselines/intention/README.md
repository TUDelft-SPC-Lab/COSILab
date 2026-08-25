# Intention Baseline: Multimodal Inference and Annotation Analysis

This folder contains the INGroup intention-recognition baseline that prompts a multimodal model on 30-second clips, plus analysis code for survey responses and model-human annotation comparison.

Two things vary independently, and neither is a property of the code. Which model runs is a `--backend` choice -- `gemma`, `qwen3b`, `qwen7b` and `qwen3omni30b` today -- and the task itself imports no torch and knows nothing about either. What is asked of it is a `--mode` choice, and the three modes are one experimental axis: how much of the group's audio is stacked into a single soundtrack, and therefore how much of it can be attributed to a person. Any mode runs on any backend.

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
    pa.py                           partially stacked audio
    fa.py                           flattened audio
    sa.py                           stacked audio (PLACEHOLDER)
    prompt_pa.json                  PA's prompt (unchanged since before modes existed)
    prompt_fa.json                  FA's prompt, labels included
    prompt_sa.json                  stub, unwritten
  __main__.py                       entry point for `python -m intention_inference`
src/models/                         the model backends the task runs on
  base.py                           the interface every backend implements
  registry.py                       backend name -> class, imported lazily
  config.py                         model_config.json -> a resolved run config
  media_io.py                       video/audio/image decoding, frame policy
  gemma/                            Gemma 4 via transformers
  qwen/                             Qwen2.5-Omni via transformers
    shared.py                       plumbing both Qwen generations use
  qwen3omni/                        Qwen3-Omni-30B-A3B via transformers
job_scripts/
  intention/                        two inference entry points plus one design utility
    cosilab_daic.sh                 one Slurm job WITHOUT personas
    persona.sh                      multi-job launcher WITH personas
    build_assignment_daic.sh        builds an optional balanced assignment
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

Each retained record is converted into one chat turn. `records.py` resolves what every mode needs from the filesystem:

- the video clip from the record's `video` field
- one audio track per speaker, from `audios[speaker_id - 1]` -- the list is positional, so speaker 1 is `audios[0]`

What differs between modes is what happens to those tracks. `records.resolve_audio_bundle` keeps the participant's own and sums everyone else's into one soundtrack (PA); `records.resolve_per_speaker_audio` hands them over untouched (FA); a whole-group mix is what SA will need. Both sit on the same `records.resolve_speaker_tracks`, so which track belongs to whom is decided in exactly one place.

The mode also decides which reference images go with it, the rendered prompt and the order of the parts. Images come from `participant_<id>.png` under `--participant-image-root`.

The group of analysis (GOA) is derived, not read from a field: the manifest has no group of its own, so it is `{participant} ∪ conversation_floor`. The union settles it whether or not `conversation_floor` already lists the participant.

Records are skipped before inference, with the reason recorded, when the video or the mode's reference media is missing or not on disk (`participant_image_not_found` for PA's single crop, `goa_participant_image_not_found` when FA is missing a photograph for *someone* in the group), when the audio list or speaker selection is malformed, when participant or conversation-floor audio cannot be resolved, when audio aggregation fails, or when a resolved path matches an `--exclude-video-substring` / `--exclude-audio-substring` filter.

Each mode ships its own prompt config, and `--prompt-config` overrides it. A prompt config provides:

- `system_prompt`: the system instruction, folded into the user text
- `user_prompt_template`: the actual intention-recognition prompt

If `user_prompt_template` (or `prompt`) is absent, the loader assembles the template from structured `intro`, `questions`, and `examples` fields. Templates are rendered with the record's flattened fields, so `{field}` and `{nested.field}` placeholders are filled from the record, `{record_json}` expands to the whole record, and unknown placeholders are left untouched.

Every mode's prompt asks the model to identify the indicated participant's intentions, including timestamps, confidence, reasoning, intensity, and counterfactual explanations. If no clear intention is visible, the model is instructed to return the no-intention format.

FA's prompt config carries more than the two keys above: its prompt is spread across a dozen text parts interleaved with the media, and each of those fragments is a key in the file. They live there rather than in Python so the exact wording of a run stays diffable. `load_prompt_config` passes through every key it does not itself normalise, and `BaseTaskMode.configure` hands the resolved config to the mode before any record is prepared. The fragments are listed under the three modes below.

The model is reached only through `models/`: `--backend` names it, `models/registry.py` resolves the name to a class, and `model_config.json` says which weights it loads with what decoding parameters. Task code never imports torch.

The question is reached the same way through `modes/`: `--mode` names it, `modes/registry.py` resolves the name to a class, and the mode owns its reference media, its prompt config and its turn layout. Add a mode by implementing `modes.base.BaseTaskMode` and adding one line to `MODE_MODULES`. `BaseTaskMode.prepare_record` is the default composition -- clip, reference media, audio, prompt -- and a mode that does not fit it overrides it and composes `records.resolve_video_media`, `records.resolve_speaker_tracks`, `records.resolve_audio_bundle` or `records.resolve_per_speaker_audio` itself -- which is exactly what `fa` does.

`sa` is registered but unimplemented: selecting it raises at startup, before the model config or the manifest is read, and says what remains to be written.

Frame sampling is one policy for every backend, set in `model_config.json` under `defaults`:

```text
video_fps         target rate (2.0)
min_video_frames  floor (4)
max_video_frames  ceiling (32)
```

`models/media_io.py` `frames_at_fps` resolves them once -- `n = duration * fps`, clamped, rounded down to even -- and every backend is handed that exact count, so a difference between two models is the model rather than how much video each one saw. A 3-second clip gives 6 frames; anything under 2 seconds lands on the floor, anything over 16 seconds on the ceiling.

Images, audio, and video frames are decoded ahead of the processor call with Pillow, librosa, and PyAV. If a Gemma processor ships no chat template, a built-in Gemma 4 fallback template is used.

### The three modes

The axis is how much of the group's audio is stacked, and therefore how much of it can be attributed to a person:

| mode | audio parts | grounding | status |
| --- | --- | --- | --- |
| `sa` | 1 -- everyone stacked, participant included | one crop of the participant | **placeholder, raises** |
| `pa` | 2 -- the participant, and everyone else stacked | one crop of the participant | runs; unchanged since before modes existed |
| `fa` | N -- one per person, each labelled with their id | a photograph per person, each labelled | runs |

The turn layout belongs to the mode, because part order is the order the model meets the placeholders in. `pa` produces:

```text
image: participant reference image
audio: participant audio
audio: mixed conversation-floor audio
video: sampled video frames
text:  system prompt + rendered user prompt
```

`fa` reads as one continuous instruction with media embedded in it. For a group of participants 1, 3 and 4 with 3 as the person of interest:

```text
text:  system prompt + "Watch the 30-second video clip... The following is a
        video that shows the conversation from top view:"
video: sampled video frames
text:  "This is the participant that you would observe: participant 3, and
        this is their audio in the conversation:"
audio: participant 3's track
text:  "This is an image of them:"
image: participant_3.png
text:  "These are all other participants that are within the same
        conversation as participant 3:"
text:  "[participant 1, audio:"    audio: 1's track
text:  ", image:"                  image: participant_1.png
text:  "],"
text:  "[participant 4, audio:"    audio: 4's track
text:  ", image:"                  image: participant_4.png
text:  "],"
text:  "You can use the gallery images to locate the participants in the
        video. ... Identify the intentions of participant 3. ..."
```

The person of interest is named up front, before anyone else is shown, and the others follow in ascending id order. The system prompt is folded into the opening text rather than the closing one, because this prompt opens by saying what the model is about to be shown.

**Ids are bound by interleaved text, not by position.** A text fragment naming the speaker precedes each media part, so the association holds regardless of how a backend lays out its placeholders. Both backends render text parts inline between media tokens, which is what makes this work.

`pa` does not do that, and is kept that way deliberately. The `<image>`, `<audio1>` and `<audio2>` markers in its prompt are literal characters, not placeholders: the real media tokens are emitted at the parts' positions, all of them ahead of the text block, so those markers resolve to nothing and the binding rests on part order alone. Fixing it would make `pa` a different condition and break comparability with results already collected under it. `fa` does not repeat the mistake, and a new mode should not either.

Every fragment above is a key in `prompt_fa.json` -- `video_intro`, `poa_audio_intro`, `poa_image_intro`, `others_intro`, `other_open`, `other_between`, `other_close`, plus `poa_intro_video_only` for `--no-audio` -- so the wording of a run stays diffable. Each takes `{participant}` (the person of interest) and `{speaker_id}` (whoever that fragment introduces); an unknown placeholder is left as itself rather than raising. `user_prompt_template` is the closing block and additionally gets `{goa_ids}`, `{other_ids}` and `{goa_size}`.

`modes/fa.py` `_segments` is the single place the order is decided: both the turn and the prompt recorded in the result file are rendered from it, so they cannot drift apart. The recorded `user` field is the whole prose with each media part shown as a marker (`<video>`, `<audio 3>`, `<gallery_image 3>`), since recording only the last of a dozen text parts would record almost nothing.

Note that `{conversation_floor}` does **not** render in any mode: it is a list, and the template flattener keeps only scalars -- which is why `fa` supplies the group as text itself.

The aggregated conversation-floor mixes are written next to the output file, under:

```text
<output_dir>/_audio_mixes/<output_stem>/<index>_<record_id>_conversation_floor.wav
```

Floor tracks whose length differs from the participant track by more than the tolerance are discarded with a warning; if nothing remains, silence is used. The mix is peak-normalized when it clips.

The output JSON contains:

- `__summary__`: run metadata, selected/retained/processed counts, and skip reasons
- `__skipped__`: records skipped before inference and why
- `results`: one item per processed record, including source/rewritten/resolved media paths, speaker IDs, audio warnings, the rendered prompt, and the model response in `assistant`

`__summary__` records which backend ran and in which mode, the resolved weights path, the decoding parameters and the frame policy, so a result file answers "what produced this" without anyone having to find the command that made it. Beyond the shared keys, each result row carries whatever the mode's `result_fields` contributes -- for `pa`, the reference image that was shown; for `fa`, the id-to-track and id-to-photograph maps, so nobody has to re-derive from part order which part was whose.

Generation errors do not abort the run: the failing record's `assistant` field is set to `[ERROR] ...` and counted in `__summary__.error_count`.

## Run Locally

```bash
uv run intention-inference \
  --backend gemma \
  --mode pa \
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
- `--participant-image-root`: folder containing `participant_<n>.png`. `pa` reads one file from it; `fa` reads one per person in the group.

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

There are no `--model` / `--max-new-tokens` / `--temperature` / `--top-p` / `--top-k` / `--do-sample` / `--enable-thinking` / `--max-video-frames` flags any more. Change a run by editing that file and resubmitting; the settings are then something that can be read, diffed and committed, and they are recorded into every result file. Setting a knob a backend does not support (`enable_thinking` on any Qwen backend) is refused at startup rather than dropped.

An absolute `model_id` that is not a directory fails the startup check before the manifest is read. A relative value is treated as a Hugging Face Hub id and therefore skips that local-directory check.

## Submit on DAIC

There are two inference entry points, each with one behavior. The assignment
builder is a separate setup utility.

```text
job_scripts/intention/
  cosilab_daic.sh           one Slurm job WITHOUT personas
  persona.sh                multi-job launcher WITH personas
  build_assignment_daic.sh  optional balanced-design builder
```

**`cosilab_daic.sh` — no personas.** Submit one inclusive manifest range as one
job. It does not tile or submit itself:

```bash
sbatch job_scripts/intention/cosilab_daic.sh \
    --backend gemma --mode pa --index-range 0-999
```

The output is `<manifest-stem>_<start>-<end>.json`. For roughly 6,000 plain
clips, submit the handful of 800–1,000-clip ranges explicitly.

**`persona.sh` — with personas.** Run it with `bash` on the login node. It reads
exactly one source and launches all required GPU jobs:

```bash
# balanced assignment: dry run, then submit
bash job_scripts/intention/persona.sh --backend qwen7b --mode pa \
    --assignment-json /path/to/assignment.json
bash job_scripts/intention/persona.sh --backend qwen7b --mode pa \
    --assignment-json /path/to/assignment.json --submit

# externally authored persona specification
bash job_scripts/intention/persona.sh --backend gemma --mode pa \
    --task-spec /path/to/personas_prolific_ids.json --submit
```

`--assignment-json` reads the balanced design (personas are `persona_NNNN.txt`,
clips are manifest positions). `--task-spec` reads the annotation pipeline's
external specification (persona text inline, clips by record id). The source is
explicit; neither is silently chosen as a default.

Jobs are packed by `--answers-per-job`, which defaults to 1,200 persona–clip
answers. A source covering 2,000 clips with six personas per clip contains
12,000 answers, so it becomes 10 jobs. If every persona has 200 clips, each job
contains six complete personas. Complete persona runs are never split because
each persona owns one output JSON; splitting one across concurrent jobs would
cause competing writes. The clip set therefore belongs in the assignment/spec,
while `persona.sh` only controls job packing.

The launcher is dry-run by default and resumes safely: it skips packs whose
outputs all exist, while the inference worker skips already-finished personas in
a partially complete pack.

There is no every-persona-every-clip option, here or in `cli.py` — that cross product is what the balanced design replaces.

**`build_assignment_daic.sh` — the design.** CPU-only and short, so it runs as an sbatch job or straight on a login node. `--mode` is required: a design is only valid for the mode it was validated against, and `persona.sh` refuses one whose mode disagrees.

```bash
sbatch job_scripts/intention/build_assignment_daic.sh --mode pa
bash   job_scripts/intention/build_assignment_daic.sh --mode pa --explain 2
```

`--explain N` prints where the first N candidate clips' media was looked for, then exits without building. Reach for it when validation rejects everything: the usual cause is a media/local prefix pair that stopped matching the manifest, which leaves the raw URL in place, so every clip looks missing when only a flag is wrong.

Run any script with `--help` for its full option list.

Default DAIC paths (`lib/intention_paths.sh` and `lib/model_backends.sh`):

```text
PROJECT_ROOT   /home/zli33/linuxhome/projects/COSILab
DATA_ROOT      /tudelft.net/staff-umbrella/neon/cosilab_project/data_clean/processed/benchmark_tasks/benchmark_1
input_json     /tudelft.net/staff-umbrella/neon/cosilab_project/B1_pipeline/annotation_clips.json
output_dir     /tudelft.net/staff-umbrella/neon/cosilab_project/B1_pipeline/model_responses/<backend>/<mode>
gemma        SIF  /tudelft.net/staff-umbrella/neon/apptainer/gemma.sif
qwen3b       SIF  /tudelft.net/staff-umbrella/neon/apptainer/qwen2.5-omni-inference.sif
qwen7b       SIF  /tudelft.net/staff-umbrella/neon/apptainer/qwen2.5-omni-inference.sif
qwen3omni30b SIF  /tudelft.net/staff-umbrella/neon/apptainer/qwen3-omni-inference.sif
```

The two Qwen2.5 sizes share one image and backend class; only their configured weights differ. Qwen3-Omni gets a separate image because `qwen3omni30b` needs a newer `transformers` than the 4.52 in the Qwen2.5 image.

`PROJECT_ROOT` is bound to `/workspace` in the container and `PYTHONPATH` is set to `/workspace/baselines/intention/src`, so the job always runs the checkout's code rather than a copy baked into the image.

Submit plain, single-job ranges:

```bash
sbatch baselines/intention/job_scripts/intention/cosilab_daic.sh --backend qwen7b --mode fa --index-range 0-99
sbatch baselines/intention/job_scripts/intention/cosilab_daic.sh --backend qwen3b --mode pa --index-range 0-999
sbatch baselines/intention/job_scripts/intention/cosilab_daic.sh --backend qwen3omni30b --mode pa --index-range 0-999
sbatch baselines/intention/job_scripts/intention/cosilab_daic.sh --backend gemma --mode pa --index-range 1000-1999 --no-audio
```

The worker's shared SLURM header requests two RTX PRO 6000 GPUs, so direct jobs expose two cards to every Qwen backend. `persona.sh` applies backend-specific requests automatically: one GPU for Gemma and two for `qwen3b`, `qwen7b` and `qwen3omni30b`. For a direct Gemma job, release the unused card with `sbatch --gres=gpu:nvidia_rtx_pro_6000:1 ...`.

All Qwen configs use `device_map: "auto"`. Accelerate can dispatch model modules across the visible GPUs, but it is memory-driven and may place the whole 3B or 7B model on GPU 0 when that model fits there; requesting two GPUs does not itself guarantee a balanced split or double inference throughput. Each Qwen load prints `Accelerate placement` with the visible CUDA count and the resolved `hf_device_map`, so check that line in the Slurm log when both-card placement matters.

Override input and output:

```bash
sbatch baselines/intention/job_scripts/intention/cosilab_daic.sh \
  --backend qwen7b \
  --mode fa \
  --input-json /path/to/annotation_clips.json \
  --output /path/to/model_responses/run.json
```

The weights are not an option here: set `backends.<name>.model_id` in `model_config.json`.

Each worker job:

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
