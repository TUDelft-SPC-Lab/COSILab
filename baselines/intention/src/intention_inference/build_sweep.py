#!/usr/bin/env python3
"""Build the prompt-variation sweep: clips x personas x instruction prompts x trials.

The question this sweep answers is how much of a model's intention judgement is
the wording of the question rather than the clip or the observer. So it holds
the clips and the personas small and fixed, and varies the instruction prompt
across the structural variants, repeating each cell enough times to see the
spread:

    5 clips  x  5 personas  x  6 instruction prompts  x  100 trials  =  15,000

The first two dimensions live in the task spec this writes -- five tasks, one per
persona, all five clips in each. The third is NOT in the spec: an instruction
prompt is run-wide, so the sweep is submitted once per prompt and each
submission files its results under that prompt's own directory. The fourth is
``--trials``, which the inference engine applies per clip.

Everything is derived from ``--seed``, so rerunning with the same seed and the
same manifest reproduces the spec byte for byte. The *answers* are not
reproducible: sampling has no seed (see --trials in cli.py), which is the point
-- an unvarying trial would measure nothing.

Candidate clips are validated with the same checks inference uses, by importing
``build_assignment``'s ``collect_valid_clip_indices`` rather than repeating
them. A clip that silently drops out at inference time would cost 600 answers
here, not one.

Sampling must be ON or every trial returns identical text, and the packaged
model_config.json has ``do_sample: false``. Rather than keep a hand-edited near
copy of that file in sync, this script derives one: the base config with
``defaults.do_sample`` set true, written beside the spec and named on the
submission with ``--model-config``. The derived file is the artefact -- it is
diffable, committable, and ``describe_model_config`` copies its values into
every result summary.

DO NOT RUN THIS DIRECTLY unless you pass all four --video-*/--audio-* prefix
flags. A missing pair leaves the manifest's covfee URLs unrewritten, so every
clip is reported unusable. Submit the job script instead, which fills the
prefixes in from job_scripts/lib/intention_paths.sh -- the same file inference
reads::

    bash job_scripts/intention/build_sweep_daic.sh
    bash job_scripts/intention/build_sweep_daic.sh --seed 7

Submitting the sweep it produces is a separate step, deliberately: the spec is
cheap to rebuild and inspect, and 15,000 answers are not.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# src/intention_inference/build_sweep.py -> src/, which is what PYTHONPATH points
# at inside the apptainer bind mount. Must run before any first-party import:
# this file is invoked by absolute path, so src/ is not on sys.path by default.
# Standard library only below, as in build_assignment.py -- no models.registry,
# so this stays runnable on a login node with no ML stack.
_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from intention_inference.build_assignment import (  # noqa: E402
    DEFAULT_PARTICIPANT_IMAGE_ROOT,
    collect_valid_clip_indices,
)
from intention_inference.manifest import (  # noqa: E402
    get_nested_value,
    load_manifest_records,
)
from intention_inference.modes.registry import available_modes, load_mode  # noqa: E402

DEFAULT_MODEL_CONFIG_PATH = Path(__file__).resolve().with_name("model_config.json")

# The design. Fixed here rather than on the command line, for the reason
# build_assignment_daic.sh fixes its own: a sweep whose shape came from whatever
# flags one submission carried is a sweep nobody can reproduce from the repo.
NUM_CLIPS = 5
NUM_PERSONAS = 5
NUM_TRIALS = 100
# The six structural variants, by the numbers in their file names.
INSTRUCTION_PROMPT_NUMBERS = (0, 1, 2, 3, 4, 5)
INSTRUCTION_PROMPT_TEMPLATE = "prompt_structural_variant_{number}.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the clip x persona x instruction-prompt x trial sweep."
    )
    parser.add_argument(
        "--mode",
        default="pa",
        choices=available_modes(),
        help=(
            "The mode the sampled clips are validated against, and the only mode the "
            "resulting spec is valid for. pa, because that is what the instruction "
            "prompts this sweep varies are written for."
        ),
    )
    parser.add_argument("--input-json", type=Path, required=True, help="Clip manifest.")
    parser.add_argument(
        "--personas-json",
        type=Path,
        required=True,
        help="personas_V2_prolific_ids.json: the pool the 5 personas are drawn from.",
    )
    parser.add_argument(
        "--instruction-prompt-dir",
        type=Path,
        required=True,
        help=f"Directory holding {INSTRUCTION_PROMPT_TEMPLATE.format(number='<N>')}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the sweep spec. The derived model config lands beside it.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Chooses the clips and the personas. Recorded in the spec's __meta__.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=NUM_TRIALS,
        help=f"Answers per clip, recorded for the submission step. Default: {NUM_TRIALS}",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=DEFAULT_MODEL_CONFIG_PATH,
        help="The config the sampling one is derived from. Default: the packaged one.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature for the derived config. Default: the base file's.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Nucleus sampling cutoff for the derived config. Default: the base file's.",
    )
    parser.add_argument("--id-key", default="id")
    parser.add_argument("--participant-image-root", type=Path,
                        default=DEFAULT_PARTICIPANT_IMAGE_ROOT)
    parser.add_argument("--media-root", type=Path, default=None)
    parser.add_argument("--media-path-prefix", default=None)
    parser.add_argument("--local-path-prefix", type=Path, default=None)
    parser.add_argument("--video-media-path-prefix", default=None)
    parser.add_argument("--video-local-path-prefix", type=Path, default=None)
    parser.add_argument("--audio-media-path-prefix", default=None)
    parser.add_argument("--audio-local-path-prefix", type=Path, default=None)
    parser.add_argument("--exclude-video-substring", action="append", default=[])
    parser.add_argument("--exclude-audio-substring", action="append", default=[])
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help=(
            "Sample without checking that the clips' media resolves. For building a "
            "spec off-cluster; a sweep submitted from one is a sweep that can lose a "
            "fifth of itself to one missing file."
        ),
    )
    return parser.parse_args()


def load_personas(personas_json_path: Path) -> list[dict[str, str]]:
    """The persona pool, as ``{"id", "persona_prompt"}`` in file order.

    Reads the bare-list form the annotation pipeline emits -- the same shape
    runs/spec.py calls the persona form -- and checks only what this script
    needs: an id that can be a file name and some persona text. The full
    validation is spec.py's, and it runs again when the sweep is submitted.
    """
    with personas_json_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or not payload:
        raise ValueError(
            f"{personas_json_path} must be a non-empty JSON array of "
            "{ID, persona_prompt, videos} entries."
        )

    personas: list[dict[str, str]] = []
    for position, entry in enumerate(payload):
        if not isinstance(entry, dict):
            raise ValueError(f"{personas_json_path}: entry [{position}] is not an object.")
        raw_id = entry.get("ID")
        persona_prompt = entry.get("persona_prompt")
        if isinstance(raw_id, bool) or not isinstance(raw_id, (str, int)):
            raise ValueError(f"{personas_json_path}: entry [{position}] has no usable 'ID'.")
        if not isinstance(persona_prompt, str) or not persona_prompt.strip():
            raise ValueError(
                f"{personas_json_path}: entry [{position}] has an empty 'persona_prompt'."
            )
        personas.append({"id": str(raw_id).strip(), "persona_prompt": persona_prompt.strip()})

    seen = {persona["id"] for persona in personas}
    if len(seen) != len(personas):
        raise ValueError(f"{personas_json_path}: duplicate 'ID' values; ids name output files.")
    return personas


def resolve_instruction_prompts(directory: Path) -> list[str]:
    """The six variant files, checked to exist before anything is written.

    Named in the spec's ``__meta__`` rather than in the tasks: an instruction
    prompt is run-wide, so these are what the submission step iterates over.
    A missing one is refused here because the alternative is discovering it on
    the sixth submission, after five have already run.
    """
    paths: list[str] = []
    missing: list[Path] = []
    for number in INSTRUCTION_PROMPT_NUMBERS:
        path = directory / INSTRUCTION_PROMPT_TEMPLATE.format(number=number)
        if not path.is_file():
            missing.append(path)
        paths.append(str(path))
    if missing:
        raise FileNotFoundError(
            "Instruction prompt(s) not found:\n"
            + "\n".join(f"  {path}" for path in missing)
        )
    return paths


def sample_clip_ids(
    *,
    all_records: list[dict[str, Any]],
    valid_indices: Sequence[int],
    rng: random.Random,
    id_key: str,
    num_clips: int,
) -> list[str]:
    """Draw the clips, and return their record ids rather than their positions.

    Ids, because a spec addresses clips by id: a position would silently mean a
    different clip if the manifest were ever reordered or appended to, and this
    spec is meant to be rerun weeks later.
    """
    if len(valid_indices) < num_clips:
        raise ValueError(
            f"Need {num_clips} usable clip(s), found {len(valid_indices)}. "
            "Check the --video-*/--audio-* prefixes resolve; a mismatched pair makes "
            "every clip look missing."
        )
    chosen = rng.sample(list(valid_indices), num_clips)
    clip_ids: list[str] = []
    for record_index in sorted(chosen):
        record_id = get_nested_value(all_records[record_index], id_key)
        if record_id is None:
            raise ValueError(
                f"Manifest record {record_index} has no {id_key!r}; a spec cannot "
                "address it."
            )
        clip_ids.append(str(record_id))
    if len(set(clip_ids)) != len(clip_ids):
        raise ValueError(
            f"Sampled clips do not have distinct {id_key!r} values: {clip_ids}. "
            "A spec cannot address a duplicated id unambiguously."
        )
    return clip_ids


def build_sampling_model_config(
    *,
    base_path: Path,
    temperature: float | None,
    top_p: float | None,
) -> dict[str, Any]:
    """The base model config with sampling on.

    Derived rather than hand-maintained: a near-copy of model_config.json would
    drift from it the first time a backend's weights path or attention setting
    changed, and that drift is invisible until two runs disagree for a reason
    nobody wrote down. Only 'defaults' is touched; every backend section, and
    every comment in the file, is carried through as-is.

    A backend whose own 'generation' block sets do_sample would still win, since
    that is how model_config.json layers -- none currently does, and a comment in
    the generated file says so.
    """
    with base_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Model config must be a JSON object: {base_path}")

    defaults = dict(payload.get("defaults") or {})
    defaults["do_sample"] = True
    if temperature is not None:
        defaults["temperature"] = temperature
    if top_p is not None:
        defaults["top_p"] = top_p

    overridden = sorted(
        name
        for name, section in (payload.get("backends") or {}).items()
        if isinstance(section, dict) and "do_sample" in (section.get("generation") or {})
    )
    derived = dict(payload)
    derived["defaults"] = defaults
    derived["_comment"] = [
        "GENERATED by intention_inference/build_sweep.py. Edit that script, or the",
        f"base config it derives from, rather than this file: {base_path}",
        "",
        "It is the base config with sampling turned on, which the sweep needs -- with",
        "do_sample false every one of its trials returns identical text, so the sweep",
        "would measure nothing. Nothing else is changed; every backend section below",
        "is the base file's verbatim.",
        "",
        (
            "Backends overriding do_sample in their own 'generation' block (which would"
            f" win over 'defaults'): {', '.join(overridden) if overridden else 'none'}."
        ),
    ]
    return derived


def main() -> None:
    args = parse_args()

    # Everything cheap that can fail, before the manifest is read: six prompt
    # paths and a persona pool cost a stat each, and getting them wrong after
    # validating several thousand clips wastes the expensive half.
    instruction_prompts = resolve_instruction_prompts(
        args.instruction_prompt_dir.expanduser().resolve()
    )
    personas = load_personas(args.personas_json.expanduser().resolve())
    if len(personas) < NUM_PERSONAS:
        raise ValueError(
            f"Need {NUM_PERSONAS} personas, {args.personas_json} has {len(personas)}."
        )
    if args.trials < 1:
        raise ValueError(f"--trials must be at least 1: {args.trials}")

    mode = load_mode(args.mode)
    input_json_path = args.input_json.expanduser().resolve()
    all_records = load_manifest_records(input_json_path)
    print(f"[INFO] Loaded {len(all_records)} record(s) from {input_json_path}")
    print(f"[INFO] Mode: {mode.name}")
    print(f"[INFO] Seed: {args.seed}")

    if args.skip_validation:
        print("[WARN] --skip-validation: clips are sampled without checking their media.")
        valid_indices: list[int] = list(range(len(all_records)))
    else:
        valid_indices, reasons = collect_valid_clip_indices(
            all_records=all_records,
            pool_start=0,
            pool_end=len(all_records) - 1,
            mode=mode,
            args=args,
            manifest_dir=input_json_path.parent,
        )
        print(f"[INFO] Usable clips: {len(valid_indices)}/{len(all_records)}")
        for reason, count in sorted(reasons.items()):
            print(f"[INFO]   rejected {reason}: {count}")

    # One generator for both draws, so the seed alone reproduces the whole spec.
    # Clips first, then personas: fixing the order means adding a dimension later
    # cannot silently reshuffle the one before it.
    rng = random.Random(args.seed)
    clip_ids = sample_clip_ids(
        all_records=all_records,
        valid_indices=valid_indices,
        rng=rng,
        id_key=args.id_key,
        num_clips=NUM_CLIPS,
    )
    chosen_personas = rng.sample(personas, NUM_PERSONAS)

    # The full spec form, not the bare persona list: only this one carries a
    # __meta__, and the sweep's provenance -- seed, ids, trials, the six prompts
    # -- is worth more than the few lines the shape costs.
    tasks = [
        {
            "task_id": f"persona_{persona['id']}",
            "persona_text": persona["persona_prompt"],
            "clips": list(clip_ids),
        }
        for persona in chosen_personas
    ]

    output_path = args.output.expanduser().resolve()
    model_config_path = output_path.with_name(f"{output_path.stem}_model_config.json")
    total_answers = NUM_CLIPS * NUM_PERSONAS * len(instruction_prompts) * args.trials

    spec = {
        "__meta__": {
            "name": output_path.stem,
            "built_by": "intention_inference/build_sweep.py",
            "seed": args.seed,
            "mode": mode.name,
            "input_json": str(input_json_path),
            "personas_json": str(args.personas_json.expanduser().resolve()),
            "clip_ids": clip_ids,
            "persona_ids": [persona["id"] for persona in chosen_personas],
            # The dimensions this file does NOT contain, recorded so the
            # submission step reads them from here rather than being told twice.
            "instruction_prompts": instruction_prompts,
            "trials": args.trials,
            "model_config": str(model_config_path),
            "total_answers": total_answers,
        },
        "tasks": tasks,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    model_config_path.write_text(
        json.dumps(
            build_sampling_model_config(
                base_path=args.model_config.expanduser().resolve(),
                temperature=args.temperature,
                top_p=args.top_p,
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"[INFO] Clips    ({NUM_CLIPS}): {', '.join(clip_ids)}")
    print(f"[INFO] Personas ({NUM_PERSONAS}): "
          f"{', '.join(persona['id'] for persona in chosen_personas)}")
    print(f"[INFO] Instruction prompts ({len(instruction_prompts)}):")
    for path in instruction_prompts:
        print(f"[INFO]   {path}")
    print(f"[INFO] Trials per clip: {args.trials}")
    print(
        f"[INFO] Total answers: {NUM_CLIPS} x {NUM_PERSONAS} x "
        f"{len(instruction_prompts)} x {args.trials} = {total_answers}"
    )
    print(f"[INFO] Sweep spec:   {output_path}")
    print(f"[INFO] Model config: {model_config_path}")
    print("[INFO] Submitting it is a separate step: one persona.sh run per")
    print("[INFO] instruction prompt, each with --trials and --model-config.")


if __name__ == "__main__":
    main()
