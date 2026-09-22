"""The instruction prompt: the question a run asks, as a file of its own.

The prompt structure-and-phrasing experiment varies how the intention question
is worded while holding the personas, the clips and the model fixed. So the
wording is not something a mode owns any more: every run reads it from a plain
text file holding one whole ``user_prompt_template``, and
``prompt_original_revised.txt`` is the baseline the variants are measured
against.

The text replaces ``user_prompt_template`` in *every* prompt config the run
resolves: the mode's own, one passed with ``--prompt-config``, and any a task
spec carries per task. That last one is what "every" is for -- a spec whose
tasks bring their own wording would otherwise quietly opt out of the wording
being measured. ``--no-instruction-prompt`` opts out on purpose and falls back
to whatever the prompt config asks.

Only ``user_prompt_template`` is replaced. A mode that spreads its prompt over
several fragments keeps those fragments, and under ``fa`` that is most of the
turn -- which is why the two functions below exist rather than one.

    incompatibility_reasons   fatal. The prompt describes a turn this mode does
                              not build, so running it would record an answer to
                              a question that was never really asked.
    mode_warnings             worth knowing, not worth refusing.

Personas are not affected by any of this. A persona is folded in ahead of the
instruction prompt as the system prompt, wherever it came from.

Standard library only: persona.sh runs this as a login-node preflight, where
there is no container and no ML stack.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from intention_inference.prompt import SafeFormatDict

__all__ = [
    "PA_MEDIA_MARKERS",
    "PA_STACKING_WORDS",
    "apply_to_prompt_config",
    "incompatibility_reasons",
    "load_instruction_prompt",
    "mode_warnings",
]


# PA's inert markers: literal text that resolves to nothing, bound to the real
# media only by part order (see modes/pa.py). Every prompt written against PA
# carries them, which is exactly why they need checking against another mode.
PA_MEDIA_MARKERS = ("<image>", "<audio1>", "<audio2>")

# Language that only makes sense when the other participants arrive as one
# summed track. PA does that; FA does not.
PA_STACKING_WORDS = ("<audio2>", "stacking", "aggregated")


def load_instruction_prompt(instruction_prompt_path: Path) -> str:
    """Read one instruction prompt, and refuse one that cannot be rendered.

    The braces are checked here rather than at the first record: the template
    goes through ``str.format_map``, so a stray ``{`` raises once per clip in a
    job that has already loaded the weights. An unknown ``{placeholder}`` is
    fine and is left as itself -- that is what :class:`SafeFormatDict` is for --
    but a literal brace has to be doubled.
    """
    text = instruction_prompt_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Instruction prompt is empty: {instruction_prompt_path}")
    try:
        text.format_map(SafeFormatDict())
    except Exception as exc:
        raise ValueError(
            f"{instruction_prompt_path}: the instruction prompt is rendered with "
            f"str.format_map and its braces do not parse ({type(exc).__name__}: {exc}). "
            "An unknown {placeholder} is left as itself and is fine; a literal brace "
            "has to be doubled as {{ or }}."
        ) from exc
    return text


def apply_to_prompt_config(
    prompt_config: Mapping[str, Any], instruction_text: str
) -> dict[str, Any]:
    """The same config with the instruction prompt as its user prompt.

    A copy rather than a mutation: a task spec's per-task configs are cached by
    path, and the mode's default config is restored around every run that brought
    its own, so overwriting in place would leak one run's wording into the next.
    """
    return {**prompt_config, "user_prompt_template": instruction_text}


def incompatibility_reasons(instruction_text: str, *, mode: str) -> list[str]:
    """Why this mode cannot honestly ask this prompt. Non-empty means refuse.

    Fatal rather than advisory, and fatal for a passed file as much as for the
    default one. The failure it prevents is silent: the run completes, writes a
    full set of answers, and every one of them answers a question about audio
    that was never in the turn. Nothing downstream could tell that apart from a
    real result, so it has to be stopped before the weights load.
    """
    if mode != "fa":
        return []

    reasons: list[str] = []

    present = [marker for marker in PA_MEDIA_MARKERS if marker in instruction_text]
    if present:
        reasons.append(
            f"It carries PA's markers {', '.join(present)}. Under PA those are inert "
            "text bound to the media by part order; under fa every audio and image "
            "part is already introduced by the id it belongs to, so they point at "
            "nothing."
        )

    if any(word in instruction_text for word in PA_STACKING_WORDS):
        reasons.append(
            "It describes a second, aggregated track for the other participants. fa "
            "stacks nothing -- every person in the group gets their own audio part -- "
            "so the prompt would describe a turn this run does not build."
        )

    return reasons


def mode_warnings(instruction_text: str, *, mode: str) -> list[str]:
    """What this prompt will and will not do under this mode.

    Warnings, not errors: each is true even of a prompt written for fa, so
    refusing on them would leave fa with nothing it could run.
    """
    if mode != "fa":
        return []

    warnings = [
        "fa spreads its prompt across many text fragments -- video_intro, the "
        "per-speaker intros, then the closing block -- and an instruction prompt "
        "replaces only the closing block. video_intro still carries prompt_fa.json's "
        "wording, so this run mixes the two.",
    ]

    if "{participant}" not in instruction_text:
        warnings.append(
            "The instruction prompt never renders {participant}. fa names everyone by "
            "id and binds each media part to one; a closing block saying only 'the "
            "indicated participant' drops that binding at the point the question is "
            "asked."
        )

    return warnings


def check(instruction_text: str, *, mode: str, source: Path | str) -> None:
    """Raise on an incompatible prompt, print what is merely worth knowing.

    One function so cli.py and the launcher preflight cannot drift about which
    of the two lists is fatal.
    """
    reasons = incompatibility_reasons(instruction_text, mode=mode)
    if reasons:
        detail = "\n".join(f"        - {reason}" for reason in reasons)
        raise ValueError(
            f"Mode {mode!r} cannot ask this instruction prompt:\n"
            f"        {source}\n"
            f"{detail}\n"
            "        Pass an instruction prompt written for this mode, or "
            "--no-instruction-prompt\n"
            "        to fall back to the prompt config's own template."
        )
    for warning in mode_warnings(instruction_text, mode=mode):
        print(f"[WARN] {warning}", file=sys.stderr)


def main() -> None:
    """Preflight one instruction prompt, for the login-node launcher.

    Prints the resolved path on stdout so the caller can use it, and anything
    wrong with it on stderr, so a sweep that cannot honestly run is answered
    before the queue accepts N jobs rather than in N job logs afterwards.
    """
    parser = argparse.ArgumentParser(description="Check one instruction prompt file.")
    parser.add_argument("--instruction-prompt", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    args = parser.parse_args()

    prompt_path = args.instruction_prompt.expanduser().resolve()
    if not prompt_path.is_file():
        sys.exit(f"[ERROR] Instruction prompt not found: {prompt_path}")
    try:
        instruction_text = load_instruction_prompt(prompt_path)
        check(instruction_text, mode=args.mode, source=prompt_path)
    except ValueError as exc:
        sys.exit(f"[ERROR] {exc}")

    print(prompt_path)


if __name__ == "__main__":
    main()
