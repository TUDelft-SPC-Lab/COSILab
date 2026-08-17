"""Reading model and decoding parameters from a config file.

The parameters that decide what a run *is* -- how many video frames the model
sees, how many tokens it may generate, whether it samples -- live in a JSON file
rather than in command-line flags, so a run's settings can be read, diffed and
committed instead of being reconstructed from an sbatch line. The file belongs to
the task, not to this module: each task keeps its own beside its code
(``intention_inference/model_config.json`` for the intention task), because a
frame budget is a property of what is being asked, and two tasks should not have
to agree on one number.

Every such file has one ``defaults`` block and one section per backend::

    defaults          decoding parameters shared by every backend
    backends.<name>.model_id     weights path/id, or null to use the backend's own
    backends.<name>.load         keyword arguments for the backend constructor
    backends.<name>.generation   decoding overrides on top of defaults
    backends.<name>.extra        backend-only knobs, e.g. gemma's enable_thinking

This module lives under ``models/`` because it produces a
:class:`models.base.GenerationConfig` and knows nothing about any particular
task -- the caller supplies the path. Standard library only: importing it must
never pull in torch, so that entrypoints which never load a model stay light.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from models.base import GenerationConfig

__all__ = [
    "EXTRA_CAPABILITY_REQUIREMENTS",
    "ModelRunConfig",
    "describe_model_config",
    "load_model_config",
    "validate_capabilities",
]


# Decoding parameters accepted in "defaults" and in a backend's "generation".
# Most are GenerationConfig fields. The three video_* / *_video_frames keys are
# not: they are the frame-sampling policy handed to prepare_video_part. They
# live here because to whoever edits the file they are the same kind of thing --
# how the model reads the clip -- and because the policy is a property of the
# run rather than of any one backend. All three backends resolve a frame count
# from them the same way, so a difference between two models is the model.
GENERATION_FIELD_TYPES: dict[str, str] = {
    "max_new_tokens": "positive int",
    "do_sample": "bool",
    "temperature": "positive number",
    "top_p": "positive number",
    "top_k": "positive int",
    "video_fps": "positive number",
    "min_video_frames": "positive int",
    "max_video_frames": "positive int",
}

# An "extra" knob that only some backends implement, and the capability a
# backend must advertise before it may be set. Mirrors what --enable-thinking
# used to check by hand.
EXTRA_CAPABILITY_REQUIREMENTS: dict[str, str] = {
    "enable_thinking": "thinking",
}


@dataclass(frozen=True)
class ModelRunConfig:
    """Everything one backend needs for one run, resolved from the file."""

    backend: str
    model_id: str | None
    load_kwargs: dict[str, Any] = field(default_factory=dict)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    # The frame-sampling policy. Defaults match Qwen's own upstream constants
    # (FPS = 2.0, FPS_MIN_FRAMES = 4) so an unconfigured run reproduces what the
    # backend would have chosen for itself.
    video_fps: float = 2.0
    min_video_frames: int = 4
    max_video_frames: int = 32
    source_path: str = ""

    def extra_is_truthy(self, key: str) -> bool:
        return bool(self.generation.extra.get(key))


def _require_mapping(value: Any, label: str, source_path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{source_path}: {label} must be a JSON object, got {type(value).__name__}.")
    # Keys starting with "_" are comments. JSON has none of its own, and a
    # parameter file nobody can annotate is a parameter file nobody trusts.
    return {key: item for key, item in value.items() if not key.startswith("_")}


def _check_positive_int(value: Any, label: str, source_path: Path) -> int:
    # bool is a subclass of int, so `isinstance(True, int)` is True: reject it
    # explicitly rather than silently running with max_new_tokens=1.
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{source_path}: {label} must be a positive integer, got {value!r}.")
    return value


def _check_positive_number(value: Any, label: str, source_path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{source_path}: {label} must be a positive number, got {value!r}.")
    return float(value)


def _check_bool(value: Any, label: str, source_path: Path) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{source_path}: {label} must be true or false, got {value!r}.")
    return value


def _merge_generation_values(
    *,
    base: dict[str, Any],
    override: dict[str, Any],
    label: str,
    source_path: Path,
) -> dict[str, Any]:
    """Validate one block of decoding parameters and layer it over the base.

    An unknown key is an error rather than a shrug: a mistyped "temprature" that
    is silently ignored is the worst thing that can happen to a config file,
    because the run looks configured and is not.
    """
    for key, value in override.items():
        if key not in GENERATION_FIELD_TYPES:
            raise ValueError(
                f"{source_path}: unknown parameter {key!r} in {label}. "
                f"Valid parameters: {', '.join(sorted(GENERATION_FIELD_TYPES))}."
            )
        where = f"{label}.{key}"
        if GENERATION_FIELD_TYPES[key] == "positive int":
            value = _check_positive_int(value, where, source_path)
        elif GENERATION_FIELD_TYPES[key] == "positive number":
            value = _check_positive_number(value, where, source_path)
        else:
            value = _check_bool(value, where, source_path)
        base[key] = value
    return base


def load_model_config(model_config_path: Path, backend: str) -> ModelRunConfig:
    """Resolve the parameters for one backend out of the config file."""
    source_path = model_config_path
    with source_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    payload = _require_mapping(payload, "the config", source_path)

    values: dict[str, Any] = _merge_generation_values(
        base={},
        override=_require_mapping(payload.get("defaults", {}), "'defaults'", source_path),
        label="defaults",
        source_path=source_path,
    )

    backends = _require_mapping(payload.get("backends", {}), "'backends'", source_path)
    if backend not in backends:
        # No falling back to defaults: running a backend the file does not
        # mention means the file and the command disagree about what is running.
        raise ValueError(
            f"{source_path}: no section for backend {backend!r}. "
            f"Sections present: {', '.join(sorted(backends)) or '<none>'}."
        )

    section = _require_mapping(backends[backend], f"'backends.{backend}'", source_path)
    values = _merge_generation_values(
        base=values,
        override=_require_mapping(
            section.get("generation", {}), f"'backends.{backend}.generation'", source_path
        ),
        label=f"backends.{backend}.generation",
        source_path=source_path,
    )

    model_id = section.get("model_id")
    if model_id is not None and (not isinstance(model_id, str) or not model_id.strip()):
        raise ValueError(
            f"{source_path}: backends.{backend}.model_id must be a non-empty string or null, "
            f"got {model_id!r}."
        )

    # Not validated here on purpose: these are the backend's own constructor
    # keywords, so the constructor is the thing that knows which are valid, and
    # an unknown one is already a loud TypeError at load time.
    load_kwargs = dict(
        _require_mapping(section.get("load", {}), f"'backends.{backend}.load'", source_path)
    )
    extra = dict(
        _require_mapping(section.get("extra", {}), f"'backends.{backend}.extra'", source_path)
    )

    video_fps = _check_positive_number(
        values.pop("video_fps", 2.0), "video_fps", source_path
    )
    min_video_frames = _check_positive_int(
        values.pop("min_video_frames", 4), "min_video_frames", source_path
    )
    max_video_frames = _check_positive_int(
        values.pop("max_video_frames", 32), "max_video_frames", source_path
    )
    if min_video_frames > max_video_frames:
        raise ValueError(
            f"{source_path}: min_video_frames ({min_video_frames}) exceeds max_video_frames "
            f"({max_video_frames}); no frame count can satisfy both."
        )
    generation = GenerationConfig(**values, extra=extra)

    return ModelRunConfig(
        backend=backend,
        model_id=model_id,
        load_kwargs=load_kwargs,
        generation=generation,
        video_fps=video_fps,
        min_video_frames=min_video_frames,
        max_video_frames=max_video_frames,
        source_path=str(source_path),
    )


def validate_capabilities(config: ModelRunConfig, capabilities: frozenset[str]) -> None:
    """Refuse knobs the chosen backend cannot honour.

    Checked before any weights load. A generation parameter that is silently
    dropped leaves a result file indistinguishable from one that never asked for
    it, which is the kind of thing nobody can reconstruct months later.
    """
    for key, required in EXTRA_CAPABILITY_REQUIREMENTS.items():
        if config.extra_is_truthy(key) and required not in capabilities:
            raise ValueError(
                f"{config.source_path}: backends.{config.backend}.extra.{key} is set, but the "
                f"{config.backend!r} backend does not support {required!r}."
            )


def describe_model_config(config: ModelRunConfig) -> dict[str, Any]:
    """The resolved parameters, for the run summary and the job log."""
    return {
        "model_config": config.source_path,
        "backend_options": dict(config.load_kwargs),
        "generation": {
            "max_new_tokens": config.generation.max_new_tokens,
            "do_sample": config.generation.do_sample,
            "temperature": config.generation.temperature,
            "top_p": config.generation.top_p,
            "top_k": config.generation.top_k,
            "extra": dict(config.generation.extra),
        },
        "video_fps": config.video_fps,
        "min_video_frames": config.min_video_frames,
        "max_video_frames": config.max_video_frames,
    }
