"""Plumbing shared by every Qwen-Omni generation.

Qwen2.5-Omni and Qwen3-Omni are different model classes with different config
trees, but everything *around* them is the same: the same chat content dicts,
the same qwen_omni_utils preprocessing, the same transformers-version quirks,
the same diagnostics for which attention backend each tower actually resolved
to. That shared half lives here so a second generation is a new engine module
rather than a second copy of it.

The rule that keeps this importable from either engine: **no concrete model or
processor class is imported here.** A container image built for one generation
does not necessarily carry the other's classes, so importing
``Qwen2_5OmniForConditionalGeneration`` in this module would make the Qwen3
backend fail inside an image that has no reason to ship it. Only torch and
transformers itself.

The two functions that walk the config tree take the paths to walk as an
argument for the same reason: the tree differs between generations (Qwen3 adds
a code2wav tower and nests the talker one level deeper), so each engine owns its
own ``ATTN_CONFIG_PATHS`` and hands it in.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import transformers

from models.base import ChatMessage

__all__ = [
    "ATTN_SUPPORT_FLAGS",
    "FPS_SHAPES",
    "apply_attn_implementation",
    "describe_attn_implementations",
    "describe_attn_support",
    "describe_sdpa_backends",
    "dtype_kwarg",
    "fps_processor_kwargs",
    "load_process_mm_info",
    "resolve_dtype",
    "to_qwen_messages",
    "transformers_version",
]

# The frame count qwen_omni_utils rounds to a multiple of is models.media_io's
# FRAME_FACTOR, and it is not restated here: one constant, in the module that
# resolves the count for every backend.

# qwen_omni_utils is NOT vendored here: no model code lives in this repo. It comes
# from the container image (job_scripts/lib/model_backends.sh names which one),
# and this env var prepends an external checkout when the code is supplied from
# outside instead.
_OMNI_UTILS_PATH_ENV = "QWEN_OMNI_UTILS_PATH"


@lru_cache(maxsize=1)
def load_process_mm_info() -> Any:
    """Import ``process_mm_info`` on first use, not at module import.

    It pulls in av, librosa and audioread. This module is imported by
    ``models.registry`` merely to read a class attribute -- capabilities are
    checked before any weights load -- so that lookup must not require the whole
    media stack to be installed.
    """
    override = os.environ.get(_OMNI_UTILS_PATH_ENV)
    if override:
        # Prepended, unlike the image's copy: an explicitly supplied path is a
        # choice, and losing silently to whatever the image happens to carry is
        # the failure this variable exists to prevent.
        if not Path(override).is_dir():
            raise FileNotFoundError(
                f"{_OMNI_UTILS_PATH_ENV}={override} is not a directory. Point it at the "
                f"directory containing the qwen_omni_utils package, or unset it to use "
                f"the copy installed in the container image."
            )
        if override not in sys.path:
            sys.path.insert(0, override)

    try:
        import qwen_omni_utils
    except ImportError as exc:
        # Two causes that look identical from the traceback, so both are named:
        # the image may not carry the package, or the override may point
        # somewhere that does not contain it.
        raise ImportError(
            f"qwen_omni_utils could not be imported. It is not vendored in this repo: "
            f"either the container image is missing it (check with `apptainer exec <sif> "
            f"python -c 'import qwen_omni_utils'`), or "
            f"{_OMNI_UTILS_PATH_ENV}={override or '<unset>'} does not contain it."
        ) from exc

    print(f"[INFO] qwen_omni_utils: {qwen_omni_utils.__file__}", flush=True)
    return qwen_omni_utils.process_mm_info


def resolve_dtype(dtype: str) -> Any:
    """Map a config string onto a torch dtype.

    "auto" follows the hardware: bfloat16 where there is a GPU, float32 where
    there is not, since bfloat16 on CPU is slow rather than merely smaller.
    """
    if dtype == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    try:
        resolved = getattr(torch, dtype)
    except AttributeError:
        raise ValueError(f"Unknown dtype: {dtype!r}") from None
    if not isinstance(resolved, torch.dtype):
        raise ValueError(f"Not a torch dtype: {dtype!r}")
    return resolved


def to_qwen_messages(
    messages: Sequence[ChatMessage],
    *,
    system_prompt: str,
) -> list[dict]:
    """Render the neutral message form into the content dicts Qwen expects.

    The Qwen boilerplate system turn is model plumbing rather than task content,
    so it is added here and the task's own system/persona prompt stays folded
    into the user text -- which keeps the prompt recorded in the results
    identical to what every other backend records.
    """
    qwen_messages: list[dict] = []
    if system_prompt:
        qwen_messages.append(
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
        )

    for message in messages:
        content: list[dict[str, Any]] = []
        for part in message.content:
            if part.type == "text":
                content.append({"type": "text", "text": part.text or ""})
            elif part.type == "image":
                content.append({"type": "image", "image": str(part.path)})
            elif part.type == "audio":
                content.append({"type": "audio", "audio": str(part.path)})
            elif part.type == "video":
                video_item: dict[str, Any] = {"type": "video", "video": str(part.path)}
                if part.num_frames is not None:
                    # Always the exact count, never a rate: handing smart_nframes
                    # an fps would let its own FPS_MIN_FRAMES / FPS_MAX_FRAMES
                    # decide, and those differ from what Gemma and Ming land on
                    # for the same clip. The rate is applied once, in
                    # models.media_io.frames_at_fps, before we get here.
                    video_item["nframes"] = part.num_frames
                content.append(video_item)
            else:
                raise ValueError(f"Unsupported media part type: {part.type!r}")
        qwen_messages.append({"role": message.role, "content": content})
    return qwen_messages


def transformers_version() -> tuple[int, int]:
    """(major, minor) of the installed transformers, tolerant of suffixes.

    Version strings in the wild include "4.52.4", "5.0.0.dev0" and "4.56.0rc1",
    so the numbers are matched rather than split on dots.
    """
    match = re.match(r"(\d+)\.(\d+)", transformers.__version__)
    if match is None:
        # Unparseable means a dev build; assume current rather than ancient.
        return (5, 0)
    return (int(match.group(1)), int(match.group(2)))


def dtype_kwarg(dtype: str) -> dict[str, Any]:
    """Name the weights-dtype keyword the way this transformers spells it.

    Renamed in 4.56: ``torch_dtype`` before, ``dtype`` from then on with
    ``torch_dtype`` kept as a deprecated alias. Passing the newer name to an
    older transformers is not ignored -- ``from_pretrained`` forwards keywords it
    does not recognise to the model constructor, which then dies with
    ``unexpected keyword argument 'dtype'`` *after* the config has loaded, which
    reads like a model problem rather than a version one.
    """
    key = "dtype" if transformers_version() >= (4, 56) else "torch_dtype"
    return {key: resolve_dtype(dtype)}


def apply_attn_implementation(
    config: Any,
    implementation: str,
    paths: Sequence[str],
) -> list[str]:
    """Write the attention backend onto every node of the nested config.

    transformers 4.52 propagates ``attn_implementation`` from the top-level
    config one level down and stops: ``thinker_config`` and ``talker_config``
    take it, ``thinker_config.vision_config`` and its siblings do not. The
    modules are built from those leaves, so a request for sdpa left the three
    towers that matter on eager -- silently, because a config that never
    received the value is not an error.

    Setting them here and handing the result to ``from_pretrained`` as
    ``config=`` closes that gap. Returns the paths it wrote to, so the caller can
    say what it asked for; ``describe_attn_implementations`` then reports what
    actually stuck, and the two together make a failed request visible.

    ``paths`` is the caller's ``ATTN_CONFIG_PATHS``: which nodes exist is a
    property of the model generation, not of this function.
    """
    applied: list[str] = []
    for path in paths:
        node = config
        for attribute in filter(None, path.split(".")):
            node = getattr(node, attribute, None)
            if node is None:
                break
        if node is None:
            continue
        try:
            node._attn_implementation = implementation
        except Exception:  # noqa: BLE001 - a node that refuses is not fatal
            continue
        applied.append(path or "config")
    return applied


def describe_attn_implementations(model: Any, paths: Sequence[str]) -> list[str]:
    """Which attention backend each tower resolved to, as "path=impl" strings.

    Purely diagnostic, and deliberately total: an unknown config layout yields a
    shorter list rather than an exception, because a print that can abort a job
    after the weights have loaded is worse than no print at all.

    Worth having because an eager tower materialises a full N-by-N score matrix,
    and the vision tower attends over patches -- four per visual token -- so its
    matrix is sixteen times the one the token count suggests. On long multi-turn
    video prompts that is the difference between fitting on the card and not.

    Note that a *requested* implementation reaching the top-level config says
    nothing about the leaves: the modules are built from
    ``thinker_config.vision_config`` and friends, and transformers 4.52 does not
    always propagate that far. Read the leaf entries, not the first one.
    """
    found: list[str] = []
    for path in paths:
        node = getattr(model, "config", None)
        for attribute in filter(None, path.split(".")):
            node = getattr(node, attribute, None)
            if node is None:
                break
        if node is None:
            continue
        implementation = getattr(node, "_attn_implementation", None)
        if implementation is not None:
            found.append(f"{path or 'config'}={implementation}")
    return found


# Class attributes transformers uses to advertise which attention backends a
# module can run. A backend a class does not declare is silently not applied --
# no exception -- which is the failure this reports.
ATTN_SUPPORT_FLAGS = ("_supports_sdpa", "_supports_flash_attn_2", "_supports_flex_attn")


def describe_attn_support(model: Any, max_depth: int = 3) -> list[str]:
    """What each tower's class says it *can* run, as "Class sdpa=… flash=…".

    Answers the question the resolved implementations cannot: whether a tower is
    on eager because nothing asked it to change, or because its class has no
    other attention path and the request was dropped. Those need different
    fixes -- the first is ours, the second needs a newer transformers -- and the
    log otherwise looks identical either way.

    Walks the loaded model rather than importing class names by module path, so
    it keeps working when upstream renames or moves them, and reports the classes
    that were actually instantiated. Deduplicated by class, shallow-limited, and
    total: any failure yields a shorter list.
    """
    found: dict[str, str] = {}

    def visit(module: Any, path: str, depth: int) -> None:
        cls = type(module)
        flags = {name: getattr(cls, name, None) for name in ATTN_SUPPORT_FLAGS}
        if any(value is not None for value in flags.values()):
            rendered = " ".join(
                f"{name.removeprefix('_supports_')}={value}"
                for name, value in flags.items()
                if value is not None
            )
            found.setdefault(cls.__name__, f"{cls.__name__}[{path or 'model'}] {rendered}")
        if depth >= max_depth:
            return
        children = getattr(module, "named_children", None)
        if children is None:
            return
        for name, child in children():
            visit(child, f"{path}.{name}" if path else name, depth + 1)

    try:
        visit(model, "", 0)
    except Exception as exc:  # noqa: BLE001 - a diagnostic must not end a job
        return [f"<unavailable: {type(exc).__name__}: {exc}>"]
    return list(found.values())


def describe_sdpa_backends() -> str:
    """Which SDPA kernels torch will consider on this machine.

    The reason this matters: ``sdpa`` is three kernels, and ``math`` materialises
    the score matrix exactly like eager. A tower can report ``_attn_implementation
    = sdpa`` and still get eager's memory profile because the other two backends
    declined -- the block-diagonal mask the vision tower builds from cu_seqlens is
    a common reason. When peak memory does not move after switching to sdpa, this
    line is the next thing to read.
    """
    probes = {
        "flash": "flash_sdp_enabled",
        "mem_efficient": "mem_efficient_sdp_enabled",
        "math": "math_sdp_enabled",
        "cudnn": "cudnn_sdp_enabled",
    }
    parts: list[str] = []
    backend_module = getattr(getattr(torch, "backends", None), "cuda", None)
    for label, attribute in probes.items():
        probe = getattr(backend_module, attribute, None)
        if probe is None:
            continue
        try:
            parts.append(f"{label}={probe()}")
        except Exception:  # noqa: BLE001 - diagnostic only
            continue
    return " ".join(parts) or "<not reported by this torch>"


FPS_SHAPES = ("scalar", "list")


def fps_processor_kwargs(sample_fps: Any, video_count: int, shape: str) -> dict[str, Any]:
    """Pass the sampling rate to the processor in one of the two shapes it takes.

    ``process_mm_info(return_video_kwargs=True)`` returns ``{"fps": [...]}``, one
    entry per video. What the processor then does with it varies: some versions
    divide by it as a scalar (``temporal_patch_size / fps``, which raises
    ``TypeError: unsupported operand type(s) for /: 'int' and 'list'`` when given
    a list), others expect one entry per video and reject a scalar.

    Which is which is NOT reliably a function of the version number -- this was
    first written as a 4.x/5.x branch and that branch was wrong for the 4.5x in
    the qwen image. So the shape is not predicted here; the caller tries one and
    falls back to the other. See ``QwenOmniModel.call_processor``.
    """
    if sample_fps is None:
        return {}
    values = list(sample_fps) if isinstance(sample_fps, (list, tuple)) else [sample_fps]
    if not values:
        return {}
    if shape == "scalar":
        return {"fps": float(values[0])}
    return {"fps": [float(value) for value in values][:video_count] or [float(values[0])]}
