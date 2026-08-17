"""The Qwen2.5-Omni backend: chat formatting, media handling and generation.

Everything transformers- and torch-specific about running Qwen2.5-Omni lives
here. Importing this module imports torch, so it is reached only through
``models.registry.load_model("qwen7b")`` -- never from task code.

Two things differ from a plain text model and are easy to get wrong:

* **Sampling knobs are ``thinker_``-prefixed.** Qwen2.5-Omni is two models, a
  "thinker" that produces text and a "talker" that produces speech.
  ``max_new_tokens`` reaches the wrong one; ``thinker_max_new_tokens`` is the
  parameter that matters here. The talker is switched off entirely by default.
* **``use_audio_in_video`` must agree in three places** -- the preprocessing
  call, the processor call and ``generate``. A mismatch does not raise; it
  interleaves audio features against the wrong placeholders.
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
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

from models.base import (
    BaseMultimodalModel,
    ChatMessage,
    GenerationConfig,
    MediaPart,
)
from models.media_io import FRAME_FACTOR, frames_at_fps

__all__ = [
    "ATTN_SUPPORT_FLAGS",
    "DEFAULT_MODEL_PATH",
    "QWEN_SYSTEM_PROMPT",
    "QwenOmniModel",
    "apply_attn_implementation",
    "describe_attn_implementations",
    "describe_attn_support",
    "describe_sdpa_backends",
    "dtype_kwarg",
    "to_qwen_messages",
    "transformers_version",
]

DEFAULT_MODEL_PATH = os.environ.get("QWEN_MODEL_PATH", "/scratch/zli33/models/Qwen2.5-Omni-7B")

# Qwen2.5-Omni's own system prompt. Its chat template injects this when the first
# turn is not a system turn, so it is passed explicitly instead: the prompt the
# model sees should not depend on which template version shipped with the image.
QWEN_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)

# qwen_omni_utils rounds any requested frame count to a multiple of this.
FRAME_FACTOR = 2

# qwen_omni_utils is NOT vendored here: no model code lives in this repo. It comes
# from the container image (job_scripts/lib/model_backends.sh names which one),
# and this env var prepends an external checkout when the code is supplied from
# outside instead.
_OMNI_UTILS_PATH_ENV = "QWEN_OMNI_UTILS_PATH"


@lru_cache(maxsize=1)
def _load_process_mm_info() -> Any:
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


# Sub-configs that carry their own _attn_implementation, as dotted paths from
# the top-level config. Qwen2.5-Omni is several towers under one config and they
# do not have to agree: transformers resolves the attention backend per module,
# so the thinker can be on sdpa while the vision tower falls back to eager.
ATTN_CONFIG_PATHS = (
    "",
    "thinker_config",
    "thinker_config.text_config",
    "thinker_config.vision_config",
    "thinker_config.audio_config",
    "talker_config",
)


def apply_attn_implementation(config: Any, implementation: str) -> list[str]:
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
    """
    applied: list[str] = []
    for path in ATTN_CONFIG_PATHS:
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


def describe_attn_implementations(model: Any) -> list[str]:
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
    for path in ATTN_CONFIG_PATHS:
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


class QwenOmniModel(BaseMultimodalModel):
    """Qwen2.5-Omni, loaded locally through transformers."""

    name = "qwen"
    # No "thinking": Qwen's thinker is the text tower, not a reasoning mode, so
    # asking for enable_thinking on this backend is still refused up front.
    capabilities = frozenset({"text", "image", "audio", "video", "multi_turn"})

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_PATH,
        *,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        use_audio_in_video: bool = False,
        system_prompt: str = QWEN_SYSTEM_PROMPT,
        disable_talker: bool = True,
        attn_implementation: str | None = None,
    ) -> None:
        super().__init__(model_id)
        if attn_implementation == "auto":
            # "auto" is a Ming convention (resolved per card) that other repos
            # carry in their model_config, so someone copying one will try it
            # here. transformers has no such value and would fail somewhere
            # less obvious.
            raise ValueError(
                "attn_implementation 'auto' is a Ming-only convention. Name the "
                "implementation for Qwen: 'sdpa', 'eager', or 'flash_attention_2'."
            )

        self.use_audio_in_video = use_audio_in_video
        self.system_prompt = system_prompt
        # Which shape the processor wants for its fps argument. None until the
        # first video record settles it; see call_processor.
        self.fps_shape: str | None = None

        print(
            f"[INFO] Loading Qwen2.5-Omni model: {model_id} "
            f"(transformers {transformers.__version__})",
            flush=True,
        )
        # Left unset by default, which is not the same as choosing eager: it
        # means whatever this transformers decides, and on the 4.52 in the qwen
        # image that is eager for every tower even though the top-level config
        # reports sdpa. Eager materialises a full N-by-N score matrix per
        # attention layer, and the vision tower attends over patches -- four per
        # visual token -- so its matrix is sixteen times the one the token count
        # suggests. That is what put a multi-turn video prompt over a 96 GB card.
        #
        # Set it in the task's model_config.json under backends.<name>.load. The
        # line below reports what each tower actually resolved to, so a request
        # that did not take is visible rather than assumed.
        load_kwargs: dict[str, Any] = {"device_map": device_map, **dtype_kwarg(dtype)}
        if attn_implementation is not None:
            # Passed as a pre-modified config rather than as the
            # attn_implementation keyword: the keyword only reaches the top two
            # levels of this model's config tree, and the towers are built from
            # the third. See apply_attn_implementation.
            config = Qwen2_5OmniForConditionalGeneration.config_class.from_pretrained(model_id)
            applied = apply_attn_implementation(config, attn_implementation)
            print(
                f"[INFO] Qwen attention {attn_implementation!r} requested on: "
                + (", ".join(applied) or "<no config node accepted it>"),
                flush=True,
            )
            load_kwargs["config"] = config

        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_id,
            **load_kwargs,
        )
        attn_implementations = describe_attn_implementations(self.model)
        print(
            f"[INFO] Qwen attention (requested {attn_implementation or '<unset>'}): "
            + (", ".join(attn_implementations) or "<none reported by this config>"),
            flush=True,
        )
        for line in describe_attn_support(self.model):
            print(f"[INFO] Qwen attention support: {line}", flush=True)
        print(f"[INFO] torch sdpa backends: {describe_sdpa_backends()}", flush=True)
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
        if disable_talker and hasattr(self.model, "disable_talker"):
            # The speech tower is several GB of weights this task never uses.
            self.model.disable_talker()
            print("[INFO] Qwen talker disabled (text-only generation)", flush=True)
        self.model.eval()

    def prepare_video_part(
        self,
        video_path: str | Path,
        max_frames: int,
        *,
        fps: float = 2.0,
        min_frames: int = 4,
    ) -> MediaPart:
        """Resolve the run's sampling policy the way every backend resolves it.

        The count is sent as ``nframes``, never as ``fps``: handing
        qwen_omni_utils a rate would let its own constants -- FPS_MIN_FRAMES,
        FPS_MAX_FRAMES -- decide, and those differ from what Gemma and Ming
        would land on. Resolving here and sending an exact count is what makes
        the three agree.
        """
        return MediaPart.video(
            video_path,
            num_frames=frames_at_fps(
                video_path, fps=fps, min_frames=min_frames, max_frames=max_frames
            ),
        )

    def build_inputs(self, qwen_messages: list[dict]) -> Any:
        process_mm_info = _load_process_mm_info()
        text = self.processor.apply_chat_template(
            qwen_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        audios, images, videos, video_kwargs = process_mm_info(
            qwen_messages,
            use_audio_in_video=self.use_audio_in_video,
            return_video_kwargs=True,
        )
        processor_kwargs: dict[str, Any] = {
            "text": text,
            "return_tensors": "pt",
            "padding": True,
            "use_audio_in_video": self.use_audio_in_video,
        }
        if audios:
            processor_kwargs["audio"] = audios
        if images:
            processor_kwargs["images"] = images
        if not videos:
            return self.processor(**processor_kwargs)

        processor_kwargs["videos"] = videos
        return self.call_processor(processor_kwargs, video_kwargs.get("fps"), len(videos))

    def call_processor(
        self,
        processor_kwargs: dict[str, Any],
        sample_fps: Any,
        video_count: int,
    ) -> Any:
        """Call the processor, settling how this transformers wants ``fps``.

        Tries one shape, falls back to the other, and remembers the winner for
        the rest of the run, so the cost is one failed call on the first record
        rather than a version table that has to be kept true.

        Worth the machinery because of how this fails: the processor runs once
        per record inside the task's per-record try/except, so the wrong shape
        does not crash the job -- it writes ``[ERROR] unsupported operand
        type(s) for /: 'int' and 'list'`` into the ``assistant`` field of every
        single record and exits 0.
        """
        candidates = FPS_SHAPES if self.fps_shape is None else (self.fps_shape,)
        last_error: Exception | None = None
        for shape in candidates:
            attempt = dict(processor_kwargs)
            attempt.update(fps_processor_kwargs(sample_fps, video_count, shape))
            try:
                inputs = self.processor(**attempt)
            except (TypeError, ValueError) as exc:
                last_error = exc
                print(f"[INFO] processor rejected fps as {shape}: {exc}", flush=True)
                continue
            if self.fps_shape != shape:
                self.fps_shape = shape
                print(f"[INFO] this transformers wants fps as a {shape}", flush=True)
            return inputs
        assert last_error is not None
        raise last_error

    def generate(self, messages: Sequence[ChatMessage], config: GenerationConfig) -> str:
        qwen_messages = to_qwen_messages(messages, system_prompt=self.system_prompt)
        inputs = self.build_inputs(qwen_messages).to(self.model.device).to(self.model.dtype)

        generation_kwargs: dict[str, Any] = {
            "thinker_max_new_tokens": config.max_new_tokens,
            "thinker_do_sample": config.do_sample,
        }
        if config.do_sample:
            generation_kwargs.update(
                {
                    "thinker_temperature": config.temperature,
                    "thinker_top_p": config.top_p,
                    "thinker_top_k": config.top_k,
                }
            )

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                use_audio_in_video=self.use_audio_in_video,
                return_audio=False,
                **generation_kwargs,
            )

        # Only the continuation: `output` contains the prompt as well, and
        # decoding all of it would put the entire task prompt and persona into
        # every recorded answer without ever raising.
        prompt_length = inputs["input_ids"].shape[1]
        decoded = self.processor.batch_decode(
            output[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0].strip() if decoded else ""
