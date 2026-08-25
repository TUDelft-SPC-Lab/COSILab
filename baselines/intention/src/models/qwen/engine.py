"""The Qwen2.5-Omni backend: chat formatting, media handling and generation.

Everything specific to *this generation* of Qwen-Omni lives here: the model and
processor classes, the shape of its config tree, and its weights path. The
plumbing it shares with Qwen3-Omni -- the chat content dicts, the qwen_omni_utils
loader, the transformers-version quirks, the attention diagnostics -- lives in
``models.qwen.shared`` and is re-exported below, so every name this module used
to define is still importable from it.

Importing this module imports torch, so it is reached only through
``models.registry.load_model("qwen3b")`` or ``load_model("qwen7b")`` -- never
from task code.

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
from collections.abc import Sequence
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
from models.media_io import frames_at_fps

# Imported rather than defined here since the Qwen3-Omni backend needs the same
# plumbing. Every one of these was a module-level name in this file before, and
# importing it keeps it one: `from models.qwen.engine import dtype_kwarg` and
# friends still resolve. resolve_dtype is unused below and imported for exactly
# that reason.
from models.qwen.shared import (
    ATTN_SUPPORT_FLAGS,
    FPS_SHAPES,
    apply_attn_implementation,
    describe_attn_implementations,
    describe_attn_support,
    describe_device_map,
    describe_sdpa_backends,
    dtype_kwarg,
    fps_processor_kwargs,
    load_process_mm_info,
    resolve_dtype,
    to_qwen_messages,
    transformers_version,
)

__all__ = [
    "ATTN_SUPPORT_FLAGS",
    "DEFAULT_MODEL_PATH",
    "QWEN_SYSTEM_PROMPT",
    "QwenOmniModel",
    "apply_attn_implementation",
    "describe_attn_implementations",
    "describe_attn_support",
    "describe_device_map",
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

# Sub-configs that carry their own _attn_implementation, as dotted paths from
# the top-level config. Qwen2.5-Omni is several towers under one config and they
# do not have to agree: transformers resolves the attention backend per module,
# so the thinker can be on sdpa while the vision tower falls back to eager.
#
# Qwen3-Omni's tree is a different shape, which is why the two functions that
# walk this take it as an argument rather than closing over one list.
ATTN_CONFIG_PATHS = (
    "",
    "thinker_config",
    "thinker_config.text_config",
    "thinker_config.vision_config",
    "thinker_config.audio_config",
    "talker_config",
)


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
            applied = apply_attn_implementation(config, attn_implementation, ATTN_CONFIG_PATHS)
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
        print(f"[INFO] Qwen Accelerate placement: {describe_device_map(self.model)}", flush=True)
        attn_implementations = describe_attn_implementations(self.model, ATTN_CONFIG_PATHS)
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
        process_mm_info = load_process_mm_info()
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
