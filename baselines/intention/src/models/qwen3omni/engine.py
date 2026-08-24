"""The Qwen3-Omni backend: chat formatting, media handling and generation.

Qwen3-Omni-30B-A3B-Instruct is a sparse-MoE omni model -- 30B total parameters,
roughly 3B active per token -- reading text, images, audio and video. It is the
same shape of thing as Qwen2.5-Omni and is driven the same way, so this module
holds only what actually differs and takes the rest from
``models.qwen.shared``:

* **Different classes.** ``Qwen3OmniMoeForConditionalGeneration`` and
  ``Qwen3OmniMoeProcessor``, which need a transformers new enough to carry them
  -- newer than the 4.52 in the Qwen2.5 image, which is why this backend has its
  own container image (job_scripts/lib/model_backends.sh).
* **A deeper config tree.** The talker nests a text config and a code-predictor
  config, and there is a third top-level tower, code2wav. See
  ``ATTN_CONFIG_PATHS``.
* **The talker is built at construction time.** See ``disable_talker`` below;
  this is the one place the Qwen2.5 recipe is actively wrong here.

Unchanged from Qwen2.5-Omni, and the reason the port is small:

* **Sampling knobs are ``thinker_``-prefixed.** ``generate`` forwards any
  ``thinker_*`` keyword to the thinker's own ``generate``, and plain
  ``max_new_tokens`` reaches the wrong model.
* **``use_audio_in_video`` must agree in three places** -- the preprocessing
  call, the processor call and ``generate``. A mismatch does not raise; it
  interleaves audio features against the wrong placeholders.
* With ``return_audio=False`` the return value is the thinker's plain token
  tensor, prompt included, so the continuation still has to be sliced off.

Importing this module imports torch, so it is reached only through
``models.registry.load_model("qwen3omni30b")`` -- never from task code.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

from models.base import (
    BaseMultimodalModel,
    ChatMessage,
    GenerationConfig,
    MediaPart,
)
from models.media_io import frames_at_fps
from models.qwen.shared import (
    FPS_SHAPES,
    apply_attn_implementation,
    describe_attn_implementations,
    describe_attn_support,
    describe_sdpa_backends,
    dtype_kwarg,
    fps_processor_kwargs,
    load_process_mm_info,
    to_qwen_messages,
)

__all__ = [
    "ATTN_CONFIG_PATHS",
    "DEFAULT_MODEL_PATH",
    "QWEN3_OMNI_SYSTEM_PROMPT",
    "Qwen3OmniModel",
]

DEFAULT_MODEL_PATH = os.environ.get(
    "QWEN3_OMNI_MODEL_PATH",
    "/tudelft.net/staff-umbrella/neon/models/Qwen3-Omni-30B-A3B-Instruct",
)

# Qwen3-Omni's own system prompt, the same string Qwen2.5-Omni uses. Its chat
# template injects this when the first turn is not a system turn, so it is
# passed explicitly instead: the prompt the model sees should not depend on
# which template version shipped with the image.
QWEN3_OMNI_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)

# Sub-configs that carry their own _attn_implementation, as dotted paths from
# the top-level config. Deeper and wider than Qwen2.5-Omni's: the talker holds a
# text config and a code-predictor config of its own, and code2wav is a third
# top-level tower. transformers resolves the attention backend per module, so
# these do not have to agree -- the thinker can be on sdpa while the vision
# tower falls back to eager -- and a leaf that is never written to is not an
# error, merely slow and large.
#
# The talker and code2wav entries stay in the list even though this task loads
# neither (see disable_talker): a run that turns the talker back on should not
# also silently lose its attention setting.
ATTN_CONFIG_PATHS = (
    "",
    "thinker_config",
    "thinker_config.text_config",
    "thinker_config.vision_config",
    "thinker_config.audio_config",
    "talker_config",
    "talker_config.text_config",
    "talker_config.code_predictor_config",
    "code2wav_config",
)


class Qwen3OmniModel(BaseMultimodalModel):
    """Qwen3-Omni, loaded locally through transformers."""

    name = "qwen3omni"
    # No "thinking": this is the Instruct checkpoint, and in any case Qwen's
    # thinker is the text tower rather than a reasoning mode, so asking for
    # enable_thinking on this backend is refused up front.
    capabilities = frozenset({"text", "image", "audio", "video", "multi_turn"})

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_PATH,
        *,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        use_audio_in_video: bool = False,
        system_prompt: str = QWEN3_OMNI_SYSTEM_PROMPT,
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
            f"[INFO] Loading Qwen3-Omni model: {model_id} "
            f"(transformers {transformers.__version__})",
            flush=True,
        )

        # The config is always loaded, not only when an attention backend was
        # requested, because disabling the talker has to happen on it. Both
        # edits ride into from_pretrained on the same object.
        config = Qwen3OmniMoeForConditionalGeneration.config_class.from_pretrained(model_id)

        if disable_talker:
            # Cleared on the config rather than by calling disable_talker()
            # afterwards, which is what the Qwen2.5 backend does. Qwen3-Omni's
            # __init__ calls enable_talker() whenever config.enable_audio_output
            # is true, so the later call frees towers that were built first --
            # and on a 30B MoE that peak is spent on a card this run already
            # fills. Setting it here means the talker and code2wav weights are
            # never materialised at all.
            config.enable_audio_output = False

        if attn_implementation is not None:
            # Applied to the config rather than passed as the
            # attn_implementation keyword: the keyword reaches only the top
            # levels of this model's config tree, and the towers are built from
            # the leaves. See models.qwen.shared.apply_attn_implementation.
            #
            # flash_attention_2 is what the model card recommends and is still
            # not the one to reach for here: the RTX PRO 6000 is Blackwell
            # (compute 12.x) and upstream flash-attn targets older
            # architectures. sdpa is the setting in model_config.json.
            applied = apply_attn_implementation(config, attn_implementation, ATTN_CONFIG_PATHS)
            print(
                f"[INFO] Qwen3-Omni attention {attn_implementation!r} requested on: "
                + (", ".join(applied) or "<no config node accepted it>"),
                flush=True,
            )

        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_id,
            config=config,
            device_map=device_map,
            **dtype_kwarg(dtype),
        )
        attn_implementations = describe_attn_implementations(self.model, ATTN_CONFIG_PATHS)
        print(
            f"[INFO] Qwen3-Omni attention (requested {attn_implementation or '<unset>'}): "
            + (", ".join(attn_implementations) or "<none reported by this config>"),
            flush=True,
        )
        for line in describe_attn_support(self.model):
            print(f"[INFO] Qwen3-Omni attention support: {line}", flush=True)
        print(f"[INFO] torch sdpa backends: {describe_sdpa_backends()}", flush=True)

        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_id)

        if disable_talker:
            # Belt and braces. enable_audio_output=False above should mean there
            # is nothing left to drop, so this reports what it found: a talker
            # that is still here means the config flag did not take, which is
            # worth seeing in the log rather than discovering as memory
            # pressure.
            if getattr(self.model, "has_talker", False):
                self.model.disable_talker()
                print(
                    "[WARN] Qwen3-Omni talker was built despite enable_audio_output=False; "
                    "dropped it after loading",
                    flush=True,
                )
            else:
                print("[INFO] Qwen3-Omni talker not built (text-only generation)", flush=True)
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
        FPS_MAX_FRAMES -- decide, and those differ from what Gemma and
        Qwen2.5-Omni land on for the same clip. Resolving here and sending an
        exact count is what makes the frame budget a property of the run rather
        than of the backend, which is the only reason two models' answers are
        comparable at all.
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
