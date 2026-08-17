"""The Gemma 4 backend: chat formatting, media handling and generation.

Everything transformers- and torch-specific about running Gemma lives here.
Importing this module imports torch, so it is reached only through
``models.registry.load_model("gemma")`` -- never from task code, and never from
``models.base``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

from models.base import (
    BaseMultimodalModel,
    ChatMessage,
    GenerationConfig,
    MediaPart,
)
from models.gemma.chat_template import normalize_message_content, render_gemma_chat_text
from models.media_io import (
    frames_at_fps,
    load_audio_array,
    load_image,
    load_video_frames,
)

__all__ = ["DEFAULT_MODEL_PATH", "GemmaModel"]

DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "/scratch/zli33/models/GemmaE4B")


def get_audio_sampling_rate(processor: Any) -> int:
    feature_extractor = getattr(processor, "feature_extractor", None)
    sampling_rate = getattr(feature_extractor, "sampling_rate", None)
    return int(sampling_rate) if sampling_rate is not None else 16000


def collect_media_inputs(
    messages: list[dict],
    audio_sampling_rate: int,
) -> tuple[list[Any], list[Any], list[Any]]:
    images: list[Any] = []
    audios: list[Any] = []
    videos: list[Any] = []
    for message in messages:
        content = message.get("content", [])
        if not isinstance(content, Sequence) or isinstance(content, (str, bytes, bytearray)):
            continue
        for item in content:
            if not isinstance(item, Mapping):
                continue
            item_type = item.get("type")
            if item_type == "image":
                image_path = item.get("image")
                if image_path is None:
                    continue
                images.append(load_image(image_path))
            elif item_type == "audio":
                audio_path = item.get("audio")
                if audio_path is None:
                    continue
                audios.append(load_audio_array(audio_path, sampling_rate=audio_sampling_rate))
            elif item_type == "video":
                video_path = item.get("video")
                if video_path is None:
                    continue
                # No silent default here: how many frames the model actually sees
                # is a result-changing parameter, so a missing count is a bug in
                # the caller (it should come from GemmaModel.prepare_video_part),
                # not something to paper over with an arbitrary number.
                num_frames = item.get("num_frames")
                if not isinstance(num_frames, int) or num_frames < 1:
                    raise ValueError(
                        f"video content item for {video_path} has no usable num_frames "
                        f"({num_frames!r}); build it with GemmaModel.prepare_video_part()."
                    )
                videos.append(load_video_frames(video_path, num_frames=num_frames))
    return images, audios, videos


def build_gemma_inputs(
    processor: Any,
    messages: list[dict],
    enable_thinking: bool,
) -> Any:
    messages = normalize_message_content(messages)
    text = render_gemma_chat_text(
        processor=processor,
        messages=messages,
        enable_thinking=enable_thinking,
    )
    images, audios, videos = collect_media_inputs(
        messages=messages,
        audio_sampling_rate=get_audio_sampling_rate(processor),
    )
    processor_kwargs: dict[str, Any] = {
        "text": text,
        "return_tensors": "pt",
        "padding": True,
    }
    if images:
        processor_kwargs["images"] = images
    if audios:
        processor_kwargs["audio"] = audios
    if videos:
        processor_kwargs["videos"] = videos
        processor_kwargs["do_sample_frames"] = False

    try:
        return processor(**processor_kwargs)
    except TypeError:
        processor_kwargs.pop("do_sample_frames", None)
        return processor(**processor_kwargs)


def get_model_input_device(model: Any) -> torch.device | str:
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def coerce_parsed_response(parsed: Any) -> str:
    if parsed is None:
        return ""
    if isinstance(parsed, str):
        return parsed.strip()
    if isinstance(parsed, Mapping):
        for key in ("answer", "response", "content", "text"):
            value = parsed.get(key)
            if isinstance(value, str):
                return value.strip()
        return json.dumps(parsed, ensure_ascii=False)
    if isinstance(parsed, Sequence) and not isinstance(parsed, (bytes, bytearray)):
        text_items = [item.strip() for item in parsed if isinstance(item, str) and item.strip()]
        if text_items:
            return text_items[-1]
    for attr in ("answer", "response", "content", "text"):
        value = getattr(parsed, attr, None)
        if isinstance(value, str):
            return value.strip()
    return str(parsed).strip()


def strip_common_special_tokens(text: str) -> str:
    replacements = (
        "<bos>",
        "<eos>",
        "<end_of_turn>",
        "<|end_of_turn|>",
        "<turn|>",
        "<|turn|>",
    )
    cleaned = text
    for token in replacements:
        cleaned = cleaned.replace(token, "")
    return cleaned.strip()


def parse_response(processor: Any, raw_response: str) -> str:
    if hasattr(processor, "parse_response"):
        parsed = coerce_parsed_response(processor.parse_response(raw_response))
        if parsed:
            return parsed
    return strip_common_special_tokens(raw_response)


def infer_turn(
    model: Any,
    processor: Any,
    messages: list[dict],
    max_new_tokens: int,
    enable_thinking: bool,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
) -> str:
    inputs = build_gemma_inputs(
        processor=processor,
        messages=messages,
        enable_thinking=enable_thinking,
    ).to(get_model_input_device(model))
    input_len = inputs["input_ids"].shape[-1]

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs.update(
            {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            }
        )

    with torch.inference_mode():
        outputs = model.generate(**inputs, **generation_kwargs)

    raw_response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    return parse_response(processor, raw_response)


class GemmaModel(BaseMultimodalModel):
    """Gemma 4 multimodal, loaded locally through transformers."""

    name = "gemma"
    capabilities = frozenset({"text", "image", "audio", "video", "thinking", "multi_turn"})

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_PATH,
        *,
        device_map: str = "auto",
        dtype: str = "auto",
    ) -> None:
        super().__init__(model_id)
        print(f"[INFO] Loading Gemma model: {model_id}", flush=True)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=device_map,
        )
        self.model.eval()

    def prepare_video_part(
        self,
        video_path: str | Path,
        max_frames: int,
        *,
        fps: float = 2.0,
        min_frames: int = 4,
    ) -> MediaPart:
        """Resolve the run's sampling policy the way every backend resolves it."""
        return MediaPart.video(
            video_path,
            num_frames=frames_at_fps(
                video_path, fps=fps, min_frames=min_frames, max_frames=max_frames
            ),
        )

    def generate(self, messages: Sequence[ChatMessage], config: GenerationConfig) -> str:
        return infer_turn(
            model=self.model,
            processor=self.processor,
            messages=to_gemma_messages(messages),
            max_new_tokens=config.max_new_tokens,
            enable_thinking=bool(config.extra.get("enable_thinking", False)),
            do_sample=config.do_sample,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
        )


def to_gemma_messages(messages: Sequence[ChatMessage]) -> list[dict]:
    """Render the neutral message form into the dicts the processor expects.

    The output is exactly the shape hand-built before the backend API existed --
    ``{"type": "image", "image": path}`` and friends -- so the prompt the model
    sees is unchanged by the refactor.
    """
    gemma_messages: list[dict] = []
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
                    video_item["num_frames"] = part.num_frames
                content.append(video_item)
            else:
                raise ValueError(f"Unsupported media part type: {part.type!r}")
        gemma_messages.append({"role": message.role, "content": content})
    return gemma_messages
