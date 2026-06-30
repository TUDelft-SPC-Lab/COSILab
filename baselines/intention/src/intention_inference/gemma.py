from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

DEFAULT_GEMMA4_CHAT_TEMPLATE = """{{ bos_token }}
{%- set loop_messages = messages -%}
{%- if messages and messages[0]['role'] in ['system', 'developer'] -%}
{{ '<|turn>system\n' }}
{%- if enable_thinking is defined and enable_thinking -%}
{{ '<|think|>\n' }}
{%- endif -%}
{%- for item in messages[0]['content'] -%}
{%- if item['type'] == 'text' -%}
{{ item['text'] | trim }}
{%- endif -%}
{%- endfor -%}
{{ '<turn|>\n' }}
{%- set loop_messages = messages[1:] -%}
{%- elif enable_thinking is defined and enable_thinking -%}
{{ '<|turn>system\n<|think|>\n<turn|>\n' }}
{%- endif -%}
{%- for message in loop_messages -%}
{%- set role = 'model' if message['role'] == 'assistant' else message['role'] -%}
{{ '<|turn>' + role + '\n' }}
{%- for item in message['content'] -%}
{%- if item['type'] == 'text' -%}
{{ item['text'] | trim }}
{%- elif item['type'] == 'image' -%}
{{ '<|image|>' }}
{%- elif item['type'] == 'audio' -%}
{{ '<|audio|>' }}
{%- elif item['type'] == 'video' -%}
{{ '<|video|>' }}
{%- endif -%}
{%- endfor -%}
{{ '<turn|>\n' }}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{ '<|turn>model\n' }}
{%- endif -%}"""


def get_video_frame_count(video_path: str | Path) -> int | None:
    try:
        import av
    except ImportError:
        return None

    try:
        with av.open(str(video_path)) as container:
            video_stream = next(
                (stream for stream in container.streams if stream.type == "video"),
                None,
            )
            if video_stream is None:
                return None
            if video_stream.frames and video_stream.frames > 0:
                return int(video_stream.frames)
            if (
                video_stream.duration is not None
                and video_stream.time_base is not None
                and video_stream.average_rate is not None
            ):
                duration_seconds = float(video_stream.duration * video_stream.time_base)
                estimated_frames = int(duration_seconds * float(video_stream.average_rate))
                if estimated_frames > 0:
                    return estimated_frames

            decoded_frames = 0
            for _ in container.decode(video=0):
                decoded_frames += 1
            return decoded_frames if decoded_frames > 0 else None
    except Exception:
        return None


def select_video_num_frames(video_path: str | Path, max_video_frames: int) -> int:
    if max_video_frames <= 0:
        raise ValueError(f"max_video_frames must be positive, got {max_video_frames}")
    frame_count = get_video_frame_count(video_path)
    if frame_count is None:
        return max_video_frames
    return max(1, min(max_video_frames, frame_count))


def load_video_frames(video_path: str | Path, num_frames: int) -> list[Any]:
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("PyAV is required to pre-decode Gemma video inputs") from exc

    decoded_frames: list[Any] = []
    with av.open(str(video_path)) as container:
        for frame in container.decode(video=0):
            decoded_frames.append(frame.to_image().convert("RGB"))

    if not decoded_frames:
        raise ValueError(f"No decodable video frames found: {video_path}")
    if len(decoded_frames) <= num_frames:
        return decoded_frames
    if num_frames == 1:
        return [decoded_frames[len(decoded_frames) // 2]]

    last_index = len(decoded_frames) - 1
    sample_indices = [
        round(index * last_index / (num_frames - 1))
        for index in range(num_frames)
    ]
    return [decoded_frames[index] for index in sample_indices]


def get_audio_sampling_rate(processor: Any) -> int:
    feature_extractor = getattr(processor, "feature_extractor", None)
    sampling_rate = getattr(feature_extractor, "sampling_rate", None)
    return int(sampling_rate) if sampling_rate is not None else 16000


def load_audio_array(audio_path: str | Path, sampling_rate: int) -> Any:
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("librosa is required to pre-decode Gemma audio inputs") from exc

    audio, _ = librosa.load(str(audio_path), sr=sampling_rate, mono=True)
    return audio


def load_image(image_path: str | Path) -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to pre-decode Gemma image inputs") from exc

    with Image.open(str(image_path)) as image:
        return image.convert("RGB")


def normalize_message_content(messages: list[dict]) -> list[dict]:
    normalized_messages: list[dict] = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            normalized_content = [{"type": "text", "text": content}]
        elif isinstance(content, Sequence) and not isinstance(content, (bytes, bytearray)):
            normalized_content = []
            for item in content:
                if isinstance(item, str):
                    normalized_content.append({"type": "text", "text": item})
                else:
                    normalized_content.append(item)
        else:
            normalized_content = [{"type": "text", "text": str(content)}]

        normalized_message = dict(message)
        normalized_message["content"] = normalized_content
        normalized_messages.append(normalized_message)
    return normalized_messages


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
                num_frames = int(item.get("num_frames") or 32)
                videos.append(load_video_frames(video_path, num_frames=num_frames))
    return images, audios, videos


def combine_system_and_user_prompt(system_prompt: str, user_prompt: str) -> str:
    system_prompt = system_prompt.strip()
    user_prompt = user_prompt.strip()
    if not system_prompt:
        return user_prompt
    return f"{system_prompt}\n\n{user_prompt}"


def render_gemma_chat_text(processor: Any, messages: list[dict], enable_thinking: bool) -> str:
    messages = normalize_message_content(messages)
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        return processor.apply_chat_template(
            messages,
            enable_thinking=enable_thinking,
            **kwargs,
        )
    except TypeError:
        try:
            return processor.apply_chat_template(messages, **kwargs)
        except ValueError as exc:
            if "chat template" not in str(exc).lower():
                raise
    except ValueError as exc:
        if "chat template" not in str(exc).lower():
            raise

    if not getattr(processor, "_vlm_social_warned_missing_chat_template", False):
        print(
            "[WARN] Processor has no chat template; using built-in Gemma 4 fallback template.",
            flush=True,
        )
        setattr(processor, "_vlm_social_warned_missing_chat_template", True)
    try:
        return processor.apply_chat_template(
            messages,
            chat_template=DEFAULT_GEMMA4_CHAT_TEMPLATE,
            enable_thinking=enable_thinking,
            **kwargs,
        )
    except TypeError:
        return processor.apply_chat_template(
            messages,
            chat_template=DEFAULT_GEMMA4_CHAT_TEMPLATE,
            **kwargs,
        )


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
