"""The Gemma 4 chat template and the rendering fallback around it.

Kept out of ``engine.py`` because the template is a 35-line Jinja blob: it is
data, and inlining it makes the engine unreadable. It is only used when the
processor ships without a chat template of its own.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = [
    "DEFAULT_GEMMA4_CHAT_TEMPLATE",
    "normalize_message_content",
    "render_gemma_chat_text",
]

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


def normalize_message_content(messages: list[dict]) -> list[dict]:
    """Represent every message content as a list of typed content parts.

    Some Gemma processor versions assume content parts are dicts and will fail
    with "string indices must be integers" when system or assistant turns are
    plain strings.
    """
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


def render_gemma_chat_text(processor: Any, messages: list[dict], enable_thinking: bool) -> str:
    """Render the Gemma chat template without processing media paths."""
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

    if not getattr(processor, "_warned_missing_chat_template", False):
        print(
            "[WARN] Processor has no chat template; using built-in Gemma 4 fallback template.",
            flush=True,
        )
        setattr(processor, "_warned_missing_chat_template", True)
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
