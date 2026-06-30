from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

DEFAULT_SYSTEM_PROMPT = "You are a helpful multimodal assistant."


class SafeFormatDict(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def load_prompt_config(prompt_config_path: Any) -> dict[str, str]:
    with prompt_config_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, Mapping):
        raise ValueError(f"Prompt config must be a JSON object: {prompt_config_path}")

    system_prompt = payload.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    if not isinstance(system_prompt, str):
        raise ValueError(f"{prompt_config_path} field 'system_prompt' must be a string.")

    user_prompt_template = payload.get("user_prompt_template", payload.get("prompt"))
    if not isinstance(user_prompt_template, str) or not user_prompt_template.strip():
        sections: list[str] = []

        intro = payload.get("intro")
        if isinstance(intro, str) and intro.strip():
            sections.append(intro.strip())

        questions = payload.get("questions")
        if isinstance(questions, Sequence) and not isinstance(
            questions, (str, bytes, bytearray)
        ):
            question_lines = ["Questions:"]
            for index, question in enumerate(questions, start=1):
                if not isinstance(question, Mapping):
                    continue
                label = question.get("label")
                prompt = question.get("prompt")
                response_format = question.get("response_format")
                parts: list[str] = []
                if isinstance(label, str) and label.strip():
                    parts.append(f"{index}. {label.strip()}")
                if isinstance(prompt, str) and prompt.strip():
                    parts.append(prompt.strip())
                if isinstance(response_format, str) and response_format.strip():
                    parts.append(f"Response format: {response_format.strip()}")
                if parts:
                    question_lines.append("\n".join(parts))
            if len(question_lines) > 1:
                sections.append("\n\n".join(question_lines))

        examples = payload.get("examples")
        if isinstance(examples, Sequence) and not isinstance(examples, (str, bytes, bytearray)):
            example_lines = ["Examples:"]
            for index, example in enumerate(examples, start=1):
                if isinstance(example, str) and example.strip():
                    example_lines.append(f"{index}. {example.strip()}")
                elif isinstance(example, Mapping):
                    title = example.get("title")
                    content = example.get("content")
                    parts: list[str] = []
                    if isinstance(title, str) and title.strip():
                        parts.append(f"{index}. {title.strip()}")
                    if isinstance(content, str) and content.strip():
                        parts.append(content.strip())
                    if parts:
                        example_lines.append("\n".join(parts))
            if len(example_lines) > 1:
                sections.append("\n\n".join(example_lines))

        user_prompt_template = "\n\n".join(section for section in sections if section.strip())
        if not user_prompt_template.strip():
            raise ValueError(
                f"{prompt_config_path} must define 'user_prompt_template' (or 'prompt'), "
                "or provide structured 'intro'/'questions'/'examples' fields."
            )

    return {
        "system_prompt": system_prompt.strip(),
        "user_prompt_template": user_prompt_template.strip(),
    }


def flatten_record_for_prompt(
    record: Mapping[str, Any],
    prefix: str = "",
) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in record.items():
        flat_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(flatten_record_for_prompt(value, prefix=flat_key))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            flattened[flat_key] = value
            if prefix == "":
                flattened.setdefault(str(key), value)
    return flattened


def render_prompt(template: str, record: Mapping[str, Any]) -> str:
    format_values = SafeFormatDict(flatten_record_for_prompt(record))
    format_values["record_json"] = json.dumps(record, ensure_ascii=False)
    return template.format_map(format_values).strip()
