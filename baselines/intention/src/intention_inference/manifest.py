from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def get_nested_value(record: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = record
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def load_manifest_records(input_json_path: Path) -> list[dict[str, Any]]:
    with input_json_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, Mapping):
        records = None
        for key in ("records", "entries", "items", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                records = value
                break
        if records is None:
            raise ValueError(
                f"{input_json_path} must be a list or contain one of: "
                "records, entries, items, data"
            )
    else:
        raise ValueError(f"Unsupported JSON payload type in {input_json_path}: {type(payload)}")

    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"Record {index} is not a JSON object: {record!r}")
        normalized.append(dict(record))
    return normalized
