from __future__ import annotations

import argparse
import csv
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .policy import ensure_dir, load_policy

SUPPORTED_ENTITY_TYPES = ("PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER")
ENTITY_PRIORITY = {
    "PHONE_NUMBER": 3,
    "EMAIL_ADDRESS": 2,
    "PERSON": 1,
}
ENTITY_TEMPLATES = {
    "PERSON": "PERSON_{n}",
    "EMAIL_ADDRESS": "[EMAIL_{n}]",
    "PHONE_NUMBER": "[PHONE_{n}]",
}
EDGE_PUNCT = " \t\n\r\"'`“”‘’,.:;!?()[]{}"


def read_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_text(path: str | Path, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def write_csv(path: str | Path, rows: List[Dict[str, Any]], headers: Iterable[str]) -> None:
    header_list = list(headers)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header_list)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in header_list})


def collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_entity_text(entity_type: str, text: str) -> str:
    text = collapse_spaces(str(text))
    if entity_type == "PERSON":
        text = text.strip(EDGE_PUNCT).casefold()
    elif entity_type == "EMAIL_ADDRESS":
        text = text.casefold()
    elif entity_type == "PHONE_NUMBER":
        text = re.sub(r"\D+", "", text)
    else:
        text = text.casefold()
    return text


def load_detection_spans(path: Path, include_types: set[str]) -> List[Dict[str, Any]]:
    data = read_json(path)
    spans = data.get("spans", []) or []
    return [span for span in spans if span.get("type") in include_types]


def build_shared_mapping(
    records: List[Dict[str, Any]],
) -> Tuple[Dict[Tuple[str, str], str], Dict[str, List[Dict[str, Any]]]]:
    counters = {entity_type: 0 for entity_type in SUPPORTED_ENTITY_TYPES}
    replacement_lookup: Dict[Tuple[str, str], str] = {}
    meta: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def sort_key(item: Dict[str, Any]) -> Tuple[str, str, str, str, float]:
        return (
            item["entity_type"],
            item["normalized_text"],
            item["section"],
            item["channel"],
            float(item.get("start_abs_seconds") or item.get("start") or 0.0),
        )

    for record in sorted(records, key=sort_key):
        entity_type = record["entity_type"]
        normalized_text = record["normalized_text"]
        key = (entity_type, normalized_text)
        if key not in replacement_lookup:
            counters[entity_type] += 1
            replacement_lookup[key] = ENTITY_TEMPLATES[entity_type].format(n=counters[entity_type])
            meta[key] = {
                "entity_type": entity_type,
                "normalized_text": normalized_text,
                "replacement": replacement_lookup[key],
                "raw_variants": set(),
                "occurrence_count": 0,
                "sections": set(),
                "channels": set(),
            }

        entry = meta[key]
        entry["occurrence_count"] += 1
        raw_text = str(record["raw_text"]).strip()
        if raw_text:
            entry["raw_variants"].add(raw_text)
        entry["sections"].add(record["section"])
        entry["channels"].add(f'{record["section"]}/{record["channel"]}')

    grouped: Dict[str, List[Dict[str, Any]]] = {entity_type: [] for entity_type in SUPPORTED_ENTITY_TYPES}
    for (entity_type, _), entry in meta.items():
        grouped[entity_type].append(
            {
                "entity_type": entry["entity_type"],
                "replacement": entry["replacement"],
                "normalized_text": entry["normalized_text"],
                "raw_variants": sorted(entry["raw_variants"]),
                "occurrence_count": entry["occurrence_count"],
                "sections": sorted(entry["sections"]),
                "channels": sorted(entry["channels"]),
            }
        )

    for entity_type in grouped:
        grouped[entity_type].sort(key=lambda item: item["replacement"])
    return replacement_lookup, grouped


def gather_records(
    sections: List[str],
    detection_base: Path,
    channel_start: int,
    channel_end: int,
    include_types: set[str],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for section in sections:
        for ch in range(channel_start, channel_end + 1):
            channel = f"ch{ch:02d}"
            spans_path = detection_base / section / "main_solo" / channel / "pii_spans.json"
            if not spans_path.exists():
                continue
            for span in load_detection_spans(spans_path, include_types=include_types):
                normalized_text = normalize_entity_text(span["type"], span.get("text", ""))
                if not normalized_text:
                    continue
                records.append(
                    {
                        "section": section,
                        "channel": channel,
                        "entity_type": span["type"],
                        "raw_text": span.get("text", ""),
                        "normalized_text": normalized_text,
                        "start": span.get("start"),
                        "start_abs_seconds": span.get("start_abs_seconds"),
                    }
                )
    return records


def index_replacements_for_channel(
    section: str,
    channel: str,
    spans: List[Dict[str, Any]],
    replacement_lookup: Dict[Tuple[str, str], str],
) -> Dict[int, List[Dict[str, Any]]]:
    indexed: Dict[int, List[Dict[str, Any]]] = {}
    for span in spans:
        normalized_text = normalize_entity_text(span["type"], span.get("text", ""))
        if not normalized_text:
            continue
        replacement = replacement_lookup.get((span["type"], normalized_text))
        if not replacement:
            continue
        segment_index = int(span["segment_index"])
        indexed.setdefault(segment_index, []).append(
            {
                "section": section,
                "channel": channel,
                "entity_type": span["type"],
                "original_text": span.get("text", ""),
                "normalized_text": normalized_text,
                "replacement": replacement,
                "score": span.get("score"),
                "start_word_index": int(span["start_word_index"]),
                "end_word_index": int(span["end_word_index"]),
                "start": span.get("start"),
                "end": span.get("end"),
                "start_abs_seconds": span.get("start_abs_seconds"),
                "end_abs_seconds": span.get("end_abs_seconds"),
                "start_abs_clock": span.get("start_abs_clock"),
                "end_abs_clock": span.get("end_abs_clock"),
            }
        )
    return indexed


def choose_non_overlapping_replacements(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ordered = sorted(
        items,
        key=lambda item: (
            int(item["start_word_index"]),
            -int(item["end_word_index"]),
            -ENTITY_PRIORITY.get(item["entity_type"], 0),
            -(float(item.get("score") or 0.0)),
        ),
    )
    chosen: List[Dict[str, Any]] = []
    current_end = -1
    for item in ordered:
        start_word_index = int(item["start_word_index"])
        end_word_index = int(item["end_word_index"])
        if start_word_index <= current_end:
            continue
        chosen.append(item)
        current_end = end_word_index
    return chosen


def pseudonymized_segment_text(words: List[Dict[str, Any]]) -> str:
    tokens = [str(word.get("word", "")).strip() for word in words if str(word.get("word", "")).strip()]
    return " ".join(tokens)


def apply_replacements_to_segment(
    segment: Dict[str, Any],
    replacements: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    updated = deepcopy(segment)
    words = updated.get("words", []) or []
    if not words or not replacements:
        return updated, []

    repl_by_start = {int(item["start_word_index"]): item for item in replacements}
    rewritten_words: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    idx = 0
    while idx < len(words):
        replacement = repl_by_start.get(idx)
        if replacement is None:
            rewritten_words.append(deepcopy(words[idx]))
            idx += 1
            continue

        end_idx = int(replacement["end_word_index"])
        span_words = words[idx : end_idx + 1]
        first = deepcopy(span_words[0])
        last = span_words[-1]
        first["word"] = replacement["replacement"]
        if "end" in last:
            first["end"] = last.get("end")
        if "end_abs_seconds" in last:
            first["end_abs_seconds"] = last.get("end_abs_seconds")
        if "end_abs_clock" in last:
            first["end_abs_clock"] = last.get("end_abs_clock")
        if "score" in first:
            scores = [float(item["score"]) for item in span_words if item.get("score") is not None]
            if scores:
                first["score"] = max(scores)
        rewritten_words.append(first)
        audit_rows.append(replacement)
        idx = end_idx + 1

    updated["words"] = rewritten_words
    updated["text"] = pseudonymized_segment_text(rewritten_words)
    return updated, audit_rows


def pseudonymize_transcript_payload(
    payload: Dict[str, Any],
    replacements_by_segment: Dict[int, List[Dict[str, Any]]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rewritten = deepcopy(payload)
    segments = rewritten.get("segments", []) or []
    all_audit_rows: List[Dict[str, Any]] = []
    for idx, segment in enumerate(segments):
        items = replacements_by_segment.get(idx, [])
        chosen = choose_non_overlapping_replacements(items)
        updated_segment, audit_rows = apply_replacements_to_segment(segment, chosen)
        segments[idx] = updated_segment
        for row in audit_rows:
            with_segment = dict(row)
            with_segment["segment_index"] = idx
            all_audit_rows.append(with_segment)
    rewritten["segments"] = segments
    return rewritten, all_audit_rows


def transcript_text_from_payload(payload: Dict[str, Any]) -> str:
    lines = [str(segment.get("text", "")).strip() for segment in payload.get("segments", []) or []]
    return "\n".join(line for line in lines if line)


def shared_map_payload(
    grouped_map: Dict[str, List[Dict[str, Any]]],
    sections: List[str],
    policy_path: str,
) -> Dict[str, Any]:
    return {
        "mode": "text_pseudonymization",
        "scope": "shared_across_sections",
        "sections": sections,
        "policy": policy_path,
        "entity_strategies": {
            "PERSON": {
                "strategy": "consistent_token",
                "template": "PERSON_{n}",
                "scope": "mingling1_plus_mingling2_all_channels",
                "match_rule": "normalized_name_string",
            },
            "EMAIL_ADDRESS": {
                "strategy": "typed_placeholder",
                "template": "[EMAIL_{n}]",
                "scope": "mingling1_plus_mingling2_all_channels",
                "match_rule": "normalized_email_string",
            },
            "PHONE_NUMBER": {
                "strategy": "typed_placeholder",
                "template": "[PHONE_{n}]",
                "scope": "mingling1_plus_mingling2_all_channels",
                "match_rule": "normalized_digits_only",
            },
        },
        "mappings": grouped_map,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript_base", required=True)
    ap.add_argument("--detection_base", required=True)
    ap.add_argument("--out_base", required=True)
    ap.add_argument("--config", default="pii_redaction/config/policy_v2.yml")
    ap.add_argument("--sections", nargs="+", required=True)
    ap.add_argument("--channel_start", type=int, default=1)
    ap.add_argument("--channel_end", type=int, default=32)
    args = ap.parse_args()

    policy = load_policy(args.config)
    include_types = set(policy["pii"]["include_types"]) & set(SUPPORTED_ENTITY_TYPES)
    transcript_base = Path(args.transcript_base)
    detection_base = Path(args.detection_base)
    out_base = Path(args.out_base)
    ensure_dir(out_base)
    ensure_dir(out_base / "shared_maps")

    records = gather_records(
        sections=args.sections,
        detection_base=detection_base,
        channel_start=args.channel_start,
        channel_end=args.channel_end,
        include_types=include_types,
    )
    replacement_lookup, grouped_map = build_shared_mapping(records)

    write_json(
        out_base / "shared_maps" / "mingling_shared_pseudonym_map.json",
        shared_map_payload(grouped_map=grouped_map, sections=args.sections, policy_path=args.config),
    )

    manifest_rows: List[Dict[str, Any]] = []
    total_written = 0
    for section in args.sections:
        for ch in range(args.channel_start, args.channel_end + 1):
            channel = f"ch{ch:02d}"
            transcript_dir = transcript_base / section / "main_solo" / channel
            detection_dir = detection_base / section / "main_solo" / channel
            out_dir = out_base / section / "main_solo" / channel
            result_abs_path = transcript_dir / "result_abs.json"
            result_path = transcript_dir / "result.json"
            detection_path = detection_dir / "pii_spans.json"
            if not result_abs_path.exists() or not result_path.exists() or not detection_path.exists():
                continue

            ensure_dir(out_dir)
            result_abs_payload = read_json(result_abs_path)
            result_payload = read_json(result_path)
            spans = load_detection_spans(detection_path, include_types=include_types)
            replacements_by_segment = index_replacements_for_channel(
                section=section,
                channel=channel,
                spans=spans,
                replacement_lookup=replacement_lookup,
            )

            pseudonymized_abs, audit_rows = pseudonymize_transcript_payload(result_abs_payload, replacements_by_segment)
            pseudonymized_local, _ = pseudonymize_transcript_payload(result_payload, replacements_by_segment)
            pseudonymized_text = transcript_text_from_payload(pseudonymized_abs)

            write_json(out_dir / "result_abs.json", pseudonymized_abs)
            write_json(out_dir / "result.json", pseudonymized_local)
            write_text(out_dir / "result.txt", pseudonymized_text + ("\n" if pseudonymized_text else ""))

            audit_headers = [
                "section",
                "channel",
                "segment_index",
                "entity_type",
                "original_text",
                "normalized_text",
                "replacement",
                "score",
                "start_word_index",
                "end_word_index",
                "start",
                "end",
                "start_abs_seconds",
                "end_abs_seconds",
                "start_abs_clock",
                "end_abs_clock",
            ]
            write_csv(out_dir / "pseudonymization_audit.csv", audit_rows, audit_headers)

            manifest_rows.append(
                {
                    "section": section,
                    "channel": channel,
                    "num_detected_spans": len(spans),
                    "num_applied_replacements": len(audit_rows),
                    "out_dir": str(out_dir),
                }
            )
            total_written += 1

    write_csv(
        out_base / "shared_maps" / "mingling_shared_pseudonymization_manifest.csv",
        manifest_rows,
        ["section", "channel", "num_detected_spans", "num_applied_replacements", "out_dir"],
    )

    print(f"sections={','.join(args.sections)}")
    print(f"shared_person_count={len(grouped_map['PERSON'])}")
    print(f"shared_email_count={len(grouped_map['EMAIL_ADDRESS'])}")
    print(f"shared_phone_count={len(grouped_map['PHONE_NUMBER'])}")
    print(f"written_channels={total_written}")


if __name__ == "__main__":
    main()
