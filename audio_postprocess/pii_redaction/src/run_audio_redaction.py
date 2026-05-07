from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from .audio import apply_beep_redaction, apply_lowpass_redaction, merge_ranges
from .detect import build_analyzer, build_word_char_spans, detect_segment_pii, map_detection_to_words
from .policy import ensure_dir, load_policy


def load_segments(result_abs_json: str) -> List[Dict[str, Any]]:
    with open(result_abs_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("segments", [])


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_csv(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    headers = [
        "type",
        "text",
        "score",
        "source",
        "segment_index",
        "start_word_index",
        "end_word_index",
        "start",
        "end",
        "start_local_time",
        "end_local_time",
        "start_abs_seconds",
        "end_abs_seconds",
        "start_abs_clock",
        "end_abs_clock",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in headers})


def format_local_time(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    minutes = total_ms // 60000
    rem_ms = total_ms % 60000
    secs = rem_ms // 1000
    ms = rem_ms % 1000
    return f"{minutes:02d}:{secs:02d}.{ms:03d}"


def detect_spans(segments: List[Dict[str, Any]], policy: Dict[str, Any], language: str) -> List[Dict[str, Any]]:
    include_types = policy["pii"]["include_types"]
    min_scores = {
        entity_type: float(policy["detectors"][entity_type]["confidence"]["min_score"])
        for entity_type in include_types
    }
    trim_punct = bool(policy["word_mapping"].get("trim_surrounding_punctuation", True))
    analyzer = build_analyzer(language=language, policy=policy)

    spans: List[Dict[str, Any]] = []
    for seg_idx, segment in enumerate(segments):
        text = str(segment.get("text", "")).strip()
        words = segment.get("words", []) or []
        if not text or not words:
            continue

        detections = detect_segment_pii(text=text, analyzer=analyzer, include_types=include_types, language=language)
        word_spans = build_word_char_spans(segment_text=text, words=words, trim_punct=trim_punct)
        if not word_spans:
            continue

        for det in detections:
            if det.score < min_scores.get(det.entity_type, 0.0):
                continue
            mapped = map_detection_to_words(det, word_spans)
            if mapped is None:
                continue
            mapped["segment_index"] = seg_idx
            mapped["start_local_time"] = format_local_time(float(mapped["start"]))
            mapped["end_local_time"] = format_local_time(float(mapped["end"]))
            spans.append(mapped)

    deduped: List[Dict[str, Any]] = []
    seen = set()
    for item in spans:
        key = (item["type"], item["start"], item["end"], item["text"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", required=True, help="Path to WhisperX result_abs.json")
    ap.add_argument("--audio", required=True, help="Path to the matching wav file")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--config", default="pii_redaction/config/policy.yml")
    ap.add_argument("--language", default="en")
    args = ap.parse_args()

    policy = load_policy(args.config)
    ensure_dir(args.out_dir)

    segments = load_segments(args.transcript)
    spans = detect_spans(segments=segments, policy=policy, language=args.language)

    outputs = policy["outputs"]
    spans_json_path = Path(args.out_dir) / outputs["spans_json_name"]
    audit_csv_path = Path(args.out_dir) / outputs["audit_csv_name"]

    write_json(
        spans_json_path,
        {
            "input_transcript": args.transcript,
            "input_audio": args.audio,
            "num_spans": len(spans),
            "spans": spans,
        },
    )
    write_csv(audit_csv_path, spans)

    audio_cfg = policy["audio_redaction"]
    merged = merge_ranges(
        spans=spans,
        pad_ms=int(audio_cfg["pad_ms"]),
        merge_gap_ms=int(audio_cfg["merge_gap_ms"]),
        min_duration_ms=int(audio_cfg["min_duration_ms"]),
    )
    out_audio_path = Path(args.out_dir) / outputs["beep_audio_name"]
    apply_beep_redaction(
        audio_path=args.audio,
        merged_ranges=merged,
        out_path=str(out_audio_path),
        frequency_hz=int(audio_cfg["beep"]["frequency_hz"]),
        target_dbfs=float(audio_cfg["beep"]["target_dbfs"]),
    )
    lowpass_audio_path = Path(args.out_dir) / outputs["lowpass_audio_name"]
    apply_lowpass_redaction(
        audio_path=args.audio,
        merged_ranges=merged,
        out_path=str(lowpass_audio_path),
        cutoff_hz=int(audio_cfg["lowpass"]["cutoff_hz"]),
    )

    print(f"Detected {len(spans)} PII spans")
    print(f"Wrote spans: {spans_json_path}")
    print(f"Wrote audit: {audit_csv_path}")
    print(f"Wrote beep audio: {out_audio_path}")
    print(f"Wrote lowpass audio: {lowpass_audio_path}")


if __name__ == "__main__":
    main()
