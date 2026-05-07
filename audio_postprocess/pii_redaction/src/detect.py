from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, Iterable, List, Tuple

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider


@dataclass
class Detection:
    entity_type: str
    score: float
    start_char: int
    end_char: int
    source: str


def build_analyzer(language: str, policy: Dict[str, Any]) -> AnalyzerEngine:
    model_name = policy.get("language", {}).get("spacy_model", "en_core_web_sm")
    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": language, "model_name": model_name}],
        }
    )
    nlp_engine = provider.create_engine()
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()

    for entity_type, cfg in policy.get("detectors", {}).items():
        if "regex" not in cfg:
            continue
        patterns = [
            Pattern(name=f"{entity_type}_pattern_{idx}", regex=regex, score=float(cfg["confidence"]["min_score"]))
            for idx, regex in enumerate(cfg["regex"], start=1)
        ]
        registry.add_recognizer(PatternRecognizer(supported_entity=entity_type, patterns=patterns))

    return AnalyzerEngine(
        nlp_engine=nlp_engine,
        registry=registry,
        supported_languages=[language],
    )


def detect_segment_pii(
    text: str,
    analyzer: AnalyzerEngine,
    include_types: Iterable[str],
    language: str,
) -> List[Detection]:
    results = analyzer.analyze(text=text, entities=list(include_types), language=language)
    detections = []
    for item in results:
        detections.append(
            Detection(
                entity_type=item.entity_type,
                score=float(item.score),
                start_char=int(item.start),
                end_char=int(item.end),
                source="presidio",
            )
        )
    return dedupe_detections(detections)


def dedupe_detections(detections: List[Detection]) -> List[Detection]:
    seen: set[Tuple[str, int, int]] = set()
    unique: List[Detection] = []
    for item in sorted(detections, key=lambda d: (d.start_char, d.end_char, -d.score)):
        key = (item.entity_type, item.start_char, item.end_char)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _normalize_token(token: str, trim_punct: bool) -> str:
    token = token.strip()
    if trim_punct:
        token = token.strip(" \t\n\r\"'`“”‘’,.:;!?()[]{}")
    return token


def build_word_char_spans(segment_text: str, words: List[Dict[str, Any]], trim_punct: bool) -> List[Dict[str, Any]]:
    spans: List[Dict[str, Any]] = []
    cursor = 0
    lower_text = segment_text.casefold()

    for idx, word in enumerate(words):
        raw = str(word.get("word", ""))
        token = _normalize_token(raw, trim_punct)
        if not token:
            continue

        token_l = token.casefold()
        found = lower_text.find(token_l, cursor)
        if found < 0:
            found = lower_text.find(token_l)
        if found < 0:
            continue

        start_char = found
        end_char = found + len(token)
        spans.append(
            {
                "word_index": idx,
                "start_char": start_char,
                "end_char": end_char,
                "word": raw,
                "start": word.get("start"),
                "end": word.get("end"),
                "start_abs_seconds": word.get("start_abs_seconds"),
                "end_abs_seconds": word.get("end_abs_seconds"),
                "start_abs_clock": word.get("start_abs_clock"),
                "end_abs_clock": word.get("end_abs_clock"),
            }
        )
        cursor = end_char
    return spans


def map_detection_to_words(detection: Detection, word_spans: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    overlapping = [
        item
        for item in word_spans
        if item["end_char"] > detection.start_char and item["start_char"] < detection.end_char
    ]
    if not overlapping:
        return None

    timed = [
        item
        for item in overlapping
        if item.get("start") is not None and item.get("end") is not None
    ]
    if not timed:
        return None

    first = timed[0]
    last = timed[-1]
    text = " ".join(item["word"] for item in timed).strip()
    return {
        "type": detection.entity_type,
        "text": text,
        "score": detection.score,
        "source": detection.source,
        "start_word_index": first["word_index"],
        "end_word_index": last["word_index"],
        "start": float(first["start"]),
        "end": float(last["end"]),
        "start_abs_seconds": first.get("start_abs_seconds"),
        "end_abs_seconds": last.get("end_abs_seconds"),
        "start_abs_clock": first.get("start_abs_clock"),
        "end_abs_clock": last.get("end_abs_clock"),
    }
