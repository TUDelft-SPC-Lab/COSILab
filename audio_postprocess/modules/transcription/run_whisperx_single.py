#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from statistics import mean

import torch
import whisperx
from faster_whisper import WhisperModel


def parse_args():
    p = argparse.ArgumentParser(description="Run WhisperX on one local audio file (offline-friendly).")
    p.add_argument("--audio", required=True, help="Input wav path")
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write json/txt/srt",
    )
    p.add_argument("--model-dir", required=True, help="Local faster-whisper model directory")
    p.add_argument(
        "--align-model-dir",
        default=None,
        help="Optional local WhisperX alignment model directory (mainly for non-English).",
    )
    p.add_argument("--beam-size", type=int, default=5)
    p.add_argument("--compute-type", default="float16", choices=["float16", "int8", "int8_float16", "float32"])
    p.add_argument("--language", default=None, help="Force ASR language code, e.g. en/nl. Default: auto-detect")
    p.add_argument("--align-language", default="en", help="Language code used for alignment model (default: en)")
    p.add_argument(
        "--align-languages",
        default=None,
        help="Comma-separated alignment language codes, e.g. en,nl. If set, overrides --align-language.",
    )
    p.add_argument("--no-align", action="store_true", help="Disable word-level alignment")
    return p.parse_args()


def write_plain_txt(result, out_txt: Path):
    lines = []
    for seg in result.get("segments", []):
        text = (seg.get("text") or "").strip()
        if text:
            lines.append(text)
    out_txt.write_text("\n".join(lines), encoding="utf-8")


def format_srt_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(float(seconds) * 1000)))
    hours = total_ms // 3_600_000
    rem = total_ms % 3_600_000
    minutes = rem // 60_000
    rem %= 60_000
    secs = rem // 1000
    ms = rem % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def write_segment_srt(result, out_srt: Path):
    blocks = []
    idx = 1
    for seg in result.get("segments", []):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = seg.get("start")
        end = seg.get("end")
        if start is None or end is None:
            continue
        blocks.append(
            "\n".join(
                [
                    str(idx),
                    f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}",
                    text,
                ]
            )
        )
        idx += 1
    out_srt.write_text("\n\n".join(blocks) + ("\n" if blocks else ""), encoding="utf-8")


def transcribe_with_faster_whisper(audio_path: Path, model_dir: str, device: str, compute_type: str, language: str | None, beam_size: int):
    asr_model = WhisperModel(model_size_or_path=model_dir, device=device, compute_type=compute_type)
    segments_iter, info = asr_model.transcribe(
        str(audio_path),
        language=language,
        beam_size=beam_size,
        vad_filter=False,
    )

    segments = []
    for seg in segments_iter:
        text = (seg.text or "").strip()
        segments.append(
            {
                "id": int(seg.id),
                "start": float(seg.start),
                "end": float(seg.end),
                "text": text,
            }
        )

    return {
        "segments": segments,
        "language": info.language,
    }


def parse_align_languages(args, detected_language: str | None):
    if args.align_languages:
        raw = args.align_languages
    elif args.align_language:
        raw = args.align_language
    elif detected_language:
        raw = detected_language
    else:
        raw = "en"

    langs = []
    for item in raw.split(","):
        lang = item.strip()
        if lang and lang not in langs:
            langs.append(lang)
    return langs


def align_once(result, audio, device: str, language_code: str, align_model_dir: str | None):
    if align_model_dir:
        align_dir = Path(align_model_dir)
        if not align_dir.exists():
            raise FileNotFoundError(f"Align model dir not found: {align_dir}")
        align_model, metadata = whisperx.load_align_model(
            language_code=language_code,
            device=device,
            model_name=str(align_dir),
        )
    else:
        align_model, metadata = whisperx.load_align_model(
            language_code=language_code,
            device=device,
        )

    aligned = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )
    return aligned


def segment_avg_word_score(segment):
    words = segment.get("words") or []
    scores = [w.get("score") for w in words if isinstance(w, dict) and isinstance(w.get("score"), (int, float))]
    if not scores:
        return -1.0
    return float(mean(scores))


def merge_aligned_results(base_result, aligned_results):
    merged_segments = []
    n_segments = len(base_result.get("segments", []))

    for idx in range(n_segments):
        best_lang = None
        best_seg = None
        best_score = float("-inf")

        for lang, aligned in aligned_results.items():
            segments = aligned.get("segments") or []
            if idx >= len(segments):
                continue
            seg = segments[idx]
            seg_score = segment_avg_word_score(seg)
            if seg_score > best_score:
                best_score = seg_score
                best_seg = seg
                best_lang = lang

        if best_seg is None:
            fallback = dict(base_result["segments"][idx])
            fallback["aligned_language"] = None
            fallback["align_score_mean"] = None
            merged_segments.append(fallback)
            continue

        chosen = dict(best_seg)
        chosen["aligned_language"] = best_lang
        chosen["align_score_mean"] = best_score if best_score > float("-inf") else None
        merged_segments.append(chosen)

    merged = dict(base_result)
    merged["segments"] = merged_segments
    merged["alignment_languages_tried"] = list(aligned_results.keys())
    return merged


def main():
    args = parse_args()

    audio_path = Path(args.audio)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not Path(args.model_dir).exists():
        raise FileNotFoundError(f"Whisper model dir not found: {args.model_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and args.compute_type == "float16":
        compute_type = "int8"
    else:
        compute_type = args.compute_type

    print(f"[info] device={device} compute_type={compute_type}")
    print(f"[info] audio={audio_path}")
    print(f"[info] model_dir={args.model_dir}")

    result = transcribe_with_faster_whisper(
        audio_path=audio_path,
        model_dir=args.model_dir,
        device=device,
        compute_type=compute_type,
        language=args.language,
        beam_size=args.beam_size,
    )
    detected_language = result.get("language")
    audio = whisperx.load_audio(str(audio_path))

    if not args.no_align:
        align_langs = parse_align_languages(args, result.get("language"))
        print(f"[info] alignment languages={','.join(align_langs)}")
        if args.align_model_dir and len(align_langs) > 1:
            raise ValueError("--align-model-dir supports single-language alignment only.")

        aligned_results = {}
        for lang in align_langs:
            try:
                aligned_results[lang] = align_once(
                    result=result,
                    audio=audio,
                    device=device,
                    language_code=lang,
                    align_model_dir=args.align_model_dir,
                )
                print(f"[info] alignment ok for language={lang}")
            except Exception as exc:
                print(f"[warn] alignment failed for language={lang}: {exc}")

        if not aligned_results:
            raise RuntimeError("All alignment languages failed.")

        if len(aligned_results) == 1:
            result = next(iter(aligned_results.values()))
        else:
            result = merge_aligned_results(result, aligned_results)

    # WhisperX align output may not contain top-level language; keep ASR language for writers.
    if detected_language and "language" not in result:
        result["language"] = detected_language

    out_json = out_dir / "result.json"
    out_txt = out_dir / "result.txt"

    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_plain_txt(result, out_txt)

    out_srt = out_dir / (audio_path.stem + ".srt")
    srt_writer = whisperx.utils.get_writer("srt", str(out_dir))
    try:
        srt_writer(
            result,
            str(audio_path),
            {
                "max_line_width": None,
                "max_line_count": None,
                "highlight_words": False,
            },
        )
    except KeyError as exc:
        if exc.args != ("words",):
            raise
        print("[warn] WhisperX SRT writer expected word timings for every segment; writing segment-level SRT fallback")
        write_segment_srt(result, out_srt)

    print(f"[done] wrote: {out_json}")
    print(f"[done] wrote: {out_txt}")
    print(f"[done] wrote: {out_srt}")


if __name__ == "__main__":
    main()
