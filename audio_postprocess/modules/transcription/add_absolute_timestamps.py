#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(
        description="Add absolute timestamps to WhisperX result.json using timeline clip meta."
    )
    ap.add_argument("--result-json", type=Path, required=True, help="Path to WhisperX result.json")
    ap.add_argument("--clip-meta", type=Path, required=True, help="Path to *_clip_meta.json")
    ap.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output path (default: sibling result_abs.json)",
    )
    return ap.parse_args()


def seconds_to_clock(sec):
    t = float(sec)
    hh = int(t // 3600)
    t -= hh * 3600
    mm = int(t // 60)
    t -= mm * 60
    ss = int(t)
    ms = int(round((t - ss) * 1000))

    if ms == 1000:
        ms = 0
        ss += 1
    if ss == 60:
        ss = 0
        mm += 1
    if mm == 60:
        mm = 0
        hh += 1

    return "{:02d}:{:02d}:{:02d}.{:03d}".format(hh, mm, ss, ms)


def add_abs_to_segments(segments, clip_start_abs):
    out = []
    for seg in segments or []:
        s = dict(seg)
        if isinstance(s.get("start"), (int, float)):
            abs_start = float(clip_start_abs + float(s["start"]))
            s["start_abs_seconds"] = abs_start
            s["start_abs_clock"] = seconds_to_clock(abs_start)
        if isinstance(s.get("end"), (int, float)):
            abs_end = float(clip_start_abs + float(s["end"]))
            s["end_abs_seconds"] = abs_end
            s["end_abs_clock"] = seconds_to_clock(abs_end)

        words = s.get("words")
        if isinstance(words, list):
            words_out = []
            for w in words:
                if not isinstance(w, dict):
                    words_out.append(w)
                    continue
                wd = dict(w)
                if isinstance(wd.get("start"), (int, float)):
                    abs_start = float(clip_start_abs + float(wd["start"]))
                    wd["start_abs_seconds"] = abs_start
                    wd["start_abs_clock"] = seconds_to_clock(abs_start)
                if isinstance(wd.get("end"), (int, float)):
                    abs_end = float(clip_start_abs + float(wd["end"]))
                    wd["end_abs_seconds"] = abs_end
                    wd["end_abs_clock"] = seconds_to_clock(abs_end)
                words_out.append(wd)
            s["words"] = words_out

        out.append(s)
    return out


def main():
    args = parse_args()

    if not args.result_json.exists():
        raise FileNotFoundError("Missing result json: {}".format(args.result_json))
    if not args.clip_meta.exists():
        raise FileNotFoundError("Missing clip meta: {}".format(args.clip_meta))

    result = json.loads(args.result_json.read_text(encoding="utf-8"))
    meta = json.loads(args.clip_meta.read_text(encoding="utf-8"))

    if "clip_start_abs_seconds" not in meta:
        raise KeyError("clip_start_abs_seconds not found in {}".format(args.clip_meta))

    clip_start_abs = float(meta["clip_start_abs_seconds"])

    out = dict(result)
    out["segments"] = add_abs_to_segments(result.get("segments", []), clip_start_abs)
    out["absolute_time_anchor"] = {
        "clip_start_abs_seconds": clip_start_abs,
        "clip_start_abs_clock": seconds_to_clock(clip_start_abs),
        "source_clip_meta": str(args.clip_meta),
        "source_result_json": str(args.result_json),
        "mapping_formula": "abs_time_seconds = clip_start_abs_seconds + local_time_seconds",
    }

    output_json = args.output_json or (args.result_json.parent / "result_abs.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[done] wrote {}".format(output_json))


if __name__ == "__main__":
    main()
