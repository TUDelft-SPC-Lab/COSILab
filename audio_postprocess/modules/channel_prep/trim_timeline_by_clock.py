#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import re
from pathlib import Path
from typing import List, Set, Tuple

import soundfile as sf


_CLOCK_RE = re.compile(r"^(\d{1,2}):(\d{2}):(\d{2})(?:\.(\d{1,6}))?$")


def clock_to_seconds(s: str) -> float:
    """
    Parse HH:MM:SS or HH:MM:SS.xxx... into seconds since midnight.
    """
    m = _CLOCK_RE.match(s.strip())
    if not m:
        raise ValueError(f"Bad clock time: {s} (expected HH:MM:SS or HH:MM:SS.xxx...)")
    hh, mm, ss, frac = m.groups()
    h = int(hh)
    mi = int(mm)
    se = int(ss)
    frac_sec = int(frac) / (10 ** len(frac)) if frac else 0.0
    return h * 3600 + mi * 60 + se + frac_sec


def seconds_to_clock(sec: float) -> str:
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


def parse_channels_arg(channels: str) -> List[int]:
    s = channels.strip()
    if not s:
        raise ValueError("Empty channels")
    wanted: Set[int] = set()
    parts = [x.strip() for x in s.split(",") if x.strip()]
    for part in parts:
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a.strip()), int(b.strip())
            if a > b:
                a, b = b, a
            wanted.update(range(a, b + 1))
        else:
            wanted.add(int(part))
    return sorted(wanted)


def choose_dtype_for_read(subtype: str) -> str:
    # For PCM_* use int32 to avoid unnecessary float round-trip during pure trim.
    if subtype.startswith("PCM_"):
        return "int32"
    return "float32"


def load_timeline_meta(meta_path: Path) -> dict:
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta file: {meta_path}")
    d = json.loads(meta_path.read_text(encoding="utf-8"))
    if d.get("view") != "timeline":
        raise RuntimeError(f"Meta file is not timeline view: {meta_path}")
    if "session_start_abs_seconds" not in d:
        raise RuntimeError(f"Missing session_start_abs_seconds in: {meta_path}")
    return d


def trim_one_wav(
    in_wav: Path,
    out_wav: Path,
    start_abs: float,
    end_abs: float,
    session_start_abs: float,
    parent_meta_path: Path,
    blocksize: int = 65536,
) -> dict:
    """
    Trim a timeline-preserving WAV by absolute clock time.
    Output remains timeline-linear:
      clip_abs_time = clip_start_abs_seconds + sample_index / sr
    """
    if end_abs <= start_abs:
        raise ValueError("end_abs must be greater than start_abs")

    info = sf.info(str(in_wav))
    sr = int(info.samplerate)
    subtype = str(info.subtype)
    total_frames = int(info.frames)

    # Convert absolute times to parent sample indices
    raw_s0 = (start_abs - session_start_abs) * sr
    raw_s1 = (end_abs - session_start_abs) * sr

    # Recommended stable rule:
    # start uses floor, end uses ceil, so boundary audio is not accidentally dropped.
    s0_req = math.floor(raw_s0)
    s1_req = math.ceil(raw_s1)

    # Clamp to valid range
    s0 = max(0, min(s0_req, total_frames))
    s1 = max(0, min(s1_req, total_frames))

    if s1 <= s0:
        raise RuntimeError(
            f"Requested range has no overlap with file: {in_wav.name} "
            f"(requested samples {s0_req}:{s1_req}, clamped to {s0}:{s1})"
        )

    out_wav.parent.mkdir(parents=True, exist_ok=True)

    read_dtype = choose_dtype_for_read(subtype)
    written = 0

    with sf.SoundFile(str(in_wav), mode="r") as fin, sf.SoundFile(
        str(out_wav),
        mode="w",
        samplerate=sr,
        channels=1,
        format="WAV",
        subtype=subtype,
    ) as fout:
        fin.seek(s0)
        remaining = s1 - s0

        while remaining > 0:
            n = min(blocksize, remaining)
            block = fin.read(n, dtype=read_dtype, always_2d=True)
            if block.size == 0:
                break

            frames_read = block.shape[0]
            remaining -= frames_read

            # input should already be mono; keep first column just in case
            fout.write(block[:, 0])
            written += frames_read

    actual_end_sample = s0 + written
    actual_end_abs = session_start_abs + (actual_end_sample / sr)
    actual_start_abs = session_start_abs + (s0 / sr)

    clip_meta = {
        "schema_version": "1.0",
        "view": "timeline_clip",
        "parent_wav": str(in_wav),
        "parent_meta": str(parent_meta_path),
        "sample_rate_hz": sr,
        "audio_subtype": subtype,
        "audio_format": "WAV",
        "requested_start_abs_seconds": float(start_abs),
        "requested_end_abs_seconds": float(end_abs),
        "requested_start_abs_clock": seconds_to_clock(start_abs),
        "requested_end_abs_clock": seconds_to_clock(end_abs),
        "clip_start_abs_seconds": float(actual_start_abs),
        "clip_end_abs_seconds": float(actual_end_abs),
        "clip_start_abs_clock": seconds_to_clock(actual_start_abs),
        "clip_end_abs_clock": seconds_to_clock(actual_end_abs),
        "parent_start_sample_requested": int(s0_req),
        "parent_end_sample_requested": int(s1_req),
        "parent_start_sample_used": int(s0),
        "parent_end_sample_used": int(actual_end_sample),
        "clip_total_samples": int(written),
        "clip_duration_seconds": float(written / sr),
        "mapping_formula": "clip_abs_time_seconds = clip_start_abs_seconds + sample_index / sr",
        "note": "Trimmed directly from a timeline-preserving WAV via global linear mapping; no per-chunk time_map required."
    }
    return clip_meta


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Trim timeline-preserving per-channel WAVs by absolute clock time "
            "(e.g., 13:30:00.000 to 16:05:00.000)."
        )
    )
    ap.add_argument("--input_dir", type=Path, required=True, help="Directory containing *_tl_chXX.wav and *_tl_full_meta.json")
    ap.add_argument("--prefix", required=True, help="Session prefix, e.g. UFX01")
    ap.add_argument("--channels", required=True, help='e.g. "1-32"')

    ap.add_argument("--start", required=True, help="Absolute clock time, e.g. 13:30:00.000")
    ap.add_argument("--end", required=True, help="Absolute clock time, e.g. 16:05:00.000")

    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--out_prefix", required=True, help="Output prefix, e.g. UFX01_trim_133000_160500")
    ap.add_argument("--blocksize", type=int, default=65536)

    args = ap.parse_args()

    start_abs = clock_to_seconds(args.start)
    end_abs = clock_to_seconds(args.end)
    if end_abs <= start_abs:
        raise ValueError("--end must be later than --start")

    chs = parse_channels_arg(args.channels)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = args.input_dir / f"{args.prefix}_tl_full_meta.json"
    full_meta = load_timeline_meta(meta_path)
    session_start_abs = float(full_meta["session_start_abs_seconds"])

    all_clip_meta = []

    for ch in chs:
        in_wav = args.input_dir / f"{args.prefix}_tl_ch{ch:02d}.wav"
        if not in_wav.exists():
            raise FileNotFoundError(f"Missing timeline WAV: {in_wav}")

        out_wav = args.out_dir / f"{args.out_prefix}_tl_ch{ch:02d}.wav"
        out_meta = args.out_dir / f"{args.out_prefix}_tl_ch{ch:02d}_clip_meta.json"

        clip_meta = trim_one_wav(
            in_wav=in_wav,
            out_wav=out_wav,
            start_abs=start_abs,
            end_abs=end_abs,
            session_start_abs=session_start_abs,
            parent_meta_path=meta_path,
            blocksize=args.blocksize,
        )
        clip_meta["channel"] = ch
        out_meta.write_text(json.dumps(clip_meta, ensure_ascii=False, indent=2), encoding="utf-8")
        all_clip_meta.append({
            "channel": ch,
            "file_name": out_wav.name,
            "meta_file": out_meta.name,
            "clip_start_abs_seconds": clip_meta["clip_start_abs_seconds"],
            "clip_end_abs_seconds": clip_meta["clip_end_abs_seconds"],
            "clip_start_abs_clock": clip_meta["clip_start_abs_clock"],
            "clip_end_abs_clock": clip_meta["clip_end_abs_clock"],
            "clip_total_samples": clip_meta["clip_total_samples"],
            "clip_duration_seconds": clip_meta["clip_duration_seconds"],
        })

        print(f"[OK] ch{ch:02d}: {out_wav.name}")
        print(f"     meta: {out_meta.name}")

    summary = {
        "schema_version": "1.0",
        "view": "timeline_clip_batch",
        "input_dir": str(args.input_dir.resolve()),
        "prefix": args.prefix,
        "out_prefix": args.out_prefix,
        "channels": chs,
        "requested_start_clock": args.start,
        "requested_end_clock": args.end,
        "requested_start_abs_seconds": float(start_abs),
        "requested_end_abs_seconds": float(end_abs),
        "requested_start_abs_clock": seconds_to_clock(start_abs),
        "requested_end_abs_clock": seconds_to_clock(end_abs),
        "parent_session_start_abs_seconds": session_start_abs,
        "files": all_clip_meta,
        "note": "Per-channel clips trimmed from timeline-preserving WAVs using global linear mapping."
    }
    summary_path = args.out_dir / f"{args.out_prefix}_tl_clip_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote summary: {summary_path.name}")


if __name__ == "__main__":
    main()
