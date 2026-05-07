#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import wave
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


class Segment(tuple):
    __slots__ = ()

    @property
    def start(self) -> float:
        return self[0]

    @property
    def end(self) -> float:
        return self[1]

    @property
    def speaker(self) -> str:
        return self[2]


def parse_rttm(rttm_path: Path) -> Tuple[str, List[Segment]]:
    segs: List[Segment] = []
    file_id: Optional[str] = None
    with rttm_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 9 or parts[0].upper() != "SPEAKER":
                continue
            file_id = parts[1]
            start = float(parts[3])
            dur = float(parts[4])
            speaker = parts[7]
            if dur <= 0:
                continue
            segs.append(Segment((start, start + dur, speaker)))
    if file_id is None:
        file_id = rttm_path.stem
    segs.sort(key=lambda x: (x.start, x.end, x.speaker))
    return file_id, segs


def resolve_keep_speakers(file_id: str, main_map: Dict[str, List[str]]) -> Optional[List[str]]:
    if file_id in main_map:
        return main_map[file_id]

    for key, value in main_map.items():
        if key == file_id:
            return value

    m = re.search(r"(?:^|_)(ch\d+)$", file_id)
    if m and m.group(1) in main_map:
        return main_map[m.group(1)]

    m2 = re.search(r"(?:^|_)ch(\d+)$", file_id)
    if m2 and m2.group(1) in main_map:
        return main_map[m2.group(1)]

    return None


def apply_pad(segs: Iterable[Segment], pad: float, t_min: float = 0.0) -> List[Segment]:
    if pad <= 0:
        return list(segs)
    out: List[Segment] = []
    for s in segs:
        out.append(Segment((max(t_min, s.start - pad), s.end + pad, s.speaker)))
    return out


def merge_segments(segs: List[Segment], merge_gap: float) -> List[Segment]:
    if not segs:
        return []
    gap = max(0.0, merge_gap)
    merged: List[Segment] = []
    cur = segs[0]
    for s in segs[1:]:
        if s.speaker == cur.speaker and s.start <= cur.end + gap:
            cur = Segment((cur.start, max(cur.end, s.end), cur.speaker))
        else:
            merged.append(cur)
            cur = s
    merged.append(cur)
    return merged


def filter_min_dur(segs: List[Segment], min_dur: float) -> List[Segment]:
    if min_dur <= 0:
        return segs
    return [s for s in segs if (s.end - s.start) >= min_dur]


def _pcm_bytes_to_int32(raw: bytes, sampwidth: int, nchannels: int) -> np.ndarray:
    if sampwidth == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.int32) - 128
    elif sampwidth == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.int32)
    elif sampwidth == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        data = (
            b[:, 0].astype(np.int32)
            | (b[:, 1].astype(np.int32) << 8)
            | (b[:, 2].astype(np.int32) << 16)
        )
        sign = 1 << 23
        data = (data ^ sign) - sign
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.int32)
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth} bytes")
    return data.reshape(-1, nchannels)


def _int32_to_pcm_bytes(audio: np.ndarray, sampwidth: int) -> bytes:
    arr = np.asarray(audio, dtype=np.int32)
    if sampwidth == 1:
        return np.clip(arr + 128, 0, 255).astype(np.uint8).tobytes()
    if sampwidth == 2:
        return np.clip(arr, -(1 << 15), (1 << 15) - 1).astype("<i2").tobytes()
    if sampwidth == 3:
        clipped = np.clip(arr, -(1 << 23), (1 << 23) - 1).astype(np.int32).reshape(-1)
        packed = np.empty((clipped.size, 3), dtype=np.uint8)
        packed[:, 0] = clipped & 0xFF
        packed[:, 1] = (clipped >> 8) & 0xFF
        packed[:, 2] = (clipped >> 16) & 0xFF
        return packed.tobytes()
    if sampwidth == 4:
        return np.clip(arr, -(1 << 31), (1 << 31) - 1).astype("<i4").tobytes()
    raise ValueError(f"Unsupported sample width: {sampwidth} bytes")


def read_wav_pcm(audio_path: Path) -> Tuple[np.ndarray, Dict[str, int]]:
    with wave.open(str(audio_path), "rb") as wav:
        meta = {
            "nchannels": wav.getnchannels(),
            "sampwidth": wav.getsampwidth(),
            "framerate": wav.getframerate(),
            "nframes": wav.getnframes(),
        }
        raw = wav.readframes(meta["nframes"])
    audio = _pcm_bytes_to_int32(raw, meta["sampwidth"], meta["nchannels"])
    return audio, meta


def write_wav_pcm(audio_path: Path, audio: np.ndarray, meta: Dict[str, int]) -> None:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(audio, dtype=np.int32)
    if arr.ndim == 1:
        arr = arr[:, None]
    with wave.open(str(audio_path), "wb") as wav:
        wav.setnchannels(meta["nchannels"])
        wav.setsampwidth(meta["sampwidth"])
        wav.setframerate(meta["framerate"])
        wav.writeframes(_int32_to_pcm_bytes(arr, meta["sampwidth"]))


def build_masks(n_samples: int, sr: int, segs: List[Segment], keep_speakers: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    keep_set = set(keep_speakers)
    main_mask = np.zeros((n_samples,), dtype=bool)
    other_mask = np.zeros((n_samples,), dtype=bool)
    for s in segs:
        i0 = max(0, min(n_samples, int(round(s.start * sr))))
        i1 = max(0, min(n_samples, int(round(s.end * sr))))
        if i1 <= i0:
            continue
        if s.speaker in keep_set:
            main_mask[i0:i1] = True
        else:
            other_mask[i0:i1] = True
    return main_mask, other_mask


def render_outputs(audio: np.ndarray, main_mask: np.ndarray, other_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    out_active = np.zeros_like(audio)
    out_solo = np.zeros_like(audio)
    out_active[main_mask, :] = audio[main_mask, :]
    solo_mask = main_mask & ~other_mask
    out_solo[solo_mask, :] = audio[solo_mask, :]
    return out_active, out_solo


def main() -> None:
    ap = argparse.ArgumentParser(description="Render main-speaker-active and main-speaker-solo audio from RTTM.")
    ap.add_argument("--audio-dir", type=Path, required=True)
    ap.add_argument("--rttm-dir", type=Path, required=True)
    ap.add_argument("--main-map", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--audio-ext", default=".wav")
    ap.add_argument("--glob", default="*.rttm")
    ap.add_argument("--pad", type=float, default=0.08)
    ap.add_argument("--merge-gap", type=float, default=0.20)
    ap.add_argument("--min-dur", type=float, default=0.30)
    args = ap.parse_args()

    main_map = json.loads(args.main_map.read_text(encoding="utf-8"))
    for k, v in list(main_map.items()):
        if isinstance(v, str):
            main_map[k] = [v]
        elif isinstance(v, list):
            main_map[k] = [str(x) for x in v]
        else:
            raise SystemExit(f"Invalid main_map value for key {k}: {type(v)}")

    rttms = sorted(args.rttm_dir.glob(args.glob))
    if not rttms:
        raise SystemExit(f"No RTTMs found in {args.rttm_dir} with glob={args.glob}")

    active_dir = args.out_root / "main_active"
    solo_dir = args.out_root / "main_solo"
    report_rows = []

    for rttm_path in rttms:
        file_id, segs = parse_rttm(rttm_path)
        keep = resolve_keep_speakers(file_id, main_map)
        if not keep:
            print(f"[SKIP] no main speaker mapping for {file_id}")
            continue

        audio_path = args.audio_dir / f"{file_id}{args.audio_ext}"
        if not audio_path.exists():
            candidates = list(args.audio_dir.glob(f"{file_id}*{args.audio_ext}"))
            if not candidates:
                print(f"[SKIP] missing audio for {file_id}")
                continue
            audio_path = candidates[0]

        audio, meta = read_wav_pcm(audio_path)
        n_samples = audio.shape[0]
        segs = apply_pad(segs, args.pad, t_min=0.0)
        segs = merge_segments(segs, args.merge_gap)
        segs = filter_min_dur(segs, args.min_dur)
        main_mask, other_mask = build_masks(n_samples, meta["framerate"], segs, keep)
        active_audio, solo_audio = render_outputs(audio, main_mask, other_mask)

        active_path = active_dir / f"{audio_path.stem}_main_active{args.audio_ext}"
        solo_path = solo_dir / f"{audio_path.stem}_main_solo{args.audio_ext}"
        write_wav_pcm(active_path, active_audio, meta)
        write_wav_pcm(solo_path, solo_audio, meta)

        report_rows.append(
            {
                "recording_id": file_id,
                "source_audio": str(audio_path),
                "active_audio": str(active_path),
                "solo_audio": str(solo_path),
                "main_speakers": keep,
                "sample_rate_hz": meta["framerate"],
                "channels": meta["nchannels"],
                "sample_width_bytes": meta["sampwidth"],
                "total_samples": n_samples,
                "active_samples": int(main_mask.sum()),
                "solo_samples": int((main_mask & ~other_mask).sum()),
                "active_seconds": float(main_mask.sum() / meta["framerate"]),
                "solo_seconds": float((main_mask & ~other_mask).sum() / meta["framerate"]),
            }
        )
        print(f"[OK] {file_id} -> active={active_path.name} solo={solo_path.name}")

    report_path = args.out_root / "render_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] wrote report: {report_path}")


if __name__ == "__main__":
    main()
