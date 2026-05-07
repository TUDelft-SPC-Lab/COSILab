from __future__ import annotations

import math
import wave
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def merge_ranges(spans: List[Dict], pad_ms: int, merge_gap_ms: int, min_duration_ms: int) -> List[Dict]:
    padded = []
    for span in spans:
        start_ms = max(0, int(round(span["start"] * 1000)) - pad_ms)
        end_ms = int(round(span["end"] * 1000)) + pad_ms
        if end_ms - start_ms < min_duration_ms:
            end_ms = start_ms + min_duration_ms
        padded.append(
            {
                "start_ms": start_ms,
                "end_ms": end_ms,
                "types": [span["type"]],
                "texts": [span["text"]],
            }
        )

    padded.sort(key=lambda x: (x["start_ms"], x["end_ms"]))
    merged: List[Dict] = []
    for item in padded:
        if not merged:
            merged.append(item)
            continue
        prev = merged[-1]
        if item["start_ms"] <= prev["end_ms"] + merge_gap_ms:
            prev["end_ms"] = max(prev["end_ms"], item["end_ms"])
            prev["types"].extend(item["types"])
            prev["texts"].extend(item["texts"])
        else:
            merged.append(item)
    return merged


def _read_wav_pcm(audio_path: str) -> Tuple[np.ndarray, Dict[str, int | str]]:
    with wave.open(str(audio_path), "rb") as wav:
        nchannels = wav.getnchannels()
        sampwidth = wav.getsampwidth()
        framerate = wav.getframerate()
        nframes = wav.getnframes()
        comptype = wav.getcomptype()
        if comptype != "NONE":
            raise ValueError(f"Unsupported WAV compression type: {comptype}")
        raw = wav.readframes(nframes)

    audio = _pcm_bytes_to_int32(raw, sampwidth=sampwidth, nchannels=nchannels)
    meta = {
        "nchannels": nchannels,
        "sampwidth": sampwidth,
        "framerate": framerate,
        "nframes": nframes,
        "comptype": comptype,
    }
    return audio, meta


def _write_wav_pcm(out_path: str, audio: np.ndarray, meta: Dict[str, int | str]) -> None:
    arr = np.asarray(audio, dtype=np.int32)
    if arr.ndim == 1:
        arr = arr[:, None]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(out), "wb") as wav:
        wav.setnchannels(int(meta["nchannels"]))
        wav.setsampwidth(int(meta["sampwidth"]))
        wav.setframerate(int(meta["framerate"]))
        wav.writeframes(_int32_to_pcm_bytes(arr, sampwidth=int(meta["sampwidth"])))


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
        clipped = np.clip(arr + 128, 0, 255).astype(np.uint8)
        return clipped.tobytes()
    if sampwidth == 2:
        clipped = np.clip(arr, -(1 << 15), (1 << 15) - 1).astype("<i2")
        return clipped.tobytes()
    if sampwidth == 3:
        clipped = np.clip(arr, -(1 << 23), (1 << 23) - 1).astype(np.int32).reshape(-1)
        packed = np.empty((clipped.size, 3), dtype=np.uint8)
        packed[:, 0] = clipped & 0xFF
        packed[:, 1] = (clipped >> 8) & 0xFF
        packed[:, 2] = (clipped >> 16) & 0xFF
        return packed.tobytes()
    if sampwidth == 4:
        clipped = np.clip(arr, -(1 << 31), (1 << 31) - 1).astype("<i4")
        return clipped.tobytes()
    raise ValueError(f"Unsupported sample width: {sampwidth} bytes")


def _full_scale(sampwidth: int) -> int:
    return (1 << (8 * sampwidth - 1)) - 1


def _ms_to_sample(ms: int, sr: int, nframes: int) -> int:
    return max(0, min(nframes, int(round(ms * sr / 1000.0))))


def _merged_ranges_to_sample_ranges(merged_ranges: List[Dict], sr: int, nframes: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for item in merged_ranges:
        start = _ms_to_sample(int(item["start_ms"]), sr, nframes)
        end = _ms_to_sample(int(item["end_ms"]), sr, nframes)
        if end > start:
            out.append((start, end))
    return out


def _dbfs_to_peak_amplitude(target_dbfs: float, sampwidth: int) -> float:
    rms = (10.0 ** (target_dbfs / 20.0)) * _full_scale(sampwidth)
    peak = rms * math.sqrt(2.0)
    return min(float(_full_scale(sampwidth)), max(1.0, peak))


def _generate_beep(nframes: int, sr: int, nchannels: int, sampwidth: int, frequency_hz: int, target_dbfs: float) -> np.ndarray:
    if nframes <= 0:
        return np.zeros((0, nchannels), dtype=np.int32)

    t = np.arange(nframes, dtype=np.float64) / float(sr)
    peak = _dbfs_to_peak_amplitude(target_dbfs, sampwidth)
    wave_mono = np.sin(2.0 * math.pi * float(frequency_hz) * t) * peak
    beep = np.rint(wave_mono).astype(np.int32)[:, None]
    return np.repeat(beep, nchannels, axis=1)


def _lowpass_fft(segment: np.ndarray, sr: int, cutoff_hz: int) -> np.ndarray:
    if segment.size == 0:
        return segment

    seg = segment.astype(np.float64)
    out = np.empty_like(seg)
    freqs = np.fft.rfftfreq(seg.shape[0], d=1.0 / sr)
    mask = freqs <= float(cutoff_hz)

    for ch in range(seg.shape[1]):
        spectrum = np.fft.rfft(seg[:, ch])
        spectrum[~mask] = 0
        filtered = np.fft.irfft(spectrum, n=seg.shape[0])
        out[:, ch] = filtered
    return np.rint(out).astype(np.int32)


def apply_beep_redaction(
    audio_path: str,
    merged_ranges: List[Dict],
    out_path: str,
    frequency_hz: int,
    target_dbfs: float,
) -> str:
    audio, meta = _read_wav_pcm(audio_path)
    out = np.array(audio, copy=True)
    ranges = _merged_ranges_to_sample_ranges(merged_ranges, sr=int(meta["framerate"]), nframes=int(meta["nframes"]))

    for start, end in ranges:
        out[start:end, :] = _generate_beep(
            nframes=end - start,
            sr=int(meta["framerate"]),
            nchannels=int(meta["nchannels"]),
            sampwidth=int(meta["sampwidth"]),
            frequency_hz=frequency_hz,
            target_dbfs=target_dbfs,
        )

    _write_wav_pcm(out_path, out, meta)
    return out_path


def apply_lowpass_redaction(
    audio_path: str,
    merged_ranges: List[Dict],
    out_path: str,
    cutoff_hz: int,
) -> str:
    audio, meta = _read_wav_pcm(audio_path)
    out = np.array(audio, copy=True)
    ranges = _merged_ranges_to_sample_ranges(merged_ranges, sr=int(meta["framerate"]), nframes=int(meta["nframes"]))

    for start, end in ranges:
        out[start:end, :] = _lowpass_fft(out[start:end, :], sr=int(meta["framerate"]), cutoff_hz=cutoff_hz)

    _write_wav_pcm(out_path, out, meta)
    return out_path
