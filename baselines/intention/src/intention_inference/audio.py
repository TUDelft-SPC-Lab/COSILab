from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def select_audio_speakers(record: Mapping[str, Any]) -> tuple[int, list[int]] | None:
    participant = record.get("participant")
    if not isinstance(participant, int) or participant <= 0:
        return None

    conversation_floor = record.get("conversation_floor", [])
    if conversation_floor is None:
        conversation_floor = []
    if not isinstance(conversation_floor, Sequence) or isinstance(
        conversation_floor, (str, bytes, bytearray)
    ):
        return None

    floor_ids: list[int] = []
    for speaker_id in conversation_floor:
        if not isinstance(speaker_id, int) or speaker_id <= 0:
            return None
        if speaker_id != participant and speaker_id not in floor_ids:
            floor_ids.append(speaker_id)
    return participant, floor_ids


def load_audio_for_mix(
    audio_path: Path, target_sampling_rate: int | None = None
) -> tuple[Any, int]:
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("librosa is required to aggregate conversation-floor audio") from exc

    audio, sampling_rate = librosa.load(str(audio_path), sr=None, mono=True)
    if target_sampling_rate is not None and sampling_rate != target_sampling_rate:
        audio = librosa.resample(
            y=audio,
            orig_sr=sampling_rate,
            target_sr=target_sampling_rate,
        )
        sampling_rate = target_sampling_rate
    return audio, int(sampling_rate)


def write_audio_for_mix(audio_path: Path, audio: Any, sampling_rate: int) -> None:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("soundfile is required to write aggregated audio") from exc

    audio_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(audio_path), audio, sampling_rate, subtype="FLOAT")


def aggregate_conversation_floor_audio(
    *,
    participant_audio_path: Path,
    conversation_floor_audio_paths: Sequence[Path],
    output_audio_path: Path,
    record_id: str,
    length_tolerance_seconds: float = 1.0,
    length_tolerance_ratio: float = 0.05,
) -> list[str]:
    import numpy as np

    warnings: list[str] = []
    participant_audio, sampling_rate = load_audio_for_mix(participant_audio_path)
    target_length = len(participant_audio)
    tolerance_samples = max(
        int(sampling_rate * length_tolerance_seconds),
        int(target_length * length_tolerance_ratio),
    )
    mixed_tracks: list[Any] = []

    for audio_path in conversation_floor_audio_paths:
        audio, _ = load_audio_for_mix(audio_path, target_sampling_rate=sampling_rate)
        length_delta = len(audio) - target_length
        if abs(length_delta) > tolerance_samples:
            warning = (
                f"record {record_id}: discarded conversation-floor audio "
                f"{audio_path} because length differs by {length_delta} sample(s)"
            )
            print(f"[WARN] {warning}", flush=True)
            warnings.append(warning)
            continue

        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        elif len(audio) > target_length:
            audio = audio[:target_length]
        mixed_tracks.append(audio)

    if not mixed_tracks:
        warning = (
            f"record {record_id}: no valid conversation-floor audio tracks to aggregate; "
            "using silence for audio2"
        )
        print(f"[WARN] {warning}", flush=True)
        warnings.append(warning)
        mixed_audio = np.zeros(target_length, dtype=np.float32)
    else:
        mixed_audio = np.sum(np.stack(mixed_tracks, axis=0), axis=0).astype(np.float32)
        peak = float(np.max(np.abs(mixed_audio))) if mixed_audio.size else 0.0
        if peak > 1.0:
            mixed_audio = mixed_audio / peak

    write_audio_for_mix(output_audio_path, mixed_audio, sampling_rate)
    return warnings
