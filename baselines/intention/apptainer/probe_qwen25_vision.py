#!/usr/bin/env python3
"""Check whether Qwen2.5-Omni vision memory still scales quadratically."""

from __future__ import annotations

import argparse
import gc
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

_INTENTION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_INTENTION_ROOT / "src"))

import torch  # noqa: E402
import transformers  # noqa: E402

from models.base import ChatMessage, MediaPart  # noqa: E402
from models.config import load_model_config  # noqa: E402
from models.media_io import FRAME_FACTOR, get_video_frame_count  # noqa: E402
from models.qwen.shared import transformers_version  # noqa: E402
from models.registry import load_model  # noqa: E402

DEFAULT_MODEL_CONFIG = _INTENTION_ROOT / "src/intention_inference/model_config.json"
_MIB = 1024 * 1024


def frame_counts(value: str) -> list[int]:
    try:
        counts = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use comma-separated integers") from exc
    if (
        len(counts) < 2
        or counts != sorted(set(counts))
        or any(count < FRAME_FACTOR or count % FRAME_FACTOR for count in counts)
    ):
        raise argparse.ArgumentTypeError(
            f"provide at least two unique, increasing multiples of {FRAME_FACTOR}"
        )
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument(
        "--image", type=Path, help="Optional PA-style participant image"
    )
    parser.add_argument("--backend", choices=("qwen3b", "qwen7b"), default="qwen3b")
    parser.add_argument(
        "--frames", type=frame_counts, default=frame_counts("4,8,16,32")
    )
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    return parser.parse_args()


def clean_cuda() -> None:
    gc.collect()
    for device in range(torch.cuda.device_count()):
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
        torch.cuda.synchronize(device)


def allocated(*, peak: bool = False) -> list[float]:
    measure = torch.cuda.max_memory_allocated if peak else torch.cuda.memory_allocated
    return [measure(device) / _MIB for device in range(torch.cuda.device_count())]


def main() -> None:
    args = parse_args()
    video = args.video.expanduser().resolve()
    image = args.image.expanduser().resolve() if args.image else None
    config_path = args.model_config.expanduser().resolve()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device is visible; submit the DAIC probe job.")
    if transformers_version() < (4, 54):
        raise SystemExit(
            f"Transformers >=4.54 is required, found {transformers.__version__}."
        )
    for label, path in (("video", video), ("model config", config_path)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    if image is not None and not image.is_file():
        raise FileNotFoundError(f"image not found: {image}")
    available = get_video_frame_count(video)
    if available is not None and args.frames[-1] > available:
        raise ValueError(f"video has {available} frames; requested {args.frames[-1]}")

    config = load_model_config(config_path, args.backend)
    if config.model_id is None or not Path(config.model_id).is_dir():
        raise FileNotFoundError(f"model directory not visible: {config.model_id}")

    print(f"[INFO] transformers = {transformers.__version__}")
    print(f"[INFO] torch        = {torch.__version__}")
    print(f"[INFO] backend      = {args.backend}")
    print(f"[INFO] video        = {video}")
    print(f"[INFO] image        = {image or '<none>'}")

    model = load_model(args.backend, config.model_id, **config.load_kwargs)
    generation = replace(config.generation, max_new_tokens=1, do_sample=False)
    clean_cuda()
    baseline = allocated()
    print(
        "[INFO] baseline MiB = "
        + ", ".join(f"d{i}:{value:.1f}" for i, value in enumerate(baseline))
    )

    results: list[tuple[int, float]] = []
    for frames in args.frames:
        clean_cuda()
        for device in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(device)

        content = [] if image is None else [MediaPart.image(image)]
        content += [
            MediaPart.video(video, num_frames=frames),
            MediaPart.text_part("Briefly describe the social interaction."),
        ]
        started = time.perf_counter()
        try:
            model.generate([ChatMessage("user", content)], generation)
            clean_cuda()
        except Exception as exc:  # noqa: BLE001 - an OOM is the probe result
            print(
                f"[frames={frames:>3}] FAIL after {time.perf_counter() - started:.1f}s: {exc}"
            )
            raise SystemExit(1) from exc

        deltas = [peak - base for peak, base in zip(allocated(peak=True), baseline)]
        total = sum(deltas)
        results.append((frames, total))
        devices = ", ".join(f"d{i}:+{value:.1f}" for i, value in enumerate(deltas))
        print(
            f"[frames={frames:>3}] above baseline MiB = {devices}; "
            f"total:+{total:.1f}; {time.perf_counter() - started:.1f}s"
        )

    slopes = [
        (
            before_frames,
            after_frames,
            math.log(after_mib / before_mib) / math.log(after_frames / before_frames),
        )
        for (before_frames, before_mib), (after_frames, after_mib) in zip(
            results, results[1:]
        )
        if before_mib > 0 and after_mib > 0
    ]
    for before, after, slope in slopes:
        print(f"[slope {before:>3}->{after:>3}] {slope:.2f}")

    max_slope = max((slope for _, _, slope in slopes), default=None)
    if max_slope is None or max_slope >= 1.6:
        raise SystemExit(
            "[RESULT] FAIL: quadratic scaling remains or slope was not measurable"
        )
    print(f"[RESULT] PASS: maximum slope {max_slope:.2f}; no quadratic step observed")


if __name__ == "__main__":
    main()
