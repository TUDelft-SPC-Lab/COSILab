#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def load_participant_map(path: Path) -> Dict[str, int]:
    mic_to_pid: Dict[str, int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            ch = f"{int(row['microphone Channels ']):02d}"
            mic_to_pid[ch] = int(row["Participant ID"])
    return mic_to_pid


def parse_labels(raw: str) -> List[str]:
    return [item.strip() for item in (raw or "").split("|") if item.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export main speaker mapping JSON and metadata from a manual selection template."
    )
    ap.add_argument("--selection-csv", type=Path, required=True)
    ap.add_argument("--participant-map-csv", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-metadata-json", type=Path, required=True)
    args = ap.parse_args()

    mic_to_pid = load_participant_map(args.participant_map_csv)
    main_map: Dict[str, List[str]] = {}
    metadata: List[dict] = []

    with args.selection_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            channel = str(row["channel"]).zfill(2)
            recording_id = (row.get("recording_id") or "").strip()
            labels = parse_labels(row.get("selected_speaker_label", ""))
            suggested = (row.get("suggested_top1_speaker_label") or "").strip()
            note = (row.get("note") or "").strip()

            if not recording_id:
                raise SystemExit(f"Missing recording_id for channel {channel}")
            if not labels:
                raise SystemExit(f"Missing selected_speaker_label for {recording_id}")

            main_map[recording_id] = labels
            metadata.append(
                {
                    "channel": channel,
                    "participant_id": mic_to_pid.get(channel),
                    "recording_id": recording_id,
                    "selected_speaker_labels": labels,
                    "suggested_top1_speaker_label": suggested,
                    "note": note,
                }
            )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_metadata_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(main_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_metadata_json.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"[OK] main map -> {args.output_json}")
    print(f"[OK] metadata -> {args.output_metadata_json}")
    print(f"[OK] rows={len(metadata)}")


if __name__ == "__main__":
    main()
