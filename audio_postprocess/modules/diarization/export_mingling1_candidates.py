#!/usr/bin/env python3
import argparse
import csv
import subprocess
from collections import defaultdict
from pathlib import Path


def parse_rttm(rttm_path: Path):
    segs = []
    with rttm_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split()
            if len(p) < 8 or p[0] != "SPEAKER":
                continue
            segs.append(
                {
                    "recording_id": p[1],
                    "start": float(p[3]),
                    "dur": float(p[4]),
                    "spk": p[7],
                }
            )
    return segs


def ffmpeg_cut(src_wav: Path, start: float, dur: float, out_wav: Path):
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(src_wav),
        "-t",
        f"{dur:.3f}",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "44100",
        "-ac",
        "1",
        str(out_wav),
    ]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description="Export mingling candidate clips from RTTM.")
    ap.add_argument(
        "--section",
        default="mingling1",
        help="Section prefix used in filenames, e.g. mingling1 or mingling2",
    )
    ap.add_argument(
        "--rttm_dir",
        required=True,
        help="Directory that contains <section>_tl_chXX.rttm",
    )
    ap.add_argument(
        "--audio_dir",
        required=True,
        help="Directory that contains <section>_tl_chXX.wav",
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Output root",
    )
    ap.add_argument("--channels", default="01-32", help="channel range, e.g. 01-32")
    ap.add_argument("--min_dur", type=float, default=2.0, help="minimum segment duration")
    ap.add_argument(
        "--top_speakers_per_channel",
        type=int,
        default=3,
        help="export top N speakers by total duration per channel",
    )
    ap.add_argument(
        "--clips_per_speaker",
        type=int,
        default=5,
        help="export top N segments per speaker by duration",
    )
    args = ap.parse_args()

    rttm_dir = Path(args.rttm_dir)
    audio_dir = Path(args.audio_dir)
    out_dir = Path(args.out_dir)
    out_clips = out_dir / "clips"
    out_csv = out_dir / "candidate_index.csv"
    out_summary = out_dir / "speaker_summary.csv"
    out_template = out_dir / "main_speaker_selection_template.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    start_ch, end_ch = args.channels.split("-")
    channels = [f"{i:02d}" for i in range(int(start_ch), int(end_ch) + 1)]

    candidate_rows = []
    summary_rows = []
    channels_top1 = {}

    for ch in channels:
        rec_id = f"{args.section}_tl_ch{ch}"
        rttm_path = rttm_dir / f"{rec_id}.rttm"
        wav_path = audio_dir / f"{rec_id}.wav"
        if not rttm_path.exists():
            print(f"[WARN] Missing RTTM: {rttm_path}")
            continue
        if not wav_path.exists():
            print(f"[WARN] Missing WAV: {wav_path}")
            continue

        segs = [s for s in parse_rttm(rttm_path) if s["dur"] >= args.min_dur]
        by_spk = defaultdict(list)
        for s in segs:
            by_spk[s["spk"]].append(s)

        spk_stats = []
        for spk, items in by_spk.items():
            total = sum(x["dur"] for x in items)
            count = len(items)
            longest = max(x["dur"] for x in items)
            spk_stats.append((spk, total, count, longest))

        spk_stats.sort(key=lambda x: x[1], reverse=True)
        top_spks = spk_stats[: args.top_speakers_per_channel]
        channels_top1[ch] = top_spks[0][0] if top_spks else ""

        for rank, (spk, total, count, longest) in enumerate(top_spks, start=1):
            summary_rows.append(
                {
                    "channel": ch,
                    "recording_id": rec_id,
                    "speaker_label": spk,
                    "rank_by_total_dur": rank,
                    "total_dur_sec": f"{total:.3f}",
                    "segment_count": count,
                    "longest_seg_sec": f"{longest:.3f}",
                }
            )
            chosen = sorted(by_spk[spk], key=lambda x: x["dur"], reverse=True)[: args.clips_per_speaker]
            for i, seg in enumerate(chosen, start=1):
                stem = f"{rec_id}_{spk}_r{rank}_clip{i:02d}_{seg['start']:.2f}_{seg['dur']:.2f}"
                out_wav = out_clips / f"ch{ch}" / spk / f"{stem}.wav"
                ffmpeg_cut(wav_path, seg["start"], seg["dur"], out_wav)
                candidate_rows.append(
                    {
                        "channel": ch,
                        "recording_id": rec_id,
                        "speaker_label": spk,
                        "rank_by_total_dur": rank,
                        "clip_rank_within_speaker": i,
                        "start_sec": f"{seg['start']:.3f}",
                        "dur_sec": f"{seg['dur']:.3f}",
                        "clip_path": str(out_wav),
                    }
                )

        print(f"[OK] {rec_id}: speakers={len(spk_stats)} exported_top={len(top_spks)}")

    with out_summary.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "channel",
                "recording_id",
                "speaker_label",
                "rank_by_total_dur",
                "total_dur_sec",
                "segment_count",
                "longest_seg_sec",
            ],
        )
        w.writeheader()
        w.writerows(summary_rows)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "channel",
                "recording_id",
                "speaker_label",
                "rank_by_total_dur",
                "clip_rank_within_speaker",
                "start_sec",
                "dur_sec",
                "clip_path",
            ],
        )
        w.writeheader()
        w.writerows(candidate_rows)

    with out_template.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "channel",
                "recording_id",
                "selected_speaker_label",
                "suggested_top1_speaker_label",
                "note",
            ],
        )
        w.writeheader()
        for ch in channels:
            rec_id = f"{args.section}_tl_ch{ch}"
            w.writerow(
                {
                    "channel": ch,
                    "recording_id": rec_id,
                    "selected_speaker_label": "",
                    "suggested_top1_speaker_label": channels_top1.get(ch, ""),
                    "note": "",
                }
            )

    print(f"[DONE] summary={out_summary}")
    print(f"[DONE] candidates={out_csv}")
    print(f"[DONE] template={out_template}")
    print(f"[DONE] clip_count={len(candidate_rows)}")


if __name__ == "__main__":
    main()
