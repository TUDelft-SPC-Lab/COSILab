#!/usr/bin/env python3
"""Reproduce Figure 4A using an external topic_composition data package (Python 3.9+)."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

TOPICS = [f"T{i}" for i in range(1, 9)]
SESSIONS = {"mingling1": "M1", "mingling2": "M2"}
GROUPS = ("Dyad", "Triad", "4+")


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique(rows, key):
    result = {row[key]: row for row in rows}
    require(len(result) == len(rows), f"Duplicate {key}")
    return result


def parse_share(value):
    """Apply the original open-coding export's share normalization rule."""
    text = str(value or "").strip()
    try:
        number = float(text.rstrip("%"))
    except ValueError:
        return None
    require(math.isfinite(number), "Non-finite raw share")
    if text.endswith("%") or number > 1:
        number /= 100
    return None if number < 0 else min(number, 1.0)


def group_size(count):
    require(count >= 2, "A conversation group must have at least two members")
    return "Dyad" if count == 2 else "Triad" if count == 3 else "4+"


def aggregate(data):
    windows = unique(read_csv(data / "windows.csv"), "window_id")
    instances = read_csv(data / "topic_instances.csv")
    unique(instances, "topic_instance_id")
    labels = unique(read_csv(data / "raw_topic_labels.csv"), "raw_topic_id")
    assignments = unique(read_csv(data / "topic_assignments.csv"), "raw_topic_id")
    taxonomy = unique(read_csv(data / "topic_taxonomy.csv"), "figure_topic_id")
    require(set(taxonomy) == set(TOPICS), "Expected the eight T1-T8 categories")
    require(set(assignments) == set(labels), "Assignment/label IDs differ")
    require(set(assignments) == {r["raw_topic_id"] for r in instances},
            "Assignments and topic instances do not cover the same raw labels")
    for row in assignments.values():
        require(row["figure_topic_id"] in taxonomy, "Unknown topic category")
        require(row["figure_topic"] == taxonomy[row["figure_topic_id"]]["figure_topic"],
                "Category name differs from the codebook")

    by_window = defaultdict(list)
    classified = []
    for row in instances:
        require(row["window_id"] in windows, "Topic references an unknown window")
        window = windows[row["window_id"]]
        for key in ("section", "parent_segment_id", "window_start_turn_index",
                    "window_end_turn_index", "window_turn_count"):
            require(row[key] == window[key], f"Window metadata mismatch: {key}")
        by_window[row["window_id"]].append(row)
        assignment = assignments[row["raw_topic_id"]]
        classified.append({**row, "figure_topic_id": assignment["figure_topic_id"],
                           "figure_topic": assignment["figure_topic"]})

    window_rows = []
    grouped = defaultdict(list)
    for window in windows.values():
        wid = window["window_id"]
        rows = by_window[wid]
        require(window["section"] in SESSIONS, "Unknown session")
        require(window["parse_status"] == "ok", f"Invalid model output: {wid}")
        require(len(rows) == int(window["topic_count"]), f"Topic count mismatch: {wid}")
        group = group_size(int(window["participant_count"]))
        require(group == window["group_size"], f"Group size mismatch: {wid}")
        if not rows:
            continue
        raw = [parse_share(r["estimated_turn_share_raw"]) for r in rows]
        total = sum(s for s in raw if s is not None)
        shares = ([round((s or 0) / total, 6) for s in raw] if total > 0
                  else [round(1 / len(rows), 6)] * len(rows))
        values = {t: 0.0 for t in TOPICS}
        for row, share in zip(rows, shares):
            require(abs(share - float(row["normalized_window_share"])) < 1e-12,
                    f"Frozen normalized share differs: {row['topic_instance_id']}")
            values[assignments[row["raw_topic_id"]]["figure_topic_id"]] += share
        require(abs(sum(values.values()) - 1) <= len(rows) * 0.5e-6 + 1e-12,
                f"Window shares do not sum to one within rounding tolerance: {wid}")
        output = {"window_id": wid, "section": window["section"],
                  "session": SESSIONS[window["section"]],
                  "parent_segment_id": window["parent_segment_id"],
                  "participant_count": window["participant_count"], "group_size": group}
        output.update({f"{t}_share": round(values[t], 6) for t in TOPICS})
        window_rows.append(output)
        grouped[(window["section"], group)].append(values)

    composition = []
    for section, session in SESSIONS.items():
        for group in GROUPS:
            values = grouped[(section, group)]
            require(bool(values), f"No topic-bearing windows for {section}/{group}")
            means = {t: sum(v[t] for v in values) / len(values) for t in TOPICS}
            row = {"section": section, "session": session, "group_size": group,
                   "topic_bearing_window_count": len(values)}
            row.update({f"{t}_share": round(means[t], 6) for t in TOPICS})
            row.update({f"{t}_percent": round(100 * means[t], 1) for t in TOPICS})
            composition.append(row)
    counts = {"windows": len(windows), "topic_bearing_windows": len(window_rows),
              "empty_windows": len(windows) - len(window_rows),
              "topic_instances": len(instances), "raw_labels": len(labels),
              "topic_categories": len(taxonomy)}
    return composition, window_rows, classified, counts


def check_reference(rows, reference):
    expected = {(r["section"], r["group_size"]): r for r in reference}
    require(len(expected) == len(reference) == len(rows), "Reference group count differs")
    max_diff = 0.0
    for row in rows:
        key = row["section"], row["group_size"]
        require(key in expected, f"Missing reference group: {key}")
        ref = expected[key]
        require(str(row["topic_bearing_window_count"]) == ref["topic_bearing_window_count"],
                f"Reference denominator differs: {key}")
        for topic in TOPICS:
            difference = abs(row[f"{topic}_share"] - float(ref[f"{topic}_share"]))
            max_diff = max(max_diff, difference)
            require(difference < 1e-12, f"Reference share differs: {key}/{topic}")
            require(row[f"{topic}_percent"] == float(ref[f"{topic}_percent"]),
                    f"Reference percentage differs: {key}/{topic}")
    return max_diff


def plot(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    indexed = {(r["session"], r["group_size"]): r for r in rows}
    ordered = [indexed[(s, g)] for g in GROUPS for s in ("M1", "M2")]
    matrix = [[100 * r[f"{t}_share"] for t in TOPICS] for r in ordered]
    cmap = LinearSegmentedColormap.from_list("topic_composition", ["#f3f8fb", "#167489"])
    fig, ax = plt.subplots(figsize=(7.5, 3.6), constrained_layout=True)
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=35, aspect="auto")
    ax.set_xticks(range(8), TOPICS)
    ax.set_yticks(range(6), [f"{r['group_size']} {r['session']}" for r in ordered])
    ax.tick_params(length=0)
    ax.set_title("A. Topic composition", loc="left", weight="bold")
    for i, values in enumerate(matrix):
        for j, value in enumerate(values):
            ax.text(j, i, f"{value:.0f}", ha="center", va="center",
                    color="white" if value >= 23 else "#23343a", fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(im, ax=ax, label="Mean window topic share (%)", shrink=0.85)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output / f"figure4a_topic_composition.{suffix}", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True,
                        help="Released topic_composition directory containing data/ and reference/")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New output directory; an existing directory is never overwritten")
    parser.add_argument("--plot", action="store_true", help="Also plot the heatmap (Matplotlib required)")
    args = parser.parse_args()
    dataset = args.dataset_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    required = [dataset / "data" / name for name in (
        "windows.csv", "topic_instances.csv", "raw_topic_labels.csv",
        "topic_assignments.csv", "topic_taxonomy.csv")]
    reference = dataset / "reference" / "figure4a_composition.csv"
    missing = [str(path) for path in required + [reference] if not path.is_file()]
    if missing:
        parser.error("Missing dataset files:\n" + "\n".join(missing))
    if args.output_dir.exists():
        parser.error(f"Output directory already exists: {args.output_dir}; choose a new directory")
    rows, windows, classified, counts = aggregate(dataset / "data")
    require(counts == {"windows": 154, "topic_bearing_windows": 148, "empty_windows": 6,
                       "topic_instances": 463, "raw_labels": 441, "topic_categories": 8},
            "Unexpected frozen dataset counts")
    difference = check_reference(rows, read_csv(reference))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "figure4a_composition.csv", rows)
    write_csv(args.output_dir / "window_topic_composition.csv", windows)
    write_csv(args.output_dir / "classified_topic_instances.csv", classified)
    if args.plot:
        plot(rows, args.output_dir)
    validation = {"status": "passed", **counts, "reference_groups_checked": len(rows),
                  "max_absolute_difference_at_six_decimal_share_precision": difference}
    (args.output_dir / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation, indent=2))
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
