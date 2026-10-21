# Topic composition: Figure 4A code MVP

This code package reproduces the topic-composition analysis in Figure 4A of the
COSILab dataset paper. It uses the frozen open-coded topics and Qwen3-14B category
assignments distributed in the dataset's `topic_composition` package. The code
normalizes estimated shares, joins raw labels to the eight topic families,
aggregates by session and group size, checks the paper reference table, and
optionally plots the heatmap.

## Data and code locations

The dataset release provides the inputs at:

```text
4TU_upload/processed/benchmark_tasks/topic_composition/
├── data/
│   ├── windows.csv
│   ├── topic_instances.csv
│   ├── raw_topic_labels.csv
│   ├── topic_assignments.csv
│   └── topic_taxonomy.csv
└── reference/
    └── figure4a_composition.csv
```

Pass that `topic_composition` directory using `--dataset-dir`; it can be stored
anywhere on your machine. Dataset rows, transcripts, generated figures, inference
logs, model weights, and the paper PDF are distributed or maintained separately.
The original full research workspace is `COSILab_topic_transition_sync_analysis`.
This folder is its portable code contribution to the COSILab repository.

## Run

Python 3.9 or newer is sufficient for table reproduction; it uses only the
standard library. From the COSILab repository root:

```bash
python3 analysis/topic_composition/reproduce.py \
  --dataset-dir /path/to/4TU_upload/processed/benchmark_tasks/topic_composition \
  --output-dir /path/to/new/topic_composition_output
```

Choose a new output directory. Input files are read only, and existing output
directories are never overwritten. The command also works from another working
directory when the script and dataset paths are supplied explicitly.

For a PNG, SVG, and PDF heatmap, install the optional plotting dependency in your
Python environment and add `--plot`:

```bash
python3 -m pip install -r analysis/topic_composition/requirements-plot.txt
python3 analysis/topic_composition/reproduce.py \
  --dataset-dir /path/to/4TU_upload/processed/benchmark_tasks/topic_composition \
  --output-dir /path/to/new/topic_composition_plot_output \
  --plot
```

The output contains:

| File | Content |
| --- | --- |
| `figure4a_composition.csv` | Six session/group-size distributions over T1–T8 |
| `window_topic_composition.csv` | Composition vectors for 148 topic-bearing windows |
| `classified_topic_instances.csv` | 463 instances joined to their figure categories |
| `validation.json` | Dataset counts and comparison with all 48 reference shares |
| `figure4a_topic_composition.{png,svg,pdf}` | Heatmap when `--plot` is supplied |

## Analysis rules

The frozen analysis contains 154 windows, 148 of which have topics, with 463
topic instances and 441 distinct raw labels. The original windowing used 300
turns, stride 250, and a final tail of fewer than 100 turns merged into the
preceding window. The prompts used in open coding, inventory creation and
fixed-category assignment are retained in `prompts/` for methodological reference.
The open-coding user template preserves Python format escapes (`{{` and `}}`).

Estimated shares are normalized within each window and rounded to six decimal
places, following the original export. After mapping `raw_topic_id` to
`figure_topic_id`, shares are summed within each window and averaged with equal
weight across topic-bearing windows in each session/group-size category. Empty
windows are excluded, and overlapping windows are retained.

Group size uses the parent conversation's annotated **`participant_count`**.
`visible_speaker_count` must not be used for this grouping. Also, the original
`topic_id` is a within-window open-code identifier; the final T1–T8 taxonomy is
identified by `figure_topic_id` in the assignment table.

This MVP performs deterministic reproduction from frozen model annotations.
It does not run LLM inference or recompute the paper's human-validation study.
The full inference workflow remains in the research workspace. The labels are
LLM-assisted descriptive estimates, rather than human ground truth.

The data definition was checked against the latest `COSI_Lab.pdf` supplied on
2026-10-21: Figure 4 on page 6, methods in Appendix E.2, and human validation in
Appendix E.4.1. Its SHA-256 is
`bbb668f593768bd2f9622220d4469a4287efd5eca47ce96604d6302c20648e61`.
The external dataset package's `provenance.json` records the detailed sources.
