"""Data-quality watchers for the LSTM/GraphFF pipeline.

Two known structural issues are counted here rather than assumed away, so each
run reports whether they actually occur in its data:

1. ``indicator_gap_report`` -- neighbour visibility gaps inside a window.
   Skynet masks the LSTM *hidden* state of an absent neighbour but not its
   *cell* state, and the cell was just updated with the -999 placeholder input.
   A neighbour that disappears and comes back therefore carries contaminated
   memory into the frames after it reappears. A neighbour that simply drops out
   and never returns is harmless: its output stays masked to zero for the rest
   of the window.

2. ``unfilled_row_report`` -- rows of the scene affinity matrix that no sample
   ever wrote. ``condense_to_group_mat`` initialises the matrix with ones and
   fills one row per (scene, self_person) sample. A sample requires the person
   to be visible across the *whole* window, but scoring selects people visible
   at the *single* evaluation frame. Anyone in the gap keeps an all-ones row,
   which reads as "grouped with everyone" and, after the ``(A + A.T)/2``
   symmetrisation, biases that scene toward false-positive merges.

Both are cheap: one vectorised pass over the indicator channel, and one pass
over the scenes.
"""

import numpy as np
import pandas as pd
import torch


def indicator_gap_report(data, feature_size):
	"""Count neighbour visibility gaps in a data tensor.

	data: (num_samples, seq_len, feature_size, num_neighbors)
	The last feature channel is the validity indicator.

	Returns (summary_dict, per_sample_dataframe).
	"""
	if data.shape[0] == 0:
		return {
			'n_samples': 0, 'n_pairs': 0, 'pairs_with_gap': 0, 'gap_frames': 0,
			'samples_with_gap': 0, 'pairs_always_present': 0,
			'pairs_never_present': 0, 'pairs_dropout_only': 0,
			'max_gap_len': 0, 'pct_pairs_with_gap': float('nan'),
			'pct_samples_with_gap': float('nan'),
		}, pd.DataFrame(columns=['sample_idx', 'gapped_neighbors', 'gap_frames'])

	present = (data[:, :, feature_size - 1, :] > 0.5)          # (N, T, K)
	num_samples, seq_len, num_neighbors = present.shape

	as_int = present.to(torch.int8)
	seen_before, _ = torch.cummax(as_int, dim=1)               # any 1 at or before t
	seen_after, _ = torch.cummax(torch.flip(as_int, dims=[1]), dim=1)
	seen_after = torch.flip(seen_after, dims=[1])              # any 1 at or after t

	# absent at t, yet present strictly earlier and strictly later -> reappearance
	gap = (~present) & (seen_before == 1) & (seen_after == 1)

	gap_per_pair = gap.sum(dim=1)                              # (N, K)
	pair_has_gap = gap_per_pair > 0
	always = present.all(dim=1)
	never = (~present).all(dim=1)
	# dropped out and never came back (or arrived late) but no sandwiched gap
	dropout_only = (~always) & (~never) & (~pair_has_gap)

	# longest run of consecutive gap frames, for a sense of how deep the gaps are
	max_gap_len = 0
	if bool(pair_has_gap.any()):
		run = torch.zeros_like(gap_per_pair)
		best = torch.zeros_like(gap_per_pair)
		for t in range(seq_len):
			run = torch.where(gap[:, t, :], run + 1, torch.zeros_like(run))
			best = torch.maximum(best, run)
		max_gap_len = int(best.max())

	sample_gapped = pair_has_gap.sum(dim=1)
	rows = torch.nonzero(sample_gapped > 0).flatten()
	per_sample = pd.DataFrame({
		'sample_idx': rows.cpu().numpy(),
		'gapped_neighbors': sample_gapped[rows].cpu().numpy(),
		'gap_frames': gap_per_pair[rows].sum(dim=1).cpu().numpy(),
	})

	n_pairs = num_samples * num_neighbors
	summary = {
		'n_samples': int(num_samples),
		'n_pairs': int(n_pairs),
		'pairs_with_gap': int(pair_has_gap.sum()),
		'gap_frames': int(gap.sum()),
		'samples_with_gap': int((sample_gapped > 0).sum()),
		'pairs_always_present': int(always.sum()),
		'pairs_never_present': int(never.sum()),
		'pairs_dropout_only': int(dropout_only.sum()),
		'max_gap_len': max_gap_len,
	}
	summary['pct_pairs_with_gap'] = 100.0 * summary['pairs_with_gap'] / n_pairs
	summary['pct_samples_with_gap'] = 100.0 * summary['samples_with_gap'] / num_samples
	return summary, per_sample


def unfilled_row_report(scene_seq_mat_dict, gt_matrices, num_nodes):
	"""Find scene rows that no sample wrote, among people scoring counts as visible.

	condense_to_group_mat seeds the matrix with ones, so an untouched row is
	exactly all ones. A filled row cannot be: its off-diagonal entries are
	sigmoid outputs, which are masked to 0 when invalid and never reach 1.0.

	Returns (summary_dict, per_scene_dataframe).
	"""
	records = []
	for scenes in scene_seq_mat_dict.keys():
		scene_idx = scenes[2] if len(scenes) >= 3 else scenes[1] - 1
		mat = scene_seq_mat_dict[scenes][-1].detach().cpu().numpy()
		gt_matrix = gt_matrices[scene_idx]
		valid_idx = np.where(np.diag(gt_matrix) == 1)[0]
		if len(valid_idx) == 0:
			continue

		unfilled = [int(p) for p in valid_idx if np.all(mat[p, :] == 1.0)]
		records.append({
			'scene_idx': int(scene_idx),
			'n_visible_at_eval_frame': int(len(valid_idx)),
			'n_filled_rows': int(len(valid_idx) - len(unfilled)),
			'n_unfilled_rows': int(len(unfilled)),
			'unfilled_person_ids': ' '.join(str(p + 1) for p in unfilled),
		})

	per_scene = pd.DataFrame.from_records(records)
	if len(per_scene) == 0:
		summary = {
			'n_scenes': 0, 'scenes_with_unfilled': 0, 'total_unfilled_rows': 0,
			'total_visible_rows': 0, 'pct_scenes_affected': float('nan'),
			'pct_rows_unfilled': float('nan'), 'worst_scene_unfilled': 0,
		}
		return summary, per_scene

	total_visible = int(per_scene['n_visible_at_eval_frame'].sum())
	total_unfilled = int(per_scene['n_unfilled_rows'].sum())
	affected = per_scene[per_scene['n_unfilled_rows'] > 0]
	summary = {
		'n_scenes': int(len(per_scene)),
		'scenes_with_unfilled': int(len(affected)),
		'total_unfilled_rows': total_unfilled,
		'total_visible_rows': total_visible,
		'pct_scenes_affected': 100.0 * len(affected) / len(per_scene),
		'pct_rows_unfilled': 100.0 * total_unfilled / total_visible if total_visible else float('nan'),
		'worst_scene_unfilled': int(per_scene['n_unfilled_rows'].max()),
	}
	# only the affected scenes are worth writing out
	return summary, affected.reset_index(drop=True)


def format_summary(title, summary):
	body = ', '.join(
		'{}={}'.format(k, round(v, 3) if isinstance(v, float) else v)
		for k, v in summary.items()
	)
	return '[watch] ' + title + ': ' + body
