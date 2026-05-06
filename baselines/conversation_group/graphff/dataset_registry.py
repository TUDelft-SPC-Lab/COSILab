import os
from pathlib import Path


DEFAULT_MINGLING_FRAME_STRIDE = 20

MINGLING_CAMERAS = {
	"mingling1": ("cam06", "cam08", "cam10"),
	"mingling2": ("cam01", "cam03"),
}


def is_mingling_dataset(dataset_path):
	dataset_root = dataset_path.split("/", 1)[0]
	return dataset_root in MINGLING_CAMERAS


def get_dataset_data_dir(dataset_path):
	return str(Path("data") / dataset_path)


def get_dataset_label(dataset_path):
	return dataset_path.replace("/", "_")


def get_dataset_frame_stride(dataset_path):
	value = os.environ.get("GRAPHFF_FRAME_STRIDE")
	if value is not None:
		return int(value)
	if is_mingling_dataset(dataset_path):
		return DEFAULT_MINGLING_FRAME_STRIDE
	return 1


def get_dataset_config(dataset_path):
	if is_mingling_dataset(dataset_path):
		dataset_root = dataset_path.split("/", 1)[0]
		return {
			"num_nodes": 32,
			"num_of_actual_people": 32,
			"feature_size": 7,
			"source_root": dataset_root,
			"camera_names": MINGLING_CAMERAS[dataset_root],
			"frame_stride": DEFAULT_MINGLING_FRAME_STRIDE,
		}

	raise ValueError("Unknown dataset: " + dataset_path)
