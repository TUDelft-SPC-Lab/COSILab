import os
from pathlib import Path
import re
import numpy as np
from scipy.io.wavfile import write

def merge_all_subfolders(parent_folder):
    """
    For each subfolder in parent_folder, merge .wav and -ts.csv files.
    """
    for root, dirs, files in os.walk(parent_folder):
        # Only process subfolders, not the parent itself
        if root == parent_folder:
            for d in dirs:
                subfolder = os.path.join(root, d)
                merge_wav_and_ts_files(subfolder)
            break
        # If you want to process recursively, remove the break and process every root
        # merge_wav_and_ts_files(root)

def merge_wav_and_ts_files(folder):
    """
    Merge all .wav and -ts.csv files in the given folder, grouped by prefix and sorted by trailing number.
    Output merged files to the same folder and delete originals.
    """
    folder = Path(folder)
    # Find all .wav and -ts.csv files
    wav_files = list(folder.glob("*MICHI*.wav")) + list(folder.glob("*MICLO*.wav"))
    ts_files = list(folder.glob("*MICHI*-ts.csv")) + list(folder.glob("*MICLO*-ts.csv"))

    def group_and_sort(files, pattern):
        groups = {}
        for f in files:
            m = re.match(pattern, f.name)
            if m:
                prefix = m.group(1)
                num = int(m.group(2))
                groups.setdefault(prefix, []).append((num, f))
        # Sort each group by the number
        for prefix in groups:
            groups[prefix].sort()
        return groups

    # Patterns: e.g., 0MICHI1.wav, 0MICHI99.wav, 0MICHI1-ts.csv
    wav_pattern = r"(\d+MICHI|\d+MICLO)(\d+)\.wav"
    ts_pattern = r"(\d+MICHI|\d+MICLO)(\d+)-ts\.csv"

    wav_groups = group_and_sort(wav_files, wav_pattern)
    ts_groups = group_and_sort(ts_files, ts_pattern)

    # Merge .wav files
    for prefix, files in wav_groups.items():
        merged_audio = []
        sample_rate = None
        num_channels = None
        for _, f in files:
            sr, data = None, None
            try:
                sr, data = read_wav(str(f))
            except Exception:
                from scipy.io.wavfile import read as read_wav
                sr, data = read_wav(str(f))
            if sample_rate is None:
                sample_rate = sr
            if num_channels is None:
                num_channels = data.shape[1] if data.ndim > 1 else 1
            merged_audio.append(data)
        merged_audio = np.concatenate(merged_audio, axis=0)
        merged_name = folder / (prefix + "_merged.wav")
        write(str(merged_name), sample_rate, merged_audio)
        # Delete originals
        for _, f in files:
            f.unlink()

    # Merge -ts.csv files
    for prefix, files in ts_groups.items():
        merged_rows = []
        idx = 0
        for _, f in files:
            with open(f, "r") as fin:
                lines = fin.readlines()
                if not merged_rows:
                    merged_rows.append(lines[0])  # header
                for line in lines[1:]:
                    parts = line.strip().split(",", 1)
                    if len(parts) == 2:
                        merged_rows.append("{},{}\n".format(idx, parts[1]))
                        idx += 1
        merged_name = folder / (prefix + "-ts_merged.csv")
        with open(merged_name, "w") as fout:
            fout.writelines(merged_rows)
        # Delete originals
        for _, f in files:
            f.unlink()

# Helper for reading wav files (for Python 2/3 compatibility)
def read_wav(filename):
    from scipy.io import wavfile
    return wavfile.read(filename)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Merge audio and timestamp files in all subfolders')
    parser.add_argument('--fn', required=True, help='Please enter the path to the parent folder')
    args = parser.parse_args()
    merge_all_subfolders(args.fn)