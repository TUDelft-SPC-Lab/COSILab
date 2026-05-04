#!/usr/bin/env python

import os
import numpy as np
from scipy.io.wavfile import write
from pathlib import Path
import struct
from parser_utilities import parse_timestamps
from tqdm import tqdm

PDM_CRYSTAL_CLK = 32e6 # Frequency that feeds the PDM clock generator
PDM_TO_PCM_DIV = 64 # https://docs.nordicsemi.com/bundle/ps_nrf52832/page/pdm.html#d931e120

#PDM clock variants taken from https://docs.nordicsemi.com/bundle/ps_nrf52832/page/pdm.html#d931e3773
#PDM_CLK_DIV = 30 # (1.067 MHz CLK variant)
#PDM_CLK_DIV = 31 # (1.032 MHz CLK variant)
#PDM_CLK_DIV = 32 # (1 MHz CLK variant)
PDM_CLK_DIV = 25 # (1.280 MHz CLK variant)

HIGH_SAMPLE_RATE = int((PDM_CRYSTAL_CLK / PDM_CLK_DIV) / PDM_TO_PCM_DIV)

# In LOW mode the midge will store one sample per DECIMATION samples
DECIMATION = 16
LOW_SAMPLE_RATE = int(HIGH_SAMPLE_RATE/DECIMATION)

def decode_audio_file(path_raw_input, input_dir=None, output_dir=None):
    """
    Decode a single audio file and save as WAV.
    
    Args:
        path_raw_input: Path to the raw audio file
    """
    if not (path_raw_input.is_file() and path_raw_input.suffix == "" and
            ("MICLO" in path_raw_input.stem or "MICHI" in path_raw_input.stem)):
        return
    
    print("Audio input file " + str(path_raw_input))
    if input_dir and output_dir:
        rel_path = path_raw_input.relative_to(input_dir)
        out_file = output_dir / rel_path.parent / (path_raw_input.stem + ".wav")
        out_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_file = path_raw_input.parent / (path_raw_input.stem + ".wav")

    buffer_dtype = np.int16 # 16-bit PCM
    if path_raw_input.stem[4:6] == "LO":
        sample_rate = LOW_SAMPLE_RATE  # Low frequency sampling
    elif path_raw_input.stem[4:6] == "HI":
        sample_rate = HIGH_SAMPLE_RATE  # High frequency sampling
    else:
        raise RuntimeError("Unknown number of channels")

    if path_raw_input.stem[0] == "0":
        num_channels = 2  # Stereo
    elif path_raw_input.stem[0] == "1":
        num_channels = 1  # Mono
    else:
        raise RuntimeError("Unknown number of channels")

    with path_raw_input.open("rb") as f:
        raw_data = f.read()
    
    audio_data = np.frombuffer(raw_data, dtype=buffer_dtype)

    if num_channels == 2:
        audio_data = audio_data.reshape(-1, 2)

    # Save the data in a WAV file
    write(filename=str(out_file), rate=sample_rate, data=audio_data)

def read_int64_little_endian(file_path):
    results = []
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(8)
            if len(chunk) < 8:
                break  # End of file
            value = struct.unpack('<q', chunk)[0]  # '<q' = little-endian signed int64
            results.append(value)
    return results

def decode_timestamp_file(path_input, input_dir=None, output_dir=None):
    """
    Decode a single timestamp file (.D file) and save results as -ts.csv files with datetime format.
    """
    if not (path_input.is_file() and path_input.suffix == ".D"):
        return
        
    print("Timestamp input file " + str(path_input))
    if input_dir and output_dir:
        rel_path = path_input.relative_to(input_dir)
        out_file = output_dir / rel_path.parent / (path_input.stem + "-ts.csv")
        out_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_file = path_input.parent / (path_input.stem + "-ts.csv")
    try:
        timestamp_data = read_int64_little_endian(path_input)
        # Convert timestamps to datetime format
        datetime_data = parse_timestamps(timestamp_data, str(path_input))
        
        # Save the datetime data as a CSV file with index and time columns
        with out_file.open("w") as f:
            f.write(",time\n")
            for i, datetime_val in enumerate(datetime_data):
                f.write("{},{}\n".format(i, datetime_val))
        
        print("Successfully decoded {} timestamps to {}".format(len(datetime_data), out_file))
    except Exception as e:
        print("Error processing {}: {}".format(path_input, e))

def process_input_file(input_path, input_dir=None, output_dir=None):
    # Process audio files
    if input_path.is_file() and input_path.suffix == "" and ("MICLO" in input_path.stem or "MICHI" in input_path.stem):
        decode_audio_file(input_path, input_dir, output_dir)
    # Process timestamp files
    elif input_path.is_file() and input_path.suffix == ".D":
        decode_timestamp_file(input_path, input_dir, output_dir)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Parser for the audio data and timestamps obtained from Mingle Midges')
    parser.add_argument('--fn', required=True, help='Please enter the path to the folder')
    parser.add_argument('--output_dir', required=False, help='Optional output directory (will mirror input structure)')
    args = parser.parse_args()

    input_dir = Path(args.fn)
    output_dir = Path(args.output_dir) if args.output_dir else None
    for root_str, _, files in tqdm(list(os.walk(input_dir)), desc="Files"):
        root = Path(root_str)
        for filename in files:
            input_path = root / filename
            process_input_file(input_path, input_dir=input_dir if output_dir else None, output_dir=output_dir)
