import pandas as pd
import numpy as np
from scipy.io import wavfile
import glob
import os
from datetime import datetime, timedelta

SAMPLES_PER_CSV_ENTRY = 1024
TIME_SEGS = [
    {"start": ["13:46:10", "13:46:20"], "end": ["14:17:30", "14:17:40"]}, 
    {"start": ["14:54:10", "14:54:20"], "end": ["15:26:00", "15:26:10"]}
]

def str_to_timedelta(time_str):
    hours, minutes, seconds = map(int, time_str.split(":"))
    return timedelta(hours=hours, minutes=minutes, seconds=seconds)

def extract_single(wav_path, csv_path, start_time, end_time, csv_time_is_chunk_end=False):
    """
    Extract audio from a single wav file using timestamps from a csv file.

    Args:
        wav_path: Path to the wav file.
        csv_path: Path to the csv file containing a 'time' column with timestamps.
        start_time: Start time of the audio to extract (timedelta, represents time-of-day).
        end_time: End time of the audio to extract (timedelta, represents time-of-day).
        csv_time_is_chunk_end: If True, treat CSV timestamps as the end time of each chunk.
                               If False (default), treat as the start time of each chunk.
    
    Returns:
        Tuple of (extracted_wav_data, sample_rate)
    """
    wav_sample_rate, wav_data = wavfile.read(wav_path)
    csv_data = pd.read_csv(csv_path)
    
    # Parse the 'time' column to datetime (handles both with and without milliseconds)
    # format='mixed' allows parsing entries like "2025-07-17 12:11:50" and "2025-07-17 12:11:50.681000"
    csv_data['time'] = pd.to_datetime(csv_data['time'], format='mixed')
    
    # Get the recording start time from CSV (first timestamp)
    recording_start = csv_data['time'].iloc[0]
    recording_end = csv_data['time'].iloc[-1]
    
    # Calculate recording duration from CSV timestamps
    recording_duration = (recording_end - recording_start).total_seconds()
    print(f"  Recording start: {recording_start}")
    print(f"  Recording end: {recording_end}")
    print(f"  Recording duration: {recording_duration:.2f}s")
    print(f"  CSV entries: {len(csv_data)}, WAV samples: {len(wav_data)}, Sample rate: {wav_sample_rate} Hz")
    print(f"  Expected WAV samples from CSV: {len(csv_data) * SAMPLES_PER_CSV_ENTRY}")
    print(f"  CSV time represents: {'end' if csv_time_is_chunk_end else 'start'} of each chunk")
    
    # Convert timestamps to seconds relative to recording start (avoids unit mismatch issues)
    csv_timestamps_sec = (csv_data['time'] - recording_start).dt.total_seconds().values
    
    # Create sample indices for CSV entries (each entry corresponds to SAMPLES_PER_CSV_ENTRY samples)
    # If csv_time_is_chunk_end=True, timestamp corresponds to end of chunk (sample index + 1023)
    # If csv_time_is_chunk_end=False, timestamp corresponds to start of chunk (sample index 0)
    if csv_time_is_chunk_end:
        # CSV row i timestamp is at sample (i+1) * SAMPLES_PER_CSV_ENTRY - 1
        csv_sample_indices = (np.arange(len(csv_data)) + 1) * SAMPLES_PER_CSV_ENTRY - 1
    else:
        # CSV row i timestamp is at sample i * SAMPLES_PER_CSV_ENTRY
        csv_sample_indices = np.arange(len(csv_data)) * SAMPLES_PER_CSV_ENTRY
    
    # Create sample indices for all WAV samples
    wav_sample_indices = np.arange(len(wav_data))
    
    # Linear interpolation to get timestamps (in seconds from recording start) for each WAV sample
    interpolated_timestamps_sec = np.interp(wav_sample_indices, csv_sample_indices, csv_timestamps_sec)
    
    # Convert start_time and end_time (timedelta representing time-of-day) to datetime
    # Use the same date as the recording
    recording_date = recording_start.date()
    target_start = datetime.combine(recording_date, datetime.min.time()) + start_time
    target_end = datetime.combine(recording_date, datetime.min.time()) + end_time
    
    print(f"  Target start: {target_start}")
    print(f"  Target end: {target_end}")
    
    # Convert target times to seconds relative to recording start
    target_start_sec = (target_start - recording_start).total_seconds()
    target_end_sec = (target_end - recording_start).total_seconds()
    
    # Find indices where interpolated time is between target_start and target_end
    mask = (interpolated_timestamps_sec >= target_start_sec) & (interpolated_timestamps_sec <= target_end_sec)
    selected_indices = np.where(mask)[0]
    
    if len(selected_indices) == 0:
        print(f"  Warning: No samples found in the specified time range")
        return wav_data[0:0], wav_sample_rate  # Return empty array
    
    start_index = selected_indices[0]
    end_index = selected_indices[-1] + 1  # +1 to include the last sample
    
    print(f"  Extracting samples {start_index} to {end_index} ({end_index - start_index} samples, {(end_index - start_index) / wav_sample_rate:.2f}s)")
    
    extracted_data = wav_data[start_index:end_index]
    return extracted_data, wav_sample_rate

def extract_folder_audo_detect(read_path, write_path, start_time, end_time, csv_time_is_chunk_end=False):
    device_number = read_path.split("\\")[-1]
    # auto-search for wav and csv files
    merged_wavs = sorted(glob.glob(os.path.join(read_path, "*.wav")))
    merged_csvs = sorted(glob.glob(os.path.join(read_path, "*.csv")))
    
    # os.makedirs(write_path, exist_ok=True)
    
    for merged_wav, merged_csv in zip(merged_wavs, merged_csvs):
        print(f"\nProcessing: {os.path.basename(merged_wav)}")
        wav_data, wav_sample_rate = extract_single(
            merged_wav, merged_csv, start_time, end_time, 
            csv_time_is_chunk_end=csv_time_is_chunk_end
        )
        # output_path = os.path.join(write_path, f"{device_number}_{os.path.basename(merged_wav)}")
        # wavfile.write(write_path, wav_sample_rate, wav_data)
        print(f"  Saved to: {write_path}")
        break  # process only one file for testing

if __name__ == "__main__":
    midge_list = [2, 3, 4, 12, 19, 24, 28, 65, 75, 81, 83, 90]
    session = 2
    signal_type = "end"
    time_seg = TIME_SEGS[session - 1][signal_type]
    for midge_idx in midge_list:
        start_time = time_seg[0]
        end_time = time_seg[1]
        start_str = ''.join(start_time.split(':'))
        end_str = ''.join(end_time.split(':'))
        read_path = f"F:\\datasets\\ingroup_extracted\\midge\\{midge_idx}"
        write_path = f"F:\\转移\\outputs\\session{session}_{signal_type}\\midge{midge_idx}_{start_str}_{end_str}.wav"
        
        csv_time_is_chunk_end = False  # Set to True if CSV timestamps represent end of each chunk
        
        start_time = str_to_timedelta(start_time)
        end_time = str_to_timedelta(end_time)
        extract_folder_audo_detect(read_path, write_path, start_time, end_time, csv_time_is_chunk_end)