import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime, date, time, timedelta, timezone
from scipy.io import wavfile
from ffmpeg import FFmpeg
import subprocess
from scipy import signal

# AUDIO_FOLDER = "E:\\data_temp\\outputs"
AUDIO_FOLDER = "E:\\转移\\outputs"
TIME_SEGS = [
    {"start": "13461000_13462000", "end": "14173000_14174000"}, 
    {"start": "14541000_14542000", "end": "15260000_15261000"}
]

def normalize_audio(data):
    """Normalize audio data to have maximum absolute value of 1.0"""
    return data / np.max(np.abs(data))

def extract_audio(video_path, output_wav, show_sample_rate=True):
    """Extract audio from video as WAV using ffmpeg without resampling."""
    # Get original sample rate using ffprobe
    if show_sample_rate:
        probe_cmd = [
            'ffprobe', '-v', 'error', 
            '-select_streams', 'a:0',
            '-show_entries', 'stream=sample_rate',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        sample_rate = result.stdout.strip()
        print(f"Original sample rate: {sample_rate} Hz")
    
    # Extract audio without resampling (removed -ar option)
    cmd = ['ffmpeg', '-y', '-i', video_path, '-vn', '-ac', '1', output_wav]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def downsample_audio(data, original_rate, target_rate=4000):
    """
    Downsample audio data to target sample rate.
    Args:
        data: Audio data array
        original_rate: Original sample rate
        target_rate: Target sample rate (default: 4000 Hz)
    Returns:
        Downsampled data and timestamps
    """
    # Calculate the number of samples in the downsampled data
    num_samples = int(len(data) * target_rate / original_rate)
    
    # Use scipy's resample function
    downsampled_data = signal.resample(data, num_samples)
    
    return downsampled_data

def check_audio_sync():
    signal_type = "end"
    camera_id = "06"
    use_channel = 12
    use_downsample = False
    # check_all = False
    use_normalize = False
    if int(camera_id) <= 5:
        time_seg_name = TIME_SEGS[1][signal_type]
    else:
        time_seg_name = TIME_SEGS[0][signal_type]

    video_path = Path(AUDIO_FOLDER, f"camera_{camera_id}_{time_seg_name}.mp4")
    output_wav = Path(AUDIO_FOLDER, f"camera_{camera_id}_{time_seg_name}.wav")
    extract_audio(video_path, output_wav)

    # # plot audio from video and microphone and overlay
    # vid_audio_path = Path(AUDIO_FOLDER, f"camera_{camera_id}_{time_seg_name}.wav")
    # mic_audio_path = Path(AUDIO_FOLDER, f"seg_{time_seg_name}.wav")
    # vid_sample_rate, vid_audio_data = wavfile.read(vid_audio_path)
    # mic_sample_rate, mic_audio_data = wavfile.read(mic_audio_path)

    # start_sec = 0
    # end_sec = 10

    # vid_audio_data = vid_audio_data[int(start_sec * vid_sample_rate): int(end_sec * vid_sample_rate)]
    # mic_audio_data = mic_audio_data[int(start_sec * mic_sample_rate): int(end_sec * mic_sample_rate), :]

    # target_sample_rate = 20000 if use_downsample else vid_sample_rate
    # vid_audio_data = downsample_audio(vid_audio_data, vid_sample_rate, target_sample_rate)
    # if use_normalize:
    #     vid_audio_data = normalize_audio(vid_audio_data)
    
    # check all channels
    # for channel in range(1, mic_audio_data.shape[1]):   
    #     mic_audio_data_channel = downsample_audio(mic_audio_data[:, channel], mic_sample_rate, target_sample_rate)
    #     if use_normalize:
    #         mic_audio_data_channel = normalize_audio(mic_audio_data_channel)
    #     corr = signal.correlate(vid_audio_data, mic_audio_data_channel, mode='full')
    #     lag = np.argmax(corr) - len(vid_audio_data) + 1
    #     print(f"channel {channel}, Lag: {lag} samples, {1000 * lag / target_sample_rate} ms")

    # check single channel
    # mic_audio_data_channel = downsample_audio(mic_audio_data[:, use_channel], mic_sample_rate, target_sample_rate)
    # if use_normalize:
    #     mic_audio_data_channel = normalize_audio(mic_audio_data_channel)
    # corr = signal.correlate(vid_audio_data, mic_audio_data_channel, mode='full')
    # lag = np.argmax(corr) - len(vid_audio_data) + 1
    # time = np.arange(len(vid_audio_data)) / target_sample_rate
    # print(f"channel {use_channel}, Lag: {lag} samples, {1000 * lag / target_sample_rate} ms")
    # plt.plot(time, vid_audio_data)
    # plt.plot(time, mic_audio_data_channel)
    # plt.show()
    # plt.savefig(Path(AUDIO_FOLDER, f"audio_overlay_{use_channel}.png"))
    # plt.close()

    # plot from all cameras
    camera_ids = ["06", "09", "10", "12", "13"]
    fig, axs = plt.subplots(len(camera_ids), 1, figsize=(10, 10), sharex=True)
    for i, camera_id in enumerate(camera_ids):
        video_path = Path(AUDIO_FOLDER, f"camera_{camera_id}_{time_seg_name}.mp4")
        vid_audio_path = Path(AUDIO_FOLDER, f"camera_{camera_id}_{time_seg_name}.wav")
        extract_audio(video_path, vid_audio_path)
        vid_sample_rate, vid_audio_data = wavfile.read(vid_audio_path)
        time = np.arange(len(vid_audio_data)) / vid_sample_rate
        axs[i].plot(time, vid_audio_data, label=f"Camera {camera_id}")
        axs[i].legend()
    plt.show()
    plt.savefig(Path(AUDIO_FOLDER, f"audio_overlay_{camera_ids}.png"))
    plt.close()   

def check_video_audio_sync():
    signal_type = "end"
    time_segs = [
        {"start": "13461000_13462000", "end": "14173000_14174000"}, 
        {"start": "14541000_14542000", "end": "15260000_15261000"}
    ]
    camera_1 = "02"
    camera_2 = "04"
 
    use_normalize = False
    if int(camera_1) <= 5:
        time_seg_name = time_segs[0][signal_type]
    else:
        time_seg_name = time_segs[1][signal_type]
    # plot audio from video and microphone and overlay
    vid_1_audio_path = Path(AUDIO_FOLDER, f"camera_{camera_1}_{time_seg_name}.wav")
    vid_2_audio_path = Path(AUDIO_FOLDER, f"camera_{camera_2}_{time_seg_name}.wav")
    vid_1_sample_rate, vid_1_audio_data = wavfile.read(vid_1_audio_path)
    vid_2_sample_rate, vid_2_audio_data = wavfile.read(vid_2_audio_path)
 
    start_sec = 0.5
    end_sec = 1.5
    vid_1_audio_data = vid_1_audio_data[int(start_sec * vid_1_sample_rate): int(end_sec * vid_1_sample_rate)]
    vid_2_audio_data = vid_2_audio_data[int(start_sec * vid_2_sample_rate): int(end_sec * vid_2_sample_rate)]
 
    target_sample_rate = 20000
    vid_1_audio_data = downsample_audio(vid_1_audio_data, vid_1_sample_rate, target_sample_rate)
    vid_2_audio_data = downsample_audio(vid_2_audio_data, vid_2_sample_rate, target_sample_rate)
    if use_normalize:
        vid_1_audio_data = normalize_audio(vid_1_audio_data)
        vid_2_audio_data = normalize_audio(vid_2_audio_data)
    corr = signal.correlate(vid_1_audio_data, vid_2_audio_data, mode='full')
    lag = np.argmax(corr) - len(vid_1_audio_data) + 1
    time = np.arange(len(vid_1_audio_data)) / target_sample_rate
    print(f"Camera {camera_1} and {camera_2} sync: {lag} samples, {1000 * lag / target_sample_rate} ms")
 
    # plot audio from video and microphone and overlay
    plt.plot(time, vid_1_audio_data, label=camera_1)
    plt.plot(time, vid_2_audio_data, label=camera_2)
    plt.legend()
    plt.show()
    plt.savefig(Path(AUDIO_FOLDER, f"video_audio_overlay_{camera_1}_{camera_2}.png"))
    plt.close()

def check_mic_audio_sync():
    signal_type = "start"
    time_seg_1 = {"start": "13461000_13462000", "end": "14173000_14174000"}
    time_seg_2 = {"start": "14541000_14542000", "end": "15260000_15261000"}
    time_seg = time_seg_1
    time_seg_name = time_seg[signal_type]
    use_normalize = False

    mic_audio_path = Path(AUDIO_FOLDER, f"seg_{time_seg_name}.wav")
    mic_sample_rate, mic_audio_data = wavfile.read(mic_audio_path)

    start_sec = 3.7
    end_sec = 4.5

    mic_audio_data = mic_audio_data[int(start_sec * mic_sample_rate): int(end_sec * mic_sample_rate), :]

    target_sample_rate = 20000

    lag_list = []
    lag_ms_heatmap = np.zeros((mic_audio_data.shape[1], mic_audio_data.shape[1]))
    # check all channels
    for channel_1 in range(1, mic_audio_data.shape[1]):   
        for channel_2 in range(channel_1 + 1, mic_audio_data.shape[1]):
            mic_audio_data_channel_1 = downsample_audio(mic_audio_data[:, channel_1], mic_sample_rate, target_sample_rate)
            mic_audio_data_channel_2 = downsample_audio(mic_audio_data[:, channel_2], mic_sample_rate, target_sample_rate)
            if use_normalize:
                mic_audio_data_channel_1 = normalize_audio(mic_audio_data_channel_1)
                mic_audio_data_channel_2 = normalize_audio(mic_audio_data_channel_2)
            corr = signal.correlate(mic_audio_data_channel_1, mic_audio_data_channel_2, mode='full')
            lag = np.argmax(corr) - len(mic_audio_data_channel_1) + 1
            lag_ms = 1000 * lag / target_sample_rate
            # print(f"channel {channel_1} and {channel_2}, Lag: {lag} samples, {1000 * lag / target_sample_rate} ms")
            lag_list.append(abs(lag_ms))
            lag_ms_heatmap[channel_1, channel_2] = abs(lag_ms) if abs(lag_ms) > 40 else 0
            lag_ms_heatmap[channel_2, channel_1] = abs(lag_ms) if abs(lag_ms) > 40 else 0
    # print(f"Average lag: {np.mean(lag_list)} ms")
    plt.imshow(lag_ms_heatmap, cmap='viridis')
    plt.colorbar()
    plt.show()
    plt.savefig(Path(AUDIO_FOLDER, "mic_audio_sync_heatmap.png"))
    plt.close()

    # check two specific channels
    # channel_1 = 3
    # channel_2 = 18
    # mic_audio_data_channel_1 = downsample_audio(mic_audio_data[:, channel_1], mic_sample_rate, target_sample_rate)
    # mic_audio_data_channel_2 = downsample_audio(mic_audio_data[:, channel_2], mic_sample_rate, target_sample_rate)
    # if use_normalize:
    #     mic_audio_data_channel_1 = normalize_audio(mic_audio_data_channel_1)
    #     mic_audio_data_channel_2 = normalize_audio(mic_audio_data_channel_2)
    # corr = signal.correlate(mic_audio_data_channel_1, mic_audio_data_channel_2, mode='full')
    # lag = np.argmax(corr) - len(mic_audio_data_channel_1) + 1
    # lag_ms = 1000 * lag / target_sample_rate
    # print(f"Channel {channel_1} and {channel_2} sync: {lag} samples, {1000 * lag / target_sample_rate} ms")

    # time = np.arange(len(mic_audio_data_channel_1)) / target_sample_rate
    # plt.plot(time, mic_audio_data_channel_1, label=channel_1)
    # plt.plot(time, mic_audio_data_channel_2, label=channel_2)
    # plt.legend()
    # plt.show()
    # plt.savefig(f"E:\\data_temp\\outputs\\mic_audio_sync_{channel_1}_{channel_2}.png")
    # plt.close()
    # return lag_ms


def plot_channel_waves(session, channels, signal_type, use_normalize=False):
    time_seg_name = TIME_SEGS[session - 1][signal_type]
    mic_audio_path = Path(AUDIO_FOLDER, f"seg_{time_seg_name}.wav")
    mic_sample_rate, mic_audio_data = wavfile.read(mic_audio_path)
    fig, axs = plt.subplots(len(channels), 1, figsize=(10, 10), sharex=True)
    for i, channel in enumerate(channels):
        mic_audio_data_channel = mic_audio_data[:, channel]
        if use_normalize:
            mic_audio_data_channel = normalize_audio(mic_audio_data_channel)
        # save single channel audio
        # wavfile.write(Path(AUDIO_FOLDER,"single_channel_audio", f"audio_{channel}_{time_seg_name}.wav"), mic_sample_rate, mic_audio_data_channel)
        # subplot within one figure
        mic_time = np.arange(len(mic_audio_data_channel)) / mic_sample_rate
        axs[i].plot(mic_time, mic_audio_data_channel, label=f"Channel {channel}")
        axs[i].legend()
    plt.savefig(Path(AUDIO_FOLDER, f"audio_overlay_{session}_{signal_type}.png"))

def plot_camera_waves(session, camera_ids, signal_type, use_normalize=False):
    time_seg_name = TIME_SEGS[session - 1][signal_type]
    fig, axs = plt.subplots(len(camera_ids), 1, figsize=(10, 10), sharex=True)
    for i, camera_id in enumerate(camera_ids):
        vid_audio_path = Path(AUDIO_FOLDER, f"camera_{camera_id}_{time_seg_name}.wav")
        # if not vid_audio_path.exists():
        video_path = Path(AUDIO_FOLDER, f"camera_{camera_id}_{time_seg_name}.mp4")
        extract_audio(video_path, vid_audio_path)
        vid_sample_rate, vid_audio_data = wavfile.read(vid_audio_path)
        vid_time = np.arange(len(vid_audio_data)) / vid_sample_rate
        axs[i].plot(vid_time, vid_audio_data, label=f"Camera {camera_id}")
        axs[i].legend()
    plt.savefig(Path(AUDIO_FOLDER, f"camera_audio_overlay_{session}_{signal_type}.png"))

if __name__ == "__main__":
    # check_mic_audio_sync()
    # check_video_audio_sync()
    # check_audio_sync()
    session = 1
    signal_type = "end"
    camera_ids = ["06", "09", "10", "12", "13", "14", "15"]
    channels = [1, 5, 10, 15, 20, 25, 30]
    use_normalize = False
    plot_channel_waves(session, channels, signal_type, use_normalize)
    plot_camera_waves(session, camera_ids, signal_type, use_normalize)
    plt.show()