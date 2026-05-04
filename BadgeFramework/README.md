# BadgeFramework

This directory contains tools for controlling Mingle/OpenBadge devices and parsing downloaded badge data.

Main files:

- `hub.py`: interactive hub for starting, stopping, erasing, checking, and synchronising multiple badges.
- `hub_utilities.py`: threaded helpers used by the hub.
- `hub_connection.py`: wrapper around BLE badge commands for one participant.
- `badge.py`: low-level OpenBadge protocol requests.
- `audio_parser.py`: parser for raw microphone data and audio timestamp files.
- `imu_parser.py`: parser for IMU, rotation, and scan binary files.

## 1. Parsing Audio and IMU Binary Data

### Audio

`audio_parser.py` processes files with no extension whose names contain `MICLO` or `MICHI`.

Example command:

```bash
python audio_parser.py --fn /path/to/downloaded_badge_files --output_dir /path/to/decoded_audio
```

For every matching raw microphone file, the parser:

1. Reads the whole file as binary.
2. Interprets the buffer as 16-bit PCM samples with `numpy.frombuffer(..., dtype=np.int16)`.
3. Uses the filename to choose the sampling mode:
   - `MICHI`: high-rate audio.
   - `MICLO`: low-rate audio.
4. Uses the first filename character to choose the channel count:
   - `0...`: stereo, reshaped to `(n_samples, 2)`.
   - `1...`: mono.
5. Writes a `.wav` file next to the input, or into the mirrored `--output_dir`.

Sampling rates are derived from the nRF52832 PDM clock settings:

```text
PDM_CRYSTAL_CLK = 32 MHz
PDM_CLK_DIV = 25
PDM_TO_PCM_DIV = 64
HIGH_SAMPLE_RATE = (32e6 / 25) / 64 = 20000 Hz
LOW_SAMPLE_RATE = HIGH_SAMPLE_RATE / 16 = 1250 Hz
```

Audio timestamp files use the `.D` suffix. `decode_timestamp_file()` reads them as repeated 8-byte little-endian signed integers:

```text
int64 little-endian timestamp
```

The shared timestamp converter in `parser_utilities.py` interprets raw timestamp values as milliseconds since the Unix epoch and converts them with:

```python
datetime.fromtimestamp(raw_timestamp / 1000)
```

The timestamp output is written as:

```text
<input_stem>-ts.csv
```

with columns:

```text
index,time
```

### IMU, Rotation, and Scan Files

`imu_parser.py` recursively processes files with no extension whose names start with one of:

```text
ACC_
GYR_
MAG_
ROT_
SCAN_
```

Example command:

```bash
python imu_parser.py \
  --fn /path/to/downloaded_badge_files \
  --acc TRUE \
  --gyr TRUE \
  --mag TRUE \
  --rot TRUE \
  --scan TRUE \
  --plot TRUE \
  --output_dir /path/to/decoded_imu
```

For `ACC_`, `GYR_`, and `MAG_`, each binary record is parsed as 24 bytes:

```text
bytes 0..7    uint64 little-endian timestamp
bytes 8..19   float32 little-endian X, Y, Z
bytes 20..23  padding, skipped
```

The output dataframe contains:

```text
time,X,Y,Z
```

For `ROT_`, each record is also 24 bytes:

```text
bytes 0..7    uint64 little-endian timestamp
bytes 8..23   float32 little-endian q1, q2, q3, q4
```

The output dataframe contains:

```text
time,X,Y,Z,W
```

For `SCAN_`, each record is parsed as 16 bytes:

```text
bytes 0..7    uint64 little-endian timestamp
bytes 8..9    uint16 little-endian sensor_id
byte 10       int8 group
byte 11       int8 rssi
bytes 12..15  unused by the parser
```

The output dataframe contains:

```text
time,SensorID,RSSI,Group
```

For every parsed sensor file, `imu_parser.py` can write both:

```text
<input_stem>.csv
<input_stem>.pkl
```

When `--plot TRUE`, it also writes simple time-series plots for accelerometer, gyroscope, magnetometer, and rotation files.

## 2. Synchronisation in `hub.py`

`hub.py` is an interactive multi-badge controller. It reads badge participant IDs and MAC addresses from:

```text
ingroup_exp.csv
```

and adds runtime state columns:

```text
Recording
Id_Set
```

The default sync interval is:

```python
sync_frequency = 5 * 60
```

so the hub attempts synchronisation every 5 minutes when sync is enabled.

### Why an Empty Command Triggers Sync

The hub wraps terminal input with:

```python
timeout_input(timeout=sync_frequency, prompt='> ')
```

from `hub_utilities.py`. If the user types a command before the timeout, that command is handled normally. If no input arrives before `sync_frequency`, `timeout_input()` returns an empty string. In the main `hub.py` loop, an empty command means:

```python
if do_synchronization is True:
    synchronise_and_check_all_devices(df, show_status=show_status_on_sync)
```

So synchronisation is timer-driven by input timeout, not by a separate background scheduler.

### What Synchronisation Sends

`synchronise_and_check_all_devices()` connects to every badge and calls:

```python
cur_connection.handle_status_request()
```

That calls `OpenBadge.get_status()` in `badge.py`. `get_status()` creates a status request containing the current hub machine time:

```python
timestamp.seconds
timestamp.ms
```

The badge receives that timestamp through the status request. This is the operation used by the hub as the periodic clock sync/check step.

After each status response, the hub checks:

- IMU recording status.
- Microphone recording status.
- Scan recording status.
- Clock sync status.
- Battery level.

It prints errors for badges that are not recording, cannot sync, have connection failures, or have battery below 10%.

### Starting Recording With Sync

The `start_all` command calls:

```python
start_recording_all_devices_with_sync(df, sync_frequency)
```

This first starts all sensors on all badges via `start_recording_all_devices(df)`, then asks:

```text
Do you want to start synchronisation? [Y/n]:
```

If the answer is yes or empty, `do_synchronization` becomes `True`, and future input timeouts trigger `synchronise_and_check_all_devices()`.

When starting devices, `hub_utilities.start_recording_all_devices()` runs across badges in up to 4 threads. For each badge it:

1. Connects by participant ID and MAC address.
2. Sets the badge ID/group at start if needed.
3. Checks current status.
4. Starts scan, microphone, and IMU recording if they are not already active.
5. Marks `Recording=True` and `Id_Set=True` in the in-memory dataframe after the threaded operation finishes.

### Manual Sync Control

The hub supports these sync-related commands:

- `start_sync`: enable periodic synchronisation without starting recording.
- `stop_sync`: disable periodic synchronisation.
- `toggle_show_status`: toggle printing full status after each sync.
- `start_all`: start all sensors, then optionally enable sync.
- `stop_all`: stop all sensors and disable sync.
- `exit`: disable sync and optionally stop recording before exiting.

Inside single-badge management mode (`midge`), periodic sync still runs on timeout when enabled. The currently connected badge is skipped and reused through `skip_id` and `conn_skip_id`, so the hub does not open a second BLE connection to the same badge while it is being managed interactively.
