# Audio Process

This folder contains the code used for the pseudonymized transcript and audio redaction workflows described in the dataset paper.

It is intentionally a code-only bundle. It does not include internal cluster scripts, raw audio, generated outputs, model weights, logs, caches, or institution-specific paths.

## External Components

This bundle keeps only the project-specific orchestration and redaction code. It does not vendor external ASR or diarization repositories.

WhisperX is used for transcription and word-level timestamp alignment:
- in the audio PII redaction workflow, before mapping detected PII text spans back to audio time ranges
- in the mingling workflow, after `main_solo` audio has been rendered for each channel
- entrypoint in this bundle: `modules/transcription/run_whisperx_single.py`
- external repository: [WhisperX](https://github.com/m-bain/whisperX)

Faster Whisper is used by the WhisperX wrapper as the local ASR backend:
- entrypoint in this bundle: `modules/transcription/run_whisperx_single.py`
- external repository: [faster-whisper](https://github.com/SYSTRAN/faster-whisper)

NeMo is used only for speaker diarization in the mingling workflow:
- it takes mingling section audio and produces RTTM speaker diarization files
- this bundle does not include NeMo code, NeMo models, or cluster execution scripts
- the included main-speaker scripts consume the RTTM files produced by external diarization
- external repository: [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)

PII detection uses Presidio and spaCy:
- entrypoints in this bundle: `pii_redaction/src/run_audio_redaction.py` and `pii_redaction/src/run_text_detection.py`
- external repositories: [Microsoft Presidio](https://github.com/microsoft/presidio), [spaCy](https://github.com/explosion/spaCy)

## Data Not Included

This folder does not include data. It expects the user to provide the needed inputs locally.

Expected inputs include:
- audio files to transcribe or redact
- WhisperX `result.json` or `result_abs.json` files with word-level timestamps
- timeline-preserving per-channel WAV files and `*_clip_meta.json` files for mingling sections
- RTTM diarization files from an external diarization step, such as NeMo
- manual main-speaker selection CSV files
- an optional participant/microphone map CSV if metadata with participant IDs is needed

Generated outputs are also not included and should normally stay outside the paper repository:
- transcripts
- PII detection audits
- redacted audio
- pseudonymized transcript branches
- caches, logs, model weights, and intermediate diarization outputs

## Included Workflows

### 1. Audio PII redaction

Use this workflow for already prepared microphone audio clips and chest-worn midge microphone audio.

Input:
- a matching audio file
- a WhisperX `result.json` or `result_abs.json` transcript with word-level timestamps

Processing:
- detect configured PII entities in transcript text
- map detected PII spans back to word timestamps
- apply audio redaction over those time ranges

Output:
- `pii_spans.json`
- `pii_audit.csv`
- `audio_redacted_beep.wav`
- `audio_redacted_lowpass_1250hz.wav`

Example:

```bash
python modules/transcription/run_whisperx_single.py \
  --audio /path/to/audio.wav \
  --output-dir /path/to/transcript_dir \
  --model-dir /path/to/faster-whisper-large-v3 \
  --compute-type float16 \
  --beam-size 5 \
  --align-languages en,de,nl

python -m pii_redaction.src.run_audio_redaction \
  --transcript /path/to/transcript_dir/result.json \
  --audio /path/to/audio.wav \
  --out_dir /path/to/redaction_out \
  --config pii_redaction/config/policy_v2.yml \
  --language en
```

### 2. Mingling main-speaker pseudonymized transcripts

Use this workflow for mingling sections where the main speaker should be isolated before transcription.

Processing:
- trim timeline-preserving per-channel audio to mingling sections
- run external NeMo diarization to create RTTM files
- export candidate clips for manual main-speaker selection
- convert the manual selection CSV into `main_speakers.json`
- render `main_active` and `main_solo` audio
- transcribe `main_solo` audio with WhisperX
- restore absolute timestamps from clip metadata
- detect text PII in `result_abs.json`
- write pseudonymized transcript copies with shared mappings across mingling sections

Current pseudonymization behavior:
- same normalized person-name string gets the same `PERSON_n`
- emails become `[EMAIL_n]`
- phone numbers become `[PHONE_n]`
- mappings are shared across both mingling sections

Example command sequence:

```bash
python modules/channel_prep/trim_timeline_by_clock.py \
  --input_dir /path/to/full_channels_norm/session_full \
  --prefix session_full \
  --channels 1-32 \
  --start 13:45:00.000 \
  --end 14:20:00.000 \
  --out_dir /path/to/mingling_sections/mingling1 \
  --out_prefix mingling1

python modules/diarization/export_mingling1_candidates.py \
  --section mingling1 \
  --rttm_dir /path/to/diarization_rttms \
  --audio_dir /path/to/mingling_sections/mingling1 \
  --out_dir /path/to/main_speaker_selection/mingling1

python modules/diarization/export_main_speaker_map.py \
  --selection-csv /path/to/main_speaker_selection_template.csv \
  --participant-map-csv /path/to/participant_mic_map.csv \
  --output-json /path/to/main_speakers.json \
  --output-metadata-json /path/to/main_speakers_metadata.json

python modules/diarization/render_main_speaker_audio.py \
  --audio-dir /path/to/mingling_sections/mingling1 \
  --rttm-dir /path/to/diarization_rttms \
  --main-map /path/to/main_speakers.json \
  --out-root /path/to/main_speaker_audio/mingling1

python modules/transcription/run_whisperx_single.py \
  --audio /path/to/main_speaker_audio/mingling1/main_solo/mingling1_tl_ch01_main_solo.wav \
  --output-dir /path/to/work/main_speaker_text/mingling1/main_solo/ch01 \
  --model-dir /path/to/faster-whisper-large-v3 \
  --compute-type float16 \
  --beam-size 5 \
  --align-languages en,de,nl

python modules/transcription/add_absolute_timestamps.py \
  --result-json /path/to/work/main_speaker_text/mingling1/main_solo/ch01/result.json \
  --clip-meta /path/to/mingling_sections/mingling1/mingling1_tl_ch01_clip_meta.json \
  --output-json /path/to/work/main_speaker_text/mingling1/main_solo/ch01/result_abs.json

python -m pii_redaction.src.run_text_detection \
  --transcript /path/to/work/main_speaker_text/mingling1/main_solo/ch01/result_abs.json \
  --out_dir /path/to/pii_detection/main_speaker_audio/mingling1/main_solo/ch01 \
  --config pii_redaction/config/policy_v2.yml \
  --language en

python -m pii_redaction.src.run_text_pseudonymization \
  --transcript_base /path/to/work/main_speaker_text \
  --detection_base /path/to/pii_detection/main_speaker_audio \
  --out_base /path/to/work/pseudonymized_main_speaker_text \
  --config pii_redaction/config/policy_v2.yml \
  --sections mingling1 mingling2 \
  --channel_start 1 \
  --channel_end 32
```

## Dependencies

Python dependencies used by the included scripts:
- `numpy`
- `PyYAML`
- `soundfile`
- `torch`
- `faster-whisper`
- `whisperx`
- `presidio-analyzer`
- `spacy`
- spaCy model `en_core_web_sm`

External command-line dependency:
- `ffmpeg`, used for candidate clip export

External assets not included:
- Whisper/faster-whisper model weights
- WhisperX alignment model cache or local alignment models
- NeMo code, diarization configs, and diarization model weights
- raw audio, transcripts, PII outputs, logs, and caches

## Included Files

- `pii_redaction/src/`: PII detection, audio redaction, and text pseudonymization code
- `pii_redaction/config/policy_v2.yml`: release-oriented PII policy
- `modules/transcription/`: WhisperX transcription and absolute timestamp restoration
- `modules/channel_prep/`: timeline audio trimming for mingling sections
- `modules/diarization/`: main-speaker candidate export, mapping, and audio rendering
