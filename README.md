# AI-Powered-Meeting-Transcription-and-Summarizer

## Overview
This project transcribes audio recordings of meetings, performs speaker diarization, and generates a structured output file with:

1. **Part One:** Multi-paragraph, third-person professional briefing  
2. **Part Two:** Main decisions extracted from discussion  
3. **Part Three:** Full transcription with speaker labels  

The pipeline uses **Faster Whisper**, **Pyannote Audio**, and Hugging Face **summarization models**.

---

## Requirements

### 1. Python

- Python 3.10+ recommended (works with 3.9+)  
- Check version:
```bash
python --version
```

### 2. CUDA (Optional for GPU)

If using GPU, install CUDA 12.1 (for newer GPUs)

PyTorch compatible version:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Required Python Libraries

Install all dependencies:
```
pip install faster-whisper==0.9.2
pip install pyannote.audio==4.0.0
pip install pydub==0.25.1
pip install torch==2.3.1
pip install transformers==4.36.2
pip install huggingface_hub==0.17.4
```
### 4. FFmpeg

Required by pydub for audio handling

Install via Windows:

[Download](https://ffmpeg.org/download.html)

Add bin folder to PATH

Test:
```
ffmpeg -version
```

---

## Setup

### 1. Clone repository or download script

### 2. Set your audio directory and file path in the script:
```
AUDIO_DIR = "D:/Stuff/ANNIE/"
AUDIO_FILE = os.path.join(AUDIO_DIR, "test.wav")
OUTPUT_FILE = os.path.join(AUDIO_DIR, "meeting_output.txt")
```

### 3. Hugging Face Login

Obtain token from: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

Add token in script:
```
HF_TOKEN = "hf_YOUR_TOKEN_HERE"
```

---

## Running the Pipeline
```
python meeting_pipeline.py
```

Steps executed automatically:

1. Transcription in 1-minute chunks using Faster Whisper

2. Speaker diarization using Pyannote Audio

3. Merge transcription with speaker labels

4. Generate professional third-person briefing (Part One)

5. Extract main decisions (Part Two)

6. Save full transcription (Part Three) in .txt

---

## Output

A single `.txt` file with the following structure:
```
===== PART ONE: SUMMARY =====
<Third-person multi-paragraph briefing>

===== PART TWO: MAIN DECISIONS =====
<List of decisions/actions>

===== PART THREE: FULL TRANSCRIPTION =====
[0.00-5.23] Speaker 1: Hello everyone...
[5.24-10.12] Speaker 2: Today we discuss...
```

---

## Common Issues & Fixes
### 1. ffmpeg not found

- Pydub requires ffmpeg.

- Fix: Download ffmpeg and add to PATH. Test with:
```
ffmpeg -version
```

### 2. CUDA not available or model runs on CPU

- Make sure GPU drivers and CUDA version match PyTorch version.

- Fix: Install compatible PyTorch version with CUDA:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Pyannote missing dependencies or build errors

- Ensure torch is installed first.

- Use Python 3.10 or 3.11 to avoid compatibility issues.

### 4. Hugging Face authentication errors

- Ensure your token is correct.

- Test login separately:
```
from huggingface_hub import login
login("hf_YOUR_TOKEN_HERE")
```

### 5. Memory issues on long audio

- Script uses chunking to avoid out-of-memory errors.

- Reduce CHUNK_LENGTH_MS if memory is limited:
```
CHUNK_LENGTH_MS = 30 * 1000  # 30 seconds
```

### 6. Long transcription fails on summarizer

Text is split into chunks automatically in the pipeline.

Ensure `max_chunk` in `generate_briefing_paragraphs()` is <=1000 characters per chunk.

---

# Notes

- Works on Windows 10+ and Linux.

- For best results, use GPU.

- Audio formats supported by pydub: WAV, MP3, FLAC, etc.

- Output text can be imported into Word or Google Docs for formatting.
