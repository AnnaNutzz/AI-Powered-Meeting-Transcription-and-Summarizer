# AI-Powered Meeting Transcriptor & Summarizer

An intelligent Python-based tool that automatically:
1. Lets you **browse and select** any audio file.
2. **Transcribes** the conversation using OpenAI Whisper.
3. Generates:
   - **PART ONE:** A clean narrative summary (third-person briefing style)
   - **PART TWO:** The main decisions made during the meeting
   - **PART THREE:** The full transcription
4. Saves everything neatly into a single `.txt` file.

---

## Features
- Automatic **OS detection** (Windows / macOS / Linux)
- GPU acceleration with **CUDA** if available
- Natural **third-person summarization** using Hugging Face `facebook/bart-large-cnn`
- User-friendly **file browser**
- Exports timestamped `.txt` output

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/meeting-transcriptor.git
cd meeting-transcriptor
```
### 2. Create a Virtual Environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
.venv\Scripts\activate          # Windows
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
If you’re using CUDA 12.1, make sure to install the matching PyTorch build:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## How to Run
```bash
python main.py
```
A file browser will open — select your .wav, .mp3, .m4a, or .flac file.

The script will:

- Transcribe the audio

- Generate the summary and decisions

- Save a file like this:

```
meeting_output_20251004_192512.txt
```

## Output File Format
```
===== PART ONE: SUMMARY =====
The discussion revolved around completing the pending work and ensuring all missing data was gathered.
Participants focused on maintaining accuracy and clarity while reviewing the final figures.
They emphasized the importance of delivering concise and well-structured reports.
In the end, they agreed to finalize the document and meet again after completion for review.

===== PART TWO: MAIN DECISIONS =====
1. The presentation will begin with a brief summary followed by key highlights only.
2. Unnecessary explanations will be avoided to maintain brevity.
3. The final report will be double-checked for data accuracy before submission.
4. The team plans to review together once it’s finalized.

===== PART THREE: FULL TRANSCRIPTION =====
<raw transcribed text>
```

---

## Dependencies & Versions

| Package | Recommended Version | Description |
|----------|---------------------|-------------|
| **Python** | 3.9+ | Required for Whisper and Transformers |
| **torch** | 2.1.0+ | GPU-accelerated PyTorch backend |
| **openai-whisper** | 20230314 | Whisper speech recognition |
| **transformers** | 4.41.1 | Hugging Face summarization model (BART/CNN) |
| **ffmpeg** | latest | Required for Whisper audio decoding |
| **tkinter** | built-in | File browser support for user-selected audio |

---

## Common Issues & Fixes
### 1. Whisper Not Found
Install directly from GitHub:
```
pip install git+https://github.com/openai/whisper.git
```

### 2. FFmpeg Missing
- Windows: Download FFmpeg and add it to PATH.

- Linux/macOS:

```
sudo apt install ffmpeg
```
or

```
brew install ffmpeg
```
### 3. CUDA Not Detected
Check PyTorch + CUDA match:
```
python -m torch.utils.collect_env
```
If not found, reinstall with the correct CUDA toolkit version.

### 4. Out-of-Memory (Low RAM/GPU)
Switch to a lighter Whisper model:

```
model = whisper.load_model("small")  # or "tiny"
```
