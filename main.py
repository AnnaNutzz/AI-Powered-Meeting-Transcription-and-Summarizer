"""
AI-Powered Meeting Summarizer
This script processes a meeting audio file to produce a structured summary with discussion points and action items.
Requirements:
- Python 3.9.13
- PyTorch with CUDA support (for GPU acceleration)
- Whisper (for transcription)
- Pyannote (for speaker diarization)
- Transformers (for summarization)
- NLTK (for text processing)
- SoundFile (for audio handling)
- Scikit-learn (for additional utilities)

Tested on:
- Dell G15 with GTX 1650 (4GB VRAM)         ✔ (small)
- REMOTE PC with RTX A6000 (48GB VRAM)      ⭕
- EAGLE PC with RTX 4060 (16GB VRAM)        ✔ (medium/large-v3)
"""


"""
# 1. Create a virtual environment (recommended)
python -m venv office_ai
# Activate it (Windows PowerShell)
.\office_ai\Scripts\Activate.ps1
# or Windows CMD
.\office_ai\Scripts\activate.bat

# 2. Install required packages (with CUDA support for RTX A6000)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install openai-whisper pyannote.audio transformers scikit-learn nltk soundfile

# 3. Download NLTK tokenizer models
python -m nltk.downloader punkt

Offline caution: After the first download of models from Hugging Face, you can set:

import os
os.environ["HF_HUB_OFFLINE"] = "1"

This ensures no network calls during inference.
"""

# imports

import whisper
import nltk
from nltk.tokenize import sent_tokenize
from pyannote.audio import Pipeline
from transformers import pipeline


nltk.download("punkt")

"""
Stage 1: Import and load Whisper --------------------------------------------
"""

# Load the local Whisper model
""" 
Large-v3 requires VRAM; fine in FP16
model_path = "/models/whisper-small" Dell G15 // 
    model_path = "/models/whisper-large-v3" A6000

Trade-offs:
    Tiny: fastest, lowest accuracy.
    Small: decent speed, good accuracy for clear audio.
    Medium: better accuracy, slightly slower; might barely fit on 4GB GPU in FP16.

"""
"""
model_path = "./models/whisper-large-v3"  # adjust if needed
whisper_model = whisper.load_model("model_path")
"""
whisper_model = whisper.load_model("medium")  # Dell G15

# Transcribe the meeting audio
audio_file = "test_new.mp3"
result = whisper_model.transcribe(audio_file, fp16=True)  # FP16 reduces VRAM usage

# Extract the full transcript
raw_text = result["text"]

# Optionally, you can inspect timestamps for each segment
segments = result.get("segments", [])
for seg in segments[:3]:  # show first 3 segments
    print(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text']}")

"""
Stage 2: Speaker diarization ------------------------------------------------
"""

# Load local pyannote model
diarizer = Pipeline.from_pretrained("./models/pyannote-speaker-diarization-3.1")

# Run diarization on the same audio file
diarization_result = diarizer(audio_file)

# Print a few speaker segments
print("Speaker Segments:")
for turn, _, speaker in diarization_result.itertracks(yield_label=True):
    print(f"{speaker}: {turn.start:.2f}s → {turn.end:.2f}s")

"""
Stage 3: Split transcript into discussion points -----------------------------------
"""

# Split full transcript into sentences
sentences = sent_tokenize(raw_text)

# Naive approach: group sentences into chunks of 5-8 sentences each
chunk_size = 3 # can be adjusted (default better 6)
discussion_points = [sentences[i:i+chunk_size] for i in range(0, len(sentences), chunk_size)]

# Prepare structured discussions
structured_points = []
for i, chunk in enumerate(discussion_points):
    structured_points.append({
        "point_number": i+1,
        "discussions": chunk
    })

# Inspect first point
print(f"Point 1 discussions: {structured_points[0]['discussions']}")

"""
Stage 4: Summarization and Decision Extraction -------------------------------
"""

# Load summarizer locally
"""
summarizer = pipeline("summarization", model="./models/bart-large-cnn", device=0)  # device=0 for A6000 GPU
summarizer = pipeline("summarization", model="facebook/bart-base", device=0)  # Dell G15
"""

summarizer = pipeline("summarization", model="./models/bart-large-cnn", device=0)  # device=0 for A6000 GPU

final_output = []

for point in structured_points:
    text_chunk = " ".join(point["discussions"])
    summary = summarizer(text_chunk, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
    
    final_output.append({
        "point_number": point["point_number"],
        "discussions": point["discussions"],
        "action_taken": summary
    })

# Inspect first summary
print(final_output[0])

"""
Stage 5: Output structured summary to text file --------------------------------
"""

output_file = "structured_summary.txt"

with open(output_file, "w", encoding="utf-8") as f:
    for point in final_output:
        f.write(f"{point['point_number']}. Point {point['point_number']}:\n")
        for idx, discussion in enumerate(point["discussions"]):
            f.write(f"   {chr(97 + idx)}. {discussion}\n")  # 97 = 'a'
        f.write(f"Action taken: {point['action_taken']}\n\n")

print(f"Structured summary written to {output_file}")
