"""
Tested on Visual Studio Code
Nvidia GeForce RTX 4060 - 16Gb RAM
Intel i9 - 12th gen
64GB RAM
"""

import os
import re
import torch
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from huggingface_hub import login
from pydub import AudioSegment
from transformers import pipeline
from tkinter import Tk, filedialog

# ------------------------
# HUGGING FACE LOGIN
# ------------------------
HF_TOKEN = "hf_token"
login(HF_TOKEN)

# ------------------------
# CONFIGURATION
# ------------------------

"""

Changes to make:

AUDIO_DIR = "the path of the directory/folder in which the transcript is to be added in
CHUNK_LENGTH_MS = reduce '60' if less memory

"""
AUDIO_DIR = "D:/"

Tk().withdraw()   
print("Please select an audio file (.wav, .mp3, etc.)")
AUDIO_FILE = filedialog.askopenfilename(
    title="Select Audio File",
    filetypes=[("Audio Files", "*.wav *.mp3 *.m4a *.flac *.ogg *.aac")]
)

if not AUDIO_FILE:
    print("No file selected. Exiting.")
    exit()

OUTPUT_FILE = os.path.join(AUDIO_DIR, "meeting_output.txt")
CHUNK_LENGTH_MS = 60 * 1000  # 1-minute chunks

# ------------------------
# DEVICE AND MODELS
# ------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

print(f"Using device: {device}, compute_type: {compute_type}")

# Whisper transcription
whisper_model = WhisperModel("medium", device=device, compute_type=compute_type)

# Pyannote diarization
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

# Hugging Face summarizer for professional briefing
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# ------------------------
# FUNCTIONS
# ------------------------

def chunk_audio(audio_path, chunk_length_ms=CHUNK_LENGTH_MS):
    """Split audio into chunks (ms)"""
    audio = AudioSegment.from_file(audio_path)
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunks.append(audio[i:i+chunk_length_ms])
    return chunks

def transcribe_chunked_audio(audio_path):
    """Transcribe audio in chunks using Faster Whisper"""
    segments_all = []
    audio_chunks = chunk_audio(audio_path)
    for idx, chunk in enumerate(audio_chunks):
        print(f"Transcribing chunk {idx+1}/{len(audio_chunks)}...")
        tmp_file = "temp_chunk.wav"
        chunk.export(tmp_file, format="wav")
        segments, _ = whisper_model.transcribe(tmp_file, beam_size=5, vad_filter=True)
        for s in segments:
            segments_all.append({
                "start": s.start,
                "end": s.end,
                "text": s.text
            })
        os.remove(tmp_file)
    return segments_all

def diarize_audio(audio_path):
    """Perform speaker diarization using Pyannote"""
    # Convert to mono 16kHz for Pyannote
    temp_file = "temp_diarize.wav"
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(temp_file, format="wav")

    diarization = diarization_pipeline(temp_file)
    os.remove(temp_file)

    diarization_result = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        diarization_result.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })
    return diarization_result

def merge_transcription_diarization(transcription, diarization):
    """Merge segments with speaker labels"""
    merged = []
    for t in transcription:
        speakers = [d["speaker"] for d in diarization if not (d["end"] < t["start"] or d["start"] > t["end"])]
        speaker_label = speakers[0] if speakers else "Unknown"
        merged.append({
            "start": t["start"],
            "end": t["end"],
            "speaker": speaker_label,
            "text": t["text"]
        })
    return merged

def generate_narrative_briefing(merged_transcription, max_chunk=1200):
    """
    Generate third-person multi-paragraph briefing.
    Fixes first-person to third-person, expands contractions, and smooths sentences.
    """
    import re

    # Combine all text
    full_text = " ".join([seg["text"] for seg in merged_transcription])
    full_text = re.sub(r'\s+', ' ', full_text)

    # Function to replace first-person with third-person
    def convert_to_third_person(text):
        # Expand common contractions first
        contractions = {
            r"\bI'm\b": "I am",
            r"\bI'll\b": "I will",
            r"\bI've\b": "I have",
            r"\bI'd\b": "I would",
            r"\bwe're\b": "we are",
            r"\bwe'll\b": "we will",
            r"\bwe've\b": "we have",
            r"\bwe'd\b": "we would",
            r"\bdon't\b": "do not",
            r"\bdoesn't\b": "does not",
            r"\bcan't\b": "cannot",
        }
        for k, v in contractions.items():
            text = re.sub(k, v, text, flags=re.IGNORECASE)

        # Convert pronouns and verbs to third-person
        text = re.sub(r'\bI\b', "The speaker", text)
        text = re.sub(r'\bwe\b', "They", text, flags=re.IGNORECASE)
        text = re.sub(r'\bmy\b', "the speaker's", text)
        text = re.sub(r'\bour\b', "their", text, flags=re.IGNORECASE)
        text = re.sub(r'\bme\b', "the speaker", text)
        text = re.sub(r'\bus\b', "them", text, flags=re.IGNORECASE)

        return text

    # Split into manageable chunks for summarization
    chunks = [full_text[i:i+max_chunk] for i in range(0, len(full_text), max_chunk)]
    paragraphs = []

    for chunk in chunks:
        chunk = convert_to_third_person(chunk)
        # Summarize chunk into paragraph
        summary = summarizer(chunk, max_length=180, min_length=60, do_sample=False)[0]['summary_text']
        # Capitalize first letter for paragraph
        summary = summary[0].upper() + summary[1:]
        paragraphs.append(summary)

    # Combine paragraphs smoothly
    return "\n\n".join(paragraphs)


def generate_summary_and_decisions(merged_transcription):
    """Generate narrative briefing and main decisions"""
    summary = generate_narrative_briefing(merged_transcription)

    # Extract decisions based on keywords
    full_text = " ".join([seg["text"] for seg in merged_transcription])
    sentences = re.split(r'(?<=[.!?]) +', full_text)
    decision_keywords = ["decided", "will", "plan", "action", "agree", "next step", "follow up"]
    decisions = [s for s in sentences if any(k.lower() in s.lower() for k in decision_keywords)]

    return summary, "\n".join(decisions)

def save_output(summary, decisions, merged_transcription, output_file=OUTPUT_FILE):
    """Save all three parts to a structured txt file"""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("===== PART ONE: SUMMARY =====\n")
        f.write(summary + "\n\n")
        f.write("===== PART TWO: MAIN DECISIONS =====\n")
        f.write(decisions + "\n\n")
        f.write("===== PART THREE: FULL TRANSCRIPTION =====\n")
        for seg in merged_transcription:
            start = round(seg["start"], 2)
            end = round(seg["end"], 2)
            f.write(f"[{start}-{end}] {seg['speaker']}: {seg['text']}\n")
    print(f"Output saved to {output_file}")

# ------------------------
# MAIN PIPELINE
# ------------------------
def run_pipeline(audio_file):
    print("Step 1: Transcription...")
    transcription = transcribe_chunked_audio(audio_file)

    print("Step 2: Speaker Diarization...")
    diarization = diarize_audio(audio_file)

    print("Step 3: Merge transcription and speakers...")
    merged_result = merge_transcription_diarization(transcription, diarization)

    print("Step 4: Generate professional narrative briefing and decisions...")
    summary, decisions = generate_summary_and_decisions(merged_result)

    print("Step 5: Save structured output...")
    save_output(summary, decisions, merged_result)

# ------------------------
# ENTRY POINT
# ------------------------
if __name__ == "__main__":
    if not os.path.exists(AUDIO_FILE):
        print(f"Audio file not found: {AUDIO_FILE}")
    else:
        run_pipeline(AUDIO_FILE)
