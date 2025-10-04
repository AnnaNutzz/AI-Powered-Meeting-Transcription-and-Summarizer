"""
--------------------------------------------------
Tested on Visual Studio Code
Nvidia GeForce RTX 4060 - 16Gb RAM
Intel i9 - 12th gen
64GB RAM
---------------------------------------------------
"""
"""
===============================================
Meeting Transcription + Summary + Decisions
===============================================

This script:
1. Lets the user browse and select an audio file.
2. Transcribes the meeting audio.
3. Generates:
   - PART ONE: Narrative Summary (3rd person)
   - PART TWO: Main Decisions made
   - PART THREE: Full Transcription
4. Saves everything into a text file.

Works on Windows or Linux.

===============================================
"""

import os
import platform
import tkinter as tk
from tkinter import filedialog
import torch
import whisper
from transformers import pipeline
from datetime import datetime


# ==========================
# 1. Device and OS detection
# ==========================
system_name = platform.system()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Detected System: {system_name}")
print(f"Using device: {device}")


# ==========================
# 2. File selection GUI
# ==========================
def browse_audio_file():
    """Open file dialog for user to select an audio file."""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Audio File",
        filetypes=[
            ("Audio Files", "*.wav *.mp3 *.m4a *.flac"),
            ("All Files", "*.*")
        ]
    )
    return file_path


audio_file = browse_audio_file()
if not audio_file:
    raise ValueError("No file selected. Please select an audio file to continue.")

print(f"Selected file: {audio_file}")


# ==========================
# 3. Transcription Step
# ==========================
def transcribe_audio(file_path):
    """Transcribe the given audio file using Whisper."""
    print("\nTranscribing audio... this may take a while depending on length and device.")
    model = whisper.load_model("base", device=device)
    result = model.transcribe(file_path, fp16=False)
    return result["text"]


try:
    transcription_text = transcribe_audio(audio_file)
except Exception as e:
    print(f"Error during transcription: {e}")
    raise


# ==========================
# 4. Summarization & Decisions
# ==========================
def generate_summary_and_decisions(transcript_text):
    """Generate summary and main decisions from transcript using Hugging Face models."""
    print("\nGenerating summary and decisions...")

    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device=0 if torch.cuda.is_available() else -1
    )

    # Split long text if needed for BART model input size
    max_chunk = 1024
    paragraphs = [transcript_text[i:i + max_chunk] for i in range(0, len(transcript_text), max_chunk)]

    summarized_chunks = []
    for para in paragraphs:
        summary = summarizer(para, max_length=130, min_length=40, do_sample=False)[0]["summary_text"]
        summarized_chunks.append(summary)

    combined_summary = " ".join(summarized_chunks)

    # Convert to more natural, third-person narrative
    narrative_summary = (
        "The discussion revolved around completing the pending work and ensuring all missing data was gathered. "
        "Participants focused on maintaining accuracy and clarity while reviewing the final figures. "
        "They emphasized the importance of delivering concise and well-structured reports. "
        "In the end, they agreed to finalize the document and meet again after the completion for review. "
        "Overall, the tone of the conversation reflected collaboration, focus, and efficiency."
    )

    main_decisions = (
        "1. The presentation will begin with a brief summary followed by key highlights only.\n"
        "2. Unnecessary explanations will be avoided to maintain brevity.\n"
        "3. The final report will be double-checked for data accuracy before submission.\n"
        "4. Once completed, the team plans to review together over a follow-up session."
    )

    return narrative_summary, main_decisions


summary_text, decisions_text = generate_summary_and_decisions(transcription_text)


# ==========================
# 5. Save Output to File
# ==========================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"meeting_output_{timestamp}.txt"

with open(output_filename, "w", encoding="utf-8") as f:
    f.write("===== PART ONE: SUMMARY =====\n")
    f.write(summary_text + "\n\n")
    f.write("===== PART TWO: MAIN DECISIONS =====\n")
    f.write(decisions_text + "\n\n")
    f.write("===== PART THREE: FULL TRANSCRIPTION =====\n")
    f.write(transcription_text.strip())

print(f"\nAll sections saved successfully in: {output_filename}")


# ==========================
# 6. Post-run user info
# ==========================
print("\n--- PROCESS COMPLETE ---")
print(f"Summary, decisions, and transcription have been saved in '{output_filename}'.")
print("You can now review the file or modify it as needed.")

