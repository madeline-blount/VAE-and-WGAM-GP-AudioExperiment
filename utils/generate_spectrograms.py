import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configuration
INPUT_DIR = "/Users/madelineblount/Documents/2024-2025 UCSD Course Materials/COGS 185/COGS185-FINAL-PROJECT/aiexperiments-sound-maker/sounds/"
OUTPUT_DIR = 'data/spectrograms'
SAMPLE_RATE = 16000
N_MELS = 128
HOP_LENGTH = 256
DURATION = 4.0  # seconds

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def audio_to_mel(audio_path, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH, duration=DURATION):
    y, sr = librosa.load(audio_path, sr=sr, duration=duration)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB  # shape: (n_mels, time_steps)

def process_all_audio(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR):
    # Recursively find all .wav files
    wav_files = []
    for root, _, files in os.walk(input_dir):
        print(f"Scanning {root}... Found files: {files}") 
        for file in files:
            if file.endswith(('.wav', '.mp3')):
                full_path = os.path.join(root, file)
                wav_files.append(full_path)

    print(f"Total WAV files found: {len(wav_files)}")  # debug print

    for filepath in tqdm(wav_files, desc="Processing audio files"):
        try:
            mel = audio_to_mel(filepath)
            # Normalize and resize
            mel = (mel + 80) / 80  # normalize to [0, 1]
            mel = mel[:, :128]
            if mel.shape[1] < 128:
                mel = np.pad(mel, ((0, 0), (0, 128 - mel.shape[1])), mode='constant')

            # Make unique filename from relative path
            rel_path = os.path.relpath(filepath, input_dir)
            rel_path = rel_path.replace('/', '_').replace('\\', '_')  # flatten folder structure
            out_path = os.path.join(output_dir, rel_path.replace('.wav', '.npy'))

            np.save(out_path, mel.astype(np.float32))
        except Exception as e:
            print(f"Failed to process {filepath}: {e}")

if __name__ == '__main__':
    process_all_audio()