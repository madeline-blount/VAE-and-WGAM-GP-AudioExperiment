import numpy as np
import matplotlib.pyplot as plt
import os

SPECTROGRAM_DIR = 'data/spectrograms'  # Adjust if different

def load_spectrogram(file_path):
    return np.load(file_path)

def show_spectrogram(mel_spec):
    plt.figure(figsize=(8, 4))
    plt.imshow(mel_spec, aspect='auto', origin='lower', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency')
    plt.show()

if __name__ == "__main__":
    files = [f for f in os.listdir(SPECTROGRAM_DIR) if f.endswith('.npy')]
    for i, file in enumerate(files[:3]):  # Show first 3 spectrograms
        print(f"Visualizing {file}")
        mel = load_spectrogram(os.path.join(SPECTROGRAM_DIR, file))
        show_spectrogram(mel)