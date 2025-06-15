import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SpectrogramDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        # List all .npy files in the directory
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Load the spectrogram numpy file
        file_path = os.path.join(self.data_dir, self.files[idx])
        mel = np.load(file_path)
        # Convert numpy array to torch tensor (float32)
        mel_tensor = torch.from_numpy(mel).float()
        # Add channel dimension for CNNs: (1, height, width)
        mel_tensor = mel_tensor.unsqueeze(0)
        return mel_tensor

# Usage example
if __name__ == "__main__":
    dataset = SpectrogramDataset('data/spectrograms')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

    # Iterate through batches
    for batch_idx, batch_data in enumerate(dataloader):
        print(f"Batch {batch_idx} shape: {batch_data.shape}")  # Expect (batch_size, 1, 128, 128)
        # For demo just break after one batch
        break