import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

# Import your VAE model and loss function
from models.vae import VAE, vae_loss  # adjust import path if needed
# Import your dataset class
from utils.pytorch_dataset import SpectrogramDataset  # adjust import path

def train_vae(model, dataloader, epochs=20, lr=1e-3, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            batch_data = batch_data.to(device)

            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch_data)
            loss = vae_loss(recon_batch, batch_data, mu, logvar)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}] Average loss: {avg_loss:.4f}")

    print("Training complete.")

if __name__ == '__main__':
    # Instantiate dataset and dataloader
    dataset = SpectrogramDataset('data/spectrograms')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

    # Instantiate model
    model = VAE(latent_dim=128)

    # Train
    train_vae(model, dataloader, epochs=20, lr=1e-3)

    torch.save(model.state_dict(), "vae_model_2.pth")
    print("Model saved as vae_model_2.pth")