import torch
from torch.utils.data import DataLoader
from torch import autograd, optim
from tqdm import tqdm
from torch.utils.data import Subset

from models.wgan_gp import Generator, Discriminator
from utils.pytorch_dataset import SpectrogramDataset

LATENT_DIM = 128
LAMBDA_GP = 10
N_CRITIC = 3
EPOCHS = 20
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gradient_penalty(D, real_data, fake_data):
    batch_size = real_data.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, device=DEVICE)
    interpolated = (epsilon * real_data + (1 - epsilon) * fake_data).requires_grad_(True)

    d_interpolated = D(interpolated)
    grad_outputs = torch.ones_like(d_interpolated)
    gradients = autograd.grad(outputs=d_interpolated, inputs=interpolated,
                              grad_outputs=grad_outputs,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp

def train():
    full_dataset = SpectrogramDataset('data/spectrograms')

    # Use only the first N samples (e.g., 1000)
    subset_indices = list(range(1000))  # <-- change to desired number
    dataset = Subset(full_dataset, subset_indices)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    G = Generator(LATENT_DIM).to(DEVICE)
    D = Discriminator().to(DEVICE)

    opt_G = optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=1e-4, betas=(0.0, 0.9))

    for epoch in range(EPOCHS):
        for i, real_data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
            real_data = real_data.to(DEVICE)
            batch_size = real_data.size(0)

            # === Train Discriminator ===
            for _ in range(N_CRITIC):
                z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
                fake_data = G(z).detach()
                d_real = D(real_data)
                d_fake = D(fake_data)
                gp = gradient_penalty(D, real_data, fake_data)
                d_loss = -d_real.mean() + d_fake.mean() + LAMBDA_GP * gp

                opt_D.zero_grad()
                d_loss.backward()
                opt_D.step()

            # === Train Generator ===
            z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
            fake_data = G(z)
            g_loss = -D(fake_data).mean()

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

        print(f"[Epoch {epoch+1}] D loss: {d_loss.item():.4f} | G loss: {g_loss.item():.4f}")

    torch.save(G.state_dict(), "wgan_gp_generator_ncrit3.pth")
    print("Generator saved as wgan_gp_generator_ncrit3.pth")

if __name__ == '__main__':
    train()