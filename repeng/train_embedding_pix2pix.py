import torch
import torch.nn as nn
import  numpy as np
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import wandb

class MLPGenerator(nn.Module):
    def __init__(self, input_dim: int, noise_dim: int = 16, hidden_dim: int = 512):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + noise_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, z):
        return self.model(torch.cat([x, z], dim=1))


class MLPDiscriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))



def train_embedding_pix2pix(pos_embds: np.ndarray, neg_embds: np.ndarray, layer: int, num_epochs: int = 1000, batch_size: int = 64) -> nn.Module:
    wandb.init(
        project="pix2pix-embedding",
        name=f"layer_{layer}",
        config={
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "layer": layer,
            "noise_dim": 16,
        },
    )
    print("Start trainig for the layer "+str(layer))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dim = pos_embds.shape[1]
    noise_dim = 16

    G = MLPGenerator(input_dim=dim, noise_dim=noise_dim).to(device)
    D = MLPDiscriminator(input_dim=dim).to(device)

    pos_tensor = torch.tensor(pos_embds, dtype=torch.float32).to(device)
    neg_tensor = torch.tensor(neg_embds, dtype=torch.float32).to(device)

    dataset = TensorDataset(pos_tensor, neg_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    g_opt = optim.Adam(G.parameters(), lr=1e-4)
    d_opt = optim.Adam(D.parameters(), lr=1e-4)
    loss_fn = nn.BCELoss()

    for epoch in range(num_epochs):
        for x_real, y_real in loader:
            x_real, y_real = x_real.to(device), y_real.to(device)
            batch_size = x_real.size(0)
            z = torch.randn(batch_size, noise_dim, device=device)

            # === Train Discriminator ===
            y_fake = G(x_real, z).detach()
            d_real = D(x_real, y_real)
            d_fake = D(x_real, y_fake)
            d_loss = loss_fn(d_real, torch.ones_like(d_real)) + loss_fn(d_fake, torch.zeros_like(d_fake))
            D.zero_grad()
            d_loss.backward()
            d_opt.step()

            # === Train Generator ===
            z = torch.randn(batch_size, noise_dim, device=device)
            y_fake = G(x_real, z)
            d_fake = D(x_real, y_fake)
            g_loss = loss_fn(d_fake, torch.ones_like(d_fake))
            G.zero_grad()
            g_loss.backward()
            g_opt.step()

            # === WanDB ===
            # Log losses
            wandb.log({
                "epoch": epoch,
                "d_loss": d_loss.item(),
                "g_loss": g_loss.item()
            })
    print("Finished training for the layer of "+str(layer))
    wandb.finish()
    return G.eval().cpu()

