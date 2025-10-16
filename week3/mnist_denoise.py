#!/usr/bin/env python3
"""
Lightweight MNIST denoising CNN using PyTorch and local CSVs (no TensorFlow).
Loads CSV MNIST, adds Gaussian noise on-the-fly, trains a tiny autoencoder, saves visualization.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class CsvMnistDataset(Dataset):
    def __init__(self, csv_path: str):
        data = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
        images_flat = data[:, 1:] / 255.0
        num_examples = images_flat.shape[0]
        images = images_flat.reshape(num_examples, 1, 28, 28)
        self.images = torch.from_numpy(images)

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.images[idx]


class NoisyWrapperDataset(Dataset):
    def __init__(self, base: Dataset, noise_factor: float, seed: int):
        self.base = base
        self.noise_factor = noise_factor
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        clean = self.base[idx]
        noise = torch.randn_like(clean, generator=self.generator)
        noisy = torch.clamp(clean + self.noise_factor * noise, 0.0, 1.0)
        return noisy, clean


class DenoisingAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def visualize(clean: torch.Tensor, noisy: torch.Tensor, denoised: torch.Tensor, out_path: str, n: int = 10) -> None:
    clean_np = clean.detach().cpu().numpy()
    noisy_np = noisy.detach().cpu().numpy()
    denoised_np = denoised.detach().cpu().numpy()

    plt.figure(figsize=(20, 6))
    for i in range(n):
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(clean_np[i, 0], cmap="gray")
        ax.axis("off")

        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(noisy_np[i, 0], cmap="gray")
        ax.axis("off")

        ax = plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(denoised_np[i, 0], cmap="gray")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    train_csv = "/home/gleb/hse/nn2025/week3/mnist_train_10p_uniform.csv"
    test_csv = "/home/gleb/hse/nn2025/week3/mnist_test_10p_uniform.csv"

    train_clean = CsvMnistDataset(train_csv)
    test_clean = CsvMnistDataset(test_csv)

    noise_factor = 0.5
    train_dataset = NoisyWrapperDataset(train_clean, noise_factor=noise_factor, seed=42)
    test_dataset = NoisyWrapperDataset(test_clean, noise_factor=noise_factor, seed=1337)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    model = DenoisingAutoencoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 5

    for _ in range(num_epochs):
        model.train()
        for noisy_batch, clean_batch in train_loader:
            output_batch = model(noisy_batch)
            loss = criterion(output_batch, clean_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        noisy_vis, clean_vis = next(iter(test_loader))
        denoised_vis = model(noisy_vis)

    out_path = "/home/gleb/hse/nn2025/week3/mnist_denoise_results.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    visualize(clean_vis, noisy_vis, denoised_vis, out_path=out_path, n=10)


if __name__ == "__main__":
    main()


