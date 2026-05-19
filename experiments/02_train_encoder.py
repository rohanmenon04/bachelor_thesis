"""
Script 2 — Encoder Training
=============================
Trains two encoders on the offline trajectory dataset (no reward signal):

  1. TransformerVQEncoder  — Transformer + VQ bottleneck (proposed model)
  2. ContinuousEncoder     — same architecture, continuous bottleneck (baseline)

Both are trained to reconstruct the 7x7x3 MiniGrid observation from their
compressed representation using per-channel cross-entropy loss.

The VQ encoder also minimises a commitment loss that keeps encoder outputs
close to their assigned codebook entry (van den Oord et al., 2017).

After training, both encoders are frozen and used by scripts 03 and 04.

Usage:
  python 02_train_encoder.py

Requires:
  data/trajectories.pkl  (from 01_collect_data.py)

Outputs:
  checkpoints/vq_encoder.pt
  checkpoints/continuous_encoder.pt
  plots/encoder_training.png
"""

import os
import pickle
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models import (
    ContinuousEncoder,
    TransformerVQEncoder,
    CODEBOOK_SIZE,
    reconstruction_loss,
)

BATCH_SIZE = 128
N_EPOCHS   =  80
LR         = 3e-4
DEVICE     = ('cuda' if torch.cuda.is_available()
               else 'mps' if torch.backends.mps.is_available()
               else 'cpu')


class ObsDataset(Dataset):
    def __init__(self, observations):
        self.obs = torch.tensor(observations, dtype=torch.long)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx]


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def train_vq(model, loader, n_epochs):
    model = model.to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    hist  = {'recon': [], 'commit': [], 'perplexity': []}

    epoch_bar = tqdm(range(n_epochs), desc='VQ encoder', unit='epoch')

    for epoch in epoch_bar:
        recon_list, commit_list, perp_list = [], [], []
        model.train()

        for batch in loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()

            logits, indices, commit = model(batch)
            recon = reconstruction_loss(logits, batch)
            loss  = recon + commit
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            recon_list.append(recon.item())
            commit_list.append(commit.item())
            perp_list.append(model.vq.get_perplexity(indices))

        mean_recon  = np.mean(recon_list)
        mean_commit = np.mean(commit_list)
        mean_perp   = np.mean(perp_list)
        pct         = 100 * mean_perp / CODEBOOK_SIZE

        hist['recon'].append(mean_recon)
        hist['commit'].append(mean_commit)
        hist['perplexity'].append(mean_perp)

        epoch_bar.set_postfix(
            recon=f'{mean_recon:.3f}',
            commit=f'{mean_commit:.3f}',
            perp=f'{mean_perp:.1f}/{CODEBOOK_SIZE} ({pct:.0f}%)',
        )

    return hist


def train_continuous(model, loader, n_epochs):
    model = model.to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    hist  = {'recon': []}

    epoch_bar = tqdm(range(n_epochs), desc='Continuous encoder', unit='epoch')

    for epoch in epoch_bar:
        recon_list = []
        model.train()

        for batch in loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            logits, _ = model(batch)
            recon = reconstruction_loss(logits, batch)
            recon.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            recon_list.append(recon.item())

        mean_recon = np.mean(recon_list)
        hist['recon'].append(mean_recon)
        epoch_bar.set_postfix(recon=f'{mean_recon:.3f}')

    return hist


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training(vq_hist, cont_hist, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Encoder Pre-Training', fontsize=12)

    ax = axes[0]
    ax.plot(vq_hist['recon'],   label='VQ reconstruction',   color='green')
    ax.plot(vq_hist['commit'],  label='VQ commitment',        color='green',  linestyle='--')
    ax.plot(cont_hist['recon'], label='Continuous reconstruction', color='orange')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(vq_hist['perplexity'], color='green')
    ax.axhline(y=CODEBOOK_SIZE, color='red', linestyle='--', label=f'Max K={CODEBOOK_SIZE}')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Codebook Perplexity')
    ax.set_title('Codebook Utilization\n(exp(entropy) — higher = less collapse)')
    ax.set_ylim(0, CODEBOOK_SIZE + 5)
    ax.legend(fontsize=8)

    ax = axes[2]
    total_vq = [r + c for r, c in zip(vq_hist['recon'], vq_hist['commit'])]
    ax.plot(total_vq,           label='VQ total',    color='green')
    ax.plot(cont_hist['recon'], label='Continuous',  color='orange')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss Comparison')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs('checkpoints/vq_encoder', exist_ok=True)
    os.makedirs('plots',       exist_ok=True)

    print(f"Device: {DEVICE}")

    with open('data/trajectories.pkl', 'rb') as f:
        data = pickle.load(f)

    dataset = ObsDataset(data['observations'])
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print(f"Dataset: {len(dataset)} observations")

    # --- VQ encoder ---
    print(f"\n--- VQ-Transformer Encoder (K={CODEBOOK_SIZE}) ---")
    vq_model = TransformerVQEncoder()
    print(f"Parameters: {sum(p.numel() for p in vq_model.parameters()):,}")
    vq_hist  = train_vq(vq_model, loader, N_EPOCHS)
    vq_model.cpu()
    torch.save(vq_model.state_dict(), 'checkpoints/vq_encoder/vq_encoder.pt')
    print("Saved checkpoints/vq_encoder/vq_encoder.pt")

    final_perp = vq_hist['perplexity'][-1]
    pct = 100 * final_perp / CODEBOOK_SIZE
    if pct < 15:
        print(f"WARNING: codebook utilization is low ({pct:.0f}%). "
              "Consider reducing commitment beta or training longer.")
    else:
        print(f"Codebook utilization: {final_perp:.1f}/{CODEBOOK_SIZE} ({pct:.0f}%) — looks healthy.")

    # --- Continuous encoder ---
    print(f"\n--- Continuous Encoder Baseline ---")
    cont_model = ContinuousEncoder()
    print(f"Parameters: {sum(p.numel() for p in cont_model.parameters()):,}")
    cont_hist  = train_continuous(cont_model, loader, N_EPOCHS)
    cont_model.cpu()
    torch.save(cont_model.state_dict(), 'checkpoints/vq_encoder/continuous_encoder.pt')
    print("Saved checkpoints/vq_encoder/continuous_encoder.pt")

    plot_training(vq_hist, cont_hist, 'plots/encoder_training.png')
    print("\nNext: python 03_linear_probe.py")


if __name__ == '__main__':
    main()
