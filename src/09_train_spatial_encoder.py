"""
Script 9 — Spatial Encoder Training (Condition I prerequisite)
==================================================================
Trains the VQSpatialEncoder, a small VQ autoencoder over agent grid
positions. The output is a discrete spatial token that pairs with the
semantic token from TransformerVQEncoder to form the fully-discrete
two-token state representation used in Condition I.

Architecture (defined in models.py — embedding+cross-entropy version):
    pos (x, y) ∈ [0, GRID)^2
        ↓ cat(x_embed, y_embed)
    z (latent_dim)
        ↓ VQ(K=32, D=32)  with EMA + dead-code restart
    z_q (latent_dim)
        ↓ MLP → 2 × GRID logits
    cross-entropy on (x, y)

Why this design rather than MLP-on-floats:
The first version (MLP encoder, MSE reconstruction over 2 floats) showed
codebook collapse to ~3/32 active codes — the MSE objective is solved by
a single code (predict the mean) so VQ has no pressure to discriminate.
Switching to the recipe used by the working semantic encoder
(discrete-int input → embedding lookup → cross-entropy reconstruction)
forces the codebook to preserve cell identity at each axis, the same
pressure that keeps the semantic encoder at 57/64 active codes.

Training data:
600 random-policy episodes plus a uniform grid-cell augmentation
(`N_GRID_COPIES` copies of every (x, y) ∈ [0, GRID)^2). The augmentation
ensures the encoder sees every reachable cell — under random policy alone
the agent spends most of its time in the start room, so cells in the
goal room are heavily under-represented in the natural distribution.

Usage:
  python 09_train_vq_spatial.py

Outputs:
  checkpoints/spatial_vq_encoder.pt    (when --model vq, default)
  checkpoints/spatial_pos_encoder.pt   (when --model pos_embed)
  plots/spatial_encoder_training.png
"""

import os

import gymnasium as gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import minigrid   # registers MiniGrid environments
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models import (
    VQSpatialEncoder,
    PositionEmbeddingEncoder,
    spatial_reconstruction_loss,
    SPATIAL_CODEBOOK_SIZE,
    SPATIAL_GRID_SIZE,
    SPATIAL_LATENT_DIM,
)


ENV_ID         = 'MiniGrid-DoorKey-5x5-v0'
N_EPISODES     = 600
MAX_STEPS      = 100
BASE_SEED      = 42
N_GRID_COPIES  = 500   # uniform augmentation: every cell repeated this many times

BATCH_SIZE     = 128
N_EPOCHS       = 40
LR             = 3e-4
DEVICE = (
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)


# ---------------------------------------------------------------------------
# Position collection
# ---------------------------------------------------------------------------

def collect_positions(env_id, n_episodes, max_steps, base_seed):
    """Roll out random policy and record integer agent positions (x, y)."""
    env = gym.make(env_id)
    positions = []

    for ep in range(n_episodes):
        env.reset(seed=base_seed + ep)
        for _ in range(max_steps):
            ax, ay = env.unwrapped.agent_pos
            positions.append([int(ax), int(ay)])
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

    env.close()
    return np.array(positions, dtype=np.int64)


def make_grid_augment(grid_size, n_copies):
    """Every (x, y) ∈ [0, grid_size)^2 repeated ``n_copies`` times."""
    cells = np.array(
        [[x, y] for x in range(grid_size) for y in range(grid_size)],
        dtype=np.int64,
    )
    return np.tile(cells, (n_copies, 1))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_spatial(model, loader, n_epochs):
    model = model.to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    hist  = {'recon': [], 'commit': [], 'perplexity': []}

    epoch_bar = tqdm(range(n_epochs), desc='VQ-spatial', unit='epoch')

    for epoch in epoch_bar:
        recon_list, commit_list, perp_list = [], [], []
        model.train()

        for (batch,) in loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()

            x_logits, y_logits, indices, commit = model(batch)
            recon = spatial_reconstruction_loss(x_logits, y_logits, batch)
            loss  = recon + commit
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            recon_list.append(recon.item())
            commit_list.append(commit.item())
            perp_list.append(model.vq.get_perplexity(indices))

        mean_recon  = float(np.mean(recon_list))
        mean_commit = float(np.mean(commit_list))
        mean_perp   = float(np.mean(perp_list))
        pct         = 100 * mean_perp / SPATIAL_CODEBOOK_SIZE

        hist['recon'].append(mean_recon)
        hist['commit'].append(mean_commit)
        hist['perplexity'].append(mean_perp)

        epoch_bar.set_postfix(
            recon=f'{mean_recon:.4f}',
            commit=f'{mean_commit:.4f}',
            perp=f'{mean_perp:.1f}/{SPATIAL_CODEBOOK_SIZE} ({pct:.0f}%)',
        )

    return hist



def train_pos_embed(model, loader, n_epochs, grid_size):
    """Pre-train PositionEmbeddingEncoder with a CE reconstruction objective."""
    model = model.to(DEVICE)
    decoder = nn.Linear(model.latent_dim, 2 * grid_size).to(DEVICE)
    opt = torch.optim.Adam(
        list(model.parameters()) + list(decoder.parameters()), lr=LR,
    )
    hist = {'recon': []}

    epoch_bar = tqdm(range(n_epochs), desc='PosEmbed', unit='epoch')
    for epoch in epoch_bar:
        recon_list = []
        model.train(); decoder.train()
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            z      = model.encode(batch)
            logits = decoder(z)
            loss   = spatial_reconstruction_loss(
                logits[:, :grid_size], logits[:, grid_size:], batch,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            recon_list.append(loss.item())
        mean_recon = float(np.mean(recon_list))
        hist['recon'].append(mean_recon)
        epoch_bar.set_postfix(recon=f'{mean_recon:.4f}')

    return hist

# ---------------------------------------------------------------------------
# Diagnostic — per-cell token assignments
# ---------------------------------------------------------------------------

def codebook_assignment_table(model, grid_size):
    """Print which token each grid cell maps to (sanity check for diversity)."""
    model.eval()
    cells = torch.tensor(
        [[x, y] for x in range(grid_size) for y in range(grid_size)],
        dtype=torch.long, device=next(model.parameters()).device,
    )
    with torch.no_grad():
        indices = model.get_token(cells)
    print("\nGrid cell → spatial token mapping:")
    grid = np.full((grid_size, grid_size), -1, dtype=int)
    for (xy, idx) in zip(cells.cpu().numpy(), indices.cpu().numpy()):
        grid[xy[1], xy[0]] = int(idx)
    for row in grid:
        print('   ' + '  '.join(f'{v:>3d}' for v in row))
    unique = np.unique(indices.cpu().numpy())
    print(f"\nUnique tokens used across {grid_size*grid_size} cells: "
          f"{len(unique)} ({sorted(unique.tolist())})")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training(hist, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Spatial VQ Encoder Pre-Training (embedding + cross-entropy)',
                 fontsize=12)

    ax = axes[0]
    ax.plot(hist['recon'],  label='Reconstruction (CE)', color='steelblue')
    ax.plot(hist['commit'], label='Commitment',          color='steelblue', linestyle='--')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(hist['perplexity'], color='steelblue')
    ax.axhline(y=SPATIAL_CODEBOOK_SIZE, color='red', linestyle='--',
               label=f'Max K={SPATIAL_CODEBOOK_SIZE}')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Codebook Perplexity')
    ax.set_title('Codebook Utilisation (exp(entropy))')
    ax.set_ylim(0, SPATIAL_CODEBOOK_SIZE + 5)
    ax.legend(fontsize=8)

    ax = axes[2]
    total = [r + c for r, c in zip(hist['recon'], hist['commit'])]
    ax.plot(total, color='steelblue')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Recon + Commitment')
    ax.set_title('Total Loss')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.close()



def plot_training_pos_embed(hist, save_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(hist['recon'], color='steelblue', label='Reconstruction (CE)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Position Embedding Pre-Training')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.close()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Pre-train a spatial encoder for Condition I.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--model', choices=['vq', 'pos_embed'], default='vq',
        help=(
            '"vq" trains VQSpatialEncoder (checkpoints/spatial_vq_encoder.pt); '
            '"pos_embed" trains PositionEmbeddingEncoder '
            '(checkpoints/spatial_pos_encoder.pt).'
        ),
    )
    args = parser.parse_args()

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    print(f"Device: {DEVICE}")

    print(f"Collecting positions from {N_EPISODES} random-policy episodes...")
    rollout = collect_positions(ENV_ID, N_EPISODES, MAX_STEPS, BASE_SEED)
    augment = make_grid_augment(SPATIAL_GRID_SIZE, N_GRID_COPIES)
    positions = np.concatenate([rollout, augment], axis=0)

    print(f"  random-policy positions: {len(rollout):,}")
    print(f"  grid-cell augmentation : {len(augment):,}  "
          f"({SPATIAL_GRID_SIZE*SPATIAL_GRID_SIZE} cells × {N_GRID_COPIES} copies)")
    print(f"  total                  : {len(positions):,}")
    uniq, cnt = np.unique(positions, axis=0, return_counts=True)
    print(f"  unique cells in mix    : {len(uniq)}  "
          f"(min={cnt.min()}, max={cnt.max()}, mean={cnt.mean():.0f})")

    dataset = TensorDataset(torch.from_numpy(positions))
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    if args.model == 'vq':
        print(f"\n--- Spatial VQ Encoder (K={SPATIAL_CODEBOOK_SIZE}) ---")
        model = VQSpatialEncoder()
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        hist  = train_spatial(model, loader, N_EPOCHS)

        model.cpu()
        torch.save(model.state_dict(), 'checkpoints/spatial_vq_encoder.pt')
        print("Saved checkpoints/spatial_vq_encoder.pt")

        final_perp = hist['perplexity'][-1]
        pct = 100 * final_perp / SPATIAL_CODEBOOK_SIZE
        print(f"Final codebook utilisation: {final_perp:.1f}/{SPATIAL_CODEBOOK_SIZE} ({pct:.0f}%)")
        if pct < 30:
            print("WARNING: codebook utilisation is low — collapse may still be present.")

        codebook_assignment_table(model, SPATIAL_GRID_SIZE)

        plot_training(hist, 'plots/spatial_encoder_training.png')
        print("\nNext: python 04_ppo_conditions.py --conditions I_Dual_VQ")

    else:  # pos_embed
        model = PositionEmbeddingEncoder()
        print(f"PositionEmbeddingEncoder — Parameters: {sum(p.numel() for p in model.parameters()):,}")
        hist  = train_pos_embed(model, loader, N_EPOCHS, SPATIAL_GRID_SIZE)
        model.cpu()
        torch.save(model.state_dict(), 'checkpoints/spatial_pos_encoder.pt')
        print("Saved checkpoints/spatial_pos_encoder.pt")
        print(f"Final reconstruction loss: {hist['recon'][-1]:.4f}")
        plot_training_pos_embed(hist, 'plots/spatial_encoder_training_pos_embed.png')
        print("\nNext: python 04_ppo_conditions.py --conditions I_Dual_PosEmbed")


if __name__ == '__main__':
    main()
