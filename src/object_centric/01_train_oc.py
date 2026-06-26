"""
Script OC-1 — Object-Centric Encoder Training
==============================================
Trains the ObjectCentricVQEncoder on the existing trajectory dataset.

The training objective combines:
  - Main reconstruction: concat(z_q_agent, z_q_key, z_q_door) → full 7×7×3 obs
  - Key auxiliary:  z_q_key  alone → key cell channels  (specialisation pressure)
  - Door auxiliary: z_q_door alone → door cell channels (specialisation pressure)
  - Commitment: sum of per-head VQ commitment losses

Training uses the same data as Script 02, so no new data collection is needed.

Usage (from project root or src/):
  python src/object_centric/01_train_oc.py

Requires:
  data/trajectories.pkl  (from src/01_collect_data.py)

Outputs:
  checkpoints/object_centric/encoder.pt
  plots/object_centric/encoder_training.png
"""

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
ROOT  = _HERE.parent.parent
sys.path.insert(0, str(_HERE))          # models_oc
sys.path.insert(0, str(_HERE.parent))   # models

from models_oc import (
    ObjectCentricVQEncoder,
    compute_training_loss,
    OC_AGENT_K, OC_KEY_K, OC_DOOR_K,
)

DATA_PATH = ROOT / 'data' / 'trajectories.pkl'
CKPT_DIR  = ROOT / 'checkpoints' / 'object_centric'
PLOT_DIR  = ROOT / 'plots' / 'object_centric'

BATCH_SIZE = 128
N_EPOCHS   = 80
LR         = 3e-4
DEVICE = (
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)


class ObsDataset(Dataset):
    def __init__(self, obs):
        self.obs = torch.tensor(obs, dtype=torch.long)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(model, loader, n_epochs):
    model = model.to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)

    keys = ('recon', 'key_aux', 'door_aux', 'commit',
            'perp_agent', 'perp_key', 'perp_door')
    hist = {k: [] for k in keys}

    epoch_bar = tqdm(range(n_epochs), desc='OC encoder', unit='epoch')

    for _ in epoch_bar:
        per_batch = {k: [] for k in keys}
        model.train()

        for batch in loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()

            logits, enc = model(batch)
            loss, breakdown = compute_training_loss(model, batch, logits, enc)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            for k, v in breakdown.items():
                per_batch[k].append(v)
            per_batch['perp_agent'].append(model.vq_agent.get_perplexity(enc['idx_agent']))
            per_batch['perp_key'].append(  model.vq_key.get_perplexity(  enc['idx_key']))
            per_batch['perp_door'].append( model.vq_door.get_perplexity( enc['idx_door']))

        for k in keys:
            hist[k].append(float(np.mean(per_batch[k])))

        epoch_bar.set_postfix(
            recon=f"{hist['recon'][-1]:.3f}",
            pa=f"{hist['perp_agent'][-1]:.1f}/{OC_AGENT_K}",
            pk=f"{hist['perp_key'][-1]:.1f}/{OC_KEY_K}",
            pd=f"{hist['perp_door'][-1]:.1f}/{OC_DOOR_K}",
        )

    return hist


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training(hist, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Object Centric VQ Encoder: Training Curves', fontsize=12)

    # Row 0: losses
    ax = axes[0, 0]
    ax.plot(hist['recon'],  label='Main recon',  color='steelblue')
    ax.plot(hist['commit'], label='Commitment',  color='steelblue', linestyle='--')
    ax.set_title('Reconstruction & Commitment'); ax.legend(fontsize=8)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')

    ax = axes[0, 1]
    ax.plot(hist['key_aux'],  label='Key cell aux',  color='darkorange')
    ax.plot(hist['door_aux'], label='Door cell aux', color='green')
    ax.set_title('Per-Head Auxiliary Losses'); ax.legend(fontsize=8)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')

    total = [r + ka + da + c for r, ka, da, c in
             zip(hist['recon'], hist['key_aux'], hist['door_aux'], hist['commit'])]
    ax = axes[0, 2]
    ax.plot(total, color='black')
    ax.set_title('Total Loss'); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')

    # Row 1: per-head perplexity
    head_cfg = [
        ('perp_agent', OC_AGENT_K, 'steelblue',   'Agent / CLS head'),
        ('perp_key',   OC_KEY_K,   'darkorange',  'Key head'),
        ('perp_door',  OC_DOOR_K,  'green',       'Door head'),
    ]
    for ax, (key, K, color, title) in zip(axes[1, :], head_cfg):
        ax.plot(hist[key], color=color)
        ax.axhline(K, color='red', linestyle='--', alpha=0.5, label=f'Max K={K}')
        ax.set_ylim(0, K + 2)
        ax.set_title(f'Perplexity: {title}')
        ax.set_xlabel('Epoch'); ax.set_ylabel('exp(H)')
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")

    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    dataset = ObsDataset(data['observations'])
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print(f"Dataset: {len(dataset):,} observations")

    model = ObjectCentricVQEncoder()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"  VQ heads — agent K={OC_AGENT_K}, key K={OC_KEY_K}, door K={OC_DOOR_K}")

    hist = train(model, loader, N_EPOCHS)

    model.cpu()
    ckpt_path = CKPT_DIR / 'encoder.pt'
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nSaved {ckpt_path}")

    print("\nFinal codebook utilisation:")
    for head, K in (('agent', OC_AGENT_K), ('key', OC_KEY_K), ('door', OC_DOOR_K)):
        p   = hist[f'perp_{head}'][-1]
        pct = 100 * p / K
        flag = ' ← LOW — possible collapse' if pct < 30 else ''
        print(f"  {head:<5} {p:.1f}/{K} ({pct:.0f}%){flag}")

    plot_training(hist, str(PLOT_DIR / 'encoder_training.png'))
    print("\nNext: python src/object_centric/02_probe_oc.py")


if __name__ == '__main__':
    main()
