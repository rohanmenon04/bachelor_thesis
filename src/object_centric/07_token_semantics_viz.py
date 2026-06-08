"""
Script OC-7 — Token Semantic Purity Visualization
==================================================
Generates a figure showing P(semantic_state=True | token k) for each
codebook entry in the key and door heads, demonstrating that the encoder
has learned hard semantic partitions in its discrete vocabulary.

Usage:
  python src/object_centric/07_token_semantics_viz.py

Output:
  plots/object_centric/token_semantics.png
"""

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

_HERE = Path(__file__).resolve().parent
ROOT  = _HERE.parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from models_oc import ObjectCentricVQEncoder, OC_KEY_K, OC_DOOR_K
from models import D_MODEL

CKPT_PATH = ROOT / 'checkpoints' / 'object_centric' / 'encoder.pt'
DATA_PATH = ROOT / 'data' / 'trajectories.pkl'
PLOT_DIR  = ROOT / 'plots' / 'object_centric'
BATCH_SIZE = 512

DEVICE = (
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)
    observations = data['observations']
    labels       = data['labels']
    carry_arr = np.array([l['carrying_key'] for l in labels])
    open_arr  = np.array([l['door_open']    for l in labels])

    model = ObjectCentricVQEncoder()
    model.load_state_dict(torch.load(CKPT_PATH, map_location='cpu'))
    model.eval().to(DEVICE)

    obs_t = torch.tensor(observations, dtype=torch.long)
    loader = DataLoader(TensorDataset(obs_t), batch_size=BATCH_SIZE, shuffle=False)

    key_all, door_all = [], []
    with torch.no_grad():
        for (batch,) in loader:
            enc = model.encode_all(batch.to(DEVICE))
            key_all.append(enc['idx_key'].cpu().numpy())
            door_all.append(enc['idx_door'].cpu().numpy())
    key_all  = np.concatenate(key_all)
    door_all = np.concatenate(door_all)

    # Compute conditional probabilities
    key_counts   = np.bincount(key_all,  minlength=OC_KEY_K)
    door_counts  = np.bincount(door_all, minlength=OC_DOOR_K)

    key_p_carry = np.array([
        carry_arr[key_all == k].mean() if key_counts[k] > 0 else 0.0
        for k in range(OC_KEY_K)
    ])
    door_p_open = np.array([
        open_arr[door_all == d].mean() if door_counts[d] > 0 else 0.0
        for d in range(OC_DOOR_K)
    ])

    # Sort by conditional probability for clarity
    key_order  = np.argsort(key_p_carry)[::-1]   # descending
    door_order = np.argsort(door_p_open)[::-1]

    # Plot
    fig, (ax_key, ax_door) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        'Object-Centric VQ Encoder — Per-Token Semantic Purity\n'
        'Object-head tokens are hard binary partitions, not soft associations',
        fontsize=12,
    )

    # --- Key head ---
    x_key = np.arange(OC_KEY_K)
    bars_k = ax_key.bar(x_key, key_p_carry[key_order],
                         color=['#e74c3c' if p > 0.5 else '#3498db'
                                for p in key_p_carry[key_order]],
                         alpha=0.85, width=0.7)
    ax_key.axhline(0.5, color='black', lw=1.0, linestyle='--', alpha=0.5, label='Decision boundary')
    ax_key.set_xticks(x_key)
    ax_key.set_xticklabels([str(key_order[i]) for i in x_key], fontsize=8)
    ax_key.set_xlabel('Key head token index (sorted by P(carrying))', fontsize=10)
    ax_key.set_ylabel('P(carrying_key = True)', fontsize=10)
    ax_key.set_ylim(-0.05, 1.1)
    ax_key.set_title(f'Key head (K={OC_KEY_K})\n'
                      f'Tokens with P=1.000: {sum(key_p_carry > 0.99)} "carrying"\n'
                      f'Tokens with P≈0.000: {sum(key_p_carry < 0.01)} "floor"',
                      fontsize=10)
    ax_key.legend(fontsize=9)
    ax_key.grid(axis='y', alpha=0.3)

    # Annotate hard-carrying tokens
    n_carry_tokens = sum(key_p_carry > 0.99)
    ax_key.annotate(
        f'{n_carry_tokens} tokens\n(100% carrying)',
        xy=(0, 1.0), xytext=(4, 0.85),
        fontsize=9, color='#e74c3c',
        arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.2),
    )

    # Count labels for each bar
    for i, ki in enumerate(key_order):
        n = key_counts[ki]
        ax_key.text(i, key_p_carry[ki] + 0.02, f'n={n}',
                    ha='center', va='bottom', fontsize=6, rotation=75, alpha=0.7)

    # --- Door head ---
    x_door = np.arange(OC_DOOR_K)
    bars_d = ax_door.bar(x_door, door_p_open[door_order],
                          color=['#e74c3c' if p > 0.5 else '#3498db'
                                 for p in door_p_open[door_order]],
                          alpha=0.85, width=0.7)
    ax_door.axhline(0.5, color='black', lw=1.0, linestyle='--', alpha=0.5, label='Decision boundary')
    ax_door.set_xticks(x_door)
    ax_door.set_xticklabels([str(door_order[i]) for i in x_door], fontsize=8)
    ax_door.set_xlabel('Door head token index (sorted by P(door open))', fontsize=10)
    ax_door.set_ylabel('P(door_open = True)', fontsize=10)
    ax_door.set_ylim(-0.05, 1.1)
    ax_door.set_title(f'Door head (K={OC_DOOR_K})\n'
                       f'Tokens with P=1.000: {sum(door_p_open > 0.99)} "open"\n'
                       f'Tokens with P≈0.000: {sum(door_p_open < 0.01)} "locked"',
                       fontsize=10)
    ax_door.legend(fontsize=9)
    ax_door.grid(axis='y', alpha=0.3)

    for i, di in enumerate(door_order):
        n = door_counts[di]
        ax_door.text(i, door_p_open[di] + 0.02, f'n={n}',
                     ha='center', va='bottom', fontsize=6, rotation=75, alpha=0.7)

    plt.tight_layout()
    save_path = str(PLOT_DIR / 'token_semantics.png')
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.close()

    # Print summary
    print("\n=== Token Purity Summary ===")
    print(f"Key head: {sum(key_p_carry > 0.99)} 'carrying' tokens, "
          f"{sum(key_p_carry < 0.01)} 'floor' tokens, "
          f"{sum((key_p_carry > 0.01) & (key_p_carry < 0.99))} mixed")
    print(f"Door head: {sum(door_p_open > 0.99)} 'open' tokens, "
          f"{sum(door_p_open < 0.01)} 'locked' tokens, "
          f"{sum((door_p_open > 0.01) & (door_p_open < 0.99))} mixed")


if __name__ == '__main__':
    main()
