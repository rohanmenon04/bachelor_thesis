"""
Script 5 — Token Visualization (Qualitative Interpretability)
==============================================================
For each active codebook entry, finds the observations that activate it
most frequently and shows them in a gallery grid.

If a token consistently activates for observations where "door_open=1"
and "carrying_key=0", that is qualitative evidence of semantic structure.

This supports RQ5 (interpretability) in the thesis: do tokens correspond
to visually and semantically coherent state clusters?

Usage:
  python 05_token_visualize.py

Requires:
  data/trajectories.pkl
  checkpoints/vq_encoder.pt

Outputs:
  plots/token_gallery.png
  plots/token_semantic_heatmap.png
"""

import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models import TransformerVQEncoder, CODEBOOK_SIZE

DEVICE      = ('cuda' if torch.cuda.is_available()
                else 'mps' if torch.backends.mps.is_available()
                else 'cpu')
BATCH_SIZE  = 512
N_NEIGHBORS = 5   # observations to show per token in the gallery
MAX_SHOW    = 16  # top-N most-used tokens to plot in gallery


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_all(model, observations):
    model.eval().to(DEVICE)
    ds     = TensorDataset(torch.tensor(observations, dtype=torch.long))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    idxs   = []
    with torch.no_grad():
        for (batch,) in loader:
            z = model.encode(batch.to(DEVICE))
            _, idx, _ = model.vq(z)
            idxs.append(idx.cpu().numpy())
    return np.concatenate(idxs)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def obs_to_rgb(obs):
    """
    Convert 7x7x3 integer observation to a simple RGB displayable image.
    Uses the type channel (0-10) as luminance and color channel for tint.
    """
    type_val  = obs[:, :, 0].astype(np.float32) / 10.0   # [0,1]
    color_val = obs[:, :, 1].astype(np.float32) / 5.0    # [0,1]

    # Map object types to rough colors for display
    r = np.clip(type_val * (1 - 0.3 * color_val), 0, 1)
    g = np.clip(type_val * (0.5 + 0.5 * color_val), 0, 1)
    b = np.clip(type_val * 0.8, 0, 1)

    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def semantic_summary(obs_indices, labels, label_keys):
    """Return mean positive rate for each label across a set of observations."""
    return {
        k: np.mean([labels[i][k] for i in obs_indices])
        for k in label_keys
    }


# ---------------------------------------------------------------------------
# Gallery plot
# ---------------------------------------------------------------------------

def plot_gallery(observations, labels, token_indices, top_tokens, save_path):
    label_keys = ['door_open', 'door_locked', 'carrying_key']
    n_rows = len(top_tokens)
    n_cols = N_NEIGHBORS + 1   # left column: token info text

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.4, n_rows * 1.5))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle('Token Gallery — Most-Activating Observations per Codebook Entry', fontsize=10)

    unique, counts = np.unique(token_indices, return_counts=True)
    count_map = dict(zip(unique, counts))

    for row_i, tok_id in enumerate(top_tokens):
        obs_idxs = np.where(token_indices == tok_id)[0]
        sem      = semantic_summary(obs_idxs[:50], labels, label_keys)

        # Left cell: token stats
        ax = axes[row_i, 0]
        summary = (
            f"Token {tok_id}\n"
            f"n={count_map.get(tok_id, 0)}\n"
            + "\n".join(f"{k}={v:.0%}" for k, v in sem.items())
        )
        ax.text(
            0.5, 0.5, summary,
            ha='center', va='center', fontsize=5.5,
            transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='#d5f5e3', alpha=0.7),
        )
        ax.axis('off')

        # Neighbour observation cells
        for col_i in range(1, n_cols):
            ax = axes[row_i, col_i]
            if col_i - 1 < len(obs_idxs):
                ax.imshow(obs_to_rgb(observations[obs_idxs[col_i - 1]]))
            ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=90, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Heatmap: token vs semantic label
# ---------------------------------------------------------------------------

def plot_semantic_heatmap(labels, token_indices, save_path):
    label_keys = ['door_open', 'door_locked', 'carrying_key']
    active_tokens = np.unique(token_indices)

    # Build matrix: (n_active, n_labels) = mean positive rate per token per label
    matrix = np.zeros((len(active_tokens), len(label_keys)))
    for i, tok in enumerate(active_tokens):
        mask = token_indices == tok
        for j, key in enumerate(label_keys):
            matrix[i, j] = np.mean([labels[k][key] for k in np.where(mask)[0]])

    fig, ax = plt.subplots(figsize=(8, max(4, len(active_tokens) * 0.25 + 2)))
    im = ax.imshow(matrix.T, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)

    ax.set_yticks(range(len(label_keys)))
    ax.set_yticklabels(label_keys, fontsize=9)
    ax.set_xticks(range(len(active_tokens)))
    ax.set_xticklabels(active_tokens, fontsize=6, rotation=90)
    ax.set_xlabel('Token Index (active codebook entries)')
    ax.set_title(
        'Semantic Heatmap: P(label=1 | token)\n'
        'Distinct columns indicate tokens specialised for specific states',
        fontsize=10,
    )

    plt.colorbar(im, ax=ax, label='Positive rate')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs('plots/edits02', exist_ok=True)

    with open('data/trajectories.pkl', 'rb') as f:
        data = pickle.load(f)
    observations = data['observations']
    labels       = data['labels']

    model = TransformerVQEncoder()
    model.load_state_dict(torch.load('checkpoints/edits01/vq_encoder.pt', map_location='cpu'))

    print("Encoding all observations...")
    token_indices = encode_all(model, observations)

    unique, counts = np.unique(token_indices, return_counts=True)
    order      = np.argsort(-counts)
    top_tokens = unique[order[:MAX_SHOW]]

    print(f"\nTop {MAX_SHOW} most-used tokens:")
    for tok, cnt in zip(unique[order[:MAX_SHOW]], counts[order[:MAX_SHOW]]):
        bar = '█' * int(cnt / len(observations) * 100)
        print(f"  Token {tok:3d}: {cnt:5d} obs  ({100*cnt/len(observations):.1f}%)  {bar}")

    plot_gallery(observations, labels, token_indices, top_tokens, 'plots/edits02/token_gallery.png')
    plot_semantic_heatmap(labels, token_indices, 'plots/edits02/token_semantic_heatmap.png')

    print("\nAll done. Plots saved to plots/edits02/")

if __name__ == '__main__':
    main()
