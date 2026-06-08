"""
Script 3 — Evaluate Representation Quality
==================================================
Tests whether the learned representations encode semantically meaningful
environment state by training a linear (logistic regression) classifier on
top of frozen representations to predict each ground-truth label.

Following Alain & Bengio (2017): a linear probe specifically tests whether
information is *linearly decodable* from the representation, which is the
standard interpretability criterion.

Three representation types are compared:
  Raw:        flattened + normalised observation pixels (upper-bound reference)
  Continuous: frozen pre-trained continuous AE latent vector
  VQ (ours):  one-hot over the discrete token index from the VQ encoder

If the VQ probe accuracy exceeds the continuous probe accuracy and is
significantly above the majority-class baseline, the tokens are encoding
semantically meaningful state information.

Usage:
  python 03_evaluate_representations.py

Requires:
  data/trajectories.pkl
  checkpoints/semantic_encoder.pt
  checkpoints/continuous_encoder.pt

Outputs:
  plots/linear_probe.png
  plots/codebook_usage.png
"""

import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

from models import (
    TransformerVQEncoder,
    ContinuousEncoder,
    CODEBOOK_SIZE,
    LATENT_DIM,
)

DEVICE     = ('cuda' if torch.cuda.is_available()
               else 'mps' if torch.backends.mps.is_available()
               else 'cpu')
BATCH_SIZE = 512
LABEL_KEYS = ['door_open', 'door_locked', 'carrying_key']


# ---------------------------------------------------------------------------
# Encoding utilities
# ---------------------------------------------------------------------------

def _make_loader(observations):
    ds = TensorDataset(torch.tensor(observations, dtype=torch.long))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)


def encode_vq(model, observations):
    """Return discrete token indices for every observation."""
    model.eval().to(DEVICE)
    indices = []
    with torch.no_grad():
        for (batch,) in _make_loader(observations):
            z = model.encode(batch.to(DEVICE))
            _, idx, _ = model.vq(z)
            indices.append(idx.cpu().numpy())
    return np.concatenate(indices)


def encode_continuous(model, observations):
    """Return continuous latent vectors for every observation."""
    model.eval().to(DEVICE)
    vecs = []
    with torch.no_grad():
        for (batch,) in _make_loader(observations):
            z = model.encode(batch.to(DEVICE))
            vecs.append(z.cpu().numpy())
    return np.concatenate(vecs)


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------

def run_probe(X, y):
    """
    Train a logistic regression on (X, y) and return test accuracy
    alongside the majority-class baseline.
    Returns (accuracy, baseline) or (None, None) if only one class exists.
    """
    if len(np.unique(y)) < 2:
        return None, None

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_tr, y_tr)
    acc      = accuracy_score(y_te, clf.predict(X_te))
    baseline = max(y_te.mean(), 1 - y_te.mean())
    return acc, baseline


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_probe_results(results, label_keys, save_path):
    valid = [k for k in label_keys if results[k]['vq'] is not None]
    if not valid:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Linear Probe: Semantic Alignment of Learned Representations', fontsize=12)

    ax = axes[0]
    x      = np.arange(len(valid))
    width  = 0.2
    conds  = ['baseline', 'raw', 'continuous', 'vq']
    colors = ['#aaaaaa', '#4c9be8', '#f4a261', '#2ecc71']
    names  = ['Majority baseline', 'Raw (upper bound)', 'Continuous AE', 'VQ-Transformer (ours)']

    for i, (cond, col, nm) in enumerate(zip(conds, colors, names)):
        vals = [results[k][cond] or 0 for k in valid]
        ax.bar(x + i * width, vals, width, label=nm, color=col, alpha=0.85)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(valid, rotation=10, ha='right')
    ax.set_ylabel('Classification Accuracy')
    ax.set_ylim(0, 1.05)
    ax.set_title('Probe Accuracy per Semantic Label')
    ax.legend(fontsize=8)
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.close()


def plot_codebook_usage(vq_indices, labels, label_keys, save_path):
    fig, axes = plt.subplots(1, len(label_keys) + 1, figsize=(16, 4))
    fig.suptitle('Codebook Usage and Semantic Separation', fontsize=11)

    ax = axes[0]
    unique, counts = np.unique(vq_indices, return_counts=True)
    ax.bar(unique, counts, color='#2ecc71', alpha=0.8)
    ax.axhline(
        y=len(vq_indices) / CODEBOOK_SIZE,
        color='red', linestyle='--', label='Uniform usage',
    )
    ax.set_xlabel('Token Index'); ax.set_ylabel('Count')
    ax.set_title(f'Overall Usage\n({len(unique)}/{CODEBOOK_SIZE} active)')
    ax.legend(fontsize=8)

    for ax, key in zip(axes[1:], label_keys):
        y = np.array([l[key] for l in labels])
        if len(np.unique(y)) < 2:
            ax.set_title(f'{key}\n(single class)')
            ax.axis('off')
            continue
        for val, col, lbl in [(0, '#4c9be8', f'{key}=0'), (1, '#e74c3c', f'{key}=1')]:
            mask = y == val
            ax.hist(vq_indices[mask], bins=CODEBOOK_SIZE, alpha=0.5, color=col, label=lbl)
        ax.set_xlabel('Token Index'); ax.set_ylabel('Count')
        ax.set_title(f'Token distribution\nby "{key}"')
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs('plots/ppo_v1', exist_ok=True)

    with open('data/trajectories.pkl', 'rb') as f:
        data = pickle.load(f)
    observations = data['observations']
    labels       = data['labels']

    # Load models
    vq_model   = TransformerVQEncoder()
    vq_model.load_state_dict(torch.load('checkpoints/semantic_encoder.pt',         map_location='cpu'))

    cont_model = ContinuousEncoder()
    cont_model.load_state_dict(torch.load('checkpoints/continuous_encoder.pt', map_location='cpu'))

    print("Encoding observations...")
    vq_indices = encode_vq(vq_model, observations)
    cont_z     = encode_continuous(cont_model, observations)

    n_active = len(np.unique(vq_indices))
    print(f"VQ: {n_active}/{CODEBOOK_SIZE} codebook entries active")

    # Feature matrices
    X_vq   = np.eye(CODEBOOK_SIZE, dtype=np.float32)[vq_indices]                      # (N, K)  one-hot
    X_cont = cont_z                                                                    # (N, LATENT_DIM)
    X_raw  = observations.reshape(len(observations), -1).astype(np.float32) / 10.0    # (N, 7*7*3) normalised

    # Run probes
    print(f"\n{'Label':<18} {'Baseline':>10} {'Raw':>10} {'Continuous':>12} {'VQ (ours)':>12}")
    print('-' * 66)

    results = {k: {'baseline': None, 'raw': None, 'continuous': None, 'vq': None}
               for k in LABEL_KEYS}

    for key in LABEL_KEYS:
        y = np.array([l[key] for l in labels])

        acc_raw,  baseline = run_probe(X_raw,  y)
        acc_cont, _        = run_probe(X_cont, y)
        acc_vq,   _        = run_probe(X_vq,   y)

        if baseline is None:
            print(f"  {key:<18} skipped (single class in dataset)")
            continue

        results[key] = {
            'baseline':   baseline,
            'raw':        acc_raw,
            'continuous': acc_cont,
            'vq':         acc_vq,
        }
        print(
            f"  {key:<18} {baseline:>10.3f} {acc_raw:>10.3f}"
            f" {acc_cont:>12.3f} {acc_vq:>12.3f}"
        )

    plot_probe_results(results, LABEL_KEYS, 'plots/ppo_v1/linear_probe.png')
    plot_codebook_usage(vq_indices, labels, LABEL_KEYS, 'plots/ppo_v1/codebook_usage.png')

    print("\nNext: python 04_ppo_comparison.py  (may take ~15-30 min)")


if __name__ == '__main__':
    main()
