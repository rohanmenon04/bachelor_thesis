"""
Script OC-2 — Linear Probes for Object-Centric Tokens
======================================================
Tests the semantic alignment of each VQ head independently using the same
linear probe methodology as Script 03 (Alain & Bengio, 2017).

For each head (agent, key, door) and for the concatenation of all three:
  - One-hot encode the discrete token indices
  - Train logistic regression on 80% of observations
  - Report test accuracy and majority-class baseline

Key hypotheses:
  - Key head should outperform CLS-alone on 'carrying_key'
    (the key token reads the key cell directly; CLS must infer absence)
  - Door head should outperform CLS-alone on 'door_open' and 'door_locked'
  - Combined probe should meet or exceed the existing Condition C result

Usage:
  python src/object_centric/02_probe_oc.py

Requires:
  data/trajectories.pkl
  checkpoints/object_centric/encoder.pt

Outputs:
  plots/object_centric/probe_results.png
  plots/object_centric/codebook_usage.png
"""

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

_HERE = Path(__file__).resolve().parent
ROOT  = _HERE.parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from models_oc import ObjectCentricVQEncoder, OC_AGENT_K, OC_KEY_K, OC_DOOR_K

DATA_PATH = ROOT / 'data' / 'trajectories.pkl'
CKPT_PATH = ROOT / 'checkpoints' / 'object_centric' / 'encoder.pt'
PLOT_DIR  = ROOT / 'plots' / 'object_centric'

BATCH_SIZE = 512
LABEL_KEYS = ['door_open', 'door_locked', 'carrying_key']
DEVICE = (
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_all_heads(model, observations):
    """Run encode_all() over the full dataset. Returns per-head index arrays."""
    model.eval().to(DEVICE)
    loader = DataLoader(
        TensorDataset(torch.tensor(observations, dtype=torch.long)),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    agent_idxs, key_idxs, door_idxs = [], [], []

    with torch.no_grad():
        for (batch,) in loader:
            enc = model.encode_all(batch.to(DEVICE))
            agent_idxs.append(enc['idx_agent'].cpu().numpy())
            key_idxs.append(  enc['idx_key'].cpu().numpy())
            door_idxs.append( enc['idx_door'].cpu().numpy())

    return {
        'agent': np.concatenate(agent_idxs),
        'key':   np.concatenate(key_idxs),
        'door':  np.concatenate(door_idxs),
    }


def to_onehot(indices, n_classes):
    n = len(indices)
    X = np.zeros((n, n_classes), dtype=np.float32)
    X[np.arange(n), indices] = 1.0
    return X


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------

def run_probe(X, y):
    if len(np.unique(y)) < 2:
        return None, None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_tr, y_tr)
    acc      = accuracy_score(y_te, clf.predict(X_te))
    baseline = max(y_te.mean(), 1 - y_te.mean())
    return float(acc), float(baseline)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_probe_results(results, save_path):
    """
    results : {'agent': {label: (acc, base)}, 'key': ..., 'door': ..., 'all': ...}
    """
    heads  = ['agent', 'key', 'door', 'all']
    labels = LABEL_KEYS
    n_l, n_h = len(labels), len(heads)

    fig, axes = plt.subplots(n_l, n_h, figsize=(3.5 * n_h, 3 * n_l), sharey=True)
    fig.suptitle('Object-Centric Heads — Linear Probe Accuracy', fontsize=12)

    for hi, head in enumerate(heads):
        axes[0, hi].set_title(head, fontsize=10, fontweight='bold')

    for li, label in enumerate(labels):
        axes[li, 0].set_ylabel(label, fontsize=9)
        for hi, head in enumerate(heads):
            ax = axes[li, hi]
            res = results[head][label]
            if res[0] is None:
                ax.text(0.5, 0.5, 'single class', ha='center',
                        va='center', transform=ax.transAxes, fontsize=8)
                ax.set_ylim(0, 1)
                continue
            acc, base = res
            bar_vals   = [base, acc]
            bar_colors = ['#cccccc', '#4c72b0']
            bars = ax.bar(['Base', 'Probe'], bar_vals, color=bar_colors, width=0.5)
            ax.set_ylim(0, 1.05)
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{bar.get_height():.3f}',
                    ha='center', va='bottom', fontsize=7,
                )

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.close()


def plot_codebook_usage(idxs, save_path):
    head_cfg = [
        ('agent', OC_AGENT_K, '#4c72b0'),
        ('key',   OC_KEY_K,   '#dd8452'),
        ('door',  OC_DOOR_K,  '#55a868'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('Codebook Usage per Head', fontsize=11)

    for ax, (head, K, color) in zip(axes, head_cfg):
        counts = np.bincount(idxs[head], minlength=K)
        active = int((counts > 0).sum())
        ax.bar(range(K), counts, color=color, alpha=0.85, width=0.8)
        ax.axhline(len(idxs[head]) / K, color='red', linestyle='--',
                   alpha=0.6, label='Uniform usage')
        ax.set_title(f'{head} head — {active}/{K} active')
        ax.set_xlabel('Token index'); ax.set_ylabel('Count')
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.close()


def plot_label_separation(idxs, label_arrays, save_path):
    """
    For each label, show per-head token histograms split by label value.
    Equivalent to the semantic heatmap in Script 05 but per head.
    """
    heads     = ['agent', 'key', 'door']
    head_K    = {'agent': OC_AGENT_K, 'key': OC_KEY_K, 'door': OC_DOOR_K}
    head_col  = {'agent': '#4c72b0',  'key': '#dd8452', 'door': '#55a868'}

    n_l, n_h = len(LABEL_KEYS), len(heads)
    fig, axes = plt.subplots(n_l, n_h, figsize=(5 * n_h, 3.5 * n_l))
    fig.suptitle('Token Distribution Split by Semantic Label', fontsize=11)

    for hi, head in enumerate(heads):
        axes[0, hi].set_title(head, fontsize=10, fontweight='bold')

    for li, label in enumerate(LABEL_KEYS):
        y = label_arrays[label]
        axes[li, 0].set_ylabel(label, fontsize=9)
        for hi, head in enumerate(heads):
            ax = axes[li, hi]
            K  = head_K[head]
            for val, col, lbl in [(0, '#4c9be8', f'{label}=0'),
                                  (1, '#e74c3c', f'{label}=1')]:
                mask = y == val
                if mask.sum() == 0:
                    continue
                ax.hist(idxs[head][mask], bins=K, range=(0, K),
                        alpha=0.55, color=col, label=lbl, density=True)
            ax.set_xlabel('Token index')
            ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    observations = data['observations']
    label_arrays = {k: np.array([l[k] for l in data['labels']]) for k in LABEL_KEYS}

    model = ObjectCentricVQEncoder()
    model.load_state_dict(torch.load(CKPT_PATH, map_location='cpu'))

    print("Encoding all observations...")
    idxs = encode_all_heads(model, observations)

    # Build feature matrices: one-hot per head + concatenated
    X = {
        'agent': to_onehot(idxs['agent'], OC_AGENT_K),
        'key':   to_onehot(idxs['key'],   OC_KEY_K),
        'door':  to_onehot(idxs['door'],  OC_DOOR_K),
    }
    X['all'] = np.concatenate([X['agent'], X['key'], X['door']], axis=1)

    print(f"\n{'Label':<15} {'Head':<8} {'Acc':>7} {'Baseline':>9} {'Δ':>7}")
    print('─' * 52)

    results = {head: {} for head in X}
    for head, feats in X.items():
        for label in LABEL_KEYS:
            acc, base = run_probe(feats, label_arrays[label])
            results[head][label] = (acc, base)
            if acc is not None:
                print(f"{label:<15} {head:<8} {acc:7.4f} {base:9.4f} {acc - base:+7.4f}")
        print()

    plot_probe_results(results, str(PLOT_DIR / 'probe_results.png'))
    plot_codebook_usage(idxs, str(PLOT_DIR / 'codebook_usage.png'))
    plot_label_separation(idxs, label_arrays, str(PLOT_DIR / 'label_separation.png'))

    print("\nNext: python src/object_centric/03_ppo_oc.py")


if __name__ == '__main__':
    main()
