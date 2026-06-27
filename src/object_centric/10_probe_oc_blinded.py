"""
Script OC-10 — Blinded-Head Probe Ablation
===========================================
Tests whether semantic alignment in the key/door heads requires those object
types to be visible in the Transformer input.

Blinding procedure:
  Before calling encode_all(), every cell whose type channel equals OC_KEY_TYPE
  (5) or OC_DOOR_TYPE (4) is replaced with 0 (floor/empty). The Transformer
  never sees key or door cells. Because _object_readout searches the (now
  masked) observation for those type IDs, it never finds them — both the key
  head and the door head always fall back to their learned absent_embed vector.

Two conditions are compared head-to-head:
  Normal  — standard encode_all() on the real observation
  Blinded — encode_all() on the type-masked observation

Expected result:
  Normal key head on carrying_key  : ~0.997
  Blinded key head on carrying_key : ~majority baseline (~0.665)

Usage:
  python src/object_centric/10_probe_oc_blinded.py

Requires:
  data/trajectories.pkl
  checkpoints/object_centric/encoder.pt

Outputs:
  plots/object_centric/blinded_probe_comparison.png
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

from models_oc import (
    ObjectCentricVQEncoder,
    OC_AGENT_K, OC_KEY_K, OC_DOOR_K,
    OC_KEY_TYPE, OC_DOOR_TYPE,
)

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

def encode_all_heads(model, observations, blinded=False):
    """
    Run encode_all() over the dataset.

    blinded=True: zero out key (type 5) and door (type 4) cells in the
    observation before encoding. _object_readout cannot find these types
    anymore, so both heads always fall back to their absent_embed.
    """
    model.eval().to(DEVICE)
    loader = DataLoader(
        TensorDataset(torch.tensor(observations, dtype=torch.long)),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    agent_idxs, key_idxs, door_idxs = [], [], []

    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            if blinded:
                batch = batch.clone()
                type_ch = batch[..., 0]               # (B, 7, 7) view
                type_ch[type_ch == OC_KEY_TYPE]  = 0
                type_ch[type_ch == OC_DOOR_TYPE] = 0
            enc = model.encode_all(batch)
            agent_idxs.append(enc['idx_agent'].cpu().numpy())
            key_idxs.append(  enc['idx_key'].cpu().numpy())
            door_idxs.append( enc['idx_door'].cpu().numpy())

    return {
        'agent': np.concatenate(agent_idxs),
        'key':   np.concatenate(key_idxs),
        'door':  np.concatenate(door_idxs),
    }


def to_onehot(indices, n_classes):
    X = np.zeros((len(indices), n_classes), dtype=np.float32)
    X[np.arange(len(indices)), indices] = 1.0
    return X


def build_features(idxs):
    X = {
        'agent': to_onehot(idxs['agent'], OC_AGENT_K),
        'key':   to_onehot(idxs['key'],   OC_KEY_K),
        'door':  to_onehot(idxs['door'],  OC_DOOR_K),
    }
    X['all'] = np.concatenate([X['agent'], X['key'], X['door']], axis=1)
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


def probe_all(X, label_arrays):
    results = {head: {} for head in X}
    for head, feats in X.items():
        for label in LABEL_KEYS:
            results[head][label] = run_probe(feats, label_arrays[label])
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(normal_results, blinded_results, save_path):
    """
    Side-by-side grouped bar chart: for each (head, label) pair, show
    majority baseline, normal probe accuracy, and blinded probe accuracy.
    """
    heads  = ['agent', 'key', 'door', 'all']
    labels = LABEL_KEYS

    fig, axes = plt.subplots(
        len(labels), len(heads),
        figsize=(3.8 * len(heads), 3.2 * len(labels)),
        sharey=True,
    )
    fig.suptitle(
        'Blinded vs Normal — Linear Probe Accuracy\n'
        '(blinded: key/door types masked to 0 before encoding)',
        fontsize=11,
    )

    for hi, head in enumerate(heads):
        axes[0, hi].set_title(head, fontsize=10, fontweight='bold')

    for li, label in enumerate(labels):
        axes[li, 0].set_ylabel(label, fontsize=9)
        for hi, head in enumerate(heads):
            ax = axes[li, hi]

            n_acc, n_base = normal_results[head][label]
            b_acc, _      = blinded_results[head][label]

            if n_acc is None:
                ax.text(0.5, 0.5, 'single class', ha='center',
                        va='center', transform=ax.transAxes, fontsize=8)
                ax.set_ylim(0, 1)
                continue

            vals   = [n_base, n_acc, b_acc if b_acc is not None else 0.0]
            colors = ['#cccccc', '#4c72b0', '#dd8452']
            xlbls  = ['Base', 'Normal', 'Blinded']
            bars   = ax.bar(xlbls, vals, color=colors, width=0.55)
            ax.set_ylim(0, 1.10)

            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{bar.get_height():.3f}',
                    ha='center', va='bottom', fontsize=7,
                )

            # Annotate the drop on the blinded bar
            if b_acc is not None and n_acc is not None:
                delta = b_acc - n_acc
                ax.text(
                    2, b_acc - 0.06,
                    f'{delta:+.3f}',
                    ha='center', va='top', fontsize=7,
                    color='#cc0000' if delta < 0 else '#007700',
                )

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

    print("Encoding — normal pass...")
    idxs_normal  = encode_all_heads(model, observations, blinded=False)

    print("Encoding — blinded pass (key/door types zeroed)...")
    idxs_blinded = encode_all_heads(model, observations, blinded=True)

    # Token usage sanity check: blinded key/door heads should be degenerate
    n_active_key_normal  = int((np.bincount(idxs_normal['key'],  minlength=OC_KEY_K)  > 0).sum())
    n_active_key_blinded = int((np.bincount(idxs_blinded['key'], minlength=OC_KEY_K)  > 0).sum())
    n_active_door_normal  = int((np.bincount(idxs_normal['door'],  minlength=OC_DOOR_K) > 0).sum())
    n_active_door_blinded = int((np.bincount(idxs_blinded['door'], minlength=OC_DOOR_K) > 0).sum())

    print(f"\nActive tokens — key  head: normal={n_active_key_normal}/{OC_KEY_K},  "
          f"blinded={n_active_key_blinded}/{OC_KEY_K}")
    print(f"Active tokens — door head: normal={n_active_door_normal}/{OC_DOOR_K}, "
          f"blinded={n_active_door_blinded}/{OC_DOOR_K}")
    print("(blinded key/door should collapse to 1 active token — the absent_embed code)")

    X_normal  = build_features(idxs_normal)
    X_blinded = build_features(idxs_blinded)

    normal_results  = probe_all(X_normal,  label_arrays)
    blinded_results = probe_all(X_blinded, label_arrays)

    # Print comparison table
    print(f"\n{'Label':<15} {'Head':<8} {'Base':>6} {'Normal':>8} {'Blinded':>9} {'Δ':>8}")
    print('─' * 62)
    for head in ['agent', 'key', 'door', 'all']:
        for label in LABEL_KEYS:
            n_acc, n_base = normal_results[head][label]
            b_acc, _      = blinded_results[head][label]
            if n_acc is None:
                continue
            delta = (b_acc - n_acc) if b_acc is not None else float('nan')
            b_str = f'{b_acc:.4f}' if b_acc is not None else '  N/A'
            print(f"{label:<15} {head:<8} {n_base:6.4f} {n_acc:8.4f} {b_str:>9} {delta:+8.4f}")
        print()

    plot_comparison(
        normal_results, blinded_results,
        save_path=str(PLOT_DIR / 'blinded_probe_comparison.png'),
    )


if __name__ == '__main__':
    main()
