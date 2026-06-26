"""
Script OC-5 — Final Analysis and Thesis-Quality Figures
========================================================
Generates all final figures from the object-centric experiment results.

Figures produced:
  1. ppo_comparison_full.png   — learning curves at full training budget
  2. ppo_comparison_200k.png   — fair comparison, all truncated to 200k steps
  3. probe_summary.png         — per-head semantic alignment (bar chart)
  4. combined_evidence.png     — two-panel: probe + PPO (main thesis figure)

Usage:
  python src/object_centric/05_final_analysis.py

Requires:
  results/object_centric/ppo_returns.npy   (incl. OC-All+Pos)
  checkpoints/object_centric/encoder.pt
  data/trajectories.pkl
"""

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
from models import D_MODEL

RESULT_PATH = ROOT / 'results' / 'object_centric' / 'ppo_returns.npy'
CKPT_PATH   = ROOT / 'checkpoints' / 'object_centric' / 'encoder.pt'
DATA_PATH   = ROOT / 'data' / 'trajectories.pkl'
PLOT_DIR    = ROOT / 'plots' / 'object_centric'
EVAL_EVERY  = 5_000
BATCH_SIZE  = 512

DEVICE = (
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)

LABEL_KEYS = ['door_open', 'door_locked', 'carrying_key']

CONDITION_STYLE = {
    'Raw':          {'color': 'black',   'label': 'Raw obs (147-dim)',                    'lw': 2.0, 'ls': '-'},
    'OC-Agent':     {'color': '#4c9be8', 'label': 'OC-Agent: CLS token (64-dim)',         'lw': 1.5, 'ls': '--'},
    'OC-All':       {'color': '#f4a261', 'label': 'OC-All: semantic only (192-dim)',       'lw': 1.5, 'ls': '-'},
    'OC-All+Pos':   {'color': '#aaaaaa', 'label': 'OC-All+Pos: oracle position (194-dim)', 'lw': 1.5, 'ls': '--'},
    'OC-All-Disc':  {'color': '#2ecc71', 'label': 'OC-All-Disc: fully discrete (224-dim)', 'lw': 2.0, 'ls': '-'},
}
PLOT_ORDER = ['Raw', 'OC-Agent', 'OC-All', 'OC-All+Pos', 'OC-All-Disc']


# ---------------------------------------------------------------------------
# Probe helpers
# ---------------------------------------------------------------------------

def encode_all_heads(model, observations):
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
    X = np.zeros((len(indices), n_classes), dtype=np.float32)
    X[np.arange(len(indices)), indices] = 1.0
    return X


def run_probe(X, y):
    if len(np.unique(y)) < 2:
        return None, None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_tr, y_tr)
    return float(accuracy_score(y_te, clf.predict(X_te))), float(max(y_te.mean(), 1 - y_te.mean()))


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _plot_curves(ax, results, budget_ckpts=None):
    """
    Plot learning curves for all conditions.
    budget_ckpts: if set, truncate all conditions to this many checkpoints.
    """
    for name in PLOT_ORDER:
        if name not in results:
            continue
        arr  = results[name]
        if budget_ckpts is not None:
            arr = arr[:, :budget_ckpts]

        style = CONDITION_STYLE.get(name, {})
        mean  = arr.mean(0)
        std   = arr.std(0)
        n_ck  = arr.shape[1]
        x     = np.arange(1, n_ck + 1) * EVAL_EVERY

        final_m = mean[-1]
        final_s = std[-1]
        n_seeds = arr.shape[0]
        steps_k = n_ck * EVAL_EVERY // 1000

        label = (f"{style.get('label', name)}  "
                 f"[{final_m:.3f}±{final_s:.3f}, "
                 f"n={n_seeds}, {steps_k}k steps]")
        ax.plot(x, mean, label=label,
                color=style.get('color', 'grey'),
                lw=style.get('lw', 1.5),
                linestyle=style.get('ls', '-'))
        ax.fill_between(x, mean - std, mean + std, alpha=0.12,
                        color=style.get('color', 'grey'))

    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel('Mean Return (10 eval eps, det)', fontsize=10)
    ax.legend(fontsize=8.5, loc='upper right')
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Figure 1: Full budget learning curves
# ---------------------------------------------------------------------------

def plot_learning_curves_full(results, save_path):
    fig, ax = plt.subplots(figsize=(11, 6))
    _plot_curves(ax, results, budget_ckpts=None)
    ax.set_xlabel('Environment Steps', fontsize=12)
    ax.set_title(
        'Object-Centric VQ Tokens: PPO Sample Efficiency\n'
        'MiniGrid-DoorKey-5x5, OC-All shown at 500k steps',
        fontsize=11,
    )

    # Annotate spatial gap arrow
    if 'OC-All' in results:
        oc_all_arr = results['OC-All']
        oc_mean_200k = oc_all_arr[:, :40].mean(0)[-1]   # mean at 200k
        ax.annotate(
            '← spatial\n   info gap',
            xy=(200_000, max(oc_mean_200k, 0.02)), xytext=(240_000, 0.25),
            fontsize=8.5, color='#f4a261',
            arrowprops=dict(arrowstyle='->', color='#f4a261', lw=1.2),
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 2: Fair comparison at 200k steps
# ---------------------------------------------------------------------------

def plot_learning_curves_200k(results, save_path):
    budget_ckpts = 40   # 40 × 5000 = 200k

    fig, ax = plt.subplots(figsize=(11, 6))
    _plot_curves(ax, results, budget_ckpts=budget_ckpts)
    ax.set_xlabel('Environment Steps', fontsize=12)
    ax.set_title(
        'Object-Centric VQ Tokens — PPO Sample Efficiency (200k step budget)\n'
        'MiniGrid-DoorKey-5x5  |  Fair comparison: all conditions at same budget',
        fontsize=11,
    )

    # Vertical line at budget
    ax.axvline(200_000, color='grey', lw=0.8, linestyle=':', alpha=0.7)
    ax.text(195_000, 0.95, '200k', ha='right', fontsize=8, color='grey')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 3: Probe summary bar chart
# ---------------------------------------------------------------------------

def plot_probe_summary(probe_results, save_path):
    heads  = ['agent', 'key', 'door', 'all']
    head_labels = {
        'agent': 'Agent head\n(CLS, K=64)',
        'key':   'Key head\n(cell, K=16)',
        'door':  'Door head\n(cell, K=8)',
        'all':   'Combined\n(all three)',
    }
    head_colors = {
        'agent': '#4c72b0', 'key': '#dd8452', 'door': '#55a868', 'all': '#8172b2',
    }

    fig, axes = plt.subplots(1, len(LABEL_KEYS), figsize=(13, 5), sharey=True)
    fig.suptitle(
        'Object-Centric VQ Encoder: Linear Probe Accuracy per Head\n'
        'One-hot token indices, logistic regression (80/20 split)',
        fontsize=11,
    )

    for ax, label in zip(axes, LABEL_KEYS):
        ax.set_title(label.replace('_', ' ').title(), fontsize=10, fontweight='bold')
        ax.set_ylim(0, 1.07)
        ax.set_xticks(range(len(heads)))
        ax.set_xticklabels([head_labels[h] for h in heads], fontsize=8)
        ax.axhline(1.0, color='black', lw=0.5, alpha=0.2)
        ax.grid(axis='y', alpha=0.25)

        for i, head in enumerate(heads):
            res = probe_results[head][label]
            if res[0] is None:
                continue
            acc, base = res
            color = head_colors[head]
            ax.bar(i - 0.18, base, width=0.32, color='#cccccc', alpha=0.9)
            ax.bar(i + 0.18, acc,  width=0.32, color=color, alpha=0.85)
            ax.text(i + 0.18, acc  + 0.01, f'{acc:.3f}',
                    ha='center', va='bottom', fontsize=7.5, fontweight='bold')
            ax.text(i - 0.18, base + 0.01, f'{base:.3f}',
                    ha='center', va='bottom', fontsize=7, color='#666666')

        if ax is axes[0]:
            ax.set_ylabel('Accuracy', fontsize=10)

    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor='#cccccc', label='Majority class baseline'),
        Patch(facecolor='#4c72b0', label='Agent (CLS)'),
        Patch(facecolor='#dd8452', label='Key (cell readout)'),
        Patch(facecolor='#55a868', label='Door (cell readout)'),
        Patch(facecolor='#8172b2', label='Combined'),
    ]
    fig.legend(handles=legend_items, loc='lower center', ncol=5,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 4: Combined evidence (main thesis figure)
# ---------------------------------------------------------------------------

def plot_combined_evidence(probe_results, ppo_results, save_path):
    fig = plt.figure(figsize=(14, 5.5))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1, 1.8], figure=fig)
    gs.update(wspace=0.35)

    # --- Left: carrying_key probe per head ---
    ax_left = fig.add_subplot(gs[0])
    heads       = ['agent', 'key', 'door', 'all']
    head_short  = ['Agent\n(CLS)', 'Key\n(cell)', 'Door\n(cell)', 'Combined']
    head_colors = ['#4c72b0', '#dd8452', '#55a868', '#8172b2']
    label       = 'carrying_key'

    accs  = [probe_results[h][label][0] for h in heads]
    bases = [probe_results[h][label][1] for h in heads]

    x = np.arange(len(heads))
    ax_left.bar(x - 0.2, bases, width=0.35, color='#cccccc', alpha=0.9, label='Majority baseline')
    ax_left.bar(x + 0.2, accs,  width=0.35, color=head_colors, alpha=0.85, label='Probe accuracy')
    for xi, acc in zip(x, accs):
        ax_left.text(xi + 0.2, acc + 0.01, f'{acc:.3f}',
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax_left.set_xticks(x)
    ax_left.set_xticklabels(head_short, fontsize=9)
    ax_left.set_ylabel('Accuracy', fontsize=10)
    ax_left.set_ylim(0, 1.10)
    ax_left.set_title('Semantic alignment:\n"carrying_key" linear probe', fontsize=10)
    ax_left.axhline(bases[0], color='#888888', lw=1.0, linestyle='--', alpha=0.6,
                    label=f'Baseline {bases[0]:.3f}')
    ax_left.legend(fontsize=8, loc='upper left')
    ax_left.grid(axis='y', alpha=0.3)

    ax_left.annotate(
        'Direct object-cell\nreadout → near-perfect',
        xy=(1 + 0.2, accs[1]), xytext=(2.1, 0.80),
        fontsize=7.5, color='#dd8452',
        arrowprops=dict(arrowstyle='->', color='#dd8452', lw=1.2),
    )

    # --- Right: PPO curves (fair 200k comparison) ---
    ax_right = fig.add_subplot(gs[1])
    budget_ckpts = 40  # 200k steps

    for name in PLOT_ORDER:
        if name not in ppo_results:
            continue
        arr   = ppo_results[name][:, :budget_ckpts]
        style = CONDITION_STYLE.get(name, {})
        mean  = arr.mean(0)
        std   = arr.std(0)
        x2    = np.arange(1, arr.shape[1] + 1) * EVAL_EVERY
        final_m = mean[-1]; final_s = std[-1]
        label_str = (f"{style.get('label', name)}\n"
                     f"[{final_m:.3f}±{final_s:.3f}]")
        ax_right.plot(x2, mean, label=label_str,
                      color=style.get('color', 'grey'),
                      lw=style.get('lw', 1.5),
                      linestyle=style.get('ls', '-'))
        ax_right.fill_between(x2, mean - std, mean + std, alpha=0.12,
                               color=style.get('color', 'grey'))

    ax_right.set_xlabel('Environment Steps', fontsize=10)
    ax_right.set_ylabel('Mean Return', fontsize=10)
    ax_right.set_title('Policy learning — 200k steps\nMiniGrid-DoorKey-5x5', fontsize=10)
    ax_right.set_ylim(-0.05, 1.05)
    ax_right.legend(fontsize=8, loc='upper left', ncol=2)
    ax_right.grid(True, alpha=0.3)

    fig.suptitle(
        'Object-Centric Discrete State Tokens: Semantic Alignment and Policy Learning',
        fontsize=12, fontweight='bold',
    )

    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    if not RESULT_PATH.exists():
        print(f"ERROR: {RESULT_PATH} not found. Run experiments first.")
        return

    ppo_results = dict(np.load(RESULT_PATH, allow_pickle=True).item())

    # Check and print summary
    print("PPO results loaded:")
    for k in PLOT_ORDER:
        if k not in ppo_results:
            print(f"  {k:<14}  (missing)")
            continue
        v = ppo_results[k]
        m200 = v[:, :40].mean(0)[-1] if v.shape[1] >= 40 else v.mean(0)[-1]
        s200 = v[:, :40].std(0)[-1]  if v.shape[1] >= 40 else v.std(0)[-1]
        print(f"  {k:<14}  shape={str(v.shape):<12}  "
              f"@200k: {m200:.3f}±{s200:.3f}  "
              f"final: {v.mean(0)[-1]:.3f}±{v.std(0)[-1]:.3f}")

    # Probes
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)
    observations = data['observations']
    label_arrays = {k: np.array([l[k] for l in data['labels']]) for k in LABEL_KEYS}

    model = ObjectCentricVQEncoder()
    model.load_state_dict(torch.load(CKPT_PATH, map_location='cpu'))
    print(f"\nEncoding {len(observations):,} observations...")
    idxs = encode_all_heads(model, observations)

    X = {
        'agent': to_onehot(idxs['agent'], OC_AGENT_K),
        'key':   to_onehot(idxs['key'],   OC_KEY_K),
        'door':  to_onehot(idxs['door'],  OC_DOOR_K),
    }
    X['all'] = np.concatenate([X['agent'], X['key'], X['door']], axis=1)

    probe_results = {h: {} for h in X}
    print(f"\n{'Label':<15} {'Head':<8} {'Acc':>7} {'Baseline':>9} {'Δ':>7}")
    print('─' * 52)
    for head, feats in X.items():
        for label in LABEL_KEYS:
            acc, base = run_probe(feats, label_arrays[label])
            probe_results[head][label] = (acc, base)
            if acc is not None:
                print(f"{label:<15} {head:<8} {acc:7.4f} {base:9.4f} {acc - base:+7.4f}")
        print()

    # Generate figures
    print("Generating figures...")
    plot_learning_curves_full(ppo_results, str(PLOT_DIR / 'ppo_comparison_full.png'))
    plot_learning_curves_200k(ppo_results, str(PLOT_DIR / 'ppo_comparison_200k.png'))
    plot_probe_summary(probe_results, str(PLOT_DIR / 'probe_summary.png'))
    plot_combined_evidence(probe_results, ppo_results, str(PLOT_DIR / 'combined_evidence.png'))

    # Print thesis summary table
    print("\n" + "=" * 65)
    print("THESIS RESULTS SUMMARY")
    print("=" * 65)
    print("\n--- Semantic Alignment (carrying_key linear probe) ---")
    for head in ['agent', 'key', 'door', 'all']:
        acc, base = probe_results[head]['carrying_key']
        tag = ' ← object-cell readout' if head == 'key' else ''
        print(f"  {head:<6} head: {acc:.4f}  (baseline {base:.4f},  Δ={acc-base:+.4f}){tag}")

    print("\n--- Policy Learning at 200k Steps (mean ± std, n seeds) ---")
    for name in PLOT_ORDER:
        if name not in ppo_results:
            continue
        v = ppo_results[name]
        n200 = min(40, v.shape[1])
        arr200 = v[:, :n200]
        m, s = arr200.mean(0)[-1], arr200.std(0)[-1]
        n_seeds = v.shape[0]
        oc_all_note = ' (full 500k shown in ppo_comparison_full.png)' if name == 'OC-All' else ''
        print(f"  {name:<14}  {m:.3f} ± {s:.3f}  ({n_seeds} seeds){oc_all_note}")

    if 'OC-All' in ppo_results and 'OC-All+Pos' in ppo_results:
        oc_all_m = ppo_results['OC-All'][:, :40].mean(0)[-1]
        oc_pos_m = ppo_results['OC-All+Pos'][:, :40].mean(0)[-1]
        if oc_all_m > 0:
            print(f"\n  Position signal improvement: "
                  f"{oc_pos_m:.3f} / {oc_all_m:.3f} = {oc_pos_m/oc_all_m:.1f}× at 200k")


if __name__ == '__main__':
    main()
