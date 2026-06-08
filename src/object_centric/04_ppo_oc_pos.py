"""
Script OC-4 — PPO with Object-Centric Tokens + Oracle Position
==============================================================
Tests the hypothesis that the spatial-information gap (not the semantic
quality of the OC tokens) is the primary reason OC-All fails to converge
efficiently.

Conditions run here:
  OC-All+Pos : frozen OC encoder (agent+key+door tokens, 192-dim)
               + oracle agent position (normalised col/row, 2-dim)
               → 194-dim flat feature vector, MultiInputPolicy

Existing results loaded for comparison:
  Raw:      results/object_centric/ppo_returns.npy  (2 seeds × 200k)
  OC-Agent: same file                               (5 seeds × 200k, all zeros)
  OC-All:   same file                               (5 seeds × 500k)

New results appended to:
  results/object_centric/ppo_returns.npy  (key: 'OC-All+Pos')
  plots/object_centric/ppo_comparison_final.png

Thesis narrative:
  Probe result  → OC tokens have genuine semantic alignment
                  (key head: 0.997 for carrying_key)
  OC-All result → semantic tokens partially support learning but
                  spatial ambiguity limits convergence
  OC-All+Pos    → semantic tokens + position = complete state description
                  → clean convergence; validates that semantic tokens
                  carry real value beyond noise

Usage:
  python src/object_centric/04_ppo_oc_pos.py [--n_seeds N] [--total_steps N]
"""

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import minigrid
import numpy as np
import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

_HERE = Path(__file__).resolve().parent
ROOT  = _HERE.parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from models_oc import ObjectCentricVQEncoder, OC_AGENT_K, OC_KEY_K, OC_DOOR_K
from models import D_MODEL

CKPT_PATH  = ROOT / 'checkpoints' / 'object_centric' / 'encoder.pt'
RESULT_DIR = ROOT / 'results' / 'object_centric'
PLOT_DIR   = ROOT / 'plots' / 'object_centric'

ENV_ID      = 'MiniGrid-DoorKey-5x5-v0'
GRID_SIZE   = 5          # DoorKey-5x5: positions in [0, GRID_SIZE)
TOTAL_STEPS = 200_000
EVAL_EVERY  =   5_000
N_EVAL_EPS  =      10
N_SEEDS     =       5


# ---------------------------------------------------------------------------
# Environment wrappers
# ---------------------------------------------------------------------------

class ImgPosObsWrapper(gym.ObservationWrapper):
    """
    Returns a Dict observation:
        'image' : (7, 7, 3) float32 — partial egocentric view
        'pos'   : (2,) float32      — (col, row) normalised to [0, 1]

    Position is read from env.unwrapped.agent_pos (oracle ground truth).
    MiniGrid's default partial view is egocentric and does NOT render the
    agent in its own FOV, so the image alone carries no world-frame position.
    """

    def __init__(self, env):
        super().__init__(env)
        img = env.observation_space['image']
        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(
                low=0.0, high=float(img.high.max()),
                shape=img.shape, dtype=np.float32,
            ),
            'pos': gym.spaces.Box(
                low=0.0, high=1.0, shape=(2,), dtype=np.float32,
            ),
        })
        self._norm = float(GRID_SIZE - 1)   # 4.0

    def observation(self, obs):
        ax, ay = self.env.unwrapped.agent_pos
        return {
            'image': obs['image'].astype(np.float32),
            'pos':   np.array([ax / self._norm, ay / self._norm],
                               dtype=np.float32),
        }


def make_env_pos(seed=0):
    env = gym.make(ENV_ID)
    env = ImgPosObsWrapper(env)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------

def _load_encoder():
    enc = ObjectCentricVQEncoder()
    enc.load_state_dict(torch.load(CKPT_PATH, map_location='cpu'))
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


class OCFullPosExtractor(BaseFeaturesExtractor):
    """
    Frozen OC encoder (agent + key + door tokens, 192-dim)
    concatenated with oracle agent position (2-dim) → 194-dim.

    Features dim breakdown:
      agent token  : D_MODEL = 64
      key token    : D_MODEL = 64
      door token   : D_MODEL = 64
      position     :         =  2
      total        :         = 194

    The position signal directly resolves the spatial-information gap
    identified in OC-All: tokens encode WHAT (semantic sub-task state)
    while position encodes WHERE (grid location). Together they form
    a complete, interpretable state description for DoorKey-5x5.
    """

    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=3 * D_MODEL + 2)
        self.enc = _load_encoder()

    def forward(self, obs):
        image = obs['image']   # (B, 7, 7, 3) float32
        pos   = obs['pos']     # (B, 2)       float32
        with torch.no_grad():
            enc = self.enc.encode_all(image.long())
        tokens = torch.cat(
            [enc['z_q_agent'], enc['z_q_key'], enc['z_q_door']], dim=-1
        )                                                           # (B, 192)
        return torch.cat([tokens, pos], dim=-1)                    # (B, 194)

    def train(self, mode=True):
        super().train(mode)
        self.enc.eval()   # prevent SB3 from re-enabling EMA updates
        return self


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_condition_pos(seed, total_steps):
    env      = make_env_pos(seed)
    eval_env = make_env_pos(seed + 100)

    policy_kwargs = {
        'features_extractor_class':  OCFullPosExtractor,
        'features_extractor_kwargs': {},
        'net_arch':                  [64, 64],
    }

    model = PPO(
        'MultiInputPolicy', env,
        policy_kwargs=policy_kwargs,
        n_steps=512, batch_size=64, n_epochs=4,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.05, vf_coef=0.5, max_grad_norm=0.5,
        learning_rate=3e-4, verbose=0,
    )

    returns = []
    for _ in range(total_steps // EVAL_EVERY):
        model.learn(total_timesteps=EVAL_EVERY, reset_num_timesteps=False,
                    progress_bar=False)
        mean_r, _ = evaluate_policy(model, eval_env, n_eval_episodes=N_EVAL_EPS,
                                    deterministic=True)
        returns.append(float(mean_r))

    env.close()
    eval_env.close()
    return returns


# ---------------------------------------------------------------------------
# Plotting — combined with existing conditions
# ---------------------------------------------------------------------------

CONDITION_STYLE = {
    'Raw':         {'color': 'black',       'label': 'Raw obs (147-dim)',              'lw': 2.0},
    'OC-Agent':    {'color': 'steelblue',   'label': 'OC agent token only (64-dim)',   'lw': 1.5, 'ls': '--'},
    'OC-All':      {'color': 'darkorange',  'label': 'OC all tokens (192-dim)',        'lw': 1.5},
    'OC-All+Pos':  {'color': '#2ecc71',     'label': 'OC all tokens + position (194-dim)', 'lw': 2.0},
}


def plot_combined(results, eval_every, save_path):
    fig, ax = plt.subplots(figsize=(11, 6))

    for name, arr in results.items():
        style = CONDITION_STYLE.get(name, {})
        color = style.get('color', 'grey')
        label = style.get('label', name)
        lw    = style.get('lw', 1.5)
        ls    = style.get('ls', '-')

        mean   = arr.mean(0)
        std    = arr.std(0)
        n_ckpt = arr.shape[1]
        x = np.arange(1, n_ckpt + 1) * eval_every

        ax.plot(x, mean, label=f'{label}  ({mean[-1]:.3f}±{std[-1]:.3f})',
                color=color, lw=lw, linestyle=ls)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=color)

    ax.set_xlabel('Environment steps', fontsize=12)
    ax.set_ylabel('Mean return (10 eval episodes)', fontsize=12)
    ax.set_title(
        'Object-Centric VQ Tokens — PPO Sample Efficiency\n'
        'MiniGrid-DoorKey-5x5  |  Spatial information gap analysis',
        fontsize=11,
    )
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--n_seeds',     type=int, default=N_SEEDS)
    parser.add_argument('--total_steps', type=int, default=TOTAL_STEPS)
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    result_path = RESULT_DIR / 'ppo_returns.npy'
    results = (
        dict(np.load(result_path, allow_pickle=True).item())
        if result_path.exists() else {}
    )

    if results:
        print("Existing results:")
        for k, v in results.items():
            m = v.mean(0)
            print(f"  {k:<12}  {v.shape}  final={m[-1]:.3f}")
    print()

    # ----- OC-All+Pos -----
    print(f"=== OC-All+Pos | {args.n_seeds} seeds × {args.total_steps:,} steps ===")
    runs = []
    for seed in range(args.n_seeds):
        print(f"  seed {seed} ...", end=' ', flush=True)
        r = train_condition_pos(seed, args.total_steps)
        runs.append(r)
        print(f"final={r[-1]:.3f}  max={max(r):.3f}")

    arr = np.array(runs)
    results['OC-All+Pos'] = arr
    print(f"\n  OC-All+Pos: mean={arr.mean(0)[-1]:.3f}  std={arr.std(0)[-1]:.3f}")

    np.save(result_path, results)
    print(f"\nSaved {result_path}")

    print('\n--- Final Performance (last checkpoint) ---')
    for name in ['Raw', 'OC-Agent', 'OC-All', 'OC-All+Pos']:
        if name in results:
            v = results[name]
            m = v.mean(0)
            s = v.std(0)
            steps_k = v.shape[1] * EVAL_EVERY // 1000
            print(f"  {name:<14}  {m[-1]:.3f} ± {s[-1]:.3f}  ({steps_k}k steps, {v.shape[0]} seeds)")

    plot_combined(results, EVAL_EVERY, str(PLOT_DIR / 'ppo_comparison_final.png'))


if __name__ == '__main__':
    main()
