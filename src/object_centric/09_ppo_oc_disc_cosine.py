"""
Script OC-9 — OC-All-Disc with Cosine LR Schedule
===================================================
Identical to OC-8 (fully discrete OC representation) except the PPO
learning rate follows a cosine annealing schedule rather than being
held constant at 3e-4.

Purpose: diagnostic — does the seed-level instability in OC-All-Disc
(0.768 ± 0.158 across 5 seeds) stem from a representation problem or
an optimisation problem? A cosine schedule decays the learning rate
smoothly to near-zero, preventing the optimiser from overshooting once
the policy begins converging. If results tighten, the instability is
an optimisation artefact. If results are similar or worse, the variance
is inherent to the representation.

Cosine schedule (SB3 convention: progress 1.0 → 0.0):
  lr(p) = lr_min + (lr_max - lr_min) * 0.5 * (1 + cos(π * (1 - p)))
  At p=1.0 (start): lr = lr_max = 3e-4
  At p=0.0 (end):   lr = lr_min = 1e-5

All other hyperparameters are identical to OC-8 to isolate the effect.

Outputs (separate dirs, existing results untouched):
  results/object_centric_cosine/ppo_returns.npy
  plots/object_centric_cosine/ppo_comparison_cosine.png

Usage:
  python src/object_centric/09_ppo_oc_disc_cosine.py [--n_seeds N] [--total_steps N]
"""

import argparse
import math
import sys
from pathlib import Path

import gymnasium as gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import minigrid
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

_HERE = Path(__file__).resolve().parent
ROOT  = _HERE.parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from models_oc import ObjectCentricVQEncoder
from models import PositionEmbeddingEncoder, D_MODEL, SPATIAL_LATENT_DIM, SPATIAL_GRID_SIZE

OC_CKPT      = ROOT / 'checkpoints' / 'object_centric' / 'encoder.pt'
SPATIAL_CKPT = ROOT / 'checkpoints' / 'spatial_pos_encoder.pt'

# Separate output dirs — existing results/plots are never touched
RESULT_DIR = ROOT / 'results' / 'object_centric_cosine'
PLOT_DIR   = ROOT / 'plots'   / 'object_centric_cosine'

ENV_ID      = 'MiniGrid-DoorKey-5x5-v0'
GRID_SIZE   = SPATIAL_GRID_SIZE   # 5
TOTAL_STEPS = 200_000
EVAL_EVERY  =   5_000
N_EVAL_EPS  =      10
N_SEEDS     =       5

FEATURES_DIM = 3 * D_MODEL + SPATIAL_LATENT_DIM   # 224

LR_MAX = 3e-4   # same starting LR as OC-8
LR_MIN = 1e-5   # cosine anneals down to this


def _cosine_lr(step: int, total_steps: int) -> float:
    """Cosine LR based on absolute step count, not SB3's progress fraction."""
    progress = step / total_steps          # 0 → 1 over training
    return LR_MIN + (LR_MAX - LR_MIN) * 0.5 * (1.0 + math.cos(math.pi * progress))


class CosineScheduleCallback(BaseCallback):
    """
    Sets the optimiser LR directly at each step using absolute step count.

    SB3's built-in LR callable uses progress_remaining relative to
    total_timesteps in learn(), which resets every chunked call and
    collapses the schedule to LR_MIN after the first chunk. This callback
    bypasses that by reading self.num_timesteps (absolute, monotonic) and
    writing directly to the optimizer param groups.
    """

    def __init__(self, total_steps: int):
        super().__init__()
        self.total_steps = total_steps

    def _on_step(self) -> bool:
        lr = _cosine_lr(self.num_timesteps, self.total_steps)
        for pg in self.model.policy.optimizer.param_groups:
            pg['lr'] = lr
        return True


# ---------------------------------------------------------------------------
# Environment  (identical to OC-8)
# ---------------------------------------------------------------------------

class ImgPosObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        img = env.observation_space['image']
        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(0.0, float(img.high.max()), img.shape, np.float32),
            'pos':   gym.spaces.Box(0.0, 1.0, (2,), np.float32),
        })
        self._norm = float(GRID_SIZE - 1)

    def observation(self, obs):
        ax, ay = self.env.unwrapped.agent_pos
        return {
            'image': obs['image'].astype(np.float32),
            'pos':   np.array([ax / self._norm, ay / self._norm], dtype=np.float32),
        }


def make_env(seed=0):
    env = gym.make(ENV_ID)
    env = ImgPosObsWrapper(env)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


# ---------------------------------------------------------------------------
# Feature extractor  (identical to OC-8 — same frozen encoders)
# ---------------------------------------------------------------------------

class OCAllDiscExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=FEATURES_DIM)

        self.enc = ObjectCentricVQEncoder()
        self.enc.load_state_dict(torch.load(OC_CKPT, map_location='cpu'))
        for p in self.enc.parameters():
            p.requires_grad_(False)
        self.enc.eval()

        self.pos_enc = PositionEmbeddingEncoder()
        self.pos_enc.load_state_dict(torch.load(SPATIAL_CKPT, map_location='cpu'))
        for p in self.pos_enc.parameters():
            p.requires_grad_(False)
        self.pos_enc.eval()

    def forward(self, obs):
        image = obs['image']
        pos   = obs['pos']
        with torch.no_grad():
            enc      = self.enc.encode_all(image.long())
            semantic = torch.cat(
                [enc['z_q_agent'], enc['z_q_key'], enc['z_q_door']], dim=-1
            )
            pos_int = (pos * (GRID_SIZE - 1)).round().long()
            e_pos   = self.pos_enc.encode(pos_int)
        return torch.cat([semantic, e_pos], dim=-1)

    def train(self, mode=True):
        super().train(mode)
        self.enc.eval()
        self.pos_enc.eval()
        return self


# ---------------------------------------------------------------------------
# Training — cosine LR
# ---------------------------------------------------------------------------

def train_seed(seed, total_steps):
    env      = make_env(seed)
    eval_env = make_env(seed + 100)

    policy_kwargs = {
        'features_extractor_class':  OCAllDiscExtractor,
        'features_extractor_kwargs': {},
        'net_arch':                  [64, 64],
    }

    # Start at LR_MAX; the callback overwrites it every step with the correct
    # cosine value. Passing LR_MAX here just initialises the optimizer.
    model = PPO(
        'MultiInputPolicy', env,
        policy_kwargs=policy_kwargs,
        n_steps=512, batch_size=64, n_epochs=4,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.05, vf_coef=0.5, max_grad_norm=0.5,
        learning_rate=LR_MAX,   # initial value only; callback takes over
        verbose=0,
    )

    cosine_cb = CosineScheduleCallback(total_steps)

    returns = []
    for _ in range(total_steps // EVAL_EVERY):
        model.learn(total_timesteps=EVAL_EVERY, reset_num_timesteps=False,
                    callback=cosine_cb, progress_bar=False)
        mean_r, _ = evaluate_policy(model, eval_env, n_eval_episodes=N_EVAL_EPS,
                                    deterministic=True)
        returns.append(float(mean_r))

    env.close()
    eval_env.close()
    return returns


# ---------------------------------------------------------------------------
# Plotting — cosine vs constant comparison
# ---------------------------------------------------------------------------

def plot_results(cosine_arr, baseline_arr, save_path):
    """Overlay cosine-schedule run against the existing constant-LR baseline."""
    fig, ax = plt.subplots(figsize=(11, 5.5))

    n_ckpts = cosine_arr.shape[1]
    x = np.arange(1, n_ckpts + 1) * EVAL_EVERY

    for arr, color, label_prefix, ls in [
        (cosine_arr,  '#e74c3c', 'OC-All-Disc (cosine LR)', '-'),
        (baseline_arr,'#2ecc71', 'OC-All-Disc (constant LR)', '--'),
    ]:
        arr = arr[:, :n_ckpts]
        mean = arr.mean(0)
        std  = arr.std(0)
        label = f"{label_prefix}  [{mean[-1]:.3f}±{std[-1]:.3f}, n={arr.shape[0]}]"
        ax.plot(x, mean, label=label, color=color, lw=2.0, linestyle=ls)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=color)

        # Individual seed traces (thin, same colour)
        for seed_r in arr:
            ax.plot(x, seed_r, color=color, lw=0.5, alpha=0.25)

    ax.set_xlabel('Environment Steps', fontsize=12)
    ax.set_ylabel('Mean Return (10 eval episodes)', fontsize=12)
    ax.set_title(
        'OC-All-Disc: Cosine vs Constant Learning Rate\n'
        'MiniGrid-DoorKey-5x5  ·  224-dim fully discrete  ·  5 seeds × 200k steps',
        fontsize=11,
    )
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)

    # LR schedule inset
    axin = ax.inset_axes([0.68, 0.08, 0.28, 0.30])
    p_vals = np.linspace(1, 0, 200)
    lr_vals = [_cosine_lr(int((1 - p) * TOTAL_STEPS), TOTAL_STEPS) for p in p_vals]
    steps   = (1 - p_vals) * TOTAL_STEPS
    axin.plot(steps, lr_vals, color='#e74c3c', lw=1.5)
    axin.axhline(LR_MAX, color='#2ecc71', lw=1.0, linestyle='--', alpha=0.8)
    axin.set_xlabel('Steps', fontsize=6.5)
    axin.set_ylabel('LR', fontsize=6.5)
    axin.set_title('LR schedule', fontsize=7)
    axin.tick_params(labelsize=5.5)
    axin.yaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
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

    print(f"OC-All-Disc Cosine LR: {args.n_seeds} seeds × {args.total_steps:,} steps")
    print(f"  LR: {LR_MAX:.0e} → {LR_MIN:.0e} (cosine)")
    print(f"  features_dim = {FEATURES_DIM}\n")

    runs = []
    for seed in range(args.n_seeds):
        print(f"  seed {seed} ...", end=' ', flush=True)
        r = train_seed(seed, args.total_steps)
        runs.append(r)
        print(f"final={r[-1]:.3f}  max={max(r):.3f}")

    cosine_arr = np.array(runs)
    np.save(RESULT_DIR / 'ppo_returns.npy', cosine_arr)
    print(f"\nSaved {RESULT_DIR / 'ppo_returns.npy'}")

    # Summary
    finals = cosine_arr[:, -1]
    peaks  = cosine_arr.max(axis=1)
    bs     = int((peaks >= 0.5).sum())
    print(f"\n=== Cosine LR Results ===")
    print(f"  Final:     {finals.mean():.3f} ± {finals.std():.3f}")
    print(f"  Mean peak: {peaks.mean():.3f} ± {peaks.std():.3f}")
    print(f"  Bootstrap: {bs}/{args.n_seeds}")
    for i, (f, p) in enumerate(zip(finals, peaks)):
        print(f"    seed {i}: final={f:.4f}  peak={p:.4f}")

    # Load existing constant-LR baseline for comparison plot
    baseline_path = ROOT / 'results' / 'object_centric' / 'ppo_returns.npy'
    if baseline_path.exists():
        existing = np.load(baseline_path, allow_pickle=True).item()
        if 'OC-All-Disc' in existing:
            baseline_arr = existing['OC-All-Disc']
            b_finals = baseline_arr[:, -1]
            print(f"\n=== Constant LR Baseline (existing) ===")
            print(f"  Final:     {b_finals.mean():.3f} ± {b_finals.std():.3f}")
            print(f"  Bootstrap: {int((baseline_arr.max(axis=1) >= 0.5).sum())}/{baseline_arr.shape[0]}")
            print(f"\n=== Delta (cosine - constant) ===")
            print(f"  Mean final: {finals.mean() - b_finals.mean():+.3f}")
            print(f"  Std:        {finals.std() - b_finals.std():+.3f}")

            plot_results(cosine_arr, baseline_arr,
                        str(PLOT_DIR / 'ppo_comparison_cosine.png'))
        else:
            print("  (OC-All-Disc key not found in baseline — skipping comparison plot)")
    else:
        print("  (No baseline file found — skipping comparison plot)")


if __name__ == '__main__':
    main()
