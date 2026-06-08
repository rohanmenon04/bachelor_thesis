"""
Script OC-8 — Fully Discrete OC Representation: Semantic + Spatial Tokens
==========================================================================
Combines the object-centric VQ encoder (3 semantic tokens) with a discrete
position embedding encoder (1 spatial token) to form a fully discrete
state representation — no continuous oracle information anywhere.

Architecture:
  obs (7×7×3) → frozen OC encoder  → e_agent (64) + e_key (64) + e_door (64)
  agent_pos   → frozen PosEmbed    → e_pos (32)
                                      ↓
               concat → 224-dim fully discrete → [64, 64] MLP → π

Semantic tokens (3 × VQ): each is a codebook lookup from a discrete integer
  index produced by a VQ bottleneck — the standard VQ-VAE mechanism.

Spatial token (1 × embedding lookup): position is already a discrete
  integer (grid cell index 0–24), so a direct embedding table is used
  rather than VQ. This is the natural analogue: the VQ mechanism is needed
  to discretize *continuous* observations; the position space is already
  discrete. Each of the 25 grid cells maps to a unique token index,
  guaranteeing zero quantization error and no codebook collapse.

Comparison:
  OC-All+Pos (194-dim)  : semantic tokens + raw float position  (oracle)
  OC-All-Disc (224-dim) : semantic tokens + discrete position token (fully discrete)

If OC-All-Disc ≈ OC-All+Pos the discretization of position is lossless.
If OC-All-Disc < OC-All+Pos the embedding needs more capacity or training.

Usage:
  python src/object_centric/08_ppo_oc_full_discrete.py [--n_seeds N] [--total_steps N]

Requires:
  checkpoints/object_centric/encoder.pt   (from OC-1)
  checkpoints/spatial_pos_encoder.pt      (from 09_train_spatial_encoder --model pos_embed)

Outputs:
  results/object_centric/ppo_returns.npy  (key: 'OC-All-Disc')
  plots/object_centric/ppo_comparison_discrete.png
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

from stable_baselines3 import PPO
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
RESULT_DIR   = ROOT / 'results' / 'object_centric'
PLOT_DIR     = ROOT / 'plots' / 'object_centric'

ENV_ID      = 'MiniGrid-DoorKey-5x5-v0'
GRID_SIZE   = SPATIAL_GRID_SIZE   # 5
TOTAL_STEPS = 200_000
EVAL_EVERY  =   5_000
N_EVAL_EPS  =      10
N_SEEDS     =       5

FEATURES_DIM = 3 * D_MODEL + SPATIAL_LATENT_DIM   # 3*64 + 32 = 224


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class ImgPosObsWrapper(gym.ObservationWrapper):
    """Dict obs: {'image': (7,7,3) float32, 'pos': (2,) float32 in [0,1]}."""

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
# Feature extractor — fully discrete
# ---------------------------------------------------------------------------

class OCAllDiscExtractor(BaseFeaturesExtractor):
    """
    Fully discrete state representation: 3 OC semantic tokens + 1 spatial token.

    Semantic component (3 VQ tokens):
      - ObjectCentricVQEncoder: agent/key/door heads, each a VQ codebook lookup
      - Frozen weights

    Spatial component (1 position token):
      - PositionEmbeddingEncoder: maps flat cell index (0–24) to a 32-dim vector
      - The index IS the discrete token — no VQ needed since the input is already
        an integer grid cell (position is inherently discrete)
      - Frozen weights

    Both encoders produce embeddings of type float32 (standard codebook vectors),
    but every dimension of the policy input is derived exclusively through
    discrete integer indices. No continuous oracle information is passed.

    features_dim = 3 × D_MODEL + SPATIAL_LATENT_DIM = 192 + 32 = 224
    """

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
        image = obs['image']   # (B, 7, 7, 3)  float32 from SB3 preprocessing
        pos   = obs['pos']     # (B, 2)         float32 in [0, 1]

        # Semantic tokens — VQ bottleneck, all three heads
        with torch.no_grad():
            enc     = self.enc.encode_all(image.long())
            semantic = torch.cat(
                [enc['z_q_agent'], enc['z_q_key'], enc['z_q_door']], dim=-1
            )   # (B, 192)

            # Spatial token — direct embedding lookup (position already discrete)
            # Convert normalised float back to integer grid coordinates
            pos_int = (pos * (GRID_SIZE - 1)).round().long()   # (B, 2) ∈ [0, 4]
            e_pos   = self.pos_enc.encode(pos_int)             # (B, 32)

        return torch.cat([semantic, e_pos], dim=-1)   # (B, 224)

    def train(self, mode=True):
        super().train(mode)
        self.enc.eval()
        self.pos_enc.eval()
        return self


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_seed(seed, total_steps):
    env      = make_env(seed)
    eval_env = make_env(seed + 100)

    policy_kwargs = {
        'features_extractor_class':  OCAllDiscExtractor,
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
# Plotting
# ---------------------------------------------------------------------------

CONDITION_STYLE = {
    'Raw':         {'color': 'black',   'label': 'Raw obs (147-dim)',                   'lw': 2.0, 'ls': '-'},
    'OC-Agent':    {'color': '#4c9be8', 'label': 'OC-Agent: CLS token only (64-dim)',   'lw': 1.5, 'ls': '--'},
    'OC-All':      {'color': '#f4a261', 'label': 'OC-All: no position (192-dim)',        'lw': 1.5, 'ls': '-'},
    'OC-All+Pos':  {'color': '#aaaaaa', 'label': 'OC-All+Pos: oracle position (194-dim)','lw': 1.5, 'ls': '--'},
    'OC-All-Disc': {'color': '#2ecc71', 'label': 'OC-All-Disc: fully discrete (224-dim)','lw': 2.0, 'ls': '-'},
}
PLOT_ORDER = ['Raw', 'OC-Agent', 'OC-All', 'OC-All+Pos', 'OC-All-Disc']


def plot_results(results, save_path, budget_ckpts=40):
    fig, ax = plt.subplots(figsize=(12, 6))

    for name in PLOT_ORDER:
        if name not in results:
            continue
        arr  = results[name][:, :budget_ckpts]
        style = CONDITION_STYLE.get(name, {})
        mean = arr.mean(0);  std = arr.std(0)
        x    = np.arange(1, arr.shape[1] + 1) * EVAL_EVERY
        label = (f"{style.get('label', name)}  "
                 f"[{mean[-1]:.3f}±{std[-1]:.3f}, n={arr.shape[0]}]")
        ax.plot(x, mean, label=label,
                color=style.get('color', 'grey'),
                lw=style.get('lw', 1.5),
                linestyle=style.get('ls', '-'))
        ax.fill_between(x, mean - std, mean + std, alpha=0.12,
                        color=style.get('color', 'grey'))

    ax.set_xlabel('Environment Steps', fontsize=12)
    ax.set_ylabel('Mean Return (10 eval episodes)', fontsize=12)
    ax.set_title(
        'Object-Centric VQ Tokens — Fully Discrete vs Oracle Position\n'
        'MiniGrid-DoorKey-5x5  |  200k step budget',
        fontsize=11,
    )
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8.5, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Annotation: oracle vs discrete comparison
    if 'OC-All+Pos' in results and 'OC-All-Disc' in results:
        pos_m  = results['OC-All+Pos'][:, :budget_ckpts].mean(0)[-1]
        disc_m = results['OC-All-Disc'][:, :budget_ckpts].mean(0)[-1]
        ax.annotate(
            f'Oracle pos: {pos_m:.3f}\nDiscrete pos: {disc_m:.3f}',
            xy=(200_000, max(pos_m, disc_m)),
            xytext=(130_000, 0.7),
            fontsize=8.5,
            arrowprops=dict(arrowstyle='->', lw=1.0, color='grey'),
            color='grey',
        )

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

    result_path = RESULT_DIR / 'ppo_returns.npy'
    results = (
        dict(np.load(result_path, allow_pickle=True).item())
        if result_path.exists() else {}
    )

    print("Existing results:")
    for k, v in results.items():
        print(f"  {k:<14}  {v.shape}  final={v.mean(0)[-1]:.3f}")
    print()

    # Run OC-All-Disc
    print(f"=== OC-All-Disc | {args.n_seeds} seeds × {args.total_steps:,} steps ===")
    print(f"    features_dim = {FEATURES_DIM}  "
          f"(3×{D_MODEL} OC semantic + {SPATIAL_LATENT_DIM} position token)")
    runs = []
    for seed in range(args.n_seeds):
        print(f"  seed {seed} ...", end=' ', flush=True)
        r = train_seed(seed, args.total_steps)
        runs.append(r)
        print(f"final={r[-1]:.3f}  max={max(r):.3f}")

    arr = np.array(runs)
    results['OC-All-Disc'] = arr
    print(f"\n  OC-All-Disc: mean={arr.mean(0)[-1]:.3f}  std={arr.std(0)[-1]:.3f}")

    np.save(result_path, results)
    print(f"\nSaved {result_path}")

    print("\n--- Final Performance (200k budget, last checkpoint) ---")
    for name in PLOT_ORDER:
        if name not in results:
            continue
        v    = results[name]
        v200 = v[:, :40]
        m, s = v200.mean(0)[-1], v200.std(0)[-1]
        bs   = sum(1 for seed in v200 if (seed > 0).any())
        print(f"  {name:<14}  {m:.3f} ± {s:.3f}  ({v.shape[0]} seeds, {bs} bootstrap)")

    if 'OC-All+Pos' in results and 'OC-All-Disc' in results:
        pos_m  = results['OC-All+Pos'][:, :40].mean(0)[-1]
        disc_m = results['OC-All-Disc'][:, :40].mean(0)[-1]
        ratio  = disc_m / pos_m if pos_m > 0 else float('nan')
        print(f"\n  Discretization cost: OC-All+Pos={pos_m:.3f} → "
              f"OC-All-Disc={disc_m:.3f}  ({ratio:.2f}×)")

    plot_results(results, str(PLOT_DIR / 'ppo_comparison_discrete.png'))


if __name__ == '__main__':
    main()
