"""
Script OC-3 — PPO Sample Efficiency with Object-Centric Tokens
==============================================================
Trains PPO under three conditions using the frozen ObjectCentricVQEncoder:

  Raw      : flattened 7×7×3 observation — baseline, reproduces Condition A
  OC-Agent : CLS token only (64-dim)     — ablation; compare directly with Condition C
  OC-All   : all three tokens concat'd   — full object-centric representation
              (agent 64 + key 64 + door 64 = 192-dim)

The OC-Agent vs OC-All delta isolates the contribution of the key and door
tokens beyond what the global CLS head already provides.

Compare these results against Condition C from results/ppo_baselines/ppo_returns.npy
to assess whether the new encoder's CLS head is stronger, and whether the
added object tokens improve further.

Usage:
  python src/object_centric/03_ppo_oc.py [--conditions NAME ...] [--n_seeds N]
  python src/object_centric/03_ppo_oc.py --list

Requires:
  checkpoints/object_centric/encoder.pt  (from OC-1)

Outputs:
  results/object_centric/ppo_returns.npy
  plots/object_centric/ppo_comparison.png
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

from models_oc import ObjectCentricVQEncoder, OC_AGENT_K, OC_KEY_K, OC_DOOR_K
from models import D_MODEL

CKPT_PATH  = ROOT / 'checkpoints' / 'object_centric' / 'encoder.pt'
RESULT_DIR = ROOT / 'results' / 'object_centric'
PLOT_DIR   = ROOT / 'plots' / 'object_centric'

ENV_ID      = 'MiniGrid-DoorKey-5x5-v0'
TOTAL_STEPS = 200_000
EVAL_EVERY  =   5_000
N_EVAL_EPS  =      10
N_SEEDS     =       2


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class ImgObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        img = env.observation_space['image']
        self.observation_space = gym.spaces.Box(
            low=0.0, high=float(img.high.max()),
            shape=img.shape, dtype=np.float32,
        )

    def observation(self, obs):
        return obs['image'].astype(np.float32)


def make_env(seed=0):
    env = gym.make(ENV_ID)
    env = ImgObsWrapper(env)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------

def _load_encoder():
    enc = ObjectCentricVQEncoder()
    enc.load_state_dict(torch.load(CKPT_PATH, map_location='cpu'))
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


class OCAgentExtractor(BaseFeaturesExtractor):
    """CLS token only — 64-dim. Direct comparison with existing Condition C."""

    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=D_MODEL)
        self.enc = _load_encoder()

    def forward(self, obs):
        with torch.no_grad():
            enc = self.enc.encode_all(obs.long())
        return enc['z_q_agent']

    def train(self, mode: bool = True):
        # Prevent SB3's train() call from re-enabling EMA updates in VQLayer.
        super().train(mode)
        self.enc.eval()
        return self


class OCFullExtractor(BaseFeaturesExtractor):
    """All three token embeddings — 192-dim (agent 64 + key 64 + door 64)."""

    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=3 * D_MODEL)
        self.enc = _load_encoder()

    def forward(self, obs):
        with torch.no_grad():
            enc = self.enc.encode_all(obs.long())
        return torch.cat([enc['z_q_agent'], enc['z_q_key'], enc['z_q_door']], dim=-1)

    def train(self, mode: bool = True):
        super().train(mode)
        self.enc.eval()
        return self


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

CONDITIONS = {
    'Raw': {
        'extractor': None,
        'net_arch':  [64, 64],
        'color':     'black',
        'label':     'Raw obs (147-dim)',
    },
    'OC-Agent': {
        'extractor': OCAgentExtractor,
        'net_arch':  [64, 64],
        'color':     'steelblue',
        'label':     'OC agent token (64-dim)',
    },
    'OC-All': {
        'extractor': OCFullExtractor,
        'net_arch':  [64, 64],
        'color':     'darkorange',
        'label':     'OC all tokens (192-dim)',
    },
}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_condition(name, cfg, seed, total_steps):
    env      = make_env(seed)
    eval_env = make_env(seed + 100)

    policy_kwargs = {}
    if cfg['extractor'] is not None:
        policy_kwargs['features_extractor_class'] = cfg['extractor']
        policy_kwargs['features_extractor_kwargs'] = {}
    policy_kwargs['net_arch'] = cfg['net_arch']

    model = PPO(
        'MlpPolicy', env,
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

def plot_results(results, total_steps, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    steps = np.arange(1, total_steps // EVAL_EVERY + 1) * EVAL_EVERY

    for name, arr in results.items():
        cfg   = CONDITIONS.get(name, {})
        color = cfg.get('color', 'grey')
        label = cfg.get('label', name)
        mean  = arr.mean(0)
        std   = arr.std(0)
        n_ckpt = arr.shape[1]
        x = np.arange(1, n_ckpt + 1) * EVAL_EVERY
        ax.plot(x, mean, label=f'{label}  ({mean[-1]:.3f}±{std[-1]:.3f})', color=color)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=color)

    ax.set_xlabel('Environment steps')
    ax.set_ylabel('Mean return (10 eval episodes)')
    ax.set_title('Object-Centric VQ Tokens — PPO Sample Efficiency\n'
                 'MiniGrid-DoorKey-5x5')
    ax.legend(fontsize=9)
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
    parser.add_argument(
        '--conditions', nargs='+', default=list(CONDITIONS.keys()),
        help='Conditions to run this invocation.',
    )
    parser.add_argument('--n_seeds',     type=int, default=N_SEEDS)
    parser.add_argument('--total_steps', type=int, default=TOTAL_STEPS)
    parser.add_argument('--list', action='store_true',
                        help='Print available conditions and exit.')
    args = parser.parse_args()

    if args.list:
        print('Available conditions:', ', '.join(CONDITIONS.keys()))
        return

    unknown = set(args.conditions) - set(CONDITIONS.keys())
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}. "
                         f"Available: {list(CONDITIONS.keys())}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    result_path = RESULT_DIR / 'ppo_returns.npy'
    results = (
        dict(np.load(result_path, allow_pickle=True).item())
        if result_path.exists() else {}
    )

    if results:
        print("Existing results:", {k: f"{v.mean(0)[-1]:.3f}" for k, v in results.items()})

    for name in args.conditions:
        cfg = CONDITIONS[name]
        print(f"\n=== {name} | {args.n_seeds} seeds × {args.total_steps:,} steps ===")
        runs = []
        for seed in range(args.n_seeds):
            print(f"  seed {seed} ...", end=' ', flush=True)
            r = train_condition(name, cfg, seed, args.total_steps)
            runs.append(r)
            print(f"final={r[-1]:.3f}")
        arr = np.array(runs)
        results[name] = arr
        print(f"  → {name}: mean={arr.mean(0)[-1]:.3f}  std={arr.std(0)[-1]:.3f}")

    np.save(result_path, results)
    print(f"\nSaved {result_path}")

    print('\n--- Final Performance ---')
    for name, arr in sorted(results.items()):
        print(f"  {name:<12} {arr.mean(0)[-1]:.3f} ± {arr.std(0)[-1]:.3f}")

    plot_results(results, args.total_steps, str(PLOT_DIR / 'ppo_comparison.png'))


if __name__ == '__main__':
    main()
