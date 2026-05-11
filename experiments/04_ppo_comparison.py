"""
Script 4 — PPO Sample Efficiency Comparison
=============================================
Trains PPO across four representation conditions and compares sample efficiency.

Condition A (Raw):        policy sees flattened 7x7x3 image directly
Condition B (Continuous): policy receives the frozen pre-trained continuous
                          AE latent vector (LATENT_DIM-dimensional)
Condition C (Discrete VQ):policy receives the codebook embedding for the
                          discrete token from the frozen VQ-Transformer
Condition D (Sentence VQ):policy receives a sentence of HISTORY_LEN+1 VQ token
                          embeddings; the current observation's token is always
                          at the final (anchor) position

Both B and C use encoders pre-trained via reconstruction in script 02.
D uses the same frozen VQ encoder but augments each step with a rolling
history of the last HISTORY_LEN observations, giving the policy temporal
context to infer agent position from trajectory.

This is the core planning-efficiency experiment: does operating over a
discrete token space improve sample efficiency compared to raw or continuous
representations?

Usage:
  python 04_ppo_comparison.py

Requires:
  checkpoints/edits01/vq_encoder.pt
  checkpoints/edits01/continuous_encoder.pt

Outputs:
  results/edits02/ppo_returns.npy   — dict of {condition: (n_seeds, n_checkpoints)} arrays
  plots/edits02/ppo_comparison.png

Estimated runtime: ~15-40 min depending on hardware (2 seeds, 50k steps each).
Increase N_SEEDS to 5 for final results (Agarwal et al. 2021 recommendation).
"""

import os
import pickle
from collections import deque

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

from models import (
    ContinuousEncoder,
    TransformerVQEncoder,
    CODEBOOK_SIZE,
    D_MODEL,
    LATENT_DIM,
)

ENV_ID      = 'MiniGrid-DoorKey-5x5-v0'
TOTAL_STEPS =  50_000
EVAL_EVERY  =   5_000
N_EVAL_EPS  =      10
N_SEEDS     =       2   # set to 5 for final results
HISTORY_LEN =       7   # past frames in sentence; sentence length = HISTORY_LEN + 1


# ---------------------------------------------------------------------------
# Environment wrapper
# ---------------------------------------------------------------------------

class ImgObsWrapper(gym.ObservationWrapper):
    """Extract the image from MiniGrid's dict observation."""

    def __init__(self, env):
        super().__init__(env)
        img_space = env.observation_space['image']
        # Declare as float32 so SB3 doesn't apply its uint8 normalisation
        self.observation_space = gym.spaces.Box(
            low=0.0, high=float(img_space.high.max()),
            shape=img_space.shape, dtype=np.float32,
        )

    def observation(self, obs):
        return obs['image'].astype(np.float32)


def make_env(seed=0):
    env = gym.make(ENV_ID)
    env = ImgObsWrapper(env)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


class HistoryWrapper(gym.Wrapper):
    """
    Augments each observation with a rolling buffer of the last HISTORY_LEN frames.

    Output shape: (HISTORY_LEN + 1, 7, 7, 3) — past frames first, current last.
    The current observation is always at the final position (the semantic anchor).
    Episodes begin with zero-padded past frames.
    """

    def __init__(self, env, history_len=HISTORY_LEN):
        super().__init__(env)
        self.history_len = history_len
        orig_shape = env.observation_space.shape        # (7, 7, 3)
        self._obs_shape = orig_shape
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=float(env.observation_space.high.max()),
            shape=(history_len + 1, *orig_shape),
            dtype=np.float32,
        )
        self._buffer = deque(maxlen=history_len)

    def _build_sentence(self, obs):
        n_pad = self.history_len - len(self._buffer)
        past  = [np.zeros(self._obs_shape, dtype=np.float32)] * n_pad + list(self._buffer)
        return np.stack(past + [obs.astype(np.float32)], axis=0)    # (H+1, 7, 7, 3)

    def reset(self, **kwargs):
        self._buffer.clear()
        obs, info = self.env.reset(**kwargs)
        sentence = self._build_sentence(obs)
        self._buffer.append(obs.astype(np.float32))
        return sentence, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        sentence = self._build_sentence(obs)
        self._buffer.append(obs.astype(np.float32))
        return sentence, reward, terminated, truncated, info


def make_env_sentence(seed=0):
    env = gym.make(ENV_ID)
    env = ImgObsWrapper(env)
    env = HistoryWrapper(env)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


# ---------------------------------------------------------------------------
# Feature extractors (SB3 interface)
# ---------------------------------------------------------------------------

class RawExtractor(BaseFeaturesExtractor):
    """Flatten and normalise raw 7x7x3 observation to [0, 1]."""

    def __init__(self, observation_space):
        flat = int(np.prod(observation_space.shape))
        super().__init__(observation_space, features_dim=flat)
        self.flatten = nn.Flatten()
        # normalisation constant: values are 0-10
        self.register_buffer('scale', torch.tensor(10.0))

    def forward(self, obs):
        return self.flatten(obs) / self.scale


class ContinuousExtractor(BaseFeaturesExtractor):
    """Pre-trained continuous AE encoder (frozen). Returns latent vector."""

    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=LATENT_DIM)
        self.enc = ContinuousEncoder()
        self.enc.load_state_dict(
            torch.load('checkpoints/edits01/continuous_encoder.pt', map_location='cpu')
        )
        for p in self.enc.parameters():
            p.requires_grad = False
        self.enc.eval()

    def forward(self, obs):
        with torch.no_grad():
            z = self.enc.encode(obs.long())
        return z


class DiscreteVQExtractor(BaseFeaturesExtractor):
    """
    Pre-trained VQ-Transformer (frozen).
    Returns the codebook embedding vector for the discrete token.
    The PPO policy heads train on top of this fixed D_MODEL-dim representation.
    """

    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=D_MODEL)
        self.enc = TransformerVQEncoder()
        self.enc.load_state_dict(
            torch.load('checkpoints/edits01/vq_encoder.pt', map_location='cpu')
        )
        for p in self.enc.parameters():
            p.requires_grad = False
        self.enc.eval()

    def forward(self, obs):
        with torch.no_grad():
            z       = self.enc.encode(obs.long())
            _, idx, _ = self.enc.vq(z)
            emb     = self.enc.vq.codebook(idx)   # (B, D_MODEL)
        return emb


class SentenceExtractor(BaseFeaturesExtractor):
    """
    Sentence of (HISTORY_LEN + 1) VQ token embeddings; current token last.

    Each of the H+1 stacked observations is encoded by the frozen VQ encoder.
    The resulting codebook embeddings are concatenated into a single vector:

        [embed(t-H), ..., embed(t-1), embed(t)]
               past frames              ^^^^^^^^
                                    semantic anchor
                                    (current state, fixed final position)

    The final D_MODEL positions always correspond to the current observation's
    token, whose semantics are interpretable via the codebook heatmap.
    Preceding positions encode recent trajectory, allowing the policy to infer
    agent position from token transitions even though no single token encodes
    grid coordinates directly.

    features_dim = (HISTORY_LEN + 1) * D_MODEL  =  8 * 64  =  512
    """

    def __init__(self, observation_space, history_len=HISTORY_LEN):
        n_tokens = history_len + 1
        super().__init__(observation_space, features_dim=n_tokens * D_MODEL)
        self.n_tokens = n_tokens

        self.enc = TransformerVQEncoder()
        self.enc.load_state_dict(
            torch.load('checkpoints/edits01/vq_encoder.pt', map_location='cpu')
        )
        for p in self.enc.parameters():
            p.requires_grad = False
        self.enc.eval()

    def forward(self, obs):
        # obs: (B, H+1, 7, 7, 3) float32
        B = obs.shape[0]
        obs_flat = obs.reshape(B * self.n_tokens, 7, 7, 3).long()
        with torch.no_grad():
            z         = self.enc.encode(obs_flat)       # (B*(H+1), D_MODEL)
            _, idx, _ = self.enc.vq(z)
            emb       = self.enc.vq.codebook(idx)       # (B*(H+1), D_MODEL)
        # Current token is always at the final D_MODEL positions
        return emb.reshape(B, self.n_tokens * D_MODEL)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

CONDITIONS = {
    'A_Raw': {
        'extractor': RawExtractor,
        'kwargs':    {},
        'make_env':  make_env,
    },
    'B_Continuous': {
        'extractor': ContinuousExtractor,
        'kwargs':    {},
        'make_env':  make_env,
    },
    'C_Discrete_VQ': {
        'extractor': DiscreteVQExtractor,
        'kwargs':    {},
        'make_env':  make_env,
    },
    'D_Sentence_VQ': {
        'extractor': SentenceExtractor,
        'kwargs':    {},
        'make_env':  make_env_sentence,
    },
}


def train_condition(name, extractor_cls, extractor_kwargs, env_factory, seed):
    env      = env_factory(seed)
    eval_env = env_factory(seed + 10_000)

    policy_kwargs = {
        'features_extractor_class':  extractor_cls,
        'features_extractor_kwargs': extractor_kwargs,
        'net_arch':                  [64, 64],
    }

    model = PPO(
        'MlpPolicy', env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        verbose=0,
        seed=seed,
    )

    returns = []
    n_ckpts = TOTAL_STEPS // EVAL_EVERY

    for i in range(n_ckpts):
        model.learn(total_timesteps=EVAL_EVERY, reset_num_timesteps=(i == 0))
        mean_r, _ = evaluate_policy(model, eval_env, n_eval_episodes=N_EVAL_EPS, warn=False)
        returns.append(mean_r)
        print(f"    step={(i+1)*EVAL_EVERY:>6d}  return={mean_r:+.2f}")

    env.close()
    eval_env.close()
    return returns


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(all_results, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    steps  = np.arange(1, len(list(all_results.values())[0][0]) + 1) * EVAL_EVERY
    colors = {
        'A_Raw':         '#4c9be8',
        'B_Continuous':  '#f4a261',
        'C_Discrete_VQ': '#2ecc71',
        'D_Sentence_VQ': '#9b59b6',
    }
    names  = {
        'A_Raw':         'A: Raw observations',
        'B_Continuous':  'B: Continuous AE (frozen)',
        'C_Discrete_VQ': 'C: Discrete VQ (single token, frozen)',
        'D_Sentence_VQ': 'D: Sentence VQ (history + anchor, frozen)',
    }

    for cond, arr in all_results.items():
        mean = arr.mean(0)
        std  = arr.std(0)
        ax.plot(steps, mean, color=colors[cond], label=names[cond], linewidth=2)
        ax.fill_between(steps, mean - std, mean + std, color=colors[cond], alpha=0.15)

    ax.set_xlabel('Environment Steps', fontsize=12)
    ax.set_ylabel('Mean Episodic Return', fontsize=12)
    ax.set_title(
        f'PPO Sample Efficiency — MiniGrid-DoorKey-5x5\n'
        f'({N_SEEDS} seed{"s" if N_SEEDS > 1 else ""}, '
        f'{N_EVAL_EPS} eval episodes per checkpoint)',
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs('results/edits02', exist_ok=True)
    os.makedirs('plots/edits02',   exist_ok=True)

    all_results = {}

    for cond_name, cfg in CONDITIONS.items():
        print(f"\n=== {cond_name} ===")
        seed_returns = []

        for seed_i in range(N_SEEDS):
            seed = seed_i * 100
            print(f"  Seed {seed_i} (seed={seed}):")
            rets = train_condition(cond_name, cfg['extractor'], cfg['kwargs'], cfg['make_env'], seed)
            seed_returns.append(rets)

        all_results[cond_name] = np.array(seed_returns)  # (N_SEEDS, n_ckpts)

    np.save('results/edits02/ppo_returns.npy', all_results, allow_pickle=True)
    print("\nSaved results/edits02/ppo_returns.npy")

    plot_comparison(all_results, 'plots/edits02/ppo_comparison.png')

    print("\n=== Final Performance (last checkpoint) ===")
    for cond, arr in all_results.items():
        m, s = arr[:, -1].mean(), arr[:, -1].std()
        print(f"  {cond}: {m:.2f} ± {s:.2f}")


if __name__ == '__main__':
    main()
