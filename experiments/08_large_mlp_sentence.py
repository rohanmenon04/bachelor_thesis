"""
Script 8 — Large MLP over Concatenated Sentence (Condition D revisited)
========================================================================
Re-tests the original sentence idea (H+1 VQ token embeddings concatenated
to a 512-dim vector) but with a policy network sized appropriately for the
input dimensionality.

Condition D (Script 04) failed because the [64, 64] policy MLP tried to
compress 512-dim input to 64-dim in a single step — 32,768 first-layer
weights that could never be meaningfully initialised from 50k steps of
sparse reward. The representation was never the problem; the network was.

This script uses net_arch=[512, 256] so the first hidden layer is at least
as wide as the input, preserving all 512 dimensions before any compression.
The full parameter comparison:

  Condition D (original):  512 → 64 → 64 → actions      ~36k policy params
  This script (D-large):   512 → 512 → 256 → actions    ~393k policy params
  Condition A for scale:   147 → 64 → 64 → actions      ~14k policy params

The extractor is identical to Condition D: frozen VQ encoder, H+1 frames
concatenated, 512-dim output, current token fixed at the final 64 positions.
EMA corruption bug fixed: train() override keeps VQ encoder in eval mode.

Usage:
  python 08_large_mlp_sentence.py

Requires:
  checkpoints/vq_encoder/vq_encoder.pt

Outputs:
  results/sentence_policy/large_mlp_sentence.npy
  plots/sentence_policy/large_mlp_sentence.png
"""

import os
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

from models import TransformerVQEncoder, D_MODEL

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENV_ID      = 'MiniGrid-DoorKey-5x5-v0'
TOTAL_STEPS =  50_000
EVAL_EVERY  =   5_000
N_EVAL_EPS  =      10
N_SEEDS     =       2

HISTORY_LEN =       7   # sentence length = HISTORY_LEN + 1 = 8
SENTENCE_DIM = (HISTORY_LEN + 1) * D_MODEL   # 512

# Policy network: first hidden layer matches input width, then halves.
# This preserves all 512 input dimensions before any compression.
NET_ARCH = [512, 256]


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class ImgObsWrapper(gym.ObservationWrapper):
    """Extract the image from MiniGrid's dict observation."""

    def __init__(self, env):
        super().__init__(env)
        img_space = env.observation_space['image']
        self.observation_space = gym.spaces.Box(
            low=0.0, high=float(img_space.high.max()),
            shape=img_space.shape, dtype=np.float32,
        )

    def observation(self, obs):
        return obs['image'].astype(np.float32)


class HistoryWrapper(gym.Wrapper):
    """
    Rolling buffer of HISTORY_LEN past frames + current frame.
    Output shape: (HISTORY_LEN + 1, 7, 7, 3), oldest frame first.
    Episodes begin with zero-padded past frames.
    """

    def __init__(self, env, history_len=HISTORY_LEN):
        super().__init__(env)
        self.history_len = history_len
        orig_shape = env.observation_space.shape
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
        return np.stack(past + [obs.astype(np.float32)], axis=0)

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


def make_env(seed=0):
    env = gym.make(ENV_ID)
    env = ImgObsWrapper(env)
    env = HistoryWrapper(env)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------

class LargeSentenceExtractor(BaseFeaturesExtractor):
    """
    Identical to Condition D's SentenceExtractor: concatenates H+1 codebook
    embeddings into a 512-dim vector, current token at the final 64 positions.

    No trainable parameters beyond the frozen VQ encoder.
    The train() override prevents SB3 from flipping the VQ encoder back into
    training mode during PPO updates, which would corrupt the codebook via
    unintended EMA updates.
    """

    def __init__(self, observation_space, history_len=HISTORY_LEN):
        n_tokens = history_len + 1
        super().__init__(observation_space, features_dim=n_tokens * D_MODEL)
        self.n_tokens = n_tokens

        self.enc = TransformerVQEncoder()
        self.enc.load_state_dict(
            torch.load('checkpoints/vq_encoder/vq_encoder.pt', map_location='cpu')
        )
        for p in self.enc.parameters():
            p.requires_grad = False
        self.enc.eval()

    def train(self, mode=True):
        super().train(mode)
        self.enc.eval()   # codebook never mutated during PPO
        return self

    def forward(self, obs):
        # obs: (B, H+1, 7, 7, 3) float32
        B = obs.shape[0]
        frames = obs.reshape(B * self.n_tokens, 7, 7, 3).long()
        with torch.no_grad():
            z         = self.enc.encode(frames)
            _, idx, _ = self.enc.vq(z)
            emb       = self.enc.vq.codebook(idx)   # (B*(H+1), D_MODEL)
        # Current token always occupies the last D_MODEL positions
        return emb.reshape(B, self.n_tokens * D_MODEL)   # (B, 512)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(seed):
    env      = make_env(seed)
    eval_env = make_env(seed + 10_000)

    policy_kwargs = {
        'features_extractor_class':  LargeSentenceExtractor,
        'features_extractor_kwargs': {'history_len': HISTORY_LEN},
        'net_arch':                  NET_ARCH,
    }

    model = PPO(
        'MlpPolicy', env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=4,
        verbose=0,
        seed=seed,
    )

    returns  = []
    n_ckpts  = TOTAL_STEPS // EVAL_EVERY

    for i in range(n_ckpts):
        model.learn(total_timesteps=EVAL_EVERY, reset_num_timesteps=(i == 0))
        mean_r, _ = evaluate_policy(model, eval_env, n_eval_episodes=N_EVAL_EPS, warn=False)
        returns.append(mean_r)
        print(f"    step={(i+1)*EVAL_EVERY:>6d}  return={mean_r:+.3f}")

    env.close()
    eval_env.close()
    return returns


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results, save_path):
    fig, ax = plt.subplots(figsize=(11, 6))
    steps = np.arange(1, results.shape[1] + 1) * EVAL_EVERY

    mean = results.mean(0)
    std  = results.std(0)
    ax.plot(steps, mean, color='#e74c3c', linewidth=2.5,
            label=f'D-large: Sentence MLP  net_arch={NET_ARCH}  (512-dim input)')
    ax.fill_between(steps, mean - std, mean + std, color='#e74c3c', alpha=0.15)

    edits02_path = 'results/ppo_baselines/ppo_returns.npy'
    if os.path.exists(edits02_path):
        ppo_data = np.load(edits02_path, allow_pickle=True).item()
        styles = {
            'A_Raw':         ('#4c9be8', 'A: Raw observations',                    '-'),
            'C_Discrete_VQ': ('#2ecc71', 'C: Discrete VQ — MLP [64,64] (no mem)', '--'),
            'D_Sentence_VQ': ('#e67e22', 'D (orig): Sentence MLP [64,64]',         ':'),
        }
        for key, (color, label, ls) in styles.items():
            arr = ppo_data.get(key)
            if arr is not None:
                m, s = arr.mean(0), arr.std(0)
                ax.plot(steps[:len(m)], m, color=color, linewidth=1.8,
                        linestyle=ls, label=label)
                ax.fill_between(steps[:len(m)], m - s, m + s,
                                color=color, alpha=0.10)

    ax.set_xlabel('Environment Steps', fontsize=12)
    ax.set_ylabel('Mean Episodic Return', fontsize=12)
    ax.set_title(
        f'Condition D revisited: Sentence MLP with larger policy network\n'
        f'MiniGrid-DoorKey-5x5  |  {N_SEEDS} seeds  |  '
        f'input=512-dim  net_arch={NET_ARCH}',
        fontsize=11,
    )
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs('results/sentence_policy', exist_ok=True)
    os.makedirs('plots/sentence_policy',   exist_ok=True)

    # Count policy MLP parameters (approximate, excluding extractor and heads)
    example_params = sum(
        np.prod(shape)
        for shape in [
            (NET_ARCH[0], SENTENCE_DIM),   # first hidden layer weights
            (NET_ARCH[0],),                 # bias
            (NET_ARCH[1], NET_ARCH[0]),     # second hidden layer weights
            (NET_ARCH[1],),                 # bias
        ]
    )

    print(f"=== Condition D-large: Sentence MLP (properly sized) ===")
    print(f"Input dim    : {SENTENCE_DIM}  ({HISTORY_LEN+1} tokens × {D_MODEL})")
    print(f"Policy arch  : {SENTENCE_DIM} → {' → '.join(str(n) for n in NET_ARCH)} → actions")
    print(f"Approx MLP params: {example_params:,}  (policy net only)")
    print(f"Original D   : {SENTENCE_DIM} → 64 → 64 — had {512*64:,} first-layer weights")
    print(f"This run     : {SENTENCE_DIM} → {NET_ARCH[0]} — has {SENTENCE_DIM*NET_ARCH[0]:,} first-layer weights")
    print()

    seed_returns = []

    for seed_i in range(N_SEEDS):
        seed = seed_i * 100
        print(f"  Seed {seed_i} (seed={seed}):")
        rets = train(seed)
        seed_returns.append(rets)

    results = np.array(seed_returns)
    np.save('results/sentence_policy/large_mlp_sentence.npy', results)
    print("\nSaved results/sentence_policy/large_mlp_sentence.npy")

    plot_results(results, 'plots/sentence_policy/large_mlp_sentence.png')

    print("\n=== Final Performance (last checkpoint) ===")
    m, s = results[:, -1].mean(), results[:, -1].std()
    print(f"  D-large (Sentence MLP {NET_ARCH}): {m:.3f} ± {s:.3f}")

    edits02_path = 'results/ppo_baselines/ppo_returns.npy'
    if os.path.exists(edits02_path):
        ppo_data = np.load(edits02_path, allow_pickle=True).item()
        for label, key in [('C [64,64]', 'C_Discrete_VQ'), ('D-orig [64,64]', 'D_Sentence_VQ')]:
            arr = ppo_data.get(key)
            if arr is not None:
                mc, sc = arr[:, -1].mean(), arr[:, -1].std()
                print(f"  Condition {label}: {mc:.3f} ± {sc:.3f}")


if __name__ == '__main__':
    main()
