"""
Script 7 — Sentence Transformer Policy (Condition F)
=====================================================
Trains a PPO policy whose observation at each timestep is a sentence of
(HISTORY_LEN + 1) VQ token embeddings, processed by a small Transformer
encoder. The current observation's token is always at the final (anchor)
position; history tokens fill the preceding positions.

The Transformer uses self-attention to let the anchor token attend to all
history positions, producing a single 64-dim vector that represents:
  "current semantic state, informed by recent trajectory"

This is the intended architecture for the sentence idea:
  - The anchor position is interpretable via the codebook (Condition C behaviour)
  - History tokens give trajectory context for planning
  - Self-attention lets the model decide which history positions are relevant
  - The output is D_MODEL-dim (64), not 512-dim — the policy MLP is not overwhelmed

Condition progression:
  C: VQ-MLP          — single token, no memory, 64-dim → MLP
  D: Sentence-MLP    — 8 tokens concatenated, 512-dim → MLP  [failed: dimensionality]
  E: LSTM-recurrent  — single token/step, recurrent hidden state [failed: no reward signal]
  F: Sentence-Transformer (this script)
                     — 8-token sentence → 2-layer self-attention → 64-dim anchor output → MLP

Architecture:
  (H+1, 7, 7, 3) obs
       ↓  frozen VQ encoder (one call per frame)
  (H+1, 64) token embeddings
       ↓  + positional encoding
  (H+1, 64) positioned sequence
       ↓  2-layer self-attention Transformer encoder
  (H+1, 64) attended outputs
       ↓  take final position (anchor)
      (64,)  ← features_dim fed to SB3 MLP policy head

Usage:
  python 07_sentence_transformer.py

Requires:
  checkpoints/edits01/vq_encoder.pt

Outputs:
  results/edits04/transformer_sequence.npy   — (N_SEEDS, n_checkpoints) array
  plots/edits04/transformer_sequence.png     — learning curve vs C / D / E
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

from models import TransformerVQEncoder, D_MODEL, CODEBOOK_SIZE

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENV_ID      = 'MiniGrid-DoorKey-5x5-v0'
TOTAL_STEPS =  50_000
EVAL_EVERY  =   5_000
N_EVAL_EPS  =      10
N_SEEDS     =       2   # set to 5 for final results
HISTORY_LEN =       7   # past frames; sentence length = HISTORY_LEN + 1 = 8

# Sentence Transformer hyperparameters
N_SENT_LAYERS = 2           # attention layers — small model, 8 tokens only
N_SENT_HEADS  = 4           # head_dim = D_MODEL // N_SENT_HEADS = 16
SENT_FFN_DIM  = D_MODEL * 2 # 128 — smaller than typical to match token count
SENT_DROPOUT  = 0.0         # no dropout: tiny model over very short sequences


# ---------------------------------------------------------------------------
# Environment wrappers
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
    Maintains a rolling buffer of the last HISTORY_LEN frames.

    Output shape: (HISTORY_LEN + 1, 7, 7, 3)
      positions 0 … H-1 : past frames, oldest first
      position  H       : current frame (semantic anchor)

    Episodes begin with zero-padded past frames.
    """

    def __init__(self, env, history_len=HISTORY_LEN):
        super().__init__(env)
        self.history_len = history_len
        orig_shape = env.observation_space.shape   # (7, 7, 3)
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
        return np.stack(past + [obs.astype(np.float32)], axis=0)   # (H+1, 7, 7, 3)

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
# Sentence Transformer feature extractor
# ---------------------------------------------------------------------------

class SentenceTransformerExtractor(BaseFeaturesExtractor):
    """
    Condition F: self-attention Transformer over H+1 VQ token embeddings.

    Processing pipeline per step:
      1. Encode each of the H+1 frames with the frozen VQ encoder
         → H+1 codebook embeddings, each D_MODEL-dim
      2. Add learned positional encodings (position 0=oldest, H=current anchor)
      3. Pass through N_SENT_LAYERS of self-attention
      4. Return the output at position H (the anchor position)

    Output dim: D_MODEL (64) — same as Condition C, so the SB3 MLP policy
    head receives a compact, fixed-size representation. The Transformer has
    already absorbed the H history tokens into the anchor's attended output.

    The only trained parameters are:
      - Positional embeddings (H+1, D_MODEL)
      - Sentence Transformer weights (attention + FFN for each layer)
    The VQ encoder is fully frozen.
    """

    def __init__(self, observation_space, history_len=HISTORY_LEN):
        # Output is D_MODEL — anchor position output after self-attention
        super().__init__(observation_space, features_dim=D_MODEL)
        self.n_tokens = history_len + 1

        # Frozen VQ encoder
        self.enc = TransformerVQEncoder()
        self.enc.load_state_dict(
            torch.load('checkpoints/edits01/vq_encoder.pt', map_location='cpu')
        )
        for p in self.enc.parameters():
            p.requires_grad = False
        self.enc.eval()

        # Positional encoding for the sentence (one entry per token position)
        self.pos_embed = nn.Embedding(self.n_tokens, D_MODEL)

        # Small Transformer encoder
        sent_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_SENT_HEADS,
            dim_feedforward=SENT_FFN_DIM,
            dropout=SENT_DROPOUT,
            batch_first=True,
            norm_first=True,   # pre-norm: more stable for small models
        )
        self.sentence_transformer = nn.TransformerEncoder(
            sent_layer, num_layers=N_SENT_LAYERS
        )

    def train(self, mode=True):
        # SB3 propagates train() to all child modules during PPO updates.
        # Keep the VQ encoder permanently in eval so its EMA codebook is never
        # mutated by RL training data — only the sentence Transformer trains.
        super().train(mode)
        self.enc.eval()
        return self

    def forward(self, obs):
        # obs: (B, H+1, 7, 7, 3) float32
        B = obs.shape[0]

        # Encode all frames in the sentence with the frozen VQ encoder
        frames = obs.reshape(B * self.n_tokens, 7, 7, 3).long()
        with torch.no_grad():
            z         = self.enc.encode(frames)          # (B*(H+1), D_MODEL)
            _, idx, _ = self.enc.vq(z)
            emb       = self.enc.vq.codebook(idx)        # (B*(H+1), D_MODEL)

        emb = emb.reshape(B, self.n_tokens, D_MODEL)     # (B, H+1, D_MODEL)

        # Add positional encoding
        pos = torch.arange(self.n_tokens, device=obs.device)
        emb = emb + self.pos_embed(pos)                  # (B, H+1, D_MODEL)

        # Self-attention over the sentence
        out = self.sentence_transformer(emb)             # (B, H+1, D_MODEL)

        # Return the anchor position output (current observation, always last)
        return out[:, -1, :]                             # (B, D_MODEL)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_condition_f(seed):
    env      = make_env(seed)
    eval_env = make_env(seed + 10_000)

    policy_kwargs = {
        'features_extractor_class':  SentenceTransformerExtractor,
        'features_extractor_kwargs': {'history_len': HISTORY_LEN},
        'net_arch':                  [64, 64],
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

    returns = []
    n_ckpts = TOTAL_STEPS // EVAL_EVERY

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

def plot_results(f_results, save_path):
    """
    Plot Condition F against the prior conditions for direct comparison.
    Loads C/D from results/edits02 and E from results/edits03 if available.
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    steps = np.arange(1, f_results.shape[1] + 1) * EVAL_EVERY

    # --- Condition F (this run) ---
    mean_f = f_results.mean(0)
    std_f  = f_results.std(0)
    ax.plot(steps, mean_f, color='#9b59b6', linewidth=2.5,
            label='F: Sentence Transformer (self-attention, anchor output)')
    ax.fill_between(steps, mean_f - std_f, mean_f + std_f,
                    color='#9b59b6', alpha=0.15)

    # --- Prior conditions (load if available) ---
    edits02_path = 'results/edits02/ppo_returns.npy'
    if os.path.exists(edits02_path):
        ppo_data = np.load(edits02_path, allow_pickle=True).item()

        cond_c = ppo_data.get('C_Discrete_VQ')
        if cond_c is not None:
            mean_c = cond_c.mean(0)
            std_c  = cond_c.std(0)
            ax.plot(steps[:len(mean_c)], mean_c, color='#2ecc71', linewidth=1.8,
                    linestyle='--', label='C: Discrete VQ — MLP (single token, no memory)')
            ax.fill_between(steps[:len(mean_c)], mean_c - std_c, mean_c + std_c,
                            color='#2ecc71', alpha=0.10)

        cond_d = ppo_data.get('D_Sentence_VQ')
        if cond_d is not None:
            mean_d = cond_d.mean(0)
            std_d  = cond_d.std(0)
            ax.plot(steps[:len(mean_d)], mean_d, color='#e67e22', linewidth=1.8,
                    linestyle=':', label='D: Sentence VQ — flat MLP (512-dim concat)')
            ax.fill_between(steps[:len(mean_d)], mean_d - std_d, mean_d + std_d,
                            color='#e67e22', alpha=0.10)

    edits03_path = 'results/edits03/lstm_returns.npy'
    if os.path.exists(edits03_path):
        cond_e = np.load(edits03_path)
        mean_e = cond_e.mean(0)
        std_e  = cond_e.std(0)
        ax.plot(steps[:len(mean_e)], mean_e, color='#95a5a6', linewidth=1.5,
                linestyle='--', label='E: LSTM (single token/step, recurrent)')
        ax.fill_between(steps[:len(mean_e)], mean_e - std_e, mean_e + std_e,
                        color='#95a5a6', alpha=0.10)

    ax.set_xlabel('Environment Steps', fontsize=12)
    ax.set_ylabel('Mean Episodic Return', fontsize=12)
    ax.set_title(
        f'Condition F: Sentence Transformer — MiniGrid-DoorKey-5x5\n'
        f'({N_SEEDS} seed{"s" if N_SEEDS > 1 else ""}, {N_EVAL_EPS} eval episodes, '
        f'history={HISTORY_LEN}, layers={N_SENT_LAYERS}, heads={N_SENT_HEADS})',
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
    os.makedirs('results/edits04', exist_ok=True)
    os.makedirs('plots/edits04',   exist_ok=True)

    n_params_sent = sum(
        p.numel() for p in
        SentenceTransformerExtractor(
            gym.spaces.Box(0, 10, shape=(HISTORY_LEN + 1, 7, 7, 3), dtype=np.float32),
            history_len=HISTORY_LEN,
        ).parameters()
        if p.requires_grad
    )
    print(f"=== Condition F: Sentence Transformer ===")
    print(f"Sentence length : {HISTORY_LEN + 1} tokens")
    print(f"Layers / heads  : {N_SENT_LAYERS} / {N_SENT_HEADS}")
    print(f"Trainable params: {n_params_sent:,}  (extractor only, VQ frozen)")
    print(f"Output dim      : {D_MODEL}  (anchor position after self-attention)")
    print(f"Total steps     : {TOTAL_STEPS:,} per seed")
    print()

    seed_returns = []

    for seed_i in range(N_SEEDS):
        seed = seed_i * 100
        print(f"  Seed {seed_i} (seed={seed}):")
        rets = train_condition_f(seed)
        seed_returns.append(rets)

    f_results = np.array(seed_returns)   # (N_SEEDS, n_ckpts)
    np.save('results/edits04/transformer_sequence.npy', f_results)
    print("\nSaved results/edits04/transformer_sequence.npy")

    plot_results(f_results, 'plots/edits04/transformer_sequence.png')

    print("\n=== Final Performance (last checkpoint) ===")
    m, s = f_results[:, -1].mean(), f_results[:, -1].std()
    print(f"  Condition F (Sentence Transformer): {m:.3f} ± {s:.3f}")

    # Print comparison if prior results available
    edits02_path = 'results/edits02/ppo_returns.npy'
    if os.path.exists(edits02_path):
        ppo_data = np.load(edits02_path, allow_pickle=True).item()
        for label, key in [('C (VQ-MLP)', 'C_Discrete_VQ'), ('D (Sentence-MLP)', 'D_Sentence_VQ')]:
            arr = ppo_data.get(key)
            if arr is not None:
                mc, sc = arr[:, -1].mean(), arr[:, -1].std()
                print(f"  Condition {label}: {mc:.3f} ± {sc:.3f}")


if __name__ == '__main__':
    main()
