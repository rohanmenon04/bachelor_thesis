"""
Script 4 — PPO Conditions
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
  python 04_ppo_conditions.py

Requires:
  checkpoints/semantic_encoder.pt
  checkpoints/continuous_encoder.pt

Outputs:
  results/ppo_returns.npy   — dict of {condition: (n_seeds, n_checkpoints)} arrays
  plots/ppo_comparison.png

Estimated runtime: ~15-40 min depending on hardware (2 seeds, 50k steps each).
Increase N_SEEDS to 5 for final results (Agarwal et al. 2021 recommendation).
"""

import argparse
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
    VQSpatialEncoder,
    PositionEmbeddingEncoder,
    CODEBOOK_SIZE,
    D_MODEL,
    LATENT_DIM,
    SPATIAL_LATENT_DIM,
)

ENV_ID      = 'MiniGrid-DoorKey-5x5-v0'
TOTAL_STEPS = 200_000   # Condition H: extended budget per architecture proposal §3.2
EVAL_EVERY  =   5_000
N_EVAL_EPS  =      10
N_SEEDS     =       2   # set to 5 for final results
HISTORY_LEN =       7   # past frames in sentence; sentence length = HISTORY_LEN + 1
GRID_SIZE   =       5   # DoorKey-5x5: positions range 0..GRID_SIZE-1 ⇒ normalise by 4


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


class PositionObsWrapper(gym.ObservationWrapper):
    """
    Augments the MiniGrid observation with the agent's true world-frame
    position (Condition G / I). Returns a Dict observation:
        {
            'image': (7, 7, 3) float32 — partial egocentric view,
            'pos':   (2,)     int64  — (x, y) world coords in [0, GRID_SIZE),
        }

    Position is read from ``env.unwrapped.agent_pos`` rather than being
    extracted from the image. MiniGrid's default partial observation does
    not render the agent in its own view, so the obs image alone does not
    carry world-frame position information. ``agent_pos`` is the
    ground-truth oracle position used by the proposal as the spatial signal.

    Integer positions are emitted (not normalised floats) so that the
    spatial VQ encoder can index its discrete embedding tables directly,
    matching the discrete-input pattern that keeps the semantic encoder's
    codebook diverse. The G extractor normalises to [0, 1] internally
    before appending to the policy feature vector.
    """

    def __init__(self, env):
        super().__init__(env)
        img_space = env.observation_space['image']
        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(
                low=0.0, high=float(img_space.high.max()),
                shape=img_space.shape, dtype=np.float32,
            ),
            'pos': gym.spaces.Box(
                low=0, high=GRID_SIZE - 1, shape=(2,), dtype=np.int64,
            ),
        })

    def observation(self, obs):
        ax, ay = self.env.unwrapped.agent_pos      # (x, y) world coords
        return {
            'image': obs['image'].astype(np.float32),
            'pos':   np.array([ax, ay], dtype=np.int64),
        }


def make_env_position(seed=0):
    env = gym.make(ENV_ID)
    env = PositionObsWrapper(env)
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
            torch.load('checkpoints/continuous_encoder.pt', map_location='cpu')
        )
        for p in self.enc.parameters():
            p.requires_grad = False
        self.enc.eval()

    def train(self, mode=True):
        super().train(mode)
        self.enc.eval()
        return self

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
            torch.load('checkpoints/semantic_encoder.pt', map_location='cpu')
        )
        for p in self.enc.parameters():
            p.requires_grad = False
        self.enc.eval()

    def train(self, mode=True):
        super().train(mode)
        self.enc.eval()
        return self

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
            torch.load('checkpoints/semantic_encoder.pt', map_location='cpu')
        )
        for p in self.enc.parameters():
            p.requires_grad = False
        self.enc.eval()

    def train(self, mode=True):
        super().train(mode)
        self.enc.eval()
        return self

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


class PositionAugmentedVQExtractor(BaseFeaturesExtractor):
    """
    Condition G: frozen VQ codebook embedding (64-dim) ⊕ oracle agent
    position (2-dim, normalised). features_dim = D_MODEL + 2 = 66.

    Operates on a Dict observation produced by ``PositionObsWrapper``.
    Position arrives as int world coords; this extractor normalises to
    [0, 1] before appending to the policy feature vector. Isolates the
    spatial information contribution: G vs C measures exactly what oracle
    position adds to the semantic token.
    """

    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=D_MODEL + 2)
        self.enc = TransformerVQEncoder()
        self.enc.load_state_dict(
            torch.load('checkpoints/semantic_encoder.pt', map_location='cpu')
        )
        for p in self.enc.parameters():
            p.requires_grad = False
        self.enc.eval()
        self.register_buffer('pos_norm', torch.tensor(float(GRID_SIZE - 1)))

    def train(self, mode=True):
        super().train(mode)
        self.enc.eval()
        return self

    def forward(self, obs):
        image    = obs['image']                         # (B, 7, 7, 3) float32
        # SB3 preprocesses int Box to float; ints survive the conversion.
        pos_norm = obs['pos'].float() / self.pos_norm   # (B, 2) ∈ [0, 1]
        with torch.no_grad():
            z         = self.enc.encode(image.long())
            _, idx, _ = self.enc.vq(z)
            emb       = self.enc.vq.codebook(idx)       # (B, D_MODEL)
        return torch.cat([emb, pos_norm], dim=1)        # (B, 66)


class DualVQExtractor(BaseFeaturesExtractor):
    """
    Condition I: dual-VQ fully-discrete state representation.

    Combines two independently trained VQ encoders:
      - TransformerVQEncoder (semantic):  obs    → 64-dim codebook embedding
      - VQSpatialEncoder    (spatial):    pos_2d → 32-dim codebook embedding

    features_dim = D_MODEL + SPATIAL_LATENT_DIM = 64 + 32 = 96.

    The two encoders' codebook embeddings are concatenated. Both encoders
    are frozen during PPO training. No continuous components appear in the
    pipeline downstream of the encoder lookups — the only continuous
    quantities are the codebook embedding tables themselves.
    """

    def __init__(self, observation_space):
        super().__init__(
            observation_space, features_dim=D_MODEL + SPATIAL_LATENT_DIM,
        )

        self.enc = TransformerVQEncoder()
        self.enc.load_state_dict(
            torch.load('checkpoints/semantic_encoder.pt', map_location='cpu')
        )
        for p in self.enc.parameters():
            p.requires_grad = False
        self.enc.eval()

        self.spatial = VQSpatialEncoder()
        self.spatial.load_state_dict(
            torch.load('checkpoints/spatial_vq_encoder.pt', map_location='cpu')
        )
        for p in self.spatial.parameters():
            p.requires_grad = False
        self.spatial.eval()

    def train(self, mode=True):
        super().train(mode)
        self.enc.eval()
        self.spatial.eval()
        return self

    def forward(self, obs):
        image   = obs['image']                          # (B, 7, 7, 3) float32
        # SB3 preprocesses int Box to float; cast back to long for embedding lookup.
        pos_int = obs['pos'].long()                     # (B, 2)       long
        with torch.no_grad():
            # Semantic token (what)
            z_sem         = self.enc.encode(image.long())
            _, idx_sem, _ = self.enc.vq(z_sem)
            emb_sem       = self.enc.vq.codebook(idx_sem)        # (B, D_MODEL)

            # Spatial token (where) — int positions index discrete embeddings
            z_sp          = self.spatial.encode(pos_int)
            _, idx_sp, _  = self.spatial.vq(z_sp)
            emb_sp        = self.spatial.vq.codebook(idx_sp)     # (B, SPATIAL_LATENT_DIM)

        return torch.cat([emb_sem, emb_sp], dim=1)               # (B, 96)



class DualPosEmbedExtractor(BaseFeaturesExtractor):
    """
    Condition I-b: semantic VQ token (64-dim) ⊕ learned position embedding (32-dim).

    Alternative to DualVQExtractor (Condition I-a). The spatial component uses
    PositionEmbeddingEncoder — a direct learned lookup table over the 25 grid
    cells — rather than a VQ-quantised spatial encoder. Eliminates VQ codebook
    collapse as a confound when comparing I-a vs I-b.

    If checkpoints/spatial_pos_encoder.pt exists the embedding is loaded and
    frozen. Otherwise it starts from random init and is learned jointly during
    PPO (only 800 parameters; converges quickly from sparse reward).

    features_dim = D_MODEL + SPATIAL_LATENT_DIM = 96 (same as I-a).
    """

    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=D_MODEL + SPATIAL_LATENT_DIM)

        self.enc = TransformerVQEncoder()
        self.enc.load_state_dict(
            torch.load('checkpoints/semantic_encoder.pt', map_location='cpu')
        )
        for p in self.enc.parameters():
            p.requires_grad = False
        self.enc.eval()

        self.spatial = PositionEmbeddingEncoder()
        ckpt = 'checkpoints/spatial_pos_encoder.pt'
        if os.path.exists(ckpt):
            self.spatial.load_state_dict(torch.load(ckpt, map_location='cpu'))
            for p in self.spatial.parameters():
                p.requires_grad = False
            self.spatial.eval()
        # No checkpoint: embedding starts random, learned jointly with PPO.

    def train(self, mode=True):
        super().train(mode)
        self.enc.eval()
        return self

    def forward(self, obs):
        image   = obs['image']
        pos_int = obs['pos'].long()
        with torch.no_grad():
            z_sem         = self.enc.encode(image.long())
            _, idx_sem, _ = self.enc.vq(z_sem)
            emb_sem       = self.enc.vq.codebook(idx_sem)     # (B, D_MODEL)
        emb_sp = self.spatial.encode(pos_int)                  # (B, SPATIAL_LATENT_DIM)
        return torch.cat([emb_sem, emb_sp], dim=1)             # (B, 96)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

CONDITIONS = {
    'A_Raw': {
        'extractor':   RawExtractor,
        'kwargs':      {},
        'make_env':    make_env,
        'policy_type': 'MlpPolicy',
    },
    'B_Continuous': {
        'extractor':   ContinuousExtractor,
        'kwargs':      {},
        'make_env':    make_env,
        'policy_type': 'MlpPolicy',
    },
    'C_Discrete_VQ': {
        'extractor':   DiscreteVQExtractor,
        'kwargs':      {},
        'make_env':    make_env,
        'policy_type': 'MlpPolicy',
    },
    'D_Sentence_VQ': {
        'extractor':   SentenceExtractor,
        'kwargs':      {},
        'make_env':    make_env_sentence,
        'policy_type': 'MlpPolicy',
    },
    'G_Position_Augmented_VQ': {
        'extractor':   PositionAugmentedVQExtractor,
        'kwargs':      {},
        'make_env':    make_env_position,
        'policy_type': 'MultiInputPolicy',
    },
    'I_Dual_VQ': {
        'extractor':   DualVQExtractor,
        'kwargs':      {},
        'make_env':    make_env_position,
        'policy_type': 'MultiInputPolicy',
    },
    'I_Dual_PosEmbed': {
        'extractor':   DualPosEmbedExtractor,
        'kwargs':      {},
        'make_env':    make_env_position,
        'policy_type': 'MultiInputPolicy',
    },
}


def train_condition(name, extractor_cls, extractor_kwargs, env_factory, policy_type, seed):
    env      = env_factory(seed)
    eval_env = env_factory(seed + 10_000)

    policy_kwargs = {
        'features_extractor_class':  extractor_cls,
        'features_extractor_kwargs': extractor_kwargs,
        'net_arch':                  [64, 64],
    }

    model = PPO(
        policy_type, env,
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

    max_ckpts = max(arr.shape[1] for arr in all_results.values())
    steps     = np.arange(1, max_ckpts + 1) * EVAL_EVERY
    colors = {
        'A_Raw':                   '#4c9be8',
        'B_Continuous':            '#f4a261',
        'C_Discrete_VQ':           '#2ecc71',
        'D_Sentence_VQ':           '#9b59b6',
        'G_Position_Augmented_VQ': '#e74c3c',
        'I_Dual_VQ':               '#1abc9c',
        'I_Dual_PosEmbed': '#e67e22',
    }
    names  = {
        'A_Raw':                   'A: Raw observations',
        'B_Continuous':            'B: Continuous AE (frozen)',
        'C_Discrete_VQ':           'C: Discrete VQ (single token, frozen)',
        'D_Sentence_VQ':           'D: Sentence VQ (history + anchor, frozen)',
        'G_Position_Augmented_VQ': 'G: VQ token + oracle position (frozen)',
        'I_Dual_VQ':        'I-a: Dual-VQ (semantic ⊕ spatial VQ, frozen)',
        'I_Dual_PosEmbed':  'I-b: Dual (semantic VQ ⊕ position embedding)',
    }

    for cond, arr in all_results.items():
        mean = arr.mean(0)
        std  = arr.std(0)
        c    = colors.get(cond, '#888888')
        lbl  = names.get(cond, cond)
        x    = np.arange(1, arr.shape[1] + 1) * EVAL_EVERY
        ax.plot(x, mean, color=c, label=lbl, linewidth=2)
        ax.fill_between(x, mean - std, mean + std, color=c, alpha=0.15)

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

def parse_args():
    parser = argparse.ArgumentParser(
        description='PPO sample-efficiency comparison across representation conditions.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--conditions', nargs='+', default=None, metavar='NAME',
        help=f'Conditions to run (subset). Default: all. '
             f'Choices: {sorted(CONDITIONS.keys())}',
    )
    parser.add_argument(
        '--n_seeds', type=int, default=N_SEEDS,
        help='Number of seeds per condition (overrides script default).',
    )
    parser.add_argument(
        '--total_steps', type=int, default=TOTAL_STEPS,
        help='Total environment steps per seed (overrides script default).',
    )
    parser.add_argument(
        '--list', action='store_true',
        help='Print the available conditions and exit.',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list:
        print('Available conditions:')
        for name in CONDITIONS:
            print(f'  {name}')
        return

    # Allow per-run overrides without editing the file.
    global N_SEEDS, TOTAL_STEPS
    N_SEEDS     = args.n_seeds
    TOTAL_STEPS = args.total_steps

    conditions_to_run = args.conditions or list(CONDITIONS.keys())
    unknown = [c for c in conditions_to_run if c not in CONDITIONS]
    if unknown:
        raise ValueError(
            f"Unknown condition(s): {unknown}. "
            f"Available: {sorted(CONDITIONS.keys())}"
        )

    os.makedirs('results/ppo_baselines', exist_ok=True)
    os.makedirs('plots/ppo_baselines',   exist_ok=True)

    # Merge with any existing results so that running a subset (e.g. just
    # `--conditions I_Dual_VQ`) preserves prior runs and only overwrites
    # the conditions actually re-run this invocation.
    results_path = 'results/ppo_baselines/ppo_returns.npy'
    if os.path.exists(results_path):
        all_results = np.load(results_path, allow_pickle=True).item()
        print(f"Loaded existing results: {list(all_results.keys())}")
    else:
        all_results = {}

    for cond_name in conditions_to_run:
        cfg = CONDITIONS[cond_name]
        print(f"\n=== {cond_name} ===")
        seed_returns = []

        for seed_i in range(N_SEEDS):
            seed = seed_i * 100
            print(f"  Seed {seed_i} (seed={seed}):")
            rets = train_condition(
                cond_name,
                cfg['extractor'], cfg['kwargs'],
                cfg['make_env'], cfg['policy_type'],
                seed,
            )
            seed_returns.append(rets)

        all_results[cond_name] = np.array(seed_returns)  # (N_SEEDS, n_ckpts)

    np.save(results_path, all_results, allow_pickle=True)
    print(f"\nSaved {results_path}")

    plot_comparison(all_results, 'plots/ppo_baselines/ppo_comparison.png')

    print("\n=== Final Performance (last checkpoint) ===")
    for cond, arr in all_results.items():
        m, s = arr[:, -1].mean(), arr[:, -1].std()
        tag = '  (re-run)' if cond in conditions_to_run else ''
        print(f"  {cond}: {m:.2f} ± {s:.2f}{tag}")


if __name__ == '__main__':
    main()
