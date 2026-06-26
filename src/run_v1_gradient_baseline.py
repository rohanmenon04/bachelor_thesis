"""
V1 Gradient-Based VQ Encoder Baseline
======================================
Reproduces the original K=32 gradient-based VQ encoder (before EMA fixes)
and benchmarks it with PPO on MiniGrid-DoorKey-5x5 at 5 seeds x 200k steps,
matching the experimental setup used for the current Table 1 conditions.

Architecture:
  - TransformerVQEncoder with K=32, beta=0.25
  - Gradient-based codebook updates (no EMA, no dead-code restart)

Outputs:
  checkpoints/semantic_encoder_v1.pt
  results/v1_gradient/ppo_returns.npy
  results/v1_gradient/probe_summary.txt

Usage:
  cd <repo root>
  python src/run_v1_gradient_baseline.py
"""

import os
import pickle
import sys
from pathlib import Path

import gymnasium as gym
import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import minigrid

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

# ── Shared constants (match models.py) ──────────────────────────────────────
N_TYPES  = 11
N_COLORS =  6
N_STATES =  3
N_CELLS  = 49   # 7×7
D_MODEL  = 64
N_HEADS  =  4
N_LAYERS =  2

# ── V1-specific constants ────────────────────────────────────────────────────
K_V1    = 32     # original codebook size
BETA_V1 = 0.25   # original (high) commitment beta — no EMA

ENV_ID      = 'MiniGrid-DoorKey-5x5-v0'
TOTAL_STEPS = 200_000
EVAL_EVERY  =   5_000
N_EVAL_EPS  =      10
N_SEEDS     =       5
BATCH_SIZE  =     128
N_EPOCHS    =      80
LR_ENC      =   3e-4

CKPT_PATH   = ROOT / 'checkpoints' / 'semantic_encoder_v1.pt'
RESULT_DIR  = ROOT / 'results' / 'v1_gradient'
DATA_PATH   = ROOT / 'data' / 'trajectories.pkl'

DEVICE = (
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)

# ── V1 VQ Layer (gradient-based, no EMA) ────────────────────────────────────

class VQLayerV1(nn.Module):
    """Original gradient-based codebook update — causes collapse on DoorKey."""

    def __init__(self, codebook_size=K_V1, d_model=D_MODEL, beta=BETA_V1):
        super().__init__()
        self.K    = codebook_size
        self.D    = d_model
        self.beta = beta
        self.codebook = nn.Embedding(codebook_size, d_model)
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z):
        dist = (
            z.pow(2).sum(1, keepdim=True)
            - 2 * (z @ self.codebook.weight.T)
            + self.codebook.weight.pow(2).sum(1)
        )
        indices = dist.argmin(dim=1)
        z_q     = self.codebook(indices)
        z_q_st  = z + (z_q - z).detach()

        # Two-term loss: codebook term + encoder commitment term
        commit_loss = (
            F.mse_loss(z.detach(), z_q)
            + self.beta * F.mse_loss(z, z_q.detach())
        )
        return z_q_st, indices, commit_loss

# ── V1 Encoder ───────────────────────────────────────────────────────────────

class TransformerVQEncoderV1(nn.Module):
    """Transformer + V1 VQ bottleneck (K=32, gradient codebook updates)."""

    def __init__(self):
        super().__init__()
        dim_t = D_MODEL // 3 + D_MODEL % 3
        dim_c = D_MODEL // 3
        dim_s = D_MODEL // 3

        self.type_embed  = nn.Embedding(N_TYPES,  dim_t)
        self.color_embed = nn.Embedding(N_COLORS, dim_c)
        self.state_embed = nn.Embedding(N_STATES, dim_s)

        self.cls_token = nn.Parameter(torch.randn(1, 1, D_MODEL))
        self.pos_embed = nn.Embedding(N_CELLS + 1, D_MODEL)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS,
            dim_feedforward=D_MODEL * 4, dropout=0.1, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=N_LAYERS)
        self.vq = VQLayerV1()

        out_dim = N_CELLS * (N_TYPES + N_COLORS + N_STATES)
        self.decoder = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL * 4), nn.ReLU(),
            nn.Linear(D_MODEL * 4, out_dim),
        )

    def encode(self, obs):
        B = obs.shape[0]
        t = self.type_embed(obs[..., 0])
        c = self.color_embed(obs[..., 1])
        s = self.state_embed(obs[..., 2])
        x = torch.cat([t, c, s], dim=-1).view(B, N_CELLS, D_MODEL)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        pos = torch.arange(N_CELLS + 1, device=obs.device)
        x = x + self.pos_embed(pos)
        return self.transformer(x)[:, 0, :]

    def forward(self, obs):
        z = self.encode(obs)
        z_q, indices, commit_loss = self.vq(z)
        logits = self.decoder(z_q).view(-1, N_CELLS, N_TYPES + N_COLORS + N_STATES)
        return logits, indices, commit_loss


def reconstruction_loss(logits, obs):
    B = obs.shape[0]
    target = obs.view(B, N_CELLS, 3)
    loss_type  = F.cross_entropy(logits[:, :, :N_TYPES].reshape(-1, N_TYPES),
                                 target[:, :, 0].reshape(-1))
    loss_color = F.cross_entropy(logits[:, :, N_TYPES:N_TYPES+N_COLORS].reshape(-1, N_COLORS),
                                 target[:, :, 1].reshape(-1))
    loss_state = F.cross_entropy(logits[:, :, N_TYPES+N_COLORS:].reshape(-1, N_STATES),
                                 target[:, :, 2].reshape(-1))
    return loss_type + loss_color + loss_state


# ── Dataset ──────────────────────────────────────────────────────────────────

class ObsDataset(Dataset):
    def __init__(self, observations):
        self.obs = torch.tensor(observations, dtype=torch.long)
    def __len__(self):
        return len(self.obs)
    def __getitem__(self, i):
        return self.obs[i]


# ── Encoder training ──────────────────────────────────────────────────────────

def train_encoder(data_path, ckpt_path):
    print("\n" + "="*60)
    print("PHASE 1: Training V1 encoder (K=32, gradient-based)")
    print("="*60)

    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    observations = data['observations']

    model = TransformerVQEncoderV1().to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR_ENC)
    loader = DataLoader(ObsDataset(observations), batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        total_recon = total_commit = 0.0
        for batch in loader:
            batch = batch.to(DEVICE)
            logits, _, commit = model(batch)
            loss = reconstruction_loss(logits, batch) + commit
            opt.zero_grad(); loss.backward(); opt.step()
            total_recon  += reconstruction_loss(logits, batch).item()
            total_commit += commit.item()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{N_EPOCHS}  "
                  f"recon={total_recon/len(loader):.4f}  "
                  f"commit={total_commit/len(loader):.4f}")

    torch.save(model.state_dict(), ckpt_path)
    print(f"  Saved → {ckpt_path}")
    return model


# ── Probe evaluation ──────────────────────────────────────────────────────────

def probe_encoder(model, data_path):
    print("\n" + "="*60)
    print("PHASE 2: Linear probe + codebook diagnostics")
    print("="*60)

    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    observations = data['observations']
    label_arrays = {k: np.array([l[k] for l in data['labels']])
                    for k in ['door_open', 'door_locked', 'carrying_key']}

    model.eval().to(DEVICE)
    loader = DataLoader(ObsDataset(observations), batch_size=512, shuffle=False)
    all_idx = []
    with torch.no_grad():
        for batch in loader:
            _, idx, _ = model(batch.to(DEVICE))
            all_idx.append(idx.cpu().numpy())
    all_idx = np.concatenate(all_idx)

    # Codebook diagnostics
    counts  = np.bincount(all_idx, minlength=K_V1)
    active  = int((counts > 0).sum())
    probs   = counts / counts.sum()
    entropy = -(probs[probs > 0] * np.log(probs[probs > 0])).sum()
    perplexity = float(np.exp(entropy))

    print(f"\n  Codebook utilisation: {active}/{K_V1} active  "
          f"({100*active/K_V1:.0f}%)")
    print(f"  Perplexity: {perplexity:.1f} / {K_V1}")

    # One-hot probe
    X = np.zeros((len(all_idx), K_V1), dtype=np.float32)
    X[np.arange(len(all_idx)), all_idx] = 1.0

    results = {}
    print(f"\n  {'Label':<16} {'Acc':>7} {'Base':>7} {'Δ':>7}")
    print("  " + "─"*40)
    for label, y in label_arrays.items():
        if len(np.unique(y)) < 2:
            continue
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_tr, y_tr)
        acc  = accuracy_score(y_te, clf.predict(X_te))
        base = max(y_te.mean(), 1 - y_te.mean())
        results[label] = (acc, base)
        print(f"  {label:<16} {acc:7.4f} {base:7.4f} {acc-base:+7.4f}")

    return active, perplexity, results


# ── PPO ───────────────────────────────────────────────────────────────────────

class ImgObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        img_space = env.observation_space['image']
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


class V1Extractor(BaseFeaturesExtractor):
    """Frozen V1 encoder — returns codebook embedding for current token."""

    def __init__(self, observation_space, ckpt_path):
        super().__init__(observation_space, features_dim=D_MODEL)
        self.enc = TransformerVQEncoderV1()
        self.enc.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
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
            emb     = self.enc.vq.codebook(idx)
        return emb


def run_ppo(ckpt_path, result_dir):
    print("\n" + "="*60)
    print(f"PHASE 3: PPO — {N_SEEDS} seeds × {TOTAL_STEPS:,} steps")
    print("="*60)

    n_evals = TOTAL_STEPS // EVAL_EVERY
    returns = np.zeros((N_SEEDS, n_evals))

    for seed in range(N_SEEDS):
        print(f"\n  Seed {seed+1}/{N_SEEDS} ...", flush=True)
        env      = make_env(seed=seed)
        eval_env = make_env(seed=seed + 100)

        policy_kwargs = dict(
            features_extractor_class=V1Extractor,
            features_extractor_kwargs=dict(ckpt_path=str(ckpt_path)),
            net_arch=[64, 64],
        )
        model = PPO(
            'MlpPolicy', env,
            policy_kwargs=policy_kwargs,
            n_steps=512, batch_size=64, n_epochs=4,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
            learning_rate=3e-4,
            verbose=0,
        )

        step = 0
        while step < TOTAL_STEPS:
            model.learn(total_timesteps=EVAL_EVERY, reset_num_timesteps=False)
            step += EVAL_EVERY
            mean_r, _ = evaluate_policy(model, eval_env, n_eval_episodes=N_EVAL_EPS,
                                        deterministic=True)
            eval_idx = step // EVAL_EVERY - 1
            returns[seed, eval_idx] = mean_r
            print(f"    step {step:>7,}  return={mean_r:.4f}", flush=True)

        env.close(); eval_env.close()

    result_dir.mkdir(parents=True, exist_ok=True)
    np.save(result_dir / 'ppo_returns.npy', returns)
    print(f"\n  Saved → {result_dir / 'ppo_returns.npy'}")
    return returns


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(active, perplexity, probe_results, ppo_returns):
    finals = ppo_returns[:, -1]
    print("\n" + "="*60)
    print("RESULTS SUMMARY — V1 Gradient Baseline (K=32)")
    print("="*60)
    print(f"\n  Codebook:  {active}/{K_V1} active ({100*active/K_V1:.0f}%)  "
          f"perplexity={perplexity:.1f}")
    print(f"\n  Probe accuracies:")
    for label, (acc, base) in probe_results.items():
        print(f"    {label:<16} {acc:.4f}  (baseline {base:.4f})")
    print(f"\n  PPO final returns (5 seeds × {TOTAL_STEPS//1000}k steps):")
    for i, r in enumerate(finals):
        print(f"    seed {i}: {r:.4f}")
    print(f"\n  Mean ± std: {finals.mean():.3f} ± {finals.std():.3f}")
    print(f"  Bootstrap rate (≥0.5): {(finals >= 0.5).sum()}/{N_SEEDS}")

    result_txt = RESULT_DIR / 'probe_summary.txt'
    with open(result_txt, 'w') as f:
        f.write(f"V1 Gradient Baseline — K={K_V1}, beta={BETA_V1}\n")
        f.write(f"Active codes: {active}/{K_V1}\n")
        f.write(f"Perplexity: {perplexity:.2f}\n")
        for label, (acc, base) in probe_results.items():
            f.write(f"{label}: acc={acc:.4f} base={base:.4f}\n")
        f.write(f"PPO mean: {finals.mean():.4f} +/- {finals.std():.4f}\n")
        f.write(f"Seed returns: {finals.tolist()}\n")
    print(f"\n  Summary saved → {result_txt}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    model = train_encoder(DATA_PATH, CKPT_PATH)
    active, perplexity, probe_results = probe_encoder(model, DATA_PATH)
    ppo_returns = run_ppo(CKPT_PATH, RESULT_DIR)
    print_summary(active, perplexity, probe_results, ppo_returns)
