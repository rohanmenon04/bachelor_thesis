"""
Script 6 — LSTM Policy over VQ Token Stream (Condition E)
==========================================================
Trains a recurrent policy (LSTM) whose only input at each timestep is the
codebook embedding for the current VQ token — one 64-dim vector per step.

This is the most faithful test of the thesis claim "planning over a discrete
embedding": the policy never sees raw pixels or continuous representations.
It receives a stream of discrete symbols (token indices mapped to embeddings)
and must integrate them over time via its hidden state.

Architecture:
  obs_t (7×7×3) → frozen VQ encoder → token index → codebook embedding (64-dim)
                                                            ↓
                                                   LSTM(hidden=64)
                                                    ↙        ↘
                                             policy head   value head
                                           (7 actions)    (scalar)

Compare with:
  Condition C (04_ppo_comparison.py): MLP on single codebook embedding — no memory
  Condition D (04_ppo_comparison.py): MLP on 8-token window — fixed history
  Condition E (this script):          LSTM over token stream — unbounded recurrent memory

The frozen encoder is the same checkpoint used in Conditions C and D.
Only the LSTM and its two linear heads are trained.

Usage:
  python 06_lstm_ppo.py

Requires:
  checkpoints/vq_encoder/vq_encoder.pt

Outputs:
  results/lstm_policy/lstm_returns.npy   — (N_SEEDS, n_checkpoints) array
  plots/lstm_policy/lstm_comparison.png  — LSTM vs Condition C learning curve
"""

import os
import pickle

import gymnasium as gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import minigrid
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from stable_baselines3.common.monitor import Monitor

from models import TransformerVQEncoder, D_MODEL, CODEBOOK_SIZE

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENV_ID        = 'MiniGrid-FourRooms-v0'
TOTAL_STEPS   = 50_000
EVAL_EVERY    =  5_000
N_EVAL_EPS    =     10
N_SEEDS       =      2   # increase to 5 for final results

LSTM_HIDDEN   =     64   # matches D_MODEL — LSTM width equals token embedding width
N_ACTIONS     =      7   # MiniGrid action space

N_ROLLOUT     =    512   # steps per PPO update (same as Condition C/D)
N_EPOCHS      =      4   # PPO update epochs per rollout
GAMMA         =   0.99
GAE_LAMBDA    =   0.95
CLIP_EPS      =    0.2
VF_COEF       =    0.5
ENT_COEF      =   0.01
MAX_GRAD_NORM =    0.5
LR            =   3e-4

DEVICE = (
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class ImgObsWrapper(gym.ObservationWrapper):
    """Extract the image array from MiniGrid's dict observation."""

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


# ---------------------------------------------------------------------------
# Frozen VQ encoder — maps obs to codebook embedding
# ---------------------------------------------------------------------------

class FrozenVQEncoder(nn.Module):
    """
    Wraps the pre-trained TransformerVQEncoder.
    Returns the D_MODEL-dim codebook embedding for the discrete token.
    No gradient flows through this module during policy training.
    """

    def __init__(self, ckpt_path='checkpoints/vq_encoder/vq_encoder.pt'):
        super().__init__()
        self.enc = TransformerVQEncoder()
        self.enc.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        for p in self.enc.parameters():
            p.requires_grad = False
        self.enc.eval()

    @torch.no_grad()
    def forward(self, obs):
        """obs: (B, 7, 7, 3) float32 → (B, D_MODEL) codebook embedding."""
        z       = self.enc.encode(obs.long())
        _, idx, _ = self.enc.vq(z)
        return self.enc.vq.codebook(idx)   # (B, D_MODEL)


# ---------------------------------------------------------------------------
# LSTM policy
# ---------------------------------------------------------------------------

class LSTMPolicy(nn.Module):
    """
    Recurrent actor-critic.
    Input:  codebook embedding (D_MODEL,) at each timestep
    Memory: LSTM hidden state (LSTM_HIDDEN,)
    Output: action logits (N_ACTIONS,) and state value (1,)
    """

    def __init__(self, input_dim=D_MODEL, hidden_dim=LSTM_HIDDEN, n_actions=N_ACTIONS):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head  = nn.Linear(hidden_dim, 1)

    def zero_state(self, batch_size=1, device='cpu'):
        """Initial LSTM hidden state (h, c), both zeros."""
        z = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        return (z, z)

    def step(self, emb, hidden):
        """
        Single-step forward pass for rollout collection.

        Args:
            emb:    (1, D_MODEL) token embedding for current obs
            hidden: (h, c) LSTM hidden state

        Returns:
            action (int), log_prob (scalar), value (scalar), new_hidden
        """
        x = emb.unsqueeze(1)                     # (1, 1, D_MODEL)
        out, hidden = self.lstm(x, hidden)        # out: (1, 1, hidden)
        h = out[:, 0, :]                          # (1, hidden)
        logits = self.policy_head(h)              # (1, n_actions)
        value  = self.value_head(h).squeeze(-1)   # (1,)

        dist    = Categorical(logits=logits)
        action  = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob, value, hidden

    def evaluate_sequences(self, emb_seqs, seq_lengths, device):
        """
        Re-run LSTM over a list of episode sequences for the PPO update.

        Args:
            emb_seqs:    list of tensors, each (T_i, D_MODEL)
            seq_lengths: list of ints

        Returns:
            all_logits: (sum(T_i), N_ACTIONS)
            all_values: (sum(T_i),)
        """
        all_logits = []
        all_values = []

        for seq in emb_seqs:
            T = seq.shape[0]
            x = seq.unsqueeze(0).to(device)               # (1, T, D_MODEL)
            h0 = self.zero_state(batch_size=1, device=device)
            out, _ = self.lstm(x, h0)                     # (1, T, hidden)
            out = out.squeeze(0)                           # (T, hidden)
            all_logits.append(self.policy_head(out))       # (T, n_actions)
            all_values.append(self.value_head(out).squeeze(-1))  # (T,)

        return torch.cat(all_logits, dim=0), torch.cat(all_values, dim=0)


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

def collect_rollout(env, policy, encoder, n_steps, device):
    """
    Collect n_steps environment transitions.

    Tracks episode boundaries so the LSTM update can re-run sequences
    from their correct start states (always zero-state at episode start).

    Returns:
        emb_seqs:      list of (T_i, D_MODEL) tensors, one per episode/chunk
        action_seqs:   list of (T_i,) tensors
        logprob_seqs:  list of (T_i,) tensors
        value_seqs:    list of (T_i,) tensors
        return_seqs:   list of (T_i,) tensors  — GAE targets
        adv_seqs:      list of (T_i,) tensors  — GAE advantages
    """
    emb_buf     = []
    action_buf  = []
    logprob_buf = []
    value_buf   = []
    reward_buf  = []
    done_buf    = []

    obs, _ = env.reset()
    hidden = policy.zero_state(batch_size=1, device=device)

    for _ in range(n_steps):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        emb   = encoder(obs_t)                            # (1, D_MODEL)

        action, log_prob, value, hidden = policy.step(emb.detach(), hidden)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        emb_buf.append(emb.squeeze(0).cpu())
        action_buf.append(action)
        logprob_buf.append(log_prob.detach().cpu())
        value_buf.append(value.detach().cpu().squeeze())
        reward_buf.append(float(reward))
        done_buf.append(float(done))

        if done:
            obs, _ = env.reset()
            hidden = policy.zero_state(batch_size=1, device=device)
        else:
            obs = next_obs

    # Bootstrap value for the last step if not done
    if not done_buf[-1]:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        emb   = encoder(obs_t)
        _, _, last_val, _ = policy.step(emb.detach(), hidden)
        last_val = last_val.detach().cpu().squeeze().item()
    else:
        last_val = 0.0

    # GAE advantage computation
    advantages = _compute_gae(
        reward_buf, value_buf, done_buf, last_val, GAMMA, GAE_LAMBDA
    )
    returns = [adv + val.item() for adv, val in zip(advantages, value_buf)]

    # Split into episode sequences at done boundaries
    emb_seqs, action_seqs, logprob_seqs, value_seqs, return_seqs, adv_seqs = \
        [], [], [], [], [], []

    seq_start = 0
    for t in range(n_steps):
        if done_buf[t] or t == n_steps - 1:
            end = t + 1
            emb_seqs.append(torch.stack(emb_buf[seq_start:end]))
            action_seqs.append(torch.tensor(action_buf[seq_start:end], dtype=torch.long))
            logprob_seqs.append(torch.stack(logprob_buf[seq_start:end]))
            value_seqs.append(torch.stack(value_buf[seq_start:end]))
            return_seqs.append(torch.tensor(returns[seq_start:end], dtype=torch.float32))
            adv_seqs.append(torch.tensor(advantages[seq_start:end], dtype=torch.float32))
            seq_start = end

    return emb_seqs, action_seqs, logprob_seqs, value_seqs, return_seqs, adv_seqs


def _compute_gae(rewards, values, dones, last_value, gamma, lam):
    """Generalized Advantage Estimation (Schulman et al., 2015)."""
    T = len(rewards)
    advantages = [0.0] * T
    gae = 0.0
    next_val = last_value

    for t in reversed(range(T)):
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t].item()
        gae   = delta + gamma * lam * next_non_terminal * gae
        advantages[t] = gae
        next_val = values[t].item()

    return advantages


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def ppo_update(policy, optimizer, emb_seqs, action_seqs, logprob_seqs,
               return_seqs, adv_seqs, device):
    """
    N_EPOCHS passes of PPO over the collected sequences.

    Each epoch re-runs the LSTM over each episode sequence from its zero
    initial state, then computes the clipped surrogate loss across all steps.
    Sequences are not minibatched — the full rollout is processed at once.
    (T=512 is small enough that this is not a bottleneck.)
    """
    # Flatten non-sequence tensors for loss computation
    actions_flat  = torch.cat(action_seqs,  dim=0).to(device)
    logprobs_flat = torch.cat(logprob_seqs, dim=0).to(device)
    returns_flat  = torch.cat(return_seqs,  dim=0).to(device)
    adv_flat      = torch.cat(adv_seqs,     dim=0).to(device)

    # Normalise advantages
    adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

    metrics = {'policy_loss': [], 'value_loss': [], 'entropy': []}

    for _ in range(N_EPOCHS):
        new_logits, new_values = policy.evaluate_sequences(emb_seqs, None, device)

        dist         = Categorical(logits=new_logits)
        new_log_prob = dist.log_prob(actions_flat)
        entropy      = dist.entropy().mean()

        ratio  = (new_log_prob - logprobs_flat).exp()
        surr1  = ratio * adv_flat
        surr2  = ratio.clamp(1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_flat
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = F.mse_loss(new_values, returns_flat)

        loss = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        metrics['policy_loss'].append(policy_loss.item())
        metrics['value_loss'].append(value_loss.item())
        metrics['entropy'].append(entropy.item())

    return {k: np.mean(v) for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(env_factory, policy, encoder, n_episodes, seed, device):
    """Run n_episodes with greedy actions, return mean episodic return."""
    env = env_factory(seed)
    returns = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        hidden = policy.zero_state(batch_size=1, device=device)
        ep_return = 0.0
        done = False

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            emb   = encoder(obs_t)
            x     = emb.unsqueeze(1)
            out, hidden = policy.lstm(x, hidden)
            logits = policy.policy_head(out[:, 0, :])
            action = logits.argmax(dim=-1).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            done = terminated or truncated

        returns.append(ep_return)

    env.close()
    return np.mean(returns)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_condition_e(env_factory, seed):
    """Train Condition E (LSTM over token stream) for one seed."""
    print(f"  Device: {DEVICE}")

    encoder = FrozenVQEncoder().to(DEVICE)
    encoder.eval()

    policy    = LSTMPolicy(D_MODEL, LSTM_HIDDEN, N_ACTIONS).to(DEVICE)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

    env = env_factory(seed)

    n_ckpts  = TOTAL_STEPS // EVAL_EVERY
    rollouts_per_eval = EVAL_EVERY // N_ROLLOUT
    returns  = []

    total_steps = 0

    for ckpt_i in range(n_ckpts):
        policy.train()

        for _ in range(rollouts_per_eval):
            emb_seqs, action_seqs, logprob_seqs, value_seqs, return_seqs, adv_seqs = \
                collect_rollout(env, policy, encoder, N_ROLLOUT, DEVICE)
            total_steps += N_ROLLOUT

            metrics = ppo_update(
                policy, optimizer,
                emb_seqs, action_seqs, logprob_seqs,
                return_seqs, adv_seqs,
                DEVICE,
            )

        policy.eval()
        mean_r = evaluate(env_factory, policy, encoder, N_EVAL_EPS, seed + 10_000, DEVICE)
        returns.append(mean_r)
        print(
            f"    step={total_steps:>6d}  return={mean_r:+.2f}  "
            f"policy_loss={metrics['policy_loss']:.3f}  value_loss={metrics['value_loss']:.3f}  "
            f"entropy={metrics['entropy']:.3f}"
        )

    env.close()
    return returns


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(lstm_results, cond_c_results, save_path):
    """
    Plot Condition E (LSTM) against Condition C (Discrete VQ, MLP) for comparison.
    cond_c_results may be None if the PPO comparison data is not available.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    steps = np.arange(1, lstm_results.shape[1] + 1) * EVAL_EVERY

    mean_e = lstm_results.mean(0)
    std_e  = lstm_results.std(0)
    ax.plot(steps, mean_e, color='#e74c3c', label='E: LSTM (token stream)', linewidth=2)
    ax.fill_between(steps, mean_e - std_e, mean_e + std_e, color='#e74c3c', alpha=0.15)

    if cond_c_results is not None:
        mean_c = cond_c_results.mean(0)
        std_c  = cond_c_results.std(0)
        ax.plot(steps, mean_c, color='#2ecc71',
                label='C: Discrete VQ — MLP (no memory)', linewidth=2, linestyle='--')
        ax.fill_between(steps, mean_c - std_c, mean_c + std_c, color='#2ecc71', alpha=0.15)

    ax.set_xlabel('Environment Steps', fontsize=12)
    ax.set_ylabel('Mean Episodic Return', fontsize=12)
    ax.set_title(
        f'Condition E: LSTM over VQ Token Stream vs Condition C\n'
        f'MiniGrid-FourRooms  '
        f'({N_SEEDS} seed{"s" if N_SEEDS > 1 else ""}, {N_EVAL_EPS} eval episodes)',
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
    os.makedirs('results/lstm_policy', exist_ok=True)
    os.makedirs('plots/lstm_policy',   exist_ok=True)

    print(f"=== Condition E: LSTM over VQ Token Stream ===")
    print(f"LSTM hidden: {LSTM_HIDDEN}  |  D_MODEL: {D_MODEL}  |  Total steps: {TOTAL_STEPS:,}")

    seed_returns = []

    for seed_i in range(N_SEEDS):
        seed = seed_i * 100
        print(f"\n  Seed {seed_i} (seed={seed}):")
        rets = train_condition_e(make_env, seed)
        seed_returns.append(rets)

    lstm_results = np.array(seed_returns)   # (N_SEEDS, n_ckpts)
    np.save('results/lstm_policy/lstm_returns.npy', lstm_results)
    print("\nSaved results/lstm_policy/lstm_returns.npy")

    # Load Condition C from previous experiment if available
    cond_c = None
    ppo_path = 'results/ppo_baselines/ppo_returns.npy'
    if os.path.exists(ppo_path):
        ppo_data = np.load(ppo_path, allow_pickle=True).item()
        cond_c = ppo_data.get('C_Discrete_VQ')
        if cond_c is not None:
            print("Loaded Condition C results for comparison.")

    plot_results(lstm_results, cond_c, 'plots/lstm_policy/lstm_comparison.png')

    print("\n=== Final Performance (last checkpoint) ===")
    m, s = lstm_results[:, -1].mean(), lstm_results[:, -1].std()
    print(f"  Condition E (LSTM): {m:.2f} ± {s:.2f}")
    if cond_c is not None:
        mc, sc = cond_c[:, -1].mean(), cond_c[:, -1].std()
        print(f"  Condition C (VQ-MLP): {mc:.2f} ± {sc:.2f}")


if __name__ == '__main__':
    main()
