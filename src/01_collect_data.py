"""
Script 1 — Data Collection
===========================
Collects random-policy trajectories from MiniGrid-DoorKey-5x5 and saves
observations alongside ground-truth semantic state labels.

The semantic labels are extracted from the environment's internal state
(not from the observation image). They are used ONLY for evaluation
(probing in script 03) — the encoder is never trained with them.

Labels collected:
  door_open:     1 if the door is currently open
  door_locked:   1 if the door is currently locked
  carrying_key:  1 if the agent is carrying the key
  key_on_ground: 1 if the key is still on the floor (not carried)

Run first, before any other script.

Usage:
  cd experiments/
  python 01_collect_data.py

Outputs:
  data/trajectories.pkl
  plots/observations.png
"""

import os
import pickle

import gymnasium as gym
import minigrid   # registers MiniGrid environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ENV_ID     = 'MiniGrid-DoorKey-5x5-v0'
N_EPISODES = 600
MAX_STEPS  = 100   # MiniGrid DoorKey-5x5 default horizon is 100


# ---------------------------------------------------------------------------
# Semantic label extraction
# ---------------------------------------------------------------------------

def get_semantic_labels(env):
    """
    Read ground-truth symbolic state from MiniGrid's internal grid.
    Called at every timestep; labels are NOT passed to the encoder.
    """
    grid     = env.unwrapped.grid
    carrying = env.unwrapped.carrying

    labels = {
        'door_open':     0,
        'door_locked':   0,
        'carrying_key':  0,
        'key_on_ground': 0,
    }

    for x in range(grid.width):
        for y in range(grid.height):
            obj = grid.get(x, y)
            if obj is None:
                continue
            if obj.type == 'door':
                labels['door_open']   = int(obj.is_open)
                labels['door_locked'] = int(obj.is_locked)
            elif obj.type == 'key':
                labels['key_on_ground'] = 1

    if carrying is not None and carrying.type == 'key':
        labels['carrying_key']  = 1
        labels['key_on_ground'] = 0

    return labels


# ---------------------------------------------------------------------------
# Trajectory collection
# ---------------------------------------------------------------------------

def collect_trajectories(env_id, n_episodes, max_steps, base_seed=42):
    env = gym.make(env_id)
    observations, label_list = [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        img = obs['image']

        for _ in range(max_steps):
            labels = get_semantic_labels(env)
            observations.append(img.copy())
            label_list.append(labels)

            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            img = obs['image']

            if terminated or truncated:
                break

    env.close()
    return np.array(observations, dtype=np.int64), label_list


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def print_stats(observations, labels):
    print(f"\nDataset size:     {len(observations)} observations")
    print(f"Observation shape: {observations.shape}  dtype={observations.dtype}")
    print(f"Value range:       [{observations.min()}, {observations.max()}]")
    print("\nSemantic label distribution (positive rate):")
    for key in labels[0]:
        rate = np.mean([l[key] for l in labels])
        bar  = '█' * int(rate * 20)
        print(f"  {key:<18} {100*rate:5.1f}%  {bar}")


def visualize(observations, labels, save_path):
    n = 10
    idxs = np.linspace(0, len(observations) - 1, n, dtype=int)

    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    fig.suptitle('MiniGrid-DoorKey-5x5: Sample Observations', fontsize=11)

    for ax, idx in zip(axes.flat, idxs):
        img = observations[idx]
        lbl = labels[idx]
        # Scale type channel (0-10) for display
        display = np.clip(img[:, :, 0:1] * 25, 0, 255).repeat(3, axis=2).astype(np.uint8)
        ax.imshow(display, vmin=0, vmax=255)
        ax.set_title(
            f"open={lbl['door_open']}  lock={lbl['door_locked']}\n"
            f"carry={lbl['carrying_key']}",
            fontsize=7,
        )
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs('data',  exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    print(f"Environment : {ENV_ID}")
    print(f"Episodes    : {N_EPISODES} (max {MAX_STEPS} steps each)")
    print("Collecting trajectories with random policy...")

    observations, labels = collect_trajectories(ENV_ID, N_EPISODES, MAX_STEPS)
    print_stats(observations, labels)
    visualize(observations, labels, 'plots/observations.png')

    dataset = {
        'observations': observations,
        'labels':       labels,
        'env_id':       ENV_ID,
    }
    with open('data/trajectories.pkl', 'wb') as f:
        pickle.dump(dataset, f)

    print("\nSaved data/trajectories.pkl")
    print("Next: python 02_train_semantic_encoder.py")


if __name__ == '__main__':
    main()
