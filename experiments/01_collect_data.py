"""
Script 1 — Data Collection
===========================
Collects random-policy trajectories from MiniGrid-Empty-5x5 and saves
observations alongside ground-truth semantic state labels.

The semantic labels are extracted from the environment's internal state
(not from the observation image). They are used ONLY for evaluation
(probing in script 03) — the encoder is never trained with them.

Labels collected:
  goal_visible:  1 if the goal tile (type=8) appears in the 7x7 observation
  near_goal:     1 if Manhattan distance from agent to goal <= 2
  facing_goal:   1 if agent's facing direction is aligned with goal direction

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


ENV_ID     = 'MiniGrid-Empty-5x5-v0'
N_EPISODES = 600
MAX_STEPS  = 100   # MiniGrid Empty-5x5 default horizon is 100


# ---------------------------------------------------------------------------
# Semantic label extraction
# ---------------------------------------------------------------------------

def get_semantic_labels(env, obs_img):
    """
    Read ground-truth symbolic state for MiniGrid-Empty-5x5.
    Called at every timestep; labels are NOT passed to the encoder.

    goal_visible: goal tile (object type 8) is present in the 7x7 observation.
    near_goal:    Manhattan distance from agent to goal <= 2.
    facing_goal:  agent's facing direction is the primary direction toward the goal.
    """
    grid      = env.unwrapped.grid
    agent_pos = env.unwrapped.agent_pos   # (x, y)
    agent_dir = env.unwrapped.agent_dir   # 0=right, 1=down, 2=left, 3=up

    # Find goal cell (always present in Empty env)
    goal_pos = None
    for x in range(grid.width):
        for y in range(grid.height):
            obj = grid.get(x, y)
            if obj is not None and obj.type == 'goal':
                goal_pos = (x, y)
                break
        if goal_pos is not None:
            break

    labels = {'goal_visible': 0, 'near_goal': 0, 'facing_goal': 0}

    if goal_pos is None:
        return labels

    # goal_visible: object-type channel value 8 = GOAL tile
    labels['goal_visible'] = int(np.any(obs_img[:, :, 0] == 8))

    # near_goal: Manhattan distance
    dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
    labels['near_goal'] = int(dist <= 2)

    # facing_goal: agent direction matches the dominant axis toward goal
    # dir 0=right(+x), 1=down(+y), 2=left(-x), 3=up(-y)
    dx = goal_pos[0] - agent_pos[0]
    dy = goal_pos[1] - agent_pos[1]
    labels['facing_goal'] = int(
        (agent_dir == 0 and dx > 0) or
        (agent_dir == 1 and dy > 0) or
        (agent_dir == 2 and dx < 0) or
        (agent_dir == 3 and dy < 0)
    )

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
            labels = get_semantic_labels(env, img)
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
    fig.suptitle('MiniGrid-Empty-5x5: Sample Observations', fontsize=11)

    for ax, idx in zip(axes.flat, idxs):
        img = observations[idx]
        lbl = labels[idx]
        # Scale type channel (0-10) for display
        display = np.clip(img[:, :, 0:1] * 25, 0, 255).repeat(3, axis=2).astype(np.uint8)
        ax.imshow(display, vmin=0, vmax=255)
        ax.set_title(
            f"vis={lbl['goal_visible']}  near={lbl['near_goal']}\n"
            f"face={lbl['facing_goal']}",
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
    print("Next: python 02_train_encoder.py")


if __name__ == '__main__':
    main()
