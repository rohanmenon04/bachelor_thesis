"""Quick results summary — print tables and run analysis after training."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULT_PATH = ROOT / 'results' / 'object_centric' / 'ppo_returns.npy'
EVAL_EVERY  = 5_000
BUDGET_200K = 40  # checkpoints


def main():
    if not RESULT_PATH.exists():
        print("No results file found."); return

    r = dict(np.load(RESULT_PATH, allow_pickle=True).item())
    order = ['Raw', 'OC-Agent', 'OC-All', 'OC-All+Pos']

    print("=" * 75)
    print("PPO RESULTS — MiniGrid-DoorKey-5x5 Object-Centric VQ Tokens")
    print("=" * 75)

    print(f"\n{'Condition':<16} {'Seeds':>5} {'Steps':>7}  "
          f"{'@200k final':>12}  {'@200k max':>10}  {'Full final':>10}")
    print("-" * 70)

    for name in order:
        if name not in r:
            print(f"  {name:<14}  [NOT RUN]")
            continue
        v = r[name]
        n_ck = v.shape[1]
        n_seeds = v.shape[0]
        n_steps = n_ck * EVAL_EVERY

        # @200k
        v200 = v[:, :BUDGET_200K]
        m200 = v200.mean(0)[-1]
        s200 = v200.std(0)[-1]
        max200_per_seed = v200.max(1)
        mean_max200 = max200_per_seed.mean()

        # Full final
        m_full = v.mean(0)[-1]
        s_full = v.std(0)[-1]

        print(f"  {name:<14}  {n_seeds:>5}  {n_steps//1000:>5}k  "
              f"  {m200:6.3f}±{s200:.3f}  "
              f"  {mean_max200:7.3f}      "
              f"  {m_full:6.3f}±{s_full:.3f}")

    # Per-seed breakdown for key conditions
    print()
    for name in ['OC-All', 'OC-All+Pos']:
        if name not in r:
            continue
        v = r[name]
        print(f"\n  {name} — per-seed details:")
        print(f"  {'Seed':>4}  {'200k final':>11}  {'200k max':>10}  "
              f"{'bootstrap step':>14}  {'full final':>10}")
        v200 = v[:, :BUDGET_200K]
        for i, (seed_200, seed_full) in enumerate(zip(v200, v)):
            m200 = seed_200[-1]
            max200 = seed_200.max()
            nz = np.where(seed_200 > 0)[0]
            bs = (nz[0] + 1) * EVAL_EVERY if len(nz) else None
            bs_str = f"{bs:>9}" if bs else "   never"
            print(f"  {i:>4}  {m200:>10.3f}  {max200:>10.3f}  "
                  f"  {bs_str}  {seed_full[-1]:>10.3f}")

    # Key comparison
    if 'OC-All' in r and 'OC-All+Pos' in r:
        oc = r['OC-All'][:, :BUDGET_200K]
        pos = r['OC-All+Pos'][:, :BUDGET_200K]
        print(f"\n  === Key comparison at 200k steps ===")
        print(f"  OC-All mean:        {oc.mean(0)[-1]:.3f} ± {oc.std(0)[-1]:.3f}")
        print(f"  OC-All+Pos mean:    {pos.mean(0)[-1]:.3f} ± {pos.std(0)[-1]:.3f}")
        n_boots_oc  = sum(1 for s in oc  if (s > 0).any())
        n_boots_pos = sum(1 for s in pos if (s > 0).any())
        print(f"  OC-All bootstrapping seeds:    {n_boots_oc}/{oc.shape[0]}")
        print(f"  OC-All+Pos bootstrapping seeds:{n_boots_pos}/{pos.shape[0]}")
        print(f"  Peak return improvement: "
              f"{oc.max(1).mean():.3f} → {pos.max(1).mean():.3f}")


if __name__ == '__main__':
    main()
