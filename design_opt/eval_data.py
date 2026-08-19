import argparse
import os
import sys
import csv
import math
sys.path.append(os.getcwd())

import yaml
from omegaconf import OmegaConf

from khrylib.utils import *
from design_opt.utils.config import Config
from design_opt.agents.genesis_agent import BodyGenAgent

project_path = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', type=str)
parser.add_argument('--epoch', default='best')
parser.add_argument('--save_video', action='store_true', default=False)
parser.add_argument('--pause_design', action='store_true', default=False)
parser.add_argument('--num_seeds', type=int, default=1)
parser.add_argument('--seed_start', type=int, default=None)
args = parser.parse_args()

train_dir = os.path.join(args.train_dir, "0")

train_config_path = os.path.join(train_dir, ".hydra", "config.yaml")

FLAGS = yaml.safe_load(open(train_config_path, 'r'))
FLAGS = OmegaConf.create(FLAGS)

cfg = Config(FLAGS, project_path, base_dir=train_dir)

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cpu')


def _read_rewards_from_eval_csv(eval_csv_path):
    rewards = []
    if not os.path.isfile(eval_csv_path):
        return rewards

    with open(eval_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rewards.append(float(row['reward']))
            except Exception:
                continue
    return rewards


def _t_critical_95(df):
    # Two-sided t critical values at 95% confidence for df in [1, 30].
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    }
    if df <= 0:
        return float('nan')
    if df in table:
        return table[df]
    return 1.96


def _mean_ci(values):
    x = np.asarray(values, dtype=np.float64)
    x = x[~np.isnan(x)]
    n = int(x.size)
    if n == 0:
        return float('nan'), float('nan'), float('nan'), 0
    mean = float(np.mean(x))
    if n == 1:
        return mean, mean, mean, 1
    std = float(np.std(x, ddof=1))
    se = std / math.sqrt(n)
    tcrit = _t_critical_95(n - 1)
    delta = tcrit * se
    return mean, mean - delta, mean + delta, n


def _save_multi_seed_summary(base_dir, records):
    summary_csv = os.path.join(base_dir, 'summary.csv')
    with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'run_idx', 'seed', 'run_dir', 'eval_csv_path',
            'episode_return', 'episode_length', 'mean_step_reward', 'final_reward'
        ])
        for rec in records:
            writer.writerow([
                rec['run_idx'],
                rec['seed'],
                rec['run_dir'],
                rec['eval_csv_path'],
                rec['episode_return'],
                rec['episode_length'],
                rec['mean_step_reward'],
                rec['final_reward'],
            ])

    values = [rec['episode_return'] for rec in records]
    mean, ci_low, ci_high, n = _mean_ci(values)

    stats_csv = os.path.join(base_dir, 'summary_stats.csv')
    with open(stats_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'n', 'mean', 'ci95_low', 'ci95_high'])
        writer.writerow(['episode_return', n, mean, ci_low, ci_high])

    reward_sequences = [np.asarray(rec['rewards'], dtype=np.float64) for rec in records]
    max_len = max((arr.size for arr in reward_sequences), default=0)
    if max_len > 0:
        reward_matrix = np.full((len(reward_sequences), max_len), np.nan, dtype=np.float64)
        for i, arr in enumerate(reward_sequences):
            reward_matrix[i, :arr.size] = arr

        step_summary_csv = os.path.join(base_dir, 'reward_step_summary.csv')
        with open(step_summary_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'n', 'mean_reward', 'ci95_low', 'ci95_high'])
            for step in range(max_len):
                step_vals = reward_matrix[:, step]
                m, lo, hi, n_step = _mean_ci(step_vals)
                writer.writerow([step, n_step, m, lo, hi])

        try:
            import matplotlib.pyplot as plt

            x = np.arange(len(values), dtype=np.int32)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.scatter(x, values, label='Per-seed episode return', color='tab:blue')
            if len(values) > 0 and not np.isnan(mean):
                ax.axhline(mean, color='tab:orange', linestyle='--', label='Mean')
            if len(values) > 1 and not (np.isnan(ci_low) or np.isnan(ci_high)):
                ax.fill_between(x, ci_low, ci_high, color='tab:orange', alpha=0.2, label='95% CI')
            ax.set_xlabel('Run Index')
            ax.set_ylabel('Episode Return')
            ax.set_title('Multi-Seed Episode Return')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')
            fig.tight_layout()
            fig.savefig(os.path.join(base_dir, 'episode_return_mean_ci.png'), dpi=140)
            plt.close(fig)

            means = []
            ci_lows = []
            ci_highs = []
            n_eff = []
            steps = np.arange(max_len, dtype=np.int32)
            for step in range(max_len):
                m, lo, hi, n_step = _mean_ci(reward_matrix[:, step])
                means.append(m)
                ci_lows.append(lo)
                ci_highs.append(hi)
                n_eff.append(n_step)

            means = np.asarray(means, dtype=np.float64)
            ci_lows = np.asarray(ci_lows, dtype=np.float64)
            ci_highs = np.asarray(ci_highs, dtype=np.float64)
            n_eff = np.asarray(n_eff, dtype=np.int32)

            valid = n_eff > 0
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(steps[valid], means[valid], label='Mean reward', color='tab:green')
            ci_valid = valid & (~np.isnan(ci_lows)) & (~np.isnan(ci_highs))
            if np.any(ci_valid):
                ax.fill_between(steps[ci_valid], ci_lows[ci_valid], ci_highs[ci_valid],
                                color='tab:green', alpha=0.2, label='95% CI')
            ax.set_xlabel('Step')
            ax.set_ylabel('Reward')
            ax.set_title('Reward Over Step (Mean +/- 95% CI Across Seeds)')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')
            fig.tight_layout()
            fig.savefig(os.path.join(base_dir, 'reward_step_mean_ci.png'), dpi=140)
            plt.close(fig)
        except Exception as exc:
            print(f'Could not generate matplotlib summary plots: {exc}')


epoch = int(args.epoch) if args.epoch.isnumeric() else args.epoch

num_seeds = max(1, int(args.num_seeds))
seed_start = cfg.seed if args.seed_start is None else int(args.seed_start)

if num_seeds == 1:
    run_seed = int(seed_start)
    cfg.seed = run_seed
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)

    agent = BodyGenAgent(cfg=cfg, dtype=dtype, device=device, seed=run_seed, num_threads=1, training=False, checkpoint=epoch)

    eval_dir = os.path.join(train_dir, 'eval_data')
    os.makedirs(eval_dir, exist_ok=True)
    eval_csv_path = os.path.join(eval_dir, 'eval_data.csv')
    agent.eval_data(out_csv_path=eval_csv_path)
else:
    multi_seed_dir = os.path.join(train_dir, 'eval_seeds')
    os.makedirs(multi_seed_dir, exist_ok=True)
    records = []

    for run_idx in range(num_seeds):
        run_seed = int(seed_start + run_idx)
        run_dir = os.path.join(multi_seed_dir, f'run_{run_idx:03d}_seed_{run_seed}')
        os.makedirs(run_dir, exist_ok=True)

        cfg.seed = run_seed
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)

        agent = BodyGenAgent(cfg=cfg, dtype=dtype, device=device, seed=run_seed, num_threads=1, training=False, checkpoint=epoch)

        eval_dir = os.path.join(run_dir, 'eval_data')
        os.makedirs(eval_dir, exist_ok=True)
        eval_csv_path = os.path.join(eval_dir, 'eval_data.csv')
        agent.eval_data(out_csv_path=eval_csv_path)

        rewards = _read_rewards_from_eval_csv(eval_csv_path)
        episode_return = float(np.sum(rewards)) if len(rewards) > 0 else float('nan')
        episode_length = int(len(rewards))
        mean_step_reward = float(np.mean(rewards)) if len(rewards) > 0 else float('nan')
        final_reward = float(rewards[-1]) if len(rewards) > 0 else float('nan')

        records.append({
            'run_idx': run_idx,
            'seed': run_seed,
            'run_dir': run_dir,
            'eval_csv_path': eval_csv_path,
            'episode_return': episode_return,
            'episode_length': episode_length,
            'mean_step_reward': mean_step_reward,
            'final_reward': final_reward,
            'rewards': rewards,
        })

    _save_multi_seed_summary(multi_seed_dir, records)
