"""Estimate the time-by-window action density of the validated policy.

The heatmap is intentionally descriptive: at each action step it reports the
empirical distribution of the window actually filled, conditional on a fill
occurring at that step.  Reset actions are counted separately and are not
treated as a fictitious window.
"""

import argparse
import csv
import json

import numpy as np

import azul_policy_search as model
from azul_policy_insights import train_current_predictor


def trace_batch(predictor, uniforms, threshold):
    """Return fill counts and reset counts for one vectorized batch."""
    episodes = uniforms.shape[0]
    x = np.zeros((episodes, model.N), dtype=np.int8)
    g = np.zeros(episodes, dtype=np.int8)
    rows = np.arange(episodes)
    fill_counts = np.zeros((model.H, model.N), dtype=np.int64)
    optional_resets = np.zeros(model.H, dtype=np.int64)
    forced_resets = np.zeros(model.H, dtype=np.int64)

    for step in range(model.H):
        h = model.H - step
        active = x < 6
        any_active = active.any(axis=1)
        left = np.where(any_active, np.argmax(active, axis=1), 0).astype(np.int8)
        p = model.P[x % 3]
        mask = (uniforms[:, step, :] < p) & active
        reachable = model.IDX[None, :] >= g[:, None]
        legal = mask & reachable

        util = model.features(x, g, h, model.H) @ model.VALIDATED_SECOND_ORDER_WEIGHTS
        legal_util = np.where(legal, util, -1e9)
        best_i = np.argmax(legal_util, axis=1).astype(np.int8)
        best_u = legal_util[rows, best_i]
        has_legal = legal.any(axis=1)

        z = model.reset_decision_features(
            x, g, h, best_i, best_u, util, p, active, legal, left
        )
        predicted_advantage = np.asarray(predictor.predict(z))
        can_reset = any_active & has_legal & (g != left)
        optional_reset = can_reset & (predicted_advantage > threshold)
        forced_reset = any_active & ~has_legal
        do_reset = optional_reset | forced_reset
        do_fill = any_active & has_legal & ~optional_reset

        optional_resets[step] = int(optional_reset.sum())
        forced_resets[step] = int(forced_reset.sum())
        if np.any(do_fill):
            chosen = best_i[do_fill]
            fill_counts[step] = np.bincount(chosen, minlength=model.N)
            rr = rows[do_fill]
            x[rr, chosen] += 1
            g[rr] = chosen
        g[do_reset] = left[do_reset]

    return fill_counts, optional_resets, forced_resets


def estimate(predictor, episodes, batch_size, seed, threshold):
    rng = np.random.default_rng(seed)
    fill_counts = np.zeros((model.H, model.N), dtype=np.int64)
    optional_resets = np.zeros(model.H, dtype=np.int64)
    forced_resets = np.zeros(model.H, dtype=np.int64)

    done = 0
    while done < episodes:
        size = min(batch_size, episodes - done)
        uniforms = rng.random((size, model.H, model.N), dtype=np.float32)
        batch = trace_batch(predictor, uniforms, threshold)
        fill_counts += batch[0]
        optional_resets += batch[1]
        forced_resets += batch[2]
        done += size
        print(json.dumps({"stage": "heatmap_progress", "done": done}), flush=True)

    fill_total = fill_counts.sum(axis=1)
    probability = fill_counts / np.maximum(1, fill_total[:, None])
    window = np.arange(1, model.N + 1, dtype=np.float64)
    mean = probability @ window
    variance = ((window[None, :] - mean[:, None]) ** 2 * probability).sum(axis=1)
    sd = np.sqrt(variance)
    return {
        "fill_counts": fill_counts,
        "fill_total": fill_total,
        "probability": probability,
        "mean": mean,
        "sd": sd,
        "optional_resets": optional_resets,
        "forced_resets": forced_resets,
    }


def write_csv(path, result, episodes):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "step", "window", "fill_count", "fill_probability",
            "fill_total", "mean_window", "sd_window",
            "optional_reset_rate", "forced_reset_rate",
        ])
        for step in range(model.H):
            for window in range(model.N):
                writer.writerow([
                    step, window + 1,
                    int(result["fill_counts"][step, window]),
                    f'{result["probability"][step, window]:.8f}',
                    int(result["fill_total"][step]),
                    f'{result["mean"][step]:.8f}',
                    f'{result["sd"][step]:.8f}',
                    f'{result["optional_resets"][step] / episodes:.8f}',
                    f'{result["forced_resets"][step] / episodes:.8f}',
                ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20261841)
    parser.add_argument("--predictor-seed", type=int, default=20260841)
    parser.add_argument("--collection-episodes", type=int, default=10000)
    parser.add_argument("--states", type=int, default=1600)
    parser.add_argument("--rollouts", type=int, default=128)
    parser.add_argument("--output", default="azul_policy_heatmap.csv")
    args = parser.parse_args()

    predictor = train_current_predictor(
        args.predictor_seed, args.collection_episodes, args.states, args.rollouts
    )
    result = estimate(
        predictor, args.episodes, args.batch_size, args.seed, args.threshold
    )
    write_csv(args.output, result, args.episodes)
    print(json.dumps({
        "stage": "heatmap_complete",
        "episodes": args.episodes,
        "threshold": args.threshold,
        "output": args.output,
        "max_probability": float(result["probability"].max()),
        "mean_window_at_steps": {
            str(step): float(result["mean"][step])
            for step in (0, 5, 10, 15, 20, 25, 30, 35, 40, 44)
        },
    }), flush=True)


if __name__ == "__main__":
    main()
