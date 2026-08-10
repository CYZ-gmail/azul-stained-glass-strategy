"""Behavioral audit of the validated fill and expanded-reset policy."""

import argparse
import json

import numpy as np

import azul_policy_search as model
import azul_reset_rollout as reset_search


def train_current_predictor(seed, collection_episodes, states, rollouts):
    """Reproduce the selected 200-feature reset regressor."""
    rng = np.random.default_rng(seed)
    uniforms = rng.random(
        (collection_episodes, model.H, model.N), dtype=np.float32
    )
    records = reset_search.collect_optional_states(
        rng, collection_episodes, uniforms
    )
    records = reset_search.sample_records(records, rng, states)
    target, target_se, _ = reset_search.rollout_targets(
        records, rng, rollouts
    )
    _, _, predictor, predictor_name = reset_search.fit_models(
        records["features"], target, target_se, rng
    )
    print(json.dumps({
        "stage": "audit_predictor",
        "model": predictor_name,
        "features": len(model.RESET_FEATURE_NAMES),
        "states": len(target),
        "rollouts_per_state": rollouts,
    }), flush=True)
    return predictor


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "se": float(values.std(ddof=1) / np.sqrt(len(values))),
    }


def forced_opening_audit(predictor, rng, episodes, threshold):
    """Compare each forced first fill with common future random masks."""
    uniforms = rng.random(
        (episodes, model.H, model.N), dtype=np.float32
    )
    score = np.empty((episodes, model.N), dtype=np.float32)
    completions = np.empty((episodes, model.N), dtype=np.int8)
    resets = np.empty((episodes, model.N), dtype=np.int16)
    for i in range(model.N):
        x = np.zeros(model.N, dtype=np.int8)
        x[i] = 1
        result = model.simulate(
            model.VALIDATED_SECOND_ORDER_WEIGHTS,
            model.VALIDATED_SECOND_ORDER_RESET,
            uniforms,
            initial_x=x,
            initial_g=i,
            start_step=1,
            reset_predictor=predictor,
            reset_threshold=threshold,
        )
        score[:, i] = result["score"]
        completions[:, i] = result["completions"]
        resets[:, i] = result["resets"]
        print(json.dumps({
            "stage": "forced_opening_progress", "window": i + 1,
            "score": float(score[:, i].mean()),
        }), flush=True)

    means = score.mean(axis=0)
    best = int(np.argmax(means))
    rows = []
    for i in range(model.N):
        diff = score[:, best].astype(np.float64) - score[:, i]
        rows.append({
            "window": i + 1,
            "score": float(means[i]),
            "score_se": _summary(score[:, i])["se"],
            "gap_to_best": float(diff.mean()),
            "gap_se": float(diff.std(ddof=1) / np.sqrt(episodes)),
            "relative_gap_percent": float(100.0 * diff.mean() / means[best]),
            "completions": float(completions[:, i].mean()),
            "resets": float(resets[:, i].mean()),
        })

    x0 = np.zeros((1, model.N), dtype=np.int8)
    g0 = np.zeros(1, dtype=np.int8)
    opening_utility = (
        model.features(x0, g0, model.H, model.H)[0]
        @ model.VALIDATED_SECOND_ORDER_WEIGHTS
    )
    print(json.dumps({
        "stage": "forced_opening",
        "episodes": episodes,
        "best_window": best + 1,
        "policy_utility_ranking": (
            np.argsort(-opening_utility) + 1
        ).astype(int).tolist(),
        "rollout_ranking": (np.argsort(-means) + 1).astype(int).tolist(),
        "rows": rows,
    }), flush=True)


def _concat(records, key, dtype=None):
    value = np.concatenate([record[key] for record in records])
    return value.astype(dtype, copy=False) if dtype is not None else value


def _group_rate(labels, decision, valid=None):
    labels = np.asarray(labels)
    decision = np.asarray(decision, dtype=bool)
    if valid is None:
        valid = np.ones(len(labels), dtype=bool)
    result = []
    for value in np.unique(labels[valid]):
        take = valid & (labels == value)
        result.append({
            "group": int(value), "n": int(take.sum()),
            "rate": float(decision[take].mean()),
        })
    return result


def _phase(step):
    return np.minimum(step // 15, 2).astype(np.int8)


def policy_trace_audit(predictor, rng, episodes, threshold):
    """Trace decisions under the learned reset policy and aggregate patterns."""
    uniforms = rng.random(
        (episodes, model.H, model.N), dtype=np.float32
    )
    x = np.zeros((episodes, model.N), dtype=np.int8)
    g = np.zeros(episodes, dtype=np.int8)
    rows = np.arange(episodes)
    score = np.zeros(episodes, dtype=np.float32)
    completions = np.zeros(episodes, dtype=np.int8)
    optional_records = []
    fill_records = []
    activated_curve = []
    first_completion_count = np.zeros(model.N, dtype=np.int64)
    second_completion_count = np.zeros(model.N, dtype=np.int64)
    first_completion_step = np.zeros(model.N, dtype=np.float64)
    second_completion_step = np.zeros(model.N, dtype=np.float64)
    optional_resets = np.zeros(episodes, dtype=np.int16)
    forced_resets = np.zeros(episodes, dtype=np.int16)

    for step in range(model.H):
        h = model.H - step
        q = x % 3
        c = x // 3
        active = x < 6
        activated = x >= 3
        activated_curve.append(float(activated.sum(axis=1).mean()))
        any_active = active.any(axis=1)
        left = np.where(any_active, np.argmax(active, axis=1), 0).astype(np.int8)
        p = model.P[q]
        mask = (uniforms[:, step, :] < p) & active
        reachable = model.IDX[None, :] >= g[:, None]
        legal = mask & reachable
        f = model.features(x, g, h, model.H)
        util = f @ model.VALIDATED_SECOND_ORDER_WEIGHTS
        legal_util = np.where(legal, util, -1e9)
        best_i = np.argmax(legal_util, axis=1).astype(np.int8)
        best_u = legal_util[rows, best_i]
        has_legal = legal.any(axis=1)

        left_window = active & (model.IDX[None, :] < g[:, None])
        expected_left_u = np.where(left_window, util * p, -1e9)
        best_left_i = np.argmax(expected_left_u, axis=1).astype(np.int8)
        best_left_u = expected_left_u[rows, best_left_i]
        can_reset = any_active & has_legal & (g != left)

        z = model.reset_decision_features(
            x, g, h, best_i, best_u, util, p, active, legal, left
        )
        predicted_advantage = np.asarray(predictor.predict(z))
        optional_reset = can_reset & (predicted_advantage > threshold)
        forced_reset = any_active & ~has_legal
        do_reset = optional_reset | forced_reset
        do_fill = any_active & has_legal & ~optional_reset

        fill_q = q[rows, best_i]
        fill_c = c[rows, best_i]
        right_activated = np.flip(
            np.cumsum(np.flip(activated, axis=1), axis=1), axis=1
        ) - activated
        fill_r = right_activated[rows, best_i]
        fill_reward = (fill_q == 2) * (1 + fill_r)
        legal_first = (legal & (c == 0)).any(axis=1)
        legal_second = (legal & (c == 1)).any(axis=1)
        reachable_active = reachable & active
        reachable_first = (reachable_active & (c == 0)).any(axis=1)
        reachable_second = (reachable_active & (c == 1)).any(axis=1)
        left_q = q[rows, best_left_i]
        left_c = c[rows, best_left_i]
        left_r = right_activated[rows, best_left_i]

        if np.any(can_reset):
            ii = np.flatnonzero(can_reset)
            optional_records.append({
                "step": np.full(len(ii), step, dtype=np.int8),
                "g": g[ii].copy(),
                "fill_i": best_i[ii].copy(),
                "fill_q": fill_q[ii].copy(),
                "fill_c": fill_c[ii].copy(),
                "fill_r": fill_r[ii].copy(),
                "fill_reward": fill_reward[ii].copy(),
                "left_i": best_left_i[ii].copy(),
                "left_q": left_q[ii].copy(),
                "left_c": left_c[ii].copy(),
                "left_r": left_r[ii].copy(),
                "activated": activated[ii].sum(axis=1).astype(np.int8),
                "left_count": left_window[ii].sum(axis=1).astype(np.int8),
                "legal_count": legal[ii].sum(axis=1).astype(np.int8),
                "legal_first": legal_first[ii].copy(),
                "legal_second": legal_second[ii].copy(),
                "reachable_first": reachable_first[ii].copy(),
                "reachable_second": reachable_second[ii].copy(),
                "utility_gap": (best_left_u[ii] - best_u[ii]).copy(),
                "predicted_advantage": predicted_advantage[ii].copy(),
                "reset": optional_reset[ii].copy(),
            })

        if np.any(do_fill):
            ii = np.flatnonzero(do_fill)
            fill_records.append({
                "step": np.full(len(ii), step, dtype=np.int8),
                "g": g[ii].copy(),
                "fill_i": best_i[ii].copy(),
                "fill_q": fill_q[ii].copy(),
                "fill_c": fill_c[ii].copy(),
                "fill_r": fill_r[ii].copy(),
                "fill_reward": fill_reward[ii].copy(),
                "activated": activated[ii].sum(axis=1).astype(np.int8),
                "legal_first": legal_first[ii].copy(),
                "legal_second": legal_second[ii].copy(),
                "move": (best_i[ii] - g[ii]).astype(np.int8),
            })

        optional_resets += optional_reset
        forced_resets += forced_reset
        g[do_reset] = left[do_reset]
        if np.any(do_fill):
            rr = rows[do_fill]
            ii = best_i[do_fill]
            old = x[rr, ii]
            completes = old % 3 == 2
            if np.any(completes):
                cr = rr[completes]
                ci = ii[completes]
                oldc = old[completes] // 3
                reward = 1 + np.array([
                    np.sum(x[row, col + 1:] >= 3)
                    for row, col in zip(cr, ci)
                ], dtype=np.float32)
                score[cr] += reward
                completions[cr] += 1
                for window in range(model.N):
                    first_take = (ci == window) & (oldc == 0)
                    second_take = (ci == window) & (oldc == 1)
                    first_completion_count[window] += int(first_take.sum())
                    second_completion_count[window] += int(second_take.sum())
                    first_completion_step[window] += step * int(first_take.sum())
                    second_completion_step[window] += step * int(second_take.sum())
            x[rr, ii] += 1
            g[rr] = ii

    opt = {
        key: _concat(optional_records, key)
        for key in optional_records[0]
    }
    fill = {
        key: _concat(fill_records, key)
        for key in fill_records[0]
    }
    reset = opt["reset"].astype(bool)
    second_fill = fill["fill_c"] == 1
    both_legal = fill["legal_first"] & fill["legal_second"]
    only_reachable_second = (
        ~opt["reachable_first"] & opt["reachable_second"]
    )
    only_legal_second = ~opt["legal_first"] & opt["legal_second"]

    completion_rows = []
    for i in range(model.N):
        completion_rows.append({
            "window": i + 1,
            "first_completion_probability": float(
                first_completion_count[i] / episodes
            ),
            "second_completion_probability": float(
                second_completion_count[i] / episodes
            ),
            "mean_first_completion_step": float(
                first_completion_step[i] / max(1, first_completion_count[i])
            ),
            "mean_second_completion_step": float(
                second_completion_step[i] / max(1, second_completion_count[i])
            ),
        })

    def rate(take):
        return {
            "n": int(take.sum()),
            "reset_rate": float(reset[take].mean()) if np.any(take) else None,
        }

    print(json.dumps({
        "stage": "policy_trace",
        "episodes": episodes,
        "score": _summary(score),
        "completions": _summary(completions),
        "optional_resets_per_episode": _summary(optional_resets),
        "forced_resets_per_episode": _summary(forced_resets),
        "optional_decision_count": int(len(reset)),
        "optional_reset_rate": float(reset.mean()),
        "reset_by_phase": _group_rate(_phase(opt["step"]), reset),
        "reset_by_worker_window": _group_rate(opt["g"] + 1, reset),
        "reset_by_fill_side": _group_rate(opt["fill_c"] + 1, reset),
        "reset_by_fill_progress": _group_rate(opt["fill_q"], reset),
        "reset_by_fill_reward": _group_rate(opt["fill_reward"], reset),
        "reset_when_best_fill_completes": rate(opt["fill_q"] == 2),
        "reset_when_best_fill_not_complete": rate(opt["fill_q"] != 2),
        "reset_when_only_legal_second": rate(only_legal_second),
        "reset_when_only_reachable_second": rate(only_reachable_second),
        "reset_when_legal_first_exists": rate(opt["legal_first"]),
        "reset_when_both_sides_legal": rate(
            opt["legal_first"] & opt["legal_second"]
        ),
        "reset_by_activated_count": _group_rate(opt["activated"], reset),
        "reset_by_left_count": _group_rate(opt["left_count"], reset),
        "fill_count": int(len(fill["step"])),
        "fill_side_by_phase": [
            {
                "phase": phase,
                "n": int((_phase(fill["step"]) == phase).sum()),
                "second_side_rate": float(second_fill[
                    _phase(fill["step"]) == phase
                ].mean()),
                "stay_rate": float((fill["move"][
                    _phase(fill["step"]) == phase
                ] == 0).mean()),
            }
            for phase in range(3)
        ],
        "both_sides_legal_fill_choices": {
            "n": int(both_legal.sum()),
            "choose_first_rate": float(
                (fill["fill_c"][both_legal] == 0).mean()
            ),
            "choose_second_rate": float(
                (fill["fill_c"][both_legal] == 1).mean()
            ),
            "by_phase": [
                {
                    "phase": phase,
                    "n": int((both_legal & (_phase(fill["step"]) == phase)).sum()),
                    "choose_first_rate": float((fill["fill_c"][
                        both_legal & (_phase(fill["step"]) == phase)
                    ] == 0).mean()) if np.any(
                        both_legal & (_phase(fill["step"]) == phase)
                    ) else None,
                }
                for phase in range(3)
            ],
        },
        "second_fill_by_window": _group_rate(
            fill["fill_i"][second_fill] + 1,
            fill["fill_q"][second_fill] == 2,
        ),
        "second_fill_completion_rate": float(
            (fill["fill_q"][second_fill] == 2).mean()
        ),
        "second_fill_mean_right_activated": float(
            fill["fill_r"][second_fill].mean()
        ),
        "completion_rows": completion_rows,
        "activated_curve": [
            {"step": step, "mean_activated": activated_curve[step]}
            for step in (0, 5, 10, 15, 20, 25, 30, 35, 40, 44)
        ],
    }), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-episodes", type=int, default=10000)
    parser.add_argument("--states", type=int, default=1600)
    parser.add_argument("--rollouts", type=int, default=128)
    parser.add_argument("--opening-episodes", type=int, default=20000)
    parser.add_argument("--trace-episodes", type=int, default=15000)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260841)
    args = parser.parse_args()

    predictor = train_current_predictor(
        args.seed, args.collection_episodes, args.states, args.rollouts
    )
    rng = np.random.default_rng(args.seed + 3000)
    forced_opening_audit(
        predictor, rng, args.opening_episodes, args.threshold
    )
    policy_trace_audit(
        predictor, rng, args.trace_episodes, args.threshold
    )


if __name__ == "__main__":
    main()
