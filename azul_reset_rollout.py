"""Learn an optional-reset rule from paired multi-step rollouts."""

import argparse
import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

import azul_policy_search as model


def collect_optional_states(rng, episodes, uniforms):
    """Collect optional-reset decisions visited by the validated policy."""
    x = np.zeros((episodes, model.N), dtype=np.int8)
    g = np.zeros(episodes, dtype=np.int8)
    rows = np.arange(episodes)
    records = []

    for step in range(model.H):
        h = model.H - step
        active = x < 6
        any_active = active.any(axis=1)
        left = np.where(any_active, np.argmax(active, axis=1), 0).astype(np.int8)
        p = model.P[x % 3]
        mask = (uniforms[:, step, :] < p) & active
        legal = mask & (model.IDX[None, :] >= g[:, None])
        f = model.features(x, g, h, model.H)
        util = f @ model.VALIDATED_SECOND_ORDER_WEIGHTS
        legal_util = np.where(legal, util, -1e9)
        best_i = np.argmax(legal_util, axis=1).astype(np.int8)
        best_u = legal_util[rows, best_i]
        has_legal = legal.any(axis=1)

        left_window = active & (model.IDX[None, :] < g[:, None])
        expected_left_u = np.where(left_window, util * p, -1e9)
        best_left_u = expected_left_u.max(axis=1)
        left_count = left_window.sum(axis=1) / 8.0
        reset_adv = (
            best_left_u - best_u
            + model.VALIDATED_SECOND_ORDER_RESET[0]
            + model.VALIDATED_SECOND_ORDER_RESET[1] * left_count
            + model.VALIDATED_SECOND_ORDER_RESET[2] * (g / 7.0)
            + model.VALIDATED_SECOND_ORDER_RESET[3] * (h / model.H)
        )
        can_reset = any_active & has_legal & (g != left)
        if np.any(can_reset):
            idx = np.flatnonzero(can_reset)
            z = model.reset_decision_features(
                x, g, h, best_i, best_u, util, p, active, legal, left
            )
            records.append({
                "x": x[idx].copy(),
                "g": g[idx].copy(),
                "step": np.full(len(idx), step, dtype=np.int8),
                "best_i": best_i[idx].copy(),
                "features": z[idx].copy(),
                "old_reset": (reset_adv[idx] > 0),
            })

        optional_reset = can_reset & (reset_adv > 0)
        forced_reset = ~has_legal
        do_reset = any_active & (optional_reset | forced_reset)
        do_fill = any_active & has_legal & ~optional_reset
        g[do_reset] = left[do_reset]
        if np.any(do_fill):
            rr = rows[do_fill]
            ii = best_i[do_fill]
            x[rr, ii] += 1
            g[rr] = ii

    merged = {
        key: np.concatenate([r[key] for r in records], axis=0)
        for key in records[0]
    }
    return merged


def sample_records(records, rng, n):
    total = len(records["g"])
    idx = rng.choice(total, size=min(n, total), replace=False)
    return {key: value[idx] for key, value in records.items()}


def rollout_targets(records, rng, rollouts):
    n = len(records["g"])
    target = np.empty(n, dtype=np.float32)
    target_se = np.empty(n, dtype=np.float32)
    immediate = np.empty(n, dtype=np.float32)

    for k in range(n):
        x = records["x"][k]
        g = int(records["g"][k])
        step = int(records["step"][k])
        i = int(records["best_i"][k])
        left = int(np.argmax(x < 6))
        future_uniforms = rng.random(
            (rollouts, model.H, model.N), dtype=np.float32
        )

        reset_result = model.simulate(
            model.VALIDATED_SECOND_ORDER_WEIGHTS,
            model.VALIDATED_SECOND_ORDER_RESET,
            future_uniforms,
            initial_x=x,
            initial_g=left,
            start_step=step + 1,
        )

        fill_x = x.copy()
        old = int(fill_x[i])
        reward = 0.0
        if old % 3 == 2:
            reward = 1.0 + float(np.sum(fill_x[i + 1:] >= 3))
        fill_x[i] += 1
        fill_result = model.simulate(
            model.VALIDATED_SECOND_ORDER_WEIGHTS,
            model.VALIDATED_SECOND_ORDER_RESET,
            future_uniforms,
            initial_x=fill_x,
            initial_g=i,
            start_step=step + 1,
        )
        diff = (
            reset_result["score"].astype(np.float64)
            - fill_result["score"].astype(np.float64)
            - reward
        )
        target[k] = diff.mean()
        target_se[k] = diff.std(ddof=1) / np.sqrt(rollouts)
        immediate[k] = reward
        if (k + 1) % 100 == 0 or k + 1 == n:
            print(json.dumps({
                "stage": "rollout_labels", "done": k + 1, "total": n,
                "positive_fraction": float(np.mean(target[:k + 1] > 0)),
                "mean_advantage": float(np.mean(target[:k + 1])),
            }), flush=True)
    return target, target_se, immediate


def _validation_metrics(name, fitted, x, y, valid):
    pred = fitted.predict(x[valid])
    metrics = {
        "stage": "model_validation", "model": name,
        "rmse": float(np.sqrt(mean_squared_error(y[valid], pred))),
        "sign_accuracy": float(np.mean((pred > 0) == (y[valid] > 0))),
        "predicted_reset_fraction": float(np.mean(pred > 0)),
        "true_reset_fraction": float(np.mean(y[valid] > 0)),
    }
    print(json.dumps(metrics), flush=True)
    return metrics


def fit_models(x, y, se, rng):
    order = rng.permutation(len(y))
    split = int(0.8 * len(y))
    train, valid = order[:split], order[split:]
    weight = 1.0 / (0.02 + se ** 2)

    ridge = Ridge(alpha=10.0)
    ridge.fit(x[train], y[train], sample_weight=weight[train])
    _validation_metrics("expanded_ridge", ridge, x, y, valid)

    # Exact ablation: the first 20 columns are the previous reset feature
    # space.  It is trained and validated on the same labels and split.
    legacy_x = x[:, :model.RESET_LEGACY_FEATURE_COUNT]
    legacy_hgb = HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=240, max_leaf_nodes=15,
        min_samples_leaf=20, l2_regularization=2.0,
        random_state=20260841,
    )
    legacy_hgb.fit(
        legacy_x[train], y[train], sample_weight=weight[train]
    )
    _validation_metrics("legacy20_hgb", legacy_hgb, legacy_x, y, valid)

    # Small structural search.  The final threshold is selected separately by
    # end-to-end episode score, so validation here chooses only the regressor.
    specifications = (
        (7, 20, 2.0), (15, 20, 2.0), (31, 20, 2.0),
        (15, 40, 2.0), (31, 40, 5.0), (31, 60, 8.0),
    )
    candidates = []
    for leaves, min_leaf, l2 in specifications:
        name = f"expanded_hgb_l{leaves}_m{min_leaf}_r{l2:g}"
        fitted = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=240,
            max_leaf_nodes=leaves,
            min_samples_leaf=min_leaf,
            l2_regularization=l2,
            random_state=20260841,
        )
        fitted.fit(x[train], y[train], sample_weight=weight[train])
        metrics = _validation_metrics(name, fitted, x, y, valid)
        candidates.append((metrics["rmse"], fitted, name))
    _, hgb, hgb_name = min(candidates, key=lambda item: item[0])
    print(json.dumps({
        "stage": "selected_regressor", "model": hgb_name,
        "feature_count": int(x.shape[1]),
        "legacy_feature_count": model.RESET_LEGACY_FEATURE_COUNT,
    }), flush=True)
    top = np.argsort(np.abs(ridge.coef_))[-20:][::-1]
    print(json.dumps({
        "stage": "ridge_coefficients",
        "intercept": float(ridge.intercept_),
        "top20_by_absolute_weight": {
            model.RESET_FEATURE_NAMES[i]: float(ridge.coef_[i]) for i in top
        },
    }), flush=True)
    return ridge, legacy_hgb, hgb, hgb_name


def threshold_sweep(predictor, uniforms, name):
    for threshold in (-0.25, -0.10, 0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75):
        result = model.simulate(
            model.VALIDATED_SECOND_ORDER_WEIGHTS,
            model.VALIDATED_SECOND_ORDER_RESET,
            uniforms,
            reset_predictor=predictor,
            reset_threshold=threshold,
        )
        print(json.dumps({
            "stage": "threshold_sweep", "model": name,
            "threshold": threshold,
            "score": float(result["score"].mean()),
            "completions": float(result["completions"].mean()),
            "optional_resets": float(result["optional_resets"].mean()),
            "forced_resets": float(result["forced_resets"].mean()),
        }), flush=True)


def final_compare(predictor, uniforms, thresholds):
    baseline = model.simulate(
        model.VALIDATED_SECOND_ORDER_WEIGHTS,
        model.VALIDATED_SECOND_ORDER_RESET,
        uniforms,
    )
    base_score = baseline["score"].astype(np.float64)
    print(json.dumps({
        "stage": "final_baseline",
        **model.summarize("second_order_old_reset", baseline),
        "optional_resets": float(baseline["optional_resets"].mean()),
        "forced_resets": float(baseline["forced_resets"].mean()),
    }), flush=True)
    for threshold in thresholds:
        result = model.simulate(
            model.VALIDATED_SECOND_ORDER_WEIGHTS,
            model.VALIDATED_SECOND_ORDER_RESET,
            uniforms,
            reset_predictor=predictor,
            reset_threshold=threshold,
        )
        score_diff = result["score"].astype(np.float64) - base_score
        reset_diff = (
            result["optional_resets"].astype(np.float64)
            - baseline["optional_resets"].astype(np.float64)
        )
        completion_diff = (
            result["completions"].astype(np.float64)
            - baseline["completions"].astype(np.float64)
        )
        print(json.dumps({
            "stage": "final_candidate", "threshold": threshold,
            **model.summarize(f"learned_reset_{threshold}", result),
            "optional_resets": float(result["optional_resets"].mean()),
            "forced_resets": float(result["forced_resets"].mean()),
            "paired_score_gain": float(score_diff.mean()),
            "paired_score_gain_se": float(
                score_diff.std(ddof=1) / np.sqrt(len(score_diff))
            ),
            "paired_optional_reset_change": float(reset_diff.mean()),
            "paired_optional_reset_change_se": float(
                reset_diff.std(ddof=1) / np.sqrt(len(reset_diff))
            ),
            "paired_completion_change": float(completion_diff.mean()),
            "paired_completion_change_se": float(
                completion_diff.std(ddof=1) / np.sqrt(len(completion_diff))
            ),
        }), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection-episodes", type=int, default=8000)
    ap.add_argument("--states", type=int, default=2000)
    ap.add_argument("--rollouts", type=int, default=96)
    ap.add_argument("--selection-episodes", type=int, default=30000)
    ap.add_argument("--final-episodes", type=int, default=0)
    ap.add_argument("--final-thresholds", default="0.1,0.2,0.35")
    ap.add_argument("--seed", type=int, default=20260841)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    collection_uniforms = rng.random(
        (args.collection_episodes, model.H, model.N), dtype=np.float32
    )
    records = collect_optional_states(
        rng, args.collection_episodes, collection_uniforms
    )
    print(json.dumps({
        "stage": "collection", "optional_states": len(records["g"]),
        "old_reset_fraction": float(records["old_reset"].mean()),
    }), flush=True)
    records = sample_records(records, rng, args.states)
    y, se, _ = rollout_targets(records, rng, args.rollouts)
    ridge, legacy_hgb, hgb, hgb_name = fit_models(
        records["features"], y, se, rng
    )
    selection_uniforms = rng.random(
        (args.selection_episodes, model.H, model.N), dtype=np.float32
    )
    baseline = model.simulate(
        model.VALIDATED_SECOND_ORDER_WEIGHTS,
        model.VALIDATED_SECOND_ORDER_RESET,
        selection_uniforms,
    )
    print(json.dumps({
        "stage": "selection_baseline",
        "score": float(baseline["score"].mean()),
        "completions": float(baseline["completions"].mean()),
        "optional_resets": float(baseline["optional_resets"].mean()),
        "forced_resets": float(baseline["forced_resets"].mean()),
    }), flush=True)
    threshold_sweep(ridge, selection_uniforms, "expanded_ridge")
    threshold_sweep(
        lambda z: legacy_hgb.predict(
            z[:, :model.RESET_LEGACY_FEATURE_COUNT]
        ),
        selection_uniforms,
        "legacy20_hgb",
    )
    threshold_sweep(hgb, selection_uniforms, hgb_name)
    if args.final_episodes:
        final_uniforms = np.random.default_rng(args.seed + 1000).random(
            (args.final_episodes, model.H, model.N), dtype=np.float32
        )
        thresholds = [float(v) for v in args.final_thresholds.split(",")]
        final_compare(hgb, final_uniforms, thresholds)


if __name__ == "__main__":
    main()
