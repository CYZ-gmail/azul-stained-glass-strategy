import argparse
import json
from dataclasses import dataclass

import numpy as np


N = 8
H = 45
BASE_FEATURE_COUNT = 17
BASE_FEATURE_NAMES = (
    "rightness", "progress", "first_side", "complete_now",
    "immediate_reward", "future_left_bonus", "stay", "flexibility",
    "early_completion", "late_progress", "second_side", "fill_bias",
    "right_edge", "rightness_squared", "R", "D", "L",
)
# Pairwise products among the six first-order features whose absolute weights
# are largest in the validated R+D+L policy.  Keeping the parent main effects
# makes this a hierarchical second-order model.
INTERACTION_PAIRS = (
    (1, 3), (1, 10), (1, 14), (1, 15), (1, 16),
    (3, 10), (3, 14), (3, 15), (3, 16),
    (10, 14), (10, 15), (10, 16),
    (14, 15), (14, 16), (15, 16),
)
INTERACTION_NAMES = (
    "progress*complete_now", "progress*second_side", "progress*R",
    "progress*D", "progress*L", "complete_now*second_side",
    "complete_now*R", "complete_now*D", "complete_now*L",
    "second_side*R", "second_side*D", "second_side*L",
    "R*D", "R*L", "D*L",
)
SQUARE_FEATURES = (1, 14, 15, 16)
SQUARE_NAMES = ("progress^2", "R^2", "D^2", "L^2")
INTERACTION_FEATURE_COUNT = BASE_FEATURE_COUNT + len(INTERACTION_PAIRS)
N_FEATURES = INTERACTION_FEATURE_COUNT + len(SQUARE_FEATURES)
ACTION_FEATURE_NAMES = BASE_FEATURE_NAMES + INTERACTION_NAMES + SQUARE_NAMES
RESET_LEGACY_FEATURE_NAMES = (
    "bias", "utility_gap", "left_count", "worker_position",
    "remaining_horizon", "fill_completes", "fill_reward",
    "left_density", "fill_density", "density_gap",
    "left_R", "fill_R", "left_progress", "fill_progress",
    "legal_count", "activated_fraction", "left_remaining_sides",
    "fill_second_side", "left_second_side", "distance_from_leftmost",
)
RESET_ACTION_BLOCKS = (
    "fill", "best_left_expected", "left_minus_fill",
    "left_pool_mean", "left_pool_max",
)
RESET_FEATURE_NAMES = RESET_LEGACY_FEATURE_NAMES + tuple(
    f"{block}:{name}"
    for block in RESET_ACTION_BLOCKS
    for name in ACTION_FEATURE_NAMES
)
RESET_LEGACY_FEATURE_COUNT = len(RESET_LEGACY_FEATURE_NAMES)
P = np.array([0.8, 0.6, 0.4], dtype=np.float32)
IDX = np.arange(N, dtype=np.int8)


def features(x: np.ndarray, g: np.ndarray, h: int, horizon: int) -> np.ndarray:
    """Return per-window action features, shape (episodes, windows, features)."""
    e = x.shape[0]
    q = x % 3
    c = x // 3
    active = x < 6
    activated = x >= 3
    right_activated = np.flip(
        np.cumsum(np.flip(activated, axis=1), axis=1), axis=1
    ) - activated
    # Number of future side completions on windows to the left. Completing a
    # first side now activates a bonus for each such future completion.
    remaining_completions = np.maximum(0, 2 - c)
    left_remaining = np.cumsum(remaining_completions, axis=1) - remaining_completions
    suffix_active = np.flip(np.cumsum(np.flip(active, axis=1), axis=1), axis=1)
    hfrac = h / horizon

    f = np.empty((e, N, N_FEATURES), dtype=np.float32)
    f[:, :, 0] = IDX / 7.0                              # rightness
    f[:, :, 1] = q / 2.0                                # current-side progress
    f[:, :, 2] = c == 0                                 # first side
    f[:, :, 3] = q == 2                                 # completes now
    f[:, :, 4] = (q == 2) * (1 + right_activated) / 8.0 # immediate reward
    f[:, :, 5] = ((q == 2) & (c == 0)) * left_remaining / 14.0
    f[:, :, 6] = IDX[None, :] == g[:, None]             # stay at current position
    f[:, :, 7] = suffix_active / 8.0                    # future reachable flexibility
    f[:, :, 8] = (q == 2) * hfrac                       # early completion interaction
    f[:, :, 9] = (q / 2.0) * (1.0 - hfrac)             # late progress urgency
    f[:, :, 10] = (c == 1)                              # second side
    f[:, :, 11] = active                                # bias for fill actions
    f[:, :, 12] = IDX[None, :] == (N - 1)               # extreme-right edge
    f[:, :, 13] = (IDX / 7.0) ** 2                       # nonlinear position effect
    # Standalone count of activated windows to the right.  Unlike feature 4,
    # this remains visible even when the candidate window is not one fill away
    # from completion, and it has no remaining-fill denominator.
    f[:, :, 14] = right_activated / 7.0
    # Completion value per remaining successful fill.  This is kept separate
    # from standalone R so joint search can learn whether the denominator helps.
    f[:, :, 15] = (1.0 + right_activated) / (3.0 - q) / 8.0
    # Phase-dependent left-harvest pressure: it grows as more first sides are
    # activated and is strongest for smaller window indices.
    activated_fraction = activated.sum(axis=1, keepdims=True) / 8.0
    f[:, :, 16] = activated_fraction * (1.0 - IDX[None, :] / 7.0)
    for k, (a, b) in enumerate(INTERACTION_PAIRS, start=BASE_FEATURE_COUNT):
        f[:, :, k] = f[:, :, a] * f[:, :, b]
    for k, a in enumerate(SQUARE_FEATURES, start=INTERACTION_FEATURE_COUNT):
        f[:, :, k] = f[:, :, a] ** 2
    return f


def reset_decision_features(x, g, h, best_i, best_u, util, p, active, legal, left):
    """Features for a learned optional-reset advantage model.

    Each row describes the choice between resetting now and taking the fill
    selected by the fixed fill policy.  All structural quantities are measured
    before the action, matching the observable decision state.
    """
    rows = np.arange(x.shape[0])
    q = x % 3
    c = x // 3
    activated = x >= 3
    right_activated = np.flip(
        np.cumsum(np.flip(activated, axis=1), axis=1), axis=1
    ) - activated
    left_window = active & (IDX[None, :] < g[:, None])

    density = (1.0 + right_activated) / (3.0 - q) / 8.0
    expected_left_u = np.where(left_window, util * p, -1e9)
    best_left_i = np.argmax(expected_left_u, axis=1)
    has_left = left_window.any(axis=1)
    best_left_u = expected_left_u[rows, best_left_i]
    best_left_u = np.where(has_left, best_left_u, best_u)

    fill_q = q[rows, best_i]
    fill_c = c[rows, best_i]
    fill_r = right_activated[rows, best_i]
    left_q = q[rows, best_left_i]
    left_c = c[rows, best_left_i]
    left_r = right_activated[rows, best_left_i]
    left_density = density[rows, best_left_i]
    left_density = np.where(has_left, left_density, 0.0)
    fill_density = density[rows, best_i]
    fill_completes = fill_q == 2
    fill_reward = fill_completes * (1.0 + fill_r) / 8.0
    left_remaining_sides = np.where(
        left_window, np.maximum(0, 2 - c), 0
    ).sum(axis=1) / 14.0

    legacy = np.empty(
        (x.shape[0], RESET_LEGACY_FEATURE_COUNT), dtype=np.float32
    )
    legacy[:, 0] = 1.0
    legacy[:, 1] = (best_left_u - best_u) / 10.0
    legacy[:, 2] = left_window.sum(axis=1) / 8.0
    legacy[:, 3] = g / 7.0
    legacy[:, 4] = h / H
    legacy[:, 5] = fill_completes
    legacy[:, 6] = fill_reward
    legacy[:, 7] = left_density
    legacy[:, 8] = fill_density
    legacy[:, 9] = left_density - fill_density
    legacy[:, 10] = np.where(has_left, left_r / 7.0, 0.0)
    legacy[:, 11] = fill_r / 7.0
    legacy[:, 12] = np.where(has_left, left_q / 2.0, 0.0)
    legacy[:, 13] = fill_q / 2.0
    legacy[:, 14] = legal.sum(axis=1) / 8.0
    legacy[:, 15] = activated.sum(axis=1) / 8.0
    legacy[:, 16] = left_remaining_sides
    legacy[:, 17] = fill_c == 1
    legacy[:, 18] = np.where(has_left, left_c == 1, False)
    legacy[:, 19] = (g - left) / 7.0

    # Match the fill policy's complete 36-dimensional feature space.  A reset
    # is valuable because it exchanges the currently selected legal fill for
    # a newly sampled set of actions to the left, so expose both the best
    # expected left candidate and summaries of the full unlocked candidate
    # pool.  Left-side action features are probability-weighted because their
    # masks will be sampled only after the reset consumes the current turn.
    action = features(x, g, h, H)
    fill_action = action[rows, best_i]
    weighted_left = action * p[:, :, None]
    best_left_action = weighted_left[rows, best_left_i]
    best_left_action = np.where(
        has_left[:, None], best_left_action, np.zeros_like(best_left_action)
    )
    left_count_raw = left_window.sum(axis=1)
    left_sum = np.where(
        left_window[:, :, None], weighted_left, 0.0
    ).sum(axis=1)
    left_mean = left_sum / np.maximum(1, left_count_raw)[:, None]
    left_max = np.where(
        left_window[:, :, None], weighted_left, -np.inf
    ).max(axis=1)
    left_max = np.where(has_left[:, None], left_max, 0.0)

    return np.concatenate((
        legacy,
        fill_action,
        best_left_action,
        best_left_action - fill_action,
        left_mean,
        left_max,
    ), axis=1).astype(np.float32, copy=False)


def simulate(weights, reset_params, uniforms, record_opening=False,
             initial_x=None, initial_g=None, start_step=0,
             harvest_params=None, reset_predictor=None,
             reset_threshold=0.0):
    episodes = uniforms.shape[0]
    horizon = uniforms.shape[1]
    if initial_x is None:
        x = np.zeros((episodes, N), dtype=np.int8)
    else:
        x = np.broadcast_to(np.asarray(initial_x, dtype=np.int8), (episodes, N)).copy()
    if initial_g is None:
        g = np.zeros(episodes, dtype=np.int8)
    else:
        g = np.full(episodes, int(initial_g), dtype=np.int8)
    score = np.zeros(episodes, dtype=np.float32)
    resets = np.zeros(episodes, dtype=np.int16)
    optional_resets = np.zeros(episodes, dtype=np.int16)
    forced_resets = np.zeros(episodes, dtype=np.int16)
    completions = np.zeros(episodes, dtype=np.int8)
    opening = np.full(episodes, -2, dtype=np.int8)

    rows = np.arange(episodes)
    for step in range(start_step, horizon):
        h = horizon - step
        active = x < 6
        any_active = active.any(axis=1)
        left = np.where(any_active, np.argmax(active, axis=1), 0).astype(np.int8)
        p = P[x % 3]
        mask = (uniforms[:, step, :] < p) & active
        reachable = IDX[None, :] >= g[:, None]
        legal = mask & reachable

        f = features(x, g, h, horizon)
        util = f @ weights
        if harvest_params is not None:
            q_now = x % 3
            c_now = x // 3
            activated_now = x >= 3
            right_activated_now = np.flip(
                np.cumsum(np.flip(activated_now, axis=1), axis=1), axis=1
            ) - activated_now
            activated_fraction = activated_now.sum(axis=1, keepdims=True) / 8.0
            # Phase-aware harvest terms. Once a window is on its second side,
            # prefer high score per remaining fill and, after many first sides
            # are activated, prefer smaller indices that can rescore more of
            # those right-side windows.
            harvest_density = (
                (c_now == 1)
                * (1.0 + right_activated_now)
                / (3.0 - q_now)
                / 8.0
            )
            left_harvest = (
                (c_now == 1)
                * activated_fraction
                * (1.0 - IDX[None, :] / 7.0)
            )
            any_completion_density = (
                (x < 6)
                * (1.0 + right_activated_now)
                / (3.0 - q_now)
                / 8.0
            )
            phase_left_any = (
                (x < 6)
                * activated_fraction
                * (1.0 - IDX[None, :] / 7.0)
            )
            util = (
                util
                + harvest_params[0] * harvest_density
                + harvest_params[1] * left_harvest
            )
            if len(harvest_params) >= 4:
                util = (
                    util
                    + harvest_params[2] * any_completion_density
                    + harvest_params[3] * phase_left_any
                )
        legal_util = np.where(legal, util, -1e9)
        best_i = np.argmax(legal_util, axis=1).astype(np.int8)
        best_u = legal_util[rows, best_i]
        has_legal = legal.any(axis=1)

        # Reset comparison: estimate the best latent priority among unfinished
        # windows to the left. Current mask is deliberately ignored because it
        # will be resampled after reset.
        left_window = active & (IDX[None, :] < g[:, None])
        expected_left_u = np.where(left_window, util * p, -1e9)
        best_left_u = expected_left_u.max(axis=1)
        left_count = left_window.sum(axis=1) / 8.0
        gfrac = g / 7.0
        hfrac = h / H
        # Positive reset advantage means reset instead of taking a legal fill.
        if reset_predictor is None:
            reset_adv = (
                best_left_u - best_u
                + reset_params[0]
                + reset_params[1] * left_count
                + reset_params[2] * gfrac
                + reset_params[3] * hfrac
            )
        else:
            z_reset = reset_decision_features(
                x, g, h, best_i, best_u, util, p, active, legal, left
            )
            if hasattr(reset_predictor, "predict"):
                reset_adv = np.asarray(reset_predictor.predict(z_reset))
            else:
                reset_adv = np.asarray(reset_predictor(z_reset))
            reset_adv = reset_adv - reset_threshold
        optional_reset = (g != left) & has_legal & (reset_adv > 0)
        forced_reset = ~has_legal
        do_reset = any_active & (optional_reset | forced_reset)
        do_fill = any_active & has_legal & ~optional_reset

        if record_opening and step == 0:
            opening[do_reset] = -1
            opening[do_fill] = best_i[do_fill]

        resets += do_reset
        optional_resets += optional_reset
        forced_resets += any_active & forced_reset
        g[do_reset] = left[do_reset]

        if np.any(do_fill):
            r = rows[do_fill]
            i = best_i[do_fill]
            old = x[r, i]
            completes = (old % 3) == 2
            if np.any(completes):
                rr = r[completes]
                ii = i[completes]
                right_bonus = np.array([
                    np.sum(x[row, col + 1:] >= 3) for row, col in zip(rr, ii)
                ], dtype=np.float32)
                score[rr] += 1.0 + right_bonus
                completions[rr] += 1
            x[r, i] += 1
            g[r] = i

    return {
        "score": score,
        "resets": resets,
        "optional_resets": optional_resets,
        "forced_resets": forced_resets,
        "completions": completions,
        "opening": opening,
        "x": x,
    }


def pad_interactions(weights):
    """Pad a 17-feature first-order policy with zero interaction weights."""
    weights = np.asarray(weights, dtype=np.float32)
    if len(weights) == N_FEATURES:
        return weights.copy()
    if len(weights) not in (BASE_FEATURE_COUNT, INTERACTION_FEATURE_COUNT):
        raise ValueError(
            f"expected {BASE_FEATURE_COUNT}, {INTERACTION_FEATURE_COUNT}, "
            f"or {N_FEATURES} weights"
        )
    return np.pad(weights, (0, N_FEATURES - len(weights)))


BASELINES = {
    "rightmost": (pad_interactions([5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                  np.array([-1e6, 0, 0, 0], np.float32)),
    "leftmost": (pad_interactions([-5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                 np.array([-1e6, 0, 0, 0], np.float32)),
    "finish_then_right": (pad_interactions([1, 1, 0, 4, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                          np.array([-1e6, 0, 0, 0], np.float32)),
    "activate_first": (pad_interactions([1, 1, 2, 5, 2, 3, 1, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0]),
                       np.array([-1e6, 0, 0, 0], np.float32)),
}

# Best jointly searched standalone-R policy reported in the document.  It was
# trained with seed 20260810 on 6,000 common-random-number trajectories and
# validated on 100,000 fresh trajectories from seed 20260811.
VALIDATED_RI_WEIGHTS = pad_interactions([
    3.1810777, 3.8479385, 1.8667953, 5.1408663, 1.2415762,
    2.7813430, 0.8493108, 0.1324626, 0.8779671, 0.1042113,
    -6.4945612, -0.1063348, -0.6765231, -2.3901970, 14.3267345,
    0.0, 0.0,
])
VALIDATED_RI_RESET = np.array([
    -2.6777754, 1.4295107, -2.8904908, -3.4088309,
], dtype=np.float32)

# Best R+D+L candidate selected across two independent restarts, then reported
# on a final untouched 100,000-trajectory test set from seed 20260822.
VALIDATED_RDL_WEIGHTS = pad_interactions([
    2.5724049, 4.4874625, 1.5506563, 6.3861532, 0.2580973,
    3.8827066, 0.2012338, -0.4794605, 1.0762638, 0.9194657,
    -5.7932153, 0.1989260, -0.1776897, -2.0331495,
    15.3965645, -7.0427556, 4.2707686,
])
VALIDATED_RDL_RESET = np.array([
    -3.1536286, 2.2941678, -3.3258555, -2.8378732,
], dtype=np.float32)

# Hierarchical second-order policy selected across two independent restarts.
# Nonzero quadratic terms are progress*second_side (18),
# complete_now*second_side (22), second_side*D (27), R*D (29), and D^2 (34).
# The reported result uses a final untouched 100,000-trajectory test set from
# seed 20260835: 45.25694 points versus 45.06444 for R+D+L on the same paths.
VALIDATED_SECOND_ORDER_WEIGHTS = np.array([
    2.2524951, 4.7895789, 1.7761585, 6.1192770, 0.3773038,
    3.5934956, 0.0320498, -0.4032657, 1.0348688, 0.9259579,
    -5.8883667, 0.1113659, -0.1068745, -1.8297492,
    15.6156254, -7.1940341, 4.5616121,
    0.0, 0.8603513, 0.0, 0.0, 0.0, 0.5629519, 0.0, 0.0,
    0.0, 0.0, -0.2047111, 0.0, 5.1079059, 0.0, 0.0,
    0.0, 0.0, -0.1878748, 0.0,
], dtype=np.float32)
VALIDATED_SECOND_ORDER_RESET = np.array([
    -3.1469793, 2.3832693, -3.3332758, -2.7523444,
], dtype=np.float32)


def summarize(name, result):
    score = result["score"]
    comp = result["completions"]
    reset = result["resets"]
    return {
        "name": name,
        "mean_score": float(score.mean()),
        "se_score": float(score.std(ddof=1) / np.sqrt(len(score))),
        "mean_completions": float(comp.mean()),
        "mean_resets": float(reset.mean()),
        "score_p10": float(np.quantile(score, 0.1)),
        "score_p50": float(np.quantile(score, 0.5)),
        "score_p90": float(np.quantile(score, 0.9)),
    }


def random_search(rng, uniforms_train, rounds=400, local_rounds=120):
    seeds = [
        np.array([1.0, 1.0, 1.0, 4.0, 2.0, 2.0, 1.0, 0.5, 0.5, 1.0, -1.0, 0.0, -1.0, 0.0, 4.0, 4.0, 4.0]),
        np.array([0.0, 2.0, 2.0, 5.0, 3.0, 3.0, 1.0, 1.0, 0.5, 2.0, -1.0, 0.0, -1.0, 0.0, 8.0, 8.0, 8.0]),
        np.array([2.0, 1.0, 1.0, 5.0, 3.0, 2.0, 1.0, 0.5, 0.5, 1.0, -1.0, 0.0, -1.0, 0.0, 12.0, 12.0, 12.0]),
        np.array([1.45605,-5.26394,7.96408,5.49344,-2.20231,-0.01717,-2.34384,-2.67869,2.14880,5.22247,-7.97209,-6.15199,-2.0,0.0,8.0,8.0,8.0]),
    ]
    best = None
    candidates = []
    for s in seeds:
        candidates.append((pad_interactions(s), np.array([-1.0, 0.0, 0.0, 0.0], np.float32)))
    candidates.append((VALIDATED_RI_WEIGHTS.copy(), VALIDATED_RI_RESET.copy()))
    # Structured warm-start grid: retain the validated R-only policy while
    # independently exploring D and L before all coefficients move jointly.
    for d_weight in [-8.0, -4.0, 0.0, 4.0, 8.0, 12.0, 20.0]:
        for l_weight in [-8.0, -4.0, 0.0, 4.0, 8.0, 12.0, 20.0]:
            w = VALIDATED_RI_WEIGHTS.copy()
            w[15] = d_weight
            w[16] = l_weight
            candidates.append((w, VALIDATED_RI_RESET.copy()))
    for _ in range(rounds):
        w = rng.normal(0, 2.5, size=N_FEATURES).astype(np.float32)
        # Completing an available side should usually matter.
        w[3] += 3.0
        w[4] += 2.0
        rp = rng.normal(0, 1.5, size=4).astype(np.float32)
        candidates.append((w, rp))

    for idx, (w, rp) in enumerate(candidates):
        result = simulate(w, rp, uniforms_train)
        val = float(result["score"].mean())
        if best is None or val > best[0]:
            best = (val, w.copy(), rp.copy())
            print(json.dumps({"candidate": idx, "train_score": val,
                              "weights": w.tolist(), "reset": rp.tolist()}), flush=True)

    # Local Gaussian refinements with decreasing scale.
    val, w, rp = best
    for scale in [2.0, 1.0, 0.5, 0.25, 0.1]:
        for _ in range(local_rounds):
            wc = w + rng.normal(0, scale, size=N_FEATURES).astype(np.float32)
            rpc = rp + rng.normal(0, scale, size=4).astype(np.float32)
            result = simulate(wc, rpc, uniforms_train)
            vc = float(result["score"].mean())
            if vc > val:
                val, w, rp = vc, wc, rpc
                print(json.dumps({"scale": scale, "train_score": val,
                                  "weights": w.tolist(), "reset": rp.tolist()}), flush=True)
    return w, rp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-episodes", type=int, default=12000)
    ap.add_argument("--test-episodes", type=int, default=200000)
    ap.add_argument("--rounds", type=int, default=350)
    ap.add_argument("--local-rounds", type=int, default=120)
    ap.add_argument("--horizon", type=int, default=H)
    ap.add_argument("--seed", type=int, default=20260810)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    uniforms_train = rng.random((args.train_episodes, args.horizon, N), dtype=np.float32)
    uniforms_test = rng.random((args.test_episodes, args.horizon, N), dtype=np.float32)

    for name, (w, rp) in BASELINES.items():
        print(json.dumps(summarize(name, simulate(w, rp, uniforms_test))), flush=True)

    w, rp = random_search(rng, uniforms_train, args.rounds, args.local_rounds)
    result = simulate(w, rp, uniforms_test, record_opening=True)
    print(json.dumps(summarize("optimized_linear_policy", result)), flush=True)
    counts = np.bincount(result["opening"] + 2, minlength=N + 2)
    print(json.dumps({"opening_counts_minus2_to_7": counts.tolist(),
                      "weights": w.tolist(), "reset": rp.tolist()}), flush=True)
    x_final = result["x"]
    print(json.dumps({
        "mean_sides_by_window": (x_final // 3).mean(axis=0).tolist(),
        "activation_probability_by_window": (x_final >= 3).mean(axis=0).tolist(),
        "double_probability_by_window": (x_final >= 6).mean(axis=0).tolist(),
        "mean_optional_resets": float(result["optional_resets"].mean()),
        "mean_forced_resets": float(result["forced_resets"].mean()),
    }), flush=True)


if __name__ == "__main__":
    main()
