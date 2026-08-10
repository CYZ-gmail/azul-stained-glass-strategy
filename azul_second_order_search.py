"""Screen and jointly optimize second-order feature interactions.

The first-order R+D+L policy remains the hierarchical parent model.  Candidate
interactions are products of six influential normalized features.  All policy
comparisons within a stage use common random numbers.
"""

import argparse
import itertools
import json

import numpy as np

import azul_policy_search as model


# progress*complete_now is exactly equal to complete_now because progress=1
# whenever complete_now=1.  It is reported during screening, but is not a new
# degree of freedom and is therefore excluded from joint search.
EXACT_ALIASES = {0: 3}
DEFAULT_GRID = (-32.0, -20.0, -12.0, -6.0, 0.0, 6.0, 12.0, 20.0, 32.0)


def paired_summary(candidate, baseline):
    diff = candidate["score"].astype(np.float64) - baseline["score"]
    return {
        "mean_gain": float(diff.mean()),
        "se_gain": float(diff.std(ddof=1) / np.sqrt(len(diff))),
        "mean_score": float(candidate["score"].mean()),
        "mean_completions": float(candidate["completions"].mean()),
        "mean_resets": float(candidate["resets"].mean()),
    }


def screen_squares(uniforms, grid=DEFAULT_GRID):
    baseline = model.simulate(
        model.VALIDATED_RDL_WEIGHTS, model.VALIDATED_RDL_RESET, uniforms
    )
    print(json.dumps({
        "stage": "square_screen_baseline",
        **model.summarize("R+D+L", baseline),
    }), flush=True)
    square_rows = []
    for square_id, name in enumerate(model.SQUARE_NAMES):
        best = None
        for coefficient in grid:
            w = model.VALIDATED_RDL_WEIGHTS.copy()
            w[model.INTERACTION_FEATURE_COUNT + square_id] = coefficient
            result = model.simulate(w, model.VALIDATED_RDL_RESET, uniforms)
            stats = paired_summary(result, baseline)
            if best is None or stats["mean_gain"] > best["mean_gain"]:
                best = {
                    "square_id": square_id, "name": name,
                    "best_coefficient": coefficient, **stats,
                }
        square_rows.append(best)
        print(json.dumps({"stage": "square_screen", **best}), flush=True)
    print(json.dumps({
        "stage": "square_ranking",
        "ranking": sorted(square_rows, key=lambda r: r["mean_gain"], reverse=True),
    }), flush=True)
    return square_rows


def screen(uniforms, grid=DEFAULT_GRID):
    baseline = model.simulate(
        model.VALIDATED_RDL_WEIGHTS, model.VALIDATED_RDL_RESET, uniforms
    )
    print(json.dumps({
        "stage": "screen_baseline",
        **model.summarize("R+D+L", baseline),
    }), flush=True)
    rows = []
    for interaction_id, name in enumerate(model.INTERACTION_NAMES):
        if interaction_id in EXACT_ALIASES:
            row = {
                "interaction_id": interaction_id,
                "name": name,
                "status": "exact_alias",
                "alias_of_base_feature": EXACT_ALIASES[interaction_id],
                "best_coefficient": 0.0,
                "mean_gain": 0.0,
                "se_gain": 0.0,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            continue
        best = None
        for coefficient in grid:
            w = model.VALIDATED_RDL_WEIGHTS.copy()
            w[model.BASE_FEATURE_COUNT + interaction_id] = coefficient
            result = model.simulate(w, model.VALIDATED_RDL_RESET, uniforms)
            stats = paired_summary(result, baseline)
            if best is None or stats["mean_gain"] > best["mean_gain"]:
                best = {
                    "interaction_id": interaction_id,
                    "name": name,
                    "status": "screened",
                    "best_coefficient": coefficient,
                    **stats,
                }
        rows.append(best)
        print(json.dumps(best), flush=True)
    ranked = sorted(
        (r for r in rows if r["status"] == "screened"),
        key=lambda r: r["mean_gain"], reverse=True,
    )
    print(json.dumps({
        "stage": "screen_ranking",
        "ranking": [
            {
                "interaction_id": r["interaction_id"],
                "name": r["name"],
                "best_coefficient": r["best_coefficient"],
                "mean_gain": r["mean_gain"],
                "se_gain": r["se_gain"],
            }
            for r in ranked
        ],
    }), flush=True)
    square_rows = screen_squares(uniforms, grid)
    return rows, square_rows


def evaluate_vector(vector, uniforms):
    w = vector[:model.N_FEATURES].astype(np.float32)
    rp = vector[model.N_FEATURES:].astype(np.float32)
    result = model.simulate(w, rp, uniforms)
    return float(result["score"].mean())


def build_warm_starts(selected_feature_indices, screened_coefficients):
    base = np.concatenate([
        model.VALIDATED_RDL_WEIGHTS,
        model.VALIDATED_RDL_RESET,
    ]).astype(np.float32)
    starts = [base.copy()]
    # One-factor warm starts.
    for feature_index in selected_feature_indices:
        v = base.copy()
        v[feature_index] = screened_coefficients[feature_index]
        starts.append(v)
    # The combined screen winner is useful even though interactions can
    # partially substitute for one another.
    v = base.copy()
    for feature_index in selected_feature_indices:
        v[feature_index] = screened_coefficients[feature_index]
    starts.append(v)
    return starts


def joint_search(rng, uniforms, selected_pairs, pair_coefficients,
                 selected_squares, square_coefficients,
                 broad_candidates=220, local_candidates=160):
    """Evolutionary random search over all parents and selected interactions."""
    selected_feature_indices = np.array(
        [model.BASE_FEATURE_COUNT + i for i in selected_pairs]
        + [model.INTERACTION_FEATURE_COUNT + i for i in selected_squares],
        dtype=np.int64,
    )
    screened_coefficients = {
        **{model.BASE_FEATURE_COUNT + i: pair_coefficients[i] for i in selected_pairs},
        **{model.INTERACTION_FEATURE_COUNT + i: square_coefficients[i]
           for i in selected_squares},
    }
    active = np.concatenate([
        np.arange(model.BASE_FEATURE_COUNT, dtype=np.int64),
        selected_feature_indices,
        np.arange(model.N_FEATURES, model.N_FEATURES + 4, dtype=np.int64),
    ])
    starts = build_warm_starts(selected_feature_indices, screened_coefficients)
    best_val = -np.inf
    best = None
    evaluations = 0

    def consider(v, label):
        nonlocal best_val, best, evaluations
        value = evaluate_vector(v, uniforms)
        evaluations += 1
        if value > best_val:
            best_val = value
            best = v.copy()
            print(json.dumps({
                "stage": "joint_search", "label": label,
                "evaluation": evaluations, "train_score": best_val,
                "selected_pairs": selected_pairs,
                "selected_squares": selected_squares,
                "weights": best[:model.N_FEATURES].tolist(),
                "reset": best[model.N_FEATURES:].tolist(),
            }), flush=True)

    for j, v in enumerate(starts):
        consider(v, f"warm_{j}")

    base = starts[0]
    for j in range(broad_candidates):
        v = base.copy()
        # Main effects and reset rule move moderately; interaction weights
        # require a wider scale because products of normalized features shrink.
        v[:model.BASE_FEATURE_COUNT] += rng.normal(
            0.0, 1.5, model.BASE_FEATURE_COUNT
        ).astype(np.float32)
        v[selected_feature_indices] = rng.normal(
            0.0, 14.0, len(selected_feature_indices)
        ).astype(np.float32)
        v[model.N_FEATURES:] += rng.normal(0.0, 1.0, 4).astype(np.float32)
        consider(v, f"broad_{j}")

    # Local evolution.  Only active coordinates move; screened-out
    # interaction coefficients remain exactly zero.
    for scale in (2.0, 1.0, 0.5, 0.25, 0.10):
        for j in range(local_candidates):
            v = best.copy()
            noise = rng.normal(0.0, scale, len(active)).astype(np.float32)
            # Interaction coefficients use a wider local metric.
            interaction_start = model.BASE_FEATURE_COUNT
            interaction_stop = interaction_start + len(selected_feature_indices)
            noise[interaction_start:interaction_stop] *= 5.0
            v[active] += noise
            consider(v, f"local_{scale}_{j}")
    print(json.dumps({
        "stage": "joint_search_final", "evaluations": evaluations,
        "train_score": best_val, "selected_pairs": selected_pairs,
        "selected_squares": selected_squares,
        "weights": best[:model.N_FEATURES].tolist(),
        "reset": best[model.N_FEATURES:].tolist(),
    }), flush=True)
    return best[:model.N_FEATURES], best[model.N_FEATURES:]


def validate_candidates(uniforms, candidates):
    base = model.simulate(
        model.VALIDATED_RDL_WEIGHTS, model.VALIDATED_RDL_RESET, uniforms
    )
    print(json.dumps({
        "stage": "validation_baseline",
        **model.summarize("R+D+L", base),
    }), flush=True)
    best = None
    for name, w, rp in candidates:
        result = model.simulate(w, rp, uniforms, record_opening=True)
        stats = paired_summary(result, base)
        row = {
            "stage": "validation_candidate", "name": name,
            **stats,
            "score_p10": float(np.quantile(result["score"], 0.1)),
            "score_p50": float(np.quantile(result["score"], 0.5)),
            "score_p90": float(np.quantile(result["score"], 0.9)),
            "weights": np.asarray(w).tolist(), "reset": np.asarray(rp).tolist(),
            "mean_sides_by_window": (result["x"] // 3).mean(axis=0).tolist(),
            "activation_probability_by_window": (result["x"] >= 3).mean(axis=0).tolist(),
            "double_probability_by_window": (result["x"] >= 6).mean(axis=0).tolist(),
            "mean_optional_resets": float(result["optional_resets"].mean()),
            "mean_forced_resets": float(result["forced_resets"].mean()),
        }
        print(json.dumps(row), flush=True)
        if best is None or row["mean_score"] > best[1]["mean_score"]:
            best = (name, row)
    return best


def parse_coefficients(text):
    result = {}
    if not text:
        return result
    for part in text.split(","):
        key, value = part.split(":")
        result[int(key)] = float(value)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("screen", "screen-squares", "search"))
    ap.add_argument("--episodes", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument("--selected", default="")
    ap.add_argument("--coefficients", default="")
    ap.add_argument("--selected-squares", default="")
    ap.add_argument("--square-coefficients", default="")
    ap.add_argument("--broad-candidates", type=int, default=220)
    ap.add_argument("--local-candidates", type=int, default=160)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    uniforms = rng.random((args.episodes, model.H, model.N), dtype=np.float32)
    if args.mode == "screen":
        screen(uniforms)
    elif args.mode == "screen-squares":
        screen_squares(uniforms)
    else:
        selected = [int(x) for x in args.selected.split(",") if x]
        coefficients = parse_coefficients(args.coefficients)
        selected_squares = [
            int(x) for x in args.selected_squares.split(",") if x
        ]
        square_coefficients = parse_coefficients(args.square_coefficients)
        missing_pairs = set(selected) - set(coefficients)
        missing_squares = set(selected_squares) - set(square_coefficients)
        if missing_pairs or missing_squares:
            raise SystemExit(
                f"missing pair coefficients {sorted(missing_pairs)}; "
                f"missing square coefficients {sorted(missing_squares)}"
            )
        joint_search(
            rng, uniforms, selected, coefficients,
            selected_squares, square_coefficients,
            args.broad_candidates, args.local_candidates,
        )


if __name__ == "__main__":
    main()
