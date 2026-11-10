"""Exact simulator scoring: squared probability error with equal episode weights."""

from collections import defaultdict
import math
import random

from .io import read_queries, read_submission, read_targets, require_ids
from .schema import PAYOUTS, SCORING_VERSION, ValidationError, integer, probabilities


def distribution_error(prediction, truth):
    p, q = probabilities(prediction), probabilities(truth, "target probabilities")
    return math.fsum((a - b) ** 2 for a, b in zip(p, q))


def mean_payout(distribution):
    return math.fsum(value * p for value, p in zip(PAYOUTS, probabilities(distribution)))


def payout_quantile(distribution, level):
    if not isinstance(level, (int, float)) or isinstance(level, bool) or not math.isfinite(level) or not 0 < level <= 1:
        raise ValidationError("Quantile level must be in (0, 1].")
    cumulative = 0.0
    for value, probability in zip(PAYOUTS, probabilities(distribution)):
        cumulative += probability
        if cumulative >= level:
            return value
    return PAYOUTS[-1]


def _percentile(values, level):
    ordered = sorted(values)
    position = (len(ordered) - 1) * level
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def bootstrap_mean_interval(values, repeats=1000, seed=314159):
    """Resample independent episodes, not correlated machine-level rows."""
    values = tuple(values)
    if not values or any(not math.isfinite(value) for value in values):
        raise ValidationError("Bootstrap requires nonempty finite episode values.")
    repeats = integer(repeats, "bootstrap repeats", 1)
    seed = integer(seed, "bootstrap seed")
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    samples = [math.fsum(rng.choices(values, k=len(values))) / len(values) for _ in range(repeats)]
    return [_percentile(samples, 0.025), _percentile(samples, 0.975)]


def score_predictions(predictions, targets, *, bootstrap_repeats=0):
    bootstrap_repeats = integer(bootstrap_repeats, "bootstrap repeats")
    target_map = {}
    for target in targets:
        if target.query_id in target_map:
            raise ValidationError(f"Duplicate target query_id: {target.query_id}.")
        target_map[target.query_id] = target
    if not target_map:
        raise ValidationError("Cannot score an empty target set.")
    require_ids(predictions, target_map)
    normalized = {key: probabilities(predictions[key], key) for key in sorted(predictions)}
    grouped = defaultdict(list)
    for key in sorted(target_map):
        grouped[target_map[key].episode_id].append(key)
    episode_errors = {}
    diagnostics = defaultdict(float)
    calibration = defaultdict(lambda: [0.0, 0.0, 0.0])
    for episode, ids in sorted(grouped.items()):
        errors = []
        weight = 1 / (len(grouped) * len(ids))
        for key in ids:
            p = normalized[key]
            target = target_map[key]
            q = target.probabilities
            errors.append(distribution_error(p, q))
            forecast, expected = mean_payout(p), mean_payout(q)
            low, high = payout_quantile(p, 0.1), payout_quantile(p, 0.9)
            diagnostics["expected_payout_mae"] += weight * abs(forecast - expected)
            diagnostics["realized_payout_mae"] += weight * abs(forecast - target.outcome)
            diagnostics["outcome_brier"] += weight * math.fsum((v - float(outcome == target.outcome)) ** 2 for outcome, v in zip(PAYOUTS, p))
            diagnostics["expected_interval_coverage_80"] += weight * math.fsum(v for outcome, v in zip(PAYOUTS, q) if low <= outcome <= high)
            diagnostics["realized_interval_coverage_80"] += weight * float(low <= target.outcome <= high)
            diagnostics["mean_interval_width_80"] += weight * (high - low)
            event_probability = 1 - p[0]
            bin_id = min(9, int(event_probability * 10))
            entry = calibration[bin_id]
            entry[0] += weight
            entry[1] += weight * event_probability
            entry[2] += weight * (1 - q[0])
        episode_errors[episode] = math.fsum(errors) / len(errors)
    error = math.fsum(episode_errors.values()) / len(episode_errors)
    bins = [{"bin": index, "weight": v[0], "forecast_nonzero": v[1] / v[0], "true_nonzero": v[2] / v[0]}
            for index, v in sorted(calibration.items())]
    diagnostics["nonzero_calibration_error"] = math.fsum(abs(v[1] - v[2]) for v in calibration.values())
    report = {
        "scoring_version": SCORING_VERSION,
        "score": max(0.0, min(100.0, 100 * (1 - error / 2))),
        "probability_error": error,
        "n_queries": len(target_map), "n_episodes": len(grouped),
        "episode_errors": episode_errors,
        "diagnostics": dict(diagnostics), "calibration_bins": bins,
    }
    if bootstrap_repeats:
        interval = bootstrap_mean_interval(episode_errors.values(), bootstrap_repeats)
        report["score_ci95"] = None if interval is None else [100 * (1 - interval[1] / 2), 100 * (1 - interval[0] / 2)]
    return report


def score_files(submission_path, target_path, query_path=None, *, bootstrap_repeats=0):
    targets = read_targets(target_path)
    if query_path is not None:
        queries = {q.query_id: q for q in read_queries(query_path)}
        require_ids({t.query_id for t in targets}, queries)
        if any(t.episode_id != queries[t.query_id].episode_id for t in targets):
            raise ValidationError("Target episode IDs do not match the query manifest.")
    predictions = read_submission(submission_path, {t.query_id for t in targets})
    return score_predictions(predictions, targets, bootstrap_repeats=bootstrap_repeats)
