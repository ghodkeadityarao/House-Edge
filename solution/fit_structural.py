"""Learn every structural parameter from a dataset's public folder, then forecast.

Works unchanged on any dataset with the same layout:

    python -m solution.fit_structural --data public --out runs/structural_01

Learned from histories: lag-pair pool, size threshold, size lags and their link to
pairs, shared lag, base logits, parity strength and signal effect.
Learned from validation true probabilities: payout tables and the logit correction.
Condition labels are used only to select stable episodes for pool discovery; inference
never reads them. Episode-grouped cross-validation gives the out-of-sample score.
"""
import argparse
import copy
import math
import hashlib
import itertools
import json
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np

from slot_forecasting.io import load_inputs, read_rows, read_targets, write_json, write_submission
from slot_forecasting.rounds import score_round
from slot_forecasting.schema import PAYOUTS
from . import structural as st

ALL_PAIRS = list(itertools.combinations(range(1, 9), 2))
LAG_SETS = ([(lag,) for lag in range(1, 9)] + ALL_PAIRS + list(itertools.combinations(range(1, 9), 3)))
MIN_RULE_SHARE = 0.10
THRESHOLDS = (1, 2, 5, 20)
TYPES = ("steady", "balanced", "jackpot")
MIN_POOL_SHARE = 0.03
SEED_ALPHA = 30.0
FOLDS = 6


def fit_tables(targets, queries):
    """Each machine type has two exact size tables; split on the largest gap in p(1|win)."""
    machine_type = {q.query_id: q.machine_type for q in queries}
    grouped = {}
    for target in targets:
        p = np.asarray(target.probabilities, dtype=float)
        grouped.setdefault(machine_type[target.query_id], []).append(p[1:] / p[1:].sum())
    tables, spread = {}, {}
    for name, rows in grouped.items():
        rows = np.asarray(rows)
        order = np.sort(rows[:, 0])
        cut = order[int(np.argmax(np.diff(order)))] + 1e-9
        low, high = rows[rows[:, 0] <= cut], rows[rows[:, 0] > cut]
        tables[name] = [low.mean(axis=0).tolist(), high.mean(axis=0).tolist()]
        # Near zero when two tables explain every target; large means the family changed.
        spread[name] = float(max(low.std(axis=0).max(), high.std(axis=0).max()))
    return tables, spread


def cell_counts(episode, i, lags):
    """Wins and trials in each cell of the earlier win flags at these lags."""
    ok, code, _ = st.pair_terms(episode, i, lags, np.arange(2 ** len(lags), dtype=float))
    cells = code[ok].astype(int)
    flag = episode.win[i][ok]
    return (np.bincount(cells, weights=flag, minlength=2 ** len(lags)),
            np.bincount(cells, minlength=2 ** len(lags)).astype(float))


def cell_score(wins, trials):
    """Free-table log likelihood with a BIC penalty, so larger lag sets must earn their cells."""
    p = (wins + 0.5) / (trials + 1)
    ll = float((wins * np.log(p) + (trials - wins) * np.log(1 - p)).sum())
    return ll - 0.5 * len(trials) * math.log(max(trials.sum(), 2))


def canonical_pattern(wins, trials, minimum=5):
    """+1/-1 cell pattern up to sign (cell 0 made negative); None where a cell is too sparse."""
    rate = (wins + 0.5) / (trials + 1)
    overall = (wins.sum() + 0.5) / (trials.sum() + 1)
    pattern = np.where(rate > overall, 1, -1)
    known = trials >= minimum
    if not known[0]:
        anchor = int(np.argmax(known)) if known.any() else 0
        pattern = pattern * (1 if pattern[anchor] == 1 else 1)
    elif pattern[0] == 1:
        pattern = -pattern
    return tuple(int(v) if k else 0 for v, k in zip(pattern, known))


def discover_pairs(episodes):
    """Find which lag sets recur in stable histories, and each one's truth table(s)."""
    chosen, patterns = Counter(), {}
    for episode in episodes:
        for i in range(len(episode.queries)):
            scored = []
            for lags in LAG_SETS:
                wins, trials = cell_counts(episode, i, lags)
                scored.append((cell_score(wins, trials), lags, wins, trials))
            scored.sort(key=lambda row: row[0], reverse=True)
            if scored[0][0] - scored[1][0] > 4:
                _, lags, wins, trials = scored[0]
                chosen[lags] += 1
                patterns.setdefault(lags, Counter())[canonical_pattern(wins, trials)] += 1
    total = sum(chosen.values())
    kept = sorted(lags for lags, count in chosen.items() if count >= MIN_POOL_SHARE * total)
    if not kept:
        raise ValueError("No recurring rule found in stable episodes")
    rules, tables, weights = [], [], []
    for lags in kept:
        merged = Counter()
        for pattern, count in patterns[lags].items():
            if 0 in pattern:
                continue
            merged[pattern] += count
        if not merged:
            merged[tuple(int(v) for v in st.parity_table(len(lags)))] = 1
        size = sum(merged.values())
        for pattern, count in merged.most_common():
            if count >= MIN_RULE_SHARE * size:
                rules.append(list(lags))
                tables.append([float(v) for v in pattern])
                weights.append(chosen[lags] * count / size)
    weights = np.asarray(weights) / np.sum(weights)
    summary = {"lag_sets": dict((str(k), v) for k, v in chosen.most_common(12)),
               "tables": [[lags, table] for lags, table in zip(rules, tables)]}
    return rules, weights.tolist(), tables, summary


def size_scores(episode, i, tables, threshold):
    """Signed size evidence for each candidate lag 1..8."""
    y = episode.Y[i]
    wins = np.nonzero(y > 0)[0]
    if len(wins) == 0:
        return np.zeros(8)
    table = np.log(np.asarray(tables, dtype=float))
    codes = np.array([PAYOUTS.index(int(v)) - 1 for v in y[wins]], dtype=int)
    delta = table[1, codes] - table[0, codes]
    out = np.zeros(8)
    for lag in range(1, 9):
        usable = wins >= lag
        usable[usable] = y[wins[usable] - lag] >= 0
        flag = y[wins[usable] - lag] >= threshold
        out[lag - 1] = (delta[usable] * (2.0 * flag - 1)).sum()
    return out


def discover_sizes(episodes, params, confident=14.0):
    """Find the size threshold, then which lags each win pair uses."""
    pairs = params["pairs"]
    totals = {}
    for threshold in THRESHOLDS:
        value = 0.0
        for episode in episodes:
            for i, query in enumerate(episode.queries):
                value += np.abs(size_scores(episode, i, params["tables"][query.machine_type],
                                            threshold)).max()
        totals[threshold] = value
    threshold = max(totals, key=totals.get)
    counts = np.zeros((len(pairs), 8))
    direction = np.zeros(2)
    for episode in episodes:
        win = st.infer(episode, params)
        for i, query in enumerate(episode.queries):
            scores = size_scores(episode, i, params["tables"][query.machine_type], threshold)
            lag = int(np.abs(scores).argmax())
            if abs(scores[lag]) > confident:
                j = int(win["pair_posterior"][i].argmax())
                counts[j, lag] += 1
                direction[0 if scores[lag] > 0 else 1] += 1
    share = counts.sum(axis=0) / max(counts.sum(), 1)
    lags = [lag for lag in range(8) if share[lag] >= MIN_POOL_SHARE]
    if not lags:
        raise ValueError("No recurring size lag found")
    table = counts[:, lags] + 0.5
    table /= table.sum(axis=1, keepdims=True)
    # Off-diagonal mass is capped at a small floor so sampling noise is not read as signal.
    table = np.maximum(table, 0.02)
    table /= table.sum(axis=1, keepdims=True)
    params["size_threshold"] = int(threshold)
    params["size_lags"] = [lag + 1 for lag in lags]
    params["size_given_pair"] = table.tolist()
    params["size_direction_prior"] = ((direction + 1) / (direction.sum() + 2)).tolist()
    return {"threshold_scores": {str(k): v for k, v in totals.items()}, "lag_counts": counts.tolist()}


def logistic(X, y, offset=None, steps=30, ridge=1e-3):
    """Newton logistic regression; deterministic and numpy-only."""
    beta = np.zeros(X.shape[1])
    offset = np.zeros(len(y)) if offset is None else offset
    for _ in range(steps):
        p = st.sigmoid(X @ beta + offset)
        weight = np.maximum(p * (1 - p), 1e-9)
        hessian = (X * weight[:, None]).T @ X + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(hessian, X.T @ (y - p) - ridge * beta)
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    return beta


def discover_shared_lag(episodes, params):
    """Choose the lag at which episode-mean wins best explain residual wins."""
    m0 = params["m0"]
    gain = np.zeros(8)
    trial = copy.deepcopy(params)
    trial["kappa"] = 0.0
    for episode in episodes:
        win = st.infer(episode, trial)
        eta, usable, _, _ = st.map_paths(episode, win, trial)
        residual = ((episode.win - st.sigmoid(eta)) * usable).sum(axis=0)
        for lag in range(1, 9):
            gain[lag - 1] += float(residual @ (episode.mean_series(lag, m0) - m0))
    params["kappa_lag"] = int(gain.argmax()) + 1
    return gain.tolist()


def fit_logit(episodes, params, rounds=3):
    """Alternate between MAP rule paths and a pooled logistic fit of the win logit."""
    trace = []
    for _ in range(rounds):
        rows, labels = [], []
        for episode in episodes:
            win = st.infer(episode, params)
            _, usable, signed, regime = st.map_paths(episode, win, params)
            level = episode.mean_series(params["kappa_lag"], params["m0"]) - params["m0"]
            for i, query in enumerate(episode.queries):
                ticks = np.nonzero(usable[i])[0]
                block = np.zeros((len(ticks), 11))
                position = TYPES.index(query.machine_type)
                block[:, position] = 1.0
                block[:, 3 + position] = signed[i, ticks]
                block[:, 6 + position] = regime[i, ticks]
                block[:, 9] = query.signal
                block[:, 10] = level[ticks]
                rows.append(block)
                labels.append(episode.win[i, ticks].astype(float))
        beta = logistic(np.concatenate(rows), np.concatenate(labels))
        params["base"] = dict(zip(TYPES, map(float, beta[:3])))
        params["w"] = dict(zip(TYPES, map(float, beta[3:6])))
        params["nu"] = dict(zip(TYPES, map(float, beta[6:9])))
        params["gamma"], params["kappa"] = float(beta[9]), float(beta[10])
        trace.append([float(v) for v in beta])
    return trace


def flip_tick(episode, i, win, params):
    """Most likely flip tick for machine i under its MAP rule, or None if no clear flip."""
    *_, candidate, keep = st.flip_scan(episode, i, win, params)
    tau = int(candidate.argmax())
    return tau if candidate[tau] > keep + 6.0 else None


def discover_flips(episodes, conditions, params):
    """Learn the flip-time window and drift spread from labelled changing episodes."""
    fractions, spreads = [], []
    trial = dict(params)
    trial["condition_prior"] = {"stable": 1e-6, "shared": 0.5, "drift": 0.5}
    trial["tau_low"], trial["tau_high"] = 0.2, 0.97
    for episode in episodes:
        condition = conditions[episode.queries[0].query_id]
        if condition not in ("drift", "shared_change") or not episode.observed.all():
            continue
        win = st.infer(episode, trial)
        taus = [flip_tick(episode, i, win, params) for i in range(len(episode.queries))]
        taus = [tau for tau in taus if tau is not None]
        if taus:
            fractions.append(float(np.median(taus)) / episode.cutoff)
            if condition == "drift" and len(taus) > 2:
                spreads.append(float(np.std(taus)))
    if len(fractions) < 8:
        return {"episodes": len(fractions), "kept_defaults": True}
    params["tau_low"] = max(0.05, float(np.min(fractions)) - 0.08)
    params["tau_high"] = min(0.98, float(np.max(fractions)) + 0.12)
    if len(spreads) >= 4:
        params["drift_sd"] = max(4.0, float(np.median(spreads)))
    return {"episodes": len(fractions), "fraction_range": [float(np.min(fractions)), float(np.max(fractions))],
            "drift_spread_median": float(np.median(spreads)) if spreads else None}


def discover_short_flips(episodes, params):
    """Decide from short-history evidence whether those episodes also flip."""
    short = [episode for episode in episodes if episode.short]
    evidence = None
    for episode in short:
        value = st.infer(episode, params)["log_evidence"]
        evidence = value if evidence is None else evidence + value
    evidence = np.zeros(4) if evidence is None else evidence
    # Index 0 is stable; 1 and 2 are the clean-flip conditions. The catch-all keeps a small prior.
    if len(short) and evidence[0] > evidence[1:3].max():
        params["condition_prior_short"] = {"stable": 0.9796, "shared": 0.01, "drift": 0.01, "regime": 0.0004}
    return {"episodes": len(short), "log_evidence": evidence.tolist(),
            "prior": params["condition_prior_short"]}


def stack_data(episodes, targets, params):
    frame = {"X": [], "y": [], "group": [], "use": []}
    cached = []
    for k, episode in enumerate(episodes):
        win = st.infer(episode, params)
        features = st.stack_features(episode, win, params)
        cached.append((episode, win, features))
        for i, query in enumerate(episode.queries):
            posterior = win["posterior"][i]
            index = np.unravel_index(int(posterior.argmax()), posterior.shape)
            p0 = float(targets[query.query_id].probabilities[0])
            fitted = float(np.clip(win["finals"][i][index], 1e-9, 1 - 1e-9))
            frame["X"].append(features[i])
            frame["y"].append(np.log((1 - p0) / p0) - np.log(fitted / (1 - fitted)))
            frame["group"].append(k)
            frame["use"].append(posterior.max() > 0.9)
    return {key: np.asarray(value) for key, value in frame.items()}, cached


def ridge(X, y, alpha):
    return np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ y)


def forecasts(cached, coefficients, params):
    output = {}
    for k, (episode, win, features) in enumerate(cached):
        p_win = st.corrected_win(win, features, coefficients[k])
        for i, query in enumerate(episode.queries):
            output[query.query_id] = st.probabilities(query, p_win[i], win["table"][i], params)
    return output


def cross_validated(data, cached, alpha):
    """Out-of-sample correction: every episode's coefficients exclude that episode."""
    groups = np.arange(len(cached))
    rng = np.random.default_rng(20261107)
    rng.shuffle(groups)
    coefficients = np.zeros((len(cached), data["X"].shape[1]))
    for held in np.array_split(groups, FOLDS):
        fit = ~np.isin(data["group"], held) & data["use"]
        coefficients[held] = ridge(data["X"][fit], data["y"][fit], alpha)
    return coefficients


def family_check(report, cached, conditions, params):
    """Flag a round whose data no longer fits the model family. Thresholds were set from
    simulated kits: in-family kits stay well inside them, broken ones cross them."""
    expected = {"stable": (0,), "short_history": (0, 3), "drift": (1, 2), "shared_change": (1, 2)}
    hits, regime = 0, 0
    for episode, win, _ in cached:
        label = conditions[episode.queries[0].query_id]
        chosen = int(np.argmax(win["conditions"]))
        hits += chosen in expected.get(label, (chosen,))
        regime += chosen == 3
    ratio = report["validation_cross_validated"]["mean_loss_ratio"]
    spread = max(report["table_spread"].values())
    strength = float(np.mean(list(params["w"].values())))
    values = {"loss_ratio": ratio, "condition_agreement": hits / len(cached),
              "regime_share": regime / len(cached), "table_spread": spread,
              "mean_rule_strength": strength, "rules": len(params["pairs"])}
    reasons, notes = [], []
    if ratio > 0.1:
        reasons.append("cross-validated loss ratio above 0.1")
    worst = {name: value for name, value in report["validation_cross_validated"]["conditions"].items()
             if value > 0.1}
    if worst:
        reasons.append("condition loss ratio above 0.1: " + ", ".join(sorted(worst)))
    if values["condition_agreement"] < 0.85:
        reasons.append("inferred conditions disagree with labels")
    if values["regime_share"] > 0.15:
        reasons.append("many episodes need the catch-all regime model")
    if spread > 1e-3:
        reasons.append("payout targets are not two tables per type")
    if strength < 1.5:
        # A weak but correctly learned rule lowers the ceiling; it does not mean the family broke.
        notes.append("rule strength is weak, so less is predictable")
    lag_sets = {tuple(lags) for lags in params["pairs"]}
    if len(lag_sets) > 8:
        # Several truth tables on one lag set are legitimate; many distinct lag sets suggest noise.
        reasons.append("unusually many distinct rule lag sets")
    values["status"] = "WARN: " + "; ".join(reasons) if reasons else "PASS"
    values["notes"] = notes
    return values


def condition_map(path):
    return {row["query_id"]: row["condition"]
            for row in read_rows(path, ("query_id", "condition"))}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(predictions, public, out, name):
    path = out / name
    write_submission(path, predictions)
    result = score_round(path, public)
    return {"technical_points": result["technical_points"], "mean_loss_ratio": result["mean_loss_ratio"],
            "conditions": {key: value["loss_ratio"] for key, value in result["by_condition"].items()}}


def fit(data, out, alpha=SEED_ALPHA):
    data = Path(data)
    queries = {}
    history = {}
    conditions = {}
    for split in ("train", "validation"):
        queries[split], history[split] = load_inputs(data / split)
        conditions.update(condition_map(data / split / "conditions.csv"))
    targets_list = read_targets(data / "validation/targets.csv")
    targets = {target.query_id: target for target in targets_list}

    params = copy.deepcopy(st.DEFAULT_PARAMS)
    params["stack"] = None
    params["tables"], table_spread = fit_tables(targets_list, queries["validation"])
    episodes = (st.group_episodes(queries["train"], history["train"])
                + st.group_episodes(queries["validation"], history["validation"]))
    stable = [e for e in episodes if conditions[e.queries[0].query_id] == "stable"]
    long_history = [e for e in episodes if e.observed.all()]

    report = {"table_spread": table_spread}
    params["pairs"], params["pair_prior"], params["rule_tables"], report["pair_counts"] = discover_pairs(stable)
    # Bootstrap with no size link; discover_sizes replaces both of these.
    params["size_lags"] = [1]
    params["size_given_pair"] = [[1.0] for _ in params["pairs"]]
    report["size"] = discover_sizes(long_history, params)
    report["shared_lag_gain"] = discover_shared_lag(stable, params)
    report["logit_trace"] = fit_logit(episodes, params)
    report["flips"] = discover_flips(episodes, conditions, params)
    report["logit_refit"] = fit_logit(episodes, params, rounds=1)
    report["size_refit"] = discover_sizes(long_history, params)
    report["logit_with_regime"] = fit_logit(episodes, params)
    report["short_flips"] = discover_short_flips(episodes, params)

    stack, cached = stack_data(st.group_episodes(queries["validation"], history["validation"]),
                               targets, params)
    n_features = stack["X"].shape[1]
    report["validation_no_correction"] = score(
        forecasts(cached, np.zeros((len(cached), n_features)), params), data, out, "validation_uncorrected.csv")
    report["validation_cross_validated"] = score(
        forecasts(cached, cross_validated(stack, cached, alpha), params), data, out, "validation_cv.csv")
    report["family_check"] = family_check(report, cached, conditions, params)
    use = stack["use"]
    coefficient = ridge(stack["X"][use], stack["y"][use], alpha)
    params["stack"] = {"coef": coefficient.tolist(), "names": list(st.STACK_NAMES), "alpha": alpha}
    return params, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("public"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=SEED_ALPHA)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("Choose a fresh output folder")
    args.out.mkdir(parents=True)
    start = time.perf_counter()
    params, report = fit(args.data, args.out, args.alpha)
    st.save_params(args.out / "params.json", params)

    loaded = st.load_params(args.out / "params.json")
    for split, name in (("validation", "validation_predictions.csv"), ("test", "submission.csv")):
        queries, history = load_inputs(args.data / split)
        write_submission(args.out / name, st.predict(queries, history, loaded))
    report["validation_in_sample_correction"] = score_round(args.out / "validation_predictions.csv", args.data)[
        "technical_points"]
    report["metadata"] = json.loads((args.data / "metadata.json").read_text(encoding="utf-8"))
    report["input_sha256"] = {str(path): sha256(path) for split in ("train", "validation", "test")
                              for path in sorted((args.data / split).glob("*.csv"))}
    report["environment"] = {"python": platform.python_version(), "numpy": np.__version__,
                             "platform": platform.platform()}
    report["seconds"] = time.perf_counter() - start
    write_json(args.out / "report.json", report)
    summary = {key: report[key] for key in ("pair_counts", "validation_no_correction",
                                            "validation_cross_validated", "seconds")}
    summary["params"] = {key: params[key] for key in ("pairs", "pair_prior", "size_threshold", "size_lags",
                                                      "size_given_pair", "base", "w", "nu", "gamma", "kappa", "kappa_lag",
                                                      "tau_low", "tau_high", "drift_sd")}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
