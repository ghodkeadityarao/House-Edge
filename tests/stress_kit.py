"""Simulated round-two kits whose rules deliberately break main-round assumptions.

Each scenario changes one thing against `baseline`, which imitates the main round including
its hidden state, so results are comparable with the real validation score.
"""
import copy
import math
from pathlib import Path
import random
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from simulated_kit import BASE, CONDITIONS, PAYOUTS, TABLES, write

BASELINE = {
    "rule": "xor",
    "pools": {"steady": [[1, 6], [2, 3], [5, 8]], "balanced": [[1, 6], [2, 3], [5, 8]], "jackpot": [[1, 6], [2, 3], [5, 8]]},
    "size_lags": {"1,6": 6, "2,3": 3, "5,8": 5},
    "size_tied": True,
    "threshold": 2,
    "shared_lags": [2],
    "flip": "late",
    "short_flips": False,
    "flip_size": False,
    "w": 2.33,
    "nu": {"steady": -0.26, "balanced": -0.43, "jackpot": -0.51},
    "machine_sd": 0.3,
    "episode_sd": 0.25,
    "tables": TABLES,
}


def scenario(**changes):
    config = copy.deepcopy(BASELINE)
    config.update(changes)
    return config


POOL_5 = [[1, 2], [1, 6], [2, 3], [4, 8], [5, 8]]
SCENARIOS = {
    "baseline": scenario(),
    "same_as_main": scenario(pools={k: [[1, 3], [2, 5], [4, 7]] for k in BASE},
                             size_lags={"1,3": 1, "2,5": 2, "4,7": 4}, shared_lags=[3]),
    "five_pairs": scenario(pools={k: POOL_5 for k in BASE},
                           size_lags={"1,2": 2, "1,6": 6, "2,3": 3, "4,8": 4, "5,8": 5}),
    "pools_by_type": scenario(pools={"steady": [[1, 6]], "balanced": [[2, 3]], "jackpot": [[5, 8]]}),
    "and_rule": scenario(rule="and"),
    "mixed_rules": scenario(rule="mixed"),
    "three_lag": scenario(rule="triple", pools={k: [[1, 3, 6], [2, 4, 7], [3, 5, 8]] for k in BASE},
                          size_lags={"1,3,6": 6, "2,4,7": 2, "3,5,8": 5}),
    "single_lag": scenario(rule="single", pools={k: [[2, 2], [3, 3], [6, 6]] for k in BASE},
                           size_lags={"2,2": 2, "3,3": 3, "6,6": 6}),
    "size_untied": scenario(size_tied=False),
    "size_threshold_5": scenario(threshold=5),
    "early_flips": scenario(flip="early"),
    "two_flips": scenario(flip="two"),
    "late_flips": scenario(flip="very_late"),
    "ramp_flips": scenario(flip="ramp"),
    "partial_flips": scenario(flip="partial"),
    "short_also_flips": scenario(short_flips=True),
    "size_also_flips": scenario(flip_size=True),
    "two_shared_lags": scenario(shared_lags=[2, 5]),
    "weak_rule": scenario(w=1.2),
    "new_tables": scenario(tables={key: (tuple(np.roll(a, 1)), tuple(np.roll(b, 1))) for key, (a, b) in TABLES.items()}),
}


def effect(config, rule_lags, rule_kind, wins, t):
    """Signed rule term for one machine at tick t, from earlier win flags only."""
    flags = [bool(t >= lag and wins[t - lag]) for lag in rule_lags]
    if rule_kind == "and":
        return 1.0 if all(flags) else -1.0
    if rule_kind == "or":
        return 1.0 if any(flags) else -1.0
    if rule_kind == "single":
        return 1.0 if flags[0] else -1.0
    return 1.0 if sum(flags) % 2 else -1.0


def sign_at(config, machine_flip, t, cutoff):
    mode = config["flip"]
    first, second = machine_flip
    if mode == "ramp":
        span = 0.15 * cutoff
        return float(np.clip((first - t) / span, -1, 1))
    if mode == "partial":
        return 1.0 if t < first else 0.25
    value = 1.0 if t < first else -1.0
    return value if t < second else -value


def simulate(rng, config, condition, cutoff, machines=8):
    types = [rng.choice(sorted(BASE)) for _ in range(machines)]
    signals = [rng.choice((-1, 0, 1)) for _ in range(machines)]
    rules, kinds = [], []
    for name in types:
        rules.append(rng.choice(config["pools"][name]))
        kinds.append(rng.choice(("xor", "and", "or")) if config["rule"] == "mixed" else config["rule"])
    signs = [rng.choice((-1, 1)) for _ in range(machines)]
    directions = [rng.randrange(2) for _ in range(machines)]
    size_lags = [config["size_lags"][",".join(map(str, rule))] if config["size_tied"]
                 else rng.choice(sorted(set(config["size_lags"].values()))) for rule in rules]
    offsets = [rng.gauss(0, config["machine_sd"]) for _ in range(machines)]
    low, high = {"early": (.3, .45), "very_late": (.85, .93)}.get(config["flip"], (.55, .77))
    centre = int(cutoff * rng.uniform(low, high))
    flips = [(10 ** 6, 10 ** 7)] * machines
    changing = condition in ("drift", "shared_change") or (condition == "short_history" and config["short_flips"])
    if changing:
        spread = 13 if condition == "drift" else 0
        for i in range(machines):
            first = int(rng.gauss(centre, spread)) if spread else centre
            second = first + int(.2 * cutoff) if config["flip"] == "two" else 10 ** 7
            flips[i] = (first, second)
    y = np.zeros((machines, cutoff + 1), dtype=int)
    target = [None] * machines
    level = 0.0
    for t in range(cutoff + 1):
        level = 0.9 * level + rng.gauss(0, config["episode_sd"] * math.sqrt(1 - .81))
        shared = sum(((y[:, t - lag] > 0).mean() - .45) if t >= lag else 0.0 for lag in config["shared_lags"])
        for i in range(machines):
            name = types[i]
            sign = signs[i] * sign_at(config, flips[i], t, cutoff)
            term = effect(config, rules[i], kinds[i], y[i] > 0, t)
            lag = size_lags[i]
            flag = t >= lag and y[i, t - lag] >= config["threshold"]
            active = flag != bool(directions[i])
            if config["flip_size"] and t >= flips[i][0]:
                active = not active
            eta = (BASE[name] + .33 * signals[i] + 1.02 * shared + offsets[i] + level
                   + sign * config["w"] * term + config["nu"][name] * (1 if active else -1))
            p = 1 / (1 + math.exp(-eta))
            row = (1 - p, *[p * float(v) for v in config["tables"][name][int(active)]])
            if t == cutoff:
                target[i] = row
            y[i, t] = rng.choices(PAYOUTS, weights=row)[0]
    return types, signals, y, target


def keep_ticks(rng, condition, cutoff):
    if condition != "short_history":
        return list(range(cutoff))
    return sorted(rng.sample(range(cutoff - 40), 24) + list(range(cutoff - 40, cutoff)))


def build(root, config, seed=11, per_condition=(24, 24, 2)):
    """Write a kit. The reference is a real pooled forecaster fitted on the kit's training data."""
    import json
    from slot_forecasting.baseline import FrequencyForecaster
    from slot_forecasting.io import load_inputs, read_labels
    rng = random.Random(seed)
    root = Path(root)
    prob = ["p_0", "p_1", "p_2", "p_5", "p_20"]
    for split, count in zip(("train", "validation", "test"), per_condition):
        queries, history, labels, conditions, targets = [], [], [], [], []
        for number in range(count * 4):
            condition = CONDITIONS[number % 4]
            cutoff = rng.randrange(192, 257)
            types, signals, y, target = simulate(rng, config, condition, cutoff)
            episode = "sim-%s-e%03d" % (split, number)
            for i, machine_type in enumerate(types):
                machine = "%s-m%d" % (episode, i)
                query = machine + "-next"
                queries.append((query, episode, machine, machine_type, signals[i], cutoff))
                history += [(episode, machine, t, y[i, t]) for t in keep_ticks(rng, condition, cutoff)]
                labels.append((query, y[i, cutoff]))
                conditions.append((query, condition))
                targets.append((query, episode, *["%.17g" % v for v in target[i]], y[i, cutoff]))
        base = root / split
        write(base / "queries.csv", ("query_id", "episode_id", "machine_id", "machine_type", "signal", "cutoff"), queries)
        write(base / "history.csv", ("episode_id", "machine_id", "tick", "payout"), history)
        write(base / "sample_submission.csv", ("query_id", *prob), [(q[0], .2, .2, .2, .2, .2) for q in queries])
        if split != "test":
            write(base / "labels.csv", ("query_id", "outcome"), labels)
            write(base / "conditions.csv", ("query_id", "condition"), conditions)
        if split == "validation":
            write(base / "targets.csv", ("query_id", "episode_id", *prob, "outcome"), targets)
    train_q, train_h = load_inputs(root / "train")
    model = FrequencyForecaster("pooled").fit(train_q, train_h, read_labels(root / "train/labels.csv", train_q))
    valid_q, valid_h = load_inputs(root / "validation")
    forecast = model.predict(valid_q, valid_h)
    write(root / "validation/reference.csv", ("query_id", *prob),
          [(key, *["%.17g" % v for v in forecast[key]]) for key in sorted(forecast)])
    (root / "metadata.json").write_text(json.dumps({
        "conditions": list(CONDITIONS), "max_points": 40, "round": "adaptation",
        "season_id": "0" * 24, "suite_version": "experimentation-suite-v3"}), encoding="utf-8")
    return root
