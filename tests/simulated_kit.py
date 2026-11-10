"""Write a miniature public kit from a known simulator with non-main rules.

Used to prove the fit pipeline learns its rules rather than assuming main-round ones.
"""
import csv
import json
import math
from pathlib import Path
import random

import numpy as np

PAYOUTS = (0, 1, 2, 5, 20)
TABLES = {
    "steady": ((.455, .436, .107, .002), (.754, .197, .048, .001)),
    "balanced": ((.214, .547, .205, .034), (.5, .348, .13, .022)),
    "jackpot": ((.121, .324, .416, .139), (.335, .245, .315, .105)),
}
BASE = {"steady": .6, "balanced": -.6, "jackpot": -1.9}
CONDITIONS = ("stable", "short_history", "drift", "shared_change")
TRUTH = {"pairs": [[1, 6], [2, 3], [5, 8]], "size_lags": [6, 3, 5], "shared_lag": 2, "threshold": 2}


def simulate_episode(rng, condition, cutoff, machines=8):
    types = [rng.choice(sorted(BASE)) for _ in range(machines)]
    signals = [rng.choice((-1, 0, 1)) for _ in range(machines)]
    rules = [rng.randrange(3) for _ in range(machines)]
    signs = [rng.choice((-1, 1)) for _ in range(machines)]
    directions = [rng.randrange(2) for _ in range(machines)]
    centre = int(cutoff * rng.uniform(.6, .75))
    flips = [10 ** 6] * machines
    if condition == "shared_change":
        flips = [centre] * machines
    elif condition == "drift":
        flips = [int(rng.gauss(centre, 12)) for _ in range(machines)]
    y = np.zeros((machines, cutoff + 1), dtype=int)
    target = [None] * machines
    for t in range(cutoff + 1):
        lagged = (y[:, t - TRUTH["shared_lag"]] > 0).mean() if t >= TRUTH["shared_lag"] else .45
        for i in range(machines):
            a, b = TRUTH["pairs"][rules[i]]
            lag = TRUTH["size_lags"][rules[i]]
            e = 2.0 * int((t >= a and y[i, t - a] > 0) != (t >= b and y[i, t - b] > 0)) - 1
            sign = signs[i] if t < flips[i] else -signs[i]
            eta = BASE[types[i]] + .33 * signals[i] + lagged - .45 + sign * 2.3 * e
            p = 1 / (1 + math.exp(-eta))
            flag = t >= lag and y[i, t - lag] >= TRUTH["threshold"]
            sizes = TABLES[types[i]][int(flag != bool(directions[i]))]
            probabilities = (1 - p, *[p * v for v in sizes])
            if t == cutoff:
                target[i] = probabilities
            y[i, t] = rng.choices(PAYOUTS, weights=probabilities)[0]
    return types, signals, y, target


def keep_ticks(rng, condition, cutoff):
    if condition != "short_history":
        return list(range(cutoff))
    return sorted(rng.sample(range(cutoff - 40), 24) + list(range(cutoff - 40, cutoff)))


def write(path, columns, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def build(root, seed=1, per_condition=(12, 12, 6)):
    rng = random.Random(seed)
    root = Path(root)
    prob = ["p_0", "p_1", "p_2", "p_5", "p_20"]
    for split, count in zip(("train", "validation", "test"), per_condition):
        queries, history, labels, conditions, targets, reference = [], [], [], [], [], []
        for number in range(count * 4):
            condition = CONDITIONS[number % 4]
            cutoff = rng.randrange(192, 257)
            types, signals, y, target = simulate_episode(rng, condition, cutoff)
            episode = "sim-%s-e%03d" % (split, number)
            for i, machine_type in enumerate(types):
                machine = "%s-m%d" % (episode, i)
                query = machine + "-next"
                queries.append((query, episode, machine, machine_type, signals[i], cutoff))
                history += [(episode, machine, t, y[i, t]) for t in keep_ticks(rng, condition, cutoff)]
                labels.append((query, y[i, cutoff]))
                conditions.append((query, condition))
                targets.append((query, episode, *["%.17g" % v for v in target[i]], y[i, cutoff]))
                reference.append((query, .5, .2, .15, .1, .05))
        base = root / split
        write(base / "queries.csv", ("query_id", "episode_id", "machine_id", "machine_type", "signal", "cutoff"), queries)
        write(base / "history.csv", ("episode_id", "machine_id", "tick", "payout"), history)
        write(base / "sample_submission.csv", ("query_id", *prob), [(q[0], .2, .2, .2, .2, .2) for q in queries])
        if split != "test":
            write(base / "labels.csv", ("query_id", "outcome"), labels)
            write(base / "conditions.csv", ("query_id", "condition"), conditions)
        if split == "validation":
            write(base / "targets.csv", ("query_id", "episode_id", *prob, "outcome"), targets)
            write(base / "reference.csv", ("query_id", *prob), reference)
    (root / "metadata.json").write_text(json.dumps({
        "conditions": list(CONDITIONS), "max_points": 40, "round": "adaptation",
        "season_id": "0" * 24, "suite_version": "experimentation-suite-v3"}), encoding="utf-8")
    return root
