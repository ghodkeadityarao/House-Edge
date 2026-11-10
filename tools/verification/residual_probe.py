"""Where does the adaptation model's remaining validation error come from, and is any of it learnable?

Uses only validation targets as an analysis target (as the fit already does for the correction);
learnability is judged out-of-fold by episode, so a gain here is not in-sample.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

data = Path(sys.argv[1])
pred_path = Path(sys.argv[2])
V = data / "validation"
read = lambda p: list(csv.DictReader(open(p, encoding="utf-8")))
queries = {r["query_id"]: r for r in read(V / "queries.csv")}
cond = {r["query_id"]: r["condition"] for r in read(V / "conditions.csv")}
cols = ["p_0", "p_1", "p_2", "p_5", "p_20"]
truth = {r["query_id"]: np.array([float(r[c]) for c in cols]) for r in read(V / "targets.csv")}
pred = {r["query_id"]: np.array([float(r[c]) for c in cols]) for r in read(pred_path)}
ref = {r["query_id"]: np.array([float(r[c]) for c in cols]) for r in read(V / "reference.csv")}
hist = defaultdict(dict)
for r in read(V / "history.csv"):
    hist[(r["episode_id"], r["machine_id"])][int(r["tick"])] = int(r["payout"])

ids = sorted(queries)
err = np.array([((pred[q] - truth[q]) ** 2).sum() for q in ids])
rerr = np.array([((ref[q] - truth[q]) ** 2).sum() for q in ids])
win_err = np.array([2 * ((1 - pred[q][0]) - (1 - truth[q][0])) ** 2 for q in ids])  # p_0 part, roughly
print("mean error model %.5f  reference %.5f" % (err.mean(), rerr.mean()))

print("\nerror share by condition x type (model error / reference error):")
groups = defaultdict(list)
for i, q in enumerate(ids):
    groups[(cond[q], queries[q]["machine_type"])].append(i)
for key in sorted(groups):
    idx = groups[key]
    print("  %-14s %-9s n=%3d  ratio %.4f  share of total error %.1f%%" % (
        key[0], key[1], len(idx), err[idx].mean() / rerr[idx].mean(), 100 * err[idx].sum() / err.sum()))

# split each query's error into win-probability part and size-given-win part
pw_t = np.array([1 - truth[q][0] for q in ids]); pw_p = np.array([1 - pred[q][0] for q in ids])
size_t = np.array([truth[q][1:] / max(1e-12, 1 - truth[q][0]) for q in ids])
size_p = np.array([pred[q][1:] / max(1e-12, 1 - pred[q][0]) for q in ids])
alt = np.array([((np.r_[1 - pw_t[i], pw_t[i] * size_p[i]] - truth[q]) ** 2).sum() for i, q in enumerate(ids)])
print("\nif win probability were exact (keep model's size split): error %.5f (%.0f%% of model error remains)" % (
    alt.mean(), 100 * alt.mean() / err.mean()))
alt2 = np.array([((np.r_[1 - pw_p[i], pw_p[i] * size_t[i]] - truth[q]) ** 2).sum() for i, q in enumerate(ids)])
print("if size split were exact (keep model's win probability): error %.5f (%.0f%% remains)" % (
    alt2.mean(), 100 * alt2.mean() / err.mean()))

# Learnability probe: residual win logit vs observable pre-cutoff features, out-of-fold by episode.
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
resid = logit(pw_t) - logit(pw_p)
X, names = [], []
for q in ids:
    r = queries[q]; h = hist[(r["episode_id"], r["machine_id"])]; c = int(r["cutoff"])
    row = []
    for lag in range(1, 9):
        v = h.get(c - lag)
        row += [0.0 if v is None else float(v > 0), 0.0 if v is None else float(v >= 2), float(v is None)]
    # pairwise products of win flags at lags 1..8 (captures any missed parity/AND rule)
    w = [0.0 if h.get(c - l) is None else float(h[c - l] > 0) * 2 - 1 for l in range(1, 9)]
    for a in range(8):
        for b in range(a + 1, 8):
            row.append(w[a] * w[b])
    row += [float(r["signal"]), float(r["machine_type"] == "steady"), float(r["machine_type"] == "jackpot"),
            logit(pw_p[len(X)]), c / 256.0]
    X.append(row)
X = np.array(X)
if not names:
    names = [f"win{l}" for l in range(1, 9)]
episodes = sorted({queries[q]["episode_id"] for q in ids})
rng = np.random.default_rng(20261108); rng.shuffle(episodes)
fold = {e: i % 6 for i, e in enumerate(episodes)}
f = np.array([fold[queries[q]["episode_id"]] for q in ids])
oof = np.zeros(len(ids))
for k in range(6):
    tr, te = f != k, f == k
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    A = np.c_[np.ones(tr.sum()), (X[tr] - mu) / sd]
    beta = np.linalg.solve(A.T @ A + 30 * np.eye(A.shape[1]), A.T @ resid[tr])
    oof[te] = np.c_[np.ones(te.sum()), (X[te] - mu) / sd] @ beta
sig = lambda z: 1 / (1 + np.exp(-z))
new_pw = sig(logit(pw_p) + oof)
new = np.array([((np.r_[1 - new_pw[i], new_pw[i] * size_p[i]] - truth[q]) ** 2).sum() for i, q in enumerate(ids)])
print("\nresidual win-logit: sd %.3f; out-of-fold ridge on 89 lag/pair/type features explains %.1f%% of its variance" % (
    resid.std(), 100 * (1 - ((resid - oof) ** 2).mean() / resid.var())))
print("error after applying that out-of-fold correction: %.5f vs %.5f now" % (new.mean(), err.mean()))
