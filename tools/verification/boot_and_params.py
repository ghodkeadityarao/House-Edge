"""Episode bootstrap of the cross-validated score, and fitted constants side by side across rounds.

argv: public_dir prediction_csv max_points main_params adaptation_params
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

public, pred_path, max_points = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
V = public / "validation"
read = lambda p: list(csv.DictReader(open(p, encoding="utf-8")))
cols = ["p_0", "p_1", "p_2", "p_5", "p_20"]
vec = lambda r: np.array([float(r[c]) for c in cols])
truth = {r["query_id"]: (r["episode_id"], vec(r)) for r in read(V / "targets.csv")}
cond = {r["query_id"]: r["condition"] for r in read(V / "conditions.csv")}
pred = {r["query_id"]: vec(r) for r in read(pred_path)}
ref = {r["query_id"]: vec(r) for r in read(V / "reference.csv")}
ep_err, ep_ref, ep_cond = defaultdict(list), defaultdict(list), {}
for q, (e, t) in truth.items():
    ep_err[e].append(((pred[q] - t) ** 2).sum()); ep_ref[e].append(((ref[q] - t) ** 2).sum()); ep_cond[e] = cond[q]
by_cond = defaultdict(list)
for e in ep_err:
    by_cond[ep_cond[e]].append((np.mean(ep_err[e]), np.mean(ep_ref[e])))
by_cond = {c: np.array(v) for c, v in by_cond.items()}


def points(groups):
    R = np.mean([g[:, 0].mean() / max(g[:, 1].mean(), 1e-9) for g in groups.values()])
    return max_points / (1 + R)


print("recomputed cross-validated points %.3f / %g" % (points(by_cond), max_points))
rng = np.random.default_rng(20261108)
boot = [points({c: g[rng.integers(0, len(g), len(g))] for c, g in by_cond.items()}) for _ in range(2000)]
print("episode bootstrap 95%% interval: %.3f - %.3f   (5th percentile %.3f)" % (
    np.percentile(boot, 2.5), np.percentile(boot, 97.5), np.percentile(boot, 5)))

pm, pa = (json.load(open(p)) for p in sys.argv[4:6])
print("\nfitted constant             main round        adaptation round")
for key in ("gamma", "kappa", "kappa_lag", "drift_sd", "tau_low", "tau_high", "pairs", "size_lags"):
    fmt = lambda v: json.dumps(v) if not isinstance(v, float) else "%.3f" % v
    print("  %-24s %-17s %s" % (key, fmt(pm.get(key)), fmt(pa.get(key))))
for key in ("base", "w", "nu"):
    for t in ("steady", "balanced", "jackpot"):
        print("  %-24s %-17.3f %.3f" % (key + " " + t, pm[key][t], pa[key][t]))
