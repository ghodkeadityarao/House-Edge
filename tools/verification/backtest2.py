"""Roll each query's cutoff back k = 1..6 ticks and score the forecast against the payout that really
happened at that tick (it is in the history). Test histories are never used by any fit, so test rows
are fully out of sample. Validation rows use histories whose targets fitted the tables/correction.

If the model's probabilities are right, observed Brier ~= expected Brier (1 - sum p^2). If the model
had overfitted validation, it would look better on validation than on test.

Run from a folder whose `solution` package matches the params. argv: public_dir params.json
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from slot_forecasting.io import load_inputs
from slot_forecasting.schema import PAYOUTS
from solution import structural as st

public, params_path = Path(sys.argv[1]), Path(sys.argv[2])
P = st.load_params(params_path)
names = ["stable", "shared", "drift", "regime"]
rows = []
for sp in ("validation", "test"):
    q, h = load_inputs(public / sp)
    seen = {(p.episode_id, p.machine_id, p.tick): p.payout for p in h}
    for k in range(1, 7):
        moved = tuple(replace(v, cutoff=v.cutoff - k) for v in q)
        for ep in st.group_episodes(moved, h):
            pred = st.predict_episode(ep, P)
            win = st.infer(ep, P)
            c = "short" if ep.short else names[int(np.argmax(win["conditions"]))]
            for v in ep.queries:
                y = seen.get((v.episode_id, v.machine_id, v.cutoff))
                if y is None:
                    continue
                p = np.array(pred[v.query_id]); o = np.array([float(x == y) for x in PAYOUTS])
                rows.append((sp, k, c, v.machine_type, v.episode_id, ((p - o) ** 2).sum(), 1 - (p ** 2).sum(),
                             1 - p[0], float(y > 0)))
sp_, k_, c_, t_, e_ = (np.array([r[i] for r in rows]) for i in range(5))
obs, exp, pw, won = (np.array([r[i] for r in rows], float) for i in range(5, 9))


def line(label, m):
    # standard error over episodes (rows within an episode are correlated)
    eps = np.unique(e_[m]); per = np.array([(obs[m & (e_ == e)] - exp[m & (e_ == e)]).mean() for e in eps])
    print("  %-26s n=%5d  observed Brier %.4f  expected %.4f  gap %+.4f (+/- %.4f)  p_win %.3f  won %.3f" % (
        label, m.sum(), obs[m].mean(), exp[m].mean(), obs[m].mean() - exp[m].mean(),
        per.std(ddof=1) / np.sqrt(len(eps)), pw[m].mean(), won[m].mean()))


for sp in ("validation", "test"):
    print(sp.upper())
    line("all", sp_ == sp)
    for c in ("stable", "shared", "drift", "regime", "short"):
        if ((sp_ == sp) & (c_ == c)).any():
            line("inferred " + c, (sp_ == sp) & (c_ == c))
    for t in ("steady", "balanced", "jackpot"):
        line("type " + t, (sp_ == sp) & (t_ == t))
print("calibration of p_win on TEST (predicted vs observed win rate):")
for lo, hi in ((0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)):
    m = (sp_ == "test") & (pw >= lo) & (pw < hi)
    if m.any():
        print("  p_win %.1f-%.1f  n=%5d  predicted %.3f  observed %.3f" % (lo, min(hi, 1), m.sum(), pw[m].mean(), won[m].mean()))
