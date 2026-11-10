"""Score a model fitted on one half of the validation episodes on the other half, once.

Run from the code workspace that produced the run (PYTHONPATH=.).
argv: full_public_dir run_dir split.json half_to_score(A_tuning|B_audit) max_points
"""
import csv
import json
import sys
from pathlib import Path

from slot_forecasting.experiments import subset_score
from slot_forecasting.io import load_inputs, read_targets
from solution import structural as st

public, run, split, half, max_points = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], float(sys.argv[5])
V = public / "validation"
keep = set(json.load(open(split))[half])
queries, history = load_inputs(V)
queries = tuple(q for q in queries if q.episode_id in keep)
history = tuple(h for h in history if h.episode_id in keep)
targets = read_targets(V / "targets.csv")
cond = {r["query_id"]: r["condition"] for r in csv.DictReader(open(V / "conditions.csv", encoding="utf-8"))}
cols = ("p_0", "p_1", "p_2", "p_5", "p_20")
ref = {r["query_id"]: tuple(float(r[c]) for c in cols) for r in csv.DictReader(open(V / "reference.csv", encoding="utf-8"))}
params = st.load_params(run / "params.json")
pred = st.predict(queries, history, params)
assert len(pred) == len(queries) == 8 * len(keep)
r = subset_score(pred, targets, cond, ref, keep, max_points)
own = json.load(open(run / "report.json"))
print("%s: fitted on the other half, scored once on %s (%d episodes): %.3f / %g   ratios %s" % (
    run.name, half, len(keep), r["technical_points"], max_points,
    {k: round(v["loss_ratio"], 4) for k, v in r["by_condition"].items()}))
print("   that half-fit's own cross-validated score on its training half: %.3f; family check %s; rules %s" % (
    own["validation_cross_validated"]["technical_points"], own.get("family_check", {}).get("status", "n/a"),
    params.get("pairs")))
