"""Make two kits whose validation split holds only one half of the validation episodes.

Run with the kit folder as the working directory (uses the kit's own episode_folds split).
A/public/validation = the kit's tuning half, B/public/validation = its audit half.
Train, test and metadata are copied unchanged.
"""
import csv
import json
import shutil
import sys
from pathlib import Path

from slot_forecasting.experiments import episode_folds
from slot_forecasting.io import read_targets

public = Path(sys.argv[1]).resolve()
dest = Path(sys.argv[2]).resolve()
assert not dest.exists()
V = public / "validation"
targets = read_targets(V / "targets.csv")
cond = {r["query_id"]: r["condition"] for r in csv.DictReader(open(V / "conditions.csv", encoding="utf-8"))}
tune, audit = episode_folds(targets, cond)
q_episode = {r["query_id"]: r["episode_id"] for r in csv.DictReader(open(V / "queries.csv", encoding="utf-8"))}
for name, keep in (("A", tune), ("B", audit)):
    out = dest / name / "public"
    shutil.copytree(public / "train", out / "train")
    shutil.copytree(public / "test", out / "test")
    shutil.copyfile(public / "metadata.json", out / "metadata.json")
    (out / "validation").mkdir()
    for f in sorted(V.glob("*.csv")):
        with open(f, encoding="utf-8", newline="") as src:
            reader = csv.reader(src)
            header = next(reader)
            key = header.index("episode_id") if "episode_id" in header else None
            qkey = header.index("query_id") if key is None else None
            rows = [r for r in reader if (r[key] if key is not None else q_episode[r[qkey]]) in keep]
        with open(out / "validation" / f.name, "w", encoding="utf-8", newline="") as dst:
            w = csv.writer(dst, lineterminator="\n")
            w.writerow(header)
            w.writerows(rows)
    print(name, len(keep), "episodes")
json.dump({"A_tuning": sorted(tune), "B_audit": sorted(audit)}, open(dest / "split.json", "w"), indent=1)
