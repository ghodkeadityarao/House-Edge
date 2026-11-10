"""60/40 event evaluation. No subjective points or improvement-over-self bonuses."""
import hashlib
import json
import re
from pathlib import Path
import tempfile

from .contest import contest_batch, score_contest_files
from .io import read_rows, write_rows, write_json
from .schema import ValidationError, finite_number, identifier

ROUND_POINTS = {"main": 60, "adaptation": 40}
VERSION = "temporal-adaptation-v3-60-40"
SUITE = "experimentation-suite-v3"
SUPPORTED_SUITES = ("temporal-suite-v2", SUITE)
POINT_COLUMNS = ("season_id", "round", "max_points", "team_id", "technical_points")


def round_metadata(public):
    metadata = json.loads((Path(public) / "metadata.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("suite_version") not in SUPPORTED_SUITES:
        raise ValidationError("This command requires supported temporal round data (v2 or v3).")
    name = metadata.get("round")
    if name not in ROUND_POINTS or metadata.get("max_points") != ROUND_POINTS[name]:
        raise ValidationError("Expected main=60 or adaptation=40 in metadata; old 80/20 kits must be updated by organizers.")
    if not isinstance(metadata.get("season_id"), str) or not re.fullmatch(r"[0-9a-f]{24}", metadata["season_id"]):
        raise ValidationError("Invalid season_id.")
    return metadata


def score_round(submission, public):
    public = Path(public)
    metadata = round_metadata(public)
    validation = public / "validation"
    result = score_contest_files(submission, validation / "targets.csv", validation / "queries.csv",
                                validation / "conditions.csv", validation / "reference.csv",
                                max_points=metadata["max_points"], version=VERSION)
    result.update(round=metadata["round"], season_id=metadata["season_id"])
    return result


def batch_round(manifest, dataset, output):
    from .batch import _publish_ready
    dataset, output = Path(dataset), Path(output).resolve()
    if output.exists():
        raise ValidationError("Choose a new round scoring directory.")
    metadata = round_metadata(dataset / "public")
    metadata_bytes = (dataset / "public/metadata.json").read_bytes()
    if json.loads(metadata_bytes) != metadata:
        raise ValidationError("Round metadata changed during scoring.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".round-") as temporary:
        ready = Path(temporary) / "ready"
        report = contest_batch(manifest, dataset, ready, max_points=metadata["max_points"], version=VERSION)
        report.update(round=metadata["round"], season_id=metadata["season_id"], max_points=metadata["max_points"],
                      metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest())
        report["scorer_sha256"]["rounds.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        rows = [{"season_id": metadata["season_id"], "round": metadata["round"], "max_points": metadata["max_points"],
                 "team_id": team["team_id"], "technical_points": team["contest"]["technical_points"]}
                for team in report["teams"] if team["status"] == "valid"]
        rows.sort(key=lambda row: (-row["technical_points"], row["team_id"]))
        write_rows(ready / "technical.csv", POINT_COLUMNS, rows)
        write_json(ready / "private/run.json", report)
        _publish_ready(ready, output)
    return report


def combine_rounds(main_file, adaptation_file):
    datasets, season_ids = {}, set()
    for name, filename in (("main", main_file), ("adaptation", adaptation_file)):
        entries = {}
        for row in read_rows(filename, POINT_COLUMNS, allow_empty=name == "adaptation"):
            team = identifier(row["team_id"], "team_id")
            if team in entries or row["round"] != name or finite_number(row["max_points"], "max_points") != ROUND_POINTS[name]:
                raise ValidationError("Duplicate team or wrong round/points file.")
            if not re.fullmatch(r"[0-9a-f]{24}", row["season_id"]):
                raise ValidationError("Invalid season_id.")
            season_ids.add(row["season_id"])
            points = finite_number(row["technical_points"], "technical_points")
            if not 0 <= points <= ROUND_POINTS[name]:
                raise ValidationError("Round points are outside their allowed range.")
            entries[team] = points
        datasets[name] = entries
    if len(season_ids) != 1 or not datasets["main"]:
        raise ValidationError("Need a nonempty main round and files from one frozen season.")
    if set(datasets["adaptation"]) - set(datasets["main"]):
        raise ValidationError("Adaptation contains a team without a valid main entry; resolve eligibility first.")
    rows = [{"team_id": team, "main_points": points, "adaptation_points": datasets["adaptation"].get(team, 0),
             "total": points + datasets["adaptation"].get(team, 0)} for team, points in datasets["main"].items()]
    rows.sort(key=lambda r: (-r["total"], r["team_id"]))
    previous, rank = None, 0
    for position, row in enumerate(rows, 1):
        if row["total"] != previous:
            rank = position
        previous, row["rank"] = row["total"], rank
    return rows
