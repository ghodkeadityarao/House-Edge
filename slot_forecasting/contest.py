"""Versioned event ranking and explicit human review; never automated AI detection."""
import math
from pathlib import Path

from .io import read_rows, read_targets, read_submission, read_queries, require_ids, write_json, write_rows
from .schema import CONDITIONS, ValidationError, finite_number
from .scoring import score_predictions

CONTEST_VERSION = "evidence-contest-v1"
REVIEW_MAX = {"hypothesis": 5, "experiments": 10, "failure_analysis": 5, "reproducibility": 10}


def contest_score(predictions, targets, groups, reference, *, max_points=70, version=CONTEST_VERSION):
    """Each condition has equal weight; a frozen reference supplies loss units."""
    max_points = finite_number(max_points, "max_points")
    if max_points <= 0 or max_points > 100:
        raise ValidationError("max_points must be in (0,100].")
    targets = tuple(targets)
    ids = {t.query_id for t in targets}
    require_ids(groups, ids)
    require_ids(predictions, ids)
    require_ids(reference, ids)
    if set(groups.values()) != set(CONDITIONS):
        raise ValidationError("Every announced condition must occur, with no unknown conditions.")
    episodes = {}
    for target in targets:
        condition = groups[target.query_id]
        if target.episode_id in episodes and episodes[target.episode_id] != condition:
            raise ValidationError("One episode cannot span multiple conditions.")
        episodes[target.episode_id] = condition
    by_condition = {}
    for condition in CONDITIONS:
        subset = [t for t in targets if groups[t.query_id] == condition]
        p = {t.query_id: predictions[t.query_id] for t in subset}
        ref = {t.query_id: reference[t.query_id] for t in subset}
        result, baseline = score_predictions(p, subset), score_predictions(ref, subset)
        denominator = max(1e-9, baseline["probability_error"])
        by_condition[condition] = {"error": result["probability_error"], "reference_error": baseline["probability_error"],
                                   "loss_ratio": result["probability_error"] / denominator,
                                   "episodes": result["n_episodes"], "queries": result["n_queries"]}
    ratio = math.fsum(v["loss_ratio"] for v in by_condition.values()) / len(CONDITIONS)
    return {"contest_version": version, "technical_points": max_points / (1 + ratio), "max_points": max_points,
            "mean_loss_ratio": ratio, "by_condition": by_condition}


def score_contest_files(submission, targets, queries, conditions, reference, *, max_points=70, version=CONTEST_VERSION):
    ts = read_targets(targets)
    qs = {q.query_id: q for q in read_queries(queries)}
    require_ids({t.query_id for t in ts}, qs)
    if any(qs[t.query_id].episode_id != t.episode_id for t in ts):
        raise ValidationError("Target episode IDs do not match the query manifest.")
    groups = {}
    for row in read_rows(conditions, ("query_id", "condition")):
        if row["query_id"] in groups:
            raise ValidationError("Duplicate condition query_id.")
        groups[row["query_id"]] = row["condition"]
    return contest_score(read_submission(submission, qs), ts, groups, read_submission(reference, qs), max_points=max_points, version=version)


def finalize_reviews(technical, review_rows):
    """Combine organizer-generated technical points with two or more human reviews."""
    if not technical:
        raise ValidationError("No eligible teams to finalize.")
    for value in technical.values():
        if not 0 <= finite_number(value, "technical_points") <= 70:
            raise ValidationError("technical_points must be in [0,70].")
    reviews = {team: {} for team in technical}
    for row in review_rows:
        team, judge = row["team_id"], row["judge_id"].strip().casefold()
        if team not in reviews or not judge or judge in reviews[team]:
            raise ValidationError("Unknown team, empty judge, or duplicate team/judge review.")
        if not row["evidence_ref"].strip():
            raise ValidationError("Every review needs a reference to inspected artifacts/notes.")
        points = []
        for key, maximum in REVIEW_MAX.items():
            value = finite_number(row[key], key)
            if not 0 <= value <= maximum:
                raise ValidationError(f"{key} must be in [0,{maximum}].")
            points.append(value)
        reviews[team][judge] = math.fsum(points)
    panels = [set(v) for v in reviews.values()]
    if any(len(panel) < 2 or panel != panels[0] for panel in panels):
        raise ValidationError("Every eligible team needs the same panel of at least two independent judges.")
    result = []
    for team, entries in reviews.items():
        human = math.fsum(entries.values()) / len(entries)
        technical_points = float(technical[team])
        result.append({"team_id": team, "technical_points": technical_points, "review_points": human,
                       "total": technical_points + human})
    result.sort(key=lambda r: (-r["total"], r["team_id"]))
    previous, rank = None, 0
    for position, row in enumerate(result, 1):
        if row["total"] != previous:
            rank = position
        previous = row["total"]
        row["rank"] = rank
    return result


def contest_batch(manifest, dataset, output, *, max_points=70, version=CONTEST_VERSION):
    """Reuse strict batch intake; save event rankings separately from legacy raw ranks."""
    from .batch import score_batch, _manifest, _publish_ready
    import tempfile
    output, dataset = Path(output).resolve(), Path(dataset)
    if output.exists():
        raise ValidationError("Choose a new output directory.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".contest-") as tmp:
        ready = Path(tmp) / "ready"
        report = score_batch(manifest, dataset / "organizer/test/targets.csv", dataset / "public/test/queries.csv", ready)
        target_bytes = (dataset / "organizer/test/targets.csv").read_bytes()
        condition_bytes = (dataset / "organizer/test/conditions.csv").read_bytes()
        reference_bytes = (dataset / "organizer/test/reference.csv").read_bytes()
        manifest_bytes = Path(manifest).read_bytes()
        targets = read_targets("targets snapshot", content=target_bytes)
        groups_rows = read_rows("conditions snapshot", ("query_id", "condition"), content=condition_bytes)
        groups = {row["query_id"]: row["condition"] for row in groups_rows}
        if len(groups) != len(groups_rows):
            raise ValidationError("Duplicate condition query_id.")
        reference = read_submission("reference snapshot", content=reference_bytes)
        # Re-read inputs only with explicit equality checks against intake hashes.
        import hashlib
        if hashlib.sha256(target_bytes).hexdigest() != report["input_sha256"]["targets"]:
            raise ValidationError("Targets changed during scoring; retry with frozen inputs.")
        if hashlib.sha256(manifest_bytes).hexdigest() != report["input_sha256"]["manifest"]:
            raise ValidationError("Manifest changed during scoring; retry with frozen inputs.")
        paths = {t: p for t, _, p in _manifest(Path(manifest), manifest_bytes)}
        # Validate common condition/reference inputs even when every team is invalid.
        contest_score(reference, targets, groups, reference, max_points=max_points, version=version)
        for team in report["teams"]:
            if team["status"] == "valid":
                content = paths[team["team_id"]].read_bytes()
                if hashlib.sha256(content).hexdigest() != team["submission_sha256"]:
                    raise ValidationError("Submission changed during scoring; retry with frozen inputs.")
                team["contest"] = contest_score(read_submission(paths[team["team_id"]], content=content), targets, groups, reference, max_points=max_points, version=version)
        report["contest_version"] = version
        report["reference_sha256"] = hashlib.sha256(reference_bytes).hexdigest()
        report["conditions_sha256"] = hashlib.sha256(condition_bytes).hexdigest()
        report["scorer_sha256"]["contest.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        rows = sorted(({"team_id": t["team_id"], "technical_points": t["contest"]["technical_points"]}
                       for t in report["teams"] if t["status"] == "valid"), key=lambda r: (-r["technical_points"], r["team_id"]))
        write_rows(ready / "technical.csv", ("team_id", "technical_points"), rows)
        # leaderboard.csv is explicitly legacy raw error, not the event award score.
        (ready / "leaderboard.csv").rename(ready / "private/legacy_leaderboard.csv")
        write_json(ready / "private/run.json", report)
        _publish_ready(ready, output)
    return report
