"""Small public experiment loop. Select on tuning episodes, then audit once."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from time import perf_counter

from .baseline import FrequencyForecaster
from .contest import contest_score
from .io import load_inputs, read_labels, read_rows, read_submission, read_targets, require_ids, write_json, write_submission
from .rounds import VERSION, round_metadata
from .schema import CONDITIONS, ValidationError
from .transition import TransitionForecaster


def starter_candidates():
    """Edit this small list, or pass your own (name, model factory) pairs."""
    candidates = [(name, lambda name=name: FrequencyForecaster(name))
                  for name in ("pooled", "recent", "pooled_recent")]
    for lag in (1, 2, 4, 8):
        for half in (24, 128):
            candidates.append((f"lag_{lag}_half_{half}",
                               lambda lag=lag, half=half: TransitionForecaster((lag,), half)))
    candidates.append(("pair_1_2_half_24", lambda: TransitionForecaster((1, 2), 24)))
    return candidates


def episode_folds(targets, groups):
    episodes = defaultdict(set)
    seen = {}
    for target in targets:
        condition = groups[target.query_id]
        if condition not in CONDITIONS:
            raise ValidationError("Unknown condition in experiment data.")
        if target.episode_id in seen and seen[target.episode_id] != condition:
            raise ValidationError("An episode cannot belong to multiple conditions.")
        seen[target.episode_id] = condition
        episodes[condition].add(target.episode_id)
    tuning, audit = set(), set()
    for condition in CONDITIONS:
        ordered = sorted(episodes[condition], key=lambda e: hashlib.sha256(e.encode()).hexdigest())
        if len(ordered) < 2:
            raise ValidationError("Experiments need at least two validation episodes per condition.")
        tuning.update(ordered[::2])
        audit.update(ordered[1::2])
    return tuning, audit


def subset_score(predictions, targets, groups, reference, episodes, max_points):
    selected = [t for t in targets if t.episode_id in episodes]
    ids = {t.query_id for t in selected}
    return contest_score({k: predictions[k] for k in ids}, selected,
                         {k: groups[k] for k in ids}, {k: reference[k] for k in ids},
                         max_points=max_points, version=VERSION)


def select_model(train, validation, groups, reference, candidates, max_points):
    """The test split is deliberately absent from this API."""
    ids = {q.query_id for q in validation.queries}
    require_ids({t.query_id for t in validation.targets}, ids)
    require_ids(groups, ids)
    require_ids(reference, ids)
    tuning, audit = episode_folds(validation.targets, groups)
    names = [name for name, _ in candidates]
    if not names or len(set(names)) != len(names):
        raise ValidationError("Candidate names must be nonempty and unique.")
    rows, models, predictions = {}, {}, {}
    for name, factory in candidates:
        start = perf_counter()
        model = factory().fit(train.queries, train.history, train.labels)
        prediction = model.predict(validation.queries, validation.history)
        require_ids(prediction, ids)
        result = subset_score(prediction, validation.targets, groups, reference, tuning, max_points)
        rows[name] = {"tuning": result, "seconds": perf_counter() - start}
        models[name], predictions[name] = model, prediction
    chosen = min(names, key=lambda name: (rows[name]["tuning"]["mean_loss_ratio"], name))
    # Audit labels are first scored after the choice is fixed. Never select on them.
    audit_result = subset_score(predictions[chosen], validation.targets, groups, reference, audit, max_points)
    report = {"chosen": chosen, "comparison": rows, "audit": audit_result,
              "tuning_episodes": sorted(tuning), "audit_episodes": sorted(audit),
              "selection_rule": "Lowest tuning loss ratio; audit scored only after selection."}
    return models[chosen], predictions[chosen], report


def run_experiments(data, output, candidates=None):
    from types import SimpleNamespace
    public, output = Path(data).resolve(), Path(output)
    metadata = round_metadata(public)
    if output.exists():
        raise ValidationError("Choose a fresh experiments output directory.")
    train_q, train_h = load_inputs(public / "train")
    val_q, val_h = load_inputs(public / "validation")
    train = SimpleNamespace(queries=train_q, history=train_h, labels=read_labels(public / "train/labels.csv", train_q))
    validation = SimpleNamespace(queries=val_q, history=val_h, targets=read_targets(public / "validation/targets.csv"))
    group_rows = read_rows(public / "validation/conditions.csv", ("query_id", "condition"))
    groups = {r["query_id"]: r["condition"] for r in group_rows}
    if len(groups) != len(group_rows):
        raise ValidationError("Duplicate condition query ID.")
    reference = read_submission(public / "validation/reference.csv")
    model, prediction, report = select_model(train, validation, groups, reference,
                                           starter_candidates() if candidates is None else candidates, metadata["max_points"])
    # Load test inputs only after model selection. There are no test labels here.
    test_q, test_h = load_inputs(public / "test")
    test = model.predict(test_q, test_h)
    require_ids(test, {q.query_id for q in test_q})
    report.update(round=metadata["round"], season_id=metadata["season_id"], max_points=metadata["max_points"])
    files = [public / name for name in ("metadata.json", "train/queries.csv", "train/history.csv", "train/labels.csv",
                                        "validation/queries.csv", "validation/history.csv", "validation/targets.csv",
                                        "validation/conditions.csv", "validation/reference.csv", "test/queries.csv", "test/history.csv")]
    report["input_sha256"] = {str(p.relative_to(public)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    report["source_sha256"] = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                               for name in ("experiments.py", "transition.py", "baseline.py", "contest.py")}
    output.mkdir(parents=True)
    write_submission(output / "validation_predictions.csv", prediction)
    write_submission(output / "submission.csv", test)
    write_json(output / "experiments.json", report)
    return report
