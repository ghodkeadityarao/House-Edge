"""Reproducible participant workflow using only released public data."""

import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

from .baseline import FrequencyForecaster, METHODS
from .io import load_inputs, read_labels, read_submission, write_json, write_submission
from .schema import SCORING_VERSION, ValidationError, integer
from .scoring import score_files


def public_directory(directory):
    """Accept either public/ itself or the release folder containing it."""
    directory = Path(directory).expanduser().resolve()
    public = directory / "public" if (directory / "public").is_dir() else directory
    required = [f"{split}/{name}.csv" for split in ("train", "validation", "test")
                for name in ("queries", "history")]
    required.append("validation/targets.csv")
    missing = [name for name in required if not (public / name).is_file()]
    if missing:
        raise ValidationError(
            f"Incomplete public dataset at {public}: missing {', '.join(missing)}. "
            "Point --data or SLOTS_DATA_DIR at the released public folder or its parent. "
            "Generate demo data separately if needed.")
    return public


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_starter(data, out, *, method="pooled", bootstrap_repeats=500):
    """Compare six fixed baselines on validation, then export the chosen method.

    The default stays pooled; the comparison never silently switches methods.
    Test labels/targets and organizer files are never opened. Training is
    deterministic, and bootstrap sampling uses the scorer's fixed seed 314159.
    """
    started = perf_counter()
    if method not in METHODS:
        raise ValidationError(f"Unknown baseline: {method}.")
    bootstrap_repeats = integer(bootstrap_repeats, "bootstrap repeats")
    public = public_directory(data)
    out = Path(out).expanduser().resolve()
    if out == public or public in out.parents or out in public.parents:
        raise ValidationError("Choose an output folder separate from the public data folder.")

    train_queries, train_history = load_inputs(public / "train")
    labels_path = public / "train/labels.csv"
    labels = read_labels(labels_path, train_queries) if labels_path.exists() else None
    validation_queries, validation_history = load_inputs(public / "validation")
    test_queries, test_history = load_inputs(public / "test")
    load_seconds = perf_counter() - started

    # Hash only inputs we consume, never recursively inspect organizer/test targets.
    input_files = [public / split / f"{name}.csv"
                   for split in ("train", "validation", "test")
                   for name in ("queries", "history")]
    input_files.append(public / "validation/targets.csv")
    if labels is not None:
        input_files.append(labels_path)
    if (public / "metadata.json").is_file():
        input_files.append(public / "metadata.json")
    contest_inputs = [public / "validation/conditions.csv", public / "validation/reference.csv"]
    has_contest = any(path.exists() for path in contest_inputs)
    event_options = {}
    if (public / "metadata.json").is_file():
        metadata = json.loads((public / "metadata.json").read_text(encoding="utf-8"))
        has_contest |= metadata.get("suite_version") in ("four-conditions-v1", "temporal-suite-v2", "experimentation-suite-v3")
        if metadata.get("suite_version") in ("temporal-suite-v2", "experimentation-suite-v3"):
            from .rounds import round_metadata, VERSION
            event_options = {"max_points": round_metadata(public)["max_points"], "version": VERSION}
    if has_contest:
        if not all(path.is_file() for path in contest_inputs):
            raise ValidationError("Event validation requires both conditions.csv and reference.csv.")
        input_files.extend(contest_inputs)
    input_hashes = {str(path.relative_to(public)): _sha256(path) for path in input_files}

    comparisons = {}
    selected = None
    for candidate in METHODS:
        candidate_started = perf_counter()
        model = FrequencyForecaster(candidate).fit(train_queries, train_history, labels)
        predictions = model.predict(validation_queries, validation_history)
        fit_predict_seconds = perf_counter() - candidate_started
        csv_path = out / "comparison" / f"{candidate}.csv"
        write_submission(csv_path, predictions)
        # Score the actual serialized CSV through the same participant CLI scorer.
        report = score_files(csv_path, public / "validation/targets.csv",
                             public / "validation/queries.csv")
        comparisons[candidate] = {
            "score": report["score"], "probability_error": report["probability_error"],
            "expected_payout_mae": report["diagnostics"]["expected_payout_mae"],
            "fit_predict_seconds": fit_predict_seconds,
        }
        if has_contest:
            from .contest import score_contest_files
            comparisons[candidate]["contest"] = score_contest_files(
                csv_path, public / "validation/targets.csv", public / "validation/queries.csv",
                public / "validation/conditions.csv", public / "validation/reference.csv", **event_options)
        if candidate == method:
            selected = model

    validation_csv = out / "validation_predictions.csv"
    write_submission(validation_csv, selected.predict(validation_queries, validation_history))
    validation_report = score_files(
        validation_csv, public / "validation/targets.csv", public / "validation/queries.csv",
        bootstrap_repeats=bootstrap_repeats)
    if has_contest:
        validation_report["contest"] = comparisons[method]["contest"]
    write_json(out / "validation_score.json", validation_report)
    model_path = out / "baseline.json"
    selected.save(model_path)
    reloaded = FrequencyForecaster.load(model_path)
    write_submission(out / "submission.csv", reloaded.predict(test_queries, test_history))
    test_predictions = read_submission(out / "submission.csv", {q.query_id for q in test_queries})
    reproduced = reloaded.predict(validation_queries, validation_history)
    original = read_submission(validation_csv, {q.query_id for q in validation_queries})
    max_difference = max(abs(p - q) for key in original
                         for p, q in zip(original[key], reproduced[key]))
    # Loading/validating normalizes tiny floating-point summation differences.
    if max_difference > 1e-14:
        raise ValidationError("Saved model did not reproduce validation probabilities within 1e-14.")

    best = min(METHODS, key=lambda name: comparisons[name].get("contest", {}).get("mean_loss_ratio", comparisons[name]["probability_error"]))
    source_root = Path(__file__).resolve().parent
    report = {
        "scoring_version": SCORING_VERSION,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "machine": platform.machine(), "model_dependencies": []},
        "randomness": {"training": "deterministic; no model seed", "bootstrap_seed": 314159,
                       "bootstrap_repeats": bootstrap_repeats,
                       "data": "External released inputs identified by SHA-256; no private generation record read."},
        "data_sha256": input_hashes,
        "source_sha256": {path.name: _sha256(path) for path in sorted(source_root.glob("*.py"))},
        "counts": {"train_queries": len(train_queries), "train_history": len(train_history),
                   "validation_queries": len(validation_queries), "test_queries": len(test_queries)},
        "training_labels_used": labels is not None,
        "settings": {"method": method, "prior_strength": selected.prior_strength,
                     "half_life": selected.half_life},
        "selection": "Explicit method (default pooled); six fixed defaults compared on validation only.",
        "best_validation_method": best,
        "comparison": comparisons,
        "validation": validation_report,
        "checks": {"saved_model_reproduces_validation": True,
                   "saved_model_max_probability_difference": max_difference,
                   "test_submission_valid": len(test_predictions) == len(test_queries)},
        "runtime_seconds": {"load": load_seconds, "total": perf_counter() - started},
        "output_sha256": {name: _sha256(out / name) for name in (
            "baseline.json", "validation_predictions.csv", "validation_score.json", "submission.csv")},
    }
    write_json(out / "run.json", report)
    return report
