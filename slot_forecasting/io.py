"""CSV/JSON files are data only; submission code is never imported by the scorer."""

import csv
import io
import json
from pathlib import Path

from .schema import (
    HISTORY_COLUMNS, LABEL_COLUMNS, PROBABILITY_COLUMNS, QUERY_COLUMNS,
    SUBMISSION_COLUMNS, TARGET_COLUMNS, Play, Query, Target, ValidationError,
    identifier, integer, payout, probabilities, unique_queries, index_history,
)


def read_rows(path, columns, *, allow_empty=False, content=None):
    try:
        stream = (io.StringIO(content.decode("utf-8-sig"), newline="") if content is not None
                  else Path(path).open("r", encoding="utf-8-sig", newline=""))
        with stream:
            reader = csv.reader(stream, strict=True)
            header = next(reader, None)
            if header is None or len(set(header)) != len(header) or set(header) != set(columns):
                raise ValidationError(f"{path}: expected exactly these unique columns: {', '.join(columns)}.")
            rows = []
            for line, row in enumerate(reader, 2):
                if len(row) != len(header) or any(not value.strip() for value in row):
                    raise ValidationError(f"{path}, row {line}: missing/extra fields or an empty value.")
                rows.append(dict(zip(header, row)))
    except (csv.Error, UnicodeError) as exc:
        raise ValidationError(f"{path}: invalid UTF-8 CSV: {exc}.") from exc
    if not rows and not allow_empty:
        raise ValidationError(f"{path}: no data rows.")
    return rows


def write_rows(path, columns, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_queries(path, *, content=None):
    queries = tuple(Query(
        r["query_id"], r["episode_id"], r["machine_id"], r["machine_type"],
        integer(r["signal"], "signal", -1), integer(r["cutoff"], "cutoff"),
    ) for r in read_rows(path, QUERY_COLUMNS, content=content))
    unique_queries(queries)
    return queries


def read_history(path):
    return tuple(Play(r["episode_id"], r["machine_id"], integer(r["tick"], "tick"), payout(r["payout"]))
                 for r in read_rows(path, HISTORY_COLUMNS, allow_empty=True))


def read_labels(path, queries):
    expected = unique_queries(queries)
    labels = {}
    for row in read_rows(path, LABEL_COLUMNS):
        key = identifier(row["query_id"], "query_id")
        if key in labels:
            raise ValidationError(f"Duplicate label query_id: {key}.")
        labels[key] = payout(row["outcome"])
    require_ids(labels, expected)
    return labels


def require_ids(actual, expected):
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra:
        raise ValidationError(f"query_id mismatch: missing={missing[:5]}, unexpected={extra[:5]}.")


def read_submission(path, expected_ids=None, *, content=None):
    predictions = {}
    for row in read_rows(path, SUBMISSION_COLUMNS, content=content):
        key = identifier(row["query_id"], "query_id")
        if key in predictions:
            raise ValidationError(f"Duplicate submission query_id: {key}.")
        predictions[key] = probabilities([row[c] for c in PROBABILITY_COLUMNS], f"row {key}")
    if expected_ids is not None:
        require_ids(predictions, expected_ids)
    return predictions


def read_targets(path, *, content=None):
    targets = tuple(Target(r["query_id"], r["episode_id"],
                           tuple(r[c] for c in PROBABILITY_COLUMNS), payout(r["outcome"]))
                    for r in read_rows(path, TARGET_COLUMNS, content=content))
    if len({t.query_id for t in targets}) != len(targets):
        raise ValidationError("Duplicate target query_id.")
    return targets


def write_submission(path, predictions):
    if not predictions:
        raise ValidationError("Cannot write an empty submission.")
    rows = []
    for key in sorted(predictions):
        identifier(key, "query_id")
        values = probabilities(predictions[key], key)
        rows.append({"query_id": key, **{c: format(p, ".17g") for c, p in zip(PROBABILITY_COLUMNS, values)}})
    write_rows(path, SUBMISSION_COLUMNS, rows)


def load_inputs(directory):
    directory = Path(directory)
    queries = read_queries(directory / "queries.csv")
    history = read_history(directory / "history.csv")
    index_history(queries, history)
    return queries, history


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
