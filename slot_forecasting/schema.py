"""The versioned prediction contract and strict input validation."""

from dataclasses import dataclass
import math
import re
from collections import defaultdict

PAYOUTS = (0, 1, 2, 5, 20)
PROBABILITY_COLUMNS = tuple(f"p_{payout}" for payout in PAYOUTS)
SCORING_VERSION = "distribution-brier-v1"
CONDITIONS = ("stable", "short_history", "drift", "shared_change")
SUITE_VERSION = "four-conditions-v1"
SUM_TOLERANCE = 1e-6
QUERY_COLUMNS = ("query_id", "episode_id", "machine_id", "machine_type", "signal", "cutoff")
HISTORY_COLUMNS = ("episode_id", "machine_id", "tick", "payout")
LABEL_COLUMNS = ("query_id", "outcome")
SUBMISSION_COLUMNS = ("query_id", *PROBABILITY_COLUMNS)
TARGET_COLUMNS = ("query_id", "episode_id", *PROBABILITY_COLUMNS, "outcome")


class ValidationError(ValueError):
    """A malformed dataset, model, or submission; never a silently repaired input."""


def identifier(value, name="identifier"):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", value):
        raise ValidationError(f"Invalid {name}: expected 1-128 letters, digits, or ._:- (starting alphanumeric).")
    return value


def finite_number(value, name="number"):
    if isinstance(value, bool):
        raise ValidationError(f"{name} must be a finite number, not a boolean.")
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValidationError(f"{name} must be a finite number.") from exc
    if not math.isfinite(result):
        raise ValidationError(f"{name} must be finite.")
    return result


def integer(value, name="integer", minimum=0):
    if isinstance(value, bool) or not re.fullmatch(r"-?\d+", str(value)):
        raise ValidationError(f"{name} must be an integer.")
    result = int(value)
    if result < minimum:
        raise ValidationError(f"{name} must be at least {minimum}.")
    return result


def payout(value):
    number = finite_number(value, "payout")
    if number not in PAYOUTS:
        raise ValidationError(f"Payout must be one of {PAYOUTS}.")
    return int(number)


def probabilities(values, name="probabilities"):
    try:
        values = tuple(values)
    except TypeError as exc:
        raise ValidationError(f"{name} must contain five probabilities.") from exc
    if len(values) != len(PAYOUTS):
        raise ValidationError(f"{name} must contain exactly {len(PAYOUTS)} probabilities.")
    result = tuple(finite_number(v, name) for v in values)
    if any(v < 0 or v > 1 for v in result):
        raise ValidationError(f"{name} must be between 0 and 1 inclusive.")
    total = math.fsum(result)
    if abs(total - 1.0) > SUM_TOLERANCE:
        raise ValidationError(f"{name} must sum to 1 (tolerance {SUM_TOLERANCE}); got {total:.12g}.")
    # Only remove documented decimal-rounding error, never clip or repair invalid rows.
    return tuple(v / total for v in result)


@dataclass(frozen=True)
class Query:
    query_id: str
    episode_id: str
    machine_id: str
    machine_type: str
    signal: int
    cutoff: int

    def __post_init__(self):
        for name in ("query_id", "episode_id", "machine_id", "machine_type"):
            identifier(getattr(self, name), name)
        if integer(self.signal, "signal", -1) not in (-1, 0, 1):
            raise ValidationError("signal must be -1, 0, or 1.")
        object.__setattr__(self, "signal", integer(self.signal, "signal", -1))
        object.__setattr__(self, "cutoff", integer(self.cutoff, "cutoff", 0))


@dataclass(frozen=True)
class Play:
    episode_id: str
    machine_id: str
    tick: int
    payout: int

    def __post_init__(self):
        identifier(self.episode_id, "episode_id")
        identifier(self.machine_id, "machine_id")
        object.__setattr__(self, "tick", integer(self.tick, "tick", 0))
        object.__setattr__(self, "payout", payout(self.payout))


@dataclass(frozen=True)
class Target:
    query_id: str
    episode_id: str
    probabilities: tuple[float, ...]
    outcome: int

    def __post_init__(self):
        identifier(self.query_id, "query_id")
        identifier(self.episode_id, "episode_id")
        object.__setattr__(self, "probabilities", probabilities(self.probabilities, "target probabilities"))
        object.__setattr__(self, "outcome", payout(self.outcome))


def unique_queries(queries):
    result = {}
    keys = set()
    for query in queries:
        if query.query_id in result:
            raise ValidationError(f"Duplicate query_id: {query.query_id}.")
        key = (query.episode_id, query.machine_id)
        if key in keys:
            raise ValidationError("v1 permits one prediction cutoff per machine per episode.")
        result[query.query_id] = query
        keys.add(key)
    if not result:
        raise ValidationError("At least one query is required.")
    return result


def index_history(queries, history):
    """Reject exposed future records rather than silently relying on model filtering."""
    query_map = unique_queries(queries)
    by_machine = {(q.episode_id, q.machine_id): q for q in query_map.values()}
    indexed = defaultdict(list)
    seen = set()
    for play in history:
        key = (play.episode_id, play.machine_id)
        if key not in by_machine:
            raise ValidationError(f"History contains an unknown machine: {key}.")
        if play.tick >= by_machine[key].cutoff:
            raise ValidationError(f"Future data leak: tick {play.tick} is at/after cutoff for {key}.")
        record = (*key, play.tick)
        if record in seen:
            raise ValidationError(f"Duplicate history observation: {record}.")
        seen.add(record)
        indexed[key].append(play)
    return {key: sorted(plays, key=lambda p: p.tick) for key, plays in indexed.items()}
