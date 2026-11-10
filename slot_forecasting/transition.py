"""An editable participant baseline: local conditional counts with optional decay."""
from .baseline import FrequencyForecaster
from .schema import PAYOUTS, ValidationError, finite_number, index_history, integer


def category(payout):
    return 0 if payout == 0 else 1 if payout == 1 else 2


def context(sequence, tick, lags):
    """Missing observations stay unknown. No current/future observations are read."""
    values = [sequence.get(tick - lag) for lag in lags]
    return None if any(v is None for v in values) else tuple(category(v) for v in values)


class TransitionForecaster:
    def __init__(self, lags=(1,), half_life=128, strength=8):
        self.lags = tuple(integer(lag, "lag", 1) for lag in lags)
        self.half_life = finite_number(half_life, "half_life")
        self.strength = finite_number(strength, "strength")
        if not self.lags or len(self.lags) > 3 or len(set(self.lags)) != len(self.lags):
            raise ValidationError("Use one to three different positive lags.")
        if self.half_life <= 0 or self.strength <= 0:
            raise ValidationError("half_life and strength must be positive.")
        self.base = FrequencyForecaster("pooled")

    def fit(self, queries, history, labels=None):
        self.base.fit(queries, history, labels)
        return self

    def predict(self, queries, history):
        queries, history = tuple(queries), tuple(history)
        indexed = index_history(queries, history)
        fallback = self.base.predict(queries, history)
        result = {}
        for query in queries:
            rows = indexed.get((query.episode_id, query.machine_id), [])
            sequence = {row.tick: row.payout for row in rows}
            wanted = context(sequence, query.cutoff, self.lags)
            counts = [self.strength * p for p in fallback[query.query_id]]
            if wanted is not None:
                for row in rows:
                    if context(sequence, row.tick, self.lags) == wanted:
                        age = query.cutoff - 1 - row.tick
                        counts[PAYOUTS.index(row.payout)] += 2 ** (-age / self.half_life)
            result[query.query_id] = tuple(v / sum(counts) for v in counts)
        return result
