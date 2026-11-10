"""Trainable frequency baselines with optional context pooling and recency weighting."""

from collections import defaultdict
import json
import math
from pathlib import Path

from .io import require_ids, write_json
from .schema import PAYOUTS, ValidationError, finite_number, index_history, payout, probabilities, unique_queries

METHODS = ("global", "context", "empirical", "recent", "pooled", "pooled_recent")


class FrequencyForecaster:
    def __init__(self, method="pooled", prior_strength=20.0, half_life=24.0):
        if method not in METHODS:
            raise ValidationError(f"Unknown baseline: {method}.")
        self.method = method
        self.prior_strength = finite_number(prior_strength, "prior_strength")
        self.half_life = finite_number(half_life, "half_life")
        if self.prior_strength <= 0 or self.half_life <= 0:
            raise ValidationError("prior_strength and half_life must be positive.")
        self.global_prior = None
        self.context_priors = {}

    @staticmethod
    def _key(query):
        return f"{query.machine_type}|{query.signal}"

    def fit(self, queries, history, labels=None):
        queries = tuple(queries)
        query_map = unique_queries(queries)
        indexed = index_history(queries, history)
        if labels is not None:
            require_ids(labels, query_map)
        counts = [0.0] * len(PAYOUTS)
        contexts = defaultdict(lambda: [0.0] * len(PAYOUTS))
        for query in sorted(query_map.values(), key=lambda q: q.query_id):
            outcomes = [p.payout for p in indexed.get((query.episode_id, query.machine_id), [])]
            if labels is not None:
                outcomes.append(payout(labels[query.query_id]))
            for outcome in outcomes:
                index = PAYOUTS.index(outcome)
                counts[index] += 1
                contexts[self._key(query)][index] += 1
        total = math.fsum(counts)
        if total == 0:
            raise ValidationError("Training needs at least one observed payout or label.")
        self.global_prior = tuple((value + 0.5) / (total + 2.5) for value in counts)
        self.context_priors = {
            key: tuple((value + 12.0 * prior) / (math.fsum(values) + 12.0)
                       for value, prior in zip(values, self.global_prior))
            for key, values in sorted(contexts.items())
        }
        return self

    def predict(self, queries, history):
        if self.global_prior is None:
            raise ValidationError("Fit or load the model before predicting.")
        queries = tuple(queries)
        query_map = unique_queries(queries)
        indexed = index_history(queries, history)
        predictions = {}
        for query in sorted(query_map.values(), key=lambda q: q.query_id):
            context = self.context_priors.get(self._key(query), self.global_prior)
            if self.method in ("global", "context"):
                predictions[query.query_id] = context if self.method == "context" else self.global_prior
                continue
            pooled = self.method in ("pooled", "pooled_recent")
            prior = context if pooled else self.global_prior
            strength = self.prior_strength if pooled else 1.0
            counts = [strength * value for value in prior]
            for play in indexed.get((query.episode_id, query.machine_id), []):
                age = query.cutoff - 1 - play.tick
                weight = 2 ** (-age / self.half_life) if self.method in ("recent", "pooled_recent") else 1.0
                counts[PAYOUTS.index(play.payout)] += weight
            total = math.fsum(counts)
            predictions[query.query_id] = tuple(value / total for value in counts)
        return predictions

    def save(self, path):
        if self.global_prior is None:
            raise ValidationError("Cannot save an unfitted model.")
        write_json(path, {"model_version": 1, "method": self.method,
                          "prior_strength": self.prior_strength, "half_life": self.half_life,
                          "global_prior": self.global_prior, "context_priors": self.context_priors})

    @classmethod
    def load(cls, path):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        expected = {"model_version", "method", "prior_strength", "half_life", "global_prior", "context_priors"}
        if (not isinstance(value, dict) or set(value) != expected
                or type(value["model_version"]) is not int or value["model_version"] != 1):
            raise ValidationError("Unsupported or malformed saved model.")
        model = cls(value["method"], value["prior_strength"], value["half_life"])
        model.global_prior = probabilities(value["global_prior"])
        if not isinstance(value["context_priors"], dict):
            raise ValidationError("context_priors must be an object.")
        model.context_priors = {key: probabilities(probs) for key, probs in value["context_priors"].items()}
        return model
