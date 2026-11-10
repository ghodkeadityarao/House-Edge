import copy
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import simulated_kit

from slot_forecasting.io import load_inputs, read_submission, write_submission
from slot_forecasting.schema import PAYOUTS, Play
from solution import fit_structural as fit, structural as st


class Data:
    root = None

    @classmethod
    def load(cls):
        if cls.root is None:
            cls.directory = tempfile.TemporaryDirectory()
            cls.root = simulated_kit.build(Path(cls.directory.name) / "kit", seed=3, per_condition=(8, 8, 2))
        return cls.root


def params():
    value = copy.deepcopy(st.DEFAULT_PARAMS)
    truth = simulated_kit.TRUTH
    value["pairs"] = truth["pairs"]
    value["size_lags"] = truth["size_lags"]
    value["size_given_pair"] = np.eye(3).tolist()
    value["kappa_lag"] = truth["shared_lag"]
    value["tables"] = {key: [list(row) for row in table] for key, table in simulated_kit.TABLES.items()}
    return value


def largest_difference(left, right):
    return max(abs(a - b) for key in left for a, b in zip(left[key], right[key]))


class Leakage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.queries, cls.history = load_inputs(Data.load() / "validation")
        cls.queries = tuple(q for q in cls.queries if q.episode_id.endswith(("000", "001", "002", "003")))
        keep = {q.machine_id for q in cls.queries}
        cls.history = tuple(p for p in cls.history if p.machine_id in keep)
        cls.params = params()
        cls.base = st.predict(cls.queries, cls.history, cls.params)

    def test_rows_at_or_after_cutoff_are_ignored(self):
        future = [Play(q.episode_id, q.machine_id, q.cutoff + step, 20) for q in self.queries for step in (0, 1, 7)]
        self.assertEqual(largest_difference(self.base, st.predict(self.queries, self.history + tuple(future), self.params)), 0)

    def test_the_last_earlier_tick_does_matter(self):
        """Proves the check above can fail: an in-range change must move forecasts."""
        target = self.queries[0]
        changed = tuple(replace(p, payout=20 if p.payout == 0 else 0)
                        if p.machine_id == target.machine_id and p.tick >= target.cutoff - 8 else p
                        for p in self.history)
        self.assertGreater(largest_difference(self.base, st.predict(self.queries, changed, self.params)), 1e-3)

    def test_other_episodes_have_no_influence(self):
        episode = self.queries[0].episode_id
        changed = tuple(replace(p, payout=0) if p.episode_id != episode else p for p in self.history)
        forecast = st.predict(self.queries, changed, self.params)
        for q in self.queries:
            if q.episode_id == episode:
                self.assertEqual(forecast[q.query_id], self.base[q.query_id])

    def test_same_tick_neighbours_are_unavailable(self):
        first = self.queries[0]
        extra = [Play(q.episode_id, q.machine_id, q.cutoff, 20) for q in self.queries
                 if q.episode_id == first.episode_id and q.machine_id != first.machine_id]
        forecast = st.predict(self.queries, self.history + tuple(extra), self.params)
        self.assertEqual(forecast[first.query_id], self.base[first.query_id])

    def test_earlier_cutoff_matches_truncated_history(self):
        moved = tuple(replace(q, cutoff=q.cutoff - 30) for q in self.queries)
        cut = {q.machine_id: q.cutoff for q in moved}
        truncated = tuple(p for p in self.history if p.tick < cut[p.machine_id])
        self.assertEqual(largest_difference(st.predict(moved, self.history, self.params),
                                            st.predict(moved, truncated, self.params)), 0)


class Missingness(unittest.TestCase):
    def test_missing_ticks_are_not_zero_payouts(self):
        queries, history = load_inputs(Data.load() / "validation")
        first = queries[0]
        one = tuple(q for q in queries if q.episode_id == first.episode_id)
        rows = tuple(p for p in history if p.episode_id == first.episode_id)
        ticks = {first.cutoff - lag for lag in range(1, 9)}
        dropped = tuple(p for p in rows if not (p.machine_id == first.machine_id and p.tick in ticks))
        zeroed = tuple(replace(p, payout=0) if p.machine_id == first.machine_id and p.tick in ticks else p for p in rows)
        value = params()
        episode = st.group_episodes(one, dropped)[0]
        self.assertTrue(episode.short)
        self.assertEqual(int(episode.Y[0, first.cutoff - 3]), -1)
        ok, _, _ = st.pair_terms(episode, 0, (1, 6))
        self.assertFalse(ok[first.cutoff - 3])
        self.assertGreater(largest_difference(st.predict(one, dropped, value), st.predict(one, zeroed, value)), 1e-6)

    def test_short_and_long_histories_both_forecast(self):
        queries, history = load_inputs(Data.load() / "validation")
        episodes = st.group_episodes(queries, history)
        self.assertTrue(any(e.short for e in episodes))
        self.assertTrue(any(not e.short for e in episodes))
        forecast = st.predict(queries, history, params())
        self.assertEqual(set(forecast), {q.query_id for q in queries})


class Output(unittest.TestCase):
    def test_probabilities_are_valid_and_reload_exactly(self):
        queries, history = load_inputs(Data.load() / "test")
        value = params()
        forecast = st.predict(queries, history, value)
        for row in forecast.values():
            self.assertEqual(len(row), len(PAYOUTS))
            self.assertTrue(all(np.isfinite(v) and 0 <= v <= 1 for v in row))
            self.assertAlmostEqual(sum(row), 1, places=12)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "params.json"
            st.save_params(path, value)
            self.assertEqual(largest_difference(forecast, st.predict(queries, history, st.load_params(path))), 0)
            csv = Path(folder) / "forecast.csv"
            write_submission(csv, forecast)
            self.assertLessEqual(largest_difference(forecast, read_submission(csv, set(forecast))), 1e-15)

    def test_inference_never_reads_condition_or_target_files(self):
        queries, history = load_inputs(Data.load() / "test")
        self.assertFalse((Data.load() / "test/conditions.csv").exists())
        self.assertTrue(st.predict(queries, history, params()))


class Inference(unittest.TestCase):
    def test_true_rules_beat_wrong_rules(self):
        """Proves inference responds to the rules: a wrong pool must lose score."""
        root = Data.load()
        queries, history = load_inputs(root / "validation")
        targets = {t.query_id: np.asarray(t.probabilities, dtype=float)
                   for t in fit.read_targets(root / "validation/targets.csv")}

        def loss(value):
            forecast = st.predict(queries, history, value)
            return np.mean([((np.asarray(forecast[k]) - targets[k]) ** 2).sum() for k in targets])

        wrong = params()
        wrong["pairs"] = [[1, 3], [2, 5], [4, 7]]
        self.assertLess(loss(params()), 0.1 * loss(wrong))

    def test_inferred_condition_matches_simulated_condition(self):
        root = Data.load()
        queries, history = load_inputs(root / "validation")
        labels = fit.condition_map(root / "validation/conditions.csv")
        names = {"stable": 0, "shared_change": 1, "drift": 2}
        value = params()
        value["drift_sd"] = 12.0
        hits, total = 0, 0
        for episode in st.group_episodes(queries, history):
            label = labels[episode.queries[0].query_id]
            if label in names:
                total += 1
                hits += int(st.infer(episode, value)["conditions"].argmax()) == names[label]
        self.assertGreaterEqual(hits / total, 0.95)


class Splits(unittest.TestCase):
    def test_held_episode_never_informs_its_own_correction(self):
        """Corrupting one episode must leave that episode's coefficients unchanged."""
        rng = np.random.default_rng(0)
        groups = np.repeat(np.arange(24), 8)
        data = {"X": rng.normal(size=(192, 5)), "y": rng.normal(size=192), "group": groups,
                "use": np.ones(192, dtype=bool)}
        cached = [None] * 24
        clean = fit.cross_validated(data, cached, 1.0)
        for episode in (0, 7, 23):
            changed = dict(data, y=np.where(groups == episode, 1e6, data["y"]))
            result = fit.cross_validated(changed, cached, 1.0)
            self.assertTrue(np.array_equal(clean[episode], result[episode]))
            self.assertFalse(np.array_equal(clean[(episode + 1) % 24], result[(episode + 1) % 24])
                             and np.array_equal(clean, result))

    def test_kit_splits_share_no_episode(self):
        root = Data.load()
        seen = {}
        for split in ("train", "validation", "test"):
            queries, _ = load_inputs(root / split)
            seen[split] = {q.episode_id for q in queries}
        self.assertFalse(seen["train"] & seen["validation"] or seen["train"] & seen["test"]
                         or seen["validation"] & seen["test"])


class Discovery(unittest.TestCase):
    def test_fit_learns_rules_that_differ_from_main(self):
        root = Data.load()
        with tempfile.TemporaryDirectory() as folder:
            learned, report = fit.fit(root, Path(folder))
        truth = simulated_kit.TRUTH
        self.assertEqual(learned["pairs"], sorted(truth["pairs"]))
        self.assertEqual(learned["size_lags"], sorted(truth["size_lags"]))
        self.assertEqual(learned["kappa_lag"], truth["shared_lag"])
        self.assertEqual(learned["size_threshold"], truth["threshold"])
        given = np.asarray(learned["size_given_pair"])
        lags = [learned["size_lags"][int(k)] for k in given.argmax(axis=1)]
        expected = dict(zip(map(tuple, truth["pairs"]), truth["size_lags"]))
        self.assertEqual(lags, [expected[tuple(pair)] for pair in learned["pairs"]])
        for value in learned["w"].values():
            self.assertAlmostEqual(value, 2.3, delta=0.25)
        self.assertGreater(report["validation_cross_validated"]["technical_points"], 38)


if __name__ == "__main__":
    unittest.main()
