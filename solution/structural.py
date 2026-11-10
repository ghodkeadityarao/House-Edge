"""Bayesian inference over the discovered simulator structure.

Win probability at tick t for machine i:
    logit = base[type] + gamma*signal + kappa*(M[t-lag] - m0) + sign(t)*w[type]*e(t) + nu[type]*r(t)
    e(t)  = +1 if win[t-a] != win[t-b] else -1, with (a, b) a machine-specific lag pair
    r(t)  = +1 if payout table 1 is active at t, else -1
    M[t]  = mean win flag across the episode's machines at tick t
sign(t) is constant ("stable"), or flips once: at one tick shared by the episode
("shared"), or at machine-specific ticks around an episode centre ("drift").

Payout sizes given a win come from one of two tables per machine type. Table 1 is active
when (payout[t-L] >= threshold) != d. L is tied to the win pair through a learned
distribution; d is machine-specific. Wins, losses and payout sizes are all evidence for the
same joint hypothesis (pair, L, d, sign, flip).

Only history strictly before the cutoff is read, and a tick whose lags are unobserved
contributes no evidence. Condition labels and true probabilities are never inputs to
inference; conditions are inferred from the histories.
"""
import json
import math
from pathlib import Path

import numpy as np

from slot_forecasting.schema import PAYOUTS
from .features import HistoryIndex

SIZES = PAYOUTS[1:]
MIN_HYPOTHESIS_PRIOR = 0.005


def log_sigmoid(x):
    return -np.logaddexp(0.0, -x)


def sigmoid(x):
    return np.exp(log_sigmoid(x))


def logsum(values, axis=None):
    peak = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(peak, axis=axis) + np.log(np.sum(np.exp(values - peak), axis=axis))


DEFAULT_PARAMS = {
    "pairs": [[1, 3], [2, 5], [4, 7]],
    "pair_prior": [1 / 3, 1 / 3, 1 / 3],
    "size_lags": [1, 2, 4],
    "size_given_pair": [[0.96, 0.02, 0.02], [0.02, 0.96, 0.02], [0.02, 0.02, 0.96]],
    "size_direction_prior": [0.5, 0.5],
    "size_threshold": 2,
    "base": {"steady": 0.5, "balanced": -0.58, "jackpot": -1.78},
    "w": {"steady": 2.3, "balanced": 2.3, "jackpot": 2.3},
    "nu": {"steady": 0.0, "balanced": 0.0, "jackpot": 0.0},
    "gamma": 0.3,
    "kappa": 0.8,
    "kappa_lag": 3,
    "m0": 0.45,
    "condition_prior": {"stable": 0.49, "shared": 0.245, "drift": 0.245, "regime": 0.02},
    "condition_prior_short": {"stable": 0.49, "shared": 0.245, "drift": 0.245, "regime": 0.02},
    "regime_levels": [1.0, 0.5, 0.25, 0.0, -0.25, -0.5, -1.0],
    "regime_switch": 0.007,
    "tau_low": 0.45,
    "tau_high": 0.9,
    "drift_sd": 15.0,
    "tables": {
        "balanced": [[0.21406615, 0.54673659, 0.20502622, 0.03417104],
                     [0.49985311, 0.34792827, 0.1304731, 0.02174552]],
        "jackpot": [[0.12078673, 0.32392068, 0.41646944, 0.13882315],
                    [0.33514603, 0.2449462, 0.31493083, 0.10497694]],
        "steady": [[0.45512201, 0.43590239, 0.10679609, 0.00217951],
                   [0.75398939, 0.19680849, 0.04821808, 0.00098404]],
    },
}


class Episode:
    """Observed outcomes for one episode as a machines-by-ticks matrix; -1 means missing."""

    def __init__(self, queries, index):
        self.queries = sorted(queries, key=lambda q: q.machine_id)
        cutoffs = {q.cutoff for q in self.queries}
        if len(cutoffs) != 1:
            raise ValueError("An episode must share one cutoff")
        self.cutoff = cutoffs.pop()
        self.Y = np.full((len(self.queries), self.cutoff), -1, dtype=np.int64)
        for i, q in enumerate(self.queries):
            for tick, payout in index.before(q.episode_id, q.machine_id, q.cutoff):
                self.Y[i, tick] = payout
        self.observed = self.Y >= 0
        self.win = (self.Y > 0).astype(np.int8)
        count = self.observed.sum(axis=0)
        wins = (self.win * self.observed).sum(axis=0)
        self.M = np.divide(wins, count, out=np.full(self.cutoff, np.nan), where=count > 0)
        self.short = not self.observed.all()

    def mean_at(self, tick, default):
        if tick < 0 or tick >= self.cutoff or math.isnan(self.M[tick]):
            return default
        return float(self.M[tick])

    def mean_series(self, lag, default):
        """M[t-lag] for every t in 0..cutoff-1, with default where unavailable."""
        out = np.full(self.cutoff, float(default))
        known = self.M[:self.cutoff - lag]
        out[lag:] = np.where(np.isnan(known), default, known)
        return out


def machine_offsets(params, query):
    return params["base"][query.machine_type] + params["gamma"] * query.signal


def parity_table(k):
    """Truth table of XOR over k win flags, as +1/-1 indexed by sum(flag_j << j)."""
    return [1.0 if bin(index).count("1") % 2 else -1.0 for index in range(2 ** k)]


def rule_tables(params):
    tables = params.get("rule_tables")
    return tables if tables else [parity_table(len(lags)) for lags in params["pairs"]]


def pair_terms(episode, i, lags, table=None):
    """Availability mask and e(t) for all ticks, plus e at the cutoff (0 if unknown).

    e(t) looks up the machine's earlier win flags at the rule's lags in a +1/-1 truth table.
    """
    lags = list(lags)
    table = np.asarray(parity_table(len(lags)) if table is None else table, dtype=float)
    T = episode.cutoff
    ok = np.zeros(T, dtype=bool)
    e = np.zeros(T)
    start = max(lags)
    if T > start:
        ok[start:] = episode.observed[i, start:]
        code = np.zeros(T - start, dtype=int)
        for position, lag in enumerate(lags):
            ok[start:] &= episode.observed[i, start - lag:T - lag]
            code += episode.win[i, start - lag:T - lag].astype(int) << position
        e[start:] = table[code]
    final = 0.0
    if T >= start and all(episode.observed[i, T - lag] for lag in lags):
        final = float(table[sum(int(episode.win[i, T - lag]) << p for p, lag in enumerate(lags))])
    return ok, e, final


def regime_terms(episode, i, lag, threshold):
    """r(t) for direction d=0 at all ticks (0 where the lag is unobserved) and at the cutoff.

    Direction d=1 is the negation, because table 1 is active when flag != d.
    """
    T = episode.cutoff
    y = episode.Y[i]
    r = np.zeros(T)
    if T > lag:
        lagged = y[:T - lag]
        r[lag:] = np.where(lagged < 0, 0.0, np.where(lagged >= threshold, 1.0, -1.0))
    final = 0.0
    if T >= lag and y[T - lag] >= 0:
        final = 1.0 if y[T - lag] >= threshold else -1.0
    return r, final


def hypotheses(params):
    """Enumerate (pair index, lag index, direction) with non-negligible prior."""
    pair_prior = np.asarray(params["pair_prior"], dtype=float)
    given = np.asarray(params["size_given_pair"], dtype=float)
    direction = np.asarray(params["size_direction_prior"], dtype=float)
    rows, log_prior = [], []
    for j in range(len(pair_prior)):
        for k in range(given.shape[1]):
            if given[j, k] < MIN_HYPOTHESIS_PRIOR:
                continue
            for d in (0, 1):
                rows.append((j, k, d))
                log_prior.append(math.log(pair_prior[j] * given[j, k] * direction[d]))
    return rows, np.asarray(log_prior)


def size_log_likelihood(episode, i, query, lag, params):
    """Log likelihood of observed win sizes under direction 0 and direction 1."""
    y = episode.Y[i]
    wins = np.nonzero(y > 0)[0]
    tables = np.log(np.asarray(params["tables"][query.machine_type], dtype=float))
    index = {value: position for position, value in enumerate(SIZES)}
    codes = np.array([index[int(v)] for v in y[wins]], dtype=int)
    usable = np.zeros(len(wins), dtype=bool)
    usable[wins >= lag] = True
    usable[usable] = y[wins[usable] - lag] >= 0
    flag = np.zeros(len(wins), dtype=bool)
    flag[usable] = y[wins[usable] - lag] >= params["size_threshold"]
    # Ticks without an observed lag use the mean of the two tables for either direction.
    unknown = codes[~usable]
    mixed = float((np.logaddexp(tables[0, unknown], tables[1, unknown]) - math.log(2)).sum())
    result = []
    for d in (0, 1):
        active = flag[usable] != bool(d)
        result.append(float(tables[active.astype(int), codes[usable]].sum()) + mixed)
    return result


def infer(episode, params):
    """Exact posterior over rule, final sign, flip tick and condition for one episode."""
    T = episode.cutoff
    pairs = [tuple(p) for p in params["pairs"]]
    tables = rule_tables(params)
    lags = params["size_lags"]
    threshold = params["size_threshold"]
    rows, log_prior = hypotheses(params)
    n, h = len(episode.queries), len(rows)
    kappa, m0 = params["kappa"], params["m0"]
    lag = params["kappa_lag"]
    shared_level = kappa * (episode.mean_series(lag, m0) - m0)
    final_level = kappa * (episode.mean_at(T - lag, m0) - m0)

    low, high = int(params["tau_low"] * T), int(params["tau_high"] * T)
    taus = np.arange(low, max(high, low + 1))
    delta = (taus[:, None] - taus[None, :]) / params["drift_sd"]
    log_kernel = -0.5 * delta ** 2
    log_kernel -= logsum(log_kernel, axis=1)[:, None]

    levels = np.asarray(params.get("regime_levels", [1.0, -1.0]), dtype=float)
    g = len(levels)
    rho = params.get("regime_switch", 0.007)
    log_move = np.full((g, g), math.log(rho / (g - 1)))
    np.fill_diagonal(log_move, math.log(1 - rho))
    stable = np.zeros((n, h, 2))
    flip = np.zeros((n, h, 2, len(taus)))
    finals = np.zeros((n, h, 2 + g))
    regime_end = np.zeros((n, h, g))
    emit = np.zeros((n, h, g, T))
    active = np.zeros((n, h))
    win = episode.win.astype(float)
    for i, query in enumerate(episode.queries):
        base = machine_offsets(params, query)
        w, nu = params["w"][query.machine_type], params["nu"][query.machine_type]
        terms = [pair_terms(episode, i, pair, table) for pair, table in zip(pairs, tables)]
        regimes = [regime_terms(episode, i, value, threshold) for value in lags]
        sizes = [size_log_likelihood(episode, i, query, value, params) for value in lags]
        for index, (j, k, d) in enumerate(rows):
            ok, e, final = terms[j]
            r, r_final = regimes[k]
            orientation = -1.0 if d else 1.0
            fixed = base + shared_level + nu * orientation * r
            eta_final = base + final_level + nu * orientation * r_final
            per_sign = []
            for s, sign in enumerate((1.0, -1.0)):
                eta = fixed + sign * w * e
                ll = win[i] * log_sigmoid(eta) + (1 - win[i]) * log_sigmoid(-eta)
                per_sign.append(np.concatenate(([0.0], np.cumsum(np.where(ok, ll, 0.0)))))
                finals[i, index, s] = float(sigmoid(np.asarray(eta_final + sign * w * final)))
            evidence = sizes[k][d]
            eta_all = fixed[None, :] + levels[:, None] * w * e[None, :]
            emit[i, index] = np.where(ok[None, :], win[i] * log_sigmoid(eta_all)
                                      + (1 - win[i]) * log_sigmoid(-eta_all), 0.0)
            regime_end[i, index] = evidence
            finals[i, index, 2:] = sigmoid(eta_final + levels * w * final)
            for s in (0, 1):
                keep, before = per_sign[s], per_sign[1 - s]
                stable[i, index, s] = keep[T] + evidence
                flip[i, index, s] = before[taus] + keep[T] - keep[taus] + evidence
            active[i, index] = 0.5 * (1.0 + orientation * r_final)

    # Sticky-regime forward pass, run once for every machine and hypothesis together.
    alpha = emit[:, :, :, 0] - math.log(g)
    for t in range(1, T):
        alpha = emit[:, :, :, t] + logsum(alpha[:, :, :, None] + log_move, axis=2)
    regime_end += alpha

    prior_stable = stable + log_prior[None, :, None] + math.log(0.5)
    prior_flip = flip + log_prior[None, :, None, None] + math.log(0.5)
    z_stable = logsum(prior_stable.reshape(n, -1), axis=1)
    z_shared = logsum(prior_flip.reshape(n, -1, len(taus)), axis=1)
    tau_shared = z_shared.sum(axis=0)
    log_shared = logsum(tau_shared) - math.log(len(taus))

    smooth = np.zeros_like(prior_flip)
    for i in range(n):
        smooth[i] = logsum(prior_flip[i][:, :, None, :] + log_kernel[None, None], axis=3)
    z_drift = logsum(smooth.reshape(n, -1, len(taus)), axis=1)
    tau_drift = z_drift.sum(axis=0)
    log_drift = logsum(tau_drift) - math.log(len(taus))

    prior_regime = regime_end + log_prior[None, :, None]
    z_regime = logsum(prior_regime.reshape(n, -1), axis=1)
    cond = params["condition_prior_short"] if episode.short else params["condition_prior"]
    log_evidence = np.array([z_stable.sum(), log_shared, log_drift, z_regime.sum()])
    weights = [cond["stable"], cond["shared"], cond["drift"], cond.get("regime", 0.0)]
    log_cond = log_evidence + np.log(np.maximum(weights, 1e-300))
    p_cond = np.exp(log_cond - logsum(log_cond))

    def tilt(table, totals, posterior):
        """Mix machine posteriors over the episode-level flip-time posterior."""
        weights = np.exp(table - totals[:, None, None, :]).transpose(0, 3, 1, 2)
        return np.einsum("t,itjs->ijs", posterior, weights)

    post_stable = np.exp(prior_stable - z_stable[:, None, None])
    post_shared = tilt(prior_flip, z_shared, np.exp(tau_shared - logsum(tau_shared)))
    post_drift = tilt(smooth, z_drift, np.exp(tau_drift - logsum(tau_drift)))
    posterior = np.zeros((n, h, 2 + g))
    posterior[:, :, :2] = p_cond[0] * post_stable + p_cond[1] * post_shared + p_cond[2] * post_drift
    posterior[:, :, 2:] = p_cond[3] * np.exp(prior_regime - z_regime[:, None, None])
    pair_posterior = np.zeros((n, len(pairs)))
    for index, (j, _, _) in enumerate(rows):
        pair_posterior[:, j] += posterior[:, index].sum(axis=1)
    return {"p_win": (posterior * finals).sum(axis=(1, 2)), "conditions": p_cond,
            "posterior": posterior, "pair_posterior": pair_posterior, "finals": finals,
            "active": active, "rows": rows, "log_evidence": log_evidence,
            "table": (posterior.sum(axis=2) * active).sum(axis=1),
            "sign_values": np.concatenate(([1.0, -1.0], levels))}


def map_rule(win, i):
    """Most probable (pair index, lag index, direction, sign value) for machine i."""
    posterior = win["posterior"][i]
    index, s = np.unravel_index(int(posterior.argmax()), posterior.shape)
    j, k, d = win["rows"][index]
    return j, k, d, float(win["sign_values"][s])


def flip_scan(episode, i, win, params):
    """Return the MAP rule's signed e(t), r(t), usable ticks and likelihood flip scan."""
    j, k, d, sign = map_rule(win, i)
    ok, e, _ = pair_terms(episode, i, params["pairs"][j], rule_tables(params)[j])
    r, _ = regime_terms(episode, i, params["size_lags"][k], params["size_threshold"])
    r = -r if d else r
    level = params["kappa"] * (episode.mean_series(params["kappa_lag"], params["m0"]) - params["m0"])
    machine_type = episode.queries[i].machine_type
    w = params["w"][machine_type]
    base = machine_offsets(params, episode.queries[i]) + level + params["nu"][machine_type] * r
    flag = episode.win[i].astype(float)

    def cumulative(direction):
        value = base + direction * w * e
        ll = flag * log_sigmoid(value) + (1 - flag) * log_sigmoid(-value)
        return np.concatenate(([0.0], np.cumsum(np.where(ok, ll, 0.0))))

    keep, other = cumulative(sign), cumulative(-sign)
    return e, r, ok, base, sign, other + keep[-1] - keep, keep[-1]


def regime_path(base, e, ok, flag, w, levels, rho):
    """Viterbi path of the sticky rule-strength level, using only the given (pre-cutoff) ticks."""
    g, T = len(levels), len(e)
    move = np.full((g, g), math.log(rho / (g - 1)))
    np.fill_diagonal(move, math.log(1 - rho))
    value = base[None, :] + levels[:, None] * w * e[None, :]
    emit = np.where(ok[None, :], flag * log_sigmoid(value) + (1 - flag) * log_sigmoid(-value), 0.0)
    score = emit[:, 0] - math.log(g)
    back = np.zeros((T, g), dtype=int)
    for t in range(1, T):
        total = score[:, None] + move
        back[t] = total.argmax(axis=0)
        score = total.max(axis=0) + emit[:, t]
    path = np.zeros(T, dtype=int)
    path[-1] = int(score.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def map_paths(episode, win, params):
    """Most probable sign path per machine: logits, usable ticks, signed e(t), and r(t)."""
    T = episode.cutoff
    n = len(episode.queries)
    eta, signed, regime = np.zeros((n, T)), np.zeros((n, T)), np.zeros((n, T))
    usable = np.zeros((n, T), dtype=bool)
    ticks = np.arange(T)
    for i, query in enumerate(episode.queries):
        e, r, ok, base, sign, candidate, keep = flip_scan(episode, i, win, params)
        if win["conditions"][3] >= 0.5:
            # The catch-all regime dominates: follow its most likely level path, so repeated or
            # partial flips are not mistaken for a weaker rule.
            levels = np.asarray(params["regime_levels"], dtype=float)
            path = regime_path(base, e, ok, episode.win[i].astype(float),
                               params["w"][query.machine_type], levels, params["regime_switch"])
            signed[i] = levels[path] * e
            eta[i] = base + params["w"][query.machine_type] * signed[i]
            regime[i] = r
            usable[i] = ok
            continue
        tau = int(candidate.argmax())
        if win["conditions"][0] >= 0.5 or candidate[tau] < keep + 3.0:
            tau = 0
        signed[i] = np.where(ticks < tau, -sign, sign) * e
        eta[i] = base + params["w"][query.machine_type] * signed[i]
        regime[i] = r
        usable[i] = ok
    return eta, usable, signed, regime


STACK_WINDOWS = (8, 16, 32, 64)
STACK_NAMES = (["steady", "balanced", "jackpot", "signal", "short", "own_all"]
               + ["own%d" % k for k in STACK_WINDOWS]
               + ["other%d" % k for k in STACK_WINDOWS]
               + ["mean8", "mean16", "mean32"])


def stack_features(episode, win, params):
    """Residual summaries, computed from pre-cutoff outcomes under the fitted paths."""
    eta, usable, _, _ = map_paths(episode, win, params)
    p = sigmoid(eta)
    residual = (episode.win - p) * usable
    variance = p * (1 - p) * usable
    T, n = episode.cutoff, len(episode.queries)
    rows = np.zeros((n, len(STACK_NAMES)))

    def recent_mean(k):
        values = episode.M[max(0, T - k):T]
        values = values[~np.isnan(values)]
        return float(values.mean()) - params["m0"] if len(values) else 0.0

    means = [recent_mean(8), recent_mean(16), recent_mean(32)]
    for i, query in enumerate(episode.queries):
        others = np.arange(n) != i
        row = [float(query.machine_type == name) for name in ("steady", "balanced", "jackpot")]
        row += [float(query.signal), float(episode.short), residual[i].sum() / (variance[i].sum() + 1)]
        for k in STACK_WINDOWS:
            low = max(0, T - k)
            row.append(residual[i, low:T].sum() / (variance[i, low:T].sum() + 1))
        for k in STACK_WINDOWS:
            low = max(0, T - k)
            row.append(residual[others, low:T].sum() / (variance[others, low:T].sum() + 1))
        rows[i] = row + means
    return rows


def corrected_win(win, features, coefficients):
    """Apply the fitted logit correction to every hypothesis, then re-mix."""
    delta = features @ np.asarray(coefficients, dtype=float)
    finals = np.clip(win["finals"], 1e-9, 1 - 1e-9)
    logits = np.log(finals / (1 - finals)) + delta[:, None, None]
    return (win["posterior"] * sigmoid(logits)).sum(axis=(1, 2))


def probabilities(query, p_win, table, params):
    tables = np.asarray(params["tables"][query.machine_type], dtype=float)
    sizes = (1 - table) * tables[0] + table * tables[1]
    p = float(np.clip(p_win, 1e-6, 1 - 1e-6))
    values = [1 - p] + [p * float(v) for v in sizes]
    total = math.fsum(values)
    return tuple(v / total for v in values)


def predict_episode(episode, params):
    win = infer(episode, params)
    p_win = win["p_win"]
    if params.get("stack"):
        p_win = corrected_win(win, stack_features(episode, win, params), params["stack"]["coef"])
    return {query.query_id: probabilities(query, p_win[i], win["table"][i], params)
            for i, query in enumerate(episode.queries)}


def group_episodes(queries, history):
    index = HistoryIndex(history)
    groups = {}
    for q in queries:
        groups.setdefault(q.episode_id, []).append(q)
    return [Episode(groups[key], index) for key in sorted(groups)]


def predict(queries, history, params):
    output = {}
    for episode in group_episodes(queries, history):
        output.update(predict_episode(episode, params))
    return output


def save_params(path, params):
    Path(path).write_text(json.dumps(params, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def load_params(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
