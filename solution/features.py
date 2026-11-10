"""Deterministic history-only features shared by training and inference."""
from bisect import bisect_left
from collections import defaultdict
from slot_forecasting.schema import PAYOUTS

class HistoryIndex:
    def __init__(self, history):
        rows = defaultdict(dict)
        for p in history:
            key = (p.episode_id, p.machine_id)
            if p.tick in rows[key]:
                raise ValueError("Duplicate history tick")
            rows[key][p.tick] = p.payout
        self.rows = {k: sorted(v.items()) for k, v in rows.items()}
        self.ticks = {k: [t for t, _ in v] for k, v in self.rows.items()}
        self.episodes = defaultdict(list)
        for e, m in sorted(rows):
            self.episodes[e].append(m)

    def before(self, episode, machine, cutoff):
        key = (episode, machine)
        n = bisect_left(self.ticks.get(key, []), cutoff)
        return self.rows.get(key, [])[:n]

    def features(self, query, cutoff=None):
        t = query.cutoff if cutoff is None else cutoff
        if t < 0:
            raise ValueError("Negative cutoff")
        rows = self.before(query.episode_id, query.machine_id, t)
        seq = dict(rows)
        f = {}
        def summary(name, values, weights=None):
            counts = [0.] * 5
            for (_, y), w in zip(values, weights if weights is not None else [1.] * len(values)):
                counts[PAYOUTS.index(y)] += w
            n = sum(counts)
            f[name + '_n'] = n
            for y, c in zip(PAYOUTS, counts):
                f[name + '_p' + str(y)] = (c + .5) / (n + 2.5)
            f[name + '_mean'] = sum(y*c for y,c in zip(PAYOUTS,counts)) / max(n, 1)
            f[name + '_win'] = (n-counts[0]) / max(n, 1)
        summary('all', rows)
        f['missing_fraction'] = 1-len(rows)/max(t,1)
        gaps = [b[0]-a[0]-1 for a,b in zip(rows,rows[1:])]
        f['gap_max'] = max(gaps,default=0)
        f['gap_mean'] = sum(gaps)/max(len(gaps),1)
        f['last_age'] = t-rows[-1][0] if rows else t+1
        for w in (8,16,32,64,128):
            summary('tick'+str(w), [(s,y) for s,y in rows if s >= t-w])
            summary('obs'+str(w), rows[-w:])
        for half in (8,24,64,128):
            summary('decay'+str(half), rows, [2**(-(t-1-s)/half) for s,_ in rows])
        for lag in range(1,9):
            value = seq.get(t-lag)
            f['lag'+str(lag)] = -1 if value is None else PAYOUTS.index(value)
            f['lag'+str(lag)+'_missing'] = int(value is None)
            matched = [(s,y) for s,y in rows if value is not None and seq.get(s-lag) == value]
            summary('conditional'+str(lag), matched)
        for a,b in ((1,2),(1,4),(2,4),(4,8)):
            va,vb = seq.get(t-a),seq.get(t-b)
            f[f'pair{a}_{b}'] = -1 if va is None or vb is None else PAYOUTS.index(va)*5+PAYOUTS.index(vb)
        for y in PAYOUTS:
            f['drift'+str(y)] = f['tick16_p'+str(y)]-f['tick128_p'+str(y)]
        others = [r for m in self.episodes.get(query.episode_id, []) if m != query.machine_id
                  for r in self.before(query.episode_id,m,t)]
        for w in (8,32,128):
            summary('peer'+str(w), [(s,y) for s,y in others if s >= t-w])
        for lag in range(1,9):
            summary('peer_lag'+str(lag), [(s,y) for s,y in others if s == t-lag])
        return f
