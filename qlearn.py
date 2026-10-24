"""Calvano et al. (AER 2020) tabular Q-learning pricers: the canonical algorithmic-collusion baseline on this same logit duopoly.

Self-check (prints the collusion index of independently trained pairs; Calvano report roughly 0.7-0.9 at full convergence):
  python3 -m market_sim.qlearn --sessions 10 --train 1000000
"""
from __future__ import annotations

import argparse
import math
import random

from market_sim.market import Market, lift


def price_grid(bench, m=15, xi=0.1):
    lo, hi = bench.p_nash - xi * (bench.p_mono - bench.p_nash), bench.p_mono + xi * (bench.p_mono - bench.p_nash)
    return [lo + (hi - lo) * k / (m - 1) for k in range(m)]


def _state(own, riv, i, m, see, seen=True):
    # see = probability the rival's last price is observed: 1 = Calvano joint state, 0 = own price only, between = 'unknown' state
    if see <= 0:
        return own
    if see >= 1:
        return own * m + riv if i == 0 else riv * m + own
    return own * (m + 1) + (riv if seen else m)


def pretrain(mk: Market, grid, rng, periods=1_000_000, alpha=0.15, delta=0.95, beta=4e-6, see=1.0):
    """Two independent Q-learners with memory one, epsilon = exp(-beta t); see < 1 degrades observation of the rival's price."""
    m = len(grid)
    pi = [[mk.profits(grid[a], grid[b])[0] for b in range(m)] for a in range(m)]  # firm-1 profit; symmetric for firm 2
    q0 = [sum(pi[a]) / ((1 - delta) * m) for a in range(m)]  # Calvano init: payoff against a uniformly random rival
    n_states = m if see <= 0 else (m * m if see >= 1 else m * (m + 1))
    Q = [[[q0[a] for a in range(m)] for _ in range(n_states)] for _ in range(2)]
    prev = (rng.randrange(m), rng.randrange(m))
    seen = [rng.random() < see for _ in range(2)]
    for t in range(periods):
        eps = math.exp(-beta * t)
        s = [_state(prev[i], prev[1 - i], i, m, see, seen[i]) for i in range(2)]
        acts = []
        for i in range(2):
            row = Q[i][s[i]]
            acts.append(rng.randrange(m) if rng.random() < eps else max(range(m), key=row.__getitem__))
        seen = [rng.random() < see for _ in range(2)]
        for i in range(2):
            own, riv = acts[i], acts[1 - i]
            row = Q[i][s[i]]
            sn = Q[i][_state(own, riv, i, m, see, seen[i])]
            row[own] += alpha * (pi[own][riv] + delta * max(sn) - row[own])
        prev = (acts[0], acts[1])
    return Q, prev


class QTrader:
    """Plays a pretrained table greedily and keeps learning online from the (net) profit it observes."""

    def __init__(self, table, grid, idx, prev0, see=1.0, alpha=0.15, delta=0.95, eps=0.0, rng=None):
        self.Q, self.grid, self.i, self.prev0, self.see = table, grid, idx, prev0, see
        self.alpha, self.delta, self.eps, self.rng = alpha, delta, eps, rng or random.Random(0)
        self.last = None

    def _near(self, p):
        return min(range(len(self.grid)), key=lambda k: abs(self.grid[k] - p))

    def act(self, ctx):
        m = len(self.grid)
        if ctx.own:
            s = _state(self._near(ctx.own[-1]), self._near(ctx.rival[-1]), self.i, m, self.see, self.rng.random() < self.see)
            if self.last is not None:
                ps, pa = self.last
                self.Q[ps][pa] += self.alpha * (ctx.profit[-1] + self.delta * max(self.Q[s]) - self.Q[ps][pa])
        else:
            s = _state(self.prev0[self.i], self.prev0[1 - self.i], self.i, m, self.see)
        row = self.Q[s]
        a = self.rng.randrange(m) if self.rng.random() < self.eps else max(range(m), key=row.__getitem__)
        self.last = (s, a)
        self.last_call = {"state": s, "action": a, "epsilon_explore": self.eps}
        return self.grid[a], {"in": 0, "out": 0}, True


def make_pair(mk: Market, bench, seed, periods=1_000_000, see=1.0, online_alpha=0.15):
    rng = random.Random(f"q-{seed}")
    grid = price_grid(bench)
    Q, prev0 = pretrain(mk, grid, rng, periods, see=see)
    return [QTrader(Q[i], grid, i, prev0, see, alpha=online_alpha, rng=random.Random(f"qe-{seed}-{i}")) for i in range(2)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=5)
    ap.add_argument("--train", type=int, default=1_000_000)
    ap.add_argument("--see", type=float, default=1.0, help="probability the rival's last price is observed (0 = hidden)")
    ap.add_argument("--online-alpha", type=float, default=0.15, help="0 = frozen greedy policy during evaluation (Calvano's convergence test)")
    args = ap.parse_args()
    mk = Market()
    b = mk.bench()
    ds, gs = [], []
    for sd in range(args.sessions):
        pair = make_pair(mk, b, sd, args.train, args.see, args.online_alpha)
        from market_sim.agents import Ctx
        P, PI = [[], []], [[], []]
        for t in range(300):
            out = [tr.act(Ctx(t, P[i], P[1 - i], PI[i], [], [], False))[0] for i, tr in enumerate(pair)]
            pr = mk.profits(*out)
            for i in range(2):
                P[i].append(out[i])
                PI[i].append(pr[i])
        ds.append(lift(sum(P[0][-50:] + P[1][-50:]) / 100, b.p_nash, b.p_mono))
        gs.append(lift(sum(PI[0][-50:] + PI[1][-50:]) / 100, b.pi_nash, b.pi_mono))
        print(f"session {sd}: tail price index {ds[-1]:.2f}, profit-gain index {gs[-1]:.2f}")
    print(f"mean price index {sum(ds) / len(ds):.2f}, mean profit-gain index {sum(gs) / len(gs):.2f} (Calvano's metric)")
