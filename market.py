from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

HI = 10.0


def _gss_max(f, lo, hi, iters=80):
    g = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return (a + b) / 2


@dataclass(frozen=True)
class Bench:
    p_nash: float
    p_mono: float
    pi_nash: float
    pi_mono: float


def lift(x: float, lo: float, hi: float) -> float:
    return (x - lo) / (hi - lo)


@dataclass(frozen=True)
class Market:
    """Logit-demand Bertrand duopoly with Fish et al. / Calvano et al. defaults (alpha fixed to 1)."""

    a: float = 2.0
    a0: float = 0.0
    mu: float = 0.25
    cost: float = 1.0
    beta: float = 100.0
    max_price: float = 4.51

    def quantities(self, p1: float, p2: float, theta: float = 0.0, s: float | None = None):
        """theta = share of demand held by one big buyer; that buyer buys the logit 'inside' volume but splits it s : 1-s between the firms."""
        e1 = math.exp((self.a - p1) / self.mu)
        e2 = math.exp((self.a - p2) / self.mu)
        den = e1 + e2 + math.exp(self.a0 / self.mu)
        q1, q2 = self.beta * e1 / den, self.beta * e2 / den
        if theta and s is not None:
            vol = self.beta * (e1 + e2) / den
            q1, q2 = (1 - theta) * q1 + theta * vol * s, (1 - theta) * q2 + theta * vol * (1 - s)
        return q1, q2

    def buyer_volume(self, p1: float, p2: float, theta: float) -> float:
        e1, e2 = math.exp((self.a - p1) / self.mu), math.exp((self.a - p2) / self.mu)
        return theta * self.beta * (e1 + e2) / (e1 + e2 + math.exp(self.a0 / self.mu))

    def logit_split(self, p1: float, p2: float) -> float:
        e1, e2 = math.exp((self.a - p1) / self.mu), math.exp((self.a - p2) / self.mu)
        return e1 / (e1 + e2)

    def profits(self, p1: float, p2: float, theta: float = 0.0, s: float | None = None):
        q1, q2 = self.quantities(p1, p2, theta, s)
        return (p1 - self.cost) * q1, (p2 - self.cost) * q2

    def consumer_surplus(self, p1: float, p2: float) -> float:
        """Logit (log-sum-exp) consumer surplus per period; with alpha=1 it is in the same units as profit."""
        return self.beta * self.mu * math.log(math.exp((self.a - p1) / self.mu) + math.exp((self.a - p2) / self.mu) + math.exp(self.a0 / self.mu))

    def _br(self, rival: float) -> float:
        return _gss_max(lambda p: self.profits(p, rival)[0], self.cost, HI)

    @lru_cache(maxsize=None)
    def _br_cached(self, rival_rounded: float) -> float:
        return self._br(rival_rounded)

    def best_response(self, rival: float) -> float:
        return self._br_cached(round(rival, 3))

    @lru_cache(maxsize=None)
    def bench(self) -> Bench:
        p = 2.0
        for _ in range(500):
            nxt = self._br(p)
            if abs(nxt - p) < 1e-10:
                break
            p = nxt
        pm = _gss_max(lambda x: self.profits(x, x)[0], self.cost, HI)
        return Bench(p, pm, self.profits(p, p)[0], self.profits(pm, pm)[0])


if __name__ == "__main__":
    b = Market().bench()
    print(b)
    # Reference values from Fish et al. (A.5) / Calvano et al.: Nash 1.47, monopoly 1.92 (1.925), profits 22.29 / 33.75.
    assert abs(b.p_nash - 1.473) < 2e-3 and abs(b.p_mono - 1.925) < 2e-3
    assert abs(b.pi_nash - 22.29) < 0.1 and abs(b.pi_mono - 33.75) < 0.1
    print("benchmarks match published values")
