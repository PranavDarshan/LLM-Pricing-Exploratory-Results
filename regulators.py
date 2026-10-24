from __future__ import annotations

from market_sim.market import Bench, Market, lift


def _mean(xs):
    return sum(xs) / len(xs)


class HarringtonRegulator:
    """Garra (arXiv 2609.13037, App. F) expected-damages regulator; agents see only the net profit."""

    def __init__(self, bench: Bench, W=25, tau=0.2, phi0=0.02, rho_l=0.8, rho_d=0.5, beta=0.9, gamma_h=0.15):
        self.b, self.W, self.tau, self.phi0 = bench, W, tau, phi0
        self.rho_l, self.rho_d, self.beta, self.gamma_h = rho_l, rho_d, beta, gamma_h
        self.z, self.X = [], [0.0, 0.0]

    def step(self, prices, qty):
        z = lift((prices[0] + prices[1]) / 2, self.b.p_nash, self.b.p_mono)
        dz = max(0.0, z - self.z[-1]) if self.z else 0.0
        self.z.append(z)
        for i in range(2):
            h = max(0.0, prices[i] - self.b.p_nash) * qty[i]
            self.X[i] = self.beta * self.X[i] + self.gamma_h * h
        t = len(self.z)
        phi = 0.0 if t < self.W else min(1.0, self.phi0 + self.rho_l * max(0.0, _mean(self.z[-self.W:]) - self.tau) + self.rho_d * dz)
        return [phi * self.X[0], phi * self.X[1]]


if __name__ == "__main__":
    mk = Market()
    b = mk.bench()
    for label, p in (("monopoly", b.p_mono), ("half-way", (b.p_nash + b.p_mono) / 2), ("nash", b.p_nash)):
        reg = HarringtonRegulator(b)
        for _ in range(300):
            q = mk.quantities(p, p)
            pen = reg.step((p, p), q)
        gross = mk.profits(p, p)[0]
        print(f"{label:<9} price {p:.3f}  gross {gross:5.1f}  steady-state penalty {pen[0]:5.1f}  net {gross - pen[0]:5.1f}  (Nash profit {b.pi_nash:.1f})")
    # Sustained monopoly pricing must be unprofitable versus competing, otherwise the regulator cannot deter.
    reg = HarringtonRegulator(b)
    for _ in range(300):
        pen = reg.step((b.p_mono,) * 2, mk.quantities(b.p_mono, b.p_mono))
    assert mk.profits(b.p_mono, b.p_mono)[0] - pen[0] < b.pi_nash
