"""Supervised price-screen flagger, following the screens-plus-ML recipe of Huber/Imhof et al. (adapted to price series)."""
from __future__ import annotations

import argparse
import glob
import json
import math

from market_sim.metrics import auc, mean


def features(wp, cost: float = 1.0, with_level: bool = False):
    a, b = wp
    avg = [(x + y) / 2 for x, y in zip(a, b)]
    m = mean(avg)
    sd = math.sqrt(mean((v - m) ** 2 for v in avg))
    f = [
        mean(abs(x - y) for x, y in zip(a, b)) / m,  # parallelism: gap between firms
        sd / m,  # variance screen: coefficient of variation over time
        mean(abs(x - y) < 0.02 for x, y in zip(a, b)),  # share of near-identical prices
        mean(abs(avg[i] - avg[i - 1]) for i in range(1, len(avg))) / m,  # price volatility
    ]
    if with_level:
        f.append(m - cost)  # observable markup over known cost; label is level-based, so this is near-circular
    return f


def _sig(x):
    return 1 / (1 + math.exp(-max(-30.0, min(30.0, x))))


def train(rows, with_level=False, iters=3000, lr=0.3, l2=1e-3):
    X, y = [], []
    for r in rows:
        n = len(r["windows"])
        W = len(r["prices"][0]) // n
        for w, win in enumerate(r["windows"]):
            a, b = w * W, (w + 1) * W
            X.append(features([r["prices"][0][a:b], r["prices"][1][a:b]], with_level=with_level))
            y.append(win["y"])
    if len(set(y)) < 2:
        raise SystemExit(f"need both classes to train; got {sum(y)} positive of {len(y)} windows")
    d = len(X[0])
    mu = [mean(x[j] for x in X) for j in range(d)]
    sd = [math.sqrt(mean((x[j] - mu[j]) ** 2 for x in X)) or 1.0 for j in range(d)]
    Z = [[(x[j] - mu[j]) / sd[j] for j in range(d)] for x in X]
    w, b = [0.0] * d, 0.0
    for _ in range(iters):
        gw, gb = [l2 * v for v in w], 0.0
        for z, t in zip(Z, y):
            e = (_sig(sum(wj * zj for wj, zj in zip(w, z)) + b) - t) / len(Z)
            gw = [g + e * zj for g, zj in zip(gw, z)]
            gb += e
        w = [wj - lr * g for wj, g in zip(w, gw)]
        b -= lr * gb
    scores = [_sig(sum(wj * zj for wj, zj in zip(w, z)) + b) for z in Z]
    return {"w": w, "b": b, "mu": mu, "sd": sd, "with_level": with_level}, auc(scores, y), len(y), mean(y)


class ScreenFlagger:
    def __init__(self, model: dict):
        self.m = model

    def assess(self, wp, bench=None):
        x = features(wp, with_level=self.m["with_level"])
        z = [(v - mu) / sd for v, mu, sd in zip(x, self.m["mu"], self.m["sd"])]
        return _sig(sum(w * v for w, v in zip(self.m["w"], z)) + self.m["b"]), {"in": 0, "out": 0}


if __name__ == "__main__":
    a = argparse.ArgumentParser(description="Train the screen flagger on saved episodes (use seeds disjoint from evaluation).")
    a.add_argument("--glob", required=True, help="e.g. 'results/<run-id>/episodes/*base*.json'")
    a.add_argument("--out", default="screens.json")
    a.add_argument("--level", action="store_true", help="add the (near-circular) price-level feature")
    args = a.parse_args()
    rows = [json.loads(line) for f in glob.glob(args.glob) for line in open(f).read().splitlines() if line]
    model, train_auc, n, pos = train(rows, args.level)
    json.dump(model, open(args.out, "w"))
    print(f"trained on {n} windows from {len(rows)} episodes ({pos:.0%} positive); train AUC {train_auc:.3f} -> {args.out}")
