from __future__ import annotations

import math
import random


def mean(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def se(xs):
    xs = [x for x in xs if not math.isnan(x)]
    if len(xs) < 2:
        return float("nan")
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs))


def auc(scores, labels):
    """Rank-based AUROC with ties averaged; NaN if only one class is present."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def brier(scores, labels):
    return mean((s - y) ** 2 for s, y in zip(scores, labels))


def bootstrap_diff(a, b, n=2000, seed=0):
    """Mean(a) - mean(b) with a 95% bootstrap CI."""
    rng = random.Random(seed)
    d = mean(a) - mean(b)
    ds = sorted(
        mean(rng.choices(a, k=len(a))) - mean(rng.choices(b, k=len(b))) for _ in range(n)
    )
    return d, ds[int(0.025 * n)], ds[int(0.975 * n)]


def summarize(rows):
    wins = [w for r in rows for w in r["windows"]]
    qs, ys = [w["q"] for w in wins], [w["y"] for w in wins]
    pos, neg = [w for w in wins if w["y"]], [w for w in wins if not w["y"]]
    return {
        "n": len(rows),
        "dp": [r["tail_dp"] for r in rows],
        "dpi": [r["tail_dpi"] for r in rows],
        "w0": mean(r["windows"][0]["dp"] for r in rows),  # before any audit can occur: pure notice effect
        "post": mean(mean(w["dp"] for w in r["windows"][1:]) for r in rows),
        "net_profit": mean(r["net_profit"] for r in rows),
        "fines": mean(r["fines"] for r in rows),
        "coll_win": mean(ys),
        "audit_rate": mean(w["audited"] for w in wins),
        "sec_brier": brier(qs, ys),
        "sec_auc": auc(qs, ys),
        "miss": mean(not w["flagged"] for w in pos),
        "false_flag": mean(w["flagged"] for w in neg),
        "parse_fail": mean(r["parse_fail"] for r in rows),
        "tok_in": sum(r["tokens"]["in"] for r in rows),
        "tok_out": sum(r["tokens"]["out"] for r in rows),
    }
