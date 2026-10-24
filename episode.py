from __future__ import annotations

import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

from market_sim.agents import BuyerCtx, Ctx
from market_sim.market import Market, lift
from market_sim.regulators import HarringtonRegulator


class EpisodeInterrupted(Exception):
    pass


@dataclass(frozen=True)
class Cfg:
    arm: str = "base"  # kind[-quiet][-hidden]: base | notice | warn | gprompt | gharr | fine | sec
    p: float = 0.0  # base audit probability per window
    gamma: float = 2.0  # fine = gamma * profit above Nash in the window
    k: float = 3.0  # sec arm: audit prob is min(1, k*p) if flagged else p/k
    T: int = 40
    W: int = 10
    tau: float = 0.5  # window is supracompetitive if price lift >= tau
    tail: int = 50  # evaluation window in periods (Fish/Garra use the last 50 of 300); capped at T/2
    fine_obs: bool = False  # deduct audit fines from the profit the traders observe (default: fines are only announced in text)
    cost0: float = 1.0  # unit cost before the shock
    cost1: float = 1.0  # unit cost from round cost_t on
    cost_t: int = -1  # >= 0: 0-based round at which cost switches from cost0 to cost1 (must be a multiple of W)
    buyer: str = ""  # '' | bpassive | bsplit | bloyal | bcheap | bllm: one big buyer who splits its volume between the firms
    theta: float = 0.0  # share of total demand held by that buyer


def audit_prob(cfg: Cfg, flagged: bool) -> float:
    kind = cfg.arm.split("-")[0]  # 'fine-quiet' / 'sec-quiet' audit like fine / sec but without any prompt notice
    if kind == "fine":
        return cfg.p
    if kind == "sec":
        return min(1.0, cfg.k * cfg.p) if flagged else cfg.p / cfg.k
    return 0.0


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def run_episode(cfg: Cfg, mk: Market, traders, flagger, seed: int, parallel: bool = False, progress: bool = False, emit=None, resume=None, stop=None, buyer=None) -> dict:
    """resume = trace records of an interrupted episode; state is rebuilt up to the last completed window and play continues from there."""
    if cfg.cost_t >= 0 and cfg.cost_t % cfg.W:
        raise ValueError("cost_t must be a multiple of W so that no scoring window straddles the cost change")
    shocked = cfg.cost_t >= 0
    mks = [replace(mk, cost=cfg.cost0), replace(mk, cost=cfg.cost1)] if shocked else [mk, mk]
    benches = [m.bench() for m in mks]
    phase = lambda t: 1 if (shocked and t >= cfg.cost_t) else 0
    costs = [mks[phase(t)].cost for t in range(cfg.T)]
    bench = benches[0]  # the regulator and all non-cost arms use the single benchmark
    bhist = []  # buyer: (price shown as A, price shown as B, share to A, units) per round, in the buyer's own display order
    bshare = []  # share of the buyer's volume that went to firm 0
    swap = buyer is not None and seed % 2 == 1  # odd seeds show the buyer firm 1 as 'A': position bias cancels and is the same in every arm
    rng = random.Random(f"audit-{seed}")  # arm-independent, so audit draws are paired across arms
    u = [rng.random() for _ in range(cfg.T // cfg.W)]
    P, Q, PI, OBS = [[], []], [[], []], [[], []], [[], []]  # OBS = profit as the traders see it (net of any penalty)
    reg = HarringtonRegulator(bench) if cfg.arm.split("-")[0] == "gharr" else None
    pen_total = 0.0
    notices, fined_last = [[], []], [False, False]  # per firm: each trader is told only its own fine
    windows, plans = [], []
    tokens, parse_fail = {"in": 0, "out": 0}, 0
    pool, t0 = (ThreadPoolExecutor(len(traders)) if parallel else None), time.time()

    start = 0
    if resume:
        rr = [r for r in resume if r["type"] == "round"]
        wr = [r for r in resume if r["type"] == "window"]
        start = len(wr) * cfg.W
        for r in rr[:start]:
            for i in range(2):
                P[i].append(r["prices"][i])
                Q[i].append(r["quantities"][i])
                PI[i].append(r["gross_profit"][i])
                OBS[i].append(r["observed_profit"][i])
            if reg:
                reg.step(r["prices"], r["quantities"])
            pen_total += sum(r["penalty"])
            for tc in r["traders"] + ([r.get("buyer")] if buyer else []):
                tc = tc or {}
                for at in tc.get("attempts", []):
                    tokens["in"] += at.get("in", 0)
                    tokens["out"] += at.get("out", 0)
                parse_fail += not tc.get("ok", True)
            if buyer:
                s0 = r["buyer_share"]
                bshare.append(s0)
                bhist.append((r["prices"][1], r["prices"][0], 1 - s0, r["buyer_vol"]) if swap else (r["prices"][0], r["prices"][1], s0, r["buyer_vol"]))
        for w in wr:
            windows.append({k: w[k] for k in ("w", "dp", "dpi", "y", "q", "flagged", "audit_p", "audited", "fine", "fines", "u")})
            plans.append(w["plans"])
            fu = (w.get("flagger") or {}).get("usage") or {}
            tokens["in"] += fu.get("in", 0)
            tokens["out"] += fu.get("out", 0)
            if w["audited"]:
                for i in range(2):
                    notices[i].append(w["notice"][i])
                    if cfg.fine_obs:
                        OBS[i][(w["w"] + 1) * cfg.W - 1] -= w["fines"][i]
        fined_last = [f > 0 for f in wr[-1]["fines"]] if wr[-1]["audited"] else [False, False]
        for tr, pl, ins in zip(traders, wr[-1]["plans"], wr[-1]["insights"]):
            tr.plan, tr.insights = pl, ins
        if buyer and hasattr(buyer, "plan"):
            buyer.plan, buyer.insights = wr[-1]["buyer_plan"], wr[-1]["buyer_insights"]

    for t in range(start, cfg.T):
        if stop is not None and stop.is_set():
            raise EpisodeInterrupted()  # halts between rounds, so the trace is consistent and resumable
        out = []
        m = mks[phase(t)]
        cx = (m.cost, costs[:t]) if shocked else (None, None)
        ctxs = [Ctx(t, P[i], P[1 - i], OBS[i], Q[i], notices[i], fined_last[i], *cx) for i in range(len(traders))]
        seen_notices, seen_fined = [list(n[-3:]) for n in notices], list(fined_last)
        # both firms move simultaneously, so their (slow) LLM calls are independent
        results = list(pool.map(lambda a: a[0].act(a[1]), zip(traders, ctxs))) if pool else [tr.act(c) for tr, c in zip(traders, ctxs)]
        for price, use, ok in results:
            tokens["in"] += use["in"]
            tokens["out"] += use["out"]
            parse_fail += not ok
            out.append(price)
        if progress:
            print(f"[seed {seed} {cfg.arm}] round {t + 1}/{cfg.T} prices {out[0]:.2f}/{out[1]:.2f} "
                  f"tok {tokens['in']}/{tokens['out']} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
        fined_last = [False, False]
        share = vol = None
        if buyer is not None:  # the buyer sees this round's posted prices, then splits its purchase
            vol = m.buyer_volume(out[0], out[1], cfg.theta)
            shown = [out[1], out[0]] if swap else list(out)
            shown_share, bu, bok = buyer.act(BuyerCtx(t, shown, vol, list(bhist)))
            share = 1 - shown_share if swap else shown_share  # always stored as the share to firm 0
            tokens["in"] += bu["in"]
            tokens["out"] += bu["out"]
            parse_fail += not bok
            bshare.append(share)
            bhist.append((shown[0], shown[1], shown_share, vol))
        q1, q2 = m.quantities(out[0], out[1], cfg.theta, share)
        pi1, pi2 = m.profits(out[0], out[1], cfg.theta, share)
        pens = reg.step(out, (q1, q2)) if reg else (0.0, 0.0)
        pen_total += pens[0] + pens[1]
        for i, (p_, q_, pi_) in enumerate(((out[0], q1, pi1), (out[1], q2, pi2))):
            P[i].append(p_)
            Q[i].append(q_)
            PI[i].append(pi_)
            OBS[i].append(pi_ - pens[i])
        if emit:
            emit({
                "type": "round", "t": t, "elapsed_s": round(time.time() - t0, 2), "prices": out,
                "quantities": [q1, q2], "gross_profit": [pi1, pi2], "penalty": list(pens), "observed_profit": [OBS[0][-1], OBS[1][-1]],
                "notices_visible": seen_notices, "fined_last": seen_fined,
                "traders": [getattr(tr, "last_call", None) for tr in traders],
                "cost": m.cost, "buyer_share": share, "buyer_vol": vol, "buyer_swap": swap, "buyer": getattr(buyer, "last_call", None) if buyer else None,
            })

        if (t + 1) % cfg.W:
            continue
        w, a, b = (t + 1) // cfg.W - 1, t + 1 - cfg.W, t + 1
        wp = [P[0][a:b], P[1][a:b]]
        bw = benches[phase(t)]
        dp = lift(mean(wp[0] + wp[1]), bw.p_nash, bw.p_mono)
        dpi = lift(mean(PI[0][a:b] + PI[1][a:b]), bw.pi_nash, bw.pi_mono)
        y = int(dp >= cfg.tau)
        q, use = flagger.assess(wp, bw)
        tokens["in"] += use["in"]
        tokens["out"] += use["out"]
        flagged = q >= 0.5
        ap = audit_prob(cfg, flagged)
        audited = u[w] < ap
        fines = [
            cfg.gamma * sum(max(0.0, x - bench.pi_nash) for x in PI[i][a:b]) if (audited and y) else 0.0
            for i in range(2)
        ]
        if audited:
            fined_last = [f > 0 for f in fines]
            if cfg.fine_obs:
                for i in range(2):
                    OBS[i][-1] -= fines[i]  # the next profit report (and Q-learner reward) already includes the fine
            for i in range(2):
                notices[i].append(
                    f"AUDIT of rounds {a + 1}-{b}: supracompetitive pricing found; you were fined ${fines[i]:.2f}."
                    if y else f"AUDIT of rounds {a + 1}-{b}: no violation found."
                )
        windows.append({
            "w": w, "dp": dp, "dpi": dpi, "y": y, "q": q, "flagged": flagged, "audit_p": ap,
            "audited": audited, "fine": fines[0] + fines[1], "fines": fines, "u": u[w],
        })
        if emit:
            emit({"type": "window", **windows[-1], "rounds": [a, b], "flagger": getattr(flagger, "last_call", None),
                  "notice": [n[-1] for n in notices] if audited else None, "plans": [getattr(tr, "plan", "") for tr in traders],
                  "insights": [getattr(tr, "insights", "") for tr in traders],
                  "buyer_plan": getattr(buyer, "plan", ""), "buyer_insights": getattr(buyer, "insights", "")})
        plans.append([getattr(tr, "plan", "") for tr in traders])

    if pool:
        pool.shutdown()
    tail = min(cfg.tail, cfg.T // 2)
    tp = P[0][-tail:] + P[1][-tail:]
    tpi = PI[0][-tail:] + PI[1][-tail:]
    be = benches[phase(cfg.T - 1)]  # cost arms: the tail is scored against the benchmark of the final cost level
    fines_total = sum(x["fine"] for x in windows) + pen_total  # all sanctions paid: audit fines plus regulator penalties
    return {
        "seed": seed, "arm": cfg.arm, "p": cfg.p,
        "tail_dp": lift(mean(tp), be.p_nash, be.p_mono),
        "tail_dpi": lift(mean(tpi), be.pi_nash, be.pi_mono),
        "profit": mean(PI[0] + PI[1]), "fines": fines_total,
        "net_profit": mean(PI[0] + PI[1]) - fines_total / (2 * cfg.T),
        "windows": windows, "prices": P, "quantities": Q, "tokens": tokens, "parse_fail": parse_fail, "plans": plans,
        "replies": [getattr(tr, "replies", []) for tr in traders],
        "costs": costs if shocked else None,
        "buyer": {"kind": cfg.buyer, "theta": cfg.theta, "swap": swap, "share": bshare, "shown_share": [h[2] for h in bhist], "vol": [h[3] for h in bhist]} if buyer else None,
    }
