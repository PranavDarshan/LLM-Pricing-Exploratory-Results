"""Free offline analysis of logged runs (no model calls).

  python3 -m market_sim.replay table  'results/<run-id>/episodes/*.json'
  python3 -m market_sim.replay flaggers 'results/<run-id>/episodes/*.json' --tau 0.5 --budgets 0.1,0.2,0.3
  python3 -m market_sim.replay text 'results/<run-id>/traces/*.jsonl'

table:    re-aggregate saved episodes per (prefix, arm, p), including window-0 and post-window-0 collusion.
flaggers: re-label every logged window at any --tau and compare flaggers (oracle threshold, variance rule, price screen);
          detection is exact on logged paths, but traders' reaction to audits is NOT replayable.
text:     share of trader-rounds whose reply mentions enforcement / collusion / rival-tracking words, by arm and window.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from market_sim.market import Market, lift
from market_sim.metrics import auc, bootstrap_diff, mean, se, summarize
from market_sim.screens import features, train, ScreenFlagger

NAME = re.compile(r"mkt_\d{8}-\d{6}(?:-\d+)?_(P\d)_(.+)_p([\d.e+-]+)\.jsonl$")
WORDS = {
    "enforcement": r"audit|\bfine[ds]?\b|regulat|penalt|surveillance|compliance|illegal|antitrust",
    "collusion": r"collu|cartel|coordinat|tacit|signal|follow.{0,12}lead|mutual",
    "rival": r"competitor|rival|comp\b|match",
}


def load_rows(pattern, llm_only=False):
    cells = defaultdict(list)
    for f in sorted(f for pat in pattern.split(",") for f in glob.glob(pat)):  # several patterns may be given, separated by commas
        if f.endswith(".json"):  # run-directory layout: results/<run>/episodes/<cell>.json
            r = json.load(open(f))
            if "cell" in r and (not llm_only or r["tokens"]["in"] > 0):
                c = r["cell"]
                cells[(c["prefix"], c["label"], float(c["p"]))].append(r)
            continue
        m = NAME.search(f)  # legacy archive files: mkt_<stamp>_<prefix>_<label>_p<p>.jsonl
        if not m:
            continue
        for line in open(f).read().splitlines():
            if line:
                r = json.loads(line)
                if not llm_only or r["tokens"]["in"] > 0:
                    cells[(m.group(1), m.group(2), float(m.group(3)))].append(r)
    return cells


def welfare(r, mk):
    """Mean per-round consumer surplus and total welfare (consumer surplus + both profits), recomputed from logged prices.
    Buyer arms: the logit surplus formula does not apply to a buyer who chooses its split, so both are NaN there."""
    P, T = r["prices"], len(r["prices"][0])
    if r.get("buyer"):
        return float("nan"), float("nan")
    costs = r.get("costs") or [mk.cost] * T
    mks = {c: replace(mk, cost=c) for c in set(costs)}
    cs = [mks[costs[t]].consumer_surplus(P[0][t], P[1][t]) for t in range(T)]
    pi = [sum(mks[costs[t]].profits(P[0][t], P[1][t])) for t in range(T)]
    return mean(cs), mean(c + p for c, p in zip(cs, pi))


def table(args):
    mk = Market()
    print(f"{'pfx':<4}{'arm':<14}{'p':>6}{'n':>4}{'dp_tail':>14}{'w0':>7}{'post':>7}{'net_pi':>8}{'fines':>8}{'audit':>7}{'cs':>8}{'welf':>8}{'war':>6}{'hi':>5}")
    for (pfx, arm, p), rows in sorted(load_rows(args.pattern, args.llm_only).items()):
        s = summarize(rows)
        cw = [welfare(r, mk) for r in rows]
        war = mean(d < -0.1 for d in s["dp"])  # share of episodes ending in a price war (tail delta below -0.1)
        hi = mean(d > 1.0 for d in s["dp"])  # share priced above the monopoly level
        print(f"{pfx:<4}{arm:<14}{p:>6.3f}{s['n']:>4}{mean(s['dp']):>8.2f}±{se(s['dp']):<5.2f}{s['w0']:>7.2f}{s['post']:>7.2f}"
              f"{s['net_profit']:>8.1f}{s['fines']:>8.0f}{s['audit_rate']:>7.2f}{mean(c for c, _ in cw):>8.1f}{mean(w for _, w in cw):>8.1f}{war:>6.2f}{hi:>5.2f}")


PRIMARY = "notice:base,fine:notice,fine-quiet:base,sec:fine=sec"


def _pbar(r):
    return [(a + b) / 2 for a, b in zip(*r["prices"])]


def _churn(r, t0, hor=10):
    """Share of firm-rounds from t0 on with a price move >= 0.05, and the share of moves >= 0.10 that are undone within hor rounds."""
    n = mv = big = undone = 0
    for own in r["prices"]:
        T = len(own)
        for t in range(max(t0, 1), T):
            d = own[t] - own[t - 1]
            n += 1
            mv += abs(d) >= 0.05
            if abs(d) >= 0.10:
                big += 1
                undone += any(abs(x - own[t - 1]) <= 0.03 for x in own[t + 1:t + 1 + hor])
    return mv / n if n else float("nan"), undone / big if big else float("nan")


def cost_metrics(r, mk, drift=0.0):
    """One cost-shock episode: how much of the change in the Nash price shows up in the market price, and how messily."""
    costs = r["costs"]
    t0 = next((t for t in range(1, len(costs)) if costs[t] != costs[t - 1]), None)
    shocked = t0 is not None
    t0 = t0 if shocked else 30
    c0, c1 = costs[0], costs[-1]
    b0, b1 = replace(mk, cost=c0).bench(), replace(mk, cost=c1).bench()
    pb, T = _pbar(r), len(costs)
    pre, end = mean(pb[t0 - 10:t0]), mean(pb[T - 15:])
    out = {"move": end - pre, "d_nash": b1.p_nash - b0.p_nash, "d_mono": b1.p_mono - b0.p_mono}
    out["churn"], out["undone"] = _churn(r, t0)
    out["dp_pre"], out["dp_end"] = lift(pre, b0.p_nash, b0.p_mono), lift(end, b1.p_nash, b1.p_mono)
    out["above_nash"] = end - b1.p_nash  # how far the final price sits above the final Nash price
    if shocked:
        dn = out["d_nash"]
        out["pass"] = (out["move"] - drift) / dn
        sg = 1 if dn > 0 else -1
        dev = [sg * (pb[t] - pre) for t in range(t0, T)]
        fin = sg * (end - pre)
        out["overshoot"] = max(dev) / fin - 1 if fin > 0.02 else float("nan")
        out["lag"] = next((t - t0 + 1 for t in range(t0, T) if sg * (pb[t] - pre) >= 0.5 * abs(dn)), 99)
    else:
        out.update({"pass": float("nan"), "overshoot": float("nan"), "lag": float("nan")})
    return out


def _warn_unequal(by):
    ns = {k: len(v) for k, v in by.items()}
    if len(set(ns.values())) > 1:
        print(f"WARNING: unequal number of episodes per arm {ns}; comparisons are only fair if the missing episodes are unrelated to the arm\n")


def perm_test(a, b, n=20000, seed=0):
    """Two-sided permutation test for a difference in means; returns (difference, p)."""
    rng = random.Random(seed)
    d, pool, k, hits = mean(a) - mean(b), list(a) + list(b), len(a), 0
    for _ in range(n):
        rng.shuffle(pool)
        hits += abs(mean(pool[:k]) - mean(pool[k:])) >= abs(d) - 1e-12
    return d, (hits + 1) / (n + 1)


def reaction(r, after=5):
    """Cost arms: share of firm-rounds in the first `after` rounds after the cost change whose price differs by >= 0.05 from the firm's pre-shock mean."""
    costs = r["costs"]
    t0 = next((t for t in range(1, len(costs)) if costs[t] != costs[t - 1]), None)
    if t0 is None:
        return float("nan")
    return mean(abs(own[t] - mean(own[t0 - 10:t0])) >= 0.05 for own in r["prices"] for t in range(t0, t0 + after))


def move_events(r, thr=0.10, win=(8, 13), stick="abs", strict=True, tmin=5, fresh=5):
    """Initiating price moves of one episode: (kind, still_in_place, vs_best_response, round, market price minus Nash).
    A move is initiating if the rival did not move the same way in that round or the one before; its window never crosses a cost change.
    fresh > 0: the firm made no OPPOSITE move of this size in the previous `fresh` rounds, so retracting an earlier move is not counted as a new move (cuts and raises are treated alike).
    still_in_place: the mean price over rounds t+win[0]..t+win[1]-1 keeps at least thr ('abs') or half of the move ('half')."""
    P, costs = r["prices"], r.get("costs")
    T = len(P[0])
    cap = next((t for t in range(1, T) if costs[t] != costs[t - 1]), T) if costs else T
    m = replace(Market(), cost=costs[0] if costs else Market().cost)
    nash, a, b, out = m.bench().p_nash, win[0], win[1], []
    for i in (0, 1):
        j = 1 - i
        for t in range(max(tmin, 2), cap - b):
            d = P[i][t] - P[i][t - 1]
            if abs(d) < thr:
                continue
            if fresh and any(s_ * (P[i][u] - P[i][u - 1]) <= -thr for u in range(max(1, t - fresh), t) for s_ in ((-1 if d < 0 else 1),)):
                continue
            s = -1 if d < 0 else 1
            if strict and (s * (P[j][t] - P[j][t - 1]) >= 0.05 or s * (P[j][t - 1] - P[j][t - 2]) >= 0.05):
                continue
            kept = s * (mean(P[i][t + a:t + b]) - P[i][t - 1])
            gap = m.best_response(P[j][t - 1]) - P[i][t - 1]
            where = "neutral" if abs(gap) < 0.05 else ("toward" if (d > 0) == (gap > 0) else "away")  # relative to the myopic best response
            out.append(("cut" if d < 0 else "raise", kept >= (thr if stick == "abs" else 0.5 * abs(d)), where, t, (P[0][t - 1] + P[1][t - 1]) / 2 - nash))
    return out


def ratchet_gap(evl, keep=lambda x: True):
    """Share of cuts still in place minus share of raises still in place; evl = one event list per episode."""
    c = [x[1] for ev in evl for x in ev if x[0] == "cut" and keep(x)]
    r = [x[1] for ev in evl for x in ev if x[0] == "raise" and keep(x)]
    return mean(c) - mean(r) if c and r else float("nan")


def _cboot(evl, stat, n=400, seed=0):
    rng, k, bs = random.Random(seed), len(evl), []
    for _ in range(n):
        v = stat([evl[rng.randrange(k)] for _ in range(k)])
        if v == v:
            bs.append(v)
    bs.sort()
    pt = stat(evl)
    return (pt, bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1]) if len(bs) > 20 else (pt, float("nan"), float("nan"))


def ratchet(args):
    """python3 -m market_sim.replay ratchet 'results/<run>/episodes/*.json'   (rival-visible arms only; hidden-rival episodes are skipped)"""
    import itertools
    skip = set(x for x in getattr(args, "exclude_arms", "").split(",") if x)
    rows = [r for rs in load_rows(args.pattern, args.llm_only).values() for r in rs if not r.get("cell", {}).get("hidden_rival") and r.get("cell", {}).get("label") not in skip]
    evl = [move_events(r) for r in rows]
    nc = sum(1 for ev in evl for x in ev if x[0] == "cut")
    nr = sum(1 for ev in evl for x in ev if x[0] == "raise")
    f = lambda t: f"{t[0]:+.2f} [{t[1]:+.2f},{t[2]:+.2f}]"
    print(f"Persistence of initiating price moves (>= 0.10; still in place over rounds t+8..t+12): {len(rows)} episodes, {nc} cuts, {nr} raises\n")
    print(f"1. cut minus raise, all moves: {f(_cboot(evl, ratchet_gap))}")
    print("2. attractor control: where the move went relative to the myopic best response to the rival's last price")
    for kind in ("cut", "raise"):
        parts = []
        for w in ("toward", "away", "neutral"):
            v = [x[1] for ev in evl for x in ev if x[0] == kind and x[2] == w]
            parts.append(f"{w} n={len(v)} in place {mean(v):.2f}")
        print(f"   {kind:<6}" + "   ".join(parts))
    within = {w: _cboot(evl, lambda s, w=w: ratchet_gap(s, lambda x: x[2] == w)) for w in ("toward", "away")}
    print(f"   cut minus raise within moves toward the best response: {f(within['toward'])}   within moves away: {f(within['away'])}")
    print("3. opening glide: cut minus raise by round of the move")
    for lo, hi in ((5, 14), (15, 24), (25, 60)):
        print(f"   rounds {lo}-{hi}: {f(_cboot(evl, lambda s, lo=lo, hi=hi: ratchet_gap(s, lambda x: lo <= x[3] <= hi)))}")
    per = [d for d in (ratchet_gap([ev]) for ev in evl) if d == d]
    rng, obs, hits, R = random.Random(5), ratchet_gap(evl), 0, 3000
    for _ in range(R):
        sh = []
        for ev in evl:
            ks = [x[0] for x in ev]
            rng.shuffle(ks)
            sh.append([(k,) + x[1:] for k, x in zip(ks, ev)])
        hits += abs(ratchet_gap(sh)) >= abs(obs) - 1e-12
    p_perm = (hits + 1) / (R + 1)
    print(f"4. within-episode: {len(per)} episodes have both kinds; difference positive in {sum(d > 0 for d in per)}, negative in {sum(d < 0 for d in per)}; permutation p (labels shuffled inside each episode) = {p_perm:.4f}")
    specs = list(itertools.product((0.05, 0.10, 0.15, 0.20), ((3, 6), (8, 13), (10, 16)), ("abs", "half"), (True, False), (5, 15), (0, 5)))
    res = [(_cboot([move_events(r, *sp) for r in rows], ratchet_gap, 100), sp) for sp in specs]
    for label, sel in (("fresh moves only (corrected rule)", lambda sp: sp[5] > 0), ("retractions counted as moves (old rule, inflated)", lambda sp: sp[5] == 0)):
        ok = [x for x, sp in res if sel(sp) and x[0] == x[0]]
        pos = sum(x[0] > 0 for x in ok)
        print(f"5. specification curve, {label}: {len(ok)} usable of {sum(1 for _, sp in res if sel(sp))}; gap > 0 in {pos}; 95% CI excludes 0 in {sum(x[1] > 0 for x in ok if x[1] == x[1])}; median {sorted(x[0] for x in ok)[len(ok) // 2]:+.2f}" if ok else f"5. specification curve, {label}: no usable specification")
    ok = [x for x, sp in res if sp[5] > 0 and x[0] == x[0]]
    pos = sum(x[0] > 0 for x in ok)
    checks = [("within moves toward the best response the CI excludes 0", within["toward"][1] > 0), ("within-episode permutation p < 0.05", p_perm < 0.05), ("gap > 0 in at least 80% of corrected specifications", bool(ok) and pos / len(ok) >= 0.8)]
    print("\nCRITERIA (fixed in code): " + "; ".join(f"{'PASS' if c else 'FAIL'}: {n}" for n, c in checks))


MIN_N = 10  # fewer episodes per arm than this is never reported as confirmatory
CONFIRM = [  # the primary contrast of each experiment is fixed here, before any confirmatory run
    ("buyer", "bllm", "nobuyer", "collusion index over the last 30 rounds", lambda r: r["tail_dp"]),
    ("cost", "costup", "costdown", "share of firm-rounds moving >= 0.05 in the 5 rounds after the shock", reaction),
]

# Freeze experiment: thresholds fixed before any confirmatory run.
FREEZE_TAIL = 30  # rounds scored (the last 30 of 60)
FREEZE_HOLD = 10  # a price counts as 'held' if unchanged (within 0.005) for this many rounds up to and including the current one
FREEZE_FRAC = 0.7  # ... and 'dominated' if the round's profit is below this fraction of the myopic best-response profit


def allocation(kind, pi, pj):
    """Share of the big buyer's volume that goes to the firm pricing pi when the rival prices pj, under the scripted buyer rules."""
    if kind == "bcheap":
        return 1.0 if pi < pj - 1e-9 else 0.0 if pj < pi - 1e-9 else 0.5
    if kind == "bsplit":
        return 0.5
    if kind == "bpassive":
        return Market().logit_split(pi, pj)
    raise ValueError(kind)


def firm_profit(r, i, pi, pj, t):
    """Profit of firm i at its own price pi against the rival's pj in round t, using the episode's true demand (cost, buyer kind, theta).
    bloyal: the buyer's firm is fixed. bllm: the buyer's counterfactual reply is unknown, so NaN."""
    costs = r.get("costs")
    m = replace(Market(), cost=costs[t] if costs else Market().cost)
    b = r.get("buyer")
    if not b:
        return m.profits(pi, pj)[0]
    if b["kind"] == "bllm":
        return float("nan")
    if b["kind"] == "bloyal":
        s = 1.0 if i == 0 else 0.0
    else:
        s = allocation(b["kind"], pi, pj)
    return m.profits(pi, pj, b["theta"], s)[0]


def best_response_profit(r, i, pj, t, grid=None):
    grid = grid or [k / 100 for k in range(100, 452)]
    return max(firm_profit(r, i, p, pj, t) for p in grid)


def check_demand(r, tol=1e-6):
    """Recomputed profit at the logged prices must equal the simulator's logged quantities x margin for every round.
    Guards every regret number against a mismatch between this file's buyer rule / theta / cost and what actually ran."""
    P, Q, T = r["prices"], r["quantities"], len(r["prices"][0])
    costs = r.get("costs") or [Market().cost] * T
    if (r.get("buyer") or {}).get("kind") == "bllm":
        return True
    for t in range(T):
        for i in (0, 1):
            logged = (P[i][t] - costs[t]) * Q[i][t]
            if abs(firm_profit(r, i, P[i][t], P[1 - i][t], t) - logged) > tol * max(1.0, abs(logged)):
                return False
    return True


def failed_rounds(r, trace_path):
    """(firm, round) pairs whose price was NOT produced by the model (parse failure after all retries repeats the previous price).
    Returns None when the trace is missing but the episode reports failures, so the caller can exclude it rather than guess."""
    if trace_path is None or not trace_path.exists():
        return set() if r.get("parse_fail", 0) == 0 else None
    bad = set()
    for line in open(trace_path):
        if not line.strip():
            continue
        x = json.loads(line)
        if x.get("type") != "round":
            continue
        for i, tc in enumerate(x["traders"]):
            if tc is not None and tc.get("ok") is False:
                bad.add((i, x["t"]))
    return bad


def freeze_metrics(r, hold=FREEZE_HOLD, frac=FREEZE_FRAC, tail=FREEZE_TAIL, bad=frozenset()):
    """One episode: per-seller freeze statistics over the last `tail` rounds. Rounds in `bad` (model failed to answer) break a hold:
    a repeated price the model did not choose is never counted as holding."""
    P, T = r["prices"], len(r["prices"][0])
    lo = T - tail
    out = {"stuck": [], "regret": [], "frozen": [], "t_freeze": [], "below_cost": [], "tail_profit": []}
    for i in (0, 1):
        j = 1 - i
        held_dom, regrets, acts = [], [], []
        for t in range(lo, T):
            run_ = range(t - hold + 1, t + 1)
            held = t - hold + 1 >= 0 and all(abs(P[i][u] - P[i][t]) < 0.005 for u in run_) and not any((i, u) in bad for u in run_)
            act, br = firm_profit(r, i, P[i][t], P[j][t], t), best_response_profit(r, i, P[j][t], t)
            acts.append(act)
            if act != act:  # bllm: no counterfactual
                held_dom.append(float("nan"))
                regrets.append(float("nan"))
                continue
            held_dom.append(float(held and br > 0 and act < frac * br))
            regrets.append((br - act) / br if br > 0 else float("nan"))
        out["stuck"].append(mean(held_dom))
        out["regret"].append(mean(regrets))
        out["tail_profit"].append(mean(acts))
        out["frozen"].append(float(all(abs(P[i][u] - P[i][T - 1]) < 0.005 and (i, u) not in bad for u in range(T - 20, T))))
        tf = T  # first round from which the price never changes again
        while tf > 0 and abs(P[i][tf - 1] - P[i][T - 1]) < 0.005:
            tf -= 1
        out["t_freeze"].append(tf + 1)
        costs = r.get("costs") or [Market().cost] * T
        out["below_cost"].append(mean(P[i][t] < costs[t] - 1e-9 for t in range(lo, T)))
    m = {k: mean(v) for k, v in out.items()}
    tp = out["tail_profit"]
    m["asym"] = abs(tp[0] - tp[1]) / max(abs(tp[0]) + abs(tp[1]), 1e-9)  # assumption-free: how unequal the two sellers' realised tail profits are
    return m


FREEZE_WORDS = {"optimum": r"optimum|optimal|equilibrium", "no_experiment": r"no (further )?(experiment|test|probe|exploration)|not worth (testing|experimenting)|no reason to (change|move|test)", "price_war": r"price war", "hold": r"\bhold\b"}
FREEZE_GRID = [(h, f, tl) for h in (5, 10, 15) for f in (0.5, 0.7, 0.9) for tl in (20, 30)]  # robustness grid around the fixed primary definition


def _trace_path(ep_path):
    p = Path(ep_path)
    return p.parent.parent / "traces" / (p.stem + ".jsonl")


def freeze_text(trace_paths):
    """Per label: share of tail seller-rounds whose reply contains each marker (descriptive only)."""
    out = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for f in trace_paths:
        if not f.exists():
            continue
        recs = [json.loads(line) for line in open(f) if line.strip()]
        label = recs[0].get("label", "?")
        rounds = [x for x in recs if x.get("type") == "round"]
        for x in rounds[len(rounds) - FREEZE_TAIL:]:
            for tc in x["traders"]:
                if not tc or not tc.get("attempts"):
                    continue
                txt = tc["attempts"][-1].get("text", "") or ""
                for k, rx in FREEZE_WORDS.items():
                    c = out[label][k]
                    c[0] += bool(re.search(rx, txt, re.I))
                    c[1] += 1
    return out


def freeze(args):
    """python3 -m market_sim.replay freeze 'results/<run>/episodes/*.json'
    PRIMARY (fixed before the run): memory ablation, X-nomem minus X, on 'stuck' = share of tail seller-rounds held >= FREEZE_HOLD rounds at < FREEZE_FRAC of the best-response profit.
    Everything else (demand shape bcheap vs nobuyer, disclosure X-told vs X, regret) is secondary and Holm-adjusted.
    Safeguards: synthetic / non-model episodes are dropped; the recomputed demand must reproduce every logged profit; rounds the model failed to answer never count as 'held';
    episodes with failures but no trace are excluded; pooling several runs is flagged."""
    files = sorted(f for pat in args.pattern.split(",") for f in glob.glob(pat) if f.endswith(".json"))
    runs = sorted({str(Path(f).parent.parent) for f in files})
    if len(runs) > 1:
        print(f"WARNING: episodes come from {len(runs)} run directories {runs}; arms are only comparable within one run\n")
    by, dropped = defaultdict(list), defaultdict(int)
    for f in files:
        r = json.load(open(f))
        if "cell" not in r:
            dropped["no cell record (legacy file)"] += 1
            continue
        man = Path(f).parent.parent / "manifest.json"
        client = json.loads(man.read_text()).get("llm_client", "ChatClient") if man.exists() else "ChatClient"
        if r["tokens"]["in"] <= 0 or client != "ChatClient":
            dropped["synthetic or no model calls"] += 1
            continue
        if not check_demand(r):
            raise SystemExit(f"{f}: recomputed demand does not reproduce the logged profits; the analysis' buyer rule / theta / cost differ from the simulator's")
        bad = failed_rounds(r, _trace_path(f))
        if bad is None:
            dropped["model failures but no trace to locate them"] += 1
            continue
        r["_bad"], r["_file"] = bad, f
        by[r["cell"]["label"]].append(r)
    for k, n in dropped.items():
        print(f"dropped {n} episode(s): {k}")
    if not by:
        raise SystemExit("no usable model episodes found")
    met = {k: [freeze_metrics(r, bad=r["_bad"]) for r in v] for k, v in by.items()}
    fails = {k: (sum(len(r["_bad"]) for r in v), sum(2 * len(r["prices"][0]) for r in v)) for k, v in by.items()}
    txt = freeze_text([_trace_path(r["_file"]) for v in by.values() for r in v])
    print(f"Freeze analysis, last {FREEZE_TAIL} rounds. stuck = share of seller-rounds held >= {FREEZE_HOLD} rounds at < {FREEZE_FRAC:.0%} of the best-response profit;")
    print("regret = (best-response profit - actual) / best-response; frozen = share of sellers whose price never changes in the last 20 rounds; t_freeze = mean round from which a seller's price never changes again;")
    print("below_cost = share of seller-rounds priced below unit cost; asym = |profit A - profit B| / (|A| + |B|) in the tail (no best-response assumption);")
    print("fail = share of seller-rounds where the model gave no usable price (such rounds never count as held). bllm episodes have no counterfactual buyer reply, so stuck/regret are NaN there.\n")
    print(f"{'arm':<16}{'n':>3}{'stuck':>7}{'regret':>8}{'asym':>6}{'frozen':>8}{'t_freeze':>9}{'below_cost':>11}{'fail':>6}   text markers (descriptive)")
    for k in sorted(met):
        ms = met[k]
        tm = txt.get(k)
        tstr = "  ".join(f"{w}={c[0] / c[1]:.2f}" for w, c in tm.items() if c[1]) if tm else "(no traces)"
        print(f"{k:<16}{len(ms):>3}{mean(m['stuck'] for m in ms):>7.2f}{mean(m['regret'] for m in ms):>8.2f}{mean(m['asym'] for m in ms):>6.2f}{mean(m['frozen'] for m in ms):>8.2f}"
              f"{mean(m['t_freeze'] for m in ms):>9.1f}{mean(m['below_cost'] for m in ms):>11.2f}{fails[k][0] / fails[k][1]:>6.3f}   {tstr}")
    if any(f[0] / f[1] > 0.05 for f in fails.values()):
        print("\nWARNING: more than 5% of seller-rounds in some arm had no model answer; compare arms' failure rates before reading anything else")
    contrasts = []
    for k in sorted(met):
        kind, mods = k.split("-")[0], k.split("-")[1:]
        if "nomem" in mods:
            contrasts.append(("memory ablation", k, "-".join([kind] + [m for m in mods if m != "nomem"])))
        if "told" in mods:
            contrasts.append(("buyer rule disclosed", k, "-".join([kind] + [m for m in mods if m != "told"])))
        if kind == "bcheap":
            contrasts.append(("winner-take-all demand", k, "-".join(["nobuyer"] + mods)))
    rows = []
    for name, a, b in contrasts:
        if a not in met or b not in met:
            continue
        for key in ("stuck", "regret", "asym"):
            x, y = [m[key] for m in met[a] if m[key] == m[key]], [m[key] for m in met[b] if m[key] == m[key]]
            if min(len(x), len(y)) < 2:
                continue
            d, p = perm_test(x, y)
            _, lo, hi = bootstrap_diff(x, y)
            primary = name == "memory ablation" and key == "stuck" and a == "bcheap-nomem"
            rows.append((primary, name, a, b, key, len(x), len(y), d, lo, hi, p))
    if not rows:
        print("\nno pre-specified contrast available (need e.g. bcheap with bcheap-nomem, bcheap-told or nobuyer in the same run)")
        return
    sec = sorted((r[10], i) for i, r in enumerate(rows) if not r[0])
    holm = {}
    for rank, (p, i) in enumerate(sec):
        holm[i] = min(1.0, max([holm[j] for _, j in sec[:rank]] + [p * (len(sec) - rank)]))
    print()
    for i, (primary, name, a, b, key, nx, ny, d, lo, hi, p) in enumerate(rows):
        ok = min(nx, ny) >= MIN_N
        tag = "PRIMARY" if primary else f"secondary, Holm p = {holm[i]:.4f}"
        print(f"{name}: {a} minus {b}, {key}: n = {nx} / {ny}   difference {d:+.3f}   95% CI [{lo:+.3f},{hi:+.3f}]   permutation p = {p:.4f}   [{tag}]   "
              + ("CONFIRMATORY" if ok else f"NOT CONFIRMATORY (< {MIN_N} per arm)"))
    if "bcheap-nomem" in met and "bcheap" in met:
        print("\nrobustness grid for the primary contrast (hold rounds, fraction of best response, tail length): difference and permutation p")
        pos = tot = sig = 0
        for h, f, tl in FREEZE_GRID:
            x = [freeze_metrics(r, h, f, tl, r["_bad"])["stuck"] for r in by["bcheap-nomem"]]
            y = [freeze_metrics(r, h, f, tl, r["_bad"])["stuck"] for r in by["bcheap"]]
            d, p = perm_test(x, y, n=4000)
            pos += d < 0
            sig += p < 0.05
            tot += 1
            print(f"   hold {h:>2}  frac {f:.1f}  tail {tl:>2}:  {d:+.3f}   p = {p:.3f}")
        print(f"   memory ablation reduces 'stuck' in {pos}/{tot} specifications; p < 0.05 in {sig}/{tot}")


def confirm(args):
    """python3 -m market_sim.replay confirm 'results/<run>/episodes/*.json'   (all arms must come from one run)"""
    by = {label: rows for (pfx, label, p), rows in load_rows(args.pattern, args.llm_only).items()}
    shown = False
    for name, a, b, what, fn in CONFIRM:
        if a not in by or b not in by:
            continue
        shown = True
        x, y = [fn(r) for r in by[a]], [fn(r) for r in by[b]]
        ok = min(len(x), len(y)) >= MIN_N
        d, p = perm_test(x, y)
        _, lo, hi = bootstrap_diff(x, y)
        print(f"{name}: {a} minus {b}, {what}")
        print(f"  n = {len(x)} / {len(y)}   difference {d:+.3f}   95% bootstrap CI [{lo:+.3f},{hi:+.3f}]   permutation p = {p:.4f}")
        print(f"  status: {'CONFIRMATORY (n >= %d per arm)' % MIN_N if ok else 'NOT CONFIRMATORY: fewer than %d episodes per arm, treat as descriptive' % MIN_N}\n")
    if not shown:
        raise SystemExit("no primary contrast found: need bllm and nobuyer, or costup and costdown")


def quality(args):
    """python3 -m market_sim.replay quality 'results/<run>/traces/*[0-9].jsonl'   answer-extraction and truncation audit per arm"""
    from market_sim.agents import PRICE_RE, SHARE_RE
    st = defaultdict(lambda: defaultdict(int))
    for f in sorted(glob.glob(args.pattern)):
        if f.endswith(".orphans.jsonl"):
            continue
        recs = [json.loads(line) for line in open(f) if line.strip()]
        arm = recs[0].get("label", "?")
        for r in recs:
            if r.get("type") != "round":
                continue
            for who, tc in (("seller", r["traders"][0]), ("seller", r["traders"][1]), ("buyer", r.get("buyer"))):
                if not tc:
                    continue
                s = st[(arm, who)]
                atts = [x for x in tc.get("attempts", []) if "text" in x]
                s["rounds"] += 1
                s["failed"] += not tc.get("ok", True)
                s["retried"] += len(atts) > 1
                s["attempts"] += len(atts)
                s["cut"] += sum(x.get("finish") == "length" for x in atts)
                if atts:
                    ms = [float(m.group(1)) for m in (PRICE_RE if who == "seller" else SHARE_RE).finditer(atts[-1]["text"])]
                    s["ambiguous"] += len(ms) > 1 and ms[0] != ms[-1]
    print(f"{'arm':<12}{'who':<8}{'rounds':>7}{'failed':>8}{'retried':>9}{'cut at max tokens':>19}{'ambiguous answer':>18}")
    bad = []
    for (arm, who), s in sorted(st.items()):
        n = s["rounds"]
        row = (s["failed"] / n, s["retried"] / n, s["cut"] / max(s["attempts"], 1), s["ambiguous"] / n)
        print(f"{arm:<12}{who:<8}{n:>7}{row[0]:>8.3f}{row[1]:>9.3f}{row[2]:>19.3f}{row[3]:>18.3f}")
        if row[0] > 0.05 or row[2] > 0.02 or row[3] > 0.01:
            bad.append(f"{arm}/{who}")
    fails = [s["failed"] / s["rounds"] for (arm, who), s in st.items() if who == "seller"]
    if fails and max(fails) - min(fails) > 0.03:
        bad.append("failure rate differs by more than 3 points across arms")
    print("\nQUALITY: " + ("WARNING for " + ", ".join(bad) + " (limits: failed <= 5%, cut <= 2%, ambiguous <= 1%, arm spread <= 3 points)" if bad else "OK (failed <= 5%, cut <= 2%, ambiguous <= 1%, arm spread <= 3 points)"))


def passthrough(args):
    """python3 -m market_sim.replay passthrough 'results/<run>/episodes/*.json'   (arms costup / costdown / costlo / costhi)"""
    mk = Market()
    by = {label: rows for (pfx, label, p), rows in load_rows(args.pattern, args.llm_only).items() if label.split("-")[0].startswith("cost")}
    if not by:
        raise SystemExit("no cost-shock cells found (arms costup / costdown / costlo / costhi)")
    _warn_unequal(by)
    ctrl = {"costup": "costlo", "costdown": "costhi"}
    drift = {}
    for k, c in ctrl.items():
        rows = by.get(c, [])
        drift[k] = mean(cost_metrics(r, mk)["move"] for r in rows) if rows else 0.0
    met = {k: [cost_metrics(r, mk, drift.get(k, 0.0)) for r in rows] for k, rows in by.items()}
    print("Cost shock: the final market price (mean of both firms, last 15 rounds) vs the 10 rounds before the shock.")
    print("pass = share of the change in the Nash price that shows up in the price, net of the no-shock control's drift (1 = full, 0 = none)")
    print("undone = share of moves >= 0.10 reversed within 10 rounds; overshoot = peak response / final response - 1; lag = rounds to half of the Nash move (99 = never)\n")
    print(f"{'arm':<10}{'n':>3}{'dNash':>7}{'drift':>7}{'pass':>7}{'dp pre':>8}{'dp end':>8}{'above Nash':>11}{'churn':>7}{'undone':>8}{'overshoot':>10}{'median lag':>11}")
    for k in sorted(met):
        ms = met[k]
        lags = sorted(m["lag"] for m in ms if m["lag"] == m["lag"])
        print(f"{k:<10}{len(ms):>3}{mean(m['d_nash'] for m in ms):>7.2f}{drift.get(k, mean(m['move'] for m in ms)):>7.2f}{mean(m['pass'] for m in ms):>7.2f}"
              f"{mean(m['dp_pre'] for m in ms):>8.2f}{mean(m['dp_end'] for m in ms):>8.2f}{mean(m['above_nash'] for m in ms):>11.2f}{mean(m['churn'] for m in ms):>7.2f}"
              f"{mean(m['undone'] for m in ms):>8.2f}{mean(m['overshoot'] for m in ms):>10.2f}{(lags[len(lags) // 2] if lags else float('nan')):>11.0f}")
    if "costup" in met and "costdown" in met:
        print("\ncostup minus costdown (episode bootstrap, 95% CI); rockets-and-feathers would show a positive 'pass' difference:")
        for key, nm in (("pass", "pass-through"), ("churn", "price-move frequency after the shock"), ("undone", "share of moves undone"), ("overshoot", "overshoot")):
            a, b = [m[key] for m in met["costup"] if m[key] == m[key]], [m[key] for m in met["costdown"] if m[key] == m[key]]
            if min(len(a), len(b)) < 3:
                print(f"  {nm}: fewer than 3 episodes per arm, no interval")
                continue
            d, lo, hi = bootstrap_diff(a, b)
            print(f"  {nm}: {d:+.2f} [{lo:+.2f},{hi:+.2f}]  (n = {len(a)} / {len(b)})")


def buyer_metrics(r, tail=30):
    """One buyer-arm episode: who ended up cooperating with whom."""
    P, T = r["prices"], len(r["prices"][0])
    lo = T - tail
    b = r.get("buyer")
    out = {"dp": r["tail_dp"], "gap": mean(abs(P[0][t] - P[1][t]) for t in range(lo, T)), "profit": r["profit"],
           "price": mean((P[0][t] + P[1][t]) / 2 for t in range(lo, T))}
    if not b:
        return {**out, "conc": float("nan"), "persist": float("nan"), "discount": float("nan"), "slope": float("nan"), "saving": float("nan"), "disp_a": float("nan"), "type": "no buyer"}
    s = b["share"]
    st = s[lo:]
    out["disp_a"] = mean((1 - x) if b.get("swap") else x for x in st)  # share to whichever seller the buyer was shown as 'A'
    out["conc"] = mean(abs(2 * x - 1) for x in st)
    out["persist"] = max(mean(x > 0.5 for x in st), mean(x < 0.5 for x in st))
    fav = 0 if mean(st) >= 0.5 else 1  # the seller that gets most of the volume
    out["discount"] = mean(P[fav][t] - P[1 - fav][t] for t in range(lo, T))  # negative = the favoured seller is cheaper
    xs, ys = [P[1][t] - P[0][t] for t in range(T)], list(s)
    mx, my = mean(xs), mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    out["slope"] = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / vx if vx > 1e-12 else float("nan")  # + = sends more to A when A is cheaper
    paid = mean(s[t] * P[0][t] + (1 - s[t]) * P[1][t] for t in range(lo, T))
    out["saving"] = mean((P[0][t] + P[1][t]) / 2 for t in range(lo, T)) - paid  # posted average minus what the buyer actually paid
    if out["dp"] < 0.25:
        out["type"] = "competitive"
    elif out["conc"] >= 0.6:
        out["type"] = ("buyer-seller pairing (loyal, cheaper favoured seller)" if out["discount"] <= -0.05 else "captured buyer (loyal, no discount)") if out["persist"] >= 0.8 else "switching buyer"
    else:
        out["type"] = "seller-seller cartel (buyer not steering)" if out["dp"] >= 0.5 else "mixed"
    return out


def buyer(args):
    """python3 -m market_sim.replay buyer 'results/<run>/episodes/*.json'   (arms nobuyer / bpassive / bsplit / bloyal / bcheap / bllm)"""
    by = {label: rows for (pfx, label, p), rows in load_rows(args.pattern, args.llm_only).items()}
    print("Buyer experiment, last 30 rounds. dp = collusion index vs the original benchmark (0 = Nash price, 1 = monopoly price).")
    print("conc = how one-sided the buyer's split is (0 = 50/50, 1 = all to one seller); persist = share of rounds the same seller gets most; discount = favoured seller's price minus the other's;")
    print("slope = change in share to A per +1.00 in (price B - price A); saving = average posted price minus the buyer's average paid price;")
    print("toA = share of volume to the seller the buyer was shown as 'A' (0.5 = no position bias; the display order alternates with the seed)\n")
    _warn_unequal(by)
    print(f"{'arm':<10}{'n':>3}{'dp':>7}{'price':>7}{'gap':>6}{'profit':>8}{'conc':>6}{'persist':>8}{'discount':>9}{'slope':>7}{'saving':>8}{'toA':>6}   types")
    for k in sorted(by):
        ms = [buyer_metrics(r) for r in by[k]]
        kinds = defaultdict(int)
        for m in ms:
            kinds[m["type"]] += 1
        print(f"{k:<10}{len(ms):>3}{mean(m['dp'] for m in ms):>7.2f}{mean(m['price'] for m in ms):>7.2f}{mean(m['gap'] for m in ms):>6.2f}{mean(m['profit'] for m in ms):>8.1f}{mean(m['conc'] for m in ms):>6.2f}"
              f"{mean(m['persist'] for m in ms):>8.2f}{mean(m['discount'] for m in ms):>9.2f}{mean(m['slope'] for m in ms):>7.2f}{mean(m['saving'] for m in ms):>8.2f}{mean(m['disp_a'] for m in ms):>6.2f}   "
              + "; ".join(f"{n} {t}" for t, n in sorted(kinds.items())))
    for k in sorted(by):
        v = [buyer_metrics(r)["disp_a"] for r in by[k]]
        if k != "nobuyer" and all(x == x for x in v) and len(v) >= 3:
            d, lo, hi = bootstrap_diff(v, [0.5] * len(v))
            print(f"  position bias {k}: share to displayed-A minus 0.5 = {d:+.2f} [{lo:+.2f},{hi:+.2f}]")
    if "nobuyer" in by:
        base = [r["tail_dp"] for r in by["nobuyer"]]
        for k in sorted(by):
            if k != "nobuyer" and min(len(base), len(by[k])) >= 3:
                d, lo, hi = bootstrap_diff([r["tail_dp"] for r in by[k]], base)
                print(f"  dp({k}) - dp(nobuyer) = {d:+.2f} [{lo:+.2f},{hi:+.2f}]")


def shock_raw(own, t0):
    """Own price change vs. the 10 rounds before the step: mean of the 5 rounds right after, and of the last 15 rounds."""
    pre = mean(own[t0 - 10:t0])
    return mean(own[t0:t0 + 5]) - pre, mean(own[len(own) - 15:]) - pre


def shock_metrics(own, riv, mk, t0, drift=(0.0, 0.0)):
    """One seller against a scripted rival that stepped at round t0. Responses are net of the no-shock control's drift
    and measured against the rational (best-response) move."""
    pre = mean(own[t0 - 10:t0])
    r_old, r_new = riv[t0 - 1], riv[t0]
    need = mk.best_response(r_new) - mk.best_response(r_old)  # how far a myopic best responder would move
    n = len(own)
    e, l = shock_raw(own, t0)
    early, late = e - drift[0], l - drift[1]
    sg = 1 if need > 0 else -1
    lag = next((s - t0 + 1 for s in range(t0, n) if sg * (own[s] - pre) >= 0.5 * abs(need)), None) if abs(need) > 0.01 else None
    return dict(shock=r_new - r_old, need=need, early=early, late=late, early_ratio=early / need if abs(need) > 0.01 else float("nan"),
                late_ratio=late / need if abs(need) > 0.01 else float("nan"), lag=lag)


def control_drift(none_eps, t0):
    """(early, late) change of the no-shock control, from [(own, riv)] episodes."""
    if not none_eps:
        return 0.0, 0.0
    raws = [shock_raw(o, t0) for o, _ in none_eps]
    return mean(r[0] for r in raws), mean(r[1] for r in raws)


def shock_contrast(cut, raise_, none, mk, t0, key, rng, B=2000):
    """Difference (cut minus raise) in drift-adjusted response / rational move. Episodes are resampled in all three arms,
    so the uncertainty of the control's drift is included. Returns (estimate, ci_low, ci_high, two-sided p)."""
    def est(c, r, z):
        d = control_drift(z, t0)
        a = [m[key] for m in (shock_metrics(o, v, mk, t0, d) for o, v in c)]
        b = [m[key] for m in (shock_metrics(o, v, mk, t0, d) for o, v in r)]
        return mean(a) - mean(b)
    point = est(cut, raise_, none)
    pick = lambda xs: [xs[rng.randrange(len(xs))] for _ in xs]
    ds = sorted(est(pick(cut), pick(raise_), pick(none) if none else none) for _ in range(B))
    lo, hi = ds[int(0.025 * B)], ds[int(0.975 * B) - 1]
    p = min(1.0, 2 * min(sum(d <= 0 for d in ds), sum(d >= 0 for d in ds)) / B)
    return point, lo, hi, max(p, 1 / B)


def shocks(args):
    """python3 -m market_sim.replay shocks 'results/<run>/episodes/*.json' --shock-t 30"""
    mk, t0 = Market(), args.shock_t
    cells = load_rows(args.pattern, args.llm_only)
    by = {label: rows for (pfx, label, p), rows in cells.items() if label.startswith("step")}
    if not by:
        raise SystemExit("no step-rival cells found (arms stepcut / stepraise / stepnone)")
    drift = control_drift([(r["prices"][0], r["prices"][1]) for r in by.get("stepnone", [])], t0)
    print(f"drift of the no-shock control relative to rounds {t0 - 9}-{t0}: first 5 rounds after {drift[0]:+.3f}, last 15 rounds {drift[1]:+.3f}\n")
    print(f"{'cell':<11}{'n':>3}{'rival step':>11}{'rational move':>14}{'early resp.':>12}{'late resp.':>11}{'early/rational':>15}{'late/rational':>14}{'half-way in 5':>14}{'median lag':>11}")
    met = {}
    for label in sorted(by):
        ms = [shock_metrics(r["prices"][0], r["prices"][1], mk, t0, drift) for r in by[label]]
        met[label] = ms
        lags = sorted(m["lag"] if m["lag"] is not None else 99 for m in ms)
        print(f"{label:<11}{len(ms):>3}{mean(m['shock'] for m in ms):>11.2f}{mean(m['need'] for m in ms):>14.3f}{mean(m['early'] for m in ms):>12.3f}{mean(m['late'] for m in ms):>11.3f}"
              f"{mean(m['early_ratio'] for m in ms):>15.2f}{mean(m['late_ratio'] for m in ms):>14.2f}{mean(m['lag'] is not None and m['lag'] <= 5 for m in ms):>14.2f}{(lags[len(lags) // 2] if lags[len(lags) // 2] < 99 else float('nan')):>11.0f}")
    if "stepcut" in met and "stepraise" in met:
        rng = random.Random(0)
        eps = {k: [(r["prices"][0], r["prices"][1]) for r in by[k]] for k in by}
        if min(len(eps["stepcut"]), len(eps["stepraise"])) < 3:
            print("\n  (fewer than 3 episodes in a shock arm: no contrast reported)")
        else:
            print()
            for key, nm in (("early_ratio", "early response / rational move"), ("late_ratio", "late response / rational move")):
                d, lo, hi, p = shock_contrast(eps["stepcut"], eps["stepraise"], eps.get("stepnone", []), mk, t0, key, rng)
                print(f"  cut minus raise, {nm}: {d:+.2f}  95% bootstrap CI [{lo:+.2f}, {hi:+.2f}]  p = {p:.3f}  (n = {len(eps['stepcut'])} / {len(eps['stepraise'])} / {len(eps.get('stepnone', []))})")
            if not eps.get("stepnone"):
                print("  WARNING: no stepnone control, so natural drift is not removed")
    print("\nmean own price relative to its pre-shock level, by round relative to the shock (-5..+14):")
    for label in sorted(by):
        paths = [[r["prices"][0][t0 + j] - mean(r["prices"][0][t0 - 10:t0]) for j in range(-5, min(15, len(r["prices"][0]) - t0))] for r in by[label]]
        print(f"  {label:<10}" + " ".join(f"{mean(p[j] for p in paths):+.2f}" for j in range(0, 20, 2)))
    print("  (columns: rounds -5, -3, -1, +1, +3, +5, +7, +9, +11, +13 after the step; the rival steps at round 0)")


def contrasts(args):
    """Pre-specified contrasts on tail collusion; each needs n >= --min-n per cell, else no test is reported."""
    groups = defaultdict(list)
    for (pfx, arm, p), rows in load_rows(args.pattern, args.llm_only).items():
        if pfx == args.prefix:
            groups[arm] += [r["tail_dp"] for r in rows]
    rng = random.Random(0)
    res = []
    for spec in args.contrasts.split(","):
        a, b = spec.split(":")
        x, y = groups.get(a, []), groups.get(b, [])
        if min(len(x), len(y)) < args.min_n:
            res.append((spec, len(x), len(y), None))
            continue
        d = mean(x) - mean(y)
        pool, k, hits, N = x + y, len(x), 0, 10000
        for _ in range(N):  # permutation test: shuffle which episodes belong to which cell
            rng.shuffle(pool)
            hits += abs(mean(pool[:k]) - mean(pool[k:])) >= abs(d) - 1e-12
        boots = sorted(mean(rng.choices(x, k=len(x))) - mean(rng.choices(y, k=len(y))) for _ in range(4000))
        res.append((spec, len(x), len(y), (d, boots[100], boots[3899], (hits + 1) / (N + 1))))
    tested = sorted((r[3][3], i) for i, r in enumerate(res) if r[3])
    holm = {}
    for rank, (p, i) in enumerate(tested):  # Holm step-down adjustment across the contrasts actually tested
        holm[i] = min(1.0, max([holm.get(j, 0) for _, j in tested[:rank]] + [p * (len(tested) - rank)]))
    print(f"{'contrast (A:B)':<24}{'nA':>4}{'nB':>4}{'A-B':>8}{'95% CI':>18}{'p':>8}{'p_holm':>8}")
    for i, (spec, na, nb, r) in enumerate(res):
        if r is None:
            print(f"{spec:<24}{na:>4}{nb:>4}   insufficient n (< {args.min_n} per cell): not tested")
        else:
            print(f"{spec:<24}{na:>4}{nb:>4}{r[0]:>8.2f}  [{r[1]:+.2f}, {r[2]:+.2f}]{r[3]:>8.3f}{holm[i]:>8.3f}")


def retries(args):
    """How often replies were truncated or retried: the share of trader-rounds whose accepted answer needed more than one attempt."""
    agg = defaultdict(lambda: {"calls": 0, "multi": 0, "att": 0, "trunc": 0, "fail": 0, "out": [], "lat": []})
    for f in sorted(glob.glob(args.pattern)):
        arm = None
        for line in open(f).read().splitlines():
            r = json.loads(line)
            if r["type"] == "header":
                arm = r.get("label") or r["cfg"]["arm"]
            elif r["type"] == "round" and arm:
                for tc in r["traders"]:
                    if not tc or "attempts" not in tc:
                        continue
                    g = agg[arm]
                    g["calls"] += 1
                    g["att"] += len(tc["attempts"])
                    g["multi"] += len(tc["attempts"]) > 1
                    g["fail"] += not tc["ok"]
                    for at in tc["attempts"]:
                        g["trunc"] += at.get("finish") == "length"
                        g["out"].append(at.get("out") or 0)
                        g["lat"].append(at.get("latency") or 0)
    if not agg:
        raise SystemExit("no trace files with model calls")
    print(f"{'arm':<14}{'calls':>7}{'att/call':>9}{'retried':>9}{'truncated':>10}{'failed':>7}{'out/att':>9}{'lat_s':>7}")
    for arm, g in sorted(agg.items()):
        print(f"{arm:<14}{g['calls']:>7}{g['att'] / g['calls']:>9.2f}{g['multi'] / g['calls']:>9.2f}{g['trunc'] / g['att']:>10.2f}{g['fail'] / g['calls']:>7.2f}"
              f"{mean(g['out']):>9.0f}{mean(g['lat']):>7.1f}")


def windows_of(rows, bench, tau):
    out = []
    for ep, r in enumerate(rows):
        W = len(r["prices"][0]) // len(r["windows"])
        for w in range(len(r["windows"])):
            wp = [r["prices"][0][w * W:(w + 1) * W], r["prices"][1][w * W:(w + 1) * W]]
            dp = lift(mean(wp[0] + wp[1]), bench.p_nash, bench.p_mono)
            out.append({"ep": ep, "wp": wp, "dp": dp, "y": int(dp >= tau)})
    return out


def cv_rule(wp):
    """Variance-screen style score: a low coefficient of variation of the average price looks collusive."""
    avg = [(a + b) / 2 for a, b in zip(*wp)]
    m = mean(avg)
    return 1 - min(1.0, math.sqrt(mean((v - m) ** 2 for v in avg)) / m / 0.1)


def flaggers(args):
    bench = Market().bench()
    cells = load_rows(args.pattern, args.llm_only)
    rows = [r for rs in cells.values() for r in rs]
    wins = windows_of(rows, bench, args.tau)
    pos = sum(w["y"] for w in wins)
    print(f"{len(rows)} episodes, {len(wins)} windows, {pos} collusive at tau={args.tau} ({pos / len(wins):.0%})")
    if pos in (0, len(wins)):
        raise SystemExit("one class only: nothing to separate; log runs with both collusive and competitive windows")
    eps = sorted({w["ep"] for w in wins})
    # price screen: trained on even-indexed episodes, scored on odd ones, and vice versa (no leakage)
    for w in wins:
        w["screen"] = None
    for fold in (0, 1):
        tr = [dict(prices=rows[e]["prices"], windows=rows[e]["windows"]) for e in eps if e % 2 != fold]
        for r, e in zip(tr, [e for e in eps if e % 2 != fold]):
            for i, win in enumerate(r["windows"]):
                win["y"] = [w["y"] for w in wins if w["ep"] == e][i]
        try:
            model, *_ = train(tr)
        except SystemExit:
            continue
        sf = ScreenFlagger(model)
        for w in wins:
            if w["ep"] % 2 == fold:
                w["screen"] = sf.assess(w["wp"])[0]
    scorers = {
        "oracle threshold": lambda w: 1 / (1 + math.exp(-10 * (w["dp"] - args.tau))),
        "variance rule": lambda w: cv_rule(w["wp"]),
        "price screen (cv)": lambda w: w["screen"] if w["screen"] is not None else 0.5,
        "random": lambda w: 0.5,
    }
    budgets = [float(b) for b in args.budgets.split(",")]
    rng = random.Random(0)
    print(f"{'flagger':<20}{'AUC':>6}  " + "  ".join(f"prec@{b:g}/rec@{b:g}" for b in budgets))
    for name, f in scorers.items():
        sc, ys = [f(w) for w in wins], [w["y"] for w in wins]
        cols = []
        for b in budgets:
            k = max(1, round(b * len(wins)))
            ps, rs = [], []
            for _ in range(200):  # ties broken at random, averaged
                order = sorted(range(len(wins)), key=lambda i: (-sc[i], rng.random()))[:k]
                hit = sum(ys[i] for i in order)
                ps.append(hit / k)
                rs.append(hit / pos)
            cols.append(f"{mean(ps):.2f}/{mean(rs):.2f}".rjust(len(f"prec@{b:g}/rec@{b:g}")))
        print(f"{name:<20}{auc(sc, ys):>6.2f}  " + "  ".join(cols))


def text(args):
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # (arm) -> (family, window) -> [hits, n]
    for f in sorted(glob.glob(args.pattern)):
        arm, win = None, 0
        for line in open(f).read().splitlines():
            r = json.loads(line)
            if r["type"] == "header":
                arm = r.get("label") or r["cfg"]["arm"]
                W = r["cfg"]["W"]
            elif r["type"] == "round" and arm:
                for tc in r["traders"]:
                    if not tc or "attempts" not in tc:
                        continue
                    final = (tc["attempts"][-1]["raw"] or "") if tc["attempts"] else ""
                    for fam, rx in WORDS.items():
                        cell = agg[arm][(fam, r["t"] // W)]
                        cell[0] += bool(re.search(rx, final, re.I))
                        cell[1] += 1
    if not agg:
        raise SystemExit("no trace files with model text (runs made before logging was added only have plans)")
    for arm, d in sorted(agg.items()):
        print(f"== {arm}")
        for fam in WORDS:
            ws = sorted(w for (f, w) in d if f == fam)
            print(f"  {fam:<12}" + " ".join(f"{d[(fam, w)][0] / d[(fam, w)][1]:.2f}" for w in ws))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("table", table), ("flaggers", flaggers), ("text", text), ("contrasts", contrasts), ("retries", retries), ("shocks", shocks), ("passthrough", passthrough), ("buyer", buyer), ("confirm", confirm), ("quality", quality), ("ratchet", ratchet), ("freeze", freeze)):
        s = sub.add_parser(name)
        s.add_argument("pattern")
        s.add_argument("--llm-only", action="store_true", help="keep only episodes that used model calls")
        if name == "shocks":
            s.add_argument("--shock-t", type=int, default=30)
        if name == "ratchet":
            s.add_argument("--exclude-arms", default="", help="comma-separated arm labels to leave out (e.g. bloyal,costup,costdown)")
        if name == "flaggers":
            s.add_argument("--tau", type=float, default=0.5)
            s.add_argument("--budgets", default="0.1,0.2,0.3")
        if name == "contrasts":
            s.add_argument("--prefix", default="P1")
            s.add_argument("--contrasts", default=PRIMARY)
            s.add_argument("--min-n", type=int, default=5)
        s.set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)
