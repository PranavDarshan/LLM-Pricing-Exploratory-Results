"""Offline self-test (no model calls, no network): python3 -m market_sim.selftest

Checks the market maths, the cost-shock and buyer experiments (scripted traders and a deterministic fake model), interrupt/resume
equality, spend accounting, the offline analyses on synthetic episodes with known answers, and that old results are unchanged.
"""
from __future__ import annotations

import contextlib
import glob
import io
import json
import os
import random
import shutil
import sys
import tempfile
import threading
import types
import zlib
from dataclasses import replace
from pathlib import Path

import llm as llm_mod
from llm import BudgetExceeded
from market_sim import replay, run
from market_sim.agents import (BestResponder, Ctx, LLMBuyer, LLMTrader, ScriptedBuyer,
                               ThresholdFlagger, paired_visibility_contexts)
from market_sim.episode import Cfg, run_episode
from market_sim.market import Market, lift

ROOT = Path(__file__).resolve().parent.parent
REAL_RESULTS = ROOT / "results"  # read-only here: the fake-model runs below go to a temp dir, never next to real data
RESULTS = Path(tempfile.mkdtemp(prefix="market_sim_selftest_"))
os.environ["MARKET_SIM_RESULTS"] = str(RESULTS)
FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


class FakeClient:
    """Deterministic stand-in for the model: the reply depends only on the prompt text."""
    seen, last = [], None

    def __init__(self, max_usd=None, usd_per_m_in=0.3, usd_per_m_out=1.2, **_):
        self.max_usd, self.rates, self.spent_usd = max_usd, (usd_per_m_in / 1e6, usd_per_m_out / 1e6), 0.0
        self._lock = threading.Lock()
        FakeClient.last = self

    def models(self):
        return ["fake-model"]

    def chat(self, model, system, user, temperature=0.7, max_tokens=2048):
        if self.max_usd is not None and self.spent_usd >= self.max_usd:
            raise BudgetExceeded("cap")
        h = zlib.crc32(user.encode())
        if "supplier A posts" in user:  # the buyer prompt
            share = (h % 101) / 100
            ans = f"{round(share * 100)}%" if h % 7 == 0 else f"{share:.2f}"
            text = f"My observations and thoughts:\nx\n\nNew content for PLANS.txt:\nbplan {h}\n\nNew content for INSIGHTS.txt:\nbins {h % 997}\n\nMy chosen share for supplier A:\n{ans}"
        else:
            text = (f"My observations and thoughts:\nx\n\nNew content for PLANS.txt:\nplan {h}\n\nNew content for INSIGHTS.txt:\nins {h % 997}\n\n"
                    f"My chosen price:\n{1.3 + (h % 700) / 1000:.2f}")
        n_in, n_out = len(user) // 4, len(text) // 4
        with self._lock:
            self.spent_usd += n_in * self.rates[0] + n_out * self.rates[1]
            FakeClient.seen.append(user)
        return text, {"in": n_in, "out": n_out, "finish": "stop", "latency": 0.0, "raw": text}


def go(argv, rid):
    sys.argv = ["run"] + argv
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        run.main()
    (RESULTS / rid / ".lock").unlink(missing_ok=True)  # same process: the lock would otherwise look held
    return buf.getvalue()


def episodes(rid):
    return {p.name: json.loads(p.read_text()) for p in sorted((RESULTS / rid / "episodes").glob("*.json"))}


def same_episode(a, b):
    ba, bb = a.get("buyer") or {}, b.get("buyer") or {}
    return (a["prices"] == b["prices"] and a["quantities"] == b["quantities"] and a["tokens"] == b["tokens"] and a["parse_fail"] == b["parse_fail"]
            and a["tail_dp"] == b["tail_dp"] and [w["dp"] for w in a["windows"]] == [w["dp"] for w in b["windows"]]
            and ba.get("share") == bb.get("share") and a.get("costs") == b.get("costs"))


def direct(cfg, buyer=None, seed=1):
    mk = Market()
    bench = mk.bench()
    tr = [BestResponder(mk, bench, random.Random(f"x-{i}")) for i in range(2)]
    return run_episode(cfg, mk, tr, ThresholdFlagger(cfg.tau, random.Random(0)), seed, buyer=buyer)


def test_visibility_contexts():
    print("paired rival-visibility prompts")
    ctx = Ctx(3, [1.5, 1.6, 1.7], [1.7, 1.8, 1.9], [10.0, 11.0, 12.0],
              [20.0, 21.0, 22.0], [], False)
    visible_ctx, hidden_ctx = paired_visibility_contexts(ctx, shock_round=2)
    single = LLMTrader(None, "fake", "P1", "", Market())
    visible_single, hidden_single = single.build_prompt(visible_ctx), single.build_prompt(hidden_ctx)
    split = LLMTrader(None, "fake", "P1", "", Market(), prompt_mode="split")
    visible_split, hidden_split = split.build_prompt(visible_ctx), split.build_prompt(hidden_ctx)
    check("single-call serialized prompts differ only at the selected shock price",
          visible_single[0] == hidden_single[0]
          and visible_single[1].replace(" - Competitor's price: 1.90\n", "", 1) == hidden_single[1]
          and " - Competitor's price: 1.70" in hidden_single[1]
          and " - Competitor's price: 1.80" in hidden_single[1]
          and "My quantity sold: 22.00" in visible_single[1] and "My quantity sold: 22.00" in hidden_single[1])
    check("split-call serialized prompts differ only at the selected shock price",
          visible_split[0] == hidden_split[0]
          and visible_split[1].replace(" - Competitor's price: 1.90\n", "", 1) == hidden_split[1])
    legacy_hidden = LLMTrader(None, "fake", "P1", "", Market(), hide_rival=True)
    _, legacy_prompt = legacy_hidden.build_prompt(ctx)
    check("legacy hide_rival still hides every rival price when no mask is supplied",
          "Competitor's price:" not in legacy_prompt)


def test_market():
    print("market maths")
    mk = Market()
    ok_same = ok_total = True
    for p1, p2 in ((1.5, 1.6), (2.0, 1.4), (1.0, 3.0), (1.47, 1.47)):
        base = mk.quantities(p1, p2)
        q = mk.quantities(p1, p2, 0.5, mk.logit_split(p1, p2))
        ok_same &= all(abs(x - y) < 1e-9 for x, y in zip(base, q))
        for s in (0.0, 0.3, 1.0):
            ok_total &= abs(sum(mk.quantities(p1, p2, 0.5, s)) - sum(base)) < 1e-9
    check("a buyer who splits by the logit shares reproduces the baseline quantities", ok_same)
    check("any split leaves total quantity unchanged (the buyer only moves volume between firms)", ok_total)
    q1, q2 = mk.quantities(1.5, 1.5, 0.5, 1.0)
    check("loyal buyer: seller A gets more than B at equal prices", q1 > q2 + 1)
    check("buyer volume equals theta x total inside volume", abs(mk.buyer_volume(1.5, 1.7, 0.4) - 0.4 * sum(mk.quantities(1.5, 1.7))) < 1e-9)
    b0, b1 = replace(mk, cost=1.0).bench(), replace(mk, cost=1.4).bench()
    check("benchmark moves with cost (Nash 1.473 -> about 1.82)", abs(b0.p_nash - 1.473) < 2e-3 and 1.80 < b1.p_nash < 1.84)


def test_scripted_episodes():
    print("scripted episodes")
    mk = Market()
    base = direct(Cfg("nobuyer", T=60, W=10))
    check("base arm quantities equal the market formula", all(abs(base["quantities"][i][t] - mk.quantities(base["prices"][0][t], base["prices"][1][t])[i]) < 1e-9 for i in (0, 1) for t in range(60)))
    passive = direct(Cfg("bpassive", T=60, W=10, buyer="bpassive", theta=0.5), ScriptedBuyer("bpassive", mk))
    check("passive buyer arm is identical to no-buyer (same prices; quantities equal up to rounding)",
          passive["prices"] == base["prices"] and passive["tail_dp"] == base["tail_dp"]
          and all(abs(x - y) < 1e-9 for i in (0, 1) for x, y in zip(passive["quantities"][i], base["quantities"][i])))
    loyal = direct(Cfg("bloyal", T=60, W=10, buyer="bloyal", theta=0.5), ScriptedBuyer("bloyal", mk), seed=0)
    ok = all(abs(loyal["quantities"][i][t] - mk.quantities(loyal["prices"][0][t], loyal["prices"][1][t], 0.5, 1.0)[i]) < 1e-9 for i in (0, 1) for t in range(60))
    check("loyal buyer arm (even seed): logged quantities follow the split formula, all volume to firm 0", ok and set(loyal["buyer"]["share"]) == {1.0} and not loyal["buyer"]["swap"])
    loyal1 = direct(Cfg("bloyal", T=60, W=10, buyer="bloyal", theta=0.5), ScriptedBuyer("bloyal", mk), seed=1)
    ok = all(abs(loyal1["quantities"][i][t] - mk.quantities(loyal1["prices"][0][t], loyal1["prices"][1][t], 0.5, 0.0)[i]) < 1e-9 for i in (0, 1) for t in range(60))
    check("loyal buyer arm (odd seed): the buyer's 'A' is firm 1, so all volume goes to firm 1", ok and set(loyal1["buyer"]["share"]) == {0.0} and loyal1["buyer"]["swap"] and set(loyal1["buyer"]["shown_share"]) == {1.0})
    check("buyer display order alternates with the seed (balanced across arms)", [direct(Cfg("bsplit", T=20, W=10, buyer="bsplit", theta=0.5), ScriptedBuyer("bsplit", mk), seed=s)["buyer"]["swap"] for s in range(4)] == [False, True, False, True])
    jobs = run.interleaved([("P1", Cfg(a)) for a in ("x", "y", "z")], range(8), random.Random("episode-order"))
    pos = {a: [i for i, j in enumerate(jobs) if j[2] == a] for a in ("x", "y", "z")}
    centres = [sum(v) / len(v) for v in pos.values()]
    check("episodes are queued in a mixed order: no arm is grouped early or late, all 24 jobs present", len(jobs) == 24 and max(centres) - min(centres) < 6 and len({(j[2], j[3]) for j in jobs}) == 24, str(centres))
    check("the queue order is reproducible", jobs == run.interleaved([("P1", Cfg(a)) for a in ("x", "y", "z")], range(8), random.Random("episode-order")))
    cheap = direct(Cfg("bcheap", T=60, W=10, buyer="bcheap", theta=0.5), ScriptedBuyer("bcheap", mk))
    P, s = cheap["prices"], cheap["buyer"]["share"]
    check("cheap buyer sends everything to the lower price", all((s[t] == 1.0) == (P[0][t] < P[1][t]) for t in range(60) if P[0][t] != P[1][t]))

    up = direct(Cfg("costup", T=60, W=10, cost0=1.0, cost1=1.4, cost_t=30))
    check("cost path: 1.0 for rounds 1-30, then 1.4", up["costs"][:30] == [1.0] * 30 and up["costs"][30:] == [1.4] * 30)
    b0, b1 = replace(mk, cost=1.0).bench(), replace(mk, cost=1.4).bench()
    wp = lambda w: [x for i in (0, 1) for x in up["prices"][i][w * 10:(w + 1) * 10]]
    check("windows are scored against the benchmark of their own cost level",
          abs(up["windows"][1]["dp"] - lift(sum(wp(1)) / 20, b0.p_nash, b0.p_mono)) < 1e-9 and abs(up["windows"][4]["dp"] - lift(sum(wp(4)) / 20, b1.p_nash, b1.p_mono)) < 1e-9)
    check("profits after the shock use the new cost", abs(up["profit"] - sum((up["prices"][i][t] - up["costs"][t]) * up["quantities"][i][t] for i in (0, 1) for t in range(60)) / 120) < 1e-9)
    try:
        direct(Cfg("costup", T=60, W=10, cost0=1.0, cost1=1.4, cost_t=25))
        check("a cost change inside a window is rejected", False)
    except ValueError:
        check("a cost change inside a window is rejected", True)


def test_fake_llm_runs():
    print("fake-model runs: prompts, resume, spend accounting")
    llm_mod.ChatClient = FakeClient
    for rid in ("_st_cost_a", "_st_cost_b", "_st_buy_a", "_st_buy_b"):
        shutil.rmtree(RESULTS / rid, ignore_errors=True)
    common = ["--traders", "llm", "--trader-model", "fake", "--episodes", "2", "--W", "10", "--history", "5", "--concurrency", "1", "--usd-per-m", "1000,0"]
    cost = ["--arms", "costup,costdown,costlo,costhi", "--T", "60", "--cost-t", "30"] + common
    buy = ["--arms", "nobuyer,bpassive,bsplit,bloyal,bcheap,bllm", "--T", "40", "--buyer-share", "0.5"] + common

    for tag, extra in (("cost", cost), ("buy", buy)):
        FakeClient.seen.clear()
        go(["--run-id", f"_st_{tag}_a"] + extra, f"_st_{tag}_a")
        full = episodes(f"_st_{tag}_a")
        spent = FakeClient.last.spent_usd
        n_expected = 8 if tag == "cost" else 12
        check(f"{tag}: all {n_expected} episodes finish", len(full) == n_expected, str(len(full)))
        go(["--run-id", f"_st_{tag}_b", "--max-usd", f"{0.45 * spent:.4f}"] + extra, f"_st_{tag}_b")
        part = episodes(f"_st_{tag}_b")
        check(f"{tag}: the spend cap interrupts the run partway", 0 <= len(part) < n_expected, str(len(part)))
        go(["--run-id", f"_st_{tag}_b", "--resume", "--max-usd", "1e9"], f"_st_{tag}_b")
        res = episodes(f"_st_{tag}_b")
        check(f"{tag}: resumed run equals the uninterrupted run, episode by episode", len(res) == n_expected and all(same_episode(full[k], res[k]) for k in full),
              str([k for k in full if k in res and not same_episode(full[k], res[k])]))
        tin = sum(r["tokens"]["in"] for r in res.values())
        tout = sum(r["tokens"]["out"] for r in res.values())
        traced = [run.trace_tokens(f) for f in (RESULTS / f"_st_{tag}_a" / "traces").glob("*[0-9].jsonl")]
        check(f"{tag}: tokens in the traces (incl. buyer calls) equal tokens in the episode files", (sum(a for a, _ in traced), sum(b for _, b in traced)) == (sum(r["tokens"]["in"] for r in full.values()), sum(r["tokens"]["out"] for r in full.values())))

    # cost arms: what the traders are told
    tr = [json.loads(line) for line in open(RESULTS / "_st_cost_a" / "traces" / "P1_costup_p0_s0.jsonl")]
    rounds = [r for r in tr if r["type"] == "round"]
    u = lambda t: rounds[t]["traders"][0]["user"]
    check("before the shock: cost line shows $1, no notice, no per-round cost rows with 1.4",
          "cost I pay to produce each unit is $1." in u(29) and "NOTICE" not in u(29) and "cost per unit: 1.4" not in u(29))
    check("at the shock: cost line shows $1.4, history keeps $1, notice names the round",
          "cost I pay to produce each unit is $1.4." in u(30) and "My cost per unit: 1" in u(30) and "NOTICE (round 31): the cost per unit changed from $1 to $1.4" in u(30))
    check("the notice disappears after three rounds", "NOTICE" in u(32) and "NOTICE" not in u(33))
    ctrl = [json.loads(line) for line in open(RESULTS / "_st_cost_a" / "traces" / "P1_costlo_p0_s0.jsonl")]
    check("control arm: cost never changes and nothing is announced", all("NOTICE" not in r["traders"][0]["user"] for r in ctrl if r["type"] == "round"))
    dn = episodes("_st_cost_a")["P1_costdown_p0_s0.json"]
    check("costdown arm: 1.4 then 1.0", dn["costs"][0] == 1.4 and dn["costs"][-1] == 1.0)

    # buyer arms
    ll = episodes("_st_buy_a")["P1_bllm_p0_s0.json"]
    check("LLM buyer: one share per round, all within [0, 1]", len(ll["buyer"]["share"]) == 40 and all(0 <= x <= 1 for x in ll["buyer"]["share"]))
    check("LLM buyer: reply formats (decimal and percent) both parse", len(set(round(x, 2) for x in ll["buyer"]["share"])) > 5)
    bt = [json.loads(line) for line in open(RESULTS / "_st_buy_a" / "traces" / "P1_bllm_p0_s0.jsonl")]
    br = [r for r in bt if r["type"] == "round"]
    check("buyer sees this round's posted prices before choosing", f"supplier A posts ${br[7]['prices'][0]:.2f}" in br[7]["buyer"]["user"])
    b1 = [json.loads(line) for line in open(RESULTS / "_st_buy_a" / "traces" / "P1_bllm_p0_s1.jsonl")]
    br1 = [r for r in b1 if r["type"] == "round"]
    check("odd seed: the buyer's 'supplier A' is firm 1", f"supplier A posts ${br1[7]['prices'][1]:.2f}" in br1[7]["buyer"]["user"] and br1[7]["buyer_swap"])
    e1 = episodes("_st_buy_a")["P1_bllm_p0_s1.json"]["buyer"]
    e0 = ll["buyer"]
    check("logged share to firm 0 = 1 - share shown for odd seeds, = share shown for even seeds", all(abs(a + b - 1) < 1e-9 for a, b in zip(e1["share"], e1["shown_share"])) and e0["share"] == e0["shown_share"])
    check("buyer history lists past shares", "Share bought from A:" in br[7]["buyer"]["user"] and "My average price paid:" in br[7]["buyer"]["user"])
    seller_prompt = br[7]["traders"][0]["user"]
    check("sellers are not told about the buyer", "buyer" not in seller_prompt.lower() and "supplier" not in seller_prompt.lower())
    check("no-buyer arm has no buyer calls", all(r.get("buyer") is None for r in (json.loads(l) for l in open(RESULTS / "_st_buy_a" / "traces" / "P1_nobuyer_p0_s0.jsonl")) if r["type"] == "round"))

    # offline analyses run on these logs
    for fn, pat, name in ((replay.passthrough, "_st_cost_a", "passthrough"), (replay.buyer, "_st_buy_a", "buyer"), (replay.confirm, "_st_buy_a", "confirm (buyer)"), (replay.confirm, "_st_cost_a", "confirm (cost)"), (replay.ratchet, "_st_buy_a", "ratchet")):
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                fn(types.SimpleNamespace(pattern=str(RESULTS / pat / "episodes" / "*.json"), llm_only=False))
            check(f"replay {name} runs on the logs", len(buf.getvalue()) > 200)
        except Exception as e:  # noqa: BLE001
            check(f"replay {name} runs on the logs", False, repr(e))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        replay.confirm(types.SimpleNamespace(pattern=str(RESULTS / "_st_buy_a" / "episodes" / "*.json"), llm_only=False))
    check("confirm refuses to call 2 episodes per arm confirmatory", "NOT CONFIRMATORY" in buf.getvalue())
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        replay.quality(types.SimpleNamespace(pattern=str(RESULTS / "_st_buy_a" / "traces" / "*[0-9].jsonl")))
    check("quality audit runs and reports per arm for sellers and the buyer", "bllm" in buf.getvalue() and "buyer" in buf.getvalue() and "QUALITY:" in buf.getvalue())
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        replay.table(types.SimpleNamespace(pattern=str(RESULTS / "_st_cost_a" / "episodes" / "*.json"), llm_only=False))
    check("replay table runs on cost-shock logs (cost-aware welfare)", "costup" in buf.getvalue())
    for rid in ("_st_cost_a", "_st_cost_b", "_st_buy_a", "_st_buy_b"):
        shutil.rmtree(RESULTS / rid, ignore_errors=True)


def test_freeze_runs():
    print("freeze experiment: arm modifiers, prompts and analysis (fake model)")
    llm_mod.ChatClient = FakeClient
    rid = "_st_freeze"
    shutil.rmtree(RESULTS / rid, ignore_errors=True)
    common = ["--traders", "llm", "--trader-model", "fake", "--episodes", "2", "--T", "60", "--W", "10", "--history", "5", "--concurrency", "1", "--buyer-share", "0.5"]
    go(["--run-id", rid, "--arms", "bcheap,bcheap-nomem,bcheap-told,bcheap-nomem-told,nobuyer,nobuyer-nomem"] + common, rid)
    eps = episodes(rid)
    check("all 12 episodes of the 6 freeze arms finish", len(eps) == 12, str(len(eps)))
    rd = lambda arm: [json.loads(l) for l in open(RESULTS / rid / "traces" / f"P1_{arm}_p0_s0.jsonl") if l.strip()]
    u = lambda arm, t: [r for r in rd(arm) if r["type"] == "round"][t]["traders"][0]["user"]
    check("bcheap: files shown, buyer not mentioned", "PLANS.txt" in u("bcheap", 5) and "large buyer" not in u("bcheap", 5))
    check("bcheap-nomem: no files anywhere in the prompt, market data and price template still present",
          "PLANS" not in u("bcheap-nomem", 5) and "INSIGHTS" not in u("bcheap-nomem", 5) and "previous copy of yourself" not in u("bcheap-nomem", 5)
          and "MARKET_DATA" in u("bcheap-nomem", 5) and "My chosen price:" in u("bcheap-nomem", 5) and "Round 5:" in u("bcheap-nomem", 5))
    check("bcheap-told: sellers are told the buyer's rule in the product information", "Half of all demand comes from one large buyer who buys everything from whichever of the two sellers posts the strictly lower price" in u("bcheap-told", 5))
    check("bcheap-nomem-told: both modifiers apply", "PLANS" not in u("bcheap-nomem-told", 5) and "large buyer" in u("bcheap-nomem-told", 5))
    check("nobuyer-nomem: memory ablation works without a buyer", "PLANS" not in u("nobuyer-nomem", 5) and eps["P1_nobuyer-nomem_p0_s0.json"]["buyer"] is None)
    hdr = rd("bcheap-nomem")[0]
    check("trace header and episode cell record the memory setting", hdr["trader_settings"][0]["memory"] == "none" and eps["P1_bcheap-nomem_p0_s0.json"]["cell"]["memory"] == "none" and eps["P1_bcheap_p0_s0.json"]["cell"]["memory"] == "files")
    check("nomem episodes still parse a price every round", all(eps[k]["parse_fail"] == 0 for k in eps if "nomem" in k))
    check("nomem arm: the saved plan stays empty (nothing is carried between rounds)", all(p == "(none yet)" for w in eps["P1_bcheap-nomem_p0_s0.json"]["plans"] for p in w))
    try:
        go(["--run-id", "_st_freeze_bad", "--arms", "base-told"] + common, "_st_freeze_bad")
        check("'told' on a non-buyer arm is rejected", False)
    except SystemExit as e:
        check("'told' on a non-buyer arm is rejected", "told" in str(e))
    shutil.rmtree(RESULTS / "_st_freeze_bad", ignore_errors=True)
    pat = str(RESULTS / rid / "episodes" / "*.json")
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            replay.freeze(types.SimpleNamespace(pattern=pat, llm_only=False))
        check("freeze REFUSES a run made with the fake model client (manifest llm_client != ChatClient)", False)
    except SystemExit as e:
        check("freeze REFUSES a run made with the fake model client (manifest llm_client != ChatClient)", "no usable" in str(e) and "synthetic" in buf.getvalue())
    man = RESULTS / rid / "manifest.json"
    m = json.loads(man.read_text())
    check("the manifest records the fake client", m["llm_client"] == "FakeClient")
    m["llm_client"] = "ChatClient"  # test-only: pretend these are real episodes so the analysis path can be exercised
    man.write_text(json.dumps(m))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        replay.freeze(types.SimpleNamespace(pattern=pat, llm_only=False))
    out = buf.getvalue()
    check("replay freeze runs: table, text markers, one PRIMARY, Holm-adjusted secondaries, robustness grid",
          "bcheap-nomem" in out and "optimum=" in out and out.count("[PRIMARY]") == 1 and "Holm p" in out and "winner-take-all demand" in out and "buyer rule disclosed" in out
          and "NOT CONFIRMATORY" in out and "robustness grid" in out and "/18 specifications" in out and "fail" in out, out[-600:])
    check("freeze: the demand consistency check passes on genuine simulator output", "does not reproduce" not in out)
    ep = RESULTS / rid / "episodes" / "P1_bcheap_p0_s0.json"
    row = json.loads(ep.read_text())
    row["buyer"]["kind"] = "bloyal"  # corrupt the recorded demand rule: the analysis must notice, not silently compute wrong regrets
    ep.write_text(json.dumps(row))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            replay.freeze(types.SimpleNamespace(pattern=pat, llm_only=False))
        check("freeze ABORTS when the recomputed demand does not reproduce the logged profits", False)
    except SystemExit as e:
        check("freeze ABORTS when the recomputed demand does not reproduce the logged profits", "does not reproduce" in str(e))
    row["buyer"]["kind"] = "bcheap"
    row["parse_fail"] = 3
    ep.write_text(json.dumps(row))
    (RESULTS / rid / "traces" / "P1_bcheap_p0_s0.jsonl").unlink()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        replay.freeze(types.SimpleNamespace(pattern=pat, llm_only=False))
    check("an episode with model failures but no trace to locate them is dropped, not guessed", "model failures but no trace" in buf.getvalue())
    shutil.rmtree(RESULTS / rid, ignore_errors=True)


def test_no_fake_data():
    print("no synthetic data in the real results directory")
    check("the self-test's run directory is outside results/", REAL_RESULTS not in RESULTS.parents and RESULTS != REAL_RESULTS)
    stray = [p for p in REAL_RESULTS.glob("_st_*")] if REAL_RESULTS.exists() else []
    check("no '_st_*' (self-test) run directories exist under results/", not stray, str(stray))
    fake = []
    for man in REAL_RESULTS.glob("*/manifest.json") if REAL_RESULTS.exists() else []:
        mj = json.loads(man.read_text())
        if mj.get("llm_client") not in (None, "ChatClient") or mj.get("args", {}).get("trader_model") == "fake":
            fake.append(man.parent.name)
    check("no run under results/ was produced by a fake model client", not fake, str(fake))


def test_freeze_metrics():
    print("freeze metrics on hand-built episodes with known answers")
    mk = Market()
    cheap = lambda p0, p1: {"prices": [p0, p1], "buyer": {"kind": "bcheap", "theta": 0.5}, "costs": None}
    loser = cheap([2.5] * 20 + [1.99] * 40, [2.4] * 20 + [1.90] * 40)  # seller 0 undercut by 9 cents for 40 rounds, seller 1 wins the buyer
    m = replay.freeze_metrics(loser)
    f0 = replay.firm_profit(loser, 0, 1.99, 1.90, 59)
    check("bcheap profit uses the cheapest-buyer rule (the loser gets none of the buyer's volume)", abs(f0 - mk.profits(1.99, 1.90, 0.5, 0.0)[0]) < 1e-9)
    check("best response to 1.90 under bcheap undercuts by one cent and earns far more", replay.best_response_profit(loser, 0, 1.90, 59) > 2.5 * f0)
    check("frozen loser: stuck = 0.5 (one of two sellers), both frozen, t_freeze = 21", abs(m["stuck"] - 0.5) < 1e-9 and m["frozen"] == 1.0 and abs(m["t_freeze"] - 21) < 1e-9, str(m))
    bad = {(0, t) for t in range(30, 60, 10)}  # the model failed to answer in rounds 31, 41, 51: the repeated price was not its choice
    mb_ = replay.freeze_metrics(loser, bad=bad)
    check("rounds the model failed to answer break a hold: the loser's stuck share falls and it no longer counts as frozen", mb_["stuck"] < m["stuck"] - 0.2 and mb_["frozen"] == 0.5, str(mb_))
    check("asym is 0 for identical sellers and large when one seller gets the whole buyer", abs(replay.freeze_metrics(cheap([1.45] * 60, [1.45] * 60))["asym"]) < 1e-9 and m["asym"] > 0.4)
    good = {"prices": [[1.6] * 60, [1.7] * 60], "quantities": [[0.0] * 60, [0.0] * 60], "buyer": {"kind": "bcheap", "theta": 0.5}, "costs": None}
    for t in range(60):
        q = mk.quantities(1.6, 1.7, 0.5, 1.0)
        good["quantities"][0][t], good["quantities"][1][t] = q
    check("demand check passes when quantities follow the rule and fails when theta is wrong", replay.check_demand(good) and not replay.check_demand({**good, "buyer": {"kind": "bcheap", "theta": 0.3}}))
    tie = cheap([1.45] * 60, [1.45] * 60)
    check("both sellers tied at Nash under bcheap are both stuck (a one-cent undercut wins the buyer)", abs(replay.freeze_metrics(tie)["stuck"] - 1.0) < 1e-9)
    below = cheap([0.98] * 60, [1.00] * 60)
    mb = replay.freeze_metrics(below)
    check("below-cost seller: below_cost 0.5", abs(mb["below_cost"] - 0.5) < 1e-9)
    pn = mk.bench().p_nash
    best = {"prices": [[pn] * 60, [pn] * 60], "buyer": None, "costs": None}
    mbest = replay.freeze_metrics(best)
    check("two frozen mutual best responders (Nash, logit demand) are frozen but not stuck, regret ~0", mbest["frozen"] == 1.0 and mbest["stuck"] == 0.0 and mbest["regret"] < 0.01, str(mbest))
    moving = {"prices": [[1.5 + 0.1 * (t % 2) for t in range(60)], [1.7] * 60], "buyer": None, "costs": None}
    check("a seller that keeps moving is never 'held', so never stuck", replay.freeze_metrics(moving)["stuck"] == 0.0)
    llmb = {"prices": [[1.5] * 60, [1.6] * 60], "buyer": {"kind": "bllm", "theta": 0.5}, "costs": None}
    check("bllm episodes report NaN stuck/regret (no counterfactual buyer reply)", replay.freeze_metrics(llmb)["stuck"] != replay.freeze_metrics(llmb)["stuck"])
    check("allocation rules: bsplit 0.5, bpassive = logit split", replay.allocation("bsplit", 1.5, 1.9) == 0.5 and abs(replay.allocation("bpassive", 1.5, 1.9) - mk.logit_split(1.5, 1.9)) < 1e-9)


def synth(kind, costs, price_fn):
    P = [[price_fn(t) for t in range(60)] for _ in range(2)]
    return {"prices": P, "costs": costs, "quantities": P, "tail_dp": 0.0, "profit": 0.0}


def test_analysis():
    print("analyses on synthetic episodes with known answers")
    rng = random.Random(3)
    same = [rng.gauss(0, 1) for _ in range(12)]
    d0, p0 = replay.perm_test(same, [x for x in same])
    sep = replay.perm_test([rng.gauss(2, 0.3) for _ in range(12)], [rng.gauss(0, 0.3) for _ in range(12)])
    check("permutation test: identical groups give p = 1, clearly separated groups give p < 0.001", abs(p0 - 1) < 1e-9 and sep[1] < 0.001)
    null_hits = sum(replay.perm_test([rng.gauss(0, 1) for _ in range(10)], [rng.gauss(0, 1) for _ in range(10)], n=300, seed=i)[1] < 0.05 for i in range(300))
    check("permutation test: false-positive rate under no effect is near 5%", 0.02 <= null_hits / 300 <= 0.09, f"{null_hits / 300:.3f}")
    r_move = {"prices": [[1.5] * 30 + [1.7] * 30] * 2, "costs": [1.0] * 30 + [1.4] * 30}
    r_stay = {"prices": [[1.5] * 60] * 2, "costs": [1.0] * 30 + [1.4] * 30}
    check("reaction measure: 1 when both firms move after the shock, 0 when neither does", replay.reaction(r_move) == 1.0 and replay.reaction(r_stay) == 0.0)
    pb = lambda swap, share: {"prices": [[1.5] * 60, [1.5] * 60], "buyer": {"share": [share] * 60, "swap": swap}, "tail_dp": 0.0, "profit": 1.0}
    check("position bias is measured in the buyer's own display order", abs(replay.buyer_metrics(pb(False, 1.0))["disp_a"] - 1.0) < 1e-9 and abs(replay.buyer_metrics(pb(True, 0.0))["disp_a"] - 1.0) < 1e-9)
    mk = Market()
    b0, b1 = replace(mk, cost=1.0).bench(), replace(mk, cost=1.4).bench()
    up_costs = [1.0] * 30 + [1.4] * 30
    track = synth("up", up_costs, lambda t: b0.p_nash if t < 30 else b1.p_nash)
    frozen = synth("up", up_costs, lambda t: b0.p_nash)
    m1, m2 = replay.cost_metrics(track, mk), replay.cost_metrics(frozen, mk)
    check("pass-through is 1 for a market that tracks the Nash price and 0 for a frozen one", abs(m1["pass"] - 1) < 1e-9 and abs(m2["pass"]) < 1e-9, f"{m1['pass']:.3f} {m2['pass']:.3f}")
    check("a market that tracks Nash has lag 1, no overshoot, no churn after the first move", m1["lag"] == 1 and abs(m1["overshoot"]) < 1e-9)
    slow = synth("up", up_costs, lambda t: b0.p_nash if t < 30 else b0.p_nash + (b1.p_nash - b0.p_nash) * min(1.0, (t - 29) / 10))
    check("a gradual response shows a longer lag", 4 <= replay.cost_metrics(slow, mk)["lag"] <= 7)
    over = synth("up", up_costs, lambda t: b0.p_nash if t < 30 else b1.p_nash + (0.2 if t < 35 else 0.0))
    check("overshoot is detected", replay.cost_metrics(over, mk)["overshoot"] > 0.4)
    drift = replay.cost_metrics(synth("lo", [1.0] * 60, lambda t: 1.5 + (0.1 if t >= 30 else 0)), mk)
    check("control episodes report their drift", abs(drift["move"] - 0.1) < 1e-9)
    adj = replay.cost_metrics(track, mk, drift=0.1)
    check("drift is netted out of the pass-through", abs(adj["pass"] - (1 - 0.1 / (b1.p_nash - b0.p_nash))) < 1e-9)
    down = synth("down", [1.4] * 30 + [1.0] * 30, lambda t: b1.p_nash if t < 30 else b0.p_nash)
    check("pass-through has the right sign for cost decreases", abs(replay.cost_metrics(down, mk)["pass"] - 1) < 1e-9)

    def brow(shares, pa, pb, dp):
        return {"prices": [[pa] * 60, [pb] * 60], "buyer": {"share": shares}, "tail_dp": dp, "profit": 20.0}
    t = lambda r: replay.buyer_metrics(r)["type"]
    check("loyal buyer + cheaper favoured seller -> buyer-seller pairing", t(brow([1.0] * 60, 1.6, 1.8, 0.5)).startswith("buyer-seller pairing"))
    check("loyal buyer with no discount -> captured buyer", t(brow([1.0] * 60, 1.9, 1.8, 0.8)).startswith("captured buyer"))
    check("50/50 buyer with both prices high -> seller-seller cartel", t(brow([0.5] * 60, 1.9, 1.9, 0.8)).startswith("seller-seller cartel"))
    check("alternating all-or-nothing buyer -> switching buyer", t(brow([1.0, 0.0] * 30, 1.9, 1.9, 0.6)) == "switching buyer")
    check("low prices -> competitive", t(brow([0.5] * 60, 1.5, 1.5, 0.05)) == "competitive")
    m = replay.buyer_metrics(brow([1.0] * 60, 1.6, 1.8, 0.5))
    check("buyer metrics: discount, concentration and saving", abs(m["discount"] + 0.2) < 1e-9 and abs(m["conc"] - 1) < 1e-9 and abs(m["saving"] - 0.1) < 1e-9)
    pa = [1.5 if t_ % 2 == 0 else 1.7 for t_ in range(60)]
    sh = [1.0 if a < 1.6 else 0.0 for a in pa]
    r = {"prices": [pa, [1.6] * 60], "buyer": {"share": sh}, "tail_dp": 0.0, "profit": 1.0}
    check("reward slope: share to A rises by 5.0 per +1.00 in (B - A) for a buyer who always picks the cheaper seller", abs(replay.buyer_metrics(r)["slope"] - 5.0) < 1e-9)


def test_ratchet():
    print("cut/raise persistence analysis: definitions and synthetic worlds with known answers")
    rival = [1.7] * 60
    ev = lambda own, **kw: replay.move_events({"prices": [own, rival]}, **kw)
    cut = [1.9] * 10 + [1.7] * 50
    e = ev(cut)
    check("a cut of 0.2 that stays: one event, in place, toward the best response", len(e) == 1 and e[0][:4] == ("cut", True, "toward", 10))
    e = ev([1.4] * 10 + [1.6] * 2 + [1.4] * 48)
    check("a raise that is retracted: not in place, and its retraction is not counted as a new cut", len(e) == 1 and e[0][0] == "raise" and e[0][1] is False)
    check("without the fresh-move rule the retraction would count as a sticky cut", len(ev([1.4] * 10 + [1.6] * 2 + [1.4] * 48, fresh=0)) == 2)
    both = {"prices": [cut, [1.9] * 10 + [1.7] * 50]}
    check("simultaneous cuts are not 'initiating' (strict) but are counted when strict=False", replay.move_events(both) == [] and len(replay.move_events(both, strict=False)) == 2)
    check("move size threshold", ev([1.9] * 10 + [1.82] * 50) == [] and len(ev([1.9] * 10 + [1.82] * 50, thr=0.05)) == 1)
    costs = [1.0] * 30 + [1.4] * 30
    r_early, r_late = {"prices": [[1.9] * 10 + [1.7] * 50, rival], "costs": costs}, {"prices": [[1.9] * 20 + [1.7] * 40, rival], "costs": costs}
    check("an event window never crosses a cost change", len(replay.move_events(r_early)) == 1 and replay.move_events(r_late) == [])
    mkev = lambda kind, st, w: (kind, st, w, 10, 0.0)
    attractor = [[mkev("cut", True, "toward")] * 20 + [mkev("raise", True, "toward")] * 10 + [mkev("raise", False, "away")] * 10]
    check("attractor world (hand-built): raw gap +0.5 but exactly 0 within moves toward the best response", abs(replay.ratchet_gap(attractor) - 0.5) < 1e-9 and abs(replay.ratchet_gap(attractor, lambda x: x[2] == "toward")) < 1e-9)
    planted = [[mkev("cut", True, w) for w in ("toward", "away")] * 10 + [mkev("raise", s, w) for s, w in ((True, "toward"), (False, "toward"), (False, "toward"), (False, "toward"), (False, "away"))] * 4]
    check("planted ratchet (hand-built): gap stays large within moves toward the best response", replay.ratchet_gap(planted, lambda x: x[2] == "toward") > 0.6)
    br = Market().best_response(1.7)

    def world(planted_ratchet, seed):
        rng, p, own, pin, wait, left = random.Random(seed), 3.0, [], None, 0, 0
        for t in range(60):
            own.append(p)
            if wait:  # one round after a planted raise the price goes back to the old level and stays there
                wait -= 1
                left = 14 if wait == 0 else left
            if left > 0:
                p, left = pin + rng.gauss(0, 0.01), left - 1
                continue
            nxt = p + 0.6 * (br - p) + rng.gauss(0, 0.12)
            if planted_ratchet and nxt - p >= 0.10 and rng.random() < 0.85:
                pin, wait = p, 1
            p = nxt
        return {"prices": [own, [1.7] * 60]}
    sym = [replay.move_events(world(False, s)) for s in range(60)]
    check("symmetric attractor world: raw gap near 0", abs(replay.ratchet_gap(sym)) < 0.3, f"{replay.ratchet_gap(sym):+.2f}")
    pl = [replay.move_events(world(True, s)) for s in range(60)]
    pl0 = [replay.move_events(world(True, s), fresh=0) for s in range(60)]
    n_cut = lambda evl: sum(1 for ev_ in evl for x in ev_ if x[0] == "cut")
    check("planted-ratchet world: raw gap clearly positive", replay.ratchet_gap(pl) > 0.4, f"{replay.ratchet_gap(pl):+.2f}")
    check("planted-ratchet world: counting retractions as cuts would add spurious sticky cuts", n_cut(pl0) > n_cut(pl), f"{n_cut(pl0)} vs {n_cut(pl)}")


def test_unchanged_old_results():
    print("old results are unchanged")
    files = sorted(glob.glob(str(REAL_RESULTS / "live-01-info-and-quiet" / "episodes" / "*.json")))
    if not files:
        print("  (no old results here, skipped)")
        return
    mk, ok = Market(), True
    for f in files:
        r = json.load(open(f))
        P, T = r["prices"], len(r["prices"][0])
        cs = [mk.consumer_surplus(P[0][t], P[1][t]) for t in range(T)]
        pi = [sum(mk.profits(P[0][t], P[1][t])) for t in range(T)]
        old = (sum(cs) / T, sum(c + p for c, p in zip(cs, pi)) / T)
        new = replay.welfare(r, mk)
        ok &= abs(old[0] - new[0]) < 1e-9 and abs(old[1] - new[1]) < 1e-9
    check(f"welfare of the {len(files)} live-01 episodes is identical to the old formula", ok)
    check("old runs without cost/buyer fields still load", all("cell" in json.load(open(f)) for f in files))


if __name__ == "__main__":
    test_market()
    test_visibility_contexts()
    test_scripted_episodes()
    test_analysis()
    test_ratchet()
    test_freeze_metrics()
    test_unchanged_old_results()
    test_fake_llm_runs()
    test_freeze_runs()
    test_no_fake_data()
    shutil.rmtree(RESULTS, ignore_errors=True)
    print("\nRESULT:", "PASS" if not FAILS else f"FAIL ({len(FAILS)}): {FAILS}")
    sys.exit(1 if FAILS else 0)
