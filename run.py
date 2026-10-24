"""Trader-vs-SEC pricing game on the Fish et al. Bertrand duopoly. Ground truth = price lift over Nash (no judge calls).

Every run lives in results/<run-id>/ : manifest.json, episodes/<cell>_s<seed>.json (finished episodes), traces/*.jsonl
(every round and model call, written live), summary.csv and table.txt. A run can be stopped at any time and resumed:

  export LLM_BASE_URL=... LLM_API_KEY=... LLM_EXTRA_BODY='{"chat_template_kwargs": {"thinking": false}}'
  python3 -m market_sim.run --run-id demo --traders llm --trader-model <id> --arms base,base-hidden,fine-quiet --episodes 2 --T 60 --history 20 --max-usd 2
  python3 -m market_sim.run --run-id demo --resume                  # finished episodes are skipped; interrupted ones continue
  python3 -m market_sim.run --run-id demo --resume --episodes 8     # extends the same run with more seeds (spend cap counts the whole run)

Arms: kind[-quiet][-hidden] with kind = base | notice | warn | gprompt | gharr | fine | sec.
  quiet = audits without any announcement; hidden = traders cannot see the rival's price; notice = announcement, no audits.
Cost-shock arms (both sellers are LLMs; cost changes for both at --cost-t, announced in the prompt):
  costup (--cost-lo -> --cost-hi), costdown (hi -> lo), costlo / costhi (no-shock controls).   Analysis: python3 -m market_sim.replay passthrough
Buyer arms (one big buyer holds --buyer-share of demand and splits it between the sellers): nobuyer | bpassive (logit split, placebo) |
  bsplit (50/50) | bloyal (always seller A) | bcheap (all to the cheaper seller) | bllm (LLM buyer).   Analysis: python3 -m market_sim.replay buyer
Free check of all of the above, no model calls: python3 -m market_sim.selftest
Free plumbing test: python3 -m market_sim.run --traders scripted --arms base,fine,sec --episodes 5 --T 40 --mock-deter 5
Fair targeted-vs-random comparison: add --match-sec. Offline analysis: python3 -m market_sim.replay --help
"""
from __future__ import annotations

import argparse
import atexit
import copy
import csv
import hashlib
import json
import os
import platform
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlparse

from market_sim.agents import BestResponder, GrimTrader, LLMBuyer, LLMFlagger, LLMTrader, ScriptedBuyer, StepRival, ThresholdFlagger, arm_note, buyer_told
from llm import BudgetExceeded
from market_sim.episode import Cfg, EpisodeInterrupted, run_episode
from market_sim.market import Market
from market_sim.metrics import bootstrap_diff, mean, se, summarize
from market_sim.qlearn import make_pair
from market_sim.replay import welfare
from market_sim.screens import ScreenFlagger

# arguments that may change between sessions of one run (extending it); everything else must match when resuming
LOOSE = {"resume", "force", "run_id", "max_usd", "concurrency", "progress", "episodes", "arms", "ps", "seed", "match_sec", "no_trace"}


# arm kinds of the two market-structure experiments
COST_KINDS = {"costup": ("lo", "hi"), "costdown": ("hi", "lo"), "costlo": ("lo", "lo"), "costhi": ("hi", "hi")}  # costlo/costhi = no-shock controls
BUYER_KINDS = ("nobuyer", "bpassive", "bsplit", "bloyal", "bcheap", "bllm")


def interleaved(configs, seeds, rng):
    """Every (cell, seed) episode in one seeded random order, so arms run side by side and endpoint drift cannot line up with an arm."""
    jobs = [(prefix, cfg, cfg.arm, s) for prefix, cfg in configs for s in seeds]
    rng.shuffle(jobs)
    return jobs


def read_records(path):
    out = []
    for line in open(path).read().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:  # a hard kill can leave one truncated last line
            break
    return out


def valid_prefix(recs, W):
    """Longest leading run of records that forms a consistent trace: header, rounds 0..n in order, windows after every W rounds."""
    if not recs or recs[0].get("type") != "header":
        return []
    out, t, w = [recs[0]], 0, 0
    for r in recs[1:]:
        if r["type"] == "round" and r["t"] == t:
            t += 1
        elif r["type"] == "window" and r["w"] == w and t == (w + 1) * W:
            w += 1
        else:
            break
        out.append(r)
    return out


def acquire_lock(run_dir):
    """One writer per run: a second process would corrupt the traces."""
    lock = run_dir / ".lock"
    if lock.exists():
        try:
            pid = int(lock.read_text())
            os.kill(pid, 0)
            sys.exit(f"run is already active (pid {pid}); stop it first or wait for it to finish")
        except (ValueError, ProcessLookupError):
            pass  # stale lock from a crashed process
        except PermissionError:
            sys.exit("run lock is held by another user's process")
    lock.write_text(str(os.getpid()))
    atexit.register(lambda: lock.unlink(missing_ok=True))


def trace_tokens(path):
    tin = tout = 0
    for r in read_records(path):
        calls = [at for tc in list(r.get("traders", [])) + [r.get("buyer")] if tc for at in tc.get("attempts", [])] if r["type"] == "round" else []
        if r["type"] == "window" and r.get("flagger"):
            calls.append((r["flagger"].get("usage") or {}))
        tin += sum(c.get("in", 0) for c in calls)
        tout += sum(c.get("out", 0) for c in calls)
    return tin, tout


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--traders", choices=["scripted", "llm", "q", "mixed"], default="scripted", help="q = Calvano Q-learning pair; mixed = one LLM against a scripted step-rival (arms stepcut / stepraise / stepnone)")
    a.add_argument("--rival-level", type=float, default=1.70, help="mixed: the scripted rival's price before the shock")
    a.add_argument("--shock-t", type=int, default=30, help="mixed: 0-based round at which the scripted rival's price steps")
    a.add_argument("--shock-size", type=float, default=0.25, help="mixed: size of the step (down for stepcut, up for stepraise)")
    a.add_argument("--cost-lo", type=float, default=1.0, help="cost arms (costup/costdown/costlo/costhi): the low unit cost")
    a.add_argument("--cost-hi", type=float, default=1.4, help="cost arms: the high unit cost")
    a.add_argument("--cost-t", type=int, default=30, help="cost arms: 0-based round at which cost changes (multiple of --W)")
    a.add_argument("--buyer-share", type=float, default=0.5, help="buyer arms (bpassive/bsplit/bloyal/bcheap/bllm): share of demand held by the big buyer")
    a.add_argument("--buyer-prompt", choices=["explore", "plain"], default="explore", help="bllm: 'plain' drops the 'explore many strategies' sentence (prompt-priming check)")
    a.add_argument("--q-train", type=int, default=1_000_000, help="pretraining periods per Q-learning pair")
    a.add_argument("--q-see", type=float, default=1.0, help="Q-learning: probability the rival's price is observed")
    a.add_argument("--q-online-alpha", type=float, default=0.15, help="Q-learning: learning rate during the episode (0 = frozen policy)")
    a.add_argument("--flagger", choices=["threshold", "screen", "llm"], default="threshold")
    a.add_argument("--screen-model", default="screens.json", help="from: python3 -m market_sim.screens")
    a.add_argument("--match-sec", action="store_true", help="after each sec cell, also run random audits at its realized audit rate")
    a.add_argument("--scripted-style", choices=["grim", "best"], default="grim")
    a.add_argument("--mock-deter", type=int, default=0, help="scripted grim trader: periods of competitive play after a fine")
    a.add_argument("--flagger-noise", type=float, default=0.0)
    a.add_argument("--trader-model", default="")
    a.add_argument("--flagger-model", default="")
    a.add_argument("--sec-info", choices=["benchmarks", "none"], default="benchmarks")
    a.add_argument("--arms", default="base,warn,fine,sec")
    a.add_argument("--ps", default="0.3", help="audit probabilities for fine/sec arms")
    a.add_argument("--prefix", default="P1", help="Fish prompt prefix: P1, P2 or both")
    a.add_argument("--episodes", type=int, default=10)
    a.add_argument("--T", type=int, default=40)
    a.add_argument("--W", type=int, default=10)
    a.add_argument("--tau", type=float, default=0.5)
    a.add_argument("--gamma", type=float, default=2.0)
    a.add_argument("--k", type=float, default=3.0)
    a.add_argument("--history", type=int, default=100)
    a.add_argument("--temperature", type=float, default=1.0)
    a.add_argument("--prompt-mode", choices=["single", "system"], default="single", help="Fish main runs: prefix and body in one user prompt")
    a.add_argument("--max-tokens", type=int, default=1500)
    a.add_argument("--tail", type=int, default=50)
    a.add_argument("--faithful", action="store_true", help="Fish/Garra protocol: T=300, history 100, last-50 evaluation, single-prompt")
    a.add_argument("--concurrency", type=int, default=8)
    a.add_argument("--progress", action="store_true", help="print one line per round to stderr")
    a.add_argument("--obs", choices=["full", "hidden"], default="full", help="hidden: traders do not see the rival's price (information-structure lever)")
    a.add_argument("--disclose-flagger", action="store_true", help="sec arm: tell traders that flagged markets are audited more often")
    a.add_argument("--no-trace", action="store_true", help="skip the full per-round trace files (results/traces/<run>/)")
    a.add_argument("--max-usd", type=float, default=None, help="abort LLM calls once estimated list-price spend reaches this")
    a.add_argument("--usd-per-m", default="0.30,1.20", help="input,output USD per million tokens used for --max-usd")
    a.add_argument("--fine-in-reward", action="store_true", help="deduct audit fines from the profit traders (and Q-learners) observe")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--run-id", default=None, help="results/<run-id>/ holds everything for this run (default: timestamp)")
    a.add_argument("--resume", action="store_true", help="continue an existing run; its saved settings are reused unless overridden")
    a.add_argument("--force", action="store_true", help="resume even though protocol settings or source files changed")
    args = a.parse_args()
    root = Path(__file__).resolve().parent.parent
    results_root = Path(os.environ.get("MARKET_SIM_RESULTS") or (root / "results"))  # the self-test points this at a temp dir so fake-model runs never sit next to real ones
    if args.resume:
        mpath = results_root / (args.run_id or "") / "manifest.json"
        if not args.run_id or not mpath.exists():
            sys.exit("--resume needs --run-id of an existing run")
        saved = json.loads(mpath.read_text())["args"]
        a.set_defaults(**{k: v for k, v in saved.items() if k not in ("resume", "force", "run_id")})
        args = a.parse_args()  # command-line flags still override the saved ones
    if args.faithful:
        args.T, args.history, args.tail, args.prompt_mode, args.temperature = 300, 100, 50, "single", 1.0

    mk = Market()
    bench = mk.bench()
    llm = None
    screen_model = json.load(open(args.screen_model)) if args.flagger == "screen" else None
    if args.traders in ("llm", "mixed") or args.flagger == "llm" or any(x.split("-")[0] == "bllm" for x in args.arms.split(",")):
        from llm import ChatClient
        rin, rout = (float(x) for x in args.usd_per_m.split(","))
        llm = ChatClient(max_usd=args.max_usd, usd_per_m_in=rin, usd_per_m_out=rout)

    def make_cfg(arm, p):
        kind, extra = arm.split("-")[0], {}
        if kind in COST_KINDS:
            lv = {"lo": args.cost_lo, "hi": args.cost_hi}
            extra = dict(cost0=lv[COST_KINDS[kind][0]], cost1=lv[COST_KINDS[kind][1]], cost_t=args.cost_t)
        elif kind in BUYER_KINDS and kind != "nobuyer":
            extra = dict(buyer=kind, theta=args.buyer_share)
        return Cfg(arm, p, args.gamma, args.k, args.T, args.W, args.tau, args.tail, args.fine_in_reward, **extra)

    configs = []
    for prefix in args.prefix.split(","):
        for arm in args.arms.split(","):
            for p in ([float(x) for x in args.ps.split(",")] if arm.split("-")[0] in ("fine", "sec") else [0.0]):
                configs.append((prefix, make_cfg(arm, p)))

    q_cache, q_lock = {}, threading.Lock()
    stop = threading.Event()  # set on Ctrl-C or budget exhaustion: running episodes halt at the next round boundary

    def q_pair(seed, see):
        with q_lock:  # pretrain once per (seed, observability) so every arm starts from the same learned pair
            if (seed, see) not in q_cache:
                q_cache[(seed, see)] = make_pair(mk, bench, seed, args.q_train, see, args.q_online_alpha)
            return copy.deepcopy(q_cache[(seed, see)])

    def episode(prefix, cfg, label, seed):
        key = f"{prefix}_{label}_p{cfg.p:.4g}_s{seed}"
        ep_file = ep_dir / f"{key}.json"
        if ep_file.exists():
            return json.loads(ep_file.read_text())  # finished in an earlier session
        hide = args.obs == "hidden" or "hidden" in cfg.arm.split("-")[1:]
        mods = cfg.arm.split("-")[1:]  # nomem = sellers keep no PLANS/INSIGHTS files; told = sellers are told the big buyer's allocation rule
        memory = "none" if "nomem" in mods else "files"
        if "told" in mods and not cfg.buyer:
            sys.exit(f"arm {cfg.arm}: 'told' only applies to buyer arms (bcheap, bsplit, bpassive, bloyal, bllm)")
        notes = lambda i: buyer_told(cfg.buyer, cfg.theta, i) if "told" in mods else ""
        rng = random.Random(f"agents-{seed}-{cfg.arm}")
        if args.traders == "q":
            traders = q_pair(seed, 0.0 if hide else args.q_see)
        elif args.traders == "mixed":
            delta = {"stepcut": -args.shock_size, "stepraise": args.shock_size}.get(cfg.arm.split("-")[0], 0.0)
            traders = [LLMTrader(llm, args.trader_model, prefix, "", mk, args.history, args.temperature, args.prompt_mode, args.max_tokens, hide_rival=hide, memory=memory),
                       StepRival(args.rival_level, args.shock_t, delta)]
        elif args.traders == "llm":
            note = arm_note(cfg.arm, cfg.p, cfg.gamma, cfg.W, args.disclose_flagger)
            traders = [LLMTrader(llm, args.trader_model, prefix, note, mk, args.history, args.temperature, args.prompt_mode, args.max_tokens, hide_rival=hide,
                                 memory=memory, product_notes=notes(i)) for i in range(2)]
        else:
            cls = GrimTrader if args.scripted_style == "grim" else BestResponder
            traders = [cls(mk, bench, random.Random(f"{seed}-{i}-{cfg.arm}"), deter=args.mock_deter) for i in range(2)]
        if args.flagger == "llm":
            flagger = LLMFlagger(llm, args.flagger_model, cfg.tau, args.sec_info)
        elif args.flagger == "screen":
            flagger = ScreenFlagger(screen_model)
        else:
            flagger = ThresholdFlagger(cfg.tau, rng, args.flagger_noise)
        buyer = None
        if cfg.buyer == "bllm":
            buyer = LLMBuyer(llm, args.trader_model, args.temperature, args.max_tokens, history=args.history, style=args.buyer_prompt)
        elif cfg.buyer:
            buyer = ScriptedBuyer(cfg.buyer, mk)
        fh, emit, resume_recs = None, None, None
        if not args.no_trace:
            tpath = trace_dir / f"{key}.jsonl"
            raw = read_records(tpath) if tpath.exists() else []
            recs = valid_prefix(raw, cfg.W)
            keep = []
            if recs and (args.traders in ("llm", "mixed") or cfg.buyer == "bllm") and any(r["type"] == "window" for r in recs):
                keep = recs[: max(i for i, r in enumerate(recs) if r["type"] == "window") + 1]  # up to the last completed window
                resume_recs = keep
            if raw[len(keep):]:  # paid-for rounds that must be replayed are kept aside so spend accounting stays exact
                with open(trace_dir / f"{key}.orphans.jsonl", "a") as of:
                    for r in raw[len(keep):]:
                        of.write(json.dumps(r, default=str) + "\n")
            fh = open(tpath, "w")
            lock = threading.Lock()

            def emit(rec):
                with lock:
                    fh.write(json.dumps(rec, default=str) + "\n")
                    fh.flush()  # a killed run keeps everything up to the last round

            for r in keep:
                emit(r)
            if not keep:
                emit({"type": "header", "prefix": prefix, "label": label, "cfg": asdict(cfg), "seed": seed, "traders": args.traders,
                      "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "hidden_rival": hide, "trader_settings": [
                          {k: getattr(tr, k) for k in ("model", "prefix", "history", "temperature", "mode", "max_tokens", "retries", "hide_rival", "memory", "product_notes") if hasattr(tr, k)}
                          for tr in traders]})
        try:
            row = run_episode(cfg, mk, traders, flagger, seed, parallel=args.traders in ("llm", "mixed"), progress=args.progress, emit=emit, resume=resume_recs, stop=stop, buyer=buyer)
        finally:
            if fh:
                fh.close()
        row["cell"] = {"prefix": prefix, "arm": cfg.arm, "label": label, "p": cfg.p, "seed": seed, "hidden_rival": hide, "memory": memory, "told": "told" in mods}
        tmp = ep_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(row))
        os.replace(tmp, ep_file)  # atomic: an episode file exists only if the episode finished
        return row

    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"  # pid keeps runs started in the same second apart
    run_dir = results_root / run_id
    ep_dir, trace_dir = run_dir / "episodes", run_dir / "traces"
    mpath = run_dir / "manifest.json"
    src = sorted(list((root / "market_sim").glob("*.py")) + [root / "llm.py"])
    code_hash = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in src}
    protocol = {k: v for k, v in vars(args).items() if k not in LOOSE}
    if mpath.exists():
        if not args.resume:
            sys.exit(f"run '{run_id}' already exists: add --resume to continue it, or choose a new --run-id")
        manifest = json.loads(mpath.read_text())
        diff = {k: (manifest["protocol"].get(k), v) for k, v in protocol.items() if manifest["protocol"].get(k) != v}
        changed = [n for n, h in code_hash.items() if manifest["source_sha256"].get(n) != h]
        if (diff or changed) and not args.force:
            sys.exit(f"cannot resume: settings changed {diff}; source files changed {changed}. --force resumes anyway (the run then mixes versions).")
    else:
        base_url = os.environ.get("LLM_BASE_URL", "")
        manifest = {
            "run_id": run_id, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "python": platform.python_version(),
            "market": {**asdict(mk), "bench": asdict(bench)},
            "llm_host": urlparse(base_url).hostname if base_url else None,  # host only; keys are never logged
            "llm_extra_body": os.environ.get("LLM_EXTRA_BODY"),
            "endpoint_models": llm.models() if llm else None,  # which weights the endpoint says it serves, to pin the model
            "llm_client": type(llm).__name__ if llm else None,  # 'ChatClient' for a real endpoint; anything else marks the run as synthetic
            "sessions": [],
        }
    manifest.update({"protocol": protocol, "args": vars(args), "source_sha256": code_hash})
    manifest["sessions"].append({"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "cmdline": " ".join(sys.argv), "resume": args.resume})
    ep_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    acquire_lock(run_dir)
    mpath.write_text(json.dumps(manifest, indent=2, default=str))
    rates = tuple(float(x) / 1e6 for x in args.usd_per_m.split(","))
    if llm is not None:  # spend of earlier sessions is rebuilt from the traces, so --max-usd bounds the whole run
        toks = [trace_tokens(f) for f in trace_dir.glob("*.jsonl")]
        llm.spent_usd = sum(i * rates[0] + o * rates[1] for i, o in toks)
        if toks:
            print(f"resuming run {run_id}: ${llm.spent_usd:.2f} already spent, {len(list(ep_dir.glob('*.json')))} episodes already finished")
    report = []

    def say(line):
        print(line)
        report.append(line)

    def write_summary():
        rows = []
        for f in sorted(ep_dir.glob("*.json")):
            r = json.loads(f.read_text())
            c = r["cell"]
            cs, wf = welfare(r, mk)
            dps = [w["dp"] for w in r["windows"]]
            rows.append({"prefix": c["prefix"], "label": c["label"], "arm": c["arm"], "p": c["p"], "seed": c["seed"], "hidden_rival": c["hidden_rival"],
                         "tail_dp": r["tail_dp"], "tail_dpi": r["tail_dpi"], "w0": dps[0], "post": mean(dps[1:]), "net_profit": r["net_profit"],
                         "fines": r["fines"], "consumer_surplus": cs, "welfare": wf, "parse_fail": r["parse_fail"],
                         "tok_in": r["tokens"]["in"], "tok_out": r["tokens"]["out"],
                         "est_usd": r["tokens"]["in"] * rates[0] + r["tokens"]["out"] * rates[1]})
        if rows:
            with open(run_dir / "summary.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        (run_dir / "table.txt").write_text("\n".join(report) + "\n")

    print(f"run: {run_dir}")
    say(f"benchmarks: Nash {bench.p_nash:.3f} / monopoly {bench.p_mono:.3f}; label: window lift >= {args.tau}")
    say(f"{'pfx':<4}{'arm':<16}{'p':>6}{'n':>4}{'dp_tail':>14}{'dpi_tail':>9}{'w0':>6}{'post':>6}{'d_vs_base [95% CI]':>26}"
        f"{'net_pi':>8}{'fines':>7}{'coll_w':>7}{'audit':>6}{'brier':>6}{'auc':>5}{'miss':>6}{'ff':>5}{'pf':>5}{'tok_in':>9}{'tok_out':>8}")
    base = {}

    ex = ThreadPoolExecutor(args.concurrency)  # one pool for all cells, so independent arms run side by side
    pending = {}

    def launch(prefix, cfg, label):
        pending[(prefix, label, cfg.p)] = [ex.submit(episode, prefix, cfg, label, s) for s in range(args.seed, args.seed + args.episodes)]

    def cell(prefix, cfg, label):
        if (prefix, label, cfg.p) not in pending:
            launch(prefix, cfg, label)
        rows = [f.result() for f in pending[(prefix, label, cfg.p)]]  # re-raises a failed episode's error here
        s = summarize(rows)
        if cfg.arm in ("base", "nobuyer"):
            base[prefix] = s["dp"]
        diff = "-"
        if prefix in base and cfg.arm not in ("base", "nobuyer"):
            if min(len(base[prefix]), len(s["dp"])) < 5:
                diff = "n<5: no CI"  # a bootstrap interval on fewer than 5 episodes would be meaningless
            else:
                d, lo, hi = bootstrap_diff(s["dp"], base[prefix])
                diff = f"{d:+.2f} [{lo:+.2f},{hi:+.2f}]"
        say(f"{prefix:<4}{label:<16}{cfg.p:>6.3f}{s['n']:>4}{mean(s['dp']):>8.2f}±{se(s['dp']):<5.2f}{mean(s['dpi']):>9.2f}{s['w0']:>6.2f}{s['post']:>6.2f}{diff:>26}"
            f"{s['net_profit']:>8.1f}{s['fines']:>7.0f}{s['coll_win']:>7.2f}{s['audit_rate']:>6.2f}{s['sec_brier']:>6.2f}"
            f"{s['sec_auc']:>5.2f}{s['miss']:>6.2f}{s['false_flag']:>5.2f}{s['parse_fail']:>5.1f}{s['tok_in']:>9}{s['tok_out']:>8}")
        write_summary()
        return s

    stopped = None
    try:
        for prefix, cfg in configs:
            pending[(prefix, cfg.arm, cfg.p)] = []
        for prefix, cfg, label, s in interleaved(configs, range(args.seed, args.seed + args.episodes), random.Random("episode-order")):
            pending[(prefix, label, cfg.p)].append(ex.submit(episode, prefix, cfg, label, s))  # matched cells (fine=sec) need their sec cell's audit rate, so they start later
        for prefix, cfg in configs:
            s = cell(prefix, cfg, cfg.arm)
            if cfg.arm.split("-")[0] == "sec" and args.match_sec:
                matched = "fine" + cfg.arm[3:]  # sec-quiet is matched against fine-quiet
                cell(prefix, Cfg(matched, s["audit_rate"], args.gamma, args.k, args.T, args.W, args.tau, args.tail, args.fine_in_reward), matched + "=sec")
    except BudgetExceeded as e:
        stopped = str(e)
    except (KeyboardInterrupt, EpisodeInterrupted):
        stopped = "interrupted"
    except Exception as e:  # e.g. the endpoint went down: whatever finished is kept and the run can be resumed
        stopped = f"error: {e!r}"
    finally:
        stop.set()  # running episodes halt at the next round boundary, leaving resumable traces
        ex.shutdown(wait=True, cancel_futures=True)
    write_summary()
    if llm is not None:
        print(f"estimated spend of this run so far (list price, no cache discount): ${llm.spent_usd:.2f}")
    if stopped:
        print(f"STOPPED ({stopped}). Everything finished is saved. Continue with:\n  python3 -m market_sim.run --run-id {run_id} --resume")
    else:
        print(f"done. Results: {run_dir}/summary.csv, table.txt, episodes/, traces/")


if __name__ == "__main__":
    main()
