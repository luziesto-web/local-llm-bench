"""K2 Q5 calibration by Astra's map (outputs/k2-two-role-test-map-20261028, plan: ../K2_CALIBRATION_PLAN.md).
Hard time limits (owner 2026-10-28: "do not let it think for ten years"): Medium episode 240 s, High 720 s; hitting the
limit is a model outcome (time_limit), never a retry. Fixed: T1.0, top_p 0.95, no `think` key (template default).

  python k2_calibrate.py --stage speed      # 4 contexts x 256 tokens, <= 12 min, no quality score
  python k2_calibrate.py --stage mscreen    # 4 configs x M2,M5, seed 11
  python k2_calibrate.py --stage mconfirm   # Medium finalist x M1,M3,M4, seeds 22,33
  python k2_calibrate.py --stage hscreen    # 4 configs x H2,H3, seed 11
  python k2_calibrate.py --stage hconfirm   # High finalist x H4,H8,H5, seed 22
  python k2_calibrate.py --stage report     # admission by the map's rules -> ../runs/k2-calibration-report.json
Finalist = most strict successes, then least total wall time, then fewest format errors, then config id.
"""
import argparse
import copy
import json
import sys
import time
import urllib.request
from pathlib import Path

import zoo_night as z

_build = z.build_plan
import high_matrix as hm  # noqa: E402  (K2 options/tag; its matrix no longer lists K2)

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs"
SEL = RUNS / "k2-calibration-selection.json"
KEY, TAG, OPTS = "k2-horizon-7b-q5", hm.K2Q5, dict(hm.K2)
MEDIUM = {"M-C8-P4": (8192, 4096), "M-C12-P4": (12288, 4096), "M-C16-P4": (16384, 4096), "M-C12-P2": (12288, 2048)}
HIGH = {"H-C12-P8": (12288, 8192), "H-C16-P8": (16384, 8192), "H-C24-P8": (24576, 8192), "H-C16-P4": (16384, 4096)}
STAGES = {
    "mscreen": ("medium", MEDIUM, ["M2", "M5"], [11]),
    "mconfirm": ("medium", None, ["M1", "M3", "M4"], [22, 33]),
    "hscreen": ("high", HIGH, ["H2", "H3"], [11]),
    "hconfirm": ("high", None, ["H4", "H8", "H5"], [22]),
}
z.runner.LEVELS["medium"]["seconds"] = 240     # Astra map: Medium deadline 240 s (High is already 720 s)
z.runner.LEVELS["high"]["seconds"] = 720


def run_name(stage):
    return f"zoo-k2-calib-{stage}-20261028"


def rows(stage):
    f = RUNS / run_name(stage) / "results.jsonl"
    out = []
    for line in (f.read_text(encoding="utf-8").splitlines() if f.exists() else []):
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return [d for d in out if d.get("outcome", "episode") == "episode"]


def finalist(stage, min_successes=1):
    per = {}
    for d in rows(stage):
        r = d.get("row") or {}
        m = per.setdefault(d["config_id"], dict(ok=0, wall=0.0, fmt=0, n=0))
        m["ok"] += bool(r.get("code_pass") and not r.get("violations"))
        m["wall"] += float(r.get("wall_seconds") or 0)
        m["fmt"] += int(r.get("format_errors") or 0)
        m["n"] += 1
    ranked = sorted(per.items(), key=lambda kv: (-kv[1]["ok"], kv[1]["wall"], kv[1]["fmt"], kv[0]))
    best = ranked[0] if ranked and ranked[0][1]["ok"] >= min_successes else None
    return (best[0] if best else None), dict(ranked)


def build_plan_for(stage):
    level, configs, tasks, seeds = STAGES[stage]
    if configs is None:            # confirm: the persisted finalist of the screen
        sel = json.loads(SEL.read_text(encoding="utf-8")) if SEL.exists() else {}
        cid = sel.get(level)
        if not cid:
            raise SystemExit(f"no {level} finalist selected: confirm stage skipped")
        configs = {cid: (MEDIUM | HIGH)[cid]}

    def build_plan(inv, only=None, profile="all", levels=z.NIGHT_LEVELS):
        if TAG not in inv:
            raise SystemExit(f"missing tag {TAG}")
        base = _build(inv, ["qwen3.5-9b"], "all", (level,))
        template = next(r for r in base["runs"] if r["level"] == level)
        runs = []
        for cid, (ctx, predict) in configs.items():
            for seed in seeds:
                for task in tasks:
                    r = copy.deepcopy(template)
                    r.update(model_key=KEY, tag=TAG, original=TAG, think=None, config_id=cid,
                             options=dict(OPTS, num_ctx=ctx, num_predict=predict, seed=seed), seed=seed,
                             task_id=task, phase=1, baked=False, deviation=False, structured=False, hint=False,
                             tool_mode="native", purpose=f"K2 calibration {stage} (Astra map)",
                             est_seconds=z.runner.LEVELS[level]["seconds"])
                    runs.append(r)
        base["runs"] = runs
        return base
    return build_plan


def speed():
    """Placement and token cost per context; one fixed 256-token generation each, 180 s per context."""
    out = []
    prompt = "Explain, step by step, how a sportsbook settles a cancelled esports match. " * 20
    for ctx in (8192, 12288, 16384, 24576):
        t = time.time()
        req = dict(model=TAG, prompt=prompt, stream=False, keep_alive="2m",
                   options=dict(OPTS, num_ctx=ctx, num_predict=256, seed=11))
        try:
            d = json.loads(urllib.request.urlopen(urllib.request.Request(
                "http://127.0.0.1:11434/api/generate", json.dumps(req).encode(),
                {"Content-Type": "application/json"}), timeout=180).read())
            ps = json.loads(urllib.request.urlopen("http://127.0.0.1:11434/api/ps", timeout=10).read())
            m = next((x for x in ps.get("models", []) if x.get("name") == TAG), {})
            ev, evd = d.get("eval_count") or 0, (d.get("eval_duration") or 0) / 1e9
            out.append(dict(num_ctx=ctx, seconds=round(time.time() - t, 1), load_s=round(d.get("load_duration", 0) / 1e9, 1),
                            prompt_tokens=d.get("prompt_eval_count"), eval_tokens=ev,
                            tok_per_s=round(ev / evd, 1) if evd else None, sample_too_short=ev < 128,
                            vram_share=round(m.get("size_vram", 0) / m["size"], 2) if m.get("size") else None))
        except Exception as e:
            out.append(dict(num_ctx=ctx, error=f"{type(e).__name__}: {e}"[:160]))
        print(json.dumps(out[-1]), flush=True)
        # unload so the next context loads cold with its own placement
        urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:11434/api/generate",
                               json.dumps(dict(model=TAG, keep_alive=0)).encode(),
                               {"Content-Type": "application/json"}), timeout=60).read()
    (RUNS / "k2-calibration-speed.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def report():
    ms_best, ms = finalist("mscreen")
    hs_best, hs = finalist("hscreen")
    mc = [d for d in rows("mconfirm")]
    hc = {d["task_id"]: bool((d.get("row") or {}).get("code_pass")) for d in rows("hconfirm")}
    viol = sum(bool((d.get("row") or {}).get("violations")) for s in STAGES for d in rows(s))
    m_ok = sum(bool((d.get("row") or {}).get("code_pass")) for d in mc)
    medium_ok = bool(ms_best) and ms[ms_best]["ok"] >= 2 and m_ok >= 5 and viol == 0
    high_ok = bool(hs_best) and hs[hs_best]["ok"] >= 2 and hc.get("H4") and hc.get("H8") and viol == 0
    rep = dict(medium=dict(finalist=ms_best, screen=ms, confirm_successes=f"{m_ok}/{len(mc)}", admitted=medium_ok),
               high=dict(finalist=hs_best, screen=hs, confirm=hc, admitted=bool(high_ok),
                         label=None if hc.get("H5") else "limited_high_H5_not_solved_within_720_seconds"),
               critical_violations=viol)
    (RUNS / "k2-calibration-report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["speed", "report", *STAGES])
    ap.add_argument("--dry-run", action="store_true", help="print the plan only")
    a = ap.parse_args()
    if a.stage == "speed":
        speed()
    elif a.stage == "report":
        report()
    else:
        z.build_plan = build_plan_for(a.stage)
        if a.dry_run:
            plan = z.build_plan(z.inventory())
            for r in plan["runs"]:
                print(r["config_id"], r["task_id"], r["seed"], r["options"]["num_ctx"], r["options"]["num_predict"],
                      z.runner.LEVELS[r["level"]]["seconds"])
            sys.exit(0)
        code = z.run(argparse.Namespace(name=run_name(a.stage), hours=6.0, new_deadline=True, models=None,
                                        profile="all", levels=[STAGES[a.stage][0]]))
        if a.stage in ("mscreen", "hscreen"):     # persist the explicit finalist for the confirm stage
            level = STAGES[a.stage][0]
            best, table = finalist(a.stage)
            sel = json.loads(SEL.read_text(encoding="utf-8")) if SEL.exists() else {}
            sel[level] = best
            SEL.write_text(json.dumps(sel, indent=1), encoding="utf-8")
            print("finalist", level, best, json.dumps(table), flush=True)
        sys.exit(code)
