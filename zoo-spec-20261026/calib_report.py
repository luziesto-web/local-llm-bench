"""Context report for High calibration: how full did the context get, how long were the replies.

Measured, not estimated: on Ollama 0.40.x prompt_eval_count is the WHOLE input of a request and
prompt_eval_cached_count its cached part (checked on light-matrix transcripts: 725 total / 692 cached). The
character-count estimate used before 2026-10-28 is gone (GPT review: JSON escaping overstates it).
Per model: max input, max reply, peak (input + reply of one request), replies cut by the cap, coverage.

Rule (user 2026-10-28; joint ctx/reply choice after the GPT review):
- coverage first: fewer complete episodes than planned, or infra errors -> "not enough data", budget unchanged;
- reply cap: any reply cut -> cap x1.5; else cap = max reply x1.3 rounded up to 2K, never under 4K and never
  under the longest reply seen;
- context: a cut or peak > 80% -> raise x1.5; peak < 40% -> cut to peak x1.5; always >= max input + new reply cap,
  rounded up to 4K, never under 8K.
The result is a PROPOSAL (status "proposed"); a budget is "validated" only after a round played with it shows no
cut, peak <= 80% and full coverage.

  python calib_report.py runs/zoo-high-calibrate-20261028 --expect 6 [--write]   -> table; --write: calibration.json
"""
import argparse
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs"


def slug(s):   # same as zoo_night.slug (folder names)
    return re.sub(r"[^A-Za-z0-9_-]+", "-", s).strip("-")


def up(value, step):
    return int(math.ceil(value / step) * step)


def episode_stats(task_dir):
    s = dict(max_input=0, max_reply=0, peak=0, cut=0, broken_lines=0)
    for line in (task_dir / "transcript.jsonl").open(encoding="utf-8"):
        try:
            ev = json.loads(line)
        except ValueError:
            s["broken_lines"] += 1
            continue
        if ev.get("event") != "response":
            continue
        p = ev.get("payload") or {}
        inp, out = p.get("prompt_eval_count") or 0, p.get("eval_count") or 0
        s.update(max_input=max(s["max_input"], inp), max_reply=max(s["max_reply"], out),
                 peak=max(s["peak"], inp + out))
        s["cut"] += p.get("done_reason") == "length"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--expect", type=int, required=True, help="planned episodes per model in this run")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    night = a.run if a.run.is_absolute() else HERE.parent / a.run
    per = {}
    for line in (night / "results.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        o = d.get("options") or {}
        # one row per model AND config (GPT review: mixing configs under one model key makes the budget meaningless)
        m = per.setdefault((d["model_key"], d["config_id"]), dict(num_ctx=o.get("num_ctx"), num_predict=o.get("num_predict"),
                                                max_input=0, max_reply=0, peak=0, cut=0, episodes=0, infra=0,
                                                solved=0))
        row = d.get("row") or {}
        if d.get("outcome", "episode") != "episode" or row.get("infra_error"):
            m["infra"] += 1
            continue
        folder = RUNS / f"{night.name}-{slug(d['model_key'])}-{slug(d['config_id'])}-s{d['seed']}-run" / d["task_id"]
        if not (folder / "transcript.jsonl").exists():
            m["infra"] += 1
            continue
        s = episode_stats(folder)
        m.update(max_input=max(m["max_input"], s["max_input"]), max_reply=max(m["max_reply"], s["max_reply"]),
                 peak=max(m["peak"], s["peak"]), cut=m["cut"] + s["cut"], episodes=m["episodes"] + 1,
                 solved=m["solved"] + bool(row.get("code_pass")))
    out = {}
    print(f"{'model':28s} {'ctx':>6s} {'input':>6s} {'reply':>6s} {'cap':>6s} {'peak':>6s} {'fill':>5s} {'cut':>3s} "
          f"{'cov':>5s} {'code':>5s}  verdict")
    for key, m in per.items():
        fill = m["peak"] / m["num_ctx"] if m["num_ctx"] else 0
        model_key = key[0]
        if not m["num_ctx"] or not m["num_predict"]:
            ctx, predict, verdict, status = m["num_ctx"], m["num_predict"], "budget not set explicitly (tag?): skipped", "unchanged"
        elif m["episodes"] < a.expect or m["infra"]:
            ctx, predict, verdict, status = m["num_ctx"], m["num_predict"], "not enough data: budget unchanged", "unchanged"
        else:
            predict = (up(m["num_predict"] * 1.5, 2048) if m["cut"]
                       else max(4096, up(m["max_reply"] * 1.3, 2048), m["max_reply"]))
            if m["cut"] or fill > 0.8:
                ctx, verdict = m["num_ctx"] * 1.5, "too small -> raise"
            elif fill < 0.4:
                ctx, verdict = m["peak"] * 1.5, "too large -> cut"
            else:
                ctx, verdict = m["num_ctx"], "ok"
            ctx = max(8192, up(max(ctx, m["max_input"] + predict), 4096))
            status = "validated" if (not m["cut"] and fill <= 0.8 and ctx == m["num_ctx"]
                                     and predict == m["num_predict"]) else "proposed"
        out[model_key] = dict(config=key[1], num_ctx=ctx, num_predict=predict, status=status,
                        measured=dict(max_input=m["max_input"], max_reply=m["max_reply"], peak=m["peak"],
                                      cut=m["cut"], episodes=m["episodes"], infra=m["infra"]))
        print(f"{key[0][:16] + '/' + key[1][:11]:28s} {m['num_ctx'] or 0:6d} {m['max_input']:6d} {m['max_reply']:6d} {m['num_predict'] or 0:6d} "
              f"{m['peak']:6d} {fill:5.0%} {m['cut']:3d} {m['episodes']:2d}/{a.expect:<2d} {m['solved']:2d}/{m['episodes']:<2d}"
              f"  {verdict}: {(ctx or 0) // 1024}K / {(predict or 0) // 1024}K ({status})")
    if a.write:
        (HERE / "calibration.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print("written", HERE / "calibration.json")


if __name__ == "__main__":
    main()
