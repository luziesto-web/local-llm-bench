"""Re-run episodes lost to Ollama tool-call parser errors (HTTP 500 "error parsing tool call" / "arg_key and
arg_value") with the harness that now retries and counts them as format errors (owner 2026-10-28).
The original run is left untouched; the re-runs go to <run>-reruns.

  python rerun_parser_failures.py zoo-high-winners-confirm-20261028
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import zoo_night as z
import high_matrix  # noqa: E402,F401  installs the -syshint system-prompt patch the original run used

RUNS = Path(__file__).resolve().parent.parent / "runs"


def lost(run):
    out = []
    for line in (RUNS / run / "results.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        e = (d.get("row") or {}).get("infra_error") or ""
        if "parsing tool call" in e or "arg_key and arg_value" in e:
            out.append((d["model_key"], d["config_id"], d["task_id"], d["seed"]))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    a = ap.parse_args()
    keys = set(lost(a.run))
    plan_file = sorted((RUNS / a.run).glob("plan-*.json"))[-1]
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    runs = [copy.deepcopy(r) for r in plan["runs"] if (r["model_key"], r["config_id"], r["task_id"], r["seed"]) in keys]
    print(f"re-running {len(runs)} of {len(keys)} lost episodes from {a.run}", flush=True)
    if not runs:
        sys.exit(0)
    for r in runs:
        r["purpose"] = (r.get("purpose") or "") + " | re-run after tool-call parser fix"

    def build_plan(inv, only=None, profile="all", levels=z.NIGHT_LEVELS):
        base = dict(plan)
        base["runs"] = runs
        return base
    z.build_plan = build_plan
    levels = sorted({r["level"] for r in runs})
    sys.exit(z.run(argparse.Namespace(name=a.run + "-reruns", hours=2.0, new_deadline=True, models=None,
                                      profile="all", levels=levels)))
