"""Medium confirmation (2026-10-27): is Granite 8B's 5/5 on seed 11 real or a lucky roll?
Top configs of the night, M1-M5, two more seeds (22, 33); seed 11 results come from the night run.
Winner gets baked as <tag>-T-M1 (user decision pending the result).

  python medium_confirm.py   -> runs/medium-confirm-20261027 (resumable, zoo_night machinery)
"""
import argparse
import sys

import zoo_night as z

KEEP = {("granite4.2-8b", "low-olldef"), ("k2-horizon-7b-q5", "baked-k2-horizon-7b-q5-t"),
        ("qwen3.5-9b", "th-coding"), ("qwen3.5-9b", "th-coding-pp15")}
z.spec.SEEDS[("screen", 1)] = [22, 33]
_build = z.build_plan


def build_plan(inv, only=None, profile="all", levels=z.NIGHT_LEVELS):
    plan = _build(inv, sorted({m for m, _ in KEEP}), profile, ("medium",))
    plan["runs"] = [r for r in plan["runs"] if (r["model_key"], r["config_id"]) in KEEP]
    for r in plan["runs"]:
        r["phase"] = 1
    found = {(r["model_key"], r["config_id"]) for r in plan["runs"]}
    if found != KEEP:
        raise SystemExit(f"missing configs: {KEEP - found}")
    return plan


z.build_plan = build_plan

if __name__ == "__main__":
    args = argparse.Namespace(name="medium-confirm-20261027", hours=4.0, new_deadline=True, models=None,
                              profile="all", levels=["medium"])
    sys.exit(z.run(args))
