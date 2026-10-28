"""q8 KV-cache re-check (user 2026-10-28): same Medium episodes as medium_confirm (seeds 22, 33), now with
OLLAMA_KV_CACHE_TYPE=q8_0. Seed-11 retest lost 3 of 9 solved episodes; this decides if that is noise.
Compare with runs/medium-confirm-20261027 (same configs and seeds, f16 cache).

  python medium_q8_recheck.py   -> runs/medium-confirm-q8-20261028 (resumable, zoo_night machinery)
"""
import argparse
import copy
import sys

import zoo_night as z

KEEP = {("granite4.2-8b", "low-olldef"), ("k2-horizon-7b-q5", "baked-k2-horizon-7b-q5-t")}
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
    # user 2026-10-28: gpt-oss:20b joins this contest "for fun" - reasoning low, 8K context / 4K reply,
    # OpenAI sampling (temperature 1.0, top_p 1.0); same tasks and seeds
    if "gpt-oss:20b" not in inv:
        raise SystemExit("missing tag gpt-oss:20b")
    for r in [r for r in plan["runs"] if r["config_id"] == "baked-k2-horizon-7b-q5-t"]:
        g = copy.deepcopy(r)
        # native tools: gpt-oss answered the JSON protocol with native tool calls and empty text (config
        # low-c8k-p4k: 32 JSONDecodeError turns on M1); forcing format=json did not change that (low-c8k-p4k-json).
        # The actions are now offered as Ollama tools, as OpenCode does. Granite and K2 stay on the JSON protocol
        # here: this contest compares them with their pre-cache JSON runs.
        g.update(model_key="gpt-oss-20b", config_id="low-c8k-p4k-native", tag="gpt-oss:20b", original="gpt-oss:20b",
                 think="low", baked=False, deviation=True, structured=False, tool_mode="native",
                 options={"temperature": 1.0, "top_p": 1.0, "num_ctx": 8192, "num_predict": 4096, "seed": r["seed"]},
                 purpose="gpt-oss in Medium: reasoning low, 8K context / 4K reply (user, for fun)", est_seconds=300)
        plan["runs"].append(g)
    order = {"granite4.2-8b": 0, "k2-horizon-7b-q5": 1, "gpt-oss-20b": 2}
    plan["runs"].sort(key=lambda r: (r["seed"], r["task_id"], order[r["model_key"]]))
    return plan


z.build_plan = build_plan

if __name__ == "__main__":
    args = argparse.Namespace(name="medium-confirm-q8-20261028", hours=6.0, new_deadline=True, models=None,
                              profile="all", levels=["medium"])
    sys.exit(z.run(args))
