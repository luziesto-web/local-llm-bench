"""Tournament with native tools (user 2026-10-28): Light (10 tasks) and Medium.

The actions are offered as Ollama tools (run_light TOOL_MODE "native", as OpenCode does) instead of the JSON
protocol that gpt-oss could not follow. One best-known config per model and level, seed 11 (same seed as the
earlier JSON nights, so the protocol effect is visible). q8 KV cache is on (global Ollama setting).
- Light: L0-L9, model by model (user: Light may run per model).
- Medium: M1-M5, task by task across all models (more intrigue).
- High models are out (user: their configs are not tested yet); K2 Q4 is not invited (its losses were
  context / reasoning budget, not format: 2 format errors in 16 episodes).

  python night_native.py   -> runs/zoo-night-native-20261028 (resumable, zoo_night machinery)
"""
import argparse
import copy
import sys

import zoo_night as z

LINEUP = {
    "light": [("gemma4-e4b", "bakedtag-gemma4-e4b-T-S2"), ("granite4.2-3b", "th-ibm")],
}
# user 2026-10-28: "for now only Light"; Medium waits (it had M1 of Granite 8B started):
# "medium": [("granite4.2-8b", "low-olldef"), ("k2-horizon-7b-q5", "baked-k2-horizon-7b-q5-t"),
#            ("qwen3.5-9b", "th-coding"), ("qwen3.5-4b", "th-coding")],
# Joined Light during the run (user 2026-10-28: "who else was a Light candidate"): Qwen3.5 4B (Light contests
# v3/v4; in the zoo it only had Medium configs) with its vendor non-thinking preset, and Gemma 4 E2B (the site
# server model, new here) with Google sampling, thinking off - both copied onto the Light tasks.
EXTRA_LIGHT = [("qwen3.5-4b", "nt-general", "qwen3.5-4b", None),
               ("gemma4-e4b", "nt-official", "gemma4-e2b", "gemma4:e2b-it-qat"),
               # user 2026-10-28: upgraded Gemmas play next to the plain ones - the "check yourself" rule in the
               # system prompt (JSON era: Gemma E4B 8/12 with it vs 4/15 without)
               ("gemma4-e4b", "nt-official-syshint", "gemma4-e4b", None),
               ("gemma4-e4b", "nt-official-syshint", "gemma4-e2b", "gemma4:e2b-it-qat")]
SEED = 11
_build = z.build_plan
import light_matrix   # noqa: E402,F401  "-syshint" configs get the self-check rule in the system prompt


def build_plan(inv, only=None, profile="all", levels=z.NIGHT_LEVELS):
    models = sorted({m for pairs in LINEUP.values() for m, _ in pairs} | {k for k, *_ in EXTRA_LIGHT})
    plan = _build(inv, models, "all", ("light", "medium"))   # medium too: Qwen 4B templates live there
    keep = {(lv, m, c) for lv, pairs in LINEUP.items() for m, c in pairs}
    runs = [r for r in plan["runs"] if (r["level"], r["model_key"], r["config_id"]) in keep and r["seed"] == SEED]
    found = {(r["level"], r["model_key"], r["config_id"]) for r in runs}
    if found != keep:
        raise SystemExit(f"missing configs: {sorted(keep - found)}")
    light_models = [m for m, _ in LINEUP["light"]]
    for source_key, cfg, key, tag in EXTRA_LIGHT:
        base_cfg = cfg[:-len("-syshint")] if cfg.endswith("-syshint") else cfg
        template = next(r for r in plan["runs"] if (r["model_key"], r["config_id"]) == (source_key, base_cfg)
                        and r["seed"] == SEED)
        for task in z.LEVEL_TASKS["light"]:
            g = copy.deepcopy(template)
            g.update(level="light", task_id=task, model_key=key, config_id=cfg,
                     est_seconds=z.est_seconds("light", key, g["think"]))
            if tag:
                g.update(tag=tag, original=tag)
            runs.append(g)
        if key not in light_models:
            light_models.append(key)
    for r in runs:
        r.update(phase=1, tool_mode="native")
    level_order = list(z.LEVEL_TASKS)

    def key(r):
        tasks = z.LEVEL_TASKS[r["level"]]
        models = light_models if r["level"] == "light" else [m for m, _ in LINEUP[r["level"]]]
        model = models.index(r["model_key"])
        task = tasks.index(r["task_id"])
        # Light: model by model; Medium: task by task
        return (level_order.index(r["level"]),) + ((model, task) if r["level"] == "light" else (task, model))

    runs.sort(key=key)
    plan["runs"] = runs
    return plan


z.build_plan = build_plan

if __name__ == "__main__":
    args = argparse.Namespace(name="zoo-night-native-20261028", hours=10.0, new_deadline=True, models=None,
                              profile="all", levels=list(LINEUP))
    sys.exit(z.run(args))
