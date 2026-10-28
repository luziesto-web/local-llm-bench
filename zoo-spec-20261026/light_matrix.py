"""Light config matrix (user 2026-10-28): the Light models were picked on 4 tasks, one seed, JSON protocol.
Screen: every config on L0-L11, seed 11, native tools, q8 KV cache; model by model (fewer reloads).
Then (separate run) the top 3 configs per model get seeds 22 and 33.

Configs per model: the vendor preset with thinking off / on, the "check yourself" system hint (it lifted Gemma
from 4/15 to 8/12 in the JSON era), and one or two sampling variants.
Budgets: thinking off 8K context / 4K reply; thinking on 16K / 8K (Light tasks are short).

  python light_matrix.py   -> runs/zoo-night-light-matrix-20261028 (resumable, zoo_night machinery)
"""
import argparse
import copy
import sys

import zoo_night as z

SYSHINT = ("When you change code and there are no tests next to it, first write a small test file yourself "
           "(candidate_tests.py, assert-based, importing the task modules), run it, fix the code until it passes, "
           "and only then give the final answer.")
NT = {"num_ctx": 8192, "num_predict": 4096}
TH = {"num_ctx": 16384, "num_predict": 8192}
GOOGLE = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}               # Gemma 4 card
IBM = {"temperature": 1.0, "top_p": 0.95}                                # Granite 4.2 card, all tasks
OLLDEF = {"temperature": 0.7, "top_p": 0.9, "top_k": 50}                 # granite4.2 Ollama tag defaults
QWEN_NT = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5}   # instruct general
QWEN_CODE = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0}  # thinking coding
QWEN_GEN = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5}   # thinking general


def gemma(key, tag):
    return [(key, tag, "nt-vendor", False, dict(GOOGLE, **NT)),
            (key, tag, "th-vendor", True, dict(GOOGLE, **TH)),
            (key, tag, "nt-vendor-syshint", False, dict(GOOGLE, **NT)),
            (key, tag, "th-vendor-syshint", True, dict(GOOGLE, **TH)),
            (key, tag, "nt-t03", False, dict(GOOGLE, temperature=0.3, **NT)),
            (key, tag, "nt-t06-syshint", False, dict(GOOGLE, temperature=0.6, **NT))]


MATRIX = (gemma("gemma4-e4b", "gemma4:e4b") + gemma("gemma4-e2b", "gemma4:e2b-it-qat") + [
    ("granite4.2-3b", "granite4.2:3b", "th-ibm", True, dict(IBM, **TH)),
    ("granite4.2-3b", "granite4.2:3b", "low-ibm", "low", dict(IBM, **TH)),
    ("granite4.2-3b", "granite4.2:3b", "nt-ibm", False, dict(IBM, **NT)),
    ("granite4.2-3b", "granite4.2:3b", "th-ibm-syshint", True, dict(IBM, **TH)),
    ("granite4.2-3b", "granite4.2:3b", "nt-ibm-syshint", False, dict(IBM, **NT)),
    ("granite4.2-3b", "granite4.2:3b", "th-olldef", True, dict(OLLDEF, **TH)),
    ("qwen3.5-4b", "qwen3.5:4b", "nt-general", False, dict(QWEN_NT, **NT)),
    ("qwen3.5-4b", "qwen3.5:4b", "th-coding", True, dict(QWEN_CODE, **TH)),
    ("qwen3.5-4b", "qwen3.5:4b", "th-general", True, dict(QWEN_GEN, **TH)),
    ("qwen3.5-4b", "qwen3.5:4b", "th-coding-syshint", True, dict(QWEN_CODE, **TH)),
    ("qwen3.5-4b", "qwen3.5:4b", "nt-general-syshint", False, dict(QWEN_NT, **NT)),
    ("qwen3.5-4b", "qwen3.5:4b", "nt-t03", False, dict(QWEN_NT, temperature=0.3, **NT)),
])
TASKS = ["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10", "L11"]
SEED = 11
_build = z.build_plan


def build_plan(inv, only=None, profile="all", levels=z.NIGHT_LEVELS):
    base = _build(inv, ["gemma4-e4b"], "all", ("light",))
    template = next(r for r in base["runs"] if r["level"] == "light")
    runs = []
    for model_key, tag, cfg, think, options in MATRIX:
        if tag not in inv:
            raise SystemExit(f"missing tag {tag}")
        for task in TASKS:
            r = copy.deepcopy(template)
            r.update(model_key=model_key, tag=tag, original=tag, config_id=cfg, think=think,
                     options=dict(options, seed=SEED), seed=SEED, task_id=task, phase=1, baked=False,
                     deviation=False, structured=False, hint=False, tool_mode="native",
                     purpose="Light config matrix (screen): " + cfg,
                     est_seconds=z.est_seconds("light", model_key, think))
            runs.append(r)
    base["runs"] = runs   # already model by model, config by config, task by task
    return base


_episode = z.runner.episode
_sys = z.runner.system_prompt


def episode(output, label, task, configuration):
    z.CURRENT["syshint"] = configuration["config_id"].split("/")[-1].endswith("-syshint")
    return _episode(output, label, task, configuration)


def system_prompt(level):
    text = _sys(level)
    return text + "\n" + SYSHINT if z.CURRENT.get("syshint") else text


z.build_plan, z.runner.episode, z.runner.system_prompt = build_plan, episode, system_prompt

if __name__ == "__main__":
    args = argparse.Namespace(name="zoo-night-light-matrix-20261028", hours=30.0, new_deadline=True, models=None,
                              profile="all", levels=["light"])
    sys.exit(z.run(args))
