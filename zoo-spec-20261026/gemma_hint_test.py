"""Gemma hint check (user request 2026-10-27): can the "write your own tests" hint live in the prompt instead of
being injected by the runner? Three configs of gemma4:e4b (nt-official sampling, thinking off), Light tasks, 3 seeds:
  nt-official          plain (control)
  nt-official-hint     runner answers run_tests on a missing candidate_tests.py with HINT (night ablation)
  nt-official-syshint  no runner help; the rule is appended to the system prompt, as a baked SYSTEM / Open WebUI
                       workspace prompt would add it

  python gemma_hint_test.py            -> runs/gemma-hint-20261027 (resumable, same zoo_night machinery)
"""
import argparse
import sys

import zoo_night as z

SYSHINT = ("When you change code and there are no tests next to it, first write a small test file yourself "
           "(candidate_tests.py, assert-based, importing the task modules), run it, fix the code until it passes, "
           "and only then give the final answer.")
KEEP = {"nt-official", "nt-official-hint", "nt-official-syshint"}

z.EXTRA_CONFIGS["gemma4-e4b"].append(z.spec.Config(
    "nt-official-syshint", False,
    {"num_ctx": 8192, "num_predict": 2048, "temperature": 1.0, "top_p": 0.95, "top_k": 64},
    "Hint in the system prompt (baked SYSTEM / workspace prompt), no runner help"))
z.spec.SEEDS[("screen", 1)] = [11, 22, 33]

_build = z.build_plan


def build_plan(inv, only=None, profile="all", levels=z.NIGHT_LEVELS):
    plan = _build(inv, ["gemma4-e4b"], profile, ("light",))
    plan["runs"] = [r for r in plan["runs"] if r["config_id"] in KEEP and not r.get("baked")]
    for r in plan["runs"]:
        r["phase"] = 1
    return plan


_episode = z.runner.episode
_sys = z.runner.system_prompt


def episode(output, label, task, configuration):
    z.CURRENT["syshint"] = configuration["config_id"].endswith("-syshint")
    return _episode(output, label, task, configuration)


def system_prompt(level):
    text = _sys(level)
    return text + "\n" + SYSHINT if z.CURRENT.get("syshint") else text


z.build_plan, z.runner.episode, z.runner.system_prompt = build_plan, episode, system_prompt

if __name__ == "__main__":
    args = argparse.Namespace(name="gemma-hint-20261027", hours=2.0, new_deadline=True, models=None,
                              profile="all", levels=["light"])
    sys.exit(z.run(args))
