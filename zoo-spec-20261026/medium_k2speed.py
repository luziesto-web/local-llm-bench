"""K2 speed-up for Medium (user decision 2026-10-27): smaller context, capped replies, Q4 vs Q5.

The Medium winner (K2 Q5 baked: num_ctx 24576, no reply cap, T1.0/top_p .95) scored 13/15 at ~268 s per task; its
Q5 weights + 24K cache do not fit the 8 GB GPU. Medium tasks filled at most ~14% of the context, so:
  k2q5-c8k        Q5, num_ctx 8192, num_predict 4096, same sampling, thinking off (as the winner)
  k2q4-c8k        Q4 (fits the GPU), same options
  k2q4-c8k-low / -medium / -high  Q4 with native reasoning levels, num_predict 6144 (room for the reasoning)
M1-M5, seed 11 only (screen); the best variant(s) get a confirmation seed afterwards.

  python medium_k2speed.py   -> runs/medium-k2speed-20261027 (resumable, zoo_night machinery)
"""
import argparse
import copy
import sys

import zoo_night as z

SAMPLING = {"temperature": 1.0, "top_p": 0.95}
VARIANTS = [
    ("k2-horizon-7b-q5", "k2q5-c8k", "hf.co/bartowski/K2-Horizon-7B-GGUF:Q5_K_M", None,
     dict(SAMPLING, num_ctx=8192, num_predict=4096)),
    ("k2-horizon-7b-q4", "k2q4-c8k", "ingloryon/k2-horizon:7b-q4_K_M", False,
     dict(SAMPLING, num_ctx=8192, num_predict=4096)),
    # Q4 accepts native reasoning levels (probe 2026-10-27: low < medium < high thinking length)
    ("k2-horizon-7b-q4", "k2q4-c8k-low", "ingloryon/k2-horizon:7b-q4_K_M", "low",
     dict(SAMPLING, num_ctx=8192, num_predict=6144)),
    ("k2-horizon-7b-q4", "k2q4-c8k-medium", "ingloryon/k2-horizon:7b-q4_K_M", "medium",
     dict(SAMPLING, num_ctx=8192, num_predict=6144)),
    ("k2-horizon-7b-q4", "k2q4-c8k-high", "ingloryon/k2-horizon:7b-q4_K_M", "high",
     dict(SAMPLING, num_ctx=8192, num_predict=6144)),
    # The 8K thinking variants spent every reply on reasoning (27-46K chars, cut at 6144 tokens, no answer):
    # user 2026-10-27 — test "high" with room for it: 12K/8K and 16K/8K (context / reply cap).
    ("k2-horizon-7b-q4", "k2q4-c12k-p8k-high", "ingloryon/k2-horizon:7b-q4_K_M", "high",
     dict(SAMPLING, num_ctx=12288, num_predict=8192)),
    ("k2-horizon-7b-q4", "k2q4-c16k-p8k-high", "ingloryon/k2-horizon:7b-q4_K_M", "high",
     dict(SAMPLING, num_ctx=16384, num_predict=8192)),
]
VARIANTS += [
    # The Medium winner (Q5, thinking off, T1.0) at a context that may fit the GPU better than its 24K (user 2026-10-27)
    ("k2-horizon-7b-q5", "k2q5-c12k-p8k", "hf.co/bartowski/K2-Horizon-7B-GGUF:Q5_K_M", None,
     dict(SAMPLING, num_ctx=12288, num_predict=8192)),
    ("k2-horizon-7b-q5", "k2q5-c16k-p8k", "hf.co/bartowski/K2-Horizon-7B-GGUF:Q5_K_M", None,
     dict(SAMPLING, num_ctx=16384, num_predict=8192)),
]
VARIANTS += [
    # user 2026-10-27: low and medium reasoning at the 12K/8K budget join the high variants task by task
    ("k2-horizon-7b-q4", "k2q4-c12k-p8k-low", "ingloryon/k2-horizon:7b-q4_K_M", "low",
     dict(SAMPLING, num_ctx=12288, num_predict=8192)),
    ("k2-horizon-7b-q4", "k2q4-c12k-p8k-medium", "ingloryon/k2-horizon:7b-q4_K_M", "medium",
     dict(SAMPLING, num_ctx=12288, num_predict=8192)),
]
GRANITE = {"temperature": 0.7, "top_p": 0.9, "top_k": 50}   # low-olldef, the Granite 8B Medium winner (11/15)
VARIANTS += [
    # user 2026-10-27: does the Medium context profile hold for Granite 8B? 8K context, reply cap 2048 vs 4096
    ("granite4.2-8b", "granite8b-low-c8k-p2k", "granite4.2:8b", "low", dict(GRANITE, num_ctx=8192, num_predict=2048)),
    ("granite4.2-8b", "granite8b-low-c8k-p4k", "granite4.2:8b", "low", dict(GRANITE, num_ctx=8192, num_predict=4096)),
]
PLAN_STAGES = [["k2q4-c8k"],
               ["k2q4-c12k-p8k-high", "k2q4-c12k-p8k-low", "k2q4-c12k-p8k-medium"],   # 16K high dropped by the user after M2
               ["granite8b-low-c8k-p2k", "granite8b-low-c8k-p4k"],
               ]   # stage by stage, task by task inside
# after the user enables the q8 KV cache: PLAN_STAGES.append(["k2q5-c12k-p8k"])  (16K judged redundant)
SEEDS = [11]   # user 2026-10-27: one seed to screen; only the best variant(s) get a second seed later
TASKS = ["M1", "M2", "M3", "M4", "M5"]
_build = z.build_plan


def build_plan(inv, only=None, profile="all", levels=z.NIGHT_LEVELS):
    base = _build(inv, ["k2-horizon-7b-q5"], profile, ("medium",))
    template = next(r for r in base["runs"] if r["config_id"] == "baked-k2-horizon-7b-q5-t")
    runs = []
    # user 2026-10-27: Q4 without reasoning first, then the two "high" variants; inside a stage task by task
    by_id = {v[1]: v for v in VARIANTS}
    for seed in SEEDS:
        for stage in PLAN_STAGES:
          for task in TASKS:
            for model_key, cfg_id, tag, think, options in (by_id[c] for c in stage):
                if tag not in inv:
                    raise SystemExit(f"missing tag {tag}")
                r = copy.deepcopy(template)
                r.update(model_key=model_key, config_id=cfg_id, tag=tag, original=tag, think=think,
                         options=dict(options, seed=seed), seed=seed, task_id=task, phase=1, baked=False,
                         deviation=True, purpose="K2 speed-up: smaller context and reply cap (Medium)",
                         est_seconds=240 if think is not None and think is not False else 150)
                runs.append(r)
    base["runs"] = runs
    return base


# user 2026-10-27: all K2 variants dropped from this run ("too tedious"); only the Granite stage is left to play.
PLAN_STAGES = [["granite8b-low-c8k-p2k", "granite8b-low-c8k-p4k"]]

z.build_plan = build_plan

if __name__ == "__main__":
    args = argparse.Namespace(name="medium-k2speed-20261027", hours=6.0, new_deadline=True, models=None,
                              profile="all", levels=["medium"])
    sys.exit(z.run(args))
