#!/usr/bin/env python3
"""Continue the Medium campaign with useful Granite axes only.

Granite invariants: original granite4.2:8b, 8192 context, 2048 output,
think='low', seed 11, T=0.7/top_p=0.9/top_k=50.  The proven control is
already present in medium-k2speed-20261027 and is not rerun.

K2 Q5 at 16K was stopped as a TKO in M1 after an unusually long run.  It is
kept in campaign results, but removed from the remaining run plan.
"""

from __future__ import annotations

import argparse
from argparse import Namespace

import zoo_night
import zoo_test_spec as spec


GRANITE = "granite4.2-8b"
K2 = "k2-horizon-7b-q5"
BASE = dict(num_ctx=8192, num_predict=2048, temperature=0.7, top_p=0.9, top_k=50)


def granite_configs() -> list[spec.Config]:
    return [
        spec.Config("low-c8k-p2k-action-schema", "low", dict(BASE),
                    "Strict action JSON schema instead of generic JSON mode", structured=True),
        spec.Config("low-c8k-p2k-repeat-100", "low", dict(BASE, repeat_penalty=1.0),
                    "Remove the runtime's repetition penalty; code reuses identifiers"),
        spec.Config("low-c8k-p2k-repeat-120", "low", dict(BASE, repeat_penalty=1.2),
                    "Strengthen repetition penalty to test looping versus code fidelity"),
        spec.Config("low-c8k-p2k-repeat-window-256", "low", dict(BASE, repeat_last_n=256),
                    "Longer lookback for the runtime's repetition penalty"),
        spec.Config("low-c8k-p2k-min-p005", "low", dict(BASE, min_p=0.05),
                    "Suppress low-probability tokens while preserving the established decoding preset"),
    ]


def k2_config() -> spec.Config:
    return spec.Config("winner-c16k", None,
                       dict(num_ctx=16384, temperature=1.0, top_p=0.95),
                       "Winning 24K Q5 profile; only context reduced to 16K")


def selected_plan(inv, only=None, profile="all", levels=("medium",)):
    full = ORIGINAL_BUILD_PLAN(inv, [GRANITE], profile, ["medium"])
    allowed_granite = {config.id for config in spec.MODELS[GRANITE].configs}
    chosen = [
        run for run in full["runs"]
        if run["model_key"] == GRANITE and run["config_id"] in allowed_granite and run["seed"] == 11
    ]
    chosen.sort(key=lambda run: (
        run["task_id"],
        run["model_key"] == K2,
        run["config_id"],
        run["seed"],
    ))
    assert len(chosen) == 25, f"Expected 25 Granite episodes, found {len(chosen)}"
    assert all(run["tag"] == "granite4.2:8b" for run in chosen if run["model_key"] == GRANITE)
    return dict(full, runs=chosen)


ORIGINAL_BUILD_PLAN = zoo_night.build_plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="medium-confirm-granite42-8k-low-sampling-20261027")
    parser.add_argument("--hours", type=float, default=6.0)
    args = parser.parse_args()

    spec.MODELS[GRANITE].configs = granite_configs()
    spec.SEEDS[("screen", 1)] = [11]
    zoo_night.build_plan = selected_plan
    zoo_night.run(Namespace(name=args.name, hours=args.hours, profile="all",
                            models=[GRANITE], levels=["medium"], new_deadline=True))


if __name__ == "__main__":
    main()
