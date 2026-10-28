#!/usr/bin/env python3
"""Fair High rerun for GLM-4.7-Flash and Nemotron with the vendor mmap profile.

Runs only the two requested models, one official vendor-mmap configuration,
seed 11, and H1-H5. Episodes are task-major and serial: both models do H1,
then both do H2, and so on. No baked variants are mixed into this rerun.
"""

from __future__ import annotations

import argparse
from argparse import Namespace

import zoo_night


MODELS = ["glm-4.7-flash", "nemotron-3.5-lightning-30b"]
ORIGINAL_BUILD_PLAN = zoo_night.build_plan


def selected_plan(inv, only=None, profile="core", levels=("high",)):
    full = ORIGINAL_BUILD_PLAN(inv, MODELS, "core", ["high"])
    # The shared core plan also contains unrelated models. Only the two
    # explicitly requested High models are prerequisites for this rescue run.
    missing = {item["model_key"] for item in full["missing"] if item["model_key"] in MODELS}
    assert not missing, f"Required model tags missing from Ollama inventory: {sorted(missing)}"
    chosen = [
        run for run in full["runs"]
        if run["model_key"] in MODELS and run["config_id"] == "vendor-mmap" and run["seed"] == 11
    ]
    model_order = {model: i for i, model in enumerate(MODELS)}
    chosen.sort(key=lambda run: (run["task_id"], model_order[run["model_key"]]))
    assert len(chosen) == 10, f"Expected 10 High episodes (2 models × H1-H5), found {len(chosen)}"
    assert all(run["options"].get("use_mmap") is True for run in chosen)
    assert all(run["options"].get("num_ctx") == 16384 for run in chosen)
    assert all(run["options"].get("num_predict") == 8192 for run in chosen)
    return dict(full, runs=chosen)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="zoo-high-mmap-rescue-20261027")
    parser.add_argument("--hours", type=float, default=12.0)
    args = parser.parse_args()
    zoo_night.build_plan = selected_plan
    zoo_night.run(Namespace(name=args.name, hours=args.hours, profile="core", models=MODELS,
                            levels=["high"], new_deadline=True))


if __name__ == "__main__":
    main()
