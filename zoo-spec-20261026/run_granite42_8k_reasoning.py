#!/usr/bin/env python3
"""Run the user-requested Granite 4.2 8B 8K reasoning matrix.

Four exact configurations are screened on the existing Medium tasks M1--M5:
8K context / 2K output and 8K context / 4K output, each with Ollama's
native ``think=\"medium\"`` and ``think=\"high\"`` values.  Results are
written through ``zoo_night`` so the live dashboard can show the campaign.
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from datetime import datetime

import zoo_night
import zoo_test_spec as spec


MODEL_KEY = "granite4.2-8b"
OFFICIAL_SAMPLING = {"temperature": 1.0, "top_p": 0.95}


def selected_configs() -> list[spec.Config]:
    """The four cells requested by the user; no baked-tag variants are added."""
    cells = []
    for effort in ("medium", "high"):
        for predict in (2048, 4096):
            cells.append(spec.Config(
                f"{effort}-ibm-c8k-p{predict}",
                effort,
                spec.opts(8192, predict, **OFFICIAL_SAMPLING),
                "User-requested 8K context / "
                f"{predict // 1024}K output; IBM official sampling; "
                f"Ollama native reasoning effort={effort}",
            ))
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name",
        default=f"medium-confirm-granite42-8k-{datetime.now():%Y%m%d}",
        help="Campaign folder under round2/runs (default: dated Medium confirmation)",
    )
    parser.add_argument(
        "--hours", type=float, default=4.0,
        help="Deadline budget; the four high/medium configurations may take over an hour",
    )
    parser.add_argument(
        "--new-deadline", action="store_true",
        help="Use a fresh deadline if resuming an existing campaign name",
    )
    args = parser.parse_args()

    # Restrict the standard, dashboard-compatible runner to precisely the four
    # requested cells.  The process is short-lived, so this does not modify the
    # benchmark specification on disk or any future invocation.
    spec.MODELS[MODEL_KEY].configs = selected_configs()
    zoo_night.run(Namespace(
        name=args.name,
        hours=args.hours,
        profile="all",
        models=[MODEL_KEY],
        levels=["medium"],
        new_deadline=args.new_deadline,
    ))


if __name__ == "__main__":
    main()
