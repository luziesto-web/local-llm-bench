#!/usr/bin/env python3
"""Screen only meaningful sampling changes for Granite 4.2 8B at 8K/2K low thinking.

The known control is deliberately excluded:
``temperature=0.7, top_p=0.9, top_k=50, think=\"low\"``.
Every cell keeps 8K context, 2K output, low thinking and seed 11.  The
matrix isolates temperature, top_p and top_k, then includes the combined
IBM sampling setting.  It runs M1--M5 through the dashboard-compatible
``zoo_night`` runner.
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from datetime import datetime

import zoo_night
import zoo_test_spec as spec


MODEL_KEY = "granite4.2-8b"
CONTEXT = 8192
OUTPUT = 2048


def selected_configs() -> list[spec.Config]:
    return [
        spec.Config(
            "low-c8k-p2k-t10-p090-k50",
            "low",
            spec.opts(CONTEXT, OUTPUT, temperature=1.0, top_p=0.90, top_k=50),
            "Sampling screen: temperature only versus known 0.7/0.9/k50 control",
        ),
        spec.Config(
            "low-c8k-p2k-t07-p095-k50",
            "low",
            spec.opts(CONTEXT, OUTPUT, temperature=0.7, top_p=0.95, top_k=50),
            "Sampling screen: top_p only versus known 0.7/0.9/k50 control",
        ),
        spec.Config(
            "low-c8k-p2k-t07-p090-k20",
            "low",
            spec.opts(CONTEXT, OUTPUT, temperature=0.7, top_p=0.90, top_k=20),
            "Sampling screen: top_k only versus known 0.7/0.9/k50 control",
        ),
        spec.Config(
            "low-c8k-p2k-t10-p095-k50",
            "low",
            spec.opts(CONTEXT, OUTPUT, temperature=1.0, top_p=0.95, top_k=50),
            "IBM temperature/top_p combined; Modelfile top_k=50 retained",
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name",
        default=f"medium-confirm-granite42-8k-low-sampling-{datetime.now():%Y%m%d}",
    )
    parser.add_argument("--hours", type=float, default=2.0)
    parser.add_argument("--new-deadline", action="store_true")
    args = parser.parse_args()

    # Isolate this screening to the four non-control sampling changes.  The
    # in-memory specification is discarded when this short-lived process exits.
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
