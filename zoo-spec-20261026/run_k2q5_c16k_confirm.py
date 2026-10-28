#!/usr/bin/env python3
"""Confirm the winning K2 Q5 Medium profile with only context reduced to 16K.

The 24K leader used the model's default reasoning behavior, temperature 1.0,
top_p 0.95 and no explicit output cap.  This confirmation preserves all of
that and changes only num_ctx from 24576 to 16384.  It uses three seeds on
M1--M5 (15 episodes), matching the established Medium comparison scale.
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from datetime import datetime

import zoo_night
import zoo_test_spec as spec


MODEL_KEY = "k2-horizon-7b-q5"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name",
        default=f"medium-confirm-k2q5-c16k-{datetime.now():%Y%m%d}",
    )
    parser.add_argument("--hours", type=float, default=3.0)
    parser.add_argument("--new-deadline", action="store_true")
    args = parser.parse_args()

    spec.MODELS[MODEL_KEY].configs = [
        spec.Config(
            "winner-c16k",
            None,
            spec.opts(16384, 2048, temperature=1.0, top_p=0.95),
            "Winner confirmation: 24K leader profile with num_ctx reduced to 16K",
        )
    ]
    # The original winning result is reported as 13/15, so use the same
    # three-seed Medium scale.  ``num_predict`` is removed below because the
    # winning 24K profile had no forced output cap; only context may change.
    spec.MODELS[MODEL_KEY].configs[0].options.pop("num_predict")
    spec.SEEDS[("screen", 1)] = [11, 22, 33]
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
