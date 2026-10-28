"""High context calibration (user 2026-10-28), before the High config matrix:
1. one config per model (the vendor default from high_matrix.MATRIX), 2 seeds, on a small, the largest and the
   log task (H0, H5, H8), budget 32K context / 16K reply;
2. calib_report.py: peak context fill and reply length per model -> too small: raise, too big: cut;
3. the adjusted budgets play one more seed (round 2 = this script with --round 2, budgets from calibration.json);
4. then the matrices run with each model's calibrated budget.

  python high_calibrate.py              -> runs/zoo-high-calibrate-20261028        (round 1, seeds 11 and 22)
  python high_calibrate.py --round 2    -> runs/zoo-high-calibrate-r2-20261028     (seed 33, calibrated budgets)
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import zoo_night as z

_build = z.build_plan
import high_matrix as hm   # noqa: E402  vendor presets (it also installs light_matrix's syshint patch: unused here)

DEFAULTS = {"gpt-oss-20b": "high", "qwen3-coder-30b": "vendor", "nemotron-3.5-lightning-30b": "vendor-th",
            "glm-4.7-flash": "agent-th", "qwen3.5-9b": "th-coding", "k2-horizon-7b-q5": "q5-template"}
TASKS = ["H0", "H5", "H8"]
CALIBRATION = Path(__file__).resolve().parent / "calibration.json"


def build_plan_for(round_no):
    seeds = {1: [11, 22], 2: [33], 3: [44], 4: [55]}[round_no]
    budgets = json.loads(CALIBRATION.read_text(encoding="utf-8")) if round_no >= 2 else {}

    def build_plan(inv, only=None, profile="all", levels=z.NIGHT_LEVELS):
        base = _build(inv, ["qwen3.5-9b"], "all", ("high",))
        template = next(r for r in base["runs"] if r["level"] == "high")
        runs = []
        for model_key, tag, cfg, think, options in hm.MATRIX:
            if DEFAULTS.get(model_key) != cfg or (round_no >= 3 and budgets[model_key]["status"] in ("validated", "hold")):
                continue
            if tag not in inv:
                raise SystemExit(f"missing tag {tag}")
            opts = dict(options)
            if round_no >= 2:
                # only the two budget numbers; calibration.json also carries config/status/measured
                opts.update(num_ctx=budgets[model_key]["num_ctx"], num_predict=budgets[model_key]["num_predict"])
            for seed in seeds:
                for task in TASKS:
                    r = copy.deepcopy(template)
                    r.update(model_key=model_key, tag=tag, original=tag, think=think,
                             config_id=f"{cfg}-c{opts['num_ctx'] // 1024}k-p{opts['num_predict'] // 1024}k",
                             options=dict(opts, seed=seed), seed=seed, task_id=task, phase=1, baked=False,
                             deviation=False, structured=False, hint=False, tool_mode="native",
                             purpose=f"High context calibration round {round_no}",
                             est_seconds=z.est_seconds("high", model_key, think))
                    runs.append(r)
        base["runs"] = runs          # model by model (heavy reloads), seed by seed, task by task
        return base
    return build_plan


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, choices=(1, 2, 3, 4), default=1)
    round_no = ap.parse_args().round
    z.build_plan = build_plan_for(round_no)
    name = "zoo-high-calibrate-20261028" if round_no == 1 else f"zoo-high-calibrate-r{round_no}-20261028"
    args = argparse.Namespace(name=name, hours=48.0, new_deadline=True, models=None, profile="all", levels=["high"])
    sys.exit(z.run(args))
