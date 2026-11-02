"""High round 2 for the L models that only had screening (owner 2026-11-02: "найти бест конфиги и устроить L конкурс").

Same tasks, seed, limit and budgets as high_round2.py (H1 H2 H4 H6 H7 H9, seed 11, 12 min, calibration-approved.json).
Finalists = top-2 screening configs, one with thinking and one without (Gemma 12B round 2: thinking configs timed out
on every hard task, so the fast nt config must get its chance):
  GLM 4.7 Flash  agent-th-syshint (3/3), agent-nt (2/3, fastest)
  Qwen3.5 9B     th-coding (3/3), nt-general (2/3)
Gemma 26B plays through gemma26_high_matrix.py --round2 th-official,nt-official.

  python l_round2_night.py --dry-run
  python l_round2_night.py --start [--hours 5]
  python l_round2_night.py --report
"""
import argparse
import sys

import zoo_night as z
import high_round2 as r2

r2.FINALISTS.clear()
r2.FINALISTS.update({"glm-4.7-flash": ["agent-th-syshint", "agent-nt"],
                     "qwen3.5-9b": ["th-coding", "nt-general"]})
r2.NAME = "l-high-round2-20261102"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--start", action="store_true", help="required to actually run")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--hours", type=float, default=5.0)
    a = ap.parse_args()
    assert z.runner.LEVELS["high"]["seconds"] == 720
    if a.report:
        r2.report()
        sys.exit(0)
    z.build_plan = r2.build_plan
    if a.dry_run:
        plan = r2.build_plan(z.inventory())
        for r in plan["runs"]:
            print(r["model_key"], r["config_id"], r["task_id"], r["think"], r["options"])
        print(len(plan["runs"]), "episodes")
        sys.exit(0)
    if not a.start:
        raise SystemExit("Not started: pass --start.")
    if (z.RUNS / r2.NAME).exists() and not a.resume:
        raise SystemExit(f"Refusing to overwrite existing run: {r2.NAME}")
    if z.runner.api("ps").get("models"):
        raise SystemExit("Ollama has a model loaded (another run?); wait for it to finish.")
    try:
        sys.exit(z.run(argparse.Namespace(name=r2.NAME, hours=a.hours, new_deadline=not a.resume, models=None,
                                          profile="all", levels=["high"])))
    finally:
        for tag in ("glm-4.7-flash:latest", "qwen3.5:9b"):
            z._api("generate", dict(model=tag, keep_alive=0), 120)
