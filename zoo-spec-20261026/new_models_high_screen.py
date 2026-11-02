"""New L candidates (pulled 2026-11-01) — High config screening (owner 2026-11-01: "отбор сначала делать").

Same design as gemma12_high_matrix.py: up to 4 configs per model x H0, H3, H8, seed 11, task by task, 12-minute
High limit, native tools, the same "check yourself" -syshint system hint. Calibration is folded into this pass: one
generous budget for everyone (32K context / 16K reply, the calibration-r1 budget of the other models) with use_mmap,
and the run records the peak context, so a separate calibration round is not needed. Budget is not a matrix axis
(Gemma 12B: 16K vs 32K changed nothing).

Configs come from each vendor's model card (sampling) plus think on/off and the hint:
  North Mini Code, Nemotron-Cascade-2  T1.0 top_p 0.95 (Cohere / NVIDIA)
  Qwen3.6 35B   th-coding T0.6 top_p 0.95 top_k 20 presence 0 (Qwen "precise coding"), th-general = factory
                (T1.0 presence 1.5), nt-instruct T0.7 top_p 0.8 top_k 20 presence 1.5
  Laguna XS 2.1 T0.7 top_k 20 (Poolside benchmark setting)
  Ornith-1.5    T0.6 top_p 0.95 top_k 20 (publisher "general"), th-bench T1.0
  Flint 40B     T1.0 top_p 0.95 top_k 20; always reasons (no think switch), so no nt- configs
Dense Qwen3.8 27B and Muse Glimmer join only if the speed probe (new_models_passport.py) shows they are usable:
pass --models.

  python new_models_high_screen.py --dry-run [--models key,key]
  python new_models_high_screen.py --start [--models key,key]     (refuses while another model is loaded)
  python new_models_high_screen.py --report
"""
import argparse
import collections
import json
import sys

import zoo_night as z
import high_matrix  # noqa: F401  (imports light_matrix: the -syshint system-prompt patch)

BUDGET = {"num_ctx": 32768, "num_predict": 16384, "use_mmap": True}
V = {"temperature": 1.0, "top_p": 0.95}
QWEN_CODE = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0}
QWEN_GEN = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5}
QWEN_NT = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5}
LAGUNA = {"temperature": 0.7, "top_k": 20}
ORNITH = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
FLINT = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}


def four(sampling):
    """th / nt x with / without the self-check hint, one vendor sampling."""
    return [("th-vendor", True, sampling), ("nt-vendor", False, sampling),
            ("th-vendor-syshint", True, sampling), ("nt-vendor-syshint", False, sampling)]


MODELS = {  # key: (tag, [(config_id, think, sampling)])  think None = no think field
    "north-mini-code": ("north-mini-code-1.0:latest", four(V)),
    "nemotron-cascade-2-30b": ("nemotron-cascade-2:30b", four(V)),
    "qwen3.6-35b": ("qwen3.6:35b", [("th-coding", True, QWEN_CODE), ("th-general", True, QWEN_GEN),
                                    ("nt-instruct", False, QWEN_NT), ("th-coding-syshint", True, QWEN_CODE)]),
    "laguna-xs-2.1": ("laguna-xs-2.1:latest", four(LAGUNA)),
    "ornith-1.5-35b": ("ornith-1.5:35b", [("th-general", True, ORNITH), ("th-bench", True, dict(ORNITH, temperature=1.0)),
                                          ("nt-general", False, ORNITH), ("th-general-syshint", True, ORNITH)]),
    "flint-40b": ("flint:40b", [("vendor", None, FLINT), ("vendor-syshint", None, FLINT)]),
    "qwen3.8-27b": ("qwen3.8:27b", [("th-coding", True, QWEN_CODE), ("nt-instruct", False, QWEN_NT),
                                    ("th-coding-syshint", True, QWEN_CODE)]),
    "muse-glimmer-30b": ("muse-glimmer:30b", [("th-vendor", True, {"temperature": 1.0, "top_p": 0.95, "top_k": 64}),
                                              ("nt-vendor", False, {"temperature": 1.0, "top_p": 0.95, "top_k": 64}),
                                              ("th-vendor-syshint", True, {"temperature": 1.0, "top_p": 0.95, "top_k": 64})]),
}
DEFAULT = ["qwen3.6-35b", "laguna-xs-2.1", "north-mini-code", "nemotron-cascade-2-30b", "flint-40b", "ornith-1.5-35b"]  # fastest first (passport 2026-11-02)
TASKS = ("H0", "H3", "H8")
ROUND2 = ("H1", "H2", "H4", "H6", "H7", "H9")    # same hard tasks and seed as zoo-high-round2-20261029
SEED = 11
NAME = "new-models-high-screen-20261101"
ACTIVE = list(DEFAULT)
ONLY = None                                      # --round2: set of "model_key/config_id" finalists


def build_plan(inv, only=None, profile="all", levels=("high",)):
    runs = []
    for key in ACTIVE:                       # model by model (one load per model), task by task inside
        tag, configs = MODELS[key]
        if tag not in inv:
            raise SystemExit(f"missing tag {tag}")
        for task in TASKS:
            for cfg, think, sampling in configs:
                if ONLY is not None and f"{key}/{cfg}" not in ONLY:
                    continue
                runs.append(dict(phase=1, level="high", model_key=key, tag=tag, original=tag, config_id=cfg,
                                 think=think, options=dict(sampling, **BUDGET, seed=SEED), seed=SEED, task_id=task,
                                 contract="explicit", structured=False, deviation=False, hint=False, baked=False,
                                 tool_mode="native", purpose="new L candidates High screening",
                                 est_seconds=z.runner.LEVELS["high"]["seconds"]))
    return dict(originals={k: MODELS[k][0] for k in ACTIVE}, missing=[], baked=[], runs=runs)


def report():
    f = z.RUNS / NAME / "results.jsonl"
    score = collections.defaultdict(lambda: [0, 0, 0.0, 0])
    for line in f.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        if d.get("outcome", "episode") != "episode":
            continue
        r = d.get("row") or {}
        s = score[f'{d["model_key"]}/{d["config_id"]}']
        s[0] += bool(r.get("automated_workflow_pass"))
        s[1] += bool(r.get("code_pass"))
        s[2] += float(r.get("wall_seconds") or 0) / 60
        s[3] += 1
    ranked = sorted(score, key=lambda c: (c.split("/")[0], -score[c][0], score[c][2]))
    print(json.dumps({c: dict(points=score[c][0], code=score[c][1], played=score[c][3], minutes=round(score[c][2], 1))
                      for c in ranked}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--start", action="store_true", help="required to actually run (owner go-ahead)")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run of the same name")
    ap.add_argument("--round2", help="comma-separated model_key/config_id finalists -> H1 H2 H4 H6 H7 H9")
    ap.add_argument("--batch", help="suffix for a second round-2 batch")
    ap.add_argument("--hours", type=float, help="deadline in hours; with --resume it sets a new deadline")
    ap.add_argument("--models", help="comma-separated model keys (default: the six MoE models)")
    a = ap.parse_args()
    assert z.runner.LEVELS["high"]["seconds"] == 720
    if a.models:
        ACTIVE = [m for m in a.models.split(",") if m]
        assert ACTIVE and set(ACTIVE) <= set(MODELS), ACTIVE
    if a.round2:
        ONLY = set(c for c in a.round2.split(",") if c)
        ACTIVE = [k for k in DEFAULT + [m for m in MODELS if m not in DEFAULT] if any(c.startswith(k + "/") for c in ONLY)]
        assert ONLY <= {f"{k}/{c[0]}" for k in MODELS for c in MODELS[k][1]}, ONLY
        TASKS, NAME = ROUND2, "new-models-high-round2-20261102" + (f"-{a.batch}" if a.batch else "")
    if a.report:
        report()
        sys.exit(0)
    z.build_plan = build_plan
    if a.dry_run:
        runs = build_plan({MODELS[k][0]: {} for k in ACTIVE})["runs"]
        for r in runs:
            print(r["model_key"], r["config_id"], r["task_id"], r["think"], r["options"])
        print(len(runs), "episodes")
        sys.exit(0)
    if not a.start:
        raise SystemExit("Not started: pass --start once the owner says go (use --dry-run to see the plan).")
    if (z.RUNS / NAME).exists() and not a.resume:
        raise SystemExit(f"Refusing to overwrite existing run: {NAME}")
    if z.runner.api("ps").get("models"):
        raise SystemExit("Ollama has a model loaded (another run?); wait for it to finish.")
    try:
        sys.exit(z.run(argparse.Namespace(name=NAME, hours=a.hours or 12.0, new_deadline=a.hours is not None or not a.resume, models=None, profile="all",
                                          levels=["high"])))
    finally:
        for k in ACTIVE:
            z._api("generate", dict(model=MODELS[k][0], keep_alive=0), 120)
