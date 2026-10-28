"""High config matrix (user 2026-10-28: matrices and a contest for High; the High configs were never screened).
Screen: every config on three representative High tasks (H0 warm-up, H3 owner-safe versioned cache,
H8 bug from a log), seed 11, native tools, q8 KV cache; model by model (heavy reloads).
The best config per model then plays the High contest on H0-H9 (separate run).

Vendor presets are the ones verified on the model cards 2026-10-28 (mastermind/MATRIX_PLAN.md), with a High
budget: 32K context, 16K reply (vendors ask for more; the mastermind gets it, a High executor should not need it).
30B models load with use_mmap (pinned-memory OOM on this PC otherwise).

  python high_matrix.py   -> runs/zoo-high-matrix-20261028 (resumable, zoo_night machinery)
"""
import argparse
import copy
import sys

import zoo_night as z

_build = z.build_plan       # the original planner; light_matrix (below) replaces z.build_plan on import
import light_matrix         # noqa: E402,F401  same "check yourself" system hint (episode/system-prompt patch)

B = {"num_ctx": 32768, "num_predict": 16384}   # calibration start; the matrix must use validated budgets (TODO)
MMAP = dict(B, use_mmap=True)
OSS = {"temperature": 1.0, "top_p": 1.0}                                             # OpenAI
CODER = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repeat_penalty": 1.05}       # Qwen3-Coder card
NEMO = {"temperature": 1.0, "top_p": 0.95}                                           # NVIDIA, all modes
GLM_AGENT = {"temperature": 0.7, "top_p": 1.0}                                       # Terminal Bench / SWE-bench
GLM_DEFAULT = {"temperature": 1.0, "top_p": 0.95}                                    # "most tasks"
QWEN_CODE = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0}
QWEN_GEN = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5}
QWEN_NT = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5}
K2 = {"temperature": 1.0, "top_p": 0.95}                                             # IFM card
K2Q5, K2Q4 = "hf.co/bartowski/K2-Horizon-7B-GGUF:Q5_K_M", "ingloryon/k2-horizon:7b-q4_K_M"

MATRIX = [
    ("gpt-oss-20b", "gpt-oss:20b", "high", "high", dict(OSS, **B)),
    ("gpt-oss-20b", "gpt-oss:20b", "medium", "medium", dict(OSS, **B)),
    ("gpt-oss-20b", "gpt-oss:20b", "low", "low", dict(OSS, **B)),
    ("gpt-oss-20b", "gpt-oss:20b", "high-syshint", "high", dict(OSS, **B)),
    ("qwen3-coder-30b", "qwen3-coder:30b", "vendor", None, dict(CODER, **MMAP)),
    ("qwen3-coder-30b", "qwen3-coder:30b", "vendor-syshint", None, dict(CODER, **MMAP)),
    ("qwen3-coder-30b", "qwen3-coder:30b", "t03", None, dict(CODER, temperature=0.3, **MMAP)),
    ("nemotron-3.5-lightning-30b", "nemotron-3.5-lightning:30b", "vendor-th", True, dict(NEMO, **MMAP)),
    ("nemotron-3.5-lightning-30b", "nemotron-3.5-lightning:30b", "vendor-nt", False, dict(NEMO, **MMAP)),
    ("nemotron-3.5-lightning-30b", "nemotron-3.5-lightning:30b", "vendor-th-syshint", True, dict(NEMO, **MMAP)),
    ("glm-4.7-flash", "glm-4.7-flash:latest", "agent-th", True, dict(GLM_AGENT, **MMAP)),
    ("glm-4.7-flash", "glm-4.7-flash:latest", "default-th", True, dict(GLM_DEFAULT, **MMAP)),
    ("glm-4.7-flash", "glm-4.7-flash:latest", "agent-nt", False, dict(GLM_AGENT, **MMAP)),
    ("glm-4.7-flash", "glm-4.7-flash:latest", "agent-th-syshint", True, dict(GLM_AGENT, **MMAP)),
    ("qwen3.5-9b", "qwen3.5:9b", "th-coding", True, dict(QWEN_CODE, **B)),
    ("qwen3.5-9b", "qwen3.5:9b", "th-general", True, dict(QWEN_GEN, **B)),
    ("qwen3.5-9b", "qwen3.5:9b", "th-coding-syshint", True, dict(QWEN_CODE, **B)),
    ("qwen3.5-9b", "qwen3.5:9b", "nt-general", False, dict(QWEN_NT, **B)),
    ("k2-horizon-7b-q5", K2Q5, "q5-template", None, dict(K2, **B)),     # Q5 tag: think parameter -> HTTP 400
    # K2 Q4 ("q4-think-high") removed 2026-10-28: the owner excluded Q4 (it also mixed quantizations under one key)
    ("k2-horizon-7b-q5", K2Q5, "q5-template-syshint", None, dict(K2, **B)),
]
TASKS = ["H0", "H3", "H8"]
SEED = 11


def build_plan(inv, only=None, profile="all", levels=z.NIGHT_LEVELS):
    base = _build(inv, ["qwen3.5-9b"], "all", ("high",))
    template = next(r for r in base["runs"] if r["level"] == "high")
    runs = []
    for model_key, tag, cfg, think, options in MATRIX:
        if tag not in inv:
            raise SystemExit(f"missing tag {tag}")
        for task in TASKS:
            r = copy.deepcopy(template)
            r.update(model_key=model_key, tag=tag, original=tag, config_id=cfg, think=think,
                     options=dict(options, seed=SEED), seed=SEED, task_id=task, phase=1, baked=False,
                     deviation=False, structured=False, hint=False, tool_mode="native",
                     purpose="High config matrix (screen): " + cfg,
                     est_seconds=z.est_seconds("high", model_key, think))
            runs.append(r)
    # owner 2026-10-28: task by task (all configs on H0, then H3, then H8); a stable sort keeps one model's
    # configs together inside a task, so a task costs one reload per model
    runs.sort(key=lambda r: TASKS.index(r["task_id"]))
    base["runs"] = runs
    return base


z.build_plan = build_plan

# owner-approved budgets per model (calibration-approved.json, 2026-10-28) replace the 32K/16K starting budget
import json  # noqa: E402
from pathlib import Path  # noqa: E402
APPROVED = json.loads((Path(__file__).resolve().parent / "calibration-approved.json").read_text(encoding="utf-8"))
MATRIX = [(k, tag, cfg, think, dict(opts, num_ctx=APPROVED[k]["num_ctx"], num_predict=APPROVED[k]["num_predict"]))
          for k, tag, cfg, think, opts in MATRIX if k in APPROVED]   # no approved budget -> not in the matrix (K2 Q5 excluded)

if __name__ == "__main__":
    args = argparse.Namespace(name="zoo-high-matrix-20261028", hours=72.0, new_deadline=True, models=None,
                              profile="all", levels=["high"])
    sys.exit(z.run(args))
