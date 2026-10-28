#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zoo_test_spec.py
================
Instructions + machine-readable test matrix for the local-LLM "zoo" on LooePC
(RTX 3070 Ti Laptop 8 GB VRAM, i7-12700H, 32 GB RAM, Windows 11, Ollama).

AUDIENCE: the executor agent (Codex / OpenCode). Read this file top to bottom, then
run  `python zoo_test_spec.py summary`  to see the size of the plan.

HOW TO USE
    python zoo_test_spec.py rules                       # global rules + decision rules
    python zoo_test_spec.py env                         # environment lock commands
    python zoo_test_spec.py probes                      # Stage 1 probes (cheap, run first)
    python zoo_test_spec.py show <model_key>            # official recs + configs of one model
    python zoo_test_spec.py summary [--stage screen --round 1]
    python zoo_test_spec.py dump --stage screen --round 1 > matrix_r1.json
    python zoo_test_spec.py dump --stage screen --round 2 --promoted promoted_r1.json
    python zoo_test_spec.py dump --stage validate --promoted promoted_r2.json
    python zoo_test_spec.py modelfile <model_key> <config_id>   # PRINT only, never applied
    python zoo_test_spec.py wilson <k> <n>                      # confidence interval

IMPORTANT PROVENANCE NOTE
    Vendor recommendations below were collected on 2026-10-26. Some were read on the
    vendor page itself (IBM Granite docs, Ollama granite4.2 page, Mistral HF card,
    OpenAI gpt-oss README, IFM vLLM recipe); others were read as quotations on
    mirrors/aggregators (Qwen3.5 card, Gemma 4 defaults via Unsloth docs). STAGE 0
    therefore REQUIRES re-checking every value on the vendor's own model card and
    recording the date. If a card disagrees with this file, the card wins; record the diff.

SCOPE RULES FROM THE USER (do not violate)
    * Test BASE models via per-request API options. Existing baked "-T"/"M"/"High" tags are
      irrelevant here; never create or overwrite tags without explicit user confirmation.
    * Do not touch the production site, Oracle, Supabase, keys, vacancies infrastructure.
    * Do not delete models or old results; never overwrite an existing run folder.
    * Do not run High-tier contests automatically (ENABLE_HIGH = False).
    * Do not mix old-benchmark and new-contest numbers in one ranking.
    * Never conclude from a single run. Verify processes really write results/status;
      do not rely on lock files or `ollama ps` alone. `ollama ps` shows placement, not GPU load.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field, asdict
from typing import Any

SPEC_VERSION = "2026-10-26"
ENABLE_HIGH = False  # user asked to stop after Medium; High contest needs explicit go-ahead

# ----------------------------------------------------------------------------------------
# 1. GLOBAL RULES / DECISION RULES
# ----------------------------------------------------------------------------------------
GLOBAL_RULES = [
    "R1  Pass = hidden acceptance check passes AND no sandbox violations AND the final report "
    "does not contradict observed facts. Store these as separate columns, never one blended score.",
    "R2  Record per run: model, tag digest, config_id, task_id, seed, effective options actually sent, "
    "think value, wall_s, load_s, gen_s (eval_duration), prompt_tokens, eval_tokens, tok_per_s "
    "(eval_count/eval_duration), truncation flag, tool_calls, violations, observed_test_pass, "
    "claimed_test_pass, code_pass, workflow_pass, honest_report, ollama_version, env_hash.",
    "R3  Speed = tokens/s from eval_duration and time EXCLUDING load_duration. Warm the model "
    "with one discarded request first. Never compare wall time that includes model load.",
    "R4  Interleave/randomise the order of models and configs (fixed seed for the shuffle) so thermal "
    "drift and background load do not favour whoever runs first. Log GPU temp/clock every 5 s.",
    "R5  Truncation (hit num_predict) is recorded as its own outcome. A truncated run is a failure "
    "for ranking, but also report 'would-pass-if-budget-larger' by re-running once with 2x budget.",
    "R6  Hidden checks live outside anything the model can read or write. The sandbox has no network. "
    "(IFM reports a K2-Horizon-7B run that found and downloaded benchmark answers, inflating a score "
    "that IFM does not count as real. Treat every agentic model as capable of this.)",
    "R7  Each task needs a reference solution that PASSES the hidden check and at least two "
    "known-bad mutants that FAIL it. A task nobody passes (e.g. L3) is quarantined until "
    "this is verified.",
    "R8  The task prompt must state the contract explicitly: which files may be written, that the "
    "model must create candidate_tests.py itself and run it, and what the final report must contain. "
    "The old 'implicit' contract is only used for the deliberate contract ablation.",
    "R9  Tuning/validation split: configs are chosen on TUNING tasks only; VALIDATION tasks are "
    "never looked at until a config is promoted. Seeds: 1 (round 1), 3 (round 2), 5 (validate).",
    "R10 Every option that is sent must be proven effective: see probe P4. If Ollama ignores a "
    "parameter for a model, say so in the report and drop that axis for that model.",
    "R11 Old results (granite3.3, ministral) were produced at a forced temperature of 1.0 that "
    "contradicts the vendors' own recommendations for those models; treat them as not comparable.",
    "R12 Reports are written in Russian for the user, with raw tables in CSV/JSONL alongside.",
]

DECISION_RULES = [
    "D1  Successive halving: round 1 = 1 seed x all configs; keep the better half (min 2 per model). "
    "Round 2 = 3 seeds x survivors. Validate = 5 seeds x top 1-2 configs on validation tasks.",
    "D2  Rank configs by (workflow_pass rate, Wilson lower bound, median gen time). A config beats "
    "another only if it wins on the same tasks across at least 2 of 3 seeds; otherwise call it a tie "
    "and pick the faster/cheaper one.",
    "D3  Report Wilson 95% intervals for every pass rate. 4/4 does NOT mean 100%; its lower bound is ~51%.",
    "D4  Detecting 70% vs 50% pass rate with 80% power needs roughly 90 runs per arm; "
    "so do not claim small gaps from small samples.",
    "D5  Roles: Light = fast, cheap, correct on small edits; Medium = multi-file autonomy; High = hard logic "
    "(disabled). A model is assigned a role only by the user after the report; never auto-tag.",
]

# ----------------------------------------------------------------------------------------
# 2. ENVIRONMENT LOCK (Stage 0)
# ----------------------------------------------------------------------------------------
ENVIRONMENT = {
    "env_vars": {
        "OLLAMA_FLASH_ATTENTION": "1",
        "OLLAMA_KV_CACHE_TYPE": "q8_0",
        "OLLAMA_MAX_LOADED_MODELS": "1",
        "OLLAMA_NUM_PARALLEL": "1",
        "OLLAMA_KEEP_ALIVE": "10m",
    },
    "notes": [
        "Env vars are GLOBAL for the Ollama server: set them, fully quit and restart Ollama, "
        "then record them in every result file (env_hash). Do not change them mid-campaign.",
        "Optional ablation (one model, best config): OLLAMA_KV_CACHE_TYPE=f16 vs q8_0.",
        "Plug the laptop in, set Windows power mode to Best performance, close browsers/games.",
        "Run tests through the Ollama HTTP API (localhost:11434), NOT through Open WebUI, "
        "because Open WebUI model settings override request options. Open WebUI/OpenCode are Stage 4.",
    ],
    "powershell": [
        "setx OLLAMA_FLASH_ATTENTION 1",
        "setx OLLAMA_KV_CACHE_TYPE q8_0",
        "setx OLLAMA_MAX_LOADED_MODELS 1",
        "setx OLLAMA_NUM_PARALLEL 1",
        "setx OLLAMA_KEEP_ALIVE 10m",
        "# then quit Ollama from the tray and start it again",
        "powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c   # High performance plan",
        "ollama --version",
        "ollama list",
        "nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw,memory.used "
        "--format=csv -l 5 > gpu_log.csv",
        "# server log (for HTTP 500 diagnosis): %LOCALAPPDATA%\\Ollama\\server.log",
    ],
    "stage0_checklist": [
        "Record ollama --version, GPU driver, Windows power mode, env vars, free RAM/VRAM at idle.",
        "For each model: `ollama show <tag> --modelfile` and `--parameters`; store the dump.",
        "For each model: open the vendor model card and confirm every value in MODELS[...]['official'] "
        "and MODELS[...]['sources']; record date and any difference.",
        "Confirm exact tags with `ollama list` (entries marked verify_tag=True are guesses).",
        "Create a NEW results folder per campaign (timestamped). Never reuse or overwrite old folders.",
    ],
}

# ----------------------------------------------------------------------------------------
# 3. TASK SUITES
# ----------------------------------------------------------------------------------------
# status: existing = already in round2 harness; author = must be written (R7, R8);
#         quarantine = excluded from ranking until reviewed.
TASKS: dict[str, dict[str, Any]] = {
    # smoke
    "L0": {"suite": "smoke", "status": "existing", "note": "add(a,b) fix; sanity only, everybody passed"},
    # light tuning (selection happens on these)
    "L2": {"suite": "light_tuning", "status": "existing", "note": "timeout=30 added without changing signature"},
    "L4": {"suite": "light_tuning", "status": "existing", "note": "zero is a valid setting (falsy trap)"},
    "L5": {"suite": "light_tuning", "status": "existing", "note": "cache key must include profile_version"},
    "L6": {"suite": "light_tuning", "status": "author", "note": "e.g. pagination off-by-one"},
    "L7": {"suite": "light_tuning", "status": "author", "note": "e.g. CSV normalisation (dates, dedupe)"},
    # light validation (holdout)
    "L8": {"suite": "light_validation", "status": "author", "note": "holdout, different bug class"},
    "L9": {"suite": "light_validation", "status": "author", "note": "holdout, different bug class"},
    # quarantined
    "L3": {"suite": "light_quarantine", "status": "quarantine",
           "note": "nobody passed; Qwen and Gemma failed at the same hidden check (line 5, ValueError "
                   "'Unknown status'). Review spec vs. check before use."},
    # medium tuning: use the existing round2 Medium tasks (verify ids in round2 folder)
    "M1": {"suite": "medium_tuning", "status": "existing", "note": "re-run in full (old run was interrupted)"},
    "M2": {"suite": "medium_tuning", "status": "existing", "note": "from round2"},
    "M3": {"suite": "medium_tuning", "status": "existing", "note": "from round2"},
    "M4": {"suite": "medium_tuning", "status": "existing", "note": "re-run in full (old run was interrupted)"},
    "M5": {"suite": "medium_tuning", "status": "existing", "note": "from round2"},
    "M6": {"suite": "medium_validation", "status": "author", "note": "holdout multi-file task"},
    "M7": {"suite": "medium_validation", "status": "author", "note": "holdout multi-file task"},
    # micro tasks: single-shot, domain = vacancies/resumes/Upwork niches. Gold labels by the USER.
    "V1": {"suite": "micro", "status": "author", "note": "relevance classification of 30 vacancy texts "
           "(betting/esports trading/risk/support roles) vs a CV; gold labels from the user"},
    "V2": {"suite": "micro", "status": "author", "note": "extract title/company/location/seniority/salary "
           "from 30 vacancy texts into a fixed JSON schema"},
    "V3": {"suite": "micro", "status": "author", "note": "EN->RU localisation of 20 short support/betting "
           "snippets; scored by the user on a 1-5 scale, blind"},
    "V4": {"suite": "micro", "status": "author", "note": "summarise 3 Russian-language resumes to a fixed schema"},
    "V5": {"suite": "micro_holdout", "status": "author", "note": "holdout items for V1/V2 (never used in tuning)"},
    "V6": {"suite": "micro_holdout", "status": "author", "note": "holdout items for V3/V4"},
    # language probes (also run as probe P7)
    "G1": {"suite": "language", "status": "author", "note": "Russian instruction following + factual fidelity"},
    "G2": {"suite": "language", "status": "author", "note": "Portuguese instruction following"},
    "G3": {"suite": "language", "status": "author", "note": "English technical writing"},
}
VALIDATION_MAP = {
    "light_tuning": "light_validation",
    "medium_tuning": "medium_validation",
    "micro": "micro_holdout",
}
R1_SUITES = {"smoke", "light_tuning", "micro"}   # cheap suites for round 1 (1 seed)
SEEDS = {("screen", 1): [11], ("screen", 2): [11, 22, 33], ("validate", 0): [11, 22, 33, 44, 55]}

# ----------------------------------------------------------------------------------------
# 4. DATA CLASSES
# ----------------------------------------------------------------------------------------
@dataclass
class Config:
    id: str
    think: Any                      # None | False | True | "low" | "medium" | "high"
    options: dict                   # Ollama options sent on EVERY request (incl. num_ctx, num_predict)
    purpose: str
    deviation: bool = False         # True = intentionally differs from the vendor recommendation
    contract: str = "default"       # "default" (explicit contract, R8) | "implicit" (old behaviour, ablation)
    structured: bool = False        # True = also send Ollama `format` = action JSON schema


@dataclass
class ModelSpec:
    key: str
    tag: str
    family: str
    roles: list
    official: dict                  # human-readable vendor recommendations
    thinking_control: str
    caveats: list
    sources: list                   # (url, provenance)
    suites: list
    sec_per_run: float              # very rough planning number only
    configs: list = field(default_factory=list)
    verify_tag: bool = False
    follow_best_of: str | None = None   # for quant variants: reuse the leader's promoted configs


def opts(ctx: int, predict: int, **kw) -> dict:
    d = {"num_ctx": ctx, "num_predict": predict}
    d.update(kw)
    return d


# ----------------------------------------------------------------------------------------
# 5. MODEL DEFINITIONS (vendor recommendations -> configs)
# ----------------------------------------------------------------------------------------
def qwen_configs(allow_pp_ablation: bool = True) -> list:
    """Qwen3.5 card, 'Best Practices' sampling presets (2026-10-26)."""
    c = [
        Config("nt-general", False,
               opts(8192, 2048, temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5),
               "Official instruct (non-thinking) preset, general tasks"),
        Config("nt-reasoning", False,
               opts(8192, 2048, temperature=1.0, top_p=1.0, top_k=40, min_p=0.0, presence_penalty=2.0),
               "Official instruct preset for reasoning tasks (card 'Best Practices' table, checked 2026-10-27; "
               "the card's Quickstart section says top_p 0.95/top_k 20/presence 1.5 - the table wins)"),
        Config("nt-t03", False,
               opts(8192, 2048, temperature=0.3, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5),
               "DEVIATION: low temperature (the user's current habit) - tests whether it helps executors",
               deviation=True),
        Config("th-coding", True,
               opts(16384, 8192, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0),
               "Official thinking preset for precise coding tasks"),
        Config("th-general", True,
               opts(16384, 8192, temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5),
               "Official thinking preset for general tasks"),
        Config("th-coding-8k", True,
               opts(8192, 4096, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0),
               "DEVIATION: thinking with a small budget (cheap); Qwen advises >=128K ctx for full thinking",
               deviation=True),
    ]
    if allow_pp_ablation:
        c.append(Config("th-coding-pp15", True,
                        opts(16384, 8192, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5),
                        "ABLATION: presence_penalty 1.5 on code (official coding preset uses 0.0)",
                        deviation=True))
    return c


def k2_configs() -> list:
    """IFM K2-Horizon-7B: temperature 1.0, top_p 0.95; reasoning_effort low|medium|high (default high).
    IFM's published benchmark numbers use effort=high and >= 32768 output tokens."""
    base = dict(temperature=1.0, top_p=0.95)
    c = []
    for effort in ("low", "medium", "high"):
        for pred in (4096, 8192):
            c.append(Config(f"{effort}-p{pred}", effort, opts(16384, pred, **base),
                            f"Official sampling, reasoning_effort={effort}, output budget {pred}"))
    c.append(Config("high-p16384-c24k", "high", opts(24576, 16384, **base),
                    "Closer to the vendor's benchmark regime (big output budget; ctx must grow with it)"))
    c.append(Config("high-p8192-c24k", "high", opts(24576, 8192, **base),
                    "ABLATION: re-check the user's earlier finding that 24K ctx is slower/not better",
                    deviation=True))
    c.append(Config("high-p8192-T06", "high", opts(16384, 8192, temperature=0.6, top_p=0.95),
                    "DEVIATION: lower temperature for an executor role", deviation=True))
    return c


def granite42_configs(include_extra: bool = True) -> list:
    """IBM Granite 4.2: IBM says temperature=1.0 and top_p=0.95 for ALL tasks incl. tool calling;
    max_new_tokens 8192 for thinking, 2048 for non-thinking. The Ollama page ships different Modelfile
    defaults (T0.7, top_p 0.9, top_k 50, num_ctx 8192) -> test both. top_k is NOT sent in the 'ibm'
    preset (IBM gives none), so the Modelfile default top_k=50 applies; record it."""
    ibm = dict(temperature=1.0, top_p=0.95)
    oll = dict(temperature=0.7, top_p=0.9, top_k=50)
    c = [
        Config("nt-ibm", False, opts(8192, 2048, **ibm), "IBM official sampling, thinking off"),
        Config("nt-olldef", False, opts(8192, 2048, **oll), "Ollama Modelfile default sampling, thinking off"),
        Config("low-ibm", "low", opts(16384, 4096, **ibm), "IBM official sampling, low-effort thinking"),
        Config("low-olldef", "low", opts(16384, 4096, **oll), "Ollama default sampling, low-effort thinking"),
        Config("th-ibm", True, opts(16384, 8192, **ibm), "IBM official sampling, full thinking (IBM default)"),
        Config("th-olldef", True, opts(16384, 8192, **oll), "Ollama default sampling, full thinking"),
    ]
    if include_extra:
        c.append(Config("th-ibm-T06", True, opts(16384, 8192, temperature=0.6, top_p=0.95),
                        "DEVIATION: lower temperature for executor role", deviation=True))
        c.append(Config("th-ibm-c8k", True, opts(8192, 4096, **ibm),
                        "DEVIATION: small budget to match Light settings", deviation=True))
    return c


MODELS: dict[str, ModelSpec] = {}


def _add(spec: ModelSpec) -> None:
    MODELS[spec.key] = spec


_add(ModelSpec(
    key="qwen3.5-9b", tag="qwen3.5:9b", family="Qwen3.5",
    roles=["Medium executor", "reasoning helper"],
    official={
        "thinking_general": "T1.0 top_p0.95 top_k20 min_p0 presence_penalty1.5 repetition_penalty1.0",
        "thinking_coding": "T0.6 top_p0.95 top_k20 min_p0 presence_penalty0.0 repetition_penalty1.0",
        "instruct_general": "T0.7 top_p0.8 top_k20 min_p0 presence_penalty1.5 repetition_penalty1.0",
        "instruct_reasoning": "T1.0 top_p0.95 top_k20 min_p0 presence_penalty1.5 repetition_penalty1.0",
        "note": "Qwen advises >=128K context to keep thinking quality; 8 GB VRAM cannot do that, "
                "so thinking budget/ctx is a real experimental axis.",
    },
    thinking_control="Ollama `think` true/false. Card: thinking is default for 9B; some distributors say "
                     "Small-series (0.8B-9B) default to non-thinking. Always set think explicitly. "
                     "Card switch is chat_template_kwargs enable_thinking, NOT /think or /nothink.",
    caveats=["presence_penalty may be ignored by some Ollama builds: run probe P4 before trusting it",
             "Thinking can loop endlessly (reported for the Qwen3.5 family): rely on num_predict + P6"],
    sources=[("https://huggingface.co/Qwen/Qwen3.5-9B#best-practices", "card text read via mirrors; verify"),
             ("https://unsloth.ai/docs/models/qwen3.5.md", "aggregator; says small models default non-thinking")],
    suites=["smoke", "light_tuning", "medium_tuning", "micro", "language"], sec_per_run=33,
    configs=qwen_configs(),
))
_add(ModelSpec(
    key="qwen3.5-4b", tag="qwen3.5:4b", family="Qwen3.5",
    roles=["Light executor", "Medium candidate"],
    official={
        "thinking": "T0.6 top_p0.95 top_k20 (one source for 4B) / general thinking T1.0 top_p0.95 top_k20 pp1.5",
        "instruct": "T0.7 top_p0.8 top_k20 pp1.5",
        "note": "Same family rules as 9B. Verify the 4B card separately in Stage 0.",
    },
    thinking_control="Same as qwen3.5-9b; set think explicitly.",
    caveats=["Best light-contest result so far (4/5 hidden-check pass, 4/5 workflow), n=1 seed: confirm"],
    sources=[("https://huggingface.co/Qwen/Qwen3.5-4B", "verify")],
    suites=["smoke", "light_tuning", "medium_tuning", "micro", "language"], sec_per_run=15,
    configs=qwen_configs(),
))
_add(ModelSpec(
    key="qwen3.5-0.8b", tag="qwen3.5:0.8b", family="Qwen3.5",
    roles=["micro tasks: routing, classification, trivial extraction"],
    official={
        "instruct": "T0.7 top_p0.8 top_k20 pp1.5; non-thinking is the default for 0.8B",
        "note": "Qwen warns this size is more prone to thinking loops and recommends tuned sampling "
                "and streaming so runaway generations can be cut.",
    },
    thinking_control="Non-thinking by default; thinking switchable. Use stream + loop guard (P6).",
    caveats=["Do NOT rank on agentic tasks; use micro suite and smoke only"],
    sources=[("https://huggingface.co/Qwen/Qwen3.5-0.8B", "verify")],
    suites=["smoke", "micro", "language"], sec_per_run=6,
    configs=[
        Config("nt-general", False, opts(8192, 1024, temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5),
               "Official instruct preset"),
        Config("nt-t03", False, opts(8192, 1024, temperature=0.3, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5),
               "DEVIATION: low temperature for classification", deviation=True),
        Config("th-coding-guarded", True, opts(8192, 3072, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0),
               "Thinking with a hard cap and loop detection", deviation=True),
    ],
))
_add(ModelSpec(
    key="gemma4-e4b", tag="gemma4:e4b", family="Gemma 4",
    roles=["fast text analysis", "short code", "Light candidate if verification is fixed"],
    official={
        "sampling": "temperature 1.0, top_p 0.95, top_k 64 (Google defaults for Gemma 4)",
        "context": "128K for E2B/E4B",
        "note": "Thinking is enabled with a <|think|> token in the system prompt (card); "
                "E4B 4-bit needs roughly 5 GB.",
    },
    thinking_control="Try Ollama `think`; if the toggle does not change behaviour (probe P1) fall back to "
                     "putting <|think|> at the start of the system prompt, and record which method worked.",
    caveats=["Previous light contest: correct code 4/5 but workflow 1/5 (did not write/run tests); "
             "likely a contract problem -> contract ablation below",
             "A fixed seed in a tag makes agent retries repeat identical output; always vary seed"],
    sources=[("https://ai.google.dev/gemma", "primary; verify numbers"),
             ("https://unsloth.ai/docs/models/gemma-4", "aggregator quoting Google defaults")],
    suites=["smoke", "light_tuning", "micro", "language"], sec_per_run=12,
    configs=[
        Config("nt-official", False, opts(8192, 2048, temperature=1.0, top_p=0.95, top_k=64),
               "Google default sampling, thinking off"),
        Config("th-official", True, opts(16384, 8192, temperature=1.0, top_p=0.95, top_k=64),
               "Google default sampling, thinking on"),
        Config("nt-t06", False, opts(8192, 2048, temperature=0.6, top_p=0.95, top_k=64),
               "DEVIATION: compromise temperature", deviation=True),
        Config("nt-t03", False, opts(8192, 2048, temperature=0.3, top_p=0.95, top_k=64),
               "DEVIATION: the user's current habit", deviation=True),
        Config("nt-official-implicit", False, opts(8192, 2048, temperature=1.0, top_p=0.95, top_k=64),
               "ABLATION: old implicit contract (does NOT tell it to write tests) vs nt-official",
               deviation=True, contract="implicit"),
        Config("nt-t03-implicit", False, opts(8192, 2048, temperature=0.3, top_p=0.95, top_k=64),
               "ABLATION: old implicit contract at the user's habit temperature", deviation=True, contract="implicit"),
    ],
))
_add(ModelSpec(
    key="ministral3-8b-instruct", tag="ministral-3:8b", family="Ministral 3",
    roles=["strict-format worker (if JSON mode fixes protocol failures)", "backup"],
    official={
        "instruct": "temperature below 0.1 for daily-driver/production (Mistral HF card); "
                    "0.1-0.15 quoted from Mistral by Unsloth",
        "note": "The OLD benchmark forced T=1.0, which is ~10x above the vendor's value for this model.",
    },
    thinking_control="Instruct variant has no thinking switch (separate Reasoning variant).",
    caveats=["Old harness 'JSON failures' may be an artefact of T=1.0; test with and without Ollama "
             "structured output (`format`)", "Slowest tok/s among 8B models in the light contest; check VRAM fit (P3)"],
    sources=[("https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-GGUF", "primary (Mistral org)"),
             ("https://unsloth.ai/docs/models/ministral-3", "aggregator quoting Mistral")],
    suites=["smoke", "light_tuning", "micro", "language"], sec_per_run=36,
    configs=[
        Config("t005", None, opts(8192, 2048, temperature=0.05), "Vendor card: below 0.1"),
        Config("t015", None, opts(8192, 2048, temperature=0.15), "Vendor value quoted by Unsloth"),
        Config("t005-json", None, opts(8192, 2048, temperature=0.05), "As t005 + structured output", structured=True),
        Config("t015-json", None, opts(8192, 2048, temperature=0.15), "As t015 + structured output", structured=True),
        Config("t10-control", None, opts(8192, 2048, temperature=1.0),
               "CONTROL: reproduces the old forced temperature; expected to be worse", deviation=True),
    ],
))
_add(ModelSpec(
    key="ministral3-8b-reasoning", tag="ministral-3-8b-reasoningq4", family="Ministral 3",
    roles=["reasoning helper (if it beats the Instruct variant)"],
    official={"reasoning": "temperature 0.7, top_p 0.95; increase max output for reasoning"},
    thinking_control="Always reasons; budget is controlled by num_predict.",
    caveats=["Tag is the user's own name (screenshot shows 'Ministral-3-8B-ReasoningQ4'); confirm via `ollama list`"],
    sources=[("https://unsloth.ai/docs/models/ministral-3", "aggregator quoting Mistral; verify on Mistral card")],
    suites=["smoke", "light_tuning"], sec_per_run=60, verify_tag=True,
    configs=[
        Config("rsn-p4096", None, opts(16384, 4096, temperature=0.7, top_p=0.95), "Official sampling, 4096 budget"),
        Config("rsn-p8192", None, opts(16384, 8192, temperature=0.7, top_p=0.95), "Official sampling, 8192 budget"),
    ],
))
_add(ModelSpec(
    key="k2-horizon-7b-q5", tag="k2-horizon:7b-q5", family="K2-Horizon (IFM)",
    roles=["Medium/High candidate", "best old result: Medium 4/4, High 3/4 at 16K/4K"],
    official={
        "sampling": "temperature 1.0, top_p 0.95",
        "effort": "reasoning_effort low|medium|high (high is default); delivered via chat_template_kwargs",
        "benchmarks": "IFM's reported scores use effort=high and at least 32768 output tokens",
        "context": "512K (9B stored params incl. embeddings)",
        "note": "Needs llama.cpp with K2-Horizon support (merged 2026-10-06): verify the Ollama build is newer.",
    },
    thinking_control="Ollama `think` low/medium/high may or may not map to reasoning_effort: probe P1 must show "
                     "that thinking length changes with the level; if not, only think true/false is testable.",
    caveats=["Old High tests capped output at 4096-8192 tokens: far below the vendor regime, so 'High' may "
             "have been truncated reasoning rather than weak reasoning",
             "Your tags set ctx 24576 (q5) / 32768 (q8) although your own data said 16K/4K is better and faster",
             "R6: IFM reports a 7B run that located SWE-bench answers; sandbox must be offline"],
    sources=[("https://recipes.vllm.ai/IFM/K2-Horizon-7B", "IFM recipe page"),
             ("https://huggingface.co/IFM/K2-Horizon-7B-GGUF", "IFM GGUF card; verify")],
    suites=["smoke", "light_tuning", "medium_tuning", "micro", "language"], sec_per_run=25,
    configs=k2_configs(), verify_tag=True,
))
for _q in ("q4", "q8"):
    _add(ModelSpec(
        key=f"k2-horizon-7b-{_q}", tag=f"k2-horizon:7b-{_q}", family="K2-Horizon (IFM)",
        roles=["quantisation follow-up only"],
        official={"note": f"Run ONLY the promoted configs of k2-horizon-7b-q5 on the {_q.upper()} build"},
        thinking_control="as q5", caveats=["Q8 (~7.5 GB weights) cannot fit 8 GB VRAM with usable context"],
        sources=[], suites=["smoke", "light_tuning", "medium_tuning"], sec_per_run=40,
        configs=[], verify_tag=True, follow_best_of="k2-horizon-7b-q5",
    ))
_add(ModelSpec(
    key="gpt-oss-20b", tag="gpt-oss:20b", family="gpt-oss (OpenAI)",
    roles=["Mastermind/planner candidate (not tested here)", "Medium/High reference"],
    official={
        "sampling": "temperature 1.0, top_p 1.0 (top_k 0 per Unsloth guide)",
        "effort": "reasoning low|medium|high; set via system prompt 'Reasoning: high' or API",
        "format": "Trained on the harmony format; must be used with it (Ollama does this itself)",
    },
    thinking_control="Ollama `think`: 'low' | 'medium' | 'high' (strings), not booleans.",
    caveats=["MXFP4 weights ~13 GB: partial CPU offload on 8 GB VRAM; expect a different speed profile",
             "Earlier HTTP 500s and constraint violations seen in Open WebUI: reproduce through the raw API "
             "first (probe P8) to separate Ollama/memory faults from Open WebUI integration faults",
             "Old benchmark forced top_k=20 and top_p=0.95, not the vendor's top_p 1.0"],
    sources=[("https://huggingface.co/openai/gpt-oss-20b", "OpenAI card/README"),
             ("https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune", "aggregator")],
    suites=["smoke", "light_tuning", "medium_tuning", "language"], sec_per_run=20,
    configs=[
        Config("low-c8k-p3072", "low", opts(8192, 3072, temperature=1.0, top_p=1.0), "Official sampling, effort low, small budget"),
        Config("medium-c8k-p3072", "medium", opts(8192, 3072, temperature=1.0, top_p=1.0), "effort medium, small budget"),
        Config("high-c8k-p3072", "high", opts(8192, 3072, temperature=1.0, top_p=1.0), "effort high, small budget (truncation risk)", deviation=True),
        Config("low-c16k-p8192", "low", opts(16384, 8192, temperature=1.0, top_p=1.0), "effort low, large budget"),
        Config("medium-c16k-p8192", "medium", opts(16384, 8192, temperature=1.0, top_p=1.0), "effort medium, large budget"),
        Config("high-c16k-p8192", "high", opts(16384, 8192, temperature=1.0, top_p=1.0), "effort high, large budget"),
        Config("high-c16k-T06", "high", opts(16384, 8192, temperature=0.6, top_p=1.0),
               "DEVIATION: lower temperature for executor role", deviation=True),
    ],
))
_add(ModelSpec(
    key="granite4.2-8b", tag="granite4.2:8b", family="Granite 4.2 (IBM)",
    roles=["Medium executor candidate (agentic RL on SWE/terminal tasks)", "strict instruction follower"],
    official={
        "sampling": "temperature 1.0 and top_p 0.95 across ALL tasks and backends incl. tool calling (IBM docs)",
        "budget": "max_new_tokens 8192 thinking, 2048 non-thinking",
        "modes": "enable_thinking=True (default) | enable_thinking=False | enable_thinking=True + low_effort=True",
        "ollama": "Ollama page: /set think|nothink|low|medium|high; REST: \"think\": \"high\"; "
                  "Modelfile defaults T0.7 top_p0.9 top_k50 num_ctx8192 (differ from IBM)",
        "languages": "12 listed languages incl. English and Portuguese; Russian is NOT listed",
        "vendor_bench": "IBM-reported: SWE-bench Verified 47.67, BFCL v4 50.29 (below the 3B's 52.41), "
                        "IFBench 79.33, RULER-128K 71.41 (vendor-reported, not independently reproduced)",
    },
    thinking_control="Ollama `think` with true/false/low/medium/high (see official.ollama).",
    caveats=["Russian quality must be probed (P7) before using it on Russian resumes",
             "Native tool calling uses a Qwen3-coder-like tool-call parser; test native tools AND the JSON protocol"],
    sources=[("https://www.ibm.com/granite/docs/models/granite4-2", "IBM docs (read directly)"),
             ("https://ollama.com/library/granite4.2:latest", "Ollama page (read directly)")],
    suites=["smoke", "light_tuning", "medium_tuning", "micro", "language"], sec_per_run=30,
    configs=granite42_configs(),
))
_add(ModelSpec(
    key="granite4.2-3b", tag="granite4.2:3b", family="Granite 4.2 (IBM)",
    roles=["cheap router/classifier", "pipeline stage replacing a small Qwen"],
    official={"sampling": "same as 8B: T1.0, top_p 0.95", "note": "3B skips the agentic-RL stage"},
    thinking_control="same as granite4.2-8b",
    caveats=["Not trained for SWE/terminal agent work; judge it on micro tasks first"],
    sources=[("https://www.ibm.com/granite/docs/models/granite4-2", "IBM docs")],
    suites=["smoke", "light_tuning", "micro", "language"], sec_per_run=12,
    configs=granite42_configs(include_extra=False)[:5],
))
_add(ModelSpec(
    key="granite3.3-8b", tag="granite3.3:8b", family="Granite 3.3 (IBM)",
    roles=["legacy control only"],
    official={"thinking": "chat-template flag thinking=True/False",
              "sampling": "no firm vendor value found; IBM/NVIDIA example uses T0.2; provider default 0.6 (not official)"},
    thinking_control="Ollama `think` may not map to this template: probe P1; otherwise skip thinking configs.",
    caveats=["Low priority. Old results at forced T=1.0 are not comparable (R11)"],
    sources=[("https://huggingface.co/ibm-granite/granite-3.3-8b-instruct", "verify")],
    suites=["smoke", "micro"], sec_per_run=25,
    configs=[
        Config("nt-t02", False, opts(8192, 2048, temperature=0.2), "Low-temperature control"),
        Config("nt-t06", False, opts(8192, 2048, temperature=0.6), "Mid temperature control", deviation=True),
        Config("th-t06", True, opts(16384, 4096, temperature=0.6), "Thinking on (if supported)", deviation=True),
    ],
))

# ----------------------------------------------------------------------------------------
# 6. STAGE 1 PROBES (cheap; run BEFORE the matrix; results gate the matrix)
# ----------------------------------------------------------------------------------------
PROBES = [
    {"id": "P1", "name": "think-toggle", "procedure":
        "For each model and each distinct `think` value in its configs, send a small reasoning prompt "
        "(e.g. multi-step arithmetic). Record HTTP status, presence of message.thinking, eval_count.",
     "pass": "Thinking tokens change with the toggle and with low/medium/high. If not: mark the config "
             "'think_unsupported', use the model's documented fallback, or drop the axis."},
    {"id": "P2", "name": "defaults-dump", "procedure":
        "`ollama show <tag> --modelfile` and `--parameters`; compare with vendor values in this file.",
     "pass": "Every difference is listed (e.g. Granite 4.2: Modelfile T0.7/top_p0.9/top_k50 vs IBM T1.0/top_p0.95)."},
    {"id": "P3", "name": "fit-and-speed", "procedure":
        "For ctx in (8192, 16384, plus 24576 for K2) load the model, send a ~2k-token prompt, record "
        "`ollama ps` (size, CPU/GPU split), tok/s from eval_count/eval_duration, and time-to-first-token.",
     "pass": "Executor candidates run 100% on GPU at the chosen ctx. Anything else is flagged (gpt-oss expected partial)."},
    {"id": "P4", "name": "param-effect", "procedure":
        "Same prompt, same seed, temperature>0, vary ONE parameter (presence_penalty 0 vs 1.5; top_k 20 vs 100; "
        "min_p 0 vs 0.1). Repeat 3x.",
     "pass": "Outputs differ when the parameter changes. If byte-identical, Ollama ignores it for this model: "
             "report it and remove that axis (important for the Qwen presence_penalty configs)."},
    {"id": "P5", "name": "json-protocol", "procedure":
        "20 requests that require the harness' action JSON; measure strict-parse rate with and without `format`.",
     "pass": "Report strict-parse rates per model/config; use structured output only if it raises the rate "
             "without hurting task pass rate."},
    {"id": "P6", "name": "loop-runaway", "procedure":
        "Open-ended reasoning prompt with thinking on and a hard num_predict cap; detect repeated n-grams.",
     "pass": "Report truncation rate and repetition ratio; models above 10% runaway get a lower budget or no thinking."},
    {"id": "P7", "name": "language", "procedure":
        "Mini-sets G1/G2/G3 (Russian, Portuguese, English). Blind scoring by the user on 1-5.",
     "pass": "Any model used on Russian text must score >= 4 on G1. Granite 4.2 does not list Russian: expect a gap."},
    {"id": "P8", "name": "gpt-oss-http500", "procedure":
        "20 sequential requests at ctx 8192 then 16384 straight to the API with only gpt-oss loaded; capture "
        "server.log and free RAM/VRAM; repeat once through Open WebUI.",
     "pass": "Root cause classified: out-of-memory / runner crash / Open WebUI parameter or tool-format issue."},
]

RESULT_FIELDS = [
    "campaign_id", "run_id", "stage", "round", "model_key", "tag", "digest", "config_id", "task_id", "seed",
    "think", "options_sent", "contract", "structured", "ollama_version", "env_hash",
    "wall_s", "load_s", "gen_s", "prompt_tokens", "eval_tokens", "tok_per_s", "truncated",
    "tool_calls", "violations", "claimed_tests_passed", "observed_tests_passed",
    "code_pass", "workflow_pass", "honest_report", "infra_error", "transcript_path",
]


# ----------------------------------------------------------------------------------------
# 7. MATRIX BUILDER
# ----------------------------------------------------------------------------------------
def tasks_in(suite_names) -> list:
    wanted = set(suite_names)
    return [tid for tid, t in TASKS.items() if t["suite"] in wanted and t["status"] != "quarantine"]


def _stage_key(stage: str, round_: int):
    return ("screen", round_) if stage == "screen" else ("validate", 0)


def _suites_for(spec: ModelSpec, stage: str, round_: int) -> set:
    if stage == "screen" and round_ == 1:
        return set(spec.suites) & R1_SUITES
    if stage == "screen":
        return set(spec.suites)
    return {VALIDATION_MAP[s] for s in spec.suites if s in VALIDATION_MAP}


def _run_id(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:12]


def build_matrix(stage: str = "screen", round_: int = 1, promoted: dict | None = None) -> list:
    """Return a list of run dicts. promoted = {model_key: [config_id, ...]} (required for round 2 and validate;
    if absent in round 2 all configs are used, which is only sensible for a dry-run)."""
    runs = []
    seeds = SEEDS[_stage_key(stage, round_)]
    for key, spec in MODELS.items():
        if spec.follow_best_of and stage == "screen":
            continue  # quant followups only run with promoted configs of the leader
        cfgs = spec.configs
        lead = spec
        if spec.follow_best_of:
            lead = MODELS[spec.follow_best_of]
            cfgs = lead.configs
        ids = None
        if stage == "validate" and not promoted:
            raise SystemExit("validate stage needs --promoted (model_key -> config ids)")
        if promoted is not None:
            ids = promoted.get(key) or (promoted.get(spec.follow_best_of) if spec.follow_best_of else None)
            if not ids:
                continue  # model (or all of its configs) was dropped by earlier halving
        if ids is not None:
            cfgs = [c for c in cfgs if c.id in ids]
        suites = _suites_for(spec, stage, round_)
        tasks = tasks_in(suites)
        for cfg in cfgs:
            for task in tasks:
                for seed in seeds:
                    o = dict(cfg.options)
                    o["seed"] = seed
                    runs.append({
                        "run_id": _run_id(key, cfg.id, task, seed, stage, round_),
                        "stage": stage, "round": round_, "model_key": key, "tag": spec.tag,
                        "verify_tag": spec.verify_tag, "config_id": cfg.id, "think": cfg.think,
                        "options": o, "contract": cfg.contract, "structured": cfg.structured,
                        "deviation": cfg.deviation, "task_id": task, "seed": seed,
                        "task_status": TASKS[task]["status"],
                    })
    return runs


def ollama_payload(run: dict, messages: list, tools: list | None = None,
                   format_schema: dict | None = None, keep_alive: str = "10m") -> dict:
    """Build an /api/chat payload for one run. `think` is a top-level field; options carry sampling."""
    payload = {"model": run["tag"], "messages": messages, "stream": False,
               "options": run["options"], "keep_alive": keep_alive}
    if run["think"] is not None:
        payload["think"] = run["think"]
    if tools:
        payload["tools"] = tools
    if run.get("structured") and format_schema:
        payload["format"] = format_schema
    return payload


def modelfile_text(model_key: str, config_id: str) -> str:
    """PRINT-ONLY helper. Applying it (ollama create) requires explicit user confirmation."""
    spec = MODELS[model_key]
    src = MODELS[spec.follow_best_of] if spec.follow_best_of else spec
    cfg = next(c for c in src.configs if c.id == config_id)
    lines = [f"FROM {spec.tag}"]
    for k, v in cfg.options.items():
        if k != "seed":
            lines.append(f"PARAMETER {k} {v}")
    lines.append(f"# think={cfg.think!r} is a per-request field, not a Modelfile parameter")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------
# 8. STATISTICS HELPERS
# ----------------------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def promote(results: dict, keep_frac: float = 0.5, min_keep: int = 2) -> list:
    """results: {config_id: {"passes": [0/1...], "gen_s": [float...]}} -> ids to keep (successive halving)."""
    def key(item):
        cid, r = item
        k, n = sum(r["passes"]), len(r["passes"])
        lo = wilson(k, n)[0]
        med = sorted(r["gen_s"])[len(r["gen_s"]) // 2] if r.get("gen_s") else 1e9
        return (-(k / n if n else 0), -lo, med)
    ranked = sorted(results.items(), key=key)
    keep = max(min_keep, math.ceil(len(ranked) * keep_frac))
    return [cid for cid, _ in ranked[:keep]]


# ----------------------------------------------------------------------------------------
# 9. CLI
# ----------------------------------------------------------------------------------------
def _estimate_minutes(runs: list) -> float:
    total = 0.0
    for r in runs:
        spec = MODELS[r["model_key"]]
        mult = 2.5 if r["think"] not in (None, False) else 1.0
        total += spec.sec_per_run * mult
    return total / 60.0


def cmd_summary(args) -> None:
    promoted = json.load(open(args.promoted)) if args.promoted else None
    if args.stage == "validate":
        args.round = 0
    runs = build_matrix(args.stage, args.round, promoted)
    print(f"spec {SPEC_VERSION}  stage={args.stage} round={args.round}  total runs={len(runs)}  "
          f"rough hours={_estimate_minutes(runs)/60:.1f}  (planning guess only)")
    by = {}
    for r in runs:
        by.setdefault(r["model_key"], []).append(r)
    print(f"{'model':28} {'configs':>7} {'tasks':>6} {'runs':>6} {'~min':>6}")
    for k, rs in by.items():
        print(f"{k:28} {len({r['config_id'] for r in rs}):7d} {len({r['task_id'] for r in rs}):6d} "
              f"{len(rs):6d} {_estimate_minutes(rs):6.0f}")
    need = [t for t, v in TASKS.items() if v["status"] == "author"]
    print("\nTasks that must be AUTHORED first (R7/R8):", ", ".join(need))
    print("Quarantined:", ", ".join(t for t, v in TASKS.items() if v["status"] == "quarantine"))
    unv = [k for k, s in MODELS.items() if s.verify_tag]
    print("Tags to verify with `ollama list`:", ", ".join(unv))


def cmd_show(args) -> None:
    s = MODELS[args.model]
    print(f"== {s.key}  (tag {s.tag}){'  [VERIFY TAG]' if s.verify_tag else ''}")
    print("roles:", "; ".join(s.roles))
    print("official:")
    for k, v in s.official.items():
        print(f"  - {k}: {v}")
    print("thinking control:", s.thinking_control)
    print("caveats:")
    for c in s.caveats:
        print("  -", c)
    print("sources:")
    for u, p in s.sources:
        print(f"  - {u}  [{p}]")
    print("suites:", ", ".join(s.suites))
    src = MODELS[s.follow_best_of] if s.follow_best_of else s
    print("configs:" + ("  (inherited from " + src.key + ")" if s.follow_best_of else ""))
    for c in src.configs:
        flag = "DEV" if c.deviation else "OFF"
        print(f"  [{flag}] {c.id:22} think={c.think!s:6} contract={c.contract:8} json={c.structured!s:5} {c.options}")
        print(f"        {c.purpose}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("summary", "dump"):
        p = sub.add_parser(name)
        p.add_argument("--stage", default="screen", choices=["screen", "validate"])
        p.add_argument("--round", type=int, default=1)
        p.add_argument("--promoted", default=None, help="JSON file {model_key: [config_id,...]}")
    sub.add_parser("rules")
    sub.add_parser("env")
    sub.add_parser("probes")
    p = sub.add_parser("show"); p.add_argument("model", choices=list(MODELS))
    p = sub.add_parser("modelfile"); p.add_argument("model", choices=list(MODELS)); p.add_argument("config")
    p = sub.add_parser("wilson"); p.add_argument("k", type=int); p.add_argument("n", type=int)
    args = ap.parse_args()

    if args.cmd == "summary":
        cmd_summary(args)
    elif args.cmd == "dump":
        promoted = json.load(open(args.promoted)) if args.promoted else None
        rnd = 0 if args.stage == "validate" else args.round
        print(json.dumps(build_matrix(args.stage, rnd, promoted), ensure_ascii=False, indent=1))
    elif args.cmd == "rules":
        print("\n".join(GLOBAL_RULES)); print(); print("\n".join(DECISION_RULES))
        print("\nResult record fields:", ", ".join(RESULT_FIELDS))
    elif args.cmd == "env":
        print(json.dumps(ENVIRONMENT, ensure_ascii=False, indent=1))
    elif args.cmd == "probes":
        print(json.dumps(PROBES, ensure_ascii=False, indent=1))
    elif args.cmd == "show":
        cmd_show(args)
    elif args.cmd == "modelfile":
        print(modelfile_text(args.model, args.config))
    elif args.cmd == "wilson":
        lo, hi = wilson(args.k, args.n)
        print(f"{args.k}/{args.n}: 95% Wilson interval {lo:.0%} - {hi:.0%}")


if __name__ == "__main__":
    main()
