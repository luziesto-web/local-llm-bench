"""Night screen of the zoo matrix (zoo_test_spec.py, stage screen, round 1) on the reviewed
round2 Light tool episodes. Everything below is decided AT LAUNCH, from the live Ollama state:

1. Inventory: /api/tags + /api/show for every tag. Read-only: never creates, edits or deletes tags.
2. Originals: the spec tags (aliases below) are the only tags the matrix runs on.
3. Baked tags: every other tag built on the same weights blob as an original. Its effective
   parameters are compared with the matrix configs of that model:
   * a config already sends the same parameters -> the baked tag is dropped from the variants
     (the original + that config covers it);
   * otherwise its parameters become a new config "baked-<tag>" that runs on the ORIGINAL tag,
     on equal terms with the other configs (phase 1).
   Baked tags whose template/SYSTEM differs from the original cannot be expressed as options;
   they are listed and run on the baked tag itself.
4. Plan + time estimate. Phase 1 = vendor configs + baked configs; phase 2 = spec deviations /
   ablations. No new episode starts if it would end after the deadline.
5. Serial episodes via run_light.episode (same tools, limits and acceptance as the contest).

  python zoo_night.py plan [--hours 7] [--models KEY ...]
  python zoo_night.py run [--name zoo-night-YYYYMMDD] [--hours 7] [--profile all|core] [--models KEY ...]
"""
import argparse
import contextlib
import ctypes
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND2 = HERE.parent
sys.path.insert(0, str(ROUND2))
sys.path.insert(0, str(HERE))
import zoo_test_spec as spec  # noqa: E402
import run_light as runner    # noqa: E402
import light10_cases           # noqa: E402,F401  L6-L11: the sandbox (light_worker) checks these, prepare must use them too
from suite import RUNS, CASES  # noqa: E402

# spec tag -> installed original. Checked 2026-10-27 (ollama list/show); re-verified at launch.
TAG_ALIASES = {
    "k2-horizon:7b-q5": "hf.co/bartowski/K2-Horizon-7B-GGUF:Q5_K_M",
    "k2-horizon:7b-q8": "hf.co/bartowski/K2-Horizon-7B-GGUF:Q8_0",
    "k2-horizon:7b-q4": "ingloryon/k2-horizon:7b-q4_K_M",
    "ministral-3-8b-reasoningq4": "hf.co/mistralai/Ministral-3-8B-Reasoning-2512-GGUF:Q4_K_M",
}
# Removed from the night plan by the user (2026-10-27), even if the tag is installed again.
# Ministral 3 (Instruct and Reasoning): DROP in ministral-trial-20261027; models stay on disk.
EXCLUDED = {"granite3.3-8b", "ministral3-8b-instruct", "ministral3-8b-reasoning"}
COMPARE_KEYS = ("num_ctx", "num_predict", "temperature", "top_p", "top_k", "min_p", "typical_p",
                "presence_penalty", "frequency_penalty", "repeat_penalty", "repeat_last_n")
# New models (HANDOFF §8), added to the night by the user on 2026-10-27. Not in zoo_test_spec yet:
# one config each = vendor-recommended sampling (OFFICIAL_NEW), Medium budget.
# Skipped at launch if the tag is not installed yet.
NEW_MODELS = {
    "qwen3-coder-30b": "qwen3-coder:30b", "nemotron-3.5-lightning-30b": "nemotron-3.5-lightning:30b",
    "devstral-24b": "devstral:24b", "bonsai-2-9b": "bonsai-2:9b", "glm-4.7-flash": "glm-4.7-flash:latest",
}
for _key, _tag in NEW_MODELS.items():
    spec.MODELS.setdefault(_key, spec.ModelSpec(
        key=_key, tag=_tag, family="new (HANDOFF §8)", roles=["Medium candidate"], official={},
        thinking_control="from tag capabilities", caveats=["parameters require verification against the vendor card"],
        sources=[], suites=["medium_tuning"], sec_per_run=0))


# Vendor recommendations, read on the official pages 2026-10-27 (think = the vendor's default mode).
OFFICIAL_NEW = {
    # huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct: non-thinking only
    "qwen3-coder-30b": (None, dict(temperature=0.7, top_p=0.8, top_k=20, repeat_penalty=1.05)),
    # huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16: "Temperature 1.0, Top_P 0.95", thinking on by default
    "nemotron-3.5-lightning-30b": (True, dict(temperature=1.0, top_p=0.95)),
    # huggingface.co/mistralai/Devstral-Small-2507 (Ollama devstral:24b = fine-tune of Mistral Small 3.1): temperature 0.15
    "devstral-24b": (None, dict(temperature=0.15)),
    # docs.prismml.com/models/bonsai-2-9b, thinking (default) column
    "bonsai-2-9b": (True, dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.0,
                               repeat_penalty=1.0)),
    # huggingface.co/zai-org/GLM-4.7-Flash: Terminal Bench / SWE-bench Verified preset (agentic coding)
    "glm-4.7-flash": (True, dict(temperature=0.7, top_p=1.0)),
}


def new_model_config(inv, key, tag):
    think, sampling = OFFICIAL_NEW[key]
    if think and "thinking" not in inv[tag]["capabilities"]:
        think = None   # the tag does not advertise thinking; keep the model default
    if key in MMAP_MODELS:
        # 2026-10-27: without mmap Ollama pins ~13 GB of RAM-side weights (CUDA_Host) and the load dies with
        # "CUDA error: out of memory"; use_mmap keeps them as a mapped file, the model loads (probe: 10-20 tok/s).
        return spec.Config("vendor-mmap", think, dict(sampling, num_ctx=16384, num_predict=8192, use_mmap=True),
                           "Vendor sampling + use_mmap (pinned-memory OOM fix on this PC); Medium budget 16K/8K")
    return spec.Config("vendor", think, dict(sampling, num_ctx=16384, num_predict=8192),
                       "Vendor-recommended sampling (official card, read 2026-10-27); Medium budget 16K/8K")


MMAP_MODELS = {"qwen3-coder-30b", "nemotron-3.5-lightning-30b", "glm-4.7-flash"}


# Each model gets tasks of the tier(s) it competes for (user decisions 2026-10-27):
# tonight = light + medium; the heavy models wait for a separate High night (`--levels high`);
# borderline models (K2, Qwen 9B) play Medium tonight AND High later.
LEVEL_TASKS = {"light": ["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9"],   # 10 since 2026-10-28
               "medium": ["M1", "M2", "M3", "M4", "M5"],
               "high": ["H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9"]}   # 10 since 2026-10-28
NIGHT_LEVELS = ("light", "medium")
GROUPS = {
    "qwen3.5-0.8b": ("light",), "gemma4-e4b": ("light",), "granite4.2-3b": ("light",),
    "ministral3-8b-instruct": ("light",),
    "qwen3.5-4b": ("medium",), "granite4.2-8b": ("medium",), "ministral3-8b-reasoning": ("medium",),
    "bonsai-2-9b": ("medium",),
    # borderline
    "qwen3.5-9b": ("medium", "high"), "k2-horizon-7b-q5": ("medium", "high"),
    "k2-horizon-7b-q4": ("medium", "high"), "k2-horizon-7b-q8": ("medium", "high"),
    # heavy: High only
    "gpt-oss-20b": ("high",), "qwen3-coder-30b": ("high",), "nemotron-3.5-lightning-30b": ("high",),
    "glm-4.7-flash": ("high",), "devstral-24b": ("high",),
}
# Rough seconds per episode. Light (12 calls / 300 s): round2 episode.json medians. Medium (30 calls / 900 s):
# medium-contest-v4 medians x1.3 (qwen, k2, granite 4.2); gpt-oss and Ministral Reasoning have no episode data (guess).
EST = {
    "light": {
        "qwen3.5-0.8b": dict(off=8, on=60), "gemma4-e4b": dict(off=20, on=120),
        "granite4.2-3b": dict(off=15, low=15, on=120), "ministral3-8b-instruct": dict(off=40),
    },
    "medium": {
        "qwen3.5-9b": dict(off=70, on=140), "qwen3.5-4b": dict(off=40, on=80),
        "k2-horizon-7b-q5": dict(off=80, low=100, medium=130, high=160),
        "k2-horizon-7b-q4": dict(off=75, low=95, medium=120, high=150),
        "k2-horizon-7b-q8": dict(off=105, low=130, medium=170, high=210),
        "gpt-oss-20b": dict(low=120, medium=200, high=300),
        "granite4.2-8b": dict(off=50, low=60, on=450), "ministral3-8b-reasoning": dict(off=250),
        # new models: no measurements at all; MoE with ~3B active (qwen3-coder, nemotron, glm) partly in RAM,
        # devstral is a dense 24B mostly in RAM (slow). Pure guesses.
        "qwen3-coder-30b": dict(off=180, on=300), "nemotron-3.5-lightning-30b": dict(off=200, on=300),
        "glm-4.7-flash": dict(off=200, on=300), "devstral-24b": dict(off=500, on=600), "bonsai-2-9b": dict(off=90, on=150),
    },
    # High (60 calls / 3600 s): K2 Q5 in high-v7 took 271-1650 s per task (median ~390). The rest are guesses.
    "high": {
        "qwen3.5-9b": dict(off=300, on=600), "k2-horizon-7b-q5": dict(off=400, low=500, medium=650, high=800),
        "k2-horizon-7b-q4": dict(off=380, low=480, medium=600, high=750),
        "k2-horizon-7b-q8": dict(off=500, low=650, medium=850, high=1000),
        "gpt-oss-20b": dict(low=500, medium=700, high=900),
        "qwen3-coder-30b": dict(off=900, on=1200), "nemotron-3.5-lightning-30b": dict(off=900, on=1200),
        "glm-4.7-flash": dict(off=900, on=1200), "devstral-24b": dict(off=1800, on=2000),
    },
}
# Moved to phase 2 to fit 7 h: on this install granite4.2:8b ships T1.0/top_p 0.95, so "olldef"
# (T0.7/top_p 0.9/top_k 50) is not the Ollama default the spec assumed, just another sampling variant.
# Also moved to phase 2 by the user (2026-10-27) to make room for the new models within 7 h:
# K2 short-output variants (p4096) and Granite 8B th-ibm.
DEMOTED = {("granite4.2-8b", "th-olldef"), ("granite4.2-8b", "th-ibm"),
           ("k2-horizon-7b-q5", "low-p4096"), ("k2-horizon-7b-q5", "medium-p4096"), ("k2-horizon-7b-q5", "high-p4096")}
HEAVY_SECONDS = 300  # phase-1 episodes this long run after the cheaper ones, so a late cut hits configs, not models
# Gemma never creates candidate_tests.py (transcripts of light v5 / medium v4). Ablation: same vendor config,
# but run_tests on a missing file answers with a hint. Separate config, never mixed into the plain one.
HINT = ("candidate_tests.py does not exist. You must create it yourself with write_file "
        "(assert-based tests importing the task modules), then call run_tests again.")
EXTRA_CONFIGS = {
    "gemma4-e4b": [spec.Config("nt-official-hint", False,
                               {"num_ctx": 8192, "num_predict": 2048, "temperature": 1.0, "top_p": 0.95, "top_k": 64},
                               "ABLATION: nt-official + hint when run_tests finds no candidate_tests.py")],
}
SWITCH_SECONDS = 30
ACTION_SCHEMA = {"type": "object", "properties": {
    "action": {"type": "string", "enum": ["list_files", "read_file", "write_file", "run_tests", "run_command", "final"]},
    "path": {"type": "string"}, "command": {"type": "string"}, "content": {"type": "string"}, "report": {"type": "string"},
    "tests_passed": {"type": "boolean"}}, "required": ["action"]}
EXPLICIT = "Write this file yourself using assert statements"
IMPLICIT = "It uses assert statements"   # approximation of the old implicit contract (spec: *-implicit)


# ------------------------------------------------------------------ inventory
def slug(s):
    return re.sub(r"[^A-Za-z0-9_-]+", "-", s).strip("-")


def _num(v):
    v = v.strip().strip('"')
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def parse_params(text):
    out = {}
    for line in (text or "").splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            k, v = parts
            if k == "stop":
                out.setdefault("stop", []).append(v.strip().strip('"'))
            else:
                out[k] = _num(v)
    return out


def inventory():
    inv = {}
    for m in runner.api("tags")["models"]:
        show = runner.api("show", dict(model=m["name"]), timeout=60)
        mf = show.get("modelfile", "")
        frm = [l.split(None, 1)[1].strip() for l in mf.splitlines() if l.startswith("FROM ")]
        inv[m["name"]] = dict(
            digest=m.get("digest"), size=m.get("size"),
            blob=Path(frm[0]).name if frm else None,
            params=parse_params(show.get("parameters")),
            template=runner.digest(show.get("template", "")),
            system=runner.digest(show.get("system", "") or "\n".join(l for l in mf.splitlines() if l.startswith("SYSTEM"))),
            capabilities=show.get("capabilities", []), details=show.get("details"))
    return inv


def effective(base, over):
    eff = {k: base[k] for k in COMPARE_KEYS if k in base}
    eff.update({k: over[k] for k in COMPARE_KEYS if k in over})
    return eff


def same(a, b):
    if set(a) != set(b):
        return False
    return all(abs(float(a[k]) - float(b[k])) < 1e-9 if isinstance(a[k], (int, float)) and isinstance(b[k], (int, float))
               else a[k] == b[k] for k in a)


def light_think(caps):
    """How run_light ran baked '-t' tags in the Light contest (reasoning=auto): thinking off when supported."""
    return False if "thinking" in caps else None


def reconcile(inv):
    originals, missing = {}, []
    for key, ms in spec.MODELS.items():
        if key in EXCLUDED:
            continue
        tag = ms.tag if ms.tag in inv else TAG_ALIASES.get(ms.tag)
        if tag in inv:
            originals[key] = tag
        else:
            missing.append(dict(model_key=key, spec_tag=ms.tag, alias=TAG_ALIASES.get(ms.tag)))
    baked_cfgs = {k: [] for k in originals}
    report = []
    by_blob = {}
    for key, tag in originals.items():
        by_blob.setdefault(inv[tag]["blob"], []).append(key)
    for name in sorted(inv):
        if name in originals.values():
            continue
        keys = by_blob.get(inv[name]["blob"])
        if not keys:
            report.append(dict(tag=name, decision="outside_matrix", reason="weights not used by any matrix model"))
            continue
        key = keys[0]
        ms, base = spec.MODELS[key], inv[originals[key]]
        diff = {k: v for k, v in inv[name]["params"].items() if k in COMPARE_KEYS and base["params"].get(k) != v}
        if inv[name]["template"] != base["template"] or inv[name]["system"] != base["system"]:
            cfg = spec.Config(f"bakedtag-{slug(name)}", None, {}, f"baked tag {name} run as-is (template/SYSTEM differ)")
            baked_cfgs[key].append((cfg, name))
            report.append(dict(tag=name, model_key=key, decision="run_baked_tag_itself",
                               reason="template or SYSTEM differs from the original; not expressible as options"))
            continue
        eff = effective(base["params"], inv[name]["params"])
        cfgs = spec.MODELS[ms.follow_best_of].configs if ms.follow_best_of else ms.configs
        dup = next((c.id for c in cfgs if same(eff, effective(base["params"], c.options))), None)
        dup = dup or next((c.id for c, _ in baked_cfgs[key] if same(eff, c.options)), None)
        if dup:
            report.append(dict(tag=name, model_key=key, decision="dropped_duplicate", same_as=dup, baked_params=diff))
            continue
        cfg = spec.Config(f"baked-{slug(name)}", light_think(base["capabilities"]), eff,
                          f"parameters of the user's baked tag {name}, run on the original tag")
        baked_cfgs[key].append((cfg, None))
        report.append(dict(tag=name, model_key=key, decision="added_config", config_id=cfg.id,
                           think=cfg.think, options=eff, baked_params=diff))
    return originals, missing, baked_cfgs, report


# ------------------------------------------------------------------ plan
def think_class(key, think):
    if key == "ministral3-8b-reasoning":
        return "off"
    if think in (None, False):
        return "off"
    return "on" if think is True else str(think)


def est_seconds(level, key, think):
    table = EST[level].get(key, {})
    return table.get(think_class(key, think)) or table.get("on") or max(table.values(), default=300)


def build_plan(inv, only=None, profile="all", levels=NIGHT_LEVELS):
    originals, missing, baked_cfgs, report = reconcile(inv)
    runs = []
    for key, tag, level in [(k, t, lv) for k, t in originals.items() for lv in GROUPS.get(k, ())]:
        if (only and key not in only) or level not in levels:
            continue
        ms = spec.MODELS[key]
        tasks = [t for t in LEVEL_TASKS[level] if t in CASES]
        seeds = spec.SEEDS[("screen", 1)]
        cfgs = [] if ms.follow_best_of else [(c, None) for c in ms.configs]   # quant followers: baked only
        cfgs += [(c, None) for c in EXTRA_CONFIGS.get(key, [])]
        if key in NEW_MODELS:
            cfgs = [(new_model_config(inv, key, tag), None)]
        for cfg, run_tag in cfgs + baked_cfgs.get(key, []):
            phase = 2 if cfg.deviation or (key, cfg.id) in DEMOTED else 1
            if profile == "core" and phase == 2:
                continue
            for task in tasks:
                for seed in seeds:
                    runs.append(dict(phase=phase, level=level, model_key=key, tag=run_tag or tag, original=tag,
                                     config_id=cfg.id, think=cfg.think, options=dict(cfg.options, seed=seed),
                                     contract=cfg.contract, structured=cfg.structured, deviation=cfg.deviation,
                                     hint=cfg.id.endswith("-hint"),
                                     baked=cfg.id.startswith(("baked-", "bakedtag-")), purpose=cfg.purpose,
                                     task_id=task, seed=seed, est_seconds=est_seconds(level, key, cfg.think)))
    order = [k for k in GROUPS if k in originals]
    # New models go first among long phase-1 episodes, so a late cut hits others.
    # Task-major ordering: finish each task across every selected model/config before advancing.
    runs.sort(key=lambda r: (list(LEVEL_TASKS).index(r["level"]),
                             LEVEL_TASKS[r["level"]].index(r["task_id"]), r["phase"],
                             r["est_seconds"] >= HEAVY_SECONDS, r["model_key"] not in NEW_MODELS,
                             order.index(r["model_key"]), r["config_id"]))
    return dict(originals=originals, missing=missing, baked=report, runs=runs)


def estimate(runs):
    secs, prev = 0, None
    for r in runs:
        secs += r["est_seconds"] + (SWITCH_SECONDS if r["tag"] != prev else 0)
        prev = r["tag"]
    return secs


def print_plan(plan, hours):
    print("Originals used by the matrix:")
    for k, t in plan["originals"].items():
        print(f"  {k:26} {t}")
    for m in plan["missing"]:
        print(f"  MISSING {m['model_key']} (spec tag {m['spec_tag']}, alias {m['alias']}): skipped")
    for k in sorted(EXCLUDED):
        print(f"  EXCLUDED {k}: removed from the night plan by the user")
    print("\nBaked / other tags:")
    for b in plan["baked"]:
        extra = b.get("same_as") or b.get("config_id") or b.get("reason", "")
        print(f"  {b['decision']:22} {b['tag']:58} {extra}")
    print("\nRuns per model (phase1 / phase2) and estimate:")
    rows = {}
    for r in plan["runs"]:
        d = rows.setdefault((r["level"], r["model_key"]), [0, 0, 0])
        d[r["phase"] - 1] += 1
        d[2] += r["est_seconds"]
    for (level, k), (p1, p2, s) in rows.items():
        print(f"  {level:7} {k:26} {p1:4} {p2:4}   ~{s / 60:5.0f} min")
    p1 = [r for r in plan["runs"] if r["phase"] == 1]
    print(f"\nTotal runs {len(plan['runs'])}: phase 1 {len(p1)} ~{estimate(p1) / 3600:.1f} h, "
          f"all ~{estimate(plan['runs']) / 3600:.1f} h; deadline {hours} h "
          f"(phase 2 runs only while time is left).")


# ------------------------------------------------------------------ run
CURRENT = {}
_api, _system_prompt, _act = runner.api, runner.system_prompt, runner.act


def _patched_act(work, key, action):
    if CURRENT.get("hint") and not (work / "candidate_tests.py").exists() and (
            action.get("action") == "run_tests" or action.get("command") == "python candidate_tests.py"):
        raise FileNotFoundError(HINT)
    return _act(work, key, action)


def _patched_api(route, payload=None, timeout=15):
    if route == "chat" and payload is not None and CURRENT.get("structured") and runner.TOOL_MODE != "native":
        payload = dict(payload, format=ACTION_SCHEMA)
    return _api(route, payload, timeout)


def _patched_system_prompt(level):
    text = _system_prompt(level)
    if CURRENT.get("implicit"):
        if EXPLICIT not in text:
            raise RuntimeError("implicit-contract patch no longer applies to run_light.SYSTEM")
        text = text.replace(EXPLICIT, IMPLICIT)
    return text


runner.api, runner.system_prompt, runner.act = _patched_api, _patched_system_prompt, _patched_act


def think_fields(r, caps):
    if r["model_key"] == "granite3.3-8b":   # Granite 3.3 thinks via a control message, not `think`
        return dict(think_request=None, control_thinking=bool(r["think"]),
                    reasoning_status="granite_control_on" if r["think"] else "control_off")
    return dict(think_request=r["think"], control_thinking=False, reasoning_status=f"think={r['think']!r}")


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


# ------------------------------------------------------------------ telemetry (for questions not asked yet)
class _MemoryStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def machine_snapshot():
    """RAM, VRAM/GPU and what Ollama holds where. Every part is optional: never breaks a run."""
    snap = dict(at=datetime.now(timezone.utc).isoformat())
    with contextlib.suppress(Exception):
        m = _MemoryStatus()
        m.dwLength = ctypes.sizeof(_MemoryStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        snap.update(ram_free_gb=round(m.ullAvailPhys / 2 ** 30, 2), ram_load_pct=m.dwMemoryLoad,
                    commit_free_gb=round(m.ullAvailPageFile / 2 ** 30, 2))
    with contextlib.suppress(Exception):
        line = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True,
                              timeout=10).stdout.strip().splitlines()[0]
        used, free, util, temp = (float(x) for x in line.split(","))
        snap.update(vram_used_mb=used, vram_free_mb=free, gpu_util_pct=util, gpu_temp_c=temp)
    with contextlib.suppress(Exception):
        snap["ollama_ps"] = [dict(name=m["name"], size=m.get("size"), size_vram=m.get("size_vram"),
                                  gpu_share=round(m["size_vram"] / m["size"], 3) if m.get("size") else None,
                                  context=m.get("context_length"))
                             for m in _api("ps", None, 10).get("models", [])]
    return snap


def transcript_stats(path, num_ctx=None):
    """Speed, thinking vs answer volume, context pressure, truncations - from the episode transcript."""
    s = dict(responses=0, eval_tokens=0, eval_seconds=0.0, prompt_eval_tokens=0, prompt_eval_seconds=0.0,
             prompt_eval_tokens_max=0, load_seconds=0.0, thinking_chars=0, content_chars=0, done_length=0,
             tool_errors=0, actions={})
    with contextlib.suppress(OSError):
        for line in path.open(encoding="utf-8"):
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get("event") == "tool_result" and "error" in (ev.get("result") or {}):
                s["tool_errors"] += 1
            if ev.get("event") != "response":
                continue
            p = ev.get("payload") or {}
            msg = p.get("message") or {}
            s["responses"] += 1
            s["eval_tokens"] += p.get("eval_count") or 0
            s["eval_seconds"] += (p.get("eval_duration") or 0) / 1e9
            s["prompt_eval_tokens"] += p.get("prompt_eval_count") or 0
            s["prompt_eval_seconds"] += (p.get("prompt_eval_duration") or 0) / 1e9
            s["prompt_eval_tokens_max"] = max(s["prompt_eval_tokens_max"], p.get("prompt_eval_count") or 0)
            s["load_seconds"] += (p.get("load_duration") or 0) / 1e9
            s["thinking_chars"] += len(msg.get("thinking") or "")
            s["content_chars"] += len(msg.get("content") or "")
            s["done_length"] += p.get("done_reason") == "length"
            with contextlib.suppress(Exception):
                action = json.loads(msg.get("content") or "").get("action", "?")
                s["actions"][action] = s["actions"].get(action, 0) + 1
    s["gen_tok_per_s"] = round(s["eval_tokens"] / s["eval_seconds"], 2) if s["eval_seconds"] else None
    s["prompt_tok_per_s"] = round(s["prompt_eval_tokens"] / s["prompt_eval_seconds"], 1) if s["prompt_eval_seconds"] else None
    # prompt_eval_count may exclude a cached prefix, so this is a lower bound of how full the context got
    s["ctx_fill_max_lower_bound"] = round(s["prompt_eval_tokens_max"] / num_ctx, 3) if num_ctx else None
    return s


def episode_telemetry(root, options, start):
    tel = dict(start=start, end=machine_snapshot(),
               transcript=transcript_stats(root / "transcript.jsonl", (options or {}).get("num_ctx")))
    with contextlib.suppress(OSError):
        if root.exists():
            write_json(root / "telemetry.json", tel)
    return tel


def run(args):
    night = RUNS / args.name
    night.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    deadline = started + timedelta(hours=args.hours)
    earlier = sorted(night.glob("plan-*.json"))
    if earlier and not args.new_deadline:   # resume: keep the first deadline, so a restart needs no time arithmetic
        deadline = datetime.fromisoformat(json.loads(earlier[0].read_text(encoding="utf-8"))["deadline"])
        print(f"Resuming {args.name}: keeping the original deadline {deadline:%Y-%m-%d %H:%M} UTC", flush=True)
    inv = inventory()
    plan = build_plan(inv, args.models, args.profile, args.levels)
    write_json(night / f"inventory-{started:%Y%m%dT%H%M%SZ}.json", inv)
    write_json(night / f"plan-{started:%Y%m%dT%H%M%SZ}.json", dict(plan, started=started, deadline=deadline,
                                                                    profile=args.profile))
    print_plan(plan, args.hours)
    results = night / "results.jsonl"
    done = skipped = 0
    prev_tag = None
    for i, r in enumerate(plan["runs"], 1):
        label = "run"
        output = RUNS / f"{args.name}-{slug(r['model_key'])}-{slug(r['config_id'])}-s{r['seed']}"
        root = RUNS / f"{output.name}-{label}" / r["task_id"]
        # pid + folder: the dashboard shows an unfinished episode as running only while this process plays it
        # (an episode killed by a restart is otherwise a "ghost" until its transcript is 2 hours old)
        status = dict(updated_at=datetime.now(timezone.utc), deadline=deadline, index=i, total=len(plan["runs"]),
                      done=done, skipped_deadline=skipped, pid=os.getpid(),
                      current=dict(model=r["model_key"], config=r["config_id"], task=r["task_id"], tag=r["tag"],
                                   folder=str(root)))
        write_json(night / "status.json", status)
        if (root / "episode.json").exists():
            done += 1
            continue                                  # resume: finished earlier
        if datetime.now(timezone.utc) + timedelta(seconds=r["est_seconds"]) > deadline:
            skipped += 1
            with results.open("a", encoding="utf-8") as f:
                f.write(json.dumps(dict(r, outcome="skipped_deadline"), ensure_ascii=False, default=str) + "\n")
            continue
        if root.exists():                             # interrupted earlier: keep it, never overwrite
            root.rename(root.with_name(f"{root.name}-interrupted-{datetime.now():%Y%m%d%H%M%S}"))
        if prev_tag and prev_tag != r["tag"]:
            with contextlib.suppress(Exception):      # unload the previous model from memory (not a delete)
                _api("generate", dict(model=prev_tag, keep_alive=0), 120)
        prev_tag = r["tag"]
        output.mkdir(exist_ok=True)
        caps = inv[r["tag"]]["capabilities"]
        configuration = dict(interface="ollama-custom-json-tools-v2", level=r["level"],
                             config_id=f"{r['model_key']}/{r['config_id']}", options=r["options"],
                             models={label: dict(tag=r["tag"], digest=inv[r["tag"]]["digest"], **think_fields(r, caps))},
                             limits=dict(calls=runner.LEVELS[r["level"]]["calls"],
                                         seconds=runner.LEVELS[r["level"]]["seconds"]),
                             night=args.name, phase=r["phase"], baked=r["baked"], contract=r["contract"],
                             structured=r["structured"], hint=r["hint"], purpose=r["purpose"],
                             tool_mode=r.get("tool_mode", "json"))
        write_json(output / "configuration.json", configuration)
        CURRENT.update(structured=r["structured"], implicit=r["contract"] == "implicit", hint=r["hint"])
        # 2026-10-28: "native" offers the actions as Ollama tools (OpenCode-like); "json" is the historical protocol
        runner.TOOL_MODE = r.get("tool_mode", "json")
        start = machine_snapshot()
        try:
            row = runner.episode(output, label, r["task_id"], configuration)
            outcome = "episode"
        except Exception as exc:                      # infrastructure, not a model score
            row, outcome = dict(error=f"{type(exc).__name__}: {exc}"), "runner_error"
        telemetry = episode_telemetry(root, r["options"], start)
        done += 1
        with results.open("a", encoding="utf-8") as f:
            f.write(json.dumps(dict(r, outcome=outcome, row=row, telemetry=telemetry),
                               ensure_ascii=False, default=str) + "\n")
    write_json(night / "status.json", dict(updated_at=datetime.now(timezone.utc), finished=True, done=done,
                                           skipped_deadline=skipped, total=len(plan["runs"])))
    print(f"ZOO_NIGHT_COMPLETE done={done} skipped_deadline={skipped} total={len(plan['runs'])}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("plan", "run"):
        sp = sub.add_parser(name)
        sp.add_argument("--hours", type=float, default=7.0)
        sp.add_argument("--profile", choices=["all", "core"], default="all")
        sp.add_argument("--models", nargs="+", choices=list(spec.MODELS))
        sp.add_argument("--levels", nargs="+", choices=list(LEVEL_TASKS), default=list(NIGHT_LEVELS),
                        help="tonight: light medium (default); the heavy night: high")
        if name == "run":
            sp.add_argument("--name", default=f"zoo-night-{datetime.now():%Y%m%d}")
            sp.add_argument("--new-deadline", action="store_true",
                            help="when resuming another day: deadline = now + --hours instead of the first launch's")
    args = p.parse_args()
    if args.command == "plan":
        inv = inventory()
        print_plan(build_plan(inv, args.models, args.profile, args.levels), args.hours)
    else:
        run(args)


if __name__ == "__main__":
    main()
