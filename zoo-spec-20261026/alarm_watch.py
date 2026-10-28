"""Emergency alarm for an unattended night. Runs next to night_supervisor.py, reads only, never acts.

It exits (which wakes Claude: the background task finishes) on the first alarm, printing ALARM and the reason:
  * Ollama does not answer for more than OLLAMA_DOWN_MIN minutes;
  * the supervisor gave up a stage or could not restart Ollama (its log says so);
  * the supervisor process is gone but it never logged "supervisor done";
  * ERROR_STREAK episodes in a row ended with runner_error / infra_error (systematic failure, e.g. a model
    that does not fit in memory) — the supervisor would keep burning the night on them;
  * free disk space below MIN_DISK_GB.
When the supervisor finishes normally it exits with DONE (no alarm). Check every CHECK_SECONDS.

  python alarm_watch.py --night zoo-night-20261027 --stage zoo-night-20261027 --stage zoo-high-20261027
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent / "runs"
CHECK_SECONDS = 60
OLLAMA_DOWN_MIN = 5
ERROR_STREAK = 4
MIN_DISK_GB = 8
IGNORE_MODELS = set()  # --ignore-model: known fast load failures (e.g. 30B OOM), already diagnosed


def ollama_ok():
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/version", timeout=10) as r:
            return r.status == 200
    except Exception:
        return False


def supervisor_alive():
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'night_supervisor.py' "
                          "-and $_.CommandLine -notmatch 'CimInstance' } | Measure-Object | Select-Object -ExpandProperty Count"],
                         capture_output=True, text=True, timeout=60).stdout.strip()
    return out not in ("", "0")


def error_streak(stage):
    path = RUNS / stage / "results.jsonl"
    try:
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()[-ERROR_STREAK:]]
    except Exception:
        return None
    if len(rows) < ERROR_STREAK:
        return None
    # Instant HTTP 400 = the tag has no thinking template for a think-config (e.g. bartowski K2 GGUF): costs 0 s,
    # not a night-burning failure, reported in the verdict as "config not applicable".
    bad = [r for r in rows if (r.get("outcome") == "runner_error" or (r.get("row") or {}).get("infra_error"))
           and "HTTP Error 400" not in str((r.get("row") or {}).get("infra_error") or "")
           and r.get("model_key") not in IGNORE_MODELS]
    if len(bad) == ERROR_STREAK:
        last = bad[-1]
        return (f"{ERROR_STREAK} episodes in a row failed for infrastructure reasons in {stage}; last: "
                f"{last.get('model_key')}/{last.get('config_id')}/{last.get('task_id')}: "
                f"{(last.get('row') or {}).get('infra_error') or (last.get('row') or {}).get('error')}")
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--night", required=True, help="first stage name (where supervisor.log lives)")
    ap.add_argument("--stage", action="append", required=True)
    ap.add_argument("--ignore-model", action="append", default=[])
    args = ap.parse_args()
    IGNORE_MODELS.update(args.ignore_model)
    log = RUNS / args.night / "supervisor.log"
    down_since, seen_streak = None, set()
    print(f"alarm watch started for {args.stage}", flush=True)
    while True:
        text = log.read_text(encoding="utf-8") if log.exists() else ""
        if "supervisor done" in text:
            print("DONE: supervisor finished normally", flush=True)
            return 0
        reasons = []
        if ollama_ok():
            down_since = None
        else:
            down_since = down_since or time.time()
            if time.time() - down_since > OLLAMA_DOWN_MIN * 60:
                reasons.append(f"Ollama has not answered for {(time.time() - down_since) / 60:.0f} min")
        for marker in ("giving up this stage", "Ollama still down", "could not start Ollama"):
            if marker in text:
                reasons.append("supervisor log: " + next(l for l in text.splitlines()[::-1] if marker in l))
        if not supervisor_alive():
            reasons.append("night_supervisor.py is not running and never logged 'supervisor done'")
        for stage in args.stage:
            streak = error_streak(stage)
            if streak and streak not in seen_streak:
                reasons.append(streak)
        free_gb = shutil.disk_usage(RUNS).free / 2 ** 30
        if free_gb < MIN_DISK_GB:
            reasons.append(f"free disk space {free_gb:.1f} GB")
        if reasons:
            print("ALARM: " + " | ".join(reasons), flush=True)
            return 1
        time.sleep(CHECK_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
