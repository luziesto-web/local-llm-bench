"""Watchdog for unattended nights: runs zoo_night.py stages one after another and keeps them alive.

Every 30 s it checks the runner. It restarts the stage (same --name, so finished episodes are kept and an
interrupted one is saved as *-interrupted-*) when the runner process died, or when it is alive but nothing
was written for STALL_MINUTES (hung). Before every (re)start it makes sure Ollama answers, starting the
Ollama app if needed. It never deletes, edits or creates models, and never touches other runs.

  python night_supervisor.py --stage zoo-night-20261027:light,medium:09:50 --stage zoo-high-20261027:high:11:50

A stage is NAME:LEVELS:HH:MM (deadline, local time today). Stages run in order; a stage ends when its
status.json says finished, at its deadline (a running episode gets GRACE_MINUTES, then is stopped and will be
resumed next time), or after MAX_RESTARTS crashes. The supervisor owns the deadline: every (re)start passes
--new-deadline, so resuming on another day just works. Log: runs/<first stage>/supervisor.log
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs"
PY = sys.executable
OLLAMA_APP = Path.home() / "AppData/Local/Programs/Ollama/ollama app.exe"
CHECK_SECONDS = 30
STALL_MINUTES = 75          # longest legit silence: a High episode may think ~60 min in one turn
MAX_RESTARTS = 12
GRACE_MINUTES = 10          # after the stage deadline, let the current episode finish this long, then stop it
LOG = None


def log(msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def ollama_ok():
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/version", timeout=10) as r:
            return r.status == 200
    except Exception:
        return False


def ensure_ollama():
    if ollama_ok():
        return True
    log("Ollama does not answer; starting the Ollama app")
    try:
        subprocess.Popen([str(OLLAMA_APP)], creationflags=getattr(subprocess, "DETACHED_PROCESS", 0))
    except Exception as exc:
        log(f"could not start Ollama: {exc}")
    for _ in range(24):
        time.sleep(5)
        if ollama_ok():
            log("Ollama is back")
            return True
    log("Ollama still down after 2 min")
    return False


def finished(name):
    try:
        return json.loads((RUNS / name / "status.json").read_text(encoding="utf-8")).get("finished") is True
    except Exception:
        return False


def last_activity(name):
    """Newest write anywhere in this stage's folders (status, results, transcripts)."""
    newest = 0.0
    for folder in RUNS.glob(name + "*"):
        for p in [folder / "status.json", folder / "results.jsonl"] + list(folder.glob("*/transcript.jsonl")):
            try:
                newest = max(newest, p.stat().st_mtime)
            except OSError:
                pass
    return newest


def stop(proc):
    proc.terminate()
    try:
        proc.wait(30)
    except subprocess.TimeoutExpired:
        proc.kill()


def run_stage(name, levels, deadline):
    restarts = 0
    while not finished(name):
        if datetime.now() >= deadline:
            log(f"{name}: stage deadline {deadline:%H:%M} reached")
            return "deadline"
        if restarts > MAX_RESTARTS:
            log(f"{name}: too many restarts ({restarts}), giving up this stage")
            return "gave_up"
        if not ensure_ollama():
            restarts += 1
            time.sleep(60)
            continue
        hours = max(0.1, (deadline - datetime.now()).total_seconds() / 3600)
        cmd = [PY, "-B", "-u", str(HERE / "zoo_night.py"), "run", "--name", name, "--levels", *levels,
               "--hours", f"{hours:.2f}", "--new-deadline"]
        out = (RUNS / name).with_name(name + "-runner.log")
        log(f"{name}: start runner (restart #{restarts}) {' '.join(cmd[3:])}")
        with out.open("a", encoding="utf-8") as fh:
            proc = subprocess.Popen(cmd, cwd=HERE, stdout=fh, stderr=subprocess.STDOUT)
        started = time.time()
        while proc.poll() is None:
            time.sleep(CHECK_SECONDS)
            if (datetime.now() - deadline).total_seconds() > GRACE_MINUTES * 60:
                log(f"{name}: past the deadline + {GRACE_MINUTES} min, stopping the runner (resumable later)")
                stop(proc)
                return "deadline"
            silent = time.time() - max(last_activity(name), started)
            if silent > STALL_MINUTES * 60:
                log(f"{name}: nothing written for {silent / 60:.0f} min, runner looks hung; restarting")
                stop(proc)
                break
        code = proc.poll()
        if finished(name):
            log(f"{name}: finished (runner exit {code})")
            return "finished"
        restarts += 1
        log(f"{name}: runner ended without finishing (exit {code}); Ollama {'ok' if ollama_ok() else 'DOWN'}")
        time.sleep(20)
    return "finished"


def main():
    global LOG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", action="append", required=True, help="NAME:LEVELS:HH:MM")
    args = ap.parse_args()
    stages = []
    for spec in args.stage:
        name, levels, hh, mm = spec.split(":")
        deadline = datetime.now().replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)
        stages.append((name, levels.split(","), deadline))
    # Keep Windows from idle-sleeping while the supervisor runs (no settings change; ends with the process).
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    except Exception:
        pass
    (RUNS / stages[0][0]).mkdir(parents=True, exist_ok=True)
    LOG = RUNS / stages[0][0] / "supervisor.log"
    log(f"supervisor start: {[(n, l, f'{d:%H:%M}') for n, l, d in stages]}")
    results = {}
    for name, levels, deadline in stages:
        results[name] = run_stage(name, levels, deadline)
    log(f"supervisor done: {results}")
    print("NIGHT_SUPERVISOR_DONE " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
