#!/usr/bin/env python3
"""Read-only live dashboard for the round-2 model contests.

Run from the round2 folder:

    python dashboard.py            # http://127.0.0.1:8765/
    python dashboard.py --port N
    python dashboard.py --dump     # print /api/state once and exit

Only the Python standard library is used.  The dashboard never writes,
renames or deletes anything under runs/.  Its only outbound request is
GET http://127.0.0.1:11434/api/ps.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import contextlib
import time
import urllib.parse
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BASE = Path(__file__).resolve().parent
RUNS = BASE / "runs"
OLD_ROOT = BASE.parent                 # old config benchmark lives next to round2

RECENT_SECONDS = 24 * 60 * 60          # contests shown on top
RUNNING_WINDOW = 2 * 60 * 60           # transcript "active" window
STALL_SECONDS = 10 * 60                # hint only
ABANDONED_SECONDS = 60 * 60            # status "running" but logs silent this long
TAIL_BYTES = 64 * 1024                 # transcript tail limit
OLD_REFRESH_SECONDS = 60               # old benchmark files change rarely
OLLAMA_PS = "http://127.0.0.1:11434/api/ps"

# Runs written by zoo_night.py (results.jsonl + plan-*.json), including its wrappers medium_confirm/gemma_hint_test.
ZOO_PREFIXES = ("zoo-night-", "zoo-high-", "medium-confirm-", "medium-k2speed-", "gemma-hint-", "cpp-")
SPECIAL_PREFIXES = ZOO_PREFIXES + ("ministral-trial-",)
NEW_MODEL_MARKERS = ("qwen3-coder", "nemotron", "devstral", "bonsai", "glm")
DISPLAY_NAME_ALIASES = {
    "q8-cache-zoo-then-high-mmap-20261027": "Кеш тест",
    "zoo-night-q8-cache-retest-20261027": "Кеш тест · Light/Medium",
    "zoo-high-mmap-q8-followup-20261027": "High mmap · Кеш тест",
    "zoo-night-native-20261028": "Турнир · встроенные инструменты",
    "zoo-night-light-matrix-20261028": "Light · матрица конфигов",
    "zoo-high-matrix-20261028": "High · матрица конфигов",
    "zoo-high-calibrate-20261028": "High · калибровка контекста",
    "zoo-high-calibrate-r2-20261028": "High · калибровка, проверка",
    "zoo-high-full-q8-cache-20261027": "Кеш тест · полный High",
}
SECTION_ORDER = {"light": 0, "medium": 1, "high": 2, "cpp": 3}
SECTION_TITLES = {"light": "Light", "medium": "Medium", "high": "High", "cpp": "C++"}

_OLD_CACHE = {"ts": 0.0, "tests": None}

TASK_RE = re.compile(r"^(L(?:[0-9]|1[0-5])|M[1-5]|H[0-9]|C[0-9]-s[0-9]+)$")   # L12-L15, H0, H6-H9: 2026-10-28   # C1-s11: C++ bench, one cell per seed
CATEGORY_ORDER = {"L": 0, "M": 1, "H": 2, "C": 3}

STATUS_CLASS = {
    "running": "running",
    "completed": "completed",
    "complete": "completed",
    "finished": "completed",
    "done": "completed",
    "stopped": "stopped",
    "cancelled": "stopped",
    "canceled": "stopped",
    "failed": "failed",
    "error": "failed",
}


# ---------------------------------------------------------------------------
# small safe helpers

def safe_mtime(path) -> float | None:
    try:
        return os.stat(path).st_mtime
    except (OSError, ValueError):
        return None


def read_json(path):
    """Return parsed JSON or None.  A partially written file is skipped."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError, UnicodeError, TypeError):
        return None


def fmt_dt(ts: float | None) -> str:
    """Exact local time for tooltips (Russian day/month order)."""
    if not ts:
        return "—"
    try:
        return datetime.fromtimestamp(ts).strftime("%d.%m.%Y %H:%M:%S")
    except (OSError, OverflowError, ValueError):
        return "—"


def fmt_ago(ts: float | None, now: float) -> str:
    """Human Russian age: «2 мин назад»."""
    if not ts:
        return "—"
    seconds = max(0, int(now - ts))
    if seconds < 5:
        return "только что"
    if seconds < 60:
        return f"{seconds} с назад"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} мин назад"
    hours = minutes // 60
    if hours < 24:
        return f"{hours} ч {minutes % 60:02d} мин назад"
    return f"{hours // 24} дн назад"


def fmt_duration(seconds: float | int | None) -> str:
    """Human Russian duration: «4 мин 32 с», «1 ч 05 мин»."""
    if seconds is None:
        return "—"
    try:
        seconds = max(0, int(seconds))
    except (TypeError, ValueError):
        return "—"
    if seconds < 60:
        return f"{seconds} с"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} мин {sec:02d} с" if sec else f"{minutes} мин"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} ч {minutes:02d} мин"


def plural_ru(number, one: str, few: str, many: str) -> str:
    try:
        number = abs(int(number))
    except (TypeError, ValueError):
        return many
    if number % 10 == 1 and number % 100 != 11:
        return one
    if 2 <= number % 10 <= 4 and not 12 <= number % 100 <= 14:
        return few
    return many


def actions_ru(number) -> str:
    if not isinstance(number, int) or isinstance(number, bool):
        return "действий: —"
    return f"{number} {plural_ru(number, 'действие', 'действия', 'действий')}"


def fmt_bytes(value) -> str:
    if not isinstance(value, (int, float)) or value <= 0:
        return "—"
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return "—"


def fmt_size_ru(value) -> str:
    if not isinstance(value, (int, float)) or value <= 0:
        return "—"
    size = float(value)
    for unit in ("Б", "КБ", "МБ", "ГБ", "ТБ"):
        if size < 1024 or unit == "ТБ":
            if unit == "Б":
                return f"{int(size)} Б"
            text = f"{size:.1f}".replace(".", ",")
            return f"{text} {unit}"
        size /= 1024
    return "—"


def state_ru(state: str, state_class: str) -> str:
    if state_class == "running":
        return "идёт"
    if state_class == "completed":
        return "завершён"
    if state_class in ("stopped", "failed"):
        return "остановлен"
    if state_class == "abandoned":
        return "заброшен"
    return "нет статуса"


ACTION_RU = {
    "list_files": "список файлов",
    "read_file": "чтение файла",
    "write_file": "запись файла",
    "run_tests": "запуск тестов",
    "run_command": "команда",
    "final": "итог",
}


def action_ru(action: str | None) -> str:
    if not action:
        return "—"
    if action in ACTION_RU:
        return ACTION_RU[action]
    if action.startswith("text:"):
        return "текстовый ответ"
    return action


def short_contest(name: str) -> str:
    """high-v7-20261027-k2q8 -> high-v7; fall back to the full name."""
    if name in DISPLAY_NAME_ALIASES:
        return DISPLAY_NAME_ALIASES[name]
    match = re.match(r"^(.*?)-\d{8}(?:-.*)?$", name)
    if match and match.group(1):
        return match.group(1)
    return name


def short_model(tag: str) -> str:
    """Compact model name for the dense headers (K2 Q8, Qwen 9B, gpt-oss...)."""
    if not tag or tag == "unknown":
        return tag or "unknown"
    match = re.search(r"(?i)^k2[^:]*:.*?q(\d+)", tag)
    if match:
        return f"K2 Q{match.group(1)}"
    lowered = tag.lower()
    if lowered.startswith("gpt-oss"):
        return "gpt-oss"
    match = re.match(r"(?i)^qwen[\d.]*:(\d+(?:\.\d+)?)b", tag)
    if match:
        return f"Qwen {match.group(1)}B"
    match = re.match(r"(?i)^granite[\d.]*:(\d+(?:\.\d+)?)b", tag)
    if match:
        return f"Granite {match.group(1)}B"
    match = re.match(r"(?i)^ministral[^:]*:(\d+(?:\.\d+)?)b", tag)
    if match:
        return f"Ministral {match.group(1)}B"
    match = re.match(r"(?i)^gemma[^:]*:([a-z0-9.]+)b", tag)
    if match:
        return f"Gemma {match.group(1).upper()}B"
    return tag


def config_short(config_id: str) -> str:
    """qwen3.5-0.8b/nt-general -> nt-general; otherwise the full id."""
    if not config_id:
        return ""
    return str(config_id).rsplit("/", 1)[-1]


def short_error(value, limit: int = 220) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        value = value.get("error") or value.get("message") or value.get("output") or value
    text = str(value).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[-1][:limit]


def cell_seconds(cell) -> float:
    if not isinstance(cell, dict):
        return 0.0
    for key in ("wall_seconds", "elapsed_seconds"):
        value = cell.get(key)
        if isinstance(value, (int, float)):
            try:
                return max(0.0, float(value))
            except (TypeError, ValueError):
                continue
    return 0.0


def level_badge(tasks, config_level=None) -> str:
    labels = []
    letters = {str(task)[:1].upper() for task in tasks}
    for letter, label in (("L", "Light"), ("M", "Medium"), ("H", "High")):
        if letter in letters:
            labels.append(label)
    if not labels and isinstance(config_level, str):
        lowered = config_level.lower()
        for letter, label in (("l", "Light"), ("m", "Medium"), ("h", "High")):
            if lowered.startswith(letter):
                labels.append(label)
                break
    return "+".join(labels) if labels else "—"


def task_sort_key(task: str):
    match = TASK_RE.match(task)
    if not match:
        return (9, 0, task)
    category = match.group(1)[0]
    number = re.match(r"\d+", match.group(1)[1:])      # "C1-s11" -> 1 (seed suffix sorts by the task string)
    return (CATEGORY_ORDER.get(category, 9), int(number.group(0)) if number else 0, task)


def bool_int(value) -> bool:
    return value is True or value == 1


# ---------------------------------------------------------------------------
# filesystem scan (read-only)

def top_dirs():
    """Immediate directories under runs/, with safe entry points."""
    result = []
    try:
        with os.scandir(RUNS) as it:
            for entry in it:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        result.append((entry.name, Path(entry.path)))
                except OSError:
                    continue
    except OSError:
        pass
    result.sort(key=lambda item: item[0])
    return result


def has_contest_files(path: Path) -> bool:
    if os.path.isfile(path / "status.json") or os.path.isfile(path / "configuration.json"):
        return True
    try:
        return any(path.glob("plan*.json"))
    except OSError:
        return False


def identity_fields(configuration, status, path: Path):
    """(night, trial) from configuration/status/plan*.json, first non-empty wins."""
    fields = {"night": None, "trial": None}
    sources = [configuration, status]
    try:
        for plan_path in sorted(path.glob("plan*.json")):
            sources.append(read_json(plan_path))
    except OSError:
        pass
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in ("night", "trial"):
            if fields[key] is not None:
                continue
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                fields[key] = value.strip()
    return fields["night"], fields["trial"]


def contest_group_name(name: str, configuration, status, path: Path):
    """Group several folders into one contest.

    Explicit night/trial wins; otherwise strip a trailing -<label> when the
    label is a key of configuration.json -> models.
    """
    night, trial = identity_fields(configuration, status, path)
    if night:
        return night, "night"
    if trial:
        return trial, "trial"
    models = configuration.get("models") if isinstance(configuration, dict) else None
    if isinstance(models, dict):
        for label in sorted(models, key=lambda item: len(str(item)), reverse=True):
            label = str(label)
            suffix = "-" + label
            if label and name.endswith(suffix) and len(name) > len(suffix):
                return name[: -len(suffix)], "label"
    return name, "folder"


def find_tasks(root: Path) -> dict[str, Path]:
    """Task subfolders (H1, M2, ...) directly inside root."""
    tasks: dict[str, Path] = {}
    try:
        with os.scandir(root) as it:
            for entry in it:
                try:
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                    if not TASK_RE.match(entry.name):
                        continue
                    tasks[entry.name] = Path(entry.path)
                except OSError:
                    continue
    except OSError:
        pass
    return dict(sorted(tasks.items(), key=lambda item: task_sort_key(item[0])))


def best_parent(name: str, candidate_names):
    matches = [candidate for candidate in candidate_names if name.startswith(candidate + "-")]
    if not matches:
        return None
    return max(matches, key=len)


def tree_max_mtime(root: Path, seed: float = 0.0) -> float:
    best = max(seed, safe_mtime(root) or 0.0)
    try:
        for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
            try:
                best = max(best, os.stat(dirpath).st_mtime)
            except OSError:
                pass
            for filename in filenames:
                mtime = safe_mtime(os.path.join(dirpath, filename))
                if mtime:
                    best = max(best, mtime)
    except OSError:
        pass
    return best


# ---------------------------------------------------------------------------
# transcript tail (never read a large transcript whole)

def tail_lines(path: Path, limit: int = TAIL_BYTES) -> list[str]:
    try:
        size = os.path.getsize(path)
        if size <= 0:
            return []
        with open(path, "rb") as fh:
            if size > limit:
                fh.seek(size - limit - 1)
                chunk = fh.read()
                newline = chunk.find(b"\n")
                if newline < 0:
                    return []
                chunk = chunk[newline + 1:]
            else:
                chunk = fh.read()
        text = chunk.decode("utf-8", "replace")
        return [line for line in text.splitlines() if line.strip()]
    except (OSError, ValueError):
        return []


def action_from_content(content):
    if isinstance(content, dict):
        action = content.get("action")
        if isinstance(action, str):
            return action
        return None
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except ValueError:
            return "text: " + content.strip().replace("\n", " ")[:100]
        if isinstance(parsed, dict) and isinstance(parsed.get("action"), str):
            return parsed["action"]
        return "text: " + content.strip().replace("\n", " ")[:100]
    return None


def transcript_summary(path: Path):
    """Inspect only the tail of a transcript.  Missing/broken lines are ignored."""
    summary = {
        "model": None,
        "turn": None,
        "last_action": None,
        "last_tool_error": None,
        "last_event": None,
    }
    for line in tail_lines(path):
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        event_name = event.get("event")
        if isinstance(event_name, str):
            summary["last_event"] = event_name
        turn = event.get("turn")
        if isinstance(turn, int) and not isinstance(turn, bool):
            summary["turn"] = turn
        payload = event.get("payload")
        if not isinstance(payload, dict):
            payload = {}
        if event_name == "request":
            model = payload.get("model")
            if isinstance(model, str):
                summary["model"] = model
        elif event_name == "response":
            message = payload.get("message")
            if not isinstance(message, dict):
                message = {}
            action = action_from_content(message.get("content"))
            if action:
                summary["last_action"] = action
            model = payload.get("model")
            if isinstance(model, str):
                summary["model"] = model
        elif event_name == "tool_result":
            result = event.get("result")
            if isinstance(result, dict):
                error = result.get("error")
                if error:
                    summary["last_tool_error"] = str(error)
    if not summary["model"]:
        # Transcripts without payload.model (e.g. the C++ bench): take the tag from <run>/../<name>/configuration.json.
        run_dir = path.parent.parent
        if run_dir.name.endswith("-run"):
            configuration = read_json(run_dir.with_name(run_dir.name[: -len("-run")]) / "configuration.json")
            entry = ((configuration or {}).get("models") or {}).get("run") if isinstance(configuration, dict) else None
            if isinstance(entry, dict) and isinstance(entry.get("tag"), str):
                summary["model"] = entry["tag"]
    return summary


# ---------------------------------------------------------------------------
# episode / contest classification

def cell_from_episode(episode: dict) -> dict:
    """Cell dict for one finished episode.json payload."""
    infra = episode.get("infra_error")
    code_pass = bool_int(episode.get("code_pass"))
    workflow_pass = bool_int(episode.get("automated_workflow_pass"))
    wall = fmt_duration(episode.get("wall_seconds"))
    calls = actions_ru(episode.get("tool_calls"))
    violations = episode.get("violations")
    truncation = bool_int(episode.get("truncation"))
    model = episode.get("model")
    acceptance = episode.get("acceptance")

    if infra:
        kind, label, icon = "infra", "I", "⚠️"
        reason = "сбой: " + (short_error(infra) or "инфраструктура")
        status_text = "Сбой инфраструктуры"
        title = f"{status_text}: {reason} · {wall} · {calls}"
    elif code_pass and workflow_pass:
        kind, label, icon = "cw", "CW", "✅"
        reason = "Решено, тесты пройдены"
        status_text = "Решено, тесты пройдены"
        title = f"{status_text} · {wall} · {calls}"
    elif code_pass:
        kind, label, icon = "code", "C", "🟡"
        reason = "Код верный, но модель не прогнала свои тесты"
        status_text = "Код верный, тесты не прогнаны"
        title = f"{reason} · {wall} · {calls}"
    else:
        kind, label, icon = "fail", "-", "❌"
        status_text = "Не решено"
        reason = ""
        if isinstance(acceptance, dict) and acceptance.get("success") is False:
            reason = short_error(acceptance.get("error") or acceptance.get("output"))
        if not reason and isinstance(violations, list) and violations:
            reason = "нарушения: " + ", ".join(str(v) for v in violations[:3])
        if not reason:
            reason = "код не прошёл проверку"
        title = f"{status_text} · {wall} · {calls}"

    if truncation:
        reason += " · контекст обрезан"
        title += "\nОбрезано по контексту."
    if isinstance(violations, list) and violations:
        title += "\nНарушения: " + ", ".join(str(v) for v in violations[:5])
    if kind == "fail":
        title += "\nПричина: " + reason
    return {
        "kind": kind,
        "label": label,
        "icon": icon,
        "title": title,
        "stalled": False,
        "wall_seconds": episode.get("wall_seconds"),
        "wall_display": wall,
        "tool_calls": episode.get("tool_calls"),
        "model": model,
        "reason": reason,
        "status_text": status_text,
        "review_required": bool_int(episode.get("review_required")),
    }


_cell_from_episode_base = cell_from_episode


def cell_from_episode(episode: dict) -> dict:  # noqa: F811 - adds the transcript path for the detail view
    cell = _cell_from_episode_base(episode)
    if isinstance(episode, dict) and episode.get("transcript"):
        cell["transcript"] = str(episode.get("transcript"))
    return cell


DETAIL_LIMITS = dict(task=20000, thinking=12000, content=20000, tool=3000, turns=80)


def _clip(text, limit):
    text = "" if text is None else str(text)
    return text if len(text) <= limit else text[:limit] + f"\n… [обрезано, всего {len(text)} символов]"


def episode_detail(transcript_arg: str) -> dict:
    """Task text and the model's turns for one episode. Read-only, only inside runs/."""
    path = Path(transcript_arg).resolve()
    path.relative_to(RUNS.resolve())
    if path.name != "transcript.jsonl" or not path.is_file():
        raise ValueError("not a transcript inside runs/")
    task_dir = path.parent
    task_text = ""
    with contextlib.suppress(OSError):
        task_text = (task_dir / "workspace" / "TASK.md").read_text(encoding="utf-8", errors="replace")
    turns, current, last_request_at = [], None, None
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith('{"event": "request"'):
                last_request_at = len(turns)   # requests repeat the whole history; only note that one was sent
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("event") == "response":
                payload = event.get("payload") or {}
                message = payload.get("message") or {}
                content = message.get("content") or ""
                action = None
                with contextlib.suppress(Exception):
                    action = json.loads(content)
                is_action = isinstance(action, dict)
                current = {"turn": event.get("turn"),
                           "thinking": _clip(message.get("thinking"), DETAIL_LIMITS["thinking"]),
                           "action": action.get("action") if is_action else None,
                           "path": action.get("path") if is_action else None,
                           "code": _clip(action.get("content"), DETAIL_LIMITS["content"]) if is_action and action.get("content") else None,
                           "report": _clip(action.get("report"), DETAIL_LIMITS["content"]) if is_action and action.get("report") else None,
                           "raw": None if is_action else _clip(content, DETAIL_LIMITS["content"]),
                           "tokens": payload.get("eval_count"), "result": None, "result_ok": True}
                turns.append(current)
            elif event.get("event") == "tool_result" and current is not None:
                result = event.get("result") or {}
                text = result.get("error") or result.get("output") or json.dumps(result, ensure_ascii=False)
                current["result"] = _clip(text, DETAIL_LIMITS["tool"])
                current["result_ok"] = not result.get("error") and result.get("success", True) is not False
    episode = None
    with contextlib.suppress(Exception):
        episode = json.loads((task_dir / "episode.json").read_text(encoding="utf-8"))
    finished = isinstance(episode, dict)
    waiting = None
    if not finished and last_request_at == len(turns):
        waiting = {"turn": len(turns), "seconds": int(time.time() - path.stat().st_mtime)}
    return {"task": _clip(task_text, DETAIL_LIMITS["task"]), "task_id": task_dir.name,
            "turns": turns[-DETAIL_LIMITS["turns"]:], "turns_total": len(turns),
            "finished": finished, "waiting": waiting,
            "code_pass": (episode or {}).get("code_pass"),
            "workflow_pass": (episode or {}).get("automated_workflow_pass")}


def pid_alive(pid) -> bool:
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        kernel = ctypes.windll.kernel32
        handle = kernel.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        try:
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259   # STILL_ACTIVE
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def episode_owner_alive(task_dir: Path):
    """Is an unfinished zoo episode still being played? None = cannot tell (not a zoo run: keep the mtime rule).

    zoo_night writes status.json with its pid and the folder of the current episode; an episode whose process
    died (a restart, a stop) or that is not the current one is interrupted, not running - otherwise it hangs
    in "Идёт сейчас" as a ghost until its transcript is two hours old.
    """
    run_name = task_dir.parent.name
    owners = [name for name, _ in top_dirs() if run_name.startswith(name + "-") and (RUNS / name / "status.json").is_file()]
    if not owners:
        return None
    status = read_json(RUNS / max(owners, key=len) / "status.json")
    if not isinstance(status, dict):
        return None
    current = status.get("current") if isinstance(status.get("current"), dict) else {}
    if status.get("finished") is True or STATUS_CLASS.get(str(status.get("state", "")).lower()) in ("stopped", "failed"):
        return False
    if "pid" in status and not pid_alive(status.get("pid")):
        return False
    folder = current.get("folder")
    if isinstance(folder, str) and folder:
        return Path(folder).resolve() == task_dir.resolve()
    # older status without pid/folder: only the episode it names can be running
    def slug(value):
        return re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")
    if current.get("task") != task_dir.name or f"-{slug(current.get('model'))}-{slug(current.get('config'))}-" not in run_name + "-":
        return False
    return None


def classify_episode(task_dir: Path, now: float):
    """Return a cell dict plus metadata for one task folder."""
    episode_path = task_dir / "episode.json"
    transcript_path = task_dir / "transcript.jsonl"
    manifest_path = task_dir / "manifest.json"
    transcript_mtime = safe_mtime(transcript_path)
    manifest_mtime = safe_mtime(manifest_path)

    if os.path.isfile(episode_path):
        episode = read_json(episode_path)
        if not isinstance(episode, dict):
            return (
                {
                    "kind": "unknown",
                    "label": "?",
                    "icon": "?",
                    "title": "episode.json есть, но прочитать его не удалось (возможно, файл ещё пишется)",
                    "stalled": False,
                    "reason": "ошибка чтения episode.json",
                    "status_text": "неизвестно",
                },
                None,
            )
        return cell_from_episode(episode), episode

    # No episode.json: active if the transcript (or, before the first write,
    # the manifest) changed recently - and, for zoo runs, only while their process plays this very episode.
    activity = transcript_mtime or manifest_mtime
    if activity and now - activity <= RUNNING_WINDOW and episode_owner_alive(task_dir) is False:
        return {
            "kind": "unknown",
            "label": "⏸",
            "icon": "⏸",
            "title": "Прервано: прогон остановлен посреди эпизода; эпизод переиграется, когда прогон продолжат",
            "stalled": False,
            "reason": "прервано (процесс прогона остановлен)",
            "status_text": "прервано",
        }, None
    if activity and now - activity <= RUNNING_WINDOW:
        transcript_created = 0.0
        with contextlib.suppress(OSError):
            st = transcript_path.stat()   # creation time: st_birthtime (Python 3.12+), not st_ctime
            transcript_created = getattr(st, "st_birthtime", 0.0) or 0.0
        # No manifest (e.g. C++ bench): count from the transcript's creation, not its last write.
        start = manifest_mtime or (transcript_created if 0 < transcript_created <= activity else activity)
        elapsed_seconds = max(0, now - start)
        elapsed = fmt_duration(elapsed_seconds)
        last_write = transcript_mtime or activity
        ago = fmt_ago(last_write, now)
        stalled = now - last_write > STALL_SECONDS
        summary = transcript_summary(transcript_path) if transcript_mtime else {
            "model": None,
            "turn": None,
            "last_action": None,
            "last_tool_error": None,
            "last_event": None,
        }
        turn = summary["turn"] if summary["turn"] is not None else "—"
        last_action = summary["last_action"]
        action_text = action_ru(last_action)
        if stalled:
            title = (
                f"Похоже на зависание: журнал молчит {ago} · идёт {elapsed}"
                f"\nход {turn} · последнее действие: {action_text}"
            )
        else:
            title = (
                f"Идёт сейчас · прошло {elapsed} · последняя запись {ago}"
                f"\nход {turn} · последнее действие: {action_text}"
            )
        return (
            {
                "kind": "run",
                "label": "RUN",
                "transcript": str(transcript_path),
                "icon": "🛑" if stalled else "⏳",
                "title": title,
                "elapsed": elapsed,
                "elapsed_seconds": int(elapsed_seconds),
                "last_write_ago": ago,
                "last_write_ago_ru": ago,
                "stalled": stalled,
                "turn": turn,
                "last_action": last_action or "—",
                "last_action_ru": action_text,
                "reason": "последнее действие: " + action_text,
                "status_text": "идёт",
                "model": summary["model"],
            },
            None,
        )

    if transcript_mtime or manifest_mtime:
        title = "Нет episode.json; журнал не менялся больше 2 ч (прервано или заброшено?)"
    else:
        title = "Пустая папка эпизода: нет manifest.json, transcript.jsonl и episode.json"
    return {
        "kind": "unknown",
        "label": "?",
        "icon": "?",
        "title": title,
        "stalled": False,
        "reason": title,
        "status_text": "неизвестно",
    }, None


def contest_status(path: Path):
    status = read_json(path / "status.json")
    if not isinstance(status, dict):
        return {
            "state": "no status",
            "state_ru": "нет статуса",
            "state_class": "unknown",
            "done": None,
            "total": None,
            "detail": "",
            "detail_ru": "",
            "title": "status.json отсутствует или не читается",
            "review_required": False,
            "finished": False,
        }

    raw_state = status.get("state")
    if isinstance(raw_state, str) and raw_state.strip():
        state = raw_state.strip()
    elif status.get("finished") is True:
        state = "finished"
    elif isinstance(status.get("current"), dict):
        state = "running"
    else:
        state = "unknown"
    state_class = STATUS_CLASS.get(state.lower(), "unknown")

    done = status.get("done")
    if not isinstance(done, int) or isinstance(done, bool):
        done = status.get("completed")
    if not isinstance(done, int) or isinstance(done, bool):
        done = None
    total = status.get("total")
    if not isinstance(total, int) or isinstance(total, bool):
        total = None

    detail_parts = []
    detail_parts_ru = []
    last_case = status.get("last_case")
    last_model = status.get("last_model")
    if last_case or last_model:
        detail_parts.append(
            "last " + " / ".join(str(v) for v in (last_model, last_case) if v)
        )
        detail_parts_ru.append(
            "последний " + " / ".join(str(v) for v in (last_model, last_case) if v)
        )
    current = status.get("current")
    if isinstance(current, dict):
        current_parts = [
            str(current.get(key))
            for key in ("model", "config", "task")
            if current.get(key)
        ]
        if current_parts:
            detail_parts.append("current " + " / ".join(current_parts))
            detail_parts_ru.append("сейчас " + " / ".join(current_parts))
    if isinstance(status.get("index"), int):
        detail_parts.append(f"index {status['index']}/{status.get('total', '?')}")
        detail_parts_ru.append(f"индекс {status['index']}/{status.get('total', '?')}")
    if status.get("finished") is True:
        detail_parts.append("finished")
        detail_parts_ru.append("завершён")
    if status.get("error"):
        detail_parts.append("error: " + str(status["error"])[:100])
        detail_parts_ru.append("ошибка: " + str(status["error"])[:100])

    try:
        title = json.dumps(status, ensure_ascii=False)
    except (TypeError, ValueError):
        title = str(status)
    if len(title) > 800:
        title = title[:800] + "…"

    return {
        "state": state,
        "state_ru": state_ru(state, state_class),
        "state_class": state_class,
        "done": done,
        "total": total,
        "detail": " · ".join(detail_parts),
        "detail_ru": " · ".join(detail_parts_ru),
        "title": title,
        "review_required": status.get("review_required") is True,
        "finished": status.get("finished") is True,
    }


def done_text(done, total) -> str:
    if done is not None and total is not None:
        return f"{done}/{total}"
    if done is not None:
        return f"{done} done"
    return "—"


def done_text_ru(done, total) -> str:
    if done is not None and total is not None:
        return f"{done} / {total}"
    if done is not None:
        return f"{done} эп."
    return "—"


# ---------------------------------------------------------------------------
# state assembly

def ollama_state():
    try:
        request = urllib.request.Request(OLLAMA_PS, headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=1.5) as response:
            raw = response.read()
        data = json.loads(raw.decode("utf-8", "replace"))
    except Exception as exc:  # noqa: BLE001 - a panel must never crash the page
        error = f"{type(exc).__name__}: {exc}"
        return {
            "ok": False,
            "error": error,
            "models": [],
            "summary": f"Ollama: недоступен — {error}",
        }

    models = []
    if isinstance(data, dict):
        for item in data.get("models") or []:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("model") or "?"
            size = item.get("size")
            size_vram = item.get("size_vram")
            split = None
            cpu_pct = None
            gpu_pct = None
            if isinstance(size, int) and isinstance(size_vram, int) and size > 0:
                gpu_pct = max(0, min(100, round(100 * size_vram / size)))
                cpu_pct = 100 - gpu_pct
                if gpu_pct == 100:
                    split = "GPU 100%"
                elif gpu_pct == 0:
                    split = "CPU 100%"
                else:
                    split = f"GPU {gpu_pct}% / CPU {cpu_pct}%"
            details = item.get("details")
            if not isinstance(details, dict):
                details = {}
            context = (
                item.get("context_length")
                or item.get("num_ctx")
                or details.get("context_length")
            )
            models.append(
                {
                    "name": str(name),
                    "size": fmt_bytes(size),
                    "size_ru": fmt_size_ru(size),
                    "split": split or "—",
                    "cpu_pct": cpu_pct,
                    "gpu_pct": gpu_pct,
                    "context": str(context) if context else "—",
                    "expires": str(item.get("expires_at") or "—"),
                }
            )
    models.sort(key=lambda model: model["name"].lower())

    if not models:
        summary = "Ollama: модели не загружены"
    else:
        parts = []
        for model in models:
            bits = [model["name"], model["size_ru"]]
            if model["cpu_pct"] is not None and model["gpu_pct"] is not None:
                bits.append(f"{model['cpu_pct']}% CPU / {model['gpu_pct']}% GPU")
            if model["context"] != "—":
                bits.append(f"ctx {model['context']}")
            parts.append(" · ".join(bits))
        summary = "Ollama: " + " | ".join(parts)

    return {"ok": True, "error": None, "models": models, "summary": summary}


SERVICE_MARKERS = (
    "smoketest",
    "selfcheck",
    "preflight",
    "adapter-validation",
    "extended-validation",
)


def model_tag_for(label: str, models_map, episode_models):
    entry = None
    if isinstance(models_map, dict):
        entry = models_map.get(label)
        if not isinstance(entry, dict) and len(models_map) == 1:
            entry = next(iter(models_map.values()))
    if isinstance(entry, dict):
        tag = entry.get("tag")
        if isinstance(tag, str) and tag.strip():
            return tag.strip()
    for model in episode_models:
        if isinstance(model, str) and model.strip():
            return model.strip()
    return label


def count_cells(rows):
    counts = {
        "cw": 0,
        "code": 0,
        "fail": 0,
        "infra": 0,
        "running": 0,
        "stalled": 0,
        "pending": 0,
        "skipped": 0,
        "unknown": 0,
    }
    for row in rows:
        for cell in row.get("cells", {}).values():
            kind = cell.get("kind")
            if kind == "run":
                counts["stalled" if cell.get("stalled") else "running"] += 1
            elif kind in counts:
                counts[kind] += 1
            else:
                counts["unknown"] += 1
    return counts


def result_text(counts):
    parts = [
        f"✅ {counts.get('cw', 0)}",
        f"🟡 {counts.get('code', 0)}",
        f"❌ {counts.get('fail', 0)}",
    ]
    if counts.get("infra"):
        parts.append(f"⚠️ {counts['infra']}")
    if counts.get("running"):
        parts.append(f"⏳ {counts['running']}")
    if counts.get("stalled"):
        parts.append(f"🛑 {counts['stalled']}")
    if counts.get("skipped"):
        parts.append(f"⏭ {counts['skipped']}")
    if counts.get("unknown"):
        parts.append(f"? {counts['unknown']}")
    return " / ".join(parts)


def service_reason(name: str, has_status: bool, has_episodes: bool):
    lowered = name.lower()
    markers = [marker for marker in SERVICE_MARKERS if marker in lowered]
    if markers:
        return "служебное имя: " + ", ".join(markers)
    if not has_status and not has_episodes:
        return "нет status.json и эпизодов"
    return ""


CONTEST_PREFIXES = ("high-", "light-contest-", "medium-contest-", "light10-")


def group_category(name: str, sources, rows, night, trial) -> str:
    """Contest or separate model test (see DASHBOARD_FIXES_2.md §4)."""
    if trial:
        return "test"
    if night or name.startswith(ZOO_PREFIXES) or name.startswith(CONTEST_PREFIXES):
        return "contest"
    tags = set()
    for row in rows:
        tag = row.get("model")
        if isinstance(tag, str) and tag.strip():
            tags.add(tag.strip())
    for source in sources:
        tags.update(source.get("model_tags") or [])
    return "contest" if len(tags) >= 2 else "test"


def group_row(contest, row):
    cells = row.get("cells") if isinstance(row.get("cells"), dict) else {}
    tasks = sorted(cells, key=task_sort_key)
    counts = count_cells([{"cells": cells}])
    wall_seconds = sum(cell_seconds(cell) for cell in cells.values())
    model = row.get("model") or contest.get("name") or "unknown"
    config = str(row.get("config") or contest.get("config_id") or "")
    row_id = "|".join(
        [
            str(contest.get("name") or ""),
            str(row.get("label") or ""),
            str(model),
            config,
        ]
    )
    return {
        "id": row_id,
        "source": contest.get("name") or "",
        "label": row.get("label") or "",
        "model": model,
        "short_model": short_model(model),
        "config": config,
        "config_short": config_short(config),
        "show_config": False,
        "cells": cells,
        "tasks": tasks,
        "counts": counts,
        "task_count": len(cells),
        "wall_seconds": int(round(wall_seconds)),
        "wall_display": fmt_duration(wall_seconds) if wall_seconds else "—",
        "incomplete": False,
        "winner": False,
        "title": f"Модель {model}" + (f"\nКонфиг: {config}" if config else "") + (
            f"\nЭпизод: {contest.get('name')} / {row.get('label')}"
        ),
    }


def pick_winners(rows, max_tasks, code_first=False):
    """Return (winner rows, best (primary, secondary) pair) per DASHBOARD_FIXES_2.md §3 + CODE_FIRST_LEVELS."""
    if max_tasks <= 0:
        return [], (0, 0)
    eligible = [row for row in rows if not row["incomplete"] and row["task_count"] == max_tasks]
    if not eligible:
        return [], (0, 0)
    best = max(rank_pair(row, code_first) for row in eligible)
    return [row for row in eligible if rank_pair(row, code_first) == best], best


def build_groups(contests, now: float):
    by_name: dict[str, list[dict]] = {}
    for contest in contests:
        group_name = contest.get("group") or contest.get("name") or "unknown"
        if group_name.startswith(SPECIAL_PREFIXES):
            continue  # built from results.jsonl / episodes.jsonl below
        by_name.setdefault(group_name, []).append(contest)

    groups = []
    for name, sources in by_name.items():
        rows = []
        for source in sorted(sources, key=lambda item: item.get("name", "")):
            for row in source.get("rows") or []:
                rows.append(group_row(source, row))

        # Draw the whole grid up front: every row gets every task of the group, and while the
        # contest runs, every task of its level.  Unplayed ones are empty squares that fill in later.
        planned = {task for row in rows for task in row["cells"]}
        running = any(
            source.get("live")
            or (source.get("state_class") == "running" and now - (source.get("updated_ts") or 0.0) <= ABANDONED_SECONDS)
            for source in sources
        )
        if running:
            for source in sources:
                planned.update(LEVEL_TASKS.get(str(source.get("config_level") or "").lower(), ()))
        for row in rows:
            row["cells"] = pad_planned(row["cells"], planned, live=running)
            row["tasks"] = sorted(row["cells"], key=task_sort_key)

        counts = count_cells(rows)
        episodes = sum(row["task_count"] for row in rows)
        task_union = sorted(
            {task for row in rows for task in row.get("tasks") or []}, key=task_sort_key
        )
        max_tasks = max((row["task_count"] for row in rows), default=0)
        wall_seconds = sum(row["wall_seconds"] for row in rows)
        tags = set()
        for row in rows:
            tags.add(row["model"])
        for source in sources:
            tags.update(source.get("model_tags") or [])
        tags.discard("")
        night = next((source.get("night") for source in sources if source.get("night")), None)
        trial = next((source.get("trial") for source in sources if source.get("trial")), None)
        category = group_category(name, sources, rows, night, trial)

        status_sources = [source for source in sources if source.get("has_status")]
        status_classes = [source.get("state_class", "unknown") for source in status_sources]
        live = any(source.get("live") for source in sources) or (
            counts.get("running", 0) > 0 or counts.get("stalled", 0) > 0
        )
        if live:
            state_class = "running"
        elif status_classes and all(value == "completed" for value in status_classes):
            state_class = "completed"
        elif any(value in ("stopped", "failed") for value in status_classes):
            state_class = "stopped"
        elif any(value == "running" for value in status_classes):
            state_class = "running"
        else:
            state_class = "unknown"
        source_ts = max((source.get("updated_ts") or 0.0) for source in sources)
        if (
            state_class == "running"
            and not live
            and now - source_ts > ABANDONED_SECONDS
        ):
            state_class = "abandoned"
        complete = state_class == "completed" and bool(status_classes) and not live

        for row in rows:
            row["incomplete"] = row["task_count"] < max_tasks
        duplicate_models = len(rows) != len({row["model"] for row in rows})
        for row in rows:
            row["show_config"] = duplicate_models and bool(row["config_short"])

        winners = []
        best = (0, 0)
        code_first = any(task_level(t) in CODE_FIRST_LEVELS for row in rows for t in row["cells"])
        if category == "contest" and complete:
            winners, best = pick_winners(rows, max_tasks, code_first)
        winner_ids = [row["id"] for row in winners]
        winner_names = []
        for row in winners:
            label = row["short_model"]
            if row["show_config"] and row["config_short"]:
                label += " · " + row["config_short"]
            if label not in winner_names:
                winner_names.append(label)
        winner_text = ""
        if winners:
            winner_text = "🏆 " + ", ".join(winner_names) + f" ({winner_ratio(best, code_first, max_tasks)})"
        best_cw = best[1] if code_first else best[0]
        for row in rows:
            row["winner"] = row["id"] in winner_ids

        rows.sort(
            key=lambda row: (
                0 if row["counts"].get("running", 0) or row["counts"].get("stalled", 0) else 1,
                tuple(-v for v in rank_pair(row, code_first)),
                row["model"].lower(),
                row["label"].lower(),
            )
        )
        sections = [
            {
                "key": "all",
                "title": "",
                "rows": rows,
                "winner_text": winner_text,
                "winner_ids": winner_ids,
                "winner_names": winner_names,
                "ratio": winner_ratio(best, code_first, max_tasks) if winners else "",
            }
        ]

        live_rows = []
        for row in rows:
            for task in row["tasks"]:
                cell = row["cells"].get(task)
                if isinstance(cell, dict) and cell.get("kind") == "run":
                    live_rows.append((row, task, cell))
        live_info = None
        if live_rows:
            live_rows.sort(key=lambda item: (bool(item[2].get("stalled")), item[1]))
            row, task, cell = live_rows[0]
            live_info = {
                "model": row["model"],
                "short_model": row["short_model"],
                "task": task,
                "elapsed": cell.get("elapsed") or fmt_duration(cell.get("elapsed_seconds")),
                "stalled": bool(cell.get("stalled")),
            }
            more = len(live_rows) - 1
            if more:
                live_info["more"] = more

        updated_ts = max((source.get("updated_ts") or 0.0) for source in sources)
        config_level = next(
            (source.get("config_level") for source in sources if source.get("config_level")),
            None,
        )
        reason = service_reason(name, bool(status_sources), bool(rows))
        if not reason:
            for source in sources:
                source_reason = service_reason(
                    source.get("name", ""),
                    bool(source.get("has_status")),
                    bool(source.get("has_episodes")),
                )
                if source_reason and "служебное имя" in source_reason:
                    reason = source_reason
                    break

        status_completed = sum(1 for value in status_classes if value == "completed")
        queue_source = next((source for source in status_sources if source.get("status_phase")), None)
        queue_progress = None
        if queue_source:
            queue_progress = {
                "phase": queue_source.get("status_phase"),
                "done": queue_source.get("status_progress_done"),
                "total": queue_source.get("status_progress_total"),
            }
        review_required = any(source.get("review_required") for source in sources)
        title_lines = [
            f"Конкурс: {name}",
            f"Состояние: {state_ru(state_class, state_class)}",
            f"Статусов завершено: {status_completed}/{len(status_classes)}",
            f"Моделей: {len(tags)} · эпизодов: {episodes}",
        ]
        if winner_text:
            title_lines.append(
                f"Победитель: {', '.join(winner_names)} ({winner_ratio(best, code_first, max_tasks)}); предварительно"
            )
            title_lines.append("Ранжирование: больше ✅, затем ✅+🟡; время и действия не учитываются.")
        groups.append(
            {
                "name": name,
                "category": category,
                "level": level_badge(task_union, config_level),
                "state": state_class,
                "state_ru": state_ru(state_class, state_class),
                "state_class": state_class,
                "live": live,
                "live_info": live_info,
                "complete": complete,
                "stopped": state_class == "stopped",
                "counts": counts,
                "result_text": result_text(counts),
                "episodes": episodes,
                "episodes_text": f"{episodes} {plural_ru(episodes, 'эпизод', 'эпизода', 'эпизодов')}",
                "task_total": max_tasks,
                "models": len(tags),
                "wall_seconds": int(round(wall_seconds)),
                "wall_display": fmt_duration(wall_seconds) if wall_seconds else "—",
                "updated_ts": updated_ts,
                "updated": fmt_dt(updated_ts),
                "updated_ago": fmt_ago(updated_ts, now),
                "recent": now - updated_ts <= RECENT_SECONDS,
                "winner": {
                    "ids": winner_ids,
                    "names": winner_names,
                    "text": winner_text,
                    "ratio": winner_ratio(best, code_first, max_tasks) if winners else "",
                    "best_cw": best_cw,
                    "total": max_tasks,
                    "provisional": bool(winners),
                },
                "winner_text": winner_text,
                "winner_ids": winner_ids,
                "sections": sections,
                "review_required": review_required,
                "rows": rows,
                "sources": [source.get("name") for source in sources],
                "queue_progress": queue_progress,
                "service": bool(reason),
                "service_reason": reason,
                "status_total": len(status_classes),
                "status_completed": status_completed,
                "title": "\n".join(title_lines),
            }
        )

    groups.sort(
        key=lambda group: (
            0 if group["live"] else 1,
            -group["updated_ts"],
            group["name"].lower(),
        )
    )
    return groups


LEVEL_TASKS = {
    "light": ("L0", "L1", "L2", "L3", "L4", "L5"),
    "medium": ("M1", "M2", "M3", "M4", "M5"),
    "high": ("H1", "H2", "H3", "H4", "H5"),
}


def pending_cell() -> dict:
    """Empty square for a planned task that has not started yet."""
    return {
        "kind": "pending", "label": "WAIT", "icon": "·",
        "title": "Запланировано; эпизод ещё не начинался",
        "status_text": "ожидает", "stalled": False,
    }


def unplayed_cell() -> dict:
    """Square for a planned task of a run that has stopped without playing it."""
    return {
        "kind": "skipped", "label": "SKIP", "icon": "⏭",
        "title": "Запланировано, но прогон остановился раньше",
        "status_text": "не сыграно", "reason": "не сыграно: прогон остановлен", "stalled": False,
    }


def pad_planned(cells: dict, planned, live: bool = True) -> dict:
    """Copy of cells with an empty square for every planned task not played yet."""
    padded = dict(cells)
    for task in planned:
        if task not in padded:
            padded[task] = pending_cell() if live else unplayed_cell()
    return padded


def task_level(task) -> str:
    text = str(task or "").upper()
    if text.startswith("L"):
        return "light"
    if text.startswith("M"):
        return "medium"
    if text.startswith("H"):
        return "high"
    if text.startswith("C"):
        return "cpp"
    return ""


def levels_badge(levels) -> str:
    order = [level for level in ("light", "medium", "high") if level in levels]
    return "+".join(SECTION_TITLES[level] for level in order) or "—"


def model_badges(tag, baked=False, hint=False, phase=None, deviation=False):
    badges = []
    if baked:
        badges.append("запечённые")
    if hint or deviation or (isinstance(phase, int) and phase >= 2):
        badges.append("абляция")
    lowered = str(tag or "").lower()
    if any(marker in lowered for marker in NEW_MODEL_MARKERS):
        badges.append("новая")
    return badges


def read_jsonl(path: Path):
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except ValueError:
                    continue
                if isinstance(value, dict):
                    rows.append(value)
    except (OSError, UnicodeError):
        pass
    return rows


def zoo_cell(line: dict) -> dict:
    """Cell for one line of a zoo-night results.jsonl."""
    outcome = str(line.get("outcome") or "episode")
    row = line.get("row") if isinstance(line.get("row"), dict) else {}
    if outcome == "skipped_deadline":
        return {
            "kind": "skipped",
            "label": "SKIP",
            "icon": "⏭",
            "title": "Пропущено: не успело до дедлайна",
            "stalled": False,
            "reason": "не успел",
            "status_text": "не успел",
            "tool_calls": None,
            "wall_seconds": 0,
            "wall_display": "—",
            "model": row.get("model"),
        }
    if row:
        cell = cell_from_episode(row)
        if outcome != "episode":
            cell["outcome"] = outcome
        return cell
    if outcome == "runner_error":
        error = short_error(line.get("error") or "runner_error")
        return {
            "kind": "infra",
            "label": "I",
            "icon": "⚠️",
            "title": "Сбой запуска: " + error,
            "stalled": False,
            "reason": "сбой: " + error,
            "status_text": "Сбой запуска",
            "tool_calls": None,
            "wall_seconds": 0,
            "wall_display": "—",
            "model": line.get("tag"),
        }
    return {
        "kind": "unknown",
        "label": "?",
        "icon": "?",
        "title": "Нет данных эпизода",
        "stalled": False,
        "reason": "нет данных",
        "status_text": "неизвестно",
        "tool_calls": None,
        "wall_seconds": 0,
        "wall_display": "—",
        "model": line.get("tag"),
    }


# Owner decision 2026-10-28: for Light and Medium the result is the working code - correct code (W + C) ranks
# first and self-verified W breaks ties ("can I trust the model's word"). High and above keep W first: there the
# model's own verification is the point.
CODE_FIRST_LEVELS = {"light", "medium"}


def rank_pair(row, code_first):
    cw = row["counts"].get("cw", 0)
    code = cw + row["counts"].get("code", 0)
    return (code, cw) if code_first else (cw, code)


def winner_ratio(best, code_first, total):
    return f"{best[0]}/{total} код, W {best[1]}" if code_first else f"{best[0]}/{total}"


def section_winners(rows, max_played, excluded=(), code_first=False):
    """Winner rows per DASHBOARD_FIXES_2.md §3 (ranking per CODE_FIRST_LEVELS), with played-task counts."""
    if max_played <= 0:
        return [], (0, 0)
    eligible = [
        row
        for row in rows
        if row.get("played", row["task_count"]) == max_played
        and row["id"] not in excluded
        and not row.get("hint")
    ]
    if not eligible:
        return [], (0, 0)
    best = max(rank_pair(row, code_first) for row in eligible)
    return [row for row in eligible if rank_pair(row, code_first) == best], best


def sort_group_rows(rows, code_first=False):
    rows.sort(
        key=lambda row: (
            0 if row["counts"].get("running", 0) or row["counts"].get("stalled", 0) else 1,
            tuple(-v for v in rank_pair(row, code_first)),
            row["model"].lower(),
            row["label"].lower(),
        )
    )


def participant_row(group_name, tag, config, cells, extra):
    played = sum(1 for cell in cells.values() if cell.get("kind") not in ("skipped", "pending"))
    wall_seconds = sum(cell_seconds(cell) for cell in cells.values())
    counts = count_cells([{"cells": cells}])
    title = f"Модель {tag}" + (f"\nКонфиг: {config}" if config else "")
    if extra.get("badges"):
        title += "\nБейджи: " + ", ".join(extra["badges"])
    return {
        "id": "|".join([group_name, tag, config]),
        "source": group_name,
        "label": config,
        "model": tag,
        "short_model": short_model(tag),
        "config": config,
        "config_short": config_short(config),
        "show_config": True,
        "badges": extra.get("badges") or [],
        "hint": bool(extra.get("hint")),
        "cells": cells,
        "tasks": sorted(cells, key=task_sort_key),
        "counts": counts,
        "task_count": len(cells),
        "played": played,
        "wall_seconds": int(round(wall_seconds)),
        "wall_display": fmt_duration(wall_seconds) if wall_seconds else "—",
        "incomplete": False,
        "winner": False,
        "old_benchmark": False,
        "title": title,
    }


def attach_running_cells(group_name, journals, participants, now, with_stage=False):
    """Add live transcripts from <group>-...-run folders to participant rows."""
    for folder_name, folder_path in journals:
        if not folder_name.startswith(group_name + "-") or not folder_name.endswith("-run"):
            continue
        config_path = RUNS / folder_name[: -len("-run")]
        configuration = read_json(config_path / "configuration.json")
        if not isinstance(configuration, dict):
            continue
        models = configuration.get("models")
        entry = None
        if isinstance(models, dict):
            entry = models.get("run")
            if not isinstance(entry, dict) and len(models) == 1:
                entry = next(iter(models.values()))
        tag = ""
        if isinstance(entry, dict) and isinstance(entry.get("tag"), str):
            tag = entry["tag"].strip()
        config = str(configuration.get("config_id") or "")
        if configuration.get("night") and "/" in config:
            config = config.split("/", 1)[1]   # zoo_night writes "<model_key>/<config>"; results.jsonl has "<config>"
        if not tag:
            continue
        level = str(configuration.get("level") or "").lower()
        stage = configuration.get("stage")
        if not isinstance(stage, int) or isinstance(stage, bool):
            stage = 1
        tasks = find_tasks(folder_path)
        for task, task_dir in tasks.items():
            if with_stage:
                key = (stage, tag, config)
            else:
                task_level_value = task_level(task) or level
                key = (task_level_value, tag, config)
            participant = participants.get(key)
            if participant is None:
                participant = participants.get((tag, config))
            if participant is None:
                continue
            if task in participant["cells"]:
                continue
            cell, _episode = classify_episode(task_dir, now)
            if cell.get("kind") == "run":
                participant["cells"][task] = cell


def build_zoo_group(name: str, path: Path, journals, now: float):
    """One zoo-night contest, built from results.jsonl with Light/Medium sections."""
    results = read_jsonl(path / "results.jsonl")
    status = read_json(path / "status.json")
    # Do not surface a user-cancelled launch that produced no completed
    # episode.  Its plan/logs deliberately remain on disk for audit, but it
    # is not a meaningful contest result and otherwise becomes a UI ghost.
    if (
        isinstance(status, dict)
        and status.get("cancelled_by_user") is True
        and not results
        and int(status.get("done") or 0) == 0
    ):
        return None
    participants = {}
    # Seed every planned model/config first, so a participant whose first episode is still running
    # (no results.jsonl line yet) still gets its row and live cell.
    plans = sorted(path.glob("plan-*.json"))
    planned = read_json(plans[-1]) if plans else None
    for run in (planned or {}).get("runs", []) if isinstance(planned, dict) else []:
        if not isinstance(run, dict) or not run.get("task_id"):
            continue
        level = str(run.get("level") or "").lower() or task_level(str(run["task_id"]))
        key = (level, str(run.get("tag") or "unknown"), str(run.get("config_id") or ""))
        participants.setdefault(key, {
            "level": level, "tag": key[1], "config": key[2], "cells": {},
            "planned_tasks": set(),
            "baked": bool(run.get("baked")), "hint": bool(run.get("hint")), "deviation": bool(run.get("deviation")),
            "phase": run.get("phase") if isinstance(run.get("phase"), int) else None, "levels": {level},
        })["planned_tasks"].add(str(run["task_id"]))
    for line in results:
        task = str(line.get("task_id") or "")
        if not task:
            continue
        level = str(line.get("level") or "").lower() or task_level(task)
        tag = str(line.get("tag") or line.get("original") or "unknown")
        row_payload = line.get("row") if isinstance(line.get("row"), dict) else {}
        if tag == "unknown" and row_payload.get("model"):
            tag = str(row_payload.get("model"))
        config = str(line.get("config_id") or "")
        key = (level, tag, config)
        participant = participants.setdefault(
            key,
            {
                "level": level,
                "tag": tag,
                "config": config,
                "cells": {},
                "planned_tasks": set(),
                "baked": False,
                "hint": False,
                "deviation": False,
                "phase": None,
                "levels": set(),
            },
        )
        participant["levels"].add(level)
        participant["baked"] = participant["baked"] or bool(line.get("baked"))
        participant["hint"] = participant["hint"] or bool(line.get("hint"))
        participant["deviation"] = participant["deviation"] or bool(line.get("deviation"))
        phase = line.get("phase")
        if isinstance(phase, int) and not isinstance(phase, bool):
            participant["phase"] = max(participant["phase"] or 0, phase)
        participant["cells"][task] = zoo_cell(line)

    attach_running_cells(name, journals, participants, now)
    # Unplayed plan squares stay "pending" only while the run can still reach them.
    open_run = (
        isinstance(status, dict)
        and str(status.get("state") or "").lower() not in {"stopped", "cancelled", "canceled", "completed", "finished"}
        and status.get("finished") is not True
        and now - tree_max_mtime(path) <= ABANDONED_SECONDS
    )
    for participant in participants.values():
        participant["cells"] = pad_planned(participant["cells"], participant.get("planned_tasks", ()), live=open_run)
    if not participants:
        return None

    section_map = {}
    for participant in participants.values():
        tag = participant["tag"]
        config = participant["config"]
        cells = participant["cells"]
        if not cells:
            continue
        row = participant_row(
            name,
            tag,
            config,
            cells,
            {
                "badges": model_badges(
                    tag,
                    participant["baked"],
                    participant["hint"],
                    participant["phase"],
                    participant["deviation"],
                ),
                "hint": participant["hint"],
            },
        )
        level = participant["level"] or task_level(row["tasks"][0])
        row["id"] = "|".join([name, level, tag, config])
        section_map.setdefault(level, []).append(row)

    sections = []
    winner_parts = []
    all_rows = []
    for level in sorted(section_map, key=lambda item: SECTION_ORDER.get(item, 9)):
        rows = section_map[level]
        max_played = max(row["played"] for row in rows)
        for row in rows:
            row["incomplete"] = row["played"] < max_played
        code_first = level in CODE_FIRST_LEVELS
        winners, best = section_winners(rows, max_played, code_first=code_first)
        best_cw = best[1] if code_first else best[0]
        winner_ids = [row["id"] for row in winners]
        winner_names = []
        for row in winners:
            label = row["short_model"]
            if row["config_short"]:
                label += " · " + row["config_short"]
            winner_names.append(label)
        for row in winners:
            row["winner"] = True
        winner_text = ""
        if winners:
            ratio = winner_ratio(best, code_first, max_played)
            winner_text = "🏆 " + ", ".join(winner_names) + f" ({ratio})"
            winner_parts.append(f"{SECTION_TITLES.get(level, level)}: {', '.join(winner_names)} ({ratio})")
        sort_group_rows(rows, code_first)
        all_rows.extend(rows)
        sections.append(
            {
                "key": level,
                "title": SECTION_TITLES.get(level, level.upper()),
                "rows": rows,
                "winner_text": winner_text,
                "winner_ids": winner_ids,
                "winner_names": winner_names,
                "ratio": winner_ratio(best, code_first, max_played) if winners else "",
                "note": "подсказка не участвует" if any(row.get("hint") for row in rows) else "",
                "max_played": max_played,
            }
        )

    counts = count_cells(all_rows)
    episodes = sum(row["played"] for row in all_rows)
    wall_seconds = sum(row["wall_seconds"] for row in all_rows)
    levels = {section["key"] for section in sections}
    finished = isinstance(status, dict) and status.get("finished") is True
    cancelled = isinstance(status, dict) and str(status.get("state") or "").lower() in {
        "stopped", "cancelled", "canceled",
    }
    # A cancelled campaign can retain a fresh partial transcript.  Its cells are
    # useful for inspection, but they must not keep the whole campaign marked
    # as live until the normal transcript-staleness window expires.
    live = not cancelled and (counts.get("running", 0) > 0 or counts.get("stalled", 0) > 0)
    if cancelled:
        state_class = "stopped"
    elif live:
        state_class = "running"
    elif finished:
        state_class = "completed"
    elif isinstance(status, dict):
        state_class = "running"
    else:
        state_class = "unknown"
    updated_ts = tree_max_mtime(path)
    if updated_ts <= 0:
        return None
    if state_class == "running" and now - updated_ts > ABANDONED_SECONDS:
        state_class = "abandoned"
    reason = service_reason(name, isinstance(status, dict), bool(results))
    winner_text = ("🏆 " + " · ".join(winner_parts)) if winner_parts else ""
    display_name = DISPLAY_NAME_ALIASES.get(name, name)
    winner_ids = [row["id"] for row in all_rows if row["winner"]]
    review = any(
        cell.get("review_required")
        for row in all_rows
        for cell in row["cells"].values()
        if isinstance(cell, dict)
    )
    live_rows = [
        (row, task, cell)
        for row in all_rows
        for task, cell in row["cells"].items()
        if isinstance(cell, dict) and cell.get("kind") == "run"
    ]
    live_info = None
    if live_rows:
        live_rows.sort(key=lambda item: (bool(item[2].get("stalled")), item[1]))
        row, task, cell = live_rows[0]
        live_info = {
            "model": row["model"],
            "short_model": row["short_model"],
            "task": task,
            "elapsed": cell.get("elapsed") or fmt_duration(cell.get("elapsed_seconds")),
            "stalled": bool(cell.get("stalled")),
        }
        if len(live_rows) > 1:
            live_info["more"] = len(live_rows) - 1
    done = status.get("done") if isinstance(status, dict) else None
    total = status.get("total") if isinstance(status, dict) else None
    title_lines = [
        f"Конкурс: {display_name}",
        f"Состояние: {state_ru(state_class, state_class)}",
        f"Моделей+конфигов: {len(all_rows)} · эпизодов: {episodes}",
        "Секции Light/Medium сравниваются только внутри себя.",
    ]
    if winner_text:
        title_lines.append("Победители: " + winner_text.replace("🏆 ", "") + "; предварительно")
    return {
        "name": display_name,
        "category": "contest",
        "zoo_night": True,
        "level": levels_badge(levels),
        "state": state_class,
        "state_ru": state_ru(state_class, state_class),
        "state_class": state_class,
        "live": live,
        "live_info": live_info,
        "complete": state_class == "completed" and not live,
        "stopped": state_class == "stopped",
        "counts": counts,
        "result_text": result_text(counts),
        "episodes": episodes,
        "episodes_text": f"{episodes} {plural_ru(episodes, 'эпизод', 'эпизода', 'эпизодов')}",
        "task_total": max((section["max_played"] for section in sections), default=0),
        "models": len({row["model"] for row in all_rows}),
        "wall_seconds": int(round(wall_seconds)),
        "wall_display": fmt_duration(wall_seconds) if wall_seconds else "—",
        "updated_ts": updated_ts,
        "updated": fmt_dt(updated_ts),
        "updated_ago": fmt_ago(updated_ts, now),
        "recent": now - updated_ts <= RECENT_SECONDS,
        "winner": {
            "ids": winner_ids,
            "names": [row["short_model"] for row in all_rows if row["winner"]],
            "text": winner_text,
            "ratio": "",
            "best_cw": 0,
            "total": 0,
            "provisional": bool(winner_ids),
        },
        "winner_text": winner_text,
        "winner_ids": winner_ids,
        "sections": sections,
        "review_required": review,
        "rows": all_rows,
        "sources": [name],
        "service": bool(reason),
        "service_reason": reason,
        "status_total": 1 if isinstance(status, dict) else 0,
        "status_completed": 1 if finished else 0,
        "done_text": f"{done}/{total}" if isinstance(done, int) and isinstance(total, int) else "—",
        "detail_ru": "",
        "title": "\n".join(title_lines),
    }


def read_text_safe(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return ""


def trial_verdict(path: Path):
    lines = []
    for line in read_text_safe(path / "summary.md").splitlines():
        text = line.strip()
        if text.startswith("- ") and ("KEEP" in text or "DROP" in text):
            cleaned = (
                text[2:]
                .replace("**", "")
                .replace(" -> ", " → ")
            )
            lines.append(cleaned)
    return lines


def build_trial_group(name: str, path: Path, journals, now: float):
    """Ministral trial: two stages, no cup, optional KEEP/DROP verdict."""
    results = read_jsonl(path / "episodes.jsonl")
    participants = {}
    for line in results:
        task = str(line.get("task_id") or "")
        row_payload = line.get("row") if isinstance(line.get("row"), dict) else {}
        if not task and isinstance(row_payload.get("case_id"), str):
            task = row_payload["case_id"]
        if not task:
            continue
        stage = line.get("stage")
        if not isinstance(stage, int) or isinstance(stage, bool):
            stage = 1
        tag = str(line.get("tag") or line.get("model_key") or (row_payload.get("model") or "unknown"))
        config = str(line.get("config_id") or "")
        key = (stage, tag, config)
        participant = participants.setdefault(
            key,
            {
                "stage": stage,
                "tag": tag,
                "config": config,
                "cells": {},
                "levels": set(),
                "hint": bool(line.get("hint")),
            },
        )
        participant["levels"].add(str(line.get("level") or "").lower() or task_level(task))
        participant["cells"][task] = cell_from_episode(row_payload) if row_payload else zoo_cell(line)

    attach_running_cells(name, journals, participants, now, with_stage=True)
    if not participants:
        return None
    # While the trial runs, every row of a stage shows the stage's whole task grid.
    trial_status = read_json(path / "status.json")
    if not (isinstance(trial_status, dict) and trial_status.get("finished") is True):
        stage_tasks = {}
        for participant in participants.values():
            stage_tasks.setdefault(participant["stage"], set()).update(participant["cells"])
        for participant in participants.values():
            participant["cells"] = pad_planned(participant["cells"], stage_tasks[participant["stage"]])

    section_map = {}
    for participant in participants.values():
        if not participant["cells"]:
            continue
        row = participant_row(
            name,
            participant["tag"],
            participant["config"],
            participant["cells"],
            {"badges": model_badges(participant["tag"], hint=participant["hint"]), "hint": participant["hint"]},
        )
        row["id"] = "|".join([name, f"stage{participant['stage']}", participant["tag"], participant["config"]])
        section_map.setdefault(participant["stage"], []).append(row)
    if not section_map:
        return None

    sections = []
    all_rows = []
    for stage in sorted(section_map):
        rows = section_map[stage]
        max_played = max(row["played"] for row in rows)
        for row in rows:
            row["incomplete"] = row["played"] < max_played
        sort_group_rows(rows)
        all_rows.extend(rows)
        sections.append(
            {
                "key": f"stage{stage}",
                "title": f"Этап {stage}",
                "rows": rows,
                "winner_text": "",
                "winner_ids": [],
                "winner_names": [],
                "ratio": "",
                "note": "без кубка",
                "max_played": max_played,
            }
        )

    counts = count_cells(all_rows)
    episodes = sum(row["played"] for row in all_rows)
    wall_seconds = sum(row["wall_seconds"] for row in all_rows)
    levels = {level for participant in participants.values() for level in participant["levels"]}
    status = read_json(path / "status.json")
    live = counts.get("running", 0) > 0 or counts.get("stalled", 0) > 0
    done = status.get("index") if isinstance(status, dict) else None
    total = status.get("total") if isinstance(status, dict) else None
    finished = isinstance(status, dict) and status.get("finished") is True
    if not finished and not live:
        finished = bool(trial_verdict(path)) or (
            isinstance(done, int) and isinstance(total, int) and done >= total > 0
        )
    state_class = "running" if live else ("completed" if finished else ("running" if isinstance(status, dict) else "unknown"))
    updated_ts = tree_max_mtime(path)
    if updated_ts <= 0:
        return None
    if state_class == "running" and isinstance(status, dict) and now - updated_ts > ABANDONED_SECONDS:
        state_class = "abandoned"
    verdict = trial_verdict(path)
    title_lines = [
        f"Испытание: {name}",
        f"Состояние: {state_ru(state_class, state_class)}",
        f"Этапов: {len(sections)} · эпизодов: {episodes}",
        "Кубка нет: это отдельное испытание модели.",
    ]
    if verdict:
        title_lines.extend(verdict)
    return {
        "name": name,
        "category": "test",
        "trial": True,
        "level": levels_badge(levels),
        "state": state_class,
        "state_ru": state_ru(state_class, state_class),
        "state_class": state_class,
        "live": live,
        "live_info": None,
        "complete": state_class == "completed" and not live,
        "stopped": state_class == "stopped",
        "counts": counts,
        "result_text": result_text(counts),
        "episodes": episodes,
        "episodes_text": f"{episodes} {plural_ru(episodes, 'эпизод', 'эпизода', 'эпизодов')}",
        "task_total": max((section["max_played"] for section in sections), default=0),
        "models": len({row["model"] for row in all_rows}),
        "wall_seconds": int(round(wall_seconds)),
        "wall_display": fmt_duration(wall_seconds) if wall_seconds else "—",
        "updated_ts": updated_ts,
        "updated": fmt_dt(updated_ts),
        "updated_ago": fmt_ago(updated_ts, now),
        "recent": now - updated_ts <= RECENT_SECONDS,
        "winner": {"ids": [], "names": [], "text": "", "ratio": "", "best_cw": 0, "total": 0, "provisional": False},
        "winner_text": "",
        "winner_ids": [],
        "sections": sections,
        "review_required": any(cell.get("review_required") for row in all_rows for cell in row["cells"].values()),
        "rows": all_rows,
        "sources": [name],
        "service": bool(service_reason(name, isinstance(status, dict), bool(results))),
        "service_reason": service_reason(name, isinstance(status, dict), bool(results)),
        "status_total": 1 if isinstance(status, dict) else 0,
        "status_completed": 1 if finished else 0,
        "done_text": f"{done}/{total}" if isinstance(done, int) and isinstance(total, int) else "—",
        "detail_ru": "",
        "verdict": verdict,
        "verdict_text": " · ".join(verdict),
        "title": "\n".join(title_lines),
    }


# ---------------------------------------------------------------------------
# old config benchmark (read-only, outside round2/runs)

def old_config_label(row: dict) -> str:
    config = str(row.get("config_id") or "").strip()
    if config:
        return config
    settings = row.get("settings") if isinstance(row.get("settings"), dict) else row
    parts = []
    think = settings.get("think") if "think" in settings else row.get("reasoning")
    if think is not None:
        parts.append("think" if think else "direct")
    num_ctx = settings.get("num_ctx") or row.get("num_ctx")
    num_predict = settings.get("num_predict") or row.get("num_predict")
    if num_ctx or num_predict:
        parts.append(f"{num_ctx or '?'}/{num_predict or '?'}")
    temperature = settings.get("temperature") if "temperature" in settings else row.get("temperature")
    if temperature is not None:
        parts.append(f"t{temperature}")
    return " · ".join(parts) or "default"


def old_cell(row: dict) -> dict:
    ok = row.get("ok")
    passed = row.get("passed")
    checks = row.get("checks")
    if passed is None and isinstance(checks, dict) and checks:
        passed = all(bool(value) for value in checks.values())
    error = row.get("error")
    done = row.get("done_reason")
    seconds = row.get("seconds")
    if ok is False or error:
        kind, icon, status_text = "infra", "⚠️", "Сбой"
        reason = short_error(error) or "ok=false"
    elif passed is False:
        kind, icon, status_text = "fail", "❌", "Не прошёл"
        reason = "не прошёл проверку"
        if done:
            reason += f" · {done}"
    else:
        kind, icon, status_text = "cw", "✅", "Пройдено"
        reason = "passed" if passed is True else "ok"
        if done and done != "stop":
            reason += f" · {done}"
    title = f"{status_text}: {reason}"
    if seconds is not None:
        title += f" · {fmt_duration(seconds)}"
    return {
        "kind": kind,
        "label": "OK" if kind == "cw" else ("X" if kind == "fail" else "I"),
        "icon": icon,
        "title": title,
        "stalled": False,
        "wall_seconds": seconds,
        "wall_display": fmt_duration(seconds),
        "tool_calls": None,
        "model": row.get("model"),
        "reason": reason,
        "status_text": status_text,
        "old_benchmark": True,
    }


def old_result_text(counts, total) -> str:
    parts = [f"✅ {counts.get('cw', 0)}", f"❌ {counts.get('fail', 0)}"]
    if counts.get("infra"):
        parts.append(f"⚠️ {counts['infra']}")
    if counts.get("unknown"):
        parts.append(f"? {counts['unknown']}")
    return " · ".join(parts) + f" из {total}"


def build_old_test(test_id: str, name: str, rows, source: str, updated_ts: float, now: float):
    model_rows = {}
    for row in rows:
        model = str(row.get("model") or row.get("model_key") or "unknown")
        config = old_config_label(row)
        key = (model, config)
        model_row = model_rows.setdefault(
            key,
            {
                "id": "|".join([test_id, model, config]),
                "source": name,
                "label": config,
                "model": model,
                "short_model": short_model(model),
                "config": config,
                "config_short": config,
                "show_config": True,
                "badges": [],
                "cells": {},
                "levels": set(),
                "old_benchmark": True,
            },
        )
        task = str(row.get("case_id") or row.get("task") or "?")
        if task in model_row["cells"]:
            index = 2
            while f"{task} · {index}" in model_row["cells"]:
                index += 1
            task = f"{task} · {index}"
        model_row["cells"][task] = old_cell(row)
        model_row["levels"].add(
            str(row.get("level") or row.get("role") or "").lower() or task_level(task)
        )
    if not model_rows:
        return None
    result_rows = []
    all_levels = set()
    for model_row in model_rows.values():
        cells = model_row.pop("cells")
        levels = model_row.pop("levels")
        all_levels.update(levels)
        wall_seconds = sum(cell_seconds(cell) for cell in cells.values())
        config = model_row["config"]
        result_rows.append(
            {
                **model_row,
                "cells": cells,
                "tasks": sorted(cells, key=task_sort_key),
                "counts": count_cells([{"cells": cells}]),
                "task_count": len(cells),
                "played": len(cells),
                "wall_seconds": int(round(wall_seconds)),
                "wall_display": fmt_duration(wall_seconds) if wall_seconds else "—",
                "incomplete": False,
                "winner": False,
                "title": f"Старый бенчмарк: {model}"
                + (f"\nКонфиг: {config}" if config else ""),
            }
        )
    result_rows.sort(key=lambda row: (row["model"].lower(), row["config"].lower()))
    counts = count_cells(result_rows)
    wall_seconds = sum(row["wall_seconds"] for row in result_rows)
    levels = {level for level in all_levels if level}
    return {
        "id": test_id,
        "name": name,
        "source": source,
        "badge": "старый бенчмарк",
        "old_benchmark": True,
        "level": levels_badge(levels),
        "counts": counts,
        "result_text": old_result_text(counts, len(rows)),
        "total": len(rows),
        "models": len({row["model"] for row in result_rows}),
        "rows": result_rows,
        "wall_seconds": int(round(wall_seconds)),
        "wall_display": fmt_duration(wall_seconds) if wall_seconds else "—",
        "updated_ts": updated_ts,
        "updated": fmt_dt(updated_ts),
        "updated_ago": fmt_ago(updated_ts, now),
        "title": f"Старый бенчмарк: {name}\n{len(rows)} запросов, без инструментов и без кубка.",
    }


def load_old_benchmarks(now: float):
    tests = []
    root = OLD_ROOT
    try:
        folders = sorted(path for path in root.iterdir() if path.is_dir() and (path / "runs.jsonl").is_file())
    except OSError:
        folders = []
    for folder in folders:
        rows = read_jsonl(folder / "runs.jsonl")
        if not rows:
            continue
        updated_ts = safe_mtime(folder / "runs.jsonl") or tree_max_mtime(folder)
        test = build_old_test(f"folder::{folder.name}", folder.name, rows, "runs.jsonl", updated_ts, now)
        if test:
            tests.append(test)
    for filename in ("results.jsonl", "light_results.jsonl", "quality_results.jsonl"):
        path = root / filename
        if not os.path.isfile(path):
            continue
        rows = read_jsonl(path)
        if not rows:
            continue
        updated_ts = safe_mtime(path) or 0.0
        test = build_old_test(f"file::{filename}", filename, rows, "root", updated_ts, now)
        if test:
            tests.append(test)
    tests.sort(key=lambda test: (0 if test["source"] == "runs.jsonl" else 1, test["name"].lower()))
    return tests


def old_benchmark_tests(now: float):
    cached = _OLD_CACHE.get("tests")
    if cached is not None and now - _OLD_CACHE.get("ts", 0.0) < OLD_REFRESH_SECONDS:
        return cached
    tests = load_old_benchmarks(now)
    _OLD_CACHE["ts"] = now
    _OLD_CACHE["tests"] = tests
    return tests


def build_special_groups(journals, now: float):
    """zoo-night and ministral-trial groups, plus their running source mapping."""
    groups = []
    source_map = {}
    seen = set()
    try:
        folders = sorted(path for path in RUNS.iterdir() if path.is_dir())
    except OSError:
        folders = []
    for path in folders:
        name = path.name
        if name in seen:
            continue
        if name.startswith(ZOO_PREFIXES) and (os.path.isfile(path / "results.jsonl") or any(path.glob("plan-*.json"))):
            group = build_zoo_group(name, path, journals, now)
            seen.add(name)
        elif name.startswith("ministral-trial-") and (
            os.path.isfile(path / "episodes.jsonl") or os.path.isfile(path / "status.json")
        ):
            group = build_trial_group(name, path, journals, now)
            seen.add(name)
        else:
            continue
        if not group:
            continue
        groups.append(group)
        for folder_name, _folder_path in journals:
            if folder_name.startswith(name + "-"):
                source_map[folder_name] = group
    return groups, source_map


def build_last_run(contests, group_by_name):
    """Newest status.json across all contests, for the idle «Идёт сейчас» block."""
    candidates = [
        contest
        for contest in contests
        if contest.get("has_status")
        and (contest.get("status_ts") or 0) > 0
        and contest.get("state_class") in ("running", "completed", "stopped", "failed", "abandoned")
    ]
    if not candidates:
        return None
    contest = max(candidates, key=lambda item: item["status_ts"])
    group = group_by_name.get(contest.get("group") or contest["name"]) or contest
    state_class = contest.get("state_class") or "unknown"
    state_text = {
        "running": "идёт",
        "completed": "завершён",
        "stopped": "остановлен",
        "failed": "остановлен",
        "abandoned": "заброшен",
    }.get(state_class, "без статуса")
    try:
        when = datetime.fromtimestamp(contest["status_ts"]).strftime("%H:%M")
    except (OSError, OverflowError, ValueError):
        when = "—"
    model_key = contest.get("status_last_model")
    tags = contest.get("status_model_tags") or {}
    model = tags.get(str(model_key)) if model_key else None
    model_label = short_model(model) if model else (str(model_key) if model_key else "")
    case = contest.get("status_last_case")
    error = short_error(contest.get("status_error"))
    error_label = error.split(":", 1)[0].strip() if error else ""
    text = f"Последний прогон: {group['name']} {state_text} в {when}"
    if error_label:
        text += f" — {error_label}"
    tail = ", ".join(str(value) for value in (model_label, case) if value)
    if tail:
        text += f" ({tail})"
    return {
        "contest": group["name"],
        "state": state_class,
        "state_text": state_text,
        "when": when,
        "error": error,
        "model": model,
        "model_label": model_label,
        "case": case,
        "updated_ts": contest["status_ts"],
        "updated": fmt_dt(contest["status_ts"]),
        "text": text,
    }


# ---------------------------------------------------------------------------
# "Выводы" tab: per-model test counts and speed against the class, plus curated text
CONCLUSIONS_FILE = BASE / "conclusions.json"
SIZE_OF_LEVEL = {"superlight": "XS", "light": "S", "medium": "M", "high": "L", "cpp": "C++"}
SIZE_ORDER = ["XS", "S", "M", "L", "C++"]


def build_conclusions():
    """Counts from every zoo-format run (results.jsonl) and SuperLight; text and roles from conclusions.json."""
    stats = {}   # (model key, size) -> [n, full_pass, code_pass, seconds_sum, seconds_n]

    def add(key, size, full, code, seconds):
        s = stats.setdefault((key, size), [0, 0, 0, 0.0, 0])
        s[0] += 1
        s[1] += bool(full)
        s[2] += bool(code)
        if isinstance(seconds, (int, float)) and seconds > 0:
            s[3] += seconds
            s[4] += 1

    try:
        folders = [p for p in RUNS.iterdir() if p.is_dir()]
    except OSError:
        folders = []
    for path in folders:
        if path.name.startswith(ZOO_PREFIXES) or path.name.startswith("cpp-"):
            for line in read_jsonl(path / "results.jsonl"):
                row = line.get("row") if isinstance(line.get("row"), dict) else {}
                if line.get("outcome", "episode") != "episode" or not row:
                    continue
                infra = str(row.get("infra_error") or "")
                if "HTTP Error 400" in infra or "HTTP Error 500" in infra:
                    continue   # config not applicable / model did not load: not the model's result
                size = SIZE_OF_LEVEL.get(str(line.get("level") or "").lower())
                key = str(line.get("original") or line.get("tag") or row.get("model") or "")
                if size and key:
                    add(key, size, row.get("automated_workflow_pass"), row.get("code_pass"), row.get("wall_seconds"))
        elif path.name.startswith("superlight-"):
            for line in read_jsonl(path / "results.jsonl"):
                if line.get("kind") == "infra":
                    continue
                add(str(line.get("model") or ""), "XS", line.get("ok"), line.get("ok"), line.get("seconds"))

    averages = {}   # size -> mean of the models' average seconds
    for size in SIZE_ORDER:
        per_model = [s[3] / s[4] for (k, sz), s in stats.items() if sz == size and s[4]]
        if per_model:
            averages[size] = sum(per_model) / len(per_model)

    curated = read_json(CONCLUSIONS_FILE) or {}
    entries = curated.get("models") if isinstance(curated.get("models"), list) else []
    known = [e.get("key") for e in entries if isinstance(e, dict)]
    keys = known + sorted({k for k, _ in stats} - set(known))
    models = []
    for key in keys:
        entry = next((e for e in entries if isinstance(e, dict) and e.get("key") == key), {})
        classes = []
        for size in SIZE_ORDER:
            s = stats.get((key, size))
            if not s:
                continue
            avg = s[3] / s[4] if s[4] else None
            classes.append({
                "size": size, "tests": s[0], "full_pass": s[1], "code_pass": s[2],
                "avg_seconds": round(avg, 2) if avg is not None else None,
                "class_avg_seconds": round(averages[size], 2) if size in averages else None,
                "speed_ratio": round(avg / averages[size], 2) if avg and averages.get(size) else None,
            })
        models.append({
            "key": key, "name": entry.get("name") or short_model(key), "roles": entry.get("roles") or [],
            "status": entry.get("status") or "", "verdict": entry.get("verdict") or "",
            "give": entry.get("give") or "", "avoid": entry.get("avoid") or "", "weak": entry.get("weak") or "",
            "classes": classes, "tests_total": sum(c["tests"] for c in classes),
        })
    return {"updated": curated.get("updated"), "roles": curated.get("roles") or [], "models": models,
            "class_averages": {k: round(v, 2) for k, v in averages.items()}}


def build_state():
    now = time.time()
    dirs = top_dirs()
    dir_paths = dict(dirs)
    hidden_cancelled = set()
    for candidate_name, candidate_path in dirs:
        status = read_json(candidate_path / "status.json")
        if not isinstance(status, dict):
            continue
        if (
            status.get("cancelled_by_user") is True
            and int(status.get("done") or 0) == 0
            and not read_jsonl(candidate_path / "results.jsonl")
        ):
            hidden_cancelled.add(candidate_name)

    def belongs_to_hidden_cancelled(name: str) -> bool:
        return any(name == parent or name.startswith(parent + "-") for parent in hidden_cancelled)

    tasks_by_dir: dict[str, dict[str, Path]] = {}
    for name, path in dirs:
        tasks_by_dir[name] = find_tasks(path)

    candidate_names = [
        name for name, path in dirs
        if has_contest_files(path) and not belongs_to_hidden_cancelled(name)
    ]
    candidate_set = set(candidate_names)

    def resolve(name: str):
        """(contest, label) for a folder that may hold task subfolders."""
        if name in candidate_set:
            return name, "(self)"
        parent = best_parent(name, candidate_names)
        if parent:
            return parent, name[len(parent) + 1:]
        return name, "(self)"

    # Rows for each contest come from its own task subfolders and from
    # sibling folders named <contest>-<label>.
    rows_by_contest: dict[str, list[tuple[str, dict[str, Path]]]] = {
        name: [] for name in candidate_names
    }
    for name, tasks in tasks_by_dir.items():
        if not tasks:
            continue
        contest, label = resolve(name)
        if contest in rows_by_contest:
            rows_by_contest[contest].append((label, tasks))

    contests = []
    for name in candidate_names:
        path = dir_paths[name]
        rows = rows_by_contest.get(name, [])
        try:
            status_info = contest_status(path)
            configuration = read_json(path / "configuration.json")
            if not isinstance(configuration, dict):
                configuration = {}
            config_id = str(configuration.get("config_id") or "")
            models_map = configuration.get("models")
            if not isinstance(models_map, dict):
                models_map = {}
            config_level = configuration.get("level")
            status_raw = read_json(path / "status.json")
            status_ts = safe_mtime(path / "status.json") or 0.0
            status_model_tags = {
                str(key): str(entry.get("tag")).strip()
                for key, entry in models_map.items()
                if isinstance(entry, dict) and isinstance(entry.get("tag"), str)
            }
            night, trial = identity_fields(configuration, status_raw, path)
            group_name, identity_kind = contest_group_name(
                name, configuration, status_raw, path
            )
            model_tags = sorted(
                {
                    str(entry.get("tag")).strip()
                    for entry in models_map.values()
                    if isinstance(entry, dict)
                    and isinstance(entry.get("tag"), str)
                    and entry.get("tag").strip()
                }
            )

            # Recover label directory paths from the row data.
            row_dirs = []
            seen_dirs = set()
            for _label, tasks in rows:
                if tasks:
                    row_dir = next(iter(tasks.values())).parent
                    if row_dir != path and row_dir not in seen_dirs:
                        seen_dirs.add(row_dir)
                        row_dirs.append(row_dir)
            updated_ts = tree_max_mtime(path)
            for row_dir in row_dirs:
                updated_ts = tree_max_mtime(row_dir, seed=updated_ts)

            if updated_ts <= 0:
                continue

            has_status = os.path.isfile(path / "status.json")
            contest = {
                "name": name,
                "short_name": short_contest(name),
                "state": status_info["state"],
                "state_ru": status_info["state_ru"],
                "state_class": status_info["state_class"],
                "done_text": done_text(status_info["done"], status_info["total"]),
                "done_text_ru": done_text_ru(status_info["done"], status_info["total"]),
                "detail": status_info["detail"],
                "detail_ru": status_info["detail_ru"],
                "status_title": status_info["title"],
                "updated_ts": updated_ts,
                "updated": fmt_dt(updated_ts),
                "updated_ago": fmt_ago(updated_ts, now),
                "rows": [],
                "tasks": [],
                "recent": now - updated_ts <= RECENT_SECONDS,
                "has_status": has_status,
                "has_episodes": bool(rows),
                "group": group_name,
                "identity_kind": identity_kind,
                "night": night,
                "trial": trial,
                "config_id": config_id,
                "config_level": config_level,
                "model_tags": model_tags,
                "status_ts": status_ts,
                "status_last_model": status_raw.get("last_model") if isinstance(status_raw, dict) else None,
                "status_last_case": status_raw.get("last_case") if isinstance(status_raw, dict) else None,
                "status_error": status_raw.get("error") if isinstance(status_raw, dict) else None,
                "status_phase": status_raw.get("phase") if isinstance(status_raw, dict) else None,
                "status_progress_done": status_info.get("done"),
                "status_progress_total": status_info.get("total"),
                "status_model_tags": status_model_tags,
                "review_required": status_info.get("review_required", False),
                "finished": status_info.get("finished", False),
            }

            all_tasks = set()
            for label, tasks in sorted(rows, key=lambda item: item[0].lower()):
                cells = {}
                episode_models = []
                for task, task_dir in sorted(tasks.items(), key=lambda item: task_sort_key(item[0])):
                    try:
                        cell, episode = classify_episode(task_dir, now)
                        if isinstance(episode, dict):
                            model = episode.get("model")
                            if isinstance(model, str) and model.strip():
                                episode_models.append(model.strip())
                    except Exception as exc:  # noqa: BLE001
                        cell = {
                            "kind": "unknown",
                            "label": "?",
                            "icon": "?",
                            "title": f"Не удалось прочитать эпизод: {type(exc).__name__}: {exc}",
                            "stalled": False,
                            "reason": f"ошибка чтения: {type(exc).__name__}",
                            "status_text": "неизвестно",
                        }
                    cells[task] = cell
                    all_tasks.add(task)
                contest["rows"].append(
                    {
                        "label": label,
                        "model": model_tag_for(label, models_map, episode_models),
                        "config": config_id,
                        "cells": cells,
                        "source": name,
                    }
                )
            contest["tasks"] = sorted(all_tasks, key=task_sort_key)
            contest["counts"] = count_cells(contest["rows"])
            contest["result_text"] = result_text(contest["counts"])
            contest["live"] = (
                contest["counts"]["running"] > 0
                or contest["counts"]["stalled"] > 0
                or (
                    status_info["state_class"] == "running"
                    and now - updated_ts <= RUNNING_WINDOW
                )
            )
            reason = service_reason(name, has_status, bool(rows))
            contest["service"] = bool(reason)
            contest["service_reason"] = reason
            contests.append(contest)
        except Exception:  # noqa: BLE001 - skip one broken contest, keep the page alive
            continue

    recent_contests = list(contests)   # owner 2026-10-28: keep every run visible; newest first (was: last 24 h only)
    recent_contests.sort(key=lambda item: item["updated_ts"], reverse=True)
    service_by_name = {contest["name"]: contest["service"] for contest in contests}

    special_groups, special_sources = build_special_groups(dirs, now)
    all_groups = build_groups(contests, now) + special_groups
    recent_groups = list(all_groups)   # owner 2026-10-28: older nights no longer drop off the page
    recent_groups.sort(
        key=lambda group: (0 if group["live"] else 1, -group["updated_ts"], group["name"].lower())
    )
    group_by_source = {}
    group_by_name = {}
    for group in all_groups:
        group_by_name[group["name"]] = group
        for source_name in group["sources"]:
            group_by_source[source_name] = group
    group_by_source.update(special_sources)
    last_run = build_last_run(contests, group_by_name)

    old_payload = []
    for test in old_benchmark_tests(now):
        item = dict(test)
        item["updated"] = fmt_dt(item.get("updated_ts"))
        item["updated_ago"] = fmt_ago(item.get("updated_ts"), now)
        old_payload.append(item)

    # "Now running": any task folder without episode.json whose transcript
    # changed during the last two hours - including orphan folders.
    running = []
    for name, tasks in tasks_by_dir.items():
        if belongs_to_hidden_cancelled(name):
            continue
        for task, task_dir in tasks.items():
            try:
                if os.path.isfile(task_dir / "episode.json"):
                    continue
                transcript_path = task_dir / "transcript.jsonl"
                transcript_mtime = safe_mtime(transcript_path)
                if not transcript_mtime or now - transcript_mtime > RUNNING_WINDOW:
                    continue
                if episode_owner_alive(task_dir) is False:
                    continue   # interrupted by a stop/restart: not running, shown as ⏸ in its contest
                contest_name, label = resolve(name)
                manifest_mtime = safe_mtime(task_dir / "manifest.json")
                born = getattr(transcript_path.stat(), "st_birthtime", 0.0) or 0.0   # no manifest: transcript creation
                start = manifest_mtime or (born if 0 < born <= transcript_mtime else transcript_mtime)
                summary = transcript_summary(transcript_path)
                last_write_age = now - transcript_mtime
                elapsed_seconds = int(max(0, now - start))
                model = summary["model"] or "unknown"
                group = group_by_source.get(contest_name)
                # A user-cancelled campaign may leave a recently touched
                # transcript without an episode.json.  Do not advertise that
                # stale partial episode in the global "Now running" panel.
                if group and group.get("state_class") in ("stopped", "failed"):
                    continue
                display_name = group["name"] if group else contest_name
                service = group["service"] if group else service_by_name.get(contest_name)
                if service is None:
                    service = bool(service_reason(contest_name, False, True))
                running.append(
                    {
                        "contest": display_name,
                        "contest_raw": contest_name,
                        "group": display_name,
                        "category": group["category"] if group else "contest",
                        "short_contest": short_contest(display_name),
                        "short_group": short_contest(display_name),
                        "label": label,
                        "model": model,
                        "short_model": short_model(model),
                        "task": task,
                        "turn": summary["turn"] if summary["turn"] is not None else "—",
                        "elapsed": fmt_duration(elapsed_seconds),
                        "elapsed_ru": fmt_duration(elapsed_seconds),
                        "elapsed_seconds": elapsed_seconds,
                        "started_ago": fmt_ago(start, now),
                        "started_ago_ru": fmt_ago(start, now),
                        "started_at": fmt_dt(start),
                        "last_action": summary["last_action"] or "—",
                        "last_action_ru": action_ru(summary["last_action"]),
                        "last_tool_error": summary["last_tool_error"] or "",
                        "last_write_ago": fmt_ago(transcript_mtime, now),
                        "last_write_ago_ru": fmt_ago(transcript_mtime, now),
                        "last_write_at": fmt_dt(transcript_mtime),
                        "last_write_seconds": int(max(0, last_write_age)),
                        "stalled": last_write_age > STALL_SECONDS,
                        "last_event": summary["last_event"] or "—",
                        "service": service,
                    }
                )
            except Exception:  # noqa: BLE001
                continue
    running.sort(key=lambda item: item["last_write_seconds"])

    return {
        "generated_ts": now,
        "generated_at": fmt_dt(now),
        "runs_dir": str(RUNS),
        "recent_seconds": RECENT_SECONDS,
        "stall_seconds": STALL_SECONDS,
        "contests": recent_contests,
        "groups": recent_groups,
        "group_counts": {
            "contest": sum(1 for group in recent_groups if group["category"] == "contest"),
            "test": sum(1 for group in recent_groups if group["category"] == "test"),
        },
        "old_benchmarks": old_payload,
        "old_benchmark_counts": {
            "tests": len(old_payload),
            "folders": sum(1 for test in old_payload if test["source"] == "runs.jsonl"),
            "root_files": sum(1 for test in old_payload if test["source"] == "root"),
        },
        "last_run": last_run,
        "running": running,
        "conclusions": build_conclusions(),
        "ollama": ollama_state(),
    }


# ---------------------------------------------------------------------------
# page

PAGE = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Конкурсы моделей — мониторинг</title>
<style>
  :root {
    --bg: #16181d;
    --panel: #1b1e24;
    --panel-2: #20242b;
    --line: #2c313a;
    --text: #d6d8dc;
    --muted: #8b929c;
    --green: #7aa98c;
    --green-bg: rgba(122, 169, 140, .10);
    --yellow: #b3a06e;
    --yellow-bg: rgba(179, 160, 110, .10);
    --red: #b57a7e;
    --red-bg: rgba(181, 122, 126, .10);
    --blue: #7699b8;
    --blue-bg: rgba(118, 153, 184, .10);
    --grey: #858d97;
    --grey-bg: rgba(133, 141, 151, .08);
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; background: var(--bg); color: var(--text); }
  body { font: 13px/1.35 "Segoe UI", system-ui, -apple-system, sans-serif; }
  .mono { font-family: Consolas, "Cascadia Mono", monospace; }
  .muted { color: var(--muted); }
  .small { font-size: 12px; }
  .hidden { display: none !important; }
  header {
    padding: 9px 16px 7px;
    border-bottom: 1px solid var(--line);
    background: #171a20;
  }
  .head-row { display: flex; gap: 18px; align-items: baseline; justify-content: space-between; }
  h1 { margin: 0; font-size: 16px; font-weight: 600; color: #e0e2e6; white-space: nowrap; }
  .ollama-line {
    color: var(--muted);
    white-space: nowrap;
    overflow-x: auto;
    max-width: 72%;
    padding-bottom: 1px;
  }
  .summary {
    margin-top: 5px;
    font-size: 14px;
    color: #e2e4e8;
    white-space: nowrap;
    overflow-x: auto;
    padding-bottom: 1px;
  }
  .legend {
    margin-top: 6px;
    display: block;
    color: var(--muted);
    font-size: 11.5px;
    line-height: 1.6;
    white-space: normal;
    overflow-x: visible;
  }
  .legend .sep { color: #454b55; }
  .head-controls {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 8px 14px;
    margin-top: 6px;
  }
  .tabs { display: flex; gap: 5px; align-items: center; }
  .tab {
    background: transparent;
    border: 1px solid var(--line);
    border-radius: 4px;
    color: var(--muted);
    font: inherit;
    font-size: 12.5px;
    padding: 2px 9px;
    cursor: pointer;
  }
  .tab:hover { background: rgba(255, 255, 255, .03); color: var(--text); }
  .tab.active { background: var(--panel-2); color: var(--text); border-color: #3a424e; }
  .tab-count { color: #6f7680; font-size: 11px; margin-left: 3px; }
  /* «Выводы» tab */
  #conclusions { display: grid; gap: 14px; }
  #conclusions h2 { font-size: 14px; margin: 4px 0 6px; color: var(--text); }
  .cc-note { color: var(--muted); font-size: 12px; margin: 0 0 4px; }
  .cc-roles { width: 100%; border-collapse: collapse; font-size: 12.5px; }
  .cc-roles td { padding: 5px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
  .cc-roles td:first-child { font-weight: 700; width: 3.5em; }
  .cc-roles code { background: var(--panel-2); padding: 0 4px; border-radius: 3px; font-size: 11.5px; }
  .cc-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 420px), 1fr)); gap: 10px; }
  .cc-card { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 10px 12px; min-width: 0; }
  .cc-card.other { opacity: .6; }
  .cc-head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px; }
  .cc-name { font-weight: 700; font-size: 14px; }
  .cc-chip { font-size: 11px; padding: 1px 6px; border-radius: 10px; background: var(--blue-bg); color: var(--blue); }
  .cc-badge { font-size: 11px; padding: 1px 6px; border-radius: 10px; margin-left: auto; }
  .cc-badge.confirmed { background: var(--green-bg); color: var(--green); }
  .cc-badge.preliminary { background: var(--yellow-bg); color: var(--yellow); }
  .cc-badge.none { background: var(--grey-bg); color: var(--grey); }
  .cc-verdict { font-size: 12.5px; line-height: 1.5; margin: 6px 0; }
  .cc-kv { font-size: 12px; color: var(--muted); margin: 2px 0; }
  .cc-kv b { color: var(--text); font-weight: 600; }
  .cc-stats { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 6px; font-variant-numeric: tabular-nums; }
  .cc-stats th, .cc-stats td { padding: 3px 6px; border-bottom: 1px solid var(--line); text-align: right; }
  .cc-stats th:first-child, .cc-stats td:first-child { text-align: left; }
  .cc-fast { color: var(--green); } .cc-slow { color: var(--red); }
  .cc-pct { font-weight: 700; }
  .cc-pct.p95 { color: #4caf6a; } .cc-pct.p80 { color: #9ccc7a; } .cc-pct.p65 { color: #d8c25a; }
  .cc-pct.p50 { color: #e0954a; } .cc-pct.p0 { color: #d9645f; }
  .tab-dot { color: var(--blue); margin-left: 4px; }
  .head-right { display: flex; align-items: center; gap: 14px; margin-left: auto; }
  .check { color: var(--muted); font-size: 12px; user-select: none; }
  .check input { vertical-align: -2px; margin-right: 3px; }
  main { padding: 11px 16px 40px; }
  h2 { margin: 18px 0 7px; font-size: 14px; font-weight: 600; color: #c9cdd3; }
  .now-block {
    margin: 2px 0 13px;
    padding: 7px 10px 8px;
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 6px;
  }
  .now-block h2 { margin: 0 0 3px; }
  .now-line { font-size: 13px; color: #e2e4e8; }
  .type-badge {
    display: inline-block;
    padding: 0 5px;
    border: 1px solid #3a4049;
    border-radius: 3px;
    color: #949ba5;
    font-size: 10px;
    line-height: 1.45;
    background: rgba(255, 255, 255, .02);
  }
  table { width: 100%; border-collapse: collapse; }
  th, td {
    border: 1px solid var(--line);
    padding: 2px 6px;
    line-height: 1.3;
    text-align: left;
    vertical-align: middle;
    white-space: nowrap;
  }
  th { background: var(--panel-2); font-weight: 600; color: #c9cdd3; }
  tbody tr:nth-child(even) td { background: rgba(255, 255, 255, .012); }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; }
  .scroll { overflow: auto; max-width: 100%; }
  .group-card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 6px;
    margin: 0 0 7px;
    overflow: hidden;
  }
  .group-card.live { border-color: #34414d; }
  .group-head {
    display: flex;
    flex-wrap: wrap;
    gap: 3px 8px;
    align-items: baseline;
    width: 100%;
    padding: 5px 9px;
    background: transparent;
    border: 0;
    color: inherit;
    font: inherit;
    text-align: left;
    cursor: pointer;
  }
  .group-head:hover { background: rgba(255, 255, 255, .03); }
  .group-head:focus-visible, .model-head:focus-visible { outline: 1px solid var(--blue); outline-offset: -1px; }
  .chev { width: 12px; color: var(--muted); display: inline-block; }
  .gname { font-weight: 600; color: #dfe2e6; }
  .level-badge {
    display: inline-block;
    padding: 0 4px;
    border: 1px solid #3a4049;
    border-radius: 3px;
    color: #8b929c;
    font-size: 10px;
    line-height: 1.45;
  }
  .badge {
    display: inline-block;
    padding: 0 6px;
    border: 1px solid var(--line);
    border-radius: 999px;
    font-size: 11px;
    color: var(--muted);
  }
  .badge.running { color: var(--blue); border-color: rgba(118, 153, 184, .38); background: var(--blue-bg); }
  .badge.completed { color: var(--green); border-color: rgba(122, 169, 140, .38); background: var(--green-bg); }
  .badge.stopped { color: var(--yellow); border-color: rgba(179, 160, 110, .38); background: var(--yellow-bg); }
  .badge.failed { color: var(--red); border-color: rgba(181, 122, 126, .38); background: var(--red-bg); }
  .badge.abandoned { color: var(--grey); border-color: rgba(133, 141, 151, .38); background: var(--grey-bg); }
  .badge.unknown { color: var(--grey); }
  .result { font-size: 12px; color: #c5c9cf; }
  .test-result { font-size: 12px; color: #c5c9cf; }
  .verdict { font-size: 12px; color: var(--yellow); }
  .winner { color: var(--yellow); font-size: 12px; }
  .winner-mark { color: var(--yellow); font-size: 12px; }
  .provisional { color: #6f7680; font-size: 10px; }
  .incomplete-note {
    color: var(--red);
    font-size: 10px;
    border: 1px dashed rgba(181, 122, 126, .38);
    border-radius: 3px;
    padding: 0 3px;
  }
  .live-note { color: var(--blue); font-size: 12px; }
  .g-extras { display: contents; }
  .gmeta { margin-left: auto; color: var(--muted); font-size: 11.5px; white-space: nowrap; }
  .service-badge {
    font-size: 10px;
    color: #6f7680;
    border: 1px dashed #3a4049;
    border-radius: 4px;
    padding: 0 4px;
  }
  .group-body { padding: 0 9px 6px; }
  .section-head {
    display: flex;
    flex-wrap: wrap;
    gap: 3px 8px;
    align-items: baseline;
    padding: 4px 0 2px;
    border-top: 1px solid #242932;
  }
  .section-title { font-weight: 600; color: #cfd4da; font-size: 12px; }
  .section-winner { color: var(--yellow); font-size: 11.5px; }
  .section-note { color: #6f7680; font-size: 10px; }
  .row-badge {
    display: inline-block;
    padding: 0 4px;
    border: 1px solid #3a4049;
    border-radius: 3px;
    color: #8b929c;
    font-size: 10px;
    line-height: 1.4;
  }
  .old-title { margin-top: 24px; }
  .old-note { margin: 0 0 7px; }
  .old-card .group-head { cursor: pointer; }
  .old-badge {
    display: inline-block;
    padding: 0 5px;
    border: 1px dashed #3a4049;
    border-radius: 4px;
    color: #7f8792;
    font-size: 10px;
    line-height: 1.45;
  }
  .empty-note { margin: 10px 0; }
  .model-block { border-top: 1px solid #242932; }
  .model-head {
    display: flex;
    flex-wrap: wrap;
    gap: 2px 7px;
    align-items: baseline;
    width: 100%;
    padding: 3px 0;
    background: transparent;
    border: 0;
    color: inherit;
    font: inherit;
    text-align: left;
    cursor: pointer;
  }
  .model-head:hover { background: rgba(255, 255, 255, .02); }
  .model-head { align-items: center; flex-wrap: nowrap; }
  .mlead { display: inline-flex; align-items: center; gap: 6px; flex: 0 0 380px; max-width: 380px;
           overflow: hidden; white-space: nowrap; }
  .mlead .mname { overflow: hidden; text-overflow: ellipsis; }
  .cellbar.big { gap: 8px; }
  .cellbar.big .cell-square { min-width: 75px; height: 75px; border-radius: 8px; }
  .cellbar.big .cell-square .sq-id { font-size: 19px; }
  .cellbar.big .cell-square .sq-min { font-size: 14px; }
  /* Phone width: the model name gets its own line and the squares wrap under it instead of running off the card. */
  @media (max-width: 640px) {
    .model-head { flex-wrap: wrap; }
    .mlead { flex: 1 1 100%; max-width: 100%; }
    .cellbar { max-width: 100%; }
    .cellbar.big { gap: 6px; }
    .cellbar.big .cell-square { min-width: 58px; height: 58px; }
    .cellbar.big .cell-square .sq-id { font-size: 16px; }
    .cellbar.big .cell-square .sq-min { font-size: 12px; }
    .cc-roles td, .cc-roles code { overflow-wrap: anywhere; white-space: normal; }
  }
  .run-card { padding: 8px 0 10px; border-bottom: 1px solid #242932; }
  .cell-square.clickable { cursor: pointer; position: relative; }
  .cell-square.clickable::after { content: "⤢"; position: absolute; top: 1px; right: 3px; font-size: 10px; opacity: .6; }
  .cell-square.clickable:hover { outline: 2px solid #8ab4f8; outline-offset: 1px; filter: brightness(1.15); }
  #detail-panel { position: fixed; top: 0; right: 0; bottom: 0; width: min(760px, 52vw); z-index: 50;
                  background: #15181d; border-left: 1px solid #333944; box-shadow: -8px 0 24px rgba(0,0,0,.45);
                  display: flex; flex-direction: column; }
  #detail-panel.hidden { display: none; }
  .detail-head { display: flex; align-items: center; gap: 10px; padding: 10px 14px; border-bottom: 1px solid #2a2f37; }
  .detail-head .x { margin-left: auto; background: none; border: 1px solid #3a4049; color: #c9cdd3;
                    border-radius: 6px; padding: 2px 10px; cursor: pointer; font-size: 16px; }
  .detail-body { overflow: auto; padding: 4px 14px 16px; flex: 1; }
  .detail-body h3 { font-size: 12.5px; color: #9aa3ad; margin: 12px 0 6px; font-weight: 600; }
  .detail-body pre { white-space: pre-wrap; word-break: break-word; background: #0f1216; border: 1px solid #262b33;
                     border-radius: 6px; padding: 8px 10px; font-size: 12px; max-height: 380px; overflow: auto; margin: 4px 0; }
  .detail-turn { border-top: 1px solid #262b33; padding: 8px 0; }
  .detail-turn .t-head { font-size: 12.5px; color: #c5c9cf; }
  .detail-body details summary { cursor: pointer; color: #9aa3ad; font-size: 12px; }
  .detail-body .res-bad { border-color: #5a3434; }
  .detail-wait { color: #8ab4f8; font-size: 12.5px; padding: 10px 0; }
  .live-dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; background: #4cc38a;
              box-shadow: 0 0 6px #4cc38a; animation: livepulse 1.6s ease-in-out infinite; }
  @keyframes livepulse { 50% { opacity: .35; } }
  .run-title { font-size: 13.5px; margin-bottom: 6px; }
  .run-meta { color: var(--muted); font-size: 12px; margin-top: 6px; }
  .run-meta .toolerror { color: #c98b8b; }
  .mname { color: #d7dade; font-size: 12.5px; }
  .mconfig { color: #7f8792; font-size: 11.5px; }
  .rowlabel-mini { color: #6f7680; font-size: 10px; }
  .mright {
    margin-left: auto;
    display: inline-flex;
    align-items: center;
    justify-content: flex-end;
    gap: 2px 9px;
    flex-wrap: wrap;
  }
  .mcounts { color: #c5c9cf; font-size: 12px; white-space: nowrap; }
  .mtime { color: var(--muted); font-size: 11.5px; }
  .cellbar { display: inline-flex; align-items: center; gap: 6px; flex-wrap: wrap; padding: 4px 0; }
  .cell-square {
    display: inline-flex;
    flex: 0 0 auto;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-width: 50px;
    height: 50px;
    padding: 2px 5px;
    border: 1px solid #3a4049;
    border-radius: 6px;
    background: transparent;
    color: #c9cdd3;
    line-height: 1.05;
    white-space: nowrap;
  }
  .cell-square .sq-id { font-size: 14px; font-weight: 700; }
  .cell-square .sq-min { font-size: 11.5px; line-height: 1.2; opacity: .9; }
  .cell-square.cw {
    color: #b7d6c3;
    border-color: rgba(122, 169, 140, .62);
    background: rgba(122, 169, 140, .24);
  }
  .cell-square.code {
    color: #ddcc9d;
    border-color: rgba(179, 160, 110, .62);
    background: rgba(179, 160, 110, .24);
  }
  .cell-square.fail {
    color: #e2b4b7;
    border-color: rgba(181, 122, 126, .62);
    background: rgba(181, 122, 126, .24);
  }
  .cell-square.infra {
    color: #d2d7dd;
    border-color: rgba(133, 141, 151, .60);
    background: rgba(133, 141, 151, .20);
  }
  .cell-square.run {
    color: #bcd5ea;
    border-color: rgba(118, 153, 184, .68);
    background: rgba(118, 153, 184, .24);
  }
  .cell-square.stall {
    color: #e2b4b7;
    border-color: rgba(181, 122, 126, .68);
    border-style: dashed;
    background: rgba(181, 122, 126, .18);
  }
  .cell-square.pending {
    color: #6f7680;
    border-color: rgba(133, 141, 151, .40);
    border-style: dotted;
  }
  .cell-square.skipped {
    color: #858d97;
    border-color: rgba(133, 141, 151, .45);
    border-style: dashed;
  }
  .cell-square.empty {
    color: #6f7680;
    border-color: #3a4049;
    border-style: dashed;
  }
  .stage-wrap { padding: 0 0 4px 18px; }
  .stages { width: 100%; table-layout: fixed; border-collapse: collapse; }
  .stages td {
    border: 0;
    border-top: 1px solid #23272f;
    padding: 2px 6px;
    font-size: 12px;
    line-height: 1.35;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .stages tr:nth-child(even) td { background: transparent; }
  .stage-task { width: 44px; font-weight: 600; color: #c9cdd3; }
  .stage-icon { width: 26px; text-align: center; font-size: 13px; }
  .stage-time { width: 92px; color: var(--muted); }
  .stage-actions { width: 104px; color: var(--muted); }
  .stage-result { color: #b9bec6; }
  .stage.cw .stage-icon { color: var(--green); }
  .stage.code .stage-icon { color: var(--yellow); }
  .stage.fail .stage-icon { color: var(--red); }
  .stage.infra .stage-icon { color: var(--grey); }
  .stage.skipped .stage-icon { color: var(--grey); }
  .stage.old_benchmark .stage-icon { color: var(--grey); }
  .stage.run .stage-icon { color: var(--blue); }
  .stage.stall .stage-icon { color: var(--red); }
  .stage.unknown .stage-icon { color: var(--muted); }
  .stalled-text { color: var(--red); }
  .stall-badge {
    display: inline-block;
    margin-left: 4px;
    padding: 0 5px;
    border-radius: 4px;
    font-size: 11px;
    color: var(--red);
    border: 1px solid rgba(181, 122, 126, .35);
    background: var(--red-bg);
  }
  .action { color: #c6cad0; }
  .toolerror { max-width: 380px; overflow: hidden; text-overflow: ellipsis; color: #c98a8e; }
  .error {
    margin: 0 0 10px;
    padding: 8px 10px;
    border: 1px solid #6b3b3e;
    border-radius: 5px;
    background: #241719;
    color: #d9a3a7;
  }
</style>
</head>
<body>
<header>
  <div class="head-row">
    <h1>Конкурсы моделей</h1>
    <div id="ollama-line" class="ollama-line mono small">Ollama: проверяется…</div>
  </div>
  <div class="legend" id="legend">
    <span>✅ решено и проверено моделью</span><span class="sep">·</span>
    <span>🟡 код верный, но модель не прогнала свои тесты</span><span class="sep">·</span>
    <span>❌ не решено</span><span class="sep">·</span>
    <span>⚠️ сбой инфраструктуры (не вина модели)</span><span class="sep">·</span>
    <span>⏳ идёт сейчас</span><span class="sep">·</span>
    <span>🛑 похоже на зависание (журнал молчит &gt; 10 мин)</span>
  </div>
  <div class="head-controls">
    <div class="tabs" id="tabs">
      <button type="button" class="tab" data-category="contest">Конкурсы<span class="tab-count" id="tab-contest-count"></span><span class="tab-dot hidden" id="tab-contest-dot">⏳</span></button>
      <button type="button" class="tab" data-category="test">Тесты моделей<span class="tab-count" id="tab-test-count"></span><span class="tab-dot hidden" id="tab-test-dot">⏳</span></button>
      <button type="button" class="tab" data-category="conclusions">Выводы<span class="tab-count" id="tab-conclusions-count"></span></button>
    </div>
    <div class="head-right">
      <label class="check">
        <input type="checkbox" id="show-service"> показать служебные
        <span id="service-count" class="muted"></span>
      </label>
      <span id="meta" class="muted small">Загрузка…</span>
    </div>
  </div>
</header>
<main>
  <div id="error" class="error hidden"></div>
  <section class="now-block" id="now-block">
    <h2 id="now-title">Идёт сейчас</h2>
    <div id="now-line" class="now-line">Загрузка…</div>
    <div id="running"></div>
  </section>
  <div id="groups"></div>
  <section id="conclusions" hidden></section>
  <aside id="detail-panel" class="hidden"><div class="detail-head"><span class="h"></span>
    <button class="x" type="button" data-close title="Закрыть (Esc)">×</button></div><div class="detail-body"></div></aside>
  <div id="old-benchmarks">
    <h2 class="old-title" id="old-title" hidden>Старый бенчмарк</h2>
    <p class="muted small old-note" id="old-note" hidden>Из родительской папки: без инструментов, без кубка, только ✅/❌/⚠️.</p>
    <div id="old-list"></div>
  </div>
</main>
<script>
const REFRESH_MS = 5000;
const TREE_KEY = "dashboard.tree.v2";
const CATEGORY_KEY = "dashboard.category.v1";
const SERVICE_KEY = "dashboard.showService.v1";

function esc(value) {
  if (value === null || value === undefined) return "";
  return String(value).replace(/[&<>"']/g, function (ch) {
    return {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[ch];
  });
}

function loadObject(key) {
  try {
    const raw = localStorage.getItem(key);
    const parsed = raw ? JSON.parse(raw) : {};
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
  } catch (err) {
    return {};
  }
}

let treeState = loadObject(TREE_KEY);
let selectedCategory = null;
try {
  const stored = localStorage.getItem(CATEGORY_KEY);
  if (stored === "contest" || stored === "test" || stored === "conclusions") selectedCategory = stored;
} catch (err) {}
let showService = false;
try { showService = localStorage.getItem(SERVICE_KEY) === "1"; } catch (err) {}
let lastState = null;
let firstRender = true;

function saveTree() {
  try { localStorage.setItem(TREE_KEY, JSON.stringify(treeState)); } catch (err) {}
}

function isOpen(key, defaultOpen) {
  if (Object.prototype.hasOwnProperty.call(treeState, key)) return !!treeState[key];
  return !!defaultOpen;
}

function setOpen(key, open) {
  treeState[key] = open;
  saveTree();
}

function pluralRu(number, one, few, many) {
  const value = Math.abs(Number(number) || 0);
  if (value % 10 === 1 && value % 100 !== 11) return one;
  if (value % 10 >= 2 && value % 10 <= 4 && (value % 100 < 12 || value % 100 > 14)) return few;
  return many;
}

function actionsRu(number) {
  if (number === null || number === undefined || number === "") return "—";
  const value = Number(number);
  if (!isFinite(value)) return "—";
  return value + " " + pluralRu(value, "действие", "действия", "действий");
}

function countsLine(counts) {
  const parts = [
    "✅ " + (counts.cw || 0),
    "🟡 " + (counts.code || 0),
    "❌ " + (counts.fail || 0),
    "⚠️ " + (counts.infra || 0)
  ];
  if (counts.running) parts.push("⏳ " + counts.running);
  if (counts.stalled) parts.push("🛑 " + counts.stalled);
  if (counts.skipped) parts.push("⏭ " + counts.skipped);
  if (counts.unknown) parts.push("? " + counts.unknown);
  return parts.join(" · ");
}

function oldCountsLine(counts) {
  const parts = [
    "✅ " + (counts.cw || 0),
    "❌ " + (counts.fail || 0)
  ];
  if (counts.infra) parts.push("⚠️ " + counts.infra);
  if (counts.unknown) parts.push("? " + counts.unknown);
  return parts.join(" · ");
}

function oldResult(c) {
  const parts = [
    "✅ " + (c.counts.cw || 0),
    "❌ " + (c.counts.fail || 0)
  ];
  if (c.counts.infra) parts.push("⚠️ " + c.counts.infra);
  if (c.counts.unknown) parts.push("? " + c.counts.unknown);
  return parts.join(" · ") + " из " + c.total;
}

function testResult(c) {
  const parts = [
    "✅ " + (c.counts.cw || 0),
    "🟡 " + (c.counts.code || 0),
    "❌ " + (c.counts.fail || 0)
  ];
  if (c.counts.infra) parts.push("⚠️ " + c.counts.infra);
  if (c.counts.running) parts.push("⏳ " + c.counts.running);
  if (c.counts.stalled) parts.push("🛑 " + c.counts.stalled);
  if (c.counts.skipped) parts.push("⏭ " + c.counts.skipped);
  if (c.counts.unknown) parts.push("? " + c.counts.unknown);
  return parts.join(" · ") + " из " + c.task_total;
}

function runningLabel(item) {
  if (!item.label || item.label === "(self)") return item.contest;
  return item.contest + " [" + item.label + "]";
}

function summaryLine(state) {
  const running = visibleRunning(state);
  if (!running.length) return "Ничего не идёт";
  const first = running[0];
  let text = (first.stalled ? "🛑 Похоже на зависание: Конкурс " : "Конкурс ") +
    first.contest +
    " · " + (first.short_model || first.model) +
    " · задача " + first.task +
    " · ход " + first.turn +
    " · " + (first.elapsed_ru || first.elapsed);
  if (running.length > 1) text += " · ещё " + (running.length - 1);
  return text;
}

function summaryTitle(state) {
  const running = visibleRunning(state);
  if (!running.length) return "Активных эпизодов нет";
  return running.map(function (r, index) {
    const mark = r.stalled ? "🛑" : "⏳";
    return (index + 1) + ". " + mark + " " + runningLabel(r) + " · " +
      r.model + " · " + r.task + " · " + (r.elapsed_ru || r.elapsed) +
      " · ход " + r.turn + " · запись " + (r.last_write_ago_ru || r.last_write_ago);
  }).join("\n");
}

function captureScroll() {
  const page = {x: window.scrollX, y: window.scrollY};
  const scrollers = {};
  document.querySelectorAll("[data-scroll-key]").forEach(function (el) {
    scrollers[el.getAttribute("data-scroll-key")] = {left: el.scrollLeft, top: el.scrollTop};
  });
  return {page: page, scrollers: scrollers};
}

function restoreScroll(saved) {
  if (!saved) return;
  window.scrollTo(saved.page.x, saved.page.y);
  document.querySelectorAll("[data-scroll-key]").forEach(function (el) {
    const pos = saved.scrollers[el.getAttribute("data-scroll-key")];
    if (pos) { el.scrollLeft = pos.left; el.scrollTop = pos.top; }
  });
}

function visibleGroups(state) {
  return state.groups.filter(function (c) { return showService || !c.service; });
}

function visibleRunning(state) {
  return state.running.filter(function (r) { return showService || !r.service; });
}

function groupsFor(state, category) {
  return visibleGroups(state).filter(function (c) { return c.category === category; });
}

function pickCategory(state) {
  if (selectedCategory === "contest" || selectedCategory === "test" || selectedCategory === "conclusions") return selectedCategory;
  const live = visibleGroups(state).filter(function (c) { return c.live; })[0];
  if (live) return live.category;
  const contest = groupsFor(state, "contest");
  return contest.length ? "contest" : "test";
}

function groupMeta(c) {
  const parts = [c.episodes_text];
  if (c.wall_display && c.wall_display !== "—") parts.push(c.wall_display);
  parts.push("обновлено " + c.updated_ago);
  return parts.join(" · ");
}

function groupExtras(c) {
  let html = "";
  if (c.queue_progress) {
    const phaseLabels = {starting: "подготовка", superlight: "SuperLight", "light-medium": "Light/Medium", "high-mmap": "High mmap", complete: "готово"};
    const progress = c.queue_progress;
    let text = phaseLabels[progress.phase] || progress.phase || "очередь";
    if (Number.isInteger(progress.done) && Number.isInteger(progress.total)) {
      text += " · " + progress.done + "/" + progress.total;
    }
    html += '<span class="live-note">' + esc(text) + '</span>';
  }
  if (c.live) {
    if (c.live_info) {
      const li = c.live_info;
      html += '<span class="live-note">' + (li.stalled ? "🛑" : "⏳") + " идёт: " +
        esc(li.short_model || li.model) + " · " + esc(li.task) + " · " + esc(li.elapsed) +
        (li.more ? " · ещё " + li.more : "") + "</span>";
    } else {
      html += '<span class="live-note">⏳ идёт</span>';
    }
  }
  if (!c.live && c.state_class === "running") {
    html += '<span class="incomplete-note" title="status.json говорит running, но свежих записей нет">нет активности</span>';
  }
  if (c.category === "contest") {
    if (c.winner_text) {
      html += '<span class="winner">' + esc(c.winner_text) + '</span>' +
        '<span class="provisional">предварительно</span>';
    }
    if (c.stopped) {
      html += '<span class="incomplete-note">не завершён</span>';
    }
    if (!c.winner_text && !c.live) {
      html += '<span class="result">' + esc(c.result_text) + '</span>';
    }
  } else {
    html += '<span class="test-result">' + esc(testResult(c)) + '</span>';
    if (c.verdict_text) {
      html += '<span class="verdict" title="Вердикт испытания из summary.md">Вердикт: ' + esc(c.verdict_text) + '</span>';
    }
  }
  return html;
}

function setText(el, text) {
  if (!el) return;
  const value = text === null || text === undefined ? "" : String(text);
  if (el.textContent !== value) el.textContent = value;
}

function setHtml(el, html) {
  if (!el) return;
  const value = html || "";
  if (el.innerHTML !== value) el.innerHTML = value;
}

function stageData(task, cell) {
  if (!cell) return null;
  const kind = cell.stalled ? "stall" : (cell.kind || "unknown");
  const time = cell.kind === "run" ? (cell.elapsed || "—") : (cell.wall_display || "—");
  let actions = "—";
  let result = "";
  if (cell.kind === "run") {
    actions = "ход " + (cell.turn === undefined || cell.turn === null ? "—" : cell.turn);
    result = "последнее действие: " + (cell.last_action_ru || cell.last_action || "—");
  } else if (cell.kind === "skipped") {
    result = cell.reason || "не успел";
  } else if (cell.kind === "unknown") {
    result = cell.title || "нет данных";
  } else {
    if (cell.tool_calls !== null && cell.tool_calls !== undefined) {
      actions = actionsRu(cell.tool_calls);
    }
    result = cell.old_benchmark
      ? (cell.reason || cell.status_text || "—")
      : "итог: " + (cell.reason || cell.status_text || "—");
  }
  return {
    kind: kind,
    time: time,
    actions: actions,
    result: result,
    icon: cell.icon || "?",
    title: cell.title || ""
  };
}

function shortMinutes(value) {
  const text = String(value || "").trim();
  if (!text || text === "—") return "";
  return text
    .replace(/\s*мин(?:ут)?/gi, "м")
    .replace(/\s*ч(?:ас)?/gi, "ч")
    .replace(/\s*с(?:ек)?/gi, "с")
    .replace(/\s+/g, "");
}

function squareInfo(task, cell) {
  if (!cell) {
    return {
      cls: "empty",
      id: task,
      min: "",
      title: task + " · ещё не играли"
    };
  }
  const data = stageData(task, cell);
  if (!data) return null;
  const kindLabels = {
    cw: "решено",
    code: "код верный, тесты не прогнаны",
    fail: "не решено",
    infra: "сбой инфраструктуры",
    run: cell.stalled ? "похоже на зависание" : "идёт",
    stall: "похоже на зависание",
    skipped: "не успел",
    pending: "запланировано",
    unknown: "нет данных"
  };
  const cls = data.kind !== "unknown" && Object.prototype.hasOwnProperty.call(kindLabels, data.kind)
    ? data.kind
    : "empty";
  const lines = [task + " · " + data.icon + " " + (kindLabels[data.kind] || "нет данных")];
  if (data.time && data.time !== "—") lines.push("время: " + data.time);
  if (data.actions && data.actions !== "—") lines.push("действия: " + data.actions);
  if (data.result) lines.push(data.result);
  if (cell.title) lines.push(cell.title);
  return {
    cls: cls,
    id: task,
    min: cell.kind === "run" ? shortMinutes(data.time) : "",
    title: lines.join("\n")
  };
}

function cellSquare(task, cell, clickable) {
  const info = squareInfo(task, cell);
  if (!info) return "";
  const open = clickable && cell && cell.transcript;
  return '<span class="cell-square ' + esc(info.cls) + (open ? ' clickable' : '') + '"' +
    (open ? ' data-transcript="' + esc(cell.transcript) + '" data-task="' + esc(task) + '" data-live="' + (cell.kind === "run" ? "1" : "") + '" role="button" tabindex="0"' : '') +
    ' title="' + esc(info.title + (open ? "\nНажмите: задание и ответы модели" : "")) + '">' +
    '<span class="sq-id">' + esc(info.id) + '</span>' +
    (info.min ? '<span class="sq-min">' + esc(info.min) + '</span>' : '') +
    '</span>';
}

function liveItemFor(row) {
  // The running episode that belongs to this model row (same group, model and config), if any.
  const items = (runningState && runningState.running) || [];
  const cfg = String(row.config || "").split("/").pop().replace(/[^A-Za-z0-9_-]+/g, "-");
  return items.find(function (it) {
    if (!row.id || !it.group || row.id.indexOf(it.group + "|") !== 0) return false;
    if (it.model !== row.model) return false;
    return !(cfg && it.contest_raw && it.contest_raw.indexOf("-" + cfg + "-") < 0);
  }) || null;
}

function liveDotHtml(row) {
  return liveItemFor(row) ? '<span class="live-dot" title="Сейчас идёт прогон этой модели и конфига"></span>' : '';
}

function cellbarHtml(row, noClick) {
  const cells = Object.assign({}, row.cells || {});
  const live = liveItemFor(row);
  if (live && (!cells[live.task] || cells[live.task].kind === "run")) {
    cells[live.task] = Object.assign({}, cells[live.task] || {}, {
      kind: "run", stalled: !!live.stalled, elapsed: live.elapsed_ru || live.elapsed,
      transcript: (cells[live.task] && cells[live.task].transcript) || live.transcript,
      title: "идёт · ход " + live.turn + " · " + (live.last_action_ru || live.last_action || "")
    });
  }
  const tasks = (row.tasks || []).slice();
  if (live && tasks.indexOf(live.task) < 0) tasks.push(live.task);
  return tasks.map(function (task) {
    return cellSquare(task, cells[task], !noClick);
  }).join("");
}

function stageRow(task, cell) {
  const data = stageData(task, cell);
  if (!data) return "";
  return '<tr class="stage ' + esc(data.kind) + '" data-task="' + esc(task) + '" title="' + esc(data.title) + '">' +
    '<td class="stage-task mono">' + esc(task) + '</td>' +
    '<td class="stage-icon">' + esc(data.icon) + '</td>' +
    '<td class="stage-time mono">' + esc(data.time) + '</td>' +
    '<td class="stage-actions">' + esc(data.actions) + '</td>' +
    '<td class="stage-result">' + esc(data.result) + '</td>' +
    '</tr>';
}

function updateStageRow(tr, task, cell) {
  const data = stageData(task, cell);
  if (!data) return;
  tr.className = "stage " + data.kind;
  tr.setAttribute("data-task", task);
  tr.title = data.title;
  setText(tr.querySelector(".stage-task"), task);
  setText(tr.querySelector(".stage-icon"), data.icon);
  setText(tr.querySelector(".stage-time"), data.time);
  setText(tr.querySelector(".stage-actions"), data.actions);
  setText(tr.querySelector(".stage-result"), data.result);
}

function modelBlock(c, row) {
  const key = "m:" + row.id;
  const open = isOpen(key, false);
  let badges = "";
  for (const badge of row.badges || []) {
    badges += '<span class="row-badge">' + esc(badge) + '</span>';
  }
  let html = '<div class="model-block" data-model="' + esc(row.id) + '">' +
    '<button type="button" class="model-head" data-model-toggle="' + esc(row.id) + '" aria-expanded="' + (open ? "true" : "false") + '" title="' + esc(row.title) + '">' +
      '<span class="mlead">' +
      '<span class="chev">' + (open ? "▾" : "▸") + '</span>' +
      '<span data-dyn="livedot">' + liveDotHtml(row) + '</span>' +
      '<span class="mname mono">' + esc(row.model) + '</span>' +
      (row.show_config && row.config_short ? '<span class="mconfig mono">· ' + esc(row.config_short) + '</span>' : '') +
      (!row.show_config && row.label && row.label !== row.model ? '<span class="rowlabel-mini">[' + esc(row.label) + ']</span>' : '') +
      badges +
      (row.winner ? '<span class="winner-mark" title="Победитель; предварительно">🏆</span><span class="provisional">предварительно</span>' : '') +
      (row.incomplete ? '<span class="incomplete-note" title="Меньше задач, чем у других: не может победить">неполный</span>' : '') +
      '</span>' +
      '<span class="cellbar" data-dyn="cellbar">' + cellbarHtml(row) + '</span>' +
      '<span class="mright">' +
        '<span class="mcounts">' + esc(row.old_benchmark ? oldCountsLine(row.counts) : countsLine(row.counts)) + '</span>' +
        '<span class="mtime mono">' + esc(row.wall_display) + '</span>' +
      '</span>' +
    '</button>' +
    '<div class="stage-wrap"' + (open ? "" : " hidden") + '>';
  if (!row.tasks.length) {
    html += '<p class="muted small empty-note">Задач нет.</p>';
  } else {
    html += '<table class="stages"><tbody>';
    for (const task of row.tasks) html += stageRow(task, row.cells[task]);
    html += '</tbody></table>';
  }
  return html + '</div></div>';
}

function groupCardHtml(c) {
  const key = "g:" + c.name;
  const open = isOpen(key, c.live);
  const typeLabel = c.category === "contest" ? "Конкурс" : "Тест модели";
  let html = '<section class="group-card' + (c.live ? " live" : "") + '" data-group="' + esc(c.name) + '">' +
    '<button type="button" class="group-head" data-group-toggle="' + esc(c.name) + '" aria-expanded="' + (open ? "true" : "false") + '" title="' + esc(c.title) + '">' +
      '<span class="chev">' + (open ? "▾" : "▸") + '</span>' +
      '<span class="gname">' + esc(c.name) + '</span>' +
      '<span class="type-badge">' + typeLabel + '</span>' +
      '<span class="level-badge">' + esc(c.level) + '</span>' +
      '<span class="badge ' + esc(c.state_class) + '">' + esc(c.state_ru) + '</span>' +
      '<span class="g-extras" data-dyn="g-extras">' + groupExtras(c) + '</span>' +
      '<span class="gmeta">' + esc(groupMeta(c)) + '</span>' +
      (c.service ? '<span class="service-badge">служебный</span>' : '') +
    '</button>' +
    '<div class="group-body"' + (open ? "" : " hidden") + '>';
  const sections = sectionsOf(c);
  if (!c.rows.length) {
    html += '<p class="muted small empty-note">Эпизоды не найдены.</p>';
  } else {
    for (const section of sections) {
      if (!section.rows || !section.rows.length) continue;
      if (section.title) {
        html += '<div class="section-head">' +
          '<span class="section-title">' + esc(section.title) + '</span>' +
          (section.winner_text ? '<span class="section-winner">' + esc(section.winner_text) + '</span>' +
            '<span class="provisional">предварительно</span>' : '') +
          (section.note ? '<span class="section-note">' + esc(section.note) + '</span>' : '') +
          '</div>';
      }
      for (const row of section.rows) html += modelBlock(c, row);
    }
  }
  return html + '</div></section>';
}

function oldCardHtml(test) {
  const key = "o:" + test.id;
  const open = isOpen(key, false);
  let html = '<section class="group-card old-card" data-old="' + esc(test.id) + '">' +
    '<button type="button" class="group-head" data-old-toggle="' + esc(test.id) + '" aria-expanded="' + (open ? "true" : "false") + '" title="' + esc(test.title) + '">' +
      '<span class="chev">' + (open ? "▾" : "▸") + '</span>' +
      '<span class="gname">' + esc(test.name) + '</span>' +
      '<span class="type-badge">Тест модели</span>' +
      '<span class="level-badge">' + esc(test.level) + '</span>' +
      '<span class="old-badge">старый бенчмарк</span>' +
      '<span class="test-result">' + esc(oldResult(test)) + '</span>' +
      '<span class="gmeta">' + esc(test.models + " " + pluralRu(test.models, "модель", "модели", "моделей") + " · " + test.wall_display + " · обновлено " + test.updated_ago) + '</span>' +
    '</button>' +
    '<div class="group-body"' + (open ? "" : " hidden") + '>';
  if (!test.rows.length) {
    html += '<p class="muted small empty-note">Запросов нет.</p>';
  } else {
    for (const row of test.rows) html += modelBlock(test, row);
  }
  return html + '</div></section>';
}

function makeNode(html) {
  const template = document.createElement("template");
  template.innerHTML = html.trim();
  return template.content.firstElementChild;
}

function groupCard(c) {
  return makeNode(groupCardHtml(c));
}

function oldCard(test) {
  return makeNode(oldCardHtml(test));
}

function sectionsOf(c) {
  return c.sections && c.sections.length
    ? c.sections
    : [{title: "", rows: c.rows || [], winner_text: "", note: ""}];
}

function allRowsOf(c) {
  const rows = [];
  for (const section of sectionsOf(c)) {
    for (const row of section.rows || []) rows.push(row);
  }
  return rows;
}

function rowSignature(row) {
  return {
    id: row.id,
    model: row.model,
    config_short: row.config_short || "",
    show_config: !!row.show_config,
    label: row.label || "",
    badges: row.badges || [],
    winner: !!row.winner,
    incomplete: !!row.incomplete,
    old: !!row.old_benchmark,
    tasks: row.tasks || []
  };
}

function groupSignature(c) {
  return JSON.stringify({
    name: c.name,
    level: c.level,
    category: c.category,
    service: !!c.service,
    sections: sectionsOf(c).map(function (section) {
      return {
        title: section.title || "",
        winner_text: section.winner_text || "",
        note: section.note || "",
        rows: (section.rows || []).map(rowSignature)
      };
    })
  });
}

function oldSignature(test) {
  return JSON.stringify({
    id: test.id,
    name: test.name,
    level: test.level,
    models: test.models,
    rows: (test.rows || []).map(rowSignature)
  });
}

function updateModelBlock(block, row, oldMode) {
  setText(block.querySelector(".mcounts"),
    (oldMode || row.old_benchmark) ? oldCountsLine(row.counts) : countsLine(row.counts));
  setText(block.querySelector(".mtime"), row.wall_display);
  setHtml(block.querySelector('[data-dyn="cellbar"]'), cellbarHtml(row));
  setHtml(block.querySelector('[data-dyn="livedot"]'), liveDotHtml(row));
  const head = block.querySelector(".model-head");
  if (head && row.title) head.title = row.title;
  const stages = {};
  block.querySelectorAll("tr.stage").forEach(function (tr) {
    stages[tr.getAttribute("data-task")] = tr;
  });
  const cells = row.cells || {};
  for (const task of row.tasks || []) {
    const tr = stages[task];
    if (tr) updateStageRow(tr, task, cells[task]);
  }
}

function updateGroupCard(card, c) {
  card.classList.toggle("live", !!c.live);
  const head = card.querySelector(".group-head");
  if (head) {
    if (c.title) head.title = c.title;
    const badge = head.querySelector(".badge");
    if (badge) {
      badge.className = "badge " + (c.state_class || "unknown");
      setText(badge, c.state_ru || "");
    }
    setText(head.querySelector(".type-badge"), c.category === "contest" ? "Конкурс" : "Тест модели");
    setHtml(head.querySelector('[data-dyn="g-extras"]'), groupExtras(c));
    setText(head.querySelector(".gmeta"), groupMeta(c));
  }
  const blocks = {};
  card.querySelectorAll(".model-block").forEach(function (block) {
    blocks[block.getAttribute("data-model")] = block;
  });
  for (const row of allRowsOf(c)) {
    const block = blocks[row.id];
    if (block) updateModelBlock(block, row, false);
  }
}

function updateOldCard(card, test) {
  const head = card.querySelector(".group-head");
  if (head && test.title) head.title = test.title;
  setText(card.querySelector(".test-result"), oldResult(test));
  setText(card.querySelector(".gmeta"),
    test.models + " " + pluralRu(test.models, "модель", "модели", "моделей") +
    " · " + test.wall_display + " · обновлено " + test.updated_ago);
  const blocks = {};
  card.querySelectorAll(".model-block").forEach(function (block) {
    blocks[block.getAttribute("data-model")] = block;
  });
  for (const row of test.rows || []) {
    const block = blocks[row.id];
    if (block) updateModelBlock(block, row, true);
  }
}

const interactionUntil = new WeakMap();
const hoveredCards = new WeakSet();

function cardOf(node) {
  return node && node.closest ? node.closest(".group-card") : null;
}

function markInteraction(node) {
  const card = cardOf(node);
  if (card) interactionUntil.set(card, Date.now() + 2500);
}

function isProtected(card) {
  if (!card) return false;
  if (hoveredCards.has(card)) return true;
  return Date.now() < (interactionUntil.get(card) || 0);
}

document.addEventListener("pointerover", function (event) {
  const card = cardOf(event.target);
  if (!card) return;
  hoveredCards.add(card);
  interactionUntil.set(card, Date.now() + 2500);
}, true);

document.addEventListener("pointerout", function (event) {
  const card = cardOf(event.target);
  if (!card) return;
  const to = event.relatedTarget;
  if (to && card.contains(to)) return;
  hoveredCards.delete(card);
}, true);

document.addEventListener("focusin", function (event) {
  markInteraction(event.target);
}, true);

function reconcile(container, items, attr, keyOf, build, signatureOf, update, emptyText) {
  const existing = new Map();
  Array.from(container.children).forEach(function (el) {
    if (el.hasAttribute(attr)) existing.set(el.getAttribute(attr), el);
  });
  const placeholder = container.querySelector(":scope > [data-empty]");
  if (!items.length) {
    existing.forEach(function (el) { el.remove(); });
    if (!placeholder) {
      const note = document.createElement("p");
      note.className = "muted empty-note";
      note.setAttribute("data-empty", "");
      note.textContent = emptyText || "Нет данных.";
      container.appendChild(note);
    } else {
      placeholder.textContent = emptyText || "Нет данных.";
    }
    return;
  }
  if (placeholder) placeholder.remove();

  let previous = null;
  for (const item of items) {
    const key = keyOf(item);
    const signature = signatureOf(item);
    let el = existing.get(key) || null;
    if (el) existing.delete(key);
    if (!el) {
      el = build(item);
      el.setAttribute(attr, key);
      el.setAttribute("data-sig", signature);
    } else if (el.getAttribute("data-sig") !== signature) {
      if (!isProtected(el)) {
        const fresh = build(item);
        fresh.setAttribute(attr, key);
        fresh.setAttribute("data-sig", signature);
        el.replaceWith(fresh);
        el = fresh;
      }
    } else if (!isProtected(el)) {
      update(el, item);
      el.setAttribute("data-sig", signature);
    }
    if (previous) {
      if (previous.nextSibling !== el) previous.after(el);
    } else if (container.firstChild !== el) {
      container.insertBefore(el, container.firstChild);
    }
    previous = el;
  }
  existing.forEach(function (el) { el.remove(); });
  Array.from(container.children).forEach(function (el) {
    if (!el.hasAttribute(attr)) el.remove();
  });
}

let runningState = null;

function runningCards(items) {
  if (!items.length) return '<p class="muted">Ничего не выполняется.</p>';
  let html = '';
  for (const item of items) {
    const contests = (runningState && runningState.contests) || [];
    const contest = contests.find(function (c) { return c.name === item.contest_raw; }) || null;
    const row = contest ? (contest.rows.find(function (r) { return r.model === item.model; }) || contest.rows[0]) : null;
    const bar = contest && row ? cellbarHtml({ tasks: contest.tasks || [], cells: row.cells || {} }, true) : '';
    const write = (item.last_write_ago_ru || item.last_write_ago || "—") +
      (item.stalled ? ' <span class="stall-badge">🛑 похоже на зависание</span>' : '');
    html += '<div class="run-card">' +
      '<div class="run-title"><b>' + esc(item.short_group || item.contest) + '</b> · <span class="mono">' +
        esc(item.model) + '</span> · задача <b>' + esc(item.task) + '</b></div>' +
      '<span class="cellbar big">' + bar + '</span>' +
      '<div class="run-meta">ход ' + esc(item.turn) + ' · прошло ' + esc(item.elapsed_ru || item.elapsed) +
        ' · последнее действие: ' + esc(item.last_action_ru || item.last_action || "—") +
        ' · последняя запись: ' + write +
        (item.last_tool_error ? ' · <span class="toolerror" title="' + esc(item.last_tool_error) + '">ошибка: ' +
          esc(item.last_tool_error.slice(0, 120)) + '</span>' : '') +
      '</div></div>';
  }
  return html;
}

let detailTimer = null;
let detailOpenKey = null;   // transcript shown in the panel; a second click on the same square closes it

function closeDetail() {
  document.getElementById("detail-panel").classList.add("hidden");
  if (detailTimer) { clearInterval(detailTimer); detailTimer = null; }
  detailOpenKey = null;
}

function renderDetail(d, task) {
  const verdict = !d.finished ? "⏳ идёт" : (d.workflow_pass ? "✅ решено и проверено" : (d.code_pass ? "🟡 код верный, без проверки" : "❌ не решено"));
  let html = '<h3>Задание</h3><pre>' + esc(d.task || "(TASK.md не найден)") + '</pre><h3>Ходы модели</h3>';
  if (d.turns_total > d.turns.length) html += '<p class="muted small">показаны последние ' + d.turns.length + ' из ' + d.turns_total + '</p>';
  for (const t of d.turns) {
    html += '<div class="detail-turn"><div class="t-head">Ход ' + esc(t.turn) + ' · <b>' + esc(t.action || "?") + '</b>' +
      (t.path ? ' <span class="mono">' + esc(t.path) + '</span>' : '') +
      (t.tokens ? ' <span class="muted small">· ' + esc(t.tokens) + ' ток.</span>' : '') + '</div>';
    if (t.thinking) html += '<details><summary>размышления (' + t.thinking.length + ' симв.)</summary><pre>' + esc(t.thinking) + '</pre></details>';
    if (t.code) html += '<pre>' + esc(t.code) + '</pre>';
    if (t.report) html += '<pre>' + esc(t.report) + '</pre>';
    if (t.raw) html += '<pre>' + esc(t.raw) + '</pre>';
    if (t.result) html += '<details' + (t.result_ok ? '' : ' open') + '><summary>ответ инструмента' + (t.result_ok ? '' : ' (ошибка)') +
      '</summary><pre class="' + (t.result_ok ? '' : 'res-bad') + '">' + esc(t.result) + '</pre></details>';
    html += '</div>';
  }
  if (d.waiting) html += '<div class="detail-wait">⏳ модель генерирует ход ' + esc(d.waiting.turn) + '… ' + esc(d.waiting.seconds) +
    ' с. Ответ появится целиком, когда ход закончится (запускатель получает ответы не по токенам).</div>';
  return {head: '<b>' + esc(task) + '</b><span>' + esc(verdict) + '</span><span class="muted small">ходов: ' + esc(d.turns_total) + '</span>', body: html};
}

function openDetail(transcript, task, live) {
  const panel = document.getElementById("detail-panel");
  const head = panel.querySelector(".detail-head .h");
  const body = panel.querySelector(".detail-body");
  head.innerHTML = '<b>' + esc(task) + '</b><span class="muted">загрузка…</span>';
  body.innerHTML = '';
  panel.classList.remove("hidden");
  if (detailTimer) { clearInterval(detailTimer); detailTimer = null; }
  let lastBody = null;
  const load = function () {
    fetch("/api/episode?t=" + encodeURIComponent(transcript), {cache: "no-store"})
      .then(function (r) { return r.json(); })
      .then(function (d) {
        if (d.error) throw new Error(d.error);
        const view = renderDetail(d, task);
        head.innerHTML = view.head;
        if (view.body !== lastBody) {
          const atBottom = body.scrollTop + body.clientHeight >= body.scrollHeight - 40;
          const openThinking = Array.from(body.querySelectorAll("details")).map(function (x) { return x.open; });
          body.innerHTML = view.body;
          body.querySelectorAll("details").forEach(function (x, i) { if (openThinking[i]) x.open = true; });
          if (live && (atBottom || lastBody === null)) body.scrollTop = body.scrollHeight;
          lastBody = view.body;
        }
        if (d.finished && detailTimer) { clearInterval(detailTimer); detailTimer = null; }
      })
      .catch(function (e) { body.innerHTML = '<p class="warn">Не удалось открыть эпизод: ' + esc(e.message) + '</p>'; });
  };
  load();
  if (live) detailTimer = setInterval(load, 3000);
}

document.addEventListener("click", function (e) {
  const sq = e.target.closest && e.target.closest(".cell-square.clickable");
  if (sq) {
    e.preventDefault(); e.stopPropagation();
    const key = sq.getAttribute("data-transcript");
    if (key === detailOpenKey) { closeDetail(); return; }
    openDetail(key, sq.getAttribute("data-task"), sq.getAttribute("data-live") === "1");
    detailOpenKey = key;
    return;
  }
  if (e.target.closest && e.target.closest("#detail-panel [data-close]")) closeDetail();
}, true);
document.addEventListener("keydown", function (e) { if (e.key === "Escape") closeDetail(); });

function runningTable(items) {
  if (!items.length) return '<p class="muted">Ничего не выполняется.</p>';
  let html = '<div class="card scroll" data-scroll-key="running"><table><thead><tr>' +
    '<th>Конкурс</th><th>Модель</th><th>Задача</th><th>Ход</th><th>Прошло</th>' +
    '<th>Последнее действие</th><th>Ошибка инструмента</th><th>Последняя запись</th>' +
    '</tr></thead><tbody>';
  for (const item of items) {
    const contest = item.label && item.label !== "(self)"
      ? esc(item.contest) + ' <span class="muted small">[' + esc(item.label) + ']</span>'
      : esc(item.contest);
    const lastWrite = item.stalled
      ? '<span class="warn stalled-text">' + esc(item.last_write_ago_ru || item.last_write_ago) + '</span>' +
        '<span class="stall-badge" title="Журнал молчит больше 10 мин. Долгая генерация может быть нормальной; это подсказка, а не приговор.">🛑 похоже на зависание</span>'
      : esc(item.last_write_ago_ru || item.last_write_ago);
    const actionTitle = "Последнее событие: " + (item.last_event || "—") +
      (item.last_action && item.last_action !== item.last_action_ru ? "; действие: " + item.last_action : "");
    const elapsedTitle = "Прошло " + item.elapsed_seconds + " с; старт: " + (item.started_at || item.started_ago_ru || item.started_ago);
    const writeTitle = "Последняя запись " + item.last_write_seconds + " с назад" + (item.last_write_at ? "; " + item.last_write_at : "");
    html += '<tr>' +
      '<td>' + contest + '</td>' +
      '<td class="mono">' + esc(item.model) + '</td>' +
      '<td>' + esc(item.task) + '</td>' +
      '<td class="mono">' + esc(item.turn) + '</td>' +
      '<td class="mono" title="' + esc(elapsedTitle) + '">' + esc(item.elapsed_ru || item.elapsed) + '</td>' +
      '<td class="action" title="' + esc(actionTitle) + '">' + esc(item.last_action_ru || item.last_action) + '</td>' +
      '<td class="toolerror" title="' + esc(item.last_tool_error) + '">' +
        (item.last_tool_error ? esc(item.last_tool_error.slice(0, 160)) : "—") + '</td>' +
      '<td class="mono" title="' + esc(writeTitle) + '">' + lastWrite + '</td>' +
      '</tr>';
  }
  return html + '</tbody></table></div>';
}

let runningSignature = null;

function runningItemsSignature(items) {
  return JSON.stringify(items.map(function (item) {
    return [
      item.contest, item.label, item.model, item.task, item.turn, item.elapsed,
      item.stalled, item.last_action, item.last_write_ago, item.last_tool_error
    ];
  }));
}

function updateRunning(items) {
  const box = document.getElementById("running");
  if (!items.length) {
    if (box.firstChild) box.replaceChildren();
    runningSignature = "[]";
    return;
  }
  if (box.matches(":hover")) return;
  const signature = runningItemsSignature(items);
  if (signature === runningSignature) return;
  box.innerHTML = runningCards(items);
  runningSignature = signature;
}

function updateNow(state) {
  const title = document.getElementById("now-title");
  const line = document.getElementById("now-line");
  const items = visibleRunning(state);
  runningState = state;
  setText(title, "Идёт сейчас");
  if (items.length) {
    setText(line, summaryLine(state));
    line.title = summaryTitle(state);
  } else if (state.last_run) {
    setText(line, state.last_run.text || "Последний прогон: нет данных");
    line.title = state.last_run.error
      ? state.last_run.text + "\n" + state.last_run.error
      : (state.last_run.text || "");
  } else {
    setText(line, "Последний прогон: нет данных");
    line.title = "";
  }
  updateRunning(items);
}

function updateTabs(state) {
  const contestGroups = groupsFor(state, "contest");
  const testGroups = groupsFor(state, "test");
  const oldTests = state.old_benchmarks || [];
  document.getElementById("tab-contest-count").textContent = " (" + contestGroups.length + ")";
  document.getElementById("tab-test-count").textContent = " (" + (testGroups.length + oldTests.length) + ")";
  const active = selectedCategory;
  document.querySelectorAll("#tabs .tab").forEach(function (btn) {
    btn.classList.toggle("active", btn.getAttribute("data-category") === active);
  });
  const contestDot = document.getElementById("tab-contest-dot");
  const testDot = document.getElementById("tab-test-dot");
  contestDot.classList.toggle("hidden", active === "contest" || !contestGroups.some(function (c) { return c.live; }));
  testDot.classList.toggle("hidden", active === "test" || !testGroups.some(function (c) { return c.live; }));
}

function updateServiceCount(state) {
  const hidden = state.groups.filter(function (c) { return c.service; }).length;
  document.getElementById("service-count").textContent = hidden ? "(" + hidden + ")" : "";
}

let conclusionsSig = null;
const SIZE_TITLE = {"XS": "SuperLight", "S": "Light", "M": "Medium", "L": "High", "C++": "C++"};

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined && text !== null) e.textContent = text;
  return e;
}

function fmtSec(v) {
  if (v === null || v === undefined) return "—";
  if (v < 1) return v.toFixed(2) + " с";
  if (v < 120) return Math.round(v) + " с";
  return (v / 60).toFixed(1) + " мин";
}

function pctClass(p) {   // 95+ green, 80-95 light green, 65-80 yellow, 50-65 orange, <50 red
  return p >= 95 ? "p95" : p >= 80 ? "p80" : p >= 65 ? "p65" : p >= 50 ? "p50" : "p0";
}

function renderConclusions(cc) {
  const sig = JSON.stringify(cc);
  if (sig === conclusionsSig) return;          // nothing changed: keep scroll and selection
  conclusionsSig = sig;
  const root = document.getElementById("conclusions");
  root.replaceChildren();

  root.appendChild(el("h2", null, "Распределение по ролям"));
  const roles = el("table", "cc-roles");
  for (const r of cc.roles || []) {
    const tr = el("tr");
    tr.appendChild(el("td", null, r.size));
    tr.appendChild(el("td", null, r.contest));
    const tags = el("td");
    (r.tags || []).forEach(function (t, i) { if (i) tags.appendChild(document.createTextNode(" ")); tags.appendChild(el("code", null, t)); });
    if (!(r.tags || []).length) tags.textContent = "—";
    tr.appendChild(tags);
    tr.appendChild(el("td", "muted", r.status));
    roles.appendChild(tr);
  }
  root.appendChild(roles);

  root.appendChild(el("h2", null, "Модели"));
  root.appendChild(el("p", "cc-note",
    "Успех: для XS/S/M — доля верного кода (✅ + 🟡), для L и C++ — доля решённого и проверенного самой моделью (✅). " +
    "Тесты — все эпизоды и запросы по всем конфигам модели (без сбоев стенда: HTTP 400/500). Скорость — среднее время " +
    "на задачу против среднего по моделям того же класса: меньше 1× быстрее одноклассников. Выводы правятся в round2/conclusions.json" +
    (cc.updated ? " (обновлено " + cc.updated + ")." : ".")));
  const grid = el("div", "cc-grid");
  for (const m of cc.models || []) {
    const card = el("div", "cc-card" + (m.verdict ? "" : " other"));
    const head = el("div", "cc-head");
    head.appendChild(el("span", "cc-name", m.name));
    (m.roles || []).forEach(function (r) { head.appendChild(el("span", "cc-chip", r)); });
    const st = m.status === "confirmed" ? ["confirmed", "подтверждено"] : m.status === "preliminary" ? ["preliminary", "предварительно"] : ["none", "без выводов"];
    head.appendChild(el("span", "cc-badge " + st[0], st[1]));
    card.appendChild(head);
    if (m.verdict) card.appendChild(el("p", "cc-verdict", m.verdict));
    [["Давать", m.give], ["Не давать", m.avoid], ["Слабые места", m.weak]].forEach(function (kv) {
      if (!kv[1] || kv[1] === "—") return;
      const p = el("p", "cc-kv"); p.appendChild(el("b", null, kv[0] + ": ")); p.appendChild(document.createTextNode(kv[1])); card.appendChild(p);
    });
    if ((m.classes || []).length) {
      const t = el("table", "cc-stats");
      const h = el("tr");
      ["Класс", "Тестов", "Успех", "Время/задача", "Против класса"].forEach(function (x) { h.appendChild(el("th", null, x)); });
      t.appendChild(h);
      for (const c of m.classes) {
        const tr = el("tr");
        tr.appendChild(el("td", null, c.size + " · " + (SIZE_TITLE[c.size] || "")));
        tr.appendChild(el("td", null, String(c.tests)));
        // XS/S/M: share of correct code (owner 2026-10-28); L/C++ and above: self-verified share
        const pct = Math.round(100 * (["XS", "S", "M"].includes(c.size) ? c.code_pass : c.full_pass) / c.tests);
        tr.appendChild(el("td", "cc-pct " + pctClass(pct), pct + "%"));
        tr.appendChild(el("td", null, fmtSec(c.avg_seconds)));
        const r = c.speed_ratio;
        tr.appendChild(el("td", r === null ? "" : r <= 0.9 ? "cc-fast" : r >= 1.1 ? "cc-slow" : "", r === null ? "—" : r.toFixed(2) + "×"));
        t.appendChild(tr);
      }
      card.appendChild(t);
    } else {
      card.appendChild(el("p", "cc-kv", "Тестов пока нет."));
    }
    grid.appendChild(card);
  }
  root.appendChild(grid);
}

function render(state, preserve) {
  const saved = preserve ? captureScroll() : null;
  lastState = state;
  selectedCategory = pickCategory(state);

  document.getElementById("meta").textContent =
    "обновлено " + state.generated_at + " · автообновление 5 с · только чтение";
  document.getElementById("ollama-line").textContent =
    state.ollama.summary || "Ollama: нет данных";

  updateNow(state);
  updateServiceCount(state);
  updateTabs(state);

  const showConclusions = selectedCategory === "conclusions";
  document.getElementById("groups").hidden = showConclusions;
  document.getElementById("conclusions").hidden = !showConclusions;
  const cc = state.conclusions || {models: []};
  document.getElementById("tab-conclusions-count").textContent = " (" + cc.models.filter(function (m) { return m.verdict; }).length + ")";
  if (showConclusions) renderConclusions(cc);
  const groups = showConclusions ? [] : groupsFor(state, selectedCategory);
  const emptyGroupText = selectedCategory === "contest"
    ? "Нет конкурсов за последние 24 часа."
    : "Нет тестов моделей за последние 24 часа.";
  reconcile(
    document.getElementById("groups"), groups, "data-group",
    function (c) { return c.name; },
    groupCard, groupSignature, updateGroupCard, emptyGroupText
  );

  const oldTests = state.old_benchmarks || [];
  const oldTitle = document.getElementById("old-title");
  const oldNote = document.getElementById("old-note");
  const oldList = document.getElementById("old-list");
  if (selectedCategory === "test" && oldTests.length) {
    oldTitle.hidden = false;
    oldNote.hidden = false;
    setText(oldTitle, "Старый бенчмарк (" + oldTests.length + ")");
    reconcile(
      oldList, oldTests, "data-old",
      function (test) { return test.id; },
      oldCard, oldSignature, updateOldCard, "Запросов нет."
    );
  } else {
    oldTitle.hidden = true;
    oldNote.hidden = true;
    if (oldList.firstChild) oldList.replaceChildren();
  }

  restoreScroll(saved);
  requestAnimationFrame(function () { restoreScroll(saved); });
}

async function tick() {
  const banner = document.getElementById("error");
  try {
    const response = await fetch("/api/state", {cache: "no-store"});
    if (!response.ok) throw new Error("HTTP " + response.status);
    const state = await response.json();
    if (state.error) throw new Error(state.error);
    render(state, !firstRender);
    firstRender = false;
    banner.classList.add("hidden");
    banner.textContent = "";
  } catch (err) {
    banner.textContent = "Обновление не удалось: " + err;
    banner.classList.remove("hidden");
  }
}

document.addEventListener("click", function (event) {
  markInteraction(event.target);
  const oldHead = event.target.closest(".old-card .group-head");
  if (oldHead) {
    const card = oldHead.closest(".old-card");
    const testId = oldHead.getAttribute("data-old-toggle");
    if (!card || !testId) return;
    const open = oldHead.getAttribute("aria-expanded") !== "true";
    oldHead.setAttribute("aria-expanded", open ? "true" : "false");
    const chev = oldHead.querySelector(".chev");
    if (chev) chev.textContent = open ? "▾" : "▸";
    const body = card.querySelector(".group-body");
    if (body) body.hidden = !open;
    setOpen("o:" + testId, open);
    return;
  }
  const groupHead = event.target.closest("#groups .group-head");
  if (groupHead) {
    const card = groupHead.closest(".group-card");
    const name = groupHead.getAttribute("data-group-toggle");
    if (!card || !name) return;
    const open = groupHead.getAttribute("aria-expanded") !== "true";
    groupHead.setAttribute("aria-expanded", open ? "true" : "false");
    const chev = groupHead.querySelector(".chev");
    if (chev) chev.textContent = open ? "▾" : "▸";
    const body = card.querySelector(".group-body");
    if (body) body.hidden = !open;
    setOpen("g:" + name, open);
    return;
  }
  const modelHead = event.target.closest(".model-head");
  if (modelHead) {
    const block = modelHead.closest(".model-block");
    const rowId = modelHead.getAttribute("data-model-toggle");
    if (!block || !rowId) return;
    const open = modelHead.getAttribute("aria-expanded") !== "true";
    modelHead.setAttribute("aria-expanded", open ? "true" : "false");
    const chev = modelHead.querySelector(".chev");
    if (chev) chev.textContent = open ? "▾" : "▸";
    const body = block.querySelector(".stage-wrap");
    if (body) body.hidden = !open;
    setOpen("m:" + rowId, open);
  }
});

document.getElementById("tabs").addEventListener("click", function (event) {
  const button = event.target.closest(".tab");
  if (!button) return;
  const category = button.getAttribute("data-category");
  if (category !== "contest" && category !== "test" && category !== "conclusions") return;
  selectedCategory = category;
  try { localStorage.setItem(CATEGORY_KEY, category); } catch (err) {}
  if (lastState) render(lastState, true);
});

const serviceCheckbox = document.getElementById("show-service");
serviceCheckbox.checked = showService;
serviceCheckbox.addEventListener("change", function () {
  showService = serviceCheckbox.checked;
  try { localStorage.setItem(SERVICE_KEY, showService ? "1" : "0"); } catch (err) {}
  if (lastState) render(lastState, true);
});

tick();
setInterval(tick, REFRESH_MS);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP server

class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class Handler(BaseHTTPRequestHandler):
    server_version = "ContestDashboard/1.0"

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
        path = urllib.parse.urlsplit(self.path).path
        if path in ("/", "/index.html"):
            self.respond(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/episode":
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            try:
                body = json.dumps(episode_detail((query.get("t") or [""])[0]), ensure_ascii=False).encode("utf-8")
                self.respond(200, body, "application/json; charset=utf-8")
            except Exception as exc:  # noqa: BLE001 - never crash, never leave runs/
                body = json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False).encode("utf-8")
                self.respond(400, body, "application/json; charset=utf-8")
            return
        if path == "/api/state":
            try:
                body = json.dumps(build_state(), ensure_ascii=False).encode("utf-8")
                self.respond(200, body, "application/json; charset=utf-8")
            except Exception as exc:  # noqa: BLE001 - always return JSON
                body = json.dumps(
                    {
                        "generated_at": fmt_dt(time.time()),
                        "generated_ts": time.time(),
                        "error": f"{type(exc).__name__}: {exc}",
                        "contests": [],
                        "running": [],
                        "ollama": {"ok": False, "error": "state build failed", "models": []},
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                self.respond(500, body, "application/json; charset=utf-8")
            return
        if path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        self.respond(404, b"not found\n", "text/plain; charset=utf-8")

    def respond(self, code: int, body: bytes, content_type: str):
        try:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def log_message(self, format, *args):  # noqa: A002 - base API
        # Keep the console quiet; the page itself reports refresh errors.
        pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Read-only live model-contest dashboard")
    parser.add_argument("--port", type=int, default=8765, help="loopback port (default 8765)")
    parser.add_argument("--dump", action="store_true", help="print /api/state JSON once and exit")
    args = parser.parse_args(argv)

    if args.dump:
        payload = json.dumps(build_state(), ensure_ascii=False, indent=2) + "\n"
        try:
            sys.stdout.buffer.write(payload.encode("utf-8"))
            sys.stdout.buffer.flush()
        except AttributeError:
            sys.stdout.write(payload)
        return 0

    server = DashboardServer(("127.0.0.1", args.port), Handler)
    print(f"Dashboard: http://127.0.0.1:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
