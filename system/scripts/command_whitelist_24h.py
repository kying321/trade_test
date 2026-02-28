#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import shlex
from typing import Any


DEFAULT_INPUT_GLOBS = [
    "output/logs/command_exec.ndjson",
    "output/logs/command_exec_*.ndjson",
    "output/logs/commands.ndjson",
    "output/logs/cmd_*.ndjson",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build 24h executable command whitelist with return-code samples")
    p.add_argument("--root", default=None, help="System root path (default: infer from script path)")
    p.add_argument("--window-hours", default="24", help="Window size in hours")
    p.add_argument("--max-samples-per-command", default="5")
    p.add_argument("--max-commands", default="50")
    p.add_argument(
        "--input",
        action="append",
        default=[],
        help="Input file/glob relative to --root; can be provided multiple times",
    )
    p.add_argument(
        "--include-tests-log",
        action="store_true",
        help="Also parse output/logs/tests_*.json as fallback command samples",
    )
    p.add_argument("--strict", action="store_true", help="Return non-zero when no events in the window")
    return p.parse_args()


def _now_local() -> datetime:
    return datetime.now().astimezone()


def _parse_timestamp(value: Any, *, local_tz: timezone | Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        sec = float(value)
        if sec > 1e12:
            sec = sec / 1000.0
        try:
            return datetime.fromtimestamp(sec, tz=timezone.utc).astimezone(local_tz)
        except Exception:
            return None
    s = str(value).strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=local_tz)
    return dt.astimezone(local_tz)


def _basename(token: str) -> str:
    return Path(str(token)).name


def _first_non_option(tokens: list[str], start: int = 0) -> str:
    for tok in tokens[start:]:
        if not str(tok).startswith("-"):
            return str(tok)
    return ""


def _normalize_command(command: str) -> tuple[str, str]:
    raw = str(command or "").strip()
    if not raw:
        return "unknown", ""
    try:
        tokens = shlex.split(raw)
    except Exception:
        tokens = raw.split()
    if not tokens:
        return "unknown", raw

    first = _basename(tokens[0]).lower()

    if first == "lie":
        sub = _first_non_option(tokens, 1)
        if sub:
            return f"lie {sub}", raw
        return "lie", raw

    if first in {"python", "python3", "python3.11", "python3.12"}:
        if "-m" in tokens:
            midx = tokens.index("-m")
            if midx + 1 < len(tokens):
                module = tokens[midx + 1]
                if module == "lie_engine.cli":
                    sub = _first_non_option(tokens, midx + 2)
                    return (f"lie {sub}" if sub else "lie"), raw
                sub = _first_non_option(tokens, midx + 2)
                return (f"python -m {module} {sub}".strip() if sub else f"python -m {module}"), raw
        script = _first_non_option(tokens, 1)
        if script:
            return f"python {_basename(script)}", raw
        return "python", raw

    if first in {"bash", "sh", "zsh"}:
        script = _first_non_option(tokens, 1)
        if script:
            return f"{first} {_basename(script)}", raw
        return first, raw

    sub = _first_non_option(tokens, 1)
    return (f"{first} {sub}" if sub else first), raw


def _to_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _extract_record(obj: dict[str, Any], *, fallback_ts: datetime | None, source_file: Path, local_tz: Any) -> dict[str, Any] | None:
    ts = None
    for key in ("timestamp", "ts", "time", "started_at", "created_at", "at"):
        if key in obj:
            ts = _parse_timestamp(obj.get(key), local_tz=local_tz)
            if ts is not None:
                break
    if ts is None:
        ts = fallback_ts
    if ts is None:
        return None

    command = ""
    if isinstance(obj.get("command"), str):
        command = str(obj.get("command", ""))
    elif isinstance(obj.get("cmd"), str):
        command = str(obj.get("cmd", ""))
    elif isinstance(obj.get("argv"), list):
        command = " ".join(str(x) for x in obj.get("argv", []) if str(x))

    rc_value = obj.get("returncode", obj.get("rc", obj.get("exit_code", None)))
    if rc_value is None and isinstance(obj.get("ok"), bool):
        rc_value = 0 if bool(obj.get("ok")) else 1
    returncode = _to_int(rc_value) if rc_value is not None else 0

    duration_ms = None
    if "duration_ms" in obj:
        duration_ms = _to_int(obj.get("duration_ms"))
    elif "duration_sec" in obj:
        duration_ms = int(round(_to_float(obj.get("duration_sec"), 0.0) * 1000.0))

    source = str(obj.get("source", "")).strip() or source_file.name
    key, command_example = _normalize_command(command)
    return {
        "timestamp": ts,
        "timestamp_iso": ts.isoformat(),
        "command_key": key,
        "command_example": command_example,
        "returncode": int(returncode),
        "success": bool(int(returncode) == 0),
        "source": source,
        "duration_ms": duration_ms,
        "source_file": str(source_file),
    }


def _load_from_json(path: Path, *, local_tz: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return out

    fallback_ts = _parse_timestamp(path.stat().st_mtime, local_tz=local_tz)

    if isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                continue
            rec = _extract_record(row, fallback_ts=fallback_ts, source_file=path, local_tz=local_tz)
            if rec is not None:
                out.append(rec)
    elif isinstance(payload, dict):
        rec = _extract_record(payload, fallback_ts=fallback_ts, source_file=path, local_tz=local_tz)
        if rec is not None:
            out.append(rec)
    return out


def _load_from_ndjson(path: Path, *, local_tz: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    fallback_ts = _parse_timestamp(path.stat().st_mtime, local_tz=local_tz)
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        rec = _extract_record(obj, fallback_ts=fallback_ts, source_file=path, local_tz=local_tz)
        if rec is not None:
            out.append(rec)
    return out


def _load_tests_fallback(path: Path, *, local_tz: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return out
    if not isinstance(payload, dict) or "returncode" not in payload:
        return out

    fallback_ts = _parse_timestamp(path.stat().st_mtime, local_tz=local_tz)
    if fallback_ts is None:
        return out
    mode = str(payload.get("mode", "")).strip().lower()
    if mode == "fast":
        cmd = "lie test-all --fast"
    elif mode == "full":
        cmd = "lie test-all"
    else:
        cmd = "lie test-all"
    rec = _extract_record(
        {
            "timestamp": fallback_ts.isoformat(),
            "command": cmd,
            "returncode": payload.get("returncode", 1),
            "source": "tests_log",
            "duration_ms": None,
        },
        fallback_ts=fallback_ts,
        source_file=path,
        local_tz=local_tz,
    )
    if rec is not None:
        out.append(rec)
    return out


def _build_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Command Whitelist 24h | {payload.get('as_of', '')}")
    lines.append("")
    lines.append(f"- window: `{payload.get('window_start', '')}` -> `{payload.get('window_end', '')}` ({payload.get('window_hours', 24)}h)")
    lines.append(f"- status: `{payload.get('status', '')}`")
    totals = payload.get("totals", {}) if isinstance(payload.get("totals", {}), dict) else {}
    lines.append(f"- total_events: `{totals.get('events', 0)}`")
    lines.append(f"- unique_commands: `{totals.get('commands', 0)}`")
    lines.append(f"- overall_success_rate: `{float(totals.get('success_rate', 0.0)):.4f}`")
    lines.append("")
    lines.append("## Whitelist")
    lines.append("")
    lines.append("| command | total | success | fail | success_rate |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in payload.get("whitelist", []):
        if not isinstance(row, dict):
            continue
        lines.append(
            "| `{}` | {} | {} | {} | {:.4f} |".format(
                str(row.get("command", "")),
                int(row.get("total_runs", 0)),
                int(row.get("success_runs", 0)),
                int(row.get("fail_runs", 0)),
                float(row.get("success_rate", 0.0)),
            )
        )
    lines.append("")
    lines.append("## Samples")
    lines.append("")
    for row in payload.get("whitelist", []):
        if not isinstance(row, dict):
            continue
        lines.append(f"### {row.get('command', '')}")
        for sample in row.get("samples", []):
            if not isinstance(sample, dict):
                continue
            lines.append(
                "- `{}` rc=`{}` source=`{}` duration_ms=`{}`".format(
                    str(sample.get("timestamp", "")),
                    int(sample.get("returncode", 0)),
                    str(sample.get("source", "")),
                    "N/A" if sample.get("duration_ms") is None else int(sample.get("duration_ms", 0)),
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = _parse_args()
    root = Path(args.root).expanduser().resolve() if args.root else Path(__file__).resolve().parents[1]
    logs_dir = root / "output" / "logs"
    review_dir = root / "output" / "review"
    logs_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    local_now = _now_local()
    local_tz = local_now.tzinfo
    window_hours = max(1, int(args.window_hours))
    window_end = local_now
    window_start = window_end - timedelta(hours=window_hours)

    input_globs = [x for x in args.input if str(x).strip()] or list(DEFAULT_INPUT_GLOBS)
    input_files: list[Path] = []
    for pattern in input_globs:
        input_files.extend(sorted(root.glob(pattern)))

    records: list[dict[str, Any]] = []
    loaded_inputs: list[str] = []
    for path in sorted({p.resolve() for p in input_files if p.exists()}):
        loaded_inputs.append(str(path))
        if path.suffix == ".ndjson":
            records.extend(_load_from_ndjson(path, local_tz=local_tz))
        elif path.suffix == ".json":
            records.extend(_load_from_json(path, local_tz=local_tz))

    tests_loaded: list[str] = []
    if bool(args.include_tests_log):
        for path in sorted(logs_dir.glob("tests_*.json")):
            tests_loaded.append(str(path.resolve()))
            records.extend(_load_tests_fallback(path, local_tz=local_tz))

    records = [
        r
        for r in records
        if isinstance(r.get("timestamp"), datetime)
        and window_start <= r["timestamp"] <= window_end
    ]
    records.sort(key=lambda x: (str(x.get("command_key", "")), str(x.get("timestamp_iso", ""))))

    grouped: dict[str, dict[str, Any]] = {}
    for rec in records:
        key = str(rec.get("command_key", "unknown"))
        group = grouped.setdefault(
            key,
            {
                "command": key,
                "command_example": str(rec.get("command_example", "")),
                "total_runs": 0,
                "success_runs": 0,
                "fail_runs": 0,
                "samples": [],
            },
        )
        group["total_runs"] += 1
        if bool(rec.get("success", False)):
            group["success_runs"] += 1
        else:
            group["fail_runs"] += 1

        if len(group["samples"]) < max(1, int(args.max_samples_per_command)):
            group["samples"].append(
                {
                    "timestamp": str(rec.get("timestamp_iso", "")),
                    "returncode": int(rec.get("returncode", 0)),
                    "source": str(rec.get("source", "")),
                    "duration_ms": rec.get("duration_ms"),
                }
            )

    whitelist = list(grouped.values())
    for row in whitelist:
        total = int(row.get("total_runs", 0))
        success = int(row.get("success_runs", 0))
        row["success_rate"] = float(success / total) if total > 0 else 0.0

    whitelist.sort(key=lambda x: (-int(x.get("total_runs", 0)), str(x.get("command", ""))))
    max_commands = max(1, int(args.max_commands))
    whitelist = whitelist[:max_commands]

    total_events = sum(int(x.get("total_runs", 0)) for x in whitelist)
    total_success = sum(int(x.get("success_runs", 0)) for x in whitelist)
    overall_success_rate = float(total_success / total_events) if total_events > 0 else 0.0

    if total_events == 0:
        status = "NO_DATA"
    elif overall_success_rate >= 0.95:
        status = "PASSED"
    elif overall_success_rate >= 0.80:
        status = "WARN"
    else:
        status = "FAILED"

    as_of = _now_local()
    stamp = as_of.strftime("%Y%m%d_%H%M%S")
    day = date.today().isoformat()
    json_path = review_dir / f"{day}_command_whitelist_24h_{stamp}.json"
    md_path = review_dir / f"{day}_command_whitelist_24h_{stamp}.md"

    payload = {
        "as_of": as_of.isoformat(),
        "status": status,
        "window_hours": window_hours,
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "inputs": loaded_inputs,
        "tests_log_inputs": tests_loaded,
        "totals": {
            "events": total_events,
            "commands": len(whitelist),
            "success_runs": total_success,
            "fail_runs": total_events - total_success,
            "success_rate": overall_success_rate,
        },
        "whitelist": whitelist,
    }

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_build_markdown(payload), encoding="utf-8")

    out = {
        "status": status,
        "review_json": str(json_path),
        "review_md": str(md_path),
        "events": total_events,
        "commands": len(whitelist),
        "success_rate": overall_success_rate,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))

    if bool(args.strict) and total_events == 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
