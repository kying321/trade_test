#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = SYSTEM_ROOT / "output"
DEFAULT_REVIEW_DIR = DEFAULT_OUTPUT_ROOT / "review"
DEFAULT_CONFIG_PATH = SYSTEM_ROOT / "config.yaml"
CLI_PATH = SYSTEM_ROOT / "src" / "lie_engine" / "cli.py"
ENGINE_PATH = SYSTEM_ROOT / "src" / "lie_engine" / "engine.py"


def parse_stamp(raw: str) -> dt.datetime:
    parsed = dt.datetime.strptime(str(raw).strip(), "%Y%m%dT%H%M%SZ")
    return parsed.replace(tzinfo=dt.timezone.utc)


def fmt_utc(value: dt.datetime | None) -> str:
    if value is None:
        return ""
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def text(value: Any) -> str:
    return str(value or "").strip()


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def latest_artifact(review_dir: Path, pattern: str) -> Path:
    candidates = sorted(review_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"missing_required_artifact:{pattern}")
    return candidates[-1]


def read_scalar_from_yaml(path: Path, key: str) -> str:
    pattern = re.compile(rf"^\s*{re.escape(key)}:\s*(.+?)\s*$", re.MULTILINE)
    text_body = path.read_text(encoding="utf-8")
    match = pattern.search(text_body)
    return match.group(1).strip() if match else ""


def build_reference_command(symbol: str) -> str:
    return (
        f"PYTHONPATH={SYSTEM_ROOT / 'src'} "
        f"python -m lie_engine.cli --config {DEFAULT_CONFIG_PATH} "
        f"micro-capture --date <YYYY-MM-DD> --symbols {symbol}"
    )


def build_blockers(config_path: Path) -> list[dict[str, Any]]:
    micro_interval = read_scalar_from_yaml(config_path, "micro_capture_daemon_interval_minutes")
    on_micro_capture = read_scalar_from_yaml(config_path, "binance_live_takeover_on_micro_capture")
    return [
        {
            "blocker_id": "historical_backfill_not_supported",
            "status": "active",
            "why": (
                "现有 CLI 只支持 `micro-capture --date YYYY-MM-DD --symbols ...`，"
                "engine.run_micro_capture 实际用的是当前 UTC run_ts 写出 artifact，"
                "没有历史时刻回放参数，所以不能补过去某个 entry 窗口的真实 micro snapshot。"
            ),
            "source_files": [str(CLI_PATH), str(ENGINE_PATH)],
        },
        {
            "blocker_id": "micro_capture_live_adjacent_hook",
            "status": "active",
            "why": (
                "engine.run_micro_capture 内部会调用 `_run_binance_live_takeover(trigger='micro_capture')`，"
                f"且 config 里 `binance_live_takeover_on_micro_capture={on_micro_capture or 'true?'}`；"
                "因此当前 workspace 的 micro-capture 不能视作纯 research-only 可随手执行路径。"
            ),
            "source_files": [str(ENGINE_PATH), str(config_path)],
        },
        {
            "blocker_id": "prospective_capture_only",
            "status": "active",
            "why": (
                "当前最多只能为未来时段做 prospective capture planning；"
                f"daemon cadence 来自 config，当前是 {micro_interval or '30'} 分钟。"
            ),
            "source_files": [str(config_path)],
        },
    ]


def build_prospective_policy(config_path: Path, priority_notes: dict[str, Any]) -> dict[str, Any]:
    pass_ratio_min = read_scalar_from_yaml(config_path, "ops_micro_capture_pass_ratio_min") or "0.70"
    schema_ok_min = read_scalar_from_yaml(config_path, "ops_micro_capture_schema_ok_ratio_min") or "0.90"
    time_sync_ok_min = read_scalar_from_yaml(config_path, "ops_micro_capture_time_sync_ok_ratio_min") or "0.90"
    cross_fail_max = read_scalar_from_yaml(config_path, "ops_micro_capture_cross_source_fail_ratio_max") or "0.35"
    min_runs = read_scalar_from_yaml(config_path, "execution_micro_capture_min_runs") or "4"
    lookback_days = read_scalar_from_yaml(config_path, "execution_micro_capture_lookback_days") or "7"
    by_symbol: dict[str, list[str]] = {}
    for row in priority_notes.get("major_capture_priority_rows", []):
        symbol = text(row.get("symbol"))
        hours = [text(item) for item in row.get("target_capture_hours_utc", []) if text(item)]
        if not symbol:
            continue
        current = by_symbol.setdefault(symbol, [])
        for item in hours:
            if item not in current:
                current.append(item)
    return {
        "mode": "prospective_reference_only",
        "symbols": sorted(by_symbol.keys()),
        "target_hours_utc_by_symbol": by_symbol,
        "reference_command_template": build_reference_command("<SYMBOLS>"),
        "done_when_thresholds": {
            "ops_micro_capture_pass_ratio_min": float(pass_ratio_min),
            "ops_micro_capture_schema_ok_ratio_min": float(schema_ok_min),
            "ops_micro_capture_time_sync_ok_ratio_min": float(time_sync_ok_min),
            "ops_micro_capture_cross_source_fail_ratio_max": float(cross_fail_max),
            "execution_micro_capture_min_runs": int(min_runs),
            "execution_micro_capture_lookback_days": int(lookback_days),
        },
        "execution_note": (
            "以下命令仅作 reference template；本轮不执行，因为 historical backfill 不支持，"
            "且 micro-capture 仍是 live-adjacent hook。"
        ),
    }


def build_checklist_rows(priority_notes: dict[str, Any], policy: dict[str, Any]) -> list[dict[str, Any]]:
    min_runs = to_int((policy.get("done_when_thresholds") or {}).get("execution_micro_capture_min_runs"), 4)
    lookback_days = to_int((policy.get("done_when_thresholds") or {}).get("execution_micro_capture_lookback_days"), 7)
    rows: list[dict[str, Any]] = []
    for row in priority_notes.get("major_capture_priority_rows", []):
        symbol = text(row.get("symbol"))
        rows.append(
            {
                "task_id": text(row.get("capture_task_id")),
                "symbol": symbol,
                "role": text(row.get("role")),
                "historical_entry_ts_utc": text(row.get("entry_ts_utc")),
                "historical_exit_reason": text(row.get("exit_reason")),
                "historical_net_r_multiple": to_float(row.get("net_r_multiple"), 0.0),
                "priority_score": to_float(row.get("priority_score"), 0.0),
                "status": "blocked_historical_backfill",
                "capture_window_utc_reference": dict(row.get("capture_window_utc") or {}),
                "future_target_hours_utc": list(row.get("target_capture_hours_utc") or []),
                "reference_command": build_reference_command(symbol),
                "done_when": (
                    f"在未来 {lookback_days} 天内，围绕 {','.join(row.get('target_capture_hours_utc', [])) or 'target hours'} UTC "
                    f"至少拿到 {min_runs} 次 {symbol} micro-capture，且 rolling stats 达到配置阈值。"
                ),
                "blocked_by": [
                    "historical_backfill_not_supported",
                    "micro_capture_live_adjacent_hook",
                ],
                "note": text(row.get("note")),
            }
        )
    return rows


def render_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Intraday Orderflow Majors Capture Execution Checklist",
        "",
        f"- generated_at_utc: `{text(payload.get('generated_at_utc'))}`",
        f"- research_decision: `{text(payload.get('research_decision'))}`",
        f"- recommended_brief: `{text(payload.get('recommended_brief'))}`",
        "",
        "## Active Blockers",
        "",
    ]
    for row in payload.get("execution_blockers", []):
        lines.append(f"- `{row['blocker_id']}` | status=`{row['status']}` | why=`{row['why']}`")
    lines.extend(["", "## Checklist Rows", ""])
    for row in payload.get("checklist_rows", []):
        lines.append(
            f"- `{row['task_id']}` | symbol=`{row['symbol']}` | status=`{row['status']}` | "
            f"future_hours=`{','.join(row.get('future_target_hours_utc', []))}` | "
            f"done_when=`{row['done_when']}`"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            f"- `{text(payload.get('research_note'))}`",
            f"- `{text(payload.get('limitation_note'))}`",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a RESEARCH_ONLY majors capture execution checklist that documents current blockers and future-only capture policy."
    )
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--priority-notes-path", default="")
    parser.add_argument("--config-path", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--stamp", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stamp_dt = parse_stamp(args.stamp)
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    priority_notes_path = Path(args.priority_notes_path).expanduser().resolve() if text(args.priority_notes_path) else latest_artifact(review_dir, "*_intraday_orderflow_majors_capture_priority_notes.json")
    config_path = Path(args.config_path).expanduser().resolve()

    priority_notes = load_json_mapping(priority_notes_path)
    execution_blockers = build_blockers(config_path)
    prospective_policy = build_prospective_policy(config_path, priority_notes)
    checklist_rows = build_checklist_rows(priority_notes, prospective_policy)
    research_decision = "prospective_capture_only_historical_gap_unfillable_with_current_cli"
    recommended_brief = (
        "majors_capture_execution_checklist:"
        f"blockers={','.join(row['blocker_id'] for row in execution_blockers)},"
        f"tasks={','.join(row['task_id'] for row in checklist_rows[:3]) if checklist_rows else '-'},"
        f"decision={research_decision}"
    )

    payload = {
        "action": "build_intraday_orderflow_majors_capture_execution_checklist",
        "ok": True,
        "status": "ok",
        "change_class": "RESEARCH_ONLY",
        "generated_at_utc": fmt_utc(stamp_dt),
        "source_artifacts": {
            "priority_notes_path": str(priority_notes_path),
            "config_path": str(config_path),
            "cli_path": str(CLI_PATH),
            "engine_path": str(ENGINE_PATH),
        },
        "execution_blockers": execution_blockers,
        "prospective_capture_policy": prospective_policy,
        "checklist_rows": checklist_rows,
        "research_decision": research_decision,
        "recommended_brief": recommended_brief,
        "research_note": (
            "这份 checklist 明确说明：当前 majors capture 只能做 prospective planning，"
            "不能把历史事件缺口误当成可即时补回的 backfill 任务。"
        ),
        "limitation_note": (
            "它只固化执行约束和 future-only policy，不会触发 micro-capture；"
            "因此它不能直接提高 coverage，只能防止下一步研究走错方向。"
        ),
    }

    json_path = review_dir / f"{args.stamp}_intraday_orderflow_majors_capture_execution_checklist.json"
    md_path = review_dir / f"{args.stamp}_intraday_orderflow_majors_capture_execution_checklist.md"
    latest_json_path = review_dir / "latest_intraday_orderflow_majors_capture_execution_checklist.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    latest_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "json_path": str(json_path),
                "md_path": str(md_path),
                "latest_json_path": str(latest_json_path),
                "research_decision": research_decision,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
