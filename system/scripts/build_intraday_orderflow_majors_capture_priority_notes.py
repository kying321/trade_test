#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = SYSTEM_ROOT / "output"
DEFAULT_REVIEW_DIR = DEFAULT_OUTPUT_ROOT / "review"


def parse_stamp(raw: str) -> dt.datetime:
    parsed = dt.datetime.strptime(str(raw).strip(), "%Y%m%dT%H%M%SZ")
    return parsed.replace(tzinfo=dt.timezone.utc)


def parse_utc(raw: str) -> dt.datetime | None:
    txt = str(raw or "").strip()
    if not txt:
        return None
    try:
        parsed = dt.datetime.fromisoformat(txt.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def fmt_utc(value: dt.datetime | None) -> str:
    if value is None:
        return ""
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def text(value: Any) -> str:
    return str(value or "").strip()


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return float(out)


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


def role_weight(role: str) -> float:
    if role == "mainline_primary":
        return 1.25
    if role == "majors_anchor":
        return 1.0
    return 0.5


def severity_weight(net_r_multiple: float) -> float:
    if net_r_multiple <= -1.0:
        return 1.25
    if net_r_multiple < 0.0:
        return 1.0
    if net_r_multiple < 0.2:
        return 0.85
    return 0.5


def build_capture_window(entry_ts_utc: str, *, before_hours: int = 4, after_hours: int = 8) -> dict[str, str]:
    entry_ts = parse_utc(entry_ts_utc)
    if entry_ts is None:
        return {"start_utc": "", "end_utc": ""}
    return {
        "start_utc": fmt_utc(entry_ts - dt.timedelta(hours=before_hours)),
        "end_utc": fmt_utc(entry_ts + dt.timedelta(hours=after_hours)),
    }


def build_major_capture_priority_rows(casebook: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for symbol_row in casebook.get("symbol_gap_rows", []):
        role = text(symbol_row.get("role"))
        if role not in {"mainline_primary", "majors_anchor"}:
            continue
        coverage_24h_ratio = to_float(symbol_row.get("coverage_24h_ratio"), 0.0)
        coverage_48h_ratio = to_float(symbol_row.get("coverage_48h_ratio"), 0.0)
        context_48h_ratio = to_float(symbol_row.get("context_available_48h_ratio"), 0.0)
        gap24 = max(0.0, 0.60 - coverage_24h_ratio)
        gap48 = max(0.0, 0.80 - coverage_48h_ratio)
        examples = list(symbol_row.get("gap_examples") or [])
        for idx, example in enumerate(examples, start=1):
            entry_ts_utc = text(example.get("entry_ts_utc"))
            net_r_multiple = to_float(example.get("net_r_multiple"), 0.0)
            delta_hours = to_float(example.get("nearest_capture_delta_hours"), 0.0)
            score = (
                100.0 * role_weight(role)
                + gap24 * 100.0
                + gap48 * 80.0
                + max(0.0, min(delta_hours, 168.0) / 24.0) * 4.0
                + severity_weight(net_r_multiple) * 10.0
                + (1.0 - context_48h_ratio) * 8.0
            )
            diagnosis = "capture_gap_blocks_inference" if delta_hours > 48.0 else "capture_present_but_context_missing"
            rows.append(
                {
                    "capture_task_id": f"{text(symbol_row.get('symbol')).lower()}_{idx}",
                    "symbol": text(symbol_row.get("symbol")),
                    "role": role,
                    "priority_score": round(score, 3),
                    "entry_ts_utc": entry_ts_utc,
                    "exit_reason": text(example.get("exit_reason")),
                    "net_r_multiple": net_r_multiple,
                    "nearest_capture_ts_utc": text(example.get("nearest_capture_ts_utc")),
                    "nearest_capture_delta_hours": delta_hours,
                    "diagnosis": diagnosis,
                    "capture_window_utc": build_capture_window(entry_ts_utc),
                    "target_capture_hours_utc": [row.get("hour_utc") for row in (symbol_row.get("top_capture_hours_utc") or [])],
                    "coverage_targets": {
                        "goal_24h_ratio": 0.60,
                        "goal_48h_ratio": 0.80,
                        "current_24h_ratio": coverage_24h_ratio,
                        "current_48h_ratio": coverage_48h_ratio,
                    },
                    "note": (
                        "先补 entry 前 4h 到后 8h 的 micro capture，并确保 context_mode/trust/veto 字段可产出；"
                        "在补齐前禁止把该事件用于 replay/veto 结论。"
                    ),
                }
            )
    rows.sort(key=lambda row: (-to_float(row.get("priority_score"), 0.0), text(row.get("symbol")), text(row.get("entry_ts_utc"))))
    return rows


def build_eth_bad_trade_notes(casebook: dict[str, Any], veto_event_study: dict[str, Any]) -> list[dict[str, Any]]:
    harmful_rules = [
        text(row.get("rule_name"))
        for row in veto_event_study.get("eth_veto_rules_48h", [])
        if text(row.get("classification")) in {"harmful_on_current_sample", "overblocking_current_sample"}
    ]
    notes: list[dict[str, Any]] = []
    diagnosis_map = {
        "capture_gap_blocks_inference": {
            "what_it_means": "当前坏交易没有足够近的 micro capture；任何 veto 解释都属于过度推断。",
            "next_action": "优先补 entry 前 4h 到后 8h 的 capture；必要时扩到后 12h。",
            "allowed_conclusion": "只能记录为采样缺口导致无法判断。",
        },
        "capture_present_but_context_missing": {
            "what_it_means": "最近邻 capture 存在，但 context_mode/context_note 没落出来。",
            "next_action": "先追查 micro context 生成链与 schema completeness，而不是改交易逻辑。",
            "allowed_conclusion": "只能记录为 context 生成缺口。",
        },
        "covered_but_low_trust_continuation": {
            "what_it_means": "事件被 capture 覆盖，但只有 low-trust continuation，当前不足以当 hard veto。",
            "next_action": "保留为弱观察标签；继续等更高 trust 或更密集连续捕获。",
            "allowed_conclusion": "只能记作低信任 continuation 风险注记。",
        },
    }
    for idx, case in enumerate((casebook.get("eth_bad_trade_casebook") or {}).get("cases", []), start=1):
        diagnosis = text(case.get("diagnosis"))
        profile = diagnosis_map.get(
            diagnosis,
            {
                "what_it_means": "当前案例证据不足，只能先保留为人工复核样本。",
                "next_action": "继续补采样并保持不拟合。",
                "allowed_conclusion": "只能做研究注记。",
            },
        )
        notes.append(
            {
                "case_id": f"eth_bad_case_{idx}",
                "entry_ts_utc": text(case.get("entry_ts_utc")),
                "exit_ts_utc": text(case.get("exit_ts_utc")),
                "exit_reason": text(case.get("exit_reason")),
                "net_r_multiple": to_float(case.get("net_r_multiple"), 0.0),
                "severity": text(case.get("severity")),
                "diagnosis": diagnosis,
                "capture_window_utc": build_capture_window(text(case.get("entry_ts_utc"))),
                "matched_context_mode": text(case.get("matched_context_mode")),
                "matched_trust_tier": text(case.get("matched_trust_tier")),
                "matched_veto_hint": text(case.get("matched_veto_hint")),
                "what_it_means": profile["what_it_means"],
                "next_action": profile["next_action"],
                "allowed_conclusion": profile["allowed_conclusion"],
                "forbidden_rules_now": harmful_rules,
            }
        )
    return notes


def render_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Intraday Orderflow Majors Capture Priority Notes",
        "",
        f"- generated_at_utc: `{text(payload.get('generated_at_utc'))}`",
        f"- research_decision: `{text(payload.get('research_decision'))}`",
        f"- recommended_brief: `{text(payload.get('recommended_brief'))}`",
        "",
        "## Majors Capture Priority",
        "",
    ]
    for row in payload.get("major_capture_priority_rows", []):
        lines.append(
            f"- `{row['capture_task_id']}` | symbol=`{row['symbol']}` | role=`{row['role']}` | "
            f"priority_score=`{float(row['priority_score']):.3f}` | entry=`{row['entry_ts_utc']}` | "
            f"delta_h=`{float(row['nearest_capture_delta_hours']):.2f}` | diagnosis=`{row['diagnosis']}`"
        )
    lines.extend(["", "## ETH Bad Trade Notes", ""])
    for row in payload.get("eth_bad_trade_notes", []):
        lines.append(
            f"- `{row['case_id']}` | entry=`{row['entry_ts_utc']}` | severity=`{row['severity']}` | "
            f"diagnosis=`{row['diagnosis']}` | next_action=`{row['next_action']}`"
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
        description="Build a RESEARCH_ONLY majors capture priority checklist and ETH bad-trade veto notes from the latest coverage gap artifacts."
    )
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--coverage-gap-casebook-path", default="")
    parser.add_argument("--veto-event-study-path", default="")
    parser.add_argument("--stamp", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stamp_dt = parse_stamp(args.stamp)
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    coverage_gap_casebook_path = Path(args.coverage_gap_casebook_path).expanduser().resolve() if text(args.coverage_gap_casebook_path) else latest_artifact(review_dir, "*_intraday_orderflow_coverage_gap_casebook.json")
    veto_event_study_path = Path(args.veto_event_study_path).expanduser().resolve() if text(args.veto_event_study_path) else latest_artifact(review_dir, "*_intraday_orderflow_veto_event_study.json")

    casebook = load_json_mapping(coverage_gap_casebook_path)
    veto_event_study = load_json_mapping(veto_event_study_path)

    major_capture_priority_rows = build_major_capture_priority_rows(casebook)
    eth_bad_trade_notes = build_eth_bad_trade_notes(casebook, veto_event_study)
    research_decision = "execute_majors_capture_priority_before_any_new_replay"
    top_tasks = ",".join(row["capture_task_id"] for row in major_capture_priority_rows[:3]) if major_capture_priority_rows else "-"
    bad_case_ids = ",".join(row["case_id"] for row in eth_bad_trade_notes[:3]) if eth_bad_trade_notes else "-"
    recommended_brief = (
        f"majors_capture_priority:top_tasks={top_tasks},"
        f"eth_bad_cases={bad_case_ids},"
        f"decision={research_decision}"
    )

    payload = {
        "action": "build_intraday_orderflow_majors_capture_priority_notes",
        "ok": True,
        "status": "ok",
        "change_class": "RESEARCH_ONLY",
        "generated_at_utc": fmt_utc(stamp_dt),
        "source_artifacts": {
            "coverage_gap_casebook_path": str(coverage_gap_casebook_path),
            "veto_event_study_path": str(veto_event_study_path),
        },
        "major_capture_priority_rows": major_capture_priority_rows,
        "eth_bad_trade_notes": eth_bad_trade_notes,
        "research_decision": research_decision,
        "recommended_brief": recommended_brief,
        "research_note": (
            "这份工件把 ETH/BTC majors 的采样缺口优先级和 ETH 坏交易逐案 note 固化为 source-owned 清单；"
            "作用是约束下一轮只能先补关键 capture，再决定是否重放。"
        ),
        "limitation_note": (
            "当前优先级仍基于已有 gap_examples 和坏交易样本，属于研究排队工具；"
            "它不能替代连续采样，也不能直接推出可交易 veto 规则。"
        ),
    }

    json_path = review_dir / f"{args.stamp}_intraday_orderflow_majors_capture_priority_notes.json"
    md_path = review_dir / f"{args.stamp}_intraday_orderflow_majors_capture_priority_notes.md"
    latest_json_path = review_dir / "latest_intraday_orderflow_majors_capture_priority_notes.json"
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
