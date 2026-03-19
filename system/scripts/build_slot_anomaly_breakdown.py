#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def resolve_system_root() -> Path:
    env_root = str(os.getenv("LIE_SYSTEM_ROOT", "")).strip() or str(
        os.getenv("FENLIE_SYSTEM_ROOT", "")
    ).strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


SYSTEM_ROOT = resolve_system_root()
DEFAULT_REVIEW_DIR = SYSTEM_ROOT / "output" / "review"


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def fmt_utc(value: dt.datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def find_latest(review_dir: Path, pattern: str) -> Path | None:
    files = sorted(review_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def unwrap_capture_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    raw = payload.get("payload", {})
    if str(payload.get("action") or "").strip() == "capture_remote_live_handoff_input" and isinstance(raw, dict):
        return raw
    return payload


def summarize_slot_anomaly(payload: dict[str, Any], *, ops_status_path: Path | None) -> dict[str, Any]:
    checks = payload.get("checks", {}) if isinstance(payload.get("checks", {}), dict) else {}
    failed_checks = [str(k) for k, v in checks.items() if not bool(v)]
    alerts = [
        str(x)
        for x in (payload.get("alerts", []) if isinstance(payload.get("alerts", []), list) else [])
        if str(x).strip()
    ]
    metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics", {}), dict) else {}
    slots = payload.get("slots", {}) if isinstance(payload.get("slots", {}), dict) else {}
    series = payload.get("series", []) if isinstance(payload.get("series", []), list) else []
    series_sample: list[dict[str, Any]] = []
    for row in series[:3]:
        if not isinstance(row, dict):
            continue
        series_sample.append(
            {
                "date": str(row.get("date") or "").strip(),
                "anomalies": int(row.get("anomalies", 0) or 0),
                "alerts": [
                    str(x)
                    for x in (row.get("alerts", []) if isinstance(row.get("alerts", []), list) else [])
                    if str(x).strip()
                ],
            }
        )
    slot_brief_parts: list[str] = []
    for name in ("premarket", "intraday", "eod"):
        row = slots.get(name, {}) if isinstance(slots.get(name, {}), dict) else {}
        ratio = row.get("anomaly_ratio")
        if ratio is None:
            continue
        try:
            ratio_text = f"{float(ratio):.2f}"
        except Exception:
            ratio_text = str(ratio)
        slot_brief_parts.append(f"{name}={ratio_text}")
    brief = (
        f"failed={','.join(failed_checks) or '-'};"
        f"alerts={','.join(alerts[:3]) or '-'};"
        f"slots={','.join(slot_brief_parts) or '-'}"
    )
    return {
        "status": "active" if bool(payload.get("active", False)) else "clear",
        "brief": brief,
        "window_days": int(payload.get("window_days", 0) or 0),
        "samples": int(payload.get("samples", 0) or 0),
        "min_samples": int(payload.get("min_samples", 0) or 0),
        "failed_checks": failed_checks,
        "alerts": alerts,
        "metrics": metrics,
        "slots": slots,
        "series_sample": series_sample,
        "artifact": str(ops_status_path) if ops_status_path else "",
    }


def prune_artifacts(
    review_dir: Path,
    *,
    current_artifact: Path,
    current_markdown: Path,
    current_checksum: Path,
    keep: int,
    ttl_hours: float,
) -> tuple[list[str], list[str]]:
    cutoff = now_utc() - dt.timedelta(hours=max(1.0, ttl_hours))
    protected = {current_artifact.name, current_markdown.name, current_checksum.name}
    candidates: list[Path] = []
    for pattern in (
        "*_slot_anomaly_breakdown.json",
        "*_slot_anomaly_breakdown.md",
        "*_slot_anomaly_breakdown_checksum.json",
    ):
        candidates.extend(review_dir.glob(pattern))
    pruned_age: list[str] = []
    survivors: list[Path] = []
    for path in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
        if path.name in protected:
            survivors.append(path)
            continue
        try:
            mtime = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)
        except OSError:
            continue
        if mtime < cutoff:
            path.unlink(missing_ok=True)
            pruned_age.append(str(path))
        else:
            survivors.append(path)
    pruned_keep: list[str] = []
    for path in survivors[keep:]:
        if path.name in protected:
            continue
        path.unlink(missing_ok=True)
        pruned_keep.append(str(path))
    return pruned_keep, pruned_age


def derive_payload(
    *,
    handoff_payload: dict[str, Any],
    handoff_path: Path,
    ops_status_payload: dict[str, Any] | None = None,
    ops_status_path: Path | None = None,
) -> dict[str, Any]:
    ready_check = handoff_payload.get("ready_check", {})
    if not isinstance(ready_check, dict):
        ready_check = {}
    ops_reconcile = ready_check.get("ops_reconcile", {})
    if not isinstance(ops_reconcile, dict):
        ops_reconcile = {}
    ops_live_gate = ready_check.get("ops_live_gate", {})
    if not isinstance(ops_live_gate, dict):
        ops_live_gate = {}

    blocking_reason_codes = [
        str(x).strip()
        for x in (ops_live_gate.get("blocking_reason_codes", []) if isinstance(ops_live_gate.get("blocking_reason_codes", []), list) else [])
        if str(x).strip()
    ]
    rollback_reason_codes = [
        str(x).strip()
        for x in (ops_live_gate.get("rollback_reason_codes", []) if isinstance(ops_live_gate.get("rollback_reason_codes", []), list) else [])
        if str(x).strip()
    ]
    gate_failed_checks = [
        str(x).strip()
        for x in (ops_live_gate.get("gate_failed_checks", []) if isinstance(ops_live_gate.get("gate_failed_checks", []), list) else [])
        if str(x).strip()
    ]

    raw_ops_status = ops_status_payload if isinstance(ops_status_payload, dict) else {}
    live_slot_anomaly = (
        raw_ops_status.get("live_slot_anomaly", {})
        if isinstance(raw_ops_status.get("live_slot_anomaly", {}), dict)
        else {}
    )
    slot_detail = (
        summarize_slot_anomaly(live_slot_anomaly, ops_status_path=ops_status_path)
        if live_slot_anomaly
        else {
            "status": "unavailable",
            "brief": "unavailable:-",
            "window_days": 0,
            "samples": 0,
            "min_samples": 0,
            "failed_checks": [],
            "alerts": [],
            "metrics": {},
            "slots": {},
            "series_sample": [],
            "artifact": str(ops_status_path) if ops_status_path else "",
        }
    )
    blocker_active = "slot_anomaly" in blocking_reason_codes or "slot_anomaly_ok" in gate_failed_checks
    ops_report_artifact = str(ops_reconcile.get("artifact_path") or "").strip()
    ops_report_age_hours = ops_reconcile.get("artifact_age_hours")
    artifact_date = str(ops_reconcile.get("artifact_date") or "").strip()
    refresh_command = (
        "cd /Users/jokenrobot/Downloads/Folders/fenlie/system && "
        "scripts/openclaw_cloud_bridge.sh live-ops-reconcile-refresh && "
        "scripts/openclaw_cloud_bridge.sh remote-live-handoff"
    )
    failed_slot_checks = slot_detail.get("failed_checks", []) if isinstance(slot_detail, dict) else []
    failed_slot_checks = [str(x).strip() for x in failed_slot_checks if str(x).strip()]
    if any(code == "missing_ratio_ok" for code in failed_slot_checks):
        repair_focus_core = "优先补齐缺失槽位数据并恢复 slot missing ratio"
    elif any(
        code in {
            "premarket_anomaly_ok",
            "intraday_anomaly_ok",
            "eod_quality_anomaly_ok",
            "eod_quality_regime_bucket_ok",
            "eod_anomaly_ok",
        }
        for code in failed_slot_checks
    ):
        repair_focus_core = "优先修复 premarket/intraday/eod quality 与 source_confidence 异常"
    else:
        repair_focus_core = "优先修复 slot_anomaly 缺陷"
    repair_focus = (
        f"{repair_focus_core}并重跑 lie ops-report --date {artifact_date} --window-days 7"
        if artifact_date
        else f"{repair_focus_core}并重跑 lie ops-report --window-days 7"
    )
    payload_gap = (
        ""
        if live_slot_anomaly
        else "slot_anomaly payload is not embedded in persisted remote ops status; deeper slot metrics still need direct ops_report capture."
    )
    diagnosis_summary = (
        "slot_anomaly is the active root blocker inside ops_live_gate; rollback_hard is only a derived wrapper."
        if blocker_active
        else "slot_anomaly is not currently active inside ops_live_gate."
    )
    diagnosis_detail = (
        f"blocking_reason_codes={','.join(blocking_reason_codes) or '-'}; "
        f"rollback_reason_codes={','.join(rollback_reason_codes) or '-'}; "
        f"gate_failed_checks={','.join(gate_failed_checks) or '-'}; "
        f"slot_failed_checks={','.join(failed_slot_checks) or '-'}; "
        f"ops_report={ops_report_artifact or '-'} age_hours={ops_report_age_hours!r}."
    )
    impact = (
        "slot_anomaly keeps rollback_hard and ops_status_red attached until slot checks recover."
        if blocker_active
        else "slot_anomaly is not contributing to the current rollback wrapper."
    )
    return {
        "generated_at": fmt_utc(now_utc()),
        "handoff_source": str(handoff_path),
        "ops_status_source": str(ops_status_path) if ops_status_path else "",
        "status": "slot_anomaly_active_root_cause" if blocker_active else "slot_anomaly_clear",
        "brief": (
            f"slot_anomaly_active_root_cause:{artifact_date or '-'}:{','.join(failed_slot_checks[:3]) or '-'}"
            if blocker_active
            else f"slot_anomaly_clear:{artifact_date or '-'}:{','.join(failed_slot_checks[:3]) or '-'}"
        ),
        "blocker_active": blocker_active,
        "ops_report_artifact": ops_report_artifact,
        "ops_report_artifact_age_hours": ops_report_age_hours,
        "ops_report_artifact_date": artifact_date,
        "rollback_level": str(ops_live_gate.get("rollback_level") or "").strip(),
        "rollback_action": str(ops_live_gate.get("rollback_action") or "").strip(),
        "blocking_reason_codes": blocking_reason_codes,
        "rollback_reason_codes": rollback_reason_codes,
        "gate_failed_checks": gate_failed_checks,
        "slot_anomaly_reason_present": "slot_anomaly" in blocking_reason_codes,
        "slot_anomaly_check_failed": "slot_anomaly_ok" in gate_failed_checks,
        "rollback_wrapper_present": "rollback_hard" in blocking_reason_codes,
        "ops_status_red_present": "ops_status_red" in blocking_reason_codes,
        "source_check": "slot_anomaly_ok",
        "source_ref": "/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/orchestration/release.py:2349",
        "rollback_ref": "/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/orchestration/release.py:4744",
        "repair_focus": repair_focus,
        "payload_gap": payload_gap,
        "slot_detail": slot_detail,
        "diagnosis": {
            "summary": diagnosis_summary,
            "detail": diagnosis_detail,
            "impact": impact,
        },
        "repair_sequence": [
            {
                "priority": 1,
                "action": repair_focus,
                "command": "",
            },
            {
                "priority": 2,
                "action": "刷新 remote ops report 与 handoff，确认 slot_anomaly 是否已离开 blocker 集。",
                "command": refresh_command,
            },
        ],
    }


def render_markdown(payload: dict[str, Any]) -> str:
    diagnosis = payload.get("diagnosis", {}) if isinstance(payload.get("diagnosis", {}), dict) else {}
    slot_detail = payload.get("slot_detail", {}) if isinstance(payload.get("slot_detail", {}), dict) else {}
    lines = [
        "# Slot Anomaly Breakdown",
        "",
        f"- handoff source: `{payload.get('handoff_source', '')}`",
        f"- ops status source: `{payload.get('ops_status_source', '') or '-'}`",
        f"- status: `{payload.get('status', '')}` brief=`{payload.get('brief', '')}`",
        f"- blocker_active: `{payload.get('blocker_active', False)}`",
        f"- ops_report: `{payload.get('ops_report_artifact', '') or '-'}` age_hours=`{payload.get('ops_report_artifact_age_hours')}`",
        f"- source_check: `{payload.get('source_check', '')}`",
        f"- blocking_reason_codes: `{', '.join(payload.get('blocking_reason_codes', [])) or '-'}`",
        f"- rollback_reason_codes: `{', '.join(payload.get('rollback_reason_codes', [])) or '-'}`",
        f"- gate_failed_checks: `{', '.join(payload.get('gate_failed_checks', [])) or '-'}`",
        f"- repair_focus: `{payload.get('repair_focus', '')}`",
        f"- payload_gap: `{payload.get('payload_gap', '')}`",
        "",
        "## Slot Detail",
        f"- status: `{slot_detail.get('status', '')}` brief=`{slot_detail.get('brief', '')}`",
        f"- failed_checks: `{', '.join(slot_detail.get('failed_checks', [])) or '-'}`",
        f"- alerts: `{', '.join(slot_detail.get('alerts', [])) or '-'}`",
        "",
        "## Diagnosis",
        f"- summary: {diagnosis.get('summary', '')}",
        f"- detail: {diagnosis.get('detail', '')}",
        f"- impact: {diagnosis.get('impact', '')}",
        "",
        "## Repair Sequence",
    ]
    for row in payload.get("repair_sequence", []):
        if not isinstance(row, dict):
            continue
        lines.append(f"- `{row.get('priority', '')}` `{row.get('action', '')}`")
        lines.append(f"  - command: `{row.get('command', '') or '-'}`")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain the remote live slot_anomaly blocker.")
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--handoff-json", type=Path, default=None)
    parser.add_argument("--ops-status-json", type=Path, default=None)
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--artifact-keep", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    review_dir = args.review_dir.expanduser().resolve()
    handoff_path = args.handoff_json.expanduser().resolve() if args.handoff_json else find_latest(review_dir, "*_remote_live_handoff.json")
    ops_status_path = (
        args.ops_status_json.expanduser().resolve()
        if args.ops_status_json
        else find_latest(review_dir, "*_remote_live_ops_reconcile_status.json")
    )
    if handoff_path is None:
        raise SystemExit("remote live handoff artifact not found")
    handoff_payload = load_json(handoff_path)
    ops_status_payload = unwrap_capture_payload(load_json(ops_status_path)) if ops_status_path else None
    payload = derive_payload(
        handoff_payload=handoff_payload,
        handoff_path=handoff_path,
        ops_status_payload=ops_status_payload,
        ops_status_path=ops_status_path,
    )
    stamp = now_utc().strftime("%Y%m%dT%H%M%SZ")
    artifact_path = review_dir / f"{stamp}_slot_anomaly_breakdown.json"
    markdown_path = review_dir / f"{stamp}_slot_anomaly_breakdown.md"
    checksum_path = review_dir / f"{stamp}_slot_anomaly_breakdown_checksum.json"
    review_dir.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    checksum_payload = {
        "artifact": str(artifact_path),
        "markdown": str(markdown_path),
        "sha256": sha256_file(artifact_path),
        "generated_at": payload.get("generated_at"),
    }
    checksum_path.write_text(json.dumps(checksum_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pruned_keep, pruned_age = prune_artifacts(
        review_dir,
        current_artifact=artifact_path,
        current_markdown=markdown_path,
        current_checksum=checksum_path,
        keep=max(1, args.artifact_keep),
        ttl_hours=args.artifact_ttl_hours,
    )
    payload["artifact"] = str(artifact_path)
    payload["markdown"] = str(markdown_path)
    payload["checksum"] = str(checksum_path)
    payload["pruned_keep"] = pruned_keep
    payload["pruned_age"] = pruned_age
    artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checksum_payload["sha256"] = sha256_file(artifact_path)
    checksum_path.write_text(json.dumps(checksum_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
