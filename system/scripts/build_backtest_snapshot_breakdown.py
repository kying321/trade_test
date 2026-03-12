#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml


def resolve_system_root() -> Path:
    env_root = str(os.getenv("LIE_SYSTEM_ROOT", "")).strip() or str(
        os.getenv("FENLIE_SYSTEM_ROOT", "")
    ).strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


SYSTEM_ROOT = resolve_system_root()
DEFAULT_REVIEW_DIR = SYSTEM_ROOT / "output" / "review"
DEFAULT_CONFIG_PATH = SYSTEM_ROOT / "config.yaml"


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


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a mapping")
    return payload


def find_latest(review_dir: Path, pattern: str) -> Path | None:
    files = sorted(review_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def parse_date(raw: str) -> dt.date:
    return dt.date.fromisoformat(str(raw).strip())


def review_backtest_start(as_of: dt.date, validation: dict[str, Any]) -> dt.date:
    lookback_raw = validation.get("review_backtest_lookback_days")
    try:
        lookback_days = int(lookback_raw)
    except (TypeError, ValueError):
        lookback_days = 0
    if lookback_days >= 30:
        return as_of - dt.timedelta(days=lookback_days - 1)
    start_raw = str(validation.get("review_backtest_start_date", "")).strip()
    if start_raw:
        try:
            return parse_date(start_raw)
        except ValueError:
            pass
    return dt.date(2015, 1, 1)


def resolve_snapshot_state(
    *,
    artifacts_dir: Path,
    start: dt.date,
    as_of: dt.date,
    max_age_days: int,
) -> dict[str, Any]:
    preferred = artifacts_dir / f"backtest_{start.isoformat()}_{as_of.isoformat()}.json"
    prefix = f"backtest_{start.isoformat()}_"
    candidates: list[tuple[dt.date, Path]] = []
    for path in artifacts_dir.glob(f"{prefix}*.json"):
        suffix = path.stem[len(prefix) :]
        try:
            end_date = parse_date(suffix)
        except ValueError:
            continue
        if end_date <= as_of:
            candidates.append((end_date, path))
    candidates.sort(key=lambda item: item[0])

    latest_end: dt.date | None = None
    latest_path: Path | None = None
    latest_age_days: int | None = None
    if candidates:
        latest_end, latest_path = candidates[-1]
        latest_age_days = (as_of - latest_end).days

    exact_exists = preferred.exists()
    fresh_enough = latest_age_days is not None and latest_age_days <= max_age_days
    if exact_exists:
        status = "exact_snapshot_missing_metrics_or_gate"
    elif latest_path is None:
        status = "snapshot_missing"
    elif fresh_enough:
        status = "fresh_fallback_available"
    else:
        status = "snapshot_stale"

    return {
        "status": status,
        "preferred_path": str(preferred),
        "preferred_exists": exact_exists,
        "latest_snapshot_path": str(latest_path) if latest_path else "",
        "latest_snapshot_end": latest_end.isoformat() if latest_end else "",
        "latest_snapshot_age_days": latest_age_days,
        "max_age_days": int(max_age_days),
        "fresh_fallback_available": bool(fresh_enough),
    }


def derive_payload(
    *,
    config_payload: dict[str, Any],
    ops_report: dict[str, Any],
    ops_report_path: Path,
) -> dict[str, Any]:
    validation = config_payload.get("validation", {}) if isinstance(config_payload.get("validation", {}), dict) else {}
    as_of = parse_date(ops_report.get("date", ""))
    review_dir = ops_report_path.parent
    artifacts_dir = review_dir.parent / "artifacts"
    start = review_backtest_start(as_of=as_of, validation=validation)
    max_age_days = int(
        validation.get(
            "ops_backtest_snapshot_max_age_days",
            validation.get("required_stable_replay_days", 3),
        )
    )
    snapshot_state = resolve_snapshot_state(
        artifacts_dir=artifacts_dir,
        start=start,
        as_of=as_of,
        max_age_days=max_age_days,
    )
    live_gate = ops_report.get("live_gate", {}) if isinstance(ops_report.get("live_gate", {}), dict) else {}
    gate_checks = ops_report.get("gate_checks", {}) if isinstance(ops_report.get("gate_checks", {}), dict) else {}
    gate_failed_checks = list(ops_report.get("gate_failed_checks", [])) if isinstance(ops_report.get("gate_failed_checks", []), list) else []
    blocking_reason_codes = list(live_gate.get("blocking_reason_codes", [])) if isinstance(live_gate.get("blocking_reason_codes", []), list) else []

    blocker_active = "backtest_snapshot" in blocking_reason_codes or "backtest_snapshot_ok" in gate_failed_checks
    diagnosis = {
        "summary": "live gate is currently blocked by a missing or stale backtest snapshot, not by a real drawdown breach.",
        "detail": (
            f"review_backtest_start_date resolves to {start.isoformat()}, so the gate expects "
            f"`backtest_{start.isoformat()}_{as_of.isoformat()}.json`. "
            f"The freshest available snapshot under output/artifacts currently ends at "
            f"{snapshot_state.get('latest_snapshot_end') or 'none'}, which is "
            f"{snapshot_state.get('latest_snapshot_age_days') if snapshot_state.get('latest_snapshot_age_days') is not None else 'N/A'} "
            f"days old against a max age of {max_age_days}."
        ),
        "impact": "ops_live_gate raises backtest_snapshot and keeps ops_status_red active until a fresh snapshot exists.",
    }

    local_backtest_cmd = (
        "cd /Users/jokenrobot/Downloads/Folders/fenlie/system && "
        f"PYTHONPATH=/Users/jokenrobot/Downloads/Folders/fenlie/system/src python3 -m lie_engine.cli "
        f"--config /Users/jokenrobot/Downloads/Folders/fenlie/system/config.yaml backtest "
        f"--start {start.isoformat()} --end {as_of.isoformat()}"
    )
    remote_backtest_cmd = (
        "ssh ubuntu@43.153.148.242 "
        f"\"cd /home/ubuntu/openclaw-system && PYTHONPATH=src python3 -m lie_engine.cli --config config.yaml backtest --start {start.isoformat()} --end {as_of.isoformat()}\""
    )
    refresh_cmd = (
        "cd /Users/jokenrobot/Downloads/Folders/fenlie/system && "
        "scripts/openclaw_cloud_bridge.sh live-ops-reconcile-refresh && "
        "scripts/openclaw_cloud_bridge.sh remote-live-handoff"
    )

    repair_sequence = [
        {
            "priority": 1,
            "action": "Rebuild the exact local backtest snapshot for the current gate date.",
            "command": local_backtest_cmd,
        },
        {
            "priority": 2,
            "action": "Rebuild the same backtest snapshot on the remote host so ops_report sees the same freshness window.",
            "command": remote_backtest_cmd,
        },
        {
            "priority": 3,
            "action": "Refresh remote ops-report and handoff after the snapshot exists.",
            "command": refresh_cmd,
        },
    ]

    return {
        "generated_at": fmt_utc(now_utc()),
        "ops_report_source": str(ops_report_path),
        "blocker_active": blocker_active,
        "as_of": as_of.isoformat(),
        "review_backtest_start": start.isoformat(),
        "snapshot_state": snapshot_state,
        "gate_checks": {
            "backtest_snapshot_ok": gate_checks.get("backtest_snapshot_ok"),
            "max_drawdown_ok": gate_checks.get("max_drawdown_ok"),
            "risk_violations_ok": gate_checks.get("risk_violations_ok"),
        },
        "gate_failed_checks": gate_failed_checks,
        "blocking_reason_codes": blocking_reason_codes,
        "config_thresholds": {
            "validation.review_backtest_start_date": str(validation.get("review_backtest_start_date", "")),
            "validation.required_stable_replay_days": int(validation.get("required_stable_replay_days", 3)),
            "validation.ops_backtest_snapshot_max_age_days": int(max_age_days),
        },
        "diagnosis": diagnosis,
        "repair_sequence": repair_sequence,
        "takeaway": "Backtest snapshot freshness is now the real blocker replacing the old synthetic max_drawdown blocker.",
    }


def render_markdown(payload: dict[str, Any]) -> str:
    snapshot_state = payload.get("snapshot_state", {}) if isinstance(payload.get("snapshot_state", {}), dict) else {}
    thresholds = payload.get("config_thresholds", {}) if isinstance(payload.get("config_thresholds", {}), dict) else {}
    diagnosis = payload.get("diagnosis", {}) if isinstance(payload.get("diagnosis", {}), dict) else {}
    lines = [
        "# Backtest Snapshot Breakdown",
        "",
        f"- ops report: `{payload.get('ops_report_source', '')}`",
        f"- blocker active: `{payload.get('blocker_active', False)}`",
        f"- as of: `{payload.get('as_of', '')}`",
        f"- takeaway: {payload.get('takeaway', '')}",
        "",
        "## Snapshot State",
        f"- review_backtest_start: `{payload.get('review_backtest_start', '')}`",
        f"- status: `{snapshot_state.get('status', '')}`",
        f"- preferred_path: `{snapshot_state.get('preferred_path', '')}`",
        f"- preferred_exists: `{snapshot_state.get('preferred_exists', False)}`",
        f"- latest_snapshot_path: `{snapshot_state.get('latest_snapshot_path', '')}`",
        f"- latest_snapshot_end: `{snapshot_state.get('latest_snapshot_end', '')}`",
        f"- latest_snapshot_age_days: `{snapshot_state.get('latest_snapshot_age_days', 'N/A')}`",
        f"- max_age_days: `{snapshot_state.get('max_age_days', 'N/A')}`",
        "",
        "## Gate Checks",
        f"- backtest_snapshot_ok: `{payload.get('gate_checks', {}).get('backtest_snapshot_ok')}`",
        f"- max_drawdown_ok: `{payload.get('gate_checks', {}).get('max_drawdown_ok')}`",
        f"- risk_violations_ok: `{payload.get('gate_checks', {}).get('risk_violations_ok')}`",
        "",
        "## Config Thresholds",
        f"- validation.review_backtest_start_date: `{thresholds.get('validation.review_backtest_start_date', '')}`",
        f"- validation.required_stable_replay_days: `{thresholds.get('validation.required_stable_replay_days', 'N/A')}`",
        f"- validation.ops_backtest_snapshot_max_age_days: `{thresholds.get('validation.ops_backtest_snapshot_max_age_days', 'N/A')}`",
        "",
        "## Diagnosis",
        f"- summary: {diagnosis.get('summary', '')}",
        f"- detail: {diagnosis.get('detail', '')}",
        f"- impact: {diagnosis.get('impact', '')}",
        "",
        "## Repair Sequence",
    ]
    for step in payload.get("repair_sequence", []):
        if not isinstance(step, dict):
            continue
        lines.append(f"{int(step.get('priority', 0))}. {step.get('action', '')}")
        lines.append(f"   - command: `{step.get('command', '')}`")
    lines.append("")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain the live gate backtest_snapshot blocker.")
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--keep", type=int, default=6)
    return parser.parse_args()


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
        "*_backtest_snapshot_breakdown.json",
        "*_backtest_snapshot_breakdown.md",
        "*_backtest_snapshot_breakdown_checksum.json",
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


def main() -> None:
    args = parse_args()
    review_dir = Path(args.review_dir).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if str(args.output_dir).strip() else review_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ops_report_path = find_latest(review_dir, "*_ops_report.json")
    if ops_report_path is None:
        raise SystemExit("no ops_report artifact found")

    payload = derive_payload(
        config_payload=load_config(config_path),
        ops_report=load_json(ops_report_path),
        ops_report_path=ops_report_path,
    )

    stamp = now_utc().strftime("%Y%m%dT%H%M%SZ")
    artifact_path = output_dir / f"{stamp}_backtest_snapshot_breakdown.json"
    markdown_path = output_dir / f"{stamp}_backtest_snapshot_breakdown.md"
    checksum_path = output_dir / f"{stamp}_backtest_snapshot_breakdown_checksum.json"

    artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")

    checksum_payload = {
        "artifact": str(artifact_path),
        "sha256": sha256_file(artifact_path),
        "generated_at": fmt_utc(now_utc()),
    }
    checksum_path.write_text(json.dumps(checksum_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    pruned_keep, pruned_age = prune_artifacts(
        output_dir,
        current_artifact=artifact_path,
        current_markdown=markdown_path,
        current_checksum=checksum_path,
        keep=max(3, int(args.keep)),
        ttl_hours=float(args.artifact_ttl_hours),
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


if __name__ == "__main__":
    main()
