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
        "*_risk_violations_breakdown.json",
        "*_risk_violations_breakdown.md",
        "*_risk_violations_breakdown_checksum.json",
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
    config_payload: dict[str, Any],
    blocker_report: dict[str, Any],
    blocker_report_path: Path,
    ops_breakdown: dict[str, Any],
    ops_breakdown_path: Path,
) -> dict[str, Any]:
    validation = config_payload.get("validation", {}) if isinstance(config_payload.get("validation", {}), dict) else {}
    dd_threshold = float(validation.get("max_drawdown_max", 0.18))
    mode_health_dd = float(validation.get("mode_health_max_drawdown_max", dd_threshold))
    mode_health_viol = int(validation.get("mode_health_max_violations", 0))
    positive_ratio_min = float(validation.get("positive_window_ratio_min", 0.70))

    root_codes = [
        str(x.get("code", "")).strip()
        for x in (ops_breakdown.get("root_causes", []) if isinstance(ops_breakdown.get("root_causes", []), list) else [])
        if isinstance(x, dict) and str(x.get("code", "")).strip()
    ]
    duplicate_active = "risk_violations" in root_codes and "max_drawdown" in root_codes

    minimal_patch = {
        "summary": "Keep max_drawdown as the single drawdown gate and stop treating drawdown-derived violations as a separate root blocker.",
        "touch_points": [
            "/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/orchestration/release.py:649",
            "/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/backtest/engine.py:421",
        ],
        "expected_effect": "rollback_hard should no longer receive both risk_violations and max_drawdown from the same drawdown breach.",
    }
    robust_patch = {
        "summary": "Split BacktestResult.violations into explicit violation codes and reserve risk_violations for non-drawdown failures such as exposure, turnover, or trade-count guard breaches.",
        "touch_points": [
            "/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/models.py:164",
            "/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/backtest/engine.py:421",
            "/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/orchestration/release.py:649",
            "/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/research/optimizer.py:169",
            "/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/research/strategy_lab.py:1805",
        ],
        "expected_effect": "risk_violations becomes an independent gate again instead of mirroring max_drawdown.",
    }

    repair_sequence = [
        {
            "priority": 1,
            "action": "Decide whether to take the minimal patch or the robust patch.",
            "reason": "Current live gate counts the same drawdown breach twice.",
        },
        {
            "priority": 2,
            "action": "Rebuild ops_report and remote handoff after the patch.",
            "command": "cd /Users/jokenrobot/Downloads/Folders/fenlie/system && scripts/openclaw_cloud_bridge.sh live-ops-reconcile-refresh && scripts/openclaw_cloud_bridge.sh remote-live-handoff",
        },
        {
            "priority": 3,
            "action": "Only after duplicate blocker removal, re-evaluate slot_anomaly and secondary failed checks.",
            "reason": "Secondary checks should not be analyzed while root blockers are duplicated.",
        },
    ]

    return {
        "generated_at": fmt_utc(now_utc()),
        "blocker_report_source": str(blocker_report_path),
        "ops_breakdown_source": str(ops_breakdown_path),
        "current_live_decision": str(blocker_report.get("current_decision", "")),
        "blocking_reason_codes": list(ops_breakdown.get("blocking_reason_codes", [])),
        "root_cause_codes": root_codes,
        "duplicate_drawdown_blocker_active": duplicate_active,
        "config_thresholds": {
            "validation.max_drawdown_max": dd_threshold,
            "validation.mode_health_max_drawdown_max": mode_health_dd,
            "validation.mode_health_max_violations": mode_health_viol,
            "validation.positive_window_ratio_min": positive_ratio_min,
        },
        "code_path_summary": {
            "release_gate": {
                "risk_violations_ok": {
                    "ref": "/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/orchestration/release.py:649",
                    "logic": "violations == 0",
                },
                "max_drawdown_ok": {
                    "ref": "/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/orchestration/release.py:648",
                    "logic": f"max_drawdown <= {dd_threshold}",
                },
            },
            "backtest_engine": {
                "violations_counter": {
                    "ref": "/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/backtest/engine.py:421",
                    "logic": f"violations += 1 when max_drawdown > {dd_threshold}",
                }
            },
            "mode_health": {
                "ref": "/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/engine.py:4883",
                "logic": f"worst_drawdown <= {mode_health_dd} and total_violations <= {mode_health_viol}",
            },
        },
        "finding": {
            "summary": "risk_violations currently mirrors max_drawdown under the current backtest implementation.",
            "detail": (
                "The live gate checks both max_drawdown_ok and risk_violations_ok. "
                "But the backtest engine currently increments violations only when max_drawdown exceeds the same validation threshold, "
                "so the same drawdown breach becomes two live blockers."
            ),
            "impact": "rollback_hard receives duplicated root blockers from a single drawdown failure.",
        },
        "patch_options": {
            "minimal_patch": minimal_patch,
            "robust_patch": robust_patch,
        },
        "repair_sequence": repair_sequence,
        "takeaway": "Fix the duplicate drawdown blocker before touching secondary ops_live_gate failures.",
    }


def render_markdown(payload: dict[str, Any]) -> str:
    thresholds = payload.get("config_thresholds", {})
    release_gate = (
        payload.get("code_path_summary", {}).get("release_gate", {})
        if isinstance(payload.get("code_path_summary", {}), dict)
        else {}
    )
    backtest_engine = (
        payload.get("code_path_summary", {}).get("backtest_engine", {})
        if isinstance(payload.get("code_path_summary", {}), dict)
        else {}
    )
    mode_health = (
        payload.get("code_path_summary", {}).get("mode_health", {})
        if isinstance(payload.get("code_path_summary", {}), dict)
        else {}
    )
    lines = [
        "# Risk Violations Breakdown",
        "",
        f"- blocker report: `{payload.get('blocker_report_source', '')}`",
        f"- ops breakdown: `{payload.get('ops_breakdown_source', '')}`",
        f"- current live decision: `{payload.get('current_live_decision', '')}`",
        f"- duplicate drawdown blocker active: `{payload.get('duplicate_drawdown_blocker_active', False)}`",
        f"- takeaway: {payload.get('takeaway', '')}",
        "",
        "## Thresholds",
        f"- validation.max_drawdown_max: `{thresholds.get('validation.max_drawdown_max', 'N/A')}`",
        f"- validation.mode_health_max_drawdown_max: `{thresholds.get('validation.mode_health_max_drawdown_max', 'N/A')}`",
        f"- validation.mode_health_max_violations: `{thresholds.get('validation.mode_health_max_violations', 'N/A')}`",
        f"- validation.positive_window_ratio_min: `{thresholds.get('validation.positive_window_ratio_min', 'N/A')}`",
        "",
        "## Code Path Summary",
        f"- release risk_violations_ok: `{release_gate.get('risk_violations_ok', {}).get('logic', '')}` @ `{release_gate.get('risk_violations_ok', {}).get('ref', '')}`",
        f"- release max_drawdown_ok: `{release_gate.get('max_drawdown_ok', {}).get('logic', '')}` @ `{release_gate.get('max_drawdown_ok', {}).get('ref', '')}`",
        f"- backtest violations counter: `{backtest_engine.get('violations_counter', {}).get('logic', '')}` @ `{backtest_engine.get('violations_counter', {}).get('ref', '')}`",
        f"- mode_health: `{mode_health.get('logic', '')}` @ `{mode_health.get('ref', '')}`",
        "",
        "## Finding",
        f"- summary: {payload.get('finding', {}).get('summary', '')}",
        f"- detail: {payload.get('finding', {}).get('detail', '')}",
        f"- impact: {payload.get('finding', {}).get('impact', '')}",
        "",
        "## Patch Options",
        f"- minimal: {payload.get('patch_options', {}).get('minimal_patch', {}).get('summary', '')}",
        f"- robust: {payload.get('patch_options', {}).get('robust_patch', {}).get('summary', '')}",
        "",
        "## Repair Sequence",
    ]
    for row in payload.get("repair_sequence", []):
        if not isinstance(row, dict):
            continue
        lines.append(f"- [{row.get('priority', '')}] {row.get('action', '')}")
        if row.get("reason"):
            lines.append(f"  - reason: {row.get('reason', '')}")
        if row.get("command"):
            lines.append(f"  - command: `{row.get('command', '')}`")
    return "\n".join(lines).strip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a source-level risk_violations breakdown report.")
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--artifact-ttl-hours", type=float, default=72.0)
    parser.add_argument("--artifact-keep", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    review_dir = args.review_dir.expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.config.expanduser().resolve()

    blocker_report_path = find_latest(review_dir, "*_live_gate_blocker_report.json")
    ops_breakdown_path = find_latest(review_dir, "*_ops_live_gate_breakdown.json")
    if blocker_report_path is None or ops_breakdown_path is None:
        raise SystemExit("required blocker artifacts are missing")

    payload = derive_payload(
        config_payload=load_config(config_path),
        blocker_report=load_json(blocker_report_path),
        blocker_report_path=blocker_report_path,
        ops_breakdown=load_json(ops_breakdown_path),
        ops_breakdown_path=ops_breakdown_path,
    )
    stamp = now_utc().strftime("%Y%m%dT%H%M%SZ")
    artifact = review_dir / f"{stamp}_risk_violations_breakdown.json"
    markdown = review_dir / f"{stamp}_risk_violations_breakdown.md"
    checksum = review_dir / f"{stamp}_risk_violations_breakdown_checksum.json"

    artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown.write_text(render_markdown(payload), encoding="utf-8")
    checksum_payload = {
        "artifact": str(artifact),
        "artifact_sha256": sha256_file(artifact),
        "markdown": str(markdown),
        "markdown_sha256": sha256_file(markdown),
        "generated_at": fmt_utc(now_utc()),
    }
    checksum.write_text(json.dumps(checksum_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pruned_keep, pruned_age = prune_artifacts(
        review_dir,
        current_artifact=artifact,
        current_markdown=markdown,
        current_checksum=checksum,
        keep=max(3, int(args.artifact_keep)),
        ttl_hours=max(1.0, float(args.artifact_ttl_hours)),
    )
    payload["artifact"] = str(artifact)
    payload["markdown"] = str(markdown)
    payload["checksum"] = str(checksum)
    payload["pruned_keep"] = pruned_keep
    payload["pruned_age"] = pruned_age
    artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
