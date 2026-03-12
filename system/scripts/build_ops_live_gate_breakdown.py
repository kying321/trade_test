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
        "*_ops_live_gate_breakdown.json",
        "*_ops_live_gate_breakdown.md",
        "*_ops_live_gate_breakdown_checksum.json",
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


def build_breakdown(handoff_payload: dict[str, Any], *, handoff_path: Path) -> dict[str, Any]:
    ready_check = handoff_payload.get("ready_check", {})
    if not isinstance(ready_check, dict):
        ready_check = {}
    ops_live_gate = ready_check.get("ops_live_gate", {})
    if not isinstance(ops_live_gate, dict):
        ops_live_gate = {}
    gate_failed_checks = [
        str(x)
        for x in (ops_live_gate.get("gate_failed_checks", []) if isinstance(ops_live_gate.get("gate_failed_checks", []), list) else [])
        if str(x).strip()
    ]
    blocking_codes = [
        str(x)
        for x in (ops_live_gate.get("blocking_reason_codes", []) if isinstance(ops_live_gate.get("blocking_reason_codes", []), list) else [])
        if str(x).strip()
    ]
    rollback_codes = [
        str(x)
        for x in (ops_live_gate.get("rollback_reason_codes", []) if isinstance(ops_live_gate.get("rollback_reason_codes", []), list) else [])
        if str(x).strip()
    ]

    root_causes = []
    if "risk_violations" in blocking_codes:
        root_causes.append(
            {
                "code": "risk_violations",
                "priority": 1,
                "kind": "root",
                "source_check": "risk_violations_ok",
                "source_ref": "/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/orchestration/release.py:649",
                "rollback_ref": "/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/orchestration/release.py:4714",
                "fix_action": "校验单笔/总暴露约束计算与执行器拦截逻辑。",
            }
        )
    if "max_drawdown" in blocking_codes:
        root_causes.append(
            {
                "code": "max_drawdown",
                "priority": 2,
                "kind": "root",
                "source_check": "max_drawdown_ok",
                "source_ref": "/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/orchestration/release.py:648",
                "rollback_ref": "/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/orchestration/release.py:4717",
                "fix_action": "下调风险预算并加严止损/保护模式触发阈值。",
            }
        )
    if "slot_anomaly" in blocking_codes:
        root_causes.append(
            {
                "code": "slot_anomaly",
                "priority": 3,
                "kind": "root",
                "source_check": "slot_anomaly_ok",
                "source_ref": "/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/orchestration/release.py:554",
                "rollback_ref": "/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/orchestration/release.py:4744",
                "fix_action": "优先修复缺失/异常槽位数据后再恢复正常评审节奏。",
            }
        )

    derived_wrappers = []
    if "rollback_hard" in blocking_codes:
        derived_wrappers.append(
            {
                "code": "rollback_hard",
                "kind": "derived",
                "source_ref": "/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/orchestration/release.py:862",
                "meaning": "硬回滚级别标签，本身不是独立根因，由 rollback_reason_codes 推导。",
                "depends_on": rollback_codes,
            }
        )
    if "ops_status_red" in blocking_codes:
        derived_wrappers.append(
            {
                "code": "ops_status_red",
                "kind": "derived",
                "source_ref": "/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/orchestration/release.py:919",
                "meaning": "由于已有 blocker 且 ops_status=red 追加的状态标签，本身不是独立根因。",
                "depends_on": blocking_codes,
            }
        )

    secondary_checks = []
    secondary_map = {
        "health_ok": ("health_degraded", "输出工件不完整，先补跑缺失槽位。"),
        "positive_window_ratio_ok": ("positive_window_ratio", "样本外正收益窗口占比不达标，先收缩信号阈值。"),
        "review_pass_gate": ("review_gate", "复盘门槛未通过，先回看参数更新幅度和因子贡献。"),
        "stable_replay_ok": ("stable_replay", "连续回放稳定性未达标，先修复 replay 失败日期。"),
        "stress_autorun_reason_drift_ok": ("stress_autorun_reason_drift", "压力自动运行原因漂移超限，先收敛自动 rerun 触发原因。"),
        "unresolved_conflict_ok": ("data_conflict", "跨源冲突未决比例超限，先补冲突仲裁规则。"),
    }
    for check in gate_failed_checks:
        if check in {"risk_violations_ok", "max_drawdown_ok", "slot_anomaly_ok", "state_stability_ok"}:
            continue
        mapped = secondary_map.get(check)
        if not mapped:
            continue
        code, action = mapped
        secondary_checks.append(
            {
                "source_check": check,
                "code": code,
                "kind": "secondary",
                "action": action,
            }
        )

    repair_order = [
        {
            "priority": 1,
            "group": "root_causes",
            "codes": [row["code"] for row in root_causes],
            "summary": "先修根因，rollback_hard 和 ops_status_red 会跟着消失。",
        },
        {
            "priority": 2,
            "group": "secondary_checks",
            "codes": [row["code"] for row in secondary_checks],
            "summary": "根因清掉后，再收 secondary gate checks，降低复发概率。",
        },
        {
            "priority": 3,
            "group": "risk_guard",
            "codes": ["ticket_missing:no_actionable_ticket"],
            "summary": "ops_live_gate 清掉后，再处理 actionable ticket 缺失。",
        },
    ]

    return {
        "generated_at": fmt_utc(now_utc()),
        "handoff_source": str(handoff_path),
        "blocking_reason_codes": blocking_codes,
        "rollback_reason_codes": rollback_codes,
        "gate_failed_checks": gate_failed_checks,
        "root_causes": root_causes,
        "derived_wrappers": derived_wrappers,
        "secondary_checks": secondary_checks,
        "repair_order": repair_order,
        "takeaway": "Current ops_live_gate blockers collapse to 3 root causes, 2 derived wrappers, and 6 secondary failed checks.",
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Ops Live Gate Breakdown",
        "",
        f"- handoff source: `{payload.get('handoff_source', '')}`",
        f"- blocker codes: `{', '.join(payload.get('blocking_reason_codes', [])) or '-'}`",
        f"- rollback codes: `{', '.join(payload.get('rollback_reason_codes', [])) or '-'}`",
        f"- takeaway: {payload.get('takeaway', '')}",
        "",
        "## Root Causes",
    ]
    for row in payload.get("root_causes", []):
        lines.append(f"- `{row.get('code', '')}`")
        lines.append(f"  - source check: `{row.get('source_check', '')}`")
        lines.append(f"  - source ref: `{row.get('source_ref', '')}`")
        lines.append(f"  - rollback ref: `{row.get('rollback_ref', '')}`")
        lines.append(f"  - fix: {row.get('fix_action', '')}")
    lines.extend(["", "## Derived Wrappers"])
    for row in payload.get("derived_wrappers", []):
        lines.append(f"- `{row.get('code', '')}`")
        lines.append(f"  - source ref: `{row.get('source_ref', '')}`")
        lines.append(f"  - meaning: {row.get('meaning', '')}")
    lines.extend(["", "## Secondary Checks"])
    for row in payload.get("secondary_checks", []):
        lines.append(f"- `{row.get('source_check', '')}` -> `{row.get('code', '')}`")
        lines.append(f"  - action: {row.get('action', '')}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ops_live_gate blocker breakdown report.")
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--handoff-json", type=Path, default=None)
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--artifact-keep", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    review_dir = args.review_dir.expanduser().resolve()
    handoff_path = args.handoff_json.expanduser().resolve() if args.handoff_json else find_latest(review_dir, "*_remote_live_handoff.json")
    if handoff_path is None:
        raise SystemExit("remote live handoff artifact not found")
    handoff_payload = load_json(handoff_path)
    payload = build_breakdown(handoff_payload, handoff_path=handoff_path)

    stamp = now_utc().strftime("%Y%m%dT%H%M%SZ")
    artifact_path = review_dir / f"{stamp}_ops_live_gate_breakdown.json"
    markdown_path = review_dir / f"{stamp}_ops_live_gate_breakdown.md"
    checksum_path = review_dir / f"{stamp}_ops_live_gate_breakdown_checksum.json"
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
