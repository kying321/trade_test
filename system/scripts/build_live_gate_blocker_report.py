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


def find_latest(review_dir: Path, pattern: str) -> Path | None:
    files = sorted(
        review_dir.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def select_research_artifact(review_dir: Path) -> Path | None:
    files = sorted(
        review_dir.glob("*_hot_universe_research.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    fallback = files[0] if files else None
    for path in files:
        try:
            payload = load_json_mapping(path)
        except Exception:
            continue
        ladder = payload.get("research_action_ladder", {})
        if not isinstance(ladder, dict):
            continue
        if any(
            isinstance(ladder.get(key), list) and len(ladder.get(key, [])) > 0
            for key in (
                "focus_primary_batches",
                "focus_with_regime_filter_batches",
                "shadow_only_batches",
                "focus_now_batches",
            )
        ):
            return path
    return fallback


def prune_artifacts(
    review_dir: Path,
    *,
    current_artifact: Path,
    current_markdown: Path,
    current_checksum: Path,
    keep: int,
    ttl_hours: float,
) -> tuple[list[str], list[str]]:
    review_dir.mkdir(parents=True, exist_ok=True)
    cutoff = now_utc() - dt.timedelta(hours=max(1.0, ttl_hours))
    protected = {current_artifact.name, current_markdown.name, current_checksum.name}
    candidates: list[Path] = []
    for pattern in (
        "*_live_gate_blocker_report.json",
        "*_live_gate_blocker_report.md",
        "*_live_gate_blocker_report_checksum.json",
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


def _unwrap_operator_handoff(handoff_payload: dict[str, Any]) -> dict[str, Any]:
    nested = handoff_payload.get("operator_handoff")
    return nested if isinstance(nested, dict) else handoff_payload


def _takeover_signal_selection(ready_check: dict[str, Any]) -> dict[str, Any]:
    guarded_exec = ready_check.get("guarded_exec", {})
    if not isinstance(guarded_exec, dict):
        return {}
    takeover = guarded_exec.get("takeover", {})
    if not isinstance(takeover, dict):
        return {}
    payload = takeover.get("payload", {})
    if not isinstance(payload, dict):
        return {}
    steps = payload.get("steps", {})
    if not isinstance(steps, dict):
        return {}
    signal_selection = steps.get("signal_selection", {})
    return signal_selection if isinstance(signal_selection, dict) else {}


def _collect_leader_symbols(research_payload: dict[str, Any], batches: list[str]) -> list[str]:
    regime_playbook = research_payload.get("regime_playbook", {})
    if not isinstance(regime_playbook, dict):
        return []
    batch_rules = regime_playbook.get("batch_rules", [])
    if not isinstance(batch_rules, list):
        return []
    leaders: list[str] = []
    seen: set[str] = set()
    wanted = set(batches)
    for row in batch_rules:
        if not isinstance(row, dict):
            continue
        if str(row.get("batch", "")).strip() not in wanted:
            continue
        for symbol in row.get("leader_symbols", []):
            tag = str(symbol).strip().upper()
            if tag and tag not in seen:
                seen.add(tag)
                leaders.append(tag)
    return leaders


def derive_report(
    handoff_payload: dict[str, Any],
    research_payload: dict[str, Any],
    *,
    handoff_path: Path,
    research_path: Path,
) -> dict[str, Any]:
    operator = _unwrap_operator_handoff(handoff_payload)
    ready_check = handoff_payload.get("ready_check", {})
    if not isinstance(ready_check, dict):
        ready_check = {}
    ops_live_gate = ready_check.get("ops_live_gate", {})
    if not isinstance(ops_live_gate, dict):
        ops_live_gate = {}
    risk_guard = ready_check.get("risk_guard", {})
    if not isinstance(risk_guard, dict):
        risk_guard = {}
    signal_selection = _takeover_signal_selection(ready_check)
    blocked_candidate = signal_selection.get("blocked_candidate", {})
    if not isinstance(blocked_candidate, dict):
        blocked_candidate = {}

    action_ladder = research_payload.get("research_action_ladder", {})
    if not isinstance(action_ladder, dict):
        action_ladder = {}
    focus_primary = [
        str(x).strip()
        for x in action_ladder.get("focus_primary_batches", [])
        if str(x).strip()
    ]
    focus_regime = [
        str(x).strip()
        for x in action_ladder.get("focus_with_regime_filter_batches", [])
        if str(x).strip()
    ]
    shadow_only = [
        str(x).strip()
        for x in action_ladder.get("shadow_only_batches", [])
        if str(x).strip()
    ]
    avoid_batches = [
        str(x).strip()
        for x in action_ladder.get("avoid_batches", [])
        if str(x).strip()
    ]
    focus_now = [
        str(x).strip()
        for x in action_ladder.get("focus_now_batches", [])
        if str(x).strip()
    ]

    commodity_focus = bool(focus_now) and all("crypto" not in item for item in focus_now)
    leaders_primary = _collect_leader_symbols(research_payload, focus_primary)
    leaders_regime = _collect_leader_symbols(research_payload, focus_regime)

    live_ready = bool(operator.get("ready", False))
    ops_gate_ok = bool(ops_live_gate.get("ok", True))
    risk_guard_reasons = [
        str(x)
        for x in (risk_guard.get("reasons", []) if isinstance(risk_guard.get("reasons", []), list) else [])
        if str(x).strip()
    ]
    gate_blockers = [
        str(x)
        for x in (
            ops_live_gate.get("blocking_reason_codes", [])
            if isinstance(ops_live_gate.get("blocking_reason_codes", []), list)
            else []
        )
        if str(x).strip()
    ]
    rollback_codes = [
        str(x)
        for x in (
            ops_live_gate.get("rollback_reason_codes", [])
            if isinstance(ops_live_gate.get("rollback_reason_codes", []), list)
            else []
        )
        if str(x).strip()
    ]

    blockers = [
        {
            "name": "ops_live_gate",
            "priority": 1,
            "status": "blocked" if not ops_gate_ok else "clear",
            "reason_codes": gate_blockers,
            "rollback_level": str(ops_live_gate.get("rollback_level", "") or ""),
            "rollback_action": str(ops_live_gate.get("rollback_action", "") or ""),
            "failed_checks": list(ops_live_gate.get("gate_failed_checks", []))
            if isinstance(ops_live_gate.get("gate_failed_checks"), list)
            else [],
        },
        {
            "name": "risk_guard",
            "priority": 2,
            "status": "blocked" if risk_guard_reasons else "clear",
            "reason_codes": risk_guard_reasons,
            "blocked_candidate": blocked_candidate,
        },
        {
            "name": "alpha_execution_mismatch",
            "priority": 3,
            "status": "active" if commodity_focus else "inactive",
            "reason_codes": ["commodity_focus_vs_crypto_spot_live"] if commodity_focus else [],
            "focus_primary_batches": focus_primary,
            "focus_regime_filter_batches": focus_regime,
            "shadow_only_batches": shadow_only,
        },
    ]

    repair_sequence = [
        {
            "priority": 1,
            "area": "ops_live_gate",
            "goal": "Clear hard rollback state before any live capital increase.",
            "actions": [
                "Inspect ops reconcile and failed checks.",
                "Reduce or reset the causes behind risk_violations / max_drawdown / slot_anomaly.",
                "Do not lift live routing until ops_live_gate.ok becomes true.",
            ],
            "reason_codes": rollback_codes or gate_blockers,
            "command": str(operator.get("next_focus_command", "") or ""),
        },
        {
            "priority": 2,
            "area": "risk_guard",
            "goal": "Recover actionable tickets with current, executable crypto signals.",
            "actions": [
                "Regenerate signal-to-order tickets with fresh market data.",
                "Drop stale or under-min-notional candidates.",
                "Keep crypto queue at watch/pilot level until micro quality recovers.",
            ],
            "reason_codes": risk_guard_reasons,
            "candidate": blocked_candidate,
            "command": str(operator.get("secondary_focus_command", "") or ""),
        },
        {
            "priority": 3,
            "area": "commodity_execution_path",
            "goal": "Align validated alpha sleeves with a paper-first commodity execution lane.",
            "actions": [
                "Build sleeve-level paper execution for metals_all / precious_metals.",
                "Add regime filter for energy_liquids before any simulated routing.",
                "Treat commodities_benchmark as shadow-only, not a primary live sleeve.",
            ],
            "focus_now_batches": focus_now,
            "leader_symbols": leaders_primary + [x for x in leaders_regime if x not in leaders_primary],
        },
    ]

    commodity_execution_path = {
        "design_status": "proposed",
        "execution_mode": "paper_first",
        "focus_primary_batches": focus_primary,
        "focus_with_regime_filter_batches": focus_regime,
        "shadow_only_batches": shadow_only,
        "avoid_batches": avoid_batches,
        "leader_symbols_primary": leaders_primary,
        "leader_symbols_regime_filter": leaders_regime,
        "stages": [
            {
                "stage": "research_sleeves",
                "batches": focus_primary,
                "rule": "Use trend-only sleeves first; keep shadow sleeves out of primary capital allocation.",
            },
            {
                "stage": "paper_ticket_lane",
                "batches": focus_primary + focus_regime,
                "rule": "Emit commodity tickets with the same ticket/risk schema used by live crypto, but route only to paper execution.",
            },
            {
                "stage": "regime_filter",
                "batches": focus_regime,
                "rule": "For energy_liquids, only allow strong-trend states; explicitly veto ranging states.",
            },
            {
                "stage": "sleeve_shadow",
                "batches": shadow_only,
                "rule": "Track for confirmation and attribution only; do not allocate as a primary sleeve.",
            },
        ],
    }

    live_decision = {
        "formal_live_ready": live_ready and ops_gate_ok and not risk_guard_reasons,
        "micro_canary_only": bool(live_ready and ops_gate_ok and not risk_guard_reasons),
        "current_decision": "do_not_start_formal_live" if (not live_ready or not ops_gate_ok or risk_guard_reasons) else "formal_live_possible",
        "summary": (
            "Do not start formal live trading yet. Clear ops_live_gate, then recover actionable tickets, then solve commodity-vs-venue mismatch."
            if (not live_ready or not ops_gate_ok or risk_guard_reasons)
            else "Formal live is structurally possible."
        ),
    }

    return {
        "generated_at": fmt_utc(now_utc()),
        "handoff_source": str(handoff_path),
        "research_source": str(research_path),
        "live_decision": live_decision,
        "operator_status_triplet": str(operator.get("operator_status_triplet", "") or ""),
        "operator_status_quad": str(operator.get("operator_status_quad", "") or ""),
        "next_focus_area": str(operator.get("next_focus_area", "") or ""),
        "next_focus_reason": str(operator.get("next_focus_reason", "") or ""),
        "secondary_focus_area": str(operator.get("secondary_focus_area", "") or ""),
        "secondary_focus_reason": str(operator.get("secondary_focus_reason", "") or ""),
        "blockers": blockers,
        "repair_sequence": repair_sequence,
        "commodity_execution_path": commodity_execution_path,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    live_decision = payload.get("live_decision", {})
    blockers = payload.get("blockers", [])
    repair_sequence = payload.get("repair_sequence", [])
    path = payload.get("commodity_execution_path", {})
    lines = [
        "# Live Gate Blocker Report",
        "",
        f"- handoff source: `{payload.get('handoff_source', '')}`",
        f"- research source: `{payload.get('research_source', '')}`",
        f"- decision: `{live_decision.get('current_decision', '')}`",
        f"- summary: {live_decision.get('summary', '')}",
        f"- status: `{payload.get('operator_status_quad') or payload.get('operator_status_triplet') or ''}`",
        f"- focus stack: `{payload.get('next_focus_area', '')} -> {payload.get('secondary_focus_area', '')}`",
        "",
        "## Blockers",
    ]
    for row in blockers:
        if not isinstance(row, dict):
            continue
        lines.append(f"- `{row.get('name', '')}`")
        lines.append(f"  - priority: `{row.get('priority', '')}`")
        lines.append(f"  - status: `{row.get('status', '')}`")
        reasons = row.get("reason_codes", [])
        lines.append(f"  - reasons: `{', '.join(reasons) if isinstance(reasons, list) and reasons else '-'}`")
        blocked_candidate = row.get("blocked_candidate", {})
        if isinstance(blocked_candidate, dict) and blocked_candidate:
            lines.append(
                "  - blocked candidate: "
                + f"`{blocked_candidate.get('symbol', '')}` / reasons=`{', '.join(blocked_candidate.get('ticket_reasons', []))}`"
            )
    lines.extend(["", "## Repair Sequence"])
    for row in repair_sequence:
        if not isinstance(row, dict):
            continue
        lines.append(f"- `{row.get('area', '')}`")
        lines.append(f"  - priority: `{row.get('priority', '')}`")
        lines.append(f"  - goal: {row.get('goal', '')}")
        cmd = str(row.get("command", "") or "").strip()
        if cmd:
            lines.append(f"  - command: `{cmd}`")
    lines.extend(["", "## Commodity Execution Path"])
    lines.append(f"- mode: `{path.get('execution_mode', '')}`")
    lines.append(f"- focus primary: `{', '.join(path.get('focus_primary_batches', [])) or '-'}`")
    lines.append(f"- focus regime filter: `{', '.join(path.get('focus_with_regime_filter_batches', [])) or '-'}`")
    lines.append(f"- shadow only: `{', '.join(path.get('shadow_only_batches', [])) or '-'}`")
    lines.append(f"- avoid: `{', '.join(path.get('avoid_batches', [])) or '-'}`")
    lines.append(f"- leader symbols: `{', '.join(path.get('leader_symbols_primary', []) + path.get('leader_symbols_regime_filter', [])) or '-'}`")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a live gate blocker + commodity path report.")
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--handoff-json", type=Path, default=None)
    parser.add_argument("--research-json", type=Path, default=None)
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--artifact-keep", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    review_dir = args.review_dir.expanduser().resolve()
    handoff_path = args.handoff_json.expanduser().resolve() if args.handoff_json else find_latest(review_dir, "*_remote_live_handoff.json")
    research_path = args.research_json.expanduser().resolve() if args.research_json else select_research_artifact(review_dir)
    if handoff_path is None:
        raise SystemExit("remote live handoff artifact not found")
    if research_path is None:
        raise SystemExit("hot universe research artifact not found")

    handoff_payload = load_json_mapping(handoff_path)
    research_payload = load_json_mapping(research_path)
    payload = derive_report(handoff_payload, research_payload, handoff_path=handoff_path, research_path=research_path)

    stamp = now_utc().strftime("%Y%m%dT%H%M%SZ")
    artifact_path = review_dir / f"{stamp}_live_gate_blocker_report.json"
    markdown_path = review_dir / f"{stamp}_live_gate_blocker_report.md"
    checksum_path = review_dir / f"{stamp}_live_gate_blocker_report_checksum.json"
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
