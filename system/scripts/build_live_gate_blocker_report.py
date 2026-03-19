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
    for path in files:
        try:
            load_json_mapping(path)
        except Exception:
            continue
        return path
    return None


def select_aligned_supporting_artifact(
    review_dir: Path,
    *,
    pattern: str,
    expected_source_field: str,
    expected_source_path: Path,
) -> tuple[Path | None, dict[str, Any]]:
    files = sorted(
        review_dir.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    expected_source = str(expected_source_path)
    for path in files:
        try:
            payload = load_json_mapping(path)
        except Exception:
            continue
        source_text = str(payload.get(expected_source_field) or "").strip()
        if source_text != expected_source:
            continue
        return path, payload
    return None, {}


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


def _dedupe_reason_codes(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


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


def _positive_profitability_window(remote_live_history: dict[str, Any]) -> tuple[str, float, int] | None:
    candidates = (
        ("30d", remote_live_history.get("pnl_30d"), remote_live_history.get("trade_count_30d")),
        ("7d", remote_live_history.get("pnl_7d"), remote_live_history.get("trade_count_7d")),
        ("24h", remote_live_history.get("pnl_24h"), remote_live_history.get("trade_count_24h")),
    )
    for label, pnl_raw, trades_raw in candidates:
        try:
            pnl = float(pnl_raw)
        except Exception:
            continue
        try:
            trades = int(trades_raw or 0)
        except Exception:
            trades = 0
        if pnl > 0.0 and trades > 0:
            return (label, pnl, trades)
    return None


OPS_LIVE_GATE_CLEARING_HINTS: dict[str, str] = {
    "rollback_hard": "clear hard rollback so ops_live_gate can leave rollback_now state",
    "risk_violations": "clear active risk violations and restore risk checks to passing",
    "max_drawdown": "bring drawdown back below the configured cap",
    "slot_anomaly": "reconcile slot state until slot anomaly checks pass",
    "backtest_snapshot": "refresh the backtest snapshot until snapshot health returns to green",
    "ops_status_red": "restore ops status from red to a healthy non-red state",
}

RISK_GUARD_CLEARING_HINTS: dict[str, str] = {
    "ticket_missing:no_actionable_ticket": "generate at least one fresh actionable ticket that survives confidence and size filters",
    "panic_cooldown_active": "wait for or explicitly clear the active panic cooldown before automated routing resumes",
    "open_exposure_above_cap": "reduce or close exposure until open exposure is back under the configured cap",
    "confidence_below_threshold": "promote a candidate whose confidence clears the active threshold",
    "size_below_min_notional": "raise effective order size or select a symbol that clears min notional constraints",
}

ACCOUNT_SCOPE_ALIGNMENT_CLEARING_HINTS: dict[str, str] = {
    "spot_remote_lane_missing": "implement a spot-native remote executable lane or stop treating spot readiness as the target execution contract",
    "portfolio_margin_um_read_only_mode": "promote portfolio_margin_um from read-only diagnostics into an explicit live execution contract before routing capital",
}

REMOTE_EXECUTION_CONTRACT_CLEARING_HINTS: dict[str, str] = {
    "shadow_executor_only_mode": "promote the remote executor from shadow_guarded into an explicit send/ack/fill contract before routing capital",
    "guarded_probe_only_mode": "guarded probe is available, but live capital still requires a source-owned send/ack/fill runtime before routing capital",
    "requested_executor_mode_not_implemented": "implement the requested non-shadow executor runtime and prove send/ack/fill behavior before routing capital",
    "unsupported_executor_mode_source": "set a supported executor mode source or implement runtime support for the requested mode before routing capital",
    "spot_remote_lane_missing": "implement a spot-native remote executable lane or stop treating spot readiness as the target execution contract",
    "portfolio_margin_um_read_only_mode": "promote portfolio_margin_um from read-only diagnostics into an explicit live execution contract before routing capital",
}


def _is_remote_live_executable_head(*, area: str, action: str, remote_market: str) -> bool:
    area_text = str(area or "").strip()
    action_text = str(action or "").strip()
    remote_market_text = str(remote_market or "").strip().lower()
    if not remote_market_text:
        return False
    if area_text != "crypto_route":
        return False
    if not action_text:
        return False
    return True


def _build_clearing_conditions(
    *,
    area: str,
    reason_codes: list[str],
    hint_map: dict[str, str],
) -> list[dict[str, Any]]:
    conditions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for code in reason_codes:
        text = str(code).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        clear_when = hint_map.get(text) or f"clear blocking condition `{text}`"
        conditions.append(
            {
                "area": area,
                "code": text,
                "clear_when": clear_when,
            }
        )
    return conditions


def _conditions_brief(conditions: list[dict[str, Any]]) -> str:
    return ", ".join(
        [str(row.get("code") or "").strip() for row in conditions if str(row.get("code") or "").strip()]
    ) or "-"


def derive_account_scope_alignment_reason_codes(
    *,
    account_scope_alignment: dict[str, Any],
    remote_live_context: dict[str, Any],
) -> list[str]:
    account_scope = account_scope_alignment if isinstance(account_scope_alignment, dict) else {}
    remote_context = remote_live_context if isinstance(remote_live_context, dict) else {}
    status = str(account_scope.get("status") or "").strip()
    remote_market = str(remote_context.get("market") or "").strip().lower()
    ready_scope_market = str(remote_context.get("ready_check_scope_market") or "").strip().lower()
    if status != "split_scope_spot_vs_portfolio_margin_um":
        return []
    if remote_market != "portfolio_margin_um":
        return []
    if ready_scope_market == "spot":
        return []
    reason_codes = ["spot_remote_lane_missing"]
    if ready_scope_market == "portfolio_margin_um":
        reason_codes.append("portfolio_margin_um_read_only_mode")
    return _dedupe_reason_codes(reason_codes)


def derive_ops_live_gate_breakdown_summary(
    *,
    ops_live_gate_breakdown_payload: dict[str, Any] | None,
    ops_live_gate_breakdown_path: Path | None,
) -> dict[str, Any]:
    payload = ops_live_gate_breakdown_payload if isinstance(ops_live_gate_breakdown_payload, dict) else {}
    root_causes = payload.get("root_causes", [])
    derived_wrappers = payload.get("derived_wrappers", [])
    secondary_checks = payload.get("secondary_checks", [])
    root_codes = [
        str(row.get("code") or "").strip()
        for row in root_causes
        if isinstance(row, dict) and str(row.get("code") or "").strip()
    ]
    wrapper_codes = [
        str(row.get("code") or "").strip()
        for row in derived_wrappers
        if isinstance(row, dict) and str(row.get("code") or "").strip()
    ]
    secondary_codes = [
        str(row.get("code") or "").strip()
        for row in secondary_checks
        if isinstance(row, dict) and str(row.get("code") or "").strip()
    ]
    if not payload:
        return {
            "status": "unavailable",
            "artifact": "",
            "brief": "unavailable:-",
            "primary_root_cause_code": "",
            "primary_root_cause_fix_action": "",
            "root_cause_codes": [],
            "derived_wrapper_codes": [],
            "secondary_codes": [],
        }
    brief = (
        f"roots={','.join(root_codes) or '-'} | "
        f"wrappers={','.join(wrapper_codes) or '-'} | "
        f"secondary={','.join(secondary_codes) or '-'}"
    )
    return {
        "status": "root_causes_identified" if root_codes else "review_required",
        "artifact": str(ops_live_gate_breakdown_path) if ops_live_gate_breakdown_path else "",
        "brief": brief,
        "primary_root_cause_code": str(payload.get("primary_root_cause_code") or "").strip(),
        "primary_root_cause_fix_action": str(payload.get("primary_root_cause_fix_action") or "").strip(),
        "root_cause_codes": root_codes,
        "derived_wrapper_codes": wrapper_codes,
        "secondary_codes": secondary_codes,
    }


def derive_slot_anomaly_breakdown_summary(
    *,
    slot_anomaly_breakdown_payload: dict[str, Any] | None,
    slot_anomaly_breakdown_path: Path | None,
) -> dict[str, Any]:
    payload = slot_anomaly_breakdown_payload if isinstance(slot_anomaly_breakdown_payload, dict) else {}
    if not payload:
        return {
            "status": "unavailable",
            "artifact": "",
            "brief": "unavailable:-",
            "repair_focus": "",
            "payload_gap": "",
        }
    return {
        "status": str(payload.get("status") or "").strip() or "review_required",
        "artifact": str(slot_anomaly_breakdown_path) if slot_anomaly_breakdown_path else "",
        "brief": str(payload.get("brief") or "").strip(),
        "repair_focus": str(payload.get("repair_focus") or "").strip(),
        "payload_gap": str(payload.get("payload_gap") or "").strip(),
    }


def _as_float(raw: Any) -> float | None:
    try:
        return float(raw)
    except Exception:
        return None


def _freshness_status(
    *,
    age_value: float | None,
    fresh_threshold: float,
    stale_threshold: float | None = None,
) -> str:
    if age_value is None:
        return "unknown"
    if age_value <= fresh_threshold:
        return "fresh"
    if stale_threshold is not None and age_value > stale_threshold:
        return "stale"
    return "aging"


def derive_source_freshness(ready_check: dict[str, Any]) -> dict[str, Any]:
    guarded_exec = dict(ready_check.get("guarded_exec") or {})
    takeover_payload = dict(dict(guarded_exec.get("takeover") or {}).get("payload") or {})
    takeover_steps = dict(takeover_payload.get("steps") or {})
    takeover_risk_guard = dict(takeover_steps.get("risk_guard") or {})
    ops_reconcile = dict(ready_check.get("ops_reconcile") or {})
    risk_guard = dict(ready_check.get("risk_guard") or {})

    ops_artifact_age_hours = _as_float(ops_reconcile.get("artifact_age_hours"))
    ops_max_age_hours = _as_float(ops_reconcile.get("max_age_hours"))
    ops_status = _freshness_status(
        age_value=ops_artifact_age_hours,
        fresh_threshold=1.0,
        stale_threshold=ops_max_age_hours,
    )

    risk_guard_age_seconds = _as_float(risk_guard.get("fuse_age_seconds"))
    if risk_guard_age_seconds is None:
        risk_guard_age_seconds = _as_float(takeover_risk_guard.get("age_seconds"))
    risk_guard_fresh = risk_guard.get("fresh")
    if risk_guard_fresh is None:
        risk_guard_fresh = takeover_risk_guard.get("fresh")
    if isinstance(risk_guard_fresh, bool):
        risk_guard_status = "fresh" if risk_guard_fresh else "stale"
    else:
        risk_guard_status = _freshness_status(
            age_value=risk_guard_age_seconds,
            fresh_threshold=120.0,
            stale_threshold=900.0,
        )

    ops_artifact = str(ops_reconcile.get("artifact_path") or "").strip()
    risk_guard_artifact = str(risk_guard.get("fuse_artifact") or "").strip()
    if not risk_guard_artifact:
        risk_guard_artifact = str(takeover_risk_guard.get("artifact") or "").strip()

    ops_brief = (
        f"ops_reconcile={ops_status}:{ops_artifact_age_hours:.3f}h"
        if ops_artifact_age_hours is not None
        else "ops_reconcile=unknown"
    )
    risk_brief = (
        f"risk_guard={risk_guard_status}:{risk_guard_age_seconds:.1f}s"
        if risk_guard_age_seconds is not None
        else "risk_guard=unknown"
    )
    return {
        "ops_reconcile_status": ops_status,
        "ops_reconcile_artifact": ops_artifact,
        "ops_reconcile_artifact_age_hours": ops_artifact_age_hours,
        "ops_reconcile_artifact_mtime_utc": str(ops_reconcile.get("artifact_mtime_utc") or "").strip(),
        "ops_reconcile_max_age_hours": ops_max_age_hours,
        "risk_guard_status": risk_guard_status,
        "risk_guard_artifact": risk_guard_artifact,
        "risk_guard_generated_at_utc": str(
            risk_guard.get("generated_at_utc") or takeover_risk_guard.get("generated_at_utc") or ""
        ).strip(),
        "risk_guard_age_seconds": risk_guard_age_seconds,
        "risk_guard_fresh": bool(risk_guard_fresh) if isinstance(risk_guard_fresh, bool) else None,
        "brief": f"{ops_brief} | {risk_brief}",
    }


def derive_clearing_lane(
    *,
    area: str,
    reason_codes: list[str],
    hint_map: dict[str, str],
) -> dict[str, Any]:
    conditions = _build_clearing_conditions(area=area, reason_codes=reason_codes, hint_map=hint_map)
    if not conditions:
        return {
            "status": "clear",
            "brief": f"clear:{area}",
            "count": 0,
            "reason_codes": [],
            "conditions": [],
            "blocker_detail": f"{area} has no active clearing conditions.",
            "done_when": f"keep {area} clear",
        }

    codes_brief = _conditions_brief(conditions)
    blocker_detail = "; ".join(
        [f"{row['code']} -> {row['clear_when']}" for row in conditions]
    )
    done_when = (
        f"{area} clears {len(conditions)} blocking condition(s): {codes_brief}"
    )
    return {
        "status": "clearing_required",
        "brief": f"clearing_required:{area}:{len(conditions)}",
        "count": len(conditions),
        "reason_codes": [str(row.get("code") or "") for row in conditions],
        "conditions": conditions,
        "conditions_brief": codes_brief,
        "blocker_detail": blocker_detail,
        "done_when": done_when,
    }


def derive_remote_live_takeover_clearing(
    *,
    ops_live_gate_clearing: dict[str, Any],
    risk_guard_clearing: dict[str, Any],
) -> dict[str, Any]:
    ops_blocked = str(ops_live_gate_clearing.get("status") or "") == "clearing_required"
    risk_blocked = str(risk_guard_clearing.get("status") or "") == "clearing_required"
    blocked_layers = [
        layer
        for layer, enabled in (
            ("ops_live_gate", ops_blocked),
            ("risk_guard", risk_blocked),
        )
        if enabled
    ]
    if not blocked_layers:
        return {
            "status": "clear",
            "brief": "clear:remote_live_takeover",
            "blocking_layers": [],
            "blocker_detail": "No active remote live takeover clearing conditions remain.",
            "done_when": "keep ops_live_gate and risk_guard clear",
        }

    blocker_parts: list[str] = []
    if ops_blocked:
        blocker_parts.append(
            f"ops_live_gate needs {str(ops_live_gate_clearing.get('conditions_brief') or '-').strip()}"
        )
    if risk_blocked:
        blocker_parts.append(
            f"risk_guard needs {str(risk_guard_clearing.get('conditions_brief') or '-').strip()}"
        )
    return {
        "status": "clearing_required",
        "brief": f"clearing_required:{'+'.join(blocked_layers)}",
        "blocking_layers": blocked_layers,
        "blocker_detail": "; ".join(blocker_parts),
        "done_when": "ops_live_gate becomes clear and risk_guard reasons become empty",
    }


def _derive_ops_live_gate_queue_from_breakdown(
    *,
    ops_live_gate_clearing: dict[str, Any],
    ops_live_gate_breakdown: dict[str, Any],
    command: str,
    goal: str,
    actions: list[str],
) -> list[dict[str, Any]]:
    if not isinstance(ops_live_gate_breakdown, dict) or not ops_live_gate_breakdown:
        return []
    conditions = (
        ops_live_gate_clearing.get("conditions", [])
        if isinstance(ops_live_gate_clearing.get("conditions", []), list)
        else []
    )
    clear_when_by_code = {
        str(row.get("code") or "").strip(): str(row.get("clear_when") or "").strip()
        for row in conditions
        if isinstance(row, dict) and str(row.get("code") or "").strip()
    }
    derived_wrapper_codes = {
        str(row.get("code") or "").strip()
        for row in ops_live_gate_breakdown.get("derived_wrappers", [])
        if isinstance(row, dict) and str(row.get("code") or "").strip()
    }
    represented_codes: set[str] = set()
    queue: list[dict[str, Any]] = []
    priority_score = 100

    def append_item(*, code: str, clear_when: str, action_name: str) -> None:
        nonlocal priority_score
        text = str(code).strip()
        if not text or text in represented_codes:
            return
        represented_codes.add(text)
        queue.append(
            {
                "area": "ops_live_gate",
                "code": text,
                "action": action_name,
                "priority_tier": "repair_queue_now",
                "priority_score": max(1, priority_score),
                "clear_when": str(clear_when).strip(),
                "goal": goal,
                "actions": list(actions),
                "command": command,
            }
        )
        priority_score -= 1

    for row in ops_live_gate_breakdown.get("root_causes", []):
        if not isinstance(row, dict):
            continue
        code = str(row.get("code") or "").strip()
        clear_when = str(row.get("fix_action") or "").strip() or clear_when_by_code.get(code, "")
        append_item(
            code=code,
            clear_when=clear_when,
            action_name="clear_ops_live_gate_root_cause",
        )

    for row in conditions:
        if not isinstance(row, dict):
            continue
        code = str(row.get("code") or "").strip()
        if not code or code in derived_wrapper_codes:
            continue
        append_item(
            code=code,
            clear_when=str(row.get("clear_when") or "").strip(),
            action_name="clear_ops_live_gate_condition",
        )

    for row in ops_live_gate_breakdown.get("secondary_checks", []):
        if not isinstance(row, dict):
            continue
        append_item(
            code=str(row.get("code") or "").strip(),
            clear_when=str(row.get("action") or "").strip(),
            action_name="clear_ops_live_gate_secondary_check",
        )

    return queue


def derive_remote_live_takeover_repair_queue(
    *,
    ops_live_gate_clearing: dict[str, Any],
    risk_guard_clearing: dict[str, Any],
    repair_sequence: list[dict[str, Any]],
    ops_live_gate_breakdown: dict[str, Any] | None = None,
) -> dict[str, Any]:
    command_by_area: dict[str, str] = {}
    goal_by_area: dict[str, str] = {}
    actions_by_area: dict[str, list[str]] = {}
    for row in repair_sequence:
        if not isinstance(row, dict):
            continue
        area = str(row.get("area") or "").strip()
        if not area:
            continue
        command_by_area[area] = str(row.get("command") or "").strip()
        goal_by_area[area] = str(row.get("goal") or "").strip()
        actions = row.get("actions", [])
        actions_by_area[area] = [str(x).strip() for x in actions if str(x).strip()] if isinstance(actions, list) else []

    queue: list[dict[str, Any]] = []
    ops_goal = goal_by_area.get("ops_live_gate", "")
    ops_actions = actions_by_area.get("ops_live_gate", [])
    ops_command = command_by_area.get("ops_live_gate", "")
    queue.extend(
        _derive_ops_live_gate_queue_from_breakdown(
            ops_live_gate_clearing=ops_live_gate_clearing,
            ops_live_gate_breakdown=ops_live_gate_breakdown or {},
            command=ops_command,
            goal=ops_goal,
            actions=ops_actions,
        )
    )
    raw_ops_fallback = not queue
    for area, lane, base_score in (
        ("ops_live_gate", ops_live_gate_clearing, 100),
        ("risk_guard", risk_guard_clearing, 90),
    ):
        if area == "ops_live_gate" and not raw_ops_fallback:
            continue
        conditions = lane.get("conditions", [])
        if not isinstance(conditions, list):
            continue
        for idx, row in enumerate(conditions, start=1):
            if not isinstance(row, dict):
                continue
            code = str(row.get("code") or "").strip()
            clear_when = str(row.get("clear_when") or "").strip()
            if not code:
                continue
            queue.append(
                {
                    "area": area,
                    "code": code,
                    "action": f"clear_{area}_condition",
                    "priority_tier": "repair_queue_now",
                    "priority_score": max(1, base_score - idx),
                    "clear_when": clear_when,
                    "goal": goal_by_area.get(area, ""),
                    "actions": actions_by_area.get(area, []),
                    "command": command_by_area.get(area, ""),
                }
            )

    queue.sort(
        key=lambda row: (
            0 if str(row.get("area") or "") == "ops_live_gate" else 1,
            -int(row.get("priority_score") or 0),
            str(row.get("code") or ""),
        )
    )
    for rank, row in enumerate(queue, start=1):
        row["rank"] = rank

    if not queue:
        return {
            "status": "inactive",
            "brief": "inactive:-",
            "count": 0,
            "head_area": "",
            "head_code": "",
            "head_action": "",
            "head_priority_score": 0,
            "head_priority_tier": "",
            "head_command": "",
            "head_clear_when": "",
            "queue_brief": "-",
            "items": [],
            "done_when": "remote live takeover clearing queue activates when blockers reappear",
        }

    head = queue[0]
    queue_brief = " | ".join(
        [
            f"{int(row.get('rank') or 0)}:{str(row.get('area') or '-')}"
            f":{str(row.get('code') or '-')}"
            f":{int(row.get('priority_score') or 0)}"
            for row in queue[:10]
        ]
    ) or "-"
    return {
        "status": "ready",
        "brief": f"ready:{str(head.get('area') or '-')}:{str(head.get('code') or '-')}:{int(head.get('priority_score') or 0)}",
        "count": len(queue),
        "head_area": str(head.get("area") or ""),
        "head_code": str(head.get("code") or ""),
        "head_action": str(head.get("action") or ""),
        "head_priority_score": int(head.get("priority_score") or 0),
        "head_priority_tier": str(head.get("priority_tier") or ""),
        "head_command": str(head.get("command") or ""),
        "head_clear_when": str(head.get("clear_when") or ""),
        "queue_brief": queue_brief,
        "items": queue,
        "done_when": "work through the queued remote live clearing conditions in rank order until the queue empties",
    }


def derive_remote_live_diagnosis(
    *,
    remote_live_history: dict[str, Any],
    account_scope_alignment: dict[str, Any],
    live_ready: bool,
    ops_gate_ok: bool,
    risk_guard_reasons: list[str],
) -> dict[str, Any]:
    market = str(remote_live_history.get("market") or "").strip() or "unknown"
    profitability_window = _positive_profitability_window(remote_live_history)
    profitability_confirmed = profitability_window is not None
    blocking_layers: list[str] = []
    if not ops_gate_ok:
        blocking_layers.append("ops_live_gate")
    if risk_guard_reasons:
        blocking_layers.append("risk_guard")
    if bool(account_scope_alignment.get("blocking", False)):
        blocking_layers.append("account_scope_alignment")

    if live_ready and not blocking_layers:
        status = "formal_live_possible"
        brief = f"formal_live_possible:{market}"
        blocker_detail = (
            "Remote account scope, live gate, and risk guard are aligned; formal live is structurally possible."
        )
        done_when = "maintain current account scope alignment and keep ops_live_gate and risk_guard clear"
    elif profitability_confirmed and blocking_layers:
        label, pnl, trades = profitability_window or ("-", 0.0, 0)
        status = "profitability_confirmed_but_auto_live_blocked"
        brief = f"{status}:{market}:{'+'.join(blocking_layers)}"
        blocker_detail = (
            f"{market} remote history confirms realized profitability "
            f"({label} pnl={pnl:.8f} across {trades} trades), but automated live remains blocked by "
            f"{', '.join(blocking_layers)}."
        )
        done_when = (
            "clear ops_live_gate and risk_guard blockers while keeping the intended ready-check scope aligned "
            "with the profitable execution account"
        )
    elif profitability_confirmed:
        label, pnl, trades = profitability_window or ("-", 0.0, 0)
        status = "profitability_confirmed_review_live_scope"
        brief = f"{status}:{market}"
        blocker_detail = (
            f"{market} remote history confirms realized profitability "
            f"({label} pnl={pnl:.8f} across {trades} trades); review live scope before enabling automation."
        )
        done_when = "confirm the intended live scope and explicitly re-run live readiness before automation"
    elif blocking_layers:
        status = "auto_live_blocked_without_profitability_confirmation"
        brief = f"{status}:{market}:{'+'.join(blocking_layers)}"
        blocker_detail = (
            "Automated live remains blocked before the system has a positive realized-profit confirmation "
            f"for the active {market} scope."
        )
        done_when = "capture a valid remote history audit and clear the active live blockers"
    else:
        status = "review_required"
        brief = f"review_required:{market}"
        blocker_detail = "Remote live state needs manual review before automation."
        done_when = "refresh remote live history and ready-check artifacts"

    return {
        "status": status,
        "brief": brief,
        "market": market,
        "profitability_confirmed": profitability_confirmed,
        "profitability_window": profitability_window[0] if profitability_window else "",
        "profitability_pnl": profitability_window[1] if profitability_window else None,
        "profitability_trade_count": profitability_window[2] if profitability_window else None,
        "blocking_layers": blocking_layers,
        "blocker_detail": blocker_detail,
        "done_when": done_when,
    }


def derive_remote_live_operator_alignment(
    *,
    cross_market_operator_payload: dict[str, Any],
    remote_live_diagnosis: dict[str, Any],
) -> dict[str, Any]:
    payload = cross_market_operator_payload if isinstance(cross_market_operator_payload, dict) else {}
    head = payload.get("operator_head", {})
    if not isinstance(head, dict) or not head:
        head = {
            "area": str(payload.get("operator_head_area") or "").strip(),
            "symbol": str(payload.get("operator_head_symbol") or "").strip().upper(),
            "action": str(payload.get("operator_head_action") or "").strip(),
            "state": str(payload.get("operator_head_state") or "").strip(),
            "priority_score": payload.get("operator_head_priority_score"),
            "priority_tier": str(payload.get("operator_head_priority_tier") or "").strip(),
            "blocker_detail": str(payload.get("operator_head_blocker_detail") or "").strip(),
            "done_when": str(payload.get("operator_head_done_when") or "").strip(),
        }
    head_area = str(head.get("area") or "").strip()
    head_symbol = str(head.get("symbol") or "").strip().upper()
    head_action = str(head.get("action") or "").strip()
    head_state = str(head.get("state") or "").strip()
    head_priority_score = int(head.get("priority_score") or 0)
    head_priority_tier = str(head.get("priority_tier") or "").strip()
    remote_status = str(remote_live_diagnosis.get("status") or "").strip()
    remote_market = str(remote_live_diagnosis.get("market") or "").strip().lower()

    if not (head_area or head_symbol or head_action):
        return {
            "status": "inactive",
            "brief": "inactive:-",
            "blocking": False,
            "head_area": "",
            "head_symbol": "",
            "head_action": "",
            "head_state": "",
            "head_priority_score": 0,
            "head_priority_tier": "",
            "remote_status": remote_status,
            "remote_market": remote_market,
            "eligible_for_remote_live": False,
            "blocker_detail": "No local cross-market operator head is currently active.",
            "done_when": "cross-market operator state produces an operator head",
        }

    if not _is_remote_live_executable_head(
        area=head_area,
        action=head_action,
        remote_market=remote_market,
    ):
        status = "local_operator_head_outside_remote_live_scope"
        brief = f"{status}:{head_area}:{head_symbol}:{remote_market or '-'}"
        blocker_detail = (
            f"Current local operator head is {head_symbol} "
            f"({head_area}/{head_action}, state={head_state or '-'}, priority={head_priority_score}), "
            f"which is outside remote-live executable scope ({remote_market or '-'}; current takeover only evaluates crypto_route heads)."
        )
        done_when = (
            "cross-market operator head moves into remote-live executable scope "
            "(for example crypto_route) while remote live diagnostics remain fresh"
        )
        return {
            "status": status,
            "brief": brief,
            "blocking": False,
            "head_area": head_area,
            "head_symbol": head_symbol,
            "head_action": head_action,
            "head_state": head_state,
            "head_priority_score": head_priority_score,
            "head_priority_tier": head_priority_tier,
            "remote_status": remote_status,
            "remote_market": remote_market,
            "eligible_for_remote_live": False,
            "blocker_detail": blocker_detail,
            "done_when": done_when,
        }

    remote_blocked = remote_status in {
        "profitability_confirmed_but_auto_live_blocked",
        "auto_live_blocked_without_profitability_confirmation",
    }
    remote_ready = remote_status == "formal_live_possible"
    if remote_blocked:
        status = "local_operator_active_remote_live_blocked"
        brief = f"{status}:{head_area}:{head_symbol}:{remote_status}"
        blocker_detail = (
            f"Current local operator head is {head_symbol} "
            f"({head_area}/{head_action}, state={head_state or '-'}, priority={head_priority_score}), "
            f"while remote automated live remains {remote_status}."
        )
        done_when = (
            f"{head_symbol} local operator head progresses and remote auto-live blockers are cleared"
        )
    elif remote_ready:
        status = "local_operator_active_remote_live_ready_review"
        brief = f"{status}:{head_area}:{head_symbol}"
        blocker_detail = (
            f"Remote auto-live is structurally ready, but the current local operator head remains "
            f"{head_symbol} ({head_area}/{head_action}, state={head_state or '-'}) and still requires operator follow-through."
        )
        done_when = f"{head_symbol} local operator head resolves or is promoted into a live-eligible action"
    else:
        status = "local_operator_active_remote_live_review_required"
        brief = f"{status}:{head_area}:{head_symbol}:{remote_status or 'review_required'}"
        blocker_detail = (
            f"Current local operator head is {head_symbol} "
            f"({head_area}/{head_action}, state={head_state or '-'}), and remote live still needs manual review "
            f"({remote_status or 'review_required'})."
        )
        done_when = f"{head_symbol} local operator head is reviewed alongside refreshed remote live diagnostics"

    return {
        "status": status,
        "brief": brief,
        "blocking": bool(remote_blocked),
        "head_area": head_area,
        "head_symbol": head_symbol,
        "head_action": head_action,
        "head_state": head_state,
        "head_priority_score": head_priority_score,
        "head_priority_tier": head_priority_tier,
        "remote_status": remote_status,
        "remote_market": remote_market,
        "eligible_for_remote_live": True,
        "blocker_detail": blocker_detail,
        "done_when": done_when,
    }


def derive_report(
    handoff_payload: dict[str, Any],
    research_payload: dict[str, Any],
    *,
    handoff_path: Path,
    contract_path: Path | None,
    contract_payload: dict[str, Any] | None,
    research_path: Path,
    cross_market_operator_path: Path | None,
    cross_market_operator_payload: dict[str, Any] | None,
    ops_live_gate_breakdown_path: Path | None,
    ops_live_gate_breakdown_payload: dict[str, Any] | None,
    slot_anomaly_breakdown_path: Path | None,
    slot_anomaly_breakdown_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    operator = _unwrap_operator_handoff(handoff_payload)
    remote_live_history = operator.get("remote_live_history", {})
    if not isinstance(remote_live_history, dict):
        remote_live_history = {}
    account_scope_alignment = operator.get("account_scope_alignment", {})
    if not isinstance(account_scope_alignment, dict):
        account_scope_alignment = {}
    execution_contract = (
        dict(contract_payload)
        if isinstance(contract_payload, dict) and contract_payload
        else operator.get("execution_contract", {})
    )
    if not isinstance(execution_contract, dict):
        execution_contract = {}
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
    risk_guard_reasons = _dedupe_reason_codes(
        [
            str(x)
            for x in (risk_guard.get("reasons", []) if isinstance(risk_guard.get("reasons", []), list) else [])
            if str(x).strip()
        ]
    )
    if not risk_guard_reasons:
        risk_guard_reasons = _dedupe_reason_codes(
            [
                str(x)
                for x in (
                    remote_live_history.get("risk_guard_reasons", [])
                    if isinstance(remote_live_history.get("risk_guard_reasons", []), list)
                    else []
                )
                if str(x).strip()
            ]
        )
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

    remote_live_context = {
        "artifact": str(remote_live_history.get("artifact") or ""),
        "status": str(remote_live_history.get("status") or ""),
        "market": str(remote_live_history.get("market") or ""),
        "ready_check_scope_market": str(operator.get("ready_check_scope_market") or ""),
        "ready_check_scope_brief": str(operator.get("ready_check_scope_brief") or ""),
        "execution_contract_status": str(
            execution_contract.get("contract_status") or execution_contract.get("status") or ""
        ),
        "execution_contract_brief": str(
            execution_contract.get("contract_brief") or execution_contract.get("brief") or ""
        ),
        "execution_contract_executor_mode": str(execution_contract.get("executor_mode") or ""),
        "execution_contract_executor_mode_source": str(
            execution_contract.get("executor_mode_source") or ""
        ),
        "execution_contract_source": str(contract_path) if contract_path else "",
        "generated_at": str(remote_live_history.get("generated_at") or ""),
        "window_brief": str(remote_live_history.get("window_brief") or ""),
        "quote_available": remote_live_history.get("quote_available"),
        "open_positions": remote_live_history.get("open_positions"),
        "risk_guard_status": str(remote_live_history.get("risk_guard_status") or ""),
        "risk_guard_reasons": list(remote_live_history.get("risk_guard_reasons") or []),
        "blocked_candidate_symbol": str(remote_live_history.get("blocked_candidate_symbol") or ""),
        "symbol_pnl_brief": str(remote_live_history.get("symbol_pnl_brief") or ""),
        "day_pnl_brief": str(remote_live_history.get("day_pnl_brief") or ""),
        "account_scope_alignment_status": str(account_scope_alignment.get("status") or ""),
        "account_scope_alignment_brief": str(account_scope_alignment.get("brief") or ""),
        "account_scope_alignment_blocking": bool(account_scope_alignment.get("blocking", False)),
        "account_scope_alignment_blocker_detail": str(
            account_scope_alignment.get("blocker_detail") or ""
        ),
    }
    account_scope_alignment_reason_codes = derive_account_scope_alignment_reason_codes(
        account_scope_alignment=account_scope_alignment,
        remote_live_context=remote_live_context,
    )
    execution_contract_reason_codes = _dedupe_reason_codes(
        [
            str(code).strip()
            for code in (
                execution_contract.get("reason_codes", [])
                if isinstance(execution_contract.get("reason_codes", []), list)
                else []
            )
            if str(code).strip()
        ]
    )
    blockers.append(
        {
            "name": "account_scope_alignment",
            "priority": 4,
            "status": "blocked" if account_scope_alignment_reason_codes else "clear",
            "reason_codes": account_scope_alignment_reason_codes,
            "alignment_status": str(account_scope_alignment.get("status") or ""),
            "alignment_brief": str(account_scope_alignment.get("brief") or ""),
            "blocker_detail": str(account_scope_alignment.get("blocker_detail") or ""),
            "done_when": str(account_scope_alignment.get("done_when") or ""),
        }
    )
    blockers.append(
        {
            "name": "remote_execution_contract",
            "priority": 5,
            "status": "blocked" if execution_contract_reason_codes else "clear",
            "reason_codes": execution_contract_reason_codes,
            "contract_status": str(
                execution_contract.get("contract_status")
                or execution_contract.get("status")
                or ""
            ),
            "contract_brief": str(
                execution_contract.get("contract_brief")
                or execution_contract.get("brief")
                or ""
            ),
            "contract_mode": str(
                execution_contract.get("contract_mode")
                or execution_contract.get("mode")
                or ""
            ),
            "executor_mode": str(execution_contract.get("executor_mode") or ""),
            "executor_mode_source": str(execution_contract.get("executor_mode_source") or ""),
            "guarded_probe_allowed": bool(execution_contract.get("guarded_probe_allowed", False)),
            "live_orders_allowed": bool(execution_contract.get("live_orders_allowed", False)),
            "blocker_detail": str(execution_contract.get("blocker_detail") or ""),
            "done_when": str(execution_contract.get("done_when") or ""),
        }
    )
    ops_live_gate_clearing = derive_clearing_lane(
        area="ops_live_gate",
        reason_codes=_dedupe_reason_codes(gate_blockers + rollback_codes),
        hint_map=OPS_LIVE_GATE_CLEARING_HINTS,
    )
    risk_guard_clearing = derive_clearing_lane(
        area="risk_guard",
        reason_codes=risk_guard_reasons,
        hint_map=RISK_GUARD_CLEARING_HINTS,
    )
    account_scope_alignment_clearing = derive_clearing_lane(
        area="account_scope_alignment",
        reason_codes=account_scope_alignment_reason_codes,
        hint_map=ACCOUNT_SCOPE_ALIGNMENT_CLEARING_HINTS,
    )
    remote_execution_contract_clearing = derive_clearing_lane(
        area="remote_execution_contract",
        reason_codes=execution_contract_reason_codes,
        hint_map=REMOTE_EXECUTION_CONTRACT_CLEARING_HINTS,
    )
    ops_live_gate_breakdown = derive_ops_live_gate_breakdown_summary(
        ops_live_gate_breakdown_payload=ops_live_gate_breakdown_payload,
        ops_live_gate_breakdown_path=ops_live_gate_breakdown_path,
    )
    slot_anomaly_breakdown = derive_slot_anomaly_breakdown_summary(
        slot_anomaly_breakdown_payload=slot_anomaly_breakdown_payload,
        slot_anomaly_breakdown_path=slot_anomaly_breakdown_path,
    )
    remote_live_takeover_clearing = derive_remote_live_takeover_clearing(
        ops_live_gate_clearing=ops_live_gate_clearing,
        risk_guard_clearing=risk_guard_clearing,
    )
    source_freshness = derive_source_freshness(ready_check)
    remote_live_takeover_repair_queue = derive_remote_live_takeover_repair_queue(
        ops_live_gate_clearing=ops_live_gate_clearing,
        risk_guard_clearing=risk_guard_clearing,
        repair_sequence=repair_sequence,
        ops_live_gate_breakdown=ops_live_gate_breakdown_payload,
    )
    remote_live_diagnosis = operator.get("remote_live_diagnosis", {})
    if not isinstance(remote_live_diagnosis, dict) or not remote_live_diagnosis:
        remote_live_diagnosis = derive_remote_live_diagnosis(
            remote_live_history=remote_live_history,
            account_scope_alignment=account_scope_alignment,
            live_ready=live_ready,
            ops_gate_ok=ops_gate_ok,
            risk_guard_reasons=risk_guard_reasons,
        )
    remote_live_operator_alignment = derive_remote_live_operator_alignment(
        cross_market_operator_payload=cross_market_operator_payload or {},
        remote_live_diagnosis=remote_live_diagnosis,
    )

    live_decision = {
        "formal_live_ready": live_ready
        and ops_gate_ok
        and not risk_guard_reasons
        and not execution_contract_reason_codes,
        "micro_canary_only": bool(
            live_ready and ops_gate_ok and not risk_guard_reasons and bool(execution_contract_reason_codes)
        ),
        "current_decision": "do_not_start_formal_live"
        if (not live_ready or not ops_gate_ok or risk_guard_reasons or execution_contract_reason_codes)
        else "formal_live_possible",
        "summary": (
            "Do not start formal live trading yet. Clear ops_live_gate, recover actionable tickets, and promote an explicit non-shadow execution contract before routing capital."
            if (not live_ready or not ops_gate_ok or risk_guard_reasons or execution_contract_reason_codes)
            else "Formal live is structurally possible."
        ),
    }

    return {
        "generated_at": fmt_utc(now_utc()),
        "handoff_source": str(handoff_path),
        "remote_execution_contract_source": str(contract_path) if contract_path else "",
        "research_source": str(research_path),
        "cross_market_operator_source": str(cross_market_operator_path) if cross_market_operator_path else "",
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
        "remote_live_context": remote_live_context,
        "source_freshness": source_freshness,
        "ops_live_gate_clearing": ops_live_gate_clearing,
        "ops_live_gate_breakdown": ops_live_gate_breakdown,
        "slot_anomaly_breakdown": slot_anomaly_breakdown,
        "risk_guard_clearing": risk_guard_clearing,
        "account_scope_alignment_clearing": account_scope_alignment_clearing,
        "remote_execution_contract": execution_contract,
        "remote_execution_contract_clearing": remote_execution_contract_clearing,
        "remote_live_takeover_clearing": remote_live_takeover_clearing,
        "remote_live_takeover_repair_queue": remote_live_takeover_repair_queue,
        "remote_live_diagnosis": remote_live_diagnosis,
        "remote_live_operator_alignment": remote_live_operator_alignment,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    live_decision = payload.get("live_decision", {})
    blockers = payload.get("blockers", [])
    repair_sequence = payload.get("repair_sequence", [])
    path = payload.get("commodity_execution_path", {})
    remote_live = payload.get("remote_live_context", {})
    source_freshness = payload.get("source_freshness", {})
    ops_clearing = payload.get("ops_live_gate_clearing", {})
    ops_breakdown = payload.get("ops_live_gate_breakdown", {})
    slot_breakdown = payload.get("slot_anomaly_breakdown", {})
    risk_clearing = payload.get("risk_guard_clearing", {})
    account_scope_clearing = payload.get("account_scope_alignment_clearing", {})
    execution_contract = payload.get("remote_execution_contract", {})
    execution_contract_clearing = payload.get("remote_execution_contract_clearing", {})
    takeover_clearing = payload.get("remote_live_takeover_clearing", {})
    takeover_repair_queue = payload.get("remote_live_takeover_repair_queue", {})
    remote_diagnosis = payload.get("remote_live_diagnosis", {})
    lines = [
        "# Live Gate Blocker Report",
        "",
        f"- handoff source: `{payload.get('handoff_source', '')}`",
        f"- research source: `{payload.get('research_source', '')}`",
        f"- execution contract source: `{payload.get('remote_execution_contract_source', '') or '-'}`",
        f"- decision: `{live_decision.get('current_decision', '')}`",
        f"- summary: {live_decision.get('summary', '')}",
        f"- status: `{payload.get('operator_status_quad') or payload.get('operator_status_triplet') or ''}`",
        f"- focus stack: `{payload.get('next_focus_area', '')} -> {payload.get('secondary_focus_area', '')}`",
        f"- remote live context: `{remote_live.get('window_brief', '') or '-'}` / risk_guard=`{remote_live.get('risk_guard_status', '') or '-'}`",
        f"- source freshness: `{source_freshness.get('brief', '') or '-'}`",
        f"- remote live diagnosis: `{remote_diagnosis.get('brief', '') or '-'}`",
        f"- local/remote operator alignment: `{((payload.get('remote_live_operator_alignment') or {}) if isinstance(payload.get('remote_live_operator_alignment'), dict) else {}).get('brief', '') or '-'}`",
        "",
        "## Remote Live Diagnosis",
        f"- status: `{remote_diagnosis.get('status', '') or '-'}` market=`{remote_diagnosis.get('market', '') or '-'}`",
        f"- profitability_confirmed: `{remote_diagnosis.get('profitability_confirmed')}` window=`{remote_diagnosis.get('profitability_window', '') or '-'}` pnl=`{remote_diagnosis.get('profitability_pnl')}` trades=`{remote_diagnosis.get('profitability_trade_count')}`",
        f"- blocking_layers: `{', '.join(remote_diagnosis.get('blocking_layers', [])) if isinstance(remote_diagnosis.get('blocking_layers', []), list) and remote_diagnosis.get('blocking_layers', []) else '-'}`",
        f"- blocker: `{remote_diagnosis.get('blocker_detail', '') or '-'}`",
        f"- done_when: `{remote_diagnosis.get('done_when', '') or '-'}`",
        "",
        "## Local / Remote Operator Alignment",
        f"- status: `{((payload.get('remote_live_operator_alignment') or {}) if isinstance(payload.get('remote_live_operator_alignment'), dict) else {}).get('status', '') or '-'}`",
        f"- brief: `{((payload.get('remote_live_operator_alignment') or {}) if isinstance(payload.get('remote_live_operator_alignment'), dict) else {}).get('brief', '') or '-'}`",
        f"- head: `{((payload.get('remote_live_operator_alignment') or {}) if isinstance(payload.get('remote_live_operator_alignment'), dict) else {}).get('head_area', '') or '-'} | {((payload.get('remote_live_operator_alignment') or {}) if isinstance(payload.get('remote_live_operator_alignment'), dict) else {}).get('head_symbol', '') or '-'} | {((payload.get('remote_live_operator_alignment') or {}) if isinstance(payload.get('remote_live_operator_alignment'), dict) else {}).get('head_action', '') or '-'} | state={((payload.get('remote_live_operator_alignment') or {}) if isinstance(payload.get('remote_live_operator_alignment'), dict) else {}).get('head_state', '') or '-'} | priority={((payload.get('remote_live_operator_alignment') or {}) if isinstance(payload.get('remote_live_operator_alignment'), dict) else {}).get('head_priority_score', 0)}`",
        f"- blocker: `{((payload.get('remote_live_operator_alignment') or {}) if isinstance(payload.get('remote_live_operator_alignment'), dict) else {}).get('blocker_detail', '') or '-'}`",
        f"- done_when: `{((payload.get('remote_live_operator_alignment') or {}) if isinstance(payload.get('remote_live_operator_alignment'), dict) else {}).get('done_when', '') or '-'}`",
        "",
        "## Remote Live Takeover Clearing",
        f"- status: `{takeover_clearing.get('status', '') or '-'}` brief=`{takeover_clearing.get('brief', '') or '-'}`",
        f"- blocking_layers: `{', '.join(takeover_clearing.get('blocking_layers', [])) if isinstance(takeover_clearing.get('blocking_layers', []), list) and takeover_clearing.get('blocking_layers', []) else '-'}`",
        f"- blocker: `{takeover_clearing.get('blocker_detail', '') or '-'}`",
        f"- done_when: `{takeover_clearing.get('done_when', '') or '-'}`",
        "",
        "## Ops Live Gate Clearing",
        f"- status: `{ops_clearing.get('status', '') or '-'}` brief=`{ops_clearing.get('brief', '') or '-'}` count=`{ops_clearing.get('count', 0)}`",
        f"- reasons: `{ops_clearing.get('conditions_brief', '') or '-'}`",
        f"- blocker: `{ops_clearing.get('blocker_detail', '') or '-'}`",
        f"- done_when: `{ops_clearing.get('done_when', '') or '-'}`",
        "",
        "## Ops Live Gate Breakdown",
        f"- status: `{ops_breakdown.get('status', '') or '-'}`",
        f"- artifact: `{ops_breakdown.get('artifact', '') or '-'}`",
        f"- brief: `{ops_breakdown.get('brief', '') or '-'}`",
        f"- primary_root_cause: `{ops_breakdown.get('primary_root_cause_code', '') or '-'}`",
        f"- primary_fix: `{ops_breakdown.get('primary_root_cause_fix_action', '') or '-'}`",
        "",
        "## Slot Anomaly Breakdown",
        f"- status: `{slot_breakdown.get('status', '') or '-'}`",
        f"- artifact: `{slot_breakdown.get('artifact', '') or '-'}`",
        f"- brief: `{slot_breakdown.get('brief', '') or '-'}`",
        f"- repair_focus: `{slot_breakdown.get('repair_focus', '') or '-'}`",
        f"- payload_gap: `{slot_breakdown.get('payload_gap', '') or '-'}`",
        "",
        "## Risk Guard Clearing",
        f"- status: `{risk_clearing.get('status', '') or '-'}` brief=`{risk_clearing.get('brief', '') or '-'}` count=`{risk_clearing.get('count', 0)}`",
        f"- reasons: `{risk_clearing.get('conditions_brief', '') or '-'}`",
        f"- blocker: `{risk_clearing.get('blocker_detail', '') or '-'}`",
        f"- done_when: `{risk_clearing.get('done_when', '') or '-'}`",
        "",
        "## Account Scope Alignment Clearing",
        f"- status: `{account_scope_clearing.get('status', '') or '-'}` brief=`{account_scope_clearing.get('brief', '') or '-'}` count=`{account_scope_clearing.get('count', 0)}`",
        f"- reasons: `{account_scope_clearing.get('conditions_brief', '') or '-'}`",
        f"- blocker: `{account_scope_clearing.get('blocker_detail', '') or '-'}`",
        f"- done_when: `{account_scope_clearing.get('done_when', '') or '-'}`",
        "",
        "## Remote Execution Contract",
        f"- brief: `{execution_contract.get('brief', '') or '-'}`",
        f"- status: `{execution_contract.get('status', '') or '-'}` mode=`{execution_contract.get('mode', '') or '-'}` live_orders_allowed=`{execution_contract.get('live_orders_allowed', False)}`",
        f"- blocker: `{execution_contract.get('blocker_detail', '') or '-'}`",
        f"- done_when: `{execution_contract.get('done_when', '') or '-'}`",
        "",
        "## Remote Execution Contract Clearing",
        f"- status: `{execution_contract_clearing.get('status', '') or '-'}` brief=`{execution_contract_clearing.get('brief', '') or '-'}` count=`{execution_contract_clearing.get('count', 0)}`",
        f"- reasons: `{execution_contract_clearing.get('conditions_brief', '') or '-'}`",
        f"- blocker: `{execution_contract_clearing.get('blocker_detail', '') or '-'}`",
        f"- done_when: `{execution_contract_clearing.get('done_when', '') or '-'}`",
        "",
        "## Remote Live Takeover Repair Queue",
        f"- status: `{takeover_repair_queue.get('status', '') or '-'}` brief=`{takeover_repair_queue.get('brief', '') or '-'}` count=`{takeover_repair_queue.get('count', 0)}`",
        f"- head: `{takeover_repair_queue.get('head_area', '') or '-'} | {takeover_repair_queue.get('head_code', '') or '-'} | {takeover_repair_queue.get('head_action', '') or '-'} | priority={takeover_repair_queue.get('head_priority_score', 0)}/{takeover_repair_queue.get('head_priority_tier', '') or '-'}`",
        f"- head_command: `{takeover_repair_queue.get('head_command', '') or '-'}`",
        f"- head_clear_when: `{takeover_repair_queue.get('head_clear_when', '') or '-'}`",
        f"- queue: `{takeover_repair_queue.get('queue_brief', '') or '-'}`",
        f"- done_when: `{takeover_repair_queue.get('done_when', '') or '-'}`",
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
    lines.extend(["", "## Remote Live Context"])
    lines.append(f"- artifact: `{remote_live.get('artifact', '') or '-'}`")
    lines.append(f"- status: `{remote_live.get('status', '') or '-'}` market=`{remote_live.get('market', '') or '-'}`")
    lines.append(
        f"- scope: `ready_check={remote_live.get('ready_check_scope_brief', '') or '-'} | alignment={remote_live.get('account_scope_alignment_brief', '') or '-'}`"
    )
    lines.append(f"- windows: `{remote_live.get('window_brief', '') or '-'}`")
    lines.append(
        f"- snapshot: `quote_available={remote_live.get('quote_available')} | open_positions={remote_live.get('open_positions')} | blocked_candidate={remote_live.get('blocked_candidate_symbol', '') or '-'}`"
    )
    lines.append(
        f"- risk_guard: `{remote_live.get('risk_guard_status', '') or '-'} | {', '.join(remote_live.get('risk_guard_reasons', [])) if isinstance(remote_live.get('risk_guard_reasons', []), list) and remote_live.get('risk_guard_reasons', []) else '-'}`"
    )
    lines.append(f"- pnl_by_symbol: `{remote_live.get('symbol_pnl_brief', '') or '-'}`")
    lines.append(f"- pnl_by_day: `{remote_live.get('day_pnl_brief', '') or '-'}`")
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
    contract_path = find_latest(review_dir, "*_remote_execution_contract_state.json")
    research_path = args.research_json.expanduser().resolve() if args.research_json else select_research_artifact(review_dir)
    cross_market_operator_path = find_latest(review_dir, "*_cross_market_operator_state.json")
    if handoff_path is None:
        raise SystemExit("remote live handoff artifact not found")
    if research_path is None:
        raise SystemExit("hot universe research artifact not found")

    handoff_payload = load_json_mapping(handoff_path)
    research_payload = load_json_mapping(research_path)
    cross_market_operator_payload = (
        load_json_mapping(cross_market_operator_path)
        if cross_market_operator_path and cross_market_operator_path.exists()
        else {}
    )
    ops_live_gate_breakdown_path, ops_live_gate_breakdown_payload = select_aligned_supporting_artifact(
        review_dir,
        pattern="*_ops_live_gate_breakdown.json",
        expected_source_field="handoff_source",
        expected_source_path=handoff_path,
    )
    slot_anomaly_breakdown_path, slot_anomaly_breakdown_payload = select_aligned_supporting_artifact(
        review_dir,
        pattern="*_slot_anomaly_breakdown.json",
        expected_source_field="handoff_source",
        expected_source_path=handoff_path,
    )
    payload = derive_report(
        handoff_payload,
        research_payload,
        handoff_path=handoff_path,
        contract_path=contract_path,
        contract_payload=load_json_mapping(contract_path) if contract_path and contract_path.exists() else {},
        research_path=research_path,
        cross_market_operator_path=cross_market_operator_path,
        cross_market_operator_payload=cross_market_operator_payload,
        ops_live_gate_breakdown_path=ops_live_gate_breakdown_path,
        ops_live_gate_breakdown_payload=ops_live_gate_breakdown_payload,
        slot_anomaly_breakdown_path=slot_anomaly_breakdown_path,
        slot_anomaly_breakdown_payload=slot_anomaly_breakdown_payload,
    )

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
