#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any


TIMESTAMP_RE = re.compile(r"^(?P<stamp>\d{8}T\d{6}Z)_")
FUTURE_STAMP_GRACE_MINUTES = 5


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse_now(raw: str | None) -> dt.datetime:
    text = str(raw or "").strip()
    if not text:
        return now_utc()
    parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def fmt_utc(value: dt.datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(dt.timezone.utc).isoformat()


def artifact_stamp(path: Path) -> str:
    match = TIMESTAMP_RE.match(path.name)
    return match.group("stamp") if match else ""


def parsed_artifact_stamp(path: Path) -> dt.datetime | None:
    stamp = artifact_stamp(path)
    if not stamp:
        return None
    return dt.datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)


def artifact_sort_key(path: Path, reference_now: dt.datetime | None = None) -> tuple[int, str, float, str]:
    stamp_dt = parsed_artifact_stamp(path)
    effective_now = reference_now or now_utc()
    future_cutoff = effective_now + dt.timedelta(minutes=FUTURE_STAMP_GRACE_MINUTES)
    is_future = bool(stamp_dt and stamp_dt > future_cutoff)
    return (0 if is_future else 1, artifact_stamp(path), path.stat().st_mtime, path.name)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latest_crypto_route_brief(review_dir: Path, reference_now: dt.datetime | None = None) -> Path:
    candidates = list(review_dir.glob("*_crypto_route_brief.json"))
    if not candidates:
        raise FileNotFoundError("no_crypto_route_brief_artifact")
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def latest_hot_universe_research(review_dir: Path, reference_now: dt.datetime | None = None) -> Path | None:
    candidates = list(review_dir.glob("*_hot_universe_research.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def latest_crypto_route_refresh(review_dir: Path, reference_now: dt.datetime | None = None) -> Path | None:
    candidates = list(review_dir.glob("*_crypto_route_refresh.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def prune_review_artifacts(
    review_dir: Path,
    *,
    stem: str,
    current_paths: list[Path],
    keep: int,
    ttl_hours: float,
    now_dt: dt.datetime | None = None,
) -> tuple[list[str], list[str]]:
    effective_now = now_dt or now_utc()
    cutoff = effective_now - dt.timedelta(hours=max(1.0, ttl_hours))
    protected = {path.name for path in current_paths}
    candidates = sorted(review_dir.glob(f"*_{stem}*"), key=lambda item: item.stat().st_mtime, reverse=True)

    pruned_age: list[str] = []
    survivors: list[Path] = []
    for path in candidates:
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


def _head_source_refresh_lane(source_payload: dict[str, Any], *, current_artifact: str = "") -> dict[str, str]:
    queue = [dict(row) for row in source_payload.get("review_priority_queue", []) if isinstance(row, dict)]
    head_symbol = str(source_payload.get("review_priority_head_symbol") or "").strip().upper()
    head_tier = str(source_payload.get("review_priority_head_tier") or "").strip() or "-"
    head_item: dict[str, Any] = {}
    for row in queue:
        row_symbol = str(row.get("symbol") or "").strip().upper()
        if head_symbol and row_symbol == head_symbol:
            head_item = row
            break
        if not head_item and int(row.get("rank") or 0) == 1:
            head_item = row
    if not head_symbol and head_item:
        head_symbol = str(head_item.get("symbol") or "").strip().upper()

    route_action = (
        str(head_item.get("route_action") or "").strip()
        or str(source_payload.get("next_focus_action") or "").strip()
        or "-"
    )
    source_artifact = str(current_artifact or "").strip() or str(source_payload.get("source_artifact") or "").strip()
    source_health = "ready" if source_artifact else "missing"
    action = "read_current_artifact" if head_symbol and source_artifact else "-"
    status = "ready" if action == "read_current_artifact" else "not_active"
    blocker_detail = "-"
    done_when = "-"
    if head_symbol and action == "read_current_artifact":
        blocker_detail = (
            f"{head_symbol} currently uses the readable crypto_route operator artifact "
            f"while priority_tier={head_tier} and route_action={route_action}."
        )
        done_when = (
            f"keep {head_symbol} on the current crypto_route operator artifact until the queue head changes "
            "or a newer route refresh is required"
        )
    return {
        "status": status,
        "brief": f"{status}:{head_symbol or '-'}:{action}",
        "symbol": head_symbol,
        "action": action,
        "source_kind": "crypto_route",
        "source_health": source_health,
        "source_artifact": source_artifact,
        "priority_tier": head_tier,
        "route_action": route_action,
        "blocker_detail": blocker_detail,
        "done_when": done_when,
    }


def _downstream_embedding_lane(
    source_payload: dict[str, Any],
    *,
    current_artifact: str = "",
    review_dir: Path | None = None,
    reference_now: dt.datetime | None = None,
) -> dict[str, str]:
    head_source_refresh = _head_source_refresh_lane(source_payload, current_artifact=current_artifact)
    symbol = str(head_source_refresh.get("symbol") or "").strip().upper()
    head_status = str(head_source_refresh.get("status") or "").strip()
    head_action = str(head_source_refresh.get("action") or "").strip()
    if not symbol:
        return {
            "status": "not_applicable",
            "brief": "not_applicable:-",
            "artifact": "",
            "as_of": "",
            "blocker_detail": "crypto route downstream embedding lane is only active when a route review head is present.",
            "done_when": "crypto route head returns before reassessing downstream embedding freshness",
        }

    if review_dir is None:
        return {
            "status": "not_assessed",
            "brief": f"not_assessed:{symbol}",
            "artifact": "",
            "as_of": "",
            "blocker_detail": "hot_universe_research artifact was not checked while building the crypto route operator brief.",
            "done_when": "rebuild the crypto route operator brief with review_dir context to assess downstream embedding freshness",
        }

    hot_research_path = latest_hot_universe_research(review_dir, reference_now)
    hot_research_as_of = parsed_artifact_stamp(hot_research_path) if hot_research_path else None
    route_as_of = parsed_artifact_stamp(Path(current_artifact)) if current_artifact else None

    if head_status in {"ready", "deferred_until_next_eligible_end_date"} and head_action in {
        "read_current_artifact",
        "wait_for_next_eligible_end_date",
    }:
        if hot_research_path and route_as_of and hot_research_as_of and hot_research_as_of < route_as_of:
            return {
                "status": "carry_over_non_blocking",
                "brief": f"carry_over_non_blocking:{symbol}",
                "artifact": str(hot_research_path),
                "as_of": fmt_utc(hot_research_as_of) or "",
                "blocker_detail": (
                    f"{symbol} current crypto route head is already {head_status} via {head_action}, "
                    f"but latest downstream hot_universe_research artifact ({hot_research_path.name}) is older "
                    "and remains broader carry-over outside route-operator scope."
                ),
                "done_when": "rerun hot_universe_research only when broader embedding freshness is required",
            }
        return {
            "status": "current_non_blocking",
            "brief": f"current_non_blocking:{symbol}",
            "artifact": str(hot_research_path or ""),
            "as_of": fmt_utc(hot_research_as_of) or "",
            "blocker_detail": (
                f"{symbol} current crypto route head is already {head_status} via {head_action}, "
                "and downstream hot_universe_research is current enough for route-operator scope."
            ),
            "done_when": "keep downstream embedding current enough for the broader hot-universe handoff",
        }

    return {
        "status": "blocked_by_head_source_refresh",
        "brief": f"blocked_by_head_source_refresh:{symbol}",
        "artifact": str(hot_research_path or ""),
        "as_of": fmt_utc(hot_research_as_of) or "",
        "blocker_detail": (
            f"{symbol} route head is not yet readable via current source refresh lane ({head_status or '-'}:{head_action or '-'})"
        ),
        "done_when": "stabilize the current crypto route head source refresh lane before reassessing downstream embedding freshness",
    }


def _latest_crypto_route_refresh_audit_lane(
    *,
    review_dir: Path | None,
    reference_now: dt.datetime | None,
) -> dict[str, Any]:
    if review_dir is None:
        return {
            "status": "not_assessed",
            "brief": "not_assessed",
            "artifact": "",
            "as_of": "",
            "native_mode": "",
            "native_step_count": 0,
            "reused_native_count": 0,
            "missing_reused_count": 0,
            "note": "latest crypto_route_refresh was not checked while building the crypto route operator brief.",
            "done_when": "rebuild the crypto route operator brief with review_dir context to assess the latest refresh audit",
        }
    refresh_path = latest_crypto_route_refresh(review_dir, reference_now)
    if refresh_path is None:
        return {
            "status": "not_available",
            "brief": "not_available",
            "artifact": "",
            "as_of": "",
            "native_mode": "",
            "native_step_count": 0,
            "reused_native_count": 0,
            "missing_reused_count": 0,
            "note": "no crypto_route_refresh artifact is available yet.",
            "done_when": "run refresh_crypto_route_state.py before expecting route refresh audit coverage",
        }
    payload = json.loads(refresh_path.read_text(encoding="utf-8"))
    steps = [dict(row) for row in payload.get("steps", []) if isinstance(row, dict)]
    native_steps = [row for row in steps if str(row.get("name") or "").startswith("native_")]
    native_step_count = len(native_steps)
    reused_native_count = sum(
        1 for row in native_steps if str(row.get("status") or "").strip() == "reused_previous_artifact"
    )
    missing_reused_count = sum(
        1 for row in native_steps if str(row.get("status") or "").strip() == "missing_reused_source"
    )
    native_mode = str(payload.get("native_refresh_mode") or "").strip()
    if native_step_count <= 0:
        status = "native_audit_unavailable"
    elif reused_native_count == native_step_count:
        status = "reused_native_inputs"
    elif reused_native_count > 0:
        status = "mixed_native_inputs"
    elif missing_reused_count > 0:
        status = "native_reuse_incomplete"
    else:
        status = "fresh_native_inputs"
    brief = (
        f"{status}:{native_mode or '-'}:{reused_native_count}/{native_step_count}"
        if native_step_count > 0
        else f"{status}:{native_mode or '-'}"
    )
    if status == "reused_native_inputs":
        note = (
            f"latest crypto_route_refresh reuses {reused_native_count}/{native_step_count} native inputs "
            f"via {native_mode or 'unknown_mode'}."
        )
        done_when = "run full native refresh only when fresh native recomputation is required"
    elif status == "fresh_native_inputs":
        note = f"latest crypto_route_refresh refreshed native inputs directly ({native_step_count}/{native_step_count})."
        done_when = "keep using the latest route refresh while it stays fresh enough"
    elif status == "mixed_native_inputs":
        note = (
            f"latest crypto_route_refresh mixes reused and refreshed native inputs "
            f"({reused_native_count}/{native_step_count} reused)."
        )
        done_when = "stabilize native refresh mode before treating route refresh inputs as uniform"
    elif status == "native_reuse_incomplete":
        note = (
            f"latest crypto_route_refresh could not reuse all expected native inputs "
            f"({missing_reused_count} missing of {native_step_count})."
        )
        done_when = "fill missing native sources or rerun guarded native refresh"
    else:
        note = "latest crypto_route_refresh did not expose any native_* steps to audit."
        done_when = "record native refresh steps before relying on route refresh audit"
    return {
        "status": status,
        "brief": brief,
        "artifact": str(refresh_path),
        "as_of": fmt_utc(parsed_artifact_stamp(refresh_path)) or str(payload.get("as_of") or ""),
        "native_mode": native_mode,
        "native_step_count": native_step_count,
        "reused_native_count": reused_native_count,
        "missing_reused_count": missing_reused_count,
        "note": note,
        "done_when": done_when,
    }


def _latest_crypto_route_refresh_reuse_gate(audit: dict[str, Any]) -> dict[str, Any]:
    status = str(audit.get("status") or "").strip()
    brief = str(audit.get("brief") or "").strip()
    native_mode = str(audit.get("native_mode") or "").strip() or "unknown"
    reused_native_count = int(audit.get("reused_native_count") or 0)
    native_step_count = int(audit.get("native_step_count") or 0)
    if status == "reused_native_inputs":
        return {
            "level": "informational",
            "status": "reuse_non_blocking",
            "brief": f"reuse_non_blocking:{native_mode}:{reused_native_count}/{native_step_count}",
            "blocking": False,
            "blocker_detail": "latest crypto_route_refresh reused all tracked native steps; current route operator brief may safely read the reused native path.",
            "done_when": "run full native refresh only when fresh native recomputation is explicitly required",
        }
    if status == "fresh_native_inputs":
        return {
            "level": "informational",
            "status": "fresh_non_blocking",
            "brief": f"fresh_non_blocking:{native_mode}:{reused_native_count}/{native_step_count}",
            "blocking": False,
            "blocker_detail": "latest crypto_route_refresh used fresh native inputs across tracked native steps.",
            "done_when": "keep using current fresh native inputs until the next required recomputation window",
        }
    if status in {"mixed_native_inputs", "native_reuse_incomplete", "native_audit_unavailable", "not_available"}:
        gate_status = (
            "mixed_requires_full_native_refresh"
            if status in {"mixed_native_inputs", "native_reuse_incomplete"}
            else "audit_missing_requires_full_native_refresh"
        )
        return {
            "level": "blocking",
            "status": gate_status,
            "brief": f"{gate_status}:{native_mode}:{reused_native_count}/{native_step_count}",
            "blocking": True,
            "blocker_detail": f"latest crypto_route_refresh audit is not fully reusable-safe ({brief or status}); force a full native refresh before trusting route-operator reuse.",
            "done_when": "rerun refresh_crypto_route_state without skip_native_refresh and confirm all native steps are either fresh or intentionally reused end-to-end",
        }
    return {
        "level": "blocking",
        "status": "unknown_requires_review",
        "brief": f"unknown_requires_review:{brief or status or 'unknown'}",
        "blocking": True,
        "blocker_detail": "latest crypto_route_refresh audit returned an unknown reuse status; manual review or a clean full native refresh is required.",
        "done_when": "rerun refresh_crypto_route_state with a known-good native refresh outcome",
    }


def build_operator_brief(
    source_payload: dict[str, Any],
    *,
    current_artifact: str = "",
    review_dir: Path | None = None,
    reference_now: dt.datetime | None = None,
) -> dict[str, Any]:
    operator_status = str(source_payload.get("operator_status") or "")
    route_stack_brief = str(source_payload.get("route_stack_brief") or "")
    next_focus_symbol = str(source_payload.get("next_focus_symbol") or "")
    next_focus_action = str(source_payload.get("next_focus_action") or "")
    next_focus_reason = str(source_payload.get("next_focus_reason") or "")
    focus_window_gate = str(source_payload.get("focus_window_gate") or "")
    focus_short_flow_combo_canonical = str(source_payload.get("focus_short_flow_combo_canonical") or "")
    focus_long_flow_combo_canonical = str(source_payload.get("focus_long_flow_combo_canonical") or "")
    focus_long_top_combo_canonical = str(source_payload.get("focus_long_top_combo_canonical") or "")
    focus_window_verdict = str(source_payload.get("focus_window_verdict") or "")
    focus_window_floor = str(source_payload.get("focus_window_floor") or "")
    price_state_window_floor = str(source_payload.get("price_state_window_floor") or "")
    comparative_window_takeaway = str(source_payload.get("comparative_window_takeaway") or "")
    xlong_flow_window_floor = str(source_payload.get("xlong_flow_window_floor") or "")
    xlong_comparative_window_takeaway = str(source_payload.get("xlong_comparative_window_takeaway") or "")
    focus_brief = str(source_payload.get("focus_brief") or "")
    next_retest_action = str(source_payload.get("next_retest_action") or "")
    next_retest_reason = str(source_payload.get("next_retest_reason") or "")
    shortline_market_state_brief = str(source_payload.get("shortline_market_state_brief") or "")
    shortline_execution_gate_brief = str(source_payload.get("shortline_execution_gate_brief") or "")
    shortline_session_map_brief = str(source_payload.get("shortline_session_map_brief") or "")
    shortline_no_trade_rule = str(source_payload.get("shortline_no_trade_rule") or "")
    shortline_cvd_semantic_status = str(source_payload.get("shortline_cvd_semantic_status") or "")
    shortline_cvd_semantic_takeaway = str(source_payload.get("shortline_cvd_semantic_takeaway") or "")
    shortline_cvd_queue_handoff_status = str(source_payload.get("shortline_cvd_queue_handoff_status") or "")
    shortline_cvd_queue_handoff_takeaway = str(source_payload.get("shortline_cvd_queue_handoff_takeaway") or "")
    shortline_cvd_queue_focus_batch = str(source_payload.get("shortline_cvd_queue_focus_batch") or "")
    shortline_cvd_queue_focus_action = str(source_payload.get("shortline_cvd_queue_focus_action") or "")
    shortline_cvd_queue_stack_brief = str(source_payload.get("shortline_cvd_queue_stack_brief") or "")
    focus_execution_state = str(source_payload.get("focus_execution_state") or "")
    focus_execution_blocker_detail = str(source_payload.get("focus_execution_blocker_detail") or "")
    focus_execution_done_when = str(source_payload.get("focus_execution_done_when") or "")
    focus_execution_micro_classification = str(source_payload.get("focus_execution_micro_classification") or "")
    focus_execution_micro_context = str(source_payload.get("focus_execution_micro_context") or "")
    focus_execution_micro_trust_tier = str(source_payload.get("focus_execution_micro_trust_tier") or "")
    focus_execution_micro_veto = str(source_payload.get("focus_execution_micro_veto") or "")
    focus_execution_micro_locality_status = str(
        source_payload.get("focus_execution_micro_locality_status") or ""
    )
    focus_execution_micro_drift_risk = str(source_payload.get("focus_execution_micro_drift_risk") or "")
    focus_execution_micro_attack_side = str(source_payload.get("focus_execution_micro_attack_side") or "")
    focus_execution_micro_attack_presence = str(
        source_payload.get("focus_execution_micro_attack_presence") or ""
    )
    focus_execution_micro_reasons = list(source_payload.get("focus_execution_micro_reasons") or [])
    focus_review_status = str(source_payload.get("focus_review_status") or "")
    focus_review_brief = str(source_payload.get("focus_review_brief") or "")
    focus_review_primary_blocker = str(source_payload.get("focus_review_primary_blocker") or "")
    focus_review_micro_blocker = str(source_payload.get("focus_review_micro_blocker") or "")
    focus_review_blocker_detail = str(source_payload.get("focus_review_blocker_detail") or "")
    focus_review_done_when = str(source_payload.get("focus_review_done_when") or "")
    focus_review_score_status = str(source_payload.get("focus_review_score_status") or "")
    focus_review_edge_score = int(source_payload.get("focus_review_edge_score") or 0)
    focus_review_structure_score = int(source_payload.get("focus_review_structure_score") or 0)
    focus_review_micro_score = int(source_payload.get("focus_review_micro_score") or 0)
    focus_review_composite_score = int(source_payload.get("focus_review_composite_score") or 0)
    focus_review_score_brief = str(source_payload.get("focus_review_score_brief") or "")
    focus_review_priority_status = str(source_payload.get("focus_review_priority_status") or "")
    focus_review_priority_score = int(source_payload.get("focus_review_priority_score") or 0)
    focus_review_priority_tier = str(source_payload.get("focus_review_priority_tier") or "")
    focus_review_priority_brief = str(source_payload.get("focus_review_priority_brief") or "")
    review_priority_queue_status = str(source_payload.get("review_priority_queue_status") or "")
    review_priority_queue_count = int(source_payload.get("review_priority_queue_count") or 0)
    review_priority_queue_brief = str(source_payload.get("review_priority_queue_brief") or "")
    review_priority_head_symbol = str(source_payload.get("review_priority_head_symbol") or "")
    review_priority_head_tier = str(source_payload.get("review_priority_head_tier") or "")
    review_priority_head_score = int(source_payload.get("review_priority_head_score") or 0)
    review_priority_queue = [dict(row) for row in source_payload.get("review_priority_queue", []) if isinstance(row, dict)]
    head_source_refresh = _head_source_refresh_lane(source_payload, current_artifact=current_artifact)
    latest_refresh_audit = _latest_crypto_route_refresh_audit_lane(
        review_dir=review_dir,
        reference_now=reference_now,
    )
    latest_refresh_reuse_gate = _latest_crypto_route_refresh_reuse_gate(latest_refresh_audit)
    downstream_embedding = _downstream_embedding_lane(
        source_payload,
        current_artifact=current_artifact,
        review_dir=review_dir,
        reference_now=reference_now,
    )
    deploy_now_symbols = list(source_payload.get("deploy_now_symbols") or [])
    watch_priority_symbols = list(source_payload.get("watch_priority_symbols") or [])
    watch_only_symbols = list(source_payload.get("watch_only_symbols") or [])

    operator_lines = [
        f"status: {operator_status or '-'}",
        f"routes: {route_stack_brief or '-'}",
        f"focus: {next_focus_symbol or '-'}",
        f"action: {next_focus_action or '-'}",
        f"focus-gate: {focus_window_gate or '-'}",
        f"focus-short-flow: {focus_short_flow_combo_canonical or '-'}",
        f"focus-long-flow: {focus_long_flow_combo_canonical or '-'}",
        f"focus-long-top: {focus_long_top_combo_canonical or '-'}",
        f"focus-window: {focus_window_verdict or '-'}",
        f"focus-window-floor: {focus_window_floor or '-'}",
        f"price-state-window-floor: {price_state_window_floor or '-'}",
        f"shortline-market-state: {shortline_market_state_brief or '-'}",
        f"shortline-trigger-stack: {shortline_execution_gate_brief or '-'}",
        f"shortline-sessions: {shortline_session_map_brief or '-'}",
        f"focus-execution-state: {focus_execution_state or '-'}",
        f"focus-execution-blocker: {focus_execution_blocker_detail or '-'}",
        f"focus-execution-done-when: {focus_execution_done_when or '-'}",
        f"micro-class: {focus_execution_micro_classification or '-'}",
        f"micro-context: {focus_execution_micro_context or '-'}",
        f"micro-trust: {focus_execution_micro_trust_tier or '-'}",
        f"micro-veto: {focus_execution_micro_veto or '-'}",
        f"micro-locality: {focus_execution_micro_locality_status or '-'}",
        f"micro-drift: {focus_execution_micro_drift_risk or '-'}",
        f"micro-attack: {':'.join(part for part in [focus_execution_micro_attack_side, focus_execution_micro_attack_presence] if part) or '-'}",
        f"cvd-queue-status: {shortline_cvd_queue_handoff_status or '-'}",
        f"cvd-queue-focus: {shortline_cvd_queue_focus_batch or '-'}:{shortline_cvd_queue_focus_action or '-'}",
        f"next-retest: {next_retest_action or '-'}",
        f"reason: {next_focus_reason or '-'}",
    ]
    if shortline_no_trade_rule:
        operator_lines.append(f"shortline-no-trade: {shortline_no_trade_rule}")
    if focus_review_status and focus_review_status != "not_active":
        operator_lines.append(f"focus-review-status: {focus_review_status}")
        operator_lines.append(f"focus-review-primary: {focus_review_primary_blocker or '-'}")
        operator_lines.append(f"focus-review-micro: {focus_review_micro_blocker or '-'}")
        if focus_review_done_when:
            operator_lines.append(f"focus-review-done-when: {focus_review_done_when}")
    if focus_review_score_status == "scored":
        operator_lines.append(
            "focus-review-scores: "
            + " | ".join(
                [
                    f"edge={focus_review_edge_score}",
                    f"structure={focus_review_structure_score}",
                    f"micro={focus_review_micro_score}",
                    f"composite={focus_review_composite_score}",
                ]
            )
        )
    if focus_review_priority_status == "ready":
        operator_lines.append(
            f"focus-review-priority: {focus_review_priority_tier or '-'} | score={focus_review_priority_score}"
        )
    if review_priority_queue_status == "ready":
        operator_lines.append(f"review-priority-queue: {review_priority_queue_brief or '-'}")
    if str(head_source_refresh.get("status") or "").strip() != "not_active":
        operator_lines.append(f"head-source-refresh: {str(head_source_refresh.get('brief') or '-')}")
    if str(latest_refresh_audit.get("status") or "").strip() not in {"", "not_assessed", "not_available"}:
        operator_lines.append(f"latest-refresh-audit: {str(latest_refresh_audit.get('brief') or '-')}")
        operator_lines.append(f"latest-refresh-reuse-gate: {str(latest_refresh_reuse_gate.get('brief') or '-')}")
    if str(downstream_embedding.get("status") or "").strip() not in {"", "not_assessed"}:
        operator_lines.append(f"downstream-embedding: {str(downstream_embedding.get('brief') or '-')}")
    if focus_execution_micro_reasons:
        operator_lines.append(f"micro-reasons: {', '.join(str(x) for x in focus_execution_micro_reasons if str(x))}")
    if shortline_cvd_semantic_status:
        operator_lines.append(f"cvd-semantic-status: {shortline_cvd_semantic_status}")
    if shortline_cvd_semantic_takeaway:
        operator_lines.append(f"cvd-semantic-note: {shortline_cvd_semantic_takeaway}")
    if shortline_cvd_queue_handoff_takeaway:
        operator_lines.append(f"cvd-queue-note: {shortline_cvd_queue_handoff_takeaway}")
    if shortline_cvd_queue_stack_brief:
        operator_lines.append(f"cvd-queue-stack: {shortline_cvd_queue_stack_brief}")
    if next_retest_reason:
        operator_lines.append(f"next-retest-reason: {next_retest_reason}")
    if comparative_window_takeaway:
        operator_lines.append(f"focus-window-note: {comparative_window_takeaway}")
    if xlong_flow_window_floor:
        operator_lines.append(f"xlong-flow-floor: {xlong_flow_window_floor}")
    if xlong_comparative_window_takeaway:
        operator_lines.append(f"xlong-flow-note: {xlong_comparative_window_takeaway}")
    if focus_brief:
        operator_lines.append(f"focus-brief: {focus_brief}")
    operator_text = " | ".join(operator_lines)
    return {
        "operator_status": operator_status,
        "route_stack_brief": route_stack_brief,
        "next_focus_symbol": next_focus_symbol,
        "next_focus_action": next_focus_action,
        "next_focus_reason": next_focus_reason,
        "focus_window_gate": focus_window_gate,
        "focus_short_flow_combo_canonical": focus_short_flow_combo_canonical,
        "focus_long_flow_combo_canonical": focus_long_flow_combo_canonical,
        "focus_long_top_combo_canonical": focus_long_top_combo_canonical,
        "focus_window_verdict": focus_window_verdict,
        "focus_window_floor": focus_window_floor,
        "price_state_window_floor": price_state_window_floor,
        "comparative_window_takeaway": comparative_window_takeaway,
        "xlong_flow_window_floor": xlong_flow_window_floor,
        "xlong_comparative_window_takeaway": xlong_comparative_window_takeaway,
        "focus_brief": focus_brief,
        "next_retest_action": next_retest_action,
        "next_retest_reason": next_retest_reason,
        "shortline_market_state_brief": shortline_market_state_brief,
        "shortline_execution_gate_brief": shortline_execution_gate_brief,
        "shortline_session_map_brief": shortline_session_map_brief,
        "shortline_no_trade_rule": shortline_no_trade_rule,
        "shortline_cvd_semantic_status": shortline_cvd_semantic_status,
        "shortline_cvd_semantic_takeaway": shortline_cvd_semantic_takeaway,
        "shortline_cvd_queue_handoff_status": shortline_cvd_queue_handoff_status,
        "shortline_cvd_queue_handoff_takeaway": shortline_cvd_queue_handoff_takeaway,
        "shortline_cvd_queue_focus_batch": shortline_cvd_queue_focus_batch,
        "shortline_cvd_queue_focus_action": shortline_cvd_queue_focus_action,
        "shortline_cvd_queue_stack_brief": shortline_cvd_queue_stack_brief,
        "focus_execution_state": focus_execution_state,
        "focus_execution_blocker_detail": focus_execution_blocker_detail,
        "focus_execution_done_when": focus_execution_done_when,
        "focus_execution_micro_classification": focus_execution_micro_classification,
        "focus_execution_micro_context": focus_execution_micro_context,
        "focus_execution_micro_trust_tier": focus_execution_micro_trust_tier,
        "focus_execution_micro_veto": focus_execution_micro_veto,
        "focus_execution_micro_locality_status": focus_execution_micro_locality_status,
        "focus_execution_micro_drift_risk": focus_execution_micro_drift_risk,
        "focus_execution_micro_attack_side": focus_execution_micro_attack_side,
        "focus_execution_micro_attack_presence": focus_execution_micro_attack_presence,
        "focus_execution_micro_reasons": focus_execution_micro_reasons,
        "focus_review_status": focus_review_status,
        "focus_review_brief": focus_review_brief,
        "focus_review_primary_blocker": focus_review_primary_blocker,
        "focus_review_micro_blocker": focus_review_micro_blocker,
        "focus_review_blocker_detail": focus_review_blocker_detail,
        "focus_review_done_when": focus_review_done_when,
        "focus_review_score_status": focus_review_score_status,
        "focus_review_edge_score": focus_review_edge_score,
        "focus_review_structure_score": focus_review_structure_score,
        "focus_review_micro_score": focus_review_micro_score,
        "focus_review_composite_score": focus_review_composite_score,
        "focus_review_score_brief": focus_review_score_brief,
        "focus_review_priority_status": focus_review_priority_status,
        "focus_review_priority_score": focus_review_priority_score,
        "focus_review_priority_tier": focus_review_priority_tier,
        "focus_review_priority_brief": focus_review_priority_brief,
        "review_priority_queue_status": review_priority_queue_status,
        "review_priority_queue_count": review_priority_queue_count,
        "review_priority_queue_brief": review_priority_queue_brief,
        "review_priority_head_symbol": review_priority_head_symbol,
        "review_priority_head_tier": review_priority_head_tier,
        "review_priority_head_score": review_priority_head_score,
        "review_priority_queue": review_priority_queue,
        "crypto_route_head_source_refresh_status": str(head_source_refresh.get("status") or ""),
        "crypto_route_head_source_refresh_brief": str(head_source_refresh.get("brief") or ""),
        "crypto_route_head_source_refresh_symbol": str(head_source_refresh.get("symbol") or ""),
        "crypto_route_head_source_refresh_action": str(head_source_refresh.get("action") or ""),
        "crypto_route_head_source_refresh_source_kind": str(head_source_refresh.get("source_kind") or ""),
        "crypto_route_head_source_refresh_source_health": str(head_source_refresh.get("source_health") or ""),
        "crypto_route_head_source_refresh_source_artifact": str(head_source_refresh.get("source_artifact") or ""),
        "crypto_route_head_source_refresh_priority_tier": str(head_source_refresh.get("priority_tier") or ""),
        "crypto_route_head_source_refresh_route_action": str(head_source_refresh.get("route_action") or ""),
        "crypto_route_head_source_refresh_blocker_detail": str(head_source_refresh.get("blocker_detail") or ""),
        "crypto_route_head_source_refresh_done_when": str(head_source_refresh.get("done_when") or ""),
        "latest_crypto_route_refresh_status": str(latest_refresh_audit.get("status") or ""),
        "latest_crypto_route_refresh_brief": str(latest_refresh_audit.get("brief") or ""),
        "latest_crypto_route_refresh_artifact": str(latest_refresh_audit.get("artifact") or ""),
        "latest_crypto_route_refresh_as_of": str(latest_refresh_audit.get("as_of") or ""),
        "latest_crypto_route_refresh_native_mode": str(latest_refresh_audit.get("native_mode") or ""),
        "latest_crypto_route_refresh_native_step_count": int(latest_refresh_audit.get("native_step_count") or 0),
        "latest_crypto_route_refresh_reused_native_count": int(latest_refresh_audit.get("reused_native_count") or 0),
        "latest_crypto_route_refresh_missing_reused_count": int(latest_refresh_audit.get("missing_reused_count") or 0),
        "latest_crypto_route_refresh_note": str(latest_refresh_audit.get("note") or ""),
        "latest_crypto_route_refresh_done_when": str(latest_refresh_audit.get("done_when") or ""),
        "latest_crypto_route_refresh_reuse_level": str(latest_refresh_reuse_gate.get("level") or ""),
        "latest_crypto_route_refresh_reuse_gate_status": str(latest_refresh_reuse_gate.get("status") or ""),
        "latest_crypto_route_refresh_reuse_gate_brief": str(latest_refresh_reuse_gate.get("brief") or ""),
        "latest_crypto_route_refresh_reuse_gate_blocking": bool(latest_refresh_reuse_gate.get("blocking")),
        "latest_crypto_route_refresh_reuse_gate_blocker_detail": str(
            latest_refresh_reuse_gate.get("blocker_detail") or ""
        ),
        "latest_crypto_route_refresh_reuse_gate_done_when": str(
            latest_refresh_reuse_gate.get("done_when") or ""
        ),
        "crypto_route_head_downstream_embedding_status": str(downstream_embedding.get("status") or ""),
        "crypto_route_head_downstream_embedding_brief": str(downstream_embedding.get("brief") or ""),
        "crypto_route_head_downstream_embedding_artifact": str(downstream_embedding.get("artifact") or ""),
        "crypto_route_head_downstream_embedding_as_of": str(downstream_embedding.get("as_of") or ""),
        "crypto_route_head_downstream_embedding_blocker_detail": str(
            downstream_embedding.get("blocker_detail") or ""
        ),
        "crypto_route_head_downstream_embedding_done_when": str(downstream_embedding.get("done_when") or ""),
        "deploy_now_symbols": deploy_now_symbols,
        "watch_priority_symbols": watch_priority_symbols,
        "watch_only_symbols": watch_only_symbols,
        "operator_lines": operator_lines,
        "operator_text": operator_text,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Crypto Route Operator Brief",
        "",
        f"- as_of: `{payload.get('as_of') or ''}`",
        f"- source_artifact: `{payload.get('source_artifact') or ''}`",
        f"- crypto_route_head_source_refresh_status: `{payload.get('crypto_route_head_source_refresh_status') or ''}`",
        f"- crypto_route_head_source_refresh_brief: `{payload.get('crypto_route_head_source_refresh_brief') or ''}`",
        f"- crypto_route_head_source_refresh_symbol: `{payload.get('crypto_route_head_source_refresh_symbol') or ''}`",
        f"- crypto_route_head_source_refresh_action: `{payload.get('crypto_route_head_source_refresh_action') or ''}`",
        f"- crypto_route_head_source_refresh_source_kind: `{payload.get('crypto_route_head_source_refresh_source_kind') or ''}`",
        f"- crypto_route_head_source_refresh_source_health: `{payload.get('crypto_route_head_source_refresh_source_health') or ''}`",
        f"- crypto_route_head_source_refresh_source_artifact: `{payload.get('crypto_route_head_source_refresh_source_artifact') or ''}`",
        f"- crypto_route_head_source_refresh_priority_tier: `{payload.get('crypto_route_head_source_refresh_priority_tier') or ''}`",
        f"- crypto_route_head_source_refresh_route_action: `{payload.get('crypto_route_head_source_refresh_route_action') or ''}`",
        f"- crypto_route_head_source_refresh_blocker_detail: `{payload.get('crypto_route_head_source_refresh_blocker_detail') or ''}`",
        f"- crypto_route_head_source_refresh_done_when: `{payload.get('crypto_route_head_source_refresh_done_when') or ''}`",
        f"- latest_crypto_route_refresh_status: `{payload.get('latest_crypto_route_refresh_status') or ''}`",
        f"- latest_crypto_route_refresh_brief: `{payload.get('latest_crypto_route_refresh_brief') or ''}`",
        f"- latest_crypto_route_refresh_artifact: `{payload.get('latest_crypto_route_refresh_artifact') or ''}`",
        f"- latest_crypto_route_refresh_as_of: `{payload.get('latest_crypto_route_refresh_as_of') or ''}`",
        f"- latest_crypto_route_refresh_native_mode: `{payload.get('latest_crypto_route_refresh_native_mode') or ''}`",
        f"- latest_crypto_route_refresh_native_step_count: `{payload.get('latest_crypto_route_refresh_native_step_count')}`",
        f"- latest_crypto_route_refresh_reused_native_count: `{payload.get('latest_crypto_route_refresh_reused_native_count')}`",
        f"- latest_crypto_route_refresh_missing_reused_count: `{payload.get('latest_crypto_route_refresh_missing_reused_count')}`",
        f"- latest_crypto_route_refresh_note: `{payload.get('latest_crypto_route_refresh_note') or ''}`",
        f"- latest_crypto_route_refresh_done_when: `{payload.get('latest_crypto_route_refresh_done_when') or ''}`",
        f"- latest_crypto_route_refresh_reuse_level: `{payload.get('latest_crypto_route_refresh_reuse_level') or ''}`",
        f"- latest_crypto_route_refresh_reuse_gate_status: `{payload.get('latest_crypto_route_refresh_reuse_gate_status') or ''}`",
        f"- latest_crypto_route_refresh_reuse_gate_brief: `{payload.get('latest_crypto_route_refresh_reuse_gate_brief') or ''}`",
        f"- latest_crypto_route_refresh_reuse_gate_blocking: `{payload.get('latest_crypto_route_refresh_reuse_gate_blocking')}`",
        f"- latest_crypto_route_refresh_reuse_gate_blocker_detail: `{payload.get('latest_crypto_route_refresh_reuse_gate_blocker_detail') or ''}`",
        f"- latest_crypto_route_refresh_reuse_gate_done_when: `{payload.get('latest_crypto_route_refresh_reuse_gate_done_when') or ''}`",
        f"- crypto_route_head_downstream_embedding_status: `{payload.get('crypto_route_head_downstream_embedding_status') or ''}`",
        f"- crypto_route_head_downstream_embedding_brief: `{payload.get('crypto_route_head_downstream_embedding_brief') or ''}`",
        f"- crypto_route_head_downstream_embedding_artifact: `{payload.get('crypto_route_head_downstream_embedding_artifact') or ''}`",
        f"- crypto_route_head_downstream_embedding_as_of: `{payload.get('crypto_route_head_downstream_embedding_as_of') or ''}`",
        f"- crypto_route_head_downstream_embedding_blocker_detail: `{payload.get('crypto_route_head_downstream_embedding_blocker_detail') or ''}`",
        f"- crypto_route_head_downstream_embedding_done_when: `{payload.get('crypto_route_head_downstream_embedding_done_when') or ''}`",
        "",
        "## Brief",
    ]
    for line in payload.get("operator_lines", []):
        lines.append(f"- {line}")
    return "\n".join(lines).strip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a short operator brief from the latest crypto route brief artifact.")
    parser.add_argument("--review-dir", required=True)
    parser.add_argument("--now", default="")
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--artifact-keep", type=int, default=12)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    runtime_now = parse_now(args.now)

    source_path = latest_crypto_route_brief(review_dir, runtime_now)
    source_payload = json.loads(source_path.read_text(encoding="utf-8"))

    stamp = runtime_now.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = review_dir / f"{stamp}_crypto_route_operator_brief.json"
    md_path = review_dir / f"{stamp}_crypto_route_operator_brief.md"
    checksum_path = review_dir / f"{stamp}_crypto_route_operator_brief_checksum.json"
    brief = build_operator_brief(
        source_payload,
        current_artifact=str(json_path),
        review_dir=review_dir,
        reference_now=runtime_now,
    )

    payload = {
        "ok": True,
        "status": "ok",
        "as_of": fmt_utc(runtime_now),
        "source_artifact": str(source_path),
        **brief,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    checksum_payload = {
        "generated_at_utc": payload["as_of"],
        "artifact_ttl_hours": max(1.0, float(args.artifact_ttl_hours)),
        "files": [
            {"path": str(json_path), "sha256": sha256_file(json_path)},
            {"path": str(md_path), "sha256": sha256_file(md_path)},
        ],
    }
    checksum_path.write_text(json.dumps(checksum_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    pruned_keep, pruned_age = prune_review_artifacts(
        review_dir,
        stem="crypto_route_operator_brief",
        current_paths=[json_path, md_path, checksum_path],
        keep=max(3, int(args.artifact_keep)),
        ttl_hours=float(args.artifact_ttl_hours),
        now_dt=runtime_now,
    )

    payload.update(
        {
            "artifact": str(json_path),
            "markdown": str(md_path),
            "checksum": str(checksum_path),
            "pruned_keep": pruned_keep,
            "pruned_age": pruned_age,
        }
    )
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
