#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml


TIMESTAMP_RE = re.compile(r"^(?P<stamp>\d{8}T\d{6}Z)_")
FUTURE_STAMP_GRACE_MINUTES = 5
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
DEFAULT_SHORTLINE_TRIGGER_STACK = (
    "4h_profile_location",
    "liquidity_sweep",
    "1m_5m_mss_or_choch",
    "15m_cvd_divergence_or_confirmation",
    "fvg_ob_breaker_retest",
    "15m_reversal_or_breakout_candle",
)
DEFAULT_SHORTLINE_SESSIONS = (
    "asia_high_low",
    "london_high_low",
    "prior_day_high_low",
    "equal_highs_lows",
)


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


def resolve_path(raw: str, *, anchor: Path) -> Path:
    path = Path(str(raw).strip())
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    return anchor / path


def normalize_string_list(raw: Any, *, default: tuple[str, ...]) -> list[str]:
    if isinstance(raw, (list, tuple)):
        items = [str(x).strip() for x in raw]
    else:
        items = [str(x).strip() for x in str(raw).split(",")]
    out = [item for item in items if item]
    return out if out else list(default)


def load_shortline_policy(config_path: Path) -> dict[str, Any]:
    payload: Any = {}
    source = "builtin_default"
    if config_path.exists():
        try:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            source = str(config_path)
        except Exception:
            payload = {}
            source = f"invalid_config:{config_path}"
    shortline = payload.get("shortline", {}) if isinstance(payload, dict) else {}
    return {
        "source": source,
        "status": str(shortline.get("status", "builtin_default")),
        "market_state_engine": str(shortline.get("market_state_engine", "bias_only_vs_setup_ready")),
        "default_market_state": str(shortline.get("default_market_state", "Bias_Only")),
        "setup_ready_state": str(shortline.get("setup_ready_state", "Setup_Ready")),
        "no_trade_rule": str(shortline.get("no_trade_rule", "no_sweep_no_mss_no_cvd_no_trade")),
        "trigger_stack": normalize_string_list(
            shortline.get("trigger_stack", DEFAULT_SHORTLINE_TRIGGER_STACK),
            default=DEFAULT_SHORTLINE_TRIGGER_STACK,
        ),
        "session_liquidity_map": normalize_string_list(
            shortline.get("session_liquidity_map", DEFAULT_SHORTLINE_SESSIONS),
            default=DEFAULT_SHORTLINE_SESSIONS,
        ),
        "execution_style": str(shortline.get("execution_style", "right_side_only")),
        "micro_structure_engine": str(shortline.get("micro_structure_engine", "ict_sweep_mss")),
        "micro_structure_timeframes": normalize_string_list(
            shortline.get("micro_structure_timeframes", ("1m", "5m")),
            default=("1m", "5m"),
        ),
    }


def latest_symbol_route_handoff(review_dir: Path, reference_now: dt.datetime | None = None) -> Path:
    candidates = list(review_dir.glob("*_binance_indicator_symbol_route_handoff.json"))
    if not candidates:
        raise FileNotFoundError("no_binance_indicator_symbol_route_handoff_artifact")
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def latest_bnb_flow_focus(review_dir: Path, reference_now: dt.datetime | None = None) -> Path | None:
    candidates = list(review_dir.glob("*_binance_indicator_bnb_flow_focus.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def latest_shortline_execution_gate(review_dir: Path, reference_now: dt.datetime | None = None) -> Path | None:
    candidates = list(review_dir.glob("*_crypto_shortline_execution_gate.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def latest_cvd_semantic_snapshot(review_dir: Path, reference_now: dt.datetime | None = None) -> Path | None:
    candidates = list(review_dir.glob("*_crypto_cvd_semantic_snapshot.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def latest_cvd_queue_handoff(review_dir: Path, reference_now: dt.datetime | None = None) -> Path | None:
    candidates = list(review_dir.glob("*_crypto_cvd_queue_handoff.json"))
    if not candidates:
        return None
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


def _semantic_time_sync_fields(cvd_semantic_snapshot_payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(cvd_semantic_snapshot_payload or {})
    status = str(payload.get("time_sync_status") or "").strip()
    classification = str(payload.get("time_sync_classification") or "").strip()
    blocker_detail = str(payload.get("time_sync_blocker_detail") or "").strip()
    remediation_hint = str(payload.get("time_sync_remediation_hint") or "").strip()
    intercept_scope = str(payload.get("time_sync_intercept_scope") or "").strip()
    fake_ip_sources = [
        str(x).strip()
        for x in list(payload.get("time_sync_fake_ip_sources") or [])
        if str(x).strip()
    ]
    threshold_breach_sources = [
        str(x).strip()
        for x in list(payload.get("time_sync_threshold_breach_sources") or [])
        if str(x).strip()
    ]
    threshold_breach_scope = str(payload.get("time_sync_threshold_breach_scope") or "").strip()
    threshold_breach_offset_sources = [
        str(x).strip()
        for x in list(payload.get("time_sync_threshold_breach_offset_sources") or [])
        if str(x).strip()
    ]
    threshold_breach_latency_sources = [
        str(x).strip()
        for x in list(payload.get("time_sync_threshold_breach_latency_sources") or [])
        if str(x).strip()
    ]
    threshold_breach_estimated_offset_ms = payload.get("time_sync_threshold_breach_estimated_offset_ms")
    threshold_breach_estimated_rtt_ms = payload.get("time_sync_threshold_breach_estimated_rtt_ms")
    if not blocker_detail and classification == "fake_ip_dns_intercept":
        blocker_detail = (
            f"fake_ip_sources={','.join(fake_ip_sources)}"
            if fake_ip_sources
            else "fake_ip_sources=unknown"
        )
    return {
        "status": status,
        "classification": classification,
        "intercept_scope": intercept_scope,
        "blocker_detail": blocker_detail,
        "remediation_hint": remediation_hint,
        "fake_ip_sources": fake_ip_sources,
        "threshold_breach_sources": threshold_breach_sources,
        "threshold_breach_scope": threshold_breach_scope,
        "threshold_breach_offset_sources": threshold_breach_offset_sources,
        "threshold_breach_latency_sources": threshold_breach_latency_sources,
        "threshold_breach_estimated_offset_ms": threshold_breach_estimated_offset_ms,
        "threshold_breach_estimated_rtt_ms": threshold_breach_estimated_rtt_ms,
    }


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


def _focus_review_lane(
    *,
    symbol: str,
    action: str,
    focus_execution_state: str,
    focus_execution_blocker_detail: str,
    focus_execution_done_when: str,
    focus_execution_micro_veto: str,
) -> dict[str, str]:
    symbol_text = str(symbol or "").strip().upper()
    action_text = str(action or "").strip()
    lane = {
        "status": "not_active",
        "brief": "not_active:-",
        "primary_blocker": "not_applicable",
        "micro_blocker": "-",
        "blocker_detail": "crypto review lane is only active when crypto focus is under deprioritize_flow review.",
        "done_when": "crypto focus returns to a deprioritize_flow review state before reassessing",
    }
    if not symbol_text or action_text != "deprioritize_flow":
        return lane

    state_text = str(focus_execution_state or "").strip()
    blocker_detail_text = str(focus_execution_blocker_detail or "").strip()
    done_when_text = str(focus_execution_done_when or "").strip()
    micro_veto_text = str(focus_execution_micro_veto or "").strip()

    status_parts = ["no_edge"]
    if state_text == "Bias_Only":
        status_parts.append("bias_only")
    if micro_veto_text and micro_veto_text != "-":
        status_parts.append("micro_veto")

    lane["status"] = f"review_{'_'.join(status_parts)}"
    lane["brief"] = f"{lane['status']}:{symbol_text}"
    lane["primary_blocker"] = "no_edge"
    lane["micro_blocker"] = micro_veto_text or "-"
    lane["blocker_detail"] = blocker_detail_text or f"{symbol_text} remains under flow review."
    lane["done_when"] = done_when_text or f"{symbol_text} regains a positive ranked flow edge or leaves review"
    return lane


def _focus_review_scores(
    *,
    symbol: str,
    action: str,
    focus_execution_state: str,
    focus_execution_micro_classification: str,
    focus_execution_micro_veto: str,
    focus_execution_micro_locality_status: str,
    focus_execution_micro_drift_risk: str,
    focus_execution_micro_attack_side: str,
    focus_execution_micro_attack_presence: str,
    review_primary_blocker: str,
) -> dict[str, Any]:
    symbol_text = str(symbol or "").strip().upper()
    action_text = str(action or "").strip()
    state_text = str(focus_execution_state or "").strip()
    micro_class_text = str(focus_execution_micro_classification or "").strip()
    micro_veto_text = str(focus_execution_micro_veto or "").strip()
    micro_locality_text = str(focus_execution_micro_locality_status or "").strip()
    micro_drift_text = str(focus_execution_micro_drift_risk or "").strip().lower()
    micro_attack_side_text = str(focus_execution_micro_attack_side or "").strip()
    micro_attack_presence_text = str(focus_execution_micro_attack_presence or "").strip()
    primary_blocker_text = str(review_primary_blocker or "").strip()
    scores: dict[str, Any] = {
        "status": "not_active",
        "edge_score": 0,
        "structure_score": 0,
        "micro_score": 0,
        "composite_score": 0,
        "brief": "not_active:edge=0|structure=0|micro=0|composite=0",
    }
    if not symbol_text or action_text != "deprioritize_flow":
        return scores

    edge_score = 35
    if primary_blocker_text == "no_edge":
        edge_score = 5
    elif primary_blocker_text == "bias_only":
        edge_score = 15

    structure_score = {
        "Setup_Ready": 100,
        "Bias_Only": 25,
    }.get(state_text, 40 if state_text else 0)

    micro_score = {
        "confirmed": 100,
        "confirm_and_veto_only": 65,
        "watch_only": 35,
    }.get(micro_class_text, 40 if micro_class_text else 0)
    if micro_veto_text == "low_sample_or_gap_risk":
        micro_score = min(micro_score, 20)
    elif micro_veto_text and micro_veto_text != "-":
        micro_score = min(micro_score, 25)
    if micro_drift_text == "true":
        micro_score = min(micro_score, 5)
    elif micro_locality_text == "outside_local_window":
        micro_score = min(micro_score, 10)
    elif micro_locality_text == "proxy_from_current_snapshot":
        micro_score = min(micro_score, 15)
    elif micro_locality_text == "local_window_ok":
        micro_score = min(100, micro_score + 10)
        if (
            micro_attack_side_text
            and micro_attack_presence_text
            and micro_attack_presence_text != "-"
        ):
            micro_score = min(100, micro_score + 5)

    composite_score = int(round((edge_score + structure_score + micro_score) / 3.0))
    scores.update(
        {
            "status": "scored",
            "edge_score": edge_score,
            "structure_score": structure_score,
            "micro_score": micro_score,
            "composite_score": composite_score,
            "brief": (
                f"scored:{symbol_text}:"
                f"edge={edge_score}|structure={structure_score}|micro={micro_score}|composite={composite_score}"
            ),
        }
    )
    return scores


def _focus_review_priority(scores: dict[str, Any]) -> dict[str, Any]:
    if str(scores.get("status") or "").strip() != "scored":
        return {
            "status": "not_active",
            "score": 0,
            "tier": "-",
            "brief": "not_active:0/100",
        }
    score = int(scores.get("composite_score") or 0)
    if score >= 70:
        tier = "high_priority_review"
    elif score >= 40:
        tier = "medium_priority_review"
    elif score >= 20:
        tier = "low_priority_review"
    else:
        tier = "deprioritized_review"
    return {
        "status": "ready",
        "score": score,
        "tier": tier,
        "brief": f"{tier}:{score}/100",
    }


def _crypto_review_priority_queue(
    *,
    source_payload: dict[str, Any],
    shortline_execution_gate_payload: dict[str, Any] | None,
    cvd_semantic_snapshot_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    route_rows = [dict(row) for row in source_payload.get("routes", []) if isinstance(row, dict)]
    next_focus_symbol = str(source_payload.get("next_focus_symbol") or "").strip().upper()
    next_focus_action = str(source_payload.get("next_focus_action") or "").strip()
    if not route_rows:
        review_symbols = [
            str(x).strip().upper() for x in source_payload.get("review_symbols", []) if str(x).strip()
        ]
        watch_priority_symbols = [
            str(x).strip().upper() for x in source_payload.get("watch_priority_symbols", []) if str(x).strip()
        ]
        watch_only_symbols = [
            str(x).strip().upper() for x in source_payload.get("watch_only_symbols", []) if str(x).strip()
        ]
        route_rows = [
            {
                "symbol": symbol,
                "status_label": "review",
                "action": next_focus_action if symbol == next_focus_symbol and next_focus_action else "deprioritize_flow",
                "reason": str(source_payload.get("next_focus_reason") or "") if symbol == next_focus_symbol else "",
            }
            for symbol in review_symbols
        ] + [
            {
                "symbol": symbol,
                "status_label": "watch_priority",
                "action": "watch_priority_until_long_window_confirms",
                "reason": "",
            }
            for symbol in watch_priority_symbols
        ] + [
            {
                "symbol": symbol,
                "status_label": "watch_only",
                "action": "watch_only",
                "reason": "",
            }
            for symbol in watch_only_symbols
        ]

    gate_by_symbol = {
        str(row.get("symbol") or "").strip().upper(): dict(row)
        for row in (shortline_execution_gate_payload or {}).get("symbols", [])
        if isinstance(row, dict) and str(row.get("symbol") or "").strip()
    }
    semantic_by_symbol = {
        str(row.get("symbol") or "").strip().upper(): dict(row)
        for row in (cvd_semantic_snapshot_payload or {}).get("symbols", [])
        if isinstance(row, dict) and str(row.get("symbol") or "").strip()
    }
    semantic_time_sync = _semantic_time_sync_fields(cvd_semantic_snapshot_payload)

    items: list[dict[str, Any]] = []
    for route in route_rows:
        symbol = str(route.get("symbol") or "").strip().upper()
        status_label = str(route.get("status_label") or "").strip()
        action = str(route.get("action") or "").strip()
        if not symbol or status_label not in {"review", "watch_priority", "watch_only"}:
            continue

        gate_row = gate_by_symbol.get(symbol, {})
        semantic_row = semantic_by_symbol.get(symbol, {})
        execution_state = str(gate_row.get("execution_state") or "Bias_Only").strip()
        micro_classification = str(semantic_row.get("classification") or "").strip()
        gate_micro = gate_row.get("micro_signals") if isinstance(gate_row.get("micro_signals"), dict) else {}
        micro_veto = str(
            semantic_row.get("cvd_veto_hint")
            or gate_micro.get("veto_hint")
            or ""
        ).strip()
        micro_locality_status = str(
            semantic_row.get("cvd_locality_status")
            or gate_micro.get("cvd_locality_status")
            or ""
        ).strip()
        micro_drift_risk = bool(
            semantic_row.get("cvd_drift_risk")
            or gate_micro.get("cvd_drift_risk")
        )
        micro_attack_side = str(
            semantic_row.get("cvd_attack_side")
            or gate_micro.get("attack_side")
            or ""
        ).strip()
        micro_attack_presence = str(
            semantic_row.get("cvd_attack_presence")
            or gate_micro.get("attack_presence")
            or ""
        ).strip()
        blocker_detail = str(gate_row.get("blocker_detail") or route.get("reason") or "").strip()
        done_when = str(gate_row.get("done_when") or "").strip()
        active_reasons = [
            str(x).strip()
            for x in list(semantic_row.get("active_reasons") or [])
            if str(x).strip()
        ]
        if "time_sync_risk" in active_reasons and str(semantic_time_sync.get("blocker_detail") or "").strip():
            time_sync_status = str(
                semantic_time_sync.get("classification")
                or semantic_time_sync.get("status")
                or "time_sync_risk"
            ).strip()
            blocker_detail = (
                f"{blocker_detail} | time-sync={time_sync_status}:{semantic_time_sync.get('blocker_detail')}"
                if blocker_detail
                else f"time-sync={time_sync_status}:{semantic_time_sync.get('blocker_detail')}"
            )
            time_sync_remediation = str(semantic_time_sync.get("remediation_hint") or "").strip()
            if time_sync_remediation:
                done_when = (
                    f"{done_when} | source: {time_sync_remediation}"
                    if done_when and done_when != "-"
                    else time_sync_remediation
                )

        base_score = {
            "review": 60,
            "watch_priority": 40,
            "watch_only": 20,
        }.get(status_label, 0)
        if action == "deprioritize_flow":
            base_score += 5
        elif action == "candidate_flow_secondary":
            base_score += 8
        elif action in {"watch_priority_until_long_window_confirms", "watch_short_window_flow_priority"}:
            base_score += 5

        structure_bonus = {
            "Setup_Ready": 25,
            "Bias_Only": 10,
        }.get(execution_state, 0)
        micro_bonus = {
            "confirmed": 15,
            "confirm_and_veto_only": 8,
            "watch_only": 3,
        }.get(micro_classification, 0)
        veto_penalty = 0
        if micro_veto == "missing_micro_capture":
            veto_penalty = 10
        elif micro_veto == "low_sample_or_gap_risk":
            veto_penalty = 5
        elif micro_veto and micro_veto != "-":
            veto_penalty = 3
        locality_bonus = 0
        locality_penalty = 0
        attack_bonus = 0
        if micro_locality_status == "local_window_ok":
            locality_bonus = 6
            if micro_attack_side and micro_attack_presence and micro_attack_presence != "-":
                attack_bonus = 4
        elif micro_locality_status == "proxy_from_current_snapshot":
            locality_penalty = 15
        elif micro_locality_status == "outside_local_window":
            locality_penalty = 18
        if micro_drift_risk:
            locality_penalty += 20

        priority_score = max(
            0,
            min(
                100,
                base_score + structure_bonus + micro_bonus + locality_bonus + attack_bonus - veto_penalty - locality_penalty,
            ),
        )
        if priority_score >= 60:
            priority_tier = "review_queue_now"
        elif priority_score >= 40:
            priority_tier = "review_queue_next"
        else:
            priority_tier = "watch_queue_only"

        items.append(
            {
                "symbol": symbol,
                "route_action": action or "-",
                "route_status_label": status_label or "-",
                "execution_state": execution_state or "-",
                "micro_classification": micro_classification or "-",
                "micro_veto": micro_veto or "-",
                "micro_locality_status": micro_locality_status or "-",
                "micro_drift_risk": micro_drift_risk,
                "micro_attack_side": micro_attack_side or "-",
                "micro_attack_presence": micro_attack_presence or "-",
                "priority_score": priority_score,
                "priority_tier": priority_tier,
                "reason": str(route.get("reason") or "").strip(),
                "blocker_detail": blocker_detail or "-",
                "done_when": done_when or "-",
            }
        )

    order_priority = {"review": 3, "watch_priority": 2, "watch_only": 1}
    items.sort(
        key=lambda row: (
            int(row.get("priority_score") or 0),
            order_priority.get(str(row.get("route_status_label") or ""), 0),
            str(row.get("symbol") or ""),
        ),
        reverse=True,
    )
    for idx, row in enumerate(items, start=1):
        row["rank"] = idx

    brief = (
        " | ".join(
            f"{int(row.get('rank') or 0)}:{row.get('symbol') or '-'}:{row.get('priority_tier') or '-'}:{int(row.get('priority_score') or 0)}"
            for row in items
        )
        if items
        else "-"
    )
    head = items[0] if items else {}
    return {
        "status": "ready" if items else "empty",
        "count": len(items),
        "items": items,
        "brief": brief,
        "head_symbol": str(head.get("symbol") or ""),
        "head_tier": str(head.get("priority_tier") or ""),
        "head_score": int(head.get("priority_score") or 0),
    }


def _crypto_route_head_source_refresh_lane(
    *,
    review_priority_queue: dict[str, Any],
    next_focus_action: str,
    current_artifact: str = "",
) -> dict[str, str]:
    head_symbol = str(review_priority_queue.get("head_symbol") or "").strip().upper()
    head_tier = str(review_priority_queue.get("head_tier") or "").strip() or "-"
    items = [dict(row) for row in review_priority_queue.get("items", []) if isinstance(row, dict)]
    head_item: dict[str, Any] = {}
    for row in items:
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
        or str(next_focus_action or "").strip()
        or "-"
    )
    source_artifact = str(current_artifact or "").strip()
    source_health = "ready" if source_artifact else "missing"
    action = "read_current_artifact" if head_symbol and source_artifact else "-"
    status = "ready" if action == "read_current_artifact" else "not_active"
    blocker_detail = "-"
    done_when = "-"
    if head_symbol and action == "read_current_artifact":
        blocker_detail = (
            f"{head_symbol} currently uses the readable crypto_route brief artifact "
            f"while priority_tier={head_tier} and route_action={route_action}."
        )
        done_when = (
            f"keep {head_symbol} on the current crypto_route brief artifact until the queue head changes "
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


def _crypto_route_head_downstream_embedding_lane(
    *,
    review_priority_queue: dict[str, Any],
    next_focus_action: str,
    current_artifact: str = "",
    review_dir: Path | None = None,
    reference_now: dt.datetime | None = None,
) -> dict[str, str]:
    head_source_refresh = _crypto_route_head_source_refresh_lane(
        review_priority_queue=review_priority_queue,
        next_focus_action=next_focus_action,
        current_artifact=current_artifact,
    )
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
            "blocker_detail": "hot_universe_research artifact was not checked while building the crypto route brief.",
            "done_when": "rebuild the crypto route brief with review_dir context to assess downstream embedding freshness",
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
                    "and remains broader carry-over outside route-brief scope."
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
                "and downstream hot_universe_research is current enough for route-brief scope."
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
            "note": "latest crypto_route_refresh was not checked while building the crypto route brief.",
            "done_when": "rebuild the crypto route brief with review_dir context to assess the latest refresh audit",
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
            "blocker_detail": "latest crypto_route_refresh reused all tracked native steps; current route brief may safely read the reused native path.",
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
            "blocker_detail": f"latest crypto_route_refresh audit is not fully reusable-safe ({brief or status}); force a full native refresh before trusting route-brief reuse.",
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


def build_brief(
    source_payload: dict[str, Any],
    bnb_focus_payload: dict[str, Any] | None = None,
    shortline_policy: dict[str, Any] | None = None,
    shortline_execution_gate_payload: dict[str, Any] | None = None,
    cvd_semantic_snapshot_payload: dict[str, Any] | None = None,
    cvd_queue_handoff_payload: dict[str, Any] | None = None,
    *,
    current_artifact: str = "",
    review_dir: Path | None = None,
    reference_now: dt.datetime | None = None,
) -> dict[str, Any]:
    deploy_now = [str(x).strip().upper() for x in source_payload.get("deploy_now_symbols", []) if str(x).strip()]
    watch_priority = [str(x).strip().upper() for x in source_payload.get("watch_priority_symbols", []) if str(x).strip()]
    watch_only = [str(x).strip().upper() for x in source_payload.get("watch_only_symbols", []) if str(x).strip()]
    review = [str(x).strip().upper() for x in source_payload.get("review_symbols", []) if str(x).strip()]
    next_focus_symbol = str(source_payload.get("next_focus_symbol", "") or "").strip().upper()
    next_focus_action = str(source_payload.get("next_focus_action", "") or "").strip()
    next_focus_reason = str(source_payload.get("next_focus_reason", "") or "").strip()
    operator_status = str(source_payload.get("operator_status", "") or "").strip()
    route_stack_brief = str(source_payload.get("route_stack_brief", "") or "").strip()
    overall_takeaway = str(source_payload.get("overall_takeaway", "") or "").strip()
    shortline = dict(shortline_policy or {})
    shortline_default_market_state = str(shortline.get("default_market_state") or "Bias_Only").strip()
    shortline_setup_ready_state = str(shortline.get("setup_ready_state") or "Setup_Ready").strip()
    shortline_no_trade_rule = str(shortline.get("no_trade_rule") or "no_sweep_no_mss_no_cvd_no_trade").strip()
    shortline_trigger_stack = [
        str(x).strip() for x in shortline.get("trigger_stack", DEFAULT_SHORTLINE_TRIGGER_STACK) if str(x).strip()
    ] or list(DEFAULT_SHORTLINE_TRIGGER_STACK)
    shortline_session_liquidity_map = [
        str(x).strip() for x in shortline.get("session_liquidity_map", DEFAULT_SHORTLINE_SESSIONS) if str(x).strip()
    ] or list(DEFAULT_SHORTLINE_SESSIONS)
    shortline_execution_style = str(shortline.get("execution_style") or "right_side_only").strip()
    shortline_market_state_brief = (
        f"{shortline_default_market_state}->{shortline_setup_ready_state} | no_trade={shortline_no_trade_rule}"
    )
    shortline_execution_gate_brief = " -> ".join(shortline_trigger_stack)
    shortline_session_map_brief = ", ".join(shortline_session_liquidity_map)
    focus_execution_state = shortline_default_market_state
    focus_execution_blocker_detail = (
        "route focus is unavailable; keep Bias_Only until a crypto route focus exists and the shortline trigger stack completes."
    )
    focus_execution_done_when = (
        f"next crypto route focus reaches {shortline_setup_ready_state} by completing {shortline_execution_gate_brief}"
    )
    shortline_execution_gate_status = str((shortline_execution_gate_payload or {}).get("status") or "")
    shortline_execution_gate_artifact = str((shortline_execution_gate_payload or {}).get("artifact") or "")
    shortline_cvd_semantic_status = str((cvd_semantic_snapshot_payload or {}).get("status") or "")
    shortline_cvd_semantic_takeaway = str((cvd_semantic_snapshot_payload or {}).get("takeaway") or "").strip()
    shortline_cvd_semantic_environment_status = str(
        (cvd_semantic_snapshot_payload or {}).get("environment_status") or ""
    ).strip()
    shortline_cvd_semantic_environment_classification = str(
        (cvd_semantic_snapshot_payload or {}).get("environment_classification") or ""
    ).strip()
    shortline_cvd_semantic_environment_blocker_detail = str(
        (cvd_semantic_snapshot_payload or {}).get("environment_blocker_detail") or ""
    ).strip()
    semantic_time_sync = _semantic_time_sync_fields(cvd_semantic_snapshot_payload)
    shortline_cvd_semantic_time_sync_status = str(semantic_time_sync.get("status") or "").strip()
    shortline_cvd_semantic_time_sync_classification = str(
        semantic_time_sync.get("classification") or ""
    ).strip()
    shortline_cvd_semantic_time_sync_intercept_scope = str(
        semantic_time_sync.get("intercept_scope") or ""
    ).strip()
    shortline_cvd_semantic_time_sync_blocker_detail = str(
        semantic_time_sync.get("blocker_detail") or ""
    ).strip()
    shortline_cvd_semantic_time_sync_remediation_hint = str(
        semantic_time_sync.get("remediation_hint") or ""
    ).strip()
    shortline_cvd_semantic_time_sync_fake_ip_sources = list(
        semantic_time_sync.get("fake_ip_sources") or []
    )
    shortline_cvd_semantic_time_sync_threshold_breach_sources = list(
        semantic_time_sync.get("threshold_breach_sources") or []
    )
    shortline_cvd_semantic_time_sync_threshold_breach_scope = str(
        semantic_time_sync.get("threshold_breach_scope") or ""
    ).strip()
    shortline_cvd_semantic_time_sync_threshold_breach_offset_sources = list(
        semantic_time_sync.get("threshold_breach_offset_sources") or []
    )
    shortline_cvd_semantic_time_sync_threshold_breach_latency_sources = list(
        semantic_time_sync.get("threshold_breach_latency_sources") or []
    )
    shortline_cvd_semantic_time_sync_threshold_breach_estimated_offset_ms = semantic_time_sync.get(
        "threshold_breach_estimated_offset_ms"
    )
    shortline_cvd_semantic_time_sync_threshold_breach_estimated_rtt_ms = semantic_time_sync.get(
        "threshold_breach_estimated_rtt_ms"
    )
    shortline_cvd_queue_handoff_status = str(
        (cvd_queue_handoff_payload or {}).get("operator_status")
        or (cvd_queue_handoff_payload or {}).get("status")
        or ""
    )
    shortline_cvd_queue_handoff_takeaway = str((cvd_queue_handoff_payload or {}).get("takeaway") or "").strip()
    shortline_cvd_queue_focus_batch = str((cvd_queue_handoff_payload or {}).get("next_focus_batch") or "").strip()
    shortline_cvd_queue_focus_action = str((cvd_queue_handoff_payload or {}).get("next_focus_action") or "").strip()
    shortline_cvd_queue_stack_brief = str((cvd_queue_handoff_payload or {}).get("queue_stack_brief") or "").strip()
    focus_execution_micro_classification = ""
    focus_execution_micro_context = ""
    focus_execution_micro_trust_tier = ""
    focus_execution_micro_veto = ""
    focus_execution_micro_locality_status = ""
    focus_execution_micro_drift_risk = ""
    focus_execution_micro_attack_side = ""
    focus_execution_micro_attack_presence = ""
    focus_execution_micro_reasons: list[str] = []
    focus_window_gate = ""
    focus_window_gate_reason = ""
    focus_window_verdict = ""
    focus_window_floor = ""
    price_state_window_floor = ""
    comparative_window_takeaway = ""
    xlong_flow_window_floor = ""
    xlong_comparative_window_takeaway = ""
    focus_short_flow_combo = ""
    focus_short_flow_combo_canonical = ""
    focus_long_flow_combo = ""
    focus_long_flow_combo_canonical = ""
    focus_long_top_combo = ""
    focus_long_top_combo_canonical = ""
    focus_brief = ""
    next_retest_action = ""
    next_retest_reason = ""
    if bnb_focus_payload and next_focus_symbol == "BNBUSDT":
        focus_window_gate = str(bnb_focus_payload.get("promotion_gate", "") or "").strip()
        focus_window_gate_reason = str(bnb_focus_payload.get("promotion_gate_reason", "") or "").strip()
        focus_short_flow_combo = str(bnb_focus_payload.get("short_flow_combo", "") or "").strip()
        focus_short_flow_combo_canonical = str(bnb_focus_payload.get("short_flow_combo_canonical", "") or "").strip()
        focus_long_flow_combo = str(bnb_focus_payload.get("long_flow_combo", "") or "").strip()
        focus_long_flow_combo_canonical = str(bnb_focus_payload.get("long_flow_combo_canonical", "") or "").strip()
        focus_long_top_combo = str(bnb_focus_payload.get("long_top_combo", "") or "").strip()
        focus_long_top_combo_canonical = str(bnb_focus_payload.get("long_top_combo_canonical", "") or "").strip()
        focus_window_verdict = str(bnb_focus_payload.get("flow_window_verdict", "") or "").strip()
        focus_window_floor = str(bnb_focus_payload.get("flow_window_floor", "") or "").strip()
        price_state_window_floor = str(bnb_focus_payload.get("price_state_window_floor", "") or "").strip()
        comparative_window_takeaway = str(bnb_focus_payload.get("comparative_window_takeaway", "") or "").strip()
        xlong_flow_window_floor = str(bnb_focus_payload.get("xlong_flow_window_floor", "") or "").strip()
        xlong_comparative_window_takeaway = str(
            bnb_focus_payload.get("xlong_comparative_window_takeaway", "") or ""
        ).strip()
        focus_brief = str(bnb_focus_payload.get("brief", "") or "").strip()
        next_retest_action = str(bnb_focus_payload.get("next_retest_action", "") or "").strip()
        next_retest_reason = str(bnb_focus_payload.get("next_retest_reason", "") or "").strip()

    if next_focus_symbol:
        if next_focus_action == "deploy_price_state_only":
            focus_execution_blocker_detail = (
                f"{next_focus_symbol} route is promoted, but execution remains {focus_execution_state} until "
                f"{shortline_execution_gate_brief} completes; {shortline_no_trade_rule}."
            )
            focus_execution_done_when = (
                f"{next_focus_symbol} reaches {shortline_setup_ready_state} by completing {shortline_execution_gate_brief}"
            )
        else:
            focus_execution_blocker_detail = (
                f"{next_focus_symbol} remains route-gated via {next_focus_action or 'review'}; keep {focus_execution_state} "
                f"until route quality improves and {shortline_execution_gate_brief} completes; {shortline_no_trade_rule}."
            )
            focus_execution_done_when = (
                f"{next_focus_symbol} route improves and reaches {shortline_setup_ready_state} by completing {shortline_execution_gate_brief}"
            )
    gate_rows = list((shortline_execution_gate_payload or {}).get("symbols") or [])
    gate_match = next(
        (
            row for row in gate_rows
            if isinstance(row, dict) and str(row.get("symbol") or "").strip().upper() == next_focus_symbol
        ),
        None,
    )
    if gate_match:
        focus_execution_state = str(gate_match.get("execution_state") or focus_execution_state).strip() or focus_execution_state
        focus_execution_blocker_detail = (
            str(gate_match.get("blocker_detail") or "").strip() or focus_execution_blocker_detail
        )
        focus_execution_done_when = (
            str(gate_match.get("done_when") or "").strip() or focus_execution_done_when
        )
        gate_micro = gate_match.get("micro_signals")
        if isinstance(gate_micro, dict):
            focus_execution_micro_veto = str(
                gate_micro.get("veto_hint") or focus_execution_micro_veto
            ).strip()
            focus_execution_micro_locality_status = str(
                gate_micro.get("cvd_locality_status") or focus_execution_micro_locality_status
            ).strip()
            if "cvd_drift_risk" in gate_micro:
                focus_execution_micro_drift_risk = "true" if bool(gate_micro.get("cvd_drift_risk")) else "false"
            focus_execution_micro_attack_side = str(
                gate_micro.get("attack_side") or focus_execution_micro_attack_side
            ).strip()
            focus_execution_micro_attack_presence = str(
                gate_micro.get("attack_presence") or focus_execution_micro_attack_presence
            ).strip()
    semantic_rows = list((cvd_semantic_snapshot_payload or {}).get("symbols") or [])
    semantic_match = next(
        (
            row for row in semantic_rows
            if isinstance(row, dict) and str(row.get("symbol") or "").strip().upper() == next_focus_symbol
        ),
        None,
    )
    if semantic_match:
        focus_execution_micro_classification = str(semantic_match.get("classification") or "").strip()
        focus_execution_micro_context = str(semantic_match.get("cvd_context_mode") or "").strip()
        focus_execution_micro_trust_tier = str(semantic_match.get("cvd_trust_tier_hint") or "").strip()
        focus_execution_micro_veto = str(semantic_match.get("cvd_veto_hint") or "").strip()
        focus_execution_micro_locality_status = str(
            semantic_match.get("cvd_locality_status") or focus_execution_micro_locality_status
        ).strip()
        if "cvd_drift_risk" in semantic_match:
            focus_execution_micro_drift_risk = "true" if bool(semantic_match.get("cvd_drift_risk")) else "false"
        focus_execution_micro_attack_side = str(
            semantic_match.get("cvd_attack_side") or focus_execution_micro_attack_side
        ).strip()
        focus_execution_micro_attack_presence = str(
            semantic_match.get("cvd_attack_presence") or focus_execution_micro_attack_presence
        ).strip()
        focus_execution_micro_reasons = [
            str(x).strip()
            for x in semantic_match.get("active_reasons", [])
            if str(x).strip()
        ] if isinstance(semantic_match.get("active_reasons"), list) else []
        micro_parts = [part for part in [
            focus_execution_micro_classification,
            focus_execution_micro_context,
            focus_execution_micro_veto,
        ] if part]
        if focus_execution_micro_reasons:
            micro_parts.append(",".join(focus_execution_micro_reasons))
        if micro_parts:
            focus_execution_blocker_detail = (
                f"{focus_execution_blocker_detail} | micro={':'.join(micro_parts)}"
            )
    if (
        "time_sync_risk" in focus_execution_micro_reasons
        and shortline_cvd_semantic_time_sync_blocker_detail
    ):
        focus_execution_blocker_detail = (
            f"{focus_execution_blocker_detail} | "
            f"time-sync={shortline_cvd_semantic_time_sync_classification or shortline_cvd_semantic_time_sync_status or 'time_sync_risk'}:"
            f"{shortline_cvd_semantic_time_sync_blocker_detail}"
        )
        if shortline_cvd_semantic_time_sync_remediation_hint:
            focus_execution_done_when = (
                f"{focus_execution_done_when} | source: {shortline_cvd_semantic_time_sync_remediation_hint}"
                if focus_execution_done_when and focus_execution_done_when != "-"
                else shortline_cvd_semantic_time_sync_remediation_hint
            )
    if (
        focus_execution_micro_veto == "missing_micro_capture"
        and shortline_cvd_semantic_environment_status == "environment_blocked"
        and shortline_cvd_semantic_environment_blocker_detail
    ):
        focus_execution_blocker_detail = (
            f"{focus_execution_blocker_detail} | env={shortline_cvd_semantic_environment_classification or 'environment_blocked'}:"
            f"{shortline_cvd_semantic_environment_blocker_detail}"
        )
    focus_review_lane = _focus_review_lane(
        symbol=next_focus_symbol,
        action=next_focus_action,
        focus_execution_state=focus_execution_state,
        focus_execution_blocker_detail=focus_execution_blocker_detail,
        focus_execution_done_when=focus_execution_done_when,
        focus_execution_micro_veto=focus_execution_micro_veto,
    )
    focus_review_scores = _focus_review_scores(
        symbol=next_focus_symbol,
        action=next_focus_action,
        focus_execution_state=focus_execution_state,
        focus_execution_micro_classification=focus_execution_micro_classification,
        focus_execution_micro_veto=focus_execution_micro_veto,
        focus_execution_micro_locality_status=focus_execution_micro_locality_status,
        focus_execution_micro_drift_risk=focus_execution_micro_drift_risk,
        focus_execution_micro_attack_side=focus_execution_micro_attack_side,
        focus_execution_micro_attack_presence=focus_execution_micro_attack_presence,
        review_primary_blocker=str(focus_review_lane.get("primary_blocker") or ""),
    )
    focus_review_priority = _focus_review_priority(focus_review_scores)
    review_priority_queue = _crypto_review_priority_queue(
        source_payload=source_payload,
        shortline_execution_gate_payload=shortline_execution_gate_payload,
        cvd_semantic_snapshot_payload=cvd_semantic_snapshot_payload,
    )
    head_source_refresh = _crypto_route_head_source_refresh_lane(
        review_priority_queue=review_priority_queue,
        next_focus_action=next_focus_action,
        current_artifact=current_artifact,
    )
    latest_refresh_audit = _latest_crypto_route_refresh_audit_lane(
        review_dir=review_dir,
        reference_now=reference_now,
    )
    latest_refresh_reuse_gate = _latest_crypto_route_refresh_reuse_gate(latest_refresh_audit)
    downstream_embedding = _crypto_route_head_downstream_embedding_lane(
        review_priority_queue=review_priority_queue,
        next_focus_action=next_focus_action,
        current_artifact=current_artifact,
        review_dir=review_dir,
        reference_now=reference_now,
    )

    brief_lines = [
        f"status: {operator_status or '-'}",
        f"routes: {route_stack_brief or '-'}",
        f"focus: {next_focus_symbol or '-'}",
        f"action: {next_focus_action or '-'}",
        f"reason: {next_focus_reason or '-'}",
        f"shortline-market-state: {shortline_market_state_brief}",
        f"shortline-trigger-stack: {shortline_execution_gate_brief}",
        f"shortline-sessions: {shortline_session_map_brief or '-'}",
        f"focus-execution-state: {focus_execution_state}",
        f"focus-execution-blocker: {focus_execution_blocker_detail}",
        f"focus-execution-done-when: {focus_execution_done_when}",
    ]
    if str(focus_review_lane.get("status") or "").strip() not in {"", "not_active"}:
        brief_lines.append(
            "focus-review: "
            + " | ".join(
                [
                    str(focus_review_lane.get("status") or "-"),
                    str(focus_review_lane.get("primary_blocker") or "-"),
                    str(focus_review_lane.get("micro_blocker") or "-"),
                ]
            )
        )
    if str(focus_review_scores.get("status") or "").strip() == "scored":
        brief_lines.append(
            "focus-review-scores: "
            + " | ".join(
                [
                    f"edge={int(focus_review_scores.get('edge_score') or 0)}",
                    f"structure={int(focus_review_scores.get('structure_score') or 0)}",
                    f"micro={int(focus_review_scores.get('micro_score') or 0)}",
                    f"composite={int(focus_review_scores.get('composite_score') or 0)}",
                ]
            )
        )
    if str(focus_review_priority.get("status") or "").strip() == "ready":
        brief_lines.append(
            "focus-review-priority: "
            + " | ".join(
                [
                    str(focus_review_priority.get("tier") or "-"),
                    f"score={int(focus_review_priority.get('score') or 0)}",
                ]
            )
        )
    if str(review_priority_queue.get("status") or "").strip() == "ready":
        brief_lines.append(f"review-priority-queue: {str(review_priority_queue.get('brief') or '-')}")
    if str(head_source_refresh.get("status") or "").strip() != "not_active":
        brief_lines.append(f"head-source-refresh: {str(head_source_refresh.get('brief') or '-')}")
    if str(latest_refresh_audit.get("status") or "").strip() not in {"", "not_assessed", "not_available"}:
        brief_lines.append(f"latest-refresh-audit: {str(latest_refresh_audit.get('brief') or '-')}")
        brief_lines.append(f"latest-refresh-reuse-gate: {str(latest_refresh_reuse_gate.get('brief') or '-')}")
    if str(downstream_embedding.get("status") or "").strip() not in {"", "not_assessed"}:
        brief_lines.append(f"downstream-embedding: {str(downstream_embedding.get('brief') or '-')}")
    if focus_execution_micro_classification:
        brief_lines.append(f"micro-class: {focus_execution_micro_classification}")
    if focus_execution_micro_context:
        brief_lines.append(f"micro-context: {focus_execution_micro_context}")
    if focus_execution_micro_trust_tier:
        brief_lines.append(f"micro-trust: {focus_execution_micro_trust_tier}")
    if focus_execution_micro_veto:
        brief_lines.append(f"micro-veto: {focus_execution_micro_veto}")
    if focus_execution_micro_locality_status:
        brief_lines.append(f"micro-locality: {focus_execution_micro_locality_status}")
    if focus_execution_micro_drift_risk:
        brief_lines.append(f"micro-drift: {focus_execution_micro_drift_risk}")
    if focus_execution_micro_attack_side or focus_execution_micro_attack_presence:
        brief_lines.append(
            "micro-attack: "
            + ":".join(
                part
                for part in [
                    focus_execution_micro_attack_side,
                    focus_execution_micro_attack_presence,
                ]
                if part
            )
        )
    if focus_execution_micro_reasons:
        brief_lines.append(f"micro-reasons: {', '.join(focus_execution_micro_reasons)}")
    if shortline_cvd_semantic_time_sync_classification or shortline_cvd_semantic_time_sync_status:
        brief_lines.append(
            "time-sync: "
            + ":".join(
                part
                for part in [
                    shortline_cvd_semantic_time_sync_classification or shortline_cvd_semantic_time_sync_status,
                    shortline_cvd_semantic_time_sync_blocker_detail,
                ]
                if part
            )
        )
    if shortline_cvd_queue_handoff_status:
        brief_lines.append(f"cvd-queue-status: {shortline_cvd_queue_handoff_status}")
    if shortline_cvd_queue_focus_batch:
        brief_lines.append(
            f"cvd-queue-focus: {shortline_cvd_queue_focus_batch}:{shortline_cvd_queue_focus_action or '-'}"
        )
    if focus_window_gate:
        brief_lines.append(f"focus-gate: {focus_window_gate}")
    if focus_short_flow_combo_canonical:
        brief_lines.append(f"focus-short-flow: {focus_short_flow_combo_canonical}")
    if focus_long_flow_combo_canonical:
        brief_lines.append(f"focus-long-flow: {focus_long_flow_combo_canonical}")
    if focus_long_top_combo_canonical:
        brief_lines.append(f"focus-long-top: {focus_long_top_combo_canonical}")
    if focus_window_verdict:
        brief_lines.append(f"focus-window: {focus_window_verdict}")
    if focus_window_floor:
        brief_lines.append(f"focus-window-floor: {focus_window_floor}")
    if price_state_window_floor:
        brief_lines.append(f"price-window-floor: {price_state_window_floor}")
    if focus_window_gate_reason:
        brief_lines.append(f"focus-gate-reason: {focus_window_gate_reason}")
    if comparative_window_takeaway:
        brief_lines.append(f"focus-window-note: {comparative_window_takeaway}")
    if xlong_flow_window_floor:
        brief_lines.append(f"xlong-flow-floor: {xlong_flow_window_floor}")
    if xlong_comparative_window_takeaway:
        brief_lines.append(f"xlong-flow-note: {xlong_comparative_window_takeaway}")
    if next_retest_action:
        brief_lines.append(f"next-retest: {next_retest_action}")
    if next_retest_reason:
        brief_lines.append(f"next-retest-reason: {next_retest_reason}")
    if focus_brief:
        brief_lines.append(f"focus-brief: {focus_brief}")
    if overall_takeaway:
        brief_lines.append(f"takeaway: {overall_takeaway}")

    return {
        "operator_status": operator_status,
        "route_stack_brief": route_stack_brief,
        "deploy_now_symbols": deploy_now,
        "watch_priority_symbols": watch_priority,
        "watch_only_symbols": watch_only,
        "review_symbols": review,
        "deploy_count": len(deploy_now),
        "watch_priority_count": len(watch_priority),
        "watch_only_count": len(watch_only),
        "review_count": len(review),
        "next_focus_symbol": next_focus_symbol,
        "next_focus_action": next_focus_action,
        "next_focus_reason": next_focus_reason,
        "shortline_source": str(shortline.get("source") or "builtin_default"),
        "shortline_status": str(shortline.get("status") or "builtin_default"),
        "shortline_market_state_engine": str(shortline.get("market_state_engine") or "bias_only_vs_setup_ready"),
        "shortline_market_state_brief": shortline_market_state_brief,
        "shortline_no_trade_rule": shortline_no_trade_rule,
        "shortline_execution_gate_brief": shortline_execution_gate_brief,
        "shortline_session_map_brief": shortline_session_map_brief,
        "shortline_execution_style": shortline_execution_style,
        "shortline_execution_gate_status": shortline_execution_gate_status,
        "shortline_execution_gate_artifact": shortline_execution_gate_artifact,
        "shortline_cvd_semantic_status": shortline_cvd_semantic_status,
        "shortline_cvd_semantic_takeaway": shortline_cvd_semantic_takeaway,
        "shortline_cvd_semantic_environment_status": shortline_cvd_semantic_environment_status,
        "shortline_cvd_semantic_environment_classification": shortline_cvd_semantic_environment_classification,
        "shortline_cvd_semantic_environment_blocker_detail": shortline_cvd_semantic_environment_blocker_detail,
        "shortline_cvd_semantic_time_sync_status": shortline_cvd_semantic_time_sync_status,
        "shortline_cvd_semantic_time_sync_classification": shortline_cvd_semantic_time_sync_classification,
        "shortline_cvd_semantic_time_sync_intercept_scope": shortline_cvd_semantic_time_sync_intercept_scope,
        "shortline_cvd_semantic_time_sync_blocker_detail": shortline_cvd_semantic_time_sync_blocker_detail,
        "shortline_cvd_semantic_time_sync_remediation_hint": shortline_cvd_semantic_time_sync_remediation_hint,
        "shortline_cvd_semantic_time_sync_fake_ip_sources": shortline_cvd_semantic_time_sync_fake_ip_sources,
        "shortline_cvd_semantic_time_sync_threshold_breach_sources": shortline_cvd_semantic_time_sync_threshold_breach_sources,
        "shortline_cvd_semantic_time_sync_threshold_breach_scope": shortline_cvd_semantic_time_sync_threshold_breach_scope,
        "shortline_cvd_semantic_time_sync_threshold_breach_offset_sources": shortline_cvd_semantic_time_sync_threshold_breach_offset_sources,
        "shortline_cvd_semantic_time_sync_threshold_breach_latency_sources": shortline_cvd_semantic_time_sync_threshold_breach_latency_sources,
        "shortline_cvd_semantic_time_sync_threshold_breach_estimated_offset_ms": shortline_cvd_semantic_time_sync_threshold_breach_estimated_offset_ms,
        "shortline_cvd_semantic_time_sync_threshold_breach_estimated_rtt_ms": shortline_cvd_semantic_time_sync_threshold_breach_estimated_rtt_ms,
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
        "focus_review_status": str(focus_review_lane.get("status") or ""),
        "focus_review_brief": str(focus_review_lane.get("brief") or ""),
        "focus_review_primary_blocker": str(focus_review_lane.get("primary_blocker") or ""),
        "focus_review_micro_blocker": str(focus_review_lane.get("micro_blocker") or ""),
        "focus_review_blocker_detail": str(focus_review_lane.get("blocker_detail") or ""),
        "focus_review_done_when": str(focus_review_lane.get("done_when") or ""),
        "focus_review_score_status": str(focus_review_scores.get("status") or ""),
        "focus_review_edge_score": int(focus_review_scores.get("edge_score") or 0),
        "focus_review_structure_score": int(focus_review_scores.get("structure_score") or 0),
        "focus_review_micro_score": int(focus_review_scores.get("micro_score") or 0),
        "focus_review_composite_score": int(focus_review_scores.get("composite_score") or 0),
        "focus_review_score_brief": str(focus_review_scores.get("brief") or ""),
        "focus_review_priority_status": str(focus_review_priority.get("status") or ""),
        "focus_review_priority_score": int(focus_review_priority.get("score") or 0),
        "focus_review_priority_tier": str(focus_review_priority.get("tier") or ""),
        "focus_review_priority_brief": str(focus_review_priority.get("brief") or ""),
        "review_priority_queue_status": str(review_priority_queue.get("status") or ""),
        "review_priority_queue_count": int(review_priority_queue.get("count") or 0),
        "review_priority_queue": list(review_priority_queue.get("items") or []),
        "review_priority_queue_brief": str(review_priority_queue.get("brief") or ""),
        "review_priority_head_symbol": str(review_priority_queue.get("head_symbol") or ""),
        "review_priority_head_tier": str(review_priority_queue.get("head_tier") or ""),
        "review_priority_head_score": int(review_priority_queue.get("head_score") or 0),
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
        "latest_crypto_route_refresh_reuse_gate_done_when": str(latest_refresh_reuse_gate.get("done_when") or ""),
        "crypto_route_head_downstream_embedding_status": str(downstream_embedding.get("status") or ""),
        "crypto_route_head_downstream_embedding_brief": str(downstream_embedding.get("brief") or ""),
        "crypto_route_head_downstream_embedding_artifact": str(downstream_embedding.get("artifact") or ""),
        "crypto_route_head_downstream_embedding_as_of": str(downstream_embedding.get("as_of") or ""),
        "crypto_route_head_downstream_embedding_blocker_detail": str(downstream_embedding.get("blocker_detail") or ""),
        "crypto_route_head_downstream_embedding_done_when": str(downstream_embedding.get("done_when") or ""),
        "focus_window_gate": focus_window_gate,
        "focus_window_gate_reason": focus_window_gate_reason,
        "focus_short_flow_combo": focus_short_flow_combo,
        "focus_short_flow_combo_canonical": focus_short_flow_combo_canonical,
        "focus_long_flow_combo": focus_long_flow_combo,
        "focus_long_flow_combo_canonical": focus_long_flow_combo_canonical,
        "focus_long_top_combo": focus_long_top_combo,
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
        "overall_takeaway": overall_takeaway,
        "brief_lines": brief_lines,
        "brief_text": "\n".join(brief_lines),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Crypto Route Brief",
        "",
        f"- as_of: `{payload.get('as_of') or ''}`",
        f"- source_artifact: `{payload.get('source_artifact') or ''}`",
        f"- shortline_execution_gate_artifact: `{payload.get('shortline_execution_gate_artifact') or ''}`",
        f"- shortline_cvd_semantic_artifact: `{payload.get('shortline_cvd_semantic_artifact') or ''}`",
        f"- shortline_cvd_queue_handoff_artifact: `{payload.get('shortline_cvd_queue_handoff_artifact') or ''}`",
        f"- operator_status: `{payload.get('operator_status') or ''}`",
        f"- route_stack: `{payload.get('route_stack_brief') or ''}`",
        f"- next_focus_symbol: `{payload.get('next_focus_symbol') or '-'}`",
        f"- next_focus_action: `{payload.get('next_focus_action') or '-'}`",
        f"- next_focus_reason: `{payload.get('next_focus_reason') or '-'}`",
        f"- shortline_execution_gate_status: `{payload.get('shortline_execution_gate_status') or '-'}`",
        f"- shortline_cvd_semantic_status: `{payload.get('shortline_cvd_semantic_status') or '-'}`",
        f"- shortline_cvd_semantic_environment_status: `{payload.get('shortline_cvd_semantic_environment_status') or '-'}`",
        f"- shortline_cvd_semantic_environment_classification: `{payload.get('shortline_cvd_semantic_environment_classification') or '-'}`",
        f"- shortline_cvd_semantic_environment_blocker_detail: `{payload.get('shortline_cvd_semantic_environment_blocker_detail') or '-'}`",
        f"- shortline_cvd_semantic_time_sync_status: `{payload.get('shortline_cvd_semantic_time_sync_status') or '-'}`",
        f"- shortline_cvd_semantic_time_sync_classification: `{payload.get('shortline_cvd_semantic_time_sync_classification') or '-'}`",
        f"- shortline_cvd_semantic_time_sync_intercept_scope: `{payload.get('shortline_cvd_semantic_time_sync_intercept_scope') or '-'}`",
        f"- shortline_cvd_semantic_time_sync_blocker_detail: `{payload.get('shortline_cvd_semantic_time_sync_blocker_detail') or '-'}`",
        f"- shortline_cvd_semantic_time_sync_remediation_hint: `{payload.get('shortline_cvd_semantic_time_sync_remediation_hint') or '-'}`",
        f"- shortline_cvd_queue_handoff_status: `{payload.get('shortline_cvd_queue_handoff_status') or '-'}`",
        f"- shortline_market_state_brief: `{payload.get('shortline_market_state_brief') or '-'}`",
        f"- shortline_execution_gate_brief: `{payload.get('shortline_execution_gate_brief') or '-'}`",
        f"- shortline_session_map_brief: `{payload.get('shortline_session_map_brief') or '-'}`",
        f"- focus_execution_state: `{payload.get('focus_execution_state') or '-'}`",
        f"- focus_execution_blocker_detail: `{payload.get('focus_execution_blocker_detail') or '-'}`",
        f"- focus_execution_done_when: `{payload.get('focus_execution_done_when') or '-'}`",
        f"- focus_execution_micro_classification: `{payload.get('focus_execution_micro_classification') or '-'}`",
        f"- focus_execution_micro_context: `{payload.get('focus_execution_micro_context') or '-'}`",
        f"- focus_execution_micro_trust_tier: `{payload.get('focus_execution_micro_trust_tier') or '-'}`",
        f"- focus_execution_micro_veto: `{payload.get('focus_execution_micro_veto') or '-'}`",
        f"- focus_execution_micro_locality_status: `{payload.get('focus_execution_micro_locality_status') or '-'}`",
        f"- focus_execution_micro_drift_risk: `{payload.get('focus_execution_micro_drift_risk') or '-'}`",
        f"- focus_execution_micro_attack_side: `{payload.get('focus_execution_micro_attack_side') or '-'}`",
        f"- focus_execution_micro_attack_presence: `{payload.get('focus_execution_micro_attack_presence') or '-'}`",
        f"- focus_review_status: `{payload.get('focus_review_status') or '-'}`",
        f"- focus_review_brief: `{payload.get('focus_review_brief') or '-'}`",
        f"- focus_review_primary_blocker: `{payload.get('focus_review_primary_blocker') or '-'}`",
        f"- focus_review_micro_blocker: `{payload.get('focus_review_micro_blocker') or '-'}`",
        f"- focus_review_blocker_detail: `{payload.get('focus_review_blocker_detail') or '-'}`",
        f"- focus_review_done_when: `{payload.get('focus_review_done_when') or '-'}`",
        f"- focus_review_score_status: `{payload.get('focus_review_score_status') or '-'}`",
        f"- focus_review_edge_score: `{payload.get('focus_review_edge_score')}`",
        f"- focus_review_structure_score: `{payload.get('focus_review_structure_score')}`",
        f"- focus_review_micro_score: `{payload.get('focus_review_micro_score')}`",
        f"- focus_review_composite_score: `{payload.get('focus_review_composite_score')}`",
        f"- focus_review_score_brief: `{payload.get('focus_review_score_brief') or '-'}`",
        f"- focus_review_priority_status: `{payload.get('focus_review_priority_status') or '-'}`",
        f"- focus_review_priority_score: `{payload.get('focus_review_priority_score')}`",
        f"- focus_review_priority_tier: `{payload.get('focus_review_priority_tier') or '-'}`",
        f"- focus_review_priority_brief: `{payload.get('focus_review_priority_brief') or '-'}`",
        f"- review_priority_queue_status: `{payload.get('review_priority_queue_status') or '-'}`",
        f"- review_priority_queue_count: `{payload.get('review_priority_queue_count')}`",
        f"- review_priority_queue_brief: `{payload.get('review_priority_queue_brief') or '-'}`",
        f"- review_priority_head_symbol: `{payload.get('review_priority_head_symbol') or '-'}`",
        f"- review_priority_head_tier: `{payload.get('review_priority_head_tier') or '-'}`",
        f"- review_priority_head_score: `{payload.get('review_priority_head_score')}`",
        f"- crypto_route_head_source_refresh_status: `{payload.get('crypto_route_head_source_refresh_status') or '-'}`",
        f"- crypto_route_head_source_refresh_brief: `{payload.get('crypto_route_head_source_refresh_brief') or '-'}`",
        f"- crypto_route_head_source_refresh_symbol: `{payload.get('crypto_route_head_source_refresh_symbol') or '-'}`",
        f"- crypto_route_head_source_refresh_action: `{payload.get('crypto_route_head_source_refresh_action') or '-'}`",
        f"- crypto_route_head_source_refresh_source_kind: `{payload.get('crypto_route_head_source_refresh_source_kind') or '-'}`",
        f"- crypto_route_head_source_refresh_source_health: `{payload.get('crypto_route_head_source_refresh_source_health') or '-'}`",
        f"- crypto_route_head_source_refresh_source_artifact: `{payload.get('crypto_route_head_source_refresh_source_artifact') or '-'}`",
        f"- crypto_route_head_source_refresh_priority_tier: `{payload.get('crypto_route_head_source_refresh_priority_tier') or '-'}`",
        f"- crypto_route_head_source_refresh_route_action: `{payload.get('crypto_route_head_source_refresh_route_action') or '-'}`",
        f"- crypto_route_head_source_refresh_blocker_detail: `{payload.get('crypto_route_head_source_refresh_blocker_detail') or '-'}`",
        f"- crypto_route_head_source_refresh_done_when: `{payload.get('crypto_route_head_source_refresh_done_when') or '-'}`",
        f"- latest_crypto_route_refresh_status: `{payload.get('latest_crypto_route_refresh_status') or '-'}`",
        f"- latest_crypto_route_refresh_brief: `{payload.get('latest_crypto_route_refresh_brief') or '-'}`",
        f"- latest_crypto_route_refresh_artifact: `{payload.get('latest_crypto_route_refresh_artifact') or '-'}`",
        f"- latest_crypto_route_refresh_as_of: `{payload.get('latest_crypto_route_refresh_as_of') or '-'}`",
        f"- latest_crypto_route_refresh_native_mode: `{payload.get('latest_crypto_route_refresh_native_mode') or '-'}`",
        f"- latest_crypto_route_refresh_native_step_count: `{payload.get('latest_crypto_route_refresh_native_step_count')}`",
        f"- latest_crypto_route_refresh_reused_native_count: `{payload.get('latest_crypto_route_refresh_reused_native_count')}`",
        f"- latest_crypto_route_refresh_missing_reused_count: `{payload.get('latest_crypto_route_refresh_missing_reused_count')}`",
        f"- latest_crypto_route_refresh_note: `{payload.get('latest_crypto_route_refresh_note') or '-'}`",
        f"- latest_crypto_route_refresh_done_when: `{payload.get('latest_crypto_route_refresh_done_when') or '-'}`",
        f"- latest_crypto_route_refresh_reuse_level: `{payload.get('latest_crypto_route_refresh_reuse_level') or '-'}`",
        f"- latest_crypto_route_refresh_reuse_gate_status: `{payload.get('latest_crypto_route_refresh_reuse_gate_status') or '-'}`",
        f"- latest_crypto_route_refresh_reuse_gate_brief: `{payload.get('latest_crypto_route_refresh_reuse_gate_brief') or '-'}`",
        f"- latest_crypto_route_refresh_reuse_gate_blocking: `{payload.get('latest_crypto_route_refresh_reuse_gate_blocking')}`",
        f"- latest_crypto_route_refresh_reuse_gate_blocker_detail: `{payload.get('latest_crypto_route_refresh_reuse_gate_blocker_detail') or '-'}`",
        f"- latest_crypto_route_refresh_reuse_gate_done_when: `{payload.get('latest_crypto_route_refresh_reuse_gate_done_when') or '-'}`",
        f"- crypto_route_head_downstream_embedding_status: `{payload.get('crypto_route_head_downstream_embedding_status') or '-'}`",
        f"- crypto_route_head_downstream_embedding_brief: `{payload.get('crypto_route_head_downstream_embedding_brief') or '-'}`",
        f"- crypto_route_head_downstream_embedding_artifact: `{payload.get('crypto_route_head_downstream_embedding_artifact') or '-'}`",
        f"- crypto_route_head_downstream_embedding_as_of: `{payload.get('crypto_route_head_downstream_embedding_as_of') or '-'}`",
        f"- crypto_route_head_downstream_embedding_blocker_detail: `{payload.get('crypto_route_head_downstream_embedding_blocker_detail') or '-'}`",
        f"- crypto_route_head_downstream_embedding_done_when: `{payload.get('crypto_route_head_downstream_embedding_done_when') or '-'}`",
        f"- shortline_cvd_queue_focus_batch: `{payload.get('shortline_cvd_queue_focus_batch') or '-'}`",
        f"- shortline_cvd_queue_focus_action: `{payload.get('shortline_cvd_queue_focus_action') or '-'}`",
        f"- focus_window_gate: `{payload.get('focus_window_gate') or '-'}`",
        f"- focus_short_flow_combo_canonical: `{payload.get('focus_short_flow_combo_canonical') or '-'}`",
        f"- focus_long_flow_combo_canonical: `{payload.get('focus_long_flow_combo_canonical') or '-'}`",
        f"- focus_long_top_combo_canonical: `{payload.get('focus_long_top_combo_canonical') or '-'}`",
        f"- focus_window_verdict: `{payload.get('focus_window_verdict') or '-'}`",
        f"- focus_window_floor: `{payload.get('focus_window_floor') or '-'}`",
        f"- price_state_window_floor: `{payload.get('price_state_window_floor') or '-'}`",
        f"- comparative_window_takeaway: `{payload.get('comparative_window_takeaway') or '-'}`",
        f"- xlong_flow_window_floor: `{payload.get('xlong_flow_window_floor') or '-'}`",
        f"- xlong_comparative_window_takeaway: `{payload.get('xlong_comparative_window_takeaway') or '-'}`",
        f"- next_retest_action: `{payload.get('next_retest_action') or '-'}`",
        f"- next_retest_reason: `{payload.get('next_retest_reason') or '-'}`",
        "",
        "## Buckets",
        f"- deploy_now: `{', '.join(payload.get('deploy_now_symbols', [])) or '-'}`",
        f"- watch_priority: `{', '.join(payload.get('watch_priority_symbols', [])) or '-'}`",
        f"- watch_only: `{', '.join(payload.get('watch_only_symbols', [])) or '-'}`",
        f"- review: `{', '.join(payload.get('review_symbols', [])) or '-'}`",
        "",
        "## Brief",
    ]
    for line in payload.get("brief_lines", []):
        lines.append(f"- {line}")
    return "\n".join(lines).strip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a compact crypto route brief from the latest symbol route handoff.")
    parser.add_argument("--review-dir", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--now", default="")
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--artifact-keep", type=int, default=12)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    review_dir = Path(args.review_dir).expanduser().resolve()
    config_path = resolve_path(args.config, anchor=Path(__file__).resolve().parents[1])
    review_dir.mkdir(parents=True, exist_ok=True)
    runtime_now = parse_now(args.now)

    source_path = latest_symbol_route_handoff(review_dir, runtime_now)
    source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    bnb_focus_path = latest_bnb_flow_focus(review_dir, runtime_now)
    bnb_focus_payload = json.loads(bnb_focus_path.read_text(encoding="utf-8")) if bnb_focus_path else None
    shortline_execution_gate_path = latest_shortline_execution_gate(review_dir, runtime_now)
    shortline_execution_gate_payload = (
        json.loads(shortline_execution_gate_path.read_text(encoding="utf-8"))
        if shortline_execution_gate_path
        else None
    )
    cvd_semantic_path = latest_cvd_semantic_snapshot(review_dir, runtime_now)
    cvd_semantic_payload = (
        json.loads(cvd_semantic_path.read_text(encoding="utf-8"))
        if cvd_semantic_path
        else None
    )
    cvd_queue_handoff_path = latest_cvd_queue_handoff(review_dir, runtime_now)
    cvd_queue_handoff_payload = (
        json.loads(cvd_queue_handoff_path.read_text(encoding="utf-8"))
        if cvd_queue_handoff_path
        else None
    )
    shortline_policy = load_shortline_policy(config_path)
    stamp = runtime_now.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = review_dir / f"{stamp}_crypto_route_brief.json"
    md_path = review_dir / f"{stamp}_crypto_route_brief.md"
    checksum_path = review_dir / f"{stamp}_crypto_route_brief_checksum.json"
    brief = build_brief(
        source_payload,
        bnb_focus_payload,
        shortline_policy,
        shortline_execution_gate_payload,
        cvd_semantic_payload,
        cvd_queue_handoff_payload,
        current_artifact=str(json_path),
        review_dir=review_dir,
        reference_now=runtime_now,
    )

    payload = {
        "ok": True,
        "status": "ok",
        "as_of": fmt_utc(runtime_now),
        "source_artifact": str(source_path),
        "config_path": str(config_path),
        "bnb_flow_focus_artifact": str(bnb_focus_path) if bnb_focus_path else None,
        "shortline_execution_gate_artifact": str(shortline_execution_gate_path) if shortline_execution_gate_path else None,
        "shortline_cvd_semantic_artifact": str(cvd_semantic_path) if cvd_semantic_path else None,
        "shortline_cvd_queue_handoff_artifact": str(cvd_queue_handoff_path) if cvd_queue_handoff_path else None,
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
        stem="crypto_route_brief",
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
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
