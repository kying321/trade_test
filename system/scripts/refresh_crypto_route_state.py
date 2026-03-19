#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = SYSTEM_ROOT / "output"
DEFAULT_REVIEW_DIR = DEFAULT_OUTPUT_ROOT / "review"
TIMESTAMP_RE = re.compile(r"^(?P<stamp>\d{8}T\d{6}Z)_")
FUTURE_STAMP_GRACE_MINUTES = 5
HARD_FAILURE_STEP_STATUSES = frozenset({"partial_failure", "failed", "timed_out", "error"})
MISSING_SOURCE_STEP_STATUSES = frozenset({"missing_reused_source"})

DEFAULT_NATIVE_GROUP_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "native_custom",
        "symbol_group": "custom",
        "symbols": "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT",
        "lookback_bars": 300,
        "sample_windows": 3,
        "window_bars": 40,
    },
    {
        "name": "native_majors",
        "symbol_group": "majors",
        "symbols": "BTCUSDT,ETHUSDT",
        "lookback_bars": 300,
        "sample_windows": 3,
        "window_bars": 40,
    },
    {
        "name": "native_beta",
        "symbol_group": "beta",
        "symbols": "SOLUSDT,BNBUSDT",
        "lookback_bars": 300,
        "sample_windows": 3,
        "window_bars": 40,
    },
    {
        "name": "native_sol",
        "symbol_group": "sol",
        "symbols": "SOLUSDT",
        "lookback_bars": 300,
        "sample_windows": 3,
        "window_bars": 40,
    },
    {
        "name": "native_bnb",
        "symbol_group": "bnb",
        "symbols": "BNBUSDT",
        "lookback_bars": 300,
        "sample_windows": 3,
        "window_bars": 40,
    },
    {
        "name": "native_majors_long",
        "symbol_group": "majors_long",
        "symbols": "BTCUSDT,ETHUSDT",
        "lookback_bars": 720,
        "sample_windows": 6,
        "window_bars": 40,
    },
    {
        "name": "native_beta_long",
        "symbol_group": "beta_long",
        "symbols": "SOLUSDT,BNBUSDT",
        "lookback_bars": 720,
        "sample_windows": 6,
        "window_bars": 40,
    },
    {
        "name": "native_sol_long",
        "symbol_group": "sol_long",
        "symbols": "SOLUSDT",
        "lookback_bars": 720,
        "sample_windows": 6,
        "window_bars": 40,
    },
    {
        "name": "native_bnb_long",
        "symbol_group": "bnb_long",
        "symbols": "BNBUSDT",
        "lookback_bars": 720,
        "sample_windows": 4,
        "window_bars": 80,
    },
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


def fmt_utc(value: dt.datetime | None) -> str:
    effective = value or now_utc()
    return effective.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def step_now(base: dt.datetime, offset_seconds: int) -> dt.datetime:
    return base + dt.timedelta(seconds=int(offset_seconds))


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


def latest_stamped_datetime(review_dir: Path, suffixes: list[str]) -> dt.datetime | None:
    latest_dt: dt.datetime | None = None
    for suffix in suffixes:
        for path in review_dir.glob(f"*_{suffix}.json"):
            stamp_dt = parsed_artifact_stamp(path)
            if stamp_dt is None:
                continue
            if latest_dt is None or stamp_dt > latest_dt:
                latest_dt = stamp_dt
    return latest_dt


def derive_runtime_now(review_dir: Path, requested_now: str | None) -> dt.datetime:
    if str(requested_now or "").strip():
        return parse_now(requested_now)
    runtime_now = now_utc()
    latest_dt = latest_stamped_datetime(
        review_dir,
        [
            "crypto_route_brief",
            "crypto_route_operator_brief",
            "binance_indicator_symbol_route_handoff",
            "binance_indicator_combo_playbook",
            "binance_indicator_source_control_report",
            "crypto_route_refresh",
        ],
    )
    future_cutoff = runtime_now + dt.timedelta(minutes=FUTURE_STAMP_GRACE_MINUTES)
    if latest_dt is not None and runtime_now <= latest_dt <= future_cutoff:
        return latest_dt + dt.timedelta(seconds=1)
    return runtime_now


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def run_json_step(*, step_name: str, cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{step_name}_failed: {(proc.stderr or proc.stdout or '').strip() or f'returncode={proc.returncode}'}"
        )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{step_name}_invalid_json: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{step_name}_invalid_payload")
    return payload


def script_path(name: str) -> Path:
    return SYSTEM_ROOT / "scripts" / name


def payload_as_of(payload: dict[str, Any]) -> dt.datetime | None:
    raw = str(payload.get("as_of") or "").strip()
    if not raw:
        return None
    try:
        return parse_now(raw)
    except ValueError:
        return None


def payload_or_artifact_as_of(payload: dict[str, Any]) -> str:
    effective = payload_as_of(payload)
    if effective is not None:
        return fmt_utc(effective)
    artifact_text = str(payload.get("artifact") or "").strip()
    if artifact_text:
        stamp_dt = parsed_artifact_stamp(Path(artifact_text))
        if stamp_dt is not None:
            return fmt_utc(stamp_dt)
    return ""


def latest_review_json_artifact(review_dir: Path, suffix: str, reference_now: dt.datetime) -> Path | None:
    candidates = sorted(
        review_dir.glob(f"*_{suffix}.json"),
        key=lambda item: artifact_sort_key(item, reference_now),
        reverse=True,
    )
    for path in candidates:
        stamp_dt = parsed_artifact_stamp(path)
        if stamp_dt is None:
            continue
        if stamp_dt <= reference_now + dt.timedelta(minutes=FUTURE_STAMP_GRACE_MINUTES):
            return path
    return None


def latest_native_group_artifact(review_dir: Path, symbol_group: str, reference_now: dt.datetime) -> Path | None:
    group_slug = str(symbol_group or "").strip().lower()
    if not group_slug:
        return None
    return latest_review_json_artifact(review_dir, f"{group_slug}_binance_indicator_combo_native_crypto", reference_now)


def latest_micro_capture_artifact(artifact_dir: Path, reference_now: dt.datetime) -> Path | None:
    candidates = sorted(
        artifact_dir.glob("*_micro_capture.json"),
        key=lambda item: artifact_sort_key(item, reference_now),
        reverse=True,
    )
    for path in candidates:
        stamp_dt = parsed_artifact_stamp(path)
        if stamp_dt is None:
            continue
        if stamp_dt <= reference_now + dt.timedelta(minutes=FUTURE_STAMP_GRACE_MINUTES):
            return path
    return None


def normalize_output_root(raw_output_root: Path, review_dir: Path) -> Path:
    candidate = raw_output_root.expanduser().resolve()
    if candidate.name == "output":
        return candidate

    review_output_root = review_dir.parent if review_dir.name == "review" else None
    if review_output_root is not None and review_output_root.name == "output":
        if review_output_root == candidate:
            return review_output_root
        if review_output_root.parent == candidate:
            return review_output_root

    nested_output = candidate / "output"
    if nested_output.exists() and nested_output.is_dir():
        return nested_output.resolve()

    return candidate


def load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def reuse_previous_artifact_payload(
    *,
    review_dir: Path,
    suffix: str,
    reference_now: dt.datetime,
    control_note: str,
) -> dict[str, Any]:
    artifact_path = latest_review_json_artifact(review_dir, suffix, reference_now)
    if artifact_path is None:
        raise FileNotFoundError(f"no_{suffix}_artifact")
    payload = load_json_mapping(artifact_path)
    payload["artifact"] = str(artifact_path)
    payload["status"] = "carry_over_previous_artifact"
    payload["control_note"] = control_note
    if not str(payload.get("as_of") or "").strip():
        stamp_dt = parsed_artifact_stamp(artifact_path)
        if stamp_dt is not None:
            payload["as_of"] = fmt_utc(stamp_dt)
    return payload


def crypto_route_head_source_refresh_lane(
    *,
    brief_payload: dict[str, Any],
    operator_brief_payload: dict[str, Any],
    brief_artifact: str,
    operator_brief_artifact: str,
) -> dict[str, Any]:
    queue = operator_brief_payload.get("review_priority_queue") or brief_payload.get("review_priority_queue") or []
    head_symbol = str(
        operator_brief_payload.get("review_priority_head_symbol")
        or brief_payload.get("review_priority_head_symbol")
        or ""
    ).strip().upper()
    head_tier = str(
        operator_brief_payload.get("review_priority_head_tier")
        or brief_payload.get("review_priority_head_tier")
        or ""
    ).strip()

    head_item: dict[str, Any] = {}
    if isinstance(queue, list):
        for row in queue:
            if not isinstance(row, dict):
                continue
            row_symbol = str(row.get("symbol") or "").strip().upper()
            if head_symbol and row_symbol == head_symbol:
                head_item = dict(row)
                break
            if not head_item and int(row.get("rank") or 0) == 1:
                head_item = dict(row)
        if not head_symbol and head_item:
            head_symbol = str(head_item.get("symbol") or "").strip().upper()

    source_artifact = operator_brief_artifact or brief_artifact
    source_kind = "crypto_route"
    source_health = "ready" if source_artifact else "missing"
    action = "read_current_artifact" if source_artifact and head_symbol else "-"
    status = "ready" if action == "read_current_artifact" else "not_active"
    tier_text = head_tier or str(head_item.get("priority_tier") or "").strip() or "-"
    route_action = (
        str(head_item.get("route_action") or "").strip()
        or str(operator_brief_payload.get("next_focus_action") or "").strip()
        or str(brief_payload.get("next_focus_action") or "").strip()
        or "-"
    )
    blocker_detail = "-"
    done_when = "-"
    if head_symbol and action == "read_current_artifact":
        blocker_detail = (
            f"{head_symbol} currently uses the freshly refreshed crypto_route artifact "
            f"while priority_tier={tier_text} and route_action={route_action}."
        )
        done_when = (
            f"keep {head_symbol} on the current crypto_route artifact until the queue head changes "
            "or a newer route refresh is required"
        )

    return {
        "status": status,
        "brief": f"{status}:{head_symbol or '-'}:{action}",
        "symbol": head_symbol,
        "action": action,
        "source_kind": source_kind,
        "source_health": source_health,
        "source_artifact": source_artifact,
        "priority_tier": tier_text,
        "route_action": route_action,
        "blocker_detail": blocker_detail,
        "done_when": done_when,
    }


def crypto_route_head_downstream_embedding_lane(
    *,
    review_dir: Path,
    reference_now: dt.datetime,
    head_source_refresh: dict[str, Any],
    route_as_of: dt.datetime | None = None,
) -> dict[str, Any]:
    head = dict(head_source_refresh or {})
    symbol = str(head.get("symbol") or "").strip().upper()
    head_status = str(head.get("status") or "").strip()
    head_action = str(head.get("action") or "").strip()
    route_artifact = str(head.get("source_artifact") or "").strip()
    route_artifact_path = Path(route_artifact) if route_artifact else None
    effective_route_as_of = route_as_of or (parsed_artifact_stamp(route_artifact_path) if route_artifact_path else None)

    if not symbol:
        return {
            "status": "not_applicable",
            "brief": "not_applicable:-",
            "artifact": "",
            "as_of": "",
            "blocker_detail": "crypto route downstream embedding lane is only active when a route review head is present.",
            "done_when": "crypto route head returns before reassessing downstream embedding freshness",
        }

    hot_research_path = latest_review_json_artifact(review_dir, "hot_universe_research", reference_now)
    hot_research_as_of = parsed_artifact_stamp(hot_research_path) if hot_research_path else None

    if head_status in {"ready", "deferred_until_next_eligible_end_date"} and head_action in {
        "read_current_artifact",
        "wait_for_next_eligible_end_date",
    }:
        if hot_research_path and effective_route_as_of and hot_research_as_of and hot_research_as_of < effective_route_as_of:
            return {
                "status": "carry_over_non_blocking",
                "brief": f"carry_over_non_blocking:{symbol}",
                "artifact": str(hot_research_path),
                "as_of": fmt_utc(hot_research_as_of),
                "blocker_detail": (
                    f"{symbol} current crypto route head is already {head_status} via {head_action}, "
                    f"but latest downstream hot_universe_research artifact ({hot_research_path.name}) is older "
                    "and remains broader carry-over outside route-refresh scope."
                ),
                "done_when": (
                    "rerun hot_universe_research and downstream hot-universe handoff only when broader embedding "
                    "freshness is required"
                ),
            }
        return {
            "status": "current_non_blocking",
            "brief": f"current_non_blocking:{symbol}",
            "artifact": str(hot_research_path or ""),
            "as_of": fmt_utc(hot_research_as_of) if hot_research_as_of else "",
            "blocker_detail": (
                f"{symbol} current crypto route head is already {head_status} via {head_action}, "
                "and downstream hot_universe_research is current enough for route-refresh scope."
            ),
            "done_when": "keep downstream embedding current enough for the broader hot-universe handoff",
        }

    return {
        "status": "blocked_by_head_source_refresh",
        "brief": f"blocked_by_head_source_refresh:{symbol}",
        "artifact": str(hot_research_path or ""),
        "as_of": fmt_utc(hot_research_as_of) if hot_research_as_of else "",
        "blocker_detail": (
            f"{symbol} route head is not yet readable via current source refresh lane ({head_status or '-'}:{head_action or '-'})"
        ),
        "done_when": "stabilize the current crypto route head source refresh lane before reassessing downstream embedding freshness",
    }


def crypto_route_refresh_reuse_gate(steps: list[dict[str, Any]], native_refresh_mode: str) -> dict[str, Any]:
    native_steps = [dict(row) for row in steps if str(row.get("name") or "").startswith("native_")]
    native_step_count = len(native_steps)
    failed_native_rows = [
        dict(row)
        for row in native_steps
        if str(row.get("status") or "").strip() in HARD_FAILURE_STEP_STATUSES
    ]
    failed_native_count = len(failed_native_rows)
    reused_native_count = sum(
        1 for row in native_steps if str(row.get("status") or "").strip() == "reused_previous_artifact"
    )
    missing_reused_count = sum(
        1 for row in native_steps if str(row.get("status") or "").strip() in MISSING_SOURCE_STEP_STATUSES
    )
    mode_text = str(native_refresh_mode or "").strip() or "unknown"
    if native_step_count <= 0:
        return {
            "level": "blocking",
            "status": "audit_missing_requires_full_native_refresh",
            "brief": f"audit_missing_requires_full_native_refresh:{mode_text}:0/0",
            "blocking": True,
            "blocker_detail": "crypto_route_refresh did not record any native_* steps, so reuse safety cannot be trusted.",
            "done_when": "rerun refresh_crypto_route_state and record native_* steps before relying on reuse safety",
        }
    if failed_native_count > 0:
        failed_step_names = ",".join(str(row.get("name") or "").strip() for row in failed_native_rows[:4])
        if failed_native_count > 4:
            failed_step_names += ",+more"
        return {
            "level": "blocking",
            "status": "native_refresh_partial_failure",
            "brief": f"native_refresh_partial_failure:{mode_text}:{failed_native_count}/{native_step_count}",
            "blocking": True,
            "blocker_detail": (
                "one or more native_* steps returned partial_failure/failed/timed_out/error "
                f"({failed_native_count}/{native_step_count}; steps={failed_step_names or '-'})"
            ),
            "done_when": "rerun full native refresh until every native_* step completes with status=ok",
        }
    if reused_native_count == native_step_count:
        return {
            "level": "informational",
            "status": "reuse_non_blocking",
            "brief": f"reuse_non_blocking:{mode_text}:{reused_native_count}/{native_step_count}",
            "blocking": False,
            "blocker_detail": "all tracked native steps were intentionally reused; current refresh remains safe for downstream consumption.",
            "done_when": "run full native refresh only when fresh native recomputation is explicitly required",
        }
    if reused_native_count == 0 and missing_reused_count == 0:
        return {
            "level": "informational",
            "status": "fresh_non_blocking",
            "brief": f"fresh_non_blocking:{mode_text}:{reused_native_count}/{native_step_count}",
            "blocking": False,
            "blocker_detail": "all tracked native steps were freshly recomputed in this refresh run.",
            "done_when": "keep using the current refresh until the next explicit recomputation window",
        }
    return {
        "level": "blocking",
        "status": "mixed_requires_full_native_refresh",
        "brief": f"mixed_requires_full_native_refresh:{mode_text}:{reused_native_count}/{native_step_count}",
        "blocking": True,
        "blocker_detail": "current refresh mixes reused, missing, or partially refreshed native inputs; force a clean full native refresh before trusting downstream reuse.",
        "done_when": "rerun refresh_crypto_route_state without skip_native_refresh and confirm native_* steps are uniformly fresh or intentionally reused end-to-end",
    }


def crypto_route_refresh_aggregate_status(steps: list[dict[str, Any]]) -> dict[str, Any]:
    failed_rows = [
        dict(row)
        for row in steps
        if str(row.get("status") or "").strip() in HARD_FAILURE_STEP_STATUSES.union(MISSING_SOURCE_STEP_STATUSES)
    ]
    if not failed_rows:
        return {
            "ok": True,
            "status": "ok",
            "failure_step_count": 0,
            "failure_step_brief": "",
        }
    failure_tokens = [
        f"{str(row.get('name') or '').strip()}:{str(row.get('status') or '').strip()}"
        for row in failed_rows[:6]
    ]
    if len(failed_rows) > 6:
        failure_tokens.append("+more")
    return {
        "ok": False,
        "status": "partial_failure",
        "failure_step_count": len(failed_rows),
        "failure_step_brief": " | ".join(token for token in failure_tokens if token),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Crypto Route Refresh",
        "",
        f"- status: `{payload.get('status') or ''}`",
        f"- failure_step_count: `{payload.get('failure_step_count')}`",
        f"- failure_step_brief: `{payload.get('failure_step_brief') or ''}`",
        f"- as_of: `{payload.get('as_of') or ''}`",
        f"- native_refresh_mode: `{payload.get('native_refresh_mode') or ''}`",
        f"- cvd_semantic_snapshot_artifact: `{payload.get('cvd_semantic_snapshot_artifact') or ''}`",
        f"- cvd_queue_handoff_artifact: `{payload.get('cvd_queue_handoff_artifact') or ''}`",
        f"- shortline_execution_gate_artifact: `{payload.get('shortline_execution_gate_artifact') or ''}`",
        f"- shortline_pattern_router_artifact: `{payload.get('shortline_pattern_router_artifact') or ''}`",
        f"- shortline_profile_location_watch_artifact: `{payload.get('shortline_profile_location_watch_artifact') or ''}`",
        f"- shortline_mss_watch_artifact: `{payload.get('shortline_mss_watch_artifact') or ''}`",
        f"- shortline_cvd_confirmation_watch_artifact: `{payload.get('shortline_cvd_confirmation_watch_artifact') or ''}`",
        f"- shortline_retest_watch_artifact: `{payload.get('shortline_retest_watch_artifact') or ''}`",
        f"- shortline_live_bars_snapshot_artifact: `{payload.get('shortline_live_bars_snapshot_artifact') or ''}`",
        f"- shortline_material_change_trigger_artifact: `{payload.get('shortline_material_change_trigger_artifact') or ''}`",
        f"- shortline_liquidity_event_trigger_artifact: `{payload.get('shortline_liquidity_event_trigger_artifact') or ''}`",
        f"- shortline_live_orderflow_snapshot_artifact: `{payload.get('shortline_live_orderflow_snapshot_artifact') or ''}`",
        f"- shortline_execution_quality_watch_artifact: `{payload.get('shortline_execution_quality_watch_artifact') or ''}`",
        f"- shortline_slippage_snapshot_artifact: `{payload.get('shortline_slippage_snapshot_artifact') or ''}`",
        f"- shortline_fill_capacity_watch_artifact: `{payload.get('shortline_fill_capacity_watch_artifact') or ''}`",
        f"- shortline_sizing_watch_artifact: `{payload.get('shortline_sizing_watch_artifact') or ''}`",
        f"- shortline_signal_quality_watch_artifact: `{payload.get('shortline_signal_quality_watch_artifact') or ''}`",
        f"- signal_to_order_tickets_artifact: `{payload.get('signal_to_order_tickets_artifact') or ''}`",
        f"- signal_to_order_tickets_signal_source_kind: `{payload.get('signal_to_order_tickets_signal_source_kind') or ''}`",
        f"- signal_to_order_tickets_symbol_scope_mode: `{payload.get('signal_to_order_tickets_symbol_scope_mode') or ''}`",
        f"- signal_to_order_tickets_proxy_price_only_count: `{payload.get('signal_to_order_tickets_proxy_price_only_count')}`",
        f"- crypto_route_brief_artifact: `{payload.get('crypto_route_brief_artifact') or ''}`",
        f"- crypto_route_operator_brief_artifact: `{payload.get('crypto_route_operator_brief_artifact') or ''}`",
        f"- source_control_artifact: `{payload.get('source_control_artifact') or ''}`",
        f"- route_handoff_artifact: `{payload.get('route_handoff_artifact') or ''}`",
        f"- system_time_sync_environment_report_artifact: `{payload.get('system_time_sync_environment_report_artifact') or ''}`",
        f"- system_time_sync_environment_report_classification: `{payload.get('system_time_sync_environment_report_classification') or ''}`",
        f"- system_time_sync_environment_report_blocker_detail: `{payload.get('system_time_sync_environment_report_blocker_detail') or ''}`",
        f"- crypto_route_refresh_reuse_level: `{payload.get('crypto_route_refresh_reuse_level') or ''}`",
        f"- crypto_route_refresh_reuse_gate_status: `{payload.get('crypto_route_refresh_reuse_gate_status') or ''}`",
        f"- crypto_route_refresh_reuse_gate_brief: `{payload.get('crypto_route_refresh_reuse_gate_brief') or ''}`",
        f"- focus_review_status: `{payload.get('focus_review_status') or ''}`",
        f"- focus_review_primary_blocker: `{payload.get('focus_review_primary_blocker') or ''}`",
        f"- focus_review_micro_blocker: `{payload.get('focus_review_micro_blocker') or ''}`",
        f"- focus_review_done_when: `{payload.get('focus_review_done_when') or ''}`",
        f"- shortline_material_change_trigger_status: `{payload.get('shortline_material_change_trigger_status') or ''}`",
        f"- shortline_material_change_trigger_brief: `{payload.get('shortline_material_change_trigger_brief') or ''}`",
        f"- shortline_material_change_trigger_decision: `{payload.get('shortline_material_change_trigger_decision') or ''}`",
        f"- shortline_material_change_trigger_rerun_recommended: `{payload.get('shortline_material_change_trigger_rerun_recommended')}`",
        f"- shortline_liquidity_event_trigger_status: `{payload.get('shortline_liquidity_event_trigger_status') or ''}`",
        f"- shortline_liquidity_event_trigger_brief: `{payload.get('shortline_liquidity_event_trigger_brief') or ''}`",
        f"- shortline_liquidity_event_trigger_decision: `{payload.get('shortline_liquidity_event_trigger_decision') or ''}`",
        f"- shortline_liquidity_event_trigger_current_signal_source: `{payload.get('shortline_liquidity_event_trigger_current_signal_source') or ''}`",
        f"- shortline_liquidity_event_trigger_current_signal_source_artifact: `{payload.get('shortline_liquidity_event_trigger_current_signal_source_artifact') or ''}`",
        f"- shortline_liquidity_event_trigger_previous_signal_source: `{payload.get('shortline_liquidity_event_trigger_previous_signal_source') or ''}`",
        f"- shortline_liquidity_event_trigger_previous_signal_source_artifact: `{payload.get('shortline_liquidity_event_trigger_previous_signal_source_artifact') or ''}`",
        f"- shortline_live_bars_snapshot_status: `{payload.get('shortline_live_bars_snapshot_status') or ''}`",
        f"- shortline_live_bars_snapshot_brief: `{payload.get('shortline_live_bars_snapshot_brief') or ''}`",
        f"- shortline_live_bars_snapshot_decision: `{payload.get('shortline_live_bars_snapshot_decision') or ''}`",
        f"- shortline_profile_location_watch_status: `{payload.get('shortline_profile_location_watch_status') or ''}`",
        f"- shortline_profile_location_watch_brief: `{payload.get('shortline_profile_location_watch_brief') or ''}`",
        f"- shortline_profile_location_watch_decision: `{payload.get('shortline_profile_location_watch_decision') or ''}`",
        f"- shortline_profile_location_watch_rotation_proximity_state: `{payload.get('shortline_profile_location_watch_rotation_proximity_state') or ''}`",
        f"- shortline_profile_location_watch_profile_rotation_alignment_band: `{payload.get('shortline_profile_location_watch_profile_rotation_alignment_band') or ''}`",
        f"- shortline_profile_location_watch_profile_rotation_next_milestone: `{payload.get('shortline_profile_location_watch_profile_rotation_next_milestone') or ''}`",
        f"- shortline_profile_location_watch_profile_rotation_confidence: `{payload.get('shortline_profile_location_watch_profile_rotation_confidence')}`",
        f"- shortline_profile_location_watch_active_rotation_targets: `{','.join(payload.get('shortline_profile_location_watch_active_rotation_targets') or [])}`",
        f"- shortline_profile_location_watch_profile_rotation_target_tag: `{payload.get('shortline_profile_location_watch_profile_rotation_target_tag') or ''}`",
        f"- shortline_profile_location_watch_profile_rotation_target_bin_distance: `{payload.get('shortline_profile_location_watch_profile_rotation_target_bin_distance')}`",
        f"- shortline_profile_location_watch_profile_rotation_target_distance_bps: `{payload.get('shortline_profile_location_watch_profile_rotation_target_distance_bps')}`",
        f"- shortline_pattern_router_status: `{payload.get('shortline_pattern_router_status') or ''}`",
        f"- shortline_pattern_router_brief: `{payload.get('shortline_pattern_router_brief') or ''}`",
        f"- shortline_pattern_router_decision: `{payload.get('shortline_pattern_router_decision') or ''}`",
        f"- shortline_pattern_router_family: `{payload.get('shortline_pattern_router_family') or ''}`",
        f"- shortline_pattern_router_stage: `{payload.get('shortline_pattern_router_stage') or ''}`",
        f"- shortline_mss_watch_status: `{payload.get('shortline_mss_watch_status') or ''}`",
        f"- shortline_mss_watch_brief: `{payload.get('shortline_mss_watch_brief') or ''}`",
        f"- shortline_mss_watch_decision: `{payload.get('shortline_mss_watch_decision') or ''}`",
        f"- shortline_cvd_confirmation_watch_status: `{payload.get('shortline_cvd_confirmation_watch_status') or ''}`",
        f"- shortline_cvd_confirmation_watch_brief: `{payload.get('shortline_cvd_confirmation_watch_brief') or ''}`",
        f"- shortline_cvd_confirmation_watch_decision: `{payload.get('shortline_cvd_confirmation_watch_decision') or ''}`",
        f"- shortline_retest_watch_status: `{payload.get('shortline_retest_watch_status') or ''}`",
        f"- shortline_retest_watch_brief: `{payload.get('shortline_retest_watch_brief') or ''}`",
        f"- shortline_retest_watch_decision: `{payload.get('shortline_retest_watch_decision') or ''}`",
        f"- shortline_live_orderflow_snapshot_status: `{payload.get('shortline_live_orderflow_snapshot_status') or ''}`",
        f"- shortline_live_orderflow_snapshot_brief: `{payload.get('shortline_live_orderflow_snapshot_brief') or ''}`",
        f"- shortline_live_orderflow_snapshot_decision: `{payload.get('shortline_live_orderflow_snapshot_decision') or ''}`",
        f"- shortline_execution_quality_watch_status: `{payload.get('shortline_execution_quality_watch_status') or ''}`",
        f"- shortline_execution_quality_watch_brief: `{payload.get('shortline_execution_quality_watch_brief') or ''}`",
        f"- shortline_execution_quality_watch_decision: `{payload.get('shortline_execution_quality_watch_decision') or ''}`",
        f"- shortline_execution_quality_watch_pattern_family: `{payload.get('shortline_execution_quality_watch_pattern_family') or ''}`",
        f"- shortline_execution_quality_watch_pattern_stage: `{payload.get('shortline_execution_quality_watch_pattern_stage') or ''}`",
        f"- shortline_slippage_snapshot_status: `{payload.get('shortline_slippage_snapshot_status') or ''}`",
        f"- shortline_slippage_snapshot_brief: `{payload.get('shortline_slippage_snapshot_brief') or ''}`",
        f"- shortline_slippage_snapshot_decision: `{payload.get('shortline_slippage_snapshot_decision') or ''}`",
        f"- shortline_slippage_snapshot_pattern_family: `{payload.get('shortline_slippage_snapshot_pattern_family') or ''}`",
        f"- shortline_slippage_snapshot_pattern_stage: `{payload.get('shortline_slippage_snapshot_pattern_stage') or ''}`",
        f"- shortline_slippage_snapshot_post_cost_viable: `{payload.get('shortline_slippage_snapshot_post_cost_viable')}`",
        f"- shortline_slippage_snapshot_estimated_roundtrip_cost_bps: `{payload.get('shortline_slippage_snapshot_estimated_roundtrip_cost_bps')}`",
        f"- shortline_fill_capacity_watch_status: `{payload.get('shortline_fill_capacity_watch_status') or ''}`",
        f"- shortline_fill_capacity_watch_brief: `{payload.get('shortline_fill_capacity_watch_brief') or ''}`",
        f"- shortline_fill_capacity_watch_decision: `{payload.get('shortline_fill_capacity_watch_decision') or ''}`",
        f"- shortline_fill_capacity_watch_pattern_family: `{payload.get('shortline_fill_capacity_watch_pattern_family') or ''}`",
        f"- shortline_fill_capacity_watch_pattern_stage: `{payload.get('shortline_fill_capacity_watch_pattern_stage') or ''}`",
        f"- shortline_fill_capacity_watch_fill_capacity_viable: `{payload.get('shortline_fill_capacity_watch_fill_capacity_viable')}`",
        f"- shortline_fill_capacity_watch_entry_headroom_bps: `{payload.get('shortline_fill_capacity_watch_entry_headroom_bps')}`",
        f"- shortline_sizing_watch_status: `{payload.get('shortline_sizing_watch_status') or ''}`",
        f"- shortline_sizing_watch_brief: `{payload.get('shortline_sizing_watch_brief') or ''}`",
        f"- shortline_sizing_watch_decision: `{payload.get('shortline_sizing_watch_decision') or ''}`",
        f"- shortline_sizing_watch_pattern_family: `{payload.get('shortline_sizing_watch_pattern_family') or ''}`",
        f"- shortline_sizing_watch_pattern_stage: `{payload.get('shortline_sizing_watch_pattern_stage') or ''}`",
        f"- shortline_signal_quality_watch_status: `{payload.get('shortline_signal_quality_watch_status') or ''}`",
        f"- shortline_signal_quality_watch_brief: `{payload.get('shortline_signal_quality_watch_brief') or ''}`",
        f"- shortline_signal_quality_watch_decision: `{payload.get('shortline_signal_quality_watch_decision') or ''}`",
        f"- shortline_signal_quality_watch_pattern_family: `{payload.get('shortline_signal_quality_watch_pattern_family') or ''}`",
        f"- shortline_signal_quality_watch_pattern_stage: `{payload.get('shortline_signal_quality_watch_pattern_stage') or ''}`",
        f"- focus_review_edge_score: `{payload.get('focus_review_edge_score')}`",
        f"- focus_review_structure_score: `{payload.get('focus_review_structure_score')}`",
        f"- focus_review_micro_score: `{payload.get('focus_review_micro_score')}`",
        f"- focus_review_composite_score: `{payload.get('focus_review_composite_score')}`",
        f"- focus_review_priority_status: `{payload.get('focus_review_priority_status') or ''}`",
        f"- focus_review_priority_score: `{payload.get('focus_review_priority_score')}`",
        f"- focus_review_priority_tier: `{payload.get('focus_review_priority_tier') or ''}`",
        f"- focus_review_priority_brief: `{payload.get('focus_review_priority_brief') or ''}`",
        f"- review_priority_queue_status: `{payload.get('review_priority_queue_status') or ''}`",
        f"- review_priority_queue_count: `{payload.get('review_priority_queue_count')}`",
        f"- review_priority_queue_brief: `{payload.get('review_priority_queue_brief') or ''}`",
        f"- review_priority_head_symbol: `{payload.get('review_priority_head_symbol') or ''}`",
        f"- review_priority_head_tier: `{payload.get('review_priority_head_tier') or ''}`",
        f"- review_priority_head_score: `{payload.get('review_priority_head_score')}`",
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
        f"- crypto_route_head_downstream_embedding_status: `{payload.get('crypto_route_head_downstream_embedding_status') or ''}`",
        f"- crypto_route_head_downstream_embedding_brief: `{payload.get('crypto_route_head_downstream_embedding_brief') or ''}`",
        f"- crypto_route_head_downstream_embedding_artifact: `{payload.get('crypto_route_head_downstream_embedding_artifact') or ''}`",
        f"- crypto_route_head_downstream_embedding_as_of: `{payload.get('crypto_route_head_downstream_embedding_as_of') or ''}`",
        f"- crypto_route_head_downstream_embedding_blocker_detail: `{payload.get('crypto_route_head_downstream_embedding_blocker_detail') or ''}`",
        f"- crypto_route_head_downstream_embedding_done_when: `{payload.get('crypto_route_head_downstream_embedding_done_when') or ''}`",
        "",
        "## Steps",
    ]
    for row in payload.get("steps", []):
        lines.append(
            f"- `{row.get('rank')}` `{row.get('name')}` status=`{row.get('status')}` artifact=`{row.get('artifact')}`"
        )
    lines.extend(
        [
            "",
            "## Refresh Reuse Gate",
            f"- level: `{payload.get('crypto_route_refresh_reuse_level') or '-'}`",
            f"- status: `{payload.get('crypto_route_refresh_reuse_gate_status') or '-'}`",
            f"- brief: `{payload.get('crypto_route_refresh_reuse_gate_brief') or '-'}`",
            f"- blocking: `{payload.get('crypto_route_refresh_reuse_gate_blocking')}`",
            f"- blocker_detail: `{payload.get('crypto_route_refresh_reuse_gate_blocker_detail') or '-'}`",
            f"- done_when: `{payload.get('crypto_route_refresh_reuse_gate_done_when') or '-'}`",
            "",
            "## Head Source Refresh",
            f"- status: `{payload.get('crypto_route_head_source_refresh_status') or '-'}`",
            f"- brief: `{payload.get('crypto_route_head_source_refresh_brief') or '-'}`",
            f"- symbol: `{payload.get('crypto_route_head_source_refresh_symbol') or '-'}`",
            f"- action: `{payload.get('crypto_route_head_source_refresh_action') or '-'}`",
            f"- source: `{payload.get('crypto_route_head_source_refresh_source_kind') or '-'}:{payload.get('crypto_route_head_source_refresh_source_health') or '-'}`",
            f"- source_artifact: `{payload.get('crypto_route_head_source_refresh_source_artifact') or '-'}`",
            f"- priority_tier: `{payload.get('crypto_route_head_source_refresh_priority_tier') or '-'}`",
            f"- route_action: `{payload.get('crypto_route_head_source_refresh_route_action') or '-'}`",
            f"- blocker_detail: `{payload.get('crypto_route_head_source_refresh_blocker_detail') or '-'}`",
            f"- done_when: `{payload.get('crypto_route_head_source_refresh_done_when') or '-'}`",
            "",
            "## Downstream Embedding",
            f"- status: `{payload.get('crypto_route_head_downstream_embedding_status') or '-'}`",
            f"- brief: `{payload.get('crypto_route_head_downstream_embedding_brief') or '-'}`",
            f"- artifact: `{payload.get('crypto_route_head_downstream_embedding_artifact') or '-'}`",
            f"- as_of: `{payload.get('crypto_route_head_downstream_embedding_as_of') or '-'}`",
            f"- blocker_detail: `{payload.get('crypto_route_head_downstream_embedding_blocker_detail') or '-'}`",
            f"- done_when: `{payload.get('crypto_route_head_downstream_embedding_done_when') or '-'}`",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refresh crypto route artifacts in a stable guarded sequence.")
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--now", default="")
    parser.add_argument("--rpm", type=int, default=20)
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument("--per-symbol-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--summarize-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--job-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--skip-native-refresh", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    review_dir = Path(args.review_dir).expanduser().resolve()
    output_root = normalize_output_root(Path(args.output_root), review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    runtime_now = derive_runtime_now(review_dir, args.now)
    python_bin = "python3"

    steps: list[dict[str, Any]] = []
    next_now = runtime_now
    native_refresh_mode = "skip_native_refresh" if bool(args.skip_native_refresh) else "full_guarded_refresh"

    def record_step(rank: int, name: str, payload: dict[str, Any], status_key: str = "status") -> None:
        steps.append(
            {
                "rank": rank,
                "name": name,
                "artifact": str(payload.get("artifact") or ""),
                "status": str(payload.get(status_key) or payload.get("ok") or ""),
                "now": payload_or_artifact_as_of(payload),
            }
        )

    def advance_now(payload: dict[str, Any]) -> None:
        nonlocal next_now
        effective = payload_as_of(payload)
        candidate = next_now + dt.timedelta(seconds=1)
        if effective is not None:
            candidate = max(candidate, effective + dt.timedelta(seconds=1))
        next_now = candidate

    for group_rank, spec in enumerate(DEFAULT_NATIVE_GROUP_SPECS, start=1):
        if bool(args.skip_native_refresh):
            reused_artifact = latest_native_group_artifact(review_dir, str(spec["symbol_group"]), runtime_now)
            reused_as_of = parsed_artifact_stamp(reused_artifact) if reused_artifact else None
            payload = {
                "ok": bool(reused_artifact),
                "status": "reused_previous_artifact" if reused_artifact else "missing_reused_source",
                "as_of": fmt_utc(reused_as_of) if reused_as_of else "",
                "artifact": str(reused_artifact or ""),
                "symbol_group": str(spec["symbol_group"]),
                "control_note": (
                    f"Native refresh skipped; reusing latest {spec['symbol_group']} native artifact."
                    if reused_artifact
                    else f"Native refresh skipped, but no existing {spec['symbol_group']} native artifact was found."
                ),
            }
            record_step(group_rank, str(spec["name"]), payload)
            continue

        step_dt = next_now
        cmd = [
            python_bin,
            str(script_path("run_backtest_binance_indicator_combo_native_crypto_guarded.py")),
            "--review-dir",
            str(review_dir),
            "--symbols",
            str(spec["symbols"]),
            "--symbol-group",
            str(spec["symbol_group"]),
            "--interval",
            "1h",
            "--lookback-bars",
            str(int(spec["lookback_bars"])),
            "--sample-windows",
            str(int(spec["sample_windows"])),
            "--window-bars",
            str(int(spec["window_bars"])),
            "--hold-bars",
            "4",
            "--binance-limit",
            "300",
            "--binance-period",
            "1h",
            "--rpm",
            str(int(args.rpm)),
            "--timeout-ms",
            str(int(args.timeout_ms)),
            "--per-symbol-timeout-seconds",
            str(float(args.per_symbol_timeout_seconds)),
            "--summarize-timeout-seconds",
            str(float(args.summarize_timeout_seconds)),
            "--job-timeout-seconds",
            str(float(args.job_timeout_seconds)),
            "--artifact-ttl-hours",
            str(float(args.artifact_ttl_hours)),
            "--artifact-keep",
            str(int(args.artifact_keep)),
            "--now",
            fmt_utc(step_dt),
        ]
        payload = run_json_step(step_name=str(spec["name"]), cmd=cmd)
        record_step(group_rank, str(spec["name"]), payload)
        advance_now(payload)

    scripted_steps: tuple[tuple[str, str], ...] = (
        ("build_source_control_report", "build_binance_indicator_source_control_report.py"),
        ("build_native_group_report", "build_binance_indicator_native_group_report.py"),
        ("build_native_lane_stability_report", "build_binance_indicator_native_lane_stability_report.py"),
        ("build_native_beta_leg_report", "build_binance_indicator_beta_leg_report.py"),
        ("build_native_lane_playbook", "build_binance_indicator_native_lane_playbook.py"),
        ("build_native_beta_leg_window_report", "build_binance_indicator_beta_leg_window_report.py"),
        ("build_bnb_flow_focus", "build_binance_indicator_bnb_flow_focus.py"),
        ("build_combo_playbook", "build_binance_indicator_combo_playbook.py"),
        ("build_symbol_route_handoff", "build_binance_indicator_symbol_route_handoff.py"),
        ("build_system_time_sync_environment_report", "build_system_time_sync_environment_report.py"),
        ("build_crypto_cvd_semantic_snapshot", "build_crypto_cvd_semantic_snapshot.py"),
        ("build_crypto_cvd_queue_handoff", "build_crypto_cvd_queue_handoff.py"),
        ("build_crypto_shortline_live_bars_snapshot", "build_crypto_shortline_live_bars_snapshot.py"),
        ("build_crypto_shortline_execution_gate", "build_crypto_shortline_execution_gate.py"),
        ("build_crypto_shortline_profile_location_watch", "build_crypto_shortline_profile_location_watch.py"),
        ("build_crypto_shortline_mss_watch", "build_crypto_shortline_mss_watch.py"),
        ("build_crypto_shortline_cvd_confirmation_watch", "build_crypto_shortline_cvd_confirmation_watch.py"),
        ("build_crypto_shortline_retest_watch", "build_crypto_shortline_retest_watch.py"),
        ("build_crypto_shortline_pattern_router", "build_crypto_shortline_pattern_router.py"),
        ("build_crypto_route_brief", "build_crypto_route_brief.py"),
        ("build_crypto_route_operator_brief", "build_crypto_route_operator_brief.py"),
        ("build_crypto_shortline_material_change_trigger", "build_crypto_shortline_material_change_trigger.py"),
        ("build_crypto_shortline_liquidity_event_trigger", "build_crypto_shortline_liquidity_event_trigger.py"),
        ("build_crypto_shortline_live_orderflow_snapshot", "build_crypto_shortline_live_orderflow_snapshot.py"),
        ("build_crypto_shortline_execution_quality_watch", "build_crypto_shortline_execution_quality_watch.py"),
        ("build_signal_to_order_tickets", "build_order_ticket.py"),
        ("build_crypto_shortline_slippage_snapshot", "build_crypto_shortline_slippage_snapshot.py"),
        ("build_crypto_shortline_fill_capacity_watch", "build_crypto_shortline_fill_capacity_watch.py"),
        ("build_crypto_shortline_sizing_watch", "build_crypto_shortline_sizing_watch.py"),
        ("build_crypto_shortline_signal_quality_watch", "build_crypto_shortline_signal_quality_watch.py"),
    )
    fallback_suffix_by_step: dict[str, str] = {
        "build_crypto_shortline_execution_gate": "crypto_shortline_execution_gate",
    }

    scripted_payloads: dict[str, dict[str, Any]] = {}
    for rank_index, (step_name, script_name) in enumerate(scripted_steps, start=len(steps) + 1):
        step_dt = next_now
        micro_capture_artifact = latest_micro_capture_artifact(output_root / "artifacts" / "micro_capture", step_dt)
        cmd = [
            python_bin,
            str(script_path(script_name)),
            "--review-dir",
            str(review_dir),
        ]
        if step_name == "build_crypto_shortline_execution_gate":
            cmd.extend(
                [
                    "--output-root",
                    str(output_root),
                    "--artifact-dir",
                    str(output_root / "artifacts" / "micro_capture"),
                    "--config",
                    str(SYSTEM_ROOT / "config.yaml"),
                    "--now",
                    fmt_utc(step_dt),
                ]
            )
        elif step_name == "build_crypto_cvd_semantic_snapshot":
            cmd.extend(
                [
                    "--artifact-dir",
                    str(output_root / "artifacts" / "micro_capture"),
                    "--now",
                    fmt_utc(step_dt),
                ]
            )
            if micro_capture_artifact is not None:
                cmd.extend(["--micro-capture-file", str(micro_capture_artifact)])
            time_sync_environment_payload = scripted_payloads.get("build_system_time_sync_environment_report") or {}
            time_sync_environment_artifact = str(time_sync_environment_payload.get("artifact") or "").strip()
            if time_sync_environment_artifact:
                cmd.extend(["--time-sync-environment-file", time_sync_environment_artifact])
        elif step_name == "build_crypto_cvd_queue_handoff":
            cmd.extend(["--now", fmt_utc(step_dt)])
        elif step_name in {
            "build_source_control_report",
            "build_native_group_report",
            "build_native_lane_stability_report",
            "build_native_beta_leg_report",
            "build_native_lane_playbook",
            "build_native_beta_leg_window_report",
            "build_bnb_flow_focus",
            "build_combo_playbook",
            "build_symbol_route_handoff",
            "build_system_time_sync_environment_report",
            "build_crypto_route_brief",
            "build_crypto_route_operator_brief",
            "build_crypto_shortline_live_bars_snapshot",
            "build_crypto_shortline_profile_location_watch",
            "build_crypto_shortline_mss_watch",
            "build_crypto_shortline_cvd_confirmation_watch",
            "build_crypto_shortline_retest_watch",
            "build_crypto_shortline_pattern_router",
            "build_crypto_shortline_material_change_trigger",
            "build_crypto_shortline_liquidity_event_trigger",
            "build_crypto_shortline_live_orderflow_snapshot",
            "build_crypto_shortline_execution_quality_watch",
            "build_crypto_shortline_slippage_snapshot",
            "build_crypto_shortline_fill_capacity_watch",
            "build_crypto_shortline_sizing_watch",
            "build_crypto_shortline_signal_quality_watch",
            "build_signal_to_order_tickets",
        }:
            if step_name == "build_crypto_shortline_live_bars_snapshot":
                cmd.extend(["--output-root", str(output_root)])
            if step_name == "build_crypto_shortline_live_orderflow_snapshot":
                cmd.extend(
                    [
                        "--artifact-dir",
                        str(output_root / "artifacts" / "micro_capture"),
                        "--config",
                        str(SYSTEM_ROOT / "config.yaml"),
                    ]
                )
                if micro_capture_artifact is not None:
                    cmd.extend(["--micro-capture-file", str(micro_capture_artifact)])
            if step_name == "build_signal_to_order_tickets":
                cmd.extend(
                    [
                        "--output-root",
                        str(output_root),
                        "--review-dir",
                        str(review_dir),
                        "--output-dir",
                        str(review_dir),
                        "--config",
                        str(SYSTEM_ROOT / "config.yaml"),
                        "--artifact-ttl-hours",
                        str(int(args.artifact_ttl_hours)),
                    ]
                )
            cmd.extend(["--now", fmt_utc(step_dt)])
        try:
            payload = run_json_step(step_name=step_name, cmd=cmd)
        except RuntimeError as exc:
            fallback_suffix = fallback_suffix_by_step.get(step_name, "")
            if not fallback_suffix or "no_local_bars_artifact" not in str(exc):
                raise
            payload = reuse_previous_artifact_payload(
                review_dir=review_dir,
                suffix=fallback_suffix,
                reference_now=runtime_now,
                control_note=(
                    "Shortline gate refresh skipped because no_local_bars_artifact; "
                    "reusing latest shortline execution gate artifact."
                ),
            )
        if step_name == "build_signal_to_order_tickets":
            payload["artifact"] = str(payload.get("json") or payload.get("artifact") or "")
            payload.setdefault("status", "ok")
        scripted_payloads[step_name] = payload
        record_step(rank_index, step_name, payload)
        advance_now(payload)

    final_now = next_now
    stamp = final_now.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = review_dir / f"{stamp}_crypto_route_refresh.json"
    md_path = review_dir / f"{stamp}_crypto_route_refresh.md"
    checksum_path = review_dir / f"{stamp}_crypto_route_refresh_checksum.json"

    source_control_payload = scripted_payloads["build_source_control_report"]
    route_handoff_payload = scripted_payloads["build_symbol_route_handoff"]
    time_sync_environment_payload = scripted_payloads["build_system_time_sync_environment_report"]
    cvd_semantic_payload = scripted_payloads["build_crypto_cvd_semantic_snapshot"]
    cvd_queue_handoff_payload = scripted_payloads["build_crypto_cvd_queue_handoff"]
    shortline_gate_payload = scripted_payloads["build_crypto_shortline_execution_gate"]
    profile_location_watch_payload = scripted_payloads["build_crypto_shortline_profile_location_watch"]
    mss_watch_payload = scripted_payloads["build_crypto_shortline_mss_watch"]
    cvd_confirmation_watch_payload = scripted_payloads["build_crypto_shortline_cvd_confirmation_watch"]
    retest_watch_payload = scripted_payloads["build_crypto_shortline_retest_watch"]
    pattern_router_payload = scripted_payloads["build_crypto_shortline_pattern_router"]
    brief_payload = scripted_payloads["build_crypto_route_brief"]
    operator_brief_payload = scripted_payloads["build_crypto_route_operator_brief"]
    live_bars_snapshot_payload = scripted_payloads["build_crypto_shortline_live_bars_snapshot"]
    material_change_trigger_payload = scripted_payloads["build_crypto_shortline_material_change_trigger"]
    liquidity_event_trigger_payload = scripted_payloads["build_crypto_shortline_liquidity_event_trigger"]
    live_orderflow_snapshot_payload = scripted_payloads["build_crypto_shortline_live_orderflow_snapshot"]
    execution_quality_watch_payload = scripted_payloads["build_crypto_shortline_execution_quality_watch"]
    slippage_snapshot_payload = scripted_payloads["build_crypto_shortline_slippage_snapshot"]
    fill_capacity_watch_payload = scripted_payloads["build_crypto_shortline_fill_capacity_watch"]
    sizing_watch_payload = scripted_payloads["build_crypto_shortline_sizing_watch"]
    signal_quality_watch_payload = scripted_payloads["build_crypto_shortline_signal_quality_watch"]
    signal_to_order_tickets_payload = scripted_payloads["build_signal_to_order_tickets"]
    head_source_refresh = crypto_route_head_source_refresh_lane(
        brief_payload=brief_payload,
        operator_brief_payload=operator_brief_payload,
        brief_artifact=str(brief_payload.get("artifact") or ""),
        operator_brief_artifact=str(operator_brief_payload.get("artifact") or ""),
    )
    head_downstream_embedding = crypto_route_head_downstream_embedding_lane(
        review_dir=review_dir,
        reference_now=final_now,
        head_source_refresh=head_source_refresh,
        route_as_of=payload_as_of(operator_brief_payload) or payload_as_of(brief_payload),
    )
    refresh_reuse_gate = crypto_route_refresh_reuse_gate(steps, native_refresh_mode)
    aggregate_status = crypto_route_refresh_aggregate_status(steps)

    payload = {
        "ok": bool(aggregate_status.get("ok")),
        "status": str(aggregate_status.get("status") or "ok"),
        "as_of": fmt_utc(final_now),
        "native_refresh_mode": native_refresh_mode,
        "failure_step_count": int(aggregate_status.get("failure_step_count") or 0),
        "failure_step_brief": str(aggregate_status.get("failure_step_brief") or ""),
        "output_root": str(output_root),
        "review_dir": str(review_dir),
        "cvd_semantic_snapshot_artifact": str(cvd_semantic_payload.get("artifact") or ""),
        "cvd_semantic_environment_status": str(cvd_semantic_payload.get("environment_status") or ""),
        "cvd_semantic_environment_classification": str(cvd_semantic_payload.get("environment_classification") or ""),
        "cvd_semantic_environment_blocker_detail": str(cvd_semantic_payload.get("environment_blocker_detail") or ""),
        "cvd_semantic_environment_remediation_hint": str(cvd_semantic_payload.get("environment_remediation_hint") or ""),
        "cvd_semantic_time_sync_status": str(cvd_semantic_payload.get("time_sync_status") or ""),
        "cvd_semantic_time_sync_classification": str(cvd_semantic_payload.get("time_sync_classification") or ""),
        "cvd_semantic_time_sync_blocker_detail": str(cvd_semantic_payload.get("time_sync_blocker_detail") or ""),
        "cvd_semantic_time_sync_remediation_hint": str(cvd_semantic_payload.get("time_sync_remediation_hint") or ""),
        "cvd_queue_handoff_artifact": str(cvd_queue_handoff_payload.get("artifact") or ""),
        "shortline_execution_gate_artifact": str(shortline_gate_payload.get("artifact") or ""),
        "shortline_pattern_router_artifact": str(pattern_router_payload.get("artifact") or ""),
        "shortline_profile_location_watch_artifact": str(
            profile_location_watch_payload.get("artifact") or ""
        ),
        "shortline_mss_watch_artifact": str(mss_watch_payload.get("artifact") or ""),
        "shortline_cvd_confirmation_watch_artifact": str(
            cvd_confirmation_watch_payload.get("artifact") or ""
        ),
        "shortline_retest_watch_artifact": str(retest_watch_payload.get("artifact") or ""),
        "shortline_live_bars_snapshot_artifact": str(
            live_bars_snapshot_payload.get("artifact") or ""
        ),
        "shortline_material_change_trigger_artifact": str(
            material_change_trigger_payload.get("artifact") or ""
        ),
        "shortline_liquidity_event_trigger_artifact": str(
            liquidity_event_trigger_payload.get("artifact") or ""
        ),
        "shortline_live_orderflow_snapshot_artifact": str(
            live_orderflow_snapshot_payload.get("artifact") or ""
        ),
        "shortline_execution_quality_watch_artifact": str(
            execution_quality_watch_payload.get("artifact") or ""
        ),
        "shortline_slippage_snapshot_artifact": str(
            slippage_snapshot_payload.get("artifact") or ""
        ),
        "shortline_fill_capacity_watch_artifact": str(
            fill_capacity_watch_payload.get("artifact") or ""
        ),
        "shortline_sizing_watch_artifact": str(
            sizing_watch_payload.get("artifact") or ""
        ),
        "shortline_signal_quality_watch_artifact": str(
            signal_quality_watch_payload.get("artifact") or ""
        ),
        "signal_to_order_tickets_artifact": str(
            signal_to_order_tickets_payload.get("artifact") or signal_to_order_tickets_payload.get("json") or ""
        ),
        "signal_to_order_tickets_signal_source_kind": str(
            (signal_to_order_tickets_payload.get("signal_source") or {}).get("kind") or ""
        ),
        "signal_to_order_tickets_symbol_scope_mode": str(
            (signal_to_order_tickets_payload.get("signal_source") or {}).get("symbol_scope_mode") or ""
        ),
        "signal_to_order_tickets_proxy_price_only_count": int(
            (signal_to_order_tickets_payload.get("summary") or {}).get("proxy_price_only_count") or 0
        ),
        "crypto_route_brief_artifact": str(brief_payload.get("artifact") or ""),
        "crypto_route_operator_brief_artifact": str(operator_brief_payload.get("artifact") or ""),
        "source_control_artifact": str(source_control_payload.get("artifact") or ""),
        "route_handoff_artifact": str(route_handoff_payload.get("artifact") or ""),
        "system_time_sync_environment_report_artifact": str(time_sync_environment_payload.get("artifact") or ""),
        "system_time_sync_environment_report_classification": str(
            time_sync_environment_payload.get("classification") or ""
        ),
        "system_time_sync_environment_report_blocker_detail": str(
            time_sync_environment_payload.get("blocker_detail") or ""
        ),
        "system_time_sync_environment_report_remediation_hint": str(
            time_sync_environment_payload.get("remediation_hint") or ""
        ),
        "crypto_route_refresh_reuse_level": str(refresh_reuse_gate.get("level") or ""),
        "crypto_route_refresh_reuse_gate_status": str(refresh_reuse_gate.get("status") or ""),
        "crypto_route_refresh_reuse_gate_brief": str(refresh_reuse_gate.get("brief") or ""),
        "crypto_route_refresh_reuse_gate_blocking": bool(refresh_reuse_gate.get("blocking")),
        "crypto_route_refresh_reuse_gate_blocker_detail": str(refresh_reuse_gate.get("blocker_detail") or ""),
        "crypto_route_refresh_reuse_gate_done_when": str(refresh_reuse_gate.get("done_when") or ""),
        "focus_review_status": str(
            operator_brief_payload.get("focus_review_status")
            or brief_payload.get("focus_review_status")
            or ""
        ),
        "focus_review_brief": str(
            operator_brief_payload.get("focus_review_brief")
            or brief_payload.get("focus_review_brief")
            or ""
        ),
        "focus_review_primary_blocker": str(
            operator_brief_payload.get("focus_review_primary_blocker")
            or brief_payload.get("focus_review_primary_blocker")
            or ""
        ),
        "focus_review_micro_blocker": str(
            operator_brief_payload.get("focus_review_micro_blocker")
            or brief_payload.get("focus_review_micro_blocker")
            or ""
        ),
        "focus_review_blocker_detail": str(
            operator_brief_payload.get("focus_review_blocker_detail")
            or brief_payload.get("focus_review_blocker_detail")
            or ""
        ),
        "focus_review_done_when": str(
            operator_brief_payload.get("focus_review_done_when")
            or brief_payload.get("focus_review_done_when")
            or ""
        ),
        "shortline_material_change_trigger_status": str(
            material_change_trigger_payload.get("trigger_status") or ""
        ),
        "shortline_material_change_trigger_brief": str(
            material_change_trigger_payload.get("trigger_brief") or ""
        ),
        "shortline_material_change_trigger_decision": str(
            material_change_trigger_payload.get("trigger_decision") or ""
        ),
        "shortline_material_change_trigger_rerun_recommended": bool(
            material_change_trigger_payload.get("rerun_recommended", False)
        ),
        "shortline_liquidity_event_trigger_status": str(
            liquidity_event_trigger_payload.get("trigger_status") or ""
        ),
        "shortline_liquidity_event_trigger_brief": str(
            liquidity_event_trigger_payload.get("trigger_brief") or ""
        ),
        "shortline_liquidity_event_trigger_decision": str(
            liquidity_event_trigger_payload.get("trigger_decision") or ""
        ),
        "shortline_liquidity_event_trigger_current_signal_source": str(
            liquidity_event_trigger_payload.get("current_signal_source") or ""
        ),
        "shortline_liquidity_event_trigger_current_signal_source_artifact": str(
            liquidity_event_trigger_payload.get("current_signal_source_artifact") or ""
        ),
        "shortline_liquidity_event_trigger_previous_signal_source": str(
            liquidity_event_trigger_payload.get("previous_signal_source") or ""
        ),
        "shortline_liquidity_event_trigger_previous_signal_source_artifact": str(
            liquidity_event_trigger_payload.get("previous_signal_source_artifact") or ""
        ),
        "shortline_live_bars_snapshot_status": str(
            live_bars_snapshot_payload.get("snapshot_status") or ""
        ),
        "shortline_live_bars_snapshot_brief": str(
            live_bars_snapshot_payload.get("snapshot_brief") or ""
        ),
        "shortline_live_bars_snapshot_decision": str(
            live_bars_snapshot_payload.get("snapshot_decision") or ""
        ),
        "shortline_profile_location_watch_status": str(
            profile_location_watch_payload.get("watch_status") or ""
        ),
        "shortline_profile_location_watch_brief": str(
            profile_location_watch_payload.get("watch_brief") or ""
        ),
        "shortline_profile_location_watch_decision": str(
            profile_location_watch_payload.get("watch_decision") or ""
        ),
        "shortline_profile_location_watch_rotation_proximity_state": str(
            profile_location_watch_payload.get("rotation_proximity_state") or ""
        ),
        "shortline_profile_location_watch_profile_rotation_alignment_band": str(
            profile_location_watch_payload.get("profile_rotation_alignment_band") or ""
        ),
        "shortline_profile_location_watch_profile_rotation_next_milestone": str(
            profile_location_watch_payload.get("profile_rotation_next_milestone") or ""
        ),
        "shortline_profile_location_watch_profile_rotation_confidence": (
            float(profile_location_watch_payload.get("profile_rotation_confidence"))
            if profile_location_watch_payload.get("profile_rotation_confidence") is not None
            else None
        ),
        "shortline_profile_location_watch_active_rotation_targets": [
            str(item).strip()
            for item in (profile_location_watch_payload.get("active_rotation_targets") or [])
            if str(item).strip()
        ],
        "shortline_profile_location_watch_profile_rotation_target_tag": str(
            profile_location_watch_payload.get("profile_rotation_target_tag") or ""
        ),
        "shortline_profile_location_watch_profile_rotation_target_bin_distance": (
            int(profile_location_watch_payload.get("profile_rotation_target_bin_distance"))
            if profile_location_watch_payload.get("profile_rotation_target_bin_distance")
            is not None
            else None
        ),
        "shortline_profile_location_watch_profile_rotation_target_distance_bps": (
            float(profile_location_watch_payload.get("profile_rotation_target_distance_bps"))
            if profile_location_watch_payload.get("profile_rotation_target_distance_bps") is not None
            else None
        ),
        "shortline_pattern_router_status": str(
            pattern_router_payload.get("pattern_status") or ""
        ),
        "shortline_pattern_router_brief": str(
            pattern_router_payload.get("pattern_brief") or ""
        ),
        "shortline_pattern_router_decision": str(
            pattern_router_payload.get("pattern_decision") or ""
        ),
        "shortline_pattern_router_family": str(
            pattern_router_payload.get("pattern_family") or ""
        ),
        "shortline_pattern_router_stage": str(
            pattern_router_payload.get("pattern_stage") or ""
        ),
        "shortline_mss_watch_status": str(
            mss_watch_payload.get("watch_status") or ""
        ),
        "shortline_mss_watch_brief": str(
            mss_watch_payload.get("watch_brief") or ""
        ),
        "shortline_mss_watch_decision": str(
            mss_watch_payload.get("watch_decision") or ""
        ),
        "shortline_cvd_confirmation_watch_status": str(
            cvd_confirmation_watch_payload.get("watch_status") or ""
        ),
        "shortline_cvd_confirmation_watch_brief": str(
            cvd_confirmation_watch_payload.get("watch_brief") or ""
        ),
        "shortline_cvd_confirmation_watch_decision": str(
            cvd_confirmation_watch_payload.get("watch_decision") or ""
        ),
        "shortline_retest_watch_status": str(
            retest_watch_payload.get("watch_status") or ""
        ),
        "shortline_retest_watch_brief": str(
            retest_watch_payload.get("watch_brief") or ""
        ),
        "shortline_retest_watch_decision": str(
            retest_watch_payload.get("watch_decision") or ""
        ),
        "shortline_live_orderflow_snapshot_status": str(
            live_orderflow_snapshot_payload.get("snapshot_status") or ""
        ),
        "shortline_live_orderflow_snapshot_brief": str(
            live_orderflow_snapshot_payload.get("snapshot_brief") or ""
        ),
        "shortline_live_orderflow_snapshot_decision": str(
            live_orderflow_snapshot_payload.get("snapshot_decision") or ""
        ),
        "shortline_execution_quality_watch_status": str(
            execution_quality_watch_payload.get("watch_status") or ""
        ),
        "shortline_execution_quality_watch_brief": str(
            execution_quality_watch_payload.get("watch_brief") or ""
        ),
        "shortline_execution_quality_watch_decision": str(
            execution_quality_watch_payload.get("watch_decision") or ""
        ),
        "shortline_execution_quality_watch_pattern_family": str(
            execution_quality_watch_payload.get("pattern_family") or ""
        ),
        "shortline_execution_quality_watch_pattern_stage": str(
            execution_quality_watch_payload.get("pattern_stage") or ""
        ),
        "shortline_slippage_snapshot_status": str(
            slippage_snapshot_payload.get("snapshot_status") or ""
        ),
        "shortline_slippage_snapshot_brief": str(
            slippage_snapshot_payload.get("snapshot_brief") or ""
        ),
        "shortline_slippage_snapshot_decision": str(
            slippage_snapshot_payload.get("snapshot_decision") or ""
        ),
        "shortline_slippage_snapshot_pattern_family": str(
            slippage_snapshot_payload.get("pattern_family") or ""
        ),
        "shortline_slippage_snapshot_pattern_stage": str(
            slippage_snapshot_payload.get("pattern_stage") or ""
        ),
        "shortline_slippage_snapshot_post_cost_viable": bool(
            slippage_snapshot_payload.get("post_cost_viable", False)
        ),
        "shortline_slippage_snapshot_estimated_roundtrip_cost_bps": (
            float(slippage_snapshot_payload.get("estimated_roundtrip_cost_bps"))
            if slippage_snapshot_payload.get("estimated_roundtrip_cost_bps") is not None
            else None
        ),
        "shortline_fill_capacity_watch_status": str(
            fill_capacity_watch_payload.get("watch_status") or ""
        ),
        "shortline_fill_capacity_watch_brief": str(
            fill_capacity_watch_payload.get("watch_brief") or ""
        ),
        "shortline_fill_capacity_watch_decision": str(
            fill_capacity_watch_payload.get("watch_decision") or ""
        ),
        "shortline_fill_capacity_watch_pattern_family": str(
            fill_capacity_watch_payload.get("pattern_family") or ""
        ),
        "shortline_fill_capacity_watch_pattern_stage": str(
            fill_capacity_watch_payload.get("pattern_stage") or ""
        ),
        "shortline_fill_capacity_watch_fill_capacity_viable": bool(
            fill_capacity_watch_payload.get("fill_capacity_viable", False)
        ),
        "shortline_fill_capacity_watch_entry_headroom_bps": (
            float(fill_capacity_watch_payload.get("entry_headroom_bps"))
            if fill_capacity_watch_payload.get("entry_headroom_bps") is not None
            else None
        ),
        "shortline_sizing_watch_status": str(
            sizing_watch_payload.get("watch_status") or ""
        ),
        "shortline_sizing_watch_brief": str(
            sizing_watch_payload.get("watch_brief") or ""
        ),
        "shortline_sizing_watch_decision": str(
            sizing_watch_payload.get("watch_decision") or ""
        ),
        "shortline_sizing_watch_pattern_family": str(
            sizing_watch_payload.get("pattern_family") or ""
        ),
        "shortline_sizing_watch_pattern_stage": str(
            sizing_watch_payload.get("pattern_stage") or ""
        ),
        "shortline_signal_quality_watch_status": str(
            signal_quality_watch_payload.get("watch_status") or ""
        ),
        "shortline_signal_quality_watch_brief": str(
            signal_quality_watch_payload.get("watch_brief") or ""
        ),
        "shortline_signal_quality_watch_decision": str(
            signal_quality_watch_payload.get("watch_decision") or ""
        ),
        "shortline_signal_quality_watch_pattern_family": str(
            signal_quality_watch_payload.get("pattern_family") or ""
        ),
        "shortline_signal_quality_watch_pattern_stage": str(
            signal_quality_watch_payload.get("pattern_stage") or ""
        ),
        "focus_review_edge_score": int(
            operator_brief_payload.get("focus_review_edge_score")
            or brief_payload.get("focus_review_edge_score")
            or 0
        ),
        "focus_review_structure_score": int(
            operator_brief_payload.get("focus_review_structure_score")
            or brief_payload.get("focus_review_structure_score")
            or 0
        ),
        "focus_review_micro_score": int(
            operator_brief_payload.get("focus_review_micro_score")
            or brief_payload.get("focus_review_micro_score")
            or 0
        ),
        "focus_review_composite_score": int(
            operator_brief_payload.get("focus_review_composite_score")
            or brief_payload.get("focus_review_composite_score")
            or 0
        ),
        "focus_review_priority_status": str(
            operator_brief_payload.get("focus_review_priority_status")
            or brief_payload.get("focus_review_priority_status")
            or ""
        ),
        "focus_review_priority_score": int(
            operator_brief_payload.get("focus_review_priority_score")
            or brief_payload.get("focus_review_priority_score")
            or 0
        ),
        "focus_review_priority_tier": str(
            operator_brief_payload.get("focus_review_priority_tier")
            or brief_payload.get("focus_review_priority_tier")
            or ""
        ),
        "focus_review_priority_brief": str(
            operator_brief_payload.get("focus_review_priority_brief")
            or brief_payload.get("focus_review_priority_brief")
            or ""
        ),
        "review_priority_queue_status": str(
            operator_brief_payload.get("review_priority_queue_status")
            or brief_payload.get("review_priority_queue_status")
            or ""
        ),
        "review_priority_queue_count": int(
            operator_brief_payload.get("review_priority_queue_count")
            or brief_payload.get("review_priority_queue_count")
            or 0
        ),
        "review_priority_queue_brief": str(
            operator_brief_payload.get("review_priority_queue_brief")
            or brief_payload.get("review_priority_queue_brief")
            or ""
        ),
        "review_priority_head_symbol": str(
            operator_brief_payload.get("review_priority_head_symbol")
            or brief_payload.get("review_priority_head_symbol")
            or ""
        ),
        "review_priority_head_tier": str(
            operator_brief_payload.get("review_priority_head_tier")
            or brief_payload.get("review_priority_head_tier")
            or ""
        ),
        "review_priority_head_score": int(
            operator_brief_payload.get("review_priority_head_score")
            or brief_payload.get("review_priority_head_score")
            or 0
        ),
        "review_priority_queue": [
            dict(row)
            for row in (
                operator_brief_payload.get("review_priority_queue")
                or brief_payload.get("review_priority_queue")
                or []
            )
            if isinstance(row, dict)
        ],
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
        "crypto_route_head_downstream_embedding_status": str(head_downstream_embedding.get("status") or ""),
        "crypto_route_head_downstream_embedding_brief": str(head_downstream_embedding.get("brief") or ""),
        "crypto_route_head_downstream_embedding_artifact": str(head_downstream_embedding.get("artifact") or ""),
        "crypto_route_head_downstream_embedding_as_of": str(head_downstream_embedding.get("as_of") or ""),
        "crypto_route_head_downstream_embedding_blocker_detail": str(
            head_downstream_embedding.get("blocker_detail") or ""
        ),
        "crypto_route_head_downstream_embedding_done_when": str(
            head_downstream_embedding.get("done_when") or ""
        ),
        "steps": steps,
        "artifact_label": f"crypto-route-refresh:{aggregate_status.get('status') or 'ok'}",
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
        stem="crypto_route_refresh",
        current_paths=[json_path, md_path, checksum_path],
        keep=max(3, int(args.artifact_keep)),
        ttl_hours=float(args.artifact_ttl_hours),
        now_dt=final_now,
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
    checksum_payload["files"][0]["sha256"] = sha256_file(json_path)
    checksum_path.write_text(json.dumps(checksum_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
