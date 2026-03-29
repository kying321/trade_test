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
DEFAULT_CONTEXT_PATH = DEFAULT_REVIEW_DIR / "NEXT_WINDOW_CONTEXT_LATEST.md"
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


def latest_review_json_artifact(
    review_dir: Path,
    stem: str,
    reference_now: dt.datetime | None = None,
) -> Path | None:
    candidates = list(review_dir.glob(f"*_{stem}.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


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
            "brooks_structure_refresh",
            "commodity_paper_execution_refresh",
            "hot_universe_operator_brief",
            "cross_market_operator_state",
        ],
    )
    if latest_dt is not None and latest_dt >= runtime_now:
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


def load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def latest_review_json_artifact(review_dir: Path, suffix: str, reference_now: dt.datetime) -> Path | None:
    candidates = sorted(
        review_dir.glob(f"*_{suffix}.json"),
        key=lambda item: artifact_sort_key(item, reference_now),
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_reused_refresh_payload(
    *,
    review_dir: Path,
    suffix: str,
    reference_now: dt.datetime,
    step_name: str,
) -> dict[str, Any]:
    artifact_path = latest_review_json_artifact(review_dir, suffix, reference_now)
    if artifact_path is None:
        raise RuntimeError(f"{step_name}_missing_reused_source")
    payload = load_json_mapping(artifact_path)
    payload.setdefault("artifact", str(artifact_path))
    payload.setdefault("status", "ok")
    payload.setdefault("as_of", payload_or_artifact_as_of(payload) or fmt_utc(reference_now))
    return payload


def resolve_embedded_artifact_source(
    *,
    payload: dict[str, Any] | None,
    key: str,
    review_dir: Path,
    suffix: str,
    reference_now: dt.datetime,
) -> Path | None:
    source = dict(payload or {})
    explicit_text = str(source.get(key) or "").strip()
    if explicit_text:
        explicit_path = Path(explicit_text).expanduser()
        if explicit_path.exists():
            return explicit_path
    return latest_review_json_artifact(review_dir, suffix, reference_now)


def load_optional_payload(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = load_json_mapping(path)
    except Exception:
        return {}
    payload.setdefault("artifact", str(path))
    return payload


def resolve_hot_brief_source(
    *,
    review_dir: Path,
    commodity_refresh: dict[str, Any],
    reference_now: dt.datetime,
) -> Path:
    explicit_path = Path(str(commodity_refresh.get("brief_artifact") or "")).expanduser()
    if explicit_path.exists():
        return explicit_path
    fallback_path = latest_review_json_artifact(
        review_dir,
        "hot_universe_operator_brief",
        reference_now,
    )
    if fallback_path and fallback_path.exists():
        return fallback_path
    if not str(commodity_refresh.get("brief_artifact") or "").strip():
        raise RuntimeError("cross_market_hot_brief_missing")
    if not explicit_path.exists():
        raise RuntimeError("cross_market_hot_brief_missing")
    return explicit_path


def _tier_rank(tier: str) -> int:
    return {
        "review_queue_now": 0,
        "review_queue_next": 1,
        "blocked_review": 2,
        "route_candidate_only": 3,
        "deprioritized_review": 4,
        "watch_queue_only": 5,
    }.get(str(tier or "").strip(), 6)


def _operator_state_rank(state: str) -> int:
    return {
        "waiting": 0,
        "review": 1,
        "watch": 2,
        "blocked": 3,
        "repair": 4,
    }.get(str(state or "").strip(), 5)


def _normalize_operator_state(
    *,
    state: str,
    priority_tier: str,
    route_status_label: str = "",
    plan_status: str = "",
) -> str:
    state_text = str(state or "").strip()
    tier_text = str(priority_tier or "").strip()
    route_label = str(route_status_label or "").strip()
    plan_text = str(plan_status or "").strip()
    if state_text:
        return state_text
    if tier_text.startswith("review_queue"):
        return "review"
    if tier_text == "watch_queue_only":
        return "watch"
    if tier_text == "blocked_review" or plan_text.startswith("blocked_"):
        return "blocked"
    if route_label.startswith("watch"):
        return "watch"
    return "review"


def _action_lane_state(*, area: str, action: str, fallback_state: str) -> str:
    fallback_text = str(fallback_state or "").strip()
    area_text = str(area or "").strip()
    action_text = str(action or "").strip()
    if fallback_text:
        return fallback_text
    if area_text in {"ops_live_gate", "risk_guard"} or action_text.startswith("clear_"):
        return "repair"
    if action_text.startswith("watch_"):
        return "watch"
    if action_text.startswith("wait_"):
        return "waiting"
    return "review"


def _review_source_age_minutes(*, reference_now: dt.datetime, source_as_of: str) -> int | None:
    text = str(source_as_of or "").strip()
    if not text:
        return None
    try:
        parsed = parse_now(text)
    except ValueError:
        return None
    delta = reference_now - parsed
    if delta.total_seconds() < 0:
        return 0
    return int(delta.total_seconds() // 60)


def _review_source_recency(*, source_age_minutes: int | None) -> str:
    if source_age_minutes is None:
        return "unknown"
    if source_age_minutes <= 15:
        return "fresh"
    if source_age_minutes <= 1440:
        return "carry_over"
    return "stale"


def _review_source_health(*, source_status: str, source_recency: str) -> str:
    status_text = str(source_status or "").strip()
    recency_text = str(source_recency or "").strip()
    if status_text == "ok" and recency_text == "fresh":
        return "ready"
    if status_text == "ok" and recency_text == "carry_over":
        return "carry_over_ok"
    if status_text == "ok" and recency_text == "stale":
        return "refresh_required"
    if status_text == "dry_run":
        return "dry_run_only"
    if status_text:
        return f"status_{status_text}"
    return "unknown"


def _review_source_refresh_action(*, source_health: str) -> str:
    health_text = str(source_health or "").strip()
    if health_text == "ready":
        return "read_current_artifact"
    if health_text == "carry_over_ok":
        return "consider_refresh_before_promotion"
    if health_text == "refresh_required":
        return "refresh_source_before_use"
    if health_text == "environment_blocked":
        return "repair_public_data_environment"
    if health_text == "dry_run_only":
        return "do_not_promote_without_non_dry_run"
    return "inspect_source_state"


def _crypto_refresh_dependency_health(refresh_payload: dict[str, Any]) -> str:
    critical_steps = {
        "build_crypto_cvd_semantic_snapshot",
        "build_crypto_cvd_queue_handoff",
        "build_crypto_shortline_execution_gate",
    }
    step_rows = [
        dict(row)
        for row in list(refresh_payload.get("steps") or [])
        if isinstance(row, dict) and str(row.get("name") or "").strip() in critical_steps
    ]
    if not step_rows:
        return ""
    statuses = {str(row.get("status") or "").strip() for row in step_rows}
    if statuses <= {"ok"}:
        return ""
    if {
        "micro_capture_missing",
        "semantic_snapshot_invalid",
        "carry_over_previous_artifact",
        "missing_reused_source",
    } & statuses:
        return "refresh_required"
    return "inspect_source_state"


def _review_row_promotion_ready(row: dict[str, Any]) -> bool:
    if "promotion_ready" in row and row.get("promotion_ready") is not None:
        return bool(row.get("promotion_ready"))
    refresh_action = str(row.get("source_refresh_action") or "").strip()
    if refresh_action in {
        "consider_refresh_before_promotion",
        "refresh_source_before_use",
        "repair_public_data_environment",
        "do_not_promote_without_non_dry_run",
        "inspect_source_state",
    }:
        return False
    return True


def _review_row_effective_action(
    *,
    action: str,
    source_refresh_action: str,
    promotion_ready: bool | None = None,
) -> str:
    action_text = str(action or "").strip()
    refresh_action_text = str(source_refresh_action or "").strip()
    ready = (
        bool(promotion_ready)
        if promotion_ready is not None
        else _review_row_promotion_ready(
            {
                "action": action_text,
                "source_refresh_action": refresh_action_text,
            }
        )
    )
    if not ready and refresh_action_text:
        return refresh_action_text
    return action_text


def _build_review_backlog_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    backlog: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    def append_row(
        *,
        area: str,
        symbol: str,
        action: str,
        status: str,
        priority_score: int,
        priority_tier: str,
        blocker_detail: str,
        done_when: str,
        reason: str = "",
        source_refresh_action: str = "",
        promotion_ready: bool | None = None,
        source_as_of: str = "",
        source_age_minutes: int | None = None,
        source_recency: str = "",
        source_status: str = "",
        source_health: str = "",
        source_kind: str = "",
        source_artifact: str = "",
    ) -> None:
        area_text = str(area or "").strip()
        symbol_text = str(symbol or "").strip().upper()
        action_text = str(action or "").strip()
        if not area_text or not symbol_text or not action_text:
            return
        dedupe_key = (area_text, symbol_text, action_text)
        if dedupe_key in seen:
            return
        seen.add(dedupe_key)
        promotion_ready_flag = (
            bool(promotion_ready)
            if promotion_ready is not None
            else not bool(str(source_refresh_action or "").strip())
        )
        effective_action_text = _review_row_effective_action(
            action=action_text,
            source_refresh_action=str(source_refresh_action or "").strip(),
            promotion_ready=promotion_ready_flag,
        )
        backlog.append(
            {
                "area": area_text,
                "symbol": symbol_text,
                "action": effective_action_text,
                "review_action": action_text,
                "effective_action": effective_action_text,
                "status": str(status or "").strip(),
                "priority_score": int(priority_score or 0),
                "priority_tier": str(priority_tier or "").strip(),
                "reason": str(reason or "").strip(),
                "blocker_detail": str(blocker_detail or "").strip(),
                "done_when": str(done_when or "").strip(),
                "source_refresh_action": str(source_refresh_action or "").strip(),
                "promotion_ready": promotion_ready_flag,
                "source_as_of": str(source_as_of or "").strip(),
                "source_age_minutes": source_age_minutes,
                "source_recency": str(source_recency or "").strip(),
                "source_status": str(source_status or "").strip(),
                "source_health": str(source_health or "").strip(),
                "source_kind": str(source_kind or "").strip(),
                "source_artifact": str(source_artifact or "").strip(),
            }
        )

    for row in rows:
        source = dict(row or {})
        if str(source.get("state") or "").strip() != "review":
            continue
        append_row(
            area=str(source.get("area") or "").strip(),
            symbol=str(source.get("symbol") or "").strip(),
            action=str(source.get("action") or "").strip(),
            status=str(source.get("status") or source.get("state") or "").strip() or "review",
            priority_score=int(source.get("priority_score") or 0),
            priority_tier=str(source.get("priority_tier") or "").strip(),
            reason=str(source.get("reason") or "").strip(),
            blocker_detail=str(source.get("blocker_detail") or "").strip(),
            done_when=str(source.get("done_when") or "").strip(),
            source_refresh_action=str(source.get("source_refresh_action") or "").strip(),
            promotion_ready=(
                bool(source.get("promotion_ready"))
                if source.get("promotion_ready") is not None
                else _review_row_promotion_ready(source)
            ),
            source_as_of=str(source.get("source_as_of") or "").strip(),
            source_age_minutes=source.get("source_age_minutes"),
            source_recency=str(source.get("source_recency") or "").strip(),
            source_status=str(source.get("source_status") or "").strip(),
            source_health=str(source.get("source_health") or "").strip(),
            source_kind=str(source.get("source_kind") or "").strip(),
            source_artifact=str(source.get("source_artifact") or "").strip(),
        )
    backlog.sort(
        key=lambda row: (
            0 if _review_row_promotion_ready(row) else 1,
            _tier_rank(str(row.get("priority_tier") or "")),
            -int(row.get("priority_score") or 0),
            str(row.get("area") or ""),
            str(row.get("symbol") or ""),
        )
    )
    for idx, row in enumerate(backlog, start=1):
        row["rank"] = idx
    return backlog


def _build_review_backlog(brief: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    brooks_status = str(brief.get("brooks_structure_operator_status") or "").strip()
    brooks_symbol = str(brief.get("brooks_structure_operator_head_symbol") or "").strip()
    if brooks_status and brooks_status != "inactive" and brooks_symbol:
        rows.append(
            {
                "area": "brooks_structure",
                "symbol": brooks_symbol,
                "action": str(brief.get("brooks_structure_operator_head_action") or "").strip(),
                "status": brooks_status,
                "state": "review",
                "priority_score": int(brief.get("brooks_structure_operator_head_priority_score") or 0),
                "priority_tier": str(brief.get("brooks_structure_operator_head_priority_tier") or "").strip(),
                "reason": str(
                    brief.get("brooks_structure_operator_head_strategy")
                    or brief.get("brooks_structure_review_head_strategy")
                    or ""
                ).strip(),
                "blocker_detail": str(brief.get("brooks_structure_operator_blocker_detail") or "").strip(),
                "done_when": str(brief.get("brooks_structure_operator_done_when") or "").strip(),
            }
        )
    secondary_state = str(brief.get("secondary_focus_state") or "").strip()
    secondary_symbol = str(brief.get("secondary_focus_symbol") or "").strip()
    if secondary_state == "review" and secondary_symbol:
        rows.append(
            {
                "area": str(brief.get("secondary_focus_area") or "crypto_route").strip() or "crypto_route",
                "symbol": secondary_symbol,
                "action": str(brief.get("secondary_focus_action") or "").strip(),
                "status": secondary_state,
                "state": secondary_state,
                "priority_score": int(brief.get("secondary_focus_priority_score") or 0),
                "priority_tier": str(brief.get("secondary_focus_priority_tier") or "").strip(),
                "reason": str(brief.get("secondary_focus_reason") or "").strip(),
                "blocker_detail": str(brief.get("secondary_focus_blocker_detail") or "").strip(),
                "done_when": str(brief.get("secondary_focus_done_when") or "").strip(),
            }
        )
    return _build_review_backlog_from_rows(rows)


def _remote_live_scope_market(live_gate_blocker_payload: dict[str, Any]) -> str:
    diagnosis = dict(live_gate_blocker_payload.get("remote_live_diagnosis") or {})
    market = str(diagnosis.get("market") or "").strip().lower()
    if market:
        return market
    return str(live_gate_blocker_payload.get("ready_check_scope_market") or "").strip().lower()


def _is_remote_live_executable_head(
    *,
    area: str,
    action: str,
    remote_market: str,
) -> bool:
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


def _build_commodity_waiting_backlog(
    *,
    review_payload: dict[str, Any],
    gap_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    review_source = dict(review_payload or {})
    gap_source = dict(gap_payload or {})
    items = [
        dict(row)
        for row in list(review_source.get("review_items") or [])
        if isinstance(row, dict)
    ]
    item_by_symbol = {
        str(row.get("symbol") or "").strip().upper(): row
        for row in items
        if str(row.get("symbol") or "").strip()
    }

    stale_dates = {
        str(symbol).strip().upper(): str(value).strip()
        for symbol, value in dict(gap_source.get("queue_symbols_with_stale_directional_signal_dates") or {}).items()
        if str(symbol).strip()
    }
    stale_ages = {
        str(symbol).strip().upper(): int(value or 0)
        for symbol, value in dict(gap_source.get("queue_symbols_with_stale_directional_signal_age_days") or {}).items()
        if str(symbol).strip()
    }

    backlog: list[dict[str, Any]] = []

    def append_row(
        *,
        symbol: str,
        area: str,
        action: str,
        reason: str,
        blocker_detail: str,
        done_when: str,
    ) -> None:
        symbol_key = str(symbol or "").strip().upper()
        if not symbol_key:
            return
        item = dict(item_by_symbol.get(symbol_key) or {})
        if not item:
            return
        queue_rank = int(item.get("queue_rank") or len(backlog) + 1)
        backlog.append(
            {
                "area": area,
                "target": str(item.get("execution_id") or symbol_key),
                "symbol": symbol_key,
                "action": action,
                "reason": reason,
                "state": "waiting",
                "priority_score": max(1, 100 - queue_rank),
                "priority_tier": "waiting_now",
                "blocker_detail": blocker_detail,
                "done_when": done_when,
                "command": "",
                "clear_when": "",
            }
        )

    close_symbol = str(review_source.get("next_close_evidence_execution_symbol") or "").strip().upper()
    if close_symbol:
        item = dict(item_by_symbol.get(close_symbol) or {})
        open_date = str(item.get("paper_open_date") or "").strip()
        blocker_detail = (
            f"paper execution evidence is present, but position is still OPEN since {open_date}; waiting for close evidence"
            if open_date
            else "paper execution evidence is present, but position is still OPEN; waiting for close evidence"
        )
        append_row(
            symbol=close_symbol,
            area="commodity_execution_close_evidence",
            action="wait_for_paper_execution_close_evidence",
            reason="paper_execution_close_evidence_pending",
            blocker_detail=blocker_detail,
            done_when=f"{close_symbol} paper_execution_status leaves OPEN and close evidence becomes available",
        )

    fill_symbol = str(review_source.get("next_fill_evidence_execution_symbol") or "").strip().upper()
    if fill_symbol:
        stale_date = stale_dates.get(fill_symbol, "")
        stale_age = int(stale_ages.get(fill_symbol, 0) or 0)
        blocker_detail = "paper execution fill evidence not written"
        if stale_date and stale_age > 0:
            blocker_detail += f"; stale directional signal {stale_age}d since {stale_date}"
        append_row(
            symbol=fill_symbol,
            area="commodity_fill_evidence",
            action="wait_for_paper_execution_fill_evidence",
            reason="paper_execution_fill_evidence_pending",
            blocker_detail=blocker_detail,
            done_when=f"{fill_symbol} gains paper evidence and leaves fill_evidence_pending_symbols",
        )

    return backlog


def _build_crypto_review_rows(
    *,
    reference_now: dt.datetime,
    refresh_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    source_rows = [
        dict(row)
        for row in list(refresh_payload.get("review_priority_queue") or [])
        if isinstance(row, dict)
    ]

    source_as_of = payload_or_artifact_as_of(refresh_payload)
    source_artifact = str(refresh_payload.get("artifact") or "").strip()
    source_status = str(refresh_payload.get("status") or "").strip()
    source_age_minutes = _review_source_age_minutes(reference_now=reference_now, source_as_of=source_as_of)
    source_recency = _review_source_recency(source_age_minutes=source_age_minutes)
    source_health = _review_source_health(source_status=source_status, source_recency=source_recency)
    dependency_health = _crypto_refresh_dependency_health(refresh_payload)
    environment_status = str(refresh_payload.get("cvd_semantic_environment_status") or "").strip()
    environment_classification = str(refresh_payload.get("cvd_semantic_environment_classification") or "").strip()
    environment_blocker_detail = str(refresh_payload.get("cvd_semantic_environment_blocker_detail") or "").strip()
    environment_remediation_hint = str(refresh_payload.get("cvd_semantic_environment_remediation_hint") or "").strip()
    if dependency_health == "refresh_required":
        source_health = "refresh_required"
    elif dependency_health == "inspect_source_state" and source_health == "ready":
        source_health = "status_dependency_unclear"
    if environment_status == "environment_blocked":
        source_health = "environment_blocked"
    source_refresh_action = _review_source_refresh_action(source_health=source_health)

    rows: list[dict[str, Any]] = []
    for row in source_rows:
        symbol = str(row.get("symbol") or "").strip().upper()
        action = str(row.get("route_action") or row.get("action") or "").strip()
        if not symbol or not action:
            continue
        state = _normalize_operator_state(
            state="",
            priority_tier=str(row.get("priority_tier") or ""),
            route_status_label=str(row.get("route_status_label") or ""),
            plan_status="",
        )
        rows.append(
            {
                "area": "crypto_route",
                "target": symbol,
                "symbol": symbol,
                "action": action,
                "reason": str(row.get("reason") or "").strip(),
                "state": state,
                "priority_score": int(row.get("priority_score") or 0),
                "priority_tier": str(row.get("priority_tier") or "").strip() or state,
                "blocker_detail": (
                    (
                        str(row.get("blocker_detail") or "").strip()
                        + " | "
                        + f"source freshness guard: crypto_route_refresh artifact is {source_status or '-'} "
                        + f"and {source_recency or '-'}"
                        + (f", age={source_age_minutes}m" if source_age_minutes is not None else "")
                        + (
                            f"; dependency_health={dependency_health}"
                            if dependency_health
                            else ""
                        )
                        + (
                            f"; environment={environment_classification or environment_status}:"
                            f"{environment_blocker_detail}"
                            if environment_status == "environment_blocked" and environment_blocker_detail
                            else ""
                        )
                    )
                    if source_refresh_action != "read_current_artifact"
                    else str(row.get("blocker_detail") or "").strip()
                ),
                "done_when": (
                    (
                        environment_remediation_hint
                        or f"{symbol} clears the public market data environment block before reuse"
                    )
                    if source_refresh_action == "repair_public_data_environment"
                    else (
                        f"{symbol} receives a fresh crypto_route_refresh artifact before promotion"
                        if source_refresh_action != "read_current_artifact"
                        else str(row.get("done_when") or "").strip()
                    )
                ),
                "command": "",
                "clear_when": "",
                "source_as_of": source_as_of,
                "source_age_minutes": source_age_minutes,
                "source_recency": source_recency,
                "source_status": source_status,
                "source_health": source_health,
                "source_kind": "crypto_route_refresh",
                "source_artifact": source_artifact,
                "source_refresh_action": source_refresh_action,
                "promotion_ready": source_refresh_action == "read_current_artifact",
            }
        )
    return rows


def _build_brooks_review_rows(
    *,
    reference_now: dt.datetime,
    refresh_payload: dict[str, Any],
    review_queue_payload: dict[str, Any],
    capability_payload: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    source_rows = [
        dict(row)
        for row in list(review_queue_payload.get("queue") or [])
        if isinstance(row, dict)
    ]
    rows: list[dict[str, Any]] = []
    if not source_rows:
        head = dict(refresh_payload.get("head") or {})
        if head:
            source_rows = [head]
    source_as_of = payload_or_artifact_as_of(review_queue_payload) or payload_or_artifact_as_of(refresh_payload)
    source_artifact = str(review_queue_payload.get("artifact") or refresh_payload.get("artifact") or "").strip()
    source_status = str(review_queue_payload.get("status") or refresh_payload.get("status") or "").strip()
    source_age_minutes = _review_source_age_minutes(reference_now=reference_now, source_as_of=source_as_of)
    source_recency = _review_source_recency(source_age_minutes=source_age_minutes)
    source_health = _review_source_health(source_status=source_status, source_recency=source_recency)
    source_refresh_action = _review_source_refresh_action(source_health=source_health)
    capability_rows = [
        dict(row)
        for row in list((capability_payload or {}).get("capabilities") or [])
        if isinstance(row, dict)
    ]
    capability_by_symbol = {
        str(row.get("symbol") or "").strip().upper(): row
        for row in capability_rows
        if str(row.get("symbol") or "").strip()
    }

    for row in source_rows:
        symbol = str(row.get("symbol") or "").strip().upper()
        action = str(row.get("execution_action") or row.get("action") or "").strip()
        if not symbol or not action:
            continue
        row_source_as_of = source_as_of
        row_source_age_minutes = source_age_minutes
        row_source_recency = source_recency
        row_source_status = source_status
        row_source_health = source_health
        row_source_kind = "brooks_structure_review_queue"
        row_source_artifact = source_artifact
        row_source_refresh_action = source_refresh_action
        row_promotion_ready = source_refresh_action == "read_current_artifact"
        blocker_detail = str(row.get("blocker_detail") or "").strip()
        done_when = str(row.get("done_when") or "").strip()

        capability_row = capability_by_symbol.get(symbol, {})
        capability_stage = str(capability_row.get("bridge_stage") or "").strip()
        capability_blocker_code = str(capability_row.get("blocker_code") or "").strip()
        capability_blocker_detail = str(capability_row.get("blocker_detail") or "").strip()
        capability_promotable = capability_stage in {"guarded_canary", "executable"}
        if capability_row and row_promotion_ready:
            row_source_as_of = payload_or_artifact_as_of(capability_payload or {}) or row_source_as_of
            row_source_age_minutes = _review_source_age_minutes(
                reference_now=reference_now,
                source_as_of=row_source_as_of,
            )
            row_source_recency = _review_source_recency(source_age_minutes=row_source_age_minutes)
            row_source_status = str((capability_payload or {}).get("status") or "").strip()
            row_source_health = _review_source_health(
                source_status=row_source_status,
                source_recency=row_source_recency,
            )
            row_source_kind = "domestic_futures_execution_bridge_capability"
            row_source_artifact = str((capability_payload or {}).get("artifact") or "").strip()
            if not capability_promotable:
                row_promotion_ready = False
                row_source_refresh_action = ""
                blocker_parts = [
                    capability_blocker_detail or blocker_detail,
                    capability_stage,
                    capability_blocker_code,
                ]
                blocker_detail = " | ".join([part for part in blocker_parts if part])
                done_when = "automated execution bridge becomes explicit before promotion"

        state = _normalize_operator_state(
            state="",
            priority_tier=str(row.get("priority_tier") or ""),
            route_status_label="",
            plan_status=str(row.get("plan_status") or ""),
        )
        rows.append(
            {
                "area": "brooks_structure",
                "target": symbol,
                "symbol": symbol,
                "action": action,
                "reason": str(row.get("strategy_id") or "brooks_structure_review").strip(),
                "state": state,
                "priority_score": int(row.get("priority_score") or 0),
                "priority_tier": str(row.get("priority_tier") or "").strip() or state,
                "blocker_detail": (
                    (
                        blocker_detail
                        + " | "
                        + f"source freshness guard: brooks_structure_review_queue artifact is {source_status or '-'} "
                        + f"and {source_recency or '-'}"
                        + (f", age={source_age_minutes}m" if source_age_minutes is not None else "")
                    )
                    if source_refresh_action != "read_current_artifact"
                    else blocker_detail
                ),
                "done_when": (
                    f"{symbol} receives a fresh brooks_structure_review_queue artifact before promotion"
                    if source_refresh_action != "read_current_artifact"
                    else done_when
                ),
                "command": "",
                "clear_when": "",
                "source_as_of": row_source_as_of,
                "source_age_minutes": row_source_age_minutes,
                "source_recency": row_source_recency,
                "source_status": row_source_status,
                "source_health": row_source_health,
                "source_kind": row_source_kind,
                "source_artifact": row_source_artifact,
                "source_refresh_action": row_source_refresh_action,
                "promotion_ready": row_promotion_ready,
            }
        )
    return rows


def _build_operator_backlog(
    *,
    commodity_waiting_rows: list[dict[str, Any]],
    crypto_review_rows: list[dict[str, Any]],
    brooks_review_rows: list[dict[str, Any]],
    remote_live_takeover_repair_queue: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    backlog: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    def append_row(
        *,
        area: str,
        target: str,
        symbol: str,
        action: str,
        reason: str,
        state: str,
        priority_score: int,
        priority_tier: str,
        blocker_detail: str,
        done_when: str,
        sort_rank: int,
        command: str = "",
        clear_when: str = "",
        source_refresh_action: str = "",
        promotion_ready: bool | None = None,
        source_as_of: str = "",
        source_age_minutes: int | None = None,
        source_recency: str = "",
        source_status: str = "",
        source_health: str = "",
        source_kind: str = "",
        source_artifact: str = "",
    ) -> None:
        area_text = str(area or "").strip()
        symbol_text = str(symbol or "").strip().upper()
        action_text = str(action or "").strip()
        if not area_text or not symbol_text or not action_text:
            return
        dedupe_key = (area_text, symbol_text, action_text)
        if dedupe_key in seen:
            return
        seen.add(dedupe_key)
        promotion_ready_flag = (
            bool(promotion_ready)
            if promotion_ready is not None
            else not bool(str(source_refresh_action or "").strip())
        )
        effective_action_text = _review_row_effective_action(
            action=action_text,
            source_refresh_action=str(source_refresh_action or "").strip(),
            promotion_ready=promotion_ready_flag,
        )
        backlog.append(
            {
                "area": area_text,
                "target": str(target or symbol_text).strip() or symbol_text,
                "symbol": symbol_text,
                "action": effective_action_text,
                "review_action": action_text,
                "effective_action": effective_action_text,
                "reason": str(reason or "").strip(),
                "state": str(state or "").strip() or "review",
                "priority_score": int(priority_score or 0),
                "priority_tier": str(priority_tier or "").strip() or "-",
                "blocker_detail": str(blocker_detail or "").strip(),
                "done_when": str(done_when or "").strip(),
                "command": str(command or "").strip(),
                "clear_when": str(clear_when or "").strip(),
                "source_refresh_action": str(source_refresh_action or "").strip(),
                "promotion_ready": promotion_ready_flag,
                "source_as_of": str(source_as_of or "").strip(),
                "source_age_minutes": source_age_minutes,
                "source_recency": str(source_recency or "").strip(),
                "source_status": str(source_status or "").strip(),
                "source_health": str(source_health or "").strip(),
                "source_kind": str(source_kind or "").strip(),
                "source_artifact": str(source_artifact or "").strip(),
                "_sort_rank": int(sort_rank),
            }
        )

    for row_rank, row in enumerate(commodity_waiting_rows, start=1):
        append_row(
            area=str(row.get("area") or ""),
            target=str(row.get("target") or row.get("symbol") or ""),
            symbol=str(row.get("symbol") or ""),
            action=str(row.get("action") or ""),
            reason=str(row.get("reason") or ""),
            state="waiting",
            priority_score=int(row.get("priority_score") or max(1, 100 - row_rank)),
            priority_tier=str(row.get("priority_tier") or "waiting_now"),
            blocker_detail=str(row.get("blocker_detail") or ""),
            done_when=str(row.get("done_when") or ""),
            command=str(row.get("command") or ""),
            clear_when=str(row.get("clear_when") or ""),
            sort_rank=row_rank,
        )

    for offset, row in enumerate(brooks_review_rows, start=1):
        append_row(
            area=str(row.get("area") or ""),
            target=str(row.get("target") or row.get("symbol") or ""),
            symbol=str(row.get("symbol") or ""),
            action=str(row.get("action") or ""),
            reason=str(row.get("reason") or ""),
            state=str(row.get("state") or "review"),
            priority_score=int(row.get("priority_score") or 0),
            priority_tier=str(row.get("priority_tier") or "review_queue_now"),
            blocker_detail=str(row.get("blocker_detail") or ""),
            done_when=str(row.get("done_when") or ""),
            command=str(row.get("command") or ""),
            clear_when=str(row.get("clear_when") or ""),
            source_refresh_action=str(row.get("source_refresh_action") or ""),
            promotion_ready=(
                bool(row.get("promotion_ready"))
                if row.get("promotion_ready") is not None
                else _review_row_promotion_ready(row)
            ),
            source_as_of=str(row.get("source_as_of") or ""),
            source_age_minutes=row.get("source_age_minutes"),
            source_recency=str(row.get("source_recency") or ""),
            source_status=str(row.get("source_status") or ""),
            source_health=str(row.get("source_health") or ""),
            source_kind=str(row.get("source_kind") or ""),
            source_artifact=str(row.get("source_artifact") or ""),
            sort_rank=100 + offset,
        )

    for offset, row in enumerate(crypto_review_rows, start=1):
        append_row(
            area=str(row.get("area") or ""),
            target=str(row.get("target") or row.get("symbol") or ""),
            symbol=str(row.get("symbol") or ""),
            action=str(row.get("action") or ""),
            reason=str(row.get("reason") or ""),
            state=str(row.get("state") or "review"),
            priority_score=int(row.get("priority_score") or 0),
            priority_tier=str(row.get("priority_tier") or "review_queue_next"),
            blocker_detail=str(row.get("blocker_detail") or ""),
            done_when=str(row.get("done_when") or ""),
            command=str(row.get("command") or ""),
            clear_when=str(row.get("clear_when") or ""),
            source_refresh_action=str(row.get("source_refresh_action") or ""),
            promotion_ready=(
                bool(row.get("promotion_ready"))
                if row.get("promotion_ready") is not None
                else _review_row_promotion_ready(row)
            ),
            source_as_of=str(row.get("source_as_of") or ""),
            source_age_minutes=row.get("source_age_minutes"),
            source_recency=str(row.get("source_recency") or ""),
            source_status=str(row.get("source_status") or ""),
            source_health=str(row.get("source_health") or ""),
            source_kind=str(row.get("source_kind") or ""),
            source_artifact=str(row.get("source_artifact") or ""),
            sort_rank=200 + offset,
        )

    repair_queue_items = []
    queue_payload = dict(remote_live_takeover_repair_queue or {})
    for row in list(queue_payload.get("items") or []):
        if isinstance(row, dict):
            repair_queue_items.append(dict(row))
    for row in repair_queue_items:
        area = str(row.get("area") or "").strip() or "remote_live_repair"
        code = str(row.get("code") or "").strip()
        action = str(row.get("action") or "").strip() or "clear_remote_live_condition"
        if not code:
            continue
        append_row(
            area=area,
            target=f"remote_live_takeover_repair:{area}:{code}",
            symbol=code.upper().replace(":", "_"),
            action=action,
            reason=str(row.get("goal") or ""),
            state="repair",
            priority_score=int(row.get("priority_score") or 0),
            priority_tier=str(row.get("priority_tier") or "repair_queue_now"),
            blocker_detail=str(row.get("clear_when") or ""),
            done_when=str(row.get("clear_when") or queue_payload.get("done_when") or ""),
            sort_rank=900 + int(row.get("rank") or (len(backlog) + 1)),
            command=str(row.get("command") or ""),
            clear_when=str(row.get("clear_when") or ""),
        )

    backlog.sort(
        key=lambda row: (
            _operator_state_rank(str(row.get("state") or "")),
            0 if _review_row_promotion_ready(row) else 1,
            _tier_rank(str(row.get("priority_tier") or "")),
            -int(row.get("priority_score") or 0),
            int(row.get("_sort_rank") or 0),
            str(row.get("area") or ""),
            str(row.get("symbol") or ""),
        )
    )
    for idx, row in enumerate(backlog, start=1):
        row["rank"] = idx
        row.pop("_sort_rank", None)
    return backlog


def _build_operator_state_lanes(backlog: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lanes: dict[str, dict[str, Any]] = {}
    for state in ("waiting", "review", "watch", "blocked", "repair"):
        rows = [dict(row) for row in backlog if str(row.get("state") or "").strip() == state]
        head = dict(rows[0]) if rows else {}
        brief = " | ".join(
            [
                f"{int(row.get('rank') or 0)}:{str(row.get('symbol') or '-')}"
                f":{str(row.get('action') or '-')}"
                f":{int(row.get('priority_score') or 0)}"
                for row in rows[:8]
            ]
        ) or "-"
        priority_total = sum(int(row.get("priority_score") or 0) for row in rows)
        lane_status = "ready" if rows else "inactive"
        if state == "review" and rows and not any(_review_row_promotion_ready(row) for row in rows):
            lane_status = "refresh_required"
        lanes[state] = {
            "status": lane_status,
            "count": len(rows),
            "rows": rows,
            "brief": brief,
            "priority_total": priority_total,
            "head": head,
            "head_symbol": str(head.get("symbol") or ""),
            "head_action": str(head.get("action") or ""),
            "head_priority_score": int(head.get("priority_score") or 0),
            "head_priority_tier": str(head.get("priority_tier") or ""),
            "head_blocker_detail": str(head.get("blocker_detail") or ""),
            "head_done_when": str(head.get("done_when") or ""),
        }
    return lanes


def _build_operator_state_lane_heads_brief(lanes: dict[str, dict[str, Any]]) -> str:
    parts: list[str] = []
    for state in ("waiting", "review", "watch", "blocked", "repair"):
        lane = dict(lanes.get(state) or {})
        head_symbol = str(lane.get("head_symbol") or "").strip().upper() or "-"
        head_action = str(lane.get("head_action") or "").strip() or "-"
        head_priority_score = int(lane.get("head_priority_score") or 0)
        parts.append(f"{state}:{head_symbol}:{head_action}:{head_priority_score}")
    return " | ".join(parts) or "-"


def _build_operator_state_lane_priority_order_brief(lanes: dict[str, dict[str, Any]]) -> str:
    ordered = sorted(
        ("waiting", "review", "watch", "blocked", "repair"),
        key=lambda state: (
            -int((lanes.get(state) or {}).get("priority_total") or 0),
            -int((lanes.get(state) or {}).get("count") or 0),
            _operator_state_rank(state),
        ),
    )
    return " > ".join(
        [
            f"{state}@{int((lanes.get(state) or {}).get('priority_total') or 0)}:{int((lanes.get(state) or {}).get('count') or 0)}"
            for state in ordered
        ]
    ) or "-"


def _operator_repair_queue_brief(items: list[dict[str, Any]], limit: int = 4) -> str:
    parts: list[str] = []
    for idx, row in enumerate(items[:limit], start=1):
        if not isinstance(row, dict):
            continue
        parts.append(
            f"{idx}:{row.get('area') or '-'}:{row.get('target') or '-'}:{row.get('action') or '-'}"
        )
    if not parts:
        return "-"
    if len(items) <= limit:
        return " | ".join(parts)
    return " | ".join(parts) + f" | +{len(items) - limit}"


def _operator_repair_checklist_brief(items: list[dict[str, Any]], limit: int = 4) -> str:
    parts: list[str] = []
    for row in items[:limit]:
        if not isinstance(row, dict):
            continue
        parts.append(
            ":".join(
                [
                    str(row.get("rank") or "-"),
                    str(row.get("state") or "-"),
                    str(row.get("symbol") or "-"),
                    str(row.get("action") or "-"),
                ]
            )
        )
    if not parts:
        return "-"
    if len(items) <= limit:
        return " | ".join(parts)
    return " | ".join(parts) + f" | +{len(items) - limit}"


def _operator_action_queue_brief(items: list[dict[str, Any]], limit: int = 4) -> str:
    parts: list[str] = []
    for idx, row in enumerate(items[:limit], start=1):
        if not isinstance(row, dict):
            continue
        parts.append(
            f"{idx}:{row.get('area') or '-'}:{row.get('target') or '-'}:{row.get('action') or '-'}"
        )
    if not parts:
        return "-"
    if len(items) <= limit:
        return " | ".join(parts)
    return " | ".join(parts) + f" | +{len(items) - limit}"


def _operator_action_checklist_brief(items: list[dict[str, Any]], limit: int = 4) -> str:
    parts: list[str] = []
    for row in items[:limit]:
        if not isinstance(row, dict):
            continue
        parts.append(
            ":".join(
                [
                    str(row.get("rank") or "-"),
                    str(row.get("state") or "-"),
                    str(row.get("symbol") or "-"),
                    str(row.get("action") or "-"),
                ]
            )
        )
    if not parts:
        return "-"
    if len(items) <= limit:
        return " | ".join(parts)
    return " | ".join(parts) + f" | +{len(items) - limit}"


def _operator_focus_slots_brief(items: list[dict[str, Any]], limit: int = 3) -> str:
    parts: list[str] = []
    for row in items[:limit]:
        if not isinstance(row, dict):
            continue
        parts.append(
            ":".join(
                [
                    str(row.get("slot") or "-"),
                    str(row.get("state") or "-"),
                    str(row.get("symbol") or "-"),
                    str(row.get("action") or "-"),
                ]
            )
        )
    if not parts:
        return "-"
    if len(items) <= limit:
        return " | ".join(parts)
    return " | ".join(parts) + f" | +{len(items) - limit}"


def _build_operator_action_lanes(
    *,
    operator_state_lanes: dict[str, dict[str, Any]],
    review_head: dict[str, Any],
    operator_repair_queue: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, str]:
    action_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    def append_row(row: dict[str, Any] | None) -> None:
        source = dict(row or {})
        area = str(source.get("area") or "").strip()
        target = str(source.get("target") or source.get("symbol") or "").strip()
        symbol = str(source.get("symbol") or "").strip().upper()
        action = str(source.get("action") or "").strip()
        if not area or not target or not action:
            return
        dedupe_key = (area, target, action)
        if dedupe_key in seen:
            return
        seen.add(dedupe_key)
        action_rows.append(
            {
                "area": area,
                "target": target,
                "symbol": symbol or target.upper(),
                "action": action,
                "reason": str(source.get("reason") or "").strip() or "-",
                "state": _action_lane_state(
                    area=area,
                    action=action,
                    fallback_state=str(source.get("state") or "").strip(),
                ),
                "priority_score": int(source.get("priority_score") or 0),
                "priority_tier": str(source.get("priority_tier") or "").strip(),
                "blocker_detail": str(source.get("blocker_detail") or "").strip(),
                "done_when": str(source.get("done_when") or "").strip(),
                "source_kind": str(source.get("source_kind") or "").strip(),
                "source_artifact": str(source.get("source_artifact") or "").strip(),
                "source_status": str(source.get("source_status") or "").strip(),
                "source_as_of": str(source.get("source_as_of") or "").strip(),
                "source_age_minutes": source.get("source_age_minutes"),
                "source_recency": str(source.get("source_recency") or "").strip(),
                "source_health": str(source.get("source_health") or "").strip(),
                "source_refresh_action": str(source.get("source_refresh_action") or "").strip(),
            }
        )

    waiting_rows = [
        dict(row)
        for row in list((operator_state_lanes.get("waiting") or {}).get("rows") or [])
        if isinstance(row, dict)
    ]
    if waiting_rows:
        append_row(waiting_rows[0])
    if len(waiting_rows) >= 2:
        append_row(waiting_rows[1])
    append_row(review_head)
    if operator_repair_queue:
        append_row(operator_repair_queue[0])

    operator_action_queue = [
        {
            "rank": idx,
            "area": str(row.get("area") or ""),
            "target": str(row.get("target") or ""),
            "symbol": str(row.get("symbol") or ""),
            "action": str(row.get("action") or ""),
            "reason": str(row.get("reason") or ""),
            "priority_score": int(row.get("priority_score") or 0),
            "priority_tier": str(row.get("priority_tier") or ""),
            "queue_rank": idx,
            "source_kind": str(row.get("source_kind") or "").strip(),
            "source_artifact": str(row.get("source_artifact") or "").strip(),
            "source_status": str(row.get("source_status") or "").strip(),
            "source_as_of": str(row.get("source_as_of") or "").strip(),
            "source_age_minutes": row.get("source_age_minutes"),
            "source_recency": str(row.get("source_recency") or "").strip(),
            "source_health": str(row.get("source_health") or "").strip(),
            "source_refresh_action": str(row.get("source_refresh_action") or "").strip(),
        }
        for idx, row in enumerate(action_rows, start=1)
    ]
    operator_action_checklist = [
        {
            "rank": idx,
            "area": str(row.get("area") or ""),
            "target": str(row.get("target") or ""),
            "symbol": str(row.get("symbol") or ""),
            "action": str(row.get("action") or ""),
            "reason": str(row.get("reason") or ""),
            "state": str(row.get("state") or ""),
            "priority_score": int(row.get("priority_score") or 0),
            "priority_tier": str(row.get("priority_tier") or ""),
            "queue_rank": idx,
            "blocker_detail": str(row.get("blocker_detail") or ""),
            "done_when": str(row.get("done_when") or ""),
            "source_kind": str(row.get("source_kind") or "").strip(),
            "source_artifact": str(row.get("source_artifact") or "").strip(),
            "source_status": str(row.get("source_status") or "").strip(),
            "source_as_of": str(row.get("source_as_of") or "").strip(),
            "source_age_minutes": row.get("source_age_minutes"),
            "source_recency": str(row.get("source_recency") or "").strip(),
            "source_health": str(row.get("source_health") or "").strip(),
            "source_refresh_action": str(row.get("source_refresh_action") or "").strip(),
        }
        for idx, row in enumerate(action_rows, start=1)
    ]
    return (
        operator_action_queue,
        operator_action_checklist,
        _operator_action_queue_brief(operator_action_queue),
        _operator_action_checklist_brief(operator_action_checklist),
    )


def _build_operator_focus_slots(
    operator_action_checklist: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    slot_names = ("primary", "followup", "secondary")
    slots: list[dict[str, Any]] = []
    for idx, row in enumerate(operator_action_checklist[: len(slot_names)]):
        if not isinstance(row, dict):
            continue
        slots.append(
            {
                "slot": slot_names[idx],
                "area": str(row.get("area") or ""),
                "target": str(row.get("target") or ""),
                "symbol": str(row.get("symbol") or ""),
                "action": str(row.get("action") or ""),
                "reason": str(row.get("reason") or ""),
                "state": str(row.get("state") or ""),
                "priority_score": int(row.get("priority_score") or 0),
                "priority_tier": str(row.get("priority_tier") or ""),
                "queue_rank": int(row.get("queue_rank") or row.get("rank") or (idx + 1)),
                "blocker_detail": str(row.get("blocker_detail") or ""),
                "done_when": str(row.get("done_when") or ""),
                "source_kind": str(row.get("source_kind") or "").strip(),
                "source_artifact": str(row.get("source_artifact") or "").strip(),
                "source_status": str(row.get("source_status") or "").strip(),
                "source_as_of": str(row.get("source_as_of") or "").strip(),
                "source_age_minutes": row.get("source_age_minutes"),
                "source_recency": str(row.get("source_recency") or "").strip(),
                "source_health": str(row.get("source_health") or "").strip(),
                "source_refresh_action": str(row.get("source_refresh_action") or "").strip(),
            }
        )
    return slots, _operator_focus_slots_brief(slots)


def _build_review_head_lane(
    review_backlog: list[dict[str, Any]],
    review_backlog_status: str,
) -> dict[str, Any]:
    head = dict(review_backlog[0]) if review_backlog else {}
    if not head:
        return {
            "status": "inactive",
            "brief": "inactive:-",
            "head": {},
            "head_rank": 0,
            "target": "-",
            "reason": "-",
            "backlog_count": 0,
            "backlog_brief": "-",
            "blocker_detail": "No cross-market review head is currently active.",
            "done_when": "cross-market operator state produces at least one review backlog item",
        }
    backlog_tail = review_backlog[1:]
    head_area = str(head.get("area") or "").strip() or "-"
    head_symbol = str(head.get("symbol") or "").strip().upper() or "-"
    head_action = str(head.get("action") or "").strip() or "-"
    head_target = str(head.get("target") or head_symbol).strip() or head_symbol
    head_reason = str(head.get("reason") or "").strip() or "cross_market_review_head"
    head_rank = int(head.get("rank") or 1)
    head_priority_score = int(head.get("priority_score") or 0)
    head_promotion_ready = _review_row_promotion_ready(head)
    head_status = (
        "review"
        if head_promotion_ready
        else "refresh_required"
    )
    backlog_brief = " | ".join(
        [
            f"{int(row.get('rank') or 0)}:{str(row.get('area') or '-')}"
            f":{str(row.get('symbol') or '-')}"
            f":{str(row.get('priority_tier') or '-')}"
            f":{int(row.get('priority_score') or 0)}"
            for row in backlog_tail[:8]
        ]
    ) or "-"
    return {
        "status": head_status,
        "brief": f"{head_status}:{head_area}:{head_symbol}:{head_action}:{head_priority_score}",
        "head": head,
        "head_rank": head_rank,
        "target": head_target,
        "reason": head_reason,
        "backlog_count": len(backlog_tail),
        "backlog_brief": backlog_brief,
        "blocker_detail": str(head.get("blocker_detail") or "").strip()
        or "Cross-market review head is active and awaiting operator review.",
        "done_when": str(head.get("done_when") or "").strip()
        or "cross-market review head is resolved, promoted, or invalidated",
    }


def _build_operator_head_lane(
    operator_backlog: list[dict[str, Any]],
    operator_backlog_status: str,
) -> dict[str, Any]:
    head = dict(operator_backlog[0]) if operator_backlog else {}
    if not head:
        return {
            "status": "inactive",
            "brief": "inactive:-",
            "head": {},
            "head_rank": 0,
            "backlog_count": 0,
            "backlog_brief": "-",
            "blocker_detail": "No cross-market operator head is currently active.",
            "done_when": "cross-market operator state produces at least one operator backlog item",
            "state": "inactive",
            "priority_tier": "",
        }
    backlog_tail = operator_backlog[1:]
    head_area = str(head.get("area") or "").strip() or "-"
    head_symbol = str(head.get("symbol") or "").strip().upper() or "-"
    head_action = str(head.get("action") or "").strip() or "-"
    head_state = str(head.get("state") or operator_backlog_status or "").strip() or "review"
    head_rank = int(head.get("rank") or 1)
    head_priority_score = int(head.get("priority_score") or 0)
    head_priority_tier = str(head.get("priority_tier") or "").strip() or "-"
    backlog_brief = " | ".join(
        [
            f"{int(row.get('rank') or 0)}:{str(row.get('state') or '-')}"
            f":{str(row.get('area') or '-')}"
            f":{str(row.get('symbol') or '-')}"
            f":{str(row.get('action') or '-')}"
            f":{int(row.get('priority_score') or 0)}"
            for row in backlog_tail[:8]
        ]
    ) or "-"
    return {
        "status": head_state,
        "brief": f"{head_state}:{head_area}:{head_symbol}:{head_action}:{head_priority_score}",
        "head": head,
        "head_rank": head_rank,
        "backlog_count": len(backlog_tail),
        "backlog_brief": backlog_brief,
        "blocker_detail": str(head.get("blocker_detail") or "").strip()
        or "Cross-market operator head is active and awaiting operator follow-through.",
        "done_when": str(head.get("done_when") or "").strip()
        or "cross-market operator head is resolved, promoted, or invalidated",
        "state": head_state,
        "priority_tier": head_priority_tier,
    }


def _build_operator_repair_head_lane(
    operator_repair_queue: list[dict[str, Any]],
    operator_repair_checklist: list[dict[str, Any]],
) -> dict[str, Any]:
    head = dict(operator_repair_queue[0]) if operator_repair_queue else {}
    checklist_head = dict(operator_repair_checklist[0]) if operator_repair_checklist else {}
    if not head:
        return {
            "status": "inactive",
            "brief": "inactive:-",
            "head": {},
            "head_rank": 0,
            "backlog_count": 0,
            "backlog_brief": "-",
            "blocker_detail": "No remote-live repair head is currently active.",
            "done_when": "remote live repair queue activates when blocker sources are available",
        }
    backlog_rows = operator_repair_queue
    head_area = str(head.get("area") or "").strip() or "-"
    head_code = str(head.get("target") or head.get("symbol") or "").strip() or "-"
    head_action = str(head.get("action") or "").strip() or "-"
    head_priority_score = int(head.get("priority_score") or 0)
    head_priority_tier = str(head.get("priority_tier") or "").strip() or "-"
    head_rank = int(head.get("rank") or 1)
    backlog_brief = " | ".join(
        [
            f"{int(row.get('rank') or 0)}:{str(row.get('area') or '-')}"
            f":{str(row.get('target') or row.get('symbol') or '-')}"
            f":{str(row.get('action') or '-')}"
            f":{int(row.get('priority_score') or 0)}"
            for row in backlog_rows[:8]
        ]
    ) or "-"
    priority_total = sum(int(row.get("priority_score") or 0) for row in backlog_rows)
    blocker_detail = str(checklist_head.get("blocker_detail") or "").strip() or str(
        head.get("clear_when") or ""
    ).strip()
    done_when = str(checklist_head.get("done_when") or "").strip() or str(head.get("clear_when") or "").strip()
    return {
        "status": "ready",
        "brief": f"ready:{head_area}:{head_code}:{head_priority_score}",
        "head": head,
        "head_rank": head_rank,
        "backlog_count": len(backlog_rows),
        "backlog_brief": backlog_brief,
        "priority_total": priority_total,
        "blocker_detail": blocker_detail,
        "done_when": done_when or "remote live repair head clears or rotates to the next queued blocker",
        "command": str(head.get("command") or "").strip(),
        "clear_when": str(head.get("clear_when") or "").strip(),
        "priority_tier": head_priority_tier,
    }


def _build_operator_repair_queue(
    queue_payload: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, str]:
    payload = dict(queue_payload or {})
    raw_items = [dict(row) for row in list(payload.get("items") or []) if isinstance(row, dict)]
    operator_repair_queue: list[dict[str, Any]] = []
    for idx, row in enumerate(raw_items, start=1):
        area = str(row.get("area") or "").strip()
        code = str(row.get("code") or "").strip()
        action = str(row.get("action") or "").strip()
        if not area or not code or not action:
            continue
        operator_repair_queue.append(
            {
                "rank": idx,
                "area": area,
                "target": code,
                "symbol": code.upper(),
                "action": action,
                "reason": str(row.get("clear_when") or row.get("goal") or payload.get("brief") or "").strip(),
                "priority_score": int(row.get("priority_score") or 0),
                "priority_tier": str(row.get("priority_tier") or "").strip(),
                "command": str(row.get("command") or "").strip(),
                "clear_when": str(row.get("clear_when") or "").strip(),
                "goal": str(row.get("goal") or "").strip(),
                "actions": [
                    str(item).strip() for item in row.get("actions", []) if str(item).strip()
                ]
                if isinstance(row.get("actions"), list)
                else [],
            }
        )
    operator_repair_checklist = [
        {
            **row,
            "rank": idx,
            "state": "repair",
            "blocker_detail": str(row.get("clear_when") or "").strip(),
            "done_when": str(row.get("clear_when") or payload.get("done_when") or "").strip(),
        }
        for idx, row in enumerate(operator_repair_queue, start=1)
    ]
    return (
        operator_repair_queue,
        operator_repair_checklist,
        _operator_repair_queue_brief(operator_repair_queue),
        _operator_repair_checklist_brief(operator_repair_checklist),
    )


def _derive_remote_live_operator_alignment(
    *,
    operator_head: dict[str, Any],
    live_gate_blocker_payload: dict[str, Any],
) -> dict[str, Any]:
    head = dict(operator_head or {})
    head_area = str(head.get("area") or "").strip()
    head_symbol = str(head.get("symbol") or "").strip().upper()
    head_action = str(head.get("action") or "").strip()
    head_state = str(head.get("state") or "").strip()
    head_priority_score = int(head.get("priority_score") or 0)
    head_priority_tier = str(head.get("priority_tier") or "").strip()
    diagnosis = dict(live_gate_blocker_payload.get("remote_live_diagnosis") or {})
    remote_status = str(diagnosis.get("status") or "").strip()
    remote_brief = str(diagnosis.get("brief") or "").strip()
    remote_market = _remote_live_scope_market(live_gate_blocker_payload)

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
            "remote_brief": remote_brief,
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
        brief = f"{status}:{head_area}:{head_symbol}:{remote_market or 'remote_scope_unknown'}"
        blocker_detail = (
            f"Current local operator head is {head_symbol} "
            f"({head_area}/{head_action}, state={head_state or '-'}, priority={head_priority_score}), "
            f"which is outside remote-live executable scope "
            f"({remote_market or 'remote_scope_unknown'}; current takeover only evaluates crypto_route heads)."
        )
        done_when = (
            "cross-market operator head moves into remote-live executable scope "
            "(for example crypto_route) while remote live diagnostics remain fresh"
        )
        blocking = False
    elif remote_status in {
        "profitability_confirmed_but_auto_live_blocked",
        "auto_live_blocked_without_profitability_confirmation",
    }:
        status = "local_operator_active_remote_live_blocked"
        brief = f"{status}:{head_area}:{head_symbol}:{remote_status or 'review_required'}"
        blocker_detail = (
            f"Current local operator head is {head_symbol} "
            f"({head_area}/{head_action}, state={head_state or '-'}, priority={head_priority_score}), "
            f"while remote automated live remains {remote_status}."
        )
        done_when = (
            f"{head_symbol} local operator head progresses and remote auto-live blockers are cleared"
        )
        blocking = True
    elif remote_status == "formal_live_possible":
        status = "local_operator_active_remote_live_ready_review"
        brief = f"{status}:{head_area}:{head_symbol}:{remote_status or 'ready'}"
        blocker_detail = (
            f"Remote auto-live is structurally ready, but the current local operator head remains "
            f"{head_symbol} ({head_area}/{head_action}, state={head_state or '-'}) and still requires operator follow-through."
        )
        done_when = f"{head_symbol} local operator head resolves or is promoted into a live-eligible action"
        blocking = False
    else:
        status = "local_operator_active_remote_live_review_required"
        brief = f"{status}:{head_area}:{head_symbol}:{remote_status or 'review_required'}"
        blocker_detail = (
            f"Current local operator head is {head_symbol} "
            f"({head_area}/{head_action}, state={head_state or '-'}), and remote live still needs manual review "
            f"({remote_status or 'review_required'})."
        )
        done_when = f"{head_symbol} local operator head is reviewed alongside refreshed remote live diagnostics"
        blocking = False

    return {
        "status": status,
        "brief": brief,
        "blocking": blocking,
        "head_area": head_area,
        "head_symbol": head_symbol,
        "head_action": head_action,
        "head_state": head_state,
        "head_priority_score": head_priority_score,
        "head_priority_tier": head_priority_tier,
        "remote_status": remote_status,
        "remote_brief": remote_brief,
        "remote_market": remote_market,
        "eligible_for_remote_live": _is_remote_live_executable_head(
            area=head_area,
            action=head_action,
            remote_market=remote_market,
        ),
        "blocker_detail": blocker_detail,
        "done_when": done_when,
    }


def _derive_remote_live_takeover_gate(
    alignment: dict[str, Any] | None,
) -> dict[str, Any]:
    row = dict(alignment or {})
    alignment_status = str(row.get("status") or "").strip()
    alignment_brief = str(row.get("brief") or "").strip()
    head_area = str(row.get("head_area") or "").strip()
    head_symbol = str(row.get("head_symbol") or "").strip().upper()
    head_action = str(row.get("head_action") or "").strip()
    remote_status = str(row.get("remote_status") or "").strip()
    blocker_detail = str(row.get("blocker_detail") or "").strip()
    done_when = str(row.get("done_when") or "").strip()
    blocking = bool(row.get("blocking", False))
    remote_market = str(row.get("remote_market") or "").strip()

    if not (head_area or head_symbol or head_action):
        return {
            "status": "inactive",
            "brief": "inactive:-",
            "blocking": False,
            "alignment_status": alignment_status,
            "head_area": "",
            "head_symbol": "",
            "head_action": "",
            "remote_status": remote_status,
            "remote_market": remote_market,
            "blocker_detail": "No cross-market operator head is currently active for remote live takeover review.",
            "done_when": "cross-market operator state produces an operator head",
        }

    if alignment_status == "local_operator_head_outside_remote_live_scope":
        status = "current_head_outside_remote_live_scope"
        brief = f"{status}:{head_area}:{head_symbol}:{remote_market or remote_status or 'scope_unknown'}"
    elif blocking:
        status = "blocked_by_remote_live_gate"
        brief = f"{status}:{head_area}:{head_symbol}:{remote_status or alignment_status or 'blocked'}"
    elif alignment_status == "local_operator_active_remote_live_ready_review":
        status = "ready_for_remote_live_review"
        brief = f"{status}:{head_area}:{head_symbol}:{remote_status or 'ready'}"
    else:
        status = "manual_review_required_before_remote_live"
        brief = f"{status}:{head_area}:{head_symbol}:{remote_status or alignment_status or 'review_required'}"

    return {
        "status": status,
        "brief": brief,
        "blocking": blocking,
        "alignment_status": alignment_status,
        "head_area": head_area,
        "head_symbol": head_symbol,
        "head_action": head_action,
        "remote_status": remote_status,
        "remote_market": remote_market,
        "blocker_detail": blocker_detail,
        "done_when": done_when,
    }


def _derive_remote_live_takeover_clearing(
    live_gate_blocker_payload: dict[str, Any],
) -> dict[str, Any]:
    source = dict(live_gate_blocker_payload.get("remote_live_takeover_clearing") or {})
    source_freshness = dict(live_gate_blocker_payload.get("source_freshness") or {})
    status = str(source.get("status") or "").strip()
    brief = str(source.get("brief") or "").strip()
    blocker_detail = str(source.get("blocker_detail") or "").strip()
    done_when = str(source.get("done_when") or "").strip()
    ops_brief = str(source.get("ops_live_gate_brief") or "").strip()
    risk_brief = str(source.get("risk_guard_brief") or "").strip()
    freshness_brief = str(source_freshness.get("brief") or "").strip()
    if not (status or brief or blocker_detail or done_when):
        return {
            "status": "inactive",
            "brief": "inactive:-",
            "blocker_detail": "Remote live takeover clearing conditions are not available from the current hot brief.",
            "done_when": "refresh live gate blocker report and hot brief sources",
            "ops_live_gate_brief": "",
            "risk_guard_brief": "",
            "source_freshness_brief": freshness_brief,
        }
    return {
        "status": status or "review_required",
        "brief": brief or f"{status or 'review_required'}:-",
        "blocker_detail": blocker_detail,
        "done_when": done_when,
        "ops_live_gate_brief": ops_brief,
        "risk_guard_brief": risk_brief,
        "source_freshness_brief": freshness_brief,
    }


def _derive_remote_live_takeover_slot_anomaly_breakdown(
    live_gate_blocker_payload: dict[str, Any],
) -> dict[str, Any]:
    row = live_gate_blocker_payload.get("slot_anomaly_breakdown")
    if not isinstance(row, dict):
        return {
            "status": "inactive",
            "brief": "inactive:-",
            "artifact": "",
            "repair_focus": "",
        }
    status = str(row.get("status") or "").strip()
    brief = str(row.get("brief") or "").strip()
    return {
        "status": status or "review_required",
        "brief": brief or f"{status or 'review_required'}:-",
        "artifact": str(row.get("artifact") or "").strip(),
        "repair_focus": str(row.get("repair_focus") or "").strip(),
    }


def _derive_remote_live_takeover_repair_queue(
    live_gate_blocker_payload: dict[str, Any],
) -> dict[str, Any]:
    row = live_gate_blocker_payload.get("remote_live_takeover_repair_queue")
    if isinstance(row, dict):
        items = [dict(item) for item in list(row.get("items") or []) if isinstance(item, dict)]
        return {
            "status": str(row.get("status") or ""),
            "brief": str(row.get("brief") or ""),
            "count": int(row.get("count") or 0),
            "head_area": str(row.get("head_area") or ""),
            "head_code": str(row.get("head_code") or ""),
            "head_action": str(row.get("head_action") or ""),
            "head_priority_score": int(row.get("head_priority_score") or 0),
            "head_priority_tier": str(row.get("head_priority_tier") or ""),
            "head_command": str(row.get("head_command") or ""),
            "head_clear_when": str(row.get("head_clear_when") or ""),
            "queue_brief": str(row.get("queue_brief") or ""),
            "items": items,
            "done_when": str(row.get("done_when") or ""),
        }

    status = ""
    brief = ""
    count = 0
    head_area = ""
    head_code = ""
    head_action = ""
    head_priority_score = 0
    head_priority_tier = ""
    head_command = ""
    head_clear_when = ""
    done_when = ""
    items: list[dict[str, Any]] = []
    if head_code:
        items.append(
            {
                "rank": 1,
                "area": head_area or "remote_live_repair",
                "code": head_code,
                "action": head_action or "clear_remote_live_condition",
                "priority_score": head_priority_score,
                "priority_tier": head_priority_tier or "repair_queue_now",
                "command": head_command,
                "clear_when": head_clear_when,
            }
        )
    queue_brief = (
        f"1:{head_area or '-'}:{head_code or '-'}:{head_priority_score}" if items else "-"
    )
    if not (status or brief or items):
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
            "done_when": "remote live repair queue activates when blocker sources are available",
        }
    return {
        "status": status or "ready",
        "brief": brief or f"ready:{head_area or '-'}:{head_code or '-'}:{head_priority_score}",
        "count": count or len(items),
        "head_area": head_area,
        "head_code": head_code,
        "head_action": head_action,
        "head_priority_score": head_priority_score,
        "head_priority_tier": head_priority_tier,
        "head_command": head_command,
        "head_clear_when": head_clear_when,
        "queue_brief": queue_brief,
        "items": items,
        "done_when": done_when,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Cross-Market Operator State",
        "",
        f"- as_of: `{payload.get('as_of') or ''}`",
        f"- downstream_refresh_mode: `{payload.get('downstream_refresh_mode') or '-'}`",
        f"- downstream_refresh_reuse_brief: `{payload.get('downstream_refresh_reuse_brief') or '-'}`",
        f"- operator_stack_brief: `{payload.get('operator_stack_brief') or '-'}`",
        f"- operator_backlog_status: `{payload.get('operator_backlog_status') or '-'}`",
        f"- operator_backlog_count: `{int(payload.get('operator_backlog_count') or 0)}`",
        f"- operator_backlog_brief: `{payload.get('operator_backlog_brief') or '-'}`",
        f"- operator_backlog_state_brief: `{payload.get('operator_backlog_state_brief') or '-'}`",
        f"- operator_backlog_priority_totals_brief: `{payload.get('operator_backlog_priority_totals_brief') or '-'}`",
        f"- operator_state_lane_heads_brief: `{payload.get('operator_state_lane_heads_brief') or '-'}`",
        f"- operator_state_lane_priority_order_brief: `{payload.get('operator_state_lane_priority_order_brief') or '-'}`",
        f"- remote_live_operator_alignment_brief: `{payload.get('remote_live_operator_alignment_brief') or '-'}`",
        f"- remote_live_takeover_gate_brief: `{payload.get('remote_live_takeover_gate_brief') or '-'}`",
        f"- remote_live_takeover_clearing_brief: `{payload.get('remote_live_takeover_clearing_brief') or '-'}`",
        f"- remote_live_takeover_slot_anomaly_breakdown_brief: `{payload.get('remote_live_takeover_slot_anomaly_breakdown_brief') or '-'}`",
        f"- remote_live_takeover_repair_queue_brief: `{payload.get('remote_live_takeover_repair_queue_brief') or '-'}`",
        f"- operator_repair_queue_brief: `{payload.get('operator_repair_queue_brief') or '-'}`",
        f"- operator_repair_checklist_brief: `{payload.get('operator_repair_checklist_brief') or '-'}`",
        "",
        "## Operator Head",
        f"- `{payload.get('operator_head_area') or '-'}` `{payload.get('operator_head_symbol') or '-'}` action=`{payload.get('operator_head_action') or '-'}`",
        f"- state=`{payload.get('operator_head_state') or '-'}` priority=`{payload.get('operator_head_priority_score') or 0}`/`{payload.get('operator_head_priority_tier') or '-'}`",
        f"- blocker: {payload.get('operator_head_blocker_detail') or '-'}",
        f"- done_when: {payload.get('operator_head_done_when') or '-'}",
        f"- review_backlog_status: `{payload.get('review_backlog_status') or '-'}`",
        f"- review_backlog_count: `{int(payload.get('review_backlog_count') or 0)}`",
        f"- review_backlog_brief: `{payload.get('review_backlog_brief') or '-'}`",
        "",
        "## Sources",
        f"- brooks_refresh: `{payload.get('brooks_refresh_artifact') or '-'}`",
        f"- commodity_refresh: `{payload.get('commodity_refresh_artifact') or '-'}`",
        f"- hot_brief: `{payload.get('hot_brief_artifact') or '-'}`",
        "",
        "## Downstream Refresh Audit",
        f"- mode: `{payload.get('downstream_refresh_mode') or '-'}`",
        f"- reuse: `{payload.get('downstream_refresh_reuse_brief') or '-'}`",
        "",
        "## Remote Live Operator Alignment",
        f"- status=`{payload.get('remote_live_operator_alignment_status') or '-'}` brief=`{payload.get('remote_live_operator_alignment_brief') or '-'}`",
        f"- head=`{payload.get('remote_live_operator_alignment_head_area') or '-'} | {payload.get('remote_live_operator_alignment_head_symbol') or '-'} | {payload.get('remote_live_operator_alignment_head_action') or '-'} | state={payload.get('remote_live_operator_alignment_head_state') or '-'} | priority={payload.get('remote_live_operator_alignment_head_priority_score') or 0}/{payload.get('remote_live_operator_alignment_head_priority_tier') or '-'}`",
        f"- blocker: `{payload.get('remote_live_operator_alignment_blocker_detail') or '-'}`",
        f"- done_when: `{payload.get('remote_live_operator_alignment_done_when') or '-'}`",
        "",
        "## Remote Live Takeover Gate",
        f"- status=`{payload.get('remote_live_takeover_gate_status') or '-'}` brief=`{payload.get('remote_live_takeover_gate_brief') or '-'}`",
        f"- head=`{payload.get('remote_live_takeover_gate_head_area') or '-'} | {payload.get('remote_live_takeover_gate_head_symbol') or '-'} | {payload.get('remote_live_takeover_gate_head_action') or '-'} | remote={payload.get('remote_live_takeover_gate_remote_status') or '-'} | alignment={payload.get('remote_live_takeover_gate_alignment_status') or '-'}`",
        f"- blocker: `{payload.get('remote_live_takeover_gate_blocker_detail') or '-'}`",
        f"- done_when: `{payload.get('remote_live_takeover_gate_done_when') or '-'}`",
        "",
        "## Remote Live Takeover Clearing",
        f"- status=`{payload.get('remote_live_takeover_clearing_status') or '-'}` brief=`{payload.get('remote_live_takeover_clearing_brief') or '-'}`",
        f"- gate_details=`ops_live_gate:{payload.get('remote_live_takeover_clearing_ops_live_gate_brief') or '-'} | risk_guard:{payload.get('remote_live_takeover_clearing_risk_guard_brief') or '-'}`",
        f"- blocker: `{payload.get('remote_live_takeover_clearing_blocker_detail') or '-'}`",
        f"- done_when: `{payload.get('remote_live_takeover_clearing_done_when') or '-'}`",
        "",
        "## Remote Live Takeover Repair Queue",
        f"- status=`{payload.get('remote_live_takeover_repair_queue_status') or '-'}` brief=`{payload.get('remote_live_takeover_repair_queue_brief') or '-'}` count=`{int(payload.get('remote_live_takeover_repair_queue_count') or 0)}`",
        f"- head=`{payload.get('remote_live_takeover_repair_queue_head_area') or '-'} | {payload.get('remote_live_takeover_repair_queue_head_code') or '-'} | {payload.get('remote_live_takeover_repair_queue_head_action') or '-'} | priority={payload.get('remote_live_takeover_repair_queue_head_priority_score') or 0}/{payload.get('remote_live_takeover_repair_queue_head_priority_tier') or '-'}`",
        f"- head_command: `{payload.get('remote_live_takeover_repair_queue_head_command') or '-'}`",
        f"- head_clear_when: `{payload.get('remote_live_takeover_repair_queue_head_clear_when') or '-'}`",
        f"- queue: `{payload.get('remote_live_takeover_repair_queue_queue_brief') or '-'}`",
        f"- done_when: `{payload.get('remote_live_takeover_repair_queue_done_when') or '-'}`",
        "",
        "## Operator Repair Queue",
        f"- queue=`{int(payload.get('operator_repair_queue_count') or 0)} | {payload.get('operator_repair_queue_brief') or '-'}`",
        f"- checklist=`{payload.get('operator_repair_checklist_brief') or '-'}`",
        "",
        "## Operator Focus Slots",
        f"- slots=`{payload.get('operator_focus_slots_brief') or '-'}`",
        "",
        "## Operator Backlog",
    ]
    for row in payload.get("operator_backlog", [])[:12]:
        lines.append(
            f"- `{row.get('rank')}` `{row.get('state')}` `{row.get('area')}` `{row.get('symbol')}` action=`{row.get('action')}` priority=`{row.get('priority_score')}`/`{row.get('priority_tier')}`"
        )
        lines.append(f"  - blocker: {row.get('blocker_detail') or '-'}")
        if str(row.get("command") or "").strip():
            lines.append(f"  - command: {row.get('command')}")
    lines.extend(
        [
            "",
            "## Operator State Lanes",
        ]
    )
    for state in ("waiting", "review", "watch", "blocked"):
        lines.append(
            f"- `{state}` status=`{payload.get(f'operator_{state}_lane_status') or '-'}` count=`{int(payload.get(f'operator_{state}_lane_count') or 0)}` priority_total=`{int(payload.get(f'operator_{state}_lane_priority_total') or 0)}` brief=`{payload.get(f'operator_{state}_lane_brief') or '-'}`"
        )
        lines.append(
            f"  - head: {(payload.get(f'operator_{state}_lane_head_symbol') or '-')}/{(payload.get(f'operator_{state}_lane_head_action') or '-')} priority={(payload.get(f'operator_{state}_lane_head_priority_score') or 0)}/{(payload.get(f'operator_{state}_lane_head_priority_tier') or '-')}"
        )
    state = "repair"
    lines.append(
        f"- `{state}` status=`{payload.get(f'operator_{state}_lane_status') or '-'}` count=`{int(payload.get(f'operator_{state}_lane_count') or 0)}` priority_total=`{int(payload.get(f'operator_{state}_lane_priority_total') or 0)}` brief=`{payload.get(f'operator_{state}_lane_brief') or '-'}`"
    )
    lines.append(
        f"  - head: {(payload.get(f'operator_{state}_lane_head_symbol') or '-')}/{(payload.get(f'operator_{state}_lane_head_action') or '-')} priority={(payload.get(f'operator_{state}_lane_head_priority_score') or 0)}/{(payload.get(f'operator_{state}_lane_head_priority_tier') or '-')}"
    )
    lines.extend(
        [
            "",
            "## Review Backlog",
        ]
    )
    for row in payload.get("review_backlog", [])[:10]:
        lines.append(
            f"- `{row.get('rank')}` `{row.get('area')}` `{row.get('symbol')}` action=`{row.get('action')}` status=`{row.get('status')}` priority=`{row.get('priority_score')}`/{row.get('priority_tier')}`"
        )
        lines.append(f"  - blocker: {row.get('blocker_detail') or '-'}")
    head = dict(payload.get("review_head") or {})
    if head:
        lines.extend(
            [
                "",
                "## Review Head",
                f"- `{head.get('area')}` `{head.get('symbol')}` action=`{head.get('action')}`",
                f"- status=`{head.get('status')}` priority=`{head.get('priority_score')}`/{head.get('priority_tier')}`",
                f"- blocker: {head.get('blocker_detail') or '-'}",
                f"- done_when: {head.get('done_when') or '-'}",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refresh cross-market operator state from Brooks and commodity/hot refresh chains.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--context-path", default=str(DEFAULT_CONTEXT_PATH))
    parser.add_argument("--min-bars", type=int, default=120)
    parser.add_argument("--asset-classes", default="")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--recent-signal-age-bars", type=int, default=3)
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--now", default="")
    parser.add_argument(
        "--skip-downstream-refresh",
        action="store_true",
        help="Reuse the latest Brooks and commodity refresh artifacts instead of rerunning the downstream chains.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = Path(args.output_root).expanduser().resolve()
    review_dir = Path(args.review_dir).expanduser().resolve()
    context_path = Path(args.context_path).expanduser().resolve()
    runtime_now = derive_runtime_now(review_dir, args.now)
    downstream_refresh_mode = "skip_downstream_refresh" if bool(args.skip_downstream_refresh) else "full_refresh"

    brooks_now = step_now(runtime_now, 0)
    commodity_now = step_now(runtime_now, 1)
    if bool(args.skip_downstream_refresh):
        brooks_refresh = load_reused_refresh_payload(
            review_dir=review_dir,
            suffix="brooks_structure_refresh",
            reference_now=runtime_now,
            step_name="refresh_brooks_structure_state",
        )
        commodity_refresh = load_reused_refresh_payload(
            review_dir=review_dir,
            suffix="commodity_paper_execution_refresh",
            reference_now=runtime_now,
            step_name="refresh_commodity_paper_execution_state",
        )
    else:
        brooks_refresh = run_json_step(
            step_name="refresh_brooks_structure_state",
            cmd=[
                "python3",
                str(script_path("refresh_brooks_structure_state.py")),
                "--output-root",
                str(output_root),
                "--review-dir",
                str(review_dir),
                "--min-bars",
                str(int(args.min_bars)),
                "--asset-classes",
                str(args.asset_classes or ""),
                "--symbols",
                str(args.symbols or ""),
                "--recent-signal-age-bars",
                str(int(args.recent_signal_age_bars)),
                "--now",
                fmt_utc(brooks_now),
            ],
        )

        commodity_refresh = run_json_step(
            step_name="refresh_commodity_paper_execution_state",
            cmd=[
                "python3",
                str(script_path("refresh_commodity_paper_execution_state.py")),
                "--output-root",
                str(output_root),
                "--review-dir",
                str(review_dir),
                "--context-path",
                str(context_path),
                "--now",
                fmt_utc(commodity_now),
            ],
        )

    hot_brief_path = resolve_hot_brief_source(
        review_dir=review_dir,
        commodity_refresh=commodity_refresh,
        reference_now=runtime_now,
    )
    commodity_review_path = resolve_embedded_artifact_source(
        payload=commodity_refresh,
        key="review_artifact",
        review_dir=review_dir,
        suffix="commodity_paper_execution_review",
        reference_now=runtime_now,
    )
    commodity_gap_path = latest_review_json_artifact(review_dir, "commodity_paper_execution_gap_report", runtime_now)
    crypto_route_refresh_path = latest_review_json_artifact(review_dir, "crypto_route_refresh", runtime_now)
    brooks_review_queue_path = resolve_embedded_artifact_source(
        payload=brooks_refresh,
        key="review_queue_artifact",
        review_dir=review_dir,
        suffix="brooks_structure_review_queue",
        reference_now=runtime_now,
    )
    brooks_domestic_futures_bridge_capability_path = resolve_embedded_artifact_source(
        payload=brooks_refresh,
        key="domestic_futures_execution_bridge_capability_artifact",
        review_dir=review_dir,
        suffix="domestic_futures_execution_bridge_capability",
        reference_now=runtime_now,
    )
    live_gate_blocker_path = latest_review_json_artifact(review_dir, "live_gate_blocker_report", runtime_now)

    commodity_review_payload = load_optional_payload(commodity_review_path)
    commodity_gap_payload = load_optional_payload(commodity_gap_path)
    crypto_route_refresh_payload = load_optional_payload(crypto_route_refresh_path)
    brooks_review_queue_payload = load_optional_payload(brooks_review_queue_path)
    brooks_domestic_futures_bridge_capability_payload = load_optional_payload(
        brooks_domestic_futures_bridge_capability_path
    )
    live_gate_blocker_payload = load_optional_payload(live_gate_blocker_path)

    reused_downstream_count = 2 if bool(args.skip_downstream_refresh) else 0
    downstream_refresh_reuse_brief = (
        f"reused_downstream_inputs:{downstream_refresh_mode}:{reused_downstream_count}/2"
        if bool(args.skip_downstream_refresh)
        else "fresh_downstream_refresh:full_refresh:0/2"
    )

    commodity_waiting_rows = _build_commodity_waiting_backlog(
        review_payload=commodity_review_payload,
        gap_payload=commodity_gap_payload,
    )
    crypto_review_rows = _build_crypto_review_rows(
        reference_now=runtime_now,
        refresh_payload=crypto_route_refresh_payload,
    )
    brooks_review_rows = _build_brooks_review_rows(
        reference_now=runtime_now,
        refresh_payload=brooks_refresh,
        review_queue_payload=brooks_review_queue_payload,
        capability_payload=brooks_domestic_futures_bridge_capability_payload,
    )

    remote_live_takeover_repair_queue = _derive_remote_live_takeover_repair_queue(live_gate_blocker_payload)
    remote_live_takeover_slot_anomaly_breakdown = _derive_remote_live_takeover_slot_anomaly_breakdown(
        live_gate_blocker_payload
    )
    (
        operator_repair_queue,
        operator_repair_checklist,
        operator_repair_queue_brief,
        operator_repair_checklist_brief,
    ) = _build_operator_repair_queue(remote_live_takeover_repair_queue)
    operator_backlog = _build_operator_backlog(
        commodity_waiting_rows=commodity_waiting_rows,
        crypto_review_rows=crypto_review_rows,
        brooks_review_rows=brooks_review_rows,
        remote_live_takeover_repair_queue=remote_live_takeover_repair_queue,
    )
    operator_head = dict(operator_backlog[0]) if operator_backlog else {}
    operator_backlog_brief = " | ".join(
        [
            f"{int(row.get('rank') or 0)}:{str(row.get('state') or '-')}"
            f":{str(row.get('area') or '-')}"
            f":{str(row.get('symbol') or '-')}"
            f":{str(row.get('action') or '-')}"
            f":{int(row.get('priority_score') or 0)}"
            for row in operator_backlog[:10]
        ]
    ) or "-"
    operator_backlog_status = "ready" if operator_backlog else "inactive"
    operator_backlog_state_counts = {
        state: sum(1 for row in operator_backlog if str(row.get("state") or "").strip() == state)
        for state in ("waiting", "review", "watch", "blocked", "repair")
    }
    operator_backlog_priority_totals = {
        state: sum(
            int(row.get("priority_score") or 0)
            for row in operator_backlog
            if str(row.get("state") or "").strip() == state
        )
        for state in ("waiting", "review", "watch", "blocked", "repair")
    }
    operator_backlog_state_brief = " | ".join(
        [
            f"{state}={int(operator_backlog_state_counts.get(state) or 0)}"
            for state in ("waiting", "review", "watch", "blocked", "repair")
        ]
    )
    operator_backlog_priority_totals_brief = " | ".join(
        [
            f"{state}={int(operator_backlog_priority_totals.get(state) or 0)}"
            for state in ("waiting", "review", "watch", "blocked", "repair")
        ]
    )
    operator_state_lanes = _build_operator_state_lanes(operator_backlog)
    operator_state_lane_heads_brief = _build_operator_state_lane_heads_brief(operator_state_lanes)
    operator_state_lane_priority_order_brief = _build_operator_state_lane_priority_order_brief(operator_state_lanes)
    remote_live_operator_alignment = _derive_remote_live_operator_alignment(
        operator_head=operator_head,
        live_gate_blocker_payload=live_gate_blocker_payload,
    )
    remote_live_takeover_gate = _derive_remote_live_takeover_gate(remote_live_operator_alignment)
    remote_live_takeover_clearing = _derive_remote_live_takeover_clearing(live_gate_blocker_payload)

    review_backlog = _build_review_backlog_from_rows(
        [
            *[dict(row) for row in brooks_review_rows if isinstance(row, dict)],
            *[dict(row) for row in crypto_review_rows if isinstance(row, dict)],
        ]
    )
    review_head = dict(review_backlog[0]) if review_backlog else {}
    review_backlog_brief = " | ".join(
        [
            f"{int(row.get('rank') or 0)}:{str(row.get('area') or '-')}"
            f":{str(row.get('symbol') or '-')}"
            f":{str(row.get('priority_tier') or '-')}"
            f":{int(row.get('priority_score') or 0)}"
            for row in review_backlog[:8]
        ]
    ) or "-"
    review_backlog_status = "ready" if review_backlog else "inactive"
    review_head_lane = _build_review_head_lane(review_backlog, review_backlog_status)
    operator_head_lane = _build_operator_head_lane(operator_backlog, operator_backlog_status)
    operator_repair_head_lane = _build_operator_repair_head_lane(
        operator_repair_queue,
        operator_repair_checklist,
    )
    (
        operator_action_queue,
        operator_action_checklist,
        operator_action_queue_brief,
        operator_action_checklist_brief,
    ) = _build_operator_action_lanes(
        operator_state_lanes=operator_state_lanes,
        review_head=review_head,
        operator_repair_queue=operator_repair_queue,
    )
    operator_focus_slots, operator_focus_slots_brief = _build_operator_focus_slots(
        operator_action_checklist
    )

    payload: dict[str, Any] = {
        "action": "refresh_cross_market_operator_state",
        "ok": True,
        "status": "ok",
        "as_of": fmt_utc(runtime_now),
        "downstream_refresh_mode": downstream_refresh_mode,
        "downstream_reused_count": reused_downstream_count,
        "downstream_refresh_reuse_brief": downstream_refresh_reuse_brief,
        "brooks_refresh_artifact": str(brooks_refresh.get("artifact") or ""),
        "brooks_refresh_status": str(brooks_refresh.get("status") or ""),
        "brooks_review_queue_artifact": str(brooks_review_queue_path or ""),
        "brooks_domestic_futures_bridge_capability_artifact": str(
            brooks_domestic_futures_bridge_capability_path or ""
        ),
        "commodity_refresh_artifact": str(commodity_refresh.get("artifact") or ""),
        "commodity_refresh_status": str(commodity_refresh.get("status") or ""),
        "commodity_review_artifact": str(commodity_review_path or ""),
        "commodity_gap_artifact": str(commodity_gap_path or ""),
        "crypto_route_refresh_artifact": str(crypto_route_refresh_path or ""),
        "live_gate_blocker_artifact": str(live_gate_blocker_path or ""),
        "hot_brief_artifact": str(hot_brief_path),
        "operator_status": str(commodity_refresh.get("operator_status") or ""),
        "operator_stack_brief": str(commodity_refresh.get("operator_stack_brief") or ""),
        "operator_backlog_status": operator_backlog_status,
        "operator_backlog_count": len(operator_backlog),
        "operator_backlog_brief": operator_backlog_brief,
        "operator_backlog_state_counts": operator_backlog_state_counts,
        "operator_backlog_state_brief": operator_backlog_state_brief,
        "operator_backlog_priority_totals": operator_backlog_priority_totals,
        "operator_backlog_priority_totals_brief": operator_backlog_priority_totals_brief,
        "operator_state_lane_heads_brief": operator_state_lane_heads_brief,
        "operator_state_lane_priority_order_brief": operator_state_lane_priority_order_brief,
        "remote_live_operator_alignment": remote_live_operator_alignment,
        "remote_live_operator_alignment_status": str(remote_live_operator_alignment.get("status") or ""),
        "remote_live_operator_alignment_brief": str(remote_live_operator_alignment.get("brief") or ""),
        "remote_live_operator_alignment_blocking": bool(remote_live_operator_alignment.get("blocking", False)),
        "remote_live_operator_alignment_head_area": str(remote_live_operator_alignment.get("head_area") or ""),
        "remote_live_operator_alignment_head_symbol": str(remote_live_operator_alignment.get("head_symbol") or ""),
        "remote_live_operator_alignment_head_action": str(remote_live_operator_alignment.get("head_action") or ""),
        "remote_live_operator_alignment_head_state": str(remote_live_operator_alignment.get("head_state") or ""),
        "remote_live_operator_alignment_head_priority_score": int(remote_live_operator_alignment.get("head_priority_score") or 0),
        "remote_live_operator_alignment_head_priority_tier": str(remote_live_operator_alignment.get("head_priority_tier") or ""),
        "remote_live_operator_alignment_remote_status": str(remote_live_operator_alignment.get("remote_status") or ""),
        "remote_live_operator_alignment_remote_brief": str(remote_live_operator_alignment.get("remote_brief") or ""),
        "remote_live_operator_alignment_blocker_detail": str(remote_live_operator_alignment.get("blocker_detail") or ""),
        "remote_live_operator_alignment_done_when": str(remote_live_operator_alignment.get("done_when") or ""),
        "remote_live_takeover_gate": remote_live_takeover_gate,
        "remote_live_takeover_gate_status": str(remote_live_takeover_gate.get("status") or ""),
        "remote_live_takeover_gate_brief": str(remote_live_takeover_gate.get("brief") or ""),
        "remote_live_takeover_gate_blocking": bool(remote_live_takeover_gate.get("blocking", False)),
        "remote_live_takeover_gate_alignment_status": str(remote_live_takeover_gate.get("alignment_status") or ""),
        "remote_live_takeover_gate_head_area": str(remote_live_takeover_gate.get("head_area") or ""),
        "remote_live_takeover_gate_head_symbol": str(remote_live_takeover_gate.get("head_symbol") or ""),
        "remote_live_takeover_gate_head_action": str(remote_live_takeover_gate.get("head_action") or ""),
        "remote_live_takeover_gate_remote_status": str(remote_live_takeover_gate.get("remote_status") or ""),
        "remote_live_takeover_gate_blocker_detail": str(remote_live_takeover_gate.get("blocker_detail") or ""),
        "remote_live_takeover_gate_done_when": str(remote_live_takeover_gate.get("done_when") or ""),
        "remote_live_takeover_clearing": remote_live_takeover_clearing,
        "remote_live_takeover_clearing_status": str(remote_live_takeover_clearing.get("status") or ""),
        "remote_live_takeover_clearing_brief": str(remote_live_takeover_clearing.get("brief") or ""),
        "remote_live_takeover_clearing_ops_live_gate_brief": str(
            remote_live_takeover_clearing.get("ops_live_gate_brief") or ""
        ),
        "remote_live_takeover_clearing_risk_guard_brief": str(
            remote_live_takeover_clearing.get("risk_guard_brief") or ""
        ),
        "remote_live_takeover_clearing_blocker_detail": str(
            remote_live_takeover_clearing.get("blocker_detail") or ""
        ),
        "remote_live_takeover_clearing_done_when": str(
            remote_live_takeover_clearing.get("done_when") or ""
        ),
        "remote_live_takeover_clearing_source_freshness_brief": str(
            remote_live_takeover_clearing.get("source_freshness_brief") or ""
        ),
        "remote_live_takeover_slot_anomaly_breakdown": remote_live_takeover_slot_anomaly_breakdown,
        "remote_live_takeover_slot_anomaly_breakdown_status": str(
            remote_live_takeover_slot_anomaly_breakdown.get("status") or ""
        ),
        "remote_live_takeover_slot_anomaly_breakdown_brief": str(
            remote_live_takeover_slot_anomaly_breakdown.get("brief") or ""
        ),
        "remote_live_takeover_slot_anomaly_breakdown_artifact": str(
            remote_live_takeover_slot_anomaly_breakdown.get("artifact") or ""
        ),
        "remote_live_takeover_slot_anomaly_breakdown_repair_focus": str(
            remote_live_takeover_slot_anomaly_breakdown.get("repair_focus") or ""
        ),
        "remote_live_takeover_repair_queue": remote_live_takeover_repair_queue,
        "remote_live_takeover_repair_queue_status": str(
            remote_live_takeover_repair_queue.get("status") or ""
        ),
        "remote_live_takeover_repair_queue_brief": str(
            remote_live_takeover_repair_queue.get("brief") or ""
        ),
        "remote_live_takeover_repair_queue_count": int(
            remote_live_takeover_repair_queue.get("count") or 0
        ),
        "remote_live_takeover_repair_queue_head_area": str(
            remote_live_takeover_repair_queue.get("head_area") or ""
        ),
        "remote_live_takeover_repair_queue_head_code": str(
            remote_live_takeover_repair_queue.get("head_code") or ""
        ),
        "remote_live_takeover_repair_queue_head_action": str(
            remote_live_takeover_repair_queue.get("head_action") or ""
        ),
        "remote_live_takeover_repair_queue_head_priority_score": int(
            remote_live_takeover_repair_queue.get("head_priority_score") or 0
        ),
        "remote_live_takeover_repair_queue_head_priority_tier": str(
            remote_live_takeover_repair_queue.get("head_priority_tier") or ""
        ),
        "remote_live_takeover_repair_queue_head_command": str(
            remote_live_takeover_repair_queue.get("head_command") or ""
        ),
        "remote_live_takeover_repair_queue_head_clear_when": str(
            remote_live_takeover_repair_queue.get("head_clear_when") or ""
        ),
        "remote_live_takeover_repair_queue_queue_brief": str(
            remote_live_takeover_repair_queue.get("queue_brief") or ""
        ),
        "remote_live_takeover_repair_queue_done_when": str(
            remote_live_takeover_repair_queue.get("done_when") or ""
        ),
        "operator_repair_queue": operator_repair_queue,
        "operator_repair_queue_brief": operator_repair_queue_brief,
        "operator_repair_queue_count": len(operator_repair_queue),
        "operator_repair_checklist": operator_repair_checklist,
        "operator_repair_checklist_brief": operator_repair_checklist_brief,
        "operator_action_queue": operator_action_queue,
        "operator_action_queue_brief": operator_action_queue_brief,
        "operator_action_checklist": operator_action_checklist,
        "operator_action_checklist_brief": operator_action_checklist_brief,
        "operator_focus_slots": operator_focus_slots,
        "operator_focus_slots_brief": operator_focus_slots_brief,
        "review_head_lane": review_head_lane,
        "review_head_lane_status": str(review_head_lane.get("status") or ""),
        "review_head_lane_brief": str(review_head_lane.get("brief") or ""),
        "review_head_lane_backlog_count": int(review_head_lane.get("backlog_count") or 0),
        "review_head_lane_backlog_brief": str(review_head_lane.get("backlog_brief") or ""),
        "review_head_lane_blocker_detail": str(review_head_lane.get("blocker_detail") or ""),
        "review_head_lane_done_when": str(review_head_lane.get("done_when") or ""),
        "operator_head_lane": operator_head_lane,
        "operator_head_lane_status": str(operator_head_lane.get("status") or ""),
        "operator_head_lane_brief": str(operator_head_lane.get("brief") or ""),
        "operator_head_lane_backlog_count": int(operator_head_lane.get("backlog_count") or 0),
        "operator_head_lane_backlog_brief": str(operator_head_lane.get("backlog_brief") or ""),
        "operator_head_lane_blocker_detail": str(operator_head_lane.get("blocker_detail") or ""),
        "operator_head_lane_done_when": str(operator_head_lane.get("done_when") or ""),
        "operator_repair_head_lane": operator_repair_head_lane,
        "operator_repair_head_lane_status": str(operator_repair_head_lane.get("status") or ""),
        "operator_repair_head_lane_brief": str(operator_repair_head_lane.get("brief") or ""),
        "operator_repair_head_lane_backlog_count": int(operator_repair_head_lane.get("backlog_count") or 0),
        "operator_repair_head_lane_backlog_brief": str(operator_repair_head_lane.get("backlog_brief") or ""),
        "operator_repair_head_lane_blocker_detail": str(operator_repair_head_lane.get("blocker_detail") or ""),
        "operator_repair_head_lane_done_when": str(operator_repair_head_lane.get("done_when") or ""),
        "operator_repair_head_lane_command": str(operator_repair_head_lane.get("command") or ""),
        "operator_repair_head_lane_clear_when": str(operator_repair_head_lane.get("clear_when") or ""),
        "operator_backlog": operator_backlog,
        "operator_head": operator_head,
        "operator_head_area": str(operator_head.get("area") or ""),
        "operator_head_symbol": str(operator_head.get("symbol") or ""),
        "operator_head_action": str(operator_head.get("action") or ""),
        "operator_head_state": str(operator_head.get("state") or ""),
        "operator_head_priority_score": operator_head.get("priority_score"),
        "operator_head_priority_tier": str(operator_head.get("priority_tier") or ""),
        "operator_head_blocker_detail": str(operator_head.get("blocker_detail") or ""),
        "operator_head_done_when": str(operator_head.get("done_when") or ""),
        "operator_waiting_lane_status": str((operator_state_lanes.get("waiting") or {}).get("status") or ""),
        "operator_waiting_lane_count": int((operator_state_lanes.get("waiting") or {}).get("count") or 0),
        "operator_waiting_lane_brief": str((operator_state_lanes.get("waiting") or {}).get("brief") or ""),
        "operator_waiting_lane_priority_total": int((operator_state_lanes.get("waiting") or {}).get("priority_total") or 0),
        "operator_waiting_lane_head_symbol": str((operator_state_lanes.get("waiting") or {}).get("head_symbol") or ""),
        "operator_waiting_lane_head_action": str((operator_state_lanes.get("waiting") or {}).get("head_action") or ""),
        "operator_waiting_lane_head_priority_score": int((operator_state_lanes.get("waiting") or {}).get("head_priority_score") or 0),
        "operator_waiting_lane_head_priority_tier": str((operator_state_lanes.get("waiting") or {}).get("head_priority_tier") or ""),
        "operator_waiting_lane_head_blocker_detail": str((operator_state_lanes.get("waiting") or {}).get("head_blocker_detail") or ""),
        "operator_waiting_lane_head_done_when": str((operator_state_lanes.get("waiting") or {}).get("head_done_when") or ""),
        "operator_review_lane_status": str((operator_state_lanes.get("review") or {}).get("status") or ""),
        "operator_review_lane_count": int((operator_state_lanes.get("review") or {}).get("count") or 0),
        "operator_review_lane_brief": str((operator_state_lanes.get("review") or {}).get("brief") or ""),
        "operator_review_lane_priority_total": int((operator_state_lanes.get("review") or {}).get("priority_total") or 0),
        "operator_review_lane_head_symbol": str((operator_state_lanes.get("review") or {}).get("head_symbol") or ""),
        "operator_review_lane_head_action": str((operator_state_lanes.get("review") or {}).get("head_action") or ""),
        "operator_review_lane_head_priority_score": int((operator_state_lanes.get("review") or {}).get("head_priority_score") or 0),
        "operator_review_lane_head_priority_tier": str((operator_state_lanes.get("review") or {}).get("head_priority_tier") or ""),
        "operator_review_lane_head_blocker_detail": str((operator_state_lanes.get("review") or {}).get("head_blocker_detail") or ""),
        "operator_review_lane_head_done_when": str((operator_state_lanes.get("review") or {}).get("head_done_when") or ""),
        "operator_watch_lane_status": str((operator_state_lanes.get("watch") or {}).get("status") or ""),
        "operator_watch_lane_count": int((operator_state_lanes.get("watch") or {}).get("count") or 0),
        "operator_watch_lane_brief": str((operator_state_lanes.get("watch") or {}).get("brief") or ""),
        "operator_watch_lane_priority_total": int((operator_state_lanes.get("watch") or {}).get("priority_total") or 0),
        "operator_watch_lane_head_symbol": str((operator_state_lanes.get("watch") or {}).get("head_symbol") or ""),
        "operator_watch_lane_head_action": str((operator_state_lanes.get("watch") or {}).get("head_action") or ""),
        "operator_watch_lane_head_priority_score": int((operator_state_lanes.get("watch") or {}).get("head_priority_score") or 0),
        "operator_watch_lane_head_priority_tier": str((operator_state_lanes.get("watch") or {}).get("head_priority_tier") or ""),
        "operator_watch_lane_head_blocker_detail": str((operator_state_lanes.get("watch") or {}).get("head_blocker_detail") or ""),
        "operator_watch_lane_head_done_when": str((operator_state_lanes.get("watch") or {}).get("head_done_when") or ""),
        "operator_blocked_lane_status": str((operator_state_lanes.get("blocked") or {}).get("status") or ""),
        "operator_blocked_lane_count": int((operator_state_lanes.get("blocked") or {}).get("count") or 0),
        "operator_blocked_lane_brief": str((operator_state_lanes.get("blocked") or {}).get("brief") or ""),
        "operator_blocked_lane_priority_total": int((operator_state_lanes.get("blocked") or {}).get("priority_total") or 0),
        "operator_blocked_lane_head_symbol": str((operator_state_lanes.get("blocked") or {}).get("head_symbol") or ""),
        "operator_blocked_lane_head_action": str((operator_state_lanes.get("blocked") or {}).get("head_action") or ""),
        "operator_blocked_lane_head_priority_score": int((operator_state_lanes.get("blocked") or {}).get("head_priority_score") or 0),
        "operator_blocked_lane_head_priority_tier": str((operator_state_lanes.get("blocked") or {}).get("head_priority_tier") or ""),
        "operator_blocked_lane_head_blocker_detail": str((operator_state_lanes.get("blocked") or {}).get("head_blocker_detail") or ""),
        "operator_blocked_lane_head_done_when": str((operator_state_lanes.get("blocked") or {}).get("head_done_when") or ""),
        "operator_repair_lane_status": str((operator_state_lanes.get("repair") or {}).get("status") or ""),
        "operator_repair_lane_count": int((operator_state_lanes.get("repair") or {}).get("count") or 0),
        "operator_repair_lane_brief": str((operator_state_lanes.get("repair") or {}).get("brief") or ""),
        "operator_repair_lane_priority_total": int((operator_state_lanes.get("repair") or {}).get("priority_total") or 0),
        "operator_repair_lane_head_symbol": str((operator_state_lanes.get("repair") or {}).get("head_symbol") or ""),
        "operator_repair_lane_head_action": str((operator_state_lanes.get("repair") or {}).get("head_action") or ""),
        "operator_repair_lane_head_priority_score": int((operator_state_lanes.get("repair") or {}).get("head_priority_score") or 0),
        "operator_repair_lane_head_priority_tier": str((operator_state_lanes.get("repair") or {}).get("head_priority_tier") or ""),
        "operator_repair_lane_head_blocker_detail": str((operator_state_lanes.get("repair") or {}).get("head_blocker_detail") or ""),
        "operator_repair_lane_head_done_when": str((operator_state_lanes.get("repair") or {}).get("head_done_when") or ""),
        "review_backlog_status": review_backlog_status,
        "review_backlog_count": len(review_backlog),
        "review_backlog_brief": review_backlog_brief,
        "review_backlog": review_backlog,
        "review_head": review_head,
        "review_head_area": str(review_head.get("area") or ""),
        "review_head_symbol": str(review_head.get("symbol") or ""),
        "review_head_action": str(review_head.get("action") or ""),
        "review_head_priority_score": review_head.get("priority_score"),
        "review_head_priority_tier": str(review_head.get("priority_tier") or ""),
        "review_head_blocker_detail": str(review_head.get("blocker_detail") or ""),
        "review_head_done_when": str(review_head.get("done_when") or ""),
        "steps": [
            {
                "name": "refresh_brooks_structure_state",
                "status": "reused_previous_artifact" if bool(args.skip_downstream_refresh) else str(brooks_refresh.get("status") or ""),
                "now": fmt_utc(brooks_now),
                "as_of": payload_or_artifact_as_of(brooks_refresh),
                "artifact": str(brooks_refresh.get("artifact") or ""),
            },
            {
                "name": "refresh_commodity_paper_execution_state",
                "status": "reused_previous_artifact" if bool(args.skip_downstream_refresh) else str(commodity_refresh.get("status") or ""),
                "now": fmt_utc(commodity_now),
                "as_of": payload_or_artifact_as_of(commodity_refresh),
                "artifact": str(commodity_refresh.get("artifact") or ""),
            },
        ],
    }

    review_dir.mkdir(parents=True, exist_ok=True)
    stamp = runtime_now.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_path = review_dir / f"{stamp}_cross_market_operator_state.json"
    markdown_path = review_dir / f"{stamp}_cross_market_operator_state.md"
    checksum_path = review_dir / f"{stamp}_cross_market_operator_state_checksum.json"
    artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    checksum_payload = {
        "artifact": str(artifact_path),
        "markdown": str(markdown_path),
        "sha256": sha256_file(artifact_path),
        "generated_at": payload.get("as_of"),
    }
    checksum_path.write_text(json.dumps(checksum_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pruned_keep, pruned_age = prune_review_artifacts(
        review_dir,
        stem="cross_market_operator_state",
        current_paths=[artifact_path, markdown_path, checksum_path],
        keep=max(1, int(args.artifact_keep)),
        ttl_hours=float(args.artifact_ttl_hours),
        now_dt=runtime_now,
    )
    hot_brief_refresh_payload: dict[str, Any] = {}
    if not args.skip_downstream_refresh:
        hot_brief_refresh_now = step_now(runtime_now, 6)
        try:
            hot_brief_refresh_payload = run_json_step(
                step_name="refresh_hot_universe_operator_brief",
                cmd=[
                    "python3",
                    str(script_path("build_hot_universe_operator_brief.py")),
                    "--review-dir",
                    str(review_dir),
                    "--now",
                    fmt_utc(hot_brief_refresh_now),
                ],
            )
            refreshed_hot_brief_path = Path(str(hot_brief_refresh_payload.get("artifact") or "")).expanduser()
            if refreshed_hot_brief_path.exists():
                hot_brief_path = refreshed_hot_brief_path
        except Exception as exc:
            hot_brief_refresh_payload = {
                "status": "failed",
                "artifact": str(hot_brief_path),
                "blocker_detail": str(exc),
            }
        payload["steps"].append(
            {
                "name": "refresh_hot_universe_operator_brief",
                "status": str(hot_brief_refresh_payload.get("status") or ""),
                "now": fmt_utc(hot_brief_refresh_now),
                "as_of": payload_or_artifact_as_of(hot_brief_refresh_payload),
                "artifact": str(hot_brief_refresh_payload.get("artifact") or hot_brief_path),
            }
        )

    payload["hot_brief_artifact"] = str(hot_brief_path)
    payload["hot_brief_refresh_status"] = str(hot_brief_refresh_payload.get("status") or "")
    payload["hot_brief_refresh_artifact"] = str(hot_brief_refresh_payload.get("artifact") or "")
    payload["hot_brief_refresh_blocker_detail"] = str(
        hot_brief_refresh_payload.get("blocker_detail") or ""
    )
    payload["artifact"] = str(artifact_path)
    payload["markdown"] = str(markdown_path)
    payload["checksum"] = str(checksum_path)
    payload["pruned_keep"] = pruned_keep
    payload["pruned_age"] = pruned_age
    artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checksum_payload["sha256"] = sha256_file(artifact_path)
    checksum_payload["generated_at"] = payload.get("as_of")
    checksum_path.write_text(json.dumps(checksum_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
