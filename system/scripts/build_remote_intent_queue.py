#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_TICKET_FRESHNESS_SECONDS = 900


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


def parse_now(raw: str | None) -> dt.datetime:
    text = str(raw or "").strip()
    if not text:
        return now_utc()
    parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def parse_utc(raw: Any) -> dt.datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def fmt_utc(value: dt.datetime | None) -> str:
    effective = value or now_utc()
    return effective.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def find_latest(review_dir: Path, pattern: str) -> Path | None:
    files = sorted(review_dir.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
    return files[0] if files else None


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def text(value: Any) -> str:
    return str(value or "").strip()


def is_remote_executable_row(row: dict[str, Any], remote_market: str) -> bool:
    area = text(row.get("area"))
    action = text(row.get("action"))
    if remote_market.strip().lower() not in {"spot", "portfolio_margin_um"}:
        return False
    if area != "crypto_route":
        return False
    return bool(action)


def join_nonempty(parts: list[str], *, sep: str = " | ") -> str:
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        item = text(part)
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return sep.join(out)


def find_preferred_candidate(
    *,
    cross_market_payload: dict[str, Any],
    scope_router_payload: dict[str, Any],
    remote_market: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    review_backlog = [
        dict(row)
        for row in as_list(cross_market_payload.get("review_backlog"))
        if isinstance(row, dict) and is_remote_executable_row(row, remote_market)
    ]
    preferred_symbol = text(scope_router_payload.get("preferred_route_symbol"))
    preferred_action = text(scope_router_payload.get("preferred_route_action"))
    if preferred_symbol:
        for row in review_backlog:
            if text(row.get("symbol")) == preferred_symbol and (
                not preferred_action or text(row.get("action")) == preferred_action
            ):
                return row, review_backlog

    review_head = as_dict(cross_market_payload.get("review_head"))
    if review_head and is_remote_executable_row(review_head, remote_market):
        return dict(review_head), review_backlog

    return (dict(review_backlog[0]) if review_backlog else {}), review_backlog


def load_ticket_state(
    *,
    review_dir: Path,
    symbol: str,
    reference_now: dt.datetime,
    freshness_seconds: int,
) -> dict[str, Any]:
    artifact = find_latest(review_dir, "*_signal_to_order_tickets.json")
    if artifact is None:
        return {
            "artifact": "",
            "artifact_status": "missing_artifact",
            "artifact_age_seconds": None,
            "ticket_match_status": "row_missing",
            "ticket_match_brief": "ticket_artifact_missing",
            "selected_row": {},
        }

    payload = load_json_mapping(artifact)
    generated_at = parse_utc(payload.get("generated_at_utc"))
    if generated_at is None:
        generated_at = dt.datetime.fromtimestamp(artifact.stat().st_mtime, tz=dt.timezone.utc)
    age_seconds = max(0.0, (reference_now - generated_at).total_seconds())
    artifact_status = (
        "stale_artifact"
        if age_seconds > float(max(1, int(freshness_seconds)))
        else "fresh_artifact"
    )
    row = {}
    for raw in as_list(payload.get("tickets")):
        if not isinstance(raw, dict):
            continue
        if text(raw.get("symbol")).upper() == symbol.upper():
            row = dict(raw)
            break
    if not row:
        return {
            "artifact": str(artifact),
            "artifact_status": artifact_status,
            "artifact_age_seconds": age_seconds,
            "ticket_match_status": "row_missing",
            "ticket_match_brief": f"{artifact_status}:ticket_row_missing:{symbol}",
            "selected_row": {},
        }

    reasons = [text(item) for item in as_list(row.get("reasons")) if text(item)]
    if bool(row.get("allowed", False)):
        match_status = "row_ready"
        match_brief = f"{artifact_status}:ticket_row_ready:{symbol}"
    else:
        match_status = "row_blocked"
        match_brief = join_nonempty(
            [f"{artifact_status}:ticket_row_blocked:{symbol}", ",".join(reasons)],
            sep=" | ",
        )
    return {
        "artifact": str(artifact),
        "artifact_status": artifact_status,
        "artifact_age_seconds": age_seconds,
        "ticket_match_status": match_status,
        "ticket_match_brief": match_brief,
        "selected_row": row,
    }


def load_risk_guard_alignment(*, live_gate_payload: dict[str, Any], preferred_symbol: str) -> dict[str, Any]:
    blockers = [
        dict(row)
        for row in as_list(live_gate_payload.get("blockers"))
        if isinstance(row, dict) and text(row.get("name")) == "risk_guard"
    ]
    risk_guard = blockers[0] if blockers else {}
    blocked_candidate = as_dict(risk_guard.get("blocked_candidate"))
    blocked_symbol = text(blocked_candidate.get("symbol"))
    if not preferred_symbol:
        status = "route_symbol_missing"
        brief = "route_symbol_missing"
    elif blocked_symbol and blocked_symbol != preferred_symbol:
        status = "risk_guard_candidate_mismatch"
        brief = f"risk_guard_candidate_mismatch:{blocked_symbol}->{preferred_symbol}"
    elif blocked_symbol and blocked_symbol == preferred_symbol:
        status = "risk_guard_candidate_matches_route_symbol"
        brief = f"risk_guard_candidate_matches_route_symbol:{preferred_symbol}"
    elif any(text(code).startswith("ticket_missing:") for code in as_list(risk_guard.get("reason_codes"))):
        status = "ticket_missing_without_candidate_symbol"
        brief = f"ticket_missing_without_candidate_symbol:{preferred_symbol}"
    else:
        status = "risk_guard_not_ticket_blocked"
        brief = "risk_guard_not_ticket_blocked"
    return {
        "status": status,
        "brief": brief,
        "blocked_candidate_symbol": blocked_symbol,
        "blocked_candidate_action": text(blocked_candidate.get("side")),
        "reason_codes": [text(code) for code in as_list(risk_guard.get("reason_codes")) if text(code)],
    }


def load_execution_contract(identity_payload: dict[str, Any]) -> dict[str, Any]:
    source = identity_payload if isinstance(identity_payload, dict) else {}
    status = text(source.get("execution_contract_status"))
    brief = text(source.get("execution_contract_brief"))
    mode = text(source.get("execution_contract_mode"))
    live_orders_allowed_raw = source.get("execution_contract_live_orders_allowed")
    live_orders_allowed = (
        bool(live_orders_allowed_raw) if live_orders_allowed_raw is not None else None
    )
    reason_codes = [
        text(code)
        for code in as_list(source.get("execution_contract_reason_codes"))
        if text(code)
    ]
    guarded_probe_allowed = bool(source.get("execution_contract_guarded_probe_allowed", False)) or (
        status == "probe_only_contract"
        or mode == "guarded_probe_only"
        or "guarded_probe_only_mode" in reason_codes
    )
    blocked = (
        not guarded_probe_allowed
        and status == "non_executable_contract"
        or bool(reason_codes)
        and not guarded_probe_allowed
        or ((live_orders_allowed is False and bool(status or brief or mode)) and not guarded_probe_allowed)
    )
    return {
        "status": status,
        "brief": brief,
        "mode": mode,
        "guarded_probe_allowed": guarded_probe_allowed,
        "live_orders_allowed": live_orders_allowed,
        "executor_mode": text(source.get("execution_contract_executor_mode")),
        "executor_mode_source": text(source.get("execution_contract_executor_mode_source")),
        "reason_codes": reason_codes,
        "blocker_detail": text(source.get("execution_contract_blocker_detail")),
        "done_when": text(source.get("execution_contract_done_when")),
        "blocked": blocked,
    }


def build_queue_status(
    *,
    preferred: dict[str, Any],
    scope_router_payload: dict[str, Any],
    ticket_state: dict[str, Any],
    execution_contract: dict[str, Any],
) -> tuple[str, str, bool]:
    symbol = text(preferred.get("symbol"))
    action = text(preferred.get("action"))
    scope_status = text(scope_router_payload.get("scope_router_status"))
    if not symbol:
        return ("no_remote_scope_candidate", "keep_remote_idle", False)
    if scope_status == "review_candidate_inside_scope_not_trade_ready":
        return ("queued_wait_trade_readiness", "hold_remote_idle_until_ticket_ready", False)
    if ticket_state.get("artifact_status") == "missing_artifact":
        return ("queued_ticket_generation_required", "generate_ticket_for_remote_candidate", False)
    if ticket_state.get("ticket_match_status") == "row_missing":
        return ("queued_ticket_generation_required", "generate_ticket_for_remote_candidate", False)
    if ticket_state.get("artifact_status") == "stale_artifact":
        return ("queued_ticket_blocked", "refresh_tickets_before_enqueue", False)
    if ticket_state.get("ticket_match_status") == "row_blocked":
        return ("queued_ticket_blocked", "repair_ticket_constraints_before_enqueue", False)
    if bool(execution_contract.get("guarded_probe_allowed")) and symbol and action:
        return ("queued_guarded_probe_ready", "run_guarded_probe_from_ticket_ready_intent", True)
    if bool(execution_contract.get("blocked")):
        return (
            "queued_execution_contract_blocked",
            "hold_remote_idle_until_execution_contract_promoted",
            False,
        )
    if symbol and action:
        return ("queued_ticket_ready", "seed_execution_journal_from_ticket_ready_intent", True)
    return ("queued_ticket_generation_required", "generate_ticket_for_remote_candidate", False)


def build_payload(
    *,
    review_dir: Path,
    cross_market_path: Path,
    cross_market_payload: dict[str, Any],
    identity_path: Path,
    identity_payload: dict[str, Any],
    scope_router_path: Path,
    scope_router_payload: dict[str, Any],
    live_gate_path: Path,
    live_gate_payload: dict[str, Any],
    reference_now: dt.datetime,
    ticket_freshness_seconds: int,
) -> dict[str, Any]:
    remote_market = text(identity_payload.get("ready_check_scope_market")) or "portfolio_margin_um"
    preferred, route_candidates = find_preferred_candidate(
        cross_market_payload=cross_market_payload,
        scope_router_payload=scope_router_payload,
        remote_market=remote_market,
    )
    preferred_symbol = text(preferred.get("symbol"))
    preferred_action = text(preferred.get("action"))
    execution_contract = load_execution_contract(identity_payload)
    ticket_state = load_ticket_state(
        review_dir=review_dir,
        symbol=preferred_symbol,
        reference_now=reference_now,
        freshness_seconds=ticket_freshness_seconds,
    )
    guard_alignment = load_risk_guard_alignment(
        live_gate_payload=live_gate_payload,
        preferred_symbol=preferred_symbol,
    )
    queue_status, queue_recommendation, intent_ready = build_queue_status(
        preferred=preferred,
        scope_router_payload=scope_router_payload,
        ticket_state=ticket_state,
        execution_contract=execution_contract,
    )

    queue_brief = ":".join(
        [
            queue_status,
            preferred_symbol or "-",
            preferred_action or "-",
            remote_market or "-",
        ]
    )
    route_candidates_brief = " | ".join(
        [
            ":".join(
                [
                    str(int(row.get("rank") or 0)),
                    text(row.get("symbol")),
                    text(row.get("action")),
                    str(int(row.get("priority_score") or 0)),
                ]
            )
            for row in route_candidates
            if text(row.get("symbol"))
        ]
    )
    blocker_detail = join_nonempty(
        [
            text(execution_contract.get("brief")),
            text(scope_router_payload.get("blocker_detail")),
            text(ticket_state.get("ticket_match_brief")),
            text(guard_alignment.get("brief")),
        ]
    )
    done_when = join_nonempty(
        [
            text(execution_contract.get("done_when")),
            text(scope_router_payload.get("done_when")),
            (
                f"generate a fresh signal_to_order_tickets row for {preferred_symbol}"
                if queue_status == "queued_ticket_generation_required" and preferred_symbol
                else ""
            ),
            (
                f"repair ticket constraints for {preferred_symbol} until allowed=true"
                if queue_status == "queued_ticket_blocked" and preferred_symbol
                else ""
            ),
            (
                f"append {preferred_symbol} remote intent into the execution journal path"
                if queue_status == "queued_ticket_ready" and preferred_symbol
                else ""
            ),
        ]
    )
    queue_rows = []
    for row in route_candidates:
        queue_rows.append(
            {
                "rank": int(row.get("rank") or 0),
                "symbol": text(row.get("symbol")),
                "action": text(row.get("action")),
                "priority_score": int(row.get("priority_score") or 0),
                "priority_tier": text(row.get("priority_tier")),
                "queue_state": queue_status if text(row.get("symbol")) == preferred_symbol else "scope_candidate_backlog",
                "reason": text(row.get("reason")),
                "blocker_detail": text(row.get("blocker_detail")),
                "done_when": text(row.get("done_when")),
            }
        )
    return {
        "action": "build_remote_intent_queue",
        "ok": True,
        "status": "ok",
        "generated_at_utc": fmt_utc(reference_now),
        "queue_status": queue_status,
        "queue_brief": queue_brief,
        "queue_recommendation": queue_recommendation,
        "intent_ready": bool(intent_ready),
        "remote_market": remote_market,
        "remote_execution_identity_brief": text(identity_payload.get("identity_brief")),
        "execution_contract_status": text(execution_contract.get("status")),
        "execution_contract_brief": text(execution_contract.get("brief")),
        "execution_contract_mode": text(execution_contract.get("mode")),
        "execution_contract_guarded_probe_allowed": bool(
            execution_contract.get("guarded_probe_allowed", False)
        ),
        "execution_contract_live_orders_allowed": execution_contract.get("live_orders_allowed"),
        "execution_contract_executor_mode": text(execution_contract.get("executor_mode")),
        "execution_contract_executor_mode_source": text(
            execution_contract.get("executor_mode_source")
        ),
        "execution_contract_reason_codes": execution_contract.get("reason_codes") or [],
        "remote_scope_router_brief": text(scope_router_payload.get("scope_router_brief")),
        "remote_scope_router_status": text(scope_router_payload.get("scope_router_status")),
        "preferred_route_symbol": preferred_symbol,
        "preferred_route_action": preferred_action,
        "preferred_route_priority_score": int(preferred.get("priority_score") or 0) if preferred else 0,
        "route_candidates_count": len(route_candidates),
        "route_candidates_brief": route_candidates_brief or "-",
        "ticket_artifact": text(ticket_state.get("artifact")),
        "ticket_artifact_status": text(ticket_state.get("artifact_status")),
        "ticket_artifact_age_seconds": ticket_state.get("artifact_age_seconds"),
        "ticket_match_status": text(ticket_state.get("ticket_match_status")),
        "ticket_match_brief": text(ticket_state.get("ticket_match_brief")),
        "ticket_selected_row": ticket_state.get("selected_row") or {},
        "guard_alignment_status": text(guard_alignment.get("status")),
        "guard_alignment_brief": text(guard_alignment.get("brief")),
        "guard_blocked_candidate_symbol": text(guard_alignment.get("blocked_candidate_symbol")),
        "guard_blocked_candidate_action": text(guard_alignment.get("blocked_candidate_action")),
        "risk_guard_reason_codes": guard_alignment.get("reason_codes") or [],
        "blocker_detail": blocker_detail,
        "done_when": done_when,
        "queue_rows_count": len(queue_rows),
        "queue_rows_brief": route_candidates_brief or "-",
        "queue_rows": queue_rows,
        "artifacts": {
            "cross_market_operator_state": str(cross_market_path),
            "remote_execution_identity_state": str(identity_path),
            "remote_scope_router_state": str(scope_router_path),
            "live_gate_blocker_report": str(live_gate_path),
            "signal_to_order_tickets": text(ticket_state.get("artifact")),
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Remote Intent Queue",
            "",
            f"- brief: `{text(payload.get('queue_brief'))}`",
            f"- recommendation: `{text(payload.get('queue_recommendation'))}`",
            f"- remote market: `{text(payload.get('remote_market'))}`",
            f"- execution contract: `{text(payload.get('execution_contract_brief'))}`",
            f"- preferred route: `{text(payload.get('preferred_route_symbol'))}:{text(payload.get('preferred_route_action'))}:{payload.get('preferred_route_priority_score')}`",
            f"- candidates: `{text(payload.get('route_candidates_brief'))}`",
            f"- ticket state: `{text(payload.get('ticket_match_brief'))}`",
            f"- guard alignment: `{text(payload.get('guard_alignment_brief'))}`",
            f"- blocker: `{text(payload.get('blocker_detail'))}`",
            f"- done_when: `{text(payload.get('done_when'))}`",
            "",
        ]
    )


def prune_artifacts(
    review_dir: Path,
    *,
    current_paths: list[Path],
    keep: int,
    ttl_hours: float,
) -> tuple[list[str], list[str]]:
    cutoff = now_utc() - dt.timedelta(hours=max(1.0, float(ttl_hours)))
    protected = {path.name for path in current_paths}
    candidates: list[Path] = []
    for pattern in (
        "*_remote_intent_queue.json",
        "*_remote_intent_queue.md",
        "*_remote_intent_queue_checksum.json",
    ):
        candidates.extend(review_dir.glob(pattern))

    existing: list[tuple[float, Path]] = []
    for path in candidates:
        try:
            existing.append((path.stat().st_mtime, path))
        except OSError:
            continue

    pruned_age: list[str] = []
    survivors: list[Path] = []
    for _, path in sorted(existing, key=lambda item: item[0], reverse=True):
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
    for path in survivors[max(1, int(keep)) :]:
        if path.name in protected:
            continue
        path.unlink(missing_ok=True)
        pruned_keep.append(str(path))
    return pruned_keep, pruned_age


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build remote intent queue for OpenClaw.")
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--ticket-freshness-seconds", type=int, default=DEFAULT_TICKET_FRESHNESS_SECONDS)
    parser.add_argument("--now", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    reference_now = parse_now(args.now)
    cross_market_path = find_latest(review_dir, "*_cross_market_operator_state.json")
    identity_path = find_latest(review_dir, "*_remote_execution_identity_state.json")
    scope_router_path = find_latest(review_dir, "*_remote_scope_router_state.json")
    live_gate_path = find_latest(review_dir, "*_live_gate_blocker_report.json")
    missing = [
        name
        for name, path in (
            ("cross_market_operator_state", cross_market_path),
            ("remote_execution_identity_state", identity_path),
            ("remote_scope_router_state", scope_router_path),
            ("live_gate_blocker_report", live_gate_path),
        )
        if path is None
    ]
    if missing:
        raise SystemExit(f"missing_required_artifacts:{','.join(missing)}")

    payload = build_payload(
        review_dir=review_dir,
        cross_market_path=cross_market_path,
        cross_market_payload=load_json_mapping(cross_market_path),
        identity_path=identity_path,
        identity_payload=load_json_mapping(identity_path),
        scope_router_path=scope_router_path,
        scope_router_payload=load_json_mapping(scope_router_path),
        live_gate_path=live_gate_path,
        live_gate_payload=load_json_mapping(live_gate_path),
        reference_now=reference_now,
        ticket_freshness_seconds=int(max(1, int(args.ticket_freshness_seconds))),
    )

    stamp = reference_now.strftime("%Y%m%dT%H%M%SZ")
    artifact = review_dir / f"{stamp}_remote_intent_queue.json"
    markdown = review_dir / f"{stamp}_remote_intent_queue.md"
    checksum = review_dir / f"{stamp}_remote_intent_queue_checksum.json"
    artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown.write_text(render_markdown(payload), encoding="utf-8")
    checksum.write_text(
        json.dumps(
            {
                "artifact": str(artifact),
                "artifact_sha256": sha256_file(artifact),
                "markdown": str(markdown),
                "markdown_sha256": sha256_file(markdown),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    pruned_keep, pruned_age = prune_artifacts(
        review_dir,
        current_paths=[artifact, markdown, checksum],
        keep=int(max(1, int(args.artifact_keep))),
        ttl_hours=float(max(1.0, float(args.artifact_ttl_hours))),
    )
    result = dict(payload)
    result.update(
        {
            "artifact": str(artifact),
            "markdown": str(markdown),
            "checksum": str(checksum),
            "pruned_due_to_keep": pruned_keep,
            "pruned_due_to_age": pruned_age,
        }
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
