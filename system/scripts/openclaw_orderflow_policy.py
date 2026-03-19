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


def text(value: Any) -> str:
    return str(value or "").strip()


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def dedupe_text(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        item = text(raw)
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def join_nonempty(parts: list[str], *, sep: str = " | ") -> str:
    return sep.join([item for item in [text(part) for part in parts] if item])


def load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def find_latest(review_dir: Path, pattern: str) -> Path | None:
    files = sorted(review_dir.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
    return files[0] if files else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        "*_remote_orderflow_policy_state.json",
        "*_remote_orderflow_policy_state.md",
        "*_remote_orderflow_policy_state_checksum.json",
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


def risk_guard_reason_codes(live_gate_payload: dict[str, Any]) -> list[str]:
    for row in as_list(live_gate_payload.get("blockers")):
        if not isinstance(row, dict):
            continue
        if text(row.get("name")) != "risk_guard":
            continue
        return dedupe_text([text(code) for code in as_list(row.get("reason_codes"))])
    return []


def live_gate_status_names(live_gate_payload: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for row in as_list(live_gate_payload.get("blockers")):
        if not isinstance(row, dict):
            continue
        if text(row.get("status")) == "blocked":
            name = text(row.get("name"))
            if name:
                out.append(name)
    return dedupe_text(out)


def build_payload(
    *,
    intent_path: Path,
    intent_payload: dict[str, Any],
    feedback_path: Path,
    feedback_payload: dict[str, Any],
    live_gate_path: Path,
    live_gate_payload: dict[str, Any],
    scope_router_path: Path | None,
    scope_router_payload: dict[str, Any],
    reference_now: dt.datetime,
) -> dict[str, Any]:
    symbol = text(intent_payload.get("preferred_route_symbol")) or text(feedback_payload.get("route_symbol"))
    action = text(intent_payload.get("preferred_route_action")) or text(feedback_payload.get("route_action"))
    remote_market = text(intent_payload.get("remote_market")) or text(feedback_payload.get("remote_market"))
    blocked_names = live_gate_status_names(live_gate_payload)
    reason_codes = dedupe_text(
        risk_guard_reason_codes(live_gate_payload)
        + [text(feedback_payload.get("dominant_guard_reason"))]
    )
    queue_status = text(intent_payload.get("queue_status"))
    feedback_status = text(feedback_payload.get("feedback_status"))
    scope_router_status = text(scope_router_payload.get("scope_router_status"))
    ticket_match_brief = text(intent_payload.get("ticket_match_brief"))
    execution_contract_status = text(intent_payload.get("execution_contract_status"))
    execution_contract_brief = text(intent_payload.get("execution_contract_brief"))
    execution_contract_mode = text(intent_payload.get("execution_contract_mode"))
    execution_contract_executor_mode = text(intent_payload.get("execution_contract_executor_mode"))
    execution_contract_executor_mode_source = text(
        intent_payload.get("execution_contract_executor_mode_source")
    )
    execution_contract_guarded_probe_allowed = bool(
        intent_payload.get("execution_contract_guarded_probe_allowed", False)
    ) or execution_contract_status == "probe_only_contract" or execution_contract_mode == "guarded_probe_only"
    execution_contract_reason_codes = dedupe_text(
        [text(code) for code in as_list(intent_payload.get("execution_contract_reason_codes"))]
    )
    execution_contract_live_orders_allowed_raw = intent_payload.get(
        "execution_contract_live_orders_allowed"
    )
    execution_contract_live_orders_allowed = (
        bool(execution_contract_live_orders_allowed_raw)
        if execution_contract_live_orders_allowed_raw is not None
        else None
    )
    runtime_promotion_requested = (
        execution_contract_mode == "promotion_requested"
        or "requested_executor_mode_not_implemented" in execution_contract_reason_codes
    )
    guarded_probe_candidate = (
        execution_contract_guarded_probe_allowed
        or queue_status == "queued_guarded_probe_ready"
        or execution_contract_mode == "guarded_probe_only"
        or "guarded_probe_only_mode" in execution_contract_reason_codes
    )
    execution_contract_blocked = (
        not guarded_probe_candidate
        and execution_contract_status == "non_executable_contract"
        or bool(execution_contract_reason_codes)
        and not guarded_probe_candidate
        or (
            execution_contract_live_orders_allowed is False
            and bool(execution_contract_status or execution_contract_brief)
            and not guarded_probe_candidate
        )
    )
    guardian_blocked = "risk_guard" in blocked_names or "ops_live_gate" in blocked_names
    learning_candidate = bool(symbol and remote_market) and (
        queue_status.startswith("queued_")
        or feedback_status.startswith("downrank_")
        or "inside_scope" in scope_router_status
    )
    if runtime_promotion_requested:
        policy_status = "shadow_policy_runtime_boundary_blocked"
        policy_decision = "hold_requested_runtime_promotion"
        policy_recommendation = "implement_requested_runtime_before_nonshadow_promotion"
    elif guarded_probe_candidate and not guardian_blocked:
        policy_status = "shadow_policy_guarded_probe_candidate"
        policy_decision = "accept_guarded_probe_candidate"
        policy_recommendation = "run_guarded_probe_before_live_promotion"
    elif execution_contract_blocked:
        policy_status = "shadow_policy_execution_contract_blocked"
        policy_decision = "accept_shadow_learning_only"
        policy_recommendation = "keep_shadow_learning_until_execution_contract_promoted"
    elif guardian_blocked or queue_status == "queued_wait_trade_readiness" or feedback_status.startswith(
        "downrank_guardian_blocked"
    ):
        if learning_candidate:
            policy_status = "shadow_policy_learning_only"
            policy_decision = "accept_shadow_learning_only"
            policy_recommendation = "continue_shadow_learning_until_guardian_clear"
        else:
            policy_status = "shadow_policy_blocked"
            policy_decision = "reject_until_guardian_clear"
            policy_recommendation = "keep_shadow_only_and_reject_stale_intent"
    else:
        policy_status = "shadow_policy_candidate_ready"
        policy_decision = "accept_shadow_candidate"
        policy_recommendation = "allow_shadow_candidate_to_progress"
    policy_brief = ":".join(
        [
            policy_status,
            symbol or "-",
            queue_status or "-",
            text(feedback_payload.get("feedback_status")) or "no_feedback",
        ]
    )
    return {
        "action": "openclaw_orderflow_policy",
        "ok": True,
        "status": "ok",
        "generated_at_utc": fmt_utc(reference_now),
        "policy_status": policy_status,
        "policy_brief": policy_brief,
        "policy_decision": policy_decision,
        "policy_recommendation": policy_recommendation,
        "route_symbol": symbol,
        "route_action": action,
        "remote_market": remote_market,
        "queue_status": queue_status,
        "feedback_status": feedback_status,
        "feedback_brief": text(feedback_payload.get("feedback_brief")),
        "feedback_recommendation": text(feedback_payload.get("feedback_recommendation")),
        "scope_router_status": scope_router_status,
        "scope_router_brief": text(scope_router_payload.get("scope_router_brief")),
        "execution_contract_status": execution_contract_status,
        "execution_contract_brief": execution_contract_brief,
        "execution_contract_guarded_probe_allowed": execution_contract_guarded_probe_allowed,
        "execution_contract_executor_mode": execution_contract_executor_mode,
        "execution_contract_executor_mode_source": execution_contract_executor_mode_source,
        "execution_contract_reason_codes": execution_contract_reason_codes,
        "execution_contract_live_orders_allowed": execution_contract_live_orders_allowed,
        "blocked_gate_names": blocked_names,
        "risk_reason_codes": reason_codes,
        "ticket_match_brief": ticket_match_brief,
        "guard_alignment_brief": text(intent_payload.get("guard_alignment_brief")),
        "shadow_learning_allowed": policy_decision == "accept_shadow_learning_only",
        "live_transport_allowed": policy_decision == "accept_shadow_candidate",
        "guarded_probe_allowed": policy_decision == "accept_guarded_probe_candidate",
        "blocker_detail": join_nonempty(
            [
                execution_contract_brief,
                text(feedback_payload.get("feedback_brief")),
                ",".join(blocked_names),
                ",".join(reason_codes),
            ]
        ),
        "done_when": (
            "policy can accept a shadow candidate only after the execution contract is live-capable, queue readiness improves, guardian blockers clear, and feedback stops down-ranking the active route"
        ),
        "artifacts": {
            "remote_intent_queue": str(intent_path),
            "remote_orderflow_feedback": str(feedback_path),
            "live_gate_blocker_report": str(live_gate_path),
            "remote_scope_router_state": str(scope_router_path) if scope_router_path else "",
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Remote Orderflow Policy State",
            "",
            f"- brief: `{text(payload.get('policy_brief'))}`",
            f"- decision: `{text(payload.get('policy_decision'))}`",
            f"- recommendation: `{text(payload.get('policy_recommendation'))}`",
            f"- queue_status: `{text(payload.get('queue_status'))}`",
            f"- execution contract: `{text(payload.get('execution_contract_brief'))}`",
            f"- feedback: `{text(payload.get('feedback_brief'))}`",
            f"- blocked_gates: `{','.join(as_list(payload.get('blocked_gate_names'))) or '-'}`",
            f"- risk_reasons: `{','.join(as_list(payload.get('risk_reason_codes'))) or '-'}`",
            f"- blocker: `{text(payload.get('blocker_detail'))}`",
            f"- done_when: `{text(payload.get('done_when'))}`",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build OpenClaw remote orderflow policy state.")
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--now", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    reference_now = parse_now(args.now)

    intent_path = find_latest(review_dir, "*_remote_intent_queue.json")
    feedback_path = find_latest(review_dir, "*_remote_orderflow_feedback.json")
    live_gate_path = find_latest(review_dir, "*_live_gate_blocker_report.json")
    scope_router_path = find_latest(review_dir, "*_remote_scope_router_state.json")
    missing = [
        name
        for name, path in (
            ("remote_intent_queue", intent_path),
            ("remote_orderflow_feedback", feedback_path),
            ("live_gate_blocker_report", live_gate_path),
        )
        if path is None
    ]
    if missing:
        raise SystemExit(f"missing_required_artifacts:{','.join(missing)}")

    payload = build_payload(
        intent_path=intent_path,
        intent_payload=load_json_mapping(intent_path),
        feedback_path=feedback_path,
        feedback_payload=load_json_mapping(feedback_path),
        live_gate_path=live_gate_path,
        live_gate_payload=load_json_mapping(live_gate_path),
        scope_router_path=scope_router_path,
        scope_router_payload=load_json_mapping(scope_router_path)
        if scope_router_path is not None and scope_router_path.exists()
        else {},
        reference_now=reference_now,
    )

    stamp = reference_now.strftime("%Y%m%dT%H%M%SZ")
    artifact = review_dir / f"{stamp}_remote_orderflow_policy_state.json"
    markdown = review_dir / f"{stamp}_remote_orderflow_policy_state.md"
    checksum = review_dir / f"{stamp}_remote_orderflow_policy_state_checksum.json"

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
        keep=max(1, int(args.artifact_keep)),
        ttl_hours=max(1.0, float(args.artifact_ttl_hours)),
    )
    payload.update(
        {
            "artifact": str(artifact),
            "markdown": str(markdown),
            "checksum": str(checksum),
            "pruned_keep": pruned_keep,
            "pruned_age": pruned_age,
        }
    )
    artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
