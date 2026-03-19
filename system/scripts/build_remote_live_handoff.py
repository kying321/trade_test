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


def load_payload_file(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    clean = text.strip()
    if not clean:
        return None
    try:
        payload = json.loads(clean)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass
    for line in reversed([x.strip() for x in clean.splitlines() if x.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def unwrap_review_input_payload(payload: dict[str, Any] | None) -> tuple[dict[str, Any] | None, int | None]:
    source = payload if isinstance(payload, dict) else None
    if not source:
        return None, None
    nested = source.get("payload")
    capture_kind = str(source.get("capture_kind") or source.get("input_kind") or "").strip()
    if capture_kind and isinstance(nested, dict):
        return dict(nested), _as_int(source.get("returncode"))
    return source, None


def resolve_review_input_artifact(
    *,
    review_dir: Path,
    explicit_path: Path | None,
    pattern: str | None = None,
) -> tuple[Path | None, dict[str, Any] | None, int | None]:
    path = explicit_path
    if path is None and pattern:
        path = find_latest_review_artifact(review_dir, pattern)
    payload = load_payload_file(path)
    unwrapped_payload, returncode = unwrap_review_input_payload(payload)
    return path, unwrapped_payload, returncode


def prune_handoffs(
    review_dir: Path,
    *,
    current_artifact: Path,
    current_checksum: Path,
    keep: int,
    ttl_hours: float,
) -> tuple[list[str], list[str]]:
    pruned_keep: list[str] = []
    pruned_age: list[str] = []
    review_dir.mkdir(parents=True, exist_ok=True)
    candidates = sorted(
        review_dir.glob("*_remote_live_handoff*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    cutoff = now_utc() - dt.timedelta(hours=max(1.0, ttl_hours))
    survivors: list[Path] = []
    protected = {current_artifact.name, current_checksum.name}
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
    artifact_like = [p for p in survivors if p.name.endswith(".json")]
    for path in artifact_like[keep:]:
        if path.name in protected:
            continue
        path.unlink(missing_ok=True)
        pruned_keep.append(str(path))
    return pruned_keep, pruned_age


def find_latest_review_artifact(review_dir: Path, pattern: str) -> Path | None:
    files = sorted(
        review_dir.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def _list_text(values: list[str], limit: int = 6) -> str:
    items = [str(v).strip() for v in values if str(v).strip()]
    if not items:
        return "-"
    if len(items) <= limit:
        return ", ".join(items)
    return ", ".join(items[:limit]) + f" (+{len(items) - limit})"


def _mapping_text(values: dict[str, Any], limit: int = 8) -> str:
    items = [
        f"{str(key).strip()}:{str(value).strip()}"
        for key, value in sorted((values or {}).items())
        if str(key).strip() and str(value).strip()
    ]
    if not items:
        return "-"
    if len(items) <= limit:
        return ", ".join(items)
    return ", ".join(items[:limit]) + f" (+{len(items) - limit})"


def _remote_live_history_window_map(payload: dict[str, Any] | None) -> dict[int, dict[str, Any]]:
    source = payload if isinstance(payload, dict) else {}
    rows = source.get("window_summaries")
    if not isinstance(rows, list):
        rows = []
    window_map: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            hours = int(row.get("window_hours") or 0)
        except Exception:
            continue
        if hours <= 0:
            continue
        window_map[hours] = dict(row)
    return window_map


def _remote_live_history_window_brief(row: dict[str, Any]) -> str:
    if not row:
        return "-"
    label = str(row.get("history_window_label") or "").strip()
    if not label:
        try:
            hours = int(row.get("window_hours") or 0)
        except Exception:
            hours = 0
        label = "24h" if hours == 24 else "7d" if hours == 168 else f"{int(hours // 24)}d" if hours else "-"
    return (
        f"{label}:{row.get('closed_pnl')}pnl/"
        f"{int(row.get('trade_count') or 0)}tr/"
        f"{int(row.get('open_positions') or 0)}open"
    )


def summarize_remote_live_history_audit(
    payload: dict[str, Any] | None,
    *,
    artifact_path: Path | None,
) -> dict[str, Any]:
    source = payload if isinstance(payload, dict) else {}
    if not source and artifact_path is None:
        return {}
    window_map = _remote_live_history_window_map(source)
    snapshot = (
        dict(window_map.get(24) or {})
        or dict(window_map.get(168) or {})
        or dict(window_map.get(720) or {})
    )
    longest = dict(window_map.get(max(window_map)) or {}) if window_map else {}
    blocked_candidate = (
        snapshot.get("blocked_candidate")
        if isinstance(snapshot.get("blocked_candidate"), dict)
        else {}
    )
    window_brief = " | ".join(
        [
            part
            for part in (
                _remote_live_history_window_brief(window_map.get(24, {})),
                _remote_live_history_window_brief(window_map.get(168, {})),
                _remote_live_history_window_brief(window_map.get(720, {})),
            )
            if part != "-"
        ]
    )
    return {
        "artifact": str(artifact_path) if artifact_path else "",
        "status": str(source.get("status") or ""),
        "market": str(source.get("market") or ""),
        "generated_at": str(source.get("generated_at_utc") or ""),
        "window_brief": window_brief,
        "quote_available": snapshot.get("quote_available"),
        "open_positions": snapshot.get("open_positions"),
        "risk_guard_status": str(snapshot.get("risk_guard_status") or ""),
        "risk_guard_reasons": list(snapshot.get("risk_guard_reasons") or []),
        "blocked_candidate_symbol": str(blocked_candidate.get("symbol") or ""),
        "pnl_24h": (window_map.get(24) or {}).get("closed_pnl"),
        "pnl_7d": (window_map.get(168) or {}).get("closed_pnl"),
        "pnl_30d": (window_map.get(720) or {}).get("closed_pnl"),
        "trade_count_24h": (window_map.get(24) or {}).get("trade_count"),
        "trade_count_7d": (window_map.get(168) or {}).get("trade_count"),
        "trade_count_30d": (window_map.get(720) or {}).get("trade_count"),
        "symbol_pnl_brief": _mapping_text(dict(longest.get("income_pnl_by_symbol") or {}), limit=8),
        "day_pnl_brief": _mapping_text(dict(longest.get("income_pnl_by_day") or {}), limit=8),
    }


def _positive_profitability_window(remote_live_history_summary: dict[str, Any]) -> tuple[str, float, int] | None:
    candidates = (
        ("30d", remote_live_history_summary.get("pnl_30d"), remote_live_history_summary.get("trade_count_30d")),
        ("7d", remote_live_history_summary.get("pnl_7d"), remote_live_history_summary.get("trade_count_7d")),
        ("24h", remote_live_history_summary.get("pnl_24h"), remote_live_history_summary.get("trade_count_24h")),
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


def derive_remote_live_diagnosis(
    *,
    remote_live_history_summary: dict[str, Any],
    account_scope_alignment: dict[str, Any],
    ready: bool,
    live_gate_ok: bool | None,
    risk_guard_attention_reasons: list[str],
) -> dict[str, Any]:
    remote_summary = remote_live_history_summary if isinstance(remote_live_history_summary, dict) else {}
    account_scope = account_scope_alignment if isinstance(account_scope_alignment, dict) else {}
    market = str(remote_summary.get("market") or "").strip() or "unknown"
    profitability_window = _positive_profitability_window(remote_summary)
    profitability_confirmed = profitability_window is not None
    blocking_layers: list[str] = []
    if live_gate_ok is False:
        blocking_layers.append("ops_live_gate")
    if risk_guard_attention_reasons:
        blocking_layers.append("risk_guard")
    if bool(account_scope.get("blocking", False)):
        blocking_layers.append("account_scope_alignment")

    if ready and not blocking_layers:
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


def summarize_ready_check_scope(payload: dict[str, Any] | None, *, fallback_market: str) -> dict[str, Any]:
    source = payload if isinstance(payload, dict) else {}
    market = str(source.get("market") or fallback_market).strip().lower() or fallback_market
    return {
        "market": market,
        "ready": bool(source.get("ready", False)),
        "reason": str(source.get("reason") or ""),
        "reasons": [str(x) for x in source.get("reasons", [])] if isinstance(source.get("reasons"), list) else [],
        "quote_available": source.get("quote_available"),
        "required_quote": source.get("required_quote"),
        "artifact": str(source.get("artifact") or ""),
        "takeover_artifact": str(source.get("takeover_artifact") or ""),
    }


def derive_account_scope_alignment(
    *,
    spot_ready_summary: dict[str, Any],
    portfolio_ready_summary: dict[str, Any],
    remote_live_history_summary: dict[str, Any],
) -> dict[str, Any]:
    remote_market = str(remote_live_history_summary.get("market") or "").strip().lower()
    spot_market = str(spot_ready_summary.get("market") or "spot").strip().lower()
    portfolio_market = str(portfolio_ready_summary.get("market") or "portfolio_margin_um").strip().lower()
    if not remote_market:
        return {
            "status": "history_market_missing",
            "brief": "history_market_missing",
            "blocking": False,
            "blocker_detail": "remote live history audit is missing market scope, so account-layer alignment cannot be verified.",
            "done_when": "refresh remote_live_history_audit and confirm the account market scope.",
        }
    if remote_market == portfolio_market and spot_market != remote_market:
        return {
            "status": "split_scope_spot_vs_portfolio_margin_um",
            "brief": "split_scope_spot_vs_portfolio_margin_um",
            "blocking": False,
            "blocker_detail": (
                "spot ready-check and portfolio_margin_um account history refer to different execution scopes; "
                "treat spot gate status and unified-account profitability as separate signals."
            ),
            "done_when": (
                "promote market-specific live routing policy or evaluate live readiness on the same market scope as the target account."
            ),
        }
    return {
        "status": "scope_aligned",
        "brief": f"scope_aligned:{remote_market}",
        "blocking": False,
        "blocker_detail": f"ready-check and account history both point to `{remote_market}`.",
        "done_when": "keep live readiness and account history on the same market scope.",
    }


def derive_execution_contract(
    *,
    ready_check_scope_market: str,
    ready_check_scope_source: str,
    openclaw_orderflow_executor_mode: str,
    openclaw_orderflow_executor_mode_source: str,
    account_scope_alignment: dict[str, Any] | None,
    remote_live_history_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    account_scope = account_scope_alignment if isinstance(account_scope_alignment, dict) else {}
    remote_summary = (
        remote_live_history_summary if isinstance(remote_live_history_summary, dict) else {}
    )
    target_market = str(ready_check_scope_market or "").strip().lower()
    target_source = str(ready_check_scope_source or "").strip().lower()
    executor_mode = str(openclaw_orderflow_executor_mode or "").strip().lower() or "shadow_guarded"
    executor_mode_source = (
        str(openclaw_orderflow_executor_mode_source or "").strip() or "default"
    )
    executable_lane_market = str(
        target_market or remote_summary.get("market") or "unknown"
    ).strip().lower()
    alignment_status = str(account_scope.get("status") or "").strip()
    reason_codes: list[str] = []
    guarded_probe_allowed = False
    if alignment_status == "split_scope_spot_vs_portfolio_margin_um" and target_market != "spot":
        reason_codes.append("spot_remote_lane_missing")
    if executable_lane_market == "portfolio_margin_um":
        reason_codes.append("portfolio_margin_um_read_only_mode")
    if executor_mode == "shadow_guarded":
        # Keep the default source-owned mode explicit instead of assuming every remote lane is
        # shadow-only forever.
        reason_codes.append("shadow_executor_only_mode")
        contract_mode = "shadow_only"
    elif executor_mode in {"spot_live_guarded", "live_guarded", "live_send_guarded"}:
        if target_market == "spot" and executable_lane_market == "spot":
            reason_codes.append("guarded_probe_only_mode")
            contract_mode = "guarded_probe_only"
            guarded_probe_allowed = True
        else:
            reason_codes.append("requested_executor_mode_not_implemented")
            contract_mode = "promotion_requested"
    else:
        reason_codes.append("unsupported_executor_mode_source")
        contract_mode = "unsupported_source_mode"
    deduped_reason_codes: list[str] = []
    seen: set[str] = set()
    for code in reason_codes:
        item = str(code or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        deduped_reason_codes.append(item)

    blocker_parts = [
        (
            "remote execution contract is probe-capable but still blocks live send"
            if guarded_probe_allowed
            else "remote execution contract remains non-executable"
        ),
        f"target={target_market or 'unknown'}",
        f"lane={executable_lane_market or 'unknown'}",
        f"scope={alignment_status or 'unknown'}",
        f"executor_mode={executor_mode}",
    ]
    if target_source:
        blocker_parts.append(f"source={target_source}")
    if guarded_probe_allowed and deduped_reason_codes == ["guarded_probe_only_mode"]:
        status = "probe_only_contract"
    else:
        status = "non_executable_contract" if deduped_reason_codes else "live_executable_contract"
    brief = ":".join(
        [
            status,
            target_market or "unknown",
            executable_lane_market or "unknown",
            ",".join(deduped_reason_codes) or "-",
        ]
    )
    return {
        "status": status,
        "brief": brief,
        "mode": contract_mode,
        "guarded_probe_allowed": guarded_probe_allowed,
        "live_orders_allowed": False if deduped_reason_codes else True,
        "target_market": target_market,
        "target_source": target_source,
        "executor_mode": executor_mode,
        "executor_mode_source": executor_mode_source,
        "executable_lane_market": executable_lane_market,
        "account_scope_alignment_status": alignment_status,
        "reason_codes": deduped_reason_codes,
        "blocker_detail": "; ".join(blocker_parts),
        "done_when": (
            "use guarded probe evidence for source-owned promotion checks, then implement a source-owned non-shadow "
            "remote send/ack/fill runtime on the target market before routing capital"
        ),
    }


def summarize_security_top_risks(findings: list[str]) -> list[str]:
    top: list[str] = []
    mapping = [
        ("RootDirectory=/RootImage=", "host_root_filesystem"),
        ("MemoryDenyWriteExecute=", "writable_executable_memory"),
        ("RestrictAddressFamilies=~AF_(INET|INET6)", "internet_socket_access"),
        ("RestrictAddressFamilies=~AF_UNIX", "local_socket_access"),
        ("ProtectProc=", "full_proc_visibility"),
        ("ProcSubset=", "full_proc_subset_visibility"),
        ("ProtectHome=", "home_directory_access"),
        ("PrivateNetwork=", "host_network_namespace"),
        ("PrivateUsers=", "shared_user_namespace"),
        ("DeviceAllow=", "device_acl_present"),
        ("IPAddressDeny=", "no_ip_allowlist"),
        ("SystemCallFilter=~@resources", "syscall_resources_allowed"),
        ("SystemCallFilter=~@privileged", "syscall_privileged_allowed"),
        ("SystemCallFilter=", "syscall_filter_not_set"),
    ]
    for line in findings:
        if "DeviceAllow=" in line and "char-rtc:r" in line and "rtc_read_allowed" not in top:
            top.append("rtc_read_allowed")
            if len(top) >= 5:
                break
            continue
        for needle, label in mapping:
            if needle in line and label not in top:
                top.append(label)
                break
        if len(top) >= 5:
            break
    return top


def build_security_recommendations(top_risks: list[str]) -> list[str]:
    mapping = {
        "host_root_filesystem": "If acceptable, review RootImage/RootDirectory isolation; otherwise keep ProtectSystem strict and document host-root dependency.",
        "writable_executable_memory": "Review whether MemoryDenyWriteExecute can be enabled without breaking Python/runtime dependencies.",
        "internet_socket_access": "Review whether RestrictAddressFamilies can drop AF_INET/AF_INET6 now that PrivateNetwork and IPAddressDeny are enabled.",
        "local_socket_access": "Review whether AF_UNIX is actually required beyond system logging and local runtime behavior.",
        "full_proc_visibility": "Review ProtectProc/ProcSubset hardening if daemon observability does not require full /proc access.",
        "full_proc_subset_visibility": "Review ProcSubset=pid if full /proc metadata is not required.",
        "home_directory_access": "Review ProtectHome after confirming project paths and runtime writes do not depend on broader home access.",
        "host_network_namespace": "Review whether PrivateNetwork can be enabled without breaking daemon isolation or future runtime expectations.",
        "shared_user_namespace": "Review PrivateUsers only if the runtime does not depend on host user namespace behavior.",
        "device_acl_present": "Check whether the remaining device ACL is platform-default only or can be further reduced safely.",
        "rtc_read_allowed": "ProtectClock implies read-only RTC access via DeviceAllow=char-rtc:r. Treat this as the tested floor unless you are prepared to remove ProtectClock.",
        "no_ip_allowlist": "Review whether IPAddressDeny plus a minimal allowlist can replace broad IP access if future runtime needs outbound connectivity.",
        "syscall_resources_allowed": "Review whether @resources is still required in SystemCallFilter after measuring runtime behavior under the current hardening profile.",
        "syscall_privileged_allowed": "Review whether @privileged can be narrowed further without breaking the daemon lifecycle or file ownership operations.",
        "syscall_filter_not_set": "Review a conservative SystemCallFilter only after runtime compatibility is measured in staging.",
    }
    return [mapping[risk] for risk in top_risks if risk in mapping][:5]


def derive_address_family_floor(
    noaf_probe_payload: dict[str, Any] | None,
) -> tuple[str, str | None, str | None]:
    payload = noaf_probe_payload if isinstance(noaf_probe_payload, dict) else {}
    status = str(payload.get("status") or "").strip() or None
    if status == "incompatible":
        return (
            "AF_UNIX",
            status,
            "Keep RestrictAddressFamilies=AF_UNIX. The latest transient RestrictAddressFamilies=none probe is incompatible on this host.",
        )
    if status == "compatible":
        return (
            "none-compatible",
            status,
            "RestrictAddressFamilies=none appears compatible in transient probe; formalize only after deliberate operator review.",
        )
    return ("unknown", status, None)


def derive_gate_status(
    *,
    ready: bool,
    ready_reasons: list[str],
    live_gate_ok: bool | None,
    live_gate_blocking_reason_codes: list[str],
    ops_status: str,
    ops_ok: bool | None,
    ops_reason_code: str,
) -> tuple[str, list[str]]:
    attention_reasons: list[str] = []
    if "ops_live_gate_blocked" in ready_reasons or live_gate_ok is False:
        attention_reasons.append("ops_live_gate_blocked")
        for code in live_gate_blocking_reason_codes:
            code_str = f"live_gate:{code}"
            if code_str not in attention_reasons:
                attention_reasons.append(code_str)
    elif ops_status in {"blocked", "stale"} or (ops_reason_code and ops_ok is False):
        attention_reasons.append("ops_reconcile_attention")
        if ops_reason_code:
            attention_reasons.append(f"ops:{ops_reason_code}")
    if ready and not attention_reasons:
        return "gate-ok", []
    if "ops_live_gate_blocked" in attention_reasons:
        return "gate-blocked", attention_reasons
    if attention_reasons:
        return "gate-review", attention_reasons
    return "gate-ok", []


def derive_risk_guard_status(
    *,
    ready: bool,
    ready_reasons: list[str],
    risk_guard_payload: dict[str, Any] | None,
) -> tuple[str, list[str]]:
    risk_guard_payload = risk_guard_payload if isinstance(risk_guard_payload, dict) else {}
    attention_reasons: list[str] = []
    risk_guard_allowed = risk_guard_payload.get("allowed")
    risk_guard_status = str(risk_guard_payload.get("status") or "")
    risk_guard_reasons = (
        [str(x) for x in risk_guard_payload.get("reasons", [])]
        if isinstance(risk_guard_payload.get("reasons"), list)
        else []
    )
    if "risk_guard_blocked" in ready_reasons or risk_guard_allowed is False or risk_guard_status == "blocked":
        attention_reasons.append("risk_guard_blocked")
        for reason in risk_guard_reasons:
            reason_str = f"risk_guard:{reason}"
            if reason_str not in attention_reasons:
                attention_reasons.append(reason_str)
        return "risk-guard-blocked", attention_reasons
    if risk_guard_payload:
        return "risk-guard-ok", []
    if ready:
        return "risk-guard-ok", []
    return "risk-guard-review", ["risk_guard_status_missing"]


def derive_notification_status(
    *,
    delivery_status: str | None,
    delivery_readiness_label: str | None,
    delivery_capabilities: dict[str, Any] | None,
) -> tuple[str, list[str]]:
    delivery_status = str(delivery_status or "").strip()
    readiness = str(delivery_readiness_label or "").strip()
    capabilities = delivery_capabilities if isinstance(delivery_capabilities, dict) else {}
    blocked_channels = (
        capabilities.get("blocked_channels")
        if isinstance(capabilities.get("blocked_channels"), dict)
        else {}
    )

    reasons: list[str] = []
    if readiness == "delivery-none" or delivery_status == "delivery_none":
        return "notify-disabled", []
    if readiness in {"telegram-ready", "feishu-ready", "all-ready"}:
        return "notify-ready", []
    if readiness == "partial-ready":
        return "notify-partial", ["notification_partial_ready"]
    if readiness in {"telegram-blocked", "feishu-blocked", "all-blocked", "credentials-missing"}:
        reasons.append("notification_blocked")
        for channel, channel_reasons in blocked_channels.items():
            if isinstance(channel_reasons, list):
                for reason in channel_reasons:
                    reason_str = f"{channel}:{reason}"
                    if reason_str not in reasons:
                        reasons.append(reason_str)
        if readiness:
            reasons.append(f"readiness:{readiness}")
        return "notify-blocked", reasons
    if readiness:
        return "notify-review", [f"readiness:{readiness}"]
    return "notify-unknown", ["notification_status_missing"]


def derive_secondary_focus(
    *,
    runtime_status_label: str,
    runtime_attention_reasons: list[str],
    gate_status_label: str,
    gate_attention_reasons: list[str],
    risk_guard_status_label: str,
    risk_guard_attention_reasons: list[str],
    notification_status_label: str,
    notification_attention_reasons: list[str],
    primary_focus_area: str,
) -> tuple[str | None, str | None]:
    candidates: list[tuple[str, str | None]] = []
    if runtime_status_label != "runtime-ok" and primary_focus_area != "runtime":
        candidates.append(
            ("runtime", runtime_attention_reasons[0] if runtime_attention_reasons else runtime_status_label)
        )
    if gate_status_label != "gate-ok" and primary_focus_area != "gate":
        candidates.append(
            ("gate", gate_attention_reasons[0] if gate_attention_reasons else gate_status_label)
        )
    if risk_guard_status_label != "risk-guard-ok" and primary_focus_area != "risk_guard":
        candidates.append(
            (
                "risk_guard",
                risk_guard_attention_reasons[0]
                if risk_guard_attention_reasons
                else risk_guard_status_label,
            )
        )
    if (
        notification_status_label in {"notify-blocked", "notify-partial", "notify-review", "notify-unknown"}
        and primary_focus_area != "notify"
    ):
        candidates.append(
            (
                "notify",
                notification_attention_reasons[0]
                if notification_attention_reasons
                else notification_status_label,
            )
    )
    return candidates[0] if candidates else (None, None)


def build_focus_stack(
    *,
    next_focus_area: str,
    next_focus_reason: str,
    secondary_focus_area: str | None,
    secondary_focus_reason: str | None,
) -> list[dict[str, str]]:
    stack: list[dict[str, str]] = []
    if next_focus_area:
        stack.append({"area": next_focus_area, "reason": next_focus_reason})
    if secondary_focus_area and secondary_focus_area != next_focus_area:
        stack.append(
            {
                "area": secondary_focus_area,
                "reason": str(secondary_focus_reason or ""),
            }
        )
    return stack


def build_focus_stack_brief(focus_stack: list[dict[str, str]]) -> str:
    areas = [str(item.get("area") or "").strip() for item in focus_stack]
    return " -> ".join(area for area in areas if area)


def build_focus_stack_summary(focus_stack: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for item in focus_stack:
        area = str(item.get("area") or "").strip()
        reason = str(item.get("reason") or "").strip()
        if not area:
            continue
        parts.append(f"{area}({reason})" if reason else area)
    return " -> ".join(parts)


def derive_security_acceptance(
    *,
    verify_ok: bool | None,
    security_returncode: int | None,
    exposure_score: float | None,
    exposure_rating: str | None,
    max_allowed_exposure: float,
) -> dict[str, Any]:
    reasons: list[str] = []
    if verify_ok is False:
        status = "failed"
        reasons.append("systemd_verify_failed")
    elif security_returncode not in (None, 0):
        status = "review"
        reasons.append("systemd_security_probe_failed")
    elif exposure_score is None:
        status = "review"
        reasons.append("security_exposure_missing")
    elif exposure_score > max_allowed_exposure:
        status = "review"
        reasons.append(
            f"security_exposure_above_threshold({exposure_score:.1f}>{max_allowed_exposure:.1f})"
        )
    else:
        status = "accepted"
    return {
        "status": status,
        "max_allowed_exposure": max_allowed_exposure,
        "observed_exposure": exposure_score,
        "observed_rating": exposure_rating,
        "reasons": reasons,
    }


def derive_next_focus_command(
    *,
    system_root: Path,
    next_focus_area: str,
    next_focus_reason: str,
) -> str:
    prefix = f"cd {system_root} && scripts/openclaw_cloud_bridge.sh"
    if next_focus_area == "runtime":
        if next_focus_reason == "daemon_not_running":
            return f"{prefix} live-risk-daemon-start"
        if next_focus_reason in {"security_acceptance_failed", "security_acceptance_review"}:
            return f"{prefix} live-risk-daemon-security-status"
        return f"{prefix} live-risk-daemon-status"
    if next_focus_area == "gate":
        return f"{prefix} live-ops-reconcile-status"
    if next_focus_area == "risk_guard":
        return f"{prefix} live-takeover-ready-check"
    if next_focus_area == "notify":
        return f"{prefix} remote-live-notification-send"
    return f"{prefix} live-takeover-probe"


def derive_next_focus_commands(
    *,
    system_root: Path,
    next_focus_area: str,
    next_focus_reason: str,
) -> list[str]:
    prefix = f"cd {system_root} && scripts/openclaw_cloud_bridge.sh"
    if next_focus_area == "runtime":
        if next_focus_reason == "daemon_not_running":
            return [
                f"{prefix} live-risk-daemon-start",
                f"{prefix} live-risk-daemon-status",
                f"{prefix} live-risk-daemon-journal",
            ]
        if next_focus_reason in {"security_acceptance_failed", "security_acceptance_review"}:
            return [
                f"{prefix} live-risk-daemon-security-status",
                f"{prefix} live-risk-daemon-status",
                f"{prefix} remote-live-handoff",
            ]
        return [
            f"{prefix} live-risk-daemon-status",
            f"{prefix} live-risk-daemon-journal",
            f"{prefix} remote-live-handoff",
        ]
    if next_focus_area == "gate":
        return [
            f"{prefix} live-ops-reconcile-status",
            f"{prefix} live-ops-reconcile-refresh",
            f"{prefix} remote-live-handoff",
        ]
    if next_focus_area == "risk_guard":
        return [
            f"{prefix} live-takeover-ready-check",
            f"{prefix} live-risk-daemon-status",
            f"{prefix} remote-live-handoff",
        ]
    if next_focus_area == "notify":
        return [
            f"{prefix} remote-live-notification-preview",
            f"{prefix} remote-live-notification-dry-run",
            f"{prefix} remote-live-notification-send",
        ]
    return [
        f"{prefix} live-takeover-probe",
        f"{prefix} live-takeover-canary",
        f"{prefix} remote-live-handoff",
    ]


def summarize_playbook_section(
    *,
    area: str,
    status_label: str,
    attention_reasons: list[str],
) -> str:
    if area == "runtime":
        if status_label == "runtime-ok":
            return "Daemon runtime, payload alignment, and security acceptance are healthy."
        if "daemon_not_running" in attention_reasons:
            return "Restore daemon service state before investigating gate or risk layers."
        if "daemon_payload_unaligned" in attention_reasons:
            return "Wait for daemon payload to align with systemd state or inspect journal lag."
        return "Runtime needs operator review before making service-level changes."
    if area == "gate":
        if status_label == "gate-ok":
            return "Ops live gate is not the current blocker."
        if "ops_live_gate_blocked" in attention_reasons:
            return "Ops live gate is the primary blocker for remote canary execution."
        return "Ops reconcile artifacts need review before attempting live actions."
    if area == "risk_guard":
        if status_label == "risk-guard-ok":
            return "Risk guard is not currently blocking the execution path."
        if "risk_guard_blocked" in attention_reasons:
            return "Risk guard is actively blocking execution and should be reviewed after gate blockers."
        return "Risk guard state is incomplete and should be rechecked."
    return "Runtime and gates are clear; proceed with guarded canary validation."


def build_operator_playbook(
    *,
    system_root: Path,
    runtime_status_label: str,
    runtime_attention_reasons: list[str],
    gate_status_label: str,
    gate_attention_reasons: list[str],
    risk_guard_status_label: str,
    risk_guard_attention_reasons: list[str],
    address_family_floor: str,
    address_family_recommendation: str | None,
    security_top_risks: list[str],
    security_recommendations: list[str],
    next_focus_area: str,
    next_focus_reason: str,
    next_focus_commands: list[str],
) -> dict[str, Any]:
    runtime_reason = (
        runtime_attention_reasons[0] if runtime_attention_reasons else runtime_status_label
    )
    gate_reason = gate_attention_reasons[0] if gate_attention_reasons else gate_status_label
    risk_guard_reason = (
        risk_guard_attention_reasons[0]
        if risk_guard_attention_reasons
        else risk_guard_status_label
    )
    clock_device_floor = "rtc-read-only" if "rtc_read_allowed" in security_top_risks else ""
    clock_device_note = (
        next(
            (
                recommendation
                for recommendation in security_recommendations
                if "ProtectClock implies read-only RTC access" in recommendation
            ),
            "",
        )
        if clock_device_floor
        else ""
    )
    sections = {
        "runtime": {
            "status_label": runtime_status_label,
            "attention_reasons": runtime_attention_reasons,
            "summary": summarize_playbook_section(
                area="runtime",
                status_label=runtime_status_label,
                attention_reasons=runtime_attention_reasons,
            ),
            "commands": derive_next_focus_commands(
                system_root=system_root,
                next_focus_area="runtime",
                next_focus_reason=runtime_reason,
            ),
            "address_family_floor": address_family_floor,
            "address_family_note": address_family_recommendation,
            "clock_device_floor": clock_device_floor,
            "clock_device_note": clock_device_note,
        },
        "gate": {
            "status_label": gate_status_label,
            "attention_reasons": gate_attention_reasons,
            "summary": summarize_playbook_section(
                area="gate",
                status_label=gate_status_label,
                attention_reasons=gate_attention_reasons,
            ),
            "commands": derive_next_focus_commands(
                system_root=system_root,
                next_focus_area="gate",
                next_focus_reason=gate_reason,
            ),
        },
        "risk_guard": {
            "status_label": risk_guard_status_label,
            "attention_reasons": risk_guard_attention_reasons,
            "summary": summarize_playbook_section(
                area="risk_guard",
                status_label=risk_guard_status_label,
                attention_reasons=risk_guard_attention_reasons,
            ),
            "commands": derive_next_focus_commands(
                system_root=system_root,
                next_focus_area="risk_guard",
                next_focus_reason=risk_guard_reason,
            ),
        },
        "canary": {
            "status_label": "canary-ready"
            if runtime_status_label == "runtime-ok"
            and gate_status_label == "gate-ok"
            and risk_guard_status_label == "risk-guard-ok"
            else "canary-blocked",
            "attention_reasons": [],
            "summary": summarize_playbook_section(
                area="canary",
                status_label="canary-ready",
                attention_reasons=[],
            ),
            "commands": derive_next_focus_commands(
                system_root=system_root,
                next_focus_area="canary",
                next_focus_reason="ready_for_canary",
            ),
        },
    }
    return {
        "active_focus": next_focus_area,
        "active_reason": next_focus_reason,
        "primary_sequence": next_focus_commands,
        "sections": sections,
    }


def build_operator_playbook_md(playbook: dict[str, Any]) -> str:
    active_focus = str(playbook.get("active_focus") or "")
    active_reason = str(playbook.get("active_reason") or "")
    primary_sequence = (
        [str(x) for x in playbook.get("primary_sequence", [])]
        if isinstance(playbook.get("primary_sequence"), list)
        else []
    )
    sections = playbook.get("sections")
    sections = sections if isinstance(sections, dict) else {}
    lines = [
        "## Active Focus",
        f"- focus: `{active_focus}`",
        f"- reason: `{active_reason}`",
        "",
        "## Primary Sequence",
    ]
    if primary_sequence:
        for idx, command in enumerate(primary_sequence, start=1):
            lines.append(f"{idx}. `{command}`")
    else:
        lines.append("- no recommended commands")
    lines.extend(["", "## Sections"])
    for area in ("runtime", "gate", "risk_guard", "canary"):
        section = sections.get(area)
        section = section if isinstance(section, dict) else {}
        status_label = str(section.get("status_label") or "")
        summary = str(section.get("summary") or "")
        address_family_floor = str(section.get("address_family_floor") or "")
        address_family_note = str(section.get("address_family_note") or "")
        clock_device_floor = str(section.get("clock_device_floor") or "")
        clock_device_note = str(section.get("clock_device_note") or "")
        attention_reasons = (
            [str(x) for x in section.get("attention_reasons", [])]
            if isinstance(section.get("attention_reasons"), list)
            else []
        )
        lines.append(f"- {area}: `{status_label}`")
        if summary:
            lines.append(f"  summary: {summary}")
        if address_family_floor:
            lines.append(f"  address-family floor: `{address_family_floor}`")
        if address_family_note:
            lines.append(f"  address-family note: {address_family_note}")
        if clock_device_floor:
            lines.append(f"  clock-device floor: `{clock_device_floor}`")
        if clock_device_note:
            lines.append(f"  clock-device note: {clock_device_note}")
        if attention_reasons:
            lines.append(
                "  reasons: " + ", ".join(f"`{reason}`" for reason in attention_reasons)
            )
    return "\n".join(lines)


def build_operator_handoff_md(
    *,
    handoff_state: str,
    summary: str,
    recommended_action: str,
    operator_status_triplet: str,
    focus_stack_brief: str | None,
    focus_stack_summary: str | None,
    secondary_focus_area: str | None,
    secondary_focus_reason: str | None,
    next_focus_area: str,
    next_focus_reason: str,
    security_status_label: str,
    security_acceptance_status: str,
    security_exposure_score: float | None,
    security_top_risks: list[str],
    address_family_floor: str,
    address_family_recommendation: str | None,
    clock_device_floor: str,
    clock_device_note: str | None,
    operator_status_quad: str,
    notification_delivery_readiness_label: str | None,
    remote_live_history_summary: dict[str, Any] | None,
    remote_live_diagnosis: dict[str, Any] | None,
    ready_check_scope_brief: str | None,
    account_scope_alignment: dict[str, Any] | None,
    operator_playbook_md: str,
) -> str:
    security_score = (
        f"{security_exposure_score:.1f}" if isinstance(security_exposure_score, float) else "n/a"
    )
    top_risks = ", ".join(f"`{risk}`" for risk in security_top_risks) if security_top_risks else "none"
    lines = [
        "# Remote Live Handoff",
        f"- state: `{handoff_state}`",
        f"- status: `{operator_status_triplet}`",
        f"- status-extended: `{operator_status_quad}`",
        f"- summary: {summary}",
        *( [f"- focus-stack: `{focus_stack_brief}`"] if focus_stack_brief else [] ),
        *( [f"- focus-stack summary: `{focus_stack_summary}`"] if focus_stack_summary else [] ),
        f"- next focus: `{next_focus_area}`",
        f"- next reason: `{next_focus_reason}`",
        f"- security: `{security_status_label}` / acceptance=`{security_acceptance_status}` / exposure=`{security_score}`",
        f"- address-family floor: `{address_family_floor}`",
        f"- security top risks: {top_risks}",
        "",
        "## Recommended Action",
        recommended_action,
        "",
        operator_playbook_md,
    ]
    remote_summary = remote_live_history_summary if isinstance(remote_live_history_summary, dict) else {}
    remote_diagnosis = remote_live_diagnosis if isinstance(remote_live_diagnosis, dict) else {}
    account_scope = account_scope_alignment if isinstance(account_scope_alignment, dict) else {}
    if remote_summary:
        lines.insert(
            10,
            "- remote-live-history: "
            + f"`{remote_summary.get('window_brief') or '-'} | "
            + f"risk_guard={remote_summary.get('risk_guard_status') or '-'} | "
            + f"reasons={_list_text(remote_summary.get('risk_guard_reasons') or [])}`",
        )
    if remote_diagnosis:
        lines.insert(
            10,
            "- remote-live-diagnosis: "
            + f"`{remote_diagnosis.get('brief') or '-'} | "
            + f"blocker={remote_diagnosis.get('blocker_detail') or '-'} | "
            + f"done_when={remote_diagnosis.get('done_when') or '-'}`",
        )
    if ready_check_scope_brief:
        lines.insert(10, f"- ready-check scope: `{ready_check_scope_brief}`")
    if account_scope:
        lines.insert(11, f"- account-scope: `{account_scope.get('brief') or '-'}`")
    if secondary_focus_area:
        lines.insert(7, f"- secondary focus: `{secondary_focus_area}`")
        if secondary_focus_reason:
            lines.insert(8, f"- secondary reason: `{secondary_focus_reason}`")
    if notification_delivery_readiness_label:
        lines.insert(7, f"- notify: `{notification_delivery_readiness_label}`")
    if address_family_recommendation:
        lines.insert(8, f"- address-family note: {address_family_recommendation}")
    if clock_device_floor:
        insert_at = 9 if address_family_recommendation else 8
        lines.insert(insert_at, f"- clock-device floor: `{clock_device_floor}`")
        if clock_device_note:
            lines.insert(insert_at + 1, f"- clock-device note: {clock_device_note}")
    return "\n".join(lines)


def build_operator_handoff_brief(
    *,
    handoff_state: str,
    operator_status_triplet: str,
    operator_status_quad: str,
    focus_stack_brief: str | None,
    address_family_floor: str,
    clock_device_floor: str,
    notification_delivery_readiness_label: str | None,
    next_focus_area: str,
    next_focus_reason: str,
    next_focus_command: str,
    secondary_focus_area: str | None,
    secondary_focus_reason: str | None,
    remote_live_history_summary: dict[str, Any] | None,
    remote_live_diagnosis: dict[str, Any] | None,
    ready_check_scope_brief: str | None,
    account_scope_alignment: dict[str, Any] | None,
) -> str:
    lines = [
        f"state: {handoff_state}",
        f"status: {operator_status_triplet}",
        f"status4: {operator_status_quad}",
        *( [f"focus-stack: {focus_stack_brief}"] if focus_stack_brief else [] ),
        f"addrfam: {address_family_floor}",
    ]
    if clock_device_floor:
        lines.append(f"clockdev: {clock_device_floor}")
    if notification_delivery_readiness_label:
        lines.append(f"notify: {notification_delivery_readiness_label}")
    remote_summary = remote_live_history_summary if isinstance(remote_live_history_summary, dict) else {}
    remote_diagnosis = remote_live_diagnosis if isinstance(remote_live_diagnosis, dict) else {}
    account_scope = account_scope_alignment if isinstance(account_scope_alignment, dict) else {}
    if remote_summary:
        lines.append(f"history: {remote_summary.get('window_brief') or '-'}")
        lines.append(
            "history-risk: "
            + f"{remote_summary.get('risk_guard_status') or '-'}:"
            + _list_text(remote_summary.get("risk_guard_reasons") or [], limit=8)
        )
    if remote_diagnosis:
        lines.append(f"history-diagnosis: {remote_diagnosis.get('brief') or '-'}")
    if ready_check_scope_brief:
        lines.append(f"scope: {ready_check_scope_brief}")
    if account_scope:
        lines.append(f"scope-align: {account_scope.get('brief') or '-'}")
    lines.extend(
        [
            f"focus: {next_focus_area}",
            f"reason: {next_focus_reason}",
            f"cmd: {next_focus_command}",
        ]
    )
    if secondary_focus_area:
        lines.append(f"focus2: {secondary_focus_area}")
        if secondary_focus_reason:
            lines.append(f"reason2: {secondary_focus_reason}")
    return "\n".join(lines)


def build_operator_notification(
    *,
    handoff_state: str,
    operator_status_triplet: str,
    focus_stack_brief: str | None,
    address_family_floor: str,
    clock_device_floor: str,
    next_focus_area: str,
    next_focus_reason: str,
    next_focus_command: str,
    security_status_label: str,
    ready: bool,
) -> dict[str, Any]:
    if ready:
        level = "info"
        title = "Remote live ready for canary"
    elif next_focus_area == "runtime":
        level = "critical"
        title = "Remote runtime needs attention"
    elif next_focus_area == "gate":
        level = "warning"
        title = "Remote gate is blocking live execution"
    elif next_focus_area == "risk_guard":
        level = "warning"
        title = "Remote risk guard is blocking execution"
    else:
        level = "info"
        title = "Remote live review required"
    tags = [
        f"state:{handoff_state}",
        f"focus:{next_focus_area}",
        f"security:{security_status_label}",
    ]
    runtime_floor_parts: list[str] = []
    if address_family_floor and address_family_floor != "unknown":
        tags.append(f"addrfam:{address_family_floor}")
        runtime_floor_parts.append(f"addrfam={address_family_floor}")
    if clock_device_floor:
        tags.append(f"clockdev:{clock_device_floor}")
        runtime_floor_parts.append(f"clockdev={clock_device_floor}")
    runtime_floor_brief = "; ".join(runtime_floor_parts)
    body_parts = [
        operator_status_triplet,
        f"focus={next_focus_area}",
        f"reason={next_focus_reason}",
        f"addrfam={address_family_floor}",
    ]
    if focus_stack_brief:
        body_parts.append(f"focusstack={focus_stack_brief}")
    if clock_device_floor:
        body_parts.append(f"clockdev={clock_device_floor}")
    body_parts.append(f"cmd={next_focus_command}")
    body = "; ".join(body_parts)
    plain_text = "\n".join(
        [
            f"[{level.upper()}] {title}",
            body,
        ]
    )
    markdown = "\n".join(
        [
            f"**{title}**",
            f"- level: `{level}`",
            f"- state: `{handoff_state}`",
            f"- status: `{operator_status_triplet}`",
            *([f"- focus-stack: `{focus_stack_brief}`"] if focus_stack_brief else []),
            f"- focus: `{next_focus_area}`",
            f"- reason: `{next_focus_reason}`",
            f"- command: `{next_focus_command}`",
        ]
    )
    return {
        "level": level,
        "title": title,
        "body": body,
        "command": next_focus_command,
        "tags": tags,
        "focus_stack_brief": str(focus_stack_brief or ""),
        "runtime_floor_brief": runtime_floor_brief,
        "plain_text": plain_text,
        "markdown": markdown,
    }


def escape_telegram_markdown_v2(text: str) -> str:
    escaped = []
    for char in text:
        if char in r"_*[]()~`>#+-=|{}.!":
            escaped.append("\\" + char)
        else:
            escaped.append(char)
    return "".join(escaped)


def build_operator_notification_templates(
    notification: dict[str, Any],
) -> dict[str, Any]:
    level = str(notification.get("level") or "")
    title = str(notification.get("title") or "")
    body = str(notification.get("body") or "")
    command = str(notification.get("command") or "")
    focus_stack_brief = str(notification.get("focus_stack_brief") or "")
    runtime_floor_brief = str(notification.get("runtime_floor_brief") or "")
    plain_text = str(notification.get("plain_text") or "")
    markdown = str(notification.get("markdown") or "")
    tags = (
        [str(x) for x in notification.get("tags", [])]
        if isinstance(notification.get("tags"), list)
        else []
    )
    telegram_lines = [
        f"*{escape_telegram_markdown_v2(title)}*",
        f"level: `{escape_telegram_markdown_v2(level)}`",
        escape_telegram_markdown_v2(body),
        f"cmd: `{escape_telegram_markdown_v2(command)}`",
    ]
    if focus_stack_brief:
        telegram_lines.append(
            f"focus-stack: `{escape_telegram_markdown_v2(focus_stack_brief)}`"
        )
    if runtime_floor_brief:
        telegram_lines.append(
            f"runtime-floor: `{escape_telegram_markdown_v2(runtime_floor_brief)}`"
        )
    feishu_text = plain_text
    if focus_stack_brief:
        feishu_text = f"{feishu_text}\nfocus-stack: {focus_stack_brief}"
    if runtime_floor_brief:
        feishu_text = f"{feishu_text}\nruntime-floor: {runtime_floor_brief}"
    return {
        "telegram": {
            "parse_mode": "MarkdownV2",
            "text": "\n".join(telegram_lines),
            "disable_web_page_preview": True,
        },
        "feishu": {
            "msg_type": "text",
            "content": {
                "text": feishu_text,
            },
        },
        "generic": {
            "level": level,
            "title": title,
            "body": body,
            "command": command,
            "tags": tags,
            "focus_stack_brief": focus_stack_brief,
            "runtime_floor_brief": runtime_floor_brief,
        },
    }


def build_operator_handoff(
    *,
    remote_host: str,
    remote_user: str,
    remote_project_dir: str,
    ready_payload: dict[str, Any] | None,
    ready_payload_spot: dict[str, Any] | None,
    ready_payload_portfolio_margin_um: dict[str, Any] | None,
    daemon_payload: dict[str, Any] | None,
    ops_payload: dict[str, Any] | None,
    journal_payload: dict[str, Any] | None,
    security_payload: dict[str, Any] | None,
    noaf_probe_payload: dict[str, Any] | None,
    notification_send_payload: dict[str, Any] | None,
    remote_live_history_payload: dict[str, Any] | None,
    remote_live_history_artifact: Path | None,
    security_accept_max_exposure: float,
    target_live_takeover_market: str,
    openclaw_orderflow_executor_mode: str,
    openclaw_orderflow_executor_mode_source: str,
) -> dict[str, Any]:
    ready_payload = ready_payload if isinstance(ready_payload, dict) else {}
    ready_payload_spot = (
        ready_payload_spot if isinstance(ready_payload_spot, dict) else {}
    )
    ready_payload_portfolio_margin_um = (
        ready_payload_portfolio_margin_um
        if isinstance(ready_payload_portfolio_margin_um, dict)
        else {}
    )
    daemon_payload = daemon_payload if isinstance(daemon_payload, dict) else {}
    ops_payload = ops_payload if isinstance(ops_payload, dict) else {}
    journal_payload = journal_payload if isinstance(journal_payload, dict) else {}
    security_payload = security_payload if isinstance(security_payload, dict) else {}
    noaf_probe_payload = noaf_probe_payload if isinstance(noaf_probe_payload, dict) else {}
    notification_send_payload = (
        notification_send_payload if isinstance(notification_send_payload, dict) else {}
    )
    remote_live_history_summary = summarize_remote_live_history_audit(
        remote_live_history_payload,
        artifact_path=remote_live_history_artifact,
    )
    ready_check_spot_summary = (
        summarize_ready_check_scope(ready_payload_spot, fallback_market="spot")
        if ready_payload_spot
        else {}
    )
    ready_check_portfolio_summary = (
        summarize_ready_check_scope(
            ready_payload_portfolio_margin_um,
            fallback_market="portfolio_margin_um",
        )
        if ready_payload_portfolio_margin_um
        else {}
    )
    account_scope_alignment = derive_account_scope_alignment(
        spot_ready_summary=ready_check_spot_summary,
        portfolio_ready_summary=ready_check_portfolio_summary,
        remote_live_history_summary=remote_live_history_summary,
    )
    target_market = str(target_live_takeover_market or "").strip().lower()
    use_explicit_ready_scope = bool(ready_payload_spot or ready_payload_portfolio_margin_um)
    effective_ready_scope_market = ""
    effective_ready_scope_source = "legacy"
    remote_history_market = str(remote_live_history_summary.get("market") or "").strip().lower()
    if target_market == "spot" and ready_payload_spot:
        ready_payload = ready_payload_spot
        effective_ready_scope_market = "spot"
        effective_ready_scope_source = "spot"
    elif target_market == "portfolio_margin_um" and ready_payload_portfolio_margin_um:
        ready_payload = ready_payload_portfolio_margin_um
        effective_ready_scope_market = "portfolio_margin_um"
        effective_ready_scope_source = "portfolio_margin_um"
    elif remote_history_market == "portfolio_margin_um" and ready_payload_portfolio_margin_um:
        ready_payload = ready_payload_portfolio_margin_um
        effective_ready_scope_market = "portfolio_margin_um"
        effective_ready_scope_source = "portfolio_margin_um"
    elif remote_history_market == "spot" and ready_payload_spot:
        ready_payload = ready_payload_spot
        effective_ready_scope_market = "spot"
        effective_ready_scope_source = "spot"
    elif ready_payload:
        effective_ready_scope_market = (
            str(ready_payload.get("market") or "").strip().lower() or "spot"
        )
    elif ready_payload_spot:
        ready_payload = ready_payload_spot
        effective_ready_scope_market = "spot"
        effective_ready_scope_source = "spot"
    elif ready_payload_portfolio_margin_um:
        ready_payload = ready_payload_portfolio_margin_um
        effective_ready_scope_market = "portfolio_margin_um"
        effective_ready_scope_source = "portfolio_margin_um"
    else:
        effective_ready_scope_market = "missing"
        effective_ready_scope_source = "missing"
    ready_check_scope_brief = (
        f"{effective_ready_scope_source}:{effective_ready_scope_market}"
        if use_explicit_ready_scope and effective_ready_scope_market and effective_ready_scope_source
        else ""
    )
    account_scope_alignment_output = account_scope_alignment if use_explicit_ready_scope else {}
    execution_contract = derive_execution_contract(
        ready_check_scope_market=effective_ready_scope_market,
        ready_check_scope_source=effective_ready_scope_source,
        openclaw_orderflow_executor_mode=openclaw_orderflow_executor_mode,
        openclaw_orderflow_executor_mode_source=openclaw_orderflow_executor_mode_source,
        account_scope_alignment=account_scope_alignment_output,
        remote_live_history_summary=remote_live_history_summary,
    )

    reasons = ready_payload.get("reasons")
    ready_reasons = reasons if isinstance(reasons, list) else []
    ready = bool(ready_payload.get("ready", False))
    risk_daemon_state = daemon_payload.get("payload")
    risk_daemon_state = risk_daemon_state if isinstance(risk_daemon_state, dict) else {}
    systemd = daemon_payload.get("systemd")
    systemd = systemd if isinstance(systemd, dict) else {}
    daemon_payload_alignment = daemon_payload.get("payload_alignment")
    daemon_payload_alignment = (
        daemon_payload_alignment if isinstance(daemon_payload_alignment, dict) else {}
    )
    risk_guard_payload = ready_payload.get("risk_guard")
    risk_guard_payload = (
        risk_guard_payload if isinstance(risk_guard_payload, dict) else {}
    )
    live_gate = ready_payload.get("ops_live_gate")
    if not isinstance(live_gate, dict):
        live_gate = ops_payload.get("live_gate") if isinstance(ops_payload.get("live_gate"), dict) else {}
    daemon_running = bool(risk_daemon_state.get("running", False)) or str(
        systemd.get("active_state", "")
    ).lower() == "active"
    daemon_payload_alignment_ok = (
        bool(daemon_payload_alignment.get("aligned"))
        if daemon_payload_alignment
        else None
    )
    daemon_payload_alignment_reasons = (
        [str(x) for x in daemon_payload_alignment.get("reasons", [])]
        if isinstance(daemon_payload_alignment.get("reasons"), list)
        else []
    )
    daemon_mode = str(daemon_payload.get("mode") or "")
    daemon_status = str(risk_daemon_state.get("status") or "")
    ops_status = str(ops_payload.get("status") or "")
    ops_reason_code = str(ops_payload.get("reason_code") or "")
    ops_ok = (
        bool(ops_payload.get("ok"))
        if ops_payload.get("ok") is not None
        else None
    )
    live_gate_ok = bool(live_gate.get("ok", False)) if live_gate else None
    live_gate_blocking_reason_codes = (
        [str(x) for x in live_gate.get("blocking_reason_codes", [])]
        if isinstance(live_gate.get("blocking_reason_codes"), list)
        else []
    )
    journal_lines = journal_payload.get("lines")
    journal_tail = (
        [str(line) for line in journal_lines[-5:]]
        if isinstance(journal_lines, list)
        else []
    )
    security_verify = security_payload.get("verify")
    security_verify = security_verify if isinstance(security_verify, dict) else {}
    security = security_payload.get("security")
    security = security if isinstance(security, dict) else {}
    security_acceptance = security_payload.get("security_acceptance")
    security_acceptance = (
        security_acceptance if isinstance(security_acceptance, dict) else {}
    )
    security_attention_reasons: list[str] = []
    security_verify_ok = bool(security_verify.get("ok", False)) if security_verify else None
    try:
        security_exposure_score = (
            float(security.get("overall_exposure"))
            if security.get("overall_exposure") is not None
            else None
        )
    except (TypeError, ValueError):
        security_exposure_score = None
    security_exposure_rating = (
        str(security.get("overall_rating", "")).strip() or None
    )
    security_findings = (
        [str(x) for x in security.get("findings", [])]
        if isinstance(security.get("findings"), list)
        else []
    )
    try:
        security_returncode = (
            int(security.get("returncode"))
            if security.get("returncode") is not None
            else None
        )
    except (TypeError, ValueError):
        security_returncode = None
    security_top_risks = summarize_security_top_risks(security_findings)
    security_recommendations = build_security_recommendations(security_top_risks)
    (
        address_family_floor,
        address_family_probe_status,
        address_family_recommendation,
    ) = derive_address_family_floor(noaf_probe_payload)
    notification_delivery_status = (
        str(notification_send_payload.get("status") or "").strip() or None
    )
    notification_delivery_readiness_label = (
        str(notification_send_payload.get("delivery_readiness_label") or "").strip() or None
    )
    notification_delivery_capabilities = (
        notification_send_payload.get("delivery_capabilities")
        if isinstance(notification_send_payload.get("delivery_capabilities"), dict)
        else None
    )
    notification_status_label, notification_attention_reasons = derive_notification_status(
        delivery_status=notification_delivery_status,
        delivery_readiness_label=notification_delivery_readiness_label,
        delivery_capabilities=notification_delivery_capabilities,
    )
    if security_acceptance:
        acceptance_status = str(security_acceptance.get("status", "")).strip() or "review"
        acceptance_reasons = (
            [str(x) for x in security_acceptance.get("reasons", [])]
            if isinstance(security_acceptance.get("reasons"), list)
            else []
        )
        try:
            acceptance_max_allowed = (
                float(security_acceptance.get("max_allowed_exposure"))
                if security_acceptance.get("max_allowed_exposure") is not None
                else float(security_accept_max_exposure)
            )
        except (TypeError, ValueError):
            acceptance_max_allowed = float(security_accept_max_exposure)
    else:
        derived_acceptance = derive_security_acceptance(
            verify_ok=security_verify_ok,
            security_returncode=security_returncode,
            exposure_score=security_exposure_score,
            exposure_rating=security_exposure_rating,
            max_allowed_exposure=float(security_accept_max_exposure),
        )
        acceptance_status = str(derived_acceptance["status"])
        acceptance_reasons = [str(x) for x in derived_acceptance["reasons"]]
        acceptance_max_allowed = float(derived_acceptance["max_allowed_exposure"])

    if security_payload:
        if acceptance_status != "accepted":
            security_attention_reasons.extend(
                [reason for reason in acceptance_reasons if reason not in security_attention_reasons]
            )
    else:
        security_attention_reasons.append("security_status_missing")
    security_status_label = (
        "security-review" if security_attention_reasons else "security-ok"
    )
    runtime_attention_reasons: list[str] = []
    if not daemon_running:
        runtime_attention_reasons.append("daemon_not_running")
    elif daemon_payload_alignment_ok is False:
        runtime_attention_reasons.append("daemon_payload_unaligned")
    if acceptance_status == "failed":
        runtime_attention_reasons.append("security_acceptance_failed")
    elif acceptance_status == "review":
        runtime_attention_reasons.append("security_acceptance_review")
    if not runtime_attention_reasons:
        runtime_status_label = "runtime-ok"
    elif "security_acceptance_failed" in runtime_attention_reasons:
        runtime_status_label = "runtime-failed"
    else:
        runtime_status_label = "runtime-review"
    gate_status_label, gate_attention_reasons = derive_gate_status(
        ready=ready,
        ready_reasons=ready_reasons,
        live_gate_ok=live_gate_ok,
        live_gate_blocking_reason_codes=live_gate_blocking_reason_codes,
        ops_status=ops_status,
        ops_ok=ops_ok,
        ops_reason_code=ops_reason_code,
    )
    risk_guard_status_label, risk_guard_attention_reasons = derive_risk_guard_status(
        ready=ready,
        ready_reasons=ready_reasons,
        risk_guard_payload=risk_guard_payload,
    )
    remote_live_diagnosis = derive_remote_live_diagnosis(
        remote_live_history_summary=remote_live_history_summary,
        account_scope_alignment=account_scope_alignment_output,
        ready=ready,
        live_gate_ok=live_gate_ok,
        risk_guard_attention_reasons=risk_guard_attention_reasons,
    )
    operator_status_triplet = (
        f"{runtime_status_label} / {gate_status_label} / {risk_guard_status_label}"
    )
    operator_status_quad = (
        f"{runtime_status_label} / {gate_status_label} / {risk_guard_status_label} / {notification_status_label}"
    )
    operator_status_summary = (
        f"runtime={runtime_status_label}; gate={gate_status_label}; risk_guard={risk_guard_status_label}"
    )
    operator_status_summary_extended = (
        f"{operator_status_summary}; notify={notification_status_label}"
    )
    if runtime_status_label != "runtime-ok":
        next_focus_area = "runtime"
        next_focus_reason = (
            runtime_attention_reasons[0] if runtime_attention_reasons else runtime_status_label
        )
    elif gate_status_label != "gate-ok":
        next_focus_area = "gate"
        next_focus_reason = (
            gate_attention_reasons[0] if gate_attention_reasons else gate_status_label
        )
    elif risk_guard_status_label != "risk-guard-ok":
        next_focus_area = "risk_guard"
        next_focus_reason = (
            risk_guard_attention_reasons[0]
            if risk_guard_attention_reasons
            else risk_guard_status_label
        )
    else:
        next_focus_area = "canary"
        next_focus_reason = "ready_for_canary"
    secondary_focus_area, secondary_focus_reason = derive_secondary_focus(
        runtime_status_label=runtime_status_label,
        runtime_attention_reasons=runtime_attention_reasons,
        gate_status_label=gate_status_label,
        gate_attention_reasons=gate_attention_reasons,
        risk_guard_status_label=risk_guard_status_label,
        risk_guard_attention_reasons=risk_guard_attention_reasons,
        notification_status_label=notification_status_label,
        notification_attention_reasons=notification_attention_reasons,
        primary_focus_area=next_focus_area,
    )
    next_focus_command = derive_next_focus_command(
        system_root=SYSTEM_ROOT,
        next_focus_area=next_focus_area,
        next_focus_reason=next_focus_reason,
    )
    next_focus_commands = derive_next_focus_commands(
        system_root=SYSTEM_ROOT,
        next_focus_area=next_focus_area,
        next_focus_reason=next_focus_reason,
    )
    secondary_focus_command = (
        derive_next_focus_command(
            system_root=SYSTEM_ROOT,
            next_focus_area=secondary_focus_area,
            next_focus_reason=str(secondary_focus_reason or ""),
        )
        if secondary_focus_area
        else None
    )
    secondary_focus_commands = (
        derive_next_focus_commands(
            system_root=SYSTEM_ROOT,
            next_focus_area=secondary_focus_area,
            next_focus_reason=str(secondary_focus_reason or ""),
        )
        if secondary_focus_area
        else []
    )
    focus_stack = build_focus_stack(
        next_focus_area=next_focus_area,
        next_focus_reason=next_focus_reason,
        secondary_focus_area=secondary_focus_area,
        secondary_focus_reason=secondary_focus_reason,
    )
    focus_stack_brief = build_focus_stack_brief(focus_stack)
    focus_stack_summary = build_focus_stack_summary(focus_stack)
    operator_playbook = build_operator_playbook(
        system_root=SYSTEM_ROOT,
        runtime_status_label=runtime_status_label,
        runtime_attention_reasons=runtime_attention_reasons,
        gate_status_label=gate_status_label,
        gate_attention_reasons=gate_attention_reasons,
        risk_guard_status_label=risk_guard_status_label,
        risk_guard_attention_reasons=risk_guard_attention_reasons,
        address_family_floor=address_family_floor,
        address_family_recommendation=address_family_recommendation,
        security_top_risks=security_top_risks,
        security_recommendations=security_recommendations,
        next_focus_area=next_focus_area,
        next_focus_reason=next_focus_reason,
        next_focus_commands=next_focus_commands,
    )
    operator_playbook_md = build_operator_playbook_md(operator_playbook)

    if ready:
        handoff_state = "ready_for_canary"
        summary = "Remote live path is ready for a guarded canary."
        recommended_action = "Run a remote probe first, then a canary only if you intentionally want live validation."
    elif daemon_running and daemon_payload_alignment and daemon_payload_alignment_ok is False:
        handoff_state = "risk_daemon_attention"
        summary = "Remote risk daemon is running, but service state and daemon payload are not aligned yet."
        recommended_action = "Wait for the daemon state file to refresh or inspect daemon journal/status for restart lag."
    elif not daemon_running:
        handoff_state = "risk_daemon_attention"
        summary = "Remote risk daemon is not confirmed running."
        recommended_action = "Inspect daemon status/journal and start the remote risk daemon before any live execution."
    elif "ops_live_gate_blocked" in ready_reasons or live_gate_ok is False:
        handoff_state = "ops_live_gate_blocked"
        summary = "Remote live gate is blocking execution even though the daemon is available."
        recommended_action = "Inspect ops reconcile status and live gate blockers before attempting any canary."
    elif "risk_guard_blocked" in ready_reasons:
        handoff_state = "risk_guard_blocked"
        summary = "Remote execution path is up, but risk guard is correctly blocking orders."
        recommended_action = "Review ticket freshness, exposure, and current risk fuse reasons before retrying."
    elif ops_status in {"blocked", "stale"} or (ops_reason_code and not bool(ops_payload.get("ok", False))):
        handoff_state = "ops_reconcile_attention"
        summary = "Remote ops reconcile status still needs operator attention."
        recommended_action = "Refresh reconcile artifacts and review gate status before any live action."
    else:
        handoff_state = "review_required"
        summary = "Remote live state needs review before the next action."
        recommended_action = "Review ready-check, daemon status, and ops reconcile outputs together."

    operator_commands = [
        f"cd {SYSTEM_ROOT} && scripts/openclaw_cloud_bridge.sh live-takeover-ready-check",
        f"cd {SYSTEM_ROOT} && scripts/openclaw_cloud_bridge.sh live-risk-daemon-status",
        f"cd {SYSTEM_ROOT} && scripts/openclaw_cloud_bridge.sh live-risk-daemon-security-status",
        f"cd {SYSTEM_ROOT} && scripts/openclaw_cloud_bridge.sh live-risk-daemon-journal",
        f"cd {SYSTEM_ROOT} && scripts/openclaw_cloud_bridge.sh live-ops-reconcile-status",
    ]
    if not daemon_running:
        operator_commands.append(
            f"cd {SYSTEM_ROOT} && scripts/openclaw_cloud_bridge.sh live-risk-daemon-start"
        )
    if handoff_state in {"ops_live_gate_blocked", "ops_reconcile_attention"}:
        operator_commands.append(
            f"cd {SYSTEM_ROOT} && scripts/openclaw_cloud_bridge.sh live-ops-reconcile-refresh"
        )
    if ready:
        operator_commands.append(
            f"cd {SYSTEM_ROOT} && scripts/openclaw_cloud_bridge.sh live-takeover-probe"
        )
        operator_commands.append(
            f"cd {SYSTEM_ROOT} && scripts/openclaw_cloud_bridge.sh live-takeover-canary"
        )
    if security_attention_reasons:
        recommended_action = (
            f"{recommended_action} Also review remote service hardening/verify output before changing daemon install settings."
        )
    operator_handoff_md = build_operator_handoff_md(
        handoff_state=handoff_state,
        summary=summary,
        recommended_action=recommended_action,
        operator_status_triplet=operator_status_triplet,
        focus_stack_brief=focus_stack_brief,
        focus_stack_summary=focus_stack_summary,
        secondary_focus_area=secondary_focus_area,
        secondary_focus_reason=secondary_focus_reason,
        next_focus_area=next_focus_area,
        next_focus_reason=next_focus_reason,
        security_status_label=security_status_label,
        security_acceptance_status=acceptance_status,
        security_exposure_score=security_exposure_score,
        security_top_risks=security_top_risks,
        address_family_floor=address_family_floor,
        address_family_recommendation=address_family_recommendation,
        clock_device_floor=(
            str(operator_playbook["sections"]["runtime"].get("clock_device_floor") or "")
            if isinstance(operator_playbook.get("sections"), dict)
            and isinstance(operator_playbook["sections"].get("runtime"), dict)
            else ""
        ),
        clock_device_note=(
            str(operator_playbook["sections"]["runtime"].get("clock_device_note") or "")
            if isinstance(operator_playbook.get("sections"), dict)
            and isinstance(operator_playbook["sections"].get("runtime"), dict)
            else ""
        ),
        operator_status_quad=operator_status_quad,
        notification_delivery_readiness_label=notification_delivery_readiness_label,
        remote_live_history_summary=remote_live_history_summary,
        remote_live_diagnosis=remote_live_diagnosis,
        ready_check_scope_brief=ready_check_scope_brief,
        account_scope_alignment=account_scope_alignment_output,
        operator_playbook_md=operator_playbook_md,
    )
    operator_handoff_brief = build_operator_handoff_brief(
        handoff_state=handoff_state,
        operator_status_triplet=operator_status_triplet,
        operator_status_quad=operator_status_quad,
        focus_stack_brief=focus_stack_brief,
        address_family_floor=address_family_floor,
        clock_device_floor=(
            str(operator_playbook["sections"]["runtime"].get("clock_device_floor") or "")
            if isinstance(operator_playbook.get("sections"), dict)
            and isinstance(operator_playbook["sections"].get("runtime"), dict)
            else ""
        ),
        notification_delivery_readiness_label=notification_delivery_readiness_label,
        next_focus_area=next_focus_area,
        next_focus_reason=next_focus_reason,
        next_focus_command=next_focus_command,
        secondary_focus_area=secondary_focus_area,
        secondary_focus_reason=secondary_focus_reason,
        remote_live_history_summary=remote_live_history_summary,
        remote_live_diagnosis=remote_live_diagnosis,
        ready_check_scope_brief=ready_check_scope_brief,
        account_scope_alignment=account_scope_alignment_output,
    )
    operator_notification = build_operator_notification(
        handoff_state=handoff_state,
        operator_status_triplet=operator_status_triplet,
        focus_stack_brief=focus_stack_brief,
        address_family_floor=address_family_floor,
        clock_device_floor=(
            str(operator_playbook["sections"]["runtime"].get("clock_device_floor") or "")
            if isinstance(operator_playbook.get("sections"), dict)
            and isinstance(operator_playbook["sections"].get("runtime"), dict)
            else ""
        ),
        next_focus_area=next_focus_area,
        next_focus_reason=next_focus_reason,
        next_focus_command=next_focus_command,
        security_status_label=security_status_label,
        ready=ready,
    )
    operator_notification_templates = build_operator_notification_templates(
        operator_notification
    )

    return {
        "handoff_state": handoff_state,
        "summary": summary,
        "recommended_action": recommended_action,
        "remote_host": remote_host,
        "remote_user": remote_user,
        "remote_project_dir": remote_project_dir,
        "target_live_takeover_market": target_market,
        "openclaw_orderflow_executor_mode": openclaw_orderflow_executor_mode,
        "openclaw_orderflow_executor_mode_source": openclaw_orderflow_executor_mode_source,
        "ready": ready,
        "ready_reasons": ready_reasons,
        "ready_check_scope_market": effective_ready_scope_market,
        "ready_check_scope_source": effective_ready_scope_source,
        "ready_check_scope_brief": ready_check_scope_brief,
        "ready_check_spot": ready_check_spot_summary or None,
        "ready_check_portfolio_margin_um": ready_check_portfolio_summary or None,
        "account_scope_alignment": account_scope_alignment_output,
        "execution_contract": execution_contract,
        "daemon_running": daemon_running,
        "daemon_mode": daemon_mode,
        "daemon_status": daemon_status,
        "daemon_payload_alignment_ok": daemon_payload_alignment_ok,
        "daemon_payload_alignment_reasons": daemon_payload_alignment_reasons,
        "daemon_payload_alignment": daemon_payload_alignment if daemon_payload_alignment else None,
        "ops_status": ops_status,
        "ops_reason_code": ops_reason_code,
        "live_gate_ok": live_gate_ok,
        "live_gate_blocking_reason_codes": live_gate_blocking_reason_codes
        if isinstance(live_gate, dict)
        else None,
        "gate_status_label": gate_status_label,
        "gate_attention_reasons": gate_attention_reasons,
        "risk_guard_status_label": risk_guard_status_label,
        "risk_guard_attention_reasons": risk_guard_attention_reasons,
        "security_verify_ok": security_verify_ok,
        "security_exposure_score": security_exposure_score,
        "security_exposure_rating": security_exposure_rating,
        "security_acceptance_status": acceptance_status,
        "security_acceptance_reasons": acceptance_reasons,
        "security_acceptance_max_allowed_exposure": acceptance_max_allowed,
        "runtime_status_label": runtime_status_label,
        "runtime_attention_reasons": runtime_attention_reasons,
        "operator_status_triplet": operator_status_triplet,
        "operator_status_quad": operator_status_quad,
        "operator_status_summary": operator_status_summary,
        "operator_status_summary_extended": operator_status_summary_extended,
        "focus_stack": focus_stack,
        "focus_stack_brief": focus_stack_brief,
        "focus_stack_summary": focus_stack_summary,
        "next_focus_area": next_focus_area,
        "next_focus_reason": next_focus_reason,
        "next_focus_command": next_focus_command,
        "next_focus_commands": next_focus_commands,
        "secondary_focus_area": secondary_focus_area,
        "secondary_focus_reason": secondary_focus_reason,
        "secondary_focus_command": secondary_focus_command,
        "secondary_focus_commands": secondary_focus_commands,
        "operator_playbook": operator_playbook,
        "operator_playbook_md": operator_playbook_md,
        "operator_handoff_md": operator_handoff_md,
        "operator_handoff_brief": operator_handoff_brief,
        "operator_notification": operator_notification,
        "operator_notification_templates": operator_notification_templates,
        "remote_live_history": remote_live_history_summary,
        "remote_live_diagnosis": remote_live_diagnosis,
        "notification_delivery_status": notification_delivery_status,
        "notification_delivery_readiness_label": notification_delivery_readiness_label,
        "notification_delivery_capabilities": notification_delivery_capabilities,
        "notification_status_label": notification_status_label,
        "notification_attention_reasons": notification_attention_reasons,
        "security_status_label": security_status_label,
        "security_attention_reasons": security_attention_reasons,
        "security_top_risks": security_top_risks,
        "security_recommendations": security_recommendations,
        "address_family_floor": address_family_floor,
        "address_family_probe_status": address_family_probe_status,
        "address_family_recommendation": address_family_recommendation,
        "address_family_probe": noaf_probe_payload if noaf_probe_payload else None,
        "security_findings": security_findings[:5],
        "journal_tail": journal_tail,
        "operator_commands": operator_commands,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a single-file handoff artifact summarizing remote live execution readiness, daemon state, ops gate, and recent journal context."
    )
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--ready-check-file", default="")
    parser.add_argument("--ready-check-returncode", type=int, default=0)
    parser.add_argument("--ready-check-spot-file", default="")
    parser.add_argument("--ready-check-spot-returncode", type=int, default=0)
    parser.add_argument("--ready-check-portfolio-margin-file", default="")
    parser.add_argument("--ready-check-portfolio-margin-returncode", type=int, default=0)
    parser.add_argument("--risk-daemon-status-file", default="")
    parser.add_argument("--risk-daemon-status-returncode", type=int, default=0)
    parser.add_argument("--ops-status-file", default="")
    parser.add_argument("--ops-status-returncode", type=int, default=0)
    parser.add_argument("--journal-file", default="")
    parser.add_argument("--journal-returncode", type=int, default=0)
    parser.add_argument("--security-status-file", default="")
    parser.add_argument("--security-status-returncode", type=int, default=0)
    parser.add_argument("--noaf-probe-file", default="")
    parser.add_argument("--notification-send-file", default="")
    parser.add_argument("--security-accept-max-exposure", default="")
    parser.add_argument("--live-takeover-market", default="")
    parser.add_argument("--openclaw-orderflow-executor-mode", default="")
    parser.add_argument("--remote-host", default="")
    parser.add_argument("--remote-user", default="")
    parser.add_argument("--remote-project-dir", default="")
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--now", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    review_dir = Path(args.review_dir).expanduser().resolve()
    now_ts = dt.datetime.fromisoformat(args.now) if str(args.now).strip() else now_utc()

    ready_path_arg = (
        Path(args.ready_check_file).expanduser().resolve() if args.ready_check_file else None
    )
    ready_spot_path_arg = (
        Path(args.ready_check_spot_file).expanduser().resolve()
        if args.ready_check_spot_file
        else None
    )
    ready_portfolio_margin_path_arg = (
        Path(args.ready_check_portfolio_margin_file).expanduser().resolve()
        if args.ready_check_portfolio_margin_file
        else None
    )
    daemon_path_arg = (
        Path(args.risk_daemon_status_file).expanduser().resolve()
        if args.risk_daemon_status_file
        else None
    )
    ops_path_arg = Path(args.ops_status_file).expanduser().resolve() if args.ops_status_file else None
    journal_path_arg = Path(args.journal_file).expanduser().resolve() if args.journal_file else None
    security_path_arg = (
        Path(args.security_status_file).expanduser().resolve()
        if args.security_status_file
        else None
    )
    noaf_probe_path_arg = (
        Path(args.noaf_probe_file).expanduser().resolve()
        if args.noaf_probe_file
        else None
    )
    notification_send_file = (
        Path(args.notification_send_file).expanduser().resolve()
        if args.notification_send_file
        else find_latest_review_artifact(review_dir, "*_remote_live_notification_send.json")
    )
    bridge_context_file, bridge_context_payload, _ = resolve_review_input_artifact(
        review_dir=review_dir,
        explicit_path=None,
        pattern="*_remote_live_bridge_context.json",
    )
    remote_live_history_file = find_latest_review_artifact(
        review_dir, "*_remote_live_history_audit.json"
    )
    bridge_context_payload = bridge_context_payload if isinstance(bridge_context_payload, dict) else {}
    remote_host = str(args.remote_host or bridge_context_payload.get("remote_host") or "").strip()
    remote_user = str(args.remote_user or bridge_context_payload.get("remote_user") or "").strip()
    remote_project_dir = str(
        args.remote_project_dir or bridge_context_payload.get("remote_project_dir") or ""
    ).strip()
    live_takeover_market = str(
        args.live_takeover_market or bridge_context_payload.get("live_takeover_market") or ""
    ).strip().lower()
    openclaw_orderflow_executor_mode_raw = str(
        args.openclaw_orderflow_executor_mode
        or bridge_context_payload.get("openclaw_orderflow_executor_mode")
        or ""
    ).strip()
    if args.openclaw_orderflow_executor_mode:
        openclaw_orderflow_executor_mode_source = "arg"
    elif bridge_context_payload.get("openclaw_orderflow_executor_mode"):
        openclaw_orderflow_executor_mode_source = "bridge_context"
    else:
        openclaw_orderflow_executor_mode_source = "default"
    openclaw_orderflow_executor_mode = (
        openclaw_orderflow_executor_mode_raw.lower()
        if openclaw_orderflow_executor_mode_raw
        else "shadow_guarded"
    )
    security_accept_max_exposure_raw = str(args.security_accept_max_exposure or "").strip()
    if not security_accept_max_exposure_raw:
        security_accept_max_exposure_raw = str(
            bridge_context_payload.get("security_accept_max_exposure") or ""
        ).strip()
    try:
        security_accept_max_exposure = (
            float(security_accept_max_exposure_raw) if security_accept_max_exposure_raw else 4.0
        )
    except (TypeError, ValueError):
        security_accept_max_exposure = 4.0

    ready_file, ready_payload, ready_returncode_auto = resolve_review_input_artifact(
        review_dir=review_dir,
        explicit_path=ready_path_arg,
        pattern="*_remote_live_ready_check.json",
    )
    ready_spot_file, ready_payload_spot, ready_spot_returncode_auto = resolve_review_input_artifact(
        review_dir=review_dir,
        explicit_path=ready_spot_path_arg,
        pattern="*_remote_live_ready_check_spot.json",
    )
    (
        ready_portfolio_margin_file,
        ready_payload_portfolio_margin_um,
        ready_portfolio_returncode_auto,
    ) = resolve_review_input_artifact(
        review_dir=review_dir,
        explicit_path=ready_portfolio_margin_path_arg,
        pattern="*_remote_live_ready_check_portfolio_margin_um.json",
    )
    daemon_file, daemon_payload, daemon_returncode_auto = resolve_review_input_artifact(
        review_dir=review_dir,
        explicit_path=daemon_path_arg,
        pattern="*_remote_live_risk_daemon_status.json",
    )
    ops_file, ops_payload, ops_returncode_auto = resolve_review_input_artifact(
        review_dir=review_dir,
        explicit_path=ops_path_arg,
        pattern="*_remote_live_ops_reconcile_status.json",
    )
    journal_file, journal_payload, journal_returncode_auto = resolve_review_input_artifact(
        review_dir=review_dir,
        explicit_path=journal_path_arg,
        pattern="*_remote_live_risk_daemon_journal.json",
    )
    security_file, security_payload, security_returncode_auto = resolve_review_input_artifact(
        review_dir=review_dir,
        explicit_path=security_path_arg,
        pattern="*_remote_live_risk_daemon_security_status.json",
    )
    noaf_probe_file, noaf_probe_payload, _ = resolve_review_input_artifact(
        review_dir=review_dir,
        explicit_path=noaf_probe_path_arg,
        pattern="*_remote_live_noaf_probe.json",
    )
    notification_send_payload = load_payload_file(notification_send_file)
    remote_live_history_payload = load_payload_file(remote_live_history_file)

    out: dict[str, Any] = {
        "action": "build_remote_live_handoff",
        "ok": True,
        "status": "ok",
        "generated_at": fmt_utc(now_ts),
        "remote_host": remote_host,
        "remote_user": remote_user,
        "remote_project_dir": remote_project_dir,
        "security_accept_max_exposure": float(security_accept_max_exposure),
        "ready_check_returncode": int(
            args.ready_check_returncode if ready_path_arg is not None else (ready_returncode_auto or 0)
        ),
        "ready_check_spot_returncode": int(
            args.ready_check_spot_returncode
            if ready_spot_path_arg is not None
            else (ready_spot_returncode_auto or 0)
        ),
        "ready_check_portfolio_margin_um_returncode": int(
            args.ready_check_portfolio_margin_returncode
            if ready_portfolio_margin_path_arg is not None
            else (ready_portfolio_returncode_auto or 0)
        ),
        "risk_daemon_status_returncode": int(
            args.risk_daemon_status_returncode
            if daemon_path_arg is not None
            else (daemon_returncode_auto or 0)
        ),
        "ops_status_returncode": int(
            args.ops_status_returncode if ops_path_arg is not None else (ops_returncode_auto or 0)
        ),
        "journal_returncode": int(
            args.journal_returncode
            if journal_path_arg is not None
            else (journal_returncode_auto or 0)
        ),
        "security_status_returncode": int(
            args.security_status_returncode
            if security_path_arg is not None
            else (security_returncode_auto or 0)
        ),
        "ready_check": ready_payload,
        "ready_check_spot": ready_payload_spot,
        "ready_check_portfolio_margin_um": ready_payload_portfolio_margin_um,
        "risk_daemon_status": daemon_payload,
        "ops_reconcile_status": ops_payload,
        "risk_daemon_journal": journal_payload,
        "risk_daemon_security_status": security_payload,
        "risk_daemon_noaf_probe": noaf_probe_payload,
        "notification_send": notification_send_payload,
        "remote_live_history_audit": remote_live_history_payload,
        "source_artifacts": {
            "ready_check": str(ready_file) if ready_file else "",
            "ready_check_spot": str(ready_spot_file) if ready_spot_file else "",
            "ready_check_portfolio_margin_um": str(ready_portfolio_margin_file)
            if ready_portfolio_margin_file
            else "",
            "risk_daemon_status": str(daemon_file) if daemon_file else "",
            "ops_status": str(ops_file) if ops_file else "",
            "journal": str(journal_file) if journal_file else "",
            "security_status": str(security_file) if security_file else "",
            "notification_send": str(notification_send_file) if notification_send_file else "",
            "remote_live_history_audit": str(remote_live_history_file)
            if remote_live_history_file
            else "",
            "noaf_probe": str(noaf_probe_file) if noaf_probe_file else "",
            "bridge_context": str(bridge_context_file) if bridge_context_file else "",
        },
        "operator_handoff": None,
        "artifact_status_label": "handoff-ok",
        "artifact_label": None,
        "artifact_tags": [],
        "artifact": None,
        "checksum": None,
        "pruned_keep": [],
        "pruned_age": [],
    }

    operator_handoff = build_operator_handoff(
        remote_host=remote_host,
        remote_user=remote_user,
        remote_project_dir=remote_project_dir,
        ready_payload=ready_payload,
        ready_payload_spot=ready_payload_spot,
        ready_payload_portfolio_margin_um=ready_payload_portfolio_margin_um,
        daemon_payload=daemon_payload,
        ops_payload=ops_payload,
        journal_payload=journal_payload,
        security_payload=security_payload,
        noaf_probe_payload=noaf_probe_payload,
        notification_send_payload=notification_send_payload,
        remote_live_history_payload=remote_live_history_payload,
        remote_live_history_artifact=remote_live_history_file,
        security_accept_max_exposure=float(security_accept_max_exposure),
        target_live_takeover_market=live_takeover_market,
        openclaw_orderflow_executor_mode=openclaw_orderflow_executor_mode,
        openclaw_orderflow_executor_mode_source=openclaw_orderflow_executor_mode_source,
    )
    out["operator_handoff"] = operator_handoff
    effective_ready_scope_market = str(
        operator_handoff.get("ready_check_scope_market") or ""
    ).strip()
    if effective_ready_scope_market == "spot" and ready_payload_spot:
        out["ready_check"] = ready_payload_spot
        out["ready_check_returncode"] = int(
            args.ready_check_spot_returncode
            if ready_spot_path_arg is not None
            else (ready_spot_returncode_auto or 0)
        )
    elif (
        effective_ready_scope_market == "portfolio_margin_um"
        and ready_payload_portfolio_margin_um
    ):
        out["ready_check"] = ready_payload_portfolio_margin_um
        out["ready_check_returncode"] = int(
            args.ready_check_portfolio_margin_returncode
            if ready_portfolio_margin_path_arg is not None
            else (ready_portfolio_returncode_auto or 0)
        )
    out["artifact_label"] = f"remote-live-handoff:{operator_handoff['handoff_state']}"
    out["artifact_tags"] = [
        "remote-live",
        "handoff",
        str(operator_handoff["handoff_state"]),
        "read-only",
    ]

    review_dir.mkdir(parents=True, exist_ok=True)
    stamp = now_ts.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_path = review_dir / f"{stamp}_remote_live_handoff.json"
    checksum_path = review_dir / f"{stamp}_remote_live_handoff_checksum.json"
    out["artifact"] = str(artifact_path)
    out["checksum"] = str(checksum_path)

    artifact_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    digest = sha256_file(artifact_path)
    checksum_path.write_text(
        json.dumps(
            {
                "generated_at": fmt_utc(now_ts),
                "artifact_ttl_hours": max(1.0, float(args.artifact_ttl_hours)),
                "files": [
                    {
                        "path": str(artifact_path),
                        "sha256": digest,
                        "size_bytes": int(artifact_path.stat().st_size),
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    pruned_keep, pruned_age = prune_handoffs(
        review_dir,
        current_artifact=artifact_path,
        current_checksum=checksum_path,
        keep=max(1, int(args.artifact_keep)),
        ttl_hours=max(1.0, float(args.artifact_ttl_hours)),
    )
    out["pruned_keep"] = pruned_keep
    out["pruned_age"] = pruned_age
    artifact_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
