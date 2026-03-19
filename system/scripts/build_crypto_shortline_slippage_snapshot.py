#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
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
TIMESTAMP_RE = re.compile(r"^(?P<stamp>\d{8}T\d{6}Z)_")
FUTURE_STAMP_GRACE_MINUTES = 5
DEFAULT_ASSUMED_ROUNDTRIP_FEE_BPS = 2.0


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse_now(raw: str | None) -> dt.datetime:
    text_value = str(raw or "").strip()
    if not text_value:
        return now_utc()
    parsed = dt.datetime.fromisoformat(text_value.replace("Z", "+00:00"))
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


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def dedupe_text(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        item = text(raw)
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def join_unique(parts: list[Any], *, sep: str = " | ") -> str:
    return sep.join(dedupe_text(parts))


def load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def artifact_stamp(path: Path) -> str:
    match = TIMESTAMP_RE.match(path.name)
    return match.group("stamp") if match else ""


def parsed_artifact_stamp(path: Path) -> dt.datetime | None:
    stamp = artifact_stamp(path)
    if not stamp:
        return None
    try:
        return dt.datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def artifact_sort_key(path: Path, reference_now: dt.datetime | None = None) -> tuple[int, str, float, str]:
    stamp_dt = parsed_artifact_stamp(path)
    effective_now = reference_now or now_utc()
    future_cutoff = effective_now + dt.timedelta(minutes=FUTURE_STAMP_GRACE_MINUTES)
    is_future = bool(stamp_dt and stamp_dt > future_cutoff)
    return (0 if is_future else 1, artifact_stamp(path), path.stat().st_mtime, path.name)


def find_latest(
    review_dir: Path,
    pattern: str,
    reference_now: dt.datetime | None = None,
) -> Path | None:
    files = sorted(
        review_dir.glob(pattern),
        key=lambda item: artifact_sort_key(item, reference_now),
        reverse=True,
    )
    if not files:
        return None
    future_cutoff = (reference_now or now_utc()) + dt.timedelta(minutes=FUTURE_STAMP_GRACE_MINUTES)
    for path in files:
        stamp_dt = parsed_artifact_stamp(path)
        if stamp_dt is None or stamp_dt <= future_cutoff:
            return path
    return files[0]


def select_ticket_surface_path(
    review_dir: Path,
    route_symbol: str,
    reference_now: dt.datetime | None = None,
) -> Path | None:
    candidates = sorted(
        review_dir.glob("*_signal_to_order_tickets.json"),
        key=lambda item: artifact_sort_key(item, reference_now),
        reverse=True,
    )
    if not candidates:
        return None
    normalized_route = text(route_symbol).upper()
    for path in candidates:
        try:
            payload = load_json_mapping(path)
        except Exception:
            continue
        for row in as_list(payload.get("tickets")):
            item = as_dict(row)
            if text(item.get("symbol")).upper() == normalized_route:
                return path
    return candidates[0]


def load_live_orderflow_snapshot(
    review_dir: Path,
    *,
    route_symbol: str,
    reference_now: dt.datetime,
) -> tuple[Path | None, dict[str, Any]]:
    path = find_latest(
        review_dir, "*_crypto_shortline_live_orderflow_snapshot.json", reference_now
    )
    if path is None or not path.exists():
        return None, {}
    payload = load_json_mapping(path)
    if text(payload.get("route_symbol")).upper() != route_symbol.upper():
        return None, {}
    return path, payload


def load_pattern_router(
    review_dir: Path,
    *,
    route_symbol: str,
    reference_now: dt.datetime,
) -> tuple[Path | None, dict[str, Any]]:
    path = find_latest(review_dir, "*_crypto_shortline_pattern_router.json", reference_now)
    if path is None or not path.exists():
        return None, {}
    payload = load_json_mapping(path)
    payload_symbol = text(payload.get("route_symbol")).upper()
    if payload_symbol and payload_symbol != route_symbol.upper():
        return None, {}
    return path, payload


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
        "*_crypto_shortline_slippage_snapshot.json",
        "*_crypto_shortline_slippage_snapshot.md",
        "*_crypto_shortline_slippage_snapshot_checksum.json",
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


def route_symbol(intent_payload: dict[str, Any], operator_payload: dict[str, Any]) -> str:
    return (
        text(intent_payload.get("preferred_route_symbol"))
        or text(operator_payload.get("review_priority_head_symbol"))
        or text(operator_payload.get("next_focus_symbol"))
    ).upper()


def route_action(intent_payload: dict[str, Any], operator_payload: dict[str, Any]) -> str:
    return text(intent_payload.get("preferred_route_action")) or text(
        operator_payload.get("next_focus_action")
    )


def find_route_row(payload: dict[str, Any], symbol: str) -> dict[str, Any]:
    for row in as_list(payload.get("tickets")):
        item = as_dict(row)
        if text(item.get("symbol")).upper() == symbol.upper():
            return item
    return {}


def bounded_penalty(value: float, lower: float, upper: float, max_penalty: float) -> float:
    if value >= upper:
        return 0.0
    if value <= lower:
        return max_penalty
    span = max(upper - lower, 1e-9)
    return ((upper - value) / span) * max_penalty


def distance_bps(*, start_price: float, end_price: float) -> float:
    start = max(0.0, float(start_price))
    end = max(0.0, float(end_price))
    if start <= 0.0 or end <= 0.0:
        return 0.0
    return abs(end - start) / start * 10000.0


def estimate_entry_cost_bps(
    *,
    max_slippage_bps: float,
    queue_imbalance: float,
    ofi_norm: float,
    micro_alignment: float,
    trade_count: int,
    evidence_score: float,
    micro_quality_ok: bool,
    trust_ok: bool,
    time_sync_ok: bool,
    cvd_veto_hint: str,
) -> float:
    budget = max(0.0, float(max_slippage_bps))
    if budget <= 0.0:
        return 0.0
    multiplier = 0.55
    multiplier += bounded_penalty(abs(queue_imbalance), 0.03, 0.12, 0.25)
    multiplier += bounded_penalty(abs(ofi_norm), 0.08, 0.20, 0.20)
    multiplier += bounded_penalty(abs(micro_alignment), 0.08, 0.18, 0.15)
    if int(trade_count) < 100:
        multiplier += 0.10
    elif int(trade_count) < 250:
        multiplier += 0.05
    if float(evidence_score) < 0.50:
        multiplier += 0.10
    elif float(evidence_score) < 0.80:
        multiplier += 0.05
    if not micro_quality_ok:
        multiplier += 0.25
    if not trust_ok:
        multiplier += 0.35
    if not time_sync_ok:
        multiplier += 0.25
    if text(cvd_veto_hint):
        multiplier += 0.30
    return round(max(0.0, budget * multiplier), 3)


def render_markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Crypto Shortline Slippage Snapshot",
            "",
            f"- brief: `{text(payload.get('snapshot_brief'))}`",
            f"- decision: `{text(payload.get('snapshot_decision'))}`",
            f"- pattern_family: `{text(payload.get('pattern_family')) or '-'}`",
            f"- pattern_stage: `{text(payload.get('pattern_stage')) or '-'}`",
            f"- entry_price: `{payload.get('entry_price')}`",
            f"- stop_price: `{payload.get('stop_price')}`",
            f"- target_price: `{payload.get('target_price')}`",
            f"- max_slippage_bps: `{payload.get('max_slippage_bps')}`",
            f"- estimated_entry_cost_bps: `{payload.get('estimated_entry_cost_bps')}`",
            f"- estimated_roundtrip_cost_bps: `{payload.get('estimated_roundtrip_cost_bps')}`",
            f"- target_distance_bps: `{payload.get('target_distance_bps')}`",
            f"- stop_distance_bps: `{payload.get('stop_distance_bps')}`",
            f"- cost_to_target_ratio: `{payload.get('cost_to_target_ratio')}`",
            f"- post_cost_viable: `{payload.get('post_cost_viable')}`",
            f"- trust_ok: `{payload.get('trust_ok')}`",
            f"- time_sync_ok: `{payload.get('time_sync_ok')}`",
            f"- cvd_veto_hint: `{text(payload.get('cvd_veto_hint'))}`",
            f"- blocker: `{text(payload.get('blocker_detail'))}`",
            f"- done_when: `{text(payload.get('done_when'))}`",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a source-owned crypto shortline slippage snapshot artifact."
    )
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

    intent_path = find_latest(review_dir, "*_remote_intent_queue.json", reference_now)
    operator_path = find_latest(review_dir, "*_crypto_route_operator_brief.json", reference_now)
    if operator_path is None:
        raise SystemExit("missing_required_artifacts:crypto_route_operator_brief")

    intent_payload = (
        load_json_mapping(intent_path) if intent_path is not None and intent_path.exists() else {}
    )
    operator_payload = load_json_mapping(operator_path)
    symbol = route_symbol(intent_payload, operator_payload)
    action = route_action(intent_payload, operator_payload)
    remote_market = text(intent_payload.get("remote_market")) or "portfolio_margin_um"

    tickets_path = select_ticket_surface_path(review_dir, symbol, reference_now)
    if tickets_path is None:
        raise SystemExit("missing_required_artifacts:signal_to_order_tickets")
    tickets_payload = load_json_mapping(tickets_path)
    route_row = find_route_row(tickets_payload, symbol)

    pattern_router_path, pattern_router_payload = load_pattern_router(
        review_dir,
        route_symbol=symbol,
        reference_now=reference_now,
    )
    live_orderflow_snapshot_path, live_orderflow_snapshot_payload = load_live_orderflow_snapshot(
        review_dir,
        route_symbol=symbol,
        reference_now=reference_now,
    )

    route_levels = as_dict(route_row.get("levels"))
    route_execution = as_dict(route_row.get("execution"))
    route_sizing = as_dict(route_row.get("sizing"))
    route_row_reasons = dedupe_text(as_list(route_row.get("reasons")))
    pattern_family = text(pattern_router_payload.get("pattern_family"))
    pattern_stage = text(pattern_router_payload.get("pattern_stage"))
    pattern_status = text(pattern_router_payload.get("pattern_status"))

    entry_price = to_float(route_levels.get("entry_price"))
    stop_price = to_float(route_levels.get("stop_price"))
    target_price = to_float(route_levels.get("target_price"))
    max_slippage_bps = to_float(route_execution.get("max_slippage_bps"))
    quote_usdt = to_float(route_sizing.get("quote_usdt"))
    queue_imbalance = to_float(live_orderflow_snapshot_payload.get("queue_imbalance"))
    ofi_norm = to_float(live_orderflow_snapshot_payload.get("ofi_norm"))
    micro_alignment = to_float(live_orderflow_snapshot_payload.get("micro_alignment"))
    trade_count = int(to_float(live_orderflow_snapshot_payload.get("trade_count")))
    evidence_score = to_float(live_orderflow_snapshot_payload.get("evidence_score"))
    micro_quality_ok = bool(live_orderflow_snapshot_payload.get("micro_quality_ok", False))
    trust_ok = bool(live_orderflow_snapshot_payload.get("trust_ok", False))
    time_sync_ok = bool(live_orderflow_snapshot_payload.get("time_sync_ok", False))
    cvd_veto_hint = text(live_orderflow_snapshot_payload.get("cvd_veto_hint"))

    target_distance_bps = round(distance_bps(start_price=entry_price, end_price=target_price), 3)
    stop_distance_bps = round(distance_bps(start_price=entry_price, end_price=stop_price), 3)
    estimated_entry_cost_bps = estimate_entry_cost_bps(
        max_slippage_bps=max_slippage_bps,
        queue_imbalance=queue_imbalance,
        ofi_norm=ofi_norm,
        micro_alignment=micro_alignment,
        trade_count=trade_count,
        evidence_score=evidence_score,
        micro_quality_ok=micro_quality_ok,
        trust_ok=trust_ok,
        time_sync_ok=time_sync_ok,
        cvd_veto_hint=cvd_veto_hint,
    )
    assumed_roundtrip_fee_bps = DEFAULT_ASSUMED_ROUNDTRIP_FEE_BPS if max_slippage_bps > 0 else 0.0
    estimated_roundtrip_cost_bps = round(
        estimated_entry_cost_bps * 2.0 + assumed_roundtrip_fee_bps,
        3,
    )
    cost_to_target_ratio = round(
        estimated_roundtrip_cost_bps / target_distance_bps, 6
    ) if target_distance_bps > 0 else None
    cost_to_stop_ratio = round(
        estimated_roundtrip_cost_bps / stop_distance_bps, 6
    ) if stop_distance_bps > 0 else None

    value_rotation_active = pattern_family == "value_rotation_scalp"
    levels_ready = entry_price > 0 and stop_price > 0 and target_price > 0 and max_slippage_bps > 0
    post_cost_viable = bool(
        levels_ready
        and target_distance_bps > 0
        and (cost_to_target_ratio or 1.0) <= 0.20
        and micro_quality_ok
        and trust_ok
        and time_sync_ok
        and not cvd_veto_hint
    )

    if not route_row:
        snapshot_status = "slippage_snapshot_missing_ticket_row"
        snapshot_decision = "refresh_ticket_surface_then_recheck_execution_gate"
        blocker_title = "Generate route ticket before post-cost shortline review"
        done_when = f"{symbol} has a fresh route ticket row with explicit entry/stop/target fields"
    elif live_orderflow_snapshot_path is None or not live_orderflow_snapshot_payload:
        snapshot_status = "slippage_snapshot_missing_live_orderflow"
        snapshot_decision = "collect_live_orderflow_snapshot_before_post_cost_check"
        blocker_title = "Collect live orderflow before post-cost shortline review"
        done_when = f"{symbol} has a matching live orderflow snapshot for post-cost validation"
    elif not levels_ready:
        snapshot_status = "slippage_snapshot_missing_execution_levels"
        snapshot_decision = "repair_execution_price_reference_then_recheck_post_cost"
        blocker_title = "Repair execution levels before post-cost shortline review"
        done_when = f"{symbol} keeps non-zero entry/stop/target and max_slippage_bps in the route ticket"
    elif post_cost_viable:
        if value_rotation_active:
            snapshot_status = "value_rotation_scalp_post_cost_viable"
            snapshot_decision = "recheck_execution_gate_after_value_rotation_post_cost_clear"
            blocker_title = "Value-rotation post-cost path clear for shortline scalp promotion"
        else:
            snapshot_status = "post_cost_viable"
            snapshot_decision = "recheck_execution_gate_after_post_cost_clear"
            blocker_title = "Post-cost path clear for shortline setup promotion"
        done_when = f"{symbol} reintroduces post-cost blockers or leaves Setup_Ready"
    else:
        if value_rotation_active:
            snapshot_status = "value_rotation_scalp_post_cost_degraded"
            snapshot_decision = "repair_value_rotation_post_cost_then_recheck_execution_gate"
            blocker_title = "Repair value-rotation post-cost before shortline scalp promotion"
            done_when = (
                f"{symbol} keeps estimated_roundtrip_cost_bps comfortably below target_distance_bps, "
                "trust/time-sync stay clear, and post_cost_viable flips true while the value-rotation scalp remains active"
            )
        else:
            snapshot_status = "post_cost_degraded"
            snapshot_decision = "repair_post_cost_then_recheck_execution_gate"
            blocker_title = "Repair shortline post-cost before setup promotion"
            done_when = (
                f"{symbol} keeps estimated_roundtrip_cost_bps comfortably below target_distance_bps "
                "and post_cost_viable flips true"
            )

    snapshot_brief = ":".join([snapshot_status, symbol or "-", snapshot_decision, remote_market or "-"])
    blocker_detail = join_unique(
        [
            f"estimated_entry_cost_bps={estimated_entry_cost_bps:g}",
            f"estimated_roundtrip_cost_bps={estimated_roundtrip_cost_bps:g}",
            f"target_distance_bps={target_distance_bps:g}",
            f"stop_distance_bps={stop_distance_bps:g}",
            (
                f"cost_to_target_ratio={cost_to_target_ratio:g}"
                if cost_to_target_ratio is not None
                else ""
            ),
            (
                f"cost_to_stop_ratio={cost_to_stop_ratio:g}"
                if cost_to_stop_ratio is not None
                else ""
            ),
            f"max_slippage_bps={max_slippage_bps:g}",
            f"assumed_roundtrip_fee_bps={assumed_roundtrip_fee_bps:g}",
            f"quote_usdt={quote_usdt:g}",
            f"queue_imbalance={queue_imbalance:g}",
            f"ofi_norm={ofi_norm:g}",
            f"micro_alignment={micro_alignment:g}",
            f"trade_count={trade_count}",
            f"evidence_score={evidence_score:g}",
            f"micro_quality_ok={str(micro_quality_ok).lower()}",
            f"trust_ok={str(trust_ok).lower()}",
            f"time_sync_ok={str(time_sync_ok).lower()}",
            f"post_cost_viable={str(post_cost_viable).lower()}",
            (f"cvd_veto_hint={cvd_veto_hint}" if cvd_veto_hint else ""),
            (
                f"ticket_row_reasons={','.join(route_row_reasons)}"
                if route_row_reasons
                else ""
            ),
            (
                f"pattern_router={pattern_status}:{pattern_family}:{pattern_stage}"
                if pattern_status or pattern_family or pattern_stage
                else ""
            ),
        ]
    )

    payload = {
        "action": "build_crypto_shortline_slippage_snapshot",
        "ok": True,
        "status": "ok",
        "generated_at_utc": fmt_utc(reference_now),
        "route_symbol": symbol,
        "route_action": action,
        "remote_market": remote_market,
        "snapshot_status": snapshot_status,
        "snapshot_brief": snapshot_brief,
        "snapshot_decision": snapshot_decision,
        "blocker_title": blocker_title,
        "blocker_target_artifact": "crypto_shortline_slippage_snapshot",
        "blocker_detail": blocker_detail,
        "next_action": snapshot_decision,
        "next_action_target_artifact": "crypto_shortline_slippage_snapshot",
        "done_when": done_when,
        "pattern_family": pattern_family,
        "pattern_stage": pattern_stage,
        "pattern_status": pattern_status,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "max_slippage_bps": max_slippage_bps,
        "estimated_entry_cost_bps": estimated_entry_cost_bps,
        "assumed_roundtrip_fee_bps": assumed_roundtrip_fee_bps,
        "estimated_roundtrip_cost_bps": estimated_roundtrip_cost_bps,
        "target_distance_bps": target_distance_bps,
        "stop_distance_bps": stop_distance_bps,
        "cost_to_target_ratio": cost_to_target_ratio,
        "cost_to_stop_ratio": cost_to_stop_ratio,
        "post_cost_viable": post_cost_viable,
        "queue_imbalance": queue_imbalance,
        "ofi_norm": ofi_norm,
        "micro_alignment": micro_alignment,
        "trade_count": trade_count,
        "evidence_score": evidence_score,
        "micro_quality_ok": micro_quality_ok,
        "trust_ok": trust_ok,
        "time_sync_ok": time_sync_ok,
        "cvd_veto_hint": cvd_veto_hint,
        "quote_usdt": quote_usdt,
        "ticket_row_reasons": route_row_reasons,
        "artifacts": {
            "remote_intent_queue": str(intent_path) if intent_path else "",
            "crypto_route_operator_brief": str(operator_path),
            "signal_to_order_tickets": str(tickets_path),
            "crypto_shortline_live_orderflow_snapshot": str(live_orderflow_snapshot_path)
            if live_orderflow_snapshot_path
            else "",
            "crypto_shortline_pattern_router": str(pattern_router_path)
            if pattern_router_path
            else "",
        },
    }

    stamp = reference_now.strftime("%Y%m%dT%H%M%SZ")
    artifact = review_dir / f"{stamp}_crypto_shortline_slippage_snapshot.json"
    markdown = review_dir / f"{stamp}_crypto_shortline_slippage_snapshot.md"
    checksum = review_dir / f"{stamp}_crypto_shortline_slippage_snapshot_checksum.json"
    artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown.write_text(render_markdown(payload), encoding="utf-8")
    checksum.write_text(
        json.dumps(
            {
                "artifact": str(artifact),
                "sha256": sha256_file(artifact),
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
        keep=args.artifact_keep,
        ttl_hours=args.artifact_ttl_hours,
    )
    payload["artifact"] = str(artifact)
    payload["markdown"] = str(markdown)
    payload["checksum"] = str(checksum)
    payload["pruned_keep"] = pruned_keep
    payload["pruned_age"] = pruned_age

    artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
