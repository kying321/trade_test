#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
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
TIMESTAMP_RE = re.compile(r"^(?P<stamp>\d{8}T\d{6}Z)_")
FUTURE_STAMP_GRACE_MINUTES = 5


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


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


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
    return dt.datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)


def artifact_sort_key(path: Path, reference_now: dt.datetime | None = None) -> tuple[int, str, float, str]:
    stamp_dt = parsed_artifact_stamp(path)
    effective_now = reference_now or now_utc()
    future_cutoff = effective_now + dt.timedelta(minutes=FUTURE_STAMP_GRACE_MINUTES)
    is_future = bool(stamp_dt and stamp_dt > future_cutoff)
    return (0 if is_future else 1, artifact_stamp(path), path.stat().st_mtime, path.name)


def find_latest(review_dir: Path, pattern: str, reference_now: dt.datetime | None = None) -> Path | None:
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
        "*_crypto_shortline_signal_quality_watch.json",
        "*_crypto_shortline_signal_quality_watch.md",
        "*_crypto_shortline_signal_quality_watch_checksum.json",
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


def find_signal_row(payload: dict[str, Any], symbol: str) -> dict[str, Any]:
    rows = as_list(as_dict(payload.get("signals")).get(symbol))
    for row in rows:
        item = as_dict(row)
        if text(item.get("symbol")).upper() == symbol.upper():
            return item
    return as_dict(rows[0]) if rows else {}


def render_markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Crypto Shortline Signal Quality Watch",
            "",
            f"- brief: `{text(payload.get('watch_brief'))}`",
            f"- decision: `{text(payload.get('watch_decision'))}`",
            f"- pattern_family: `{text(payload.get('pattern_family')) or '-'}`",
            f"- pattern_stage: `{text(payload.get('pattern_stage')) or '-'}`",
            f"- live_orderflow_snapshot_brief: `{text(payload.get('live_orderflow_snapshot_brief'))}`",
            f"- sizing_watch_brief: `{text(payload.get('sizing_watch_brief'))}`",
            f"- confidence_below_threshold: `{payload.get('confidence_below_threshold')}`",
            f"- convexity_below_threshold: `{payload.get('convexity_below_threshold')}`",
            f"- size_below_min_notional: `{payload.get('size_below_min_notional')}`",
            f"- signal_confidence: `{payload.get('signal_confidence')}`",
            f"- signal_convexity_ratio: `{payload.get('signal_convexity_ratio')}`",
            f"- quote_usdt: `{payload.get('quote_usdt')}`",
            f"- min_notional_usdt: `{payload.get('min_notional_usdt')}`",
            f"- micro_alignment: `{payload.get('micro_alignment')}`",
            f"- queue_imbalance: `{payload.get('queue_imbalance')}`",
            f"- ofi_norm: `{payload.get('ofi_norm')}`",
            f"- time_sync_ok: `{payload.get('time_sync_ok')}`",
            f"- micro_quality_ok: `{payload.get('micro_quality_ok')}`",
            f"- blocker: `{text(payload.get('blocker_detail'))}`",
            f"- done_when: `{text(payload.get('done_when'))}`",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a source-owned crypto shortline signal-quality watch artifact."
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
    signal_source_path = find_latest(
        review_dir, "*_crypto_shortline_signal_source.json", reference_now
    )
    diagnosis_path = find_latest(
        review_dir, "*_crypto_shortline_ticket_constraint_diagnosis.json", reference_now
    )
    sizing_watch_path = find_latest(
        review_dir, "*_crypto_shortline_sizing_watch.json", reference_now
    )
    if operator_path is None:
        raise SystemExit("missing_required_artifacts:crypto_route_operator_brief")

    intent_payload = (
        load_json_mapping(intent_path) if intent_path is not None and intent_path.exists() else {}
    )
    operator_payload = load_json_mapping(operator_path)
    symbol = route_symbol(intent_payload, operator_payload)
    tickets_path = select_ticket_surface_path(review_dir, symbol, reference_now)
    if tickets_path is None:
        raise SystemExit("missing_required_artifacts:signal_to_order_tickets")

    tickets_payload = load_json_mapping(tickets_path)
    signal_source_payload = (
        load_json_mapping(signal_source_path)
        if signal_source_path is not None and signal_source_path.exists()
        else {}
    )
    diagnosis_payload = (
        load_json_mapping(diagnosis_path)
        if diagnosis_path is not None and diagnosis_path.exists()
        else {}
    )
    sizing_watch_payload = (
        load_json_mapping(sizing_watch_path)
        if sizing_watch_path is not None and sizing_watch_path.exists()
        else {}
    )
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

    action = route_action(intent_payload, operator_payload)
    remote_market = text(intent_payload.get("remote_market")) or "portfolio_margin_um"
    route_row = find_route_row(tickets_payload, symbol)
    route_signal = as_dict(route_row.get("signal"))
    signal_row = find_signal_row(signal_source_payload, symbol) or route_signal
    route_sizing = as_dict(route_row.get("sizing"))
    route_row_reasons = dedupe_text(as_list(route_row.get("reasons")))

    confidence_below_threshold = "confidence_below_threshold" in route_row_reasons
    convexity_below_threshold = "convexity_below_threshold" in route_row_reasons
    size_below_min_notional = "size_below_min_notional" in route_row_reasons
    signal_confidence = to_float(route_signal.get("confidence"), to_float(signal_row.get("confidence")))
    signal_convexity_ratio = to_float(
        route_signal.get("convexity_ratio"), to_float(signal_row.get("convexity_ratio"))
    )
    conviction = to_float(route_sizing.get("conviction"))
    quote_usdt = to_float(route_sizing.get("quote_usdt"))
    min_notional_usdt = to_float(route_sizing.get("min_notional_usdt"))
    risk_budget_usdt = to_float(route_sizing.get("risk_budget_usdt"))
    diagnosis_brief = text(diagnosis_payload.get("diagnosis_brief"))
    diagnosis_decision = text(diagnosis_payload.get("diagnosis_decision"))
    sizing_watch_brief = text(sizing_watch_payload.get("watch_brief"))
    sizing_watch_status = text(sizing_watch_payload.get("watch_status"))
    sizing_watch_decision = text(sizing_watch_payload.get("watch_decision"))
    pattern_family = text(pattern_router_payload.get("pattern_family"))
    pattern_stage = text(pattern_router_payload.get("pattern_stage"))
    pattern_status = text(pattern_router_payload.get("pattern_status"))
    live_orderflow_snapshot_brief = text(live_orderflow_snapshot_payload.get("snapshot_brief"))
    live_orderflow_snapshot_status = text(live_orderflow_snapshot_payload.get("snapshot_status"))
    live_orderflow_snapshot_decision = text(live_orderflow_snapshot_payload.get("snapshot_decision"))
    micro_quality_ok = bool(live_orderflow_snapshot_payload.get("micro_quality_ok", False))
    time_sync_ok = bool(live_orderflow_snapshot_payload.get("time_sync_ok", False))
    queue_imbalance = to_float(live_orderflow_snapshot_payload.get("queue_imbalance"))
    ofi_norm = to_float(live_orderflow_snapshot_payload.get("ofi_norm"))
    micro_alignment = to_float(live_orderflow_snapshot_payload.get("micro_alignment"))
    cvd_delta_ratio = to_float(live_orderflow_snapshot_payload.get("cvd_delta_ratio"))
    cvd_context_mode = text(live_orderflow_snapshot_payload.get("cvd_context_mode"))
    cvd_veto_hint = text(live_orderflow_snapshot_payload.get("cvd_veto_hint"))
    cvd_locality_status = text(live_orderflow_snapshot_payload.get("cvd_locality_status"))
    cvd_attack_presence = text(live_orderflow_snapshot_payload.get("cvd_attack_presence"))

    value_rotation_active = pattern_family == "value_rotation_scalp"

    if not any([confidence_below_threshold, convexity_below_threshold]):
        watch_status = "signal_quality_watch_clear"
        watch_decision = "recheck_execution_gate_after_signal_quality_clear"
        blocker_title = "Shortline signal quality clear for setup promotion"
        done_when = (
            f"{symbol} reintroduces confidence/convexity blockers or leaves Setup_Ready"
        )
    elif confidence_below_threshold and convexity_below_threshold:
        if value_rotation_active:
            watch_status = "value_rotation_scalp_signal_quality_confidence_convexity_below_threshold"
            watch_decision = "improve_value_rotation_signal_quality_then_recheck_execution_gate"
            blocker_title = "Improve value-rotation signal quality before shortline scalp promotion"
            done_when = (
                f"{symbol} clears confidence_below_threshold and convexity_below_threshold while "
                "the value-rotation scalp remains active"
            )
        else:
            watch_status = "signal_quality_confidence_convexity_below_threshold"
            watch_decision = "improve_shortline_signal_quality_then_recheck_execution_gate"
            blocker_title = "Improve shortline signal quality before shortline setup promotion"
            done_when = (
                f"{symbol} clears confidence_below_threshold and convexity_below_threshold on the route ticket"
            )
    elif confidence_below_threshold:
        if value_rotation_active:
            watch_status = "value_rotation_scalp_signal_confidence_below_threshold"
            watch_decision = "improve_value_rotation_signal_quality_then_recheck_execution_gate"
            blocker_title = "Improve value-rotation signal quality before shortline scalp promotion"
            done_when = (
                f"{symbol} clears confidence_below_threshold while the value-rotation scalp remains active"
            )
        else:
            watch_status = "signal_confidence_below_threshold"
            watch_decision = "improve_shortline_signal_quality_then_recheck_execution_gate"
            blocker_title = "Improve shortline signal quality before shortline setup promotion"
            done_when = (
                f"{symbol} clears confidence_below_threshold on the route ticket"
            )
    else:
        if value_rotation_active:
            watch_status = "value_rotation_scalp_signal_convexity_below_threshold"
            watch_decision = "improve_value_rotation_signal_quality_then_recheck_execution_gate"
            blocker_title = "Improve value-rotation signal quality before shortline scalp promotion"
            done_when = (
                f"{symbol} clears convexity_below_threshold while the value-rotation scalp remains active"
            )
        else:
            watch_status = "signal_convexity_below_threshold"
            watch_decision = "improve_shortline_signal_quality_then_recheck_execution_gate"
            blocker_title = "Improve shortline signal quality before shortline setup promotion"
            done_when = (
                f"{symbol} clears convexity_below_threshold on the route ticket"
            )

    watch_brief = ":".join([watch_status, symbol or "-", watch_decision, remote_market or "-"])
    blocker_detail = join_unique(
        [
            f"signal_confidence={signal_confidence:g}",
            f"signal_convexity_ratio={signal_convexity_ratio:g}",
            f"conviction={conviction:g}",
            f"quote_usdt={quote_usdt:g}",
            f"min_notional_usdt={min_notional_usdt:g}",
            f"risk_budget_usdt={risk_budget_usdt:g}",
            f"confidence_below_threshold={str(confidence_below_threshold).lower()}",
            f"convexity_below_threshold={str(convexity_below_threshold).lower()}",
            f"size_below_min_notional={str(size_below_min_notional).lower()}",
            (
                f"ticket_row_reasons={','.join(route_row_reasons)}"
                if route_row_reasons
                else ""
            ),
            live_orderflow_snapshot_brief,
            (
                f"live_orderflow_snapshot_status={live_orderflow_snapshot_status}"
                if live_orderflow_snapshot_status
                else ""
            ),
            (
                f"live_orderflow_snapshot_decision={live_orderflow_snapshot_decision}"
                if live_orderflow_snapshot_decision
                else ""
            ),
            f"micro_quality_ok={str(micro_quality_ok).lower()}",
            f"time_sync_ok={str(time_sync_ok).lower()}",
            f"queue_imbalance={queue_imbalance:g}",
            f"ofi_norm={ofi_norm:g}",
            f"micro_alignment={micro_alignment:g}",
            f"cvd_delta_ratio={cvd_delta_ratio:g}",
            f"cvd_context_mode={cvd_context_mode}" if cvd_context_mode else "",
            f"cvd_veto_hint={cvd_veto_hint}" if cvd_veto_hint else "",
            f"cvd_locality_status={cvd_locality_status}" if cvd_locality_status else "",
            f"cvd_attack_presence={cvd_attack_presence}" if cvd_attack_presence else "",
            (
                f"sizing_watch={sizing_watch_brief}:{sizing_watch_decision}"
                if sizing_watch_brief or sizing_watch_decision
                else ""
            ),
            (
                f"pattern_router={pattern_status}:{pattern_family}:{pattern_stage}"
                if pattern_status or pattern_family or pattern_stage
                else ""
            ),
            diagnosis_brief,
            diagnosis_decision,
        ]
    )

    payload = {
        "action": "build_crypto_shortline_signal_quality_watch",
        "ok": True,
        "status": "ok",
        "generated_at_utc": fmt_utc(reference_now),
        "route_symbol": symbol,
        "route_action": action,
        "remote_market": remote_market,
        "watch_status": watch_status,
        "watch_brief": watch_brief,
        "watch_decision": watch_decision,
        "blocker_title": blocker_title,
        "blocker_target_artifact": "crypto_shortline_signal_quality_watch",
        "blocker_detail": blocker_detail,
        "next_action": watch_decision,
        "next_action_target_artifact": "crypto_shortline_signal_quality_watch",
        "done_when": done_when,
        "pattern_family": pattern_family,
        "pattern_stage": pattern_stage,
        "pattern_status": pattern_status,
        "confidence_below_threshold": confidence_below_threshold,
        "convexity_below_threshold": convexity_below_threshold,
        "size_below_min_notional": size_below_min_notional,
        "signal_confidence": signal_confidence,
        "signal_convexity_ratio": signal_convexity_ratio,
        "conviction": conviction,
        "quote_usdt": quote_usdt,
        "min_notional_usdt": min_notional_usdt,
        "risk_budget_usdt": risk_budget_usdt,
        "sizing_watch_status": sizing_watch_status,
        "sizing_watch_brief": sizing_watch_brief,
        "sizing_watch_decision": sizing_watch_decision,
        "live_orderflow_snapshot_status": live_orderflow_snapshot_status,
        "live_orderflow_snapshot_brief": live_orderflow_snapshot_brief,
        "live_orderflow_snapshot_decision": live_orderflow_snapshot_decision,
        "micro_quality_ok": micro_quality_ok,
        "time_sync_ok": time_sync_ok,
        "queue_imbalance": queue_imbalance,
        "ofi_norm": ofi_norm,
        "micro_alignment": micro_alignment,
        "cvd_delta_ratio": cvd_delta_ratio,
        "cvd_context_mode": cvd_context_mode,
        "cvd_veto_hint": cvd_veto_hint,
        "cvd_locality_status": cvd_locality_status,
        "cvd_attack_presence": cvd_attack_presence,
        "ticket_row_reasons": route_row_reasons,
        "artifacts": {
            "remote_intent_queue": str(intent_path) if intent_path else "",
            "crypto_route_operator_brief": str(operator_path),
            "signal_to_order_tickets": str(tickets_path),
            "crypto_shortline_signal_source": str(signal_source_path) if signal_source_path else "",
            "crypto_shortline_live_orderflow_snapshot": str(live_orderflow_snapshot_path)
            if live_orderflow_snapshot_path
            else "",
            "crypto_shortline_pattern_router": str(pattern_router_path)
            if pattern_router_path
            else "",
            "crypto_shortline_sizing_watch": str(sizing_watch_path)
            if sizing_watch_path
            else "",
            "crypto_shortline_ticket_constraint_diagnosis": str(diagnosis_path)
            if diagnosis_path
            else "",
        },
    }

    stamp = reference_now.strftime("%Y%m%dT%H%M%SZ")
    artifact = review_dir / f"{stamp}_crypto_shortline_signal_quality_watch.json"
    markdown = review_dir / f"{stamp}_crypto_shortline_signal_quality_watch.md"
    checksum = review_dir / f"{stamp}_crypto_shortline_signal_quality_watch_checksum.json"
    artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown.write_text(render_markdown(payload), encoding="utf-8")
    checksum.write_text(
        json.dumps(
            {
                "artifact": str(artifact),
                "artifact_sha256": sha256_file(artifact),
                "markdown": str(markdown),
                "markdown_sha256": sha256_file(markdown),
                "generated_at_utc": fmt_utc(reference_now),
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
                "generated_at_utc": fmt_utc(reference_now),
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
