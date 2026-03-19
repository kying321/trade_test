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


def find_route_row(tickets_payload: dict[str, Any], symbol: str) -> dict[str, Any]:
    normalized = text(symbol).upper()
    for row in as_list(tickets_payload.get("tickets")):
        item = as_dict(row)
        if text(item.get("symbol")).upper() == normalized:
            return item
    return {}


def bounded_component(value: float, limit: float) -> float:
    effective_limit = max(1e-9, float(limit))
    return max(0.0, min(abs(float(value)) / effective_limit, 1.0))


def compute_fill_capacity_score(
    *,
    snapshot_status: str,
    micro_quality_ok: bool,
    trust_ok: bool,
    time_sync_ok: bool,
    fill_capacity_viable: bool,
    entry_budget_usage_ratio: float | None,
    trade_count: int,
    evidence_score: float,
    cvd_veto_hint: str,
) -> float:
    score = 0.0
    if snapshot_status == "live_orderflow_snapshot_ready":
        score += 20.0
    if fill_capacity_viable:
        score += 25.0
    usage_ratio = max(0.0, float(entry_budget_usage_ratio or 0.0))
    score += 15.0 * max(0.0, min(1.0 - usage_ratio, 1.0))
    if micro_quality_ok:
        score += 10.0
    if trust_ok:
        score += 10.0
    if time_sync_ok:
        score += 5.0
    score += 7.5 * bounded_component(float(trade_count), 200.0)
    score += 7.5 * max(0.0, min(float(evidence_score), 1.0))
    if cvd_veto_hint:
        score -= 10.0
    return max(0.0, min(round(score, 3), 100.0))


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
        "*_crypto_shortline_fill_capacity_watch.json",
        "*_crypto_shortline_fill_capacity_watch.md",
        "*_crypto_shortline_fill_capacity_watch_checksum.json",
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


def render_markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Crypto Shortline Fill Capacity Watch",
            "",
            f"- brief: `{text(payload.get('watch_brief'))}`",
            f"- decision: `{text(payload.get('watch_decision'))}`",
            f"- pattern_family: `{text(payload.get('pattern_family')) or '-'}`",
            f"- pattern_stage: `{text(payload.get('pattern_stage')) or '-'}`",
            f"- quote_usdt: `{payload.get('quote_usdt')}`",
            f"- max_slippage_bps: `{payload.get('max_slippage_bps')}`",
            f"- estimated_entry_cost_bps: `{payload.get('estimated_entry_cost_bps')}`",
            f"- entry_headroom_bps: `{payload.get('entry_headroom_bps')}`",
            f"- entry_budget_usage_ratio: `{payload.get('entry_budget_usage_ratio')}`",
            f"- fill_capacity_score: `{payload.get('fill_capacity_score')}`",
            f"- fill_capacity_viable: `{payload.get('fill_capacity_viable')}`",
            f"- trust_ok: `{payload.get('trust_ok')}`",
            f"- micro_quality_ok: `{payload.get('micro_quality_ok')}`",
            f"- blocker: `{text(payload.get('blocker_detail'))}`",
            f"- done_when: `{text(payload.get('done_when'))}`",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a source-owned crypto shortline fill-capacity watch artifact."
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

    pattern_router_path = find_latest(review_dir, "*_crypto_shortline_pattern_router.json", reference_now)
    live_orderflow_snapshot_path = find_latest(
        review_dir, "*_crypto_shortline_live_orderflow_snapshot.json", reference_now
    )
    slippage_snapshot_path = find_latest(
        review_dir, "*_crypto_shortline_slippage_snapshot.json", reference_now
    )
    pattern_router_payload = (
        load_json_mapping(pattern_router_path)
        if pattern_router_path is not None and pattern_router_path.exists()
        else {}
    )
    live_orderflow_payload = (
        load_json_mapping(live_orderflow_snapshot_path)
        if live_orderflow_snapshot_path is not None and live_orderflow_snapshot_path.exists()
        else {}
    )
    slippage_snapshot_payload = (
        load_json_mapping(slippage_snapshot_path)
        if slippage_snapshot_path is not None and slippage_snapshot_path.exists()
        else {}
    )

    route_levels = as_dict(route_row.get("levels"))
    route_execution = as_dict(route_row.get("execution"))
    route_sizing = as_dict(route_row.get("sizing"))
    route_row_reasons = dedupe_text(as_list(route_row.get("reasons")))
    pattern_family = text(pattern_router_payload.get("pattern_family"))
    pattern_stage = text(pattern_router_payload.get("pattern_stage"))
    pattern_status = text(pattern_router_payload.get("pattern_status"))

    quote_usdt = to_float(route_sizing.get("quote_usdt"))
    max_slippage_bps = to_float(route_execution.get("max_slippage_bps"))
    estimated_entry_cost_bps = to_float(slippage_snapshot_payload.get("estimated_entry_cost_bps"))
    snapshot_status = text(live_orderflow_payload.get("snapshot_status"))
    snapshot_brief = text(live_orderflow_payload.get("snapshot_brief"))
    snapshot_decision = text(live_orderflow_payload.get("snapshot_decision"))
    slippage_status = text(slippage_snapshot_payload.get("snapshot_status"))
    slippage_brief = text(slippage_snapshot_payload.get("snapshot_brief"))
    slippage_decision = text(slippage_snapshot_payload.get("snapshot_decision"))
    micro_quality_ok = bool(live_orderflow_payload.get("micro_quality_ok", False))
    trust_ok = bool(live_orderflow_payload.get("trust_ok", False))
    time_sync_ok = bool(live_orderflow_payload.get("time_sync_ok", False))
    trade_count = int(to_float(live_orderflow_payload.get("trade_count")))
    evidence_score = to_float(live_orderflow_payload.get("evidence_score"))
    cvd_veto_hint = text(live_orderflow_payload.get("cvd_veto_hint"))

    entry_headroom_bps = (
        round(max(0.0, max_slippage_bps - estimated_entry_cost_bps), 3)
        if max_slippage_bps > 0
        else 0.0
    )
    entry_budget_usage_ratio = (
        round(estimated_entry_cost_bps / max_slippage_bps, 6)
        if max_slippage_bps > 0
        else None
    )
    value_rotation_active = pattern_family == "value_rotation_scalp"
    fill_capacity_viable = bool(
        route_row
        and quote_usdt > 0.0
        and max_slippage_bps > 0.0
        and estimated_entry_cost_bps > 0.0
        and estimated_entry_cost_bps <= max_slippage_bps
        and entry_headroom_bps >= 1.0
        and micro_quality_ok
        and trust_ok
        and time_sync_ok
        and not cvd_veto_hint
    )
    fill_capacity_score = compute_fill_capacity_score(
        snapshot_status=snapshot_status,
        micro_quality_ok=micro_quality_ok,
        trust_ok=trust_ok,
        time_sync_ok=time_sync_ok,
        fill_capacity_viable=fill_capacity_viable,
        entry_budget_usage_ratio=entry_budget_usage_ratio,
        trade_count=trade_count,
        evidence_score=evidence_score,
        cvd_veto_hint=cvd_veto_hint,
    )

    if not route_row:
        watch_status = "fill_capacity_missing_ticket_row"
        watch_decision = "refresh_ticket_surface_then_recheck_fill_capacity"
        blocker_title = "Generate route ticket before fill-capacity review"
        done_when = f"{symbol} has a fresh route ticket row with explicit entry and sizing fields"
    elif live_orderflow_snapshot_path is None or not live_orderflow_payload:
        watch_status = "fill_capacity_missing_live_orderflow"
        watch_decision = "capture_live_orderflow_before_fill_capacity_check"
        blocker_title = "Collect live orderflow before fill-capacity review"
        done_when = f"{symbol} has a matching live orderflow snapshot for fill-capacity validation"
    elif slippage_snapshot_path is None or not slippage_snapshot_payload:
        watch_status = "fill_capacity_missing_post_cost_source"
        watch_decision = "build_post_cost_snapshot_then_recheck_fill_capacity"
        blocker_title = "Build post-cost source before fill-capacity review"
        done_when = f"{symbol} has a slippage/post-cost source with estimated entry cost"
    elif fill_capacity_viable:
        if value_rotation_active:
            watch_status = "value_rotation_scalp_fill_capacity_clear"
            watch_decision = "recheck_execution_gate_after_value_rotation_fill_capacity_clear"
            blocker_title = "Value-rotation fill capacity clear for shortline scalp promotion"
        else:
            watch_status = "fill_capacity_watch_clear"
            watch_decision = "recheck_execution_gate_after_fill_capacity_clear"
            blocker_title = "Fill capacity clear for shortline setup promotion"
        done_when = f"{symbol} reintroduces fill-capacity degradation or leaves Setup_Ready"
    else:
        if value_rotation_active:
            watch_status = "value_rotation_scalp_fill_capacity_constrained"
            watch_decision = "repair_value_rotation_fill_capacity_then_recheck_execution_gate"
            blocker_title = "Repair value-rotation fill capacity before shortline scalp promotion"
            done_when = (
                f"{symbol} keeps entry cost within slippage budget, trust/time-sync stay clear, "
                "and fill_capacity_viable flips true while the value-rotation scalp remains active"
            )
        else:
            watch_status = "fill_capacity_constrained"
            watch_decision = "repair_fill_capacity_then_recheck_execution_gate"
            blocker_title = "Repair shortline fill capacity before setup promotion"
            done_when = f"{symbol} keeps entry cost inside slippage budget and fill_capacity_viable flips true"

    watch_brief = ":".join([watch_status, symbol or "-", watch_decision, remote_market or "-"])
    blocker_detail = join_unique(
        [
            snapshot_brief,
            slippage_brief,
            (
                f"pattern_router={pattern_status}:{pattern_family}:{pattern_stage}"
                if pattern_status or pattern_family or pattern_stage
                else ""
            ),
            f"quote_usdt={quote_usdt:g}",
            f"max_slippage_bps={max_slippage_bps:g}",
            f"estimated_entry_cost_bps={estimated_entry_cost_bps:g}",
            f"entry_headroom_bps={entry_headroom_bps:g}",
            (
                f"entry_budget_usage_ratio={entry_budget_usage_ratio:g}"
                if entry_budget_usage_ratio is not None
                else ""
            ),
            f"fill_capacity_score={fill_capacity_score:g}",
            f"fill_capacity_viable={str(fill_capacity_viable).lower()}",
            f"micro_quality_ok={str(micro_quality_ok).lower()}",
            f"trust_ok={str(trust_ok).lower()}",
            f"time_sync_ok={str(time_sync_ok).lower()}",
            f"trade_count={trade_count}",
            f"evidence_score={evidence_score:g}",
            (f"cvd_veto_hint={cvd_veto_hint}" if cvd_veto_hint else ""),
            (
                f"ticket_row_reasons={','.join(route_row_reasons)}"
                if route_row_reasons
                else ""
            ),
            f"snapshot_decision={snapshot_decision}" if snapshot_decision else "",
            f"slippage_decision={slippage_decision}" if slippage_decision else "",
        ]
    )

    payload = {
        "action": "build_crypto_shortline_fill_capacity_watch",
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
        "blocker_target_artifact": "crypto_shortline_fill_capacity_watch",
        "blocker_detail": blocker_detail,
        "next_action": watch_decision,
        "next_action_target_artifact": "crypto_shortline_fill_capacity_watch",
        "done_when": done_when,
        "pattern_family": pattern_family,
        "pattern_stage": pattern_stage,
        "pattern_status": pattern_status,
        "snapshot_status": snapshot_status,
        "snapshot_brief": snapshot_brief,
        "snapshot_decision": snapshot_decision,
        "slippage_snapshot_status": slippage_status,
        "slippage_snapshot_brief": slippage_brief,
        "slippage_snapshot_decision": slippage_decision,
        "quote_usdt": quote_usdt,
        "max_slippage_bps": max_slippage_bps,
        "estimated_entry_cost_bps": estimated_entry_cost_bps,
        "entry_headroom_bps": entry_headroom_bps,
        "entry_budget_usage_ratio": entry_budget_usage_ratio,
        "fill_capacity_score": fill_capacity_score,
        "fill_capacity_viable": fill_capacity_viable,
        "micro_quality_ok": micro_quality_ok,
        "trust_ok": trust_ok,
        "time_sync_ok": time_sync_ok,
        "trade_count": trade_count,
        "evidence_score": evidence_score,
        "cvd_veto_hint": cvd_veto_hint,
        "ticket_row_reasons": route_row_reasons,
        "artifacts": {
            "remote_intent_queue": str(intent_path) if intent_path else "",
            "crypto_route_operator_brief": str(operator_path),
            "signal_to_order_tickets": str(tickets_path),
            "crypto_shortline_live_orderflow_snapshot": str(live_orderflow_snapshot_path)
            if live_orderflow_snapshot_path
            else "",
            "crypto_shortline_slippage_snapshot": str(slippage_snapshot_path)
            if slippage_snapshot_path
            else "",
            "crypto_shortline_pattern_router": str(pattern_router_path)
            if pattern_router_path
            else "",
        },
    }

    stamp = reference_now.strftime("%Y%m%dT%H%M%SZ")
    artifact = review_dir / f"{stamp}_crypto_shortline_fill_capacity_watch.json"
    markdown = review_dir / f"{stamp}_crypto_shortline_fill_capacity_watch.md"
    checksum = review_dir / f"{stamp}_crypto_shortline_fill_capacity_watch_checksum.json"
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
        keep=max(1, int(args.artifact_keep)),
        ttl_hours=float(args.artifact_ttl_hours),
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
