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
        "*_crypto_shortline_sizing_watch.json",
        "*_crypto_shortline_sizing_watch.md",
        "*_crypto_shortline_sizing_watch_checksum.json",
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


def render_markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Crypto Shortline Sizing Watch",
            "",
            f"- brief: `{text(payload.get('watch_brief'))}`",
            f"- decision: `{text(payload.get('watch_decision'))}`",
            f"- pattern_family: `{text(payload.get('pattern_family')) or '-'}`",
            f"- pattern_stage: `{text(payload.get('pattern_stage')) or '-'}`",
            f"- quote_usdt: `{payload.get('quote_usdt')}`",
            f"- min_notional_usdt: `{payload.get('min_notional_usdt')}`",
            f"- size_shortfall_usdt: `{payload.get('size_shortfall_usdt')}`",
            f"- size_shortfall_ratio: `{payload.get('size_shortfall_ratio')}`",
            f"- conviction: `{payload.get('conviction')}`",
            f"- risk_budget_usdt: `{payload.get('risk_budget_usdt')}`",
            f"- blocker: `{text(payload.get('blocker_detail'))}`",
            f"- done_when: `{text(payload.get('done_when'))}`",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a source-owned crypto shortline sizing watch artifact."
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
    tickets_path = select_ticket_surface_path(review_dir, symbol, reference_now)
    if tickets_path is None:
        raise SystemExit("missing_required_artifacts:signal_to_order_tickets")
    tickets_payload = load_json_mapping(tickets_path)
    pattern_router_path, pattern_router_payload = load_pattern_router(
        review_dir,
        route_symbol=symbol,
        reference_now=reference_now,
    )

    action = route_action(intent_payload, operator_payload)
    remote_market = text(intent_payload.get("remote_market")) or "portfolio_margin_um"
    route_row = find_route_row(tickets_payload, symbol)
    route_row_reasons = dedupe_text(as_list(route_row.get("reasons")))
    route_sizing = as_dict(route_row.get("sizing"))
    pattern_family = text(pattern_router_payload.get("pattern_family"))
    pattern_stage = text(pattern_router_payload.get("pattern_stage"))
    pattern_status = text(pattern_router_payload.get("pattern_status"))

    size_below_min_notional = "size_below_min_notional" in route_row_reasons
    conviction = to_float(route_sizing.get("conviction"))
    quote_usdt = to_float(route_sizing.get("quote_usdt"))
    min_notional_usdt = to_float(route_sizing.get("min_notional_usdt"))
    risk_budget_usdt = to_float(route_sizing.get("risk_budget_usdt"))
    equity_usdt = to_float(route_sizing.get("equity_usdt"))
    max_alloc_pct = to_float(route_sizing.get("max_alloc_pct"))
    size_shortfall_usdt = max(0.0, min_notional_usdt - quote_usdt)
    size_shortfall_ratio = (
        size_shortfall_usdt / min_notional_usdt if min_notional_usdt > 0 else 0.0
    )

    value_rotation_active = pattern_family == "value_rotation_scalp"

    if not route_row:
        watch_status = "ticket_sizing_row_missing"
        watch_decision = "refresh_ticket_surface_then_recheck_execution_gate"
        blocker_title = "Generate route ticket sizing before shortline setup promotion"
        done_when = f"{symbol} has a fresh ticket row with explicit sizing fields"
    elif not size_below_min_notional:
        watch_status = "ticket_sizing_clear"
        watch_decision = "recheck_execution_gate_after_size_clear"
        blocker_title = "Shortline ticket sizing clear for setup promotion"
        done_when = f"{symbol} reintroduces size_below_min_notional or leaves Setup_Ready"
    else:
        if value_rotation_active:
            watch_status = "value_rotation_scalp_ticket_size_below_min_notional"
            watch_decision = "raise_value_rotation_effective_size_then_recheck_execution_gate"
            blocker_title = "Raise effective value-rotation size before shortline scalp promotion"
            done_when = (
                f"{symbol} keeps quote_usdt above min_notional_usdt while the value-rotation scalp "
                "remains active and the route ticket drops size_below_min_notional"
            )
        else:
            watch_status = "ticket_size_below_min_notional"
            watch_decision = "raise_effective_shortline_size_then_recheck_execution_gate"
            blocker_title = "Raise effective shortline size before shortline setup promotion"
            done_when = (
                f"{symbol} keeps quote_usdt above min_notional_usdt and the route ticket drops "
                "size_below_min_notional"
            )

    watch_brief = ":".join([watch_status, symbol or "-", watch_decision, remote_market or "-"])
    blocker_detail = join_unique(
        [
            f"quote_usdt={quote_usdt:g}",
            f"min_notional_usdt={min_notional_usdt:g}",
            f"size_shortfall_usdt={size_shortfall_usdt:g}",
            f"size_shortfall_ratio={size_shortfall_ratio:g}",
            f"conviction={conviction:g}",
            f"risk_budget_usdt={risk_budget_usdt:g}",
            f"equity_usdt={equity_usdt:g}",
            f"max_alloc_pct={max_alloc_pct:g}",
            f"size_below_min_notional={str(size_below_min_notional).lower()}",
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
        "action": "build_crypto_shortline_sizing_watch",
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
        "blocker_target_artifact": "crypto_shortline_sizing_watch",
        "blocker_detail": blocker_detail,
        "next_action": watch_decision,
        "next_action_target_artifact": "crypto_shortline_sizing_watch",
        "done_when": done_when,
        "pattern_family": pattern_family,
        "pattern_stage": pattern_stage,
        "pattern_status": pattern_status,
        "size_below_min_notional": size_below_min_notional,
        "conviction": conviction,
        "quote_usdt": quote_usdt,
        "min_notional_usdt": min_notional_usdt,
        "risk_budget_usdt": risk_budget_usdt,
        "equity_usdt": equity_usdt,
        "max_alloc_pct": max_alloc_pct,
        "size_shortfall_usdt": size_shortfall_usdt,
        "size_shortfall_ratio": size_shortfall_ratio,
        "ticket_row_reasons": route_row_reasons,
        "artifacts": {
            "remote_intent_queue": str(intent_path) if intent_path else "",
            "crypto_route_operator_brief": str(operator_path),
            "signal_to_order_tickets": str(tickets_path),
            "crypto_shortline_pattern_router": str(pattern_router_path)
            if pattern_router_path
            else "",
        },
    }

    stamp = reference_now.strftime("%Y%m%dT%H%M%SZ")
    artifact = review_dir / f"{stamp}_crypto_shortline_sizing_watch.json"
    markdown = review_dir / f"{stamp}_crypto_shortline_sizing_watch.md"
    checksum = review_dir / f"{stamp}_crypto_shortline_sizing_watch_checksum.json"
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
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
