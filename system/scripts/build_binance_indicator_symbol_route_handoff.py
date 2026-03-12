#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any


TIMESTAMP_RE = re.compile(r"^(?P<stamp>\d{8}T\d{6}Z)_")
FUTURE_STAMP_GRACE_MINUTES = 5
COMBO_CANONICAL_ALIASES = {
    "ad_breakout": "cvd_breakout",
    "ad_rsi_breakout": "cvd_rsi_breakout",
    "ad_rsi_vol_breakout": "cvd_rsi_vol_breakout",
    "ad_rsi_reclaim": "cvd_rsi_reclaim",
    "taker_oi_ad_breakout": "taker_oi_cvd_breakout",
    "taker_oi_ad_rsi_breakout": "taker_oi_cvd_rsi_breakout",
}


def canonical_combo_id(combo_id: Any) -> str:
    combo = str(combo_id or "")
    return COMBO_CANONICAL_ALIASES.get(combo, combo)


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


def artifact_stamp(path: Path) -> str:
    match = TIMESTAMP_RE.match(path.name)
    return match.group("stamp") if match else ""


def parsed_artifact_stamp(path: Path) -> dt.datetime | None:
    stamp = artifact_stamp(path)
    if not stamp:
        return None
    return dt.datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)


def artifact_sort_key(path: Path) -> tuple[int, str, float, str]:
    stamp_dt = parsed_artifact_stamp(path)
    future_cutoff = now_utc() + dt.timedelta(minutes=FUTURE_STAMP_GRACE_MINUTES)
    is_future = bool(stamp_dt and stamp_dt > future_cutoff)
    return (0 if is_future else 1, artifact_stamp(path), path.stat().st_mtime, path.name)


def latest_combo_playbook(review_dir: Path) -> Path:
    candidates = list(review_dir.glob("*_binance_indicator_combo_playbook.json"))
    if not candidates:
        raise FileNotFoundError("no_binance_indicator_combo_playbook_artifact")
    return max(candidates, key=artifact_sort_key)


def latest_beta_leg_window_report(review_dir: Path) -> tuple[Path | None, dict[str, Any] | None]:
    candidates = list(review_dir.glob("*_binance_indicator_native_beta_leg_window_report.json"))
    if not candidates:
        return None, None
    path = max(candidates, key=artifact_sort_key)
    return path, json.loads(path.read_text(encoding="utf-8"))


def latest_bnb_flow_focus(review_dir: Path) -> tuple[Path | None, dict[str, Any] | None]:
    candidates = list(review_dir.glob("*_binance_indicator_bnb_flow_focus.json"))
    if not candidates:
        return None, None
    best_path: Path | None = None
    best_payload: dict[str, Any] | None = None
    best_key: tuple[int, tuple[int, str, float, str]] | None = None
    for path in candidates:
        payload = json.loads(path.read_text(encoding="utf-8"))
        richness = 0
        if str(payload.get("source_mode") or "").strip() == "direct_bnb_native":
            richness += 2
        if str(payload.get("flow_window_floor") or "").strip():
            richness += 1
        if str(payload.get("comparative_window_takeaway") or "").strip():
            richness += 1
        candidate_key = (richness, artifact_sort_key(path))
        if best_key is None or candidate_key > best_key:
            best_key = candidate_key
            best_path = path
            best_payload = payload
    return best_path, best_payload


def prune_review_artifacts(
    review_dir: Path,
    *,
    stem: str,
    current_paths: list[Path],
    keep: int,
    ttl_hours: float,
) -> tuple[list[str], list[str]]:
    cutoff = now_utc() - dt.timedelta(hours=max(1.0, ttl_hours))
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


def classify_route(
    symbol: str,
    route: dict[str, Any],
    window_leg: dict[str, Any] | None = None,
    bnb_focus: dict[str, Any] | None = None,
) -> dict[str, Any]:
    action = str(route.get("action") or "").strip()
    deployment = str(route.get("deployment") or "").strip()
    lane = str(route.get("lane") or "").strip()
    reason = str(route.get("reason") or "").strip()

    if action == "deploy_price_state_only":
        status_label = "deploy_now"
        route_summary = "Use price-state trigger path now."
    elif action == "candidate_flow_secondary":
        status_label = "review"
        route_summary = "Manual review before promoting this beta flow-secondary path."
    elif action in {"watch_short_window_flow_priority", "watch_priority_until_long_window_confirms"}:
        status_label = "watch_priority"
        route_summary = "Keep in research watch; this is the first beta flow leg to revisit."
    elif action in {"watch_short_window_flow_only", "watch_only"}:
        status_label = "watch_only"
        route_summary = "Keep in research watch; do not promote this leg yet."
    else:
        status_label = "review"
        route_summary = "Review this route manually before promotion."

    payload = {
        "symbol": symbol,
        "lane": lane,
        "deployment": deployment,
        "action": action,
        "status_label": status_label,
        "reason": reason,
        "route_summary": route_summary,
    }
    if route.get("flow_combo"):
        payload["flow_combo"] = str(route.get("flow_combo"))
        payload["flow_combo_canonical"] = str(route.get("flow_combo_canonical") or "") or canonical_combo_id(route.get("flow_combo"))
    if route.get("flow_return") is not None:
        payload["flow_return"] = float(route.get("flow_return") or 0.0)
    if window_leg:
        payload["short_flow_combo"] = str(window_leg.get("short_flow_combo") or "") or None
        payload["short_flow_combo_canonical"] = (
            str(window_leg.get("short_flow_combo_canonical") or "")
            or canonical_combo_id(window_leg.get("short_flow_combo"))
            or None
        )
        payload["long_flow_combo"] = str(window_leg.get("long_flow_combo") or "") or None
        payload["long_flow_combo_canonical"] = (
            str(window_leg.get("long_flow_combo_canonical") or "")
            or canonical_combo_id(window_leg.get("long_flow_combo"))
            or None
        )
        payload["short_top_combo"] = str(window_leg.get("short_top_combo") or "") or None
        payload["short_top_combo_canonical"] = (
            str(window_leg.get("short_top_combo_canonical") or "")
            or canonical_combo_id(window_leg.get("short_top_combo"))
            or None
        )
        payload["long_top_combo"] = str(window_leg.get("long_top_combo") or "") or None
        payload["long_top_combo_canonical"] = (
            str(window_leg.get("long_top_combo_canonical") or "")
            or canonical_combo_id(window_leg.get("long_top_combo"))
            or None
        )
        payload["window_action"] = str(window_leg.get("action") or "")
        payload["window_action_reason"] = str(window_leg.get("action_reason") or "")
        payload["flow_window_verdict"] = str(window_leg.get("flow_window_verdict") or "")
        payload["price_state_window_verdict"] = str(window_leg.get("price_state_window_verdict") or "")
        if window_leg.get("flow_window_floor"):
            payload["flow_window_floor"] = str(window_leg.get("flow_window_floor") or "")
        if window_leg.get("price_state_window_floor"):
            payload["price_state_window_floor"] = str(window_leg.get("price_state_window_floor") or "")
        if window_leg.get("comparative_window_takeaway"):
            payload["comparative_window_takeaway"] = str(window_leg.get("comparative_window_takeaway") or "")
        if payload["window_action_reason"]:
            payload["reason"] = payload["window_action_reason"]
    if bnb_focus and symbol == "BNBUSDT":
        payload["promotion_gate"] = str(bnb_focus.get("promotion_gate") or "")
        payload["promotion_gate_reason"] = str(bnb_focus.get("promotion_gate_reason") or "")
        payload["focus_short_flow_combo"] = (
            str(bnb_focus.get("short_flow_combo") or "") or str(payload.get("flow_combo") or "") or None
        )
        payload["focus_short_flow_combo_canonical"] = (
            str(bnb_focus.get("short_flow_combo_canonical") or "")
            or str(payload.get("flow_combo_canonical") or "")
            or None
        )
        payload["focus_long_flow_combo"] = str(bnb_focus.get("long_flow_combo") or "") or None
        payload["focus_long_flow_combo_canonical"] = str(bnb_focus.get("long_flow_combo_canonical") or "") or None
        payload["focus_long_top_combo"] = str(bnb_focus.get("long_top_combo") or "") or None
        payload["focus_long_top_combo_canonical"] = str(bnb_focus.get("long_top_combo_canonical") or "") or None
        payload["focus_window_verdict"] = str(bnb_focus.get("flow_window_verdict") or "")
        payload["focus_window_floor"] = str(bnb_focus.get("flow_window_floor") or "")
        payload["price_state_window_floor"] = str(bnb_focus.get("price_state_window_floor") or "")
        payload["comparative_window_takeaway"] = str(bnb_focus.get("comparative_window_takeaway") or "")
        payload["xlong_flow_window_floor"] = str(bnb_focus.get("xlong_flow_window_floor") or "")
        payload["xlong_comparative_window_takeaway"] = str(
            bnb_focus.get("xlong_comparative_window_takeaway") or ""
        )
        payload["focus_brief"] = str(bnb_focus.get("brief") or "")
        if payload["promotion_gate_reason"]:
            payload["reason"] = payload["promotion_gate_reason"]
    return payload


def classify_handoff(
    source_payload: dict[str, Any],
    beta_leg_window_payload: dict[str, Any] | None = None,
    bnb_focus_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    symbol_routes = source_payload.get("symbol_routes") or {}
    ordered_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
    window_legs = dict((beta_leg_window_payload or {}).get("legs") or {})
    routes = [
        classify_route(
            symbol,
            dict(symbol_routes.get(symbol) or {}),
            dict(window_legs.get(symbol) or {}) or None,
            dict(bnb_focus_payload or {}) or None,
        )
        for symbol in ordered_symbols
        if symbol_routes.get(symbol)
    ]

    deploy_now_symbols = [row["symbol"] for row in routes if row["status_label"] == "deploy_now"]
    watch_priority_symbols = [row["symbol"] for row in routes if row["status_label"] == "watch_priority"]
    watch_only_symbols = [row["symbol"] for row in routes if row["status_label"] == "watch_only"]
    review_symbols = [row["symbol"] for row in routes if row["status_label"] == "review"]

    next_focus_symbol = (
        review_symbols[0]
        if review_symbols
        else (watch_priority_symbols[0] if watch_priority_symbols else (watch_only_symbols[0] if watch_only_symbols else None))
    )
    next_focus_action = None
    next_focus_reason = None
    for row in routes:
        if row["symbol"] == next_focus_symbol:
            next_focus_action = row["action"]
            next_focus_reason = row["reason"]
            break

    operator_status = "deploy-price-state-plus-beta-watch"
    if not deploy_now_symbols:
        operator_status = "watch-all"
    elif review_symbols:
        operator_status = "deploy-price-state-plus-beta-review"
    elif not watch_priority_symbols and not watch_only_symbols:
        operator_status = "deploy-only"

    route_stack_brief_parts: list[str] = []
    if deploy_now_symbols:
        route_stack_brief_parts.append("deploy:" + ",".join(deploy_now_symbols))
    if review_symbols:
        route_stack_brief_parts.append("review:" + ",".join(review_symbols))
    if watch_priority_symbols:
        route_stack_brief_parts.append("watch-priority:" + ",".join(watch_priority_symbols))
    if watch_only_symbols:
        route_stack_brief_parts.append("watch:" + ",".join(watch_only_symbols))

    route_stack_brief = " | ".join(route_stack_brief_parts)
    overall_takeaway = "Deploy BTC/ETH on price-state only."
    if review_symbols and "BNBUSDT" in review_symbols:
        overall_takeaway += " Review BNB first as the leading beta flow-secondary candidate, and keep SOL behind it until longer-window stability improves."
    elif watch_priority_symbols and "BNBUSDT" in watch_priority_symbols:
        overall_takeaway += " Keep BNB as the first beta flow watch leg, and keep SOL behind BNB until longer-window flow stability improves."
    elif watch_only_symbols:
        overall_takeaway += " Keep beta names in watch mode until longer-window flow stability improves."

    payload = {
        "routes": routes,
        "deploy_now_symbols": deploy_now_symbols,
        "watch_priority_symbols": watch_priority_symbols,
        "watch_only_symbols": watch_only_symbols,
        "review_symbols": review_symbols,
        "next_focus_symbol": next_focus_symbol,
        "next_focus_action": next_focus_action,
        "next_focus_reason": next_focus_reason,
        "operator_status": operator_status,
        "route_stack_brief": route_stack_brief,
        "overall_takeaway": overall_takeaway,
        "beta_leg_window_report_artifact": str((beta_leg_window_payload or {}).get("artifact") or "") or None,
    }
    if bnb_focus_payload:
        bnb_route = next((row for row in routes if row.get("symbol") == "BNBUSDT"), {})
        payload["bnb_flow_focus_artifact"] = str(bnb_focus_payload.get("artifact") or "") or None
        payload["focus_window_gate"] = str(bnb_focus_payload.get("promotion_gate") or "") or ""
        payload["focus_window_gate_reason"] = str(bnb_focus_payload.get("promotion_gate_reason") or "") or ""
        payload["focus_short_flow_combo"] = (
            str(bnb_focus_payload.get("short_flow_combo") or "") or str(bnb_route.get("flow_combo") or "") or None
        )
        payload["focus_short_flow_combo_canonical"] = (
            str(bnb_focus_payload.get("short_flow_combo_canonical") or "")
            or str(bnb_route.get("flow_combo_canonical") or "")
            or None
        )
        payload["focus_long_flow_combo"] = str(bnb_focus_payload.get("long_flow_combo") or "") or None
        payload["focus_long_flow_combo_canonical"] = str(bnb_focus_payload.get("long_flow_combo_canonical") or "") or None
        payload["focus_long_top_combo"] = str(bnb_focus_payload.get("long_top_combo") or "") or None
        payload["focus_long_top_combo_canonical"] = str(bnb_focus_payload.get("long_top_combo_canonical") or "") or None
        payload["focus_window_verdict"] = str(bnb_focus_payload.get("flow_window_verdict") or "") or ""
        payload["focus_window_floor"] = str(bnb_focus_payload.get("flow_window_floor") or "") or ""
        payload["price_state_window_floor"] = str(bnb_focus_payload.get("price_state_window_floor") or "") or ""
        payload["comparative_window_takeaway"] = str(bnb_focus_payload.get("comparative_window_takeaway") or "") or ""
        payload["xlong_flow_window_floor"] = str(bnb_focus_payload.get("xlong_flow_window_floor") or "") or ""
        payload["xlong_comparative_window_takeaway"] = (
            str(bnb_focus_payload.get("xlong_comparative_window_takeaway") or "") or ""
        )
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Binance Indicator Symbol Route Handoff",
        "",
        f"- as_of: `{payload.get('as_of') or ''}`",
        f"- source_artifact: `{payload.get('source_artifact') or ''}`",
        f"- operator_status: `{payload.get('operator_status') or ''}`",
        f"- route_stack: `{payload.get('route_stack_brief') or ''}`",
        f"- next_focus_symbol: `{payload.get('next_focus_symbol') or '-'}`",
        f"- next_focus_action: `{payload.get('next_focus_action') or '-'}`",
        "",
        "## Routes",
    ]
    for row in payload.get("routes", []):
        lines.append(
            f"- `{row.get('symbol')}` lane=`{row.get('lane')}` deployment=`{row.get('deployment')}` action=`{row.get('action')}` status=`{row.get('status_label')}`"
        )
        lines.append(f"  - reason: {row.get('reason')}")
        lines.append(f"  - summary: {row.get('route_summary')}")
        if row.get("flow_combo"):
            lines.append(f"  - flow_combo: `{row.get('flow_combo')}`")
        if row.get("flow_combo_canonical"):
            lines.append(f"  - flow_combo_canonical: `{row.get('flow_combo_canonical')}`")
        if row.get("flow_return") is not None:
            lines.append(f"  - flow_return: `{float(row.get('flow_return') or 0.0):.4f}`")
        if row.get("short_flow_combo_canonical"):
            lines.append(
                f"  - short_flow_window: `{row.get('short_flow_combo') or '-'}` canonical=`{row.get('short_flow_combo_canonical')}`"
            )
        if row.get("long_flow_combo_canonical"):
            lines.append(
                f"  - long_flow_window: `{row.get('long_flow_combo') or '-'}` canonical=`{row.get('long_flow_combo_canonical')}`"
            )
        if row.get("short_top_combo_canonical"):
            lines.append(
                f"  - short_top_window: `{row.get('short_top_combo') or '-'}` canonical=`{row.get('short_top_combo_canonical')}`"
            )
        if row.get("long_top_combo_canonical"):
            lines.append(
                f"  - long_top_window: `{row.get('long_top_combo') or '-'}` canonical=`{row.get('long_top_combo_canonical')}`"
            )
        if row.get("promotion_gate"):
            lines.append(f"  - promotion_gate: `{row.get('promotion_gate')}`")
        if row.get("focus_short_flow_combo_canonical"):
            lines.append(f"  - focus_short_flow_combo_canonical: `{row.get('focus_short_flow_combo_canonical')}`")
        if row.get("focus_long_flow_combo_canonical"):
            lines.append(f"  - focus_long_flow_combo_canonical: `{row.get('focus_long_flow_combo_canonical')}`")
        if row.get("focus_long_top_combo_canonical"):
            lines.append(f"  - focus_long_top_combo_canonical: `{row.get('focus_long_top_combo_canonical')}`")
        if row.get("focus_window_verdict"):
            lines.append(f"  - focus_window_verdict: `{row.get('focus_window_verdict')}`")
        if row.get("focus_window_floor"):
            lines.append(f"  - focus_window_floor: `{row.get('focus_window_floor')}`")
        if row.get("price_state_window_floor"):
            lines.append(f"  - price_state_window_floor: `{row.get('price_state_window_floor')}`")
        if row.get("comparative_window_takeaway"):
            lines.append(f"  - comparative_window_takeaway: {row.get('comparative_window_takeaway')}")
        if row.get("xlong_flow_window_floor"):
            lines.append(f"  - xlong_flow_window_floor: `{row.get('xlong_flow_window_floor')}`")
        if row.get("xlong_comparative_window_takeaway"):
            lines.append(f"  - xlong_comparative_window_takeaway: {row.get('xlong_comparative_window_takeaway')}")

    lines.extend(
        [
            "",
            "## Buckets",
            f"- deploy_now: `{', '.join(payload.get('deploy_now_symbols', [])) or '-'}`",
            f"- watch_priority: `{', '.join(payload.get('watch_priority_symbols', [])) or '-'}`",
            f"- watch_only: `{', '.join(payload.get('watch_only_symbols', [])) or '-'}`",
            f"- review: `{', '.join(payload.get('review_symbols', [])) or '-'}`",
            "",
            "## Overall Takeaway",
            f"- {payload.get('overall_takeaway') or ''}",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a symbol-level handoff from the latest Binance indicator combo playbook artifact.")
    parser.add_argument("--review-dir", required=True)
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--artifact-keep", type=int, default=12)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    runtime_now = now_utc()

    source_path = latest_combo_playbook(review_dir)
    source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    beta_leg_window_path, beta_leg_window_payload = latest_beta_leg_window_report(review_dir)
    bnb_focus_path, bnb_focus_payload = latest_bnb_flow_focus(review_dir)
    handoff = classify_handoff(source_payload, beta_leg_window_payload, bnb_focus_payload)

    stamp = runtime_now.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = review_dir / f"{stamp}_binance_indicator_symbol_route_handoff.json"
    md_path = review_dir / f"{stamp}_binance_indicator_symbol_route_handoff.md"
    checksum_path = review_dir / f"{stamp}_binance_indicator_symbol_route_handoff_checksum.json"

    payload = {
        "ok": True,
        "status": "ok",
        "as_of": fmt_utc(runtime_now),
        "source_artifact": str(source_path),
        "beta_leg_window_report_artifact": str(beta_leg_window_path) if beta_leg_window_path else None,
        "bnb_flow_focus_artifact": str(bnb_focus_path) if bnb_focus_path else None,
        **handoff,
        "artifact_label": "binance-indicator-symbol-route-handoff:ok",
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")

    checksum_payload = {
        "artifact": str(json_path),
        "artifact_sha256": sha256_file(json_path),
        "markdown": str(md_path),
        "markdown_sha256": sha256_file(md_path),
        "generated_at": fmt_utc(runtime_now),
    }
    checksum_path.write_text(json.dumps(checksum_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    pruned_keep, pruned_age = prune_review_artifacts(
        review_dir,
        stem="binance_indicator_symbol_route_handoff",
        current_paths=[json_path, md_path, checksum_path],
        keep=max(3, int(args.artifact_keep)),
        ttl_hours=max(24.0, float(args.artifact_ttl_hours)),
    )

    payload["artifact"] = str(json_path)
    payload["markdown"] = str(md_path)
    payload["checksum"] = str(checksum_path)
    payload["pruned_keep"] = pruned_keep
    payload["pruned_age"] = pruned_age
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checksum_payload["artifact_sha256"] = sha256_file(json_path)
    checksum_path.write_text(json.dumps(checksum_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
