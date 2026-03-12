#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any

COMBO_CANONICAL_ALIASES = {
    "ad_breakout": "cvd_breakout",
    "ad_rsi_breakout": "cvd_rsi_breakout",
    "ad_rsi_vol_breakout": "cvd_rsi_vol_breakout",
    "ad_rsi_reclaim": "cvd_rsi_reclaim",
    "taker_oi_ad_breakout": "taker_oi_cvd_breakout",
    "taker_oi_ad_rsi_breakout": "taker_oi_cvd_rsi_breakout",
}


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


def canonical_combo_id(combo_id: Any) -> str:
    combo = str(combo_id or "")
    return COMBO_CANONICAL_ALIASES.get(combo, combo)


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


TIMESTAMP_RE = re.compile(r"^(?P<stamp>\d{8}T\d{6}Z)_")
FUTURE_STAMP_GRACE_MINUTES = 5


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


def latest_artifact_by_suffix(review_dir: Path, suffix: str) -> Path | None:
    candidates = list(review_dir.glob(f"*_{suffix}.json"))
    if not candidates:
        return None
    return max(candidates, key=artifact_sort_key)


def latest_backtest_artifact(review_dir: Path) -> Path:
    path = latest_artifact_by_suffix(review_dir, "binance_indicator_combo_etf")
    if path is None:
        raise FileNotFoundError("no_binance_indicator_combo_etf_artifact")
    return path


def latest_native_lane_playbook(review_dir: Path) -> tuple[Path | None, dict[str, Any] | None]:
    path = latest_artifact_by_suffix(review_dir, "binance_indicator_native_lane_playbook")
    if path is None:
        return None, None
    return path, json.loads(path.read_text(encoding="utf-8"))


def latest_beta_leg_window_report(review_dir: Path) -> tuple[Path | None, dict[str, Any] | None]:
    path = latest_artifact_by_suffix(review_dir, "binance_indicator_native_beta_leg_window_report")
    if path is None:
        return None, None
    return path, json.loads(path.read_text(encoding="utf-8"))


def classify_playbook(
    source_payload: dict[str, Any],
    native_lane_payload: dict[str, Any] | None = None,
    beta_leg_window_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    crypto_ranked = list(source_payload.get("crypto_family", {}).get("ranked_combos", []))
    commodity_ranked = list(source_payload.get("commodity_family", {}).get("ranked_combos", []))
    crypto_discarded = list(source_payload.get("crypto_family", {}).get("discarded_combos", []))
    commodity_discarded = list(source_payload.get("commodity_family", {}).get("discarded_combos", []))

    adopt_now: list[dict[str, Any]] = []
    research_only: list[dict[str, Any]] = []
    context_only: list[dict[str, Any]] = []
    discard_now: list[dict[str, Any]] = []
    retest_native_crypto: list[dict[str, Any]] = []

    if crypto_ranked:
        top = crypto_ranked[0]
        adopt_now.append(
            {
                "family": "crypto",
                "combo_id": top.get("combo_id"),
                "combo_id_canonical": canonical_combo_id(top.get("combo_id")),
                "reason": "Best ETF-proxy timing result across crypto proxies; treat as price-state trigger candidate.",
                "return": float(top.get("avg_total_return") or 0.0),
                "timely_hit_rate": float(top.get("avg_timely_hit_rate") or 0.0),
            }
        )

    if commodity_ranked:
        top = commodity_ranked[0]
        research_only.append(
            {
                "family": "commodity",
                "combo_id": top.get("combo_id"),
                "combo_id_canonical": canonical_combo_id(top.get("combo_id")),
                "reason": "Best relative commodity ETF combo, but aggregate return stayed non-positive in this sample.",
                "return": float(top.get("avg_total_return") or 0.0),
                "timely_hit_rate": float(top.get("avg_timely_hit_rate") or 0.0),
            }
        )

    for row in crypto_discarded:
        combo_id = str(row.get("combo_id") or "")
        base = {
            "family": "crypto",
            "combo_id": combo_id,
            "combo_id_canonical": canonical_combo_id(combo_id),
            "reason": str(row.get("discard_reason") or ""),
            "trade_count": int(row.get("trade_count") or 0),
            "timely_hit_rate": float(row.get("avg_timely_hit_rate") or 0.0),
            "avg_lag_bars": row.get("avg_lag_bars"),
        }
        if combo_id.startswith("taker_oi") or combo_id.startswith("crowding_filtered"):
            context_only.append(
                {
                    **base,
                    "usage": "context_or_veto_only",
                    "note": "Sparse or delayed on ETF proxies; keep as crowding/force filter, not trigger.",
                }
            )
            retest_native_crypto.append(
                {
                    **base,
                    "retest_scope": "native_24_7_crypto_only",
                    "note": "Re-test on exchange-native bars before rejecting the factor family.",
                }
            )
        else:
            discard_now.append(
                {
                    **base,
                    "usage": "discard_for_etf_timing",
                    "note": "Failed timely support/resistance test in this sample.",
                }
            )

    for row in commodity_discarded:
        discard_now.append(
            {
                "family": "commodity",
                "combo_id": str(row.get("combo_id") or ""),
                "combo_id_canonical": canonical_combo_id(row.get("combo_id")),
                "reason": str(row.get("discard_reason") or ""),
                "trade_count": int(row.get("trade_count") or 0),
                "timely_hit_rate": float(row.get("avg_timely_hit_rate") or 0.0),
                "avg_lag_bars": row.get("avg_lag_bars"),
                "usage": "discard_for_etf_timing",
                "note": "Too laggy for ETF support/resistance timing in this sample.",
            }
        )

    measured_vs_practitioner = {
        "crypto": {
            "measured": source_payload.get("crypto_takeaway"),
            "practitioner": source_payload.get("crypto_practitioner_note"),
        },
        "commodity": {
            "measured": source_payload.get("commodity_takeaway"),
            "practitioner": source_payload.get("commodity_practitioner_note"),
        },
    }

    native_crypto_lanes = {}
    symbol_routes: dict[str, Any] = {}
    beta_window_legs = dict((beta_leg_window_payload or {}).get("legs") or {})
    beta_window_focus_symbol = str((beta_leg_window_payload or {}).get("next_focus_symbol") or "")
    beta_window_focus_action = str((beta_leg_window_payload or {}).get("next_focus_action") or "")
    beta_window_focus_reason = str((beta_leg_window_payload or {}).get("next_focus_reason") or "")

    if native_lane_payload:
        native_crypto_lanes = {
            "majors": {
                "deployment_status": native_lane_payload.get("lanes", {}).get("majors", {}).get("deployment_status"),
                "stability_status": native_lane_payload.get("lanes", {}).get("majors", {}).get("stability_status"),
                "lane_takeaway": native_lane_payload.get("lanes", {}).get("majors", {}).get("lane_takeaway"),
            },
            "beta": {
                "deployment_status": native_lane_payload.get("lanes", {}).get("beta", {}).get("deployment_status"),
                "stability_status": native_lane_payload.get("lanes", {}).get("beta", {}).get("stability_status"),
                "lane_takeaway": native_lane_payload.get("lanes", {}).get("beta", {}).get("lane_takeaway"),
                "leg_constraint": native_lane_payload.get("lanes", {}).get("beta", {}).get("leg_constraint"),
                "leg_constraint_takeaway": native_lane_payload.get("lanes", {}).get("beta", {}).get("leg_constraint_takeaway"),
                "beta_leg_routes": native_lane_payload.get("lanes", {}).get("beta", {}).get("beta_leg_routes"),
                "beta_leg_focus": native_lane_payload.get("lanes", {}).get("beta", {}).get("beta_leg_focus"),
                "beta_leg_window_focus": {
                    "symbol": beta_window_focus_symbol,
                    "action": beta_window_focus_action,
                    "reason": beta_window_focus_reason,
                }
                if beta_window_focus_symbol
                else None,
            },
            "recommended_live_research_split": native_lane_payload.get("recommended_live_research_split", {}),
            "overall_takeaway": native_lane_payload.get("overall_takeaway"),
        }

        symbol_routes["BTCUSDT"] = {
            "lane": "majors",
            "deployment": "price_state_primary_only",
            "action": "deploy_price_state_only",
            "reason": "Majors lane remains stable on price-state while flow stays non-positive.",
        }
        symbol_routes["ETHUSDT"] = {
            "lane": "majors",
            "deployment": "price_state_primary_only",
            "action": "deploy_price_state_only",
            "reason": "Majors lane remains stable on price-state while flow stays non-positive.",
        }
        beta_routes = native_crypto_lanes.get("beta", {}).get("beta_leg_routes") or {}
        beta_focus = native_crypto_lanes.get("beta", {}).get("beta_leg_focus") or {}
        for symbol, leg_key in (("SOLUSDT", "sol"), ("BNBUSDT", "bnb")):
            leg_route = dict(beta_routes.get(leg_key) or {})
            window_leg = dict(beta_window_legs.get(symbol) or {})
            action = "watch_short_window_flow_only"
            if str(beta_focus.get("symbol") or "") == symbol.replace("USDT", ""):
                action = "watch_short_window_flow_priority"
            if window_leg.get("action"):
                action = str(window_leg.get("action"))
            symbol_routes[symbol] = {
                "lane": "beta",
                "deployment": native_crypto_lanes.get("beta", {}).get("deployment_status"),
                "action": action,
                "reason": window_leg.get("action_reason")
                or leg_route.get("stability_takeaway")
                or native_crypto_lanes.get("beta", {}).get("lane_takeaway"),
                "flow_combo": leg_route.get("short_flow_combo"),
                "flow_combo_canonical": leg_route.get("short_flow_combo_canonical"),
                "flow_return": leg_route.get("short_flow_return"),
                "flow_window_verdict": window_leg.get("flow_window_verdict"),
                "price_state_window_verdict": window_leg.get("price_state_window_verdict"),
                "beta_window_action": window_leg.get("action"),
                "beta_window_action_reason": window_leg.get("action_reason"),
            }

    overall_takeaway = (
        "Use RSI plus fixed-range profile structure for ETF proxy timing, use CVD-lite as the active shortline confirmation layer, keep Binance taker/OI as context-veto on ETF proxies, "
        "and re-test flow-heavy stacks on native 24/7 crypto bars before promoting them."
    )
    if native_crypto_lanes:
        beta_status = native_crypto_lanes.get("beta", {}).get("deployment_status")
        if beta_status == "price_state_plus_flow_secondary_basket_only":
            overall_takeaway = (
                "Use RSI plus fixed-range profile structure for ETF proxy timing, use CVD-lite as the active shortline confirmation layer, keep Binance taker/OI as context-veto on ETF proxies, "
                "and on native 24/7 crypto bars keep majors on price-state while promoting taker/OI only as a beta-basket secondary confirmation family, "
                "not as a standalone SOL or BNB lane."
            )
        elif beta_status == "flow_secondary_research_hold":
            overall_takeaway = (
                "Use RSI plus fixed-range profile structure for ETF proxy timing, use CVD-lite as the active shortline confirmation layer, keep Binance taker/OI as context-veto on ETF proxies, "
                "and on native 24/7 crypto bars keep majors on price-state while treating beta taker/OI as a fragile research-hold secondary layer until longer-window stability improves."
            )
        else:
            overall_takeaway = (
                "Use RSI plus fixed-range profile structure for ETF proxy timing, use CVD-lite as the active shortline confirmation layer, keep Binance taker/OI as context-veto on ETF proxies, "
                "and on native 24/7 crypto bars keep majors on price-state while promoting taker/OI only as a beta-only secondary confirmation family."
            )

    return {
        "adopt_now": adopt_now,
        "research_only": research_only,
        "context_only": context_only,
        "discard_now": discard_now,
        "retest_native_crypto": retest_native_crypto,
        "measured_vs_practitioner": measured_vs_practitioner,
        "native_crypto_lanes": native_crypto_lanes,
        "beta_leg_window_takeaway": (beta_leg_window_payload or {}).get("overall_takeaway"),
        "symbol_routes": symbol_routes,
        "overall_takeaway": overall_takeaway,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Binance Indicator Combo Playbook",
        "",
        f"- as_of: `{payload.get('as_of') or ''}`",
        f"- source_artifact: `{payload.get('source_artifact') or ''}`",
        "",
        "## Adopt Now",
    ]
    for row in payload.get("adopt_now", []):
        lines.append(
            f"- `{row.get('family')}:{row.get('combo_id')}` canonical=`{row.get('combo_id_canonical')}` return=`{row.get('return'):.4f}` timely=`{row.get('timely_hit_rate'):.2%}`"
        )
        lines.append(f"  - reason: {row.get('reason')}")

    lines.extend(["", "## Research Only"])
    for row in payload.get("research_only", []):
        lines.append(
            f"- `{row.get('family')}:{row.get('combo_id')}` canonical=`{row.get('combo_id_canonical')}` return=`{row.get('return'):.4f}` timely=`{row.get('timely_hit_rate'):.2%}`"
        )
        lines.append(f"  - reason: {row.get('reason')}")

    lines.extend(["", "## Context Or Veto Only"])
    for row in payload.get("context_only", []):
        lines.append(
            f"- `{row.get('family')}:{row.get('combo_id')}` canonical=`{row.get('combo_id_canonical')}` trades=`{row.get('trade_count')}` timely=`{row.get('timely_hit_rate'):.2%}` lag=`{row.get('avg_lag_bars')}`"
        )
        lines.append(f"  - note: {row.get('note')}")

    lines.extend(["", "## Discard Now"])
    for row in payload.get("discard_now", []):
        lines.append(
            f"- `{row.get('family')}:{row.get('combo_id')}` canonical=`{row.get('combo_id_canonical')}` reason=`{row.get('reason')}` timely=`{row.get('timely_hit_rate'):.2%}` lag=`{row.get('avg_lag_bars')}`"
        )

    lines.extend(["", "## Re-test On Native Crypto"])
    for row in payload.get("retest_native_crypto", []):
        lines.append(
            f"- `{row.get('combo_id')}` canonical=`{row.get('combo_id_canonical')}` trades=`{row.get('trade_count')}` timely=`{row.get('timely_hit_rate'):.2%}`"
        )
        lines.append(f"  - note: {row.get('note')}")

    lines.extend(["", "## Measured Vs Practitioner"])
    for family_key in ("crypto", "commodity"):
        row = payload.get("measured_vs_practitioner", {}).get(family_key, {})
        lines.append(f"- `{family_key}` measured: {row.get('measured')}")
        lines.append(f"  - practitioner: {row.get('practitioner')}")

    native_lanes = payload.get("native_crypto_lanes") or {}
    if native_lanes:
        lines.extend(["", "## Native Crypto Lanes"])
        for lane_key in ("majors", "beta"):
            lane = native_lanes.get(lane_key, {})
            lines.append(
                f"- `{lane_key}` deployment=`{lane.get('deployment_status')}` stability=`{lane.get('stability_status')}`"
            )
            lines.append(f"  - takeaway: {lane.get('lane_takeaway')}")
            if lane_key == "beta" and lane.get("beta_leg_focus"):
                focus = lane.get("beta_leg_focus") or {}
                lines.append(
                    f"  - beta-leg-focus: `{focus.get('symbol')}` use=`{focus.get('recommended_use')}` short_flow_return=`{float(focus.get('short_flow_return') or 0.0):.4f}`"
                )
            if lane_key == "beta" and lane.get("beta_leg_window_focus"):
                focus = lane.get("beta_leg_window_focus") or {}
                lines.append(
                    f"  - beta-window-focus: `{focus.get('symbol')}` action=`{focus.get('action')}`"
                )
                lines.append(f"    - reason: {focus.get('reason')}")
        lines.append(
            f"- split: `{native_lanes.get('recommended_live_research_split')}`"
        )
        if payload.get("beta_leg_window_takeaway"):
            lines.append(f"- beta-window-takeaway: {payload.get('beta_leg_window_takeaway')}")

    symbol_routes = payload.get("symbol_routes") or {}
    if symbol_routes:
        lines.extend(["", "## Symbol Routes"])
        for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"):
            route = symbol_routes.get(symbol, {})
            if not route:
                continue
            lines.append(
                f"- `{symbol}` lane=`{route.get('lane')}` deployment=`{route.get('deployment')}` action=`{route.get('action')}`"
            )
            lines.append(f"  - reason: {route.get('reason')}")
            if route.get("flow_combo_canonical"):
                lines.append(f"  - flow_combo_canonical: `{route.get('flow_combo_canonical')}`")
            if route.get("flow_window_verdict"):
                lines.append(
                    f"  - window: flow=`{route.get('flow_window_verdict')}` price-state=`{route.get('price_state_window_verdict')}`"
                )

    lines.extend(["", "## Overall Takeaway", f"- {payload.get('overall_takeaway')}"])
    return "\n".join(lines).strip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Derive a measured-vs-practitioner playbook from Binance indicator ETF backtest artifacts.")
    parser.add_argument("--review-dir", required=True)
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--artifact-keep", type=int, default=12)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    runtime_now = now_utc()

    source_path = latest_backtest_artifact(review_dir)
    source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    native_lane_path, native_lane_payload = latest_native_lane_playbook(review_dir)
    beta_leg_window_path, beta_leg_window_payload = latest_beta_leg_window_report(review_dir)
    playbook = classify_playbook(source_payload, native_lane_payload, beta_leg_window_payload)

    stamp = runtime_now.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = review_dir / f"{stamp}_binance_indicator_combo_playbook.json"
    md_path = review_dir / f"{stamp}_binance_indicator_combo_playbook.md"
    checksum_path = review_dir / f"{stamp}_binance_indicator_combo_playbook_checksum.json"

    payload = {
        "ok": True,
        "status": "ok",
        "as_of": fmt_utc(runtime_now),
        "source_artifact": str(source_path),
        "native_lane_artifact": str(native_lane_path) if native_lane_path else None,
        "beta_leg_window_artifact": str(beta_leg_window_path) if beta_leg_window_path else None,
        **playbook,
        "artifact_label": "binance-indicator-combo-playbook:ok",
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
        stem="binance_indicator_combo_playbook",
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
