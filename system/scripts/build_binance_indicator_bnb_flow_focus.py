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
FLOW_PREFIXES = ("taker_oi", "crowding_filtered")
PRICE_STATE_IDS = ("rsi_breakout", "cvd_breakout", "cvd_rsi_breakout")
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


def artifact_status_rank(payload: dict[str, Any]) -> int:
    status = str(payload.get("status") or "").strip().lower()
    if status == "ok" or payload.get("ok") is True:
        return 2
    if status:
        return 0
    return 1


def is_price_state_combo(combo_id: Any) -> bool:
    return canonical_combo_id(combo_id) in PRICE_STATE_IDS


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


def fmt_utc(value: dt.datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(dt.timezone.utc).isoformat()


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latest_beta_leg_window_report(review_dir: Path, reference_now: dt.datetime | None = None) -> Path:
    candidates = list(review_dir.glob("*_binance_indicator_native_beta_leg_window_report.json"))
    if not candidates:
        raise FileNotFoundError("no_binance_indicator_native_beta_leg_window_report_artifact")
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def latest_bnb_native_artifact(
    review_dir: Path,
    symbol_group: str,
    reference_now: dt.datetime | None = None,
) -> Path | None:
    candidates = list(review_dir.glob(f"*_{symbol_group}_binance_indicator_combo_native_crypto.json"))
    if not candidates:
        return None
    ranked: list[tuple[int, tuple[int, str, float, str], Path]] = []
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            rank = 1
        else:
            rank = artifact_status_rank(payload)
        ranked.append((rank, artifact_sort_key(path, reference_now), path))
    ranked.sort(reverse=True)
    return ranked[0][2]


def _rank_combo_row(row: dict[str, Any]) -> tuple[float, float, int]:
    return (
        float(row.get("avg_total_return") or 0.0),
        float(row.get("avg_timely_hit_rate") or 0.0),
        int(row.get("trade_count") or 0),
    )


def extract_native_family_best(
    payload: dict[str, Any],
    *,
    family: str,
) -> dict[str, Any]:
    native_family = dict(payload.get("native_crypto_family") or {})
    ranked = list(native_family.get("ranked_combos") or [])
    discarded = list(native_family.get("discarded_combos") or [])
    if family == "flow":
        wanted = lambda combo_id: str(combo_id or "").startswith(FLOW_PREFIXES)
    else:
        wanted = is_price_state_combo

    ranked_matches = [row for row in ranked if wanted(row.get("combo_id"))]
    discarded_matches = [row for row in discarded if wanted(row.get("combo_id"))]

    if ranked_matches:
        best = max(ranked_matches, key=_rank_combo_row)
        return {
            "combo": str(best.get("combo_id") or ""),
            "combo_canonical": canonical_combo_id(best.get("combo_id")),
            "source": "ranked",
            "return": float(best.get("avg_total_return") or 0.0),
            "timely_hit_rate": float(best.get("avg_timely_hit_rate") or 0.0),
            "trade_count": int(best.get("trade_count") or 0),
        }
    if discarded_matches:
        best = max(discarded_matches, key=_rank_combo_row)
        return {
            "combo": str(best.get("combo_id") or ""),
            "combo_canonical": canonical_combo_id(best.get("combo_id")),
            "source": "discarded",
            "return": float(best.get("avg_total_return") or 0.0),
            "timely_hit_rate": float(best.get("avg_timely_hit_rate") or 0.0),
            "trade_count": int(best.get("trade_count") or 0),
        }
    return {
        "combo": "",
        "combo_canonical": "",
        "source": "missing",
        "return": 0.0,
        "timely_hit_rate": 0.0,
        "trade_count": 0,
    }


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


def derive_focus_payload_from_beta_window(source_payload: dict[str, Any]) -> dict[str, Any]:
    leg = dict((source_payload.get("legs") or {}).get("BNBUSDT") or {})
    flow_verdict = str(leg.get("flow_window_verdict") or "")
    price_verdict = str(leg.get("price_state_window_verdict") or "")
    action = str(leg.get("action") or "")
    action_reason = str(leg.get("action_reason") or "")

    if flow_verdict == "stable_across_windows" and price_verdict == "stable_across_windows":
        promotion_gate = "candidate_flow_secondary"
        promotion_gate_reason = "BNB keeps both flow and price-state stable when the window is extended."
        readiness = "promotion_candidate"
    else:
        promotion_gate = "blocked_until_long_window_confirms"
        promotion_gate_reason = action_reason or "BNB still needs longer-window confirmation before promotion."
        readiness = "watch_priority"

    short_flow_return = float(leg.get("short_flow_return") or 0.0)
    long_flow_return = float(leg.get("long_flow_return") or 0.0)
    short_flow_timely = float(leg.get("short_flow_timely_hit_rate") or 0.0)
    long_flow_timely = float(leg.get("long_flow_timely_hit_rate") or 0.0)

    return {
        "symbol": "BNBUSDT",
        "source_mode": "beta_window_fallback",
        "operator_status": readiness,
        "promotion_gate": promotion_gate,
        "promotion_gate_reason": promotion_gate_reason,
        "action": action,
        "action_reason": action_reason,
        "flow_window_verdict": flow_verdict,
        "price_state_window_verdict": price_verdict,
        "short_flow_combo": leg.get("short_flow_combo"),
        "short_flow_combo_canonical": canonical_combo_id(leg.get("short_flow_combo")),
        "short_flow_return": short_flow_return,
        "short_flow_timely_hit_rate": short_flow_timely,
        "long_flow_combo": leg.get("long_flow_combo"),
        "long_flow_combo_canonical": canonical_combo_id(leg.get("long_flow_combo")),
        "long_flow_return": long_flow_return,
        "long_flow_timely_hit_rate": long_flow_timely,
        "short_top_combo": leg.get("short_top_combo"),
        "short_top_combo_canonical": canonical_combo_id(leg.get("short_top_combo")),
        "short_top_return": float(leg.get("short_top_return") or 0.0),
        "long_top_combo": leg.get("long_top_combo"),
        "long_top_combo_canonical": canonical_combo_id(leg.get("long_top_combo")),
        "long_top_return": float(leg.get("long_top_return") or 0.0),
        "flow_return_delta": float(leg.get("flow_return_delta") or 0.0),
        "flow_timely_hit_rate_delta": float(leg.get("flow_timely_hit_rate_delta") or 0.0),
        "next_retest_action": "rerun_bnb_native_long_window",
        "next_retest_reason": "Confirm whether BNB can keep ranked positive flow after extending the native sample window.",
        "brief": (
            "BNB remains the highest-value beta flow watch leg, but long-window confirmation is still missing."
        ),
    }


def derive_window_verdict(short_return: float, long_return: float) -> tuple[str, str]:
    if short_return > 0.0 and long_return > 0.0:
        if long_return >= short_return * 0.5:
            return "stable_across_windows", "The edge stays positive without material decay after extending the window."
        return "degrades_on_long_window", "The edge remains positive on the long window, but it loses most of its short-window strength."
    if short_return > 0.0 and long_return <= 0.0:
        return "degrades_on_long_window", "The edge is positive on the short window but turns non-positive after extending the window."
    if short_return <= 0.0 and long_return > 0.0:
        return "improves_on_long_window", "The edge only emerges after extending the window."
    return "non_positive_both_windows", "The edge stays non-positive across both windows."


def derive_focus_payload_from_direct_native(
    short_payload: dict[str, Any],
    long_payload: dict[str, Any],
    xlong_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    short_flow = extract_native_family_best(short_payload, family="flow")
    long_flow = extract_native_family_best(long_payload, family="flow")
    short_price = extract_native_family_best(short_payload, family="price_state")
    long_price = extract_native_family_best(long_payload, family="price_state")
    xlong_flow = extract_native_family_best(xlong_payload, family="flow") if xlong_payload else None
    xlong_price = extract_native_family_best(xlong_payload, family="price_state") if xlong_payload else None

    flow_verdict, flow_note = derive_window_verdict(short_flow["return"], long_flow["return"])
    price_verdict, price_note = derive_window_verdict(short_price["return"], long_price["return"])

    if long_flow["return"] > 0.0 and long_flow["source"] == "ranked" and long_flow["timely_hit_rate"] >= 0.6:
        promotion_gate = "candidate_flow_secondary"
        promotion_gate_reason = "BNB keeps a ranked positive flow edge after extending the window."
        readiness = "promotion_candidate"
    else:
        promotion_gate = "blocked_until_long_window_confirms"
        promotion_gate_reason = (
            "BNB keeps the best beta flow on the short window, but the long-window sample only preserves a weak floor."
            if long_flow["return"] > 0.0
            else "BNB loses the positive flow edge once the native window is extended."
        )
        readiness = "watch_priority"

    flow_window_floor = "positive_but_weaker" if long_flow["return"] > 0.0 else "non_positive"
    flow_window_floor_note = (
        "Flow stays slightly positive on the extended BNB-only sample, but most of the short-window edge fades."
        if long_flow["return"] > 0.0
        else "Flow no longer stays positive on the extended BNB-only sample."
    )
    price_state_window_floor = "positive" if long_price["return"] > 0.0 else "negative"
    price_state_window_floor_note = (
        "Price-state stays positive on the extended BNB-only sample."
        if long_price["return"] > 0.0
        else "Price-state turns negative on the extended BNB-only sample."
    )
    comparative_window_takeaway = (
        "Long-window flow holds up better than long-window price-state, but it is still too weak for promotion."
        if long_flow["return"] > long_price["return"]
        else "Long-window price-state still dominates, so flow should remain in watch mode."
    )
    next_retest_action = "rerun_bnb_native_long_window"
    next_retest_reason = "Extend the native BNB-only window again before promoting flow beyond watch priority."
    xlong_flow_window_floor = ""
    xlong_flow_window_floor_note = ""
    xlong_comparative_window_takeaway = ""
    if xlong_flow and xlong_price:
        if xlong_flow["return"] > 0.0 and xlong_flow["source"] == "ranked" and xlong_flow["timely_hit_rate"] >= 0.5:
            xlong_flow_window_floor = "ranked_positive"
            xlong_flow_window_floor_note = "Extra-long flow remains ranked and positive."
            xlong_comparative_window_takeaway = (
                "Extra-long flow remains ranked and positive, but promotion still requires repeated confirmation."
            )
        elif xlong_flow["return"] > 0.0:
            xlong_flow_window_floor = "laggy_positive_only"
            xlong_flow_window_floor_note = (
                "Extra-long flow only survives as a discarded laggy signal; treat it as observational, not promotable."
            )
            xlong_comparative_window_takeaway = (
                "Extra-long flow keeps a raw positive return, but only in discarded laggy form; keep BNB in watch priority."
            )
            promotion_gate = "blocked_until_long_window_confirms"
            promotion_gate_reason = (
                "BNB keeps a raw positive flow return on the extra-long window, but only in discarded laggy form."
            )
            readiness = "watch_priority"
            next_retest_action = "wait_for_more_bnb_native_data"
            next_retest_reason = (
                "Extra-long native flow is still laggy; wait for fresher data before retrying promotion."
            )
        else:
            xlong_flow_window_floor = "non_positive"
            xlong_flow_window_floor_note = "Extra-long flow loses the positive floor entirely."
            xlong_comparative_window_takeaway = (
                "Extra-long flow loses the positive floor entirely, so BNB should stay in watch priority."
            )
            promotion_gate = "blocked_until_long_window_confirms"
            promotion_gate_reason = "BNB loses the positive flow edge on the extra-long native sample."
            readiness = "watch_priority"
            next_retest_action = "wait_for_more_bnb_native_data"
            next_retest_reason = (
                "Extra-long native flow is non-positive; wait for materially new data before rerunning."
            )

    return {
        "symbol": "BNBUSDT",
        "source_mode": "direct_bnb_native",
        "operator_status": readiness,
        "promotion_gate": promotion_gate,
        "promotion_gate_reason": promotion_gate_reason,
        "action": "watch_priority_until_long_window_confirms" if promotion_gate != "candidate_flow_secondary" else "candidate_flow_secondary",
        "action_reason": promotion_gate_reason,
        "flow_window_verdict": flow_verdict,
        "flow_window_note": flow_note,
        "price_state_window_verdict": price_verdict,
        "price_state_window_note": price_note,
        "flow_window_floor": flow_window_floor,
        "flow_window_floor_note": flow_window_floor_note,
        "price_state_window_floor": price_state_window_floor,
        "price_state_window_floor_note": price_state_window_floor_note,
        "comparative_window_takeaway": comparative_window_takeaway,
        "xlong_flow_window_floor": xlong_flow_window_floor,
        "xlong_flow_window_floor_note": xlong_flow_window_floor_note,
        "xlong_comparative_window_takeaway": xlong_comparative_window_takeaway,
        "short_flow_combo": short_flow["combo"],
        "short_flow_combo_canonical": short_flow["combo_canonical"],
        "short_flow_source": short_flow["source"],
        "short_flow_return": short_flow["return"],
        "short_flow_timely_hit_rate": short_flow["timely_hit_rate"],
        "long_flow_combo": long_flow["combo"],
        "long_flow_combo_canonical": long_flow["combo_canonical"],
        "long_flow_source": long_flow["source"],
        "long_flow_return": long_flow["return"],
        "long_flow_timely_hit_rate": long_flow["timely_hit_rate"],
        "xlong_flow_combo": xlong_flow["combo"] if xlong_flow else "",
        "xlong_flow_combo_canonical": xlong_flow["combo_canonical"] if xlong_flow else "",
        "xlong_flow_source": xlong_flow["source"] if xlong_flow else "",
        "xlong_flow_return": xlong_flow["return"] if xlong_flow else 0.0,
        "xlong_flow_timely_hit_rate": xlong_flow["timely_hit_rate"] if xlong_flow else 0.0,
        "flow_return_delta": long_flow["return"] - short_flow["return"],
        "flow_timely_hit_rate_delta": long_flow["timely_hit_rate"] - short_flow["timely_hit_rate"],
        "short_top_combo": short_price["combo"],
        "short_top_combo_canonical": short_price["combo_canonical"],
        "short_top_return": short_price["return"],
        "long_top_combo": long_price["combo"],
        "long_top_combo_canonical": long_price["combo_canonical"],
        "long_top_return": long_price["return"],
        "xlong_top_combo": xlong_price["combo"] if xlong_price else "",
        "xlong_top_combo_canonical": xlong_price["combo_canonical"] if xlong_price else "",
        "xlong_top_return": xlong_price["return"] if xlong_price else 0.0,
        "next_retest_action": next_retest_action,
        "next_retest_reason": next_retest_reason,
        "brief": (
            "BNB keeps the best short-window beta flow, but the extra-long sample only preserves laggy or fragile confirmation."
            if xlong_flow
            else "BNB keeps the best short-window beta flow and a weak long-window flow floor, but promotion still needs stronger long-window confirmation."
        ),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Binance Indicator BNB Flow Focus",
        "",
        f"- as_of: `{payload.get('as_of') or ''}`",
        f"- source_artifact: `{payload.get('source_artifact') or ''}`",
        f"- operator_status: `{payload.get('operator_status') or ''}`",
        f"- promotion_gate: `{payload.get('promotion_gate') or ''}`",
        f"- action: `{payload.get('action') or ''}`",
        "",
        "## Window Evidence",
        f"- flow_window_verdict: `{payload.get('flow_window_verdict') or ''}`",
        f"- price_state_window_verdict: `{payload.get('price_state_window_verdict') or ''}`",
        f"- flow_window_floor: `{payload.get('flow_window_floor') or '-'}`",
        f"- price_state_window_floor: `{payload.get('price_state_window_floor') or '-'}`",
        f"- short_flow: `{payload.get('short_flow_combo') or '-'}` return=`{float(payload.get('short_flow_return') or 0.0):.4f}` timely=`{float(payload.get('short_flow_timely_hit_rate') or 0.0):.2%}`",
        f"- long_flow: `{payload.get('long_flow_combo') or '-'}` return=`{float(payload.get('long_flow_return') or 0.0):.4f}` timely=`{float(payload.get('long_flow_timely_hit_rate') or 0.0):.2%}`",
        f"- xlong_flow: `{payload.get('xlong_flow_combo') or '-'}` source=`{payload.get('xlong_flow_source') or '-'}` return=`{float(payload.get('xlong_flow_return') or 0.0):.4f}` timely=`{float(payload.get('xlong_flow_timely_hit_rate') or 0.0):.2%}`",
        f"- short_top: `{payload.get('short_top_combo') or '-'}` return=`{float(payload.get('short_top_return') or 0.0):.4f}`",
        f"- long_top: `{payload.get('long_top_combo') or '-'}` return=`{float(payload.get('long_top_return') or 0.0):.4f}`",
        f"- xlong_top: `{payload.get('xlong_top_combo') or '-'}` return=`{float(payload.get('xlong_top_return') or 0.0):.4f}`",
        f"- canonical_short_flow: `{payload.get('short_flow_combo_canonical') or '-'}`",
        f"- canonical_long_flow: `{payload.get('long_flow_combo_canonical') or '-'}`",
        f"- canonical_short_top: `{payload.get('short_top_combo_canonical') or '-'}`",
        f"- canonical_long_top: `{payload.get('long_top_combo_canonical') or '-'}`",
        "",
        "## Gate",
        f"- reason: {payload.get('promotion_gate_reason') or ''}",
        f"- flow-floor-note: {payload.get('flow_window_floor_note') or ''}",
        f"- price-floor-note: {payload.get('price_state_window_floor_note') or ''}",
        f"- comparative-takeaway: {payload.get('comparative_window_takeaway') or ''}",
        f"- xlong-flow-floor: {payload.get('xlong_flow_window_floor') or '-'}",
        f"- xlong-flow-floor-note: {payload.get('xlong_flow_window_floor_note') or ''}",
        f"- xlong-comparative-takeaway: {payload.get('xlong_comparative_window_takeaway') or ''}",
        f"- next_retest_action: `{payload.get('next_retest_action') or ''}`",
        f"- next_retest_reason: {payload.get('next_retest_reason') or ''}",
        "",
        "## Brief",
        f"- {payload.get('brief') or ''}",
    ]
    return "\n".join(lines).strip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a focused BNB long-window flow report from the latest beta leg window artifact.")
    parser.add_argument("--review-dir", required=True)
    parser.add_argument("--now", default="")
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--artifact-keep", type=int, default=12)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    runtime_now = parse_now(args.now)

    source_paths: list[str] = []
    short_native_path = latest_bnb_native_artifact(review_dir, "bnb_short", runtime_now)
    long_native_path = latest_bnb_native_artifact(review_dir, "bnb_long", runtime_now)
    xlong_native_path = latest_bnb_native_artifact(review_dir, "bnb_xlong", runtime_now)
    if short_native_path and long_native_path:
        short_payload = json.loads(short_native_path.read_text(encoding="utf-8"))
        long_payload = json.loads(long_native_path.read_text(encoding="utf-8"))
        xlong_payload = json.loads(xlong_native_path.read_text(encoding="utf-8")) if xlong_native_path else None
        focus = derive_focus_payload_from_direct_native(short_payload, long_payload, xlong_payload)
        source_paths = [str(short_native_path), str(long_native_path)]
        if xlong_native_path:
            source_paths.append(str(xlong_native_path))
            source_path = xlong_native_path
        else:
            source_path = long_native_path
    else:
        source_path = latest_beta_leg_window_report(review_dir, runtime_now)
        source_payload = json.loads(source_path.read_text(encoding="utf-8"))
        focus = derive_focus_payload_from_beta_window(source_payload)
        source_paths = [str(source_path)]

    stamp = runtime_now.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = review_dir / f"{stamp}_binance_indicator_bnb_flow_focus.json"
    md_path = review_dir / f"{stamp}_binance_indicator_bnb_flow_focus.md"
    checksum_path = review_dir / f"{stamp}_binance_indicator_bnb_flow_focus_checksum.json"

    payload = {
        "ok": True,
        "status": "ok",
        "as_of": fmt_utc(runtime_now),
        "source_artifact": str(source_path),
        "source_artifacts": source_paths,
        **focus,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    checksum_payload = {
        "generated_at_utc": payload["as_of"],
        "artifact_ttl_hours": max(1.0, float(args.artifact_ttl_hours)),
        "files": [
            {"path": str(json_path), "sha256": sha256_file(json_path)},
            {"path": str(md_path), "sha256": sha256_file(md_path)},
        ],
    }
    checksum_path.write_text(json.dumps(checksum_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    pruned_keep, pruned_age = prune_review_artifacts(
        review_dir,
        stem="binance_indicator_bnb_flow_focus",
        current_paths=[json_path, md_path, checksum_path],
        keep=max(3, int(args.artifact_keep)),
        ttl_hours=float(args.artifact_ttl_hours),
        now_dt=runtime_now,
    )

    payload.update({
        "artifact": str(json_path),
        "markdown": str(md_path),
        "checksum": str(checksum_path),
        "pruned_keep": pruned_keep,
        "pruned_age": pruned_age,
    })
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checksum_payload["files"][0]["sha256"] = sha256_file(json_path)
    checksum_path.write_text(json.dumps(checksum_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
