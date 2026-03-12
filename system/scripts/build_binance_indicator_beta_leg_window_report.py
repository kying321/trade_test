#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any

TIMESTAMP_RE = re.compile(r"(?P<stamp>\d{8}T\d{6}Z)")
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


def artifact_stamp(path: Path) -> dt.datetime:
    match = TIMESTAMP_RE.search(path.name)
    if not match:
        return dt.datetime.min.replace(tzinfo=dt.timezone.utc)
    return dt.datetime.strptime(match.group("stamp"), "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)


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


def latest_beta_leg_report(review_dir: Path) -> tuple[Path, dict[str, Any]]:
    candidates = sorted(
        review_dir.glob("*_binance_indicator_native_beta_leg_report.json"),
        key=lambda item: (artifact_stamp(item), item.name),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("missing_binance_indicator_native_beta_leg_report")
    path = candidates[0]
    return path, json.loads(path.read_text(encoding="utf-8"))


def classify_flow_window(leg_payload: dict[str, Any]) -> tuple[str, str]:
    short_leg = dict(leg_payload.get("short") or {})
    long_leg = dict(leg_payload.get("long") or {})
    short_source = str(short_leg.get("flow_source") or "")
    long_source = str(long_leg.get("flow_source") or "")
    short_return = float(short_leg.get("flow_return") or 0.0)
    long_return = float(long_leg.get("flow_return") or 0.0)

    if short_source == "ranked" and short_return > 0.0 and long_source == "ranked" and long_return > 0.0:
        return "stable_across_windows", "Flow stays ranked and positive after extending the window."
    if short_source == "ranked" and short_return > 0.0 and long_source != "ranked":
        return "degrades_on_long_window", "Flow is positive in the short sample but drops out of the ranked set on the long window."
    if short_source == "ranked" and short_return > 0.0 and long_source == "ranked" and long_return <= 0.0:
        return "turns_negative_on_long_window", "Flow remains ranked on the long window but loses positive return."
    return "no_positive_flow_edge", "Flow does not produce a positive ranked edge even in the short sample."


def classify_price_window(leg_payload: dict[str, Any]) -> tuple[str, str]:
    short_leg = dict(leg_payload.get("short") or {})
    long_leg = dict(leg_payload.get("long") or {})
    short_return = float(short_leg.get("top_return") or 0.0)
    long_return = float(long_leg.get("top_return") or 0.0)
    if short_return > 0.0 and long_return > 0.0:
        return "stable_across_windows", "Price-state remains positive on both short and long windows."
    if short_return > 0.0 and long_return <= 0.0:
        return "degrades_on_long_window", "Price-state is positive on the short sample but turns non-positive on the long window."
    if short_return <= 0.0 and long_return > 0.0:
        return "improves_on_long_window", "Price-state improves when the window is extended."
    return "non_positive_both_windows", "Price-state remains non-positive across both windows."


def summarize_leg(symbol: str, leg_payload: dict[str, Any]) -> dict[str, Any]:
    short_leg = dict(leg_payload.get("short") or {})
    long_leg = dict(leg_payload.get("long") or {})
    flow_verdict, flow_note = classify_flow_window(leg_payload)
    price_verdict, price_note = classify_price_window(leg_payload)
    short_flow_return = float(short_leg.get("flow_return") or 0.0)
    long_flow_return = float(long_leg.get("flow_return") or 0.0)
    short_flow_timely = float(short_leg.get("flow_timely_hit_rate") or 0.0)
    long_flow_timely = float(long_leg.get("flow_timely_hit_rate") or 0.0)

    if symbol == "BNBUSDT":
        if flow_verdict == "degrades_on_long_window":
            action = "watch_priority_until_long_window_confirms"
            action_reason = "BNB has the best short-window beta flow, but it degrades when the window is extended."
        elif flow_verdict == "stable_across_windows":
            action = "candidate_flow_secondary"
            action_reason = "BNB keeps a positive ranked flow edge across windows."
        else:
            action = "watch_only"
            action_reason = flow_note
    else:
        if flow_verdict == "degrades_on_long_window":
            action = "watch_only"
            action_reason = "SOL shows short-window flow only and loses ranked flow on the long window."
        elif flow_verdict == "stable_across_windows":
            action = "candidate_flow_secondary"
            action_reason = "SOL keeps a positive ranked flow edge across windows."
        else:
            action = "deprioritize_flow"
            action_reason = flow_note

    return {
        "symbol": symbol,
        "flow_window_verdict": flow_verdict,
        "flow_window_note": flow_note,
        "price_state_window_verdict": price_verdict,
        "price_state_window_note": price_note,
        "short_flow_combo": short_leg.get("flow_combo"),
        "short_flow_combo_canonical": str(short_leg.get("flow_combo_canonical") or "") or canonical_combo_id(short_leg.get("flow_combo")),
        "short_flow_source": short_leg.get("flow_source"),
        "short_flow_return": short_flow_return,
        "short_flow_timely_hit_rate": short_flow_timely,
        "long_flow_combo": long_leg.get("flow_combo"),
        "long_flow_combo_canonical": str(long_leg.get("flow_combo_canonical") or "") or canonical_combo_id(long_leg.get("flow_combo")),
        "long_flow_source": long_leg.get("flow_source"),
        "long_flow_return": long_flow_return,
        "long_flow_timely_hit_rate": long_flow_timely,
        "flow_return_delta": long_flow_return - short_flow_return,
        "flow_timely_hit_rate_delta": long_flow_timely - short_flow_timely,
        "short_top_combo": short_leg.get("top_combo"),
        "short_top_combo_canonical": str(short_leg.get("top_combo_canonical") or "") or canonical_combo_id(short_leg.get("top_combo")),
        "short_top_return": float(short_leg.get("top_return") or 0.0),
        "long_top_combo": long_leg.get("top_combo"),
        "long_top_combo_canonical": str(long_leg.get("top_combo_canonical") or "") or canonical_combo_id(long_leg.get("top_combo")),
        "long_top_return": float(long_leg.get("top_return") or 0.0),
        "action": action,
        "action_reason": action_reason,
        "stability_status": dict(leg_payload.get("stability") or {}).get("status"),
        "stability_takeaway": dict(leg_payload.get("stability") or {}).get("takeaway"),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Binance Native Beta Leg Window Report",
        "",
        f"- as_of: `{payload.get('as_of') or ''}`",
        f"- source_artifact: `{payload.get('source_artifact') or ''}`",
        f"- basket_verdict: `{payload.get('basket_verdict') or ''}`",
        f"- next_focus_symbol: `{payload.get('next_focus_symbol') or '-'}`",
        f"- next_focus_action: `{payload.get('next_focus_action') or '-'}`",
        "",
        "## Window Stability",
    ]
    for symbol in ("BNBUSDT", "SOLUSDT"):
        row = payload.get("legs", {}).get(symbol, {})
        lines.extend([
            f"### {symbol}",
            f"- flow: `{row.get('flow_window_verdict')}`",
            f"- flow-note: {row.get('flow_window_note')}",
            f"- price-state: `{row.get('price_state_window_verdict')}`",
            f"- short-flow: `{row.get('short_flow_combo')}` canonical=`{row.get('short_flow_combo_canonical')}` source=`{row.get('short_flow_source')}` return=`{float(row.get('short_flow_return') or 0.0):.4f}` timely=`{float(row.get('short_flow_timely_hit_rate') or 0.0):.2%}`",
            f"- long-flow: `{row.get('long_flow_combo')}` canonical=`{row.get('long_flow_combo_canonical')}` source=`{row.get('long_flow_source')}` return=`{float(row.get('long_flow_return') or 0.0):.4f}` timely=`{float(row.get('long_flow_timely_hit_rate') or 0.0):.2%}`",
            f"- short-top: `{row.get('short_top_combo')}` canonical=`{row.get('short_top_combo_canonical')}` return=`{float(row.get('short_top_return') or 0.0):.4f}`",
            f"- long-top: `{row.get('long_top_combo')}` canonical=`{row.get('long_top_combo_canonical')}` return=`{float(row.get('long_top_return') or 0.0):.4f}`",
            f"- action: `{row.get('action')}`",
            f"- action-reason: {row.get('action_reason')}",
            "",
        ])
    lines.extend([
        "## Overall Takeaway",
        f"- {payload.get('overall_takeaway')}",
    ])
    return "\n".join(lines).strip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a short-vs-long beta leg window report for BNB/SOL flow stability.")
    parser.add_argument("--review-dir", required=True)
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--artifact-keep", type=int, default=12)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    runtime_now = now_utc()

    source_path, source_payload = latest_beta_leg_report(review_dir)
    source_legs = dict(source_payload.get("legs") or {})
    legs = {
        "BNBUSDT": summarize_leg("BNBUSDT", dict(source_legs.get("bnb") or {})),
        "SOLUSDT": summarize_leg("SOLUSDT", dict(source_legs.get("sol") or {})),
    }

    focus_symbol = "BNBUSDT"
    if str(legs["BNBUSDT"].get("flow_window_verdict")) != "degrades_on_long_window" and str(legs["SOLUSDT"].get("flow_window_verdict")) == "degrades_on_long_window":
        focus_symbol = "SOLUSDT"

    payload = {
        "ok": True,
        "status": "ok",
        "as_of": fmt_utc(runtime_now),
        "source_artifact": str(source_path),
        "basket_verdict": source_payload.get("basket_verdict"),
        "legs": legs,
        "next_focus_symbol": focus_symbol,
        "next_focus_action": legs[focus_symbol].get("action"),
        "next_focus_reason": legs[focus_symbol].get("action_reason"),
        "overall_takeaway": "Treat beta flow as short-window fragile until a leg keeps ranked positive flow on the long window.",
        "artifact_label": "binance-native-beta-leg-window-report:ok",
    }

    stamp = runtime_now.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = review_dir / f"{stamp}_binance_indicator_native_beta_leg_window_report.json"
    md_path = review_dir / f"{stamp}_binance_indicator_native_beta_leg_window_report.md"
    checksum_path = review_dir / f"{stamp}_binance_indicator_native_beta_leg_window_report_checksum.json"

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
        stem="binance_indicator_native_beta_leg_window_report",
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
