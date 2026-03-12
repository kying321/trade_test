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


def latest_group_artifact(review_dir: Path, group_name: str) -> tuple[Path, dict[str, Any]]:
    candidates = sorted(
        review_dir.glob("*_binance_indicator_combo_native_crypto.json"),
        key=lambda item: (artifact_stamp(item), item.name),
        reverse=True,
    )
    fallback: tuple[Path, dict[str, Any]] | None = None
    for path in candidates:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if str(payload.get("symbol_group") or "") == group_name:
            rank = artifact_status_rank(payload)
            if rank >= 2:
                return path, payload
            if fallback is None and rank >= 1:
                fallback = (path, payload)
    if fallback is not None:
        return fallback
    raise FileNotFoundError(f"missing_native_group_artifact:{group_name}")


def latest_beta_stability_artifact(review_dir: Path) -> tuple[Path | None, dict[str, Any] | None]:
    candidates = sorted(
        review_dir.glob("*_binance_indicator_native_lane_stability_report.json"),
        key=lambda item: (artifact_stamp(item), item.name),
        reverse=True,
    )
    if not candidates:
        return None, None
    path = candidates[0]
    payload = json.loads(path.read_text(encoding="utf-8"))
    return path, payload


def best_flow_anywhere(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    ranked = list(payload.get("native_crypto_family", {}).get("ranked_combos", []))
    for row in ranked:
        combo_id = str(row.get("combo_id") or "")
        if combo_id.startswith("taker_oi") or combo_id.startswith("crowding_filtered"):
            return row, "ranked"

    discarded = list(payload.get("native_crypto_family", {}).get("discarded_combos", []))
    for row in discarded:
        combo_id = str(row.get("combo_id") or "")
        if combo_id.startswith("taker_oi") or combo_id.startswith("crowding_filtered"):
            return row, "discarded"
    return None, "missing"


def top_row(payload: dict[str, Any]) -> dict[str, Any]:
    ranked = list(payload.get("native_crypto_family", {}).get("ranked_combos", []))
    return ranked[0] if ranked else {}


def leg_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    top = top_row(payload)
    flow, source = best_flow_anywhere(payload)
    return {
        "symbol_group": str(payload.get("symbol_group") or ""),
        "top_combo": str(top.get("combo_id") or ""),
        "top_combo_canonical": canonical_combo_id(top.get("combo_id")),
        "top_return": float(top.get("avg_total_return") or 0.0),
        "top_timely_hit_rate": float(top.get("avg_timely_hit_rate") or 0.0),
        "top_trade_count": int(top.get("trade_count") or 0),
        "flow_combo": str(flow.get("combo_id") or "") if flow else None,
        "flow_combo_canonical": canonical_combo_id(flow.get("combo_id")) if flow else None,
        "flow_source": source,
        "flow_return": float(flow.get("avg_total_return") or 0.0) if flow else None,
        "flow_timely_hit_rate": float(flow.get("avg_timely_hit_rate") or 0.0) if flow else None,
        "flow_trade_count": int(flow.get("trade_count") or 0) if flow else 0,
        "flow_discard_reason": str(flow.get("discard_reason") or "") if flow else None,
    }


def classify_leg(short_leg: dict[str, Any], long_leg: dict[str, Any]) -> dict[str, Any]:
    short_source = str(short_leg.get("flow_source") or "")
    long_source = str(long_leg.get("flow_source") or "")
    short_flow = float(short_leg.get("flow_return") or 0.0)
    long_flow = float(long_leg.get("flow_return") or 0.0)

    if short_source == "ranked" and short_flow > 0.0 and long_source == "ranked" and long_flow > 0.0:
        status = "stable_single_leg_flow_secondary"
        takeaway = "This leg keeps positive ranked taker/OI value across short and extended windows."
    elif short_source == "ranked" and short_flow > 0.0 and long_source != "ranked":
        status = "short_window_only_flow"
        takeaway = "This leg shows flow recovery only in the short sample; do not promote it alone."
    elif short_source == "ranked" and short_flow > 0.0 and long_source == "ranked" and long_flow <= 0.0:
        status = "fragile_single_leg_flow"
        takeaway = "This leg keeps ranked flow in the long sample but does not maintain positive return."
    else:
        status = "no_single_leg_flow_secondary"
        takeaway = "This leg does not support promoting taker/OI on a standalone basis."

    return {
        "status": status,
        "takeaway": takeaway,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Binance Native Beta Leg Report",
        "",
        f"- as_of: `{payload.get('as_of') or ''}`",
        "",
        "## Leg Stability",
        f"- SOL: `{payload['legs']['sol']['stability']['status']}`",
        f"- BNB: `{payload['legs']['bnb']['stability']['status']}`",
        f"- basket takeaway: {payload.get('overall_takeaway')}",
        "",
    ]
    for leg_name in ("sol", "bnb"):
        leg = payload["legs"][leg_name]
        lines.extend(
            [
                f"## {leg_name.upper()}",
                f"- short: top=`{leg['short']['top_combo']}` top_canonical=`{leg['short']['top_combo_canonical']}` top_return=`{leg['short']['top_return']:.4f}` flow=`{leg['short']['flow_combo']}` flow_canonical=`{leg['short']['flow_combo_canonical']}` flow_source=`{leg['short']['flow_source']}` flow_return=`{float(leg['short']['flow_return'] or 0.0):.4f}`",
                f"- long: top=`{leg['long']['top_combo']}` top_canonical=`{leg['long']['top_combo_canonical']}` top_return=`{leg['long']['top_return']:.4f}` flow=`{leg['long']['flow_combo']}` flow_canonical=`{leg['long']['flow_combo_canonical']}` flow_source=`{leg['long']['flow_source']}` flow_return=`{float(leg['long']['flow_return'] or 0.0):.4f}`",
                f"- takeaway: {leg['stability']['takeaway']}",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare SOL vs BNB native crypto flow recovery.")
    parser.add_argument("--review-dir", required=True)
    parser.add_argument("--short-sol-group", default="sol")
    parser.add_argument("--short-bnb-group", default="bnb")
    parser.add_argument("--long-sol-group", default="sol_long")
    parser.add_argument("--long-bnb-group", default="bnb_long")
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--artifact-keep", type=int, default=12)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    runtime_now = now_utc()

    short_sol_path, short_sol_payload = latest_group_artifact(review_dir, str(args.short_sol_group))
    short_bnb_path, short_bnb_payload = latest_group_artifact(review_dir, str(args.short_bnb_group))
    long_sol_path, long_sol_payload = latest_group_artifact(review_dir, str(args.long_sol_group))
    long_bnb_path, long_bnb_payload = latest_group_artifact(review_dir, str(args.long_bnb_group))
    stability_path, stability_payload = latest_beta_stability_artifact(review_dir)

    legs = {
        "sol": {
            "short": leg_snapshot(short_sol_payload),
            "long": leg_snapshot(long_sol_payload),
        },
        "bnb": {
            "short": leg_snapshot(short_bnb_payload),
            "long": leg_snapshot(long_bnb_payload),
        },
    }
    legs["sol"]["stability"] = classify_leg(legs["sol"]["short"], legs["sol"]["long"])
    legs["bnb"]["stability"] = classify_leg(legs["bnb"]["short"], legs["bnb"]["long"])

    beta_lane_status = (
        (stability_payload or {}).get("stability", {}).get("beta", {}).get("status")
        if stability_payload
        else None
    )
    stable_leg_supported = any(
        str(legs[leg_name]["stability"]["status"]) == "stable_single_leg_flow_secondary"
        for leg_name in ("sol", "bnb")
    )
    fragile_leg_present = any(
        str(legs[leg_name]["stability"]["status"]) in {"short_window_only_flow", "fragile_single_leg_flow"}
        for leg_name in ("sol", "bnb")
    )

    if stable_leg_supported:
        basket_verdict = "single_leg_supported"
        overall_takeaway = "At least one beta leg supports standalone flow-secondary deployment."
    elif str(beta_lane_status or "") == "stable_flow_secondary":
        basket_verdict = "basket_only_flow_secondary"
        overall_takeaway = (
            "Beta flow survives only at the basket level. Do not promote SOL or BNB alone as a stable flow-secondary lane."
        )
    elif fragile_leg_present or str(beta_lane_status or "") == "fragile_flow_secondary":
        basket_verdict = "fragile_leg_only_flow"
        overall_takeaway = (
            "Beta flow recovery remains short-window or fragile at the leg level. Keep SOL and BNB in research-watch mode only."
        )
    else:
        basket_verdict = "no_leg_flow_support"
        overall_takeaway = "No beta leg supports promoting taker/OI beyond research-watch mode."

    payload = {
        "ok": True,
        "status": "ok",
        "as_of": fmt_utc(runtime_now),
        "source_artifacts": {
            "short_sol": str(short_sol_path),
            "short_bnb": str(short_bnb_path),
            "long_sol": str(long_sol_path),
            "long_bnb": str(long_bnb_path),
            "beta_stability_report": str(stability_path) if stability_path else None,
        },
        "legs": legs,
        "basket_verdict": basket_verdict,
        "overall_takeaway": overall_takeaway,
        "artifact_label": "binance-native-beta-leg-report:ok",
    }

    stamp = runtime_now.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = review_dir / f"{stamp}_binance_indicator_native_beta_leg_report.json"
    md_path = review_dir / f"{stamp}_binance_indicator_native_beta_leg_report.md"
    checksum_path = review_dir / f"{stamp}_binance_indicator_native_beta_leg_report_checksum.json"
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
        stem="binance_indicator_native_beta_leg_report",
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
