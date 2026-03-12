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


def top_row(payload: dict[str, Any]) -> dict[str, Any]:
    ranked = list(payload.get("native_crypto_family", {}).get("ranked_combos", []))
    return ranked[0] if ranked else {}


def best_flow_row(payload: dict[str, Any]) -> dict[str, Any] | None:
    ranked = list(payload.get("native_crypto_family", {}).get("ranked_combos", []))
    for row in ranked:
        combo_id = str(row.get("combo_id") or "")
        if combo_id.startswith("taker_oi") or combo_id.startswith("crowding_filtered"):
            return row
    return None


def lane_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    top = top_row(payload)
    flow = best_flow_row(payload)
    return {
        "symbol_group": str(payload.get("symbol_group") or ""),
        "top_combo": str(top.get("combo_id") or ""),
        "top_combo_canonical": canonical_combo_id(top.get("combo_id")),
        "top_return": float(top.get("avg_total_return") or 0.0),
        "top_timely_hit_rate": float(top.get("avg_timely_hit_rate") or 0.0),
        "top_trade_count": int(top.get("trade_count") or 0),
        "flow_combo": str(flow.get("combo_id") or "") if flow else None,
        "flow_combo_canonical": canonical_combo_id(flow.get("combo_id")) if flow else None,
        "flow_return": float(flow.get("avg_total_return") or 0.0) if flow else None,
        "flow_timely_hit_rate": float(flow.get("avg_timely_hit_rate") or 0.0) if flow else None,
        "flow_trade_count": int(flow.get("trade_count") or 0) if flow else 0,
    }


def classify_stability(short_lane: dict[str, Any], long_lane: dict[str, Any], *, lane_name: str) -> dict[str, Any]:
    short_flow = float(short_lane.get("flow_return") or 0.0)
    long_flow = float(long_lane.get("flow_return") or 0.0)
    short_top = float(short_lane.get("top_return") or 0.0)
    long_top = float(long_lane.get("top_return") or 0.0)

    if lane_name == "beta":
        if short_flow > 0.0 and long_flow > 0.0:
            status = "stable_flow_secondary"
            takeaway = "Beta names keep positive taker/OI secondary value across both short and extended windows."
        elif short_flow > 0.0 or long_flow > 0.0:
            status = "fragile_flow_secondary"
            takeaway = "Beta taker/OI turns positive only in part of the sample; keep it secondary but treat as fragile."
        else:
            status = "no_flow_secondary"
            takeaway = "Beta taker/OI did not stay positive; do not promote it beyond research."
    else:
        if short_top > 0.0 and long_top > 0.0 and short_flow <= 0.0 and long_flow <= 0.0:
            status = "stable_price_state_primary"
            takeaway = "Majors remain price-state first across both short and extended windows while taker/OI stays non-positive."
        elif short_flow > 0.0 or long_flow > 0.0:
            status = "flow_recheck_needed"
            takeaway = "Majors flow stack improved in at least one window; re-check before keeping a hard majors exclusion."
        else:
            status = "price_state_primary"
            takeaway = "Majors still favor price-state over flow, but short/long samples are not both positive."

    return {
        "status": status,
        "takeaway": takeaway,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Binance Native Crypto Lane Stability Report",
        "",
        f"- as_of: `{payload.get('as_of') or ''}`",
        "",
        "## Stability",
        f"- majors: `{payload['stability']['majors']['status']}`",
        f"- beta: `{payload['stability']['beta']['status']}`",
        f"- overall: {payload.get('overall_takeaway')}",
        "",
    ]
    for lane_name in ("majors", "beta"):
        lane = payload["lanes"][lane_name]
        lines.extend(
            [
                f"## {lane_name.title()}",
                f"- short: top=`{lane['short']['top_combo']}` top_canonical=`{lane['short']['top_combo_canonical']}` top_return=`{lane['short']['top_return']:.4f}` flow=`{lane['short']['flow_combo']}` flow_canonical=`{lane['short']['flow_combo_canonical']}` flow_return=`{float(lane['short']['flow_return'] or 0.0):.4f}`",
                f"- long: top=`{lane['long']['top_combo']}` top_canonical=`{lane['long']['top_combo_canonical']}` top_return=`{lane['long']['top_return']:.4f}` flow=`{lane['long']['flow_combo']}` flow_canonical=`{lane['long']['flow_combo_canonical']}` flow_return=`{float(lane['long']['flow_return'] or 0.0):.4f}`",
                f"- takeaway: {payload['stability'][lane_name]['takeaway']}",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare short vs extended majors/beta native crypto lanes.")
    parser.add_argument("--review-dir", required=True)
    parser.add_argument("--short-majors-group", default="majors")
    parser.add_argument("--short-beta-group", default="beta")
    parser.add_argument("--long-majors-group", default="majors_long")
    parser.add_argument("--long-beta-group", default="beta_long")
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--artifact-keep", type=int, default=12)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    runtime_now = now_utc()

    short_majors_path, short_majors_payload = latest_group_artifact(review_dir, str(args.short_majors_group))
    short_beta_path, short_beta_payload = latest_group_artifact(review_dir, str(args.short_beta_group))
    long_majors_path, long_majors_payload = latest_group_artifact(review_dir, str(args.long_majors_group))
    long_beta_path, long_beta_payload = latest_group_artifact(review_dir, str(args.long_beta_group))

    lanes = {
        "majors": {
            "short": lane_snapshot(short_majors_payload),
            "long": lane_snapshot(long_majors_payload),
        },
        "beta": {
            "short": lane_snapshot(short_beta_payload),
            "long": lane_snapshot(long_beta_payload),
        },
    }
    stability = {
        "majors": classify_stability(lanes["majors"]["short"], lanes["majors"]["long"], lane_name="majors"),
        "beta": classify_stability(lanes["beta"]["short"], lanes["beta"]["long"], lane_name="beta"),
    }

    if stability["beta"]["status"] == "stable_flow_secondary" and stability["majors"]["status"] == "stable_price_state_primary":
        overall_takeaway = "Keep majors on price-state only and promote taker/OI as a beta-only secondary layer; the split survives the longer window."
    else:
        overall_takeaway = "The majors/beta split still needs caution under the longer window; keep deployment conservative."

    payload = {
        "ok": True,
        "status": "ok",
        "as_of": fmt_utc(runtime_now),
        "source_artifacts": {
            "short_majors": str(short_majors_path),
            "short_beta": str(short_beta_path),
            "long_majors": str(long_majors_path),
            "long_beta": str(long_beta_path),
        },
        "lanes": lanes,
        "stability": stability,
        "overall_takeaway": overall_takeaway,
        "artifact_label": "binance-native-crypto-lane-stability:ok",
    }

    stamp = runtime_now.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = review_dir / f"{stamp}_binance_indicator_native_lane_stability_report.json"
    md_path = review_dir / f"{stamp}_binance_indicator_native_lane_stability_report.md"
    checksum_path = review_dir / f"{stamp}_binance_indicator_native_lane_stability_report_checksum.json"
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
        stem="binance_indicator_native_lane_stability_report",
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
