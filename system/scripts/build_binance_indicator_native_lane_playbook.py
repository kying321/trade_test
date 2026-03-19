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


def latest_group_report(review_dir: Path) -> tuple[Path, dict[str, Any]]:
    candidates = sorted(
        review_dir.glob("*_binance_indicator_native_group_report.json"),
        key=lambda item: (artifact_stamp(item), item.name),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("missing_binance_indicator_native_group_report")
    path = candidates[0]
    return path, json.loads(path.read_text(encoding="utf-8"))


def latest_group_artifact(review_dir: Path, group_name: str) -> tuple[Path, dict[str, Any]]:
    candidates = sorted(
        review_dir.glob("*_binance_indicator_combo_native_crypto.json"),
        key=lambda item: (artifact_stamp(item), item.name),
        reverse=True,
    )
    fallback: tuple[Path, dict[str, Any]] | None = None
    partial_fallback: tuple[Path, dict[str, Any]] | None = None
    for path in candidates:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if str(payload.get("symbol_group") or "") == group_name:
            rank = artifact_status_rank(payload)
            if rank >= 2:
                return path, payload
            if fallback is None and rank >= 1:
                fallback = (path, payload)
            if partial_fallback is None:
                partial_fallback = (path, payload)
    if fallback is not None:
        return fallback
    if partial_fallback is not None:
        return partial_fallback
    raise FileNotFoundError(f"missing_native_group_artifact:{group_name}")


def latest_stability_report(review_dir: Path) -> tuple[Path | None, dict[str, Any] | None]:
    candidates = sorted(
        review_dir.glob("*_binance_indicator_native_lane_stability_report.json"),
        key=lambda item: (artifact_stamp(item), item.name),
        reverse=True,
    )
    if not candidates:
        return None, None
    path = candidates[0]
    return path, json.loads(path.read_text(encoding="utf-8"))


def latest_beta_leg_report(review_dir: Path) -> tuple[Path | None, dict[str, Any] | None]:
    candidates = sorted(
        review_dir.glob("*_binance_indicator_native_beta_leg_report.json"),
        key=lambda item: (artifact_stamp(item), item.name),
        reverse=True,
    )
    if not candidates:
        return None, None
    path = candidates[0]
    return path, json.loads(path.read_text(encoding="utf-8"))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_group_payload(
    review_dir: Path,
    configured_path: str | None,
    fallback_group: str,
) -> tuple[Path, dict[str, Any], str]:
    if configured_path:
        path = Path(configured_path).expanduser().resolve()
        if path.exists():
            return path, load_json(path), "configured_path"
    path, payload = latest_group_artifact(review_dir, fallback_group)
    return path, payload, "latest_group_fallback"


def metric_signature(row: dict[str, Any], precision: int = 10) -> tuple[float, float, int]:
    return (
        round(float(row.get("avg_total_return") or 0.0), precision),
        round(float(row.get("avg_timely_hit_rate") or 0.0), precision),
        int(row.get("trade_count") or 0),
    )


def equivalent_family(
    ranked_rows: list[dict[str, Any]],
    *,
    anchor_combo: str | None = None,
) -> list[dict[str, Any]]:
    if not ranked_rows:
        return []
    if anchor_combo is None:
        anchor = ranked_rows[0]
    else:
        anchor_canonical = canonical_combo_id(anchor_combo)
        anchor = next(
            (row for row in ranked_rows if canonical_combo_id(row.get("combo_id")) == anchor_canonical),
            None,
        )
        if anchor is None:
            return []
    signature = metric_signature(anchor)
    family = [row for row in ranked_rows if metric_signature(row) == signature]
    return family


def first_by_prefix(ranked_rows: list[dict[str, Any]], prefixes: tuple[str, ...]) -> dict[str, Any] | None:
    for row in ranked_rows:
        combo_id = str(row.get("combo_id") or "")
        if combo_id.startswith(prefixes):
            return row
    return None


def family_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "combo_ids": [str(row.get("combo_id") or "") for row in rows],
        "canonical_combo_ids": [canonical_combo_id(row.get("combo_id")) for row in rows],
        "avg_total_return": float(rows[0].get("avg_total_return") or 0.0) if rows else 0.0,
        "avg_timely_hit_rate": float(rows[0].get("avg_timely_hit_rate") or 0.0) if rows else 0.0,
        "trade_count": int(rows[0].get("trade_count") or 0) if rows else 0,
        "family_size": len(rows),
    }


def derive_beta_leg_routes(beta_leg_report: dict[str, Any] | None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not beta_leg_report:
        return None, None
    legs = dict(beta_leg_report.get("legs") or {})
    if not legs:
        return None, None

    routes: dict[str, Any] = {}
    focus_entry: dict[str, Any] | None = None
    for leg_name, leg_payload in legs.items():
        stability = dict(leg_payload.get("stability") or {})
        short_leg = dict(leg_payload.get("short") or {})
        status = str(stability.get("status") or "unknown")
        short_flow_return = float(short_leg.get("flow_return") or 0.0)
        short_flow_source = str(short_leg.get("flow_source") or "")

        if status == "stable_single_leg_flow_secondary":
            route_status = "candidate_single_leg_flow_secondary"
            recommended_use = "allow_beta_leg_secondary_confirmation"
        elif status == "short_window_only_flow":
            route_status = "research_watch_short_window_only"
            recommended_use = "watch_short_window_only"
        elif status == "fragile_single_leg_flow":
            route_status = "research_watch_fragile"
            recommended_use = "watch_fragile_flow_only"
        else:
            route_status = "deprioritize_flow_leg"
            recommended_use = "deprioritize_flow_leg"

        route = {
            "status": route_status,
            "recommended_use": recommended_use,
            "stability_status": status,
            "stability_takeaway": str(stability.get("takeaway") or ""),
            "short_flow_combo": short_leg.get("flow_combo"),
            "short_flow_combo_canonical": canonical_combo_id(short_leg.get("flow_combo")),
            "short_flow_source": short_flow_source or None,
            "short_flow_return": short_flow_return,
        }
        routes[str(leg_name)] = route

        if short_flow_source == "ranked" and short_flow_return > 0.0:
            if focus_entry is None or short_flow_return > float(focus_entry.get("short_flow_return") or 0.0):
                focus_entry = {
                    "symbol": str(leg_name).upper(),
                    "route_status": route_status,
                    "recommended_use": recommended_use,
                    "short_flow_return": short_flow_return,
                    "reason": str(stability.get("takeaway") or ""),
                }

    return routes, focus_entry


def derive_lane(
    group_name: str,
    payload: dict[str, Any],
    group_status: dict[str, Any],
    stability_entry: dict[str, Any] | None,
    beta_leg_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ranked_rows = list(payload.get("native_crypto_family", {}).get("ranked_combos", []))
    top_family = equivalent_family(ranked_rows)
    top_combo = str(top_family[0].get("combo_id") or "") if top_family else None
    flow_anchor = group_status.get("taker_best_combo")
    flow_family = equivalent_family(ranked_rows, anchor_combo=str(flow_anchor or ""))
    flow_status = str(group_status.get("status") or "")
    stability_status = str((stability_entry or {}).get("status") or "")
    stability_takeaway = str((stability_entry or {}).get("takeaway") or "")

    crowding_incremental_value = "none_observed"
    if len(flow_family) == 1 and flow_family and str(flow_family[0].get("combo_id") or "") == "crowding_filtered_breakout":
        crowding_incremental_value = "only_flow_variant"

    if group_name == "beta" and stability_status == "stable_flow_secondary":
        deployment_status = "price_state_plus_flow_secondary"
        lane_takeaway = "Keep price-state breakout as the trigger and add taker/OI family as beta-only secondary confirmation."
    elif group_name == "beta" and flow_status == "taker_positive_secondary":
        deployment_status = "flow_secondary_research_hold"
        lane_takeaway = "Flow family improved, but keep it in research hold until longer-window stability is confirmed."
    elif group_name == "majors" and stability_status == "stable_price_state_primary":
        deployment_status = "price_state_primary_only"
        lane_takeaway = "Keep majors on price-state triggers only; flow family stays context/watch until it turns positive."
    elif group_name == "majors" and flow_status.startswith("taker_"):
        deployment_status = "price_state_primary_watch_flow"
        lane_takeaway = "Keep majors on price-state triggers; flow stack is still watch-only until stability improves."
    else:
        deployment_status = "research_only"
        lane_takeaway = "Do not promote this lane beyond research."

    leg_constraint = None
    leg_constraint_takeaway = None
    beta_leg_routes = None
    beta_leg_focus = None
    if group_name == "beta" and beta_leg_report:
        beta_leg_routes, beta_leg_focus = derive_beta_leg_routes(beta_leg_report)
        basket_verdict = str(beta_leg_report.get("basket_verdict") or "")
        if basket_verdict == "basket_only_flow_secondary":
            leg_constraint = basket_verdict
            leg_constraint_takeaway = str(beta_leg_report.get("overall_takeaway") or "")
            if deployment_status == "price_state_plus_flow_secondary":
                deployment_status = "price_state_plus_flow_secondary_basket_only"
            lane_takeaway = (
                f"{lane_takeaway} {leg_constraint_takeaway}".strip()
                if leg_constraint_takeaway
                else lane_takeaway
            )
        elif basket_verdict in {"single_leg_supported", "fragile_leg_only_flow", "no_leg_flow_support"}:
            leg_constraint = basket_verdict
            leg_constraint_takeaway = str(beta_leg_report.get("overall_takeaway") or "")
            lane_takeaway = (
                f"{lane_takeaway} {leg_constraint_takeaway}".strip()
                if leg_constraint_takeaway
                else lane_takeaway
            )

    return {
        "group": group_name,
        "deployment_status": deployment_status,
        "top_combo": top_combo,
        "top_combo_canonical": canonical_combo_id(top_combo),
        "flow_anchor_combo": flow_anchor,
        "flow_anchor_combo_canonical": canonical_combo_id(flow_anchor),
        "top_family": family_summary(top_family),
        "flow_family": family_summary(flow_family),
        "flow_status": flow_status,
        "stability_status": stability_status or "unavailable",
        "stability_takeaway": stability_takeaway or "No stability report available.",
        "crowding_incremental_value": crowding_incremental_value,
        "lane_takeaway": lane_takeaway,
        "leg_constraint": leg_constraint,
        "leg_constraint_takeaway": leg_constraint_takeaway,
        "beta_leg_routes": beta_leg_routes,
        "beta_leg_focus": beta_leg_focus,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Binance Native Crypto Lane Playbook",
        "",
        f"- as_of: `{payload.get('as_of') or ''}`",
        f"- group_report: `{payload.get('source_artifacts', {}).get('group_report') or ''}`",
        f"- stability_report: `{payload.get('source_artifacts', {}).get('stability_report') or ''}`",
        "",
        "## Deployment Summary",
        f"- majors: `{payload.get('lanes', {}).get('majors', {}).get('deployment_status')}`",
        f"- beta: `{payload.get('lanes', {}).get('beta', {}).get('deployment_status')}`",
        f"- overall: {payload.get('overall_takeaway')}",
        "",
    ]
    for lane_key in ("majors", "beta"):
        lane = payload.get("lanes", {}).get(lane_key, {})
        top_family = lane.get("top_family", {})
        flow_family = lane.get("flow_family", {})
        lines.extend(
            [
                f"## {lane_key.title()}",
                f"- deployment: `{lane.get('deployment_status')}`",
                f"- stability: `{lane.get('stability_status')}`",
                f"- top-family: `{', '.join(top_family.get('combo_ids', []))}`",
                f"- top-family-canonical: `{', '.join(top_family.get('canonical_combo_ids', []))}`",
                f"- flow-family: `{', '.join(flow_family.get('combo_ids', []))}`",
                f"- flow-family-canonical: `{', '.join(flow_family.get('canonical_combo_ids', []))}`",
                f"- crowding incremental value: `{lane.get('crowding_incremental_value')}`",
                f"- leg constraint: `{lane.get('leg_constraint')}`",
                f"- stability takeaway: {lane.get('stability_takeaway')}",
                f"- leg constraint takeaway: {lane.get('leg_constraint_takeaway')}",
                f"- takeaway: {lane.get('lane_takeaway')}",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Derive majors vs beta native crypto deployment lanes.")
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

    group_report_path, group_report = latest_group_report(review_dir)
    stability_report_path, stability_report = latest_stability_report(review_dir)
    beta_leg_report_path, beta_leg_report = latest_beta_leg_report(review_dir)
    majors_path, majors_payload, majors_resolution = resolve_group_payload(
        review_dir,
        str(group_report.get("source_artifacts", {}).get("majors") or ""),
        "majors",
    )
    beta_path, beta_payload, beta_resolution = resolve_group_payload(
        review_dir,
        str(group_report.get("source_artifacts", {}).get("beta") or ""),
        "beta",
    )

    stability_lanes = (stability_report or {}).get("stability", {})
    majors_lane = derive_lane(
        "majors",
        majors_payload,
        group_report["groups"]["majors"],
        stability_lanes.get("majors"),
        None,
    )
    beta_lane = derive_lane(
        "beta",
        beta_payload,
        group_report["groups"]["beta"],
        stability_lanes.get("beta"),
        beta_leg_report,
    )

    if beta_lane["deployment_status"] == "price_state_plus_flow_secondary_basket_only":
        overall_takeaway = (
            "Keep majors on price-state only. For beta, use taker/OI only as a basket-level secondary confirmation family; "
            "do not promote SOL or BNB as standalone flow-secondary lanes."
        )
    elif beta_lane["deployment_status"] == "price_state_plus_flow_secondary":
        overall_takeaway = (
            "Promote taker/OI only on beta names as a secondary confirmation family; keep majors on price-state breakout until flow stacks turn positive."
        )
    elif beta_lane["deployment_status"] == "flow_secondary_research_hold":
        overall_takeaway = (
            "Keep majors on price-state only. For beta, flow has recovered enough to stay in research hold, but longer-window stability is still missing and at least one leg remains short-window-only."
        )
    else:
        overall_takeaway = (
            "Keep both lanes on research-grade deployment until stability confirms beta flow and majors continue to reject flow leadership."
        )

    payload = {
        "ok": True,
        "status": "ok",
        "as_of": fmt_utc(runtime_now),
        "source_artifacts": {
            "group_report": str(group_report_path),
            "stability_report": str(stability_report_path) if stability_report_path else None,
            "beta_leg_report": str(beta_leg_report_path) if beta_leg_report_path else None,
            "majors": str(majors_path),
            "beta": str(beta_path),
        },
        "source_resolution": {
            "majors": majors_resolution,
            "beta": beta_resolution,
        },
        "lanes": {
            "majors": majors_lane,
            "beta": beta_lane,
        },
        "recommended_live_research_split": {
            "majors": majors_lane["deployment_status"],
            "beta": beta_lane["deployment_status"],
        },
        "overall_takeaway": overall_takeaway,
        "artifact_label": "binance-native-crypto-lane-playbook:ok",
    }

    stamp = runtime_now.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = review_dir / f"{stamp}_binance_indicator_native_lane_playbook.json"
    md_path = review_dir / f"{stamp}_binance_indicator_native_lane_playbook.md"
    checksum_path = review_dir / f"{stamp}_binance_indicator_native_lane_playbook_checksum.json"
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
        stem="binance_indicator_native_lane_playbook",
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
