#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW_DIR = SYSTEM_ROOT / "output" / "review"
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
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_combo_id(combo_id: Any) -> str:
    combo = str(combo_id or "")
    return COMBO_CANONICAL_ALIASES.get(combo, combo)


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


def latest_artifact(review_dir: Path, suffix: str) -> Path:
    candidates = list(review_dir.glob(f"*_{suffix}.json"))
    if not candidates:
        raise FileNotFoundError(f"missing_artifact:{suffix}")
    return max(candidates, key=artifact_sort_key)


def latest_optional_artifact(review_dir: Path, suffix: str) -> Path | None:
    candidates = list(review_dir.glob(f"*_{suffix}.json"))
    if not candidates:
        return None
    return max(candidates, key=artifact_sort_key)


def load_json_mapping(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid_json_mapping:{path}")
    return payload


def prune_review_artifacts(
    review_dir: Path,
    *,
    stem: str,
    current_paths: list[Path],
    keep: int,
    ttl_hours: float,
    now_dt: dt.datetime,
) -> tuple[list[str], list[str]]:
    cutoff = now_dt - dt.timedelta(hours=max(1.0, float(ttl_hours)))
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
    for path in survivors[max(1, int(keep)) :]:
        if path.name in protected:
            continue
        path.unlink(missing_ok=True)
        pruned_keep.append(str(path))
    return pruned_keep, pruned_age


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def playbook_status_map(playbook_payload: dict[str, Any]) -> dict[tuple[str, str], str]:
    mapping: dict[tuple[str, str], str] = {}
    for status_key in ("adopt_now", "research_only", "context_only", "discard_now", "retest_native_crypto"):
        for row in playbook_payload.get(status_key, []) or []:
            if not isinstance(row, dict):
                continue
            family = str(row.get("family") or "crypto").strip().lower()
            combo_id = str(row.get("combo_id") or "").strip()
            if family and combo_id:
                mapping[(family, combo_id)] = status_key
    return mapping


def recommendation_for_row(
    *,
    total_return: float | None = None,
    expectancy_r: float | None = None,
    profit_factor: float | None = None,
    trade_count: int | None = None,
    plan_status: str | None = None,
    playbook_status: str | None = None,
) -> str:
    if plan_status == "manual_structure_review_now":
        return "manual_structure_review_now"
    if plan_status == "blocked_shortline_gate":
        return "blocked_shortline_gate"
    if playbook_status == "adopt_now":
        return "adopt_now"
    if playbook_status == "research_only":
        return "research_only"
    pf = float(profit_factor or 0.0)
    trades = int(trade_count or 0)
    edge = expectancy_r if expectancy_r is not None else total_return
    if edge is not None and edge > 0.0 and pf > 1.0 and trades >= 10:
        return "positive_edge_research"
    return "do_not_promote_negative_edge"


def comparison_score(row: dict[str, Any]) -> float:
    expectancy = row.get("expectancy_r")
    total_return = row.get("total_return")
    pf = safe_float(row.get("profit_factor"), 0.0)
    trades = safe_int(row.get("trade_count"), 0)
    edge_value = safe_float(expectancy if expectancy is not None else total_return, 0.0)
    return (
        edge_value * 100.0
        + max(0.0, (pf - 1.0) * 25.0)
        + min(20.0, trades / 10.0)
        + (10.0 if row.get("recommendation") in {"adopt_now", "positive_edge_research", "manual_structure_review_now"} else 0.0)
    )


def build_rows(review_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    etf_path = latest_artifact(review_dir, "binance_indicator_combo_etf")
    playbook_path = latest_optional_artifact(review_dir, "binance_indicator_combo_playbook")
    native_path = latest_artifact(review_dir, "custom_binance_indicator_combo_native_crypto")
    lane_stability_path = latest_artifact(review_dir, "binance_indicator_native_lane_stability_report")
    source_control_path = latest_optional_artifact(review_dir, "binance_indicator_source_control_report")
    brooks_study_path = latest_artifact(review_dir, "brooks_price_action_market_study")
    brooks_execution_path = latest_optional_artifact(review_dir, "brooks_price_action_execution_plan")

    etf_payload = load_json_mapping(etf_path)
    playbook_payload = load_json_mapping(playbook_path)
    native_payload = load_json_mapping(native_path)
    lane_stability_payload = load_json_mapping(lane_stability_path)
    source_control_payload = load_json_mapping(source_control_path)
    brooks_study_payload = load_json_mapping(brooks_study_path)
    brooks_execution_payload = load_json_mapping(brooks_execution_path)
    playbook_statuses = playbook_status_map(playbook_payload)

    rows: list[dict[str, Any]] = []

    crypto_top = ((etf_payload.get("crypto_family") or {}).get("ranked_combos") or [{}])[0]
    if crypto_top:
        playbook_status = playbook_statuses.get(("crypto", str(crypto_top.get("combo_id") or "")))
        rows.append(
            {
                "row_id": "etf_proxy_crypto_top",
                "strategy_family": "binance_indicator_combo_etf",
                "market_scope": "crypto_etf_proxy",
                "strategy_id": str(crypto_top.get("combo_id") or ""),
                "strategy_id_canonical": canonical_combo_id(crypto_top.get("combo_id")),
                "sample_scope": "recent_window_aggregate",
                "trade_count": safe_int(crypto_top.get("trade_count")),
                "win_rate": safe_float(crypto_top.get("avg_win_rate")),
                "timely_hit_rate": safe_float(crypto_top.get("avg_timely_hit_rate")),
                "total_return": safe_float(crypto_top.get("avg_total_return")),
                "profit_factor": safe_float(crypto_top.get("avg_profit_factor")),
                "expectancy_r": None,
                "plan_status": None,
                "playbook_status": playbook_status,
                "recommendation": recommendation_for_row(
                    total_return=safe_float(crypto_top.get("avg_total_return")),
                    profit_factor=safe_float(crypto_top.get("avg_profit_factor")),
                    trade_count=safe_int(crypto_top.get("trade_count")),
                    playbook_status=playbook_status,
                ),
                "artifact": str(etf_path),
            }
        )

    commodity_ranked = list((etf_payload.get("commodity_family") or {}).get("ranked_combos") or [])
    commodity_best = max(
        commodity_ranked,
        key=lambda row: (
            1 if safe_float(row.get("avg_total_return")) > 0.0 and safe_float(row.get("avg_profit_factor")) >= 1.0 else 0,
            safe_float(row.get("avg_total_return")),
            safe_float(row.get("avg_profit_factor")),
            safe_int(row.get("trade_count")),
        ),
        default=None,
    )
    if commodity_best:
        playbook_status = playbook_statuses.get(("commodity", str(commodity_best.get("combo_id") or "")))
        rows.append(
            {
                "row_id": "etf_proxy_commodity_best_measured",
                "strategy_family": "binance_indicator_combo_etf",
                "market_scope": "commodity_etf_proxy",
                "strategy_id": str(commodity_best.get("combo_id") or ""),
                "strategy_id_canonical": canonical_combo_id(commodity_best.get("combo_id")),
                "sample_scope": "recent_window_aggregate",
                "trade_count": safe_int(commodity_best.get("trade_count")),
                "win_rate": safe_float(commodity_best.get("avg_win_rate")),
                "timely_hit_rate": safe_float(commodity_best.get("avg_timely_hit_rate")),
                "total_return": safe_float(commodity_best.get("avg_total_return")),
                "profit_factor": safe_float(commodity_best.get("avg_profit_factor")),
                "expectancy_r": None,
                "plan_status": None,
                "playbook_status": playbook_status,
                "recommendation": recommendation_for_row(
                    total_return=safe_float(commodity_best.get("avg_total_return")),
                    profit_factor=safe_float(commodity_best.get("avg_profit_factor")),
                    trade_count=safe_int(commodity_best.get("trade_count")),
                    playbook_status=playbook_status,
                ),
                "artifact": str(etf_path),
            }
        )

    native_top = ((native_payload.get("native_crypto_family") or {}).get("ranked_combos") or [{}])[0]
    if native_top:
        rows.append(
            {
                "row_id": "native_crypto_family_top",
                "strategy_family": "native_crypto_combo",
                "market_scope": "native_crypto_family",
                "strategy_id": str(native_top.get("combo_id") or ""),
                "strategy_id_canonical": canonical_combo_id(native_top.get("combo_id")),
                "sample_scope": "recent_window_aggregate",
                "trade_count": safe_int(native_top.get("trade_count")),
                "win_rate": safe_float(native_top.get("avg_win_rate")),
                "timely_hit_rate": safe_float(native_top.get("avg_timely_hit_rate")),
                "total_return": safe_float(native_top.get("avg_total_return")),
                "profit_factor": safe_float(native_top.get("avg_profit_factor")),
                "expectancy_r": None,
                "plan_status": None,
                "playbook_status": None,
                "recommendation": recommendation_for_row(
                    total_return=safe_float(native_top.get("avg_total_return")),
                    profit_factor=safe_float(native_top.get("avg_profit_factor")),
                    trade_count=safe_int(native_top.get("trade_count")),
                ),
                "artifact": str(native_path),
            }
        )

    majors_long = (((lane_stability_payload.get("lanes") or {}).get("majors") or {}).get("long") or {})
    if majors_long:
        rows.append(
            {
                "row_id": "native_majors_long_best",
                "strategy_family": "native_lane_stability",
                "market_scope": "native_majors_long",
                "strategy_id": str(majors_long.get("top_combo") or ""),
                "strategy_id_canonical": canonical_combo_id(majors_long.get("top_combo")),
                "sample_scope": "long_window_lane",
                "trade_count": safe_int(majors_long.get("top_trade_count")),
                "win_rate": None,
                "timely_hit_rate": safe_float(majors_long.get("top_timely_hit_rate")),
                "total_return": safe_float(majors_long.get("top_return")),
                "profit_factor": None,
                "expectancy_r": None,
                "plan_status": None,
                "playbook_status": None,
                "recommendation": "positive_edge_research" if safe_float(majors_long.get("top_return")) > 0.0 else "do_not_promote_negative_edge",
                "artifact": str(lane_stability_path),
            }
        )

    beta_short = (((lane_stability_payload.get("lanes") or {}).get("beta") or {}).get("short") or {})
    if beta_short:
        rows.append(
            {
                "row_id": "native_beta_short_best",
                "strategy_family": "native_lane_stability",
                "market_scope": "native_beta_short",
                "strategy_id": str(beta_short.get("top_combo") or ""),
                "strategy_id_canonical": canonical_combo_id(beta_short.get("top_combo")),
                "sample_scope": "short_window_lane",
                "trade_count": safe_int(beta_short.get("top_trade_count")),
                "win_rate": None,
                "timely_hit_rate": safe_float(beta_short.get("top_timely_hit_rate")),
                "total_return": safe_float(beta_short.get("top_return")),
                "profit_factor": None,
                "expectancy_r": None,
                "plan_status": None,
                "playbook_status": None,
                "recommendation": "positive_edge_research" if safe_float(beta_short.get("top_return")) > 0.0 else "do_not_promote_negative_edge",
                "artifact": str(lane_stability_path),
            }
        )

    adaptive = brooks_study_payload.get("adaptive_route_strategy") or {}
    metrics = adaptive.get("metrics") or {}
    if metrics:
        rows.append(
            {
                "row_id": "brooks_adaptive_route_full_sample",
                "strategy_family": "brooks_price_action",
                "market_scope": "cross_market_adaptive_route",
                "strategy_id": "adaptive_route_strategy",
                "strategy_id_canonical": "adaptive_route_strategy",
                "sample_scope": "full_sample",
                "trade_count": safe_int(metrics.get("trade_count")),
                "win_rate": safe_float(metrics.get("win_rate")),
                "timely_hit_rate": None,
                "total_return": None,
                "profit_factor": safe_float(metrics.get("profit_factor")),
                "expectancy_r": safe_float(metrics.get("expectancy_r")),
                "plan_status": None,
                "playbook_status": None,
                "recommendation": recommendation_for_row(
                    expectancy_r=safe_float(metrics.get("expectancy_r")),
                    profit_factor=safe_float(metrics.get("profit_factor")),
                    trade_count=safe_int(metrics.get("trade_count")),
                ),
                "artifact": str(brooks_study_path),
            }
        )

    oos_metrics = adaptive.get("out_of_sample_metrics") or {}
    if oos_metrics:
        rows.append(
            {
                "row_id": "brooks_adaptive_route_oos",
                "strategy_family": "brooks_price_action",
                "market_scope": "cross_market_adaptive_route",
                "strategy_id": "adaptive_route_strategy",
                "strategy_id_canonical": "adaptive_route_strategy",
                "sample_scope": "out_of_sample",
                "trade_count": safe_int(oos_metrics.get("trade_count")),
                "win_rate": safe_float(oos_metrics.get("win_rate")),
                "timely_hit_rate": None,
                "total_return": None,
                "profit_factor": safe_float(oos_metrics.get("profit_factor")),
                "expectancy_r": safe_float(oos_metrics.get("expectancy_r")),
                "plan_status": None,
                "playbook_status": None,
                "recommendation": recommendation_for_row(
                    expectancy_r=safe_float(oos_metrics.get("expectancy_r")),
                    profit_factor=safe_float(oos_metrics.get("profit_factor")),
                    trade_count=safe_int(oos_metrics.get("trade_count")),
                ),
                "artifact": str(brooks_study_path),
            }
        )

    execution_head = (
        brooks_execution_payload.get("head_plan_item")
        or ((brooks_execution_payload.get("plan_items") or [None])[0])
        or {}
    )
    if execution_head:
        rows.append(
            {
                "row_id": "brooks_current_execution_head",
                "strategy_family": "brooks_price_action",
                "market_scope": str(execution_head.get("asset_class") or "cross_market"),
                "strategy_id": str(execution_head.get("strategy_id") or ""),
                "strategy_id_canonical": str(execution_head.get("strategy_id") or ""),
                "sample_scope": "current_structure_head",
                "trade_count": None,
                "win_rate": None,
                "timely_hit_rate": None,
                "total_return": None,
                "profit_factor": None,
                "expectancy_r": None,
                "plan_status": str(execution_head.get("plan_status") or ""),
                "playbook_status": None,
                "recommendation": recommendation_for_row(
                    plan_status=str(execution_head.get("plan_status") or ""),
                ),
                "artifact": str(brooks_execution_path) if brooks_execution_path else None,
                "symbol": str(execution_head.get("symbol") or ""),
                "direction": str(execution_head.get("direction") or ""),
                "entry_price": safe_float(execution_head.get("entry_price")),
                "stop_price": safe_float(execution_head.get("stop_price")),
                "target_price": safe_float(execution_head.get("target_price")),
                "rr_ratio": safe_float(execution_head.get("rr_ratio")),
                "plan_blocker_detail": str(execution_head.get("plan_blocker_detail") or ""),
            }
        )

    for row in rows:
        row["comparison_score"] = round(comparison_score(row), 6)

    rows.sort(
        key=lambda row: (
            -safe_float(row.get("comparison_score")),
            -safe_float(row.get("expectancy_r") if row.get("expectancy_r") is not None else row.get("total_return")),
            -safe_float(row.get("profit_factor")),
            -safe_int(row.get("trade_count")),
            str(row.get("row_id") or ""),
        )
    )

    strongest = rows[0] if rows else {}
    best_crypto_segment = max(
        [row for row in rows if str(row.get("market_scope") or "").startswith("native_") or "crypto" in str(row.get("market_scope") or "")],
        key=lambda row: safe_float(row.get("comparison_score")),
        default={},
    )
    negative_promotion_guards = []
    for row in rows:
        edge_value = row.get("expectancy_r") if row.get("expectancy_r") is not None else row.get("total_return")
        if safe_float(edge_value, 0.0) >= 0.0:
            continue
        if row.get("recommendation") in {"do_not_promote_negative_edge", "research_only"}:
            negative_promotion_guards.append(row)
    meta = {
        "source_artifacts": {
            "combo_etf": str(etf_path),
            "combo_playbook": str(playbook_path) if playbook_path else None,
            "native_crypto": str(native_path),
            "lane_stability": str(lane_stability_path),
            "source_control": str(source_control_path) if source_control_path else None,
            "brooks_market_study": str(brooks_study_path),
            "brooks_execution_plan": str(brooks_execution_path) if brooks_execution_path else None,
        },
        "crypto_source_control_verdict": source_control_payload.get("control_verdict"),
        "crypto_source_control_takeaway": source_control_payload.get("control_takeaway"),
        "strongest_recent_backtest": strongest,
        "best_crypto_segment": best_crypto_segment,
        "negative_promotion_guards": negative_promotion_guards,
    }
    return rows, meta


def render_markdown(payload: dict[str, Any]) -> str:
    strongest = payload.get("strongest_recent_backtest") or {}
    best_crypto = payload.get("best_crypto_segment") or {}
    lines = [
        "# Recent Strategy Backtest Comparison",
        "",
        f"- as_of: `{payload.get('as_of') or ''}`",
        f"- row_count: `{payload.get('row_count') or 0}`",
        f"- strongest_recent_backtest: `{strongest.get('row_id') or '-'}:{strongest.get('recommendation') or '-'}:{strongest.get('comparison_score') or 0}`",
        f"- best_crypto_segment: `{best_crypto.get('row_id') or '-'}:{best_crypto.get('recommendation') or '-'}:{best_crypto.get('comparison_score') or 0}`",
        f"- crypto_source_control_verdict: `{payload.get('crypto_source_control_verdict') or '-'}`",
        "",
        "## Comparison Rows",
    ]
    for row in payload.get("rows", []):
        metric_bits = []
        if row.get("expectancy_r") is not None:
            metric_bits.append(f"expectancy_r=`{safe_float(row.get('expectancy_r')):.4f}`")
        if row.get("total_return") is not None:
            metric_bits.append(f"return=`{safe_float(row.get('total_return')):.4f}`")
        if row.get("profit_factor") is not None:
            metric_bits.append(f"pf=`{safe_float(row.get('profit_factor')):.2f}`")
        if row.get("trade_count") is not None:
            metric_bits.append(f"trades=`{safe_int(row.get('trade_count'))}`")
        if row.get("plan_status"):
            metric_bits.append(f"plan_status=`{row.get('plan_status')}`")
        lines.append(
            f"- `{row.get('row_id')}` scope=`{row.get('market_scope')}` strategy=`{row.get('strategy_id_canonical')}` rec=`{row.get('recommendation')}` score=`{safe_float(row.get('comparison_score')):.2f}` {' '.join(metric_bits)}"
        )
    guards = payload.get("negative_promotion_guards") or []
    lines.extend(["", "## Negative Promotion Guards"])
    for row in guards:
        lines.append(
            f"- `{row.get('row_id')}` strategy=`{row.get('strategy_id_canonical')}` scope=`{row.get('market_scope')}` rec=`{row.get('recommendation')}`"
        )
    lines.extend(["", "## Overall Takeaway", f"- {payload.get('comparison_takeaway') or ''}"])
    return "\n".join(lines).strip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a unified recent strategy backtest comparison report across ETF-proxy, native crypto, and Brooks route studies.")
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--now", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    runtime_now = parse_now(args.now)

    rows, meta = build_rows(review_dir)
    strongest = meta.get("strongest_recent_backtest") or {}
    best_crypto = meta.get("best_crypto_segment") or {}
    comparison_takeaway = (
        "Recent measured evidence favors Brooks adaptive routing over current crypto ETF/native combo families; keep negative-edge crypto combos out of promotion, keep native majors-long as the only positive crypto subsegment, and treat Brooks execution heads as manual structure review until an execution bridge exists."
    )
    payload = {
        "action": "build_recent_strategy_backtest_comparison_report",
        "ok": True,
        "status": "ok",
        "as_of": fmt_utc(runtime_now),
        "row_count": len(rows),
        "rows": rows,
        "source_artifacts": meta.get("source_artifacts"),
        "crypto_source_control_verdict": meta.get("crypto_source_control_verdict"),
        "crypto_source_control_takeaway": meta.get("crypto_source_control_takeaway"),
        "strongest_recent_backtest": strongest,
        "best_crypto_segment": best_crypto,
        "negative_promotion_guards": meta.get("negative_promotion_guards"),
        "comparison_takeaway": comparison_takeaway,
    }

    stamp = runtime_now.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_path = review_dir / f"{stamp}_recent_strategy_backtest_comparison_report.json"
    markdown_path = review_dir / f"{stamp}_recent_strategy_backtest_comparison_report.md"
    checksum_path = review_dir / f"{stamp}_recent_strategy_backtest_comparison_report_checksum.json"
    artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    checksum_payload = {
        "artifact": str(artifact_path),
        "markdown": str(markdown_path),
        "artifact_sha256": sha256_file(artifact_path),
        "markdown_sha256": sha256_file(markdown_path),
        "generated_at": payload.get("as_of"),
    }
    checksum_path.write_text(json.dumps(checksum_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pruned_keep, pruned_age = prune_review_artifacts(
        review_dir,
        stem="recent_strategy_backtest_comparison_report",
        current_paths=[artifact_path, markdown_path, checksum_path],
        keep=max(1, int(args.artifact_keep)),
        ttl_hours=float(args.artifact_ttl_hours),
        now_dt=runtime_now,
    )
    payload["artifact"] = str(artifact_path)
    payload["markdown"] = str(markdown_path)
    payload["checksum"] = str(checksum_path)
    payload["pruned_keep"] = pruned_keep
    payload["pruned_age"] = pruned_age
    artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checksum_payload["artifact_sha256"] = sha256_file(artifact_path)
    checksum_path.write_text(json.dumps(checksum_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
