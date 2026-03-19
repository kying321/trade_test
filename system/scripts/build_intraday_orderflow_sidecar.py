#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = SYSTEM_ROOT / "output"
DEFAULT_REVIEW_DIR = DEFAULT_OUTPUT_ROOT / "review"
DEFAULT_MICRO_CAPTURE_DIR = DEFAULT_OUTPUT_ROOT / "artifacts" / "micro_capture"


def parse_stamp(raw: str) -> dt.datetime:
    parsed = dt.datetime.strptime(str(raw).strip(), "%Y%m%dT%H%M%SZ")
    return parsed.replace(tzinfo=dt.timezone.utc)


def parse_utc(raw: str) -> dt.datetime | None:
    txt = str(raw or "").strip()
    if not txt:
        return None
    try:
        parsed = dt.datetime.fromisoformat(txt.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def fmt_utc(value: dt.datetime | None) -> str:
    if value is None:
        return ""
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def text(value: Any) -> str:
    return str(value or "").strip()


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return float(out)


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def latest_artifact(review_dir: Path, pattern: str) -> Path:
    candidates = sorted(review_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"missing_required_artifact:{pattern}")
    return candidates[-1]


def safe_ratio(numer: float, denom: float) -> float:
    if denom <= 0:
        return 0.0
    return float(numer / denom)


def list_micro_capture_artifacts(micro_capture_dir: Path, limit: int) -> list[Path]:
    candidates = sorted(micro_capture_dir.glob("*_micro_capture.json"))
    if limit > 0:
        return candidates[-limit:]
    return candidates


def choose_research_logic_gate(*, sample_count: int, schema_ok_ratio: float, evidence_score_median: float, dominant_context_mode: str) -> str:
    if sample_count < 3:
        return "insufficient_samples"
    if schema_ok_ratio < 0.95:
        return "schema_unstable"
    if evidence_score_median < 0.8:
        return "weak_evidence"
    if not dominant_context_mode:
        return "missing_context_mode"
    return "ready_context_veto_only"


def summarize_symbol(
    *,
    symbol: str,
    samples: list[dict[str, Any]],
    cross_source_rows: list[dict[str, Any]],
    ladder_entry: dict[str, Any] | None,
) -> dict[str, Any]:
    sample_count = len(samples)
    captured_ts = [parse_utc(row.get("_captured_at_utc")) for row in samples]
    captured_ts = [x for x in captured_ts if x is not None]
    schema_ok_ratio = safe_ratio(sum(1 for row in samples if bool(row.get("schema_ok", False))), sample_count)
    sync_ok_ratio = safe_ratio(sum(1 for row in samples if bool(row.get("sync_ok", False))), sample_count)
    gap_ok_ratio = safe_ratio(sum(1 for row in samples if bool(row.get("gap_ok", False))), sample_count)
    time_sync_ok_ratio = safe_ratio(sum(1 for row in samples if bool(row.get("time_sync_ok", False))), sample_count)
    evidence_scores = [to_float(row.get("evidence_score"), 0.0) for row in samples]
    queue_imbalances = [to_float(row.get("queue_imbalance"), 0.0) for row in samples]
    ofi_norms = [to_float(row.get("ofi_norm"), 0.0) for row in samples]
    micro_alignments = [to_float(row.get("micro_alignment"), 0.0) for row in samples]
    trade_counts = [to_int(row.get("trade_count"), 0) for row in samples]
    context_counts = Counter(text(row.get("cvd_context_mode")) for row in samples if text(row.get("cvd_context_mode")))
    trust_counts = Counter(text(row.get("cvd_trust_tier_hint")) for row in samples if text(row.get("cvd_trust_tier_hint")))
    veto_counts = Counter(text(row.get("cvd_veto_hint")) for row in samples if text(row.get("cvd_veto_hint")))
    source_counts = Counter(text(row.get("source")) for row in samples if text(row.get("source")))

    cross_source_pass_ratio = safe_ratio(sum(1 for row in cross_source_rows if bool(row.get("pass", False))), len(cross_source_rows))
    cross_source_sync_ok_ratio = safe_ratio(sum(1 for row in cross_source_rows if bool(row.get("sync_ok", False))), len(cross_source_rows))
    cross_source_gap_ok_ratio = safe_ratio(sum(1 for row in cross_source_rows if bool(row.get("gap_ok", False))), len(cross_source_rows))
    cross_source_status_counts = Counter(text(row.get("row_status")) for row in cross_source_rows if text(row.get("row_status")))

    dominant_context_mode = context_counts.most_common(1)[0][0] if context_counts else ""
    dominant_trust_tier = trust_counts.most_common(1)[0][0] if trust_counts else ""
    dominant_veto_hint = veto_counts.most_common(1)[0][0] if veto_counts else ""

    research_logic_gate = choose_research_logic_gate(
        sample_count=sample_count,
        schema_ok_ratio=schema_ok_ratio,
        evidence_score_median=median(evidence_scores) if evidence_scores else 0.0,
        dominant_context_mode=dominant_context_mode,
    )
    role = text((ladder_entry or {}).get("role"))
    if research_logic_gate == "ready_context_veto_only" and role in {"mainline_primary", "majors_anchor"}:
        recommended_use = "price_state_plus_orderflow_context_veto"
    elif research_logic_gate == "ready_context_veto_only":
        recommended_use = "beta_flow_watch_only"
    else:
        recommended_use = "collect_more_sidecar_samples"

    latest_sample = max(samples, key=lambda row: parse_utc(row.get("_captured_at_utc")) or dt.datetime.min.replace(tzinfo=dt.timezone.utc))
    return {
        "symbol": symbol,
        "role": role,
        "priority": to_int((ladder_entry or {}).get("priority"), 0),
        "sample_count": int(sample_count),
        "sources_seen": dict(source_counts),
        "schema_ok_ratio": float(schema_ok_ratio),
        "sync_ok_ratio": float(sync_ok_ratio),
        "gap_ok_ratio": float(gap_ok_ratio),
        "time_sync_ok_ratio": float(time_sync_ok_ratio),
        "evidence_score_median": float(median(evidence_scores) if evidence_scores else 0.0),
        "trade_count_median": int(median(trade_counts) if trade_counts else 0),
        "queue_imbalance_median": float(median(queue_imbalances) if queue_imbalances else 0.0),
        "ofi_norm_median": float(median(ofi_norms) if ofi_norms else 0.0),
        "micro_alignment_median": float(median(micro_alignments) if micro_alignments else 0.0),
        "dominant_context_mode": dominant_context_mode,
        "context_mode_counts": dict(context_counts),
        "dominant_trust_tier": dominant_trust_tier,
        "trust_tier_counts": dict(trust_counts),
        "dominant_veto_hint": dominant_veto_hint,
        "veto_hint_counts": dict(veto_counts),
        "cross_source_audit_count": int(len(cross_source_rows)),
        "cross_source_pass_ratio": float(cross_source_pass_ratio),
        "cross_source_sync_ok_ratio": float(cross_source_sync_ok_ratio),
        "cross_source_gap_ok_ratio": float(cross_source_gap_ok_ratio),
        "cross_source_status_counts": dict(cross_source_status_counts),
        "latest_sample": {
            "captured_at_utc": text(latest_sample.get("_captured_at_utc")),
            "source": text(latest_sample.get("source")),
            "cvd_context_mode": text(latest_sample.get("cvd_context_mode")),
            "cvd_context_note": text(latest_sample.get("cvd_context_note")),
            "cvd_trust_tier_hint": text(latest_sample.get("cvd_trust_tier_hint")),
            "queue_imbalance": to_float(latest_sample.get("queue_imbalance"), 0.0),
            "ofi_norm": to_float(latest_sample.get("ofi_norm"), 0.0),
            "micro_alignment": to_float(latest_sample.get("micro_alignment"), 0.0),
            "trade_count": to_int(latest_sample.get("trade_count"), 0),
        },
        "research_logic_gate": research_logic_gate,
        "recommended_use": recommended_use,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Intraday Orderflow Sidecar",
        "",
        f"- generated_at_utc: `{text(payload.get('generated_at_utc'))}`",
        f"- research_decision: `{text(payload.get('research_decision'))}`",
        f"- recommended_brief: `{text(payload.get('recommended_brief'))}`",
        f"- source_micro_capture_count: `{int(payload.get('source_micro_capture_count', 0) or 0)}`",
        "",
        "## Symbol Summary",
        "",
    ]
    for row in payload.get("symbol_rows", []):
        lines.extend(
            [
                f"### {row['symbol']}",
                f"- role: `{row['role']}`",
                f"- research_logic_gate: `{row['research_logic_gate']}`",
                f"- recommended_use: `{row['recommended_use']}`",
                f"- sample_count: `{row['sample_count']}`",
                f"- schema_ok_ratio: `{float(row['schema_ok_ratio']):.2%}`",
                f"- evidence_score_median: `{float(row['evidence_score_median']):.3f}`",
                f"- dominant_context_mode: `{text(row['dominant_context_mode'])}`",
                f"- cross_source_pass_ratio: `{float(row['cross_source_pass_ratio']):.2%}`",
                f"- latest_sample: `{json.dumps(row['latest_sample'], ensure_ascii=False)}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Notes",
            "",
            f"- `{text(payload.get('research_note'))}`",
            f"- `{text(payload.get('limitation_note'))}`",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a RESEARCH_ONLY intraday orderflow sidecar from local micro_capture artifacts."
    )
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--micro-capture-dir", default=str(DEFAULT_MICRO_CAPTURE_DIR))
    parser.add_argument("--blueprint-path", default="")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--stamp", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stamp_dt = parse_stamp(args.stamp)
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    micro_capture_dir = Path(args.micro_capture_dir).expanduser().resolve()
    blueprint_path = Path(args.blueprint_path).expanduser().resolve() if text(args.blueprint_path) else latest_artifact(review_dir, "*_intraday_orderflow_strategy_blueprint.json")
    blueprint = load_json_mapping(blueprint_path)

    symbol_input = [item.strip().upper() for item in text(args.symbols).split(",") if item.strip()]
    if symbol_input:
        target_symbols = symbol_input
    else:
        target_symbols = [text(row.get("symbol")).upper() for row in blueprint.get("symbol_depth_ladder", []) if text(row.get("symbol"))]
    if not target_symbols:
        raise ValueError("missing_target_symbols")

    micro_paths = list_micro_capture_artifacts(micro_capture_dir, int(args.limit))
    if not micro_paths:
        raise FileNotFoundError(f"missing_micro_capture_artifacts:{micro_capture_dir}")

    symbol_samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    symbol_cross_source_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    capture_context_counts = Counter()

    for path in micro_paths:
        payload = load_json_mapping(path)
        captured_at_utc = fmt_utc(parse_utc(payload.get("captured_at_utc")))
        selected_micro = payload.get("selected_micro") or []
        if isinstance(selected_micro, list):
            for raw in selected_micro:
                if not isinstance(raw, dict):
                    continue
                symbol = text(raw.get("symbol")).upper()
                if symbol not in target_symbols:
                    continue
                row = dict(raw)
                row["_artifact_path"] = str(path)
                row["_captured_at_utc"] = captured_at_utc
                symbol_samples[symbol].append(row)
                capture_context_counts[text(raw.get("cvd_context_mode"))] += 1
        cross_source_rows = ((payload.get("cross_source_audit") or {}).get("rows") or [])
        if isinstance(cross_source_rows, list):
            for raw in cross_source_rows:
                if not isinstance(raw, dict):
                    continue
                symbol = text(raw.get("symbol")).upper()
                if symbol not in target_symbols:
                    continue
                row = dict(raw)
                row["_artifact_path"] = str(path)
                row["_captured_at_utc"] = captured_at_utc
                symbol_cross_source_rows[symbol].append(row)

    ladder_map = {text(row.get("symbol")).upper(): row for row in blueprint.get("symbol_depth_ladder", []) if isinstance(row, dict)}
    symbol_rows = [
        summarize_symbol(
            symbol=symbol,
            samples=symbol_samples.get(symbol, []),
            cross_source_rows=symbol_cross_source_rows.get(symbol, []),
            ladder_entry=ladder_map.get(symbol),
        )
        for symbol in target_symbols
        if symbol_samples.get(symbol)
    ]
    if not symbol_rows:
        raise ValueError("no_target_symbols_found_in_micro_capture_artifacts")
    symbol_rows.sort(key=lambda row: (int(row.get("priority", 99)), row["symbol"]))

    ready_symbols = [row["symbol"] for row in symbol_rows if row["research_logic_gate"] == "ready_context_veto_only"]
    blocked_symbols = [row["symbol"] for row in symbol_rows if row["research_logic_gate"] != "ready_context_veto_only"]
    research_decision = (
        "build_orderflow_context_veto_research_pack"
        if {"ETHUSDT", "BTCUSDT"}.issubset(set(ready_symbols))
        else "collect_more_orderflow_sidecar_samples_before_pack"
    )
    recommended_brief = (
        f"orderflow_sidecar:ready={','.join(ready_symbols) if ready_symbols else '-'},"
        f"blocked={','.join(blocked_symbols) if blocked_symbols else '-'},"
        f"decision={research_decision}"
    )

    payload = {
        "action": "build_intraday_orderflow_sidecar",
        "ok": True,
        "status": "ok",
        "change_class": "RESEARCH_ONLY",
        "generated_at_utc": fmt_utc(stamp_dt),
        "blueprint_path": str(blueprint_path),
        "source_micro_capture_dir": str(micro_capture_dir),
        "source_micro_capture_count": int(len(micro_paths)),
        "target_symbols": target_symbols,
        "capture_context_counts": dict(capture_context_counts),
        "symbol_rows": symbol_rows,
        "ready_symbols": ready_symbols,
        "blocked_symbols": blocked_symbols,
        "research_decision": research_decision,
        "recommended_brief": recommended_brief,
        "research_note": (
            "这份 sidecar 只聚合本地 micro_capture 工件，为 ETH/BTC/BNB/SOL 提供统一订单流研究字段；"
            "它的角色是 research-only context/veto，不是 execution 输入。"
        ),
        "limitation_note": (
            "当前 sidecar 来自历史 micro_capture 快照，不是连续逐笔研究集；"
            "time_sync / cross_source 风险仍在，因此只能先支持逻辑验证与样本分层，不可直接升级为回测真值或 live 依赖。"
        ),
    }

    json_path = review_dir / f"{args.stamp}_intraday_orderflow_sidecar.json"
    md_path = review_dir / f"{args.stamp}_intraday_orderflow_sidecar.md"
    latest_json_path = review_dir / "latest_intraday_orderflow_sidecar.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    latest_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"json_path": str(json_path), "md_path": str(md_path), "latest_json_path": str(latest_json_path), "ready_symbols": ready_symbols}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
