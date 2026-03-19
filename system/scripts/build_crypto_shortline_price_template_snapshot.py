#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW_DIR = SYSTEM_ROOT / "output" / "review"
DEFAULT_OUTPUT_ROOT = SYSTEM_ROOT / "output"


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


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


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


def find_latest(review_dir: Path, pattern: str) -> Path | None:
    files = sorted(review_dir.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
    return files[0] if files else None


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
        "*_crypto_shortline_price_template_snapshot.json",
        "*_crypto_shortline_price_template_snapshot.md",
        "*_crypto_shortline_price_template_snapshot_checksum.json",
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


def read_bars_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


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


def find_gate_row(payload: dict[str, Any], symbol: str) -> dict[str, Any]:
    for row in as_list(payload.get("symbols")):
        item = as_dict(row)
        if text(item.get("symbol")).upper() == symbol.upper():
            return item
    return {}


def candidate_symbols(
    route_symbol_value: str,
    execution_gate_payload: dict[str, Any],
    live_bars_snapshot_payload: dict[str, Any],
) -> list[str]:
    ordered: list[str] = []
    for raw in [route_symbol_value, *as_list(live_bars_snapshot_payload.get("materialized_symbols"))]:
        symbol = text(raw).upper()
        if symbol and symbol not in ordered:
            ordered.append(symbol)
    for row in as_list(execution_gate_payload.get("symbols")):
        symbol = text(as_dict(row).get("symbol")).upper()
        if symbol and symbol not in ordered:
            ordered.append(symbol)
    return ordered


def derive_side(gate_row: dict[str, Any], live_snapshot_payload: dict[str, Any]) -> str:
    micro = as_dict(gate_row.get("micro_signals"))
    attack_side = text(micro.get("attack_side")).lower()
    if attack_side == "buyers":
        return "LONG"
    if attack_side == "sellers":
        return "SHORT"
    alignment = safe_float(micro.get("micro_alignment"), 0.0)
    if alignment > 0:
        return "LONG"
    if alignment < 0:
        return "SHORT"
    structure = as_dict(live_snapshot_payload.get("structure_signals"))
    if bool(structure.get("sweep_long")) or bool(structure.get("candle_long")):
        return "LONG"
    if bool(structure.get("sweep_short")) or bool(structure.get("candle_short")):
        return "SHORT"
    return "LONG"


def sort_symbol_bars(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    work = frame.copy()
    if "symbol" not in work.columns:
        return work.iloc[0:0]
    work = work[work["symbol"].astype(str).str.upper() == symbol.upper()].copy()
    if work.empty:
        return work
    if "ts" in work.columns:
        work["_ts_sort"] = pd.to_datetime(work["ts"], utc=True, errors="coerce")
        work = work.sort_values("_ts_sort")
    return work


def build_template_from_bars(frame: pd.DataFrame, side: str) -> dict[str, Any]:
    if frame.empty or len(frame) < 6:
        return {
            "template_ready": False,
            "reason": "insufficient_bars",
            "entry_price": 0.0,
            "stop_price": 0.0,
            "target_price": 0.0,
            "risk_per_unit": 0.0,
            "reward_multiple": 0.0,
            "reference_bar_count": int(len(frame)),
        }
    work = frame.tail(8).copy()
    for column in ("open", "high", "low", "close", "volume"):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=["high", "low", "close"])
    if work.empty or len(work) < 6:
        return {
            "template_ready": False,
            "reason": "invalid_bar_data",
            "entry_price": 0.0,
            "stop_price": 0.0,
            "target_price": 0.0,
            "risk_per_unit": 0.0,
            "reward_multiple": 0.0,
            "reference_bar_count": int(len(work)),
        }
    latest = work.iloc[-1]
    recent = work.tail(5)
    avg_range = float((recent["high"] - recent["low"]).clip(lower=0.0).mean())
    avg_range = max(avg_range, 1e-6)
    latest_close = float(latest["close"])
    if side == "SHORT":
        swing_extreme = float(recent["high"].max())
        stop_price = max(swing_extreme, latest_close + avg_range * 1.2)
        risk = stop_price - latest_close
        target_price = latest_close - max(risk * 2.0, avg_range * 1.8)
    else:
        swing_extreme = float(recent["low"].min())
        stop_price = min(swing_extreme, latest_close - avg_range * 1.2)
        risk = latest_close - stop_price
        target_price = latest_close + max(risk * 2.0, avg_range * 1.8)
    risk = float(max(risk, 0.0))
    ready = latest_close > 0.0 and stop_price > 0.0 and target_price > 0.0 and risk > 0.0
    reward_multiple = abs(target_price - latest_close) / risk if ready else 0.0
    return {
        "template_ready": ready,
        "reason": "live_bars_template_ready" if ready else "non_positive_levels",
        "entry_price": round(latest_close, 8),
        "stop_price": round(stop_price, 8),
        "target_price": round(target_price, 8),
        "risk_per_unit": round(risk, 8),
        "reward_multiple": round(float(reward_multiple), 6),
        "reference_bar_count": int(len(work)),
        "latest_bar_ts": text(latest.get("ts")),
        "avg_range": round(avg_range, 8),
    }


def build_templates_by_symbol(frame: pd.DataFrame, symbols: list[str]) -> dict[str, dict[str, Any]]:
    templates: dict[str, dict[str, Any]] = {}
    for symbol in symbols:
        symbol_frame = sort_symbol_bars(frame, symbol)
        symbol_templates: dict[str, Any] = {}
        if symbol_frame.empty:
            for side in ("LONG", "SHORT"):
                symbol_templates[side] = {
                    "template_ready": False,
                    "reason": "symbol_missing_in_bars_artifact",
                    "entry_price": 0.0,
                    "stop_price": 0.0,
                    "target_price": 0.0,
                    "risk_per_unit": 0.0,
                    "reward_multiple": 0.0,
                    "reference_bar_count": 0,
                }
        else:
            for side in ("LONG", "SHORT"):
                symbol_templates[side] = build_template_from_bars(symbol_frame, side)
        templates[symbol] = symbol_templates
    return templates


def render_markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Crypto Shortline Price Template Snapshot",
            "",
            f"- brief: `{text(payload.get('template_brief'))}`",
            f"- decision: `{text(payload.get('template_decision'))}`",
            f"- side: `{text(payload.get('template_side'))}`",
            f"- ready: `{payload.get('template_ready')}`",
            f"- entry: `{payload.get('entry_price')}`",
            f"- stop: `{payload.get('stop_price')}`",
            f"- target: `{payload.get('target_price')}`",
            f"- blocker: `{text(payload.get('blocker_detail'))}`",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a source-owned crypto shortline price template snapshot from live bars."
    )
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--now", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    review_dir = Path(args.review_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    reference_now = parse_now(args.now)

    intent_path = find_latest(review_dir, "*_remote_intent_queue.json")
    operator_path = find_latest(review_dir, "*_crypto_route_operator_brief.json")
    execution_gate_path = find_latest(review_dir, "*_crypto_shortline_execution_gate.json")
    live_bars_snapshot_path = find_latest(review_dir, "*_crypto_shortline_live_bars_snapshot.json")
    if operator_path is None or execution_gate_path is None:
        raise SystemExit("missing_required_artifacts:crypto_route_operator_brief_or_execution_gate")

    intent_payload = load_json_mapping(intent_path) if intent_path and intent_path.exists() else {}
    operator_payload = load_json_mapping(operator_path)
    execution_gate_payload = load_json_mapping(execution_gate_path)
    live_bars_snapshot_payload = (
        load_json_mapping(live_bars_snapshot_path)
        if live_bars_snapshot_path and live_bars_snapshot_path.exists()
        else {}
    )

    symbol = route_symbol(intent_payload, operator_payload)
    action = route_action(intent_payload, operator_payload)
    remote_market = text(intent_payload.get("remote_market")) or "portfolio_margin_um"
    gate_row = find_gate_row(execution_gate_payload, symbol)
    side = derive_side(gate_row, live_bars_snapshot_payload)
    bars_path = Path(text(live_bars_snapshot_payload.get("bars_artifact"))).expanduser()
    symbols = candidate_symbols(symbol, execution_gate_payload, live_bars_snapshot_payload)

    template_status = "price_template_snapshot_missing_live_bars"
    template_decision = "collect_live_bars_snapshot_before_building_price_template"
    blocker_title = "Collect live bars snapshot before building executable price template"
    blocker_detail = ""
    template_ready = False
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    risk_per_unit = 0.0
    reward_multiple = 0.0
    reference_bar_count = 0
    latest_bar_ts = ""
    avg_range = 0.0
    templates_by_symbol: dict[str, dict[str, Any]] = {}

    if live_bars_snapshot_payload and text(live_bars_snapshot_payload.get("snapshot_status")) == "bars_live_snapshot_ready" and bars_path.exists():
        try:
            frame = read_bars_file(bars_path)
            templates_by_symbol = build_templates_by_symbol(frame, symbols)
            template = as_dict(as_dict(templates_by_symbol.get(symbol)).get(side))
        except Exception as exc:
            template = {"template_ready": False, "reason": f"bars_read_error:{exc}"}
        template_ready = bool(template.get("template_ready"))
        entry_price = safe_float(template.get("entry_price"), 0.0)
        stop_price = safe_float(template.get("stop_price"), 0.0)
        target_price = safe_float(template.get("target_price"), 0.0)
        risk_per_unit = safe_float(template.get("risk_per_unit"), 0.0)
        reward_multiple = safe_float(template.get("reward_multiple"), 0.0)
        reference_bar_count = int(template.get("reference_bar_count") or 0)
        latest_bar_ts = text(template.get("latest_bar_ts"))
        avg_range = safe_float(template.get("avg_range"), 0.0)
        if template_ready:
            template_status = "price_template_snapshot_ready_live_bars"
            template_decision = "refresh_signal_source_with_live_price_template"
            blocker_title = "Use live bars price template before shortline setup promotion"
        else:
            template_status = "price_template_snapshot_blocked_live_bars"
            template_decision = "repair_live_price_template_inputs_then_refresh_signal_source"
            blocker_title = "Repair live bars price template before shortline setup promotion"
        blocker_detail = join_unique(
            [
                f"template_reason={text(template.get('reason')) or '-'}",
                f"bars_artifact={bars_path.name}",
                f"reference_bar_count={reference_bar_count}",
                f"side={side}",
                f"entry_price={entry_price:g}",
                f"stop_price={stop_price:g}",
                f"target_price={target_price:g}",
                f"reward_multiple={reward_multiple:g}",
                f"avg_range={avg_range:g}",
            ]
        )
    else:
        blocker_detail = join_unique(
            [
                f"live_bars_snapshot_status={text(live_bars_snapshot_payload.get('snapshot_status')) or 'missing'}",
                f"bars_artifact={bars_path.name if bars_path else '-'}",
            ]
        )

    template_brief = ":".join([template_status, symbol or "-", template_decision, remote_market or "-"])
    payload = {
        "action": "build_crypto_shortline_price_template_snapshot",
        "ok": True,
        "status": "ok",
        "generated_at_utc": fmt_utc(reference_now),
        "route_symbol": symbol,
        "route_action": action,
        "remote_market": remote_market,
        "template_status": template_status,
        "template_brief": template_brief,
        "template_decision": template_decision,
        "template_side": side,
        "template_ready": template_ready,
        "template_symbols": symbols,
        "template_symbol_count": len(symbols),
        "templates": templates_by_symbol,
        "price_reference_kind": "shortline_live_bars_template",
        "price_reference_source": "",
        "execution_price_ready": template_ready,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "risk_per_unit": risk_per_unit,
        "reward_multiple": reward_multiple,
        "reference_bar_count": reference_bar_count,
        "latest_bar_ts": latest_bar_ts,
        "avg_range": avg_range,
        "blocker_title": blocker_title,
        "blocker_target_artifact": "crypto_shortline_price_template_snapshot",
        "blocker_detail": blocker_detail,
        "next_action": template_decision,
        "next_action_target_artifact": "crypto_shortline_price_template_snapshot",
        "done_when": f"{symbol} has executable entry/stop/target from current live bars snapshot",
        "artifacts": {
            "remote_intent_queue": str(intent_path) if intent_path else "",
            "crypto_route_operator_brief": str(operator_path),
            "crypto_shortline_execution_gate": str(execution_gate_path),
            "crypto_shortline_live_bars_snapshot": str(live_bars_snapshot_path)
            if live_bars_snapshot_path
            else "",
            "bars_live_snapshot": str(bars_path) if bars_path else "",
        },
    }

    stamp = reference_now.strftime("%Y%m%dT%H%M%SZ")
    artifact = review_dir / f"{stamp}_crypto_shortline_price_template_snapshot.json"
    markdown = review_dir / f"{stamp}_crypto_shortline_price_template_snapshot.md"
    checksum = review_dir / f"{stamp}_crypto_shortline_price_template_snapshot_checksum.json"
    payload["price_reference_source"] = str(artifact)
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
