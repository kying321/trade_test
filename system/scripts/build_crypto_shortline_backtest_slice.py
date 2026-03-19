#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml


def resolve_system_root() -> Path:
    env_root = str(os.getenv("LIE_SYSTEM_ROOT", "")).strip() or str(
        os.getenv("FENLIE_SYSTEM_ROOT", "")
    ).strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


SYSTEM_ROOT = resolve_system_root()
DEFAULT_REVIEW_DIR = SYSTEM_ROOT / "output" / "review"
DEFAULT_CONFIG_PATH = SYSTEM_ROOT / "config.yaml"
DEFAULT_SUPPORTED_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT")
DEFAULT_TRIGGER_STACK = (
    "4h_profile_location",
    "liquidity_sweep",
    "1m_5m_mss_or_choch",
    "15m_cvd_divergence_or_confirmation",
    "fvg_ob_breaker_retest",
    "15m_reversal_or_breakout_candle",
)
DEFAULT_SESSION_MAP = (
    "asia_high_low",
    "london_high_low",
    "prior_day_high_low",
    "equal_highs_lows",
)


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
        "*_crypto_shortline_backtest_slice.json",
        "*_crypto_shortline_backtest_slice.md",
        "*_crypto_shortline_backtest_slice_checksum.json",
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


def load_shortline_policy(config_path: Path) -> dict[str, Any]:
    payload: Any = {}
    if config_path.exists():
        try:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    shortline = payload.get("shortline", {}) if isinstance(payload, dict) else {}
    holding_window = as_dict(shortline.get("holding_window_minutes"))
    return {
        "structure_timeframe": text(shortline.get("structure_timeframe")) or "4h",
        "execution_timeframe": text(shortline.get("execution_timeframe")) or "15m",
        "micro_structure_timeframes": dedupe_text(
            as_list(shortline.get("micro_structure_timeframes"))
        )
        or list(("1m", "5m")),
        "holding_window_minutes": {
            "min": int(holding_window.get("min") or 15),
            "max": int(holding_window.get("max") or 180),
        },
        "trigger_stack": dedupe_text(as_list(shortline.get("trigger_stack")))
        or list(DEFAULT_TRIGGER_STACK),
        "session_liquidity_map": dedupe_text(as_list(shortline.get("session_liquidity_map")))
        or list(DEFAULT_SESSION_MAP),
        "supported_symbols": dedupe_text(as_list(shortline.get("supported_symbols")))
        or list(DEFAULT_SUPPORTED_SYMBOLS),
        "no_trade_rule": text(shortline.get("no_trade_rule")) or "no_sweep_no_mss_no_cvd_no_trade",
    }


def find_gate_row(gate_payload: dict[str, Any], symbol: str) -> dict[str, Any]:
    for row in as_list(gate_payload.get("symbols")):
        item = as_dict(row)
        if text(item.get("symbol")).upper() == symbol.upper():
            return item
    return {}


def find_queue_batch(queue_payload: dict[str, Any], batch_name: str, symbol: str) -> dict[str, Any]:
    for key in ("runtime_queue", "batch_runtime_profiles"):
        for row in as_list(queue_payload.get(key)):
            item = as_dict(row)
            if batch_name and text(item.get("batch")) == batch_name:
                return item
            if symbol.upper() in {text(x).upper() for x in as_list(item.get("eligible_symbols"))}:
                return item
    return {}


def find_matching_symbol(queue_batch: dict[str, Any], symbol: str) -> dict[str, Any]:
    for row in as_list(queue_batch.get("matching_symbols")):
        item = as_dict(row)
        if text(item.get("symbol")).upper() == symbol.upper():
            return item
    return {}


def build_universe(
    *,
    route_symbol: str,
    supported_symbols: list[str],
    gate_payload: dict[str, Any],
    queue_batch: dict[str, Any],
) -> list[str]:
    gate_symbols = [text(as_dict(row).get("symbol")).upper() for row in as_list(gate_payload.get("symbols"))]
    eligible_symbols = [text(x).upper() for x in as_list(queue_batch.get("eligible_symbols"))]
    ordered: list[str] = []
    if route_symbol:
        ordered.append(route_symbol.upper())
    for symbol in supported_symbols:
        symbol_upper = text(symbol).upper()
        if symbol_upper == route_symbol.upper():
            continue
        if gate_symbols and symbol_upper not in gate_symbols:
            continue
        if eligible_symbols and symbol_upper not in eligible_symbols:
            continue
        ordered.append(symbol_upper)
    if len(ordered) < 2:
        for symbol_upper in gate_symbols:
            if symbol_upper in ordered:
                continue
            ordered.append(symbol_upper)
    return dedupe_text(ordered)[:4]


def comparison_row(
    *,
    symbol: str,
    gate_row: dict[str, Any],
    queue_batch: dict[str, Any],
) -> dict[str, Any]:
    batch_match = find_matching_symbol(queue_batch, symbol)
    missing_gates = dedupe_text(as_list(gate_row.get("missing_gates")))
    micro_signals = as_dict(gate_row.get("micro_signals"))
    return {
        "symbol": symbol,
        "execution_state": text(gate_row.get("execution_state")),
        "route_action": text(gate_row.get("route_action")),
        "classification": text(batch_match.get("classification")) or text(gate_row.get("execution_state")),
        "micro_context": text(batch_match.get("cvd_context_mode")) or text(micro_signals.get("context")),
        "micro_veto": text(batch_match.get("cvd_veto_hint")) or ",".join(
            dedupe_text(as_list(gate_row.get("focus_execution_micro_reasons")))
        ),
        "missing_gates": missing_gates,
    }


def build_payload(
    *,
    crypto_route_operator_path: Path,
    crypto_route_operator_payload: dict[str, Any],
    shortline_gate_path: Path,
    shortline_gate_payload: dict[str, Any],
    cvd_queue_path: Path,
    cvd_queue_payload: dict[str, Any],
    config_path: Path,
    policy: dict[str, Any],
    reference_now: dt.datetime,
) -> dict[str, Any]:
    selected_symbol = (
        text(crypto_route_operator_payload.get("review_priority_head_symbol"))
        or text(crypto_route_operator_payload.get("next_focus_symbol"))
        or text(crypto_route_operator_payload.get("focus_review_brief")).split(":")[1]
        if ":" in text(crypto_route_operator_payload.get("focus_review_brief"))
        else ""
    ).upper()
    selected_action = text(crypto_route_operator_payload.get("next_focus_action"))
    selected_gate_row = find_gate_row(shortline_gate_payload, selected_symbol)
    selected_batch_name = text(cvd_queue_payload.get("next_focus_batch"))
    selected_batch = find_queue_batch(cvd_queue_payload, selected_batch_name, selected_symbol)
    selected_batch_match = find_matching_symbol(selected_batch, selected_symbol)
    slice_universe = build_universe(
        route_symbol=selected_symbol,
        supported_symbols=policy.get("supported_symbols", []),
        gate_payload=shortline_gate_payload,
        queue_batch=selected_batch,
    )

    comparison_rows: list[dict[str, Any]] = []
    for symbol in slice_universe:
        comparison_rows.append(
            comparison_row(
                symbol=symbol,
                gate_row=find_gate_row(shortline_gate_payload, symbol),
                queue_batch=selected_batch,
            )
        )

    execution_state = text(selected_gate_row.get("execution_state")) or text(
        selected_batch_match.get("classification")
    )
    if execution_state == "Setup_Ready":
        slice_status = "selected_setup_ready_orderflow_slice"
        research_decision = "run_setup_ready_orderflow_cross_section_backtest"
    elif selected_symbol and selected_gate_row:
        slice_status = "selected_watch_only_orderflow_slice"
        research_decision = "run_watch_only_orderflow_cross_section_backtest"
    else:
        slice_status = "shortline_slice_unavailable"
        research_decision = "inspect_shortline_sources"

    slice_brief = ":".join(
        [
            slice_status,
            selected_symbol or "-",
            selected_batch_name or "-",
            f"{policy.get('structure_timeframe')}->{policy.get('execution_timeframe')}",
        ]
    )
    missing_gates = dedupe_text(as_list(selected_gate_row.get("missing_gates")))
    blocker_detail = join_unique(
        [
            text(crypto_route_operator_payload.get("focus_review_blocker_detail")),
            (
                f"queue={text(cvd_queue_payload.get('queue_status'))}:{selected_batch_name}:{text(cvd_queue_payload.get('next_focus_action'))}"
                if text(cvd_queue_payload.get("queue_status")) or selected_batch_name
                else ""
            ),
            (
                f"execution_state={execution_state}; missing_gates={','.join(missing_gates)}"
                if execution_state or missing_gates
                else ""
            ),
        ]
    )
    hypothesis_brief = join_unique(
        [
            "measure whether watch-only shortline symbols can graduate into Setup_Ready before ticket generation becomes necessary",
            (
                f"current_focus={selected_symbol}:{selected_action}"
                if selected_symbol or selected_action
                else ""
            ),
            (
                f"queue_focus={selected_batch_name}:{text(cvd_queue_payload.get('next_focus_action'))}"
                if selected_batch_name or text(cvd_queue_payload.get("next_focus_action"))
                else ""
            ),
        ]
    )
    done_when = (
        f"{selected_symbol or 'route symbol'} has a reusable shortline backtest slice with deterministic trigger-stack inputs "
        "and the comparison universe is ready for event-timestamp-anchored replay"
    )
    return {
        "action": "build_crypto_shortline_backtest_slice",
        "ok": True,
        "status": "ok",
        "generated_at_utc": fmt_utc(reference_now),
        "slice_status": slice_status,
        "slice_brief": slice_brief,
        "research_decision": research_decision,
        "selected_symbol": selected_symbol,
        "selected_action": selected_action,
        "selected_execution_state": execution_state,
        "selected_micro_context": text(selected_batch_match.get("cvd_context_mode"))
        or text(as_dict(selected_gate_row.get("micro_signals")).get("context")),
        "selected_micro_veto": text(selected_batch_match.get("cvd_veto_hint"))
        or text(crypto_route_operator_payload.get("focus_execution_micro_veto")),
        "selected_focus_batch": selected_batch_name,
        "selected_focus_action": text(cvd_queue_payload.get("next_focus_action")),
        "selected_queue_stack_brief": text(cvd_queue_payload.get("queue_stack_brief")),
        "market_state_brief": text(crypto_route_operator_payload.get("shortline_market_state_brief")),
        "execution_gate_brief": text(crypto_route_operator_payload.get("shortline_execution_gate_brief")),
        "no_trade_rule": text(policy.get("no_trade_rule")),
        "structure_timeframe": text(policy.get("structure_timeframe")),
        "execution_timeframe": text(policy.get("execution_timeframe")),
        "micro_structure_timeframes": list(policy.get("micro_structure_timeframes", [])),
        "holding_window_minutes": as_dict(policy.get("holding_window_minutes")),
        "trigger_stack": list(policy.get("trigger_stack", [])),
        "session_liquidity_map": list(policy.get("session_liquidity_map", [])),
        "slice_universe": slice_universe,
        "slice_universe_count": len(slice_universe),
        "slice_universe_brief": ",".join(slice_universe),
        "comparison_rows": comparison_rows,
        "focus_review_status": text(crypto_route_operator_payload.get("focus_review_status")),
        "focus_review_brief": text(crypto_route_operator_payload.get("focus_review_brief")),
        "focus_review_blocker_detail": text(
            crypto_route_operator_payload.get("focus_review_blocker_detail")
        ),
        "focus_review_done_when": text(
            crypto_route_operator_payload.get("focus_review_done_when")
        ),
        "focus_review_priority_score": crypto_route_operator_payload.get(
            "focus_review_priority_score"
        ),
        "queue_status": text(cvd_queue_payload.get("queue_status")),
        "queue_takeaway": text(cvd_queue_payload.get("takeaway")),
        "semantic_status": text(cvd_queue_payload.get("semantic_status")),
        "semantic_takeaway": text(cvd_queue_payload.get("semantic_takeaway")),
        "hypothesis_brief": hypothesis_brief,
        "blocker_detail": blocker_detail,
        "done_when": done_when,
        "artifacts": {
            "crypto_route_operator_brief": str(crypto_route_operator_path),
            "crypto_shortline_execution_gate": str(shortline_gate_path),
            "crypto_cvd_queue_handoff": str(cvd_queue_path),
            "config": str(config_path),
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Crypto Shortline Backtest Slice",
            "",
            f"- brief: `{text(payload.get('slice_brief'))}`",
            f"- research_decision: `{text(payload.get('research_decision'))}`",
            f"- selected_symbol: `{text(payload.get('selected_symbol'))}`",
            f"- focus_batch: `{text(payload.get('selected_focus_batch'))}`",
            f"- focus_action: `{text(payload.get('selected_focus_action'))}`",
            f"- universe: `{text(payload.get('slice_universe_brief'))}`",
            f"- trigger_stack: `{', '.join(as_list(payload.get('trigger_stack')))}`",
            f"- blocker: `{text(payload.get('blocker_detail'))}`",
            f"- done_when: `{text(payload.get('done_when'))}`",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build crypto shortline backtest slice artifact.")
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--now", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config).expanduser().resolve()
    reference_now = parse_now(args.now)

    crypto_route_operator_path = find_latest(review_dir, "*_crypto_route_operator_brief.json")
    shortline_gate_path = find_latest(review_dir, "*_crypto_shortline_execution_gate.json")
    cvd_queue_path = find_latest(review_dir, "*_crypto_cvd_queue_handoff.json")
    missing = [
        name
        for name, path in (
            ("crypto_route_operator_brief", crypto_route_operator_path),
            ("crypto_shortline_execution_gate", shortline_gate_path),
            ("crypto_cvd_queue_handoff", cvd_queue_path),
        )
        if path is None
    ]
    if missing:
        raise SystemExit(f"missing_required_artifacts:{','.join(missing)}")

    payload = build_payload(
        crypto_route_operator_path=crypto_route_operator_path,
        crypto_route_operator_payload=load_json_mapping(crypto_route_operator_path),
        shortline_gate_path=shortline_gate_path,
        shortline_gate_payload=load_json_mapping(shortline_gate_path),
        cvd_queue_path=cvd_queue_path,
        cvd_queue_payload=load_json_mapping(cvd_queue_path),
        config_path=config_path,
        policy=load_shortline_policy(config_path),
        reference_now=reference_now,
    )

    stamp = reference_now.strftime("%Y%m%dT%H%M%SZ")
    artifact = review_dir / f"{stamp}_crypto_shortline_backtest_slice.json"
    markdown = review_dir / f"{stamp}_crypto_shortline_backtest_slice.md"
    checksum = review_dir / f"{stamp}_crypto_shortline_backtest_slice_checksum.json"
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
