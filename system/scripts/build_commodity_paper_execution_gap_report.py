#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

import yaml


TIMESTAMP_RE = re.compile(r"^(?P<stamp>\d{8}T\d{6}Z)_")
FUTURE_STAMP_GRACE_MINUTES = 5


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


def latest_review_artifact(review_dir: Path, suffix: str, reference_now: dt.datetime | None = None) -> Path | None:
    candidates = list(review_dir.glob(f"*_{suffix}.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _list_text(values: list[str], limit: int = 8) -> str:
    items = [str(v).strip() for v in values if str(v).strip()]
    if not items:
        return "-"
    if len(items) <= limit:
        return ", ".join(items)
    return ", ".join(items[:limit]) + f" (+{len(items) - limit})"


def _mapping_text(values: dict[str, Any], limit: int = 8) -> str:
    items = [
        f"{str(key).strip().upper()}:{str(value).strip()}"
        for key, value in sorted((values or {}).items())
        if str(key).strip() and str(value).strip()
    ]
    if not items:
        return "-"
    if len(items) <= limit:
        return ", ".join(items)
    return ", ".join(items[:limit]) + f" (+{len(items) - limit})"


def _safe_int(raw: Any) -> int | None:
    try:
        value = int(raw)
    except Exception:
        return None
    return int(value)


def _watch_item_sort_key(item: dict[str, Any]) -> tuple[int, str, str]:
    age = _safe_int(item.get("signal_age_days"))
    symbol = str(item.get("symbol") or "").strip().upper()
    signal_date = str(item.get("signal_date") or "").strip()
    return (-(age if age is not None else -1), signal_date, symbol)


def _watch_items_text(items: list[dict[str, Any]], limit: int = 8) -> str:
    parts: list[str] = []
    for row in items[:limit]:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        signal_date = str(row.get("signal_date") or "").strip()
        age = _safe_int(row.get("signal_age_days"))
        if not symbol:
            continue
        detail = symbol
        if age is not None:
            detail += f":{age}d"
        if signal_date:
            detail += f"@{signal_date}"
        parts.append(detail)
    if not parts:
        return "-"
    if len(items) <= limit:
        return ", ".join(parts)
    return ", ".join(parts) + f" (+{len(items) - limit})"


def resolve_optional_path(raw: str | None, default_path: Path) -> Path | None:
    text = str(raw or "").strip()
    path = Path(text).expanduser().resolve() if text else default_path
    return path if path.exists() else None


def resolve_explicit_path(raw: str | None) -> Path | None:
    text = str(raw or "").strip()
    if not text:
        return None
    path = Path(text).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def default_config_path(review_dir: Path) -> Path:
    return review_dir.parents[1] / "config.yaml"


def default_paper_execution_ledger_path(review_dir: Path) -> Path:
    return review_dir.parent / "logs" / "paper_execution_ledger.jsonl"


def default_paper_positions_path(review_dir: Path) -> Path:
    return review_dir.parent / "artifacts" / "paper_positions_open.json"


def default_paper_db_path(review_dir: Path) -> Path:
    return review_dir.parent / "artifacts" / "lie_engine.db"


def load_json_mapping(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def load_config_core_symbols(path: Path | None) -> tuple[list[str], list[str]]:
    if path is None:
        return [], []
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        return [], []
    universe = payload.get("universe", {})
    if not isinstance(universe, dict):
        return [], []
    core_rows = universe.get("core", [])
    if not isinstance(core_rows, list):
        core_rows = []
    domestic_rows = universe.get("domestic_futures_paper", [])
    if not isinstance(domestic_rows, list):
        domestic_rows = []
    symbols: list[str] = []
    asset_classes: list[str] = []
    for row in [*core_rows, *domestic_rows]:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        asset_class = str(row.get("asset_class") or "").strip().lower()
        if symbol:
            symbols.append(symbol)
        if asset_class:
            asset_classes.append(asset_class)
    return symbols, asset_classes


def load_ledger_symbol_counts(path: Path | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    if path is None:
        return counts
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            if str(payload.get("domain") or "").strip() != "paper_execution":
                continue
            symbol = str(payload.get("symbol") or "").strip().upper()
            if symbol:
                counts[symbol] = counts.get(symbol, 0) + 1
    return counts


def load_open_position_symbol_counts(path: Path | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    payload = load_json_mapping(path)
    positions = payload.get("positions", []) if isinstance(payload.get("positions"), list) else []
    for row in positions:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        if symbol:
            counts[symbol] = counts.get(symbol, 0) + 1
    return counts


def load_sqlite_symbol_counts(path: Path | None, table_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    if path is None:
        return counts
    try:
        with sqlite3.connect(path) as conn:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
                (table_name,),
            ).fetchone()
            if not exists:
                return counts
            rows = conn.execute(
                f"SELECT symbol, COUNT(*) AS cnt FROM {table_name} GROUP BY symbol"
            ).fetchall()
    except sqlite3.Error:
        return counts
    for symbol, count in rows:
        symbol_text = str(symbol or "").strip().upper()
        if symbol_text:
            counts[symbol_text] = int(count or 0)
    return counts


def symbols_subset(values: dict[str, int], symbols: list[str]) -> dict[str, int]:
    return {symbol: int(values.get(symbol, 0) or 0) for symbol in symbols}


def derive_gap_report(
    *,
    execution_queue: dict[str, Any],
    execution_review: dict[str, Any],
    execution_retro: dict[str, Any],
    execution_bridge: dict[str, Any],
    config_core_symbols: list[str],
    config_asset_classes: list[str],
    trade_plan_counts: dict[str, int],
    executed_plan_counts: dict[str, int],
    ledger_counts: dict[str, int],
    open_position_counts: dict[str, int],
) -> dict[str, Any]:
    execution_symbols = [str(x).strip().upper() for x in execution_queue.get("execution_symbols", []) if str(x).strip()]
    next_execution_id = str(execution_queue.get("next_execution_id") or "").strip()
    next_execution_symbol = str(execution_queue.get("next_execution_symbol") or "").strip().upper()
    execution_batch = str(execution_queue.get("execution_batch") or "").strip()

    missing_from_core = [symbol for symbol in execution_symbols if symbol not in set(config_core_symbols)]
    trade_plan_subset = symbols_subset(trade_plan_counts, execution_symbols)
    executed_plan_subset = symbols_subset(executed_plan_counts, execution_symbols)
    ledger_subset = symbols_subset(ledger_counts, execution_symbols)
    open_position_subset = symbols_subset(open_position_counts, execution_symbols)
    queue_symbols_with_trade_plans = [symbol for symbol in execution_symbols if int(trade_plan_subset.get(symbol, 0) or 0) > 0]
    queue_symbols_without_trade_plans = [
        symbol for symbol in execution_symbols if int(trade_plan_subset.get(symbol, 0) or 0) <= 0
    ]

    queue_symbols_with_any_evidence: list[str] = []
    queue_symbols_without_any_evidence: list[str] = []
    for symbol in execution_symbols:
        has_evidence = bool(
            trade_plan_subset.get(symbol, 0)
            or executed_plan_subset.get(symbol, 0)
            or ledger_subset.get(symbol, 0)
            or open_position_subset.get(symbol, 0)
        )
        if has_evidence:
            queue_symbols_with_any_evidence.append(symbol)
        else:
            queue_symbols_without_any_evidence.append(symbol)

    bridge_status = str(execution_bridge.get("bridge_status") or "").strip()
    bridge_items = [row for row in execution_bridge.get("bridge_items", []) if isinstance(row, dict)]
    blocked_bridge_statuses = {
        "blocked_missing_directional_signal",
        "blocked_stale_directional_signal",
        "blocked_proxy_price_reference_only",
        "bridge_partially_bridged_missing_remainder",
        "bridge_partially_bridged_stale_remainder",
        "bridge_partially_bridged_proxy_remainder",
    }
    queue_symbols_already_bridged = [
        str(row.get("symbol") or "").strip().upper()
        for row in bridge_items
        if str(row.get("bridge_status") or "").strip() == "already_bridged"
        and str(row.get("symbol") or "").strip()
    ]
    queue_symbols_with_proxy_price_reference_only = [
        str(row.get("symbol") or "").strip().upper()
        for row in bridge_items
        if str(row.get("bridge_status") or "").strip() in blocked_bridge_statuses
        and "proxy_price_reference_only" in [str(x).strip() for x in row.get("bridge_reasons", []) if str(x).strip()]
        and str(row.get("symbol") or "").strip()
    ]
    queue_symbols_with_stale_directional_signal = [
        str(row.get("symbol") or "").strip().upper()
        for row in bridge_items
        if str(row.get("bridge_status") or "").strip() in blocked_bridge_statuses
        and "stale_signal" in [str(x).strip() for x in row.get("bridge_reasons", []) if str(x).strip()]
        and str(row.get("symbol") or "").strip()
    ]
    queue_symbols_with_stale_directional_signal_dates = {
        str(row.get("symbol") or "").strip().upper(): str(row.get("signal_date") or "").strip()
        for row in bridge_items
        if str(row.get("bridge_status") or "").strip() in blocked_bridge_statuses
        and "stale_signal" in [str(x).strip() for x in row.get("bridge_reasons", []) if str(x).strip()]
        and str(row.get("symbol") or "").strip()
        and str(row.get("signal_date") or "").strip()
    }
    queue_symbols_with_stale_directional_signal_age_days = {
        str(row.get("symbol") or "").strip().upper(): int(_safe_int(row.get("signal_age_days")) or 0)
        for row in bridge_items
        if str(row.get("bridge_status") or "").strip() in blocked_bridge_statuses
        and "stale_signal" in [str(x).strip() for x in row.get("bridge_reasons", []) if str(x).strip()]
        and str(row.get("symbol") or "").strip()
        and _safe_int(row.get("signal_age_days")) is not None
    }
    stale_directional_signal_watch_items = sorted(
        [
            {
                "execution_id": str(row.get("execution_id") or "").strip(),
                "symbol": str(row.get("symbol") or "").strip().upper(),
                "signal_date": str(row.get("signal_date") or "").strip(),
                "signal_age_days": _safe_int(row.get("signal_age_days")),
            }
            for row in bridge_items
            if str(row.get("bridge_status") or "").strip() in blocked_bridge_statuses
            and "stale_signal" in [str(x).strip() for x in row.get("bridge_reasons", []) if str(x).strip()]
            and str(row.get("symbol") or "").strip()
        ],
        key=_watch_item_sort_key,
    )
    queue_symbols_missing_directional_signal = [
        str(row.get("symbol") or "").strip().upper()
        for row in bridge_items
        if str(row.get("bridge_status") or "").strip() in blocked_bridge_statuses
        and "stale_signal" not in [str(x).strip() for x in row.get("bridge_reasons", []) if str(x).strip()]
        and str(row.get("symbol") or "").strip()
    ]

    reason_codes: list[str] = []
    if missing_from_core:
        reason_codes.append("queue_symbols_missing_from_core_universe")
    if config_core_symbols and all(asset_class == "crypto" for asset_class in config_asset_classes):
        reason_codes.append("core_universe_crypto_only")
    if queue_symbols_with_proxy_price_reference_only:
        reason_codes.append("queue_symbols_with_proxy_price_reference_only")
    if queue_symbols_with_stale_directional_signal:
        reason_codes.append("queue_symbols_with_stale_directional_signal")
    if queue_symbols_missing_directional_signal:
        reason_codes.append("queue_symbols_missing_directional_signal")
    if queue_symbols_without_trade_plans:
        reason_codes.append("queue_symbols_missing_from_trade_plans")
    if queue_symbols_without_any_evidence:
        reason_codes.append("queue_symbols_missing_from_paper_execution_evidence")

    gap_active = bool(reason_codes)
    root_cause_lines: list[str] = []
    if missing_from_core:
        root_cause_lines.append(
            "Queue symbols are absent from config core universe: "
            + _list_text(missing_from_core)
            + "."
        )
    if config_core_symbols and all(asset_class == "crypto" for asset_class in config_asset_classes):
        root_cause_lines.append(
            "Config core universe remains crypto-only, so commodity queue symbols are outside the default paper runtime path."
        )
    if queue_symbols_with_proxy_price_reference_only:
        root_cause_lines.append(
            "Commodity directional tickets still use proxy-market prices rather than executable instrument prices for queue symbols: "
            + _list_text(queue_symbols_with_proxy_price_reference_only)
            + "."
        )
    if queue_symbols_already_bridged:
        root_cause_lines.append(
            "Queue symbols already bridged into paper execution evidence: "
            + _list_text(queue_symbols_already_bridged)
            + "."
        )
    if queue_symbols_with_stale_directional_signal:
        root_cause_lines.append(
            "Fresh directional combo triggers are absent; the latest bridgeable commodity signals are stale for queue symbols: "
            + _list_text(queue_symbols_with_stale_directional_signal)
            + "."
        )
    if queue_symbols_with_stale_directional_signal_dates:
        root_cause_lines.append(
            "Latest stale directional trigger dates by symbol: "
            + _mapping_text(queue_symbols_with_stale_directional_signal_dates)
            + "."
        )
    if queue_symbols_with_stale_directional_signal_age_days:
        root_cause_lines.append(
            "Latest stale directional trigger ages in days by symbol: "
            + _mapping_text(queue_symbols_with_stale_directional_signal_age_days)
            + "."
        )
    if queue_symbols_missing_directional_signal:
        root_cause_lines.append(
            "Fresh signal-to-order tickets still return signal_not_found for queue symbols, so the bridge cannot derive side/levels."
        )
    if queue_symbols_without_trade_plans:
        if len(queue_symbols_without_trade_plans) == len(execution_symbols):
            root_cause_lines.append(
                "No queue symbols appear in sqlite trade_plans, so no paper execution handoff reached the engine database."
            )
        else:
            root_cause_lines.append(
                "Queue symbols still missing from sqlite trade_plans: "
                + _list_text(queue_symbols_without_trade_plans)
                + "."
            )
    if queue_symbols_without_any_evidence:
        if len(queue_symbols_without_any_evidence) == len(execution_symbols):
            root_cause_lines.append(
                "No queue symbols appear in paper execution ledger, open paper positions, or executed_plans."
            )
        else:
            root_cause_lines.append(
                "Queue symbols still missing from paper execution ledger, open paper positions, or executed_plans: "
                + _list_text(queue_symbols_without_any_evidence)
                + "."
            )
    if not root_cause_lines:
        root_cause_lines.append("Queue symbols are present in both config universe and paper execution evidence sources.")

    recommended_actions = [
        (
            "Apply the ready commodity bridge items into paper execution evidence, then reassess the remaining stale queue symbols."
            if bridge_status == "bridge_ready"
            else (
                "Continue paper review/retro for already bridged symbols while the remaining queue symbols stay blocked."
                if bridge_status
                in {
                    "bridge_partially_bridged_missing_remainder",
                    "bridge_partially_bridged_stale_remainder",
                    "bridge_partially_bridged_proxy_remainder",
                }
                else (
                    "Keep commodity queue in paper-only bridge review mode until config/runtime coverage and execution evidence are both present."
                    if bridge_status
                    else "Keep commodity queue in research/paper-planning mode until a real paper execution bridge exists."
                )
            )
        )
    ]
    if queue_symbols_with_proxy_price_reference_only:
        recommended_actions.append(
            "Add a commodity price-normalization path or retarget the queue to proxy instruments before allowing bridge apply."
        )
    recommended_actions.extend(
        [
            "Decide explicitly whether to extend config/runtime universe to commodity symbols or keep commodity flow report-only.",
            "Only promote queue -> review/retro after queue symbols appear in trade_plans or paper execution evidence sources.",
            "Restore or generate directional commodity signals before attempting queue bridge apply.",
        ]
    )
    if queue_symbols_with_stale_directional_signal:
        recommended_actions[-1] = (
            "Keep stale queue symbols in watch until they generate a fresh commodity breakout/reclaim trigger."
            if bridge_status == "bridge_ready"
            else "Wait for or generate a fresh commodity breakout/reclaim trigger before attempting queue bridge apply."
        )

    summary_lines = [
        f"gap-status: {'blocking_gap_active' if gap_active else 'gap_clear'}",
        (
            "decision: do_not_assume_commodity_paper_execution_active"
            if gap_active
            else "decision: commodity_paper_execution_path_present"
        ),
        f"execution-batch: {execution_batch or '-'}",
        f"execution-symbols: {_list_text(execution_symbols)}",
        f"next-execution-id: {next_execution_id or '-'}",
        f"next-execution-symbol: {next_execution_symbol or '-'}",
        f"missing-from-core: {_list_text(missing_from_core)}",
        f"already-bridged-symbols: {_list_text(queue_symbols_already_bridged)}",
        f"symbols-with-trade-plans: {_list_text(queue_symbols_with_trade_plans)}",
        f"symbols-without-trade-plans: {_list_text(queue_symbols_without_trade_plans)}",
        f"symbols-with-evidence: {_list_text(queue_symbols_with_any_evidence)}",
        f"symbols-without-evidence: {_list_text(queue_symbols_without_any_evidence)}",
        f"symbols-with-proxy-price-reference-only: {_list_text(queue_symbols_with_proxy_price_reference_only)}",
        f"symbols-with-stale-directional-signal: {_list_text(queue_symbols_with_stale_directional_signal)}",
        f"stale-directional-signal-dates: {_mapping_text(queue_symbols_with_stale_directional_signal_dates)}",
        f"stale-directional-signal-age-days: {_mapping_text(queue_symbols_with_stale_directional_signal_age_days)}",
        f"stale-directional-watch: {_watch_items_text(stale_directional_signal_watch_items)}",
        f"symbols-missing-directional-signal: {_list_text(queue_symbols_missing_directional_signal)}",
        f"reason-codes: {_list_text(reason_codes)}",
    ]

    return {
        "gap_status": "blocking_gap_active" if gap_active else "gap_clear",
        "current_decision": (
            "do_not_assume_commodity_paper_execution_active"
            if gap_active
            else "commodity_paper_execution_path_present"
        ),
        "execution_batch": execution_batch,
        "execution_symbols": execution_symbols,
        "next_execution_id": next_execution_id,
        "next_execution_symbol": next_execution_symbol,
        "execution_queue_status": str(execution_queue.get("execution_queue_status") or ""),
        "execution_review_status": str(execution_review.get("execution_review_status") or ""),
        "execution_retro_status": str(execution_retro.get("execution_retro_status") or ""),
        "execution_bridge_status": bridge_status,
        "core_universe_symbols": config_core_symbols,
        "core_universe_asset_classes": sorted({x for x in config_asset_classes if x}),
        "queue_symbols_missing_from_core_universe": missing_from_core,
        "queue_symbols_already_bridged": queue_symbols_already_bridged,
        "trade_plan_symbol_counts": trade_plan_subset,
        "queue_symbols_with_trade_plans": queue_symbols_with_trade_plans,
        "queue_symbols_without_trade_plans": queue_symbols_without_trade_plans,
        "executed_plan_symbol_counts": executed_plan_subset,
        "paper_execution_ledger_symbol_counts": ledger_subset,
        "open_position_symbol_counts": open_position_subset,
        "queue_symbols_with_any_evidence": queue_symbols_with_any_evidence,
        "queue_symbols_without_any_evidence": queue_symbols_without_any_evidence,
        "queue_symbols_with_proxy_price_reference_only": queue_symbols_with_proxy_price_reference_only,
        "queue_symbols_with_stale_directional_signal": queue_symbols_with_stale_directional_signal,
        "queue_symbols_with_stale_directional_signal_dates": queue_symbols_with_stale_directional_signal_dates,
        "queue_symbols_with_stale_directional_signal_age_days": queue_symbols_with_stale_directional_signal_age_days,
        "stale_directional_signal_watch_items": stale_directional_signal_watch_items,
        "queue_symbols_missing_directional_signal": queue_symbols_missing_directional_signal,
        "gap_reason_codes": reason_codes,
        "root_cause_lines": root_cause_lines,
        "recommended_actions": recommended_actions,
        "summary_lines": summary_lines,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Commodity Paper Execution Gap Report",
        "",
        f"- as_of: `{payload.get('as_of') or ''}`",
        f"- gap_status: `{payload.get('gap_status') or ''}`",
        f"- current_decision: `{payload.get('current_decision') or ''}`",
        f"- source_execution_queue_artifact: `{payload.get('source_execution_queue_artifact') or ''}`",
        f"- source_execution_review_artifact: `{payload.get('source_execution_review_artifact') or ''}`",
        f"- source_execution_retro_artifact: `{payload.get('source_execution_retro_artifact') or ''}`",
        f"- source_execution_bridge_artifact: `{payload.get('source_execution_bridge_artifact') or ''}`",
        f"- source_config_path: `{payload.get('source_config_path') or ''}`",
        f"- source_paper_execution_ledger_path: `{payload.get('source_paper_execution_ledger_path') or ''}`",
        f"- source_paper_positions_path: `{payload.get('source_paper_positions_path') or ''}`",
        f"- source_paper_db_path: `{payload.get('source_paper_db_path') or ''}`",
        f"- execution_batch: `{payload.get('execution_batch') or '-'}`",
        f"- execution_symbols: `{_list_text(payload.get('execution_symbols', []))}`",
        f"- next_execution_id: `{payload.get('next_execution_id') or '-'}`",
        f"- next_execution_symbol: `{payload.get('next_execution_symbol') or '-'}`",
        f"- execution_bridge_status: `{payload.get('execution_bridge_status') or '-'}`",
        "",
        "## Summary",
    ]
    for line in payload.get("summary_lines", []):
        lines.append(f"- {line}")
    lines.extend(["", "## Root Cause"])
    for line in payload.get("root_cause_lines", []):
        lines.append(f"- {line}")
    lines.extend(["", "## Evidence"])
    lines.append(f"- core_universe_symbols: `{_list_text(payload.get('core_universe_symbols', []), limit=20)}`")
    lines.append(
        f"- queue_symbols_missing_from_core_universe: `{_list_text(payload.get('queue_symbols_missing_from_core_universe', []))}`"
    )
    lines.append(f"- queue_symbols_already_bridged: `{_list_text(payload.get('queue_symbols_already_bridged', []))}`")
    lines.append(f"- queue_symbols_with_trade_plans: `{_list_text(payload.get('queue_symbols_with_trade_plans', []))}`")
    lines.append(
        f"- queue_symbols_without_trade_plans: `{_list_text(payload.get('queue_symbols_without_trade_plans', []))}`"
    )
    lines.append(
        f"- queue_symbols_with_proxy_price_reference_only: `{_list_text(payload.get('queue_symbols_with_proxy_price_reference_only', []))}`"
    )
    lines.append(
        f"- queue_symbols_with_stale_directional_signal: `{_list_text(payload.get('queue_symbols_with_stale_directional_signal', []))}`"
    )
    lines.append(
        f"- queue_symbols_with_stale_directional_signal_dates: `{_mapping_text(payload.get('queue_symbols_with_stale_directional_signal_dates', {}))}`"
    )
    lines.append(
        f"- queue_symbols_with_stale_directional_signal_age_days: `{_mapping_text(payload.get('queue_symbols_with_stale_directional_signal_age_days', {}))}`"
    )
    lines.append(
        f"- stale_directional_signal_watch_items: `{_watch_items_text(payload.get('stale_directional_signal_watch_items', []), limit=20)}`"
    )
    lines.append(
        f"- queue_symbols_missing_directional_signal: `{_list_text(payload.get('queue_symbols_missing_directional_signal', []))}`"
    )
    lines.append(f"- trade_plan_symbol_counts: `{payload.get('trade_plan_symbol_counts') or {}}`")
    lines.append(f"- executed_plan_symbol_counts: `{payload.get('executed_plan_symbol_counts') or {}}`")
    lines.append(f"- paper_execution_ledger_symbol_counts: `{payload.get('paper_execution_ledger_symbol_counts') or {}}`")
    lines.append(f"- open_position_symbol_counts: `{payload.get('open_position_symbol_counts') or {}}`")
    lines.extend(["", "## Recommended Actions"])
    for line in payload.get("recommended_actions", []):
        lines.append(f"- {line}")
    return "\n".join(lines).strip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a commodity paper execution gap report.")
    parser.add_argument("--review-dir", required=True)
    parser.add_argument("--now", default="")
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--execution-queue-json", default="")
    parser.add_argument("--execution-review-json", default="")
    parser.add_argument("--execution-retro-json", default="")
    parser.add_argument("--execution-bridge-json", default="")
    parser.add_argument("--config-path", default="")
    parser.add_argument("--paper-ledger-path", default="")
    parser.add_argument("--paper-positions-path", default="")
    parser.add_argument("--paper-db-path", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    runtime_now = parse_now(args.now)

    execution_queue_path = resolve_explicit_path(args.execution_queue_json) or latest_review_artifact(
        review_dir, "commodity_paper_execution_queue", runtime_now
    )
    if execution_queue_path is None:
        raise FileNotFoundError("no_commodity_paper_execution_queue")
    execution_review_path = resolve_explicit_path(args.execution_review_json) or latest_review_artifact(
        review_dir, "commodity_paper_execution_review", runtime_now
    )
    execution_retro_path = resolve_explicit_path(args.execution_retro_json) or latest_review_artifact(
        review_dir, "commodity_paper_execution_retro", runtime_now
    )
    execution_bridge_path = resolve_explicit_path(args.execution_bridge_json) or latest_review_artifact(
        review_dir, "commodity_paper_execution_bridge", runtime_now
    )

    config_path = resolve_optional_path(args.config_path, default_config_path(review_dir))
    paper_ledger_path = resolve_optional_path(args.paper_ledger_path, default_paper_execution_ledger_path(review_dir))
    paper_positions_path = resolve_optional_path(args.paper_positions_path, default_paper_positions_path(review_dir))
    paper_db_path = resolve_optional_path(args.paper_db_path, default_paper_db_path(review_dir))

    execution_queue = load_json_mapping(execution_queue_path)
    execution_review = load_json_mapping(execution_review_path)
    execution_retro = load_json_mapping(execution_retro_path)
    execution_bridge = load_json_mapping(execution_bridge_path)
    config_core_symbols, config_asset_classes = load_config_core_symbols(config_path)
    gap_report = derive_gap_report(
        execution_queue=execution_queue,
        execution_review=execution_review,
        execution_retro=execution_retro,
        execution_bridge=execution_bridge,
        config_core_symbols=config_core_symbols,
        config_asset_classes=config_asset_classes,
        trade_plan_counts=load_sqlite_symbol_counts(paper_db_path, "trade_plans"),
        executed_plan_counts=load_sqlite_symbol_counts(paper_db_path, "executed_plans"),
        ledger_counts=load_ledger_symbol_counts(paper_ledger_path),
        open_position_counts=load_open_position_symbol_counts(paper_positions_path),
    )

    stamp = runtime_now.strftime("%Y%m%dT%H%M%SZ")
    json_path = review_dir / f"{stamp}_commodity_paper_execution_gap_report.json"
    md_path = review_dir / f"{stamp}_commodity_paper_execution_gap_report.md"
    checksum_path = review_dir / f"{stamp}_commodity_paper_execution_gap_report_checksum.json"
    payload = {
        "ok": True,
        "status": "ok",
        "as_of": fmt_utc(runtime_now),
        "source_execution_queue_artifact": str(execution_queue_path),
        "source_execution_review_artifact": str(execution_review_path) if execution_review_path else "",
        "source_execution_retro_artifact": str(execution_retro_path) if execution_retro_path else "",
        "source_execution_bridge_artifact": str(execution_bridge_path) if execution_bridge_path else "",
        "source_config_path": str(config_path) if config_path else "",
        "source_paper_execution_ledger_path": str(paper_ledger_path) if paper_ledger_path else "",
        "source_paper_positions_path": str(paper_positions_path) if paper_positions_path else "",
        "source_paper_db_path": str(paper_db_path) if paper_db_path else "",
        **gap_report,
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
        stem="commodity_paper_execution_gap_report",
        current_paths=[json_path, md_path, checksum_path],
        keep=max(3, int(args.artifact_keep)),
        ttl_hours=float(args.artifact_ttl_hours),
        now_dt=runtime_now,
    )
    payload.update(
        {
            "artifact": str(json_path),
            "markdown": str(md_path),
            "checksum": str(checksum_path),
            "pruned_keep": pruned_keep,
            "pruned_age": pruned_age,
        }
    )
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
