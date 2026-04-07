from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

import akshare as ak
import pandas as pd

from lie_engine.data.providers import PublicInternetResearchProvider
from lie_engine.research.fuel_oil_2607_input_packet import build_fuel_oil_2607_input_packet
from lie_engine.research.fuel_oil_2607_scenario import build_fuel_oil_2607_scenario_tree
from lie_engine.research.fuel_oil_2607_summary import build_fuel_oil_2607_summary
from lie_engine.research.fuel_oil_2607_trade_space import build_fuel_oil_2607_trade_space
from lie_engine.research.fuel_oil_2607_transmission import build_fuel_oil_2607_transmission_map
from lie_engine.research.fuel_oil_2607_strategy import build_fuel_oil_2607_strategy_matrix
from lie_engine.research.fuel_oil_2607_validation import build_fuel_oil_2607_validation_ring


DEFAULT_REPORT_PATH = Path("/Users/jokenrobot/Downloads/燃油期货2607分析与交易策略.txt")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_pipeline(
    *,
    output_root: Path,
    input_packet: Mapping[str, Any],
    generated_at: str,
) -> Mapping[str, Path]:
    scenario_tree = build_fuel_oil_2607_scenario_tree(input_packet=input_packet, generated_at=generated_at)
    transmission_map = build_fuel_oil_2607_transmission_map(
        input_packet=input_packet,
        scenario_tree=scenario_tree,
        generated_at=generated_at,
    )
    validation_ring = build_fuel_oil_2607_validation_ring(
        input_packet=input_packet,
        scenario_tree=scenario_tree,
        transmission_map=transmission_map,
        generated_at=generated_at,
    )
    trade_space = build_fuel_oil_2607_trade_space(
        input_packet=input_packet,
        scenario_tree=scenario_tree,
        validation_ring=validation_ring,
        generated_at=generated_at,
    )
    strategy_matrix = build_fuel_oil_2607_strategy_matrix(
        input_packet=input_packet,
        scenario_tree=scenario_tree,
        validation_ring=validation_ring,
        trade_space=trade_space,
        generated_at=generated_at,
    )
    summary = build_fuel_oil_2607_summary(
        input_packet=input_packet,
        scenario_tree=scenario_tree,
        validation_ring=validation_ring,
        trade_space=trade_space,
        strategy_matrix=strategy_matrix,
        generated_at=generated_at,
    )

    review_dir = output_root / "review"
    paths = {
        "input_packet": review_dir / "latest_fuel_oil_2607_input_packet.json",
        "scenario_tree": review_dir / "latest_fuel_oil_2607_scenario_tree.json",
        "transmission_map": review_dir / "latest_fuel_oil_2607_transmission_map.json",
        "validation_ring": review_dir / "latest_fuel_oil_2607_validation_ring.json",
        "trade_space": review_dir / "latest_fuel_oil_2607_trade_space.json",
        "strategy_matrix": review_dir / "latest_fuel_oil_2607_strategy_matrix.json",
        "summary": review_dir / "latest_fuel_oil_2607_summary.json",
    }
    _write_json(paths["input_packet"], dict(input_packet))
    _write_json(paths["scenario_tree"], scenario_tree)
    _write_json(paths["transmission_map"], transmission_map)
    _write_json(paths["validation_ring"], validation_ring)
    _write_json(paths["trade_space"], trade_space)
    _write_json(paths["strategy_matrix"], strategy_matrix)
    _write_json(paths["summary"], summary)
    return paths


def _filter_by_date(df: pd.DataFrame, *, as_of: date, lookback_days: int) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    ts_col = "date" if "date" in out.columns else "ts" if "ts" in out.columns else None
    if ts_col is None:
        return out
    out[ts_col] = pd.to_datetime(out[ts_col], errors="coerce")
    out = out.dropna(subset=[ts_col])
    mask = (out[ts_col].dt.date <= as_of) & (out[ts_col].dt.date >= as_of - timedelta(days=lookback_days))
    return out.loc[mask].reset_index(drop=True)


def _fetch_daily(symbol: str, *, as_of: date, lookback_days: int = 220) -> pd.DataFrame:
    try:
        raw = ak.futures_zh_daily_sina(symbol=symbol)
    except Exception:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "hold", "settle"])
    return _filter_by_date(raw, as_of=as_of, lookback_days=lookback_days)


def _fetch_spot(symbol: str) -> Mapping[str, Any]:
    try:
        raw = ak.futures_zh_spot(symbol=symbol, market="CF", adjust="0")
    except Exception:
        return {}
    if raw is None or raw.empty:
        return {}
    return raw.iloc[0].to_dict()


def _fetch_member_rank(contract: str, *, as_of: date) -> dict[str, pd.DataFrame]:
    payload: dict[str, pd.DataFrame] = {}
    for rank_symbol in ("成交量", "多单持仓", "空单持仓"):
        frame = pd.DataFrame()
        for offset in range(0, 6):
            query_date = (as_of - timedelta(days=offset)).strftime("%Y%m%d")
            try:
                candidate = ak.futures_hold_pos_sina(symbol=rank_symbol, contract=contract, date=query_date)
            except Exception:
                continue
            if candidate is not None and not candidate.empty:
                frame = candidate
                break
        payload[rank_symbol] = frame
    return payload


def build_live_input_packet(
    *,
    as_of: date,
    contract_focus: str,
    benchmark_contract: str,
    deferred_contract: str,
    generated_at: str,
    report_path: Path | None = None,
) -> dict[str, Any]:
    provider = PublicInternetResearchProvider(request_timeout_ms=5000, rate_limit_per_minute=30)
    macro_frame = provider.fetch_macro(as_of - timedelta(days=120), as_of)
    contract_daily = _fetch_daily(contract_focus, as_of=as_of)
    benchmark_daily = _fetch_daily(benchmark_contract, as_of=as_of)
    deferred_daily = _fetch_daily(deferred_contract, as_of=as_of)
    report_text = ""
    effective_report_path = report_path or DEFAULT_REPORT_PATH
    if effective_report_path.exists():
        report_text = effective_report_path.read_text(encoding="utf-8", errors="ignore")
    return build_fuel_oil_2607_input_packet(
        contract_focus=contract_focus,
        benchmark_contract=benchmark_contract,
        deferred_contract=deferred_contract,
        macro_frame=macro_frame,
        contract_daily=contract_daily,
        benchmark_daily=benchmark_daily,
        deferred_daily=deferred_daily,
        spot_snapshot=_fetch_spot(contract_focus),
        benchmark_spot_snapshot=_fetch_spot(benchmark_contract),
        member_rank_payload=_fetch_member_rank(contract_focus, as_of=as_of),
        report_text=report_text,
        generated_at=generated_at,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FU2607 structured reasoning and write latest review artifacts.")
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--contract", default="FU2607")
    parser.add_argument("--benchmark", default="SC2607")
    parser.add_argument("--deferred", default="FU2609")
    parser.add_argument("--as-of", required=True, help="YYYY-MM-DD")
    parser.add_argument("--now", required=True)
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    args = parser.parse_args()

    as_of = date.fromisoformat(args.as_of)
    input_packet = build_live_input_packet(
        as_of=as_of,
        contract_focus=args.contract,
        benchmark_contract=args.benchmark,
        deferred_contract=args.deferred,
        generated_at=args.now,
        report_path=Path(args.report_path),
    )
    artifacts = run_pipeline(
        output_root=Path(args.output_root),
        input_packet=input_packet,
        generated_at=args.now,
    )
    print(
        json.dumps(
            {
                "contract_focus": input_packet.get("contract_focus"),
                "primary_hint": input_packet.get("report_digest", ""),
                "artifacts": {k: str(v) for k, v in artifacts.items()},
                "coverage_ratio": (input_packet.get("coverage", {}) or {}).get("coverage_ratio"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
