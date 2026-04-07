from __future__ import annotations

from datetime import datetime

import pandas as pd

from lie_engine.research.fuel_oil_2607_input_packet import build_fuel_oil_2607_input_packet
from lie_engine.research.fuel_oil_2607_scenario import build_fuel_oil_2607_scenario_tree
from lie_engine.research.fuel_oil_2607_summary import build_fuel_oil_2607_summary
from lie_engine.research.fuel_oil_2607_trade_space import build_fuel_oil_2607_trade_space
from lie_engine.research.fuel_oil_2607_transmission import build_fuel_oil_2607_transmission_map
from lie_engine.research.fuel_oil_2607_strategy import build_fuel_oil_2607_strategy_matrix
from lie_engine.research.fuel_oil_2607_validation import build_fuel_oil_2607_validation_ring


def _bars(
    symbol: str,
    closes: list[float],
    volumes: list[float],
    holds: list[float],
    *,
    start: str = "2026-01-02",
) -> pd.DataFrame:
    idx = pd.date_range(start=start, periods=len(closes), freq="B")
    return pd.DataFrame(
        {
            "date": idx,
            "ts": idx,
            "symbol": symbol,
            "open": closes,
            "high": [x + 25 for x in closes],
            "low": [x - 25 for x in closes],
            "close": closes,
            "settle": closes,
            "volume": volumes,
            "hold": holds,
        }
    )


def _macro_frame(
    *,
    inventory: float,
    inventory_delta: float,
    lfu_inventory: float,
    lfu_inventory_delta: float,
    bcti: float,
    bdti: float,
    cargo_yoy: float,
    port_yoy: float,
    date_text: str = "2026-04-07",
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp(date_text),
                "fuel_oil_inventory": inventory,
                "fuel_oil_inventory_delta": inventory_delta,
                "lfu_inventory": lfu_inventory,
                "lfu_inventory_delta": lfu_inventory_delta,
                "bcti_index": bcti,
                "bdti_index": bdti,
                "cargo_volume_yoy": cargo_yoy,
                "coastal_port_throughput_yoy": port_yoy,
            }
        ]
    )


def _member_rank(long_1: float, short_1: float) -> dict[str, pd.DataFrame]:
    return {
        "成交量": pd.DataFrame(
            [
                {"名次": 1, "会员简称": "中信期货", "成交量": 5000, "比上交易增减": 800},
                {"名次": 2, "会员简称": "国泰君安", "成交量": 3200, "比上交易增减": -200},
            ]
        ),
        "多单持仓": pd.DataFrame(
            [
                {"名次": 1, "会员简称": "中信期货", "多单持仓": long_1, "比上交易增减": 600},
                {"名次": 2, "会员简称": "瑞银期货", "多单持仓": 3000, "比上交易增减": 200},
            ]
        ),
        "空单持仓": pd.DataFrame(
            [
                {"名次": 1, "会员简称": "五矿期货", "空单持仓": short_1, "比上交易增减": -350},
                {"名次": 2, "会员简称": "国泰君安", "空单持仓": 2400, "比上交易增减": 100},
            ]
        ),
    }


def test_build_fuel_oil_2607_input_packet_extracts_fundamental_technical_and_member_signals() -> None:
    fuel_bars = _bars(
        "FU2607",
        closes=[3800 + i * 8 for i in range(80)],
        volumes=[12000 + i * 60 for i in range(80)],
        holds=[21000 + i * 40 for i in range(80)],
    )
    crude_bars = _bars(
        "SC2607",
        closes=[640 + i * 0.4 for i in range(80)],
        volumes=[9000 + i * 20 for i in range(80)],
        holds=[12000 + i * 15 for i in range(80)],
    )
    deferred_bars = _bars(
        "FU2609",
        closes=[3700 + i * 6 for i in range(80)],
        volumes=[8000 + i * 30 for i in range(80)],
        holds=[18000 + i * 10 for i in range(80)],
    )
    packet = build_fuel_oil_2607_input_packet(
        contract_focus="FU2607",
        benchmark_contract="SC2607",
        deferred_contract="FU2609",
        macro_frame=_macro_frame(
            inventory=182_000,
            inventory_delta=-4_500,
            lfu_inventory=135_000,
            lfu_inventory_delta=-2_000,
            bcti=812,
            bdti=1060,
            cargo_yoy=5.2,
            port_yoy=4.7,
        ),
        contract_daily=fuel_bars,
        benchmark_daily=crude_bars,
        deferred_daily=deferred_bars,
        spot_snapshot={
            "current_price": 4440.0,
            "last_settle_price": 4418.0,
            "volume": 23000,
            "hold": 26500,
        },
        benchmark_spot_snapshot={
            "current_price": 706.5,
            "last_settle_price": 699.2,
            "volume": 6800,
            "hold": 13220,
        },
        member_rank_payload=_member_rank(long_1=4200, short_1=3100),
        report_text="燃油框架修正后采用动态情景概率、证据优先、优先裂解与跨期而非固定点位。",
        generated_at="2026-04-07T15:00:00Z",
    )

    assert packet["generated_at_utc"] == "2026-04-07T15:00:00Z"
    assert packet["contract_focus"] == "FU2607"
    assert packet["fundamental_snapshot"]["inventory_signal"] == "tightening"
    assert packet["technical_snapshot"]["daily_trend"] == "up"
    assert packet["technical_snapshot"]["weekly_trend"] == "up"
    assert packet["technical_snapshot"]["rsi14"] > 50.0
    assert packet["technical_snapshot"]["calendar_spread"] > 0.0
    assert packet["participant_snapshot"]["net_top2"] > 0.0
    assert "refinery_margin" in packet["coverage"]["missing_fields"]


def test_fuel_oil_2607_reasoning_pipeline_outputs_base_scenario_and_long_bias() -> None:
    packet = build_fuel_oil_2607_input_packet(
        contract_focus="FU2607",
        benchmark_contract="SC2607",
        deferred_contract="FU2609",
        macro_frame=_macro_frame(
            inventory=178_000,
            inventory_delta=-6_000,
            lfu_inventory=132_000,
            lfu_inventory_delta=-1_500,
            bcti=850,
            bdti=1088,
            cargo_yoy=6.1,
            port_yoy=5.4,
        ),
        contract_daily=_bars("FU2607", [3850 + i * 7 for i in range(90)], [12000 + i * 45 for i in range(90)], [20000 + i * 35 for i in range(90)]),
        benchmark_daily=_bars("SC2607", [650 + i * 0.25 for i in range(90)], [8500 + i * 18 for i in range(90)], [11000 + i * 12 for i in range(90)]),
        deferred_daily=_bars("FU2609", [3740 + i * 5 for i in range(90)], [7000 + i * 20 for i in range(90)], [17000 + i * 8 for i in range(90)]),
        spot_snapshot={"current_price": 4472.0, "last_settle_price": 4430.0, "volume": 24000, "hold": 27050},
        benchmark_spot_snapshot={"current_price": 709.0, "last_settle_price": 702.0, "volume": 7000, "hold": 13500},
        member_rank_payload=_member_rank(long_1=4600, short_1=2900),
        report_text="主线修正为观察-诊断-执行分层，避免静态结论。",
        generated_at="2026-04-07T16:00:00Z",
    )
    scenario_tree = build_fuel_oil_2607_scenario_tree(input_packet=packet, generated_at="2026-04-07T16:00:00Z")
    transmission_map = build_fuel_oil_2607_transmission_map(
        input_packet=packet,
        scenario_tree=scenario_tree,
        generated_at="2026-04-07T16:00:00Z",
    )
    validation_ring = build_fuel_oil_2607_validation_ring(
        input_packet=packet,
        scenario_tree=scenario_tree,
        transmission_map=transmission_map,
        generated_at="2026-04-07T16:00:00Z",
    )
    trade_space = build_fuel_oil_2607_trade_space(
        input_packet=packet,
        scenario_tree=scenario_tree,
        validation_ring=validation_ring,
        generated_at="2026-04-07T16:00:00Z",
    )
    strategy_matrix = build_fuel_oil_2607_strategy_matrix(
        input_packet=packet,
        scenario_tree=scenario_tree,
        validation_ring=validation_ring,
        trade_space=trade_space,
        generated_at="2026-04-07T16:00:00Z",
    )
    summary = build_fuel_oil_2607_summary(
        input_packet=packet,
        scenario_tree=scenario_tree,
        validation_ring=validation_ring,
        trade_space=trade_space,
        strategy_matrix=strategy_matrix,
        generated_at="2026-04-07T16:00:00Z",
    )

    assert scenario_tree["primary_scenario"] == "base_repricing"
    assert abs(sum(row["path_probability"] for row in scenario_tree["scenario_nodes"]) - 1.0) < 1e-9
    assert transmission_map["primary_chain"] == "inventory_tightening_chain"
    assert validation_ring["boundary_pressure"] in {"supportive", "balanced"}
    assert trade_space["weighted_range"]["lower"] < packet["price_snapshot"]["last_price"] < trade_space["weighted_range"]["upper"]
    assert strategy_matrix["preferred_bias"] == "long"
    assert "裂解" in strategy_matrix["priority_strategies"][0]["strategy_name"]
    assert summary["headline"]
    assert summary["primary_scenario_brief"] == "base_repricing"


def test_fuel_oil_2607_reasoning_pipeline_detects_bearish_macro_path() -> None:
    packet = build_fuel_oil_2607_input_packet(
        contract_focus="FU2607",
        benchmark_contract="SC2607",
        deferred_contract="FU2609",
        macro_frame=_macro_frame(
            inventory=240_000,
            inventory_delta=9_000,
            lfu_inventory=165_000,
            lfu_inventory_delta=4_000,
            bcti=620,
            bdti=800,
            cargo_yoy=-7.5,
            port_yoy=-5.8,
        ),
        contract_daily=_bars("FU2607", [4480 - i * 8 for i in range(80)], [18000 - i * 50 for i in range(80)], [28000 - i * 45 for i in range(80)]),
        benchmark_daily=_bars("SC2607", [720 - i * 0.5 for i in range(80)], [7600 - i * 15 for i in range(80)], [14000 - i * 18 for i in range(80)]),
        deferred_daily=_bars("FU2609", [4420 - i * 6 for i in range(80)], [9000 - i * 25 for i in range(80)], [19000 - i * 15 for i in range(80)]),
        spot_snapshot={"current_price": 3875.0, "last_settle_price": 3920.0, "volume": 9200, "hold": 21500},
        benchmark_spot_snapshot={"current_price": 661.0, "last_settle_price": 670.0, "volume": 5200, "hold": 11800},
        member_rank_payload=_member_rank(long_1=2200, short_1=4400),
        report_text="熊市路径不再写死概率，而由需求与库存信号决定。",
        generated_at="2026-04-07T16:30:00Z",
    )
    scenario_tree = build_fuel_oil_2607_scenario_tree(input_packet=packet, generated_at="2026-04-07T16:30:00Z")
    validation_ring = build_fuel_oil_2607_validation_ring(
        input_packet=packet,
        scenario_tree=scenario_tree,
        transmission_map=build_fuel_oil_2607_transmission_map(
            input_packet=packet,
            scenario_tree=scenario_tree,
            generated_at="2026-04-07T16:30:00Z",
        ),
        generated_at="2026-04-07T16:30:00Z",
    )
    trade_space = build_fuel_oil_2607_trade_space(
        input_packet=packet,
        scenario_tree=scenario_tree,
        validation_ring=validation_ring,
        generated_at="2026-04-07T16:30:00Z",
    )
    strategy_matrix = build_fuel_oil_2607_strategy_matrix(
        input_packet=packet,
        scenario_tree=scenario_tree,
        validation_ring=validation_ring,
        trade_space=trade_space,
        generated_at="2026-04-07T16:30:00Z",
    )

    assert scenario_tree["primary_scenario"] == "macro_bear_slump"
    assert validation_ring["boundary_pressure"] == "fragile"
    assert trade_space["directional_bias"] == "down"
    assert strategy_matrix["preferred_bias"] == "short"
    assert strategy_matrix["priority_strategies"][0]["direction"] in {"short", "relative_short"}
