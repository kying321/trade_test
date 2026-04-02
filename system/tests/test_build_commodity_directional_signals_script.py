from __future__ import annotations

import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_commodity_directional_signals.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_commodity_directional_signals_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _frame(symbol: str = "BU2606") -> pd.DataFrame:
    rows = []
    start = dt.date(2026, 2, 10)
    price = 4100.0
    for idx in range(30):
        day = start + dt.timedelta(days=idx)
        open_px = price + idx * 8.0
        close_px = open_px + 12.0
        rows.append(
            {
                "ts": pd.Timestamp(day),
                "symbol": symbol,
                "open": open_px,
                "high": close_px + 15.0,
                "low": open_px - 10.0,
                "close": close_px,
                "volume": 100000 + idx * 1000,
                "source": "akshare.futures_zh_daily_sina:BU0",
                "asset_class": "future",
            }
        )
    return pd.DataFrame(rows)


def test_build_directional_ticket_emits_long_ticket_for_uptrend() -> None:
    mod = _load_module()
    ticket = mod.build_directional_ticket(
        symbol="BU2606",
        frame=_frame(),
        as_of=dt.date(2026, 3, 10),
        max_age_days=14,
    )
    assert ticket["symbol"] == "BU2606"
    assert ticket["allowed"] is True
    assert ticket["reasons"] == []
    assert ticket["signal"]["side"] == "LONG"
    assert ticket["signal"]["execution_price_ready"] is True
    assert ticket["signal"]["price_reference_kind"] == "contract_native_daily"
    assert ticket["signal"]["price_reference_source"].startswith("akshare.futures_zh_daily_sina")
    assert ticket["levels"]["entry_price"] > ticket["levels"]["stop_price"]
    assert ticket["levels"]["target_price"] > ticket["levels"]["entry_price"]


def test_build_directional_ticket_allows_minor_pullback_inside_uptrend() -> None:
    mod = _load_module()
    frame = _frame()
    frame.loc[len(frame) - 2, "close"] = 4543.0
    frame.loc[len(frame) - 1, "close"] = 4532.0
    frame.loc[len(frame) - 1, "open"] = 4647.0
    frame.loc[len(frame) - 1, "high"] = 4647.0
    frame.loc[len(frame) - 1, "low"] = 4433.0

    ticket = mod.build_directional_ticket(
        symbol="BU2606",
        frame=frame,
        as_of=dt.date(2026, 3, 11),
        max_age_days=14,
    )

    assert ticket["allowed"] is True
    assert ticket["signal"]["side"] == "LONG"


def test_main_writes_signal_and_ticket_artifacts(tmp_path: Path, monkeypatch, capsys) -> None:
    mod = _load_module()
    monkeypatch.setattr(mod, "fetch_future_daily", lambda symbol, start, end: _frame(symbol))

    rc = mod.main(
        [
            "--review-dir",
            str(tmp_path / "review"),
            "--output-root",
            str(tmp_path / "output"),
            "--date",
            "2026-03-10",
            "--symbols",
            "BU2606",
            "--max-age-days",
            "14",
            "--enable-state-carry",
            "--state-carry-max-age-days",
            "5",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["signal_count"] == 1
    assert payload["ticket_count"] == 1
    assert Path(payload["json"]).exists()
    assert Path(payload["signal_tickets_json"]).exists()
    tickets_payload = json.loads(Path(payload["signal_tickets_json"]).read_text(encoding="utf-8"))
    assert tickets_payload["symbols"] == ["BU2606"]
    assert tickets_payload["summary"]["ticket_count"] == 1
    assert tickets_payload["tickets"][0]["symbol"] == "BU2606"
