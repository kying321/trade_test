from __future__ import annotations

from datetime import date, datetime, timezone
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lie_engine.data.providers import BinanceSpotPublicProvider


class _FakeHTTPResponse:
    def __init__(self, payload, status: int = 200) -> None:
        self._payload = json.dumps(payload).encode("utf-8")
        self.status = int(status)

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class BinancePublicProviderTests(unittest.TestCase):
    def test_fetch_l2_depth_schema(self) -> None:
        provider = BinanceSpotPublicProvider(rate_limit_per_minute=120)

        def _fake_urlopen(req, timeout=0, context=None):  # noqa: ANN001
            url = str(getattr(req, "full_url", req))
            if "/api/v3/depth" in url:
                return _FakeHTTPResponse(
                    {
                        "lastUpdateId": 1027024,
                        "bids": [["4.00000000", "431.00000000"]],
                        "asks": [["4.00000200", "12.00000000"]],
                    }
                )
            raise AssertionError(f"unexpected url: {url}")

        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            out = provider.fetch_l2("BTCUSDT", datetime(2026, 2, 28, 12, 0, 0), datetime(2026, 2, 28, 12, 1, 0), depth=20)
        self.assertEqual(len(out), 1)
        self.assertIn("event_ts_ms", out.columns)
        self.assertIn("recv_ts_ms", out.columns)
        self.assertIn("bids", out.columns)
        self.assertIn("asks", out.columns)
        self.assertEqual(str(out.loc[0, "symbol"]), "BTCUSDT")
        self.assertEqual(str(out.loc[0, "exchange"]), "binance_spot")

    def test_fetch_trades_side_mapping(self) -> None:
        provider = BinanceSpotPublicProvider(rate_limit_per_minute=120)

        def _fake_urlopen(req, timeout=0, context=None):  # noqa: ANN001
            url = str(getattr(req, "full_url", req))
            if "/api/v3/aggTrades" in url:
                return _FakeHTTPResponse(
                    [
                        {"a": 1, "p": "100.0", "q": "0.2", "T": 1700000001000, "m": False},
                        {"a": 2, "p": "99.8", "q": "0.3", "T": 1700000002000, "m": True},
                    ]
                )
            raise AssertionError(f"unexpected url: {url}")

        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            out = provider.fetch_trades(
                "BTCUSDT",
                datetime.fromtimestamp(1700000001, tz=timezone.utc),
                datetime.fromtimestamp(1700000002, tz=timezone.utc),
                limit=100,
            )
        self.assertEqual(len(out), 2)
        self.assertEqual(str(out.iloc[0]["side"]), "BUY")
        self.assertEqual(str(out.iloc[1]["side"]), "SELL")
        self.assertIn("trade_id", out.columns)

    def test_fetch_ohlcv_uses_timeout_ceiling_5s(self) -> None:
        provider = BinanceSpotPublicProvider(rate_limit_per_minute=120, request_timeout_ms=9000)
        timeouts: list[float] = []

        def _fake_urlopen(req, timeout=0, context=None):  # noqa: ANN001
            timeouts.append(float(timeout))
            url = str(getattr(req, "full_url", req))
            if "/api/v3/klines" in url:
                return _FakeHTTPResponse(
                    [
                        [
                            1700000000000,
                            "100.0",
                            "110.0",
                            "90.0",
                            "105.0",
                            "123.0",
                            1700000059999,
                            "0",
                            1,
                            "0",
                            "0",
                            "0",
                        ]
                    ]
                )
            raise AssertionError(f"unexpected url: {url}")

        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            out = provider.fetch_ohlcv("BTCUSDT", date(2026, 2, 28), date(2026, 2, 28), freq="1d")
        self.assertEqual(len(out), 1)
        self.assertTrue(bool(timeouts))
        self.assertTrue(all(t <= 5.0 for t in timeouts))

    def test_fetch_time_sync_sample_parses_server_time(self) -> None:
        provider = BinanceSpotPublicProvider(rate_limit_per_minute=120)

        def _fake_urlopen(req, timeout=0, context=None):  # noqa: ANN001
            url = str(getattr(req, "full_url", req))
            if "/api/v3/time" in url:
                return _FakeHTTPResponse({"serverTime": 1700000001234})
            raise AssertionError(f"unexpected url: {url}")

        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            out = provider.fetch_time_sync_sample()
        self.assertEqual(int(out["server_ts_ms"]), 1700000001234)
        self.assertIn("offset_abs_ms", out)
        self.assertIn("rtt_ms", out)


if __name__ == "__main__":
    unittest.main()
