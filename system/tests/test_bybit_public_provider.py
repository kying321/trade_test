from __future__ import annotations

from datetime import date, datetime, timezone
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lie_engine.data.providers import BybitSpotPublicProvider


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


class _FakeOpener:
    def __init__(self, handler):
        self._handler = handler

    def open(self, req, timeout=0):
        return self._handler(req, timeout=timeout)


class BybitPublicProviderTests(unittest.TestCase):
    def test_fetch_l2_depth_schema(self) -> None:
        provider = BybitSpotPublicProvider(rate_limit_per_minute=120)

        def _fake_urlopen(req, timeout=0, context=None):  # noqa: ANN001
            url = str(getattr(req, "full_url", req))
            if "/v5/market/orderbook" in url:
                return _FakeHTTPResponse(
                    {
                        "retCode": 0,
                        "result": {
                            "s": "BTCUSDT",
                            "b": [["100.0", "1.2"]],
                            "a": [["100.1", "0.8"]],
                            "ts": 1700000001000,
                            "seq": 101,
                        },
                    }
                )
            raise AssertionError(f"unexpected url: {url}")

        with patch("urllib.request.build_opener", return_value=_FakeOpener(_fake_urlopen)):
            out = provider.fetch_l2("BTCUSDT", datetime(2026, 2, 28, 12, 0, 0), datetime(2026, 2, 28, 12, 1, 0), depth=20)
        self.assertEqual(len(out), 1)
        self.assertEqual(str(out.loc[0, "exchange"]), "bybit_spot")
        self.assertEqual(str(out.loc[0, "symbol"]), "BTCUSDT")
        self.assertIn("event_ts_ms", out.columns)
        self.assertIn("recv_ts_ms", out.columns)

    def test_fetch_trades_filters_by_window(self) -> None:
        provider = BybitSpotPublicProvider(rate_limit_per_minute=120)

        def _fake_urlopen(req, timeout=0, context=None):  # noqa: ANN001
            url = str(getattr(req, "full_url", req))
            if "/v5/market/recent-trade" in url:
                return _FakeHTTPResponse(
                    {
                        "retCode": 0,
                        "result": {
                            "list": [
                                {"execId": "1", "price": "100.0", "size": "0.2", "side": "Buy", "time": "1700000001000"},
                                {"execId": "2", "price": "99.8", "size": "0.3", "side": "Sell", "time": "1700000002000"},
                                {"execId": "3", "price": "99.6", "size": "0.1", "side": "Buy", "time": "1699999990000"},
                            ]
                        },
                    }
                )
            raise AssertionError(f"unexpected url: {url}")

        with patch("urllib.request.build_opener", return_value=_FakeOpener(_fake_urlopen)):
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

    def test_fetch_trades_allows_small_future_clock_skew(self) -> None:
        provider = BybitSpotPublicProvider(rate_limit_per_minute=120, request_timeout_ms=5000)

        def _fake_urlopen(req, timeout=0, context=None):  # noqa: ANN001
            url = str(getattr(req, "full_url", req))
            if "/v5/market/recent-trade" in url:
                return _FakeHTTPResponse(
                    {
                        "retCode": 0,
                        "result": {
                            "list": [
                                {"execId": "1", "price": "100.0", "size": "0.2", "side": "Buy", "time": "1700000060000"},
                                {"execId": "2", "price": "99.8", "size": "0.3", "side": "Sell", "time": "1700000205000"},
                            ]
                        },
                    }
                )
            raise AssertionError(f"unexpected url: {url}")

        with patch("urllib.request.build_opener", return_value=_FakeOpener(_fake_urlopen)):
            out = provider.fetch_trades(
                "BTCUSDT",
                datetime.fromtimestamp(1700000001, tz=timezone.utc),
                datetime.fromtimestamp(1700000002, tz=timezone.utc),
                limit=100,
            )
        # request_timeout_ms=5000 => future skew allowance=100000ms.
        self.assertEqual(len(out), 1)
        self.assertEqual(str(out.iloc[0]["trade_id"]), "1")

    def test_fetch_ohlcv_uses_timeout_ceiling_5s(self) -> None:
        provider = BybitSpotPublicProvider(rate_limit_per_minute=120, request_timeout_ms=9000)
        timeouts: list[float] = []

        def _fake_urlopen(req, timeout=0, context=None):  # noqa: ANN001
            timeouts.append(float(timeout))
            url = str(getattr(req, "full_url", req))
            if "/v5/market/kline" in url:
                return _FakeHTTPResponse(
                    {
                        "retCode": 0,
                        "result": {
                            "list": [
                                ["1700000000000", "100.0", "110.0", "90.0", "105.0", "123.0", "0"],
                            ]
                        },
                    }
                )
            raise AssertionError(f"unexpected url: {url}")

        with patch("urllib.request.build_opener", return_value=_FakeOpener(_fake_urlopen)):
            out = provider.fetch_ohlcv("BTCUSDT", date(2026, 2, 28), date(2026, 2, 28), freq="1d")
        self.assertEqual(len(out), 1)
        self.assertEqual(str(out.loc[0, "asset_class"]), "crypto")
        self.assertTrue(bool(timeouts))
        self.assertTrue(all(t <= 5.0 for t in timeouts))

    def test_fetch_time_sync_sample_parses_market_time(self) -> None:
        provider = BybitSpotPublicProvider(rate_limit_per_minute=120)

        def _fake_urlopen(req, timeout=0, context=None):  # noqa: ANN001
            url = str(getattr(req, "full_url", req))
            if "/v5/market/time" in url:
                return _FakeHTTPResponse({"retCode": 0, "result": {"timeNano": "1700000001234000000", "timeSecond": "1700000001"}})
            raise AssertionError(f"unexpected url: {url}")

        with patch("urllib.request.build_opener", return_value=_FakeOpener(_fake_urlopen)):
            out = provider.fetch_time_sync_sample()
        self.assertEqual(int(out["server_ts_ms"]), 1700000001234)
        self.assertIn("offset_abs_ms", out)
        self.assertIn("rtt_ms", out)

    def test_public_provider_bypasses_env_proxy_by_default(self) -> None:
        provider = BybitSpotPublicProvider(rate_limit_per_minute=120)
        captured_handlers = []

        def _fake_build_opener(*handlers):  # noqa: ANN001
            captured_handlers.extend(list(handlers))
            return _FakeOpener(
                lambda req, timeout=0: _FakeHTTPResponse(
                    {"retCode": 0, "result": {"timeNano": "1700000001234000000", "timeSecond": "1700000001"}}
                )
            )

        with patch("urllib.request.build_opener", side_effect=_fake_build_opener):
            provider.fetch_time_sync_sample()

        proxy_handlers = [handler for handler in captured_handlers if handler.__class__.__name__ == "ProxyHandler"]
        self.assertTrue(proxy_handlers)
        self.assertEqual(getattr(proxy_handlers[0], "proxies", None), {})


if __name__ == "__main__":
    unittest.main()
