from __future__ import annotations

from contextlib import ExitStack
from datetime import date, datetime
from pathlib import Path
import sys
import types
import unittest

import pandas as pd
from unittest.mock import patch, Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lie_engine.data.providers import (
    BinanceSpotPublicProvider,
    BybitSpotPublicProvider,
    PublicInternetResearchProvider,
)


class PublicInternetResearchProviderTests(unittest.TestCase):
    def test_fetch_freight_index_falls_back_to_direct_series_when_aggregate_source_breaks(self) -> None:
        provider = PublicInternetResearchProvider()
        fake_akshare = types.SimpleNamespace(
            macro_china_freight_index=lambda: (_ for _ in ()).throw(UnicodeDecodeError("gbk", b"\x88", 0, 1, "illegal multibyte sequence")),
            macro_shipping_bdi=lambda: pd.DataFrame(
                {
                    "日期": pd.to_datetime(["2026-04-01", "2026-04-07"]),
                    "最新值": [2030.0, 2095.0],
                }
            ),
            macro_shipping_bcti=lambda: pd.DataFrame(
                {
                    "日期": pd.to_datetime(["2026-04-01", "2026-04-07"]),
                    "最新值": [1994.0, 1969.0],
                }
            ),
            macro_china_bdti_index=lambda: pd.DataFrame(
                {
                    "日期": pd.to_datetime(["2026-04-01", "2026-04-07"]),
                    "最新值": [3678.0, 3639.0],
                }
            ),
        )

        with patch.dict(sys.modules, {"akshare": fake_akshare}):
            out = provider._fetch_freight_index(oldest_date=date(2026, 4, 1), latest_date=date(2026, 4, 7))

        self.assertFalse(out.empty)
        self.assertEqual(
            list(out.columns),
            [
                "截止日期",
                "波罗的海综合运价指数BDI",
                "油轮运价指数成品油运价指数BCTI",
                "油轮运价指数原油运价指数BDTI",
                "波罗的海超级大灵便型船BSI指数",
            ],
        )
        latest = out[out["截止日期"] == pd.Timestamp("2026-04-07")]
        self.assertEqual(len(latest), 1)
        self.assertAlmostEqual(float(latest.iloc[0]["波罗的海综合运价指数BDI"]), 2095.0, places=6)
        self.assertAlmostEqual(float(latest.iloc[0]["油轮运价指数成品油运价指数BCTI"]), 1969.0, places=6)
        self.assertAlmostEqual(float(latest.iloc[0]["油轮运价指数原油运价指数BDTI"]), 3639.0, places=6)
        self.assertTrue(pd.isna(latest.iloc[0]["波罗的海超级大灵便型船BSI指数"]))

    def test_fetch_society_electricity_http_fallback_parses_jsonp(self) -> None:
        provider = PublicInternetResearchProvider()
        body = """/*<script>location.href='//sina.com';</script>*/
SINAREMOTECALLCALLBACK1601557771972(({count:'2',data:[
['2026.2','165460000','6.1','','','2230000','7.4','102790000','6.3','32310000','8.3','','','','','',''],
['2026.1','100000000','5.0','','','1200000','6.0','70000000','5.8','21000000','7.2','','','','','','']
]}))"""

        fake_resp = Mock()
        fake_resp.status_code = 200
        fake_resp.text = body

        with patch("requests.get", return_value=fake_resp):
            out = provider._fetch_society_electricity_http_fallback(oldest_date=date(2026, 1, 1), latest_date=date(2026, 3, 31))

        self.assertFalse(out.empty)
        self.assertEqual(list(out.columns), ["统计时间", "全社会用电量", "全社会用电量同比", "第二产业用电量", "第二产业用电量同比", "date"])
        latest = out[out["date"] == pd.Timestamp("2026-02-01")]
        self.assertEqual(len(latest), 1)
        self.assertAlmostEqual(float(latest.iloc[0]["全社会用电量"]), 165460000.0, places=6)
        self.assertAlmostEqual(float(latest.iloc[0]["第二产业用电量同比"]), 6.3, places=6)

    def test_crypto_public_providers_do_not_emit_placeholder_macro_rows(self) -> None:
        for provider in [BinanceSpotPublicProvider(), BybitSpotPublicProvider()]:
            out = provider.fetch_macro(start=date(2026, 3, 1), end=date(2026, 3, 31))
            self.assertTrue(out.empty)

    def test_fetch_ohlcv_supports_domestic_future_daily(self) -> None:
        provider = PublicInternetResearchProvider()
        future_df = pd.DataFrame(
            {
                "ts": pd.to_datetime(["2026-03-27", "2026-03-30"]),
                "symbol": ["BU2606", "BU2606"],
                "open": [3120.0, 3140.0],
                "high": [3150.0, 3180.0],
                "low": [3100.0, 3130.0],
                "close": [3145.0, 3170.0],
                "volume": [100_000.0, 120_000.0],
                "source": ["akshare.futures_zh_daily_sina:BU0", "akshare.futures_zh_daily_sina:BU0"],
                "asset_class": ["future", "future"],
            }
        )

        with patch("lie_engine.research.real_data.fetch_future_daily", return_value=future_df):
            out = provider.fetch_ohlcv(
                symbol="BU2606",
                start=date(2026, 3, 27),
                end=date(2026, 3, 30),
                freq="1d",
            )

        self.assertEqual(len(out), 2)
        self.assertEqual(set(out["symbol"]), {"BU2606"})
        self.assertEqual(set(out["asset_class"]), {"future"})
        self.assertEqual(set(out["source"]), {provider.name})
        self.assertIn("source_detail", out.columns)
        self.assertTrue(str(out.iloc[0]["source_detail"]).startswith("akshare.futures_zh_daily_sina"))

    def test_fetch_macro_merges_cpi_ppi_and_lpr_with_forward_filled_lpr(self) -> None:
        provider = PublicInternetResearchProvider()
        cpi = pd.DataFrame(
            {
                "日期": pd.to_datetime(["2026-03-09", "2026-04-10"]),
                "今值": [0.2, 0.5],
                "预测值": [0.1, 0.4],
                "前值": [0.0, 0.2],
            }
        )
        ppi = pd.DataFrame(
            {
                "日期": pd.to_datetime(["2026-03-09", "2026-04-10"]),
                "今值": [-2.2, -2.0],
                "预测值": [-2.1, -1.9],
                "前值": [-2.3, -2.2],
            }
        )
        lpr = pd.DataFrame(
            {
                "TRADE_DATE": pd.to_datetime(["2026-02-20", "2026-03-20"]),
                "LPR1Y": [3.10, 3.00],
            }
        )

        with (
            patch.object(PublicInternetResearchProvider, "_fetch_jin10_indicator", side_effect=[cpi, ppi]),
            patch.object(PublicInternetResearchProvider, "_fetch_lpr_table", return_value=lpr),
        ):
            out = provider.fetch_macro(start=date(2026, 3, 1), end=date(2026, 3, 31))

        self.assertFalse(out.empty)
        self.assertEqual(list(out.columns[:5]), ["date", "cpi_yoy", "ppi_yoy", "lpr_1y", "source"])
        march_release = out[out["date"] == pd.Timestamp("2026-03-09")]
        self.assertEqual(len(march_release), 1)
        self.assertAlmostEqual(float(march_release.iloc[0]["cpi_yoy"]), 0.2, places=6)
        self.assertAlmostEqual(float(march_release.iloc[0]["ppi_yoy"]), -2.2, places=6)
        self.assertAlmostEqual(float(march_release.iloc[0]["lpr_1y"]), 3.10, places=6)
        self.assertTrue((out["source"] == provider.name).all())
        self.assertIn("cpi_source", out.columns)
        self.assertIn("ppi_source", out.columns)
        self.assertIn("lpr_source", out.columns)
        self.assertNotIn(pd.Timestamp("2026-04-10"), set(pd.to_datetime(out["date"])))

    def test_fetch_macro_includes_energy_and_commodity_index_columns(self) -> None:
        provider = PublicInternetResearchProvider()
        cpi = pd.DataFrame({"日期": pd.to_datetime(["2026-03-09"]), "今值": [0.2], "预测值": [0.1], "前值": [0.0]})
        ppi = pd.DataFrame({"日期": pd.to_datetime(["2026-03-09"]), "今值": [-2.2], "预测值": [-2.1], "前值": [-2.3]})
        lpr = pd.DataFrame({"TRADE_DATE": pd.to_datetime(["2026-03-20"]), "LPR1Y": [3.0]})
        energy = pd.DataFrame(
            {
                "日期": pd.to_datetime(["2026-03-29", "2026-03-30"]),
                "沿海六大电库存": [1280.0, 1275.0],
                "日耗": [79.0, 80.5],
                "存煤可用天数": [16.2, 15.8],
            }
        )
        commodity = pd.DataFrame(
            {
                "日期": pd.to_datetime(["2026-03-30"]),
                "最新值": [1420.0],
                "涨跌幅": [1.25],
            }
        )

        with (
            patch.object(PublicInternetResearchProvider, "_fetch_jin10_indicator", side_effect=[cpi, ppi]),
            patch.object(PublicInternetResearchProvider, "_fetch_lpr_table", return_value=lpr),
            patch.object(PublicInternetResearchProvider, "_fetch_daily_energy", return_value=energy),
            patch.object(PublicInternetResearchProvider, "_fetch_commodity_price_index", return_value=commodity),
        ):
            out = provider.fetch_macro(start=date(2026, 3, 1), end=date(2026, 3, 31))

        self.assertIn("coastal_power_coal_inventory", out.columns)
        self.assertIn("coastal_power_coal_daily_burn", out.columns)
        self.assertIn("coastal_power_coal_days", out.columns)
        self.assertIn("commodity_price_index", out.columns)
        self.assertIn("commodity_price_index_pct_chg", out.columns)
        row = out[out["date"] == pd.Timestamp("2026-03-30")]
        self.assertEqual(len(row), 1)
        self.assertAlmostEqual(float(row.iloc[0]["coastal_power_coal_inventory"]), 1275.0, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["coastal_power_coal_daily_burn"]), 80.5, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["coastal_power_coal_days"]), 15.8, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["commodity_price_index"]), 1420.0, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["commodity_price_index_pct_chg"]), 1.25, places=6)

    def test_fetch_macro_includes_energy_and_freight_indices(self) -> None:
        provider = PublicInternetResearchProvider()
        cpi = pd.DataFrame({"日期": pd.to_datetime(["2026-03-09"]), "今值": [0.2], "预测值": [0.1], "前值": [0.0]})
        ppi = pd.DataFrame({"日期": pd.to_datetime(["2026-03-09"]), "今值": [-2.2], "预测值": [-2.1], "前值": [-2.3]})
        lpr = pd.DataFrame({"TRADE_DATE": pd.to_datetime(["2026-03-20"]), "LPR1Y": [3.0]})
        energy = pd.DataFrame(columns=["日期", "沿海六大电库存", "日耗", "存煤可用天数"])
        commodity = pd.DataFrame(columns=["日期", "最新值", "涨跌幅"])
        energy_index = pd.DataFrame(
            {
                "日期": pd.to_datetime(["2026-03-30"]),
                "最新值": [1007.0],
                "涨跌幅": [0.0],
            }
        )
        freight = pd.DataFrame(
            {
                "截止日期": pd.to_datetime(["2026-03-26"]),
                "波罗的海综合运价指数BDI": [2014.0],
                "油轮运价指数成品油运价指数BCTI": [1936.0],
                "油轮运价指数原油运价指数BDTI": [3716.0],
                "波罗的海超级大灵便型船BSI指数": [1205.0],
            }
        )

        with (
            patch.object(PublicInternetResearchProvider, "_fetch_jin10_indicator", side_effect=[cpi, ppi]),
            patch.object(PublicInternetResearchProvider, "_fetch_lpr_table", return_value=lpr),
            patch.object(PublicInternetResearchProvider, "_fetch_daily_energy", return_value=energy),
            patch.object(PublicInternetResearchProvider, "_fetch_commodity_price_index", return_value=commodity),
            patch.object(PublicInternetResearchProvider, "_fetch_energy_index", return_value=energy_index),
            patch.object(PublicInternetResearchProvider, "_fetch_freight_index", return_value=freight),
        ):
            out = provider.fetch_macro(start=date(2026, 3, 1), end=date(2026, 3, 31))

        self.assertIn("energy_index", out.columns)
        self.assertIn("energy_index_pct_chg", out.columns)
        self.assertIn("bdi_index", out.columns)
        self.assertIn("bcti_index", out.columns)
        self.assertIn("bdti_index", out.columns)
        self.assertIn("bsi_index", out.columns)
        row = out[out["date"] == pd.Timestamp("2026-03-30")]
        self.assertEqual(len(row), 1)
        self.assertAlmostEqual(float(row.iloc[0]["energy_index"]), 1007.0, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["energy_index_pct_chg"]), 0.0, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["bdi_index"]), 2014.0, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["bcti_index"]), 1936.0, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["bdti_index"]), 3716.0, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["bsi_index"]), 1205.0, places=6)

    def test_fetch_macro_includes_oil_price_adjustment_columns(self) -> None:
        provider = PublicInternetResearchProvider()
        cpi = pd.DataFrame({"日期": pd.to_datetime(["2026-03-09"]), "今值": [0.2], "预测值": [0.1], "前值": [0.0]})
        ppi = pd.DataFrame({"日期": pd.to_datetime(["2026-03-09"]), "今值": [-2.2], "预测值": [-2.1], "前值": [-2.3]})
        lpr = pd.DataFrame({"TRADE_DATE": pd.to_datetime(["2026-03-20"]), "LPR1Y": [3.0]})
        energy = pd.DataFrame(columns=["日期", "沿海六大电库存", "日耗", "存煤可用天数"])
        commodity = pd.DataFrame(columns=["日期", "最新值", "涨跌幅"])
        energy_index = pd.DataFrame(columns=["日期", "最新值", "涨跌幅"])
        freight = pd.DataFrame(columns=["截止日期", "波罗的海综合运价指数BDI", "油轮运价指数成品油运价指数BCTI", "油轮运价指数原油运价指数BDTI", "波罗的海超级大灵便型船BSI指数"])
        oil_hist = pd.DataFrame(
            {
                "调整日期": pd.to_datetime(["2026-03-10", "2026-03-24"]),
                "汽油价格": [8745.0, 9905.0],
                "柴油价格": [7720.0, 8835.0],
                "汽油涨跌": [695.0, 1160.0],
                "柴油涨跌": [670.0, 1115.0],
            }
        )

        with (
            patch.object(PublicInternetResearchProvider, "_fetch_jin10_indicator", side_effect=[cpi, ppi]),
            patch.object(PublicInternetResearchProvider, "_fetch_lpr_table", return_value=lpr),
            patch.object(PublicInternetResearchProvider, "_fetch_daily_energy", return_value=energy),
            patch.object(PublicInternetResearchProvider, "_fetch_commodity_price_index", return_value=commodity),
            patch.object(PublicInternetResearchProvider, "_fetch_energy_index", return_value=energy_index),
            patch.object(PublicInternetResearchProvider, "_fetch_freight_index", return_value=freight),
            patch.object(PublicInternetResearchProvider, "_fetch_oil_hist", return_value=oil_hist),
        ):
            out = provider.fetch_macro(start=date(2026, 3, 1), end=date(2026, 3, 31))

        self.assertIn("gasoline_price", out.columns)
        self.assertIn("diesel_price", out.columns)
        self.assertIn("gasoline_price_delta", out.columns)
        self.assertIn("diesel_price_delta", out.columns)
        row = out[out["date"] == pd.Timestamp("2026-03-24")]
        self.assertEqual(len(row), 1)
        self.assertAlmostEqual(float(row.iloc[0]["gasoline_price"]), 9905.0, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["diesel_price"]), 8835.0, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["gasoline_price_delta"]), 1160.0, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["diesel_price_delta"]), 1115.0, places=6)

    def test_fetch_macro_includes_oil_detail_regional_means(self) -> None:
        provider = PublicInternetResearchProvider()
        cpi = pd.DataFrame({"日期": pd.to_datetime(["2026-03-09"]), "今值": [0.2], "预测值": [0.1], "前值": [0.0]})
        ppi = pd.DataFrame({"日期": pd.to_datetime(["2026-03-09"]), "今值": [-2.2], "预测值": [-2.1], "前值": [-2.3]})
        lpr = pd.DataFrame({"TRADE_DATE": pd.to_datetime(["2026-03-20"]), "LPR1Y": [3.0]})
        empty_daily = pd.DataFrame(columns=["日期", "沿海六大电库存", "日耗", "存煤可用天数"])
        empty_index = pd.DataFrame(columns=["日期", "最新值", "涨跌幅"])
        empty_freight = pd.DataFrame(columns=["截止日期", "波罗的海综合运价指数BDI", "油轮运价指数成品油运价指数BCTI", "油轮运价指数原油运价指数BDTI", "波罗的海超级大灵便型船BSI指数"])
        oil_hist = pd.DataFrame(
            {
                "调整日期": pd.to_datetime(["2026-03-24"]),
                "汽油价格": [9905.0],
                "柴油价格": [8835.0],
                "汽油涨跌": [1160.0],
                "柴油涨跌": [1115.0],
            }
        )
        oil_detail = pd.DataFrame(
            {
                "日期": ["2026-03-24", "2026-03-24"],
                "地区": ["北京", "上海"],
                "V_0": [8.83, 8.70],
                "V_92": [9.30, 9.18],
                "V_95": [9.90, 9.75],
            }
        )

        with ExitStack() as stack:
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_jin10_indicator", side_effect=[cpi, ppi]))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_lpr_table", return_value=lpr))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_daily_energy", return_value=empty_daily))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_commodity_price_index", return_value=empty_index))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_energy_index", return_value=empty_index))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_freight_index", return_value=empty_freight))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_oil_hist", return_value=oil_hist))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_oil_detail", return_value=oil_detail))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_construction_index", return_value=empty_index))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_construction_price_index", return_value=empty_index))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_real_estate_index", return_value=empty_index))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_society_electricity", return_value=pd.DataFrame(columns=["统计时间"])))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_new_house_price", return_value=pd.DataFrame(columns=["日期", "城市"])))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_industrial_production_yoy", return_value=pd.DataFrame(columns=["日期", "今值"])))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_exports_yoy", return_value=pd.DataFrame(columns=["日期", "今值"])))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_imports_yoy", return_value=pd.DataFrame(columns=["日期", "今值"])))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_pmi_manufacturing", return_value=pd.DataFrame(columns=["日期", "今值"])))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_non_man_pmi", return_value=pd.DataFrame(columns=["日期", "今值"])))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_asphalt_inventory", return_value=pd.DataFrame(columns=["日期", "库存", "增减"])))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_lfu_inventory", return_value=pd.DataFrame(columns=["日期", "库存", "增减"])))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_fuel_oil_inventory", return_value=pd.DataFrame(columns=["日期", "库存", "增减"])))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_rebar_inventory", return_value=pd.DataFrame(columns=["日期", "库存", "增减"])))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_hotcoil_inventory", return_value=pd.DataFrame(columns=["日期", "库存", "增减"])))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_coking_coal_inventory", return_value=pd.DataFrame(columns=["日期", "库存", "增减"])))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_coke_inventory", return_value=pd.DataFrame(columns=["日期", "库存", "增减"])))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_iron_ore_inventory", return_value=pd.DataFrame(columns=["日期", "库存", "增减"])))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_glass_inventory", return_value=pd.DataFrame(columns=["日期", "库存", "增减"])))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_soda_ash_inventory", return_value=pd.DataFrame(columns=["日期", "库存", "增减"])))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_pvc_inventory", return_value=pd.DataFrame(columns=["日期", "库存", "增减"])))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_pp_inventory", return_value=pd.DataFrame(columns=["日期", "库存", "增减"])))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_methanol_inventory", return_value=pd.DataFrame(columns=["日期", "库存", "增减"])))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_eg_inventory", return_value=pd.DataFrame(columns=["日期", "库存", "增减"])))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_lpg_inventory", return_value=pd.DataFrame(columns=["日期", "库存", "增减"])))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_society_traffic_volume", return_value=pd.DataFrame(columns=["统计时间"])))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_money_supply", return_value=pd.DataFrame(columns=["月份"])))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_new_financial_credit", return_value=pd.DataFrame(columns=["月份"])))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_bank_financing_index", return_value=pd.DataFrame(columns=["日期"])))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_fx_reserves", return_value=pd.DataFrame(columns=["日期"])))
            out = provider.fetch_macro(start=date(2026, 3, 1), end=date(2026, 3, 31))

        self.assertIn("diesel_price_regional_mean", out.columns)
        self.assertIn("gasoline_92_price_regional_mean", out.columns)
        self.assertIn("gasoline_95_price_regional_mean", out.columns)
        row = out[out["date"] == pd.Timestamp("2026-03-24")]
        self.assertEqual(len(row), 1)
        self.assertAlmostEqual(float(row.iloc[0]["diesel_price_regional_mean"]), (8.83 + 8.70) / 2, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["gasoline_92_price_regional_mean"]), (9.30 + 9.18) / 2, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["gasoline_95_price_regional_mean"]), (9.90 + 9.75) / 2, places=6)

    def test_fetch_macro_includes_construction_indices(self) -> None:
        provider = PublicInternetResearchProvider()
        cpi = pd.DataFrame({"日期": pd.to_datetime(["2026-03-09"]), "今值": [0.2], "预测值": [0.1], "前值": [0.0]})
        ppi = pd.DataFrame({"日期": pd.to_datetime(["2026-03-09"]), "今值": [-2.2], "预测值": [-2.1], "前值": [-2.3]})
        lpr = pd.DataFrame({"TRADE_DATE": pd.to_datetime(["2026-03-20"]), "LPR1Y": [3.0]})
        empty_daily = pd.DataFrame(columns=["日期", "沿海六大电库存", "日耗", "存煤可用天数"])
        empty_index = pd.DataFrame(columns=["日期", "最新值", "涨跌幅"])
        empty_freight = pd.DataFrame(columns=["截止日期", "波罗的海综合运价指数BDI", "油轮运价指数成品油运价指数BCTI", "油轮运价指数原油运价指数BDTI", "波罗的海超级大灵便型船BSI指数"])
        empty_oil = pd.DataFrame(columns=["调整日期", "汽油价格", "柴油价格", "汽油涨跌", "柴油涨跌"])
        construction_index = pd.DataFrame(
            {
                "日期": pd.to_datetime(["2026-03-29"]),
                "最新值": [926.0],
                "涨跌幅": [0.0],
            }
        )
        construction_price = pd.DataFrame(
            {
                "日期": pd.to_datetime(["2024-11-26"]),
                "最新值": [1140.89],
                "涨跌幅": [1.205535],
            }
        )

        with (
            patch.object(PublicInternetResearchProvider, "_fetch_jin10_indicator", side_effect=[cpi, ppi]),
            patch.object(PublicInternetResearchProvider, "_fetch_lpr_table", return_value=lpr),
            patch.object(PublicInternetResearchProvider, "_fetch_daily_energy", return_value=empty_daily),
            patch.object(PublicInternetResearchProvider, "_fetch_commodity_price_index", return_value=empty_index),
            patch.object(PublicInternetResearchProvider, "_fetch_energy_index", return_value=empty_index),
            patch.object(PublicInternetResearchProvider, "_fetch_freight_index", return_value=empty_freight),
            patch.object(PublicInternetResearchProvider, "_fetch_oil_hist", return_value=empty_oil),
            patch.object(PublicInternetResearchProvider, "_fetch_construction_index", return_value=construction_index),
            patch.object(PublicInternetResearchProvider, "_fetch_construction_price_index", return_value=construction_price),
        ):
            out = provider.fetch_macro(start=date(2026, 3, 1), end=date(2026, 3, 31))

        self.assertIn("construction_index", out.columns)
        self.assertIn("construction_index_pct_chg", out.columns)
        self.assertIn("construction_price_index", out.columns)
        self.assertIn("construction_price_index_pct_chg", out.columns)
        row = out[out["date"] == pd.Timestamp("2026-03-29")]
        self.assertEqual(len(row), 1)
        self.assertAlmostEqual(float(row.iloc[0]["construction_index"]), 926.0, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["construction_index_pct_chg"]), 0.0, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["construction_price_index"]), 1140.89, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["construction_price_index_pct_chg"]), 1.205535, places=6)

    def test_fetch_macro_includes_real_estate_and_electricity_proxies(self) -> None:
        provider = PublicInternetResearchProvider()
        cpi = pd.DataFrame({"日期": pd.to_datetime(["2026-03-09"]), "今值": [0.2], "预测值": [0.1], "前值": [0.0]})
        ppi = pd.DataFrame({"日期": pd.to_datetime(["2026-03-09"]), "今值": [-2.2], "预测值": [-2.1], "前值": [-2.3]})
        lpr = pd.DataFrame({"TRADE_DATE": pd.to_datetime(["2026-03-20"]), "LPR1Y": [3.0]})
        empty_daily = pd.DataFrame(columns=["日期", "沿海六大电库存", "日耗", "存煤可用天数"])
        empty_index = pd.DataFrame(columns=["日期", "最新值", "涨跌幅"])
        empty_freight = pd.DataFrame(columns=["截止日期", "波罗的海综合运价指数BDI", "油轮运价指数成品油运价指数BCTI", "油轮运价指数原油运价指数BDTI", "波罗的海超级大灵便型船BSI指数"])
        empty_oil = pd.DataFrame(columns=["调整日期", "汽油价格", "柴油价格", "汽油涨跌", "柴油涨跌"])
        real_estate = pd.DataFrame(
            {
                "日期": pd.to_datetime(["2025-12-01"]),
                "最新值": [91.45],
                "涨跌幅": [-0.478833],
            }
        )
        society_electricity = pd.DataFrame(
            {
                "统计时间": ["2026.2"],
                "全社会用电量": [165460000.0],
                "全社会用电量同比": [6.1],
                "第二产业用电量": [102790000.0],
                "第二产业用电量同比": [6.3],
            }
        )

        with (
            patch.object(PublicInternetResearchProvider, "_fetch_jin10_indicator", side_effect=[cpi, ppi]),
            patch.object(PublicInternetResearchProvider, "_fetch_lpr_table", return_value=lpr),
            patch.object(PublicInternetResearchProvider, "_fetch_daily_energy", return_value=empty_daily),
            patch.object(PublicInternetResearchProvider, "_fetch_commodity_price_index", return_value=empty_index),
            patch.object(PublicInternetResearchProvider, "_fetch_energy_index", return_value=empty_index),
            patch.object(PublicInternetResearchProvider, "_fetch_freight_index", return_value=empty_freight),
            patch.object(PublicInternetResearchProvider, "_fetch_oil_hist", return_value=empty_oil),
            patch.object(PublicInternetResearchProvider, "_fetch_construction_index", return_value=empty_index),
            patch.object(PublicInternetResearchProvider, "_fetch_construction_price_index", return_value=empty_index),
            patch.object(PublicInternetResearchProvider, "_fetch_real_estate_index", return_value=real_estate),
            patch.object(PublicInternetResearchProvider, "_fetch_society_electricity", return_value=society_electricity),
        ):
            out = provider.fetch_macro(start=date(2026, 2, 1), end=date(2026, 3, 31))

        self.assertIn("real_estate_index", out.columns)
        self.assertIn("real_estate_index_pct_chg", out.columns)
        self.assertIn("society_electricity_total", out.columns)
        self.assertIn("society_electricity_yoy", out.columns)
        self.assertIn("secondary_industry_electricity", out.columns)
        self.assertIn("secondary_industry_electricity_yoy", out.columns)
        row = out[out["date"] == pd.Timestamp("2026-02-01")]
        self.assertEqual(len(row), 1)
        self.assertAlmostEqual(float(row.iloc[0]["real_estate_index"]), 91.45, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["real_estate_index_pct_chg"]), -0.478833, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["society_electricity_total"]), 165460000.0, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["society_electricity_yoy"]), 6.1, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["secondary_industry_electricity"]), 102790000.0, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["secondary_industry_electricity_yoy"]), 6.3, places=6)

    def test_fetch_macro_includes_new_house_price_proxies(self) -> None:
        provider = PublicInternetResearchProvider()
        cpi = pd.DataFrame({"日期": pd.to_datetime(["2026-03-09"]), "今值": [0.2], "预测值": [0.1], "前值": [0.0]})
        ppi = pd.DataFrame({"日期": pd.to_datetime(["2026-03-09"]), "今值": [-2.2], "预测值": [-2.1], "前值": [-2.3]})
        lpr = pd.DataFrame({"TRADE_DATE": pd.to_datetime(["2026-03-20"]), "LPR1Y": [3.0]})
        empty_daily = pd.DataFrame(columns=["日期", "沿海六大电库存", "日耗", "存煤可用天数"])
        empty_index = pd.DataFrame(columns=["日期", "最新值", "涨跌幅"])
        empty_freight = pd.DataFrame(columns=["截止日期", "波罗的海综合运价指数BDI", "油轮运价指数成品油运价指数BCTI", "油轮运价指数原油运价指数BDTI", "波罗的海超级大灵便型船BSI指数"])
        empty_oil = pd.DataFrame(columns=["调整日期", "汽油价格", "柴油价格", "汽油涨跌", "柴油涨跌"])
        empty_society = pd.DataFrame(columns=["统计时间", "全社会用电量", "全社会用电量同比", "第二产业用电量", "第二产业用电量同比"])
        new_house = pd.DataFrame(
            {
                "日期": pd.to_datetime(["2026-02-01", "2026-02-01"]),
                "城市": ["北京", "上海"],
                "新建商品住宅价格指数-同比": [97.7, 104.2],
                "新建商品住宅价格指数-环比": [100.2, 100.2],
                "二手住宅价格指数-同比": [91.6, 93.8],
                "二手住宅价格指数-环比": [100.3, 100.2],
            }
        )

        with (
            patch.object(PublicInternetResearchProvider, "_fetch_jin10_indicator", side_effect=[cpi, ppi]),
            patch.object(PublicInternetResearchProvider, "_fetch_lpr_table", return_value=lpr),
            patch.object(PublicInternetResearchProvider, "_fetch_daily_energy", return_value=empty_daily),
            patch.object(PublicInternetResearchProvider, "_fetch_commodity_price_index", return_value=empty_index),
            patch.object(PublicInternetResearchProvider, "_fetch_energy_index", return_value=empty_index),
            patch.object(PublicInternetResearchProvider, "_fetch_freight_index", return_value=empty_freight),
            patch.object(PublicInternetResearchProvider, "_fetch_oil_hist", return_value=empty_oil),
            patch.object(PublicInternetResearchProvider, "_fetch_construction_index", return_value=empty_index),
            patch.object(PublicInternetResearchProvider, "_fetch_construction_price_index", return_value=empty_index),
            patch.object(PublicInternetResearchProvider, "_fetch_real_estate_index", return_value=empty_index),
            patch.object(PublicInternetResearchProvider, "_fetch_society_electricity", return_value=empty_society),
            patch.object(PublicInternetResearchProvider, "_fetch_new_house_price", return_value=new_house),
        ):
            out = provider.fetch_macro(start=date(2026, 2, 1), end=date(2026, 3, 31))

        self.assertIn("new_house_price_yoy", out.columns)
        self.assertIn("new_house_price_mom", out.columns)
        self.assertIn("resale_house_price_yoy", out.columns)
        self.assertIn("resale_house_price_mom", out.columns)
        row = out[out["date"] == pd.Timestamp("2026-02-01")]
        self.assertEqual(len(row), 1)
        self.assertAlmostEqual(float(row.iloc[0]["new_house_price_yoy"]), (97.7 + 104.2) / 2, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["new_house_price_mom"]), 100.2, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["resale_house_price_yoy"]), (91.6 + 93.8) / 2, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["resale_house_price_mom"]), (100.3 + 100.2) / 2, places=6)

    def test_fetch_macro_includes_activity_trade_and_pmi_proxies(self) -> None:
        provider = PublicInternetResearchProvider()
        cpi = pd.DataFrame({"日期": pd.to_datetime(["2025-08-09"]), "今值": [0.0], "预测值": [-0.1], "前值": [0.1]})
        ppi = pd.DataFrame({"日期": pd.to_datetime(["2025-08-09"]), "今值": [-3.6], "预测值": [-3.4], "前值": [-3.6]})
        lpr = pd.DataFrame({"TRADE_DATE": pd.to_datetime(["2025-08-20"]), "LPR1Y": [3.0]})
        empty_daily = pd.DataFrame(columns=["日期", "沿海六大电库存", "日耗", "存煤可用天数"])
        empty_index = pd.DataFrame(columns=["日期", "最新值", "涨跌幅"])
        empty_freight = pd.DataFrame(columns=["截止日期", "波罗的海综合运价指数BDI", "油轮运价指数成品油运价指数BCTI", "油轮运价指数原油运价指数BDTI", "波罗的海超级大灵便型船BSI指数"])
        empty_oil = pd.DataFrame(columns=["调整日期", "汽油价格", "柴油价格", "汽油涨跌", "柴油涨跌"])
        empty_society = pd.DataFrame(columns=["统计时间", "全社会用电量", "全社会用电量同比", "第二产业用电量", "第二产业用电量同比"])
        empty_new_house = pd.DataFrame(columns=["日期", "城市", "新建商品住宅价格指数-同比", "新建商品住宅价格指数-环比", "二手住宅价格指数-同比", "二手住宅价格指数-环比"])
        industrial = pd.DataFrame({"日期": pd.to_datetime(["2025-08-15"]), "今值": [5.7]})
        exports = pd.DataFrame({"日期": pd.to_datetime(["2025-08-07"]), "今值": [7.2]})
        imports = pd.DataFrame({"日期": pd.to_datetime(["2025-08-07"]), "今值": [4.1]})
        pmi = pd.DataFrame({"日期": pd.to_datetime(["2025-08-31"]), "今值": [49.4]})
        non_man = pd.DataFrame({"日期": pd.to_datetime(["2025-08-31"]), "今值": [50.3]})

        with (
            patch.object(PublicInternetResearchProvider, "_fetch_jin10_indicator", side_effect=[cpi, ppi]),
            patch.object(PublicInternetResearchProvider, "_fetch_lpr_table", return_value=lpr),
            patch.object(PublicInternetResearchProvider, "_fetch_daily_energy", return_value=empty_daily),
            patch.object(PublicInternetResearchProvider, "_fetch_commodity_price_index", return_value=empty_index),
            patch.object(PublicInternetResearchProvider, "_fetch_energy_index", return_value=empty_index),
            patch.object(PublicInternetResearchProvider, "_fetch_freight_index", return_value=empty_freight),
            patch.object(PublicInternetResearchProvider, "_fetch_oil_hist", return_value=empty_oil),
            patch.object(PublicInternetResearchProvider, "_fetch_construction_index", return_value=empty_index),
            patch.object(PublicInternetResearchProvider, "_fetch_construction_price_index", return_value=empty_index),
            patch.object(PublicInternetResearchProvider, "_fetch_real_estate_index", return_value=empty_index),
            patch.object(PublicInternetResearchProvider, "_fetch_society_electricity", return_value=empty_society),
            patch.object(PublicInternetResearchProvider, "_fetch_new_house_price", return_value=empty_new_house),
            patch.object(PublicInternetResearchProvider, "_fetch_industrial_production_yoy", return_value=industrial),
            patch.object(PublicInternetResearchProvider, "_fetch_exports_yoy", return_value=exports),
            patch.object(PublicInternetResearchProvider, "_fetch_imports_yoy", return_value=imports),
            patch.object(PublicInternetResearchProvider, "_fetch_pmi_manufacturing", return_value=pmi),
            patch.object(PublicInternetResearchProvider, "_fetch_non_man_pmi", return_value=non_man),
        ):
            out = provider.fetch_macro(start=date(2025, 8, 1), end=date(2025, 8, 31))

        self.assertIn("industrial_production_yoy", out.columns)
        self.assertIn("exports_yoy", out.columns)
        self.assertIn("imports_yoy", out.columns)
        self.assertIn("pmi_manufacturing", out.columns)
        self.assertIn("pmi_non_manufacturing", out.columns)
        row = out[out["date"] == pd.Timestamp("2025-08-31")]
        self.assertEqual(len(row), 1)
        self.assertAlmostEqual(float(row.iloc[0]["industrial_production_yoy"]), 5.7, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["exports_yoy"]), 7.2, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["imports_yoy"]), 4.1, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["pmi_manufacturing"]), 49.4, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["pmi_non_manufacturing"]), 50.3, places=6)

    def test_fetch_macro_includes_asphalt_inventory_proxy(self) -> None:
        provider = PublicInternetResearchProvider()
        cpi = pd.DataFrame({"日期": pd.to_datetime(["2026-03-09"]), "今值": [0.2], "预测值": [0.1], "前值": [0.0]})
        ppi = pd.DataFrame({"日期": pd.to_datetime(["2026-03-09"]), "今值": [-2.2], "预测值": [-2.1], "前值": [-2.3]})
        lpr = pd.DataFrame({"TRADE_DATE": pd.to_datetime(["2026-03-20"]), "LPR1Y": [3.0]})
        empty_daily = pd.DataFrame(columns=["日期", "沿海六大电库存", "日耗", "存煤可用天数"])
        empty_index = pd.DataFrame(columns=["日期", "最新值", "涨跌幅"])
        empty_freight = pd.DataFrame(columns=["截止日期", "波罗的海综合运价指数BDI", "油轮运价指数成品油运价指数BCTI", "油轮运价指数原油运价指数BDTI", "波罗的海超级大灵便型船BSI指数"])
        empty_oil = pd.DataFrame(columns=["调整日期", "汽油价格", "柴油价格", "汽油涨跌", "柴油涨跌"])
        empty_society = pd.DataFrame(columns=["统计时间", "全社会用电量", "全社会用电量同比", "第二产业用电量", "第二产业用电量同比"])
        empty_new_house = pd.DataFrame(columns=["日期", "城市", "新建商品住宅价格指数-同比", "新建商品住宅价格指数-环比", "二手住宅价格指数-同比", "二手住宅价格指数-环比"])
        empty_value = pd.DataFrame(columns=["日期", "今值"])
        inventory = pd.DataFrame(
            {
                "日期": pd.to_datetime(["2026-03-24", "2026-03-25"]),
                "库存": [4180.0, 4205.0],
                "增减": [0.0, 25.0],
            }
        )

        with (
            patch.object(PublicInternetResearchProvider, "_fetch_jin10_indicator", side_effect=[cpi, ppi]),
            patch.object(PublicInternetResearchProvider, "_fetch_lpr_table", return_value=lpr),
            patch.object(PublicInternetResearchProvider, "_fetch_daily_energy", return_value=empty_daily),
            patch.object(PublicInternetResearchProvider, "_fetch_commodity_price_index", return_value=empty_index),
            patch.object(PublicInternetResearchProvider, "_fetch_energy_index", return_value=empty_index),
            patch.object(PublicInternetResearchProvider, "_fetch_freight_index", return_value=empty_freight),
            patch.object(PublicInternetResearchProvider, "_fetch_oil_hist", return_value=empty_oil),
            patch.object(PublicInternetResearchProvider, "_fetch_construction_index", return_value=empty_index),
            patch.object(PublicInternetResearchProvider, "_fetch_construction_price_index", return_value=empty_index),
            patch.object(PublicInternetResearchProvider, "_fetch_real_estate_index", return_value=empty_index),
            patch.object(PublicInternetResearchProvider, "_fetch_society_electricity", return_value=empty_society),
            patch.object(PublicInternetResearchProvider, "_fetch_new_house_price", return_value=empty_new_house),
            patch.object(PublicInternetResearchProvider, "_fetch_industrial_production_yoy", return_value=empty_value),
            patch.object(PublicInternetResearchProvider, "_fetch_exports_yoy", return_value=empty_value),
            patch.object(PublicInternetResearchProvider, "_fetch_imports_yoy", return_value=empty_value),
            patch.object(PublicInternetResearchProvider, "_fetch_pmi_manufacturing", return_value=empty_value),
            patch.object(PublicInternetResearchProvider, "_fetch_non_man_pmi", return_value=empty_value),
            patch.object(PublicInternetResearchProvider, "_fetch_asphalt_inventory", return_value=inventory),
        ):
            out = provider.fetch_macro(start=date(2026, 3, 1), end=date(2026, 3, 31))

        self.assertIn("asphalt_inventory", out.columns)
        self.assertIn("asphalt_inventory_delta", out.columns)
        row = out[out["date"] == pd.Timestamp("2026-03-25")]
        self.assertEqual(len(row), 1)
        self.assertAlmostEqual(float(row.iloc[0]["asphalt_inventory"]), 4205.0, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["asphalt_inventory_delta"]), 25.0, places=6)

    def test_fetch_macro_includes_chain_inventory_proxies(self) -> None:
        provider = PublicInternetResearchProvider()
        cpi = pd.DataFrame({"日期": pd.to_datetime(["2026-03-09"]), "今值": [0.2], "预测值": [0.1], "前值": [0.0]})
        ppi = pd.DataFrame({"日期": pd.to_datetime(["2026-03-09"]), "今值": [-2.2], "预测值": [-2.1], "前值": [-2.3]})
        lpr = pd.DataFrame({"TRADE_DATE": pd.to_datetime(["2026-03-20"]), "LPR1Y": [3.0]})
        empty_daily = pd.DataFrame(columns=["日期", "沿海六大电库存", "日耗", "存煤可用天数"])
        empty_index = pd.DataFrame(columns=["日期", "最新值", "涨跌幅"])
        empty_freight = pd.DataFrame(columns=["截止日期", "波罗的海综合运价指数BDI", "油轮运价指数成品油运价指数BCTI", "油轮运价指数原油运价指数BDTI", "波罗的海超级大灵便型船BSI指数"])
        empty_oil = pd.DataFrame(columns=["调整日期", "汽油价格", "柴油价格", "汽油涨跌", "柴油涨跌"])
        empty_society = pd.DataFrame(columns=["统计时间", "全社会用电量", "全社会用电量同比", "第二产业用电量", "第二产业用电量同比"])
        empty_new_house = pd.DataFrame(columns=["日期", "城市", "新建商品住宅价格指数-同比", "新建商品住宅价格指数-环比", "二手住宅价格指数-同比", "二手住宅价格指数-环比"])
        empty_value = pd.DataFrame(columns=["日期", "今值"])
        inventory = pd.DataFrame(columns=["日期", "库存", "增减"])
        lfu = pd.DataFrame({"日期": pd.to_datetime(["2026-03-27"]), "库存": [51960.0], "增减": [-3210.0]})
        fu = pd.DataFrame({"日期": pd.to_datetime(["2025-12-31"]), "库存": [201390.0], "增减": [0.0]})
        rb = pd.DataFrame({"日期": pd.to_datetime(["2026-03-27"]), "库存": [83113.0], "增减": [1525.0]})
        hc = pd.DataFrame({"日期": pd.to_datetime(["2026-03-27"]), "库存": [539561.0], "增减": [5882.0]})

        with ExitStack() as stack:
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_jin10_indicator", side_effect=[cpi, ppi]))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_lpr_table", return_value=lpr))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_daily_energy", return_value=empty_daily))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_commodity_price_index", return_value=empty_index))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_energy_index", return_value=empty_index))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_freight_index", return_value=empty_freight))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_oil_hist", return_value=empty_oil))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_construction_index", return_value=empty_index))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_construction_price_index", return_value=empty_index))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_real_estate_index", return_value=empty_index))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_society_electricity", return_value=empty_society))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_new_house_price", return_value=empty_new_house))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_industrial_production_yoy", return_value=empty_value))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_exports_yoy", return_value=empty_value))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_imports_yoy", return_value=empty_value))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_pmi_manufacturing", return_value=empty_value))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_non_man_pmi", return_value=empty_value))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_asphalt_inventory", return_value=inventory))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_lfu_inventory", return_value=lfu))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_fuel_oil_inventory", return_value=fu))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_rebar_inventory", return_value=rb))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_hotcoil_inventory", return_value=hc))
            out = provider.fetch_macro(start=date(2025, 12, 1), end=date(2026, 3, 31))

        self.assertIn("lfu_inventory", out.columns)
        self.assertIn("fuel_oil_inventory", out.columns)
        self.assertIn("rebar_inventory", out.columns)
        self.assertIn("hotcoil_inventory", out.columns)
        row = out[out["date"] == pd.Timestamp("2026-03-27")]
        self.assertEqual(len(row), 1)
        self.assertAlmostEqual(float(row.iloc[0]["lfu_inventory"]), 51960.0, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["rebar_inventory"]), 83113.0, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["hotcoil_inventory"]), 539561.0, places=6)

    def test_fetch_macro_includes_upstream_black_chain_inventory_proxies(self) -> None:
        provider = PublicInternetResearchProvider()
        cpi = pd.DataFrame({"日期": pd.to_datetime(["2026-03-09"]), "今值": [0.2], "预测值": [0.1], "前值": [0.0]})
        ppi = pd.DataFrame({"日期": pd.to_datetime(["2026-03-09"]), "今值": [-2.2], "预测值": [-2.1], "前值": [-2.3]})
        lpr = pd.DataFrame({"TRADE_DATE": pd.to_datetime(["2026-03-20"]), "LPR1Y": [3.0]})
        empty_daily = pd.DataFrame(columns=["日期", "沿海六大电库存", "日耗", "存煤可用天数"])
        empty_index = pd.DataFrame(columns=["日期", "最新值", "涨跌幅"])
        empty_freight = pd.DataFrame(columns=["截止日期", "波罗的海综合运价指数BDI", "油轮运价指数成品油运价指数BCTI", "油轮运价指数原油运价指数BDTI", "波罗的海超级大灵便型船BSI指数"])
        empty_oil = pd.DataFrame(columns=["调整日期", "汽油价格", "柴油价格", "汽油涨跌", "柴油涨跌"])
        empty_society = pd.DataFrame(columns=["统计时间", "全社会用电量", "全社会用电量同比", "第二产业用电量", "第二产业用电量同比"])
        empty_new_house = pd.DataFrame(columns=["日期", "城市", "新建商品住宅价格指数-同比", "新建商品住宅价格指数-环比", "二手住宅价格指数-同比", "二手住宅价格指数-环比"])
        empty_value = pd.DataFrame(columns=["日期", "今值"])
        empty_inv = pd.DataFrame(columns=["日期", "库存", "增减"])
        jm = pd.DataFrame({"日期": pd.to_datetime(["2026-03-19"]), "库存": [0.0], "增减": [-300.0]})
        j = pd.DataFrame({"日期": pd.to_datetime(["2026-03-27"]), "库存": [1060.0], "增减": [0.0]})
        i = pd.DataFrame({"日期": pd.to_datetime(["2026-03-27"]), "库存": [3200.0], "增减": [0.0]})

        with ExitStack() as stack:
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_jin10_indicator", side_effect=[cpi, ppi]))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_lpr_table", return_value=lpr))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_daily_energy", return_value=empty_daily))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_commodity_price_index", return_value=empty_index))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_energy_index", return_value=empty_index))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_freight_index", return_value=empty_freight))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_oil_hist", return_value=empty_oil))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_construction_index", return_value=empty_index))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_construction_price_index", return_value=empty_index))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_real_estate_index", return_value=empty_index))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_society_electricity", return_value=empty_society))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_new_house_price", return_value=empty_new_house))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_industrial_production_yoy", return_value=empty_value))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_exports_yoy", return_value=empty_value))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_imports_yoy", return_value=empty_value))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_pmi_manufacturing", return_value=empty_value))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_non_man_pmi", return_value=empty_value))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_asphalt_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_lfu_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_fuel_oil_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_rebar_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_hotcoil_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_coking_coal_inventory", return_value=jm))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_coke_inventory", return_value=j))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_iron_ore_inventory", return_value=i))
            out = provider.fetch_macro(start=date(2026, 3, 1), end=date(2026, 3, 31))

        self.assertIn("coking_coal_inventory", out.columns)
        self.assertIn("coke_inventory", out.columns)
        self.assertIn("iron_ore_inventory", out.columns)
        row = out[out["date"] == pd.Timestamp("2026-03-27")]
        self.assertEqual(len(row), 1)
        self.assertAlmostEqual(float(row.iloc[0]["coke_inventory"]), 1060.0, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["iron_ore_inventory"]), 3200.0, places=6)

    def test_fetch_macro_includes_chemical_and_building_inventory_proxies(self) -> None:
        provider = PublicInternetResearchProvider()
        cpi = pd.DataFrame({"日期": pd.to_datetime(["2026-03-09"]), "今值": [0.2], "预测值": [0.1], "前值": [0.0]})
        ppi = pd.DataFrame({"日期": pd.to_datetime(["2026-03-09"]), "今值": [-2.2], "预测值": [-2.1], "前值": [-2.3]})
        lpr = pd.DataFrame({"TRADE_DATE": pd.to_datetime(["2026-03-20"]), "LPR1Y": [3.0]})
        empty_daily = pd.DataFrame(columns=["日期", "沿海六大电库存", "日耗", "存煤可用天数"])
        empty_index = pd.DataFrame(columns=["日期", "最新值", "涨跌幅"])
        empty_freight = pd.DataFrame(columns=["截止日期", "波罗的海综合运价指数BDI", "油轮运价指数成品油运价指数BCTI", "油轮运价指数原油运价指数BDTI", "波罗的海超级大灵便型船BSI指数"])
        empty_oil = pd.DataFrame(columns=["调整日期", "汽油价格", "柴油价格", "汽油涨跌", "柴油涨跌"])
        empty_society = pd.DataFrame(columns=["统计时间", "全社会用电量", "全社会用电量同比", "第二产业用电量", "第二产业用电量同比"])
        empty_new_house = pd.DataFrame(columns=["日期", "城市", "新建商品住宅价格指数-同比", "新建商品住宅价格指数-环比", "二手住宅价格指数-同比", "二手住宅价格指数-环比"])
        empty_value = pd.DataFrame(columns=["日期", "今值"])
        empty_inv = pd.DataFrame(columns=["日期", "库存", "增减"])
        glass = pd.DataFrame({"日期": pd.to_datetime(["2026-03-27"]), "库存": [100.0], "增减": [0.0]})
        soda = pd.DataFrame({"日期": pd.to_datetime(["2026-03-27"]), "库存": [1363.0], "增减": [0.0]})
        pvc = pd.DataFrame({"日期": pd.to_datetime(["2026-03-27"]), "库存": [32344.0], "增减": [-2820.0]})
        pp = pd.DataFrame({"日期": pd.to_datetime(["2026-03-27"]), "库存": [7597.0], "增减": [-5416.0]})
        ma = pd.DataFrame({"日期": pd.to_datetime(["2026-03-27"]), "库存": [7835.0], "增减": [-146.0]})
        eg = pd.DataFrame({"日期": pd.to_datetime(["2026-03-27"]), "库存": [5146.0], "增减": [-1740.0]})
        pg = pd.DataFrame({"日期": pd.to_datetime(["2026-03-27"]), "库存": [1300.0], "增减": [0.0]})

        with ExitStack() as stack:
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_jin10_indicator", side_effect=[cpi, ppi]))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_lpr_table", return_value=lpr))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_daily_energy", return_value=empty_daily))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_commodity_price_index", return_value=empty_index))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_energy_index", return_value=empty_index))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_freight_index", return_value=empty_freight))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_oil_hist", return_value=empty_oil))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_construction_index", return_value=empty_index))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_construction_price_index", return_value=empty_index))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_real_estate_index", return_value=empty_index))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_society_electricity", return_value=empty_society))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_new_house_price", return_value=empty_new_house))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_industrial_production_yoy", return_value=empty_value))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_exports_yoy", return_value=empty_value))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_imports_yoy", return_value=empty_value))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_pmi_manufacturing", return_value=empty_value))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_non_man_pmi", return_value=empty_value))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_asphalt_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_lfu_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_fuel_oil_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_rebar_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_hotcoil_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_coking_coal_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_coke_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_iron_ore_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_glass_inventory", return_value=glass))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_soda_ash_inventory", return_value=soda))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_pvc_inventory", return_value=pvc))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_pp_inventory", return_value=pp))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_methanol_inventory", return_value=ma))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_eg_inventory", return_value=eg))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_lpg_inventory", return_value=pg))
            out = provider.fetch_macro(start=date(2026, 3, 1), end=date(2026, 3, 31))

        self.assertIn("glass_inventory", out.columns)
        self.assertIn("soda_ash_inventory", out.columns)
        self.assertIn("pvc_inventory", out.columns)
        self.assertIn("pp_inventory", out.columns)
        self.assertIn("methanol_inventory", out.columns)
        self.assertIn("eg_inventory", out.columns)
        self.assertIn("lpg_inventory", out.columns)
        row = out[out["date"] == pd.Timestamp("2026-03-27")]
        self.assertEqual(len(row), 1)
        self.assertAlmostEqual(float(row.iloc[0]["glass_inventory"]), 100.0, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["soda_ash_inventory"]), 1363.0, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["pvc_inventory"]), 32344.0, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["pp_inventory"]), 7597.0, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["methanol_inventory"]), 7835.0, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["eg_inventory"]), 5146.0, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["lpg_inventory"]), 1300.0, places=6)

    def test_fetch_macro_includes_society_traffic_proxies(self) -> None:
        provider = PublicInternetResearchProvider()
        cpi = pd.DataFrame({"日期": pd.to_datetime(["2026-03-09"]), "今值": [0.2], "预测值": [0.1], "前值": [0.0]})
        ppi = pd.DataFrame({"日期": pd.to_datetime(["2026-03-09"]), "今值": [-2.2], "预测值": [-2.1], "前值": [-2.3]})
        lpr = pd.DataFrame({"TRADE_DATE": pd.to_datetime(["2026-03-20"]), "LPR1Y": [3.0]})
        empty_daily = pd.DataFrame(columns=["日期", "沿海六大电库存", "日耗", "存煤可用天数"])
        empty_index = pd.DataFrame(columns=["日期", "最新值", "涨跌幅"])
        empty_freight = pd.DataFrame(columns=["截止日期", "波罗的海综合运价指数BDI", "油轮运价指数成品油运价指数BCTI", "油轮运价指数原油运价指数BDTI", "波罗的海超级大灵便型船BSI指数"])
        empty_oil = pd.DataFrame(columns=["调整日期", "汽油价格", "柴油价格", "汽油涨跌", "柴油涨跌"])
        empty_society = pd.DataFrame(columns=["统计时间", "全社会用电量", "全社会用电量同比", "第二产业用电量", "第二产业用电量同比"])
        empty_new_house = pd.DataFrame(columns=["日期", "城市", "新建商品住宅价格指数-同比", "新建商品住宅价格指数-环比", "二手住宅价格指数-同比", "二手住宅价格指数-环比"])
        empty_value = pd.DataFrame(columns=["日期", "今值"])
        empty_inv = pd.DataFrame(columns=["日期", "库存", "增减"])
        traffic = pd.DataFrame(
            {
                "统计时间": ["2026.2"],
                "货运量": [475214.0],
                "货运量同比增长": [4.5],
                "货物周转量": [43210.0],
                "公里货物周转量同比增长": [5.1],
                "沿海主要港口货物吞吐量": [87654.0],
                "沿海主要港口货物吞吐量同比增长": [3.4],
                "其中:外贸货物吞吐量": [23456.0],
                "其中:外贸货物吞吐量同比增长": [2.2],
            }
        )

        with ExitStack() as stack:
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_jin10_indicator", side_effect=[cpi, ppi]))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_lpr_table", return_value=lpr))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_daily_energy", return_value=empty_daily))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_commodity_price_index", return_value=empty_index))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_energy_index", return_value=empty_index))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_freight_index", return_value=empty_freight))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_oil_hist", return_value=empty_oil))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_construction_index", return_value=empty_index))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_construction_price_index", return_value=empty_index))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_real_estate_index", return_value=empty_index))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_society_electricity", return_value=empty_society))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_new_house_price", return_value=empty_new_house))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_industrial_production_yoy", return_value=empty_value))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_exports_yoy", return_value=empty_value))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_imports_yoy", return_value=empty_value))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_pmi_manufacturing", return_value=empty_value))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_non_man_pmi", return_value=empty_value))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_asphalt_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_lfu_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_fuel_oil_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_rebar_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_hotcoil_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_coking_coal_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_coke_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_iron_ore_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_glass_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_soda_ash_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_pvc_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_pp_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_methanol_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_eg_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_lpg_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_society_traffic_volume", return_value=traffic))
            out = provider.fetch_macro(start=date(2026, 2, 1), end=date(2026, 3, 31))

        self.assertIn("cargo_volume", out.columns)
        self.assertIn("cargo_volume_yoy", out.columns)
        self.assertIn("cargo_turnover", out.columns)
        self.assertIn("cargo_turnover_yoy", out.columns)
        self.assertIn("coastal_port_throughput", out.columns)
        self.assertIn("coastal_port_throughput_yoy", out.columns)
        self.assertIn("coastal_port_foreign_trade_throughput", out.columns)
        self.assertIn("coastal_port_foreign_trade_throughput_yoy", out.columns)
        row = out[out["date"] == pd.Timestamp("2026-02-01")]
        self.assertEqual(len(row), 1)
        self.assertAlmostEqual(float(row.iloc[0]["cargo_volume"]), 475214.0, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["cargo_volume_yoy"]), 4.5, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["cargo_turnover"]), 43210.0, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["cargo_turnover_yoy"]), 5.1, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["coastal_port_throughput"]), 87654.0, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["coastal_port_foreign_trade_throughput"]), 23456.0, places=6)

    def test_fetch_macro_includes_passenger_and_postal_proxies(self) -> None:
        provider = PublicInternetResearchProvider()
        cpi = pd.DataFrame({"日期": pd.to_datetime(["2026-03-09"]), "今值": [0.2], "预测值": [0.1], "前值": [0.0]})
        ppi = pd.DataFrame({"日期": pd.to_datetime(["2026-03-09"]), "今值": [-2.2], "预测值": [-2.1], "前值": [-2.3]})
        lpr = pd.DataFrame({"TRADE_DATE": pd.to_datetime(["2026-03-20"]), "LPR1Y": [3.0]})
        empty_daily = pd.DataFrame(columns=["日期", "沿海六大电库存", "日耗", "存煤可用天数"])
        empty_index = pd.DataFrame(columns=["日期", "最新值", "涨跌幅"])
        empty_freight = pd.DataFrame(columns=["截止日期", "波罗的海综合运价指数BDI", "油轮运价指数成品油运价指数BCTI", "油轮运价指数原油运价指数BDTI", "波罗的海超级大灵便型船BSI指数"])
        empty_oil = pd.DataFrame(columns=["调整日期", "汽油价格", "柴油价格", "汽油涨跌", "柴油涨跌"])
        empty_society = pd.DataFrame(columns=["统计时间", "全社会用电量", "全社会用电量同比", "第二产业用电量", "第二产业用电量同比"])
        empty_new_house = pd.DataFrame(columns=["日期", "城市", "新建商品住宅价格指数-同比", "新建商品住宅价格指数-环比", "二手住宅价格指数-同比", "二手住宅价格指数-环比"])
        empty_value = pd.DataFrame(columns=["日期", "今值"])
        empty_inv = pd.DataFrame(columns=["日期", "库存", "增减"])
        passenger = pd.DataFrame({"统计时间": ["2026.2"], "客座率": [87.2], "载运率": [72.0]})
        postal = pd.DataFrame({"统计时间": ["2026.2"], "特快专递": [1211000.0], "特快专递同比增长": [-10.9], "电信业务总量": [1800.0], "电信业务总量同比增长": [9.1]})

        with ExitStack() as stack:
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_jin10_indicator", side_effect=[cpi, ppi]))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_lpr_table", return_value=lpr))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_daily_energy", return_value=empty_daily))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_commodity_price_index", return_value=empty_index))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_energy_index", return_value=empty_index))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_freight_index", return_value=empty_freight))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_oil_hist", return_value=empty_oil))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_oil_detail", return_value=pd.DataFrame(columns=["日期", "地区", "V_0", "V_92", "V_95"])))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_construction_index", return_value=empty_index))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_construction_price_index", return_value=empty_index))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_real_estate_index", return_value=empty_index))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_society_electricity", return_value=empty_society))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_new_house_price", return_value=empty_new_house))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_industrial_production_yoy", return_value=empty_value))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_exports_yoy", return_value=empty_value))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_imports_yoy", return_value=empty_value))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_pmi_manufacturing", return_value=empty_value))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_non_man_pmi", return_value=empty_value))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_asphalt_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_lfu_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_fuel_oil_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_rebar_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_hotcoil_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_coking_coal_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_coke_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_iron_ore_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_glass_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_soda_ash_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_pvc_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_pp_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_methanol_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_eg_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_lpg_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_society_traffic_volume", return_value=pd.DataFrame(columns=["统计时间"])))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_money_supply", return_value=pd.DataFrame(columns=["月份"])))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_new_financial_credit", return_value=pd.DataFrame(columns=["月份"])))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_bank_financing_index", return_value=pd.DataFrame(columns=["日期"])))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_fx_reserves", return_value=pd.DataFrame(columns=["日期"])))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_passenger_load_factor", return_value=passenger))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_postal_telecom", return_value=postal))
            out = provider.fetch_macro(start=date(2026, 2, 1), end=date(2026, 3, 31))

        self.assertIn("passenger_load_factor", out.columns)
        self.assertIn("cargo_load_factor", out.columns)
        self.assertIn("express_delivery_volume", out.columns)
        self.assertIn("express_delivery_yoy", out.columns)
        self.assertIn("telecom_business_total", out.columns)
        self.assertIn("telecom_business_yoy", out.columns)
        row = out[out["date"] == pd.Timestamp("2026-02-01")]
        self.assertEqual(len(row), 1)
        self.assertAlmostEqual(float(row.iloc[0]["passenger_load_factor"]), 87.2, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["cargo_load_factor"]), 72.0, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["express_delivery_volume"]), 1211000.0, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["express_delivery_yoy"]), -10.9, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["telecom_business_total"]), 1800.0, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["telecom_business_yoy"]), 9.1, places=6)

    def test_fetch_macro_includes_liquidity_and_credit_proxies(self) -> None:
        provider = PublicInternetResearchProvider()
        cpi = pd.DataFrame({"日期": pd.to_datetime(["2026-03-09"]), "今值": [0.2], "预测值": [0.1], "前值": [0.0]})
        ppi = pd.DataFrame({"日期": pd.to_datetime(["2026-03-09"]), "今值": [-2.2], "预测值": [-2.1], "前值": [-2.3]})
        lpr = pd.DataFrame({"TRADE_DATE": pd.to_datetime(["2026-03-20"]), "LPR1Y": [3.0]})
        empty_daily = pd.DataFrame(columns=["日期", "沿海六大电库存", "日耗", "存煤可用天数"])
        empty_index = pd.DataFrame(columns=["日期", "最新值", "涨跌幅"])
        empty_freight = pd.DataFrame(columns=["截止日期", "波罗的海综合运价指数BDI", "油轮运价指数成品油运价指数BCTI", "油轮运价指数原油运价指数BDTI", "波罗的海超级大灵便型船BSI指数"])
        empty_oil = pd.DataFrame(columns=["调整日期", "汽油价格", "柴油价格", "汽油涨跌", "柴油涨跌"])
        empty_society = pd.DataFrame(columns=["统计时间", "全社会用电量", "全社会用电量同比", "第二产业用电量", "第二产业用电量同比"])
        empty_new_house = pd.DataFrame(columns=["日期", "城市", "新建商品住宅价格指数-同比", "新建商品住宅价格指数-环比", "二手住宅价格指数-同比", "二手住宅价格指数-环比"])
        empty_value = pd.DataFrame(columns=["日期", "今值"])
        empty_inv = pd.DataFrame(columns=["日期", "库存", "增减"])
        money_supply = pd.DataFrame(
            {
                "月份": ["2026年02月份"],
                "货币和准货币(M2)-数量(亿元)": [3492159.91],
                "货币和准货币(M2)-同比增长": [9.0],
                "货币(M1)-数量(亿元)": [1159258.82],
                "货币(M1)-同比增长": [5.9],
                "流通中的现金(M0)-数量(亿元)": [151436.41],
                "流通中的现金(M0)-同比增长": [14.1],
            }
        )
        new_credit = pd.DataFrame({"月份": ["2026年02月份"], "当月": [8458.0], "当月-同比增长": [29.564951], "累计": [57474.0]})
        bank_financing = pd.DataFrame({"日期": pd.to_datetime(["2026-02-01"]), "最新值": [2106.0], "涨跌幅": [-22.772277]})
        fx_reserves = pd.DataFrame({"日期": pd.to_datetime(["2025-08-07"]), "今值": [32920.0]})

        with ExitStack() as stack:
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_jin10_indicator", side_effect=[cpi, ppi]))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_lpr_table", return_value=lpr))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_daily_energy", return_value=empty_daily))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_commodity_price_index", return_value=empty_index))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_energy_index", return_value=empty_index))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_freight_index", return_value=empty_freight))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_oil_hist", return_value=empty_oil))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_construction_index", return_value=empty_index))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_construction_price_index", return_value=empty_index))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_real_estate_index", return_value=empty_index))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_society_electricity", return_value=empty_society))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_new_house_price", return_value=empty_new_house))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_industrial_production_yoy", return_value=empty_value))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_exports_yoy", return_value=empty_value))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_imports_yoy", return_value=empty_value))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_pmi_manufacturing", return_value=empty_value))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_non_man_pmi", return_value=empty_value))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_asphalt_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_lfu_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_fuel_oil_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_rebar_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_hotcoil_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_coking_coal_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_coke_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_iron_ore_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_glass_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_soda_ash_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_pvc_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_pp_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_methanol_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_eg_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_lpg_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_society_traffic_volume", return_value=pd.DataFrame(columns=['统计时间','货运量','货运量同比增长','货物周转量','公里货物周转量同比增长','沿海主要港口货物吞吐量','沿海主要港口货物吞吐量同比增长','其中:外贸货物吞吐量','其中:外贸货物吞吐量同比增长'])))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_money_supply", return_value=money_supply))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_new_financial_credit", return_value=new_credit))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_bank_financing_index", return_value=bank_financing))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_fx_reserves", return_value=fx_reserves))
            out = provider.fetch_macro(start=date(2026, 2, 1), end=date(2026, 3, 31))

        self.assertIn("m2_level", out.columns)
        self.assertIn("m2_yoy", out.columns)
        self.assertIn("m1_level", out.columns)
        self.assertIn("m1_yoy", out.columns)
        self.assertIn("m0_level", out.columns)
        self.assertIn("m0_yoy", out.columns)
        self.assertIn("new_financial_credit_monthly", out.columns)
        self.assertIn("new_financial_credit_yoy", out.columns)
        self.assertIn("new_financial_credit_cum", out.columns)
        self.assertIn("bank_financing_index", out.columns)
        self.assertIn("bank_financing_index_pct_chg", out.columns)
        self.assertIn("fx_reserves", out.columns)
        row = out[out["date"] == pd.Timestamp("2026-02-01")]
        self.assertEqual(len(row), 1)
        self.assertAlmostEqual(float(row.iloc[0]["m2_level"]), 3492159.91, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["m2_yoy"]), 9.0, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["m1_level"]), 1159258.82, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["m1_yoy"]), 5.9, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["m0_level"]), 151436.41, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["m0_yoy"]), 14.1, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["new_financial_credit_monthly"]), 8458.0, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["new_financial_credit_yoy"]), 29.564951, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["new_financial_credit_cum"]), 57474.0, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["bank_financing_index"]), 2106.0, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["bank_financing_index_pct_chg"]), -22.772277, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["fx_reserves"]), 32920.0, places=6)

    def test_fetch_macro_includes_demand_price_and_confidence_proxies(self) -> None:
        provider = PublicInternetResearchProvider()
        cpi = pd.DataFrame({"日期": pd.to_datetime(["2026-03-09"]), "今值": [0.2], "预测值": [0.1], "前值": [0.0]})
        ppi = pd.DataFrame({"日期": pd.to_datetime(["2026-03-09"]), "今值": [-2.2], "预测值": [-2.1], "前值": [-2.3]})
        lpr = pd.DataFrame({"TRADE_DATE": pd.to_datetime(["2026-03-20"]), "LPR1Y": [3.0]})
        empty_daily = pd.DataFrame(columns=["日期", "沿海六大电库存", "日耗", "存煤可用天数"])
        empty_index = pd.DataFrame(columns=["日期", "最新值", "涨跌幅"])
        empty_freight = pd.DataFrame(columns=["截止日期", "波罗的海综合运价指数BDI", "油轮运价指数成品油运价指数BCTI", "油轮运价指数原油运价指数BDTI", "波罗的海超级大灵便型船BSI指数"])
        empty_oil = pd.DataFrame(columns=["调整日期", "汽油价格", "柴油价格", "汽油涨跌", "柴油涨跌"])
        empty_society = pd.DataFrame(columns=["统计时间", "全社会用电量", "全社会用电量同比", "第二产业用电量", "第二产业用电量同比"])
        empty_new_house = pd.DataFrame(columns=["日期", "城市", "新建商品住宅价格指数-同比", "新建商品住宅价格指数-环比", "二手住宅价格指数-同比", "二手住宅价格指数-环比"])
        empty_value = pd.DataFrame(columns=["日期", "今值"])
        empty_inv = pd.DataFrame(columns=["日期", "库存", "增减"])
        fixed_asset = pd.DataFrame(
            {
                "月份": ["2026年02月份"],
                "当月": [41200.0],
                "同比增长": [4.2],
                "环比增长": [1.1],
                "自年初累计": [52721.0],
            }
        )
        retail_sales = pd.DataFrame(
            {
                "月份": ["2026年02月份"],
                "当月": [43039.5],
                "同比增长": [3.2],
                "环比增长": [1.8],
                "累计": [86079.0],
                "累计-同比增长": [2.8],
            }
        )
        enterprise_goods_price = pd.DataFrame(
            {
                "月份": ["2026年02月份"],
                "总指数-指数值": [100.1],
                "总指数-同比增长": [0.7],
                "总指数-环比增长": [0.2],
                "煤油电-指数值": [94.1],
                "煤油电-同比增长": [-1.8],
                "煤油电-环比增长": [0.5],
            }
        )
        consumer_confidence = pd.DataFrame(
            {
                "月份": ["2026年02月份"],
                "消费者信心指数-指数值": [91.2],
                "消费者信心指数-同比增长": [3.9],
                "消费者信心指数-环比增长": [0.7],
                "消费者满意指数-指数值": [89.0],
                "消费者满意指数-同比增长": [4.1],
                "消费者满意指数-环比增长": [0.4],
                "消费者预期指数-指数值": [92.4],
                "消费者预期指数-同比增长": [4.2],
                "消费者预期指数-环比增长": [0.8],
            }
        )

        with ExitStack() as stack:
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_jin10_indicator", side_effect=[cpi, ppi]))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_lpr_table", return_value=lpr))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_daily_energy", return_value=empty_daily))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_commodity_price_index", return_value=empty_index))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_energy_index", return_value=empty_index))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_freight_index", return_value=empty_freight))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_oil_hist", return_value=empty_oil))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_oil_detail", return_value=pd.DataFrame(columns=["日期", "地区", "V_0", "V_92", "V_95"])))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_construction_index", return_value=empty_index))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_construction_price_index", return_value=empty_index))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_real_estate_index", return_value=empty_index))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_society_electricity", return_value=empty_society))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_new_house_price", return_value=empty_new_house))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_industrial_production_yoy", return_value=empty_value))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_exports_yoy", return_value=empty_value))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_imports_yoy", return_value=empty_value))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_pmi_manufacturing", return_value=empty_value))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_non_man_pmi", return_value=empty_value))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_asphalt_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_lfu_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_fuel_oil_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_rebar_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_hotcoil_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_coking_coal_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_coke_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_iron_ore_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_glass_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_soda_ash_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_pvc_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_pp_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_methanol_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_eg_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_lpg_inventory", return_value=empty_inv))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_society_traffic_volume", return_value=pd.DataFrame(columns=["统计时间"])))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_money_supply", return_value=pd.DataFrame(columns=["月份"])))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_new_financial_credit", return_value=pd.DataFrame(columns=["月份"])))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_bank_financing_index", return_value=pd.DataFrame(columns=["日期"])))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_fx_reserves", return_value=pd.DataFrame(columns=["日期"])))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_passenger_load_factor", return_value=pd.DataFrame(columns=["统计时间"])))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_postal_telecom", return_value=pd.DataFrame(columns=["统计时间"])))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_fixed_asset_investment", return_value=fixed_asset, create=True))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_consumer_goods_retail", return_value=retail_sales, create=True))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_enterprise_goods_price", return_value=enterprise_goods_price, create=True))
            stack.enter_context(patch.object(PublicInternetResearchProvider, "_fetch_consumer_confidence", return_value=consumer_confidence, create=True))
            out = provider.fetch_macro(start=date(2026, 2, 1), end=date(2026, 3, 31))

        expected_cols = [
            "fixed_asset_investment_monthly",
            "fixed_asset_investment_yoy",
            "fixed_asset_investment_cum",
            "retail_sales_monthly",
            "retail_sales_yoy",
            "retail_sales_cum",
            "retail_sales_cum_yoy",
            "enterprise_goods_price_index",
            "enterprise_goods_price_yoy",
            "energy_goods_price_index",
            "energy_goods_price_yoy",
            "consumer_confidence_index",
            "consumer_satisfaction_index",
            "consumer_expectation_index",
        ]
        for col in expected_cols:
            self.assertIn(col, out.columns)

        row = out[out["date"] == pd.Timestamp("2026-02-01")]
        self.assertEqual(len(row), 1)
        self.assertAlmostEqual(float(row.iloc[0]["fixed_asset_investment_monthly"]), 41200.0, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["fixed_asset_investment_yoy"]), 4.2, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["fixed_asset_investment_cum"]), 52721.0, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["retail_sales_monthly"]), 43039.5, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["retail_sales_cum_yoy"]), 2.8, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["enterprise_goods_price_index"]), 100.1, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["energy_goods_price_yoy"]), -1.8, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["consumer_confidence_index"]), 91.2, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["consumer_satisfaction_index"]), 89.0, places=6)
        self.assertAlmostEqual(float(row.iloc[0]["consumer_expectation_index"]), 92.4, places=6)

    def test_fetch_news_combines_baidu_calendar_and_shmet_flash_for_zh_only(self) -> None:
        provider = PublicInternetResearchProvider()
        baidu = pd.DataFrame(
            {
                "日期": [date(2026, 3, 30)],
                "时间": ["15:10"],
                "地区": ["中国"],
                "事件": ["中国3月30日上期所每日仓单变动-铜(吨)"],
                "公布": [None],
                "预期": [None],
                "前值": [-9365.0],
                "重要性": [2],
            }
        )
        shmet = pd.DataFrame(
            {
                "发布时间": [
                    pd.Timestamp("2026-03-30 01:52:06+08:00"),
                    pd.Timestamp("2026-03-28 23:59:00+08:00"),
                ],
                "内容": [
                    "【以媒：美官员称到下周初将有足够兵力来执行对伊朗的地面行动】SHMET03月30日讯，据报道所有选项都在考虑之中。",
                    "窗口外旧快讯",
                ],
            }
        )
        cctv = pd.DataFrame(
            {
                "date": ["20260330"],
                "title": ["今年前两个月我国电子商务稳定发展"],
                "content": ["数字消费持续活跃，产业电商成为增长主动力。"],
            }
        )

        def _fake_baidu_fetch(self, current_date: date) -> pd.DataFrame:  # type: ignore[no-untyped-def]
            if current_date == date(2026, 3, 30):
                return baidu
            return pd.DataFrame(columns=baidu.columns)

        def _fake_cctv_fetch(self, current_date: date) -> pd.DataFrame:  # type: ignore[no-untyped-def]
            if current_date == date(2026, 3, 30):
                return cctv
            return pd.DataFrame(columns=cctv.columns)

        with (
            patch.object(PublicInternetResearchProvider, "_fetch_baidu_calendar", new=_fake_baidu_fetch),
            patch.object(PublicInternetResearchProvider, "_fetch_shmet_flash", return_value=shmet),
            patch.object(PublicInternetResearchProvider, "_fetch_cctv_transcript", new=_fake_cctv_fetch),
        ):
            out = provider.fetch_news(
                start_ts=datetime(2026, 3, 29, 0, 0),
                end_ts=datetime(2026, 3, 30, 23, 59),
                lang="zh",
            )
            out_en = provider.fetch_news(
                start_ts=datetime(2026, 3, 29, 0, 0),
                end_ts=datetime(2026, 3, 30, 23, 59),
                lang="en",
            )

        self.assertEqual(out_en, [])
        self.assertEqual(len(out), 3)
        titles = {item.title for item in out}
        self.assertTrue(any("仓单变动" in title for title in titles))
        self.assertTrue(any("伊朗" in item.content for item in out))
        self.assertTrue(any("电子商务稳定发展" in title for title in titles))
        self.assertTrue(all(item.source == provider.name for item in out))
        self.assertTrue(all(item.lang == "zh" for item in out))
        self.assertTrue(all(0.0 <= float(item.confidence) <= 1.0 for item in out))


if __name__ == "__main__":
    unittest.main()
