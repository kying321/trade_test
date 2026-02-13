from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lie_engine.config import SystemSettings, validate_settings


class ConfigValidationTests(unittest.TestCase):
    def test_validate_settings_ok(self) -> None:
        settings = SystemSettings(
            raw={
                "timezone": "Asia/Shanghai",
                "schedule": {
                    "premarket": "08:40",
                    "intraday_slots": ["10:30", "14:30"],
                    "eod": "15:10",
                    "nightly_review": "20:30",
                },
                "thresholds": {
                    "hurst_trend": 0.6,
                    "hurst_mean_revert": 0.4,
                    "atr_extreme": 2.0,
                    "signal_confidence_min": 60.0,
                    "convexity_min": 3.0,
                },
                "risk": {
                    "max_single_risk_pct": 2.0,
                    "max_total_exposure_pct": 50.0,
                    "max_symbol_pct": 15.0,
                    "max_theme_pct": 25.0,
                    "safety_bucket_pct": 85.0,
                    "convexity_bucket_pct": 15.0,
                },
                "validation": {
                    "data_completeness_min": 0.99,
                    "unresolved_conflict_max": 0.005,
                    "positive_window_ratio_min": 0.70,
                    "max_drawdown_max": 0.18,
                    "required_stable_replay_days": 3,
                    "cooldown_consecutive_losses": 3,
                    "major_event_window_hours": 24,
                    "factor_lookback_days": 120,
                },
                "universe": {
                    "core": [
                        {"symbol": "300750", "asset_class": "equity"},
                        {"symbol": "SC2603", "asset_class": "future"},
                    ]
                },
                "paths": {"output": "output", "sqlite": "output/artifacts/lie_engine.db"},
            }
        )
        out = validate_settings(settings)
        self.assertTrue(out["ok"])
        self.assertEqual(out["summary"]["errors"], 0)

    def test_validate_settings_detects_errors(self) -> None:
        settings = SystemSettings(
            raw={
                "timezone": "Invalid/Timezone",
                "schedule": {"premarket": "25:61", "intraday_slots": [], "eod": "aa:bb", "nightly_review": "23:00"},
                "thresholds": {"hurst_trend": 0.3, "hurst_mean_revert": 0.4, "atr_extreme": -1.0, "convexity_min": 0.0},
                "risk": {"max_single_risk_pct": 20, "max_symbol_pct": 10, "max_total_exposure_pct": 5},
                "validation": {"data_completeness_min": 1.1, "required_stable_replay_days": 0},
                "universe": {"core": [{"symbol": "BAD"}]},
                "paths": {"output": "", "sqlite": ""},
            }
        )
        out = validate_settings(settings)
        self.assertFalse(out["ok"])
        self.assertGreater(out["summary"]["errors"], 0)


if __name__ == "__main__":
    unittest.main()
