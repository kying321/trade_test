from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _load_stability_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_strategy_stability.py"
    spec = importlib.util.spec_from_file_location("run_strategy_stability_for_tests", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class StrategyStabilityScriptTests(unittest.TestCase):
    def test_parse_windows_unique_sorted(self) -> None:
        mod = _load_stability_module()
        out = mod._parse_windows("180,365,180")
        self.assertEqual(out, [365, 180])

    def test_param_drift_score_zero_on_equal(self) -> None:
        mod = _load_stability_module()
        base = {"hold_days": 8, "exposure_scale": 0.12}
        test = {"hold_days": 8, "exposure_scale": 0.12}
        score = float(mod._param_drift_score(base, test, ["hold_days", "exposure_scale"]))
        self.assertAlmostEqual(score, 0.0, places=8)

    def test_stability_score_prefers_consistent_rows(self) -> None:
        mod = _load_stability_module()
        consistent_rows = [
            {
                "best_accepted": True,
                "validation_max_drawdown": 0.03,
                "validation_annual_return": 0.08,
                "validation_trades": 12,
                "exposure_cap_applied": 0.10,
                "best_params": {"hold_days": 8, "exposure_scale": 0.10},
            },
            {
                "best_accepted": True,
                "validation_max_drawdown": 0.04,
                "validation_annual_return": 0.06,
                "validation_trades": 9,
                "exposure_cap_applied": 0.11,
                "best_params": {"hold_days": 8, "exposure_scale": 0.11},
            },
        ]
        unstable_rows = [
            {
                "best_accepted": False,
                "validation_max_drawdown": 0.15,
                "validation_annual_return": -0.03,
                "validation_trades": 0,
                "exposure_cap_applied": 0.08,
                "best_params": {"hold_days": 4, "exposure_scale": 0.08},
            },
            {
                "best_accepted": True,
                "validation_max_drawdown": 0.22,
                "validation_annual_return": -0.01,
                "validation_trades": 2,
                "exposure_cap_applied": 0.30,
                "best_params": {"hold_days": 16, "exposure_scale": 0.30},
            },
        ]
        stable = mod._stability_score(consistent_rows, 0.05)
        unstable = mod._stability_score(unstable_rows, 0.05)
        self.assertGreater(float(stable.get("score", 0.0)), float(unstable.get("score", 0.0)))


if __name__ == "__main__":
    unittest.main()
