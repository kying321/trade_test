from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lie_engine.regime import compute_atr_zscore, infer_hmm_state, latest_multi_scale_hurst
from tests.helpers import make_bars


class RegimeTests(unittest.TestCase):
    def test_hurst_trend_higher_than_random(self) -> None:
        rng = np.random.default_rng(1)
        eps1 = rng.normal(0.0, 1.0, 700)
        eps2 = rng.normal(0.0, 1.0, 700)

        # Positive-persistence AR(1) path vs anti-persistent path.
        r_pos = np.zeros(700)
        r_neg = np.zeros(700)
        for i in range(1, 700):
            r_pos[i] = 0.85 * r_pos[i - 1] + eps1[i]
            r_neg[i] = -0.65 * r_neg[i - 1] + eps2[i]

        trend_prices = 100 + np.cumsum(r_pos)
        mean_revert_prices = 100 + np.cumsum(r_neg)
        ht = latest_multi_scale_hurst(trend_prices)
        hr = latest_multi_scale_hurst(mean_revert_prices)
        self.assertGreater(ht, hr)
        self.assertGreaterEqual(ht, 0.50)

    def test_hmm_outputs_probabilities(self) -> None:
        bars = make_bars("300750", n=260)
        probs = infer_hmm_state(bars)
        self.assertIn("bull", probs)
        self.assertIn("range", probs)
        self.assertIn("bear", probs)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=5)

    def test_atr_zscore_finite(self) -> None:
        bars = make_bars("300750", n=200)
        z = compute_atr_zscore(bars)
        self.assertTrue(np.isfinite(z))


if __name__ == "__main__":
    unittest.main()
