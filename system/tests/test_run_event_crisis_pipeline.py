from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from lie_engine.research import run_event_crisis_pipeline


def _fake_row() -> dict[str, object]:
    return {
        "event_id": "evt-1",
        "event_ts": "2026-03-25T12:00:00Z",
        "event_classes": ["credit_deterioration"],
    }


def test_run_event_crisis_pipeline_writes_latest_artifacts(tmp_path: Path) -> None:
    artifacts = run_event_crisis_pipeline.run_pipeline(
        output_root=tmp_path,
        mode="snapshot",
        event_rows=[_fake_row()],
        market_inputs={"credit_liquidity_stress_score": 0.6},
        generated_at=dt.datetime(2026, 3, 25, 12, 0, tzinfo=dt.timezone.utc),
    )

    for path in artifacts.values():
        assert path.exists()
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)

