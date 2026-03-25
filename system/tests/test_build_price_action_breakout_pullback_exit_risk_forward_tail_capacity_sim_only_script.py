from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_price_action_breakout_pullback_exit_risk_forward_tail_capacity_sim_only.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("fenlie_exit_risk_forward_tail_capacity_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_dataset(path: Path, *, days: int = 60, cadence_minutes: int = 15) -> None:
    periods = days * int((24 * 60) / cadence_minutes)
    ts = pd.date_range("2026-01-20T12:00:00Z", periods=periods, freq=f"{cadence_minutes}min", tz="UTC")
    rows = []
    price = 3000.0
    for index, stamp in enumerate(ts):
        open_px = price
        high_px = open_px * 1.001
        low_px = open_px * 0.999
        close_px = open_px * (1.0002 if index % 2 == 0 else 0.9998)
        rows.append(
            {
                "ts": stamp.isoformat().replace("+00:00", "Z"),
                "symbol": "ETHUSDT",
                "open": round(open_px, 6),
                "high": round(high_px, 6),
                "low": round(low_px, 6),
                "close": round(close_px, 6),
                "volume": 1000 + (index % 17),
            }
        )
        price = close_px
    pd.DataFrame(rows).to_csv(path, index=False)


def test_classify_capacity_decision_marks_45d_plus_as_requires_longer_dataset() -> None:
    module = _load_module()
    capacity_rows = [
        {"train_days": 30, "slice_count": 3},
        {"train_days": 40, "slice_count": 2},
        {"train_days": 45, "slice_count": 1},
        {"train_days": 50, "slice_count": 1},
        {"train_days": 55, "slice_count": 0},
        {"train_days": 60, "slice_count": 0},
    ]

    summary = module.summarize_capacity_rows(capacity_rows, min_credible_slices=2, min_robust_slices=3)

    assert summary == {
        "max_credible_train_days": 40,
        "max_robust_train_days": 30,
        "first_insufficient_train_days": 45,
        "zero_slice_train_days": [55, 60],
    }
    assert module.classify_capacity_decision(summary) == (
        "exit_risk_forward_tail_capacity_supports_40d_watch_45d_plus_requires_longer_dataset"
    )


def test_classify_capacity_decision_marks_current_grid_as_fully_covered() -> None:
    module = _load_module()
    capacity_rows = [
        {"train_days": 30, "slice_count": 6},
        {"train_days": 40, "slice_count": 5},
        {"train_days": 45, "slice_count": 4},
        {"train_days": 50, "slice_count": 4},
        {"train_days": 55, "slice_count": 3},
        {"train_days": 60, "slice_count": 3},
    ]

    summary = module.summarize_capacity_rows(capacity_rows, min_credible_slices=2, min_robust_slices=3)

    assert summary == {
        "max_credible_train_days": 60,
        "max_robust_train_days": 60,
        "first_insufficient_train_days": 0,
        "zero_slice_train_days": [],
    }
    assert module.classify_capacity_decision(summary) == (
        "exit_risk_forward_tail_capacity_supports_current_grid_no_tail_extension_required"
    )
    assert module.build_research_note(
        capacity_rows,
        validation_days=10,
        summary=summary,
    ) == (
        "当前 exit/risk non-overlapping 10d validation 对 30d~60d 候选 train window 仍有足够前推切片覆盖；"
        "当前 public dataset 已能支撑这组 more-tail 容量审计，不需要把继续扩更长 tail 当作默认下一步。"
    )


def test_builder_writes_exit_risk_tail_capacity_artifact_and_latest_alias(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = review_dir / "20260321T115755Z_public_intraday_crypto_bars_dataset.csv"
    _write_dataset(dataset_path, days=60, cadence_minutes=15)

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--dataset-path",
            str(dataset_path),
            "--symbol",
            "ETHUSDT",
            "--validation-days",
            "10",
            "--step-days",
            "10",
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260323T050500Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    output = json.loads(proc.stdout)
    json_path = Path(output["json_path"])
    latest_json_path = Path(output["latest_json_path"])
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    latest_payload = json.loads(latest_json_path.read_text(encoding="utf-8"))

    rows = {int(row["train_days"]): int(row["slice_count"]) for row in payload["capacity_rows"]}
    assert rows == {
        30: 3,
        40: 2,
        45: 1,
        50: 1,
        55: 0,
        60: 0,
    }
    assert payload["action"] == "build_price_action_breakout_pullback_exit_risk_forward_tail_capacity_sim_only"
    assert payload["change_class"] == "SIM_ONLY"
    assert payload["validation_window_mode"] == "non_overlapping"
    assert payload["max_credible_train_days"] == 40
    assert payload["max_robust_train_days"] == 30
    assert payload["first_insufficient_train_days"] == 45
    assert payload["zero_slice_train_days"] == [55, 60]
    assert payload["research_decision"] == (
        "exit_risk_forward_tail_capacity_supports_40d_watch_45d_plus_requires_longer_dataset"
    )
    assert payload["recommended_brief"] == (
        "ETHUSDT:exit_risk_tail_capacity:non_overlapping:max_credible=40d,max_robust=30d,"
        "first_insufficient=45d,decision=exit_risk_forward_tail_capacity_supports_40d_watch_45d_plus_requires_longer_dataset"
    )
    assert payload["research_note"] == (
        "当前 exit/risk non-overlapping 10d validation 仍可把 train window 扩到 40d，但 45d/50d 只剩 1 个 slice，"
        "55d+ 已无可用 slice；因此当前 public 60d dataset 只够把 45d+ 当 watch，不足以继续 more-tail 主证据。"
    )
    assert latest_payload["research_decision"] == payload["research_decision"]


def test_builder_marks_90d_dataset_as_current_grid_fully_covered(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = review_dir / "20260323T194500Z_public_intraday_crypto_bars_dataset.csv"
    _write_dataset(dataset_path, days=90, cadence_minutes=15)

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--dataset-path",
            str(dataset_path),
            "--symbol",
            "ETHUSDT",
            "--validation-days",
            "10",
            "--step-days",
            "10",
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260324T043000Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    output = json.loads(proc.stdout)
    payload = json.loads(Path(output["json_path"]).read_text(encoding="utf-8"))

    rows = {int(row["train_days"]): int(row["slice_count"]) for row in payload["capacity_rows"]}
    assert rows == {
        30: 6,
        40: 5,
        45: 4,
        50: 4,
        55: 3,
        60: 3,
    }
    assert payload["max_credible_train_days"] == 60
    assert payload["max_robust_train_days"] == 60
    assert payload["first_insufficient_train_days"] == 0
    assert payload["zero_slice_train_days"] == []
    assert payload["research_decision"] == (
        "exit_risk_forward_tail_capacity_supports_current_grid_no_tail_extension_required"
    )
    assert payload["research_note"] == (
        "当前 exit/risk non-overlapping 10d validation 对 30d~60d 候选 train window 仍有足够前推切片覆盖；"
        "当前 public dataset 已能支撑这组 more-tail 容量审计，不需要把继续扩更长 tail 当作默认下一步。"
    )
    assert "  只剩 1 个 slice" not in payload["research_note"]
    assert "45d+ 当 watch" not in payload["research_note"]
