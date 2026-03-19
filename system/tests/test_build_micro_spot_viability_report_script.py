from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml


def _load_module():
    root = Path(__file__).resolve().parents[1]
    mod_path = root / "scripts" / "build_micro_spot_viability_report.py"
    spec = importlib.util.spec_from_file_location("build_micro_spot_viability_report", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_report_marks_threshold_only_pass_but_size_still_blocked(tmp_path: Path) -> None:
    mod = _load_module()
    review_dir = tmp_path / "review"
    research_root = tmp_path / "research"
    review_dir.mkdir(parents=True, exist_ok=True)
    (research_root / "strategy_lab_20260310_111640").mkdir(parents=True, exist_ok=True)

    ticket_payload = {
        "thresholds": {
            "min_confidence": 68.7,
            "min_convexity": 2.93,
            "threshold_source": str(tmp_path / "artifacts" / "params_live.yaml"),
            "threshold_source_kind": "params_live",
        },
        "tickets": [
            {
                "symbol": "SOLUSDT",
                "allowed": False,
                "reasons": [
                    "confidence_below_threshold",
                    "convexity_below_threshold",
                    "size_below_min_notional",
                ],
                "signal": {"confidence": 17.11533, "convexity_ratio": 2.5},
                "levels": {"risk_per_unit_pct": 6.46892655367231},
                "sizing": {
                    "equity_usdt": 70.0,
                    "risk_multiplier": 0.44223149807836404,
                    "conviction": 0.1711533,
                    "risk_budget_usdt": 0.02384215478191754,
                    "quote_usdt": 0.3685643140960182,
                    "min_notional_usdt": 5.0,
                },
            }
        ],
    }
    ticket_path = review_dir / "20260317T181502Z_signal_to_order_tickets.json"
    ticket_path.write_text(json.dumps(ticket_payload, ensure_ascii=False) + "\n", encoding="utf-8")

    candidates_csv = research_root / "strategy_lab_20260310_111640" / "candidates.csv"
    candidates_csv.write_text(
        "\n".join(
            [
                "name,rationale,params,train_metrics,validation_metrics,factor_alignment_train,factor_alignment_validation,score,review_metrics,factor_alignment_review,robustness_score,accepted",
                "crypto_swing_flow_02,crypto,\"{'signal_confidence_min': 8.0, 'convexity_min': 0.7, 'hold_days': 1, 'max_daily_trades': 5}\",{},\"{'trades': 19}\",0,0,0.48,{},0,0.80,True",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = mod.build_report(
        review_dir=review_dir,
        research_root=research_root,
        ticket_json=str(ticket_path),
        symbol="SOLUSDT",
        reference_now=mod.now_utc(),
    )
    candidate = payload["lowest_threshold_accepted_crypto_candidate"]
    viability = payload["viability"]
    assert candidate["name"] == "crypto_swing_flow_02"
    assert candidate["signal_confidence_min"] == 8.0
    assert candidate["convexity_min"] == 0.7
    assert viability["threshold_only_passes"] is True
    assert viability["size_still_blocked"] is True
    assert viability["required_base_risk_pct_current_signal"] > 6.0


def test_script_main_writes_report_artifacts(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    output_root = tmp_path / "output"
    review_dir = output_root / "review"
    research_root = output_root / "research"
    review_dir.mkdir(parents=True, exist_ok=True)
    (research_root / "strategy_lab_20260310_111640").mkdir(parents=True, exist_ok=True)

    (review_dir / "20260317T181502Z_signal_to_order_tickets.json").write_text(
        json.dumps(
            {
                "thresholds": {
                    "min_confidence": 68.7,
                    "min_convexity": 2.93,
                    "threshold_source": str(output_root / "artifacts" / "params_live.yaml"),
                    "threshold_source_kind": "params_live",
                },
                "tickets": [
                    {
                        "symbol": "SOLUSDT",
                        "allowed": False,
                        "reasons": [
                            "confidence_below_threshold",
                            "convexity_below_threshold",
                            "size_below_min_notional",
                        ],
                        "signal": {"confidence": 17.11533, "convexity_ratio": 2.5},
                        "levels": {"risk_per_unit_pct": 6.46892655367231},
                        "sizing": {
                            "equity_usdt": 70.0,
                            "risk_multiplier": 0.44223149807836404,
                            "conviction": 0.1711533,
                            "risk_budget_usdt": 0.02384215478191754,
                            "quote_usdt": 0.3685643140960182,
                            "min_notional_usdt": 5.0,
                        },
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (research_root / "strategy_lab_20260310_111640" / "candidates.csv").write_text(
        "\n".join(
            [
                "name,rationale,params,train_metrics,validation_metrics,factor_alignment_train,factor_alignment_validation,score,review_metrics,factor_alignment_review,robustness_score,accepted",
                "crypto_swing_flow_02,crypto,\"{'signal_confidence_min': 8.0, 'convexity_min': 0.7, 'hold_days': 1, 'max_daily_trades': 5}\",{},\"{'trades': 19}\",0,0,0.48,{},0,0.80,True",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_micro_spot_viability_report.py",
            "--review-dir",
            str(review_dir),
            "--research-root",
            str(research_root),
            "--symbol",
            "SOLUSDT",
            "--now",
            "2026-03-17T18:19:24Z",
        ],
    )
    rc = mod.main()
    assert rc == 0
    generated = sorted(review_dir.glob("*_micro_spot_viability_report.json"))
    assert generated
    payload = json.loads(generated[-1].read_text(encoding="utf-8"))
    assert payload["focus_symbol"] == "SOLUSDT"
    assert payload["accepted_crypto_candidate_count"] == 1
    assert payload["viability"]["size_still_blocked"] is True
    markdown = sorted(review_dir.glob("*_micro_spot_viability_report.md"))
    assert markdown
    checksum = sorted(review_dir.glob("*_micro_spot_viability_report_checksum.json"))
    assert checksum


def test_build_report_raises_when_explicit_ticket_path_missing(tmp_path: Path) -> None:
    mod = _load_module()
    review_dir = tmp_path / "review"
    research_root = tmp_path / "research"
    review_dir.mkdir(parents=True, exist_ok=True)
    research_root.mkdir(parents=True, exist_ok=True)

    missing = review_dir / "missing_signal_to_order_tickets.json"
    try:
        mod.build_report(
            review_dir=review_dir,
            research_root=research_root,
            ticket_json=str(missing),
            symbol="SOLUSDT",
            reference_now=mod.now_utc(),
        )
    except FileNotFoundError as exc:
        assert str(missing) in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")
