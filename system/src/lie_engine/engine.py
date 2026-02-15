from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from contextlib import closing
import json
from pathlib import Path
import re
import sqlite3
from typing import Any

import pandas as pd
import yaml

from lie_engine.backtest import BacktestConfig, run_walk_forward_backtest
from lie_engine.config import SystemSettings, assert_valid_settings, load_settings
from lie_engine.data import DataBus, build_provider_stack
from lie_engine.data.storage import append_sqlite, write_csv, write_json, write_markdown
from lie_engine.models import BacktestResult, NewsEvent, RegimeLabel, ReviewDelta, TradePlan
from lie_engine.orchestration import (
    ArchitectureOrchestrator,
    DependencyOrchestrator,
    GuardAssessment,
    ObservabilityOrchestrator,
    ReleaseOrchestrator,
    SchedulerOrchestrator,
    TestingOrchestrator,
    estimate_factor_contrib_120d,
)
from lie_engine.orchestration.guards import black_swan_assessment, loss_cooldown_active, major_event_window
from lie_engine.regime import compute_atr_zscore, derive_regime_consensus, infer_hmm_state, latest_multi_scale_hurst
from lie_engine.research import run_research_backtest as run_research_pipeline
from lie_engine.research import run_strategy_lab as run_strategy_lab_pipeline
from lie_engine.reporting import render_daily_briefing, render_review_report, write_run_manifest
from lie_engine.review import ReviewThresholds, bounded_bayesian_update, build_review_delta
from lie_engine.risk import RiskManager, infer_edge_from_trades
from lie_engine.signal import SignalEngineConfig, expand_universe, scan_signals


@dataclass(slots=True)
class EngineContext:
    settings: SystemSettings
    root: Path
    output_dir: Path
    sqlite_path: Path


class LieEngine:
    def __init__(self, config_path: str | Path | None = None) -> None:
        settings = load_settings(config_path)
        assert_valid_settings(settings)
        root = Path(config_path).resolve().parent if config_path else Path(__file__).resolve().parents[2]
        output_dir = root / settings.paths.get("output", "output")
        sqlite_path = root / settings.paths.get("sqlite", "output/artifacts/lie_engine.db")
        self.ctx = EngineContext(settings=settings, root=root, output_dir=output_dir, sqlite_path=sqlite_path)

        data_cfg = settings.raw.get("data", {}) if isinstance(settings.raw, dict) else {}
        provider_profile = str(data_cfg.get("provider_profile", "opensource_dual"))
        self.providers = build_provider_stack(provider_profile)
        self.data_bus = DataBus(
            providers=self.providers,
            output_dir=self.ctx.output_dir,
            sqlite_path=self.ctx.sqlite_path,
            completeness_min=float(settings.validation.get("data_completeness_min", 0.99)),
            conflict_max=float(settings.validation.get("unresolved_conflict_max", 0.005)),
            source_confidence_min=float(settings.validation.get("source_confidence_min", 0.75)),
            low_confidence_source_ratio_max=float(settings.validation.get("low_confidence_source_ratio_max", 0.40)),
        )

        self.risk_manager = RiskManager(
            max_single_risk_pct=float(settings.risk.get("max_single_risk_pct", 2.0)),
            max_total_exposure_pct=float(settings.risk.get("max_total_exposure_pct", 50.0)),
            max_symbol_pct=float(settings.risk.get("max_symbol_pct", 15.0)),
            max_theme_pct=float(settings.risk.get("max_theme_pct", 25.0)),
        )

    @property
    def settings(self) -> SystemSettings:
        return self.ctx.settings

    def _core_symbols(self) -> list[str]:
        return [x["symbol"] for x in self.settings.universe.get("core", [])]

    def _theme_map(self) -> dict[str, str]:
        out = {}
        for item in self.settings.universe.get("core", []):
            out[item["symbol"]] = item.get("theme", "unknown")
        return out

    @staticmethod
    def _load_json_safely(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    @staticmethod
    def _clamp_float(v: float, lo: float, hi: float) -> float:
        return float(max(lo, min(hi, v)))

    @staticmethod
    def _clamp_int(v: float | int, lo: int, hi: int) -> int:
        return int(max(lo, min(hi, int(round(float(v))))))

    @staticmethod
    def _parse_iso_date(value: Any) -> date | None:
        if not value:
            return None
        try:
            return date.fromisoformat(str(value))
        except Exception:
            return None

    @staticmethod
    def _parse_backtest_run_id_dates(run_id: str) -> tuple[date | None, date | None]:
        m = re.match(r"^(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})$", str(run_id).strip())
        if not m:
            return None, None
        try:
            return date.fromisoformat(m.group(1)), date.fromisoformat(m.group(2))
        except Exception:
            return None, None

    def _mode_history_stats(self, *, as_of: date, lookback_days: int | None = None) -> dict[str, Any]:
        lb = int(lookback_days) if isinstance(lookback_days, int) and lookback_days > 0 else int(self.settings.validation.get("mode_stats_lookback_days", 365))
        lb = max(30, lb)
        manifest_dir = self.ctx.output_dir / "artifacts" / "manifests"
        if not manifest_dir.exists():
            return {"as_of": as_of.isoformat(), "lookback_days": lb, "modes": {}}

        buckets: dict[str, list[dict[str, float]]] = {}
        for path in sorted(manifest_dir.glob("backtest_*.json")):
            payload = self._load_json_safely(path)
            if not payload:
                continue
            run_type = str(payload.get("run_type", "")).strip()
            if run_type and run_type != "backtest":
                continue
            run_id = str(payload.get("run_id", "")).strip()
            _, end_d = self._parse_backtest_run_id_dates(run_id)
            if end_d is None:
                _, end_d = self._parse_backtest_run_id_dates(path.stem.replace("backtest_", "", 1))
            if end_d is None:
                continue
            if end_d > as_of:
                continue
            if (as_of - end_d).days > lb:
                continue

            metrics = payload.get("metrics", {})
            checks = payload.get("checks", {})
            if not isinstance(metrics, dict):
                continue
            mode = str(metrics.get("runtime_mode", "base")).strip() or "base"
            row = {
                "annual_return": float(metrics.get("annual_return", 0.0)),
                "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
                "win_rate": float(metrics.get("win_rate", 0.0)),
                "profit_factor": float(metrics.get("profit_factor", 0.0)),
                "trades": float(metrics.get("trades", 0.0)),
                "violations": float((checks or {}).get("violations", 0.0)),
            }
            buckets.setdefault(mode, []).append(row)

        modes: dict[str, Any] = {}
        for mode, rows in buckets.items():
            if not rows:
                continue
            n = float(len(rows))
            modes[mode] = {
                "samples": int(len(rows)),
                "avg_annual_return": float(sum(r["annual_return"] for r in rows) / n),
                "avg_win_rate": float(sum(r["win_rate"] for r in rows) / n),
                "avg_profit_factor": float(sum(r["profit_factor"] for r in rows) / n),
                "avg_trades": float(sum(r["trades"] for r in rows) / n),
                "worst_drawdown": float(max(r["max_drawdown"] for r in rows)),
                "total_violations": int(sum(int(r["violations"]) for r in rows)),
            }
        return {"as_of": as_of.isoformat(), "lookback_days": lb, "modes": modes}

    def _review_backtest_start(self, as_of: date) -> date:
        default_start = date(2015, 1, 1)
        val = self.settings.validation if isinstance(self.settings.validation, dict) else {}

        lookback_raw = val.get("review_backtest_lookback_days")
        if isinstance(lookback_raw, (int, float)) and int(lookback_raw) > 0:
            lookback_days = max(30, int(lookback_raw))
            return max(default_start, as_of - timedelta(days=lookback_days))

        start_raw = val.get("review_backtest_start_date")
        parsed = self._parse_iso_date(start_raw)
        if parsed is None:
            return default_start
        if parsed > as_of:
            return default_start
        return max(default_start, parsed)

    @staticmethod
    def _default_mode_profiles() -> dict[str, dict[str, float]]:
        return {
            "ultra_short": {
                "signal_confidence_min": 56.0,
                "convexity_min": 1.6,
                "hold_days": 2.0,
                "max_daily_trades": 4.0,
            },
            "swing": {
                "signal_confidence_min": 60.0,
                "convexity_min": 2.2,
                "hold_days": 8.0,
                "max_daily_trades": 2.0,
            },
            "long": {
                "signal_confidence_min": 54.0,
                "convexity_min": 2.8,
                "hold_days": 18.0,
                "max_daily_trades": 1.0,
            },
        }

    def _mode_from_regime(self, consensus: RegimeLabel) -> str:
        default_map = {
            RegimeLabel.STRONG_TREND: "long",
            RegimeLabel.WEAK_TREND: "swing",
            RegimeLabel.RANGE: "ultra_short",
            RegimeLabel.DOWNTREND: "swing",
            RegimeLabel.UNCERTAIN: "swing",
            RegimeLabel.EXTREME_VOL: "ultra_short",
        }
        raw_runtime = self.settings.raw.get("mode_runtime", {}) if isinstance(self.settings.raw, dict) else {}
        raw_map = raw_runtime.get("regime_to_mode", {}) if isinstance(raw_runtime, dict) else {}

        mode = default_map.get(consensus, "swing")
        if isinstance(raw_map, dict):
            for key in (consensus.value, consensus.name):
                if key in raw_map and str(raw_map.get(key)).strip():
                    mode = str(raw_map.get(key)).strip()
                    break
        if mode not in {"ultra_short", "swing", "long"}:
            mode = "swing"
        return mode

    def _resolve_runtime_params(self, *, regime, live_params: dict[str, float]) -> dict[str, float | str]:
        base = {
            "signal_confidence_min": self._clamp_float(
                float(live_params.get("signal_confidence_min", self.settings.thresholds.get("signal_confidence_min", 60.0))),
                20.0,
                95.0,
            ),
            "convexity_min": self._clamp_float(
                float(live_params.get("convexity_min", self.settings.thresholds.get("convexity_min", 3.0))),
                0.5,
                5.0,
            ),
            "hold_days": float(self._clamp_int(live_params.get("hold_days", 5.0), 1, 20)),
            "max_daily_trades": float(self._clamp_int(live_params.get("max_daily_trades", 2.0), 1, 5)),
        }
        if not bool(self.settings.validation.get("use_mode_profiles", False)):
            return {"mode": "base"} | base

        mode = self._mode_from_regime(regime.consensus)
        profiles = self._resolved_mode_profiles()
        profile = profiles.get(mode, profiles["swing"])
        blend = self._clamp_float(float(self.settings.validation.get("mode_profile_blend_with_live", 0.55)), 0.0, 1.0)
        merged_conf = (1.0 - blend) * float(base["signal_confidence_min"]) + blend * float(profile["signal_confidence_min"])
        merged_conv = (1.0 - blend) * float(base["convexity_min"]) + blend * float(profile["convexity_min"])
        merged_hold = (1.0 - blend) * float(base["hold_days"]) + blend * float(profile["hold_days"])
        merged_trades = (1.0 - blend) * float(base["max_daily_trades"]) + blend * float(profile["max_daily_trades"])
        return {
            "mode": mode,
            "signal_confidence_min": self._clamp_float(merged_conf, 20.0, 95.0),
            "convexity_min": self._clamp_float(merged_conv, 0.5, 5.0),
            "hold_days": float(self._clamp_int(merged_hold, 1, 20)),
            "max_daily_trades": float(self._clamp_int(merged_trades, 1, 5)),
        }

    def _resolved_mode_profiles(self) -> dict[str, dict[str, float]]:
        profiles = self._default_mode_profiles()
        raw_profiles = self.settings.raw.get("mode_profiles", {}) if isinstance(self.settings.raw, dict) else {}
        if isinstance(raw_profiles, dict):
            for mode_name, values in raw_profiles.items():
                if mode_name not in profiles or not isinstance(values, dict):
                    continue
                for key in ("signal_confidence_min", "convexity_min", "hold_days", "max_daily_trades"):
                    if key in values and isinstance(values.get(key), (int, float)):
                        profiles[mode_name][key] = float(values[key])
        return profiles

    def _load_latest_strategy_candidate(self, as_of: date) -> dict[str, Any]:
        if not bool(self.settings.validation.get("review_use_strategy_lab", True)):
            return {}
        lookback_days = int(self.settings.validation.get("strategy_lab_manifest_lookback_days", 45))
        manifest_dir = self.ctx.output_dir / "artifacts" / "manifests"
        if not manifest_dir.exists():
            return {}

        manifests = sorted(manifest_dir.glob("strategy_lab_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for mp in manifests:
            manifest = self._load_json_safely(mp)
            if not manifest:
                continue
            artifacts = manifest.get("artifacts", {})
            if not isinstance(artifacts, dict):
                continue
            summary_ref = artifacts.get("summary")
            if not summary_ref:
                continue
            summary_path = Path(str(summary_ref))
            if not summary_path.is_absolute():
                summary_path = self.ctx.root / summary_path
            summary = self._load_json_safely(summary_path)
            if not summary:
                continue
            cutoff = self._parse_iso_date(summary.get("cutoff_date"))
            if cutoff is None:
                metadata = manifest.get("metadata", {}) if isinstance(manifest.get("metadata", {}), dict) else {}
                cutoff = self._parse_iso_date(metadata.get("cutoff_date"))
            if cutoff is None:
                continue
            if cutoff > as_of:
                continue
            if (as_of - cutoff).days > max(1, lookback_days):
                continue

            best = summary.get("best_candidate", {})
            if not isinstance(best, dict):
                best = {}
            if (not best) or (not bool(best.get("accepted", False))):
                cands = summary.get("candidates", [])
                if isinstance(cands, list):
                    accepted = [c for c in cands if isinstance(c, dict) and bool(c.get("accepted", False))]
                    if accepted:
                        accepted.sort(key=lambda x: float(x.get("score", -1e9)), reverse=True)
                        best = accepted[0]
            if not isinstance(best, dict) or not best:
                continue
            params = best.get("params", {})
            if not isinstance(params, dict) or not params:
                continue
            return {
                "manifest_path": str(mp),
                "summary_path": str(summary_path),
                "cutoff_date": cutoff.isoformat(),
                "candidate": best,
            }
        return {}

    def _autorun_strategy_lab_for_review(self, as_of: date) -> dict[str, Any]:
        if not bool(self.settings.validation.get("review_autorun_strategy_lab_if_missing", False)):
            return {}
        lookback_days = max(60, int(self.settings.validation.get("strategy_lab_autorun_lookback_days", 180)))
        start = max(date(2015, 1, 1), as_of - timedelta(days=lookback_days))
        max_symbols = max(1, int(self.settings.validation.get("strategy_lab_autorun_max_symbols", 60)))
        report_symbol_cap = max(1, int(self.settings.validation.get("strategy_lab_autorun_report_symbol_cap", 20)))
        workers = max(1, int(self.settings.validation.get("strategy_lab_autorun_workers", 4)))
        review_days = max(0, int(self.settings.validation.get("strategy_lab_autorun_review_days", 3)))
        candidate_count = max(1, int(self.settings.validation.get("strategy_lab_autorun_candidate_count", 6)))
        return self.run_strategy_lab(
            start=start,
            end=as_of,
            max_symbols=max_symbols,
            report_symbol_cap=report_symbol_cap,
            workers=workers,
            review_days=review_days,
            candidate_count=candidate_count,
        )

    def _merge_strategy_candidate_into_review(
        self,
        *,
        review: ReviewDelta,
        current_params: dict[str, float],
        candidate_payload: dict[str, Any],
    ) -> None:
        cand = candidate_payload.get("candidate", {})
        if not isinstance(cand, dict):
            return
        params = cand.get("params", {})
        if not isinstance(params, dict):
            return

        step = self._clamp_float(float(self.settings.validation.get("strategy_lab_merge_step", 0.18)), 0.01, 0.60)
        updated: list[str] = []
        fields: list[tuple[str, float, float, bool]] = [
            ("signal_confidence_min", 30.0, 90.0, False),
            ("convexity_min", 1.0, 4.0, False),
            ("hold_days", 1.0, 20.0, True),
            ("max_daily_trades", 1.0, 5.0, True),
        ]
        for key, lo, hi, is_int in fields:
            if key not in params:
                continue
            target = float(params.get(key, 0.0))
            target = self._clamp_float(target, lo, hi)
            base = float(review.parameter_changes.get(key, current_params.get(key, target)))
            merged = bounded_bayesian_update(base, target, lo, hi, step=step)
            if is_int:
                review.parameter_changes[key] = float(self._clamp_int(merged, int(lo), int(hi)))
            else:
                review.parameter_changes[key] = float(merged)
            review.change_reasons[key] = f"融合 strategy-lab 最优候选 `{cand.get('name', 'unknown')}`（小步收敛，step={step:.2f}）"
            updated.append(key)

        if updated:
            review.notes.append(
                "strategy_lab_candidate="
                + f"{cand.get('name', 'unknown')}; cutoff={candidate_payload.get('cutoff_date', '')}; "
                + f"manifest={candidate_payload.get('manifest_path', '')}"
            )

    def _quality_snapshot(self, as_of: date) -> dict[str, Any]:
        path = self.ctx.output_dir / "artifacts" / f"{as_of.isoformat()}_quality.json"
        return self._load_json_safely(path)

    def _backtest_snapshot(self, as_of: date) -> dict[str, Any]:
        path = self.ctx.output_dir / "artifacts" / f"backtest_{date(2015, 1, 1).isoformat()}_{as_of.isoformat()}.json"
        return self._load_json_safely(path)

    @staticmethod
    def _filter_model_path_bars(bars: pd.DataFrame) -> pd.DataFrame:
        if bars.empty or "data_conflict_flag" not in bars.columns:
            return bars
        return bars[bars["data_conflict_flag"] == 0].copy()

    def _major_event_window(self, as_of: date, news: list[NewsEvent]) -> bool:
        lookback_hours = int(self.settings.validation.get("major_event_window_hours", 24))
        return major_event_window(as_of=as_of, news=news, lookback_hours=lookback_hours)

    def _loss_cooldown_active(self, recent_trades: pd.DataFrame) -> bool:
        cooldown_losses = int(self.settings.validation.get("cooldown_consecutive_losses", 3))
        return loss_cooldown_active(recent_trades=recent_trades, cooldown_losses=cooldown_losses)

    def _black_swan_assessment(self, regime, sentiment: dict[str, float], news: list[NewsEvent]) -> tuple[float, list[str], bool]:
        threshold = float(self.settings.validation.get("black_swan_score_threshold", 70.0))
        return black_swan_assessment(
            atr_z=float(regime.atr_z),
            sentiment=sentiment,
            news=news,
            threshold=threshold,
        )

    def _evaluate_guards(
        self,
        *,
        as_of: date,
        regime,
        quality_passed: bool,
        sentiment: dict[str, float],
        news: list[NewsEvent],
        recent_trades: pd.DataFrame,
    ) -> GuardAssessment:
        score, items, trigger = self._black_swan_assessment(regime=regime, sentiment=sentiment, news=news)
        event_window = self._major_event_window(as_of=as_of, news=news)
        cooldown = self._loss_cooldown_active(recent_trades=recent_trades)

        reasons: list[str] = []
        if event_window:
            reasons.append("重大事件窗口：暂停新增仓位")
        if not quality_passed:
            reasons.append("数据质量未通过门禁：进入保护模式并禁止开新仓")
        if regime.consensus in {RegimeLabel.UNCERTAIN, RegimeLabel.EXTREME_VOL}:
            reasons.append("体制冲突/极端波动：不做方向强判")
        if cooldown:
            reasons.append("连亏冷却期未结束：暂停新增仓位")
        if trigger:
            reasons.append("黑天鹅评分超阈值：保护模式生效")

        return GuardAssessment(
            black_swan_score=score,
            black_swan_items=items,
            black_swan_trigger=trigger,
            major_event_window=event_window,
            cooldown_active=cooldown,
            non_trade_reasons=reasons,
        )

    def _estimate_factor_contrib_120d(self, as_of: date) -> dict[str, float]:
        lookback_days = int(self.settings.validation.get("factor_lookback_days", 120))
        start = as_of - timedelta(days=max(lookback_days * 2, 220))
        symbols = self._core_symbols()
        bars, ingest = self._run_ingestion_range(start=start, end=as_of, symbols=symbols)
        bars = self._filter_model_path_bars(bars)
        return estimate_factor_contrib_120d(
            bars=bars,
            ingest=ingest,
            providers=self.providers,
            lookback_days=lookback_days,
        )

    def _run_ingestion(self, as_of: date, symbols: list[str]) -> tuple[pd.DataFrame, Any]:
        start = as_of - timedelta(days=420)
        start_ts = datetime.combine(as_of, time(0, 0)) - timedelta(days=1)
        end_ts = datetime.combine(as_of, time(23, 59))

        result = self.data_bus.ingest(
            symbols=symbols,
            start=start,
            end=as_of,
            start_ts=start_ts,
            end_ts=end_ts,
            langs=("zh", "en"),
        )
        expanded = expand_universe(
            core_symbols=symbols,
            bars=result.normalized_bars,
            max_additions=int(self.settings.universe.get("max_dynamic_additions", 5)),
        )
        if set(expanded) - set(symbols):
            result = self.data_bus.ingest(
                symbols=expanded,
                start=start,
                end=as_of,
                start_ts=start_ts,
                end_ts=end_ts,
                langs=("zh", "en"),
            )
            symbols = expanded

        self.data_bus.persist(as_of=as_of, result=result)
        model_bars = self._filter_model_path_bars(result.normalized_bars)
        return model_bars, result

    def _run_ingestion_range(self, start: date, end: date, symbols: list[str]) -> tuple[pd.DataFrame, Any]:
        start_ts = datetime.combine(start, time(0, 0))
        end_ts = datetime.combine(end, time(23, 59))

        result = self.data_bus.ingest(
            symbols=symbols,
            start=start,
            end=end,
            start_ts=start_ts,
            end_ts=end_ts,
            langs=("zh", "en"),
        )
        expanded = expand_universe(
            core_symbols=symbols,
            bars=result.normalized_bars,
            max_additions=int(self.settings.universe.get("max_dynamic_additions", 5)),
        )
        if set(expanded) - set(symbols):
            result = self.data_bus.ingest(
                symbols=expanded,
                start=start,
                end=end,
                start_ts=start_ts,
                end_ts=end_ts,
                langs=("zh", "en"),
            )

        self.data_bus.persist(as_of=end, result=result)
        model_bars = self._filter_model_path_bars(result.normalized_bars)
        return model_bars, result

    def _regime_from_bars(self, as_of: date, bars: pd.DataFrame):
        if bars.empty:
            return derive_regime_consensus(
                as_of=as_of,
                hurst=0.5,
                hmm_probs={"bull": 0.33, "range": 0.34, "bear": 0.33},
                atr_z=0.0,
                trend_thr=float(self.settings.thresholds.get("hurst_trend", 0.6)),
                mean_thr=float(self.settings.thresholds.get("hurst_mean_revert", 0.4)),
                atr_extreme=float(self.settings.thresholds.get("atr_extreme", 2.0)),
            )

        market_proxy = bars.groupby("ts", as_index=False).agg({"close": "mean", "high": "mean", "low": "mean", "open": "mean", "volume": "mean"})
        market_proxy = market_proxy.sort_values("ts")

        hurst = latest_multi_scale_hurst(market_proxy["close"].to_numpy())
        hmm_probs = infer_hmm_state(market_proxy[["close", "volume"]].assign(high=market_proxy["close"], low=market_proxy["close"]))
        atr_z = compute_atr_zscore(market_proxy[["open", "high", "low", "close"]])

        regime = derive_regime_consensus(
            as_of=as_of,
            hurst=hurst,
            hmm_probs=hmm_probs,
            atr_z=atr_z,
            trend_thr=float(self.settings.thresholds.get("hurst_trend", 0.6)),
            mean_thr=float(self.settings.thresholds.get("hurst_mean_revert", 0.4)),
            atr_extreme=float(self.settings.thresholds.get("atr_extreme", 2.0)),
        )
        return regime

    def _load_recent_trades(self, n: int = 200) -> pd.DataFrame:
        if not self.ctx.sqlite_path.exists():
            return pd.DataFrame(columns=["date", "symbol", "pnl"])

        query = f"SELECT * FROM executed_plans ORDER BY rowid DESC LIMIT {int(n)}"
        with closing(sqlite3.connect(self.ctx.sqlite_path)) as conn:
            try:
                df = pd.read_sql_query(query, conn)
            except Exception:
                return pd.DataFrame(columns=["date", "symbol", "pnl"])
        return df

    def _paper_positions_state_path(self) -> Path:
        return self.ctx.output_dir / "artifacts" / "paper_positions_open.json"

    def _load_open_paper_positions(self) -> list[dict[str, Any]]:
        payload = self._load_json_safely(self._paper_positions_state_path())
        rows = payload.get("positions", []) if isinstance(payload.get("positions", []), list) else []
        out: list[dict[str, Any]] = []
        for row in rows:
            if isinstance(row, dict):
                out.append(dict(row))
        return out

    def _save_open_paper_positions(self, *, as_of: date, positions: list[dict[str, Any]]) -> None:
        write_json(
            self._paper_positions_state_path(),
            {
                "as_of": as_of.isoformat(),
                "positions": positions,
            },
        )

    @staticmethod
    def _bars_daily_snapshot(bars: pd.DataFrame) -> dict[str, dict[str, float]]:
        if bars.empty:
            return {}
        cols = {"symbol", "ts", "open", "high", "low", "close"}
        if not cols.issubset(set(bars.columns)):
            return {}
        snap = bars[["symbol", "ts", "open", "high", "low", "close"]].copy()
        snap["ts"] = pd.to_datetime(snap["ts"], errors="coerce")
        snap = snap.dropna(subset=["symbol", "ts"])
        if snap.empty:
            return {}
        for c in ("open", "high", "low", "close"):
            snap[c] = pd.to_numeric(snap[c], errors="coerce")
        snap = snap.dropna(subset=["open", "high", "low", "close"])
        if snap.empty:
            return {}
        snap = snap.sort_values(["symbol", "ts"])
        agg = (
            snap.groupby("symbol", as_index=False)
            .agg(
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
            )
        )
        out: dict[str, dict[str, float]] = {}
        for row in agg.to_dict(orient="records"):
            sym = str(row.get("symbol", "")).strip()
            if not sym:
                continue
            out[sym] = {
                "open": float(row.get("open", 0.0)),
                "high": float(row.get("high", 0.0)),
                "low": float(row.get("low", 0.0)),
                "close": float(row.get("close", 0.0)),
            }
        return out

    def _settle_open_paper_positions(self, *, as_of: date, bars: pd.DataFrame) -> dict[str, Any]:
        positions = self._load_open_paper_positions()
        if not positions:
            return {
                "closed_df": pd.DataFrame(
                    columns=[
                        "date",
                        "open_date",
                        "symbol",
                        "side",
                        "direction",
                        "runtime_mode",
                        "mode",
                        "size_pct",
                        "risk_pct",
                        "entry_price",
                        "exit_price",
                        "pnl",
                        "pnl_pct",
                        "exit_reason",
                        "hold_days",
                        "holding_days",
                        "status",
                    ]
                ),
                "remaining_positions": [],
                "summary": {
                    "open_before": 0,
                    "closed_count": 0,
                    "closed_pnl": 0.0,
                    "missing_symbol_count": 0,
                },
            }

        snapshot = self._bars_daily_snapshot(bars)
        remaining: list[dict[str, Any]] = []
        closed_rows: list[dict[str, Any]] = []
        missing_symbol_count = 0

        for pos in positions:
            symbol = str(pos.get("symbol", "")).strip()
            side = str(pos.get("side", "LONG")).strip().upper()
            if not symbol or symbol not in snapshot:
                remaining.append(pos)
                missing_symbol_count += 1 if symbol else 0
                continue

            px = snapshot[symbol]
            entry = float(pos.get("entry_price", px["close"]))
            stop = float(pos.get("stop_price", entry))
            target = float(pos.get("target_price", entry))
            size_pct = float(pos.get("size_pct", 0.0))
            risk_pct = float(pos.get("risk_pct", 0.0))
            hold_days = max(1, int(pos.get("hold_days", 1)))
            open_date = self._parse_iso_date(pos.get("open_date")) or as_of
            holding_days = max(0, (as_of - open_date).days)
            runtime_mode = str(pos.get("runtime_mode", "base")).strip() or "base"

            exit_price = float(px["close"])
            exit_reason = ""

            if side == "SHORT":
                if float(px["high"]) >= stop:
                    exit_price = stop
                    exit_reason = "stop_loss"
                elif float(px["low"]) <= target:
                    exit_price = target
                    exit_reason = "take_profit"
            else:
                if float(px["low"]) <= stop:
                    exit_price = stop
                    exit_reason = "stop_loss"
                elif float(px["high"]) >= target:
                    exit_price = target
                    exit_reason = "take_profit"

            if not exit_reason and holding_days >= hold_days:
                exit_price = float(px["close"])
                exit_reason = "time_stop"

            if not exit_reason:
                remaining.append(pos)
                continue

            signed_ret = (exit_price - entry) / max(entry, 1e-9)
            if side == "SHORT":
                signed_ret = -signed_ret
            pnl = float((size_pct / 100.0) * signed_ret)

            closed_rows.append(
                {
                    "date": as_of.isoformat(),
                    "open_date": open_date.isoformat(),
                    "symbol": symbol,
                    "side": side,
                    "direction": side,
                    "runtime_mode": runtime_mode,
                    "mode": runtime_mode,
                    "size_pct": size_pct,
                    "risk_pct": risk_pct,
                    "entry_price": entry,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "pnl_pct": float(signed_ret),
                    "exit_reason": exit_reason,
                    "hold_days": hold_days,
                    "holding_days": holding_days,
                    "status": "CLOSED",
                }
            )

        closed_df = pd.DataFrame(closed_rows)
        if not closed_df.empty:
            append_sqlite(self.ctx.sqlite_path, "executed_plans", closed_df)

        return {
            "closed_df": closed_df,
            "remaining_positions": remaining,
            "summary": {
                "open_before": len(positions),
                "closed_count": int(len(closed_rows)),
                "closed_pnl": float(closed_df["pnl"].sum()) if not closed_df.empty else 0.0,
                "missing_symbol_count": int(missing_symbol_count),
            },
        }

    @staticmethod
    def _build_open_positions_from_plans(
        *,
        as_of: date,
        plans: list[TradePlan],
        runtime_mode: str,
        hold_days: int,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for p in plans:
            if str(p.status).upper() != "ACTIVE":
                continue
            out.append(
                {
                    "open_date": as_of.isoformat(),
                    "symbol": str(p.symbol),
                    "side": str(p.side.value),
                    "size_pct": float(p.size_pct),
                    "risk_pct": float(p.risk_pct),
                    "entry_price": float(p.entry_price),
                    "stop_price": float(p.stop_price),
                    "target_price": float(p.target_price),
                    "runtime_mode": str(runtime_mode or "base"),
                    "hold_days": int(max(1, hold_days)),
                    "status": "OPEN",
                }
            )
        return out

    def _symbol_exposure_snapshot(self) -> tuple[dict[str, float], dict[str, float], float]:
        if not self.ctx.sqlite_path.exists():
            return {}, {}, 0.0
        with closing(sqlite3.connect(self.ctx.sqlite_path)) as conn:
            try:
                pos = pd.read_sql_query(
                    "SELECT symbol, size_pct, status FROM latest_positions "
                    "WHERE date = (SELECT MAX(date) FROM latest_positions)",
                    conn,
                )
            except Exception:
                try:
                    pos = pd.read_sql_query("SELECT symbol, size_pct, status FROM latest_positions", conn)
                except Exception:
                    return {}, {}, 0.0
        if pos.empty:
            return {}, {}, 0.0

        pos = pos[pos["status"] == "ACTIVE"]
        by_symbol = pos.groupby("symbol")["size_pct"].sum().to_dict()
        theme_map = self._theme_map()
        by_theme: dict[str, float] = {}
        for sym, val in by_symbol.items():
            theme = theme_map.get(sym, "dynamic")
            by_theme.setdefault(theme, 0.0)
            by_theme[theme] += float(val)
        total = float(sum(by_symbol.values()))
        return by_symbol, by_theme, total

    def run_eod(self, as_of: date) -> dict[str, Any]:
        symbols = self._core_symbols()
        bars, ingest = self._run_ingestion(as_of, symbols)
        paper_exec = self._settle_open_paper_positions(as_of=as_of, bars=bars)
        regime = self._regime_from_bars(as_of=as_of, bars=bars)
        live_params = self._load_live_params()
        runtime_params = self._resolve_runtime_params(regime=regime, live_params=live_params)

        signal_cfg = SignalEngineConfig(
            confidence_min=float(runtime_params["signal_confidence_min"]),
            convexity_min=float(runtime_params["convexity_min"]),
        )
        signals = scan_signals(bars=bars, regime=regime.consensus, cfg=signal_cfg)

        recent_trades = self._load_recent_trades(300)
        win_rate, payoff = infer_edge_from_trades(recent_trades)
        guard = self._evaluate_guards(
            as_of=as_of,
            regime=regime,
            quality_passed=bool(ingest.quality.passed),
            sentiment=ingest.sentiment,
            news=ingest.news,
            recent_trades=recent_trades,
        )
        runtime_mode = str(runtime_params.get("mode", "base"))
        mode_history = self._mode_history_stats(as_of=as_of)
        mode_health = self._evaluate_mode_health(runtime_mode=runtime_mode, mode_history=mode_history)
        risk_control = self._execution_risk_control(
            source_confidence_score=float(ingest.quality.source_confidence_score),
            mode_health=mode_health,
        )

        by_symbol, by_theme, used_exposure = self._symbol_exposure_snapshot()
        budget = self.risk_manager.build_budget(account_equity=1_000_000.0, used_exposure_pct=used_exposure)

        plans: list[TradePlan] = []
        theme_map = self._theme_map()

        if not guard.trade_blocked:
            for s in signals:
                theme = theme_map.get(s.symbol, "dynamic")
                plan = self.risk_manager.size_signal(
                    signal=s,
                    win_rate=win_rate,
                    payoff=payoff,
                    budget=budget,
                    symbol_exposure_pct=by_symbol.get(s.symbol, 0.0),
                    theme_exposure_pct=by_theme.get(theme, 0.0),
                    protection_mode=regime.protection_mode or guard.black_swan_trigger,
                    risk_multiplier=float(risk_control.get("risk_multiplier", 1.0)),
                )
                if plan:
                    plans.append(plan)
                    by_symbol[s.symbol] = by_symbol.get(s.symbol, 0.0) + plan.size_pct
                    by_theme[theme] = by_theme.get(theme, 0.0) + plan.size_pct
                    budget.used_exposure_pct += plan.size_pct

        next_events = [
            "08:40 盘前事件核查",
            "10:30 / 14:30 盘中信息校准",
            "20:30 盘后复盘和参数重估",
        ]
        avg_conf = float(pd.Series([float(s.confidence) for s in signals]).mean()) if signals else 0.0
        avg_conv = float(pd.Series([float(s.convexity_ratio) for s in signals]).mean()) if signals else 0.0
        mode_feedback = {
            "runtime_mode": runtime_mode,
            "runtime_params": {
                "signal_confidence_min": float(runtime_params["signal_confidence_min"]),
                "convexity_min": float(runtime_params["convexity_min"]),
                "hold_days": float(runtime_params["hold_days"]),
                "max_daily_trades": float(runtime_params["max_daily_trades"]),
            },
            "today": {
                "signals": int(len(signals)),
                "plans": int(len(plans)),
                "avg_confidence": avg_conf,
                "avg_convexity": avg_conv,
                "closed_trades": int((paper_exec.get("summary", {}) or {}).get("closed_count", 0)),
                "closed_pnl": float((paper_exec.get("summary", {}) or {}).get("closed_pnl", 0.0)),
            },
            "history": mode_history,
            "mode_health": mode_health,
            "risk_control": risk_control,
        }

        daily_md = render_daily_briefing(
            as_of=as_of,
            regime=regime,
            signals=signals,
            plans=plans,
            quality=ingest.quality,
            black_swan_items=guard.black_swan_items,
            next_events=next_events,
            black_swan_score=guard.black_swan_score,
            non_trade_reasons=guard.non_trade_reasons,
            mode_feedback=mode_feedback,
        )

        daily_dir = self.ctx.output_dir / "daily"
        briefing_path = daily_dir / f"{as_of.isoformat()}_briefing.md"
        signals_path = daily_dir / f"{as_of.isoformat()}_signals.json"
        positions_path = daily_dir / f"{as_of.isoformat()}_positions.csv"
        mode_feedback_path = daily_dir / f"{as_of.isoformat()}_mode_feedback.json"

        write_markdown(briefing_path, daily_md)
        write_json(signals_path, [s.to_dict() for s in signals])
        write_json(mode_feedback_path, mode_feedback)
        plans_df = pd.DataFrame([p.to_dict() for p in plans])
        if plans_df.empty:
            plans_df = pd.DataFrame(
                columns=[
                    "symbol",
                    "side",
                    "size_pct",
                    "risk_pct",
                    "entry_price",
                    "stop_price",
                    "target_price",
                    "hedge_leg",
                    "reason",
                    "status",
                ]
            )
        write_csv(positions_path, plans_df)

        signals_df = pd.DataFrame([s.to_dict() for s in signals])
        if signals_df.empty:
            signals_df = pd.DataFrame(
                columns=[
                    "symbol",
                    "side",
                    "regime",
                    "position_score",
                    "structure_score",
                    "momentum_score",
                    "confidence",
                    "convexity_ratio",
                    "entry_price",
                    "stop_price",
                    "target_price",
                    "can_short",
                    "notes",
                ]
            )
        append_sqlite(self.ctx.sqlite_path, "signals", signals_df)
        append_sqlite(self.ctx.sqlite_path, "trade_plans", plans_df.assign(date=as_of.isoformat()))

        # latest positions snapshot for risk budgeting
        append_sqlite(
            self.ctx.sqlite_path,
            "latest_positions",
            plans_df.assign(date=as_of.isoformat()),
        )
        open_positions = self._build_open_positions_from_plans(
            as_of=as_of,
            plans=plans,
            runtime_mode=runtime_mode,
            hold_days=int(runtime_params["hold_days"]),
        )
        active_positions = list(paper_exec.get("remaining_positions", [])) + open_positions
        self._save_open_paper_positions(as_of=as_of, positions=active_positions)
        manifest_path = write_run_manifest(
            output_dir=self.ctx.output_dir,
            run_type="eod",
            run_id=as_of.isoformat(),
            artifacts={
                "briefing": str(briefing_path),
                "signals": str(signals_path),
                "positions": str(positions_path),
                "mode_feedback": str(mode_feedback_path),
            },
            metrics={
                "signals": int(len(signals)),
                "plans": int(len(plans)),
                "black_swan_score": float(guard.black_swan_score),
                "regime": str(regime.consensus.value),
                "runtime_mode": runtime_mode,
                "risk_multiplier": float(risk_control.get("risk_multiplier", 1.0)),
                "closed_trades": int((paper_exec.get("summary", {}) or {}).get("closed_count", 0)),
                "closed_pnl": float((paper_exec.get("summary", {}) or {}).get("closed_pnl", 0.0)),
                "open_positions": int(len(active_positions)),
            },
            checks={
                "quality_passed": bool(ingest.quality.passed),
                "trade_blocked": bool(guard.trade_blocked),
            },
        )

        return {
            "date": as_of.isoformat(),
            "regime": regime.to_dict(),
            "signals": len(signals),
            "plans": len(plans),
            "briefing": str(briefing_path),
            "manifest": str(manifest_path),
            "quality_passed": ingest.quality.passed,
            "source_confidence_score": float(ingest.quality.source_confidence_score),
            "low_confidence_source_ratio": float(ingest.quality.low_confidence_source_ratio),
            "low_confidence_sources": list(ingest.source_confidence.low_confidence_sources),
            "mode_feedback": str(mode_feedback_path),
            "risk_multiplier": float(risk_control.get("risk_multiplier", 1.0)),
            "non_trade_reasons": guard.non_trade_reasons,
            "black_swan_score": guard.black_swan_score,
            "runtime_mode": runtime_mode,
            "closed_trades": int((paper_exec.get("summary", {}) or {}).get("closed_count", 0)),
            "closed_pnl": float((paper_exec.get("summary", {}) or {}).get("closed_pnl", 0.0)),
            "open_positions": int(len(active_positions)),
        }

    def run_premarket(self, as_of: date) -> dict[str, Any]:
        symbols = self._core_symbols()
        bars, ingest = self._run_ingestion(as_of, symbols)
        regime = self._regime_from_bars(as_of=as_of, bars=bars)
        live_params = self._load_live_params()
        runtime_params = self._resolve_runtime_params(regime=regime, live_params=live_params)
        runtime_mode = str(runtime_params.get("mode", "base"))
        mode_history = self._mode_history_stats(as_of=as_of)
        mode_health = self._evaluate_mode_health(runtime_mode=runtime_mode, mode_history=mode_history)
        risk_control = self._execution_risk_control(
            source_confidence_score=float(ingest.quality.source_confidence_score),
            mode_health=mode_health,
        )
        recent_trades = self._load_recent_trades(300)
        guard = self._evaluate_guards(
            as_of=as_of,
            regime=regime,
            quality_passed=bool(ingest.quality.passed),
            sentiment=ingest.sentiment,
            news=ingest.news,
            recent_trades=recent_trades,
        )

        payload = {
            "date": as_of.isoformat(),
            "slot": "premarket",
            "quality": ingest.quality.to_dict(),
            "source_confidence": ingest.source_confidence.to_dict(),
            "regime": regime.to_dict(),
            "news_count": len(ingest.news),
            "black_swan_score": guard.black_swan_score,
            "major_event_window": guard.major_event_window,
            "runtime_mode": runtime_mode,
            "mode_health": mode_health,
            "risk_control": risk_control,
            "risk_multiplier": float(risk_control.get("risk_multiplier", 1.0)),
            "protection_mode": regime.protection_mode or guard.black_swan_trigger or guard.major_event_window or (not ingest.quality.passed),
        }

        path = self.ctx.output_dir / "logs" / f"{as_of.isoformat()}_premarket.json"
        write_json(path, payload)
        manifest_path = write_run_manifest(
            output_dir=self.ctx.output_dir,
            run_type="premarket",
            run_id=as_of.isoformat(),
            artifacts={"premarket_log": str(path)},
            metrics={
                "runtime_mode": runtime_mode,
                "risk_multiplier": float(risk_control.get("risk_multiplier", 1.0)),
                "source_confidence_score": float(ingest.quality.source_confidence_score),
                "black_swan_score": float(guard.black_swan_score),
                "news_count": int(len(ingest.news)),
            },
            checks={
                "quality_passed": bool(ingest.quality.passed),
                "major_event_window": bool(guard.major_event_window),
                "protection_mode": bool(payload.get("protection_mode", False)),
            },
        )
        payload["manifest"] = str(manifest_path)
        return payload

    def run_intraday_check(self, as_of: date, slot: str) -> dict[str, Any]:
        symbols = self._core_symbols()
        bars, ingest = self._run_ingestion(as_of, symbols)
        regime = self._regime_from_bars(as_of=as_of, bars=bars)
        live_params = self._load_live_params()
        runtime_params = self._resolve_runtime_params(regime=regime, live_params=live_params)
        runtime_mode = str(runtime_params.get("mode", "base"))
        mode_history = self._mode_history_stats(as_of=as_of)
        mode_health = self._evaluate_mode_health(runtime_mode=runtime_mode, mode_history=mode_history)
        risk_control = self._execution_risk_control(
            source_confidence_score=float(ingest.quality.source_confidence_score),
            mode_health=mode_health,
        )
        recent_trades = self._load_recent_trades(300)
        guard = self._evaluate_guards(
            as_of=as_of,
            regime=regime,
            quality_passed=bool(ingest.quality.passed),
            sentiment=ingest.sentiment,
            news=ingest.news,
            recent_trades=recent_trades,
        )

        if slot not in {"10:30", "14:30"}:
            slot = "10:30"

        payload = {
            "date": as_of.isoformat(),
            "slot": slot,
            "atr_z": regime.atr_z,
            "protection_mode": regime.protection_mode or guard.black_swan_trigger or guard.major_event_window or (not ingest.quality.passed),
            "quality_flags": ingest.quality.flags,
            "source_confidence_score": float(ingest.quality.source_confidence_score),
            "low_confidence_source_ratio": float(ingest.quality.low_confidence_source_ratio),
            "runtime_mode": runtime_mode,
            "mode_health": mode_health,
            "risk_control": risk_control,
            "risk_multiplier": float(risk_control.get("risk_multiplier", 1.0)),
            "black_swan_score": guard.black_swan_score,
            "major_event_window": guard.major_event_window,
        }
        path = self.ctx.output_dir / "logs" / f"{as_of.isoformat()}_intraday_{slot.replace(':', '')}.json"
        write_json(path, payload)
        run_id = f"{as_of.isoformat()}_{slot.replace(':', '')}"
        manifest_path = write_run_manifest(
            output_dir=self.ctx.output_dir,
            run_type="intraday_check",
            run_id=run_id,
            artifacts={"intraday_log": str(path)},
            metrics={
                "runtime_mode": runtime_mode,
                "risk_multiplier": float(risk_control.get("risk_multiplier", 1.0)),
                "source_confidence_score": float(ingest.quality.source_confidence_score),
                "black_swan_score": float(guard.black_swan_score),
                "atr_z": float(regime.atr_z),
            },
            checks={
                "quality_passed": bool(ingest.quality.passed),
                "major_event_window": bool(guard.major_event_window),
                "protection_mode": bool(payload.get("protection_mode", False)),
            },
        )
        payload["manifest"] = str(manifest_path)
        return payload

    def run_backtest(self, start: date, end: date) -> BacktestResult:
        symbols = self._core_symbols()
        # include dynamic candidates for wider stress
        symbols = expand_universe(symbols, pd.DataFrame(), int(self.settings.universe.get("max_dynamic_additions", 5)))
        bars, _ = self._run_ingestion_range(start=start, end=end, symbols=symbols)
        params = self._load_live_params()
        runtime_regime = self._regime_from_bars(as_of=end, bars=bars)
        runtime_params = self._resolve_runtime_params(regime=runtime_regime, live_params=params)
        signal_conf = float(runtime_params["signal_confidence_min"])
        convexity_min = float(runtime_params["convexity_min"])
        hold_days = self._clamp_int(runtime_params["hold_days"], 1, 20)
        max_daily_trades = self._clamp_int(runtime_params["max_daily_trades"], 1, 5)
        bt_cfg = BacktestConfig(
            signal_confidence_min=signal_conf,
            convexity_min=convexity_min,
            max_daily_trades=0 if signal_conf >= 90.0 else max_daily_trades,
            hold_days=hold_days,
        )

        result = run_walk_forward_backtest(
            bars=bars,
            start=start,
            end=end,
            trend_thr=float(self.settings.thresholds.get("hurst_trend", 0.6)),
            mean_thr=float(self.settings.thresholds.get("hurst_mean_revert", 0.4)),
            atr_extreme=float(self.settings.thresholds.get("atr_extreme", 2.0)),
            cfg_template=bt_cfg,
            train_years=3,
            valid_years=1,
            step_months=3,
        )

        out_path = self.ctx.output_dir / "artifacts" / f"backtest_{start.isoformat()}_{end.isoformat()}.json"
        write_json(out_path, result.to_dict())
        append_sqlite(self.ctx.sqlite_path, "backtest_runs", pd.DataFrame([result.to_dict()]))
        write_run_manifest(
            output_dir=self.ctx.output_dir,
            run_type="backtest",
            run_id=f"{start.isoformat()}_{end.isoformat()}",
            artifacts={"result": str(out_path)},
            metrics={
                "annual_return": float(result.annual_return),
                "max_drawdown": float(result.max_drawdown),
                "win_rate": float(result.win_rate),
                "profit_factor": float(result.profit_factor),
                "trades": int(result.trades),
                "runtime_mode": str(runtime_params.get("mode", "base")),
            },
            checks={"violations": int(result.violations)},
        )
        return result

    def run_research_backtest(
        self,
        *,
        start: date,
        end: date,
        hours_budget: float = 10.0,
        max_symbols: int = 120,
        report_symbol_cap: int = 40,
        workers: int = 8,
        max_trials_per_mode: int = 500,
        seed: int = 42,
        modes: list[str] | None = None,
        review_days: int = 5,
    ) -> dict[str, Any]:
        summary = run_research_pipeline(
            output_root=self.ctx.output_dir,
            core_symbols=self._core_symbols(),
            start=start,
            end=end,
            hours_budget=hours_budget,
            max_symbols=max_symbols,
            report_symbol_cap=report_symbol_cap,
            workers=workers,
            max_trials_per_mode=max_trials_per_mode,
            seed=seed,
            modes=modes,
            review_days=review_days,
        )
        payload = summary.to_dict()
        manifest_path = write_run_manifest(
            output_dir=self.ctx.output_dir,
            run_type="research_backtest",
            run_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            artifacts={
                "summary": str(Path(summary.output_dir) / "summary.json"),
                "report": str(Path(summary.output_dir) / "report.md"),
                "best_params": str(Path(summary.output_dir) / "best_params.yaml"),
            },
            metrics={
                "hours_budget": float(payload.get("hours_budget", 0.0)),
                "elapsed_seconds": float(payload.get("elapsed_seconds", 0.0)),
                "universe_count": int(payload.get("universe_count", 0)),
                "bars_rows": int(payload.get("bars_rows", 0)),
            },
            checks={
                "strict_cutoff_enforced": bool(payload.get("data_fetch_stats", {}).get("strict_cutoff_enforced", False)),
                "mode_count": int(len(payload.get("mode_summaries", []))),
            },
            metadata={
                "cutoff_date": str(payload.get("cutoff_date", "")),
                "review_days": int(payload.get("review_days", 0)),
            },
        )
        payload["manifest"] = str(manifest_path)
        return payload

    def run_strategy_lab(
        self,
        *,
        start: date,
        end: date,
        max_symbols: int = 120,
        report_symbol_cap: int = 40,
        workers: int = 8,
        review_days: int = 5,
        candidate_count: int = 10,
    ) -> dict[str, Any]:
        summary = run_strategy_lab_pipeline(
            output_root=self.ctx.output_dir,
            core_symbols=self._core_symbols(),
            start=start,
            end=end,
            max_symbols=max_symbols,
            report_symbol_cap=report_symbol_cap,
            workers=workers,
            review_days=review_days,
            candidate_count=candidate_count,
        )
        payload = summary.to_dict()
        manifest_path = write_run_manifest(
            output_dir=self.ctx.output_dir,
            run_type="strategy_lab",
            run_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            artifacts={
                "summary": str(Path(summary.output_dir) / "summary.json"),
                "report": str(Path(summary.output_dir) / "report.md"),
                "best_strategy": str(Path(summary.output_dir) / "best_strategy.yaml"),
            },
            metrics={
                "universe_count": int(payload.get("universe_count", 0)),
                "bars_rows": int(payload.get("bars_rows", 0)),
                "candidate_count": int(len(payload.get("candidates", []))),
            },
            checks={
                "has_best_candidate": bool(payload.get("best_candidate", {})),
                "strict_cutoff_enforced": bool(payload.get("data_fetch_stats", {}).get("strict_cutoff_enforced", False)),
            },
            metadata={
                "cutoff_date": str(payload.get("cutoff_date", "")),
                "review_days": int(payload.get("review_days", 0)),
            },
        )
        payload["manifest"] = str(manifest_path)
        return payload

    def _load_live_params(self) -> dict[str, float]:
        p = self.ctx.output_dir / "artifacts" / "params_live.yaml"
        if not p.exists():
            return {
                "win_rate": 0.45,
                "payoff": 2.0,
                "signal_confidence_min": float(self.settings.thresholds.get("signal_confidence_min", 60.0)),
                "convexity_min": float(self.settings.thresholds.get("convexity_min", 3.0)),
                "hold_days": 5.0,
                "max_daily_trades": 2.0,
            }
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))}

    def _backup_live_params(self, as_of: date) -> str:
        p = self.ctx.output_dir / "artifacts" / "params_live.yaml"
        if not p.exists():
            return "initial_seed"
        backup = self.ctx.output_dir / "artifacts" / f"params_live_backup_{as_of.isoformat()}.yaml"
        write_markdown(backup, p.read_text(encoding="utf-8"))
        return str(backup)

    def _resolve_review_runtime_mode(self, *, start: date, as_of: date) -> str:
        manifest_path = self.ctx.output_dir / "artifacts" / "manifests" / f"backtest_{start.isoformat()}_{as_of.isoformat()}.json"
        payload = self._load_json_safely(manifest_path)
        metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics", {}), dict) else {}
        mode = str(metrics.get("runtime_mode", "base")).strip() or "base"
        return mode

    def _evaluate_mode_health(self, *, runtime_mode: str, mode_history: dict[str, Any]) -> dict[str, Any]:
        validation = self.settings.validation if isinstance(self.settings.validation, dict) else {}
        modes = mode_history.get("modes", {}) if isinstance(mode_history.get("modes", {}), dict) else {}
        stats = modes.get(runtime_mode, {})
        if not isinstance(stats, dict):
            stats = {}

        samples = int(stats.get("samples", 0))
        min_samples = max(1, int(validation.get("mode_health_min_samples", 1)))
        min_profit_factor = float(validation.get("mode_health_min_profit_factor", 1.0))
        min_win_rate = float(validation.get("mode_health_min_win_rate", 0.40))
        max_drawdown = float(validation.get("mode_health_max_drawdown_max", validation.get("max_drawdown_max", 0.18)))
        max_violations = max(0, int(validation.get("mode_health_max_violations", 0)))

        if samples < min_samples:
            return {
                "passed": True,
                "active": False,
                "runtime_mode": runtime_mode,
                "reason": "insufficient_samples",
                "samples": samples,
                "min_samples": min_samples,
            }

        pf = float(stats.get("avg_profit_factor", 0.0))
        wr = float(stats.get("avg_win_rate", 0.0))
        dd = float(stats.get("worst_drawdown", 1.0))
        viol = int(stats.get("total_violations", 0))
        failed_rules: list[str] = []
        if pf < min_profit_factor:
            failed_rules.append("profit_factor")
        if wr < min_win_rate:
            failed_rules.append("win_rate")
        if dd > max_drawdown:
            failed_rules.append("max_drawdown")
        if viol > max_violations:
            failed_rules.append("violations")

        return {
            "passed": len(failed_rules) == 0,
            "active": True,
            "runtime_mode": runtime_mode,
            "samples": samples,
            "thresholds": {
                "min_profit_factor": min_profit_factor,
                "min_win_rate": min_win_rate,
                "max_drawdown_max": max_drawdown,
                "max_violations": max_violations,
                "min_samples": min_samples,
            },
            "stats": {
                "avg_profit_factor": pf,
                "avg_win_rate": wr,
                "worst_drawdown": dd,
                "total_violations": viol,
            },
            "failed_rules": failed_rules,
        }

    def _apply_mode_health_guard(
        self,
        *,
        review: ReviewDelta,
        current_params: dict[str, float],
        mode_health: dict[str, Any],
    ) -> None:
        if bool(mode_health.get("passed", True)):
            return
        if "MODE_HEALTH_DEGRADED" not in review.defects:
            review.defects.append("MODE_HEALTH_DEGRADED")
        review.pass_gate = False

        conf = float(review.parameter_changes.get("signal_confidence_min", current_params.get("signal_confidence_min", 60.0)))
        tightened_conf = self._clamp_float(conf + 3.0, 50.0, 90.0)
        review.parameter_changes["signal_confidence_min"] = tightened_conf
        review.change_reasons["signal_confidence_min"] = (
            str(review.change_reasons.get("signal_confidence_min", "")).strip()
            + "；模式健康劣化触发保护收敛（提升信号置信阈值）"
        ).strip("；")

        trades = self._clamp_int(
            review.parameter_changes.get("max_daily_trades", current_params.get("max_daily_trades", 2.0)),
            1,
            5,
        )
        reduced_trades = float(max(1, trades - 1))
        review.parameter_changes["max_daily_trades"] = reduced_trades
        review.change_reasons["max_daily_trades"] = "模式健康劣化触发保护收敛（下调单日交易次数）"

        hold_days = self._clamp_int(
            review.parameter_changes.get("hold_days", current_params.get("hold_days", 5.0)),
            1,
            20,
        )
        shorter_hold = float(max(1, hold_days - 2))
        review.parameter_changes["hold_days"] = shorter_hold
        review.change_reasons["hold_days"] = "模式健康劣化触发保护收敛（缩短持有期暴露）"

        failed_rules = mode_health.get("failed_rules", [])
        review.notes.append(
            "mode_health_guard="
            + f"runtime_mode={mode_health.get('runtime_mode', 'base')}; "
            + f"failed_rules={failed_rules}; samples={mode_health.get('samples', 0)}"
        )

    def _apply_mode_adaptive_update(
        self,
        *,
        review: ReviewDelta,
        current_params: dict[str, float],
        runtime_mode: str,
        mode_history: dict[str, Any],
    ) -> dict[str, Any]:
        val = self.settings.validation if isinstance(self.settings.validation, dict) else {}
        enabled = bool(val.get("mode_adaptive_update_enabled", True))
        summary: dict[str, Any] = {
            "enabled": enabled,
            "applied": False,
            "runtime_mode": runtime_mode,
        }
        if not enabled:
            summary["reason"] = "disabled"
            return summary

        modes = mode_history.get("modes", {}) if isinstance(mode_history.get("modes", {}), dict) else {}
        stats = modes.get(runtime_mode, {})
        if not isinstance(stats, dict) or not stats:
            summary["reason"] = "missing_mode_stats"
            return summary

        min_samples = max(1, int(val.get("mode_adaptive_update_min_samples", 3)))
        samples = int(stats.get("samples", 0))
        summary["samples"] = samples
        summary["min_samples"] = min_samples
        if samples < min_samples:
            summary["reason"] = "insufficient_samples"
            return summary

        step = self._clamp_float(float(val.get("mode_adaptive_update_step", 0.08)), 0.01, 0.50)
        good_pf = float(val.get("mode_adaptive_good_profit_factor", 1.25))
        bad_pf = float(val.get("mode_adaptive_bad_profit_factor", 1.00))
        good_wr = self._clamp_float(float(val.get("mode_adaptive_good_win_rate", 0.50)), 0.0, 1.0)
        bad_wr = self._clamp_float(float(val.get("mode_adaptive_bad_win_rate", 0.42)), 0.0, 1.0)
        default_bad_dd = float(val.get("max_drawdown_max", 0.18))
        good_dd = self._clamp_float(float(val.get("mode_adaptive_good_drawdown_max", default_bad_dd * 0.67)), 0.0, 1.0)
        bad_dd = self._clamp_float(float(val.get("mode_adaptive_bad_drawdown_max", default_bad_dd)), 0.0, 1.0)

        pf = float(stats.get("avg_profit_factor", 0.0))
        wr = float(stats.get("avg_win_rate", 0.0))
        dd = float(stats.get("worst_drawdown", 1.0))
        viol = int(stats.get("total_violations", 0))
        summary["stats"] = {
            "avg_profit_factor": pf,
            "avg_win_rate": wr,
            "worst_drawdown": dd,
            "total_violations": viol,
        }

        direction = "neutral"
        if pf >= good_pf and wr >= good_wr and dd <= good_dd and viol == 0:
            direction = "expand"
        elif pf < bad_pf or wr < bad_wr or dd > bad_dd or viol > 0:
            direction = "tighten"
        summary["direction"] = direction
        summary["thresholds"] = {
            "good_profit_factor": good_pf,
            "bad_profit_factor": bad_pf,
            "good_win_rate": good_wr,
            "bad_win_rate": bad_wr,
            "good_drawdown_max": good_dd,
            "bad_drawdown_max": bad_dd,
            "step": step,
        }
        if direction == "neutral":
            summary["reason"] = "no_regime_shift_signal"
            return summary

        profiles = self._resolved_mode_profiles()
        profile = profiles.get(runtime_mode, profiles.get("swing", self._default_mode_profiles()["swing"]))

        base_conf = float(review.parameter_changes.get("signal_confidence_min", current_params.get("signal_confidence_min", 60.0)))
        base_conv = float(review.parameter_changes.get("convexity_min", current_params.get("convexity_min", self.settings.thresholds.get("convexity_min", 3.0))))
        base_hold = float(review.parameter_changes.get("hold_days", current_params.get("hold_days", 5.0)))
        base_trades = float(review.parameter_changes.get("max_daily_trades", current_params.get("max_daily_trades", 2.0)))

        if direction == "expand":
            evidence_conf = min(base_conf - 1.5, float(profile.get("signal_confidence_min", base_conf)))
            evidence_conv = min(base_conv - 0.10, float(profile.get("convexity_min", base_conv)))
            evidence_hold = max(base_hold + 1.0, float(profile.get("hold_days", base_hold)))
            evidence_trades = max(base_trades + 1.0, float(profile.get("max_daily_trades", base_trades)))
        else:
            evidence_conf = max(base_conf + 3.0, float(profile.get("signal_confidence_min", base_conf)))
            evidence_conv = max(base_conv + 0.25, float(profile.get("convexity_min", base_conv)))
            evidence_hold = min(base_hold - 1.0, float(profile.get("hold_days", base_hold)))
            evidence_trades = min(base_trades - 1.0, float(profile.get("max_daily_trades", base_trades)))

        new_conf = bounded_bayesian_update(base_conf, evidence_conf, 30.0, 90.0, step=step)
        new_conv = bounded_bayesian_update(base_conv, evidence_conv, 1.0, 4.0, step=step)
        new_hold = float(self._clamp_int(bounded_bayesian_update(base_hold, evidence_hold, 1.0, 20.0, step=step), 1, 20))
        new_trades = float(self._clamp_int(bounded_bayesian_update(base_trades, evidence_trades, 1.0, 5.0, step=step), 1, 5))

        review.parameter_changes["signal_confidence_min"] = float(new_conf)
        review.parameter_changes["convexity_min"] = float(new_conv)
        review.parameter_changes["hold_days"] = float(new_hold)
        review.parameter_changes["max_daily_trades"] = float(new_trades)
        reason = f"模式自适应更新（mode={runtime_mode}, direction={direction}, step={step:.2f}）"
        for key in ("signal_confidence_min", "convexity_min", "hold_days", "max_daily_trades"):
            prev_reason = str(review.change_reasons.get(key, "")).strip()
            review.change_reasons[key] = (prev_reason + "；" + reason).strip("；")
        review.notes.append(
            "mode_adaptive_update="
            + f"mode={runtime_mode}; direction={direction}; samples={samples}; "
            + f"pf={pf:.3f}; wr={wr:.2%}; dd={dd:.2%}; viol={viol}; step={step:.2f}"
        )

        summary["applied"] = True
        summary["updates"] = {
            "signal_confidence_min": float(new_conf),
            "convexity_min": float(new_conv),
            "hold_days": float(new_hold),
            "max_daily_trades": float(new_trades),
        }
        return summary

    def _execution_risk_control(
        self,
        *,
        source_confidence_score: float,
        mode_health: dict[str, Any],
    ) -> dict[str, Any]:
        val = self.settings.validation if isinstance(self.settings.validation, dict) else {}
        min_mult = self._clamp_float(float(val.get("execution_min_risk_multiplier", 0.2)), 0.0, 1.0)
        source_floor = self._clamp_float(float(val.get("source_confidence_floor_risk_multiplier", 0.35)), min_mult, 1.0)
        mode_degraded_mult = self._clamp_float(float(val.get("mode_health_risk_multiplier", 0.5)), min_mult, 1.0)
        mode_unknown_mult = self._clamp_float(float(val.get("mode_health_insufficient_sample_risk_multiplier", 0.85)), min_mult, 1.0)
        source_min = self._clamp_float(float(val.get("source_confidence_min", 0.75)), 0.01, 1.0)

        score = self._clamp_float(float(source_confidence_score), 0.0, 1.0)
        source_mult = 1.0
        if score < source_min:
            source_mult = source_floor + (1.0 - source_floor) * (score / source_min)
        source_mult = self._clamp_float(source_mult, min_mult, 1.0)

        mode_mult = 1.0
        mode_reason = "healthy"
        if not bool(mode_health.get("passed", True)):
            mode_mult = mode_degraded_mult
            mode_reason = "degraded"
        elif not bool(mode_health.get("active", False)):
            mode_mult = mode_unknown_mult
            mode_reason = "insufficient_samples"

        final_mult = self._clamp_float(source_mult * mode_mult, min_mult, 1.0)
        return {
            "risk_multiplier": final_mult,
            "source_multiplier": source_mult,
            "mode_multiplier": mode_mult,
            "mode_reason": mode_reason,
            "source_confidence_score": score,
        }

    def run_review(self, as_of: date) -> ReviewDelta:
        start = self._review_backtest_start(as_of=as_of)
        bt = self.run_backtest(start=start, end=as_of)
        params = self._load_live_params()
        rollback_anchor = self._backup_live_params(as_of)
        factor_weights = {
            "macro": 0.20,
            "industry": 0.18,
            "news": 0.17,
            "sentiment": 0.15,
            "fundamental": 0.10,
            "technical": 0.20,
        }
        factor_contrib = self._estimate_factor_contrib_120d(as_of=as_of)
        thresholds = ReviewThresholds(
            positive_window_ratio_min=float(self.settings.validation.get("positive_window_ratio_min", 0.70)),
            max_drawdown_max=float(self.settings.validation.get("max_drawdown_max", 0.18)),
        )
        review = build_review_delta(
            as_of=as_of,
            backtest=bt,
            current_params=params,
            factor_weights=factor_weights,
            factor_contrib=factor_contrib,
            thresholds=thresholds,
        )
        strategy_candidate = self._load_latest_strategy_candidate(as_of=as_of)
        strategy_lab_autorun: dict[str, Any] = {}
        if not strategy_candidate and bool(self.settings.validation.get("review_autorun_strategy_lab_if_missing", False)):
            strategy_lab_autorun = self._autorun_strategy_lab_for_review(as_of=as_of)
            if strategy_lab_autorun:
                strategy_candidate = self._load_latest_strategy_candidate(as_of=as_of)
                review.notes.append(
                    "strategy_lab_autorun="
                    + f"manifest={strategy_lab_autorun.get('manifest', '')}; "
                    + f"candidate_count={len(strategy_lab_autorun.get('candidates', []))}"
                )
        if strategy_candidate:
            self._merge_strategy_candidate_into_review(
                review=review,
                current_params=params,
                candidate_payload=strategy_candidate,
            )
        else:
            review.notes.append("No accepted strategy-lab candidate found within lookback window")

        runtime_mode = self._resolve_review_runtime_mode(start=start, as_of=as_of)
        mode_history = self._mode_history_stats(as_of=as_of)
        mode_health = self._evaluate_mode_health(runtime_mode=runtime_mode, mode_history=mode_history)
        mode_adaptive = self._apply_mode_adaptive_update(
            review=review,
            current_params=params,
            runtime_mode=runtime_mode,
            mode_history=mode_history,
        )
        self._apply_mode_health_guard(
            review=review,
            current_params=params,
            mode_health=mode_health,
        )
        review.notes.append(
            "mode_health="
            + f"runtime_mode={runtime_mode}; passed={mode_health.get('passed', True)}; "
            + f"active={mode_health.get('active', False)}"
        )
        review.rollback_anchor = rollback_anchor

        report = render_review_report(as_of=as_of, backtest=bt, review=review)
        review_dir = self.ctx.output_dir / "review"
        review_path = review_dir / f"{as_of.isoformat()}_review.md"
        delta_path = review_dir / f"{as_of.isoformat()}_param_delta.yaml"
        write_markdown(review_path, report)

        prev_params = params
        deltas = {
            k: float(review.parameter_changes.get(k, 0.0) - prev_params.get(k, 0.0))
            for k in review.parameter_changes.keys()
        }
        audit_payload = review.to_dict()
        audit_payload["previous_parameters"] = prev_params
        audit_payload["parameter_deltas"] = deltas
        audit_payload["impact_window_days"] = review.impact_window_days
        audit_payload["rollback_anchor"] = rollback_anchor
        audit_payload["strategy_lab_candidate"] = strategy_candidate
        audit_payload["strategy_lab_autorun"] = strategy_lab_autorun
        audit_payload["runtime_mode"] = runtime_mode
        audit_payload["mode_history"] = mode_history
        audit_payload["mode_health"] = mode_health
        audit_payload["mode_adaptive"] = mode_adaptive
        audit_payload["generated_at"] = datetime.now().isoformat()
        write_markdown(delta_path, yaml.safe_dump(audit_payload, allow_unicode=True, sort_keys=False))

        live_params_path = self.ctx.output_dir / "artifacts" / "params_live.yaml"
        write_markdown(live_params_path, yaml.safe_dump(review.parameter_changes, allow_unicode=True, sort_keys=False))

        append_sqlite(self.ctx.sqlite_path, "review_runs", pd.DataFrame([audit_payload]))
        manifest_path = write_run_manifest(
            output_dir=self.ctx.output_dir,
            run_type="review",
            run_id=as_of.isoformat(),
            artifacts={
                "review_report": str(review_path),
                "param_delta": str(delta_path),
                "live_params": str(live_params_path),
            },
            metrics={
                "pass_gate": bool(review.pass_gate),
                "defects": int(len(review.defects)),
                "changed_params": int(len(review.parameter_changes)),
                "runtime_mode": runtime_mode,
                "mode_health_passed": bool(mode_health.get("passed", True)),
                "mode_adaptive_applied": bool(mode_adaptive.get("applied", False)),
                "mode_adaptive_direction": str(mode_adaptive.get("direction", "none")),
            },
            checks={
                "positive_window_ratio": float(bt.positive_window_ratio),
                "max_drawdown": float(bt.max_drawdown),
                "violations": int(bt.violations),
            },
        )
        review.notes.append(f"manifest={manifest_path}")
        return review

    def test_all(
        self,
        *,
        fast: bool = False,
        fast_ratio: float = 0.10,
        fast_shard_index: int = 0,
        fast_shard_total: int = 1,
        fast_seed: str = "lie-fast-v1",
    ) -> dict[str, Any]:
        return self._testing_orchestrator().test_all(
            fast=fast,
            fast_ratio=fast_ratio,
            fast_shard_index=fast_shard_index,
            fast_shard_total=fast_shard_total,
            fast_seed=fast_seed,
        )

    def stable_replay_check(self, as_of: date, days: int | None = None) -> dict[str, Any]:
        return self._observability_orchestrator().stable_replay_check(as_of=as_of, days=days)

    def _release_orchestrator(self) -> ReleaseOrchestrator:
        return ReleaseOrchestrator(
            settings=self.settings,
            output_dir=self.ctx.output_dir,
            quality_snapshot=lambda d: self._quality_snapshot(d),
            backtest_snapshot=lambda d: self._backtest_snapshot(d),
            run_review=lambda d: self.run_review(d),
            health_check=lambda d, require_review: self.health_check(d, require_review=require_review),
            stable_replay_check=lambda d, days: self.stable_replay_check(d, days=days),
            test_all=lambda **kwargs: self.test_all(**kwargs),
            load_json_safely=lambda p: self._load_json_safely(p),
            sqlite_path=self.ctx.sqlite_path,
        )

    def _architecture_orchestrator(self) -> ArchitectureOrchestrator:
        return ArchitectureOrchestrator(
            settings=self.settings,
            output_dir=self.ctx.output_dir,
            health_check=lambda d, require_review: self.health_check(d, require_review=require_review),
        )

    def _dependency_orchestrator(self) -> DependencyOrchestrator:
        return DependencyOrchestrator(
            settings=self.settings,
            source_root=self.ctx.root / "src",
            output_dir=self.ctx.output_dir,
        )

    def _observability_orchestrator(self) -> ObservabilityOrchestrator:
        return ObservabilityOrchestrator(
            settings=self.settings,
            output_dir=self.ctx.output_dir,
            sqlite_path=self.ctx.sqlite_path,
            run_eod=lambda d: self.run_eod(d),
        )

    def _testing_orchestrator(self) -> TestingOrchestrator:
        return TestingOrchestrator(
            root=self.ctx.root,
            output_dir=self.ctx.output_dir,
        )

    def _scheduler_orchestrator(self) -> SchedulerOrchestrator:
        return SchedulerOrchestrator(
            settings=self.settings,
            output_dir=self.ctx.output_dir,
            run_premarket=lambda d: self.run_premarket(d),
            run_intraday_check=lambda d, slot: self.run_intraday_check(d, slot=slot),
            run_eod=lambda d: self.run_eod(d),
            run_review_cycle=lambda d, max_rounds: self.run_review_cycle(as_of=d, max_rounds=max_rounds),
            ops_report=lambda d, window_days: self.ops_report(as_of=d, window_days=window_days),
        )

    def gate_report(
        self,
        as_of: date,
        run_tests: bool = False,
        run_review_if_missing: bool = True,
    ) -> dict[str, Any]:
        return self._release_orchestrator().gate_report(
            as_of=as_of,
            run_tests=run_tests,
            run_review_if_missing=run_review_if_missing,
        )

    def ops_report(self, as_of: date, window_days: int = 7) -> dict[str, Any]:
        return self._release_orchestrator().ops_report(as_of=as_of, window_days=window_days)

    def review_until_pass(self, as_of: date, max_rounds: int = 3) -> dict[str, Any]:
        return self._release_orchestrator().review_until_pass(as_of=as_of, max_rounds=max_rounds)

    def run_review_cycle(self, as_of: date, max_rounds: int = 2, ops_window_days: int | None = None) -> dict[str, Any]:
        return self._release_orchestrator().run_review_cycle(
            as_of=as_of,
            max_rounds=max_rounds,
            ops_window_days=ops_window_days,
        )

    def run_slot(self, as_of: date, slot: str, max_review_rounds: int = 2) -> dict[str, Any]:
        return self._scheduler_orchestrator().run_slot(
            as_of=as_of,
            slot=slot,
            max_review_rounds=max_review_rounds,
        )

    def run_session(self, as_of: date, include_review: bool = True, max_review_rounds: int = 2) -> dict[str, Any]:
        return self._scheduler_orchestrator().run_session(
            as_of=as_of,
            include_review=include_review,
            max_review_rounds=max_review_rounds,
        )

    def run_daemon(
        self,
        poll_seconds: int = 30,
        max_cycles: int | None = None,
        max_review_rounds: int = 2,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        return self._scheduler_orchestrator().run_daemon(
            poll_seconds=poll_seconds,
            max_cycles=max_cycles,
            max_review_rounds=max_review_rounds,
            dry_run=dry_run,
        )

    def health_check(self, as_of: date | None = None, require_review: bool = True) -> dict[str, Any]:
        return self._observability_orchestrator().health_check(as_of=as_of, require_review=require_review)

    def architecture_audit(self, as_of: date | None = None) -> dict[str, Any]:
        return self._architecture_orchestrator().architecture_audit(as_of=as_of)

    def dependency_audit(self, as_of: date | None = None) -> dict[str, Any]:
        return self._dependency_orchestrator().dependency_audit(as_of=as_of)
