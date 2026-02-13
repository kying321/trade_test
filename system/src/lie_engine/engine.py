from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time as time_module
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

from lie_engine.backtest import BacktestConfig, run_walk_forward_backtest
from lie_engine.config import SystemSettings, load_settings
from lie_engine.data import DataBus, OpenSourcePrimaryProvider, OpenSourceSecondaryProvider
from lie_engine.data.storage import append_sqlite, write_csv, write_json, write_markdown
from lie_engine.models import BacktestResult, NewsEvent, RegimeLabel, ReviewDelta, TradePlan
from lie_engine.orchestration import GuardAssessment, estimate_factor_contrib_120d
from lie_engine.orchestration.guards import black_swan_assessment, loss_cooldown_active, major_event_window
from lie_engine.regime import compute_atr_zscore, derive_regime_consensus, infer_hmm_state, latest_multi_scale_hurst
from lie_engine.reporting import render_daily_briefing, render_review_report
from lie_engine.review import ReviewThresholds, build_review_delta
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
        root = Path(config_path).resolve().parent if config_path else Path(__file__).resolve().parents[2]
        output_dir = root / settings.paths.get("output", "output")
        sqlite_path = root / settings.paths.get("sqlite", "output/artifacts/lie_engine.db")
        self.ctx = EngineContext(settings=settings, root=root, output_dir=output_dir, sqlite_path=sqlite_path)

        self.providers = [OpenSourcePrimaryProvider(), OpenSourceSecondaryProvider()]
        self.data_bus = DataBus(
            providers=self.providers,
            output_dir=self.ctx.output_dir,
            sqlite_path=self.ctx.sqlite_path,
            completeness_min=float(settings.validation.get("data_completeness_min", 0.99)),
            conflict_max=float(settings.validation.get("unresolved_conflict_max", 0.005)),
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

    def _schedule_cfg(self) -> dict[str, Any]:
        schedule = self.settings.schedule
        return {
            "premarket": str(schedule.get("premarket", "08:40")),
            "intraday_slots": [str(x) for x in schedule.get("intraday_slots", ["10:30", "14:30"])],
            "eod": str(schedule.get("eod", "15:10")),
            "nightly_review": str(schedule.get("nightly_review", "20:30")),
        }

    @staticmethod
    def _parse_hhmm(slot: str) -> time:
        hh, mm = slot.split(":")
        return time(hour=int(hh), minute=int(mm))

    @staticmethod
    def _load_json_safely(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

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

    def _symbol_exposure_snapshot(self) -> tuple[dict[str, float], dict[str, float], float]:
        if not self.ctx.sqlite_path.exists():
            return {}, {}, 0.0
        with closing(sqlite3.connect(self.ctx.sqlite_path)) as conn:
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
        regime = self._regime_from_bars(as_of=as_of, bars=bars)

        signal_cfg = SignalEngineConfig(
            confidence_min=float(self.settings.thresholds.get("signal_confidence_min", 60.0)),
            convexity_min=float(self.settings.thresholds.get("convexity_min", 3.0)),
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
        )

        daily_dir = self.ctx.output_dir / "daily"
        briefing_path = daily_dir / f"{as_of.isoformat()}_briefing.md"
        signals_path = daily_dir / f"{as_of.isoformat()}_signals.json"
        positions_path = daily_dir / f"{as_of.isoformat()}_positions.csv"

        write_markdown(briefing_path, daily_md)
        write_json(signals_path, [s.to_dict() for s in signals])
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

        return {
            "date": as_of.isoformat(),
            "regime": regime.to_dict(),
            "signals": len(signals),
            "plans": len(plans),
            "briefing": str(briefing_path),
            "quality_passed": ingest.quality.passed,
            "non_trade_reasons": guard.non_trade_reasons,
            "black_swan_score": guard.black_swan_score,
        }

    def run_premarket(self, as_of: date) -> dict[str, Any]:
        symbols = self._core_symbols()
        bars, ingest = self._run_ingestion(as_of, symbols)
        regime = self._regime_from_bars(as_of=as_of, bars=bars)
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
            "regime": regime.to_dict(),
            "news_count": len(ingest.news),
            "black_swan_score": guard.black_swan_score,
            "major_event_window": guard.major_event_window,
            "protection_mode": regime.protection_mode or guard.black_swan_trigger or guard.major_event_window or (not ingest.quality.passed),
        }

        path = self.ctx.output_dir / "logs" / f"{as_of.isoformat()}_premarket.json"
        write_json(path, payload)
        return payload

    def run_intraday_check(self, as_of: date, slot: str) -> dict[str, Any]:
        symbols = self._core_symbols()
        bars, ingest = self._run_ingestion(as_of, symbols)
        regime = self._regime_from_bars(as_of=as_of, bars=bars)
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
            "black_swan_score": guard.black_swan_score,
            "major_event_window": guard.major_event_window,
        }
        path = self.ctx.output_dir / "logs" / f"{as_of.isoformat()}_intraday_{slot.replace(':', '')}.json"
        write_json(path, payload)
        return payload

    def run_backtest(self, start: date, end: date) -> BacktestResult:
        symbols = self._core_symbols()
        # include dynamic candidates for wider stress
        symbols = expand_universe(symbols, pd.DataFrame(), int(self.settings.universe.get("max_dynamic_additions", 5)))
        bars, _ = self._run_ingestion_range(start=start, end=end, symbols=symbols)
        params = self._load_live_params()
        signal_conf = float(params.get("signal_confidence_min", self.settings.thresholds.get("signal_confidence_min", 60.0)))
        bt_cfg = BacktestConfig(
            signal_confidence_min=signal_conf,
            convexity_min=float(self.settings.thresholds.get("convexity_min", 3.0)),
            max_daily_trades=0 if signal_conf >= 90.0 else 2,
            hold_days=5,
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
        return result

    def _load_live_params(self) -> dict[str, float]:
        p = self.ctx.output_dir / "artifacts" / "params_live.yaml"
        if not p.exists():
            return {
                "win_rate": 0.45,
                "payoff": 2.0,
                "signal_confidence_min": float(self.settings.thresholds.get("signal_confidence_min", 60.0)),
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

    def run_review(self, as_of: date) -> ReviewDelta:
        start = date(2015, 1, 1)
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
        audit_payload["generated_at"] = datetime.now().isoformat()
        write_markdown(delta_path, yaml.safe_dump(audit_payload, allow_unicode=True, sort_keys=False))

        live_params_path = self.ctx.output_dir / "artifacts" / "params_live.yaml"
        write_markdown(live_params_path, yaml.safe_dump(review.parameter_changes, allow_unicode=True, sort_keys=False))

        append_sqlite(self.ctx.sqlite_path, "review_runs", pd.DataFrame([audit_payload]))
        return review

    def test_all(self) -> dict[str, Any]:
        env = dict(os.environ)
        env["PYTHONWARNINGS"] = "ignore::ResourceWarning"
        proc = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-t", ".", "-v"],
            cwd=self.ctx.root,
            text=True,
            capture_output=True,
            env=env,
        )
        payload = {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
        log_path = self.ctx.output_dir / "logs" / f"tests_{datetime.now():%Y%m%d_%H%M%S}.json"
        write_json(log_path, payload)
        return payload

    def stable_replay_check(self, as_of: date, days: int | None = None) -> dict[str, Any]:
        replay_days = int(days or self.settings.validation.get("required_stable_replay_days", 3))
        replay_days = max(1, replay_days)

        checks: list[dict[str, Any]] = []
        all_passed = True
        for i in range(replay_days):
            d = as_of - timedelta(days=i)
            try:
                self.run_eod(d)
                health = self.health_check(d, require_review=(i == 0))
                day_ok = bool(health["status"] == "healthy")
                checks.append({"date": d.isoformat(), "ok": day_ok, "health": health})
                if not day_ok:
                    all_passed = False
            except Exception as exc:
                checks.append({"date": d.isoformat(), "ok": False, "error": str(exc)})
                all_passed = False

        return {
            "as_of": as_of.isoformat(),
            "replay_days": replay_days,
            "passed": all_passed,
            "checks": checks,
        }

    def gate_report(
        self,
        as_of: date,
        run_tests: bool = False,
        run_review_if_missing: bool = True,
    ) -> dict[str, Any]:
        d = as_of.isoformat()
        review_delta_path = self.ctx.output_dir / "review" / f"{d}_param_delta.yaml"
        if run_review_if_missing and not review_delta_path.exists():
            self.run_review(as_of)

        quality = self._quality_snapshot(as_of)
        backtest = self._backtest_snapshot(as_of)
        health = self.health_check(as_of, require_review=True)
        replay = self.stable_replay_check(as_of)

        tests_ok = True
        tests_payload: dict[str, Any] = {}
        if run_tests:
            tests_payload = self.test_all()
            tests_ok = bool(tests_payload.get("returncode", 1) == 0)

        review_pass = False
        if review_delta_path.exists():
            try:
                review_delta = yaml.safe_load(review_delta_path.read_text(encoding="utf-8")) or {}
            except Exception:
                review_delta = {}
            review_pass = bool(review_delta.get("pass_gate", False))

        completeness = float(quality.get("completeness", 0.0))
        unresolved = float(quality.get("unresolved_conflict_ratio", 1.0))
        positive_ratio = float(backtest.get("positive_window_ratio", 0.0))
        max_drawdown = float(backtest.get("max_drawdown", 1.0))
        violations = int(backtest.get("violations", 999))

        completeness_ok = completeness >= float(self.settings.validation.get("data_completeness_min", 0.99))
        unresolved_ok = unresolved <= float(self.settings.validation.get("unresolved_conflict_max", 0.005))
        positive_ok = positive_ratio >= float(self.settings.validation.get("positive_window_ratio_min", 0.70))
        drawdown_ok = max_drawdown <= float(self.settings.validation.get("max_drawdown_max", 0.18))
        violations_ok = violations == 0
        health_ok = bool(health.get("status") == "healthy")
        replay_ok = bool(replay.get("passed", False))

        checks = {
            "review_pass_gate": review_pass,
            "tests_ok": tests_ok,
            "health_ok": health_ok,
            "stable_replay_ok": replay_ok,
            "data_completeness_ok": completeness_ok,
            "unresolved_conflict_ok": unresolved_ok,
            "positive_window_ratio_ok": positive_ok,
            "max_drawdown_ok": drawdown_ok,
            "risk_violations_ok": violations_ok,
        }
        overall = all(checks.values())
        out = {
            "date": d,
            "passed": overall,
            "checks": checks,
            "metrics": {
                "completeness": completeness,
                "unresolved_conflict_ratio": unresolved,
                "positive_window_ratio": positive_ratio,
                "max_drawdown": max_drawdown,
                "violations": violations,
            },
            "health": health,
            "stable_replay": replay,
            "tests": tests_payload if run_tests else {"skipped": True},
        }

        if overall:
            alert_path = self.ctx.output_dir / "logs" / f"review_loop_alert_{d}.json"
            if alert_path.exists():
                try:
                    alert_path.unlink()
                except OSError:
                    pass

        report_path = self.ctx.output_dir / "review" / f"{d}_gate_report.json"
        write_json(report_path, out)
        return out

    def _latest_test_result(self) -> dict[str, Any]:
        logs_dir = self.ctx.output_dir / "logs"
        candidates = sorted(logs_dir.glob("tests_*.json"))
        if not candidates:
            return {"found": False}
        latest = candidates[-1]
        payload = self._load_json_safely(latest)
        return {
            "found": True,
            "path": str(latest),
            "returncode": payload.get("returncode"),
            "has_output": bool(payload.get("stdout") or payload.get("stderr")),
        }

    def ops_report(self, as_of: date, window_days: int = 7) -> dict[str, Any]:
        d = as_of.isoformat()
        wd = max(1, int(window_days))

        scheduler_state = self._load_json_safely(self.ctx.output_dir / "logs" / "scheduler_state.json")
        latest_tests = self._latest_test_result()
        gate = self.gate_report(as_of=as_of, run_tests=False, run_review_if_missing=False)

        history = []
        healthy_days = 0
        for i in range(wd):
            day = as_of - timedelta(days=i)
            require_review = (i == 0)
            h = self.health_check(day, require_review=require_review)
            ok = h["status"] == "healthy"
            healthy_days += 1 if ok else 0
            history.append(
                {
                    "date": day.isoformat(),
                    "healthy": ok,
                    "missing": h.get("missing", []),
                }
            )

        history.reverse()
        health_ratio = healthy_days / wd
        status = "green"
        if not gate["passed"] or health_ratio < 0.8:
            status = "red"
        elif health_ratio < 1.0:
            status = "yellow"

        summary = {
            "date": d,
            "status": status,
            "window_days": wd,
            "health_ratio": health_ratio,
            "gate_passed": gate["passed"],
            "latest_tests": latest_tests,
            "scheduler": {
                "date": scheduler_state.get("date"),
                "executed_slots": scheduler_state.get("executed", []),
                "history_count": len(scheduler_state.get("history", [])),
            },
            "history": history,
        }

        report_json = self.ctx.output_dir / "review" / f"{d}_ops_report.json"
        write_json(report_json, summary)

        lines: list[str] = []
        lines.append(f"# 运维状态报告 | {d}")
        lines.append("")
        lines.append(f"- 状态: `{status}`")
        lines.append(f"- 发布门槛通过: `{gate['passed']}`")
        lines.append(f"- 健康天数占比({wd}日): `{health_ratio:.2%}`")
        lines.append(f"- 最近测试: `{latest_tests.get('returncode', 'N/A')}`")
        lines.append(f"- 调度已执行槽位: `{', '.join(summary['scheduler']['executed_slots']) if summary['scheduler']['executed_slots'] else 'NONE'}`")
        lines.append("")
        lines.append("## 最近健康历史")
        for item in history:
            lines.append(f"- {item['date']}: {'OK' if item['healthy'] else 'DEGRADED'} | missing={item['missing']}")
        lines.append("")
        lines.append("## 门槛检查")
        for k, v in gate["checks"].items():
            lines.append(f"- `{k}`: `{v}`")

        report_md = self.ctx.output_dir / "review" / f"{d}_ops_report.md"
        write_markdown(report_md, "\n".join(lines) + "\n")
        summary["paths"] = {"json": str(report_json), "md": str(report_md)}
        return summary

    @staticmethod
    def _extract_failed_tests(test_payload: dict[str, Any]) -> list[str]:
        stderr = str(test_payload.get("stderr", "") or "")
        failed: list[str] = []
        for line in stderr.splitlines():
            txt = line.strip()
            if txt.endswith("... FAIL") or txt.endswith("... ERROR"):
                failed.append(txt.split(" ... ")[0].strip())
            elif txt.startswith("FAIL: ") or txt.startswith("ERROR: "):
                failed.append(txt.split(": ", 1)[1].strip())
        # keep stable order while deduplicating
        out: list[str] = []
        seen: set[str] = set()
        for item in failed:
            if item and item not in seen:
                seen.add(item)
                out.append(item)
        return out

    def _build_defect_plan(
        self,
        as_of: date,
        round_no: int,
        review: ReviewDelta,
        tests: dict[str, Any],
        gate: dict[str, Any],
    ) -> dict[str, Any]:
        checks = gate.get("checks", {})
        metrics = gate.get("metrics", {})
        failed_tests = self._extract_failed_tests(tests)

        defects: list[dict[str, Any]] = []
        if not bool(checks.get("data_completeness_ok", True)):
            defects.append(
                {
                    "category": "data",
                    "code": "DATA_COMPLETENESS",
                    "message": "关键字段完整率未达门槛",
                    "action": "优先修复抓取字段映射并重跑 run-premarket/run-eod。",
                }
            )
        if not bool(checks.get("unresolved_conflict_ok", True)):
            defects.append(
                {
                    "category": "data",
                    "code": "DATA_CONFLICT",
                    "message": "跨源冲突未决比例超限",
                    "action": "定位冲突字段来源并补充冲突仲裁规则后重跑。",
                }
            )
        if not bool(review.pass_gate):
            defects.append(
                {
                    "category": "model",
                    "code": "REVIEW_GATE",
                    "message": "复盘门槛未通过",
                    "action": "检查参数更新幅度和因子贡献，必要时回滚到上一个稳定参数。",
                }
            )
        if not bool(checks.get("positive_window_ratio_ok", True)):
            defects.append(
                {
                    "category": "model",
                    "code": "POSITIVE_WINDOW_RATIO",
                    "message": "样本外正收益窗口占比不达标",
                    "action": "收缩信号阈值与仓位，先做局部回测再跑全量 walk-forward。",
                }
            )
        if not bool(checks.get("max_drawdown_ok", True)):
            defects.append(
                {
                    "category": "risk",
                    "code": "MAX_DRAWDOWN",
                    "message": "最大回撤超过 18% 上限",
                    "action": "下调风险预算并加严止损/保护模式触发阈值。",
                }
            )
        if not bool(checks.get("risk_violations_ok", True)):
            defects.append(
                {
                    "category": "risk",
                    "code": "RISK_VIOLATION",
                    "message": "风险约束出现违规",
                    "action": "校验单笔/总暴露约束计算与执行器拦截逻辑。",
                }
            )
        if int(tests.get("returncode", 0)) != 0:
            defects.append(
                {
                    "category": "execution",
                    "code": "TEST_FAILURE",
                    "message": f"自动测试失败，失败用例数={len(failed_tests)}",
                    "action": "先跑失败用例局部回归，再跑 lie test-all 进行全量验收。",
                    "failed_tests": failed_tests[:20],
                }
            )
        if not bool(checks.get("stable_replay_ok", True)):
            defects.append(
                {
                    "category": "execution",
                    "code": "STABLE_REPLAY",
                    "message": "连续回放稳定性未达标",
                    "action": "定位失败日期的产物缺失或波动异常，修复后重跑 stable-replay。",
                }
            )
        if not bool(checks.get("health_ok", True)):
            defects.append(
                {
                    "category": "report",
                    "code": "HEALTH_DEGRADED",
                    "message": "输出工件不完整",
                    "action": "补跑缺失槽位并校验 output/daily 与 output/review 文件契约。",
                }
            )

        if not defects:
            defects.append(
                {
                    "category": "unknown",
                    "code": "UNCLASSIFIED",
                    "message": "存在失败但未命中分类规则",
                    "action": "人工审查 gate_report 与 tests 日志后补充规则。",
                }
            )

        plan = {
            "date": as_of.isoformat(),
            "round": round_no,
            "defect_count": len(defects),
            "defects": defects,
            "metrics": metrics,
            "checks": checks,
            "next_actions": [
                "仅针对缺陷模块执行局部测试/回放",
                "局部通过后执行 lie test-all",
                "门槛通过后重跑 gate-report 与 ops-report",
            ],
        }

        review_dir = self.ctx.output_dir / "review"
        json_path = review_dir / f"{as_of.isoformat()}_defect_plan_round{round_no}.json"
        md_path = review_dir / f"{as_of.isoformat()}_defect_plan_round{round_no}.md"
        write_json(json_path, plan)

        lines: list[str] = []
        lines.append(f"# 缺陷修正计划 | {as_of.isoformat()} | Round {round_no}")
        lines.append("")
        lines.append(f"- 缺陷数量: `{len(defects)}`")
        lines.append(f"- 最大回撤: `{metrics.get('max_drawdown', 'N/A')}`")
        lines.append(f"- 正收益窗口占比: `{metrics.get('positive_window_ratio', 'N/A')}`")
        lines.append("")
        lines.append("## 缺陷分类")
        for item in defects:
            lines.append(f"- [{item['category']}] `{item['code']}`: {item['message']} | action={item['action']}")
            if item.get("failed_tests"):
                lines.append(f"- 失败用例: {', '.join(item['failed_tests'])}")
        lines.append("")
        lines.append("## 修正顺序")
        for idx, action in enumerate(plan["next_actions"], start=1):
            lines.append(f"{idx}. {action}")

        write_markdown(md_path, "\n".join(lines) + "\n")
        return {"json": str(json_path), "md": str(md_path)}

    def review_until_pass(self, as_of: date, max_rounds: int = 3) -> dict[str, Any]:
        alert_path = self.ctx.output_dir / "logs" / f"review_loop_alert_{as_of.isoformat()}.json"
        if int(max_rounds) <= 0:
            return {
                "passed": False,
                "skipped": True,
                "reason": "max_rounds must be >= 1",
                "rounds": [],
            }

        rounds = []
        for i in range(int(max_rounds)):
            review = self.run_review(as_of)
            tests = self.test_all()
            gate = self.gate_report(as_of=as_of, run_tests=False, run_review_if_missing=False)
            ok = bool(gate["passed"] and tests["returncode"] == 0 and review.pass_gate)
            rounds.append(
                {
                    "round": i + 1,
                    "pass_gate": review.pass_gate,
                    "tests_ok": tests["returncode"] == 0,
                    "stable_replay_ok": gate["checks"]["stable_replay_ok"],
                    "stable_replay_days": gate["stable_replay"]["replay_days"],
                    "gate_passed": gate["passed"],
                }
            )
            if ok:
                release_path = self.ctx.output_dir / "artifacts" / f"release_ready_{as_of.isoformat()}.json"
                write_json(release_path, {"date": as_of.isoformat(), "passed": True, "rounds": rounds})
                if alert_path.exists():
                    try:
                        alert_path.unlink()
                    except OSError:
                        pass
                return {"passed": True, "skipped": False, "rounds": rounds}
            plan_paths = self._build_defect_plan(
                as_of=as_of,
                round_no=i + 1,
                review=review,
                tests=tests,
                gate=gate,
            )
            rounds[-1]["defect_plan"] = plan_paths
        fail_payload = {"passed": False, "skipped": False, "rounds": rounds}
        write_json(alert_path, fail_payload)
        return fail_payload

    def run_review_cycle(self, as_of: date, max_rounds: int = 2, ops_window_days: int | None = None) -> dict[str, Any]:
        replay_days = int(self.settings.validation.get("required_stable_replay_days", 3))
        ops_days = int(ops_window_days or replay_days)
        review_loop = self.review_until_pass(as_of=as_of, max_rounds=max_rounds)
        gate = self.gate_report(as_of=as_of, run_tests=False, run_review_if_missing=False)
        ops = self.ops_report(as_of=as_of, window_days=ops_days)
        health = self.health_check(as_of=as_of, require_review=True)
        return {
            "date": as_of.isoformat(),
            "review_loop": review_loop,
            "gate_report": gate,
            "ops_report": ops,
            "health": health,
        }

    def run_slot(self, as_of: date, slot: str, max_review_rounds: int = 2) -> dict[str, Any]:
        cfg = self._schedule_cfg()
        slot = str(slot)

        if slot in {"premarket", cfg["premarket"]}:
            return {"slot": "premarket", "result": self.run_premarket(as_of)}

        if slot in {"eod", cfg["eod"]}:
            return {"slot": "eod", "result": self.run_eod(as_of)}

        if slot in {"review", cfg["nightly_review"]}:
            review_cycle = self.run_review_cycle(as_of=as_of, max_rounds=max_review_rounds)
            return {"slot": "review", "result": review_cycle}

        if slot in {"ops", "ops-report"}:
            replay_days = int(self.settings.validation.get("required_stable_replay_days", 3))
            ops = self.ops_report(as_of=as_of, window_days=replay_days)
            return {"slot": "ops", "result": ops}

        intraday_slots = cfg["intraday_slots"]
        if slot in intraday_slots:
            return {"slot": f"intraday:{slot}", "result": self.run_intraday_check(as_of, slot=slot)}

        if slot in {"intraday_1", "intraday-1"} and intraday_slots:
            s = intraday_slots[0]
            return {"slot": f"intraday:{s}", "result": self.run_intraday_check(as_of, slot=s)}

        if slot in {"intraday_2", "intraday-2"} and len(intraday_slots) >= 2:
            s = intraday_slots[1]
            return {"slot": f"intraday:{s}", "result": self.run_intraday_check(as_of, slot=s)}

        raise ValueError(f"Unsupported slot: {slot}")

    def run_session(self, as_of: date, include_review: bool = True, max_review_rounds: int = 2) -> dict[str, Any]:
        cfg = self._schedule_cfg()
        out: dict[str, Any] = {"date": as_of.isoformat(), "schedule": cfg, "steps": {}}

        out["steps"]["premarket"] = self.run_premarket(as_of)
        for slot in cfg["intraday_slots"]:
            out["steps"][f"intraday_{slot}"] = self.run_intraday_check(as_of, slot=slot)
        out["steps"]["eod"] = self.run_eod(as_of)

        if include_review:
            out["steps"]["review_cycle"] = self.run_review_cycle(as_of=as_of, max_rounds=max_review_rounds)
        else:
            out["steps"]["review_cycle"] = {"skipped": True}
        return out

    def run_daemon(self, poll_seconds: int = 30, max_cycles: int | None = None, max_review_rounds: int = 2) -> dict[str, Any]:
        cfg = self._schedule_cfg()
        slots = [cfg["premarket"], *cfg["intraday_slots"], cfg["eod"], cfg["nightly_review"]]
        tz = ZoneInfo(self.settings.timezone)
        state_path = self.ctx.output_dir / "logs" / "scheduler_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)

        def load_state() -> dict[str, Any]:
            if not state_path.exists():
                return {"date": None, "executed": [], "history": []}
            try:
                return json.loads(state_path.read_text(encoding="utf-8"))
            except Exception:
                return {"date": None, "executed": [], "history": []}

        def save_state(st: dict[str, Any]) -> None:
            state_path.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")

        state = load_state()
        cycles = 0

        while True:
            now = datetime.now(tz)
            d = now.date().isoformat()
            if state.get("date") != d:
                state = {"date": d, "executed": [], "history": []}

            for slot in slots:
                if slot in state["executed"]:
                    continue
                if now.time() >= self._parse_hhmm(slot):
                    try:
                        result = self.run_slot(as_of=now.date(), slot=slot, max_review_rounds=max_review_rounds)
                        status = "ok"
                    except Exception as exc:
                        result = {"error": str(exc)}
                        status = "error"
                    state["executed"].append(slot)
                    state["history"].append(
                        {
                            "ts": now.isoformat(),
                            "slot": slot,
                            "status": status,
                            "result": result,
                        }
                    )
                    save_state(state)

            if max_cycles is not None and cycles >= max_cycles:
                break

            cycles += 1
            time_module.sleep(max(1, int(poll_seconds)))

        return state

    def health_check(self, as_of: date | None = None, require_review: bool = True) -> dict[str, Any]:
        tz = ZoneInfo(self.settings.timezone)
        target = as_of or datetime.now(tz).date()
        d = target.isoformat()

        daily_dir = self.ctx.output_dir / "daily"
        review_dir = self.ctx.output_dir / "review"
        required = {
            "daily_briefing": daily_dir / f"{d}_briefing.md",
            "daily_signals": daily_dir / f"{d}_signals.json",
            "daily_positions": daily_dir / f"{d}_positions.csv",
            "sqlite": self.ctx.sqlite_path,
        }
        if require_review:
            required["review_report"] = review_dir / f"{d}_review.md"
            required["review_delta"] = review_dir / f"{d}_param_delta.yaml"

        checks = {k: p.exists() for k, p in required.items()}
        missing = [k for k, v in checks.items() if not v]
        status = "healthy" if not missing else "degraded"

        return {
            "date": d,
            "status": status,
            "checks": checks,
            "missing": missing,
        }
