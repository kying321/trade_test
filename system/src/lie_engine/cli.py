from __future__ import annotations

import argparse
from datetime import date, datetime
import json

from lie_engine.config import load_settings, validate_settings
from lie_engine.engine import LieEngine
from lie_engine.models import BacktestResult, ReviewDelta


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main() -> None:
    parser = argparse.ArgumentParser(prog="lie", description="Liè Antifragile Trading System CLI")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_eod = sub.add_parser("run-eod", help="Run end-of-day pipeline")
    p_eod.add_argument("--date", required=True)

    p_pm = sub.add_parser("run-premarket", help="Run premarket checks")
    p_pm.add_argument("--date", required=True)

    p_ic = sub.add_parser("run-intraday-check", help="Run intraday checkpoint")
    p_ic.add_argument("--date", required=True)
    p_ic.add_argument("--slot", required=True, help="HH:MM")

    p_bt = sub.add_parser("backtest", help="Run walk-forward backtest")
    p_bt.add_argument("--start", required=True)
    p_bt.add_argument("--end", required=True)

    p_rbt = sub.add_parser("research-backtest", help="Run budgeted real-data multi-mode research backtest")
    p_rbt.add_argument("--start", required=True)
    p_rbt.add_argument("--end", required=True)
    p_rbt.add_argument("--hours", default="10")
    p_rbt.add_argument("--max-symbols", default="120")
    p_rbt.add_argument("--report-symbol-cap", default="40")
    p_rbt.add_argument("--workers", default="8")
    p_rbt.add_argument("--max-trials-per-mode", default="500")
    p_rbt.add_argument("--seed", default="42")
    p_rbt.add_argument("--modes", default="ultra_short,swing,long", help="Comma-separated modes")
    p_rbt.add_argument("--review-days", default="5", help="Post-cutoff review window days (not used for backtest)")

    p_sl = sub.add_parser("strategy-lab", help="Learn and validate new strategy candidates from market+reports")
    p_sl.add_argument("--start", required=True)
    p_sl.add_argument("--end", required=True)
    p_sl.add_argument("--max-symbols", default="120")
    p_sl.add_argument("--report-symbol-cap", default="40")
    p_sl.add_argument("--workers", default="8")
    p_sl.add_argument("--review-days", default="5")
    p_sl.add_argument("--candidate-count", default="10")

    p_rv = sub.add_parser("review", help="Run post-market review and parameter update")
    p_rv.add_argument("--date", required=True)

    p_rc = sub.add_parser("run-review-cycle", help="Run review-loop + gate-report + ops-report")
    p_rc.add_argument("--date", required=True)
    p_rc.add_argument("--max-rounds", default="2")
    p_rc.add_argument("--ops-window-days", default=None)

    p_rs = sub.add_parser("run-slot", help="Run one scheduled slot")
    p_rs.add_argument("--date", required=True)
    p_rs.add_argument("--slot", required=True, help="premarket|10:30|14:30|eod|review|ops")
    p_rs.add_argument("--max-review-rounds", default="2")

    p_ss = sub.add_parser("run-session", help="Run full scheduled session for one date")
    p_ss.add_argument("--date", required=True)
    p_ss.add_argument("--skip-review", action="store_true")
    p_ss.add_argument("--max-review-rounds", default="2")

    p_dm = sub.add_parser("run-daemon", help="Run polling scheduler daemon")
    p_dm.add_argument("--poll-seconds", default="30")
    p_dm.add_argument("--max-cycles", default=None)
    p_dm.add_argument("--max-review-rounds", default="2")
    p_dm.add_argument("--dry-run", action="store_true", help="Preview due slots without execution/state mutation")

    p_hc = sub.add_parser("health-check", help="Check daily output and artifacts health")
    p_hc.add_argument("--date", default=None)

    p_sr = sub.add_parser("stable-replay", help="Check required stable replay days")
    p_sr.add_argument("--date", required=True)
    p_sr.add_argument("--days", default=None)

    p_gr = sub.add_parser("gate-report", help="Run release gate report")
    p_gr.add_argument("--date", required=True)
    p_gr.add_argument("--run-tests", action="store_true")
    p_gr.add_argument("--run-review-if-missing", action="store_true")

    p_or = sub.add_parser("ops-report", help="Run operations health summary report")
    p_or.add_argument("--date", required=True)
    p_or.add_argument("--window-days", default="7")

    sub.add_parser("validate-config", help="Validate config schema and risk bounds")

    p_aa = sub.add_parser("architecture-audit", help="Run architecture audit report")
    p_aa.add_argument("--date", required=False, default=None)

    p_da = sub.add_parser("dependency-audit", help="Run dependency layer audit report")
    p_da.add_argument("--date", required=False, default=None)

    p_ta = sub.add_parser("test-all", help="Run test suite")
    p_ta.add_argument("--fast", action="store_true", help="Run deterministic subset for quick feedback")
    p_ta.add_argument("--fast-ratio", default="0.10", help="Subset ratio in (0,1], e.g. 0.05")
    p_ta.add_argument("--fast-shard-index", default="0", help="Shard index for parallel agents")
    p_ta.add_argument("--fast-shard-total", default="1", help="Shard total for parallel agents")
    p_ta.add_argument("--fast-seed", default="lie-fast-v1", help="Deterministic sampling seed")

    p_loop = sub.add_parser("review-loop", help="Run review->tests loop until pass or max rounds")
    p_loop.add_argument("--date", required=True)
    p_loop.add_argument("--max-rounds", default="3")

    args = parser.parse_args()
    if args.cmd == "validate-config":
        settings = load_settings(args.config)
        out = validate_settings(settings)
    else:
        eng = LieEngine(config_path=args.config)

    if args.cmd == "validate-config":
        pass
    elif args.cmd == "run-eod":
        out = eng.run_eod(as_of=_parse_date(args.date))
    elif args.cmd == "run-premarket":
        out = eng.run_premarket(as_of=_parse_date(args.date))
    elif args.cmd == "run-intraday-check":
        out = eng.run_intraday_check(as_of=_parse_date(args.date), slot=args.slot)
    elif args.cmd == "backtest":
        result: BacktestResult = eng.run_backtest(start=_parse_date(args.start), end=_parse_date(args.end))
        out = result.to_dict()
    elif args.cmd == "research-backtest":
        out = eng.run_research_backtest(
            start=_parse_date(args.start),
            end=_parse_date(args.end),
            hours_budget=float(args.hours),
            max_symbols=int(args.max_symbols),
            report_symbol_cap=int(args.report_symbol_cap),
            workers=int(args.workers),
            max_trials_per_mode=int(args.max_trials_per_mode),
            seed=int(args.seed),
            modes=[x.strip() for x in str(args.modes).split(",") if x.strip()],
            review_days=int(args.review_days),
        )
    elif args.cmd == "strategy-lab":
        out = eng.run_strategy_lab(
            start=_parse_date(args.start),
            end=_parse_date(args.end),
            max_symbols=int(args.max_symbols),
            report_symbol_cap=int(args.report_symbol_cap),
            workers=int(args.workers),
            review_days=int(args.review_days),
            candidate_count=int(args.candidate_count),
        )
    elif args.cmd == "review":
        result2: ReviewDelta = eng.run_review(as_of=_parse_date(args.date))
        out = result2.to_dict()
    elif args.cmd == "run-review-cycle":
        out = eng.run_review_cycle(
            as_of=_parse_date(args.date),
            max_rounds=int(args.max_rounds),
            ops_window_days=int(args.ops_window_days) if args.ops_window_days not in {None, "", "none", "None"} else None,
        )
    elif args.cmd == "run-slot":
        out = eng.run_slot(
            as_of=_parse_date(args.date),
            slot=args.slot,
            max_review_rounds=int(args.max_review_rounds),
        )
    elif args.cmd == "run-session":
        out = eng.run_session(
            as_of=_parse_date(args.date),
            include_review=not bool(args.skip_review),
            max_review_rounds=int(args.max_review_rounds),
        )
    elif args.cmd == "run-daemon":
        max_cycles = None if args.max_cycles in {None, "", "none", "None"} else int(args.max_cycles)
        out = eng.run_daemon(
            poll_seconds=int(args.poll_seconds),
            max_cycles=max_cycles,
            max_review_rounds=int(args.max_review_rounds),
            dry_run=bool(args.dry_run),
        )
    elif args.cmd == "health-check":
        out = eng.health_check(as_of=_parse_date(args.date) if args.date else None)
    elif args.cmd == "stable-replay":
        out = eng.stable_replay_check(
            as_of=_parse_date(args.date),
            days=int(args.days) if args.days not in {None, "", "none", "None"} else None,
        )
    elif args.cmd == "gate-report":
        out = eng.gate_report(
            as_of=_parse_date(args.date),
            run_tests=bool(args.run_tests),
            run_review_if_missing=bool(args.run_review_if_missing),
        )
    elif args.cmd == "ops-report":
        out = eng.ops_report(
            as_of=_parse_date(args.date),
            window_days=int(args.window_days),
        )
    elif args.cmd == "architecture-audit":
        out = eng.architecture_audit(as_of=_parse_date(args.date) if args.date else None)
    elif args.cmd == "dependency-audit":
        out = eng.dependency_audit(as_of=_parse_date(args.date) if args.date else None)
    elif args.cmd == "test-all":
        out = eng.test_all(
            fast=bool(args.fast),
            fast_ratio=float(args.fast_ratio),
            fast_shard_index=int(args.fast_shard_index),
            fast_shard_total=int(args.fast_shard_total),
            fast_seed=str(args.fast_seed),
        )
    elif args.cmd == "review-loop":
        out = eng.review_until_pass(as_of=_parse_date(args.date), max_rounds=int(args.max_rounds))
    else:
        raise ValueError(f"Unknown command: {args.cmd}")

    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
