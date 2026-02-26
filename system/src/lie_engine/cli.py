from __future__ import annotations

import argparse
from datetime import date, datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

from lie_engine.config import load_settings, validate_settings
from lie_engine.engine import LieEngine
from lie_engine.models import BacktestResult, ReviewDelta


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main() -> None:
    argv0 = Path(sys.argv[0]).name.strip().lower() if sys.argv else ""
    prog = argv0 if argv0 in {"fenlie", "lie"} else "fenlie"
    parser = argparse.ArgumentParser(prog=prog, description="Fenlie Antifragile Trading System CLI")
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

    p_hp = sub.add_parser("run-halfhour-pulse", help="Run one 30-minute pulse scheduler cycle")
    p_hp.add_argument("--date", required=False, default=None, help="YYYY-MM-DD (default: local today)")
    p_hp.add_argument("--slot", required=False, default=None, help="HH:MM (default: local current time)")
    p_hp.add_argument("--max-review-rounds", default="2")
    p_hp.add_argument("--max-slot-runs", default="2")
    p_hp.add_argument("--slot-retry-max", default="2")
    p_hp.add_argument("--ops-every-n-pulses", default="4")
    p_hp.add_argument("--force", action="store_true")
    p_hp.add_argument("--dry-run", action="store_true")

    p_ss = sub.add_parser("run-session", help="Run full scheduled session for one date")
    p_ss.add_argument("--date", required=True)
    p_ss.add_argument("--skip-review", action="store_true")
    p_ss.add_argument("--max-review-rounds", default="2")

    p_dm = sub.add_parser("run-daemon", help="Run polling scheduler daemon")
    p_dm.add_argument("--poll-seconds", default="30")
    p_dm.add_argument("--max-cycles", default=None)
    p_dm.add_argument("--max-review-rounds", default="2")
    p_dm.add_argument("--dry-run", action="store_true", help="Preview due slots without execution/state mutation")

    p_hd = sub.add_parser("run-halfhour-daemon", help="Run polling daemon for 30-minute pulse scheduler")
    p_hd.add_argument("--poll-seconds", default="30")
    p_hd.add_argument("--max-cycles", default=None)
    p_hd.add_argument("--max-review-rounds", default="2")
    p_hd.add_argument("--max-slot-runs", default="2")
    p_hd.add_argument("--slot-retry-max", default="2")
    p_hd.add_argument("--ops-every-n-pulses", default="4")
    p_hd.add_argument("--dry-run", action="store_true", help="Preview current bucket behavior without state mutation")

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

    p_ar = sub.add_parser("autorun-retro", help="Run automation execution retrospective report")
    p_ar.add_argument("--date", required=False, default=None)
    p_ar.add_argument("--window-days", default="7")

    p_gb = sub.add_parser("guardrail-burnin", help="Run degradation-guardrail burn-in replay and threshold tuning")
    p_gb.add_argument("--date", required=True)
    p_gb.add_argument("--days", default="3")
    p_gb.add_argument("--no-stable-replay", action="store_true")
    p_gb.add_argument("--no-auto-tune", action="store_true")

    p_gd = sub.add_parser("guardrail-drift-audit", help="Run degradation-guardrail threshold drift audit")
    p_gd.add_argument("--date", required=True)
    p_gd.add_argument("--window-days", default="56")

    p_ce = sub.add_parser(
        "compact-executed-plans",
        help="Compact duplicated executed_plans rows in bounded date chunks",
    )
    p_ce.add_argument("--start", required=True, help="YYYY-MM-DD")
    p_ce.add_argument("--end", required=True, help="YYYY-MM-DD")
    p_ce.add_argument("--chunk-days", default="30")
    p_ce.add_argument("--max-delete-rows", default=None)
    p_ce.add_argument("--apply", action="store_true", help="Apply deletion (default is dry-run)")

    p_vcr = sub.add_parser(
        "verify-compaction-restore",
        help="Verify executed_plans compaction rollback chain on latest or specified run_id",
    )
    p_vcr.add_argument("--run-id", default=None)
    p_vcr.add_argument("--keep-temp-db", action="store_true")

    p_dbm = sub.add_parser("db-maintain", help="Run SQLite stats/retention/vacuum maintenance")
    p_dbm.add_argument("--date", default=None, help="YYYY-MM-DD (default: today in config timezone)")
    p_dbm.add_argument("--retention-days", default=None, help="Delete rows older than N days")
    p_dbm.add_argument("--tables", default=None, help="Comma-separated table list")
    p_dbm.add_argument("--vacuum", action="store_true", help="Run VACUUM (apply mode only)")
    p_dbm.add_argument("--analyze", action="store_true", help="Run ANALYZE (apply mode only)")
    p_dbm.add_argument("--apply", action="store_true", help="Apply retention/vacuum (default dry-run)")

    p_sm = sub.add_parser("stress-matrix", help="Run mode-aware stress matrix report")
    p_sm.add_argument("--date", required=True)
    p_sm.add_argument("--modes", default="ultra_short,swing,long", help="Comma-separated runtime modes")

    p_seam = sub.add_parser(
        "stress-exec-approval-manifest",
        help="Create/validate approval manifest for stress execution-friction controlled apply",
    )
    p_seam.add_argument("--date", default=None, help="YYYY-MM-DD (default: today in config timezone)")
    p_seam.add_argument("--proposal-id", default=None, help="Override proposal_id (auto-assist from proposal artifact when omitted)")
    p_seam.add_argument("--proposal-path", default=None, help="Optional proposal json path override")
    p_seam.add_argument("--manifest-path", default=None, help="Optional approval manifest path override")
    p_seam.add_argument("--approved-at", default=None, help="ISO timestamp; default now in config timezone")
    p_seam.add_argument("--reject", action="store_true", help="Write approval with approved=false (lint only by default)")
    p_seam.add_argument("--validate-only", action="store_true", help="Run lint and preview without writing")

    p_fham = sub.add_parser(
        "frontend-hard-fail-approval-manifest",
        help="Create/validate approval manifest for frontend snapshot trend hard-fail controlled apply",
    )
    p_fham.add_argument("--date", default=None, help="YYYY-MM-DD (default: today in config timezone)")
    p_fham.add_argument(
        "--proposal-id",
        default=None,
        help="Override proposal_id (auto-assist from proposal artifact when omitted)",
    )
    p_fham.add_argument("--proposal-path", default=None, help="Optional proposal json path override")
    p_fham.add_argument("--manifest-path", default=None, help="Optional approval manifest path override")
    p_fham.add_argument("--approved-at", default=None, help="ISO timestamp; default now in config timezone")
    p_fham.add_argument("--reject", action="store_true", help="Write approval with approved=false (lint only by default)")
    p_fham.add_argument("--validate-only", action="store_true", help="Run lint and preview without writing")

    sub.add_parser("validate-config", help="Validate config schema and risk bounds")

    p_aa = sub.add_parser("architecture-audit", help="Run architecture audit report")
    p_aa.add_argument("--date", required=False, default=None)

    p_da = sub.add_parser("dependency-audit", help="Run dependency layer audit report")
    p_da.add_argument("--date", required=False, default=None)

    p_ta = sub.add_parser("test-all", help="Run test suite")
    p_ta.add_argument("--tier", default="deep", choices=["fast", "standard", "deep"], help="Test tier profile")
    p_ta.add_argument("--fast", action="store_true", help="Run deterministic subset for quick feedback")
    p_ta.add_argument("--standard-ratio", default="0.35", help="Subset ratio for standard tier in (0,1]")
    p_ta.add_argument("--fast-ratio", default="0.10", help="Subset ratio in (0,1], e.g. 0.05")
    p_ta.add_argument("--fast-shard-index", default="0", help="Shard index for parallel agents")
    p_ta.add_argument("--fast-shard-total", default="1", help="Shard total for parallel agents")
    p_ta.add_argument("--fast-seed", default="lie-fast-v1", help="Deterministic sampling seed")
    p_ta.add_argument(
        "--fast-tail-priority",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force tail-risk tests into fast/standard subsets",
    )
    p_ta.add_argument(
        "--fast-tail-floor",
        default="3",
        help="Minimum number of tail-priority tests forced into fast subset",
    )
    p_ta.add_argument(
        "--isolate-shard-workspace",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run tests in an ephemeral shard workspace (default: auto on shard_total>1)",
    )
    p_ta.add_argument(
        "--shard-workspace-root",
        default=None,
        help="Optional root directory for shard ephemeral workspaces",
    )

    p_tc = sub.add_parser("test-chaos", help="Run chaos-tier resilience tests")
    p_tc.add_argument("--max-tests", default="24", help="Cap selected chaos tests")
    p_tc.add_argument("--fast-shard-index", default="0", help="Shard index for chaos parallel agents")
    p_tc.add_argument("--fast-shard-total", default="1", help="Shard total for chaos parallel agents")
    p_tc.add_argument("--seed", default="lie-chaos-v1", help="Deterministic chaos selection seed")
    p_tc.add_argument(
        "--isolate-shard-workspace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run chaos tests in an ephemeral shard workspace",
    )
    p_tc.add_argument(
        "--shard-workspace-root",
        default=None,
        help="Optional root directory for chaos shard ephemeral workspaces",
    )
    p_tc.add_argument(
        "--include-probes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run chaos fault-injection probes (config/sqlite/json/kill-recovery)",
    )
    p_tc.add_argument(
        "--probe-timeout-seconds",
        default="30",
        help="Per-probe timeout seconds",
    )

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
    elif args.cmd == "run-halfhour-pulse":
        pulse_date = _parse_date(args.date) if args.date else datetime.now(ZoneInfo(eng.settings.timezone)).date()
        out = eng.run_halfhour_pulse(
            as_of=pulse_date,
            slot=args.slot,
            max_review_rounds=int(args.max_review_rounds),
            max_slot_runs=int(args.max_slot_runs),
            slot_retry_max=int(args.slot_retry_max),
            ops_every_n_pulses=int(args.ops_every_n_pulses),
            force=bool(args.force),
            dry_run=bool(args.dry_run),
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
    elif args.cmd == "run-halfhour-daemon":
        max_cycles = None if args.max_cycles in {None, "", "none", "None"} else int(args.max_cycles)
        out = eng.run_halfhour_daemon(
            poll_seconds=int(args.poll_seconds),
            max_cycles=max_cycles,
            max_review_rounds=int(args.max_review_rounds),
            max_slot_runs=int(args.max_slot_runs),
            slot_retry_max=int(args.slot_retry_max),
            ops_every_n_pulses=int(args.ops_every_n_pulses),
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
    elif args.cmd == "autorun-retro":
        retro_date = _parse_date(args.date) if args.date else datetime.now(ZoneInfo(eng.settings.timezone)).date()
        out = eng.run_autorun_retro(
            as_of=retro_date,
            window_days=int(args.window_days),
        )
    elif args.cmd == "guardrail-burnin":
        out = eng.degradation_guardrail_burnin(
            as_of=_parse_date(args.date),
            days=int(args.days),
            run_stable_replay=not bool(args.no_stable_replay),
            auto_tune=not bool(args.no_auto_tune),
        )
    elif args.cmd == "guardrail-drift-audit":
        out = eng.degradation_guardrail_threshold_drift_audit(
            as_of=_parse_date(args.date),
            window_days=int(args.window_days),
        )
    elif args.cmd == "compact-executed-plans":
        max_delete_rows = (
            None
            if args.max_delete_rows in {None, "", "none", "None"}
            else int(args.max_delete_rows)
        )
        out = eng.compact_executed_plans_duplicates(
            start=_parse_date(args.start),
            end=_parse_date(args.end),
            chunk_days=int(args.chunk_days),
            dry_run=not bool(args.apply),
            max_delete_rows=max_delete_rows,
        )
    elif args.cmd == "verify-compaction-restore":
        run_id = None if args.run_id in {None, "", "none", "None"} else str(args.run_id)
        out = eng.verify_executed_plans_compaction_restore(
            run_id=run_id,
            keep_temp_db=bool(args.keep_temp_db),
        )
    elif args.cmd == "db-maintain":
        db_date = _parse_date(args.date) if args.date else datetime.now(ZoneInfo(eng.settings.timezone)).date()
        retention_days = (
            None
            if args.retention_days in {None, "", "none", "None"}
            else int(args.retention_days)
        )
        tables = (
            [x.strip() for x in str(args.tables).split(",") if x.strip()]
            if args.tables not in {None, "", "none", "None"}
            else None
        )
        out = eng.maintain_sqlite(
            as_of=db_date,
            retention_days=retention_days,
            tables=tables,
            vacuum=bool(args.vacuum),
            analyze=bool(args.analyze),
            apply=bool(args.apply),
        )
    elif args.cmd == "stress-matrix":
        out = eng.run_mode_stress_matrix(
            as_of=_parse_date(args.date),
            modes=[x.strip() for x in str(args.modes).split(",") if x.strip()],
        )
    elif args.cmd == "stress-exec-approval-manifest":
        out = eng.stress_exec_controlled_apply_approval_manifest(
            as_of=_parse_date(args.date) if args.date else None,
            proposal_id=args.proposal_id,
            approved=(not bool(args.reject)),
            approved_at=args.approved_at,
            proposal_path=args.proposal_path,
            manifest_path=args.manifest_path,
            validate_only=bool(args.validate_only),
        )
    elif args.cmd == "frontend-hard-fail-approval-manifest":
        out = eng.frontend_hard_fail_controlled_apply_approval_manifest(
            as_of=_parse_date(args.date) if args.date else None,
            proposal_id=args.proposal_id,
            approved=(not bool(args.reject)),
            approved_at=args.approved_at,
            proposal_path=args.proposal_path,
            manifest_path=args.manifest_path,
            validate_only=bool(args.validate_only),
        )
    elif args.cmd == "architecture-audit":
        out = eng.architecture_audit(as_of=_parse_date(args.date) if args.date else None)
    elif args.cmd == "dependency-audit":
        out = eng.dependency_audit(as_of=_parse_date(args.date) if args.date else None)
    elif args.cmd == "test-all":
        tier = "fast" if bool(args.fast) else str(args.tier)
        out = eng.test_all(
            fast=bool(args.fast),
            tier=tier,
            standard_ratio=float(args.standard_ratio),
            fast_ratio=float(args.fast_ratio),
            fast_shard_index=int(args.fast_shard_index),
            fast_shard_total=int(args.fast_shard_total),
            fast_seed=str(args.fast_seed),
            fast_tail_priority=bool(args.fast_tail_priority),
            fast_tail_floor=int(args.fast_tail_floor),
            isolate_shard_workspace=args.isolate_shard_workspace,
            shard_workspace_root=args.shard_workspace_root,
        )
    elif args.cmd == "test-chaos":
        out = eng.test_chaos(
            max_tests=int(args.max_tests),
            fast_shard_index=int(args.fast_shard_index),
            fast_shard_total=int(args.fast_shard_total),
            seed=str(args.seed),
            isolate_shard_workspace=bool(args.isolate_shard_workspace),
            shard_workspace_root=args.shard_workspace_root,
            include_probes=bool(args.include_probes),
            probe_timeout_seconds=int(args.probe_timeout_seconds),
        )
    elif args.cmd == "review-loop":
        out = eng.review_until_pass(as_of=_parse_date(args.date), max_rounds=int(args.max_rounds))
    else:
        raise ValueError(f"Unknown command: {args.cmd}")

    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
