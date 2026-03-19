#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACT_TTL_HOURS = 168


def _load_guard_module():
    script_dir = Path(__file__).resolve().parent
    mod_path = script_dir / "guarded_exec.py"
    spec = importlib.util.spec_from_file_location("guarded_exec", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load guarded_exec module: {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GE = _load_guard_module()
panic_close_all = GE.panic_close_all
PanicTriggered = GE.PanicTriggered
resolve_workspace = GE.resolve_workspace
run_json_command = GE.run_json_command
sha256_file = GE.sha256_file
write_json = GE.write_json


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def now_utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def evict_old_pipeline_artifacts(*, directory: Path, ttl_hours: int, keep_names: set[str]) -> int:
    cutoff = datetime.now(timezone.utc).timestamp() - (max(1, int(ttl_hours)) * 3600.0)
    removed = 0
    for pattern in ("*_ticketed_canary_pipeline.json", "*_ticketed_canary_pipeline_checksum.json"):
        for path in directory.glob(pattern):
            if path.name in keep_names:
                continue
            try:
                if path.stat().st_mtime >= cutoff:
                    continue
            except Exception:
                continue
            try:
                path.unlink()
                removed += 1
            except Exception:
                continue
    return removed


def build_ticket_cmd(*, system_root: Path, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(system_root / "scripts" / "build_order_ticket.py"),
        "--config",
        str(system_root / "config.yaml"),
        "--output-root",
        str(system_root / "output"),
        "--review-dir",
        str(system_root / "output" / "review"),
        "--output-dir",
        str(system_root / "output" / "review"),
        "--symbols",
        str(args.symbols),
        "--max-age-days",
        str(max(1, int(args.live_max_age_days))),
    ]
    if str(args.date).strip():
        cmd.extend(["--date", str(args.date).strip()])
    if float(args.equity_usdt) > 0.0:
        cmd.extend(["--equity-usdt", f"{float(args.equity_usdt):.8f}"])
    if args.live_min_confidence is not None:
        cmd.extend(["--min-confidence", str(float(args.live_min_confidence))])
    if args.live_min_convexity is not None:
        cmd.extend(["--min-convexity", str(float(args.live_min_convexity))])
    return cmd


def build_plan_cmd(*, system_root: Path, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(system_root / "scripts" / "build_fast_trade_skill_plan.py"),
        "--review-dir",
        str(system_root / "output" / "review"),
        "--output-dir",
        str(system_root / "output" / "review"),
        "--config",
        str(system_root / "config.yaml"),
        "--symbols",
        str(args.symbols),
        "--max-age-days",
        str(max(1, int(args.live_max_age_days))),
        "--decision-ttl-seconds",
        str(max(30, int(args.decision_ttl_seconds))),
    ]
    if float(args.equity_usdt) > 0.0:
        cmd.extend(["--equity-usdt", f"{float(args.equity_usdt):.8f}"])
    if args.live_min_confidence is not None:
        cmd.extend(["--min-confidence", str(float(args.live_min_confidence))])
    if args.live_min_convexity is not None:
        cmd.extend(["--min-convexity", str(float(args.live_min_convexity))])
    return cmd


def build_guard_cmd(
    *,
    system_root: Path,
    args: argparse.Namespace,
    mode: str,
    quote_usdt: float,
    order_symbol: str,
    order_side: str,
    order_quantity: float,
    risk_intent: str,
) -> list[str]:
    cmd = [
        sys.executable,
        str(system_root / "scripts" / "guarded_exec.py"),
        "--workspace",
        str(system_root.parent),
        "--mode",
        mode,
        "--market",
        str(args.market),
        "--canary-quote-usdt",
        f"{float(quote_usdt):.4f}",
        "--rate-limit-per-minute",
        str(max(1, int(args.rate_limit_per_minute))),
        "--timeout-ms",
        str(min(5000, max(100, int(args.timeout_ms)))),
        "--max-drawdown",
        f"{float(args.max_drawdown):.6f}",
        "--trade-window-hours",
        str(max(1, int(args.trade_window_hours))),
        "--risk-fuse-max-age-seconds",
        str(max(1, int(args.risk_fuse_max_age_seconds))),
        "--panic-cooldown-seconds",
        str(max(1, int(args.panic_cooldown_seconds))),
        "--max-daily-loss-ratio",
        f"{float(args.max_daily_loss_ratio):.6f}",
        "--max-open-exposure-ratio",
        f"{float(args.max_open_exposure_ratio):.6f}",
        "--risk-intent",
        str(risk_intent),
    ]
    if str(args.date).strip():
        cmd.extend(["--date", str(args.date).strip()])
    if bool(args.allow_daemon_env_fallback):
        cmd.append("--allow-daemon-env-fallback")
    if mode == "canary" and bool(args.run_live_canary):
        cmd.append("--allow-live-order")
    if order_symbol:
        cmd.extend(["--order-symbol", order_symbol])
    if order_side:
        cmd.extend(["--order-side", order_side])
    if order_quantity > 0.0:
        cmd.extend(["--order-quantity", f"{float(order_quantity):.8f}"])
    return cmd


def extract_open_fill(guard_payload: dict[str, Any]) -> dict[str, Any]:
    takeover = guard_payload.get("takeover", {}) if isinstance(guard_payload.get("takeover", {}), dict) else {}
    payload = takeover.get("payload", {}) if isinstance(takeover.get("payload", {}), dict) else {}
    steps = payload.get("steps", {}) if isinstance(payload.get("steps", {}), dict) else {}
    canary = steps.get("canary_order", {}) if isinstance(steps.get("canary_order", {}), dict) else {}
    plan = steps.get("canary_plan", {}) if isinstance(steps.get("canary_plan", {}), dict) else {}
    side = str(canary.get("side", plan.get("side", ""))).strip().upper()
    return {
        "executed": bool(canary.get("executed", False)),
        "symbol": str(canary.get("symbol", plan.get("symbol", ""))).strip().upper(),
        "side": side,
        "quantity": float(canary.get("quantity", plan.get("quantity", 0.0)) or 0.0),
        "reverse_side": "SELL" if side == "BUY" else "BUY",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="One-click ticketed canary pipeline for local guarded execution.")
    parser.add_argument("--workspace", default="")
    parser.add_argument("--date", default="")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XAUUSD")
    parser.add_argument("--market", choices=["spot", "futures_usdm"], default="spot")
    parser.add_argument("--canary-quote-usdt", type=float, default=5.0)
    parser.add_argument("--equity-usdt", type=float, default=0.0)
    parser.add_argument("--use-selected-quote", action="store_true")
    parser.add_argument("--live-min-confidence", type=float, default=None)
    parser.add_argument("--live-min-convexity", type=float, default=None)
    parser.add_argument("--live-max-age-days", type=int, default=14)
    parser.add_argument("--decision-ttl-seconds", type=int, default=300)
    parser.add_argument("--run-live-canary", action="store_true")
    parser.add_argument("--auto-unwind-live-canary", action="store_true")
    parser.add_argument("--rate-limit-per-minute", type=int, default=10)
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument("--max-drawdown", type=float, default=0.05)
    parser.add_argument("--trade-window-hours", type=int, default=24)
    parser.add_argument("--allow-daemon-env-fallback", action="store_true")
    parser.add_argument("--risk-fuse-max-age-seconds", type=int, default=300)
    parser.add_argument("--panic-cooldown-seconds", type=int, default=1800)
    parser.add_argument("--max-daily-loss-ratio", type=float, default=0.05)
    parser.add_argument("--max-open-exposure-ratio", type=float, default=0.50)
    parser.add_argument("--artifact-ttl-hours", type=int, default=DEFAULT_ARTIFACT_TTL_HOURS)
    args = parser.parse_args()

    _, system_root = resolve_workspace(args.workspace)
    output_root = system_root / "output"
    review_dir = output_root / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    timeout_seconds = max(10.0, min(120.0, float(max(100, int(args.timeout_ms))) / 1000.0 * 6.0))

    payload: dict[str, Any]
    try:
        ticket_build = run_json_command(cmd=build_ticket_cmd(system_root=system_root, args=args), cwd=system_root, timeout_seconds=timeout_seconds)
        if int(ticket_build.get("returncode", 0)) != 0:
            raise RuntimeError(f"build_order_ticket failed rc={ticket_build.get('returncode', 0)}")

        plan_build = run_json_command(cmd=build_plan_cmd(system_root=system_root, args=args), cwd=system_root, timeout_seconds=timeout_seconds)
        if int(plan_build.get("returncode", 0)) != 0:
            raise RuntimeError(f"build_fast_trade_skill_plan failed rc={plan_build.get('returncode', 0)}")

        plan_payload = plan_build.get("payload", {}) if isinstance(plan_build.get("payload", {}), dict) else {}
        selected = plan_payload.get("selected", {}) if isinstance(plan_payload.get("selected", {}), dict) else {}
        selected_executable = bool(selected.get("executable", False))
        selected_quote = float(selected.get("quote_usdt", 0.0) or 0.0)
        if bool(args.use_selected_quote) and selected_quote > 0.0:
            exec_quote = float(selected_quote)
        else:
            exec_quote = min(float(args.canary_quote_usdt), selected_quote) if selected_quote > 0.0 else float(args.canary_quote_usdt)
        if exec_quote <= 0.0:
            exec_quote = float(args.canary_quote_usdt)

        guarded = {"returncode": 0, "payload": {}, "stdout": "", "stderr": ""}
        auto_unwind = {"attempted": False, "returncode": 0, "payload": {}}
        status = "skipped_no_executable_candidate"

        if selected_executable:
            mode = "canary" if bool(args.run_live_canary) else "probe"
            guarded = run_json_command(
                cmd=build_guard_cmd(
                    system_root=system_root,
                    args=args,
                    mode=mode,
                    quote_usdt=exec_quote,
                    order_symbol=str(selected.get("symbol", "")),
                    order_side=str(selected.get("side", "BUY")),
                    order_quantity=0.0,
                    risk_intent="normal",
                ),
                cwd=system_root,
                timeout_seconds=timeout_seconds,
            )
            if int(guarded.get("returncode", 0)) not in {0}:
                raise RuntimeError(f"guarded_exec failed rc={guarded.get('returncode', 0)}")
            guard_payload = guarded.get("payload", {}) if isinstance(guarded.get("payload", {}), dict) else {}
            status = str(guard_payload.get("status", "guarded_completed"))

            fill = extract_open_fill(guard_payload)
            if bool(args.auto_unwind_live_canary) and bool(args.run_live_canary) and bool(fill.get("executed", False)) and float(fill.get("quantity", 0.0)) > 0.0:
                unwind_result = run_json_command(
                    cmd=build_guard_cmd(
                        system_root=system_root,
                        args=args,
                        mode="canary",
                        quote_usdt=max(0.0001, exec_quote),
                        order_symbol=str(fill.get("symbol", "")),
                        order_side=str(fill.get("reverse_side", "SELL")),
                        order_quantity=float(fill.get("quantity", 0.0)),
                        risk_intent="reduce_only",
                    ),
                    cwd=system_root,
                    timeout_seconds=timeout_seconds,
                )
                auto_unwind = {
                    "attempted": True,
                    "returncode": int(unwind_result.get("returncode", 0)),
                    "payload": unwind_result.get("payload", {}),
                }
                if int(unwind_result.get("returncode", 0)) != 0:
                    panic_close_all(
                        output_root,
                        reason="ticketed_canary_auto_unwind_failure",
                        detail=f"rc={unwind_result.get('returncode', 0)}; stderr={str(unwind_result.get('stderr', ''))[:400]}",
                    )

        payload = {
            "generated_at_utc": now_utc_iso(),
            "action": "ticketed_canary_pipeline",
            "workspace": str(system_root.parent),
            "system_root": str(system_root),
            "run_live_canary": bool(args.run_live_canary),
            "auto_unwind_live_canary": bool(args.auto_unwind_live_canary),
            "status": status,
            "selected": selected,
            "effective_canary_quote_usdt": float(exec_quote),
            "equity_usdt": float(args.equity_usdt),
            "use_selected_quote": bool(args.use_selected_quote),
            "ticket_build": {
                "returncode": int(ticket_build.get("returncode", 0)),
                "payload": ticket_build.get("payload", {}),
            },
            "plan_build": {
                "returncode": int(plan_build.get("returncode", 0)),
                "payload": plan_payload,
            },
            "guarded_exec": {
                "returncode": int(guarded.get("returncode", 0)),
                "payload": guarded.get("payload", {}),
            },
            "auto_unwind": {
                "attempted": bool(auto_unwind.get("attempted", False)),
                "returncode": int(auto_unwind.get("returncode", 0)),
                "payload": auto_unwind.get("payload", {}),
            },
        }

        stamp = now_utc_compact()
        out_json = review_dir / f"{stamp}_ticketed_canary_pipeline.json"
        out_checksum = review_dir / f"{stamp}_ticketed_canary_pipeline_checksum.json"
        write_json(out_json, payload)
        evicted = evict_old_pipeline_artifacts(
            directory=review_dir,
            ttl_hours=max(1, int(args.artifact_ttl_hours)),
            keep_names={out_json.name, out_checksum.name},
        )
        payload["governance"] = {
            "artifact_ttl_hours": int(max(1, int(args.artifact_ttl_hours))),
            "evicted_files": int(evicted),
        }
        payload["artifact"] = str(out_json)
        write_json(out_json, payload)
        digest, size_bytes = sha256_file(out_json)
        write_json(
            out_checksum,
            {
                "generated_at_utc": now_utc_iso(),
                "files": [{"path": str(out_json), "sha256": digest, "size_bytes": int(size_bytes)}],
            },
        )
        payload["checksum"] = str(out_checksum)
        write_json(out_json, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    except PanicTriggered as exc:
        payload = {
            "generated_at_utc": now_utc_iso(),
            "action": "ticketed_canary_pipeline",
            "status": "panic",
            "panic": str(exc),
            "workspace": str(system_root.parent),
            "system_root": str(system_root),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2
    except Exception as exc:
        payload = {
            "generated_at_utc": now_utc_iso(),
            "action": "ticketed_canary_pipeline",
            "status": "error",
            "error": str(exc),
            "workspace": str(system_root.parent),
            "system_root": str(system_root),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
