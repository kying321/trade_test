#!/usr/bin/env python3
"""Safe paper-state reset utility.

Default is dry-run; use --write to persist.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict

from lie_root_resolver import resolve_lie_system_root

DEFAULT_STATE_PATH = resolve_lie_system_root() / "output" / "state" / "spot_paper_state.json"


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "date": dt.date.today().isoformat(),
            "cash_usdt": 100.0,
            "eth_qty": 0.0,
            "avg_cost": 0.0,
            "equity_peak": 100.0,
            "daily_realized_pnl": 0.0,
            "consecutive_losses": 0,
            "last_loss_ts": None,
        }
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_state(st: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "date": str(st.get("date") or dt.date.today().isoformat()),
        "cash_usdt": round(_safe_float(st.get("cash_usdt"), 0.0), 8),
        "eth_qty": round(_safe_float(st.get("eth_qty"), 0.0), 8),
        "avg_cost": round(_safe_float(st.get("avg_cost"), 0.0), 8),
        "equity_peak": round(_safe_float(st.get("equity_peak"), 0.0), 8),
        "daily_realized_pnl": round(_safe_float(st.get("daily_realized_pnl"), 0.0), 8),
        "consecutive_losses": _safe_int(st.get("consecutive_losses"), 0),
        "last_loss_ts": str(st.get("last_loss_ts") or "").strip() or None,
    }


def backup_state(path: Path, current: Dict[str, Any]) -> Path:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    bdir = path.parent / "backup"
    bdir.mkdir(parents=True, exist_ok=True)
    out = bdir / f"spot_paper_state_{ts}.json"
    out.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-path", default=str(DEFAULT_STATE_PATH))
    ap.add_argument("--set-cash", type=float)
    ap.add_argument("--set-eth", type=float)
    ap.add_argument("--set-avg-cost", type=float)
    ap.add_argument("--set-consecutive-losses", type=int)
    ap.add_argument("--set-daily-realized-pnl", type=float)
    ap.add_argument("--set-last-loss-ts")
    ap.add_argument("--set-date")
    ap.add_argument("--reset-guardrail", action="store_true", help="set consecutive_losses=0 and daily_realized_pnl=0")
    ap.add_argument("--clear-last-loss-ts", action="store_true", help="clear last_loss_ts")
    ap.add_argument("--reset-equity-peak", action="store_true", help="set equity_peak to current equity")
    ap.add_argument("--write", action="store_true", help="persist changes (default is dry-run)")
    args = ap.parse_args()

    path = Path(args.state_path)
    current = normalize_state(load_state(path))
    target = dict(current)

    if args.set_cash is not None:
        target["cash_usdt"] = round(float(args.set_cash), 8)
    if args.set_eth is not None:
        target["eth_qty"] = round(float(args.set_eth), 8)
    if args.set_avg_cost is not None:
        target["avg_cost"] = round(float(args.set_avg_cost), 8)
    if args.set_consecutive_losses is not None:
        target["consecutive_losses"] = int(args.set_consecutive_losses)
    if args.set_daily_realized_pnl is not None:
        target["daily_realized_pnl"] = round(float(args.set_daily_realized_pnl), 8)
    if args.set_last_loss_ts:
        target["last_loss_ts"] = str(args.set_last_loss_ts).strip()
    if args.set_date:
        target["date"] = str(args.set_date)

    if args.reset_guardrail:
        target["consecutive_losses"] = 0
        target["daily_realized_pnl"] = 0.0
        target["last_loss_ts"] = None
    if args.clear_last_loss_ts:
        target["last_loss_ts"] = None

    if args.reset_equity_peak:
        target["equity_peak"] = round(target["cash_usdt"] + target["eth_qty"] * target["avg_cost"], 8)

    target = normalize_state(target)

    payload = {
        "state_path": str(path),
        "write": bool(args.write),
        "current": current,
        "target": target,
    }

    if args.write:
        path.parent.mkdir(parents=True, exist_ok=True)
        bkp = backup_state(path, current)
        path.write_text(json.dumps(target, ensure_ascii=False) + "\n", encoding="utf-8")
        payload["backup_path"] = str(bkp)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
