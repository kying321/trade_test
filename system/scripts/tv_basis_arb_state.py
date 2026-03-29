#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from binance_live_common import now_utc_iso, read_json, write_json

SUPPORTED_STATUSES = (
    "entry_pending",
    "spot_buy_submitting",
    "spot_buy_filled_perp_pending",
    "perp_short_submitting",
    "open_hedged",
    "exit_pending",
    "needs_recovery",
    "closed",
)
_ACTIVE_RECOVERY_STATUSES = {"needs_recovery"}
_ALLOWED_PREDECESSORS: dict[str, set[str]] = {
    "spot_buy_submitting": {"entry_pending"},
    "spot_buy_filled_perp_pending": {"spot_buy_submitting"},
    "perp_short_submitting": {"spot_buy_filled_perp_pending"},
    "open_hedged": {"perp_short_submitting"},
    "exit_pending": {"open_hedged"},
    "needs_recovery": {
        "spot_buy_submitting",
        "spot_buy_filled_perp_pending",
        "perp_short_submitting",
        "open_hedged",
        "exit_pending",
    },
    "closed": {"exit_pending", "needs_recovery"},
}


class StateConflictError(RuntimeError):
    pass


class IllegalTransitionError(StateConflictError):
    pass


def _normalize_rows(payload: Any, key: str) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, dict):
        return {}
    rows = payload.get(key, {})
    if not isinstance(rows, dict):
        return {}
    normalized: dict[str, dict[str, Any]] = {}
    for row_key, row in rows.items():
        if isinstance(row, dict):
            normalized[str(row_key)] = dict(row)
    return normalized


def load_idempotency_state(path: Path) -> dict[str, Any]:
    payload = read_json(path, {})
    return {
        "updated_at_utc": str(payload.get("updated_at_utc", "")) if isinstance(payload, dict) else "",
        "attempts": _normalize_rows(payload, "attempts"),
    }


def save_idempotency_state(path: Path, payload: dict[str, Any]) -> None:
    write_json(
        path,
        {
            "updated_at_utc": now_utc_iso(),
            "attempts": _normalize_rows(payload, "attempts"),
        },
    )


def load_positions_state(path: Path) -> dict[str, Any]:
    payload = read_json(path, {})
    return {
        "updated_at_utc": str(payload.get("updated_at_utc", "")) if isinstance(payload, dict) else "",
        "positions": _normalize_rows(payload, "positions"),
    }


def save_positions_state(path: Path, payload: dict[str, Any]) -> None:
    write_json(
        path,
        {
            "updated_at_utc": now_utc_iso(),
            "positions": _normalize_rows(payload, "positions"),
        },
    )


def load_recovery_state(path: Path) -> dict[str, Any]:
    payload = read_json(path, {})
    return {
        "updated_at_utc": str(payload.get("updated_at_utc", "")) if isinstance(payload, dict) else "",
        "recoveries": _normalize_rows(payload, "recoveries"),
    }


def save_recovery_state(path: Path, payload: dict[str, Any]) -> None:
    write_json(
        path,
        {
            "updated_at_utc": now_utc_iso(),
            "recoveries": _normalize_rows(payload, "recoveries"),
        },
    )


def _position_key(strategy_id: str, symbol: str, idempotency_key: str) -> str:
    seed = f"{strategy_id}:{symbol.upper()}:{idempotency_key}:tv_basis_arb_v1"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:28]


def _missing_leg(side: str) -> dict[str, Any]:
    return {
        "side": side,
        "status": "missing",
    }


def _dict_contains(superset: Any, subset: Any) -> bool:
    if isinstance(subset, dict):
        if not isinstance(superset, dict):
            return False
        for key, value in subset.items():
            if key not in superset or not _dict_contains(superset[key], value):
                return False
        return True
    return superset == subset


class TvBasisArbStateLedger:
    def __init__(self, *, output_root: Path | str) -> None:
        self.output_root = Path(output_root)
        self.state_root = self.output_root / "state"
        self.idempotency_path = self.state_root / "tv_basis_arb_idempotency.json"
        self.positions_path = self.state_root / "tv_basis_arb_positions.json"
        self.recovery_path = self.state_root / "tv_basis_arb_recovery.json"
        self._bootstrap()

    def _bootstrap(self) -> None:
        if not self.idempotency_path.exists():
            save_idempotency_state(self.idempotency_path, {"attempts": {}})
        if not self.positions_path.exists():
            save_positions_state(self.positions_path, {"positions": {}})
        if not self.recovery_path.exists():
            save_recovery_state(self.recovery_path, {"recoveries": {}})

    def _get_attempt(self, idempotency_key: str) -> dict[str, Any]:
        attempts = load_idempotency_state(self.idempotency_path)["attempts"]
        attempt = attempts.get(idempotency_key)
        if not isinstance(attempt, dict):
            raise KeyError(f"unknown attempt:{idempotency_key}")
        return dict(attempt)

    def _save_attempt(self, attempt: dict[str, Any]) -> dict[str, Any]:
        payload = load_idempotency_state(self.idempotency_path)
        attempts = payload["attempts"]
        attempt_key = str(attempt["attempt_key"])
        previous = attempts.get(attempt_key, {}) if isinstance(attempts.get(attempt_key), dict) else {}
        updated = dict(previous)
        updated.update(attempt)
        updated["attempt_key"] = attempt_key
        updated["updated_at_utc"] = now_utc_iso()
        if "created_at_utc" not in updated:
            updated["created_at_utc"] = updated["updated_at_utc"]
        attempts[attempt_key] = updated
        payload["attempts"] = attempts
        save_idempotency_state(self.idempotency_path, payload)
        return dict(updated)

    def _save_position(self, position: dict[str, Any]) -> dict[str, Any]:
        payload = load_positions_state(self.positions_path)
        positions = payload["positions"]
        position_key = str(position["position_key"])
        previous = positions.get(position_key, {}) if isinstance(positions.get(position_key), dict) else {}
        updated = dict(previous)
        updated.update(position)
        updated["position_key"] = position_key
        updated["updated_at_utc"] = now_utc_iso()
        if "created_at_utc" not in updated:
            updated["created_at_utc"] = updated["updated_at_utc"]
        positions[position_key] = updated
        payload["positions"] = positions
        save_positions_state(self.positions_path, payload)
        return dict(updated)

    def _save_recovery(self, recovery: dict[str, Any]) -> dict[str, Any]:
        payload = load_recovery_state(self.recovery_path)
        recoveries = payload["recoveries"]
        recovery_key = str(recovery["position_key"])
        previous = recoveries.get(recovery_key, {}) if isinstance(recoveries.get(recovery_key), dict) else {}
        updated = dict(previous)
        updated.update(recovery)
        updated["position_key"] = recovery_key
        updated["updated_at_utc"] = now_utc_iso()
        if "created_at_utc" not in updated:
            updated["created_at_utc"] = updated["updated_at_utc"]
        recoveries[recovery_key] = updated
        payload["recoveries"] = recoveries
        save_recovery_state(self.recovery_path, payload)
        return dict(updated)

    def _get_position(self, position_key: str) -> dict[str, Any]:
        positions = load_positions_state(self.positions_path)["positions"]
        position = positions.get(position_key)
        if not isinstance(position, dict):
            raise KeyError(f"unknown position:{position_key}")
        return dict(position)

    def _assert_begin_entry_replay_matches(
        self,
        *,
        attempt: dict[str, Any],
        strategy_id: str,
        symbol: str,
        requested_notional_usdt: float,
        target_base_qty: float | None,
        max_quote_budget_usdt: float | None,
        execution_venue: str | None,
        tv_timestamp: str,
    ) -> None:
        if str(attempt.get("strategy_id", "")) != str(strategy_id):
            raise StateConflictError("idempotency payload mismatch:strategy_id")
        if str(attempt.get("symbol", "")).upper() != str(symbol).upper():
            raise StateConflictError("idempotency payload mismatch:symbol")
        if float(attempt.get("requested_notional_usdt", 0.0)) != float(requested_notional_usdt):
            raise StateConflictError("idempotency payload mismatch:requested_notional_usdt")
        if target_base_qty is not None and float(attempt.get("target_base_qty", 0.0)) != float(target_base_qty):
            raise StateConflictError("idempotency payload mismatch:target_base_qty")
        if max_quote_budget_usdt is not None and float(attempt.get("max_quote_budget_usdt", 0.0)) != float(max_quote_budget_usdt):
            raise StateConflictError("idempotency payload mismatch:max_quote_budget_usdt")
        if execution_venue is not None and str(attempt.get("execution_venue", "")) != str(execution_venue):
            raise StateConflictError("idempotency payload mismatch:execution_venue")
        if str(attempt.get("tv_timestamp", "")) != str(tv_timestamp):
            raise StateConflictError("idempotency payload mismatch:tv_timestamp")

    def _assert_transition_allowed(self, *, current_status: str, target_status: str) -> None:
        if target_status not in SUPPORTED_STATUSES:
            raise ValueError(f"unsupported status:{target_status}")
        if current_status == target_status:
            return
        allowed = _ALLOWED_PREDECESSORS.get(target_status, set())
        if current_status not in allowed:
            raise IllegalTransitionError(f"illegal transition:{current_status}->{target_status}")

    def _assert_replay_patch_matches(
        self,
        *,
        current_attempt: dict[str, Any],
        current_position: dict[str, Any],
        attempt_patch: dict[str, Any] | None,
        position_patch: dict[str, Any] | None,
    ) -> None:
        if attempt_patch and not _dict_contains(current_attempt, attempt_patch):
            raise StateConflictError("replay mismatch:attempt_patch")
        if position_patch and not _dict_contains(current_position, position_patch):
            raise StateConflictError("replay mismatch:position_patch")

    def _resolve_recovery_for_position(
        self,
        *,
        position_key: str,
        resolved_status: str,
        close_reason: str,
    ) -> dict[str, Any] | None:
        payload = load_recovery_state(self.recovery_path)
        current = payload["recoveries"].get(position_key)
        if not isinstance(current, dict):
            return None
        recovery = dict(current)
        recovery["status"] = str(resolved_status)
        recovery["resolved_at_utc"] = now_utc_iso()
        recovery["resolved_by"] = "record_closed"
        recovery["close_reason"] = str(close_reason)
        return self._save_recovery(recovery)

    def can_start_entry(self, *, strategy_id: str, symbol: str) -> tuple[bool, str]:
        recoveries = load_recovery_state(self.recovery_path)["recoveries"]
        want_symbol = str(symbol).upper()
        for recovery in recoveries.values():
            if not isinstance(recovery, dict):
                continue
            if str(recovery.get("strategy_id", "")) != str(strategy_id):
                continue
            if str(recovery.get("symbol", "")).upper() != want_symbol:
                continue
            if str(recovery.get("status", "")).strip().lower() in _ACTIVE_RECOVERY_STATUSES:
                return False, "recovery_required"
        return True, ""

    def begin_entry(
        self,
        *,
        strategy_id: str,
        symbol: str,
        idempotency_key: str,
        requested_notional_usdt: float,
        target_base_qty: float | None = None,
        max_quote_budget_usdt: float | None = None,
        execution_venue: str | None = None,
        tv_timestamp: str,
    ) -> dict[str, Any]:
        existing_attempt = self._get_attempt_or_none(idempotency_key)
        if existing_attempt is not None:
            self._assert_begin_entry_replay_matches(
                attempt=existing_attempt,
                strategy_id=strategy_id,
                symbol=symbol,
                requested_notional_usdt=requested_notional_usdt,
                target_base_qty=target_base_qty,
                max_quote_budget_usdt=max_quote_budget_usdt,
                execution_venue=execution_venue,
                tv_timestamp=tv_timestamp,
            )
            return existing_attempt
        allowed, reason = self.can_start_entry(strategy_id=strategy_id, symbol=symbol)
        if not allowed:
            raise StateConflictError(f"recovery exists:{reason}")

        symbol_txt = str(symbol).upper()
        position_key = _position_key(strategy_id, symbol_txt, idempotency_key)
        attempt = self._save_attempt(
            {
                "attempt_key": idempotency_key,
                "strategy_id": str(strategy_id),
                "symbol": symbol_txt,
                "status": "entry_pending",
                "phase": "entry_pending",
                "tv_timestamp": str(tv_timestamp),
                "requested_notional_usdt": float(requested_notional_usdt),
                "target_base_qty": None if target_base_qty is None else float(target_base_qty),
                "max_quote_budget_usdt": None if max_quote_budget_usdt is None else float(max_quote_budget_usdt),
                "execution_venue": None if execution_venue is None else str(execution_venue),
                "position_key": position_key,
                "spot_leg": _missing_leg("spot_buy"),
                "perp_leg": _missing_leg("perp_short"),
            }
        )
        self._save_position(
            {
                "position_key": position_key,
                "attempt_key": idempotency_key,
                "strategy_id": str(strategy_id),
                "symbol": symbol_txt,
                "status": "entry_pending",
                "tv_timestamp": str(tv_timestamp),
                "requested_notional_usdt": float(requested_notional_usdt),
                "target_base_qty": None if target_base_qty is None else float(target_base_qty),
                "max_quote_budget_usdt": None if max_quote_budget_usdt is None else float(max_quote_budget_usdt),
                "execution_venue": None if execution_venue is None else str(execution_venue),
                "spot_leg": dict(attempt["spot_leg"]),
                "perp_leg": dict(attempt["perp_leg"]),
            }
        )
        return attempt

    def _get_attempt_or_none(self, idempotency_key: str) -> dict[str, Any] | None:
        try:
            return self._get_attempt(idempotency_key)
        except KeyError:
            return None

    def _update_attempt_and_position(
        self,
        *,
        idempotency_key: str,
        status: str,
        phase: str | None = None,
        attempt_patch: dict[str, Any] | None = None,
        position_patch: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        current_attempt = self._get_attempt(idempotency_key)
        current_position = self._get_position(str(current_attempt["position_key"]))
        current_status = str(current_attempt.get("status", ""))
        self._assert_transition_allowed(current_status=current_status, target_status=status)
        if current_status == status:
            self._assert_replay_patch_matches(
                current_attempt=current_attempt,
                current_position=current_position,
                attempt_patch=attempt_patch,
                position_patch=position_patch,
            )
            return current_attempt, current_position
        updated_attempt = dict(current_attempt)
        updated_attempt["status"] = status
        updated_attempt["phase"] = str(phase or status)
        if attempt_patch:
            updated_attempt.update(attempt_patch)
        saved_attempt = self._save_attempt(updated_attempt)

        updated_position = dict(current_position)
        updated_position.update(
            {
                "position_key": saved_attempt["position_key"],
                "attempt_key": idempotency_key,
                "strategy_id": saved_attempt["strategy_id"],
                "symbol": saved_attempt["symbol"],
                "status": status,
                "tv_timestamp": saved_attempt["tv_timestamp"],
                "requested_notional_usdt": saved_attempt["requested_notional_usdt"],
                "target_base_qty": saved_attempt.get("target_base_qty"),
                "max_quote_budget_usdt": saved_attempt.get("max_quote_budget_usdt"),
                "execution_venue": saved_attempt.get("execution_venue"),
                "spot_leg": dict(saved_attempt.get("spot_leg", _missing_leg("spot_buy"))),
                "perp_leg": dict(saved_attempt.get("perp_leg", _missing_leg("perp_short"))),
            }
        )
        if position_patch:
            updated_position.update(position_patch)
        saved_position = self._save_position(updated_position)
        return saved_attempt, saved_position

    def record_spot_buy_submitting(self, *, idempotency_key: str, spot_order_id: str) -> dict[str, Any]:
        saved_attempt, _ = self._update_attempt_and_position(
            idempotency_key=idempotency_key,
            status="spot_buy_submitting",
            attempt_patch={
                "spot_leg": {
                    "side": "spot_buy",
                    "status": "submitting",
                    "order_id": str(spot_order_id),
                }
            },
        )
        return saved_attempt

    def record_spot_buy_fill(
        self,
        *,
        idempotency_key: str,
        spot_order_id: str,
        filled_base_qty: float,
        filled_quote_usdt: float,
        partial_fill: bool = False,
    ) -> dict[str, Any]:
        saved_attempt, saved_position = self._update_attempt_and_position(
            idempotency_key=idempotency_key,
            status="spot_buy_filled_perp_pending",
            attempt_patch={
                "spot_leg": {
                    "side": "spot_buy",
                    "status": "filled",
                    "order_id": str(spot_order_id),
                    "filled_base_qty": float(filled_base_qty),
                    "filled_quote_usdt": float(filled_quote_usdt),
                    "partial_fill": bool(partial_fill),
                }
            },
        )
        return saved_position | {"attempt_key": saved_attempt["attempt_key"]}

    def record_perp_short_submitting(
        self,
        *,
        idempotency_key: str,
        perp_order_id: str,
        target_base_qty: float,
    ) -> dict[str, Any]:
        saved_attempt, _ = self._update_attempt_and_position(
            idempotency_key=idempotency_key,
            status="perp_short_submitting",
            attempt_patch={
                "perp_leg": {
                    "side": "perp_short",
                    "status": "submitting",
                    "order_id": str(perp_order_id),
                    "target_base_qty": float(target_base_qty),
                }
            },
        )
        return saved_attempt

    def record_open_hedged(
        self,
        *,
        idempotency_key: str,
        perp_order_id: str,
        filled_base_qty: float,
        avg_entry_price: float,
        basis_bps: float,
    ) -> dict[str, Any]:
        _, saved_position = self._update_attempt_and_position(
            idempotency_key=idempotency_key,
            status="open_hedged",
            attempt_patch={
                "perp_leg": {
                    "side": "perp_short",
                    "status": "filled",
                    "order_id": str(perp_order_id),
                    "filled_base_qty": float(filled_base_qty),
                    "avg_entry_price": float(avg_entry_price),
                }
            },
            position_patch={
                "basis_bps": float(basis_bps),
            },
        )
        return saved_position

    def record_exit_pending(self, *, idempotency_key: str, reason: str = "") -> dict[str, Any]:
        _, saved_position = self._update_attempt_and_position(
            idempotency_key=idempotency_key,
            status="exit_pending",
            attempt_patch={"exit_reason": str(reason)},
            position_patch={"exit_reason": str(reason)},
        )
        return saved_position

    def record_closed(self, *, idempotency_key: str, close_reason: str = "") -> dict[str, Any]:
        _, saved_position = self._update_attempt_and_position(
            idempotency_key=idempotency_key,
            status="closed",
            attempt_patch={"close_reason": str(close_reason)},
            position_patch={"close_reason": str(close_reason), "closed_at_utc": now_utc_iso()},
        )
        self._resolve_recovery_for_position(
            position_key=str(saved_position["position_key"]),
            resolved_status="closed",
            close_reason=close_reason,
        )
        return saved_position

    def record_needs_recovery(
        self,
        *,
        idempotency_key: str,
        reason: str,
        failure_phase: str,
        recovery_action: str,
    ) -> dict[str, Any]:
        saved_attempt, saved_position = self._update_attempt_and_position(
            idempotency_key=idempotency_key,
            status="needs_recovery",
            phase=failure_phase,
            attempt_patch={
                "failure_reason": str(reason),
                "recovery_action": str(recovery_action),
            },
            position_patch={
                "failure_reason": str(reason),
                "recovery_action": str(recovery_action),
            },
        )
        recovery = self._save_recovery(
            {
                "position_key": saved_position["position_key"],
                "attempt_key": saved_attempt["attempt_key"],
                "strategy_id": saved_position["strategy_id"],
                "symbol": saved_position["symbol"],
                "status": "needs_recovery",
                "reason": str(reason),
                "failure_phase": str(failure_phase),
                "recovery_action": str(recovery_action),
                "target_base_qty": saved_position.get("target_base_qty"),
                "max_quote_budget_usdt": saved_position.get("max_quote_budget_usdt"),
                "execution_venue": saved_position.get("execution_venue"),
                "spot_leg": dict(saved_position.get("spot_leg", _missing_leg("spot_buy"))),
                "perp_leg": dict(saved_position.get("perp_leg", _missing_leg("perp_short"))),
            }
        )
        return recovery
