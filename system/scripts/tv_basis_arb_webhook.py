from __future__ import annotations

import hashlib
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from tv_basis_arb_common import (
    TvBasisWebhookSignal,
    load_strategy_definition,
    parse_tv_basis_webhook_payload,
)
from tv_basis_arb_executor import TvBasisArbExecutor
from tv_basis_arb_gate import build_market_snapshot, evaluate_tv_basis_gate
from tv_basis_arb_state import load_positions_state, load_recovery_state

_STRATEGY_RUNTIME_POLICY: dict[str, dict[str, float | None]] = {
    "tv_basis_btc_spot_perp_v1": {
        "requested_notional_usdt": None,
        "exit_basis_bps": 4.0,
        "max_holding_seconds": 3600.0,
    }
}


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sanitize_filename(value: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    result = "".join(ch if ch in allowed else "_" for ch in value)
    return result or "signal"


def _deterministic_hash(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]


def _unique_suffix(signal: TvBasisWebhookSignal, payload: Dict[str, Any]) -> str:
    parts: list[str] = []
    if signal.alert_id:
        parts.append(_sanitize_filename(signal.alert_id))
    parts.append(_deterministic_hash(payload))
    parts.append(uuid.uuid4().hex[:6])
    return "_".join(parts)


def signal_artifact_path(output_root: Path, signal: TvBasisWebhookSignal, payload: Dict[str, Any]) -> Path:
    timestamp_safe = _sanitize_filename(signal.tv_timestamp)
    suffix = _unique_suffix(signal, payload)
    artifact_dir = output_root / "review" / "tv_basis_arb"
    return artifact_dir / f"{timestamp_safe}_{signal.strategy_id}_{signal.event_type}_{suffix}.json"


def gate_artifact_path(signal_path: Path) -> Path:
    return signal_path.with_name(f"{signal_path.stem}_gate.json")


def execution_artifact_path(output_root: Path) -> Path:
    return output_root / "state" / "tv_basis_arb_idempotency.json"


def position_artifact_path(output_root: Path) -> Path:
    return output_root / "state" / "tv_basis_arb_positions.json"


def closeout_artifact_path(output_root: Path) -> Path:
    return output_root / "state" / "tv_basis_arb_recovery.json"


def _build_signal_payload(signal: TvBasisWebhookSignal) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "strategy_id": signal.strategy_id,
        "symbol": signal.symbol,
        "event_type": signal.event_type,
        "tv_timestamp": signal.tv_timestamp,
    }
    if signal.alert_id is not None:
        data["alert_id"] = signal.alert_id
    return data


def write_signal_artifact(output_root: Path, signal: TvBasisWebhookSignal, payload: Dict[str, Any]) -> Path:
    target = signal_artifact_path(output_root, signal, payload)
    _write_json(target, _build_signal_payload(signal))
    return target


def _current_runtime_policy(strategy_id: str) -> dict[str, float]:
    strategy = load_strategy_definition(strategy_id)
    gate = strategy.get("gate", {})
    if not isinstance(gate, dict):
        raise ValueError(f"missing gate config:{strategy_id}")
    policy = dict(_STRATEGY_RUNTIME_POLICY.get(strategy_id, {}))
    requested_notional = policy.get("requested_notional_usdt")
    if requested_notional is None:
        requested_notional = gate.get("max_notional_usdt", 0.0)
    exit_basis_bps = policy.get("exit_basis_bps")
    if exit_basis_bps is None:
        exit_basis_bps = float(gate.get("min_basis_bps", 0.0)) / 2.0
    max_holding_seconds = policy.get("max_holding_seconds")
    if max_holding_seconds is None:
        max_holding_seconds = 3600.0
    return {
        "requested_notional_usdt": float(requested_notional),
        "exit_basis_bps": float(exit_basis_bps),
        "max_holding_seconds": float(max_holding_seconds),
    }


def _iso_to_utc(ts: str) -> datetime:
    normalized = str(ts).strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _holding_time_seconds(entry_ts: str, exit_ts: str) -> float:
    delta = _iso_to_utc(exit_ts) - _iso_to_utc(entry_ts)
    return max(0.0, delta.total_seconds())


def _basis_bps_from_snapshot(snapshot: dict[str, Any]) -> float:
    spot_price = float(snapshot.get("spot_price", 0.0) or 0.0)
    perp_mark_price = float(snapshot.get("perp_mark_price", 0.0) or 0.0)
    if spot_price <= 0.0:
        return 0.0
    return ((perp_mark_price - spot_price) / spot_price) * 10_000.0


def _entry_idempotency_key(signal: TvBasisWebhookSignal) -> str:
    payload = {
        "strategy_id": signal.strategy_id,
        "symbol": signal.symbol,
        "event_type": signal.event_type,
        "tv_timestamp": signal.tv_timestamp,
        "alert_id": signal.alert_id or "",
    }
    return f"tv-basis-entry-{_deterministic_hash(payload)}"


def _latest_open_position(*, output_root: Path, strategy_id: str, symbol: str) -> dict[str, Any] | None:
    positions_payload = load_positions_state(output_root / "state" / "tv_basis_arb_positions.json")
    candidates: list[dict[str, Any]] = []
    for position in positions_payload["positions"].values():
        if not isinstance(position, dict):
            continue
        if str(position.get("strategy_id", "")) != str(strategy_id):
            continue
        if str(position.get("symbol", "")).upper() != str(symbol).upper():
            continue
        if str(position.get("status", "")) != "open_hedged":
            continue
        candidates.append(dict(position))
    if not candidates:
        return None
    candidates.sort(
        key=lambda row: (
            str(row.get("tv_timestamp", "")),
            str(row.get("updated_at_utc", "")),
            str(row.get("created_at_utc", "")),
        )
    )
    return candidates[-1]


def _latest_pending_recovery(*, output_root: Path, strategy_id: str, symbol: str) -> dict[str, Any] | None:
    recoveries_payload = load_recovery_state(closeout_artifact_path(output_root))
    candidates: list[dict[str, Any]] = []
    for recovery in recoveries_payload["recoveries"].values():
        if not isinstance(recovery, dict):
            continue
        if str(recovery.get("strategy_id", "")) != str(strategy_id):
            continue
        if str(recovery.get("symbol", "")).upper() != str(symbol).upper():
            continue
        if str(recovery.get("status", "")).strip().lower() != "needs_recovery":
            continue
        candidates.append(dict(recovery))
    if not candidates:
        return None
    candidates.sort(
        key=lambda row: (
            str(row.get("updated_at_utc", "")),
            str(row.get("created_at_utc", "")),
            str(row.get("position_key", "")),
        )
    )
    return candidates[-1]


def _write_gate_artifact(signal_path: Path, payload: dict[str, Any]) -> Path:
    target = gate_artifact_path(signal_path)
    _write_json(target, payload)
    return target


def _review_artifact_family(output_root: Path, *, include_closeout: bool) -> dict[str, str]:
    family = {
        "execution_artifact_path": str(execution_artifact_path(output_root)),
        "position_artifact_path": str(position_artifact_path(output_root)),
    }
    if include_closeout:
        family["closeout_artifact_path"] = str(closeout_artifact_path(output_root))
    return family


def _attach_review_artifact_family(payload: dict[str, Any], output_root: Path, *, include_closeout: bool) -> dict[str, str]:
    family = _review_artifact_family(output_root, include_closeout=include_closeout)
    payload.update(family)
    return family


def _base_gate_payload(
    *,
    signal: TvBasisWebhookSignal,
    signal_path: Path,
    runtime_policy: dict[str, float],
    requested_notional_usdt: float,
    max_holding_seconds: float,
    holding_time_seconds: float,
) -> dict[str, Any]:
    return {
        "strategy_id": signal.strategy_id,
        "symbol": signal.symbol,
        "event_type": signal.event_type,
        "tv_timestamp": signal.tv_timestamp,
        "signal_artifact_path": str(signal_path),
        "requested_notional_usdt": float(requested_notional_usdt),
        "holding_time_seconds": float(holding_time_seconds),
        "max_holding_seconds": float(max_holding_seconds),
        "runtime_policy": dict(runtime_policy),
    }


def _handle_entry_check(
    *,
    output_root: Path,
    signal: TvBasisWebhookSignal,
    signal_path: Path,
    spot_client: Any | None,
    perp_client: Any | None,
) -> dict[str, Any]:
    runtime_policy = _current_runtime_policy(signal.strategy_id)
    requested_notional_usdt = float(runtime_policy["requested_notional_usdt"])
    market_snapshot = build_market_snapshot(
        symbol=signal.symbol,
        spot_client=spot_client,
        perp_client=perp_client,
    )
    gate = evaluate_tv_basis_gate(
        strategy_id=signal.strategy_id,
        requested_notional_usdt=requested_notional_usdt,
        market_snapshot=market_snapshot,
    )
    artifact_payload = _base_gate_payload(
        signal=signal,
        signal_path=signal_path,
        runtime_policy=runtime_policy,
        requested_notional_usdt=requested_notional_usdt,
        max_holding_seconds=float(runtime_policy["max_holding_seconds"]),
        holding_time_seconds=0.0,
    )
    artifact_payload.update(
        {
            "artifact_kind": "entry_gate",
            "action": "gate_blocked",
            "idempotency_key": _entry_idempotency_key(signal),
            "passed": bool(gate["passed"]),
            "reasons": list(gate["reasons"]),
            "basis_bps": float(gate["basis_bps"]),
            "mark_index_spread_bps": float(gate["mark_index_spread_bps"]),
            "open_interest_usdt": float(gate["open_interest_usdt"]),
            "thresholds": dict(gate["thresholds"]),
            "snapshot_ts_utc": gate.get("snapshot_ts_utc"),
            "snapshot_time_ms": gate.get("snapshot_time_ms"),
        }
    )
    execution: dict[str, Any] | None = None
    status = "gate_blocked"
    if bool(gate["passed"]):
        executor = TvBasisArbExecutor(
            output_root=output_root,
            spot_client=spot_client,
            perp_client=perp_client,
        )
        execution = executor.execute_entry(
            strategy_id=signal.strategy_id,
            symbol=signal.symbol,
            idempotency_key=str(artifact_payload["idempotency_key"]),
            requested_notional_usdt=requested_notional_usdt,
            tv_timestamp=signal.tv_timestamp,
        )
        artifact_payload["action"] = "execute_entry"
        artifact_payload["execution_status"] = execution.get("status")
        artifact_payload["position_key"] = execution.get("position", {}).get("position_key")
        artifact_family = _attach_review_artifact_family(
            artifact_payload,
            output_root,
            include_closeout=str(execution.get("status", "")) == "needs_recovery",
        )
        status = str(execution.get("status", "gate_blocked"))
    else:
        artifact_family = {}
    gate_path = _write_gate_artifact(signal_path, artifact_payload)
    result = {
        "status": status,
        "signal_artifact_path": str(signal_path),
        "gate_artifact_path": str(gate_path),
        "gate": artifact_payload,
        "execution": execution,
    }
    result.update(artifact_family)
    return result


def _handle_exit_check(
    *,
    output_root: Path,
    signal: TvBasisWebhookSignal,
    signal_path: Path,
    spot_client: Any | None,
    perp_client: Any | None,
) -> dict[str, Any]:
    runtime_policy = _current_runtime_policy(signal.strategy_id)
    recovery = _latest_pending_recovery(output_root=output_root, strategy_id=signal.strategy_id, symbol=signal.symbol)
    position = _latest_open_position(output_root=output_root, strategy_id=signal.strategy_id, symbol=signal.symbol)
    requested_notional_usdt = float(
        position.get("requested_notional_usdt", runtime_policy["requested_notional_usdt"]) if isinstance(position, dict) else runtime_policy["requested_notional_usdt"]
    )
    holding_time_seconds = (
        _holding_time_seconds(str(position.get("tv_timestamp", signal.tv_timestamp)), signal.tv_timestamp)
        if isinstance(position, dict)
        else 0.0
    )
    artifact_payload = _base_gate_payload(
        signal=signal,
        signal_path=signal_path,
        runtime_policy=runtime_policy,
        requested_notional_usdt=requested_notional_usdt,
        max_holding_seconds=float(runtime_policy["max_holding_seconds"]),
        holding_time_seconds=holding_time_seconds,
    )
    artifact_payload["artifact_kind"] = "exit_gate"
    artifact_payload["action"] = "no_open_position"
    artifact_payload["close_reason"] = ""
    artifact_payload["exit_basis_bps"] = float(runtime_policy["exit_basis_bps"])
    execution: dict[str, Any] | None = None
    status = "no_open_position"
    artifact_family: dict[str, str] = {}

    if isinstance(recovery, dict):
        artifact_payload.update(
            {
                "action": "recovery_required",
                "position_key": recovery.get("position_key"),
                "attempt_key": recovery.get("attempt_key"),
                "recovery_reason": recovery.get("reason"),
                "recovery_action": recovery.get("recovery_action"),
                "failure_phase": recovery.get("failure_phase"),
            }
        )
        artifact_family = _attach_review_artifact_family(artifact_payload, output_root, include_closeout=True)
        gate_path = _write_gate_artifact(signal_path, artifact_payload)
        result = {
            "status": "needs_recovery",
            "signal_artifact_path": str(signal_path),
            "gate_artifact_path": str(gate_path),
            "gate": artifact_payload,
            "execution": None,
            "recovery": recovery,
        }
        result.update(artifact_family)
        return result

    if isinstance(position, dict):
        market_snapshot = build_market_snapshot(
            symbol=signal.symbol,
            spot_client=spot_client,
            perp_client=perp_client,
        )
        current_basis_bps = _basis_bps_from_snapshot(market_snapshot)
        basis_reverted = current_basis_bps <= float(runtime_policy["exit_basis_bps"])
        max_holding_exceeded = holding_time_seconds >= float(runtime_policy["max_holding_seconds"])
        close_reason = ""
        reasons: list[str] = []
        if basis_reverted:
            close_reason = "basis_reverted"
            reasons.append("basis_reverted")
        if max_holding_exceeded:
            if not close_reason:
                close_reason = "max_holding_time_exceeded"
            reasons.append("max_holding_time_exceeded")
        artifact_payload.update(
            {
                "attempt_key": position.get("attempt_key"),
                "position_key": position.get("position_key"),
                "basis_bps": float(current_basis_bps),
                "snapshot_ts_utc": market_snapshot.get("snapshot_ts_utc"),
                "snapshot_time_ms": market_snapshot.get("snapshot_time_ms"),
                "should_exit": bool(reasons),
                "reasons": reasons,
                "close_reason": close_reason,
            }
        )
        if reasons:
            executor = TvBasisArbExecutor(
                output_root=output_root,
                spot_client=spot_client,
                perp_client=perp_client,
            )
            execution = executor.execute_exit(
                idempotency_key=str(position["attempt_key"]),
                close_reason=close_reason,
            )
            artifact_payload["action"] = "execute_exit"
            artifact_payload["execution_status"] = execution.get("status")
            artifact_family = _attach_review_artifact_family(
                artifact_payload,
                output_root,
                include_closeout=str(execution.get("status", "")) == "needs_recovery",
            )
            status = str(execution.get("status", "hold_position"))
        else:
            artifact_payload["action"] = "hold_position"
            status = "hold_position"

    gate_path = _write_gate_artifact(signal_path, artifact_payload)
    result = {
        "status": status,
        "signal_artifact_path": str(signal_path),
        "gate_artifact_path": str(gate_path),
        "gate": artifact_payload,
        "execution": execution,
    }
    result.update(artifact_family)
    return result


def handle_webhook(
    payload: Dict[str, Any],
    *,
    output_root: Path | str,
    spot_client: Any | None = None,
    perp_client: Any | None = None,
) -> dict[str, Any]:
    root = Path(output_root)
    signal = parse_tv_basis_webhook_payload(payload)
    signal_path = write_signal_artifact(root, signal, payload)
    if signal.event_type == "entry_check":
        return _handle_entry_check(
            output_root=root,
            signal=signal,
            signal_path=signal_path,
            spot_client=spot_client,
            perp_client=perp_client,
        )
    return _handle_exit_check(
        output_root=root,
        signal=signal,
        signal_path=signal_path,
        spot_client=spot_client,
        perp_client=perp_client,
    )
