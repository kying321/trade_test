#!/usr/bin/env python3
"""State Vector Cortex (IRPOTA) with real sensor bindings.

Sensor bindings:
- R: psutil CPU/RAM/disk + token/context queue pressure.
- P: live_multimodal HMM entropy/log-likelihood/shift (with safe fallback).
- O: ccxt account optionality (fuel/exposure) with paper-state fallback.
- I/T/A: integrity checks + trust/alignment guardrails.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from lie_root_resolver import resolve_lie_system_root
from signal_registry import (
    extract_cortex_threshold_overrides,
    extract_runtime_limits,
    load_signal_registry,
    resolve_signal_registry_path,
    validate_signal_registry,
)

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional runtime dep
    psutil = None

try:
    import ccxt  # type: ignore
except Exception:  # pragma: no cover - optional runtime dep
    ccxt = None


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


class Mode(Enum):
    SURVIVAL = "SURVIVAL"
    STABILIZE = "STABILIZE"
    HIBERNATE = "HIBERNATE"
    OBSERVE = "OBSERVE"
    LEARN = "LEARN"
    ACT = "ACT"
    ROLLBACK = "ROLLBACK"


@dataclass
class IRPOTAState:
    I: float  # Integrity
    R: float  # Resources
    P: float  # Predictability
    O: float  # Optionality
    T: int  # Trust Tier (0=highest, 3=hostile)
    A: bool  # Alignment
    signals: Dict[str, Any] = field(default_factory=dict)


class StateVectorCortex:
    def __init__(self, lie_root: Optional[Path] = None):
        # Critical Thresholds (from Appendix B)
        self.I_critical = 0.60
        self.I_warning = 0.80

        self.R_critical = 0.25
        self.R_warning = 0.45

        self.P_critical = 0.30
        self.P_warning = 0.55
        self.P_shock_threshold = 0.25

        self.O_critical = 0.25
        self.O_warning = 0.50

        # State tracking for hysteresis and shock detection
        self._prev_P = 1.0
        self._current_mode = Mode.OBSERVE
        self._hysteresis_counter = 0
        self._minimum_dwell_time = 3

        # Recovery window
        self._recovery_remaining = 0
        self._recovery_duration = 5

        self._cycle = 0

        self.lie_root = lie_root or resolve_lie_system_root()
        self.artifacts_dir = self.lie_root / "output" / "artifacts"
        self.logs_dir = self.lie_root / "output" / "logs"
        self.state_dir = self.lie_root / "output" / "state"

        self._signal_registry_path = resolve_signal_registry_path()
        self._signal_registry = load_signal_registry(self._signal_registry_path)
        self._signal_registry_validation = validate_signal_registry(self._signal_registry)

        threshold_overrides = extract_cortex_threshold_overrides(self._signal_registry)
        self.I_critical = _clamp(threshold_overrides.get("I_critical", self.I_critical))
        self.I_warning = _clamp(threshold_overrides.get("I_warning", self.I_warning))
        self.R_critical = _clamp(threshold_overrides.get("R_critical", self.R_critical))
        self.R_warning = _clamp(threshold_overrides.get("R_warning", self.R_warning))
        self.P_critical = _clamp(threshold_overrides.get("P_critical", self.P_critical))
        self.P_warning = _clamp(threshold_overrides.get("P_warning", self.P_warning))
        self.P_shock_threshold = _clamp(
            threshold_overrides.get("P_shock_threshold", self.P_shock_threshold)
        )
        self.O_critical = _clamp(threshold_overrides.get("O_critical", self.O_critical))
        self.O_warning = _clamp(threshold_overrides.get("O_warning", self.O_warning))

        runtime_limits = extract_runtime_limits(self._signal_registry)
        dq = int(runtime_limits.get("token_queue_hard_limit", 8))
        db = int(runtime_limits.get("context_token_budget", 128000))
        self._token_queue_hard_limit = int(
            os.getenv("CORTEX_TOKEN_QUEUE_HARD_LIMIT", str(max(1, dq)))
        )
        self._context_token_budget = int(
            os.getenv("CORTEX_CONTEXT_TOKEN_BUDGET", str(max(1, db)))
        )
        self._token_log_candidates = [
            Path(os.getenv("CORTEX_TOKEN_LOG", "")) if os.getenv("CORTEX_TOKEN_LOG") else None,
            self.logs_dir / "llm_usage.jsonl",
            self.logs_dir / "token_usage.jsonl",
            Path("/Users/jokenrobot/.openclaw/logs/llm_usage.jsonl"),
            Path("/Users/jokenrobot/.openclaw/logs/token_usage.jsonl"),
        ]
        self._session_dir_candidates = [
            Path("/Users/jokenrobot/.openclaw/agents/main/sessions"),
            Path("/Users/jokenrobot/.openclaw/agents/trader/sessions"),
            Path("/Users/jokenrobot/.openclaw/agents/pi/sessions"),
        ]
        self._queue_file_candidates = [
            Path("/Users/jokenrobot/.openclaw/workspace/queue.md"),
            Path("/Users/jokenrobot/.openclaw/workspaces/pi/queue.md"),
            Path("/Users/jokenrobot/.openclaw/workspaces/trader/queue.md"),
        ]

    def _safe_load_json(self, path: Path) -> Optional[Dict[str, Any]]:
        try:
            if not path.exists():
                return None
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _integrity_score(self) -> Tuple[float, Dict[str, Any]]:
        score = 1.0
        hard_violation = False
        signals: Dict[str, Any] = {}

        checks = [
            (self.artifacts_dir / "params_live.yaml", 0.10, False, "params_live_exists"),
            (self.artifacts_dir / "live_multimodal.json", 0.10, False, "multimodal_exists"),
            (self.logs_dir / "halfhour_daemon_state.json", 0.10, False, "daemon_state_exists"),
            (self.state_dir / "spot_paper_state.json", 0.05, False, "paper_state_exists"),
        ]
        for path, penalty, hard, name in checks:
            exists = path.exists()
            signals[name] = exists
            if not exists:
                score -= penalty
                if hard:
                    hard_violation = True

        # Structured parse checks for critical runtime files.
        mm = self._safe_load_json(self.artifacts_dir / "live_multimodal.json")
        if mm is None and (self.artifacts_dir / "live_multimodal.json").exists():
            hard_violation = True
            score -= 0.35
            signals["multimodal_parse"] = "failed"
        else:
            signals["multimodal_parse"] = "ok" if mm is not None else "missing"

        daemon = self._safe_load_json(self.logs_dir / "halfhour_daemon_state.json")
        if daemon is None and (self.logs_dir / "halfhour_daemon_state.json").exists():
            hard_violation = True
            score -= 0.25
            signals["daemon_parse"] = "failed"
        else:
            signals["daemon_parse"] = "ok" if daemon is not None else "missing"

        # Manual override for chaos testing.
        if os.getenv("CORTEX_FORCE_INTEGRITY_FAIL", "0") == "1":
            hard_violation = True
            score = 0.0
            signals["forced_integrity_fail"] = True

        if hard_violation:
            return 0.0, signals
        return _clamp(score), signals

    def _read_latest_jsonl(self, path: Path, max_lines: int = 80) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            return None

        for raw in reversed(lines[-max_lines:]):
            raw = raw.strip()
            if not raw:
                continue
            if raw.startswith("{"):
                try:
                    return json.loads(raw)
                except Exception:
                    continue
        return None

    def _extract_session_token_ratio(self) -> Tuple[float, str]:
        latest_file: Optional[Path] = None
        latest_mtime = -1.0
        for d in self._session_dir_candidates:
            if not d.exists():
                continue
            for p in d.glob("*.jsonl"):
                try:
                    m = p.stat().st_mtime
                except Exception:
                    continue
                if m > latest_mtime:
                    latest_mtime = m
                    latest_file = p

        if latest_file is None:
            return 0.0, "none"

        try:
            lines = latest_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            return 0.0, str(latest_file)

        token_used = 0.0
        for raw in reversed(lines[-200:]):
            raw = raw.strip()
            if not raw or not raw.startswith("{"):
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            msg = obj.get("message") if isinstance(obj, dict) else None
            if not isinstance(msg, dict):
                continue
            usage = msg.get("usage")
            if not isinstance(usage, dict):
                continue
            token_used = _safe_float(usage.get("totalTokens"), 0.0)
            if token_used > 0:
                break

        ratio = _clamp(token_used / max(1.0, float(self._context_token_budget)))
        return ratio, str(latest_file)

    def _extract_queue_depth(self) -> Tuple[float, str]:
        for p in self._queue_file_candidates:
            if not p.exists():
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            # Prefer markdown task items; fallback to non-empty line count.
            task_cnt = len(re.findall(r"^\s*[-*]\s+\[[ xX]\]", text, flags=re.MULTILINE))
            if task_cnt <= 0:
                lines = [ln for ln in text.splitlines() if ln.strip()]
                task_cnt = len(lines)
            return float(task_cnt), str(p)
        return 0.0, "none"

    def _extract_token_pressure(self) -> Dict[str, Any]:
        queue_depth = 0.0
        context_ratio = 0.0
        token_ratio = 0.0
        source = "none"

        queue_key_candidates = [
            "queue_depth",
            "pending_requests",
            "pending",
            "inflight",
            "request_queue_depth",
        ]
        ratio_key_candidates = [
            "context_utilization",
            "context_ratio",
            "context_usage_ratio",
            "token_usage_ratio",
            "token_budget_ratio",
        ]

        for candidate in self._token_log_candidates:
            if not candidate:
                continue
            payload = self._read_latest_jsonl(candidate)
            if not payload:
                continue

            for key in queue_key_candidates:
                if key in payload:
                    queue_depth = max(queue_depth, _safe_float(payload.get(key), 0.0))
            for key in ratio_key_candidates:
                if key in payload:
                    v = _safe_float(payload.get(key), 0.0)
                    if v > 1.0:
                        v = v / 100.0
                    context_ratio = max(context_ratio, _clamp(v))

            used = _safe_float(payload.get("tokens_used"), 0.0)
            budget = _safe_float(payload.get("tokens_budget"), 0.0)
            if budget > 0:
                token_ratio = max(token_ratio, _clamp(used / budget))

            source = str(candidate)
            break

        # Session log fallback: pull totalTokens from the latest OpenClaw session.
        if source == "none":
            session_ratio, session_src = self._extract_session_token_ratio()
            if session_ratio > 0:
                token_ratio = max(token_ratio, session_ratio)
                source = session_src

        # Weak regex fallback for simple text logs.
        if source == "none":
            for candidate in self._token_log_candidates:
                if not candidate or not candidate.exists():
                    continue
                try:
                    text = candidate.read_text(encoding="utf-8", errors="ignore")[-3000:]
                except Exception:
                    continue
                q = re.findall(r"queue[_\s-]*depth\D+(\d+)", text, flags=re.IGNORECASE)
                c = re.findall(r"context[_\s-]*(?:ratio|usage|utilization)\D+([0-9.]+)", text, flags=re.IGNORECASE)
                if q:
                    queue_depth = max(queue_depth, _safe_float(q[-1], 0.0))
                if c:
                    v = _safe_float(c[-1], 0.0)
                    if v > 1.0:
                        v = v / 100.0
                    context_ratio = max(context_ratio, _clamp(v))
                if q or c:
                    source = str(candidate)
                    break

        queue_depth_fs, queue_src = self._extract_queue_depth()
        if queue_depth_fs > 0:
            queue_depth = max(queue_depth, queue_depth_fs)
            if source == "none":
                source = queue_src

        queue_pressure = _clamp(queue_depth / max(1.0, float(self._token_queue_hard_limit)))
        context_pressure = max(context_ratio, token_ratio)
        pressure = max(queue_pressure, context_pressure)

        return {
            "queue_depth": queue_depth,
            "context_ratio": context_ratio,
            "token_ratio": token_ratio,
            "pressure": pressure,
            "source": source,
        }

    def _resource_score(self) -> Tuple[float, Dict[str, Any]]:
        cpu_pct = 0.0
        mem_pct = 0.0
        disk_pct = 0.0

        if psutil is not None:
            try:
                cpu_pct = float(psutil.cpu_percent(interval=0.15))
            except Exception:
                cpu_pct = 0.0
            try:
                mem_pct = float(psutil.virtual_memory().percent)
            except Exception:
                mem_pct = 0.0
            try:
                disk_pct = float(psutil.disk_usage(str(self.lie_root)).percent)
            except Exception:
                disk_pct = 0.0

        cpu_pressure = _clamp((cpu_pct - 65.0) / 35.0)
        mem_pressure = _clamp((mem_pct - 75.0) / 25.0)
        disk_pressure = _clamp((disk_pct - 85.0) / 15.0)

        token = self._extract_token_pressure()
        token_pressure = _safe_float(token.get("pressure"), 0.0)

        rpi = 0.35 * cpu_pressure + 0.25 * mem_pressure + 0.15 * disk_pressure + 0.25 * token_pressure

        queue_depth = _safe_float(token.get("queue_depth"), 0.0)
        token_overload = queue_depth >= float(self._token_queue_hard_limit)
        if token_overload:
            # Hard collapse when context queue is fully saturated.
            rpi = max(rpi, 0.97)

        R = 1.0 - _clamp(rpi)
        return _clamp(R), {
            "cpu_pct": cpu_pct,
            "mem_pct": mem_pct,
            "disk_pct": disk_pct,
            "cpu_pressure": cpu_pressure,
            "mem_pressure": mem_pressure,
            "disk_pressure": disk_pressure,
            "token": token,
            "token_overload": token_overload,
            "rpi": _clamp(rpi),
        }

    def _predictability_score(self) -> Tuple[float, Dict[str, Any]]:
        mm_path = self.artifacts_dir / "live_multimodal.json"
        mm = self._safe_load_json(mm_path) or {}

        entropy = None
        for key in ("entropy", "hmm_entropy", "regime_entropy", "distribution_entropy"):
            if key in mm:
                entropy = _safe_float(mm.get(key), 0.0)
                break

        ll = None
        for key in ("log_likelihood", "hmm_log_likelihood", "ll"):
            if key in mm:
                ll = _safe_float(mm.get(key), -2.0)
                break

        shift = None
        for key in ("distribution_shift", "psi", "kl_divergence", "drift_score"):
            if key in mm:
                shift = _safe_float(mm.get(key), 0.0)
                break

        micro = abs(_safe_float(mm.get("micro_imbalance"), 0.0))
        sent = abs(_safe_float(mm.get("sentiment_pca"), 0.0))
        onchain = abs(_safe_float(mm.get("onchain_proxy"), 0.0))
        churn = _clamp((micro + min(1.0, sent / 3.0) + min(1.0, onchain / 3.0)) / 3.0)

        if entropy is None:
            entropy = churn * 1.2
        entropy_norm = _clamp(entropy / 1.5)

        if ll is None:
            ll_score = 1.0 - churn
        else:
            # ll in [-3.5, -0.5] -> [0, 1]
            ll_score = _clamp((ll + 3.5) / 3.0)

        if shift is None:
            shift = churn
        shift_pen = _clamp(shift)

        P = _clamp(0.50 * ll_score + 0.35 * (1.0 - entropy_norm) + 0.15 * (1.0 - shift_pen))
        return P, {
            "source": str(mm_path),
            "entropy": entropy,
            "log_likelihood": ll,
            "shift": shift,
            "churn": churn,
            "ll_score": ll_score,
            "entropy_norm": entropy_norm,
            "shift_penalty": shift_pen,
        }

    def _optionality_from_ccxt(self) -> Optional[Dict[str, Any]]:
        if ccxt is None:
            return None

        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_SECRET")
        if not api_key or not api_secret:
            return None

        try:
            ex = ccxt.binance(
                {
                    "apiKey": api_key,
                    "secret": api_secret,
                    "enableRateLimit": True,
                    "timeout": 12000,
                }
            )
            bal = ex.fetch_balance()
            free = bal.get("free", {}) if isinstance(bal, dict) else {}
            used = bal.get("used", {}) if isinstance(bal, dict) else {}
            total = bal.get("total", {}) if isinstance(bal, dict) else {}

            free_usdt = _safe_float(free.get("USDT"), 0.0)
            used_usdt = _safe_float(used.get("USDT"), 0.0)
            total_usdt = _safe_float(total.get("USDT"), free_usdt + used_usdt)

            open_orders = 0
            try:
                open_orders = len(ex.fetch_open_orders("ETH/USDT"))
            except Exception:
                open_orders = 0

            denom = max(1.0, total_usdt)
            fuel_ratio = _clamp(free_usdt / denom)
            exposure_ratio = _clamp(used_usdt / denom)

            return {
                "source": "ccxt",
                "free_usdt": free_usdt,
                "used_usdt": used_usdt,
                "total_usdt": total_usdt,
                "fuel_ratio": fuel_ratio,
                "exposure_ratio": exposure_ratio,
                "open_orders": open_orders,
            }
        except Exception:
            return None

    def _optionality_from_paper(self) -> Dict[str, Any]:
        state_path = self.state_dir / "spot_paper_state.json"
        state = self._safe_load_json(state_path) or {}

        cash = _safe_float(state.get("cash_usdt"), 0.0)
        eth_qty = _safe_float(state.get("eth_qty"), 0.0)
        avg_cost = _safe_float(state.get("avg_cost"), 0.0)

        px = avg_cost if avg_cost > 0 else 2000.0
        equity = max(1.0, cash + eth_qty * px)
        fuel_ratio = _clamp(cash / equity)
        exposure_ratio = _clamp(1.0 - fuel_ratio)

        return {
            "source": "paper_state",
            "state_path": str(state_path),
            "cash_usdt": cash,
            "eth_qty": eth_qty,
            "equity": equity,
            "fuel_ratio": fuel_ratio,
            "exposure_ratio": exposure_ratio,
            "open_orders": 0,
        }

    def _optionality_score(self) -> Tuple[float, Dict[str, Any]]:
        data = self._optionality_from_ccxt() or self._optionality_from_paper()

        fuel_ratio = _safe_float(data.get("fuel_ratio"), 0.0)
        exposure_ratio = _safe_float(data.get("exposure_ratio"), 1.0)
        open_orders = _safe_float(data.get("open_orders"), 0.0)

        kill_switch_exists = (self.artifacts_dir / "params_live.yaml").exists()
        exit_buffer = _clamp(1.0 - exposure_ratio)
        order_pressure = _clamp(open_orders / 10.0)

        O = _clamp(
            0.55 * fuel_ratio + 0.30 * exit_buffer + 0.15 * (1.0 if kill_switch_exists else 0.0) - 0.10 * order_pressure
        )

        data.update(
            {
                "kill_switch_exists": kill_switch_exists,
                "exit_buffer": exit_buffer,
                "order_pressure": order_pressure,
            }
        )
        return O, data

    def _simulate_sensor_readings(self) -> IRPOTAState:
        """Real sensor bindings (legacy method name kept for compatibility)."""
        I, i_signals = self._integrity_score()
        R, r_signals = self._resource_score()
        P, p_signals = self._predictability_score()
        O, o_signals = self._optionality_score()

        # Trust & alignment gates (manual override supported for testing).
        T = int(_clamp(_safe_float(os.getenv("CORTEX_TRUST_TIER", "0"), 0.0), 0.0, 3.0))
        A = os.getenv("CORTEX_ALIGNMENT_OK", "1") != "0"

        state = IRPOTAState(
            I=I,
            R=R,
            P=P,
            O=O,
            T=T,
            A=A,
            signals={
                "integrity": i_signals,
                "resources": r_signals,
                "predictability": p_signals,
                "optionality": o_signals,
                "registry": {
                    "path": str(self._signal_registry_path),
                    "ok": bool(self._signal_registry_validation.get("ok")),
                    "signal_count": int(self._signal_registry_validation.get("signal_count") or 0),
                    "errors": list(self._signal_registry_validation.get("errors") or [])[:5],
                },
            },
        )
        return state

    def _check_hysteresis(self, requested_mode: Optional[Mode], state: IRPOTAState) -> bool:
        """Enforce minimum dwell time and safe-exit thresholds when leaving critical modes."""
        del requested_mode
        critical_modes = {Mode.SURVIVAL, Mode.HIBERNATE, Mode.ROLLBACK, Mode.STABILIZE}

        if self._current_mode in critical_modes:
            if self._hysteresis_counter < self._minimum_dwell_time:
                self._hysteresis_counter += 1
                return False

            can_exit = True
            if self._current_mode == Mode.SURVIVAL and (not state.A or state.I < self.I_warning):
                can_exit = False
            elif self._current_mode == Mode.HIBERNATE and state.R < self.R_warning:
                can_exit = False
            elif self._current_mode == Mode.STABILIZE and (state.I < self.I_warning or state.O < self.O_warning):
                can_exit = False

            if not can_exit:
                return False

        return True

    def _act_allowed(self, state: IRPOTAState, shock: bool, is_micro_probe: bool = False) -> bool:
        if not state.A:
            return False
            
        # [PHASE 11] Monkey Micro-Probe Electric Shock Ping
        # If the agent is intentionally firing a $1 trash metric test to 
        # shock the system out of LLM stasis, bypass the standard warning 
        # thresholds. Only absolute critical death blocks this.
        if is_micro_probe:
            if state.I < self.I_critical or state.R < self.R_critical:
                return False
            return True
            
        if state.I < self.I_warning:
            return False
        if state.R < self.R_warning:
            return False
        if state.O < self.O_warning:
            return False
        if state.T >= 3:
            return False
        if state.P < self.P_warning and not shock:
            return False
            
        # [PHASE 11] Doomsday Shock Vector (-30% Synthetic Stress Test)
        # Mathematically enforce a 30% capital Optionality plunge. If the
        # simulated state drops below the physical life support boundary 
        # (O_critical), shatter the ACT certificate immediately.
        doomsday_optionality = state.O * 0.70
        if doomsday_optionality < self.O_critical:
            return False
            
        return True

    def _learn_allowed(self, state: IRPOTAState, shock: bool) -> bool:
        if not state.A:
            return False
        if state.I < self.I_warning:
            return False
        if state.R < self.R_warning:
            return False
        if shock:
            return False
        return True

    def _enter_mode(self, new_mode: Mode) -> None:
        if self._current_mode != new_mode:
            self._current_mode = new_mode
            self._hysteresis_counter = 0

    def eval_irpota(self) -> Tuple[str, Dict[str, Any]]:
        """Core orchestration gate; must run before any actuation."""
        self._cycle += 1

        forced_mode = os.getenv("CORTEX_FORCE_MODE", "").strip().upper()
        if forced_mode in Mode.__members__:
            m = Mode[forced_mode]
            self._enter_mode(m)
            return m.value, {
                "reason": "forced_mode",
                "trigger": "env:CORTEX_FORCE_MODE",
                "cycle": self._cycle,
            }

        s = self._simulate_sensor_readings()

        # Shock = sudden P drop.
        delta_P = self._prev_P - s.P
        shock = delta_P >= self.P_shock_threshold

        if not s.A:
            self._enter_mode(Mode.SURVIVAL)
            self._prev_P = s.P
            return Mode.SURVIVAL.value, {
                "reason": "priority_1",
                "trigger": "alignment_failed",
                "cycle": self._cycle,
                "state": s.__dict__,
            }

        if not self._check_hysteresis(None, s):
            self._prev_P = s.P
            return self._current_mode.value, {
                "reason": "hysteresis_lock",
                "trigger": "dwell_time_or_warning_threshold",
                "cycle": self._cycle,
                "state": s.__dict__,
            }

        if s.I < self.I_critical:
            self._enter_mode(Mode.STABILIZE)
            self._prev_P = s.P
            return Mode.STABILIZE.value, {
                "reason": "priority_2",
                "trigger": "integrity_critical",
                "cycle": self._cycle,
                "state": s.__dict__,
            }

        if s.R < self.R_critical:
            self._enter_mode(Mode.HIBERNATE)
            self._prev_P = s.P
            return Mode.HIBERNATE.value, {
                "reason": "priority_3",
                "trigger": "resources_critical",
                "cycle": self._cycle,
                "state": s.__dict__,
            }

        if shock:
            self._recovery_remaining = self._recovery_duration
            self._enter_mode(Mode.OBSERVE)
            self._prev_P = s.P
            return Mode.OBSERVE.value, {
                "reason": "priority_4",
                "trigger": "predictability_shock",
                "shock": shock,
                "delta_p": round(delta_P, 6),
                "cycle": self._cycle,
                "state": s.__dict__,
            }

        if s.O < self.O_warning:
            self._enter_mode(Mode.STABILIZE)
            self._prev_P = s.P
            return Mode.STABILIZE.value, {
                "reason": "priority_5",
                "trigger": "optionality_warning",
                "cycle": self._cycle,
                "state": s.__dict__,
            }

        if self._recovery_remaining > 0:
            self._recovery_remaining -= 1
            self._enter_mode(Mode.OBSERVE)
            self._prev_P = s.P
            return Mode.OBSERVE.value, {
                "reason": "priority_6",
                "trigger": "recovery_window",
                "recovery_remaining": self._recovery_remaining,
                "cycle": self._cycle,
                "state": s.__dict__,
            }

        is_micro_probe = os.getenv("CORTEX_IS_MICRO_PROBE", "0") == "1"

        if self._act_allowed(s, shock, is_micro_probe=is_micro_probe):
            self._enter_mode(Mode.ACT)
            self._prev_P = s.P
            return Mode.ACT.value, {
                "reason": "priority_7",
                "trigger": "act_allowed",
                "is_micro_probe": is_micro_probe,
                "cycle": self._cycle,
                "state": s.__dict__,
            }

        if self._learn_allowed(s, shock):
            self._enter_mode(Mode.LEARN)
            self._prev_P = s.P
            return Mode.LEARN.value, {
                "reason": "priority_7",
                "trigger": "learn_allowed",
                "cycle": self._cycle,
                "state": s.__dict__,
            }

        self._enter_mode(Mode.OBSERVE)
        self._prev_P = s.P
        return Mode.OBSERVE.value, {
            "reason": "fallback",
            "trigger": "gates_failed",
            "cycle": self._cycle,
            "state": s.__dict__,
        }


if __name__ == "__main__":
    cortex = StateVectorCortex()
    mode, debug_info = cortex.eval_irpota()
    print(json.dumps({"mode": mode, "debug": debug_info}, ensure_ascii=False, indent=2))
