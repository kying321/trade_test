#!/usr/bin/env python3
"""Cybernetic control-kernel primitives for Pi.

This module is intentionally strategy-agnostic. It implements only safety/control
infrastructure:
- Actuator dual-bus routing (normal/reflex reduce-only)
- Certificate objects (C0/C1/C2/C3/Cmu)
- Doomsday shock vector (monotonic non-decreasing persistence)
- Hard invariant probes (A/I/O/T/R)
- Append-only audit ledger with local ring-buffer fallback
- Probe budget sub-ledger and loss-cap enforcement
- Local knowledge isolation pipeline (quarantine -> sandbox -> canary)
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from lie_root_resolver import resolve_lie_system_root


def _now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _now_iso() -> str:
    return _now_utc().isoformat()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return int(default)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> bool:
    tmp_path: Optional[Path] = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        tmp_path = Path(raw)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp_path, path)
        return True
    except Exception:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        return False


def _load_json_dict(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


def _default_system_root() -> Path:
    return resolve_lie_system_root()


class ActionBus(str, Enum):
    NORMAL = "normal"
    REFLEX = "reflex"


class CertificateLevel(str, Enum):
    C0 = "C0"  # Reflex emergency lane (reduce-only)
    C1 = "C1"  # Normal action permit
    C2 = "C2"  # Simulate-only permit
    C3 = "C3"  # Hard reject
    CMU = "Cmu"  # Micro-probe permit


@dataclass(frozen=True)
class ControlCertificate:
    level: str
    actuator_bus: str
    reduce_only: bool
    action_class: str
    mode: str
    policy: str
    reason: str
    issued_ts: str = field(default_factory=_now_iso)
    ttl_sec: int = 300
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "actuator_bus": self.actuator_bus,
            "reduce_only": bool(self.reduce_only),
            "action_class": self.action_class,
            "mode": self.mode,
            "policy": self.policy,
            "reason": self.reason,
            "issued_ts": self.issued_ts,
            "ttl_sec": int(self.ttl_sec),
            "metadata": dict(self.metadata or {}),
        }


def resolve_actuator_bus(
    *,
    mode: str,
    policy: str,
    reduce_only_candidate: bool,
) -> Tuple[str, bool, str]:
    critical = mode in {"SURVIVAL", "HIBERNATE", "ROLLBACK", "STABILIZE"}
    if critical and reduce_only_candidate:
        return ActionBus.REFLEX.value, True, "critical_reduce_only_allowlist"
    if reduce_only_candidate and policy == "micro_probe":
        return ActionBus.REFLEX.value, True, "micro_probe_reduce_only"
    return ActionBus.NORMAL.value, False, "default"


def issue_certificate(
    *,
    mode: str,
    policy: str,
    action_class: str,
    actuator_bus: str,
    reduce_only: bool,
    reason: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> ControlCertificate:
    ttl_sec = max(10, _safe_int(os.getenv("CORTEX_CERT_TTL_SEC", "300"), 300))
    if policy == "reject":
        level = CertificateLevel.C3.value
    elif policy == "simulate":
        level = CertificateLevel.C2.value
    elif policy == "micro_probe":
        level = CertificateLevel.CMU.value
    elif actuator_bus == ActionBus.REFLEX.value and reduce_only:
        level = CertificateLevel.C0.value
    else:
        level = CertificateLevel.C1.value

    return ControlCertificate(
        level=level,
        actuator_bus=actuator_bus,
        reduce_only=bool(reduce_only),
        action_class=action_class,
        mode=mode,
        policy=policy,
        reason=reason,
        ttl_sec=ttl_sec,
        metadata=dict(metadata or {}),
    )


@dataclass(frozen=True)
class DoomsdayShockVector:
    price_shock_pct: float = 0.30
    gap_shock_pct: float = 0.10
    slippage_multiplier: float = 2.0
    liquidity_haircut: float = 0.60
    fee_spike_pct: float = 0.01
    version: int = 1
    updated_ts: str = field(default_factory=_now_iso)

    def projected_loss(self, notional: float) -> float:
        notional_abs = abs(float(notional))
        shock = max(0.0, float(self.price_shock_pct) + float(self.gap_shock_pct))
        friction = max(1.0, float(self.slippage_multiplier)) * max(
            0.1, min(1.0, float(self.liquidity_haircut))
        )
        fee = max(0.0, float(self.fee_spike_pct))
        return notional_abs * (shock * friction + fee)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "price_shock_pct": float(self.price_shock_pct),
            "gap_shock_pct": float(self.gap_shock_pct),
            "slippage_multiplier": float(self.slippage_multiplier),
            "liquidity_haircut": float(self.liquidity_haircut),
            "fee_spike_pct": float(self.fee_spike_pct),
            "version": int(self.version),
            "updated_ts": self.updated_ts,
        }


class DoomsdayVectorStore:
    """Persistent store for a monotonic non-decreasing shock vector."""

    def __init__(self, path: Optional[Path] = None):
        self.path = path or Path(
            os.getenv(
                "CORTEX_DOOMSDAY_VECTOR_PATH",
                str(_default_system_root() / "output" / "logs" / "cortex_doomsday_vector.json"),
            )
        )

    def _build_candidate(self, current: Dict[str, Any]) -> Tuple[DoomsdayShockVector, Dict[str, Any], bool]:
        fields = {
            "price_shock_pct": ("CORTEX_D_PRICE_SHOCK_PCT", 0.30),
            "gap_shock_pct": ("CORTEX_D_GAP_SHOCK_PCT", 0.10),
            "slippage_multiplier": ("CORTEX_D_SLIPPAGE_MULTIPLIER", 2.0),
            "liquidity_haircut": ("CORTEX_D_LIQUIDITY_HAIRCUT", 0.60),
            "fee_spike_pct": ("CORTEX_D_FEE_SPIKE_PCT", 0.01),
        }
        monotonic_floor_applied = False
        changed = False
        effective: Dict[str, float] = {}
        for key, (env_name, default) in fields.items():
            baseline = _safe_float(current.get(key), default)
            env_raw = os.getenv(env_name)
            if env_raw is None or str(env_raw).strip() == "":
                candidate = baseline
            else:
                candidate = _safe_float(env_raw, baseline)
                if candidate < baseline:
                    monotonic_floor_applied = True
                candidate = max(baseline, candidate)
            if abs(candidate - baseline) > 1e-12:
                changed = True
            effective[key] = candidate

        version = max(1, _safe_int(current.get("version"), 1))
        if changed:
            version += 1
        vector = DoomsdayShockVector(
            price_shock_pct=effective["price_shock_pct"],
            gap_shock_pct=effective["gap_shock_pct"],
            slippage_multiplier=effective["slippage_multiplier"],
            liquidity_haircut=effective["liquidity_haircut"],
            fee_spike_pct=effective["fee_spike_pct"],
            version=version,
            updated_ts=_now_iso(),
        )
        meta = {
            "path": str(self.path),
            "changed": changed,
            "monotonic_floor_applied": monotonic_floor_applied,
        }
        return vector, meta, changed

    def load(self) -> Tuple[DoomsdayShockVector, Dict[str, Any]]:
        current = _load_json_dict(self.path) if self.path.exists() else {}
        vector, meta, changed = self._build_candidate(current)
        if (not self.path.exists()) or changed:
            _atomic_write_json(self.path, vector.as_dict())
        return vector, meta


def evaluate_doomsday_guard(
    *,
    notional: float,
    vector: DoomsdayShockVector,
    certificate: Optional[ControlCertificate] = None,
) -> Dict[str, Any]:
    max_loss = max(0.0, _safe_float(os.getenv("CORTEX_DOOMSDAY_MAX_LOSS_USDT", "2.0"), 2.0))
    projected_loss = float(vector.projected_loss(notional))
    hard_block = bool(
        projected_loss > max_loss
        and (certificate is None or certificate.level in {CertificateLevel.C0.value, CertificateLevel.C1.value})
    )
    return {
        "notional": abs(float(notional)),
        "projected_loss": projected_loss,
        "max_loss_usdt": max_loss,
        "hard_block": hard_block,
        "vector": vector.as_dict(),
    }


def probe_hard_invariants(state: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "status": "pass",
        "passed": True,
        "failed_metrics": [],
        "warn_metrics": [],
        "metrics": {},
    }

    min_i = _safe_float(os.getenv("CORTEX_PROBE_MIN_I", "0.60"), 0.60)
    min_o = _safe_float(os.getenv("CORTEX_PROBE_MIN_O", "0.25"), 0.25)
    min_r = _safe_float(os.getenv("CORTEX_PROBE_MIN_R", "0.25"), 0.25)
    max_t = _safe_int(os.getenv("CORTEX_PROBE_MAX_T", "1"), 1)

    def _metric_fail(metric: str, value: Any, threshold: Any, check: str, status: str) -> None:
        out["metrics"][metric] = {
            "value": value,
            "threshold": threshold,
            "check": check,
            "status": status,
        }
        if status == "fail":
            out["failed_metrics"].append(metric)
        elif status == "warn":
            out["warn_metrics"].append(metric)

    a_val = state.get("A")
    if a_val is None:
        _metric_fail("A", None, True, "eq", "warn")
    else:
        _metric_fail("A", bool(a_val), True, "eq", "pass" if bool(a_val) else "fail")

    i_val = state.get("I")
    if i_val is None:
        _metric_fail("I", None, min_i, "ge", "warn")
    else:
        i_float = _safe_float(i_val, 0.0)
        _metric_fail("I", i_float, min_i, "ge", "pass" if i_float >= min_i else "fail")

    o_val = state.get("O")
    if o_val is None:
        _metric_fail("O", None, min_o, "ge", "warn")
    else:
        o_float = _safe_float(o_val, 0.0)
        _metric_fail("O", o_float, min_o, "ge", "pass" if o_float >= min_o else "fail")

    t_val = state.get("T")
    if t_val is None:
        _metric_fail("T", None, max_t, "le", "warn")
    else:
        t_int = _safe_int(t_val, 3)
        _metric_fail("T", t_int, max_t, "le", "pass" if t_int <= max_t else "fail")

    r_val = state.get("R")
    if r_val is None:
        _metric_fail("R", None, min_r, "ge", "warn")
    else:
        r_float = _safe_float(r_val, 0.0)
        _metric_fail("R", r_float, min_r, "ge", "pass" if r_float >= min_r else "fail")

    failed = out["failed_metrics"]
    warned = out["warn_metrics"]
    if failed:
        out["status"] = "fail"
        out["passed"] = False
    elif warned:
        out["status"] = "warn"
        out["passed"] = True
    return out


class AppendOnlyAuditLedger:
    """Append-only JSONL ledger with local ring-buffer fallback."""

    def __init__(
        self,
        *,
        path: Path,
        ring_path: Optional[Path] = None,
        ring_limit: int = 512,
    ):
        self.path = path
        self.ring_path = ring_path or path.with_name(f"{path.stem}.ring.jsonl")
        self.ring_limit = max(16, int(ring_limit))

    def _append_jsonl(self, path: Path, payload: Dict[str, Any]) -> bool:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            return True
        except Exception:
            return False

    def _append_ring(self, payload: Dict[str, Any]) -> bool:
        lines = []
        if self.ring_path.exists():
            try:
                lines = self.ring_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            except Exception:
                lines = []
        lines.append(json.dumps(payload, ensure_ascii=False))
        lines = lines[-self.ring_limit :]
        text = "\n".join(lines) + ("\n" if lines else "")
        try:
            self.ring_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.ring_path.with_suffix(self.ring_path.suffix + ".tmp")
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, self.ring_path)
            return True
        except Exception:
            return False

    def append(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        primary_ok = self._append_jsonl(self.path, payload)
        if primary_ok:
            return {
                "primary_ok": True,
                "fallback_used": False,
                "ring_ok": True,
                "path": str(self.path),
                "ring_path": str(self.ring_path),
            }
        ring_payload = {
            "ts": _now_iso(),
            "domain": "audit_ledger_ring_fallback",
            "primary_path": str(self.path),
            "payload": payload,
        }
        ring_ok = self._append_ring(ring_payload)
        return {
            "primary_ok": False,
            "fallback_used": True,
            "ring_ok": bool(ring_ok),
            "path": str(self.path),
            "ring_path": str(self.ring_path),
        }


class ProbeBudgetGuard:
    """Probe sub-ledger and loss-cap enforcement for Cmu operations."""

    def __init__(self, path: Optional[Path] = None):
        self.path = path or Path(
            os.getenv(
                "CORTEX_PROBE_BUDGET_STATE_PATH",
                str(_default_system_root() / "output" / "state" / "cortex_probe_budget.json"),
            )
        )

    def _default_state(self) -> Dict[str, Any]:
        return {
            "date": _now_utc().date().isoformat(),
            "probe_notional_used": 0.0,
            "probe_projected_loss_used": 0.0,
            "halted": False,
            "last_updated_ts": _now_iso(),
        }

    def _load(self) -> Dict[str, Any]:
        state = _load_json_dict(self.path) if self.path.exists() else {}
        if not state:
            state = self._default_state()
        if str(state.get("date") or "") != _now_utc().date().isoformat():
            state = self._default_state()
        return state

    def inspect(
        self,
        *,
        notional: float,
        projected_loss: float,
        certificate: Optional[ControlCertificate],
    ) -> Dict[str, Any]:
        state = self._load()
        enforce = str(os.getenv("CORTEX_PROBE_BUDGET_ENFORCE", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        notional_cap = max(0.0, _safe_float(os.getenv("CORTEX_PROBE_NOTIONAL_CAP_USDT", "25"), 25.0))
        loss_cap = max(0.0, _safe_float(os.getenv("CORTEX_PROBE_LOSS_CAP_USDT", "2"), 2.0))

        cert_level = (certificate.level if certificate is not None else "").strip()
        cmu = cert_level == CertificateLevel.CMU.value
        if not cmu:
            return {
                "applicable": False,
                "enforce": enforce,
                "allowed": True,
                "reason": "non_probe_certificate",
                "state_path": str(self.path),
                "state": state,
            }

        used_notional = _safe_float(state.get("probe_notional_used"), 0.0)
        used_loss = _safe_float(state.get("probe_projected_loss_used"), 0.0)
        halted = bool(state.get("halted", False))
        next_notional = used_notional + abs(float(notional))
        next_loss = used_loss + max(0.0, float(projected_loss))
        blocked = bool(halted or next_notional > notional_cap or next_loss > loss_cap)
        reason = "ok"
        if halted:
            reason = "probe_budget_halted"
        elif next_loss > loss_cap:
            reason = "probe_loss_cap_exceeded"
        elif next_notional > notional_cap:
            reason = "probe_notional_cap_exceeded"

        if not blocked:
            state["probe_notional_used"] = round(next_notional, 8)
            state["probe_projected_loss_used"] = round(next_loss, 8)
        elif enforce:
            state["halted"] = True
        state["last_updated_ts"] = _now_iso()
        _atomic_write_json(self.path, state)

        return {
            "applicable": True,
            "enforce": enforce,
            "allowed": (not blocked) or (not enforce),
            "would_block": blocked,
            "reason": reason,
            "state_path": str(self.path),
            "caps": {"notional_cap_usdt": notional_cap, "loss_cap_usdt": loss_cap},
            "next": {"notional_used": next_notional, "projected_loss_used": next_loss},
            "state": state,
        }


class KnowledgeIsolationPipeline:
    """Local-only external knowledge isolation flow."""

    def __init__(self, root: Optional[Path] = None):
        workspace = Path(__file__).resolve().parents[1]
        self.root = root or Path(
            os.getenv(
                "CORTEX_KNOWLEDGE_PIPELINE_ROOT",
                str(workspace / "tmp" / "knowledge_pipeline"),
            )
        )
        self.quarantine_dir = self.root / "quarantine"
        self.sandbox_dir = self.root / "sandbox"
        self.canary_dir = self.root / "canary"
        self.index_path = self.root / "knowledge_index.json"
        self.ledger = AppendOnlyAuditLedger(
            path=self.root / "knowledge_pipeline_events.jsonl",
            ring_path=self.root / "knowledge_pipeline_events.ring.jsonl",
            ring_limit=256,
        )

    def _ensure_dirs(self) -> None:
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        self.canary_dir.mkdir(parents=True, exist_ok=True)

    def _load_index(self) -> Dict[str, Any]:
        return _load_json_dict(self.index_path) if self.index_path.exists() else {}

    def _save_index(self, index: Dict[str, Any]) -> None:
        _atomic_write_json(self.index_path, index)

    def _record(self, event: Dict[str, Any]) -> None:
        payload = dict(event)
        payload.setdefault("ts", _now_iso())
        payload.setdefault("domain", "knowledge_pipeline")
        self.ledger.append(payload)

    def ingest(self, source_path: Path, *, source: str = "external") -> Dict[str, Any]:
        self._ensure_dirs()
        if not source_path.exists() or (not source_path.is_file()):
            out = {
                "ok": False,
                "stage": "quarantine",
                "reason": "source_missing",
                "source_path": str(source_path),
            }
            self._record(out)
            return out

        raw = source_path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()[:16]
        item_id = f"{_now_utc().strftime('%Y%m%d%H%M%S')}_{digest}"
        dst = self.quarantine_dir / f"{item_id}_{source_path.name}"
        shutil.copy2(source_path, dst)

        index = self._load_index()
        entry = {
            "item_id": item_id,
            "source": source,
            "source_path": str(source_path),
            "quarantine_path": str(dst),
            "sandbox_path": "",
            "canary_path": "",
            "sha256_16": digest,
            "stage": "quarantine",
            "created_ts": _now_iso(),
            "updated_ts": _now_iso(),
        }
        index[item_id] = entry
        self._save_index(index)

        out = {
            "ok": True,
            "stage": "quarantine",
            "item_id": item_id,
            "path": str(dst),
        }
        self._record(out)
        return out

    def sandbox_validate(
        self,
        item_id: str,
        *,
        validator: Optional[Callable[[Dict[str, Any], Path], Tuple[bool, str]]] = None,
    ) -> Dict[str, Any]:
        self._ensure_dirs()
        index = self._load_index()
        item = index.get(item_id) if isinstance(index.get(item_id), dict) else None
        if item is None:
            out = {"ok": False, "stage": "sandbox", "reason": "item_not_found", "item_id": item_id}
            self._record(out)
            return out

        src = Path(str(item.get("quarantine_path") or ""))
        if (not src.exists()) or (not src.is_file()):
            out = {
                "ok": False,
                "stage": "sandbox",
                "reason": "quarantine_payload_missing",
                "item_id": item_id,
            }
            self._record(out)
            return out

        max_bytes = max(1024, _safe_int(os.getenv("CORTEX_KNOWLEDGE_MAX_BYTES", "5242880"), 5242880))
        suffix_ok = src.suffix.lower() in {
            ".md",
            ".txt",
            ".json",
            ".yaml",
            ".yml",
            ".csv",
            ".log",
        }
        if (src.stat().st_size <= 0) or (src.stat().st_size > max_bytes) or (not suffix_ok):
            item["stage"] = "sandbox_rejected"
            item["updated_ts"] = _now_iso()
            index[item_id] = item
            self._save_index(index)
            out = {
                "ok": False,
                "stage": "sandbox",
                "reason": "sandbox_basic_validation_failed",
                "item_id": item_id,
            }
            self._record(out)
            return out

        if validator is not None:
            try:
                ok, reason = validator(item, src)
            except Exception as exc:
                ok, reason = False, f"validator_error:{type(exc).__name__}"
            if not ok:
                item["stage"] = "sandbox_rejected"
                item["updated_ts"] = _now_iso()
                index[item_id] = item
                self._save_index(index)
                out = {
                    "ok": False,
                    "stage": "sandbox",
                    "reason": reason or "validator_rejected",
                    "item_id": item_id,
                }
                self._record(out)
                return out

        dst = self.sandbox_dir / src.name
        shutil.copy2(src, dst)
        item["sandbox_path"] = str(dst)
        item["stage"] = "sandbox_validated"
        item["updated_ts"] = _now_iso()
        index[item_id] = item
        self._save_index(index)
        out = {"ok": True, "stage": "sandbox", "reason": "validated", "item_id": item_id, "path": str(dst)}
        self._record(out)
        return out

    def promote_canary(self, item_id: str) -> Dict[str, Any]:
        self._ensure_dirs()
        index = self._load_index()
        item = index.get(item_id) if isinstance(index.get(item_id), dict) else None
        if item is None:
            out = {"ok": False, "stage": "canary", "reason": "item_not_found", "item_id": item_id}
            self._record(out)
            return out
        if str(item.get("stage") or "") != "sandbox_validated":
            out = {
                "ok": False,
                "stage": "canary",
                "reason": "sandbox_not_validated",
                "item_id": item_id,
            }
            self._record(out)
            return out

        src = Path(str(item.get("sandbox_path") or ""))
        if (not src.exists()) or (not src.is_file()):
            out = {
                "ok": False,
                "stage": "canary",
                "reason": "sandbox_payload_missing",
                "item_id": item_id,
            }
            self._record(out)
            return out

        dst = self.canary_dir / src.name
        shutil.copy2(src, dst)
        item["canary_path"] = str(dst)
        item["stage"] = "canary"
        item["updated_ts"] = _now_iso()
        index[item_id] = item
        self._save_index(index)
        out = {"ok": True, "stage": "canary", "reason": "promoted", "item_id": item_id, "path": str(dst)}
        self._record(out)
        return out
