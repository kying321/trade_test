from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from contextlib import closing
import csv
import math
import sqlite3
from typing import Any, Callable, ClassVar

import yaml

from lie_engine.config import SystemSettings
from lie_engine.data.storage import write_json, write_markdown
from lie_engine.models import ReviewDelta
from lie_engine.orchestration.artifact_governance import apply_dated_artifact_governance


@dataclass(slots=True)
class ReleaseOrchestrator:
    settings: SystemSettings
    output_dir: Path
    quality_snapshot: Callable[[date], dict[str, Any]]
    backtest_snapshot: Callable[[date], dict[str, Any]]
    run_review: Callable[[date], ReviewDelta]
    health_check: Callable[[date, bool], dict[str, Any]]
    stable_replay_check: Callable[[date, int | None], dict[str, Any]]
    test_all: Callable[..., dict[str, Any]]
    load_json_safely: Callable[[Path], dict[str, Any]]
    sqlite_path: Path | None = None
    run_stress_matrix: Callable[[date, list[str] | None], dict[str, Any]] | None = None
    ARTIFACT_GOVERNANCE_DEFAULTS: ClassVar[dict[str, dict[str, str]]] = {
        "stress_autorun_history": {
            "json_glob": "*_stress_autorun_history.json",
            "md_glob": "*_stress_autorun_history.md",
            "checksum_index_filename": "stress_autorun_history_checksum_index.json",
        },
        "stress_autorun_reason_drift": {
            "json_glob": "*_stress_autorun_reason_drift.json",
            "md_glob": "*_stress_autorun_reason_drift.md",
            "checksum_index_filename": "stress_autorun_reason_drift_checksum_index.json",
        },
        "temporal_autofix_patch": {
            "json_glob": "*_temporal_autofix_patch.json",
            "md_glob": "*_temporal_autofix_patch.md",
            "checksum_index_filename": "temporal_autofix_patch_checksum_index.json",
        },
        "reconcile_row_diff": {
            "json_glob": "*_reconcile_row_diff.json",
            "md_glob": "*_reconcile_row_diff.md",
            "checksum_index_filename": "reconcile_row_diff_checksum_index.json",
        },
    }
    ARTIFACT_GOVERNANCE_LEGACY_KEYS: ClassVar[dict[str, dict[str, Any]]] = {
        "temporal_autofix_patch": {
            "retention_key": "ops_temporal_audit_autofix_patch_retention_days",
            "retention_default": 30,
            "checksum_key": "ops_temporal_audit_autofix_patch_checksum_index_enabled",
            "checksum_default": True,
        },
        "stress_autorun_history": {
            "retention_key": "ops_stress_autorun_history_retention_days",
            "retention_default": 30,
            "checksum_key": "ops_stress_autorun_history_checksum_index_enabled",
            "checksum_default": True,
        },
        "stress_autorun_reason_drift": {
            "retention_key": "ops_stress_autorun_reason_drift_retention_days",
            "retention_default": 30,
            "checksum_key": "ops_stress_autorun_reason_drift_checksum_index_enabled",
            "checksum_default": True,
        },
        "reconcile_row_diff": {
            "retention_key": "ops_reconcile_broker_row_diff_artifact_retention_days",
            "retention_default": 30,
            "checksum_key": "ops_reconcile_broker_row_diff_artifact_checksum_index_enabled",
            "checksum_default": True,
        },
    }

    @staticmethod
    def _safe_float(v: Any, default: float = 0.0) -> float:
        try:
            return float(v)
        except Exception:
            return float(default)

    @staticmethod
    def _safe_bool(v: Any, default: bool = False) -> bool:
        if isinstance(v, bool):
            return bool(v)
        if isinstance(v, (int, float)):
            return bool(v)
        txt = str(v).strip().lower()
        if txt in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if txt in {"0", "false", "f", "no", "n", "off"}:
            return False
        return bool(default)

    @staticmethod
    def _extract_failed_tests(test_payload: dict[str, Any]) -> list[str]:
        listed = test_payload.get("failed_tests", [])
        if isinstance(listed, list):
            out = [str(x).strip() for x in listed if str(x).strip()]
            if out:
                return out
        stderr = str(test_payload.get("stderr", "") or "")
        if not stderr:
            stderr = str(test_payload.get("stderr_excerpt", "") or "")
        stdout = str(test_payload.get("stdout", "") or "")
        if not stdout:
            stdout = str(test_payload.get("stdout_excerpt", "") or "")
        text = f"{stderr}\n{stdout}"
        failed: list[str] = []
        for line in text.splitlines():
            txt = line.strip()
            if txt.endswith("... FAIL") or txt.endswith("... ERROR"):
                failed.append(txt.split(" ... ")[0].strip())
            elif txt.startswith("FAIL: ") or txt.startswith("ERROR: "):
                failed.append(txt.split(": ", 1)[1].strip())
        out: list[str] = []
        seen: set[str] = set()
        for item in failed:
            if item and item not in seen:
                seen.add(item)
                out.append(item)
        return out

    @classmethod
    def _is_timeout_payload(cls, test_payload: dict[str, Any]) -> bool:
        if bool(test_payload.get("timed_out", False)):
            return True
        try:
            if int(test_payload.get("returncode", 0)) == 124:
                return True
        except Exception:
            pass
        failed = cls._extract_failed_tests(test_payload)
        return "__timeout__" in set(failed)

    def _artifact_governance_profile(
        self,
        *,
        profile_name: str,
        fallback_retention_days: int,
        fallback_checksum_index_enabled: bool,
    ) -> dict[str, Any]:
        defaults = self.ARTIFACT_GOVERNANCE_DEFAULTS.get(str(profile_name), {})
        default_json_glob = str(defaults.get("json_glob", "")).strip() or f"*_{profile_name}.json"
        default_md_glob = str(defaults.get("md_glob", "")).strip() or f"*_{profile_name}.md"
        default_index_filename = (
            str(defaults.get("checksum_index_filename", "")).strip() or f"{profile_name}_checksum_index.json"
        )
        val = self.settings.validation if isinstance(self.settings.validation, dict) else {}
        raw_profiles = (
            val.get("ops_artifact_governance_profiles", {})
            if isinstance(val.get("ops_artifact_governance_profiles", {}), dict)
            else {}
        )
        profile_raw = (
            raw_profiles.get(profile_name, {})
            if isinstance(raw_profiles.get(profile_name, {}), dict)
            else {}
        )
        json_glob = str(profile_raw.get("json_glob", default_json_glob)).strip() or default_json_glob
        md_glob = str(profile_raw.get("md_glob", default_md_glob)).strip() or default_md_glob
        checksum_index_filename = (
            str(profile_raw.get("checksum_index_filename", default_index_filename)).strip()
            or default_index_filename
        )
        retention_days = max(
            1,
            int(
                self._safe_float(
                    profile_raw.get("retention_days", fallback_retention_days),
                    fallback_retention_days,
                )
            ),
        )
        checksum_index_enabled = self._safe_bool(
            profile_raw.get("checksum_index_enabled", fallback_checksum_index_enabled),
            bool(fallback_checksum_index_enabled),
        )
        return {
            "profile_name": str(profile_name),
            "json_glob": str(json_glob),
            "md_glob": str(md_glob),
            "checksum_index_filename": str(checksum_index_filename),
            "retention_days": int(retention_days),
            "checksum_index_enabled": bool(checksum_index_enabled),
            "profile_override": bool(profile_name in raw_profiles),
        }

    def _apply_artifact_governance(
        self,
        *,
        as_of: date,
        review_dir: Path,
        profile_name: str,
        fallback_retention_days: int,
        fallback_checksum_index_enabled: bool,
    ) -> dict[str, Any]:
        policy = self._artifact_governance_profile(
            profile_name=profile_name,
            fallback_retention_days=fallback_retention_days,
            fallback_checksum_index_enabled=fallback_checksum_index_enabled,
        )
        result = apply_dated_artifact_governance(
            as_of=as_of,
            directory=review_dir,
            json_glob=str(policy.get("json_glob", "")),
            md_glob=str(policy.get("md_glob", "")),
            retention_days=int(policy.get("retention_days", fallback_retention_days)),
            checksum_index_enabled=bool(policy.get("checksum_index_enabled", fallback_checksum_index_enabled)),
            checksum_index_filename=str(policy.get("checksum_index_filename", "")),
        )
        result["profile_name"] = str(profile_name)
        result["profile_override"] = bool(policy.get("profile_override", False))
        result["json_glob"] = str(policy.get("json_glob", ""))
        result["md_glob"] = str(policy.get("md_glob", ""))
        result["checksum_index_filename"] = str(policy.get("checksum_index_filename", ""))
        return result

    def _artifact_governance_legacy_defaults(self, *, profile_name: str) -> dict[str, Any]:
        val = self.settings.validation if isinstance(self.settings.validation, dict) else {}
        mapping = self.ARTIFACT_GOVERNANCE_LEGACY_KEYS.get(str(profile_name), {})
        retention_key = str(mapping.get("retention_key", "")).strip()
        retention_default = int(self._safe_float(mapping.get("retention_default", 30), 30))
        checksum_key = str(mapping.get("checksum_key", "")).strip()
        checksum_default = self._safe_bool(mapping.get("checksum_default", True), True)
        retention_days = max(
            1,
            int(
                self._safe_float(
                    val.get(retention_key, retention_default) if retention_key else retention_default,
                    retention_default,
                )
            ),
        )
        checksum_index_enabled = self._safe_bool(
            val.get(checksum_key, checksum_default) if checksum_key else checksum_default,
            checksum_default,
        )
        return {
            "retention_days": int(retention_days),
            "checksum_index_enabled": bool(checksum_index_enabled),
        }

    def _artifact_governance_metrics(
        self,
        *,
        as_of: date,
        temporal_audit: dict[str, Any],
        stress_autorun_history: dict[str, Any],
        stress_autorun_reason_drift: dict[str, Any],
        reconcile_drift: dict[str, Any],
    ) -> dict[str, Any]:
        val = self.settings.validation if isinstance(self.settings.validation, dict) else {}
        strict_mode_enabled = self._safe_bool(val.get("ops_artifact_governance_strict_mode_enabled", False), False)
        baseline_profiles = (
            val.get("ops_artifact_governance_profile_baseline", {})
            if isinstance(val.get("ops_artifact_governance_profile_baseline", {}), dict)
            else {}
        )
        artifacts_temporal = (
            temporal_audit.get("artifacts", {}) if isinstance(temporal_audit.get("artifacts", {}), dict) else {}
        )
        artifact_temporal_autofix = (
            artifacts_temporal.get("autofix_patch", {})
            if isinstance(artifacts_temporal.get("autofix_patch", {}), dict)
            else {}
        )
        artifacts_history = (
            stress_autorun_history.get("artifacts", {})
            if isinstance(stress_autorun_history.get("artifacts", {}), dict)
            else {}
        )
        artifact_stress_history = (
            artifacts_history.get("history", {})
            if isinstance(artifacts_history.get("history", {}), dict)
            else {}
        )
        artifacts_reason = (
            stress_autorun_reason_drift.get("artifacts", {})
            if isinstance(stress_autorun_reason_drift.get("artifacts", {}), dict)
            else {}
        )
        artifact_stress_reason = (
            artifacts_reason.get("reason_drift", {})
            if isinstance(artifacts_reason.get("reason_drift", {}), dict)
            else {}
        )
        artifacts_reconcile = (
            reconcile_drift.get("artifacts", {}) if isinstance(reconcile_drift.get("artifacts", {}), dict) else {}
        )
        artifact_reconcile_row_diff = (
            artifacts_reconcile.get("row_diff", {})
            if isinstance(artifacts_reconcile.get("row_diff", {}), dict)
            else {}
        )

        profile_inputs: list[dict[str, Any]] = [
            {
                "profile": "temporal_autofix_patch",
                "active": bool(temporal_audit.get("active", False)),
                "artifact": artifact_temporal_autofix,
            },
            {
                "profile": "stress_autorun_history",
                "active": bool(stress_autorun_history.get("active", False)),
                "artifact": artifact_stress_history,
            },
            {
                "profile": "stress_autorun_reason_drift",
                "active": bool(stress_autorun_reason_drift.get("active", False)),
                "artifact": artifact_stress_reason,
            },
            {
                "profile": "reconcile_row_diff",
                "active": bool(reconcile_drift.get("active", False)),
                "artifact": artifact_reconcile_row_diff,
            },
        ]

        profile_rows: list[dict[str, Any]] = []
        required_missing = 0
        policy_mismatch = 0
        legacy_drift = 0
        baseline_defined = 0
        baseline_drift = 0
        alerts: list[str] = []

        for row in profile_inputs:
            profile_name = str(row.get("profile", "")).strip()
            active = bool(row.get("active", False))
            artifact = row.get("artifact", {}) if isinstance(row.get("artifact", {}), dict) else {}
            legacy = self._artifact_governance_legacy_defaults(profile_name=profile_name)
            policy = self._artifact_governance_profile(
                profile_name=profile_name,
                fallback_retention_days=int(legacy.get("retention_days", 30)),
                fallback_checksum_index_enabled=bool(legacy.get("checksum_index_enabled", True)),
            )

            artifact_written = bool(artifact.get("written", False))
            artifact_retention_days = int(self._safe_float(artifact.get("retention_days", 0), 0))
            artifact_checksum_enabled = bool(artifact.get("checksum_index_enabled", False))
            artifact_index_path = str(artifact.get("checksum_index_path", "")).strip()
            expected_index_filename = str(policy.get("checksum_index_filename", "")).strip()

            has_artifact_payload = bool(artifact)
            has_any_artifact_trace = bool(
                artifact_written
                or str(artifact.get("json", "")).strip()
                or str(artifact.get("md", "")).strip()
                or artifact_index_path
            )
            required_profile_missing = bool(active and (not has_artifact_payload))
            if required_profile_missing:
                required_missing += 1

            retention_mismatch = bool(has_any_artifact_trace and artifact_retention_days != int(policy.get("retention_days", 0)))
            checksum_switch_mismatch = bool(
                has_any_artifact_trace and artifact_checksum_enabled != bool(policy.get("checksum_index_enabled", False))
            )
            index_filename_mismatch = False
            if artifact_index_path and expected_index_filename:
                index_filename_mismatch = bool(Path(artifact_index_path).name != expected_index_filename)

            this_policy_mismatch = bool(
                retention_mismatch or checksum_switch_mismatch or index_filename_mismatch
            )
            if this_policy_mismatch:
                policy_mismatch += 1

            legacy_retention_days = int(legacy.get("retention_days", 30))
            legacy_checksum_enabled = bool(legacy.get("checksum_index_enabled", True))
            this_legacy_drift = bool(
                int(policy.get("retention_days", legacy_retention_days)) != legacy_retention_days
                or bool(policy.get("checksum_index_enabled", legacy_checksum_enabled)) != legacy_checksum_enabled
            )
            if this_legacy_drift:
                legacy_drift += 1

            baseline_raw = (
                baseline_profiles.get(profile_name, {})
                if isinstance(baseline_profiles.get(profile_name, {}), dict)
                else {}
            )
            baseline_compare_keys = {
                str(k)
                for k in baseline_raw.keys()
                if str(k)
                in {"json_glob", "md_glob", "checksum_index_filename", "retention_days", "checksum_index_enabled"}
            }
            baseline_drift_fields: list[str] = []
            if baseline_compare_keys:
                baseline_defined += 1
                if "json_glob" in baseline_compare_keys:
                    baseline_json_glob = str(baseline_raw.get("json_glob", policy.get("json_glob", ""))).strip()
                    if baseline_json_glob != str(policy.get("json_glob", "")).strip():
                        baseline_drift_fields.append("json_glob")
                if "md_glob" in baseline_compare_keys:
                    baseline_md_glob = str(baseline_raw.get("md_glob", policy.get("md_glob", ""))).strip()
                    if baseline_md_glob != str(policy.get("md_glob", "")).strip():
                        baseline_drift_fields.append("md_glob")
                if "checksum_index_filename" in baseline_compare_keys:
                    baseline_index_name = str(
                        baseline_raw.get("checksum_index_filename", policy.get("checksum_index_filename", ""))
                    ).strip()
                    if baseline_index_name != str(policy.get("checksum_index_filename", "")).strip():
                        baseline_drift_fields.append("checksum_index_filename")
                if "retention_days" in baseline_compare_keys:
                    baseline_retention_days = max(
                        1,
                        int(
                            self._safe_float(
                                baseline_raw.get("retention_days", policy.get("retention_days", 0)),
                                policy.get("retention_days", 0),
                            )
                        ),
                    )
                    if baseline_retention_days != int(policy.get("retention_days", 0)):
                        baseline_drift_fields.append("retention_days")
                if "checksum_index_enabled" in baseline_compare_keys:
                    baseline_checksum_enabled = self._safe_bool(
                        baseline_raw.get("checksum_index_enabled", policy.get("checksum_index_enabled", False)),
                        bool(policy.get("checksum_index_enabled", False)),
                    )
                    if baseline_checksum_enabled != bool(policy.get("checksum_index_enabled", False)):
                        baseline_drift_fields.append("checksum_index_enabled")

            this_baseline_drift = bool(len(baseline_drift_fields) > 0)
            if this_baseline_drift:
                baseline_drift += 1

            profile_rows.append(
                {
                    "profile": profile_name,
                    "active": bool(active),
                    "profile_override": bool(policy.get("profile_override", False)),
                    "legacy_retention_days": int(legacy_retention_days),
                    "legacy_checksum_index_enabled": bool(legacy_checksum_enabled),
                    "policy_retention_days": int(policy.get("retention_days", legacy_retention_days)),
                    "policy_checksum_index_enabled": bool(
                        policy.get("checksum_index_enabled", legacy_checksum_enabled)
                    ),
                    "policy_json_glob": str(policy.get("json_glob", "")),
                    "policy_md_glob": str(policy.get("md_glob", "")),
                    "policy_checksum_index_filename": str(expected_index_filename),
                    "artifact_present": bool(has_artifact_payload),
                    "artifact_written": bool(artifact_written),
                    "artifact_retention_days": int(artifact_retention_days),
                    "artifact_checksum_index_enabled": bool(artifact_checksum_enabled),
                    "artifact_checksum_index_path": str(artifact_index_path),
                    "required_profile_missing": bool(required_profile_missing),
                    "retention_mismatch": bool(retention_mismatch),
                    "checksum_switch_mismatch": bool(checksum_switch_mismatch),
                    "index_filename_mismatch": bool(index_filename_mismatch),
                    "policy_mismatch": bool(this_policy_mismatch),
                    "legacy_policy_drift": bool(this_legacy_drift),
                    "baseline_defined": bool(len(baseline_compare_keys) > 0),
                    "baseline_compare_keys": sorted(baseline_compare_keys),
                    "baseline_policy_drift": bool(this_baseline_drift),
                    "baseline_drift_fields": baseline_drift_fields,
                }
            )

        if required_missing > 0:
            alerts.append("artifact_governance_profile_missing")
        if policy_mismatch > 0:
            alerts.append("artifact_governance_policy_mismatch")
        if legacy_drift > 0:
            alerts.append("artifact_governance_legacy_policy_drift")
        if baseline_drift > 0:
            alerts.append("artifact_governance_baseline_drift")
        strict_mode_blocked = bool(
            strict_mode_enabled
            and (required_missing > 0 or policy_mismatch > 0 or legacy_drift > 0 or baseline_drift > 0)
        )
        if strict_mode_blocked:
            alerts.append("artifact_governance_strict_mode_blocked")

        checks = {
            "required_profiles_present_ok": bool(required_missing == 0),
            "policy_alignment_ok": bool(policy_mismatch == 0),
            "legacy_alignment_ok": bool((not strict_mode_enabled) or (legacy_drift == 0)),
            "baseline_freeze_ok": bool((not strict_mode_enabled) or (baseline_drift == 0)),
            "strict_mode_ok": bool((not strict_mode_enabled) or (not strict_mode_blocked)),
        }

        return {
            "active": True,
            "as_of": as_of.isoformat(),
            "checks": checks,
            "metrics": {
                "profiles_total": int(len(profile_rows)),
                "profiles_active": int(sum(1 for x in profile_rows if bool(x.get("active", False)))),
                "profiles_with_override": int(sum(1 for x in profile_rows if bool(x.get("profile_override", False)))),
                "required_missing_profiles": int(required_missing),
                "policy_mismatch_profiles": int(policy_mismatch),
                "legacy_policy_drift_profiles": int(legacy_drift),
                "baseline_defined_profiles": int(baseline_defined),
                "baseline_drift_profiles": int(baseline_drift),
                "strict_mode_enabled": bool(strict_mode_enabled),
                "strict_mode_blocked": bool(strict_mode_blocked),
            },
            "alerts": alerts,
            "profiles": profile_rows,
        }

    def gate_report(
        self,
        as_of: date,
        run_tests: bool = False,
        run_review_if_missing: bool = True,
    ) -> dict[str, Any]:
        d = as_of.isoformat()
        review_delta_path = self.output_dir / "review" / f"{d}_param_delta.yaml"
        if run_review_if_missing and not review_delta_path.exists():
            self.run_review(as_of)

        quality = self.quality_snapshot(as_of)
        backtest = self.backtest_snapshot(as_of)
        health = self.health_check(as_of, True)
        replay = self.stable_replay_check(as_of, None)
        state_stability = self._state_stability_metrics(as_of=as_of)
        state_active = bool(state_stability.get("active", False))
        state_checks = state_stability.get("checks", {}) if isinstance(state_stability.get("checks", {}), dict) else {}
        state_stability_ok = all(bool(v) for v in state_checks.values()) if state_active else True
        temporal_audit = self._temporal_audit_metrics(as_of=as_of)
        temporal_active = bool(temporal_audit.get("active", False))
        temporal_checks = temporal_audit.get("checks", {}) if isinstance(temporal_audit.get("checks", {}), dict) else {}
        temporal_audit_ok = all(bool(v) for v in temporal_checks.values()) if temporal_active else True
        slot_anomaly = self._slot_anomaly_metrics(as_of=as_of)
        slot_active = bool(slot_anomaly.get("active", False))
        slot_checks = slot_anomaly.get("checks", {}) if isinstance(slot_anomaly.get("checks", {}), dict) else {}
        slot_anomaly_ok = all(bool(v) for v in slot_checks.values()) if slot_active else True
        mode_drift = self._mode_drift_metrics(as_of=as_of)
        drift_active = bool(mode_drift.get("active", False))
        drift_checks = mode_drift.get("checks", {}) if isinstance(mode_drift.get("checks", {}), dict) else {}
        mode_drift_ok = all(bool(v) for v in drift_checks.values()) if drift_active else True
        stress_matrix_trend = self._stress_matrix_trend_metrics(as_of=as_of)
        stress_active = bool(stress_matrix_trend.get("active", False))
        stress_checks = (
            stress_matrix_trend.get("checks", {})
            if isinstance(stress_matrix_trend.get("checks", {}), dict)
            else {}
        )
        stress_matrix_trend_ok = all(bool(v) for v in stress_checks.values()) if stress_active else True
        stress_autorun_history = self._stress_autorun_history_metrics(as_of=as_of)
        stress_autorun_history_active = bool(stress_autorun_history.get("active", False))
        stress_autorun_history_checks = (
            stress_autorun_history.get("checks", {})
            if isinstance(stress_autorun_history.get("checks", {}), dict)
            else {}
        )
        stress_autorun_history_ok = (
            all(bool(v) for v in stress_autorun_history_checks.values()) if stress_autorun_history_active else True
        )
        stress_autorun_adaptive = self._stress_autorun_adaptive_saturation_metrics(as_of=as_of)
        stress_autorun_adaptive_active = bool(stress_autorun_adaptive.get("active", False))
        stress_autorun_adaptive_checks = (
            stress_autorun_adaptive.get("checks", {})
            if isinstance(stress_autorun_adaptive.get("checks", {}), dict)
            else {}
        )
        stress_autorun_adaptive_ok = (
            all(bool(v) for v in stress_autorun_adaptive_checks.values()) if stress_autorun_adaptive_active else True
        )
        stress_autorun_reason_drift = self._stress_autorun_adaptive_reason_drift_metrics(as_of=as_of)
        stress_autorun_reason_drift_active = bool(stress_autorun_reason_drift.get("active", False))
        stress_autorun_reason_drift_checks = (
            stress_autorun_reason_drift.get("checks", {})
            if isinstance(stress_autorun_reason_drift.get("checks", {}), dict)
            else {}
        )
        stress_autorun_reason_drift_ok = (
            all(bool(v) for v in stress_autorun_reason_drift_checks.values())
            if stress_autorun_reason_drift_active
            else True
        )
        reconcile_drift = self._reconcile_drift_metrics(as_of=as_of)
        reconcile_active = bool(reconcile_drift.get("active", False))
        reconcile_checks = reconcile_drift.get("checks", {}) if isinstance(reconcile_drift.get("checks", {}), dict) else {}
        reconcile_drift_ok = all(bool(v) for v in reconcile_checks.values()) if reconcile_active else True
        artifact_governance = self._artifact_governance_metrics(
            as_of=as_of,
            temporal_audit=temporal_audit,
            stress_autorun_history=stress_autorun_history,
            stress_autorun_reason_drift=stress_autorun_reason_drift,
            reconcile_drift=reconcile_drift,
        )
        artifact_governance_active = bool(artifact_governance.get("active", False))
        artifact_governance_checks = (
            artifact_governance.get("checks", {})
            if isinstance(artifact_governance.get("checks", {}), dict)
            else {}
        )
        artifact_governance_ok = (
            all(bool(v) for v in artifact_governance_checks.values()) if artifact_governance_active else True
        )

        tests_ok = True
        tests_payload: dict[str, Any] = {}
        if run_tests:
            tests_payload = self.test_all()
            tests_ok = bool(tests_payload.get("returncode", 1) == 0)

        review_pass = False
        mode_health_ok = True
        if review_delta_path.exists():
            try:
                review_delta = yaml.safe_load(review_delta_path.read_text(encoding="utf-8")) or {}
            except Exception:
                review_delta = {}
            review_pass = bool(review_delta.get("pass_gate", False))
            mode_health = review_delta.get("mode_health", {}) if isinstance(review_delta.get("mode_health", {}), dict) else {}
            mode_health_ok = bool(mode_health.get("passed", True))

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
            "mode_health_ok": mode_health_ok,
            "state_stability_ok": state_stability_ok,
            "temporal_audit_ok": temporal_audit_ok,
            "slot_anomaly_ok": slot_anomaly_ok,
            "mode_drift_ok": mode_drift_ok,
            "stress_matrix_trend_ok": stress_matrix_trend_ok,
            "stress_autorun_history_ok": stress_autorun_history_ok,
            "stress_autorun_adaptive_ok": stress_autorun_adaptive_ok,
            "stress_autorun_reason_drift_ok": stress_autorun_reason_drift_ok,
            "reconcile_drift_ok": reconcile_drift_ok,
            "artifact_governance_ok": artifact_governance_ok,
            "tests_ok": tests_ok,
            "health_ok": health_ok,
            "stable_replay_ok": replay_ok,
            "data_completeness_ok": completeness_ok,
            "unresolved_conflict_ok": unresolved_ok,
            "positive_window_ratio_ok": positive_ok,
            "max_drawdown_ok": drawdown_ok,
            "risk_violations_ok": violations_ok,
        }
        rollback_recommendation = self._rollback_recommendation(
            as_of=as_of,
            checks=checks,
            state_stability=state_stability,
            temporal_audit=temporal_audit,
            slot_anomaly=slot_anomaly,
            mode_drift=mode_drift,
            reconcile_drift=reconcile_drift,
        )
        checks["rollback_anchor_ready"] = bool(rollback_recommendation.get("anchor_ready", True))
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
            "state_stability": state_stability,
            "temporal_audit": temporal_audit,
            "slot_anomaly": slot_anomaly,
            "mode_drift": mode_drift,
            "stress_matrix_trend": stress_matrix_trend,
            "stress_autorun_history": stress_autorun_history,
            "stress_autorun_adaptive": stress_autorun_adaptive,
            "stress_autorun_reason_drift": stress_autorun_reason_drift,
            "reconcile_drift": reconcile_drift,
            "artifact_governance": artifact_governance,
            "rollback_recommendation": rollback_recommendation,
            "tests": tests_payload if run_tests else {"skipped": True},
        }

        if overall:
            alert_path = self.output_dir / "logs" / f"review_loop_alert_{d}.json"
            if alert_path.exists():
                try:
                    alert_path.unlink()
                except OSError:
                    pass

        report_path = self.output_dir / "review" / f"{d}_gate_report.json"
        write_json(report_path, out)
        return out

    def _run_tests(
        self,
        *,
        fast: bool,
        fast_ratio: float,
        fast_shard_index: int,
        fast_shard_total: int,
        fast_seed: str,
    ) -> dict[str, Any]:
        if not fast:
            return self.test_all()
        try:
            return self.test_all(
                fast=True,
                fast_ratio=float(fast_ratio),
                fast_shard_index=int(fast_shard_index),
                fast_shard_total=int(fast_shard_total),
                fast_seed=str(fast_seed),
            )
        except TypeError:
            # Backward compatibility for legacy callables that don't accept kwargs.
            return self.test_all()

    def _latest_test_result(self) -> dict[str, Any]:
        logs_dir = self.output_dir / "logs"
        candidates = sorted(logs_dir.glob("tests_*.json"))
        if not candidates:
            return {"found": False}
        latest = candidates[-1]
        payload = self.load_json_safely(latest)
        return {
            "found": True,
            "path": str(latest),
            "returncode": payload.get("returncode"),
            "has_output": bool(payload.get("stdout") or payload.get("stderr") or payload.get("stdout_excerpt") or payload.get("stderr_excerpt")),
        }

    def _load_review_loop_history_series(self, *, as_of: date, window_days: int) -> list[dict[str, Any]]:
        wd = max(1, int(window_days))
        out: list[dict[str, Any]] = []
        artifacts_dir = self.output_dir / "artifacts"
        logs_dir = self.output_dir / "logs"

        for i in range(wd):
            day = as_of - timedelta(days=i)
            dstr = day.isoformat()
            release_payload = self.load_json_safely(artifacts_dir / f"release_ready_{dstr}.json")
            alert_payload = self.load_json_safely(logs_dir / f"review_loop_alert_{dstr}.json")
            payload: dict[str, Any] = {}
            source = ""
            if isinstance(release_payload, dict) and release_payload:
                payload = release_payload
                source = "release_ready"
            elif isinstance(alert_payload, dict) and alert_payload:
                payload = alert_payload
                source = "review_loop_alert"
            if (not payload) or (not source):
                continue

            rounds = payload.get("rounds", []) if isinstance(payload.get("rounds", []), list) else []
            for idx, round_item in enumerate(rounds):
                if not isinstance(round_item, dict):
                    continue
                auto = (
                    round_item.get("stress_matrix_autorun", {})
                    if isinstance(round_item.get("stress_matrix_autorun", {}), dict)
                    else {}
                )
                if not auto:
                    continue
                round_no = int(self._safe_float(round_item.get("round", idx + 1), idx + 1))
                if round_no <= 0:
                    round_no = idx + 1
                reason_codes = auto.get("reason_codes", []) if isinstance(auto.get("reason_codes", []), list) else []
                adaptive_payload = auto.get("adaptive", {}) if isinstance(auto.get("adaptive", {}), dict) else {}
                max_runs = int(self._safe_float(auto.get("max_runs", 0), 0))
                max_runs_base = int(self._safe_float(auto.get("max_runs_base", max_runs), max_runs))
                out.append(
                    {
                        "date": dstr,
                        "source": source,
                        "round": int(round_no),
                        "triggered": bool(auto.get("triggered", False)),
                        "attempted": bool(auto.get("attempted", auto.get("ran", False))),
                        "ran": bool(auto.get("ran", False)),
                        "skipped_reason": str(auto.get("skipped_reason", "")).strip(),
                        "error": str(auto.get("error", "")).strip(),
                        "reason_codes": [str(x).strip() for x in reason_codes if str(x).strip()],
                        "runs_used": int(self._safe_float(auto.get("runs_used", 0), 0)),
                        "next_allowed_round": int(self._safe_float(auto.get("next_allowed_round", 0), 0)),
                        "cooldown_remaining_rounds": int(self._safe_float(auto.get("cooldown_remaining_rounds", 0), 0)),
                        "max_runs": int(max_runs),
                        "max_runs_base": int(max_runs_base),
                        "adaptive_reason": str(adaptive_payload.get("reason", "")).strip(),
                        "adaptive_factor": float(self._safe_float(adaptive_payload.get("factor", 1.0), 1.0)),
                        "adaptive_trigger_density": float(
                            self._safe_float(adaptive_payload.get("trigger_density", 0.0), 0.0)
                        ),
                    }
                )
        out.sort(key=lambda row: (str(row.get("date", "")), int(self._safe_float(row.get("round", 0), 0))))
        return out

    def _write_stress_autorun_history_artifact(
        self,
        *,
        as_of: date,
        series: list[dict[str, Any]],
        metrics: dict[str, Any],
        retention_days: int,
        checksum_index_enabled: bool = True,
    ) -> dict[str, Any]:
        policy = self._artifact_governance_profile(
            profile_name="stress_autorun_history",
            fallback_retention_days=retention_days,
            fallback_checksum_index_enabled=checksum_index_enabled,
        )
        keep_days = int(policy.get("retention_days", max(1, int(retention_days))))
        index_enabled = bool(policy.get("checksum_index_enabled", checksum_index_enabled))
        total_rounds = len(series)
        if total_rounds <= 0:
            return {
                "written": False,
                "json": "",
                "md": "",
                "total_rounds": 0,
                "retention_days": int(keep_days),
                "rotated_out_count": 0,
                "rotated_out_dates": [],
                "rotation_failed": False,
                "checksum_index_enabled": bool(index_enabled),
                "checksum_index_written": False,
                "checksum_index_path": "",
                "checksum_index_entries": 0,
                "checksum_index_failed": False,
                "reason": "no_history",
            }

        skip_reason_counts: dict[str, int] = {}
        trigger_reason_counts: dict[str, int] = {}
        for row in series:
            if bool(row.get("triggered", False)) and (not bool(row.get("ran", False))):
                key = str(row.get("skipped_reason", "")).strip() or "unknown"
                skip_reason_counts[key] = int(skip_reason_counts.get(key, 0) + 1)
            if bool(row.get("triggered", False)):
                for code in row.get("reason_codes", []):
                    key = str(code).strip()
                    if key:
                        trigger_reason_counts[key] = int(trigger_reason_counts.get(key, 0) + 1)

        top_skip_reasons = [
            {"reason": str(k), "count": int(v)}
            for k, v in sorted(skip_reason_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[:20]
        ]
        top_trigger_reasons = [
            {"reason": str(k), "count": int(v)}
            for k, v in sorted(trigger_reason_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[:20]
        ]
        payload = {
            "date": as_of.isoformat(),
            "metrics": metrics,
            "top_skip_reasons": top_skip_reasons,
            "top_trigger_reasons": top_trigger_reasons,
            "series": series[-120:],
        }
        review_dir = self.output_dir / "review"
        json_path = review_dir / f"{as_of.isoformat()}_stress_autorun_history.json"
        md_path = review_dir / f"{as_of.isoformat()}_stress_autorun_history.md"

        lines: list[str] = []
        lines.append(f"# Stress Autorun History | {as_of.isoformat()}")
        lines.append("")
        lines.append(f"- rounds_total: `{int(metrics.get('rounds_total', 0))}`")
        lines.append(f"- triggered_rounds: `{int(metrics.get('triggered_rounds', 0))}`")
        lines.append(f"- ran_rounds: `{int(metrics.get('ran_rounds', 0))}`")
        lines.append(f"- skipped_rounds: `{int(metrics.get('skipped_rounds', 0))}`")
        lines.append(
            "- trigger_density/run_rate/cooldown_efficiency: "
            + f"`{self._safe_float(metrics.get('trigger_density', 0.0), 0.0):.2%}` / "
            + f"`{self._safe_float(metrics.get('run_rate_when_triggered', 0.0), 0.0):.2%}` / "
            + f"`{self._safe_float(metrics.get('cooldown_efficiency', 0.0), 0.0):.2%}`"
        )
        lines.append("")
        lines.append("## Top Skip Reasons")
        if top_skip_reasons:
            for item in top_skip_reasons:
                lines.append(f"- `{item['reason']}` x `{item['count']}`")
        else:
            lines.append("- NONE")
        lines.append("")
        lines.append("## Top Trigger Reasons")
        if top_trigger_reasons:
            for item in top_trigger_reasons:
                lines.append(f"- `{item['reason']}` x `{item['count']}`")
        else:
            lines.append("- NONE")
        lines.append("")
        lines.append("## Recent Rounds")
        for row in series[-40:]:
            lines.append(
                "- "
                + f"{row.get('date', '')}#{int(self._safe_float(row.get('round', 0), 0))} "
                + f"triggered={bool(row.get('triggered', False))} "
                + f"ran={bool(row.get('ran', False))} "
                + f"skip={str(row.get('skipped_reason', '') or 'N/A')} "
                + f"reasons={','.join(row.get('reason_codes', [])) or 'N/A'}"
            )

        try:
            write_json(json_path, payload)
            write_markdown(md_path, "\n".join(lines) + "\n")
        except Exception as exc:
            return {
                "written": False,
                "json": "",
                "md": "",
                "total_rounds": int(total_rounds),
                "retention_days": int(keep_days),
                "rotated_out_count": 0,
                "rotated_out_dates": [],
                "rotation_failed": False,
                "checksum_index_enabled": bool(index_enabled),
                "checksum_index_written": False,
                "checksum_index_path": "",
                "checksum_index_entries": 0,
                "checksum_index_failed": False,
                "reason": f"write_failed:{type(exc).__name__}:{exc}",
            }

        governance = self._apply_artifact_governance(
            as_of=as_of,
            review_dir=review_dir,
            profile_name="stress_autorun_history",
            fallback_retention_days=keep_days,
            fallback_checksum_index_enabled=index_enabled,
        )

        return {
            "written": True,
            "json": str(json_path),
            "md": str(md_path),
            "total_rounds": int(total_rounds),
            "retention_days": int(governance.get("retention_days", keep_days)),
            "rotated_out_count": int(governance.get("rotated_out_count", 0)),
            "rotated_out_dates": [str(x) for x in governance.get("rotated_out_dates", [])],
            "rotation_failed": bool(governance.get("rotation_failed", False)),
            "checksum_index_enabled": bool(governance.get("checksum_index_enabled", index_enabled)),
            "checksum_index_written": bool(governance.get("checksum_index_written", False)),
            "checksum_index_path": str(governance.get("checksum_index_path", "")),
            "checksum_index_entries": int(governance.get("checksum_index_entries", 0)),
            "checksum_index_failed": bool(governance.get("checksum_index_failed", False)),
            "reason": str(governance.get("reason", "")),
        }

    def _write_stress_autorun_reason_drift_artifact(
        self,
        *,
        as_of: date,
        series: list[dict[str, Any]],
        metrics: dict[str, Any],
        window_trace: list[dict[str, Any]],
        retention_days: int,
        checksum_index_enabled: bool = True,
    ) -> dict[str, Any]:
        policy = self._artifact_governance_profile(
            profile_name="stress_autorun_reason_drift",
            fallback_retention_days=retention_days,
            fallback_checksum_index_enabled=checksum_index_enabled,
        )
        keep_days = int(policy.get("retention_days", max(1, int(retention_days))))
        index_enabled = bool(policy.get("checksum_index_enabled", checksum_index_enabled))
        total_rounds = len(series)
        transition_counts = (
            metrics.get("transition_counts", {}) if isinstance(metrics.get("transition_counts", {}), dict) else {}
        )
        top_transitions = [
            {"transition": str(k), "count": int(v)}
            for k, v in sorted(transition_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[:20]
        ]
        payload = {
            "date": as_of.isoformat(),
            "metrics": metrics,
            "top_transitions": top_transitions,
            "window_trace": window_trace[-160:],
            "series": series[-120:],
        }
        review_dir = self.output_dir / "review"
        json_path = review_dir / f"{as_of.isoformat()}_stress_autorun_reason_drift.json"
        md_path = review_dir / f"{as_of.isoformat()}_stress_autorun_reason_drift.md"

        lines: list[str] = []
        lines.append(f"# Stress Autorun Reason Drift | {as_of.isoformat()}")
        lines.append("")
        lines.append(f"- rounds_total: `{int(metrics.get('rounds_total', 0))}`")
        lines.append(
            f"- baseline/recent rounds: `{int(metrics.get('baseline_rounds', 0))}` / `{int(metrics.get('recent_rounds', 0))}`"
        )
        lines.append(
            "- reason_mix_gap/change_point_gap: "
            + f"`{self._safe_float(metrics.get('reason_mix_gap', 0.0), 0.0):.3f}` / "
            + f"`{self._safe_float(metrics.get('change_point_gap', 0.0), 0.0):.3f}`"
        )
        lines.append("")
        lines.append("## Top Transitions")
        if top_transitions:
            for item in top_transitions:
                lines.append(f"- `{item['transition']}` x `{item['count']}`")
        else:
            lines.append("- NONE")
        lines.append("")
        lines.append("## Window Trace")
        if window_trace:
            for row in window_trace[-40:]:
                lines.append(
                    "- "
                    + f"{str(row.get('end_date', ''))}#{int(self._safe_float(row.get('end_round', 0), 0))} "
                    + f"mix_gap={self._safe_float(row.get('reason_mix_gap', 0.0), 0.0):.3f} "
                    + f"cp_gap={self._safe_float(row.get('change_point_gap', 0.0), 0.0):.3f} "
                    + "baseline(high/low)="
                    + f"{self._safe_float(row.get('baseline_high_ratio', 0.0), 0.0):.2%}/"
                    + f"{self._safe_float(row.get('baseline_low_ratio', 0.0), 0.0):.2%} "
                    + "recent(high/low)="
                    + f"{self._safe_float(row.get('recent_high_ratio', 0.0), 0.0):.2%}/"
                    + f"{self._safe_float(row.get('recent_low_ratio', 0.0), 0.0):.2%}"
                )
        else:
            lines.append("- NONE")
        lines.append("")
        lines.append("## Recent Reasons")
        for row in series[-40:]:
            lines.append(
                "- "
                + f"{str(row.get('date', ''))}#{int(self._safe_float(row.get('round', 0), 0))} "
                + f"reason={str(row.get('adaptive_reason', '') or 'N/A')} "
                + f"bucket={str(row.get('reason_bucket', '') or 'N/A')}"
            )

        try:
            write_json(json_path, payload)
            write_markdown(md_path, "\n".join(lines) + "\n")
        except Exception as exc:
            return {
                "written": False,
                "json": "",
                "md": "",
                "total_rounds": int(total_rounds),
                "transition_count": int(len(top_transitions)),
                "window_trace_points": int(len(window_trace)),
                "retention_days": int(keep_days),
                "rotated_out_count": 0,
                "rotated_out_dates": [],
                "rotation_failed": False,
                "checksum_index_enabled": bool(index_enabled),
                "checksum_index_written": False,
                "checksum_index_path": "",
                "checksum_index_entries": 0,
                "checksum_index_failed": False,
                "reason": f"write_failed:{type(exc).__name__}:{exc}",
            }

        governance = self._apply_artifact_governance(
            as_of=as_of,
            review_dir=review_dir,
            profile_name="stress_autorun_reason_drift",
            fallback_retention_days=keep_days,
            fallback_checksum_index_enabled=index_enabled,
        )

        return {
            "written": True,
            "json": str(json_path),
            "md": str(md_path),
            "total_rounds": int(total_rounds),
            "transition_count": int(len(top_transitions)),
            "window_trace_points": int(len(window_trace)),
            "retention_days": int(governance.get("retention_days", keep_days)),
            "rotated_out_count": int(governance.get("rotated_out_count", 0)),
            "rotated_out_dates": [str(x) for x in governance.get("rotated_out_dates", [])],
            "rotation_failed": bool(governance.get("rotation_failed", False)),
            "checksum_index_enabled": bool(governance.get("checksum_index_enabled", index_enabled)),
            "checksum_index_written": bool(governance.get("checksum_index_written", False)),
            "checksum_index_path": str(governance.get("checksum_index_path", "")),
            "checksum_index_entries": int(governance.get("checksum_index_entries", 0)),
            "checksum_index_failed": bool(governance.get("checksum_index_failed", False)),
            "reason": str(governance.get("reason", "")),
        }

    def _stress_autorun_history_metrics(self, *, as_of: date) -> dict[str, Any]:
        val = self.settings.validation if isinstance(self.settings.validation, dict) else {}
        enabled = bool(val.get("ops_stress_autorun_history_enabled", True))
        if not enabled:
            return {
                "active": False,
                "enabled": False,
                "window_days": 0,
                "samples": 0,
                "min_samples": 0,
                "checks": {},
                "thresholds": {},
                "metrics": {},
                "alerts": [],
                "artifacts": {},
                "series": [],
            }

        window_days = max(1, int(val.get("ops_stress_autorun_history_window_days", 30)))
        min_rounds = max(1, int(val.get("ops_stress_autorun_history_min_rounds", 3)))
        history_artifact_policy = self._artifact_governance_profile(
            profile_name="stress_autorun_history",
            fallback_retention_days=max(1, int(val.get("ops_stress_autorun_history_retention_days", 30))),
            fallback_checksum_index_enabled=self._safe_bool(
                val.get("ops_stress_autorun_history_checksum_index_enabled", True),
                True,
            ),
        )
        retention_days = int(history_artifact_policy.get("retention_days", 30))
        checksum_index_enabled = bool(history_artifact_policy.get("checksum_index_enabled", True))
        series = self._load_review_loop_history_series(as_of=as_of, window_days=window_days)
        rounds_total = len(series)
        triggered_rounds = 0
        attempted_rounds = 0
        ran_rounds = 0
        skipped_rounds = 0
        cooldown_skip_rounds = 0
        max_runs_skip_rounds = 0
        runner_unavailable_skip_rounds = 0
        errored_rounds = 0
        payload_days: set[str] = set()
        triggered_days: set[str] = set()
        skip_reason_counts: dict[str, int] = {}
        trigger_reason_counts: dict[str, int] = {}

        for row in series:
            dstr = str(row.get("date", "")).strip()
            if dstr:
                payload_days.add(dstr)
            triggered = bool(row.get("triggered", False))
            attempted = bool(row.get("attempted", False))
            ran = bool(row.get("ran", False))
            skip_reason = str(row.get("skipped_reason", "")).strip()
            err = str(row.get("error", "")).strip()
            reason_codes = row.get("reason_codes", []) if isinstance(row.get("reason_codes", []), list) else []

            if triggered:
                triggered_rounds += 1
                if dstr:
                    triggered_days.add(dstr)
                for code in reason_codes:
                    key = str(code).strip()
                    if key:
                        trigger_reason_counts[key] = int(trigger_reason_counts.get(key, 0) + 1)
            if attempted:
                attempted_rounds += 1
            if ran:
                ran_rounds += 1
            if triggered and (not ran):
                skipped_rounds += 1
                reason_key = skip_reason or "unknown"
                skip_reason_counts[reason_key] = int(skip_reason_counts.get(reason_key, 0) + 1)
                if reason_key == "cooldown_active":
                    cooldown_skip_rounds += 1
                elif reason_key == "max_runs_reached":
                    max_runs_skip_rounds += 1
                elif reason_key == "runner_unavailable":
                    runner_unavailable_skip_rounds += 1
            if err:
                errored_rounds += 1

        active = rounds_total >= min_rounds
        trigger_density = self._ratio(triggered_rounds, rounds_total)
        attempt_rate_when_triggered = self._ratio(attempted_rounds, triggered_rounds)
        run_rate_when_triggered = self._ratio(ran_rounds, triggered_rounds)
        cooldown_efficiency = self._ratio(cooldown_skip_rounds, cooldown_skip_rounds + ran_rounds)
        checks: dict[str, bool] = {}
        alerts: list[str] = []
        if not active:
            alerts.append("stress_autorun_history_insufficient_rounds")
        if active and triggered_rounds <= 0:
            alerts.append("stress_autorun_history_no_triggers")

        metrics = {
            "payload_days": int(len(payload_days)),
            "triggered_days": int(len(triggered_days)),
            "rounds_total": int(rounds_total),
            "triggered_rounds": int(triggered_rounds),
            "attempted_rounds": int(attempted_rounds),
            "ran_rounds": int(ran_rounds),
            "skipped_rounds": int(skipped_rounds),
            "cooldown_skip_rounds": int(cooldown_skip_rounds),
            "max_runs_skip_rounds": int(max_runs_skip_rounds),
            "runner_unavailable_skip_rounds": int(runner_unavailable_skip_rounds),
            "errored_rounds": int(errored_rounds),
            "trigger_density": float(trigger_density),
            "attempt_rate_when_triggered": float(attempt_rate_when_triggered),
            "run_rate_when_triggered": float(run_rate_when_triggered),
            "cooldown_efficiency": float(cooldown_efficiency),
            "skip_reason_counts": dict(sorted(skip_reason_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))),
            "trigger_reason_counts": dict(
                sorted(trigger_reason_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
            ),
        }
        artifact = self._write_stress_autorun_history_artifact(
            as_of=as_of,
            series=series,
            metrics=metrics,
            retention_days=retention_days,
            checksum_index_enabled=checksum_index_enabled,
        )
        artifact_failed = (
            rounds_total > 0
            and (not bool(artifact.get("written", False)))
            and str(artifact.get("reason", "")).startswith("write_failed")
        )
        artifact_rotation_failed = bool(artifact.get("rotation_failed", False))
        artifact_checksum_index_failed = bool(artifact.get("checksum_index_failed", False))
        if artifact_failed:
            alerts.append("stress_autorun_history_artifact_failed")
        if artifact_rotation_failed:
            alerts.append("stress_autorun_history_artifact_rotation_failed")
        if artifact_checksum_index_failed:
            alerts.append("stress_autorun_history_artifact_checksum_index_failed")

        if active:
            checks["artifact_rotation_ok"] = bool(not artifact_rotation_failed)
            checks["artifact_checksum_index_ok"] = bool((not checksum_index_enabled) or (not artifact_checksum_index_failed))

        metrics["artifact_written"] = bool(artifact.get("written", False))
        metrics["artifact_failed"] = bool(artifact_failed)
        metrics["artifact_total_rounds"] = int(artifact.get("total_rounds", 0))
        metrics["artifact_retention_days"] = int(artifact.get("retention_days", retention_days))
        metrics["artifact_rotated_out_count"] = int(artifact.get("rotated_out_count", 0))
        metrics["artifact_rotation_failed"] = bool(artifact_rotation_failed)
        metrics["artifact_checksum_index_enabled"] = bool(
            artifact.get("checksum_index_enabled", checksum_index_enabled)
        )
        metrics["artifact_checksum_index_written"] = bool(artifact.get("checksum_index_written", False))
        metrics["artifact_checksum_index_entries"] = int(artifact.get("checksum_index_entries", 0))
        metrics["artifact_checksum_index_failed"] = bool(artifact_checksum_index_failed)

        return {
            "active": bool(active),
            "enabled": True,
            "window_days": int(window_days),
            "samples": int(rounds_total),
            "min_samples": int(min_rounds),
            "checks": checks,
            "thresholds": {
                "ops_stress_autorun_history_window_days": int(window_days),
                "ops_stress_autorun_history_min_rounds": int(min_rounds),
                "ops_stress_autorun_history_retention_days": int(retention_days),
                "ops_stress_autorun_history_checksum_index_enabled": bool(checksum_index_enabled),
            },
            "metrics": metrics,
            "alerts": alerts,
            "artifacts": {
                "history": {
                    "written": bool(artifact.get("written", False)),
                    "json": str(artifact.get("json", "")),
                    "md": str(artifact.get("md", "")),
                    "total_rounds": int(artifact.get("total_rounds", 0)),
                    "retention_days": int(artifact.get("retention_days", retention_days)),
                    "rotated_out_count": int(artifact.get("rotated_out_count", 0)),
                    "rotated_out_dates": [str(x) for x in artifact.get("rotated_out_dates", [])],
                    "rotation_failed": bool(artifact.get("rotation_failed", False)),
                    "checksum_index_enabled": bool(
                        artifact.get("checksum_index_enabled", checksum_index_enabled)
                    ),
                    "checksum_index_written": bool(artifact.get("checksum_index_written", False)),
                    "checksum_index_path": str(artifact.get("checksum_index_path", "")),
                    "checksum_index_entries": int(artifact.get("checksum_index_entries", 0)),
                    "checksum_index_failed": bool(artifact.get("checksum_index_failed", False)),
                    "reason": str(artifact.get("reason", "")),
                }
            },
            "series": series[-60:],
        }

    def _stress_autorun_adaptive_saturation_metrics(self, *, as_of: date) -> dict[str, Any]:
        val = self.settings.validation if isinstance(self.settings.validation, dict) else {}
        enabled = bool(val.get("ops_stress_autorun_adaptive_monitor_enabled", True))
        if not enabled:
            return {
                "active": False,
                "enabled": False,
                "window_days": 0,
                "samples": 0,
                "min_samples": 0,
                "checks": {},
                "thresholds": {},
                "metrics": {},
                "alerts": [],
                "series": [],
            }

        window_days = max(1, int(val.get("ops_stress_autorun_adaptive_monitor_window_days", 30)))
        min_rounds = max(1, int(val.get("ops_stress_autorun_adaptive_monitor_min_rounds", 3)))
        ratio_floor = max(
            0.0,
            self._safe_float(val.get("ops_stress_autorun_adaptive_effective_base_ratio_floor", 0.50), 0.50),
        )
        ratio_ceiling = max(
            float(ratio_floor),
            self._safe_float(val.get("ops_stress_autorun_adaptive_effective_base_ratio_ceiling", 2.00), 2.00),
        )
        throttle_ratio_max = min(
            1.0,
            max(
                0.0,
                self._safe_float(val.get("ops_stress_autorun_adaptive_throttle_ratio_max", 0.85), 0.85),
            ),
        )
        expand_ratio_max = min(
            1.0,
            max(
                0.0,
                self._safe_float(val.get("ops_stress_autorun_adaptive_expand_ratio_max", 0.85), 0.85),
            ),
        )

        raw_series = self._load_review_loop_history_series(as_of=as_of, window_days=window_days)
        series: list[dict[str, Any]] = []
        reason_counts: dict[str, int] = {}
        for row in raw_series:
            if not isinstance(row, dict):
                continue
            max_runs_base = int(self._safe_float(row.get("max_runs_base", 0), 0))
            max_runs = int(self._safe_float(row.get("max_runs", 0), 0))
            if max_runs_base <= 0:
                continue
            ratio = float(max_runs) / float(max_runs_base)
            reason = str(row.get("adaptive_reason", "")).strip() or "unknown"
            reason_counts[reason] = int(reason_counts.get(reason, 0) + 1)
            series.append(
                {
                    "date": str(row.get("date", "")),
                    "round": int(self._safe_float(row.get("round", 0), 0)),
                    "max_runs": int(max_runs),
                    "max_runs_base": int(max_runs_base),
                    "effective_base_ratio": float(ratio),
                    "adaptive_reason": reason,
                    "adaptive_factor": float(self._safe_float(row.get("adaptive_factor", 1.0), 1.0)),
                    "adaptive_trigger_density": float(
                        self._safe_float(row.get("adaptive_trigger_density", 0.0), 0.0)
                    ),
                    "triggered": bool(row.get("triggered", False)),
                    "ran": bool(row.get("ran", False)),
                    "skipped_reason": str(row.get("skipped_reason", "")).strip(),
                }
            )

        rounds_total = len(series)
        active = rounds_total >= min_rounds
        ratio_sum = sum(float(row.get("effective_base_ratio", 0.0)) for row in series)
        ratio_avg = (ratio_sum / rounds_total) if rounds_total > 0 else 0.0
        ratio_min = min((float(row.get("effective_base_ratio", 0.0)) for row in series), default=0.0)
        ratio_max = max((float(row.get("effective_base_ratio", 0.0)) for row in series), default=0.0)
        ratio_latest = float(series[-1].get("effective_base_ratio", 0.0)) if series else 0.0
        throttle_rounds = sum(1 for row in series if float(row.get("effective_base_ratio", 0.0)) < 1.0)
        expand_rounds = sum(1 for row in series if float(row.get("effective_base_ratio", 0.0)) > 1.0)
        neutral_rounds = max(0, int(rounds_total - throttle_rounds - expand_rounds))
        throttle_ratio = self._ratio(throttle_rounds, rounds_total)
        expand_ratio = self._ratio(expand_rounds, rounds_total)

        checks: dict[str, bool] = {}
        alerts: list[str] = []
        if not active:
            alerts.append("stress_autorun_adaptive_insufficient_rounds")
        else:
            checks["effective_base_ratio_floor_ok"] = bool(ratio_avg >= ratio_floor)
            checks["effective_base_ratio_ceiling_ok"] = bool(ratio_avg <= ratio_ceiling)
            checks["throttle_ratio_ok"] = bool(throttle_ratio <= throttle_ratio_max)
            checks["expand_ratio_ok"] = bool(expand_ratio <= expand_ratio_max)
            if not checks["effective_base_ratio_floor_ok"]:
                alerts.append("stress_autorun_adaptive_ratio_low")
            if not checks["effective_base_ratio_ceiling_ok"]:
                alerts.append("stress_autorun_adaptive_ratio_high")
            if not checks["throttle_ratio_ok"]:
                alerts.append("stress_autorun_adaptive_throttle_ratio_high")
            if not checks["expand_ratio_ok"]:
                alerts.append("stress_autorun_adaptive_expand_ratio_high")

        return {
            "active": bool(active),
            "enabled": True,
            "window_days": int(window_days),
            "samples": int(rounds_total),
            "min_samples": int(min_rounds),
            "checks": checks,
            "thresholds": {
                "ops_stress_autorun_adaptive_monitor_window_days": int(window_days),
                "ops_stress_autorun_adaptive_monitor_min_rounds": int(min_rounds),
                "ops_stress_autorun_adaptive_effective_base_ratio_floor": float(ratio_floor),
                "ops_stress_autorun_adaptive_effective_base_ratio_ceiling": float(ratio_ceiling),
                "ops_stress_autorun_adaptive_throttle_ratio_max": float(throttle_ratio_max),
                "ops_stress_autorun_adaptive_expand_ratio_max": float(expand_ratio_max),
            },
            "metrics": {
                "rounds_total": int(rounds_total),
                "effective_base_ratio_avg": float(ratio_avg),
                "effective_base_ratio_min": float(ratio_min),
                "effective_base_ratio_max": float(ratio_max),
                "effective_base_ratio_latest": float(ratio_latest),
                "throttle_rounds": int(throttle_rounds),
                "expand_rounds": int(expand_rounds),
                "neutral_rounds": int(neutral_rounds),
                "throttle_ratio": float(throttle_ratio),
                "expand_ratio": float(expand_ratio),
                "reason_counts": dict(sorted(reason_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))),
            },
            "alerts": alerts,
            "series": series[-60:],
        }

    def _stress_autorun_adaptive_reason_drift_metrics(self, *, as_of: date) -> dict[str, Any]:
        val = self.settings.validation if isinstance(self.settings.validation, dict) else {}
        enabled = bool(val.get("ops_stress_autorun_reason_drift_enabled", True))
        if not enabled:
            return {
                "active": False,
                "enabled": False,
                "window_days": 0,
                "samples": 0,
                "min_samples": 0,
                "checks": {},
                "thresholds": {},
                "metrics": {},
                "alerts": [],
                "series": [],
            }

        def _bucket_ratios(rows: list[dict[str, Any]]) -> dict[str, float]:
            total = len(rows)
            high = sum(1 for item in rows if str(item.get("reason_bucket", "")) == "high")
            low = sum(1 for item in rows if str(item.get("reason_bucket", "")) == "low")
            other = max(0, int(total - high - low))
            return {
                "high": self._ratio(high, total),
                "low": self._ratio(low, total),
                "other": self._ratio(other, total),
            }

        window_days = max(1, int(val.get("ops_stress_autorun_reason_drift_window_days", 30)))
        min_rounds = max(2, int(val.get("ops_stress_autorun_reason_drift_min_rounds", 6)))
        recent_rounds_cfg = max(1, int(val.get("ops_stress_autorun_reason_drift_recent_rounds", 4)))
        reason_artifact_policy = self._artifact_governance_profile(
            profile_name="stress_autorun_reason_drift",
            fallback_retention_days=max(1, int(val.get("ops_stress_autorun_reason_drift_retention_days", 30))),
            fallback_checksum_index_enabled=self._safe_bool(
                val.get("ops_stress_autorun_reason_drift_checksum_index_enabled", True),
                True,
            ),
        )
        retention_days = int(reason_artifact_policy.get("retention_days", 30))
        checksum_index_enabled = bool(reason_artifact_policy.get("checksum_index_enabled", True))
        mix_gap_max = min(
            1.0,
            max(
                0.0,
                self._safe_float(val.get("ops_stress_autorun_reason_drift_mix_gap_max", 0.35), 0.35),
            ),
        )
        change_point_gap_max = min(
            1.0,
            max(
                0.0,
                self._safe_float(val.get("ops_stress_autorun_reason_drift_change_point_gap_max", 0.45), 0.45),
            ),
        )

        raw_series = self._load_review_loop_history_series(as_of=as_of, window_days=window_days)
        series: list[dict[str, Any]] = []
        reason_counts: dict[str, int] = {}
        for row in raw_series:
            if not isinstance(row, dict):
                continue
            reason = str(row.get("adaptive_reason", "")).strip()
            if not reason:
                continue
            if reason == "high_density_throttle":
                bucket = "high"
            elif reason == "low_density_expand":
                bucket = "low"
            else:
                bucket = "other"
            reason_counts[reason] = int(reason_counts.get(reason, 0) + 1)
            series.append(
                {
                    "date": str(row.get("date", "")),
                    "round": int(self._safe_float(row.get("round", 0), 0)),
                    "adaptive_reason": reason,
                    "reason_bucket": bucket,
                }
            )

        transition_counts: dict[str, int] = {}
        for idx in range(1, len(series)):
            prev_reason = str(series[idx - 1].get("adaptive_reason", "")).strip() or "unknown"
            curr_reason = str(series[idx].get("adaptive_reason", "")).strip() or "unknown"
            key = f"{prev_reason}->{curr_reason}"
            transition_counts[key] = int(transition_counts.get(key, 0) + 1)

        rounds_total = len(series)
        recent_rounds = min(recent_rounds_cfg, rounds_total)
        required_rounds = max(min_rounds, recent_rounds_cfg + 1)
        active = rounds_total >= required_rounds and recent_rounds > 0 and (rounds_total - recent_rounds) > 0

        baseline_series = series[:-recent_rounds] if recent_rounds > 0 else []
        recent_series = series[-recent_rounds:] if recent_rounds > 0 else []

        baseline_ratio = _bucket_ratios(baseline_series)
        recent_ratio = _bucket_ratios(recent_series)
        high_gap = abs(float(recent_ratio.get("high", 0.0)) - float(baseline_ratio.get("high", 0.0)))
        low_gap = abs(float(recent_ratio.get("low", 0.0)) - float(baseline_ratio.get("low", 0.0)))
        mix_gap = (high_gap + low_gap) * 0.5
        change_point_gap = max(high_gap, low_gap)
        window_trace: list[dict[str, Any]] = []
        if recent_rounds_cfg > 0 and rounds_total > recent_rounds_cfg:
            for end_idx in range(recent_rounds_cfg, rounds_total + 1):
                recent_slice = series[end_idx - recent_rounds_cfg : end_idx]
                baseline_slice = series[: end_idx - recent_rounds_cfg]
                if not baseline_slice:
                    continue
                b_ratio = _bucket_ratios(baseline_slice)
                r_ratio = _bucket_ratios(recent_slice)
                hg = abs(float(r_ratio.get("high", 0.0)) - float(b_ratio.get("high", 0.0)))
                lg = abs(float(r_ratio.get("low", 0.0)) - float(b_ratio.get("low", 0.0)))
                window_trace.append(
                    {
                        "end_date": str(recent_slice[-1].get("date", "")),
                        "end_round": int(self._safe_float(recent_slice[-1].get("round", 0), 0)),
                        "baseline_rounds": int(len(baseline_slice)),
                        "recent_rounds": int(len(recent_slice)),
                        "baseline_high_ratio": float(b_ratio.get("high", 0.0)),
                        "baseline_low_ratio": float(b_ratio.get("low", 0.0)),
                        "recent_high_ratio": float(r_ratio.get("high", 0.0)),
                        "recent_low_ratio": float(r_ratio.get("low", 0.0)),
                        "reason_mix_gap": float((hg + lg) * 0.5),
                        "change_point_gap": float(max(hg, lg)),
                    }
                )

        checks: dict[str, bool] = {}
        alerts: list[str] = []
        if not active:
            alerts.append("stress_autorun_reason_drift_insufficient_rounds")
        else:
            checks["reason_mix_gap_ok"] = bool(mix_gap <= mix_gap_max)
            checks["change_point_gap_ok"] = bool(change_point_gap <= change_point_gap_max)
            if not checks["reason_mix_gap_ok"]:
                alerts.append("stress_autorun_reason_mix_drift")
            if not checks["change_point_gap_ok"]:
                alerts.append("stress_autorun_reason_change_point")

        metrics = {
            "rounds_total": int(rounds_total),
            "baseline_rounds": int(len(baseline_series)),
            "recent_rounds": int(len(recent_series)),
            "baseline_high_ratio": float(baseline_ratio.get("high", 0.0)),
            "baseline_low_ratio": float(baseline_ratio.get("low", 0.0)),
            "baseline_other_ratio": float(baseline_ratio.get("other", 0.0)),
            "recent_high_ratio": float(recent_ratio.get("high", 0.0)),
            "recent_low_ratio": float(recent_ratio.get("low", 0.0)),
            "recent_other_ratio": float(recent_ratio.get("other", 0.0)),
            "reason_mix_gap": float(mix_gap),
            "change_point_gap": float(change_point_gap),
            "reason_counts": dict(sorted(reason_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))),
            "transition_counts": dict(
                sorted(transition_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
            ),
            "window_trace_points": int(len(window_trace)),
        }
        artifact = self._write_stress_autorun_reason_drift_artifact(
            as_of=as_of,
            series=series,
            metrics=metrics,
            window_trace=window_trace,
            retention_days=retention_days,
            checksum_index_enabled=checksum_index_enabled,
        )
        artifact_failed = (
            rounds_total > 0
            and (not bool(artifact.get("written", False)))
            and str(artifact.get("reason", "")).startswith("write_failed")
        )
        artifact_rotation_failed = bool(artifact.get("rotation_failed", False))
        artifact_checksum_index_failed = bool(artifact.get("checksum_index_failed", False))
        if artifact_failed:
            alerts.append("stress_autorun_reason_drift_artifact_failed")
        if artifact_rotation_failed:
            alerts.append("stress_autorun_reason_drift_artifact_rotation_failed")
        if artifact_checksum_index_failed:
            alerts.append("stress_autorun_reason_drift_artifact_checksum_index_failed")

        if active:
            checks["artifact_rotation_ok"] = bool(not artifact_rotation_failed)
            checks["artifact_checksum_index_ok"] = bool((not checksum_index_enabled) or (not artifact_checksum_index_failed))

        metrics["artifact_written"] = bool(artifact.get("written", False))
        metrics["artifact_failed"] = bool(artifact_failed)
        metrics["artifact_total_rounds"] = int(artifact.get("total_rounds", 0))
        metrics["artifact_retention_days"] = int(artifact.get("retention_days", retention_days))
        metrics["artifact_rotated_out_count"] = int(artifact.get("rotated_out_count", 0))
        metrics["artifact_rotation_failed"] = bool(artifact_rotation_failed)
        metrics["artifact_checksum_index_enabled"] = bool(
            artifact.get("checksum_index_enabled", checksum_index_enabled)
        )
        metrics["artifact_checksum_index_written"] = bool(artifact.get("checksum_index_written", False))
        metrics["artifact_checksum_index_entries"] = int(artifact.get("checksum_index_entries", 0))
        metrics["artifact_checksum_index_failed"] = bool(artifact_checksum_index_failed)

        return {
            "active": bool(active),
            "enabled": True,
            "window_days": int(window_days),
            "samples": int(rounds_total),
            "min_samples": int(required_rounds),
            "checks": checks,
            "thresholds": {
                "ops_stress_autorun_reason_drift_window_days": int(window_days),
                "ops_stress_autorun_reason_drift_min_rounds": int(min_rounds),
                "ops_stress_autorun_reason_drift_recent_rounds": int(recent_rounds_cfg),
                "ops_stress_autorun_reason_drift_mix_gap_max": float(mix_gap_max),
                "ops_stress_autorun_reason_drift_change_point_gap_max": float(change_point_gap_max),
                "ops_stress_autorun_reason_drift_retention_days": int(retention_days),
                "ops_stress_autorun_reason_drift_checksum_index_enabled": bool(checksum_index_enabled),
            },
            "metrics": metrics,
            "alerts": alerts,
            "artifacts": {
                "reason_drift": {
                    "written": bool(artifact.get("written", False)),
                    "json": str(artifact.get("json", "")),
                    "md": str(artifact.get("md", "")),
                    "total_rounds": int(artifact.get("total_rounds", 0)),
                    "transition_count": int(artifact.get("transition_count", 0)),
                    "window_trace_points": int(artifact.get("window_trace_points", 0)),
                    "retention_days": int(artifact.get("retention_days", retention_days)),
                    "rotated_out_count": int(artifact.get("rotated_out_count", 0)),
                    "rotated_out_dates": [str(x) for x in artifact.get("rotated_out_dates", [])],
                    "rotation_failed": bool(artifact.get("rotation_failed", False)),
                    "checksum_index_enabled": bool(
                        artifact.get("checksum_index_enabled", checksum_index_enabled)
                    ),
                    "checksum_index_written": bool(artifact.get("checksum_index_written", False)),
                    "checksum_index_path": str(artifact.get("checksum_index_path", "")),
                    "checksum_index_entries": int(artifact.get("checksum_index_entries", 0)),
                    "checksum_index_failed": bool(artifact.get("checksum_index_failed", False)),
                    "reason": str(artifact.get("reason", "")),
                }
            },
            "series": series[-60:],
            "window_trace": window_trace[-60:],
        }

    def _stress_autorun_adaptive_max_runs(
        self,
        *,
        as_of: date,
        base_max_runs: int,
        current_rounds: list[dict[str, Any]],
        enabled: bool,
        window_days: int,
        min_rounds: int,
        low_density_threshold: float,
        high_density_threshold: float,
        low_density_factor: float,
        high_density_factor: float,
        min_runs_floor: int,
        max_runs_cap: int,
    ) -> dict[str, Any]:
        base = max(0, int(base_max_runs))
        floor_runs = max(0, int(min_runs_floor))
        cap_runs = max(floor_runs, int(max_runs_cap))
        if cap_runs <= 0 and base > 0:
            cap_runs = base

        history_window = max(1, int(window_days))
        required_rounds = max(1, int(min_rounds))
        low_th = max(0.0, min(1.0, float(low_density_threshold)))
        high_th = max(0.0, min(1.0, float(high_density_threshold)))
        if low_th > high_th:
            low_th, high_th = high_th, low_th
        low_factor = max(0.0, float(low_density_factor))
        high_factor = max(0.0, float(high_density_factor))

        history_series = self._load_review_loop_history_series(as_of=as_of, window_days=history_window)
        history_rounds_total = len(history_series)
        history_triggered_rounds = sum(1 for row in history_series if bool(row.get("triggered", False)))

        current_rounds_total = 0
        current_triggered_rounds = 0
        for row in current_rounds:
            if not isinstance(row, dict):
                continue
            auto = (
                row.get("stress_matrix_autorun", {})
                if isinstance(row.get("stress_matrix_autorun", {}), dict)
                else {}
            )
            if not auto:
                continue
            current_rounds_total += 1
            if bool(auto.get("triggered", False)):
                current_triggered_rounds += 1

        rounds_total = int(history_rounds_total + current_rounds_total)
        triggered_rounds = int(history_triggered_rounds + current_triggered_rounds)
        trigger_density = self._ratio(triggered_rounds, rounds_total) if rounds_total > 0 else 0.0

        factor = 1.0
        reason = "adaptive_disabled"
        effective_max_runs = int(base)

        if base <= 0:
            reason = "base_max_runs_zero"
            effective_max_runs = 0
        elif not bool(enabled):
            reason = "adaptive_disabled"
            effective_max_runs = int(base)
        else:
            if rounds_total < required_rounds:
                reason = "insufficient_rounds"
                factor = 1.0
            elif trigger_density >= high_th:
                reason = "high_density_throttle"
                factor = float(high_factor)
            elif trigger_density <= low_th:
                reason = "low_density_expand"
                factor = float(low_factor)
            else:
                reason = "mid_density_neutral"
                factor = 1.0

            scaled = int(round(float(base) * float(factor)))
            scaled = max(floor_runs, scaled)
            scaled = min(cap_runs, scaled)
            effective_max_runs = max(0, int(scaled))

        return {
            "enabled": bool(enabled),
            "reason": str(reason),
            "factor": float(factor),
            "base_max_runs": int(base),
            "effective_max_runs": int(effective_max_runs),
            "window_days": int(history_window),
            "min_rounds": int(required_rounds),
            "low_density_threshold": float(low_th),
            "high_density_threshold": float(high_th),
            "low_density_factor": float(low_factor),
            "high_density_factor": float(high_factor),
            "min_runs_floor": int(floor_runs),
            "max_runs_cap": int(cap_runs),
            "history_rounds_total": int(history_rounds_total),
            "history_triggered_rounds": int(history_triggered_rounds),
            "current_rounds_total": int(current_rounds_total),
            "current_triggered_rounds": int(current_triggered_rounds),
            "rounds_total": int(rounds_total),
            "triggered_rounds": int(triggered_rounds),
            "trigger_density": float(trigger_density),
        }

    def _load_mode_feedback_series(self, *, as_of: date, window_days: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        wd = max(1, int(window_days))
        daily_dir = self.output_dir / "daily"
        for i in range(wd):
            day = as_of - timedelta(days=i)
            path = daily_dir / f"{day.isoformat()}_mode_feedback.json"
            payload = self.load_json_safely(path)
            if not payload:
                continue
            risk_control = payload.get("risk_control", {}) if isinstance(payload.get("risk_control", {}), dict) else {}
            mode_health = payload.get("mode_health", {}) if isinstance(payload.get("mode_health", {}), dict) else {}
            out.append(
                {
                    "date": day.isoformat(),
                    "runtime_mode": str(payload.get("runtime_mode", "")).strip(),
                    "risk_multiplier": self._safe_float(risk_control.get("risk_multiplier", 1.0), 1.0),
                    "source_confidence_score": self._safe_float(risk_control.get("source_confidence_score", 1.0), 1.0),
                    "mode_health_passed": bool(mode_health.get("passed", True)),
                }
            )
        out.reverse()
        return out

    def _effective_sqlite_path(self) -> Path:
        if self.sqlite_path is not None:
            return Path(self.sqlite_path)
        raw = str(self.settings.paths.get("sqlite", "output/artifacts/lie_engine.db")).strip()
        path = Path(raw)
        if path.is_absolute():
            return path
        return self.output_dir.parent / path

    def _latest_mode_feedback_payload(self, *, as_of: date, lookback_days: int = 30) -> dict[str, Any]:
        daily_dir = self.output_dir / "daily"
        for i in range(max(1, int(lookback_days))):
            day = as_of - timedelta(days=i)
            path = daily_dir / f"{day.isoformat()}_mode_feedback.json"
            payload = self.load_json_safely(path)
            if payload:
                return payload
        return {}

    def _slot_config(self) -> dict[str, Any]:
        schedule = self.settings.schedule if isinstance(self.settings.schedule, dict) else {}
        intraday_slots = schedule.get("intraday_slots", ["10:30", "14:30"])
        if not isinstance(intraday_slots, list):
            intraday_slots = ["10:30", "14:30"]
        out_slots: list[str] = []
        for raw in intraday_slots:
            txt = str(raw).strip()
            if txt:
                out_slots.append(txt)
        if not out_slots:
            out_slots = ["10:30", "14:30"]
        return {"intraday_slots": out_slots}

    @staticmethod
    def _regime_bucket(label: Any) -> str:
        txt = str(label or "").strip().lower()
        if not txt:
            return "unknown"
        if any(k in txt for k in ("极端波动", "extreme_vol", "extreme-vol", "extreme vol")):
            return "extreme_vol"
        if any(k in txt for k in ("强趋势", "弱趋势", "下跌趋势", "trend", "downtrend")):
            return "trend"
        if any(k in txt for k in ("震荡", "不确定", "range", "uncertain")):
            return "range"
        return "unknown"

    def _slot_regime_threshold_map(self, raw: Any, *, default: float) -> dict[str, float]:
        out = {
            "trend": float(default),
            "range": float(default),
            "extreme_vol": float(default),
        }
        if not isinstance(raw, dict):
            return out
        for k, v in raw.items():
            bucket = self._regime_bucket(k)
            if bucket not in out:
                continue
            candidate = self._safe_float(v, out[bucket])
            if 0.0 <= candidate <= 1.0:
                out[bucket] = float(candidate)
        return out

    def _load_slot_regime_threshold_overrides(self) -> dict[str, Any]:
        path = self.output_dir / "artifacts" / "slot_regime_thresholds_live.yaml"
        if not path.exists():
            return {}
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _slot_anomaly_metrics(self, *, as_of: date) -> dict[str, Any]:
        val = self.settings.validation if isinstance(self.settings.validation, dict) else {}
        cfg = self._slot_config()
        intraday_slots = list(cfg.get("intraday_slots", ["10:30", "14:30"]))
        window_days = max(1, int(val.get("ops_slot_window_days", 7)))
        min_samples = max(1, int(val.get("ops_slot_min_samples", 3)))
        pre_ratio_max = self._safe_float(val.get("ops_slot_premarket_anomaly_ratio_max", 0.50), 0.50)
        intraday_ratio_max = self._safe_float(val.get("ops_slot_intraday_anomaly_ratio_max", 0.50), 0.50)
        eod_ratio_max = self._safe_float(val.get("ops_slot_eod_anomaly_ratio_max", 0.50), 0.50)
        eod_quality_ratio_max = self._safe_float(val.get("ops_slot_eod_quality_anomaly_ratio_max", eod_ratio_max), eod_ratio_max)
        eod_risk_ratio_max = self._safe_float(val.get("ops_slot_eod_risk_anomaly_ratio_max", eod_ratio_max), eod_ratio_max)
        eod_quality_ratio_by_regime = self._slot_regime_threshold_map(
            val.get("ops_slot_eod_quality_anomaly_ratio_max_by_regime", {}),
            default=eod_quality_ratio_max,
        )
        eod_risk_ratio_by_regime = self._slot_regime_threshold_map(
            val.get("ops_slot_eod_risk_anomaly_ratio_max_by_regime", {}),
            default=eod_risk_ratio_max,
        )
        use_live_regime_thresholds = bool(val.get("ops_slot_use_live_regime_thresholds", True))
        live_overrides_applied = False
        if use_live_regime_thresholds:
            live_payload = self._load_slot_regime_threshold_overrides()
            if live_payload:
                eod_quality_ratio_by_regime = self._slot_regime_threshold_map(
                    live_payload.get("ops_slot_eod_quality_anomaly_ratio_max_by_regime", {}),
                    default=eod_quality_ratio_max,
                )
                eod_risk_ratio_by_regime = self._slot_regime_threshold_map(
                    live_payload.get("ops_slot_eod_risk_anomaly_ratio_max_by_regime", {}),
                    default=eod_risk_ratio_max,
                )
                live_overrides_applied = True
        missing_ratio_max = self._safe_float(val.get("ops_slot_missing_ratio_max", 0.35), 0.35)
        source_floor = self._safe_float(
            val.get("ops_slot_source_confidence_floor", val.get("source_confidence_min", 0.75)),
            0.75,
        )
        risk_floor = self._safe_float(
            val.get("ops_slot_risk_multiplier_floor", val.get("execution_min_risk_multiplier", 0.20)),
            0.20,
        )

        def _slot_template() -> dict[str, Any]:
            return {"expected": 0, "observed": 0, "missing": 0, "anomalies": 0}

        slots = {
            "premarket": _slot_template(),
            "intraday": _slot_template(),
            "eod": _slot_template(),
        }
        series: list[dict[str, Any]] = []
        active_days = 0
        total_expected = 0
        total_observed = 0
        total_missing = 0
        total_anomalies = 0
        eod_quality_anomaly_slots = 0
        eod_risk_anomaly_slots = 0
        eod_regime_buckets = {
            "trend": {"days": 0, "quality_anomalies": 0, "risk_anomalies": 0},
            "range": {"days": 0, "quality_anomalies": 0, "risk_anomalies": 0},
            "extreme_vol": {"days": 0, "quality_anomalies": 0, "risk_anomalies": 0},
        }
        eod_unknown_regime_days = 0

        logs_dir = self.output_dir / "logs"
        manifest_dir = self.output_dir / "artifacts" / "manifests"

        for i in range(window_days):
            day = as_of - timedelta(days=i)
            dstr = day.isoformat()
            local = {
                "premarket": _slot_template(),
                "intraday": _slot_template(),
                "eod": _slot_template(),
            }
            day_alerts: list[str] = []

            # premarket
            local["premarket"]["expected"] += 1
            pre = self.load_json_safely(logs_dir / f"{dstr}_premarket.json")
            if pre:
                local["premarket"]["observed"] += 1
                quality = pre.get("quality", {}) if isinstance(pre.get("quality", {}), dict) else {}
                q_flags = quality.get("flags", []) if isinstance(quality.get("flags", []), list) else []
                score = self._safe_float(quality.get("source_confidence_score", pre.get("source_confidence_score", 1.0)), 1.0)
                risk_mult = self._safe_float(pre.get("risk_multiplier", 1.0), 1.0)
                reasons: list[str] = []
                if not bool(quality.get("passed", True)):
                    reasons.append("quality_failed")
                if q_flags:
                    reasons.append("quality_flags")
                if score < source_floor:
                    reasons.append("source_confidence_low")
                if risk_mult < risk_floor:
                    reasons.append("risk_multiplier_low")
                if reasons:
                    local["premarket"]["anomalies"] += 1
                    day_alerts.append("premarket:" + "+".join(reasons[:2]))
            else:
                local["premarket"]["missing"] += 1
                day_alerts.append("premarket:missing")

            # intraday
            for slot in intraday_slots:
                local["intraday"]["expected"] += 1
                intraday = self.load_json_safely(logs_dir / f"{dstr}_intraday_{slot.replace(':', '')}.json")
                if intraday:
                    local["intraday"]["observed"] += 1
                    q_flags = intraday.get("quality_flags", []) if isinstance(intraday.get("quality_flags", []), list) else []
                    score = self._safe_float(intraday.get("source_confidence_score", 1.0), 1.0)
                    risk_mult = self._safe_float(intraday.get("risk_multiplier", 1.0), 1.0)
                    reasons = []
                    if q_flags:
                        reasons.append("quality_flags")
                    if score < source_floor:
                        reasons.append("source_confidence_low")
                    if risk_mult < risk_floor:
                        reasons.append("risk_multiplier_low")
                    if reasons:
                        local["intraday"]["anomalies"] += 1
                        day_alerts.append("intraday:" + "+".join(reasons[:2]))
                else:
                    local["intraday"]["missing"] += 1
                    day_alerts.append(f"intraday:{slot}:missing")

            # eod manifest
            local["eod"]["expected"] += 1
            eod = self.load_json_safely(manifest_dir / f"eod_{dstr}.json")
            if eod:
                local["eod"]["observed"] += 1
                checks = eod.get("checks", {}) if isinstance(eod.get("checks", {}), dict) else {}
                metrics = eod.get("metrics", {}) if isinstance(eod.get("metrics", {}), dict) else {}
                regime_bucket = self._regime_bucket(metrics.get("regime", ""))
                if regime_bucket in eod_regime_buckets:
                    eod_regime_buckets[regime_bucket]["days"] = int(eod_regime_buckets[regime_bucket]["days"]) + 1
                else:
                    eod_unknown_regime_days += 1
                reasons = []
                if not bool(checks.get("quality_passed", True)):
                    reasons.append("quality_failed")
                    eod_quality_anomaly_slots += 1
                    if regime_bucket in eod_regime_buckets:
                        eod_regime_buckets[regime_bucket]["quality_anomalies"] = (
                            int(eod_regime_buckets[regime_bucket]["quality_anomalies"]) + 1
                        )
                risk_mult = self._safe_float(metrics.get("risk_multiplier", 1.0), 1.0)
                if risk_mult < risk_floor:
                    reasons.append("risk_multiplier_low")
                    eod_risk_anomaly_slots += 1
                    if regime_bucket in eod_regime_buckets:
                        eod_regime_buckets[regime_bucket]["risk_anomalies"] = (
                            int(eod_regime_buckets[regime_bucket]["risk_anomalies"]) + 1
                        )
                if reasons:
                    local["eod"]["anomalies"] += 1
                    day_alerts.append("eod:" + "+".join(reasons[:2]))
            else:
                local["eod"]["missing"] += 1
                day_alerts.append("eod:missing")

            day_expected = sum(int(local[k]["expected"]) for k in local)
            day_observed = sum(int(local[k]["observed"]) for k in local)
            day_missing = sum(int(local[k]["missing"]) for k in local)
            day_anomalies = sum(int(local[k]["anomalies"]) for k in local)

            # only evaluate days that have at least one produced slot artifact
            if day_observed <= 0:
                continue

            active_days += 1
            total_expected += day_expected
            total_observed += day_observed
            total_missing += day_missing
            total_anomalies += day_anomalies
            for key in slots:
                for field in ("expected", "observed", "missing", "anomalies"):
                    slots[key][field] = int(slots[key][field]) + int(local[key][field])
            series.append(
                {
                    "date": dstr,
                    "expected": day_expected,
                    "observed": day_observed,
                    "missing": day_missing,
                    "anomalies": day_anomalies,
                    "alerts": day_alerts[:6],
                }
            )

        def _ratio(n: int, d: int) -> float:
            return float(n / d) if d > 0 else 0.0

        pre_ratio = _ratio(int(slots["premarket"]["anomalies"]), int(slots["premarket"]["expected"]))
        intraday_ratio = _ratio(int(slots["intraday"]["anomalies"]), int(slots["intraday"]["expected"]))
        eod_ratio = _ratio(int(slots["eod"]["anomalies"]), int(slots["eod"]["expected"]))
        eod_quality_ratio = _ratio(eod_quality_anomaly_slots, int(slots["eod"]["expected"]))
        eod_risk_ratio = _ratio(eod_risk_anomaly_slots, int(slots["eod"]["expected"]))
        missing_ratio = _ratio(total_missing, total_expected)
        eod_regime_stats: dict[str, dict[str, Any]] = {}
        eod_quality_regime_breach_count = 0
        eod_risk_regime_breach_count = 0
        for bucket in ("trend", "range", "extreme_vol"):
            row = eod_regime_buckets[bucket]
            days = int(row.get("days", 0))
            q_anom = int(row.get("quality_anomalies", 0))
            r_anom = int(row.get("risk_anomalies", 0))
            q_ratio = _ratio(q_anom, days)
            r_ratio = _ratio(r_anom, days)
            q_thr = float(eod_quality_ratio_by_regime.get(bucket, eod_quality_ratio_max))
            r_thr = float(eod_risk_ratio_by_regime.get(bucket, eod_risk_ratio_max))
            q_ok = bool(days <= 0 or q_ratio <= q_thr)
            r_ok = bool(days <= 0 or r_ratio <= r_thr)
            if days > 0 and not q_ok:
                eod_quality_regime_breach_count += 1
            if days > 0 and not r_ok:
                eod_risk_regime_breach_count += 1
            eod_regime_stats[bucket] = {
                "days": days,
                "quality_anomalies": q_anom,
                "risk_anomalies": r_anom,
                "quality_anomaly_ratio": q_ratio,
                "risk_anomaly_ratio": r_ratio,
                "quality_threshold": q_thr,
                "risk_threshold": r_thr,
                "quality_ok": q_ok,
                "risk_ok": r_ok,
            }

        active = active_days >= min_samples
        checks = {
            "missing_ratio_ok": True,
            "premarket_anomaly_ok": True,
            "intraday_anomaly_ok": True,
            "eod_quality_anomaly_ok": True,
            "eod_risk_anomaly_ok": True,
            "eod_quality_regime_bucket_ok": True,
            "eod_risk_regime_bucket_ok": True,
            "eod_anomaly_ok": True,
        }
        alerts: list[str] = []
        if active:
            checks["missing_ratio_ok"] = bool(missing_ratio <= missing_ratio_max)
            checks["premarket_anomaly_ok"] = bool(pre_ratio <= pre_ratio_max)
            checks["intraday_anomaly_ok"] = bool(intraday_ratio <= intraday_ratio_max)
            checks["eod_quality_anomaly_ok"] = bool(eod_quality_ratio <= eod_quality_ratio_max)
            checks["eod_risk_anomaly_ok"] = bool(eod_risk_ratio <= eod_risk_ratio_max)
            checks["eod_quality_regime_bucket_ok"] = bool(eod_quality_regime_breach_count == 0)
            checks["eod_risk_regime_bucket_ok"] = bool(eod_risk_regime_breach_count == 0)
            checks["eod_anomaly_ok"] = bool(
                eod_ratio <= eod_ratio_max
                and checks["eod_quality_anomaly_ok"]
                and checks["eod_risk_anomaly_ok"]
                and checks["eod_quality_regime_bucket_ok"]
                and checks["eod_risk_regime_bucket_ok"]
            )
            if not checks["missing_ratio_ok"]:
                alerts.append("slot_missing_ratio_high")
            if not checks["premarket_anomaly_ok"]:
                alerts.append("slot_premarket_anomaly_high")
            if not checks["intraday_anomaly_ok"]:
                alerts.append("slot_intraday_anomaly_high")
            if not checks["eod_quality_anomaly_ok"]:
                alerts.append("slot_eod_quality_anomaly_high")
            if not checks["eod_risk_anomaly_ok"]:
                alerts.append("slot_eod_risk_anomaly_high")
            if not checks["eod_quality_regime_bucket_ok"]:
                alerts.append("slot_eod_quality_regime_bucket_anomaly_high")
            if not checks["eod_risk_regime_bucket_ok"]:
                alerts.append("slot_eod_risk_regime_bucket_anomaly_high")
            if not checks["eod_anomaly_ok"]:
                alerts.append("slot_eod_anomaly_high")
        else:
            alerts.append("insufficient_slot_samples")

        for key in slots:
            expected = int(slots[key]["expected"])
            anomalies = int(slots[key]["anomalies"])
            missing = int(slots[key]["missing"])
            slots[key]["anomaly_ratio"] = _ratio(anomalies, expected)
            slots[key]["missing_ratio"] = _ratio(missing, expected)
        slots["eod"]["quality_anomalies"] = int(eod_quality_anomaly_slots)
        slots["eod"]["risk_anomalies"] = int(eod_risk_anomaly_slots)
        slots["eod"]["quality_anomaly_ratio"] = eod_quality_ratio
        slots["eod"]["risk_anomaly_ratio"] = eod_risk_ratio
        slots["eod"]["regime_buckets"] = eod_regime_stats

        series.reverse()
        return {
            "active": active,
            "window_days": window_days,
            "samples": active_days,
            "min_samples": min_samples,
            "metrics": {
                "expected_slots": total_expected,
                "observed_slots": total_observed,
                "missing_slots": total_missing,
                "anomaly_slots": total_anomalies,
                "missing_ratio": missing_ratio,
                "premarket_anomaly_ratio": pre_ratio,
                "intraday_anomaly_ratio": intraday_ratio,
                "eod_anomaly_ratio": eod_ratio,
                "eod_quality_anomaly_ratio": eod_quality_ratio,
                "eod_risk_anomaly_ratio": eod_risk_ratio,
                "eod_unknown_regime_days": int(eod_unknown_regime_days),
                "eod_quality_regime_bucket_breaches": int(eod_quality_regime_breach_count),
                "eod_risk_regime_bucket_breaches": int(eod_risk_regime_breach_count),
            },
            "thresholds": {
                "ops_slot_missing_ratio_max": missing_ratio_max,
                "ops_slot_premarket_anomaly_ratio_max": pre_ratio_max,
                "ops_slot_intraday_anomaly_ratio_max": intraday_ratio_max,
                "ops_slot_eod_anomaly_ratio_max": eod_ratio_max,
                "ops_slot_eod_quality_anomaly_ratio_max": eod_quality_ratio_max,
                "ops_slot_eod_risk_anomaly_ratio_max": eod_risk_ratio_max,
                "ops_slot_eod_quality_anomaly_ratio_max_by_regime": eod_quality_ratio_by_regime,
                "ops_slot_eod_risk_anomaly_ratio_max_by_regime": eod_risk_ratio_by_regime,
                "ops_slot_source_confidence_floor": source_floor,
                "ops_slot_risk_multiplier_floor": risk_floor,
                "ops_slot_use_live_regime_thresholds": bool(use_live_regime_thresholds),
                "live_regime_thresholds_applied": bool(live_overrides_applied),
            },
            "checks": checks,
            "alerts": alerts,
            "slots": slots,
            "series": series[-10:],
        }

    @staticmethod
    def _ratio(num: float, den: float) -> float:
        n = float(num)
        d = float(den)
        return float(n / d) if d > 0 else 0.0

    @staticmethod
    def _sqlite_table_exists(conn: sqlite3.Connection, table: str) -> bool:
        try:
            with closing(conn.cursor()) as cur:
                cur.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
                    (str(table),),
                )
                return cur.fetchone() is not None
        except Exception:
            return False

    def _csv_plan_summary(self, *, day: str) -> dict[str, Any]:
        path = self.output_dir / "daily" / f"{day}_positions.csv"
        if not path.exists():
            return {"found": False, "path": str(path), "rows": 0, "active_rows": 0, "exposure": 0.0}
        rows = 0
        active_rows = 0
        exposure = 0.0
        try:
            with path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows += 1
                    status = str(row.get("status", "ACTIVE") or "ACTIVE").strip().upper()
                    if status == "ACTIVE":
                        active_rows += 1
                    exposure += abs(self._safe_float(row.get("size_pct", 0.0), 0.0))
        except Exception:
            return {"found": False, "path": str(path), "rows": 0, "active_rows": 0, "exposure": 0.0}
        return {
            "found": True,
            "path": str(path),
            "rows": int(rows),
            "active_rows": int(active_rows),
            "exposure": float(exposure),
        }

    @staticmethod
    def _normalize_broker_symbol(raw: Any) -> str:
        txt = str(raw or "").strip().upper()
        if not txt:
            return ""
        txt = "".join(txt.split())
        allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
        return "".join(ch for ch in txt if ch in allowed)

    @staticmethod
    def _normalize_broker_side(raw: Any) -> str:
        txt = str(raw or "").strip().upper()
        if not txt:
            return ""
        if txt in {"LONG", "BUY", "B", "1", "+1"}:
            return "LONG"
        if txt in {"SHORT", "SELL", "S", "-1"}:
            return "SHORT"
        if txt in {"FLAT", "BOTH", "NET", "NONE", "0", "NEUTRAL"}:
            return "FLAT"
        return ""

    def _canonicalize_broker_snapshot(self, *, payload: dict[str, Any]) -> dict[str, Any]:
        def _finite(v: Any) -> float | None:
            try:
                x = float(v)
            except Exception:
                return None
            if not math.isfinite(x):
                return None
            return float(x)

        source = str(payload.get("source", "")).strip() or "unknown"
        positions_raw = payload.get("positions", [])
        if not isinstance(positions_raw, list):
            positions_raw = []

        positions_out: list[dict[str, Any]] = []
        symbol_noncanonical_count = 0
        side_noncanonical_count = 0
        for row in positions_raw:
            if not isinstance(row, dict):
                continue
            symbol_raw = str(row.get("symbol", "")).strip()
            symbol_norm = self._normalize_broker_symbol(symbol_raw)
            symbol_upper = "".join(symbol_raw.upper().split())
            if symbol_norm and symbol_norm != symbol_upper:
                symbol_noncanonical_count += 1

            side_raw = str(row.get("side", "")).strip()
            side_norm = self._normalize_broker_side(side_raw)
            qty = _finite(row.get("qty", 0.0))
            qty_val = float(qty) if qty is not None else 0.0
            if not side_norm:
                if qty_val > 0.0:
                    side_norm = "LONG"
                elif qty_val < 0.0:
                    side_norm = "SHORT"
                else:
                    side_norm = "FLAT"
            if side_raw and side_norm != side_raw.upper():
                side_noncanonical_count += 1

            qty_abs = abs(qty_val)
            entry_price = _finite(row.get("entry_price"))
            market_price = _finite(row.get("market_price"))
            notional = _finite(row.get("notional"))
            if notional is None:
                px_ref = market_price if market_price is not None else (entry_price if entry_price is not None else 0.0)
                notional = abs(qty_abs * float(px_ref))
            else:
                notional = abs(float(notional))

            positions_out.append(
                {
                    "symbol": symbol_norm,
                    "side": side_norm,
                    "qty": float(qty_abs),
                    "entry_price": float(entry_price) if entry_price is not None else 0.0,
                    "market_price": float(market_price) if market_price is not None else 0.0,
                    "notional": float(notional),
                    "raw_symbol": symbol_raw,
                    "raw_side": side_raw,
                }
            )

        open_positions_raw = _finite(payload.get("open_positions", len(positions_out)))
        open_positions = int(round(open_positions_raw)) if open_positions_raw is not None else int(len(positions_out))
        open_positions = int(max(open_positions, len(positions_out), 0))

        closed_count_raw = _finite(payload.get("closed_count", 0))
        closed_count = int(round(closed_count_raw)) if closed_count_raw is not None else 0
        closed_count = int(max(closed_count, 0))

        closed_pnl_raw = _finite(payload.get("closed_pnl", 0.0))
        closed_pnl = float(closed_pnl_raw) if closed_pnl_raw is not None else 0.0

        return {
            "source": source,
            "open_positions": int(open_positions),
            "closed_count": int(closed_count),
            "closed_pnl": float(closed_pnl),
            "positions": positions_out,
            "normalization": {
                "symbol_noncanonical_count": int(symbol_noncanonical_count),
                "side_noncanonical_count": int(side_noncanonical_count),
                "position_rows": int(len(positions_out)),
            },
        }

    def _canonicalize_system_open_positions(self, *, rows: list[Any]) -> list[dict[str, Any]]:
        def _finite(v: Any) -> float | None:
            try:
                x = float(v)
            except Exception:
                return None
            if not math.isfinite(x):
                return None
            return float(x)

        out: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            symbol = self._normalize_broker_symbol(row.get("symbol", ""))
            if not symbol:
                continue

            side_raw = row.get("side")
            if side_raw is None:
                side_raw = row.get("direction", "")
            side = self._normalize_broker_side(side_raw)

            qty = _finite(row.get("qty"))
            if qty is None:
                qty = _finite(row.get("size_pct"))
            if qty is None:
                qty = _finite(row.get("position_size"))
            qty_val = abs(float(qty)) if qty is not None else 0.0

            if not side:
                if qty is not None:
                    if float(qty) > 0.0:
                        side = "LONG"
                    elif float(qty) < 0.0:
                        side = "SHORT"
                    else:
                        side = "FLAT"
                else:
                    side = "LONG"

            notional = _finite(row.get("notional"))
            if notional is None:
                notional = _finite(row.get("size_pct"))
            if notional is None:
                notional = qty_val
            notional_val = abs(float(notional)) if notional is not None else 0.0

            out.append(
                {
                    "symbol": symbol,
                    "side": side,
                    "qty": float(qty_val),
                    "notional": float(notional_val),
                }
            )
        return out

    def _row_diff_symbol_alias_map(self) -> dict[str, str]:
        val = self.settings.validation if isinstance(self.settings.validation, dict) else {}
        raw = val.get("ops_reconcile_broker_row_diff_symbol_alias_map", {})
        if not isinstance(raw, dict):
            return {}
        out: dict[str, str] = {}
        for k, v in raw.items():
            src = self._normalize_broker_symbol(k)
            dst = self._normalize_broker_symbol(v)
            if src and dst:
                out[src] = dst
        return out

    def _row_diff_side_alias_map(self) -> dict[str, str]:
        val = self.settings.validation if isinstance(self.settings.validation, dict) else {}
        raw = val.get("ops_reconcile_broker_row_diff_side_alias_map", {})
        if not isinstance(raw, dict):
            return {}
        out: dict[str, str] = {}
        for k, v in raw.items():
            src = str(k or "").strip().upper()
            dst = self._normalize_broker_side(v)
            if src and dst:
                out[src] = dst
        return out

    def _compare_position_rows(
        self,
        *,
        broker_rows: list[Any],
        system_rows: list[Any],
        broker_source: str,
    ) -> dict[str, Any]:
        def _finite(v: Any) -> float | None:
            try:
                x = float(v)
            except Exception:
                return None
            if not math.isfinite(x):
                return None
            return float(x)

        symbol_alias_map = self._row_diff_symbol_alias_map()
        side_alias_map = self._row_diff_side_alias_map()

        def _aggregate(rows: list[Any]) -> tuple[dict[str, int], dict[str, float], int, dict[str, int]]:
            counts: dict[str, int] = {}
            notionals: dict[str, float] = {}
            accepted_rows = 0
            stats = {
                "input_rows": 0,
                "alias_hits": 0,
                "symbol_alias_hits": 0,
                "side_alias_hits": 0,
                "unresolved_rows": 0,
            }
            for row in rows:
                if not isinstance(row, dict):
                    continue
                stats["input_rows"] += 1
                symbol = self._normalize_broker_symbol(row.get("symbol", ""))
                symbol_alias_hit = False
                if symbol_alias_map:
                    mapped_symbol = symbol_alias_map.get(symbol, symbol)
                    if mapped_symbol != symbol:
                        symbol = mapped_symbol
                        symbol_alias_hit = True
                        stats["symbol_alias_hits"] += 1
                side_raw = str(row.get("side", "")).strip().upper()
                side = self._normalize_broker_side(row.get("side", ""))
                side_alias_hit = False
                if side_alias_map:
                    side_alias = side_alias_map.get(side_raw, "")
                    if side_alias and ((not side) or (side_alias != side)):
                        side = side_alias
                        side_alias_hit = True
                        stats["side_alias_hits"] += 1
                if symbol_alias_hit or side_alias_hit:
                    stats["alias_hits"] += 1
                if not symbol or not side:
                    stats["unresolved_rows"] += 1
                    continue
                key = f"{symbol}|{side}"
                counts[key] = int(counts.get(key, 0) + 1)

                notional = _finite(row.get("notional"))
                if notional is None:
                    notional = _finite(row.get("qty"))
                weight = abs(float(notional)) if notional is not None else 1.0
                notionals[key] = float(notionals.get(key, 0.0) + weight)
                accepted_rows += 1
            return counts, notionals, accepted_rows, stats

        broker_counts, broker_notionals, broker_rows_used, broker_stats = _aggregate(broker_rows)
        system_counts, system_notionals, system_rows_used, system_stats = _aggregate(system_rows)
        broker_keys = set(broker_counts.keys())
        system_keys = set(system_counts.keys())
        union_keys = broker_keys | system_keys
        union_n = len(union_keys)
        symmetric_diff_n = len(broker_keys ^ system_keys)

        key_mismatch_ratio = self._ratio(symmetric_diff_n, max(1, union_n))
        count_gap = 0.0
        for key in union_keys:
            count_gap += abs(float(broker_counts.get(key, 0) - system_counts.get(key, 0)))
        count_gap_ratio = self._ratio(count_gap, max(1.0, float(sum(system_counts.values()))))

        source_norm = str(broker_source).strip().lower()
        notional_comparable = bool(source_norm.startswith("paper_engine"))
        notional_gap_ratio = 0.0
        if notional_comparable and union_keys:
            total_system_notional = float(sum(system_notionals.values()))
            total_broker_notional = float(sum(broker_notionals.values()))
            denom = max(total_system_notional, total_broker_notional, 1e-9)
            gap = 0.0
            for key in union_keys:
                gap += abs(float(broker_notionals.get(key, 0.0) - system_notionals.get(key, 0.0)))
            notional_gap_ratio = float(gap / denom)

        missing_on_broker = sorted(list(system_keys - broker_keys))[:10]
        extra_on_broker = sorted(list(broker_keys - system_keys))[:10]
        alias_hits = int(broker_stats.get("alias_hits", 0) + system_stats.get("alias_hits", 0))
        symbol_alias_hits = int(
            broker_stats.get("symbol_alias_hits", 0) + system_stats.get("symbol_alias_hits", 0)
        )
        side_alias_hits = int(broker_stats.get("side_alias_hits", 0) + system_stats.get("side_alias_hits", 0))
        input_rows = int(broker_stats.get("input_rows", 0) + system_stats.get("input_rows", 0))
        unresolved_rows = int(
            broker_stats.get("unresolved_rows", 0) + system_stats.get("unresolved_rows", 0)
        )
        compared_rows = int(broker_rows_used + system_rows_used)
        alias_hit_rate = self._ratio(alias_hits, compared_rows) if compared_rows > 0 else 0.0
        unresolved_row_ratio = self._ratio(unresolved_rows, input_rows) if input_rows > 0 else 0.0
        unresolved_key_ratio = float(key_mismatch_ratio)

        return {
            "key_mismatch_ratio": float(key_mismatch_ratio),
            "count_gap_ratio": float(count_gap_ratio),
            "notional_gap_ratio": float(notional_gap_ratio),
            "notional_comparable": bool(notional_comparable),
            "broker_rows": int(broker_rows_used),
            "system_rows": int(system_rows_used),
            "compared_rows": int(compared_rows),
            "missing_on_broker": missing_on_broker,
            "extra_on_broker": extra_on_broker,
            "alias_hits": int(alias_hits),
            "symbol_alias_hits": int(symbol_alias_hits),
            "side_alias_hits": int(side_alias_hits),
            "alias_hit_rate": float(alias_hit_rate),
            "unresolved_rows": int(unresolved_rows),
            "input_rows": int(input_rows),
            "unresolved_row_ratio": float(unresolved_row_ratio),
            "unresolved_keys": int(symmetric_diff_n),
            "union_keys": int(union_n),
            "unresolved_key_ratio": float(unresolved_key_ratio),
        }

    def _write_reconcile_row_diff_artifact(
        self,
        *,
        as_of: date,
        series: list[dict[str, Any]],
        retention_days: int = 30,
        checksum_index_enabled: bool = True,
    ) -> dict[str, Any]:
        policy = self._artifact_governance_profile(
            profile_name="reconcile_row_diff",
            fallback_retention_days=retention_days,
            fallback_checksum_index_enabled=checksum_index_enabled,
        )
        keep_days = int(policy.get("retention_days", max(1, int(retention_days))))
        index_enabled = bool(policy.get("checksum_index_enabled", checksum_index_enabled))
        review_dir = self.output_dir / "review"

        def _governance() -> dict[str, Any]:
            return self._apply_artifact_governance(
                as_of=as_of,
                review_dir=review_dir,
                profile_name="reconcile_row_diff",
                fallback_retention_days=keep_days,
                fallback_checksum_index_enabled=index_enabled,
            )

        missing_counts: dict[str, int] = {}
        extra_counts: dict[str, int] = {}
        sample_rows = 0
        breach_rows = 0

        for row in series:
            if not isinstance(row, dict):
                continue
            broker = row.get("broker", {}) if isinstance(row.get("broker", {}), dict) else {}
            row_diff = broker.get("row_diff", {}) if isinstance(broker.get("row_diff", {}), dict) else {}
            if not bool(row_diff.get("active", False)):
                continue
            if bool(row_diff.get("skipped", True)):
                continue
            sample_rows += 1
            if bool(row_diff.get("breached", False)):
                breach_rows += 1
            for key in row_diff.get("missing_on_broker", []):
                txt = str(key or "").strip()
                if txt:
                    missing_counts[txt] = int(missing_counts.get(txt, 0) + 1)
            for key in row_diff.get("extra_on_broker", []):
                txt = str(key or "").strip()
                if txt:
                    extra_counts[txt] = int(extra_counts.get(txt, 0) + 1)

        if sample_rows <= 0:
            governance = _governance()
            return {
                "written": False,
                "json": "",
                "md": "",
                "sample_rows": 0,
                "breach_rows": 0,
                "retention_days": int(governance.get("retention_days", keep_days)),
                "rotated_out_count": int(governance.get("rotated_out_count", 0)),
                "rotated_out_dates": [str(x) for x in governance.get("rotated_out_dates", [])],
                "rotation_failed": bool(governance.get("rotation_failed", False)),
                "checksum_index_enabled": bool(governance.get("checksum_index_enabled", index_enabled)),
                "checksum_index_written": bool(governance.get("checksum_index_written", False)),
                "checksum_index_path": str(governance.get("checksum_index_path", "")),
                "checksum_index_entries": int(governance.get("checksum_index_entries", 0)),
                "checksum_index_failed": bool(governance.get("checksum_index_failed", False)),
                "reason": "no_row_diff_samples",
            }
        if breach_rows <= 0 and (not missing_counts) and (not extra_counts):
            governance = _governance()
            return {
                "written": False,
                "json": "",
                "md": "",
                "sample_rows": int(sample_rows),
                "breach_rows": int(breach_rows),
                "retention_days": int(governance.get("retention_days", keep_days)),
                "rotated_out_count": int(governance.get("rotated_out_count", 0)),
                "rotated_out_dates": [str(x) for x in governance.get("rotated_out_dates", [])],
                "rotation_failed": bool(governance.get("rotation_failed", False)),
                "checksum_index_enabled": bool(governance.get("checksum_index_enabled", index_enabled)),
                "checksum_index_written": bool(governance.get("checksum_index_written", False)),
                "checksum_index_path": str(governance.get("checksum_index_path", "")),
                "checksum_index_entries": int(governance.get("checksum_index_entries", 0)),
                "checksum_index_failed": bool(governance.get("checksum_index_failed", False)),
                "reason": "no_row_diff_breach",
            }

        def _top_counts(bucket: dict[str, int], n: int = 20) -> list[dict[str, Any]]:
            items = sorted(bucket.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
            return [{"key": str(k), "count": int(v)} for k, v in items[:n]]

        top_missing = _top_counts(missing_counts)
        top_extra = _top_counts(extra_counts)
        hints: list[str] = []
        for item in top_missing[:5]:
            key = str(item.get("key", ""))
            if "|" in key:
                sym, side = key.split("|", 1)
            else:
                sym, side = key, "UNKNOWN"
            hints.append(f"missing_on_broker: {sym}|{side} -> 检查 symbol/side 映射及仓位同步顺序")
        for item in top_extra[:5]:
            key = str(item.get("key", ""))
            if "|" in key:
                sym, side = key.split("|", 1)
            else:
                sym, side = key, "UNKNOWN"
            hints.append(f"extra_on_broker: {sym}|{side} -> 检查 broker 持仓口径(净仓/逐腿)与去重规则")

        payload = {
            "date": as_of.isoformat(),
            "sample_rows": int(sample_rows),
            "breach_rows": int(breach_rows),
            "top_missing_on_broker": top_missing,
            "top_extra_on_broker": top_extra,
            "hints": hints[:10],
        }
        json_path = review_dir / f"{as_of.isoformat()}_reconcile_row_diff.json"
        md_path = review_dir / f"{as_of.isoformat()}_reconcile_row_diff.md"

        lines: list[str] = []
        lines.append(f"# Reconcile Row-Diff Drilldown | {as_of.isoformat()}")
        lines.append("")
        lines.append(f"- sample_rows: `{int(sample_rows)}`")
        lines.append(f"- breach_rows: `{int(breach_rows)}`")
        lines.append("")
        lines.append("## Top Missing On Broker")
        if top_missing:
            for item in top_missing:
                lines.append(f"- `{item['key']}` x `{item['count']}`")
        else:
            lines.append("- NONE")
        lines.append("")
        lines.append("## Top Extra On Broker")
        if top_extra:
            for item in top_extra:
                lines.append(f"- `{item['key']}` x `{item['count']}`")
        else:
            lines.append("- NONE")
        lines.append("")
        lines.append("## Hints")
        if hints:
            for txt in hints[:10]:
                lines.append(f"- {txt}")
        else:
            lines.append("- NONE")

        try:
            write_json(json_path, payload)
            write_markdown(md_path, "\n".join(lines) + "\n")
        except Exception as exc:
            return {
                "written": False,
                "json": "",
                "md": "",
                "sample_rows": int(sample_rows),
                "breach_rows": int(breach_rows),
                "retention_days": int(keep_days),
                "rotated_out_count": 0,
                "rotated_out_dates": [],
                "rotation_failed": False,
                "checksum_index_enabled": bool(index_enabled),
                "checksum_index_written": False,
                "checksum_index_path": "",
                "checksum_index_entries": 0,
                "checksum_index_failed": False,
                "reason": f"write_failed:{type(exc).__name__}:{exc}",
            }

        governance = _governance()
        return {
            "written": True,
            "json": str(json_path),
            "md": str(md_path),
            "sample_rows": int(sample_rows),
            "breach_rows": int(breach_rows),
            "retention_days": int(governance.get("retention_days", keep_days)),
            "rotated_out_count": int(governance.get("rotated_out_count", 0)),
            "rotated_out_dates": [str(x) for x in governance.get("rotated_out_dates", [])],
            "rotation_failed": bool(governance.get("rotation_failed", False)),
            "checksum_index_enabled": bool(governance.get("checksum_index_enabled", index_enabled)),
            "checksum_index_written": bool(governance.get("checksum_index_written", False)),
            "checksum_index_path": str(governance.get("checksum_index_path", "")),
            "checksum_index_entries": int(governance.get("checksum_index_entries", 0)),
            "checksum_index_failed": bool(governance.get("checksum_index_failed", False)),
            "reason": str(governance.get("reason", "")),
        }

    def _write_temporal_autofix_artifact(
        self,
        *,
        as_of: date,
        series: list[dict[str, Any]],
        retention_days: int = 30,
        checksum_index_enabled: bool = True,
    ) -> dict[str, Any]:
        policy = self._artifact_governance_profile(
            profile_name="temporal_autofix_patch",
            fallback_retention_days=retention_days,
            fallback_checksum_index_enabled=checksum_index_enabled,
        )
        keep_days = int(policy.get("retention_days", max(1, int(retention_days))))
        index_enabled = bool(policy.get("checksum_index_enabled", checksum_index_enabled))
        events: list[dict[str, Any]] = []
        reason_counts: dict[str, int] = {}
        applied_count = 0
        failed_count = 0
        skipped_count = 0

        for row in series:
            if not isinstance(row, dict):
                continue
            autofix = row.get("autofix", {}) if isinstance(row.get("autofix", {}), dict) else {}
            attempted = bool(autofix.get("attempted", False))
            applied = bool(autofix.get("applied", False))
            reason = str(autofix.get("reason", "")).strip()
            error = str(autofix.get("error", "")).strip()
            if not attempted and not applied and (not reason) and (not error):
                continue

            patch_delta = autofix.get("patch_delta", {}) if isinstance(autofix.get("patch_delta", {}), dict) else {}
            strict_delta = autofix.get("strict_delta", {}) if isinstance(autofix.get("strict_delta", {}), dict) else {}
            event = {
                "date": str(row.get("date", "")),
                "run_type": str(row.get("run_type", "")),
                "run_id": str(row.get("run_id", "")),
                "manifest_path": str(row.get("manifest_path", "")),
                "attempted": bool(attempted),
                "applied": bool(applied),
                "reason": reason,
                "error": error,
                "patched_fields": list(autofix.get("patched_fields", [])),
                "strict_patched": bool(autofix.get("strict_patched", False)),
                "summary_path": str(autofix.get("summary_path", "")),
                "patch_delta": patch_delta,
                "strict_delta": strict_delta,
            }
            events.append(event)
            if applied:
                applied_count += 1
            elif error:
                failed_count += 1
            else:
                skipped_count += 1
            bucket = reason if reason else ("applied" if applied else ("failed" if error else "unknown"))
            reason_counts[bucket] = int(reason_counts.get(bucket, 0) + 1)

        total_events = len(events)
        if total_events <= 0:
            return {
                "written": False,
                "json": "",
                "md": "",
                "total_events": 0,
                "applied_count": 0,
                "failed_count": 0,
                "skipped_count": 0,
                "reason": "no_autofix_activity",
            }

        top_reasons = [
            {"reason": str(k), "count": int(v)}
            for k, v in sorted(reason_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[:20]
        ]
        payload = {
            "date": as_of.isoformat(),
            "total_events": int(total_events),
            "applied_count": int(applied_count),
            "failed_count": int(failed_count),
            "skipped_count": int(skipped_count),
            "top_reasons": top_reasons,
            "events": events[:50],
        }
        review_dir = self.output_dir / "review"
        json_path = review_dir / f"{as_of.isoformat()}_temporal_autofix_patch.json"
        md_path = review_dir / f"{as_of.isoformat()}_temporal_autofix_patch.md"

        lines: list[str] = []
        lines.append(f"# Temporal Autofix Patch Report | {as_of.isoformat()}")
        lines.append("")
        lines.append(f"- total_events: `{int(total_events)}`")
        lines.append(f"- applied_count: `{int(applied_count)}`")
        lines.append(f"- failed_count: `{int(failed_count)}`")
        lines.append(f"- skipped_count: `{int(skipped_count)}`")
        lines.append("")
        lines.append("## Top Reasons")
        if top_reasons:
            for item in top_reasons:
                lines.append(f"- `{item['reason']}` x `{item['count']}`")
        else:
            lines.append("- NONE")
        lines.append("")
        lines.append("## Event Trace")
        for item in events[:30]:
            lines.append(
                "- "
                + f"{item['date']} {item['run_type']}:{item['run_id']} "
                + f"attempted={item['attempted']} applied={item['applied']} reason={item['reason'] or 'N/A'}"
            )
            lines.append(f"  - manifest: `{item['manifest_path'] or 'N/A'}`")
            lines.append(f"  - summary: `{item['summary_path'] or 'N/A'}`")
            patch_delta = item.get("patch_delta", {}) if isinstance(item.get("patch_delta", {}), dict) else {}
            if patch_delta:
                for key in sorted(patch_delta.keys()):
                    delta = patch_delta.get(key, {}) if isinstance(patch_delta.get(key, {}), dict) else {}
                    before = str(delta.get("before", ""))
                    after = str(delta.get("after", ""))
                    source = str(delta.get("source", ""))
                    lines.append(
                        f"  - patch `{key}`: `{before or 'EMPTY'}` -> `{after}`"
                        + (f" (source={source})" if source else "")
                    )
            strict_delta = item.get("strict_delta", {}) if isinstance(item.get("strict_delta", {}), dict) else {}
            if strict_delta:
                lines.append(
                    "  - strict_cutoff: "
                    + f"`{bool(strict_delta.get('before', False))}` -> `{bool(strict_delta.get('after', False))}`"
                    + (
                        f" (source={str(strict_delta.get('source', '')).strip()})"
                        if str(strict_delta.get("source", "")).strip()
                        else ""
                    )
                )
            if str(item.get("error", "")).strip():
                lines.append(f"  - error: `{str(item.get('error', ''))}`")
        if total_events > 30:
            lines.append("")
            lines.append(f"- truncated: `{total_events - 30}` more events in JSON artifact")

        try:
            write_json(json_path, payload)
            write_markdown(md_path, "\n".join(lines) + "\n")
        except Exception as exc:
            return {
                "written": False,
                "json": "",
                "md": "",
                "total_events": int(total_events),
                "applied_count": int(applied_count),
                "failed_count": int(failed_count),
                "skipped_count": int(skipped_count),
                "retention_days": int(keep_days),
                "rotated_out_count": 0,
                "rotated_out_dates": [],
                "rotation_failed": False,
                "checksum_index_enabled": bool(index_enabled),
                "checksum_index_written": False,
                "checksum_index_path": "",
                "checksum_index_entries": 0,
                "checksum_index_failed": False,
                "reason": f"write_failed:{type(exc).__name__}:{exc}",
            }

        governance = self._apply_artifact_governance(
            as_of=as_of,
            review_dir=review_dir,
            profile_name="temporal_autofix_patch",
            fallback_retention_days=keep_days,
            fallback_checksum_index_enabled=index_enabled,
        )

        return {
            "written": True,
            "json": str(json_path),
            "md": str(md_path),
            "total_events": int(total_events),
            "applied_count": int(applied_count),
            "failed_count": int(failed_count),
            "skipped_count": int(skipped_count),
            "retention_days": int(governance.get("retention_days", keep_days)),
            "rotated_out_count": int(governance.get("rotated_out_count", 0)),
            "rotated_out_dates": [str(x) for x in governance.get("rotated_out_dates", [])],
            "rotation_failed": bool(governance.get("rotation_failed", False)),
            "checksum_index_enabled": bool(governance.get("checksum_index_enabled", index_enabled)),
            "checksum_index_written": bool(governance.get("checksum_index_written", False)),
            "checksum_index_path": str(governance.get("checksum_index_path", "")),
            "checksum_index_entries": int(governance.get("checksum_index_entries", 0)),
            "checksum_index_failed": bool(governance.get("checksum_index_failed", False)),
            "reason": str(governance.get("reason", "")),
        }

    def _lint_broker_snapshot_contract(
        self,
        *,
        payload: dict[str, Any],
        closed_pnl_abs_hard_max: float,
        qty_abs_hard_max: float,
        notional_abs_hard_max: float,
        price_abs_hard_max: float,
    ) -> dict[str, Any]:
        schema_errors: list[str] = []
        numeric_errors: list[str] = []
        symbol_errors: list[str] = []
        symbol_noncanonical_count = 0
        side_noncanonical_count = 0

        def _finite(v: Any) -> float | None:
            try:
                x = float(v)
            except Exception:
                return None
            if not math.isfinite(x):
                return None
            return float(x)

        source = str(payload.get("source", "")).strip()
        if not source:
            schema_errors.append("source_missing")

        positions_raw = payload.get("positions", [])
        if not isinstance(positions_raw, list):
            schema_errors.append("positions_not_list")
            positions: list[Any] = []
        else:
            positions = positions_raw

        open_positions = _finite(payload.get("open_positions", len(positions)))
        closed_count = _finite(payload.get("closed_count", 0))
        closed_pnl = _finite(payload.get("closed_pnl", 0.0))
        if open_positions is None:
            numeric_errors.append("open_positions_not_numeric")
            open_positions_int = 0
        else:
            open_positions_int = int(round(open_positions))
            if open_positions_int < 0:
                numeric_errors.append("open_positions_negative")
        if closed_count is None:
            numeric_errors.append("closed_count_not_numeric")
        else:
            if int(round(closed_count)) < 0:
                numeric_errors.append("closed_count_negative")
        if closed_pnl is None:
            numeric_errors.append("closed_pnl_not_numeric")
        else:
            if abs(closed_pnl) > float(max(0.0, closed_pnl_abs_hard_max)):
                numeric_errors.append("closed_pnl_abs_hard_breach")

        if open_positions_int < len(positions):
            numeric_errors.append("open_positions_less_than_position_rows")

        for idx, row in enumerate(positions):
            if not isinstance(row, dict):
                schema_errors.append(f"position_row_not_object:{idx}")
                continue

            symbol_raw = str(row.get("symbol", "")).strip()
            symbol_norm = self._normalize_broker_symbol(symbol_raw)
            if not symbol_norm:
                symbol_errors.append(f"symbol_missing:{idx}")
            else:
                symbol_upper = "".join(symbol_raw.upper().split())
                if symbol_norm != symbol_upper:
                    symbol_noncanonical_count += 1

            side_raw = str(row.get("side", "")).strip()
            side = self._normalize_broker_side(side_raw)
            if side_raw and not side:
                schema_errors.append(f"side_invalid:{idx}")
            elif side_raw and side != side_raw.upper():
                side_noncanonical_count += 1

            qty = _finite(row.get("qty", 0.0))
            if qty is None:
                numeric_errors.append(f"qty_not_numeric:{idx}")
            else:
                if abs(qty) > float(max(0.0, qty_abs_hard_max)):
                    numeric_errors.append(f"qty_abs_hard_breach:{idx}")

            notional = _finite(row.get("notional", 0.0))
            if notional is None:
                numeric_errors.append(f"notional_not_numeric:{idx}")
            else:
                if abs(notional) > float(max(0.0, notional_abs_hard_max)):
                    numeric_errors.append(f"notional_abs_hard_breach:{idx}")

            for px_key in ("entry_price", "market_price"):
                raw_px = row.get(px_key)
                if raw_px is None:
                    continue
                txt_px = str(raw_px).strip()
                if txt_px == "":
                    continue
                px = _finite(raw_px)
                if px is None:
                    numeric_errors.append(f"{px_key}_not_numeric:{idx}")
                    continue
                if px < 0.0:
                    numeric_errors.append(f"{px_key}_negative:{idx}")
                if px > float(max(0.0, price_abs_hard_max)):
                    numeric_errors.append(f"{px_key}_abs_hard_breach:{idx}")

        return {
            "schema_ok": bool(len(schema_errors) == 0),
            "numeric_ok": bool(len(numeric_errors) == 0),
            "symbol_ok": bool(len(symbol_errors) == 0),
            "schema_errors": schema_errors[:20],
            "numeric_errors": numeric_errors[:20],
            "symbol_errors": symbol_errors[:20],
            "symbol_noncanonical_count": int(symbol_noncanonical_count),
            "side_noncanonical_count": int(side_noncanonical_count),
            "position_rows": int(len(positions)),
            "open_positions": int(open_positions_int),
        }

    def _reconcile_drift_metrics(self, *, as_of: date) -> dict[str, Any]:
        val = self.settings.validation if isinstance(self.settings.validation, dict) else {}
        window_days = max(1, int(val.get("ops_reconcile_window_days", 7)))
        min_samples = max(1, int(val.get("ops_reconcile_min_samples", 3)))
        missing_ratio_max = self._safe_float(val.get("ops_reconcile_missing_ratio_max", 0.35), 0.35)
        plan_gap_ratio_max = self._safe_float(val.get("ops_reconcile_plan_gap_ratio_max", 0.10), 0.10)
        closed_count_gap_ratio_max = self._safe_float(val.get("ops_reconcile_closed_count_gap_ratio_max", 0.10), 0.10)
        closed_pnl_gap_abs_max = abs(self._safe_float(val.get("ops_reconcile_closed_pnl_gap_abs_max", 0.001), 0.001))
        open_gap_ratio_max = self._safe_float(val.get("ops_reconcile_open_gap_ratio_max", 0.25), 0.25)
        broker_gap_ratio_max = self._safe_float(val.get("ops_reconcile_broker_gap_ratio_max", 0.10), 0.10)
        broker_pnl_gap_abs_max = abs(
            self._safe_float(val.get("ops_reconcile_broker_pnl_gap_abs_max", closed_pnl_gap_abs_max), closed_pnl_gap_abs_max)
        )
        broker_missing_ratio_max = self._safe_float(val.get("ops_reconcile_broker_missing_ratio_max", 0.50), 0.50)
        broker_contract_schema_invalid_ratio_max = self._safe_float(
            val.get("ops_reconcile_broker_contract_schema_invalid_ratio_max", 0.10),
            0.10,
        )
        broker_contract_numeric_invalid_ratio_max = self._safe_float(
            val.get("ops_reconcile_broker_contract_numeric_invalid_ratio_max", 0.10),
            0.10,
        )
        broker_contract_symbol_invalid_ratio_max = self._safe_float(
            val.get("ops_reconcile_broker_contract_symbol_invalid_ratio_max", 0.10),
            0.10,
        )
        broker_contract_symbol_noncanonical_ratio_max = self._safe_float(
            val.get("ops_reconcile_broker_contract_symbol_noncanonical_ratio_max", 0.40),
            0.40,
        )
        broker_closed_pnl_abs_hard_max = abs(
            self._safe_float(val.get("ops_reconcile_broker_closed_pnl_abs_hard_max", 1e9), 1e9)
        )
        broker_position_qty_abs_hard_max = abs(
            self._safe_float(val.get("ops_reconcile_broker_position_qty_abs_hard_max", 1e9), 1e9)
        )
        broker_position_notional_abs_hard_max = abs(
            self._safe_float(val.get("ops_reconcile_broker_position_notional_abs_hard_max", 1e10), 1e10)
        )
        broker_price_abs_hard_max = abs(
            self._safe_float(val.get("ops_reconcile_broker_price_abs_hard_max", 1e8), 1e8)
        )
        broker_row_diff_min_samples = max(1, int(val.get("ops_reconcile_broker_row_diff_min_samples", 1)))
        broker_row_diff_breach_ratio_max = self._safe_float(
            val.get("ops_reconcile_broker_row_diff_breach_ratio_max", 0.20),
            0.20,
        )
        broker_row_diff_key_mismatch_max = self._safe_float(
            val.get("ops_reconcile_broker_row_diff_key_mismatch_max", 0.25),
            0.25,
        )
        broker_row_diff_count_gap_max = self._safe_float(
            val.get("ops_reconcile_broker_row_diff_count_gap_max", 0.25),
            0.25,
        )
        broker_row_diff_notional_gap_max = self._safe_float(
            val.get("ops_reconcile_broker_row_diff_notional_gap_max", 0.50),
            0.50,
        )
        broker_row_diff_alias_monitor_enabled = bool(
            val.get("ops_reconcile_broker_row_diff_alias_monitor_enabled", True)
        )
        broker_row_diff_alias_hit_rate_min = self._safe_float(
            val.get("ops_reconcile_broker_row_diff_alias_hit_rate_min", 0.0),
            0.0,
        )
        broker_row_diff_unresolved_key_ratio_max = self._safe_float(
            val.get("ops_reconcile_broker_row_diff_unresolved_key_ratio_max", 0.30),
            0.30,
        )
        broker_row_diff_asof_only = bool(val.get("ops_reconcile_broker_row_diff_asof_only", True))
        row_diff_artifact_policy = self._artifact_governance_profile(
            profile_name="reconcile_row_diff",
            fallback_retention_days=max(
                1,
                int(val.get("ops_reconcile_broker_row_diff_artifact_retention_days", 30)),
            ),
            fallback_checksum_index_enabled=self._safe_bool(
                val.get("ops_reconcile_broker_row_diff_artifact_checksum_index_enabled", True),
                True,
            ),
        )
        broker_row_diff_artifact_retention_days = int(row_diff_artifact_policy.get("retention_days", 30))
        broker_row_diff_artifact_checksum_index_enabled = bool(
            row_diff_artifact_policy.get("checksum_index_enabled", True)
        )
        broker_row_diff_symbol_alias_size = len(self._row_diff_symbol_alias_map())
        broker_row_diff_side_alias_size = len(self._row_diff_side_alias_map())
        require_broker_snapshot = bool(val.get("ops_reconcile_require_broker_snapshot", False))
        emit_canonical_view = bool(val.get("ops_reconcile_broker_contract_emit_canonical_view", True))
        canonical_dir_raw = str(
            val.get("ops_reconcile_broker_contract_canonical_dir", "artifacts/broker_snapshot_canonical")
        ).strip()
        if canonical_dir_raw:
            canonical_dir_path = Path(canonical_dir_raw)
            if not canonical_dir_path.is_absolute():
                canonical_dir_path = self.output_dir / canonical_dir_path
        else:
            canonical_dir_path = self.output_dir / "artifacts" / "broker_snapshot_canonical"

        db_path = self._effective_sqlite_path()
        conn: sqlite3.Connection | None = None
        has_latest_positions = False
        has_executed_plans = False
        if db_path.exists():
            try:
                conn = sqlite3.connect(db_path)
                has_latest_positions = self._sqlite_table_exists(conn, "latest_positions")
                has_executed_plans = self._sqlite_table_exists(conn, "executed_plans")
            except Exception:
                conn = None

        series: list[dict[str, Any]] = []
        samples = 0
        missing_days = 0
        plan_gap_breach_days = 0
        closed_count_gap_breach_days = 0
        closed_pnl_gap_breach_days = 0
        open_gap_breach_days = 0
        open_gap_samples = 0
        total_plan_gap = 0.0
        total_closed_count_gap = 0.0
        total_closed_pnl_gap = 0.0
        total_open_gap = 0.0
        broker_expected_days = 0
        broker_missing_days = 0
        broker_samples = 0
        broker_count_breach_days = 0
        broker_pnl_breach_days = 0
        broker_contract_schema_invalid_days = 0
        broker_contract_numeric_invalid_days = 0
        broker_contract_symbol_invalid_days = 0
        broker_symbol_noncanonical_total = 0
        broker_side_noncanonical_total = 0
        broker_symbol_rows_total = 0
        broker_canonical_eligible_days = 0
        broker_canonical_written_days = 0
        broker_canonical_write_fail_days = 0
        broker_canonical_symbol_noncanonical_total = 0
        broker_canonical_side_noncanonical_total = 0
        broker_row_diff_samples = 0
        broker_row_diff_breach_days = 0
        broker_row_diff_skipped_days = 0
        broker_row_diff_key_mismatch_total = 0.0
        broker_row_diff_count_gap_total = 0.0
        broker_row_diff_notional_gap_total = 0.0
        broker_row_diff_notional_samples = 0
        broker_row_diff_canonical_preferred_days = 0
        broker_row_diff_alias_hits_total = 0
        broker_row_diff_symbol_alias_hits_total = 0
        broker_row_diff_side_alias_hits_total = 0
        broker_row_diff_compared_rows_total = 0
        broker_row_diff_unresolved_rows_total = 0
        broker_row_diff_input_rows_total = 0
        broker_row_diff_unresolved_keys_total = 0
        broker_row_diff_union_keys_total = 0
        total_broker_count_gap = 0.0
        total_broker_pnl_gap = 0.0

        manifest_dir = self.output_dir / "artifacts" / "manifests"
        paper_state = self.load_json_safely(self.output_dir / "artifacts" / "paper_positions_open.json")
        paper_state_as_of = str(paper_state.get("as_of", "")).strip()
        paper_positions = paper_state.get("positions", []) if isinstance(paper_state.get("positions", []), list) else []
        broker_dir = self.output_dir / "artifacts" / "broker_snapshot"
        broker_exists_by_day: dict[str, bool] = {}
        broker_any_exists = False
        for i in range(window_days):
            day = as_of - timedelta(days=i)
            dstr = day.isoformat()
            exists = (broker_dir / f"{dstr}.json").exists()
            broker_exists_by_day[dstr] = bool(exists)
            if exists:
                broker_any_exists = True
        broker_monitor_enabled = bool(require_broker_snapshot or broker_any_exists)

        for i in range(window_days):
            day = as_of - timedelta(days=i)
            dstr = day.isoformat()
            manifest = self.load_json_safely(manifest_dir / f"eod_{dstr}.json")
            if not manifest:
                continue

            metrics = manifest.get("metrics", {}) if isinstance(manifest.get("metrics", {}), dict) else {}
            required_metric_keys = {"plans", "closed_trades", "closed_pnl", "open_positions"}
            if not required_metric_keys.issubset(set(metrics.keys())):
                continue

            samples += 1
            manifest_plans = int(self._safe_float(metrics.get("plans", 0), 0.0))
            manifest_closed_count = int(self._safe_float(metrics.get("closed_trades", 0), 0.0))
            manifest_closed_pnl = self._safe_float(metrics.get("closed_pnl", 0.0), 0.0)
            manifest_open_count = int(self._safe_float(metrics.get("open_positions", 0), 0.0))

            csv_summary = self._csv_plan_summary(day=dstr)
            plan_db_count: int | None = None
            closed_db_count: int | None = None
            closed_db_pnl: float | None = None
            day_missing = False
            day_alerts: list[str] = []
            plan_gap_breached = False
            closed_count_gap_breached = False
            closed_pnl_gap_breached = False
            open_gap_breached = False

            if conn is not None and has_latest_positions:
                try:
                    with closing(conn.cursor()) as cur:
                        cur.execute(
                            "SELECT COUNT(*) FROM latest_positions WHERE date = ?",
                            (dstr,),
                        )
                        row = cur.fetchone()
                        plan_db_count = int(row[0] if row else 0)
                except Exception:
                    plan_db_count = None
            if plan_db_count is None:
                if manifest_plans <= 0:
                    plan_db_count = 0
                else:
                    day_missing = True
                    day_alerts.append("latest_positions_missing")

            if conn is not None and has_executed_plans:
                try:
                    with closing(conn.cursor()) as cur:
                        cur.execute(
                            "SELECT COUNT(*), COALESCE(SUM(pnl), 0.0) FROM executed_plans "
                            "WHERE date = ? AND (status = 'CLOSED' OR status IS NULL)",
                            (dstr,),
                        )
                        row = cur.fetchone()
                        closed_db_count = int(row[0] if row else 0)
                        closed_db_pnl = self._safe_float(row[1] if row else 0.0, 0.0)
                except Exception:
                    closed_db_count = None
                    closed_db_pnl = None
            if closed_db_count is None:
                if manifest_closed_count <= 0 and abs(manifest_closed_pnl) <= 1e-12:
                    closed_db_count = 0
                    closed_db_pnl = 0.0
                else:
                    day_missing = True
                    day_alerts.append("executed_plans_missing")

            if not bool(csv_summary.get("found", False)):
                day_missing = True
                day_alerts.append("positions_csv_missing")

            plan_gap_ratio = 0.0
            closed_count_gap_ratio = 0.0
            closed_pnl_gap_abs = 0.0
            open_gap_ratio = 0.0
            broker_count_gap_ratio = 0.0
            broker_pnl_gap_abs = 0.0
            broker_found = False
            broker_row_diff: dict[str, Any] = {
                "active": False,
                "skipped": True,
                "reason": "",
                "source": "",
                "key_mismatch_ratio": 0.0,
                "count_gap_ratio": 0.0,
                "notional_gap_ratio": 0.0,
                "notional_comparable": False,
                "broker_rows": 0,
                "system_rows": 0,
                "alias_hits": 0,
                "symbol_alias_hits": 0,
                "side_alias_hits": 0,
                "alias_hit_rate": 0.0,
                "unresolved_rows": 0,
                "input_rows": 0,
                "unresolved_row_ratio": 0.0,
                "unresolved_keys": 0,
                "union_keys": 0,
                "unresolved_key_ratio": 0.0,
                "missing_on_broker": [],
                "extra_on_broker": [],
                "breached": False,
            }
            broker_contract_lint: dict[str, Any] = {
                "schema_ok": True,
                "numeric_ok": True,
                "symbol_ok": True,
                "schema_errors": [],
                "numeric_errors": [],
                "symbol_errors": [],
                "symbol_noncanonical_count": 0,
                "side_noncanonical_count": 0,
                "position_rows": 0,
            }
            canonical_view: dict[str, Any] = {
                "enabled": bool(emit_canonical_view),
                "eligible": False,
                "written": False,
                "write_error": "",
                "path": "",
                "symbol_noncanonical_count": 0,
                "side_noncanonical_count": 0,
                "position_rows": 0,
            }
            canonical_payload: dict[str, Any] | None = None

            if plan_db_count is not None:
                plan_gap_ratio = self._ratio(abs(plan_db_count - manifest_plans), max(1, manifest_plans))
                total_plan_gap += plan_gap_ratio
                if plan_gap_ratio > plan_gap_ratio_max:
                    plan_gap_breached = True
                    day_alerts.append("plan_count_gap_high")
                if bool(csv_summary.get("found", False)):
                    csv_count = int(self._safe_float(csv_summary.get("rows", 0), 0.0))
                    csv_db_gap = self._ratio(abs(csv_count - plan_db_count), max(1, plan_db_count))
                    if csv_db_gap > plan_gap_ratio_max:
                        plan_gap_breached = True
                        day_alerts.append("plan_csv_db_gap_high")

            if closed_db_count is not None and closed_db_pnl is not None:
                closed_count_gap_ratio = self._ratio(
                    abs(closed_db_count - manifest_closed_count),
                    max(1, manifest_closed_count),
                )
                closed_pnl_gap_abs = abs(closed_db_pnl - manifest_closed_pnl)
                total_closed_count_gap += closed_count_gap_ratio
                total_closed_pnl_gap += closed_pnl_gap_abs
                if closed_count_gap_ratio > closed_count_gap_ratio_max:
                    closed_count_gap_breached = True
                    day_alerts.append("closed_count_gap_high")
                if closed_pnl_gap_abs > closed_pnl_gap_abs_max:
                    closed_pnl_gap_breached = True
                    day_alerts.append("closed_pnl_gap_high")

            if day == as_of and paper_state_as_of == dstr:
                open_gap_samples += 1
                open_gap_ratio = self._ratio(abs(len(paper_positions) - manifest_open_count), max(1, manifest_open_count))
                total_open_gap += open_gap_ratio
                if open_gap_ratio > open_gap_ratio_max:
                    open_gap_breached = True
                    day_alerts.append("open_count_gap_high")
            elif day == as_of:
                if manifest_open_count <= 0:
                    open_gap_samples += 1
                    open_gap_ratio = 0.0
                else:
                    day_missing = True
                    day_alerts.append("paper_state_missing_or_stale")

            if broker_monitor_enabled:
                broker_expected_days += 1
                broker_payload = self.load_json_safely(broker_dir / f"{dstr}.json")
                if broker_payload:
                    broker_found = True
                    broker_samples += 1
                    broker_contract_lint = self._lint_broker_snapshot_contract(
                        payload=broker_payload,
                        closed_pnl_abs_hard_max=broker_closed_pnl_abs_hard_max,
                        qty_abs_hard_max=broker_position_qty_abs_hard_max,
                        notional_abs_hard_max=broker_position_notional_abs_hard_max,
                        price_abs_hard_max=broker_price_abs_hard_max,
                    )
                    broker_symbol_noncanonical_total += int(broker_contract_lint.get("symbol_noncanonical_count", 0))
                    broker_side_noncanonical_total += int(broker_contract_lint.get("side_noncanonical_count", 0))
                    broker_symbol_rows_total += int(broker_contract_lint.get("position_rows", 0))
                    if not bool(broker_contract_lint.get("schema_ok", True)):
                        broker_contract_schema_invalid_days += 1
                        day_alerts.append("broker_contract_schema_invalid")
                    if not bool(broker_contract_lint.get("numeric_ok", True)):
                        broker_contract_numeric_invalid_days += 1
                        day_alerts.append("broker_contract_numeric_invalid")
                    if not bool(broker_contract_lint.get("symbol_ok", True)):
                        broker_contract_symbol_invalid_days += 1
                        day_alerts.append("broker_contract_symbol_invalid")
                    canonical_eligible = bool(
                        broker_contract_lint.get("schema_ok", True)
                        and broker_contract_lint.get("numeric_ok", True)
                        and broker_contract_lint.get("symbol_ok", True)
                    )
                    if canonical_eligible:
                        broker_canonical_eligible_days += 1
                    canonical_view["eligible"] = bool(canonical_eligible)
                    if bool(emit_canonical_view) and canonical_eligible:
                        canonical_payload = self._canonicalize_broker_snapshot(payload=broker_payload)
                        canonical_view["symbol_noncanonical_count"] = int(
                            (canonical_payload.get("normalization", {}) or {}).get("symbol_noncanonical_count", 0)
                        )
                        canonical_view["side_noncanonical_count"] = int(
                            (canonical_payload.get("normalization", {}) or {}).get("side_noncanonical_count", 0)
                        )
                        canonical_view["position_rows"] = int(
                            (canonical_payload.get("normalization", {}) or {}).get("position_rows", 0)
                        )
                        canonical_path = canonical_dir_path / f"{dstr}.json"
                        canonical_view["path"] = str(canonical_path)
                        try:
                            write_json(canonical_path, canonical_payload)
                            canonical_view["written"] = True
                            broker_canonical_written_days += 1
                            broker_canonical_symbol_noncanonical_total += int(
                                canonical_view.get("symbol_noncanonical_count", 0)
                            )
                            broker_canonical_side_noncanonical_total += int(
                                canonical_view.get("side_noncanonical_count", 0)
                            )
                        except Exception as exc:
                            broker_canonical_write_fail_days += 1
                            canonical_view["write_error"] = f"{type(exc).__name__}:{exc}"
                            day_alerts.append("broker_contract_canonical_write_failed")
                    broker_positions = (
                        broker_payload.get("positions", [])
                        if isinstance(broker_payload.get("positions", []), list)
                        else []
                    )
                    broker_open_count = int(
                        self._safe_float(
                            broker_payload.get("open_positions", len(broker_positions)),
                            len(broker_positions),
                        )
                    )
                    broker_closed_pnl = self._safe_float(
                        broker_payload.get("closed_pnl", 0.0),
                        0.0,
                    )
                    broker_count_gap_ratio = self._ratio(
                        abs(broker_open_count - manifest_open_count),
                        max(1, manifest_open_count),
                    )
                    broker_pnl_gap_abs = abs(broker_closed_pnl - manifest_closed_pnl)
                    total_broker_count_gap += broker_count_gap_ratio
                    total_broker_pnl_gap += broker_pnl_gap_abs
                    if broker_count_gap_ratio > broker_gap_ratio_max:
                        broker_count_breach_days += 1
                        day_alerts.append("broker_open_count_gap_high")
                    if broker_pnl_gap_abs > broker_pnl_gap_abs_max:
                        broker_pnl_breach_days += 1
                        day_alerts.append("broker_closed_pnl_gap_high")

                    compare_this_day = bool((not broker_row_diff_asof_only) or (day == as_of))
                    if compare_this_day:
                        broker_row_diff["active"] = True
                        if paper_state_as_of != dstr:
                            broker_row_diff_skipped_days += 1
                            broker_row_diff["reason"] = "system_state_missing_or_stale"
                        else:
                            broker_compare_payload: dict[str, Any] | None = None
                            compare_source = ""
                            canonical_path_txt = str(canonical_view.get("path", "")).strip()
                            if canonical_path_txt:
                                broker_compare_payload = self.load_json_safely(Path(canonical_path_txt))
                                if broker_compare_payload:
                                    compare_source = "canonical_file"
                            if (not broker_compare_payload) and canonical_payload:
                                broker_compare_payload = canonical_payload
                                compare_source = "canonical_inline"
                            if not broker_compare_payload:
                                broker_compare_payload = self._canonicalize_broker_snapshot(payload=broker_payload)
                                compare_source = "canonical_fallback"

                            broker_rows_for_diff = (
                                broker_compare_payload.get("positions", [])
                                if isinstance(broker_compare_payload.get("positions", []), list)
                                else []
                            )
                            system_rows_for_diff = self._canonicalize_system_open_positions(rows=paper_positions)
                            row_cmp = self._compare_position_rows(
                                broker_rows=broker_rows_for_diff,
                                system_rows=system_rows_for_diff,
                                broker_source=str(
                                    broker_compare_payload.get("source", broker_payload.get("source", ""))
                                ).strip(),
                            )
                            broker_row_diff_samples += 1
                            broker_row_diff_key_mismatch_total += self._safe_float(
                                row_cmp.get("key_mismatch_ratio", 0.0), 0.0
                            )
                            broker_row_diff_count_gap_total += self._safe_float(
                                row_cmp.get("count_gap_ratio", 0.0), 0.0
                            )
                            broker_row_diff_alias_hits_total += int(
                                self._safe_float(row_cmp.get("alias_hits", 0), 0)
                            )
                            broker_row_diff_symbol_alias_hits_total += int(
                                self._safe_float(row_cmp.get("symbol_alias_hits", 0), 0)
                            )
                            broker_row_diff_side_alias_hits_total += int(
                                self._safe_float(row_cmp.get("side_alias_hits", 0), 0)
                            )
                            broker_row_diff_compared_rows_total += int(
                                self._safe_float(
                                    row_cmp.get(
                                        "compared_rows",
                                        self._safe_float(row_cmp.get("broker_rows", 0), 0)
                                        + self._safe_float(row_cmp.get("system_rows", 0), 0),
                                    ),
                                    0,
                                )
                            )
                            broker_row_diff_unresolved_rows_total += int(
                                self._safe_float(row_cmp.get("unresolved_rows", 0), 0)
                            )
                            broker_row_diff_input_rows_total += int(
                                self._safe_float(row_cmp.get("input_rows", 0), 0)
                            )
                            broker_row_diff_unresolved_keys_total += int(
                                self._safe_float(row_cmp.get("unresolved_keys", 0), 0)
                            )
                            broker_row_diff_union_keys_total += int(
                                self._safe_float(row_cmp.get("union_keys", 0), 0)
                            )
                            if bool(row_cmp.get("notional_comparable", False)):
                                broker_row_diff_notional_samples += 1
                                broker_row_diff_notional_gap_total += self._safe_float(
                                    row_cmp.get("notional_gap_ratio", 0.0), 0.0
                                )
                            if compare_source in {"canonical_file", "canonical_inline"}:
                                broker_row_diff_canonical_preferred_days += 1

                            broker_row_diff["skipped"] = False
                            broker_row_diff["source"] = compare_source
                            broker_row_diff["key_mismatch_ratio"] = self._safe_float(
                                row_cmp.get("key_mismatch_ratio", 0.0), 0.0
                            )
                            broker_row_diff["count_gap_ratio"] = self._safe_float(
                                row_cmp.get("count_gap_ratio", 0.0), 0.0
                            )
                            broker_row_diff["notional_gap_ratio"] = self._safe_float(
                                row_cmp.get("notional_gap_ratio", 0.0), 0.0
                            )
                            broker_row_diff["notional_comparable"] = bool(row_cmp.get("notional_comparable", False))
                            broker_row_diff["broker_rows"] = int(row_cmp.get("broker_rows", 0))
                            broker_row_diff["system_rows"] = int(row_cmp.get("system_rows", 0))
                            broker_row_diff["alias_hits"] = int(self._safe_float(row_cmp.get("alias_hits", 0), 0))
                            broker_row_diff["symbol_alias_hits"] = int(
                                self._safe_float(row_cmp.get("symbol_alias_hits", 0), 0)
                            )
                            broker_row_diff["side_alias_hits"] = int(
                                self._safe_float(row_cmp.get("side_alias_hits", 0), 0)
                            )
                            broker_row_diff["alias_hit_rate"] = self._safe_float(
                                row_cmp.get("alias_hit_rate", 0.0), 0.0
                            )
                            broker_row_diff["unresolved_rows"] = int(
                                self._safe_float(row_cmp.get("unresolved_rows", 0), 0)
                            )
                            broker_row_diff["input_rows"] = int(self._safe_float(row_cmp.get("input_rows", 0), 0))
                            broker_row_diff["unresolved_row_ratio"] = self._safe_float(
                                row_cmp.get("unresolved_row_ratio", 0.0), 0.0
                            )
                            broker_row_diff["unresolved_keys"] = int(
                                self._safe_float(row_cmp.get("unresolved_keys", 0), 0)
                            )
                            broker_row_diff["union_keys"] = int(self._safe_float(row_cmp.get("union_keys", 0), 0))
                            broker_row_diff["unresolved_key_ratio"] = self._safe_float(
                                row_cmp.get("unresolved_key_ratio", row_cmp.get("key_mismatch_ratio", 0.0)),
                                0.0,
                            )
                            broker_row_diff["missing_on_broker"] = list(row_cmp.get("missing_on_broker", []))[:10]
                            broker_row_diff["extra_on_broker"] = list(row_cmp.get("extra_on_broker", []))[:10]

                            broker_row_diff_breached = bool(
                                broker_row_diff["key_mismatch_ratio"] > broker_row_diff_key_mismatch_max
                                or broker_row_diff["count_gap_ratio"] > broker_row_diff_count_gap_max
                                or (
                                    broker_row_diff["notional_comparable"]
                                    and broker_row_diff["notional_gap_ratio"] > broker_row_diff_notional_gap_max
                                )
                            )
                            broker_row_diff["breached"] = bool(broker_row_diff_breached)
                            if broker_row_diff_breached:
                                broker_row_diff_breach_days += 1
                                day_alerts.append("broker_row_diff_high")
                else:
                    broker_missing_days += 1
                    day_alerts.append("broker_snapshot_missing")
                    if require_broker_snapshot:
                        day_missing = True

            if plan_gap_breached:
                plan_gap_breach_days += 1
            if closed_count_gap_breached:
                closed_count_gap_breach_days += 1
            if closed_pnl_gap_breached:
                closed_pnl_gap_breach_days += 1
            if open_gap_breached:
                open_gap_breach_days += 1

            if day_missing:
                missing_days += 1

            series.append(
                {
                    "date": dstr,
                    "missing": bool(day_missing),
                    "manifest": {
                        "plans": manifest_plans,
                        "closed_trades": manifest_closed_count,
                        "closed_pnl": manifest_closed_pnl,
                        "open_positions": manifest_open_count,
                    },
                    "csv": {
                        "found": bool(csv_summary.get("found", False)),
                        "rows": int(self._safe_float(csv_summary.get("rows", 0), 0.0)),
                    },
                    "db": {
                        "latest_positions_rows": int(plan_db_count) if plan_db_count is not None else None,
                        "executed_closed_rows": int(closed_db_count) if closed_db_count is not None else None,
                        "executed_closed_pnl": float(closed_db_pnl) if closed_db_pnl is not None else None,
                    },
                    "gaps": {
                        "plan_count_ratio": plan_gap_ratio,
                        "closed_count_ratio": closed_count_gap_ratio,
                        "closed_pnl_abs": closed_pnl_gap_abs,
                        "open_count_ratio": open_gap_ratio,
                        "broker_open_count_ratio": broker_count_gap_ratio,
                        "broker_closed_pnl_abs": broker_pnl_gap_abs,
                    },
                    "broker": {
                        "monitor_enabled": bool(broker_monitor_enabled),
                        "found": bool(broker_found),
                        "contract_lint": {
                            "schema_ok": bool(broker_contract_lint.get("schema_ok", True)),
                            "numeric_ok": bool(broker_contract_lint.get("numeric_ok", True)),
                            "symbol_ok": bool(broker_contract_lint.get("symbol_ok", True)),
                            "schema_errors": list(broker_contract_lint.get("schema_errors", []))[:5],
                            "numeric_errors": list(broker_contract_lint.get("numeric_errors", []))[:5],
                            "symbol_errors": list(broker_contract_lint.get("symbol_errors", []))[:5],
                            "symbol_noncanonical_count": int(broker_contract_lint.get("symbol_noncanonical_count", 0)),
                            "side_noncanonical_count": int(broker_contract_lint.get("side_noncanonical_count", 0)),
                            "position_rows": int(broker_contract_lint.get("position_rows", 0)),
                        },
                        "canonical_view": {
                            "enabled": bool(canonical_view.get("enabled", False)),
                            "eligible": bool(canonical_view.get("eligible", False)),
                            "written": bool(canonical_view.get("written", False)),
                            "write_error": str(canonical_view.get("write_error", "")),
                            "path": str(canonical_view.get("path", "")),
                            "symbol_noncanonical_count": int(canonical_view.get("symbol_noncanonical_count", 0)),
                            "side_noncanonical_count": int(canonical_view.get("side_noncanonical_count", 0)),
                            "position_rows": int(canonical_view.get("position_rows", 0)),
                        },
                        "row_diff": {
                            "active": bool(broker_row_diff.get("active", False)),
                            "skipped": bool(broker_row_diff.get("skipped", True)),
                            "reason": str(broker_row_diff.get("reason", "")),
                            "source": str(broker_row_diff.get("source", "")),
                            "key_mismatch_ratio": self._safe_float(
                                broker_row_diff.get("key_mismatch_ratio", 0.0), 0.0
                            ),
                            "count_gap_ratio": self._safe_float(
                                broker_row_diff.get("count_gap_ratio", 0.0), 0.0
                            ),
                            "notional_gap_ratio": self._safe_float(
                                broker_row_diff.get("notional_gap_ratio", 0.0), 0.0
                            ),
                            "notional_comparable": bool(broker_row_diff.get("notional_comparable", False)),
                            "broker_rows": int(broker_row_diff.get("broker_rows", 0)),
                            "system_rows": int(broker_row_diff.get("system_rows", 0)),
                            "alias_hits": int(broker_row_diff.get("alias_hits", 0)),
                            "symbol_alias_hits": int(broker_row_diff.get("symbol_alias_hits", 0)),
                            "side_alias_hits": int(broker_row_diff.get("side_alias_hits", 0)),
                            "alias_hit_rate": self._safe_float(broker_row_diff.get("alias_hit_rate", 0.0), 0.0),
                            "unresolved_rows": int(broker_row_diff.get("unresolved_rows", 0)),
                            "input_rows": int(broker_row_diff.get("input_rows", 0)),
                            "unresolved_row_ratio": self._safe_float(
                                broker_row_diff.get("unresolved_row_ratio", 0.0), 0.0
                            ),
                            "unresolved_keys": int(broker_row_diff.get("unresolved_keys", 0)),
                            "union_keys": int(broker_row_diff.get("union_keys", 0)),
                            "unresolved_key_ratio": self._safe_float(
                                broker_row_diff.get("unresolved_key_ratio", 0.0), 0.0
                            ),
                            "missing_on_broker": list(broker_row_diff.get("missing_on_broker", []))[:10],
                            "extra_on_broker": list(broker_row_diff.get("extra_on_broker", []))[:10],
                            "breached": bool(broker_row_diff.get("breached", False)),
                        },
                    },
                    "alerts": day_alerts[:8],
                }
            )

        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

        active = samples >= min_samples
        missing_ratio = self._ratio(missing_days, samples)
        plan_gap_breach_ratio = self._ratio(plan_gap_breach_days, samples)
        closed_count_gap_breach_ratio = self._ratio(closed_count_gap_breach_days, samples)
        closed_pnl_gap_breach_ratio = self._ratio(closed_pnl_gap_breach_days, samples)
        open_gap_breach_ratio = self._ratio(open_gap_breach_days, open_gap_samples) if open_gap_samples > 0 else 0.0
        avg_plan_gap_ratio = self._ratio(total_plan_gap, samples)
        avg_closed_count_gap_ratio = self._ratio(total_closed_count_gap, samples)
        avg_closed_pnl_gap_abs = self._ratio(total_closed_pnl_gap, samples)
        avg_open_gap_ratio = self._ratio(total_open_gap, open_gap_samples) if open_gap_samples > 0 else 0.0
        broker_missing_ratio = self._ratio(broker_missing_days, broker_expected_days) if broker_expected_days > 0 else 0.0
        broker_count_breach_ratio = self._ratio(broker_count_breach_days, broker_samples) if broker_samples > 0 else 0.0
        broker_pnl_breach_ratio = self._ratio(broker_pnl_breach_days, broker_samples) if broker_samples > 0 else 0.0
        broker_contract_schema_invalid_ratio = (
            self._ratio(broker_contract_schema_invalid_days, broker_samples) if broker_samples > 0 else 0.0
        )
        broker_contract_numeric_invalid_ratio = (
            self._ratio(broker_contract_numeric_invalid_days, broker_samples) if broker_samples > 0 else 0.0
        )
        broker_contract_symbol_invalid_ratio = (
            self._ratio(broker_contract_symbol_invalid_days, broker_samples) if broker_samples > 0 else 0.0
        )
        broker_contract_symbol_noncanonical_ratio = (
            self._ratio(broker_symbol_noncanonical_total, broker_symbol_rows_total)
            if broker_symbol_rows_total > 0
            else 0.0
        )
        broker_contract_side_noncanonical_ratio = (
            self._ratio(broker_side_noncanonical_total, broker_symbol_rows_total)
            if broker_symbol_rows_total > 0
            else 0.0
        )
        broker_canonical_write_fail_ratio = (
            self._ratio(broker_canonical_write_fail_days, broker_canonical_eligible_days)
            if broker_canonical_eligible_days > 0
            else 0.0
        )
        broker_canonical_symbol_noncanonical_ratio = (
            self._ratio(broker_canonical_symbol_noncanonical_total, broker_symbol_rows_total)
            if broker_symbol_rows_total > 0
            else 0.0
        )
        broker_canonical_side_noncanonical_ratio = (
            self._ratio(broker_canonical_side_noncanonical_total, broker_symbol_rows_total)
            if broker_symbol_rows_total > 0
            else 0.0
        )
        broker_row_diff_breach_ratio = (
            self._ratio(broker_row_diff_breach_days, broker_row_diff_samples) if broker_row_diff_samples > 0 else 0.0
        )
        broker_row_diff_avg_key_mismatch = (
            self._ratio(broker_row_diff_key_mismatch_total, broker_row_diff_samples)
            if broker_row_diff_samples > 0
            else 0.0
        )
        broker_row_diff_avg_count_gap = (
            self._ratio(broker_row_diff_count_gap_total, broker_row_diff_samples)
            if broker_row_diff_samples > 0
            else 0.0
        )
        broker_row_diff_avg_notional_gap = (
            self._ratio(broker_row_diff_notional_gap_total, broker_row_diff_notional_samples)
            if broker_row_diff_notional_samples > 0
            else 0.0
        )
        broker_row_diff_canonical_preferred_ratio = (
            self._ratio(broker_row_diff_canonical_preferred_days, broker_row_diff_samples)
            if broker_row_diff_samples > 0
            else 0.0
        )
        broker_row_diff_alias_hit_rate = (
            self._ratio(broker_row_diff_alias_hits_total, broker_row_diff_compared_rows_total)
            if broker_row_diff_compared_rows_total > 0
            else 0.0
        )
        broker_row_diff_unresolved_row_ratio = (
            self._ratio(broker_row_diff_unresolved_rows_total, broker_row_diff_input_rows_total)
            if broker_row_diff_input_rows_total > 0
            else 0.0
        )
        broker_row_diff_unresolved_key_ratio = (
            self._ratio(broker_row_diff_unresolved_keys_total, broker_row_diff_union_keys_total)
            if broker_row_diff_union_keys_total > 0
            else 0.0
        )
        avg_broker_count_gap_ratio = self._ratio(total_broker_count_gap, broker_samples) if broker_samples > 0 else 0.0
        avg_broker_pnl_gap_abs = self._ratio(total_broker_pnl_gap, broker_samples) if broker_samples > 0 else 0.0

        checks = {
            "missing_ratio_ok": True,
            "plan_count_gap_ok": True,
            "closed_count_gap_ok": True,
            "closed_pnl_gap_ok": True,
            "open_count_gap_ok": True,
            "broker_missing_ratio_ok": True,
            "broker_count_gap_ok": True,
            "broker_pnl_gap_ok": True,
            "broker_contract_schema_ok": True,
            "broker_contract_numeric_ok": True,
            "broker_contract_symbol_ok": True,
            "broker_contract_canonical_view_ok": True,
            "broker_row_diff_ok": True,
            "broker_row_diff_alias_drift_ok": True,
            "broker_row_diff_artifact_rotation_ok": True,
            "broker_row_diff_artifact_checksum_index_ok": True,
        }
        alerts: list[str] = []
        if active:
            checks["missing_ratio_ok"] = bool(missing_ratio <= missing_ratio_max)
            checks["plan_count_gap_ok"] = bool(plan_gap_breach_ratio <= plan_gap_ratio_max)
            checks["closed_count_gap_ok"] = bool(closed_count_gap_breach_ratio <= closed_count_gap_ratio_max)
            checks["closed_pnl_gap_ok"] = bool(closed_pnl_gap_breach_ratio <= closed_count_gap_ratio_max)
            if open_gap_samples > 0:
                checks["open_count_gap_ok"] = bool(open_gap_breach_ratio <= open_gap_ratio_max)
            if broker_monitor_enabled:
                checks["broker_missing_ratio_ok"] = bool(broker_missing_ratio <= broker_missing_ratio_max)
                if broker_samples > 0:
                    checks["broker_count_gap_ok"] = bool(broker_count_breach_ratio <= broker_gap_ratio_max)
                    checks["broker_pnl_gap_ok"] = bool(broker_pnl_breach_ratio <= broker_gap_ratio_max)
                    checks["broker_contract_schema_ok"] = bool(
                        broker_contract_schema_invalid_ratio <= broker_contract_schema_invalid_ratio_max
                    )
                    checks["broker_contract_numeric_ok"] = bool(
                        broker_contract_numeric_invalid_ratio <= broker_contract_numeric_invalid_ratio_max
                    )
                    checks["broker_contract_symbol_ok"] = bool(
                        broker_contract_symbol_invalid_ratio <= broker_contract_symbol_invalid_ratio_max
                        and broker_contract_symbol_noncanonical_ratio <= broker_contract_symbol_noncanonical_ratio_max
                    )
                    if emit_canonical_view and broker_canonical_eligible_days > 0:
                        checks["broker_contract_canonical_view_ok"] = bool(
                            broker_canonical_write_fail_days == 0
                            and broker_canonical_written_days >= broker_canonical_eligible_days
                        )
                    if broker_row_diff_samples >= broker_row_diff_min_samples:
                        checks["broker_row_diff_ok"] = bool(
                            broker_row_diff_breach_ratio <= broker_row_diff_breach_ratio_max
                        )
                        if broker_row_diff_alias_monitor_enabled:
                            alias_configured = bool(
                                broker_row_diff_symbol_alias_size > 0 or broker_row_diff_side_alias_size > 0
                            )
                            alias_hit_rate_ok = True
                            if (
                                alias_configured
                                and broker_row_diff_alias_hit_rate_min > 0.0
                                and broker_row_diff_compared_rows_total > 0
                            ):
                                alias_hit_rate_ok = bool(
                                    broker_row_diff_alias_hit_rate >= broker_row_diff_alias_hit_rate_min
                                )
                            unresolved_key_ok = bool(
                                broker_row_diff_unresolved_key_ratio <= broker_row_diff_unresolved_key_ratio_max
                            )
                            checks["broker_row_diff_alias_drift_ok"] = bool(alias_hit_rate_ok and unresolved_key_ok)
            if not checks["missing_ratio_ok"]:
                alerts.append("reconcile_missing_ratio_high")
            if not checks["plan_count_gap_ok"]:
                alerts.append("reconcile_plan_count_gap_high")
            if not checks["closed_count_gap_ok"]:
                alerts.append("reconcile_closed_count_gap_high")
            if not checks["closed_pnl_gap_ok"]:
                alerts.append("reconcile_closed_pnl_gap_high")
            if not checks["open_count_gap_ok"]:
                alerts.append("reconcile_open_count_gap_high")
            if not checks["broker_missing_ratio_ok"]:
                alerts.append("reconcile_broker_missing_ratio_high")
            if not checks["broker_count_gap_ok"]:
                alerts.append("reconcile_broker_open_count_gap_high")
            if not checks["broker_pnl_gap_ok"]:
                alerts.append("reconcile_broker_closed_pnl_gap_high")
            if not checks["broker_contract_schema_ok"]:
                alerts.append("reconcile_broker_contract_schema_invalid")
            if not checks["broker_contract_numeric_ok"]:
                alerts.append("reconcile_broker_contract_numeric_invalid")
            if not checks["broker_contract_symbol_ok"]:
                alerts.append("reconcile_broker_contract_symbol_invalid")
            if not checks["broker_contract_canonical_view_ok"]:
                alerts.append("reconcile_broker_contract_canonical_view_failed")
            if not checks["broker_row_diff_ok"]:
                alerts.append("reconcile_broker_row_diff_high")
            if not checks["broker_row_diff_alias_drift_ok"]:
                alerts.append("reconcile_broker_row_diff_alias_drift")
        else:
            alerts.append("insufficient_reconcile_samples")

        series.reverse()
        row_diff_artifact = self._write_reconcile_row_diff_artifact(
            as_of=as_of,
            series=series,
            retention_days=broker_row_diff_artifact_retention_days,
            checksum_index_enabled=broker_row_diff_artifact_checksum_index_enabled,
        )
        row_diff_artifact_failed = (
            (int(broker_row_diff_breach_days) > 0)
            and (not bool(row_diff_artifact.get("written", False)))
            and str(row_diff_artifact.get("reason", "")).startswith("write_failed")
        )
        row_diff_artifact_rotation_failed = bool(row_diff_artifact.get("rotation_failed", False))
        row_diff_artifact_checksum_index_failed = bool(row_diff_artifact.get("checksum_index_failed", False))
        checks["broker_row_diff_artifact_rotation_ok"] = bool(not row_diff_artifact_rotation_failed)
        checks["broker_row_diff_artifact_checksum_index_ok"] = bool(
            (not broker_row_diff_artifact_checksum_index_enabled) or (not row_diff_artifact_checksum_index_failed)
        )
        if row_diff_artifact_failed and ("reconcile_broker_row_diff_artifact_failed" not in alerts):
            alerts.append("reconcile_broker_row_diff_artifact_failed")
        if row_diff_artifact_rotation_failed and ("reconcile_broker_row_diff_artifact_rotation_failed" not in alerts):
            alerts.append("reconcile_broker_row_diff_artifact_rotation_failed")
        if row_diff_artifact_checksum_index_failed and (
            "reconcile_broker_row_diff_artifact_checksum_index_failed" not in alerts
        ):
            alerts.append("reconcile_broker_row_diff_artifact_checksum_index_failed")
        return {
            "active": active,
            "window_days": window_days,
            "samples": samples,
            "min_samples": min_samples,
            "metrics": {
                "missing_days": missing_days,
                "missing_ratio": missing_ratio,
                "plan_gap_breach_ratio": plan_gap_breach_ratio,
                "closed_count_gap_breach_ratio": closed_count_gap_breach_ratio,
                "closed_pnl_gap_breach_ratio": closed_pnl_gap_breach_ratio,
                "open_gap_breach_ratio": open_gap_breach_ratio,
                "avg_plan_gap_ratio": avg_plan_gap_ratio,
                "avg_closed_count_gap_ratio": avg_closed_count_gap_ratio,
                "avg_closed_pnl_gap_abs": avg_closed_pnl_gap_abs,
                "avg_open_gap_ratio": avg_open_gap_ratio,
                "open_gap_samples": open_gap_samples,
                "broker_monitor_enabled": bool(broker_monitor_enabled),
                "broker_expected_days": broker_expected_days,
                "broker_missing_days": broker_missing_days,
                "broker_missing_ratio": broker_missing_ratio,
                "broker_samples": broker_samples,
                "broker_count_breach_ratio": broker_count_breach_ratio,
                "broker_pnl_breach_ratio": broker_pnl_breach_ratio,
                "broker_contract_schema_invalid_days": broker_contract_schema_invalid_days,
                "broker_contract_numeric_invalid_days": broker_contract_numeric_invalid_days,
                "broker_contract_symbol_invalid_days": broker_contract_symbol_invalid_days,
                "broker_contract_schema_invalid_ratio": broker_contract_schema_invalid_ratio,
                "broker_contract_numeric_invalid_ratio": broker_contract_numeric_invalid_ratio,
                "broker_contract_symbol_invalid_ratio": broker_contract_symbol_invalid_ratio,
                "broker_symbol_noncanonical_total": broker_symbol_noncanonical_total,
                "broker_side_noncanonical_total": broker_side_noncanonical_total,
                "broker_symbol_rows_total": broker_symbol_rows_total,
                "broker_contract_symbol_noncanonical_ratio": broker_contract_symbol_noncanonical_ratio,
                "broker_contract_side_noncanonical_ratio": broker_contract_side_noncanonical_ratio,
                "broker_canonical_eligible_days": broker_canonical_eligible_days,
                "broker_canonical_written_days": broker_canonical_written_days,
                "broker_canonical_write_fail_days": broker_canonical_write_fail_days,
                "broker_canonical_write_fail_ratio": broker_canonical_write_fail_ratio,
                "broker_canonical_symbol_noncanonical_total": broker_canonical_symbol_noncanonical_total,
                "broker_canonical_side_noncanonical_total": broker_canonical_side_noncanonical_total,
                "broker_canonical_symbol_noncanonical_ratio": broker_canonical_symbol_noncanonical_ratio,
                "broker_canonical_side_noncanonical_ratio": broker_canonical_side_noncanonical_ratio,
                "broker_row_diff_samples": broker_row_diff_samples,
                "broker_row_diff_min_samples": broker_row_diff_min_samples,
                "broker_row_diff_skipped_days": broker_row_diff_skipped_days,
                "broker_row_diff_breach_days": broker_row_diff_breach_days,
                "broker_row_diff_breach_ratio": broker_row_diff_breach_ratio,
                "broker_row_diff_avg_key_mismatch": broker_row_diff_avg_key_mismatch,
                "broker_row_diff_avg_count_gap": broker_row_diff_avg_count_gap,
                "broker_row_diff_notional_samples": broker_row_diff_notional_samples,
                "broker_row_diff_avg_notional_gap": broker_row_diff_avg_notional_gap,
                "broker_row_diff_canonical_preferred_days": broker_row_diff_canonical_preferred_days,
                "broker_row_diff_canonical_preferred_ratio": broker_row_diff_canonical_preferred_ratio,
                "broker_row_diff_alias_hits_total": int(broker_row_diff_alias_hits_total),
                "broker_row_diff_symbol_alias_hits_total": int(broker_row_diff_symbol_alias_hits_total),
                "broker_row_diff_side_alias_hits_total": int(broker_row_diff_side_alias_hits_total),
                "broker_row_diff_compared_rows_total": int(broker_row_diff_compared_rows_total),
                "broker_row_diff_alias_hit_rate": float(broker_row_diff_alias_hit_rate),
                "broker_row_diff_unresolved_rows_total": int(broker_row_diff_unresolved_rows_total),
                "broker_row_diff_input_rows_total": int(broker_row_diff_input_rows_total),
                "broker_row_diff_unresolved_row_ratio": float(broker_row_diff_unresolved_row_ratio),
                "broker_row_diff_unresolved_keys_total": int(broker_row_diff_unresolved_keys_total),
                "broker_row_diff_union_keys_total": int(broker_row_diff_union_keys_total),
                "broker_row_diff_unresolved_key_ratio": float(broker_row_diff_unresolved_key_ratio),
                "broker_row_diff_symbol_alias_size": int(broker_row_diff_symbol_alias_size),
                "broker_row_diff_side_alias_size": int(broker_row_diff_side_alias_size),
                "broker_row_diff_artifact_written": bool(row_diff_artifact.get("written", False)),
                "broker_row_diff_artifact_sample_rows": int(row_diff_artifact.get("sample_rows", 0)),
                "broker_row_diff_artifact_breach_rows": int(row_diff_artifact.get("breach_rows", 0)),
                "broker_row_diff_artifact_retention_days": int(
                    row_diff_artifact.get("retention_days", broker_row_diff_artifact_retention_days)
                ),
                "broker_row_diff_artifact_rotated_out_count": int(row_diff_artifact.get("rotated_out_count", 0)),
                "broker_row_diff_artifact_rotation_failed": bool(row_diff_artifact_rotation_failed),
                "broker_row_diff_artifact_checksum_index_enabled": bool(
                    row_diff_artifact.get(
                        "checksum_index_enabled",
                        broker_row_diff_artifact_checksum_index_enabled,
                    )
                ),
                "broker_row_diff_artifact_checksum_index_written": bool(
                    row_diff_artifact.get("checksum_index_written", False)
                ),
                "broker_row_diff_artifact_checksum_index_entries": int(
                    row_diff_artifact.get("checksum_index_entries", 0)
                ),
                "broker_row_diff_artifact_checksum_index_failed": bool(row_diff_artifact_checksum_index_failed),
                "broker_row_diff_artifact_failed": bool(row_diff_artifact_failed),
                "avg_broker_count_gap_ratio": avg_broker_count_gap_ratio,
                "avg_broker_pnl_gap_abs": avg_broker_pnl_gap_abs,
            },
            "thresholds": {
                "ops_reconcile_missing_ratio_max": missing_ratio_max,
                "ops_reconcile_plan_gap_ratio_max": plan_gap_ratio_max,
                "ops_reconcile_closed_count_gap_ratio_max": closed_count_gap_ratio_max,
                "ops_reconcile_closed_pnl_gap_abs_max": closed_pnl_gap_abs_max,
                "ops_reconcile_open_gap_ratio_max": open_gap_ratio_max,
                "ops_reconcile_broker_gap_ratio_max": broker_gap_ratio_max,
                "ops_reconcile_broker_pnl_gap_abs_max": broker_pnl_gap_abs_max,
                "ops_reconcile_broker_missing_ratio_max": broker_missing_ratio_max,
                "ops_reconcile_broker_contract_schema_invalid_ratio_max": broker_contract_schema_invalid_ratio_max,
                "ops_reconcile_broker_contract_numeric_invalid_ratio_max": broker_contract_numeric_invalid_ratio_max,
                "ops_reconcile_broker_contract_symbol_invalid_ratio_max": broker_contract_symbol_invalid_ratio_max,
                "ops_reconcile_broker_contract_symbol_noncanonical_ratio_max": broker_contract_symbol_noncanonical_ratio_max,
                "ops_reconcile_broker_closed_pnl_abs_hard_max": broker_closed_pnl_abs_hard_max,
                "ops_reconcile_broker_position_qty_abs_hard_max": broker_position_qty_abs_hard_max,
                "ops_reconcile_broker_position_notional_abs_hard_max": broker_position_notional_abs_hard_max,
                "ops_reconcile_broker_price_abs_hard_max": broker_price_abs_hard_max,
                "ops_reconcile_require_broker_snapshot": bool(require_broker_snapshot),
                "ops_reconcile_broker_contract_emit_canonical_view": bool(emit_canonical_view),
                "ops_reconcile_broker_contract_canonical_dir": str(canonical_dir_path),
                "ops_reconcile_broker_row_diff_min_samples": broker_row_diff_min_samples,
                "ops_reconcile_broker_row_diff_breach_ratio_max": broker_row_diff_breach_ratio_max,
                "ops_reconcile_broker_row_diff_key_mismatch_max": broker_row_diff_key_mismatch_max,
                "ops_reconcile_broker_row_diff_count_gap_max": broker_row_diff_count_gap_max,
                "ops_reconcile_broker_row_diff_notional_gap_max": broker_row_diff_notional_gap_max,
                "ops_reconcile_broker_row_diff_alias_monitor_enabled": bool(broker_row_diff_alias_monitor_enabled),
                "ops_reconcile_broker_row_diff_alias_hit_rate_min": broker_row_diff_alias_hit_rate_min,
                "ops_reconcile_broker_row_diff_unresolved_key_ratio_max": broker_row_diff_unresolved_key_ratio_max,
                "ops_reconcile_broker_row_diff_asof_only": bool(broker_row_diff_asof_only),
                "ops_reconcile_broker_row_diff_artifact_retention_days": int(
                    broker_row_diff_artifact_retention_days
                ),
                "ops_reconcile_broker_row_diff_artifact_checksum_index_enabled": bool(
                    broker_row_diff_artifact_checksum_index_enabled
                ),
                "ops_reconcile_broker_row_diff_symbol_alias_size": int(broker_row_diff_symbol_alias_size),
                "ops_reconcile_broker_row_diff_side_alias_size": int(broker_row_diff_side_alias_size),
            },
            "checks": checks,
            "alerts": alerts,
            "artifacts": {
                "row_diff": {
                    "written": bool(row_diff_artifact.get("written", False)),
                    "json": str(row_diff_artifact.get("json", "")),
                    "md": str(row_diff_artifact.get("md", "")),
                    "sample_rows": int(row_diff_artifact.get("sample_rows", 0)),
                    "breach_rows": int(row_diff_artifact.get("breach_rows", 0)),
                    "retention_days": int(
                        row_diff_artifact.get("retention_days", broker_row_diff_artifact_retention_days)
                    ),
                    "rotated_out_count": int(row_diff_artifact.get("rotated_out_count", 0)),
                    "rotated_out_dates": [str(x) for x in row_diff_artifact.get("rotated_out_dates", [])],
                    "rotation_failed": bool(row_diff_artifact.get("rotation_failed", False)),
                    "checksum_index_enabled": bool(
                        row_diff_artifact.get(
                            "checksum_index_enabled",
                            broker_row_diff_artifact_checksum_index_enabled,
                        )
                    ),
                    "checksum_index_written": bool(row_diff_artifact.get("checksum_index_written", False)),
                    "checksum_index_path": str(row_diff_artifact.get("checksum_index_path", "")),
                    "checksum_index_entries": int(row_diff_artifact.get("checksum_index_entries", 0)),
                    "checksum_index_failed": bool(row_diff_artifact.get("checksum_index_failed", False)),
                    "reason": str(row_diff_artifact.get("reason", "")),
                }
            },
            "series": series[-10:],
        }

    def _rollback_candidates(self, *, as_of: date, lookback_days: int) -> list[dict[str, str]]:
        artifacts_dir = self.output_dir / "artifacts"
        review_dir = self.output_dir / "review"
        out: list[dict[str, str]] = []
        seen_paths: set[str] = set()
        max_days = max(1, int(lookback_days))

        for path in sorted(artifacts_dir.glob("params_live_backup_*.yaml"), reverse=True):
            tag = path.stem.replace("params_live_backup_", "").strip()
            try:
                d = date.fromisoformat(tag)
            except Exception:
                continue
            if d > as_of:
                continue
            key = str(path)
            if key in seen_paths:
                continue
            seen_paths.add(key)
            out.append({"date": d.isoformat(), "path": key, "source": "params_backup"})
            if len(out) >= max_days:
                break

        for i in range(max_days):
            day = as_of - timedelta(days=i)
            payload = self.load_json_safely(review_dir / f"{day.isoformat()}_param_delta.yaml")
            anchor = str(payload.get("rollback_anchor", "")).strip()
            if not anchor or anchor == "initial_seed":
                continue
            key = str(Path(anchor))
            if key in seen_paths:
                continue
            if not Path(anchor).exists():
                continue
            seen_paths.add(key)
            out.append({"date": day.isoformat(), "path": key, "source": "review_delta"})
            if len(out) >= max_days:
                break

        out.sort(key=lambda x: str(x.get("date", "")), reverse=True)
        return out

    def _rollback_recommendation(
        self,
        *,
        as_of: date,
        checks: dict[str, Any],
        state_stability: dict[str, Any],
        temporal_audit: dict[str, Any],
        slot_anomaly: dict[str, Any],
        mode_drift: dict[str, Any],
        reconcile_drift: dict[str, Any],
    ) -> dict[str, Any]:
        score = 0
        reason_codes: list[str] = []

        if not bool(checks.get("risk_violations_ok", True)):
            score += 4
            reason_codes.append("risk_violations")
        if not bool(checks.get("max_drawdown_ok", True)):
            score += 3
            reason_codes.append("max_drawdown")
        if not bool(checks.get("stable_replay_ok", True)):
            score += 2
            reason_codes.append("stable_replay")
        if not bool(checks.get("health_ok", True)):
            score += 2
            reason_codes.append("health_degraded")
        if not bool(checks.get("state_stability_ok", True)):
            score += 2
            reason_codes.append("state_stability")
        if not bool(checks.get("temporal_audit_ok", True)):
            score += 2
            reason_codes.append("temporal_audit")
        if not bool(checks.get("mode_drift_ok", True)):
            score += 2
            reason_codes.append("mode_drift")
        if not bool(checks.get("reconcile_drift_ok", True)):
            score += 2
            reason_codes.append("reconcile_drift")
        if not bool(checks.get("slot_anomaly_ok", True)):
            score += 1
            reason_codes.append("slot_anomaly")
        if not bool(checks.get("tests_ok", True)):
            score += 1
            reason_codes.append("tests")
        if not bool(checks.get("review_pass_gate", True)):
            score += 1
            reason_codes.append("review_gate")

        hard_reasons = {"risk_violations", "max_drawdown"}
        has_hard_reason = any(code in hard_reasons for code in reason_codes)
        level = "none"
        if has_hard_reason or score >= 7:
            level = "hard"
        elif score >= 4:
            level = "soft"

        candidates = self._rollback_candidates(as_of=as_of, lookback_days=30)
        target_anchor = ""
        if level != "none":
            for item in candidates:
                tag = str(item.get("date", "")).strip()
                if tag and tag < as_of.isoformat():
                    target_anchor = str(item.get("path", "")).strip()
                    break
            if not target_anchor and candidates:
                target_anchor = str(candidates[0].get("path", "")).strip()

        anchor_ready = True
        if level != "none":
            anchor_ready = bool(target_anchor and Path(target_anchor).exists())
        action = "no_rollback"
        if level == "soft":
            action = "rollback_to_last_stable_anchor_after_partial_recheck"
        elif level == "hard":
            action = "rollback_now_and_lock_parameter_updates"

        state_alerts = state_stability.get("alerts", []) if isinstance(state_stability.get("alerts", []), list) else []
        temporal_alerts = temporal_audit.get("alerts", []) if isinstance(temporal_audit.get("alerts", []), list) else []
        slot_alerts = slot_anomaly.get("alerts", []) if isinstance(slot_anomaly.get("alerts", []), list) else []
        drift_alerts = mode_drift.get("alerts", []) if isinstance(mode_drift.get("alerts", []), list) else []
        reconcile_alerts = (
            reconcile_drift.get("alerts", []) if isinstance(reconcile_drift.get("alerts", []), list) else []
        )

        return {
            "active": level != "none",
            "level": level,
            "score": score,
            "reason_codes": reason_codes,
            "action": action,
            "target_anchor": target_anchor,
            "anchor_ready": bool(anchor_ready),
            "cooldown_days": 3 if level == "hard" else (1 if level == "soft" else 0),
            "candidates": candidates[:10],
            "alerts": list(state_alerts[:2])
            + list(temporal_alerts[:2])
            + list(slot_alerts[:2])
            + list(drift_alerts[:2])
            + list(reconcile_alerts[:2]),
        }

    def _live_mode_metrics(self, *, as_of: date, window_days: int) -> dict[str, dict[str, float]]:
        db_path = self._effective_sqlite_path()
        if not db_path.exists():
            return {}

        start = as_of - timedelta(days=max(1, int(window_days)) - 1)
        sql = (
            "SELECT date, runtime_mode, mode, pnl "
            "FROM executed_plans "
            "WHERE date >= ? AND date <= ? "
            "ORDER BY date ASC"
        )
        try:
            with closing(sqlite3.connect(db_path)) as conn:
                conn.row_factory = sqlite3.Row
                with closing(conn.cursor()) as cur:
                    cur.execute(sql, (start.isoformat(), as_of.isoformat()))
                    rows = cur.fetchall()
        except Exception:
            return {}

        buckets: dict[str, dict[str, float]] = {}
        for row in rows:
            mode_raw = str(row["runtime_mode"] or row["mode"] or "").strip()
            mode = mode_raw or "base"
            pnl = self._safe_float(row["pnl"], 0.0)
            b = buckets.setdefault(
                mode,
                {
                    "trades": 0.0,
                    "wins": 0.0,
                    "gross_profit": 0.0,
                    "gross_loss": 0.0,
                },
            )
            b["trades"] += 1.0
            if pnl > 0:
                b["wins"] += 1.0
                b["gross_profit"] += pnl
            elif pnl < 0:
                b["gross_loss"] += abs(pnl)

        out: dict[str, dict[str, float]] = {}
        for mode, b in buckets.items():
            trades = max(1.0, float(b["trades"]))
            gp = float(b["gross_profit"])
            gl = float(b["gross_loss"])
            if gl > 1e-9:
                pf = gp / gl
            else:
                pf = 10.0 if gp > 0 else 0.0
            out[mode] = {
                "trades": float(b["trades"]),
                "win_rate": float(b["wins"] / trades),
                "profit_factor": float(pf),
            }
        return out

    def _mode_drift_metrics(self, *, as_of: date) -> dict[str, Any]:
        val = self.settings.validation if isinstance(self.settings.validation, dict) else {}
        window_days = max(1, int(val.get("mode_drift_window_days", 120)))
        min_live_trades = max(1, int(val.get("mode_drift_min_live_trades", 30)))
        max_wr_gap = self._safe_float(val.get("mode_drift_win_rate_max_gap", 0.12), 0.12)
        max_pf_gap = self._safe_float(val.get("mode_drift_profit_factor_max_gap", 0.40), 0.40)
        focus_runtime_only = bool(val.get("mode_drift_focus_runtime_mode_only", True))

        latest_feedback = self._latest_mode_feedback_payload(as_of=as_of, lookback_days=max(30, window_days))
        runtime_mode = str(latest_feedback.get("runtime_mode", "")).strip()
        history = latest_feedback.get("history", {}) if isinstance(latest_feedback.get("history", {}), dict) else {}
        baseline_modes = history.get("modes", {}) if isinstance(history.get("modes", {}), dict) else {}
        live_modes = self._live_mode_metrics(as_of=as_of, window_days=window_days)

        scope_modes: list[str]
        if focus_runtime_only and runtime_mode:
            scope_modes = [runtime_mode]
        else:
            names = set(str(k).strip() for k in baseline_modes.keys()) | set(str(k).strip() for k in live_modes.keys())
            scope_modes = sorted([x for x in names if x])

        checks = {
            "samples_ok": True,
            "win_rate_gap_ok": True,
            "profit_factor_gap_ok": True,
        }
        alerts: list[str] = []
        mode_rows: dict[str, Any] = {}
        compared_modes = 0
        active_modes = 0

        for mode in scope_modes:
            baseline = baseline_modes.get(mode, {}) if isinstance(baseline_modes.get(mode, {}), dict) else {}
            live = live_modes.get(mode, {}) if isinstance(live_modes.get(mode, {}), dict) else {}
            base_wr = self._safe_float(baseline.get("avg_win_rate", 0.0), 0.0)
            base_pf = self._safe_float(baseline.get("avg_profit_factor", 0.0), 0.0)
            live_wr = self._safe_float(live.get("win_rate", 0.0), 0.0)
            live_pf = self._safe_float(live.get("profit_factor", 0.0), 0.0)
            live_trades = int(self._safe_float(live.get("trades", 0.0), 0.0))
            baseline_samples = int(self._safe_float(baseline.get("samples", 0.0), 0.0))

            row = {
                "baseline": {
                    "samples": baseline_samples,
                    "win_rate": base_wr,
                    "profit_factor": base_pf,
                },
                "live": {
                    "trades": live_trades,
                    "win_rate": live_wr,
                    "profit_factor": live_pf,
                },
                "gaps": {
                    "win_rate_abs": abs(live_wr - base_wr),
                    "profit_factor_abs": abs(live_pf - base_pf),
                },
                "checks": {
                    "baseline_ok": baseline_samples > 0,
                    "samples_ok": live_trades >= min_live_trades,
                    "win_rate_gap_ok": True,
                    "profit_factor_gap_ok": True,
                },
                "active": False,
                "reason": "",
            }

            if baseline_samples <= 0:
                row["reason"] = "missing_backtest_baseline"
                checks["samples_ok"] = False
                alerts.append(f"mode_drift_missing_baseline:{mode}")
            elif live_trades < min_live_trades:
                row["reason"] = "insufficient_live_trades"
                checks["samples_ok"] = False
                alerts.append(f"mode_drift_insufficient_live:{mode}")
            else:
                active_modes += 1
                row["active"] = True
                compared_modes += 1
                wr_gap = self._safe_float(row["gaps"]["win_rate_abs"], 0.0)
                pf_gap = self._safe_float(row["gaps"]["profit_factor_abs"], 0.0)
                row["checks"]["win_rate_gap_ok"] = bool(wr_gap <= max_wr_gap)
                row["checks"]["profit_factor_gap_ok"] = bool(pf_gap <= max_pf_gap)
                if not bool(row["checks"]["win_rate_gap_ok"]):
                    checks["win_rate_gap_ok"] = False
                    alerts.append(f"mode_drift_win_rate:{mode}")
                if not bool(row["checks"]["profit_factor_gap_ok"]):
                    checks["profit_factor_gap_ok"] = False
                    alerts.append(f"mode_drift_profit_factor:{mode}")
                row["reason"] = "ok" if (row["checks"]["win_rate_gap_ok"] and row["checks"]["profit_factor_gap_ok"]) else "drift_breach"
            mode_rows[mode] = row

        active = bool(compared_modes > 0)
        if not scope_modes:
            alerts.append("mode_drift_scope_empty")
        elif not active:
            alerts.append("mode_drift_inactive")

        return {
            "active": active,
            "window_days": window_days,
            "runtime_mode": runtime_mode,
            "focus_runtime_mode_only": focus_runtime_only,
            "min_live_trades": min_live_trades,
            "checks": checks,
            "thresholds": {
                "mode_drift_win_rate_max_gap": max_wr_gap,
                "mode_drift_profit_factor_max_gap": max_pf_gap,
            },
            "summary": {
                "scope_modes": int(len(scope_modes)),
                "compared_modes": int(compared_modes),
                "active_modes": int(active_modes),
            },
            "alerts": alerts,
            "modes": mode_rows,
        }

    def _stress_matrix_trend_metrics(self, *, as_of: date) -> dict[str, Any]:
        val = self.settings.validation if isinstance(self.settings.validation, dict) else {}
        enabled = bool(val.get("ops_stress_matrix_trend_enabled", True))
        if not enabled:
            return {
                "active": False,
                "enabled": False,
                "window_runs": 0,
                "samples": 0,
                "min_samples": 0,
                "checks": {},
                "thresholds": {},
                "metrics": {},
                "alerts": [],
                "series": [],
            }

        window_runs = max(2, int(val.get("ops_stress_matrix_trend_window_runs", 8)))
        min_samples = max(2, int(val.get("ops_stress_matrix_trend_min_runs", 3)))
        robustness_drop_max = self._safe_float(val.get("ops_stress_matrix_robustness_drop_max", 0.15), 0.15)
        annual_drop_max = self._safe_float(val.get("ops_stress_matrix_annual_return_drop_max", 0.08), 0.08)
        drawdown_rise_max = self._safe_float(val.get("ops_stress_matrix_drawdown_rise_max", 0.08), 0.08)
        fail_ratio_max = self._safe_float(val.get("ops_stress_matrix_fail_ratio_max", 0.50), 0.50)

        review_dir = self.output_dir / "review"
        runs: list[dict[str, Any]] = []
        for path in sorted(review_dir.glob("*_mode_stress_matrix.json")):
            dtag = path.name.replace("_mode_stress_matrix.json", "")
            try:
                d = date.fromisoformat(dtag)
            except Exception:
                continue
            if d > as_of:
                continue
            payload = self.load_json_safely(path)
            if not payload:
                continue
            matrix = payload.get("matrix", []) if isinstance(payload.get("matrix", []), list) else []
            summary = payload.get("mode_summary", []) if isinstance(payload.get("mode_summary", []), list) else []
            best_mode = str(payload.get("best_mode", "")).strip()
            best_row = None
            for row in summary:
                if not isinstance(row, dict):
                    continue
                if str(row.get("mode", "")).strip() == best_mode:
                    best_row = row
                    break
            if best_row is None and summary and isinstance(summary[0], dict):
                best_row = summary[0]

            matrix_rows = int(len(matrix))
            fail_rows = 0
            for row in matrix:
                if not isinstance(row, dict):
                    continue
                status = str(row.get("status", "ok")).strip().lower()
                if status != "ok":
                    fail_rows += 1

            runs.append(
                {
                    "date": d.isoformat(),
                    "best_mode": best_mode,
                    "robustness_score": self._safe_float(
                        ((best_row or {}).get("robustness_score", 0.0) if isinstance(best_row, dict) else 0.0),
                        0.0,
                    ),
                    "avg_annual_return": self._safe_float(
                        ((best_row or {}).get("avg_annual_return", 0.0) if isinstance(best_row, dict) else 0.0),
                        0.0,
                    ),
                    "worst_drawdown": self._safe_float(
                        ((best_row or {}).get("worst_drawdown", 0.0) if isinstance(best_row, dict) else 0.0),
                        0.0,
                    ),
                    "fail_ratio": self._ratio(fail_rows, matrix_rows),
                    "matrix_rows": matrix_rows,
                    "fail_rows": fail_rows,
                }
            )

        runs = sorted(runs, key=lambda x: str(x.get("date", "")))
        if len(runs) > window_runs:
            runs = runs[-window_runs:]
        samples = len(runs)
        active = bool(samples >= min_samples)

        checks = {
            "robustness_drop_ok": True,
            "annual_return_drop_ok": True,
            "drawdown_rise_ok": True,
            "fail_ratio_ok": True,
        }
        alerts: list[str] = []
        metrics: dict[str, Any] = {
            "current": {},
            "baseline": {},
            "delta": {},
        }
        if active:
            current = runs[-1]
            baseline_rows = runs[:-1]
            if not baseline_rows:
                alerts.append("stress_matrix_baseline_missing")
            else:
                b_n = float(len(baseline_rows))
                baseline = {
                    "robustness_score": float(sum(self._safe_float(x.get("robustness_score", 0.0), 0.0) for x in baseline_rows) / b_n),
                    "avg_annual_return": float(sum(self._safe_float(x.get("avg_annual_return", 0.0), 0.0) for x in baseline_rows) / b_n),
                    "worst_drawdown": float(sum(self._safe_float(x.get("worst_drawdown", 0.0), 0.0) for x in baseline_rows) / b_n),
                    "fail_ratio": float(sum(self._safe_float(x.get("fail_ratio", 0.0), 0.0) for x in baseline_rows) / b_n),
                }
                delta = {
                    "robustness_drop": float(
                        self._safe_float(baseline.get("robustness_score", 0.0), 0.0)
                        - self._safe_float(current.get("robustness_score", 0.0), 0.0)
                    ),
                    "annual_return_drop": float(
                        self._safe_float(baseline.get("avg_annual_return", 0.0), 0.0)
                        - self._safe_float(current.get("avg_annual_return", 0.0), 0.0)
                    ),
                    "drawdown_rise": float(
                        self._safe_float(current.get("worst_drawdown", 0.0), 0.0)
                        - self._safe_float(baseline.get("worst_drawdown", 0.0), 0.0)
                    ),
                    "fail_ratio_rise": float(
                        self._safe_float(current.get("fail_ratio", 0.0), 0.0)
                        - self._safe_float(baseline.get("fail_ratio", 0.0), 0.0)
                    ),
                }
                checks["robustness_drop_ok"] = bool(self._safe_float(delta.get("robustness_drop", 0.0), 0.0) <= robustness_drop_max)
                checks["annual_return_drop_ok"] = bool(self._safe_float(delta.get("annual_return_drop", 0.0), 0.0) <= annual_drop_max)
                checks["drawdown_rise_ok"] = bool(self._safe_float(delta.get("drawdown_rise", 0.0), 0.0) <= drawdown_rise_max)
                checks["fail_ratio_ok"] = bool(self._safe_float(current.get("fail_ratio", 0.0), 0.0) <= fail_ratio_max)
                if not checks["robustness_drop_ok"]:
                    alerts.append("stress_matrix_robustness_drop")
                if not checks["annual_return_drop_ok"]:
                    alerts.append("stress_matrix_annual_return_drop")
                if not checks["drawdown_rise_ok"]:
                    alerts.append("stress_matrix_drawdown_rise")
                if not checks["fail_ratio_ok"]:
                    alerts.append("stress_matrix_fail_ratio_high")
                metrics = {
                    "current": current,
                    "baseline": baseline,
                    "delta": delta,
                }
        else:
            alerts.append("stress_matrix_insufficient_samples")

        return {
            "active": active,
            "enabled": True,
            "window_runs": int(window_runs),
            "samples": int(samples),
            "min_samples": int(min_samples),
            "checks": checks,
            "thresholds": {
                "ops_stress_matrix_robustness_drop_max": float(robustness_drop_max),
                "ops_stress_matrix_annual_return_drop_max": float(annual_drop_max),
                "ops_stress_matrix_drawdown_rise_max": float(drawdown_rise_max),
                "ops_stress_matrix_fail_ratio_max": float(fail_ratio_max),
            },
            "metrics": metrics,
            "alerts": alerts,
            "series": runs[-10:],
        }

    def _temporal_audit_metrics(self, *, as_of: date) -> dict[str, Any]:
        val = self.settings.validation if isinstance(self.settings.validation, dict) else {}
        enabled = bool(val.get("ops_temporal_audit_enabled", True))
        if not enabled:
            return {
                "active": False,
                "enabled": False,
                "samples": 0,
                "min_samples": 0,
                "checks": {},
                "thresholds": {},
                "metrics": {},
                "alerts": [],
                "series": [],
            }

        lookback_days = max(1, int(val.get("ops_temporal_audit_lookback_days", 45)))
        min_samples = max(1, int(val.get("ops_temporal_audit_min_samples", 1)))
        missing_ratio_max = self._safe_float(val.get("ops_temporal_audit_missing_ratio_max", 0.20), 0.20)
        leak_ratio_max = self._safe_float(val.get("ops_temporal_audit_leak_ratio_max", 0.0), 0.0)
        autofix_enabled = bool(val.get("ops_temporal_audit_autofix_enabled", True))
        autofix_max_writes = max(0, int(val.get("ops_temporal_audit_autofix_max_writes", 3)))
        autofix_fix_strict_cutoff = bool(val.get("ops_temporal_audit_autofix_fix_strict_cutoff", True))
        autofix_require_safe = bool(val.get("ops_temporal_audit_autofix_require_safe", True))
        autofix_artifact_policy = self._artifact_governance_profile(
            profile_name="temporal_autofix_patch",
            fallback_retention_days=max(1, int(val.get("ops_temporal_audit_autofix_patch_retention_days", 30))),
            fallback_checksum_index_enabled=self._safe_bool(
                val.get("ops_temporal_audit_autofix_patch_checksum_index_enabled", True),
                True,
            ),
        )
        autofix_patch_retention_days = int(autofix_artifact_policy.get("retention_days", 30))
        autofix_patch_checksum_index_enabled = bool(
            autofix_artifact_policy.get("checksum_index_enabled", True)
        )

        def _parse_date(v: Any) -> date | None:
            if not v:
                return None
            try:
                return date.fromisoformat(str(v))
            except Exception:
                return None

        def _parse_ts_date(v: Any) -> date | None:
            if not v:
                return None
            text = str(v).strip()
            if not text:
                return None
            ts = text
            if ts.endswith("Z"):
                ts = ts[:-1] + "+00:00"
            try:
                return date.fromisoformat(ts)
            except Exception:
                pass
            try:
                return date.fromisoformat(ts[:10])
            except Exception:
                return None

        def _pick_text_with_source(*vals: tuple[str, Any]) -> tuple[str, str]:
            for source, item in vals:
                text = str(item or "").strip()
                if text:
                    return text, str(source).strip()
            return "", ""

        def _resolve_summary_path(ref: Any) -> Path | None:
            raw = str(ref or "").strip()
            if not raw:
                return None
            p = Path(raw)
            if p.is_absolute():
                return p
            cands = [self.output_dir / p, self.output_dir.parent / p]
            for cand in cands:
                if cand.exists():
                    return cand
            return cands[0]

        def _candidate_temporal_ok(fields: dict[str, str]) -> tuple[bool, str]:
            cutoff_date = _parse_date(fields.get("cutoff_date"))
            cutoff_ts = _parse_ts_date(fields.get("cutoff_ts"))
            bar_max = _parse_ts_date(fields.get("bar_max_ts"))
            news_max = _parse_ts_date(fields.get("news_max_ts"))
            report_max = _parse_ts_date(fields.get("report_max_ts"))
            if cutoff_date is None or cutoff_ts is None or bar_max is None or news_max is None or report_max is None:
                return False, "candidate_missing_or_invalid"
            if autofix_require_safe:
                for probe in (cutoff_ts, bar_max, news_max, report_max):
                    if probe > cutoff_date:
                        return False, "unsafe_temporal_candidate"
            return True, ""

        manifest_dir = self.output_dir / "artifacts" / "manifests"
        manifests = sorted(list(manifest_dir.glob("strategy_lab_*.json")) + list(manifest_dir.glob("research_backtest_*.json")))
        series: list[dict[str, Any]] = []
        samples = 0
        missing_count = 0
        leak_count = 0
        strict_disabled_count = 0
        autofix_attempted_count = 0
        autofix_applied_count = 0
        autofix_skipped_count = 0
        autofix_failed_count = 0
        autofix_writes = 0

        for path in manifests:
            payload = self.load_json_safely(path)
            if not payload:
                continue
            created_text = str(payload.get("created_at", "")).strip()
            created_date = _parse_ts_date(created_text)
            if created_date is None:
                continue
            if created_date > as_of:
                continue
            if (as_of - created_date).days > lookback_days:
                continue

            samples += 1
            metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata", {}), dict) else {}
            checks = payload.get("checks", {}) if isinstance(payload.get("checks", {}), dict) else {}
            artifacts = payload.get("artifacts", {}) if isinstance(payload.get("artifacts", {}), dict) else {}

            summary_payload: dict[str, Any] = {}
            summary_fetch_stats: dict[str, Any] = {}
            summary_path = _resolve_summary_path(artifacts.get("summary"))
            if summary_path is not None:
                summary_payload = self.load_json_safely(summary_path)
                if not isinstance(summary_payload, dict):
                    summary_payload = {}
                summary_fetch_stats = (
                    summary_payload.get("data_fetch_stats", {})
                    if isinstance(summary_payload.get("data_fetch_stats", {}), dict)
                    else {}
                )

            strict_cutoff = bool(checks.get("strict_cutoff_enforced", False))
            cutoff_date = _parse_date(metadata.get("cutoff_date"))
            cutoff_ts = _parse_ts_date(metadata.get("cutoff_ts"))
            bar_max = _parse_ts_date(metadata.get("bar_max_ts"))
            news_max = _parse_ts_date(metadata.get("news_max_ts"))
            report_max = _parse_ts_date(metadata.get("report_max_ts"))
            missing = bool(
                cutoff_date is None
                or cutoff_ts is None
                or bar_max is None
                or news_max is None
                or report_max is None
            )

            autofix_detail: dict[str, Any] = {
                "enabled": bool(autofix_enabled),
                "attempted": False,
                "applied": False,
                "reason": "",
                "error": "",
                "patched_fields": [],
                "strict_patched": False,
                "manifest_path": str(path),
                "summary_path": str(summary_path) if summary_path is not None else "",
                "patch_delta": {},
                "strict_delta": {},
            }

            if autofix_enabled and (missing or (autofix_fix_strict_cutoff and (not strict_cutoff))):
                autofix_attempted_count += 1
                autofix_detail["attempted"] = True
                if autofix_writes >= autofix_max_writes:
                    autofix_skipped_count += 1
                    autofix_detail["reason"] = "max_writes_reached"
                else:
                    candidate_sources: dict[str, str] = {}
                    candidate_fields = {
                        "cutoff_date": "",
                        "cutoff_ts": "",
                        "bar_max_ts": "",
                        "news_max_ts": "",
                        "report_max_ts": "",
                    }
                    for key in ("cutoff_date", "cutoff_ts", "bar_max_ts", "news_max_ts", "report_max_ts"):
                        txt, source = _pick_text_with_source(
                            ("metadata", metadata.get(key)),
                            ("summary", summary_payload.get(key)),
                            ("summary_fetch_stats", summary_fetch_stats.get(key)),
                        )
                        candidate_fields[key] = txt
                        candidate_sources[key] = source
                    if (not candidate_fields["cutoff_ts"]) and candidate_fields["cutoff_date"]:
                        candidate_fields["cutoff_ts"] = f"{candidate_fields['cutoff_date']}T23:59:59"
                        candidate_sources["cutoff_ts"] = "derived_cutoff_date"

                    current_fields = {
                        "cutoff_date": str(metadata.get("cutoff_date", "")).strip(),
                        "cutoff_ts": str(metadata.get("cutoff_ts", "")).strip(),
                        "bar_max_ts": str(metadata.get("bar_max_ts", "")).strip(),
                        "news_max_ts": str(metadata.get("news_max_ts", "")).strip(),
                        "report_max_ts": str(metadata.get("report_max_ts", "")).strip(),
                    }
                    patch_fields: dict[str, str] = {}
                    patch_delta: dict[str, dict[str, Any]] = {}
                    for key in ("cutoff_date", "cutoff_ts", "bar_max_ts", "news_max_ts", "report_max_ts"):
                        cur_txt = str(current_fields.get(key, "")).strip()
                        cand_txt = str(candidate_fields.get(key, "")).strip()
                        if (not cur_txt) and cand_txt:
                            patch_fields[key] = cand_txt
                            patch_delta[key] = {
                                "before": cur_txt,
                                "after": cand_txt,
                                "source": str(candidate_sources.get(key, "")),
                            }
                    autofix_detail["patch_delta"] = patch_delta

                    strict_from_summary = bool(
                        summary_fetch_stats.get(
                            "strict_cutoff_enforced",
                            summary_payload.get("strict_cutoff_enforced", False),
                        )
                    )
                    strict_source = (
                        "summary_fetch_stats"
                        if "strict_cutoff_enforced" in summary_fetch_stats
                        else ("summary" if "strict_cutoff_enforced" in summary_payload else "")
                    )
                    strict_patch = bool((not strict_cutoff) and autofix_fix_strict_cutoff and strict_from_summary)
                    if strict_patch:
                        autofix_detail["strict_delta"] = {
                            "before": bool(strict_cutoff),
                            "after": True,
                            "source": strict_source,
                        }

                    if (not patch_fields) and (not strict_patch):
                        autofix_skipped_count += 1
                        autofix_detail["reason"] = "no_patchable_fields"
                    else:
                        merged_fields = {
                            "cutoff_date": str(metadata.get("cutoff_date", "")).strip(),
                            "cutoff_ts": str(metadata.get("cutoff_ts", "")).strip(),
                            "bar_max_ts": str(metadata.get("bar_max_ts", "")).strip(),
                            "news_max_ts": str(metadata.get("news_max_ts", "")).strip(),
                            "report_max_ts": str(metadata.get("report_max_ts", "")).strip(),
                        }
                        for k, v in patch_fields.items():
                            merged_fields[k] = str(v).strip()
                        safe_ok, safe_reason = _candidate_temporal_ok(merged_fields)
                        if not safe_ok:
                            autofix_skipped_count += 1
                            autofix_detail["reason"] = safe_reason
                        else:
                            patched_payload = dict(payload)
                            patched_meta = dict(metadata)
                            patched_checks = dict(checks)
                            for k, v in patch_fields.items():
                                patched_meta[k] = str(v)
                            if strict_patch:
                                patched_checks["strict_cutoff_enforced"] = True
                            patched_payload["metadata"] = patched_meta
                            patched_payload["checks"] = patched_checks
                            try:
                                write_json(path, patched_payload)
                                payload = patched_payload
                                metadata = patched_meta
                                checks = patched_checks
                                strict_cutoff = bool(checks.get("strict_cutoff_enforced", False))
                                cutoff_date = _parse_date(metadata.get("cutoff_date"))
                                cutoff_ts = _parse_ts_date(metadata.get("cutoff_ts"))
                                bar_max = _parse_ts_date(metadata.get("bar_max_ts"))
                                news_max = _parse_ts_date(metadata.get("news_max_ts"))
                                report_max = _parse_ts_date(metadata.get("report_max_ts"))
                                missing = bool(
                                    cutoff_date is None
                                    or cutoff_ts is None
                                    or bar_max is None
                                    or news_max is None
                                    or report_max is None
                                )
                                autofix_applied_count += 1
                                autofix_writes += 1
                                autofix_detail["applied"] = True
                                autofix_detail["reason"] = "applied"
                                autofix_detail["patched_fields"] = sorted(list(patch_fields.keys()))
                                autofix_detail["strict_patched"] = bool(strict_patch)
                            except Exception as exc:
                                autofix_failed_count += 1
                                autofix_detail["reason"] = "write_failed"
                                autofix_detail["error"] = f"{type(exc).__name__}:{exc}"

            strict_cutoff = bool(checks.get("strict_cutoff_enforced", False))
            cutoff_date = _parse_date(metadata.get("cutoff_date"))
            cutoff_ts = _parse_ts_date(metadata.get("cutoff_ts"))
            bar_max = _parse_ts_date(metadata.get("bar_max_ts"))
            news_max = _parse_ts_date(metadata.get("news_max_ts"))
            report_max = _parse_ts_date(metadata.get("report_max_ts"))
            missing = bool(
                cutoff_date is None
                or cutoff_ts is None
                or bar_max is None
                or news_max is None
                or report_max is None
            )
            leak = False
            if not missing and cutoff_date is not None:
                for probe in (cutoff_ts, bar_max, news_max, report_max):
                    if probe is not None and probe > cutoff_date:
                        leak = True
                        break
            if missing:
                missing_count += 1
            if leak:
                leak_count += 1
            if not strict_cutoff:
                strict_disabled_count += 1

            series.append(
                {
                    "date": created_date.isoformat(),
                    "run_type": str(payload.get("run_type", "")),
                    "run_id": str(payload.get("run_id", "")),
                    "manifest_path": str(path),
                    "missing": bool(missing),
                    "leak": bool(leak),
                    "strict_cutoff_enforced": bool(strict_cutoff),
                    "cutoff_date": metadata.get("cutoff_date", ""),
                    "cutoff_ts": metadata.get("cutoff_ts", ""),
                    "bar_max_ts": metadata.get("bar_max_ts", ""),
                    "news_max_ts": metadata.get("news_max_ts", ""),
                    "report_max_ts": metadata.get("report_max_ts", ""),
                    "autofix": autofix_detail,
                }
            )

        active = bool(samples >= min_samples)
        missing_ratio = self._ratio(missing_count, samples)
        leak_ratio = self._ratio(leak_count, samples)
        strict_disabled_ratio = self._ratio(strict_disabled_count, samples)
        autofix_attempted_ratio = self._ratio(autofix_attempted_count, samples)
        autofix_applied_ratio = self._ratio(autofix_applied_count, samples)
        autofix_failed_ratio = self._ratio(autofix_failed_count, max(1, autofix_attempted_count))
        checks = {
            "missing_ratio_ok": True,
            "leak_ratio_ok": True,
            "strict_cutoff_ok": True,
        }
        alerts: list[str] = []
        if active:
            checks["missing_ratio_ok"] = bool(missing_ratio <= missing_ratio_max)
            checks["leak_ratio_ok"] = bool(leak_ratio <= leak_ratio_max)
            checks["strict_cutoff_ok"] = bool(strict_disabled_count == 0)
            if not checks["missing_ratio_ok"]:
                alerts.append("temporal_audit_missing_high")
            if not checks["leak_ratio_ok"]:
                alerts.append("temporal_audit_leak_detected")
            if not checks["strict_cutoff_ok"]:
                alerts.append("temporal_audit_strict_cutoff_disabled")
            if autofix_failed_count > 0:
                alerts.append("temporal_audit_autofix_failed")
        else:
            alerts.append("temporal_audit_insufficient_samples")

        autofix_artifact = self._write_temporal_autofix_artifact(
            as_of=as_of,
            series=series,
            retention_days=autofix_patch_retention_days,
            checksum_index_enabled=autofix_patch_checksum_index_enabled,
        )
        autofix_artifact_failed = (
            int(autofix_attempted_count) > 0
            and (not bool(autofix_artifact.get("written", False)))
            and str(autofix_artifact.get("reason", "")).startswith("write_failed")
        )
        autofix_artifact_rotation_failed = bool(autofix_artifact.get("rotation_failed", False))
        autofix_artifact_checksum_index_failed = bool(autofix_artifact.get("checksum_index_failed", False))
        if autofix_artifact_failed and ("temporal_audit_autofix_artifact_failed" not in alerts):
            alerts.append("temporal_audit_autofix_artifact_failed")
        if autofix_artifact_rotation_failed and ("temporal_audit_autofix_rotation_failed" not in alerts):
            alerts.append("temporal_audit_autofix_rotation_failed")
        if autofix_artifact_checksum_index_failed and ("temporal_audit_autofix_checksum_index_failed" not in alerts):
            alerts.append("temporal_audit_autofix_checksum_index_failed")

        return {
            "active": active,
            "enabled": True,
            "lookback_days": int(lookback_days),
            "autofix_enabled": bool(autofix_enabled),
            "samples": int(samples),
            "min_samples": int(min_samples),
            "checks": checks,
            "thresholds": {
                "ops_temporal_audit_missing_ratio_max": float(missing_ratio_max),
                "ops_temporal_audit_leak_ratio_max": float(leak_ratio_max),
                "ops_temporal_audit_autofix_max_writes": int(autofix_max_writes),
                "ops_temporal_audit_autofix_fix_strict_cutoff": bool(autofix_fix_strict_cutoff),
                "ops_temporal_audit_autofix_require_safe": bool(autofix_require_safe),
                "ops_temporal_audit_autofix_patch_retention_days": int(autofix_patch_retention_days),
                "ops_temporal_audit_autofix_patch_checksum_index_enabled": bool(
                    autofix_patch_checksum_index_enabled
                ),
            },
            "metrics": {
                "missing_count": int(missing_count),
                "missing_ratio": float(missing_ratio),
                "leak_count": int(leak_count),
                "leak_ratio": float(leak_ratio),
                "strict_cutoff_disabled_count": int(strict_disabled_count),
                "strict_cutoff_disabled_ratio": float(strict_disabled_ratio),
                "autofix_attempted_count": int(autofix_attempted_count),
                "autofix_attempted_ratio": float(autofix_attempted_ratio),
                "autofix_applied_count": int(autofix_applied_count),
                "autofix_applied_ratio": float(autofix_applied_ratio),
                "autofix_failed_count": int(autofix_failed_count),
                "autofix_failed_ratio": float(autofix_failed_ratio),
                "autofix_skipped_count": int(autofix_skipped_count),
                "autofix_writes": int(autofix_writes),
                "autofix_artifact_written": bool(autofix_artifact.get("written", False)),
                "autofix_artifact_total_events": int(autofix_artifact.get("total_events", 0)),
                "autofix_artifact_applied_count": int(autofix_artifact.get("applied_count", 0)),
                "autofix_artifact_failed_count": int(autofix_artifact.get("failed_count", 0)),
                "autofix_artifact_skipped_count": int(autofix_artifact.get("skipped_count", 0)),
                "autofix_artifact_failed": bool(autofix_artifact_failed),
                "autofix_artifact_retention_days": int(autofix_artifact.get("retention_days", autofix_patch_retention_days)),
                "autofix_artifact_rotated_out_count": int(autofix_artifact.get("rotated_out_count", 0)),
                "autofix_artifact_rotation_failed": bool(autofix_artifact_rotation_failed),
                "autofix_artifact_checksum_index_enabled": bool(
                    autofix_artifact.get("checksum_index_enabled", autofix_patch_checksum_index_enabled)
                ),
                "autofix_artifact_checksum_index_written": bool(
                    autofix_artifact.get("checksum_index_written", False)
                ),
                "autofix_artifact_checksum_index_entries": int(
                    autofix_artifact.get("checksum_index_entries", 0)
                ),
                "autofix_artifact_checksum_index_failed": bool(autofix_artifact_checksum_index_failed),
            },
            "alerts": alerts,
            "artifacts": {
                "autofix_patch": {
                    "written": bool(autofix_artifact.get("written", False)),
                    "json": str(autofix_artifact.get("json", "")),
                    "md": str(autofix_artifact.get("md", "")),
                    "total_events": int(autofix_artifact.get("total_events", 0)),
                    "applied_count": int(autofix_artifact.get("applied_count", 0)),
                    "failed_count": int(autofix_artifact.get("failed_count", 0)),
                    "skipped_count": int(autofix_artifact.get("skipped_count", 0)),
                    "retention_days": int(autofix_artifact.get("retention_days", autofix_patch_retention_days)),
                    "rotated_out_count": int(autofix_artifact.get("rotated_out_count", 0)),
                    "rotated_out_dates": [
                        str(x) for x in autofix_artifact.get("rotated_out_dates", [])
                    ],
                    "rotation_failed": bool(autofix_artifact.get("rotation_failed", False)),
                    "checksum_index_enabled": bool(
                        autofix_artifact.get("checksum_index_enabled", autofix_patch_checksum_index_enabled)
                    ),
                    "checksum_index_written": bool(autofix_artifact.get("checksum_index_written", False)),
                    "checksum_index_path": str(autofix_artifact.get("checksum_index_path", "")),
                    "checksum_index_entries": int(autofix_artifact.get("checksum_index_entries", 0)),
                    "checksum_index_failed": bool(autofix_artifact.get("checksum_index_failed", False)),
                    "reason": str(autofix_artifact.get("reason", "")),
                }
            },
            "series": series[-10:],
        }

    def _state_stability_metrics(self, *, as_of: date) -> dict[str, Any]:
        val = self.settings.validation if isinstance(self.settings.validation, dict) else {}
        window_days = max(3, int(val.get("mode_switch_window_days", 20)))
        min_samples = max(1, int(val.get("ops_state_min_samples", 5)))
        switch_rate_max = self._safe_float(val.get("mode_switch_max_rate", 0.45), 0.45)
        risk_floor = self._safe_float(
            val.get("ops_risk_multiplier_floor", val.get("execution_min_risk_multiplier", 0.20)),
            0.20,
        )
        risk_drift_max = self._safe_float(val.get("ops_risk_multiplier_drift_max", 0.30), 0.30)
        source_floor = self._safe_float(
            val.get("ops_source_confidence_floor", val.get("source_confidence_min", 0.75)),
            0.75,
        )
        mode_health_fail_days_max = max(0, int(val.get("ops_mode_health_fail_days_max", 2)))

        rows = self._load_mode_feedback_series(as_of=as_of, window_days=window_days)
        samples = len(rows)
        modes = [str(x.get("runtime_mode", "")).strip() for x in rows if str(x.get("runtime_mode", "")).strip()]
        risk_values = [self._safe_float(x.get("risk_multiplier", 1.0), 1.0) for x in rows]
        source_values = [self._safe_float(x.get("source_confidence_score", 1.0), 1.0) for x in rows]
        mode_health_fail_days = sum(1 for x in rows if not bool(x.get("mode_health_passed", True)))

        switch_count = 0
        if len(modes) >= 2:
            switch_count = sum(1 for i in range(1, len(modes)) if modes[i] != modes[i - 1])
        switch_rate = float(switch_count / max(1, len(modes) - 1)) if len(modes) >= 2 else 0.0

        risk_min = min(risk_values) if risk_values else 1.0
        risk_avg = (sum(risk_values) / len(risk_values)) if risk_values else 1.0
        source_min = min(source_values) if source_values else 1.0
        source_avg = (sum(source_values) / len(source_values)) if source_values else 1.0

        risk_drift = 0.0
        if len(risk_values) >= 6:
            risk_drift = (sum(risk_values[-3:]) / 3.0) - (sum(risk_values[-6:-3]) / 3.0)
        elif len(risk_values) >= 4:
            risk_drift = (sum(risk_values[-2:]) / 2.0) - (sum(risk_values[-4:-2]) / 2.0)

        active = samples >= min_samples
        checks = {
            "switch_rate_ok": True,
            "risk_multiplier_floor_ok": True,
            "risk_multiplier_drift_ok": True,
            "source_confidence_floor_ok": True,
            "mode_health_fail_days_ok": True,
        }
        alerts: list[str] = []
        if active:
            checks["switch_rate_ok"] = bool(switch_rate <= switch_rate_max)
            checks["risk_multiplier_floor_ok"] = bool(risk_min >= risk_floor)
            checks["risk_multiplier_drift_ok"] = bool(abs(risk_drift) <= risk_drift_max)
            checks["source_confidence_floor_ok"] = bool(source_min >= source_floor)
            checks["mode_health_fail_days_ok"] = bool(mode_health_fail_days <= mode_health_fail_days_max)
            if not checks["switch_rate_ok"]:
                alerts.append("mode_switch_rate_high")
            if not checks["risk_multiplier_floor_ok"]:
                alerts.append("risk_multiplier_too_low")
            if not checks["risk_multiplier_drift_ok"]:
                alerts.append("risk_multiplier_drift_high")
            if not checks["source_confidence_floor_ok"]:
                alerts.append("source_confidence_too_low")
            if not checks["mode_health_fail_days_ok"]:
                alerts.append("mode_health_fail_days_high")
        else:
            alerts.append("insufficient_mode_feedback_samples")

        return {
            "active": active,
            "window_days": window_days,
            "samples": samples,
            "min_samples": min_samples,
            "metrics": {
                "switch_count": switch_count,
                "switch_rate": switch_rate,
                "risk_multiplier_min": risk_min,
                "risk_multiplier_avg": risk_avg,
                "risk_multiplier_drift": risk_drift,
                "source_confidence_min": source_min,
                "source_confidence_avg": source_avg,
                "mode_health_fail_days": mode_health_fail_days,
            },
            "thresholds": {
                "mode_switch_max_rate": switch_rate_max,
                "ops_risk_multiplier_floor": risk_floor,
                "ops_risk_multiplier_drift_max": risk_drift_max,
                "ops_source_confidence_floor": source_floor,
                "ops_mode_health_fail_days_max": mode_health_fail_days_max,
            },
            "checks": checks,
            "alerts": alerts,
            "series": rows[-10:],
        }

    def ops_report(self, as_of: date, window_days: int = 7) -> dict[str, Any]:
        d = as_of.isoformat()
        wd = max(1, int(window_days))

        scheduler_state = self.load_json_safely(self.output_dir / "logs" / "scheduler_state.json")
        latest_tests = self._latest_test_result()
        gate = self.gate_report(as_of=as_of, run_tests=False, run_review_if_missing=False)
        state_stability = self._state_stability_metrics(as_of=as_of)
        state_checks = state_stability.get("checks", {}) if isinstance(state_stability.get("checks", {}), dict) else {}
        state_all_ok = all(bool(v) for v in state_checks.values())
        temporal_audit = gate.get("temporal_audit", {}) if isinstance(gate.get("temporal_audit", {}), dict) else {}
        temporal_active = bool(temporal_audit.get("active", False))
        temporal_checks = (
            temporal_audit.get("checks", {}) if isinstance(temporal_audit.get("checks", {}), dict) else {}
        )
        temporal_all_ok = all(bool(v) for v in temporal_checks.values()) if temporal_active else True
        slot_anomaly = gate.get("slot_anomaly", {}) if isinstance(gate.get("slot_anomaly", {}), dict) else {}
        slot_active = bool(slot_anomaly.get("active", False))
        slot_checks = slot_anomaly.get("checks", {}) if isinstance(slot_anomaly.get("checks", {}), dict) else {}
        slot_all_ok = all(bool(v) for v in slot_checks.values()) if slot_active else True
        mode_drift = gate.get("mode_drift", {}) if isinstance(gate.get("mode_drift", {}), dict) else {}
        drift_active = bool(mode_drift.get("active", False))
        drift_checks = mode_drift.get("checks", {}) if isinstance(mode_drift.get("checks", {}), dict) else {}
        drift_all_ok = all(bool(v) for v in drift_checks.values()) if drift_active else True
        stress_matrix_trend = (
            gate.get("stress_matrix_trend", {}) if isinstance(gate.get("stress_matrix_trend", {}), dict) else {}
        )
        stress_active = bool(stress_matrix_trend.get("active", False))
        stress_checks = (
            stress_matrix_trend.get("checks", {}) if isinstance(stress_matrix_trend.get("checks", {}), dict) else {}
        )
        stress_all_ok = all(bool(v) for v in stress_checks.values()) if stress_active else True
        stress_autorun_history = (
            gate.get("stress_autorun_history", {})
            if isinstance(gate.get("stress_autorun_history", {}), dict)
            else {}
        )
        stress_autorun_adaptive = (
            gate.get("stress_autorun_adaptive", {})
            if isinstance(gate.get("stress_autorun_adaptive", {}), dict)
            else {}
        )
        stress_auto_adaptive_active = bool(stress_autorun_adaptive.get("active", False))
        stress_auto_adaptive_checks = (
            stress_autorun_adaptive.get("checks", {})
            if isinstance(stress_autorun_adaptive.get("checks", {}), dict)
            else {}
        )
        stress_auto_adaptive_all_ok = (
            all(bool(v) for v in stress_auto_adaptive_checks.values()) if stress_auto_adaptive_active else True
        )
        stress_autorun_reason_drift = (
            gate.get("stress_autorun_reason_drift", {})
            if isinstance(gate.get("stress_autorun_reason_drift", {}), dict)
            else {}
        )
        stress_reason_drift_active = bool(stress_autorun_reason_drift.get("active", False))
        stress_reason_drift_checks = (
            stress_autorun_reason_drift.get("checks", {})
            if isinstance(stress_autorun_reason_drift.get("checks", {}), dict)
            else {}
        )
        stress_reason_drift_all_ok = (
            all(bool(v) for v in stress_reason_drift_checks.values()) if stress_reason_drift_active else True
        )
        reconcile_drift = gate.get("reconcile_drift", {}) if isinstance(gate.get("reconcile_drift", {}), dict) else {}
        reconcile_active = bool(reconcile_drift.get("active", False))
        reconcile_checks = (
            reconcile_drift.get("checks", {}) if isinstance(reconcile_drift.get("checks", {}), dict) else {}
        )
        reconcile_all_ok = all(bool(v) for v in reconcile_checks.values()) if reconcile_active else True
        artifact_governance = (
            gate.get("artifact_governance", {}) if isinstance(gate.get("artifact_governance", {}), dict) else {}
        )
        artifact_governance_active = bool(artifact_governance.get("active", False))
        artifact_governance_checks = (
            artifact_governance.get("checks", {})
            if isinstance(artifact_governance.get("checks", {}), dict)
            else {}
        )
        artifact_governance_all_ok = (
            all(bool(v) for v in artifact_governance_checks.values()) if artifact_governance_active else True
        )
        rollback_rec = (
            gate.get("rollback_recommendation", {})
            if isinstance(gate.get("rollback_recommendation", {}), dict)
            else {}
        )
        rollback_level = str(rollback_rec.get("level", "none")).strip().lower() or "none"
        rollback_active = bool(rollback_rec.get("active", False))
        rollback_anchor_ready = bool(rollback_rec.get("anchor_ready", True))

        history = []
        healthy_days = 0
        for i in range(wd):
            day = as_of - timedelta(days=i)
            require_review = i == 0
            h = self.health_check(day, require_review)
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
        if (
            not gate["passed"]
            or health_ratio < 0.8
            or (bool(state_stability.get("active", False)) and not state_all_ok)
            or (temporal_active and not temporal_all_ok)
            or (slot_active and not slot_all_ok)
            or (drift_active and not drift_all_ok)
            or (stress_active and not stress_all_ok)
            or (stress_auto_adaptive_active and not stress_auto_adaptive_all_ok)
            or (stress_reason_drift_active and not stress_reason_drift_all_ok)
            or (reconcile_active and not reconcile_all_ok)
            or (artifact_governance_active and not artifact_governance_all_ok)
            or rollback_level == "hard"
            or (rollback_active and not rollback_anchor_ready)
        ):
            status = "red"
        elif (
            health_ratio < 1.0
            or (not bool(state_stability.get("active", False)))
            or (not temporal_active)
            or (not slot_active)
            or (not drift_active)
            or (not stress_active)
            or (not stress_auto_adaptive_active)
            or (not stress_reason_drift_active)
            or (not reconcile_active)
            or (not artifact_governance_active)
            or rollback_level == "soft"
        ):
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
            "state_stability": state_stability,
            "temporal_audit": temporal_audit,
            "slot_anomaly": slot_anomaly,
            "mode_drift": mode_drift,
            "stress_matrix_trend": stress_matrix_trend,
            "stress_autorun_history": stress_autorun_history,
            "stress_autorun_adaptive": stress_autorun_adaptive,
            "stress_autorun_reason_drift": stress_autorun_reason_drift,
            "reconcile_drift": reconcile_drift,
            "artifact_governance": artifact_governance,
            "rollback_recommendation": rollback_rec,
            "history": history,
        }

        report_json = self.output_dir / "review" / f"{d}_ops_report.json"
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
        lines.append("## 状态稳定性")
        lines.append(f"- active: `{state_stability.get('active', False)}`")
        lines.append(f"- samples: `{state_stability.get('samples', 0)}` / min=`{state_stability.get('min_samples', 0)}`")
        metrics = state_stability.get("metrics", {}) if isinstance(state_stability.get("metrics", {}), dict) else {}
        lines.append(
            "- switch_rate: "
            + f"`{self._safe_float(metrics.get('switch_rate', 0.0), 0.0):.2%}` "
            + f"(count={int(metrics.get('switch_count', 0))})"
        )
        lines.append(
            "- risk_multiplier(min/avg/drift): "
            + f"`{self._safe_float(metrics.get('risk_multiplier_min', 1.0), 1.0):.3f}` / "
            + f"`{self._safe_float(metrics.get('risk_multiplier_avg', 1.0), 1.0):.3f}` / "
            + f"`{self._safe_float(metrics.get('risk_multiplier_drift', 0.0), 0.0):+.3f}`"
        )
        lines.append(
            "- source_confidence(min/avg): "
            + f"`{self._safe_float(metrics.get('source_confidence_min', 1.0), 1.0):.2%}` / "
            + f"`{self._safe_float(metrics.get('source_confidence_avg', 1.0), 1.0):.2%}`"
        )
        lines.append(f"- mode_health_fail_days: `{int(metrics.get('mode_health_fail_days', 0))}`")
        lines.append(f"- alerts: `{', '.join(state_stability.get('alerts', [])) if state_stability.get('alerts') else 'NONE'}`")
        for k, v in state_checks.items():
            lines.append(f"- `{k}`: `{v}`")
        lines.append("")
        lines.append("## 时间审计")
        lines.append(f"- active: `{temporal_audit.get('active', False)}`")
        lines.append(
            f"- samples: `{int(temporal_audit.get('samples', 0))}` / min=`{int(temporal_audit.get('min_samples', 0))}`"
        )
        temporal_metrics = (
            temporal_audit.get("metrics", {}) if isinstance(temporal_audit.get("metrics", {}), dict) else {}
        )
        temporal_artifacts = (
            temporal_audit.get("artifacts", {}) if isinstance(temporal_audit.get("artifacts", {}), dict) else {}
        )
        temporal_autofix_artifact = (
            temporal_artifacts.get("autofix_patch", {})
            if isinstance(temporal_artifacts.get("autofix_patch", {}), dict)
            else {}
        )
        lines.append(
            "- missing/leak/strict_disabled: "
            + f"`{self._safe_float(temporal_metrics.get('missing_ratio', 0.0), 0.0):.2%}` / "
            + f"`{self._safe_float(temporal_metrics.get('leak_ratio', 0.0), 0.0):.2%}` / "
            + f"`{self._safe_float(temporal_metrics.get('strict_cutoff_disabled_ratio', 0.0), 0.0):.2%}`"
        )
        lines.append(
            "- autofix(attempted/applied/failed/skipped): "
            + f"`{int(self._safe_float(temporal_metrics.get('autofix_attempted_count', 0), 0))}` / "
            + f"`{int(self._safe_float(temporal_metrics.get('autofix_applied_count', 0), 0))}` / "
            + f"`{int(self._safe_float(temporal_metrics.get('autofix_failed_count', 0), 0))}` / "
            + f"`{int(self._safe_float(temporal_metrics.get('autofix_skipped_count', 0), 0))}`"
        )
        lines.append(
            "- autofix_artifact(written/events/applied/failed/skipped): "
            + f"`{bool(temporal_autofix_artifact.get('written', False))}` / "
            + f"`{int(self._safe_float(temporal_autofix_artifact.get('total_events', 0), 0))}` / "
            + f"`{int(self._safe_float(temporal_autofix_artifact.get('applied_count', 0), 0))}` / "
            + f"`{int(self._safe_float(temporal_autofix_artifact.get('failed_count', 0), 0))}` / "
            + f"`{int(self._safe_float(temporal_autofix_artifact.get('skipped_count', 0), 0))}`"
        )
        lines.append(
            "- autofix_retention(days/rotated/rotation_failed): "
            + f"`{int(self._safe_float(temporal_autofix_artifact.get('retention_days', 0), 0))}` / "
            + f"`{int(self._safe_float(temporal_autofix_artifact.get('rotated_out_count', 0), 0))}` / "
            + f"`{bool(temporal_autofix_artifact.get('rotation_failed', False))}`"
        )
        lines.append(
            "- autofix_checksum_index(enabled/written/entries/failed): "
            + f"`{bool(temporal_autofix_artifact.get('checksum_index_enabled', False))}` / "
            + f"`{bool(temporal_autofix_artifact.get('checksum_index_written', False))}` / "
            + f"`{int(self._safe_float(temporal_autofix_artifact.get('checksum_index_entries', 0), 0))}` / "
            + f"`{bool(temporal_autofix_artifact.get('checksum_index_failed', False))}`"
        )
        temporal_autofix_artifact_md = str(temporal_autofix_artifact.get("md", "")).strip()
        temporal_autofix_artifact_index = str(temporal_autofix_artifact.get("checksum_index_path", "")).strip()
        temporal_autofix_artifact_reason = str(temporal_autofix_artifact.get("reason", "")).strip()
        lines.append(
            f"- autofix_artifact_md: `{temporal_autofix_artifact_md if temporal_autofix_artifact_md else 'N/A'}`"
        )
        lines.append(
            f"- autofix_checksum_index: `{temporal_autofix_artifact_index if temporal_autofix_artifact_index else 'N/A'}`"
        )
        if temporal_autofix_artifact_reason:
            lines.append(f"- autofix_artifact_reason: `{temporal_autofix_artifact_reason}`")
        lines.append(
            f"- alerts: `{', '.join(temporal_audit.get('alerts', [])) if temporal_audit.get('alerts') else 'NONE'}`"
        )
        for k, v in temporal_checks.items():
            lines.append(f"- `{k}`: `{v}`")
        lines.append("")
        lines.append("## 分时异常")
        lines.append(f"- active: `{slot_anomaly.get('active', False)}`")
        lines.append(
            f"- samples: `{int(slot_anomaly.get('samples', 0))}` / min=`{int(slot_anomaly.get('min_samples', 0))}`"
        )
        slot_metrics = slot_anomaly.get("metrics", {}) if isinstance(slot_anomaly.get("metrics", {}), dict) else {}
        lines.append(
            "- slots(expected/observed/missing): "
            + f"`{int(slot_metrics.get('expected_slots', 0))}` / "
            + f"`{int(slot_metrics.get('observed_slots', 0))}` / "
            + f"`{int(slot_metrics.get('missing_slots', 0))}`"
        )
        lines.append(
            "- anomaly_ratio(pre/intra/eod): "
            + f"`{self._safe_float(slot_metrics.get('premarket_anomaly_ratio', 0.0), 0.0):.2%}` / "
            + f"`{self._safe_float(slot_metrics.get('intraday_anomaly_ratio', 0.0), 0.0):.2%}` / "
            + f"`{self._safe_float(slot_metrics.get('eod_anomaly_ratio', 0.0), 0.0):.2%}`"
        )
        lines.append(
            "- eod_anomaly_split(quality/risk): "
            + f"`{self._safe_float(slot_metrics.get('eod_quality_anomaly_ratio', 0.0), 0.0):.2%}` / "
            + f"`{self._safe_float(slot_metrics.get('eod_risk_anomaly_ratio', 0.0), 0.0):.2%}`"
        )
        slots_payload = slot_anomaly.get("slots", {}) if isinstance(slot_anomaly.get("slots", {}), dict) else {}
        eod_payload = slots_payload.get("eod", {}) if isinstance(slots_payload.get("eod", {}), dict) else {}
        regime_buckets = (
            eod_payload.get("regime_buckets", {})
            if isinstance(eod_payload.get("regime_buckets", {}), dict)
            else {}
        )
        if regime_buckets:
            trend = regime_buckets.get("trend", {}) if isinstance(regime_buckets.get("trend", {}), dict) else {}
            range_ = regime_buckets.get("range", {}) if isinstance(regime_buckets.get("range", {}), dict) else {}
            extreme = (
                regime_buckets.get("extreme_vol", {})
                if isinstance(regime_buckets.get("extreme_vol", {}), dict)
                else {}
            )
            lines.append(
                "- eod_regime_buckets(quality trend/range/extreme): "
                + f"`{self._safe_float(trend.get('quality_anomaly_ratio', 0.0), 0.0):.2%}` / "
                + f"`{self._safe_float(range_.get('quality_anomaly_ratio', 0.0), 0.0):.2%}` / "
                + f"`{self._safe_float(extreme.get('quality_anomaly_ratio', 0.0), 0.0):.2%}`"
            )
            lines.append(
                "- eod_regime_buckets(risk trend/range/extreme): "
                + f"`{self._safe_float(trend.get('risk_anomaly_ratio', 0.0), 0.0):.2%}` / "
                + f"`{self._safe_float(range_.get('risk_anomaly_ratio', 0.0), 0.0):.2%}` / "
                + f"`{self._safe_float(extreme.get('risk_anomaly_ratio', 0.0), 0.0):.2%}`"
            )
        lines.append(
            f"- missing_ratio: `{self._safe_float(slot_metrics.get('missing_ratio', 0.0), 0.0):.2%}`"
        )
        lines.append(f"- alerts: `{', '.join(slot_anomaly.get('alerts', [])) if slot_anomaly.get('alerts') else 'NONE'}`")
        for k, v in slot_checks.items():
            lines.append(f"- `{k}`: `{v}`")
        lines.append("")
        lines.append("## 模式漂移")
        lines.append(f"- active: `{mode_drift.get('active', False)}`")
        lines.append(
            f"- scope/compared: `{int((mode_drift.get('summary', {}) or {}).get('scope_modes', 0))}` / "
            + f"`{int((mode_drift.get('summary', {}) or {}).get('compared_modes', 0))}`"
        )
        lines.append(
            f"- runtime_mode: `{str(mode_drift.get('runtime_mode', '') or 'N/A')}` | "
            + f"focus_runtime_only=`{bool(mode_drift.get('focus_runtime_mode_only', True))}`"
        )
        lines.append(
            f"- min_live_trades: `{int(mode_drift.get('min_live_trades', 0))}` | "
            + f"window_days=`{int(mode_drift.get('window_days', 0))}`"
        )
        lines.append(f"- alerts: `{', '.join(mode_drift.get('alerts', [])) if mode_drift.get('alerts') else 'NONE'}`")
        for k, v in drift_checks.items():
            lines.append(f"- `{k}`: `{v}`")
        lines.append("")
        lines.append("## Stress Matrix 趋势")
        lines.append(f"- active: `{stress_matrix_trend.get('active', False)}`")
        lines.append(
            f"- runs: `{int(stress_matrix_trend.get('samples', 0))}` / min=`{int(stress_matrix_trend.get('min_samples', 0))}`"
        )
        stress_metrics = (
            stress_matrix_trend.get("metrics", {})
            if isinstance(stress_matrix_trend.get("metrics", {}), dict)
            else {}
        )
        stress_current = (
            stress_metrics.get("current", {})
            if isinstance(stress_metrics.get("current", {}), dict)
            else {}
        )
        stress_baseline = (
            stress_metrics.get("baseline", {})
            if isinstance(stress_metrics.get("baseline", {}), dict)
            else {}
        )
        stress_delta = (
            stress_metrics.get("delta", {})
            if isinstance(stress_metrics.get("delta", {}), dict)
            else {}
        )
        lines.append(
            "- robustness(current/base/drop): "
            + f"`{self._safe_float(stress_current.get('robustness_score', 0.0), 0.0):.4f}` / "
            + f"`{self._safe_float(stress_baseline.get('robustness_score', 0.0), 0.0):.4f}` / "
            + f"`{self._safe_float(stress_delta.get('robustness_drop', 0.0), 0.0):.4f}`"
        )
        lines.append(
            "- annual_return(current/base/drop): "
            + f"`{self._safe_float(stress_current.get('avg_annual_return', 0.0), 0.0):.2%}` / "
            + f"`{self._safe_float(stress_baseline.get('avg_annual_return', 0.0), 0.0):.2%}` / "
            + f"`{self._safe_float(stress_delta.get('annual_return_drop', 0.0), 0.0):.2%}`"
        )
        lines.append(
            "- worst_drawdown(current/base/rise): "
            + f"`{self._safe_float(stress_current.get('worst_drawdown', 0.0), 0.0):.2%}` / "
            + f"`{self._safe_float(stress_baseline.get('worst_drawdown', 0.0), 0.0):.2%}` / "
            + f"`{self._safe_float(stress_delta.get('drawdown_rise', 0.0), 0.0):.2%}`"
        )
        lines.append(
            "- fail_ratio(current/base/rise): "
            + f"`{self._safe_float(stress_current.get('fail_ratio', 0.0), 0.0):.2%}` / "
            + f"`{self._safe_float(stress_baseline.get('fail_ratio', 0.0), 0.0):.2%}` / "
            + f"`{self._safe_float(stress_delta.get('fail_ratio_rise', 0.0), 0.0):.2%}`"
        )
        lines.append(
            f"- alerts: `{', '.join(stress_matrix_trend.get('alerts', [])) if stress_matrix_trend.get('alerts') else 'NONE'}`"
        )
        for k, v in stress_checks.items():
            lines.append(f"- `{k}`: `{v}`")
        lines.append("")
        lines.append("## Stress Autorun 历史")
        lines.append(f"- active: `{stress_autorun_history.get('active', False)}`")
        lines.append(
            f"- rounds: `{int(stress_autorun_history.get('samples', 0))}` / min=`{int(stress_autorun_history.get('min_samples', 0))}`"
        )
        stress_auto_metrics = (
            stress_autorun_history.get("metrics", {})
            if isinstance(stress_autorun_history.get("metrics", {}), dict)
            else {}
        )
        stress_auto_artifacts = (
            stress_autorun_history.get("artifacts", {})
            if isinstance(stress_autorun_history.get("artifacts", {}), dict)
            else {}
        )
        stress_auto_history_artifact = (
            stress_auto_artifacts.get("history", {})
            if isinstance(stress_auto_artifacts.get("history", {}), dict)
            else {}
        )
        lines.append(
            "- trigger_density/attempt_rate/run_rate: "
            + f"`{self._safe_float(stress_auto_metrics.get('trigger_density', 0.0), 0.0):.2%}` / "
            + f"`{self._safe_float(stress_auto_metrics.get('attempt_rate_when_triggered', 0.0), 0.0):.2%}` / "
            + f"`{self._safe_float(stress_auto_metrics.get('run_rate_when_triggered', 0.0), 0.0):.2%}`"
        )
        lines.append(
            "- skips(cooldown/max_runs/runner_unavailable): "
            + f"`{int(self._safe_float(stress_auto_metrics.get('cooldown_skip_rounds', 0), 0))}` / "
            + f"`{int(self._safe_float(stress_auto_metrics.get('max_runs_skip_rounds', 0), 0))}` / "
            + f"`{int(self._safe_float(stress_auto_metrics.get('runner_unavailable_skip_rounds', 0), 0))}`"
        )
        lines.append(
            "- cooldown_efficiency: "
            + f"`{self._safe_float(stress_auto_metrics.get('cooldown_efficiency', 0.0), 0.0):.2%}`"
        )
        lines.append(
            "- history_artifact(written/rounds): "
            + f"`{bool(stress_auto_history_artifact.get('written', False))}` / "
            + f"`{int(self._safe_float(stress_auto_history_artifact.get('total_rounds', 0), 0))}`"
        )
        lines.append(
            "- history_retention(days/rotated/rotation_failed): "
            + f"`{int(self._safe_float(stress_auto_history_artifact.get('retention_days', 0), 0))}` / "
            + f"`{int(self._safe_float(stress_auto_history_artifact.get('rotated_out_count', 0), 0))}` / "
            + f"`{bool(stress_auto_history_artifact.get('rotation_failed', False))}`"
        )
        lines.append(
            "- history_checksum_index(enabled/written/entries/failed): "
            + f"`{bool(stress_auto_history_artifact.get('checksum_index_enabled', False))}` / "
            + f"`{bool(stress_auto_history_artifact.get('checksum_index_written', False))}` / "
            + f"`{int(self._safe_float(stress_auto_history_artifact.get('checksum_index_entries', 0), 0))}` / "
            + f"`{bool(stress_auto_history_artifact.get('checksum_index_failed', False))}`"
        )
        stress_auto_history_md = str(stress_auto_history_artifact.get("md", "")).strip()
        stress_auto_history_index = str(stress_auto_history_artifact.get("checksum_index_path", "")).strip()
        stress_auto_history_reason = str(stress_auto_history_artifact.get("reason", "")).strip()
        lines.append(f"- history_artifact_md: `{stress_auto_history_md if stress_auto_history_md else 'N/A'}`")
        lines.append(f"- history_checksum_index: `{stress_auto_history_index if stress_auto_history_index else 'N/A'}`")
        if stress_auto_history_reason:
            lines.append(f"- history_artifact_reason: `{stress_auto_history_reason}`")
        lines.append(
            f"- alerts: `{', '.join(stress_autorun_history.get('alerts', [])) if stress_autorun_history.get('alerts') else 'NONE'}`"
        )
        lines.append("")
        lines.append("## Stress Autorun 自适应饱和度")
        lines.append(f"- active: `{stress_autorun_adaptive.get('active', False)}`")
        lines.append(
            f"- rounds: `{int(stress_autorun_adaptive.get('samples', 0))}` / min=`{int(stress_autorun_adaptive.get('min_samples', 0))}`"
        )
        stress_auto_adaptive_metrics = (
            stress_autorun_adaptive.get("metrics", {})
            if isinstance(stress_autorun_adaptive.get("metrics", {}), dict)
            else {}
        )
        lines.append(
            "- effective/base ratio(avg/min/max/latest): "
            + f"`{self._safe_float(stress_auto_adaptive_metrics.get('effective_base_ratio_avg', 0.0), 0.0):.3f}` / "
            + f"`{self._safe_float(stress_auto_adaptive_metrics.get('effective_base_ratio_min', 0.0), 0.0):.3f}` / "
            + f"`{self._safe_float(stress_auto_adaptive_metrics.get('effective_base_ratio_max', 0.0), 0.0):.3f}` / "
            + f"`{self._safe_float(stress_auto_adaptive_metrics.get('effective_base_ratio_latest', 0.0), 0.0):.3f}`"
        )
        lines.append(
            "- throttle/expand ratio: "
            + f"`{self._safe_float(stress_auto_adaptive_metrics.get('throttle_ratio', 0.0), 0.0):.2%}` / "
            + f"`{self._safe_float(stress_auto_adaptive_metrics.get('expand_ratio', 0.0), 0.0):.2%}`"
        )
        lines.append(
            "- rounds(throttle/expand/neutral): "
            + f"`{int(self._safe_float(stress_auto_adaptive_metrics.get('throttle_rounds', 0), 0))}` / "
            + f"`{int(self._safe_float(stress_auto_adaptive_metrics.get('expand_rounds', 0), 0))}` / "
            + f"`{int(self._safe_float(stress_auto_adaptive_metrics.get('neutral_rounds', 0), 0))}`"
        )
        lines.append(
            f"- alerts: `{', '.join(stress_autorun_adaptive.get('alerts', [])) if stress_autorun_adaptive.get('alerts') else 'NONE'}`"
        )
        for k, v in stress_auto_adaptive_checks.items():
            lines.append(f"- `{k}`: `{v}`")
        lines.append("")
        lines.append("## Stress Autorun 原因漂移")
        lines.append(f"- active: `{stress_autorun_reason_drift.get('active', False)}`")
        lines.append(
            f"- rounds: `{int(stress_autorun_reason_drift.get('samples', 0))}` / min=`{int(stress_autorun_reason_drift.get('min_samples', 0))}`"
        )
        stress_reason_metrics = (
            stress_autorun_reason_drift.get("metrics", {})
            if isinstance(stress_autorun_reason_drift.get("metrics", {}), dict)
            else {}
        )
        stress_reason_artifacts = (
            stress_autorun_reason_drift.get("artifacts", {})
            if isinstance(stress_autorun_reason_drift.get("artifacts", {}), dict)
            else {}
        )
        stress_reason_artifact = (
            stress_reason_artifacts.get("reason_drift", {})
            if isinstance(stress_reason_artifacts.get("reason_drift", {}), dict)
            else {}
        )
        lines.append(
            "- reason_mix_gap/change_point_gap: "
            + f"`{self._safe_float(stress_reason_metrics.get('reason_mix_gap', 0.0), 0.0):.3f}` / "
            + f"`{self._safe_float(stress_reason_metrics.get('change_point_gap', 0.0), 0.0):.3f}`"
        )
        lines.append(
            "- baseline_ratio(high/low/other): "
            + f"`{self._safe_float(stress_reason_metrics.get('baseline_high_ratio', 0.0), 0.0):.2%}` / "
            + f"`{self._safe_float(stress_reason_metrics.get('baseline_low_ratio', 0.0), 0.0):.2%}` / "
            + f"`{self._safe_float(stress_reason_metrics.get('baseline_other_ratio', 0.0), 0.0):.2%}`"
        )
        lines.append(
            "- recent_ratio(high/low/other): "
            + f"`{self._safe_float(stress_reason_metrics.get('recent_high_ratio', 0.0), 0.0):.2%}` / "
            + f"`{self._safe_float(stress_reason_metrics.get('recent_low_ratio', 0.0), 0.0):.2%}` / "
            + f"`{self._safe_float(stress_reason_metrics.get('recent_other_ratio', 0.0), 0.0):.2%}`"
        )
        lines.append(
            f"- alerts: `{', '.join(stress_autorun_reason_drift.get('alerts', [])) if stress_autorun_reason_drift.get('alerts') else 'NONE'}`"
        )
        lines.append(
            "- reason_drift_artifact(written/rounds/transitions/windows): "
            + f"`{bool(stress_reason_artifact.get('written', False))}` / "
            + f"`{int(self._safe_float(stress_reason_artifact.get('total_rounds', 0), 0))}` / "
            + f"`{int(self._safe_float(stress_reason_artifact.get('transition_count', 0), 0))}` / "
            + f"`{int(self._safe_float(stress_reason_artifact.get('window_trace_points', 0), 0))}`"
        )
        lines.append(
            "- reason_drift_retention(days/rotated/rotation_failed): "
            + f"`{int(self._safe_float(stress_reason_artifact.get('retention_days', 0), 0))}` / "
            + f"`{int(self._safe_float(stress_reason_artifact.get('rotated_out_count', 0), 0))}` / "
            + f"`{bool(stress_reason_artifact.get('rotation_failed', False))}`"
        )
        lines.append(
            "- reason_drift_checksum_index(enabled/written/entries/failed): "
            + f"`{bool(stress_reason_artifact.get('checksum_index_enabled', False))}` / "
            + f"`{bool(stress_reason_artifact.get('checksum_index_written', False))}` / "
            + f"`{int(self._safe_float(stress_reason_artifact.get('checksum_index_entries', 0), 0))}` / "
            + f"`{bool(stress_reason_artifact.get('checksum_index_failed', False))}`"
        )
        reason_artifact_md = str(stress_reason_artifact.get("md", "")).strip()
        reason_artifact_index = str(stress_reason_artifact.get("checksum_index_path", "")).strip()
        reason_artifact_reason = str(stress_reason_artifact.get("reason", "")).strip()
        lines.append(f"- reason_drift_artifact_md: `{reason_artifact_md if reason_artifact_md else 'N/A'}`")
        lines.append(f"- reason_drift_checksum_index: `{reason_artifact_index if reason_artifact_index else 'N/A'}`")
        if reason_artifact_reason:
            lines.append(f"- reason_drift_artifact_reason: `{reason_artifact_reason}`")
        for k, v in stress_reason_drift_checks.items():
            lines.append(f"- `{k}`: `{v}`")
        lines.append("")
        lines.append("## Artifact Governance")
        lines.append(f"- active: `{artifact_governance.get('active', False)}`")
        artifact_metrics = (
            artifact_governance.get("metrics", {}) if isinstance(artifact_governance.get("metrics", {}), dict) else {}
        )
        lines.append(
            "- profiles(total/active/override): "
            + f"`{int(self._safe_float(artifact_metrics.get('profiles_total', 0), 0))}` / "
            + f"`{int(self._safe_float(artifact_metrics.get('profiles_active', 0), 0))}` / "
            + f"`{int(self._safe_float(artifact_metrics.get('profiles_with_override', 0), 0))}`"
        )
        lines.append(
            "- policy(required_missing/policy_mismatch/legacy_drift/baseline_drift): "
            + f"`{int(self._safe_float(artifact_metrics.get('required_missing_profiles', 0), 0))}` / "
            + f"`{int(self._safe_float(artifact_metrics.get('policy_mismatch_profiles', 0), 0))}` / "
            + f"`{int(self._safe_float(artifact_metrics.get('legacy_policy_drift_profiles', 0), 0))}` / "
            + f"`{int(self._safe_float(artifact_metrics.get('baseline_drift_profiles', 0), 0))}`"
        )
        lines.append(
            "- strict_mode(enabled/blocked): "
            + f"`{bool(artifact_metrics.get('strict_mode_enabled', False))}` / "
            + f"`{bool(artifact_metrics.get('strict_mode_blocked', False))}`"
        )
        lines.append(
            f"- alerts: `{', '.join(artifact_governance.get('alerts', [])) if artifact_governance.get('alerts') else 'NONE'}`"
        )
        for k, v in artifact_governance_checks.items():
            lines.append(f"- `{k}`: `{v}`")
        lines.append("")
        lines.append("## 对账漂移")
        lines.append(f"- active: `{reconcile_drift.get('active', False)}`")
        lines.append(
            f"- samples: `{int(reconcile_drift.get('samples', 0))}` / min=`{int(reconcile_drift.get('min_samples', 0))}`"
        )
        reconcile_metrics = (
            reconcile_drift.get("metrics", {}) if isinstance(reconcile_drift.get("metrics", {}), dict) else {}
        )
        reconcile_artifacts = (
            reconcile_drift.get("artifacts", {}) if isinstance(reconcile_drift.get("artifacts", {}), dict) else {}
        )
        row_diff_artifact = (
            reconcile_artifacts.get("row_diff", {})
            if isinstance(reconcile_artifacts.get("row_diff", {}), dict)
            else {}
        )
        lines.append(
            "- breach_ratio(plan/closed/open): "
            + f"`{self._safe_float(reconcile_metrics.get('plan_gap_breach_ratio', 0.0), 0.0):.2%}` / "
            + f"`{self._safe_float(reconcile_metrics.get('closed_count_gap_breach_ratio', 0.0), 0.0):.2%}` / "
            + f"`{self._safe_float(reconcile_metrics.get('open_gap_breach_ratio', 0.0), 0.0):.2%}`"
        )
        lines.append(
            "- broker(missing/breach_count/breach_pnl): "
            + f"`{self._safe_float(reconcile_metrics.get('broker_missing_ratio', 0.0), 0.0):.2%}` / "
            + f"`{self._safe_float(reconcile_metrics.get('broker_count_breach_ratio', 0.0), 0.0):.2%}` / "
            + f"`{self._safe_float(reconcile_metrics.get('broker_pnl_breach_ratio', 0.0), 0.0):.2%}`"
        )
        lines.append(
            "- broker_contract(schema/numeric/symbol/noncanonical): "
            + f"`{self._safe_float(reconcile_metrics.get('broker_contract_schema_invalid_ratio', 0.0), 0.0):.2%}` / "
            + f"`{self._safe_float(reconcile_metrics.get('broker_contract_numeric_invalid_ratio', 0.0), 0.0):.2%}` / "
            + f"`{self._safe_float(reconcile_metrics.get('broker_contract_symbol_invalid_ratio', 0.0), 0.0):.2%}` / "
            + f"`{self._safe_float(reconcile_metrics.get('broker_contract_symbol_noncanonical_ratio', 0.0), 0.0):.2%}`"
        )
        lines.append(
            "- broker_canonical(eligible/written/fail): "
            + f"`{int(self._safe_float(reconcile_metrics.get('broker_canonical_eligible_days', 0), 0))}` / "
            + f"`{int(self._safe_float(reconcile_metrics.get('broker_canonical_written_days', 0), 0))}` / "
            + f"`{int(self._safe_float(reconcile_metrics.get('broker_canonical_write_fail_days', 0), 0))}`"
        )
        lines.append(
            "- broker_canonical_normalized(symbol/side): "
            + f"`{self._safe_float(reconcile_metrics.get('broker_canonical_symbol_noncanonical_ratio', 0.0), 0.0):.2%}` / "
            + f"`{self._safe_float(reconcile_metrics.get('broker_canonical_side_noncanonical_ratio', 0.0), 0.0):.2%}`"
        )
        lines.append(
            "- broker_row_diff(samples/breach/key/count/notional/canonical_pref): "
            + f"`{int(self._safe_float(reconcile_metrics.get('broker_row_diff_samples', 0), 0))}` / "
            + f"`{self._safe_float(reconcile_metrics.get('broker_row_diff_breach_ratio', 0.0), 0.0):.2%}` / "
            + f"`{self._safe_float(reconcile_metrics.get('broker_row_diff_avg_key_mismatch', 0.0), 0.0):.2%}` / "
            + f"`{self._safe_float(reconcile_metrics.get('broker_row_diff_avg_count_gap', 0.0), 0.0):.2%}` / "
            + f"`{self._safe_float(reconcile_metrics.get('broker_row_diff_avg_notional_gap', 0.0), 0.0):.2%}` / "
            + f"`{self._safe_float(reconcile_metrics.get('broker_row_diff_canonical_preferred_ratio', 0.0), 0.0):.2%}`"
        )
        lines.append(
            "- broker_row_diff_aliases(symbol/side): "
            + f"`{int(self._safe_float(reconcile_metrics.get('broker_row_diff_symbol_alias_size', 0), 0))}` / "
            + f"`{int(self._safe_float(reconcile_metrics.get('broker_row_diff_side_alias_size', 0), 0))}`"
        )
        lines.append(
            "- broker_row_diff_alias_drift(hit/unresolved_key/check): "
            + f"`{self._safe_float(reconcile_metrics.get('broker_row_diff_alias_hit_rate', 0.0), 0.0):.2%}` / "
            + f"`{self._safe_float(reconcile_metrics.get('broker_row_diff_unresolved_key_ratio', 0.0), 0.0):.2%}` / "
            + f"`{bool(reconcile_checks.get('broker_row_diff_alias_drift_ok', True))}`"
        )
        lines.append(
            "- broker_row_diff_artifact(written/sample/breach): "
            + f"`{bool(row_diff_artifact.get('written', False))}` / "
            + f"`{int(self._safe_float(row_diff_artifact.get('sample_rows', 0), 0))}` / "
            + f"`{int(self._safe_float(row_diff_artifact.get('breach_rows', 0), 0))}`"
        )
        lines.append(
            "- broker_row_diff_artifact_retention(days/rotated/rotation_failed): "
            + f"`{int(self._safe_float(row_diff_artifact.get('retention_days', 0), 0))}` / "
            + f"`{int(self._safe_float(row_diff_artifact.get('rotated_out_count', 0), 0))}` / "
            + f"`{bool(row_diff_artifact.get('rotation_failed', False))}`"
        )
        lines.append(
            "- broker_row_diff_artifact_checksum(enabled/written/entries/failed): "
            + f"`{bool(row_diff_artifact.get('checksum_index_enabled', False))}` / "
            + f"`{bool(row_diff_artifact.get('checksum_index_written', False))}` / "
            + f"`{int(self._safe_float(row_diff_artifact.get('checksum_index_entries', 0), 0))}` / "
            + f"`{bool(row_diff_artifact.get('checksum_index_failed', False))}`"
        )
        row_diff_artifact_md = str(row_diff_artifact.get("md", "")).strip()
        row_diff_artifact_index = str(row_diff_artifact.get("checksum_index_path", "")).strip()
        row_diff_artifact_reason = str(row_diff_artifact.get("reason", "")).strip()
        lines.append(
            f"- broker_row_diff_artifact_md: `{row_diff_artifact_md if row_diff_artifact_md else 'N/A'}`"
        )
        lines.append(
            f"- broker_row_diff_artifact_checksum_index: `{row_diff_artifact_index if row_diff_artifact_index else 'N/A'}`"
        )
        if row_diff_artifact_reason:
            lines.append(f"- broker_row_diff_artifact_reason: `{row_diff_artifact_reason}`")
        lines.append(
            "- missing_ratio: "
            + f"`{self._safe_float(reconcile_metrics.get('missing_ratio', 0.0), 0.0):.2%}`"
        )
        lines.append(
            f"- alerts: `{', '.join(reconcile_drift.get('alerts', [])) if reconcile_drift.get('alerts') else 'NONE'}`"
        )
        for k, v in reconcile_checks.items():
            lines.append(f"- `{k}`: `{v}`")
        lines.append("")
        lines.append("## 回滚建议")
        lines.append(f"- active: `{rollback_active}`")
        lines.append(f"- level: `{rollback_level}`")
        lines.append(f"- score: `{int(self._safe_float(rollback_rec.get('score', 0), 0))}`")
        lines.append(f"- action: `{str(rollback_rec.get('action', 'no_rollback'))}`")
        lines.append(f"- anchor_ready: `{rollback_anchor_ready}`")
        lines.append(f"- target_anchor: `{str(rollback_rec.get('target_anchor', '') or 'N/A')}`")
        reason_codes = rollback_rec.get("reason_codes", []) if isinstance(rollback_rec.get("reason_codes", []), list) else []
        lines.append(f"- reason_codes: `{', '.join(reason_codes) if reason_codes else 'NONE'}`")
        lines.append("")
        lines.append("## 最近健康历史")
        for item in history:
            lines.append(f"- {item['date']}: {'OK' if item['healthy'] else 'DEGRADED'} | missing={item['missing']}")
        lines.append("")
        lines.append("## 门槛检查")
        for k, v in gate["checks"].items():
            lines.append(f"- `{k}`: `{v}`")

        report_md = self.output_dir / "review" / f"{d}_ops_report.md"
        write_markdown(report_md, "\n".join(lines) + "\n")
        summary["paths"] = {"json": str(report_json), "md": str(report_md)}
        return summary

    def _build_defect_plan(
        self,
        as_of: date,
        round_no: int,
        review: ReviewDelta,
        tests: dict[str, Any],
        gate: dict[str, Any],
        state_stability: dict[str, Any] | None = None,
        temporal_audit: dict[str, Any] | None = None,
        slot_anomaly: dict[str, Any] | None = None,
        mode_drift: dict[str, Any] | None = None,
        stress_matrix_trend: dict[str, Any] | None = None,
        reconcile_drift: dict[str, Any] | None = None,
        rollback_recommendation: dict[str, Any] | None = None,
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
        if not bool(checks.get("mode_health_ok", True)):
            defects.append(
                {
                    "category": "model",
                    "code": "MODE_HEALTH",
                    "message": "当前运行模式历史表现退化，触发模式健康门禁",
                    "action": "收敛信号阈值并降低交易频率，补跑对应模式窗口回测后再放开。",
                }
            )
        if not bool(checks.get("slot_anomaly_ok", True)):
            defects.append(
                {
                    "category": "execution",
                    "code": "SLOT_ANOMALY",
                    "message": "固定时段执行链路出现异常聚集",
                    "action": "优先修复缺失/异常槽位数据后再恢复正常评审节奏。",
                }
            )
        if not bool(checks.get("mode_drift_ok", True)):
            defects.append(
                {
                    "category": "model",
                    "code": "MODE_DRIFT",
                    "message": "实盘与回测模式表现出现漂移",
                    "action": "先核验 live/backtest 口径一致性，再收紧该模式参数并复跑近窗回测。",
                }
            )
        if not bool(checks.get("temporal_audit_ok", True)):
            defects.append(
                {
                    "category": "data",
                    "code": "TEMPORAL_AUDIT",
                    "message": "时间审计链路存在缺陷（截止日/时间戳缺失或越界）",
                    "action": "先修复 manifest 时间审计字段与 strict_cutoff 标记，再恢复参数更新流程。",
                }
            )
        if not bool(checks.get("reconcile_drift_ok", True)):
            defects.append(
                {
                    "category": "execution",
                    "code": "RECONCILE_DRIFT",
                    "message": "执行链路对账存在漂移",
                    "action": "优先核对 manifest/daily/sqlite 三方一致性并修复缺失写盘。",
                }
            )
        if not bool(checks.get("rollback_anchor_ready", True)):
            defects.append(
                {
                    "category": "risk",
                    "code": "ROLLBACK_ANCHOR_MISSING",
                    "message": "触发回滚条件但缺少可用回滚锚点",
                    "action": "立即补齐 params_live_backup 与 review rollback_anchor 审计链路。",
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
            if self._is_timeout_payload(tests):
                timeout_fallback = (
                    tests.get("timeout_fallback", {})
                    if isinstance(tests.get("timeout_fallback", {}), dict)
                    else {}
                )
                timeout_fallback_ok = bool(int(timeout_fallback.get("returncode", 1)) == 0) if timeout_fallback else False
                fallback_mode = str(timeout_fallback.get("mode", "N/A")) if timeout_fallback else "N/A"
                fallback_summary = str(timeout_fallback.get("summary_line", "")) if timeout_fallback else ""
                defects.append(
                    {
                        "category": "execution",
                        "code": "TEST_TIMEOUT",
                        "message": (
                            "自动测试超时，已触发降级快测兜底；"
                            + f"fallback_ok={timeout_fallback_ok}, mode={fallback_mode}"
                        ),
                        "action": (
                            "优先按超时兜底快测结果定位高耗时模块，"
                            + "拆分后先局部回归，再重跑 lie test-all 全量验收。"
                        ),
                        "failed_tests": failed_tests[:20],
                        "timeout_fallback_summary": fallback_summary,
                    }
                )
            else:
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
        state_payload = state_stability if isinstance(state_stability, dict) else {}
        state_active = bool(state_payload.get("active", False))
        state_checks = state_payload.get("checks", {}) if isinstance(state_payload.get("checks", {}), dict) else {}
        state_metrics = state_payload.get("metrics", {}) if isinstance(state_payload.get("metrics", {}), dict) else {}
        state_thresholds = state_payload.get("thresholds", {}) if isinstance(state_payload.get("thresholds", {}), dict) else {}
        if state_active:
            if not bool(state_checks.get("switch_rate_ok", True)):
                defects.append(
                    {
                        "category": "model",
                        "code": "STATE_MODE_SWITCH",
                        "message": (
                            "模式切换率过高："
                            + f"{self._safe_float(state_metrics.get('switch_rate', 0.0), 0.0):.2%} > "
                            + f"{self._safe_float(state_thresholds.get('mode_switch_max_rate', 1.0), 1.0):.2%}"
                        ),
                        "action": "降低状态切换敏感度（提高确认窗口/滞后阈值），并复跑近窗模式稳定性。",
                    }
                )
            if not bool(state_checks.get("risk_multiplier_floor_ok", True)):
                defects.append(
                    {
                        "category": "risk",
                        "code": "STATE_RISK_MULT_FLOOR",
                        "message": (
                            "执行风险乘子触及过低区间："
                            + f"min={self._safe_float(state_metrics.get('risk_multiplier_min', 1.0), 1.0):.3f} < "
                            + f"floor={self._safe_float(state_thresholds.get('ops_risk_multiplier_floor', 0.0), 0.0):.3f}"
                        ),
                        "action": "优先修复源置信度与模式健康退化根因，避免执行层持续被动降杠杆。",
                    }
                )
            if not bool(state_checks.get("risk_multiplier_drift_ok", True)):
                defects.append(
                    {
                        "category": "risk",
                        "code": "STATE_RISK_MULT_DRIFT",
                        "message": (
                            "执行风险乘子漂移超限："
                            + f"drift={self._safe_float(state_metrics.get('risk_multiplier_drift', 0.0), 0.0):+.3f}, "
                            + f"max={self._safe_float(state_thresholds.get('ops_risk_multiplier_drift_max', 0.0), 0.0):.3f}"
                        ),
                        "action": "检查近期状态切换与数据质量是否同步恶化，必要时进入保护模式并收敛交易频次。",
                    }
                )
            if not bool(state_checks.get("source_confidence_floor_ok", True)):
                defects.append(
                    {
                        "category": "data",
                        "code": "STATE_SOURCE_CONFIDENCE",
                        "message": (
                            "源置信度下穿门槛："
                            + f"min={self._safe_float(state_metrics.get('source_confidence_min', 1.0), 1.0):.2%} < "
                            + f"floor={self._safe_float(state_thresholds.get('ops_source_confidence_floor', 0.0), 0.0):.2%}"
                        ),
                        "action": "优先处理低置信源与跨源冲突仲裁，必要时临时降级数据源组合。",
                    }
                )
            if not bool(state_checks.get("mode_health_fail_days_ok", True)):
                defects.append(
                    {
                        "category": "model",
                        "code": "STATE_MODE_HEALTH_DAYS",
                        "message": (
                            "模式健康失败天数超限："
                            + f"{int(self._safe_float(state_metrics.get('mode_health_fail_days', 0), 0))} > "
                            + f"{int(self._safe_float(state_thresholds.get('ops_mode_health_fail_days_max', 0), 0))}"
                        ),
                        "action": "收敛当前模式参数并补跑对应资产/模式窗口回测，确认恢复后再放开仓位。",
                    }
                )

        temporal_payload = temporal_audit if isinstance(temporal_audit, dict) else {}
        temporal_active = bool(temporal_payload.get("active", False))
        temporal_checks = (
            temporal_payload.get("checks", {}) if isinstance(temporal_payload.get("checks", {}), dict) else {}
        )
        temporal_metrics = (
            temporal_payload.get("metrics", {}) if isinstance(temporal_payload.get("metrics", {}), dict) else {}
        )
        temporal_thresholds = (
            temporal_payload.get("thresholds", {})
            if isinstance(temporal_payload.get("thresholds", {}), dict)
            else {}
        )
        temporal_artifacts = (
            temporal_payload.get("artifacts", {})
            if isinstance(temporal_payload.get("artifacts", {}), dict)
            else {}
        )
        temporal_autofix_artifact = (
            temporal_artifacts.get("autofix_patch", {})
            if isinstance(temporal_artifacts.get("autofix_patch", {}), dict)
            else {}
        )
        if temporal_active:
            if not bool(temporal_checks.get("missing_ratio_ok", True)):
                defects.append(
                    {
                        "category": "data",
                        "code": "TEMPORAL_AUDIT_MISSING",
                        "message": (
                            "时间审计字段缺失率超限："
                            + f"{self._safe_float(temporal_metrics.get('missing_ratio', 0.0), 0.0):.2%} > "
                            + f"{self._safe_float(temporal_thresholds.get('ops_temporal_audit_missing_ratio_max', 1.0), 1.0):.2%}"
                        ),
                        "action": "补齐 cutoff_ts/bar_max_ts/news_max_ts/report_max_ts 后重跑 strategy-lab/research。",
                    }
                )
            if not bool(temporal_checks.get("leak_ratio_ok", True)):
                defects.append(
                    {
                        "category": "data",
                        "code": "TEMPORAL_AUDIT_LEAK",
                        "message": (
                            "时间泄漏率超限："
                            + f"{self._safe_float(temporal_metrics.get('leak_ratio', 0.0), 0.0):.2%} > "
                            + f"{self._safe_float(temporal_thresholds.get('ops_temporal_audit_leak_ratio_max', 1.0), 1.0):.2%}"
                        ),
                        "action": "立即冻结参数自动更新，修复时间切分后重跑全链路回测与复盘。",
                    }
                )
            if not bool(temporal_checks.get("strict_cutoff_ok", True)):
                defects.append(
                    {
                        "category": "data",
                        "code": "TEMPORAL_AUDIT_STRICT",
                        "message": "发现 strict_cutoff_enforced=false 的实验产物",
                        "action": "仅允许 strict_cutoff_enforced=true 产物进入 review 融合路径。",
                    }
                )
            if int(self._safe_float(temporal_metrics.get("autofix_failed_count", 0), 0)) > 0:
                autofix_artifact_md = str(temporal_autofix_artifact.get("md", "")).strip()
                artifact_hint = f"详见 {autofix_artifact_md}。" if autofix_artifact_md else ""
                defects.append(
                    {
                        "category": "data",
                        "code": "TEMPORAL_AUDIT_AUTOFIX_FAILED",
                        "message": "时间审计自动修复写盘失败",
                        "action": "核查 manifests 写盘权限与 summary 路径映射，修复后重跑 temporal audit。" + artifact_hint,
                    }
                )
            if bool(temporal_metrics.get("autofix_artifact_rotation_failed", False)):
                defects.append(
                    {
                        "category": "execution",
                        "code": "TEMPORAL_AUDIT_AUTOFIX_ROTATION",
                        "message": "temporal autofix 工件轮转失败，旧工件可能堆积且影响审计一致性",
                        "action": "核查 review 目录权限与日期命名规则，修复后重跑 gate/ops 以重建轮转状态。",
                    }
                )
            if bool(temporal_metrics.get("autofix_artifact_checksum_index_failed", False)):
                index_path = str(temporal_autofix_artifact.get("checksum_index_path", "")).strip()
                index_hint = f"目标索引: {index_path}。" if index_path else ""
                defects.append(
                    {
                        "category": "execution",
                        "code": "TEMPORAL_AUDIT_AUTOFIX_CHECKSUM_INDEX",
                        "message": "temporal autofix checksum 索引写入失败，审计追溯完整性下降",
                        "action": "修复索引写盘链路后重跑 temporal audit 与 ops 报告。" + index_hint,
                    }
                )

        slot_payload = slot_anomaly if isinstance(slot_anomaly, dict) else {}
        slot_active = bool(slot_payload.get("active", False))
        slot_checks = slot_payload.get("checks", {}) if isinstance(slot_payload.get("checks", {}), dict) else {}
        slot_metrics = slot_payload.get("metrics", {}) if isinstance(slot_payload.get("metrics", {}), dict) else {}
        slot_thresholds = slot_payload.get("thresholds", {}) if isinstance(slot_payload.get("thresholds", {}), dict) else {}
        if slot_active:
            if not bool(slot_checks.get("missing_ratio_ok", True)):
                defects.append(
                    {
                        "category": "execution",
                        "code": "SLOT_MISSING_RATIO",
                        "message": (
                            "分时槽位缺失率超限："
                            + f"{self._safe_float(slot_metrics.get('missing_ratio', 0.0), 0.0):.2%} > "
                            + f"{self._safe_float(slot_thresholds.get('ops_slot_missing_ratio_max', 1.0), 1.0):.2%}"
                        ),
                        "action": "补齐缺失槽位日志/manifest，并核对调度器与数据写盘链路。",
                    }
                )
            if not bool(slot_checks.get("premarket_anomaly_ok", True)):
                defects.append(
                    {
                        "category": "execution",
                        "code": "SLOT_PREMARKET_ANOMALY",
                        "message": (
                            "盘前异常率超限："
                            + f"{self._safe_float(slot_metrics.get('premarket_anomaly_ratio', 0.0), 0.0):.2%} > "
                            + f"{self._safe_float(slot_thresholds.get('ops_slot_premarket_anomaly_ratio_max', 1.0), 1.0):.2%}"
                        ),
                        "action": "优先修复盘前数据质量/保护模式触发路径，避免开盘前风控误触发。",
                    }
                )
            if not bool(slot_checks.get("intraday_anomaly_ok", True)):
                defects.append(
                    {
                        "category": "execution",
                        "code": "SLOT_INTRADAY_ANOMALY",
                        "message": (
                            "盘中异常率超限："
                            + f"{self._safe_float(slot_metrics.get('intraday_anomaly_ratio', 0.0), 0.0):.2%} > "
                            + f"{self._safe_float(slot_thresholds.get('ops_slot_intraday_anomaly_ratio_max', 1.0), 1.0):.2%}"
                        ),
                        "action": "复核盘中校准任务的输入完整性与保护模式阈值，必要时降级盘中校准频率。",
                    }
                )
            if not bool(slot_checks.get("eod_quality_anomaly_ok", True)):
                defects.append(
                    {
                        "category": "execution",
                        "code": "SLOT_EOD_QUALITY_ANOMALY",
                        "message": (
                            "收盘主计算质量异常率超限："
                            + f"{self._safe_float(slot_metrics.get('eod_quality_anomaly_ratio', 0.0), 0.0):.2%} > "
                            + f"{self._safe_float(slot_thresholds.get('ops_slot_eod_quality_anomaly_ratio_max', 1.0), 1.0):.2%}"
                        ),
                        "action": "优先修复 EOD 质量门禁失败根因（数据完整性/冲突仲裁）后再放开门槛。",
                    }
                )
            if not bool(slot_checks.get("eod_risk_anomaly_ok", True)):
                defects.append(
                    {
                        "category": "risk",
                        "code": "SLOT_EOD_RISK_ANOMALY",
                        "message": (
                            "收盘主计算风险异常率超限："
                            + f"{self._safe_float(slot_metrics.get('eod_risk_anomaly_ratio', 0.0), 0.0):.2%} > "
                            + f"{self._safe_float(slot_thresholds.get('ops_slot_eod_risk_anomaly_ratio_max', 1.0), 1.0):.2%}"
                        ),
                        "action": "检查风险乘子与保护模式阈值是否过严，避免过度触发风险降级。",
                    }
                )
            if not bool(slot_checks.get("eod_quality_regime_bucket_ok", True)):
                defects.append(
                    {
                        "category": "model",
                        "code": "SLOT_EOD_QUALITY_REGIME_BUCKET",
                        "message": (
                            "收盘质量异常在体制分桶中超限：breaches="
                            + f"{int(self._safe_float(slot_metrics.get('eod_quality_regime_bucket_breaches', 0), 0))}"
                        ),
                        "action": "按 trend/range/extreme_vol 分桶复核 EOD 质量阈值，避免单一阈值掩盖体制差异。",
                    }
                )
            if not bool(slot_checks.get("eod_risk_regime_bucket_ok", True)):
                defects.append(
                    {
                        "category": "risk",
                        "code": "SLOT_EOD_RISK_REGIME_BUCKET",
                        "message": (
                            "收盘风险异常在体制分桶中超限：breaches="
                            + f"{int(self._safe_float(slot_metrics.get('eod_risk_regime_bucket_breaches', 0), 0))}"
                        ),
                        "action": "按体制分桶校正风险乘子下限与保护模式阈值，减少误伤与漏判。",
                    }
                )
            if not bool(slot_checks.get("eod_anomaly_ok", True)):
                defects.append(
                    {
                        "category": "execution",
                        "code": "SLOT_EOD_ANOMALY",
                        "message": (
                            "收盘主计算异常率超限："
                            + f"{self._safe_float(slot_metrics.get('eod_anomaly_ratio', 0.0), 0.0):.2%} > "
                            + f"{self._safe_float(slot_thresholds.get('ops_slot_eod_anomaly_ratio_max', 1.0), 1.0):.2%}"
                        ),
                        "action": "检查 EOD 主流程质量门禁与风险乘子链路，确保收盘工件稳定产出。",
                    }
                )

        drift_payload = mode_drift if isinstance(mode_drift, dict) else {}
        drift_active = bool(drift_payload.get("active", False))
        drift_checks = drift_payload.get("checks", {}) if isinstance(drift_payload.get("checks", {}), dict) else {}
        drift_modes = drift_payload.get("modes", {}) if isinstance(drift_payload.get("modes", {}), dict) else {}
        if drift_active:
            if not bool(drift_checks.get("win_rate_gap_ok", True)):
                offenders: list[str] = []
                for mode, row in drift_modes.items():
                    if not isinstance(row, dict):
                        continue
                    row_checks = row.get("checks", {}) if isinstance(row.get("checks", {}), dict) else {}
                    if bool(row.get("active", False)) and not bool(row_checks.get("win_rate_gap_ok", True)):
                        offenders.append(str(mode))
                defects.append(
                    {
                        "category": "model",
                        "code": "MODE_DRIFT_WIN_RATE",
                        "message": "模式胜率偏离回测基线",
                        "action": "优先收紧对应模式信号阈值，并校验成交成本/滑点建模是否低估。",
                        "modes": offenders[:10],
                    }
                )
            if not bool(drift_checks.get("profit_factor_gap_ok", True)):
                offenders = []
                for mode, row in drift_modes.items():
                    if not isinstance(row, dict):
                        continue
                    row_checks = row.get("checks", {}) if isinstance(row.get("checks", {}), dict) else {}
                    if bool(row.get("active", False)) and not bool(row_checks.get("profit_factor_gap_ok", True)):
                        offenders.append(str(mode))
                defects.append(
                    {
                        "category": "model",
                        "code": "MODE_DRIFT_PROFIT_FACTOR",
                        "message": "模式盈亏比偏离回测基线",
                        "action": "复核止损/止盈执行偏差与行情跳空影响，必要时降低模式风险乘子。",
                        "modes": offenders[:10],
                    }
                )

        stress_payload = stress_matrix_trend if isinstance(stress_matrix_trend, dict) else {}
        stress_active = bool(stress_payload.get("active", False))
        stress_checks = (
            stress_payload.get("checks", {}) if isinstance(stress_payload.get("checks", {}), dict) else {}
        )
        if stress_active:
            if not bool(stress_checks.get("robustness_drop_ok", True)):
                defects.append(
                    {
                        "category": "model",
                        "code": "STRESS_MATRIX_ROBUSTNESS",
                        "message": "Stress Matrix 鲁棒性评分显著下滑",
                        "action": "优先回溯模式参数变更与数据口径变化，回滚到最近稳定参数后重跑 stress matrix。",
                    }
                )
            if not bool(stress_checks.get("annual_return_drop_ok", True)):
                defects.append(
                    {
                        "category": "model",
                        "code": "STRESS_MATRIX_ANNUAL_RETURN",
                        "message": "Stress Matrix 年化收益相对基线下滑超限",
                        "action": "收紧高频模式风险预算并复核事件窗口配置。",
                    }
                )
            if not bool(stress_checks.get("drawdown_rise_ok", True)):
                defects.append(
                    {
                        "category": "risk",
                        "code": "STRESS_MATRIX_DRAWDOWN",
                        "message": "Stress Matrix 最大回撤上升超限",
                        "action": "提高防御仓占比并校正止损/保护模式阈值后重测。",
                    }
                )
            if not bool(stress_checks.get("fail_ratio_ok", True)):
                defects.append(
                    {
                        "category": "execution",
                        "code": "STRESS_MATRIX_FAIL_RATIO",
                        "message": "Stress Matrix 窗口失败率过高",
                        "action": "排查窗口数据缺失与模式路由逻辑，修复后全量复算 stress matrix。",
                    }
                )

        stress_history_payload = (
            gate.get("stress_autorun_history", {})
            if isinstance(gate.get("stress_autorun_history", {}), dict)
            else {}
        )
        stress_history_active = bool(stress_history_payload.get("active", False))
        stress_history_checks = (
            stress_history_payload.get("checks", {})
            if isinstance(stress_history_payload.get("checks", {}), dict)
            else {}
        )
        stress_history_artifacts = (
            stress_history_payload.get("artifacts", {})
            if isinstance(stress_history_payload.get("artifacts", {}), dict)
            else {}
        )
        stress_history_artifact = (
            stress_history_artifacts.get("history", {})
            if isinstance(stress_history_artifacts.get("history", {}), dict)
            else {}
        )
        if stress_history_active:
            if not bool(stress_history_checks.get("artifact_rotation_ok", True)):
                defects.append(
                    {
                        "category": "execution",
                        "code": "STRESS_AUTORUN_HISTORY_ARTIFACT_ROTATION",
                        "message": "Stress history 工件轮转失败，历史审计窗口可能失控增长。",
                        "action": "修复 review 目录权限与日期命名后重跑 gate/ops，恢复 retention 轮转。",
                    }
                )
            if not bool(stress_history_checks.get("artifact_checksum_index_ok", True)):
                history_index_path = str(stress_history_artifact.get("checksum_index_path", "")).strip()
                history_index_hint = f"目标索引: {history_index_path}。" if history_index_path else ""
                defects.append(
                    {
                        "category": "execution",
                        "code": "STRESS_AUTORUN_HISTORY_CHECKSUM_INDEX",
                        "message": "Stress history checksum 索引写入失败，工件追溯完整性下降。",
                        "action": "修复 checksum 索引写盘链路后重跑 stress history 审计。" + history_index_hint,
                    }
                )

        stress_auto_adaptive_payload = (
            gate.get("stress_autorun_adaptive", {})
            if isinstance(gate.get("stress_autorun_adaptive", {}), dict)
            else {}
        )
        stress_auto_adaptive_active = bool(stress_auto_adaptive_payload.get("active", False))
        stress_auto_adaptive_checks = (
            stress_auto_adaptive_payload.get("checks", {})
            if isinstance(stress_auto_adaptive_payload.get("checks", {}), dict)
            else {}
        )
        stress_auto_adaptive_metrics = (
            stress_auto_adaptive_payload.get("metrics", {})
            if isinstance(stress_auto_adaptive_payload.get("metrics", {}), dict)
            else {}
        )
        stress_auto_adaptive_thresholds = (
            stress_auto_adaptive_payload.get("thresholds", {})
            if isinstance(stress_auto_adaptive_payload.get("thresholds", {}), dict)
            else {}
        )
        if stress_auto_adaptive_active:
            if not bool(stress_auto_adaptive_checks.get("effective_base_ratio_floor_ok", True)):
                defects.append(
                    {
                        "category": "risk",
                        "code": "STRESS_AUTORUN_ADAPTIVE_RATIO_LOW",
                        "message": (
                            "Stress autorun 动态上限均值偏低："
                            + f"{self._safe_float(stress_auto_adaptive_metrics.get('effective_base_ratio_avg', 0.0), 0.0):.3f} < "
                            + f"{self._safe_float(stress_auto_adaptive_thresholds.get('ops_stress_autorun_adaptive_effective_base_ratio_floor', 0.0), 0.0):.3f}"
                        ),
                        "action": "放宽 adaptive 高密度降档系数或提升 base max-runs，避免压力期诊断算力长期不足。",
                    }
                )
            if not bool(stress_auto_adaptive_checks.get("effective_base_ratio_ceiling_ok", True)):
                defects.append(
                    {
                        "category": "risk",
                        "code": "STRESS_AUTORUN_ADAPTIVE_RATIO_HIGH",
                        "message": (
                            "Stress autorun 动态上限均值偏高："
                            + f"{self._safe_float(stress_auto_adaptive_metrics.get('effective_base_ratio_avg', 0.0), 0.0):.3f} > "
                            + f"{self._safe_float(stress_auto_adaptive_thresholds.get('ops_stress_autorun_adaptive_effective_base_ratio_ceiling', 0.0), 0.0):.3f}"
                        ),
                        "action": "收紧 adaptive 低密度扩张系数或降低 cap，防止评审回路在低质量阶段过度重跑。",
                    }
                )
            if not bool(stress_auto_adaptive_checks.get("throttle_ratio_ok", True)):
                defects.append(
                    {
                        "category": "execution",
                        "code": "STRESS_AUTORUN_ADAPTIVE_THROTTLE",
                        "message": (
                            "Stress autorun 高频降档占比过高："
                            + f"{self._safe_float(stress_auto_adaptive_metrics.get('throttle_ratio', 0.0), 0.0):.2%} > "
                            + f"{self._safe_float(stress_auto_adaptive_thresholds.get('ops_stress_autorun_adaptive_throttle_ratio_max', 1.0), 1.0):.2%}"
                        ),
                        "action": "优先复核触发密度统计窗口与 stress 趋势阈值，避免噪声触发导致持续降档。",
                    }
                )
            if not bool(stress_auto_adaptive_checks.get("expand_ratio_ok", True)):
                defects.append(
                    {
                        "category": "execution",
                        "code": "STRESS_AUTORUN_ADAPTIVE_EXPAND",
                        "message": (
                            "Stress autorun 高频扩张占比过高："
                            + f"{self._safe_float(stress_auto_adaptive_metrics.get('expand_ratio', 0.0), 0.0):.2%} > "
                            + f"{self._safe_float(stress_auto_adaptive_thresholds.get('ops_stress_autorun_adaptive_expand_ratio_max', 1.0), 1.0):.2%}"
                        ),
                        "action": "收敛低密度扩张阈值并加严 cap，防止低触发期被误判为可无限重跑。",
                    }
                )

        stress_reason_payload = (
            gate.get("stress_autorun_reason_drift", {})
            if isinstance(gate.get("stress_autorun_reason_drift", {}), dict)
            else {}
        )
        stress_reason_active = bool(stress_reason_payload.get("active", False))
        stress_reason_checks = (
            stress_reason_payload.get("checks", {})
            if isinstance(stress_reason_payload.get("checks", {}), dict)
            else {}
        )
        stress_reason_metrics = (
            stress_reason_payload.get("metrics", {})
            if isinstance(stress_reason_payload.get("metrics", {}), dict)
            else {}
        )
        stress_reason_thresholds = (
            stress_reason_payload.get("thresholds", {})
            if isinstance(stress_reason_payload.get("thresholds", {}), dict)
            else {}
        )
        stress_reason_artifacts = (
            stress_reason_payload.get("artifacts", {})
            if isinstance(stress_reason_payload.get("artifacts", {}), dict)
            else {}
        )
        stress_reason_artifact = (
            stress_reason_artifacts.get("reason_drift", {})
            if isinstance(stress_reason_artifacts.get("reason_drift", {}), dict)
            else {}
        )
        if stress_reason_active:
            if not bool(stress_reason_checks.get("reason_mix_gap_ok", True)):
                defects.append(
                    {
                        "category": "model",
                        "code": "STRESS_AUTORUN_REASON_MIX",
                        "message": (
                            "Stress autorun 自适应原因分布漂移超限："
                            + f"{self._safe_float(stress_reason_metrics.get('reason_mix_gap', 0.0), 0.0):.3f} > "
                            + f"{self._safe_float(stress_reason_thresholds.get('ops_stress_autorun_reason_drift_mix_gap_max', 1.0), 1.0):.3f}"
                        ),
                        "action": "复核 high_density/low_density 触发阈值与统计窗口，避免 reason mix 在短窗内失真。",
                    }
                )
            if not bool(stress_reason_checks.get("change_point_gap_ok", True)):
                defects.append(
                    {
                        "category": "execution",
                        "code": "STRESS_AUTORUN_REASON_CHANGE_POINT",
                        "message": (
                            "Stress autorun 自适应原因出现突变："
                            + f"{self._safe_float(stress_reason_metrics.get('change_point_gap', 0.0), 0.0):.3f} > "
                            + f"{self._safe_float(stress_reason_thresholds.get('ops_stress_autorun_reason_drift_change_point_gap_max', 1.0), 1.0):.3f}"
                        ),
                        "action": "触发变点复核：先锁定最近参数变更与数据口径，再决定是否回滚 adaptive 配置。",
                    }
                )
            if bool(stress_reason_metrics.get("artifact_rotation_failed", False)):
                defects.append(
                    {
                        "category": "execution",
                        "code": "STRESS_AUTORUN_REASON_ARTIFACT_ROTATION",
                        "message": "Stress reason-drift 工件轮转失败，历史审计窗口可能持续膨胀。",
                        "action": "修复 review 目录权限与日期命名后重跑 gate/ops，恢复 retention 轮转。",
                    }
                )
            if bool(stress_reason_metrics.get("artifact_checksum_index_failed", False)):
                reason_index_path = str(stress_reason_artifact.get("checksum_index_path", "")).strip()
                reason_index_hint = f"目标索引: {reason_index_path}。" if reason_index_path else ""
                defects.append(
                    {
                        "category": "execution",
                        "code": "STRESS_AUTORUN_REASON_CHECKSUM_INDEX",
                        "message": "Stress reason-drift checksum 索引写入失败，工件追溯完整性下降。",
                        "action": "修复 checksum 索引写盘链路后重跑 reason-drift 审计。" + reason_index_hint,
                    }
                )

        reconcile_payload = reconcile_drift if isinstance(reconcile_drift, dict) else {}
        reconcile_active = bool(reconcile_payload.get("active", False))
        reconcile_checks = (
            reconcile_payload.get("checks", {}) if isinstance(reconcile_payload.get("checks", {}), dict) else {}
        )
        reconcile_metrics = (
            reconcile_payload.get("metrics", {}) if isinstance(reconcile_payload.get("metrics", {}), dict) else {}
        )
        reconcile_thresholds = (
            reconcile_payload.get("thresholds", {})
            if isinstance(reconcile_payload.get("thresholds", {}), dict)
            else {}
        )
        reconcile_artifacts = (
            reconcile_payload.get("artifacts", {})
            if isinstance(reconcile_payload.get("artifacts", {}), dict)
            else {}
        )
        reconcile_row_diff_artifact = (
            reconcile_artifacts.get("row_diff", {})
            if isinstance(reconcile_artifacts.get("row_diff", {}), dict)
            else {}
        )
        if reconcile_active:
            if not bool(reconcile_checks.get("missing_ratio_ok", True)):
                defects.append(
                    {
                        "category": "execution",
                        "code": "RECONCILE_MISSING_RATIO",
                        "message": (
                            "对账缺失率超限："
                            + f"{self._safe_float(reconcile_metrics.get('missing_ratio', 0.0), 0.0):.2%} > "
                            + f"{self._safe_float(reconcile_thresholds.get('ops_reconcile_missing_ratio_max', 1.0), 1.0):.2%}"
                        ),
                        "action": "优先补齐 daily/manifest/sqlite 缺失数据并重跑当日 EOD。",
                    }
                )
            if not bool(reconcile_checks.get("plan_count_gap_ok", True)):
                defects.append(
                    {
                        "category": "execution",
                        "code": "RECONCILE_PLAN_COUNT",
                        "message": "计划单数量对账漂移超限",
                        "action": "核查 latest_positions 与 daily positions 写入口径，修复后重跑对账。",
                    }
                )
            if not bool(reconcile_checks.get("closed_count_gap_ok", True)):
                defects.append(
                    {
                        "category": "execution",
                        "code": "RECONCILE_CLOSED_COUNT",
                        "message": "平仓笔数对账漂移超限",
                        "action": "核查 executed_plans 回写条件与 manifest 统计口径。",
                    }
                )
            if not bool(reconcile_checks.get("closed_pnl_gap_ok", True)):
                defects.append(
                    {
                        "category": "execution",
                        "code": "RECONCILE_CLOSED_PNL",
                        "message": "平仓盈亏对账漂移超限",
                        "action": "核查平仓收益写盘精度与汇总逻辑，修复后回放当日数据。",
                    }
                )
            if not bool(reconcile_checks.get("open_count_gap_ok", True)):
                defects.append(
                    {
                        "category": "execution",
                        "code": "RECONCILE_OPEN_COUNT",
                        "message": "当前持仓数量与 open_state 对账漂移超限",
                        "action": "重建 paper_positions_open 状态并校验 EOD open_positions 指标。",
                    }
                )
            if not bool(reconcile_checks.get("broker_missing_ratio_ok", True)):
                defects.append(
                    {
                        "category": "execution",
                        "code": "RECONCILE_BROKER_MISSING",
                        "message": (
                            "Broker 快照缺失率超限："
                            + f"{self._safe_float(reconcile_metrics.get('broker_missing_ratio', 0.0), 0.0):.2%} > "
                            + f"{self._safe_float(reconcile_thresholds.get('ops_reconcile_broker_missing_ratio_max', 1.0), 1.0):.2%}"
                        ),
                        "action": "补齐 artifacts/broker_snapshot/YYYY-MM-DD.json 后再执行对账判定。",
                    }
                )
            if not bool(reconcile_checks.get("broker_count_gap_ok", True)):
                defects.append(
                    {
                        "category": "execution",
                        "code": "RECONCILE_BROKER_COUNT",
                        "message": "Broker 持仓数量与系统 open_positions 对账漂移超限",
                        "action": "核查 broker 仓位口径（净仓/总仓）并统一映射规则。",
                    }
                )
            if not bool(reconcile_checks.get("broker_pnl_gap_ok", True)):
                defects.append(
                    {
                        "category": "execution",
                        "code": "RECONCILE_BROKER_PNL",
                        "message": "Broker 收益与系统 closed_pnl 对账漂移超限",
                        "action": "核查手续费/滑点/币种换算口径后重放当日对账。",
                    }
                )
            if not bool(reconcile_checks.get("broker_contract_schema_ok", True)):
                defects.append(
                    {
                        "category": "execution",
                        "code": "RECONCILE_BROKER_CONTRACT_SCHEMA",
                        "message": "Broker 快照契约 schema 校验失败率超限",
                        "action": "修复 broker_snapshot 结构字段（source/open_positions/closed_count/closed_pnl/positions）。",
                    }
                )
            if not bool(reconcile_checks.get("broker_contract_numeric_ok", True)):
                defects.append(
                    {
                        "category": "execution",
                        "code": "RECONCILE_BROKER_CONTRACT_NUMERIC",
                        "message": "Broker 快照契约数值范围校验失败率超限",
                        "action": "修复 broker 数值字段口径与边界（qty/notional/price/closed_pnl）。",
                    }
                )
            if not bool(reconcile_checks.get("broker_contract_symbol_ok", True)):
                defects.append(
                    {
                        "category": "execution",
                        "code": "RECONCILE_BROKER_CONTRACT_SYMBOL",
                        "message": "Broker 快照 symbol 规范化校验失败率超限",
                        "action": "统一 symbol 规范（大写、去空白、仅保留 A-Z0-9._-）并修复 side 枚举映射。",
                    }
                )
            if not bool(reconcile_checks.get("broker_contract_canonical_view_ok", True)):
                defects.append(
                    {
                        "category": "execution",
                        "code": "RECONCILE_BROKER_CANONICAL_VIEW",
                        "message": "Broker canonical 视图写入失败",
                        "action": "修复 artifacts/broker_snapshot_canonical 写盘权限或路径配置并重跑对账。",
                    }
                )
            if not bool(reconcile_checks.get("broker_row_diff_ok", True)):
                row_diff_md = str(reconcile_row_diff_artifact.get("md", "")).strip()
                row_diff_reason = str(reconcile_row_diff_artifact.get("reason", "")).strip()
                row_diff_hint = ""
                if row_diff_md:
                    row_diff_hint = f"查看行级差异工件：{row_diff_md}。"
                elif row_diff_reason:
                    row_diff_hint = f"行级差异工件未写入：{row_diff_reason}。"
                defects.append(
                    {
                        "category": "execution",
                        "code": "RECONCILE_BROKER_ROW_DIFF",
                        "message": "Broker 与系统持仓行级对账偏差超限",
                        "action": (
                            "优先核查 symbol/side 映射与 canonical 口径，再按键值明细修复仓位对账链路。"
                            + row_diff_hint
                        ),
                    }
                )
            if not bool(reconcile_checks.get("broker_row_diff_alias_drift_ok", True)):
                defects.append(
                    {
                        "category": "execution",
                        "code": "RECONCILE_BROKER_ROW_DIFF_ALIAS_DRIFT",
                        "message": "Broker 行级对账 alias 漂移超限（命中率/未解析键比率异常）",
                        "action": "校准 symbol/side alias 映射并复核 unresolved keys，再重跑对账与行级工件导出。",
                    }
                )
            if not bool(reconcile_checks.get("broker_row_diff_artifact_rotation_ok", True)):
                defects.append(
                    {
                        "category": "execution",
                        "code": "RECONCILE_BROKER_ROW_DIFF_ARTIFACT_ROTATION",
                        "message": "Broker 行级差异工件轮转失败，review 目录保留策略失效。",
                        "action": "修复 output/review 写权限与文件命名后重跑 gate/ops，恢复 retention 清理。",
                    }
                )
            if not bool(reconcile_checks.get("broker_row_diff_artifact_checksum_index_ok", True)):
                row_diff_index = str(reconcile_row_diff_artifact.get("checksum_index_path", "")).strip()
                row_diff_index_hint = f"目标索引: {row_diff_index}。" if row_diff_index else ""
                defects.append(
                    {
                        "category": "execution",
                        "code": "RECONCILE_BROKER_ROW_DIFF_ARTIFACT_CHECKSUM_INDEX",
                        "message": "Broker 行级差异工件 checksum 索引写入失败，审计追溯链路不完整。",
                        "action": "修复 checksum 索引写盘后重跑对账工件导出。" + row_diff_index_hint,
                    }
                )

        artifact_governance_payload = (
            gate.get("artifact_governance", {})
            if isinstance(gate.get("artifact_governance", {}), dict)
            else {}
        )
        artifact_governance_active = bool(artifact_governance_payload.get("active", False))
        artifact_governance_checks = (
            artifact_governance_payload.get("checks", {})
            if isinstance(artifact_governance_payload.get("checks", {}), dict)
            else {}
        )
        artifact_governance_metrics = (
            artifact_governance_payload.get("metrics", {})
            if isinstance(artifact_governance_payload.get("metrics", {}), dict)
            else {}
        )
        if artifact_governance_active:
            if not bool(artifact_governance_checks.get("required_profiles_present_ok", True)):
                defects.append(
                    {
                        "category": "execution",
                        "code": "ARTIFACT_GOVERNANCE_PROFILE_MISSING",
                        "message": "Artifact governance 缺少活跃 profile 的工件观测，审计链路不完整。",
                        "action": "补齐对应 profile 工件导出并校验 artifacts 路径配置。",
                    }
                )
            if not bool(artifact_governance_checks.get("policy_alignment_ok", True)):
                defects.append(
                    {
                        "category": "execution",
                        "code": "ARTIFACT_GOVERNANCE_POLICY_MISMATCH",
                        "message": "Artifact governance 策略与实际工件存在错配（retention/checksum/index 命名漂移）。",
                        "action": "统一 profile 策略声明与工件写入行为后重跑 gate/ops。",
                    }
                )
            if int(artifact_governance_metrics.get("legacy_policy_drift_profiles", 0)) > 0:
                defects.append(
                    {
                        "category": "report",
                        "code": "ARTIFACT_GOVERNANCE_LEGACY_DRIFT",
                        "message": "Artifact governance profile 与 legacy 键存在漂移（已进入声明式策略模式）。",
                        "action": "确认是否保留 legacy 键；如无需兼容，逐步收敛到 profiles 单一配置源。",
                    }
                )
            if int(artifact_governance_metrics.get("baseline_drift_profiles", 0)) > 0:
                defects.append(
                    {
                        "category": "report",
                        "code": "ARTIFACT_GOVERNANCE_BASELINE_DRIFT",
                        "message": "Artifact governance profile 与基线配置不一致（baseline freeze 漂移）。",
                        "action": "核对 ops_artifact_governance_profile_baseline 与 profiles 声明，统一后重跑 gate/ops。",
                    }
                )
            if not bool(artifact_governance_checks.get("strict_mode_ok", True)):
                defects.append(
                    {
                        "category": "risk",
                        "code": "ARTIFACT_GOVERNANCE_STRICT_BLOCKED",
                        "message": "Artifact governance 严格模式阻断发布（存在 profile/policy/baseline 漂移）。",
                        "action": "先修复 governance 漂移并确认 strict_mode_ok=true，再继续 review-loop。",
                    }
                )

        rollback_payload = rollback_recommendation if isinstance(rollback_recommendation, dict) else {}
        rollback_level = str(rollback_payload.get("level", "none")).strip().lower() or "none"
        rollback_active = bool(rollback_payload.get("active", False))
        rollback_anchor_ready = bool(rollback_payload.get("anchor_ready", True))
        if rollback_level in {"soft", "hard"}:
            defects.append(
                {
                    "category": "risk",
                    "code": "ROLLBACK_RECOMMENDED",
                    "message": f"系统建议参数回滚（level={rollback_level}）",
                    "action": "按回滚建议执行参数回退并冻结自适应更新，完成局部回放后再恢复。",
                    "target_anchor": str(rollback_payload.get("target_anchor", "")),
                    "reason_codes": list(rollback_payload.get("reason_codes", []))[:10],
                }
            )
            if rollback_level == "hard":
                defects.append(
                    {
                        "category": "risk",
                        "code": "ROLLBACK_HARD",
                        "message": "触发硬回滚条件",
                        "action": "立即回滚并人工确认后再开放自动参数更新。",
                    }
                )
        if rollback_active and not rollback_anchor_ready:
            defects.append(
                {
                    "category": "risk",
                    "code": "ROLLBACK_ANCHOR_UNAVAILABLE",
                    "message": "回滚建议已触发但缺少可用锚点",
                    "action": "优先修复回滚锚点链路并补写备份文件。",
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

        default_actions = [
            "仅针对缺陷模块执行局部测试/回放",
            "局部通过后执行 lie test-all",
            "门槛通过后重跑 gate-report 与 ops-report",
        ]
        next_actions = list(default_actions)
        if any(str(x.get("code", "")).startswith("STATE_") for x in defects):
            next_actions = [
                f"优先修复 state_stability 缺陷并重跑 lie ops-report --date {as_of.isoformat()} --window-days 7",
                "状态稳定后再进行参数修正与回测收敛",
            ] + default_actions
        if any(str(x.get("code", "")).startswith("SLOT_") for x in defects):
            next_actions = [
                f"优先修复 slot_anomaly 缺陷并重跑 lie ops-report --date {as_of.isoformat()} --window-days 7",
                "补齐缺失槽位后再进行策略回测与参数更新。",
            ] + [x for x in next_actions if x not in default_actions] + default_actions
        if any(str(x.get("code", "")).startswith("MODE_DRIFT_") for x in defects):
            next_actions = [
                f"优先修复 mode_drift 缺陷并重跑 lie ops-report --date {as_of.isoformat()} --window-days 7",
                "确认 live/backtest 口径一致后再放开模式自适应更新",
            ] + [x for x in next_actions if x not in default_actions] + default_actions
        if any(str(x.get("code", "")).startswith("RECONCILE_") for x in defects):
            next_actions = [
                f"优先修复 reconcile_drift 缺陷并重跑 lie ops-report --date {as_of.isoformat()} --window-days 7",
                "对账一致后再执行参数更新与全量回测。",
            ] + [x for x in next_actions if x not in default_actions] + default_actions
        if any(str(x.get("code", "")).startswith("ARTIFACT_GOVERNANCE_") for x in defects):
            next_actions = [
                f"优先修复 artifact governance 策略漂移并重跑 lie gate-report --date {as_of.isoformat()}",
                "确认 profiles 与工件写入一致后，再执行 lie ops-report 与 review-loop。",
            ] + [x for x in next_actions if x not in default_actions] + default_actions
        if any(str(x.get("code", "")).startswith("ROLLBACK_") for x in defects):
            next_actions = [
                f"优先执行回滚建议并重跑 lie gate-report --date {as_of.isoformat()}",
                "回滚后先局部回放，再执行 lie test-all 与 review-loop。",
            ] + [x for x in next_actions if x not in default_actions] + default_actions
        if any(str(x.get("code", "")).startswith("TEST_TIMEOUT") for x in defects):
            next_actions = [
                "优先执行 deterministic fast shard 快测，定位超时模块与慢用例。",
                "拆分慢用例并完成局部回归后，再执行 lie test-all 全量验收。",
            ] + [x for x in next_actions if x not in default_actions] + default_actions

        plan = {
            "date": as_of.isoformat(),
            "round": round_no,
            "defect_count": len(defects),
            "defects": defects,
            "metrics": metrics,
            "checks": checks,
            "state_stability": state_payload,
            "temporal_audit": temporal_payload,
            "slot_anomaly": slot_payload,
            "mode_drift": drift_payload,
            "stress_matrix_trend": stress_payload,
            "reconcile_drift": reconcile_payload,
            "artifact_governance": artifact_governance_payload,
            "rollback_recommendation": rollback_payload,
            "next_actions": next_actions,
        }

        review_dir = self.output_dir / "review"
        json_path = review_dir / f"{as_of.isoformat()}_defect_plan_round{round_no}.json"
        md_path = review_dir / f"{as_of.isoformat()}_defect_plan_round{round_no}.md"
        write_json(json_path, plan)

        lines: list[str] = []
        lines.append(f"# 缺陷修正计划 | {as_of.isoformat()} | Round {round_no}")
        lines.append("")
        lines.append(f"- 缺陷数量: `{len(defects)}`")
        lines.append(f"- 最大回撤: `{metrics.get('max_drawdown', 'N/A')}`")
        lines.append(f"- 正收益窗口占比: `{metrics.get('positive_window_ratio', 'N/A')}`")
        if state_payload:
            lines.append(
                "- 状态稳定性: "
                + f"active={state_payload.get('active', False)}, alerts={state_payload.get('alerts', [])}"
            )
        if temporal_payload:
            lines.append(
                "- 时间审计: "
                + f"active={temporal_payload.get('active', False)}, alerts={temporal_payload.get('alerts', [])}"
            )
        if slot_payload:
            lines.append(
                "- 分时异常: "
                + f"active={slot_payload.get('active', False)}, alerts={slot_payload.get('alerts', [])}"
            )
        if drift_payload:
            lines.append(
                "- 模式漂移: "
                + f"active={drift_payload.get('active', False)}, alerts={drift_payload.get('alerts', [])}"
            )
        if stress_payload:
            lines.append(
                "- Stress Matrix 趋势: "
                + f"active={stress_payload.get('active', False)}, alerts={stress_payload.get('alerts', [])}"
            )
        if reconcile_payload:
            lines.append(
                "- 对账漂移: "
                + f"active={reconcile_payload.get('active', False)}, alerts={reconcile_payload.get('alerts', [])}"
            )
        if rollback_payload:
            lines.append(
                "- 回滚建议: "
                + f"level={rollback_payload.get('level', 'none')}, target={rollback_payload.get('target_anchor', '')}"
            )
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
        alert_path = self.output_dir / "logs" / f"review_loop_alert_{as_of.isoformat()}.json"
        if int(max_rounds) <= 0:
            return {
                "passed": False,
                "skipped": True,
                "reason": "max_rounds must be >= 1",
                "rounds": [],
            }

        rounds = []
        val = self.settings.validation if isinstance(self.settings.validation, dict) else {}
        fast_enabled = bool(val.get("review_loop_fast_test_enabled", True))
        fast_ratio = float(val.get("review_loop_fast_ratio", 0.10))
        fast_shard_index = int(val.get("review_loop_fast_shard_index", 0))
        fast_shard_total = int(val.get("review_loop_fast_shard_total", 1))
        fast_seed = str(val.get("review_loop_fast_seed", "lie-fast-v1"))
        fast_then_full = bool(val.get("review_loop_fast_then_full", True))
        timeout_fallback_enabled = bool(val.get("review_loop_timeout_fallback_enabled", True))
        timeout_fallback_ratio = max(0.01, min(1.0, float(val.get("review_loop_timeout_fallback_ratio", fast_ratio))))
        timeout_fallback_shard_total = max(1, int(val.get("review_loop_timeout_fallback_shard_total", fast_shard_total)))
        timeout_fallback_shard_index = int(val.get("review_loop_timeout_fallback_shard_index", fast_shard_index))
        if timeout_fallback_shard_index < 0:
            timeout_fallback_shard_index = 0
        if timeout_fallback_shard_index >= timeout_fallback_shard_total:
            timeout_fallback_shard_index = timeout_fallback_shard_total - 1
        timeout_fallback_seed = str(val.get("review_loop_timeout_fallback_seed", f"{fast_seed}-timeout"))
        stress_autorun_enabled = bool(val.get("review_loop_stress_matrix_autorun_enabled", True))
        stress_autorun_on_mode_drift = bool(val.get("review_loop_stress_matrix_autorun_on_mode_drift", True))
        stress_autorun_on_stress_breach = bool(val.get("review_loop_stress_matrix_autorun_on_stress_breach", True))
        stress_autorun_max_runs = max(0, int(val.get("review_loop_stress_matrix_autorun_max_runs", 1)))
        stress_autorun_adaptive_enabled = bool(
            val.get("review_loop_stress_matrix_autorun_adaptive_enabled", True)
        )
        stress_autorun_adaptive_window_days = max(
            1,
            int(val.get("review_loop_stress_matrix_autorun_adaptive_window_days", 30)),
        )
        stress_autorun_adaptive_min_rounds = max(
            1,
            int(val.get("review_loop_stress_matrix_autorun_adaptive_min_rounds", 6)),
        )
        stress_autorun_adaptive_low_density_threshold = self._safe_float(
            val.get("review_loop_stress_matrix_autorun_adaptive_low_density_threshold", 0.20),
            0.20,
        )
        stress_autorun_adaptive_high_density_threshold = self._safe_float(
            val.get("review_loop_stress_matrix_autorun_adaptive_high_density_threshold", 0.60),
            0.60,
        )
        stress_autorun_adaptive_low_density_factor = self._safe_float(
            val.get("review_loop_stress_matrix_autorun_adaptive_low_density_factor", 1.5),
            1.5,
        )
        stress_autorun_adaptive_high_density_factor = self._safe_float(
            val.get("review_loop_stress_matrix_autorun_adaptive_high_density_factor", 0.5),
            0.5,
        )
        stress_autorun_adaptive_min_runs_floor = max(
            0,
            int(val.get("review_loop_stress_matrix_autorun_adaptive_min_runs_floor", 0)),
        )
        stress_autorun_adaptive_max_runs_cap = max(
            int(stress_autorun_adaptive_min_runs_floor),
            int(
                val.get(
                    "review_loop_stress_matrix_autorun_adaptive_max_runs_cap",
                    max(3, int(stress_autorun_max_runs)),
                )
            ),
        )
        stress_autorun_modes_raw = val.get("review_loop_stress_matrix_autorun_modes", [])
        stress_autorun_modes: list[str] = []
        if isinstance(stress_autorun_modes_raw, list):
            for raw_mode in stress_autorun_modes_raw:
                m = str(raw_mode).strip()
                if m and m not in stress_autorun_modes:
                    stress_autorun_modes.append(m)
        stress_autorun_runs = 0
        stress_autorun_cooldown_rounds = max(0, int(val.get("review_loop_stress_matrix_autorun_cooldown_rounds", 0)))
        stress_autorun_backoff_multiplier = max(
            1.0,
            self._safe_float(val.get("review_loop_stress_matrix_autorun_backoff_multiplier", 1.0), 1.0),
        )
        stress_autorun_backoff_max_rounds = max(
            int(stress_autorun_cooldown_rounds),
            int(val.get("review_loop_stress_matrix_autorun_backoff_max_rounds", stress_autorun_cooldown_rounds)),
        )
        stress_autorun_current_cooldown = int(stress_autorun_cooldown_rounds)
        stress_autorun_next_allowed_round = 1

        for i in range(int(max_rounds)):
            round_no = i + 1
            review = self.run_review(as_of)
            run_fast = bool(i == 0 and fast_enabled)
            tests = self._run_tests(
                fast=run_fast,
                fast_ratio=fast_ratio,
                fast_shard_index=fast_shard_index,
                fast_shard_total=fast_shard_total,
                fast_seed=fast_seed,
            )
            fast_tests = tests if run_fast else {}
            full_tests = tests if (not run_fast) else {}
            if run_fast and tests.get("returncode", 1) == 0 and fast_then_full:
                full_tests = self._run_tests(
                    fast=False,
                    fast_ratio=fast_ratio,
                    fast_shard_index=fast_shard_index,
                    fast_shard_total=fast_shard_total,
                    fast_seed=fast_seed,
                )
                tests = full_tests
            tests_timeout = self._is_timeout_payload(tests)
            timeout_fallback: dict[str, Any] = {}
            if tests_timeout and timeout_fallback_enabled:
                timeout_fallback = self._run_tests(
                    fast=True,
                    fast_ratio=timeout_fallback_ratio,
                    fast_shard_index=timeout_fallback_shard_index,
                    fast_shard_total=timeout_fallback_shard_total,
                    fast_seed=timeout_fallback_seed,
                )
                tests = dict(tests)
                tests["timeout_fallback"] = timeout_fallback
                tests["timeout_fallback_used"] = True

            gate = self.gate_report(as_of=as_of, run_tests=False, run_review_if_missing=False)
            state_stability = self._state_stability_metrics(as_of=as_of)
            state_checks = state_stability.get("checks", {}) if isinstance(state_stability.get("checks", {}), dict) else {}
            state_active = bool(state_stability.get("active", False))
            state_ok = all(bool(v) for v in state_checks.values()) if state_active else True
            temporal_audit = gate.get("temporal_audit", {}) if isinstance(gate.get("temporal_audit", {}), dict) else {}
            temporal_checks = (
                temporal_audit.get("checks", {}) if isinstance(temporal_audit.get("checks", {}), dict) else {}
            )
            temporal_active = bool(temporal_audit.get("active", False))
            temporal_ok = all(bool(v) for v in temporal_checks.values()) if temporal_active else True
            slot_anomaly = gate.get("slot_anomaly", {}) if isinstance(gate.get("slot_anomaly", {}), dict) else {}
            slot_checks = slot_anomaly.get("checks", {}) if isinstance(slot_anomaly.get("checks", {}), dict) else {}
            slot_active = bool(slot_anomaly.get("active", False))
            slot_ok = all(bool(v) for v in slot_checks.values()) if slot_active else True
            mode_drift = gate.get("mode_drift", {}) if isinstance(gate.get("mode_drift", {}), dict) else {}
            drift_checks = mode_drift.get("checks", {}) if isinstance(mode_drift.get("checks", {}), dict) else {}
            drift_active = bool(mode_drift.get("active", False))
            drift_ok = all(bool(v) for v in drift_checks.values()) if drift_active else True
            stress_matrix_trend = (
                gate.get("stress_matrix_trend", {})
                if isinstance(gate.get("stress_matrix_trend", {}), dict)
                else {}
            )
            stress_checks = (
                stress_matrix_trend.get("checks", {})
                if isinstance(stress_matrix_trend.get("checks", {}), dict)
                else {}
            )
            stress_active = bool(stress_matrix_trend.get("active", False))
            stress_ok = all(bool(v) for v in stress_checks.values()) if stress_active else True
            stress_autorun_adaptive = self._stress_autorun_adaptive_max_runs(
                as_of=as_of,
                base_max_runs=stress_autorun_max_runs,
                current_rounds=rounds,
                enabled=stress_autorun_adaptive_enabled,
                window_days=stress_autorun_adaptive_window_days,
                min_rounds=stress_autorun_adaptive_min_rounds,
                low_density_threshold=stress_autorun_adaptive_low_density_threshold,
                high_density_threshold=stress_autorun_adaptive_high_density_threshold,
                low_density_factor=stress_autorun_adaptive_low_density_factor,
                high_density_factor=stress_autorun_adaptive_high_density_factor,
                min_runs_floor=stress_autorun_adaptive_min_runs_floor,
                max_runs_cap=stress_autorun_adaptive_max_runs_cap,
            )
            stress_autorun_effective_max_runs = max(0, int(stress_autorun_adaptive.get("effective_max_runs", 0)))
            stress_autorun_payload: dict[str, Any] = {
                "enabled": bool(stress_autorun_enabled),
                "triggered": False,
                "attempted": False,
                "ran": False,
                "reason_codes": [],
                "skipped_reason": "",
                "error": "",
                "max_runs": int(stress_autorun_effective_max_runs),
                "max_runs_base": int(stress_autorun_max_runs),
                "runs_used": int(stress_autorun_runs),
                "modes": list(stress_autorun_modes),
                "adaptive": {
                    "enabled": bool(stress_autorun_adaptive.get("enabled", False)),
                    "reason": str(stress_autorun_adaptive.get("reason", "")),
                    "factor": float(self._safe_float(stress_autorun_adaptive.get("factor", 1.0), 1.0)),
                    "trigger_density": float(
                        self._safe_float(stress_autorun_adaptive.get("trigger_density", 0.0), 0.0)
                    ),
                    "rounds_total": int(self._safe_float(stress_autorun_adaptive.get("rounds_total", 0), 0)),
                    "triggered_rounds": int(
                        self._safe_float(stress_autorun_adaptive.get("triggered_rounds", 0), 0)
                    ),
                    "window_days": int(self._safe_float(stress_autorun_adaptive.get("window_days", 0), 0)),
                    "min_rounds": int(self._safe_float(stress_autorun_adaptive.get("min_rounds", 0), 0)),
                    "min_runs_floor": int(
                        self._safe_float(stress_autorun_adaptive.get("min_runs_floor", 0), 0)
                    ),
                    "max_runs_cap": int(self._safe_float(stress_autorun_adaptive.get("max_runs_cap", 0), 0)),
                    "low_density_threshold": float(
                        self._safe_float(
                            stress_autorun_adaptive.get("low_density_threshold", 0.0),
                            0.0,
                        )
                    ),
                    "high_density_threshold": float(
                        self._safe_float(
                            stress_autorun_adaptive.get("high_density_threshold", 0.0),
                            0.0,
                        )
                    ),
                    "low_density_factor": float(
                        self._safe_float(stress_autorun_adaptive.get("low_density_factor", 0.0), 0.0)
                    ),
                    "high_density_factor": float(
                        self._safe_float(stress_autorun_adaptive.get("high_density_factor", 0.0), 0.0)
                    ),
                },
                "cooldown_rounds_base": int(stress_autorun_cooldown_rounds),
                "cooldown_rounds_current": int(stress_autorun_current_cooldown),
                "cooldown_remaining_rounds": int(max(0, stress_autorun_next_allowed_round - round_no)),
                "next_allowed_round": int(stress_autorun_next_allowed_round),
                "backoff_multiplier": float(stress_autorun_backoff_multiplier),
                "backoff_max_rounds": int(stress_autorun_backoff_max_rounds),
                "output": {},
            }
            stress_trigger_codes: list[str] = []
            if stress_autorun_on_mode_drift and drift_active and (not drift_ok):
                stress_trigger_codes.append("mode_drift")
            if stress_autorun_on_stress_breach and stress_active and (not stress_ok):
                stress_trigger_codes.append("stress_trend")
            if stress_autorun_enabled and stress_trigger_codes:
                stress_autorun_payload["triggered"] = True
                stress_autorun_payload["reason_codes"] = list(stress_trigger_codes)
                if self.run_stress_matrix is None:
                    stress_autorun_payload["skipped_reason"] = "runner_unavailable"
                elif round_no < stress_autorun_next_allowed_round:
                    stress_autorun_payload["skipped_reason"] = "cooldown_active"
                elif stress_autorun_runs >= stress_autorun_effective_max_runs:
                    stress_autorun_payload["skipped_reason"] = "max_runs_reached"
                else:
                    stress_autorun_payload["attempted"] = True
                    stress_autorun_runs += 1
                    try:
                        stress_out = self.run_stress_matrix(
                            as_of,
                            (list(stress_autorun_modes) if stress_autorun_modes else None),
                        )
                        stress_autorun_payload["ran"] = True
                        out_paths = stress_out.get("paths", {}) if isinstance(stress_out.get("paths", {}), dict) else {}
                        stress_autorun_payload["output"] = {
                            "best_mode": str(stress_out.get("best_mode", "")),
                            "mode_count": int(self._safe_float(stress_out.get("mode_count", 0), 0.0)),
                            "window_count": int(self._safe_float(stress_out.get("window_count", 0), 0.0)),
                            "json": str(out_paths.get("json", "")),
                            "md": str(out_paths.get("md", "")),
                        }
                    except Exception as exc:
                        stress_autorun_payload["error"] = f"{type(exc).__name__}:{exc}"
                    if stress_autorun_current_cooldown > 0:
                        stress_autorun_next_allowed_round = round_no + stress_autorun_current_cooldown + 1
                        next_cooldown = int(math.ceil(stress_autorun_current_cooldown * stress_autorun_backoff_multiplier))
                        stress_autorun_current_cooldown = min(
                            stress_autorun_backoff_max_rounds,
                            max(stress_autorun_cooldown_rounds, next_cooldown),
                        )
                    else:
                        stress_autorun_next_allowed_round = round_no + 1
            stress_autorun_payload["runs_used"] = int(stress_autorun_runs)
            stress_autorun_payload["cooldown_rounds_current"] = int(stress_autorun_current_cooldown)
            stress_autorun_payload["next_allowed_round"] = int(stress_autorun_next_allowed_round)
            stress_autorun_payload["cooldown_remaining_rounds"] = int(
                max(0, stress_autorun_next_allowed_round - round_no)
            )
            reconcile_drift = gate.get("reconcile_drift", {}) if isinstance(gate.get("reconcile_drift", {}), dict) else {}
            reconcile_checks = (
                reconcile_drift.get("checks", {}) if isinstance(reconcile_drift.get("checks", {}), dict) else {}
            )
            reconcile_active = bool(reconcile_drift.get("active", False))
            reconcile_ok = all(bool(v) for v in reconcile_checks.values()) if reconcile_active else True
            rollback_rec = (
                gate.get("rollback_recommendation", {})
                if isinstance(gate.get("rollback_recommendation", {}), dict)
                else {}
            )
            ok = bool(gate["passed"] and tests["returncode"] == 0 and review.pass_gate)
            tests_mode = "fast+full" if (run_fast and fast_then_full and full_tests) else ("fast" if run_fast else "full")
            if timeout_fallback:
                tests_mode = tests_mode + "+timeout-fast"
            rounds.append(
                {
                    "round": round_no,
                    "tests_mode": tests_mode,
                    "pass_gate": review.pass_gate,
                    "tests_ok": tests["returncode"] == 0,
                    "tests_timeout": bool(tests_timeout),
                    "stable_replay_ok": gate["checks"]["stable_replay_ok"],
                    "stable_replay_days": gate["stable_replay"]["replay_days"],
                    "gate_passed": gate["passed"],
                    "state_stability_active": state_active,
                    "state_stability_ok": state_ok,
                    "state_alerts": list(state_stability.get("alerts", [])),
                    "temporal_audit_active": temporal_active,
                    "temporal_audit_ok": temporal_ok,
                    "temporal_audit_alerts": list(temporal_audit.get("alerts", [])),
                    "slot_anomaly_active": slot_active,
                    "slot_anomaly_ok": slot_ok,
                    "slot_alerts": list(slot_anomaly.get("alerts", [])),
                    "mode_drift_active": drift_active,
                    "mode_drift_ok": drift_ok,
                    "mode_drift_alerts": list(mode_drift.get("alerts", [])),
                    "stress_matrix_trend_active": stress_active,
                    "stress_matrix_trend_ok": stress_ok,
                    "stress_matrix_trend_alerts": list(stress_matrix_trend.get("alerts", [])),
                    "stress_matrix_autorun": stress_autorun_payload,
                    "reconcile_drift_active": reconcile_active,
                    "reconcile_drift_ok": reconcile_ok,
                    "reconcile_drift_alerts": list(reconcile_drift.get("alerts", [])),
                    "rollback_level": str(rollback_rec.get("level", "none")),
                    "rollback_action": str(rollback_rec.get("action", "no_rollback")),
                    "rollback_anchor_ready": bool(rollback_rec.get("anchor_ready", True)),
                    "timeout_fallback_used": bool(timeout_fallback),
                    "timeout_fallback_ok": bool(timeout_fallback and int(timeout_fallback.get("returncode", 1)) == 0),
                }
            )
            if fast_tests:
                rounds[-1]["fast_tests"] = {
                    "returncode": int(fast_tests.get("returncode", 1)),
                    "summary_line": str(fast_tests.get("summary_line", "")),
                    "tests_ran": int(fast_tests.get("tests_ran", 0)),
                    "failed_tests": list(fast_tests.get("failed_tests", []))[:20],
                }
            if full_tests:
                rounds[-1]["full_tests"] = {
                    "returncode": int(full_tests.get("returncode", 1)),
                    "summary_line": str(full_tests.get("summary_line", "")),
                    "tests_ran": int(full_tests.get("tests_ran", 0)),
                    "failed_tests": list(full_tests.get("failed_tests", []))[:20],
                }
            if timeout_fallback:
                rounds[-1]["timeout_fallback"] = {
                    "returncode": int(timeout_fallback.get("returncode", 1)),
                    "summary_line": str(timeout_fallback.get("summary_line", "")),
                    "tests_ran": int(timeout_fallback.get("tests_ran", 0)),
                    "failed_tests": list(timeout_fallback.get("failed_tests", []))[:20],
                    "fast_ratio": float(timeout_fallback_ratio),
                    "fast_shard_index": int(timeout_fallback_shard_index),
                    "fast_shard_total": int(timeout_fallback_shard_total),
                    "fast_seed": timeout_fallback_seed,
                }
            if ok:
                release_path = self.output_dir / "artifacts" / f"release_ready_{as_of.isoformat()}.json"
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
                state_stability=state_stability,
                temporal_audit=temporal_audit,
                slot_anomaly=slot_anomaly,
                mode_drift=mode_drift,
                stress_matrix_trend=stress_matrix_trend,
                reconcile_drift=reconcile_drift,
                rollback_recommendation=rollback_rec,
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
        health = self.health_check(as_of, True)
        return {
            "date": as_of.isoformat(),
            "review_loop": review_loop,
            "gate_report": gate,
            "ops_report": ops,
            "health": health,
        }
