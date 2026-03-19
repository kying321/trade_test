#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml


def resolve_system_root() -> Path:
    env_root = str(os.getenv("LIE_SYSTEM_ROOT", "")).strip() or str(
        os.getenv("FENLIE_SYSTEM_ROOT", "")
    ).strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


SYSTEM_ROOT = resolve_system_root()
DEFAULT_OUTPUT_ROOT = SYSTEM_ROOT / "output"
DEFAULT_REVIEW_DIR = DEFAULT_OUTPUT_ROOT / "review"
DEFAULT_MICRO_CAPTURE_DIR = DEFAULT_OUTPUT_ROOT / "artifacts" / "micro_capture"
DEFAULT_TIME_SYNC_DIR = DEFAULT_OUTPUT_ROOT / "artifacts" / "time_sync"
DEFAULT_CONFIG_PATH = SYSTEM_ROOT / "config.yaml"
DEFAULT_CVD_LOCAL_WINDOW_MINUTES = 15
DEFAULT_CVD_REFERENCE_MAX_AGE_MINUTES = 15
DEFAULT_CVD_ATTACK_ALIGNMENT_MIN_ABS = 0.05


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def fmt_utc(value: dt.datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(raw: str, *, anchor: Path) -> Path:
    path = Path(str(raw).strip())
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    return anchor / path


def safe_float(raw: Any) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not value == value:
        return None
    return value


def load_cvd_policy(config_path: Path) -> dict[str, Any]:
    payload: Any = {}
    if config_path.exists():
        try:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    shortline = payload.get("shortline", {}) if isinstance(payload, dict) else {}
    return {
        "cvd_local_window_minutes": int(
            shortline.get("cvd_local_window_minutes", DEFAULT_CVD_LOCAL_WINDOW_MINUTES)
            or DEFAULT_CVD_LOCAL_WINDOW_MINUTES
        ),
        "cvd_reference_max_age_minutes": int(
            shortline.get(
                "cvd_reference_max_age_minutes",
                shortline.get("cvd_local_window_minutes", DEFAULT_CVD_REFERENCE_MAX_AGE_MINUTES),
            )
            or DEFAULT_CVD_REFERENCE_MAX_AGE_MINUTES
        ),
        "cvd_attack_alignment_min_abs": float(
            shortline.get("cvd_attack_alignment_min_abs", DEFAULT_CVD_ATTACK_ALIGNMENT_MIN_ABS)
            or DEFAULT_CVD_ATTACK_ALIGNMENT_MIN_ABS
        ),
        "cvd_key_level_only": bool(shortline.get("cvd_key_level_only", True)),
        "cvd_drift_guard_enabled": bool(shortline.get("cvd_drift_guard_enabled", True)),
    }


def load_json_mapping(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def parse_timestamp(raw: Any) -> dt.datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def find_latest_micro_capture(artifact_dir: Path) -> Path | None:
    files = sorted(
        artifact_dir.glob("*_micro_capture.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def find_latest_time_sync_probe(artifact_dir: Path) -> Path | None:
    files = sorted(
        artifact_dir.glob("*_time_sync_probe.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def find_latest_time_sync_environment_report(review_dir: Path) -> Path | None:
    files = sorted(
        review_dir.glob("*_system_time_sync_environment_report.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def strip_time_sync_risk(note: str) -> str:
    parts = [part.strip() for part in str(note or "").split("|") if part.strip()]
    return "|".join(part for part in parts if part != "time_sync_risk")


def local_cvd_features(row: dict[str, Any], *, policy: dict[str, Any]) -> dict[str, Any]:
    note = str(row.get("cvd_context_note", "")).strip().lower()
    alignment = float(row.get("micro_alignment") or 0.0)
    age_minutes = safe_float(
        row.get("cvd_reference_age_minutes")
        or row.get("cvd_local_reference_age_minutes")
        or row.get("cvd_local_age_minutes")
    )
    local_window_minutes = int(policy.get("cvd_local_window_minutes") or DEFAULT_CVD_LOCAL_WINDOW_MINUTES)
    max_age_minutes = int(
        policy.get("cvd_reference_max_age_minutes") or DEFAULT_CVD_REFERENCE_MAX_AGE_MINUTES
    )
    locality_status = str(row.get("cvd_locality_status", "")).strip()
    local_window_ok = True if age_minutes is None else age_minutes <= float(local_window_minutes)
    if not locality_status:
        locality_status = (
            "proxy_from_current_snapshot"
            if age_minutes is None
            else ("local_window_ok" if local_window_ok else "outside_local_window")
        )
    drift_risk = bool(row.get("cvd_drift_risk", False))
    if not drift_risk and "drift" in note:
        drift_risk = True
    if (
        bool(policy.get("cvd_drift_guard_enabled", True))
        and age_minutes is not None
        and age_minutes > float(max_age_minutes)
    ):
        drift_risk = True
    near_key_level_raw = row.get("cvd_near_key_level")
    key_level_confirmed = True if near_key_level_raw in {None, ""} else bool(near_key_level_raw)
    attack_side = str(row.get("cvd_attack_side", "")).strip().lower()
    attack_threshold = float(
        policy.get("cvd_attack_alignment_min_abs") or DEFAULT_CVD_ATTACK_ALIGNMENT_MIN_ABS
    )
    if attack_side not in {"buyers", "sellers", "balanced"}:
        if alignment >= attack_threshold:
            attack_side = "buyers"
        elif alignment <= -attack_threshold:
            attack_side = "sellers"
        else:
            attack_side = "balanced"
    attack_presence = str(row.get("cvd_attack_presence", "")).strip().lower()
    if not attack_presence:
        attack_presence = {
            "buyers": "buyers_attacking",
            "sellers": "sellers_attacking",
            "balanced": "no_clear_attack",
        }.get(attack_side, "no_clear_attack")
    return {
        "cvd_locality_status": locality_status,
        "cvd_reference_age_minutes": age_minutes,
        "cvd_local_window_ok": local_window_ok,
        "cvd_drift_risk": drift_risk,
        "cvd_key_level_confirmed": key_level_confirmed,
        "cvd_attack_side": attack_side,
        "cvd_attack_presence": attack_presence,
    }


def classify_symbol(row: dict[str, Any], *, policy: dict[str, Any]) -> tuple[str, list[str], dict[str, Any]]:
    context = str(row.get("cvd_context_mode", "unclear")).strip() or "unclear"
    trust = str(row.get("cvd_trust_tier_hint", "unavailable")).strip() or "unavailable"
    veto = str(row.get("cvd_veto_hint", "")).strip()
    note = str(row.get("cvd_context_note", "")).strip()
    local = local_cvd_features(row, policy=policy)
    reasons: list[str] = []
    if not bool(row.get("time_sync_ok", True)):
        reasons.append("time_sync_risk")
    if trust in {"single_exchange_low", "unavailable"}:
        reasons.append("trust_low")
    if veto:
        reasons.append(veto)
    if "time_sync_risk" in note and "time_sync_risk" not in reasons:
        reasons.append("time_sync_risk")
    if bool(local.get("cvd_drift_risk", False)):
        reasons.append("cvd_drift_risk")
    if bool(policy.get("cvd_key_level_only", True)) and not bool(local.get("cvd_key_level_confirmed", True)):
        reasons.append("cvd_not_confirmed_near_key_level")

    if reasons:
        return "watch_only", reasons, local
    if context == "continuation":
        return "trend_confirmation_watch", reasons, local
    if context in {"reversal", "absorption", "failed_auction"}:
        return "reversal_absorption_watch", reasons, local
    return "unclear", reasons, local


def build_symbol_rows(
    selected_micro: list[dict[str, Any]],
    *,
    policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, list[str]], dict[str, int], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    buckets: dict[str, list[str]] = {
        "trend_confirmation_watch": [],
        "reversal_absorption_watch": [],
        "watch_only": [],
        "unclear": [],
    }
    locality_counts: dict[str, int] = {}
    attack_side_counts: dict[str, int] = {}
    for row in selected_micro:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        classification, reasons, local = classify_symbol(row, policy=policy)
        buckets.setdefault(classification, []).append(symbol)
        locality = str(local.get("cvd_locality_status", "")).strip() or "unknown"
        attack_side = str(local.get("cvd_attack_side", "")).strip() or "unknown"
        locality_counts[locality] = int(locality_counts.get(locality, 0)) + 1
        attack_side_counts[attack_side] = int(attack_side_counts.get(attack_side, 0)) + 1
        rows.append(
            {
                "symbol": symbol,
                "classification": classification,
                "cvd_context_mode": str(row.get("cvd_context_mode", "unclear")),
                "cvd_trust_tier_hint": str(row.get("cvd_trust_tier_hint", "unavailable")),
                "cvd_veto_hint": str(row.get("cvd_veto_hint", "")),
                "time_sync_ok": bool(row.get("time_sync_ok", False)),
                "schema_ok": bool(row.get("schema_ok", False)),
                "sync_ok": bool(row.get("sync_ok", False)),
                "trade_count": int(row.get("trade_count", 0) or 0),
                "evidence_score": float(row.get("evidence_score", 0.0) or 0.0),
                "cvd_context_note": str(row.get("cvd_context_note", "")),
                "active_reasons": reasons,
                "cvd_locality_status": locality,
                "cvd_reference_age_minutes": local.get("cvd_reference_age_minutes"),
                "cvd_local_window_ok": bool(local.get("cvd_local_window_ok", False)),
                "cvd_drift_risk": bool(local.get("cvd_drift_risk", False)),
                "cvd_key_level_confirmed": bool(local.get("cvd_key_level_confirmed", False)),
                "cvd_attack_side": attack_side,
                "cvd_attack_presence": str(local.get("cvd_attack_presence", "")),
            }
        )
    return rows, buckets, locality_counts, attack_side_counts


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Crypto CVD Semantic Snapshot",
        "",
        f"- source artifact: `{payload.get('source_artifact') or ''}`",
        f"- as_of: `{payload.get('as_of') or ''}`",
        f"- status: `{payload.get('source_status') or ''}`",
        f"- pass: `{payload.get('source_pass')}`",
        f"- takeaway: {payload.get('takeaway') or ''}",
        "",
        "## Semantic Summary",
        f"- context counts: `{json.dumps(payload.get('context_counts', {}), ensure_ascii=False, sort_keys=True)}`",
        f"- trust counts: `{json.dumps(payload.get('trust_counts', {}), ensure_ascii=False, sort_keys=True)}`",
        f"- veto counts: `{json.dumps(payload.get('veto_hint_counts', {}), ensure_ascii=False, sort_keys=True)}`",
        f"- locality counts: `{json.dumps(payload.get('locality_counts', {}), ensure_ascii=False, sort_keys=True)}`",
        f"- attack counts: `{json.dumps(payload.get('attack_side_counts', {}), ensure_ascii=False, sort_keys=True)}`",
        f"- time_sync_status: `{payload.get('time_sync_status') or '-'}`",
        f"- time_sync_classification: `{payload.get('time_sync_classification') or '-'}`",
        f"- time_sync_intercept_scope: `{payload.get('time_sync_intercept_scope') or '-'}`",
        f"- time_sync_blocker_detail: `{payload.get('time_sync_blocker_detail') or '-'}`",
        f"- time_sync_remediation_hint: `{payload.get('time_sync_remediation_hint') or '-'}`",
        f"- time_sync_environment_classification: `{payload.get('time_sync_environment_classification') or '-'}`",
        f"- time_sync_environment_blocker_detail: `{payload.get('time_sync_environment_blocker_detail') or '-'}`",
        "",
        "## Queue Classification",
        f"- trend-confirmation-watch: `{', '.join(payload.get('trend_confirmation_watch', [])) or '-'}`",
        f"- reversal-absorption-watch: `{', '.join(payload.get('reversal_absorption_watch', [])) or '-'}`",
        f"- watch-only: `{', '.join(payload.get('watch_only_symbols', [])) or '-'}`",
        f"- unclear: `{', '.join(payload.get('unclear_symbols', [])) or '-'}`",
        "",
        "## Symbols",
    ]
    for row in payload.get("symbols", []):
        if not isinstance(row, dict):
            continue
        lines.extend(
            [
                f"- `{row.get('symbol')}`",
                f"  - class: `{row.get('classification')}`",
                f"  - context: `{row.get('cvd_context_mode')}`",
                f"  - trust: `{row.get('cvd_trust_tier_hint')}`",
                f"  - veto: `{row.get('cvd_veto_hint') or '-'}`",
                f"  - time_sync_ok: `{row.get('time_sync_ok')}`",
                f"  - trade_count: `{row.get('trade_count')}`",
                f"  - evidence_score: `{row.get('evidence_score')}`",
                f"  - locality: `{row.get('cvd_locality_status')}`",
                f"  - ref_age_min: `{row.get('cvd_reference_age_minutes')}`",
                f"  - drift_risk: `{row.get('cvd_drift_risk')}`",
                f"  - attack: `{row.get('cvd_attack_side')} | {row.get('cvd_attack_presence')}`",
                f"  - reasons: `{', '.join(row.get('active_reasons', [])) or '-'}`",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def prune_artifacts(
    review_dir: Path,
    *,
    current_artifact: Path,
    current_checksum: Path,
    current_markdown: Path,
    keep: int,
    ttl_hours: float,
) -> tuple[list[str], list[str]]:
    review_dir.mkdir(parents=True, exist_ok=True)
    cutoff = now_utc() - dt.timedelta(hours=max(1.0, ttl_hours))
    protected = {current_artifact.name, current_checksum.name, current_markdown.name}
    candidates: list[Path] = []
    for pattern in (
        "*_crypto_cvd_semantic_snapshot.json",
        "*_crypto_cvd_semantic_snapshot_checksum.json",
        "*_crypto_cvd_semantic_snapshot.md",
    ):
        candidates.extend(review_dir.glob(pattern))

    pruned_age: list[str] = []
    survivors: list[Path] = []
    for path in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
        if path.name in protected:
            survivors.append(path)
            continue
        try:
            mtime = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)
        except OSError:
            continue
        if mtime < cutoff:
            path.unlink(missing_ok=True)
            pruned_age.append(str(path))
        else:
            survivors.append(path)

    pruned_keep: list[str] = []
    json_like = [p for p in survivors if p.suffix in {".json", ".md"}]
    for path in json_like[keep:]:
        if path.name in protected:
            continue
        path.unlink(missing_ok=True)
        pruned_keep.append(str(path))
    return pruned_keep, pruned_age


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a semantic snapshot from the latest micro capture artifact.")
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--artifact-dir", default=str(DEFAULT_MICRO_CAPTURE_DIR))
    parser.add_argument("--micro-capture-file", default="")
    parser.add_argument("--time-sync-probe-file", default="")
    parser.add_argument("--time-sync-environment-file", default="")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--artifact-keep", type=int, default=16)
    parser.add_argument("--now", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    review_dir = Path(args.review_dir).expanduser().resolve()
    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    config_path = resolve_path(args.config, anchor=SYSTEM_ROOT)
    policy = load_cvd_policy(config_path)
    now_ts = dt.datetime.fromisoformat(args.now) if str(args.now).strip() else now_utc()

    source_path = (
        Path(args.micro_capture_file).expanduser().resolve()
        if str(args.micro_capture_file).strip()
        else find_latest_micro_capture(artifact_dir)
    )
    source_payload = load_json_mapping(source_path)
    time_sync_probe_dir = artifact_dir.parent / "time_sync"
    time_sync_probe_path = (
        Path(args.time_sync_probe_file).expanduser().resolve()
        if str(args.time_sync_probe_file).strip()
        else find_latest_time_sync_probe(time_sync_probe_dir) if time_sync_probe_dir.exists() else None
    )
    time_sync_probe_payload = load_json_mapping(time_sync_probe_path) or {}
    time_sync_environment_path = (
        Path(args.time_sync_environment_file).expanduser().resolve()
        if str(args.time_sync_environment_file).strip()
        else find_latest_time_sync_environment_report(review_dir)
    )
    time_sync_environment_payload = load_json_mapping(time_sync_environment_path) or {}

    status = "ok"
    ok = True
    selected_micro = []
    if source_payload is None:
        status = "micro_capture_missing"
        ok = False
    else:
        selected_micro = source_payload.get("selected_micro", [])
        if not isinstance(selected_micro, list) or not selected_micro:
            status = "selected_micro_missing"
            ok = False

    source_captured_at = parse_timestamp(
        (source_payload or {}).get("captured_at_utc") or (source_payload or {}).get("as_of")
    )
    probe_captured_at = parse_timestamp(
        time_sync_probe_payload.get("captured_at_utc") or time_sync_probe_payload.get("as_of")
    )
    probe_newer_than_capture = bool(
        probe_captured_at and (source_captured_at is None or probe_captured_at >= source_captured_at)
    )
    semantics = source_payload.get("cvd_semantics", {}) if isinstance(source_payload, dict) else {}
    context_counts = semantics.get("context_counts", {}) if isinstance(semantics, dict) else {}
    trust_counts = semantics.get("trust_counts", {}) if isinstance(semantics, dict) else {}
    veto_hint_counts = semantics.get("veto_hint_counts", {}) if isinstance(semantics, dict) else {}
    system_time_sync = (
        source_payload.get("system_time_sync", {})
        if isinstance(source_payload, dict) and isinstance(source_payload.get("system_time_sync", {}), dict)
        else {}
    )
    environment_diagnostics = (
        source_payload.get("environment_diagnostics", {})
        if isinstance(source_payload, dict) and isinstance(source_payload.get("environment_diagnostics", {}), dict)
        else {}
    )
    environment_status = str(environment_diagnostics.get("status", "")).strip()
    environment_classification = str(environment_diagnostics.get("classification", "")).strip()
    environment_blocker_detail = str(environment_diagnostics.get("blocker_detail", "")).strip()
    environment_remediation_hint = str(environment_diagnostics.get("remediation_hint", "")).strip()
    if time_sync_environment_payload and probe_newer_than_capture:
        environment_status = str(time_sync_environment_payload.get("status", "")).strip() or environment_status
        environment_classification = (
            str(time_sync_environment_payload.get("classification", "")).strip() or environment_classification
        )
        environment_blocker_detail = (
            str(time_sync_environment_payload.get("blocker_detail", "")).strip() or environment_blocker_detail
        )
        environment_remediation_hint = (
            str(time_sync_environment_payload.get("remediation_hint", "")).strip() or environment_remediation_hint
        )
    time_sync_status = str(system_time_sync.get("status", "")).strip()
    time_sync_classification = str(system_time_sync.get("classification", "")).strip()
    time_sync_remediation_hint = str(system_time_sync.get("remediation_hint", "")).strip()
    time_sync_intercept_scope = str(system_time_sync.get("fake_ip_intercept_scope", "")).strip()
    time_sync_fake_ip_sources = [
        str(x).strip()
        for x in list(system_time_sync.get("fake_ip_sources", []))
        if str(x).strip()
    ]
    time_sync_threshold_breach_sources = [
        str(x).strip()
        for x in list(system_time_sync.get("threshold_breach_sources", []))
        if str(x).strip()
    ]
    time_sync_threshold_breach_scope = str(system_time_sync.get("threshold_breach_scope", "")).strip()
    time_sync_threshold_breach_offset_sources = [
        str(x).strip()
        for x in list(system_time_sync.get("threshold_breach_offset_sources", []))
        if str(x).strip()
    ]
    time_sync_threshold_breach_latency_sources = [
        str(x).strip()
        for x in list(system_time_sync.get("threshold_breach_latency_sources", []))
        if str(x).strip()
    ]
    time_sync_threshold_breach_estimated_offset_ms = system_time_sync.get("threshold_breach_estimated_offset_ms")
    time_sync_threshold_breach_estimated_rtt_ms = system_time_sync.get("threshold_breach_estimated_rtt_ms")
    if time_sync_probe_payload and probe_newer_than_capture:
        system_time_sync = dict(time_sync_probe_payload)
        time_sync_status = str(system_time_sync.get("status", "")).strip()
        time_sync_classification = str(system_time_sync.get("classification", "")).strip()
        time_sync_remediation_hint = str(system_time_sync.get("remediation_hint", "")).strip()
        time_sync_intercept_scope = str(system_time_sync.get("fake_ip_intercept_scope", "")).strip()
        time_sync_fake_ip_sources = [
            str(x).strip()
            for x in list(system_time_sync.get("fake_ip_sources", []))
            if str(x).strip()
        ]
        time_sync_threshold_breach_sources = [
            str(x).strip()
            for x in list(system_time_sync.get("threshold_breach_sources", []))
            if str(x).strip()
        ]
        time_sync_threshold_breach_scope = str(system_time_sync.get("threshold_breach_scope", "")).strip()
        time_sync_threshold_breach_offset_sources = [
            str(x).strip()
            for x in list(system_time_sync.get("threshold_breach_offset_sources", []))
            if str(x).strip()
        ]
        time_sync_threshold_breach_latency_sources = [
            str(x).strip()
            for x in list(system_time_sync.get("threshold_breach_latency_sources", []))
            if str(x).strip()
        ]
        time_sync_threshold_breach_estimated_offset_ms = system_time_sync.get(
            "threshold_breach_estimated_offset_ms"
        )
        time_sync_threshold_breach_estimated_rtt_ms = system_time_sync.get(
            "threshold_breach_estimated_rtt_ms"
        )
    effective_selected_micro = selected_micro if isinstance(selected_micro, list) else []
    if probe_newer_than_capture and effective_selected_micro:
        effective_time_sync_ok = bool(system_time_sync.get("pass", False))
        rebased_rows: list[dict[str, Any]] = []
        for raw_row in effective_selected_micro:
            if not isinstance(raw_row, dict):
                continue
            row = dict(raw_row)
            row["time_sync_ok"] = effective_time_sync_ok
            if effective_time_sync_ok:
                row["cvd_context_note"] = strip_time_sync_risk(str(row.get("cvd_context_note", "")))
            rebased_rows.append(row)
        effective_selected_micro = rebased_rows
    symbol_rows, buckets, locality_counts, attack_side_counts = build_symbol_rows(
        effective_selected_micro,
        policy=policy,
    )
    time_sync_environment_status = str(time_sync_environment_payload.get("status", "")).strip()
    time_sync_environment_classification = str(
        time_sync_environment_payload.get("classification", "")
    ).strip()
    time_sync_environment_blocker_detail = str(
        time_sync_environment_payload.get("blocker_detail", "")
    ).strip()
    time_sync_environment_remediation_hint = str(
        time_sync_environment_payload.get("remediation_hint", "")
    ).strip()
    time_sync_blocker_detail = ""
    if time_sync_classification == "fake_ip_dns_intercept":
        blocker_parts: list[str] = []
        if time_sync_intercept_scope:
            blocker_parts.append(f"intercept_scope={time_sync_intercept_scope}")
        blocker_parts.append(
            f"fake_ip_sources={','.join(time_sync_fake_ip_sources)}"
            if time_sync_fake_ip_sources
            else "fake_ip_sources=unknown"
        )
        time_sync_blocker_detail = "; ".join(blocker_parts)
    elif time_sync_classification == "threshold_breach":
        blocker_parts: list[str] = []
        if time_sync_threshold_breach_scope:
            blocker_parts.append(f"scope={time_sync_threshold_breach_scope}")
        if isinstance(time_sync_threshold_breach_estimated_offset_ms, (int, float)):
            blocker_parts.append(f"est_offset_ms={float(time_sync_threshold_breach_estimated_offset_ms):.3f}")
        if isinstance(time_sync_threshold_breach_estimated_rtt_ms, (int, float)):
            blocker_parts.append(f"est_rtt_ms={float(time_sync_threshold_breach_estimated_rtt_ms):.3f}")
        blocker_parts.extend(time_sync_threshold_breach_sources)
        time_sync_blocker_detail = "; ".join(blocker_parts) if blocker_parts else "threshold_breach_sources=unknown"
    elif time_sync_status and not bool(system_time_sync.get("pass", True)):
        ok_sources = int(system_time_sync.get("ok_sources", 0) or 0)
        available_sources = int(system_time_sync.get("available_sources", 0) or 0)
        time_sync_blocker_detail = f"ok_sources={ok_sources}/{available_sources}"
    if time_sync_environment_classification and time_sync_environment_blocker_detail:
        env_detail = f"env={time_sync_environment_classification}:{time_sync_environment_blocker_detail}"
        time_sync_blocker_detail = (
            f"{time_sync_blocker_detail} | {env_detail}" if time_sync_blocker_detail else env_detail
        )
    if time_sync_environment_remediation_hint:
        time_sync_remediation_hint = (
            f"{time_sync_remediation_hint} | env: {time_sync_environment_remediation_hint}"
            if time_sync_remediation_hint
            else time_sync_environment_remediation_hint
        )

    watch_only_symbols = buckets.get("watch_only", [])
    if not ok:
        if environment_status == "environment_blocked":
            takeaway = (
                "No usable micro-capture snapshot was available because the current public market data path is environment-blocked."
            )
        else:
            takeaway = "No usable micro-capture snapshot was available."
    elif len(watch_only_symbols) == len(symbol_rows):
        takeaway = "All current CVD-lite observations are downgraded to watch-only; keep them as review filters until micro quality recovers."
    elif buckets.get("trend_confirmation_watch"):
        takeaway = "Use current CVD-lite as trend confirmation for the symbols listed under trend-confirmation-watch."
    elif buckets.get("reversal_absorption_watch"):
        takeaway = "Current CVD-lite is more useful as reversal/absorption watch than continuation confirmation."
    else:
        takeaway = "Current CVD-lite snapshot is mixed and should remain advisory."

    out: dict[str, Any] = {
        "action": "build_crypto_cvd_semantic_snapshot",
        "ok": ok,
        "status": status,
        "generated_at": fmt_utc(now_ts),
        "source_artifact": None if source_path is None else str(source_path),
        "time_sync_probe_artifact": None if time_sync_probe_path is None else str(time_sync_probe_path),
        "time_sync_probe_override_active": bool(time_sync_probe_payload and probe_newer_than_capture),
        "time_sync_environment_artifact": None
        if time_sync_environment_path is None
        else str(time_sync_environment_path),
        "as_of": (
            source_payload.get("captured_at_utc")
            or source_payload.get("as_of")
            if isinstance(source_payload, dict)
            else None
        ),
        "source_status": source_payload.get("status") if isinstance(source_payload, dict) else None,
        "source_pass": bool(source_payload.get("pass", False)) if isinstance(source_payload, dict) else False,
        "environment_status": environment_status,
        "environment_classification": environment_classification,
        "environment_blocker_detail": environment_blocker_detail,
        "environment_remediation_hint": environment_remediation_hint,
        "time_sync_status": time_sync_status,
        "time_sync_classification": time_sync_classification,
        "time_sync_intercept_scope": time_sync_intercept_scope,
        "time_sync_blocker_detail": time_sync_blocker_detail,
        "time_sync_remediation_hint": time_sync_remediation_hint,
        "time_sync_environment_status": time_sync_environment_status,
        "time_sync_environment_classification": time_sync_environment_classification,
        "time_sync_environment_blocker_detail": time_sync_environment_blocker_detail,
        "time_sync_environment_remediation_hint": time_sync_environment_remediation_hint,
        "time_sync_fake_ip_sources": time_sync_fake_ip_sources,
        "time_sync_threshold_breach_sources": time_sync_threshold_breach_sources,
        "time_sync_threshold_breach_scope": time_sync_threshold_breach_scope,
        "time_sync_threshold_breach_offset_sources": time_sync_threshold_breach_offset_sources,
        "time_sync_threshold_breach_latency_sources": time_sync_threshold_breach_latency_sources,
        "time_sync_threshold_breach_estimated_offset_ms": time_sync_threshold_breach_estimated_offset_ms,
        "time_sync_threshold_breach_estimated_rtt_ms": time_sync_threshold_breach_estimated_rtt_ms,
        "symbols_requested": int(source_payload.get("symbols_requested", 0) or 0) if isinstance(source_payload, dict) else 0,
        "symbols_selected": int(source_payload.get("symbols_selected", 0) or 0) if isinstance(source_payload, dict) else 0,
        "context_counts": context_counts if isinstance(context_counts, dict) else {},
        "trust_counts": trust_counts if isinstance(trust_counts, dict) else {},
        "veto_hint_counts": veto_hint_counts if isinstance(veto_hint_counts, dict) else {},
        "locality_counts": locality_counts,
        "attack_side_counts": attack_side_counts,
        "focus_stack_brief": "trend_confirmation -> reversal_absorption -> watch_only",
        "trend_confirmation_watch": buckets.get("trend_confirmation_watch", []),
        "reversal_absorption_watch": buckets.get("reversal_absorption_watch", []),
        "watch_only_symbols": watch_only_symbols,
        "unclear_symbols": buckets.get("unclear", []),
        "symbols": symbol_rows,
        "takeaway": takeaway,
        "artifact_status_label": "cvd-semantic-snapshot-ok" if ok else status,
        "artifact_label": None,
        "artifact_tags": [],
        "artifact": None,
        "markdown": None,
        "checksum": None,
        "pruned_keep": [],
        "pruned_age": [],
    }

    suffix = str(out.get("source_status") or status)
    out["artifact_label"] = f"crypto-cvd-semantic-snapshot:{suffix}"
    out["artifact_tags"] = ["crypto-cvd", "micro-capture", str(suffix)]

    review_dir.mkdir(parents=True, exist_ok=True)
    stamp = now_ts.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_path = review_dir / f"{stamp}_crypto_cvd_semantic_snapshot.json"
    markdown_path = review_dir / f"{stamp}_crypto_cvd_semantic_snapshot.md"
    checksum_path = review_dir / f"{stamp}_crypto_cvd_semantic_snapshot_checksum.json"
    out["artifact"] = str(artifact_path)
    out["markdown"] = str(markdown_path)
    out["checksum"] = str(checksum_path)

    artifact_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(out), encoding="utf-8")
    checksum_path.write_text(
        json.dumps(
            {
                "generated_at": fmt_utc(now_ts),
                "artifact_ttl_hours": max(1.0, float(args.artifact_ttl_hours)),
                "files": [
                    {
                        "path": str(artifact_path),
                        "sha256": sha256_file(artifact_path),
                        "size_bytes": int(artifact_path.stat().st_size),
                    },
                    {
                        "path": str(markdown_path),
                        "sha256": sha256_file(markdown_path),
                        "size_bytes": int(markdown_path.stat().st_size),
                    },
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    pruned_keep, pruned_age = prune_artifacts(
        review_dir,
        current_artifact=artifact_path,
        current_checksum=checksum_path,
        current_markdown=markdown_path,
        keep=max(1, int(args.artifact_keep)),
        ttl_hours=max(1.0, float(args.artifact_ttl_hours)),
    )
    out["pruned_keep"] = pruned_keep
    out["pruned_age"] = pruned_age
    artifact_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
