#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import ipaddress
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = SYSTEM_ROOT / "output"
DEFAULT_REVIEW_DIR = DEFAULT_OUTPUT_ROOT / "review"
DEFAULT_TIME_SYNC_DIR = DEFAULT_OUTPUT_ROOT / "artifacts" / "time_sync"
DEFAULT_CLASH_CONFIG = Path(
    "/Users/jokenrobot/Library/Application Support/io.github.clash-verge-rev.clash-verge-rev/clash-verge.yaml"
)
DEFAULT_CLASH_MERGE = Path(
    "/Users/jokenrobot/Library/Application Support/io.github.clash-verge-rev.clash-verge-rev/profiles/Merge.yaml"
)
TIMESTAMP_RE = re.compile(r"^(?P<stamp>\d{8}T\d{6}Z)_")
FAKE_IP_NET = ipaddress.ip_network("198.18.0.0/15")
SRC_RE = re.compile(r'Received time .* from "([^"]+)"')
LOG_TS_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})")
MAX_REASONABLE_NTP_DELAY_MS = 60_000.0


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse_now(raw: str | None) -> dt.datetime:
    text = str(raw or "").strip()
    if not text:
        return now_utc()
    parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def fmt_utc(value: dt.datetime | None) -> str:
    effective = value or now_utc()
    return effective.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def parsed_artifact_stamp(path: Path) -> dt.datetime | None:
    match = TIMESTAMP_RE.match(path.name)
    if not match:
        return None
    return dt.datetime.strptime(match.group("stamp"), "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)


def artifact_sort_key(path: Path, reference_now: dt.datetime) -> tuple[int, str, float, str]:
    stamp_dt = parsed_artifact_stamp(path)
    future_cutoff = reference_now + dt.timedelta(minutes=5)
    is_future = bool(stamp_dt and stamp_dt > future_cutoff)
    return (0 if is_future else 1, match_stamp(path), path.stat().st_mtime, path.name)


def match_stamp(path: Path) -> str:
    match = TIMESTAMP_RE.match(path.name)
    return match.group("stamp") if match else ""


def latest_time_sync_probe(artifact_dir: Path, reference_now: dt.datetime) -> Path | None:
    candidates = sorted(
        artifact_dir.glob("*_time_sync_probe.json"),
        key=lambda item: artifact_sort_key(item, reference_now),
        reverse=True,
    )
    for path in candidates:
        stamp_dt = parsed_artifact_stamp(path)
        if stamp_dt is not None and stamp_dt <= reference_now + dt.timedelta(minutes=5):
            return path
    return None


def load_json_mapping(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def load_yaml_mapping(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prune_artifacts(
    review_dir: Path,
    *,
    current_paths: list[Path],
    keep: int,
    ttl_hours: float,
) -> tuple[list[str], list[str]]:
    cutoff = now_utc() - dt.timedelta(hours=max(1.0, float(ttl_hours)))
    protected = {path.name for path in current_paths}
    candidates: list[Path] = []
    for pattern in (
        "*_system_time_sync_environment_report.json",
        "*_system_time_sync_environment_report.md",
        "*_system_time_sync_environment_report_checksum.json",
    ):
        candidates.extend(review_dir.glob(pattern))
    pruned_age: list[str] = []
    survivors: list[Path] = []
    existing_candidates: list[tuple[float, Path]] = []
    for path in candidates:
        try:
            existing_candidates.append((path.stat().st_mtime, path))
        except OSError:
            continue
    for _, path in sorted(existing_candidates, key=lambda item: item[0], reverse=True):
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
    for path in survivors[max(1, int(keep)) :]:
        if path.name in protected:
            continue
        path.unlink(missing_ok=True)
        pruned_keep.append(str(path))
    return pruned_keep, pruned_age


def is_fake_ip(addr: str) -> bool:
    try:
        return ipaddress.ip_address(str(addr).strip()) in FAKE_IP_NET
    except ValueError:
        return False


def fake_ip_filter_matches_host(host: str, rules: list[str]) -> bool:
    target = str(host or "").strip().lower()
    if not target:
        return False
    for raw_rule in list(rules or []):
        rule = str(raw_rule or "").strip().lower()
        if not rule:
            continue
        if rule.startswith("+."):
            suffix = rule[2:]
            if target == suffix or target.endswith(f".{suffix}"):
                return True
        elif target == rule:
            return True
    return False


def read_timed_log_text(*, lookback_minutes: int, timed_log_file: Path | None) -> str:
    if timed_log_file and timed_log_file.exists():
        return timed_log_file.read_text(encoding="utf-8")
    cmd = [
        "log",
        "show",
        "--style",
        "compact",
        "--last",
        f"{int(max(1, lookback_minutes))}m",
        "--predicate",
        'process == "timed"',
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    return str(proc.stdout or "")


def parse_log_line_ts(line: str) -> dt.datetime | None:
    match = LOG_TS_RE.match(str(line or ""))
    if not match:
        return None
    try:
        parsed = dt.datetime.strptime(match.group("ts"), "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        return None
    return parsed.replace(tzinfo=dt.datetime.now().astimezone().tzinfo or dt.timezone.utc)


def parse_fetch_time_row(line: str, line_ts: dt.datetime | None) -> dict[str, Any] | None:
    marker = "cmd,fetchTime,"
    if marker not in line:
        return None
    suffix = line.split(marker, 1)[1]
    tokens = [segment.strip() for segment in suffix.split(",")]
    if len(tokens) < 2:
        return None
    fields: dict[str, str] = {}
    index = 0
    while index + 1 < len(tokens):
        key = tokens[index]
        value = tokens[index + 1]
        if key:
            fields[key] = value
        index += 2
    ip_text = str(fields.get("ip") or "").strip()
    if not ip_text:
        return None
    delay_ms: float | None = None
    delay_unreliable = False
    delay_text = str(fields.get("delay") or "").strip()
    result_text = str(fields.get("result") or "").strip()
    if delay_text:
        try:
            parsed_delay_ms = round(float(delay_text) * 1000.0, 3)
        except ValueError:
            parsed_delay_ms = None
        if parsed_delay_ms is not None:
            if result_text not in {"", "0"} or parsed_delay_ms < 0 or parsed_delay_ms > MAX_REASONABLE_NTP_DELAY_MS:
                delay_unreliable = True
            else:
                delay_ms = parsed_delay_ms
    return {
        "ip": ip_text,
        "delay_ms": delay_ms,
        "delay_unreliable": delay_unreliable,
        "fake_ip_detected": is_fake_ip(ip_text),
        "result": result_text,
        "ts": line_ts.isoformat() if line_ts else "",
    }


def parse_timed_log(log_text: str) -> dict[str, Any]:
    source_events: list[dict[str, Any]] = []
    recent_fetch_rows: list[dict[str, Any]] = []
    for line in str(log_text or "").splitlines():
        line_ts = parse_log_line_ts(line)
        src_match = SRC_RE.search(line)
        if src_match:
            source = str(src_match.group(1)).strip()
            if source:
                source_events.append({"source": source, "ts": line_ts.isoformat() if line_ts else ""})
        fetch_row = parse_fetch_time_row(line, line_ts)
        if fetch_row:
            recent_fetch_rows.append(fetch_row)
    recent_fake_ip_rows = [row for row in recent_fetch_rows if bool(row.get("fake_ip_detected"))]
    recent_ntp_fake_ip = recent_fake_ip_rows[-1] if recent_fake_ip_rows else {}
    recent_sources: list[str] = []
    recent_source_sequence: list[str] = []
    for event in source_events:
        source = str(event.get("source") or "").strip()
        if source and source not in recent_sources:
            recent_sources.append(source)
        if source and (not recent_source_sequence or recent_source_sequence[-1] != source):
            recent_source_sequence.append(source)
    recent_source = str(source_events[-1].get("source") or "").strip() if source_events else ""
    last_ntp_event_ts: dt.datetime | None = None
    recent_apns_after_ntp_ts: dt.datetime | None = None
    for event in reversed(source_events):
        source = str(event.get("source") or "").strip()
        event_ts_text = str(event.get("ts") or "").strip()
        try:
            event_ts = dt.datetime.fromisoformat(event_ts_text) if event_ts_text else None
        except ValueError:
            event_ts = None
        if source == "NTP" and event_ts is not None:
            last_ntp_event_ts = event_ts
            break
    if last_ntp_event_ts is not None:
        for event in source_events:
            source = str(event.get("source") or "").strip()
            event_ts_text = str(event.get("ts") or "").strip()
            try:
                event_ts = dt.datetime.fromisoformat(event_ts_text) if event_ts_text else None
            except ValueError:
                event_ts = None
            if source == "APNS" and event_ts is not None and event_ts > last_ntp_event_ts:
                recent_apns_after_ntp_ts = event_ts
                break
    recent_ntp_fallback_seconds: float | None = None
    if last_ntp_event_ts is not None and recent_apns_after_ntp_ts is not None:
        recent_ntp_fallback_seconds = round(
            (recent_apns_after_ntp_ts - last_ntp_event_ts).total_seconds(),
            3,
        )
    return {
        "recent_sources": recent_sources,
        "recent_source_sequence": recent_source_sequence,
        "recent_source_transition_count": max(0, len(recent_source_sequence) - 1),
        "recent_source": recent_source,
        "recent_source_ts": str(source_events[-1].get("ts") or "") if source_events else "",
        "recent_ntp_event_ts": last_ntp_event_ts.isoformat() if last_ntp_event_ts else "",
        "recent_apns_after_ntp_ts": recent_apns_after_ntp_ts.isoformat() if recent_apns_after_ntp_ts else "",
        "recent_ntp_fallback_seconds": recent_ntp_fallback_seconds,
        "recent_fetch_rows": recent_fetch_rows[-8:],
        "recent_fake_ip_fetch_rows": recent_fake_ip_rows[-8:],
        "recent_ntp_fake_ip": recent_ntp_fake_ip,
    }


def parse_row_ts(value: Any) -> dt.datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return dt.datetime.fromisoformat(text)
    except ValueError:
        return None


def build_clash_summary(config_path: Path | None, merge_path: Path | None) -> dict[str, Any]:
    active = load_yaml_mapping(config_path)
    merge = load_yaml_mapping(merge_path)
    active_dns = dict(active.get("dns") or {}) if isinstance(active, dict) else {}
    active_tun = dict(active.get("tun") or {}) if isinstance(active, dict) else {}
    merge_dns = dict(merge.get("dns") or {}) if isinstance(merge, dict) else {}
    merge_tun = dict(merge.get("tun") or {}) if isinstance(merge, dict) else {}
    active_rules = [
        str(item).strip()
        for item in list(active.get("rules") or [])
        if str(item).strip()
    ] if isinstance(active, dict) else []
    merge_rules = [
        str(item).strip()
        for item in list(merge.get("rules") or [])
        if str(item).strip()
    ] if isinstance(merge, dict) else []
    rules = active_rules or merge_rules
    fake_ip_filter = [
        str(x).strip()
        for x in list(active_dns.get("fake-ip-filter") or merge_dns.get("fake-ip-filter") or [])
        if str(x).strip()
    ]
    rules_upper = [rule.upper() for rule in rules]
    return {
        "active_config_path": str(config_path) if config_path else "",
        "merge_config_path": str(merge_path) if merge_path else "",
        "dns_enhanced_mode": str(active_dns.get("enhanced-mode") or merge_dns.get("enhanced-mode") or "").strip(),
        "tun_enable": bool(active_tun.get("enable", merge_tun.get("enable", False))),
        "tun_stack": str(active_tun.get("stack") or merge_tun.get("stack") or "").strip(),
        "tun_auto_route": bool(active_tun.get("auto-route", merge_tun.get("auto-route", False))),
        "fake_ip_filter": fake_ip_filter,
        "timed_direct_rule_present": any(rule == "PROCESS-NAME,TIMED,DIRECT" for rule in rules_upper),
        "sntp_direct_rule_present": any(rule == "PROCESS-NAME,SNTP,DIRECT" for rule in rules_upper),
    }


def classify_environment(
    *,
    probe_payload: dict[str, Any],
    timed_summary: dict[str, Any],
    clash_summary: dict[str, Any],
) -> dict[str, str]:
    dns_rows = list(probe_payload.get("dns_diagnostics") or [])
    fake_ip_hosts = [
        str(row.get("source") or "").strip()
        for row in dns_rows
        if isinstance(row, dict) and bool(row.get("fake_ip_detected")) and str(row.get("source") or "").strip()
    ]
    recent_source = str(timed_summary.get("recent_source") or "").strip()
    recent_ntp_fake_ip = dict(timed_summary.get("recent_ntp_fake_ip") or {})
    fake_ip_ip = str(recent_ntp_fake_ip.get("ip") or "").strip()
    fake_ip_delay_ms = recent_ntp_fake_ip.get("delay_ms")
    fake_ip_delay_unreliable = bool(recent_ntp_fake_ip.get("delay_unreliable"))
    fake_ip_result = str(recent_ntp_fake_ip.get("result") or "").strip()
    dns_mode = str(clash_summary.get("dns_enhanced_mode") or "").strip()
    tun_stack = str(clash_summary.get("tun_stack") or "").strip()
    probe_classification = str(probe_payload.get("classification") or "").strip()
    probe_available_sources = int(probe_payload.get("available_sources") or 0)
    probe_scope = str(probe_payload.get("threshold_breach_scope") or "").strip()
    probe_offset_ms = probe_payload.get("threshold_breach_estimated_offset_ms")
    recent_sources = [str(item).strip() for item in list(timed_summary.get("recent_sources") or []) if str(item).strip()]
    recent_source_sequence = [
        str(item).strip() for item in list(timed_summary.get("recent_source_sequence") or []) if str(item).strip()
    ]
    recent_ntp_success_detected = recent_source == "APNS" and "NTP" in recent_sources
    recent_ntp_fallback_seconds = timed_summary.get("recent_ntp_fallback_seconds")
    recent_fetch_rows = [row for row in list(timed_summary.get("recent_fetch_rows") or []) if isinstance(row, dict)]
    recent_direct_ntp_rows = [
        row
        for row in recent_fetch_rows
        if str(row.get("result") or "").strip() == "0" and not bool(row.get("fake_ip_detected"))
    ]
    recent_direct_ntp_ips: list[str] = []
    recent_direct_ntp_delay_values: list[float] = []
    for row in recent_direct_ntp_rows:
        ip_text = str(row.get("ip") or "").strip()
        if ip_text and ip_text not in recent_direct_ntp_ips:
            recent_direct_ntp_ips.append(ip_text)
        delay_value = row.get("delay_ms")
        if isinstance(delay_value, (int, float)):
            recent_direct_ntp_delay_values.append(float(delay_value))
    latest_success_row: dict[str, Any] = {}
    for row in reversed(recent_fetch_rows):
        if str(row.get("result") or "").strip() == "0":
            latest_success_row = row
            break
    latest_success_ip = str(latest_success_row.get("ip") or "").strip()
    latest_success_is_direct = bool(latest_success_row) and not bool(latest_success_row.get("fake_ip_detected"))
    latest_success_delay_ms = latest_success_row.get("delay_ms")
    latest_fake_ip_row: dict[str, Any] = {}
    for row in reversed(recent_fetch_rows):
        if str(row.get("result") or "").strip() == "0" and bool(row.get("fake_ip_detected")):
            latest_fake_ip_row = row
            break
    latest_fake_ip_ts = parse_row_ts(latest_fake_ip_row.get("ts"))
    latest_direct_row: dict[str, Any] = {}
    for row in reversed(recent_fetch_rows):
        if str(row.get("result") or "").strip() == "0" and not bool(row.get("fake_ip_detected")):
            latest_direct_row = row
            break
    latest_direct_ts = parse_row_ts(latest_direct_row.get("ts"))
    direct_ntp_recovered = bool(latest_success_row) and latest_success_is_direct and (
        latest_fake_ip_ts is None or (latest_direct_ts is not None and latest_direct_ts >= latest_fake_ip_ts)
    )
    fake_ip_filter_rules = [str(item).strip() for item in list(clash_summary.get("fake_ip_filter") or []) if str(item).strip()]
    timed_direct_rule_present = bool(clash_summary.get("timed_direct_rule_present"))
    sntp_direct_rule_present = bool(clash_summary.get("sntp_direct_rule_present"))
    probe_sources = [
        str(row.get("source") or "").strip()
        for row in dns_rows
        if isinstance(row, dict) and str(row.get("source") or "").strip()
    ]
    covered_probe_sources = [
        source for source in probe_sources if fake_ip_filter_matches_host(source, fake_ip_filter_rules)
    ]
    clash_time_exemptions_present = bool(covered_probe_sources) and timed_direct_rule_present and sntp_direct_rule_present
    residual_path_hint = (
        "fake_ip_filter_present_stale_cache_or_tun_intercept" if fake_ip_ip and covered_probe_sources else ""
    )

    classification = "none"
    blocker_detail = ""
    remediation_hint = ""
    if recent_source == "NTP" and direct_ntp_recovered:
        classification = "none"
        blocker_detail = ""
        remediation_hint = ""
        residual_path_hint = ""
    elif recent_source == "NTP" and fake_ip_ip:
        classification = "timed_ntp_via_fake_ip"
        parts = [
            f"timed_source={recent_source}",
            f"ntp_ip={fake_ip_ip}",
        ]
        if isinstance(fake_ip_delay_ms, (int, float)):
            parts.append(f"delay_ms={float(fake_ip_delay_ms):.3f}")
        if fake_ip_hosts:
            parts.append(f"fake_ip_hosts={','.join(fake_ip_hosts)}")
        if dns_mode:
            parts.append(f"clash_dns_mode={dns_mode}")
        if tun_stack:
            parts.append(f"tun_stack={tun_stack}")
        blocker_detail = "; ".join(parts)
        remediation_hint = (
            "exclude macOS timed / UDP 123 from Clash TUN fake-ip handling or provide a direct NTP path, "
            "then rerun time_sync_probe"
        )
    elif recent_source == "APNS":
        parts = ["timed_source=APNS", "os_time_service is not using NTP as the active source"]
        if not fake_ip_hosts:
            parts.append("dns_ntp_hosts_resolve_cleanly")
        if fake_ip_ip:
            parts.append(f"recent_fake_ip_attempt={fake_ip_ip}")
            if fake_ip_result:
                parts.append(f"recent_fake_ip_result={fake_ip_result}")
            if isinstance(fake_ip_delay_ms, (int, float)):
                parts.append(f"recent_fake_ip_delay_ms={float(fake_ip_delay_ms):.3f}")
            elif fake_ip_delay_unreliable:
                parts.append("recent_fake_ip_delay_ms=unreliable")
            if residual_path_hint:
                parts.append(f"residual_path_hint={residual_path_hint}")
        if probe_classification == "threshold_breach" and probe_available_sources > 0:
            classification = (
                "timed_apns_fallback_ntp_reachable_fake_ip_residual"
                if fake_ip_ip
                else (
                    "timed_apns_fallback_ntp_reachable_flapping"
                    if recent_ntp_success_detected
                    else "timed_apns_fallback_ntp_reachable"
                )
            )
            parts.append(f"probe_scope={probe_scope or '-'}")
            if isinstance(probe_offset_ms, (int, float)):
                parts.append(f"probe_offset_ms={float(probe_offset_ms):.3f}")
            if recent_ntp_success_detected:
                parts.append("recent_ntp_success_observed")
                if recent_source_sequence:
                    parts.append(f"recent_source_sequence={'>'.join(recent_source_sequence)}")
                if isinstance(recent_ntp_fallback_seconds, (int, float)):
                    parts.append(f"recent_ntp_fallback_seconds={float(recent_ntp_fallback_seconds):.3f}")
                if recent_direct_ntp_ips:
                    parts.append("recent_ntp_direct_hits_present")
                    parts.append(f"recent_ntp_ips={','.join(recent_direct_ntp_ips)}")
                if recent_direct_ntp_delay_values:
                    parts.append(f"recent_ntp_delay_ms_min={min(recent_direct_ntp_delay_values):.3f}")
                    parts.append(f"recent_ntp_delay_ms_max={max(recent_direct_ntp_delay_values):.3f}")
            if clash_time_exemptions_present and not fake_ip_ip:
                residual_path_hint = "clash_exemptions_present_os_time_service_not_recovered"
                parts.append("clash_time_exemptions_present")
                parts.append(f"residual_path_hint={residual_path_hint}")
            blocker_detail = "; ".join(parts)
            if fake_ip_ip:
                remediation_hint = (
                    "direct NTP sources are reachable, but macOS timed is still using APNS fallback and recently attempted a fake-IP/proxy path; "
                    "re-enable network time, restart Clash/TUN or clear stale fake-IP state if filter rules are already present, clear proxy handling for timed / UDP 123 if needed, force a one-time admin resync, and rerun time_sync_probe"
                )
            elif clash_time_exemptions_present:
                remediation_hint = (
                    "direct NTP sources are reachable and Clash already exempts timed/sntp plus the probe hosts; "
                    "prioritize re-enabling macOS network time, toggling automatic time off/on, forcing a one-time admin resync, and rebooting/restarting the OS time service path before editing Clash rules again, then rerun time_sync_probe"
                )
                if recent_ntp_success_detected:
                    prefix = "timed recently reached NTP but fell back to APNS again"
                    if isinstance(recent_ntp_fallback_seconds, (int, float)):
                        prefix += f" after about {float(recent_ntp_fallback_seconds):.3f}s"
                    if recent_direct_ntp_ips:
                        prefix += f"; direct NTP fetches from timed already succeeded to {','.join(recent_direct_ntp_ips)}"
                        if recent_direct_ntp_delay_values:
                            prefix += (
                                f" with recent delay range {min(recent_direct_ntp_delay_values):.3f}-{max(recent_direct_ntp_delay_values):.3f}ms"
                            )
                    remediation_hint = remediation_hint.replace(
                        "prioritize",
                        f"{prefix}; prioritize",
                        1,
                    )
            else:
                remediation_hint = (
                    "direct NTP sources are reachable, but macOS timed is still using APNS fallback; "
                    "re-enable network time, force a one-time admin resync if needed, clear any stale fake-IP/proxy path for timed if recent fake-IP attempts persist, and rerun time_sync_probe"
                )
        else:
            classification = "timed_apns_fallback"
            blocker_detail = "; ".join(parts)
            remediation_hint = (
                "manual clock resync or administrator-level network time repair may be required; "
                "restore a direct NTP path for macOS timed, clear any stale fake-IP/proxy path if recent fake-IP attempts persist, before relying on APNS fallback, then rerun time_sync_probe"
            )
    elif fake_ip_hosts:
        classification = "dns_fake_ip_residual"
        blocker_detail = f"fake_ip_hosts={','.join(fake_ip_hosts)}"
        remediation_hint = (
            "remove fake-ip DNS interception for remaining NTP hosts or ensure those hosts are excluded from proxy DNS, "
            "then rerun time_sync_probe"
        )
    return {
        "classification": classification,
        "blocker_detail": blocker_detail,
        "remediation_hint": remediation_hint,
        "covered_probe_sources": covered_probe_sources,
        "residual_path_hint": residual_path_hint,
        "recent_ntp_success_detected": "true" if recent_ntp_success_detected else "",
        "recent_direct_ntp_ips": recent_direct_ntp_ips,
        "recent_direct_ntp_delay_ms_min": (
            round(min(recent_direct_ntp_delay_values), 3) if recent_direct_ntp_delay_values else ""
        ),
        "recent_direct_ntp_delay_ms_max": (
            round(max(recent_direct_ntp_delay_values), 3) if recent_direct_ntp_delay_values else ""
        ),
        "latest_success_ip": latest_success_ip,
        "latest_success_is_direct": "true" if latest_success_is_direct else "",
        "latest_success_delay_ms": round(float(latest_success_delay_ms), 3)
        if isinstance(latest_success_delay_ms, (int, float))
        else "",
        "direct_ntp_recovered": "true" if direct_ntp_recovered else "",
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# System Time Sync Environment Report",
        "",
        f"- status: `{payload.get('status') or ''}`",
        f"- classification: `{payload.get('classification') or '-'}`",
        f"- blocker_detail: `{payload.get('blocker_detail') or '-'}`",
        f"- remediation_hint: `{payload.get('remediation_hint') or '-'}`",
        f"- time_sync_probe_artifact: `{payload.get('time_sync_probe_artifact') or '-'}`",
        "",
        "## Proxy",
        f"- http_proxy: `{payload.get('http_proxy') or '-'}`",
        f"- https_proxy: `{payload.get('https_proxy') or '-'}`",
        "",
        "## Clash",
        f"- dns_enhanced_mode: `{payload.get('clash_dns_enhanced_mode') or '-'}`",
        f"- tun_enable: `{payload.get('clash_tun_enable')}`",
        f"- tun_stack: `{payload.get('clash_tun_stack') or '-'}`",
        f"- tun_auto_route: `{payload.get('clash_tun_auto_route')}`",
        f"- fake_ip_filter: `{','.join(payload.get('clash_fake_ip_filter') or []) or '-'}`",
        "",
        "## Timed",
        f"- recent_source: `{payload.get('timed_recent_source') or '-'}`",
        f"- recent_ntp_fake_ip: `{payload.get('timed_recent_ntp_fake_ip') or '-'}`",
        f"- recent_ntp_fake_ip_delay_ms: `{payload.get('timed_recent_ntp_fake_ip_delay_ms') or '-'}`",
        f"- recent_sources: `{','.join(payload.get('timed_recent_sources') or []) or '-'}`",
        "",
        "## DNS Diagnostics",
    ]
    for row in payload.get("dns_diagnostics") or []:
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- `{row.get('source')}` -> `{','.join(row.get('resolved_addrs') or []) or '-'}` | fake_ip=`{row.get('fake_ip_detected')}`"
        )
    return "\n".join(lines).strip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a system time-sync environment report for crypto route diagnostics.")
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--artifact-dir", default=str(DEFAULT_TIME_SYNC_DIR))
    parser.add_argument("--time-sync-probe-file", default="")
    parser.add_argument("--timed-log-file", default="")
    parser.add_argument("--clash-config-file", default=str(DEFAULT_CLASH_CONFIG))
    parser.add_argument("--clash-merge-file", default=str(DEFAULT_CLASH_MERGE))
    parser.add_argument("--lookback-minutes", type=int, default=20)
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--artifact-keep", type=int, default=16)
    parser.add_argument("--now", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    now_ts = parse_now(args.now)
    review_dir = Path(args.review_dir).expanduser().resolve()
    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    probe_path = (
        Path(args.time_sync_probe_file).expanduser().resolve()
        if str(args.time_sync_probe_file).strip()
        else latest_time_sync_probe(artifact_dir, now_ts)
    )
    timed_log_file = Path(args.timed_log_file).expanduser().resolve() if str(args.timed_log_file).strip() else None
    clash_config_path = (
        Path(args.clash_config_file).expanduser().resolve() if str(args.clash_config_file).strip() else None
    )
    clash_merge_path = (
        Path(args.clash_merge_file).expanduser().resolve() if str(args.clash_merge_file).strip() else None
    )

    probe_payload = load_json_mapping(probe_path)
    timed_summary = parse_timed_log(read_timed_log_text(lookback_minutes=args.lookback_minutes, timed_log_file=timed_log_file))
    clash_summary = build_clash_summary(clash_config_path, clash_merge_path)
    classification = classify_environment(
        probe_payload=probe_payload,
        timed_summary=timed_summary,
        clash_summary=clash_summary,
    )

    stamp = now_ts.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    review_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = review_dir / f"{stamp}_system_time_sync_environment_report.json"
    markdown_path = review_dir / f"{stamp}_system_time_sync_environment_report.md"
    checksum_path = review_dir / f"{stamp}_system_time_sync_environment_report_checksum.json"

    recent_ntp_fake_ip = dict(timed_summary.get("recent_ntp_fake_ip") or {})
    payload: dict[str, Any] = {
        "action": "build_system_time_sync_environment_report",
        "ok": True,
        "status": "ok",
        "generated_at": fmt_utc(now_ts),
        "time_sync_probe_artifact": str(probe_path) if probe_path else "",
        "time_sync_probe_as_of": str(probe_payload.get("as_of") or ""),
        "classification": classification["classification"],
        "blocker_detail": classification["blocker_detail"],
        "remediation_hint": classification["remediation_hint"],
        "http_proxy": str(os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy") or "").strip(),
        "https_proxy": str(os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or "").strip(),
        "dns_diagnostics": list(probe_payload.get("dns_diagnostics") or []),
        "timed_recent_source": str(timed_summary.get("recent_source") or ""),
        "timed_recent_sources": list(timed_summary.get("recent_sources") or []),
        "timed_recent_source_sequence": list(timed_summary.get("recent_source_sequence") or []),
        "timed_recent_source_transition_count": timed_summary.get("recent_source_transition_count"),
        "timed_recent_ntp_success_detected": bool(classification.get("recent_ntp_success_detected")),
        "timed_recent_ntp_event_ts": str(timed_summary.get("recent_ntp_event_ts") or ""),
        "timed_recent_apns_after_ntp_ts": str(timed_summary.get("recent_apns_after_ntp_ts") or ""),
        "timed_recent_ntp_fallback_seconds": timed_summary.get("recent_ntp_fallback_seconds"),
        "timed_recent_fetch_rows": list(timed_summary.get("recent_fetch_rows") or []),
        "timed_recent_fake_ip_fetch_rows": list(timed_summary.get("recent_fake_ip_fetch_rows") or []),
        "timed_recent_ntp_fake_ip": str(recent_ntp_fake_ip.get("ip") or ""),
        "timed_recent_ntp_fake_ip_delay_ms": recent_ntp_fake_ip.get("delay_ms"),
        "timed_recent_direct_ntp_ips": list(classification.get("recent_direct_ntp_ips") or []),
        "timed_recent_direct_ntp_delay_ms_min": classification.get("recent_direct_ntp_delay_ms_min") or "",
        "timed_recent_direct_ntp_delay_ms_max": classification.get("recent_direct_ntp_delay_ms_max") or "",
        "latest_success_ip": str(classification.get("latest_success_ip") or ""),
        "latest_success_is_direct": bool(classification.get("latest_success_is_direct")),
        "latest_success_delay_ms": classification.get("latest_success_delay_ms") or "",
        "direct_ntp_recovered": bool(classification.get("direct_ntp_recovered")),
        "clash_config_path": str(clash_summary.get("active_config_path") or ""),
        "clash_merge_path": str(clash_summary.get("merge_config_path") or ""),
        "clash_dns_enhanced_mode": str(clash_summary.get("dns_enhanced_mode") or ""),
        "clash_tun_enable": bool(clash_summary.get("tun_enable", False)),
        "clash_tun_stack": str(clash_summary.get("tun_stack") or ""),
        "clash_tun_auto_route": bool(clash_summary.get("tun_auto_route", False)),
        "clash_fake_ip_filter": list(clash_summary.get("fake_ip_filter") or []),
        "clash_timed_direct_rule_present": bool(clash_summary.get("timed_direct_rule_present")),
        "clash_sntp_direct_rule_present": bool(clash_summary.get("sntp_direct_rule_present")),
        "covered_probe_sources": list(classification.get("covered_probe_sources") or []),
        "residual_path_hint": str(classification.get("residual_path_hint") or ""),
        "artifact": str(artifact_path),
        "markdown": str(markdown_path),
        "checksum": str(checksum_path),
        "pruned_keep": [],
        "pruned_age": [],
    }

    artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
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
        current_paths=[artifact_path, markdown_path, checksum_path],
        keep=max(1, int(args.artifact_keep)),
        ttl_hours=max(1.0, float(args.artifact_ttl_hours)),
    )
    payload["pruned_keep"] = pruned_keep
    payload["pruned_age"] = pruned_age
    artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
