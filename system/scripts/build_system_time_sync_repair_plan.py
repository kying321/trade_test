#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = SYSTEM_ROOT / "output"
DEFAULT_REVIEW_DIR = DEFAULT_OUTPUT_ROOT / "review"
DEFAULT_TIME_SYNC_DIR = DEFAULT_OUTPUT_ROOT / "artifacts" / "time_sync"
TIMESTAMP_RE = re.compile(r"^(?P<stamp>\d{8}T\d{6}Z)_")
FUTURE_STAMP_GRACE_MINUTES = 5
APPLE_AUTO_TIME_URL = "https://support.apple.com/my-mm/guide/mac-help/mchlp2996/mac"
APPLE_TIME_SETTINGS_URL = "https://support.apple.com/en-lamr/guide/mac-help/mchlp1124/mac"
APPLE_TIME_TROUBLESHOOT_URL = "https://support.apple.com/en-mt/101619"
MIHOMO_DNS_DOC_URL = "https://wiki.metacubex.one/en/config/dns/"
CLASH_META_NTP_DOC_URL = "https://hokoory.github.io/clash-mate-doc/config/ntp/"


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


def artifact_stamp(path: Path) -> str:
    match = TIMESTAMP_RE.match(path.name)
    return match.group("stamp") if match else ""


def parsed_artifact_stamp(path: Path) -> dt.datetime | None:
    stamp = artifact_stamp(path)
    if not stamp:
        return None
    return dt.datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)


def artifact_sort_key(path: Path, reference_now: dt.datetime) -> tuple[int, str, float, str]:
    stamp_dt = parsed_artifact_stamp(path)
    future_cutoff = reference_now + dt.timedelta(minutes=FUTURE_STAMP_GRACE_MINUTES)
    is_future = bool(stamp_dt and stamp_dt > future_cutoff)
    return (0 if is_future else 1, artifact_stamp(path), path.stat().st_mtime, path.name)


def latest_review_json_artifact(review_dir: Path, suffix: str, reference_now: dt.datetime) -> Path | None:
    candidates = sorted(
        review_dir.glob(f"*_{suffix}.json"),
        key=lambda item: artifact_sort_key(item, reference_now),
        reverse=True,
    )
    return candidates[0] if candidates else None


def latest_time_sync_probe(artifact_dir: Path, reference_now: dt.datetime) -> Path | None:
    candidates = sorted(
        artifact_dir.glob("*_time_sync_probe.json"),
        key=lambda item: artifact_sort_key(item, reference_now),
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_json_mapping(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prune_review_artifacts(
    review_dir: Path,
    *,
    stem: str,
    current_paths: list[Path],
    keep: int,
    ttl_hours: float,
    now_dt: dt.datetime,
) -> tuple[list[str], list[str]]:
    cutoff = now_dt - dt.timedelta(hours=max(1.0, float(ttl_hours)))
    protected = {path.name for path in current_paths}
    candidates = sorted(review_dir.glob(f"*_{stem}*"), key=lambda item: item.stat().st_mtime, reverse=True)
    pruned_age: list[str] = []
    survivors: list[Path] = []
    for path in candidates:
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


def safe_text(value: Any) -> str:
    return str(value or "").strip()


def review_source_payload(cross_market_payload: dict[str, Any], hot_payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(cross_market_payload, dict) and safe_text(cross_market_payload.get("review_head_brief")):
        return {
            "symbol": safe_text(cross_market_payload.get("review_head_symbol")),
            "brief": safe_text(cross_market_payload.get("review_head_brief")),
            "blocker_detail": safe_text(cross_market_payload.get("review_head_blocker_detail")),
            "done_when": safe_text(cross_market_payload.get("review_head_done_when")),
        }
    return {
        "symbol": safe_text(hot_payload.get("cross_market_review_head_symbol")),
        "brief": safe_text(hot_payload.get("cross_market_review_head_brief")),
        "blocker_detail": safe_text(hot_payload.get("cross_market_review_head_blocker_detail")),
        "done_when": safe_text(hot_payload.get("cross_market_review_head_done_when")),
    }


def derive_symbol(review_payload: dict[str, Any]) -> str:
    direct = safe_text(review_payload.get("symbol"))
    if direct:
        return direct
    brief = safe_text(review_payload.get("brief"))
    parts = [segment.strip() for segment in brief.split(":") if segment.strip()]
    return parts[2] if len(parts) >= 3 else ""


def derive_scope(blocker_detail: str, probe_payload: dict[str, Any]) -> str:
    marker = "time-sync=threshold_breach:scope="
    if marker in blocker_detail:
        return blocker_detail.split(marker, 1)[1].split(";", 1)[0].strip()
    return safe_text(probe_payload.get("threshold_breach_scope")) or safe_text(probe_payload.get("classification"))


def merge_blocker_detail(review_blocker_detail: str, environment_blocker_detail: str) -> str:
    review_text = safe_text(review_blocker_detail)
    env_text = safe_text(environment_blocker_detail)
    if not env_text:
        return review_text
    base_review = review_text.split(" | env=", 1)[0].strip()
    if base_review:
        return f"{base_review} | env={env_text}"
    return f"env={env_text}"


def build_steps(
    *,
    env_payload: dict[str, Any],
    review_payload: dict[str, Any],
    review_dir: Path,
    output_root: Path,
) -> list[dict[str, Any]]:
    env_classification = safe_text(env_payload.get("classification"))
    remediation_hint = safe_text(env_payload.get("remediation_hint"))
    residual_path_hint = safe_text(env_payload.get("residual_path_hint"))
    clash_config_path = safe_text(env_payload.get("clash_config_path"))
    clash_merge_path = safe_text(env_payload.get("clash_merge_path"))
    clash_dns_mode = safe_text(env_payload.get("clash_dns_enhanced_mode"))
    clash_tun_stack = safe_text(env_payload.get("clash_tun_stack"))
    recent_ntp_success_detected = bool(env_payload.get("timed_recent_ntp_success_detected"))
    recent_direct_ntp_ips = [
        safe_text(item)
        for item in list(env_payload.get("timed_recent_direct_ntp_ips") or [])
        if safe_text(item)
    ]
    recent_direct_ntp_delay_ms_min = env_payload.get("timed_recent_direct_ntp_delay_ms_min")
    recent_direct_ntp_delay_ms_max = env_payload.get("timed_recent_direct_ntp_delay_ms_max")
    recent_ntp_fallback_seconds = env_payload.get("timed_recent_ntp_fallback_seconds")
    if not residual_path_hint and env_classification == "timed_apns_fallback_ntp_reachable_fake_ip_residual":
        residual_path_hint = "fake_ip_filter_present_stale_cache_or_tun_intercept"
    if not residual_path_hint and "residual_path_hint=fake_ip_filter_present_stale_cache_or_tun_intercept" in safe_text(
        env_payload.get("blocker_detail")
    ):
        residual_path_hint = "fake_ip_filter_present_stale_cache_or_tun_intercept"
    if not residual_path_hint and "residual_path_hint=clash_exemptions_present_os_time_service_not_recovered" in safe_text(
        env_payload.get("blocker_detail")
    ):
        residual_path_hint = "clash_exemptions_present_os_time_service_not_recovered"
    probe_cmd = (
        f"python3 {SYSTEM_ROOT / 'scripts' / 'build_system_time_sync_environment_report.py'} "
        f"--review-dir {review_dir} --artifact-dir {output_root / 'artifacts' / 'time_sync'}"
    )
    refresh_cmd = (
        f"python3 {SYSTEM_ROOT / 'scripts' / 'refresh_crypto_route_state.py'} "
        f"--review-dir {review_dir} --output-root {output_root} --skip-native-refresh"
    )
    verify_cmd = (
        f"python3 {SYSTEM_ROOT / 'scripts' / 'build_system_time_sync_repair_verification_report.py'} "
        f"--review-dir {review_dir} --output-root {output_root} --artifact-dir {output_root / 'artifacts' / 'time_sync'}"
    )
    steps: list[dict[str, Any]] = [
        {
            "rank": 1,
            "type": "read_only",
            "title": "Verify current macOS time-source state",
            "details": (
                "Check whether `timed` is still on APNS fallback and whether network time is enabled before making any repair."
            ),
            "admin_required": False,
            "commands": [
                "/usr/bin/log show --style compact --last 20m --predicate 'process == \"timed\"'",
                "/usr/sbin/scutil --dns",
            ],
        },
        {
            "rank": 2,
            "type": "manual_ui",
            "title": "Restore automatic date/time in macOS settings",
            "details": (
                "Open System Settings > General > Date & Time, turn on automatic date and time, choose a directly reachable network time server for your region, and turn on automatic time zone."
            ),
            "admin_required": True,
            "references": [
                APPLE_AUTO_TIME_URL,
                APPLE_TIME_SETTINGS_URL,
                APPLE_TIME_TROUBLESHOOT_URL,
            ],
        },
        {
            "rank": 3,
            "type": "admin_cli",
            "title": "Re-enable network time from Terminal if UI changes do not stick",
            "details": (
                "These commands require administrator access. Use a directly reachable server such as `ntp.aliyun.com` if `time.apple.com` remains unsuitable in your network path."
            ),
            "admin_required": True,
            "commands": [
                "sudo /usr/sbin/systemsetup -getusingnetworktime",
                "sudo /usr/sbin/systemsetup -getnetworktimeserver",
                "sudo /usr/sbin/systemsetup -setusingnetworktime on",
                "sudo /usr/sbin/systemsetup -setnetworktimeserver ntp.aliyun.com",
                "sudo /bin/launchctl print system/com.apple.timed",
                "sudo /bin/launchctl kickstart -k system/com.apple.timed",
            ],
        },
        {
            "rank": 4,
            "type": "admin_cli",
            "title": "Force a one-time clock resync if skew remains above threshold",
            "details": (
                "If automatic settings are enabled but the clock is still off by tens of milliseconds, perform a privileged SNTP sync once, then re-check `timed`."
            ),
            "admin_required": True,
            "commands": [
                "sudo /usr/bin/sntp -sS ntp.aliyun.com",
                "sudo /usr/bin/sntp -sS pool.ntp.org",
            ],
        },
        {
            "rank": 5,
            "type": "verify",
            "title": "Verify skew and rerun Fenlie source refresh",
            "details": (
                "The repair is complete only when offset stays within configured limits and downstream crypto review no longer carries a time-sync blocker."
            ),
            "admin_required": False,
            "commands": [
                "/usr/bin/sntp -d ntp.aliyun.com",
                "/usr/bin/sntp -d pool.ntp.org",
                probe_cmd,
                refresh_cmd,
                verify_cmd,
            ],
        },
    ]
    if env_classification in {
        "timed_apns_fallback",
        "timed_apns_fallback_ntp_reachable",
        "timed_apns_fallback_ntp_reachable_flapping",
        "timed_apns_fallback_ntp_reachable_fake_ip_residual",
    }:
        steps[1]["details"] += " Current diagnostics show macOS `timed` is falling back to APNS instead of active NTP."
    if env_classification == "timed_apns_fallback_ntp_reachable":
        steps[2]["details"] += " Current diagnostics indicate direct NTP sources are reachable; the issue is that macOS has not resumed using them as the active source."
    if env_classification == "timed_apns_fallback_ntp_reachable_flapping":
        steps[2]["details"] += " Current diagnostics indicate direct NTP sources are reachable and `timed` can briefly use them, but the OS time service is falling back to APNS again instead of staying on NTP."
    if env_classification == "timed_apns_fallback_ntp_reachable_fake_ip_residual":
        steps[2]["details"] += " Current diagnostics indicate direct NTP sources are reachable, but macOS has not resumed using them as the active source and recent timed attempts still hit a fake-IP/proxy path."
    if residual_path_hint == "clash_exemptions_present_os_time_service_not_recovered":
        steps[2]["details"] += (
            " Clash already exempts timed/sntp and the known NTP hosts, so prioritize toggling automatic time off/on, "
            "re-enabling network time, forcing a one-time admin resync, and restarting or rebooting the macOS time-service path before editing Clash again."
        )
        steps[2]["details"] += " Current launchd label for the time service is `system/com.apple.timed`."
        if recent_ntp_success_detected:
            steps[2]["details"] += " Recent logs show timed briefly reached NTP and then fell back to APNS again"
            if isinstance(recent_ntp_fallback_seconds, (int, float)):
                steps[2]["details"] += f" after about {float(recent_ntp_fallback_seconds):.3f}s"
            steps[2]["details"] += ", which is a strong signal that restarting the OS time-service path is worth doing after re-enabling network time."
    if recent_ntp_success_detected and recent_direct_ntp_ips:
        direct_ntp_detail = (
            f" Recent direct NTP fetches from timed already succeeded to {','.join(recent_direct_ntp_ips)}"
        )
        if isinstance(recent_direct_ntp_delay_ms_min, (int, float)) and isinstance(
            recent_direct_ntp_delay_ms_max, (int, float)
        ):
            direct_ntp_detail += (
                f" with recent delay range {float(recent_direct_ntp_delay_ms_min):.3f}-{float(recent_direct_ntp_delay_ms_max):.3f}ms"
            )
        if isinstance(recent_ntp_fallback_seconds, (int, float)):
            direct_ntp_detail += f" before falling back to APNS after about {float(recent_ntp_fallback_seconds):.3f}s"
        direct_ntp_detail += (
            ", so verification should focus on whether the macOS time-service path stays on NTP after the admin repair."
        )
        steps[4]["details"] += direct_ntp_detail
    if residual_path_hint == "fake_ip_filter_present_stale_cache_or_tun_intercept":
        steps[2]["details"] += " Because the fake-ip filter already covers the probe sources, prefer restarting Clash core / toggling TUN off and on / clearing stale fake-ip state before assuming the filter list itself is wrong."
        if clash_dns_mode or clash_tun_stack:
            steps[2]["details"] += (
                f" Current network stack hint: dns_mode={clash_dns_mode or '-'}, tun_stack={clash_tun_stack or '-'}."
            )
        steps[2]["details"] += " Mihomo docs state that fake-ip-filtered domains should not receive fake-ip mappings, so a continuing fake-ip hit suggests stale cache/TUN interception or a path outside the expected DNS flow."
        inspect_commands = list(steps[2].get("commands") or [])
        if clash_config_path:
            inspect_commands.append(
                f"rg -n 'fake-ip-filter|time.apple.com|time.windows.com|ntp.aliyun.com|pool.ntp.org|enhanced-mode|tun:' '{clash_config_path}'"
            )
        if clash_merge_path:
            inspect_commands.append(f"rg -n 'fake-ip-filter|enhanced-mode|tun:' '{clash_merge_path}'")
        if inspect_commands:
            steps[2]["commands"] = inspect_commands
    if remediation_hint:
        steps[4]["details"] += f" Current remediation hint: {remediation_hint}"
    if safe_text(review_payload.get("done_when")):
        steps[4]["details"] += f" Fenlie done_when: {safe_text(review_payload.get('done_when'))}"
    return steps


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# System Time Sync Repair Plan",
        "",
        f"- generated_at_utc: `{payload.get('generated_at_utc') or ''}`",
        f"- status: `{payload.get('status') or ''}`",
        f"- plan_brief: `{payload.get('plan_brief') or '-'}`",
        f"- review_head_brief: `{payload.get('review_head_brief') or '-'}`",
        f"- blocker_detail: `{payload.get('blocker_detail') or '-'}`",
        f"- environment_blocker_detail: `{payload.get('environment_blocker_detail') or '-'}`",
        f"- done_when: `{payload.get('done_when') or '-'}`",
        f"- admin_required: `{payload.get('admin_required')}`",
        f"- time_sync_probe_artifact: `{payload.get('time_sync_probe_artifact') or '-'}`",
        f"- environment_report_artifact: `{payload.get('environment_report_artifact') or '-'}`",
        f"- cross_market_artifact: `{payload.get('cross_market_artifact') or '-'}`",
        f"- verification_status: `{payload.get('verification_status') or '-'}`",
        f"- verification_brief: `{payload.get('verification_brief') or '-'}`",
        f"- verification_artifact: `{payload.get('verification_artifact') or '-'}`",
        "",
        "## References",
    ]
    for row in payload.get("references") or []:
        if not isinstance(row, dict):
            continue
        lines.append(f"- `{row.get('title')}`: `{row.get('url')}`")
    lines.extend(["", "## Steps"])
    for step in payload.get("steps") or []:
        if not isinstance(step, dict):
            continue
        lines.extend(
            [
                "",
                f"### {int(step.get('rank') or 0)}. {safe_text(step.get('title'))}",
                f"- type: `{safe_text(step.get('type'))}`",
                f"- admin_required: `{bool(step.get('admin_required'))}`",
                f"- details: {safe_text(step.get('details'))}",
            ]
        )
        commands = [safe_text(cmd) for cmd in list(step.get("commands") or []) if safe_text(cmd)]
        if commands:
            lines.append(f"- commands: `{ ' | '.join(commands) }`")
        refs = [safe_text(ref) for ref in list(step.get("references") or []) if safe_text(ref)]
        if refs:
            lines.append(f"- references: `{ ' | '.join(refs) }`")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a manual repair plan for macOS/system time-sync issues.")
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_TIME_SYNC_DIR)
    parser.add_argument("--cross-market-file", default="")
    parser.add_argument("--hot-brief-file", default="")
    parser.add_argument("--environment-report-file", default="")
    parser.add_argument("--time-sync-probe-file", default="")
    parser.add_argument("--now", default="")
    parser.add_argument("--artifact-keep", type=int, default=16)
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    args = parser.parse_args()

    reference_now = parse_now(args.now)
    review_dir = args.review_dir.resolve()
    output_root = args.output_root.resolve()
    artifact_dir = args.artifact_dir.resolve()

    cross_market_path = (
        Path(args.cross_market_file).expanduser().resolve()
        if str(args.cross_market_file).strip()
        else latest_review_json_artifact(review_dir, "cross_market_operator_state", reference_now)
    )
    hot_path = (
        Path(args.hot_brief_file).expanduser().resolve()
        if str(args.hot_brief_file).strip()
        else latest_review_json_artifact(review_dir, "hot_universe_operator_brief", reference_now)
    )
    env_path = (
        Path(args.environment_report_file).expanduser().resolve()
        if str(args.environment_report_file).strip()
        else latest_review_json_artifact(review_dir, "system_time_sync_environment_report", reference_now)
    )
    probe_path = (
        Path(args.time_sync_probe_file).expanduser().resolve()
        if str(args.time_sync_probe_file).strip()
        else latest_time_sync_probe(artifact_dir, reference_now)
    )

    cross_market_payload = load_json_mapping(cross_market_path)
    hot_payload = load_json_mapping(hot_path)
    env_payload = load_json_mapping(env_path)
    probe_payload = load_json_mapping(probe_path)
    review_payload = review_source_payload(cross_market_payload, hot_payload)

    review_head_brief = safe_text(review_payload.get("brief"))
    review_blocker_detail = safe_text(review_payload.get("blocker_detail"))
    environment_blocker_detail = safe_text(env_payload.get("blocker_detail"))
    blocker_detail = merge_blocker_detail(review_blocker_detail, environment_blocker_detail)
    done_when = safe_text(review_payload.get("done_when"))
    symbol = derive_symbol(review_payload)
    scope = derive_scope(review_blocker_detail or blocker_detail, probe_payload)
    env_classification = safe_text(env_payload.get("classification"))
    plan_suffix = env_classification or scope or "manual_repair"
    plan_brief = f"manual_time_repair_required:{symbol}:{plan_suffix}" if symbol else f"manual_time_repair_required:{plan_suffix}"
    admin_required = bool(
        env_classification
        in {
            "timed_apns_fallback",
            "timed_apns_fallback_ntp_reachable",
            "timed_apns_fallback_ntp_reachable_flapping",
            "timed_apns_fallback_ntp_reachable_fake_ip_residual",
            "timed_ntp_via_fake_ip",
        }
        or "administrator-level" in safe_text(env_payload.get("remediation_hint"))
    )
    steps = build_steps(
        env_payload=env_payload,
        review_payload=review_payload,
        review_dir=review_dir,
        output_root=output_root,
    )

    payload: dict[str, Any] = {
        "action": "build_system_time_sync_repair_plan",
        "ok": True,
        "status": "run_now" if "time-sync=threshold_breach:" in blocker_detail or scope else "verify_only",
        "generated_at_utc": fmt_utc(reference_now),
        "review_head_symbol": symbol,
        "review_head_brief": review_head_brief,
        "plan_brief": plan_brief,
        "blocker_detail": blocker_detail,
        "environment_blocker_detail": environment_blocker_detail,
        "done_when": done_when,
        "scope": scope,
        "admin_required": admin_required,
        "time_sync_probe_artifact": str(probe_path) if probe_path else "",
        "environment_report_artifact": str(env_path) if env_path else "",
        "cross_market_artifact": str(cross_market_path) if cross_market_path else "",
        "time_sync_probe_classification": safe_text(probe_payload.get("classification")),
        "environment_classification": env_classification,
        "environment_remediation_hint": safe_text(env_payload.get("remediation_hint")),
        "verification_artifact": "",
        "verification_status": "",
        "verification_brief": "",
        "references": [
            {"title": "Set the date and time automatically on your Mac", "url": APPLE_AUTO_TIME_URL},
            {"title": "Change Date & Time settings on Mac", "url": APPLE_TIME_SETTINGS_URL},
            {"title": "If you can't change the time or time zone on your Apple device", "url": APPLE_TIME_TROUBLESHOOT_URL},
            {"title": "mihomo DNS configuration", "url": MIHOMO_DNS_DOC_URL},
            {"title": "Clash.Meta NTP configuration", "url": CLASH_META_NTP_DOC_URL},
        ],
        "steps": steps,
    }

    verification_path = latest_review_json_artifact(
        review_dir,
        "system_time_sync_repair_verification_report",
        reference_now,
    )
    verification_payload = load_json_mapping(verification_path)
    payload["verification_artifact"] = str(verification_path) if verification_path else ""
    payload["verification_status"] = safe_text(verification_payload.get("status"))
    payload["verification_brief"] = safe_text(verification_payload.get("verification_brief"))

    stem = f"{reference_now.strftime('%Y%m%dT%H%M%SZ')}_system_time_sync_repair_plan"
    artifact_path = review_dir / f"{stem}.json"
    markdown_path = review_dir / f"{stem}.md"
    checksum_path = review_dir / f"{stem}_checksum.json"

    artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    checksum_path.write_text(
        json.dumps(
            {
                "artifact": str(artifact_path),
                "artifact_sha256": sha256_file(artifact_path),
                "markdown": str(markdown_path),
                "markdown_sha256": sha256_file(markdown_path),
                "generated_at_utc": fmt_utc(reference_now),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    pruned_keep, pruned_age = prune_review_artifacts(
        review_dir,
        stem="system_time_sync_repair_plan",
        current_paths=[artifact_path, markdown_path, checksum_path],
        keep=args.artifact_keep,
        ttl_hours=args.artifact_ttl_hours,
        now_dt=reference_now,
    )
    payload.update(
        {
            "artifact": str(artifact_path),
            "markdown": str(markdown_path),
            "checksum": str(checksum_path),
            "pruned_keep": pruned_keep,
            "pruned_age": pruned_age,
        }
    )
    artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checksum_path.write_text(
        json.dumps(
            {
                "artifact": str(artifact_path),
                "artifact_sha256": sha256_file(artifact_path),
                "markdown": str(markdown_path),
                "markdown_sha256": sha256_file(markdown_path),
                "generated_at_utc": fmt_utc(reference_now),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
