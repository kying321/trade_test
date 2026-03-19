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


def derive_symbol(hot_payload: dict[str, Any], plan_payload: dict[str, Any]) -> str:
    direct = safe_text(hot_payload.get("cross_market_review_head_symbol")) or safe_text(plan_payload.get("review_head_symbol"))
    if direct:
        return direct
    brief = safe_text(hot_payload.get("cross_market_review_head_brief")) or safe_text(plan_payload.get("review_head_brief"))
    parts = [segment.strip() for segment in brief.split(":") if segment.strip()]
    return parts[2] if len(parts) >= 3 else ""


def review_source_payload(
    *,
    cross_market_payload: dict[str, Any],
    hot_payload: dict[str, Any],
    repair_plan_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "cross_market_review_head_symbol": safe_text(cross_market_payload.get("review_head_symbol"))
        or safe_text(hot_payload.get("cross_market_review_head_symbol"))
        or safe_text(repair_plan_payload.get("review_head_symbol")),
        "cross_market_review_head_brief": safe_text(cross_market_payload.get("review_head_brief"))
        or safe_text(hot_payload.get("cross_market_review_head_brief"))
        or safe_text(repair_plan_payload.get("review_head_brief")),
        "cross_market_review_head_blocker_detail": safe_text(cross_market_payload.get("review_head_blocker_detail"))
        or safe_text(hot_payload.get("cross_market_review_head_blocker_detail")),
        "cross_market_review_head_done_when": safe_text(cross_market_payload.get("review_head_done_when"))
        or safe_text(hot_payload.get("cross_market_review_head_done_when"))
        or safe_text(repair_plan_payload.get("done_when")),
    }


def derive_scope(blocker_detail: str, probe_payload: dict[str, Any]) -> str:
    marker = "time-sync=threshold_breach:scope="
    if marker in blocker_detail:
        return blocker_detail.split(marker, 1)[1].split(";", 1)[0].strip()
    return safe_text(probe_payload.get("threshold_breach_scope")) or safe_text(probe_payload.get("classification"))


def derive_current_repair_plan_brief(
    *,
    symbol: str,
    hot_payload: dict[str, Any],
    env_payload: dict[str, Any],
    probe_payload: dict[str, Any],
    repair_plan_payload: dict[str, Any],
) -> str:
    blocker_detail = safe_text(hot_payload.get("cross_market_review_head_blocker_detail"))
    env_classification = safe_text(env_payload.get("classification"))
    scope = derive_scope(blocker_detail, probe_payload)
    plan_suffix = env_classification or scope
    if plan_suffix:
        return (
            f"manual_time_repair_required:{symbol}:{plan_suffix}"
            if symbol
            else f"manual_time_repair_required:{plan_suffix}"
        )
    return safe_text(repair_plan_payload.get("plan_brief")) or "manual_time_repair_required"


def derive_check_rows(
    *,
    hot_payload: dict[str, Any],
    env_payload: dict[str, Any],
    probe_payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    blocker_detail = safe_text(hot_payload.get("cross_market_review_head_blocker_detail"))
    probe_classification = safe_text(probe_payload.get("classification"))
    env_classification = safe_text(env_payload.get("classification"))

    check_rows: list[dict[str, Any]] = []
    passing: list[str] = []
    failing: list[str] = []

    probe_ok = probe_classification in {"", "none"}
    probe_label = "probe_clear" if probe_ok else "probe_blocked"
    if probe_ok:
        passing.append(probe_label)
    else:
        failing.append(probe_label)
    check_rows.append(
        {
            "id": "time_sync_probe",
            "status": "passed" if probe_ok else "failed",
            "detail": (
                "time_sync_probe classification is clear."
                if probe_ok
                else (
                    f"classification={probe_classification or '-'}; "
                    f"scope={safe_text(probe_payload.get('threshold_breach_scope')) or '-'}; "
                    f"est_offset_ms={safe_text(probe_payload.get('threshold_breach_estimated_offset_ms')) or '-'}; "
                    f"est_rtt_ms={safe_text(probe_payload.get('threshold_breach_estimated_rtt_ms')) or '-'}"
                )
            ),
        }
    )

    env_ok = env_classification in {"", "none"} or (
        probe_ok and env_classification.startswith("timed_apns_fallback")
    )
    env_label = "environment_clear" if env_ok else "environment_blocked"
    if env_ok:
        passing.append(env_label)
    else:
        failing.append(env_label)
    check_rows.append(
        {
            "id": "system_time_environment",
            "status": "passed" if env_ok else "failed",
            "detail": (
                (
                    "system_time_sync_environment_report classification is clear."
                    if env_classification in {"", "none"}
                    else (
                        "time_sync_probe is clear and the remaining environment state is APNS fallback only; "
                        "no active fake-ip NTP blocker remains."
                    )
                )
                if env_ok
                else (
                    f"classification={env_classification or '-'}; "
                    f"blocker_detail={safe_text(env_payload.get('blocker_detail')) or '-'}"
                )
            ),
        }
    )

    review_blocker_present = "time-sync=threshold_breach:" in blocker_detail
    review_label = "review_head_time_sync_clear" if not review_blocker_present else "review_head_time_sync_blocked"
    if review_blocker_present:
        failing.append(review_label)
    else:
        passing.append(review_label)
    check_rows.append(
        {
            "id": "review_head_time_sync_blocker",
            "status": "passed" if not review_blocker_present else "failed",
            "detail": (
                "review head blocker no longer includes a time-sync threshold breach."
                if not review_blocker_present
                else blocker_detail
            ),
        }
    )

    return check_rows, passing, failing


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# System Time Sync Repair Verification Report",
        "",
        f"- generated_at_utc: `{payload.get('generated_at_utc') or ''}`",
        f"- status: `{payload.get('status') or ''}`",
        f"- verification_brief: `{payload.get('verification_brief') or '-'}`",
        f"- review_head_brief: `{payload.get('review_head_brief') or '-'}`",
        f"- repair_plan_brief: `{payload.get('repair_plan_brief') or '-'}`",
        f"- time_sync_probe_artifact: `{payload.get('time_sync_probe_artifact') or '-'}`",
        f"- environment_report_artifact: `{payload.get('environment_report_artifact') or '-'}`",
        f"- cross_market_artifact: `{payload.get('cross_market_artifact') or '-'}`",
        f"- hot_brief_artifact: `{payload.get('hot_brief_artifact') or '-'}`",
        "",
        "## Checks",
    ]
    for row in payload.get("check_rows") or []:
        if not isinstance(row, dict):
            continue
        lines.extend(
            [
                "",
                f"### {safe_text(row.get('id'))}",
                f"- status: `{safe_text(row.get('status'))}`",
                f"- detail: {safe_text(row.get('detail')) or '-'}",
            ]
        )
    lines.extend(
        [
            "",
            "## Next Actions",
        ]
    )
    for item in payload.get("next_actions") or []:
        text = safe_text(item)
        if text:
            lines.append(f"- {text}")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify whether system time sync repair is actually cleared.")
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_TIME_SYNC_DIR)
    parser.add_argument("--cross-market-file", default="")
    parser.add_argument("--hot-brief-file", default="")
    parser.add_argument("--environment-report-file", default="")
    parser.add_argument("--time-sync-probe-file", default="")
    parser.add_argument("--repair-plan-file", default="")
    parser.add_argument("--now", default="")
    parser.add_argument("--artifact-keep", type=int, default=16)
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    args = parser.parse_args()

    reference_now = parse_now(args.now)
    review_dir = args.review_dir.resolve()
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
    repair_plan_path = (
        Path(args.repair_plan_file).expanduser().resolve()
        if str(args.repair_plan_file).strip()
        else latest_review_json_artifact(review_dir, "system_time_sync_repair_plan", reference_now)
    )

    cross_market_payload = load_json_mapping(cross_market_path)
    hot_payload = load_json_mapping(hot_path)
    env_payload = load_json_mapping(env_path)
    probe_payload = load_json_mapping(probe_path)
    repair_plan_payload = load_json_mapping(repair_plan_path)
    review_payload = review_source_payload(
        cross_market_payload=cross_market_payload,
        hot_payload=hot_payload,
        repair_plan_payload=repair_plan_payload,
    )

    symbol = derive_symbol(review_payload, repair_plan_payload)
    review_head_brief = safe_text(review_payload.get("cross_market_review_head_brief")) or safe_text(
        repair_plan_payload.get("review_head_brief")
    )
    check_rows, passing_checks, failing_checks = derive_check_rows(
        hot_payload=review_payload,
        env_payload=env_payload,
        probe_payload=probe_payload,
    )
    repair_plan_brief = derive_current_repair_plan_brief(
        symbol=symbol,
        hot_payload=review_payload,
        env_payload=env_payload,
        probe_payload=probe_payload,
        repair_plan_payload=repair_plan_payload,
    )
    status = "cleared" if not failing_checks else "blocked"
    verification_suffix = "time_sync_ok" if status == "cleared" else "+".join(failing_checks)
    verification_brief = (
        f"{status}:{symbol}:{verification_suffix}" if symbol else f"{status}:{verification_suffix}"
    )
    next_actions = (
        [
            "Rerun the full Fenlie source refresh chain now that system time sync is clear.",
            f"python3 {SYSTEM_ROOT / 'scripts' / 'refresh_crypto_route_state.py'} --review-dir {review_dir} --output-root {args.output_root.resolve()} --skip-native-refresh",
        ]
        if status == "cleared"
        else [
            "Keep the manual/admin repair path active; the system time sync blocker is still present.",
            repair_plan_brief,
            safe_text(review_payload.get("cross_market_review_head_done_when"))
            or safe_text(repair_plan_payload.get("done_when"))
            or "synchronize system clock and rerun time_sync_probe",
        ]
    )

    payload: dict[str, Any] = {
        "action": "build_system_time_sync_repair_verification_report",
        "ok": True,
        "status": status,
        "generated_at_utc": fmt_utc(reference_now),
        "review_head_symbol": symbol,
        "review_head_brief": review_head_brief,
        "repair_plan_brief": repair_plan_brief,
        "verification_brief": verification_brief,
        "time_sync_probe_artifact": str(probe_path) if probe_path else "",
        "environment_report_artifact": str(env_path) if env_path else "",
        "cross_market_artifact": str(cross_market_path) if cross_market_path else "",
        "hot_brief_artifact": str(hot_path) if hot_path else "",
        "repair_plan_artifact": str(repair_plan_path) if repair_plan_path else "",
        "time_sync_probe_classification": safe_text(probe_payload.get("classification")),
        "environment_classification": safe_text(env_payload.get("classification")),
        "review_head_blocker_detail": safe_text(review_payload.get("cross_market_review_head_blocker_detail")),
        "passing_checks": passing_checks,
        "failing_checks": failing_checks,
        "check_rows": check_rows,
        "next_actions": next_actions,
        "cleared": status == "cleared",
    }

    stem = f"{reference_now.strftime('%Y%m%dT%H%M%SZ')}_system_time_sync_repair_verification_report"
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
        stem="system_time_sync_repair_verification_report",
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
