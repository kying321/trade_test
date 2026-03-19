#!/usr/bin/env python3
"""Rebuild local PI half-hour readiness artifacts from STATE.md event logs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


STATE_EVENT_PREFIXES = {
    "PI_CYCLE_EVENT=": ("state_pi_cycle", 1),
    "LIE_DRYRUN_EVENT=": ("state_lie_dryrun", 3),
}
DEFAULT_THRESHOLDS = {
    "coverage_min": 0.95,
    "max_missing_buckets": 16,
    "max_largest_missing_block_hours": 2.0,
    "max_allowlisted_scheduler_outage_buckets": 6,
    "max_foreign_overlap_buckets": 0,
    "max_after_last_allowlisted_buckets": 0,
}
ARTIFACT_PREFIX = "_pi_paper_mode_readiness_7d"
LAUNCHD_START_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})Z\] pi_cycle_launchd start$")


@dataclass
class EventRow:
    ts_utc: datetime
    bucket_utc_30m: datetime
    source: str
    source_priority: int
    confidence: float
    provenance_path: str
    provenance_line: int
    payload: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state-md",
        default="",
        help="Path to STATE.md. Defaults to $STATE_MD, ../STATE.md, then ~/.openclaw/workspaces/pi/STATE.md.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/review",
        help="Directory for readiness json/md/csv artifacts.",
    )
    parser.add_argument(
        "--window-days",
        type=float,
        default=7.0,
        help="Historical window in days. Ignored when --window-hours is set.",
    )
    parser.add_argument(
        "--window-hours",
        type=float,
        default=0.0,
        help="Historical window in hours for tests/bootstrap. Overrides --window-days when > 0.",
    )
    parser.add_argument(
        "--now-utc",
        default="",
        help="Optional RFC3339 UTC timestamp for deterministic rebuilds, e.g. 2026-03-08T10:30:00Z.",
    )
    parser.add_argument(
        "--artifact-ttl-days",
        type=int,
        default=30,
        help="Delete matching readiness artifacts older than this many days.",
    )
    parser.add_argument(
        "--launchd-log",
        default=str(Path.home() / ".openclaw" / "logs" / "pi_cycle_launchd.log"),
        help="Optional launchd log used to compute warmup coverage from scheduler starts.",
    )
    parser.add_argument(
        "--warmup-window-hours",
        type=float,
        default=float(os.getenv("LIE_PAPER_MODE_WARMUP_WINDOW_HOURS", "24")),
        help="Recent launchd coverage window for warmup gate.",
    )
    parser.add_argument(
        "--warmup-coverage-min",
        type=float,
        default=float(os.getenv("LIE_PAPER_MODE_WARMUP_COVERAGE_MIN", "0.80")),
        help="Minimum launchd bucket coverage required for warmup gate.",
    )
    parser.add_argument(
        "--warmup-max-missing-buckets",
        type=int,
        default=int(float(os.getenv("LIE_PAPER_MODE_WARMUP_MAX_MISSING_BUCKETS", "8"))),
        help="Maximum missing buckets allowed in the warmup window.",
    )
    parser.add_argument(
        "--warmup-max-largest-missing-block-hours",
        type=float,
        default=float(os.getenv("LIE_PAPER_MODE_WARMUP_MAX_LARGEST_MISSING_BLOCK_HOURS", "2.0")),
        help="Maximum contiguous missing block allowed in the warmup window.",
    )
    parser.add_argument(
        "--coverage-min",
        type=float,
        default=DEFAULT_THRESHOLDS["coverage_min"],
        help="Minimum bucket coverage required for ready status.",
    )
    parser.add_argument(
        "--max-missing-buckets",
        type=int,
        default=DEFAULT_THRESHOLDS["max_missing_buckets"],
        help="Maximum allowed missing half-hour buckets.",
    )
    parser.add_argument(
        "--max-largest-missing-block-hours",
        type=float,
        default=DEFAULT_THRESHOLDS["max_largest_missing_block_hours"],
        help="Maximum allowed contiguous missing block length in hours.",
    )
    parser.add_argument(
        "--max-allowlisted-scheduler-outage-buckets",
        type=int,
        default=DEFAULT_THRESHOLDS["max_allowlisted_scheduler_outage_buckets"],
        help="Maximum allowed buckets inferred as allowlisted scheduler outage.",
    )
    parser.add_argument(
        "--max-foreign-overlap-buckets",
        type=int,
        default=DEFAULT_THRESHOLDS["max_foreign_overlap_buckets"],
        help="Maximum allowed buckets covered only by foreign jobs.",
    )
    parser.add_argument(
        "--max-after-last-allowlisted-buckets",
        type=int,
        default=DEFAULT_THRESHOLDS["max_after_last_allowlisted_buckets"],
        help="Maximum allowed missing buckets after the last allowlisted event.",
    )
    return parser.parse_args()


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def now_utc(args: argparse.Namespace) -> datetime:
    if str(args.now_utc or "").strip():
        return parse_utc(str(args.now_utc).strip())
    return datetime.now(timezone.utc)


def align_bucket_30m(ts: datetime) -> datetime:
    minute = 0 if ts.minute < 30 else 30
    return ts.replace(minute=minute, second=0, microsecond=0)


def fmt_utc(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).isoformat()


def resolve_state_md(raw: str) -> Path:
    candidates: list[Path] = []
    if str(raw or "").strip():
        candidates.append(Path(str(raw)).expanduser())
    state_md_from_env = ""
    for key in ("STATE_MD", "LIE_STATE_MD"):
        value = str(os.getenv(key, "")).strip()
        if value:
            state_md_from_env = value
            break
    if state_md_from_env:
        candidates.append(Path(state_md_from_env).expanduser())
    candidates.extend(
        [
            Path.cwd().parent / "STATE.md",
            Path.cwd() / "STATE.md",
            Path.home() / ".openclaw" / "workspaces" / "pi" / "STATE.md",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError("state_md_missing")


def read_state_events(state_md: Path, start_utc: datetime, end_utc: datetime) -> list[EventRow]:
    rows: list[EventRow] = []
    for lineno, raw in enumerate(
        state_md.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        line = raw.strip()
        matched = None
        for prefix, meta in STATE_EVENT_PREFIXES.items():
            if line.startswith(prefix):
                matched = (prefix, meta)
                break
        if matched is None:
            continue
        prefix, (source, priority) = matched
        try:
            payload = json.loads(line[len(prefix) :])
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        ts_raw = payload.get("ts")
        if not ts_raw:
            continue
        try:
            ts_utc = parse_utc(str(ts_raw))
        except Exception:
            continue
        if ts_utc < start_utc or ts_utc >= end_utc + timedelta(minutes=30):
            continue
        rows.append(
            EventRow(
                ts_utc=ts_utc,
                bucket_utc_30m=align_bucket_30m(ts_utc),
                source=source,
                source_priority=priority,
                confidence=1.0 if source == "state_pi_cycle" else 0.95,
                provenance_path=str(state_md),
                provenance_line=lineno,
                payload=payload,
            )
        )
    rows.sort(key=lambda row: (row.bucket_utc_30m, row.source_priority, row.ts_utc))
    return rows


def choose_bucket_rows(rows: Iterable[EventRow]) -> dict[datetime, EventRow]:
    chosen: dict[datetime, EventRow] = {}
    for row in rows:
        current = chosen.get(row.bucket_utc_30m)
        if current is None:
            chosen[row.bucket_utc_30m] = row
            continue
        if (row.source_priority, row.ts_utc) < (current.source_priority, current.ts_utc):
            chosen[row.bucket_utc_30m] = row
    return chosen


def iter_expected_buckets(start_utc: datetime, end_utc: datetime) -> list[datetime]:
    buckets: list[datetime] = []
    cur = start_utc
    while cur < end_utc:
        buckets.append(cur)
        cur += timedelta(minutes=30)
    return buckets


def payload_to_matrix_row(bucket: datetime, row: EventRow | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "bucket_utc_30m": fmt_utc(bucket),
        "data_present": bool(row is not None),
        "data_reason": "missing",
        "source": "",
        "source_priority": None,
        "confidence": None,
        "source_ts_utc": None,
        "raw_ts": None,
        "events_in_bucket": 0,
        "pi_cycle_status": None,
        "core_returncode": None,
        "core_action": None,
        "core_bucket": None,
        "guardrail_hit": None,
        "guardrail_reasons": None,
        "would_run_pulse": None,
        "pulse_ran": None,
        "pulse_reason": None,
        "daemon_mode": None,
        "daemon_command": None,
        "due_slots": None,
        "paper_equity": None,
        "paper_cash_usdt": None,
        "paper_eth_qty": None,
        "paper_consecutive_losses": None,
        "provenance_path": None,
        "provenance_line": None,
    }
    if row is None:
        return out

    payload = row.payload
    out["data_reason"] = f"source:{row.source}"
    out["source"] = row.source
    out["source_priority"] = row.source_priority
    out["confidence"] = row.confidence
    out["source_ts_utc"] = fmt_utc(row.ts_utc)
    out["raw_ts"] = str(payload.get("ts") or "")
    out["events_in_bucket"] = 1
    out["provenance_path"] = row.provenance_path
    out["provenance_line"] = row.provenance_line

    if row.source == "state_pi_cycle":
        out["pi_cycle_status"] = payload.get("status")
        out["core_returncode"] = payload.get("core_execution_returncode")
        out["core_action"] = payload.get("core_execution_action")
        out["guardrail_hit"] = payload.get("core_execution_guardrail_hit")
        reasons = payload.get("core_execution_guardrail_reasons")
        out["guardrail_reasons"] = "|".join(str(x) for x in reasons) if isinstance(reasons, list) else None
        out["paper_equity"] = None
        out["paper_cash_usdt"] = None
        out["paper_eth_qty"] = None
        out["paper_consecutive_losses"] = None
        return out

    out["pi_cycle_status"] = f"lie_dryrun:{payload.get('decision')}"
    out["core_action"] = payload.get("action")
    out["core_bucket"] = payload.get("bucket")
    guardrails = payload.get("guardrails") if isinstance(payload.get("guardrails"), dict) else {}
    paper = payload.get("paper") if isinstance(payload.get("paper"), dict) else {}
    lie = payload.get("lie") if isinstance(payload.get("lie"), dict) else {}
    daemon = lie.get("daemon") if isinstance(lie.get("daemon"), dict) else {}
    pulse = lie.get("pulse") if isinstance(lie.get("pulse"), dict) else {}
    out["guardrail_hit"] = guardrails.get("hit")
    reasons = guardrails.get("reasons")
    out["guardrail_reasons"] = "|".join(str(x) for x in reasons) if isinstance(reasons, list) else None
    out["would_run_pulse"] = daemon.get("would_run_pulse")
    out["pulse_ran"] = pulse.get("ran")
    out["pulse_reason"] = daemon.get("pulse_reason") or pulse.get("reason")
    out["daemon_mode"] = daemon.get("mode")
    out["daemon_command"] = daemon.get("command")
    due_slots = daemon.get("due_slots")
    out["due_slots"] = "|".join(str(x) for x in due_slots) if isinstance(due_slots, list) else None
    out["paper_equity"] = paper.get("equity")
    out["paper_cash_usdt"] = paper.get("cash_usdt")
    out["paper_eth_qty"] = paper.get("eth_qty")
    out["paper_consecutive_losses"] = paper.get("consecutive_losses")
    return out


def build_missing_blocks(expected_buckets: list[datetime], present: set[datetime]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    start_idx: int | None = None
    for idx, bucket in enumerate(expected_buckets):
        if bucket not in present and start_idx is None:
            start_idx = idx
        if bucket in present and start_idx is not None:
            end_idx = idx
            blocks.append(_build_missing_block(expected_buckets, start_idx, end_idx))
            start_idx = None
    if start_idx is not None:
        blocks.append(_build_missing_block(expected_buckets, start_idx, len(expected_buckets)))
    return blocks


def _build_missing_block(expected_buckets: list[datetime], start_idx: int, end_idx: int) -> dict[str, Any]:
    start_bucket = expected_buckets[start_idx]
    last_bucket = expected_buckets[end_idx - 1]
    bucket_count = end_idx - start_idx
    return {
        "start_bucket_utc": fmt_utc(start_bucket),
        "end_bucket_utc": fmt_utc(last_bucket),
        "bucket_count": bucket_count,
        "duration_hours": round(bucket_count * 0.5, 3),
    }


def build_gap_forensics(expected_buckets: list[datetime], present: set[datetime]) -> list[dict[str, Any]]:
    present_sorted = sorted(present)
    rows: list[dict[str, Any]] = []
    if not expected_buckets:
        return rows
    last_present = present_sorted[-1] if present_sorted else None
    for bucket in expected_buckets:
        if bucket in present:
            continue
        prev_ts = max((ts for ts in present_sorted if ts < bucket), default=None)
        next_ts = min((ts for ts in present_sorted if ts > bucket), default=None)
        prev_gap = round((bucket - prev_ts).total_seconds() / 3600.0, 3) if prev_ts else None
        next_gap = round((next_ts - bucket).total_seconds() / 3600.0, 3) if next_ts else None
        span = None
        if prev_gap is not None and next_gap is not None:
            span = round(prev_gap + next_gap, 3)
        if last_present is not None and bucket > last_present:
            cause = "after_last_allowlisted_run"
        else:
            cause = "allowlisted_scheduler_outage"
        rows.append(
            {
                "bucket_utc_30m": fmt_utc(bucket),
                "allowlisted_run_overlap_count": 0,
                "foreign_run_overlap_count": 0,
                "foreign_job_names": "",
                "nearest_prev_allowlisted_run_ts": fmt_utc(prev_ts) if prev_ts else "",
                "nearest_next_allowlisted_run_ts": fmt_utc(next_ts) if next_ts else "",
                "prev_gap_hours": prev_gap,
                "next_gap_hours": next_gap,
                "nearest_allowlisted_span_hours": span,
                "inferred_cause": cause,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_launchd_buckets(log_path: Path, start_utc: datetime, end_utc: datetime) -> set[datetime]:
    if not log_path.exists():
        return set()
    buckets: set[datetime] = set()
    for raw in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = LAUNCHD_START_RE.match(raw.strip())
        if not match:
            continue
        ts_utc = parse_utc(match.group(1) + "Z")
        if start_utc <= ts_utc < end_utc + timedelta(minutes=30):
            buckets.add(align_bucket_30m(ts_utc))
    return buckets


def build_warmup_gate(
    *,
    log_path: Path,
    end_utc: datetime,
    window_hours: float,
    coverage_min: float,
    max_missing_buckets: int,
    max_largest_missing_block_hours: float,
) -> dict[str, Any]:
    start_utc = end_utc - timedelta(hours=window_hours)
    expected_buckets = iter_expected_buckets(start_utc, end_utc)
    present_buckets = read_launchd_buckets(log_path, start_utc=start_utc, end_utc=end_utc)
    missing_blocks = build_missing_blocks(expected_buckets, present_buckets)
    coverage = round(len(present_buckets) / len(expected_buckets), 6) if expected_buckets else 0.0
    missing_buckets = len(expected_buckets) - len(present_buckets)
    largest_missing_block_hours = round(
        max((float(block["duration_hours"]) for block in missing_blocks), default=0.0), 3
    )
    checks = [
        {
            "name": "coverage_min",
            "ok": coverage >= coverage_min,
            "actual": coverage,
            "expected": f">={coverage_min}",
        },
        {
            "name": "missing_buckets_max",
            "ok": missing_buckets <= max_missing_buckets,
            "actual": missing_buckets,
            "expected": f"<={max_missing_buckets}",
        },
        {
            "name": "largest_missing_block_hours_max",
            "ok": largest_missing_block_hours <= max_largest_missing_block_hours,
            "actual": largest_missing_block_hours,
            "expected": f"<={max_largest_missing_block_hours}",
        },
    ]
    return {
        "enabled": True,
        "log_path": str(log_path),
        "window_hours": round(window_hours, 3),
        "window_start_utc": fmt_utc(start_utc),
        "window_end_utc": fmt_utc(end_utc),
        "expected_buckets": len(expected_buckets),
        "buckets_with_data": len(present_buckets),
        "missing_buckets": missing_buckets,
        "coverage": coverage,
        "largest_missing_block_hours": largest_missing_block_hours,
        "checks": checks,
        "ok": all(bool(row["ok"]) for row in checks),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prune_old_artifacts(output_dir: Path, ttl_days: int) -> list[str]:
    if ttl_days <= 0:
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=ttl_days)
    removed: list[str] = []
    for path in output_dir.glob(f"*{ARTIFACT_PREFIX}*"):
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except Exception:
            continue
        if mtime < cutoff:
            path.unlink(missing_ok=True)
            removed.append(str(path))
    return removed


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# PI Paper Mode Readiness (7D)",
        "",
        f"- status: `{payload['status']}`",
        f"- ready_for_paper_mode: `{payload['ready_for_paper_mode']}`",
        f"- fail_reasons: `{payload['fail_reasons']}`",
        "",
        "## Checks",
        "",
    ]
    for row in payload["checks"]:
        lines.append(
            f"- `{row['name']}` | ok=`{row['ok']}` | actual=`{row['actual']}` | expected=`{row['expected']}`"
        )
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            f"- coverage: `{payload['coverage']}`",
            f"- missing_buckets: `{payload['missing_buckets']}`",
            f"- largest_missing_block_hours: `{payload['largest_missing_block_hours']}`",
            f"- allowlisted_scheduler_outage_buckets: `{payload['allowlisted_scheduler_outage_buckets']}`",
            f"- foreign_overlap_buckets: `{payload['foreign_overlap_buckets']}`",
            f"- after_last_allowlisted_buckets: `{payload['after_last_allowlisted_buckets']}`",
            f"- gap_cause_counts: `{payload['gap_cause_counts']}`",
            "",
            "## Warmup Gate",
            "",
            f"- ok: `{payload['warmup_gate']['ok']}`",
            f"- window_hours: `{payload['warmup_gate']['window_hours']}`",
            f"- coverage: `{payload['warmup_gate']['coverage']}`",
            f"- missing_buckets: `{payload['warmup_gate']['missing_buckets']}`",
            f"- largest_missing_block_hours: `{payload['warmup_gate']['largest_missing_block_hours']}`",
            "",
            "## Artifacts",
            f"- readiness_json: `{payload['artifacts']['paper_readiness_json']}`",
            f"- matrix: `{payload['artifacts']['matrix_csv']}`",
            f"- missing_blocks: `{payload['artifacts']['missing_blocks_csv']}`",
            f"- gap_forensics: `{payload['artifacts']['gap_forensics_csv']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    current = align_bucket_30m(now_utc(args))
    window_hours = float(args.window_hours) if float(args.window_hours) > 0 else float(args.window_days) * 24.0
    start_utc = current - timedelta(hours=window_hours)
    state_md = resolve_state_md(str(args.state_md or ""))
    output_dir = Path(str(args.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_state_events(state_md, start_utc=start_utc, end_utc=current)
    chosen = choose_bucket_rows(rows)
    expected_buckets = iter_expected_buckets(start_utc, current)
    present_buckets = set(chosen.keys())
    matrix_rows = [payload_to_matrix_row(bucket, chosen.get(bucket)) for bucket in expected_buckets]
    missing_blocks = build_missing_blocks(expected_buckets, present_buckets)
    gap_forensics = build_gap_forensics(expected_buckets, present_buckets)
    warmup_gate = build_warmup_gate(
        log_path=Path(str(args.launchd_log)).expanduser().resolve(),
        end_utc=current,
        window_hours=float(args.warmup_window_hours),
        coverage_min=float(args.warmup_coverage_min),
        max_missing_buckets=int(args.warmup_max_missing_buckets),
        max_largest_missing_block_hours=float(args.warmup_max_largest_missing_block_hours),
    )

    gap_cause_counts: dict[str, int] = {}
    for row in gap_forensics:
        cause = str(row["inferred_cause"])
        gap_cause_counts[cause] = gap_cause_counts.get(cause, 0) + 1

    missing_buckets = len(expected_buckets) - len(present_buckets)
    coverage = round(len(present_buckets) / len(expected_buckets), 6) if expected_buckets else 0.0
    largest_missing_block_hours = round(
        max((float(block["duration_hours"]) for block in missing_blocks), default=0.0), 3
    )
    allowlisted_scheduler_outage_buckets = int(gap_cause_counts.get("allowlisted_scheduler_outage", 0))
    foreign_overlap_buckets = int(gap_cause_counts.get("foreign_jobs_only_allowlisted_gap", 0))
    after_last_allowlisted_buckets = int(gap_cause_counts.get("after_last_allowlisted_run", 0))

    thresholds = {
        "coverage_min": float(args.coverage_min),
        "max_missing_buckets": int(args.max_missing_buckets),
        "max_largest_missing_block_hours": float(args.max_largest_missing_block_hours),
        "max_allowlisted_scheduler_outage_buckets": int(args.max_allowlisted_scheduler_outage_buckets),
        "max_foreign_overlap_buckets": int(args.max_foreign_overlap_buckets),
        "max_after_last_allowlisted_buckets": int(args.max_after_last_allowlisted_buckets),
    }

    checks = [
        {
            "name": "coverage_min",
            "ok": coverage >= thresholds["coverage_min"],
            "actual": coverage,
            "expected": f">={thresholds['coverage_min']}",
        },
        {
            "name": "missing_buckets_max",
            "ok": missing_buckets <= thresholds["max_missing_buckets"],
            "actual": missing_buckets,
            "expected": f"<={thresholds['max_missing_buckets']}",
        },
        {
            "name": "largest_missing_block_hours_max",
            "ok": largest_missing_block_hours <= thresholds["max_largest_missing_block_hours"],
            "actual": largest_missing_block_hours,
            "expected": f"<={thresholds['max_largest_missing_block_hours']}",
        },
        {
            "name": "allowlisted_scheduler_outage_buckets_max",
            "ok": allowlisted_scheduler_outage_buckets <= thresholds["max_allowlisted_scheduler_outage_buckets"],
            "actual": allowlisted_scheduler_outage_buckets,
            "expected": f"<={thresholds['max_allowlisted_scheduler_outage_buckets']}",
        },
        {
            "name": "foreign_overlap_buckets_max",
            "ok": foreign_overlap_buckets <= thresholds["max_foreign_overlap_buckets"],
            "actual": foreign_overlap_buckets,
            "expected": f"<={thresholds['max_foreign_overlap_buckets']}",
        },
        {
            "name": "after_last_allowlisted_buckets_max",
            "ok": after_last_allowlisted_buckets <= thresholds["max_after_last_allowlisted_buckets"],
            "actual": after_last_allowlisted_buckets,
            "expected": f"<={thresholds['max_after_last_allowlisted_buckets']}",
        },
    ]
    fail_reasons = [str(row["name"]) for row in checks if not bool(row["ok"])]
    ready_for_paper_mode = len(fail_reasons) == 0
    status = "ready" if ready_for_paper_mode else "blocked"

    report_date = current.date().isoformat()
    readiness_json = output_dir / f"{report_date}{ARTIFACT_PREFIX}.json"
    readiness_md = output_dir / f"{report_date}{ARTIFACT_PREFIX}.md"
    matrix_csv = output_dir / f"{report_date}_pi_halfhour_matrix_7d_reconstructed.csv"
    missing_blocks_csv = output_dir / f"{report_date}_pi_halfhour_missing_blocks_7d_reconstructed.csv"
    gap_forensics_csv = output_dir / f"{report_date}_pi_halfhour_gap_forensics_7d_reconstructed.csv"
    checksum_json = output_dir / f"{report_date}{ARTIFACT_PREFIX}_checksum.json"

    write_csv(matrix_csv, matrix_rows)
    write_csv(missing_blocks_csv, missing_blocks)
    write_csv(gap_forensics_csv, gap_forensics)

    payload: dict[str, Any] = {
        "generated_at": fmt_utc(datetime.now(timezone.utc)),
        "report_ts": fmt_utc(current),
        "window_start_utc": fmt_utc(start_utc),
        "window_end_utc": fmt_utc(current),
        "report_window_hours": round(window_hours, 3),
        "state_md": str(state_md),
        "expected_buckets": len(expected_buckets),
        "buckets_with_data": len(present_buckets),
        "missing_buckets": missing_buckets,
        "coverage": coverage,
        "largest_missing_block_hours": largest_missing_block_hours,
        "allowlisted_runs_in_window": len(rows),
        "allowlisted_missing_overlap_count": 0,
        "allowlisted_scheduler_outage_buckets": allowlisted_scheduler_outage_buckets,
        "foreign_overlap_buckets": foreign_overlap_buckets,
        "after_last_allowlisted_buckets": after_last_allowlisted_buckets,
        "gap_cause_counts": gap_cause_counts,
        "thresholds": thresholds,
        "checks": checks,
        "ready_for_paper_mode": ready_for_paper_mode,
        "status": status,
        "fail_reasons": fail_reasons,
        "warmup_gate": warmup_gate,
        "artifacts": {
            "paper_readiness_json": str(readiness_json),
            "paper_readiness_md": str(readiness_md),
            "matrix_csv": str(matrix_csv),
            "missing_blocks_csv": str(missing_blocks_csv),
            "gap_forensics_csv": str(gap_forensics_csv),
            "checksum_json": str(checksum_json),
        },
    }
    readiness_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readiness_md.write_text(render_md(payload), encoding="utf-8")

    checksum_payload = {
        "generated_at": fmt_utc(datetime.now(timezone.utc)),
        "artifacts": {
            "paper_readiness_json": sha256_file(readiness_json),
            "paper_readiness_md": sha256_file(readiness_md),
            "matrix_csv": sha256_file(matrix_csv),
            "missing_blocks_csv": sha256_file(missing_blocks_csv),
            "gap_forensics_csv": sha256_file(gap_forensics_csv),
        },
        "pruned_artifacts": prune_old_artifacts(output_dir, int(args.artifact_ttl_days)),
    }
    checksum_json.write_text(
        json.dumps(checksum_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    out = {
        "ok": True,
        "status": status,
        "ready_for_paper_mode": ready_for_paper_mode,
        "paper_readiness_json": str(readiness_json),
        "paper_readiness_md": str(readiness_md),
        "matrix_csv": str(matrix_csv),
        "missing_blocks_csv": str(missing_blocks_csv),
        "gap_forensics_csv": str(gap_forensics_csv),
        "checksum_json": str(checksum_json),
        "coverage": coverage,
        "missing_buckets": missing_buckets,
        "largest_missing_block_hours": largest_missing_block_hours,
        "fail_reasons": fail_reasons,
        "warmup_gate": warmup_gate,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
