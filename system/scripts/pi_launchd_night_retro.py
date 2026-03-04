#!/usr/bin/env python3
"""Build launchd night retro report from log + whitelist samples."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

STAMP_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})Z\] (.*)$")
START_RE = re.compile(r"\bpi_cycle_launchd start\b")
DONE_RE = re.compile(r"\bpi_cycle_launchd done rc=(\d+)\b")
FUSE_RE = re.compile(r"\bpi_cycle_launchd fuse rc=(\d+) reason=([^\s]+)\b")


def parse_utc(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def fmt_utc(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_launchd_window(log_path: Path, start: datetime, end: datetime) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for raw in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = STAMP_RE.match(raw)
        if not m:
            continue
        ts = parse_utc(m.group(1) + "Z")
        if not (start <= ts <= end):
            continue
        records.append({"ts": ts, "msg": m.group(2), "line": raw})
    records.sort(key=lambda x: x["ts"])
    return records


def summarize_launchd(records: List[Dict[str, object]]) -> Dict[str, object]:
    starts = [r for r in records if START_RE.search(str(r["msg"]))]
    done_rows = []
    fuse_rows = []
    for rec in records:
        msg = str(rec["msg"])
        m = DONE_RE.search(msg)
        if m:
            done_rows.append({"ts_utc": fmt_utc(rec["ts"]), "rc": int(m.group(1))})
        m = FUSE_RE.search(msg)
        if m:
            fuse_rows.append(
                {
                    "ts_utc": fmt_utc(rec["ts"]),
                    "rc": int(m.group(1)),
                    "reason": m.group(2),
                    "line": rec["line"],
                }
            )
    key_rows = []
    key_markers = (
        "pi_cycle_launchd start",
        "pi_cycle_launchd done rc=",
        "pi_cycle_launchd fuse rc=",
        "ASSERT PASS:",
        "ENSURE PASS:",
        "FUSE:",
        "info=cloud_pass_loaded_from_keychain",
    )
    for rec in records:
        line = str(rec["line"])
        if any(m in line for m in key_markers):
            key_rows.append({"ts_utc": fmt_utc(rec["ts"]), "line": line})
    return {
        "start_count": len(starts),
        "done_count": len(done_rows),
        "fuse_count": len(fuse_rows),
        "done_rc_breakdown": dict(Counter(r["rc"] for r in done_rows)),
        "fuse_rc_breakdown": dict(Counter(r["rc"] for r in fuse_rows)),
        "fuse_reason_breakdown": dict(Counter(r["reason"] for r in fuse_rows)),
        "done_rows": done_rows,
        "fuse_rows": fuse_rows,
        "key_events": key_rows,
    }


def load_sample_rows(path: Path, start: datetime, end: datetime) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not path.exists():
        return rows
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            row = json.loads(raw)
            ts = parse_utc(str(row["timestamp_utc"]))
        except Exception:
            continue
        if not (start <= ts <= end):
            continue
        row["_ts"] = ts
        rows.append(row)
    rows.sort(key=lambda x: x["_ts"])
    return rows


def summarize_samples(sample_logs: List[Path], start: datetime, end: datetime) -> Dict[str, object]:
    all_rows: List[Dict[str, object]] = []
    by_path = {}
    for path in sample_logs:
        rows = load_sample_rows(path, start, end)
        by_path[str(path)] = rows
        all_rows.extend(rows)
    all_rows.sort(key=lambda x: x["_ts"])

    by_action: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in all_rows:
        by_action[str(row.get("action", "unknown"))].append(row)

    summary_rows = []
    for action in sorted(by_action.keys()):
        grp = by_action[action]
        samples = len(grp)
        success = sum(1 for r in grp if int(r.get("rc", 1)) == 0)
        nonzero = [
            {"timestamp_utc": str(r.get("timestamp_utc")), "rc": int(r.get("rc", 0))}
            for r in grp
            if int(r.get("rc", 0)) != 0
        ]
        tail = [
            {"timestamp_utc": str(r.get("timestamp_utc")), "rc": int(r.get("rc", 0))}
            for r in grp[-5:]
        ]
        summary_rows.append(
            {
                "action": action,
                "success": success,
                "samples": samples,
                "success_rate": round(success / samples, 4) if samples else 0.0,
                "recent_nonzero": nonzero[-5:],
                "recent_samples": tail,
            }
        )

    total_samples = len(all_rows)
    total_success = sum(1 for r in all_rows if int(r.get("rc", 1)) == 0)
    return {
        "total_samples": total_samples,
        "total_success": total_success,
        "total_success_rate": round(total_success / total_samples, 4) if total_samples else 0.0,
        "actions": summary_rows,
        "source_logs": [str(p) for p in sample_logs],
    }


def load_latest_whitelist_artifact(review_dir: Path) -> Dict[str, object]:
    items = sorted(review_dir.glob("*_openclaw_bridge_whitelist_24h.json"), reverse=True)
    if not items:
        return {"artifact_path": None}
    latest = items[0]
    payload = json.loads(latest.read_text(encoding="utf-8"))
    return {
        "artifact_path": str(latest),
        "generated_at_utc": payload.get("generated_at_utc"),
        "total_success_rate": payload.get("total_success_rate"),
        "gate_pass": (payload.get("gate") or {}).get("pass"),
        "gate_reasons": (payload.get("gate") or {}).get("reasons", []),
        "commands": payload.get("commands", []),
    }


def render_markdown(payload: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("# PI Launchd 夜间自动化复盘")
    lines.append("")
    lines.append("## 时间窗")
    window = payload["window_utc"]
    lines.append(f"- start: {window['start']}")
    lines.append(f"- end:   {window['end']}")
    lines.append("")

    launchd = payload["launchd"]
    lines.append("## Launchd 汇总")
    lines.append(
        f"- starts={launchd['start_count']} "
        f"dones={launchd['done_count']} "
        f"fuses={launchd['fuse_count']}"
    )
    lines.append(f"- done_rc_breakdown={launchd['done_rc_breakdown']}")
    lines.append(f"- fuse_rc_breakdown={launchd['fuse_rc_breakdown']}")
    lines.append(f"- fuse_reason_breakdown={launchd['fuse_reason_breakdown']}")
    lines.append("")

    samples = payload["samples"]
    lines.append("## 样本汇总")
    lines.append(
        f"- total_success={samples['total_success']}/{samples['total_samples']} "
        f"rate={samples['total_success_rate']}"
    )
    for row in samples["actions"]:
        lines.append(
            f"- {row['action']}: success={row['success']}/{row['samples']} "
            f"rate={row['success_rate']} recent_nonzero={row['recent_nonzero']}"
        )
    lines.append("")

    latest = payload["latest_whitelist_artifact"]
    lines.append("## 最新白名单工件")
    lines.append(f"- artifact: {latest['artifact_path']}")
    lines.append(f"- generated_at_utc: {latest.get('generated_at_utc')}")
    lines.append(f"- total_success_rate: {latest.get('total_success_rate')}")
    lines.append(f"- gate_pass: {latest.get('gate_pass')}")
    lines.append(f"- gate_reasons: {latest.get('gate_reasons')}")
    lines.append("")

    lines.append("## 关键事件")
    for row in launchd["key_events"]:
        lines.append(f"- {row['line']}")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--launchd-log",
        default=str(Path.home() / ".openclaw/logs/pi_cycle_launchd.log"),
        help="Path to pi_cycle_launchd.log",
    )
    parser.add_argument(
        "--sample-log",
        action="append",
        default=[],
        help="Whitelist sample jsonl path (can repeat).",
    )
    parser.add_argument(
        "--review-dir",
        default=str(Path.home() / ".openclaw/workspaces/pi/fenlie-system/output/review"),
        help="Review output directory.",
    )
    parser.add_argument(
        "--start-utc",
        help="Inclusive UTC start in RFC3339 Z format, e.g. 2026-03-03T19:50:00Z",
    )
    parser.add_argument(
        "--end-utc",
        help="Inclusive UTC end in RFC3339 Z format, e.g. 2026-03-04T03:10:00Z",
    )
    parser.add_argument("--window-hours", type=float, default=12.0)
    parser.add_argument(
        "--out-prefix",
        default="pi_launchd_night_retro",
        help="Output filename prefix under review dir.",
    )
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    end = parse_utc(args.end_utc) if args.end_utc else now
    start = parse_utc(args.start_utc) if args.start_utc else end - timedelta(hours=float(args.window_hours))
    if end <= start:
        raise SystemExit("invalid time window: end must be greater than start")

    launchd_log = Path(args.launchd_log)
    review_dir = Path(args.review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)

    sample_logs = [Path(p) for p in args.sample_log]
    if not sample_logs:
        default_sample = review_dir.parent / "logs/openclaw_bridge_whitelist_samples.jsonl"
        sample_logs = [default_sample]

    records = read_launchd_window(launchd_log, start, end)
    launchd = summarize_launchd(records)
    samples = summarize_samples(sample_logs, start, end)
    latest = load_latest_whitelist_artifact(review_dir)

    payload = {
        "generated_at_utc": fmt_utc(now),
        "window_utc": {"start": fmt_utc(start), "end": fmt_utc(end)},
        "launchd_log": str(launchd_log),
        "launchd": launchd,
        "samples": samples,
        "latest_whitelist_artifact": latest,
    }

    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    out_json = review_dir / f"{stamp}_{args.out_prefix}.json"
    out_md = review_dir / f"{stamp}_{args.out_prefix}.md"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(render_markdown(payload), encoding="utf-8")
    print(str(out_json))
    print(str(out_md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
