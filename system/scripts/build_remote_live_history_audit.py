#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse_now(raw: str) -> dt.datetime:
    text = str(raw or "").strip()
    if not text:
        return now_utc()
    value = text.replace("Z", "+00:00")
    parsed = dt.datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_payload_text(text: str) -> dict[str, Any]:
    clean = str(text or "").strip()
    if not clean:
        return {}
    candidates = [clean]
    first = clean.find("{")
    last = clean.rfind("}")
    if 0 <= first < last:
        candidates.append(clean[first : last + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return payload
        except Exception:
            continue
    for line in reversed([x.strip() for x in clean.splitlines() if x.strip()]):
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {"raw": clean}


def resolve_system_root(raw: str) -> Path:
    if str(raw or "").strip():
        return Path(str(raw).strip()).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def parse_windows(raw: str) -> list[int]:
    out: list[int] = []
    for token in str(raw or "").split(","):
        token_clean = token.strip()
        if not token_clean:
            continue
        value = int(token_clean)
        if value > 0 and value not in out:
            out.append(value)
    return out or [24, 168, 720]


def resolve_probe_mode(*, requested: str, window_hours: int) -> str:
    mode = str(requested or "auto").strip().lower()
    if mode == "auto":
        return "remote_async" if int(window_hours) > 168 else "direct"
    if mode in {"direct", "remote_capture", "remote_async"}:
        return mode
    return "direct"


def recommended_probe_rate_limit_per_minute(*, window_hours: int) -> int:
    hours = int(max(1, window_hours))
    if hours > 168:
        return 30
    if hours > 24:
        return 15
    return 10


def parse_probe_file_args(rows: list[str] | None) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for row in rows or []:
        text = str(row or "").strip()
        if not text or "=" not in text:
            continue
        lhs, rhs = text.split("=", 1)
        try:
            key = int(lhs.strip())
        except Exception:
            continue
        path = Path(rhs.strip()).expanduser().resolve()
        out[key] = path
    return out


def load_payload_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing": True, "path": str(path)}
    return parse_payload_text(path.read_text(encoding="utf-8"))


def run_probe(*, system_root: Path, market: str, window_hours: int, timeout_seconds: float, probe_mode: str) -> dict[str, Any]:
    script_path = system_root / "scripts" / "openclaw_cloud_bridge.sh"
    env = os.environ.copy()
    env["LIVE_TAKEOVER_MARKET"] = str(market)
    env["LIVE_TAKEOVER_TRADE_WINDOW_HOURS"] = str(int(window_hours))
    env["LIVE_TAKEOVER_PROBE_CAPTURE_MODE"] = str(probe_mode)
    env["LIVE_TAKEOVER_RATE_LIMIT_PER_MINUTE"] = str(recommended_probe_rate_limit_per_minute(window_hours=int(window_hours)))
    proc = subprocess.run(
        [str(script_path), "live-takeover-probe"],
        cwd=str(system_root.parent),
        text=True,
        capture_output=True,
        env=env,
        timeout=max(5.0, float(timeout_seconds)),
        check=False,
    )
    payload = parse_payload_text(str(proc.stdout or ""))
    if isinstance(payload, dict):
        payload = dict(payload)
        payload["__probe_returncode"] = int(proc.returncode)
        payload["__probe_stderr"] = str(proc.stderr or "")
        payload["__probe_stdout"] = str(proc.stdout or "")
        payload["__probe_mode"] = str(probe_mode)
    return {
        "returncode": int(proc.returncode),
        "stdout": str(proc.stdout or ""),
        "stderr": str(proc.stderr or ""),
        "payload": payload if isinstance(payload, dict) else {},
    }


def _trade_telemetry(payload: dict[str, Any]) -> dict[str, Any]:
    guarded = payload.get("guarded_exec", {}) if isinstance(payload.get("guarded_exec", {}), dict) else {}
    takeover = guarded.get("takeover", {}) if isinstance(guarded.get("takeover", {}), dict) else {}
    takeover_payload = takeover.get("payload", {}) if isinstance(takeover.get("payload", {}), dict) else {}
    steps = takeover_payload.get("steps", {}) if isinstance(takeover_payload.get("steps", {}), dict) else {}
    return steps if isinstance(steps, dict) else {}


def build_window_summary(*, window_hours: int, probe_payload: dict[str, Any], probe_source: str) -> dict[str, Any]:
    guarded = probe_payload.get("guarded_exec", {}) if isinstance(probe_payload.get("guarded_exec", {}), dict) else {}
    takeover = guarded.get("takeover", {}) if isinstance(guarded.get("takeover", {}), dict) else {}
    takeover_payload = takeover.get("payload", {}) if isinstance(takeover.get("payload", {}), dict) else {}
    steps = _trade_telemetry(probe_payload)
    account = steps.get("account_overview", {}) if isinstance(steps.get("account_overview", {}), dict) else {}
    live_snapshot = steps.get("live_snapshot", {}) if isinstance(steps.get("live_snapshot", {}), dict) else {}
    telemetry = steps.get("trade_telemetry", {}) if isinstance(steps.get("trade_telemetry", {}), dict) else {}
    risk_guard = steps.get("risk_guard", {}) if isinstance(steps.get("risk_guard", {}), dict) else {}
    signal_selection = steps.get("signal_selection", {}) if isinstance(steps.get("signal_selection", {}), dict) else {}
    blocked_candidate = signal_selection.get("blocked_candidate", {}) if isinstance(signal_selection.get("blocked_candidate", {}), dict) else {}
    status = str(probe_payload.get("status", "unknown"))
    market = str(takeover_payload.get("market", "") or probe_payload.get("market", ""))
    closed_pnl = float(live_snapshot.get("closed_pnl", 0.0) or 0.0)
    probe_mode = str(probe_payload.get("__probe_mode", "direct")).strip() or "direct"
    probe_transport = probe_payload.get("probe_transport", {}) if isinstance(probe_payload.get("probe_transport", {}), dict) else {}
    probe_returncode = int(probe_payload.get("__probe_returncode", 0) or 0)
    guarded_panic = str(guarded.get("panic", "")).strip()
    probe_panic = str(probe_payload.get("panic", "")).strip() or guarded_panic
    probe_stderr = str(probe_payload.get("__probe_stderr", "")).strip()
    probe_raw = str(probe_payload.get("raw", "")).strip()
    probe_error_detail = ""
    if status != "probe_completed":
        for candidate in (probe_panic, probe_raw, probe_stderr):
            if str(candidate).strip():
                probe_error_detail = str(candidate).strip()
                break
        if not probe_error_detail and probe_returncode:
            probe_error_detail = f"probe_returncode={probe_returncode}"
    return {
        "window_hours": int(window_hours),
        "probe_source": str(probe_source),
        "probe_status": status,
        "probe_executed": bool(probe_payload.get("executed", False)),
        "market": market,
        "probe_mode": probe_mode,
        "probe_transport": probe_transport,
        "probe_returncode": probe_returncode,
        "probe_panic": probe_panic,
        "probe_error_detail": probe_error_detail,
        "guard_artifact": str(guarded.get("artifact", "")),
        "takeover_artifact": str(takeover_payload.get("artifact", "")),
        "quote_available": float(account.get("quote_available", 0.0) or 0.0),
        "position_count": int(account.get("position_count", live_snapshot.get("open_positions", 0)) or 0),
        "open_positions": int(live_snapshot.get("open_positions", 0) or 0),
        "closed_count": int(live_snapshot.get("closed_count", 0) or 0),
        "closed_pnl": closed_pnl,
        "trade_count": int(telemetry.get("trades", 0) or 0),
        "income_rows": int(telemetry.get("income_rows", 0) or 0),
        "query_slice_hours": int(telemetry.get("query_slice_hours", 0) or 0),
        "trade_slice_count": int(telemetry.get("trade_slice_count", 0) or 0),
        "income_slice_count": int(telemetry.get("income_slice_count", 0) or 0),
        "trade_count_by_symbol": dict(telemetry.get("trade_count_by_symbol", {})),
        "income_count_by_symbol": dict(telemetry.get("income_count_by_symbol", {})),
        "income_pnl_by_symbol": dict(telemetry.get("income_pnl_by_symbol", {})),
        "income_pnl_by_day": dict(telemetry.get("income_pnl_by_day", {})),
        "risk_guard_status": str(risk_guard.get("status", "")),
        "risk_guard_allowed": bool(risk_guard.get("allowed", False)),
        "risk_guard_reasons": [str(x) for x in risk_guard.get("reasons", [])] if isinstance(risk_guard.get("reasons", []), list) else [],
        "blocked_candidate": blocked_candidate,
        "history_window_label": "24h" if int(window_hours) == 24 else "7d" if int(window_hours) == 168 else f"{int(window_hours // 24)}d",
    }


def build_markdown(*, market: str, windows: list[dict[str, Any]], generated_at: str) -> str:
    lines = [
        "# Remote Live History Audit",
        "",
        f"- generated_at_utc: `{generated_at}`",
        f"- market: `{market}`",
    ]
    for row in windows:
        lines.extend(
            [
                "",
                f"## {row['history_window_label']}",
                "",
                f"- status: `{row['probe_status']}`",
                f"- probe_mode: `{row['probe_mode']}`",
                f"- probe_returncode: `{row['probe_returncode']}`",
                f"- quote_available: `{row['quote_available']}`",
                f"- open_positions: `{row['open_positions']}`",
                f"- closed_count: `{row['closed_count']}`",
                f"- closed_pnl: `{row['closed_pnl']}`",
                f"- trades: `{row['trade_count']}`",
                f"- income_rows: `{row['income_rows']}`",
                f"- slice_hours: `{row['query_slice_hours']}`",
                f"- trade_slice_count: `{row['trade_slice_count']}`",
                f"- income_slice_count: `{row['income_slice_count']}`",
                f"- income_pnl_by_symbol: `{json.dumps(row['income_pnl_by_symbol'], ensure_ascii=False, sort_keys=True)}`",
                f"- income_pnl_by_day: `{json.dumps(row['income_pnl_by_day'], ensure_ascii=False, sort_keys=True)}`",
            ]
        )
        if str(row.get("probe_panic", "")).strip():
            lines.append(f"- probe_panic: `{row['probe_panic']}`")
        if str(row.get("probe_error_detail", "")).strip():
            lines.append(f"- probe_error_detail: `{row['probe_error_detail']}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build remote live history audit from guarded takeover probes.")
    parser.add_argument("--system-root", default="")
    parser.add_argument("--review-dir", default="")
    parser.add_argument("--market", default="portfolio_margin_um")
    parser.add_argument("--windows-hours", default="24,168,720")
    parser.add_argument("--probe-json", action="append", default=[], help="Optional mapping like 24=/path/to/probe.json")
    parser.add_argument("--probe-mode", choices=["auto", "direct", "remote_capture", "remote_async"], default="auto")
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--now", default="")
    args = parser.parse_args()

    system_root = resolve_system_root(str(args.system_root))
    review_dir = Path(args.review_dir).expanduser().resolve() if str(args.review_dir).strip() else system_root / "output" / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    market = str(args.market).strip().lower() or "portfolio_margin_um"
    windows_hours = parse_windows(str(args.windows_hours))
    probe_files = parse_probe_file_args(args.probe_json)
    now_dt = parse_now(str(args.now))
    stamp = now_dt.strftime("%Y%m%dT%H%M%SZ")

    window_summaries: list[dict[str, Any]] = []
    statuses: list[str] = []
    for window_hours in windows_hours:
        if window_hours in probe_files:
            payload = load_payload_file(probe_files[window_hours])
            source = str(probe_files[window_hours])
        else:
            probe_mode = resolve_probe_mode(requested=str(args.probe_mode), window_hours=window_hours)
            run_out = run_probe(
                system_root=system_root,
                market=market,
                window_hours=window_hours,
                timeout_seconds=float(args.timeout_seconds),
                probe_mode=probe_mode,
            )
            payload = run_out.get("payload", {}) if isinstance(run_out.get("payload", {}), dict) else {}
            if not payload and isinstance(run_out, dict):
                payload = {
                    "status": "probe_failed",
                    "executed": False,
                    "raw": str(run_out.get("stdout", "")),
                    "stderr": str(run_out.get("stderr", "")),
                    "returncode": int(run_out.get("returncode", 1) or 1),
                }
            source = "live-takeover-probe"
        summary = build_window_summary(window_hours=window_hours, probe_payload=payload, probe_source=source)
        window_summaries.append(summary)
        statuses.append(str(summary.get("probe_status", "")))

    ok = all(status == "probe_completed" for status in statuses)
    audit_status = "ok" if ok else "partial_failure"
    payload = {
        "generated_at_utc": now_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "market": market,
        "windows_hours": windows_hours,
        "status": audit_status,
        "ok": ok,
        "window_summaries": window_summaries,
    }

    artifact_path = review_dir / f"{stamp}_remote_live_history_audit.json"
    markdown_path = review_dir / f"{stamp}_remote_live_history_audit.md"
    checksum_path = review_dir / f"{stamp}_remote_live_history_audit_checksum.json"
    payload["artifact"] = str(artifact_path)
    payload["markdown_artifact"] = str(markdown_path)
    write_json(artifact_path, payload)
    markdown_path.write_text(build_markdown(market=market, windows=window_summaries, generated_at=payload["generated_at_utc"]), encoding="utf-8")
    checksum = {
        "artifact": str(artifact_path),
        "markdown_artifact": str(markdown_path),
        "sha256": sha256_file(artifact_path),
        "generated_at_utc": payload["generated_at_utc"],
    }
    write_json(checksum_path, checksum)
    write_json(review_dir / "latest_remote_live_history_audit.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
