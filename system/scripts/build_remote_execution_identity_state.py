#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def resolve_system_root() -> Path:
    env_root = str(os.getenv("LIE_SYSTEM_ROOT", "")).strip() or str(
        os.getenv("FENLIE_SYSTEM_ROOT", "")
    ).strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


SYSTEM_ROOT = resolve_system_root()
DEFAULT_REVIEW_DIR = SYSTEM_ROOT / "output" / "review"


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def find_latest(review_dir: Path, pattern: str) -> Path | None:
    files = sorted(review_dir.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
    return files[0] if files else None


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def text(value: Any) -> str:
    return str(value or "").strip()


def unwrap_operator_handoff(payload: dict[str, Any]) -> dict[str, Any]:
    nested = payload.get("operator_handoff")
    return nested if isinstance(nested, dict) else payload


def load_execution_contract(
    *,
    handoff_payload: dict[str, Any],
    contract_payload: dict[str, Any],
) -> dict[str, Any]:
    contract = as_dict(contract_payload)
    if contract:
        return {
            "status": text(contract.get("contract_status")),
            "brief": text(contract.get("contract_brief")),
            "mode": text(contract.get("contract_mode")),
            "guarded_probe_allowed": bool(contract.get("guarded_probe_allowed", False)),
            "live_orders_allowed": bool(contract.get("live_orders_allowed", False)),
            "reason_codes": [text(code) for code in as_list(contract.get("reason_codes")) if text(code)],
            "executor_mode": text(contract.get("executor_mode")),
            "executor_mode_source": text(contract.get("executor_mode_source")),
            "blocker_detail": text(contract.get("blocker_detail")),
            "done_when": text(contract.get("done_when")),
        }
    operator = unwrap_operator_handoff(handoff_payload)
    return as_dict(operator.get("execution_contract"))


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
        "*_remote_execution_identity_state.json",
        "*_remote_execution_identity_state.md",
        "*_remote_execution_identity_state_checksum.json",
    ):
        candidates.extend(review_dir.glob(pattern))

    existing: list[tuple[float, Path]] = []
    for path in candidates:
        try:
            existing.append((path.stat().st_mtime, path))
        except OSError:
            continue

    pruned_age: list[str] = []
    survivors: list[Path] = []
    for _, path in sorted(existing, key=lambda item: item[0], reverse=True):
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


def build_payload(
    *,
    handoff_path: Path,
    handoff_payload: dict[str, Any],
    contract_path: Path | None,
    contract_payload: dict[str, Any],
    live_gate_path: Path,
    live_gate_payload: dict[str, Any],
    history_path: Path | None,
    history_payload: dict[str, Any],
    reference_now: dt.datetime,
) -> dict[str, Any]:
    operator = unwrap_operator_handoff(handoff_payload)
    account_scope = as_dict(operator.get("account_scope_alignment"))
    execution_contract = load_execution_contract(
        handoff_payload=handoff_payload,
        contract_payload=contract_payload,
    )
    remote_diag = as_dict(operator.get("remote_live_diagnosis"))
    live_decision = as_dict(live_gate_payload.get("live_decision"))
    blocker_names = [
        text(row.get("name"))
        for row in as_list(live_gate_payload.get("blockers"))
        if isinstance(row, dict) and text(row.get("name"))
    ]
    ready_check_scope_market = text(operator.get("ready_check_scope_market")) or text(
        history_payload.get("market")
    )
    remote_history_market = text(history_payload.get("market")) or ready_check_scope_market
    identity_brief = ":".join(
        [
            text(operator.get("remote_host")) or "remote-host-missing",
            ready_check_scope_market or "market-missing",
            text(account_scope.get("brief")) or "scope-missing",
            text(remote_diag.get("status")) or "diagnosis-missing",
        ]
    )
    return {
        "action": "build_remote_execution_identity_state",
        "ok": True,
        "status": "ok",
        "generated_at_utc": fmt_utc(reference_now),
        "identity_brief": identity_brief,
        "remote_host": text(operator.get("remote_host")),
        "remote_user": text(operator.get("remote_user")),
        "remote_project_dir": text(operator.get("remote_project_dir")),
        "handoff_state": text(operator.get("handoff_state")),
        "operator_status_quad": text(operator.get("operator_status_quad"))
        or text(live_gate_payload.get("operator_status_quad")),
        "operator_status_triplet": text(live_gate_payload.get("operator_status_triplet")),
        "ready_check_scope_market": ready_check_scope_market,
        "ready_check_scope_source": text(operator.get("ready_check_scope_source")),
        "ready_check_scope_brief": text(operator.get("ready_check_scope_brief")),
        "remote_history_market": remote_history_market,
        "remote_history_window_brief": text(operator.get("remote_live_history", {}).get("window_brief"))
        or text(history_payload.get("window_brief")),
        "account_scope_alignment_status": text(account_scope.get("status")),
        "account_scope_alignment_brief": text(account_scope.get("brief")),
        "account_scope_alignment_blocking": bool(account_scope.get("blocking", False)),
        "account_scope_alignment_blocker_detail": text(account_scope.get("blocker_detail")),
        "account_scope_alignment_done_when": text(account_scope.get("done_when")),
        "execution_contract_status": text(execution_contract.get("status")),
        "execution_contract_brief": text(execution_contract.get("brief")),
        "execution_contract_mode": text(execution_contract.get("mode")),
        "execution_contract_executor_mode": text(execution_contract.get("executor_mode")),
        "execution_contract_executor_mode_source": text(
            execution_contract.get("executor_mode_source")
        ),
        "execution_contract_guarded_probe_allowed": bool(
            execution_contract.get("guarded_probe_allowed", False)
        ),
        "execution_contract_live_orders_allowed": bool(
            execution_contract.get("live_orders_allowed", False)
        ),
        "execution_contract_reason_codes": [
            text(code)
            for code in as_list(execution_contract.get("reason_codes"))
            if text(code)
        ],
        "execution_contract_blocker_detail": text(execution_contract.get("blocker_detail")),
        "execution_contract_done_when": text(execution_contract.get("done_when")),
        "remote_live_diagnosis_status": text(remote_diag.get("status")),
        "remote_live_diagnosis_brief": text(remote_diag.get("brief")),
        "remote_live_diagnosis_blocker_detail": text(remote_diag.get("blocker_detail")),
        "remote_live_diagnosis_done_when": text(remote_diag.get("done_when")),
        "remote_profitability_window": text(remote_diag.get("profitability_window")),
        "remote_profitability_pnl": remote_diag.get("profitability_pnl"),
        "remote_profitability_trade_count": remote_diag.get("profitability_trade_count"),
        "live_decision": text(live_decision.get("current_decision")),
        "live_decision_summary": text(live_decision.get("summary")),
        "blocking_layers": blocker_names,
        "done_when": text(account_scope.get("done_when")) or text(remote_diag.get("done_when")),
        "artifacts": {
            "remote_live_handoff": str(handoff_path),
            "remote_execution_contract_state": str(contract_path) if contract_path else "",
            "live_gate_blocker_report": str(live_gate_path),
            "remote_live_history_audit": str(history_path) if history_path else "",
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Remote Execution Identity State",
            "",
            f"- brief: `{text(payload.get('identity_brief'))}`",
            f"- remote host: `{text(payload.get('remote_host'))}`",
            f"- remote project: `{text(payload.get('remote_project_dir'))}`",
            f"- scope: `{text(payload.get('ready_check_scope_brief'))}`",
            f"- history market: `{text(payload.get('remote_history_market'))}`",
            f"- scope alignment: `{text(payload.get('account_scope_alignment_brief'))}`",
            f"- execution contract: `{text(payload.get('execution_contract_brief'))}`",
            f"- diagnosis: `{text(payload.get('remote_live_diagnosis_brief'))}`",
            f"- live decision: `{text(payload.get('live_decision'))}`",
            f"- blockers: `{', '.join(payload.get('blocking_layers') or []) or '-'}`",
            f"- done_when: `{text(payload.get('done_when'))}`",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build remote execution identity state for OpenClaw.")
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--now", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    reference_now = parse_now(args.now)
    handoff_path = find_latest(review_dir, "*_remote_live_handoff.json")
    contract_path = find_latest(review_dir, "*_remote_execution_contract_state.json")
    live_gate_path = find_latest(review_dir, "*_live_gate_blocker_report.json")
    history_latest = review_dir / "latest_remote_live_history_audit.json"
    history_path = history_latest if history_latest.exists() else find_latest(review_dir, "*_remote_live_history_audit.json")
    missing = [
        name
        for name, path in (
            ("remote_live_handoff", handoff_path),
            ("live_gate_blocker_report", live_gate_path),
        )
        if path is None
    ]
    if missing:
        raise SystemExit(f"missing_required_artifacts:{','.join(missing)}")

    payload = build_payload(
        handoff_path=handoff_path,
        handoff_payload=load_json_mapping(handoff_path),
        contract_path=contract_path,
        contract_payload=load_json_mapping(contract_path) if contract_path and contract_path.exists() else {},
        live_gate_path=live_gate_path,
        live_gate_payload=load_json_mapping(live_gate_path),
        history_path=history_path,
        history_payload=load_json_mapping(history_path) if history_path and history_path.exists() else {},
        reference_now=reference_now,
    )

    stamp = reference_now.strftime("%Y%m%dT%H%M%SZ")
    artifact = review_dir / f"{stamp}_remote_execution_identity_state.json"
    markdown = review_dir / f"{stamp}_remote_execution_identity_state.md"
    checksum = review_dir / f"{stamp}_remote_execution_identity_state_checksum.json"
    artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown.write_text(render_markdown(payload), encoding="utf-8")
    checksum.write_text(
        json.dumps(
            {
                "artifact": str(artifact),
                "artifact_sha256": sha256_file(artifact),
                "markdown": str(markdown),
                "markdown_sha256": sha256_file(markdown),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    pruned_keep, pruned_age = prune_artifacts(
        review_dir,
        current_paths=[artifact, markdown, checksum],
        keep=max(1, int(args.artifact_keep)),
        ttl_hours=max(1.0, float(args.artifact_ttl_hours)),
    )
    payload.update(
        {
            "artifact": str(artifact),
            "markdown": str(markdown),
            "checksum": str(checksum),
            "pruned_keep": pruned_keep,
            "pruned_age": pruned_age,
        }
    )
    artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checksum.write_text(
        json.dumps(
            {
                "artifact": str(artifact),
                "artifact_sha256": sha256_file(artifact),
                "markdown": str(markdown),
                "markdown_sha256": sha256_file(markdown),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
