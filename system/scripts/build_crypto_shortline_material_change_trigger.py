#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
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
TIMESTAMP_RE = re.compile(r"^(?P<stamp>\d{8}T\d{6}Z)_")


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse_now(raw: str | None) -> dt.datetime:
    text_value = str(raw or "").strip()
    if not text_value:
        return now_utc()
    parsed = dt.datetime.fromisoformat(text_value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def fmt_utc(value: dt.datetime | None) -> str:
    effective = value or now_utc()
    return effective.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def text(value: Any) -> str:
    return str(value or "").strip()


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def dedupe_text(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        item = text(raw)
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def join_unique(parts: list[Any], *, sep: str = " | ") -> str:
    return sep.join(dedupe_text(parts))


def load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def artifact_stamp(path: Path) -> str:
    match = TIMESTAMP_RE.match(path.name)
    return match.group("stamp") if match else ""


def parse_artifact_stamp(path: Path) -> dt.datetime | None:
    stamp = artifact_stamp(path)
    if not stamp:
        return None
    try:
        return dt.datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def find_latest(review_dir: Path, pattern: str) -> Path | None:
    files = list(review_dir.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda item: (artifact_stamp(item), item.stat().st_mtime, item.name))


def find_latest_at_or_before(
    review_dir: Path,
    pattern: str,
    cutoff: dt.datetime,
) -> Path | None:
    candidates: list[Path] = []
    for path in review_dir.glob(pattern):
        stamp = parse_artifact_stamp(path)
        if stamp is None:
            try:
                mtime = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)
            except OSError:
                continue
            if mtime <= cutoff:
                candidates.append(path)
            continue
        if stamp <= cutoff:
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda item: (artifact_stamp(item), item.stat().st_mtime, item.name))


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
        "*_crypto_shortline_material_change_trigger.json",
        "*_crypto_shortline_material_change_trigger.md",
        "*_crypto_shortline_material_change_trigger_checksum.json",
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


def find_gate_row(payload: dict[str, Any], symbol: str) -> dict[str, Any]:
    for row in as_list(payload.get("symbols")):
        item = as_dict(row)
        if text(item.get("symbol")).upper() == symbol.upper():
            return item
    return {}


def find_batch_row(payload: dict[str, Any], symbol: str) -> dict[str, Any]:
    for key in ("runtime_queue", "batch_runtime_profiles"):
        for row in as_list(payload.get(key)):
            item = as_dict(row)
            eligible = {text(x).upper() for x in as_list(item.get("eligible_symbols"))}
            if symbol.upper() in eligible:
                return item
    return {}


def find_matching_symbol(batch_row: dict[str, Any], symbol: str) -> dict[str, Any]:
    for row in as_list(batch_row.get("matching_symbols")):
        item = as_dict(row)
        if text(item.get("symbol")).upper() == symbol.upper():
            return item
    return {}


def gate_signature(payload: dict[str, Any], symbol: str) -> dict[str, Any]:
    row = find_gate_row(payload, symbol)
    structure = as_dict(row.get("structure_signals"))
    micro = as_dict(row.get("micro_signals"))
    return {
        "execution_state": text(row.get("execution_state")),
        "route_state": text(row.get("route_state")),
        "missing_gates": dedupe_text(as_list(row.get("missing_gates"))),
        "structure_signals": {
            "sweep_long": bool(structure.get("sweep_long", False)),
            "sweep_short": bool(structure.get("sweep_short", False)),
            "mss_long": bool(structure.get("mss_long", False)),
            "mss_short": bool(structure.get("mss_short", False)),
            "fvg_long": bool(structure.get("fvg_long", False)),
            "fvg_short": bool(structure.get("fvg_short", False)),
        },
        "micro_signals": {
            "cvd_ready": bool(micro.get("cvd_ready", False)),
            "quality_ok": bool(micro.get("quality_ok", False)),
            "trust_ok": bool(micro.get("trust_ok", False)),
            "context": text(micro.get("context")),
            "veto_hint": text(micro.get("veto_hint")),
            "attack_side": text(micro.get("attack_side")),
            "attack_presence": text(micro.get("attack_presence")),
            "attack_confirmation_ok": bool(micro.get("attack_confirmation_ok", False)),
        },
        "blocker_detail": text(row.get("blocker_detail")),
    }


def operator_signature(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "focus_review_status": text(payload.get("focus_review_status")),
        "next_focus_action": text(payload.get("next_focus_action")),
        "review_priority_head_symbol": text(payload.get("review_priority_head_symbol")),
        "focus_review_brief": text(payload.get("focus_review_brief")),
    }


def queue_signature(payload: dict[str, Any], symbol: str) -> dict[str, Any]:
    batch = find_batch_row(payload, symbol)
    match = find_matching_symbol(batch, symbol)
    return {
        "queue_status": text(payload.get("queue_status")),
        "semantic_status": text(payload.get("semantic_status")),
        "next_focus_batch": text(payload.get("next_focus_batch")),
        "next_focus_action": text(payload.get("next_focus_action")),
        "classification": text(match.get("classification")),
        "cvd_context_mode": text(match.get("cvd_context_mode")),
        "cvd_veto_hint": text(match.get("cvd_veto_hint")),
        "cvd_attack_side": text(match.get("cvd_attack_side")),
        "active_reasons": dedupe_text(as_list(match.get("active_reasons"))),
    }


def compare_signatures(
    *,
    baseline_gate: dict[str, Any],
    current_gate: dict[str, Any],
    baseline_operator: dict[str, Any],
    current_operator: dict[str, Any],
    baseline_queue: dict[str, Any],
    current_queue: dict[str, Any],
) -> list[dict[str, Any]]:
    comparisons = (
        ("execution_state", baseline_gate.get("execution_state"), current_gate.get("execution_state")),
        ("route_state", baseline_gate.get("route_state"), current_gate.get("route_state")),
        ("missing_gates", baseline_gate.get("missing_gates"), current_gate.get("missing_gates")),
        (
            "structure_signals",
            baseline_gate.get("structure_signals"),
            current_gate.get("structure_signals"),
        ),
        (
            "micro_signals",
            baseline_gate.get("micro_signals"),
            current_gate.get("micro_signals"),
        ),
        (
            "focus_review_status",
            baseline_operator.get("focus_review_status"),
            current_operator.get("focus_review_status"),
        ),
        (
            "queue_focus",
            baseline_queue,
            current_queue,
        ),
    )
    changes: list[dict[str, Any]] = []
    for name, before, after in comparisons:
        if before == after:
            continue
        changes.append({"dimension": name, "before": before, "after": after})
    return changes


def change_summary(changes: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for change in changes:
        name = text(change.get("dimension"))
        before = change.get("before")
        after = change.get("after")
        if isinstance(before, list) and isinstance(after, list):
            out.append(f"{name}:{','.join(before) or '-'}->{','.join(after) or '-'}")
        elif isinstance(before, dict) and isinstance(after, dict):
            out.append(name)
        else:
            out.append(f"{name}:{text(before) or '-'}->{text(after) or '-'}")
    return out


def render_markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Crypto Shortline Material Change Trigger",
            "",
            f"- brief: `{text(payload.get('trigger_brief'))}`",
            f"- status: `{text(payload.get('trigger_status'))}`",
            f"- decision: `{text(payload.get('trigger_decision'))}`",
            f"- rerun_recommended: `{bool(payload.get('rerun_recommended', False))}`",
            f"- route_symbol: `{text(payload.get('route_symbol'))}`",
            f"- anchor_backtest_brief: `{text(payload.get('anchor_backtest_brief'))}`",
            f"- changed_dimensions: `{','.join(as_list(payload.get('changed_dimensions'))) or '-'}`",
            f"- change_summary: `{join_unique(as_list(payload.get('change_summary')), sep=' ; ') or '-'}`",
            f"- blocker_detail: `{text(payload.get('blocker_detail'))}`",
            f"- done_when: `{text(payload.get('done_when'))}`",
            "",
        ]
    )


def build_payload(
    *,
    cross_section_backtest_path: Path | None,
    cross_section_backtest_payload: dict[str, Any],
    baseline_gate_path: Path | None,
    baseline_gate_payload: dict[str, Any],
    current_gate_path: Path | None,
    current_gate_payload: dict[str, Any],
    baseline_operator_path: Path | None,
    baseline_operator_payload: dict[str, Any],
    current_operator_path: Path | None,
    current_operator_payload: dict[str, Any],
    baseline_queue_path: Path | None,
    baseline_queue_payload: dict[str, Any],
    current_queue_path: Path | None,
    current_queue_payload: dict[str, Any],
    reference_now: dt.datetime,
) -> dict[str, Any]:
    route_symbol = (
        text(cross_section_backtest_payload.get("selected_symbol"))
        or text(current_operator_payload.get("review_priority_head_symbol"))
        or text(current_operator_payload.get("next_focus_symbol"))
    ).upper()
    route_action = text(current_operator_payload.get("next_focus_action")) or text(
        current_queue_payload.get("next_focus_action")
    )
    anchor_brief = text(cross_section_backtest_payload.get("backtest_brief"))
    anchor_status = text(cross_section_backtest_payload.get("backtest_status"))
    anchor_decision = text(cross_section_backtest_payload.get("research_decision"))
    anchor_edge = text(cross_section_backtest_payload.get("selected_edge_status"))

    if cross_section_backtest_path is None or not route_symbol:
        trigger_status = "material_change_anchor_missing"
        trigger_decision = "build_cross_section_anchor_before_material_change_trigger"
        blocker_title = "Build a shortline backtest anchor before evaluating material change"
        blocker_detail = join_unique(
            [
                "cross_section_backtest_missing" if cross_section_backtest_path is None else "",
                "route_symbol_missing" if not route_symbol else "",
            ]
        )
        done_when = "latest crypto shortline cross-section backtest exists with a selected symbol anchor"
        rerun_recommended = False
        changes: list[dict[str, Any]] = []
    elif current_gate_path is None or baseline_gate_path is None:
        trigger_status = "material_change_anchor_incomplete"
        trigger_decision = "collect_execution_gate_anchor_before_material_change_trigger"
        blocker_title = "Collect a comparable execution gate anchor before evaluating material change"
        blocker_detail = join_unique(
            [
                "current_execution_gate_missing" if current_gate_path is None else "",
                "baseline_execution_gate_missing" if baseline_gate_path is None else "",
                anchor_brief,
            ]
        )
        done_when = "baseline and current crypto shortline execution gate artifacts both exist around the backtest anchor"
        rerun_recommended = False
        changes = []
    else:
        baseline_gate = gate_signature(baseline_gate_payload, route_symbol)
        current_gate = gate_signature(current_gate_payload, route_symbol)
        baseline_operator = operator_signature(baseline_operator_payload)
        current_operator = operator_signature(current_operator_payload)
        baseline_queue = queue_signature(baseline_queue_payload, route_symbol)
        current_queue = queue_signature(current_queue_payload, route_symbol)
        changes = compare_signatures(
            baseline_gate=baseline_gate,
            current_gate=current_gate,
            baseline_operator=baseline_operator,
            current_operator=current_operator,
            baseline_queue=baseline_queue,
            current_queue=current_queue,
        )
        rerun_recommended = bool(changes)
        if rerun_recommended:
            trigger_status = "material_orderflow_change_detected"
            trigger_decision = "rerun_shortline_execution_gate_and_recheck_ticket_actionability"
            blocker_title = "Material orderflow change detected; refresh execution gate"
            blocker_detail = join_unique(
                [
                    anchor_brief,
                    f"changes={join_unique(change_summary(changes), sep=',')}",
                ]
            )
            done_when = (
                f"{route_symbol} refreshes the execution gate after the detected material change and ticket actionability is re-evaluated"
            )
        else:
            trigger_status = "no_material_orderflow_change_since_cross_section_anchor"
            trigger_decision = "wait_for_material_orderflow_change_before_rerun"
            blocker_title = "Wait for material orderflow change before refreshing execution gate"
            blocker_detail = join_unique(
                [
                    anchor_brief,
                    f"current_gate={text(current_gate.get('execution_state'))}:{','.join(as_list(current_gate.get('missing_gates')))}",
                    "no_delta_since_anchor",
                ]
            )
            done_when = (
                f"{route_symbol} shows any delta in execution_state, route_state, missing_gates, focus_review_status, or CVD queue posture relative to the cross-section anchor"
            )

    trigger_brief = ":".join(
        [
            trigger_status,
            route_symbol or "-",
            trigger_decision,
            route_action or "-",
        ]
    )
    return {
        "action": "build_crypto_shortline_material_change_trigger",
        "ok": True,
        "status": "ok",
        "generated_at_utc": fmt_utc(reference_now),
        "route_symbol": route_symbol,
        "route_action": route_action,
        "anchor_backtest_brief": anchor_brief,
        "anchor_backtest_status": anchor_status,
        "anchor_backtest_decision": anchor_decision,
        "anchor_selected_edge_status": anchor_edge,
        "trigger_status": trigger_status,
        "trigger_brief": trigger_brief,
        "trigger_decision": trigger_decision,
        "rerun_recommended": rerun_recommended,
        "blocker_title": blocker_title,
        "blocker_detail": blocker_detail,
        "done_when": done_when,
        "changed_dimensions": [text(row.get("dimension")) for row in changes],
        "change_summary": change_summary(changes),
        "anchor_backtest_artifact": str(cross_section_backtest_path) if cross_section_backtest_path else "",
        "baseline_execution_gate_artifact": str(baseline_gate_path) if baseline_gate_path else "",
        "current_execution_gate_artifact": str(current_gate_path) if current_gate_path else "",
        "baseline_route_operator_artifact": str(baseline_operator_path) if baseline_operator_path else "",
        "current_route_operator_artifact": str(current_operator_path) if current_operator_path else "",
        "baseline_cvd_queue_artifact": str(baseline_queue_path) if baseline_queue_path else "",
        "current_cvd_queue_artifact": str(current_queue_path) if current_queue_path else "",
        "artifacts": {
            "crypto_shortline_cross_section_backtest": str(cross_section_backtest_path)
            if cross_section_backtest_path
            else "",
            "baseline_crypto_shortline_execution_gate": str(baseline_gate_path)
            if baseline_gate_path
            else "",
            "current_crypto_shortline_execution_gate": str(current_gate_path)
            if current_gate_path
            else "",
            "baseline_crypto_route_operator_brief": str(baseline_operator_path)
            if baseline_operator_path
            else "",
            "current_crypto_route_operator_brief": str(current_operator_path)
            if current_operator_path
            else "",
            "baseline_crypto_cvd_queue_handoff": str(baseline_queue_path) if baseline_queue_path else "",
            "current_crypto_cvd_queue_handoff": str(current_queue_path) if current_queue_path else "",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a source-owned trigger for rerunning crypto shortline execution gate on material orderflow change."
    )
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
    cross_section_backtest_path = find_latest(
        review_dir, "*_crypto_shortline_cross_section_backtest.json"
    )
    cross_section_backtest_payload = (
        load_json_mapping(cross_section_backtest_path)
        if cross_section_backtest_path is not None and cross_section_backtest_path.exists()
        else {}
    )
    anchor_time = parse_artifact_stamp(cross_section_backtest_path) if cross_section_backtest_path else None
    baseline_gate_path = (
        find_latest_at_or_before(review_dir, "*_crypto_shortline_execution_gate.json", anchor_time)
        if anchor_time is not None
        else None
    )
    baseline_operator_path = (
        find_latest_at_or_before(review_dir, "*_crypto_route_operator_brief.json", anchor_time)
        if anchor_time is not None
        else None
    )
    baseline_queue_path = (
        find_latest_at_or_before(review_dir, "*_crypto_cvd_queue_handoff.json", anchor_time)
        if anchor_time is not None
        else None
    )
    current_gate_path = find_latest(review_dir, "*_crypto_shortline_execution_gate.json")
    current_operator_path = find_latest(review_dir, "*_crypto_route_operator_brief.json")
    current_queue_path = find_latest(review_dir, "*_crypto_cvd_queue_handoff.json")

    payload = build_payload(
        cross_section_backtest_path=cross_section_backtest_path,
        cross_section_backtest_payload=cross_section_backtest_payload,
        baseline_gate_path=baseline_gate_path,
        baseline_gate_payload=load_json_mapping(baseline_gate_path)
        if baseline_gate_path is not None and baseline_gate_path.exists()
        else {},
        current_gate_path=current_gate_path,
        current_gate_payload=load_json_mapping(current_gate_path)
        if current_gate_path is not None and current_gate_path.exists()
        else {},
        baseline_operator_path=baseline_operator_path,
        baseline_operator_payload=load_json_mapping(baseline_operator_path)
        if baseline_operator_path is not None and baseline_operator_path.exists()
        else {},
        current_operator_path=current_operator_path,
        current_operator_payload=load_json_mapping(current_operator_path)
        if current_operator_path is not None and current_operator_path.exists()
        else {},
        baseline_queue_path=baseline_queue_path,
        baseline_queue_payload=load_json_mapping(baseline_queue_path)
        if baseline_queue_path is not None and baseline_queue_path.exists()
        else {},
        current_queue_path=current_queue_path,
        current_queue_payload=load_json_mapping(current_queue_path)
        if current_queue_path is not None and current_queue_path.exists()
        else {},
        reference_now=reference_now,
    )

    stamp = reference_now.strftime("%Y%m%dT%H%M%SZ")
    artifact = review_dir / f"{stamp}_crypto_shortline_material_change_trigger.json"
    markdown = review_dir / f"{stamp}_crypto_shortline_material_change_trigger.md"
    checksum = review_dir / f"{stamp}_crypto_shortline_material_change_trigger_checksum.json"
    payload["artifact"] = str(artifact)
    payload["markdown"] = str(markdown)
    payload["checksum"] = str(checksum)

    artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown.write_text(render_markdown(payload), encoding="utf-8")
    checksum.write_text(
        json.dumps(
            {
                "artifact": str(artifact),
                "sha256": sha256_file(artifact),
                "generated_at_utc": fmt_utc(reference_now),
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
        keep=args.artifact_keep,
        ttl_hours=args.artifact_ttl_hours,
    )
    payload["pruned_keep"] = pruned_keep
    payload["pruned_age"] = pruned_age
    artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
