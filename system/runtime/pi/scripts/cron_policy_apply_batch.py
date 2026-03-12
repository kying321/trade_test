#!/usr/bin/env python3
"""Apply cron policy patch draft by rollout batch.

Default behavior is non-destructive: generate a patched jobs candidate file and
an execution report. Source jobs.json is only updated when --apply is set.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lie_root_resolver import resolve_lie_system_root

SYSTEM_ROOT = resolve_lie_system_root()
DEFAULT_DRAFT = SYSTEM_ROOT / "output" / "review" / f"{dt.date.today().isoformat()}_cron_policy_patch_draft.json"
DEFAULT_JOBS = Path(os.path.expanduser("~/.openclaw/cron/jobs.json"))
DEFAULT_OUTPUT = SYSTEM_ROOT / "output" / "review" / f"{dt.date.today().isoformat()}_cron_policy_apply_batch.json"
DEFAULT_CANDIDATE = (
    SYSTEM_ROOT / "output" / "review" / f"{dt.date.today().isoformat()}_cron_jobs_batch_candidate.json"
)


def _read_json_dict(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"json_not_object:{path}")
    return obj


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _set_path(obj: Dict[str, Any], path: str, value: Any) -> Tuple[Any, bool]:
    parts = [p for p in str(path or "").strip().split(".") if p]
    if not parts:
        raise ValueError("empty_path")
    cur: Dict[str, Any] = obj
    for seg in parts[:-1]:
        nxt = cur.get(seg)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[seg] = nxt
        cur = nxt
    leaf = parts[-1]
    before = cur.get(leaf)
    changed = before != value
    if changed:
        cur[leaf] = value
    return before, changed


def _find_job_index(jobs: List[Dict[str, Any]], job_id: str, name: str) -> Tuple[Optional[int], str]:
    if job_id:
        for i, job in enumerate(jobs):
            if str(job.get("id") or "") == job_id:
                return i, "id"
    if name:
        by_name = [i for i, job in enumerate(jobs) if str(job.get("name") or "") == name]
        if len(by_name) == 1:
            return by_name[0], "name"
        if len(by_name) > 1:
            return None, "name_ambiguous"
    return None, "not_found"


def _normalize_batch_id(raw: str) -> str:
    text = str(raw or "").strip()
    if text.startswith("run_batch:"):
        return text.split(":", 1)[1].strip()
    return text


def _resolve_batch_id(draft_obj: Dict[str, Any], requested_batch: str) -> str:
    batch_id = _normalize_batch_id(requested_batch)
    if batch_id and batch_id != "auto":
        return batch_id

    isolation_rollout = (
        draft_obj.get("isolation_rollout") if isinstance(draft_obj.get("isolation_rollout"), dict) else {}
    )
    isolation_next = str(isolation_rollout.get("next_batch") or "").strip()
    if isolation_next:
        return isolation_next

    rollout = draft_obj.get("rollout_plan") if isinstance(draft_obj.get("rollout_plan"), dict) else {}
    batches = rollout.get("batches") if isinstance(rollout.get("batches"), list) else []
    for item in batches:
        if not isinstance(item, dict):
            continue
        try:
            count = int(item.get("count") or 0)
            ops = int(item.get("ops") or 0)
        except Exception:
            count = 0
            ops = 0
        if count > 0 and ops > 0:
            return str(item.get("id") or "").strip()
    return "batch_0_proxy_isolation"


def _collect_batch_plan_ids(draft_obj: Dict[str, Any], batch_id: str) -> List[str]:
    out: List[str] = []
    if batch_id == "batch_0_proxy_isolation":
        iso_rollout = (
            draft_obj.get("isolation_rollout") if isinstance(draft_obj.get("isolation_rollout"), dict) else {}
        )
        plan_ids = iso_rollout.get("plan_ids") if isinstance(iso_rollout.get("plan_ids"), list) else []
        out.extend(str(x).strip() for x in plan_ids if str(x).strip())
        if out:
            return list(dict.fromkeys(out))
        iso_plan = draft_obj.get("isolation_plan") if isinstance(draft_obj.get("isolation_plan"), dict) else {}
        recs = iso_plan.get("recommendations") if isinstance(iso_plan.get("recommendations"), list) else []
        for rec in recs:
            if not isinstance(rec, dict):
                continue
            if not bool(rec.get("actionable")):
                continue
            pid = str(rec.get("plan_id") or "").strip()
            if pid:
                out.append(pid)
        return list(dict.fromkeys(out))

    rollout = draft_obj.get("rollout_plan") if isinstance(draft_obj.get("rollout_plan"), dict) else {}
    batches = rollout.get("batches") if isinstance(rollout.get("batches"), list) else []
    for item in batches:
        if not isinstance(item, dict):
            continue
        if str(item.get("id") or "").strip() != batch_id:
            continue
        plan_ids = item.get("plan_ids") if isinstance(item.get("plan_ids"), list) else []
        out.extend(str(x).strip() for x in plan_ids if str(x).strip())
        break
    return list(dict.fromkeys(out))


def _select_actions(draft_obj: Dict[str, Any], batch_id: str, plan_ids: List[str]) -> List[Dict[str, Any]]:
    actions = draft_obj.get("actions") if isinstance(draft_obj.get("actions"), list) else []
    if not actions:
        return []
    if plan_ids:
        action_by_plan: Dict[str, Dict[str, Any]] = {}
        for action in actions:
            if not isinstance(action, dict):
                continue
            plan_id = str(action.get("plan_id") or "").strip()
            if not plan_id or plan_id in action_by_plan:
                continue
            action_by_plan[plan_id] = action
        selected: List[Dict[str, Any]] = []
        for plan_id in plan_ids:
            pid = str(plan_id or "").strip()
            if not pid:
                continue
            action = action_by_plan.get(pid)
            if not isinstance(action, dict):
                continue
            if int(action.get("operation_count") or 0) <= 0:
                continue
            selected.append(action)
        return selected
    if batch_id == "batch_0_proxy_isolation":
        return [
            a
            for a in actions
            if isinstance(a, dict)
            and bool(a.get("isolation_recommendation"))
            and int(a.get("operation_count") or 0) > 0
        ]
    return [
        a
        for a in actions
        if isinstance(a, dict)
        and str(a.get("rollout_batch") or "").strip() == batch_id
        and int(a.get("operation_count") or 0) > 0
    ]


def build_batch_candidate(
    draft_obj: Dict[str, Any],
    jobs_obj: Dict[str, Any],
    *,
    batch_id: str,
    allow_core_batch: bool,
    max_actions: int = 0,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    resolved_batch = _resolve_batch_id(draft_obj, batch_id)
    plan_ids = _collect_batch_plan_ids(draft_obj, resolved_batch)
    selected_actions = _select_actions(draft_obj, resolved_batch, plan_ids)
    if max_actions > 0:
        selected_actions = selected_actions[: max(0, int(max_actions))]

    jobs_copy = copy.deepcopy(jobs_obj)
    jobs_list = jobs_copy.get("jobs") if isinstance(jobs_copy.get("jobs"), list) else []

    action_results: List[Dict[str, Any]] = []
    matched_jobs = 0
    changed_jobs = 0
    failed_jobs = 0
    blocked_jobs = 0
    operations_total = 0
    operations_applied = 0
    operations_changed = 0

    for action in selected_actions:
        aid = str(action.get("id") or "").strip()
        aname = str(action.get("name") or "").strip()
        plan_id = str(action.get("plan_id") or "").strip() or None
        core_job = bool(action.get("core_job", False))
        changes = action.get("changes") if isinstance(action.get("changes"), list) else []
        result = {
            "plan_id": plan_id,
            "id": aid or None,
            "name": aname or None,
            "core_job": core_job,
            "operation_count": len(changes),
            "operation_applied_count": 0,
            "operation_changed_count": 0,
            "status": "ok",
            "errors": [],
        }

        if resolved_batch == "batch_3_core_guarded" and core_job and not allow_core_batch:
            result["status"] = "blocked"
            result["errors"] = ["core_batch_requires_allow_core_batch"]
            blocked_jobs += 1
            action_results.append(result)
            continue

        idx, source = _find_job_index(jobs_list, job_id=aid, name=aname)
        result["match_source"] = source
        if idx is None:
            result["status"] = "failed"
            result["errors"] = ["job_not_found" if source != "name_ambiguous" else "job_name_ambiguous"]
            failed_jobs += 1
            action_results.append(result)
            continue

        matched_jobs += 1
        job = jobs_list[idx]
        job_changed = False
        for item in changes:
            operations_total += 1
            if not isinstance(item, dict):
                result["status"] = "failed"
                result["errors"].append("invalid_change_item")
                continue
            path = str(item.get("path") or "").strip()
            if not path:
                result["status"] = "failed"
                result["errors"].append("missing_change_path")
                continue
            value = item.get("after")
            try:
                _, changed = _set_path(job, path=path, value=value)
            except Exception as e:
                result["status"] = "failed"
                result["errors"].append(f"set_failed:{type(e).__name__}")
                continue
            operations_applied += 1
            result["operation_applied_count"] = int(result["operation_applied_count"] or 0) + 1
            if changed:
                operations_changed += 1
                result["operation_changed_count"] = int(result["operation_changed_count"] or 0) + 1
                job_changed = True
        if result["status"] == "failed":
            failed_jobs += 1
        elif job_changed:
            changed_jobs += 1
        action_results.append(result)

    status = "noop"
    if selected_actions:
        status = "ok"
        if failed_jobs > 0 or blocked_jobs > 0:
            status = "degraded"

    report = {
        "envelope_version": "1.0",
        "domain": "cron_policy_batch_apply",
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": status,
        "batch_id": resolved_batch,
        "selected_plan_ids": plan_ids,
        "summary": {
            "selected_actions": len(selected_actions),
            "matched_jobs": matched_jobs,
            "changed_jobs": changed_jobs,
            "failed_jobs": failed_jobs,
            "blocked_jobs": blocked_jobs,
            "operations_total": operations_total,
            "operations_applied": operations_applied,
            "operations_changed": operations_changed,
        },
        "actions": action_results,
    }
    return report, jobs_copy


def run_apply_batch(
    *,
    draft_path: Path,
    jobs_path: Path,
    output_path: Path,
    candidate_jobs_output: Optional[Path],
    batch_id: str,
    apply: bool,
    allow_core_batch: bool,
    allow_degraded_apply: bool,
    max_actions: int,
    backup_path: Optional[Path] = None,
) -> Dict[str, Any]:
    draft_obj = _read_json_dict(draft_path)
    jobs_obj = _read_json_dict(jobs_path)
    report, candidate = build_batch_candidate(
        draft_obj,
        jobs_obj,
        batch_id=batch_id,
        allow_core_batch=allow_core_batch,
        max_actions=max_actions,
    )
    report["source"] = {"draft_path": str(draft_path), "jobs_path": str(jobs_path)}
    report["mode"] = "apply" if apply else "dry_run"
    report["allow_core_batch"] = bool(allow_core_batch)
    report["allow_degraded_apply"] = bool(allow_degraded_apply)
    report["applied_to_source"] = False
    report["apply_skipped_reason"] = None

    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    failed_jobs = int(summary.get("failed_jobs") or 0)
    blocked_jobs = int(summary.get("blocked_jobs") or 0)
    degraded = str(report.get("status") or "") == "degraded"

    if candidate_jobs_output is not None:
        _write_json_atomic(candidate_jobs_output, candidate)
        report["candidate_jobs_output"] = str(candidate_jobs_output)

    if apply:
        if blocked_jobs > 0:
            report["apply_skipped_reason"] = "blocked_jobs_present"
        elif degraded and failed_jobs > 0 and not allow_degraded_apply:
            report["apply_skipped_reason"] = "degraded_requires_allow_degraded_apply"
        else:
            raw_jobs_text = jobs_path.read_text(encoding="utf-8")
            backup = (
                backup_path
                if backup_path is not None
                else jobs_path.with_name(
                    f"{jobs_path.name}.bak.{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
            )
            backup.parent.mkdir(parents=True, exist_ok=True)
            backup.write_text(raw_jobs_text, encoding="utf-8")
            _write_json_atomic(jobs_path, candidate)
            report["applied_to_source"] = True
            report["backup_path"] = str(backup)

    _write_json_atomic(output_path, report)
    report["output_path"] = str(output_path)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft-path", default=str(DEFAULT_DRAFT))
    ap.add_argument("--jobs-path", default=str(DEFAULT_JOBS))
    ap.add_argument("--batch-id", default="auto")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--candidate-jobs-output", default=str(DEFAULT_CANDIDATE))
    ap.add_argument("--no-candidate-jobs-output", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--backup-path", default="")
    ap.add_argument("--allow-core-batch", action="store_true")
    ap.add_argument("--allow-degraded-apply", action="store_true")
    ap.add_argument("--max-actions", type=int, default=0)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    draft_path = Path(args.draft_path).expanduser().resolve()
    jobs_path = Path(args.jobs_path).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    candidate_path = (
        None
        if bool(args.no_candidate_jobs_output)
        else Path(args.candidate_jobs_output).expanduser().resolve()
    )
    backup_path = (
        Path(str(args.backup_path).strip()).expanduser().resolve()
        if str(args.backup_path).strip()
        else None
    )

    out = run_apply_batch(
        draft_path=draft_path,
        jobs_path=jobs_path,
        output_path=output_path,
        candidate_jobs_output=candidate_path,
        batch_id=str(args.batch_id or "auto"),
        apply=bool(args.apply),
        allow_core_batch=bool(args.allow_core_batch),
        allow_degraded_apply=bool(args.allow_degraded_apply),
        max_actions=max(0, int(args.max_actions or 0)),
        backup_path=backup_path,
    )
    print(json.dumps(out, ensure_ascii=False))
    if bool(args.strict):
        if str(out.get("status") or "") == "degraded":
            return 1
        if bool(args.apply) and not bool(out.get("applied_to_source")):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
