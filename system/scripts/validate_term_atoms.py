#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import yaml


REQUIRED_TOP_KEYS = {"version", "truth_baseline", "atoms"}
REQUIRED_BASELINE_KEYS = {"l2_min_fields", "trade_min_fields", "time_sync", "gap_fuse", "drop_rate", "retention_days"}
REQUIRED_ATOM_KEYS = {
    "id",
    "family",
    "aliases",
    "objective",
    "formula_type",
    "feature_formula",
    "invalidation",
    "metrics",
    "input_fields",
    "status",
}
ALLOWED_FAMILY = {"ict", "al_brooks", "lie_pdf"}
ALLOWED_FORMULA_TYPES = {"proxy_ohlcv", "hybrid", "l2_native"}
ALLOWED_STATUS = {"active", "proxy", "pending"}
REQUIRED_TIME_SYNC_KEYS = {"ntp_max_offset_ms", "cross_source_tolerance_ms", "max_acceptable_rtt_ms"}
REQUIRED_GAP_KEYS = {"continuous_gap_ms", "consecutive_gaps", "stale_stream_ms"}
REQUIRED_DROP_KEYS = {"warn_pct", "fuse_pct"}
REQUIRED_RETENTION_KEYS = {"l2", "trades", "news_labels"}


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Registry root must be a mapping.")
    return payload


def _validate_metric(metric: Any, atom_id: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(metric, dict):
        return [f"{atom_id}: metric must be mapping"]
    for key in ("name", "direction", "eval_window"):
        if key not in metric:
            errors.append(f"{atom_id}: metric missing key `{key}`")
    if str(metric.get("direction", "")) not in {"higher_better", "lower_better"}:
        errors.append(f"{atom_id}: metric.direction must be higher_better or lower_better")
    return errors


def validate_registry(payload: dict[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    errors: list[str] = []
    for key in sorted(REQUIRED_TOP_KEYS - set(payload.keys())):
        errors.append(f"missing top-level key `{key}`")

    truth = payload.get("truth_baseline")
    if not isinstance(truth, dict):
        errors.append("truth_baseline must be mapping")
        truth = {}
    else:
        for key in sorted(REQUIRED_BASELINE_KEYS - set(truth.keys())):
            errors.append(f"truth_baseline missing `{key}`")

    time_sync = truth.get("time_sync", {})
    if isinstance(time_sync, dict):
        for key in sorted(REQUIRED_TIME_SYNC_KEYS - set(time_sync.keys())):
            errors.append(f"time_sync missing `{key}`")
    else:
        errors.append("time_sync must be mapping")
        time_sync = {}

    gap_fuse = truth.get("gap_fuse", {})
    if isinstance(gap_fuse, dict):
        for key in sorted(REQUIRED_GAP_KEYS - set(gap_fuse.keys())):
            errors.append(f"gap_fuse missing `{key}`")
    else:
        errors.append("gap_fuse must be mapping")
        gap_fuse = {}

    drop_rate = truth.get("drop_rate", {})
    if isinstance(drop_rate, dict):
        for key in sorted(REQUIRED_DROP_KEYS - set(drop_rate.keys())):
            errors.append(f"drop_rate missing `{key}`")
    else:
        errors.append("drop_rate must be mapping")
        drop_rate = {}

    retention = truth.get("retention_days", {})
    if isinstance(retention, dict):
        for key in sorted(REQUIRED_RETENTION_KEYS - set(retention.keys())):
            errors.append(f"retention_days missing `{key}`")
    else:
        errors.append("retention_days must be mapping")

    if isinstance(time_sync, dict):
        if float(time_sync.get("ntp_max_offset_ms", 999999.0)) > 5.0:
            errors.append("time_sync.ntp_max_offset_ms must be <= 5")
        if float(time_sync.get("cross_source_tolerance_ms", 0.0)) <= 0.0:
            errors.append("time_sync.cross_source_tolerance_ms must be > 0")

    if isinstance(drop_rate, dict):
        warn = float(drop_rate.get("warn_pct", -1.0))
        fuse = float(drop_rate.get("fuse_pct", -1.0))
        if warn < 0.0 or fuse < 0.0:
            errors.append("drop_rate values must be >= 0")
        if warn > fuse:
            errors.append("drop_rate.warn_pct must be <= drop_rate.fuse_pct")

    atoms = payload.get("atoms")
    if not isinstance(atoms, list):
        errors.append("atoms must be a list")
        atoms = []

    ids: set[str] = set()
    alias_seen: set[str] = set()
    family_counter: dict[str, int] = {}
    formula_counter: dict[str, int] = {}
    status_counter: dict[str, int] = {}
    l2_atom_count = 0

    for i, atom in enumerate(atoms):
        if not isinstance(atom, dict):
            errors.append(f"atoms[{i}] must be mapping")
            continue
        for key in sorted(REQUIRED_ATOM_KEYS - set(atom.keys())):
            errors.append(f"atoms[{i}] missing `{key}`")
        atom_id = str(atom.get("id", f"atoms[{i}]"))
        if atom_id in ids:
            errors.append(f"duplicate atom id `{atom_id}`")
        ids.add(atom_id)

        family = str(atom.get("family", ""))
        if family not in ALLOWED_FAMILY:
            errors.append(f"{atom_id}: unsupported family `{family}`")
        family_counter[family] = family_counter.get(family, 0) + 1

        formula_type = str(atom.get("formula_type", ""))
        if formula_type not in ALLOWED_FORMULA_TYPES:
            errors.append(f"{atom_id}: unsupported formula_type `{formula_type}`")
        formula_counter[formula_type] = formula_counter.get(formula_type, 0) + 1
        if formula_type == "l2_native":
            l2_atom_count += 1

        status = str(atom.get("status", ""))
        if status not in ALLOWED_STATUS:
            errors.append(f"{atom_id}: unsupported status `{status}`")
        status_counter[status] = status_counter.get(status, 0) + 1

        aliases = atom.get("aliases", [])
        if not isinstance(aliases, list) or not aliases:
            errors.append(f"{atom_id}: aliases must be non-empty list")
        else:
            for alias in aliases:
                alias_str = str(alias).strip().lower()
                if not alias_str:
                    errors.append(f"{atom_id}: alias cannot be empty")
                    continue
                if alias_str in alias_seen:
                    errors.append(f"{atom_id}: duplicate alias `{alias_str}` across registry")
                alias_seen.add(alias_str)

        metrics = atom.get("metrics", [])
        if not isinstance(metrics, list) or not metrics:
            errors.append(f"{atom_id}: metrics must be non-empty list")
        else:
            for m in metrics:
                errors.extend(_validate_metric(m, atom_id))

        invalidation = atom.get("invalidation", [])
        if not isinstance(invalidation, list) or not invalidation:
            errors.append(f"{atom_id}: invalidation must be non-empty list")

        inputs = atom.get("input_fields", [])
        if not isinstance(inputs, list) or not inputs:
            errors.append(f"{atom_id}: input_fields must be non-empty list")

        if not str(atom.get("feature_formula", "")).strip():
            errors.append(f"{atom_id}: feature_formula cannot be empty")
        if not str(atom.get("objective", "")).strip():
            errors.append(f"{atom_id}: objective cannot be empty")

    for family in sorted(ALLOWED_FAMILY):
        if family_counter.get(family, 0) == 0:
            errors.append(f"family `{family}` has zero atoms")
    if l2_atom_count == 0:
        errors.append("at least one l2_native atom is required")

    summary = {
        "ok": len(errors) == 0,
        "version": str(payload.get("version", "")),
        "atoms_total": len(atoms),
        "families": family_counter,
        "formula_types": formula_counter,
        "status": status_counter,
        "l2_atoms": l2_atom_count,
        "errors": errors,
    }
    return len(errors) == 0, errors, summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate term atom registry schema and minimum coverage constraints.")
    parser.add_argument(
        "--path",
        default="config/term_atoms.yaml",
        help="Path to term atom registry yaml, relative to system/ by default.",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional output file path for JSON summary.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = Path(__file__).resolve().parents[1]
    target = Path(args.path)
    if not target.is_absolute():
        target = root / target
    if not target.exists():
        print(json.dumps({"ok": False, "errors": [f"file not found: {target}"]}, ensure_ascii=False, indent=2))
        return 2

    try:
        payload = _load_yaml(target)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "errors": [f"yaml load error: {exc}"]}, ensure_ascii=False, indent=2))
        return 2

    ok, _, summary = validate_registry(payload)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(rendered)

    if args.json_out:
        out = Path(args.json_out)
        if not out.is_absolute():
            out = root / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered + "\n", encoding="utf-8")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
