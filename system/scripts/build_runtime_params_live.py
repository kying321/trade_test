#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = "config.yaml"
DEFAULT_SIGNAL_CONFIDENCE_MIN = 60.0
DEFAULT_CONVEXITY_MIN = 3.0
DEFAULT_HOLD_DAYS = 5.0
DEFAULT_MAX_DAILY_TRADES = 2.0


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def parse_now(raw: str) -> datetime | None:
    text_value = str(raw or "").strip()
    if not text_value:
        return None
    try:
        parsed = datetime.fromisoformat(text_value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_date(raw: str) -> date | None:
    text_value = str(raw or "").strip()
    if not text_value:
        return None
    try:
        return date.fromisoformat(text_value)
    except ValueError:
        return None


def resolve_path(raw: str, *, anchor: Path) -> Path:
    path = Path(str(raw or "").strip())
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    return anchor / path


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def read_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def clamp_float(value: Any, lo: float, hi: float) -> float:
    candidate = to_float(value, lo)
    if candidate < lo:
        return float(lo)
    if candidate > hi:
        return float(hi)
    return float(candidate)


def clamp_int(value: Any, lo: int, hi: int) -> int:
    try:
        candidate = int(round(float(value)))
    except Exception:
        candidate = int(lo)
    if candidate < lo:
        return int(lo)
    if candidate > hi:
        return int(hi)
    return int(candidate)


def default_mode_profiles() -> dict[str, dict[str, float]]:
    return {
        "ultra_short": {
            "signal_confidence_min": 56.0,
            "convexity_min": 1.6,
            "hold_days": 2.0,
            "max_daily_trades": 4.0,
        },
        "swing": {
            "signal_confidence_min": 60.0,
            "convexity_min": 2.2,
            "hold_days": 8.0,
            "max_daily_trades": 2.0,
        },
        "long": {
            "signal_confidence_min": 54.0,
            "convexity_min": 2.8,
            "hold_days": 18.0,
            "max_daily_trades": 1.0,
        },
    }


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    return read_yaml_mapping(config_path)


def resolved_mode_profiles(config_payload: dict[str, Any]) -> dict[str, dict[str, float]]:
    profiles = default_mode_profiles()
    raw_profiles = config_payload.get("mode_profiles", {}) if isinstance(config_payload, dict) else {}
    if isinstance(raw_profiles, dict):
        for mode_name, values in raw_profiles.items():
            if mode_name not in profiles or not isinstance(values, dict):
                continue
            for key in ("signal_confidence_min", "convexity_min", "hold_days", "max_daily_trades"):
                if key in values:
                    profiles[mode_name][key] = float(values[key])
    return profiles


def load_latest_mode_feedback(output_root: Path, *, as_of: date | None) -> tuple[dict[str, Any], Path | None]:
    candidates: list[Path] = []
    if as_of is not None:
        candidates.extend(
            [
                output_root / "daily" / f"{as_of.isoformat()}_mode_feedback.json",
                output_root / "state" / "output" / "daily" / f"{as_of.isoformat()}_mode_feedback.json",
            ]
        )
    fallback_dir = output_root / "daily"
    if fallback_dir.exists():
        candidates.extend(sorted(fallback_dir.glob("*_mode_feedback.json"), reverse=True))
    state_fallback_dir = output_root / "state" / "output" / "daily"
    if state_fallback_dir.exists():
        candidates.extend(sorted(state_fallback_dir.glob("*_mode_feedback.json"), reverse=True))
    seen: set[Path] = set()
    for path in candidates:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        payload = read_json(path, {})
        if isinstance(payload, dict):
            return payload, path
    return {}, None


def resolve_runtime_params(*, config_payload: dict[str, Any], output_root: Path, as_of: date | None) -> dict[str, Any]:
    thresholds = config_payload.get("thresholds", {}) if isinstance(config_payload, dict) else {}
    validation = config_payload.get("validation", {}) if isinstance(config_payload, dict) else {}
    params_live_path = output_root / "artifacts" / "params_live.yaml"
    params_live = read_yaml_mapping(params_live_path) if params_live_path.exists() else {}
    mode_feedback, mode_feedback_path = load_latest_mode_feedback(output_root, as_of=as_of)

    base = {
        "signal_confidence_min": clamp_float(
            params_live.get("signal_confidence_min", thresholds.get("signal_confidence_min", DEFAULT_SIGNAL_CONFIDENCE_MIN)),
            20.0,
            95.0,
        ),
        "convexity_min": clamp_float(
            params_live.get("convexity_min", thresholds.get("convexity_min", DEFAULT_CONVEXITY_MIN)),
            0.5,
            5.0,
        ),
        "hold_days": float(
            clamp_int(params_live.get("hold_days", DEFAULT_HOLD_DAYS), 1, 20)
        ),
        "max_daily_trades": float(
            clamp_int(params_live.get("max_daily_trades", DEFAULT_MAX_DAILY_TRADES), 1, 5)
        ),
    }

    use_mode_profiles = bool(validation.get("use_mode_profiles", False))
    runtime_mode = str(mode_feedback.get("runtime_mode", "")).strip()
    profiles = resolved_mode_profiles(config_payload)
    blend = clamp_float(validation.get("mode_profile_blend_with_live", 0.55), 0.0, 1.0)

    if not use_mode_profiles:
        resolved = {"mode": "base"} | base
        status = "base_live_params"
        profile = {}
    else:
        if runtime_mode not in profiles:
            runtime_mode = "swing"
        profile = profiles.get(runtime_mode, profiles["swing"])
        resolved = {
            "mode": runtime_mode,
            "signal_confidence_min": clamp_float(
                (1.0 - blend) * float(base["signal_confidence_min"]) + blend * float(profile["signal_confidence_min"]),
                20.0,
                95.0,
            ),
            "convexity_min": clamp_float(
                (1.0 - blend) * float(base["convexity_min"]) + blend * float(profile["convexity_min"]),
                0.5,
                5.0,
            ),
            "hold_days": float(
                clamp_int((1.0 - blend) * float(base["hold_days"]) + blend * float(profile["hold_days"]), 1, 20)
            ),
            "max_daily_trades": float(
                clamp_int(
                    (1.0 - blend) * float(base["max_daily_trades"]) + blend * float(profile["max_daily_trades"]),
                    1,
                    5,
                )
            ),
        }
        status = "resolved_runtime_params_live"

    return {
        "generated_at_utc": now_utc().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "as_of": as_of.isoformat() if as_of is not None else "",
        "status": status,
        "source_kind": "runtime_params_live",
        "use_mode_profiles": bool(use_mode_profiles),
        "blend": float(blend),
        "runtime_mode": str(resolved.get("mode", "")),
        "signal_confidence_min": float(resolved["signal_confidence_min"]),
        "convexity_min": float(resolved["convexity_min"]),
        "hold_days": float(resolved["hold_days"]),
        "max_daily_trades": float(resolved["max_daily_trades"]),
        "base_live_params": base,
        "mode_profile": profile,
        "source_params_live": str(params_live_path) if params_live_path.exists() else "",
        "source_mode_feedback": str(mode_feedback_path) if mode_feedback_path is not None else "",
        "source_config": str(config_payload.get("__config_path__", "")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build source-owned runtime_params_live artifact from params_live and mode feedback.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--output-dir", default="output/review")
    parser.add_argument("--date", default="")
    parser.add_argument("--now", default="")
    args = parser.parse_args()

    system_root = Path(__file__).resolve().parents[1]
    config_path = resolve_path(args.config, anchor=system_root)
    output_root = resolve_path(args.output_root, anchor=system_root)
    output_dir = resolve_path(args.output_dir, anchor=system_root)
    reference_now = parse_now(args.now) or now_utc()
    as_of = parse_date(args.date)

    config_payload = load_config(config_path)
    config_payload["__config_path__"] = str(config_path)
    payload = resolve_runtime_params(config_payload=config_payload, output_root=output_root, as_of=as_of)
    payload["generated_at_utc"] = reference_now.strftime("%Y-%m-%dT%H:%M:%SZ")

    artifacts_path = output_root / "artifacts" / "runtime_params_live.json"
    stamped_path = output_dir / f"{reference_now.strftime('%Y%m%dT%H%M%SZ')}_runtime_params_live.json"
    write_json(artifacts_path, payload)
    write_json(stamped_path, payload)
    print(json.dumps({"artifact": str(artifacts_path), "review": str(stamped_path), "runtime_mode": payload["runtime_mode"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
