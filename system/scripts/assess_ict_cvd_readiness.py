#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

from lie_engine.data.factory import SUPPORTED_PROVIDER_PROFILES, build_provider_stack


def resolve_system_root() -> Path:
    env_root = str(os.getenv("LIE_SYSTEM_ROOT", "")).strip() or str(
        os.getenv("FENLIE_SYSTEM_ROOT", "")
    ).strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


SYSTEM_ROOT = resolve_system_root()
DEFAULT_CONFIG = SYSTEM_ROOT / "config.yaml"
DEFAULT_SPEC = SYSTEM_ROOT / "config" / "ict_cvd_factor_spec.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    payload = yaml.safe_load(text) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping at {path}")
    return payload


def _truthy(value: Any) -> bool:
    return bool(value)


def _get_nested(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _provider_summary(profile: str) -> dict[str, Any]:
    providers = build_provider_stack(profile)
    classes = [type(p).__name__ for p in providers]
    public_trade_side = any(name in {"BinanceSpotPublicProvider", "BybitSpotPublicProvider"} for name in classes)
    public_l2_snapshot = public_trade_side
    dual_public = {
        "BinanceSpotPublicProvider",
        "BybitSpotPublicProvider",
    }.issubset(set(classes))
    opensource_only = set(classes).issubset({"OpenSourcePrimaryProvider", "OpenSourceSecondaryProvider"})
    cvd_lite_ready = bool(public_trade_side and public_l2_snapshot)
    readiness_label = "cvd-unavailable"
    if cvd_lite_ready and dual_public:
        readiness_label = "cvd-lite-ready"
    elif cvd_lite_ready:
        readiness_label = "cvd-lite-partial"
    elif opensource_only:
        readiness_label = "cvd-unavailable"
    return {
        "profile": profile,
        "provider_classes": classes,
        "public_trade_side_available": public_trade_side,
        "public_l2_snapshot_available": public_l2_snapshot,
        "dual_public_cross_check_available": dual_public,
        "strict_cvd_ready": False,
        "cvd_lite_ready": cvd_lite_ready,
        "readiness_label": readiness_label,
        "limitations": [
            "No persistent websocket diff-depth local book reconstruction in current stack.",
            "No replayable full-session order-flow store.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assess whether the current Fenlie stack can support ICT+CVD.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--spec", default=str(DEFAULT_SPEC))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = Path(args.config).expanduser().resolve()
    spec_path = Path(args.spec).expanduser().resolve()

    config_payload = _load_yaml(config_path)
    spec_payload = _load_yaml(spec_path)

    data_cfg = _get_nested(config_payload, "data", default={}) or {}
    validation_cfg = _get_nested(config_payload, "validation", default={}) or {}
    profile = str(data_cfg.get("provider_profile") or "opensource_dual").strip().lower()
    if profile not in SUPPORTED_PROVIDER_PROFILES:
        raise ValueError(f"Unsupported data.provider_profile: {profile!r}")

    provider_assessment = _provider_summary(profile)
    supported_profiles = set(_get_nested(spec_payload, "integration", "supported_profiles", default=[]) or [])
    unsupported_profiles = set(_get_nested(spec_payload, "integration", "unsupported_profiles", default=[]) or [])

    micro_cross_source = _truthy(validation_cfg.get("micro_cross_source_audit_enabled"))
    micro_trade_count = int(validation_cfg.get("micro_min_trade_count", 0) or 0)
    time_sync_required = bool(_get_nested(spec_payload, "data_requirements", "gates", "time_sync_ok_required", default=True))
    trade_count_floor = int(_get_nested(spec_payload, "data_requirements", "gates", "trade_count_min_floor", default=20))

    reasons: list[str] = []
    warnings: list[str] = []
    cvd_lite_ready = bool(provider_assessment["cvd_lite_ready"])

    if profile not in supported_profiles:
        cvd_lite_ready = False
        reasons.append(f"profile_not_supported:{profile}")
    if profile in unsupported_profiles:
        cvd_lite_ready = False
        reasons.append(f"profile_explicitly_unsupported:{profile}")
    if not micro_cross_source:
        warnings.append("micro_cross_source_audit_disabled")
    if micro_trade_count < trade_count_floor:
        warnings.append(
            f"micro_trade_count_below_floor(config={micro_trade_count},floor={trade_count_floor})"
        )
    if not provider_assessment["dual_public_cross_check_available"]:
        warnings.append("dual_public_cross_check_unavailable")
    if not time_sync_required:
        warnings.append("time_sync_gate_disabled")

    readiness_label = "cvd-unavailable"
    if cvd_lite_ready and provider_assessment["dual_public_cross_check_available"]:
        readiness_label = "cvd-lite-ready"
    elif cvd_lite_ready:
        readiness_label = "cvd-lite-partial"

    current_profile_assessment = {
        "profile": profile,
        "cvd_lite_ready": cvd_lite_ready,
        "strict_cvd_ready": False,
        "readiness_label": readiness_label,
        "reasons": reasons,
        "warnings": warnings,
        "micro_cross_source_audit_enabled": micro_cross_source,
        "micro_min_trade_count": micro_trade_count,
        "trade_count_floor": trade_count_floor,
        "time_sync_required": time_sync_required,
        "rollout_recommendation": (
            "Integrate ICT+CVD as confirm-and-veto only."
            if cvd_lite_ready
            else "Do not enable ICT+CVD until a public-spot trade/L2 profile is active."
        ),
    }

    profile_matrix = [_provider_summary(p) for p in sorted(SUPPORTED_PROVIDER_PROFILES)]

    payload = {
        "action": "assess_ict_cvd_readiness",
        "config_path": str(config_path),
        "spec_path": str(spec_path),
        "current_profile": profile,
        "current_profile_assessment": current_profile_assessment,
        "provider_assessment": provider_assessment,
        "profile_matrix": profile_matrix,
        "factor_ids": [
            item["id"]
            for item in (spec_payload.get("factors") or [])
            if isinstance(item, dict) and item.get("id")
        ],
        "data_sufficiency_conclusion": {
            "cvd_lite_supported_now": cvd_lite_ready,
            "strict_cvd_supported_now": False,
            "summary": (
                "Fenlie can support ICT+CVD-lite confirmation factors now."
                if cvd_lite_ready
                else "Current config does not support usable ICT+CVD-lite."
            ),
            "strict_gap": [
                "No websocket diff-depth local book reconstruction.",
                "No replayable long-session order-flow store.",
                "Absorption and failed-auction logic must remain proxy-grade.",
            ],
        },
        "repo_evidence": {
            "provider_factory": "/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/data/factory.py",
            "provider_impl": "/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/data/providers.py",
            "cross_source": "/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/signal/cross_source.py",
            "microstructure": "/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/signal/microstructure.py",
            "engine": "/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/engine.py",
            "signal_engine": "/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/signal/engine.py",
        },
        "primary_sources": {
            "binance_rest_market_data": "https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints",
            "binance_websocket_local_book": "https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams",
            "bybit_orderbook": "https://bybit-exchange.github.io/docs/v5/market/orderbook",
            "bybit_recent_trades": "https://bybit-exchange.github.io/docs/v5/market/recent-trade",
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
