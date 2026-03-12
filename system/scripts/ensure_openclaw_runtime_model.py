#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL_ID = "gpt-5.4"
DEFAULT_API = "openai-responses"
DEFAULT_BASE_URL = "http://127.0.0.1:9999/v1"
DEFAULT_CONTEXT_WINDOW = 391000
DEFAULT_MAX_TOKENS = 128000


def now_utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_model_spec(model_id: str, *, context_window: int, max_tokens: int) -> dict[str, Any]:
    return {
        "id": model_id,
        "name": model_id,
        "reasoning": True,
        "input": ["text", "image"],
        "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0,
        },
        "contextWindow": int(context_window),
        "maxTokens": int(max_tokens),
    }


def ensure_provider_model(
    payload: dict[str, Any],
    *,
    provider: str,
    model_id: str,
    api: str,
    base_url: str,
    context_window: int,
    max_tokens: int,
) -> tuple[bool, list[str]]:
    changed = False
    touched: list[str] = []
    models = payload.setdefault("models", {})
    if not isinstance(models, dict):
        raise ValueError("config.models_not_dict")
    providers = models.setdefault("providers", {})
    if not isinstance(providers, dict):
        raise ValueError("config.models.providers_not_dict")

    provider_cfg = providers.setdefault(provider, {})
    if not isinstance(provider_cfg, dict):
        raise ValueError(f"config.models.providers.{provider}_not_dict")

    if provider_cfg.get("api") != api:
        provider_cfg["api"] = api
        changed = True
        touched.append(f"models.providers.{provider}.api")
    if provider_cfg.get("baseUrl") != base_url:
        provider_cfg["baseUrl"] = base_url
        changed = True
        touched.append(f"models.providers.{provider}.baseUrl")

    model_rows = provider_cfg.setdefault("models", [])
    if not isinstance(model_rows, list):
        raise ValueError(f"config.models.providers.{provider}.models_not_list")

    target = None
    for row in model_rows:
        if isinstance(row, dict) and str(row.get("id", "")).strip() == model_id:
            target = row
            break
    if target is None:
        target = {}
        model_rows.append(target)
        changed = True
        touched.append(f"models.providers.{provider}.models[{model_id}]")

    desired = build_model_spec(model_id, context_window=context_window, max_tokens=max_tokens)
    for key, value in desired.items():
        if target.get(key) != value:
            target[key] = value
            changed = True
            touched.append(f"models.providers.{provider}.models[{model_id}].{key}")

    return changed, touched


def ensure_default_aliases(payload: dict[str, Any], *, provider: str, model_id: str) -> tuple[bool, list[str]]:
    changed = False
    touched: list[str] = []
    agents = payload.setdefault("agents", {})
    if not isinstance(agents, dict):
        raise ValueError("config.agents_not_dict")
    defaults = agents.setdefault("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("config.agents.defaults_not_dict")
    model_entries = defaults.setdefault("models", {})
    if not isinstance(model_entries, dict):
        raise ValueError("config.agents.defaults.models_not_dict")

    for key in (f"{provider}/{model_id}", model_id):
        entry = model_entries.get(key)
        if entry is None:
            model_entries[key] = {}
            changed = True
            touched.append(f"agents.defaults.models[{key}]")
        elif not isinstance(entry, dict):
            model_entries[key] = {}
            changed = True
            touched.append(f"agents.defaults.models[{key}]")
    return changed, touched


def main() -> int:
    parser = argparse.ArgumentParser(description="Ensure OpenClaw runtime config can resolve a target model through the local OpenAI-compatible proxy.")
    parser.add_argument("--config", default=str(Path.home() / ".openclaw" / "openclaw.json"))
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--context-window", type=int, default=DEFAULT_CONTEXT_WINDOW)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    config_path = Path(str(args.config)).expanduser().resolve()
    out: dict[str, Any] = {
        "action": "ensure_openclaw_runtime_model",
        "ok": False,
        "changed": False,
        "config_path": str(config_path),
        "provider": str(args.provider).strip(),
        "model_id": str(args.model_id).strip(),
        "model_ref": f"{str(args.provider).strip()}/{str(args.model_id).strip()}",
        "base_url": str(args.base_url).strip(),
        "api": str(args.api).strip(),
        "backup_path": None,
        "touched_paths": [],
    }

    try:
        if not config_path.exists():
            raise FileNotFoundError(f"config_not_found:{config_path}")
        raw = read_json(config_path)
        if not isinstance(raw, dict):
            raise ValueError("config_root_not_dict")
        before_sha = sha256_file(config_path)
        touched: list[str] = []

        changed_provider, touched_provider = ensure_provider_model(
            raw,
            provider=str(args.provider).strip(),
            model_id=str(args.model_id).strip(),
            api=str(args.api).strip(),
            base_url=str(args.base_url).strip(),
            context_window=int(args.context_window),
            max_tokens=int(args.max_tokens),
        )
        changed_aliases, touched_aliases = ensure_default_aliases(
            raw,
            provider=str(args.provider).strip(),
            model_id=str(args.model_id).strip(),
        )
        touched.extend(touched_provider)
        touched.extend(touched_aliases)
        changed = bool(changed_provider or changed_aliases)

        backup_path: Path | None = None
        if changed:
            if not args.no_backup:
                backup_path = config_path.with_name(f"{config_path.name}.bak_{now_utc_compact()}_runtime_model")
                backup_path.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
            write_json(config_path, raw)
        after_sha = sha256_file(config_path)

        out.update(
            {
                "ok": True,
                "changed": changed,
                "backup_path": str(backup_path) if backup_path is not None else None,
                "before_sha256": before_sha,
                "after_sha256": after_sha,
                "touched_paths": touched,
            }
        )
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        out["error"] = str(exc)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
