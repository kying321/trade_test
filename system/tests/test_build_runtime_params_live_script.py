from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml


def _load_module():
    root = Path(__file__).resolve().parents[1]
    mod_path = root / "scripts" / "build_runtime_params_live.py"
    spec = importlib.util.spec_from_file_location("build_runtime_params_live", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_runtime_params_live_blends_params_live_with_mode_profile(tmp_path: Path) -> None:
    mod = _load_module()
    output_root = tmp_path / "output"
    (output_root / "artifacts").mkdir(parents=True, exist_ok=True)
    (output_root / "daily").mkdir(parents=True, exist_ok=True)
    (output_root / "artifacts" / "params_live.yaml").write_text(
        yaml.safe_dump({"signal_confidence_min": 73.24, "convexity_min": 4.2}, sort_keys=False),
        encoding="utf-8",
    )
    (output_root / "daily" / "2026-03-17_mode_feedback.json").write_text(
        json.dumps({"runtime_mode": "swing"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "validation": {"use_mode_profiles": True, "mode_profile_blend_with_live": 0.55},
                "mode_profiles": {"swing": {"signal_confidence_min": 60.0, "convexity_min": 2.2}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    payload = mod.resolve_runtime_params(
        config_payload={**mod.load_config(config_path), "__config_path__": str(config_path)},
        output_root=output_root,
        as_of=mod.parse_date("2026-03-17"),
    )
    assert payload["source_kind"] == "runtime_params_live"
    assert payload["runtime_mode"] == "swing"
    assert abs(payload["signal_confidence_min"] - 65.958) < 1e-6
    assert abs(payload["convexity_min"] - 3.1) < 1e-6


def test_script_main_writes_artifact_and_review_copy(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    output_root = tmp_path / "output"
    review_dir = output_root / "review"
    (output_root / "artifacts").mkdir(parents=True, exist_ok=True)
    (output_root / "daily").mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    (output_root / "artifacts" / "params_live.yaml").write_text(
        yaml.safe_dump({"signal_confidence_min": 68.0, "convexity_min": 2.9}, sort_keys=False),
        encoding="utf-8",
    )
    (output_root / "daily" / "2026-03-17_mode_feedback.json").write_text(
        json.dumps({"runtime_mode": "ultra_short"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"validation": {"use_mode_profiles": True, "mode_profile_blend_with_live": 0.55}}, sort_keys=False),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_runtime_params_live.py",
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--output-dir",
            str(review_dir),
            "--date",
            "2026-03-17",
            "--now",
            "2026-03-17T18:30:00Z",
        ],
    )
    rc = mod.main()
    assert rc == 0
    artifact = output_root / "artifacts" / "runtime_params_live.json"
    review = review_dir / "20260317T183000Z_runtime_params_live.json"
    assert artifact.exists()
    assert review.exists()
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["runtime_mode"] == "ultra_short"
    assert payload["source_kind"] == "runtime_params_live"
