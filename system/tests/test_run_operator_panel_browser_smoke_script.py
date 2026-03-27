from __future__ import annotations

import contextlib
import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_operator_panel_browser_smoke.py"


def load_module():
    spec = importlib.util.spec_from_file_location("operator_panel_browser_smoke_script", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_operator_panel_expectations_reads_geostrategy_briefs(tmp_path: Path) -> None:
    mod = load_module()
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir(parents=True)
    (dist_dir / "operator_task_visual_panel_data.json").write_text(
        json.dumps(
            {
                "summary": {
                    "event_crisis_primary_theater_brief": "usd_liquidity_and_sanctions",
                    "event_crisis_dominant_chain_brief": "credit_intermediary_chain",
                    "event_crisis_safety_margin_brief": "system_margin=0.42",
                    "event_crisis_hard_boundary_brief": "new_risk_hard_block",
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    payload = mod.load_operator_panel_expectations(dist_dir=dist_dir)

    assert payload["route"] == "/operator_task_visual_panel.html"
    assert payload["markers"] == [
        "事件危机地缘层",
        "usd_liquidity_and_sanctions",
        "credit_intermediary_chain",
        "system_margin=0.42",
        "new_risk_hard_block",
    ]


def test_build_operator_panel_smoke_spec_includes_route_and_geostrategy_markers(tmp_path: Path) -> None:
    mod = load_module()
    screenshot_path = tmp_path / "operator-panel-smoke.png"
    result_path = tmp_path / "operator-panel-smoke.json"

    spec = mod.build_operator_panel_smoke_spec(
        base_url="http://127.0.0.1:4173/",
        screenshot_path=screenshot_path,
        result_path=result_path,
        route="/operator_task_visual_panel.html",
        markers=[
            "事件危机地缘层",
            "usd_liquidity_and_sanctions",
            "credit_intermediary_chain",
            "system_margin=0.42",
            "new_risk_hard_block",
        ],
    )

    assert "operator_task_visual_panel.html" in spec
    assert "事件危机地缘层" in spec
    assert "usd_liquidity_and_sanctions" in spec
    assert "credit_intermediary_chain" in spec
    assert "system_margin=0.42" in spec
    assert "new_risk_hard_block" in spec
    assert str(screenshot_path) in spec
    assert str(result_path) in spec
    assert "page.screenshot" in spec


def test_build_artifact_payload_reports_panel_assertion(tmp_path: Path) -> None:
    mod = load_module()
    report_path = tmp_path / "report.json"
    screenshot_path = tmp_path / "panel.png"

    payload = mod.build_artifact_payload(
        workspace=tmp_path,
        report_path=report_path,
        screenshot_path=screenshot_path,
        base_url="http://127.0.0.1:4173/",
        server_ready_seconds=0.33,
        build_result={"returncode": 0},
        smoke_result={
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "playwright_result": {
                "final_url": "http://127.0.0.1:4173/operator_task_visual_panel.html",
                "visible_markers": [
                    "事件危机地缘层",
                    "usd_liquidity_and_sanctions",
                    "credit_intermediary_chain",
                    "system_margin=0.42",
                    "new_risk_hard_block",
                ],
            },
        },
        route="/operator_task_visual_panel.html",
        markers=[
            "事件危机地缘层",
            "usd_liquidity_and_sanctions",
            "credit_intermediary_chain",
            "system_margin=0.42",
            "new_risk_hard_block",
        ],
    )

    assert payload["action"] == "operator_panel_browser_smoke"
    assert payload["ok"] is True
    assert payload["change_class"] == "RESEARCH_ONLY"
    assert payload["route"] == "/operator_task_visual_panel.html"
    assert payload["visible_markers"] == [
        "事件危机地缘层",
        "usd_liquidity_and_sanctions",
        "credit_intermediary_chain",
        "system_margin=0.42",
        "new_risk_hard_block",
    ]
    assert payload["final_url"] == "http://127.0.0.1:4173/operator_task_visual_panel.html"
    assert payload["server_ready_seconds"] == 0.33


def test_main_writes_operator_panel_browser_smoke_report(monkeypatch, tmp_path: Path, capsys) -> None:
    mod = load_module()
    workspace = tmp_path / "workspace"
    system_root = workspace / "system"
    web_root = system_root / "dashboard" / "web"
    dist_dir = web_root / "dist"
    review_dir = system_root / "output" / "review"
    dist_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    (dist_dir / "operator_task_visual_panel.html").write_text("<html></html>\n", encoding="utf-8")
    (dist_dir / "operator_task_visual_panel_data.json").write_text(
        json.dumps(
            {
                "summary": {
                    "event_crisis_primary_theater_brief": "usd_liquidity_and_sanctions",
                    "event_crisis_dominant_chain_brief": "credit_intermediary_chain",
                    "event_crisis_safety_margin_brief": "system_margin=0.42",
                    "event_crisis_hard_boundary_brief": "new_risk_hard_block",
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    seen: dict[str, object] = {}
    monkeypatch.setattr(
        mod,
        "ensure_success",
        lambda **kwargs: (
            seen.update({"name": kwargs["name"], "cmd": kwargs["cmd"], "cwd": str(kwargs["cwd"])}) or
            {"returncode": 0, "cmd": kwargs["cmd"], "cwd": str(kwargs["cwd"])}
        ),
    )
    monkeypatch.setattr(mod, "choose_port", lambda host: 4173)
    monkeypatch.setattr(mod, "wait_http_ready", lambda url, timeout_seconds: 0.42)
    monkeypatch.setattr(mod, "http_server", lambda **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(
        mod,
        "run_playwright_smoke",
        lambda **kwargs: {
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "playwright_result": {
                "final_url": "http://127.0.0.1:4173/operator_task_visual_panel.html",
                "visible_markers": kwargs["markers"],
            },
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_operator_panel_browser_smoke.py",
            "--workspace",
            str(workspace),
        ],
    )

    mod.main()
    payload = json.loads(capsys.readouterr().out)

    assert payload["ok"] is True
    assert payload["route"] == "/operator_task_visual_panel.html"
    assert payload["visible_markers"] == [
        "事件危机地缘层",
        "usd_liquidity_and_sanctions",
        "credit_intermediary_chain",
        "system_margin=0.42",
        "new_risk_hard_block",
    ]
    assert seen["name"] == "operator_panel_refresh"
    assert seen["cmd"][:4] == [
        mod.current_python_executable(),
        str(system_root / "scripts" / "run_operator_panel_refresh.py"),
        "--workspace",
        str(workspace),
    ]
    assert seen["cmd"][4] == "--now"
    report_path = Path(payload["report_path"])
    assert report_path.exists()
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["action"] == "operator_panel_browser_smoke"
