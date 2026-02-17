# Agent Shift Report | 2026-02-17

## Scope
- 连续完成两项优先级：
  - `stress-autorun adaptive reason drift monitor`
  - `stress-autorun reason-drift artifact export`

## Delivered
- Reason drift 监控上线（gate/ops/review 全链路）：
  - 新门控：`stress_autorun_reason_drift_ok`
  - 指标：`reason_mix_gap`、`change_point_gap`、baseline/recent reason 占比。
  - 告警：`stress_autorun_reason_mix_drift`、`stress_autorun_reason_change_point`。
- 缺陷计划接入：
  - `STRESS_AUTORUN_REASON_MIX`
  - `STRESS_AUTORUN_REASON_CHANGE_POINT`
- 审计工件上线：
  - `output/review/YYYY-MM-DD_stress_autorun_reason_drift.json`
  - `output/review/YYYY-MM-DD_stress_autorun_reason_drift.md`
  - 工件内容包含：
    - top reason transitions
    - per-window drift trace（mix_gap/change_point_gap）
    - recent reason 序列
- ops 报告新增章节：
  - `Stress Autorun 原因漂移`（含工件写入状态与路径）。

## Files Updated
- `src/lie_engine/orchestration/release.py`
- `src/lie_engine/config/validation.py`
- `tests/test_release_orchestrator.py`
- `tests/test_config_validation.py`
- `config.yaml`
- `config.daemon.test.yaml`
- `README.md`
- `docs/PROGRESS.md`

## Validation
- 定向测试：
  - `python3 -m unittest tests.test_release_orchestrator.ReleaseOrchestratorTests.test_gate_report_fails_on_stress_autorun_reason_drift tests.test_release_orchestrator.ReleaseOrchestratorTests.test_ops_report_flags_stress_autorun_reason_drift tests.test_release_orchestrator.ReleaseOrchestratorTests.test_review_until_pass_defect_plan_includes_stress_autorun_reason_drift_breaches -v`
  - `3 passed`
- 相关全量回归：
  - `python3 -m unittest tests.test_config_validation tests.test_release_orchestrator -v`
  - `46 passed`
- 配置校验：
  - `lie validate-config` -> `ok=true, errors=0, warnings=0`
- 全量测试：
  - `lie test-all` -> `selected=131, ran=131, failed=0`
  - 日志：`output/logs/tests_20260217_160618.json`

## Environment Note
- `python3 -m pip install -e .` 仍受本机 SSL 证书链问题影响（拉取 `setuptools>=68` 失败）。
- `python3 -m pip install -e . --no-build-isolation` 仍失败（缺失 `setuptools.build_meta`）。
- 不影响当前本地 `lie` 命令执行与回归验证。

## Next Priority
1. Add stress-autorun reason-drift retention + checksum index policy（对历史 drift 工件做轮转与哈希审计）.
