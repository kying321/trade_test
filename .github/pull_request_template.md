## Merge Scope
- [ ] 仅包含主代码与必要配置（不含 `system/output/**`、`__pycache__`、`*.egg-info`、`*.pdf`）
- [ ] 说明本次变更的失败域（Failure Domain）与爆炸半径（Blast Radius）

## Mandatory 3 (Merge Blocking)
- [ ] `validate-config` 已执行并附真实输出（`ok: true` 或错误堆栈）
- [ ] `test-all --fast` 已执行并附真实日志路径（`system/output/logs/tests_*.json`）
- [ ] 已生成 24h 命令白名单报告并附路径（含时间戳/返回码/成功率样本）

命令参考：
```bash
cd system
PYTHONPATH=src python -m lie_engine.cli --config config.yaml validate-config
PYTHONPATH=src python -m lie_engine.cli --config config.yaml test-all --fast --fast-ratio 0.10
python scripts/exec_with_audit.py --source manual --tag gate-check -- lie gate-report --date 2026-02-28
python scripts/command_whitelist_24h.py --include-tests-log
```

## Rollback (Merge Blocking)
- [ ] 已提供一键回滚命令（可直接复制执行）
- [ ] 已说明回滚触发条件（何时必须熔断/回滚）

## 48h Follow-up 2 (Post-Merge Required)
- [ ] 覆盖率对比（合并前后，按关键模块）
- [ ] 故障注入恢复时延（P50 / P95 / P99）

## Physical Proof
- [ ] 贴出关键终端片段（时间戳 + 返回码）
- [ ] 贴出关键产物路径（JSON/MD）

## Notes
- 任何新增依赖、环境变量、端口、核心路由改动，需单独标记为高风险并补充干跑与恢复计划。
