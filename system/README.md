# 离厄反脆弱交易系统

架构复盘文档：

```bash
docs/ARCHITECTURE_REVIEW.md
```

运行：

```bash
python3 -m pip install -e .
lie run-eod --date 2026-02-13
lie test-all
lie validate-config
lie architecture-audit --date 2026-02-13
# 10小时预算的实盘多模式研究回测（新闻+研报因子）
lie research-backtest --start 2015-01-01 --end 2026-02-13 --hours 10 --max-symbols 120 --max-trials-per-mode 500 --review-days 5
# 单模式定向诊断（可选：ultra_short,swing,long）
lie research-backtest --start 2015-01-01 --end 2026-02-13 --hours 2 --modes swing
```

调度：

```bash
# 单槽位执行
lie run-slot --date 2026-02-13 --slot 08:40
lie run-slot --date 2026-02-13 --slot 15:10
lie run-slot --date 2026-02-13 --slot ops

# 单日全流程
lie run-session --date 2026-02-13
lie run-review-cycle --date 2026-02-13 --max-rounds 2
# 生产建议 max-rounds>=1；max-rounds=0 仅用于跳过复审循环的快速联调

# 守护调度（按 config.yaml 固定时段触发）
lie run-daemon --poll-seconds 30

# 本地 cron 安装/卸载
./infra/local/install_cron.sh
./infra/local/uninstall_cron.sh

# 健康检查与失败重试
lie health-check --date 2026-02-13
lie stable-replay --date 2026-02-13 --days 3
lie gate-report --date 2026-02-13 --run-tests
lie ops-report --date 2026-02-13 --window-days 7
./infra/local/healthcheck.sh 2026-02-13
./infra/local/retry_slot.sh 2026-02-13 15:10

# 审查未通过时会自动生成：
# output/review/YYYY-MM-DD_defect_plan_roundN.json
# output/review/YYYY-MM-DD_defect_plan_roundN.md
```
