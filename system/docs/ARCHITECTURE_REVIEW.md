# 架构整理与复盘（2026-02-13）

## 目标
- 保持“反脆弱优先”不变量：先活下来，再追求收益。
- 将编排与策略规则解耦，减少单点复杂度。
- 在不破坏既有 CLI 契约的前提下提升可维护性。

## 本轮发现
- `src/lie_engine/engine.py` 同时承担编排、门禁规则、因子重估、审计写入，职责过载。
- 门禁逻辑（黑天鹅评分/事件窗口/冷却期）与 EOD、盘前、盘中流程耦合，难以独立验证。
- 因子边际贡献计算嵌入引擎内部，后续替换模型成本高。

## 本轮改造
1. 新增编排策略子模块：
   - `src/lie_engine/orchestration/guards.py`
   - `src/lie_engine/orchestration/factor_contrib.py`
   - `src/lie_engine/orchestration/__init__.py`
2. `LieEngine` 保持外部接口不变，但内部改为调用编排子模块：
   - 门禁评估统一走 `build_guard_assessment(...)`
   - 120 日因子重估统一走 `estimate_factor_contrib_120d(...)`
3. 保持兼容性：
   - `LieEngine._major_event_window/_loss_cooldown_active/_black_swan_assessment/_estimate_factor_contrib_120d` 仍保留（作为薄封装）
   - 既有测试中的 monkeypatch 不受破坏

## 架构边界（更新后）
- `data/*`：采集、标准化、质量门禁、存储。
- `regime/*`：体制识别（Hurst/HMM/ATR_Z）。
- `signal/*`：信号引擎与 B/S 候选点。
- `risk/*`：仓位、止损、约束执行。
- `backtest/*`：事件驱动回测与 walk-forward。
- `review/*`：参数更新与审计输出。
- `orchestration/*`：跨模块业务规则编排（门禁、因子重估等可演进逻辑）。
- `engine.py`：流程编排器，仅连接模块，不承载重规则。

## 不变量（必须长期保持）
- 冲突数据不进入模型主路径。
- 不确定体制/极端波动/重大事件窗口默认保护模式。
- 参数更新必须可追溯：变更原因、影响窗口、回滚锚点。
- 所有发布判断必须通过 `run-review-cycle` 闭环。

## 后续建议（下一轮）
1. 继续缩减 `engine.py`：将 `gate_report/ops_report/review_loop` 拆至 `orchestration/release.py`。
2. 将默认模拟数据 Provider 与真实公开源 Provider 解耦成独立 profile（dev/prod）。
3. 引入架构回归测试：校验关键模块不出现跨层反向依赖。

