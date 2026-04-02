# Fenlie Shortlist

## Identity
Fenlie 是以 ETHUSDT / 15m / single-symbol breakout-pullback 为主线的策略研究与操作协同平台，长期保持源头 artifact（exit/risk、hold-forward、contract）控制的权威链。

## Authority Order
1. source artifacts（dataset、builder outputs、handoff JSON）
2. compact handoff/context（NEXT_WINDOW_CONTEXT_LATEST、handoff 中的 control fields、contracts 中定义的长期 authority/boundary 规则）
3. UI/operator summaries
4. 本地记忆与 chat 口述说明

## Current Main Priorities
- 保持 `ETHUSDT / 15m / single-symbol / breakout-pullback` 研究主线，强调 price-action breakout-pullback 的 exit/risk 以及 hold-forward arbitration。
- 当前主要由 source artifacts 提供 exit/risk、hold-forward arbitration 的 guard/control fields，memory_shortlist 仅作为摘要/导航；具体 `challenge_pair` 命名与解释请参考 `memory/contracts/RESEARCH_CONTRACTS.md`。

## Low-Value
- geometry_delta 截面推理（跨 symbol / cross-section 的 breakout 变体）
- VPVR proxy 之类的密度推导

## Deprioritized Paths
- reclaim structure filter 的后验节点
- daily_stop_r 作为主要止损路径
这些方向目前不再作为主线研究。

## Cold Start
Main-agent bootstrap path：
1. `AGENTS.md`
2. `FENLIE_CODEX_MEMORY.md`
3. `memory/MEMORY_INDEX.md`
4. `memory/SHORTLIST.md`
5. 具体任务相关的 contract/source artifact

子代理与人工不要把上面当成统一序列；请按 `memory/MEMORY_INDEX.md` 中的角色化读序选择最小必要读取范围。

## “继续 / 下一步” Semantics
“继续”只在先决条件（state/queue/gate/source ownership）发生变化后才触发；优先处理能改变任何 source-owned gate、状态、手工 queue 或 user-visible 能力的未解决高危问题。连续两轮都没改变这些内容就需要停止并总结。

## Operator Preference
- 默认高自主推进：少拆解、少过程解释、少频繁确认。
- 除非涉及高风险不可逆动作、权限缺失、source-of-truth 冲突或真实 blocker，否则直接推进主线并在关键节点再汇报。
- 用户未要求详细计划时，不主动展开过细步骤分解。

## Never Treat As Source-of-Truth
记忆树是解释型手册——它提醒你如何恢复与判断，但永远不能代替 source artifacts、handoff 或 live contract。本文件中提及的优先级与规则必须回到对应 artifact 验证后再采取行动。
