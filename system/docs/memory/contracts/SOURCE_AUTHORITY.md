# Source Authority Contract

## Purpose
定义 Fenlie 在 source artifact、handoff/context、UI summary、memory 之间的权威边界，确保“谁有权裁决 queue/gate/head”在所有消费者中一致且可审计。

## Authority Chain
1. source-of-truth artifacts（builder 输出、contract 字段、可执行控制字段）
2. compact handoff/context（面向消费层的压缩控制面，必须回指 source）
3. UI/operator summaries（仅解释与展示）
4. chat/memory 文档（仅导航与规则提示）

上层可以解释下层，但不得覆盖下层裁决。

## Canonical Heads
- canonical source head 由 source artifact 或其受控 handoff 的 active 行定义。
- `baseline_retained` 可以是 canonical source head：它表示“基线被继续保留并成为当前有效头”，不是“无结论”。
- canonical head 的变更必须由 source 侧字段驱动，不允许 UI/brief 自行宣布“已切换”。

## Consumer Obligations
- consumer 必须读取 source-owned 控制字段（如 source head、gate、queue owner、allowed/blocked）。
- consumer 不得重复推导 queue/gate/head，不得通过文案拼接或本地缓存替代 source 判定。
- 如发现 source 与 handoff 不一致，consumer 必须降级为“待仲裁/待复核”，而不是自选更“合理”的一侧。

## Superseded vs Active Rows
- active row：当前可被执行链消费的唯一有效裁决行。
- superseded row：已被真实 anchor 替换，不再具备执行权威。
- `superseded_anchor` 仅在发生真实 anchor 替换时填写。
- baseline-retained 或 inconclusive 场景下，不得伪造 superseded 关系来制造“已晋升/已淘汰”叙事。

## Anti-Patterns
- 在 UI summary 中硬编码“当前 head/queue/gate”并绕过 source 字段。
- 通过 consumer 侧二次聚合重算 queue 状态、gate 结论或 head 归属。
- 把 brief/聊天结论当成 active row 使用。
- 因为 sidecar 或单次观察结果，直接改写 canonical head。

## Related Artifacts
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/MEMORY_INDEX.md`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/DURABLE_RULES.md`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/FENLIE_CODEX_MEMORY.md`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/OPENCLAW_CLOUD_BRIDGE.md`
- `system/output/review/*_handoff*.json`（source-owned handoff/control 字段）
