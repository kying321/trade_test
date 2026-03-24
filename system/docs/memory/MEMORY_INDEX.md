# Fenlie Memory Index

## Purpose
Fenlie 的记忆树旨在提供一个清晰、可分层读取的导航器，帮助主代理、子代理与人工在不侵入 source-of-truth 的前提下快速找到需要的规则与契约。

## Authority Order
1. source-of-truth artifacts（代码/数据/手动路径）
2. compact handoff/context（如 NEXT_WINDOW_CONTEXT_LATEST、source-owned artifacts 中的 control fields）
3. UI/operator summaries
4. chat 记忆与本地记忆手册
任何文档都不得超越以上顺序替代信源。

## Cold Start Read Order
- 主代理默认读取：`AGENTS.md` → `FENLIE_CODEX_MEMORY.md` → `memory/MEMORY_INDEX.md` → `memory/SHORTLIST.md` → 相关 contract。
- 子代理默认读取：`memory/SHORTLIST.md` → 与任务直接相关的 contract，必要时再补 `memory/DURABLE_RULES.md`。
- 人类默认序列：`memory/SHORTLIST.md` → `memory/DURABLE_RULES.md` → 争议时再退回到具体 contract。

## Task-to-Document Map
- 快速恢复及身份指引：`memory/SHORTLIST.md`
- 关于长期规范与可变边界：`memory/DURABLE_RULES.md`
- 深层 contract（source、live、research 等）：`memory/contracts/` 目录中的各章
- 总体导航与文件布局：本文件

## What Belongs Here vs Source Artifacts
本目录仅记录长期规则、导航与读取协议。禁止写入：
- 实时 ready-check/queue/gate/live 状态
- 单次回测窗口或 pnl 结论
- 短期 blocker/未稳定 review
这类信息应保留在 source artifacts、hand-off 文档或专门的 controller 里。

## Update Rules
- 本索引只需在导航结构或读取协议改变时更新。
- 长期规则、深层契约将逐步移至 `memory/DURABLE_RULES.md` 与 `memory/contracts/`（Task 2 计划完成后正式落地）。
- 当前 `FENLIE_CODEX_MEMORY.md` 仍承担 bootstrap 与高价值摘要，Task 3 将进一步收敛其职责为指向新树的兼容 shim。

## Memory Tree Layout
```
memory/
├── MEMORY_INDEX.md        ← 本文件，导航 + 协议
├── SHORTLIST.md          ← 一屏恢复（身份、主线、禁行、低价值）
├── DURABLE_RULES.md      ← 长期规则（操作风格、边界、验证、停止、更新）
└── contracts/            ← 计划由 Task 2 填充的 deep contract 章节点
```
Memory tree 允许主代理全量阅读，也可让子代理仅截取所需文件。
