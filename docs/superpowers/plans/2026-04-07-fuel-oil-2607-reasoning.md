# Fuel Oil 2607 Reasoning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把燃油 2607 静态报告改造成可执行、可测试、可暴露到 Fenlie review/dashboard 的结构化研究链。

**Architecture:** 先建立 `fuel_oil_2607_input_packet -> scenario -> transmission -> validation -> trade_space -> strategy -> summary` 七段式纯函数链，再用单独 runner 负责数据抓取和工件落盘。dashboard 只暴露 latest artifact 路径，不重算逻辑。

**Tech Stack:** Python 3, pytest, AkShare, pandas, existing Fenlie review artifact conventions

---

### Task 1: 建立失败测试

**Files:**
- Create: `/Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_fuel_oil_2607_reasoning.py`
- Create: `/Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_run_fuel_oil_2607_reasoning.py`
- Modify: `/Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_build_dashboard_frontend_snapshot_script.py`

- [ ] Step 1: 为输入包/情景树/交易空间写失败测试
- [ ] Step 2: 运行定向 pytest，确认失败原因是模块缺失

### Task 2: 实现燃油 2607 专用推演模块

**Files:**
- Create: `/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/research/fuel_oil_2607_input_packet.py`
- Create: `/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/research/fuel_oil_2607_scenario.py`
- Create: `/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/research/fuel_oil_2607_transmission.py`
- Create: `/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/research/fuel_oil_2607_validation.py`
- Create: `/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/research/fuel_oil_2607_trade_space.py`
- Create: `/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/research/fuel_oil_2607_strategy.py`
- Create: `/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/research/fuel_oil_2607_summary.py`

- [ ] Step 1: 先实现最小纯函数让测试转绿
- [ ] Step 2: 再补 coverage/missing/source 字段
- [ ] Step 3: 跑定向 pytest 保持绿色

### Task 3: 接入 runner 与 latest 工件落盘

**Files:**
- Create: `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_fuel_oil_2607_reasoning.py`
- Test: `/Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_run_fuel_oil_2607_reasoning.py`

- [ ] Step 1: 写 runner 纯管线 `run_pipeline`
- [ ] Step 2: 加真实抓取 helper（AkShare + PublicInternetResearchProvider）
- [ ] Step 3: 跑测试并真实执行一次，确认 review/latest 工件存在

### Task 4: 暴露到 dashboard snapshot

**Files:**
- Modify: `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/build_dashboard_frontend_snapshot.py`
- Test: `/Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_build_dashboard_frontend_snapshot_script.py`

- [ ] Step 1: 新增 fuel oil 2607 artifact 发现逻辑
- [ ] Step 2: 跑对应 pytest

### Task 5: 文档收口

**Files:**
- Modify: `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/PROGRESS.md`

- [ ] Step 1: 记录本轮变更、验证、风险、回滚点
