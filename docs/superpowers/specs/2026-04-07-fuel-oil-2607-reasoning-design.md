# Fuel Oil 2607 Reasoning Design

## Goal

把 `/Users/jokenrobot/Downloads/燃油期货2607分析与交易策略.txt` 从静态长报告改造成 Fenlie 内可重复运行、可验证、可追踪的结构化研究链，避免把硬编码叙述直接塞进系统。

## 已确认约束

- 用户已明确要求“分析并修正这个框架然后吸收并入fenlie系统”。
- 现有系统已经有通用 `commodity_reasoning_*` 原语，可复用但不能继续停留在通用品类摘要。
- 现有公开数据链已能提供：
  - `fuel_oil_inventory` / `fuel_oil_inventory_delta`
  - `bcti_index` / `bdti_index`
  - `cargo_volume` / `coastal_port_throughput`
  - FU/SC 合约日线与实时盘口（AkShare/Sina）
- 不应把报告中的固定概率、固定点位、固定结论原样写入系统。

## 设计

### 1. 新建专用燃油 2607 推演链

新增 `fuel_oil_2607_*` 模块，而不是污染通用 `commodity_reasoning_*`：

- `fuel_oil_2607_input_packet.py`
- `fuel_oil_2607_scenario.py`
- `fuel_oil_2607_transmission.py`
- `fuel_oil_2607_validation.py`
- `fuel_oil_2607_trade_space.py`
- `fuel_oil_2607_strategy.py`
- `fuel_oil_2607_summary.py`

### 2. 输入层做“结构化证据包”，不是写死结论

输入包统一承载：

- 基本面：库存、航运、港口吞吐、原油对标、裂解相对强弱
- 技术面：日线/周线趋势、均线、MACD、RSI、支撑阻力、成交量与持仓变化
- 席位面：成交/多头/空头前排集中度与净意向
- 覆盖度：哪些字段真实可得，哪些字段当前缺失

### 3. 情景层做动态再权重

保留原报告的三条主路径语义，但把固定概率改为基于输入包的路径分数：

- `base_repricing`
- `geopolitical_bull_shock`
- `macro_bear_slump`

### 4. 产出层拆为 7 个工件

- `latest_fuel_oil_2607_input_packet.json`
- `latest_fuel_oil_2607_scenario_tree.json`
- `latest_fuel_oil_2607_transmission_map.json`
- `latest_fuel_oil_2607_validation_ring.json`
- `latest_fuel_oil_2607_trade_space.json`
- `latest_fuel_oil_2607_strategy_matrix.json`
- `latest_fuel_oil_2607_summary.json`

### 5. 系统吸收方式

- 提供独立 runner：`system/scripts/run_fuel_oil_2607_reasoning.py`
- review 目录直接产出 latest 工件
- `build_dashboard_frontend_snapshot.py` 增加 artifact 暴露，保证前端/工作台可以消费

## 取舍

- 本轮先做 `RESEARCH_ONLY`，不触碰 live execution path。
- 本轮不强做所有缺失产业字段（如炼厂利润/开工率），改为显式 coverage 缺口。
- 本轮优先把“报告变结构化链条”，不直接做 UI 新页面。

## 验证

- 单元测试覆盖输入包、情景树、交易空间、runner、dashboard artifact 暴露。
- 真实 runner 至少跑一遍，确认能落 review 工件。
