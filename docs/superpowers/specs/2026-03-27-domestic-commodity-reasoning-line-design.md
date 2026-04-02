# Domestic Commodity Reasoning Line Design

## 1. Goal

为 Fenlie 增加一条**独立于交易执行**的“国内大宗商品逻辑推理分析线”，用于：

- 基于事件/研究类 source artifacts 构建情景树
- 推导事件到板块、品种、具体合约的传导链
- 输出结论摘要、作用范围、边界强度与失效条件
- 在 dashboard / operator 中提供单独入口和摘要视图

该线路**不直接赋予执行 authority**，只提供推理、解释、验证与风险边界信息。

## 2. Scope

### 2.1 首版覆盖对象

首版目标为**国内大宗商品通用框架**，输出粒度固定为：

`板块 -> 品种 -> 具体合约`

首个验证样本可以使用：

- `BU2606`（沥青 2606）

但设计不绑定单一合约。

### 2.2 首版输入范围

首版主输入只读以下两类：

- 事件类 source artifacts
- 研究类 source artifacts

### 2.3 首版不做的事

- 不做下单、风控压缩、持仓建议或资金动作
- 不直接接 live execution
- 不把行情侧因子作为 source authority
- 不把同花顺期货通或任意第三方 GUI 工具变成主权输入源

## 3. Authority Boundary

该线路必须遵守 Fenlie 既有 authority order：

1. source-of-truth artifacts
2. compact handoff/context
3. UI/operator summaries
4. chat memory

其中：

- **事件/研究类 source artifacts** 是主权输入
- **截面数据 / 截面新闻 / 同花顺期货通指标 / 自定义公式结果** 只能作为**验证环**输入
- 验证环可以：
  - 收缩结论
  - 标记边界变弱
  - 提供反证
  - 触发 review
- 验证环**不能**直接提升结论级别或替代 source authority

## 4. Recommended Architecture

推荐采用**双环校验型**结构。

### 4.1 主链

`事件/研究 artifacts -> 情景树 -> 传导链 -> 板块判断 -> 品种判断 -> 合约判断 -> 结论摘要`

### 4.2 校验环

`截面新闻 + 截面数据 + 可获得的行情侧观测 -> 反向验证主链的作用范围 / 边界强度 / 持续性 / 失效条件`

### 4.3 设计意图

该结构用于避免：

- 线性顺推导致的自洽陷阱
- 单一路径解释导致的范围漂移
- “看起来合理”但缺少边界约束的推理输出

## 5. Output Artifacts

首版新增以下 artifacts：

### 5.1 `commodity_reasoning_scenario_tree.json`

用途：表达当前主情景、次情景、触发条件与切换条件。

最小字段：

- `generated_at_utc`
- `root_theme`
- `primary_scenario`
- `secondary_scenarios[]`
- `scenario_nodes[]`

`scenario_nodes[]` 每项最小字段：

- `scenario_id`
- `label`
- `level`
- `trigger_conditions[]`
- `invalidators[]`
- `confidence_score`

### 5.2 `commodity_reasoning_transmission_map.json`

用途：表达从事件到板块/品种/合约的传导路径。

最小字段：

- `generated_at_utc`
- `primary_chain`
- `chains[]`

`chains[]` 每项最小字段：

- `chain_id`
- `from_event`
- `sector`
- `commodity`
- `contract`
- `path_nodes[]`
- `range_scope`
- `boundary_strength`
- `confidence_score`

### 5.3 `commodity_reasoning_boundary_strength.json`

用途：表达每条传导链在当前样本中的作用范围、边界强度、脆弱点。

最小字段：

- `generated_at_utc`
- `range_summary`
- `boundary_rows[]`

`boundary_rows[]` 每项最小字段：

- `target_level`  (`sector|commodity|contract`)
- `target_id`
- `range_scope`
- `boundary_strength`
- `persistence_strength`
- `fragility_flags[]`
- `counter_evidence[]`

### 5.4 `commodity_reasoning_summary.json`

用途：给 operator/dashboard 提供紧凑摘要。

最小字段：

- `generated_at_utc`
- `headline`
- `primary_scenario_brief`
- `primary_chain_brief`
- `range_scope_brief`
- `boundary_strength_brief`
- `invalidator_brief`
- `contracts_in_focus[]`

## 6. Reasoning Model

### 6.1 情景树层

情景树用于回答：

- 当前最可能发生的结构场景是什么
- 次级路径有哪些
- 哪些条件会触发切换
- 哪些条件会使当前情景失效

### 6.2 传导链层

传导链用于回答：

- 事件冲击如何先传到板块
- 板块内哪些品种最先承压/受益
- 进一步落到哪些具体合约
- 当前传导是窄范围还是广范围

### 6.3 边界强度层

边界强度用于回答：

- 当前推理只对局部合约成立，还是对整个板块成立
- 影响是脉冲式、阶段式，还是有持续性
- 哪些反证会迅速推翻当前链路

## 7. Validation Ring

### 7.1 允许引入的验证类信息

在首版中，以下内容允许作为**验证环**：

- 截面新闻
- 截面数据
- 可获得的商品/期货横截面观测
- 同花顺期货通中可提取的指标观测
- 自定义指标/源码编辑中导出的派生结论

### 7.2 验证环的职责

验证环只做以下事情：

- 验证主链是否覆盖正确范围
- 判断边界强度是否被高估
- 提供 counter-evidence
- 标记推理是否进入 review_required

### 7.3 验证环禁止项

验证环禁止：

- 直接改写 source-owned primary scenario
- 直接赋予交易执行建议
- 在缺 source authority 时单独给出高置信结论

## 8. Dashboard / Operator Integration

### 8.1 Integration Goal

在现有 dashboard/operator 中增加一条**独立分析泳道**，不复用 event crisis live-guard 路径。

### 8.2 Dashboard

dashboard 首版应展示：

- 主情景
- 主传导链
- 作用范围
- 边界强度
- 失效条件
- 关注合约列表

### 8.3 Operator

operator 首版只透传：

- `primary_scenario_brief`
- `primary_chain_brief`
- `range_scope_brief`
- `boundary_strength_brief`
- `invalidator_brief`

operator 不重算 reasoning authority。

## 9. Change Class and Safety

该线路首版默认 change class：

- `RESEARCH_ONLY`

原因：

- 这是解释/推理/验证层
- 不涉及 live execution authority
- 不涉及 order routing / capital path

## 10. First-Version Rollout

建议 rollout 顺序：

1. 建立 reasoning artifacts 合同
2. 接入事件/研究类 source artifacts
3. 输出情景树 / 传导链 / 边界强度 / summary
4. 用 `BU2606` 做首个验证样本
5. 再挂 dashboard/operator

## 11. Risks

首版主要风险：

- 情景树过宽，导致解释过度
- 传导链粒度不稳定，导致合约层结论漂移
- 验证环过强，反向污染 source authority
- 过早引入行情因子，导致分析线变成半执行链

## 12. Decision

首版设计决策固定为：

- 做“国内大宗商品通用框架”
- 粒度做到 `板块 -> 品种 -> 合约`
- 主输入只读事件/研究类 source artifacts
- 截面数据 / 新闻 / 同花顺期货通指标只做验证环
- 输出主形态为：
  - 情景树
  - 传导链
  - 结论摘要
  - 作用范围
  - 边界强度
  - 失效条件
