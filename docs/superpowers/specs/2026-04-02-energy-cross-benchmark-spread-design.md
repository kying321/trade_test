# Energy Cross-Benchmark Spread Design

## 1. Goal

为 Fenlie 增加一个**最小 source-owned 能源跨基准价差 artifact**，用于稳定表达：

- SC 原油相对 Brent / WTI 的价格差
- FU / LU 相对 Brent / WTI 的桶当量价格差
- 显式汇率输入与换算假设
- 每个输入价格自己的 `as_of` 与 `source`

该设计首版只解决**可审计价格层**，不直接做归因、交易建议、UI 展示或 live path 接入。

## 2. Scope

### 2.1 首版覆盖对象

首版固定覆盖以下输入：

- `SC`（国内原油，CNY/bbl）
- `FU`（高硫燃料油，CNY/t）
- `LU`（低硫燃料油，CNY/t）
- `Brent`（USD/bbl）
- `WTI`（USD/bbl）
- `USD/CNY`

### 2.2 首版产出内容

首版只产出三层：

1. `inputs`：原始输入价格 + 时间戳 + source
2. `derived`：统一换算后的 USD/bbl 或 USD/t 数值
3. `spreads`：跨基准价差结果

### 2.3 首版不做的事

- 不做 BDTI / BCTI / 燃油库存 / 原油加工量等解释层
- 不做 event crisis / commodity reasoning 自动消费
- 不做 dashboard/operator panel 展示
- 不做自动抓网页或外部 API 聚合
- 不做交易信号或风险建议

## 3. Authority Boundary

该 artifact 必须遵守 Fenlie 的 authority order：

1. source-of-truth artifacts
2. compact handoff/context
3. UI/operator summaries
4. chat memory

因此首版明确要求：

- 该 builder **不直接重算 UI 状态**
- 该 builder **不从 chat 文本取值**
- 该 builder 使用**repo-owned explicit input JSON** 作为主权输入
- consumer 未来只能**只读消费**该 artifact，不能在 UI 中重新推导核心价差

## 4. Time Semantics

首版采用：

- `snapshot_mode = latest_available_composite`

含义：

- 每个输入字段都允许使用“最近可得公开锚点”
- 不强制所有输入共享完全相同时间戳
- 每个输入必须显式保留：
  - `price`
  - `unit`
  - `as_of`
  - `source`

设计原因：

- 比严格同一时点快照更稳健
- 更适合首版 source-owned 落地
- 避免因为单一时间对不齐而整包失效

## 5. Recommended Architecture

推荐采用单脚本、显式输入、固定公式的最小闭包：

`explicit input JSONs -> normalize -> derive -> spread artifact -> checksum`

### 5.1 输入方式

脚本显式接受：

- `--sc-json`
- `--fu-json`
- `--lu-json`
- `--brent-json`
- `--wti-json`
- `--fx-json`

每个输入 JSON 最小格式：

```json
{
  "price": 656.7,
  "as_of": "2026-04-01T14:33:00+08:00",
  "source": "yicai",
  "unit": "CNY/bbl"
}
```

首版要求：

- `price` 必须存在且可转 float
- `as_of` 必须存在且可解析
- `source` 必须存在
- `unit` 必须与该品种预期单位一致

### 5.2 输出 artifact

首版输出：

- `*_energy_cross_benchmark_spread.json`
- `*_energy_cross_benchmark_spread_checksum.json`
- `latest_energy_cross_benchmark_spread.json`

不产出 markdown，避免首版过度扩张。

## 6. Artifact Schema

```json
{
  "ok": true,
  "status": "ok",
  "as_of": "2026-04-02T12:00:00Z",
  "snapshot_mode": "latest_available_composite",
  "inputs": {
    "sc": {"price": 656.7, "unit": "CNY/bbl", "as_of": "...", "source": "..."},
    "fu": {"price": 4001.0, "unit": "CNY/t", "as_of": "...", "source": "..."},
    "lu": {"price": 4772.0, "unit": "CNY/t", "as_of": "...", "source": "..."},
    "brent": {"price": 109.12, "unit": "USD/bbl", "as_of": "...", "source": "..."},
    "wti": {"price": 112.60, "unit": "USD/bbl", "as_of": "...", "source": "..."},
    "usd_cny": {"price": 6.90, "unit": "CNY/USD", "as_of": "...", "source": "..."}
  },
  "assumptions": {
    "fu_bbl_per_ton": 6.35,
    "lu_bbl_per_ton": 6.35
  },
  "derived": {
    "sc_usd_per_bbl": 95.17,
    "fu_usd_per_ton": 579.86,
    "lu_usd_per_ton": 691.59,
    "fu_usd_per_bbl_equiv": 91.32,
    "lu_usd_per_bbl_equiv": 108.91
  },
  "spreads": {
    "sc_minus_brent_usd_per_bbl": -13.95,
    "sc_minus_wti_usd_per_bbl": -17.43,
    "fu_bbl_equiv_minus_brent_usd_per_bbl": -17.80,
    "fu_bbl_equiv_minus_wti_usd_per_bbl": -21.28,
    "lu_bbl_equiv_minus_brent_usd_per_bbl": -0.21,
    "lu_bbl_equiv_minus_wti_usd_per_bbl": -3.69
  },
  "artifact": "...",
  "checksum": "..."
}
```

## 7. Calculation Rules

### 7.1 Fixed assumptions

首版固定：

- `fu_bbl_per_ton = 6.35`
- `lu_bbl_per_ton = 6.35`

理由：

- 保持首版最小闭包
- 避免引入动态密度 / 产品规格分叉
- 后续若要增强，可再拆到解释层

### 7.2 Core formulas

- `sc_usd_per_bbl = sc_cny_per_bbl / usd_cny`
- `fu_usd_per_ton = fu_cny_per_ton / usd_cny`
- `lu_usd_per_ton = lu_cny_per_ton / usd_cny`
- `fu_usd_per_bbl_equiv = fu_usd_per_ton / fu_bbl_per_ton`
- `lu_usd_per_bbl_equiv = lu_usd_per_ton / lu_bbl_per_ton`

### 7.3 Spread formulas

- `sc_minus_brent_usd_per_bbl = sc_usd_per_bbl - brent_usd_per_bbl`
- `sc_minus_wti_usd_per_bbl = sc_usd_per_bbl - wti_usd_per_bbl`
- `fu_bbl_equiv_minus_brent_usd_per_bbl = fu_usd_per_bbl_equiv - brent_usd_per_bbl`
- `fu_bbl_equiv_minus_wti_usd_per_bbl = fu_usd_per_bbl_equiv - wti_usd_per_bbl`
- `lu_bbl_equiv_minus_brent_usd_per_bbl = lu_usd_per_bbl_equiv - brent_usd_per_bbl`
- `lu_bbl_equiv_minus_wti_usd_per_bbl = lu_usd_per_bbl_equiv - wti_usd_per_bbl`

## 8. Failure Behavior

首版 builder 必须：

- 缺任何必需输入时直接失败
- 单位不匹配时直接失败
- `as_of` 解析失败时直接失败
- 不做 silent fallback 到 chat memory 或 hardcoded market values

允许的唯一 hardcoded 值：

- `fu_bbl_per_ton`
- `lu_bbl_per_ton`
- `snapshot_mode`

## 9. Test Strategy

### 9.1 首版测试覆盖

`test_build_energy_cross_benchmark_spread_script.py` 至少覆盖：

1. 最小输入可成功产出 artifact
2. 公式计算正确
3. 各 input 的 `as_of` / `source` 被保留
4. 缺失 input / 错误单位会失败
5. `latest_energy_cross_benchmark_spread.json` 被更新

### 9.2 Validation command

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
PYTHONPATH=src pytest -q tests/test_build_energy_cross_benchmark_spread_script.py
```

## 10. Consumer Plan (Deferred)

首版只产出 artifact，不接 consumer。

后续优先级：

1. `build_operator_task_visual_panel.py` 只读消费
2. `build_dashboard_frontend_snapshot.py` 只读透传
3. `commodity_reasoning_summary` / `event_crisis_operator_summary` 再决定是否引用

## 11. Why This Is The Smallest Useful Cut

这个设计是当前最小、最稳的切口，因为它：

- 满足你要的“source-owned”
- 不把抓取链、解释链、UI 链一次性绑死
- 能快速为后续 consumer 提供稳定 contract
- 符合 Fenlie 里“prefer source-owned state, avoid UI re-derivation”的规则
