# Fenlie Read-Only Share Directory Manifest

> 版本：2026-03-24  
> 用途：定义 Fenlie 对外共享盘 / data room / Notion 附件区的建议目录结构，避免把工作草稿、尽调深层材料或错误版本直接暴露给外部。  
> change class：`DOC_ONLY`

---

## 1. 一句话原则

外部共享目录必须是**只读、分层、可审计**的。

不要把整个 `/Users/jokenrobot/Downloads/Folders/fenlie/docs/investor` 直接暴露给外部。

---

## 2. 推荐目录结构

建议未来外部共享目录固定为：

```text
Fenlie_External_Share/
├── 01_Overview/
├── 02_Preread/
├── 03_Followup/
├── 04_Diligence/
└── 99_Internal_Not_For_Send/
```

其中：

- `01_Overview`：首次触达最小包
- `02_Preread`：会前预读
- `03_Followup`：会后补发
- `04_Diligence`：已进入认真评估后再开放
- `99_Internal_Not_For_Send`：只做内部缓存，不对外开放

---

## 3. 每层建议放什么

### `01_Overview/`

建议放：

- `Fenlie_Executive_Summary_2026-03-24.pdf`
- `Fenlie_Market_Landscape_2026-03-24.pdf`

对应内部源文件：

- `2026-03-24-fenlie-executive-summary.md`
- `2026-03-24-fenlie-market-and-competitive-landscape.md`

---

### `02_Preread/`

建议放：

- `Fenlie_Investor_Deck_Preread_2026-03-24.pdf`
- `Fenlie_Product_Architecture_Visual_2026-03-24.pdf`
- `Fenlie_12_18_Month_Roadmap_2026-03-24.pdf`

对应内部源文件：

- `2026-03-24-fenlie-investor-deck-preread-copy.md`
- `2026-03-24-fenlie-product-architecture-visual-brief.md`
- `2026-03-24-fenlie-12-18-month-milestone-roadmap.md`

---

### `03_Followup/`

建议放：

- `Fenlie_Investor_Deck_Followup_2026-03-24.pdf`
- `Fenlie_Business_Plan_2026-03-24.pdf`
- `Fenlie_Fundraise_And_Runway_2026-03-24.pdf`
- `Fenlie_Execution_Maturity_Note_2026-03-24.pdf`

对应内部源文件：

- `2026-03-24-fenlie-investor-deck-followup-copy.md`
- `2026-03-24-fenlie-business-plan.md`
- `2026-03-24-fenlie-fundraise-scenarios-and-runway.md`
- `2026-03-24-fenlie-execution-maturity-note.md`

---

### `04_Diligence/`

建议放：

- `Fenlie_Data_Room_Index_2026-03-24.pdf`
- `Fenlie_Evidence_Matrix_2026-03-24.pdf`
- `Fenlie_Diligence_Readiness_2026-03-24.pdf`
- `Fenlie_Risk_Governance_2026-03-24.pdf`
- `Fenlie_Pilot_Profile_2026-03-24.pdf`
- `Fenlie_Pilot_Acceptance_Template_2026-03-24.pdf`
- `Fenlie_Investor_FAQ_Expanded_2026-03-24.pdf`

对应内部源文件：

- `2026-03-24-fenlie-data-room-index-and-request-list.md`
- `2026-03-24-fenlie-investor-evidence-matrix.md`
- `2026-03-24-fenlie-diligence-readiness-checklist.md`
- `2026-03-24-fenlie-risk-and-governance-principles-one-pager.md`
- `2026-03-24-fenlie-pilot-customer-profile-one-pager.md`
- `2026-03-24-fenlie-pilot-milestone-and-acceptance-template.md`
- `2026-03-24-fenlie-investor-faq-expanded.md`

---

### `99_Internal_Not_For_Send/`

建议放：

- markdown 源文稿
- visual brief
- email pack
- send-pack manifest
- 版本校验清单

这一层不应直接对外分享。

---

## 4. 权限规则

### 默认规则

- 外部目录全部只读
- 不给对方编辑权限
- 不给整个工作区根目录
- 不给包含临时草稿的共享入口

### 开放顺序

1. 先开 `01_Overview`
2. 再开 `02_Preread`
3. 会后再开 `03_Followup`
4. 确认进入认真评估后再开 `04_Diligence`

---

## 5. 命名规则

对外成品建议统一使用：

- 英文文件名
- 带日期
- 不出现 `v_final_final2`
- 不直接暴露内部 markdown 文件名

---

## 6. 红线

不应进入外部共享目录的内容包括：

1. 工作中草稿
2. 内部聊天记录
3. 未审计经营数字
4. 虚构 traction 或估值推演
5. 任何 live execution 运行说明或敏感配置

---

## 7. 当前状态

截至 2026-03-24：

- investor narrative：已具备
- send pack governance：已具备
- deck preread / followup 文稿：已具备
- read-only share 目录规则：已具备
- 外部共享成品目录：**仍待落地**

