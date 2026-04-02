# Fenlie CPA Entry Visibility Design

> Date: 2026-04-03  
> Change class: `DOC_ONLY`

## Goal
让 `CPA 管理` 从“存在路由但不易发现”提升到“首屏可见、跨页可达、视觉层级明确”，且保持 Fenlie 现有壳层语言，不引入割裂的独立视觉体系。

## Chosen Approach
采用 **B 方案**：
1. 顶栏高亮入口
2. 总览页首屏强调卡片
3. 图谱主页快捷入口
4. 侧栏 utilities 固定入口

不采用悬浮按钮，避免对现有控制台阅读节奏造成持续打断。

## UX Intent
- 新用户：进入系统后 1 屏内能看见 CPA 入口
- 熟用户：在任意页面都能快速回到 CPA 管理
- 视觉语义：CPA 属于高频管理工具，但不是系统唯一主任务，因此应“明显但不喧宾夺主”

## Surface Changes
### 1. GlobalTopbar
- 保留主导航里的 `CPA 管理`
- 额外增加一个高亮型 action button，文案使用 `打开 CPA 管理`
- 该按钮放在搜索按钮附近，形成“全局工具入口”区

### 2. OverviewPage
- 在“下一步”入口区新增 CPA 强调卡片
- 文案突出：认证文件、配额、删除保护
- 与 `操作终端 / 研究工作区 / 契约验收`并列，但视觉上略强调

### 3. GraphHomePage
- 在现有快捷入口区域加入 CPA 快捷入口
- 文案指向“认证文件 / 配额 / 删除保护”
- 不把 CPA 变成图谱中心节点，只作为操作捷径

### 4. ContextSidebar
- 在 utilities 区加入固定 `CPA 管理 / Auth & Quota`
- 即使当前页面不在 CPA 域，也允许快速跳转
- 若当前已在 `/cpa`，仍保留链接但可通过 active 文案或高亮弱提示避免迷失

## Visual Rules
- 复用现有 `ActionButton / ActionLink / PanelCard / DomainTab / FilterChip` 风格
- 强调方式优先使用：
  - 边框强调
  - kicker 文案
  - 次级说明
  - 稍强的按钮色阶
- 不新增独立主题色系统，不做悬浮浮层，不做 toast 级干扰

## Testing
1. Topbar test：存在“打开 CPA 管理”高亮入口并指向 `/cpa`
2. Sidebar test：utilities 中存在 `CPA 管理 / Auth & Quota`
3. OverviewPage test：首屏入口区存在 CPA 强调卡片
4. GraphHomePage test：快捷入口区存在 CPA 管理入口
5. App-level smoke：`#/cpa` 仍可正常进入，不破坏现有主导航高亮

## Risks
- 若顶栏和主导航同时放 CPA，可能重复；因此高亮按钮应承担“快捷动作”语义，而不是再做第二个普通 tab
- Overview 首屏入口过多会稀释节奏，因此 CPA 卡片需维持与现有卡片同尺寸，不做 oversized hero
