# CPA Automation Comparison

更新时间：2026-04-07（Asia/Shanghai）

## 目的

记录 Fenlie 当前 CPA 主线与第三方开源候选项目的可执行性、平台约束、参数依赖和成功率证据，避免后续线程重复试错。

---

## 一、对比对象

### A. Fenlie 当前主线

来源：

- `/Users/jokenrobot/Downloads/Folders/MAC工具/data/fenlie_main_thread_handoff_20260407.md`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/cpa_channels/`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/build_cpa_control_plane_snapshot.py`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_cpa_guarded_action.py`

当前主线能力：

- source-owned CPA channel kernel
- source-owned CPA control plane snapshot
- guarded actions / grouped drilldown / recent receipts

### B. 第三方候选：MAACodeX

来源：

- 仓库：`https://github.com/ZeroAd-06/MAACodeX`
- 本地探查目录：
  - `/Users/jokenrobot/Downloads/Folders/third_party/MAACodeX`

测试时 HEAD：

- `293ea0240bbba8703b59d70e12d1b7f3a8a98531`

---

## 二、MAACodeX 已验证事实

### 1. 安装/准备阶段：通过

在当前机器上已完成：

- `git clone`
- `git submodule update --init --recursive`
- `npm ci`
- `python3 -m pip install -r ./scripts/requirements.txt`
- `python3 ./scripts/configure.py`

其中 OCR 配置明确成功：

- 输出：`OCR model configured.`

### 2. 运行阶段：当前环境不可用

#### 尝试 1：不指定浏览器

命令：

```bash
npm run run:codex -- --config /tmp/maa-test-task.json
```

结果：

- 失败
- 报错：
  - `failed to locate Chrome/Chromium automatically, set browser_executable in config or CHROME_PATH`

#### 尝试 2：指定本机 Chrome

配置：

- `browser_executable = /Applications/Google Chrome.app/Contents/MacOS/Google Chrome`

结果：

- 仍失败
- 报错：
  - `spawnSync powershell.exe ENOENT`

### 3. 平台约束是硬阻塞，不是偶发错误

直接证据：

- `README.md` 明确标注：
  - `Platform-Windows`
  - `在 Windows 桌面浏览器中自动执行`
- `assets/interface.json` 中 controller 类型为：
  - `Win32`
- `scripts/install.py` 明确限制：
  - `MAACodex release packaging is only supported for Windows.`

结论：

- 在当前 **macOS** 环境下，MAACodeX 无法进入真实任务执行阶段
- 当前失败是 **平台不匹配**，不是单纯配置错误

### 4. 参数仍不完整

MAACodeX 要求至少 4 个关键凭证：

- `relay_csrf_token`
- `relay_session_id`
- `relay_cf_api_url`
- `relay_cf_token`

当前只提供了：

- `csrftoken`
- `sessionid`

仍缺：

- `relay_cf_api_url`
- `relay_cf_token`

所以即使平台匹配，也还不能完成完整注册/验证码/登录链

---

## 三、成功率对比

### 1. 当前机器（macOS）上的可执行成功率

#### MAACodeX

- 安装/准备阶段：`5 / 5`
- 任务执行阶段：`0 / 2`

结论：

- 当前机器上的 **任务执行成功率 = 0%**

#### Fenlie / MAC工具 当前 live

根据 handoff：

- 当前 live 验收：
  - `active_target_authfiles = 0`

结论：

- 当前 live 环境也不能直接作为成功链使用

### 2. 历史已验证成功率

#### Fenlie / MAC工具 主线

已验证历史成功快照：

- `/Users/jokenrobot/Downloads/Folders/MAC工具/data/active20_acceptance_20260329_212018.json`
- `/Users/jokenrobot/Downloads/Folders/MAC工具/data/registered_success_active20.csv`

结论：

- 历史成功 `20 / 20`
- 历史闭环成功率：**100%**

#### MAACodeX

本轮没有拿到任何真实任务成功证据。

---

## 四、当前主线选择建议

### 结论

**不要把 MAACodeX 作为当前 Fenlie CPA 主线替代。**

原因：

1. 当前平台不匹配（Windows/Win32/PowerShell 依赖）
2. 参数不完整（缺少 CF temp mail 2 个关键参数）
3. 没有取得当前环境下的真实任务成功证据
4. Fenlie / MAC工具 主线已有明确历史成功集和 source-owned handoff

### 正确定位

MAACodeX 当前应被定位为：

- **Windows 专用候选对照组**
- 只有在以下条件齐备时才值得继续做成功率比较：
  - Windows 环境
  - MaaFramework runtime 准备完整
  - `relay_cf_api_url`
  - `relay_cf_token`
  - 单独的批量成功率测试样本

---

## 五、主线程操作建议

### 如果目标是继续推进 Fenlie 主线

继续沿用：

- `MAC工具` 历史成功快照
- `Fenlie cpa_channels kernel`
- `Fenlie cpa_control_plane_snapshot`

### 如果目标是继续验证 MAACodeX

应切到独立对照环境：

- Windows
- Chrome
- 完整 relay/CF mail 4 参数
- 不与当前 Fenlie live 管理面共用结论

---

## 六、source-of-truth 结论

当前关于 CPA 自动化能力的 authority 顺序应为：

1. `MAC工具` 历史成功快照（20-active）
2. Fenlie 主仓 `cpa_channels` / `cpa_control_plane_snapshot`
3. 当前 live inventory / review queue
4. MAACodeX 对照探测结果

MAACodeX 当前只能证明：

- 安装准备可完成
- 当前 macOS 环境下无法进入真实任务执行

不能证明：

- 它在 Fenlie 当前环境下优于现有主线

