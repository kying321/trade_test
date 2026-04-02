# Energy Cross-Benchmark Spread Artifact Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 Fenlie 增加一个最小 source-owned 能源跨基准价差 artifact，稳定产出 SC/FU/LU 相对 Brent/WTI 的价格层对比结果，并保留显式输入时间戳与来源。

**Architecture:** 采用 repo-owned 单脚本 builder + 单元测试 + review artifact 的最小闭包。builder 只消费显式 input JSON，不直接抓网页、不接 UI、不做解释层；输出 `*_energy_cross_benchmark_spread.json`、checksum 和 latest 指针，供后续 consumer 只读消费。

**Tech Stack:** Python 3、repo-owned scripts、JSON review artifacts、Pytest

---

## 0. 范围与边界

- change class 固定为 `RESEARCH_ONLY`
- 只实现 source-owned artifact，不接 consumer
- 只覆盖：SC / FU / LU / Brent / WTI / USDCNY
- 时间语义固定为：`latest_available_composite`
- 不做自动抓取，不接数据库，不接 dashboard，不接 operator panel
- 不做解释层，不做交易建议

## 1. 目标文件结构

### Create

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/build_energy_cross_benchmark_spread.py`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_build_energy_cross_benchmark_spread_script.py`

### Output

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/*_energy_cross_benchmark_spread.json`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/*_energy_cross_benchmark_spread_checksum.json`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/latest_energy_cross_benchmark_spread.json`

## 2. Task 1：写 failing tests 锁定 artifact contract

**Files:**
- Create: `/Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_build_energy_cross_benchmark_spread_script.py`

- [ ] **Step 1: 写最小成功用例 failing test**

```python
from __future__ import annotations

import json
import subprocess
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_energy_cross_benchmark_spread.py"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_energy_cross_benchmark_spread_writes_artifact(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(review_dir / "sc.json", {"price": 656.7, "unit": "CNY/bbl", "as_of": "2026-04-01T14:33:00+08:00", "source": "yicai"})
    _write_json(review_dir / "fu.json", {"price": 4001.0, "unit": "CNY/t", "as_of": "2026-04-01T14:33:00+08:00", "source": "yicai"})
    _write_json(review_dir / "lu.json", {"price": 4772.0, "unit": "CNY/t", "as_of": "2026-04-01T14:33:00+08:00", "source": "yicai"})
    _write_json(review_dir / "brent.json", {"price": 109.12, "unit": "USD/bbl", "as_of": "2026-04-02T13:02:00Z", "source": "reuters"})
    _write_json(review_dir / "wti.json", {"price": 112.60, "unit": "USD/bbl", "as_of": "2026-04-02T13:02:00Z", "source": "reuters"})
    _write_json(review_dir / "fx.json", {"price": 6.90, "unit": "CNY/USD", "as_of": "2026-04-02T09:00:00+08:00", "source": "manual"})

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir", str(review_dir),
            "--sc-json", str(review_dir / "sc.json"),
            "--fu-json", str(review_dir / "fu.json"),
            "--lu-json", str(review_dir / "lu.json"),
            "--brent-json", str(review_dir / "brent.json"),
            "--wti-json", str(review_dir / "wti.json"),
            "--fx-json", str(review_dir / "fx.json"),
            "--now", "2026-04-02T14:00:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["snapshot_mode"] == "latest_available_composite"
    assert payload["inputs"]["sc"]["source"] == "yicai"
    assert payload["inputs"]["brent"]["source"] == "reuters"
    assert Path(payload["artifact"]).exists()
    assert (review_dir / "latest_energy_cross_benchmark_spread.json").exists()
```

- [ ] **Step 2: 写公式正确性 failing test**

```python
def test_build_energy_cross_benchmark_spread_derives_spreads_correctly(tmp_path: Path) -> None:
    ...
    payload = json.loads(proc.stdout)
    assert round(float(payload["derived"]["sc_usd_per_bbl"]), 2) == 95.17
    assert round(float(payload["derived"]["fu_usd_per_bbl_equiv"]), 2) == 91.32
    assert round(float(payload["spreads"]["sc_minus_brent_usd_per_bbl"]), 2) == -13.95
    assert round(float(payload["spreads"]["lu_bbl_equiv_minus_wti_usd_per_bbl"]), 2) == -3.69
```

- [ ] **Step 3: 写失败路径 test**

```python
def test_build_energy_cross_benchmark_spread_rejects_unit_mismatch(tmp_path: Path) -> None:
    ...
    bad_fx = {"price": 6.90, "unit": "USD/CNY", "as_of": "2026-04-02T09:00:00+08:00", "source": "manual"}
    ...
    assert proc.returncode != 0
    assert "unit_mismatch" in (proc.stderr or proc.stdout)
```

- [ ] **Step 4: 跑 tests 确认红**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
PYTHONPATH=src pytest -q tests/test_build_energy_cross_benchmark_spread_script.py
```

Expected: FAIL（脚本不存在）

## 3. Task 2：最小实现 builder（RESEARCH_ONLY）

**Files:**
- Create: `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/build_energy_cross_benchmark_spread.py`

- [ ] **Step 1: 实现 parser 与显式输入参数**

要求：

- 支持：
  - `--review-dir`
  - `--sc-json`
  - `--fu-json`
  - `--lu-json`
  - `--brent-json`
  - `--wti-json`
  - `--fx-json`
  - `--now`
  - `--artifact-ttl-hours`
  - `--artifact-keep`
- 所有 price inputs 必填

- [ ] **Step 2: 实现 input 校验 helper**

要求：

- 校验 `price / unit / as_of / source` 都存在
- 校验单位严格匹配：
  - `SC -> CNY/bbl`
  - `FU -> CNY/t`
  - `LU -> CNY/t`
  - `Brent -> USD/bbl`
  - `WTI -> USD/bbl`
  - `USD/CNY -> CNY/USD`
- 任一项失败直接报错，不 silent fallback

- [ ] **Step 3: 实现 derive / spread 计算**

要求：

- 固定 assumptions：
  - `fu_bbl_per_ton = 6.35`
  - `lu_bbl_per_ton = 6.35`
- 输出：
  - `derived`
  - `spreads`
- 浮点保留原始值，不在 builder 内预先四舍五入为字符串

- [ ] **Step 4: 实现 artifact 写入**

要求：

- 写入：
  - `*_energy_cross_benchmark_spread.json`
  - `*_energy_cross_benchmark_spread_checksum.json`
- 同步更新：
  - `latest_energy_cross_benchmark_spread.json`
- checksum 至少包含：
  - artifact path
  - sha256
  - generated_at_utc

- [ ] **Step 5: 打印 JSON payload 到 stdout**

要求：

- stdout 输出最终 artifact payload
- 便于 tests 直接读取

## 4. Task 3：跑 tests 转绿（RESEARCH_ONLY）

**Files:**
- Test: `/Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_build_energy_cross_benchmark_spread_script.py`

- [ ] **Step 1: 跑 targeted tests**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
PYTHONPATH=src pytest -q tests/test_build_energy_cross_benchmark_spread_script.py
```

Expected: PASS

- [ ] **Step 2: 跑 public data refresh 现有回归，确认没有误伤**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
PYTHONPATH=src pytest -q tests/test_run_public_data_refresh_script.py
```

Expected: PASS

## 5. Task 4：文档化最小 contract（RESEARCH_ONLY）

**Files:**
- Modify: `/Users/jokenrobot/Downloads/Folders/fenlie/docs/superpowers/specs/2026-04-02-energy-cross-benchmark-spread-design.md`

- [ ] **Step 1: 若实现中字段名有微调，同步更新 spec**

要求：

- spec 中字段名、单位、时间语义与最终代码一致
- 不新增解释层承诺

- [ ] **Step 2: Commit**

```bash
git add \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/build_energy_cross_benchmark_spread.py \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_build_energy_cross_benchmark_spread_script.py \
  /Users/jokenrobot/Downloads/Folders/fenlie/docs/superpowers/specs/2026-04-02-energy-cross-benchmark-spread-design.md \
  /Users/jokenrobot/Downloads/Folders/fenlie/docs/superpowers/plans/2026-04-02-energy-cross-benchmark-spread-implementation-plan.md

git commit -m "feat(research): add energy cross benchmark spread artifact"
```

## 6. 执行说明

- 首版只做到 source-owned artifact
- 不在本计划内接入任何 consumer
- 后续若要接 `build_operator_task_visual_panel.py`，应另开一份小计划，只做只读消费，不在 UI 里重算 spread
