#!/usr/bin/env python3
"""Hippocampus Darwin Compressor.

Protocol:
- Delete low-salience daily logs aggressively.
- Preserve only pain/euphoria/survival logs.
- Extract strict six-tuple genes and inject them into memory/genes.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

TRADER_ROOT = Path(
    os.getenv("TRADER_WORKSPACE_ROOT", str(Path(__file__).resolve().parents[1]))
)
MEMORY_DIR = Path(os.getenv("HIPPOCAMPUS_MEMORY_DIR", str(TRADER_ROOT / "memory")))
ARCHIVE_DIR = MEMORY_DIR / "archive"
GENE_DIR = MEMORY_DIR / "genes"
GENE_INDEX = GENE_DIR / "index.jsonl"
GENE_CONFLICTS = GENE_DIR / "conflicts.jsonl"
ACTIVE_RULES = GENE_DIR / "active_rules.json"

SALIENCE_WEIGHTS = {
    "SURVIVAL": 30,
    "HIBERNATE": 22,
    "STABILIZE": 18,
    "LIQUIDATION": 25,
    "CRITICAL": 20,
    "ERROR": 16,
    "FAIL": 12,
    "PAIN": 15,
    "DRAWDOWN": 16,
    "STOP": 8,
    "LOSS": 10,
    "亏": 10,
    "痛": 12,
    "EUPHORIA": 12,
    "WIN": 8,
    "盈利": 8,
    "CONVEXITY": 14,
    "SHOCK": 12,
    "THRESHOLD": 10,
}


def list_daily_logs(memory_dir: Path) -> List[Path]:
    return sorted(memory_dir.glob("20??-??-??.md"))


def _extract_numeric_stress(text: str) -> float:
    score = 0.0
    # numbers like -12.4%, +8.2%, drawdown 3.4, pnl -120
    for match in re.findall(r"[-+]?\d+(?:\.\d+)?\s*%", text):
        v = abs(float(match.replace("%", "").strip()))
        score += min(20.0, v)
    for match in re.findall(r"(?:pnl|drawdown|亏损|盈利|loss|profit)\s*[:=]?\s*[-+]?\d+(?:\.\d+)?", text, flags=re.IGNORECASE):
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", match)
        if nums:
            score += min(20.0, abs(float(nums[-1])))
    return min(score, 40.0)


def calculate_salience(path: Path) -> Dict[str, Any]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    upper = raw.upper()

    score = 0.0
    hits: Dict[str, int] = {}
    for token, weight in SALIENCE_WEIGHTS.items():
        cnt = upper.count(token.upper())
        if cnt > 0:
            hits[token] = cnt
            score += cnt * weight

    score += _extract_numeric_stress(raw)

    return {
        "path": str(path),
        "score": round(score, 3),
        "hits": hits,
        "text": raw,
    }


def _pick_line(lines: List[str], patterns: List[str], default: str) -> str:
    for line in lines:
        if not line:
            continue
        if line.startswith("#"):
            continue
        if line.startswith("|"):
            continue
        if re.fullmatch(r"[-=*_`~\s]+", line):
            continue
        if len(line.strip()) < 6:
            continue
        for p in patterns:
            if re.search(p, line, flags=re.IGNORECASE):
                return line.strip()
    return default


def _pick_prefixed(lines: List[str], keys: List[str]) -> Optional[str]:
    keys_l = [k.lower() for k in keys]
    for line in lines:
        l = line.strip()
        if not l:
            continue
        # normalize markdown bullets/emphasis and chinese colon
        norm = re.sub(r"^\s*[-*]\s*", "", l)
        norm = norm.replace("：", ":")
        norm = norm.replace("**", "").replace("__", "")
        low = norm.lower()
        for k in keys_l:
            if low.startswith(k):
                return norm
    return None


def _extract_gene_heuristic(text: str, source: str) -> Dict[str, str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    invariant_pref = _pick_prefixed(lines, ["Invariant:", "原则:", "规则:", "风险警示:"])
    trigger_pref = _pick_prefixed(lines, ["Trigger:", "触发:", "条件:"])
    action_pref = _pick_prefixed(lines, ["Action:", "动作:", "应对:"])
    rationale_pref = _pick_prefixed(lines, ["Rationale:", "原理:", "原因:"])
    failure_pref = _pick_prefixed(lines, ["Failure domain:", "失效域:", "失效条件:"])

    invariant = _pick_line(
        lines,
        [r"必须", r"always", r"never", r"原则", r"止损", r"risk", r"风险警示", r"底线"],
        "Invariant: 下行有界优先于收益追逐，任何异常先保全可选性。",
    )
    if invariant_pref:
        invariant = invariant_pref

    trigger = _pick_line(
        lines,
        [r"当", r"if", r"when", r"threshold", r"突破", r"跌破", r"shock", r"error", r"触发", r"条件"],
        "Trigger: 当预测度骤降或系统资源拥塞时触发防守模式。",
    )
    if trigger_pref:
        trigger = trigger_pref

    action = _pick_line(
        lines,
        [r"执行", r"action", r"stop", r"减仓", r"平仓", r"切换", r"freeze", r"断开", r"买入", r"卖出", r"开仓", r"加仓", r"观望", r"休眠", r"hibernat", r"stabilize"],
        "Action: 降级为 OBSERVE，仅允许微探针或模拟执行。",
    )
    if action_pref:
        action = action_pref

    rationale = _pick_line(
        lines,
        [r"因为", r"因此", r"so that", r"cause", r"由于", r"原因", r"原理"],
        "Rationale: 先阻断尾部损失，再等待状态恢复后逐步重建暴露。",
    )
    if rationale_pref:
        rationale = rationale_pref

    evidence_bits = [f"source={source}"]
    for line in lines:
        if line.startswith("#") or line.startswith("|"):
            continue
        if re.search(r"(亏|盈|loss|profit|drawdown|error|latency|shock|threshold)", line, flags=re.IGNORECASE):
            evidence_bits.append(line)
        if len(evidence_bits) >= 3:
            break
    evidence = " | ".join(evidence_bits)

    failure_domain = _pick_line(
        lines,
        [r"除非", r"unless", r"失效", r"无效", r"仅在", r"前提"],
        "Failure domain: 仅在数据完整、执行链可回滚且流动性正常时适用。",
    )
    if failure_pref:
        failure_domain = failure_pref

    return {
        "Invariant": invariant,
        "Trigger": trigger,
        "Action": action,
        "Rationale": rationale,
        "Evidence": evidence,
        "Failure domain": failure_domain,
    }


def _extract_gene_with_llm(text: str, source: str) -> Optional[Dict[str, str]]:
    if os.getenv("HIPPOCAMPUS_USE_LLM", "0") != "1":
        return None

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return None

    model = os.getenv("HIPPOCAMPUS_LLM_MODEL", "gpt-4o-mini")
    base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

    prompt = (
        "You are a strict memory compressor. "
        "Extract one survival gene as JSON with exact keys: "
        "Invariant, Trigger, Action, Rationale, Evidence, Failure domain. "
        "No extra keys, no markdown.\n"
        f"source={source}\n"
        "text:\n"
        f"{text[:6000]}"
    )

    try:
        r = requests.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": "Return strict JSON only."},
                    {"role": "user", "content": prompt},
                ],
                "response_format": {"type": "json_object"},
            },
            timeout=25,
        )
        if r.status_code >= 300:
            return None

        payload = r.json()
        content = (
            payload.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        if not content:
            return None
        obj = json.loads(content)
        if not isinstance(obj, dict):
            return None
        return {k: str(obj.get(k, "")).strip() for k in ["Invariant", "Trigger", "Action", "Rationale", "Evidence", "Failure domain"]}
    except Exception:
        return None


def extract_gene(text: str, source: str) -> Dict[str, str]:
    llm_gene = _extract_gene_with_llm(text, source)
    gene = llm_gene if llm_gene else _extract_gene_heuristic(text, source)

    required = ["Invariant", "Trigger", "Action", "Rationale", "Evidence", "Failure domain"]
    for key in required:
        if not gene.get(key):
            gene[key] = f"{key}: unavailable"
    return gene


def _norm_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff _:-]", "", s)
    return s


def _hash_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def _gene_signature(gene: Dict[str, str]) -> str:
    inv = _norm_text(gene.get("Invariant", ""))
    trig = _norm_text(gene.get("Trigger", ""))
    act = _norm_text(gene.get("Action", ""))
    return _hash_text(f"{inv}|{trig}|{act}")


def _trigger_signature(gene: Dict[str, str]) -> str:
    trig = _norm_text(gene.get("Trigger", ""))
    return _hash_text(trig)


def _action_polarity(action: str) -> str:
    a = _norm_text(action)
    risk_off_kw = [
        "减仓",
        "平仓",
        "清仓",
        "停止",
        "freeze",
        "flat",
        "observe",
        "stabilize",
        "hibernate",
        "derisk",
        "de risk",
        "stop",
        "cut",
    ]
    risk_on_kw = [
        "加仓",
        "开仓",
        "做多",
        "追涨",
        "满仓",
        "buy",
        "long",
        "enter",
        "increase",
        "act",
        "all in",
    ]
    if any(k in a for k in risk_off_kw):
        return "risk_off"
    if any(k in a for k in risk_on_kw):
        return "risk_on"
    return "other"


def _load_gene_catalog() -> List[Dict[str, Any]]:
    catalog: List[Dict[str, Any]] = []
    if not GENE_INDEX.exists():
        return catalog
    for raw in GENE_INDEX.read_text(encoding="utf-8", errors="ignore").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        gene = obj.get("gene", {}) if isinstance(obj, dict) else {}
        if not isinstance(gene, dict):
            continue
        record = {
            "signature": obj.get("signature") or _gene_signature(gene),
            "trigger_signature": obj.get("trigger_signature") or _trigger_signature(gene),
            "action_polarity": obj.get("action_polarity") or _action_polarity(gene.get("Action", "")),
            "gene_path": obj.get("gene_path"),
            "source_log": obj.get("source_log"),
        }
        catalog.append(record)
    return catalog


def _load_gene_entries() -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if not GENE_INDEX.exists():
        return entries
    for raw in GENE_INDEX.read_text(encoding="utf-8", errors="ignore").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        gene = obj.get("gene", {})
        if not isinstance(gene, dict):
            continue
        obj["signature"] = obj.get("signature") or _gene_signature(gene)
        obj["trigger_signature"] = obj.get("trigger_signature") or _trigger_signature(gene)
        obj["action_polarity"] = obj.get("action_polarity") or _action_polarity(gene.get("Action", ""))
        entries.append(obj)
    return entries


def _ts_from_filename(path: Path) -> str:
    m = re.match(r"^(\d{4}-\d{2}-\d{2})", path.name)
    if m:
        return f"{m.group(1)}T00:00:00+00:00"
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _load_archive_gene_entries(max_files: int = 120) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if not ARCHIVE_DIR.exists():
        return entries
    files = sorted(ARCHIVE_DIR.glob("20??-??-??.md"))[-max_files:]
    for p in files:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        gene = extract_gene(text, source=p.name)
        entry = {
            "ts": _ts_from_filename(p),
            "source_log": p.name,
            "gene": gene,
            "signature": _gene_signature(gene),
            "trigger_signature": _trigger_signature(gene),
            "action_polarity": _action_polarity(gene.get("Action", "")),
            "source": "archive_rebuild",
        }
        entries.append(entry)
    return entries


def _polarity_priority(p: str) -> int:
    # Antifragile preference: risk_off rules outrank risk_on for same trigger.
    if p == "risk_off":
        return 3
    if p == "other":
        return 2
    if p == "risk_on":
        return 1
    return 0


def _is_noise_line(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return True
    if s.startswith("#") or s.startswith("|"):
        return True
    if re.fullmatch(r"[-=*_`~\s]+", s):
        return True
    if re.match(r"^(?:\d+[\.\)]\s*)?[第]?\s*\d+\s*(?:章|节)?\s*$", s):
        return True
    return False


def _gene_quality_ok(gene: Dict[str, Any]) -> bool:
    action = str(gene.get("Action") or "")
    trigger = str(gene.get("Trigger") or "")
    invariant = str(gene.get("Invariant") or "")
    if _is_noise_line(action) or _is_noise_line(trigger) or _is_noise_line(invariant):
        return False
    if len(action.strip()) < 8 or len(trigger.strip()) < 8:
        return False
    return True


def rebuild_active_rules() -> Dict[str, Any]:
    GENE_DIR.mkdir(parents=True, exist_ok=True)
    index_entries = _load_gene_entries()
    archive_entries = _load_archive_gene_entries()
    entries = index_entries + archive_entries
    quality_fallback = False
    strict_entries: List[Dict[str, Any]] = []
    for e in entries:
        gene_obj = e.get("gene") if isinstance(e.get("gene"), dict) else {}
        if _gene_quality_ok(gene_obj):
            strict_entries.append(e)
    if not strict_entries and entries:
        strict_entries = entries
        quality_fallback = True

    selected: Dict[str, Dict[str, Any]] = {}
    for e in strict_entries:
        trig = str(e.get("trigger_signature") or "")
        if not trig:
            continue
        cur = selected.get(trig)
        if cur is None:
            selected[trig] = e
            continue

        p_new = _polarity_priority(str(e.get("action_polarity") or ""))
        p_cur = _polarity_priority(str(cur.get("action_polarity") or ""))
        if p_new > p_cur:
            selected[trig] = e
            continue
        if p_new == p_cur:
            if str(e.get("ts") or "") > str(cur.get("ts") or ""):
                selected[trig] = e

    active = sorted(selected.values(), key=lambda x: str(x.get("ts") or ""), reverse=True)
    payload = {
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        "active_count": len(active),
        "source_total_entries": len(entries),
        "source_index_entries": len(index_entries),
        "source_archive_entries": len(archive_entries),
        "quality_fallback": quality_fallback,
        "rules": [
            {
                "signature": r.get("signature"),
                "trigger_signature": r.get("trigger_signature"),
                "action_polarity": r.get("action_polarity"),
                "source_log": r.get("source_log"),
                "ts": r.get("ts"),
                "gene": r.get("gene"),
            }
            for r in active
        ],
    }
    ACTIVE_RULES.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"active_count": len(active), "source_total_entries": len(entries), "path": str(ACTIVE_RULES)}


def _classify_gene(gene: Dict[str, str], catalog: List[Dict[str, Any]]) -> Dict[str, Any]:
    sig = _gene_signature(gene)
    trig_sig = _trigger_signature(gene)
    polarity = _action_polarity(gene.get("Action", ""))

    for rec in catalog:
        if rec.get("signature") == sig:
            return {"status": "duplicate", "signature": sig, "trigger_signature": trig_sig, "action_polarity": polarity, "against": rec}

    for rec in catalog:
        if rec.get("trigger_signature") != trig_sig:
            continue
        existing_pol = str(rec.get("action_polarity") or "other")
        if polarity != "other" and existing_pol != "other" and polarity != existing_pol:
            return {"status": "conflict", "signature": sig, "trigger_signature": trig_sig, "action_polarity": polarity, "against": rec}

    return {"status": "new", "signature": sig, "trigger_signature": trig_sig, "action_polarity": polarity, "against": None}


def _archive_log(log_path: Path) -> Path:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    archived_path = ARCHIVE_DIR / log_path.name
    shutil.move(str(log_path), str(archived_path))
    return archived_path


def _record_conflict(
    *,
    log_path: Path,
    salience: Dict[str, Any],
    gene: Dict[str, str],
    classify: Dict[str, Any],
) -> None:
    GENE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_log": str(log_path.name),
        "salience_score": salience.get("score"),
        "gene": gene,
        "signature": classify.get("signature"),
        "trigger_signature": classify.get("trigger_signature"),
        "action_polarity": classify.get("action_polarity"),
        "conflict_with": classify.get("against"),
    }
    with GENE_CONFLICTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def persist_gene(
    log_path: Path,
    salience: Dict[str, Any],
    gene: Dict[str, str],
    catalog: List[Dict[str, Any]],
) -> Dict[str, Any]:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    GENE_DIR.mkdir(parents=True, exist_ok=True)

    classify = _classify_gene(gene, catalog)
    status = str(classify.get("status"))

    # Always archive salient source logs to avoid reprocessing loops.
    _archive_log(log_path)

    if status == "duplicate":
        return {"status": "duplicate", "gene_path": None, "classify": classify}

    if status == "conflict":
        _record_conflict(log_path=log_path, salience=salience, gene=gene, classify=classify)
        return {"status": "conflict", "gene_path": None, "classify": classify}

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    gene_path = GENE_DIR / f"{ts}_{log_path.stem}_gene.json"

    payload = {
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_log": str(log_path.name),
        "salience_score": salience.get("score"),
        "salience_hits": salience.get("hits"),
        "signature": classify.get("signature"),
        "trigger_signature": classify.get("trigger_signature"),
        "action_polarity": classify.get("action_polarity"),
        "gene_path": str(gene_path),
        "gene": gene,
    }

    gene_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with GENE_INDEX.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    catalog.append(
        {
            "signature": payload["signature"],
            "trigger_signature": payload["trigger_signature"],
            "action_polarity": payload["action_polarity"],
            "gene_path": str(gene_path),
            "source_log": payload["source_log"],
        }
    )
    return {"status": "new", "gene_path": gene_path, "classify": classify}


def hippocampus_cycle() -> Dict[str, Any]:
    logs = list_daily_logs(MEMORY_DIR)
    if not logs:
        active = rebuild_active_rules()
        return {"status": "ok", "message": "no_logs", "processed": 0, "active_rules": active}

    threshold = float(os.getenv("HIPPOCAMPUS_SALIENCE_THRESHOLD", "24"))

    deleted = 0
    preserved = 0
    dedup_skipped = 0
    conflict_skipped = 0
    genes: List[str] = []
    catalog = _load_gene_catalog()

    for lp in logs:
        salience = calculate_salience(lp)
        score = float(salience["score"])

        if score < threshold:
            lp.unlink(missing_ok=True)
            deleted += 1
            continue

        gene = extract_gene(salience["text"], source=lp.name)
        out = persist_gene(lp, salience, gene, catalog=catalog)
        status = str(out.get("status") or "")
        if status == "new":
            preserved += 1
            if out.get("gene_path"):
                genes.append(str(out["gene_path"]))
        elif status == "duplicate":
            dedup_skipped += 1
        elif status == "conflict":
            conflict_skipped += 1

    summary = {
        "status": "ok",
        "threshold": threshold,
        "processed": len(logs),
        "deleted": deleted,
        "preserved": preserved,
        "dedup_skipped": dedup_skipped,
        "conflict_skipped": conflict_skipped,
        "genes": genes,
    }
    summary["active_rules"] = rebuild_active_rules()
    return summary


def main() -> None:
    summary = hippocampus_cycle()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
