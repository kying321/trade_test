import React, { Component, useEffect, useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";

const SNAPSHOT_PATH = "/data/fenlie_dashboard_snapshot.json";
const MAIN_TABS = [
  { id: "overview", label: "总览" },
  { id: "backtests", label: "回测工作台" },
  { id: "explorer", label: "工件穿透" },
  { id: "raw", label: "原始 JSON" },
];
const DETAIL_TABS = [
  { id: "summary", label: "摘要" },
  { id: "fields", label: "字段" },
  { id: "raw", label: "Raw" },
];

function formatNumber(value, digits = 3) {
  if (value === null || value === undefined || value === "") return "—";
  const num = Number(value);
  if (!Number.isFinite(num)) return String(value);
  return num.toLocaleString("zh-CN", {
    maximumFractionDigits: digits,
    minimumFractionDigits: Math.abs(num) < 10 && digits > 0 ? Math.min(2, digits) : 0,
  });
}

function formatPercent(value, digits = 2) {
  if (value === null || value === undefined || value === "") return "—";
  const num = Number(value);
  if (!Number.isFinite(num)) return String(value);
  return `${(num * 100).toFixed(digits)}%`;
}

function safeDisplayValue(value) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  if (Array.isArray(value)) {
    if (!value.length) return "[]";
    const primitivesOnly = value.every((item) => ["string", "number", "boolean"].includes(typeof item));
    return primitivesOnly ? value.join(", ") : `[${value.length} items]`;
  }
  if (typeof value === "object") {
    if (typeof value.brief === "string") return value.brief;
    if (value.symbol || value.action || value.state) {
      return [value.symbol, value.action, value.state].filter(Boolean).join(" / ");
    }
    return `{${Object.keys(value).length} keys}`;
  }
  return String(value);
}

function statusTone(value) {
  const text = String(value || "").toLowerCase();
  if (!text) return "neutral";
  if (["ok", "allowed", "active", "passed", "promising_small_sample", "mixed_positive", "ready"].some((key) => text.includes(key))) {
    return "positive";
  }
  if (["blocked", "demoted", "failed", "negative", "harmful", "overblocking", "error"].some((key) => text.includes(key))) {
    return "negative";
  }
  if (["watch", "candidate", "mixed", "partial", "warning", "inconclusive", "local", "defer"].some((key) => text.includes(key))) {
    return "warning";
  }
  return "neutral";
}

function badgeClass(value) {
  return `badge badge-${statusTone(value)}`;
}

function cardList(items = []) {
  return items.filter((item) => item && item.value !== undefined && item.value !== null && item.value !== "");
}

function downsampleCurve(curve = []) {
  if (!Array.isArray(curve)) return [];
  return curve.map((point) => ({
    date: point.date ?? point.ts ?? point.time ?? "",
    equity: Number(point.equity ?? point.value ?? 0),
  }));
}

function buildEquityOption(backtest) {
  const curve = downsampleCurve(backtest?.equity_curve_sample || []);
  let peak = 0;
  const drawdowns = curve.map((point) => {
    peak = Math.max(peak, point.equity);
    const dd = peak > 0 ? point.equity / peak - 1 : 0;
    return { date: point.date, drawdown: dd };
  });
  return {
    backgroundColor: "transparent",
    tooltip: { trigger: "axis" },
    legend: { top: 0, textStyle: { color: "#d7e6ff" } },
    grid: [
      { left: 50, right: 20, top: 40, height: "52%" },
      { left: 50, right: 20, top: "72%", height: "16%" },
    ],
    xAxis: [
      {
        type: "category",
        boundaryGap: false,
        data: curve.map((item) => item.date),
        axisLine: { lineStyle: { color: "rgba(160,185,215,0.35)" } },
        axisLabel: { color: "#aac1e6", hideOverlap: true },
      },
      {
        type: "category",
        boundaryGap: false,
        gridIndex: 1,
        data: drawdowns.map((item) => item.date),
        axisLine: { lineStyle: { color: "rgba(160,185,215,0.35)" } },
        axisLabel: { color: "#aac1e6", hideOverlap: true },
      },
    ],
    yAxis: [
      {
        type: "value",
        scale: true,
        axisLabel: { color: "#aac1e6" },
        splitLine: { lineStyle: { color: "rgba(160,185,215,0.12)" } },
      },
      {
        type: "value",
        gridIndex: 1,
        axisLabel: { color: "#aac1e6", formatter: (value) => `${(value * 100).toFixed(0)}%` },
        splitLine: { lineStyle: { color: "rgba(160,185,215,0.12)" } },
      },
    ],
    series: [
      {
        name: "权益曲线",
        type: "line",
        smooth: true,
        showSymbol: false,
        data: curve.map((item) => item.equity),
        lineStyle: { width: 2.2, color: "#7ee0c6" },
        areaStyle: { color: "rgba(126,224,198,0.12)" },
      },
      {
        name: "回撤",
        type: "line",
        xAxisIndex: 1,
        yAxisIndex: 1,
        smooth: true,
        showSymbol: false,
        data: drawdowns.map((item) => item.drawdown),
        lineStyle: { width: 1.8, color: "#ff8b8b" },
        areaStyle: { color: "rgba(255,139,139,0.12)" },
      },
    ],
  };
}

function buildComparisonOption(rows = []) {
  const prepared = rows.slice(0, 10);
  return {
    backgroundColor: "transparent",
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
    legend: { top: 0, textStyle: { color: "#d7e6ff" } },
    grid: { left: 50, right: 20, top: 42, bottom: 80 },
    xAxis: {
      type: "category",
      data: prepared.map((row) => row.strategy_id_canonical || row.strategy_id || row.row_id),
      axisLabel: { color: "#aac1e6", rotate: 24, interval: 0 },
      axisLine: { lineStyle: { color: "rgba(160,185,215,0.35)" } },
    },
    yAxis: [
      {
        type: "value",
        axisLabel: { color: "#aac1e6", formatter: (value) => `${(value * 100).toFixed(0)}%` },
        splitLine: { lineStyle: { color: "rgba(160,185,215,0.12)" } },
      },
      {
        type: "value",
        axisLabel: { color: "#aac1e6" },
        splitLine: { show: false },
      },
    ],
    series: [
      {
        name: "总收益",
        type: "bar",
        data: prepared.map((row) => Number(row.total_return ?? 0)),
        itemStyle: { color: "#6ea8ff" },
      },
      {
        name: "Expectancy(R)",
        type: "line",
        yAxisIndex: 1,
        smooth: true,
        data: prepared.map((row) => Number(row.expectancy_r ?? 0)),
        itemStyle: { color: "#7ee0c6" },
        lineStyle: { width: 2.2 },
      },
    ],
  };
}

function KeyValueGrid({ items = [], compact = false }) {
  return (
    <div className={`kv-grid ${compact ? "kv-grid-compact" : ""}`}>
      {items.map((item) => (
        <div className="metric-card" key={item.label}>
          <span className="metric-label">{item.label}</span>
          <strong className="metric-value">{safeDisplayValue(item.value)}</strong>
          {item.hint ? <span className="metric-hint">{item.hint}</span> : null}
        </div>
      ))}
    </div>
  );
}

function SimpleTable({ columns = [], rows = [], empty = "暂无数据" }) {
  if (!rows.length) return <div className="empty-card">{empty}</div>;
  return (
    <div className="table-shell">
      <table className="data-table">
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column.key}>{column.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, rowIndex) => (
            <tr key={row.id || row.path || rowIndex}>
              {columns.map((column) => (
                <td key={column.key}>
                  {column.render ? column.render(row[column.key], row) : safeDisplayValue(row[column.key])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function describeNodeType(value) {
  if (Array.isArray(value)) return `array(${value.length})`;
  if (value === null) return "null";
  return typeof value;
}

function summarizeNode(value) {
  if (Array.isArray(value)) {
    if (!value.length) return "[]";
    const primitiveOnly = value.every((item) => item === null || ["string", "number", "boolean"].includes(typeof item));
    return primitiveOnly ? value.map((item) => safeDisplayValue(item)).join(", ") : `[${value.length} items]`;
  }
  if (value && typeof value === "object") {
    if (typeof value.brief === "string") return value.brief;
    return `{${Object.keys(value).length} keys}`;
  }
  return safeDisplayValue(value);
}

function readNodeAtPath(payload, path = []) {
  return path.reduce((current, segment) => {
    if (current === null || current === undefined) return undefined;
    return current[segment];
  }, payload);
}

function buildFieldRows(node, path = []) {
  if (!node || typeof node !== "object") return [];
  const entries = Array.isArray(node)
    ? node.map((value, index) => [String(index), value])
    : Object.entries(node);
  return entries.map(([key, value]) => ({
    id: [...path, key].join(".") || key,
    key,
    path: [...path, key],
    pathLabel: [...path, key].join("."),
    type: describeNodeType(value),
    summary: summarizeNode(value),
    expandable: Boolean(value) && typeof value === "object",
    value,
  }));
}

function FieldExplorer({ payload }) {
  const [path, setPath] = useState([]);

  useEffect(() => {
    setPath([]);
  }, [payload]);

  if (!payload || typeof payload !== "object") return <div className="empty-card">未选择可展开的 payload。</div>;

  const currentNode = readNodeAtPath(payload, path);
  const rows = buildFieldRows(currentNode, path);
  const isLeaf = currentNode === null || typeof currentNode !== "object";
  const displayPath = path.length ? path.join(".") : "root";

  return (
    <div className="detail-section">
      <div className="field-shell">
        <div className="field-toolbar">
          <div className="field-path">
            <span className="note-title">field path</span>
            <div className="breadcrumb-track">
              <button className="breadcrumb-button" onClick={() => setPath([])} disabled={!path.length}>
                root
              </button>
              {path.map((segment, index) => (
                <React.Fragment key={`${segment}-${index}`}>
                  <span className="breadcrumb-sep">/</span>
                  <button
                    className="breadcrumb-button"
                    onClick={() => setPath(path.slice(0, index + 1))}
                    disabled={index === path.length - 1}
                  >
                    {segment}
                  </button>
                </React.Fragment>
              ))}
            </div>
          </div>
          <div className="field-toolbar-actions">
            <button
              className="ghost-button"
              disabled={!path.length}
              onClick={() => setPath((current) => current.slice(0, -1))}
              data-testid="field-back-button"
            >
              字段回退
            </button>
            <button className="ghost-button" disabled={!path.length} onClick={() => setPath([])}>
              回到 root
            </button>
          </div>
        </div>

        {isLeaf ? (
          <div className="note-block">
            <div className="note-title">leaf value</div>
            <div className="field-leaf-value">{safeDisplayValue(currentNode)}</div>
          </div>
        ) : (
          <div className="table-shell">
            <table className="data-table">
              <thead>
                <tr>
                  <th>字段</th>
                  <th>类型</th>
                  <th>摘要</th>
                  <th>动作</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.id}>
                    <td>{row.key}</td>
                    <td>{row.type}</td>
                    <td>{row.summary}</td>
                    <td>
                      {row.expandable ? (
                        <button
                          className="ghost-button field-action-button"
                          onClick={() => setPath(row.path)}
                          data-testid={`field-drill-${row.pathLabel.replaceAll(".", "-")}`}
                        >
                          进入
                        </button>
                      ) : (
                        <span className="metric-hint">leaf</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        <div className="note-block">
          <div className="note-title">current node</div>
          <div>{displayPath}</div>
          <div className="note-sub">
            {isLeaf ? "当前节点已到叶子字段，可使用“字段回退”返回上层。" : `当前层共 ${rows.length} 个字段。`}
          </div>
        </div>
      </div>

      <pre className="json-viewer">{JSON.stringify(currentNode ?? {}, null, 2)}</pre>
    </div>
  );
}

function buildArtifactRows(snapshot) {
  const payloads = snapshot?.artifact_payloads || {};
  const catalog = snapshot?.catalog || [];
  const rows = [];
  const seenPayloadKeys = new Set();

  catalog.forEach((item) => {
    const payloadEntry = item.payload_key ? payloads[item.payload_key] : null;
    if (payloadEntry) seenPayloadKeys.add(item.payload_key);
    rows.push({
      id: item.id || item.payload_key || item.path,
      payloadKey: item.payload_key || null,
      label: payloadEntry?.label || item.label || item.payload_key || item.id,
      category: item.category || payloadEntry?.category || "uncategorized",
      status: item.status || payloadEntry?.summary?.status || "",
      changeClass: item.change_class || payloadEntry?.summary?.change_class || "",
      path: item.path || payloadEntry?.path || "",
      researchDecision: item.research_decision || payloadEntry?.summary?.research_decision || "",
      takeaway: item.takeaway || item.recommended_brief || payloadEntry?.summary?.recommended_brief || "",
      generatedAt: item.generated_at_utc || payloadEntry?.summary?.generated_at_utc || "",
      topLevelKeys: item.top_level_keys || payloadEntry?.summary?.top_level_keys || [],
      hasPayload: Boolean(payloadEntry),
      payload: payloadEntry?.payload || null,
      summaryPayload: payloadEntry?.summary || item,
    });
  });

  Object.entries(payloads).forEach(([key, entry]) => {
    if (seenPayloadKeys.has(key)) return;
    rows.push({
      id: key,
      payloadKey: key,
      label: entry.label,
      category: entry.category || "uncategorized",
      status: entry.summary?.status || "",
      changeClass: entry.summary?.change_class || "",
      path: entry.path || "",
      researchDecision: entry.summary?.research_decision || "",
      takeaway: entry.summary?.recommended_brief || "",
      generatedAt: entry.summary?.generated_at_utc || "",
      topLevelKeys: entry.summary?.top_level_keys || [],
      hasPayload: true,
      payload: entry.payload,
      summaryPayload: entry.summary,
    });
  });

  return rows;
}

function buildDomainRows(artifactRows) {
  const grouped = new Map();
  artifactRows.forEach((row) => {
    const key = row.category || "uncategorized";
    if (!grouped.has(key)) {
      grouped.set(key, {
        id: key,
        label: key,
        count: 0,
        payloadCount: 0,
        positive: 0,
        warning: 0,
        negative: 0,
        headline: "",
      });
    }
    const item = grouped.get(key);
    item.count += 1;
    item.payloadCount += row.hasPayload ? 1 : 0;
    item[statusTone(row.status || row.researchDecision)] += 1;
    if (!item.headline && (row.researchDecision || row.takeaway)) {
      item.headline = row.researchDecision || row.takeaway;
    }
  });
  return [
    {
      id: "all",
      label: "all",
      count: artifactRows.length,
      payloadCount: artifactRows.filter((row) => row.hasPayload).length,
      positive: 0,
      warning: 0,
      negative: 0,
      headline: "查看全部非敏感域与工件。",
    },
    ...Array.from(grouped.values()).sort((a, b) => b.count - a.count || a.label.localeCompare(b.label)),
  ];
}

function pickDefaultArtifact(rows, domain) {
  const candidates = rows.filter((row) => domain === "all" || row.category === domain);
  return candidates[0]?.id || "";
}

function Breadcrumbs({ items, onBack, canGoBack }) {
  return (
    <div className="breadcrumb-bar">
      <button className="ghost-button" disabled={!canGoBack} onClick={onBack}>
        ← 回退
      </button>
      <div className="breadcrumb-track">
        {items.map((item, index) => (
          <React.Fragment key={`${item.label}-${index}`}>
            {index > 0 ? <span className="breadcrumb-sep">/</span> : null}
            <button className="breadcrumb-button" onClick={item.onClick} disabled={item.disabled}>
              {item.label}
            </button>
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

function DomainRail({ domains, selectedDomain, onSelect }) {
  return (
    <div className="stack-list" data-testid="domain-rail">
      {domains.map((domain) => (
        <button
          key={domain.id}
          className={`stack-item ${selectedDomain === domain.id ? "stack-item-active" : ""}`}
          onClick={() => onSelect(domain.id)}
          data-testid={`domain-item-${domain.id}`}
        >
          <div className="stack-item-top">
            <strong>{domain.label}</strong>
            <span className={badgeClass(domain.id === "all" ? "neutral" : domain.headline)}>{domain.count}</span>
          </div>
          <div className="stack-item-sub">
            payload {domain.payloadCount} / total {domain.count}
          </div>
          {domain.headline ? <div className="stack-item-note">{domain.headline}</div> : null}
        </button>
      ))}
    </div>
  );
}

function ArtifactList({ rows, selectedArtifactId, onSelect }) {
  if (!rows.length) return <div className="empty-card">当前筛选下没有工件。</div>;
  return (
    <div className="stack-list" data-testid="artifact-list">
      {rows.map((row) => (
        <button
          key={row.id}
          className={`stack-item ${selectedArtifactId === row.id ? "stack-item-active" : ""}`}
          onClick={() => onSelect(row.id)}
          data-testid={`artifact-item-${row.payloadKey || row.id}`}
        >
          <div className="stack-item-top">
            <strong>{row.label}</strong>
            <span className={badgeClass(row.status || row.changeClass || row.category)}>{row.status || row.category}</span>
          </div>
          <div className="stack-item-sub">{row.path}</div>
          <div className="stack-item-note">{row.researchDecision || row.takeaway || "无额外说明"}</div>
        </button>
      ))}
    </div>
  );
}

function ArtifactSummary({ artifact, payload, detailTab, onDetailTabChange }) {
  const summary = artifact?.summaryPayload || {};
  const topFields = cardList([
    { label: "label", value: artifact?.label },
    { label: "category", value: artifact?.category },
    { label: "status", value: artifact?.status },
    { label: "change_class", value: artifact?.changeClass },
    { label: "research_decision", value: artifact?.researchDecision },
    { label: "generated_at_utc", value: artifact?.generatedAt },
    { label: "payload", value: artifact?.hasPayload ? "loaded" : "summary-only" },
    { label: "path", value: artifact?.path },
  ]);

  return (
    <div className="detail-shell">
      <div className="panel-header">
        <h2>{artifact?.label || "未选择工件"}</h2>
        <span className={badgeClass(artifact?.status || artifact?.researchDecision || artifact?.category)}>{artifact?.status || artifact?.category || "—"}</span>
      </div>
      <KeyValueGrid items={topFields} compact />
      <div className="detail-tabs">
        {DETAIL_TABS.map((tab) => (
          <button
            key={tab.id}
            className={`detail-tab ${detailTab === tab.id ? "detail-tab-active" : ""}`}
            onClick={() => onDetailTabChange(tab.id)}
            data-testid={`detail-tab-${tab.id}`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {detailTab === "summary" ? (
        <div className="detail-section">
          <div className="note-block">
            <div className="note-title">takeaway</div>
            <div>{artifact?.takeaway || artifact?.researchDecision || "—"}</div>
          </div>
          <div className="note-block">
            <div className="note-title">top_level_keys</div>
            <div>{(artifact?.topLevelKeys || []).join(", ") || "—"}</div>
          </div>
          <SimpleTable
            columns={[
              { key: "field", label: "field" },
              { key: "value", label: "value" },
            ]}
            rows={Object.entries(summary)
              .slice(0, 12)
              .map(([field, value]) => ({ field, value: safeDisplayValue(value) }))}
            empty="没有可展示的 summary 字段。"
          />
        </div>
      ) : null}

      {detailTab === "fields" ? (
        artifact?.hasPayload ? <FieldExplorer payload={payload} /> : <div className="empty-card">当前工件只有 catalog/summary，没有加载 payload。</div>
      ) : null}

      {detailTab === "raw" ? (
        artifact?.hasPayload ? <pre className="json-viewer">{JSON.stringify(payload || {}, null, 2)}</pre> : <pre className="json-viewer">{JSON.stringify(summary || {}, null, 2)}</pre>
      ) : null}
    </div>
  );
}

function OverviewPanel({ snapshot, comparisonReport, holdHandoff, operatorPanel, domains, onJumpExplorer, onJumpBacktests }) {
  const overviewCards = cardList([
    { label: "前端端口", value: snapshot?.ui_routes?.frontend_dev, hint: "preview / dev entry" },
    { label: "快照生成", value: snapshot?.meta?.generated_at_utc },
    { label: "非敏感工件数", value: formatNumber(snapshot?.meta?.artifact_payload_count, 0) },
    { label: "回测工件数", value: formatNumber(snapshot?.meta?.backtest_artifact_count, 0) },
    { label: "当前 baseline", value: holdHandoff?.active_baseline || "—" },
    { label: "局部候选", value: holdHandoff?.local_candidate || "—" },
  ]);

  return (
    <section className="section-grid">
      <div className="panel-card span-12">
        <div className="panel-header">
          <h2>三层阅读框架</h2>
          <span className={badgeClass("read_only")}>framework</span>
        </div>
        <div className="framework-strip">
          <button className="framework-card" onClick={onJumpBacktests}>
            <span className="metric-label">Layer 1</span>
            <strong>全局概览 / 回测工作台</strong>
            <span className="metric-hint">先看表现、样本、主线结论。</span>
          </button>
          <button className="framework-card" onClick={() => onJumpExplorer("research")}>
            <span className="metric-label">Layer 2</span>
            <strong>域级穿透</strong>
            <span className="metric-hint">research / backtest / operator-panel / sim-only。</span>
          </button>
          <button className="framework-card" onClick={() => onJumpExplorer("all")}>
            <span className="metric-label">Layer 3</span>
            <strong>工件 → 字段 → Raw</strong>
            <span className="metric-hint">支持面包屑、回退、字段展开和 raw drill-down。</span>
          </button>
        </div>
      </div>

      <div className="panel-card span-7">
        <div className="panel-header">
          <h2>核心概览</h2>
          <span className={badgeClass(comparisonReport?.status)}>{comparisonReport?.status || "—"}</span>
        </div>
        <KeyValueGrid items={overviewCards} />
        <div className="chart-shell">
          <ReactECharts option={buildComparisonOption(comparisonReport?.rows || [])} style={{ height: 340 }} />
        </div>
      </div>

      <div className="panel-card span-5">
        <div className="panel-header">
          <h2>主线摘要</h2>
          <span className={badgeClass(holdHandoff?.research_decision)}>{holdHandoff?.source_head_status || "canonical"}</span>
        </div>
        <KeyValueGrid
          items={cardList([
            { label: "active_baseline", value: holdHandoff?.active_baseline },
            { label: "transfer_watch", value: holdHandoff?.transfer_watch || [] },
            { label: "demoted", value: holdHandoff?.demoted_candidate || [] },
            { label: "operator_head", value: operatorPanel?.summary?.operator_head_brief },
          ])}
          compact
        />
        <div className="note-block">
          <div className="note-title">recommended_brief</div>
          <div>{holdHandoff?.recommended_brief || holdHandoff?.research_note || "—"}</div>
        </div>
      </div>

      <div className="panel-card span-12">
        <div className="panel-header">
          <h2>域级地图</h2>
          <span className={badgeClass("ok")}>{domains.length - 1} domains</span>
        </div>
        <div className="domain-summary-grid">
          {domains.filter((domain) => domain.id !== "all").map((domain) => (
            <button key={domain.id} className="domain-summary-card" onClick={() => onJumpExplorer(domain.id)}>
              <div className="stack-item-top">
                <strong>{domain.label}</strong>
                <span className={badgeClass(domain.headline)}>{domain.count}</span>
              </div>
              <div className="stack-item-sub">payload {domain.payloadCount}</div>
              <div className="stack-item-note">{domain.headline || "—"}</div>
            </button>
          ))}
        </div>
      </div>
    </section>
  );
}

function BacktestWorkbench({ backtests, selectedBacktest, onSelectBacktest, comparisonReport }) {
  return (
    <section className="section-grid">
      <div className="panel-card span-4">
        <div className="panel-header">
          <h2>回测工作台</h2>
          <span className={badgeClass("ok")}>{backtests.length} artifacts</span>
        </div>
        <div className="control-stack">
          <select value={selectedBacktest?.id || ""} onChange={(event) => onSelectBacktest(event.target.value)}>
            {backtests.map((item) => (
              <option key={item.id} value={item.id}>
                {item.label}
              </option>
            ))}
          </select>
        </div>
        <KeyValueGrid
          items={cardList([
            { label: "start", value: selectedBacktest?.start },
            { label: "end", value: selectedBacktest?.end },
            { label: "trades", value: formatNumber(selectedBacktest?.trades, 0) },
            { label: "total_return", value: formatPercent(selectedBacktest?.total_return) },
            { label: "annual_return", value: formatPercent(selectedBacktest?.annual_return) },
            { label: "max_drawdown", value: formatPercent(selectedBacktest?.max_drawdown) },
            { label: "profit_factor", value: formatNumber(selectedBacktest?.profit_factor) },
            { label: "positive_window_ratio", value: formatPercent(selectedBacktest?.positive_window_ratio) },
          ])}
          compact
        />
      </div>
      <div className="panel-card span-8">
        <div className="panel-header">
          <h2>权益 / 回撤</h2>
          <span className={badgeClass("sampled")}>sampled</span>
        </div>
        <div className="chart-shell">
          <ReactECharts option={buildEquityOption(selectedBacktest)} style={{ height: 430 }} />
        </div>
      </div>
      <div className="panel-card span-12">
        <div className="panel-header">
          <h2>最近策略回测对比</h2>
          <span className={badgeClass(comparisonReport?.comparison_takeaway)}>{comparisonReport?.row_count || 0} rows</span>
        </div>
        <SimpleTable
          columns={[
            { key: "strategy_id_canonical", label: "strategy" },
            { key: "market_scope", label: "scope" },
            { key: "trade_count", label: "trades", render: (value) => formatNumber(value, 0) },
            { key: "win_rate", label: "win_rate", render: (value) => formatPercent(value) },
            { key: "total_return", label: "total_return", render: (value) => formatPercent(value) },
            { key: "profit_factor", label: "pf", render: (value) => formatNumber(value) },
            { key: "expectancy_r", label: "expectancy_r", render: (value) => formatNumber(value) },
            { key: "recommendation", label: "recommendation" },
          ]}
          rows={comparisonReport?.rows || []}
        />
      </div>
    </section>
  );
}

function ExplorerPanel({
  domains,
  selectedDomain,
  selectedArtifact,
  filteredArtifacts,
  search,
  onSearch,
  onSelectDomain,
  onSelectArtifact,
  breadcrumbs,
  onBack,
  canGoBack,
  detailTab,
  onDetailTabChange,
}) {
  return (
    <section className="section-grid">
      <div className="panel-card span-12">
        <Breadcrumbs items={breadcrumbs} onBack={onBack} canGoBack={canGoBack} />
      </div>

      <div className="panel-card span-3">
        <div className="panel-header">
          <h2>域导航</h2>
          <span className={badgeClass(selectedDomain)}>{selectedDomain}</span>
        </div>
        <DomainRail domains={domains} selectedDomain={selectedDomain} onSelect={onSelectDomain} />
      </div>

      <div className="panel-card span-4">
        <div className="panel-header">
          <h2>工件列表</h2>
          <span className={badgeClass("read_only")}>{filteredArtifacts.length}</span>
        </div>
        <div className="control-stack">
          <input
            type="search"
            value={search}
            onChange={(event) => onSearch(event.target.value)}
            placeholder="按 label / status / decision / path 搜索"
            data-testid="artifact-search"
          />
        </div>
        <ArtifactList rows={filteredArtifacts} selectedArtifactId={selectedArtifact?.id} onSelect={onSelectArtifact} />
      </div>

      <div className="panel-card span-5">
        {selectedArtifact ? (
          <ArtifactSummary
            artifact={selectedArtifact}
            payload={selectedArtifact.payload}
            detailTab={detailTab}
            onDetailTabChange={onDetailTabChange}
          />
        ) : (
          <div className="empty-state">
            <h2>先选一个工件</h2>
            <p>当前支持从域 → 工件 → 字段 → Raw 四层穿透，并且可回退。</p>
          </div>
        )}
      </div>
    </section>
  );
}

class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, message: "" };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, message: error?.message || "unknown render error" };
  }

  componentDidCatch(error) {
    console.error("Fenlie dashboard render error:", error);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="app-shell">
          <div className="error-boundary">
            <p className="eyebrow">Fenlie / UI Recovery</p>
            <h1>前端渲染失败，已进入回退框架</h1>
            <p>错误信息：{this.state.message}</p>
            <div className="hero-actions">
              <button className="primary-button" onClick={() => window.location.reload()}>
                重新加载
              </button>
              <a className="ghost-link" href={SNAPSHOT_PATH} target="_blank" rel="noreferrer">
                直接打开 JSON 快照
              </a>
            </div>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

function DashboardApp() {
  const [snapshot, setSnapshot] = useState(null);
  const [error, setError] = useState("");
  const [activeTab, setActiveTab] = useState("overview");
  const [selectedBacktestId, setSelectedBacktestId] = useState("");
  const [selectedDomain, setSelectedDomain] = useState("all");
  const [selectedArtifactId, setSelectedArtifactId] = useState("");
  const [artifactSearch, setArtifactSearch] = useState("");
  const [detailTab, setDetailTab] = useState("summary");

  const loadSnapshot = async () => {
    try {
      setError("");
      const response = await fetch(`${SNAPSHOT_PATH}?ts=${Date.now()}`, { cache: "no-store" });
      if (!response.ok) throw new Error(`snapshot fetch failed: ${response.status}`);
      const payload = await response.json();
      setSnapshot(payload);
    } catch (err) {
      setError(err?.message || "snapshot load failed");
    }
  };

  useEffect(() => {
    loadSnapshot();
  }, []);

  const comparisonReport = snapshot?.artifact_payloads?.recent_strategy_backtests?.payload || {};
  const holdHandoff = snapshot?.artifact_payloads?.hold_selection_handoff?.payload || {};
  const operatorPanel = snapshot?.artifact_payloads?.operator_panel?.payload || {};

  const backtests = snapshot?.backtest_artifacts || [];
  useEffect(() => {
    if (!selectedBacktestId && backtests.length) setSelectedBacktestId(backtests[0].id);
  }, [backtests, selectedBacktestId]);
  const selectedBacktest = backtests.find((item) => item.id === selectedBacktestId) || backtests[0];

  const artifactRows = useMemo(() => buildArtifactRows(snapshot), [snapshot]);
  const domains = useMemo(() => buildDomainRows(artifactRows), [artifactRows]);

  const filteredArtifacts = useMemo(() => {
    const keyword = artifactSearch.trim().toLowerCase();
    return artifactRows.filter((row) => {
      if (selectedDomain !== "all" && row.category !== selectedDomain) return false;
      if (!keyword) return true;
      return JSON.stringify(row).toLowerCase().includes(keyword);
    });
  }, [artifactRows, selectedDomain, artifactSearch]);

  useEffect(() => {
    if (!artifactRows.length) return;
    const hasCurrent = filteredArtifacts.some((row) => row.id === selectedArtifactId);
    if (!hasCurrent) {
      setSelectedArtifactId(filteredArtifacts[0]?.id || "");
      setDetailTab("summary");
    }
  }, [artifactRows, filteredArtifacts, selectedArtifactId]);

  const selectedArtifact = filteredArtifacts.find((row) => row.id === selectedArtifactId)
    || artifactRows.find((row) => row.id === selectedArtifactId)
    || null;

  const breadcrumbs = [
    { label: "概览", onClick: () => setActiveTab("overview") },
    ...(activeTab === "explorer" ? [{ label: selectedDomain, onClick: () => { setSelectedArtifactId(""); setDetailTab("summary"); } }] : []),
    ...(activeTab === "explorer" && selectedArtifact ? [{ label: selectedArtifact.label, onClick: () => setDetailTab("summary") }] : []),
    ...(activeTab === "explorer" && selectedArtifact && detailTab !== "summary" ? [{ label: detailTab, onClick: () => {} , disabled: true }] : []),
  ];

  const canGoBack = activeTab === "explorer" && (detailTab !== "summary" || Boolean(selectedArtifact) || selectedDomain !== "all");
  const handleBack = () => {
    if (detailTab !== "summary") {
      setDetailTab("summary");
      return;
    }
    if (selectedArtifactId) {
      setSelectedArtifactId("");
      return;
    }
    if (selectedDomain !== "all") {
      setSelectedDomain("all");
    }
  };

  const jumpToExplorer = (domain = "all") => {
    setActiveTab("explorer");
    setSelectedDomain(domain);
    setSelectedArtifactId(pickDefaultArtifact(artifactRows, domain));
    setDetailTab("summary");
  };

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Fenlie / Visual Backtest / Read-only</p>
          <h1>前端可视化回测与多层次穿透面板</h1>
          <p className="hero-copy">
            当前页面采用四层结构：总览 → 域 → 工件 → 字段 / Raw。
            所有交互只读消费 non-sensitive snapshot，不触碰 live execution path；并内置 ErrorBoundary 与回退按钮，避免整页黑屏。
          </p>
        </div>
        <div className="hero-actions">
          <button className="primary-button" onClick={loadSnapshot}>刷新快照</button>
          <button className="ghost-link" onClick={() => jumpToExplorer("all")}>进入穿透浏览</button>
          <a className="ghost-link" href={snapshot?.ui_routes?.operator_panel || "/operator_task_visual_panel.html"} target="_blank" rel="noreferrer">
            打开 Operator Panel
          </a>
          <a className="ghost-link" href={SNAPSHOT_PATH} target="_blank" rel="noreferrer">
            打开 JSON 快照
          </a>
        </div>
      </header>

      {error ? <div className="error-banner">加载失败：{error}</div> : null}

      <nav className="tab-bar" data-testid="main-tabs">
        {MAIN_TABS.map((tab) => (
          <button
            key={tab.id}
            className={`tab-button ${activeTab === tab.id ? "tab-button-active" : ""}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      {activeTab === "overview" ? (
        <OverviewPanel
          snapshot={snapshot}
          comparisonReport={comparisonReport}
          holdHandoff={holdHandoff}
          operatorPanel={operatorPanel}
          domains={domains}
          onJumpExplorer={jumpToExplorer}
          onJumpBacktests={() => setActiveTab("backtests")}
        />
      ) : null}

      {activeTab === "backtests" ? (
        <BacktestWorkbench
          backtests={backtests}
          selectedBacktest={selectedBacktest}
          onSelectBacktest={setSelectedBacktestId}
          comparisonReport={comparisonReport}
        />
      ) : null}

      {activeTab === "explorer" ? (
        <ExplorerPanel
          domains={domains}
          selectedDomain={selectedDomain}
          selectedArtifact={selectedArtifact}
          filteredArtifacts={filteredArtifacts}
          search={artifactSearch}
          onSearch={setArtifactSearch}
          onSelectDomain={(domain) => {
            setSelectedDomain(domain);
            setSelectedArtifactId(pickDefaultArtifact(artifactRows, domain));
            setDetailTab("summary");
          }}
          onSelectArtifact={(artifactId) => {
            setSelectedArtifactId(artifactId);
            setDetailTab("summary");
          }}
          breadcrumbs={breadcrumbs}
          onBack={handleBack}
          canGoBack={canGoBack}
          detailTab={detailTab}
          onDetailTabChange={setDetailTab}
        />
      ) : null}

      {activeTab === "raw" ? (
        <section className="section-grid">
          <div className="panel-card span-12">
            <div className="panel-header">
              <h2>快照级 Raw 输出</h2>
              <span className={badgeClass(snapshot?.status)}>{snapshot?.status || "—"}</span>
            </div>
            <pre className="json-viewer">{JSON.stringify(snapshot || {}, null, 2)}</pre>
          </div>
        </section>
      ) : null}
    </div>
  );
}

export default function App() {
  return (
    <ErrorBoundary>
      <DashboardApp />
    </ErrorBoundary>
  );
}
