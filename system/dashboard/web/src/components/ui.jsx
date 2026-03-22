import React from "react";
import { badgeClass, safeDisplayValue } from "../lib/formatters";
import { labelFor } from "../utils/dictionary";

const t = (key) => labelFor(key);

export function Badge({ value, children }) {
  return <span className={badgeClass(value)}>{children || safeDisplayValue(value)}</span>;
}

export function PanelCard({ title, tone, meta, children, className = "" }) {
  return (
    <section className={`panel-card ${className}`.trim()}>
      {(title || meta) ? (
        <div className="panel-header">
          <div>
            {title ? <h2>{title}</h2> : null}
            {meta ? <p className="panel-meta">{meta}</p> : null}
          </div>
          {tone ? <Badge value={tone} /> : null}
        </div>
      ) : null}
      {children}
    </section>
  );
}

export function MetricGrid({ items = [], compact = false }) {
  return (
    <div className={`metric-grid ${compact ? "metric-grid-compact" : ""}`.trim()}>
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

export function EmptyCard({ title = t("legacy_empty_title"), body = t("legacy_empty_body"), action }) {
  return (
    <div className="empty-card">
      <h3>{title}</h3>
      <p>{body}</p>
      {action}
    </div>
  );
}

export function SimpleTable({ columns = [], rows = [], empty = t("legacy_empty_title") }) {
  if (!rows.length) return <EmptyCard title={empty} body="" />;
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
                <td key={column.key}>{column.render ? column.render(row[column.key], row) : safeDisplayValue(row[column.key])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function TabBar({ items, active, onChange, testId }) {
  return (
    <nav className="tab-bar" data-testid={testId}>
      {items.map((tab) => (
        <button
          key={tab.id}
          className={`tab-button ${active === tab.id ? "tab-button-active" : ""}`.trim()}
          onClick={() => onChange(tab.id)}
        >
          {tab.label}
        </button>
      ))}
    </nav>
  );
}

export function SegmentedControl({ items, active, onChange }) {
  return (
    <div className="segmented-control">
      {items.map((item) => (
        <button
          key={item.id}
          className={`segment ${active === item.id ? "segment-active" : ""}`.trim()}
          onClick={() => onChange(item.id)}
        >
          {item.label}
        </button>
      ))}
    </div>
  );
}
