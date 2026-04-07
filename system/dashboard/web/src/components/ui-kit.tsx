import { Children, isValidElement, useCallback, useEffect, useLayoutEffect, useRef, useState, type CSSProperties, type MouseEvent, type ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { compactPath, formatAgeMinutes, formatNumber, formatPercent, safeDisplayValue, statusTone } from '../utils/formatters';
import type { MetricItem, Tone, ViewColumnSchema, ViewFieldSchema, ViewValueKind } from '../types/contracts';
import { displayText, explainLabel, labelFor } from '../utils/dictionary';
import { buildDrilldownRowId } from '../utils/focus-links';

function formatMaybeMetric(value: unknown): string {
  if (typeof value === 'number') {
    if (Math.abs(value) <= 1) return formatPercent(value, 2);
    return formatNumber(value, 2);
  }
  if (typeof value === 'string') return displayText(value);
  return safeDisplayValue(value);
}

function flattenNodeText(children: ReactNode): string {
  if (children === null || children === undefined || typeof children === 'boolean') return '';
  if (typeof children === 'string' || typeof children === 'number') return String(children);
  if (Array.isArray(children)) return children.map((child) => flattenNodeText(child)).join('');
  if (isValidElement<{ children?: ReactNode }>(children)) return flattenNodeText(children.props.children);
  return Children.toArray(children).map((child) => flattenNodeText(child)).join('');
}

function resolveRawTitle(children: ReactNode, raw?: string) {
  if (raw) return raw;
  const flattened = flattenNodeText(children).trim();
  return flattened || undefined;
}

export function ClampText({
  children,
  maxLines = 2,
  expandable = true,
  critical = false,
  className = '',
  raw,
}: {
  children: ReactNode;
  maxLines?: number;
  expandable?: boolean;
  critical?: boolean;
  className?: string;
  raw?: string;
}) {
  const copyRef = useRef<HTMLSpanElement | null>(null);
  const [expanded, setExpanded] = useState(false);
  const [overflowing, setOverflowing] = useState(false);
  const title = resolveRawTitle(children, raw);

  const measureCollapsedOverflow = useCallback((node: HTMLSpanElement) => {
    const alreadyClamped = node.classList.contains('is-clamped');
    if (!alreadyClamped) node.classList.add('is-clamped');
    const nextOverflowing = node.scrollHeight > node.clientHeight + 1 || node.scrollWidth > node.clientWidth + 1;
    if (!alreadyClamped) node.classList.remove('is-clamped');
    return nextOverflowing;
  }, []);

  const measureOverflow = useCallback(() => {
    const node = copyRef.current;
    if (!node) {
      setOverflowing(false);
      return;
    }
    const nextOverflowing = measureCollapsedOverflow(node);
    setOverflowing(nextOverflowing);
    if (!nextOverflowing) setExpanded(false);
  }, [measureCollapsedOverflow]);

  useLayoutEffect(() => {
    measureOverflow();
  }, [children, maxLines, measureOverflow]);

  useEffect(() => {
    if (typeof window === 'undefined') return undefined;
    const handleResize = () => measureOverflow();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [measureOverflow]);

  useEffect(() => {
    const node = copyRef.current;
    if (!node || typeof ResizeObserver === 'undefined') return undefined;

    const observer = new ResizeObserver(() => {
      measureOverflow();
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, [measureOverflow]);

  const canExpand = critical || expandable;
  const showToggle = canExpand && (overflowing || expanded);

  return (
    <span className={`clamp-text ${className}`.trim()} data-expanded={expanded ? 'true' : 'false'}>
      <span
        ref={copyRef}
        className={`clamp-copy ${!expanded ? 'is-clamped' : ''}`.trim()}
        style={{ '--clamp-lines': String(maxLines) } as CSSProperties}
        title={title}
      >
        {children}
      </span>
      {showToggle ? (
        <button
          type="button"
          className="clamp-toggle"
          aria-label={expanded ? '收起完整内容' : '展开完整内容'}
          onClick={() => setExpanded((current) => !current)}
        >
          {expanded ? '收起' : '展开'}
        </button>
      ) : null}
    </span>
  );
}

function MetricValue({
  value,
  showRaw = false,
}: {
  value: unknown;
  showRaw?: boolean;
}) {
  if (typeof value === 'number') {
    return <>{formatMaybeMetric(value)}</>;
  }
  if (typeof value === 'string') {
    return showRaw ? <ValueText value={value} showRaw /> : <>{displayText(value)}</>;
  }
  const rendered = safeDisplayValue(value);
  return showRaw ? <ValueText value={rendered} showRaw /> : <>{rendered}</>;
}

function renderStructuredValue(value: unknown, kind: ViewValueKind | undefined, showRaw: boolean, critical = false) {
  switch (kind) {
    case 'badge':
      return <Badge value={value as string | undefined} />;
    case 'percent':
      return <>{formatPercent(value, 2)}</>;
    case 'number':
      return <>{formatNumber(value, 2)}</>;
    case 'sparkline':
      return <Sparkline points={value as Array<{ equity?: number }> | undefined} />;
    case 'path':
      return <PathText value={value} showRaw={showRaw} critical={critical} />;
    case 'text':
    default:
      return <ValueText value={value} showRaw={showRaw} critical={critical} />;
  }
}

export function toneClass(value: Tone | string | undefined): string {
  const tone = typeof value === 'string' && ['positive', 'warning', 'negative', 'neutral'].includes(value)
    ? (value as Tone)
    : statusTone(value);
  return `tone-${tone}`;
}

export function ValueText({
  value,
  className = '',
  showRaw = true,
  expandable = true,
  critical = false,
}: {
  value: unknown;
  className?: string;
  showRaw?: boolean;
  expandable?: boolean;
  critical?: boolean;
}) {
  const raw = safeDisplayValue(value);
  const { primary, secondary } = explainLabel(raw);
  return (
    <span className={`value-text ${className}`.trim()} title={raw}>
      <ClampText className="value-primary" raw={raw} expandable={expandable} critical={critical}>{primary}</ClampText>
      {showRaw && secondary ? <small className="value-raw">{secondary}</small> : null}
    </span>
  );
}

export function Badge({ value, children }: { value?: unknown; children?: ReactNode }) {
  const hasChildren = children !== null && children !== undefined;
  const raw = hasChildren ? (resolveRawTitle(children) ?? safeDisplayValue(value)) : safeDisplayValue(value);
  const display = hasChildren ? children : displayText(raw);
  return <span className={`badge ${toneClass(value as string | undefined)}`} title={raw}>{display}</span>;
}

export function PanelCard({
  title,
  kicker,
  meta,
  actions,
  children,
  className = '',
  maximizable = true,
}: {
  title: ReactNode;
  kicker?: string;
  meta?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  maximizable?: boolean;
}) {
  const titleText = resolveRawTitle(title) || '当前面板';
  const [maximized, setMaximized] = useState(false);

  useEffect(() => {
    if (typeof document === 'undefined') return undefined;
    if (!maximized) {
      document.body.classList.remove('panel-card-lock');
      return undefined;
    }
    document.body.classList.add('panel-card-lock');
    return () => document.body.classList.remove('panel-card-lock');
  }, [maximized]);

  return (
    <section
      className={`panel-card ${maximized ? 'panel-card-maximized' : ''} ${className}`.trim()}
      data-maximized={maximized ? 'true' : 'false'}
    >
      <header className="panel-card-head">
        <div className="panel-card-copy">
          {kicker ? <p className="panel-kicker">{kicker}</p> : null}
          <h2 className="panel-card-title"><ClampText>{title}</ClampText></h2>
          {meta ? <p className="panel-note panel-card-meta"><ClampText>{meta}</ClampText></p> : null}
        </div>
        {(actions || maximizable) ? (
          <div className="panel-actions panel-card-head-actions">
            {actions}
            {maximizable ? (
              <button
                type="button"
                className="panel-card-maximize-button"
                aria-label={`${maximized ? '还原' : '最大化'}面板：${titleText}`}
                title={`${maximized ? '还原' : '最大化'}面板：${titleText}`}
                onClick={() => setMaximized((current) => !current)}
              >
                <span aria-hidden="true" className="panel-card-maximize-glyph">{maximized ? '🗗' : '⛶'}</span>
                <span className="panel-card-maximize-label">{maximized ? '还原' : '最大化'}</span>
              </button>
            ) : null}
          </div>
        ) : null}
      </header>
      <div className="panel-card-body">{children}</div>
    </section>
  );
}

export function MetricStrip({
  items,
  showRawValues = false,
}: {
  items: MetricItem[];
  showRawValues?: boolean;
}) {
  return (
    <div className="metric-strip">
      {items.map((item) => (
        <article className={`metric-tile ${toneClass(item.tone)}`} key={item.id}>
          <span className="metric-label">{item.label}</span>
          <strong className="metric-value"><MetricValue value={item.value} showRaw={showRawValues} /></strong>
          {item.hint ? <ValueText value={item.hint} className="metric-hint" showRaw={false} /> : null}
        </article>
      ))}
    </div>
  );
}

export function FreshnessList({
  rows,
}: {
  rows: Array<{ id: string; label: string; timestamp?: string; ageMinutes?: number | null; state: string; artifact?: string }>;
}) {
  return (
    <div className="freshness-list">
      {rows.map((row) => (
        <article className={`freshness-row ${toneClass(row.state)}`} key={row.id}>
          <div>
            <strong>{row.label}</strong>
            <p>{row.timestamp || '—'}</p>
          </div>
          <div className="freshness-meta">
            <Badge value={row.state} />
            <span>{formatAgeMinutes(row.ageMinutes)}</span>
          </div>
        </article>
      ))}
    </div>
  );
}

export function DenseTable({
  columns,
  rows,
  empty = labelFor('empty_data'),
}: {
  columns: Array<ViewColumnSchema & { render?: (row: Record<string, unknown>, index: number) => ReactNode }>;
  rows: Array<Record<string, unknown>>;
  empty?: string;
}) {
  if (!rows.length) return <div className="empty-block">{empty}</div>;
  return (
    <div className="table-wrap">
      <table className="dense-table">
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column.key}>{column.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={String(row.id || row.label || row.path || index)}>
              {columns.map((column) => (
                <td key={column.key}>{column.render ? column.render(row, index) : renderStructuredValue(row[column.key], column.kind, column.showRaw ?? false)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function JsonBlock({ value }: { value: unknown }) {
  return <pre className="json-block">{JSON.stringify(value, null, 2)}</pre>;
}

export function PathText({
  value,
  showRaw = false,
  keepSegments = 4,
  expandable = true,
  critical = false,
}: {
  value: unknown;
  showRaw?: boolean;
  keepSegments?: number;
  expandable?: boolean;
  critical?: boolean;
}) {
  const raw = safeDisplayValue(value);
  const primary = compactPath(raw, keepSegments);
  const showSecondary = showRaw && raw !== primary && raw !== '—';
  return (
    <span className="value-text" title={raw}>
      <ClampText className="value-primary" raw={raw} expandable={expandable} critical={critical}>{primary}</ClampText>
      {showSecondary ? <small className="value-raw">{raw}</small> : null}
    </span>
  );
}

export function DrilldownSection({
  title,
  summary,
  meta,
  tone,
  defaultOpen = false,
  className = '',
  children,
}: {
  title: string;
  summary?: string;
  meta?: ReactNode;
  tone?: Tone | string;
  defaultOpen?: boolean;
  className?: string;
  children: ReactNode;
}) {
  return (
    <details className={`drill-section ${toneClass(tone)} ${className}`.trim()} open={defaultOpen}>
      <summary className="drill-summary">
        <div className="drill-summary-copy">
          <span className="drill-title"><ClampText raw={title} expandable={false}>{title}</ClampText></span>
          {summary ? <strong className="drill-brief"><ClampText raw={summary} expandable={false}>{summary}</ClampText></strong> : null}
        </div>
        {meta ? <div className="drill-summary-meta">{meta}</div> : null}
      </summary>
      <div className="drill-body">{children}</div>
    </details>
  );
}

export function AccordionCard({
  id,
  title,
  summary,
  meta,
  tone,
  open,
  onToggle,
  className = '',
  children,
}: {
  id: string;
  title: ReactNode;
  summary?: ReactNode;
  meta?: ReactNode;
  tone?: Tone | string;
  open: boolean;
  onToggle: () => void;
  className?: string;
  children: ReactNode;
}) {
  return (
    <section
      className={`drill-card accordion-card ${toneClass(tone)} ${open ? 'open' : ''} ${className}`.trim()}
      data-accordion-id={id}
      data-state={open ? 'open' : 'closed'}
    >
      <button
        type="button"
        className="drill-card-summary accordion-trigger"
        aria-expanded={open}
        aria-controls={`${id}-panel`}
        onClick={onToggle}
      >
        <div className="drill-card-copy">
          <strong><ClampText expandable={false}>{title}</ClampText></strong>
          {summary ? <span><ClampText expandable={false}>{summary}</ClampText></span> : null}
        </div>
        <div className="drill-card-meta">
          {meta}
        </div>
      </button>
      <div className="drill-body accordion-body" id={`${id}-panel`} hidden={!open}>
        {children}
      </div>
    </section>
  );
}

export function KeyValueGrid({
  rows,
}: {
  rows: Array<ViewFieldSchema & { value: unknown; critical?: boolean }>;
}) {
  return (
    <div className="kv-grid">
      {rows.map((row) => (
        <div className="kv-row" key={row.key}>
          <span className="kv-key">{row.label}</span>
          <strong className="kv-value">
            {renderStructuredValue(row.value, row.kind, row.showRaw ?? true, row.critical ?? false)}
          </strong>
        </div>
      ))}
    </div>
  );
}

export function DrilldownList({
  rows,
  empty = labelFor('empty_drilldown_items'),
  titleKey,
  summaryKeys = [],
  statusKey,
  rowIdentityKeys,
  focusedRowId,
  rowHrefBuilder,
  fields,
  defaultOpenCount = 1,
}: {
  rows: Array<Record<string, unknown>>;
  empty?: string;
  titleKey: string;
  summaryKeys?: string[];
  statusKey?: string;
  rowIdentityKeys?: string[];
  focusedRowId?: string;
  rowHrefBuilder?: (row: Record<string, unknown>, rowId: string) => string;
  fields: ViewFieldSchema[];
  defaultOpenCount?: number;
}) {
  if (!rows.length) return <div className="empty-block">{empty}</div>;
  return (
    <div className="drill-list">
      {rows.map((row, index) => {
        const summary = summaryKeys
          .map((key) => displayText(safeDisplayValue(row[key])))
          .filter((value) => value !== '—')
          .join(' ｜ ');
        const status = statusKey ? safeDisplayValue(row[statusKey]) : undefined;
        const titleText = explainLabel(safeDisplayValue(row[titleKey]));
        const rowId = buildDrilldownRowId(row, rowIdentityKeys);
        const focused = Boolean(focusedRowId && rowId === focusedRowId);
        const rowHref = rowHrefBuilder && rowId ? rowHrefBuilder(row, rowId) : undefined;
        const stopToggle = (event: MouseEvent<HTMLAnchorElement>) => {
          event.stopPropagation();
        };
        return (
          <details className={`drill-card ${toneClass(status)} ${focused ? 'drill-card-focused' : ''}`.trim()} key={String(row.id || row[titleKey] || index)} open={focused || index < defaultOpenCount}>
            <summary className="drill-card-summary">
              <div className="drill-card-copy">
                <strong><ClampText raw={safeDisplayValue(row[titleKey])} expandable={false}>{titleText.primary}</ClampText></strong>
                {titleText.secondary ? <small className="drill-card-raw">{titleText.secondary}</small> : null}
                {summary ? <span><ClampText raw={summary} expandable={false}>{summary}</ClampText></span> : null}
              </div>
              <div className="drill-card-meta">
                {rowHref ? (
                  <Link
                    className={`chip-button drill-card-link ${focused ? 'active' : ''}`.trim()}
                    to={rowHref}
                    onClick={stopToggle}
                    onMouseDown={stopToggle}
                  >
                    {focused ? labelFor('terminal_drill_focus_current') : labelFor('terminal_drill_focus_link')}
                  </Link>
                ) : null}
                {status ? <Badge value={status} /> : null}
              </div>
            </summary>
            <KeyValueGrid
              rows={fields.map((field) => ({
                key: field.key,
                label: field.label,
                value: row[field.key],
                showRaw: field.showRaw,
              }))}
            />
          </details>
        );
      })}
    </div>
  );
}

export function Sparkline({ points }: { points?: Array<{ equity?: number }> }) {
  const series = (points || []).map((item) => Number(item.equity || 0)).filter((value) => Number.isFinite(value));
  if (series.length < 2) return <span className="sparkline-empty">—</span>;
  const min = Math.min(...series);
  const max = Math.max(...series);
  const range = max - min || 1;
  const path = series
    .map((value, index) => {
      const x = (index / (series.length - 1)) * 100;
      const y = 100 - ((value - min) / range) * 100;
      return `${index === 0 ? 'M' : 'L'} ${x} ${y}`;
    })
    .join(' ');
  return (
    <svg className="sparkline" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
      <path d={path} vectorEffect="non-scaling-stroke" />
    </svg>
  );
}
