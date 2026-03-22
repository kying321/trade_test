import type { Tone } from '../types/contracts';
import { labelFor } from './dictionary';

export function formatNumber(value: unknown, digits = 2): string {
  if (value === null || value === undefined || value === '') return '—';
  const num = Number(value);
  if (!Number.isFinite(num)) return String(value);
  return num.toLocaleString('zh-CN', {
    maximumFractionDigits: digits,
    minimumFractionDigits: Math.abs(num) < 10 && digits > 0 ? Math.min(2, digits) : 0,
  });
}

export function formatPercent(value: unknown, digits = 2): string {
  if (value === null || value === undefined || value === '') return '—';
  const num = Number(value);
  if (!Number.isFinite(num)) return String(value);
  return `${(num * 100).toFixed(digits)}%`;
}

export function safeDisplayValue(value: unknown): string {
  if (value === null || value === undefined || value === '') return '—';
  if (typeof value === 'boolean') return labelFor(String(value));
  if (typeof value === 'string' || typeof value === 'number') return String(value);
  if (Array.isArray(value)) {
    if (!value.length) return '[]';
    const primitiveOnly = value.every((item) => item === null || ['string', 'number', 'boolean'].includes(typeof item));
    return primitiveOnly ? value.map((item) => safeDisplayValue(item)).join(', ') : `[${value.length} ${labelFor('array_summary_suffix')}]`;
  }
  if (typeof value === 'object') return `{${Object.keys(value as object).length} ${labelFor('object_summary_suffix')}}`;
  return String(value);
}

export function statusTone(value: unknown): Tone {
  const text = String(value || '').toLowerCase();
  if (!text) return 'neutral';
  if (['ok', 'active', 'allowed', 'ready', 'healthy', 'passed', 'fresh', 'stable', 'positive', 'live', 'clear', 'keep'].some((key) => text.includes(key))) {
    return 'positive';
  }
  if (['blocked', 'failed', 'error', 'degraded', 'negative', 'stale', 'halt', 'deny', 'missing', 'breach'].some((key) => text.includes(key))) {
    return 'negative';
  }
  if (['warning', 'watch', 'candidate', 'pending', 'mixed', 'partial', 'defer', 'queued', 'stub', 'unknown', 'unwired'].some((key) => text.includes(key))) {
    return 'warning';
  }
  return 'neutral';
}

export function formatDateTime(value?: string): string {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString('zh-CN', {
    hour12: false,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}

export function compactPath(value: unknown, keepSegments = 4): string {
  if (value === null || value === undefined || value === '') return '—';
  const text = String(value);
  if (!text.includes('/')) return text;
  const segments = text.split('/').filter(Boolean);
  if (segments.length <= keepSegments) return text;
  return `…/${segments.slice(-keepSegments).join('/')}`;
}

export function ageMinutes(value?: string): number | null {
  if (!value) return null;
  const ts = Date.parse(value);
  if (Number.isNaN(ts)) return null;
  return Math.max(0, Math.round((Date.now() - ts) / 60000));
}

export function freshnessState(minutes: number | null, staleAfter = 240, warnAfter = 60): 'fresh' | 'warning' | 'stale' | 'unknown' {
  if (minutes === null) return 'unknown';
  if (minutes >= staleAfter) return 'stale';
  if (minutes >= warnAfter) return 'warning';
  return 'fresh';
}

export function formatAgeMinutes(value: number | null | undefined): string {
  if (value === null || value === undefined) return labelFor('unwired');
  if (value < 60) return `${value}${labelFor('unit_minute_short')}`;
  const hours = Math.floor(value / 60);
  const minutes = value % 60;
  return `${hours}${labelFor('unit_hour_short')} ${minutes}${labelFor('unit_minute_short')}`;
}

export function toArray<T>(value: unknown): T[] {
  return Array.isArray(value) ? (value as T[]) : [];
}

export function toRecord<T extends Record<string, unknown>>(value: unknown): T | null {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as T) : null;
}
