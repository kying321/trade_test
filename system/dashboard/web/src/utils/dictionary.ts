import type { SurfaceKey } from '../types/contracts';
import sharedTermContract from '../contracts/dashboard-term-contract.json';

export type LabelExplanation = {
  primary: string;
  secondary?: string;
  translated: boolean;
};

const LABELS: Record<string, string> = sharedTermContract.labels as Record<string, string>;


const TOKEN_LABELS: Record<string, string> = sharedTermContract.tokenLabels as Record<string, string>;


const SENTENCE_REPLACEMENTS: Array<[RegExp, string]> = (sharedTermContract.sentenceReplacements || []).map(({ pattern, flags, replacement }) => [new RegExp(pattern, flags), replacement]);


function exactLabel(value: string): string | undefined {
  const raw = value.trim();
  if (!raw) return undefined;
  return LABELS[raw] || LABELS[raw.toLowerCase()];
}

function translateToken(token: string): string {
  const raw = token.trim();
  if (!raw) return raw;
  const exact = exactLabel(raw);
  if (exact) return exact;
  const lower = raw.toLowerCase();
  if (TOKEN_LABELS[lower]) return TOKEN_LABELS[lower];
  if (/^[A-Z0-9]+$/.test(raw)) return raw;
  return raw;
}

function isUrlLike(raw: string): boolean {
  return /^https?:\/\//.test(raw);
}

function isPathLike(raw: string): boolean {
  return (raw.includes('/') || raw.includes('\\')) && (raw.includes('.') || raw.startsWith('/'));
}

function isTimestampLike(raw: string): boolean {
  return /^\d{4}-\d{2}-\d{2}(?:[T\s].*)?$/.test(raw)
    || /^\d{4}\/\d{2}\/\d{2}/.test(raw)
    || /T\d{2}:\d{2}:\d{2}/.test(raw)
    || /^\d{2}:\d{2}(?::\d{2})?$/.test(raw);
}

function isNaturalSentence(raw: string): boolean {
  return raw.includes(' ') && raw.split(/\s+/).length >= 4 && /[A-Za-z]{3,}/.test(raw) && !/^[A-Z0-9_:\-=|.+/]+$/.test(raw);
}

function looksCodeLike(raw: string): boolean {
  return /[_-]/.test(raw) || (raw.includes(':') && !raw.includes('://')) || raw.includes('=');
}

function humanizeSimpleCode(raw: string): string | null {
  if (!/[_-]/.test(raw)) return null;
  const exact = exactLabel(raw);
  if (exact) return exact;
  const tokens = raw.split(/[_-]+/).filter(Boolean);
  if (!tokens.length) return null;
  const translated = tokens.map(translateToken);
  const translatedCount = translated.filter((token, index) => token !== tokens[index]).length;
  const untranslatedAlpha = tokens.filter((token, index) => {
    const current = translated[index];
    return current === token && /^[a-z]+$/i.test(token);
  }).length;

  if (!translatedCount) return null;
  if (tokens.length > 8 && translatedCount < tokens.length - 1) return null;
  if (raw.length > 56 && untranslatedAlpha > 0) return null;
  if (untranslatedAlpha > 1 && tokens.length > 3) return null;
  return translated.join('');
}

function renderFragment(raw: string): string {
  const exact = exactLabel(raw);
  if (exact) return exact;
  const token = translateToken(raw);
  if (token !== raw) return token;
  const simple = humanizeSimpleCode(raw);
  if (simple) return simple;
  return raw;
}

function humanizeJoinedPhrase(raw: string, delimiter: string, displayDelimiter = ` ${delimiter} `): string | null {
  if (!raw.includes(delimiter)) return null;
  const parts = raw.split(delimiter).map((part) => part.trim()).filter(Boolean);
  if (parts.length < 2 || parts.length > 6) return null;
  const rendered = parts.map((part) => renderFragment(part));
  const changed = rendered.some((part, index) => part !== parts[index]);
  return changed ? rendered.join(displayDelimiter) : null;
}

function humanizeColonPhrase(raw: string): string | null {
  if (!raw.includes(':') || raw.includes('://') || isTimestampLike(raw)) return null;
  const parts = raw.split(':').filter(Boolean);
  if (parts.length < 2 || parts.length > 6 || parts.some((part) => part.length > 120)) return null;
  const rendered = parts.map((part) => renderFragment(part));
  const changed = rendered.some((part, index) => part !== parts[index]);
  return changed ? rendered.join(' / ') : null;
}

function splitOnce(raw: string, marker: string): [string, string] | null {
  const index = raw.indexOf(marker);
  if (index < 0) return null;
  return [raw.slice(0, index), raw.slice(index + marker.length)];
}

function humanizeAssignmentSegment(segment: string): string {
  const trimmed = segment.trim();
  if (!trimmed) return trimmed;
  const assignment = splitOnce(trimmed, '=');
  if (!assignment) {
    return humanizeColonPhrase(trimmed) || renderFragment(trimmed);
  }

  const [left, right] = assignment;
  const leftParts = left.split(':').filter(Boolean);
  const field = leftParts.pop() || left;
  const prefix = leftParts.map((part) => renderFragment(part)).filter(Boolean).join(' / ');
  const fieldLabel = renderFragment(field);
  const valueLabel = displayText(right.trim());
  const suffix = `${fieldLabel}：${valueLabel}`;
  return prefix ? `${prefix} / ${suffix}` : suffix;
}

function humanizeStructuredSummary(raw: string): string | null {
  if ((!raw.includes('=') && !raw.includes('|') && !raw.includes('｜') && !raw.includes(',')) || raw.length > 420) return null;
  const separator = raw.includes('｜') ? '｜' : raw.includes('|') ? '|' : ',';
  const segments = raw.split(separator).map((part) => part.trim()).filter(Boolean);
  if (segments.length < 2) return null;
  const rendered = segments.map(humanizeAssignmentSegment);
  const changed = rendered.some((part, index) => part !== segments[index]);
  return changed ? rendered.join(' ｜ ') : null;
}

function humanizePriorityOrder(raw: string): string | null {
  if (!raw.includes('>') || !raw.includes('@')) return null;
  const segments = raw.split('>').map((part) => part.trim()).filter(Boolean);
  if (!segments.length || segments.length > 8) return null;

  const rendered = segments.map((segment) => {
    const match = segment.match(/^([A-Za-z_]+)@(\d+(?:\.\d+)?):(\d+)$/);
    if (!match) return null;
    const [, state, priority, count] = match;
    return `${renderFragment(state)} ${priority} ${labelFor('unit_points')} / ${count} ${labelFor('unit_items')}`;
  });

  if (rendered.some((value) => !value)) return null;
  return rendered.join(' ＞ ');
}

function replaceKnownCodesInSentence(raw: string): string {
  let output = raw;
  for (const [pattern, replacement] of SENTENCE_REPLACEMENTS) {
    output = output.replace(pattern, replacement);
  }
  const keys = Object.keys(LABELS)
    .filter((key) => key.length >= 6 && /[_-]/.test(key))
    .sort((left, right) => right.length - left.length);

  for (const key of keys) {
    if (!output.includes(key)) continue;
    output = output.replaceAll(key, LABELS[key]);
  }
  return output;
}

function humanizeText(raw: string): string {
  const exact = exactLabel(raw);
  if (exact) return exact;
  if (!raw) return '—';
  if (isUrlLike(raw) || isPathLike(raw) || isTimestampLike(raw)) return raw;

  const structured = humanizeStructuredSummary(raw);
  if (structured) return structured;

  const slash = humanizeJoinedPhrase(raw, ' / ');
  if (slash) return slash;

  const plus = humanizeJoinedPhrase(raw, '+', ' + ');
  if (plus) return plus;

  const priority = humanizePriorityOrder(raw);
  if (priority) return priority;

  const colon = humanizeColonPhrase(raw);
  if (colon) return colon;

  if (isNaturalSentence(raw)) {
    return replaceKnownCodesInSentence(raw);
  }

  const simple = humanizeSimpleCode(raw);
  return simple || raw;
}

export function explainLabel(value: string | undefined): LabelExplanation {
  if (!value) return { primary: '—', translated: false };
  const raw = String(value).trim();
  if (!raw) return { primary: '—', translated: false };
  const primary = humanizeText(raw);
  const translated = primary !== raw;
  const secondary = translated && looksCodeLike(raw) && !isPathLike(raw) && !isUrlLike(raw) && raw.length <= 220 ? raw : undefined;
  return { primary, secondary, translated };
}

export function displayText(value: string | undefined): string {
  return explainLabel(value).primary;
}

export function labelFor(value: string | undefined): string {
  return displayText(value);
}

export function surfaceLabel(surface: SurfaceKey): string {
  return labelFor(surface === 'internal' ? 'terminal_internal' : 'terminal_public');
}
