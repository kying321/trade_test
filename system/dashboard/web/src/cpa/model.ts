export const CPA_ERROR_LEDGER_STORAGE_KEY = 'fenlie_cpa_error_ledger_v1';
export const CPA_USAGE_URL = 'https://chatgpt.com/backend-api/wham/usage';

export const STRUCTURAL_ERROR_BUCKETS = [
  '401',
  '402',
  'invalid_json',
  'missing_auth_index',
  'missing_account_id',
  'other_structural_error',
] as const;

export const TRANSIENT_ERROR_BUCKETS = [
  'network_error',
  'timeout',
  'upstream_5xx',
  'transient_unknown',
] as const;

export const CPA_ERROR_BUCKETS = [...STRUCTURAL_ERROR_BUCKETS, ...TRANSIENT_ERROR_BUCKETS] as const;

export type CpaErrorBucket = (typeof CPA_ERROR_BUCKETS)[number];
export type CpaQueryStatus = 'idle' | 'loading' | 'ok' | 'error';

export type CpaAuthFile = Record<string, unknown> & {
  name: string;
  disabled?: boolean;
  type?: string;
  provider?: string;
  auth_index?: string | number | null;
  id_token?: Record<string, unknown> | null;
};

export type CpaLedgerEntry = {
  last_error_type?: CpaErrorBucket;
  same_error_streak: number;
  total_error_count: number;
  last_error_at?: string;
  last_success_at?: string;
  eligible_for_delete: boolean;
};

export type CpaLedgerStore = {
  version: 1;
  servers: Record<string, Record<string, CpaLedgerEntry>>;
};

export type CpaRowRecord = {
  name: string;
  file: CpaAuthFile;
  disabled: boolean;
  planType: string | null;
  status: CpaQueryStatus;
  error: string;
  errorBucket?: CpaErrorBucket;
  shortRemaining: number | null;
  remaining: number | null;
  resetLabel: string;
  ledger?: Partial<CpaLedgerEntry>;
};

export type CpaDisplayRow = CpaRowRecord & {
  sameErrorStreak: number;
  totalErrorCount: number;
  eligibleForDelete: boolean;
  lastErrorType?: CpaErrorBucket;
  lastErrorAt?: string;
  lastSuccessAt?: string;
};

export type CpaDeleteBucketOption = {
  value: CpaErrorBucket;
  label: string;
  deletable: boolean;
};

export const CPA_DELETE_BUCKET_OPTIONS: CpaDeleteBucketOption[] = [
  { value: '401', label: '401', deletable: true },
  { value: '402', label: '402', deletable: true },
  { value: 'invalid_json', label: 'invalid_json', deletable: true },
  { value: 'missing_auth_index', label: 'missing_auth_index', deletable: true },
  { value: 'missing_account_id', label: 'missing_account_id', deletable: true },
  { value: 'other_structural_error', label: 'other_structural_error', deletable: true },
  { value: 'network_error', label: 'network_error（仅统计）', deletable: false },
  { value: 'timeout', label: 'timeout（仅统计）', deletable: false },
  { value: 'upstream_5xx', label: 'upstream_5xx（仅统计）', deletable: false },
  { value: 'transient_unknown', label: 'transient_unknown（仅统计）', deletable: false },
];

function emptyLedgerEntry(): CpaLedgerEntry {
  return {
    same_error_streak: 0,
    total_error_count: 0,
    eligible_for_delete: false,
  };
}

export function loadCpaErrorLedger(raw?: string | null): CpaLedgerStore {
  if (raw === undefined) {
    if (typeof window === 'undefined') return { version: 1, servers: {} };
    return loadCpaErrorLedger(window.localStorage.getItem(CPA_ERROR_LEDGER_STORAGE_KEY));
  }
  if (raw === null) {
    return { version: 1, servers: {} };
  }
  if (raw !== '') {
    try {
      const parsed = JSON.parse(raw) as CpaLedgerStore;
      if (parsed.version === 1 && parsed.servers && typeof parsed.servers === 'object') return parsed;
    } catch {
      return { version: 1, servers: {} };
    }
    return { version: 1, servers: {} };
  }
  return { version: 1, servers: {} };
}

export function persistCpaErrorLedger(ledger: CpaLedgerStore) {
  if (typeof window === 'undefined') return;
  window.localStorage.setItem(CPA_ERROR_LEDGER_STORAGE_KEY, JSON.stringify(ledger));
}

export function buildManagementBase(server: string): string {
  let raw = String(server || '').trim().replace(/\/+$/, '');
  if (!raw) return '';
  if (!/^https?:\/\//i.test(raw)) raw = `http://${raw}`;
  return `${raw}/v0/management`;
}

export function buildManagementHeaders(password: string) {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };
  const pw = String(password || '').trim();
  if (pw) headers.Authorization = `Bearer ${pw}`;
  return headers;
}

export function isCodexFile(file: CpaAuthFile): boolean {
  const provider = String(file.type || file.provider || '').toLowerCase();
  if (provider === 'codex') return true;
  const name = String(file.name || '').toLowerCase();
  return name.startsWith('token_oc') || name.includes('codex');
}

export function resolveCodexAccountId(file: CpaAuthFile): string | null {
  const token = file.id_token;
  if (token && typeof token === 'object') {
    const raw = (token.chatgpt_account_id || token.chatgptAccountId) as string | undefined;
    if (raw) return String(raw).trim();
  }
  for (const key of ['chatgpt_account_id', 'chatgptAccountId', 'account_id', 'accountId'] as const) {
    const raw = file[key];
    if (raw) return String(raw).trim();
  }
  return null;
}

export function formatReset(raw: unknown): string {
  if (!raw) return '';
  const date = new Date(typeof raw === 'number' ? raw * 1000 : String(raw));
  if (Number.isNaN(date.getTime())) return '';
  const now = new Date();
  const diff = date.getTime() - now.getTime();
  if (diff < 0) return '已重置';
  const pad = (value: number) => String(value).padStart(2, '0');
  const label = `${date.getMonth() + 1}/${date.getDate()} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
  const h = Math.floor(diff / 3_600_000);
  const m = Math.floor((diff % 3_600_000) / 60_000);
  return `${label} (${h > 0 ? `${h}h` : ''}${m}m后)`;
}

function parseWindowRemaining(windowData: Record<string, unknown> | null | undefined, blocked = false): number | null {
  if (!windowData) return null;
  const raw = windowData.used_percent ?? windowData.usedPercent;
  const usedPct = raw !== undefined && raw !== null ? Number(raw) : blocked ? 100 : Number.NaN;
  if (Number.isNaN(usedPct)) return null;
  return Math.max(0, 100 - usedPct);
}

export function parseWeeklyQuota(payload: unknown): {
  planType: string | null;
  shortRemaining: number | null;
  remaining: number | null;
  resetLabel: string;
} {
  if (!payload || typeof payload !== 'object') {
    return { planType: null, shortRemaining: null, remaining: null, resetLabel: '' };
  }
  const record = payload as Record<string, unknown>;
  const planType = typeof record.plan_type === 'string' ? record.plan_type : typeof record.planType === 'string' ? record.planType : null;
  const rateLimit = (record.rate_limit || record.rateLimit) as Record<string, unknown> | undefined;
  if (!rateLimit) return { planType, shortRemaining: null, remaining: null, resetLabel: '' };
  const shortWindow = (rateLimit.secondary_window || rateLimit.secondaryWindow) as Record<string, unknown> | undefined;
  const primaryWindow = (rateLimit.primary_window || rateLimit.primaryWindow) as Record<string, unknown> | undefined;
  const blocked = Boolean(rateLimit.limit_reached || rateLimit.limitReached) || rateLimit.allowed === false;
  return {
    planType,
    shortRemaining: parseWindowRemaining(shortWindow, blocked),
    remaining: parseWindowRemaining(primaryWindow, blocked),
    resetLabel: formatReset(primaryWindow?.reset_at || primaryWindow?.resetAt || primaryWindow?.resets_at || primaryWindow?.resetsAt),
  };
}

export function classifyCpaErrorBucket(error: unknown): CpaErrorBucket {
  const message = String(error instanceof Error ? error.message : error || '').toLowerCase();
  if (/(^|\D)401(\D|$)/.test(message)) return '401';
  if (/(^|\D)402(\D|$)/.test(message)) return '402';
  if (message.includes('invalid_json') || message.includes('json decode') || message.includes('json_payload')) return 'invalid_json';
  if (message.includes('auth_index')) return 'missing_auth_index';
  if (message.includes('chatgpt-account-id') || message.includes('account-id') || message.includes('account_id')) return 'missing_account_id';
  if (message.includes('timeout') || message.includes('timed out')) return 'timeout';
  if (/(^|\D)5\d\d(\D|$)/.test(message)) return 'upstream_5xx';
  if (message.includes('failed to fetch') || message.includes('network')) return 'network_error';
  if (/(^|\D)4\d\d(\D|$)/.test(message)) return 'other_structural_error';
  return 'transient_unknown';
}

export function isStructuralErrorBucket(bucket: CpaErrorBucket | undefined | null): boolean {
  return Boolean(bucket && STRUCTURAL_ERROR_BUCKETS.includes(bucket as (typeof STRUCTURAL_ERROR_BUCKETS)[number]));
}

function cloneLedger(ledger: CpaLedgerStore): CpaLedgerStore {
  return {
    version: 1,
    servers: Object.fromEntries(
      Object.entries(ledger.servers || {}).map(([server, entries]) => [server, { ...entries }]),
    ),
  };
}

export function getLedgerEntry(ledger: CpaLedgerStore, serverBase: string, fileName: string): CpaLedgerEntry {
  return ledger.servers?.[serverBase]?.[fileName] || emptyLedgerEntry();
}

export function reduceLedgerOnQuotaError(
  ledger: CpaLedgerStore,
  serverBase: string,
  fileName: string,
  bucket: CpaErrorBucket,
  at: string,
): CpaLedgerStore {
  const next = cloneLedger(ledger);
  const current = getLedgerEntry(next, serverBase, fileName);
  const streak = current.last_error_type === bucket ? current.same_error_streak + 1 : 1;
  next.servers[serverBase] ||= {};
  next.servers[serverBase][fileName] = {
    ...current,
    last_error_type: bucket,
    same_error_streak: streak,
    total_error_count: current.total_error_count + 1,
    last_error_at: at,
    eligible_for_delete: isStructuralErrorBucket(bucket) && streak >= 3,
  };
  return next;
}

export function reduceLedgerOnQuotaSuccess(
  ledger: CpaLedgerStore,
  serverBase: string,
  fileName: string,
  at: string,
): CpaLedgerStore {
  const next = cloneLedger(ledger);
  const current = getLedgerEntry(next, serverBase, fileName);
  next.servers[serverBase] ||= {};
  next.servers[serverBase][fileName] = {
    ...current,
    same_error_streak: 0,
    eligible_for_delete: false,
    last_success_at: at,
  };
  return next;
}

export function removeLedgerEntry(ledger: CpaLedgerStore, serverBase: string, fileName: string): CpaLedgerStore {
  const next = cloneLedger(ledger);
  if (next.servers[serverBase]) {
    delete next.servers[serverBase][fileName];
  }
  return next;
}

export function createRowsFromAuthFiles(files: CpaAuthFile[]): CpaRowRecord[] {
  return files.filter(isCodexFile).map((file) => ({
    name: String(file.name),
    file,
    disabled: Boolean(file.disabled),
    planType: typeof file.id_token?.plan_type === 'string' ? String(file.id_token.plan_type).toLowerCase() : null,
    status: 'idle',
    error: '',
    shortRemaining: null,
    remaining: null,
    resetLabel: '',
  }));
}

export function decorateRowsWithLedger(rows: CpaRowRecord[], ledger: CpaLedgerStore, serverBase: string): CpaDisplayRow[] {
  return rows.map((row) => {
    const current = getLedgerEntry(ledger, serverBase, row.name);
    return {
      ...row,
      sameErrorStreak: current.same_error_streak,
      totalErrorCount: current.total_error_count,
      eligibleForDelete: current.eligible_for_delete,
      lastErrorType: current.last_error_type,
      lastErrorAt: current.last_error_at,
      lastSuccessAt: current.last_success_at,
      ledger: current,
    };
  });
}

export function getDeleteCandidates(rows: CpaRowRecord[], bucket: CpaErrorBucket | ''): CpaRowRecord[] {
  if (!bucket || !isStructuralErrorBucket(bucket)) return [];
  return rows.filter((row) => row.errorBucket === bucket && Boolean(row.ledger?.eligible_for_delete));
}

export function summarizeErrorBuckets(rows: CpaRowRecord[]): Record<string, number> {
  return rows.reduce<Record<string, number>>((acc, row) => {
    if (row.status !== 'error' || !row.errorBucket) return acc;
    acc[row.errorBucket] = (acc[row.errorBucket] || 0) + 1;
    return acc;
  }, {});
}
