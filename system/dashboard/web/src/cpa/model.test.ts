import { describe, expect, it } from 'vitest';
import {
  CPA_ERROR_LEDGER_STORAGE_KEY,
  classifyCpaErrorBucket,
  getDeleteCandidates,
  isStructuralErrorBucket,
  loadCpaErrorLedger,
  reduceLedgerOnQuotaError,
  reduceLedgerOnQuotaSuccess,
  type CpaRowRecord,
} from './model';

describe('cpa model helpers', () => {
  it('classifies structural and transient error buckets', () => {
    expect(classifyCpaErrorBucket(new Error('api-call 失败: HTTP 401 – unauthorized'))).toBe('401');
    expect(classifyCpaErrorBucket(new Error('api-call 失败: HTTP 402 – payment required'))).toBe('402');
    expect(classifyCpaErrorBucket(new Error('invalid_json_payload'))).toBe('invalid_json');
    expect(classifyCpaErrorBucket(new Error('缺少 auth_index'))).toBe('missing_auth_index');
    expect(classifyCpaErrorBucket(new Error('无法获取 Chatgpt-Account-Id'))).toBe('missing_account_id');
    expect(classifyCpaErrorBucket(new Error('HTTP 503'))).toBe('upstream_5xx');
    expect(classifyCpaErrorBucket(new TypeError('Failed to fetch'))).toBe('network_error');
    expect(classifyCpaErrorBucket(new Error('timeout while contacting upstream'))).toBe('timeout');
    expect(isStructuralErrorBucket('401')).toBe(true);
    expect(isStructuralErrorBucket('network_error')).toBe(false);
  });

  it('marks only same structural error repeated three times as delete-eligible', () => {
    const base = 'http://localhost:8787/v0/management';
    const name = 'token_oc_demo.json';
    let ledger = loadCpaErrorLedger();

    ledger = reduceLedgerOnQuotaError(ledger, base, name, '401', '2026-04-02T10:00:00Z');
    ledger = reduceLedgerOnQuotaError(ledger, base, name, '401', '2026-04-02T10:01:00Z');
    ledger = reduceLedgerOnQuotaError(ledger, base, name, '401', '2026-04-02T10:02:00Z');

    expect(ledger.servers[base][name]).toMatchObject({
      last_error_type: '401',
      same_error_streak: 3,
      total_error_count: 3,
      eligible_for_delete: true,
    });

    ledger = reduceLedgerOnQuotaSuccess(ledger, base, name, '2026-04-02T10:03:00Z');
    expect(ledger.servers[base][name]).toMatchObject({
      last_error_type: '401',
      same_error_streak: 0,
      total_error_count: 3,
      eligible_for_delete: false,
      last_success_at: '2026-04-02T10:03:00Z',
    });
  });

  it('resets streak when error type changes and never marks network errors deletable', () => {
    const base = 'http://localhost:8787/v0/management';
    const name = 'token_oc_demo.json';
    let ledger = loadCpaErrorLedger();

    ledger = reduceLedgerOnQuotaError(ledger, base, name, '401', '2026-04-02T10:00:00Z');
    ledger = reduceLedgerOnQuotaError(ledger, base, name, 'invalid_json', '2026-04-02T10:01:00Z');
    expect(ledger.servers[base][name]).toMatchObject({
      last_error_type: 'invalid_json',
      same_error_streak: 1,
      total_error_count: 2,
      eligible_for_delete: false,
    });

    ledger = reduceLedgerOnQuotaError(ledger, base, name, 'network_error', '2026-04-02T10:02:00Z');
    ledger = reduceLedgerOnQuotaError(ledger, base, name, 'network_error', '2026-04-02T10:03:00Z');
    ledger = reduceLedgerOnQuotaError(ledger, base, name, 'network_error', '2026-04-02T10:04:00Z');
    expect(ledger.servers[base][name]).toMatchObject({
      last_error_type: 'network_error',
      same_error_streak: 3,
      eligible_for_delete: false,
    });
  });

  it('returns delete candidates only for eligible rows in the selected structural bucket', () => {
    const rows: CpaRowRecord[] = [
      {
        name: 'token_oc_a.json',
        file: { name: 'token_oc_a.json' },
        disabled: false,
        planType: 'plus',
        status: 'error',
        error: '401 unauthorized',
        errorBucket: '401',
        shortRemaining: null,
        remaining: null,
        resetLabel: '',
        ledger: {
          last_error_type: '401',
          same_error_streak: 3,
          total_error_count: 3,
          eligible_for_delete: true,
        },
      },
      {
        name: 'token_oc_b.json',
        file: { name: 'token_oc_b.json' },
        disabled: false,
        planType: 'plus',
        status: 'error',
        error: 'network',
        errorBucket: 'network_error',
        shortRemaining: null,
        remaining: null,
        resetLabel: '',
        ledger: {
          last_error_type: 'network_error',
          same_error_streak: 4,
          total_error_count: 4,
          eligible_for_delete: false,
        },
      },
    ];

    expect(getDeleteCandidates(rows, '401').map((row) => row.name)).toEqual(['token_oc_a.json']);
    expect(getDeleteCandidates(rows, 'network_error')).toEqual([]);
  });

  it('uses versioned localStorage schema', () => {
    const ledger = loadCpaErrorLedger();
    expect(ledger.version).toBe(1);
    expect(CPA_ERROR_LEDGER_STORAGE_KEY).toBe('fenlie_cpa_error_ledger_v1');
  });
});
