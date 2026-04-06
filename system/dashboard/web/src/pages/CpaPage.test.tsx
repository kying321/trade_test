import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { CPA_ERROR_LEDGER_STORAGE_KEY } from '../cpa/model';
import { CpaPage } from './CpaPage';

function mockAuthFiles() {
  return [
    {
      name: 'token_oc_alpha.json',
      email: 'hpgwqc62wtwg@fuuu.fun',
      type: 'codex',
      auth_index: '1',
      disabled: false,
      id_token: { plan_type: 'plus', chatgpt_account_id: 'acct_1' },
    },
    {
      name: 'non_chat_file.json',
      type: 'chatgpt',
      auth_index: '2',
      disabled: false,
    },
  ];
}

describe('CpaPage', () => {
  beforeEach(() => {
    window.localStorage.clear();
    vi.restoreAllMocks();
    vi.stubGlobal('confirm', vi.fn(() => true));
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('renders config fields and loads codex auth files into a fenlie-style table', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/data/cpa_control_plane_snapshot.json')) {
        return {
          ok: true,
          json: async () => ({
            summary: {
              historical_success_total: 20,
              active_target_authfiles: 20,
              inventory_total: 72,
              retry_candidate_total: 5,
              blocked_about_you_total: 8,
              no_retry_deactivated_total: 11,
              new_unmounted_total: 6,
              historical_success_in_usable_active_total: 1,
              historical_success_non_active_total: 19,
              historical_success_missing_from_inventory_total: 0,
            },
            historical_success_emails: ['hpgwqc62wtwg@fuuu.fun', 'oezru5cjfn@fuuu.fun'],
            guarded_actions: [
              {
                id: 'acceptance_replay_success20',
                label: '验收回放 / 历史成功20',
                command: 'cd /Users/jokenrobot/Downloads/Folders/MAC工具 && python3 check_five_account_acceptance.py --csv data/registered_success_active20.csv --target-count 20 --store data/pipeline_store.sqlite3 --pretty',
              },
            ],
            groups: {
              retry_candidate_rows: [
                {
                  email: 'retry1@fuuu.fun',
                  bucket: 'retry_candidate',
                  reason: 'needs remount',
                  artifact_title: '欢迎回来',
                  artifact_flags: 'consent,welcome_back',
                },
              ],
              blocked_about_you_rows: [
                {
                  email: 'about1@fuuu.fun',
                  bucket: 'blocked_about_you',
                  reason: 'stuck on age/about-you',
                  artifact_title: '确认一下你的年龄',
                  artifact_flags: 'about_you,consent',
                },
              ],
              no_retry_deactivated_rows: [
                {
                  email: 'dead1@fuuu.fun',
                  bucket: 'no_retry_deactivated',
                  reason: 'deactivated / disabled',
                  artifact_title: '糟糕，出错了！',
                  artifact_flags: 'deactivated',
                },
              ],
              new_unmounted_rows: [
                {
                  email: 'new1@fuuu.fun',
                  timestamp: '2026-03-29 18:51:17',
                },
              ],
            },
            latest_kernel_run_id: 'run-kernel-1',
          }),
        };
      }
      if (url.includes('/auth-files')) {
        return {
          ok: true,
          json: async () => mockAuthFiles(),
        };
      }
      throw new Error(`unexpected fetch ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    render(<CpaPage />);
    fireEvent.change(screen.getByPlaceholderText('请输入服务器地址'), { target: { value: '127.0.0.1:8787' } });
    fireEvent.change(screen.getByPlaceholderText('请输入管理员密码'), { target: { value: 'secret' } });
    fireEvent.click(screen.getByRole('button', { name: '开始连接' }));

    expect(await screen.findByRole('heading', { name: 'CPA 管理' })).toBeTruthy();
    expect(await screen.findByText(/历史成功快照/)).toBeTruthy();
    expect(screen.getByText(/20 个历史成功账号/)).toBeTruthy();
    expect(screen.getByText(/Guarded Actions/)).toBeTruthy();
    expect(screen.getByText(/验收回放 \/ 历史成功20/)).toBeTruthy();
    expect(screen.getByText(/retry_candidate 队列/)).toBeTruthy();
    expect(screen.getByText(/blocked_about_you 队列/)).toBeTruthy();
    expect(screen.getByText(/no_retry_deactivated 队列/)).toBeTruthy();
    expect(screen.getByText(/new_unmounted 池/)).toBeTruthy();
    expect(screen.getByText(/retry1@fuuu.fun/)).toBeTruthy();
    expect(screen.getByText(/about1@fuuu.fun/)).toBeTruthy();
    expect(screen.getByText(/dead1@fuuu.fun/)).toBeTruthy();
    expect(screen.getByText(/new1@fuuu.fun/)).toBeTruthy();
    expect(await screen.findByText('token_oc_alpha')).toBeTruthy();
    expect(screen.getByText(/live 已连接 1/)).toBeTruthy();
    expect(screen.getByText(/历史成功命中 1/)).toBeTruthy();
    expect(screen.queryByText('non_chat_file')).toBeNull();
    expect(screen.getByRole('button', { name: '查询额度' })).toBeTruthy();
    expect(screen.getByRole('button', { name: '删除401账号' })).toBeTruthy();
    expect(screen.getByRole('button', { name: '删除402账号' })).toBeTruthy();
    expect(screen.getByLabelText('删除错误桶')).toBeTruthy();
  });

  it('marks a row deletable only after the same structural error appears three times', async () => {
    const apiCallBodies = [
      { status_code: 401, body: JSON.stringify({ error: { message: 'unauthorized' } }) },
      { status_code: 401, body: JSON.stringify({ error: { message: 'unauthorized' } }) },
      { status_code: 401, body: JSON.stringify({ error: { message: 'unauthorized' } }) },
      { status_code: 200, body: JSON.stringify({ plan_type: 'plus', rate_limit: { primary_window: { used_percent: 10, reset_at: '2026-04-03T10:00:00Z' }, secondary_window: { used_percent: 20 } } }) },
    ];
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/data/cpa_control_plane_snapshot.json')) {
        return {
          ok: true,
          json: async () => ({ summary: {} }),
        };
      }
      if (url.includes('/auth-files')) {
        return {
          ok: true,
          json: async () => mockAuthFiles(),
        };
      }
      if (url.includes('/api-call')) {
        const next = apiCallBodies.shift();
        return {
          ok: true,
          json: async () => next,
        };
      }
      throw new Error(`unexpected fetch ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    render(<CpaPage />);
    fireEvent.change(screen.getByPlaceholderText('请输入服务器地址'), { target: { value: '127.0.0.1:8787' } });
    fireEvent.click(screen.getByRole('button', { name: '开始连接' }));
    await screen.findByText('token_oc_alpha');

    const runQuery = async () => {
      fireEvent.click(screen.getByRole('button', { name: '查询额度' }));
      await waitFor(() => {
        expect(screen.getByText(/连续同错/)).toBeTruthy();
      });
    };

    await runQuery();
    await runQuery();
    await runQuery();

    const row = screen.getByRole('row', { name: /token_oc_alpha/i });
    expect(within(row).getByText('可删')).toBeTruthy();
    expect(within(row).getAllByText('3').length).toBeGreaterThan(0);

    const deleteSelect = screen.getByLabelText('删除错误桶');
    fireEvent.change(deleteSelect, { target: { value: '401' } });
    expect((deleteSelect as HTMLSelectElement).value).toBe('401');

    fireEvent.click(screen.getByRole('button', { name: '查询额度' }));
    await waitFor(() => {
      expect(screen.queryByText('可删')).toBeNull();
    });

    const ledger = JSON.parse(window.localStorage.getItem(CPA_ERROR_LEDGER_STORAGE_KEY) || '{}');
    expect(JSON.stringify(ledger)).toContain('last_success_at');
  });

  it('never marks network failures deletable even when they repeat', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/data/cpa_control_plane_snapshot.json')) {
        return {
          ok: true,
          json: async () => ({ summary: {} }),
        };
      }
      if (url.includes('/auth-files')) {
        return {
          ok: true,
          json: async () => mockAuthFiles(),
        };
      }
      if (url.includes('/api-call')) {
        throw new TypeError('Failed to fetch');
      }
      throw new Error(`unexpected fetch ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    render(<CpaPage />);
    fireEvent.change(screen.getByPlaceholderText('请输入服务器地址'), { target: { value: '127.0.0.1:8787' } });
    fireEvent.click(screen.getByRole('button', { name: '开始连接' }));
    await screen.findByText('token_oc_alpha');

    fireEvent.click(screen.getByRole('button', { name: '查询额度' }));
    await waitFor(() => {
      expect(within(screen.getByTestId('cpa-error-buckets')).getByText(/network_error/i)).toBeTruthy();
    });

    fireEvent.click(screen.getByRole('button', { name: '查询额度' }));
    await waitFor(() => {
      expect(within(screen.getByTestId('cpa-error-buckets')).getByText(/network_error/i)).toBeTruthy();
    });
    fireEvent.click(screen.getByRole('button', { name: '查询额度' }));
    await waitFor(() => {
      expect(within(screen.getByTestId('cpa-error-buckets')).getByText(/network_error/i)).toBeTruthy();
    });

    expect(screen.queryByText('可删')).toBeNull();
    const bucketSummary = screen.getByTestId('cpa-error-buckets');
    expect(within(bucketSummary).getByText(/network_error/i)).toBeTruthy();
  });
});
