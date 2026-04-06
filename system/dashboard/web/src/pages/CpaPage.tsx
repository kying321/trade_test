import { useEffect, useMemo, useState } from 'react';
import { ActionButton } from '../components/control-primitives';
import { Badge, PanelCard } from '../components/ui-kit';
import {
  buildManagementBase,
  buildManagementHeaders,
  classifyCpaErrorBucket,
  CPA_DELETE_BUCKET_OPTIONS,
  CPA_ERROR_LEDGER_STORAGE_KEY,
  CPA_USAGE_URL,
  createRowsFromAuthFiles,
  decorateRowsWithLedger,
  getDeleteCandidates,
  isStructuralErrorBucket,
  loadCpaErrorLedger,
  parseWeeklyQuota,
  persistCpaErrorLedger,
  reduceLedgerOnQuotaError,
  reduceLedgerOnQuotaSuccess,
  removeLedgerEntry,
  resolveCodexAccountId,
  summarizeErrorBuckets,
  type CpaAuthFile,
  type CpaDisplayRow,
  type CpaErrorBucket,
  type CpaLedgerStore,
  type CpaRowRecord,
} from '../cpa/model';

const PAGE_SIZE = 1000;
const QUOTA_CONCURRENCY = 10;

type StatusTone = 'info' | 'error' | 'success';

type StatusMessage = {
  tone: StatusTone;
  text: string;
};

type CpaControlSnapshot = {
  summary?: Record<string, unknown>;
  latest_kernel_run_id?: string;
  historical_success_emails?: string[];
  guarded_actions?: Array<Record<string, unknown>>;
};

function trimJsonSuffix(name: string) {
  return name.replace(/\.json$/i, '');
}

function resolveLiveFileEmail(file: CpaAuthFile): string {
  for (const key of ['email', 'account', 'name'] as const) {
    const raw = String(file[key] || '').trim().toLowerCase();
    if (!raw) continue;
    const match = raw.match(/[a-z0-9._%+-]+@[a-z0-9.-]+/);
    if (match) return match[0];
  }
  return '';
}

function planTone(planType: string | null | undefined) {
  if (planType === 'free') return 'neutral';
  if (planType === 'plus') return 'positive';
  if (planType === 'team') return 'warning';
  return 'neutral';
}

async function fetchAuthFiles(base: string, headers: Record<string, string>) {
  const resp = await fetch(`${base}/auth-files`, { headers });
  if (!resp.ok) {
    let detail = '';
    try {
      const payload = await resp.json();
      detail = payload?.error || '';
    } catch {
      detail = '';
    }
    throw new Error(`获取认证文件失败: HTTP ${resp.status}${detail ? ` – ${detail}` : ''}`);
  }
  const data = await resp.json();
  if (Array.isArray(data)) return data as CpaAuthFile[];
  if (Array.isArray(data.files)) return data.files as CpaAuthFile[];
  return [];
}

async function apiToggleAuth(base: string, headers: Record<string, string>, name: string, disabled: boolean) {
  const resp = await fetch(`${base}/auth-files/status`, {
    method: 'PATCH',
    headers,
    body: JSON.stringify({ name, disabled }),
  });
  if (!resp.ok) {
    let detail = '';
    try {
      const payload = await resp.json();
      detail = payload?.error || '';
    } catch {
      detail = '';
    }
    throw new Error(`操作失败: HTTP ${resp.status}${detail ? ` – ${detail}` : ''}`);
  }
}

async function apiDeleteAuth(base: string, headers: Record<string, string>, name: string) {
  const resp = await fetch(`${base}/auth-files?name=${encodeURIComponent(name)}`, {
    method: 'DELETE',
    headers,
  });
  if (!resp.ok) {
    let detail = '';
    try {
      const payload = await resp.json();
      detail = payload?.error || '';
    } catch {
      detail = '';
    }
    throw new Error(`删除失败: HTTP ${resp.status}${detail ? ` – ${detail}` : ''}`);
  }
}

async function fetchCodexQuota(base: string, headers: Record<string, string>, file: CpaAuthFile) {
  const authIndex = file.auth_index;
  if (authIndex == null) throw new Error('缺少 auth_index');
  const accountId = resolveCodexAccountId(file);
  if (!accountId) throw new Error('无法获取 Chatgpt-Account-Id');

  const payload = {
    auth_index: String(authIndex),
    method: 'GET',
    url: CPA_USAGE_URL,
    header: {
      Authorization: 'Bearer $TOKEN$',
      'Content-Type': 'application/json',
      'User-Agent': 'codex_cli_rs/0.76.0 (Debian 13.0.0; x86_64) WindowsTerminal',
      'Chatgpt-Account-Id': accountId,
    },
  };

  const resp = await fetch(`${base}/api-call`, {
    method: 'POST',
    headers,
    body: JSON.stringify(payload),
  });
  if (!resp.ok) {
    let detail = '';
    try {
      const result = await resp.json();
      detail = result?.error || '';
    } catch {
      detail = '';
    }
    throw new Error(`api-call 失败: HTTP ${resp.status}${detail ? ` – ${detail}` : ''}`);
  }

  const result = await resp.json();
  const statusCode = Number(result?.status_code ?? result?.statusCode ?? 0);
  if (statusCode < 200 || statusCode >= 300) {
    let message = `上游返回 ${statusCode}`;
    try {
      const body = typeof result.body === 'string' ? JSON.parse(result.body) : result.body;
      const error = body?.error;
      message = (typeof error === 'object' ? error?.message : error) || body?.message || message;
    } catch {
      // ignore json parse failure
    }
    throw new Error(`上游返回 ${statusCode}${message ? ` – ${message}` : ''}`);
  }

  if (typeof result.body === 'string') {
    try {
      return JSON.parse(result.body);
    } catch {
      return result.body;
    }
  }
  return result.body;
}

async function runWithConcurrency<T>(items: T[], limit: number, worker: (item: T) => Promise<void>) {
  let cursor = 0;
  const runners = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (cursor < items.length) {
      const current = items[cursor++];
      await worker(current);
    }
  });
  await Promise.all(runners);
}

function matchFilters(row: CpaDisplayRow, filters: {
  search: string;
  plan: string;
  status: string;
  query: string;
}) {
  if (filters.search && !row.name.toLowerCase().includes(filters.search)) return false;
  if (filters.plan && (row.planType || '').toLowerCase() !== filters.plan) return false;
  if (filters.status === 'enabled' && row.disabled) return false;
  if (filters.status === 'disabled' && !row.disabled) return false;
  if (filters.query === 'success' && row.status !== 'ok') return false;
  if (filters.query === 'failed' && row.status !== 'error') return false;
  return true;
}

export function CpaPage() {
  const [server, setServer] = useState('');
  const [password, setPassword] = useState('');
  const [controlSnapshot, setControlSnapshot] = useState<CpaControlSnapshot | null>(null);
  const [rows, setRows] = useState<CpaRowRecord[]>([]);
  const [status, setStatus] = useState<StatusMessage | null>(null);
  const [connecting, setConnecting] = useState(false);
  const [querying, setQuerying] = useState(false);
  const [ledger, setLedger] = useState<CpaLedgerStore>(() => loadCpaErrorLedger());
  const [search, setSearch] = useState('');
  const [planFilter, setPlanFilter] = useState('');
  const [statusFilter, setStatusFilter] = useState('');
  const [queryFilter, setQueryFilter] = useState('');
  const [deleteBucket, setDeleteBucket] = useState<CpaErrorBucket | ''>('');
  const [page, setPage] = useState(1);

  useEffect(() => {
    persistCpaErrorLedger(ledger);
  }, [ledger]);

  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const resp = await fetch('/data/cpa_control_plane_snapshot.json');
        if (!resp.ok) return;
        const payload = await resp.json();
        if (!active || !payload || typeof payload !== 'object') return;
        setControlSnapshot(payload as CpaControlSnapshot);
      } catch {
        // degrade-only
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  const base = useMemo(() => buildManagementBase(server), [server]);
  const headers = useMemo(() => buildManagementHeaders(password), [password]);

  const displayRows = useMemo(
    () => decorateRowsWithLedger(rows, ledger, base),
    [base, ledger, rows],
  );

  const filteredRows = useMemo(() => displayRows.filter((row) => matchFilters(row, {
    search: search.trim().toLowerCase(),
    plan: planFilter,
    status: statusFilter,
    query: queryFilter,
  })), [displayRows, planFilter, queryFilter, search, statusFilter]);

  const pageCount = Math.max(1, Math.ceil(filteredRows.length / PAGE_SIZE));
  const currentPage = Math.min(page, pageCount);
  const pageRows = filteredRows.slice((currentPage - 1) * PAGE_SIZE, currentPage * PAGE_SIZE);
  const bucketCounts = useMemo(() => summarizeErrorBuckets(displayRows), [displayRows]);
  const deleteCandidates = useMemo(() => getDeleteCandidates(displayRows, deleteBucket), [deleteBucket, displayRows]);
  const eligibleDeleteCount = useMemo(() => displayRows.filter((row) => row.eligibleForDelete).length, [displayRows]);
  const selectedDeleteOption = CPA_DELETE_BUCKET_OPTIONS.find((option) => option.value === deleteBucket);
  const bucketDeleteEnabled = Boolean(deleteBucket && selectedDeleteOption?.deletable && deleteCandidates.length > 0);

  const updateRowByName = (name: string, updater: (row: CpaRowRecord) => CpaRowRecord) => {
    setRows((current) => current.map((row) => (row.name === name ? updater(row) : row)));
  };

  const removeRowByName = (name: string) => {
    setRows((current) => current.filter((row) => row.name !== name));
  };

  const handleConnect = async () => {
    if (!base) {
      setStatus({ tone: 'error', text: '请先输入服务器地址' });
      return;
    }
    setConnecting(true);
    setStatus({ tone: 'info', text: '正在获取认证文件列表...' });
    try {
      const files = await fetchAuthFiles(base, headers);
      const nextRows = createRowsFromAuthFiles(files);
      if (!nextRows.length) {
        setRows([]);
        setStatus({ tone: 'error', text: '未找到任何 Codex 类型的认证文件' });
      } else {
        setRows(nextRows);
        setPage(1);
        setStatus({ tone: 'success', text: `找到 ${nextRows.length} 个 Codex 文件` });
      }
    } catch (error) {
      setStatus({ tone: 'error', text: error instanceof Error ? error.message : '连接失败' });
    } finally {
      setConnecting(false);
    }
  };

  const handleQueryQuota = async () => {
    if (!base) {
      setStatus({ tone: 'error', text: '请先输入服务器地址' });
      return;
    }
    if (!filteredRows.length) {
      setStatus({ tone: 'info', text: '当前筛选条件下没有可查询的账号' });
      return;
    }
    setQuerying(true);
    setStatus({ tone: 'info', text: `开始查询额度，共 ${filteredRows.length} 个...` });
    const now = () => new Date().toISOString();
    filteredRows.forEach((row) => {
      updateRowByName(row.name, (current) => ({
        ...current,
        status: 'loading',
        error: '',
        errorBucket: undefined,
        shortRemaining: null,
        remaining: null,
        resetLabel: '',
      }));
    });

    let okCount = 0;
    let errorCount = 0;
    await runWithConcurrency(filteredRows, QUOTA_CONCURRENCY, async (row) => {
      try {
        const payload = await fetchCodexQuota(base, headers, row.file);
        const parsed = parseWeeklyQuota(payload);
        updateRowByName(row.name, (current) => ({
          ...current,
          status: 'ok',
          error: '',
          errorBucket: undefined,
          planType: parsed.planType ? parsed.planType.toLowerCase() : current.planType,
          shortRemaining: parsed.shortRemaining,
          remaining: parsed.remaining,
          resetLabel: parsed.resetLabel,
        }));
        setLedger((current) => reduceLedgerOnQuotaSuccess(current, base, row.name, now()));
        okCount += 1;
      } catch (error) {
        const bucket = classifyCpaErrorBucket(error);
        updateRowByName(row.name, (current) => ({
          ...current,
          status: 'error',
          error: error instanceof Error ? error.message : String(error),
          errorBucket: bucket,
          shortRemaining: null,
          remaining: null,
          resetLabel: '',
        }));
        setLedger((current) => reduceLedgerOnQuotaError(current, base, row.name, bucket, now()));
        errorCount += 1;
      }
    });

    setQuerying(false);
    setStatus({
      tone: okCount > 0 ? 'success' : 'error',
      text: `额度查询完成：${okCount} 成功，${errorCount} 失败`,
    });
  };

  const handleToggleAuth = async (row: CpaDisplayRow) => {
    try {
      await apiToggleAuth(base, headers, row.name, !row.disabled);
      updateRowByName(row.name, (current) => ({ ...current, disabled: !current.disabled }));
    } catch (error) {
      setStatus({ tone: 'error', text: error instanceof Error ? error.message : '状态切换失败' });
    }
  };

  const handleDeleteAuth = async (row: CpaDisplayRow) => {
    if (!window.confirm(`确认删除「${row.name}」？此操作不可撤销。`)) return;
    try {
      await apiDeleteAuth(base, headers, row.name);
      removeRowByName(row.name);
      setLedger((current) => removeLedgerEntry(current, base, row.name));
      setStatus({ tone: 'success', text: `已删除 ${row.name}` });
    } catch (error) {
      setStatus({ tone: 'error', text: error instanceof Error ? error.message : '删除失败' });
    }
  };

  const handleBatchDelete = async (bucket: CpaErrorBucket, preserveLegacy = false) => {
    const targets = preserveLegacy
      ? displayRows.filter((row) => row.errorBucket === bucket)
      : getDeleteCandidates(displayRows, bucket);
    if (!targets.length) {
      setStatus({ tone: 'info', text: preserveLegacy ? `没有需要删除的 ${bucket} 账号` : '当前错误桶下没有达到删除阈值的文件' });
      return;
    }
    if (!window.confirm(`将删除 ${targets.length} 个 ${bucket} 文件，确认继续？`)) return;
    let ok = 0;
    let fail = 0;
    for (const row of targets) {
      try {
        await apiDeleteAuth(base, headers, row.name);
        removeRowByName(row.name);
        setLedger((current) => removeLedgerEntry(current, base, row.name));
        ok += 1;
      } catch {
        fail += 1;
      }
    }
    setStatus({
      tone: ok > 0 ? 'success' : 'error',
      text: `批量操作完成：成功删除 ${ok} 个 ${bucket} 文件${fail ? `，失败 ${fail} 个` : ''}`,
    });
  };

  const handleEnablePositiveQuota = async () => {
    const targets = displayRows.filter((row) => row.status === 'ok' && row.disabled && (row.remaining ?? 0) > 0 && (row.shortRemaining ?? 0) > 0);
    if (!targets.length) {
      setStatus({ tone: 'info', text: '没有需要启用的账号（需同时满足周额度和5小时额度都大于 0）' });
      return;
    }
    let ok = 0;
    let fail = 0;
    for (const row of targets) {
      try {
        await apiToggleAuth(base, headers, row.name, false);
        updateRowByName(row.name, (current) => ({ ...current, disabled: false }));
        ok += 1;
      } catch {
        fail += 1;
      }
    }
    setStatus({ tone: ok > 0 ? 'success' : 'error', text: `批量操作完成：成功启用 ${ok} 个账号${fail ? `，失败 ${fail} 个` : ''}` });
  };

  const handleDisableNoQuota = async () => {
    const targets = displayRows.filter((row) => row.status === 'ok' && !row.disabled && (((row.remaining ?? 1) <= 0) || ((row.shortRemaining ?? 1) <= 0)));
    if (!targets.length) {
      setStatus({ tone: 'info', text: '没有需要关闭的无额度账号' });
      return;
    }
    let ok = 0;
    let fail = 0;
    for (const row of targets) {
      try {
        await apiToggleAuth(base, headers, row.name, true);
        updateRowByName(row.name, (current) => ({ ...current, disabled: true }));
        ok += 1;
      } catch {
        fail += 1;
      }
    }
    setStatus({ tone: ok > 0 ? 'success' : 'error', text: `批量操作完成：成功禁用 ${ok} 个无额度账号${fail ? `，失败 ${fail} 个` : ''}` });
  };

  const bucketSummaryEntries = Object.entries(bucketCounts).sort((a, b) => a[0].localeCompare(b[0]));
  const controlSummary = (controlSnapshot?.summary || {}) as Record<string, unknown>;
  const historicalSuccessEmails = Array.isArray(controlSnapshot?.historical_success_emails)
    ? controlSnapshot?.historical_success_emails.map((value) => String(value || '').trim().toLowerCase()).filter(Boolean)
    : [];
  const liveEmails = rows.map((row) => resolveLiveFileEmail(row.file)).filter(Boolean);
  const liveEmailSet = new Set(liveEmails);
  const historicalHitCount = historicalSuccessEmails.filter((email) => liveEmailSet.has(email)).length;
  const historicalMissingInLiveCount = Math.max(0, historicalSuccessEmails.length - historicalHitCount);
  const unexpectedLiveCount = liveEmails.filter((email) => !historicalSuccessEmails.includes(email)).length;
  const guardedActions = Array.isArray(controlSnapshot?.guarded_actions) ? controlSnapshot?.guarded_actions : [];

  return (
    <div className="workspace-grid single-column cpa-page">
      {controlSnapshot ? (
        <PanelCard title="CPA 渠道状态" kicker="source-owned / historical acceptance + inventory" meta="历史成功快照、当前 inventory 与主仓 CPA kernel 统一只读摘要">
          <div className="chip-row cpa-summary-grid">
            <span className="summary-chip">{`${Number(controlSummary.historical_success_total || 0)} 个历史成功账号`}</span>
            <span className="summary-chip">{`active 验收快照 ${Number(controlSummary.active_target_authfiles || 0)}`}</span>
            <span className="summary-chip">{`inventory ${Number(controlSummary.inventory_total || 0)}`}</span>
            <span className="summary-chip">{`retry_candidate ${Number(controlSummary.retry_candidate_total || 0)}`}</span>
            <span className="summary-chip">{`blocked_about_you ${Number(controlSummary.blocked_about_you_total || 0)}`}</span>
            <span className="summary-chip">{`no_retry_deactivated ${Number(controlSummary.no_retry_deactivated_total || 0)}`}</span>
            <span className="summary-chip">{`new_unmounted ${Number(controlSummary.new_unmounted_total || 0)}`}</span>
            <span className="summary-chip">{`kernel accounts ${Number(controlSummary.latest_kernel_accounts_total || 0)}`}</span>
            {controlSnapshot.latest_kernel_run_id ? <span className="summary-chip">{`kernel run ${controlSnapshot.latest_kernel_run_id}`}</span> : null}
          </div>
        </PanelCard>
      ) : null}
      {controlSnapshot ? (
        <PanelCard title="CPA 对账视图" kicker="source-owned / reconciliation" meta="先看历史成功集与 inventory 的偏移，再决定是否触发 guarded live workflow。">
          <div className="chip-row cpa-summary-grid">
            <span className="summary-chip">{`历史成功在 usable_active ${Number(controlSummary.historical_success_in_usable_active_total || 0)}`}</span>
            <span className="summary-chip">{`历史成功非 active ${Number(controlSummary.historical_success_non_active_total || 0)}`}</span>
            <span className="summary-chip">{`历史成功缺口 ${Number(controlSummary.historical_success_missing_from_inventory_total || 0)}`}</span>
            {rows.length ? <span className="summary-chip">{`live 已连接 ${liveEmails.length}`}</span> : null}
            {rows.length ? <span className="summary-chip">{`历史成功命中 ${historicalHitCount}`}</span> : null}
            {rows.length ? <span className="summary-chip">{`历史成功缺口 ${historicalMissingInLiveCount}`}</span> : null}
            {rows.length ? <span className="summary-chip">{`unexpected live ${unexpectedLiveCount}`}</span> : null}
          </div>
        </PanelCard>
      ) : null}
      {guardedActions.length ? (
        <PanelCard title="Guarded Actions" kicker="live-guard-only / command suggestions" meta="这些动作来自 handoff 与 source-owned CPA control snapshot，只给出建议命令，不在页面中直接执行。">
          {guardedActions.map((action) => (
            <div className="empty-block" key={String(action.id || action.label || Math.random())}>
              <strong>{String(action.label || action.id || 'action')}</strong>
              <div>{String(action.description || '')}</div>
              <div>{`风险级别：${String(action.risk_class || 'LIVE_GUARD_ONLY')}`}</div>
              <pre className="json-block">{String(action.command || '')}</pre>
            </div>
          ))}
        </PanelCard>
      ) : null}
      <PanelCard title="CPA 管理" kicker="auth-files / quota / delete-guard" meta="认证文件、额度查询与误删保护">
        <div className="cpa-config-row">
          <label className="cpa-field">
            <span>服务器地址</span>
            <input className="search-input" placeholder="请输入服务器地址" value={server} onChange={(event) => setServer(event.target.value)} />
          </label>
          <label className="cpa-field">
            <span>管理员密码</span>
            <input className="search-input" placeholder="请输入管理员密码" type="password" value={password} onChange={(event) => setPassword(event.target.value)} />
          </label>
          <div className="cpa-field cpa-action-field">
            <ActionButton onClick={handleConnect} disabled={connecting}>{connecting ? '连接中...' : rows.length ? '重新连接' : '开始连接'}</ActionButton>
          </div>
        </div>

        {status ? <div className={`cpa-status-bar ${status.tone}`.trim()}>{status.text}</div> : null}

        {rows.length ? (
          <>
            <div className="cpa-toolbar-row">
              <input className="search-input" placeholder="搜索文件名..." value={search} onChange={(event) => { setSearch(event.target.value); setPage(1); }} />
              <label className="cpa-inline-select">
                <span>套餐</span>
                <select className="cpa-select" value={planFilter} onChange={(event) => { setPlanFilter(event.target.value); setPage(1); }}>
                  <option value="">全部套餐</option>
                  <option value="free">Free</option>
                  <option value="plus">Plus</option>
                  <option value="team">Team</option>
                </select>
              </label>
              <label className="cpa-inline-select">
                <span>启用状态</span>
                <select className="cpa-select" value={statusFilter} onChange={(event) => { setStatusFilter(event.target.value); setPage(1); }}>
                  <option value="">全部状态</option>
                  <option value="enabled">已启用</option>
                  <option value="disabled">已禁用</option>
                </select>
              </label>
              <label className="cpa-inline-select">
                <span>查询结果</span>
                <select className="cpa-select" value={queryFilter} onChange={(event) => { setQueryFilter(event.target.value); setPage(1); }}>
                  <option value="">全部查询结果</option>
                  <option value="success">查询成功</option>
                  <option value="failed">查询失败</option>
                </select>
              </label>
              <label className="cpa-inline-select">
                <span>分页</span>
                <select className="cpa-select" value={String(currentPage)} onChange={(event) => setPage(Number(event.target.value) || 1)}>
                  {Array.from({ length: pageCount }, (_, index) => (
                    <option key={index + 1} value={index + 1}>{`第 ${index + 1} 页`}</option>
                  ))}
                </select>
              </label>
            </div>

            <div className="cpa-toolbar-row">
              <ActionButton onClick={handleQueryQuota} disabled={querying}>{querying ? '查询中...' : '查询额度'}</ActionButton>
              <ActionButton onClick={handleEnablePositiveQuota}>启用有额度账号</ActionButton>
              <ActionButton onClick={handleDisableNoQuota}>关闭无额度账号</ActionButton>
              <ActionButton onClick={() => void handleBatchDelete('401', true)}>删除401账号</ActionButton>
              <ActionButton onClick={() => void handleBatchDelete('402', true)}>删除402账号</ActionButton>
              <label className="cpa-inline-select">
                <span>删除错误桶</span>
                <select
                  aria-label="删除错误桶"
                  className="cpa-select"
                  value={deleteBucket}
                  onChange={(event) => setDeleteBucket((event.target.value || '') as CpaErrorBucket | '')}
                >
                  <option value="">选择错误桶</option>
                  {CPA_DELETE_BUCKET_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>{option.label}</option>
                  ))}
                </select>
              </label>
              <ActionButton disabled={!bucketDeleteEnabled} onClick={() => deleteBucket && void handleBatchDelete(deleteBucket)}>按错误桶删除</ActionButton>
            </div>

            <div className="chip-row cpa-summary-grid">
              <span className="summary-chip">共 {displayRows.length} 个 Codex 文件</span>
              <span className="summary-chip">成功 {displayRows.filter((row) => row.status === 'ok').length}</span>
              <span className="summary-chip">失败 {displayRows.filter((row) => row.status === 'error').length}</span>
              <span className="summary-chip">已禁用 {displayRows.filter((row) => row.disabled).length}</span>
              <span className="summary-chip">可删候选 {eligibleDeleteCount}</span>
              {deleteBucket ? <span className="summary-chip">当前错误桶候选 {deleteCandidates.length}</span> : null}
            </div>
            <div className="chip-row cpa-summary-grid" data-testid="cpa-error-buckets">
              {bucketSummaryEntries.length ? bucketSummaryEntries.map(([bucket, count]) => (
                <span className="summary-chip" key={bucket}>{`${bucket}：${count}`}</span>
              )) : <span className="summary-chip">暂无错误桶</span>}
            </div>

            <div className="table-wrap">
              <table className="dense-table">
                <thead>
                  <tr>
                    <th>文件名</th>
                    <th>套餐</th>
                    <th>查询状态</th>
                    <th>5小时额度</th>
                    <th>周限额剩余</th>
                    <th>重置时间</th>
                    <th>最近错误类型</th>
                    <th>连续同错</th>
                    <th>累计错误</th>
                    <th>删除资格</th>
                    <th>操作</th>
                  </tr>
                </thead>
                <tbody>
                  {pageRows.map((row) => {
                    const highlightDelete = deleteBucket && row.lastErrorType === deleteBucket && row.eligibleForDelete;
                    return (
                      <tr className={highlightDelete ? 'cpa-row-highlight' : ''} key={row.name}>
                        <td>{trimJsonSuffix(row.name)}</td>
                        <td><Badge value={planTone(row.planType)}>{row.planType ? row.planType.toUpperCase() : '未知'}</Badge></td>
                        <td>
                          {row.status === 'ok' ? <Badge value="positive">成功</Badge> : row.status === 'error' ? <Badge value="negative">失败</Badge> : row.status === 'loading' ? <Badge value="warning">查询中</Badge> : <Badge value="neutral">待查询</Badge>}
                        </td>
                        <td>{row.shortRemaining !== null ? `${Math.round(row.shortRemaining)}%` : '—'}</td>
                        <td>{row.remaining !== null ? `${Math.round(row.remaining)}%` : '—'}</td>
                        <td>{row.resetLabel || '—'}</td>
                        <td>{row.lastErrorType || row.errorBucket || '—'}{row.error && row.status === 'error' ? <div className="cpa-error-note">{row.error}</div> : null}</td>
                        <td>{row.sameErrorStreak}</td>
                        <td>{row.totalErrorCount}</td>
                        <td>{row.eligibleForDelete ? <Badge value="warning">可删</Badge> : <span className="cpa-metric-muted">—</span>}</td>
                        <td>
                          <div className="cpa-row-actions">
                            <ActionButton onClick={() => void handleToggleAuth(row)}>{row.disabled ? '启用' : '禁用'}</ActionButton>
                            <ActionButton onClick={() => void handleDeleteAuth(row)}>删除</ActionButton>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </>
        ) : (
          <div className="empty-block">请填写服务器信息后点击「开始连接」</div>
        )}
      </PanelCard>
    </div>
  );
}
