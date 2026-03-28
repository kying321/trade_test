import { ActionLink, EntityRowButton, FilterChip } from '../components/control-primitives';
import { Badge, ClampText, PanelCard } from '../components/ui-kit';
import type { TerminalReadModel } from '../types/contracts';
import { groupSearchResults, buildSearchCatalog, searchCatalog } from './catalog';
import { buildSearchLink } from './links';
import type { SearchResultEntry, SearchScope } from './types';

const CATEGORY_LABELS: Record<'module' | 'route' | 'artifact', string> = {
  module: '模块',
  route: '路由',
  artifact: '工件',
};

function resultHint(result: SearchResultEntry): string {
  return result.matchedFields.map((field) => {
    if (field === 'title') return '标题命中';
    if (field === 'subtitle') return '说明命中';
    if (field === 'destination') return '路径命中';
    return '关键词命中';
  }).join(' / ');
}

export function SearchScopeChips({
  scope,
  onChange,
}: {
  scope: SearchScope;
  onChange: (next: SearchScope) => void;
}) {
  const scopes: SearchScope[] = ['all', 'module', 'route', 'artifact'];
  return (
    <div className="chip-row search-scope-row">
      {scopes.map((item) => (
        <FilterChip
          key={item}
          className="chip-button"
          active={scope === item}
          onClick={() => onChange(item)}
        >
          {item === 'all' ? '全部' : CATEGORY_LABELS[item]}
        </FilterChip>
      ))}
    </div>
  );
}

function SearchResultGroup({
  title,
  results,
  activeId,
  onPick,
}: {
  title: string;
  results: SearchResultEntry[];
  activeId?: string;
  onPick: (entry: SearchResultEntry) => void;
}) {
  if (!results.length) return null;
  return (
    <section className="search-result-group">
      <div className="search-result-group-head">
        <strong>{title}</strong>
        <span>{results.length}</span>
      </div>
      <div className="stack-list search-result-list">
        {results.map((result) => (
          <EntityRowButton
            key={result.id}
            className={`search-result-button ${activeId === result.id ? 'active' : ''}`.trim()}
            title={<ClampText raw={result.title} expandable={false}>{result.title}</ClampText>}
            subtitle={<ClampText raw={result.subtitle || result.destination} expandable={false}>{result.subtitle || result.destination}</ClampText>}
            active={activeId === result.id}
            onClick={() => onPick(result)}
          >
            <span className="nav-link-copy search-result-copy">
              <small className="search-result-hint">{resultHint(result)}</small>
            </span>
            <Badge value={result.category}>{CATEGORY_LABELS[result.category]}</Badge>
          </EntityRowButton>
        ))}
      </div>
    </section>
  );
}

export function SearchResultsContent({
  model,
  query,
  scope,
  activeId,
  onPick,
  onQueryChange,
  onScopeChange,
  showViewAll = false,
}: {
  model?: TerminalReadModel | null;
  query: string;
  scope: SearchScope;
  activeId?: string;
  onPick: (entry: SearchResultEntry) => void;
  onQueryChange: (next: string) => void;
  onScopeChange: (next: SearchScope) => void;
  showViewAll?: boolean;
}) {
  const catalog = buildSearchCatalog(model || undefined);
  const results = searchCatalog(catalog, query, scope);
  const grouped = groupSearchResults(showViewAll ? results.slice(0, 12) : results);
  const viewAllHref = buildSearchLink(query, scope);
  return (
    <>
      <SearchScopeChips scope={scope} onChange={onScopeChange} />
      <input
        autoFocus
        className="search-input global-search-input"
        value={query}
        onChange={(event) => onQueryChange(event.target.value)}
        placeholder="输入模块、路由、工件、source head 或关键字"
      />
      {query.trim() ? (
        <div className="search-result-groups">
          <SearchResultGroup title="模块" results={grouped.module} activeId={activeId} onPick={onPick} />
          <SearchResultGroup title="路由" results={grouped.route} activeId={activeId} onPick={onPick} />
          <SearchResultGroup title="工件" results={grouped.artifact} activeId={activeId} onPick={onPick} />
        </div>
      ) : (
        <div className="empty-block">输入关键词后显示分类结果。</div>
      )}
      {showViewAll && query.trim() ? (
        <div className="button-row search-actions">
          <ActionLink className="button" to={viewAllHref}>查看全部结果</ActionLink>
        </div>
      ) : null}
      {query.trim() && !results.length ? (
        <div className="empty-block">未找到匹配结果，请尝试中文标题、路由名、artifact id 或路径片段。</div>
      ) : null}
    </>
  );
}

export function SearchPageContent({
  model,
  query,
  scope,
  onPick,
  onQueryChange,
  onScopeChange,
}: {
  model?: TerminalReadModel | null;
  query: string;
  scope: SearchScope;
  onPick: (entry: SearchResultEntry) => void;
  onQueryChange: (next: string) => void;
  onScopeChange: (next: SearchScope) => void;
}) {
  return (
    <div className="workspace-grid search-page-grid">
      <PanelCard
        title="全局关键词搜索"
        kicker="global search / module + route + artifact"
        meta="跨页面、跨模块、跨工件的本地索引搜索，不接后端服务。"
      >
        <SearchResultsContent
          model={model}
          query={query}
          scope={scope}
          onPick={onPick}
          onQueryChange={onQueryChange}
          onScopeChange={onScopeChange}
        />
      </PanelCard>
    </div>
  );
}
