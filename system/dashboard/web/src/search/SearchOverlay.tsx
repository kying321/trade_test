import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ActionButton } from '../components/control-primitives';
import type { TerminalReadModel } from '../types/contracts';
import { buildSearchCatalog, searchCatalog } from './catalog';
import { SearchResultsContent } from './SearchResults';
import type { SearchResultEntry, SearchScope } from './types';

function canCaptureShortcut(event: KeyboardEvent): boolean {
  const target = event.target as HTMLElement | null;
  if (!target) return true;
  const tag = typeof target.tagName === 'string' ? target.tagName.toLowerCase() : '';
  if (tag === 'input' || tag === 'textarea' || tag === 'select') return false;
  if (target.isContentEditable) return false;
  return true;
}

export function GlobalSearchOverlay({
  isOpen,
  model,
  query,
  scope,
  onClose,
  onQueryChange,
  onScopeChange,
}: {
  isOpen: boolean;
  model?: TerminalReadModel | null;
  query: string;
  scope: SearchScope;
  onClose: () => void;
  onQueryChange: (next: string) => void;
  onScopeChange: (next: SearchScope) => void;
}) {
  const navigate = useNavigate();
  const results = useMemo(() => searchCatalog(buildSearchCatalog(model || undefined), query, scope).slice(0, 12), [model, query, scope]);
  const [activeIndex, setActiveIndex] = useState(0);

  useEffect(() => {
    if (!isOpen) return;
    setActiveIndex(0);
  }, [isOpen, query, scope]);

  useEffect(() => {
    if (!isOpen) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (!canCaptureShortcut(event) && event.key !== 'Escape') return;
      if (event.key === 'Escape') {
        event.preventDefault();
        onClose();
        return;
      }
      if (!results.length) return;
      if (event.key === 'ArrowDown') {
        event.preventDefault();
        setActiveIndex((prev) => (prev + 1) % results.length);
        return;
      }
      if (event.key === 'ArrowUp') {
        event.preventDefault();
        setActiveIndex((prev) => (prev - 1 + results.length) % results.length);
        return;
      }
      if (event.key === 'Enter') {
        event.preventDefault();
        const next = results[activeIndex];
        if (next) {
          navigate(next.destination);
          onClose();
        }
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [activeIndex, isOpen, navigate, onClose, results]);

  if (!isOpen) return null;

  const activeId = results[activeIndex]?.id;
  const handlePick = (entry: SearchResultEntry) => {
    navigate(entry.destination);
    onClose();
  };

  return (
    <div className="search-overlay-backdrop" role="dialog" aria-modal="true" aria-label="全局搜索">
      <div className="search-overlay-card">
        <div className="search-overlay-head">
          <div>
            <p className="panel-kicker">Cmd/Ctrl+K</p>
            <strong>全局关键词搜索</strong>
          </div>
          <ActionButton className="chip-button" onClick={onClose}>关闭</ActionButton>
        </div>
        <SearchResultsContent
          model={model}
          query={query}
          scope={scope}
          activeId={activeId}
          onPick={handlePick}
          onQueryChange={onQueryChange}
          onScopeChange={onScopeChange}
          showViewAll
        />
      </div>
    </div>
  );
}
