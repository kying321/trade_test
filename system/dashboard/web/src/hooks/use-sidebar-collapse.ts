import { useCallback, useEffect, useMemo, useState } from 'react';

export const SIDEBAR_COLLAPSE_STORAGE_KEY = 'fenlie-sidebar-collapsed';
const SIDEBAR_COLLAPSE_QUERY = '(max-width: 1380px)';

function parseStoredCollapse(raw: string | null | undefined): boolean {
  return raw === 'true';
}

function loadStoredCollapse(): boolean {
  if (typeof window === 'undefined') return false;
  return parseStoredCollapse(window.localStorage.getItem(SIDEBAR_COLLAPSE_STORAGE_KEY));
}

function resolveCompactLayout(): boolean {
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return false;
  return window.matchMedia(SIDEBAR_COLLAPSE_QUERY).matches;
}

export function useSidebarCollapse() {
  const [storedCollapsed, setStoredCollapsed] = useState<boolean>(() => loadStoredCollapse());
  const [compactLayout, setCompactLayout] = useState<boolean>(() => resolveCompactLayout());
  const [compactCollapsed, setCompactCollapsed] = useState<boolean>(() => resolveCompactLayout());

  useEffect(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return undefined;
    const media = window.matchMedia(SIDEBAR_COLLAPSE_QUERY);
    const handleChange = () => {
      const matches = media.matches;
      setCompactLayout(matches);
      if (matches) setCompactCollapsed(true);
    };
    handleChange();
    if (typeof media.addEventListener === 'function') {
      media.addEventListener('change', handleChange);
      return () => media.removeEventListener('change', handleChange);
    }
    media.addListener(handleChange);
    return () => media.removeListener(handleChange);
  }, []);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    window.localStorage.setItem(SIDEBAR_COLLAPSE_STORAGE_KEY, storedCollapsed ? 'true' : 'false');
  }, [storedCollapsed]);

  const setCollapsed = useCallback((next: boolean) => {
    if (compactLayout) {
      setCompactCollapsed(next);
      return;
    }
    setStoredCollapsed(next);
  }, [compactLayout]);

  const toggleCollapsed = useCallback(() => {
    if (compactLayout) {
      setCompactCollapsed((current) => !current);
      return;
    }
    setStoredCollapsed((current) => !current);
  }, [compactLayout]);

  const canCollapse = true;
  const collapsed = compactLayout ? compactCollapsed : storedCollapsed;

  return useMemo(() => ({
    collapsed,
    canCollapse,
    setCollapsed,
    toggleCollapsed,
  }), [collapsed, canCollapse, setCollapsed, toggleCollapsed]);
}
