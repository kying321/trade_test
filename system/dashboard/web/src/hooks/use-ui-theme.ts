import { useCallback, useEffect, useMemo, useState } from 'react';
import type { ThemePreference, ResolvedTheme } from '../app-shell/GlobalTopbar';

export const THEME_STORAGE_KEY = 'fenlie-theme-preference';

function parseThemePreference(raw: string | null | undefined): ThemePreference | null {
  if (raw === 'system' || raw === 'dark' || raw === 'light') return raw;
  return null;
}

function resolveSystemTheme(): ResolvedTheme {
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return 'dark';
  return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
}

function loadStoredTheme(): ThemePreference {
  if (typeof window === 'undefined') return 'system';
  return parseThemePreference(window.localStorage.getItem(THEME_STORAGE_KEY)) || 'system';
}

export function useUiTheme(urlThemeOverride: string | null | undefined) {
  const [storedPreference, setStoredPreference] = useState<ThemePreference>(() => loadStoredTheme());
  const [systemTheme, setSystemTheme] = useState<ResolvedTheme>(() => resolveSystemTheme());

  const urlPreference = parseThemePreference(urlThemeOverride);
  const preference = urlPreference || storedPreference;
  const resolvedTheme: ResolvedTheme = preference === 'system' ? systemTheme : preference;

  useEffect(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return undefined;
    const media = window.matchMedia('(prefers-color-scheme: light)');
    const handleChange = () => setSystemTheme(media.matches ? 'light' : 'dark');
    handleChange();
    if (typeof media.addEventListener === 'function') {
      media.addEventListener('change', handleChange);
      return () => media.removeEventListener('change', handleChange);
    }
    media.addListener(handleChange);
    return () => media.removeListener(handleChange);
  }, []);

  useEffect(() => {
    if (typeof document === 'undefined') return;
    document.documentElement.dataset.theme = resolvedTheme;
    document.documentElement.style.colorScheme = resolvedTheme;
  }, [resolvedTheme]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    window.localStorage.setItem(THEME_STORAGE_KEY, storedPreference);
  }, [storedPreference]);

  const setPreference = useCallback((next: ThemePreference) => {
    setStoredPreference(next);
  }, []);

  return useMemo(() => ({
    preference,
    resolvedTheme,
    setPreference,
  }), [preference, resolvedTheme, setPreference]);
}
