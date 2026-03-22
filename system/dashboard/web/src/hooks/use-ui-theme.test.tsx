import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { THEME_STORAGE_KEY, useUiTheme } from './use-ui-theme';

function stubMatchMedia(initialLight = false) {
  let prefersLight = initialLight;
  const listeners = new Set<(event: MediaQueryListEvent) => void>();
  const mediaQuery = '(prefers-color-scheme: light)';
  const media = {
    get matches() {
      return prefersLight;
    },
    media: mediaQuery,
    onchange: null,
    addEventListener: (_event: string, listener: (event: MediaQueryListEvent) => void) => listeners.add(listener),
    removeEventListener: (_event: string, listener: (event: MediaQueryListEvent) => void) => listeners.delete(listener),
    addListener: (listener: (event: MediaQueryListEvent) => void) => listeners.add(listener),
    removeListener: (listener: (event: MediaQueryListEvent) => void) => listeners.delete(listener),
    dispatchEvent: () => true,
  };

  vi.stubGlobal('matchMedia', vi.fn().mockImplementation(() => media));

  return {
    setLight(next: boolean) {
      prefersLight = next;
      const event = { matches: prefersLight, media: mediaQuery } as MediaQueryListEvent;
      listeners.forEach((listener) => listener(event));
    },
  };
}

describe('useUiTheme', () => {
  beforeEach(() => {
    window.localStorage.clear();
    delete document.documentElement.dataset.theme;
    document.documentElement.style.colorScheme = '';
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it('defaults to system preference and resolves from matchMedia', async () => {
    stubMatchMedia(true);
    const { result } = renderHook(() => useUiTheme(undefined));

    await waitFor(() => {
      expect(result.current.preference).toBe('system');
      expect(result.current.resolvedTheme).toBe('light');
      expect(document.documentElement.dataset.theme).toBe('light');
      expect(document.documentElement.style.colorScheme).toBe('light');
    });
  });

  it('persists explicit preference and updates the document theme', async () => {
    stubMatchMedia(true);
    const { result } = renderHook(() => useUiTheme(undefined));

    await act(async () => {
      result.current.setPreference('dark');
    });

    await waitFor(() => {
      expect(result.current.preference).toBe('dark');
      expect(result.current.resolvedTheme).toBe('dark');
      expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe('dark');
      expect(document.documentElement.dataset.theme).toBe('dark');
      expect(document.documentElement.style.colorScheme).toBe('dark');
    });
  });

  it('lets url override beat stored preference while preserving the stored value', async () => {
    window.localStorage.setItem(THEME_STORAGE_KEY, 'dark');
    stubMatchMedia(false);
    const { result } = renderHook(() => useUiTheme('light'));

    await waitFor(() => {
      expect(result.current.preference).toBe('light');
      expect(result.current.resolvedTheme).toBe('light');
    });

    await act(async () => {
      result.current.setPreference('dark');
    });

    await waitFor(() => {
      expect(result.current.preference).toBe('light');
      expect(result.current.resolvedTheme).toBe('light');
      expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe('dark');
    });
  });

  it('reacts to system theme changes while preference stays on system', async () => {
    const media = stubMatchMedia(false);
    const { result } = renderHook(() => useUiTheme(undefined));

    await waitFor(() => {
      expect(result.current.preference).toBe('system');
      expect(result.current.resolvedTheme).toBe('dark');
    });

    act(() => {
      media.setLight(true);
    });

    await waitFor(() => {
      expect(result.current.resolvedTheme).toBe('light');
      expect(document.documentElement.dataset.theme).toBe('light');
    });
  });
});
