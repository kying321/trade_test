import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { SIDEBAR_COLLAPSE_STORAGE_KEY, useSidebarCollapse } from './use-sidebar-collapse';

function stubMatchMedia(matches = false) {
  let current = matches;
  const listeners = new Set<(event: MediaQueryListEvent) => void>();
  const mediaQuery = '(max-width: 1380px)';
  const media = {
    get matches() {
      return current;
    },
    media: mediaQuery,
    onchange: null,
    addEventListener: (_event: string, listener: (event: MediaQueryListEvent) => void) => listeners.add(listener),
    removeEventListener: (_event: string, listener: (event: MediaQueryListEvent) => void) => listeners.delete(listener),
    addListener: (listener: (event: MediaQueryListEvent) => void) => listeners.add(listener),
    removeListener: (listener: (event: MediaQueryListEvent) => void) => listeners.delete(listener),
    dispatchEvent: () => true,
  };

  vi.stubGlobal('matchMedia', vi.fn().mockImplementation((query: string) => {
    if (query === mediaQuery) return media;
    return {
      ...media,
      media: query,
      matches: false,
    };
  }));

  return {
    setCompact(next: boolean) {
      current = next;
      const event = { matches: current, media: mediaQuery } as MediaQueryListEvent;
      listeners.forEach((listener) => listener(event));
    },
  };
}

describe('useSidebarCollapse', () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it('defaults to expanded desktop state and persists explicit collapse intent', async () => {
    stubMatchMedia(false);
    const { result } = renderHook(() => useSidebarCollapse());

    await waitFor(() => {
      expect(result.current.collapsed).toBe(false);
      expect(result.current.canCollapse).toBe(true);
    });

    await act(async () => {
      result.current.setCollapsed(true);
    });

    await waitFor(() => {
      expect(result.current.collapsed).toBe(true);
      expect(window.localStorage.getItem(SIDEBAR_COLLAPSE_STORAGE_KEY)).toBe('true');
    });
  });

  it('forces expanded state on single-column breakpoint while keeping remembered preference', async () => {
    const media = stubMatchMedia(false);
    window.localStorage.setItem(SIDEBAR_COLLAPSE_STORAGE_KEY, 'true');
    const { result } = renderHook(() => useSidebarCollapse());

    await waitFor(() => {
      expect(result.current.collapsed).toBe(true);
      expect(result.current.canCollapse).toBe(true);
    });

    act(() => {
      media.setCompact(true);
    });

    await waitFor(() => {
      expect(result.current.collapsed).toBe(false);
      expect(result.current.canCollapse).toBe(false);
      expect(window.localStorage.getItem(SIDEBAR_COLLAPSE_STORAGE_KEY)).toBe('true');
    });
  });
});
