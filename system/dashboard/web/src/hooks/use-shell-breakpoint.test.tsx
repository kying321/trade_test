import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import { resolveShellTier, useShellBreakpoint } from './use-shell-breakpoint';

function mockWindowWidth(width: number) {
  Object.defineProperty(window, 'innerWidth', {
    configurable: true,
    writable: true,
    value: width,
  });
}

describe('useShellBreakpoint', () => {
  afterEach(() => {
    mockWindowWidth(1024);
  });

  it('returns medium tier between 960 and 1279 width', async () => {
    mockWindowWidth(1100);
    const { result } = renderHook(() => useShellBreakpoint());

    await waitFor(() => {
      expect(result.current.tier).toBe('m');
      expect(result.current.width).toBe(1100);
    });
  });

  it('updates shell tier when window size changes', async () => {
    mockWindowWidth(1420);
    const { result } = renderHook(() => useShellBreakpoint());

    await waitFor(() => {
      expect(result.current.tier).toBe('l');
    });

    act(() => {
      mockWindowWidth(820);
      window.dispatchEvent(new Event('resize'));
    });

    await waitFor(() => {
      expect(result.current.tier).toBe('s');
      expect(result.current.width).toBe(820);
    });
  });

  it('maps tier boundaries deterministically', () => {
    expect(resolveShellTier(960)).toBe('m');
    expect(resolveShellTier(1279)).toBe('m');
    expect(resolveShellTier(1280)).toBe('l');
    expect(resolveShellTier(1600)).toBe('xl');
  });
});
