import { useEffect, useMemo, useState } from 'react';

export type ShellTier = 'xl' | 'l' | 'm' | 's';

export type ShellBreakpointState = {
  width: number;
  tier: ShellTier;
};

export const SHELL_BREAKPOINTS = {
  sMax: 959,
  mMax: 1279,
  lMax: 1599,
} as const;

function resolveWindowWidth(): number {
  if (typeof window === 'undefined') return SHELL_BREAKPOINTS.mMax + 1;
  return window.innerWidth;
}

export function resolveShellTier(width: number): ShellTier {
  if (width <= SHELL_BREAKPOINTS.sMax) return 's';
  if (width <= SHELL_BREAKPOINTS.mMax) return 'm';
  if (width <= SHELL_BREAKPOINTS.lMax) return 'l';
  return 'xl';
}

export function useShellBreakpoint(): ShellBreakpointState {
  const [width, setWidth] = useState<number>(() => resolveWindowWidth());

  useEffect(() => {
    if (typeof window === 'undefined') return undefined;
    const handleResize = () => setWidth(window.innerWidth);
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return useMemo(() => ({
    width,
    tier: resolveShellTier(width),
  }), [width]);
}
