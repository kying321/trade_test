import type { AnchorHTMLAttributes, ButtonHTMLAttributes, ReactNode } from 'react';
import { NavLink, type NavLinkProps, type NavLinkRenderProps, useMatch, useResolvedPath } from 'react-router-dom';

function joinControlClass(base: string, extra?: string) {
  return [base, extra].filter(Boolean).join(' ');
}

function normalizeNavRenderState(
  state: NavLinkRenderProps,
  forceActive?: boolean,
): NavLinkRenderProps {
  if (!forceActive || state.isActive) return state;
  return {
    ...state,
    isActive: true,
  };
}

function normalizeNavClassName(
  base: string,
  className?: NavLinkProps['className'],
  forceActive?: boolean,
): NavLinkProps['className'] {
  if (typeof className === 'function') {
    return (state) => {
      const effectiveState = normalizeNavRenderState(state, forceActive);
      return (
      joinControlClass(
        joinControlClass(base, effectiveState.isActive ? 'active' : undefined),
        className(effectiveState),
      ));
    };
  }

  return (state) =>
    joinControlClass(
      joinControlClass(base, state.isActive || forceActive ? 'active' : undefined),
      className,
    );
}

function normalizeNavStyle(
  style: NavLinkProps['style'],
  forceActive?: boolean,
): NavLinkProps['style'] {
  if (typeof style !== 'function') return style;
  return (state) => style(normalizeNavRenderState(state, forceActive));
}

function normalizeNavChildren(
  children: NavLinkProps['children'],
  forceActive?: boolean,
): NavLinkProps['children'] {
  if (typeof children !== 'function') return children;
  return (state) => children(normalizeNavRenderState(state, forceActive));
}

export function DomainTab({ className, active, children, style, ...props }: NavLinkProps & { active?: boolean }) {
  const resolvedPath = useResolvedPath(props.to);
  const routeActive = Boolean(useMatch({
    path: resolvedPath.pathname,
    end: props.end ?? false,
    caseSensitive: props.caseSensitive,
  }));
  const selected = routeActive || Boolean(active);

  return (
    <NavLink
      {...props}
      role="tab"
      aria-selected={selected}
      className={normalizeNavClassName('control-domain-tab', className, active)}
      style={normalizeNavStyle(style, active)}
    >
      {normalizeNavChildren(children, active)}
    </NavLink>
  );
}

type ActionLinkProps = Omit<NavLinkProps, 'type'> &
  Omit<AnchorHTMLAttributes<HTMLAnchorElement>, keyof NavLinkProps | 'type'>;

export function ActionLink({ className, ...props }: ActionLinkProps) {
  return <NavLink {...props} className={normalizeNavClassName('control-action-link', className)} />;
}

export function SegmentedOption({
  className,
  active,
  type,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { active?: boolean }) {
  return (
    <button
      type={type ?? 'button'}
      {...props}
      aria-pressed={Boolean(active)}
      data-active={active ? 'true' : 'false'}
      className={joinControlClass('control-segment-option', className)}
    />
  );
}

export function ActionButton({ className, type, ...props }: ButtonHTMLAttributes<HTMLButtonElement>) {
  return <button type={type ?? 'button'} {...props} className={joinControlClass('control-action-button', className)} />;
}

export function FilterChip({
  className,
  active,
  type,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { active?: boolean }) {
  return (
    <button
      type={type ?? 'button'}
      {...props}
      aria-pressed={Boolean(active)}
      data-active={active ? 'true' : 'false'}
      className={joinControlClass('control-filter-chip', className)}
    />
  );
}

export function EntityRowButton({
  title,
  subtitle,
  active,
  className,
  type,
  children,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  title: string;
  subtitle?: string;
  active?: boolean;
  children?: ReactNode;
}) {
  return (
    <button
      type={type ?? 'button'}
      {...props}
      aria-pressed={Boolean(active)}
      data-active={active ? 'true' : 'false'}
      className={joinControlClass('control-entity-row', className)}
    >
      <span className="control-entity-row-title">{title}</span>
      {subtitle ? <small className="control-entity-row-subtitle">{subtitle}</small> : null}
      {children}
    </button>
  );
}
