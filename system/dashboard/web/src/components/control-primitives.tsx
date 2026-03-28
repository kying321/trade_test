import type { AnchorHTMLAttributes, ButtonHTMLAttributes, ReactNode } from 'react';
import { Link, NavLink, type NavLinkProps, type NavLinkRenderProps, useMatch, useResolvedPath } from 'react-router-dom';

function joinControlClass(base: string, extra?: string) {
  return [base, extra].filter(Boolean).join(' ');
}

function normalizeNavClassName(
  base: string,
  className?: NavLinkProps['className'],
  forceActive?: boolean,
): NavLinkProps['className'] {
  if (typeof className === 'function') {
    return (state) => {
      const effectiveState = {
        ...state,
        isActive: state.isActive || Boolean(forceActive),
      };
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

function resolveNavClassName(
  base: string,
  className: NavLinkProps['className'] | undefined,
  renderState: NavLinkRenderProps,
) {
  if (typeof className === 'function') {
    return joinControlClass(
      joinControlClass(base, renderState.isActive ? 'active' : undefined),
      className(renderState),
    );
  }

  return joinControlClass(
    joinControlClass(base, renderState.isActive ? 'active' : undefined),
    className,
  );
}

function resolveNavStyle(
  style: NavLinkProps['style'] | undefined,
  renderState: NavLinkRenderProps,
) {
  if (typeof style === 'function') return style(renderState);
  return style;
}

export function DomainTab({
  className,
  active,
  children,
  style,
  to,
  end,
  caseSensitive,
  ...props
}: NavLinkProps & { active?: boolean }) {
  const resolvedPath = useResolvedPath(to);
  const routeMatch = useMatch({
    path: resolvedPath.pathname,
    end,
    caseSensitive,
  });
  const renderState: NavLinkRenderProps = {
    isActive: Boolean(active || routeMatch),
    isPending: false,
    isTransitioning: false,
  };

  return (
    <Link
      {...props}
      to={to}
      aria-current={renderState.isActive ? 'page' : undefined}
      className={resolveNavClassName('control-domain-tab', className, renderState)}
      style={resolveNavStyle(style, renderState)}
    >
      {typeof children === 'function' ? children(renderState) : children}
    </Link>
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
