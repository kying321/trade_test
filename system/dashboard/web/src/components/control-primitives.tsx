import type { AnchorHTMLAttributes, ButtonHTMLAttributes } from 'react';
import { NavLink, type NavLinkProps } from 'react-router-dom';

function joinControlClass(base: string, extra?: string) {
  return [base, extra].filter(Boolean).join(' ');
}

function normalizeNavClassName(
  base: string,
  className?: NavLinkProps['className'],
  forceActive?: boolean,
): NavLinkProps['className'] {
  if (typeof className === 'function') {
    return (state) =>
      joinControlClass(
        joinControlClass(base, state.isActive || forceActive ? 'active' : undefined),
        className(state),
      );
  }

  return (state) =>
    joinControlClass(
      joinControlClass(base, state.isActive || forceActive ? 'active' : undefined),
      className,
    );
}

export function DomainTab({ className, active, ...props }: NavLinkProps & { active?: boolean }) {
  return <NavLink {...props} className={normalizeNavClassName('control-domain-tab', className, active)} />;
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

export function EntityRowButton({ title, subtitle, active, onClick }: {
  title: string;
  subtitle?: string;
  active?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      aria-pressed={Boolean(active)}
      data-active={active ? 'true' : 'false'}
      className="control-entity-row"
      onClick={onClick}
    >
      <span>{title}</span>
      {subtitle ? <small>{subtitle}</small> : null}
    </button>
  );
}
