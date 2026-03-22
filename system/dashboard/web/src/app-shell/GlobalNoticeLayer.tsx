type GlobalNoticeLayerProps = {
  warnings?: string[];
  error?: string;
};

export function GlobalNoticeLayer({ warnings = [], error }: GlobalNoticeLayerProps) {
  if (!warnings.length && !error) return null;
  return (
    <section className={`global-notice-layer ${warnings.length ? 'is-fallback' : ''}`.trim()}>
      {error ? <p className="error-banner">加载失败：{error}</p> : null}
      {warnings.map((warning) => (
        <p key={warning}>{warning}</p>
      ))}
    </section>
  );
}
