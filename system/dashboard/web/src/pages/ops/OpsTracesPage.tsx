import { Link } from 'react-router-dom';
import { KeyValueGrid, PanelCard } from '../../components/ui-kit';
import { buildOpsPageLink } from '../../navigation/route-contract';
import { buildTerminalFocusKey, buildWorkspaceLink, type SharedFocusState } from '../../utils/focus-links';
import { safeDisplayValue, compactPath } from '../../utils/formatters';
import type { TerminalReadModel } from '../../types/contracts';

type OpsTracesPageProps = {
  model: TerminalReadModel;
  focus: SharedFocusState;
};

function resolveArtifactId(model: TerminalReadModel, focus: SharedFocusState): string {
  if (focus.artifact) return safeDisplayValue(focus.artifact);
  const focusKey = buildTerminalFocusKey({
    panel: focus.panel,
    section: focus.section,
    row: focus.row,
  });
  const sectionKey = buildTerminalFocusKey({
    panel: focus.panel,
    section: focus.section,
  });
  return safeDisplayValue(model.navigation.terminalPrimaryArtifact[focusKey] || model.navigation.terminalPrimaryArtifact[sectionKey]);
}

export function OpsTracesPage({ model, focus }: OpsTracesPageProps) {
  const artifactId = resolveArtifactId(model, focus);
  const artifactRow = artifactId
    ? model.workspace.artifactRows.find((row) => [row.id, row.payload_key, row.label].map((value) => safeDisplayValue(value)).includes(artifactId))
    : null;
  const sourceHead = artifactId
    ? model.workspace.sourceHeads.find((row) => [row.id, row.label, row.path].map((value) => safeDisplayValue(value)).includes(artifactId))
      || model.workspace.sourceHeads.find((row) => safeDisplayValue(row.id) === artifactId)
    : null;
  const traceLinks = artifactId !== '—'
    ? (model.workspace.artifactHandoffs[artifactId] || [])
    : [];

  return (
    <section className="ops-traces-page" aria-label="ops-traces-page">
      <div className="workspace-grid single-column">
        <PanelCard
          title="链路上下文"
          kicker="trace / context"
          meta="固定展示当前 panel/section/row/artifact 与 source-owned 路径，不在其他页面重复解释。"
          actions={<Link className="button" to={buildOpsPageLink('risk', focus)}>回风险驾驶舱</Link>}
        >
          <KeyValueGrid
            rows={[
              { key: 'panel', label: '当前面板', value: focus.panel || '—', showRaw: true },
              { key: 'section', label: '当前分层', value: focus.section || '—', showRaw: true },
              { key: 'row', label: '当前行焦点', value: focus.row || '—', showRaw: true },
              { key: 'artifact', label: '当前工件', value: artifactId || '—', showRaw: true },
              { key: 'focus-key', label: '穿透键', value: buildTerminalFocusKey({ panel: focus.panel, section: focus.section, row: focus.row }) || '—', showRaw: true },
              { key: 'snapshot', label: '快照路径', value: model.surface.snapshotPath || '—', kind: 'path', showRaw: true },
              { key: 'artifact-path', label: '工件路径', value: artifactRow?.path || '—', kind: 'path', showRaw: true },
              { key: 'source-head-path', label: '源头路径', value: sourceHead?.path || '—', kind: 'path', showRaw: true },
            ]}
          />
        </PanelCard>

        <PanelCard
          title="证据跳转"
          kicker="trace / readonly jumps"
          meta="所有追踪跳转都保持只读。"
          actions={<Link className="button" to={buildOpsPageLink('workflow', focus)}>进入工作流</Link>}
        >
          <div className="button-row">
            {artifactId !== '—' ? <Link className="button" to={buildWorkspaceLink('artifacts', { artifact: artifactId })}>查看工件池</Link> : null}
            {artifactId !== '—' ? <Link className="button" to={buildWorkspaceLink('raw', { artifact: artifactId })}>查看原始层</Link> : null}
            {traceLinks.map((link) => (
              <Link className="button" key={link.id} to={link.to}>{link.label}</Link>
            ))}
          </div>
        </PanelCard>

        <PanelCard
          title="回退链"
          kicker="trace / fallback chain"
          meta="直接暴露当前 fallbackChain，便于核对 snapshot 与原始证据回落路径。"
        >
          <div className="stack-list">
            {model.workspace.fallbackChain.length ? model.workspace.fallbackChain.map((item, index) => (
              <div className="empty-block" key={`${item}-${index}`}>
                <strong>#{index + 1}</strong> {compactPath(item)}
              </div>
            )) : <div className="empty-block">未声明回退链</div>}
          </div>
        </PanelCard>
      </div>
    </section>
  );
}
