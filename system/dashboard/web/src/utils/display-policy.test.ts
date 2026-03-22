import { describe, expect, it } from 'vitest';
import {
  shouldShowRawCompactPath,
  shouldShowRawDrilldownField,
  shouldShowRawKeyValueField,
  shouldShowRawMetricValues,
  shouldShowRawTableColumn,
  shouldShowRawValueText,
} from './display-policy';

describe('display policy contract', () => {
  it('internal surface exposes raw audit fields for configured terminal sections', () => {
    expect(shouldShowRawMetricValues('internal', 'terminal.signalRisk.gateScores')).toBe(true);
    expect(shouldShowRawTableColumn('internal', 'terminal.signalRisk.focusSlots', 'action')).toBe(true);
    expect(shouldShowRawDrilldownField('internal', 'terminal.signalRisk.controlChain', 'blocking_reason')).toBe(true);
  });

  it('public surface keeps the same fields redacted at the ui policy layer', () => {
    expect(shouldShowRawMetricValues('public', 'terminal.signalRisk.gateScores')).toBe(false);
    expect(shouldShowRawTableColumn('public', 'terminal.signalRisk.focusSlots', 'action')).toBe(false);
    expect(shouldShowRawDrilldownField('public', 'terminal.signalRisk.controlChain', 'blocking_reason')).toBe(false);
  });

  it('workspace source-head policy is centrally declared', () => {
    expect(shouldShowRawKeyValueField('internal', 'workspace.contracts.sourceHeads', 'summary')).toBe(true);
    expect(shouldShowRawValueText('internal', 'workspace.contracts.sourceHeads.summary', 'label')).toBe(true);
    expect(shouldShowRawValueText('public', 'workspace.contracts.sourceHeads.summary', 'label')).toBe(false);
  });

  it('supports wildcard policy for dynamic artifact payload fields', () => {
    expect(shouldShowRawKeyValueField('internal', 'workspace.artifacts.payload', 'any_dynamic_key')).toBe(true);
    expect(shouldShowRawValueText('internal', 'workspace.artifacts.topLevelKeys', 'generated_at_utc')).toBe(true);
    expect(shouldShowRawKeyValueField('public', 'workspace.artifacts.payload', 'any_dynamic_key')).toBe(false);
  });

  it('covers artifact title/path policy through the same contract', () => {
    expect(shouldShowRawValueText('internal', 'workspace.artifacts.list', 'title')).toBe(true);
    expect(shouldShowRawCompactPath('internal', 'workspace.artifacts.list', 'path')).toBe(true);
    expect(shouldShowRawValueText('public', 'workspace.artifacts.inspector', 'title')).toBe(false);
  });
});
