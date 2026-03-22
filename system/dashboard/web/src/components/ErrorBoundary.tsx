import { Component, type ReactNode } from 'react';
import { labelFor } from '../utils/dictionary';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  message: string;
}

export class ErrorBoundary extends Component<Props, State> {
  public constructor(props: Props) {
    super(props);
    this.state = { hasError: false, message: '' };
  }

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, message: error?.message || labelFor('unknown_render_error') };
  }

  public componentDidCatch(error: Error): void {
    console.error('Fenlie terminal render error:', error);
  }

  public render(): ReactNode {
    if (!this.state.hasError) return this.props.children;
    return (
      <div className="app-shell terminal-shell fallback-shell">
        <section className="panel-card fallback-card">
          <div className="panel-card-head">
            <div>
              <p className="panel-kicker">{labelFor('fallback_shell_kicker')}</p>
              <h1>{labelFor('fallback_shell_title')}</h1>
            </div>
          </div>
          <p className="panel-note">{labelFor('error_message_label')}：{this.state.message}</p>
          <div className="button-row">
            <button className="button button-primary" onClick={() => window.location.reload()}>
              {labelFor('reload_page')}
            </button>
            <a className="button" href="/data/fenlie_dashboard_snapshot.json" target="_blank" rel="noreferrer">
              {labelFor('open_public_snapshot')}
            </a>
            <a className="button" href="/operator_task_visual_panel.html" target="_blank" rel="noreferrer">
              {labelFor('open_operator_panel')}
            </a>
          </div>
        </section>
      </div>
    );
  }
}
