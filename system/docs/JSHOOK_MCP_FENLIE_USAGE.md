# JSHookMCP In Fenlie

## Status

JSHookMCP is configured in Codex at:

- `/Users/jokenrobot/.codex/config.toml`

Configured server:

```toml
[mcp_servers.jshook]
command = "npx"
args = ["-y", "@jshookmcp/jshook"]
enabled = true
```

The server is intended for browser-side JavaScript inspection, workflow automation, and network capture. It is not part of the Fenlie live execution path.

## Best-Fit Use Cases

### 1. Web workflow reverse engineering

Use JSHookMCP when a target capability exists only in a browser workflow and not in a documented API. Good examples:

- exchange web console navigation
- third-party trading dashboard request capture
- websocket/event schema discovery
- browser-side token/session flow observation

### 2. OpenClaw remote diagnostics

Use it to observe browser-visible state around remote execution tooling without touching signed live routing:

- compare remote web panel status with Fenlie source artifacts
- inspect shadow execution dashboards or admin panels
- capture network requests behind operator-only pages

### 3. UI and browser regression checks

Use it to automate browser checks for Fenlie-adjacent surfaces:

- operator dashboard rendering checks
- web-based control panel button flow validation
- page load, auth redirect, and websocket reconnect behavior

### 4. Research-only contract discovery

Use it to learn how a site structures requests before building a proper guarded client:

- REST endpoint patterns
- websocket subscribe/unsubscribe payloads
- pagination and filtering behavior
- browser-side anti-automation obstacles

## Fenlie-Specific Scenarios

### OpenClaw orderflow preparation

JSHookMCP is useful for discovering and validating browser-visible orderflow context around OpenClaw:

- inspect remote dashboard request sequences
- compare displayed order state with `remote_intent_queue`, `remote_orderflow_feedback`, and `remote_execution_ack_state`
- capture browser-side evidence when a remote admin panel disagrees with source artifacts

### Exchange web surface diagnostics

JSHookMCP can help when exchange behavior is only visible in the browser:

- inspect websocket payload shape from exchange web UIs
- capture read-only network traces for research
- verify whether a page exposes data not present in public API responses

### Third-party tooling integration research

If Fenlie later needs to interoperate with charting terminals, analytics dashboards, or broker consoles, JSHookMCP is a good discovery tool before writing production integration code.

## Do Not Use It For

Do not put JSHookMCP directly in these paths:

- live order routing
- signed order placement
- risk guard enforcement
- portfolio mutation
- queue ownership mutation
- any task that should already be handled by LiE/OpenClaw guarded API clients

Reason:

- browser hooks are less deterministic than direct guarded clients
- live trading paths in Fenlie require mutex, idempotency, rate limiting, timeout, and panic semantics already enforced elsewhere
- browser session state is a poor source of truth for execution

## Recommended Operating Pattern

1. Use JSHookMCP to observe or reverse engineer a browser workflow.
2. Extract only the stable request/response contract or diagnostic evidence.
3. Convert that knowledge into a normal Fenlie source artifact or guarded client.
4. Keep JSHookMCP out of the final live execution path.

## Suggested Prompts

- "Use JSHookMCP to inspect the browser network flow for the remote admin panel and summarize request endpoints, websocket channels, and auth redirects."
- "Use JSHookMCP to compare the exchange web position view with Fenlie `remote_execution_ack_state` and report mismatches."
- "Use JSHookMCP to capture browser websocket payloads for the target page and map them into a source-owned artifact design."

## Restart Note

Codex usually needs a restart or MCP server refresh before a newly added server appears in the MCP server list.
