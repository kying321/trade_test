#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  system/scripts/newapi_model_probe.sh [options]

Options:
  --base-url URL          API base URL (default: https://x666.me)
  --api-key-env NAME      API key env name (default: NEWAPI_API_KEY; fallback: X666_API_KEY)
  --models CSV            Requested model list (default: gpt-5.3-codex,gemini-3.1-pro-preview-bs)
  --required-models CSV   Required models for gate (default: gpt-5.3-codex)
  --optional-models CSV   Optional models for gate (default: gemini-3.1-pro-preview-bs)
  --samples N             Samples per model (default: 3)
  --retry-transient N     Retries for transient failures (default: 1)
  --retry-backoff-ms N    Backoff between transient retries in ms (default: 300)
  --max-tokens N          max_tokens for ping request (default: 8)
  --isolation-config-path PATH
                          Config path to flip `binance_live_takeover_enabled` on hard fail
                          (default: system/config.yaml; supports absolute path)
  --disable-isolation-write
                          Do not mutate config on hard fail (still exits 1)
  --output-dir PATH       Artifact dir relative to repo root (default: system/output/review)
  -h, --help              Show help

Notes:
  - Applies model alias mapping: gemini-pro-3.1 -> gemini-3.1-pro-preview-bs
  - Uses token bucket limiter (10 req/min) and hard 5s timeout.
USAGE
}

base_url="https://x666.me"
api_key_env="NEWAPI_API_KEY"
models_csv="gpt-5.3-codex,gemini-3.1-pro-preview-bs"
required_models_csv="gpt-5.3-codex"
optional_models_csv="gemini-3.1-pro-preview-bs"
samples=3
retry_transient=1
retry_backoff_ms=300
max_tokens=8
isolation_config_path="system/config.yaml"
isolation_write_enabled=true
output_dir="system/output/review"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url)
      shift
      base_url="${1:-}"
      ;;
    --api-key-env)
      shift
      api_key_env="${1:-}"
      ;;
    --models)
      shift
      models_csv="${1:-}"
      ;;
    --required-models)
      shift
      required_models_csv="${1:-$required_models_csv}"
      ;;
    --optional-models)
      shift
      optional_models_csv="${1:-$optional_models_csv}"
      ;;
    --samples)
      shift
      samples="${1:-3}"
      ;;
    --retry-transient)
      shift
      retry_transient="${1:-1}"
      ;;
    --retry-backoff-ms)
      shift
      retry_backoff_ms="${1:-300}"
      ;;
    --max-tokens)
      shift
      max_tokens="${1:-8}"
      ;;
    --isolation-config-path)
      shift
      isolation_config_path="${1:-system/config.yaml}"
      ;;
    --disable-isolation-write)
      isolation_write_enabled=false
      ;;
    --output-dir)
      shift
      output_dir="${1:-system/output/review}"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown arg '$1'" >&2
      usage
      exit 2
      ;;
  esac
  shift || true
done

if ! [[ "$samples" =~ ^[0-9]+$ ]] || (( samples <= 0 )); then
  echo "ERROR: --samples must be a positive integer." >&2
  exit 2
fi
if ! [[ "$retry_transient" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --retry-transient must be a non-negative integer." >&2
  exit 2
fi
if ! [[ "$retry_backoff_ms" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --retry-backoff-ms must be a non-negative integer." >&2
  exit 2
fi
if ! [[ "$max_tokens" =~ ^[0-9]+$ ]] || (( max_tokens <= 0 )); then
  echo "ERROR: --max-tokens must be a positive integer." >&2
  exit 2
fi
if [[ "$isolation_write_enabled" != "true" && "$isolation_write_enabled" != "false" ]]; then
  echo "ERROR: internal isolation_write_enabled must be true/false." >&2
  exit 2
fi
if [[ -z "${isolation_config_path// }" ]]; then
  echo "ERROR: --isolation-config-path must not be empty." >&2
  exit 2
fi

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$repo_root" ]]; then
  echo "ERROR: must run inside git repository." >&2
  exit 2
fi

mutex_lock_path="${repo_root}/system/output/state/run-halfhour-pulse.lock"
mutex_timeout_seconds=5
mutex_lock_acquired=0

release_run_halfhour_mutex() {
  local lock_pid=""
  if (( mutex_lock_acquired != 1 )); then
    return 0
  fi
  if [[ -f "$mutex_lock_path" ]]; then
    lock_pid="$(cat "$mutex_lock_path" 2>/dev/null || true)"
    if [[ "$lock_pid" == "$$" ]]; then
      rm -f "$mutex_lock_path" 2>/dev/null || true
    fi
  fi
  mutex_lock_acquired=0
}

acquire_run_halfhour_mutex() {
  local deadline now lock_pid
  mkdir -p "$(dirname "$mutex_lock_path")"
  deadline=$(( $(date +%s) + mutex_timeout_seconds ))
  while true; do
    if ( set -o noclobber; echo "$$" > "$mutex_lock_path" ) 2>/dev/null; then
      mutex_lock_acquired=1
      return 0
    fi
    lock_pid="$(cat "$mutex_lock_path" 2>/dev/null || true)"
    if [[ -n "$lock_pid" ]] && ! kill -0 "$lock_pid" 2>/dev/null; then
      rm -f "$mutex_lock_path" 2>/dev/null || true
      continue
    fi
    now="$(date +%s)"
    if (( now >= deadline )); then
      echo "ERROR: run-halfhour-pulse mutex timeout: ${mutex_timeout_seconds}s" >&2
      return 1
    fi
    sleep 0.1
  done
}

read_key_from_env_file() {
  local env_file="$1"
  local var_name="$2"
  local line value
  [[ -f "$env_file" ]] || return 1
  line="$(grep -E "^[[:space:]]*${var_name}=" "$env_file" | tail -n1 || true)"
  [[ -n "$line" ]] || return 1
  value="${line#*=}"
  value="${value%$'\r'}"
  case "$value" in
    \"*\")
      value="${value#\"}"
      value="${value%\"}"
      ;;
    \'*\')
      value="${value#\'}"
      value="${value%\'}"
      ;;
  esac
  printf '%s' "$value"
}

read_key_from_openclaw_json() {
  local json_file="$1"
  [[ -f "$json_file" ]] || return 1
  jq -r '
    [
      {source:"OPENCLAW_JSON:skills.entries.openai.apiKey",value:(.skills.entries.openai.apiKey // empty)},
      {source:"OPENCLAW_JSON:skills.entries.openrouter.apiKey",value:(.skills.entries.openrouter.apiKey // empty)},
      {source:"OPENCLAW_JSON:skills.entries.google-antigravity.apiKey",value:(.skills.entries["google-antigravity"].apiKey // empty)},
      {source:"OPENCLAW_JSON:skills.entries.google-gemini.apiKey",value:(.skills.entries["google-gemini"].apiKey // empty)}
    ]
    | .[]
    | select(.value | type == "string" and length > 0)
    | [.source, .value] | @tsv
  ' "$json_file" 2>/dev/null || true
}

declare -a api_keys=()
declare -a api_key_sources=()

append_api_key_candidate() {
  local key="$1"
  local source="$2"
  local i
  [[ -n "$key" ]] || return 0
  for i in "${!api_keys[@]}"; do
    if [[ "${api_keys[$i]}" == "$key" ]]; then
      return 0
    fi
  done
  api_keys+=("$key")
  api_key_sources+=("$source")
}

# Priority 1: process env vars.
append_api_key_candidate "${!api_key_env:-}" "$api_key_env"
if [[ "$api_key_env" == "NEWAPI_API_KEY" ]]; then
  append_api_key_candidate "${X666_API_KEY:-}" "X666_API_KEY"
fi

# Priority 2: ~/.openclaw/.env fallback.
openclaw_env="${HOME}/.openclaw/.env"
append_api_key_candidate "$(read_key_from_env_file "$openclaw_env" "$api_key_env" || true)" "OPENCLAW_ENV:${api_key_env}"
if [[ "$api_key_env" == "NEWAPI_API_KEY" ]]; then
  append_api_key_candidate "$(read_key_from_env_file "$openclaw_env" "X666_API_KEY" || true)" "OPENCLAW_ENV:X666_API_KEY"
fi

# Priority 3: ~/.openclaw/openclaw.json skill entries.
if (( ${#api_keys[@]} == 0 )); then
  openclaw_json="${HOME}/.openclaw/openclaw.json"
  while IFS=$'\t' read -r key_source key_value; do
    append_api_key_candidate "$key_value" "$key_source"
  done < <(read_key_from_openclaw_json "$openclaw_json")
fi

if (( ${#api_keys[@]} == 0 )); then
  echo "ERROR: missing API key; set ${api_key_env} (or X666_API_KEY), or configure ~/.openclaw/.env / ~/.openclaw/openclaw.json." >&2
  exit 2
fi
api_key="${api_keys[0]}"
api_key_env="${api_key_sources[0]}"
api_key_sources_csv="$(
  IFS=','
  echo "${api_key_sources[*]}"
)"

normalize_model_name() {
  local raw="$1"
  case "$raw" in
    gemini-pro-3.1)
      printf '%s' "gemini-3.1-pro-preview-bs"
      ;;
    *)
      printf '%s' "$raw"
      ;;
  esac
}

# Token bucket: 10 requests per minute.
bucket_capacity=10
bucket_tokens=$bucket_capacity
bucket_refill_interval_sec=6
bucket_last_refill="$(date +%s)"

bucket_refill() {
  local now elapsed refill
  now="$(date +%s)"
  elapsed=$(( now - bucket_last_refill ))
  if (( elapsed >= bucket_refill_interval_sec )); then
    refill=$(( elapsed / bucket_refill_interval_sec ))
    bucket_tokens=$(( bucket_tokens + refill ))
    if (( bucket_tokens > bucket_capacity )); then
      bucket_tokens=$bucket_capacity
    fi
    bucket_last_refill=$(( bucket_last_refill + refill * bucket_refill_interval_sec ))
  fi
}

bucket_acquire() {
  bucket_refill
  while (( bucket_tokens <= 0 )); do
    sleep 1
    bucket_refill
  done
  bucket_tokens=$(( bucket_tokens - 1 ))
}

ts="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${repo_root}/${output_dir}"
tmpdir="$(mktemp -d)"
cleanup_probe_runtime() {
  rm -rf "$tmpdir" 2>/dev/null || true
  release_run_halfhour_mutex
}
trap cleanup_probe_runtime EXIT
results_jsonl="${tmpdir}/records.jsonl"

endpoint="${base_url%/}/v1/chat/completions"

is_transient_http_status() {
  local status="$1"
  case "$status" in
    408|429|500|502|503|504)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

build_normalized_csv_list() {
  local csv="$1"
  local out="" raw trimmed norm
  local -a items
  IFS=',' read -r -a items <<< "$csv"
  for raw in "${items[@]}"; do
    trimmed="$(printf '%s' "$raw" | xargs)"
    [[ -z "$trimmed" ]] && continue
    norm="$(normalize_model_name "$trimmed")"
    if [[ -z "$out" ]]; then
      out="$norm"
    else
      out="${out},${norm}"
    fi
  done
  printf '%s' "$out"
}

run_probe() {
  local requested_model="$1"
  local effective_model="$2"
  local sample_index="$3"
  local request_ts payload response curl_rc http_status body head_snippet prompt_tokens
  local ok err_kind err_message transient_failure retried
  local attempt max_attempts attempt_count sleep_seconds
  local api_key_idx api_key_for_attempt api_key_source_for_attempt

  payload="$(jq -nc \
    --arg model "$effective_model" \
    --argjson max_tokens "$max_tokens" \
    '{model:$model,messages:[{role:"user",content:"ping"}],max_tokens:$max_tokens}')"

  max_attempts=$(( retry_transient + 1 ))
  attempt=1
  attempt_count=0
  retried=false
  while (( attempt <= max_attempts )); do
    attempt_count=$attempt
    api_key_idx=$(( (attempt - 1) % ${#api_keys[@]} ))
    api_key_for_attempt="${api_keys[$api_key_idx]}"
    api_key_source_for_attempt="${api_key_sources[$api_key_idx]}"
    bucket_acquire
    request_ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    set +e
    response="$(
      curl -sS \
        --connect-timeout 5 \
        --max-time 5 \
        -A "fenlie-model-probe/1.0" \
        "${endpoint}" \
        -H "Authorization: Bearer ${api_key_for_attempt}" \
        -H 'Content-Type: application/json' \
        --data-raw "${payload}" \
        -w '\nHTTPSTATUS:%{http_code}\n'
    )"
    curl_rc=$?
    set -e

    http_status=""
    body=""
    if (( curl_rc == 0 )); then
      http_status="$(printf '%s' "$response" | sed -n 's/^HTTPSTATUS://p' | tail -n1)"
      body="$(printf '%s' "$response" | sed '/^HTTPSTATUS:/d')"
    fi

    head_snippet="$(printf '%s' "$body" | head -c 180 | tr '\n' ' ')"
    prompt_tokens="$(printf '%s' "$body" | jq -r '.usage.prompt_tokens // empty' 2>/dev/null || true)"

    ok=false
    err_kind=""
    err_message=""
    transient_failure=false
    if (( curl_rc != 0 )); then
      err_kind="curl_error"
      err_message="curl_exit_${curl_rc}"
      transient_failure=true
    elif [[ "$http_status" =~ ^2[0-9][0-9]$ ]]; then
      ok=true
    else
      err_kind="http_error"
      err_message="status_${http_status:-unknown}"
      if is_transient_http_status "${http_status:-}"; then
        transient_failure=true
      fi
    fi

    if [[ "$ok" == "true" ]]; then
      break
    fi
    if [[ "$transient_failure" != "true" ]] || (( attempt >= max_attempts )); then
      break
    fi
    retried=true
    if (( retry_backoff_ms > 0 )); then
      sleep_seconds="$(awk -v ms="$retry_backoff_ms" 'BEGIN { printf "%.3f", (ms / 1000.0) }')"
      sleep "$sleep_seconds"
    fi
    attempt=$(( attempt + 1 ))
  done

  jq -nc \
    --arg timestamp_utc "$request_ts" \
    --arg requested_model "$requested_model" \
    --arg effective_model "$effective_model" \
    --arg api_key_source "$api_key_source_for_attempt" \
    --argjson sample_index "$sample_index" \
    --argjson api_key_slot "$((api_key_idx + 1))" \
    --argjson attempts "$attempt_count" \
    --argjson max_attempts "$max_attempts" \
    --argjson curl_rc "$curl_rc" \
    --arg http_status "${http_status:-}" \
    --argjson ok "$ok" \
    --argjson retried "$retried" \
    --argjson transient_failure "$transient_failure" \
    --arg error_kind "$err_kind" \
    --arg error_message "$err_message" \
    --arg response_head "$head_snippet" \
    --arg prompt_tokens "${prompt_tokens:-}" \
    '{
      timestamp_utc:$timestamp_utc,
      sample_index:$sample_index,
      api_key_source:(if $api_key_source=="" then null else $api_key_source end),
      api_key_slot:$api_key_slot,
      requested_model:$requested_model,
      effective_model:$effective_model,
      curl_rc:$curl_rc,
      attempts:$attempts,
      max_attempts:$max_attempts,
      http_status:(if $http_status=="" then null else ($http_status|tonumber? // null) end),
      ok:$ok,
      retried:$retried,
      transient_failure:$transient_failure,
      error_kind:(if $error_kind=="" then null else $error_kind end),
      error_message:(if $error_message=="" then null else $error_message end),
      prompt_tokens:(if $prompt_tokens=="" then null else ($prompt_tokens|tonumber? // null) end),
      response_head:$response_head
    }' >> "$results_jsonl"
}

IFS=',' read -r -a model_items <<< "$models_csv"
for raw in "${model_items[@]}"; do
  requested="$(printf '%s' "$raw" | xargs)"
  if [[ -z "$requested" ]]; then
    continue
  fi
  effective="$(normalize_model_name "$requested")"
  for (( i = 1; i <= samples; i++ )); do
    run_probe "$requested" "$effective" "$i"
  done
done

artifact_json="${repo_root}/${output_dir}/${ts}_newapi_model_probe.json"
artifact_md="${repo_root}/${output_dir}/${ts}_newapi_model_probe.md"
required_models_effective_csv="$(build_normalized_csv_list "$required_models_csv")"
optional_models_effective_csv="$(build_normalized_csv_list "$optional_models_csv")"

jq -s \
  --arg generated_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg base_url "${base_url%/}" \
  --arg endpoint "$endpoint" \
  --arg api_key_env "$api_key_env" \
  --arg api_key_sources_csv "$api_key_sources_csv" \
  --arg isolation_config_path "$isolation_config_path" \
  --argjson isolation_write_enabled "$isolation_write_enabled" \
  --arg required_models_csv "$required_models_effective_csv" \
  --arg optional_models_csv "$optional_models_effective_csv" \
  --argjson timeout_ms 5000 \
  --argjson bucket_capacity "$bucket_capacity" \
  --argjson bucket_refill_interval_sec "$bucket_refill_interval_sec" \
  --argjson retry_transient "$retry_transient" \
  --argjson retry_backoff_ms "$retry_backoff_ms" \
  '
  def pct(s;t): if t == 0 then 0 else ((s / t) * 100) end;
  def split_csv(s):
    if s == "" then []
    else (s | split(",") | map(gsub("^\\s+|\\s+$"; "") | select(length > 0)))
    end;
  . as $records
  | (split_csv($required_models_csv)) as $required
  | (split_csv($optional_models_csv)) as $optional
  | (
      $records
      | group_by(.requested_model)
      | map(
          . as $grp
          | ($grp[0].effective_model) as $m
          | {
              requested_model:$grp[0].requested_model,
              effective_model:$m,
              tier:(
                if (($required | index($m)) != null) then "required"
                elif (($optional | index($m)) != null) then "optional"
                else "required"
                end
              ),
              samples:($grp | length),
              success:($grp | map(select(.ok == true)) | length),
              failure:($grp | map(select(.ok != true)) | length),
              retried_samples:($grp | map(select(.retried == true)) | length),
              transient_failures:($grp | map(select(.ok != true and .transient_failure == true)) | length),
              success_rate_pct:pct(($grp | map(select(.ok == true)) | length); ($grp | length))
            }
        )
    ) as $summary
  | (
      {
        required_effective:(
          if ($required | length) > 0 then $required
          else ($summary | map(select(.tier == "required")) | map(.effective_model))
          end
        ),
        available_effective:($summary | map(.effective_model)),
        optional_total:($summary | map(select(.tier == "optional")) | length),
        optional_failing_models:($summary | map(select(.tier == "optional" and .success == 0)) | map(.effective_model))
      }
      | . + {
          required_total:(.required_effective | length),
          required_passing:(
            [.required_effective[] as $req
             | select(
                 ($summary | map(select(.effective_model == $req and .success > 0)) | length) > 0
               )
             | $req
            ]
            | length
          ),
          required_missing_models:(
            [.required_effective[] as $req
             | select((.available_effective | index($req)) == null)
             | $req
            ]
          )
        }
    ) as $gate_counts
  | {
      generated_at_utc:$generated_at_utc,
      base_url:$base_url,
      endpoint:$endpoint,
      api_key_env:$api_key_env,
      api_key_sources:(split_csv($api_key_sources_csv)),
      api_key_candidate_count:(split_csv($api_key_sources_csv) | length),
      policy:{
        timeout_ms:$timeout_ms,
        retry_transient:$retry_transient,
        retry_backoff_ms:$retry_backoff_ms,
        isolation_config_path:$isolation_config_path,
        isolation_write_enabled:$isolation_write_enabled,
        model_tiers:{
          required:$required,
          optional:$optional
        },
        token_bucket:{
          capacity:$bucket_capacity,
          refill_token_every_seconds:$bucket_refill_interval_sec
        }
      },
      records:$records,
      summary:$summary,
      gate:(
        $gate_counts
        | . + {
            status:(
              if .required_total == 0 then
                (if ($summary | length) == 0 then "empty"
                 elif ($summary | all(.success > 0)) then "pass"
                 else "fail" end)
              elif ((.required_missing_models | length) > 0) then "fail"
              elif .required_passing < .required_total then "fail"
              elif ((.optional_failing_models | length) > 0) then "degraded"
              else "pass"
              end
            ),
            gate_ok:(
              if .required_total == 0 then
                (($summary | length) > 0 and ($summary | all(.success > 0)))
              else
                ((.required_missing_models | length) == 0) and (.required_passing == .required_total)
              end
            )
          }
      )
    }' "$results_jsonl" > "$artifact_json"

{
  echo "# NewAPI Model Probe (${ts})"
  echo
  echo "- base_url: \`${base_url%/}\`"
  echo "- endpoint: \`${endpoint}\`"
  echo "- api_key_env: \`${api_key_env}\`"
  echo "- api_key_sources: \`${api_key_sources_csv}\`"
  echo "- timeout_ms: \`5000\`"
  echo "- retry_transient: \`${retry_transient}\`"
  echo "- retry_backoff_ms: \`${retry_backoff_ms}\`"
  echo "- isolation_config_path: \`${isolation_config_path}\`"
  echo "- isolation_write_enabled: \`${isolation_write_enabled}\`"
  echo "- required_models: \`${required_models_effective_csv}\`"
  echo "- optional_models: \`${optional_models_effective_csv}\`"
  echo "- token_bucket: \`capacity=${bucket_capacity}, refill_every=${bucket_refill_interval_sec}s\`"
  echo "- models (requested): \`${models_csv}\`"
  echo "- samples_per_model: \`${samples}\`"
  echo
  echo "## Summary"
  jq -r '.summary[] | "- [\(.tier)] \(.requested_model) => \(.effective_model): success=\(.success)/\(.samples), retried=\(.retried_samples), transient_failures=\(.transient_failures), success_rate=\(.success_rate_pct|tostring)%"' "$artifact_json"
  echo
  echo "## Gate"
  jq -r '"- status: \(.gate.status), gate_ok: \(.gate.gate_ok), required=\(.gate.required_passing)/\(.gate.required_total), required_missing=\((.gate.required_missing_models|join(","))), optional_failures=\((.gate.optional_failing_models|join(",")))"' "$artifact_json"
  echo
  echo "## Artifact"
  echo "- JSON: ${artifact_json}"
} > "$artifact_md"

echo "$artifact_json"

probe_status="$(jq -r '.gate.status // "fail"' "$artifact_json")"
if [[ "$probe_status" == "fail" || "$probe_status" == "empty" ]]; then
  if [[ "$isolation_config_path" == /* ]]; then
    isolation_config="$isolation_config_path"
  else
    isolation_config="${repo_root}/${isolation_config_path}"
  fi
  echo "CRITICAL: Live Models failed connectivity check. Triggering ISOLATION." >&2
  if [[ "$isolation_write_enabled" != "true" ]]; then
      echo "Isolation config write disabled by flag; skip mutation." >&2
      exit 1
  fi
  if ! acquire_run_halfhour_mutex; then
      echo "ERROR: failed to acquire run-halfhour-pulse lock; isolation aborted." >&2
      exit 1
  fi
  if [[ -f "$isolation_config" ]]; then
      if grep -Eq '^[[:space:]]*binance_live_takeover_enabled:[[:space:]]*true([[:space:]]*#.*)?$' "$isolation_config"; then
          sed -i.bak 's/^\([[:space:]]*binance_live_takeover_enabled:[[:space:]]*\)true/\1false/' "$isolation_config"
          echo "Live Takeover isolated via ${isolation_config} modification." >&2
      else
          echo "Live Takeover already isolated in ${isolation_config}." >&2
      fi
  else
      echo "WARN: isolation config missing (${isolation_config}); skip." >&2
  fi
  release_run_halfhour_mutex
  exit 1
fi
