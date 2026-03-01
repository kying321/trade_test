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
  --samples N             Samples per model (default: 3)
  --max-tokens N          max_tokens for ping request (default: 8)
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
samples=3
max_tokens=8
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
    --samples)
      shift
      samples="${1:-3}"
      ;;
    --max-tokens)
      shift
      max_tokens="${1:-8}"
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
if ! [[ "$max_tokens" =~ ^[0-9]+$ ]] || (( max_tokens <= 0 )); then
  echo "ERROR: --max-tokens must be a positive integer." >&2
  exit 2
fi

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$repo_root" ]]; then
  echo "ERROR: must run inside git repository." >&2
  exit 2
fi

api_key="${!api_key_env:-}"
if [[ -z "$api_key" && "$api_key_env" == "NEWAPI_API_KEY" ]]; then
  api_key="${X666_API_KEY:-}"
  if [[ -n "$api_key" ]]; then
    api_key_env="X666_API_KEY"
  fi
fi
if [[ -z "$api_key" ]]; then
  echo "ERROR: missing API key; set ${api_key_env} (or X666_API_KEY)." >&2
  exit 2
fi

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
trap 'rm -rf "$tmpdir"' EXIT
results_jsonl="${tmpdir}/records.jsonl"

endpoint="${base_url%/}/v1/chat/completions"

run_probe() {
  local requested_model="$1"
  local effective_model="$2"
  local sample_index="$3"
  local request_ts payload response curl_rc http_status body head_snippet prompt_tokens
  local ok err_kind err_message

  payload="$(jq -nc \
    --arg model "$effective_model" \
    --argjson max_tokens "$max_tokens" \
    '{model:$model,messages:[{role:"user",content:"ping"}],max_tokens:$max_tokens}')"

  bucket_acquire
  request_ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  set +e
  response="$(
    curl -sS \
      --connect-timeout 5 \
      --max-time 5 \
      -A "fenlie-model-probe/1.0" \
      "${endpoint}" \
      -H "Authorization: Bearer ${api_key}" \
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
  if (( curl_rc != 0 )); then
    err_kind="curl_error"
    err_message="curl_exit_${curl_rc}"
  elif [[ "$http_status" =~ ^2[0-9][0-9]$ ]]; then
    ok=true
  else
    err_kind="http_error"
    err_message="status_${http_status:-unknown}"
  fi

  jq -nc \
    --arg timestamp_utc "$request_ts" \
    --arg requested_model "$requested_model" \
    --arg effective_model "$effective_model" \
    --argjson sample_index "$sample_index" \
    --argjson curl_rc "$curl_rc" \
    --arg http_status "${http_status:-}" \
    --argjson ok "$ok" \
    --arg error_kind "$err_kind" \
    --arg error_message "$err_message" \
    --arg response_head "$head_snippet" \
    --arg prompt_tokens "${prompt_tokens:-}" \
    '{
      timestamp_utc:$timestamp_utc,
      sample_index:$sample_index,
      requested_model:$requested_model,
      effective_model:$effective_model,
      curl_rc:$curl_rc,
      http_status:(if $http_status=="" then null else ($http_status|tonumber? // null) end),
      ok:$ok,
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

jq -s \
  --arg generated_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg base_url "${base_url%/}" \
  --arg endpoint "$endpoint" \
  --arg api_key_env "$api_key_env" \
  --argjson timeout_ms 5000 \
  --argjson bucket_capacity "$bucket_capacity" \
  --argjson bucket_refill_interval_sec "$bucket_refill_interval_sec" \
  '
  def pct(s;t): if t == 0 then 0 else ((s / t) * 100) end;
  . as $records
  | {
      generated_at_utc:$generated_at_utc,
      base_url:$base_url,
      endpoint:$endpoint,
      api_key_env:$api_key_env,
      policy:{
        timeout_ms:$timeout_ms,
        token_bucket:{
          capacity:$bucket_capacity,
          refill_token_every_seconds:$bucket_refill_interval_sec
        }
      },
      records:$records,
      summary:(
        $records
        | group_by(.requested_model)
        | map({
            requested_model:.[0].requested_model,
            effective_model:.[0].effective_model,
            samples:length,
            success:(map(select(.ok == true)) | length),
            failure:(map(select(.ok != true)) | length),
            success_rate_pct:pct((map(select(.ok == true)) | length); length)
          })
      )
    }' "$results_jsonl" > "$artifact_json"

{
  echo "# NewAPI Model Probe (${ts})"
  echo
  echo "- base_url: \`${base_url%/}\`"
  echo "- endpoint: \`${endpoint}\`"
  echo "- api_key_env: \`${api_key_env}\`"
  echo "- timeout_ms: \`5000\`"
  echo "- token_bucket: \`capacity=${bucket_capacity}, refill_every=${bucket_refill_interval_sec}s\`"
  echo "- models (requested): \`${models_csv}\`"
  echo "- samples_per_model: \`${samples}\`"
  echo
  echo "## Summary"
  jq -r '.summary[] | "- \(.requested_model) => \(.effective_model): success=\(.success)/\(.samples), success_rate=\(.success_rate_pct|tostring)%"' "$artifact_json"
  echo
  echo "## Artifact"
  echo "- JSON: ${artifact_json}"
} > "$artifact_md"

echo "$artifact_json"
