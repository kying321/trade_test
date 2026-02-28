#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/baseline_promotion_rollback_drill.sh [options]

Options:
  --promotion-file PATH   Baseline promotion json file.
  --output-dir PATH       Drill artifact output dir (default: output/review).
  -h, --help              Show help.
EOF
}

promotion_file=""
output_dir="output/review"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --promotion-file)
      shift
      promotion_file="${1:-}"
      ;;
    --output-dir)
      shift
      output_dir="${1:-}"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
  shift
done

if [[ -z "$promotion_file" ]]; then
  promotion_file="$(ls -1t output/review/*_baseline_promotion.json 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "$promotion_file" ]]; then
  echo "ERROR: baseline_promotion artifact not found." >&2
  exit 2
fi

mkdir -p "$output_dir"

json_payload="$(
python3 - <<'PY' "$promotion_file"
import json
import os
import sys

p = sys.argv[1]
with open(p, "r", encoding="utf-8") as f:
    data = json.load(f)

required = ("active_path", "rollback_anchor")
missing = [k for k in required if not str(data.get(k, "")).strip()]
if missing:
    print(f"ERROR: missing fields in {p}: {','.join(missing)}", file=sys.stderr)
    sys.exit(3)

active_path = str(data.get("active_path", "")).strip()
rollback_anchor = str(data.get("rollback_anchor", "")).strip()

if not os.path.exists(active_path):
    print(f"ERROR: active baseline path missing: {active_path}", file=sys.stderr)
    sys.exit(4)
if not os.path.exists(rollback_anchor):
    print(f"ERROR: rollback anchor missing: {rollback_anchor}", file=sys.stderr)
    sys.exit(5)

print(json.dumps(
    {
        "promotion_file": p,
        "active_path": active_path,
        "rollback_anchor": rollback_anchor,
        "as_of": str(data.get("as_of", "")),
        "round": int(data.get("round", 0) or 0),
    },
    ensure_ascii=False,
))
PY
)"

active_path="$(python3 - <<'PY' "$json_payload"
import json,sys
print(json.loads(sys.argv[1])["active_path"])
PY
)"
rollback_anchor="$(python3 - <<'PY' "$json_payload"
import json,sys
print(json.loads(sys.argv[1])["rollback_anchor"])
PY
)"

tmp_backup="$(mktemp "${TMPDIR:-/tmp}/baseline_rollback_drill_active.XXXXXX")"
restored=0
cleanup() {
  if [[ $restored -eq 0 && -f "$tmp_backup" && -f "$active_path" ]]; then
    cp -f "$tmp_backup" "$active_path" || true
  fi
  rm -f "$tmp_backup" || true
}
trap cleanup EXIT

cp -f "$active_path" "$tmp_backup"

orig_sha="$(shasum -a 256 "$active_path" | awk '{print $1}')"
anchor_sha="$(shasum -a 256 "$rollback_anchor" | awk '{print $1}')"

cp -f "$rollback_anchor" "$active_path"
drill_sha="$(shasum -a 256 "$active_path" | awk '{print $1}')"
if [[ "$drill_sha" != "$anchor_sha" ]]; then
  echo "ERROR: drill write mismatch: active sha != rollback anchor sha" >&2
  exit 6
fi

cp -f "$tmp_backup" "$active_path"
restored=1
final_sha="$(shasum -a 256 "$active_path" | awk '{print $1}')"
if [[ "$final_sha" != "$orig_sha" ]]; then
  echo "ERROR: restore mismatch: final sha != original sha" >&2
  exit 7
fi

ts="$(date -u +%FT%TZ)"
day_stamp="$(date +%F)"
json_out="$output_dir/${day_stamp}_baseline_rollback_drill.json"
md_out="$output_dir/${day_stamp}_baseline_rollback_drill.md"
python3 - <<'PY' "$json_payload" "$ts" "$orig_sha" "$anchor_sha" "$json_out"
import json,sys
payload=json.loads(sys.argv[1])
payload.update(
    {
        "status": "passed",
        "executed_at_utc": sys.argv[2],
        "active_original_sha256": sys.argv[3],
        "rollback_anchor_sha256": sys.argv[4],
        "validation": {
            "drill_write_match_anchor": True,
            "restore_match_original": True,
        },
    }
)
with open(sys.argv[5], "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
PY

cat >"$md_out" <<EOF
# Baseline Rollback Drill

- status: \`passed\`
- executed_at_utc: \`$ts\`
- promotion_file: \`$promotion_file\`
- active_path: \`$active_path\`
- rollback_anchor: \`$rollback_anchor\`
- active_original_sha256: \`$orig_sha\`
- rollback_anchor_sha256: \`$anchor_sha\`
- validation:
  - \`drill_write_match_anchor=true\`
  - \`restore_match_original=true\`
EOF

echo "PASSED baseline rollback drill"
echo "artifact_json=$json_out"
