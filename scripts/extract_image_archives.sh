#!/usr/bin/env bash
# Extract recovered ZIP/RAR/7Z archives that contain image content.
# Each archive's contents go into its own subfolder (basename, sans extension)
# under OUTPUT_DIR to prevent name collisions.
#
# Filtering: only archives whose first_10_entries (from verify_archives.sh
# output) include any of the configured image extensions are extracted.
# This skips archives that are just .dll/.exe/.txt/etc.
#
# Usage:
#   ./extract_image_archives.sh --tsv FILE --output DIR [--workers N]
#                              [--include-ext "jpg,jpeg,png,gif,mpo,heic"]
#                              [--dry-run] [--limit N]
#
# Safety:
#   - Dry-run by default (prints plan, doesn't extract)
#   - Skips archives whose target subfolder already exists (idempotent — safe re-run)
#   - Writes a transcript log of every success / failure
#   - --limit guardrail (default 10000)

set -uo pipefail

TSV=""
OUTPUT_DIR=""
WORKERS="${WORKERS:-4}"
INCLUDE_EXT="jpg,jpeg,png,gif,mpo,heic"
DRY_RUN=1
LIMIT=10000
TRANSCRIPT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tsv) TSV="$2"; shift 2 ;;
    --output) OUTPUT_DIR="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --include-ext) INCLUDE_EXT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --commit) DRY_RUN=0; shift ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --transcript) TRANSCRIPT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$TSV" || -z "$OUTPUT_DIR" ]]; then
  echo "Usage: $0 --tsv FILE --output DIR [--workers N] [--include-ext EXTS] [--commit] [--limit N]" >&2
  exit 2
fi
if [[ ! -f "$TSV" ]]; then echo "TSV not found: $TSV" >&2; exit 1; fi

# Build pipe-or regex from include-ext: "jpg,png" -> "\.(jpg|png)(\||$)"
ext_pattern=$(echo "$INCLUDE_EXT" | tr 'A-Z,' 'a-z|')
# We want to match the extensions inside the pipe-separated first_10_entries
# Each entry ends in either | (more entries follow) or end-of-line.
# Case-insensitive via tolower() in awk.

TS="$(date +%Y%m%dT%H%M%S)"
TRANSCRIPT="${TRANSCRIPT:-extract_archives_transcript_${TS}.log}"

echo "TSV:          $TSV"
echo "Output:       $OUTPUT_DIR"
echo "Workers:      $WORKERS"
echo "Include ext:  $INCLUDE_EXT"
echo "Mode:         $([[ $DRY_RUN -eq 1 ]] && echo 'DRY RUN' || echo 'COMMIT (will extract)')"
echo "Transcript:   $TRANSCRIPT"
echo ""

mkdir -p "$OUTPUT_DIR"

# Build the plan: archives that pass the filter.
PLAN=$(mktemp)
awk -F'\t' -v ext="$ext_pattern" '
  NR == 1 { next }
  $1 != "VALID" { next }
  {
    # Lowercase the entries column for matching
    entries = tolower($5)
    # Match if any entry has one of our include extensions (followed by | or end)
    if (entries ~ ("\\.(" ext ")(\\||$)")) {
      print $2 "\t" $4
    }
  }
' "$TSV" > "$PLAN"

plan_count=$(wc -l < "$PLAN")
echo "Archives matching filter: $plan_count"
echo ""

if [[ $plan_count -eq 0 ]]; then
  echo "Nothing to extract. Adjust --include-ext or check the TSV."
  rm -f "$PLAN"
  exit 0
fi
if [[ $plan_count -gt $LIMIT ]]; then
  echo "ABORT: plan size $plan_count exceeds --limit $LIMIT. Raise --limit to proceed."
  rm -f "$PLAN"
  exit 3
fi

if [[ $DRY_RUN -eq 1 ]]; then
  echo "=== DRY RUN — nothing extracted ==="
  echo "First 10 archives that would be extracted:"
  head -10 "$PLAN" | awk -F'\t' '{ printf "  %s (%s)\n", $1, $2 }'
  if [[ $plan_count -gt 10 ]]; then
    echo "  ... and $((plan_count - 10)) more"
  fi
  echo ""
  echo "To execute: re-run with --commit"
  rm -f "$PLAN"
  exit 0
fi

# Real extraction.
extract_one() {
  local f="$1"
  local fmt="$2"
  local base subdir
  base=$(basename "$f")
  # Strip the last extension to make the subfolder name
  subdir="$OUTPUT_DIR/${base%.*}"

  if [[ -d "$subdir" ]]; then
    echo "SKIP_EXISTS	$f"
    return 0
  fi

  mkdir -p "$subdir"
  case "$fmt" in
    zip)
      if unzip -q -o "$f" -d "$subdir" 2>/dev/null; then
        echo "OK	$f"
      else
        # Cleanup empty subdir on failure
        rmdir "$subdir" 2>/dev/null
        echo "FAIL	$f"
      fi
      ;;
    rar)
      if unrar x -inul -o+ "$f" "$subdir/" >/dev/null 2>&1; then
        echo "OK	$f"
      else
        rmdir "$subdir" 2>/dev/null
        echo "FAIL	$f"
      fi
      ;;
    7z|tar|gz|bz2|tgz|tbz|xz)
      if 7z x -y -o"$subdir" "$f" >/dev/null 2>&1; then
        echo "OK	$f"
      else
        rmdir "$subdir" 2>/dev/null
        echo "FAIL	$f"
      fi
      ;;
    *)
      rmdir "$subdir" 2>/dev/null
      echo "UNKNOWN_FORMAT	$f"
      ;;
  esac
}
export -f extract_one
export OUTPUT_DIR

echo "Extracting $plan_count archives. Transcript → $TRANSCRIPT"
echo "(this can take a while — each archive opens once)"
echo ""

xargs -d '\n' -n 1 -P "$WORKERS" -I {} bash -c '
  IFS=$'"'"'\t'"'"' read -r path fmt <<< "$1"
  extract_one "$path" "$fmt"
' _ {} < "$PLAN" | tee "$TRANSCRIPT"

echo ""
echo "Summary:"
awk -F'\t' '{count[$1]++} END {for (s in count) printf "  %-15s : %d\n", s, count[s]}' "$TRANSCRIPT" | sort

rm -f "$PLAN"
