#!/usr/bin/env bash
# Verify archive validity + sample contents for recovered ZIP/RAR/7Z files.
#
# For each archive:
#   - Reads the central directory (fast — doesn't extract or full-test)
#   - Captures status: VALID / EMPTY / BROKEN / UNKNOWN_FORMAT
#   - Lists the first 10 entry names (pipe-separated in the output TSV)
#
# This is the archive equivalent of the Tier 1 ffprobe pass on videos.
# It catches truncated archives, broken headers, missing central directories.
# It does NOT detect mid-archive corruption (would need full `unzip -t` etc.,
# which reads the entire file — minutes per GB).
#
# Usage:
#   ./verify_archives.sh INPUT_DIR [OUTPUT_TSV]
#   WORKERS=8 ./verify_archives.sh INPUT_DIR
#
# Output is TSV (tab-separated) so entry names containing commas survive intact:
#   status<TAB>path<TAB>size_mb<TAB>format<TAB>first_10_entries
# where first_10_entries are pipe-separated within their column.
#
# Required tools (install via Entware):
#   sudo opkg install unzip unrar p7zip

set -uo pipefail

INPUT_DIR="${1:-}"
OUT="${2:-archive_verification_$(date +%Y%m%dT%H%M%S).tsv}"
WORKERS="${WORKERS:-4}"

if [[ -z "$INPUT_DIR" || ! -d "$INPUT_DIR" ]]; then
  echo "Usage: $0 INPUT_DIR [OUTPUT_TSV]" >&2
  exit 2
fi

# Check available tools — warn about missing ones but don't fail.
HAVE_UNZIP=0; HAVE_UNRAR=0; HAVE_7Z=0
command -v unzip  >/dev/null 2>&1 && HAVE_UNZIP=1
command -v unrar  >/dev/null 2>&1 && HAVE_UNRAR=1
command -v 7z     >/dev/null 2>&1 && HAVE_7Z=1
echo "Tools: unzip=$HAVE_UNZIP unrar=$HAVE_UNRAR 7z=$HAVE_7Z  (install missing with 'opkg install unzip unrar p7zip')"
echo "Workers: $WORKERS"
echo "Output: $OUT"
echo ""

verify_one() {
  local f="$1"
  local ext="${f##*.}"
  local lower
  lower=$(echo "$ext" | tr '[:upper:]' '[:lower:]')
  local size_mb status entries
  size_mb=$(stat -c '%s' "$f" 2>/dev/null | awk '{printf "%.1f", $1/1048576}')

  case "$lower" in
    zip)
      if command -v unzip >/dev/null; then
        entries=$(unzip -Z1 "$f" 2>/dev/null | head -10 | tr '\n' '|' | sed 's/|$//')
        rc=$?
        if [[ $rc -eq 0 && -n "$entries" ]]; then status="VALID"
        elif [[ $rc -eq 0 ]]; then status="EMPTY"
        else status="BROKEN"; entries=""; fi
      else
        status="NO_TOOL"; entries=""
      fi
      ;;
    rar)
      if command -v unrar >/dev/null; then
        entries=$(unrar lb "$f" 2>/dev/null | head -10 | tr '\n' '|' | sed 's/|$//')
        rc=$?
        if [[ $rc -eq 0 && -n "$entries" ]]; then status="VALID"
        elif [[ $rc -eq 0 ]]; then status="EMPTY"
        else status="BROKEN"; entries=""; fi
      else
        status="NO_TOOL"; entries=""
      fi
      ;;
    7z|tar|gz|bz2|tgz|tbz|xz)
      if command -v 7z >/dev/null; then
        entries=$(7z l -ba -slt "$f" 2>/dev/null | grep '^Path = ' | head -10 | sed 's/^Path = //' | tr '\n' '|' | sed 's/|$//')
        rc=$?
        if [[ $rc -eq 0 && -n "$entries" ]]; then status="VALID"
        elif [[ $rc -eq 0 ]]; then status="EMPTY"
        else status="BROKEN"; entries=""; fi
      else
        status="NO_TOOL"; entries=""
      fi
      ;;
    *)
      status="UNKNOWN_FORMAT"; entries=""
      ;;
  esac

  printf '%s\t%s\t%s\t%s\t%s\n' "$status" "$f" "$size_mb" "$lower" "$entries"
}
export -f verify_one

# Header for the TSV
printf 'status\tpath\tsize_mb\tformat\tfirst_10_entries\n' > "$OUT"

# Build file list (so we can show a count)
TMPLIST=$(mktemp)
find "$INPUT_DIR" -type f \( \
    -iname "*.zip" -o -iname "*.rar" -o -iname "*.7z" \
    -o -iname "*.tar" -o -iname "*.gz" -o -iname "*.bz2" \
    -o -iname "*.tgz" -o -iname "*.tbz" -o -iname "*.xz" \
    \) -print0 2>/dev/null > "$TMPLIST"
total=$(tr -cd '\0' < "$TMPLIST" | wc -c)
echo "Found $total archive files. Validating..."
echo ""

xargs -0 -n 1 -P "$WORKERS" -I {} bash -c 'verify_one "$1"' _ {} < "$TMPLIST" >> "$OUT" 2>/dev/null

rm -f "$TMPLIST"

echo ""
echo "Summary:"
awk -F'\t' 'NR>1 { count[$1]++ } END { for (s in count) printf "  %-15s : %d\n", s, count[s] }' "$OUT" | sort

echo ""
echo "Inspect:"
echo "  grep -P '^VALID\\t' $OUT | head -20            # see what's in valid archives"
echo "  grep -P '^BROKEN\\t' $OUT | wc -l              # count corrupted"
echo "  awk -F'\\t' '\$1==\"VALID\" {print \$5}' $OUT | head -50  # peek at contents"
