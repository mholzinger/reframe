#!/usr/bin/env bash
# Tier 0 council check: sample files that the SKIP_FILENAME_PATTERNS +
# SKIP_FILENAME_MAX_SIZE filter would drop, and look for any sign that
# they're real camera photos (camera EXIF, large dimensions) rather than
# 4K Stogram / Instagram archive captures.
#
# Usage:
#   ./verify_skipped.sh [DUMP_PATH] [SAMPLE_SIZE] [MAX_SIZE_KB]
#
# Defaults match the running config on the NAS.
#
# Verdict the user wants:
#   - ALL sampled files small (<200KB), no Make/Model EXIF, 300/750/1080 dims → filter is safe.
#   - ANY sampled file with Make/Model EXIF or non-Instagram dimensions → STOP, redesign the filter.

set -euo pipefail

DUMP_PATH="${1:-/volume1/_Backups/4TDrive/EaseUS 03-24 0934}"
SAMPLE_SIZE="${2:-50}"
MAX_SIZE_KB="${3:-500}"
OUT="/tmp/verify_skipped_$(date +%Y%m%dT%H%M%S).txt"

if ! command -v exiftool >/dev/null 2>&1; then
  echo "exiftool not installed. On Synology with Entware: opkg install exiftool" >&2
  exit 1
fi

if [[ ! -d "$DUMP_PATH" ]]; then
  echo "Dump path does not exist: $DUMP_PATH" >&2
  exit 1
fi

echo "Sampling $SAMPLE_SIZE files from $DUMP_PATH"
echo "Filter: name matches FILE[0-9]+\\.JPG AND size <= ${MAX_SIZE_KB}KB"
echo "Output: $OUT"
echo ""

find "$DUMP_PATH" -type f -name 'FILE[0-9]*.JPG' -size -"${MAX_SIZE_KB}"k 2>/dev/null \
  | shuf -n "$SAMPLE_SIZE" > /tmp/_sample_paths.txt

if [[ ! -s /tmp/_sample_paths.txt ]]; then
  echo "No files matched. Either the dump is empty or no files fit the filter." >&2
  exit 0
fi

while IFS= read -r f; do
  echo "=== $f ==="
  exiftool -s -ImageSize -FileSize -Make -Model -SerialNumber -Software -DateTimeOriginal -LensModel "$f" 2>/dev/null \
    | grep -v '^$' || true
  echo ""
done < /tmp/_sample_paths.txt | tee "$OUT" >/dev/null

# --- Verdict summary ---
total=$(wc -l < /tmp/_sample_paths.txt)
with_make=$(grep -c '^Make' "$OUT" || true)
with_model=$(grep -c '^Model' "$OUT" || true)
with_serial=$(grep -c '^SerialNumber' "$OUT" || true)
with_lens=$(grep -c '^LensModel' "$OUT" || true)

# Instagram-typical dimensions: 300x300, 1080x1080, 1080x1349, 750x750, 640x640, 150x300, 151x300, 1080x2220, 1080x718
ig_dims=$(grep -cE '^ImageSize *: *(300x300|1080x1080|1080x1349|750x750|640x640|150x300|151x300|1080x2220|1080x718|1080x2144)' "$OUT" || true)
non_ig_dims=$(grep -cE '^ImageSize' "$OUT" | xargs -I {} echo "{}-${ig_dims}" | bc 2>/dev/null || echo "?")

echo "============================================"
echo "VERIFICATION SUMMARY"
echo "============================================"
echo "Sampled files       : $total"
echo "With Make EXIF      : $with_make   ${with_make:+ ← suspicious if > 0}"
echo "With Model EXIF     : $with_model   ${with_model:+ ← suspicious if > 0}"
echo "With SerialNumber   : $with_serial   ${with_serial:+ ← VERY suspicious if > 0}"
echo "With LensModel      : $with_lens   ${with_lens:+ ← VERY suspicious if > 0}"
echo "Instagram-typical dims : $ig_dims of $total"
echo "Non-Instagram dims     : $non_ig_dims"
echo ""
if [[ "$with_make" -gt 0 ]] || [[ "$with_serial" -gt 0 ]] || [[ "$with_lens" -gt 0 ]]; then
  echo "VERDICT: STOP. At least one skipped file looks like a real camera photo."
  echo "         The filter is silently throwing away real photos. Investigate $OUT."
  exit 2
elif [[ "$ig_dims" -lt $((total * 80 / 100)) ]]; then
  echo "VERDICT: CAUTION. Less than 80% have Instagram-typical dimensions."
  echo "         Eyeball $OUT to confirm the others aren't real photos."
  exit 1
else
  echo "VERDICT: SAFE. Filter behavior looks correct — all sampled files match"
  echo "         the Instagram archive profile (no camera EXIF, IG dimensions)."
  exit 0
fi
