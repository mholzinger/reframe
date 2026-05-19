#!/usr/bin/env bash
# Validate recovered video files via ffprobe (Tier 1 — header check)
# and optionally ffmpeg snippet decoding (Tier 2 — start/end frames).
#
# Tier 1 catches: corrupted MP4 atoms, missing duration, invalid codecs.
# Tier 2 catches: truncation, broken file edges.
# Neither catches mid-stream corruption — use ffmpeg null-decode on samples.
#
# Usage:
#   ./validate_videos.sh /path/to/videos                       # Tier 1 only
#   ./validate_videos.sh /path/to/videos --deep                # Tier 1 + 2
#   WORKERS=8 ./validate_videos.sh /path/to/videos             # tune parallelism
#
# Output: video_validation_<timestamp>.csv with columns
#   tier1_status,tier2_status,path,duration_sec,codec
# where tier1_status is OK|SUSPICIOUS|BROKEN
#   and tier2_status is OK|HEAD_FAIL|TAIL_FAIL|BOTH_FAIL|SKIPPED|"" (empty if not deep)

set -uo pipefail

INPUT_DIR="${1:-}"
DEEP=0
shift || true
for arg in "$@"; do
  case "$arg" in
    --deep) DEEP=1 ;;
    *) echo "Unknown arg: $arg" >&2; exit 2 ;;
  esac
done

if [[ -z "$INPUT_DIR" || ! -d "$INPUT_DIR" ]]; then
  echo "Usage: $0 INPUT_DIR [--deep]" >&2
  exit 2
fi

if ! command -v ffprobe >/dev/null; then
  echo "ffprobe not found. On Synology with Entware: sudo opkg install ffmpeg" >&2
  exit 1
fi
if [[ $DEEP -eq 1 ]] && ! command -v ffmpeg >/dev/null; then
  echo "--deep requires ffmpeg (sudo opkg install ffmpeg)" >&2
  exit 1
fi

WORKERS="${WORKERS:-4}"
TS="$(date +%Y%m%dT%H%M%S)"
OUT="video_validation_${TS}.csv"

echo "Validating videos under: $INPUT_DIR"
echo "Workers: $WORKERS  Deep mode: $DEEP"
echo "Output: $OUT"
echo ""

echo "tier1_status,tier2_status,path,duration_sec,codec" > "$OUT"

# Build the list of files first so we can show a count.
TMPLIST="$(mktemp)"
find "$INPUT_DIR" -type f \( \
    -iname "*.mp4" -o -iname "*.m4v" -o -iname "*.mkv" \
    -o -iname "*.mov" -o -iname "*.avi" -o -iname "*.webm" \
    \) -print0 2>/dev/null > "$TMPLIST"
total=$(tr -cd '\0' < "$TMPLIST" | wc -c)
echo "Found $total video files. Running Tier 1 (ffprobe)..."
echo ""

export DEEP

# Per-file worker.
process_one() {
  local f="$1"
  # Tier 1: ffprobe header + first video stream
  local info
  info=$(ffprobe -v error -select_streams v:0 \
    -show_entries stream=codec_name:format=duration \
    -of default=noprint_wrappers=1:nokey=1 "$f" 2>/dev/null)
  local t1 duration codec
  if [[ $? -ne 0 || -z "$info" ]]; then
    t1="BROKEN"; duration=""; codec=""
  else
    codec=$(echo "$info" | head -1)
    duration=$(echo "$info" | tail -1)
    if [[ -z "$duration" || "$duration" == "N/A" ]]; then
      t1="SUSPICIOUS"
    else
      t1="OK"
    fi
  fi

  local t2=""
  if [[ "${DEEP:-0}" == "1" && "$t1" == "OK" ]]; then
    # Tier 2: decode 3 seconds at start + last 3 seconds. Errors mean corruption.
    local head_ok=0 tail_ok=0
    ffmpeg -v error -nostdin -ss 0 -t 3 -i "$f" -f null - </dev/null >/dev/null 2>&1 && head_ok=1
    ffmpeg -v error -nostdin -sseof -3 -i "$f" -f null - </dev/null >/dev/null 2>&1 && tail_ok=1
    if [[ $head_ok == 1 && $tail_ok == 1 ]]; then
      t2="OK"
    elif [[ $head_ok == 0 && $tail_ok == 0 ]]; then
      t2="BOTH_FAIL"
    elif [[ $head_ok == 0 ]]; then
      t2="HEAD_FAIL"
    else
      t2="TAIL_FAIL"
    fi
  elif [[ "${DEEP:-0}" == "1" ]]; then
    t2="SKIPPED"
  fi

  # CSV-escape: wrap path in quotes; escape any quotes in path.
  local escaped_path="${f//\"/\"\"}"
  printf '%s,%s,"%s",%s,%s\n' "$t1" "$t2" "$escaped_path" "$duration" "$codec"
}
export -f process_one

xargs -0 -n 1 -P "$WORKERS" -I {} bash -c 'process_one "$1"' _ {} < "$TMPLIST" >> "$OUT"

rm -f "$TMPLIST"

echo ""
echo "Done. Summary:"
echo ""
awk -F',' 'NR>1 { t1[$1]++ } END { for (s in t1) printf "  Tier 1 %-12s : %d\n", s, t1[s] }' "$OUT" | sort
if [[ $DEEP -eq 1 ]]; then
  echo ""
  awk -F',' 'NR>1 && $2 != "" { t2[$2]++ } END { for (s in t2) printf "  Tier 2 %-12s : %d\n", s, t2[s] }' "$OUT" | sort
fi
echo ""
echo "CSV: $OUT"
echo ""
echo "Inspect:"
echo "  grep '^BROKEN' \"$OUT\"           # files with bad headers"
echo "  grep '^SUSPICIOUS' \"$OUT\"       # OK header but missing duration"
if [[ $DEEP -eq 1 ]]; then
  echo "  grep '^OK,\\(HEAD\\|TAIL\\|BOTH\\)_FAIL' \"$OUT\"  # decode failures"
fi
