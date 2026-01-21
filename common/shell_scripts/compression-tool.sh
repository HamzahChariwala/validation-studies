#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 -c|-d <folder>"
  echo "  -c   compress:  *.json -> *.json.gz (original .json removed)"
  echo "  -d   decompress: *.json.gz -> *.json (original .gz removed)"
  exit 1
}

if [[ $# -ne 2 ]]; then
  usage
fi

MODE="$1"
DIR="$2"

if [[ ! -d "$DIR" ]]; then
  echo "Error: '$DIR' is not a directory"
  exit 1
fi

case "$MODE" in
  -c|-d) ;;
  *) usage ;;
esac

# --- helpers ---
human_dir_size() {
  # Cross-platform-ish:
  # - GNU du: du -sb
  # - macOS/BSD du: no -b, use -sk and convert
  if du -sb "$1" >/dev/null 2>&1; then
    # bytes
    du -sb "$1" | awk '{print $1}'
  else
    # KiB -> bytes
    local kib
    kib=$(du -sk "$1" | awk '{print $1}')
    echo $((kib * 1024))
  fi
}

human_bytes() {
  # prints a human-friendly size (B, KiB, MiB, GiB, TiB)
  awk -v b="$1" 'BEGIN{
    split("B KiB MiB GiB TiB",u," ");
    i=1;
    while (b>=1024 && i<5) { b/=1024; i++ }
    printf "%.2f %s", b, u[i]
  }'
}

BEFORE_BYTES="$(human_dir_size "$DIR")"
echo "Folder: $DIR"
echo "Before: $(human_bytes "$BEFORE_BYTES")"

# --- operation ---
if [[ "$MODE" == "-c" ]]; then
  find "$DIR" -type f -name "*.json" ! -name "*.json.gz" -print0 |
  while IFS= read -r -d '' file; do
    echo "Compressing: $file"
    # gzip replaces file with file.gz (deletes original) unless -k is used
    gzip -f "$file"
  done
elif [[ "$MODE" == "-d" ]]; then
  find "$DIR" -type f -name "*.json.gz" -print0 |
  while IFS= read -r -d '' file; do
    echo "Decompressing: $file"
    # gunzip replaces file.gz with file (deletes .gz) unless -k is used
    gunzip -f "$file"
  done
fi

AFTER_BYTES="$(human_dir_size "$DIR")"
echo "After:  $(human_bytes "$AFTER_BYTES")"

# Show delta
if [[ "$AFTER_BYTES" -le "$BEFORE_BYTES" ]]; then
  SAVED=$((BEFORE_BYTES - AFTER_BYTES))
  echo "Change: -$(human_bytes "$SAVED")"
else
  GROWN=$((AFTER_BYTES - BEFORE_BYTES))
  echo "Change: +$(human_bytes "$GROWN")"
fi

echo "Done."

