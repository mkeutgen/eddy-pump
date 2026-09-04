#!/usr/bin/env bash
# fetch_caches.sh — put the study fleet cache (and, optionally, the old labels) on this machine.
#
# Belongs in the eddy-pump repository at scripts/fetch_caches.sh. It downloads the data deposit,
# verifies every file against the deposit's own SHA256SUMS, and extracts the fleet cache so that
# `make verify-candidates` can re-detect the six candidate pools.
#
# The repository ships the 27 MB of saved candidate lists in git, so it computes the study's rates
# WITHOUT this cache. You need the cache only to re-detect the candidates from the grids, and the
# --extras (old labels) only to train the classifier on more panels.
#
# Two sources, in priority order:
#   1. DEPOSIT_DIR=/path/to/zenodo-study-deposit   — a local copy (what to use before the DOI exists)
#   2. ZENODO_RECORD=<numeric id>                  — download from the published Zenodo record
#
# Destination:
#   CACHE_DIR=/path   where residual_cache_v4/ is written (default: $HOME/eddy-pump-cache)
#
# The identity file in the repository records the absolute path of the machine the cache was built
# on, which is not this machine's path. So the last thing this script prints is the one line to
# export -- EDDY_PUMP_CACHE -- which tells every study tool where the grids are here. The
# fingerprint still decides whether they are the right grids.
#
# Flags:
#   --extras     also extract old_labels/, letter_v1_candidates/ and labels_raw (classifier + provenance)
#   --keep       keep the downloaded tarballs after extracting (default: remove them)
#
# Exit codes: 0 ok · 1 failure/misuse · 2 not wired to a source yet.
set -euo pipefail

ZENODO_RECORD="${ZENODO_RECORD:-}"      # RESERVED — set to the published record id when the DOI is live
CACHE_DIR="${CACHE_DIR:-$HOME/eddy-pump-cache}"
FINGERPRINT="11fce215c7db6cebabfbb0d233d17ed7b6674e388bcfb5b91153449ec35f8299"

FILES=(fleet_cache_residual_v4.tar.zst old_labels.tar.zst labels_raw.tar letter_v1_candidates.tar.zst fleet_manifest.csv SHA256SUMS)

want_extras=0; keep=0
for a in "$@"; do
  case "$a" in
    --extras) want_extras=1 ;;
    --keep)   keep=1 ;;
    *) echo "unknown flag: $a" >&2; exit 1 ;;
  esac
done

msg(){ printf '%s\n' "$*" >&2; }

# 1. Get the deposit into a working directory ($WORK holds the tarballs + SHA256SUMS).
if [[ -n "${DEPOSIT_DIR:-}" ]]; then
  [[ -f "$DEPOSIT_DIR/SHA256SUMS" ]] || { msg "DEPOSIT_DIR=$DEPOSIT_DIR has no SHA256SUMS"; exit 1; }
  WORK="$DEPOSIT_DIR"; downloaded=0
  msg "source: local deposit $WORK"
elif [[ -n "$ZENODO_RECORD" ]]; then
  WORK="$(mktemp -d)"; downloaded=1
  base="https://zenodo.org/records/$ZENODO_RECORD/files"
  msg "source: Zenodo record $ZENODO_RECORD -> $WORK"
  for f in "${FILES[@]}"; do
    msg "  downloading $f"
    curl -fL --retry 3 -o "$WORK/$f" "$base/$f?download=1"
  done
else
  cat >&2 <<TXT
fetch_caches.sh is not wired to a source yet.

  Before the DOI is published, point it at a local copy of the deposit:
      DEPOSIT_DIR=/path/to/zenodo-study-deposit scripts/fetch_caches.sh

  After it is published, set the record id:
      ZENODO_RECORD=1234567 scripts/fetch_caches.sh

  It writes residual_cache_v4/ under CACHE_DIR (now: $CACHE_DIR).
TXT
  exit 2
fi

# 2. Verify. Check only the files we will use, against the deposit's SHA256SUMS.
msg "verifying against SHA256SUMS ..."
check=(fleet_cache_residual_v4.tar.zst)
[[ $want_extras -eq 1 ]] && check+=(old_labels.tar.zst labels_raw.tar letter_v1_candidates.tar.zst)
( cd "$WORK" && grep -E "  ($(IFS='|'; echo "${check[*]}"))\$" SHA256SUMS | shasum -a 256 -c - )

# 3. Extract the fleet cache.
mkdir -p "$CACHE_DIR"
msg "extracting the fleet cache into $CACHE_DIR ..."
zstd -dc "$WORK/fleet_cache_residual_v4.tar.zst" | tar -x -C "$CACHE_DIR"
grids="$(find "$CACHE_DIR/residual_cache_v4" -name '*_fine.parquet' | wc -l | tr -d ' ')"
msg "  extracted $grids fine grids (expected 2542)"

# 4. Optional extras.
if [[ $want_extras -eq 1 ]]; then
  ex="$CACHE_DIR/deposit_extras"; mkdir -p "$ex"
  msg "extracting the old labels and candidate tables into $ex ..."
  zstd -dc "$WORK/old_labels.tar.zst"           | tar -x -C "$ex"
  zstd -dc "$WORK/letter_v1_candidates.tar.zst" | tar -x -C "$ex"
  tar -x -f "$WORK/labels_raw.tar" -C "$ex"
  cp "$WORK/fleet_manifest.csv" "$ex/"
fi

# 5. Clean up downloads.
if [[ ${downloaded:-0} -eq 1 && $keep -eq 0 ]]; then rm -rf "$WORK"; fi

cat >&2 <<TXT

done. the fleet cache is at $CACHE_DIR/residual_cache_v4

  tell the study where it is -- put this in your shell profile:

      export EDDY_PUMP_CACHE=$CACHE_DIR/residual_cache_v4

  science check (the fingerprint the study refuses to run without):
      fine grids           2542
      fine_grids_sha256    $FINGERPRINT
  confirm by re-detecting the six pools from the extracted grids:
      EDDY_PUMP_CACHE=$CACHE_DIR/residual_cache_v4 make verify-candidates
TXT
