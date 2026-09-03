#!/usr/bin/env bash
# fetch_companion.sh — the earlier subduction study's (the companion's) tables.
#
# reads  mkeutgen/globargo at a pinned commit, or its Zenodo record
# writes $GLOBARGO_DATA (default ~/Documents/globargo/data)
# The study's downward classifier trains on the companion's reviewed detections; these are training
# evidence only, never a rate. The two files the pipeline reads:
#   detected_physical_subd_events.csv                 (scores.py: the companion's R-detections)
#   manually_verified_physical_subd_events.csv        (also vendored at data/external/ as reference events)
#
# STUB. Prints the manual steps and exits 2 (a deliberate "not wired yet", distinct from 1 = failure).
# TODO(DOI): fetch from the Zenodo record and verify a checksum, then drop the "by hand" block.
set -euo pipefail
GLOBARGO_COMMIT="1c3330a"
GLOBARGO_DOI="10.5281/zenodo.17236580"
DEST="${GLOBARGO_DATA:-$HOME/Documents/globargo/data}"

cat <<TXT
fetch_companion.sh is not wired to the record yet — do it by hand (two paths), then re-run the tests.

  source    github.com/mkeutgen/globargo @ $GLOBARGO_COMMIT   (Zenodo $GLOBARGO_DOI)
  dest      \$GLOBARGO_DATA (now: $DEST)
  needed    detected_physical_subd_events.csv
            manually_verified_physical_subd_events.csv

  A. FROM THE COMPANION REPO (or its Zenodo record):

       git clone --depth 1 https://github.com/mkeutgen/globargo /tmp/globargo
       git -C /tmp/globargo fetch --depth 1 origin $GLOBARGO_COMMIT && git -C /tmp/globargo checkout $GLOBARGO_COMMIT
       mkdir -p "$DEST"
       cp /tmp/globargo/data/*.csv "$DEST"/
       export GLOBARGO_DATA="$DEST"

  B. FROM AN EXISTING LOCAL CHECKOUT. Nothing is copied; scores.py reads \$GLOBARGO_DATA in place:

       export GLOBARGO_DATA=/path/to/globargo/data
       # e.g.  export GLOBARGO_DATA=~/Documents/globargo/data

  Verify (and keep GLOBARGO_DATA exported for the run):

       ls "\$GLOBARGO_DATA"/detected_physical_subd_events.csv
TXT
exit 2
