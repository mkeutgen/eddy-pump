#!/usr/bin/env bash
# fetch_companion.sh — the companion paper's tables (the downward limb this study is paired with).
#
# reads  mkeutgen/globargo at a pinned commit, or its Zenodo record
# writes $GLOBARGO_DATA (default ~/Documents/globargo/data): seven CSVs, ~30 MB
# Letter item: every side-by-side comparison — Figures 2, 3, S5, S6, Text S19
# Phase: not a stage; run once, before the pipeline
#
# STUB. Prints the exact manual steps and exits 2 (a deliberate "not wired yet", distinct
# from 1 = misuse/failure).
# TODO(DOI): fetch from the Zenodo record rather than a local clone, verify against
#            data/companion/SHA256SUMS, and drop the "by hand" block.
set -euo pipefail
GLOBARGO_COMMIT="1c3330a"
GLOBARGO_DOI="10.5281/zenodo.17236580"
DEST="${GLOBARGO_DATA:-$HOME/Documents/globargo/data}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cat <<TXT
fetch_companion.sh is not wired to the record yet — do it by hand (two paths), then re-run preflight.

  source    github.com/mkeutgen/globargo @ $GLOBARGO_COMMIT   (Zenodo $GLOBARGO_DOI)
  dest      \$GLOBARGO_DATA (now: $DEST)
  needed    manually_verified_physical_subd_events.csv
            df_carbon_subduction_anom.csv
            df_carbon_subduction_anom_with_poc_fromgali.csv
            (+ the four the figure scripts read for the downward limb)

  A. FROM THE COMPANION REPO (or its Zenodo record):

       git clone --depth 1 https://github.com/mkeutgen/globargo /tmp/globargo
       git -C /tmp/globargo fetch --depth 1 origin $GLOBARGO_COMMIT && git -C /tmp/globargo checkout $GLOBARGO_COMMIT
       mkdir -p "$DEST"
       cp /tmp/globargo/data/*.csv "$DEST"/
       # no checksum file ships for these yet — TODO(DOI), see data/companion/README.md
       export GLOBARGO_DATA="$DEST"

  B. FROM AN EXISTING LOCAL CHECKOUT (what the fresh-machine acceptance run used).  Nothing is
     copied; the stages read \$GLOBARGO_DATA in place, read-only:

       export GLOBARGO_DATA=/path/to/globargo/data
       # e.g.  export GLOBARGO_DATA=~/Documents/globargo/data

  EITHER WAY, verify before starting the pipeline (and keep GLOBARGO_DATA exported for the run):

       ls "\$GLOBARGO_DATA"/manually_verified_physical_subd_events.csv
       bash "$REPO/production/run_all.sh" --preflight

  data/companion/companion_carbon_poc.csv is NOT fetched: it is a hand export of an .Rds this
  repo's Python cannot read, and it is vendored here with its provenance (data/companion/README.md).
TXT
exit 2
