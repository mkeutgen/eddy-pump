# eddy-pump — the net-carbon study of the submesoscale eddy pump.
#
# The pipeline is study.mk, one rule per step:
#   make study-help          the rules, in order
#   make -n rates            print what would run to refresh the rate report; `make rates` runs it
#   make verify-candidates   re-detect the six pools and compare with the saved lists (~36 min)
#   make draw-batch | review BATCH=... | calibrate SHEET=... | ingest BATCH=...   the labelling loop
#   make test                the tests (also `make check`)
#   make fetch-caches        the fleet cache from the data deposit (needed only to re-detect)
#   make fetch-companion     the earlier subduction study's tables ($GLOBARGO_DATA)
#
# Cut fresh from mkeutgen/eddy-pump-archive at tag archive-2026-09-03 (commit cbd6fd6). The archive
# holds the full history and the retired GRL letter (its own tag letter-v1); the data deposit holds
# the fleet cache and the old labels. See README.md and docs/PLAN.md.

# The targets that call python must all resolve the SAME interpreter. Precedence: an already-activated
# venv, then $OBDUCTION_VENV, then the default.
OBDUCTION_VENV ?= $(HOME)/Documents/release/.venv
export OBDUCTION_VENV
PY      ?= $(if $(VIRTUAL_ENV),$(VIRTUAL_ENV)/bin/python,$(OBDUCTION_VENV)/bin/python)

.PHONY: help test check fetch fetch-caches fetch-companion

help:
	@sed -n '2,12p' Makefile | sed 's/^# \{0,1\}//'

include study.mk

test:
	$(PY) -m pytest -q

check: test

fetch: fetch-caches fetch-companion

# The study fleet cache, from the data deposit. Before the DOI is published, point it at a local
# copy: DEPOSIT_DIR=/path/to/zenodo-study-deposit make fetch-caches
fetch-caches:
	bash scripts/fetch_caches.sh

fetch-companion:
	bash scripts/fetch_companion.sh
