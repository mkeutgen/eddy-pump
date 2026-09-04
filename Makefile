# eddy-pump — the net-carbon study of the submesoscale eddy pump.
#
# The pipeline is study.mk, one rule per step:
#   make study-help          the pipeline rules, in the order you run them
#   make -n rates            print what would run to refresh the rate report; `make rates` runs it
#   make test                the tests (also `make check`)
#   The labelling loop, calibration first:
#     make draw-batch BATCH=<name>       draw a sample and render its panels
#     make calibrate SHEET=<sheet.csv>   check the re-labelled 42 calibration panels; must pass first
#     make review BATCH=<sheet.csv>      label a sheet, blind
#     make load BATCH=<name>             load the labelled batch into the label table
#   make verify-candidates   re-detect the six pools and compare with the saved lists (~36 min)
#   make check-cache         rebuild four floats and compare their grids with the bound cache (~1 min)
#   make build-cache OUT=… RAW=…   build a fleet cache somewhere new (days)
#   make verify-cache OUT=…        read a built cache back and say whether it can be trusted
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
	@sed -n '2,17p' Makefile | sed 's/^# \{0,1\}//'

include study.mk

test:
	$(PY) -m pytest -q

check: test

fetch: fetch-caches fetch-companion

# The study fleet cache, from the data deposit. Before the DOI is published, point it at a local
# copy: DEPOSIT_DIR=/path/to/zenodo-study-deposit make fetch-caches
# Once it is on disk somewhere other than the path CACHE_IDENTITY.json records, set EDDY_PUMP_CACHE
# to that directory and every rule reads it there.
fetch-caches:
	bash scripts/fetch_caches.sh

fetch-companion:
	bash scripts/fetch_companion.sh
