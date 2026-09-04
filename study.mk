# study.mk -- the net-carbon study's pipeline, written as make rules. Included by the Makefile.
#
# Read a rule as: "to make <target>, you need <inputs>; run <command>". `make -n rates` prints what
# would run without running it. Two kinds of target: file targets, rebuilt only when an input is
# newer (features, scores, rates), and deliberate acts, which never run on their own (writing the
# saved candidate lists, drawing a batch, loading a labelled sheet, building the fleet cache).
#
# The labelling loop, in the order you run it. The calibration comes first: re-label a fresh blind
# copy of the 42 calibration panels and pass, then label the batch, then load it. The loader looks
# for a calibration that passed and was finished before the batch sheet, and refuses otherwise.
#   make draw-batch BATCH=<name>       draw the sample and render its panels
#                                      (`draw_batch.py --list` prints the batch names it knows)
#   make calibrate SHEET=<sheet.csv>   check the re-labelled 42 panels; must PASS, and must be first
#   make review BATCH=<sheet.csv>      label a sheet, blind (the calibration copy, then the batch)
#   make load BATCH=<name>             load the labelled batch; refuses unless the calibration passed
#   make rates                         the rate per limb, its denominator and its error bar
# The rest: verify-candidates, freeze-candidates, features, scores, build-cache, verify-cache,
#   check-cache. Slow: verify-candidates / freeze-candidates ~36 min, features ~54 min, build-cache
#   days. Everything else takes seconds.
# A cache build is a deliberate act, never automatic: nothing depends on build-cache, and it refuses
#   to write into the cache the saved candidate lists are bound to.
#
# The whole pipeline runs on the study's one label table (data/labels/study_reviews.parquet +
# study_batches.yaml + draws/*.yaml). verify-candidates / freeze-candidates / features need the
# bound fleet cache; the rest run from the label table and the saved lists.

SAVED := data/candidates/net_carbon_v1
RESULTS := results/net_carbon_v1
AUD   := data/labels/audit
IDENT := data/candidates/net_carbon_v1/CACHE_IDENTITY.json
# The six saved candidate lists (full tables); every label points at a row of them.
POOLS := $(wildcard data/candidates/net_carbon_v1/*.parquet)
# The bound fleet cache, read from the identity file, so the labelling app draws panels from the
# same grids the candidates were detected on. Get it with scripts/fetch_caches.sh. The identity
# file records one machine's absolute path; on any other machine set EDDY_PUMP_CACHE to where the
# cache actually sits and it is used instead.
BOUND_CACHE = $(if $(EDDY_PUMP_CACHE),$(EDDY_PUMP_CACHE),$(shell $(PY) -c 'import json;print(json.load(open("$(IDENT)"))["cache"]["path"])'))

.PHONY: study-help verify-candidates freeze-candidates features scores draw-batch review calibrate load rates build-cache verify-cache check-cache

study-help:
	@sed -n '2,21p' study.mk | sed 's/^# \{0,1\}//'

# 1. The candidate lists. Detect the six pools from the bound cache and compare with the saved
#    lists (the check); or write the saved lists (a deliberate act, never automatic).
verify-candidates:
	$(PY) pipeline/detect.py --verify
freeze-candidates:
	$(PY) pipeline/detect.py --write

# 2. Features for every candidate of the two physical pools. One run writes both files.
$(RESULTS)/features/physical_obduction.parquet: $(SAVED)/CANDIDATES.json $(IDENT)
	$(PY) pipeline/features.py
features: $(RESULTS)/features/physical_obduction.parquet

# 3. A score for every candidate: it decides the strata a batch is drawn from, never a number.
$(RESULTS)/scores/physical_obduction.parquet: $(RESULTS)/features/physical_obduction.parquet \
        data/external/manually_verified_physical_subd_events.csv
	$(PY) pipeline/scores.py
scores: $(RESULTS)/scores/physical_obduction.parquet

# 4. Draw a labelling batch and render its panels. A random draw: never automatic. BATCH is the
#    name the batch will have, e.g. rate_obduction_02.
draw-batch: $(RESULTS)/scores/physical_obduction.parquet
	@test -n "$(BATCH)" || (echo "usage: make draw-batch BATCH=<batch name, e.g. rate_obduction_02>"; exit 1)
	$(PY) pipeline/draw_batch.py $(BATCH) --render

# 5. The calibration, first in every session: check a fresh blind re-labelling of the 42 calibration
#    panels against the saved answers. It must say PASS before any batch panel is labelled.
calibrate:
	@test -n "$(SHEET)" || (echo "usage: make calibrate SHEET=<the re-labelled calibration sheet>"; exit 1)
	$(PY) pipeline/draw_batch.py --report $(SHEET)

# 6. Label a sheet, blind — the calibration copy first, then the batch. POOL is read from the batch
#    name (rate_obduction_01 -> physical_obduction).
review:
	@test -n "$(BATCH)" || (echo "usage: make review BATCH=results/net_carbon_v1/labeling/<name>/<name>.csv"; exit 1)
	ARGOPOD_RESIDUAL_CACHE=$(BOUND_CACHE) $(PY) -m argopod.cli review \
	  --config config/review/$(or $(POOL),physical_$(word 2,$(subst _, ,$(notdir $(basename $(BATCH)))))).yaml \
	  --batch $(BATCH)

# 7. Load a labelled sheet into the study's label table. It refuses unless a calibration of the same
#    pool passed and was finished before this sheet.
load:
	@test -n "$(BATCH)" || (echo "usage: make load BATCH=<batch name, e.g. rate_obduction_01>"; exit 1)
	$(PY) pipeline/load_batch.py $(BATCH)

# 8. The rate report: the rate per limb, its denominator and its error bar.
$(AUD)/rate_status.csv: data/labels/study_reviews.parquet $(wildcard data/labels/draws/*.yaml)
	$(PY) pipeline/rates.py
rates: $(AUD)/rate_status.csv

# 9. The fleet cache: build it, read it back, or check a rebuild against the bound one. Never
#    automatic — no other rule needs any of these, and a build refuses the bound cache's own
#    directory, because rewriting those grids would move the fingerprint every number stands on.
#    The recipe is config/events.yaml, the float list config/fleet.csv.
build-cache:
	@test -n "$(OUT)" -a -n "$(RAW)" || (echo "usage: make build-cache OUT=<new cache dir> RAW=<staged raw frames> [WORKERS=n] [TIER=n] [LIMIT=n] [RESUME=1]"; exit 1)
	$(PY) pipeline/build_cache.py --out $(OUT) --raw $(RAW) \
	  $(if $(WORKERS),--workers $(WORKERS)) $(if $(TIER),--tier $(TIER)) \
	  $(if $(LIMIT),--limit $(LIMIT)) $(if $(RESUME),--resume)
verify-cache:
	@test -n "$(OUT)" || (echo "usage: make verify-cache OUT=<cache dir>"; exit 1)
	$(PY) pipeline/build_cache.py --verify --out $(OUT)
#    Rebuild four floats, one per grid kind, and compare every grid byte for byte (~1 min).
#    One list, not four arguments: $(or ...) takes the first non-empty of TWO things.
CHECK_WMOS := 2901074,1901339,1901378,6903247
CHECK_RAW  := $(HOME)/Documents/release/cache_paper/raw
check-cache:
	$(PY) pipeline/build_cache.py --raw $(or $(RAW),$(CHECK_RAW)) --check $(or $(WMO),$(CHECK_WMOS))
