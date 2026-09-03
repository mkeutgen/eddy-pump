# study.mk -- the net-carbon study's pipeline, written as make rules. Included by the Makefile.
#
# Read a rule as: "to make <target>, you need <inputs>; run <command>". `make -n rates` prints
# what would run without running it. Two kinds of target:
#   file targets   rebuilt only when an input is newer: features, scores, rates
#   explicit acts  never run on their own: freezing the candidate lists, drawing a batch, loading
#                  a labelled sheet
# Slow steps: verify-candidates / freeze-candidates ~36 min, features ~54 min. The rest: seconds.
# Targets: verify-candidates freeze-candidates features scores draw-batch review calibrate ingest rates
#
# The whole pipeline runs on the study's one label table (data/labels/study_reviews.parquet +
# study_batches.yaml + draws/*.yaml). verify-candidates / freeze-candidates / features need the
# bound fleet cache; the rest run from the label table and the saved lists.

SAVED := data/candidates/net_carbon_v1
OUT   := results/net_carbon_v1
AUD   := data/labels/audit
IDENT := data/candidates/net_carbon_v1/CACHE_IDENTITY.json
# The six saved candidate lists (full tables); every label points at a row of them.
POOLS := $(wildcard data/candidates/net_carbon_v1/*.parquet)
# The bound fleet cache, read from the identity file, so the labelling app draws panels from the
# same grids the candidates were detected on. Get it with scripts/fetch_caches.sh.
BOUND_CACHE = $(shell $(PY) -c 'import json;print(json.load(open("$(IDENT)"))["cache"]["path"])')

.PHONY: study-help verify-candidates freeze-candidates features scores draw-batch review calibrate ingest rates

study-help:
	@sed -n '2,9p' study.mk | sed 's/^# \{0,1\}//'

# 1. The candidate lists. Detect the six pools from the bound cache and compare with the saved
#    lists (the check); or write the saved lists (a deliberate act, never automatic).
verify-candidates:
	$(PY) pipeline/detect.py --verify
freeze-candidates:
	$(PY) pipeline/detect.py --write

# 2. Features for every candidate of the two physical pools. One run writes both files.
$(OUT)/features/physical_obduction.parquet: $(SAVED)/CANDIDATES.json $(IDENT)
	$(PY) pipeline/features.py
features: $(OUT)/features/physical_obduction.parquet

# 3. A score for every candidate: it decides the strata a batch is drawn from, never a number.
$(OUT)/scores/physical_obduction.parquet: $(OUT)/features/physical_obduction.parquet \
        data/external/manually_verified_physical_subd_events.csv
	$(PY) pipeline/scores.py
scores: $(OUT)/scores/physical_obduction.parquet

# 4. Draw the labelling batches and render their panels. A random draw: never automatic.
draw-batch: $(OUT)/scores/physical_obduction.parquet
	$(PY) pipeline/draw_batch.py --render

# 5. Label a batch, blind. POOL is read from the batch id (rate_obduction_01 -> physical_obduction).
review:
	@test -n "$(BATCH)" || (echo "usage: make review BATCH=results/net_carbon_v1/labeling/<id>/<id>.csv"; exit 1)
	ARGOPOD_RESIDUAL_CACHE=$(BOUND_CACHE) $(PY) -m argopod.cli review \
	  --config config/review/$(or $(POOL),physical_$(word 2,$(subst _, ,$(notdir $(basename $(BATCH)))))).yaml \
	  --batch $(BATCH)
#    Check a blind re-labelling of the 42 calibration panels against the frozen reference.
calibrate:
	@test -n "$(SHEET)" || (echo "usage: make calibrate SHEET=<re-labelled calibration sheet>"; exit 1)
	$(PY) pipeline/draw_batch.py --report $(SHEET)

# 6. Load a labelled sheet into the study's label table (refuses unless the calibration passed).
ingest:
	@test -n "$(BATCH)" || (echo "usage: make ingest BATCH=<batch id, e.g. rate_obduction_01>"; exit 1)
	$(PY) pipeline/ingest_batch.py $(BATCH)

# 7. The rate report: the rate per limb, its denominator and its error bar.
$(AUD)/rate_status.csv: data/labels/study_reviews.parquet $(wildcard data/labels/draws/*.yaml)
	$(PY) pipeline/rates.py
rates: $(AUD)/rate_status.csv
