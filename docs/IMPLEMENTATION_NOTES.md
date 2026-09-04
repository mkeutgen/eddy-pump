# IMPLEMENTATION_NOTES — how the machine works

How the study is built and what every file and column means. The plan is `docs/PLAN.md`; the
decisions in force are `docs/DECISIONS.md`.

## The words

Read this first. Every other document uses these words and only these.

- **event** — a real water-mass injection at one depth level of one profile. A candidate becomes an
  event when a human looks at its panel and says yes.
- **candidate** — a level the detector flagged. Its key is `(WMO, CYCLE_NUMBER,
  round(PRES_ADJUSTED))` plus the pool. Most candidates are not events; that is the point.
- **pool** — one of the six event definitions, two limbs times three channels. A tracer pool is its
  directional parent plus one tracer term, so its candidates are a subset of the parent's.
- **limb** — subduction, water pushed down, or obduction, water lifted up.
- **channel** — physical, nitrate or carbon. It says which tracer term the pool adds on top of AOU
  and absolute salinity.
- **grid kind** — which channels a float's cached grid carries: `paper_phys`, `paper_bbp`,
  `paper_nit` or `paper_all`. The `cache:` block of `config/events.yaml` lists the four and their
  channels; `config/fleet.csv` says which one each float was promised.
- **the bound cache** — the one set of residual grids the saved candidate lists were detected on:
  2,542 fine grids, fingerprint `11fce215…`. `data/candidates/net_carbon_v1/CACHE_IDENTITY.json`
  names it and the code refuses any other cache. That file records one machine's absolute path; on
  another machine set `EDDY_PUMP_CACHE` to where the cache actually sits.
- **fingerprint** — a sha256 that names a thing so it cannot change quietly. The cache fingerprint
  hashes the grid file *names*, not their contents; `make check-cache` is what proves the contents.
- **open region** — the part of the upward pool that was sampled at random: 171,578 of its 186,275
  candidate levels. The upward rate is measured on this part alone.
- **held region** — the other 14,697 levels of the upward pool. They were picked earlier by the
  retired letter's stricter 1.96 σ cut, so no random sample covers them. They wait for a draw of
  their own under this study's criterion.
- **calibration set** — 42 panels per limb whose answers are saved. Before every labelling session
  the labeller re-labels a fresh blind copy and must pass.
- **batch** — one drawn sample of candidates, with its rendered panels and its worksheet, the `.csv`
  a labeller fills in. The make rules call that file `SHEET=`; it is the batch's worksheet, not a
  separate thing.
- **inclusion probability (π)** — the chance a candidate had of being drawn, π = n/N inside its
  stratum. Every label enters a rate weighted by 1/π.
- **stratum** — one of the ten score bands a pool is cut into before a draw. The score decides which
  stratum a panel came from and nothing else.
- **denominator** — the candidate levels a rate is a fraction of. Every rate names its own.
- **rate** — the weighted fraction of a pool's candidates that a human called real, from a
  probability sample, with a denominator and an error bar.
- **spec id** — a short hash of a pool's channels and detector settings, e.g. `v1:31619ba8…`.
  `config/events.yaml` records one per pool. The loader refuses a pool that no longer hashes to its
  id, because every label and every saved list is keyed to it.

## Where things are

| path | what |
|---|---|
| `config/events.yaml` | the one source: the six event definitions, how the fleet cache is built (the `cache:` block, the ranges, the nine floats left out), the recorded spec ids |
| `config/fleet.csv` | the fleet-cache build list: 2,574 floats, the grid kind each is promised, and the wave it was built in. Row for row the bound cache's own record |
| `config/criteria.yaml` | the criterion a human labels against (four clauses) and the two calibration sets |
| `config/review/*.yaml` | the two panel layouts for the labelling app |
| `data/candidates/net_carbon_v1/` | the six saved lists (full Parquet tables), a `.json` sidecar per list, `CANDIDATES.json` (the six in one page), `CACHE_IDENTITY.json` (the bound cache) |
| `data/labels/study_reviews.parquet`, `study_batches.yaml`, `draws/` | the study's own labels: the batches, the reviews, the draw records with every inclusion probability, the saved calibration answers |
| `data/external/` | the earlier subduction study's verified events (they train the downward classifier and act as reference events) and the upward calibration reference; the earlier study's detection table that names the held region (`letter_pool_features.parquet`, hash checked) |
| `src/eddy_pump/` | the code (below) |
| `pipeline/` | the seven pipeline scripts |
| `tests/` | the tests and the numbers they share (`pins.yaml`) |

## The code (`src/eddy_pump/`)

- `vocabulary.py` — the fixed vocabulary: the channels, the two limbs, the tracers.
- `spec.py` — the event spec objects and the declared field lists that `spec_id` hashes.
- `study.py` — the study object: the pools, the cache identity, the excluded floats, the output
  policy, and `cache_policy()` — the whole fleet-cache recipe, as argopod consumes it.
- `manifest.py` — `load_manifest()`: reads `config/events.yaml` into the study object; `REPO_ROOT`.
- `candidates.py` — the detector wrapper: it detects a pool from the bound cache and saves or reads
  a list. `content_hash()` is a pool's identity, the sha256 of its sorted key triples.
  `verify_saved()` compares a fresh detection with the saved list.
- `batches.py` — the batch design: the score-stratified uniform draw and its inclusion probabilities.
- `labels.py` — the study's one label table (query API; the only door a rate may use).
- `criteria.py` — loads `config/criteria.yaml`; the criterion in force.
- `classifier.py` — the triage model: train, evaluate, score, calibrate. It orders panels and never
  enters a number.
- `domain.py` — a re-export shim (removed in the "tests and documents" step of the plan).

## The pipeline (`pipeline/`, run through `study.mk`)

1. `build_cache.py` — build or check the fleet cache. See below; it is never part of a run.
2. `detect.py` — detect or verify the six pools.
3. `features.py` — a feature row per candidate.
4. `scores.py` — a triage score per candidate.
5. `draw_batch.py <batch> --render` — draw a probability sample and render its panels.
6. `argopod.cli review` — a human labels the panels, blind.
7. `load_batch.py <batch>` — load a labelled sheet, once the calibration has passed.
8. `rates.py` — the rate per limb, its denominator and its error bar.

**The order inside a session matters.** The calibration comes first: a fresh blind copy of the 42
panels (`python pipeline/draw_batch.py --repass calib_<limb>_…`), labelled, then checked with
`make calibrate SHEET=…`. Only then is the batch labelled. `load_batch.py` looks for a calibration
of the same pool that passed and was finished *before* the batch sheet, and refuses otherwise.

`make study-help` prints the rules. Slow steps: `verify-candidates` ~36 min, `features` ~54 min.

## Rebuilding the fleet cache

The cache is one pair of residual grids per float, a fine grid and a coarse one. It is the expensive
half of everything. Each float is downloaded once, its derived channels computed once, its levels
binned once. Every detection afterwards just reads the result. The bound cache is 2,542 fine grids,
fingerprint `11fce215…`.

**Who does what.** The building is generic and lives in argopod (`argopod.cache`, v0.5.2): a float
list plus a recipe in, a directory of grids out. The study owns the recipe and the list.

**Where each part of the recipe comes from**, all of it in `config/events.yaml`:

| part | where it is written |
|---|---|
| the four grid kinds and the channels each carries | the `cache:` block, its grid-kind map |
| the dates kept, 2009-01-08 to 2026-03-15 | the `cache:` block, `window:` |
| a placeholder value becomes "no reading" before anything is derived from it | the `cache:` block, `fill_policy: mask` |
| a cycle with no delayed-mode column reads the raw one, that cycle only | the `cache:` block, `adjusted_fallback: cycle` |
| the ceilings `make verify-cache` uses | the `cache:` block, `residual_ceilings:` |
| the plausible ranges on the columns floats arrive with | `raw_inputs:` |
| the plausible ranges on the channels the study detects on | each channel's own `prefilter:` |
| the backscatter smoother | `tracers.carbon.prefilter.pre_median_filter` |
| the nine floats left out, each with its reason | `excluded_floats:` |

`Study.cache_policy()` joins them into the one object argopod builds from, so nothing is written
down twice. The bin widths, the smoothing window and the column names are argopod's own defaults,
and the bound cache was built under them. The one detector knob this study moves acts when
candidates are found, never when a grid is filled. `tests/test_cache_policy.py` names all fifteen
of those defaults and fails if one drifts.

**The float list** is `config/fleet.csv`: one row per float for 2,574 floats, each with the grid
kind it is promised. The counts are 1,181 `paper_phys`, 953 `paper_all`, 382 `paper_bbp` and 58
`paper_nit`, built in four waves. It matches the bound cache's own `MANIFEST.csv` row for row.

**The raw frames.** `~/Documents/release/cache_paper/raw` holds 2,559 staged downloads, 16 GB, one
parquet per float. They live on this machine only — they are not in the data deposit and not in git.
A float with no staged frame is downloaded during the build. `ARGOPOD_SKIP_ERDDAP=1` sends those
downloads straight to the GDAC files instead of trying the ERDDAP endpoint first. That saves minutes
per float when the endpoint is slow.

**The commands** (`pipeline/build_cache.py` behind all three):

```bash
make build-cache OUT=<new dir> RAW=~/Documents/release/cache_paper/raw   # add WORKERS= TIER= LIMIT= RESUME=1
make verify-cache OUT=<cache dir>                                       # read it back; exit 1 if it cannot be trusted
make check-cache                                                        # rebuild four floats and compare, ~1 min
```

A build never targets the cache the saved candidate lists are bound to. Rewriting those grids would
move the fingerprint every saved number stands on, so the script refuses that directory outright.
Build somewhere new and compare the two.

**What lands beside the grids.**

- `MANIFEST.csv` — one row per float: the grid kind promised and the one its data earned, the row
  and cycle counts, the timings. It also says how many placeholder values were masked and how much
  the ranges removed. A float that did not build gets its reason, and the nine floats left out get
  a row each.
- `PROVENANCE.json` — the whole recipe, the argopod version and the run, so a built cache states how
  it was made.
- `HEALTH.csv` — written by `make verify-cache`, per grid and per channel. How many placeholder
  values survived (any is a failure), and how many levels are over that channel's ceiling. Also
  what fraction of the rows are both finite and plausible.

**The check.** `make check-cache` rebuilds four floats — one per grid kind — into a temporary
directory and compares the sha256 of every grid file with the bound cache's. It prints SAME or DIFF
per file and exits 1 on any DIFF. Nothing survives the run. `WMO=` picks other floats, `RAW=` another
staged directory.

## The fingerprints (what holds a number still)

- Each pool's `.json` sidecar records `content_sha256` (the hash of the sorted key triples) and
  `file_sha256`. Re-detecting must reproduce `content_sha256` exactly.
- `CACHE_IDENTITY.json` records `fine_grids` (2542) and `fine_grids_sha256` (`11fce215…`). That hash
  is taken over the sorted fine-grid file **names**, so it catches a missing, added or renamed
  float and nothing else. It says the cache holds the same 2,542 floats; it does not say the grids
  inside them are unchanged. Only `make check-cache` says that, by rebuilding four floats and
  comparing every byte. The code refuses a cache whose identity does not match.
- `tests/pins.yaml` holds the numbers the tests share (pool sizes, the cache fingerprint, the six
  spec ids), each with a why.

## Traps

- The absolute cache path in `CACHE_IDENTITY.json` and `CANDIDATES.json` points at one machine. Get
  the cache from the data deposit with `scripts/fetch_caches.sh`, and set `EDDY_PUMP_CACHE` to its
  directory if it lands anywhere else. (Dropping the absolute path is a plan item.)
- The upward rate covers the open region alone: the held region, 14,697 of the pool's 186,275
  levels, is in no probability sample yet. The downward pool is sampled whole. `rates.py` reports
  each rate over the levels its sample covers, and never extrapolates to the rest.
