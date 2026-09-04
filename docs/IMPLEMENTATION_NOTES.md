# IMPLEMENTATION_NOTES — how the machine works

How the study is built and what every file and column means. The plan is `docs/PLAN.md`; the
rulings are `docs/DECISIONS.md`. (Freshly seeded 2026-09-03; the "tests and documents" step in the
plan expands this file.)

## The words

- **candidate** — a level the detector flags: an event key `(WMO, CYCLE_NUMBER, round(PRES_ADJUSTED))`
  in a pool.
- **pool** — one of the six event definitions (two limbs × three channels). A child pool (a tracer
  channel) is its directional parent plus one tracer term; its candidates are a subset of the parent's.
- **limb** — subduction (down) or obduction (up).
- **rate** — the weighted fraction of a pool's candidates that a human called real, from a
  probability sample, with a denominator and an error bar.
- **the anchor / calibration set** — the 42 panels a labeller re-labels blind before a session; the
  frozen answers gate a session in.

## Where things are

| path | what |
|---|---|
| `config/events.yaml` | the one canonical source: the six event definitions, how the fleet cache is built (the `cache:` block, the ranges, the nine floats left out), the pinned spec ids |
| `config/fleet.csv` | the fleet-cache build list: 2,574 floats, the grid flavour each is promised, and the wave it was built in. Row for row the bound cache's own record |
| `config/criteria.yaml` | the labelling criterion (four clauses) and the calibration references |
| `config/review/*.yaml` | the two panel layouts for the labelling app |
| `data/candidates/net_carbon_v1/` | the six saved lists (full Parquet tables), a `.json` sidecar per list, `CANDIDATES.json` (the six in one page), `CACHE_IDENTITY.json` (the bound cache) |
| `data/labels/study_reviews.parquet`, `study_batches.yaml`, `draws/` | the study's own labels: the batches, the reviews, the draw records with every inclusion probability, the frozen calibration answers |
| `data/external/` | the earlier subduction study's verified events (train the downward classifier; the reference events) and the calibration reference |
| `src/eddy_pump/` | the code (below) |
| `pipeline/` | the seven pipeline scripts |
| `tests/` | the tests and the pins they share (`pins.yaml`) |

## The code (`src/eddy_pump/`)

- `vocabulary.py` — the fixed vocabulary: the channels, the two directions, the tracers.
- `spec.py` — the event spec objects and the declared field lists that `spec_id` hashes.
- `study.py` — the study object: the pools, the cache identity, the excluded floats, the output
  policy, and `cache_policy()` — the whole fleet-cache recipe, as argopod consumes it.
- `manifest.py` — `load_manifest()`: reads `config/events.yaml` into the study object; `REPO_ROOT`.
- `candidates.py` — the detector wrapper: detects a pool from the bound cache, saves/reads a list,
  `content_hash()` (the sha256 of the sorted key triples — a pool's identity), `verify_saved()`.
- `batches.py` — the batch design: the score-stratified uniform draw and its inclusion probabilities.
- `labels.py` — the study's one label table (query API; the only door a rate may use).
- `criteria.py` — loads `config/criteria.yaml`; the active criterion.
- `domain.py` — a re-export shim (removed in the "tests and documents" plan step).

## The pipeline (`pipeline/`, run through `study.mk`)

`build_cache.py` (build or check the fleet cache; see below, and never as part of a run) →
`detect.py` (detect / verify the six pools) → `features.py` (a feature row per candidate) →
`scores.py` (a triage score per candidate) → `draw_batch.py` (draw a probability sample, render its
panels) → label blind (`argopod.cli review`) → `ingest_batch.py` (load a labelled sheet after the
calibration passes) → `rates.py` (the rate per limb, its denominator and error bar).

`make study-help` prints the rules. Slow steps: `verify-candidates` ~36 min, `features` ~54 min.

## Rebuilding the fleet cache

The cache is one pair of residual grids per float — a fine grid and a coarse one — and it is the
expensive half of everything: each float is downloaded once, its derived channels computed once, its
levels binned once, and every detection afterwards reads the result. The bound cache is 2,542 fine
grids, fingerprint `11fce215…`.

**Who does what.** The building is generic and lives in argopod (`argopod.cache`, v0.5.1): a float
list plus a recipe in, a directory of grids out. The study owns the recipe and the list.

**Where each part of the recipe comes from**, all of it in `config/events.yaml`:

| part | where it is written |
|---|---|
| the four grid flavours and the channels each carries | the `cache:` block, `labels:` |
| the dates kept, 2009-01-08 to 2026-03-15 | the `cache:` block, `window:` |
| a placeholder value becomes "no reading" before anything is derived from it | the `cache:` block, `fill_policy: mask` |
| a cycle with no delayed-mode column reads the raw one, that cycle only | the `cache:` block, `adjusted_fallback: cycle` |
| the ceilings `make verify-cache` uses | the `cache:` block, `residual_ceilings:` |
| the plausible ranges on the columns floats arrive with | `raw_inputs:` |
| the plausible ranges on the channels the study detects on | each channel's own `prefilter:` |
| the backscatter smoother | `tracers.carbon.prefilter.pre_median_filter` |
| the nine floats left out, each with its reason | `excluded_floats:` |

`Study.cache_policy()` joins them into the one object argopod builds from, so nothing is written
down twice. The bin widths, the smoothing window and the column names are argopod's own defaults:
the bound cache was built under them, and the one detector knob this study moves acts when
candidates are found, never when a grid is filled. `tests/test_cache_policy.py` names all fifteen
of those and fails if one drifts.

**The float list** is `config/fleet.csv`: `WMO,label,tier` for 2,574 floats — 1,181 promised
`paper_phys`, 953 `paper_all`, 382 `paper_bbp`, 58 `paper_nit`, in four waves. It matches the bound
cache's own `MANIFEST.csv` row for row.

**The raw frames.** `~/Documents/release/cache_paper/raw` holds 2,559 staged downloads, 16 GB, one
parquet per float. They live on this machine only — they are not in the data deposit and not in git.
A float with no staged frame is downloaded during the build; `ARGOPOD_SKIP_ERDDAP=1` makes those
downloads go straight to the GDAC files instead of trying the ERDDAP endpoint first, which saves
minutes per float when that endpoint is slow.

**The commands** (`pipeline/build_cache.py` behind all three):

```bash
make build-cache OUT=<new dir> RAW=~/Documents/release/cache_paper/raw   # add WORKERS= TIER= LIMIT= RESUME=1
make verify-cache OUT=<cache dir>                                       # read it back; exit 1 if it cannot be trusted
make check-cache                                                        # rebuild four floats and compare, ~1 min
```

A build never targets the cache the saved candidate lists are bound to: rewriting those grids would
move the fingerprint every frozen number stands on, so the script refuses that directory outright.
Build somewhere new and compare the two.

**What lands beside the grids.** `MANIFEST.csv` — one row per float: the flavour promised and the
flavour its data earned, row and cycle counts, how many placeholder values were masked and how much
the ranges removed, timings, and for a float that did not build, why. The nine floats left out get a
row each carrying their ruling. `PROVENANCE.json` — the whole recipe, the argopod version and the
run, so a built cache states how it was made. `HEALTH.csv` — written by `make verify-cache`: per
grid and per channel, how many placeholder values survived (any is a failure), how many levels are
over that channel's ceiling, and what fraction of the rows are both finite and plausible.

**The check.** `make check-cache` rebuilds four floats — one per grid flavour — into a temporary
directory and compares the sha256 of every grid file with the bound cache's. It prints SAME or DIFF
per file and exits 1 on any DIFF. Nothing survives the run. `WMO=` picks other floats, `RAW=` another
staged directory.

## The fingerprints (what freezes a number)

- Each pool's `.json` sidecar records `content_sha256` (the key-triple hash) and `file_sha256`.
  Re-detecting must reproduce `content_sha256` exactly.
- `CACHE_IDENTITY.json` records `fine_grids` (2542) and `fine_grids_sha256` (`11fce215…`). The code
  refuses a cache that does not match.
- `tests/pins.yaml` holds the numbers the tests share (pool sizes, the cache fingerprint, the six
  spec ids), each with a why.

## Traps

- The absolute cache path in `CACHE_IDENTITY.json` and `CANDIDATES.json` points at one machine; the
  cache comes from the data deposit via `scripts/fetch_caches.sh`. (Dropping the absolute path is a
  plan item.)
- The upward rate is the open region alone: its former ≥ 1.96 σ region (14,697 of the pool's 186,275
  levels) is not yet in a probability sample and awaits a fresh draw. The downward pool is sampled
  whole. `rates.py` reports each rate over the levels its sample covers, never extrapolating.
