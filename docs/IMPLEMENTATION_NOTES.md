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
| `config/events.yaml` | the one canonical source: the six event definitions, the cache policy (ranges, the nine excluded floats), the pinned spec ids |
| `config/criteria.yaml` | the labelling criterion (four clauses) and the calibration references |
| `config/review/*.yaml` | the two panel layouts for the labelling app |
| `data/candidates/net_carbon_v1/` | the six saved lists (full Parquet tables), a `.json` sidecar per list, `CANDIDATES.json` (the six in one page), `CACHE_IDENTITY.json` (the bound cache) |
| `data/labels/study_reviews.parquet`, `study_batches.yaml`, `draws/` | the study's own labels: the batches, the reviews, the draw records with every inclusion probability, the frozen calibration answers |
| `data/external/` | the earlier subduction study's verified events (train the downward classifier; the reference events) and the calibration reference |
| `src/eddy_pump/` | the code (below) |
| `pipeline/` | the six pipeline scripts |
| `tests/` | the tests and the pins they share (`pins.yaml`) |

## The code (`src/eddy_pump/`)

- `vocabulary.py` — the fixed vocabulary: the channels, the two directions, the tracers.
- `spec.py` — the event spec objects and the declared field lists that `spec_id` hashes.
- `study.py` — the study object: the pools, the cache identity, the excluded floats, the output policy.
- `manifest.py` — `load_manifest()`: reads `config/events.yaml` into the study object; `REPO_ROOT`.
- `candidates.py` — the detector wrapper: detects a pool from the bound cache, saves/reads a list,
  `content_hash()` (the sha256 of the sorted key triples — a pool's identity), `verify_saved()`.
- `batches.py` — the batch design: the score-stratified uniform draw and its inclusion probabilities.
- `labels.py` — the label table (one layer, after the "one label layer" plan step).
- `criteria.py` — loads `config/criteria.yaml`; the active criterion.
- `domain.py` — a re-export shim (removed in the "tests and documents" plan step).

## The pipeline (`pipeline/`, run through `study.mk`)

`detect.py` (detect / verify the six pools) → `features.py` (a feature row per candidate) →
`scores.py` (a triage score per candidate) → `draw_batch.py` (draw a probability sample, render its
panels) → label blind (`argopod.cli review`) → `ingest_batch.py` (load a labelled sheet after the
calibration passes) → `rates.py` (the rate per limb, its denominator and error bar).

`make study-help` prints the rules. Slow steps: `verify-candidates` ~36 min, `features` ~54 min.

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
- The labelling loop and the rate report still read the pre-study label layer; the "one label layer"
  plan step rewrites them. Until then only `verify-candidates`, `freeze-candidates` and `features`
  run end to end.
