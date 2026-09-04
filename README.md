# eddy-pump

The net carbon export of the submesoscale eddy pump, measured from BGC-Argo floats.

The eddy pump pushes surface water down (subduction) and brings deep water up (obduction). This
study detects those injection events in Argo profiles. A human labels a probability sample of
them, and the labels become a rate per limb. The paper's number is the **net**: gross carbon
subduction minus the carbon returned by obduction, in three channels — physical, nitrate, carbon.
Flux quantification comes from a MOM6-COBALT model experiment (the OSSE, a sibling project); the
floats give the census and the return fraction.

This repository is built on `argopod`, the generic detect → review → triage library. Generic
machinery lives there; the science choices live here.

## The six candidate pools

Two limbs × three channels. Each channel is its directional parent plus one tracer term, and each
child's candidates are a subset of its parent's.

| pool | candidate levels |
|---|---|
| physical / obduction (up) | 186,275 |
| physical / subduction (down) | 133,307 |
| nitrate / obduction | 13,329 |
| nitrate / subduction | 7,503 |
| carbon / obduction | 11,400 |
| carbon / subduction | 12,081 |
| **total** | **363,895** |

The lists are saved in full as Parquet under `data/candidates/net_carbon_v1/` (27 MB). A clone
computes the rates from these without the 4.6 GB fleet cache; you need the cache only to re-detect
the candidates from the grids.

## How a number is made

Every published number is a pure function of the human labels. A candidate is a level the detector
flags. A human labels a **probability sample** of candidates as real or not. The rate is the
weighted fraction of the real ones, with its denominator and its error bar. The classifier only
orders which panels a human sees first; it never enters a number.

Two rates measured so far, both awaiting an adversarial review:

- **upward** 12.9 % ± 17 % relative, over the open region — the 171,578 of the pool's 186,275
  candidate levels that were sampled at random. The other 14,697 are the held region: picked
  earlier by the retired letter's stricter 1.96 σ cut. No random sample covers them yet, so they
  wait for a draw of their own.
- **downward** 18.7 % ± 14 % relative, over all 133,307 candidate levels of that pool.

Every rate names its denominator; a uniform-random sample measures a rate, a score-ranked one never
does. "Complete" always says against which pool. The words used here are defined at the top of
`docs/IMPLEMENTATION_NOTES.md`.

## Run it

Python is the shared venv at `~/Documents/release/.venv` (3.14); no R.

```bash
make study-help                      # the pipeline rules, in the order you run them
make test                            # the tests
make -n rates                        # print what refreshing the rate report would run
```

The labelling loop, in order. The calibration comes first, always: the loader wants a calibration
that passed and was finished before the batch sheet.

```bash
make draw-batch BATCH=<name>         # draw a sample and render its panels
python pipeline/draw_batch.py --repass calib_<limb>_…   # a fresh blind copy of the 42 calibration panels
make review BATCH=<the copy>.csv     # label that copy, blind
make calibrate SHEET=<the copy>.csv  # must PASS before you touch the batch
make review BATCH=<the batch>.csv    # label the batch, blind
make load BATCH=<name>               # load the labelled batch into the label table
make rates                           # the rate per limb, its denominator and its error bar
```

The fleet cache, which you need only to re-detect the candidates from the grids:

```bash
DEPOSIT_DIR=/path/to/zenodo-study-deposit make fetch-caches   # until the Zenodo DOI is published
make verify-candidates               # re-detect the six pools and compare with the saved lists (~36 min)
make check-cache                     # rebuild four floats and compare their grids with the bound cache (~1 min)
```

`CACHE_IDENTITY.json` records one machine's absolute path to that cache. If it lands somewhere else,
set `EDDY_PUMP_CACHE` to its directory and every rule reads it there.

Rebuilding the whole fleet cache is a deliberate act and never part of a run:
`make build-cache OUT=<new dir> RAW=<staged raw frames>`, then `make verify-cache OUT=<new dir>`.
It refuses to write into the cache the saved lists are bound to. See `docs/IMPLEMENTATION_NOTES.md`.

## Where things are

| where | what |
|---|---|
| this repo | the study: the saved candidate lists, the human labels, the event definitions, the pipeline, the classifier |
| `mkeutgen/eddy-pump-archive` (tag `archive-2026-09-03`, commit `cbd6fd6`) | the full history that built this, and the retired GRL letter (its own tag `letter-v1`) |
| the data deposit (Zenodo, DOI pending) | the fleet cache (4.6 GB), the old labels, the old paper's candidate tables |
| `mkeutgen/argopod` | the generic library, the fleet-cache builder included (fixed at v0.5.2) |

## Provenance

Cut fresh from `mkeutgen/eddy-pump-archive` at tag `archive-2026-09-03` (commit `cbd6fd6`) on
2026-09-03, keeping only what the net-carbon study needs. The plan and the decisions in force are in
`docs/`. This is research code under active development, not a finished product.
