# eddy-pump

The net carbon export of the submesoscale eddy pump, measured from BGC-Argo floats.

The eddy pump pushes surface water down (subduction) and brings deep water up (obduction). This
study detects those injection events in Argo profiles, has a human label a probability sample of
them, and turns the labels into a rate per limb. The paper's number is the **net**: gross carbon
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
flags; a human labels a **probability sample** of candidates as real or not; the rate is the
weighted fraction of real ones, with its denominator and error bar. The classifier only orders
which panels a human sees first — it never enters a number.

Two rates measured so far (both physical, awaiting an adversarial review):

- **upward** 12.9 % of the open region (171,578 of the pool's 186,275 candidate levels), ±17 % relative — the rest of the pool awaits a fresh draw
- **downward** 18.7 % of 133,307 candidate levels, ±14 % relative

Every rate names its denominator; a uniform-random sample measures a rate, a score-ranked one never
does. "Complete" always says against which pool.

## Run it

Python is the shared venv at `~/Documents/release/.venv` (3.14); no R.

```bash
make study-help        # the pipeline rules, in order
make test              # the tests
make -n rates          # print what refreshing the rate report would run
make verify-candidates # re-detect the six pools and compare with the saved lists (needs the cache, ~36 min)
```

The labelling loop, blind: `make draw-batch`, `make review BATCH=…`, `make calibrate SHEET=…`,
`make ingest BATCH=…`, then `make rates`.

To re-detect from the grids you need the fleet cache from the data deposit:

```bash
DEPOSIT_DIR=/path/to/zenodo-study-deposit make fetch-caches   # until the Zenodo DOI is published
```

## Where things are

| where | what |
|---|---|
| this repo | the study: the saved candidate lists, the human labels, the definitions, the pipeline, the classifier |
| `mkeutgen/eddy-pump-archive` (tag `archive-2026-09-03`, commit `cbd6fd6`) | the full history that built this, and the retired GRL letter (its own tag `letter-v1`) |
| the data deposit (Zenodo, DOI pending) | the fleet cache (4.6 GB), the old labels, the old paper's candidate tables |
| `mkeutgen/argopod` | the generic library (pinned v0.5.0; v0.5.1 in progress) |

## Provenance

Cut fresh from `mkeutgen/eddy-pump-archive` at tag `archive-2026-09-03` (commit `cbd6fd6`) on
2026-09-03, keeping only what the net-carbon study needs. The plan and the rulings in force are in
`docs/`. This is research code under active development, not a finished product.
