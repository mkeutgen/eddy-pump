# eddy-pump — Claude guide

Write for me like I have ADHD: short, lead with the answer, detail in the artifact. Plain words: no
repository jargon in a reply or a page. Words I do not want to read: anchor, adjudicate, ruling,
gate, ledger, ingest, pin, fence, freeze, manifest, persona. Say what the thing is instead. One idea
per sentence, and no metaphors.

## What this is

The net-carbon study of the submesoscale eddy pump, built on `argopod` (the generic detect → review
→ triage library). It detects water-mass injection events in BGC-Argo profiles: two limbs
(subduction down, obduction up), three channels (physical, nitrate, carbon), six candidate pools.
A human labels a probability sample of them. Every published number is a pure function of those
labels. The paper is the net carbon export: gross subduction minus the return by obduction. Flux
quantification comes from the MOM6-COBALT model experiment (the OSSE, `../osse/`); the floats give
the rates.

This repository was cut fresh on 2026-09-03 from `mkeutgen/eddy-pump-archive` (tag
`archive-2026-09-03`, commit `cbd6fd6`), keeping only what the study needs. The archive holds the
full history and the retired GRL letter (its own tag `letter-v1`). The data deposit holds the fleet
cache and the old labels.

**There is exactly ONE plan: `docs/PLAN.md`** — what is settled, the next steps in order, what is
open. `docs/STATUS.md` is the dated log (read its top entry first). `docs/IMPLEMENTATION_NOTES.md`
is how the machine works and what every file and column means. Its first section defines every
word this repository uses; read that glossary before writing anything. `docs/DECISIONS.md` is the
decisions in force, each with one line of why and a link to the archive commit that has the
evidence. Never create a second plan.

## Where things are

| where | what |
|---|---|
| this repo | the study: `data/candidates/net_carbon_v1/` (the six saved lists), the human labels, the pipeline (`pipeline/`), the code (`src/eddy_pump/`) |
| `~/Documents/release/eddy-pump` (GitHub `mkeutgen/eddy-pump-archive`) | the full history that built this; the retired letter at tag `letter-v1`; the old record (`docs/DECISIONS.md`, 7,000+ lines) cited by commit |
| the data deposit (Zenodo, DOI pending; local `~/Documents/release/zenodo-study-deposit`) | the fleet cache (4.6 GB, fingerprint `11fce215…`), the old labels, the old candidate tables |
| `~/Documents/release/argopod` | the generic library, the fleet-cache builder included (fixed at v0.5.2) |
| `~/Documents/release/.venv` | THE python (3.14); the study needs no R |

## The rules that matter

- Every science number is a pure function of the human labels. The classifier only decides which
  panels a human sees first; it never enters a number.
- Every rate names its denominator, and every precision names how the sample was drawn. A
  uniform-random precision measures a rate; a score-ranked one is barred from every rate.
- Fingerprints are what keep a number from moving: the six candidate lists' key hashes (each pool's
  `.json` sidecar) and the cache identity
  (`data/candidates/net_carbon_v1/CACHE_IDENTITY.json`). If a fingerprint moves, the code is wrong,
  not the number. The cache fingerprint hashes the grid file *names*, so it only says the cache
  holds the same 2,542 floats. `make check-cache` is what proves the grids themselves are
  unchanged.
- Never quote a number from memory or an old document — read it from the file that produces it.
  When Maxime asks "are you sure?", there has always been a real bug: break your own result before
  defending it.
- Event key: `(WMO, CYCLE_NUMBER, round(PRES_ADJUSTED))` plus the pool.
- Generic machinery belongs in `argopod`; science choices belong here.
- No relabelling, live Argo fetch or cache rebuild without explicit instruction; nothing is deleted
  before its retrieval location is written down.

## Working here

`make study-help` prints the pipeline rules. The labelling loop runs in this order, and the
calibration is always first:

1. `make draw-batch BATCH=…` — draw the sample and render its panels.
2. A fresh blind copy of the 42 calibration panels, labelled, then `make calibrate SHEET=…`. It must
   say PASS. The loader wants a calibration finished before the batch sheet.
3. `make review BATCH=…` — label the batch, blind.
4. `make load BATCH=…`, then `make rates`.

`make check-cache` rebuilds four floats and checks their grids against the bound cache. Tests:
`make test`.

Models: Opus for day-to-day work; Fable only to orchestrate or to audit a number before it ships;
every subagent runs on Opus.

The whole pipeline runs on the study's one label table (`study_reviews.parquet` +
`study_batches.yaml` + `draws/*.yaml`); there is no old label layer in the code. The upward rate
covers the open region alone — the 171,578 levels of that pool that were sampled at random. The
other 14,697 are the held region. They were picked earlier by the retired letter's stricter 1.96 σ
cut, and they wait for a draw of their own (`docs/PLAN.md`).
