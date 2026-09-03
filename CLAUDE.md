# eddy-pump — Claude guide

Write for me like I have ADHD: short, lead with the answer, detail in the artifact. Plain words: no
repository jargon in a reply or a page (not "anchor", "adjudicate", "ruling", "gate", "ledger",
"ingest" — say what the thing is). One idea per sentence.

## What this is

The net-carbon study of the submesoscale eddy pump, built on `argopod` (the generic detect → review
→ triage library). It detects water-mass injection events in BGC-Argo profiles — both limbs
(subduction down, obduction up), three channels (physical, nitrate, carbon), six candidate pools —
and a human labels a probability sample. Every published number is a pure function of the human
labels. The paper is the net carbon export: gross subduction minus the return by obduction. Flux
quantification comes from the MOM6-COBALT model experiment (the OSSE, `../osse/`); the floats give
the rates.

This repository was cut fresh on 2026-09-03 from `mkeutgen/eddy-pump-archive` (tag
`archive-2026-09-03`, commit `cbd6fd6`), keeping only what the study needs. The archive holds the
full history and the retired GRL letter (its own tag `letter-v1`). The data deposit holds the fleet
cache and the old labels.

**There is exactly ONE plan: `docs/PLAN.md`** — what is settled, the next steps in order, what is
open. `docs/STATUS.md` is the dated log (read its top row first); `docs/IMPLEMENTATION_NOTES.md` is
how the machine works and what every file and column means; `docs/DECISIONS.md` is the rulings in
force, each with one line of why and a link to the archive commit that has the evidence. Never
create a second plan.

## Where things are

| where | what |
|---|---|
| this repo | the study: `data/candidates/net_carbon_v1/` (the six saved lists), the human labels, the pipeline (`pipeline/`), the code (`src/eddy_pump/`) |
| `~/Documents/release/eddy-pump-archive` | the full history that built this; the retired letter at tag `letter-v1`; the old record (`docs/DECISIONS.md`, 7,000+ lines) cited by commit |
| the data deposit (Zenodo, DOI pending; local `~/Documents/release/zenodo-study-deposit`) | the fleet cache (4.6 GB, fingerprint `11fce215…`), the old labels, the old candidate tables |
| `~/Documents/release/argopod` | the generic library (pinned v0.5.0; v0.5.1 in progress) |
| `~/Documents/release/.venv` | THE python (3.14); the study needs no R |

## The rules that matter

- Every science number is a pure function of the human labels. The classifier only decides which
  panels a human sees first; it never enters a number.
- Every rate names its denominator, and every precision names how the sample was drawn. A
  uniform-random precision measures a rate; a score-ranked one is barred from every rate.
- Numbers are frozen by fingerprints: the six candidate lists' key hashes (each pool's `.json`
  sidecar), the cache identity (`data/candidates/net_carbon_v1/CACHE_IDENTITY.json`). If a
  fingerprint moves, the code is wrong, not the number.
- Never quote a number from memory or an old document — read it from the file that produces it.
  When Maxime asks "are you sure?", there has always been a real bug: break your own result before
  defending it.
- Event key: `(WMO, CYCLE_NUMBER, round(PRES_ADJUSTED))` plus the pool.
- Generic machinery belongs in `argopod`; science choices belong here.
- No relabelling, live Argo fetch or cache rebuild without explicit instruction; nothing is deleted
  before its retrieval location is written down.

## Working here

`make study-help` prints the pipeline rules. The study: `make draw-batch` (draw the labelling
batch), `make review BATCH=…` (label, blind), `make calibrate SHEET=…` (check a blind re-labelling
of the 42 calibration panels), then `make ingest BATCH=…` and `make rates`. Tests: `make test`.

Models: Opus for day-to-day work; Fable only to orchestrate or to audit a number before it ships;
every subagent runs on Opus.

The whole pipeline runs on the study's one label table (`study_reviews.parquet` +
`study_batches.yaml` + `draws/*.yaml`); there is no old label layer in the code. The upward rate is
the open region alone — the old ≥ 1.96 σ region of the pool awaits a fresh draw under the study's
criterion (`docs/PLAN.md`).
