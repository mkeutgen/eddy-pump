# PLAN — what is next (2026-09-04)

**This is the only plan.** It says what is settled, what comes next in what order, and what is still
open. The decisions in force are `docs/DECISIONS.md`, each naming the archive entry with the
evidence. The dated log is `docs/STATUS.md`. How the machine works, and what every word here means,
is `docs/IMPLEMENTATION_NOTES.md`. If this file grows past two screens, something is in the wrong
file.

## The goal

One paper on the **net** carbon export of the submesoscale eddy pump. The net is gross carbon
subduction minus the carbon returned by obduction, in the physical, nitrate and carbon channels.
Flux quantification comes from the MOM6-COBALT OSSE; the floats give the census and the return
fraction.

## Where things stand

This repository was cut fresh on 2026-09-03 from `mkeutgen/eddy-pump-archive` (tag
`archive-2026-09-03`, commit `cbd6fd6`), keeping only what the study needs. The six saved candidate
lists load, and their key hashes match the archive. The whole pipeline runs on the study's one label
table; no other label layer is left in the code.

Two physical rates are measured, both awaiting an adversarial review:

- **downward** 18.7 % of 133,307 candidate levels, ±14 % relative.
- **upward** 12.9 % of 186,275 candidate levels, ±16.5 % relative, from two random samples: the
  open region (171,578 levels) and the held region (14,697 levels, picked earlier by the retired
  letter's stricter 1.96 σ cut, labelled 2026-09-04). About 143 more open-region panels reach ±15 %.

## What is settled

Seven decisions of the fresh-repository plan, all confirmed 2026-09-03:

1. A new repository — this one.
2. The old labels go to the deposit, and the held region is re-sampled under this study's criterion.
3. A fresh upward calibration set, labelled twice.
4. The six candidate lists stay in git.
5. The cache builder moves to argopod.
6. The names, as they now stand.
7. The classifier does triage now, a calibrated census next, a wider net after the OSSE. The
   paper's rates stay human-labelled.

The cache recipe, the criterion, the sampling design and the ±15 % precision target carry over from
the archive unchanged. *(Archive: "The fresh repository confirmed", 2026-09-03.)*

## The next steps, in order

Each step names what it produces and the one number that matters. Estimates are agent working time.

1. **One label layer, one design — done 2026-09-03.** One label table, one criterion, two
   calibration sets, and a classifier trained on the study's own labels. The downward rate
   reproduces to the bit; the upward becomes the open region alone. See `docs/STATUS.md`.
2. **The upward limb on the study's own footing — done 2026-09-04.** `calib_obduction_v1`: 42
   fresh upward calibration panels, labelled twice blind (40 of 42 agree, kappa 0.95), the two
   disagreements decided by hand, the answers frozen at 18/42. `rate_obduction_02`: the held region
   (14,697 levels) labelled, 128 panels, controls clean, acceptance flat with position. The upward
   rate now covers the whole pool: 12.9 % of 186,275, ±16.5 % (was 12.9 % of 171,578, ±17.4 %).
   Left: about 143 more open-region panels reach ±15 % (see "Still open").
3. **The classifier as a module of its own — done 2026-09-04.** `src/eddy_pump/classifier.py`:
   fit / evaluate / score / calibrate, folds by float, the model saved with a record of how it was
   made. The scores it writes are byte-identical to before (`SCORES_SHA256`). Left: `census.py`, the
   calibrated census, once the upward limb stands on its own footing. *Half a day.*
4. **The cache builder moved to argopod — done 2026-09-04.** The builder is generic and lives in
   the library; the study keeps the recipe (`config/events.yaml`) and the float list
   (`config/fleet.csv`). **Four floats, one per grid kind, rebuild byte for byte identical to the
   bound cache:** `make check-cache`, about a minute. See `docs/STATUS.md`.
5. **Tests and documents of the new repository.** Mostly done by the audit of 2026-09-04: 140
   tests (from 102) in strict mode, a 15-word glossary, STATUS ten lines per day, the rejected words
   out of the prose. Left: remove the `domain.py` shim and rewire its two tests; re-measure the cold
   read. *Check: a cold read answers the seven questions in under 400 lines.* *Half a day.*
6. **The adversarial review of both rates — done 2026-09-04.** Both hold as estimates of what
   the reviewer called real: two independent recomputations reproduce them to eight decimals, and
   every hash in the chain matches. But the reviewer drifted within the two first sessions by two to
   three times the sampling error, for different reasons on each limb, and the net of the two limbs
   changes sign across that band (+2,509 / +984 / −248 levels). Honest intervals: downward 0.123 to
   0.249, upward 0.087 to 0.168. The record: `data/labels/audit/RATE_REVIEW_2026-09-04.md`.
   **No net is quoted before step 7.**
7. **Measure and correct the drift** *(your labelling: 2 × 100 panels, about 40 minutes; agent half
   a day).* A blind re-judgement of 100 random panels from each first batch, shuffled, labelled in
   sessions of at most 120 panels with controls interleaved by construction. The flip rate by original
   position gives the correction and its uncertainty. Before drawing: regenerate the 2026-08-27
   upward score file from the archive's recipe (commit e7a625c) and save each draw's stratum
   membership beside its record, so the open-region strata can be reused. *Check: both rates and
   the net carry an interval that includes the drift.*

Blocked by nothing here: **the data deposit upload** (yours — the local deposit is staged at
`~/Documents/release/zenodo-study-deposit`, 2.9 GB, verified; upload it, get the DOI, set
`ZENODO_RECORD` in `scripts/fetch_caches.sh`). **The OSSE** (`../osse/`) proceeds in parallel.

## Still open

| decision | recommendation on file | who |
|---|---|---|
| The particle test on carbon subduction (≥ 1.00 σ backscatter) | keep it at 1.00 and report it; it costs about 3 % of the reference carbon events (archive record) | user |
| Whether the paper census sits beside the earlier GRL letter's | beside; undecided | user |
| Sizes of the positive-control, dipole and carbon-obduction arms | from the coverage the study's own samples show, not before | advisor |
| About 143 more open-region upward panels, to bring the pool rate inside ±15 % | after step 7, in sessions of at most 120 panels; about 1.1 hours | user |
| Session rules in the protocol: at most 120 panels per session, controls interleaved by construction, the calibration copy labelled the day before | adopt; the drift the review found is the reason | user |

## Hard rules

1. Every scientific rate comes from a probability sample with a declared frame and inclusion
   probability. Score-selected labels are training only, and every rate function rejects them.
2. A rate comes only from the study's own probability samples under `phys_net_carbon_v1`. Labels
   fetched from the deposit — the earlier letter's — are training evidence for the classifier, never
   a rate.
3. Large grids, raw profiles and raw sheets stay out of git. The saved candidate lists, the event
   definitions and the fingerprints stay in.
4. No relabelling, live Argo fetch or cache rebuild without explicit instruction. Nothing is deleted
   before its retrieval location is written down.
