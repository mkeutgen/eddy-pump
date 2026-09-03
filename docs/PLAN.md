# PLAN — what is next (2026-09-03)

**This is the only plan.** It says what is settled, what comes next in what order, and what is still
open. The rulings in force are `docs/DECISIONS.md`, each linking the archive commit with the
evidence; the dated log is `docs/STATUS.md`; how the machine works is `docs/IMPLEMENTATION_NOTES.md`.
If this file grows past two screens, something is in the wrong file.

## The goal

One paper on the **net** carbon export of the submesoscale eddy pump: gross carbon subduction minus
carbon return by obduction, in the physical / nitrate / carbon channels. Flux quantification comes
from the MOM6-COBALT OSSE; the floats give the census and the return fraction.

## Where things stand

This repository was cut fresh on 2026-09-03 from `mkeutgen/eddy-pump-archive` (tag
`archive-2026-09-03`, commit `cbd6fd6`), keeping only what the study needs. The six saved candidate
lists load and their key hashes match the archive; `load_manifest()` gives the six pools. The
labelling loop and the rate report still carry the pre-study label layer — the first step below
removes it.

Two physical rates are measured (both awaiting an adversarial review): upward 12.9 % of 186,275,
±16 %; downward 18.7 % of 133,307, ±14 %.

## What is settled

The seven decisions of the fresh-repository plan, all confirmed 2026-09-03 (archive
`docs/DECISIONS.md` §30): a new repository (this one); the old labels to the deposit and the former
upward held region re-sampled under the study's criterion; a fresh upward calibration set labelled
twice; the six lists kept in git; the cache builder to argopod; the names; the classifier's
three-stage role (triage now, a calibrated census next, a wider net after the OSSE) with the paper's
rates staying human-labelled. The cache policy, the criterion, the sampling design and the ±15 %
precision target carry over from the archive unchanged.

## The next steps, in order

Each step names what it produces and the one number that matters. Estimates are agent working time.

1. **One label layer, one design.** `labels.py` reads one table; the batch drawer and the rate
   report lose the held-region logic (~150 lines); `criteria.yaml` keeps one criterion and two
   calibration references; the ingest stops checking the old layer; every "old / legacy / letter /
   paper line" word leaves the code. *Check: the downward rate reproduces exactly (18.7 % of
   133,307, ±14 %); the tests pass.* *One day.*
2. **The upward limb on the study's own footing** *(your labelling, ~2 hours).* Draw a fresh
   42-panel upward calibration set, labelled twice blind and adjudicated; draw the former held
   region (14,697 levels, ~90 panels, one per float) and label it. *Check: the pool rate's error bar
   is inside ±15 %; the new rate is compared with today's 12.9 % ± 16 % and the difference recorded.*
   *Agent 1 hour.*
3. **The classifier as a module of its own.** `classifier.py` with `train` / `evaluate` / `score` /
   `calibrate` (folds by float; metrics only on the probability samples; a saved model + manifest).
   Then `census.py` (the calibrated census) once the upward limb is on its own footing. *Check: the
   honest out-of-fold AUC and the calibration curve are in the manifest; the rates do not change.*
   *One day.*
4. **The cache builder moves to argopod.** A generic "build a fleet cache from a float list and a
   policy" tool; the study passes its policy from `events.yaml`; argopod `0bc30e3` tagged v0.5.1 and
   pinned here. *Check: one float rebuilt reproduces its grid byte for byte.* *One day.*
5. **Tests and documents of the new repository.** ~8 test files with one pins file and the strict
   mode; the record holds only the rulings in force; remove the `domain.py` shim and rewire its two
   tests. *Check: a cold read answers the seven questions in under 400 lines; the tests pass in
   strict mode.* *One day.*
6. **The adversarial review of both rates** before either is quoted (archive plan step 1). Produces
   a confirmed or corrected `RATE_STATUS.md`. *Half a day.*

Blocked by nothing here: **the data deposit upload** (yours — the local deposit is staged at
`~/Documents/release/zenodo-study-deposit`, 2.9 GB, verified; upload it, get the DOI, set
`ZENODO_RECORD` in `scripts/fetch_caches.sh`). **The OSSE** (`../osse/`) proceeds in parallel.

## Still open

| decision | recommendation on file | who |
|---|---|---|
| The particle gate on carbon subduction (≥ 1.00 σ backscatter) | keep at 1.00 and report; it costs ~3 % of the reference carbon events (archive record) | user |
| Whether the paper census sits beside the letter's | beside; undecided | user |
| Sizes of the positive-control, dipole and carbon-obduction arms | from the coverage report, not before | advisor |

## Hard rules

1. Every scientific rate comes from a probability sample with a declared frame and inclusion
   probability; score-selected labels are training only and every rate function rejects them.
2. Old labels are directly reusable only when pool identity **and** criterion version match;
   otherwise they are calibration, training or audit evidence.
3. Large grids, raw profiles and raw sheets stay out of git; the saved candidate lists, manifests
   and fingerprints stay in.
4. No relabelling, live Argo fetch or cache rebuild without explicit instruction; nothing is deleted
   before its retrieval location is written down.
