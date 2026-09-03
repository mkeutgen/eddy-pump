# DECISIONS — the rulings in force

The rulings the study rests on, each in one line with why, and a link to the archive commit that
holds the evidence. The full record — every ruling with its evidence, append-only — is in the
archive `mkeutgen/eddy-pump-archive` at `docs/DECISIONS.md` (7,000+ lines); this file is only what is
still in force. When a new ruling is made here, add it with its date and its evidence.

Archive reference for everything below: `mkeutgen/eddy-pump-archive`, tag `archive-2026-09-03`
(commit `cbd6fd6`), `docs/DECISIONS.md`.

## How this repository was born

Cut fresh on 2026-09-03 from the archive, keeping only what the net-carbon study needs, after Maxime
confirmed the seven decisions of the fresh-repository plan (archive record §30). What left the
repository — the old labels, the reuse audit, the old candidate tables, the cache builder, the old
record — lives in the archive and the data deposit. **Why:** a reader should see the study, not the
road that built it.

## The pools and the cache

- **Six candidate pools, two limbs × three channels**, frozen as the full Parquet tables under
  `data/candidates/net_carbon_v1/` (186,275 / 133,307 / 13,329 / 7,503 / 11,400 / 12,081 = 363,895).
  Each child is its directional parent plus one tracer term, and its candidates are a subset of the
  parent's. **Why:** one saved copy every label points into. *(Archive: "Study-aware detection",
  the readability audit.)*
- **The fleet cache is one declared build** — fill masked, plausible ranges, nitrate clipped, nine
  floats out, a per-cycle salinity fallback and a backscatter smoother: 2,542 fine grids, fingerprint
  `11fce215…` (`data/candidates/net_carbon_v1/CACHE_IDENTITY.json`). It is not in git; it is in the
  data deposit. **Why:** the grids cannot be rebuilt bit-identically from a fresh Argo pull. *(Archive:
  the cache rulings, the fourth build.)*
- **The salinity magnitude is a reported covariate, never a gate.** **Why:** the noise floor was
  refused against the verified events. *(Archive: the salinity floor.)*

## The labels and the numbers

- **Every science number is a pure function of the human labels.** The classifier only orders which
  panels a human sees first; it never enters a number. **Why:** honesty by construction.
- **Every rate comes from a probability sample with a declared frame and inclusion probability.**
  Score-selected labels are training only and every rate function rejects them. A uniform-random
  precision measures a rate; a score-ranked one is 7–15× higher on the same ocean and is barred.
  **Why:** the two kinds of number must never be confused. *(Archive: the labelling levers.)*
- **The review criterion is the four-clause protocol on both limbs, `phys_net_carbon_v1`;** clause 4
  applies to both limbs. **Why:** one criterion, one design. *(Archive: the criterion.)*
- **The precision target is ±15 % relative per limb** — assumed, pending the advisor. *(Archive: the
  precision target.)*
- **The old (earlier GRL letter) labels left the repository; a rate comes only from the study's own
  probability samples under `phys_net_carbon_v1`.** Old labels may be fetched from the deposit as
  classifier training evidence only, never a rate — key overlap is not interchangeability. **Why:**
  one criterion, one design; the old labels were judged under a different rule. *(Archive: the reuse
  audit, §30.)*
- **Every companion subduction event is a reference event;** one the detector misses is a detector
  defect, never a bad label. *(Archive: decisions taken.)*

## The classifier

Three stages: triage now (orders candidates into score strata), a calibrated census next
(`census.py`, precision and recall from the labelled samples), a wider candidate net after the OSSE.
The paper's rates stay human-labelled; every probability-sample label is both a rate's sample and the
classifier's out-of-fold validation set. **Why:** it reaches "a classifier that finds the events"
without a number that cannot be checked against a label. *(Archive: §30.)*

## The event key

`(WMO, CYCLE_NUMBER, round(PRES_ADJUSTED))` plus the pool. **Why:** the level is the unit of a rate;
the pool disambiguates a level that is a candidate in more than one.
