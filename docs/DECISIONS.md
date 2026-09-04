# DECISIONS — the decisions in force

What the study rests on, each in one line with why, and a pointer to the archive entry that holds
the evidence. The full record — every decision with its evidence, only ever added to — is in the
archive, `mkeutgen/eddy-pump-archive`, `docs/DECISIONS.md` (7,000+ lines). This file is only what is
still in force. When a new decision is made here, add it with its date and its evidence.

Archive reference for everything below: `mkeutgen/eddy-pump-archive`, tag `archive-2026-09-03`
(commit `cbd6fd6`), `docs/DECISIONS.md`. The words used here are defined in
`docs/IMPLEMENTATION_NOTES.md`.

## How this repository was born

Cut fresh on 2026-09-03 from the archive, keeping only what the net-carbon study needs, after Maxime
confirmed the seven decisions of the fresh-repository plan. What left the repository lives in the
archive and the data deposit. That is the old labels, the reuse audit, the old candidate tables, the
cache builder and the old record. **Why:** a reader should see the study, not the road that built
it.
*(Archive: "The fresh repository confirmed", 2026-09-03.)*

## The pools and the cache

- **Six candidate pools, two limbs × three channels**, saved in full as Parquet tables under
  `data/candidates/net_carbon_v1/` (186,275 / 133,307 / 13,329 / 7,503 / 11,400 / 12,081 = 363,895).
  Each tracer pool is its directional parent plus one tracer term, so its candidates are a subset of
  the parent's. **Why:** one saved copy that every label points into. *(Archive: "Study-aware
  detection", the readability audit.)*
- **The fleet cache is one declared build.** Placeholder values masked, plausible ranges applied,
  nitrate clipped, nine floats left out, plus a per-cycle salinity fallback and a backscatter
  smoother. It is 2,542 fine grids, fingerprint `11fce215…`
  (`data/candidates/net_carbon_v1/CACHE_IDENTITY.json`). It is not in git; it is in the data
  deposit. **Why:** the grids cannot be rebuilt bit for bit from a fresh Argo pull. *(Archive: the
  cache decisions, the fourth build.)*
- **The cache builder is generic and lives in argopod** (`argopod.cache`, v0.5.2). The study keeps
  only what is its own. That is the recipe — the four grid kinds and their channels, the dates, the
  placeholder rule, the check ceilings — in the `cache:` block of `config/events.yaml`. Code joins
  it to the plausible ranges, the nine floats left out and the backscatter smoother written in the
  same file. The study also keeps the float list, `config/fleet.csv`: 2,574 floats, each promised a
  grid kind. A build never targets the bound cache's own directory. **Why:** the study should hold
  the choices and none of the plumbing, and the recipe should be readable in one place. *(Evidence,
  2026-09-04: four floats, one per grid kind — 2901074, 1901339, 1901378, 6903247 — rebuild to
  grids whose sha256 match the bound cache exactly. The baseline was the old script's own rebuild
  on today's libraries, and `argopod.cache.cache_identity` returns the same fingerprint the study
  checks. The old script, `scripts/build_paper_cache.py` at `eddy-pump-archive` tag
  `archive-2026-09-03`, is retired.)*
- **A batch is drawn once.** `pipeline/draw_batch.py` takes the batch name and refuses a batch
  that already has a draw record or an entry in the label table; there is no override. **Why:** the
  draw record is the frame and the error bar a rate stands on, and a second draw under the same
  name would replace it silently. Found 2026-09-04: the held region had left the code, so a redraw
  of the first upward batch would have recorded 186,275 levels instead of 171,578 while the same
  576 labels stayed. *(Test: `tests/test_batches.py`, the refusal.)*
- **The held region is a fixed list of keys from the earlier study's detector**, not a cut on this
  study's residuals: `data/external/letter_pool_features.parquet` (sha256 `67b129d8…`, checked
  before every draw), the levels that score ≥ 1.96 σ on both AOU and salinity in that table.
  14,697 levels in five pressure bands (4,931 / 4,777 / 2,158 / 1,493 / 1,338). This study's own
  columns give 14,688, 47 of them different. **Why:** the open region of the first upward batch is
  the complement of exactly this list; the two must partition the pool. *(Test:
  `test_the_region_the_second_upward_batch_samples_is_the_one_the_first_draw_held_back`.)*
- **Nine floats are left out of the cache**, each named in `config/events.yaml` with one sentence
  of why. Two have salinity that breaks intermittently; seven have a nitrate sensor that reads high
  everywhere. **Why:** a plausible range removes the levels that cross it and certifies the wrong
  ones that do not. *(Archive: the entries of 2026-08-25, which hold the per-float statistics, the
  co-located floats and the candidate counts.)*
- **The salinity magnitude is a reported covariate, never a test a candidate must pass.** **Why:**
  the noise floor was refused against the verified events. *(Archive: the salinity floor.)*

## The labels and the numbers

- **Every science number is a pure function of the human labels.** The classifier only orders which
  panels a human sees first; it never enters a number. **Why:** honesty by construction.
- **A number reads its list with the hash checked.** The rate report and the draw read a saved
  list through `read_saved(verify=True)`: the key hash, the spec and the cache block must match
  the sidecar. The feature step records the cache fingerprint it measured, never the one it was
  told. **Why:** a check that compares a value with itself cannot fail (found 2026-09-04).
- **Every rate comes from a probability sample with a declared frame and inclusion probability.**
  Score-selected labels are training only and every rate function rejects them. A uniform-random
  precision measures a rate; a score-ranked one is 7–15× higher on the same ocean and is barred.
  **Why:** the two kinds of number must never be confused. *(Archive: the labelling levers.)*
- **The criterion a human labels against is the four-clause protocol on both limbs,
  `phys_net_carbon_v1`;** clause 4 applies to both limbs. **Why:** one criterion, one design.
  *(Archive: the criterion.)*
- **The precision target is ±15 % relative per limb** — assumed, pending the advisor. *(Archive: the
  precision target.)*
- **The old (earlier GRL letter) labels left the repository; a rate comes only from the study's own
  probability samples under `phys_net_carbon_v1`.** Old labels may be fetched from the deposit as
  classifier training evidence only, never a rate — key overlap is not interchangeability. **Why:**
  one criterion, one design; the old labels were judged under a different rule. *(Archive: the reuse
  audit, and "The fresh repository confirmed", 2026-09-03.)*
- **There is one label table**: `data/labels/study_reviews.parquet` with `study_batches.yaml` and
  `draws/*.yaml`. No other label layer exists in the code. **Why:** one door a rate may use, so a
  number cannot quietly mix two rules. *(Done 2026-09-03; see `docs/STATUS.md`.)*
- **The upward rate covers the whole pool from two random samples** (since 2026-09-04): the open
  region (171,578 levels) and the held region (the other 14,697, picked earlier by the retired
  letter's stricter 1.96 σ cut, drawn as `rate_obduction_02` under this study's criterion). No rate
  extrapolates onto it. **Why:** a rate may only cover the levels its sample covers.
  *(Done 2026-09-03; see `docs/STATUS.md` and `docs/PLAN.md`.)*
- **Every companion subduction event is a reference event;** one the detector misses is a detector
  defect, never a bad label. *(Archive: decisions taken.)*

## The classifier

Three stages. Triage now: it orders candidates into score strata. A calibrated census next
(`census.py`), with precision and recall from the labelled samples. A wider candidate net after the
OSSE. The paper's rates stay human-labelled. Every probability-sample label is both a rate's sample
and the classifier's out-of-fold validation set. **Why:** it reaches "a classifier that finds the
events" without a number that cannot be checked against a label. *(Archive: "The fresh repository
confirmed", 2026-09-03.)*

## The event key

`(WMO, CYCLE_NUMBER, round(PRES_ADJUSTED))` plus the pool. **Why:** the level is the unit of a rate;
the pool tells apart a level that is a candidate in more than one.
