# STATUS — what was done, on which day

The dated log: one entry per working day, newest first. Every number here was read from the file
that produces it, on the date shown. The plan is `docs/PLAN.md`; the decisions in force are
`docs/DECISIONS.md`; the words are defined in `docs/IMPLEMENTATION_NOTES.md`.

## 2026-09-04, night

**The adversarial review of both rates: they hold as measured; the net does not.**

- Three independent checks: the arithmetic and every hash reproduce (both rates to eight decimals;
  lists, cache, scores, draws, sheets). No score, control or calibration row reaches a rate.
- The reviewer drifted within both first sessions by more than the sampling error (2.9× downward,
  1.8× upward), a moved threshold on one limb and fatigue on the other. The net of accepted levels
  reads +2,509 / +984 / −248 across the band. Intervals: downward 0.123–0.249, upward 0.087–0.168.
- Record: `data/labels/audit/RATE_REVIEW_2026-09-04.md`. Next: a blind re-judgement of 100 panels
  per batch to measure the drift, and the 2026-08-27 upward score file regenerated from the
  archive's recipe so the open-region strata can be reused.

## 2026-09-04, evening

**The upward limb stands on its own footing: both batches labelled, the whole pool sampled.**

- `calib_obduction_v1`: two blind passes, 40 of 42 agree; the two "unsure" of pass 1 decided by
  hand (both no); reference frozen at 18/42; pass 2 against it: kappa 0.95, base rate on target, PASS.
- `rate_obduction_02`, the held region, 128 panels: 30 accepted, 95 rejected, 3 unsure. Positive
  controls 15/16, negative 3/20 (under the 20 % ceiling); acceptance flat with position (p = 0.95).
- The upward rate now covers the whole pool: 12.9 % of 186,275 levels (0.1288, ±16.5 % relative;
  before: 12.9 % of 171,578, ±17.4 %). `make rates` says 143 more open-region panels reach ±15 %
  (about 1.1 h). The downward rate is unchanged.

## 2026-09-04, later

**Three audits, every finding fixed, in both repositories.**

- The batch draw takes a batch name and refuses one that already has a record. The blind copy is
  reproducible. The rate report and the draw check the saved list's hash. The feature step records
  the fingerprint it measured. `EDDY_PUMP_CACHE` overrides the one-machine cache path. Re-detection:
  all six pools exact. The scores: byte-identical.
- The two upward batches are drawn and rendered: `calib_obduction_v1` (42 panels) and
  `rate_obduction_02` (128 panels: 90 science, 38 controls, 83 floats, ±40 % on the held region's
  own rate). The held region comes from the earlier study's detection table
  (`data/external/letter_pool_features.parquet`, hash checked): this study's own residuals give
  14,688 of its 14,697 levels, so the list is kept as data.
- argopod 0.5.2: one grid builder, a checker that fails on an empty cache, a run that survives a
  dead worker, provenance with library versions, `label` renamed `grid_kind`. Parity holds. Tests:
  study 140 (from 102), argopod 529 (from 516). The words: rejected terms out of the prose, STATUS
  ten lines per day, a 15-word glossary.

## 2026-09-04

**The fleet-cache builder moved into argopod, and the study now says how its cache is built.**

- The builder is generic, so it lives in the library (`argopod.cache`, v0.5.1). The study keeps the
  recipe (`config/events.yaml`) and the float list (`config/fleet.csv`, 2,574 floats).
- The check: four floats, one per grid kind — 2901074, 1901339, 1901378, 6903247 — rebuild to 8
  grid files. Every sha256 matches the bound cache exactly. `make check-cache`, about a minute.
- Tests: 102 pass, 84 before, and the same in strict mode. Changed: the cache recipe and float
  list, `study.py`, `manifest.py`, `pipeline/build_cache.py`, the make rules, one new test file.

## 2026-09-03, later

**One label layer, one design: the old machinery left the code.**

- The downward rate reproduces to the bit: 0.18738977257005085 — 18.7 % of 133,307 candidate
  levels, ±14 % relative.
- The upward rate becomes the open region alone: 0.12867083136731428 — 12.9 % of 171,578, ±17 %.
  The held region, 14,697 levels, waits for a draw of its own.
- The upward classifier now trains on the study's own 576 obduction labels (out-of-fold AUC 0.75;
  the downward 0.86 is unchanged). The score never enters a rate. Tests: 84 pass.

## 2026-09-03

**The repository seeded from the archive.**

- Cut from `mkeutgen/eddy-pump-archive`, tag `archive-2026-09-03`, commit `cbd6fd6`, keeping only
  the study's essentials: 73 files, 28 MB. `production/` became `pipeline/`.
- The check passed: the six pools load, and each pool's spec id, row count and key hash match the
  archive. All 14 candidate files are byte-identical to it.
- The labelling loop and the rate report still carried the pre-study label layer at this point. It
  was removed the same day, in the entry above.
