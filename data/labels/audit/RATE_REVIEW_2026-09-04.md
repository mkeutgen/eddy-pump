# The adversarial review of the two physical rates, 2026-09-04

Three independent checks on Opus, each with its own code, verified by Fable. The numbers under
review are the ones `pipeline/rates.py` wrote to `rate_status.csv` on 2026-09-04 (repository at
commit b31e1d8):

| limb | rate | of | design 95 % half-width | batches |
|---|---:|---:|---:|---|
| downward (physical/subduction) | 0.18739 | 133,307 levels | ±13.7 % | rate_subduction_01 |
| upward (physical/obduction) | 0.12882 | 186,275 levels | ±16.5 % | rate_obduction_01 (open region), rate_obduction_02 (held region) |

## Verdict

**Both rates hold as estimates of what this reviewer called real. The net does not.**

- The arithmetic holds. Two independent recomputations reproduce both rates and all three
  standard errors to eight decimals. Every inclusion probability equals n_h / N_h. The strata
  partition each pool key for key. No duplicate level, no control, no calibration row enters a
  rate. The headline error bar uses the design variance, which is not the friendliest of the three.
- The provenance holds. The six lists, the cache fingerprint, the feature and score files, the
  eight draw records and the ten labelled sheets all hash to what the records say. Nesting of the
  four tracer pools is exact. No classifier score reaches the estimator.
- **The reviewer drifted within each session, by more than the sampling error.** Acceptance fell
  with position in both first batches, and the fall is not stratum mix (within-stratum test,
  p = 0.003 downward, 0.008 upward). The held-region batch shows no trend.

## The one thing to know

The drift band is 2.9 times the error bar downward and 1.8 times upward. The two limbs drifted
for different reasons, so they do not cancel:

- downward: the threshold moved. Low-score panels stopped being accepted (0.22 to 0.10 across the
  session); high-score acceptance did not change.
- upward, open region: fatigue. High-score panels stopped being accepted (0.83 to 0.33). The
  session ran at about 7 seconds per panel over 619 panels.

The reviewer's own re-look agrees: 32 of the 42 new upward calibration panels were panels already
judged in the open-region batch, and on the second look 4 of 21 accepts became rejects while 0 of
11 rejects became accepts.

The net (accepted levels down minus up) changes sign across the band:

| reading | down | up | net |
|---|---:|---:|---:|
| as labelled | 24,980 | 23,996 | +984 |
| the whole session like its first half | 29,744 | 27,235 | +2,509 |
| the whole session like its second half | 19,825 | 20,074 | −248 |

Honest intervals, drift plus sampling: **downward 0.123 to 0.249, upward 0.087 to 0.168.**
Do not quote a net until the drift is measured and corrected.

## What the paper must say the rates are

A rate is the fraction of candidate LEVELS the detector flagged at 1.50 σ that a human called real.
One level is one depth in one dive. It is not a fraction of profiles, of floats or of the ocean,
and it says nothing about events the detector never flagged. Candidates sit on a 40 dbar detection
grid at 22 pressures from 140 to 980 dbar: nothing shallower, nothing below 1000 dbar. The frame is
2,542 floats (2,574 promised, nine excluded by name, 23 with no grid), profiles from 2009-01-08 to
2026-03-15. `accepted_levels_estimated` is a count of levels, never of events.

## Smaller findings

1. The score file that cut the open-region deciles on 2026-08-27 no longer exists: the upward
   classifier was retrained on 2026-09-03. The rate is untouched (N_h and π are frozen in the draw
   record), but the decile membership of the unsampled levels cannot be re-derived, so no more
   panels can be added to those strata until the file is regenerated. Its recipe is in the archive
   at commit e7a625c (`SCORES_SHA256`: HistGradientBoosting, seed 0, 168 features, 4,824 labels).
2. Negative controls in rate_obduction_01 were not interleaved: 17 of 20 sit in the second half.
   They also come from the same reviewer's earlier verdicts, so they cannot see this reviewer drift.
   The downward controls come from the companion catalogue and are clean.
3. The upward calibration reference was decided the same afternoon as pass 2 by the same reviewer:
   kappa 0.95 is self-consistency within one day, not an independent check.
4. `pipeline/rates.py:109-118`: the pace and the "anchor" columns mix batches (7 s/panel upward,
   102 s downward, anchor calib_obduction_b6); `uncertain_excluded` counts control rows too.
5. argopod `review/session.py:435`: the trend test runs on all decided rows, controls included.
6. rate_obduction_02 caps negative controls at score < 0.5; rate_obduction_01 did not, so their
   20 % ceilings are not comparable.
7. `calib_obduction_v1_pass2.yaml` is missing from `DRAWS_SHA256`.
8. Stale sentences: `docs/DECISIONS.md`, `docs/IMPLEMENTATION_NOTES.md`, `docs/LABELING_PROTOCOL.md`
   still said the upward rate is the open region alone; `README.md` said "two rates" and listed one.
   Fixed 2026-09-04 with this review.
9. Two of the three labelled rate sheets exist only as local, gitignored files (and in the archive's
   working tree). The label table in git carries the labels; the sheets are the raw record.

## What would have caught the drift earlier

A rule in the protocol and a check in the loader: a session is at most about 120 panels; controls
are interleaved by construction (one every N rows, checked by the loader); and each rate batch gets a
blind re-judgement of a random subset (about 100 panels, shuffled) whose flip rate by original
position is reported beside the rate. The loader should refuse a batch whose within-stratum
position trend fails at p < 0.01 until that re-judgement exists.

## Sensitivity, for the record

- uncertain rows all accepted / all rejected: downward 0.1835 to 0.1962; upward 0.1279 to 0.1324.
- upward zero-count strata at their rule-of-three ceiling: 0.1566.
- standard errors: design 0.0131 / 0.0109; stratified bootstrap 0.0130 / 0.0104; naive float
  bootstrap 0.0140 / 0.0131.
