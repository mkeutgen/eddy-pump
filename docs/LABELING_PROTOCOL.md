# The labelling protocol — both limbs

One human labels candidate panels against four written clauses. The criterion is `phys_net_carbon_v1`
in `config/criteria.yaml`; it applies to both limbs (decided 2026-08-26). A subduction event is a
negative AOU anomaly — young, oxygen-rich water pushed down — beside a salinity anomaly of either
sign; an obduction event is a positive AOU anomaly — old water lifted. Everything else is the same.

**Why a written criterion.** The profile does contain the signal: within one consistent session a
classifier reaches an AUC of 0.80–0.90. What broke earlier sessions was the criterion drifting
between them — in one blind test the acceptance rate slid from 15 % to 5 %. This protocol locks it.
Consistency, not volume.

## The criterion

On the panel, an event has:
1. **Colocated AOU and absolute-salinity peaks** at the same depth, both clearly present (AOU of the
   limb's sign; salinity either sign).
2. **Below the mixed layer.**
3. **Compact** — vertical extent well under ~200 m.
4. **Stands out against an otherwise regular background** — a distinct bulge, not one wiggle among
   many in a messy profile.

These are necessary, not sufficient. The real decision is in the borderline zone, which is what the
calibration panels pin down: when unsure, match the call you made on them, not your gut of the day.

## The calibration panels

42 panels per limb. Upward: `calib_obduction_b6` (18 of 42 accepted; one pass, and the panels come
from the earlier study, so `calib_obduction_v1` replaces it — 42 panels from this study's own pool,
to be labelled twice blind and then decided by hand). Downward: `calib_subduction_v1` (17 of 42; two
blind passes, κ 0.76, the five disagreements decided by hand on 2026-09-01). The frozen answers are
`data/labels/draws/<batch>.reference.yaml`.

## Every session

```bash
python pipeline/draw_batch.py --repass calib_<limb>_…   # a fresh blind copy of the 42 panels
make review BATCH=results/net_carbon_v1/labeling/<copy>/<copy>.csv
make calibrate SHEET=<the labelled copy>          # must PASS: kappa > 0.6, base rate on target
make review BATCH=results/net_carbon_v1/labeling/<batch>/<batch>.csv   # then the batch, blind
make load BATCH=<batch>                            # refuses unless the calibration passed
make rates                                         # the rate report
```

If `make calibrate` says DRIFTED: re-examine the listed disagreements, decide your rule, re-label
until you pass — before touching new data.

## Drawing a batch

```bash
python pipeline/draw_batch.py --list                    # the names it knows and which are drawn
make draw-batch BATCH=rate_obduction_02                 # draw one, render its panels
python pipeline/draw_batch.py rate_obduction_02 --dry-run   # print the plan, write nothing
```

A batch must be named; there is no default. The script **refuses** to draw a batch that already has
a record in `data/labels/draws/` or a row in `data/labels/study_batches.yaml`, and no flag overrides
that: the record says which levels the rate stands on, and drawing the same name again would swap
that frame for another one without a word. To try a different design, give it a new name.

The names it knows today:

| name | what it draws |
|---|---|
| `calib_obduction_b6` | the first 42 upward calibration panels, carried over from the earlier study |
| `calib_obduction_v1` | 42 fresh upward calibration panels from this study's own pool |
| `calib_subduction_v1` | 42 downward calibration panels |
| `rate_obduction_01` | the upward rate over the whole pool, ten score deciles |
| `rate_obduction_02` | the 14,697 upward levels the first upward draw held back, five pressure bands |
| `rate_subduction_01` | the downward rate over the whole pool, ten score deciles |

## What a batch is

A probability sample of the pool: the candidates are cut into ten strata by classifier score; the
number of panels per stratum comes from Neyman allocation (more where acceptance is uncertain, at
least 5 % in every stratum), sized for a target half-width of ±15 % relative; within a stratum, one
panel per float, every candidate with the same inclusion probability π = n/N. Twenty positive and
twenty negative blind controls are interleaved. The worksheet carries the key, the position and the
coordinates and nothing else; the answer key sits beside it as `ANSWER_KEY_do_not_open.csv` and is
opened only when the sheet is finished. The draw record (`data/labels/draws/<batch>.yaml`) holds
every stratum's N, n and π. `data/labels/draws/BATCHES.md` (generated) lists the batches drawn.

While labelling, watch the acceptance rate: on the upward limb the pool rate is about 13 %, on the
downward about 19 %; a drift toward 5 % (too strict) or 30 % (too loose) means recalibrate. The
load reads acceptance against position in the session and the controls; both are recorded, never
used to drop a label. Unsure (2) is allowed and is excluded from the rate, counted.

Two batches are shaped differently and say so in their own records. `rate_obduction_02` samples the
former held region, not the whole pool: its strata are five pressure bands rather than score
deciles, and its size is a budget of about 90 panels rather than a solve for ±15 %, because the
region is only 7.9 % of the pool by levels. The two calibration sets have no science rows at all —
42 panels each, chosen to sit half in the clear and half on the borderline.

## What the labels become

The rate of a pool is the weighted mean of the accepted labels, each weighted by 1/π, with a
design-based standard error; its denominator is the candidate levels the sample covers. The upward
rate is the open-region probability sample alone; the 14,697 levels of the pool that the first
upward draw held back are not yet sampled and are drawn by `rate_obduction_02`. The classifier's score decides only
which stratum a panel came from; it never enters the number. The current rates and their error bars
are `data/labels/audit/RATE_STATUS.md`. An arm stops when its limb is inside ±15 % relative.
