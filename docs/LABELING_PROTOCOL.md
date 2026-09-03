# The labelling protocol — both limbs

One human labels candidate panels against four written clauses. The criterion is `phys_net_carbon_v1`
in `config/criteria.yaml`; it applies to both limbs (ruled 2026-08-26). A subduction event is a
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

42 panels per limb. Upward: `calib_obduction_b6` (18 of 42 accepted; one pass). Downward:
`calib_subduction_v1` (17 of 42; two blind passes, κ 0.76, the five disagreements decided by hand
on 2026-09-01). The frozen answers are `data/labels/draws/<batch>.reference.yaml`.

## Every session

```bash
python production/build_batches.py --repass calib_<limb>_…   # a fresh blind copy of the 42 panels
make review BATCH=results/net_carbon_v1/labeling/<copy>/<copy>.csv
make calibrate SHEET=<the labelled copy>          # must PASS: kappa > 0.6, base rate on target
make review BATCH=results/net_carbon_v1/labeling/<batch>/<batch>.csv   # then the batch, blind
make ingest BATCH=<batch>                          # refuses unless the calibration passed
make rates                                         # the rate report
```

If `make calibrate` says DRIFTED: re-examine the listed disagreements, decide your rule, re-label
until you pass — before touching new data.

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
ingest reads acceptance against position in the session and the controls; both are recorded, never
used to drop a label. Unsure (2) is allowed and is excluded from the rate, counted.

## What the labels become

The rate of a pool is the weighted mean of the accepted labels, each weighted by 1/π, with a
design-based standard error; its denominator is the pool's candidate levels. On the upward limb the
region the old labels already cover (the old ≥ 1.96 σ pool) is held at its direct sample and
combined with the open region by their share of the pool. The classifier's score decides only which
stratum a panel came from; it never enters the number. The current rates and their error bars are
`data/labels/audit/RATE_STATUS.md`. An arm stops when its limb is inside ±15 % relative.
