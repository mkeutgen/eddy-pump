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

## Session rules

One sheet is one sitting. These rules exist because the review of 2026-09-04 found the reading
moving inside the two first rate sessions, by two to three times the sampling error, and the net of
the two limbs changes sign across that band.

- **At most 120 panels in a sheet.** A draw that needs more is written as several sheets of the same
  draw — `<batch>_s1`, `_s2`, and so on, one sitting each. The draw is untouched: same panels, same
  order, same inclusion probabilities, so the rate is the one a single sheet would have given.
  `--allow-long` writes one long sheet instead, and the record says so.
- **The controls are spread through the sheet by the draw, not by luck.** They sit at evenly spaced
  positions, and each arm is spread across those positions, so every stretch of the sheet carries
  its share of both. In the first upward sheet a plain shuffle put 17 of the 20 negative controls in
  the second half, where they could not see the reading change that half was showing.
- **The calibration copy is labelled on an earlier day than the sheet.** Doing both in one afternoon
  makes the check the same reading it is meant to test. The loader warns rather than refuses: it may
  be the right thing to do on the day.
- **Every rate sheet gets a blind second look afterwards** — 100 of its own panels, re-shown and
  re-judged (see below). Draw it before loading the batch.

What the loader refuses, and what it only records:

| what it reads | what happens |
|---|---|
| the 42 calibration panels were not re-labelled PASS before the sheet | refuses |
| acceptance fell from the first half of a sitting to the second, holding the stratum fixed (p < 0.01) | refuses, unless a blind second look of that batch has been drawn, or `--accept-drift` is given — which the record then carries |
| a control arm is bunched into part of the sheet (p < 0.01) | records it, loud |
| the calibration copy was labelled the same day as the sheet | records it, loud |

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
| `rejudge_obduction_01` | a blind second look at 100 panels of `rate_obduction_01`, 50 from each half |
| `rejudge_subduction_01` | a blind second look at 100 panels of `rate_subduction_01`, 50 from each half |

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

## The blind second look

After every rate sheet, 100 of its own panels come back: 50 from the first half of the sitting and
50 from the second, drawn at random, shuffled together with 10 positive and 10 negative controls,
their panels drawn again, in a new order with new positions. Nothing on the sheet says which panel
this is or what it was called the first time. The saved answers keep the first verdict, where the
panel sat in the first sheet, which half that was, its stratum and its score.

```bash
make draw-batch BATCH=rejudge_obduction_01     # or rejudge_subduction_01
make review BATCH=results/net_carbon_v1/labeling/rejudge_obduction_01/rejudge_obduction_01.csv
make load BATCH=rejudge_obduction_01
make rates
```

**What it measures.** How often the second look agrees with the first, split by which half of the
first sitting the panel came from. If the reading held, the two halves agree the same way. If it
moved, panels from the second half come back differently — and by how much is the size of the drift.

**It never enters a rate on its own.** It decides nothing, and its record says so (`decides: false`).
What it does is correct the rate of the sheet it re-judged. Each science row of that sheet stops
contributing its own accept or reject and contributes instead the chance the second look accepts a
panel of that first verdict from that half. The rate is then formed exactly as before.

**How it is reported.** Beside the rate as labelled, never instead of it:
`rate_drift_corrected` in `data/labels/audit/rate_status.csv`, with its own error bar, which carries
both the sampling error of the sheet and how well 100 panels pin the second look. Read it as *the
rate if the reviewer had read every panel the way the blind second look reads it*. The net of the
two limbs — downward minus upward, in accepted candidate levels — is quoted only when both limbs
carry that correction. Until then `RATE_STATUS.md` says the net is not quoted.

## What the labels become

The rate of a pool is the weighted mean of the accepted labels, each weighted by 1/π, with a
design-based standard error; its denominator is the candidate levels the sample covers. The upward
rate combines two probability samples that partition the pool: the open region
(`rate_obduction_01`) and the held region (`rate_obduction_02`, labelled 2026-09-04). The classifier's score decides only
which stratum a panel came from; it never enters the number. The current rates and their error bars
are `data/labels/audit/RATE_STATUS.md`. An arm stops when its limb is inside ±15 % relative.
