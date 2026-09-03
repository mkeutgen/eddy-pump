# The study's labelling batches — built 2026-08-27

Built by `pipeline/draw_batch.py`; the design is `eddy_pump.batches`. Worksheets, keys and panels are under `results/net_carbon_v1/labeling/<batch_id>/` (not in Git); the draw records beside this page are the ledger's input when a sheet comes back labelled.

| batch | role | rows | of which science / controls | hours | expected precision | gate before labelling |
|---|---|---:|---|---:|---|---|
| `calib_obduction_b6` | calibration | 42 | 0 / calibration 42 | 0.3 | — | — |
| `calib_subduction_v1` | calibration | 42 | 0 / calibration 42 | 0.3 | — | — |
| `rate_obduction_01` | analysis | 619 | 579 / positive 20 / negative 20 | 4.9 | ±15.0% relative on the pool rate (target ±15%) | calib_obduction_b6 PASS the same day |
| `rate_subduction_01` | analysis | 832 | 792 / positive 20 / negative 20 | 6.6 | ±15.0% relative on the pool rate (target ±15%) | calib_subduction_v1 adjudicated and frozen, then re-labelled blind: PASS |

## How to label one

```
make review BATCH=results/net_carbon_v1/labeling/<batch_id>/<batch_id>.csv   # the keyboard app, blind
python pipeline/draw_batch.py --report results/net_carbon_v1/labeling/calib_obduction_b6/calib_obduction_b6.csv
```

The worksheet carries the key, the position and the coordinates — nothing else. The answer key beside it (`ANSWER_KEY_do_not_open.csv`) is opened by `argopod session` only once the sheet is finished.

## `rate_obduction_01` — the allocation

Planning base rate 0.1366; target variance 0.000109; the open draw contributes 9.87e-05. Neyman against proportional at the same n: ×0.80. (Frozen draw record; the held region it also drew is no longer credited by the rate — the open strata below are the sample.)

| stratum | N | floats | score range | planned p | n | π = n/N |
|---|---:|---:|---|---:|---:|---:|
| open|d0 | 15,880 | 1,480 | 0.0001–0.0014 | 0.016 | 29 | 0.00183 |
| open|d1 | 16,993 | 1,598 | 0.0014–0.0024 | 0.017 | 29 | 0.00171 |
| open|d2 | 17,227 | 1,630 | 0.0024–0.0038 | 0.018 | 29 | 0.00168 |
| open|d3 | 17,372 | 1,597 | 0.0038–0.0057 | 0.021 | 32 | 0.00184 |
| open|d4 | 17,386 | 1,543 | 0.0057–0.0088 | 0.034 | 41 | 0.00236 |
| open|d5 | 17,430 | 1,563 | 0.0088–0.0141 | 0.061 | 54 | 0.00310 |
| open|d6 | 17,407 | 1,512 | 0.0141–0.0246 | 0.132 | 76 | 0.00437 |
| open|d7 | 17,479 | 1,450 | 0.0246–0.0518 | 0.167 | 84 | 0.00481 |
| open|d8 | 17,488 | 1,406 | 0.0518–0.1612 | 0.261 | 99 | 0.00566 |
| open|d9 | 16,916 | 1,201 | 0.1612–0.9983 | 0.633 | 106 | 0.00627 |


## `rate_subduction_01` — the allocation

Planning base rate 0.1366; target variance 0.000109; the open draw contributes 9.92e-05. Neyman against proportional at the same n: ×0.80. (The downward pool is sampled whole; no held region.)

| stratum | N | floats | score range | planned p | n | π = n/N |
|---|---:|---:|---|---:|---:|---:|
| open|d0 | 13,331 | 979 | 0.0016–0.0200 | 0.005 | 40 | 0.00300 |
| open|d1 | 13,331 | 1,231 | 0.0200–0.0346 | 0.009 | 40 | 0.00300 |
| open|d2 | 13,331 | 1,346 | 0.0346–0.0545 | 0.021 | 40 | 0.00300 |
| open|d3 | 13,330 | 1,448 | 0.0545–0.0830 | 0.033 | 48 | 0.00360 |
| open|d4 | 13,331 | 1,514 | 0.0830–0.1246 | 0.071 | 70 | 0.00525 |
| open|d5 | 13,331 | 1,555 | 0.1246–0.1826 | 0.105 | 83 | 0.00623 |
| open|d6 | 13,330 | 1,529 | 0.1826–0.2631 | 0.153 | 98 | 0.00735 |
| open|d7 | 13,331 | 1,554 | 0.2631–0.3814 | 0.211 | 111 | 0.00833 |
| open|d8 | 13,331 | 1,491 | 0.3814–0.5666 | 0.311 | 126 | 0.00945 |
| open|d9 | 13,330 | 1,274 | 0.5666–0.9607 | 0.448 | 136 | 0.01020 |
