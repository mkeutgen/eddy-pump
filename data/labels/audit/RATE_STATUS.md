# RATE STATUS — the physical rates from the human labels *(2026-09-08)*

Producer: `pipeline/rates.py`. Criterion `phys_net_carbon_v1`. Every rate is a weighted Hájek mean of human verdicts with declared inclusion probabilities; the denominator is the candidate levels the sample covers. The next batch goes to **`net_carbon_v1/physical/obduction` (the wider half-width)**.

## `net_carbon_v1/physical/obduction` — 12.9% of 186,275 candidate levels, ±17% relative (target ±15%)

Sampled region (186,275 levels): stratified mean 0.1288, design-based SE 0.0109 (stratified bootstrap 0.0103; the naive float bootstrap 0.0131 is the conservative sensitivity and overstates a stratified draw) on 665 target verdicts over 469 floats (6 uncertain excluded; 3 strata with no accept, floored at the Jeffreys mean in the variance).

Rate: **0.1288 ± 0.0213** (≈ 23,996 accepted candidate LEVELS in the sampled region — not events; a cycle-level estimand needs its own denominator). Sampling precision only, at the session-average instrument. At the variance realised, the sample needs about **808 target panels** for ±15%: 143 more (1.1 h at the planning pace, 0.3 h at the realised 7 s/panel). **Drift band**: the rate would read 0.146 if the whole session had read like its first half, 0.108 like its second — a systematic term the sampling interval does not contain.

**Drift-corrected**: 0.0923 ± 0.0440 (≈ 17,187 accepted levels) — the rate if the reviewer had read every panel the way the blind re-judgement reads it. Error bar: the sampling error 0.0109 and how well rejudge_obduction_01 pins the second look 0.0196, added in quadrature. 576 of 665 science rows are corrected; rate_obduction_02 has no re-judgement and keeps its verdicts as labelled.

| what the reviewer first called it | half of the sitting | re-judged | called real again | second looks |
|---|---|---:|---:|---|
| not real | first | 39 | 1 (3%) | rejudge_obduction_01 1/39 |
| not real | second | 41 | 1 (2%) | rejudge_obduction_01 1/41 |
| real | first | 11 | 5 (45%) | rejudge_obduction_01 5/11 |
| real | second | 8 | 5 (62%) | rejudge_obduction_01 5/8 |

Session flags (recorded, never a filter):
- rate_obduction_01: acceptance fell with position (Mann-Whitney p = 0.006; 28% in the first quarter, 15% in the last; the rate would read 0.146 like the first half, 0.108 like the second)
- rate_obduction_01: positive controls 14/20 against the blind re-judgement history 48/61 (79%): Fisher p = 0.54 — within the instrument's own noise
- rate_obduction_01: negative controls 4/20 against their blind history 4/24 (17%): Fisher p = 1.00; 2 of the 2 with score >= 0.5 accepted (plausible detector misses)

| stratum | N | n | accepted | rate | share of variance |
|---|---:|---:|---:|---:|---:|
| former_held|200-260 | 4,777 | 30 | 4 | 0.133 | 2% |
| former_held|260-400 | 2,158 | 15 | 2 | 0.133 | 1% |
| former_held|400-600 | 1,493 | 10 | 3 | 0.300 | 1% |
| former_held|600-1000 | 1,338 | 8 | 1 | 0.125 | 1% |
| former_held|<=200 | 4,931 | 26 | 2 | 0.077 | 2% |
| open|d0 | 15,880 | 29 | 0 | 0.000 | 4% |
| open|d1 | 16,993 | 29 | 0 | 0.000 | 4% |
| open|d2 | 17,227 | 29 | 0 | 0.000 | 4% |
| open|d3 | 17,372 | 32 | 1 | 0.031 | 7% |
| open|d4 | 17,386 | 41 | 2 | 0.049 | 9% |
| open|d5 | 17,430 | 53 | 4 | 0.075 | 10% |
| open|d6 | 17,407 | 76 | 11 | 0.145 | 12% |
| open|d7 | 17,479 | 83 | 13 | 0.157 | 12% |
| open|d8 | 17,488 | 99 | 27 | 0.273 | 15% |
| open|d9 | 16,916 | 105 | 58 | 0.552 | 17% |

## `net_carbon_v1/physical/subduction` — 18.7% of 133,307 candidate levels, ±14% relative (target ±15%)

Sampled region (133,307 levels): stratified mean 0.1874, design-based SE 0.0131 (stratified bootstrap 0.0130; the naive float bootstrap 0.0140 is the conservative sensitivity and overstates a stratified draw) on 779 target verdicts over 548 floats (15 uncertain excluded; 1 strata with no accept, floored at the Jeffreys mean in the variance).

Rate: **0.1874 ± 0.0257** (≈ 24,980 accepted candidate LEVELS in the sampled region — not events; a cycle-level estimand needs its own denominator). Sampling precision only, at the session-average instrument. At the variance realised, the sample needs about **652 target panels** for ±15%: 0 more. **Drift band**: the rate would read 0.223 if the whole session had read like its first half, 0.149 like its second — a systematic term the sampling interval does not contain.

**Drift-corrected**: not reported — the calibration copy labelled before rejudge_subduction_01 (on 2026-09-08) read stricter than the reference: 13/42 accepted against the reference's 17/42, κ 0.73 — so that second look does not correct a rate. Its rows stay in the label table. A fresh second look, labelled after a calibration copy that passes, is what fills this in.

Session flags (recorded, never a filter):
- rate_subduction_01: acceptance fell with position (Mann-Whitney p = 0.005; 27% in the first quarter, 19% in the last; the rate would read 0.223 like the first half, 0.149 like the second)

| stratum | N | n | accepted | rate | share of variance |
|---|---:|---:|---:|---:|---:|
| open|d0 | 13,331 | 40 | 0 | 0.000 | 2% |
| open|d1 | 13,331 | 39 | 4 | 0.103 | 14% |
| open|d2 | 13,331 | 40 | 4 | 0.100 | 13% |
| open|d3 | 13,330 | 48 | 8 | 0.167 | 17% |
| open|d4 | 13,331 | 70 | 9 | 0.129 | 9% |
| open|d5 | 13,331 | 81 | 7 | 0.086 | 6% |
| open|d6 | 13,330 | 98 | 16 | 0.163 | 8% |
| open|d7 | 13,331 | 110 | 23 | 0.209 | 9% |
| open|d8 | 13,331 | 122 | 43 | 0.352 | 11% |
| open|d9 | 13,330 | 131 | 74 | 0.565 | 11% |

## The net — not quoted

The net of the two limbs is not quoted. The 2026-09-04 review found the reviewer's reading moving within each of the two first sittings by two to three times the sampling error, and the net changes sign across that band. Both limbs need a second look that counts before it is quoted (docs/PLAN.md). Missing on:
- **subduction** — not reported — the calibration copy labelled before rejudge_subduction_01 (on 2026-09-08) read stricter than the reference: 13/42 accepted against the reference's 17/42, κ 0.73 — so that second look does not correct a rate. Its rows stay in the label table. A fresh second look, labelled after a calibration copy that passes, is what fills this in.
