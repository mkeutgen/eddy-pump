# RATE STATUS — the physical rates from the human labels *(2026-09-04)*

Producer: `pipeline/rates.py`. Criterion `phys_net_carbon_v1`. Every rate is a weighted Hájek mean of human verdicts with declared inclusion probabilities; the denominator is the candidate levels the sample covers. The next batch goes to **`net_carbon_v1/physical/obduction` (the wider half-width)**.

## `net_carbon_v1/physical/obduction` — 12.9% of 186,275 candidate levels, ±17% relative (target ±15%)

Sampled region (186,275 levels): stratified mean 0.1288, design-based SE 0.0109 (stratified bootstrap 0.0103; the naive float bootstrap 0.0131 is the conservative sensitivity and overstates a stratified draw) on 665 target verdicts over 469 floats (6 uncertain excluded; 3 strata with no accept, floored at the Jeffreys mean in the variance).

Rate: **0.1288 ± 0.0213** (≈ 23,996 accepted candidate LEVELS in the sampled region — not events; a cycle-level estimand needs its own denominator). Sampling precision only, at the session-average instrument. At the variance realised, the sample needs about **808 target panels** for ±15%: 143 more (1.1 h at the planning pace, 0.3 h at the realised 7 s/panel). **Drift band**: the rate would read 0.146 if the whole session had read like its first half, 0.108 like its second — a systematic term the sampling interval does not contain.

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
