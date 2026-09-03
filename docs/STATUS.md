# STATUS — what was done, on which day (2026-09-03)

The plan is `docs/PLAN.md`; the rulings in force are `docs/DECISIONS.md`. This file is the dated
log: one entry per working day, newest first, saying what was done and what was verified. Every
number here was read from its producer on the date shown.

## The log

| date | what was done | what moved | where |
|---|---|---|---|
| **2026-09-03** | **The repository seeded.** Cut fresh from `mkeutgen/eddy-pump-archive` (tag `archive-2026-09-03`, commit `cbd6fd6`), keeping only the study's essentials: the six saved candidate lists, the study's own labels, the definitions, the pipeline and the code. `production/` became `pipeline/`; the six scripts got plain names (`detect.py`, `features.py`, `scores.py`, `draw_batch.py`, `ingest_batch.py`, `rates.py`). Left behind (deposit / archive): the old labels (`verdicts.csv`, the two-layer ledger, the four curated files), the reuse audit and its outputs, the old paper's three candidate tables, `data/legacy/`, the cache builder, `legacy_labels.py`, the old record. 73 files, 28 MB. **Gate passed**: `load_manifest()` loads the six pools; each pool's spec id, row count and key hash (the sidecar `content_sha256`) match the archive; all 14 candidate files are byte-identical to the archive. The labelling loop and the rate report still carry the pre-study label layer — the "one label layer" step (`docs/PLAN.md`) removes it. | the whole tree; first commit | the seed (`docs/DECISIONS.md`, "How this repository was born") |
