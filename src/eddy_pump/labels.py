"""The label table's query API — the only door a rate may use.

reads  data/labels/{study_batches.yaml, study_reviews.parquet} (drawn by pipeline/draw_batch.py,
       ingested once labelled by pipeline/ingest_batch.py)
writes nothing
`resolve_reviews(pool, criterion)`: one decision per candidate through the explicit supersedes chain.
`analysis_sample(pool, criterion)`: the probability sample a rate may use (role analysis, a probability
design, the target arm). `labelled_keys()`: the keys already labelled, the exclusion set for a new batch.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from .manifest import REPO_ROOT

LABELS = REPO_ROOT / "data" / "labels"
# The study's one label layer: batches drawn by pipeline/draw_batch.py and ingested, once labelled,
# by pipeline/ingest_batch.py. Every review carries the inclusion probability of its draw.
STUDY_BATCHES = LABELS / "study_batches.yaml"
STUDY_REVIEWS = LABELS / "study_reviews.parquet"

ROLES = ("analysis", "training", "catalogue", "audit", "calibration", "control", "snapshot", "answer_key")
PROBABILITY_DESIGNS = ("probability",)
_BATCH_COLS = ["batch_id", "role", "decides", "criterion_version", "sampling_design", "sampling_frame"]


def _flatten(raw_batches: list[dict]) -> list[dict]:
    rows = []
    for b in raw_batches:
        r = {k: v for k, v in b.items() if not isinstance(v, dict)}
        for k in ("sampling",):
            for kk, vv in (b.get(k) or {}).items():
                r[f"{k}_{kk}"] = vv
        rows.append(r)
    return rows


@lru_cache(maxsize=1)
def load_batches() -> pd.DataFrame:
    """One row per study batch, flattened: `sampling.*` becomes `sampling_*`."""
    import yaml

    if not STUDY_BATCHES.exists():
        raise FileNotFoundError(f"missing {STUDY_BATCHES} — draw a batch (pipeline/draw_batch.py) and "
                                f"ingest it (pipeline/ingest_batch.py) first")
    rows = _flatten(yaml.safe_load(STUDY_BATCHES.read_text(encoding="utf-8"))["batches"])
    B = pd.DataFrame(rows)
    bad = set(B.role) - set(ROLES)
    if bad:
        raise ValueError(f"{STUDY_BATCHES}: unknown roles {sorted(bad)}")
    if not B.batch_id.is_unique:
        raise ValueError("a batch id is repeated")
    return B


@lru_cache(maxsize=1)
def load_reviews() -> pd.DataFrame:
    """Every study review, in sheet order, with its batch's role, design and frame."""
    if not STUDY_REVIEWS.exists():
        raise FileNotFoundError(f"missing {STUDY_REVIEWS} — ingest a labelled batch first "
                                f"(pipeline/ingest_batch.py)")
    R = pd.read_parquet(STUDY_REVIEWS)
    R = R.drop(columns=["study_id"], errors="ignore")   # the batch record is the authority for it
    B = load_batches()[_BATCH_COLS].drop(columns=["role", "criterion_version"])
    return R.merge(B, on="batch_id", how="left", validate="many_to_one")


def batches_of(pool_id: str) -> pd.DataFrame:
    B = load_batches()
    return B[B.pool_id == pool_id]


def _resolved(pool_id: str) -> pd.DataFrame:
    """Decided reviews of the pool in batches that decide, minus every superseded one."""
    R = load_reviews()
    d = R[(R.pool_id == pool_id) & R.decision.isin([0, 1]) & R.candidate_id.notna()
          & R.decides.fillna(False).astype(bool)]
    if d.empty and (R[(R.pool_id == pool_id) & R.decision.isin([0, 1])]).shape[0]:
        raise ValueError(f"{pool_id}: decided reviews exist but no batch of this pool carries a "
                         f"precedence relation (`decides`); nothing can be resolved")
    superseded = set(R.supersedes_review_id.dropna())
    d = d[~d.review_id.isin(superseded)]
    if d.candidate_id.duplicated().any():
        dup = d[d.candidate_id.duplicated(keep=False)].sort_values("candidate_id")
        raise ValueError(f"{pool_id}: resolution is not unique for {dup.candidate_id.nunique()} candidates; "
                         f"the supersedes chain is incomplete:\n{dup[['candidate_id', 'batch_id', 'review_id']].head()}")
    return d.reset_index(drop=True)


def resolve_reviews(pool_id: str, criterion_version: str) -> pd.DataFrame:
    """One decided review per candidate, under ONE criterion, by the explicit supersedes chain.

    The criterion is required: a candidate whose resolving review was judged under a different
    criterion (or under none the table can vouch for) is not returned. That is the rule —
    reuse keys on pool identity AND criterion — enforced by the signature.
    """
    if not criterion_version:
        raise ValueError("resolve_reviews needs a criterion_version")
    d = _resolved(pool_id)
    return d[d.criterion_version == criterion_version].reset_index(drop=True)


def analysis_sample(pool_id: str, criterion_version: str) -> pd.DataFrame:
    """The reviews a RATE may be computed from.

    Role `analysis` (a uniform draw), a probability design, judged under `criterion_version`,
    decided 0/1, and on the target arm only — a blind batch's constructed positives and
    negatives are controls, not draws. Raises if the pool has no such batch, or if any analysis
    batch is not a probability draw. A score-selected, catalogue, calibration, control, audit or
    snapshot batch never enters a rate.
    """
    if not criterion_version:
        raise ValueError("analysis_sample needs a criterion_version")
    B = batches_of(pool_id)
    A = B[B.role == "analysis"]
    if A.empty:
        raise ValueError(f"{pool_id}: no batch with role 'analysis' — nothing here may feed a rate")
    bad = A[~A.sampling_design.isin(PROBABILITY_DESIGNS)]
    if not bad.empty:
        raise ValueError(f"{pool_id}: analysis batches without a probability design: {bad.batch_id.tolist()}")
    A = A[A.criterion_version == criterion_version]
    if A.empty:
        raise ValueError(f"{pool_id}: no analysis batch judged under {criterion_version!r}")
    R = load_reviews()
    d = R[R.batch_id.isin(A.batch_id) & R.decision.isin([0, 1]) & R.candidate_id.notna()
          & (R.control_arm.isna() | (R.control_arm == "target"))]
    if d.candidate_id.duplicated().any():
        dup = d[d.candidate_id.duplicated(keep=False)]
        raise ValueError(f"{pool_id}: {dup.candidate_id.nunique()} candidates appear in more than one analysis batch — a "
                         f"probability sample counts each unit once; resolve before any rate:\n{dup[['candidate_id', 'batch_id']].head()}")
    return d.reset_index(drop=True)


@lru_cache(maxsize=1)
def _batches_with_label_column() -> frozenset:
    B = load_batches()
    return frozenset(b for b, cols in zip(B.batch_id, B.columns_kept) if "LABEL" in cols)


def labelled_keys(exclude_batch_prefix: str = "nitrate/",
                  exclude_roles: tuple[str, ...] = ("answer_key",)) -> set:
    """Every event key the study has labelled — the exclusion set for a new batch.

    A sheet WITH a LABEL column contributes its decided rows (0/1/2); a sheet WITHOUT one
    contributes every row. The keys are the normalised (WMO, CYCLE_NUMBER, round(PRES)) triples,
    as floats. A probability draw never EXCLUDES them from its frame — they are a covariate,
    `previously_judged`; this is for reporting.
    """
    R = load_reviews()
    has_label = R.batch_id.isin(_batches_with_label_column())
    keep = (~R.batch_id.str.startswith(exclude_batch_prefix)) & ~R.role.isin(exclude_roles)
    keep &= (has_label & R.decision.isin([0, 1, 2])) | ~has_label
    d = R[keep & R.key_wmo.notna() & R.key_cycle.notna() & R.key_pres.notna()]
    return set(zip(d.key_wmo.astype(float), d.key_cycle.astype(float), d.key_pres.astype(float)))
