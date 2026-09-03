"""The label table's query API — the only door a rate may use.

reads  data/labels/{batches.yaml, reviews.parquet} (the old sheets, by production/build_ledger.py) and
       data/labels/{study_batches.yaml, study_reviews.parquet} (the study's, by ingest_study_batch.py)
writes nothing
`resolve_reviews(pool, criterion)`: one decision per candidate through the explicit supersedes chain.
`analysis_sample(pool, criterion)`: the probability sample a rate may use (role analysis, a probability
design, the target arm). `training_sample(pool)`: the classifier's labels. `labelled_keys()`: the
exclusion set for a new batch. `legacy_catalogue` / `legacy_random_sessions`: equivalence tests only.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from .manifest import REPO_ROOT

LABELS = REPO_ROOT / "data" / "labels"
BATCHES = LABELS / "batches.yaml"
REVIEWS = LABELS / "reviews.parquet"
# The STUDY layer: batches drawn by production/build_batches.py and ingested, once labelled, by
# production/ingest_study_batch.py. Same record shape, same review columns plus the inclusion
# probability every study review carries. Absent until the first sheet comes back.
STUDY_BATCHES = LABELS / "study_batches.yaml"
STUDY_REVIEWS = LABELS / "study_reviews.parquet"
LEGACY_STUDY_ID = "legacy_letter_v1"

ROLES = ("analysis", "training", "catalogue", "audit", "calibration", "control", "snapshot", "answer_key")
PROBABILITY_DESIGNS = ("probability",)
_BATCH_COLS = ["batch_id", "role", "decides", "criterion_version", "sampling_design", "sampling_frame",
               "legacy_precedence_rank", "legacy_nitrate_rank", "legacy_random_session_index",
               "legacy_priority_index", "legacy_is_adjudication", "study_id"]


def _flatten(raw_batches: list[dict]) -> list[dict]:
    rows = []
    for b in raw_batches:
        r = {k: v for k, v in b.items() if not isinstance(v, dict)}
        for k in ("sampling", "legacy"):
            for kk, vv in (b.get(k) or {}).items():
                r[f"{k}_{kk}"] = vv
        rows.append(r)
    return rows


@lru_cache(maxsize=1)
def load_batches() -> pd.DataFrame:
    """One row per batch, legacy then study, flattened: `sampling.*` and `legacy.*` become
    `sampling_*`, `legacy_*`. `study_id` tells the two layers apart."""
    import yaml

    if not BATCHES.exists():
        raise FileNotFoundError(f"missing {BATCHES} — run: python production/build_ledger.py")
    rows = _flatten(yaml.safe_load(BATCHES.read_text(encoding="utf-8"))["batches"])
    if STUDY_BATCHES.exists():
        rows += _flatten(yaml.safe_load(STUDY_BATCHES.read_text(encoding="utf-8"))["batches"])
    B = pd.DataFrame(rows)
    bad = set(B.role) - set(ROLES)
    if bad:
        raise ValueError(f"{BATCHES}: unknown roles {sorted(bad)}")
    if not B.batch_id.is_unique:
        raise ValueError("a batch id appears in both layers")
    return B


@lru_cache(maxsize=1)
def load_reviews() -> pd.DataFrame:
    """Every review, legacy then study, in sheet order, with its batch's role, design, frame,
    legacy positions and study id."""
    if not REVIEWS.exists():
        raise FileNotFoundError(f"missing {REVIEWS} — run: python production/build_ledger.py")
    R = pd.read_parquet(REVIEWS)
    if STUDY_REVIEWS.exists():
        S = pd.read_parquet(STUDY_REVIEWS)
        R = pd.concat([R, S], ignore_index=True)
    R = R.drop(columns=["study_id"], errors="ignore")   # the batch record is the authority for it
    B = load_batches()[_BATCH_COLS].drop(columns=["role", "criterion_version"])
    return R.merge(B, on="batch_id", how="left", validate="many_to_one")


def legacy_only(df: pd.DataFrame) -> pd.DataFrame:
    """The Letter's layer alone — what the legacy equivalence gates are pinned on."""
    return df[df.study_id == LEGACY_STUDY_ID].reset_index(drop=True)


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
    criterion (or under none the ledger can vouch for) is not returned. That is the rule —
    reuse keys on pool identity AND criterion — enforced by the signature.
    """
    if not criterion_version:
        raise ValueError("resolve_reviews needs a criterion_version; use legacy_catalogue() for the mixed legacy view")
    d = _resolved(pool_id)
    return d[d.criterion_version == criterion_version].reset_index(drop=True)


def legacy_catalogue(pool_id: str) -> pd.DataFrame:
    """The Letter's catalogue as it was: every criterion mixed, first verdict wins. GATE ONLY."""
    return _resolved(pool_id)


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


def legacy_random_sessions(pool_id: str) -> pd.DataFrame:
    """The three sessions the Letter called its honest evaluation set. GATE ONLY: one of them
    (representative_batch_1) predates the anchor and carries no criterion the ledger can vouch for."""
    R = load_reviews()
    d = R[(R.pool_id == pool_id) & R.legacy_random_session_index.notna()
          & R.decision.isin([0, 1]) & R.candidate_id.notna()]
    return d.reset_index(drop=True)


def training_sample(pool_id: str) -> pd.DataFrame:
    """The legacy training walk: the three random sessions in their legacy order, then the five
    priority batches in theirs, first review per candidate, then the blind adjudication's
    corrections applied. Reproduces `obduction.labels.consistent_labels()`; the order is data
    (`legacy.random_session_index`, `legacy.priority_index`, `legacy.is_adjudication`)."""
    B = batches_of(pool_id)
    R = load_reviews()
    random_ = B[B.legacy_random_session_index.notna()].sort_values("legacy_random_session_index")
    priority = B[B.legacy_priority_index.notna()].sort_values("legacy_priority_index")
    if (priority.role != "training").any():
        raise ValueError(f"{pool_id}: a priority batch without role 'training': "
                         f"{priority[priority.role != 'training'].batch_id.tolist()}")
    parts = [R[(R.batch_id == n) & R.decision.isin([0, 1]) & R.candidate_id.notna()]
             for n in random_.batch_id.tolist() + priority.batch_id.tolist()]
    L = pd.concat(parts, ignore_index=True).drop_duplicates("candidate_id")
    adj_batches = B[B.legacy_is_adjudication.fillna(False).astype(bool)].batch_id
    adj = R[R.batch_id.isin(adj_batches) & R.decision.isin([0, 1])]
    corr = dict(zip(adj.candidate_id, adj.decision))
    L = L.assign(decision=[corr.get(c, y) for c, y in zip(L.candidate_id, L.decision)])
    return L.reset_index(drop=True)


@lru_cache(maxsize=1)
def _batches_with_label_column() -> frozenset:
    B = load_batches()
    return frozenset(b for b, cols in zip(B.batch_id, B.columns_kept) if "LABEL" in cols)


def labelled_keys(exclude_batch_prefix: str = "nitrate/",
                  exclude_roles: tuple[str, ...] = ("answer_key",), include_study: bool = False) -> set:
    """Every event key decided in ANY legacy batch — the exclusion set for new batches.

    Reproduces the legacy `all_labeled_keys()` exactly, including its two quirks, so that the
    equivalence gate can hold: the `nitrate/` directory is left out (a different candidate table,
    a different criterion) but the other nitrate and carbon sheets are IN; a sheet WITH a LABEL
    column contributes its decided rows (0/1/2), a sheet WITHOUT one contributes every row. The
    keys are the normalised (WMO, CYCLE_NUMBER, round(PRES)) triples, as floats, as legacy did.
    `include_study=True` adds the study layer's decided keys (a probability draw never EXCLUDES
    them from its frame — they are a covariate, `previously_judged`; this is for reporting).
    """
    R = load_reviews()
    if not include_study:
        R = legacy_only(R)
    has_label = R.batch_id.isin(_batches_with_label_column())
    keep = (~R.batch_id.str.startswith(exclude_batch_prefix)) & ~R.role.isin(exclude_roles)
    keep &= (has_label & R.decision.isin([0, 1, 2])) | ~has_label
    d = R[keep & R.key_wmo.notna() & R.key_cycle.notna() & R.key_pres.notna()]
    return set(zip(d.key_wmo.astype(float), d.key_cycle.astype(float), d.key_pres.astype(float)))


def verdicts_view() -> pd.DataFrame:
    """The legacy `verdicts.csv` rebuilt from the ledger, column for column, row for row."""
    R = pd.read_parquet(REVIEWS)
    order = ["event", "WMO", "CYCLE_NUMBER", "PRES_ADJUSTED", "LABEL", "batch_id", "rank", "tier",
             "sampling_mode", "stratum", "control_arm", "blind", "SAMPLE_ID", "src", "score"]
    return R[order].rename(columns={"batch_id": "batch"})
