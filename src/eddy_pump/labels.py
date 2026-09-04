"""The label table's query API — the only door a rate may use.

reads  data/labels/{study_batches.yaml, study_reviews.parquet} (drawn by pipeline/draw_batch.py,
       loaded once labelled by pipeline/load_batch.py)
writes nothing
`analysis_sample(pool, criterion)`: the probability sample a rate may use (role analysis, a
probability design, a batch that decides, the target arm). `labelled_keys()`: the keys already
labelled, reported beside a new batch as a covariate.

Two copies of `role` and `criterion_version` exist: the review row carries what the sheet was
written with, the batch record carries what the batch was drawn under. `load_reviews` keeps both,
under the suffixes `_review` and `_batch`, and raises if they disagree — the batch record is the
authority, and a silent difference between the two would let a rate filter on one while a check
read the other.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from .manifest import REPO_ROOT

LABELS = REPO_ROOT / "data" / "labels"
# The study's one label layer: batches drawn by pipeline/draw_batch.py and loaded, once labelled,
# by pipeline/load_batch.py. Every review carries the inclusion probability of its draw.
STUDY_BATCHES = LABELS / "study_batches.yaml"
STUDY_REVIEWS = LABELS / "study_reviews.parquet"

ROLES = ("analysis", "training", "catalogue", "audit", "calibration", "control", "snapshot", "answer_key")
PROBABILITY_DESIGNS = ("probability",)
_BATCH_COLS = ["batch_id", "role", "decides", "criterion_version", "sampling_design", "sampling_frame"]
#: The columns both the review row and the batch record carry. Merged under a suffix, checked equal.
_TWO_COPIES = ("role", "criterion_version")


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
                                f"load it once labelled (pipeline/load_batch.py) first")
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
    """Every study review, in sheet order, with its batch's role, design, frame and precedence.

    `role` and `criterion_version` arrive twice — once from the sheet's own rows, once from the
    batch record — and are kept as `role_review` / `role_batch` and `criterion_version_review` /
    `criterion_version_batch`. They must agree row for row; a disagreement raises here rather than
    letting one filter read the sheet and another read the record.
    """
    if not STUDY_REVIEWS.exists():
        raise FileNotFoundError(f"missing {STUDY_REVIEWS} — load a labelled batch first "
                                f"(pipeline/load_batch.py)")
    R = pd.read_parquet(STUDY_REVIEWS)
    R = R.drop(columns=["study_id"], errors="ignore")   # the batch record is the authority for it
    B = load_batches()[_BATCH_COLS]
    M = R.merge(B, on="batch_id", how="left", validate="many_to_one", suffixes=("_review", "_batch"))
    for col in _TWO_COPIES:
        a, b = M[f"{col}_review"], M[f"{col}_batch"]
        both = a.notna() & b.notna()
        off = both & (a.astype("string") != b.astype("string"))
        if off.any():
            bad = M.loc[off, ["batch_id", f"{col}_review", f"{col}_batch"]].drop_duplicates()
            raise ValueError(f"{col}: the sheet's rows and the batch record disagree for "
                             f"{int(off.sum())} reviews; the batch record is the authority and the "
                             f"two must be written together:\n{bad.head()}")
    return M


def batches_of(pool_id: str) -> pd.DataFrame:
    B = load_batches()
    return B[B.pool_id == pool_id]


def analysis_sample(pool_id: str, criterion_version: str) -> pd.DataFrame:
    """The reviews a rate may be computed from.

    Role `analysis` (a uniform draw), a probability design, a batch that decides, judged under
    `criterion_version`, decided 0/1, and on the target arm only — a blind batch's constructed
    positives and negatives are controls, not draws. Raises if the pool has no such batch, or if
    any analysis batch is not a probability draw. A score-selected, catalogue, calibration,
    control, audit or snapshot batch never enters a rate.
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
    # `decides` is the batch record's precedence flag: a batch that does not decide is evidence,
    # never a number. The rate filters on it as well as on the role.
    A = A[A.decides.fillna(False).astype(bool)]
    if A.empty:
        raise ValueError(f"{pool_id}: every analysis batch under {criterion_version!r} is marked as "
                         f"not deciding (`decides: false`) — none of them may feed a rate")
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


def labelled_keys(exclude_roles: tuple[str, ...] = ("answer_key",)) -> set:
    """Every event key the study has labelled — reported beside a new batch.

    A sheet with a LABEL column contributes its decided rows (0/1/2); a sheet without one
    contributes every row. The keys are the normalised (WMO, CYCLE_NUMBER, round(PRES)) triples,
    as floats. A probability draw never excludes them from its frame — they are a covariate,
    `previously_judged`; this is for reporting.
    """
    R = load_reviews()
    has_label = R.batch_id.isin(_batches_with_label_column())
    role = R.role_batch.fillna(R.role_review)
    keep = ~role.isin(exclude_roles)
    keep &= (has_label & R.decision.isin([0, 1, 2])) | ~has_label
    d = R[keep & R.key_wmo.notna() & R.key_cycle.notna() & R.key_pres.notna()]
    return set(zip(d.key_wmo.astype(float), d.key_cycle.astype(float), d.key_pres.astype(float)))
