"""The label table's query API: what a rate is allowed to read, and what it must refuse.

`eddy_pump.labels` is the only door a rate may use. Two things it must get right, both proved here
on a small table built in the test, and then checked once against the real one:

1. `role` and `criterion_version` exist twice — on the sheet's own rows and on the batch record.
   They must agree. `load_reviews` keeps both under `_review` / `_batch` and raises otherwise, so a
   filter reading one and a check reading the other cannot disagree in silence.
2. `analysis_sample` returns only what a rate may divide by: role analysis, a probability design, a
   batch that decides, the criterion asked for, decided 0/1, the target arm. A control row, a
   calibration set, and a batch marked as not deciding are each refused.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from eddy_pump import labels as L

REPO = Path(__file__).resolve().parents[1]
POOL = "net_carbon_v1/physical/obduction"
CRIT = "phys_net_carbon_v1"


# --------------------------------------------------------------------------------------------- #
# a small table, written in the test
# --------------------------------------------------------------------------------------------- #
def _batch(batch_id, role="analysis", decides=True, criterion=CRIT, design="probability", pool=POOL):
    return {"batch_id": batch_id, "pool_id": pool, "role": role, "decides": decides,
            "criterion_version": criterion, "columns_kept": ["LABEL", "SAMPLE_ID"],
            "sampling": {"design": design, "frame": "a test frame"}}


def _reviews(rows):
    return pd.DataFrame(rows)


def _row(batch_id, cand, decision=1, role="analysis", criterion=CRIT, arm="target", pool=POOL, wmo=1, pres=300):
    return {"batch_id": batch_id, "candidate_id": cand, "decision": decision, "role": role,
            "criterion_version": criterion, "control_arm": arm, "pool_id": pool,
            "key_wmo": wmo, "key_cycle": 10, "key_pres": pres, "inclusion_probability": 0.01,
            "design_stratum": "open|d9", "review_id": cand}


@pytest.fixture
def table(tmp_path, monkeypatch):
    """Point `eddy_pump.labels` at a table this test writes, and clear its caches."""
    def install(batches, reviews):
        bp, rp = tmp_path / "study_batches.yaml", tmp_path / "study_reviews.parquet"
        bp.write_text(yaml.safe_dump({"batches": batches}), encoding="utf-8")
        _reviews(reviews).to_parquet(rp, index=False)
        monkeypatch.setattr(L, "STUDY_BATCHES", bp)
        monkeypatch.setattr(L, "STUDY_REVIEWS", rp)
        for f in (L.load_batches, L.load_reviews, L._batches_with_label_column):
            f.cache_clear()
    yield install
    for f in (L.load_batches, L.load_reviews, L._batches_with_label_column):
        f.cache_clear()


def test_the_two_copies_of_role_and_criterion_are_both_kept_and_must_agree(table):
    table([_batch("b1")], [_row("b1", "c1"), _row("b1", "c2", decision=0)])
    R = L.load_reviews()
    assert {"role_review", "role_batch", "criterion_version_review", "criterion_version_batch"} <= set(R.columns)
    assert (R.role_review == R.role_batch).all()
    assert (R.criterion_version_review == R.criterion_version_batch).all()


def test_a_sheet_row_that_disagrees_with_its_batch_record_raises(table):
    table([_batch("b1", role="analysis")],
          [_row("b1", "c1"), _row("b1", "c2", role="calibration")])
    with pytest.raises(ValueError, match="role: the sheet's rows and the batch record disagree"):
        L.load_reviews()


def test_a_criterion_that_disagrees_with_its_batch_record_raises(table):
    table([_batch("b1")], [_row("b1", "c1"), _row("b1", "c2", criterion="something_else")])
    with pytest.raises(ValueError, match="criterion_version: the sheet's rows and the batch record disagree"):
        L.load_reviews()


def test_a_rate_reads_only_decided_target_rows_of_a_probability_batch(table):
    table([_batch("b1")],
          [_row("b1", "c1"), _row("b1", "c2", decision=0), _row("b1", "c3", decision=2),
           _row("b1", "c4", arm="pos_ctrl"), _row("b1", "c5", arm="neg_ctrl")])
    A = L.analysis_sample(POOL, CRIT)
    assert sorted(A.candidate_id) == ["c1", "c2"], "uncertain and both control arms are out"


def test_a_batch_that_does_not_decide_never_feeds_a_rate(table):
    table([_batch("b1", decides=False)], [_row("b1", "c1"), _row("b1", "c2", decision=0)])
    with pytest.raises(ValueError, match="not deciding"):
        L.analysis_sample(POOL, CRIT)


def test_a_batch_that_decides_is_read_beside_one_that_does_not(table):
    table([_batch("b1"), _batch("b2", decides=False)],
          [_row("b1", "c1"), _row("b2", "c9")])
    A = L.analysis_sample(POOL, CRIT)
    assert sorted(A.candidate_id) == ["c1"]


def test_a_calibration_batch_and_a_non_probability_design_are_refused(table):
    table([_batch("cal", role="calibration")], [_row("cal", "c1", role="calibration")])
    with pytest.raises(ValueError, match="no batch with role 'analysis'"):
        L.analysis_sample(POOL, CRIT)
    table([_batch("b1", design="score_selected")], [_row("b1", "c1")])
    with pytest.raises(ValueError, match="without a probability design"):
        L.analysis_sample(POOL, CRIT)


def test_a_rate_refuses_a_criterion_no_batch_was_judged_under(table):
    table([_batch("b1")], [_row("b1", "c1")])
    with pytest.raises(ValueError, match="no analysis batch judged under"):
        L.analysis_sample(POOL, "some_other_criterion")


def test_one_candidate_in_two_analysis_batches_is_refused(table):
    table([_batch("b1"), _batch("b2")], [_row("b1", "c1"), _row("b2", "c1")])
    with pytest.raises(ValueError, match="more than one analysis batch"):
        L.analysis_sample(POOL, CRIT)


def test_labelled_keys_counts_uncertain_and_skips_an_answer_key(table):
    table([_batch("b1"), _batch("ak", role="answer_key")],
          [_row("b1", "c1", wmo=1), _row("b1", "c2", decision=2, wmo=2),
           _row("b1", "c3", decision=None, wmo=3), _row("ak", "c4", role="answer_key", wmo=4)])
    keys = L.labelled_keys()
    assert {w for w, _, _ in keys} == {1.0, 2.0}, "uncertain counts as judged; blank and answer keys do not"


# --------------------------------------------------------------------------------------------- #
# the real table
# --------------------------------------------------------------------------------------------- #
real = pytest.mark.skipif(not L.STUDY_REVIEWS.exists(),
                          reason="the study label table is not on this machine")


@real
def test_the_real_table_agrees_with_itself_and_its_rate_sample_is_clean():
    R = L.load_reviews()          # raises if the two copies disagree anywhere
    assert len(R) and (R.role_review.notna() | R.role_batch.notna()).all()
    A = L.analysis_sample(POOL, CRIT)
    assert A.decision.isin([0, 1]).all()
    assert (A.control_arm.isna() | (A.control_arm == "target")).all()
    assert A.inclusion_probability.between(0, 1, inclusive="right").all()
    assert not A.candidate_id.duplicated().any()
    B = L.load_batches()
    used = B[B.batch_id.isin(A.batch_id)]
    assert (used.role == "analysis").all() and used.decides.all()
