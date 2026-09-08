"""The session rules and the drift correction — plan step 7 *(2026-09-07)*.

The review of 2026-09-04 found the reviewer's reading moving inside each of the two first rate
sessions, by two to three times the sampling error, and the net of the two limbs changing sign
across that band. Nothing in the draw or the loader guarded against it. These tests hold the guards
that were added.

Seven things, the last three added on 2026-09-08 after the loader let a second look in over a
calibration copy that had read stricter than the reference:

1. **The controls are spread by the draw.** Every arm sits at evenly spaced positions, so a fresh
   draw passes the loader's position check at every seed; a sheet whose controls were dropped where
   a shuffle put them — 17 of 20 in the second half, which is what the first upward sheet did —
   fails it.
2. **A sheet is one sitting.** At most 120 panels; a longer draw is cut into sheets of one draw,
   with the same rows in the same order, so the rate cannot move.
3. **The blind second look.** 100 science rows, 50 from each half of the source sheet, 20 controls,
   a blind worksheet; the flip table reads what the second look did to the first.
4. **The corrected rate.** On a synthetic sheet whose second half was read at half the acceptance,
   the correction puts the rate back where the first half had it, and the refusal to load fires when
   the drift is there and nothing has measured it.
5. **The calibration copy labelled before the sheet is the one that decides.** Not an earlier copy
   that passed. If it read DRIFTED the sheet is refused, naming the copy and its numbers; loaded by
   hand, the record says so, and a second look loaded that way corrects no rate.
6. **Two second looks of one sheet are read together.** Their cell counts add and the error bar
   narrows by about √2; one whose calibration copy did not pass is left out of the correction and
   keeps its rows.
7. **The second look shaped by the first verdict.** The four cells of the correction are strata of
   the draw, shared out so the correction's error is as small as it can be at 100 panels.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from eddy_pump import batches as B

REPO = Path(__file__).resolve().parents[1]
DRAWS = REPO / "data/labels/draws"


def _loader():
    """`pipeline/load_batch.py` as a module (it is a script, not a package)."""
    sys.path.insert(0, str(REPO / "pipeline"))
    sys.path.insert(0, str(REPO / "src"))
    import load_batch

    return load_batch


def _draw_script():
    sys.path.insert(0, str(REPO / "pipeline"))
    sys.path.insert(0, str(REPO / "src"))
    import draw_batch

    return draw_batch


def _sheet(n_science: int, n_pos: int, n_neg: int, rng, interleave: bool = True):
    """A worksheet's stratum column in review order, from the real assembling path."""
    sci = pd.DataFrame({"WMO": np.arange(n_science) + 1_000_000, "CYCLE_NUMBER": 3.0,
                        "PRES_ADJUSTED": np.arange(n_science) * 1.0 + 200,
                        "design_stratum": "open|d5", "score": 0.5, "inclusion_probability": 0.01,
                        "candidate_id": [f"s{i}" for i in range(n_science)],
                        "LATITUDE": 0.0, "LONGITUDE": 0.0, "TIME": "2020-01-01"})
    n_c = n_pos + n_neg
    ctrl = pd.DataFrame({"WMO": np.arange(n_c) + 9_000_000, "CYCLE_NUMBER": 4.0,
                         "PRES_ADJUSTED": np.arange(n_c) * 1.0 + 200,
                         "stratum": [B.POS_CTRL] * n_pos + [B.NEG_CTRL] * n_neg, "src": "test",
                         "REF_LABEL": [1] * n_pos + [0] * n_neg, "score": 0.5,
                         "candidate_id": [f"c{i}" for i in range(n_c)],
                         "LATITUDE": 0.0, "LONGITUDE": 0.0, "TIME": "2020-01-01"})
    batch = B.Batch("t", sci, ctrl, "physical_obduction", rng, interleave=interleave)
    ws, key = batch.assemble()
    return ws, key, batch


# --------------------------------------------------------------------------------------------- #
# 1. the controls are spread by the draw, not by luck
# --------------------------------------------------------------------------------------------- #
def test_a_fresh_draw_spreads_every_control_arm_over_the_whole_sheet():
    """Each arm sits on its own evenly spaced ladder, so the loader's position check passes at
    every seed — not at 99 % of them, which is what a genuinely random placement would give."""
    for seed in range(60):
        rng = np.random.default_rng(seed)
        ws, key, batch = _sheet(579, 20, 20, rng)
        c = key[key.stratum.isin(B.CONTROL_STRATA)]
        rep = B.control_position_check(c.SAMPLE_ID.to_numpy(), c.stratum.to_numpy(), len(key))
        assert rep["verdict"] == "PASS", (seed, rep)
        assert rep["smallest_p"] > 0.5, (seed, rep["smallest_p"])
        for arm in rep["arms"].values():   # no quarter of the sheet is empty of an arm
            assert min(arm["quarters_of_the_sheet"]) >= 4
        assert batch.control_placement["stride_rows"] == pytest.approx(619 / 40)
        assert "offset" not in str(batch.control_placement).lower() or "not recorded" in str(batch.control_placement)


def test_a_plain_shuffle_can_bunch_an_arm_and_the_loader_says_so():
    """The first upward sheet put 17 of its 20 negative controls in the second half. That is what
    the check has to catch — and it does, at the p it actually scores."""
    sid = np.r_[np.array([40, 120, 200]), np.linspace(320, 615, 17).astype(int)]   # 3 early, 17 late
    rep = B.control_position_check(sid, np.array([B.NEG_CTRL] * 20), 619)
    assert rep["verdict"] == "FAIL" and rep["smallest_p"] < B.CONTROL_POSITION_ALPHA
    # A plain shuffle scatters the arms wherever it likes: over sixty seeds its worst sheet is far
    # from flat, while the interleaved draw's worst sheet is barely distinguishable from flat.
    def worst(interleave):
        out = 1.0
        for seed in range(60):
            ws, key, _ = _sheet(579, 20, 20, np.random.default_rng(seed), interleave=interleave)
            c = key[key.stratum.isin(B.CONTROL_STRATA)]
            out = min(out, B.control_position_check(c.SAMPLE_ID.to_numpy(), c.stratum.to_numpy(), len(key))["smallest_p"])
        return out
    assert worst(False) < 0.2 < 0.5 < worst(True), (worst(False), worst(True))


def test_the_real_upward_sheet_is_the_one_that_fails_and_the_other_two_pass():
    """The three rate sheets on disk, read through the loader's own checks: the first upward sheet
    fails on its negative controls (the 2026-09-04 finding), the other two pass."""
    LB = _loader()
    want = {"rate_obduction_01": "FAIL", "rate_subduction_01": "PASS", "rate_obduction_02": "PASS"}
    for bid, verdict in want.items():
        rec_path = DRAWS / f"{bid}.yaml"
        if not rec_path.exists():
            pytest.skip(f"{bid} is not on this machine")
        rec = yaml.safe_load(rec_path.read_text())
        if not B.resolve_recorded_path(rec["worksheet"]["path"]).exists():
            pytest.skip("the labelled sheets are not on this machine")
        ws, key, parts = LB.read_sheets(rec)
        sess = LB.session_block(rec, parts, ws, key)
        assert sess["control_positions"]["verdict"] == verdict, bid


# --------------------------------------------------------------------------------------------- #
# 2. a sheet is one sitting
# --------------------------------------------------------------------------------------------- #
def test_the_session_cap_cuts_a_long_draw_into_equal_sittings():
    D = _draw_script()
    assert D.MAX_SHEET_ROWS == 120
    assert D.session_sizes(120) == [120] and D.session_sizes(1) == [1]
    for n in (121, 240, 241, 619, 832, 1000):
        sizes = D.session_sizes(n)
        assert sum(sizes) == n and max(sizes) <= 120
        assert max(sizes) - min(sizes) <= 1, "the sittings are as equal as they can be"
        assert len(sizes) == -(-n // 120), "and there are as few of them as they can be"


def test_a_split_draw_is_the_same_rows_in_the_same_order(tmp_path):
    """The cut changes the sitting, never the sample: the sheets joined back are the sheet a single
    sitting would have been, row for row, and every control arm is still spread through each."""
    D = _draw_script()
    rng = np.random.default_rng(4)
    ws, key, _ = _sheet(579, 20, 20, rng)
    whole = D.write_session_sheets(tmp_path / "whole", ws, key, "b", allow_long=True)
    assert whole["sessions"] is None and whole["worksheet"]["rows"] == 619
    split = D.write_session_sheets(tmp_path / "split", ws, key, "b", allow_long=False)
    assert len(split["sessions"]) == 6 and split["worksheet"]["rows"] == 619
    back = pd.concat([pd.read_csv(B.resolve_recorded_path(x["worksheet"]["path"]))
                      for x in split["sessions"]], ignore_index=True)
    assert back.equals(pd.read_csv(B.resolve_recorded_path(whole["worksheet"]["path"])))
    keys = pd.concat([pd.read_csv(B.resolve_recorded_path(x["answer_key"]["path"]))
                      for x in split["sessions"]], ignore_index=True)
    assert (keys.inclusion_probability.fillna(-1).to_numpy() == key.inclusion_probability.fillna(-1).to_numpy()).all()
    for x in split["sessions"]:
        k = pd.read_csv(B.resolve_recorded_path(x["answer_key"]["path"]))
        assert len(k) <= 120 and k.stratum.isin(B.CONTROL_STRATA).sum() >= 5


def test_the_loader_reads_the_sittings_of_one_draw_back_as_one_sheet(tmp_path):
    D, LB = _draw_script(), _loader()
    rng = np.random.default_rng(5)
    ws, key, _ = _sheet(200, 10, 10, rng)
    ws["LABEL"] = np.where(ws.SAMPLE_ID < 110, 1, 0)
    split = D.write_session_sheets(tmp_path / "split", ws, key, "b", allow_long=False)
    rec = {"batch_id": "b", **split}
    got_ws, got_key, parts = LB.read_sheets(rec)
    assert len(parts) == 2 and len(got_ws) == 220 and len(got_key) == 220
    assert (got_ws.SAMPLE_ID.to_numpy() == np.arange(220)).all()
    assert LB.joint_sha([{"sheet_id": x["sheet_id"], "sha256": x["sha256"]} for x in parts]) != parts[0]["sha256"]
    assert LB.joint_sha([{"sheet_id": "one", "sha256": "abc"}]) == "abc", "one sheet keeps its own hash"
    # the halves are cut inside each sitting, never across the join
    m = got_key.merge(got_ws[["SAMPLE_ID", "LABEL"]], on="SAMPLE_ID")
    m["LABEL"] = pd.to_numeric(m.LABEL, errors="coerce")
    m = LB.add_position_columns(m)
    assert set(m.groupby("_sheet")._half_cut_at.nunique()) == {1}
    assert m._half_cut_at.nunique() == 2, "each sitting has its own middle"
    # and the control check reads positions inside the sitting, not the draw's own numbering
    sess = LB.session_block({"batch_id": "b", "role": "analysis", "controls": {"positive": {}, "negative": {}}},
                            parts, got_ws, got_key)
    assert set(sess["control_positions"]["per_sheet"]) == {"b_s1", "b_s2"}
    assert sess["control_positions"]["verdict"] == "PASS"
    for rep in sess["control_positions"]["per_sheet"].values():
        assert rep["verdict"] == "PASS" and rep["smallest_p"] > 0.2, rep
        for arm in rep["arms"].values():
            assert sum(arm["quarters_of_the_sheet"]) == arm["n"], "a position fell outside its own sheet"


# --------------------------------------------------------------------------------------------- #
# 3. the blind second look
# --------------------------------------------------------------------------------------------- #
def test_the_flip_table_reads_a_second_look_that_moved_in_the_second_half():
    """A made-up re-judgement: the first half agrees, the second half's accepts half come back as
    rejects. The table must say so, and the Fisher test must see it."""
    original = np.r_[np.ones(25, int), np.zeros(25, int), np.ones(25, int), np.zeros(25, int)]
    half = np.array(["first"] * 50 + ["second"] * 50)
    rejudged = original.copy()
    rejudged[75:] = 1                       # the second half's rejects all come back as accepts
    ft = B.flip_table(original, rejudged, half)
    assert ft["by_half"]["first"]["accept_to_reject"] == 0 and ft["by_half"]["first"]["reject_to_accept"] == 0
    assert ft["by_half"]["second"]["reject_to_accept"] == 25
    assert ft["cells"]["0|first"] == {"n": 25, "accepted": 0}
    assert ft["cells"]["0|second"] == {"n": 25, "accepted": 25}
    assert ft["fisher_p_first_vs_second_half_among_originally_rejected"] < 1e-9
    assert ft["fisher_p_first_vs_second_half_among_originally_accepted"] == 1.0
    assert ft["overall"]["n"] == 100 and ft["overall"]["accept_to_accept"] == 50


def test_the_rejudgement_design_takes_fifty_panels_from_each_half_and_stays_blind():
    """The design, run against the real labelled upward sheet: 100 science rows, half from each
    half, 20 controls, a worksheet that says nothing, and a saved answer sheet that says everything.
    """
    D = _draw_script()
    if not (DRAWS / "rate_obduction_01.labelled.yaml").exists():
        pytest.skip("rate_obduction_01 is not on this machine")
    t, meta = D.source_science_rows("rate_obduction_01")
    if not len(t):
        pytest.skip("the labelled upward sheet is not on this machine")
    assert meta["decided_science_rows"] == len(t) == 576
    assert meta["halves"]["first"]["n"] == meta["halves"]["second"]["n"] == 288
    assert set(t.original_half) == {"first", "second"} and t.original_label.isin([0, 1]).all()
    # the same halves the loader's session read cuts
    lab = yaml.safe_load((DRAWS / "rate_obduction_01.labelled.yaml").read_text())["session"]["target_halves"]
    for side in ("first", "second"):
        assert lab[side]["n"] == meta["halves"][side]["n"]
        assert abs(lab[side]["accept"] - meta["halves"][side]["originally_accepted"] / 288) < 1e-9


@pytest.mark.parametrize("bid", ["rejudge_obduction_01", "rejudge_subduction_01"])
def test_the_rejudgement_batches_plan_the_way_the_protocol_says(bid, tmp_path):
    """Both designs, built for real but written nowhere: 100 science rows split 50/50 by half, ten
    controls of each arm, 120 rows — exactly the session cap — and a blind worksheet."""
    D = _draw_script()
    src = D.DESIGNS[bid]["rejudges"]
    if not (DRAWS / f"{src}.labelled.yaml").exists():
        pytest.skip(f"{src} is not on this machine")
    plan = D.Plan(0.15, D.HELD_PANELS)
    rng = np.random.default_rng([20260827, D.DESIGNS[bid]["seed_index"]])
    batch, design = plan.build(bid, rng)
    ws, key = batch.assemble()
    assert design["role"] == "rejudgement" and design["decides"] is False and design["rejudges"] == src
    assert design["n_science"] == 100 and design["n_controls"] == {"positive": 10, "negative": 10}
    assert len(ws) == 120 <= D.MAX_SHEET_ROWS, "a re-judgement is one sitting"
    assert list(ws.columns) == B.WORKSHEET_COLS and ws.LABEL.isna().all()
    assert not (set(ws.columns) & B.BLIND_FORBIDDEN)
    sci = key[key.stratum == B.TARGET]
    assert len(sci) == 100 and sci.original_half.value_counts().to_dict() == {"first": 50, "second": 50}
    assert sci.original_label.isin([0, 1]).all() and sci.original_sample_id.is_unique
    assert sci.design_stratum.notna().all() and sci.score.notna().all()
    assert [h["n"] for h in design["halves"]] == [50, 50]
    for h in design["halves"]:
        assert abs(h["inclusion_probability"] - 50 / h["N"]) < 1e-12
    c = key[key.stratum.isin(B.CONTROL_STRATA)]
    rep = B.control_position_check(c.SAMPLE_ID.to_numpy(), c.stratum.to_numpy(), len(key))
    assert rep["verdict"] == "PASS", "the controls of a re-judgement are spread like any other sheet"


def test_the_loader_reads_a_labelled_rejudgement_end_to_end(tmp_path):
    """Draw `rejudge_obduction_01` for real, label it in the test with a made-up second look that
    tightened on the second half, and read it back through the loader: the flip table it writes is
    the one the correction then uses."""
    D, LB = _draw_script(), _loader()
    if not (DRAWS / "rate_obduction_01.labelled.yaml").exists():
        pytest.skip("rate_obduction_01 is not on this machine")
    plan = D.Plan(0.15, D.HELD_PANELS)
    batch, design = plan.build("rejudge_obduction_01", np.random.default_rng([20260827, 6]))
    ws, key = batch.assemble()
    # the made-up second look: it keeps every first-half verdict and rejects every second-half accept
    m = key[["SAMPLE_ID", "stratum", "original_label", "original_half"]].copy()
    lab = np.where(m.stratum == B.TARGET, m.original_label.fillna(0), (m.stratum == B.POS_CTRL).astype(float))
    lab = np.where((m.stratum == B.TARGET) & (m.original_half == "second") & (m.original_label == 1), 0, lab)
    ws["LABEL"] = lab.astype(int)
    sheets = D.write_session_sheets(tmp_path, ws, key, "rejudge_obduction_01", allow_long=False)
    rec = {**design, "batch_id": "rejudge_obduction_01", **sheets,
           "control_placement": batch.control_placement}
    got_ws, got_key, parts = LB.read_sheets(rec)
    sess = LB.session_block(rec, parts, got_ws, got_key)
    assert sess["kind"] == "rejudgement"
    rj = sess["rejudgement"]
    assert rj["rejudges"] == "rate_obduction_01" and rj["n_rejudged_and_decided"] == 100
    assert rj["by_half"]["first"]["accept_to_reject"] == 0
    assert rj["by_half"]["second"]["accept_to_reject"] == rj["cells"]["1|second"]["n"] > 0
    assert rj["cells"]["1|second"]["accepted"] == 0
    assert rj["cells"]["1|first"]["accepted"] == rj["cells"]["1|first"]["n"]
    assert rj["fisher_p_first_vs_second_half_among_originally_accepted"] < 0.01
    assert sum(c["n"] for c in rj["cells"].values()) == 100
    assert sess["control_positions"]["verdict"] == "PASS"


# --------------------------------------------------------------------------------------------- #
# 4. the corrected rate, and the refusal
# --------------------------------------------------------------------------------------------- #
def _drifted_sheet(rng, n=400, p_first=0.30, p_second=0.15):
    """A sheet of two strata read at `p_first` in its first half and `p_second` in its second."""
    sid = np.arange(n)
    second = sid >= n // 2
    y = (rng.uniform(size=n) < np.where(second, p_second, p_first)).astype(int)
    stratum = np.where(sid % 2 == 0, "open|d8", "open|d9")   # stratum is not confounded with position
    return sid, second, y, stratum


def test_the_correction_puts_a_halved_second_half_back_where_the_first_half_read_it():
    """The reviewer read the second half of the sheet at half the acceptance of the first. A blind
    second look reads every panel the way the first half was read. The corrected rate must come back
    to the first half's reading, and its error bar must be wider than the sampling error alone."""
    rng = np.random.default_rng(3)
    sid, second, y, stratum = _drifted_sheet(rng)
    N_h = {"open|d8": 100_000, "open|d9": 100_000}
    raw = B.stratified_rate(y.astype(float), np.full(len(y), 0.002), stratum, sid, N_h, n_boot=200)
    # the second look: it accepts every panel the first half accepted and none it rejected, and in
    # the second half it accepts the ones the tired reading had let go
    cells = {"1|first": {"n": 30, "accepted": 30}, "0|first": {"n": 70, "accepted": 0},
             "1|second": {"n": 30, "accepted": 30}, "0|second": {"n": 70, "accepted": 12}}
    cell = np.array([f"{v}|{'second' if s else 'first'}" for v, s in zip(y, second)])
    corr = B.drift_corrected_rate(y.astype(float), cell, stratum, N_h, cells, raw["se_design"], n_boot=3000, seed=1)
    first_half = float(y[~second].mean())
    assert corr["rate"] > raw["rate"], "correcting a reading that tightened must raise the rate"
    assert abs(corr["rate"] - first_half) < 0.04, (corr["rate"], first_half)
    assert corr["se_total"] > corr["se_design"] and corr["se_rejudgement"] > 0
    assert abs(corr["se_total"] ** 2 - corr["se_design"] ** 2 - corr["se_rejudgement"] ** 2) < 1e-12
    assert corr["rows_corrected"] == len(y) and corr["rows_kept_as_labelled"] == 0
    assert corr["half_width_95"] == pytest.approx(B.Z * corr["se_total"])


def test_a_second_look_that_agrees_with_the_first_leaves_the_rate_where_it_was():
    rng = np.random.default_rng(9)
    sid, second, y, stratum = _drifted_sheet(rng, p_first=0.2, p_second=0.2)
    N_h = {"open|d8": 100_000, "open|d9": 100_000}
    raw = B.stratified_rate(y.astype(float), np.full(len(y), 0.002), stratum, sid, N_h, n_boot=200)
    cells = {f"{v}|{h}": {"n": 100, "accepted": 100 * v} for v in (0, 1) for h in ("first", "second")}
    cell = np.array([f"{v}|{'second' if s else 'first'}" for v, s in zip(y, second)])
    corr = B.drift_corrected_rate(y.astype(float), cell, stratum, N_h, cells, raw["se_design"], n_boot=500, seed=1)
    assert abs(corr["rate"] - raw["rate"]) < 1e-12, "a second look that flips nothing changes no number"
    assert corr["se_rejudgement"] < 1e-12 and corr["se_total"] == pytest.approx(raw["se_design"])


def test_a_row_of_a_batch_nobody_looked_at_twice_keeps_its_own_verdict():
    rng = np.random.default_rng(11)
    sid, second, y, stratum = _drifted_sheet(rng, n=200)
    N_h = {"open|d8": 100_000, "open|d9": 100_000}
    cells = {"1|first": {"n": 20, "accepted": 20}, "0|first": {"n": 20, "accepted": 10}}
    cell = np.array([f"{v}|first" if i < 100 else "" for i, v in enumerate(y)])
    corr = B.drift_corrected_rate(y.astype(float), cell, stratum, N_h, cells, 0.01, n_boot=200, seed=1)
    assert corr["rows_corrected"] == 100 and corr["rows_kept_as_labelled"] == 100
    only_raw = B.drift_corrected_rate(y.astype(float), np.array([""] * len(y)), stratum, N_h, {}, 0.01, n_boot=10)
    assert abs(only_raw["rate"] - float(np.mean([y[stratum == h].mean() for h in ("open|d8", "open|d9")]))) < 1e-12
    assert only_raw["se_rejudgement"] == 0.0


def test_the_within_stratum_trend_sees_a_moved_reading_and_not_a_moved_stratum_mix():
    """Two sheets. In one the reading moved; in the other only the mix of strata did, and the
    acceptance dropped with it. The test must fire on the first and not on the second."""
    rng = np.random.default_rng(2)
    sid, second, y, stratum = _drifted_sheet(rng, n=600, p_first=0.30, p_second=0.12)
    moved = B.cochran_mantel_haenszel(y.astype(float), second, stratum)
    assert moved["p"] < B.DRIFT_ALPHA and moved["strata_used"] == 2
    # the same total drop, but every panel of the second half comes from the low-acceptance stratum
    n = 600
    second2 = np.arange(n) >= n // 2
    stratum2 = np.where(second2, "open|d1", "open|d9")
    y2 = (rng.uniform(size=n) < np.where(second2, 0.12, 0.30)).astype(int)
    mix = B.cochran_mantel_haenszel(y2.astype(float), second2, stratum2)
    assert mix["p"] is None or mix["p"] > 0.05, "a stratum that sits in one half carries no information"
    assert mix["chi2"] is None or mix["strata_used"] == 0
    assert float(y2[~second2].mean()) - float(y2[second2].mean()) > 0.1, "the raw drop is there all the same"


def test_the_loader_refuses_a_rate_batch_whose_reading_moved_and_says_what_to_do(monkeypatch, tmp_path):
    LB = _loader()
    monkeypatch.setattr(LB, "DRAWS", tmp_path)   # no re-judgement drawn yet: the real one exists since 2026-09-07
    rec = {"batch_id": "rate_obduction_01", "role": "analysis"}
    sess = {"within_stratum_position_trend": {"p": 0.004, "verdict": "DRIFTED",
                                              "accept_first_half": 0.24, "accept_second_half": 0.16}}
    with pytest.raises(SystemExit) as e:
        LB.refuse_if_the_reading_drifted(rec, dict(sess), False)
    msg = str(e.value)
    assert "acceptance fell across the sitting" in msg and "rejudge_obduction_01" in msg
    assert "--accept-drift" in msg
    # by hand, and the record carries it
    s2 = dict(sess)
    LB.refuse_if_the_reading_drifted(rec, s2, True)
    assert s2["drift_decision"]["accepted_by_hand"] is True and "--accept-drift" in s2["drift_decision"]["loaded_because"]
    # a flat sheet is loaded without a word
    s3 = {"within_stratum_position_trend": {"p": 0.4, "verdict": "FLAT"}}
    LB.refuse_if_the_reading_drifted(rec, s3, False)
    assert s3["drift_decision"]["accepted_by_hand"] is False and "did not fall" in s3["drift_decision"]["loaded_because"]
    # a calibration or re-judgement sheet is not a rate and is never refused on this
    LB.refuse_if_the_reading_drifted({"batch_id": "x", "role": "calibration"}, {}, False)


def test_a_drawn_rejudgement_is_what_lets_a_drifted_batch_load(monkeypatch, tmp_path):
    """The way out of the refusal is to draw the blind second look, not to argue with it."""
    LB = _loader()
    monkeypatch.setattr(LB, "DRAWS", tmp_path)
    (tmp_path / "rejudge_obduction_01.yaml").write_text(yaml.safe_dump(
        {"batch_id": "rejudge_obduction_01", "role": "rejudgement", "rejudges": "rate_obduction_01",
         "built": "2026-09-07T10:00:00-04:00"}))
    sess = {"within_stratum_position_trend": {"p": 0.004, "verdict": "DRIFTED",
                                              "accept_first_half": 0.24, "accept_second_half": 0.16}}
    LB.refuse_if_the_reading_drifted({"batch_id": "rate_obduction_01", "role": "analysis"}, sess, False)
    d = sess["drift_decision"]
    assert d["rejudgements"] == [{"batch_id": "rejudge_obduction_01", "drawn": "2026-09-07T10:00:00-04:00",
                                  "loaded": False}]
    assert "blind re-judgement of this batch exists" in d["loaded_because"]
    assert LB.rejudgements_of("rate_subduction_01") == []


def test_the_rate_report_leaves_the_corrected_columns_empty_until_a_second_look_is_loaded():
    p = REPO / "data/labels/audit/rate_status.csv"
    if not p.exists():
        pytest.skip("no rate report on this machine")
    T = pd.read_csv(p)
    for c in ("rate_drift_corrected", "rate_drift_corrected_half_width_95", "accepted_levels_drift_corrected",
              "net_accepted_levels_drift_corrected", "net_accepted_levels_half_width_95", "rejudgement_batches"):
        assert c in T.columns, c
    if not (DRAWS / "rejudge_obduction_01.labelled.yaml").exists():
        assert T.rate_drift_corrected.isna().all(), "no second look has been loaded; nothing may be corrected"
        page = (REPO / "data/labels/audit/RATE_STATUS.md").read_text()
        assert "## The net — not quoted" in page


# --------------------------------------------------------------------------------------------- #
# 5. the calibration copy labelled before the sheet is the one that decides
# --------------------------------------------------------------------------------------------- #
def _calibration_copy(d: Path, bid: str, pool_id: str, when: str, verdict: str, kappa: float = 0.85,
                      accepted: int = 17, n: int = 42, reference: int = 17, undecided: int = 0) -> None:
    """A calibration copy on disk as the loader reads it: its draw record and its loaded record."""
    (d / f"{bid}.yaml").write_text(yaml.safe_dump({"batch_id": bid, "pool_id": pool_id, "role": "calibration"}))
    decided = n - undecided
    (d / f"{bid}.labelled.yaml").write_text(yaml.safe_dump(
        {"batch_id": bid, "worksheet_mtime": when,
         "session": {"verdict": verdict, "kappa": kappa, "n_reference": n, "n_decided": decided,
                     "n_undecided": undecided, "base_rate_observed": accepted / decided,
                     "base_rate_reference": reference / n}}))


POOL = "net_carbon_v1/physical/subduction"


def test_the_most_recent_calibration_copy_decides_not_an_older_one_that_passed(monkeypatch, tmp_path):
    """The hole this closes: on 2026-09-08 the loader took a copy that had passed a week earlier and
    ignored the copy labelled that morning, which read DRIFTED. The copy labelled before a session is
    what says whether the session counts."""
    LB = _loader()
    monkeypatch.setattr(LB, "DRAWS", tmp_path)
    _calibration_copy(tmp_path, "calib_subduction_v1_pass2", POOL, "2026-09-01T14:23:56-04:00", "PASS",
                      kappa=0.86, accepted=20)
    _calibration_copy(tmp_path, "calib_subduction_v1_pass3", POOL, "2026-09-08T09:19:36-04:00", "DRIFTED",
                      kappa=0.73, accepted=13, undecided=2)
    rec = {"batch_id": "rejudge_subduction_01", "pool_id": POOL, "role": "rejudgement"}
    with pytest.raises(SystemExit) as e:
        LB.calibration_check_for(rec, "2026-09-08T09:22:29-04:00", False, False)
    msg = str(e.value)
    assert "calib_subduction_v1_pass3" in msg and "13/42" in msg and "17/42" in msg
    assert "DRIFTED" in msg and "20% relative stricter" in msg and "κ 0.73" in msg
    assert "--repass calib_subduction_v1" in msg and "--accept-calibration-drift" in msg
    assert "calib_subduction_v1_pass2" not in msg.split("Label a fresh")[0], "the older PASS is no defence"
    # the same two copies, and a sheet labelled BEFORE the drifted one: then the PASS is the check
    check = LB.calibration_check_for(rec, "2026-09-02T10:00:00-04:00", False, False)
    assert check["batch"] == "calib_subduction_v1_pass2" and check["verdict"] == "PASS"
    assert check["earlier_day"] is True and check["counts_for_correction"] is True
    assert check["accepted_by_hand"] is False


def test_a_drifted_calibration_copy_can_be_accepted_by_hand_and_the_record_carries_it(monkeypatch, tmp_path):
    LB = _loader()
    monkeypatch.setattr(LB, "DRAWS", tmp_path)
    _calibration_copy(tmp_path, "calib_subduction_v1_pass3", POOL, "2026-09-08T09:19:36-04:00", "DRIFTED",
                      kappa=0.73, accepted=13, undecided=2)
    rec = {"batch_id": "rejudge_subduction_01", "pool_id": POOL, "role": "rejudgement"}
    check = LB.calibration_check_for(rec, "2026-09-08T09:22:29-04:00", True, False)
    assert check["batch"] == "calib_subduction_v1_pass3" and check["verdict"] == "DRIFTED"
    assert check["base_rate_you"] == "13/42" and check["base_rate_reference"] == "17/42"
    assert abs(check["base_rate_drift_relative"] + 0.197) < 0.01 and check["undecided"] == 2
    assert check["accepted_by_hand"] is True and check["kappa"] == 0.73
    assert check["counts_for_correction"] is False and "did not pass" in check["does_not_count_because"]
    assert check["earlier_day"] is False and "the same day" in check["warning"]
    # and it may be made to count, but only by saying so as well
    both = LB.calibration_check_for(rec, "2026-09-08T09:22:29-04:00", True, True)
    assert both["counts_for_correction"] is True and "--count-for-correction" in both["counts_by_hand"]
    # a rate sheet has no such field: only a second look corrects a rate
    rate = LB.calibration_check_for({"batch_id": "rate_subduction_01", "pool_id": POOL, "role": "analysis"},
                                    "2026-09-08T09:22:29-04:00", True, False)
    assert "counts_for_correction" not in rate and rate["accepted_by_hand"] is True


def test_no_calibration_copy_at_all_is_refused(monkeypatch, tmp_path):
    LB = _loader()
    monkeypatch.setattr(LB, "DRAWS", tmp_path)
    rec = {"batch_id": "rate_subduction_01", "pool_id": POOL, "role": "analysis"}
    with pytest.raises(SystemExit) as e:
        LB.calibration_check_for(rec, "2026-09-08T09:22:29-04:00", False, False)
    assert "no scored copy" in str(e.value)
    check = LB.calibration_check_for(rec, "2026-09-08T09:22:29-04:00", True, False)
    assert check["batch"] is None and check["accepted_by_hand"] is True
    # a copy of the OTHER limb is not this limb's check
    _calibration_copy(tmp_path, "calib_obduction_v1_pass3", "net_carbon_v1/physical/obduction",
                      "2026-09-08T08:05:43-04:00", "PASS")
    with pytest.raises(SystemExit):
        LB.calibration_check_for(rec, "2026-09-08T09:22:29-04:00", False, False)
    assert [c["batch"] for c in LB.calibration_copies(POOL)] == []


def test_the_pass_that_became_the_saved_answers_is_not_a_check(monkeypatch, tmp_path):
    """That pass scores κ = 1 against itself, which says nothing about whether the reading held. The
    check is the next blind copy — which is what the upward held-region sheet was really read against
    (`calib_obduction_v1_pass2`, κ 0.95), not the freezing pass seven minutes later."""
    LB = _loader()
    monkeypatch.setattr(LB, "DRAWS", tmp_path)
    _calibration_copy(tmp_path, "calib_x_pass2", POOL, "2026-09-04T17:20:58-04:00", "PASS", kappa=0.95)
    _calibration_copy(tmp_path, "calib_x", POOL, "2026-09-04T17:28:03-04:00", "PASS", kappa=1.0)
    l = yaml.safe_load((tmp_path / "calib_x.labelled.yaml").read_text())
    l["session"]["reference_is_this_pass"] = True
    (tmp_path / "calib_x.labelled.yaml").write_text(yaml.safe_dump(l))
    assert [c["batch"] for c in LB.calibration_copies(POOL)] == ["calib_x_pass2"]
    check = LB.calibration_check_for({"batch_id": "rate_x", "pool_id": POOL, "role": "analysis"},
                                     "2026-09-04T17:52:04-04:00", False, False)
    assert check["batch"] == "calib_x_pass2" and check["kappa"] == 0.95
    # and the real records on disk read the same way
    if (DRAWS / "rate_obduction_02.labelled.yaml").exists():
        assert yaml.safe_load((DRAWS / "rate_obduction_02.labelled.yaml").read_text()
                              )["anchor"]["batch_id"] == "calib_obduction_v1_pass2"


def test_the_loaded_records_of_today_say_which_limb_may_correct_a_rate():
    """The two second looks on disk: the upward one counts, the downward one does not, and the rate
    report says why in the reviewer's own numbers."""
    for bid, counts in (("rejudge_obduction_01", True), ("rejudge_subduction_01", False)):
        p = DRAWS / f"{bid}.labelled.yaml"
        if not p.exists():
            pytest.skip(f"{bid} is not on this machine")
        cal = yaml.safe_load(p.read_text())["calibration_check"]
        assert cal["counts_for_correction"] is counts, bid
        assert cal["batch"].startswith("calib_") and cal["verdict"] in ("PASS", "DRIFTED")
    csv = REPO / "data/labels/audit/rate_status.csv"
    if not csv.exists():
        pytest.skip("no rate report on this machine")
    T = pd.read_csv(csv).set_index("pool_id")
    up, down = T.loc["net_carbon_v1/physical/obduction"], T.loc["net_carbon_v1/physical/subduction"]
    assert up.rejudgement_batches == "rejudge_obduction_01" and up.rate_drift_corrected > 0
    assert pd.isna(down.rate_drift_corrected) and down.rejudgements_not_counted == "rejudge_subduction_01"
    assert "13/42" in down.no_drift_correction_because and "17/42" in down.no_drift_correction_because
    page = (REPO / "data/labels/audit/RATE_STATUS.md").read_text()
    assert "## The net — not quoted" in page and "13/42" in page


# --------------------------------------------------------------------------------------------- #
# 6. two second looks of one sheet are read together
# --------------------------------------------------------------------------------------------- #
def test_two_second_looks_of_one_sheet_pool_their_cells_and_the_interval_shrinks():
    """A synthetic pair: the same four cells, re-judged twice at the same rates. Pooling doubles
    every count, so the correction lands on the same rate with a smaller error bar — about 1/√2 of
    it, because the cell rates are pinned by twice as many panels."""
    rng = np.random.default_rng(3)
    sid, second, y, stratum = _drifted_sheet(rng)
    N_h = {"open|d8": 100_000, "open|d9": 100_000}
    cell = np.array([f"{v}|{'second' if s else 'first'}" for v, s in zip(y, second)])
    one = {"1|first": {"n": 30, "accepted": 24}, "0|first": {"n": 40, "accepted": 4},
           "1|second": {"n": 15, "accepted": 12}, "0|second": {"n": 15, "accepted": 3}}
    two = {c: {"n": v["n"], "accepted": v["accepted"]} for c, v in one.items()}   # a second sheet, same rates
    alone = B.drift_corrected_rate(y.astype(float), cell, stratum, N_h, one, 0.013, n_boot=6000, seed=1)
    pooled = B.drift_corrected_rate(y.astype(float), cell, stratum, N_h,
                                    {c: [dict(one[c], batch_id="rj_01"), dict(two[c], batch_id="rj_02")] for c in one},
                                    0.013, n_boot=6000, seed=1)
    assert pooled["rate"] == pytest.approx(alone["rate"], abs=1e-12), "the same rates pool to the same rate"
    for c, v in pooled["cell_rates"].items():
        assert v["n"] == 2 * one[c]["n"] and v["accepted"] == 2 * one[c]["accepted"]
        assert [x["batch_id"] for x in v["pooled_from"]] == ["rj_01", "rj_02"]
    assert pooled["se_rejudgement"] < alone["se_rejudgement"]
    assert pooled["se_rejudgement"] == pytest.approx(alone["se_rejudgement"] / np.sqrt(2), rel=0.05)
    assert pooled["se_total"] < alone["se_total"] and pooled["half_width_95"] < alone["half_width_95"]
    # one block or a list of one are the same thing
    same = B.drift_corrected_rate(y.astype(float), cell, stratum, N_h, {c: [one[c]] for c in one}, 0.013,
                                  n_boot=6000, seed=1)
    assert same["rate"] == alone["rate"] and same["se_rejudgement"] == pytest.approx(alone["se_rejudgement"])
    assert B.pool_cells({"n": 5, "accepted": 2}) == {"n": 5, "accepted": 2}
    assert B.pool_cells([{"n": 5, "accepted": 2}, {"n": 0, "accepted": 0}]) == {"n": 5, "accepted": 2}


def test_the_rate_report_pools_two_loaded_second_looks_of_the_same_sheet(monkeypatch, tmp_path):
    """`loaded_rejudgements` hands the correction every second look of a sheet, oldest first, and
    leaves out the ones whose calibration copy did not pass."""
    sys.path.insert(0, str(REPO / "pipeline"))
    import rates as R

    monkeypatch.setattr(R, "DRAWS", tmp_path)
    for bid, when, counts in (("rj_a", "2026-09-08T09:00:00-04:00", True),
                              ("rj_b", "2026-09-09T09:00:00-04:00", True),
                              ("rj_c", "2026-09-10T09:00:00-04:00", False)):
        (tmp_path / f"{bid}.yaml").write_text(yaml.safe_dump(
            {"batch_id": bid, "role": "rejudgement", "rejudges": "rate_x"}))
        (tmp_path / f"{bid}.labelled.yaml").write_text(yaml.safe_dump(
            {"batch_id": bid, "worksheet_mtime": when,
             "calibration_check": {"batch": "calib_x", "verdict": "PASS" if counts else "DRIFTED",
                                   "kappa": 0.8, "base_rate_you": "13/42", "base_rate_reference": "17/42",
                                   "counts_for_correction": counts,
                                   "does_not_count_because": "the calibration copy did not pass"},
             "session": {"rejudgement": {"cells": {"1|first": {"n": 10, "accepted": 7}},
                                         "overall": {"n": 10, "accept_to_reject": 3, "reject_to_accept": 0}}}}))
    got, skipped = R.loaded_rejudgements(["rate_x"])
    assert [x["batch_id"] for x in got["rate_x"]] == ["rj_a", "rj_b"], "oldest first, and rj_c is left out"
    assert [x["batch_id"] for x in skipped] == ["rj_c"] and skipped[0]["base_rate_you"] == "13/42"
    assert R.loaded_rejudgements(["rate_y"]) == ({}, [])


# --------------------------------------------------------------------------------------------- #
# 7. the second look shaped by the first verdict
# --------------------------------------------------------------------------------------------- #
def test_the_allocation_by_cell_beats_the_uniform_one_and_respects_its_floor_and_its_rows():
    """The shape comes out of the correction's own variance: panels ∝ what the cell is worth × the
    spread of its flip rate. It must beat the uniform shape it replaces, keep the floor, and never
    ask for more rows than a cell has."""
    share = {"0|first": 0.396, "0|second": 0.416, "1|first": 0.110, "1|second": 0.077}
    q = {"0|first": 0.064, "0|second": 0.012, "1|first": 0.682, "1|second": 0.750}
    avail = {"0|first": 241, "0|second": 269, "1|first": 99, "1|second": 70}
    n = B.rejudgement_allocation(share, q, avail, 100)
    assert sum(n.values()) == 100 and all(v >= B.REJUDGE_CELL_FLOOR for v in n.values())
    assert all(n[c] <= avail[c] for c in n)
    uniform = {"0|first": 36.0, "0|second": 39.8, "1|first": 14.0, "1|second": 10.2}   # what a uniform draw gives
    assert B.rejudgement_correction_se(share, q, n) < B.rejudgement_correction_se(share, q, uniform)
    # it roughly doubles the panels the reviewer first called real — 19 of 100 under the uniform draw
    assert 30 <= n["1|first"] + n["1|second"] <= 45
    # but not further: the cells of panels first called NOT real carry four fifths of the weight, so
    # a shape with 60 of the 100 panels among the accepted ones is worse than the uniform draw
    heavy = {"1|first": 30, "1|second": 30, "0|first": 20, "0|second": 20}
    assert B.rejudgement_correction_se(share, q, heavy) > B.rejudgement_correction_se(share, q, uniform)
    # a cell with almost nothing left is capped there and the rest is shared out
    thin = B.rejudgement_allocation(share, q, {**avail, "0|first": 20}, 100)
    assert thin["0|first"] == 20 and sum(thin.values()) == 100
    with pytest.raises(ValueError):
        B.rejudgement_allocation(share, q, {c: 10 for c in share}, 100)
    with pytest.raises(ValueError):
        B.rejudgement_allocation(share, q, avail, 4 * B.REJUDGE_CELL_FLOOR - 1)


@pytest.mark.parametrize("bid", ["rejudge_subduction_02", "rejudge_obduction_02"])
def test_the_by_verdict_second_look_draws_its_four_cells_and_leaves_the_first_look_alone(bid, tmp_path):
    """Both redos, built for real and written nowhere: 100 science rows in four cells, none of them a
    panel the first second look already re-judged, every cell with its own inclusion probability."""
    D = _draw_script()
    src = D.DESIGNS[bid]["rejudges"]
    prior = D.DESIGNS[bid]["prior"]
    if not (DRAWS / f"{prior}.labelled.yaml").exists():
        pytest.skip(f"{prior} is not on this machine")
    plan = D.Plan(0.15, D.HELD_PANELS)
    rng = np.random.default_rng([20260827, D.DESIGNS[bid]["seed_index"]])
    batch, design = plan.build(bid, rng)
    ws, key = batch.assemble()
    assert design["role"] == "rejudgement" and design["decides"] is False and design["by_verdict"] is True
    assert design["n_science"] == 100 and len(ws) == 120 <= D.MAX_SHEET_ROWS
    assert list(ws.columns) == B.WORKSHEET_COLS and ws.LABEL.isna().all() and not (set(ws.columns) & B.BLIND_FORBIDDEN)
    sci = key[key.stratum == B.TARGET]
    assert len(sci) == 100 and sci.original_label.isin([0, 1]).all()
    cells = {c["cell"]: c for c in design["cells"]}
    assert set(cells) == {"0|first", "0|second", "1|first", "1|second"}
    drawn = (sci.original_label.astype(int).astype(str) + "|" + sci.original_half).value_counts().to_dict()
    for c, x in cells.items():
        assert drawn[c] == x["n"] >= B.REJUDGE_CELL_FLOOR
        assert x["inclusion_probability"] == pytest.approx(x["n"] / x["N"])
        assert sci.inclusion_probability[(sci.original_label.astype(int).astype(str) + "|" + sci.original_half) == c].eq(
            x["inclusion_probability"]).all()
    assert sum(x["n"] for x in cells.values()) == 100
    # the weights are the share of the pool the re-judged sheet's own strata carry: all of the
    # downward pool, the open region of the upward one. The cells of panels first called not real
    # carry most of it, which is why the shape does not go all the way to the accepted ones.
    weight = sum(x["weight_in_the_corrected_rate"] for x in cells.values())
    assert 0.9 < weight <= 1.0 + 1e-9
    assert sum(x["weight_in_the_corrected_rate"] for c, x in cells.items() if c.startswith("0")) > 0.7 * weight
    # the panels the first second look saw are left out
    first = pd.read_csv(B.resolve_recorded_path(
        yaml.safe_load((DRAWS / f"{prior}.yaml").read_text())["answer_key"]["path"]))
    seen = set(B.key3(first[first.stratum == B.TARGET]))
    assert not (set(B.key3(sci)) & seen), "a panel already seen twice is not a fresh second look"
    assert design["source"]["excluded_already_rejudged"]["rows"] == 100
    # and the shape is better than the uniform one it replaces
    e = design["correction_error"]
    assert e["se_this_shape"] < e["se_if_drawn_uniformly_by_half"]
    assert e["se_pooled_with"] == prior and e["se_pooled_with_that_sheet"] < e["se_this_shape"]
    assert 0.015 < e["se_this_shape"] < 0.025
    c = key[key.stratum.isin(B.CONTROL_STRATA)]
    rep = B.control_position_check(c.SAMPLE_ID.to_numpy(), c.stratum.to_numpy(), len(key))
    assert rep["verdict"] == "PASS", "the controls of a redo are spread like any other sheet"
