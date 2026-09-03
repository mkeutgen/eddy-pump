"""The review criterion is an object with an id, and the study names one *(plan step 1, 2026-08-26)*.

Hard rule 3 of the plan makes reuse key on criterion version. These tests prove that the three
criteria in `config/criteria.yaml` say what the sources say, that the active study names a
criterion covering both limbs, that a PROPOSED criterion cannot open a batch, and that the two
limbs of the active criterion differ in nothing but the AOU sign.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from eddy_pump.criteria import (
    CRITERIA_PATH,
    active_criterion,
    load_criteria,
    require_ruled,
    study_criterion_version,
)

REPO = Path(__file__).resolve().parents[1]
OLD_ANCHOR = REPO / "data/labels/external/calibration_reference_b6.csv"   # the 42 calibration panels, tracked since 2026-09-02


@pytest.fixture(scope="module")
def criteria():
    return load_criteria()


def test_three_criteria_exist_with_the_expected_status(criteria):
    assert {"phys_companion_2024", "phys_obduction_letter_b6", "phys_net_carbon_v1"} <= set(criteria)
    # the three historical ids the ledger keys legacy sheets on (plan step 2)
    assert {"phys_obduction_precalibration", "nitrate_obduction_letter", "carbon_obduction_letter"} <= set(criteria)
    assert all(criteria[c].status == "historical" for c in
               ("phys_obduction_precalibration", "nitrate_obduction_letter", "carbon_obduction_letter"))
    assert criteria["phys_companion_2024"].status == "historical"
    assert criteria["phys_obduction_letter_b6"].status == "historical"
    assert criteria["phys_net_carbon_v1"].status in ("proposed", "ruled")


def test_the_companion_criterion_is_three_clauses_on_one_limb_at_cycle_level(criteria):
    """The 2024 sentence: colocated peaks, below the mixed layer, under 200 m. Cycle unit, no anchor."""
    c = criteria["phys_companion_2024"]
    assert c.applies_to == ("subduction",)
    assert sorted(c.clauses) == [1, 2, 3]
    assert c.unit_of_judgement == "cycle"
    assert c.aou_sign("subduction") == "negative"
    assert c.raw["anchor"] == "none"


def test_the_obduction_criterion_is_the_protocols_four_clauses_verbatim(criteria):
    """Clause 4 is quoted from docs/LABELING_PROTOCOL.md, not paraphrased."""
    c = criteria["phys_obduction_letter_b6"]
    protocol = (REPO / "docs" / "LABELING_PROTOCOL.md").read_text(encoding="utf-8")
    assert "Stands out against an otherwise regular background" in protocol
    assert c.clauses[4].startswith("Stands out against an otherwise regular background")
    assert sorted(c.clauses) == [1, 2, 3, 4]
    assert c.unit_of_judgement == "level"
    assert c.aou_sign("obduction") == "positive"


def test_the_obduction_anchor_identity_is_recorded_and_matches_the_file_when_present(criteria):
    a = criteria["phys_obduction_letter_b6"].raw["anchor"]
    assert a["n_events"] == 42
    assert a["session"] == "representative/representative_batch_6"
    if not OLD_ANCHOR.exists():
        pytest.skip("data/labels/external/calibration_reference_b6.csv is missing")
    digest = hashlib.sha256(OLD_ANCHOR.read_bytes()).hexdigest()[:16]
    assert digest == a["sha256_16"], "the calibration reference on disk is not the one the criterion names"
    assert sum(1 for _ in OLD_ANCHOR.open()) - 1 == 42


def test_the_active_criterion_covers_both_limbs_and_extends_the_obduction_one_verbatim(criteria):
    """Clauses unchanged from the Letter's protocol; applied to the downward limb too."""
    c = active_criterion()
    assert c.id == study_criterion_version() == "phys_net_carbon_v1"
    assert c.two_limb
    assert c.clauses == criteria["phys_obduction_letter_b6"].clauses
    assert c.aou_sign("subduction") == "negative"
    assert c.aou_sign("obduction") == "positive"


def test_the_two_limbs_differ_only_in_the_aou_sign(criteria):
    """Mirrors the spec-level rule: the two physical parents differ only in directional sign."""
    c = criteria["phys_net_carbon_v1"]
    sub, obd = dict(c.limb_sign["subduction"]), dict(c.limb_sign["obduction"])
    assert sub.pop("AOU") != obd.pop("AOU")
    assert sub == obd == {"ABS_SAL": "either"}


def test_clause_four_is_the_only_thing_that_separates_the_active_criterion_from_the_companions(criteria):
    """Strictly tightening: the active criterion's clauses 1-3 say what the companion's 1-3 say."""
    c = criteria["phys_net_carbon_v1"]
    assert c.raw["relation"]["tighter_than"] == "phys_companion_2024"
    assert c.raw["relation"]["extends"] == "phys_obduction_letter_b6"
    assert len(c.clauses) == len(criteria["phys_companion_2024"].clauses) + 1
    assert c.raw["reuse_under_this_criterion"]["phys_companion_2024"] == "calibration_or_audit"


def test_a_proposed_criterion_cannot_open_a_batch(criteria):
    c = criteria["phys_net_carbon_v1"]
    if c.is_ruled:
        assert require_ruled(c) is c
        return
    with pytest.raises(ValueError, match="not ruled"):
        require_ruled(c)


def test_the_loader_refuses_a_criterion_that_paraphrases_instead_of_quoting(tmp_path):
    """A criterion must carry its clauses, its limbs, its signs and its unit — nothing implicit."""
    bad = tmp_path / "criteria.yaml"
    bad.write_text(
        "criteria:\n  - id: x\n    status: ruled\n    applies_to: [subduction]\n"
        "    limb_sign: {subduction: {AOU: negative}}\n    clauses: {1: a}\n    unit_of_judgement: level\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="AOU/ABS_SAL sign"):
        load_criteria(bad)
    bad.write_text(
        "criteria:\n  - id: x\n    status: proposed\n    applies_to: [subduction]\n"
        "    limb_sign: {subduction: {AOU: negative, ABS_SAL: either}}\n    clauses: {1: a, 3: c}\n"
        "    unit_of_judgement: level\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="without gaps"):
        load_criteria(bad)


def test_the_criteria_file_is_where_the_plan_says(criteria):
    assert CRITERIA_PATH == REPO / "config" / "criteria.yaml"


def test_the_anchors_recorded_base_rate_is_the_reference_files(criteria):
    """config/criteria.yaml said 22/42 for the b6 anchor until 2026-08-27; the frozen reference holds 18."""
    import re
    import pandas as pd

    b6 = criteria["phys_obduction_letter_b6"].raw["anchor"]
    if not OLD_ANCHOR.exists():
        pytest.skip("data/labels/external/calibration_reference_b6.csv is missing")
    assert hashlib.sha256(OLD_ANCHOR.read_bytes()).hexdigest().startswith(b6["sha256_16"])
    ref = pd.read_csv(OLD_ANCHOR)
    k, n = map(int, re.match(r"(\d+)/(\d+)", str(b6["base_rate"])).groups())
    assert (k, n) == (int(ref.REF_LABEL.sum()), len(ref)) == (18, 42)
