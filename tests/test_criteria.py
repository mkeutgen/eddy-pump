"""The review criterion is an object with an id, and the study names exactly one.

The study has ONE criterion, `phys_net_carbon_v1`, covering both limbs (only the AOU sign differs);
its clauses are quoted from the protocol, not paraphrased; a PROPOSED criterion cannot open a batch;
and the loader refuses a criterion that leaves out its clauses, limbs, signs or unit.
"""

from __future__ import annotations

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


@pytest.fixture(scope="module")
def criteria():
    return load_criteria()


def test_only_the_study_criterion_exists(criteria):
    """One criterion, phys_net_carbon_v1; the historical ones left with the old labels."""
    assert set(criteria) == {"phys_net_carbon_v1"}
    assert criteria["phys_net_carbon_v1"].status in ("proposed", "ruled")


def test_the_criterion_is_the_protocols_four_clauses_verbatim(criteria):
    """Clause 4 is quoted from docs/LABELING_PROTOCOL.md, not paraphrased."""
    c = criteria["phys_net_carbon_v1"]
    protocol = (REPO / "docs" / "LABELING_PROTOCOL.md").read_text(encoding="utf-8")
    assert "Stands out against an otherwise regular background" in protocol
    assert c.clauses[4].startswith("Stands out against an otherwise regular background")
    assert sorted(c.clauses) == [1, 2, 3, 4]
    assert c.unit_of_judgement == "level"
    assert c.aou_sign("obduction") == "positive"
    assert c.aou_sign("subduction") == "negative"


def test_the_active_criterion_covers_both_limbs_and_names_its_two_anchors(criteria):
    c = active_criterion()
    assert c.id == study_criterion_version() == "phys_net_carbon_v1"
    assert c.two_limb
    assert sorted(c.clauses) == [1, 2, 3, 4]
    assert c.raw["anchors"] == {
        "obduction": "data/external/calibration_reference_b6.csv",
        "subduction": "data/labels/draws/calib_subduction_v1.reference.yaml",
    }


def test_the_two_limbs_differ_only_in_the_aou_sign(criteria):
    """Mirrors the spec-level rule: the two physical parents differ only in directional sign."""
    c = criteria["phys_net_carbon_v1"]
    sub, obd = dict(c.limb_sign["subduction"]), dict(c.limb_sign["obduction"])
    assert sub.pop("AOU") != obd.pop("AOU")
    assert sub == obd == {"ABS_SAL": "either"}


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
