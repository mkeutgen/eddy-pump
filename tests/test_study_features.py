"""Every candidate of both physical pools carries features and a score — plan step 4 *(2026-08-26)*.

Pins the two tracked manifests, `data/features/net_carbon_v1/FEATURES_SHA256` and
`SCORES_SHA256`: features for every candidate of both pools from the bound cache, under the
pools' pinned spec ids; an upward classifier trained on the study's own obduction labels, with
honest float-grouped out-of-fold metrics; and a downward score transferred through a data-decided
sign alignment and checked against the companion's verified events. The score only orders which
panels a human sees first; it never enters a rate.
The large parquet files live under `results/` and are checked only when present.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pins import PINS  # noqa: E402

from eddy_pump.manifest import load_manifest

REPO = Path(__file__).resolve().parents[1]
MAN = REPO / "data/features/net_carbon_v1"
pytestmark = pytest.mark.skipif(not (MAN / "SCORES_SHA256").exists(), reason="run pipeline/features.py and pipeline/scores.py")


def _load(name):
    return json.loads((MAN / name).read_text().split("\n", 1)[1])


@pytest.fixture(scope="module")
def feats():
    return _load("FEATURES_SHA256")


@pytest.fixture(scope="module")
def scores():
    return _load("SCORES_SHA256")


def test_features_cover_every_candidate_of_both_pools_under_the_pinned_specs(feats):
    study = load_manifest()
    assert feats["study_id"] == study.study_id == "net_carbon_v1"
    assert feats["cache"]["fine_grids_sha256"] == study.cache.fine_grids_sha256
    by = {p["pool_id"]: p for p in feats["pools"]}
    assert set(by) == {"net_carbon_v1/physical/obduction", "net_carbon_v1/physical/subduction"}
    spec_of = {p.pool_id: p.spec.spec_id for p in study.pools}
    for pid, p in by.items():
        assert p["spec_id"] == spec_of[pid]
        assert p["rows"] == p["candidates"] and p["floats_missing"] == 0
        assert p["columns"] == 176
    assert by["net_carbon_v1/physical/obduction"]["rows"] == PINS["pool_rows"]["physical_obduction"]
    assert by["net_carbon_v1/physical/subduction"]["rows"] == PINS["pool_rows"]["physical_subduction"]


def test_the_upward_score_is_trained_on_the_studys_obduction_labels_and_measured_honestly(scores):
    u = scores["upward"]
    assert u["pool_rows"] == PINS["pool_rows"]["physical_obduction"]
    assert set(u["labels_by_source"]) == {"obduction_reviews"}
    assert u["obduction_reviews"] == 576   # the open-region analysis sample under phys_net_carbon_v1
    assert "grouped by float" in u["cv"]
    assert u["auc_oof_on_obduction_reviews"] > 0.7    # honest OOF on 576 study labels; refined in the classifier step
    assert u["rho_oof_on_obduction_reviews"] > 0.3
    cal = u["decile_calibration_on_obduction_reviews"]
    obs = [cal[k]["observed"] for k in sorted(cal, key=int)]
    assert obs[0] < 0.05 and obs[-1] > 0.4 and obs[-1] > obs[0]
    assert "latitude" in scores["features"]["excluded_ids"] and "longitude" in scores["features"]["excluded_ids"]


def test_the_downward_score_is_trained_on_the_companions_reviewed_detections_training_only(scores):
    """The companion reviewed every one of its R-detections; 13,471 of those keys are candidates of
    the active subduction pool and carry its verdict. A different criterion and a different frame,
    so training only (hard rule 5) -- which is all a score is for."""
    d = scores["downward"]
    assert d["pool_rows"] == 133_307
    assert "training only" in d["trained_on"] and "companion" in d["trained_on"]
    assert d["labelled_rows"] == 13_543 and d["accepted"] == 3_983
    assert set(d["labels_by_source"]) == {"companion_verified", "companion_rejected", "companion_detected_not_verified"}
    assert "grouped by float" in d["cv"]
    assert d["auc_oof_on_companion_labels"] > 0.8
    assert d["rho_oof_on_companion_labels"] > 0.5
    cal = d["decile_calibration_on_companion_labels"]
    obs = [cal[k]["observed"] for k in sorted(cal, key=int)]
    assert obs[-1] > obs[-2] > obs[-3] and obs[0] < 0.03
    # the transferred upward model is kept as a comparison and does worse
    tc = d["transfer_comparison"]
    assert tc["auc_verified_vs_rejected"] > 0.7 and tc["auc_verified_vs_pool"] > 0.65
    al = d["alignment"]
    assert "AOU_res_at_det" in al["flipped"] and "AOU_min_res" in al["swapped_and_negated"]
    assert d["companion_in_pool"]["verified"] == 3_983 and d["companion_in_pool"]["rejected"] == 233


def test_the_score_files_match_the_manifest_when_present(scores):
    sdir = REPO / "results/net_carbon_v1/scores"
    if not sdir.exists():
        pytest.skip("scores not on this machine")
    for name, sha in scores["files"].items():
        f = sdir / f"{name}.parquet"
        assert hashlib.sha256(f.read_bytes()).hexdigest() == sha, name
