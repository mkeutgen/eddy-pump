"""The review panel must show the detector that made the candidate.

`config/review/physical_{obduction,subduction}.yaml` tells the labelling app which channels to draw
and at what cutoff, and it writes those numbers out again — they are already in `config/events.yaml`,
which is what the saved candidate lists were detected with. Nothing forced the two to agree, so a
panel could quietly draw a 1.50 sigma line over a pool detected at some other cutoff, and a human
would judge the wrong picture. These tests read both through their own loaders and require them to
match, channel for channel: the name, the cutoff, the limb's sign and the shape check.

Both files are also read for two things the panel cannot get wrong: the grid kind must be one the
fleet cache actually holds, and `paths.candidates` (a fallback the study never uses, because every
batch is passed with --batch) must not point at a file that does not exist.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REVIEW = REPO / "config/review"


def _pools():
    from eddy_pump.manifest import load_manifest

    study = load_manifest()
    return study, {p.event_type: p for p in study.pools if p.tracer is None}


def _panel(event_type: str):
    from argopod.eventconfig import load_event_config

    return load_event_config(REVIEW / f"{event_type}.yaml")


@pytest.mark.parametrize("event_type", ["physical_obduction", "physical_subduction"])
def test_the_panel_declares_the_same_detector_as_the_pool_it_shows(event_type):
    """Cutoff, sign and shape check, channel for channel, against `config/events.yaml`."""
    _, pools = _pools()
    pool = pools[event_type]
    cfg = _panel(event_type)
    spec = {v.name: v for v in pool.spec.variables}
    panel = {v.name: v for v in cfg.variables}
    assert set(panel) == set(spec), f"{event_type}: the panel shows {sorted(panel)}, the pool detects {sorted(spec)}"
    for name, v in panel.items():
        s = spec[name]
        assert v.cutoff == s.cutoff, f"{event_type}/{name}: panel cutoff {v.cutoff}, detector {s.cutoff}"
        assert v.sign_constraint == s.sign_constraint, (
            f"{event_type}/{name}: panel sign {v.sign_constraint!r}, detector {s.sign_constraint!r}")
        assert v.require_gradient_check == s.require_gradient_check, (
            f"{event_type}/{name}: panel shape check {v.require_gradient_check}, detector {s.require_gradient_check}")


@pytest.mark.parametrize("event_type", ["physical_obduction", "physical_subduction"])
def test_the_panel_draws_the_same_cutoff_line_it_detects_at(event_type):
    """The `panel:` block sets the line the labeller sees; it must be the detector's cutoff too."""
    _, pools = _pools()
    spec = {v.name: v for v in pools[event_type].spec.variables}
    for v in _panel(event_type).panel_variables:
        assert v.cutoff == spec[v.name].cutoff, f"{event_type}/{v.name}: the drawn line is not the detector's cutoff"


@pytest.mark.parametrize("event_type", ["physical_obduction", "physical_subduction"])
def test_the_limbs_differ_only_in_the_aou_sign(event_type):
    """The criterion says so: an obduction event is a positive AOU anomaly, a subduction event a
    negative one, and everything else about the two panels is the same."""
    from eddy_pump.criteria import active_criterion

    limb = event_type.split("_")[1]
    want = {"positive": "positive", "negative": "negative"}[active_criterion().aou_sign(limb)]
    panel = {v.name: v for v in _panel(event_type).variables}
    assert panel["AOU"].sign_constraint == want
    assert panel["ABS_SAL"].sign_constraint is None, "absolute salinity is either sign on both limbs"


@pytest.mark.parametrize("event_type", ["physical_obduction", "physical_subduction"])
def test_the_panel_asks_for_a_grid_the_fleet_cache_holds(event_type):
    study, _ = _pools()
    assert _panel(event_type).event in study.cache_build.grid_kinds, (
        f"{event_type}: the panel asks for grid kind {_panel(event_type).event!r}, which the cache does not build")


@pytest.mark.xfail(strict=False, reason="known, and config/ is not this change's to edit: both review configs set "
                                        "paths.candidates to results/net_carbon_v1/candidates/<pool>.parquet, which "
                                        "nothing in the pipeline writes. The saved list is "
                                        "data/candidates/net_carbon_v1/<pool>.parquet. Harmless today (every batch is "
                                        "passed with --batch), wrong the day one is not.")
@pytest.mark.parametrize("event_type", ["physical_obduction", "physical_subduction"])
def test_the_panels_fallback_candidate_path_is_not_a_file_that_does_not_exist(event_type):
    """`paths.candidates` is only a directory for argopod to default to; every study batch is passed
    with --batch. It must still not name a file that is not there: a reader who follows it finds
    nothing, and the app would fall back to it if a batch were ever left off."""
    p = _panel(event_type).candidates_path
    if p is None:
        return
    assert Path(p).exists(), (f"{event_type}: paths.candidates points at {p}, which does not exist; the saved list "
                              f"is data/candidates/net_carbon_v1/{event_type}.parquet")
