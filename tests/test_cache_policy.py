"""The fleet-cache recipe the study hands to argopod, and the float list it hands with it.

The build itself lives in argopod (`argopod.cache`). What this repository owns is the recipe —
which grids, which channels, which dates, which floats — and these tests hold it to the shape the
bound cache was actually built under. They are synthetic and offline: no cache, no raw frames, no
network. Whether a rebuilt grid is byte-identical to the bound one is `make check-cache`.

Four things can go wrong quietly here.

1. THE RECIPE STOPS SAYING WHAT WAS BUILT. The bound cache's 2,542 grids cannot be rebuilt from a
   fresh Argo pull, so the recipe is the only surviving description of them. A flavour that loses a
   channel, a date that moves, a placeholder rule that flips: each would build a different cache
   under the same name.

2. THE GRID KNOBS DRIFT. `Study.cache_policy()` passes argopod's DEFAULT `DetectionParams`, not the
   study's `params:` block, because the frozen cache was built under the defaults and the one knob
   the study moves acts at detection time. If a bin width or a smoothing window ever arrived from
   the study's block instead, every grid would change and nothing would say so.

3. A SETTING GETS TWO AUTHORS. The ranges, the floats left out and the backscatter smoother are
   declared once each, elsewhere in `config/events.yaml`, and joined into the policy in code. A
   copy of any of them inside the `cache:` block is refused rather than merged.

4. THE FLOAT LIST AND THE RECIPE DISAGREE. `config/fleet.csv` promises each float a flavour by
   name. A promised flavour the recipe does not declare builds nothing, quietly.
"""

from __future__ import annotations

import dataclasses
import pathlib

import pandas as pd
import pytest

from pins import PINS  # noqa: E402

from argopod import DetectionParams

from eddy_pump.manifest import MANIFEST_PATH, REPO_ROOT, load_manifest

REPO = pathlib.Path(__file__).resolve().parents[1]
FLEET = REPO / "config" / "fleet.csv"

#: The four grid flavours and the channels each carries, in order. Written out rather than read
#: from the manifest: a test that reads the file it is checking cannot notice the file changing.
EXPECTED_LABELS = {
    "paper_phys": ("AOU", "ABS_SAL", "SIGMA0"),
    "paper_bbp": ("AOU", "ABS_SAL", "BBP700_ADJUSTED", "SIGMA0"),
    "paper_nit": ("AOU", "ABS_SAL", "NITRATE_ADJUSTED", "SIGMA0"),
    "paper_all": ("AOU", "ABS_SAL", "NITRATE_ADJUSTED", "BBP700_ADJUSTED", "SIGMA0"),
}

#: The dates the bound cache keeps, both ends included.
EXPECTED_WINDOW = ("2009-01-08", "2026-03-15")

#: Every field of `DetectionParams` that reaches a grid: the two bin widths, the trimmed mean, the
#: scale, the column-name contract and the placeholder values. Named one by one so a new argopod
#: field is a deliberate addition here rather than a silent gap.
GRID_FIELDS = (
    "bin_width_fine", "bin_width_coarse",
    "trimmed_mean_window", "trimmed_mean_min_periods", "trimmed_mean_lo", "trimmed_mean_hi",
    "scale_window", "scale_min_periods", "scale_ref_range",
    "pressure_col", "cycle_col", "group_col", "meta_cols",
    "fill_values", "fill_abs_min",
)


@pytest.fixture(scope="module")
def study():
    return load_manifest()


@pytest.fixture(scope="module")
def policy(study):
    return study.cache_policy()


@pytest.fixture(scope="module")
def fleet():
    return pd.read_csv(FLEET)


# --------------------------------------------------------------------------- #
# 1. the recipe says what was built
# --------------------------------------------------------------------------- #
def test_the_four_grid_flavours_carry_exactly_the_channels_they_always_did(policy):
    assert dict(policy.labels) == EXPECTED_LABELS


def test_a_float_earns_the_richest_flavour_its_data_fits(policy):
    """The rule that decides a float's two file names, checked on the four cases that exist."""
    assert policy.label_for(["AOU", "ABS_SAL", "SIGMA0"]) == "paper_phys"
    assert policy.label_for(["AOU", "ABS_SAL", "SIGMA0", "BBP700_ADJUSTED"]) == "paper_bbp"
    assert policy.label_for(["AOU", "ABS_SAL", "SIGMA0", "NITRATE_ADJUSTED"]) == "paper_nit"
    assert policy.label_for(
        ["AOU", "ABS_SAL", "SIGMA0", "NITRATE_ADJUSTED", "BBP700_ADJUSTED"]) == "paper_all"


def test_the_window_and_the_placeholder_rule_are_the_ones_the_cache_was_built_under(policy):
    """A placeholder value becomes "no reading" BEFORE anything is derived from it: a saturation
    computed from a filled oxygen looks like an ordinary number and matches no range afterwards."""
    assert policy.window == EXPECTED_WINDOW
    assert policy.fill_policy == "mask"
    assert policy.params.fill_policy == "mask"
    assert policy.adjusted_fallback == "cycle"


def test_the_ceilings_gate_four_channels_and_report_the_fifth(policy):
    assert dict(policy.residual_ceilings) == {
        "AOU": 1000.0, "ABS_SAL": 1000.0, "NITRATE_ADJUSTED": 1000.0, "SIGMA0": 1000.0,
        "BBP700_ADJUSTED": None,
    }


# --------------------------------------------------------------------------- #
# 2. the grid knobs are argopod's defaults, and stay there
# --------------------------------------------------------------------------- #
def test_every_knob_that_reaches_a_grid_is_still_argopods_default(policy):
    """The frozen cache was built under the defaults, so the recipe must keep passing them.

    The study's own `params:` block moves one knob — the local-extremum test — and that knob acts
    when candidates are detected, never when a bin is filled. Passing the study's block here would
    be invisible today and would rewrite every grid the day a detection knob starts touching one.
    """
    default = DetectionParams()
    drifted = {f: (getattr(policy.params, f), getattr(default, f))
               for f in GRID_FIELDS if getattr(policy.params, f) != getattr(default, f)}
    assert not drifted, f"grid knobs no longer at argopod's default: {drifted}"


def test_the_detection_knob_the_study_moves_is_not_in_the_cache_recipe(study, policy):
    """The study sets `extremum_ratio_threshold: 0.0`; the recipe must still carry the default."""
    assert study.params.extremum_ratio_threshold == 0.0
    assert policy.params.extremum_ratio_threshold == DetectionParams().extremum_ratio_threshold


# --------------------------------------------------------------------------- #
# 3. every setting has one author
# --------------------------------------------------------------------------- #
def test_the_ranges_come_from_the_manifest_in_the_order_the_study_gives_them(study, policy):
    """Raw ranges act on the frame the floats arrive in; derived ranges on the channels the study
    detects on, at the only moment those channels exist."""
    ranges = study.cache_build_ranges()
    assert [v.name for v in policy.raw_ranges] == ["DOXY_ADJUSTED", "PSAL_ADJUSTED",
                                                   "TEMP_ADJUSTED"]
    assert [v.name for v in policy.derived_ranges] == ["AOU", "BBP700_ADJUSTED",
                                                       "NITRATE_ADJUSTED"]
    assert policy.raw_ranges == ranges["raw"]
    assert policy.derived_ranges == ranges["derived"]


def test_the_only_smoother_is_the_one_backscatter_asks_for(policy):
    assert [v.name for v in policy.prefilters] == ["BBP700_ADJUSTED"]
    assert all(v.pre_median_filter for v in policy.prefilters)


def test_the_nine_floats_left_out_reach_the_recipe_with_their_reasons(policy):
    expected = sorted(PINS["excluded_floats"]["salinity"] + PINS["excluded_floats"]["nitrate"])
    assert len(expected) == 9
    assert sorted(int(e.wmo) for e in policy.exclusions) == expected
    for e in policy.exclusions:
        assert e.ruled and e.ruled_by and e.reason, f"{e.wmo} arrives without its ruling"


@pytest.mark.parametrize("key", ["raw_ranges", "derived_ranges", "prefilters", "exclusions",
                                 "params"])
def test_a_setting_copied_into_the_cache_block_is_refused(tmp_path, key):
    """Each of these is declared once, elsewhere in the same file. A copy would let the file and
    the build disagree with nothing to say which one ran."""
    text = MANIFEST_PATH.read_text(encoding="utf-8").replace(
        "cache:\n  labels:", f"cache:\n  {key}: []\n  labels:", 1)
    with pytest.raises(ValueError, match=key):
        load_manifest(_manifest_copy(tmp_path, text))


def test_an_unknown_cache_key_names_the_ones_that_are_valid(tmp_path):
    text = MANIFEST_PATH.read_text(encoding="utf-8").replace(
        "cache:\n  labels:", "cache:\n  no_such_key: 3\n  labels:", 1)
    with pytest.raises(ValueError, match="unknown cache key"):
        load_manifest(_manifest_copy(tmp_path, text))
    with pytest.raises(ValueError, match="residual_ceilings"):
        load_manifest(_manifest_copy(tmp_path / "second", text))


def test_a_manifest_that_does_not_say_how_its_cache_is_built_is_refused(tmp_path):
    """The cache is what every candidate stands on. A recipe half in a file and half in code
    cannot be reproduced, so the block is required rather than defaulted."""
    text = MANIFEST_PATH.read_text(encoding="utf-8")
    start, end = text.index("\ncache:\n"), text.index("\nspec_version:")
    without = text[:start] + text[end:]
    assert "\ncache:\n" not in without
    with pytest.raises(ValueError, match="no 'cache' block"):
        load_manifest(_manifest_copy(tmp_path, without))


# --------------------------------------------------------------------------- #
# 4. the float list and the recipe agree
# --------------------------------------------------------------------------- #
def test_the_float_list_is_the_fleet_the_bound_cache_was_built_from(fleet):
    pinned = PINS["fleet"]
    assert list(fleet.columns) == ["WMO", "label", "tier"]
    assert len(fleet) == pinned["floats"]
    assert fleet.WMO.nunique() == pinned["floats"]
    assert fleet.label.value_counts().to_dict() == pinned["labels"]
    assert sorted(int(t) for t in fleet.tier.unique()) == pinned["tiers"]


def test_every_flavour_the_float_list_promises_is_one_the_recipe_declares(fleet, policy):
    """A promised flavour the recipe does not know builds nothing and says nothing."""
    assert set(fleet.label) <= set(policy.labels)


# --------------------------------------------------------------------------- #
# helper
# --------------------------------------------------------------------------- #
def _manifest_copy(tmp_path: pathlib.Path, text: str) -> pathlib.Path:
    """A mutated manifest in a throwaway tree whose repo root is still this repo.

    Relative paths in the manifest resolve against `<manifest>/../..`, so the copy has to sit two
    levels below a directory carrying this repo's `data/`. A symlink is enough and costs nothing.
    """
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True)
    for name in ("data", "results"):
        (root / name).symlink_to(REPO_ROOT / name)
    path = root / "config" / "events.yaml"
    path.write_text(text, encoding="utf-8")
    return path
