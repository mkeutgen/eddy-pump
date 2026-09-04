"""The study's six saved candidate lists: their identity holds, and the refusals fire.

Pinned on `data/candidates/net_carbon_v1/`: one list per pool (the full table, sorted by key,
Zstd Parquet) with a sidecar naming the pinned spec id, the bound cache, the row count and a hash of
the key set; the six row counts; every row's `candidate_id` recomputes from the pool and the key;
every child is a subset of its own directional parent; a wrong cache and a wrong spec are refused
before anything is written. Whether the lists reproduce from the cache is `make verify-candidates`
(about 36 minutes), not a unit test.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from pins import PINS  # noqa: E402

from eddy_pump import candidates as C
from eddy_pump.domain import CacheIdentity
from eddy_pump.manifest import load_manifest

REPO = Path(__file__).resolve().parents[1]
SAVED = REPO / "data/candidates/net_carbon_v1"
pytestmark = pytest.mark.skipif(not (SAVED / C.SUMMARY).exists(), reason="run pipeline/detect.py --write")

ROWS = PINS["pool_rows"]
CACHE_SHA = PINS["cache_fingerprint"]


@pytest.fixture(scope="module")
def study():
    return load_manifest()


def test_six_saved_lists_with_the_frozen_row_counts_and_matching_sidecars(study):
    for p in study.pools:
        t, side = C.read_saved(study, p)
        assert len(t) == ROWS[p.event_type] == side["rows"], p.pool_id
        assert side["content_sha256"] == C.content_hash(t)
        assert side["file_sha256"] == hashlib.sha256(C.saved_path(study, p).read_bytes()).hexdigest()
        assert set(C.KEYS + C.MEASURES + C.ID_COLS) <= set(t.columns)
        assert (t.pool_id == p.pool_id).all() and (t.spec_id == p.spec_id).all() and (t.EVENT_TYPE == p.event_type).all()


def test_every_candidate_id_recomputes_from_the_pool_and_the_key(study):
    for p in study.pools:
        t, _ = C.read_saved(study, p, columns=C.KEYS + ["candidate_id"])
        sample = t.sample(min(500, len(t)), random_state=0)
        for r in sample.itertuples(index=False):
            assert p.key(int(r.WMO), int(round(r.CYCLE_NUMBER)), float(r.PRES_ADJUSTED)).candidate_id == r.candidate_id
        assert t.candidate_id.is_unique


def test_children_are_subsets_of_their_own_directional_parents(study):
    keys = {p.pool_id: C._keys(C.read_saved(study, p, columns=C.KEYS)[0]) for p in study.pools}
    for p in study.pools:
        if p.parent is not None:
            assert keys[p.pool_id] <= keys[p.parent.pool_id], p.pool_id
            assert p.parent.direction == p.direction


def test_every_sidecar_and_the_summary_name_the_bound_cache_and_the_pinned_spec(study):
    man = json.loads((SAVED / C.SUMMARY).read_text())
    assert man["cache"]["fine_grids_sha256"] == study.cache.fine_grids_sha256 == CACHE_SHA
    assert {l["pool_id"] for l in man["pools"]} == {p.pool_id for p in study.pools}
    for p in study.pools:
        _, side = C.read_saved(study, p, columns=C.KEYS)
        assert side["spec_id"] == p.spec_id and side["pool_id"] == p.pool_id and side["study_id"] == "net_carbon_v1"
        assert study.cache.matches(side["cache"])
        assert side["event_type"] == p.event_type and "_paper" not in side["event_type"]


def test_a_wrong_cache_is_refused_before_anything_runs(study, tmp_path):
    from dataclasses import replace

    wrong = replace(study, cache=CacheIdentity(path=tmp_path, fine_grids=0, fine_grids_sha256="0" * 64))
    with pytest.raises(ValueError, match="not the one .* is bound to"):
        C.require_bound_cache(wrong)
    with pytest.raises(ValueError, match="not the one .* is bound to"):
        C.detect_study(wrong, [])


def test_a_saved_list_from_another_spec_or_cache_is_refused(study, tmp_path):
    p = study.pools[0]
    t, side = C.read_saved(study, p, columns=C.KEYS)
    d = tmp_path / "net_carbon_v1"
    d.mkdir()
    t.to_parquet(d / f"{p.event_type}.parquet", index=False)
    (d / f"{p.event_type}.json").write_text(json.dumps(dict(side, spec_id="v1:0000000000000000")))
    with pytest.raises(ValueError, match="written under spec"):
        C.verify_saved(study, p, t, root=tmp_path)
    (d / f"{p.event_type}.json").write_text(json.dumps(dict(side, cache=dict(side["cache"], fine_grids_sha256="f" * 64))))
    with pytest.raises(ValueError, match="another cache"):
        C.verify_saved(study, p, t, root=tmp_path)


def test_verify_reports_exact_against_the_saved_list_itself(study):
    p = study.pools[0]
    t, _ = C.read_saved(study, p, columns=C.KEYS)
    v = C.verify_saved(study, p, t)
    assert v["exact"] and v["saved_only"] == 0 and v["new_only"] == 0 and v["saved"] == ROWS[p.event_type]


def test_a_child_without_its_parent_is_refused(study):
    child = next(p for p in study.pools if p.parent is not None)
    with pytest.raises(ValueError, match="without their directional parents"):
        C.detect_study(study, [child])


# --------------------------------------------------------------------------- #
# reading the grids: one door, and nothing swallowed
# --------------------------------------------------------------------------- #
def _fake_pair(d: Path, wmo: int, flavour: str, columns: list[str], corrupt: bool = False) -> None:
    """A coarse+fine pair for one float, valid or unreadable."""
    for grid in ("coarse", "fine"):
        path = d / f"{wmo}_{flavour}_{grid}.parquet"
        if corrupt:
            path.write_text("this is not a parquet file")
        else:
            frame = pd.DataFrame({c: pd.Series(dtype="float64") for c in columns})
            frame.to_parquet(path, index=False)


def test_a_grid_that_is_not_there_names_the_file_it_is_not_in(tmp_path):
    """argopod's provider owns the file layout, so the message names the paths, not the float."""
    with pytest.raises(FileNotFoundError) as exc:
        C.read_grids(tmp_path, 1900001, "paper_phys")
    assert "1900001_paper_phys_coarse.parquet" in str(exc.value)
    assert "1900001_paper_phys_fine.parquet" in str(exc.value)


def test_every_float_in_the_cache_gets_one_flavour_and_the_choice_is_not_the_globs(tmp_path):
    for wmo, flavour in ((1900001, "paper_phys"), (1900002, "paper_all")):
        _fake_pair(tmp_path, wmo, flavour, ["PRES_ADJUSTED"])
    # a coarse grid with no fine twin is not a float the cache can serve
    (tmp_path / "1900003_paper_nit_coarse.parquet").write_text("x")
    assert C.float_flavours(tmp_path) == {1900001: "paper_phys", 1900002: "paper_all"}
    # two flavours for one float would be a broken cache; the answer is still the same everywhere
    grids = {"paper_phys": {77: "paper_phys"}, "paper_all": {77: "paper_all"}}
    assert C.float_flavours(tmp_path, grids) == {77: "paper_all"}


def test_a_float_whose_grids_will_not_open_is_counted_and_named_not_skipped(study, tmp_path):
    """The old read swallowed every error, so a cache half on disk produced a short list that
    looked complete. Now the float is recorded, and `pipeline/detect.py` stops on the record."""
    pool = study.pool("physical", "obduction")
    columns = ["CYCLE_NUMBER", "PRES_ADJUSTED"] + \
        [f"{v.name}_SCALE_RES_ROB" for v in pool.spec.variables]
    _fake_pair(tmp_path, 1, "toy", columns)                  # readable: the flavour probe reads this
    _fake_pair(tmp_path, 2, "toy", columns, corrupt=True)
    grids = {"toy": {1: "toy", 2: "toy"}}

    failures: list[C.GridFailure] = []
    C.detect_pool(pool, study, grids, cache_dir=tmp_path, failures=failures)
    unreadable = [f for f in failures if f.wmo == 2]
    assert len(unreadable) == 1
    assert unreadable[0].stage == "read" and unreadable[0].flavour == "toy"
    assert unreadable[0].pool_id == pool.pool_id
    assert "2" in str(unreadable[0]) and "read" in str(unreadable[0])

    # and with nowhere to record it, the failure is raised rather than lost
    with pytest.raises(Exception):
        C.detect_pool(pool, study, grids, cache_dir=tmp_path)
