"""The active domain layer: one manifest, six pools, and the two invariants the study rests on.

reads  config/events.yaml, data/candidates/net_carbon_v1/CACHE_IDENTITY.json
writes nothing

Six things can go wrong silently here, and each has a test below.

1. The manifest stops describing the detectors that produced the saved lists. `config/events.yaml`
   is the one source now, so it must load into exactly the six specs whose ids the saved candidate
   lists were written under. A manifest that is merely *plausible* would silently re-cut every pool
   the first time anything detected from it. Held by the six pinned `spec_id`s.

2. The two physical parents drift apart. The apples-to-apples claim of the whole net-carbon paper
   is that gross subduction and obduction return are measured by one instrument and a sign. A
   cutoff that moves on one limb only turns the difference of two censuses into the difference of
   two detectors, and nothing downstream says so.

3. A child stops being its parent plus one term, so it cannot nest. That is not hypothetical: the
   superseded `carbon_subduction` (ABS_SAL 1.2 against the parent's 1.50) admitted 273 levels with
   no `physical_subduction` parent at all. The assertion is on the spec, before any detection.

4. Two pools collide on a key. The older key `(WMO, CYCLE, round(PRES), EVENT_TYPE)` is ambiguous
   between the limbs and between the studies, which is why the event type used to have to carry
   the study in it. `pool_id` and `candidate_id` exist to end that, so they are tested for
   distinctness and — the part that actually bites — for staying the same run after run.

5. An id moves between processes. Python's builtin `hash()` is salted per interpreter, so an id
   built with it differs on every run and no label can ever be joined back to its candidate. The
   candidate-id test asserts a literal digest and re-computes it in two subprocesses under
   different PYTHONHASHSEEDs.

6. An id moves because the upstream library grew a field. It did: argopod added nine knobs this
   study does not use, and because `spec_id` digested `dataclasses.asdict()` of argopod's own
   dataclasses, all six ids moved with not one cutoff, sign or detector knob changed. The digest
   now covers an explicit list owned by this repository
   (`eddy_pump.domain.DECLARED_VARIABLE_FIELDS` / `DECLARED_PARAM_FIELDS`). Section 8 tests both
   halves of that: a grown upstream field must not move an id, and a field added to the declared
   list must — plus the case that makes selecting fields safe rather than merely convenient, a
   spec that sets an undeclared field being refused by name instead of hashed around.
"""
import dataclasses
import os
import pathlib
import subprocess
import sys

import pytest

from pins import PINS  # noqa: E402

from argopod import DetectionParams, VariableConfig

from eddy_pump import (
    CANONICAL_EVENT_TYPES,
    DECLARED_PARAM_FIELDS,
    DECLARED_VARIABLE_FIELDS,
    CandidateKey,
    Direction,
    EventSpec,
    OutputRootPolicy,
    Tracer,
    load_manifest,
    undeclared_settings,
)
from eddy_pump import domain as _domain
from eddy_pump.manifest import MANIFEST_PATH, REPO_ROOT


STUDY_ID = "net_carbon_v1"

#: The six pools, in the manifest's own order. Written out rather than generated, so a pool that
#: silently disappears from the manifest fails here instead of shrinking a comprehension.
EXPECTED_POOL_IDS = (
    "net_carbon_v1/physical/obduction",
    "net_carbon_v1/nitrate/obduction",
    "net_carbon_v1/carbon/obduction",
    "net_carbon_v1/physical/subduction",
    "net_carbon_v1/nitrate/subduction",
    "net_carbon_v1/carbon/subduction",
)


@pytest.fixture(scope="module")
def study():
    return load_manifest()


# --------------------------------------------------------------------------- #
# 0. the manifest loads, and it is the one source
# --------------------------------------------------------------------------- #
def test_the_manifest_is_where_the_plan_says_it_is():
    assert MANIFEST_PATH == REPO_ROOT / "config" / "events.yaml"
    assert MANIFEST_PATH.exists()


def test_the_manifest_produces_exactly_six_detector_configurations(study):
    configs = study.detector_configs()
    assert len(configs) == 6
    assert tuple(configs) == EXPECTED_POOL_IDS
    for pool_id, cfg in configs.items():
        # The two argopod types and nothing else — this is what argopod.detect consumes.
        assert isinstance(cfg.params, DetectionParams), pool_id
        assert cfg.variables and all(hasattr(v, "cutoff") for v in cfg.variables), pool_id
        assert cfg.pool_id == pool_id
    assert {c.params for c in configs.values()} == {study.params}


def test_the_study_is_bound_to_the_fleet_cache_it_names(study):
    """The identity is read from CACHE_IDENTITY.json, never retyped into the manifest."""
    assert study.study_id == STUDY_ID
    binding = REPO_ROOT / "data" / "candidates" / "net_carbon_v1" / "CACHE_IDENTITY.json"
    assert study.cache.source == binding
    assert study.cache.fine_grids > 0 and len(study.cache.fine_grids_sha256) == 64
    # the same fingerprint, re-read: `matches` compares content and ignores the path (a
    # co-developer keeps the fleet cache elsewhere and must not be refused the correct cache)
    from eddy_pump import CacheIdentity
    assert study.cache.matches(CacheIdentity.from_json(binding))
    assert not study.cache.matches({"fine_grids": 721, "fine_grids_sha256": "0" * 64})


def test_this_machine_says_where_the_cache_is_and_the_fingerprint_still_decides(monkeypatch, tmp_path):
    """`CACHE_IDENTITY.json` records the absolute path of the machine the cache was built on,
    which is nobody else's path. `$EDDY_PUMP_CACHE` says where the same cache is here.

    It moves the path and nothing else. The fingerprint is what decides whether the grids are the
    right grids, so pointing this at the wrong directory fails loudly rather than quietly detecting
    something else — which is the whole reason a co-developer may be trusted to set it.
    """
    from eddy_pump import candidates as C
    from eddy_pump.study import CACHE_DIR_ENV

    monkeypatch.delenv(CACHE_DIR_ENV, raising=False)
    recorded = load_manifest().cache
    assert recorded.path == recorded.recorded_path   # nothing set: the recorded path stands

    here = tmp_path / "residual_cache_v4"
    here.mkdir()
    monkeypatch.setenv(CACHE_DIR_ENV, str(here))
    moved = load_manifest().cache
    assert moved.path == here                        # the grids are read from here
    assert moved.recorded_path == recorded.recorded_path != here   # provenance still names there
    assert (moved.fine_grids, moved.fine_grids_sha256) == (recorded.fine_grids,
                                                           recorded.fine_grids_sha256)
    # and the fingerprint is still checked: an empty directory is not the cache, wherever it is
    with pytest.raises(ValueError, match="not the one .* is bound to"):
        C.require_bound_cache(load_manifest())

    monkeypatch.setenv(CACHE_DIR_ENV, "~/eddy-pump-cache/residual_cache_v4")
    assert load_manifest().cache.path == pathlib.Path.home() / "eddy-pump-cache/residual_cache_v4"

    monkeypatch.setenv(CACHE_DIR_ENV, "   ")         # set but empty says nothing
    assert load_manifest().cache.path == recorded.recorded_path


def test_the_active_study_never_writes_into_the_letters_tree(study):
    """The retired output trees are read-only: the active study writes only under its own root."""
    assert study.output.root == REPO_ROOT / "results" / "net_carbon_v1"
    assert study.output.resolve("candidates", "x.parquet").is_relative_to(study.output.root)

    # rule 1: nothing escapes the study root, whatever the parts say
    with pytest.raises(ValueError, match="escapes the study's output root"):
        study.output.resolve("..", "obduction", "catalogue.csv")

    # rule 2: and even a path inside a permissive root is refused when it lands in the retired
    # tree. Exercised with a root wide enough for the escape check to pass, so the second rule
    # is the one under test rather than the first.
    wide = OutputRootPolicy(root=REPO_ROOT / "results",
                            forbidden_roots=study.output.forbidden_roots)
    assert wide.resolve("net_carbon_v1", "candidates").name == "candidates"
    with pytest.raises(ValueError, match="a retired output tree"):
        wide.resolve("obduction", "catalogue.csv")
    assert (REPO_ROOT / "results" / "obduction").resolve() in {
        p.resolve() for p in study.output.forbidden_roots}


# --------------------------------------------------------------------------- #
# 1. the one detector knob this study moves, and no other
# --------------------------------------------------------------------------- #

def test_the_paper_setting_is_the_only_knob_the_study_moves(study):
    letter, paper = dataclasses.asdict(DetectionParams()), dataclasses.asdict(study.params)
    assert {k for k in letter if letter[k] != paper[k]} == {"extremum_ratio_threshold"}
    assert paper["extremum_ratio_threshold"] == 0.0


# --------------------------------------------------------------------------- #
# 2. the two limbs are one instrument and a sign
# --------------------------------------------------------------------------- #
def test_the_two_physical_parents_differ_only_in_the_aou_sign(study):
    """One instrument and a sign. Asserted by stripping the signs and demanding identity, so a
    cutoff, a gradient check, a valid_range or a channel that moved on one limb only shows up."""
    up = study.pool("physical", Direction.OBDUCTION).spec
    down = study.pool("physical", Direction.SUBDUCTION).spec

    assert up.channels == down.channels == ("AOU", "ABS_SAL")
    stripped_up = [dataclasses.replace(v, sign_constraint=None) for v in up.variables]
    stripped_down = [dataclasses.replace(v, sign_constraint=None) for v in down.variables]
    assert stripped_up == stripped_down

    # and the sign is the ONE difference, in the direction the physics says
    assert [v.sign_constraint for v in up.variables] == ["positive", None]
    assert [v.sign_constraint for v in down.variables] == ["negative", None]
    assert Direction.OBDUCTION.aou_sign == "positive"
    assert Direction.SUBDUCTION.aou_sign == "negative"
    assert Direction.SUBDUCTION.opposite is Direction.OBDUCTION

    # the same statement in one line: exactly one field of one channel differs
    differing = [(a.name, f.name) for a, b in zip(up.variables, down.variables)
                 for f in dataclasses.fields(a)
                 if getattr(a, f.name) != getattr(b, f.name)]
    assert differing == [("AOU", "sign_constraint")]


def test_the_manifest_cannot_declare_a_sign_on_the_shared_aou_channel(tmp_path):
    """The sign has ONE author — the limb. A second one is how the parents drift apart."""
    text = MANIFEST_PATH.read_text(encoding="utf-8").replace(
        "    name: AOU\n", "    name: AOU\n    sign_constraint: positive\n", 1)
    broken = _manifest_copy(tmp_path, text)
    with pytest.raises(ValueError, match="must not declare a sign_constraint"):
        load_manifest(broken)


def test_the_manifest_cannot_contradict_the_code_s_aou_convention(tmp_path):
    text = MANIFEST_PATH.read_text(encoding="utf-8").replace(
        "    aou_sign: negative", "    aou_sign: positive", 1)
    broken = _manifest_copy(tmp_path, text)
    with pytest.raises(ValueError, match="aou_sign"):
        load_manifest(broken)


# --------------------------------------------------------------------------- #
# 3. NESTING
# --------------------------------------------------------------------------- #
def test_each_child_is_its_own_directional_parent_plus_exactly_one_tracer_term(study):
    """No more, no fewer — and the physical channels are the PARENT'S, field for field.

    Compared by NAME rather than by position. That was once load-bearing: `nitrate/obduction`
    carried its tracer term in the MIDDLE of the channel list (`tracer_position: 1`) where every
    other child appends it, inherited from `NITRATE_OBDUCTION_VARIABLES`. The N3 ruling of
    2026-08-24 (docs/DECISIONS.md §1, the nitrate variable-order ruling; `config/events.yaml` note (a)) moved the module to the append
    order and retired the key, so all four children are now `[parent…, tracer]` positionally too.
    The name-keyed comparison stays, because it asserts the thing that actually matters — the set
    of channels and every field on them is the parent's — without re-asserting an order that
    reaches only `PROVENANCE.json` and the triage feature list.
    """
    assert len(study.children) == 4
    for child in study.children:
        parent = child.parent
        assert parent is not None and parent.tracer is None
        assert parent.direction == child.direction, f"{child.pool_id} nests across the limbs"
        assert parent.pool_id == f"{STUDY_ID}/physical/{child.direction.value}"

        by_name = {v.name: v for v in child.spec.variables}
        # the parent's channels, verbatim
        assert [by_name[v.name] for v in parent.spec.variables] == list(parent.spec.variables), (
            f"{child.pool_id} does not carry {parent.pool_id}'s channels verbatim")
        # plus exactly one term
        extra = [v for v in child.spec.variables
                 if v.name not in {x.name for x in parent.spec.variables}]
        assert len(extra) == 1, f"{child.pool_id} adds {len(extra)} terms, not one"
        assert len(child.spec.variables) == len(parent.spec.variables) + 1

        term = extra[0]
        assert term.name == {Tracer.NITRATE: "NITRATE_ADJUSTED",
                             Tracer.CARBON: "BBP700_ADJUSTED"}[child.tracer]
        assert term.cutoff == 1.00, "the biotracer rule is 1.00 on both limbs, both tracers"


def test_the_biotracer_sign_flips_with_the_limb_and_the_cutoff_does_not(study):
    def term(channel, direction):
        pool = study.pool(channel, direction)
        parent = {v.name for v in pool.parent.spec.variables}
        return next(v for v in pool.spec.variables if v.name not in parent)

    assert term("nitrate", "obduction").sign_constraint == "positive"   # deep water: NO3-rich
    assert term("nitrate", "subduction").sign_constraint == "negative"  # surface water: NO3-poor
    # a particle EXCESS is the carbon signature on BOTH limbs
    assert term("carbon", "obduction").sign_constraint == "positive"
    assert term("carbon", "subduction").sign_constraint == "positive"
    # the CEILING ruling of 2026-08-25 brought this down from 100; the policy and band did not
    # move. Both limbs must carry the identical range or `Study.channel_ranges()` raises.
    assert term("nitrate", "obduction").valid_range == (0.0, 48.0)
    assert term("nitrate", "subduction").valid_range == (0.0, 48.0)
    assert term("carbon", "obduction").pre_median_filter is True
    assert term("carbon", "obduction").require_gradient_check is False


def test_the_superseded_carbon_subduction_declaration_cannot_come_back(study):
    """The one spec in the repo that could not nest: ABS_SAL 1.2 / BBP 0.7 admitted 273 levels
    with no physical_subduction parent. A revert must fail here, not in a census."""
    spec = study.pool("carbon", "subduction").spec
    assert (spec.channel("ABS_SAL").cutoff, spec.channel("BBP700_ADJUSTED").cutoff) == (1.50, 1.00)


def test_a_tracer_pool_without_its_own_limbs_parent_is_refused(tmp_path):
    """A child returned un-nested looks like a product and is a superset of one."""
    text = MANIFEST_PATH.read_text(encoding="utf-8")
    start = text.index("  - channel: physical\n    direction: subduction")
    end = text.index("  - channel: nitrate\n    direction: subduction")
    broken = _manifest_copy(tmp_path, text[:start] + text[end:])
    with pytest.raises(ValueError, match="has no physical/subduction parent"):
        load_manifest(broken)


# --------------------------------------------------------------------------- #
# 4. pool identity — stable strings, all distinct, no collisions
# --------------------------------------------------------------------------- #
def test_pool_ids_are_the_exact_strings_the_plan_specifies(study):
    assert study.pool_ids == EXPECTED_POOL_IDS
    for pool in study.pools:
        assert pool.pool_id == f"{STUDY_ID}/{pool.channel}/{pool.direction.value}"
        assert study[pool.pool_id] is pool


def test_no_two_pools_collide_on_any_identity(study):
    for attr in ("pool_id", "event_type", "spec_id"):
        values = [getattr(p, attr) for p in study.pools]
        assert len(set(values)) == 6, f"two pools share a {attr}: {sorted(values)}"


def test_event_type_means_the_proposal_class_only_and_never_the_product_line(study):
    """The key collision this whole refactor exists to remove.

    The legacy adapter stamps `physical_obduction_paper` into its fourth key field
    (the retired paper line's `_paper` tokens) because two product lines reused three event names. Here
    EVENT_TYPE is DERIVED from `pool_id`, so a line suffix cannot be expressed — the manifest
    records the legacy token separately, as provenance, and it is NOT the pool's EVENT_TYPE.
    """
    for pool in study.pools:
        assert pool.event_type in CANONICAL_EVENT_TYPES
        assert not pool.event_type.endswith("_paper")
        assert pool.event_type == f"{pool.channel}_{pool.direction.value}"
    with pytest.raises(ValueError, match="product line"):
        CandidateKey(f"{STUDY_ID}/physical_obduction_paper/obduction", 1, 1, 100.0)


# --------------------------------------------------------------------------- #
# 5. candidate_id — stable across processes, and it separates the pools
# --------------------------------------------------------------------------- #
# One fixed input, one literal digest. If a future change to the hashing moves this, every label
# ever written against a candidate_id is orphaned — so it must be a deliberate, visible edit.
_FIXED = (f"{STUDY_ID}/physical/subduction", 1902303, 12, 250.4)
_FIXED_PAYLOAD = "net_carbon_v1/physical/subduction|1902303|12|250"
_FIXED_ID = "096abcc91e816054fbd747dffe6656af"


def test_candidate_id_is_the_frozen_digest_of_a_spelled_out_payload():
    key = CandidateKey(*_FIXED)
    assert key.id_payload == _FIXED_PAYLOAD
    assert key.candidate_id == _FIXED_ID
    assert len(key.candidate_id) == 32
    assert key.event_type == "physical_subduction"
    assert key.pres_rounded == 250


def test_candidate_id_is_identical_in_other_processes_under_other_hash_seeds():
    """Python's builtin `hash()` is salted per interpreter (PYTHONHASHSEED). An id built with it
    would differ between two runs of the same code over the same row. hashlib is not salted, and
    this is the test that proves the implementation actually uses it."""
    code = (
        "from eddy_pump import CandidateKey;"
        f"print(CandidateKey({_FIXED[0]!r}, {_FIXED[1]}, {_FIXED[2]}, {_FIXED[3]}).candidate_id)"
    )
    for seed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed,
                   PYTHONPATH=str(REPO_ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             env=env, check=True)
        assert out.stdout.strip() == _FIXED_ID, f"PYTHONHASHSEED={seed}"


def test_the_same_level_in_two_pools_gets_two_ids(study):
    """The reason the id takes `pool_id` at all: subduction and obduction pools overlap on the
    four readable fields, so those alone are ambiguous (docs/IMPLEMENTATION_NOTES.md §1)."""
    ids = {p.pool_id: p.key(1902303, 12, 250.4).candidate_id for p in study.pools}
    assert len(set(ids.values())) == 6, ids
    assert ids[f"{STUDY_ID}/physical/subduction"] == _FIXED_ID
    assert ids[f"{STUDY_ID}/physical/obduction"] == "8d58bd6ac27516a690126924d8757033"
    assert ids[f"{STUDY_ID}/carbon/subduction"] == "bc5a3a6c959c711c2cadddd1dfe036e4"


def test_pressure_enters_the_id_rounded_by_the_legacy_rule(study):
    """`round(PRES_ADJUSTED)`, half-to-even — the same rule `obduction.detect_all._keys` and
    `config.HIERARCHY_KEY_PRES_DECIMALS` use, so a legacy row and an active row describing one
    level land on one rounded pressure."""
    pool = study.pool("physical", "subduction")
    assert [pool.key(1, 1, p).pres_rounded for p in (250.4, 250.5, 250.6, 251.5)] == [
        250, 250, 251, 252]
    assert pool.key(1, 1, 250.4).candidate_id == pool.key(1, 1, 249.8).candidate_id


def test_a_malformed_pool_id_is_refused_at_construction():
    for bad in ("net_carbon_v1/physical", "net_carbon_v1/physical/sideways",
                "net_carbon_v1//subduction", "physical/subduction"):
        with pytest.raises(ValueError):
            CandidateKey(bad, 1, 1, 100.0)


# --------------------------------------------------------------------------- #
# 6. spec_id — moves when the spec moves, and only then
# --------------------------------------------------------------------------- #
#: The six pinned ids, as `config/events.yaml` declares them. The loader already refuses a
#: manifest whose pin disagrees with its content; these literals name the values so a re-pin is
#: a visible edit in a diff rather than a number nobody wrote down.
#:
#: Re-pinned 2026-08-25 by the plausible-range ruling, all six at once, and the previous values
#: are kept below so the move is readable rather than merely asserted. Two causes, both deliberate:
#: `range_policy` and `range_tolerance` joined `DECLARED_VARIABLE_FIELDS` (this study now sets
#: them), and AOU — a channel of all six specs — gained the `valid_range` it never had. No cutoff,
#: sign constraint, gradient flag or detector knob moved.
SPEC_IDS_BEFORE_THE_RANGE_RULING = {
    "net_carbon_v1/physical/obduction": "v1:eb851366426ddcb4",
    "net_carbon_v1/nitrate/obduction": "v1:17a644a4c0d15eab",
    "net_carbon_v1/carbon/obduction": "v1:7e1bd22d3ee73b6d",
    "net_carbon_v1/physical/subduction": "v1:871ea64319d348ba",
    "net_carbon_v1/nitrate/subduction": "v1:07726d8b4f902263",
    "net_carbon_v1/carbon/subduction": "v1:270223d6a04ffdd9",
}

#: RE-PINNED AGAIN 2026-08-26 BY THE NITRATE-FILTER RULING — but only TWO of the six, and that
#: contrast is the point. The oxygen ruling moved all six because it touched AOU, a channel every
#: spec carries. This one repairs `NITRATE_ADJUSTED`'s range POLICY (argopod's whole-cycle
#: `drop_cycle` default -> `clip`, band 2.0 umol/kg), and only the two nitrate specs carry that
#: channel. A physical or carbon pin that moved here would mean the edit reached a spec it had no
#: business reaching.
SPEC_IDS_BEFORE_THE_NITRATE_RULING = {
    "net_carbon_v1/physical/obduction": "v1:31619ba88c252ed8",
    "net_carbon_v1/nitrate/obduction": "v1:f066d38eb0f36753",
    "net_carbon_v1/carbon/obduction": "v1:feb637c336821f05",
    "net_carbon_v1/physical/subduction": "v1:6b8de00d7d53e118",
    "net_carbon_v1/nitrate/subduction": "v1:29725acf720740f0",
    "net_carbon_v1/carbon/subduction": "v1:768ee191d3cfcd9f",
}

#: The pins the nitrate range POLICY ruling of 2026-08-26 produced, before the CEILING ruling
#: of 2026-08-25 moved the same two pools again by changing 100 to 48.
SPEC_IDS_BEFORE_THE_CEILING_RULING = PINS["spec_ids_before_the_ceiling_ruling"]

EXPECTED_SPEC_IDS = PINS["spec_ids"]

#: The pools whose channel set contains NITRATE_ADJUSTED — the only ones the 2026-08-26 ruling
#: may touch.
NITRATE_POOLS = {"net_carbon_v1/nitrate/obduction", "net_carbon_v1/nitrate/subduction"}


def test_the_six_spec_ids_are_the_pinned_ones(study):
    assert {p.pool_id: p.spec_id for p in study.pools} == EXPECTED_SPEC_IDS


def test_the_nitrate_ruling_moved_exactly_the_two_nitrate_pins_and_no_other(study):
    """The re-pin of 2026-08-26, asserted as a SCOPE and not as two new literals.

    A range policy is a property of ONE channel, so it may move exactly the pins of the pools that
    declare that channel. The four other pools must read identically to what they read before the
    ruling — that is the whole difference from the oxygen ruling of the day before, which touched
    AOU and therefore had to move all six.
    """
    now = {p.pool_id: p.spec_id for p in study.pools}
    moved = {k for k, v in SPEC_IDS_BEFORE_THE_NITRATE_RULING.items() if now[k] != v}
    assert moved == NITRATE_POOLS
    assert {p.pool_id for p in study.pools
            if "NITRATE_ADJUSTED" in p.spec.channels} == NITRATE_POOLS
    for pool_id in set(now) - NITRATE_POOLS:
        assert now[pool_id] == SPEC_IDS_BEFORE_THE_NITRATE_RULING[pool_id], pool_id
    assert len(set(now.values())) == 6, "two pools collided on one id"


def test_all_six_ids_moved_with_the_range_ruling_and_none_stayed_behind(study):
    """The re-pin of 2026-08-25, asserted as a MOVE rather than as six new literals.

    Six pools, six ids, six different old values: the ruling touched every spec, so a pin that
    still reads its pre-ruling value would mean one spec did not receive the range — which on
    AOU, a channel of all six, could only happen by a mistake.
    """
    now = {p.pool_id: p.spec_id for p in study.pools}
    assert set(now) == set(SPEC_IDS_BEFORE_THE_RANGE_RULING)
    for pool_id, before in SPEC_IDS_BEFORE_THE_RANGE_RULING.items():
        assert now[pool_id] != before, f"{pool_id} did not move"
    assert len(set(now.values())) == 6, "two pools collided on one id"
    assert not (set(now.values()) & set(SPEC_IDS_BEFORE_THE_RANGE_RULING.values()))


def test_the_ruling_moved_the_ranges_and_left_every_detect_time_gate_alone(study):
    """What moved and what did not, field by field — the claim the re-pin is justified by.

    Every channel of every pool must still carry exactly the cutoff, sign constraint, gradient
    check and second-derivative threshold it had before the ruling. If one of those moved, the
    re-pin is hiding a detector change behind a data-hygiene ruling.
    """
    gates = {
        ("physical", "obduction"): [("AOU", 1.50, "positive", True),
                                    ("ABS_SAL", 1.50, None, True)],
        ("physical", "subduction"): [("AOU", 1.50, "negative", True),
                                     ("ABS_SAL", 1.50, None, True)],
        ("nitrate", "obduction"): [("AOU", 1.50, "positive", True),
                                   ("ABS_SAL", 1.50, None, True),
                                   ("NITRATE_ADJUSTED", 1.00, "positive", True)],
        ("nitrate", "subduction"): [("AOU", 1.50, "negative", True),
                                    ("ABS_SAL", 1.50, None, True),
                                    ("NITRATE_ADJUSTED", 1.00, "negative", True)],
        ("carbon", "obduction"): [("AOU", 1.50, "positive", True),
                                  ("ABS_SAL", 1.50, None, True),
                                  ("BBP700_ADJUSTED", 1.00, "positive", False)],
        ("carbon", "subduction"): [("AOU", 1.50, "negative", True),
                                   ("ABS_SAL", 1.50, None, True),
                                   ("BBP700_ADJUSTED", 1.00, "positive", False)],
    }
    for (channel, direction), expected in gates.items():
        spec = study.pool(channel, direction).spec
        got = [(v.name, v.cutoff, v.sign_constraint, v.require_gradient_check)
               for v in spec.variables]
        assert got == expected, f"{channel}/{direction}: a DETECT-TIME gate moved"
        for v in spec.variables:
            assert v.second_deriv_threshold == 0.001, v.name
    # and N1 in particular is still unruled: the carbon gate is still 1.00 on both limbs
    for direction in ("obduction", "subduction"):
        assert study.pool("carbon", direction).spec.channel(
            "BBP700_ADJUSTED").cutoff == 1.00


def test_ABS_SAL_is_deliberately_still_unranged(study):
    """The residual hole, named so it cannot be mistaken for an oversight.

    The ruling of 2026-08-25 covered dissolved oxygen and particle backscatter. It did NOT cover
    temperature or salinity, so `ABS_SAL` — a detection channel of all six pools, derived from a
    `PSAL_ADJUSTED` that reaches 77.57 on this fleet — still declares no range, and neither do
    `SIGMA0` or the raw T/S columns behind them. That is a decision the user has not taken, and
    a test that quietly ranged them would be taking it for them.
    """
    for pool in study.pools:
        assert pool.spec.channel("ABS_SAL").valid_range is None, pool.pool_id
    assert "ABS_SAL" not in {v.name for v in study.channel_ranges()}
    assert "ABS_SAL" not in {v.name for v in study.input_ranges}


def test_spec_id_does_not_change_when_the_spec_does_not(study):
    """Same content, freshly built object: same id. No object identity, no timestamp, no dict
    iteration order may reach the digest."""
    for pool in study.pools:
        rebuilt = EventSpec(name="anything-at-all", direction=pool.direction, tracer=pool.tracer,
                            variables=tuple(pool.spec.variables), params=pool.spec.params,
                            version=pool.spec.version)
        assert rebuilt.spec_id == pool.spec_id
        assert rebuilt.content_digest == pool.spec.content_digest
        # and asking twice is the same answer
        assert pool.spec.spec_id == pool.spec.spec_id


@pytest.mark.parametrize("field,value", [("cutoff", 1.75),
                                         ("sign_constraint", None),
                                         ("require_gradient_check", False),
                                         ("valid_range", (0.0, 90.0)),
                                         ("pre_median_filter", True),
                                         ("second_deriv_threshold", 0.002)])
def test_spec_id_changes_when_a_variable_changes(study, field, value):
    pool = study.pool("nitrate", "subduction")
    head, *tail = pool.spec.variables
    moved = dataclasses.replace(pool.spec, variables=(dataclasses.replace(head, **{field: value}),
                                                      *tail))
    assert moved.spec_id != pool.spec_id, f"{field} moved and the spec_id did not"


def test_spec_id_does_NOT_change_when_only_the_declaration_order_changes(study):
    """The ruling of 2026-08-24, and the reason it matters.

    Detection is a joint-AND read as a SET: `obduction.detect_all` feeds `EVENT_SPECS` to
    `detect_from_grids` (it opens no YAML), and `argopod.eventconfig` reads
    `{v.name for v in cfg.variables}`. Which order nitrate obduction's channels were declared in
    was an open question when this test was written, and it was settled on 2026-08-24 (N3) AFTER
    the six ids were pinned. This property is why that was possible: `v1:17a644a4c0d15eab` did not
    move when the channels did. Had `spec_id` been order-sensitive, the ruling would have moved
    every nitrate id and orphaned every label keyed to one, for no scientific reason at all.
    """
    for pool in study.pools:
        reordered = dataclasses.replace(pool.spec,
                                        variables=tuple(reversed(pool.spec.variables)))
        assert reordered.declared_order == tuple(reversed(pool.spec.declared_order))
        assert reordered.spec_id == pool.spec_id, pool.pool_id
        assert reordered.content_digest == pool.spec.content_digest, pool.pool_id


def test_spec_id_changes_when_the_detector_or_the_version_changes(study):
    pool = study.pool("physical", "subduction")
    other_detector = dataclasses.replace(pool.spec, params=DetectionParams())
    assert other_detector.spec_id != pool.spec_id

    bumped = dataclasses.replace(pool.spec, version="v2")
    assert bumped.spec_id != pool.spec_id
    assert bumped.content_digest == pool.spec.content_digest   # the version, not the content


# --------------------------------------------------------------------------- #
# 7. the build-time prefilter terms, kept separate from the detect-time gate
# --------------------------------------------------------------------------- #
def test_the_prefilter_terms_are_declared_where_they_act(study):
    """`valid_range` and `pre_median_filter` are the only two VariableConfig fields consumed by
    `argopod.detect.prefilter.apply_prefilters`, which runs at CACHE-BUILD time. They are declared
    under their channel's `prefilter:` block, folded into the one VariableConfig argopod expects,
    and reported by `EventSpec.prefilter_terms` so nobody mistakes a declared term for an applied
    one — both caches were built with prefilters off."""
    AOU = {"valid_range": [-600.0, 600.0], "range_policy": "mask"}
    for direction in ("obduction", "subduction"):
        assert study.pool("physical", direction).spec.prefilter_terms == {"AOU": AOU}
        assert study.pool("nitrate", direction).spec.prefilter_terms == {
            "AOU": AOU,
            # RULED 2026-08-26. Until then this carried no policy of its own and digested as
            # argopod's whole-cycle `drop_cycle` default, which `Study.channel_ranges()`
            # withholds from the cache builder — so the range was declared and reached nothing.
            # `clip` is a sample policy, so it is now handed over and applied.
            # The ceiling came down from 100 to 48 on 2026-08-25; policy and band unchanged.
            "NITRATE_ADJUSTED": {"valid_range": [0.0, 48.0], "range_policy": "clip",
                                 "range_tolerance": 2.0}}
        assert study.pool("carbon", direction).spec.prefilter_terms == {
            "AOU": AOU,
            "BBP700_ADJUSTED": {"valid_range": [0.0, 0.1], "range_policy": "clip",
                                "range_tolerance": 0.005, "pre_median_filter": True}}


def test_a_prefilter_term_declared_at_detection_level_is_refused(tmp_path):
    text = MANIFEST_PATH.read_text(encoding="utf-8").replace(
        "    channel: NITRATE_ADJUSTED\n",
        "    channel: NITRATE_ADJUSTED\n    valid_range: [0.0, 100.0]\n", 1)
    broken = _manifest_copy(tmp_path, text)
    with pytest.raises(ValueError, match="CACHE-BUILD time"):
        load_manifest(broken)


def test_spec_id_is_content_only_and_not_the_pools_name(study):
    """Two pools declaring the same detector must get the same id — they are one instrument
    pointed at different water. The name must not leak into the digest."""
    pool = study.pool("carbon", "obduction")
    renamed = dataclasses.replace(pool.spec, name="something_else")
    assert renamed.spec_id == pool.spec_id


def test_a_manifest_whose_pin_disagrees_with_its_content_refuses_to_load(tmp_path):
    """The pin is the point: edit a cutoff without re-pinning and the load fails, naming both."""
    text = MANIFEST_PATH.read_text(encoding="utf-8").replace(
        "    - {name: ABS_SAL, cutoff: 1.50, require_gradient_check: true}",
        "    - {name: ABS_SAL, cutoff: 1.75, require_gradient_check: true}", 1)
    broken = _manifest_copy(tmp_path, text)
    with pytest.raises(ValueError, match="pins spec_id"):
        load_manifest(broken)


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


# --------------------------------------------------------------------------- #
# 8. spec_id covers a declared field list owned here, not argopod's inventory
# --------------------------------------------------------------------------- #
# What happened. `EventSpec.content` used to digest `dataclasses.asdict()` of argopod's
# `VariableConfig` and `DetectionParams`. argopod then grew nine fields this study neither sets
# nor needs — `range_policy`, `range_tolerance`, `range_tolerance_fraction`,
# `artifact_cycle_fraction` on VariableConfig; `fill_policy`, `fill_values`, `fill_abs_min`,
# `fill_columns`, `fill_warn` on DetectionParams — and all six ids moved at once. Nothing
# scientific had changed. A `spec_id` that churns on somebody else's release note cannot be what
# a label, a saved list, a catalogue and a flux are keyed to, so the digest now covers a list this
# repository declares. These tests hold that fix to all four of its promises.


@dataclasses.dataclass(frozen=True)
class _GrownVariableConfig(VariableConfig):
    """argopod's VariableConfig with one more field, simulated without touching argopod.

    A hand-written subclass is the honest simulation: the nine real fields all arrived with
    defaults and this study leaves every one of them alone, which is precisely the case that used
    to move six pins.
    """

    a_knob_this_study_does_not_use: str = "argopod-default"
    #: And one that JSON cannot render at all. The digest must not merely IGNORE an undeclared
    #: field, it must never touch it: "cannot move a spec_id" has to include "cannot stop one
    #: being computed". This is why `declared_content` reads field by field instead of filtering
    #: `dataclasses.asdict()`.
    an_opaque_object_argopod_may_hold: object = object()


@dataclasses.dataclass(frozen=True)
class _GrownDetectionParams(DetectionParams):
    """The same simulation for the detector block."""

    another_knob_this_study_does_not_use: float = 0.0


def _grown(spec: EventSpec) -> EventSpec:
    """`spec` rebuilt on the grown dataclasses, field for field, nothing else changed."""
    def widen(obj, cls, base):
        return cls(**{f.name: getattr(obj, f.name) for f in dataclasses.fields(base)})

    return dataclasses.replace(
        spec,
        variables=tuple(widen(v, _GrownVariableConfig, VariableConfig) for v in spec.variables),
        params=widen(spec.params, _GrownDetectionParams, DetectionParams),
    )


#: Of the nine fields argopod grew, the SEVEN this study still does not set. `range_policy` and
#: `range_tolerance` left this list on 2026-08-25 — the study now sets both, so they were added to
#: `DECLARED_VARIABLE_FIELDS` and all six pins moved. That is rule 2 of the declared-field block
#: happening on purpose, not the regression coming back; the regression was a pin following an
#: upstream INVENTORY, and these seven are what still proves it cannot.
STILL_UNDECLARED = {
    VariableConfig: ("range_tolerance_fraction", "artifact_cycle_fraction"),
    DetectionParams: ("fill_policy", "fill_values", "fill_abs_min", "fill_columns", "fill_warn"),
}

#: The two that were promoted, and the ruling that promoted them.
DELIBERATELY_DECLARED_2026_08_25 = {VariableConfig: ("range_policy", "range_tolerance")}


def test_the_seven_fields_this_study_still_does_not_set_are_undeclared(study):
    """The regression, named. Each is a field of the argopod dataclass AND absent from this
    repository's declared list — which is the only reason an argopod release cannot churn a pin."""
    declared = {VariableConfig: DECLARED_VARIABLE_FIELDS, DetectionParams: DECLARED_PARAM_FIELDS}
    seen = 0
    for cls, names in STILL_UNDECLARED.items():
        upstream = {f.name for f in dataclasses.fields(cls)}
        for name in names:
            if name not in upstream:
                continue          # argopod may drop one; the point is only made by those present
            seen += 1
            assert name not in declared[cls], (
                f"{cls.__name__}.{name} is declared, so it is IN the digest — that re-pins "
                f"every spec_id it appears on and must be a deliberate, written-down act")
    assert seen, "none of the seven fields is present; the simulation below still covers the rule"
    # and the study sets none of THOSE, on any pool — an undeclared field left at its default
    for pool in study.pools:
        for v in pool.spec.variables:
            assert undeclared_settings(v, DECLARED_VARIABLE_FIELDS) == {}, pool.pool_id
        assert undeclared_settings(pool.spec.params, DECLARED_PARAM_FIELDS) == {}, pool.pool_id


def test_the_two_promoted_fields_are_declared_and_the_study_really_sets_them(study):
    """The other side of the same coin, and the justification for the six moved pins.

    Declaring a field is only defensible when the study SETS it. `range_policy` is set to
    something other than argopod's default on the two channels the 2026-08-25 ruling names, and
    `range_tolerance` is set explicitly wherever a clip band exists — so the digest covers a real
    difference and not a row of defaults.
    """
    for cls, names in DELIBERATELY_DECLARED_2026_08_25.items():
        upstream = {f.name for f in dataclasses.fields(cls)}
        for name in names:
            assert name in upstream, f"argopod no longer carries {cls.__name__}.{name}"
            assert name in DECLARED_VARIABLE_FIELDS, name
    policies = {v.name: v.range_policy for p in study.pools for v in p.spec.variables}
    assert policies["AOU"] == "mask"
    assert policies["BBP700_ADJUSTED"] == "clip"
    assert policies["ABS_SAL"] == "drop_cycle"          # untouched: it declares no range at all
    assert policies["NITRATE_ADJUSTED"] == "clip"       # ruled 2026-08-26, closing N4
    tolerances = {v.name: v.range_tolerance for p in study.pools for v in p.spec.variables}
    assert tolerances["BBP700_ADJUSTED"] == 0.005
    assert tolerances["NITRATE_ADJUSTED"] == 2.0
    # NOTHING this study declares may still delete a whole dive on one bad sample
    assert "drop_cycle" not in {v.range_policy for p in study.pools for v in p.spec.variables
                                if v.valid_range is not None}
    assert tolerances["AOU"] is None, "a 'mask' policy declares no near-boundary band"
    # the fraction is never consulted, which is why it stays undeclared
    for pool in study.pools:
        for v in pool.spec.variables:
            if v.valid_range is not None and v.range_policy == "clip":
                assert v.range_tolerance is not None, (
                    f"{v.name} clips on a tolerance derived from range_tolerance_fraction, which "
                    f"the digest does not cover")


def test_a_field_argopod_grows_does_not_move_one_spec_id(study):
    """The bug, as a test. Widen both upstream dataclasses, set nothing new, and every id holds.

    Under the old `dataclasses.asdict()` digest this failed on all six pools at once — which is
    exactly what happened when the nine real fields landed.
    """
    for pool in study.pools:
        widened = _grown(pool.spec)
        assert isinstance(widened.variables[0], _GrownVariableConfig)
        assert widened.variables[0].a_knob_this_study_does_not_use == "argopod-default"
        # the JSON-hostile one included: an undeclared field is never touched, only unread
        assert not isinstance(widened.variables[0].an_opaque_object_argopod_may_hold, str)
        assert widened.spec_id == pool.spec_id, pool.pool_id
        assert widened.content_digest == pool.spec.content_digest, pool.pool_id
    assert {p.pool_id: _grown(p.spec).spec_id for p in study.pools} == EXPECTED_SPEC_IDS


def test_adding_a_field_to_the_declared_list_does_move_the_spec_id(monkeypatch, study):
    """The other half: re-pinning must be POSSIBLE, and it is a one-line, reviewable edit.

    Same object as the test above — the only thing that changes is the list this repository
    declares. That is the whole design: the ids follow the list, not argopod's inventory.
    """
    pool = study.pool("physical", "subduction")
    before = _grown(pool.spec).spec_id
    assert before == pool.spec_id

    monkeypatch.setattr(_domain, "DECLARED_VARIABLE_FIELDS",
                        DECLARED_VARIABLE_FIELDS + ("a_knob_this_study_does_not_use",))
    after = _grown(pool.spec).spec_id
    assert after != before, "the declared list grew and the spec_id did not follow it"

    # and the detector half moves every pool, because every pool shares one params block
    monkeypatch.setattr(_domain, "DECLARED_VARIABLE_FIELDS", DECLARED_VARIABLE_FIELDS)
    monkeypatch.setattr(_domain, "DECLARED_PARAM_FIELDS",
                        DECLARED_PARAM_FIELDS + ("another_knob_this_study_does_not_use",))
    moved = {p.pool_id: _grown(p.spec).spec_id for p in study.pools}
    assert all(moved[k] != v for k, v in EXPECTED_SPEC_IDS.items())


def test_a_spec_that_sets_an_undeclared_field_is_refused_by_name(study):
    """The case that makes selecting fields safe rather than merely convenient.

    A list that only SELECTED fields would be worse than the bug it replaces: a spec could set an
    undeclared knob, detect differently, and hash identically to one that does not — the manifest
    claiming one instrument while the pin certifies another. So a value on any field outside the
    declared list is refused, and the message names the field.
    """
    pool = study.pool("nitrate", "subduction")
    head, *tail = _grown(pool.spec).variables
    with pytest.raises(ValueError, match="a_knob_this_study_does_not_use"):
        EventSpec(name=pool.spec.name, direction=pool.direction, tracer=pool.tracer,
                  variables=(dataclasses.replace(
                      head, a_knob_this_study_does_not_use="clip"), *tail),
                  params=pool.spec.params, version=pool.spec.version)

    with pytest.raises(ValueError, match="another_knob_this_study_does_not_use"):
        EventSpec(name=pool.spec.name, direction=pool.direction, tracer=pool.tracer,
                  variables=pool.spec.variables,
                  params=_GrownDetectionParams(
                      **{f.name: getattr(pool.spec.params, f.name)
                         for f in dataclasses.fields(DetectionParams)},
                      another_knob_this_study_does_not_use=1.0),
                  version=pool.spec.version)

    # and the refusal says WHY, not just what
    with pytest.raises(ValueError, match="provenance hole"):
        EventSpec(name=pool.spec.name, direction=pool.direction, tracer=pool.tracer,
                  variables=(dataclasses.replace(
                      head, a_knob_this_study_does_not_use="clip"), *tail),
                  params=pool.spec.params, version=pool.spec.version)


def _an_undeclared_upstream_field(cls, declared):
    """One real argopod field this repository does not declare, with a value != its default.

    Chosen from the live dataclass rather than hardcoded, so the test states the rule and does not
    quietly pass the day argopod renames a field.

    It accepts a numeric default as well as a string, and it has to. Until 2026-08-25 the first
    undeclared string-valued `VariableConfig` field was `range_policy`; the range ruling declared
    it, and a string-only search then found nothing and skipped the test — silently retiring the
    check that the loader refuses an undeclared field. `range_tolerance_fraction` (a float, 0.02)
    keeps it running.
    """
    for f in dataclasses.fields(cls):
        if f.name in declared:
            continue
        if isinstance(f.default, str):
            return f.name, f.default + "__not_the_default"
        if isinstance(f.default, float) and not isinstance(f.default, bool):
            return f.name, repr(f.default + 1.0)
    return None, None


def test_a_manifest_that_sets_an_undeclared_channel_field_is_refused_by_name(tmp_path):
    """The loader's own catch, which can name the file and the key as well as the field."""
    field, value = _an_undeclared_upstream_field(VariableConfig, DECLARED_VARIABLE_FIELDS)
    if field is None:
        pytest.skip("argopod carries no undeclared scalar VariableConfig field today")
    text = MANIFEST_PATH.read_text(encoding="utf-8").replace(
        "    name: AOU\n", f"    name: AOU\n    {field}: {value}\n", 1)
    broken = _manifest_copy(tmp_path, text)
    with pytest.raises(ValueError, match=field):
        load_manifest(broken)
    with pytest.raises(ValueError, match="DECLARED_VARIABLE_FIELDS"):
        load_manifest(broken)


def test_a_manifest_that_sets_an_undeclared_detector_knob_is_refused_by_name(tmp_path):
    field, value = _an_undeclared_upstream_field(DetectionParams, DECLARED_PARAM_FIELDS)
    if field is None:
        pytest.skip("argopod carries no undeclared scalar DetectionParams field today")
    text = MANIFEST_PATH.read_text(encoding="utf-8").replace(
        "params:\n  extremum_ratio_threshold: 0.0\n",
        f"params:\n  extremum_ratio_threshold: 0.0\n  {field}: {value}\n", 1)
    broken = _manifest_copy(tmp_path, text)
    with pytest.raises(ValueError, match=field):
        load_manifest(broken)
    with pytest.raises(ValueError, match="DECLARED_PARAM_FIELDS"):
        load_manifest(broken)
    # a key argopod does not know at all still fails the way it always did
    text = MANIFEST_PATH.read_text(encoding="utf-8").replace(
        "params:\n  extremum_ratio_threshold: 0.0\n",
        "params:\n  extremum_ratio_threshold: 0.0\n  no_such_knob: 3\n", 1)
    with pytest.raises(ValueError, match="unknown params key"):
        load_manifest(_manifest_copy(tmp_path / "second", text))


def test_a_declared_field_that_vanished_upstream_is_refused_not_skipped(study):
    """The mirror hazard. If the list names a field argopod has removed, quietly skipping it
    would shrink the digest and move every id with no visible edit anywhere."""
    pool = study.pool("carbon", "obduction")
    from eddy_pump.domain import declared_content
    with pytest.raises(ValueError, match="no longer fields of VariableConfig"):
        declared_content(pool.spec.variables[0],
                         DECLARED_VARIABLE_FIELDS + ("a_field_argopod_removed",), "a channel")


def test_the_manifests_own_pins_and_the_computed_ids_are_the_same_six(study):
    """The file and the code must agree, and the literals above must agree with both.

    This used to assert that the six pins had NOT moved, which was the acceptance condition of the
    declared-field fix. It is now the weaker, permanent statement — the file, the loader and this
    test name one set of six — because the pins DID move on 2026-08-25, for a written-down reason.
    The "did they move, and did all six" question has its own test above.
    """
    import yaml
    raw = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    pinned = {f"{STUDY_ID}/{e['channel']}/{e['direction']}": e["spec_id"] for e in raw["pools"]}
    assert pinned == EXPECTED_SPEC_IDS
    assert {p.pool_id: p.spec_id for p in study.pools} == EXPECTED_SPEC_IDS
    assert study.pool("nitrate", "obduction").spec_id == EXPECTED_SPEC_IDS[
        "net_carbon_v1/nitrate/obduction"]
