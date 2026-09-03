"""Detection specs, candidate pools and candidate keys: what a spec_id and a candidate_id are.

Entry points: `EventSpec` (a detector plus its content-derived `spec_id`), `CandidatePool`
(study, channel, direction, and its parent), `CandidateKey` (one depth level, with its
`candidate_id`), and `declared_content` / `undeclared_settings`, which fix which argopod
fields the spec digest covers.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass

from argopod import DetectionParams, VariableConfig

from .vocabulary import CHANNELS, Direction, Tracer, channel_of

__all__ = [
    "DECLARED_VARIABLE_FIELDS",
    "DECLARED_PARAM_FIELDS",
    "declared_content",
    "undeclared_settings",
    "EventSpec",
    "CandidatePool",
    "CandidateKey",
    "PRES_DECIMALS",
    "CANDIDATE_ID_HEX",
]


# --------------------------------------------------------------------------- #
# what `spec_id` covers — the declared field list, owned HERE
# --------------------------------------------------------------------------- #
# The digest covers an explicit list of field names, owned by this repository rather than taken
# from whatever argopod happens to carry. Adding a name below is a deliberate re-pin: it moves
# the spec_id of every spec the field appears on. A value on a field NOT listed below is refused
# by name, at EventSpec construction and again in eddy_pump.manifest.
# The reasoning: docs/DECISIONS.md

#: The `argopod.VariableConfig` fields that ARE this study's scientific content: the channel's
#: name, its detect-time gate (cutoff, sign constraint, gradient check, second-derivative
#: threshold), its build-time prefilter terms (`pre_median_filter`, `valid_range`,
#: `range_policy`, `range_tolerance`), and `depth_scale`.
#: `range_tolerance_fraction` and `artifact_cycle_fraction` are deliberately absent.
DECLARED_VARIABLE_FIELDS: tuple[str, ...] = (
    "name",
    "cutoff",
    "sign_constraint",
    "require_gradient_check",
    "second_deriv_threshold",
    "pre_median_filter",
    "valid_range",
    "range_policy",
    "range_tolerance",
    "depth_scale",
)

#: The `argopod.DetectionParams` fields that ARE this study's detector — all twenty-two, so the
#: whole detector setting is pinned and not merely the knobs this study moves off the default.
DECLARED_PARAM_FIELDS: tuple[str, ...] = (
    "bin_width_fine",
    "bin_width_coarse",
    "pres_min",
    "pres_min_col",
    "pres_max",
    "gradient_window",
    "extremum_half_width",
    "extremum_shoulder_dbar",
    "extremum_ratio_threshold",
    "extremum_require_both_shoulders",
    "detect_grid",
    "scale_window",
    "scale_min_periods",
    "scale_ref_range",
    "trimmed_mean_window",
    "trimmed_mean_min_periods",
    "trimmed_mean_lo",
    "trimmed_mean_hi",
    "pressure_col",
    "cycle_col",
    "group_col",
    "meta_cols",
)

#: Returned by :func:`_upstream_default` for a field with no default at all.
_NO_DEFAULT = object()


def _upstream_default(field: dataclasses.Field) -> object:
    """The value an upstream field takes when nobody sets it, or :data:`_NO_DEFAULT`."""
    if field.default is not dataclasses.MISSING:
        return field.default
    if field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        return field.default_factory()  # type: ignore[misc]
    return _NO_DEFAULT


def declared_content(obj, declared: tuple[str, ...], what: str) -> dict:
    """The JSON-able rendering of `obj` over `declared`, and ONLY over `declared`.

    A name in `declared` that the upstream dataclass no longer carries raises rather than being
    skipped. Fields are read one at a time with `getattr`, never filtered out of
    `dataclasses.asdict(obj)`, so an unrenderable undeclared field cannot break the digest.
    """
    present = {f.name for f in dataclasses.fields(obj)}
    missing = [n for n in declared if n not in present]
    if missing:
        raise ValueError(
            f"{what}: the declared field(s) {missing} are no longer fields of "
            f"{type(obj).__name__}. The list this repository digests names a field upstream has "
            f"removed or renamed; every spec_id would move silently. Adjudicate the removal and "
            f"edit the list — and the pins — deliberately.")
    return {name: getattr(obj, name) for name in declared}


def undeclared_settings(obj, declared: tuple[str, ...]) -> dict[str, object]:
    """``{field: value}`` for upstream fields OUTSIDE `declared` that carry a non-default value.

    A field with no default at all is always reported, because "unset" is not a state it has.
    """
    out: dict[str, object] = {}
    for f in dataclasses.fields(obj):
        if f.name in declared:
            continue
        default = _upstream_default(f)
        value = getattr(obj, f.name)
        if default is _NO_DEFAULT or value != default:
            out[f.name] = value
    return out


def _refuse_undeclared(obj, declared: tuple[str, ...], constant: str, what: str) -> None:
    """Raise, naming every undeclared field that has been given a value."""
    bad = undeclared_settings(obj, declared)
    if not bad:
        return
    shown = ", ".join(f"{k}={v!r}" for k, v in sorted(bad.items()))
    raise ValueError(
        f"{what} sets {sorted(bad)}, which eddy_pump.domain.{constant} does not declare, so "
        f"spec_id does not cover it ({shown}). A value the spec pin cannot see is a provenance "
        f"hole: two specs that detect differently would hash the same. Either add the field to "
        f"{constant} — a DELIBERATE act that re-pins every spec_id it appears on — or leave it "
        f"at argopod's default.")


# --------------------------------------------------------------------------- #
# the detection spec
# --------------------------------------------------------------------------- #
def _canonical_json(payload: object) -> str:
    """The one rendering every digest in this module is taken over.

    ``sort_keys`` removes dict iteration order from the digest; the compact separators remove
    whitespace; ``ensure_ascii`` removes the encoding.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True)
class EventSpec:
    """One detector: the variables with their cutoffs, signs, gradient checks and ranges,
    plus the global detector knobs, plus the content-derived `spec_id` that pins both.

    ``variables`` and ``params`` are argopod's own types (:class:`argopod.VariableConfig`,
    :class:`argopod.DetectionParams`), so a spec loaded from the manifest can be handed straight
    to ``argopod.detect`` with nothing in between. `spec_id` is a function of the detector
    content alone: two pools declaring the same detector get the same `spec_id`.
    """

    name: str
    direction: Direction
    tracer: Tracer | None
    variables: tuple[VariableConfig, ...]
    params: DetectionParams
    version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "variables", tuple(self.variables))
        if not self.variables:
            raise ValueError(f"{self.name}: a spec with no variables detects nothing")
        names = [v.name for v in self.variables]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.name}: a channel is declared twice: {names}")
        if not self.version:
            raise ValueError(f"{self.name}: spec_id must be versioned; got an empty version")
        # a spec may not set a field the digest cannot see — checked here as well as in the
        # loader, so a spec built in Python is held to the same contract as the manifest
        for v in self.variables:
            _refuse_undeclared(v, DECLARED_VARIABLE_FIELDS, "DECLARED_VARIABLE_FIELDS",
                               f"{self.name}: channel {v.name!r}")
        _refuse_undeclared(self.params, DECLARED_PARAM_FIELDS, "DECLARED_PARAM_FIELDS",
                           f"{self.name}: the detector params block")

    # --- the channels ------------------------------------------------------ #
    @property
    def channels(self) -> tuple[str, ...]:
        """The channel names, in declaration order."""
        return tuple(v.name for v in self.variables)

    @property
    def declared_order(self) -> tuple[str, ...]:
        """The order the manifest declares the channels in — PRESENTATION, never identity.

        It travels into `PROVENANCE.json` and the CLI's triage feature list, and is deliberately
        excluded from :attr:`content_digest`. The reasoning: docs/DECISIONS.md
        """
        return self.channels

    @property
    def prefilter_terms(self) -> dict[str, dict]:
        """The CACHE-BUILD-TIME terms, per channel — the range, its policy, its tolerance, and
        the median smoother.

        `range_policy` is reported ONLY beside a `valid_range`, because without a range it is
        argopod's default and decides nothing. `range_tolerance` is reported only when the policy
        can actually consult it — ``"mask"`` declares no near-boundary band at all.
        """
        out = {}
        for v in self.variables:
            terms = {}
            if v.valid_range is not None:
                terms["valid_range"] = list(v.valid_range)
                terms["range_policy"] = v.range_policy
                if v.range_policy != "mask" and v.range_tolerance is not None:
                    terms["range_tolerance"] = float(v.range_tolerance)
            if v.pre_median_filter:
                terms["pre_median_filter"] = True
            if terms:
                out[v.name] = terms
        return out

    def channel(self, name: str) -> VariableConfig:
        """This spec's entry for one channel. Raises rather than returning ``None``."""
        for v in self.variables:
            if v.name == name:
                return v
        raise KeyError(f"{self.name} has no channel {name!r}; it has {list(self.channels)}")

    # --- identity ---------------------------------------------------------- #
    @property
    def content(self) -> dict:
        """The JSON-able rendering the digest is taken over. SEMANTIC content only.

        The channels are sorted by name and rendered over :data:`DECLARED_VARIABLE_FIELDS`, the
        detector over :data:`DECLARED_PARAM_FIELDS`. `name`, `direction` and `tracer` are
        deliberately excluded. The reasoning: docs/DECISIONS.md
        """
        return {
            "channels": [declared_content(v, DECLARED_VARIABLE_FIELDS,
                                          f"{self.name}: channel {v.name!r}")
                         for v in sorted(self.variables, key=lambda v: v.name)],
            "params": declared_content(self.params, DECLARED_PARAM_FIELDS,
                                       f"{self.name}: detector params"),
        }

    @property
    def content_digest(self) -> str:
        """Full sha256 of :attr:`content`. Stable across processes, machines and Pythons."""
        return hashlib.sha256(_canonical_json(self.content).encode("utf-8")).hexdigest()

    @property
    def spec_id(self) -> str:
        """``{version}:{16 hex}`` — the versioned, content-derived spec pin."""
        return f"{self.version}:{self.content_digest[:16]}"


# --------------------------------------------------------------------------- #
# the pools
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CandidatePool:
    """(study, channel, direction) — the frame a rate is estimated against.

    The two physical pools are the primary human-review frames and have no parent. The four
    tracer pools are nested subsets of their OWN directional physical parent — `nitrate` under
    `subduction` nests in `physical/subduction`, never in `physical/obduction`, and a cross-limb
    parent is refused at construction.
    """

    study_id: str
    direction: Direction
    tracer: Tracer | None
    spec: EventSpec
    parent: CandidatePool | None = None

    def __post_init__(self) -> None:
        if self.spec.direction != self.direction:
            raise ValueError(
                f"{self.pool_id}: spec {self.spec.name!r} declares direction "
                f"{self.spec.direction.value!r}")
        if self.spec.tracer != self.tracer:
            raise ValueError(
                f"{self.pool_id}: spec {self.spec.name!r} declares tracer {self.spec.tracer!r}")
        if self.tracer is None:
            if self.parent is not None:
                raise ValueError(
                    f"{self.pool_id}: a physical pool is a primary frame and has no parent")
        else:
            if self.parent is None:
                raise ValueError(
                    f"{self.pool_id}: a tracer pool is a nested subset and must name its parent")
            if self.parent.tracer is not None:
                raise ValueError(
                    f"{self.pool_id}: parent {self.parent.pool_id} is not a physical pool")
            if self.parent.direction != self.direction:
                raise ValueError(
                    f"{self.pool_id}: parent {self.parent.pool_id} is on the other limb — a "
                    f"tracer pool nests in its OWN directional parent")
            if self.parent.study_id != self.study_id:
                raise ValueError(
                    f"{self.pool_id}: parent {self.parent.pool_id} belongs to another study")

    # --- identity ---------------------------------------------------------- #
    @property
    def channel(self) -> str:
        return channel_of(self.tracer)

    @property
    def pool_id(self) -> str:
        """``{study_id}/{channel}/{direction}`` — the pool's stable, readable identity."""
        return f"{self.study_id}/{self.channel}/{self.direction.value}"

    @property
    def event_type(self) -> str:
        """The scientific proposal class, and ONLY that: ``{channel}_{direction}``."""
        return f"{self.channel}_{self.direction.value}"

    @property
    def spec_id(self) -> str:
        return self.spec.spec_id

    # --- relations --------------------------------------------------------- #
    @property
    def ancestry(self) -> tuple[CandidatePool, ...]:
        """This pool then its parents, outermost last. One level deep today."""
        out, node = [], self
        while node is not None:
            out.append(node)
            node = node.parent
        return tuple(out)

    def key(self, wmo: int, cycle_number: int, pres_adjusted: float) -> CandidateKey:
        """A :class:`CandidateKey` in this pool. The only construction that cannot mistype an id."""
        return CandidateKey(self.pool_id, wmo, cycle_number, pres_adjusted)

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.pool_id


# --------------------------------------------------------------------------- #
# the candidate key
# --------------------------------------------------------------------------- #
#: Decimals `PRES_ADJUSTED` is rounded to before it enters an id. Zero, matching the legacy dedup
#: rule (`obduction.config.HIERARCHY_KEY_PRES_DECIMALS`).
PRES_DECIMALS = 0

#: Hex characters of the candidate sha256 kept. 32 is 128 bits.
CANDIDATE_ID_HEX = 32


@dataclass(frozen=True)
class CandidateKey:
    """One depth level on one profile, in one pool.

    The four readable fields are ``WMO``, ``CYCLE_NUMBER``, :attr:`pres_rounded` and
    :attr:`event_type`; the machine identity is :attr:`candidate_id`.

    ``EVENT_TYPE`` is DERIVED from `pool_id`, never stored, so it always means the scientific
    proposal class. A `pool_id` whose channel token is not one of :data:`CHANNELS` is refused at
    construction. The reasoning: docs/DECISIONS.md
    """

    pool_id: str
    wmo: int
    cycle_number: int
    pres_adjusted: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "pool_id", str(self.pool_id))
        object.__setattr__(self, "wmo", int(self.wmo))
        object.__setattr__(self, "cycle_number", int(self.cycle_number))
        object.__setattr__(self, "pres_adjusted", float(self.pres_adjusted))
        self._parts()  # validate the pool_id grammar now, not at first id request

    def _parts(self) -> tuple[str, str, str]:
        parts = self.pool_id.split("/")
        if len(parts) != 3 or not all(parts):
            raise ValueError(
                f"pool_id {self.pool_id!r} is not "
                f"'{{study_id}}/{{physical|nitrate|carbon}}/{{subduction|obduction}}'")
        study_id, channel, direction = parts
        if channel not in CHANNELS:
            raise ValueError(
                f"pool_id {self.pool_id!r} has channel {channel!r}; the channel token is one of "
                f"{list(CHANNELS)}. A token like 'physical_obduction_paper' is the product line "
                f"leaking back into the key — the line belongs to the study, not to the channel.")
        if direction not in tuple(d.value for d in Direction):
            raise ValueError(
                f"pool_id {self.pool_id!r} has direction {direction!r}; expected one of "
                f"{[d.value for d in Direction]}")
        return study_id, channel, direction

    # --- the readable fields ----------------------------------------------- #
    @property
    def study_id(self) -> str:
        return self._parts()[0]

    @property
    def channel(self) -> str:
        return self._parts()[1]

    @property
    def direction(self) -> Direction:
        return Direction(self._parts()[2])

    @property
    def pres_rounded(self) -> int:
        """``round(PRES_ADJUSTED)``, half-to-even — the same rule the legacy dedup key uses."""
        return int(round(self.pres_adjusted, PRES_DECIMALS))

    @property
    def event_type(self) -> str:
        """``{channel}_{direction}``. Always in :data:`CANONICAL_EVENT_TYPES`, by construction."""
        _, channel, direction = self._parts()
        return f"{channel}_{direction}"

    # --- the machine identity ---------------------------------------------- #
    @property
    def id_payload(self) -> str:
        """The exact string the digest is taken over. Spelled out so a future reader can
        reproduce an id by hand: ``pool_id|WMO|CYCLE_NUMBER|round(PRES_ADJUSTED)``."""
        return f"{self.pool_id}|{self.wmo:d}|{self.cycle_number:d}|{self.pres_rounded:d}"

    @property
    def candidate_id(self) -> str:
        """sha256 of :attr:`id_payload`, first :data:`CANDIDATE_ID_HEX` hex digits.

        `hashlib`, never the builtin ``hash()``, which is salted per process.
        """
        digest = hashlib.sha256(self.id_payload.encode("utf-8")).hexdigest()
        return digest[:CANDIDATE_ID_HEX]

    def as_row(self) -> dict:
        """The identity columns an active output carries for this level."""
        return {
            "candidate_id": self.candidate_id,
            "pool_id": self.pool_id,
            "WMO": self.wmo,
            "CYCLE_NUMBER": self.cycle_number,
            "PRES_ADJUSTED": self.pres_adjusted,
            "EVENT_TYPE": self.event_type,
        }
