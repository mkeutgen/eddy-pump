"""One study: the cache it is bound to, where it may write, its pools and its declared fleet.

Entry points: `Study` (pool lookup, excluded floats, cache-build ranges, `cache_policy`,
`detector_configs`), `CacheIdentity`, `CacheBuild`, `OutputRootPolicy`, `DetectorConfig` and
`ExcludedFloat`.
"""
from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from argopod import DetectionParams, VariableConfig

from .spec import CandidatePool
from .vocabulary import Direction, Tracer, channel_of

__all__ = [
    "CacheIdentity",
    "CacheBuild",
    "OutputRootPolicy",
    "DetectorConfig",
    "ExcludedFloat",
    "Study",
]


@dataclass(frozen=True)
class CacheIdentity:
    """WHICH residual cache a study's frozen keys were built from.

    Read from a `CACHE_IDENTITY.json` written by the legacy adapter
    (`obduction.detect_all.write_cache_binding`). The fingerprint is the count of
    `*_fine.parquet` grids and the sha256 of their sorted names.
    """

    path: Path
    fine_grids: int
    fine_grids_sha256: str
    bound: str | None = None
    argopod: str | None = None
    line: str | None = None
    source: Path | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> CacheIdentity:
        path = Path(path)
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        cache = raw.get("cache")
        if not isinstance(cache, dict):
            raise ValueError(f"{path}: no 'cache' block — this is not a CACHE_IDENTITY.json")
        missing = [k for k in ("path", "fine_grids", "fine_grids_sha256") if k not in cache]
        if missing:
            raise ValueError(f"{path}: cache block is missing {missing}")
        return cls(
            path=Path(str(cache["path"])),
            fine_grids=int(cache["fine_grids"]),
            fine_grids_sha256=str(cache["fine_grids_sha256"]),
            bound=raw.get("bound"),
            argopod=raw.get("argopod"),
            line=raw.get("line"),
            source=path,
        )

    def matches(self, other: CacheIdentity | dict) -> bool:
        """True when the two fingerprints describe the same cache CONTENT, path ignored."""
        if isinstance(other, CacheIdentity):
            other = {"fine_grids": other.fine_grids,
                     "fine_grids_sha256": other.fine_grids_sha256}
        return (self.fine_grids, self.fine_grids_sha256) == (
            int(other["fine_grids"]), str(other["fine_grids_sha256"]))


@dataclass(frozen=True)
class CacheBuild:
    """HOW the fleet cache is built — the half of the recipe that is a science choice.

    Read from the `cache:` block of `config/events.yaml`. The other half is already written down
    elsewhere in the same file — the plausible ranges (`raw_inputs:` and each channel's
    `prefilter:`), the floats left out (`excluded_floats:`) and the backscatter smoother — and
    :meth:`Study.cache_policy` joins the two, so nothing is declared twice.

    Fields
    ------
    labels
        Grid flavour -> the channels that grid carries, in order. A float earns the richest
        flavour its data fits, and the flavour names its two files.
    window
        `(since, until)` ISO dates. A profile outside them is dropped; both ends are kept.
    fill_policy
        `"mask"` turns a placeholder value into "no reading" before anything is derived from it.
    adjusted_fallback
        What a cycle with no delayed-mode column falls back to.
    residual_ceilings
        Channel -> the largest scaled residual `make verify-cache` accepts. `None` means the
        channel is reported but not checked.
    """

    labels: Mapping[str, tuple[str, ...]]
    window: tuple[str | None, str | None] = (None, None)
    fill_policy: str = "mask"
    adjusted_fallback: str = "cycle"
    residual_ceilings: Mapping[str, float | None] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        labels = {str(k): tuple(str(c) for c in v) for k, v in dict(self.labels).items()}
        if not labels:
            raise ValueError("the cache block declares no grid flavours")
        for name, channels in labels.items():
            if not channels:
                raise ValueError(f"cache flavour {name!r} declares no channels")
        object.__setattr__(self, "labels", labels)
        since, until = tuple(self.window)
        object.__setattr__(self, "window", (
            None if since is None else str(since), None if until is None else str(until)))
        object.__setattr__(self, "residual_ceilings", {
            str(k): (None if v is None else float(v))
            for k, v in dict(self.residual_ceilings).items()})

    @property
    def channels(self) -> tuple[str, ...]:
        """Every channel any flavour carries, in declaration order, each once."""
        seen: dict[str, None] = {}
        for channels in self.labels.values():
            for name in channels:
                seen.setdefault(name, None)
        return tuple(seen)


@dataclass(frozen=True)
class OutputRootPolicy:
    """Where a study may write, and — louder — where it may not.

    :meth:`resolve` is the ONLY sanctioned way to name an output path. It refuses anything that
    escapes :attr:`root` (a ``..`` in the parts) and anything that lands inside a forbidden root.
    """

    root: Path
    forbidden_roots: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(self, "forbidden_roots",
                           tuple(Path(p) for p in self.forbidden_roots))

    def resolve(self, *parts: str) -> Path:
        """An absolute path under :attr:`root`, or ``ValueError`` naming the rule it broke."""
        root = self.root.resolve()
        out = root.joinpath(*parts).resolve()
        if out != root and not out.is_relative_to(root):
            raise ValueError(
                f"{out} escapes the study's output root {root} — an active output may not be "
                f"written outside it")
        for forbidden in self.forbidden_roots:
            f = forbidden.resolve()
            if out == f or out.is_relative_to(f):
                raise ValueError(
                    f"{out} is inside {f}, which is frozen legacy output — the active study "
                    f"never writes there")
        return out


@dataclass(frozen=True)
class DetectorConfig:
    """One pool's detector, in the two argopod types and nothing else.

    This is what `argopod.detect` consumes. Handed out by :meth:`Study.detector_configs` so a
    caller can run the six detectors without importing this module's vocabulary at all.
    """

    pool_id: str
    event_type: str
    spec_id: str
    variables: tuple[VariableConfig, ...]
    params: DetectionParams


@dataclass(frozen=True)
class ExcludedFloat:
    """One float the study has declared out of scope, with the reason and the date it was ruled.

    A declared absence, not a filter: `config/events.yaml` holds the records and nothing else may
    exclude a float. The reasoning: docs/DECISIONS.md

    Fields
    ------
    wmo
        The float's WMO identifier.
    ruled / ruled_by
        When the exclusion was decided and by whom. Both required.
    reason
        Why this float is out, in prose, aimed at a reviewer rather than at a maintainer.
    evidence
        What was measured, and where it can be re-measured. Optional but expected.
    """

    wmo: int
    ruled: str
    ruled_by: str
    reason: str
    evidence: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "wmo", int(self.wmo))
        for field_name in ("ruled", "ruled_by", "reason"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(
                    f"excluded float {self.wmo}: {field_name!r} is empty. An exclusion without a "
                    f"date, an author and a reason is a filter wearing a data file's clothes.")

    def as_record(self) -> dict[str, object]:
        """A flat mapping, for `PROVENANCE.json` and for a `MANIFEST.csv` row."""
        return {"WMO": self.wmo, "ruled": self.ruled, "ruled_by": self.ruled_by,
                "reason": " ".join(self.reason.split()),
                "evidence": " ".join(self.evidence.split())}


@dataclass(frozen=True)
class Study:
    """One study: an id, the cache identity it is bound to, its output-root policy, its pools."""

    study_id: str
    cache: CacheIdentity
    output: OutputRootPolicy
    pools: tuple[CandidatePool, ...]
    spec_version: str
    params: DetectionParams
    manifest_path: Path | None = None
    #: Plausible ranges on RAW columns that are not detection channels of any pool — today
    #: `DOXY_ADJUSTED`. NOT spec content and in no `spec_id`: they are pinned by the cache
    #: identity instead. See `config/events.yaml`, the `raw_inputs:` block.
    input_ranges: tuple[VariableConfig, ...] = ()
    #: Floats declared out of the study, with a reason and a date each. NOT spec content and in
    #: no `spec_id`, for the same reason as :attr:`input_ranges`. See `config/events.yaml`, the
    #: `excluded_floats:` block.
    excluded_floats: tuple[ExcludedFloat, ...] = ()
    #: How the fleet cache is built. See `config/events.yaml`, the `cache:` block. Always present
    #: on a study that came from the manifest — the loader refuses a file without the block.
    cache_build: CacheBuild | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "pools", tuple(self.pools))
        object.__setattr__(self, "input_ranges", tuple(self.input_ranges))
        object.__setattr__(self, "excluded_floats", tuple(self.excluded_floats))
        wmos = [e.wmo for e in self.excluded_floats]
        if len(set(wmos)) != len(wmos):
            raise ValueError(
                f"a float is excluded twice: {sorted(w for w in set(wmos) if wmos.count(w) > 1)}. "
                f"Two records for one float means two reasons, and no rule for which is the one "
                f"that was ruled.")
        ids = [p.pool_id for p in self.pools]
        if len(set(ids)) != len(ids):
            raise ValueError(f"two pools share a pool_id: {sorted(ids)}")
        foreign = [i for p, i in zip(self.pools, ids) if p.study_id != self.study_id]
        if foreign:
            raise ValueError(f"pools from another study: {foreign}")

    # --- lookup ------------------------------------------------------------ #
    @property
    def pool_ids(self) -> tuple[str, ...]:
        return tuple(p.pool_id for p in self.pools)

    def __getitem__(self, pool_id: str) -> CandidatePool:
        for p in self.pools:
            if p.pool_id == pool_id:
                return p
        raise KeyError(f"{pool_id!r} is not a pool of {self.study_id}: {list(self.pool_ids)}")

    def pool(self, channel: str | Tracer | None, direction: str | Direction) -> CandidatePool:
        """The pool for one (channel, direction), by the tokens a human would type."""
        channel = channel_of(channel) if isinstance(channel, (Tracer, type(None))) else channel
        return self[f"{self.study_id}/{channel}/{Direction(direction).value}"]

    @property
    def parents(self) -> tuple[CandidatePool, ...]:
        """The primary human-review frames: the pools with no tracer term."""
        return tuple(p for p in self.pools if p.tracer is None)

    @property
    def children(self) -> tuple[CandidatePool, ...]:
        """The nested tracer subsets."""
        return tuple(p for p in self.pools if p.tracer is not None)

    # --- the declared fleet ------------------------------------------------ #
    @property
    def excluded_wmos(self) -> frozenset[int]:
        """The WMOs the study has declared out of scope. Empty when nothing is excluded."""
        return frozenset(e.wmo for e in self.excluded_floats)

    def exclusion(self, wmo: int) -> ExcludedFloat | None:
        """The record for one WMO, so a caller can print the REASON and not just the number."""
        for e in self.excluded_floats:
            if e.wmo == int(wmo):
                return e
        return None

    def exclusion_records(self) -> list[dict[str, object]]:
        """Every exclusion as a flat mapping — what provenance and a manifest row are built from."""
        return [e.as_record() for e in self.excluded_floats]

    # --- the cache-build ranges -------------------------------------------- #
    def channel_ranges(self, *, surgical_only: bool = True) -> tuple[VariableConfig, ...]:
        """Every DETECTION channel of every pool that declares a `valid_range`, de-duplicated.

        Each entry is a RANGE CARRIER, not the spec's channel: it keeps the name and the four
        range fields and drops `cutoff`, `sign_constraint`, `require_gradient_check` and
        `pre_median_filter`. A channel that appears in more than one pool must declare the same
        range in all of them or this raises.

        `surgical_only` (the default) keeps only the policies that act on a SAMPLE — `"clip"` and
        `"mask"` — and drops `"drop_cycle"`, which deletes an entire dive. Pass
        ``surgical_only=False`` to get the dropping policies back.
        The reasoning: docs/DECISIONS.md
        """
        out: dict[str, VariableConfig] = {}
        for pool in self.pools:
            for v in pool.spec.variables:
                if v.valid_range is None:
                    continue
                if surgical_only and v.range_policy == "drop_cycle":
                    continue
                carrier = VariableConfig(
                    name=v.name, valid_range=v.valid_range, range_policy=v.range_policy,
                    range_tolerance=v.range_tolerance,
                    range_tolerance_fraction=v.range_tolerance_fraction,
                    artifact_cycle_fraction=v.artifact_cycle_fraction)
                seen = out.get(v.name)
                if seen is not None and seen != carrier:
                    raise ValueError(
                        f"{self.study_id}: channel {v.name!r} declares two different ranges "
                        f"across the pools ({seen} vs {carrier}). One cache serves all six "
                        f"pools, so a channel has one range or the manifest cannot be built.")
                out[v.name] = carrier
        return tuple(out[k] for k in sorted(out))

    def cache_build_ranges(self) -> dict[str, tuple[VariableConfig, ...]]:
        """The two lists a cache build needs, keyed by WHEN they run.

        ``"raw"``      applied to the raw frame, BEFORE `compute_derived_variables`.
        ``"derived"``  applied after derivation and before `downscale`, on the detection channels.
        """
        return {"raw": self.input_ranges, "derived": self.channel_ranges()}

    def cache_policy(self):
        """The whole fleet-cache recipe, as the `argopod.cache.CachePolicy` a build consumes.

        Four things are joined here, each from the one place it is written down: the grid
        flavours, the dates, the placeholder rule and the check ceilings from the `cache:` block;
        the plausible ranges from `raw_inputs:` and the channels' own `prefilter:` blocks; the
        floats left out from `excluded_floats:`; the backscatter smoother from the channel that
        asks for it.

        THE GRID KNOBS ARE ARGOPOD'S DEFAULTS, not :attr:`params`. The frozen cache was built
        under the defaults, and the one knob this study moves — the local-extremum test — acts at
        detection time and never touches a grid. Passing the study's block instead would be
        harmless today and silent damage the day a detection knob starts mattering to a bin.
        """
        # Local import: nothing but a cache build needs these, so `import eddy_pump` stays cheap.
        from argopod.cache import CachePolicy, Exclusion

        if self.cache_build is None:
            raise ValueError(
                f"{self.study_id} carries no cache block, so there is no policy to build under. "
                f"It comes from config/events.yaml `cache:`; load the study with load_manifest().")
        build = self.cache_build
        ranges = self.cache_build_ranges()
        # The one smoother, taken from the channel that declares it rather than re-typed: a
        # second author for it is how the file and the build drift apart.
        prefilters = tuple(
            VariableConfig(name, pre_median_filter=True)
            for name in sorted({v.name for p in self.pools for v in p.spec.variables
                                if v.pre_median_filter}))
        return CachePolicy(
            labels=dict(build.labels),
            params=dataclasses.replace(DetectionParams(), fill_policy=build.fill_policy),
            window=build.window,
            raw_ranges=ranges["raw"],
            derived_ranges=ranges["derived"],
            prefilters=prefilters,
            adjusted_fallback=build.adjusted_fallback,
            exclusions=tuple(
                Exclusion(wmo=r["WMO"], ruled=r["ruled"], ruled_by=r["ruled_by"],
                          reason=r["reason"], evidence=r["evidence"])
                for r in self.exclusion_records()),
            residual_ceilings=dict(build.residual_ceilings),
        )

    # --- the detectors ----------------------------------------------------- #
    def detector_configs(self) -> dict[str, DetectorConfig]:
        """``{pool_id: DetectorConfig}`` — the study's detectors, one per pool."""
        return {
            p.pool_id: DetectorConfig(
                pool_id=p.pool_id,
                event_type=p.event_type,
                spec_id=p.spec_id,
                variables=p.spec.variables,
                params=p.spec.params,
            )
            for p in self.pools
        }
