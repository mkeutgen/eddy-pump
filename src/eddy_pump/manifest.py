"""Load `config/events.yaml`, the one canonical event-spec source, into domain objects.

reads  config/events.yaml, and the CACHE_IDENTITY.json that manifest names
writes nothing

Entry point: `load_manifest()`, aliased `load_study()`, returning a Study with its pools. It
refuses a manifest whose declared spec_id disagrees with the digest of its own content, whose
per-limb AOU sign disagrees with the code, or which sets a field the spec digest cannot see.
"""
from __future__ import annotations

import dataclasses
import typing
import os
from pathlib import Path

from argopod import DetectionParams, VariableConfig

from .spec import (
    DECLARED_PARAM_FIELDS,
    DECLARED_VARIABLE_FIELDS,
    CandidatePool,
    EventSpec,
)
from .study import CacheIdentity, ExcludedFloat, OutputRootPolicy, Study
from .vocabulary import CHANNELS, PHYSICAL, Direction, Tracer

__all__ = ["REPO_ROOT", "MANIFEST_PATH", "GLOBARGO_DATA", "load_manifest", "load_study"]

#: `src/eddy_pump/manifest.py` -> parents[2] is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: The canonical manifest. One file, at a fixed place.
MANIFEST_PATH = REPO_ROOT / "config" / "events.yaml"

#: The earlier subduction study's tables (Keutgen De Greef, Resplandy & Poupon 2026; the
#: `mkeutgen/globargo` repository). One default, overridable with $GLOBARGO_DATA; every study script
#: that reads those tables imports this instead of typing the path.
GLOBARGO_DATA = Path(os.environ.get("GLOBARGO_DATA", "~/Documents/globargo/data")).expanduser()


# --------------------------------------------------------------------------- #
# small parsers
# --------------------------------------------------------------------------- #
#: The `VariableConfig` fields consumed by `argopod.detect.prefilter.apply_prefilters`, which runs
#: at CACHE-BUILD time. The manifest keeps them in their own `prefilter:` block rather than mixed
#: in with the detect-time gate. The reasoning: docs/DECISIONS.md
PREFILTER_FIELDS: tuple[str, ...] = (
    "valid_range", "pre_median_filter", "range_policy", "range_tolerance",
)


def _as_variable(item: dict, where: str) -> VariableConfig:
    """One manifest mapping -> :class:`argopod.VariableConfig`.

    Mirrors `argopod.eventconfig._as_variable`: tuple-ify `valid_range` and let an unknown key
    raise. A key argopod knows but `DECLARED_VARIABLE_FIELDS` does not is refused too.

    The build-time terms must arrive under a `prefilter:` sub-mapping and are refused at the
    channel's top level; they are folded back into the one `VariableConfig` here, because that is
    where argopod keeps them.
    """
    if not isinstance(item, dict) or "name" not in item:
        raise ValueError(f"{where}: each entry must be a mapping with a 'name' key, got {item!r}")
    kwargs = dict(item)
    misplaced = sorted(set(kwargs) & set(PREFILTER_FIELDS))
    if misplaced:
        raise ValueError(
            f"{where}: {misplaced} act at CACHE-BUILD time, not at detection time — declare them "
            f"under this channel's 'prefilter:' block so the manifest says when they act")
    prefilter = kwargs.pop("prefilter", None) or {}
    if not isinstance(prefilter, dict):
        raise ValueError(f"{where}: 'prefilter' must be a mapping, got {prefilter!r}")
    unknown = sorted(set(prefilter) - set(PREFILTER_FIELDS))
    if unknown:
        raise ValueError(
            f"{where}: 'prefilter' takes only {list(PREFILTER_FIELDS)} — the fields "
            f"argopod.detect.prefilter.apply_prefilters reads; got {unknown}")
    kwargs.update(prefilter)
    upstream = {f.name for f in dataclasses.fields(VariableConfig)}
    undeclared = sorted((set(kwargs) & upstream) - set(DECLARED_VARIABLE_FIELDS))
    if undeclared:
        raise ValueError(
            f"{where}: {undeclared} are argopod VariableConfig fields that "
            f"eddy_pump.domain.DECLARED_VARIABLE_FIELDS does not declare, so `spec_id` does not "
            f"cover them. Setting one here would build a detector the pin cannot describe — the "
            f"manifest claiming one instrument while the digest certifies another. Add the field "
            f"to DECLARED_VARIABLE_FIELDS (a deliberate act: it re-pins every spec_id the field "
            f"appears on) or leave it at argopod's default and say nothing here.")
    if kwargs.get("valid_range") is not None:
        lo, hi = kwargs["valid_range"]
        kwargs["valid_range"] = (float(lo), float(hi))
    if kwargs.get("depth_scale") is not None:
        kwargs["depth_scale"] = tuple((float(p), float(f)) for p, f in kwargs["depth_scale"])
    try:
        return VariableConfig(**kwargs)
    except TypeError as exc:
        raise ValueError(f"{where}: bad entry {item!r}: {exc}") from exc


def _as_params(block, where: str) -> DetectionParams:
    """The `params:` mapping -> :class:`argopod.DetectionParams`.

    Unknown keys raise. Numbers are coerced against the dataclass's own resolved annotations,
    because YAML is untyped and `DetectionParams` would otherwise hold the string ``"0.0"``.
    """
    if not block:
        return DetectionParams()
    if not isinstance(block, dict):
        raise ValueError(f"{where}: 'params' must be a mapping, got {type(block).__name__}")
    valid = {f.name for f in dataclasses.fields(DetectionParams)}
    unknown = sorted(set(block) - valid)
    if unknown:
        raise ValueError(
            f"{where}: unknown params key(s) {unknown}. Valid keys are the DetectionParams "
            f"fields: {', '.join(sorted(valid))}")
    undeclared = sorted(set(block) - set(DECLARED_PARAM_FIELDS))
    if undeclared:
        raise ValueError(
            f"{where}: {undeclared} are argopod DetectionParams fields that "
            f"eddy_pump.domain.DECLARED_PARAM_FIELDS does not declare, so `spec_id` does not "
            f"cover them. A detector knob the pin cannot see is a provenance hole: the six pools "
            f"would hash identically whichever value it took. Add the field to "
            f"DECLARED_PARAM_FIELDS (a deliberate act: it re-pins all six spec_ids) or leave it "
            f"at argopod's default.")
    hints = typing.get_type_hints(DetectionParams)
    kwargs = {}
    for key, value in block.items():
        hint = hints[key]
        text = str(hint)
        if value is None:
            kwargs[key] = None
        elif isinstance(value, bool):
            kwargs[key] = value
        elif isinstance(value, (int, float)) and "float" in text:
            kwargs[key] = float(value)
        elif isinstance(value, (int, float)) and "int" in text:
            if isinstance(value, float) and not value.is_integer():
                raise ValueError(f"{where}: params {key!r} wants a whole number, got {value!r}")
            kwargs[key] = int(value)
        elif isinstance(value, (list, tuple)):
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    try:
        return DetectionParams(**kwargs)
    except TypeError as exc:
        raise ValueError(f"{where}: bad 'params' block {block!r}: {exc}") from exc


def _as_input_range(item: dict, where: str, spec_channels: set[str]) -> VariableConfig:
    """One `raw_inputs:` entry -> a :class:`argopod.VariableConfig` used ONLY as a range carrier.

    A raw input is not a detection channel: nothing gates on it, no `cutoff` applies to it, and it
    never reaches `spec_id`. Four things are refused — a detect-time key, a missing `valid_range`,
    `range_policy: drop_cycle`, and a name that is a detection channel of some pool.
    The reasoning: docs/DECISIONS.md
    """
    if not isinstance(item, dict) or "name" not in item:
        raise ValueError(f"{where}: each raw_inputs entry must be a mapping with a 'name' key, "
                         f"got {item!r}")
    kwargs = dict(item)
    name = str(kwargs["name"])
    if name in spec_channels:
        raise ValueError(
            f"{where}: raw_inputs declares {name!r}, which is a DETECTION CHANNEL of this study. "
            f"Its range belongs in that channel's own 'prefilter:' block, where `spec_id` covers "
            f"it; declaring it here as well would give one column two authors.")
    allowed = {"name", *PREFILTER_FIELDS}
    unknown = sorted(set(kwargs) - allowed)
    if unknown:
        raise ValueError(
            f"{where}: raw_inputs.{name} sets {unknown}. A raw input is a range carrier, not a "
            f"detector: only {sorted(allowed)} may be set on one.")
    if kwargs.get("valid_range") is None:
        raise ValueError(
            f"{where}: raw_inputs.{name} declares no valid_range, so it declares nothing")
    lo, hi = kwargs["valid_range"]
    kwargs["valid_range"] = (float(lo), float(hi))
    if kwargs.get("range_policy", "drop_cycle") == "drop_cycle":
        raise ValueError(
            f"{where}: raw_inputs.{name} must name range_policy 'clip' or 'mask'. argopod's "
            f"default 'drop_cycle' deletes the WHOLE cycle — every other channel of it — on one "
            f"bad sample, which is the shape ruling N4 calls wrong and is never what a raw-input "
            f"range is for.")
    try:
        return VariableConfig(**kwargs)
    except TypeError as exc:
        raise ValueError(f"{where}: bad raw_inputs entry {item!r}: {exc}") from exc


def _as_excluded_float(item: dict, where: str) -> ExcludedFloat:
    """One `excluded_floats:` entry -> an :class:`~eddy_pump.study.ExcludedFloat`.

    A record that is not a mapping, one with no `wmo`, one missing `ruled`, `ruled_by` or
    `reason`, and one carrying an unknown key are all refused. `evidence` is optional in the type
    and expected in practice. The reasoning: docs/DECISIONS.md
    """
    if not isinstance(item, dict) or "wmo" not in item:
        raise ValueError(f"{where}: each excluded_floats entry must be a mapping with a 'wmo' "
                         f"key, got {item!r}")
    allowed = {"wmo", "ruled", "ruled_by", "reason", "evidence"}
    unknown = sorted(set(item) - allowed)
    if unknown:
        raise ValueError(
            f"{where}: excluded_floats entry for {item['wmo']} sets {unknown}. Only "
            f"{sorted(allowed)} may be set — the record is the whole statement.")
    try:
        wmo = int(item["wmo"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: excluded_floats wmo {item['wmo']!r} is not an integer "
                         f"WMO") from exc
    missing = [k for k in ("ruled", "ruled_by", "reason") if not str(item.get(k, "")).strip()]
    if missing:
        raise ValueError(
            f"{where}: excluded_floats entry for {wmo} declares no {', '.join(missing)}. Dropping "
            f"a float from a study is a scientific claim; it carries a date, an author and a "
            f"reason or it is not declared at all.")
    return ExcludedFloat(wmo=wmo, ruled=str(item["ruled"]), ruled_by=str(item["ruled_by"]),
                         reason=str(item["reason"]), evidence=str(item.get("evidence", "")))


def _resolve(path_str, base: Path) -> Path:
    """A manifest path -> absolute. Relative paths resolve against the REPO ROOT."""
    p = Path(str(path_str)).expanduser()
    return p if p.is_absolute() else (base / p).resolve()


# --------------------------------------------------------------------------- #
# the spec builders — where symmetry and nesting are made structural
# --------------------------------------------------------------------------- #
def _parent_variables(physical: dict, direction: Direction,
                      where: str) -> tuple[VariableConfig, ...]:
    """The physical parent's channels for one limb: the AOU channel at that limb's sign,
    followed by the shared, sign-free channels VERBATIM.

    Called twice with the same `physical` block and two different directions, so the only thing
    that can differ between the two parents is the sign.
    """
    aou = dict(physical["aou"])
    if "sign_constraint" in aou:
        raise ValueError(
            f"{where}: physical.aou must not declare a sign_constraint — the sign is the LIMB's "
            f"("f"directions.<limb>.aou_sign), and a second author for it is how the two parents "
            f"drift apart")
    aou["sign_constraint"] = direction.aou_sign
    out = [_as_variable(aou, f"{where}: physical.aou")]
    out += [_as_variable(v, f"{where}: physical.shared") for v in physical.get("shared", ())]
    return tuple(out)


def _tracer_term(tracers: dict, tracer: Tracer, direction: Direction,
                 where: str) -> VariableConfig:
    """The ONE joint-AND term a child adds to its parent.

    The cutoff is one number for both limbs; the sign is declared per limb, and both limbs must
    be named or this raises.
    """
    block = tracers.get(tracer.value)
    if block is None:
        raise ValueError(f"{where}: no tracer term declared for {tracer.value!r}")
    item = {k: v for k, v in block.items() if k != "sign_constraint"}
    item["name"] = item.pop("channel")
    signs = block.get("sign_constraint")
    if not isinstance(signs, dict) or set(signs) != {d.value for d in Direction}:
        raise ValueError(
            f"{where}: tracers.{tracer.value}.sign_constraint must name BOTH limbs "
            f"({[d.value for d in Direction]}); the sign is the thing that flips")
    item["sign_constraint"] = signs[direction.value]
    return _as_variable(item, f"{where}: tracers.{tracer.value}")


def _child_variables(parent: EventSpec, term: VariableConfig) -> tuple[VariableConfig, ...]:
    """The parent's channels VERBATIM, then `term` appended. There is no other shape."""
    return tuple(parent.variables) + (term,)


# --------------------------------------------------------------------------- #
# the loader
# --------------------------------------------------------------------------- #
def load_manifest(path: str | Path | None = None) -> Study:
    """Parse the canonical manifest into a :class:`~eddy_pump.study.Study`.

    Relative paths in the manifest resolve against the REPO ROOT (the manifest lives in
    `config/`, so its own directory is not the right base for `data/...`).
    """
    import yaml  # pyyaml arrives with argopod[cli]; imported here so `import eddy_pump` is cheap

    path = Path(MANIFEST_PATH if path is None else path).expanduser().resolve()
    where = str(path)
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: expected a YAML mapping at top level")
    root = path.parent.parent

    # --- the study -------------------------------------------------------- #
    study_block = raw.get("study") or {}
    for required in ("study_id", "cache_identity", "output_root"):
        if required not in study_block:
            raise ValueError(f"{where}: study block is missing {required!r}")
    study_id = str(study_block["study_id"])
    cache = CacheIdentity.from_json(_resolve(study_block["cache_identity"], root))
    output = OutputRootPolicy(
        root=_resolve(study_block["output_root"], root),
        forbidden_roots=tuple(_resolve(p, root)
                              for p in study_block.get("forbidden_output_roots", ())),
    )

    spec_version = str(raw.get("spec_version") or "")
    if not spec_version:
        raise ValueError(f"{where}: 'spec_version' is required — a spec_id is versioned")
    params = _as_params(raw.get("params"), where)

    # --- the AOU sign convention, declared and cross-checked ---------------- #
    directions_block = raw.get("directions") or {}
    if set(directions_block) != {d.value for d in Direction}:
        raise ValueError(
            f"{where}: 'directions' must declare exactly {[d.value for d in Direction]}, "
            f"got {sorted(directions_block)}")
    declared_parents: dict[Direction, str] = {}
    for name, block in directions_block.items():
        block = block or {}
        declared = block.get("aou_sign")
        expected = Direction(name).aou_sign
        if declared != expected:
            raise ValueError(
                f"{where}: directions.{name}.aou_sign is {declared!r}, but the code convention "
                f"(eddy_pump.domain.Direction.aou_sign) is {expected!r}. One of the two is wrong "
                f"and neither may be changed to match the other without a ruling.")
        parent_ref = block.get("physical_parent")
        if not parent_ref:
            raise ValueError(
                f"{where}: directions.{name} must name its physical_parent — a limb without a "
                f"primary review frame has no denominator")
        declared_parents[Direction(name)] = str(parent_ref)

    physical = raw.get("physical") or {}
    if "aou" not in physical:
        raise ValueError(f"{where}: 'physical.aou' is required — it is the channel the limb flips")
    tracers = raw.get("tracers") or {}

    # --- the pools --------------------------------------------------------- #
    pool_specs = raw.get("pools") or []
    if not pool_specs:
        raise ValueError(f"{where}: no pools declared")

    parents: dict[Direction, CandidatePool] = {}
    pools: list[CandidatePool] = []
    # Physical parents first, in declaration order, so a child can always name its parent.
    for entry in sorted(pool_specs, key=lambda e: e.get("channel") != PHYSICAL):
        channel = str(entry.get("channel"))
        if channel not in CHANNELS:
            raise ValueError(f"{where}: pool channel {channel!r} is not one of {list(CHANNELS)}")
        direction = Direction(str(entry.get("direction")))
        name = f"{channel}_{direction.value}"
        # 'tracer_position' is retired; a child appends its one tracer term. Refused rather than
        # ignored, so the manifest cannot claim a channel order the study does not build.
        if "tracer_position" in entry:
            raise ValueError(
                f"{where}: pool {name} declares 'tracer_position', retired by the ruling of "
                f"(docs/DECISIONS.md, the nitrate variable-order ruling). Every child appends its one tracer term to "
                f"its physical parent — see config/events.yaml note (a).")

        if channel == PHYSICAL:
            variables = _parent_variables(physical, direction, where)
            tracer = None
            parent = None
        else:
            tracer = Tracer(channel)
            parent = parents.get(direction)
            if parent is None:
                raise ValueError(
                    f"{where}: pool {name} has no physical/{direction.value} parent declared — a "
                    f"tracer pool is a nested subset and cannot exist without its own limb's frame")
            term = _tracer_term(tracers, tracer, direction, where)
            variables = _child_variables(parent.spec, term)

        spec = EventSpec(name=name, direction=direction, tracer=tracer,
                         variables=variables, params=params, version=spec_version)
        declared_id = entry.get("spec_id")
        if declared_id is not None and str(declared_id) != spec.spec_id:
            raise ValueError(
                f"{where}: pool {name} pins spec_id {declared_id!r} but its content hashes to "
                f"{spec.spec_id!r}. The spec moved. Re-pin deliberately — a spec_id is what a "
                f"label, an anchor and a flux are all keyed to.")

        pool = CandidatePool(study_id=study_id, direction=direction, tracer=tracer,
                             spec=spec, parent=parent)
        if channel == PHYSICAL:
            wanted = declared_parents[direction]
            if wanted != f"{PHYSICAL}/{direction.value}":
                raise ValueError(
                    f"{where}: directions.{direction.value}.physical_parent is {wanted!r}, but "
                    f"the physical pool on that limb is {PHYSICAL}/{direction.value}")
            parents[direction] = pool
        pools.append(pool)

    # Declaration order, not construction order: the manifest's own order is the one a reader saw.
    order = {f"{e.get('channel')}_{e.get('direction')}": i for i, e in enumerate(pool_specs)}
    pools.sort(key=lambda p: order.get(p.event_type, len(order)))

    # --- the raw-input ranges ---------------------------------------------- #
    # Parsed AFTER the pools: a raw input may not name a channel any pool detects on, and that
    # set is not known until the pools exist.
    spec_channels = {v.name for p in pools for v in p.spec.variables}
    raw_block = raw.get("raw_inputs") or []
    if not isinstance(raw_block, list):
        raise ValueError(f"{where}: 'raw_inputs' must be a list of mappings, got "
                         f"{type(raw_block).__name__}")
    input_ranges = tuple(_as_input_range(item, where, spec_channels) for item in raw_block)
    names = [v.name for v in input_ranges]
    if len(set(names)) != len(names):
        raise ValueError(f"{where}: raw_inputs declares a column twice: {sorted(names)}")

    # --- the declared fleet ------------------------------------------------ #
    excl_block = raw.get("excluded_floats") or []
    if not isinstance(excl_block, list):
        raise ValueError(f"{where}: 'excluded_floats' must be a list of mappings, got "
                         f"{type(excl_block).__name__}")
    excluded = tuple(_as_excluded_float(item, where) for item in excl_block)

    return Study(study_id=study_id, cache=cache, output=output, pools=tuple(pools),
                 spec_version=spec_version, params=params, manifest_path=path,
                 input_ranges=input_ranges, excluded_floats=excluded)


#: Readable alias — `load_study()` says what comes back.
load_study = load_manifest
