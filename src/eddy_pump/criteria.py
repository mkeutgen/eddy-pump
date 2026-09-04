"""The human review criteria: `config/criteria.yaml` loaded, checked and versioned.

reads  config/criteria.yaml, config/events.yaml
writes nothing

Entry points: `load_criteria()` for all of them by id, `active_criterion()` for the one the
study declares, and `require_ruled()`, the gate the batch builder calls. Only the YAML is
authority; nothing here types a clause.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .manifest import MANIFEST_PATH, REPO_ROOT

CRITERIA_PATH = REPO_ROOT / "config" / "criteria.yaml"

STATUSES = ("historical", "proposed", "ruled")
LIMBS = ("subduction", "obduction")
REQUIRED = ("id", "status", "applies_to", "limb_sign", "clauses", "unit_of_judgement")


@dataclass(frozen=True)
class Criterion:
    id: str
    status: str
    applies_to: tuple[str, ...]
    limb_sign: Mapping[str, Mapping[str, str]]
    clauses: Mapping[int, str]
    unit_of_judgement: str
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def is_ruled(self) -> bool:
        """Only a ruled criterion may open a batch."""
        return self.status == "ruled"

    @property
    def two_limb(self) -> bool:
        return set(self.applies_to) == set(LIMBS)

    def aou_sign(self, limb: str) -> str:
        return str(self.limb_sign[limb]["AOU"])


def _as_criterion(block: Mapping[str, Any], where: str) -> Criterion:
    for key in REQUIRED:
        if key not in block:
            raise ValueError(f"{where}: criterion {block.get('id', '?')!r} is missing {key!r}")
    cid = str(block["id"])
    status = str(block["status"])
    if status not in STATUSES:
        raise ValueError(f"{where}: criterion {cid!r} has status {status!r}, not one of {STATUSES}")
    applies = tuple(str(x) for x in block["applies_to"])
    bad = [x for x in applies if x not in LIMBS]
    if bad or not applies:
        raise ValueError(f"{where}: criterion {cid!r} applies_to {applies!r}; limbs are {LIMBS}")
    signs = block["limb_sign"]
    for limb in applies:
        if limb not in signs or "AOU" not in signs[limb] or "ABS_SAL" not in signs[limb]:
            raise ValueError(f"{where}: criterion {cid!r} declares no AOU/ABS_SAL sign for {limb!r}")
        if signs[limb]["AOU"] not in ("negative", "positive"):
            raise ValueError(f"{where}: criterion {cid!r}: AOU sign for {limb!r} must be negative or positive")
    clauses = {int(k): str(v) for k, v in dict(block["clauses"]).items()}
    if sorted(clauses) != list(range(1, len(clauses) + 1)):
        raise ValueError(f"{where}: criterion {cid!r} clauses must be numbered 1..n without gaps")
    if status == "proposed" and not any(k.startswith("proposed") for k in block):
        raise ValueError(f"{where}: criterion {cid!r} is proposed but says neither when nor by whom")
    return Criterion(
        id=cid, status=status, applies_to=applies, limb_sign=signs, clauses=clauses,
        unit_of_judgement=str(block["unit_of_judgement"]), raw=dict(block),
    )


def load_criteria(path: str | Path | None = None) -> dict[str, Criterion]:
    """All criteria in `config/criteria.yaml`, by id. Refuses a duplicate id."""
    import yaml

    p = Path(CRITERIA_PATH if path is None else path)
    where = str(p)
    with open(p, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    blocks = (raw or {}).get("criteria")
    if not isinstance(blocks, list) or not blocks:
        raise ValueError(f"{where}: expected a non-empty 'criteria' list")
    out: dict[str, Criterion] = {}
    for block in blocks:
        c = _as_criterion(block, where)
        if c.id in out:
            raise ValueError(f"{where}: duplicate criterion id {c.id!r}")
        out[c.id] = c
    return out


def study_criterion_version(manifest_path: str | Path | None = None) -> str:
    """The `criterion_version` the study block of the manifest declares. Required."""
    import yaml

    p = Path(MANIFEST_PATH if manifest_path is None else manifest_path)
    with open(p, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    study = (raw or {}).get("study") or {}
    if "criterion_version" not in study:
        raise ValueError(f"{p}: the study block declares no 'criterion_version'")
    return str(study["criterion_version"])


def active_criterion(manifest_path: str | Path | None = None,
                     criteria_path: str | Path | None = None) -> Criterion:
    """The study's criterion, checked: it exists and it applies to both limbs."""
    version = study_criterion_version(manifest_path)
    criteria = load_criteria(criteria_path)
    if version not in criteria:
        raise ValueError(f"criterion_version {version!r} is not in {CRITERIA_PATH}: {sorted(criteria)}")
    c = criteria[version]
    if not c.two_limb:
        raise ValueError(f"criterion {version!r} applies to {c.applies_to}; the study has both limbs")
    return c


def require_ruled(c: Criterion) -> Criterion:
    """The batch builder's gate: a batch cannot open under a criterion nobody has ruled."""
    if not c.is_ruled:
        raise ValueError(
            f"criterion {c.id!r} is {c.status!r}, not ruled — no batch may be built under it "
            f"(mark it ruled in config/criteria.yaml; the criterion is decided in docs/DECISIONS.md, "
            f"'The labels and the numbers')"
        )
    return c
