"""eddy_pump — the active science layer for the net-carbon study of the submesoscale eddy pump.

    from eddy_pump import load_manifest
    study = load_manifest()                      # config/events.yaml
    study.pool("physical", "subduction").pool_id # 'net_carbon_v1/physical/subduction'
    study.detector_configs()                     # {pool_id: argopod variables + params}
"""
from __future__ import annotations

from .manifest import MANIFEST_PATH, REPO_ROOT, load_manifest, load_study
from .spec import (
    DECLARED_PARAM_FIELDS,
    DECLARED_VARIABLE_FIELDS,
    CandidateKey,
    CandidatePool,
    EventSpec,
    declared_content,
    undeclared_settings,
)
from .study import CacheIdentity, DetectorConfig, ExcludedFloat, OutputRootPolicy, Study
from .vocabulary import CANONICAL_EVENT_TYPES, CHANNELS, PHYSICAL, Direction, Tracer

__all__ = [
    "CANONICAL_EVENT_TYPES",
    "CHANNELS",
    "DECLARED_PARAM_FIELDS",
    "DECLARED_VARIABLE_FIELDS",
    "PHYSICAL",
    "CacheIdentity",
    "CandidateKey",
    "CandidatePool",
    "DetectorConfig",
    "Direction",
    "EventSpec",
    "ExcludedFloat",
    "MANIFEST_PATH",
    "OutputRootPolicy",
    "REPO_ROOT",
    "Study",
    "Tracer",
    "declared_content",
    "load_manifest",
    "load_study",
    "undeclared_settings",
]
