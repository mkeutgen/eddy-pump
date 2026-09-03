"""Compatibility re-exports; the domain objects live in three modules now.

`vocabulary` holds Direction, Tracer, channel_of and the pool-identity tokens; `spec` holds
EventSpec, CandidatePool, CandidateKey and the declared-field machinery; `study` holds Study,
CacheIdentity, OutputRootPolicy, DetectorConfig and ExcludedFloat. Assigning a name on this
module also assigns it on the module that defines it, so a caller may still patch it here.
"""
from __future__ import annotations

import sys
from types import ModuleType

from . import spec as _spec
from . import study as _study
from . import vocabulary as _vocabulary
from .spec import *  # noqa: F401,F403
from .study import *  # noqa: F401,F403
from .vocabulary import *  # noqa: F401,F403

__all__ = [*_vocabulary.__all__, *_spec.__all__, *_study.__all__]


def _set(module: ModuleType, name: str, value: object) -> None:
    for target in (_vocabulary, _spec, _study):
        if hasattr(target, name):
            setattr(target, name, value)
    ModuleType.__setattr__(module, name, value)


sys.modules[__name__].__class__ = type("_DomainCompat", (ModuleType,), {"__setattr__": _set})
