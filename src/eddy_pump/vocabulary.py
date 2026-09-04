"""The vocabulary a pool identity is spelled in: limbs, tracer channels, event types.

Entry points: `Direction` (a limb, and its AOU sign), `Tracer` (the term a child pool adds),
`channel_of`, and the constants `PHYSICAL`, `CHANNELS` and `CANONICAL_EVENT_TYPES`.
"""
from __future__ import annotations

from enum import StrEnum

__all__ = [
    "Direction",
    "Tracer",
    "PHYSICAL",
    "CHANNELS",
    "CANONICAL_EVENT_TYPES",
    "channel_of",
]


class Direction(StrEnum):
    """A limb of the pump, and the AOU sign convention that defines it.

    The strings are argopod's ``VariableConfig.sign_constraint`` vocabulary, not ours, so the
    value can be handed straight to a detector spec.
    """

    SUBDUCTION = "subduction"
    OBDUCTION = "obduction"

    @property
    def aou_sign(self) -> str:
        """``"negative"`` for subduction, ``"positive"`` for obduction.

        Subducted water is recently ventilated, so its apparent oxygen utilisation is negative;
        obducted water is old and respired, so AOU is positive.
        """
        return "negative" if self is Direction.SUBDUCTION else "positive"

    @property
    def opposite(self) -> Direction:
        """The other limb — the one this limb's return flux is measured against."""
        return Direction.OBDUCTION if self is Direction.SUBDUCTION else Direction.SUBDUCTION


class Tracer(StrEnum):
    """The tracer channel that extends a physical parent into a child pool.

    ``None`` — not a member of this enum — is the physical parents' own value: they carry no
    tracer term. A child is its directional parent plus exactly one of these terms, never two.
    """

    NITRATE = "nitrate"
    CARBON = "carbon"


#: The channel token of a pool with no tracer. One string, used in `pool_id`, in `EVENT_TYPE`
#: and in config/events.yaml, so a reader greps for the same word everywhere.
PHYSICAL = "physical"

#: Every channel token a `pool_id` may carry, in the order a reader expects: parent first.
CHANNELS: tuple[str, ...] = (PHYSICAL, Tracer.NITRATE.value, Tracer.CARBON.value)

#: The six event definitions — the whole vocabulary of `EVENT_TYPE`. A token outside this set is
#: refused wherever a pool_id is parsed. Why: docs/DECISIONS.md
CANONICAL_EVENT_TYPES: frozenset[str] = frozenset(
    f"{channel}_{direction.value}" for channel in CHANNELS for direction in Direction
)


def channel_of(tracer: Tracer | None) -> str:
    """The `pool_id` channel token for a tracer (or for the physical parents' ``None``)."""
    return PHYSICAL if tracer is None else tracer.value
