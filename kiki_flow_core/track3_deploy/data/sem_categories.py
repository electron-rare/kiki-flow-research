"""32 default semantic categories for the sem:code species.

Derived from WordNet top-level lexicographer files, trimmed to 32 cognitively-relevant categories.
"""

from __future__ import annotations

SEM_CATEGORIES: tuple[str, ...] = (
    "person",
    "animal",
    "plant",
    "body",
    "food",
    "artifact",
    "substance",
    "shape",
    "location",
    "group",
    "quantity",
    "time",
    "event",
    "act",
    "state",
    "attribute",
    "cognition",
    "communication",
    "feeling",
    "motive",
    "perception",
    "possession",
    "process",
    "relation",
    "phenomenon",
    "Tops",
    "motion",
    "change",
    "contact",
    "creation",
    "social",
    "other",
)

N_SEM = 32

assert len(SEM_CATEGORIES) == N_SEM
