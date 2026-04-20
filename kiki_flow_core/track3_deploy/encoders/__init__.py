"""Text encoders for the bridge surrogate ablation sweep."""

from __future__ import annotations

from collections.abc import Callable

from kiki_flow_core.track3_deploy.encoders.base import TextEncoder

ENCODER_REGISTRY: dict[str, type[TextEncoder]] = {}


def register(name: str) -> Callable[[type[TextEncoder]], type[TextEncoder]]:
    """Class decorator to register an encoder by name in ENCODER_REGISTRY."""

    def deco(cls: type[TextEncoder]) -> type[TextEncoder]:
        ENCODER_REGISTRY[name] = cls
        return cls

    return deco
