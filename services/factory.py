from __future__ import annotations

from services.contracts import ServiceBundle
from services.service_bundle import build_service_bundle


_bundle: ServiceBundle | None = None


def get_service_bundle() -> ServiceBundle:
    global _bundle
    if _bundle is None:
        _bundle = build_service_bundle()
    return _bundle
