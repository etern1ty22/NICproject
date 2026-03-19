from __future__ import annotations

from pathlib import Path

from nic_vrptw.core.models import VRPTWInstance
from nic_vrptw.core.utils import stable_digest

from .solomon import parse_solomon_instance
from .vrplib_parser import parse_vrplib_instance


def load_instance(path: str | Path, format: str | None = None) -> VRPTWInstance:
    # Keep format dispatch centralized here so new dataset adapters can be added
    # without touching experiment code or solver code.
    instance_path = Path(path)
    detected_format = (format or _detect_format(instance_path)).lower()
    if detected_format in {"solomon", "homberger"}:
        return parse_solomon_instance(instance_path)
    if detected_format in {"vrplib", "ortec"}:
        return parse_vrplib_instance(instance_path)
    raise ValueError(f"Unsupported instance format: {detected_format}")


def fingerprint_instance(instance: VRPTWInstance) -> str:
    return stable_digest(instance.to_payload())


def _detect_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".vrp", ".vrplib"}:
        return "vrplib"
    return "solomon"
