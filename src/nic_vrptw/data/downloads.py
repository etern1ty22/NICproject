from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

import yaml


class DatasetDownloadError(RuntimeError):
    """Raised when a manifest entry cannot be resolved or verified."""


def download_dataset(
    manifest_path: str | Path,
    dataset_id: str,
    output_dir: str | Path,
) -> Path:
    manifest = _load_manifest(Path(manifest_path))
    datasets = manifest.get("datasets", {})
    if dataset_id not in datasets:
        raise DatasetDownloadError(f"Dataset '{dataset_id}' was not found in manifest {manifest_path}.")

    spec = datasets[dataset_id]
    url = spec.get("url")
    sha256 = spec.get("sha256")
    filename = spec.get("filename")
    if not url or not sha256 or not filename:
        raise DatasetDownloadError(f"Dataset '{dataset_id}' must define url, sha256, and filename.")

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename

    payload = _download_bytes(url)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != sha256:
        raise DatasetDownloadError(
            f"Checksum mismatch for dataset '{dataset_id}': expected {sha256}, got {digest}."
        )

    target_path.write_bytes(payload)
    return target_path


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        raise DatasetDownloadError(f"Manifest does not exist: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _download_bytes(url: str) -> bytes:
    parsed = urlparse(url)
    if parsed.scheme in {"", "file"}:
        local_path = Path(parsed.path)
        if not local_path.exists():
            raise DatasetDownloadError(f"Local dataset file does not exist: {local_path}")
        return local_path.read_bytes()

    try:
        with urlopen(url) as response:
            return response.read()
    except URLError as exc:  # pragma: no cover - depends on external connectivity
        raise DatasetDownloadError(f"Failed to download dataset from {url}: {exc}") from exc
