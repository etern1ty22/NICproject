from __future__ import annotations

import hashlib
import io
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen
import zipfile

import yaml


class DatasetDownloadError(RuntimeError):
    """Raised when a manifest entry cannot be resolved or verified."""


def download_dataset(
    manifest_path: str | Path,
    dataset_id: str,
    output_dir: str | Path,
) -> Path:
    manifest_path = Path(manifest_path)
    manifest = _load_manifest(manifest_path)
    datasets = manifest.get("datasets", {})
    if dataset_id not in datasets:
        raise DatasetDownloadError(f"Dataset '{dataset_id}' was not found in manifest {manifest_path}.")

    spec = datasets[dataset_id]
    url = spec.get("url")
    relative_path = spec.get("path")
    sha256 = spec.get("sha256")
    filename = spec.get("filename")
    archive_member = spec.get("archive_member")
    if bool(url) == bool(relative_path):
        raise DatasetDownloadError(f"Dataset '{dataset_id}' must define exactly one of url or path.")
    if not sha256 or not filename:
        raise DatasetDownloadError(f"Dataset '{dataset_id}' must define sha256 and filename.")

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename

    payload = _read_source_bytes(spec=spec, manifest_path=manifest_path)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != sha256:
        raise DatasetDownloadError(
            f"Checksum mismatch for dataset '{dataset_id}': expected {sha256}, got {digest}."
        )

    output_bytes = _extract_archive_member(payload, archive_member) if archive_member else payload
    target_path.write_bytes(output_bytes)
    return target_path


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        raise DatasetDownloadError(f"Manifest does not exist: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _read_source_bytes(spec: dict, manifest_path: Path) -> bytes:
    relative_path = spec.get("path")
    if relative_path:
        local_path = (manifest_path.parent / relative_path).resolve()
        if not local_path.exists():
            raise DatasetDownloadError(f"Local dataset file does not exist: {local_path}")
        return local_path.read_bytes()

    url = spec["url"]
    return _download_bytes(url)


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


def _extract_archive_member(payload: bytes, archive_member: str) -> bytes:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            with archive.open(archive_member) as handle:
                return handle.read()
    except KeyError as exc:
        raise DatasetDownloadError(f"Archive member '{archive_member}' was not found in the downloaded archive.") from exc
    except zipfile.BadZipFile as exc:
        raise DatasetDownloadError("archive_member was specified but the downloaded payload is not a valid zip archive.") from exc
