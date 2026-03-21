import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import _bootstrap  # noqa: F401
import yaml

from nic_vrptw.data.downloads import DatasetDownloadError, download_dataset


class DownloadTests(unittest.TestCase):
    def test_download_dataset_sends_browser_like_user_agent_for_http_urls(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            payload = b"fixture-payload"
            digest = hashlib.sha256(payload).hexdigest()
            manifest_path = tmp / "manifest.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "datasets": {
                            "demo": {
                                "url": "https://example.com/demo.txt",
                                "filename": "downloaded.txt",
                                "sha256": digest,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            class _FakeResponse:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def read(self) -> bytes:
                    return payload

            with patch("nic_vrptw.data.downloads.urlopen", return_value=_FakeResponse()) as mocked_urlopen:
                output_path = download_dataset(manifest_path, "demo", tmp / "downloads")

            request = mocked_urlopen.call_args.args[0]
            self.assertEqual(request.full_url, "https://example.com/demo.txt")
            self.assertIn("Mozilla/5.0", request.headers["User-agent"])
            self.assertEqual(request.headers["Accept"], "*/*")
            self.assertEqual(output_path.read_bytes(), payload)

    def test_download_dataset_verifies_checksum_for_relative_fixture_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_path = tmp / "source.txt"
            source_path.write_text("fixture-payload", encoding="utf-8")
            digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
            manifest_path = tmp / "manifest.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "datasets": {
                            "demo": {
                                "path": "source.txt",
                                "filename": "downloaded.txt",
                                "sha256": digest,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            output_path = download_dataset(manifest_path, "demo", tmp / "downloads")
            self.assertEqual(output_path.read_text(encoding="utf-8"), "fixture-payload")

    def test_download_dataset_supports_file_uri(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_path = tmp / "source.txt"
            source_path.write_text("fixture-payload", encoding="utf-8")
            digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
            manifest_path = tmp / "manifest.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "datasets": {
                            "demo": {
                                "url": source_path.resolve().as_uri(),
                                "filename": "downloaded.txt",
                                "sha256": digest,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            output_path = download_dataset(manifest_path, "demo", tmp / "downloads")
            self.assertEqual(output_path.read_text(encoding="utf-8"), "fixture-payload")

    def test_download_dataset_extracts_archive_member(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            archive_path = tmp / "archive.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("inner/demo.txt", "archived-payload")
            digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
            manifest_path = tmp / "manifest.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "datasets": {
                            "demo": {
                                "path": "archive.zip",
                                "archive_member": "inner/demo.txt",
                                "filename": "demo.txt",
                                "sha256": digest,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            output_path = download_dataset(manifest_path, "demo", tmp / "downloads")
            self.assertEqual(output_path.read_text(encoding="utf-8"), "archived-payload")

    def test_download_dataset_fails_on_checksum_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_path = tmp / "source.txt"
            source_path.write_text("fixture-payload", encoding="utf-8")
            manifest_path = tmp / "manifest.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "datasets": {
                            "demo": {
                                "url": source_path.resolve().as_uri(),
                                "filename": "downloaded.txt",
                                "sha256": "0" * 64,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(DatasetDownloadError):
                download_dataset(manifest_path, "demo", tmp / "downloads")

    def test_download_dataset_fails_on_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest_path = tmp / "manifest.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "datasets": {
                            "demo": {
                                "url": (tmp / "missing.txt").resolve().as_uri(),
                                "filename": "downloaded.txt",
                                "sha256": "0" * 64,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(DatasetDownloadError):
                download_dataset(manifest_path, "demo", tmp / "downloads")

    def test_download_dataset_fails_on_directory_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest_path = tmp / "manifest.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "datasets": {
                            "demo": {
                                "path": ".",
                                "filename": "downloaded.txt",
                                "sha256": "0" * 64,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(DatasetDownloadError) as exc_info:
                download_dataset(manifest_path, "demo", tmp / "downloads")
            self.assertIn("does not exist or is a directory", str(exc_info.exception))


if __name__ == "__main__":
    unittest.main()
