import hashlib
from pathlib import Path
import tempfile
import unittest
import zipfile

import _bootstrap  # noqa: F401
import yaml

from nic_vrptw.data.downloads import DatasetDownloadError, download_dataset


class DownloadTests(unittest.TestCase):
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
                                "url": f"file://{source_path}",
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
                                "url": f"file://{tmp / 'missing.txt'}",
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


if __name__ == "__main__":
    unittest.main()
