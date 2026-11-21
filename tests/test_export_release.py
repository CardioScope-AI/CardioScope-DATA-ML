"""Publication safety tests; real-model verification is run separately in Docker."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.export_release import (
    FILES,
    assert_scores,
    export_release,
    read_json,
    sha256,
    verify_snapshot,
)


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()

        self.output = self.root / "releases"
        self.dataset = self.root / "data.csv"
        self.dataset.write_bytes(b"synthetic dataset")

        for name in FILES:
            (self.source / name).write_bytes(
                ("synthetic:" + name).encode()
            )

    def export(self, name="v1"):
        return export_release(
            self.source,
            self.dataset,
            self.output,
            name,
        )

    def make_snapshot(self, metadata, report):
        return {
            "metadata.json": json.dumps(metadata).encode("utf-8"),
            "test_evaluation.json": json.dumps(report).encode("utf-8"),
        }

    def valid_schema_metadata(self, dataset_hash):
        return {
            "schema_version": 1,
            "dataset_sha256": dataset_hash,
            "split": {
                "test_size": 0.2,
                "random_state": 42,
                "stratify": True,
            },
        }

    def valid_schema_report(self, dataset_hash):
        return {
            "schema_version": 1,
            "dataset_sha256": dataset_hash,
        }

    def test_manifest_binds_exact_copies_and_refuses_overwrite(self):
        with patch(
            "scripts.export_release.verify_snapshot",
            return_value={
                "dataset_sha256": sha256(self.dataset.read_bytes())
            },
        ):
            destination = self.export()

            manifest = json.loads(
                (destination / "release.json").read_text(
                    encoding="utf-8"
                )
            )

            hash_keys = (
                "artifact_sha256",
                "metadata_sha256",
                "test_evaluation_sha256",
            )

            for name, key in zip(FILES, hash_keys):
                self.assertEqual(
                    (destination / name).read_bytes(),
                    (self.source / name).read_bytes(),
                )
                self.assertEqual(
                    manifest[key],
                    sha256((destination / name).read_bytes()),
                )

            before = (destination / "release.json").read_bytes()

            with self.assertRaises(FileExistsError):
                self.export()

            self.assertEqual(
                (destination / "release.json").read_bytes(),
                before,
            )
            self.assertTrue(self.export("v2").is_dir())

    def test_invalid_or_missing_source_never_publishes(self):
        with patch(
            "scripts.export_release.verify_snapshot",
            side_effect=ValueError("invalid model"),
        ):
            with self.assertRaises(ValueError):
                self.export()

        self.assertFalse((self.output / "v1").exists())

        (self.source / "pipeline.joblib").unlink()

        with self.assertRaises(FileNotFoundError):
            self.export()

        self.assertFalse((self.output / "v1").exists())

    def test_path_traversal_and_existing_lock_refused(self):
        invalid_names = ("../escape", "a/b", "", r"a\b", ".", "a b")

        for name in invalid_names:
            with self.assertRaises(ValueError):
                self.export(name)

        self.output.mkdir()
        lock = self.output / ".v1.lock"
        lock.write_text("another publisher", encoding="utf-8")

        with patch(
            "scripts.export_release.verify_snapshot",
            return_value={},
        ):
            with self.assertRaises(FileExistsError):
                self.export()

        self.assertTrue(lock.exists())
        self.assertFalse((self.output / "v1").exists())
        self.assertEqual(list(self.output.glob(".release-*")), [])

    def test_failed_write_does_not_publish_partial_release(self):
        original = Path.write_bytes

        def fail_report(path, data):
            if path.name == "test_evaluation.json":
                raise OSError("synthetic disk failure")
            return original(path, data)

        with (
            patch(
                "scripts.export_release.verify_snapshot",
                return_value={},
            ),
            patch.object(Path, "write_bytes", fail_report),
        ):
            with self.assertRaises(OSError):
                self.export()

        self.assertFalse((self.output / "v1").exists())
        self.assertEqual(list(self.output.glob(".release-*")), [])

    def test_mismatching_scores_and_duplicate_keys_refused(self):
        invalid_values = (True, float("nan"), 0.9, -1, 2)

        for value in invalid_values:
            with self.assertRaises(ValueError):
                assert_scores(
                    {"accuracy": value},
                    {"accuracy": 0.8},
                )

        with self.assertRaises(ValueError):
            read_json(b'{"a":1,"a":2}')

    def test_invalid_metadata_schema_version_is_refused(self):
        dataset = b"synthetic dataset"
        dataset_hash = sha256(dataset)

        metadata = self.valid_schema_metadata(dataset_hash)
        metadata["schema_version"] = 2
        report = self.valid_schema_report(dataset_hash)

        with self.assertRaises(ValueError):
            verify_snapshot(
                self.make_snapshot(metadata, report),
                dataset,
            )

    def test_mismatching_dataset_fingerprint_is_refused(self):
        dataset = b"synthetic dataset"

        metadata = self.valid_schema_metadata("wrong-fingerprint")
        report = self.valid_schema_report(sha256(dataset))

        with self.assertRaises(ValueError):
            verify_snapshot(
                self.make_snapshot(metadata, report),
                dataset,
            )

    def test_mismatching_report_fingerprint_is_refused(self):
        dataset = b"synthetic dataset"
        dataset_hash = sha256(dataset)

        metadata = self.valid_schema_metadata(dataset_hash)
        report = self.valid_schema_report("wrong-fingerprint")

        with self.assertRaises(ValueError):
            verify_snapshot(
                self.make_snapshot(metadata, report),
                dataset,
            )

    def test_invalid_split_protocol_is_refused(self):
        dataset = b"synthetic dataset"
        dataset_hash = sha256(dataset)

        metadata = self.valid_schema_metadata(dataset_hash)
        metadata["split"]["test_size"] = 0.3
        report = self.valid_schema_report(dataset_hash)

        with self.assertRaises(ValueError):
            verify_snapshot(
                self.make_snapshot(metadata, report),
                dataset,
            )


if __name__ == "__main__":
    unittest.main()