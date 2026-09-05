import json
from pathlib import Path
import tempfile
import unittest

import suite_manifest


class SuiteManifestTests(unittest.TestCase):
    def make_manifest(self, root: Path) -> Path:
        tasks_root = root / "task-data"
        for task, content in (("alpha", "first"), ("beta", "second")):
            task_dir = tasks_root / task
            task_dir.mkdir(parents=True)
            (task_dir / "task.toml").write_text(f'[task]\nname = "org/{task}"\n')
            (task_dir / "instruction.md").write_text(content)
        manifest = {
            "schema_version": 1,
            "id": "test-suite",
            "version": "1.0.0",
            "name": "Test suite",
            "sources": {
                "old": {
                    "task_root": "task-data",
                    "dataset": "org/dataset",
                    "version": "2.1",
                    "repository": "https://example.test/tasks.git",
                    "revision": "abc123",
                }
            },
            "tiers": {"smoke": ["alpha"], "full": ["alpha", "beta"]},
            "tasks": {},
        }
        for task in ("alpha", "beta"):
            manifest["tasks"][task] = {
                "source": "old",
                "upstream_name": f"org/{task}",
                "content_sha256": suite_manifest.task_tree_digest(tasks_root / task),
            }
        path = root / "suite.json"
        path.write_text(json.dumps(manifest))
        return path

    def test_load_resolves_tiers_groups_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite = suite_manifest.SuiteManifest.load(self.make_manifest(root))
            self.assertEqual(suite.tasks_for("smoke"), ["alpha"])
            self.assertEqual(
                suite.grouped_tasks(["beta", "alpha"]),
                [(root / "task-data", ["beta", "alpha"])],
            )
            identity = suite.task_identity("alpha")
            self.assertEqual(identity["upstream_name"], "org/alpha")
            self.assertEqual(identity["provenance"]["revision"], "abc123")
            self.assertRegex(suite.manifest_hash, r"^[0-9a-f]{64}$")

    def test_manifest_hash_is_independent_of_checkout_location_and_name(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_path = self.make_manifest(Path(first))
            second_path = self.make_manifest(Path(second))
            raw = json.loads(second_path.read_text())
            raw["name"] = "Renamed for display"
            second_path.write_text(json.dumps(raw))
            self.assertEqual(
                suite_manifest.SuiteManifest.load(first_path).manifest_hash,
                suite_manifest.SuiteManifest.load(second_path).manifest_hash,
            )

    def test_modified_task_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.make_manifest(root)
            (root / "task-data" / "alpha" / "instruction.md").write_text("changed")
            with self.assertRaisesRegex(
                suite_manifest.SuiteManifestError, "content digest mismatch"
            ):
                suite_manifest.SuiteManifest.load(path)

    def test_source_revision_file_must_match_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.make_manifest(root)
            (root / "task-data" / "REVISION").write_text("different")
            with self.assertRaisesRegex(
                suite_manifest.SuiteManifestError, "revision does not match"
            ):
                suite_manifest.SuiteManifest.load(path)

    def test_executable_bit_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.make_manifest(root)
            instruction = root / "task-data" / "alpha" / "instruction.md"
            instruction.chmod(instruction.stat().st_mode | 0o100)
            with self.assertRaisesRegex(
                suite_manifest.SuiteManifestError, "content digest mismatch"
            ):
                suite_manifest.SuiteManifest.load(path)

    def test_unknown_and_duplicate_requested_tasks_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            suite = suite_manifest.SuiteManifest.load(
                self.make_manifest(Path(directory))
            )
            with self.assertRaisesRegex(suite_manifest.SuiteManifestError, "not part"):
                suite.resolve_task("missing")
            with self.assertRaisesRegex(suite_manifest.SuiteManifestError, "duplicate"):
                suite.grouped_tasks(["alpha", "alpha"])

    def test_unknown_manifest_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_manifest(Path(directory))
            raw = json.loads(path.read_text())
            raw["ignored_identity_field"] = "unsafe"
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(
                suite_manifest.SuiteManifestError, "unsupported field"
            ):
                suite_manifest.SuiteManifest.load(path)

    def test_repository_manifests_match_vendored_tasks(self):
        root = Path(__file__).resolve().parents[1]
        expected = {
            "legacy-mini20.json": 20,
            "core19.json": 19,
        }
        for name, count in expected.items():
            with self.subTest(manifest=name):
                suite = suite_manifest.SuiteManifest.load(root / "suites" / name)
                self.assertEqual(len(suite.tasks_for("full")), count)
                self.assertEqual(len(suite.tasks), count)

    def test_default_manifest_matches_canonical_subset_files(self):
        root = Path(__file__).resolve().parents[1]
        suite = suite_manifest.SuiteManifest.load(root / "suites" / "core19.json")
        for tier in ("smoke", "full"):
            expected = [
                line.strip()
                for line in (root / "subsets" / f"{tier}.txt").read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
            self.assertEqual(suite.tasks_for(tier), expected)

    def test_core19_removes_only_build_pov_ray(self):
        root = Path(__file__).resolve().parents[1]
        legacy = suite_manifest.SuiteManifest.load(
            root / "suites" / "legacy-mini20.json"
        )
        core = suite_manifest.SuiteManifest.load(root / "suites" / "core19.json")
        removed = {"build-pov-ray"}
        self.assertEqual(
            core.tasks_for("full"),
            [task for task in legacy.tasks_for("full") if task not in removed],
        )


if __name__ == "__main__":
    unittest.main()
