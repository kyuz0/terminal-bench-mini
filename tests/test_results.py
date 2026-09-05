import json
from pathlib import Path
import tempfile
import unittest

import results


class ResultsTests(unittest.TestCase):
    def test_empty_index_rebuild_does_not_create_a_suite_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results.rebuild_all_indexes(root)
            self.assertFalse((root / "suites").exists())

    def test_hosted_task_name_wins_over_generic_materialized_path(self):
        self.assertEqual(
            results.task_id_from_trial(
                {
                    "task_name": "terminal-bench/fin-saccr-rwa",
                    "task_id": {"path": "/cache/datasets/5b2103ac/task"},
                    "trial_name": "fin-saccr-rwa__abc123",
                }
            ),
            "fin-saccr-rwa",
        )

    def test_task_id_falls_back_to_parent_of_generic_task_path(self):
        self.assertEqual(
            results.task_id_from_trial(
                {"task_id": {"path": "/datasets/mvcc-lsm-compaction/task"}}
            ),
            "mvcc-lsm-compaction",
        )

    def test_conflicting_suite_identity_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "disagree"):
            results.metadata_suite(
                {
                    "suite": {"id": "one", "manifest_hash": "a" * 64},
                    "evaluation_profile": {
                        "suite": {"id": "two", "manifest_hash": "b" * 64}
                    },
                }
            )

    def test_terminal_harness_uses_harbor_path_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = root / "jobs" / "job"
            job.mkdir(parents=True)
            (job / "result.json").write_text("{}")
            self.make_trial(job, "task__one", task="task", reward=1)
            meta = self.run_meta()
            meta["harness"] = "Harbor"
            model_dir, _ = results.export_job(
                job,
                results_root=root / "export",
                repo_root=root,
                run_meta=meta,
            )
            result = results.read_json(model_dir / "results-task.json")
            self.assertIn("harbor_job", result)
            self.assertIn("harbor_jobs", result)
            self.assertIn("harbor_paths", result["attempts"][0])
            self.assertIsNone(result["attempts"][0]["endpoint"])
            self.assertNotIn("pier_job", result)

    def test_export_reuses_existing_descriptive_platform_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = root / "jobs" / "job"
            job.mkdir(parents=True)
            (job / "result.json").write_text("{}")
            self.make_trial(job, "task__one", task="task", reward=1)
            results_root = root / "export"
            results.write_json(
                results_root / "strix-halo" / "platform.json",
                {"id": "strix-halo", "name": "AMD Strix Halo"},
            )
            meta = self.run_meta()
            meta["platform"] = {"id": "strix-halo", "name": "strix-halo"}

            model_dir, _ = results.export_job(
                job,
                results_root=results_root,
                repo_root=root,
                run_meta=meta,
            )

            exported = results.read_json(model_dir / "results-task.json")
            self.assertEqual(
                exported["platform"],
                {"id": "strix-halo", "name": "AMD Strix Halo"},
            )
            self.assertEqual(
                results.read_json(model_dir / "run-meta.json")["platform"],
                {"id": "strix-halo", "name": "AMD Strix Halo"},
            )

    def make_trial(
        self,
        job: Path,
        name: str,
        *,
        task: str = "task-a",
        reward: int = 0,
        exception: dict | None = None,
    ) -> None:
        trial = job / name
        (trial / "agent").mkdir(parents=True)
        (trial / "artifacts").mkdir()
        (trial / "verifier").mkdir()
        (trial / "agent" / "trajectory.json").write_text(
            json.dumps({"schema_version": "ATIF-v1.7", "steps": [{"source": "agent"}]})
        )
        (trial / "agent" / "mini-swe-agent.trajectory.json").write_text("{}")
        (trial / "artifacts" / "model.patch").write_text("diff")
        (trial / "verifier" / "ctrf.json").write_text("{}")
        (trial / "verifier" / "test-stdout.txt").write_text("tests")
        payload = {
            "task_name": f"datacurve/{task}",
            "task_id": {"path": f"/tasks/{task}"},
            "trial_name": name,
            "started_at": "2026-01-01T00:00:00+00:00",
            "finished_at": "2026-01-01T00:00:02+00:00",
            "agent_result": {
                "n_input_tokens": 100,
                "n_cache_tokens": 80,
                "n_output_tokens": 20,
                "peak_context_tokens": 90,
                "n_agent_steps": 3,
            },
            "n_agent_steps": 3,
            "verifier_result": {"rewards": {"reward": reward, "partial": 0.5}},
            "exception_info": exception,
        }
        (trial / "result.json").write_text(json.dumps(payload))

    def run_meta(self) -> dict:
        profile = {
            "benchmark": "Terminal-Bench-Local",
            "model_id": "org/model-q8",
            "engine": "llama.cpp",
            "engine_version": None,
            "backend": "rocm",
            "backend_version": "7.14",
        }
        return {
            "platform": {"id": "strix-halo", "name": "Strix Halo"},
            "model": {
                "name": "Model",
                "id": "org/model-q8",
                "endpoint_metadata": {"n_ctx": 128000},
            },
            "engine": "llama.cpp",
            "engine_version": None,
            "backend": "rocm",
            "backend_version": "7.14",
            "quant": "q8",
            "inference_profile": "mtp",
            "tag": "long-context",
            "evaluation_profile": profile,
            "profile_hash": results.evaluation_profile_hash(profile),
        }

    def test_legacy_rocm_profile_matches_canonical_runtime_identity(self):
        legacy = {
            "benchmark": "Terminal-Bench-Local",
            "backend": "llama.cpp",
            "rocm_version": "7.14",
        }
        canonical = {
            "benchmark": "Terminal-Bench-Local",
            "engine": "llama.cpp",
            "engine_version": None,
            "backend": "rocm",
            "backend_version": "7.14",
        }
        self.assertTrue(results.matching_evaluation_profiles(legacy, canonical))

    def test_export_aggregates_attempts_and_copies_atif_transcripts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = root / "jobs" / "job-a"
            job.mkdir(parents=True)
            (job / "result.json").write_text("{}")
            self.make_trial(job, "task-a__one", reward=0)
            self.make_trial(job, "task-a__two", reward=1)
            model_dir, summary = results.export_job(
                job,
                results_root=root / "benchmark_results",
                repo_root=root,
                run_meta=self.run_meta(),
            )
            result = results.read_json(model_dir / "results-task-a.json")
            self.assertTrue(result["completed"])
            self.assertTrue(result["passed"])
            self.assertEqual(result["model"]["name"], "Model")
            self.assertEqual(result["succeeded_at_attempt"], 2)
            self.assertEqual(result["tokens"], {"input": 200, "cached": 160, "output": 40})
            self.assertEqual(len(result["attempts"]), 2)
            self.assertTrue((model_dir / "transcript-task-a-attempt1.json").is_file())
            self.assertTrue((model_dir / "transcript-task-a-attempt2.json").is_file())
            self.assertEqual(summary["passed_tasks"], 1)
            self.assertEqual(
                results.cached_tasks(model_dir, ["task-a"], self.run_meta()["profile_hash"]),
                ["task-a"],
            )
            index = results.read_json(root / "benchmark_results" / "index.json")
            self.assertEqual(len(index["platforms"]), 1)
            self.assertEqual(len(index["models"]), 1)
            self.assertEqual(index["tasks"][0]["id"], "task-a")

    def test_manifest_suite_exports_to_isolated_namespace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = root / "jobs" / "job-suite"
            job.mkdir(parents=True)
            (job / "result.json").write_text("{}")
            self.make_trial(job, "task-a__one", reward=1)
            meta = self.run_meta()
            suite = {
                "schema_version": 1,
                "id": "test-suite",
                "version": "1.0.0",
                "manifest_hash": "a" * 64,
            }
            provenance = {
                "id": "task-a",
                "source": "test-source",
                "upstream_name": "terminal-bench/task-a",
                "content_sha256": "sha256:" + "b" * 64,
            }
            meta["suite"] = suite
            meta["task_provenance"] = {"task-a": provenance}
            meta["evaluation_profile"] = {
                **meta["evaluation_profile"],
                "suite": suite,
            }
            meta["profile_hash"] = results.evaluation_profile_hash(
                meta["evaluation_profile"]
            )
            results_root = root / "benchmark_results"

            model_dir, summary = results.export_job(
                job,
                results_root=results_root,
                repo_root=root,
                run_meta=meta,
            )

            scope = results_root / "suites" / ("test-suite-" + "a" * 64)
            self.assertTrue(model_dir.is_relative_to(scope))
            self.assertEqual(
                results.results_root_from_model_dir(model_dir, meta), results_root
            )
            self.assertFalse((results_root / "strix-halo").exists())
            self.assertEqual(results.read_json(scope / "suite.json"), suite)
            self.assertEqual(results.read_json(scope / "index.json")["suite"], suite)
            catalog = results.read_json(results_root / "suites" / "index.json")
            self.assertEqual(catalog["suites"][0]["id"], "test-suite")
            rebuilt = results.rebuild_all_indexes(results_root)
            self.assertEqual(rebuilt["legacy"]["models"], [])
            self.assertEqual(len(rebuilt["suite_catalog"]["suites"]), 1)
            result = results.read_json(model_dir / "results-task-a.json")
            self.assertEqual(result["suite"], suite)
            self.assertEqual(result["task_provenance"], provenance)
            self.assertEqual(summary["suite"], suite)

    def test_suite_result_paths_isolate_structurally_colliding_tags(self):
        suite = {"id": "suite", "manifest_hash": "a" * 64}
        first_profile = results.evaluation_profile_hash(
            {"quant": "Q4", "inference_profile": "mtp-long"}
        )
        second_profile = results.evaluation_profile_hash(
            {"quant": "Q4-mtp", "inference_profile": "long"}
        )
        # The legacy display tag is identical; structured profile identity is not.
        display_tag = "engine-backend-Q4-mtp-long-agent"
        first = results.model_results_dir(
            Path("results"),
            "local",
            "model",
            display_tag,
            suite=suite,
            profile_hash=first_profile,
        )
        second = results.model_results_dir(
            Path("results"),
            "local",
            "model",
            display_tag,
            suite=suite,
            profile_hash=second_profile,
        )
        self.assertNotEqual(first, second)

    def test_manifest_suite_export_requires_task_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = root / "jobs" / "job-suite"
            job.mkdir(parents=True)
            (job / "result.json").write_text("{}")
            self.make_trial(job, "task-a__one", reward=1)
            meta = self.run_meta()
            suite = {
                "schema_version": 1,
                "id": "test-suite",
                "version": "1.0.0",
                "manifest_hash": "a" * 64,
            }
            meta["suite"] = suite
            meta["evaluation_profile"] = {
                **meta["evaluation_profile"],
                "suite": suite,
            }
            meta["profile_hash"] = results.evaluation_profile_hash(
                meta["evaluation_profile"]
            )
            with self.assertRaisesRegex(ValueError, "no valid provenance"):
                results.export_job(
                    job,
                    results_root=root / "benchmark_results",
                    repo_root=root,
                    run_meta=meta,
                )

    def test_conditional_attempt_export_merges_separate_harbor_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = root / "jobs"
            first = jobs / "job-first"
            second = jobs / "job-second"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            (first / "result.json").write_text("{}")
            (second / "result.json").write_text("{}")
            self.make_trial(first, "task-a__first", reward=0)
            self.make_trial(second, "task-a__second", reward=1)
            results_root = root / "benchmark_results"
            meta = self.run_meta()

            model_dir, _ = results.export_job(
                first,
                results_root=results_root,
                repo_root=root,
                run_meta=meta,
            )
            self.assertEqual(
                results.tasks_requiring_attempt(
                    model_dir, ["task-a"], meta["profile_hash"], 2
                ),
                ["task-a"],
            )

            results.export_job(
                second,
                results_root=results_root,
                repo_root=root,
                run_meta=meta,
                merge_existing_attempts=True,
            )
            result = results.read_json(model_dir / "results-task-a.json")
            self.assertTrue(result["passed"])
            self.assertEqual(result["succeeded_at_attempt"], 2)
            self.assertEqual(len(result["attempts"]), 2)
            self.assertEqual(len(result["harbor_jobs"]), 2)
            results.export_job(
                first,
                results_root=results_root,
                repo_root=root,
                run_meta=meta,
                merge_existing_attempts=True,
            )
            result = results.read_json(model_dir / "results-task-a.json")
            self.assertEqual(len(result["attempts"]), 2)
            self.assertEqual(
                results.tasks_requiring_attempt(
                    model_dir, ["task-a"], meta["profile_hash"], 2
                ),
                [],
            )

    def test_harness_error_is_exported_but_not_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = root / "jobs" / "job-b"
            job.mkdir(parents=True)
            (job / "result.json").write_text("{}")
            self.make_trial(
                job,
                "task-a__error",
                exception={"exception_type": "RuntimeError", "message": "broken"},
            )
            model_dir, summary = results.export_job(
                job,
                results_root=root / "benchmark_results",
                repo_root=root,
                run_meta=self.run_meta(),
            )
            result = results.read_json(model_dir / "results-task-a.json")
            self.assertFalse(result["completed"])
            self.assertEqual(summary["total_tasks"], 1)
            self.assertEqual(summary["passed_tasks"], 0)
            self.assertEqual(summary["pass_rate"], 0)
            self.assertEqual(summary["total_duration_ms"], 2000)
            self.assertEqual(
                summary["tokens"], {"input": 100, "cached": 80, "output": 20}
            )
            self.assertEqual(
                results.cached_tasks(model_dir, ["task-a"], self.run_meta()["profile_hash"]),
                [],
            )

    def test_model_directory_is_stable_and_platform_readable(self):
        path = results.model_results_dir(
            Path("benchmark_results"),
            "Strix Halo",
            "/models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf",
            "Long-Context",
        )
        self.assertEqual(path.parts[-2], "strix-halo")
        self.assertRegex(
            path.name,
            r"^Qwen3\.6-35B-A3B-UD-Q8_K_XL-[0-9a-f]{8}-Long-Context_results$",
        )

    def test_long_model_directory_retains_quantization_suffix(self):
        model_id = "/models/" + "Very-Long-Model-" * 20 + "UD-Q8_K_XL.gguf"
        path = results.model_results_dir(Path("benchmark_results"), "local", model_id, None)
        self.assertIn("Very-Long-Model", path.name)
        self.assertIn("UD-Q8_K_XL", path.name)

    def test_long_identity_tags_with_same_prefix_have_distinct_directories(self):
        prefix = "llama.cpp-rocm-" + "very-long-quant-" * 4
        first = results.model_results_dir(
            Path("benchmark_results"), "local", "model", prefix + "mtp"
        )
        second = results.model_results_dir(
            Path("benchmark_results"), "local", "model", prefix + "no-mtp"
        )
        self.assertNotEqual(first, second)
        self.assertIn("mtp", first.name)
        self.assertIn("no-mtp", second.name)
        self.assertEqual(len(results.identity_tag_component(prefix + "mtp")), 48)

    def test_long_platform_ids_with_same_prefix_have_distinct_directories(self):
        prefix = "lab-cluster-with-a-very-long-shared-platform-prefix-"
        first = results.model_results_dir(
            Path("benchmark_results"), prefix + "one", "model", None
        )
        second = results.model_results_dir(
            Path("benchmark_results"), prefix + "two", "model", None
        )
        self.assertNotEqual(first.parent, second.parent)
        self.assertLessEqual(len(first.parent.name), 48)

    def test_lossy_short_identity_tag_sanitization_cannot_collide(self):
        first = results.model_results_dir(
            Path("benchmark_results"), "local", "model", "quant/a"
        )
        second = results.model_results_dir(
            Path("benchmark_results"), "local", "model", "quant a"
        )
        self.assertNotEqual(first, second)

    def test_model_name_preserves_case_quantization_and_drops_gguf_extension(self):
        self.assertEqual(
            results.model_name(
                "/models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf"
            ),
            "Qwen3.6-35B-A3B-UD-Q8_K_XL",
        )

    def test_schema_v2_model_tag_is_read_as_quant(self):
        self.assertEqual(results.metadata_quant({"model_tag": "UD-Q4_K_XL"}), "UD-Q4_K_XL")
        self.assertEqual(results.metadata_quant({"quant": "MXFP4"}), "MXFP4")


if __name__ == "__main__":
    unittest.main()
