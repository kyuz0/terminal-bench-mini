import json
from pathlib import Path
import tempfile
import unittest

import results


class ResultsTests(unittest.TestCase):
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
            self.assertNotIn("pier_job", result)

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
