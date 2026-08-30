import tempfile
from pathlib import Path
import unittest
from unittest import mock

import terminal_bench


class TerminalBenchRunnerTests(unittest.TestCase):
    def test_run_defaults_to_full_and_two_conditional_attempts(self):
        args = terminal_bench.parse_args(
            [
                "run",
                "--platform",
                "test-local",
                "--engine",
                "llama.cpp",
                "--backend",
                "vulkan",
            ]
        )
        self.assertEqual(args.tier, "full")
        self.assertEqual(args.attempts, 2)

    def test_run_requires_platform_engine_and_backend(self):
        with self.assertRaises(SystemExit):
            terminal_bench.parse_args(
                ["run", "--engine", "llama.cpp", "--backend", "vulkan"]
            )
        with self.assertRaises(SystemExit):
            terminal_bench.parse_args(["run", "--platform", "test-local"])
        with self.assertRaises(SystemExit):
            terminal_bench.parse_args(
                ["run", "--platform", "test-local", "--backend", "vulkan"]
            )
        with self.assertRaises(SystemExit):
            terminal_bench.parse_args(
                ["run", "--platform", "test-local", "--engine", "llama.cpp"]
            )

    def test_tiers_are_nested_and_tasks_exist(self):
        smoke = set(terminal_bench.load_tier("smoke"))
        full = set(terminal_bench.load_tier("full"))
        self.assertLessEqual(smoke, full)
        self.assertEqual((len(smoke), len(full)), (1, 20))
        terminal_bench.validate_tasks(terminal_bench.DEFAULT_TASKS_DIR, sorted(full))

    def test_smoke_avoids_local_endpoint_port_collision(self):
        self.assertEqual(terminal_bench.load_tier("smoke"), ["git-leak-recovery"])

    def test_config_is_sequential_and_has_no_token_or_call_cap(self):
        config = terminal_bench.build_config(
            job_name="test",
            tasks_dir=Path("/tasks"),
            tasks=["task-a"],
            model="/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
            endpoint="http://localhost:8080/v1",
            api_key="local",
            concurrency=1,
            context_length=262144,
            agent_timeout_seconds=10800,
            keep_containers=False,
        )
        agent = config["agents"][0]
        self.assertEqual(config["n_concurrent_trials"], 1)
        self.assertEqual(config["n_attempts"], 1)
        self.assertEqual(agent["override_timeout_sec"], 10800)
        self.assertEqual(agent["name"], "terminus-2")
        self.assertNotIn("max_tokens", agent["kwargs"])
        self.assertNotIn("max_turns", agent["kwargs"])
        self.assertNotIn("cost_limit", agent["kwargs"])
        self.assertTrue(agent["kwargs"]["enable_summarize"])
        self.assertEqual(
            agent["kwargs"]["model_info"]["max_input_tokens"], 262144
        )
        self.assertEqual(agent["kwargs"]["api_base"], "http://localhost:8080/v1")
        self.assertEqual(
            agent["model_name"],
            "openai//models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
        )

    def test_context_length_can_be_discovered_or_overridden(self):
        self.assertEqual(
            terminal_bench.model_context_length({"meta": {"n_ctx": 126976}}),
            126976,
        )
        self.assertEqual(terminal_bench.model_context_length({}, 262144), 262144)
        with self.assertRaises(terminal_bench.RunnerError):
            terminal_bench.model_context_length({})

    def test_agent_has_separate_result_namespace(self):
        self.assertEqual(
            terminal_bench.result_tag("mtp", "llama.cpp", "vulkan"),
            "llama.cpp-vulkan-mtp-terminus-2",
        )

    def test_compute_backends_have_distinct_result_directories(self):
        vulkan_tag = terminal_bench.result_tag("q4", "llama.cpp", "vulkan")
        rocm_tag = terminal_bench.result_tag("q4", "llama.cpp", "rocm")
        self.assertNotEqual(vulkan_tag, rocm_tag)
        vulkan_dir = terminal_bench.result_store.model_results_dir(
            Path("results"), "strix-halo", "model.gguf", vulkan_tag
        )
        rocm_dir = terminal_bench.result_store.model_results_dir(
            Path("results"), "strix-halo", "model.gguf", rocm_tag
        )
        self.assertNotEqual(vulkan_dir, rocm_dir)
        self.assertIn("llama.cpp-vulkan", vulkan_dir.name)
        self.assertIn("llama.cpp-rocm", rocm_dir.name)

    def test_model_tag_is_preserved_in_default_job_and_retry_names(self):
        model_tag = (
            "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-"
            "chat-v2-imatrix-0731"
        )
        job_name = terminal_bench.make_job_name(
            "full",
            "deepseek-v4-flash",
            model_tag,
            engine="llama.cpp",
            backend="vulkan",
        )
        self.assertIn(model_tag, job_name)
        self.assertIn("llama.cpp--vulkan", job_name)
        self.assertEqual(
            terminal_bench.attempt_job_name(job_name, 2), f"{job_name}-attempt2"
        )

    def test_attempt_budget_is_not_part_of_evaluation_identity(self):
        profile = terminal_bench.evaluation_profile(
            tasks_dir=terminal_bench.DEFAULT_TASKS_DIR,
            model="model",
            model_metadata={},
            endpoint="http://localhost:8080/v1",
            engine="llama.cpp",
            engine_version="b1234",
            backend="vulkan",
            backend_version="1.4.304",
            inference_profile=None,
            context_length=262144,
            agent_timeout_seconds=10800,
        )
        self.assertNotIn("attempts", profile)
        self.assertNotIn("attempt_policy", profile)
        self.assertEqual(profile["engine"], "llama.cpp")
        self.assertEqual(profile["engine_version"], "b1234")
        self.assertEqual(profile["backend"], "vulkan")
        self.assertEqual(profile["backend_version"], "1.4.304")
        self.assertNotIn("rocm_version", profile)

    def test_rocm_version_alias_only_applies_to_rocm_backend(self):
        args = terminal_bench.parse_args(
            [
                "run",
                "--platform",
                "test-local",
                "--engine",
                "llama.cpp",
                "--backend",
                "rocm",
                "--rocm-version",
                "7.14",
            ]
        )
        self.assertEqual(
            terminal_bench.runtime_args(args),
            ("llama.cpp", None, "rocm", "7.14"),
        )
        args.backend = "vulkan"
        with self.assertRaises(terminal_bench.RunnerError):
            terminal_bench.runtime_args(args)

    def test_retry_state_selects_only_failed_tasks_below_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            profile = {"benchmark": "Terminal-Bench-Local"}
            profile_hash = terminal_bench.result_store.evaluation_profile_hash(profile)
            meta = {"profile_hash": profile_hash, "evaluation_profile": profile}
            terminal_bench.result_store.write_json(model_dir / "run-meta.json", meta)
            for task, passed, attempts in (
                ("passed", True, 1),
                ("failed-once", False, 1),
                ("failed-twice", False, 2),
            ):
                terminal_bench.result_store.write_json(
                    model_dir / f"results-{task}.json",
                    {
                        "task": task,
                        "passed": passed,
                        "attempts": [{}] * attempts,
                        "profile_hash": profile_hash,
                        "evaluation_profile": profile,
                    },
                )
            _, pending, completed_round = terminal_bench.retry_state(model_dir, 2)
            self.assertEqual(pending, ["failed-once"])
            self.assertEqual(completed_round, 1)

    def test_retry_state_accepts_legacy_attempt_policy_profiles(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            current = {"benchmark": "Terminal-Bench-Local", "model_id": "model"}
            legacy = {**current, "attempts": 1, "attempt_policy": "stop_on_pass"}
            terminal_bench.result_store.write_json(
                model_dir / "run-meta.json",
                {
                    "profile_hash": terminal_bench.result_store.evaluation_profile_hash(current),
                    "evaluation_profile": current,
                },
            )
            for task, profile in (("old", legacy), ("new", current)):
                terminal_bench.result_store.write_json(
                    model_dir / f"results-{task}.json",
                    {
                        "task": task,
                        "passed": False,
                        "attempts": [{}],
                        "profile_hash": terminal_bench.result_store.json_hash(profile),
                        "evaluation_profile": profile,
                    },
                )
            meta, pending, completed_round = terminal_bench.retry_state(model_dir, 2)
            self.assertEqual(pending, ["new", "old"])
            self.assertEqual(completed_round, 1)
            self.assertEqual(meta["evaluation_profile"], current)

    def test_conditional_attempt_resumes_existing_job_instead_of_recreating_it(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs_dir = Path(directory) / "jobs"
            group = "benchmark-job"
            attempt_job = jobs_dir / f"{group}-attempt2"
            attempt_job.mkdir(parents=True)
            profile_hash = "profile-hash"
            existing_meta = {
                "job_name": attempt_job.name,
                "attempt_group": group,
                "attempt_round": 2,
                "profile_hash": profile_hash,
            }
            terminal_bench.result_store.write_json(
                attempt_job / "runner-meta.json", existing_meta
            )

            meta = {
                "job_name": group,
                "attempt_group": group,
                "attempt_round": 1,
                "max_attempts": 2,
                "requested_tasks": ["task-a"],
                "platform": {"id": "test-local"},
                "model": {"id": "model"},
                "result_tag": "test",
                "profile_hash": profile_hash,
                "evaluation_profile": {},
            }
            config = {
                "job_name": group,
                "datasets": [{"task_names": ["task-a"]}],
            }
            resumed_meta = {**existing_meta, **meta, "attempt_round": 2}

            with (
                mock.patch.object(terminal_bench, "JOBS_DIR", jobs_dir),
                mock.patch.object(
                    terminal_bench.result_store,
                    "tasks_requiring_attempt",
                    return_value=["task-a"],
                ),
                mock.patch.object(
                    terminal_bench,
                    "resume_harbor_job",
                    return_value=(0, True, resumed_meta),
                ) as resume_job,
                mock.patch.object(terminal_bench, "execute_harbor_job") as execute_job,
            ):
                result = terminal_bench.continue_conditional_attempts(
                    meta=meta,
                    base_config=config,
                    completed_round=1,
                    results_root=Path(directory) / "results",
                    runtime="podman",
                )

            self.assertEqual(result, 0)
            resume_job.assert_called_once_with(
                job_dir=attempt_job,
                results_root=Path(directory) / "results",
                runtime="podman",
            )
            execute_job.assert_not_called()

    def test_tier_validation_rejects_missing_task(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(terminal_bench.RunnerError):
                terminal_bench.validate_tasks(Path(directory), ["missing"])


if __name__ == "__main__":
    unittest.main()
