import json
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import terminal_bench


class TerminalBenchRunnerTests(unittest.TestCase):
    def test_harbor_version_must_match_exact_pin(self):
        completed = mock.Mock(returncode=0, stdout="0.20.0\n", stderr="")
        with mock.patch.object(
            terminal_bench.subprocess, "run", return_value=completed
        ) as run:
            self.assertEqual(
                terminal_bench.installed_harbor_version(["/usr/bin/harbor"]),
                "0.20.0",
            )
        run.assert_called_once_with(
            ["/usr/bin/harbor", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )

        completed.stdout = "0.22.0\n"
        with (
            mock.patch.object(terminal_bench.subprocess, "run", return_value=completed),
            self.assertRaisesRegex(
                terminal_bench.RunnerError,
                "Harbor 0.20.0 is required, but 0.22.0 is installed",
            ),
        ):
            terminal_bench.installed_harbor_version(["harbor"])

    def test_harbor_version_check_reports_command_failure(self):
        completed = mock.Mock(returncode=1, stdout="", stderr="broken installation")
        with (
            mock.patch.object(terminal_bench.subprocess, "run", return_value=completed),
            self.assertRaisesRegex(
                terminal_bench.RunnerError, "version check failed: broken installation"
            ),
        ):
            terminal_bench.installed_harbor_version(["harbor"])

    def test_container_runtime_requires_compose(self):
        docker_version = mock.Mock(
            returncode=0, stdout="Docker version 27.0.0\n", stderr=""
        )
        compose_version = mock.Mock(
            returncode=0, stdout="Docker Compose version v2.29.0\n", stderr=""
        )
        with (
            mock.patch.object(
                terminal_bench.shutil, "which", return_value="/usr/bin/docker"
            ),
            mock.patch.object(
                terminal_bench.subprocess,
                "run",
                side_effect=[docker_version, compose_version],
            ) as run,
        ):
            runtime, description = terminal_bench.container_runtime()
        self.assertEqual(runtime, "docker")
        self.assertIn("Docker Compose version v2.29.0", description)
        self.assertEqual(
            run.call_args_list[1],
            mock.call(
                ["/usr/bin/docker", "compose", "version"],
                capture_output=True,
                text=True,
                check=False,
            ),
        )

        missing_compose = mock.Mock(
            returncode=1, stdout="", stderr="docker: 'compose' is not a command"
        )
        with (
            mock.patch.object(
                terminal_bench.shutil, "which", return_value="/usr/bin/docker"
            ),
            mock.patch.object(
                terminal_bench.subprocess,
                "run",
                side_effect=[docker_version, missing_compose],
            ),
            self.assertRaisesRegex(
                terminal_bench.RunnerError, "docker compose version failed"
            ),
        ):
            terminal_bench.container_runtime()

    def test_job_name_must_be_one_safe_filename_component(self):
        self.assertEqual(
            terminal_bench.validate_job_name("candidate-01"), "candidate-01"
        )
        for invalid in ("../escape", "/tmp/escape", ".hidden", "a/b", "a" * 221):
            with self.subTest(invalid=invalid), self.assertRaises(
                terminal_bench.RunnerError
            ):
                terminal_bench.validate_job_name(invalid)

        with self.assertRaises(SystemExit):
            terminal_bench.parse_args(
                [
                    "run",
                    "--platform",
                    "test-local",
                    "--model-name",
                    "test-model",
                    "--engine",
                    "llama.cpp",
                    "--backend",
                    "rocm",
                    "--job-name",
                    "../escape",
                ]
            )

    def test_resume_or_retry_rejects_a_different_harbor_protocol(self):
        terminal_bench.validate_stored_harbor_version(
            {"evaluation_profile": {"harbor_version": "0.20.0"}}
        )
        terminal_bench.validate_stored_harbor_version({})
        with self.assertRaisesRegex(
            terminal_bench.RunnerError,
            "created with Harbor 0.22.0.*required Harbor 0.20.0",
        ):
            terminal_bench.validate_stored_harbor_version(
                {"evaluation_profile": {"harbor_version": "0.22.0"}}
            )

    def test_run_defaults_to_full_and_two_conditional_attempts(self):
        args = terminal_bench.parse_args(
            [
                "run",
                "--platform",
                "test-local",
                "--model-name",
                "Test-Model",
                "--engine",
                "llama.cpp",
                "--backend",
                "vulkan",
            ]
        )
        self.assertEqual(args.tier, "full")
        self.assertEqual(args.attempts, 2)

    def test_default_and_named_suite_selection(self):
        default = terminal_bench.load_suite(None)
        self.assertEqual(default.id, "core19")
        self.assertEqual(len(default.tasks_for("full")), 19)

        legacy = terminal_bench.load_suite("legacy-mini20")
        self.assertEqual(legacy.id, "legacy-mini20")
        self.assertEqual(len(legacy.tasks_for("full")), 20)

    def test_suite_rejects_unknown_tier_or_task(self):
        suite = terminal_bench.load_suite("core19")
        with self.assertRaisesRegex(terminal_bench.RunnerError, "no tier 'missing'"):
            terminal_bench.selected_suite_tasks(
                mock.Mock(tier="missing", task=None), suite
            )
        with self.assertRaisesRegex(terminal_bench.RunnerError, "not part of suite"):
            terminal_bench.selected_suite_tasks(
                mock.Mock(tier="full", task="not-a-task"), suite
            )

    def test_doctor_validates_only_the_selected_suite_tier(self):
        args = terminal_bench.parse_args(
            [
                "doctor",
                "--suite",
                "core19",
                "--tier",
                "smoke",
                "--model",
                "model",
                "--context-length",
                "262144",
                "--skip-endpoint-check",
            ]
        )
        with (
            mock.patch.object(
                terminal_bench,
                "container_runtime",
                return_value=("docker", "Docker test"),
            ),
            mock.patch.object(
                terminal_bench, "harbor_command", return_value=["harbor"]
            ),
            mock.patch.object(
                terminal_bench, "installed_harbor_version", return_value="0.20.0"
            ),
            mock.patch.object(terminal_bench, "validate_tasks") as validate_tasks,
        ):
            *_, suite, tasks = terminal_bench.check_doctor(args)

        self.assertEqual(suite.id, "core19")
        self.assertEqual(tasks, ["git-leak-recovery"])
        validate_tasks.assert_called_once_with(
            terminal_bench.ROOT / "tasks", ["git-leak-recovery"]
        )

    def test_tasks_dir_is_only_a_legacy_default_root_alias(self):
        suite = terminal_bench.load_suite(None, terminal_bench.DEFAULT_TASKS_DIR)
        self.assertEqual(suite.id, "legacy-mini20")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                terminal_bench.RunnerError, "cannot override a suite manifest"
            ):
                terminal_bench.load_suite(None, Path(directory))

    def test_suite_and_tasks_dir_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            terminal_bench.parse_args(
                [
                    "list",
                    "--suite",
                    "legacy-mini20",
                    "--tasks-dir",
                    "tasks",
                ]
            )

    def test_multiple_endpoints_accept_commas_and_spaces(self):
        args = terminal_bench.parse_args(
            [
                "run",
                "--endpoints",
                "http://host-a:8000/v1,http://host-b:8000/v1,",
                "http://host-c:8000/v1",
                "--platform",
                "test-local",
                "--model-name",
                "Test-Model",
                "--engine",
                "llama.cpp",
                "--backend",
                "vulkan",
            ]
        )
        self.assertEqual(
            terminal_bench.connection_endpoints(args),
            [
                "http://host-a:8000/v1",
                "http://host-b:8000/v1",
                "http://host-c:8000/v1",
            ],
        )

    def test_endpoint_and_endpoints_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            terminal_bench.parse_args(
                [
                    "run",
                    "--endpoint",
                    "http://host-a:8000/v1",
                    "--endpoints",
                    "http://host-b:8000/v1",
                    "--platform",
                    "test-local",
                    "--model-name",
                    "Test-Model",
                    "--engine",
                    "llama.cpp",
                    "--backend",
                    "vulkan",
                ]
            )

    def test_run_requires_platform_engine_and_backend(self):
        with self.assertRaises(SystemExit):
            terminal_bench.parse_args(
                [
                    "run",
                    "--model-name",
                    "Test-Model",
                    "--engine",
                    "llama.cpp",
                    "--backend",
                    "vulkan",
                ]
            )
        with self.assertRaises(SystemExit):
            terminal_bench.parse_args(
                ["run", "--platform", "test-local", "--model-name", "Test-Model"]
            )
        with self.assertRaises(SystemExit):
            terminal_bench.parse_args(
                [
                    "run",
                    "--platform",
                    "test-local",
                    "--model-name",
                    "Test-Model",
                    "--backend",
                    "vulkan",
                ]
            )
        with self.assertRaises(SystemExit):
            terminal_bench.parse_args(
                [
                    "run",
                    "--platform",
                    "test-local",
                    "--model-name",
                    "Test-Model",
                    "--engine",
                    "llama.cpp",
                ]
            )

    def test_run_requires_canonical_model_name(self):
        with self.assertRaises(SystemExit):
            terminal_bench.parse_args(
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

    def test_tiers_are_nested_and_tasks_exist(self):
        smoke = set(terminal_bench.load_tier("smoke"))
        full = set(terminal_bench.load_tier("full"))
        self.assertLessEqual(smoke, full)
        self.assertEqual((len(smoke), len(full)), (1, 19))
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

    def test_config_display_redacts_api_key_and_file_is_private(self):
        config = terminal_bench.build_config(
            job_name="secret-test",
            tasks_dir=Path("/tasks"),
            tasks=["task-a"],
            model="model",
            endpoint="http://localhost:8080/v1",
            api_key="super-secret",
            concurrency=1,
            context_length=262144,
            agent_timeout_seconds=10800,
            keep_containers=False,
        )
        displayed = terminal_bench.display_config(config)
        self.assertEqual(
            displayed["agents"][0]["kwargs"]["llm_kwargs"]["api_key"],
            "<redacted>",
        )
        self.assertEqual(
            config["agents"][0]["kwargs"]["llm_kwargs"]["api_key"],
            "super-secret",
        )
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(terminal_bench, "ROOT", Path(directory)):
                path = terminal_bench.write_config(config)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(path.parent.stat().st_mode & 0o777, 0o700)

    def test_tasks_are_round_robin_partitioned_across_endpoints(self):
        self.assertEqual(
            terminal_bench.partition_tasks(["a", "b", "c", "d", "e"], 3),
            [["a", "d"], ["b", "e"], ["c"]],
        )

    def test_core19_duration_weights_balance_and_frontload_long_tasks(self):
        tasks = terminal_bench.load_suite("core19").tasks_for("full")
        shards = terminal_bench.partition_tasks(
            tasks,
            2,
            weights=terminal_bench.CORE19_TASK_ESTIMATED_MINUTES,
        )
        loads = [
            sum(terminal_bench.CORE19_TASK_ESTIMATED_MINUTES[task] for task in shard)
            for shard in shards
        ]
        self.assertEqual(shards[0][0], "break-filter-js-from-html")
        self.assertEqual(shards[1][0], "llm-inference-batching-scheduler")
        self.assertLess(abs(loads[0] - loads[1]), 5)
        self.assertEqual(sorted(task for shard in shards for task in shard), sorted(tasks))

    def test_distributed_jobs_use_one_endpoint_and_shard_each(self):
        meta = {
            "job_name": "group",
            "attempt_group": "group",
            "endpoint": "http://host-a/v1",
            "endpoints": ["http://host-a/v1", "http://host-b/v1"],
        }
        config = terminal_bench.build_config(
            job_name="group",
            tasks_dir=Path("/tasks"),
            tasks=["a", "b", "c"],
            model="model",
            endpoint="http://host-a/v1",
            api_key="local",
            concurrency=1,
            context_length=262144,
            agent_timeout_seconds=10800,
            keep_containers=False,
        )
        jobs = terminal_bench.build_attempt_jobs(
            meta=meta,
            base_config=config,
            tasks=["a", "b", "c"],
            attempt=1,
        )
        self.assertEqual(
            [job[0]["job_name"] for job in jobs],
            ["group-endpoint1", "group-endpoint2"],
        )
        self.assertEqual(
            [job[0]["datasets"][0]["task_names"] for job in jobs],
            [["a", "c"], ["b"]],
        )
        self.assertEqual(
            [job[0]["agents"][0]["kwargs"]["api_base"] for job in jobs],
            ["http://host-a/v1", "http://host-b/v1"],
        )

    def test_core19_distributed_jobs_record_weighted_scheduling(self):
        tasks = [
            "pypi-server",
            "fix-git",
            "break-filter-js-from-html",
            "llm-inference-batching-scheduler",
        ]
        meta = {
            "job_name": "group",
            "attempt_group": "group",
            "endpoint": "http://host-a/v1",
            "endpoints": ["http://host-a/v1", "http://host-b/v1"],
            "suite": {"id": "core19"},
        }
        config = terminal_bench.build_config(
            job_name="group",
            tasks_dir=Path("/tasks"),
            tasks=tasks,
            model="model",
            endpoint="http://host-a/v1",
            api_key="local",
            concurrency=1,
            context_length=262144,
            agent_timeout_seconds=10800,
            keep_containers=False,
        )
        jobs = terminal_bench.build_attempt_jobs(
            meta=meta,
            base_config=config,
            tasks=tasks,
            attempt=1,
        )
        self.assertEqual(
            [job[1]["scheduling"]["strategy"] for job in jobs],
            ["historical-duration-lpt", "historical-duration-lpt"],
        )
        self.assertEqual(
            [job[1]["executed_tasks"] for job in jobs],
            [
                ["break-filter-js-from-html", "pypi-server"],
                ["llm-inference-batching-scheduler", "fix-git"],
            ],
        )

    def test_distributed_jobs_preserve_multiple_dataset_roots(self):
        meta = {
            "job_name": "group",
            "attempt_group": "group",
            "endpoint": "http://host-a/v1",
            "endpoints": ["http://host-a/v1", "http://host-b/v1"],
        }
        config = terminal_bench.build_config(
            job_name="group",
            tasks_dir=Path("/unused"),
            tasks=["group-a1", "group-a2", "group-b1", "group-b2"],
            dataset_groups=[
                (Path("/tasks-a"), ["group-a1", "group-a2"]),
                (Path("/tasks-b"), ["group-b1", "group-b2"]),
            ],
            model="model",
            endpoint="http://host-a/v1",
            api_key="local",
            concurrency=1,
            context_length=262144,
            agent_timeout_seconds=10800,
            keep_containers=False,
        )
        jobs = terminal_bench.build_attempt_jobs(
            meta=meta,
            base_config=config,
            tasks=["group-a1", "group-a2", "group-b1", "group-b2"],
            attempt=1,
        )
        self.assertEqual(
            jobs[0][0]["datasets"],
            [
                {"path": "/tasks-a", "task_names": ["group-a1"]},
                {"path": "/tasks-b", "task_names": ["group-b1"]},
            ],
        )
        self.assertEqual(
            jobs[1][0]["datasets"],
            [
                {"path": "/tasks-a", "task_names": ["group-a2"]},
                {"path": "/tasks-b", "task_names": ["group-b2"]},
            ],
        )

    def test_dataset_selection_rejects_missing_or_ambiguous_tasks(self):
        with self.assertRaisesRegex(terminal_bench.RunnerError, "absent"):
            terminal_bench.select_config_datasets(
                [{"path": "/tasks", "task_names": ["a"]}], ["b"]
            )
        with self.assertRaisesRegex(terminal_bench.RunnerError, "more than one"):
            terminal_bench.select_config_datasets(
                [
                    {"path": "/tasks", "task_names": ["a"]},
                    {"path": "/tasks-b", "task_names": ["a"]},
                ],
                ["a"],
            )

    def test_distributed_children_launch_before_any_child_is_waited(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs_dir = root / "jobs"
            events = []
            launches = []

            class FakeProcess:
                def __init__(self, name):
                    self.name = name

                def wait(self, timeout=None):
                    events.append(f"wait-{self.name}")
                    (jobs_dir / self.name / "result.json").write_text(
                        '{"n_total_trials": 1, "stats": {'
                        '"n_completed_trials": 1, "n_errored_trials": 0, '
                        '"n_running_trials": 0, "n_pending_trials": 0, '
                        '"n_cancelled_trials": 0}}'
                    )
                    return 0

                def poll(self):
                    return 0

                def send_signal(self, sig):
                    pass

            names = iter(["child-a", "child-b"])

            def launch(*args, **kwargs):
                name = next(names)
                events.append(f"launch-{name}")
                launches.append(kwargs)
                return FakeProcess(name)

            job_specs = [
                (
                    {"job_name": name},
                    {"job_name": name, "model": {"id": "model"}},
                )
                for name in ("child-a", "child-b")
            ]
            summary = {
                "passed_tasks": 2,
                "total_tasks": 2,
                "pass_rate": 1.0,
            }
            with (
                mock.patch.object(terminal_bench, "ROOT", root),
                mock.patch.object(terminal_bench, "JOBS_DIR", jobs_dir),
                mock.patch.object(terminal_bench, "harbor_command", return_value=["harbor"]),
                mock.patch.object(terminal_bench, "command_environment", return_value={}),
                mock.patch.object(terminal_bench.subprocess, "Popen", side_effect=launch),
                mock.patch.object(terminal_bench, "summarize_job"),
                mock.patch.object(
                    terminal_bench.result_store,
                    "export_job",
                    return_value=(root / "results", summary),
                ),
            ):
                return_code, exported = terminal_bench.execute_harbor_jobs(
                    jobs=job_specs,
                    results_root=root / "results",
                    runtime="podman",
                    merge_existing_attempts=False,
                )

            self.assertEqual(return_code, 0)
            self.assertTrue(exported)
            self.assertEqual(
                events,
                ["launch-child-a", "launch-child-b", "wait-child-a", "wait-child-b"],
            )
            self.assertEqual((jobs_dir / "child-a").stat().st_mode & 0o777, 0o700)
            self.assertEqual((jobs_dir / "child-b").stat().st_mode & 0o777, 0o700)
            self.assertEqual(len(launches), 2)
            for launched in launches:
                self.assertIs(launched["stderr"], terminal_bench.subprocess.STDOUT)
                self.assertTrue(launched["stdout"].closed)

    def test_interrupt_cleanup_removes_only_matching_trial_containers(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs_dir = Path(directory) / "jobs"
            (jobs_dir / "job-a" / "task-a__AbC123").mkdir(parents=True)
            (jobs_dir / "job-a" / "unrelated__Xyz789").mkdir()
            config = {
                "job_name": "job-a",
                "environment": {"delete": True},
                "datasets": [{"task_names": ["task-a"]}],
            }
            listed = mock.Mock(
                returncode=0,
                stdout=(
                    "container-a\ttask-a__abc123__env\n"
                    "container-b\tunrelated__xyz789__env\n"
                ),
                stderr="",
            )
            removed = mock.Mock(returncode=0, stdout="container-a\n", stderr="")
            with (
                mock.patch.object(terminal_bench, "JOBS_DIR", jobs_dir),
                mock.patch.object(
                    terminal_bench.shutil, "which", return_value="/usr/bin/docker"
                ),
                mock.patch.object(
                    terminal_bench, "command_environment", return_value={}
                ),
                mock.patch.object(
                    terminal_bench.subprocess,
                    "run",
                    side_effect=[listed, removed],
                ) as run,
            ):
                terminal_bench.cleanup_interrupted_containers(
                    [config], runtime="podman"
                )

            self.assertEqual(run.call_count, 2)
            self.assertEqual(
                run.call_args_list[1].args[0],
                ["/usr/bin/docker", "rm", "-f", "container-a"],
            )

    def test_discard_cancelled_trials_preserves_real_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            job_dir = Path(directory)
            cancelled = job_dir / "task-a__cancelled"
            timeout = job_dir / "task-b__timeout"
            failed = job_dir / "task-c__failed"
            for trial_dir in (cancelled, timeout, failed):
                trial_dir.mkdir()
            terminal_bench.result_store.write_json(
                cancelled / "result.json",
                {
                    "trial_name": cancelled.name,
                    "exception_info": {"exception_type": "CancelledError"},
                    "agent_result": {
                        "n_input_tokens": 10,
                        "n_cache_tokens": 4,
                        "n_output_tokens": 3,
                    },
                },
            )
            terminal_bench.result_store.write_json(
                timeout / "result.json",
                {"exception_info": {"exception_type": "AgentTimeoutError"}},
            )
            terminal_bench.result_store.write_json(
                failed / "result.json",
                {
                    "exception_info": None,
                    "verifier_result": {"rewards": {"reward": 0}},
                },
            )
            terminal_bench.result_store.write_json(
                job_dir / "result.json",
                {
                    "n_total_trials": 3,
                    "finished_at": None,
                    "stats": {
                        "n_completed_trials": 2,
                        "n_errored_trials": 1,
                        "n_cancelled_trials": 1,
                        "n_pending_trials": 1,
                        "n_input_tokens": 30,
                        "n_cache_tokens": 14,
                        "n_output_tokens": 8,
                        "evals": {
                            "benchmark": {
                                "n_trials": 1,
                                "n_errors": 1,
                                "metrics": [{"mean": 0.5}],
                                "reward_stats": {
                                    "reward": {"1.0": [failed.name]}
                                },
                                "exception_stats": {
                                    "CancelledError": [cancelled.name]
                                },
                            }
                        },
                    },
                },
            )

            removed = terminal_bench.discard_cancelled_trials(job_dir)

            self.assertEqual(removed, ["task-a__cancelled"])
            self.assertFalse(cancelled.exists())
            self.assertTrue(timeout.exists())
            self.assertTrue(failed.exists())
            aggregate = terminal_bench.result_store.read_json(job_dir / "result.json")
            self.assertEqual(aggregate["stats"]["n_completed_trials"], 1)
            self.assertEqual(aggregate["stats"]["n_errored_trials"], 0)
            self.assertEqual(aggregate["stats"]["n_cancelled_trials"], 0)
            self.assertEqual(aggregate["stats"]["n_pending_trials"], 2)
            self.assertEqual(aggregate["stats"]["n_input_tokens"], 20)
            self.assertEqual(aggregate["stats"]["n_cache_tokens"], 10)
            self.assertEqual(aggregate["stats"]["n_output_tokens"], 5)
            eval_stats = aggregate["stats"]["evals"]["benchmark"]
            self.assertEqual(eval_stats["n_errors"], 0)
            self.assertEqual(eval_stats["metrics"], [{"mean": 1.0}])
            self.assertNotIn("exception_stats", eval_stats)

    def test_live_status_shows_progress_active_task_and_elapsed_time(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs_dir = Path(directory)
            job_dir = jobs_dir / "job-a"
            job_dir.mkdir()
            (job_dir / "result.json").write_text(
                json.dumps(
                    {
                        "n_total_trials": 2,
                        "stats": {
                            "n_completed_trials": 1,
                            "n_running_trials": 1,
                            "n_pending_trials": 0,
                            "n_errored_trials": 0,
                            "evals": {
                                "terminal-bench": {
                                    "reward_stats": {
                                        "reward": {"1.0": ["task-a"]}
                                    }
                                }
                            },
                        },
                    }
                )
            )
            config = {
                "job_name": "job-a",
                "datasets": [{"task_names": ["task-a", "task-b"]}],
            }
            meta = {"endpoint": "http://localhost:8000/v1"}
            with (
                mock.patch.object(terminal_bench, "JOBS_DIR", jobs_dir),
                mock.patch.object(
                    terminal_bench,
                    "active_trial",
                    return_value=("task-b", 125),
                ),
            ):
                status = terminal_bench.child_live_status(config, meta, None)

            self.assertIn("1/2", status)
            self.assertIn("pass 1/1 (100%)", status)
            self.assertIn("task-b 2:05", status)
            self.assertIn("http://localhost:8000/v1", status)
            with mock.patch.object(terminal_bench, "JOBS_DIR", jobs_dir):
                overall = terminal_bench.overall_live_status([config])
            self.assertIn("current pass 1/1 (100%)", overall)

    def test_resumed_dashboard_includes_preserved_campaign_progress(self):
        child_counts = [
            {
                "total": 7,
                "completed": 2,
                "running": 1,
                "pending": 4,
                "errors": 0,
                "passed": 1,
                "graded": 2,
            },
            {
                "total": 7,
                "completed": 1,
                "running": 1,
                "pending": 5,
                "errors": 0,
                "passed": 1,
                "graded": 1,
            },
        ]
        baseline = {
            "total": 19,
            "completed": 5,
            "passed": 5,
            "graded": 5,
            "errors": 0,
        }
        with mock.patch.object(
            terminal_bench, "live_job_counts", side_effect=child_counts
        ):
            status = terminal_bench.overall_live_status(
                [{"job_name": "a"}, {"job_name": "b"}], baseline
            )

        self.assertIn("8/19", status)
        self.assertIn("current pass 7/8 (88%)", status)
        self.assertIn("2 running", status)
        self.assertIn("9 pending", status)

    def test_job_is_terminal_rejects_live_result_and_accepts_both_counter_schemas(self):
        live = {
            "n_total_trials": 1,
            "stats": {
                "n_completed_trials": 0,
                "n_errored_trials": 0,
                "n_running_trials": 1,
                "n_pending_trials": 0,
                "n_cancelled_trials": 0,
            },
        }
        self.assertFalse(terminal_bench.harbor_result_is_terminal(live))

        inclusive = {
            "n_total_trials": 20,
            "stats": {
                "n_completed_trials": 20,
                "n_errored_trials": 2,
                "n_running_trials": 0,
                "n_pending_trials": 0,
                "n_cancelled_trials": 0,
            },
        }
        self.assertTrue(terminal_bench.harbor_result_is_terminal(inclusive))

        exclusive = {
            "n_total_trials": 20,
            "stats": {
                "n_completed_trials": 17,
                "n_errored_trials": 2,
                "n_running_trials": 0,
                "n_pending_trials": 0,
                "n_cancelled_trials": 1,
            },
        }
        self.assertTrue(terminal_bench.harbor_result_is_terminal(exclusive))

        with tempfile.TemporaryDirectory() as directory:
            job_dir = Path(directory)
            (job_dir / "result.json").write_text(json.dumps(live))
            self.assertFalse(terminal_bench.job_is_terminal(job_dir))

    def test_resume_orchestrator_resumes_a_child_with_live_result_json(self):
        live = {
            "n_total_trials": 1,
            "stats": {
                "n_completed_trials": 0,
                "n_errored_trials": 0,
                "n_running_trials": 1,
                "n_pending_trials": 0,
                "n_cancelled_trials": 0,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs_dir = root / "jobs"
            parent_dir = jobs_dir / "group"
            child_dir = jobs_dir / "child-a"
            parent_dir.mkdir(parents=True)
            child_dir.mkdir()
            (parent_dir / "orchestrator.json").write_text(
                json.dumps({"rounds": {"1": ["child-a"]}})
            )
            meta = {"job_name": "child-a"}
            (child_dir / "runner-meta.json").write_text(json.dumps(meta))
            (child_dir / "config.json").write_text(
                json.dumps({"job_name": "child-a"})
            )
            (child_dir / "result.json").write_text(json.dumps(live))

            with (
                mock.patch.object(terminal_bench, "JOBS_DIR", jobs_dir),
                mock.patch.object(
                    terminal_bench,
                    "resume_harbor_job",
                    return_value=(1, False, meta),
                ) as resume_job,
                mock.patch.object(terminal_bench.result_store, "export_job") as export,
            ):
                return_code = terminal_bench.resume_orchestrator_job(
                    parent_dir=parent_dir,
                    results_root=root / "results",
                    runtime="podman",
                )

            self.assertEqual(return_code, 1)
            resume_job.assert_called_once_with(
                job_dir=child_dir,
                results_root=root / "results",
                runtime="podman",
            )
            export.assert_not_called()

    def test_resume_can_redistribute_only_unfinished_single_job_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs_dir = root / "jobs"
            job_dir = jobs_dir / "group"
            job_dir.mkdir(parents=True)
            profile = {
                "model_id": "model",
                "model_metadata": {"id": "model"},
                "endpoint": "http://host-a/v1",
                "agent": {"context_length": 262144},
            }
            meta = {
                "job_name": "group",
                "attempt_group": "group",
                "attempt_round": 1,
                "max_attempts": 2,
                "requested_tasks": ["task-a", "task-b", "task-c", "task-d"],
                "executed_tasks": ["task-a", "task-b", "task-c", "task-d"],
                "endpoint": "http://host-a/v1",
                "endpoints": ["http://host-a/v1"],
                "model": {"id": "model", "endpoint_metadata": {"id": "model"}},
                "evaluation_profile": profile,
                "profile_hash": terminal_bench.result_store.evaluation_profile_hash(profile),
            }
            config = {
                "job_name": "group",
                "n_concurrent_trials": 1,
                "environment": {"delete": True},
                "agents": [
                    {
                        "kwargs": {
                            "api_base": "http://host-a/v1",
                            "llm_kwargs": {"api_key": "stored"},
                        }
                    }
                ],
                "datasets": [
                    {"path": "/tasks", "task_names": ["task-a", "task-b", "task-c", "task-d"]}
                ],
            }
            terminal_bench.result_store.write_json(job_dir / "runner-meta.json", meta)
            terminal_bench.result_store.write_json(job_dir / "config.json", config)
            for task in ("task-a", "task-b"):
                trial = job_dir / f"{task}__finished"
                trial.mkdir()
                terminal_bench.result_store.write_json(
                    trial / "result.json",
                    {"task_name": task, "trial_name": trial.name},
                )
            interrupted = job_dir / "task-c__interrupted"
            interrupted.mkdir()
            terminal_bench.result_store.write_json(
                interrupted / "result.json",
                {
                    "task_name": "task-c",
                    "trial_name": interrupted.name,
                    "exception_info": {"exception_type": "CancelledError"},
                },
            )
            new_endpoints = ["http://host-a/v1", "http://host-b/v1"]
            with (
                mock.patch.object(terminal_bench, "JOBS_DIR", jobs_dir),
                mock.patch.object(
                    terminal_bench, "harbor_job_process_is_live", return_value=False
                ),
                mock.patch.object(terminal_bench, "validate_resume_endpoints") as validate,
                mock.patch.object(
                    terminal_bench.result_store,
                    "export_job",
                    return_value=(root / "results" / "model", {"passed_tasks": 2, "total_tasks": 2, "pass_rate": 1.0}),
                ) as export,
                mock.patch.object(
                    terminal_bench,
                    "execute_attempt_round",
                    return_value=(0, True),
                ) as execute,
                mock.patch.object(
                    terminal_bench,
                    "continue_conditional_attempts",
                    return_value=0,
                ) as continue_attempts,
            ):
                return_code = terminal_bench.resume_with_endpoint_redistribution(
                    job_dir=job_dir,
                    endpoints=new_endpoints,
                    results_root=root / "results",
                    runtime="podman",
                    api_key="new-key",
                    skip_endpoint_check=False,
                    concurrency=None,
                )

            self.assertEqual(return_code, 0)
            self.assertFalse(interrupted.exists())
            validate.assert_called_once_with(
                meta, new_endpoints, "new-key", skip_check=False
            )
            export.assert_called_once()
            execute_kwargs = execute.call_args.kwargs
            self.assertEqual(execute_kwargs["tasks"], ["task-c", "task-d"])
            self.assertEqual(
                execute_kwargs["campaign_progress"],
                {
                    "total": 4,
                    "completed": 2,
                    "passed": 0,
                    "graded": 0,
                    "errors": 0,
                },
            )
            self.assertEqual(execute_kwargs["attempt"], 1)
            self.assertTrue(execute_kwargs["merge_existing_attempts"])
            self.assertEqual(execute_kwargs["meta"]["endpoints"], new_endpoints)
            self.assertEqual(execute_kwargs["meta"]["profile_hash"], meta["profile_hash"])
            self.assertEqual(
                execute_kwargs["meta"]["evaluation_profile"],
                meta["evaluation_profile"],
            )
            self.assertEqual(
                execute_kwargs["base_config"]["agents"][0]["kwargs"]["llm_kwargs"]["api_key"],
                "new-key",
            )
            continue_attempts.assert_called_once()

    def test_resume_endpoint_override_parser_accepts_multiple_urls(self):
        args = terminal_bench.parse_args(
            [
                "resume",
                "jobs/example",
                "--endpoints",
                "http://host-a/v1,http://host-b/v1",
                "--concurrency",
                "2",
            ]
        )
        self.assertEqual(args.endpoints, ["http://host-a/v1,http://host-b/v1"])
        self.assertEqual(args.concurrency, 2)

    def test_resume_redistribution_requires_an_endpoint(self):
        with self.assertRaisesRegex(
            terminal_bench.RunnerError, "requires at least one"
        ):
            terminal_bench.resume_with_endpoint_redistribution(
                job_dir=Path("jobs/example"),
                endpoints=[],
                results_root=Path("results"),
                runtime="podman",
                api_key=None,
                skip_endpoint_check=False,
                concurrency=None,
            )

    def test_single_replacement_endpoint_uses_a_new_child_job(self):
        meta = {
            "job_name": "group",
            "attempt_group": "group",
            "endpoints": ["http://host-b/v1"],
            "topology_migration": {"source_job": "group"},
        }
        config = {
            "job_name": "group",
            "agents": [{"kwargs": {"api_base": "http://host-a/v1"}}],
            "datasets": [{"path": "/tasks", "task_names": ["task-a"]}],
        }

        jobs = terminal_bench.build_attempt_jobs(
            meta=meta, base_config=config, tasks=["task-a"], attempt=1
        )

        self.assertEqual(jobs[0][0]["job_name"], "group-resumed")
        self.assertEqual(
            jobs[0][0]["agents"][0]["kwargs"]["api_base"],
            "http://host-b/v1",
        )

    def test_migrated_single_endpoint_round_is_recorded_as_an_orchestrator(self):
        jobs = [
            (
                {"job_name": "group-resumed"},
                {
                    "endpoints": ["http://host-b/v1"],
                    "topology_migration": {"source_job": "group"},
                },
            )
        ]
        meta = {
            "job_name": "group",
            "endpoints": ["http://host-b/v1"],
            "topology_migration": {"source_job": "group"},
        }
        with (
            mock.patch.object(terminal_bench, "build_attempt_jobs", return_value=jobs),
            mock.patch.object(
                terminal_bench, "record_orchestrator_round", return_value=Path("jobs/group")
            ) as record,
            mock.patch.object(
                terminal_bench, "execute_harbor_jobs", return_value=(0, True)
            ) as execute,
        ):
            result = terminal_bench.execute_attempt_round(
                meta=meta,
                base_config={"datasets": []},
                tasks=["task-a"],
                attempt=1,
                results_root=Path("results"),
                runtime="podman",
                merge_existing_attempts=True,
            )

        self.assertEqual(result, (0, True))
        record.assert_called_once()
        execute.assert_called_once()

    def test_jobs_command_folds_retry_children_into_readable_campaign(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs_dir = Path(directory) / "jobs"
            parent = jobs_dir / "campaign"
            retry = jobs_dir / "campaign-attempt2"
            parent.mkdir(parents=True)
            retry.mkdir()
            profile = {
                "agent": {"context_length": 262144},
                "agent_timeout_seconds": 10800,
                "engine": "llama.cpp",
                "backend": "rocm",
                "quant": "Q4",
            }
            base_meta = {
                "job_name": "campaign",
                "attempt_group": "campaign",
                "attempt_round": 1,
                "created_at": "2026-09-05T09:00:00+00:00",
                "requested_tasks": ["task-a", "task-b"],
                "max_attempts": 2,
                "suite": {"id": "core19"},
                "tier": "full",
                "model": {"id": "served-model", "name": "Readable Model"},
                "platform": {"id": "strix-halo", "name": "Strix Halo"},
                "engine": "llama.cpp",
                "backend": "rocm",
                "backend_version": "10.0",
                "quant": "Q4",
                "inference_profile": "mtp",
                "endpoint": "http://host-a/v1",
                "endpoints": ["http://host-a/v1"],
                "evaluation_profile": profile,
            }
            config = {
                "job_name": "campaign",
                "n_concurrent_trials": 1,
                "environment": {"delete": True},
                "agents": [{"kwargs": {"llm_kwargs": {"api_key": "local"}}}],
                "datasets": [{"path": "/tasks", "task_names": ["task-a", "task-b"]}],
            }
            terminal_bench.result_store.write_json(parent / "runner-meta.json", base_meta)
            terminal_bench.result_store.write_json(parent / "config.json", config)
            retry_meta = {**base_meta, "job_name": retry.name, "attempt_round": 2}
            terminal_bench.result_store.write_json(retry / "runner-meta.json", retry_meta)
            terminal_bench.result_store.write_json(retry / "config.json", config)
            for job_dir, task, reward in (
                (parent, "task-a", 1),
                (parent, "task-b", 0),
                (retry, "task-b", 1),
            ):
                trial = job_dir / f"{task}__trial"
                trial.mkdir()
                terminal_bench.result_store.write_json(
                    trial / "result.json",
                    {
                        "task_name": task,
                        "trial_name": trial.name,
                        "finished_at": "2026-09-05T10:00:00+00:00",
                        "verifier_result": {"rewards": {"reward": reward}},
                    },
                )

            with (
                mock.patch.object(terminal_bench, "JOBS_DIR", jobs_dir),
                mock.patch.object(terminal_bench, "live_harbor_jobs", return_value=set()),
            ):
                campaigns = terminal_bench.discover_job_campaigns()

            self.assertEqual(len(campaigns), 1)
            campaign = campaigns[0]
            self.assertEqual(campaign["name"], "campaign")
            self.assertEqual(campaign["status"], "COMPLETE")
            self.assertEqual(campaign["completed_tasks"], 2)
            self.assertEqual(campaign["passed_tasks"], 2)
            self.assertEqual(campaign["model_name"], "Readable Model")
            self.assertEqual(campaign["quant"], "Q4")

    def test_rerun_arguments_reconstruct_current_job_identity(self):
        campaign = {
            "meta": {
                "suite": {"id": "core19"},
                "tier": "full",
                "requested_tasks": ["task-a"],
                "model": {"id": "served", "name": "Readable"},
                "platform": {"id": "strix-halo", "name": "Strix Halo"},
                "tag": None,
                "evaluation_profile": {
                    "agent_timeout_seconds": 10800,
                },
            },
            "config": {
                "n_concurrent_trials": 1,
                "environment": {"delete": True},
                "agents": [{"kwargs": {"llm_kwargs": {"api_key": "local"}}}],
            },
            "endpoints": ["http://host-a/v1"],
            "engine": "llama.cpp",
            "engine_version": None,
            "backend": "rocm",
            "backend_version": "10.0",
            "quant": "Q4",
            "inference_profile": "mtp",
            "context_length": 262144,
            "max_attempts": 2,
        }

        argv = terminal_bench.rerun_arguments(campaign)

        self.assertEqual(argv[:3], ["run", "--tier", "full"])
        self.assertIn("--rerun", argv)
        self.assertEqual(argv[argv.index("--platform") + 1], "strix-halo")
        self.assertEqual(argv[argv.index("--quant") + 1], "Q4")
        self.assertNotIn("--suite", argv)

    def test_migrated_single_job_can_become_orchestrator_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs_dir = Path(directory) / "jobs"
            parent = jobs_dir / "group"
            parent.mkdir(parents=True)
            (parent / "runner-meta.json").write_text("{}")
            (parent / "config.json").write_text("{}")
            migration = {"source_job": "group"}
            jobs = [
                (
                    {"job_name": "group-endpoint1"},
                    {
                        "endpoints": ["http://host-a/v1", "http://host-b/v1"],
                        "topology_migration": migration,
                    },
                )
            ]
            with mock.patch.object(terminal_bench, "JOBS_DIR", jobs_dir):
                resolved = terminal_bench.record_orchestrator_round(
                    group="group",
                    endpoints=["http://host-a/v1", "http://host-b/v1"],
                    attempt=1,
                    jobs=jobs,
                    results_root=Path(directory) / "results",
                    datasets=[{"path": "/tasks"}],
                )

            self.assertEqual(resolved, parent)
            manifest = json.loads((parent / "orchestrator.json").read_text())
            self.assertEqual(manifest["topology_migration"], migration)
            self.assertEqual(manifest["rounds"]["1"], ["group-endpoint1"])

    def test_multi_endpoint_discovery_requires_matching_metadata(self):
        endpoints = ["http://host-a/v1", "http://host-b/v1"]
        matching = {
            endpoints[0]: ("model", {"id": "model", "meta": {"n_ctx": 262144}}),
            endpoints[1]: ("model", {"id": "model", "meta": {"n_ctx": 262144}}),
        }
        with mock.patch.object(
            terminal_bench,
            "discover_model",
            side_effect=lambda endpoint, api_key, preferred=None: matching[endpoint],
        ):
            model, metadata, context = terminal_bench.discover_matching_model(
                endpoints, "local", preferred="model"
            )
        self.assertEqual((model, context), ("model", 262144))
        self.assertEqual(metadata, matching[endpoints[0]][1])

        mismatched = dict(matching)
        mismatched[endpoints[1]] = (
            "model",
            {"id": "model", "meta": {"n_ctx": 131072}},
        )
        with mock.patch.object(
            terminal_bench,
            "discover_model",
            side_effect=lambda endpoint, api_key, preferred=None: mismatched[endpoint],
        ):
            with self.assertRaisesRegex(terminal_bench.RunnerError, "context tokens"):
                terminal_bench.discover_matching_model(
                    endpoints, "local", preferred="model"
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
            terminal_bench.result_tag(
                "UD-Q4_K_XL", "llama.cpp", "vulkan", "mtp", "long-context"
            ),
            "llama.cpp-vulkan-UD-Q4_K_XL-mtp-long-context-terminus-2",
        )

    def test_run_identity_keeps_quant_profile_and_tag_separate(self):
        args = terminal_bench.parse_args(
            [
                "run",
                "--platform",
                "test-local",
                "--model-name",
                "DeepSeek-V4-Flash-0731",
                "--engine",
                "DwarfStar",
                "--backend",
                "rocm",
                "--quant",
                "IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8",
                "--inference-profile",
                "DSpark",
                "--tag",
                "chat-v2-imatrix-0731",
            ]
        )
        self.assertEqual(
            terminal_bench.run_identity_args(args),
            (
                "DeepSeek-V4-Flash-0731",
                "IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8",
                "DSpark",
                "chat-v2-imatrix-0731",
            ),
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

    def test_quant_profile_and_tag_are_preserved_in_job_names(self):
        quant = "IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8"
        job_name = terminal_bench.make_job_name(
            "full",
            "deepseek-v4-flash",
            quant,
            engine="llama.cpp",
            backend="vulkan",
            inference_profile="DSpark",
            tag="chat-v2-imatrix-0731",
        )
        self.assertIn(quant, job_name)
        self.assertIn("DSpark", job_name)
        self.assertIn("chat-v2-imatrix-0731", job_name)
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
            quant="UD-Q4_K_XL",
            inference_profile=None,
            tag=None,
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
        self.assertNotIn("endpoints", profile)

        distributed = terminal_bench.evaluation_profile(
            tasks_dir=terminal_bench.DEFAULT_TASKS_DIR,
            model="model",
            model_metadata={},
            endpoint="http://host-b/v1",
            endpoints=["http://host-b/v1", "http://host-a/v1"],
            engine="llama.cpp",
            engine_version="b1234",
            backend="vulkan",
            backend_version="1.4.304",
            quant="UD-Q4_K_XL",
            inference_profile=None,
            tag=None,
            context_length=262144,
            agent_timeout_seconds=10800,
        )
        self.assertEqual(
            distributed["endpoints"],
            ["http://host-a/v1", "http://host-b/v1"],
        )

    def test_rocm_version_alias_only_applies_to_rocm_backend(self):
        args = terminal_bench.parse_args(
            [
                "run",
                "--platform",
                "test-local",
                "--model-name",
                "Test-Model",
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

    def test_retry_infers_and_requires_the_recorded_suite(self):
        core = terminal_bench.load_suite("core19")
        meta = {"suite": core.identity()}
        resolved = terminal_bench.suite_for_result(meta, None, None)
        self.assertEqual(resolved.identity(), core.identity())

        with self.assertRaisesRegex(
            terminal_bench.RunnerError, "does not match the result set"
        ):
            terminal_bench.suite_for_result(meta, "legacy-mini20", None)

        legacy = terminal_bench.suite_for_result({}, None, None)
        self.assertEqual(legacy.id, "legacy-mini20")
        explicit_legacy = terminal_bench.suite_for_result(
            {}, "legacy-mini20", None
        )
        self.assertEqual(explicit_legacy.id, "legacy-mini20")
        with self.assertRaisesRegex(
            terminal_bench.RunnerError, "Legacy unscoped results"
        ):
            terminal_bench.suite_for_result({}, "core19", None)

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
