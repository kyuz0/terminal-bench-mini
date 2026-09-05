import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("docs_build_data", REPO_ROOT / "docs" / "build_data.py")
build_data = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(build_data)


class DocsDataTests(unittest.TestCase):
    def test_time_estimates_accept_minutes_or_v4_hours(self):
        self.assertEqual(
            build_data.estimate_minutes({"expert_time_estimate_min": 45}, "expert"),
            45,
        )
        self.assertEqual(
            build_data.estimate_minutes({"expert_time_estimate_hours": 1.5}, "expert"),
            90,
        )
        self.assertIsNone(build_data.estimate_minutes({}, "expert"))

    def test_verifier_excerpt_keeps_concise_failure_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "test-stdout.txt"
            output.write_text(
                "noise\n"
                "E   AssertionError: expected a valid query\n"
                "E   assert False\n"
                "FAILED tests/test_outputs.py::test_query - AssertionError\n"
            )
            self.assertEqual(
                build_data.verifier_excerpt(output),
                [
                    "AssertionError: expected a valid query",
                    "assert False",
                    "FAILED tests/test_outputs.py::test_query - AssertionError",
                ],
            )

    def test_zero_token_agent_timeout_is_classified_as_endpoint_stall(self):
        outcome, reason = build_data.classify_attempt(
            {
                "passed": False,
                "tokens": {"input": 0, "output": 0},
                "agent_steps": 0,
                "exception": {
                    "exception_type": "AgentTimeoutError",
                    "exception_message": "Agent execution timed out after 10800.0 seconds",
                },
            },
            [],
        )
        self.assertEqual(outcome, "endpoint-stall")
        self.assertIn("before the endpoint returned model tokens", reason)

    def test_dataset_preserves_identity_and_prioritizes_within_attempts(self):
        dataset = build_data.build_dataset(
            REPO_ROOT, "kyuz0/terminal-bench-local"
        )
        self.assertEqual(dataset["benchmark"]["taskCount"], 19)
        self.assertEqual(
            dataset["benchmark"]["defaultMetric"], "pass-within-attempts"
        )
        self.assertEqual(len(dataset["tasks"]), 19)
        self.assertGreaterEqual(len(dataset["models"]), 1)
        for model in dataset["models"]:
            self.assertIn("platform", model)
            self.assertIn("engine", model)
            self.assertIn("backend", model)
            self.assertIn("backendVersion", model)
            self.assertIn("quant", model)
            self.assertIn("inferenceProfile", model)
            self.assertIn("tag", model)
            self.assertEqual(model["passAt1"], sum(result["passAt1"] for result in model["results"]))
            self.assertLessEqual(model["passAt1"], model["passedWithinAttempts"])

        ordering = [
            (
                -model["passWithinAttemptsRate"],
                -model["passAt1Rate"],
                str(model["name"]).lower(),
            )
            for model in dataset["models"]
        ]
        self.assertEqual(ordering, sorted(ordering))

        dwarfstar_runs = [
            model for model in dataset["models"] if model["engine"] == "DwarfStar"
        ]
        self.assertTrue(dwarfstar_runs)
        self.assertTrue(all(model["backend"] == "rocm" for model in dwarfstar_runs))
        self.assertEqual(
            {model["backendVersion"] for model in dwarfstar_runs}, {"7.14", "10.0"}
        )

        qwen_baseline = next(
            model
            for model in dataset["models"]
            if model["name"] == "Qwen3.6-35B-A3B"
        )
        self.assertEqual(qwen_baseline["quant"], "UD-Q4_K_XL")
        self.assertEqual(qwen_baseline["engine"], "llama.cpp")
        self.assertEqual(qwen_baseline["backend"], "rocm")
        self.assertEqual(qwen_baseline["backendVersion"], "7.14")

        deepseek_runs = [
            model for model in dataset["models"] if model["name"] == "DeepSeek-V4-Flash-0731"
        ]
        self.assertEqual(len(deepseek_runs), 3)
        self.assertEqual(
            {model["quant"] for model in deepseek_runs},
            {"MXFP4", "UD-IQ3_XXS", "IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8"},
        )
        dspark = next(model for model in deepseek_runs if model["inferenceProfile"] == "DSpark")
        self.assertEqual(dspark["tag"], "chat-v2-imatrix-0731")

        qwen38_runs = [model for model in dataset["models"] if model["name"] == "Qwen3.8-27B"]
        self.assertEqual(len(qwen38_runs), 3)

    def test_model_record_filters_results_to_the_selected_tier(self):
        result_dir = next(
            (REPO_ROOT / "results" / "suites").glob("core19-*/*/*_results")
        )
        complete = build_data.model_record(
            REPO_ROOT, result_dir, "kyuz0/terminal-bench-mini"
        )
        task_id = complete["results"][0]["taskId"]
        filtered = build_data.model_record(
            REPO_ROOT,
            result_dir,
            "kyuz0/terminal-bench-mini",
            selected_tasks={task_id},
        )
        self.assertEqual(filtered["totalTasks"], 1)
        self.assertEqual([row["taskId"] for row in filtered["results"]], [task_id])

    def test_model_record_does_not_expose_internal_migration_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            result_dir = repo_root / "results" / "test" / "model_results"
            result_dir.mkdir(parents=True)
            (result_dir / "summary.json").write_text(
                '{"model":{"name":"test"},"results":[]}'
            )
            (result_dir / "run-meta.json").write_text(
                """{
                  "model": {"name": "test", "id": "test"},
                  "platform": {"id": "test", "name": "test"},
                  "historical_projection": {
                    "kind": "legacy-mini20-to-core19",
                    "status": "complete",
                    "source_result_directory": "results/source/model_results"
                  }
                }"""
            )
            model = build_data.model_record(
                repo_root, result_dir, "kyuz0/terminal-bench-mini"
            )

        self.assertNotIn("historicalProjection", model)

    def test_dataset_includes_task_instructions_but_never_solution_paths(self):
        dataset = build_data.build_dataset(
            REPO_ROOT, "kyuz0/terminal-bench-local", None
        )
        self.assertTrue(all(task["instruction"] for task in dataset["tasks"]))
        serialized = str(dataset)
        self.assertNotIn("/solution/", serialized)
        self.assertNotIn("tasks/solution", serialized)

    def test_default_dataset_is_core19(self):
        dataset = build_data.build_dataset(REPO_ROOT, "kyuz0/terminal-bench-mini")
        self.assertEqual(dataset["benchmark"]["suite"]["id"], "core19")
        self.assertEqual(dataset["benchmark"]["taskCount"], 19)
        self.assertEqual(len(dataset["tasks"]), 19)

        with mock.patch("sys.argv", ["build_data.py"]):
            self.assertEqual(build_data.parse_args().suite, "core19")

if __name__ == "__main__":
    unittest.main()
