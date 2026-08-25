import importlib.util
from pathlib import Path
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("docs_build_data", REPO_ROOT / "docs" / "build_data.py")
build_data = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(build_data)


class DocsDataTests(unittest.TestCase):
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
        dataset = build_data.build_dataset(REPO_ROOT, "kyuz0/terminal-bench-local")
        self.assertEqual(dataset["benchmark"]["taskCount"], 20)
        self.assertEqual(
            dataset["benchmark"]["defaultMetric"], "pass-within-attempts"
        )
        self.assertEqual(len(dataset["tasks"]), 20)
        self.assertGreaterEqual(len(dataset["models"]), 1)
        for model in dataset["models"]:
            self.assertIn("platform", model)
            self.assertIn("engine", model)
            self.assertIn("backend", model)
            self.assertIn("backendVersion", model)
            self.assertIn("modelTag", model)
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
        self.assertTrue(
            all(model["backendVersion"] == "7.14" for model in dwarfstar_runs)
        )

        qwen_baseline = next(
            model
            for model in dataset["models"]
            if model["name"] == "Qwen3.6-35B-A3B-UD-Q4_K_XL"
        )
        self.assertEqual(qwen_baseline["engine"], "llama.cpp")
        self.assertEqual(qwen_baseline["backend"], "rocm")
        self.assertEqual(qwen_baseline["backendVersion"], "7.14")

    def test_dataset_includes_task_instructions_but_never_solution_paths(self):
        dataset = build_data.build_dataset(REPO_ROOT, "kyuz0/terminal-bench-local")
        self.assertTrue(all(task["instruction"] for task in dataset["tasks"]))
        serialized = str(dataset)
        self.assertNotIn("/solution/", serialized)
        self.assertNotIn("tasks/solution", serialized)


if __name__ == "__main__":
    unittest.main()
