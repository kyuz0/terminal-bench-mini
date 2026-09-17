import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import terminal_bench as runner


class TimeoutRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.jobs = self.root / "jobs"
        self.parent = self.jobs / "campaign"
        self.child = self.jobs / "campaign-endpoint1"
        self.parent.mkdir(parents=True)
        self.child.mkdir()
        self.tasks = ["pass", "timeout", "verifier", "endpoint", "cancelled", "unfinished"]
        self.profile = {"agent_timeout_seconds": 10800, "agent": {"context_length": 262144}}
        self.meta = {
            "job_name": self.child.name, "attempt_group": self.parent.name,
            "attempt_round": 1, "max_attempts": 2, "requested_tasks": self.tasks,
            "model": {"id": "glm", "name": "GLM"},
            "platform": {"id": "test", "name": "Test"},
            "quant": "Q2", "engine": "test", "backend": "cpu",
            "endpoint": "http://old/v1", "endpoints": ["http://old/v1"],
            "evaluation_profile": self.profile,
            "profile_hash": runner.result_store.evaluation_profile_hash(self.profile),
        }
        self.config = {
            "job_name": self.child.name, "n_concurrent_trials": 1,
            "datasets": [{"path": "/tasks", "task_names": self.tasks}],
            "agents": [{"override_timeout_sec": 10800, "kwargs": {
                "api_base": "http://old/v1", "llm_kwargs": {"api_key": "local"}}}],
        }
        self.manifest = {"attempt_group": self.parent.name,
                         "endpoints": ["http://old/v1"],
                         "datasets": self.config["datasets"],
                         "rounds": {"1": [self.child.name]}}
        self.write(self.parent / "orchestrator.json", self.manifest)
        self.write(self.child / "runner-meta.json", self.meta)
        self.write(self.child / "config.json", self.config)
        for task, reward, error in [
            ("pass", 1, None), ("timeout", 0, "AgentTimeoutError"),
            ("verifier", 0, None), ("endpoint", None, "ConnectionError"),
            ("cancelled", None, "CancelledError"),
        ]:
            self.write(self.child / task / "result.json", {
                "task_name": task, "task_id": {"path": f"/tasks/{task}"},
                "trial_name": task, "verifier_result": {"rewards": {"reward": reward}},
                "config": {"agent": {"override_timeout_sec": 10800}},
                "exception_info": {"exception_type": error} if error else None,
            })
        for name, value in [("JOBS_DIR", self.jobs), ("ROOT", self.root)]:
            patch = mock.patch.object(runner, name, value)
            patch.start()
            self.addCleanup(patch.stop)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def recover(self, **overrides):
        args = dict(parent_dir=self.parent, endpoints=["http://a/v1", "http://b/v1"],
                    agent_timeout=18000, results_root=self.root / "results",
                    runtime="test", api_key=None, concurrency=None)
        args.update(overrides)
        return runner.recover_orchestrator_job(**args)

    def test_recovery_preserves_evidence_failures_and_actual_timeout(self):
        before = {p: p.read_bytes() for p in self.child.rglob("*.json")}
        with mock.patch.object(runner, "harbor_job_process_is_live", return_value=False), \
             mock.patch.object(runner, "validate_resume_endpoints"), \
             mock.patch.object(runner, "execute_attempt_round", return_value=(1, False)) as execute:
            self.assertEqual(self.recover(), 1)
        call = execute.call_args.kwargs
        self.assertEqual(call["tasks"], ["timeout", "cancelled", "unfinished"])
        self.assertEqual(call["campaign_progress"], dict(total=6, completed=3, passed=1, graded=2, errors=1))
        self.assertEqual(call["meta"]["max_attempts"], 2)
        self.assertEqual(call["base_config"]["agents"][0]["override_timeout_sec"], 18000)
        self.assertNotEqual(call["meta"]["profile_hash"], self.meta["profile_hash"])
        self.assertEqual(before, {p: p.read_bytes() for p in self.child.rglob("*.json")})
        model_dir = runner.result_store.model_results_dir_for_run(self.root / "results", call["meta"])
        self.assertFalse((model_dir / "results-timeout.json").exists())
        self.assertFalse((model_dir / "results-cancelled.json").exists())
        passed = runner.result_store.read_json(model_dir / "results-pass.json")
        self.assertEqual(passed["attempts"][0]["agent_timeout_seconds"], 10800)
        self.assertEqual(passed["evaluation_profile"]["agent_timeout_seconds"], 18000)
        self.assertTrue((model_dir / "results-verifier.json").exists())
        self.assertTrue((model_dir / "results-endpoint.json").exists())
        generated = runner.build_attempt_jobs(meta=call["meta"], base_config=call["base_config"], tasks=call["tasks"], attempt=1)
        self.assertTrue(all("-recovery1-" in c["job_name"] for c, _ in generated))
        self.assertTrue(all(m["attempt_group"] == self.parent.name for _, m in generated))
        with mock.patch.object(runner, "live_harbor_jobs", return_value=set()):
            campaign = runner.discover_job_campaigns()[0]
        self.assertEqual(campaign["completed_tasks"], 3)

    def test_preflight_rejections_leave_campaign_untouched(self):
        before = (self.parent / "orchestrator.json").read_bytes()
        with mock.patch.object(runner, "harbor_job_process_is_live", return_value=True):
            with self.assertRaisesRegex(runner.RunnerError, "still running"):
                self.recover()
        with mock.patch.object(runner, "harbor_job_process_is_live", return_value=False), \
             mock.patch.object(runner, "validate_resume_endpoints", side_effect=runner.RunnerError("mismatch")):
            with self.assertRaisesRegex(runner.RunnerError, "mismatch"):
                self.recover()
            with self.assertRaisesRegex(runner.RunnerError, "exceed"):
                self.recover(agent_timeout=10800)
        self.assertEqual(before, (self.parent / "orchestrator.json").read_bytes())
        self.assertFalse((self.parent / "recovery-archive").exists())

    def test_second_round_and_pending_preparation_are_rejected(self):
        for change in [{"rounds": {"1": [self.child.name], "2": ["retry"]}}, {"prepared_recovery": {"tasks": ["timeout"]}}]:
            manifest = copy.deepcopy(self.manifest)
            manifest.update(change)
            self.write(self.parent / "orchestrator.json", manifest)
            with self.assertRaisesRegex(runner.RunnerError, "stopped first-round"):
                self.recover()

    def test_prepare_then_repeat_keeps_timeout_passes_without_reviving_old_failures(self):
        with mock.patch.object(runner, "harbor_job_process_is_live", return_value=False), \
             mock.patch.object(runner, "validate_resume_endpoints") as validate, \
             mock.patch.object(runner, "execute_attempt_round") as execute:
            self.assertEqual(self.recover(prepare_only=True), 0)
            execute.assert_not_called()
            validate.assert_not_called()
        manifest = runner.result_store.read_json(self.parent / "orchestrator.json")
        self.assertTrue(manifest["prepared_recovery"])

        def interrupted_round(**kwargs):
            for config, meta in kwargs["jobs"]:
                child = self.jobs / config["job_name"]
                self.write(child / "runner-meta.json", meta)
                self.write(child / "config.json", config)
                for task in runner.config_task_names(config):
                    if task == "cancelled":
                        continue
                    self.write(child / task / "result.json", {
                        "task_name": task, "task_id": {"path": f"/tasks/{task}"},
                        "trial_name": f"new-{task}",
                        "verifier_result": {"rewards": {"reward": 1 if task == "timeout" else 0}},
                        "exception_info": {"exception_type": "AgentTimeoutError"},
                        "config": {"agent": {"override_timeout_sec": 18000}},
                    })
            return 130, False

        with mock.patch.object(runner, "harbor_job_process_is_live", return_value=False), \
             mock.patch.object(runner, "validate_resume_endpoints") as validate, \
             mock.patch.object(runner, "execute_harbor_jobs", side_effect=interrupted_round):
            self.assertEqual(runner.resume_orchestrator_job(
                parent_dir=self.parent, results_root=self.root / "results", runtime="test"), 130)
            validate.assert_called_once()
        with mock.patch.object(runner, "harbor_job_process_is_live", return_value=False), \
             mock.patch.object(runner, "execute_harbor_jobs") as execute:
            self.assertEqual(self.recover(agent_timeout=25200, prepare_only=True), 0)
            execute.assert_not_called()
        meta = runner.load_resume_meta(self.parent)
        manifest = runner.result_store.read_json(self.parent / "orchestrator.json")
        self.assertEqual(meta["execution_suffix"], "-recovery2")
        self.assertEqual(manifest["prepared_recovery"]["tasks"], ["cancelled", "unfinished"])
        self.assertEqual(manifest["recovery"]["reset_tasks"], ["unfinished"])
        model_dir = runner.result_store.model_results_dir_for_run(self.root / "results", meta)
        result = runner.result_store.read_json(model_dir / "results-timeout.json")
        self.assertEqual(len(result["attempts"]), 1)
        self.assertTrue(result["passed"])
        self.assertEqual(result["attempts"][0]["agent_timeout_seconds"], 18000)
        self.assertTrue((model_dir / "results-pass.json").exists())
        self.assertFalse((model_dir / "results-unfinished.json").exists())
        self.assertTrue((self.parent / "recovery-archive-2" / "orchestrator.json").exists())
