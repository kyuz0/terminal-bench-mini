from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path
import json
import os
import tempfile
import unittest
from unittest import mock


SHIM = Path(__file__).parents[1] / "compat" / "podman" / "docker"
SPEC = spec_from_loader("podman_docker_shim", SourceFileLoader("podman_docker_shim", str(SHIM)))
assert SPEC and SPEC.loader
shim = module_from_spec(SPEC)
SPEC.loader.exec_module(shim)


class PodmanDockerShimTests(unittest.TestCase):
    def test_removes_project_directory_and_preserves_other_arguments(self):
        arguments, directory = shim.translated(
            [
                "compose",
                "--project-name",
                "trial",
                "--project-directory",
                "/task/environment",
                "-f",
                "/tmp/base.yaml",
                "up",
                "--wait",
            ]
        )
        self.assertEqual(directory, Path("/task/environment"))
        self.assertEqual(
            arguments,
            ["compose", "--project-name", "trial", "-f", "/tmp/base.yaml", "up", "--wait"],
        )

    def test_non_compose_command_is_unchanged(self):
        arguments = ["info", "--format", "{{.OSType}}"]
        self.assertEqual(shim.translated(arguments), (arguments, None))

    def test_translates_compose_cp_to_container_cp(self):
        arguments = [
            "compose",
            "--project-name",
            "trial",
            "-f",
            "/tmp/base.yaml",
            "cp",
            "/tmp/source",
            "main:/tmp/target",
        ]
        self.assertEqual(
            shim.translated_compose_cp(arguments),
            ["cp", "/tmp/source", "trial_main_1:/tmp/target"],
        )

    def test_adds_host_network_only_for_agent_image(self):
        arguments = ["compose", "--project-name", "trial", "-f", "/tmp/base.yaml", "up"]
        with mock.patch.dict(
            os.environ,
            {
                "TBENCH_AGENT_NETWORK_MODE": "host",
                "CONTEXT_DIR": "/tmp/trial/agent-build-context",
            },
        ):
            translated = shim.add_agent_network_overlay(arguments)
        self.assertEqual(translated[-3], "-f")
        self.assertEqual(Path(translated[-2]).name, "agent-network-host.json")
        self.assertEqual(translated[-1], "up")

        with mock.patch.dict(
            os.environ,
            {
                "TBENCH_AGENT_NETWORK_MODE": "host",
                "CONTEXT_DIR": "/tmp/task/environment",
            },
        ):
            self.assertEqual(shim.add_agent_network_overlay(arguments), arguments)

        with mock.patch.dict(
            os.environ,
            {
                "TBENCH_AGENT_NETWORK_MODE": "bridge",
                "CONTEXT_DIR": "/tmp/trial/agent-build-context",
            },
        ):
            translated = shim.add_agent_network_overlay(arguments)
        self.assertEqual(Path(translated[-2]).name, "agent-network-bridge.json")

    def test_adds_isolated_host_loopback_network_for_terminal_bench(self):
        arguments = ["compose", "--project-name", "trial", "-f", "/tmp/base.yaml", "up"]
        with mock.patch.dict(
            os.environ,
            {"TBENCH_CONTAINER_NETWORK_MODE": "podman-loopback"},
            clear=False,
        ):
            translated = shim.add_agent_network_overlay(arguments)
        self.assertEqual(Path(translated[-2]).name, "terminal-network-podman-loopback.json")

    def test_qualifies_docker_hub_image_references(self):
        self.assertEqual(
            shim.qualify_image_reference("alexgshaw/task:latest"),
            "docker.io/alexgshaw/task:latest",
        )
        self.assertEqual(
            shim.qualify_image_reference("ubuntu:24.04"),
            "docker.io/ubuntu:24.04",
        )
        self.assertEqual(
            shim.qualify_image_reference("public.ecr.aws/org/task:latest"),
            "public.ecr.aws/org/task:latest",
        )

    def test_makes_json_bind_mount_directories_writable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mount = root / "agent"
            mount.mkdir(mode=0o755)
            compose = root / "mounts.json"
            compose.write_text(
                json.dumps(
                    {
                        "services": {
                            "main": {
                                "volumes": [
                                    {
                                        "type": "bind",
                                        "source": str(mount),
                                        "target": "/logs/agent",
                                    }
                                ]
                            }
                        }
                    }
                )
            )
            with (
                mock.patch.dict(os.environ, {"TBENCH_JOBS_DIR": str(root)}),
                mock.patch.object(shim, "has_container_file_label", return_value=False),
                mock.patch.object(shim, "relabel_bind_source") as relabel,
            ):
                shim.prepare_bind_mounts(["compose", "-f", str(compose), "up"])
            self.assertEqual(os.stat(mount).st_mode & 0o777, 0o777)
            relabel.assert_called_once_with(mount.resolve())

    def test_ignores_bind_mount_outside_scoped_jobs_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = root / "jobs"
            jobs.mkdir()
            outside = root / "outside"
            outside.mkdir(mode=0o755)
            compose = root / "mounts.json"
            compose.write_text(
                json.dumps(
                    {
                        "services": {
                            "main": {
                                "volumes": [
                                    {"type": "bind", "source": str(outside), "target": "/logs"}
                                ]
                            }
                        }
                    }
                )
            )
            with (
                mock.patch.dict(os.environ, {"TBENCH_JOBS_DIR": str(jobs)}),
                mock.patch.object(shim, "relabel_bind_source") as relabel,
            ):
                shim.prepare_bind_mounts(["compose", "-f", str(compose), "up"])
            self.assertEqual(os.stat(outside).st_mode & 0o777, 0o755)
            relabel.assert_not_called()

    def test_already_prepared_rootless_mount_needs_no_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            source.chmod(0o777)
            compose = source.parent / f"{source.name}-mounts.json"
            compose.write_text(
                json.dumps(
                    {
                        "services": {
                            "main": {
                                "volumes": [
                                    {"type": "bind", "source": str(source), "target": "/logs"}
                                ]
                            }
                        }
                    }
                )
            )
            with (
                mock.patch.dict(os.environ, {"TBENCH_JOBS_DIR": str(source)}),
                mock.patch.object(shim, "has_container_file_label", return_value=True),
                mock.patch.object(Path, "chmod") as chmod,
                mock.patch.object(shim, "relabel_bind_source") as relabel,
            ):
                shim.prepare_bind_mounts(["compose", "-f", str(compose), "up"])
            chmod.assert_not_called()
            relabel.assert_not_called()
            compose.unlink()


if __name__ == "__main__":
    unittest.main()
