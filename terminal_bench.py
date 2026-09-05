#!/usr/bin/env python3
"""Run Terminal-Bench-Local against an OpenAI-compatible local model."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import time
from typing import Any
from urllib import error, request

import results as result_store
from suite_manifest import SuiteManifest, SuiteManifestError


ROOT = Path(__file__).resolve().parent
DEFAULT_TASKS_DIR = ROOT / "tasks"
SUITES_DIR = ROOT / "suites"
DEFAULT_SUITE = "core19"
LEGACY_SUITE = "legacy-mini20"
DEFAULT_ENDPOINT = "http://localhost:8080/v1"
DEFAULT_AGENT_TIMEOUT_SECONDS = 3 * 60 * 60
DEFAULT_ATTEMPTS = 2
AGENT_NAME = "terminus-2"
AGENT_VERSION = "2.0.0"
SUMMARIZATION_FREE_TOKENS = 8_000
HARBOR_VERSION = "0.20.0"
HARBOR_PACKAGE = f"harbor=={HARBOR_VERSION}"
MINIMUM_PYTHON = (3, 11)
RECOMMENDED_CONTEXT_LENGTH = 262_144
JOBS_DIR = ROOT / "jobs"
DEFAULT_RESULTS_DIR = ROOT / "results"
TIERS = {
    "smoke": ROOT / "subsets" / "smoke.txt",
    "full": ROOT / "subsets" / "full.txt",
}
JOB_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,219}")

# Mean attempt durations, in minutes, from the nine finished Core-19 result
# sets available when weighted scheduling was introduced. Means intentionally
# retain long-tail and timeout cost because the objective is wall-clock balance.
CORE19_TASK_ESTIMATED_MINUTES = {
    "break-filter-js-from-html": 90.8,
    "llm-inference-batching-scheduler": 86.6,
    "fix-ocaml-gc": 77.3,
    "mailman": 66.7,
    "regex-log": 62.5,
    "sparql-university": 51.0,
    "extract-elf": 39.2,
    "cobol-modernization": 38.8,
    "headless-terminal": 36.0,
    "build-cython-ext": 34.6,
    "overfull-hbox": 28.1,
    "mteb-retrieve": 22.0,
    "configure-git-webserver": 19.8,
    "nginx-request-logging": 11.6,
    "sqlite-with-gcov": 10.2,
    "git-leak-recovery": 9.3,
    "fix-git": 9.1,
    "openssl-selfsigned-cert": 7.7,
    "pypi-server": 7.5,
}


class RunnerError(RuntimeError):
    pass


def validate_python_version(version_info: tuple[int, ...] | None = None) -> None:
    current = version_info or sys.version_info
    if tuple(current[:2]) < MINIMUM_PYTHON:
        required = ".".join(str(value) for value in MINIMUM_PYTHON)
        actual = ".".join(str(value) for value in current[:3])
        raise RunnerError(
            f"Python {required} or newer is required by {HARBOR_PACKAGE}; "
            f"this runner is using Python {actual}"
        )


def endpoint_url(base: str, route: str) -> str:
    return f"{base.rstrip('/')}/{route.lstrip('/')}"


def get_json(url: str, api_key: str, timeout: float = 10) -> dict[str, Any]:
    req = request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RunnerError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RunnerError(f"Cannot query {url}: {exc}") from exc


def model_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    entries = payload.get("data") or payload.get("models") or []
    return [entry for entry in entries if isinstance(entry, dict)]


def model_id(entry: dict[str, Any]) -> str | None:
    for key in ("id", "model", "name"):
        value = entry.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def discover_model(
    endpoint: str, api_key: str, preferred: str | None = None
) -> tuple[str, dict[str, Any]]:
    entries = model_entries(get_json(endpoint_url(endpoint, "models"), api_key))
    models = [(model_id(entry), entry) for entry in entries]
    models = [(name, entry) for name, entry in models if name]
    if not models:
        raise RunnerError(f"No model IDs returned by {endpoint_url(endpoint, 'models')}")
    if preferred:
        for name, entry in models:
            if name == preferred:
                return name, entry
        names = ", ".join(name for name, _ in models)
        raise RunnerError(f"Model {preferred!r} is not advertised by the endpoint: {names}")
    if len(models) > 1:
        names = ", ".join(name for name, _ in models)
        raise RunnerError(f"Endpoint serves multiple models; pass --model explicitly: {names}")
    return models[0]


def parse_endpoints(values: list[str] | None) -> list[str]:
    """Normalize comma- or whitespace-separated endpoint arguments."""
    endpoints = [
        endpoint.strip()
        for value in values or []
        for endpoint in value.split(",")
        if endpoint.strip()
    ]
    if not endpoints:
        raise RunnerError("--endpoints must contain at least one URL")
    if len(endpoints) != len(set(endpoints)):
        raise RunnerError("--endpoints contains a duplicate URL")
    return endpoints


def connection_endpoints(
    args: argparse.Namespace, *, fallback: list[str] | None = None
) -> list[str]:
    if getattr(args, "endpoints", None) is not None:
        return parse_endpoints(args.endpoints)
    if getattr(args, "endpoint", None):
        return [str(args.endpoint)]
    if fallback:
        return fallback
    return [os.getenv("TBENCH_ENDPOINT", DEFAULT_ENDPOINT)]


def discover_matching_model(
    endpoints: list[str],
    api_key: str,
    preferred: str | None = None,
    context_override: int | None = None,
) -> tuple[str, dict[str, Any], int]:
    """Validate that every endpoint advertises the same model and context."""
    discovered: list[tuple[str, dict[str, Any], int]] = []
    selected = preferred
    for endpoint in endpoints:
        model, metadata = discover_model(endpoint, api_key, preferred=selected)
        selected = selected or model
        context = model_context_length(metadata, context_override)
        discovered.append((model, metadata, context))

    model, metadata, context = discovered[0]
    for endpoint, (candidate, candidate_metadata, candidate_context) in zip(
        endpoints[1:], discovered[1:]
    ):
        if candidate != model:
            raise RunnerError(
                f"Endpoint {endpoint} advertises model {candidate!r}; expected {model!r}"
            )
        if candidate_context != context:
            raise RunnerError(
                f"Endpoint {endpoint} advertises {candidate_context:,} context tokens; "
                f"expected {context:,}"
            )
        if stable_model_metadata(candidate_metadata) != stable_model_metadata(metadata):
            raise RunnerError(
                f"Endpoint {endpoint} metadata for model {model!r} does not match "
                f"{endpoints[0]}"
            )
    return model, metadata, context


def container_runtime() -> tuple[str, str]:
    docker = shutil.which("docker")
    if not docker:
        raise RunnerError("docker-compatible CLI not found in PATH")
    version = subprocess.run(
        [docker, "--version"], capture_output=True, text=True, check=False
    )
    description = (version.stdout or version.stderr).strip()
    if version.returncode != 0:
        raise RunnerError(f"docker --version failed: {description}")
    compose = subprocess.run(
        [docker, "compose", "version"], capture_output=True, text=True, check=False
    )
    compose_description = (compose.stdout or compose.stderr).strip()
    if compose.returncode != 0:
        raise RunnerError(
            "docker compose version failed: "
            f"{compose_description or 'Compose v2 or a Compose provider is missing'}"
        )
    kind = "podman" if "podman" in description.lower() else "docker"
    return kind, f"{description}; {compose_description}"


def harbor_command() -> list[str]:
    if harbor := shutil.which("harbor"):
        return [harbor]
    if uvx := shutil.which("uvx"):
        return [uvx, "--from", HARBOR_PACKAGE, "harbor"]
    if uv := shutil.which("uv"):
        return [uv, "tool", "run", "--from", HARBOR_PACKAGE, "harbor"]
    raise RunnerError(
        "Neither harbor nor uv/uvx is installed. Install uv, or run "
        f"`uv tool install {HARBOR_PACKAGE}`."
    )


def installed_harbor_version(command: list[str] | None = None) -> str:
    """Return the Harbor CLI version and require the benchmark's exact pin."""
    harbor = command or harbor_command()
    try:
        completed = subprocess.run(
            [*harbor, "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise RunnerError(f"Cannot execute Harbor: {exc}") from exc
    output = (completed.stdout or completed.stderr).strip()
    if completed.returncode != 0:
        raise RunnerError(f"Harbor version check failed: {output or 'no output'}")
    version = next(
        (line.strip() for line in completed.stdout.splitlines() if line.strip()),
        "",
    )
    if version != HARBOR_VERSION:
        displayed = version or output or "unknown"
        raise RunnerError(
            f"Harbor {HARBOR_VERSION} is required, but {displayed} is installed. "
            f"Use `{HARBOR_PACKAGE}` for this benchmark."
        )
    return version


def validate_stored_harbor_version(meta: dict[str, Any]) -> None:
    profile = meta.get("evaluation_profile") or {}
    stored = profile.get("harbor_version") or meta.get("harbor_version")
    if stored and str(stored) != HARBOR_VERSION:
        raise RunnerError(
            f"This job was created with Harbor {stored}; it cannot be resumed or "
            f"merged with the required Harbor {HARBOR_VERSION} protocol"
        )


def load_tier(name: str) -> list[str]:
    path = TIERS[name]
    if not path.is_file():
        raise RunnerError(f"Missing tier definition: {path}")
    tasks = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not tasks or len(tasks) != len(set(tasks)):
        raise RunnerError(f"Tier {name!r} is empty or contains duplicate task IDs")
    return tasks


def suite_manifest_path(value: str | Path | None) -> Path:
    if value is None:
        return SUITES_DIR / f"{DEFAULT_SUITE}.json"
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return candidate
    if candidate.suffix == ".json" or len(candidate.parts) > 1:
        raise RunnerError(f"Suite manifest not found: {candidate}")
    built_in = SUITES_DIR / f"{candidate.name}.json"
    if not built_in.is_file():
        available = ", ".join(path.stem for path in sorted(SUITES_DIR.glob("*.json")))
        raise RunnerError(
            f"Unknown suite {candidate.name!r}; available built-ins: {available}"
        )
    return built_in


def load_suite(value: str | Path | None, tasks_dir: Path | None = None) -> SuiteManifest:
    # Preserve the old ``--tasks-dir tasks`` spelling as an explicit request for
    # the legacy mini-20. With no source option, the current core is selected.
    selected = LEGACY_SUITE if value is None and tasks_dir is not None else value
    path = suite_manifest_path(selected)
    try:
        suite = SuiteManifest.load(path)
    except SuiteManifestError as exc:
        raise RunnerError(str(exc)) from exc
    core_path = (SUITES_DIR / f"{DEFAULT_SUITE}.json").resolve()
    if path.resolve() == core_path:
        for tier in TIERS:
            if suite.tasks_for(tier) != load_tier(tier):
                raise RunnerError(
                    f"Default suite tier {tier!r} does not match canonical "
                    f"{TIERS[tier]}"
                )
    if tasks_dir is not None:
        requested_root = tasks_dir.expanduser().resolve()
        roots = {task.root for task in suite.tasks.values()}
        if roots != {requested_root}:
            raise RunnerError(
                "--tasks-dir cannot override a suite manifest's task roots; "
                "select or create a suite manifest instead"
            )
    return suite


def selected_suite_tasks(args: argparse.Namespace, suite: SuiteManifest) -> list[str]:
    task = getattr(args, "task", None)
    if task:
        try:
            suite.resolve_task(task)
        except SuiteManifestError as exc:
            raise RunnerError(str(exc)) from exc
        return [task]
    tier = str(getattr(args, "tier", None) or "full")
    try:
        return suite.tasks_for(tier)
    except SuiteManifestError as exc:
        raise RunnerError(str(exc)) from exc


def validate_tasks(tasks_dir: Path, tasks: list[str]) -> None:
    missing = [task for task in tasks if not (tasks_dir / task / "task.toml").is_file()]
    if missing:
        raise RunnerError(
            f"{len(missing)} task(s) are absent below {tasks_dir}: {', '.join(missing)}"
        )


def checkout_revision(tasks_dir: Path) -> str | None:
    revision_file = tasks_dir / "REVISION"
    if revision_file.is_file():
        revision = revision_file.read_text().strip()
        if revision:
            return revision
    completed = subprocess.run(
        ["git", "-C", str(tasks_dir), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    revision = completed.stdout.strip()
    return revision if completed.returncode == 0 and revision else None


def stable_model_metadata(value: Any) -> Any:
    volatile = {"created", "created_at", "updated", "updated_at", "timestamp"}
    if isinstance(value, dict):
        return {
            key: stable_model_metadata(item)
            for key, item in value.items()
            if key not in volatile
        }
    if isinstance(value, list):
        return [stable_model_metadata(item) for item in value]
    return value


def harbor_model_name(raw_model: str) -> str:
    return raw_model if raw_model.startswith("openai/") else f"openai/{raw_model}"


def make_job_name(
    tier: str,
    model: str,
    quant: str | None = None,
    *,
    engine: str | None = None,
    backend: str | None = None,
    inference_profile: str | None = None,
    tag: str | None = None,
) -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    identity_parts = [model]
    identity_parts.extend(value for value in (quant, inference_profile, tag) if value)
    identity_parts.extend(value for value in (engine, backend) if value)
    identity = "--".join(identity_parts)
    slug = result_store.stable_slug(identity, limit=140)
    return validate_job_name(
        f"{stamp}-terminal-bench-local-{tier}-{AGENT_NAME}-{slug}"
    )


def validate_job_name(value: str) -> str:
    """Require one conservative filename component for all generated paths."""

    if not JOB_NAME_PATTERN.fullmatch(value):
        raise RunnerError(
            "Job names must be 1-220 ASCII letters, digits, dots, underscores, "
            "or dashes, and must start with a letter or digit"
        )
    return value


def job_name_argument(value: str) -> str:
    try:
        return validate_job_name(value)
    except RunnerError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def result_tag(
    quant: str | None,
    engine: str,
    backend: str,
    inference_profile: str | None = None,
    tag: str | None = None,
) -> str:
    parts = [engine, backend]
    parts.extend(value for value in (quant, inference_profile, tag) if value)
    parts.append(AGENT_NAME)
    return "-".join(parts)


def run_identity_args(
    args: argparse.Namespace,
) -> tuple[str, str | None, str | None, str | None]:
    model_name = str(args.model_name or "").strip()
    if not model_name:
        raise RunnerError("--model-name must be non-empty")
    quant = str(args.quant or "").strip() or None
    inference_profile = str(args.inference_profile or "").strip() or None
    tag = str(args.tag or "").strip() or None
    return model_name, quant, inference_profile, tag


def model_context_length(
    metadata: dict[str, Any], override: int | None = None
) -> int:
    if override is not None:
        if override < 1:
            raise RunnerError("--context-length must be positive")
        return override
    candidates = (
        (metadata.get("meta") or {}).get("n_ctx"),
        metadata.get("n_ctx"),
        metadata.get("context_length"),
        metadata.get("max_model_len"),
    )
    for value in candidates:
        if isinstance(value, int) and value > 0:
            return value
    raise RunnerError(
        "The endpoint did not advertise its context length; pass --context-length"
    )


def evaluation_profile(
    *,
    tasks_dir: Path,
    model: str,
    model_metadata: dict[str, Any],
    endpoint: str,
    endpoints: list[str] | None = None,
    engine: str,
    engine_version: str | None,
    backend: str,
    backend_version: str | None,
    quant: str | None,
    inference_profile: str | None,
    tag: str | None,
    context_length: int,
    agent_timeout_seconds: int,
    suite: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile = {
        "benchmark": "Terminal-Bench-Local",
        "harbor_version": HARBOR_PACKAGE.removeprefix("harbor=="),
        "model_id": model,
        "model_metadata": stable_model_metadata(model_metadata),
        "endpoint": endpoint,
        "engine": engine,
        "engine_version": engine_version,
        "backend": backend,
        "backend_version": backend_version,
        "quant": quant,
        "inference_profile": inference_profile,
        "tag": tag,
        "agent": {
            "name": AGENT_NAME,
            "version": AGENT_VERSION,
            "context_length": context_length,
            "context_summarization": True,
            "summarization_free_tokens": SUMMARIZATION_FREE_TOKENS,
        },
        "agent_timeout_seconds": agent_timeout_seconds,
    }
    if suite:
        profile["suite"] = suite
    else:
        profile["terminal_bench_version"] = "2.1"
        profile["terminal_bench_revision"] = checkout_revision(tasks_dir)
    if endpoints and len(endpoints) > 1:
        profile["endpoints"] = sorted(endpoints)
    return profile


def make_run_meta(
    *,
    job_name: str,
    tier: str,
    requested_tasks: list[str],
    executed_tasks: list[str],
    platform: str,
    platform_name: str | None,
    model: str,
    model_name: str,
    model_metadata: dict[str, Any],
    engine: str,
    engine_version: str | None,
    backend: str,
    backend_version: str | None,
    quant: str | None,
    inference_profile: str | None,
    tag: str | None,
    endpoint: str,
    endpoints: list[str] | None,
    runtime: str,
    runtime_description: str,
    profile: dict[str, Any],
    max_attempts: int,
    suite: dict[str, Any] | None = None,
    task_provenance: dict[str, dict[str, Any]] | None = None,
    attempt_group: str | None = None,
    attempt_round: int = 1,
) -> dict[str, Any]:
    return {
        "schema_version": result_store.SCHEMA_VERSION,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "benchmark": "Terminal-Bench-Local",
        "harness": "Harbor",
        "terminal_bench_version": None if suite else "2.1",
        "suite": suite,
        "task_provenance": task_provenance or {},
        "tier": tier,
        "requested_tasks": requested_tasks,
        "executed_tasks": executed_tasks,
        "task_list_hash": hashlib.sha256("\n".join(requested_tasks).encode()).hexdigest(),
        "platform": {"id": platform, "name": platform_name or platform},
        "model": {
            "name": model_name,
            "id": model,
            "endpoint_metadata": model_metadata,
        },
        "engine": engine,
        "engine_version": engine_version,
        "backend": backend,
        "backend_version": backend_version,
        "quant": quant,
        "inference_profile": inference_profile,
        "tag": tag,
        "result_tag": result_tag(quant, engine, backend, inference_profile, tag),
        "endpoint": endpoint,
        "endpoints": endpoints or [endpoint],
        "container_runtime": {"type": runtime, "description": runtime_description},
        "job_name": job_name,
        "attempt_group": attempt_group or job_name,
        "attempt_round": attempt_round,
        "attempt_policy": "stop_on_pass",
        "max_attempts": max_attempts,
        "evaluation_profile": profile,
        "profile_hash": result_store.evaluation_profile_hash(profile),
    }


def build_config(
    *,
    job_name: str,
    tasks_dir: Path,
    tasks: list[str],
    model: str,
    endpoint: str,
    api_key: str,
    concurrency: int,
    context_length: int,
    agent_timeout_seconds: int,
    keep_containers: bool,
    dataset_groups: list[tuple[Path, list[str]]] | None = None,
) -> dict[str, Any]:
    datasets = (
        [
            {"path": str(dataset_root), "task_names": dataset_tasks}
            for dataset_root, dataset_tasks in dataset_groups
            if dataset_tasks
        ]
        if dataset_groups is not None
        else [{"path": str(tasks_dir), "task_names": tasks}]
    )
    configured_tasks = [
        task for dataset in datasets for task in dataset["task_names"]
    ]
    if (
        len(configured_tasks) != len(tasks)
        or len(set(configured_tasks)) != len(configured_tasks)
        or set(configured_tasks) != set(tasks)
    ):
        raise RunnerError(
            "Dataset groups must contain every selected task exactly once and "
            "contain no duplicates"
        )
    return {
        "job_name": job_name,
        "jobs_dir": str(JOBS_DIR),
        "n_attempts": 1,
        "n_concurrent_trials": concurrency,
        "environment": {"type": "docker", "delete": not keep_containers},
        "agents": [
            {
                "name": AGENT_NAME,
                "model_name": harbor_model_name(model),
                "override_timeout_sec": agent_timeout_seconds,
                "kwargs": {
                    "api_base": endpoint,
                    "enable_summarize": True,
                    "proactive_summarization_threshold": SUMMARIZATION_FREE_TOKENS,
                    "model_info": {"max_input_tokens": context_length},
                    "llm_kwargs": {"api_key": api_key},
                },
            }
        ],
        "datasets": datasets,
    }


def select_config_datasets(
    datasets: list[dict[str, Any]], selected_tasks: list[str]
) -> list[dict[str, Any]]:
    """Select tasks from a multi-root Harbor dataset catalog."""
    locations: dict[str, int] = {}
    for index, dataset in enumerate(datasets):
        for task in dataset.get("task_names") or []:
            if task in locations:
                raise RunnerError(f"Task {task!r} occurs in more than one dataset root")
            locations[task] = index
    missing = [task for task in selected_tasks if task not in locations]
    if missing:
        raise RunnerError(
            "Selected task(s) are absent from the configured datasets: "
            + ", ".join(missing)
        )

    grouped: list[list[str]] = [[] for _ in datasets]
    for task in selected_tasks:
        grouped[locations[task]].append(task)
    selected: list[dict[str, Any]] = []
    for dataset, task_names in zip(datasets, grouped):
        if not task_names:
            continue
        item = copy.deepcopy(dataset)
        item["task_names"] = task_names
        selected.append(item)
    return selected


def partition_tasks(
    tasks: list[str], count: int, weights: dict[str, float] | None = None
) -> list[list[str]]:
    if count < 1:
        raise RunnerError("At least one endpoint is required")
    if weights and all(task in weights and weights[task] > 0 for task in tasks):
        shards: list[list[str]] = [[] for _ in range(count)]
        loads = [0.0] * count
        ordered = sorted(
            enumerate(tasks),
            key=lambda item: (-weights[item[1]], item[0]),
        )
        for _, task in ordered:
            index = min(range(count), key=lambda candidate: (loads[candidate], candidate))
            shards[index].append(task)
            loads[index] += weights[task]
        return shards
    return [tasks[index::count] for index in range(count)]


def write_config(config: dict[str, Any]) -> Path:
    job_name = validate_job_name(str(config["job_name"]))
    config_dir = ROOT / ".runner" / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_dir.chmod(0o700)
    path = config_dir / f"{job_name}.json"
    result_store.write_json(path, config)
    path.chmod(0o600)
    return path


def display_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a printable config with endpoint credentials removed."""
    value = copy.deepcopy(config)
    for agent in value.get("agents") or []:
        llm_kwargs = (agent.get("kwargs") or {}).get("llm_kwargs") or {}
        if "api_key" in llm_kwargs:
            llm_kwargs["api_key"] = "<redacted>"
    return value


def runner_meta_path(job_name: str) -> Path:
    validate_job_name(job_name)
    return ROOT / ".runner" / "run-meta" / f"{job_name}.json"


def save_runner_meta(job_name: str, meta: dict[str, Any]) -> None:
    result_store.write_json(runner_meta_path(job_name), meta)


def command_environment(runtime: str) -> dict[str, str]:
    environment = os.environ.copy()
    real_docker = shutil.which("docker")
    if not real_docker:
        raise RunnerError("docker-compatible CLI not found in PATH")
    environment["TBENCH_REAL_DOCKER"] = real_docker
    environment["TBENCH_CONTAINER_RUNTIME"] = runtime
    environment["TBENCH_JOBS_DIR"] = str(JOBS_DIR)
    if runtime == "podman":
        environment["TBENCH_CONTAINER_NETWORK_MODE"] = "podman-loopback"
    else:
        environment["TBENCH_AGENT_NETWORK_MODE"] = "bridge"
    compat_dir = ROOT / "compat" / "podman"
    environment["PATH"] = f"{compat_dir}{os.pathsep}{environment.get('PATH', '')}"
    return environment


def cleanup_interrupted_containers(
    configs: list[dict[str, Any]], *, runtime: str
) -> None:
    """Remove only Compose containers belonging to interrupted Harbor trials."""
    projects: set[str] = set()
    for config in configs:
        if not bool((config.get("environment") or {}).get("delete", True)):
            continue
        job_name = validate_job_name(str(config["job_name"]))
        job_dir = JOBS_DIR / job_name
        task_names = {
            str(task).casefold()
            for dataset in config.get("datasets") or []
            for task in dataset.get("task_names") or []
        }
        if not job_dir.is_dir():
            continue
        for trial_dir in job_dir.iterdir():
            if not trial_dir.is_dir() or "__" not in trial_dir.name:
                continue
            task_name = trial_dir.name.split("__", 1)[0].casefold()
            if task_name in task_names:
                projects.add(f"{trial_dir.name.casefold()}__env")

    if not projects:
        return
    docker = shutil.which("docker")
    if not docker:
        print(
            "Warning: cannot clean interrupted containers: docker-compatible "
            "CLI not found.",
            file=sys.stderr,
        )
        return
    environment = command_environment(runtime)
    try:
        listed = subprocess.run(
            [
                docker,
                "ps",
                "-a",
                "--format",
                '{{.ID}}\t{{.Label "com.docker.compose.project"}}',
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        print(f"Warning: interrupted-container cleanup failed: {exc}", file=sys.stderr)
        return
    if listed.returncode != 0:
        detail = (listed.stderr or listed.stdout).strip()
        print(
            f"Warning: cannot list interrupted containers: {detail or 'unknown error'}",
            file=sys.stderr,
        )
        return
    container_ids = [
        container_id
        for line in listed.stdout.splitlines()
        if "\t" in line
        for container_id, project in [line.split("\t", 1)]
        if project.casefold() in projects
    ]
    if not container_ids:
        return
    removed = subprocess.run(
        [docker, "rm", "-f", *container_ids],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if removed.returncode == 0:
        print(
            f"Cleaned {len(container_ids)} interrupted task container(s).",
            file=sys.stderr,
        )
    else:
        detail = (removed.stderr or removed.stdout).strip()
        print(
            f"Warning: interrupted-container cleanup failed: "
            f"{detail or 'unknown error'}",
            file=sys.stderr,
        )


def discard_cancelled_trials(job_dir: Path) -> list[str]:
    """Remove Ctrl+C trial artifacts so those tasks restart on resume.

    Harbor writes a final-looking ``CancelledError`` result when a running
    agent is interrupted. It is not a completed benchmark attempt. Harbor's
    resume command applies the same filter; doing it in the wrapper also keeps
    endpoint redistribution from exporting the cancellation as a failure.
    """
    removed: list[str] = []
    removed_results: list[dict[str, Any]] = []
    if not job_dir.is_dir():
        return removed
    for trial_dir in sorted(job_dir.iterdir()):
        result_path = trial_dir / "result.json"
        if not trial_dir.is_dir() or not result_path.is_file():
            continue
        try:
            result = result_store.read_json(result_path)
        except (OSError, ValueError):
            continue
        exception = result.get("exception_info") or {}
        if exception.get("exception_type") != "CancelledError":
            continue
        removed_results.append(result)
        shutil.rmtree(trial_dir)
        removed.append(trial_dir.name)
    if removed:
        reconcile_discarded_cancellations(job_dir, removed_results)
        print(
            f"Discarded {len(removed)} interrupted trial(s); "
            "they will restart from scratch on resume."
        )
    return removed


def reconcile_discarded_cancellations(
    job_dir: Path, removed_results: list[dict[str, Any]]
) -> None:
    """Remove discarded Ctrl+C trials from Harbor's transient aggregate."""
    path = job_dir / "result.json"
    if not removed_results or not path.is_file():
        return
    try:
        aggregate = result_store.read_json(path)
    except (OSError, ValueError):
        return
    stats = aggregate.get("stats")
    if not isinstance(stats, dict):
        return

    count = len(removed_results)
    total = int(aggregate.get("n_total_trials") or 0)
    stats["n_completed_trials"] = max(
        int(stats.get("n_completed_trials") or 0) - count, 0
    )
    stats["n_errored_trials"] = max(
        int(stats.get("n_errored_trials") or 0) - count, 0
    )
    stats["n_cancelled_trials"] = max(
        int(stats.get("n_cancelled_trials") or 0) - count, 0
    )
    stats["n_pending_trials"] = min(
        int(stats.get("n_pending_trials") or 0) + count, total
    )
    for key, result_key in (
        ("n_input_tokens", "n_input_tokens"),
        ("n_cache_tokens", "n_cache_tokens"),
        ("n_output_tokens", "n_output_tokens"),
    ):
        discarded = sum(
            int(((result.get("agent_result") or {}).get(result_key)) or 0)
            for result in removed_results
        )
        stats[key] = max(int(stats.get(key) or 0) - discarded, 0)

    removed_names = {
        str(result.get("trial_name"))
        for result in removed_results
        if result.get("trial_name")
    }
    for eval_stats in (stats.get("evals") or {}).values():
        if not isinstance(eval_stats, dict):
            continue
        exception_stats = eval_stats.get("exception_stats") or {}
        cancelled_names = list(exception_stats.get("CancelledError") or [])
        matched = len(removed_names.intersection(cancelled_names))
        remaining_cancelled = [
            name for name in cancelled_names if name not in removed_names
        ]
        if remaining_cancelled:
            exception_stats["CancelledError"] = remaining_cancelled
        else:
            exception_stats.pop("CancelledError", None)
        if exception_stats:
            eval_stats["exception_stats"] = exception_stats
        else:
            eval_stats.pop("exception_stats", None)
        eval_stats["n_errors"] = max(
            int(eval_stats.get("n_errors") or 0) - matched, 0
        )
        denominator = int(eval_stats.get("n_trials") or 0) + int(
            eval_stats.get("n_errors") or 0
        )
        rewards = ((eval_stats.get("reward_stats") or {}).get("reward") or {})
        reward_total = sum(
            float(reward) * len(names) for reward, names in rewards.items()
        )
        for metric in eval_stats.get("metrics") or []:
            if isinstance(metric, dict) and "mean" in metric:
                metric["mean"] = reward_total / denominator if denominator else 0.0

    if isinstance(aggregate.get("trial_results"), list):
        aggregate["trial_results"] = [
            result
            for result in aggregate["trial_results"]
            if str(result.get("trial_name")) not in removed_names
        ]
    aggregate["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    aggregate["finished_at"] = None
    result_store.write_json(path, aggregate)


def format_live_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"


def active_trial(job_dir: Path) -> tuple[str, float] | None:
    """Return the newest unfinished trial name and elapsed wall time."""
    if not job_dir.is_dir():
        return None
    candidates: list[tuple[float, str]] = []
    for trial_dir in job_dir.iterdir():
        if not trial_dir.is_dir() or "__" not in trial_dir.name:
            continue
        if (trial_dir / "result.json").is_file():
            continue
        marker = trial_dir / "lock.json"
        try:
            started = marker.stat().st_mtime if marker.is_file() else trial_dir.stat().st_mtime
        except OSError:
            continue
        candidates.append((started, trial_dir.name.split("__", 1)[0]))
    if not candidates:
        return None
    started, task = max(candidates)
    return task, max(0.0, time.time() - started)


def live_job_counts(config: dict[str, Any]) -> dict[str, int]:
    job_dir = JOBS_DIR / str(config["job_name"])
    total = sum(
        len(dataset.get("task_names") or [])
        for dataset in config.get("datasets") or []
    )
    completed = running = errors = 0
    pending = total
    passed_tasks: set[str] = set()
    graded_tasks: set[str] = set()
    path = job_dir / "result.json"
    if path.is_file():
        try:
            result = result_store.read_json(path)
            stats = result.get("stats") or {}
            total = int(result.get("n_total_trials") or total)
            completed = int(stats.get("n_completed_trials") or 0)
            running = int(stats.get("n_running_trials") or 0)
            pending = int(stats.get("n_pending_trials") or 0)
            errors = int(stats.get("n_errored_trials") or 0)
            for eval_stats in (stats.get("evals") or {}).values():
                rewards = (
                    ((eval_stats or {}).get("reward_stats") or {}).get("reward")
                    or {}
                )
                for value, names in rewards.items():
                    task_names = {str(name) for name in names}
                    graded_tasks.update(task_names)
                    try:
                        passed = float(value) == 1.0
                    except (TypeError, ValueError):
                        passed = False
                    if passed:
                        passed_tasks.update(task_names)
        except (OSError, TypeError, ValueError):
            pass
    return {
        "total": total,
        "completed": completed,
        "running": running,
        "pending": pending,
        "errors": errors,
        "passed": len(passed_tasks),
        "graded": len(graded_tasks),
    }


def live_success(passed: int, graded: int) -> str:
    return f"{passed}/{graded} ({passed / graded:.0%})" if graded else "n/a"


def child_live_status(
    config: dict[str, Any], meta: dict[str, Any], return_code: int | None
) -> str:
    job_dir = JOBS_DIR / str(config["job_name"])
    endpoint = str(meta.get("endpoint") or "unknown endpoint")
    counts = live_job_counts(config)

    width = 18
    filled = round(width * counts["completed"] / counts["total"]) if counts["total"] else 0
    bar = f"[{'#' * filled}{'-' * (width - filled)}]"
    trial = active_trial(job_dir)
    if return_code is not None:
        activity = "finished" if return_code == 0 else f"exited {return_code}"
    elif trial:
        activity = f"{trial[0]} {format_live_elapsed(trial[1])}"
    elif counts["running"]:
        activity = "starting task"
    else:
        activity = "starting"
    return (
        f"{endpoint}  {bar} {counts['completed']}/{counts['total']}  | "
        f"pass {live_success(counts['passed'], counts['graded'])} | {activity} | "
        f"{counts['pending']} pending | {counts['errors']} errors"
    )


def overall_live_status(
    configs: list[dict[str, Any]],
    campaign_progress: dict[str, int] | None = None,
) -> str:
    job_counts = [live_job_counts(config) for config in configs]
    combined = {
        key: sum(counts[key] for counts in job_counts)
        for key in ("total", "completed", "running", "pending", "errors", "passed", "graded")
    }
    if campaign_progress:
        combined["total"] = campaign_progress["total"]
        for key in ("completed", "errors", "passed", "graded"):
            combined[key] += campaign_progress.get(key, 0)
        combined["pending"] = max(
            combined["total"] - combined["completed"] - combined["running"], 0
        )
    width = 24
    filled = round(width * combined["completed"] / combined["total"]) if combined["total"] else 0
    bar = f"[{'#' * filled}{'-' * (width - filled)}]"
    return (
        f"Overall {bar} {combined['completed']}/{combined['total']} | "
        f"current pass {live_success(combined['passed'], combined['graded'])} | "
        f"{combined['running']} running | {combined['pending']} pending | "
        f"{combined['errors']} errors"
    )


class TerminalDashboard:
    """Small dependency-free in-place display owned solely by the parent."""

    def __init__(self) -> None:
        self.enabled = sys.stdout.isatty() and os.environ.get("TERM") != "dumb"
        self._line_count = 0
        if self.enabled:
            sys.stdout.write("\x1b[?25l")
            sys.stdout.flush()

    def render(self, lines: list[str]) -> None:
        if not self.enabled:
            return
        if self._line_count:
            sys.stdout.write(f"\x1b[{self._line_count}F")
        columns = max(20, shutil.get_terminal_size((120, 24)).columns)
        for line in lines:
            display = line[: columns - 1]
            sys.stdout.write(f"\x1b[2K{display}\n")
        self._line_count = len(lines)
        sys.stdout.flush()

    def close(self) -> None:
        if self.enabled:
            sys.stdout.write("\x1b[?25h")
            sys.stdout.flush()
            self.enabled = False


def check_doctor(
    args: argparse.Namespace,
) -> tuple[
    str,
    str,
    str,
    dict[str, Any],
    int,
    list[str],
    SuiteManifest,
    list[str],
]:
    validate_python_version()
    suite = load_suite(args.suite, args.tasks_dir)
    selected_tasks = selected_suite_tasks(args, suite)
    task_groups = suite.grouped_tasks(selected_tasks)
    for tasks_dir, tasks in task_groups:
        validate_tasks(tasks_dir, tasks)
    runtime, runtime_description = container_runtime()
    harbor = harbor_command()
    harbor_version = installed_harbor_version(harbor)
    endpoints = connection_endpoints(args)
    model = args.model
    metadata: dict[str, Any] = {}
    if not args.skip_endpoint_check:
        discovered, metadata, context = discover_matching_model(
            endpoints,
            args.api_key,
            preferred=model,
            context_override=args.context_length,
        )
        model = model or discovered
    if not model:
        raise RunnerError("Cannot determine model ID; pass --model")
    if args.skip_endpoint_check:
        context = model_context_length(metadata, args.context_length)
    if len(endpoints) == 1:
        print(f"Endpoint:           {endpoints[0]}")
    else:
        print(f"Endpoints:          {len(endpoints)} validated")
        for index, endpoint in enumerate(endpoints, 1):
            print(f"  [{index}] {endpoint}")
    print(f"Model:              {model}")
    print(f"Container runtime:  {runtime_description}")
    print(
        f"Suite:              {suite.id} {suite.version} "
        f"({suite.manifest_hash[:12]})"
    )
    print(
        f"Tasks:              {len(selected_tasks)} selected ID(s) across "
        f"{len(task_groups)} root(s)"
    )
    for tasks_dir, tasks in task_groups:
        print(f"  {tasks_dir}: {len(tasks)} task(s)")
    print(f"Harbor:             {harbor_version} ({' '.join(harbor)})")
    print(f"Model context:      {context:,} tokens")
    if args.context_length is None and context < RECOMMENDED_CONTEXT_LENGTH:
        print(
            f"Warning: endpoint advertises less than the recommended "
            f"{RECOMMENDED_CONTEXT_LENGTH:,}-token context.",
            file=sys.stderr,
        )
    return (
        model,
        runtime,
        runtime_description,
        metadata,
        context,
        endpoints,
        suite,
        selected_tasks,
    )


def summarize_job(job_dir: Path) -> None:
    path = job_dir if job_dir.name == "result.json" else job_dir / "result.json"
    if not path.is_file():
        raise RunnerError(f"Harbor result not found: {path}")
    result = result_store.read_json(path)
    stats = result.get("stats") or {}
    total = int(result.get("n_total_trials") or 0)
    completed = int(stats.get("n_completed_trials") or 0)
    errors = int(stats.get("n_errored_trials") or 0)
    print(f"Job:       {path.parent}")
    print(f"Trials:    {completed}/{total} completed")
    print(f"Errors:    {errors}")
    print(
        f"Tokens:    {int(stats.get('n_input_tokens') or 0):,} input, "
        f"{int(stats.get('n_output_tokens') or 0):,} output"
    )
    for key, eval_stats in (stats.get("evals") or {}).items():
        rewards = ((eval_stats or {}).get("reward_stats") or {}).get("reward") or {}
        passed = sum(len(names) for value, names in rewards.items() if float(value) == 1.0)
        graded = sum(len(names) for names in rewards.values())
        value = f"{passed}/{graded} ({passed / graded:.1%})" if graded else "n/a"
        print(f"Passed:    {value} [{key}]")


def harbor_result_is_terminal(result: dict[str, Any]) -> bool:
    """Return whether Harbor's aggregate counters describe a finished job.

    Harbor writes ``result.json`` while a job is still running. Older releases
    counted errored and cancelled trials inside ``n_completed_trials`` while
    newer schemas may report them as disjoint counters, so accept either
    internally consistent terminal representation.
    """

    stats = result.get("stats")
    if not isinstance(stats, dict):
        return False
    try:
        total = int(result["n_total_trials"])
        completed = int(stats["n_completed_trials"])
        errored = int(stats["n_errored_trials"])
        running = int(stats["n_running_trials"])
        pending = int(stats["n_pending_trials"])
        cancelled = int(stats["n_cancelled_trials"])
    except (KeyError, TypeError, ValueError):
        return False
    counts = (total, completed, errored, running, pending, cancelled)
    if total <= 0 or any(value < 0 for value in counts):
        return False
    if running != 0 or pending != 0:
        return False
    return completed == total or completed + errored + cancelled == total


def job_is_terminal(job_dir: Path) -> bool:
    """Check terminal state from Harbor counters, not file existence alone."""

    path = job_dir / "result.json"
    if not path.is_file():
        return False
    try:
        result = result_store.read_json(path)
    except (OSError, ValueError):
        return False
    return harbor_result_is_terminal(result)


def secure_job_directory(job_dir: Path, *, create: bool = False) -> None:
    """Keep Harbor's copied config, including any API key, owner-readable only."""

    if create:
        job_dir.mkdir(parents=True)
    job_dir.chmod(0o700)


def summarize_orchestrator(parent_dir: Path) -> None:
    path = parent_dir / "orchestrator.json"
    if not path.is_file():
        raise RunnerError(f"Orchestrator metadata not found: {path}")
    manifest = result_store.read_json(path)
    print(f"Orchestrator: {parent_dir}")
    print(f"Endpoints:    {len(manifest.get('endpoints') or [])}")
    for attempt, child_names in sorted(
        (manifest.get("rounds") or {}).items(), key=lambda item: int(item[0])
    ):
        child_names = [validate_job_name(str(name)) for name in child_names]
        completed = sum(
            job_is_terminal(JOBS_DIR / child_name) for child_name in child_names
        )
        print(f"Attempt {attempt}:  {completed}/{len(child_names)} child job(s) complete")
        for child_name in child_names:
            state = (
                "complete"
                if job_is_terminal(JOBS_DIR / child_name)
                else "incomplete"
            )
            print(f"  {child_name}: {state}")


def print_results_summary(model_dir: Path, summary: dict[str, Any]) -> None:
    print(f"Results:   {model_dir}")
    print(
        f"Aggregate: {summary['passed_tasks']}/{summary['total_tasks']} "
        f"({summary['pass_rate']:.1%})"
    )


def execute_harbor_job(
    *,
    config: dict[str, Any],
    meta: dict[str, Any],
    results_root: Path,
    runtime: str,
    merge_existing_attempts: bool,
    dry_run: bool = False,
) -> tuple[int, bool]:
    job_name = validate_job_name(str(config["job_name"]))
    job_dir = JOBS_DIR / job_name
    if job_dir.exists():
        raise RunnerError(
            f"Job directory already exists: {job_dir}. "
            f"Use `./terminal_bench.py resume {job_dir}` to continue it."
        )
    save_runner_meta(job_name, meta)
    config_path = write_config(config)
    print(f"Job:                {job_name}")
    print(f"Config:             {config_path}")
    if dry_run:
        print(json.dumps(display_config(config), indent=2))
        return 0, False
    secure_job_directory(job_dir, create=True)
    result_store.write_json(job_dir / "runner-meta.json", meta)
    command = [*harbor_command(), "run", "--config", str(config_path), "--yes"]
    print(f"Running:            {' '.join(command)}", flush=True)
    try:
        completed = subprocess.run(
            command, cwd=ROOT, env=command_environment(runtime), check=False
        )
    except KeyboardInterrupt:
        print("\nRun interrupted; Harbor received the interrupt.", file=sys.stderr)
        cleanup_interrupted_containers([config], runtime=runtime)
        discard_cancelled_trials(job_dir)
        return 130, False
    exported = False
    if job_is_terminal(job_dir):
        summarize_job(job_dir)
        result_store.write_json(job_dir / "runner-meta.json", meta)
        model_dir, summary = result_store.export_job(
            job_dir,
            results_root=results_root,
            repo_root=ROOT,
            run_meta=meta,
            merge_existing_attempts=merge_existing_attempts,
        )
        print_results_summary(model_dir, summary)
        exported = True
    if completed.returncode == 0 and not exported:
        print(
            f"Harbor exited before {job_name} reached a terminal state; "
            f"resume it from {job_dir}.",
            file=sys.stderr,
        )
        return 1, False
    return completed.returncode, exported


def attempt_job_name(group: str, attempt: int) -> str:
    validate_job_name(group)
    return validate_job_name(group if attempt == 1 else f"{group}-attempt{attempt}")


def endpoint_job_name(
    group: str, attempt: int, index: int, count: int, *, migrated: bool = False
) -> str:
    name = attempt_job_name(group, attempt)
    if count == 1 and migrated:
        return validate_job_name(f"{name}-resumed")
    return validate_job_name(
        name if count == 1 else f"{name}-endpoint{index + 1}"
    )


def build_attempt_jobs(
    *,
    meta: dict[str, Any],
    base_config: dict[str, Any],
    tasks: list[str],
    attempt: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    configured_endpoint = (
        ((base_config.get("agents") or [{}])[0].get("kwargs") or {}).get("api_base")
        if base_config.get("agents")
        else None
    )
    fallback_endpoint = meta.get("endpoint") or configured_endpoint
    endpoints = list(
        meta.get("endpoints")
        or ([fallback_endpoint] if fallback_endpoint else [""])
    )
    group = str(meta.get("attempt_group") or meta["job_name"])
    migrated = bool(meta.get("topology_migration"))
    jobs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    suite_id = str((meta.get("suite") or {}).get("id") or "")
    weights = CORE19_TASK_ESTIMATED_MINUTES if suite_id == DEFAULT_SUITE else None
    weighted = bool(
        weights and all(task in weights and weights[task] > 0 for task in tasks)
    )
    shards = partition_tasks(tasks, len(endpoints), weights=weights)
    for index, shard in enumerate(shards):
        if not shard:
            continue
        job_name = endpoint_job_name(
            group, attempt, index, len(endpoints), migrated=migrated
        )
        config = copy.deepcopy(base_config)
        config["job_name"] = job_name
        config["n_attempts"] = 1
        if config.get("agents"):
            config["agents"][0]["kwargs"]["api_base"] = endpoints[index]
        config["datasets"] = select_config_datasets(
            base_config.get("datasets") or [], shard
        )
        child_meta = copy.deepcopy(meta)
        child_meta.update(
            {
                "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "job_name": job_name,
                "executed_tasks": shard,
                "endpoint": endpoints[index],
                "endpoints": endpoints,
                "endpoint_index": index + 1,
                "endpoint_count": len(endpoints),
                "attempt_group": group,
                "attempt_round": attempt,
            }
        )
        if weighted and weights:
            child_meta["scheduling"] = {
                "strategy": "historical-duration-lpt",
                "estimated_minutes": round(sum(weights[task] for task in shard), 1),
            }
        if attempt > 1:
            child_meta["previous_attempt_round"] = attempt - 1
        jobs.append((config, child_meta))
    return jobs


def orchestrator_path(group: str) -> Path:
    validate_job_name(group)
    return JOBS_DIR / group / "orchestrator.json"


def record_orchestrator_round(
    *,
    group: str,
    endpoints: list[str],
    attempt: int,
    jobs: list[tuple[dict[str, Any], dict[str, Any]]],
    results_root: Path,
    datasets: list[dict[str, Any]],
) -> Path:
    path = orchestrator_path(group)
    if path.is_file():
        manifest = result_store.read_json(path)
        if (
            manifest.get("attempt_group") != group
            or manifest.get("endpoints") != endpoints
        ):
            raise RunnerError(
                f"Orchestrator metadata does not match this run: {path.parent}"
            )
    else:
        migration = next(
            (
                child_meta.get("topology_migration")
                for _, child_meta in jobs
                if child_meta.get("topology_migration")
            ),
            None,
        )
        migrated_single_parent = bool(
            path.parent.exists()
            and isinstance(migration, dict)
            and migration.get("source_job") == group
            and (path.parent / "runner-meta.json").is_file()
            and (path.parent / "config.json").is_file()
        )
        if path.parent.exists() and not migrated_single_parent:
            raise RunnerError(f"Job directory already exists: {path.parent}")
        manifest = {
            "schema_version": result_store.SCHEMA_VERSION,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "attempt_group": group,
            "endpoints": endpoints,
            "results_root": str(results_root),
            "datasets": datasets,
            "rounds": {},
        }
        if migrated_single_parent:
            manifest["topology_migration"] = migration
    if "datasets" not in manifest:
        manifest["datasets"] = datasets
    elif manifest.get("datasets") != datasets:
        raise RunnerError(
            f"Orchestrator dataset catalog does not match this run: {path.parent}"
        )
    manifest["rounds"][str(attempt)] = [config["job_name"] for config, _ in jobs]
    result_store.write_json(path, manifest)
    return path.parent


def execute_harbor_jobs(
    *,
    jobs: list[tuple[dict[str, Any], dict[str, Any]]],
    results_root: Path,
    runtime: str,
    merge_existing_attempts: bool,
    dry_run: bool = False,
    campaign_progress: dict[str, int] | None = None,
) -> tuple[int, bool]:
    """Run Harbor jobs concurrently without mixing their terminal renderers."""
    for config, _ in jobs:
        job_name = validate_job_name(str(config["job_name"]))
        job_dir = JOBS_DIR / job_name
        if job_dir.exists():
            raise RunnerError(
                f"Job directory already exists: {job_dir}. Resume the parent "
                "orchestrator to continue it."
            )

    prepared: list[tuple[dict[str, Any], dict[str, Any], Path]] = []
    for config, meta in jobs:
        job_name = validate_job_name(str(config["job_name"]))
        save_runner_meta(job_name, meta)
        config_path = write_config(config)
        print(f"Child job:          {job_name}")
        print(f"Config:             {config_path}")
        prepared.append((config, meta, config_path))
    if dry_run:
        for config, _, _ in prepared:
            print(json.dumps(display_config(config), indent=2))
        return 0, False

    harbor = harbor_command()
    environment = command_environment(runtime)
    processes: list[
        tuple[subprocess.Popen[Any], dict[str, Any], dict[str, Any], Any]
    ] = []
    dashboard: TerminalDashboard | None = None
    try:
        print(
            f"Starting {len(prepared)} Harbor children concurrently. "
            "Child output is isolated into per-job logs.",
            flush=True,
        )
        for index, (config, meta, config_path) in enumerate(prepared, 1):
            job_dir = JOBS_DIR / config["job_name"]
            secure_job_directory(job_dir, create=True)
            result_store.write_json(job_dir / "runner-meta.json", meta)
            command = [*harbor, "run", "--config", str(config_path), "--yes"]
            console_path = job_dir / "harbor-console.log"
            console = console_path.open("w", encoding="utf-8")
            try:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=console,
                    stderr=subprocess.STDOUT,
                )
            except BaseException:
                console.close()
                raise
            processes.append((process, config, meta, console))
            task_count = sum(
                len(dataset.get("task_names") or [])
                for dataset in config.get("datasets") or []
            )
            endpoint = str(meta.get("endpoint") or "unknown endpoint")
            estimated_minutes = (meta.get("scheduling") or {}).get(
                "estimated_minutes"
            )
            estimate = (
                f", estimated {format_live_elapsed(float(estimated_minutes) * 60)}"
                if estimated_minutes is not None
                else ""
            )
            print(
                f"Child {index}/{len(prepared)}: {task_count} task(s){estimate} "
                f"on {endpoint}",
                flush=True,
            )
            print(f"  Log: {console_path}", flush=True)

        return_codes: list[int | None] = [None] * len(processes)
        pending = set(range(len(processes)))
        dashboard = TerminalDashboard()
        next_status = time.monotonic() + 60
        while pending:
            for index in list(pending):
                process, config, meta, console = processes[index]
                code = process.poll()
                if code is None:
                    continue
                return_codes[index] = process.wait()
                console.close()
                pending.remove(index)
                if not dashboard.enabled:
                    endpoint = str(meta.get("endpoint") or "unknown endpoint")
                    print(
                        f"Child {index + 1}/{len(processes)} finished "
                        f"with exit code {code}: {endpoint}",
                        flush=True,
                    )

            now = time.monotonic()
            if dashboard.enabled:
                dashboard.render(
                    [
                        "Terminal-Bench live progress (1s refresh; Ctrl+C cleans up)",
                        overall_live_status(
                            [config for _, config, _, _ in processes],
                            campaign_progress,
                        ),
                        *[
                            f"  [{index + 1}] "
                            f"{child_live_status(config, meta, return_codes[index])}"
                            for index, (_, config, meta, _) in enumerate(processes)
                        ],
                    ]
                )
            elif pending and now >= next_status:
                print("Harbor status:", flush=True)
                print(
                    f"  {overall_live_status([config for _, config, _, _ in processes], campaign_progress)}",
                    flush=True,
                )
                for index, (_, config, meta, _) in enumerate(processes):
                    print(
                        f"  [{index + 1}] "
                        f"{child_live_status(config, meta, return_codes[index])}",
                        flush=True,
                    )
                next_status = now + 60
            if pending:
                time.sleep(1)
        if dashboard.enabled:
            dashboard.render(
                [
                    "Terminal-Bench live progress (complete)",
                    overall_live_status(
                        [config for _, config, _, _ in processes],
                        campaign_progress,
                    ),
                    *[
                        f"  [{index + 1}] "
                        f"{child_live_status(config, meta, return_codes[index])}"
                        for index, (_, config, meta, _) in enumerate(processes)
                    ],
                ]
            )
        dashboard.close()
    except KeyboardInterrupt:
        if dashboard:
            dashboard.close()
        print("\nRun interrupted; Harbor children received the interrupt.", file=sys.stderr)
        for process, _, _, _ in processes:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
        for process, _, _, console in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            console.close()
        cleanup_interrupted_containers(
            [config for _, config, _, _ in processes], runtime=runtime
        )
        for _, config, _, _ in processes:
            discard_cancelled_trials(JOBS_DIR / str(config["job_name"]))
        return 130, False
    except OSError as exc:
        if dashboard:
            dashboard.close()
        for process, _, _, console in processes:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
            console.close()
        raise RunnerError(f"Cannot start Harbor child job: {exc}") from exc

    all_exported = True
    final_summary: tuple[Path, dict[str, Any]] | None = None
    for _, config, meta, _ in processes:
        job_dir = JOBS_DIR / config["job_name"]
        if not job_is_terminal(job_dir):
            all_exported = False
            continue
        summarize_job(job_dir)
        result_store.write_json(job_dir / "runner-meta.json", meta)
        final_summary = result_store.export_job(
            job_dir,
            results_root=results_root,
            repo_root=ROOT,
            run_meta=meta,
            merge_existing_attempts=merge_existing_attempts,
        )
    if final_summary:
        print_results_summary(*final_summary)
    return_code = next((code for code in return_codes if code not in (None, 0)), 0)
    if return_code == 0 and not all_exported:
        print(
            "Harbor exited before every child reached a terminal state; "
            "resume the orchestrator.",
            file=sys.stderr,
        )
        return_code = 1
    return return_code, all_exported


def execute_attempt_round(
    *,
    meta: dict[str, Any],
    base_config: dict[str, Any],
    tasks: list[str],
    attempt: int,
    results_root: Path,
    runtime: str,
    merge_existing_attempts: bool,
    dry_run: bool = False,
    campaign_progress: dict[str, int] | None = None,
) -> tuple[int, bool]:
    jobs = build_attempt_jobs(
        meta=meta, base_config=base_config, tasks=tasks, attempt=attempt
    )
    endpoints = list(jobs[0][1]["endpoints"])
    if len(endpoints) == 1 and not meta.get("topology_migration"):
        config, child_meta = jobs[0]
        job_dir = JOBS_DIR / config["job_name"]
        if job_dir.exists() and not dry_run:
            existing_meta = load_resume_meta(job_dir)
            if (
                str(existing_meta.get("attempt_group") or "")
                != str(child_meta.get("attempt_group") or "")
                or int(existing_meta.get("attempt_round") or 1) != attempt
                or str(existing_meta.get("profile_hash") or "")
                != str(child_meta.get("profile_hash") or "")
            ):
                raise RunnerError(
                    "Existing conditional-attempt job does not match the expected "
                    f"attempt state: {job_dir}"
                )
            secure_job_directory(job_dir)
            if job_is_terminal(job_dir):
                print(
                    f"Attempt {attempt}:         exporting completed job "
                    f"{config['job_name']}"
                )
                summarize_job(job_dir)
                model_dir, summary = result_store.export_job(
                    job_dir,
                    results_root=results_root,
                    repo_root=ROOT,
                    run_meta=existing_meta,
                    merge_existing_attempts=True,
                )
                print_results_summary(model_dir, summary)
                return 0, True
            print(
                f"Attempt {attempt}:         resuming existing job "
                f"{config['job_name']}"
            )
            return_code, exported, _ = resume_harbor_job(
                job_dir=job_dir,
                results_root=results_root,
                runtime=runtime,
            )
            return return_code, exported
        return execute_harbor_job(
            config=config,
            meta=child_meta,
            results_root=results_root,
            runtime=runtime,
            merge_existing_attempts=merge_existing_attempts,
            dry_run=dry_run,
        )
    group = str(meta.get("attempt_group") or meta["job_name"])
    if not dry_run:
        parent = record_orchestrator_round(
            group=group,
            endpoints=endpoints,
            attempt=attempt,
            jobs=jobs,
            results_root=results_root,
            datasets=base_config.get("datasets") or [],
        )
        print(f"Orchestrator:       {parent}")
    return execute_harbor_jobs(
        jobs=jobs,
        results_root=results_root,
        runtime=runtime,
        merge_existing_attempts=merge_existing_attempts,
        dry_run=dry_run,
        campaign_progress=campaign_progress,
    )


def continue_conditional_attempts(
    *,
    meta: dict[str, Any],
    base_config: dict[str, Any],
    completed_round: int,
    results_root: Path,
    runtime: str,
    dry_run: bool = False,
) -> int:
    profile = meta.get("evaluation_profile") or {}
    # Older result sets stored the retry budget inside the evaluation profile.
    # New runs keep it in run metadata so changing the budget does not change
    # benchmark identity and prevent attempts from being merged.
    max_attempts = int(meta.get("max_attempts") or profile.get("attempts") or 1)
    if completed_round >= max_attempts:
        return 0
    requested_tasks = list(meta.get("requested_tasks") or [])
    model_dir = result_store.model_results_dir_for_run(results_root, meta)
    group = str(meta.get("attempt_group") or meta["job_name"])
    for attempt in range(completed_round + 1, max_attempts + 1):
        pending = result_store.tasks_requiring_attempt(
            model_dir, requested_tasks, meta["profile_hash"], max_attempts
        )
        if not pending:
            print(f"Attempt {attempt}:         not needed; every task already passed")
            return 0
        print(
            f"Attempt {attempt}:         retrying {len(pending)} "
            "non-passing task(s)"
        )
        return_code, exported = execute_attempt_round(
            meta=meta,
            base_config=base_config,
            tasks=pending,
            attempt=attempt,
            results_root=results_root,
            runtime=runtime,
            merge_existing_attempts=True,
            dry_run=dry_run,
        )
        if return_code != 0 or not exported:
            return return_code
    return 0


def retry_state(
    model_dir: Path, max_attempts: int
) -> tuple[dict[str, Any], list[str], int]:
    """Load a uniform result set and return only failed tasks below the cap."""
    meta_path = model_dir / "run-meta.json"
    if not meta_path.is_file():
        raise RunnerError(f"Result metadata not found: {meta_path}")
    meta = result_store.read_json(meta_path)
    result_paths = sorted(model_dir.glob("results-*.json"))
    if not result_paths:
        raise RunnerError(f"No task results found below {model_dir}")

    rows = [result_store.read_json(path) for path in result_paths]
    profile_hashes = {
        result_store.evaluation_profile_hash(row.get("evaluation_profile") or {})
        for row in rows
    }
    if len(profile_hashes) != 1:
        raise RunnerError(
            "Result directory contains multiple evaluation profiles; "
            "cannot safely merge retries"
        )
    profile_hash = profile_hashes.pop()
    if result_store.evaluation_profile_hash(
        meta.get("evaluation_profile") or {}
    ) != profile_hash:
        # A single-task add-on can replace run-meta.json. Its profile must still
        # match every accumulated task before this command is allowed to merge.
        raise RunnerError("run-meta.json does not match the task result profile")

    pending = sorted(
        str(row["task"])
        for row in rows
        if not row.get("passed")
        and len(row.get("attempts") or []) < max_attempts
    )
    completed_round = min(
        (len(row.get("attempts") or []) for row in rows if str(row.get("task")) in pending),
        default=max_attempts,
    )
    meta = copy.deepcopy(meta)
    meta["evaluation_profile"] = result_store.evaluation_identity(
        meta.get("evaluation_profile") or {}
    )
    meta["profile_hash"] = profile_hash
    return meta, pending, completed_round


def suite_for_result(
    meta: dict[str, Any],
    suite_value: str | Path | None,
    tasks_dir: Path | None,
) -> SuiteManifest:
    try:
        stored_suite = result_store.metadata_suite(meta)
    except ValueError as exc:
        raise RunnerError(str(exc)) from exc
    if suite_value is None and tasks_dir is None and stored_suite:
        suite_id = str(stored_suite.get("id") or "")
        if not (SUITES_DIR / f"{suite_id}.json").is_file():
            raise RunnerError(
                "This result uses a custom suite; pass --suite with its manifest path"
            )
        suite_value = suite_id
    elif suite_value is None and tasks_dir is None:
        # Results exported before suite identities existed are mini-20 results,
        # regardless of the current default for new runs.
        suite_value = LEGACY_SUITE
    suite = load_suite(suite_value, tasks_dir)
    if stored_suite and suite.identity() != stored_suite:
        raise RunnerError(
            "Selected suite manifest does not match the result set's suite identity"
        )
    if not stored_suite and suite.id != LEGACY_SUITE:
        raise RunnerError(
            f"Legacy unscoped results can only be retried with {LEGACY_SUITE}"
        )
    return suite


def retry_failed(args: argparse.Namespace) -> int:
    if args.max_attempts < 2 or args.concurrency < 1:
        raise RunnerError("--max-attempts must be at least 2 and --concurrency positive")
    validate_python_version()
    installed_harbor_version()
    model_dir = args.result_set.resolve()
    meta, pending, completed_round = retry_state(model_dir, args.max_attempts)
    validate_stored_harbor_version(meta)
    if not pending:
        print(f"No failed tasks remain below the {args.max_attempts}-attempt limit.")
        return 0

    suite = suite_for_result(meta, args.suite, args.tasks_dir)
    task_groups = suite.grouped_tasks(pending)
    for tasks_dir, tasks in task_groups:
        validate_tasks(tasks_dir, tasks)
    stored_endpoints = list(meta.get("endpoints") or [])
    if not stored_endpoints and meta.get("endpoint"):
        stored_endpoints = [str(meta["endpoint"])]
    if not stored_endpoints and not args.endpoint and args.endpoints is None:
        raise RunnerError("Result metadata has no endpoint; pass --endpoint or --endpoints")
    endpoints = connection_endpoints(args, fallback=stored_endpoints)
    endpoint = endpoints[0]
    model = str((meta.get("model") or {}).get("id") or "")
    if not model:
        raise RunnerError("Result metadata has no model ID")
    runtime, runtime_description = container_runtime()
    profile = meta.get("evaluation_profile") or {}
    agent_profile = profile.get("agent") or {}
    context_length = int(agent_profile.get("context_length") or 0)
    agent_timeout = int(profile.get("agent_timeout_seconds") or 0)
    if context_length < 1 or agent_timeout < 1:
        raise RunnerError("Stored evaluation profile lacks context length or agent timeout")
    if not args.skip_endpoint_check:
        discover_matching_model(
            endpoints,
            args.api_key,
            preferred=model,
            context_override=context_length,
        )

    runtime_identity = result_store.runtime_identity(meta)
    base_name = validate_job_name(
        args.job_name
        or make_job_name(
            "retry",
            model,
            result_store.metadata_quant(meta),
            engine=runtime_identity.get("engine"),
            backend=runtime_identity.get("backend"),
            inference_profile=result_store.metadata_inference_profile(meta),
            tag=meta.get("tag"),
        )
    )
    retry_meta = copy.deepcopy(meta)
    retry_meta.update(
        {
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "tier": "retry",
            "requested_tasks": pending,
            "executed_tasks": [],
            "task_list_hash": hashlib.sha256("\n".join(pending).encode()).hexdigest(),
            "endpoint": endpoint,
            "endpoints": endpoints,
            "container_runtime": {
                "type": runtime,
                "description": runtime_description,
            },
            "job_name": base_name,
            "attempt_group": base_name,
            "attempt_round": completed_round,
            "max_attempts": args.max_attempts,
        }
    )
    config = build_config(
        job_name=base_name,
        tasks_dir=task_groups[0][0],
        tasks=pending,
        model=model,
        endpoint=endpoint,
        api_key=args.api_key,
        concurrency=args.concurrency,
        context_length=context_length,
        agent_timeout_seconds=agent_timeout,
        keep_containers=args.keep_containers,
        dataset_groups=task_groups,
    )
    print(f"Results:            {model_dir}")
    print(f"Failed tasks:       {len(pending)}")
    for task in pending:
        print(f"  {task}")
    print(f"Attempts:           up to {args.max_attempts}; stop after first pass")
    if len(endpoints) == 1:
        print(f"Concurrency:        {args.concurrency}")
    else:
        print(
            f"Concurrency:        {args.concurrency} per endpoint; "
            f"up to {args.concurrency * len(endpoints)} total"
        )
    return continue_conditional_attempts(
        meta=retry_meta,
        base_config=config,
        completed_round=completed_round,
        results_root=result_store.results_root_from_model_dir(model_dir, retry_meta),
        runtime=runtime,
        dry_run=args.dry_run,
    )


def load_resume_meta(job_dir: Path) -> dict[str, Any]:
    candidates = [job_dir / "runner-meta.json", runner_meta_path(job_dir.name)]
    for path in candidates:
        if path.is_file():
            return result_store.read_json(path)
    raise RunnerError(f"Runner metadata not found for {job_dir}")


def config_task_names(config: dict[str, Any]) -> list[str]:
    return [
        str(task)
        for dataset in config.get("datasets") or []
        for task in dataset.get("task_names") or []
    ]


def completed_trial_progress(
    rows: list[tuple[Path, dict[str, Any]]], *, total: int
) -> dict[str, int]:
    completed: set[str] = set()
    passed: set[str] = set()
    graded: set[str] = set()
    errors: set[str] = set()
    for _, result in rows:
        try:
            task = result_store.task_id_from_trial(result)
        except ValueError:
            continue
        completed.add(task)
        reward = (
            ((result.get("verifier_result") or {}).get("rewards") or {}).get(
                "reward"
            )
        )
        if reward is not None:
            graded.add(task)
        if reward == 1:
            passed.add(task)
        if result.get("exception_info"):
            errors.add(task)
    return {
        "total": total,
        "completed": len(completed),
        "passed": len(passed),
        "graded": len(graded),
        "errors": len(errors),
    }


def harbor_job_process_is_live(job_name: str) -> bool:
    """Return whether a Harbor run/resume process still references this job."""
    try:
        completed = subprocess.run(
            ["ps", "-eo", "args="],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    if completed.returncode != 0:
        return False
    return any(
        job_name in line
        and (" harbor run " in f" {line} " or " harbor job resume " in f" {line} ")
        for line in completed.stdout.splitlines()
    )


def validate_resume_endpoints(
    meta: dict[str, Any], endpoints: list[str], api_key: str, *, skip_check: bool
) -> None:
    if skip_check:
        return
    model_id = str((meta.get("model") or {}).get("id") or "")
    if not model_id:
        raise RunnerError("Stored run metadata has no model ID")
    profile = meta.get("evaluation_profile") or {}
    stored_metadata = stable_model_metadata(
        profile.get("model_metadata")
        or (meta.get("model") or {}).get("endpoint_metadata")
        or {}
    )
    stored_context = int(((profile.get("agent") or {}).get("context_length") or 0))
    if stored_context < 1:
        raise RunnerError("Stored run metadata has no valid model context length")
    for endpoint in endpoints:
        discovered_model, metadata = discover_model(
            endpoint, api_key, preferred=model_id
        )
        if discovered_model != model_id:
            raise RunnerError(
                f"Endpoint {endpoint} advertises {discovered_model!r}; "
                f"expected {model_id!r}"
            )
        if stable_model_metadata(metadata) != stored_metadata:
            raise RunnerError(
                f"Endpoint {endpoint} model metadata does not match the stored run"
            )
        try:
            advertised_context = model_context_length(metadata)
        except RunnerError:
            advertised_context = None
        if advertised_context is not None and advertised_context != stored_context:
            raise RunnerError(
                f"Endpoint {endpoint} advertises {advertised_context:,} context "
                f"tokens; expected {stored_context:,}"
            )


def resume_with_endpoint_redistribution(
    *,
    job_dir: Path,
    endpoints: list[str],
    results_root: Path,
    runtime: str,
    api_key: str | None,
    skip_endpoint_check: bool,
    concurrency: int | None,
) -> int:
    """Convert an interrupted single Harbor job into a distributed orchestrator."""
    if not endpoints:
        raise RunnerError("Endpoint-changing resume requires at least one URL")
    if job_dir.parent.resolve() != JOBS_DIR.resolve():
        raise RunnerError(
            "Endpoint-changing resume requires a job directly below the repository's "
            f"jobs directory: {JOBS_DIR}"
        )
    job_name = validate_job_name(job_dir.name)
    if harbor_job_process_is_live(job_name):
        raise RunnerError(
            f"Harbor is still running for {job_name}; interrupt it with Ctrl+C and "
            "wait for cleanup before changing endpoints"
        )
    meta = load_resume_meta(job_dir)
    validate_stored_harbor_version(meta)
    group = str(meta.get("attempt_group") or job_name)
    if group != job_name:
        raise RunnerError(
            "Change endpoints by resuming the original single job or its existing "
            "orchestrator parent, not an endpoint child"
        )
    config = saved_job_config(job_name, job_dir)
    discard_cancelled_trials(job_dir)
    stored_endpoints = list(meta.get("endpoints") or [meta.get("endpoint")])
    if endpoints == stored_endpoints:
        raise RunnerError(
            "The requested endpoints match the stored topology; resume without "
            "--endpoint/--endpoints"
        )
    if concurrency is not None:
        if concurrency < 1:
            raise RunnerError("--concurrency must be positive")
        config["n_concurrent_trials"] = concurrency
    stored_api_key = str(
        (((config.get("agents") or [{}])[0].get("kwargs") or {})
        .get("llm_kwargs") or {})
        .get("api_key")
        or "local"
    )
    operational_api_key = api_key or stored_api_key
    validate_resume_endpoints(
        meta,
        endpoints,
        operational_api_key,
        skip_check=skip_endpoint_check,
    )
    if config.get("agents"):
        llm_kwargs = config["agents"][0]["kwargs"].setdefault("llm_kwargs", {})
        llm_kwargs["api_key"] = operational_api_key

    configured_tasks = config_task_names(config)
    finished_rows = result_store.trial_results(job_dir)
    campaign_progress = completed_trial_progress(
        finished_rows, total=len(configured_tasks)
    )
    finished_tasks = {
        result_store.task_id_from_trial(result) for _, result in finished_rows
    }
    unexpected = finished_tasks - set(configured_tasks)
    if unexpected:
        raise RunnerError(
            "Finished trial results are not present in the stored job config: "
            + ", ".join(sorted(unexpected))
        )
    if finished_rows:
        model_dir, summary = result_store.export_job(
            job_dir,
            results_root=results_root,
            repo_root=ROOT,
            run_meta=meta,
            merge_existing_attempts=True,
        )
        print(f"Results:             {model_dir}")
        completed = campaign_progress["completed"]
        passed = campaign_progress["passed"]
        pass_text = f"{passed}/{completed} ({passed / completed:.1%})" if completed else "n/a"
        print(
            f"Campaign progress:   {completed}/{len(configured_tasks)} completed; "
            f"pass {pass_text}"
        )
    remaining = [task for task in configured_tasks if task not in finished_tasks]
    print(f"Preserved:           {len(finished_tasks)} finished task(s)")
    print(f"Redistributing:      {len(remaining)} unfinished task(s)")

    migrated_meta = copy.deepcopy(meta)
    migrated_meta["endpoint"] = endpoints[0]
    migrated_meta["endpoints"] = endpoints
    migrated_meta["executed_tasks"] = remaining
    migrated_meta["topology_migration"] = {
        "source_job": job_name,
        "previous_endpoints": stored_endpoints,
        "new_endpoints": endpoints,
        "preserved_tasks": [
            task for task in configured_tasks if task in finished_tasks
        ],
        "redistributed_tasks": remaining,
        "migrated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    attempt = int(meta.get("attempt_round") or 1)
    if remaining:
        return_code, exported = execute_attempt_round(
            meta=migrated_meta,
            base_config=config,
            tasks=remaining,
            attempt=attempt,
            results_root=results_root,
            runtime=runtime,
            merge_existing_attempts=True,
            campaign_progress=campaign_progress,
        )
        if return_code != 0 or not exported:
            return return_code
    return continue_conditional_attempts(
        meta=migrated_meta,
        base_config=config,
        completed_round=attempt,
        results_root=results_root,
        runtime=runtime,
    )


def resume_harbor_job(
    *,
    job_dir: Path,
    results_root: Path,
    runtime: str,
) -> tuple[int, bool, dict[str, Any]]:
    config_path = job_dir / "config.json"
    if not config_path.is_file():
        raise RunnerError(f"Harbor job config not found: {config_path}")
    secure_job_directory(job_dir)
    meta = load_resume_meta(job_dir)
    validate_stored_harbor_version(meta)
    discard_cancelled_trials(job_dir)
    command = [*harbor_command(), "job", "resume", "--job-path", str(job_dir)]
    print(f"Resuming:           {' '.join(command)}", flush=True)
    try:
        completed = subprocess.run(
            command, cwd=ROOT, env=command_environment(runtime), check=False
        )
    except KeyboardInterrupt:
        print("\nResume interrupted; Harbor received the interrupt.", file=sys.stderr)
        cleanup_interrupted_containers(
            [result_store.read_json(config_path)], runtime=runtime
        )
        return 130, False, meta

    exported = False
    if job_is_terminal(job_dir):
        summarize_job(job_dir)
        model_dir, summary = result_store.export_job(
            job_dir,
            results_root=results_root,
            repo_root=ROOT,
            run_meta=meta,
            merge_existing_attempts=True,
        )
        print_results_summary(model_dir, summary)
        exported = True
    if completed.returncode == 0 and not exported:
        print(
            f"Harbor exited before {job_dir.name} reached a terminal state; "
            "it remains resumable.",
            file=sys.stderr,
        )
        return 1, False, meta
    return completed.returncode, exported, meta


def saved_job_config(job_name: str, job_dir: Path) -> dict[str, Any]:
    validate_job_name(job_name)
    candidates = [
        job_dir / "config.json",
        ROOT / ".runner" / "configs" / f"{job_name}.json",
    ]
    for path in candidates:
        if path.is_file():
            return result_store.read_json(path)
    raise RunnerError(f"Harbor job config not found for {job_dir}")


def resume_orchestrator_job(
    *,
    parent_dir: Path,
    results_root: Path,
    runtime: str,
) -> int:
    manifest_path = parent_dir / "orchestrator.json"
    if not manifest_path.is_file():
        raise RunnerError(f"Orchestrator metadata not found: {manifest_path}")
    manifest = result_store.read_json(manifest_path)
    rounds = manifest.get("rounds") or {}
    if not rounds:
        raise RunnerError(f"Orchestrator has no child jobs: {manifest_path}")
    latest_round = max(int(value) for value in rounds)
    child_names = list(rounds[str(latest_round)])
    print(f"Resuming orchestrator: {parent_dir}")

    latest_meta: dict[str, Any] | None = None
    base_config: dict[str, Any] | None = None
    for child_name in child_names:
        validate_job_name(child_name)
        job_dir = JOBS_DIR / child_name
        meta = load_resume_meta(job_dir)
        validate_stored_harbor_version(meta)
        config = saved_job_config(child_name, job_dir)
        latest_meta = latest_meta or meta
        base_config = base_config or config
        if not job_dir.exists():
            return_code, exported = execute_harbor_job(
                config=config,
                meta=meta,
                results_root=results_root,
                runtime=runtime,
                merge_existing_attempts=True,
            )
            if return_code != 0 or not exported:
                return return_code
            continue
        secure_job_directory(job_dir)
        if job_is_terminal(job_dir):
            model_dir, summary = result_store.export_job(
                job_dir,
                results_root=results_root,
                repo_root=ROOT,
                run_meta=meta,
                merge_existing_attempts=True,
            )
            print_results_summary(model_dir, summary)
            continue
        return_code, exported, _ = resume_harbor_job(
            job_dir=job_dir,
            results_root=results_root,
            runtime=runtime,
        )
        if return_code != 0 or not exported:
            return return_code

    if latest_meta is None or base_config is None:
        raise RunnerError(f"Orchestrator has no usable child metadata: {parent_dir}")
    if manifest.get("datasets"):
        base_config["datasets"] = copy.deepcopy(manifest["datasets"])
    return continue_conditional_attempts(
        meta=latest_meta,
        base_config=base_config,
        completed_round=latest_round,
        results_root=results_root,
        runtime=runtime,
    )


def optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = result_store.read_json(path)
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def job_datetime(value: Any) -> dt.datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed


def display_job_datetime(value: dt.datetime | None) -> str:
    if value is None:
        return "unknown"
    return value.astimezone().strftime("%Y-%m-%d %H:%M %Z")


def live_harbor_jobs(names: set[str]) -> set[str]:
    if not names:
        return set()
    try:
        completed = subprocess.run(
            ["ps", "-eo", "args="], capture_output=True, text=True, check=False
        )
    except OSError:
        return set()
    if completed.returncode != 0:
        return set()
    live: set[str] = set()
    for line in completed.stdout.splitlines():
        padded = f" {line} "
        if " harbor run " not in padded and " harbor job resume " not in padded:
            continue
        live.update(name for name in names if name in line)
    return live


def campaign_context(job_dir: Path) -> tuple[dict[str, Any], dict[str, Any], list[Path]]:
    """Return representative metadata/config and every job belonging to a campaign."""
    manifest = optional_json(job_dir / "orchestrator.json") or {}
    member_names = {
        str(name)
        for names in (manifest.get("rounds") or {}).values()
        for name in names
    }
    for candidate in JOBS_DIR.iterdir() if JOBS_DIR.is_dir() else []:
        if not candidate.is_dir():
            continue
        candidate_meta = optional_json(candidate / "runner-meta.json")
        if candidate_meta and str(candidate_meta.get("attempt_group") or "") == job_dir.name:
            member_names.add(candidate.name)
    members = [job_dir]
    members.extend(
        JOBS_DIR / name for name in sorted(member_names) if name != job_dir.name
    )

    meta = optional_json(job_dir / "runner-meta.json")
    config = optional_json(job_dir / "config.json")
    for member in members:
        meta = meta or optional_json(member / "runner-meta.json")
        config = config or optional_json(member / "config.json")
        if meta and config:
            break
    if meta is None:
        meta = {}
    if config is None:
        config = {}
    if manifest.get("datasets"):
        config = copy.deepcopy(config)
        config["datasets"] = copy.deepcopy(manifest["datasets"])
    return meta, config, members


def discover_job_campaigns() -> list[dict[str, Any]]:
    """Discover top-level campaigns and summarize them for human selection."""
    if not JOBS_DIR.is_dir():
        return []
    directories = [path for path in JOBS_DIR.iterdir() if path.is_dir()]
    metas = {
        path.name: optional_json(path / "runner-meta.json") for path in directories
    }
    parent_names = {
        path.name for path in directories if (path / "orchestrator.json").is_file()
    }
    for path in directories:
        meta = metas[path.name]
        if not meta:
            continue
        group = str(meta.get("attempt_group") or meta.get("job_name") or path.name)
        if group == path.name or not (JOBS_DIR / group).is_dir():
            parent_names.add(path.name)

    all_names = {path.name for path in directories}
    live_names = live_harbor_jobs(all_names)
    campaigns: list[dict[str, Any]] = []
    for name in parent_names:
        job_dir = JOBS_DIR / name
        meta, config, members = campaign_context(job_dir)
        manifest = optional_json(job_dir / "orchestrator.json") or {}
        requested = [str(task) for task in meta.get("requested_tasks") or []]
        if not requested:
            requested = config_task_names(config)
        total = len(requested)
        attempts_by_task: dict[str, set[int]] = {}
        passed_tasks: set[str] = set()
        timestamps: list[dt.datetime] = []
        aggregate_times: list[dt.datetime] = []
        for member in members:
            member_meta = optional_json(member / "runner-meta.json") or meta
            attempt = int(member_meta.get("attempt_round") or 1)
            aggregate = optional_json(member / "result.json") or {}
            for key in ("finished_at", "updated_at"):
                parsed = job_datetime(aggregate.get(key))
                if parsed:
                    aggregate_times.append(parsed)
            if not member.is_dir():
                continue
            for _, result in result_store.trial_results(member):
                exception = result.get("exception_info") or {}
                if exception.get("exception_type") == "CancelledError":
                    continue
                try:
                    task = result_store.task_id_from_trial(result)
                except ValueError:
                    continue
                attempts_by_task.setdefault(task, set()).add(attempt)
                reward = (((result.get("verifier_result") or {}).get("rewards") or {}).get("reward"))
                if reward == 1:
                    passed_tasks.add(task)
                parsed = job_datetime(result.get("finished_at"))
                if parsed:
                    timestamps.append(parsed)

        profile = meta.get("evaluation_profile") or {}
        max_attempts = int(
            meta.get("max_attempts") or profile.get("attempts") or 1
        )
        policy_complete = bool(requested) and all(
            task in passed_tasks
            or max(attempts_by_task.get(task) or {0}) >= max_attempts
            for task in requested
        )
        member_live = any(member.name in live_names for member in members)
        status = "RUNNING" if member_live else "COMPLETE" if policy_complete else "INCOMPLETE"

        started_candidates = [
            value
            for value in (
                job_datetime(meta.get("created_at")),
                job_datetime(manifest.get("created_at")),
                *(
                    job_datetime((optional_json(member / "result.json") or {}).get("started_at"))
                    for member in members
                ),
            )
            if value is not None
        ]
        started = min(started_candidates) if started_candidates else dt.datetime.fromtimestamp(
            job_dir.stat().st_mtime, tz=dt.timezone.utc
        )
        last_activity = max([*timestamps, *aggregate_times], default=None)
        endpoints = list(
            manifest.get("endpoints")
            or meta.get("endpoints")
            or ([meta.get("endpoint")] if meta.get("endpoint") else [])
        )
        model = meta.get("model") or {}
        platform = meta.get("platform") or {}
        agent_profile = profile.get("agent") or {}
        campaigns.append(
            {
                "job_dir": job_dir,
                "name": name,
                "status": status,
                "complete": policy_complete,
                "live": member_live,
                "started": started,
                "last_activity": last_activity,
                "completed_tasks": len(attempts_by_task),
                "passed_tasks": len(passed_tasks),
                "total_tasks": total,
                "max_attempts": max_attempts,
                "meta": meta,
                "config": config,
                "members": members,
                "orchestrated": bool(manifest),
                "endpoints": endpoints,
                "suite": (meta.get("suite") or {}).get("id") or "legacy",
                "tier": meta.get("tier") or "unknown",
                "model_name": model.get("name") or model.get("id") or "unknown model",
                "model_id": model.get("id") or "unknown",
                "quant": meta.get("quant") or profile.get("quant") or "unspecified",
                "inference_profile": meta.get("inference_profile") or profile.get("inference_profile") or "default",
                "platform": platform.get("name") if isinstance(platform, dict) else platform,
                "platform_id": platform.get("id") if isinstance(platform, dict) else platform,
                "engine": meta.get("engine") or profile.get("engine") or "unknown",
                "engine_version": meta.get("engine_version") or profile.get("engine_version"),
                "backend": meta.get("backend") or profile.get("backend") or "unknown",
                "backend_version": meta.get("backend_version") or profile.get("backend_version"),
                "context_length": agent_profile.get("context_length"),
            }
        )
    return sorted(campaigns, key=lambda item: item["started"], reverse=True)


def print_job_campaigns(campaigns: list[dict[str, Any]]) -> None:
    if not campaigns:
        print("No benchmark jobs found.")
        return
    print(f"Benchmark campaigns — newest first ({len(campaigns)} shown)\n")
    for index, campaign in enumerate(campaigns, 1):
        total = campaign["total_tasks"]
        progress = (
            f"{campaign['completed_tasks']}/{total} attempted, "
            f"{campaign['passed_tasks']}/{total} passed"
            if total
            else "progress unavailable"
        )
        last_label = (
            "Finished"
            if campaign["complete"]
            else "Updated"
            if campaign["live"]
            else "Stopped"
        )
        runtime = f"{campaign['engine']} / {campaign['backend']}"
        if campaign["backend_version"]:
            runtime += f" {campaign['backend_version']}"
        print(
            f"[{index}] {campaign['status']}  {progress}  "
            f"{campaign['suite']}/{campaign['tier']}"
        )
        print(
            f"    Started {display_job_datetime(campaign['started'])}  ·  "
            f"{last_label} {display_job_datetime(campaign['last_activity'])}"
        )
        print(
            f"    {campaign['model_name']}  ·  quant {campaign['quant']}  ·  "
            f"profile {campaign['inference_profile']}"
        )
        context = campaign.get("context_length")
        context_text = f"  ·  context {int(context):,}" if context else ""
        print(f"    Served: {campaign['model_id']}{context_text}")
        print(
            f"    {campaign['platform'] or campaign['platform_id'] or 'unknown platform'}"
            f"  ·  {runtime}"
        )
        print(f"    Endpoints: {', '.join(campaign['endpoints']) or 'unknown'}")
        print(f"    Job: {campaign['name']}\n")


def prompt_yes_no(prompt: str, *, default: bool = False) -> bool:
    suffix = " [Y/n] " if default else " [y/N] "
    answer = input(prompt + suffix).strip().casefold()
    if not answer:
        return default
    return answer in {"y", "yes"}


def prompt_resume_endpoints(campaign: dict[str, Any]) -> list[str]:
    endpoints = list(campaign["endpoints"])
    print("Stored endpoint topology:")
    for index, endpoint in enumerate(endpoints, 1):
        print(f"  [{index}] {endpoint}")
    if not prompt_yes_no("Change endpoints before resuming?"):
        return endpoints
    if campaign["orchestrated"]:
        raise RunnerError(
            "This is already a distributed orchestrator; its endpoint topology "
            "cannot currently be changed during resume"
        )
    entered = input("New endpoint URL(s), comma-separated: ").strip()
    return parse_endpoints([entered])


def rerun_arguments(campaign: dict[str, Any]) -> list[str]:
    """Reconstruct a fresh run from a campaign's stored benchmark identity."""
    meta = campaign["meta"]
    config = campaign["config"]
    profile = meta.get("evaluation_profile") or {}
    model = meta.get("model") or {}
    platform = meta.get("platform") or {}
    platform_id = platform.get("id") if isinstance(platform, dict) else platform
    platform_name = platform.get("name") if isinstance(platform, dict) else None
    required = {
        "model ID": model.get("id"),
        "model name": model.get("name"),
        "platform": platform_id,
        "engine": campaign.get("engine"),
        "backend": campaign.get("backend"),
        "context length": campaign.get("context_length"),
    }
    missing = [label for label, value in required.items() if not value]
    if missing:
        raise RunnerError(
            "Cannot reconstruct this legacy job; missing " + ", ".join(missing)
        )
    endpoints = list(campaign["endpoints"])
    if not endpoints:
        raise RunnerError("Cannot reconstruct this job; no stored endpoint")
    requested = [str(task) for task in meta.get("requested_tasks") or []]
    tier = str(meta.get("tier") or "")
    argv = ["run"]
    suite_id = str((meta.get("suite") or {}).get("id") or "")
    if not suite_id:
        legacy_suite = load_suite(LEGACY_SUITE)
        if requested and set(requested) <= set(legacy_suite.tasks):
            suite_id = LEGACY_SUITE
        else:
            raise RunnerError(
                "Cannot safely reconstruct this legacy job because its task suite "
                "is not available"
            )
    if suite_id != DEFAULT_SUITE:
        argv.extend(["--suite", suite_id])
    if tier in {"full", "smoke"}:
        argv.extend(["--tier", tier])
    elif len(requested) == 1:
        argv.extend(["--task", requested[0]])
    else:
        raise RunnerError(
            "Cannot reconstruct a non-tier job containing multiple selected tasks"
        )
    if len(endpoints) == 1:
        argv.extend(["--endpoint", endpoints[0]])
    else:
        argv.extend(["--endpoints", ",".join(endpoints)])
    argv.extend(
        [
            "--model", str(model["id"]),
            "--context-length", str(campaign["context_length"]),
            "--platform", str(platform_id),
            "--model-name", str(model["name"]),
            "--engine", str(campaign["engine"]),
            "--backend", str(campaign["backend"]),
            "--attempts", str(campaign["max_attempts"]),
            "--concurrency", str(config.get("n_concurrent_trials") or 1),
            "--agent-timeout", str(profile.get("agent_timeout_seconds") or DEFAULT_AGENT_TIMEOUT_SECONDS),
            "--rerun",
        ]
    )
    optional_values = (
        ("--platform-name", platform_name if platform_name != platform_id else None),
        ("--engine-version", campaign.get("engine_version")),
        ("--backend-version", campaign.get("backend_version")),
        ("--quant", None if campaign.get("quant") == "unspecified" else campaign.get("quant")),
        ("--inference-profile", None if campaign.get("inference_profile") == "default" else campaign.get("inference_profile")),
        ("--tag", meta.get("tag") or profile.get("tag")),
    )
    for flag, value in optional_values:
        if value:
            argv.extend([flag, str(value)])
    api_key = (
        (((config.get("agents") or [{}])[0].get("kwargs") or {}).get("llm_kwargs") or {}).get("api_key")
    )
    if api_key:
        argv.extend(["--api-key", str(api_key)])
    if not bool((config.get("environment") or {}).get("delete", True)):
        argv.append("--keep-containers")
    return argv


def manage_jobs(args: argparse.Namespace) -> int:
    campaigns = discover_job_campaigns()
    if not args.all:
        campaigns = campaigns[: args.limit]
    print_job_campaigns(campaigns)
    if args.list_only or not sys.stdin.isatty() or not sys.stdout.isatty() or not campaigns:
        return 0
    try:
        selection = input("Select a campaign number, or q to quit: ").strip().casefold()
        if selection in {"", "q", "quit"}:
            return 0
        try:
            campaign = campaigns[int(selection) - 1]
        except (ValueError, IndexError):
            raise RunnerError("Invalid campaign selection")
        print(f"\nSelected: {campaign['model_name']} · {campaign['quant']}")
        print(f"Job:      {campaign['name']}")
        if campaign["live"]:
            print("This campaign is currently running; no restart action is available.")
            return 0
        actions = "[r] resume, [n] rerun from scratch, [q] quit" if not campaign["complete"] else "[n] rerun from scratch, [q] quit"
        action = input(f"Action ({actions}): ").strip().casefold()
        if action in {"", "q", "quit"}:
            return 0
        if action in {"r", "resume"} and not campaign["complete"]:
            endpoints = prompt_resume_endpoints(campaign)
            if not prompt_yes_no("Start resume now?"):
                return 0
            resume_argv = [
                "resume",
                str(campaign["job_dir"]),
                "--results-dir",
                str(args.results_dir),
            ]
            if endpoints != campaign["endpoints"]:
                flag = "--endpoint" if len(endpoints) == 1 else "--endpoints"
                resume_argv.extend([flag, ",".join(endpoints)])
            return main(resume_argv)
        if action in {"n", "new", "rerun"}:
            print("This creates a new raw job and reruns every selected task.")
            if not prompt_yes_no("Start a fresh rerun now?"):
                return 0
            return main(rerun_arguments(campaign))
        raise RunnerError("Invalid action")
    except (EOFError, KeyboardInterrupt):
        print()
        return 130


def add_connection_args(parser: argparse.ArgumentParser) -> None:
    connection = parser.add_mutually_exclusive_group()
    connection.add_argument(
        "--endpoint",
        help=f"Host-visible OpenAI-compatible base URL (default: {DEFAULT_ENDPOINT})",
    )
    connection.add_argument(
        "--endpoints",
        nargs="+",
        help="OpenAI-compatible base URLs; comma- or space-separated",
    )
    parser.add_argument("--model", help="Served model ID; auto-detected for one-model endpoints")
    parser.add_argument("--api-key", default=os.getenv("TBENCH_API_KEY", "local"))
    parser.add_argument("--skip-endpoint-check", action="store_true")
    parser.add_argument(
        "--context-length",
        type=int,
        help="Model context capacity; normally discovered from GET /models",
    )


def add_suite_args(
    parser: argparse.ArgumentParser, *, include_selection: bool = False
) -> None:
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--suite",
        help=(
            f"Built-in suite ID or manifest path (default: {DEFAULT_SUITE})"
        ),
    )
    source.add_argument(
        "--tasks-dir",
        type=Path,
        help="Legacy alias valid only for the default vendored tasks directory",
    )
    if include_selection:
        parser.add_argument("--tier", default="full", help="Suite tier (default: full)")
        parser.add_argument("--task", help="Run or validate one explicit suite task")


def add_result_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--platform",
        required=True,
        help="Platform identifier recorded in results (required)",
    )
    parser.add_argument("--platform-name")
    parser.add_argument(
        "--model-name",
        required=True,
        help="Canonical human-readable model family/revision (required)",
    )
    parser.add_argument(
        "--engine",
        required=True,
        help="Inference engine/server implementation, for example llama.cpp or DwarfStar (required)",
    )
    parser.add_argument("--engine-version")
    parser.add_argument(
        "--backend",
        required=True,
        help="Compute backend used by the engine, for example rocm, vulkan, cuda, metal or cpu (required)",
    )
    parser.add_argument("--backend-version")
    parser.add_argument(
        "--quant",
        help="Quantization or numeric-format variant only, for example UD-Q4_K_XL",
    )
    parser.add_argument(
        "--rocm-version",
        help="Deprecated alias for --backend-version; valid only with --backend rocm",
    )
    parser.add_argument("--inference-profile")
    parser.add_argument(
        "--tag",
        help="Optional non-quant, non-profile variant label",
    )


def runtime_args(args: argparse.Namespace) -> tuple[str, str | None, str, str | None]:
    engine = str(args.engine or "").strip()
    backend = str(args.backend or "").strip()
    if not engine or not backend:
        raise RunnerError("--engine and --backend must be non-empty")
    backend_version = args.backend_version
    if args.rocm_version:
        if backend.casefold() != "rocm":
            raise RunnerError("--rocm-version can only be used with --backend rocm")
        if backend_version and backend_version != args.rocm_version:
            raise RunnerError("--backend-version and --rocm-version disagree")
        backend_version = args.rocm_version
    return engine, args.engine_version, backend, backend_version


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Validate endpoint, Harbor, runtime and tasks")
    add_connection_args(doctor)
    add_suite_args(doctor, include_selection=True)
    list_parser = sub.add_parser("list", help="Print task tiers")
    add_suite_args(list_parser)
    run = sub.add_parser("run", help="Run Terminal-Bench-Local")
    add_connection_args(run)
    add_suite_args(run, include_selection=True)
    add_result_args(run)
    run.add_argument(
        "--attempts",
        type=int,
        default=DEFAULT_ATTEMPTS,
        help="Maximum attempts per task; later attempts run only after failure (default: 2)",
    )
    run.add_argument("--concurrency", type=int, default=1)
    run.add_argument("--agent-timeout", type=int, default=DEFAULT_AGENT_TIMEOUT_SECONDS)
    run.add_argument("--keep-containers", action="store_true")
    run.add_argument("--job-name", type=job_name_argument)
    run.add_argument("--rerun", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    summary = sub.add_parser("summary", help="Summarize a Harbor job")
    summary.add_argument("job", type=Path)
    jobs = sub.add_parser(
        "jobs", help="List benchmark campaigns and interactively resume or rerun one"
    )
    jobs.add_argument(
        "--limit", type=int, default=20, help="Newest campaigns to show (default: 20)"
    )
    jobs.add_argument("--all", action="store_true", help="Show every campaign")
    jobs.add_argument(
        "--list-only", action="store_true", help="Print jobs without prompting"
    )
    jobs.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    resume = sub.add_parser(
        "resume",
        help="Resume a job, optionally redistributing unfinished tasks to new endpoints",
    )
    resume.add_argument("job", type=Path)
    resume.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    resume_connection = resume.add_mutually_exclusive_group()
    resume_connection.add_argument(
        "--endpoint",
        help="Replace the endpoint while restarting unfinished trials",
    )
    resume_connection.add_argument(
        "--endpoints",
        nargs="+",
        help="Redistribute unfinished tasks; comma- or space-separated",
    )
    resume.add_argument("--api-key", default=os.getenv("TBENCH_API_KEY"))
    resume.add_argument("--skip-endpoint-check", action="store_true")
    resume.add_argument(
        "--concurrency",
        type=int,
        help="Per-endpoint concurrency override; stored value is preserved by default",
    )
    retry = sub.add_parser(
        "retry-failed", help="Append retries for failed tasks in an existing result set"
    )
    retry.add_argument("result_set", type=Path)
    retry.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_ATTEMPTS,
        help="Maximum accumulated attempts per failed task (default: 2)",
    )
    retry_connection = retry.add_mutually_exclusive_group()
    retry_connection.add_argument("--endpoint", help="Operational endpoint override")
    retry_connection.add_argument(
        "--endpoints",
        nargs="+",
        help="Operational endpoint overrides; comma- or space-separated",
    )
    retry.add_argument("--api-key", default=os.getenv("TBENCH_API_KEY", "local"))
    retry.add_argument("--skip-endpoint-check", action="store_true")
    add_suite_args(retry)
    retry.add_argument("--concurrency", type=int, default=1)
    retry.add_argument("--keep-containers", action="store_true")
    retry.add_argument("--job-name", type=job_name_argument)
    retry.add_argument("--dry-run", action="store_true")
    results = sub.add_parser("results", help="Rebuild the stable results index")
    results.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args(argv)


def print_tiers(suite: SuiteManifest) -> None:
    full = set(suite.tiers.get("full") or ())
    print(
        f"suite: {suite.id} {suite.version} "
        f"[{suite.manifest_hash[:12]}] ({suite.name})"
    )
    for name in suite.tiers:
        tasks = suite.tasks_for(name)
        nested = set(tasks) <= full if full else name == "full"
        print(f"{name}: {len(tasks)} task(s), nested={nested}")
        for task in tasks:
            print(f"  {task}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "list":
            print_tiers(load_suite(args.suite, args.tasks_dir))
            return 0
        if args.command == "summary":
            job = args.job.resolve()
            if (job / "orchestrator.json").is_file():
                summarize_orchestrator(job)
            else:
                summarize_job(job)
            return 0
        if args.command == "jobs":
            if args.limit < 1:
                raise RunnerError("--limit must be positive")
            return manage_jobs(args)
        if args.command == "results":
            root = args.results_dir.resolve()
            rebuilt = result_store.rebuild_all_indexes(root)
            index = rebuilt["legacy"]
            suite_catalog = rebuilt["suite_catalog"]
            print(f"Results:   {root}")
            print(f"Suites:    {len(suite_catalog['suites'])}")
            print(f"Legacy platforms: {len(index['platforms'])}")
            print(f"Legacy models:    {len(index['models'])}")
            print(f"Legacy tasks:     {len(index['tasks'])}")
            return 0
        if args.command == "resume":
            job_dir = args.job.resolve()
            validate_python_version()
            installed_harbor_version()
            runtime, _ = container_runtime()
            results_root = args.results_dir.resolve()
            endpoint_override = args.endpoint is not None or args.endpoints is not None
            if (job_dir / "orchestrator.json").is_file():
                if endpoint_override:
                    manifest = result_store.read_json(job_dir / "orchestrator.json")
                    requested_endpoints = connection_endpoints(args)
                    if requested_endpoints != list(manifest.get("endpoints") or []):
                        raise RunnerError(
                            "Changing endpoints is supported when converting an "
                            "interrupted single job; an existing distributed "
                            "orchestrator cannot yet be repartitioned"
                        )
                return resume_orchestrator_job(
                    parent_dir=job_dir,
                    results_root=results_root,
                    runtime=runtime,
                )
            child_meta = load_resume_meta(job_dir)
            group = str(child_meta.get("attempt_group") or "")
            if group:
                validate_job_name(group)
            parent = orchestrator_path(group).parent if group else None
            if parent and (parent / "orchestrator.json").is_file():
                if endpoint_override:
                    raise RunnerError(
                        "Resume the orchestrator parent without changing its endpoint "
                        "topology"
                    )
                return resume_orchestrator_job(
                    parent_dir=parent,
                    results_root=results_root,
                    runtime=runtime,
                )
            if endpoint_override:
                return resume_with_endpoint_redistribution(
                    job_dir=job_dir,
                    endpoints=connection_endpoints(args),
                    results_root=results_root,
                    runtime=runtime,
                    api_key=args.api_key,
                    skip_endpoint_check=args.skip_endpoint_check,
                    concurrency=args.concurrency,
                )
            return_code, exported, meta = resume_harbor_job(
                job_dir=job_dir,
                results_root=results_root,
                runtime=runtime,
            )
            if return_code == 0 and exported:
                return continue_conditional_attempts(
                    meta=meta,
                    base_config=result_store.read_json(job_dir / "config.json"),
                    completed_round=int(meta.get("attempt_round") or 1),
                    results_root=results_root,
                    runtime=runtime,
                )
            return return_code
        if args.command == "retry-failed":
            return retry_failed(args)

        run_runtime = runtime_args(args) if args.command == "run" else None

        (
            model,
            runtime,
            runtime_description,
            metadata,
            context_length,
            endpoints,
            suite,
            requested,
        ) = check_doctor(args)
        if args.command == "doctor":
            print("Doctor checks passed.")
            return 0
        if run_runtime is None:
            raise RunnerError(f"Unsupported command: {args.command}")
        engine, engine_version, backend, backend_version = run_runtime
        model_name, quant, inference_profile, tag = run_identity_args(args)
        if args.attempts < 1 or args.concurrency < 1 or args.agent_timeout < 1:
            raise RunnerError("--attempts, --concurrency and --agent-timeout must be positive")
        tier = "task" if args.task else args.tier
        task_groups = suite.grouped_tasks(requested)
        tasks_dir = task_groups[0][0]
        suite_identity = suite.identity()
        profile = evaluation_profile(
            tasks_dir=tasks_dir,
            model=model,
            model_metadata=metadata,
            endpoint=endpoints[0],
            endpoints=endpoints,
            engine=engine,
            engine_version=engine_version,
            backend=backend,
            backend_version=backend_version,
            quant=quant,
            inference_profile=inference_profile,
            tag=tag,
            context_length=context_length,
            agent_timeout_seconds=args.agent_timeout,
            suite=suite_identity,
        )
        profile_hash = result_store.evaluation_profile_hash(profile)
        results_root = args.results_dir.resolve()
        model_dir = result_store.model_results_dir(
            results_root,
            args.platform,
            model,
            result_tag(quant, engine, backend, inference_profile, tag),
            suite=suite_identity,
            profile_hash=profile_hash,
        )
        cached = [] if args.rerun else result_store.cached_tasks(
            model_dir, requested, profile_hash
        )
        tasks = [task for task in requested if task not in cached]
        if cached:
            print(f"Cached:             {len(cached)}/{len(requested)} matching task(s)")
        if not tasks:
            print("All requested tasks have matching completed results; use --rerun to run again.")
            return 0
        job_name = validate_job_name(
            args.job_name
            or make_job_name(
                f"{suite.id}-{tier}",
                model,
                quant,
                engine=engine,
                backend=backend,
                inference_profile=inference_profile,
                tag=tag,
            )
        )
        meta = make_run_meta(
            job_name=job_name,
            tier=tier,
            requested_tasks=requested,
            executed_tasks=tasks,
            platform=args.platform,
            platform_name=args.platform_name,
            model=model,
            model_name=model_name,
            model_metadata=metadata,
            engine=engine,
            engine_version=engine_version,
            backend=backend,
            backend_version=backend_version,
            quant=quant,
            inference_profile=inference_profile,
            tag=tag,
            endpoint=endpoints[0],
            endpoints=endpoints,
            runtime=runtime,
            runtime_description=runtime_description,
            profile=profile,
            max_attempts=args.attempts,
            suite=suite_identity,
            task_provenance={
                task: suite.task_identity(task) for task in requested
            },
            attempt_group=job_name,
        )
        config = build_config(
            job_name=job_name,
            tasks_dir=tasks_dir,
            tasks=tasks,
            model=model,
            endpoint=endpoints[0],
            api_key=args.api_key,
            concurrency=args.concurrency,
            context_length=context_length,
            agent_timeout_seconds=args.agent_timeout,
            keep_containers=args.keep_containers,
            dataset_groups=suite.grouped_tasks(tasks),
        )
        print(
            f"Suite:              {suite.id} {suite.version} "
            f"({suite.manifest_hash[:12]})"
        )
        print(f"Tier:               {tier} ({len(tasks)} task(s) to run)")
        print(
            f"Engine:             {engine}"
            f"{f' {engine_version}' if engine_version else ''}"
        )
        print(
            f"Compute backend:    {backend}"
            f"{f' {backend_version}' if backend_version else ''}"
        )
        print(f"Display model:      {model_name}")
        print(f"Quant:              {quant or 'not specified'}")
        print(f"Inference profile:  {inference_profile or 'default / not specified'}")
        if tag:
            print(f"Tag:                {tag}")
        print(f"Attempts:           up to {args.attempts}; stop after first pass")
        if len(endpoints) == 1:
            print(f"Concurrency:        {args.concurrency}")
        else:
            print(
                f"Concurrency:        {args.concurrency} per endpoint; "
                f"up to {args.concurrency * len(endpoints)} total"
            )
        print(f"Agent:              {AGENT_NAME} {AGENT_VERSION}")
        return_code, exported = execute_attempt_round(
            meta=meta,
            base_config=config,
            tasks=tasks,
            attempt=1,
            results_root=results_root,
            runtime=runtime,
            merge_existing_attempts=False,
            dry_run=args.dry_run,
        )
        if args.dry_run or return_code != 0 or not exported:
            return return_code
        return continue_conditional_attempts(
            meta=meta,
            base_config=config,
            completed_round=1,
            results_root=results_root,
            runtime=runtime,
        )
    except RunnerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
