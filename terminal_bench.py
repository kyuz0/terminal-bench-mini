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
import shutil
import subprocess
import sys
from typing import Any
from urllib import error, request

import results as result_store


ROOT = Path(__file__).resolve().parent
DEFAULT_TASKS_DIR = ROOT / "tasks"
DEFAULT_ENDPOINT = "http://localhost:8080/v1"
DEFAULT_AGENT_TIMEOUT_SECONDS = 3 * 60 * 60
DEFAULT_ATTEMPTS = 2
AGENT_NAME = "terminus-2"
AGENT_VERSION = "2.0.0"
SUMMARIZATION_FREE_TOKENS = 8_000
HARBOR_PACKAGE = "harbor==0.20.0"
JOBS_DIR = ROOT / "jobs"
DEFAULT_RESULTS_DIR = ROOT / "results"
TIERS = {
    "smoke": ROOT / "subsets" / "smoke.txt",
    "full": ROOT / "subsets" / "full.txt",
}


class RunnerError(RuntimeError):
    pass


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
    kind = "podman" if "podman" in description.lower() else "docker"
    return kind, description


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
    model_tag: str | None = None,
    *,
    engine: str | None = None,
    backend: str | None = None,
) -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    identity_parts = [model]
    if model_tag:
        identity_parts.append(model_tag)
    identity_parts.extend(value for value in (engine, backend) if value)
    identity = "--".join(identity_parts)
    slug = result_store.stable_slug(identity, limit=140)
    return f"{stamp}-terminal-bench-local-{tier}-{AGENT_NAME}-{slug}"


def result_tag(model_tag: str | None, engine: str, backend: str) -> str:
    parts = [engine, backend]
    if model_tag:
        parts.append(model_tag)
    parts.append(AGENT_NAME)
    return "-".join(parts)


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
    engine: str,
    engine_version: str | None,
    backend: str,
    backend_version: str | None,
    inference_profile: str | None,
    context_length: int,
    agent_timeout_seconds: int,
) -> dict[str, Any]:
    return {
        "benchmark": "Terminal-Bench-Local",
        "terminal_bench_version": "2.1",
        "terminal_bench_revision": checkout_revision(tasks_dir),
        "harbor_version": HARBOR_PACKAGE.removeprefix("harbor=="),
        "model_id": model,
        "model_metadata": stable_model_metadata(model_metadata),
        "endpoint": endpoint,
        "engine": engine,
        "engine_version": engine_version,
        "backend": backend,
        "backend_version": backend_version,
        "inference_profile": inference_profile,
        "agent": {
            "name": AGENT_NAME,
            "version": AGENT_VERSION,
            "context_length": context_length,
            "context_summarization": True,
            "summarization_free_tokens": SUMMARIZATION_FREE_TOKENS,
        },
        "agent_timeout_seconds": agent_timeout_seconds,
    }


def make_run_meta(
    *,
    job_name: str,
    tier: str,
    requested_tasks: list[str],
    executed_tasks: list[str],
    platform: str,
    platform_name: str | None,
    model: str,
    model_metadata: dict[str, Any],
    engine: str,
    engine_version: str | None,
    backend: str,
    backend_version: str | None,
    model_tag: str | None,
    inference_profile: str | None,
    endpoint: str,
    runtime: str,
    runtime_description: str,
    profile: dict[str, Any],
    max_attempts: int,
    attempt_group: str | None = None,
    attempt_round: int = 1,
) -> dict[str, Any]:
    return {
        "schema_version": result_store.SCHEMA_VERSION,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "benchmark": "Terminal-Bench-Local",
        "harness": "Harbor",
        "terminal_bench_version": "2.1",
        "tier": tier,
        "requested_tasks": requested_tasks,
        "executed_tasks": executed_tasks,
        "task_list_hash": hashlib.sha256("\n".join(requested_tasks).encode()).hexdigest(),
        "platform": {"id": platform, "name": platform_name or platform},
        "model": {
            "name": result_store.model_name(model),
            "id": model,
            "endpoint_metadata": model_metadata,
        },
        "engine": engine,
        "engine_version": engine_version,
        "backend": backend,
        "backend_version": backend_version,
        "model_tag": model_tag,
        "result_tag": result_tag(model_tag, engine, backend),
        "inference_profile": inference_profile,
        "endpoint": endpoint,
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
) -> dict[str, Any]:
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
        "datasets": [{"path": str(tasks_dir), "task_names": tasks}],
    }


def write_config(config: dict[str, Any]) -> Path:
    config_dir = ROOT / ".runner" / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"{config['job_name']}.json"
    result_store.write_json(path, config)
    return path


def runner_meta_path(job_name: str) -> Path:
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


def check_doctor(
    args: argparse.Namespace,
) -> tuple[str, str, str, dict[str, Any], int]:
    tasks_dir = args.tasks_dir.resolve()
    all_tasks = sorted(set().union(*(load_tier(name) for name in TIERS)))
    validate_tasks(tasks_dir, all_tasks)
    runtime, runtime_description = container_runtime()
    model = args.model
    metadata: dict[str, Any] = {}
    if not args.skip_endpoint_check:
        discovered, metadata = discover_model(args.endpoint, args.api_key, preferred=model)
        model = model or discovered
    if not model:
        raise RunnerError("Cannot determine model ID; pass --model")
    print(f"Endpoint:           {args.endpoint}")
    print(f"Model:              {model}")
    print(f"Container runtime:  {runtime_description}")
    print(f"Tasks:              {tasks_dir} ({len(all_tasks)} selected IDs validated)")
    print(f"Harbor command:     {' '.join(harbor_command())}")
    context = model_context_length(metadata, args.context_length)
    print(f"Model context:      {context:,} tokens")
    return model, runtime, runtime_description, metadata, context


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
    job_name = config["job_name"]
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
        print(json.dumps(config, indent=2))
        return 0, False
    job_dir.mkdir(parents=True)
    result_store.write_json(job_dir / "runner-meta.json", meta)
    command = [*harbor_command(), "run", "--config", str(config_path), "--yes"]
    print(f"Running:            {' '.join(command)}", flush=True)
    try:
        completed = subprocess.run(
            command, cwd=ROOT, env=command_environment(runtime), check=False
        )
    except KeyboardInterrupt:
        print("\nRun interrupted; Harbor received the interrupt.", file=sys.stderr)
        return 130, False
    exported = False
    if (job_dir / "result.json").is_file():
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
    return completed.returncode, exported


def attempt_job_name(group: str, attempt: int) -> str:
    return group if attempt == 1 else f"{group}-attempt{attempt}"


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
    model_dir = result_store.model_results_dir(
        results_root, meta["platform"]["id"], meta["model"]["id"], meta.get("result_tag")
    )
    group = str(meta.get("attempt_group") or meta["job_name"])
    current_meta = meta
    for attempt in range(completed_round + 1, max_attempts + 1):
        pending = result_store.tasks_requiring_attempt(
            model_dir, requested_tasks, meta["profile_hash"], max_attempts
        )
        if not pending:
            print(f"Attempt {attempt}:         not needed; every task already passed")
            return 0
        job_name = attempt_job_name(group, attempt)
        config = copy.deepcopy(base_config)
        config["job_name"] = job_name
        config["n_attempts"] = 1
        config["datasets"][0]["task_names"] = pending
        current_meta = copy.deepcopy(current_meta)
        current_meta.update(
            {
                "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "job_name": job_name,
                "executed_tasks": pending,
                "attempt_group": group,
                "attempt_round": attempt,
                "previous_attempt_job": attempt_job_name(group, attempt - 1),
            }
        )
        job_dir = JOBS_DIR / job_name
        if job_dir.exists() and not dry_run:
            existing_meta = load_resume_meta(job_dir)
            expected_group = str(current_meta.get("attempt_group") or "")
            existing_group = str(existing_meta.get("attempt_group") or "")
            expected_profile = str(current_meta.get("profile_hash") or "")
            existing_profile = str(existing_meta.get("profile_hash") or "")
            existing_round = int(existing_meta.get("attempt_round") or 1)
            if (
                existing_group != expected_group
                or existing_round != attempt
                or existing_profile != expected_profile
            ):
                raise RunnerError(
                    f"Existing conditional-attempt job does not match the "
                    f"expected attempt state: {job_dir}"
                )
            print(f"Attempt {attempt}:         resuming existing job {job_name}")
            return_code, exported, current_meta = resume_harbor_job(
                job_dir=job_dir,
                results_root=results_root,
                runtime=runtime,
            )
        else:
            print(
                f"Attempt {attempt}:         retrying {len(pending)} "
                "non-passing task(s)"
            )
            return_code, exported = execute_harbor_job(
                config=config,
                meta=current_meta,
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


def retry_failed(args: argparse.Namespace) -> int:
    if args.max_attempts < 2 or args.concurrency < 1:
        raise RunnerError("--max-attempts must be at least 2 and --concurrency positive")
    model_dir = args.result_set.resolve()
    meta, pending, completed_round = retry_state(model_dir, args.max_attempts)
    if not pending:
        print(f"No failed tasks remain below the {args.max_attempts}-attempt limit.")
        return 0

    tasks_dir = args.tasks_dir.resolve()
    validate_tasks(tasks_dir, pending)
    endpoint = args.endpoint or str(meta.get("endpoint") or "")
    if not endpoint:
        raise RunnerError("Result metadata has no endpoint; pass --endpoint")
    model = str((meta.get("model") or {}).get("id") or "")
    if not model:
        raise RunnerError("Result metadata has no model ID")
    if not args.skip_endpoint_check:
        discover_model(endpoint, args.api_key, preferred=model)

    runtime, runtime_description = container_runtime()
    profile = meta.get("evaluation_profile") or {}
    agent_profile = profile.get("agent") or {}
    context_length = int(agent_profile.get("context_length") or 0)
    agent_timeout = int(profile.get("agent_timeout_seconds") or 0)
    if context_length < 1 or agent_timeout < 1:
        raise RunnerError("Stored evaluation profile lacks context length or agent timeout")

    runtime_identity = result_store.runtime_identity(meta)
    base_name = args.job_name or make_job_name(
        "retry",
        model,
        meta.get("model_tag"),
        engine=runtime_identity.get("engine"),
        backend=runtime_identity.get("backend"),
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
        tasks_dir=tasks_dir,
        tasks=pending,
        model=model,
        endpoint=endpoint,
        api_key=args.api_key,
        concurrency=args.concurrency,
        context_length=context_length,
        agent_timeout_seconds=agent_timeout,
        keep_containers=args.keep_containers,
    )
    print(f"Results:            {model_dir}")
    print(f"Failed tasks:       {len(pending)}")
    for task in pending:
        print(f"  {task}")
    print(f"Attempts:           up to {args.max_attempts}; stop after first pass")
    print(f"Concurrency:        {args.concurrency}")
    return continue_conditional_attempts(
        meta=retry_meta,
        base_config=config,
        completed_round=completed_round,
        results_root=model_dir.parent.parent,
        runtime=runtime,
        dry_run=args.dry_run,
    )


def load_resume_meta(job_dir: Path) -> dict[str, Any]:
    candidates = [job_dir / "runner-meta.json", runner_meta_path(job_dir.name)]
    for path in candidates:
        if path.is_file():
            return result_store.read_json(path)
    raise RunnerError(f"Runner metadata not found for {job_dir}")


def resume_harbor_job(
    *,
    job_dir: Path,
    results_root: Path,
    runtime: str,
) -> tuple[int, bool, dict[str, Any]]:
    config_path = job_dir / "config.json"
    if not config_path.is_file():
        raise RunnerError(f"Harbor job config not found: {config_path}")
    meta = load_resume_meta(job_dir)
    command = [*harbor_command(), "job", "resume", "--job-path", str(job_dir)]
    print(f"Resuming:           {' '.join(command)}", flush=True)
    try:
        completed = subprocess.run(
            command, cwd=ROOT, env=command_environment(runtime), check=False
        )
    except KeyboardInterrupt:
        print("\nResume interrupted; Harbor received the interrupt.", file=sys.stderr)
        return 130, False, meta

    exported = False
    if (job_dir / "result.json").is_file():
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
    return completed.returncode, exported, meta


def add_connection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--endpoint",
        default=os.getenv("TBENCH_ENDPOINT", DEFAULT_ENDPOINT),
        help=f"Host-visible OpenAI-compatible base URL (default: {DEFAULT_ENDPOINT})",
    )
    parser.add_argument("--model", help="Served model ID; auto-detected for one-model endpoints")
    parser.add_argument("--api-key", default=os.getenv("TBENCH_API_KEY", "local"))
    parser.add_argument("--skip-endpoint-check", action="store_true")
    parser.add_argument(
        "--context-length",
        type=int,
        help="Model context capacity; normally discovered from GET /models",
    )
    parser.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS_DIR)


def add_result_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--platform",
        required=True,
        help="Platform identifier recorded in results (required)",
    )
    parser.add_argument("--platform-name")
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
    parser.add_argument("--model-tag")
    parser.add_argument(
        "--rocm-version",
        help="Deprecated alias for --backend-version; valid only with --backend rocm",
    )
    parser.add_argument("--inference-profile")


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
    list_parser = sub.add_parser("list", help="Print task tiers")
    list_parser.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS_DIR)
    run = sub.add_parser("run", help="Run Terminal-Bench-Local")
    add_connection_args(run)
    add_result_args(run)
    run.add_argument(
        "--tier", choices=TIERS, default="full", help="Task tier to run (default: full)"
    )
    run.add_argument("--task", help="Run one explicit Terminal-Bench 2.1 task")
    run.add_argument(
        "--attempts",
        type=int,
        default=DEFAULT_ATTEMPTS,
        help="Maximum attempts per task; later attempts run only after failure (default: 2)",
    )
    run.add_argument("--concurrency", type=int, default=1)
    run.add_argument("--agent-timeout", type=int, default=DEFAULT_AGENT_TIMEOUT_SECONDS)
    run.add_argument("--keep-containers", action="store_true")
    run.add_argument("--job-name")
    run.add_argument("--rerun", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    summary = sub.add_parser("summary", help="Summarize a Harbor job")
    summary.add_argument("job", type=Path)
    resume = sub.add_parser("resume", help="Resume and export an interrupted Harbor job")
    resume.add_argument("job", type=Path)
    resume.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
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
    retry.add_argument("--endpoint", help="Operational endpoint override")
    retry.add_argument("--api-key", default=os.getenv("TBENCH_API_KEY", "local"))
    retry.add_argument("--skip-endpoint-check", action="store_true")
    retry.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS_DIR)
    retry.add_argument("--concurrency", type=int, default=1)
    retry.add_argument("--keep-containers", action="store_true")
    retry.add_argument("--job-name")
    retry.add_argument("--dry-run", action="store_true")
    results = sub.add_parser("results", help="Rebuild the stable results index")
    results.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args(argv)


def print_tiers(tasks_dir: Path) -> None:
    full = set(load_tier("full"))
    for name in TIERS:
        tasks = load_tier(name)
        validate_tasks(tasks_dir, tasks)
        print(f"{name}: {len(tasks)} task(s), nested={set(tasks) <= full}")
        for task in tasks:
            print(f"  {task}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "list":
            print_tiers(args.tasks_dir.resolve())
            return 0
        if args.command == "summary":
            summarize_job(args.job.resolve())
            return 0
        if args.command == "results":
            root = args.results_dir.resolve()
            index = result_store.rebuild_index(root)
            print(f"Results:   {root}")
            print(f"Platforms: {len(index['platforms'])}")
            print(f"Models:    {len(index['models'])}")
            print(f"Tasks:     {len(index['tasks'])}")
            return 0
        if args.command == "resume":
            job_dir = args.job.resolve()
            runtime, _ = container_runtime()
            results_root = args.results_dir.resolve()
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
        ) = check_doctor(args)
        if args.command == "doctor":
            print("Doctor checks passed.")
            return 0
        if run_runtime is None:
            raise RunnerError(f"Unsupported command: {args.command}")
        engine, engine_version, backend, backend_version = run_runtime
        if args.attempts < 1 or args.concurrency < 1 or args.agent_timeout < 1:
            raise RunnerError("--attempts, --concurrency and --agent-timeout must be positive")
        tier = "task" if args.task else args.tier
        requested = [args.task] if args.task else load_tier(args.tier)
        tasks_dir = args.tasks_dir.resolve()
        validate_tasks(tasks_dir, requested)
        profile = evaluation_profile(
            tasks_dir=tasks_dir,
            model=model,
            model_metadata=metadata,
            endpoint=args.endpoint,
            engine=engine,
            engine_version=engine_version,
            backend=backend,
            backend_version=backend_version,
            inference_profile=args.inference_profile,
            context_length=context_length,
            agent_timeout_seconds=args.agent_timeout,
        )
        profile_hash = result_store.evaluation_profile_hash(profile)
        results_root = args.results_dir.resolve()
        model_dir = result_store.model_results_dir(
            results_root,
            args.platform,
            model,
            result_tag(args.model_tag, engine, backend),
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
        job_name = args.job_name or make_job_name(
            tier,
            model,
            args.model_tag,
            engine=engine,
            backend=backend,
        )
        meta = make_run_meta(
            job_name=job_name,
            tier=tier,
            requested_tasks=requested,
            executed_tasks=tasks,
            platform=args.platform,
            platform_name=args.platform_name,
            model=model,
            model_metadata=metadata,
            engine=engine,
            engine_version=engine_version,
            backend=backend,
            backend_version=backend_version,
            model_tag=args.model_tag,
            inference_profile=args.inference_profile,
            endpoint=args.endpoint,
            runtime=runtime,
            runtime_description=runtime_description,
            profile=profile,
            max_attempts=args.attempts,
            attempt_group=job_name,
        )
        config = build_config(
            job_name=job_name,
            tasks_dir=tasks_dir,
            tasks=tasks,
            model=model,
            endpoint=args.endpoint,
            api_key=args.api_key,
            concurrency=args.concurrency,
            context_length=context_length,
            agent_timeout_seconds=args.agent_timeout,
            keep_containers=args.keep_containers,
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
        print(f"Attempts:           up to {args.attempts}; stop after first pass")
        print(f"Concurrency:        {args.concurrency}")
        print(f"Agent:              {AGENT_NAME} {AGENT_VERSION}")
        return_code, exported = execute_harbor_job(
            config=config,
            meta=meta,
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
