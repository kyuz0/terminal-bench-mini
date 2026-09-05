"""Stable, report-friendly result exports layered on top of Harbor jobs."""

from __future__ import annotations

from collections import defaultdict
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any


SCHEMA_VERSION = 4


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def model_name(model_id: str) -> str:
    """Return the human-facing model basename without a GGUF file suffix."""
    basename = model_id.rstrip("/").rsplit("/", 1)[-1]
    return re.sub(r"\.gguf$", "", basename, flags=re.IGNORECASE) or "local-model"


def model_path_component(model_id: str, limit: int = 180) -> str:
    """Make a safe path component without hiding case or a quantization suffix."""
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", model_name(model_id)).strip("-.")
    if not slug:
        slug = "local-model"
    if len(slug) <= limit:
        return slug

    # Quantization is normally at the end of a model name, so retain both ends
    # rather than blindly truncating the suffix.
    suffix_length = min(72, limit // 2)
    prefix_length = limit - suffix_length - 2
    return f"{slug[:prefix_length].rstrip('-.')}--{slug[-suffix_length:].lstrip('-.')}"


def stable_slug(value: str, limit: int = 180) -> str:
    slug = model_path_component(value, limit=limit)
    digest = hashlib.sha256(value.encode()).hexdigest()[:8]
    return f"{slug}-{digest}"


def path_slug(value: str, limit: int = 72) -> str:
    basename = value.rstrip("/").rsplit("/", 1)[-1]
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", basename).strip("-.").lower()
    slug = slug or "local"
    if len(slug) <= limit:
        return slug
    digest = hashlib.sha256(value.encode()).hexdigest()[:8]
    return f"{slug[: limit - 9].rstrip('-.')}-{digest}"


def identity_tag_component(value: str, limit: int = 48) -> str:
    """Return a readable tag without allowing truncation collisions."""
    tag = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-.") or "tag"
    lossy = tag != value.strip("-.")
    if len(tag) <= limit and not lossy:
        return tag
    digest = hashlib.sha256(value.encode()).hexdigest()[:8]
    if len(tag) + len(digest) + 1 <= limit:
        return f"{tag}-{digest}"
    available = limit - len(digest) - 3
    prefix_length = (available + 1) // 2
    suffix_length = available - prefix_length
    prefix = tag[:prefix_length].rstrip("-.")
    suffix = tag[-suffix_length:].lstrip("-.")
    return f"{prefix}--{suffix}-{digest}"


def json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def evaluation_identity(profile: dict[str, Any]) -> dict[str, Any]:
    """Return benchmark identity without operational retry policy."""
    identity = {
        key: value
        for key, value in profile.items()
        if key not in {"attempts", "attempt_policy"}
    }
    # Schema v1 used `backend` for the inference engine and had a ROCm-only
    # version field. Canonicalize it so equivalent historical ROCm runs still
    # compare correctly with schema v2 engine/backend identities.
    if "engine" not in identity and (
        "backend" in identity or "rocm_version" in identity
    ):
        legacy_engine = identity.get("backend")
        legacy_rocm_version = identity.pop("rocm_version", None)
        identity["engine"] = legacy_engine
        identity["engine_version"] = None
        identity["backend"] = "rocm" if legacy_rocm_version else None
        identity["backend_version"] = legacy_rocm_version
    elif "engine" in identity:
        identity.pop("rocm_version", None)
        identity.setdefault("engine_version", None)
        identity.setdefault("backend_version", None)
    return identity


def evaluation_profile_hash(profile: dict[str, Any]) -> str:
    return json_hash(evaluation_identity(profile))


def matching_evaluation_profiles(
    left: dict[str, Any] | None, right: dict[str, Any] | None
) -> bool:
    return evaluation_profile_hash(left or {}) == evaluation_profile_hash(right or {})


def runtime_identity(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return canonical engine and compute-backend fields for any schema version."""
    profile = metadata.get("evaluation_profile") or {}
    if metadata.get("engine") or profile.get("engine"):
        return {
            "engine": metadata.get("engine") or profile.get("engine"),
            "engine_version": metadata.get("engine_version")
            or profile.get("engine_version"),
            "backend": metadata.get("backend") or profile.get("backend"),
            "backend_version": metadata.get("backend_version")
            or profile.get("backend_version"),
        }
    legacy_engine = metadata.get("backend") or profile.get("backend")
    legacy_rocm_version = metadata.get("rocm_version") or profile.get("rocm_version")
    return {
        "engine": legacy_engine,
        "engine_version": None,
        "backend": "rocm" if legacy_rocm_version else None,
        "backend_version": legacy_rocm_version,
    }


def metadata_quant(metadata: dict[str, Any]) -> str | None:
    """Return the dedicated quant field, with a schema-v2 fallback."""
    value = metadata.get("quant")
    if value is None:
        value = metadata.get("model_tag")
    return str(value) if value else None


def metadata_inference_profile(metadata: dict[str, Any]) -> str | None:
    value = metadata.get("inference_profile")
    if value is None:
        value = (metadata.get("evaluation_profile") or {}).get("inference_profile")
    return str(value) if value else None


def metadata_suite(metadata: dict[str, Any]) -> dict[str, Any] | None:
    """Return the immutable suite identity recorded for a manifest-backed run."""
    direct = metadata.get("suite")
    profiled = (metadata.get("evaluation_profile") or {}).get("suite")
    values = [value for value in (direct, profiled) if value is not None]
    if not values:
        return None
    normalized: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("Suite identity must be a JSON object")
        suite_id = value.get("id")
        manifest_hash = value.get("manifest_hash")
        if not isinstance(suite_id, str) or not suite_id:
            raise ValueError("Suite identity has no usable id")
        if not isinstance(manifest_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", manifest_hash
        ):
            raise ValueError("Suite identity has no valid manifest_hash")
        normalized.append(dict(value))
    if any(value != normalized[0] for value in normalized[1:]):
        raise ValueError("Run metadata and evaluation profile disagree on suite identity")
    return normalized[0]


def reconcile_platform_metadata(
    existing: dict[str, Any], incoming: dict[str, Any]
) -> dict[str, Any]:
    """Keep a descriptive platform name when the stable platform ID matches."""
    existing_identity = {key: value for key, value in existing.items() if key != "name"}
    incoming_identity = {key: value for key, value in incoming.items() if key != "name"}
    if existing_identity != incoming_identity:
        raise ValueError("Platform IDs or identity metadata differ")
    platform_id = str(existing.get("id") or "")
    existing_name = str(existing.get("name") or platform_id)
    incoming_name = str(incoming.get("name") or platform_id)
    if existing_name == incoming_name:
        return dict(existing)
    if incoming_name == platform_id:
        return dict(existing)
    if existing_name == platform_id:
        return {**existing, "name": incoming_name}
    raise ValueError(
        f"Conflicting descriptive names for platform {platform_id!r}: "
        f"{existing_name!r} and {incoming_name!r}"
    )


def suite_results_root(
    results_root: Path, suite: dict[str, Any] | None
) -> Path:
    """Return an isolated root for a suite, preserving legacy unscoped paths."""
    if not suite:
        return results_root
    suite_id = str(suite["id"])
    manifest_hash = str(suite["manifest_hash"])
    namespace = f"{path_slug(suite_id)}-{manifest_hash}"
    return results_root / "suites" / namespace


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def model_results_dir(
    results_root: Path,
    platform: str,
    model_id: str,
    identity_tag: str | None,
    *,
    suite: dict[str, Any] | None = None,
    profile_hash: str | None = None,
) -> Path:
    if suite and not (
        isinstance(profile_hash, str) and re.fullmatch(r"[0-9a-f]{64}", profile_hash)
    ):
        raise ValueError("Suite-backed result path requires a valid profile hash")
    directory = stable_slug(model_id, limit=110 if suite else 180)
    if identity_tag:
        directory += f"-{identity_tag_component(identity_tag)}"
    if suite:
        directory += f"-{profile_hash}"
    scope = suite_results_root(results_root, suite)
    return scope / path_slug(platform, limit=48) / f"{directory}_results"


def model_results_dir_for_run(results_root: Path, metadata: dict[str, Any]) -> Path:
    """Resolve a model result directory from complete run metadata."""
    identity_tag = metadata.get("result_tag") or metadata_quant(metadata)
    model = metadata.get("model") or {}
    platform = metadata.get("platform") or {}
    return model_results_dir(
        results_root,
        str(platform["id"]),
        str(model["id"]),
        str(identity_tag) if identity_tag else None,
        suite=metadata_suite(metadata),
        profile_hash=str(metadata.get("profile_hash") or ""),
    )


def results_root_from_model_dir(model_dir: Path, metadata: dict[str, Any]) -> Path:
    """Recover the caller-facing results root from a normalized model directory."""
    model_dir = model_dir.resolve()
    scope_root = model_dir.parent.parent
    suite = metadata_suite(metadata)
    if not suite:
        return scope_root
    if scope_root.parent.name != "suites":
        raise ValueError(f"Suite result directory is outside a suites namespace: {model_dir}")
    results_root = scope_root.parent.parent
    expected = suite_results_root(results_root, suite)
    if expected != scope_root:
        raise ValueError(
            f"Suite result directory does not match recorded suite identity: {model_dir}"
        )
    return results_root


def task_id_from_trial(result: dict[str, Any]) -> str:
    task_name = result.get("task_name")
    if isinstance(task_name, str) and task_name.rstrip("/"):
        candidate = task_name.rstrip("/").rsplit("/", 1)[-1]
        # Hosted Harbor materializes registry tasks below a directory literally
        # named `task`; its namespaced task_name retains the actual task slug.
        return candidate
    task_id = result.get("task_id")
    if isinstance(task_id, dict):
        for key in ("name", "ref"):
            value = task_id.get(key)
            if not isinstance(value, str) or not value.rstrip("/"):
                continue
            candidate = value.rstrip("/").rsplit("/", 1)[-1]
            if candidate:
                return candidate
    trial_name = result.get("trial_name")
    if isinstance(trial_name, str) and trial_name:
        return trial_name.split("__", 1)[0]
    if isinstance(task_id, dict):
        value = task_id.get("path")
        if isinstance(value, str) and value.rstrip("/"):
            path = Path(value.rstrip("/"))
            candidate = path.parent.name if path.name == "task" else path.name
            if candidate:
                return candidate
    raise ValueError("Trial result has no usable task ID")


def trial_results(job_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    rows: list[tuple[Path, dict[str, Any]]] = []
    for directory in sorted(job_dir.iterdir()):
        path = directory / "result.json"
        if directory.is_dir() and path.is_file():
            rows.append((directory, read_json(path)))
    return rows


def relative_to_repo(path: Path, repo_root: Path) -> str:
    return Path(os.path.relpath(path.resolve(), repo_root.resolve())).as_posix()


def timing_duration_ms(result: dict[str, Any]) -> int | None:
    started = result.get("started_at")
    finished = result.get("finished_at")
    if not isinstance(started, str) or not isinstance(finished, str):
        return None
    try:
        start = dt.datetime.fromisoformat(started.replace("Z", "+00:00"))
        end = dt.datetime.fromisoformat(finished.replace("Z", "+00:00"))
    except ValueError:
        return None
    return max(round((end - start).total_seconds() * 1000), 0)


def normalized_attempt(
    trial_dir: Path,
    result: dict[str, Any],
    *,
    attempt: int,
    model_dir: Path,
    repo_root: Path,
    transcript_name: str,
    endpoint: str | None = None,
    paths_key: str = "pier_paths",
) -> dict[str, Any]:
    agent_result = result.get("agent_result") or {}
    verifier_result = result.get("verifier_result") or {}
    rewards = verifier_result.get("rewards") or {}
    reward = rewards.get("reward")
    exception = result.get("exception_info")

    transcript_source = trial_dir / "agent" / "trajectory.json"
    transcript_target = model_dir / transcript_name
    if transcript_source.is_file():
        shutil.copy2(transcript_source, transcript_target)

    transcript_steps: list[dict[str, Any]] = []
    if transcript_source.is_file():
        try:
            transcript = read_json(transcript_source)
            transcript_steps = [
                step for step in transcript.get("steps", []) if isinstance(step, dict)
            ]
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    agent_steps = result.get("n_agent_steps") or agent_result.get("n_agent_steps")
    if agent_steps is None and transcript_steps:
        agent_steps = sum(step.get("source") == "agent" for step in transcript_steps)
    peak_context = agent_result.get("peak_context_tokens")
    if peak_context is None and transcript_steps:
        prompt_tokens = [
            (step.get("metrics") or {}).get("prompt_tokens")
            for step in transcript_steps
            if isinstance(step.get("metrics"), dict)
        ]
        prompt_tokens = [value for value in prompt_tokens if isinstance(value, int)]
        peak_context = max(prompt_tokens, default=None)

    native_trajectory = trial_dir / "agent" / "trajectory.json"
    patch = trial_dir / "artifacts" / "model.patch"
    verifier_ctrf = trial_dir / "verifier" / "ctrf.json"
    verifier_output = trial_dir / "verifier" / "test-stdout.txt"

    def optional_path(path: Path) -> str | None:
        return relative_to_repo(path, repo_root) if path.is_file() else None

    return {
        "attempt": attempt,
        "trial_name": result.get("trial_name") or trial_dir.name,
        "passed": reward == 1,
        "reward": reward,
        "rewards": rewards,
        "duration_ms": timing_duration_ms(result),
        "tokens": {
            "input": agent_result.get("n_input_tokens"),
            "cached": agent_result.get("n_cache_tokens"),
            "output": agent_result.get("n_output_tokens"),
            "peak_context": peak_context,
        },
        "agent_steps": agent_steps,
        "exception": exception,
        "endpoint": endpoint,
        "transcript": transcript_name if transcript_source.is_file() else None,
        paths_key: {
            "trial": relative_to_repo(trial_dir, repo_root),
            "native_trajectory": optional_path(native_trajectory),
            "model_patch": optional_path(patch),
            "verifier_ctrf": optional_path(verifier_ctrf),
            "verifier_output": optional_path(verifier_output),
        },
    }


def summarize_model_dir(model_dir: Path, run_meta: dict[str, Any]) -> dict[str, Any]:
    results = []
    for path in sorted(model_dir.glob("results-*.json")):
        try:
            results.append(read_json(path))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    attempted = results
    passed = sum(bool(row.get("passed")) for row in attempted)

    def total(path: tuple[str, ...]) -> int:
        value = 0
        for row in attempted:
            current: Any = row
            for key in path:
                current = current.get(key) if isinstance(current, dict) else None
            if isinstance(current, (int, float)):
                value += int(current)
        return value

    runtime = runtime_identity(run_meta)
    suite = metadata_suite(run_meta)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "platform": run_meta["platform"],
        "model": run_meta["model"],
        **runtime,
        "quant": metadata_quant(run_meta),
        "inference_profile": metadata_inference_profile(run_meta),
        "tag": run_meta.get("tag"),
        "result_tag": run_meta.get("result_tag"),
        "suite": suite,
        "profile_hash": run_meta.get("profile_hash"),
        "total_tasks": len(attempted),
        "passed_tasks": passed,
        "pass_rate": passed / len(attempted) if attempted else 0,
        "total_duration_ms": total(("duration_ms",)),
        "tokens": {
            "input": total(("tokens", "input")),
            "cached": total(("tokens", "cached")),
            "output": total(("tokens", "output")),
        },
        "results": [path.name for path in sorted(model_dir.glob("results-*.json"))],
    }
    if run_meta.get("historical_projection") is not None:
        summary["historical_projection"] = copy.deepcopy(
            run_meta["historical_projection"]
        )
    write_json(model_dir / "summary.json", summary)
    return summary


def rebuild_index(
    results_root: Path, *, suite: dict[str, Any] | None = None
) -> dict[str, Any]:
    suite_path = results_root / "suite.json"
    if suite_path.is_file():
        recorded_suite = metadata_suite({"suite": read_json(suite_path)})
        if suite is not None and suite != recorded_suite:
            raise ValueError(f"Requested suite does not match {suite_path}")
        suite = recorded_suite
    platforms: list[dict[str, Any]] = []
    models: list[dict[str, Any]] = []
    tasks: dict[str, dict[str, Any]] = defaultdict(dict)
    if results_root.is_dir():
        for platform_dir in sorted(path for path in results_root.iterdir() if path.is_dir()):
            platform_path = platform_dir / "platform.json"
            if not platform_path.is_file():
                continue
            platform = read_json(platform_path)
            platforms.append(platform)
            for model_dir in sorted(platform_dir.glob("*_results")):
                meta_path = model_dir / "run-meta.json"
                if not meta_path.is_file():
                    continue
                meta = read_json(meta_path)
                run_suite = metadata_suite(meta)
                if run_suite != suite:
                    raise ValueError(
                        f"Result metadata suite does not match its index scope: {meta_path}"
                    )
                summary = summarize_model_dir(model_dir, meta)
                model_key = f"{platform['id']}::{meta['model']['id']}"
                identity_tag = meta.get("result_tag") or metadata_quant(meta)
                if identity_tag:
                    model_key += f"::{identity_tag}"
                if suite:
                    model_key += f"::{meta.get('profile_hash')}"
                models.append(
                    {
                        "id": model_key,
                        "result_directory": relative_to_repo(model_dir, results_root),
                        **{key: summary.get(key) for key in (
                            "platform", "model", "engine", "engine_version", "backend",
                            "backend_version", "quant", "inference_profile", "tag",
                            "result_tag", "suite", "profile_hash", "total_tasks",
                            "passed_tasks", "pass_rate", "total_duration_ms", "tokens",
                        )},
                    }
                )
                for result_path in sorted(model_dir.glob("results-*.json")):
                    result = read_json(result_path)
                    task_id = result.get("task")
                    if isinstance(task_id, str):
                        tasks[task_id][model_key] = {
                            "passed": result.get("passed"),
                            "reward": result.get("reward"),
                            "duration_ms": result.get("duration_ms"),
                            "provenance": result.get("task_provenance"),
                            "result": relative_to_repo(result_path, results_root),
                            "transcript": result.get("transcript"),
                        }
    models.sort(key=lambda row: (-float(row.get("pass_rate") or 0), row["id"]))
    index = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "suite": suite,
        "platforms": platforms,
        "models": models,
        "tasks": [{"id": task, "results": values} for task, values in sorted(tasks.items())],
    }
    write_json(results_root / "index.json", index)
    return index


def rebuild_suite_catalog(results_root: Path) -> dict[str, Any]:
    """Index isolated suite namespaces without combining their aggregates."""
    entries: list[dict[str, Any]] = []
    suites_root = results_root / "suites"
    if suites_root.is_dir():
        for namespace in sorted(path for path in suites_root.iterdir() if path.is_dir()):
            suite_path = namespace / "suite.json"
            if not suite_path.is_file():
                continue
            suite = metadata_suite({"suite": read_json(suite_path)})
            if suite_results_root(results_root, suite) != namespace:
                raise ValueError(
                    f"Suite namespace does not match suite.json identity: {namespace}"
                )
            index_path = namespace / "index.json"
            entries.append(
                {
                    **suite,
                    "result_directory": relative_to_repo(namespace, results_root),
                    "index": relative_to_repo(index_path, results_root)
                    if index_path.is_file()
                    else None,
                }
            )
    catalog = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "suites": entries,
    }
    if entries or (suites_root / "index.json").is_file():
        write_json(suites_root / "index.json", catalog)
    return catalog


def rebuild_all_indexes(results_root: Path) -> dict[str, Any]:
    """Rebuild legacy and suite-scoped indexes without aggregating them together."""
    legacy = rebuild_index(results_root)
    suites_root = results_root / "suites"
    if suites_root.is_dir():
        for namespace in sorted(path for path in suites_root.iterdir() if path.is_dir()):
            suite_path = namespace / "suite.json"
            if not suite_path.is_file():
                continue
            suite = metadata_suite({"suite": read_json(suite_path)})
            if suite_results_root(results_root, suite) != namespace:
                raise ValueError(
                    f"Suite namespace does not match suite.json identity: {namespace}"
                )
            rebuild_index(namespace, suite=suite)
    catalog = rebuild_suite_catalog(results_root)
    return {"legacy": legacy, "suite_catalog": catalog}


def export_job(
    job_dir: Path,
    *,
    results_root: Path,
    repo_root: Path,
    run_meta: dict[str, Any],
    merge_existing_attempts: bool = False,
) -> tuple[Path, dict[str, Any]]:
    job_dir = job_dir.resolve()
    results_root = results_root.resolve()
    run_meta = {
        **run_meta,
        "model": dict(run_meta["model"]),
        "platform": dict(run_meta["platform"]),
    }
    run_meta.update(runtime_identity(run_meta))
    run_meta["quant"] = metadata_quant(run_meta)
    run_meta["inference_profile"] = metadata_inference_profile(run_meta)
    run_meta.setdefault("tag", None)
    run_meta.pop("model_tag", None)
    run_meta["model"].setdefault("name", model_name(run_meta["model"]["id"]))
    platform = run_meta["platform"]
    model = run_meta["model"]
    suite = metadata_suite(run_meta)
    scope_root = suite_results_root(results_root, suite)
    grouped: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for trial_dir, result in trial_results(job_dir):
        grouped[task_id_from_trial(result)].append((trial_dir, result))
    task_provenance_by_task = run_meta.get("task_provenance") or {}
    if suite:
        for task_id in grouped:
            provenance = task_provenance_by_task.get(task_id)
            if (
                not isinstance(provenance, dict)
                or provenance.get("id") != task_id
                or not re.fullmatch(
                    r"sha256:[0-9a-f]{64}",
                    str(provenance.get("content_sha256") or ""),
                )
            ):
                raise ValueError(
                    f"Suite-backed result has no valid provenance for task {task_id!r}"
                )
    model_dir = model_results_dir(
        results_root,
        platform["id"],
        model["id"],
        run_meta.get("result_tag") or run_meta.get("quant"),
        suite=suite,
        profile_hash=run_meta.get("profile_hash"),
    )
    suite_path = scope_root / "suite.json"
    if suite_path.is_file() and read_json(suite_path) != suite:
        raise ValueError(f"Suite namespace contains a different identity: {suite_path}")
    platform_path = scope_root / path_slug(platform["id"], limit=48) / "platform.json"
    if platform_path.is_file():
        try:
            platform = reconcile_platform_metadata(read_json(platform_path), platform)
        except ValueError as exc:
            raise ValueError(
                f"Platform path collision or metadata mismatch: {platform_path}: {exc}"
            ) from exc
        run_meta["platform"] = platform
    model_dir.mkdir(parents=True, exist_ok=True)
    if suite:
        write_json(suite_path, suite)
    write_json(platform_path, platform)

    harness = str(run_meta.get("harness") or "Harbor")
    harness_key = re.sub(r"[^a-z0-9]+", "_", harness.lower()).strip("_") or "harness"
    job_key = f"{harness_key}_job"
    jobs_key = f"{harness_key}_jobs"
    paths_key = f"{harness_key}_paths"
    exported_meta = {
        **run_meta,
        "schema_version": SCHEMA_VERSION,
        "exported_at": utc_now(),
        job_key: relative_to_repo(job_dir, repo_root),
    }
    write_json(model_dir / "run-meta.json", exported_meta)
    runs_dir = model_dir / "runs"
    write_json(runs_dir / f"{job_dir.name}.json", exported_meta)

    exported_tasks = []
    for task_id, attempts in sorted(grouped.items()):
        task_provenance = task_provenance_by_task.get(task_id)
        attempts.sort(key=lambda item: str(item[1].get("started_at") or item[0].name))
        result_path = model_dir / f"results-{task_id}.json"
        previous: dict[str, Any] | None = None
        if merge_existing_attempts and result_path.is_file():
            candidate = read_json(result_path)
            if matching_evaluation_profiles(
                candidate.get("evaluation_profile"), run_meta.get("evaluation_profile")
            ):
                previous = candidate
        normalized_attempts = list((previous or {}).get("attempts") or [])
        existing_trial_names = {
            row.get("trial_name") for row in normalized_attempts if row.get("trial_name")
        }
        next_attempt = len(normalized_attempts) + 1
        for trial_dir, result in attempts:
            trial_name = result.get("trial_name") or trial_dir.name
            if trial_name in existing_trial_names:
                continue
            index = next_attempt
            next_attempt += 1
            transcript_name = (
                f"transcript-{task_id}.json"
                if index == 1 and len(attempts) == 1
                else f"transcript-{task_id}-attempt{index}.json"
            )
            normalized_attempts.append(
                normalized_attempt(
                    trial_dir,
                    result,
                    attempt=index,
                    model_dir=model_dir,
                    repo_root=repo_root,
                    transcript_name=transcript_name,
                    endpoint=run_meta.get("endpoint"),
                    paths_key=paths_key,
                )
            )
            existing_trial_names.add(trial_name)
        successful = next((row for row in normalized_attempts if row["passed"]), None)
        selected = successful or normalized_attempts[-1]
        completed = successful is not None or all(
            row["exception"] is None for row in normalized_attempts
        )
        harness_jobs = list((previous or {}).get(jobs_key) or [])
        previous_job = (previous or {}).get(job_key)
        if previous_job and previous_job not in harness_jobs:
            harness_jobs.append(previous_job)
        current_job = relative_to_repo(job_dir, repo_root)
        if current_job not in harness_jobs:
            harness_jobs.append(current_job)
        normalized = {
            "schema_version": SCHEMA_VERSION,
            "task": task_id,
            "completed": completed,
            "passed": bool(successful),
            "reward": selected["reward"],
            "rewards": selected["rewards"],
            "duration_ms": sum(row["duration_ms"] or 0 for row in normalized_attempts),
            "tokens": {
                key: sum((row["tokens"].get(key) or 0) for row in normalized_attempts)
                for key in ("input", "cached", "output")
            },
            "agent_steps": sum((row.get("agent_steps") or 0) for row in normalized_attempts),
            "attempts": normalized_attempts,
            "succeeded_at_attempt": successful["attempt"] if successful else None,
            "transcript": selected["transcript"],
            "model": model,
            "platform": platform,
            "engine": run_meta.get("engine"),
            "engine_version": run_meta.get("engine_version"),
            "backend": run_meta.get("backend"),
            "backend_version": run_meta.get("backend_version"),
            "quant": run_meta.get("quant"),
            "inference_profile": run_meta.get("inference_profile"),
            "tag": run_meta.get("tag"),
            "result_tag": run_meta.get("result_tag"),
            "suite": suite,
            "task_provenance": task_provenance,
            "endpoints": run_meta.get("endpoints")
            or ([run_meta["endpoint"]] if run_meta.get("endpoint") else []),
            "evaluation_profile": run_meta["evaluation_profile"],
            "profile_hash": run_meta["profile_hash"],
            job_key: current_job,
            jobs_key: harness_jobs,
            "exported_at": utc_now(),
        }
        write_json(result_path, normalized)
        exported_tasks.append(task_id)

    run_summary = {
        "schema_version": SCHEMA_VERSION,
        "job_name": job_dir.name,
        "tasks": exported_tasks,
        "suite": suite,
        "profile_hash": run_meta["profile_hash"],
        "exported_at": utc_now(),
    }
    write_json(runs_dir / f"{job_dir.name}-summary.json", run_summary)
    summary = summarize_model_dir(model_dir, exported_meta)
    rebuild_index(scope_root, suite=suite)
    if suite:
        rebuild_suite_catalog(results_root)
    return model_dir, summary


def cached_tasks(model_dir: Path, tasks: list[str], profile_hash: str) -> list[str]:
    cached = []
    for task in tasks:
        path = model_dir / f"results-{task}.json"
        if not path.is_file():
            continue
        try:
            result = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        projection = result.get("historical_projection")
        if isinstance(projection, dict) and projection.get("cache_eligible") is False:
            continue
        if (
            result.get("completed")
            and evaluation_profile_hash(result.get("evaluation_profile") or {})
            == profile_hash
        ):
            cached.append(task)
    return cached


def tasks_requiring_attempt(
    model_dir: Path,
    tasks: list[str],
    profile_hash: str,
    max_attempts: int,
) -> list[str]:
    """Return tasks that have not passed and remain below the attempt limit."""
    pending = []
    for task in tasks:
        path = model_dir / f"results-{task}.json"
        if not path.is_file():
            pending.append(task)
            continue
        try:
            result = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            pending.append(task)
            continue
        if evaluation_profile_hash(result.get("evaluation_profile") or {}) != profile_hash:
            pending.append(task)
            continue
        if result.get("passed"):
            continue
        if len(result.get("attempts") or []) < max_attempts:
            pending.append(task)
    return pending
