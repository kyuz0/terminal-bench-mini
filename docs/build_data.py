#!/usr/bin/env python3
"""Build the compact, static dataset consumed by the GitHub Pages UI."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import tomllib
from urllib.parse import quote


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from suite_manifest import SuiteManifest
import results as result_store


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
DEFAULT_DOCS_SUITE = "core19"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def github_url(repository: str, path: str) -> str:
    encoded = quote(path.replace("\\", "/"), safe="/")
    return f"https://github.com/{repository}/blob/main/{encoded}"


def compact_text(value: object, limit: int = 360) -> str:
    text = ANSI_RE.sub("", str(value or ""))
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def verifier_excerpt(path: Path | None) -> list[str]:
    """Extract concise pytest evidence without publishing the complete job tree."""
    if path is None or not path.is_file():
        return []

    text = ANSI_RE.sub("", path.read_text(errors="replace"))
    candidates: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"^E\s+", stripped):
            candidates.append(re.sub(r"^E\s+", "", stripped))
        elif stripped.startswith("FAILED "):
            candidates.append(stripped)

    evidence: list[str] = []
    for candidate in candidates:
        candidate = compact_text(candidate)
        if candidate and candidate not in evidence:
            evidence.append(candidate)
        if len(evidence) == 3:
            break
    return evidence


def resolve_job_evidence(repo_root: Path, harbor_path: object) -> Path | None:
    if not harbor_path:
        return None
    candidate = Path(str(harbor_path))
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    try:
        candidate.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return None
    return candidate


def classify_attempt(attempt: dict, evidence: list[str]) -> tuple[str, str]:
    if attempt.get("passed") or attempt.get("reward") == 1:
        return "passed", "Verifier awarded the required reward of 1."

    exception = attempt.get("exception") or {}
    exception_type = exception.get("exception_type") or exception.get("type")
    message = exception.get("exception_message") or exception.get("message")
    tokens = attempt.get("tokens") or {}
    produced_tokens = (tokens.get("input") or 0) + (tokens.get("output") or 0)
    steps = attempt.get("agent_steps") or 0

    if exception_type == "AgentTimeoutError":
        if produced_tokens == 0 and steps == 0:
            return (
                "endpoint-stall",
                "The configured agent timeout expired before the endpoint returned model tokens.",
            )
        return "agent-timeout", compact_text(message) or "The agent exceeded its time limit."

    if exception_type:
        return "harness-error", compact_text(f"{exception_type}: {message}")

    if evidence:
        return "verifier-failure", evidence[0]
    return "verifier-failure", "The task completed, but the verifier awarded reward 0."


def estimate_minutes(metadata: dict, audience: str) -> float | int | None:
    """Normalize Terminal-Bench 2.x minute and 4.x hour estimates."""
    minutes = metadata.get(f"{audience}_time_estimate_min")
    if isinstance(minutes, (int, float)):
        return minutes
    hours = metadata.get(f"{audience}_time_estimate_hours")
    if isinstance(hours, (int, float)):
        return hours * 60
    return None


def task_metadata(
    repo_root: Path,
    task_id: str,
    repository: str,
    task_dir: Path | None = None,
) -> dict:
    task_dir = task_dir or repo_root / "tasks" / task_id
    task_toml = task_dir / "task.toml"
    payload = tomllib.loads(task_toml.read_text())
    task = payload.get("task", {})
    metadata = payload.get("metadata", {})
    instruction_path = task_dir / "instruction.md"
    return {
        "id": task_id,
        "name": task.get("name", f"terminal-bench/{task_id}"),
        "description": task.get("description", ""),
        "keywords": task.get("keywords", []),
        "category": metadata.get("category", "uncategorized"),
        "difficulty": metadata.get("difficulty", "unknown"),
        "tags": metadata.get("tags", []),
        "expertMinutes": estimate_minutes(metadata, "expert"),
        "juniorMinutes": estimate_minutes(metadata, "junior"),
        "instruction": instruction_path.read_text().strip(),
        "sourceUrl": github_url(repository, task_dir.relative_to(repo_root).as_posix()),
        "instructionUrl": github_url(
            repository, instruction_path.relative_to(repo_root).as_posix()
        ),
    }


def build_attempt(
    repo_root: Path,
    result_dir: Path,
    attempt: dict,
    repository: str,
) -> dict:
    harbor_paths = attempt.get("harbor_paths") or {}
    evidence_path = resolve_job_evidence(repo_root, harbor_paths.get("verifier_output"))
    evidence = verifier_excerpt(evidence_path)
    outcome_type, reason = classify_attempt(attempt, evidence)
    transcript = attempt.get("transcript")
    transcript_path = result_dir / transcript if transcript else None

    return {
        "number": attempt.get("attempt"),
        "trialName": attempt.get("trial_name"),
        "passed": bool(attempt.get("passed")),
        "reward": attempt.get("reward"),
        "rewards": attempt.get("rewards") or {},
        "durationMs": attempt.get("duration_ms") or 0,
        "tokens": attempt.get("tokens") or {},
        "agentSteps": attempt.get("agent_steps") or 0,
        "outcomeType": outcome_type,
        "reason": reason,
        "evidence": evidence,
        "exception": {
            "type": (attempt.get("exception") or {}).get("exception_type"),
            "message": compact_text(
                (attempt.get("exception") or {}).get("exception_message")
                or (attempt.get("exception") or {}).get("message")
            ),
        }
        if attempt.get("exception")
        else None,
        "transcriptUrl": github_url(
            repository, transcript_path.relative_to(repo_root).as_posix()
        )
        if transcript_path and transcript_path.is_file()
        else None,
    }


def result_record(
    repo_root: Path,
    result_dir: Path,
    result_path: Path,
    repository: str,
) -> dict:
    result = read_json(result_path)
    attempts = [
        build_attempt(repo_root, result_dir, attempt, repository)
        for attempt in result.get("attempts", [])
    ]
    passed = bool(result.get("passed"))
    if passed:
        succeeded = result.get("succeeded_at_attempt") or next(
            (attempt["number"] for attempt in attempts if attempt["passed"]), None
        )
        reason = (
            f"Passed on attempt {succeeded}; the verifier awarded reward 1."
            if succeeded
            else "The verifier awarded the required reward of 1."
        )
        outcome_type = "passed"
    elif attempts:
        labels = [attempt["reason"] for attempt in attempts]
        reason = " ".join(
            f"Attempt {index + 1}: {label}" for index, label in enumerate(labels)
        )
        outcome_type = attempts[-1]["outcomeType"]
    else:
        reason = "No attempt data was exported."
        outcome_type = "missing"

    return {
        "taskId": result.get("task"),
        "passed": passed,
        "passAt1": bool(attempts and attempts[0]["passed"]),
        "reward": result.get("reward"),
        "rewards": result.get("rewards") or {},
        "completed": bool(result.get("completed")),
        "succeededAtAttempt": result.get("succeeded_at_attempt"),
        "durationMs": result.get("duration_ms") or 0,
        "tokens": result.get("tokens") or {},
        "agentSteps": result.get("agent_steps") or 0,
        "outcomeType": outcome_type,
        "reason": reason,
        "attempts": attempts,
        "resultUrl": github_url(
            repository, result_path.relative_to(repo_root).as_posix()
        ),
    }


def model_record(
    repo_root: Path,
    result_dir: Path,
    repository: str,
    selected_tasks: set[str] | None = None,
) -> dict:
    summary = read_json(result_dir / "summary.json")
    run_meta_path = result_dir / "run-meta.json"
    run_meta = read_json(run_meta_path) if run_meta_path.is_file() else {}
    profile = run_meta.get("evaluation_profile") or {}
    model = run_meta.get("model") or summary.get("model") or {}
    endpoint_metadata = model.get("endpoint_metadata") or {}
    endpoint_meta = endpoint_metadata.get("meta") or {}

    results = []
    for filename in summary.get("results", []):
        path = result_dir / filename
        if path.is_file():
            record = result_record(repo_root, result_dir, path, repository)
            if selected_tasks is None or record["taskId"] in selected_tasks:
                results.append(record)

    pass_at_1 = sum(1 for result in results if result["passAt1"])
    passed_within_attempts = sum(1 for result in results if result["passed"])
    total = len(results)
    relative_dir = result_dir.relative_to(repo_root).as_posix()
    platform = run_meta.get("platform") or summary.get("platform") or {}
    if run_meta.get("engine") or profile.get("engine"):
        engine = run_meta.get("engine") or profile.get("engine")
        engine_version = run_meta.get("engine_version") or profile.get("engine_version")
        backend = run_meta.get("backend") or profile.get("backend")
        backend_version = run_meta.get("backend_version") or profile.get(
            "backend_version"
        )
    else:
        # Schema v1 called the inference engine `backend` and could only
        # describe a ROCm compute backend.
        engine = run_meta.get("backend") or summary.get("backend") or profile.get("backend")
        engine_version = None
        backend_version = run_meta.get("rocm_version") or profile.get("rocm_version")
        backend = "rocm" if backend_version else None
    agent = profile.get("agent") or {}
    context_length = (
        agent.get("context_length")
        or endpoint_metadata.get("context_length")
        or endpoint_meta.get("n_ctx")
    )
    quant = (
        run_meta.get("quant")
        or summary.get("quant")
        or run_meta.get("model_tag")
        or summary.get("model_tag")
    )
    inference_profile = (
        run_meta.get("inference_profile")
        or summary.get("inference_profile")
        or profile.get("inference_profile")
    )
    return {
        "id": relative_dir,
        "name": model.get("name") or summary.get("model", {}).get("name"),
        "modelId": model.get("id") or profile.get("model_id"),
        "quant": quant,
        "inferenceProfile": inference_profile,
        "tag": run_meta.get("tag", summary.get("tag")),
        "resultTag": run_meta.get("result_tag", summary.get("result_tag")),
        "platform": platform,
        "engine": engine,
        "engineVersion": engine_version,
        "backend": backend,
        "backendVersion": backend_version,
        "endpointOwner": endpoint_metadata.get("owned_by"),
        "contextLength": context_length,
        "modelSizeBytes": endpoint_meta.get("size"),
        "parameterCount": endpoint_meta.get("n_params"),
        "quantizationType": endpoint_meta.get("ftype"),
        "agent": agent,
        "harborVersion": profile.get("harbor_version"),
        "terminalBenchVersion": profile.get("terminal_bench_version"),
        "terminalBenchRevision": profile.get("terminal_bench_revision"),
        "profileHash": run_meta.get("profile_hash"),
        "exportedAt": run_meta.get("exported_at") or summary.get("generated_at"),
        "totalTasks": total,
        "passAt1": pass_at_1,
        "passAt1Rate": pass_at_1 / total if total else 0,
        "passedWithinAttempts": passed_within_attempts,
        "passWithinAttemptsRate": passed_within_attempts / total if total else 0,
        "totalDurationMs": sum(result["durationMs"] for result in results),
        "totalTokens": {
            key: sum((result.get("tokens") or {}).get(key, 0) or 0 for result in results)
            for key in ("input", "cached", "output")
        },
        "results": results,
        "summaryUrl": github_url(repository, f"{relative_dir}/summary.json"),
        "runMetaUrl": github_url(repository, f"{relative_dir}/run-meta.json"),
    }


def docs_suite_path(repo_root: Path, value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    return (repo_root / "suites" / f"{candidate}.json").resolve()


def build_dataset(
    repo_root: Path,
    repository: str,
    suite_value: str | Path | None = DEFAULT_DOCS_SUITE,
    tier: str = "full",
) -> dict:
    suite = (
        SuiteManifest.load(docs_suite_path(repo_root, suite_value))
        if suite_value is not None
        else None
    )
    if suite:
        task_ids = suite.tasks_for(tier)
        tasks = [
            task_metadata(
                repo_root,
                task_id,
                repository,
                task_dir=suite.resolve_task(task_id).path,
            )
            for task_id in task_ids
        ]
        identity = suite.identity()
        result_root = result_store.suite_results_root(
            repo_root / "results", identity
        )
        benchmark_name = suite.name
    else:
        task_ids = [
            line.strip()
            for line in (repo_root / "subsets" / "full.txt").read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        tasks = [task_metadata(repo_root, task_id, repository) for task_id in task_ids]
        identity = None
        result_root = repo_root / "results"
        benchmark_name = "Terminal Bench Mini"
    result_dirs = sorted(result_root.glob("*/*_results"))
    models = []
    selected_task_ids = set(task_ids)
    for result_dir in result_dirs:
        if not (result_dir / "summary.json").is_file():
            continue
        model = model_record(
            repo_root, result_dir, repository, selected_tasks=selected_task_ids
        )
        if model["totalTasks"]:
            models.append(model)
    models.sort(
        key=lambda model: (
            -model["passWithinAttemptsRate"],
            -model["passAt1Rate"],
            str(model["name"]).lower(),
        )
    )
    exported_times = [model.get("exportedAt") for model in models if model.get("exportedAt")]
    generated_at = max(exported_times) if exported_times else datetime.now(timezone.utc).isoformat()
    return {
        "schemaVersion": 2,
        "generatedAt": generated_at,
        "repository": repository,
        "benchmark": {
            "name": benchmark_name,
            "taskCount": len(tasks),
            "defaultMetric": "pass-within-attempts",
            "official": False,
            "suite": identity,
            "tier": tier if suite else "full",
        },
        "tasks": tasks,
        "models": models,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repository", default="kyuz0/terminal-bench-mini")
    parser.add_argument(
        "--suite",
        default=DEFAULT_DOCS_SUITE,
        help=f"Built-in suite ID or manifest path (default: {DEFAULT_DOCS_SUITE})",
    )
    parser.add_argument("--tier", default="full", help="Suite tier (default: full)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output = args.output or repo_root / "docs" / "data.json"
    payload = build_dataset(repo_root, args.repository, args.suite, args.tier)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(
        f"Wrote {output} with {len(payload['models'])} model runs "
        f"and {len(payload['tasks'])} tasks."
    )


if __name__ == "__main__":
    main()
