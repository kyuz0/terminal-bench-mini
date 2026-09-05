"""Versioned suite manifests and deterministic task-content identities."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable


MANIFEST_SCHEMA_VERSION = 1
TREE_HASH_ALGORITHM = "terminal-bench-local-task-tree-v1"
_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class SuiteManifestError(ValueError):
    """Raised when a suite manifest is invalid or does not match its task tree."""


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise SuiteManifestError(f"Suite manifest contains duplicate key {key!r}")
        value[key] = item
    return value


def _reject_unknown_keys(
    mapping: dict[str, Any], allowed: set[str], context: str
) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise SuiteManifestError(
            f"{context} contains unsupported field(s): {', '.join(unknown)}"
        )


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def task_tree_digest(task_dir: Path) -> str:
    """Hash every file in a task directory using a stable, path-aware format.

    The digest covers the complete task package, including its environment,
    verifier, solution and documentation. Directory locations and incidental
    file metadata are excluded; the executable bit is retained because it can
    change task behavior.
    """

    task_dir = task_dir.resolve()
    if not task_dir.is_dir():
        raise SuiteManifestError(f"Task directory does not exist: {task_dir}")
    paths = sorted(
        path
        for path in task_dir.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    if not paths:
        raise SuiteManifestError(f"Task directory contains no files: {task_dir}")
    symlinks = sorted(path for path in task_dir.rglob("*") if path.is_symlink())
    if symlinks:
        relative = symlinks[0].relative_to(task_dir).as_posix()
        raise SuiteManifestError(
            f"Task directory contains unsupported symlink {relative!r}: {task_dir}"
        )

    digest = hashlib.sha256()
    digest.update(TREE_HASH_ALGORITHM.encode() + b"\0")
    for path in paths:
        relative = path.relative_to(task_dir).as_posix().encode()
        content = hashlib.sha256(path.read_bytes()).digest()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(b"x" if path.stat().st_mode & 0o111 else b"-")
        digest.update(content)
    return f"sha256:{digest.hexdigest()}"


def _required_string(mapping: dict[str, Any], key: str, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SuiteManifestError(f"{context}.{key} must be a non-empty string")
    return value.strip()


def _digest(mapping: dict[str, Any], key: str, context: str) -> str:
    value = _required_string(mapping, key, context)
    if not _DIGEST_PATTERN.fullmatch(value):
        raise SuiteManifestError(f"{context}.{key} must be a sha256:<64 hex> digest")
    return value


@dataclass(frozen=True)
class SuiteTask:
    id: str
    root: Path
    source_id: str
    upstream_name: str
    content_sha256: str
    upstream_digest: str | None
    provenance: dict[str, Any]

    @property
    def path(self) -> Path:
        return self.root / self.id

    def identity(self) -> dict[str, Any]:
        value = {
            "id": self.id,
            "source": self.source_id,
            "upstream_name": self.upstream_name,
            "content_sha256": self.content_sha256,
        }
        if self.upstream_digest:
            value["upstream_digest"] = self.upstream_digest
        return value


@dataclass(frozen=True)
class SuiteManifest:
    path: Path
    id: str
    version: str
    name: str
    sources: dict[str, dict[str, Any]]
    tiers: dict[str, tuple[str, ...]]
    tasks: dict[str, SuiteTask]
    manifest_hash: str

    @classmethod
    def load(cls, path: Path, *, verify_content: bool = True) -> "SuiteManifest":
        path = path.resolve()
        try:
            raw = json.loads(
                path.read_text(), object_pairs_hook=_object_without_duplicates
            )
        except OSError as exc:
            raise SuiteManifestError(f"Cannot read suite manifest {path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise SuiteManifestError(f"Invalid JSON in suite manifest {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise SuiteManifestError(f"Suite manifest must be a JSON object: {path}")
        _reject_unknown_keys(
            raw,
            {"schema_version", "id", "version", "name", "description", "sources", "tiers", "tasks"},
            "suite",
        )
        if type(raw.get("schema_version")) is not int or raw.get(
            "schema_version"
        ) != MANIFEST_SCHEMA_VERSION:
            raise SuiteManifestError(
                f"Unsupported suite manifest schema {raw.get('schema_version')!r}; "
                f"expected {MANIFEST_SCHEMA_VERSION}"
            )

        suite_id = _required_string(raw, "id", "suite")
        if not _ID_PATTERN.fullmatch(suite_id):
            raise SuiteManifestError(
                "suite.id must use lowercase letters, digits, dots, underscores or dashes"
            )
        version = _required_string(raw, "version", "suite")
        name = _required_string(raw, "name", "suite")

        raw_sources = raw.get("sources")
        if not isinstance(raw_sources, dict) or not raw_sources:
            raise SuiteManifestError("suite.sources must be a non-empty object")
        sources: dict[str, dict[str, Any]] = {}
        source_roots: dict[str, Path] = {}
        for source_id, source in raw_sources.items():
            context = f"sources.{source_id}"
            if not isinstance(source_id, str) or not _ID_PATTERN.fullmatch(source_id):
                raise SuiteManifestError(f"Invalid source ID: {source_id!r}")
            if not isinstance(source, dict):
                raise SuiteManifestError(f"{context} must be an object")
            _reject_unknown_keys(
                source,
                {"task_root", "dataset", "version", "repository", "revision"},
                context,
            )
            task_root = _required_string(source, "task_root", context)
            provenance = {
                "dataset": _required_string(source, "dataset", context),
                "version": _required_string(source, "version", context),
                "repository": _required_string(source, "repository", context),
                "revision": _required_string(source, "revision", context),
            }
            root = (path.parent / task_root).resolve()
            revision_path = root / "REVISION"
            if revision_path.is_file():
                actual_revision = revision_path.read_text().strip()
                if actual_revision != provenance["revision"]:
                    raise SuiteManifestError(
                        f"{context}.revision does not match {revision_path}: "
                        f"manifest {provenance['revision']!r}, actual {actual_revision!r}"
                    )
            sources[source_id] = provenance
            source_roots[source_id] = root

        raw_tasks = raw.get("tasks")
        if not isinstance(raw_tasks, dict) or not raw_tasks:
            raise SuiteManifestError("suite.tasks must be a non-empty object")
        tasks: dict[str, SuiteTask] = {}
        canonical_tasks: dict[str, dict[str, Any]] = {}
        for task_id, task in raw_tasks.items():
            context = f"tasks.{task_id}"
            if not isinstance(task_id, str) or not _ID_PATTERN.fullmatch(task_id):
                raise SuiteManifestError(f"Invalid task ID: {task_id!r}")
            if not isinstance(task, dict):
                raise SuiteManifestError(f"{context} must be an object")
            _reject_unknown_keys(
                task,
                {"source", "upstream_name", "upstream_digest", "content_sha256"},
                context,
            )
            source_id = _required_string(task, "source", context)
            if source_id not in sources:
                raise SuiteManifestError(f"{context}.source references unknown {source_id!r}")
            upstream_name = _required_string(task, "upstream_name", context)
            content_sha256 = _digest(task, "content_sha256", context)
            upstream_digest = task.get("upstream_digest")
            if upstream_digest is not None:
                upstream_digest = _digest(task, "upstream_digest", context)
            resolved = SuiteTask(
                id=task_id,
                root=source_roots[source_id],
                source_id=source_id,
                upstream_name=upstream_name,
                content_sha256=content_sha256,
                upstream_digest=upstream_digest,
                provenance=dict(sources[source_id]),
            )
            if not (resolved.path / "task.toml").is_file():
                raise SuiteManifestError(
                    f"Task {task_id!r} has no task.toml below {resolved.root}"
                )
            if verify_content:
                actual = task_tree_digest(resolved.path)
                if actual != content_sha256:
                    raise SuiteManifestError(
                        f"Task {task_id!r} content digest mismatch: manifest "
                        f"{content_sha256}, actual {actual}"
                    )
            tasks[task_id] = resolved
            canonical_tasks[task_id] = resolved.identity()

        raw_tiers = raw.get("tiers")
        if not isinstance(raw_tiers, dict) or not raw_tiers:
            raise SuiteManifestError("suite.tiers must be a non-empty object")
        tiers: dict[str, tuple[str, ...]] = {}
        referenced: set[str] = set()
        for tier_id, tier_tasks in raw_tiers.items():
            context = f"tiers.{tier_id}"
            if not isinstance(tier_id, str) or not _ID_PATTERN.fullmatch(tier_id):
                raise SuiteManifestError(f"Invalid tier ID: {tier_id!r}")
            if not isinstance(tier_tasks, list) or not tier_tasks:
                raise SuiteManifestError(f"{context} must be a non-empty array")
            if not all(isinstance(task, str) for task in tier_tasks):
                raise SuiteManifestError(f"{context} must contain only task IDs")
            if len(tier_tasks) != len(set(tier_tasks)):
                raise SuiteManifestError(f"{context} contains duplicate task IDs")
            unknown = [task for task in tier_tasks if task not in tasks]
            if unknown:
                raise SuiteManifestError(
                    f"{context} references unknown task(s): {', '.join(unknown)}"
                )
            tiers[tier_id] = tuple(tier_tasks)
            referenced.update(tier_tasks)
        unreferenced = sorted(set(tasks) - referenced)
        if unreferenced:
            raise SuiteManifestError(
                "suite.tasks contains task(s) absent from every tier: "
                + ", ".join(unreferenced)
            )

        canonical_sources = {
            source_id: sources[source_id] for source_id in sorted(sources)
        }
        canonical = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "id": suite_id,
            "version": version,
            "task_hash_algorithm": TREE_HASH_ALGORITHM,
            "sources": canonical_sources,
            "tiers": {tier: list(tiers[tier]) for tier in sorted(tiers)},
            "tasks": {
                task_id: canonical_tasks[task_id] for task_id in sorted(canonical_tasks)
            },
        }
        return cls(
            path=path,
            id=suite_id,
            version=version,
            name=name,
            sources=sources,
            tiers=tiers,
            tasks=tasks,
            manifest_hash=_json_hash(canonical),
        )

    def tasks_for(self, tier: str) -> list[str]:
        try:
            return list(self.tiers[tier])
        except KeyError as exc:
            available = ", ".join(sorted(self.tiers))
            raise SuiteManifestError(
                f"Suite {self.id!r} has no tier {tier!r}; available: {available}"
            ) from exc

    def resolve_task(self, task_id: str) -> SuiteTask:
        try:
            return self.tasks[task_id]
        except KeyError as exc:
            raise SuiteManifestError(
                f"Task {task_id!r} is not part of suite {self.id!r}"
            ) from exc

    def grouped_tasks(self, requested: Iterable[str]) -> list[tuple[Path, list[str]]]:
        requested = list(requested)
        if len(requested) != len(set(requested)):
            raise SuiteManifestError("Requested task list contains duplicate task IDs")
        grouped: dict[Path, list[str]] = {}
        for task_id in requested:
            task = self.resolve_task(task_id)
            grouped.setdefault(task.root, []).append(task_id)
        return list(grouped.items())

    def identity(self) -> dict[str, Any]:
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "id": self.id,
            "version": self.version,
            "task_hash_algorithm": TREE_HASH_ALGORITHM,
            "manifest_hash": self.manifest_hash,
        }

    def task_identity(self, task_id: str) -> dict[str, Any]:
        task = self.resolve_task(task_id)
        return {**task.identity(), "provenance": dict(task.provenance)}
