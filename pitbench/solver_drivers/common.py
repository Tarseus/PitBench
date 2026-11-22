from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import resource
import subprocess
import time
from enum import Enum
from pathlib import Path
from typing import Any


class ScenarioUnsupported(RuntimeError):
    """The selected solver API cannot execute this declared scenario."""


class ScenarioSession:
    """One solver object in one worker; implementations live with other drivers."""

    @classmethod
    def load(cls, import_path: str):
        import importlib

        module, name = import_path.split(":", 1)
        implementation = getattr(importlib.import_module(module), name)
        if not issubclass(implementation, cls):
            raise TypeError("scenario session plugin must implement ScenarioSession")
        return implementation

    @classmethod
    def identity(cls) -> dict:
        raise NotImplementedError

    @classmethod
    def probe(cls) -> dict:
        import inspect

        sources = {
            "session": hashlib.sha256(
                Path(inspect.getfile(cls)).read_bytes()
            ).hexdigest(),
            "support": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        return {
            **cls.identity(),
            "session_code_sha256": hashlib.sha256(
                json.dumps(sources, sort_keys=True).encode()
            ).hexdigest(),
        }

    def load_model(self, model: dict) -> None:
        raise NotImplementedError

    def update_model(self, previous: dict, model: dict) -> None:
        raise ScenarioUnsupported("in-place updates are unavailable for this session")

    def solve(self, emit) -> dict:
        raise NotImplementedError

    def reset_search(self) -> None:
        raise ScenarioUnsupported("search reset is unavailable for this session")

    def close(self) -> None:
        """Release resources held by a scenario session."""

    def collect_garbage(self) -> None:
        """Release language-managed temporary objects when supported."""

    def measurement_pid(self) -> int:
        return os.getpid()


def _package_artifacts(package_name: str) -> dict:
    package = Path(importlib.import_module(package_name).__file__).resolve().parent
    files = {
        str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package.rglob("*"))
        if path.is_file() and (path.suffix == ".py" or path.suffix == ".so")
    }
    native = {name: digest for name, digest in files.items() if name.endswith(".so")}
    if not native:
        raise ValueError("source binding has no native solver artifact")

    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    return {"binding_sha256": digest(native), "package_code_sha256": digest(files)}


def _process_artifacts(source: Path, repository_plugin: str) -> dict:
    """Bind an executable-based scenario to the actual built solver artifacts."""
    from pitbench.repositories.base import RepositoryPluginRegistry

    plugin = RepositoryPluginRegistry.load(repository_plugin)
    artifacts = {}
    for pattern in plugin.agent_artifact_globs:
        for path in sorted(source.glob(pattern)):
            if path.is_file():
                resolved = path.resolve()
                if not resolved.is_relative_to(source.resolve()):
                    raise ValueError("scenario artifact is outside its source workspace")
                artifacts[path.relative_to(source).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not artifacts:
        raise ValueError("source-bound process has no built solver artifacts")
    support = {}
    adapter_globs = getattr(plugin, "scenario_artifact_globs", ())
    for pattern in adapter_globs:
        for path in sorted(source.glob(pattern)):
            if path.is_file():
                if not path.resolve().is_relative_to(source.resolve()):
                    raise ValueError("scenario adapter is outside its source workspace")
                support[path.relative_to(source).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    if adapter_globs and not support:
        raise ValueError("source-bound scenario adapter has not been built")
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return {"binding_sha256": digest(artifacts), "package_code_sha256": digest(support)}


def record_source_binding(
    *,
    source: Path,
    package: str | None,
    repository_plugin: str,
    version: str,
    destination: Path,
    native_library: Path | None = None,
    process_artifacts: bool = False,
) -> dict:
    """Called by the evaluator after its source build, never by an Agent claim."""
    from pitbench.agent_cli import _candidate_patch
    from pitbench.repositories.base import (
        RepositoryPluginRegistry,
        ignore_generated_outputs,
    )

    source = source.resolve()
    if package is not None:
        module_path = Path(importlib.import_module(package).__file__).resolve()
        if not module_path.is_relative_to(source):
            raise ValueError("scenario package is outside its source workspace")
    ignore_generated_outputs(
        source, RepositoryPluginRegistry.load(repository_plugin).agent_build_paths
    )
    record = {
        "source": str(source),
        "package": package,
        "repository_plugin": repository_plugin,
        "version": version,
        "binding_git_hash": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip(),
        "source_patch_sha256": hashlib.sha256(_candidate_patch(source)).hexdigest(),
        **(_package_artifacts(package) if package is not None else
           _process_artifacts(source, repository_plugin) if process_artifacts else
           _native_artifacts(native_library, source)),
    }
    if process_artifacts:
        record["binding_kind"] = "process_artifacts"
        record["artifact_version"] = 2
    if native_library is not None:
        record["native_library"] = str(native_library.resolve())
        if record["native_version"] != version:
            raise ValueError("native binding version differs from the source task")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2) + "\n")
    return record


def source_binding_identity(
    cls, *, package: str | None, repository_plugin: str, representations: list[str]
) -> dict:
    from pitbench.agent_cli import _candidate_patch

    path = os.environ.get("PITBENCH_SCENARIO_BINDING")
    if not path:
        raise ValueError(
            "PITBENCH_SCENARIO_BINDING must name an evaluator-owned build receipt"
        )
    record = json.loads(Path(path).read_text())
    if record["package"] != package or record["repository_plugin"] != repository_plugin:
        raise ValueError("binding receipt belongs to another solver")
    source = Path(record["source"])
    if package is not None:
        module_path = Path(importlib.import_module(package).__file__).resolve()
        if not module_path.is_relative_to(source.resolve()):
            raise ValueError("scenario package is outside its source workspace")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    if (
        commit != record["binding_git_hash"]
        or hashlib.sha256(_candidate_patch(source)).hexdigest()
        != record["source_patch_sha256"]
    ):
        raise ValueError("source changed after the binding build")
    artifacts = (
        _package_artifacts(package) if package is not None else
        _process_artifacts(source, repository_plugin) if record.get("binding_kind") == "process_artifacts" else
        _native_artifacts(Path(record["native_library"]), source)
    )
    checked_artifacts = artifacts
    if record.get("binding_kind") == "process_artifacts" and record.get("artifact_version", 1) < 2:
        # Older receipts included the evaluator's Python files. Their changes
        # do not alter the already-built solver, so retain the solver check and
        # report the current compiled adapter identity without rejecting it.
        checked_artifacts = {"binding_sha256": artifacts["binding_sha256"]}
    if any(checked_artifacts[key] != record[key] for key in checked_artifacts):
        raise ValueError("solver package changed after its source build")
    return {
        "plugin": f"{cls.__module__}:{cls.__name__}",
        "repository_plugin": repository_plugin,
        "representations": representations,
        "version": record["version"],
        "binding_git_hash": commit,
        "provenance_kind": "evaluator_source_build_receipt",
        **artifacts,
    }


def _native_artifacts(library: Path, source: Path) -> dict:
    import ctypes

    binding = ctypes.CDLL(str(library.resolve()))
    for name in (
        "pbs_version",
        "pbs_git_hash",
        "pbs_solver_library",
        "pbs_adapter_sha256",
    ):
        getattr(binding, name).restype = ctypes.c_char_p
    linked = Path(binding.pbs_solver_library().decode()).resolve()
    if not linked.is_relative_to(source.resolve()):
        raise ValueError(
            "scenario binding loaded a solver library outside its source build"
        )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    if binding.pbs_git_hash().decode() != commit:
        raise ValueError("native instrumentation belongs to another source commit")
    adapter = Path(__file__).resolve().parents[2] / "adapters/cpp/ScenarioSession.cpp"
    if (
        binding.pbs_adapter_sha256().decode()
        != hashlib.sha256(adapter.read_bytes()).hexdigest()
    ):
        raise ValueError(
            "native instrumentation source differs from its compiled artifact"
        )
    artifacts = {
        "bridge": hashlib.sha256(library.read_bytes()).hexdigest(),
        "solver": hashlib.sha256(linked.read_bytes()).hexdigest(),
    }
    return {
        "binding_sha256": hashlib.sha256(
            json.dumps(artifacts, sort_keys=True).encode()
        ).hexdigest(),
        "package_code_sha256": hashlib.sha256(
            Path(__file__).with_name("run.py").read_bytes()
        ).hexdigest(),
        "native_version": binding.pbs_version().decode(),
    }


class SolverTermination(str, Enum):
    OPTIMAL = "optimal"
    TIME_LIMIT = "time_limit"
    ERROR = "solver_error"
    OTHER = "other"


def process_resources(*, child_process: bool = False) -> dict[str, Any]:
    """Snapshot one solver process, before result parsing or verification.

    The default measures the fresh Python worker through solver return. The
    child mode is for a fresh driver that has run exactly one native solver
    subprocess: its RSS is that subprocess's lifetime peak, not the maximum
    of unrelated driver/child peaks or a process-tree memory measurement.
    """
    usage = resource.getrusage(
        resource.RUSAGE_CHILDREN if child_process else resource.RUSAGE_SELF
    )
    return {
        "cpu_time_sec": usage.ru_utime + usage.ru_stime,
        "peak_rss_bytes": int(usage.ru_maxrss * 1024),
        "resource_scope": (
            "single_native_child_lifetime"
            if child_process
            else "worker_process_start_through_solver_return"
        ),
    }


def parser(*, solver: bool = False, trajectory: bool = True) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    if solver:
        result.add_argument("--solver", required=True)
    result.add_argument("--instance", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    if trajectory:
        result.add_argument("--trajectory", type=Path, required=True)
    result.add_argument("--seed", type=int, required=True)
    result.add_argument("--budget", type=float, required=True)
    result.add_argument("--threads", type=int, required=True)
    return result


def write_result(
    path: Path,
    *,
    started: float,
    valid: bool,
    objective: float | None = None,
    error: str | None = None,
    **metrics: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "valid": valid,
                "objective": objective,
                "wall_time_sec": time.perf_counter() - started,
                "error": error,
                **metrics,
            },
            indent=2,
        )
    )


def write_solution(output: Path, payload: dict[str, Any]) -> Path:
    path = output.with_suffix(".solution.json")
    path.write_text(json.dumps(payload, indent=2))
    return path


def failure_reason(error: Exception) -> str | None:
    """Preserve explicit failures across a driver subprocess boundary."""
    if isinstance(error, subprocess.TimeoutExpired):
        return "timed_out"
    if isinstance(error, MemoryError):
        return "out_of_memory"
    if isinstance(error, ParameterRejected):
        return "parameter_rejected"
    return None


def append_trajectory(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload))
        handle.write("\n")


class ParameterRejected(ValueError):
    """A solver rejected parameters that the declared domain allowed."""


def finite(value):
    value = float(value)
    return value if math.isfinite(value) else None
