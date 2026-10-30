from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pitbench.agent_diagnostics import (
    development_diagnosis as _development_diagnosis,
)
from pitbench.agent_diagnostics import (
    finite_number as _finite_number,
)
from pitbench.agent_diagnostics import (
    matching_base_feedback as _matching_base_feedback,
)
from pitbench.agent_diagnostics import (
    paired_feedback as _paired_feedback,
)
from pitbench.agent_diagnostics import (
    solver_behavior as _solver_behavior,
)

_CANDIDATE_PATHS = (
    ".",
    ":(exclude).pitbench",
    ":(exclude).pitbench/**",
)
_DEVELOPMENT_OUTPUT_LOCK = threading.Lock()


def hypothesis_command(args: argparse.Namespace) -> int:
    from pitbench.schema.research import CandidateHypothesis

    config = _config()
    declaration = CandidateHypothesis.model_validate_json(Path(args.plan).read_text())
    catalog = json.loads(
        (Path(__file__).parent / "schema/objective_catalog.json").read_text()
    )
    metrics = {
        metric
        for experiment in declaration.experiments
        for metric in experiment.metric_ids
    }
    if metrics - set(catalog["metrics"]):
        raise SystemExit("hypothesis uses unknown metric IDs")
    patch = _candidate_patch(_repository())
    if not patch:
        raise SystemExit("candidate hypothesis requires a nonempty source patch")
    patch_hash = hashlib.sha256(patch).hexdigest()
    destination = (
        _runs_root(config) / "hypotheses" / patch_hash / f"{declaration.id}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": "1.0",
        "task_id": config["task_id"],
        "patch_sha256": patch_hash,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": declaration.model_dump(mode="json"),
    }
    with destination.open("x") as output:
        json.dump(record, output, indent=2)
    print(
        json.dumps(
            {
                **record,
                "record_path": str(destination),
                "record_sha256": _sha256(destination),
            }
        )
    )
    return 0


def _candidate_experiment(
    config: dict, args: argparse.Namespace, patch: bytes, patch_hash: str
) -> dict | None:
    hypothesis_id = getattr(args, "hypothesis_id", None)
    experiment_id = getattr(args, "experiment_id", None)
    required = (
        config.get("candidate_hypotheses_required", False)
        and args.label == "agent"
        and bool(patch)
    )
    if required and (not hypothesis_id or not experiment_id):
        raise SystemExit(
            "register the candidate hypothesis first and select --hypothesis-id and --experiment-id"
        )
    if not hypothesis_id and not experiment_id:
        return None
    if not hypothesis_id or not experiment_id:
        raise SystemExit("hypothesis and experiment IDs must be supplied together")
    if any(
        not value
        or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for c in value
        )
        for value in (hypothesis_id, experiment_id)
    ):
        raise SystemExit("invalid hypothesis or experiment ID")
    path = _runs_root(config) / "hypotheses" / patch_hash / f"{hypothesis_id}.json"
    if not path.is_file():
        raise SystemExit("no registered hypothesis matches the current source patch")
    record = json.loads(path.read_text())
    if record["task_id"] != config["task_id"] or record["patch_sha256"] != patch_hash:
        raise SystemExit(
            "registered hypothesis identity differs from this task or patch"
        )
    if experiment_id not in {
        item["id"] for item in record["hypothesis"]["experiments"]
    }:
        raise SystemExit("experiment ID is absent from the registered hypothesis")
    return {
        "hypothesis_id": hypothesis_id,
        "experiment_id": experiment_id,
        "patch_sha256": patch_hash,
        "record_sha256": _sha256(path),
        "record_path": str(path),
    }


def _print_development_run(payload: dict) -> None:
    with _DEVELOPMENT_OUTPUT_LOCK:
        print(json.dumps(payload), flush=True)


def _path(environment: str, default: str) -> Path:
    return Path(os.environ.get(environment, default)).resolve()


def _runs_root(config: dict[str, Any]) -> Path:
    configured = _path("PITBENCH_RUNS", "/pitbench/runs")

    def prepare(path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=path):
            pass
        return path

    try:
        return prepare(configured)
    except OSError as configured_error:
        fallback = (
            Path(tempfile.gettempdir())
            / "pitbench-runs"
            / str(config.get("task_id", "unknown-task"))
        )
        try:
            path = prepare(fallback)
        except OSError as fallback_error:
            raise OSError(
                "PitBench could not write development results to either "
                f"{configured} ({configured_error}) or fallback {fallback} "
                f"({fallback_error})"
            ) from fallback_error
        print(
            f"PITBENCH_RUNS is not writable at {configured} "
            f"({configured_error}); using {path}",
            file=sys.stderr,
        )
        return path


def _config() -> dict[str, Any]:
    return json.loads(
        _path("PITBENCH_AGENT_CONFIG", "/pitbench/config.json").read_text()
    )


def _instances(
    selected: str | None, config: dict[str, Any] | None = None
) -> list[Path]:
    root = _path("PITBENCH_DEV_INSTANCES", "/pitbench/dev_instances")
    paths = sorted(
        path for path in root.glob("dev_*.*") if path.name != "instance_set_config.yaml"
    )
    if selected is None:
        return paths
    targets = (config or {}).get("development_targets", {})
    matches = [
        path
        for path in paths
        if path.stem == selected
        or path.name == selected
        or targets.get(path.name, {}).get("id") == selected
    ]
    if not matches:
        custom = Path(selected).expanduser()
        if custom.is_file():
            return [custom.resolve()]
    if not matches:
        raise SystemExit(f"development instance or file does not exist: {selected}")
    return matches


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_identity(config: dict[str, Any]) -> dict[str, Any]:
    repository = _repository()
    files: dict[str, str] = {}
    missing: list[str] = []
    for pattern in config.get("artifact_globs", []):
        matches = sorted(
            path
            for path in repository.glob(pattern)
            if path.is_file()
            and not path.name.endswith(("-sources.jar", "-javadoc.jar"))
        )
        if not matches:
            missing.append(pattern)
        for path in matches:
            files[str(path.relative_to(repository))] = _sha256(path)
    fingerprint = (
        hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        if files
        else None
    )
    return {"files": files, "fingerprint": fingerprint, "missing": missing}


def _target_for(config: dict[str, Any], instance: Path, digest: str) -> dict[str, Any]:
    targets = config.get("development_targets", {})
    direct = targets.get(instance.name)
    if direct is not None and direct.get("sha256") == digest:
        return direct
    return next(
        (item for item in targets.values() if item.get("sha256") == digest),
        {"id": instance.stem, "target": None},
    )


def _runner_available(config: dict[str, Any]) -> tuple[bool, str]:
    requirement = config.get("runner_requirement")
    if requirement and requirement.startswith("import:"):
        module = requirement.split(":", 1)[1]
        try:
            importlib.import_module(module)
        except Exception as exc:
            return False, f"{module} import failed: {exc}"
        return True, f"{module} import"
    if requirement and requirement.startswith("env:"):
        variable = requirement.split(":", 1)[1]
        available = bool(os.environ.get(variable))
        return available, variable if available else f"{variable} is not configured"
    if requirement and requirement.startswith("file:"):
        executable = _path("PITBENCH_REPO", "/workspace/repo") / requirement[5:]
        return executable.is_file(), str(executable)
    return True, "runner has no additional requirement"


def inspect_command(_: argparse.Namespace) -> int:
    config = _config()
    available, detail = _runner_available(config)
    workspace = _workspace_capability(config, _repository())
    payload = {
        "task_id": config["task_id"],
        "task_type": config["task_type"],
        "problem_family": config["problem_family"],
        "budgets_sec": config["budgets_sec"],
        "primary_budget_sec": config["primary_budget_sec"],
        "performance_protocol": config["performance_protocol"],
        "exact_time_basis": config["exact_time_basis"],
        "development_seeds": config["development_seeds"],
        "threads": config["threads"],
        "development_parallel_runs": config.get("development_parallel_runs", 1),
        "research_cpus": config.get("research_cpus"),
        "development_instances": [path.name for path in _instances(None)],
        "development_targets": config.get("development_targets", {}),
        "development_instance_directory": str(
            _path("PITBENCH_DEV_INSTANCES", "/pitbench/dev_instances")
        ),
        "runner_available": available,
        "runner_detail": detail,
        "runner_command_template": config["runner"],
        "runner_artifacts": _artifact_identity(config),
        "build_commands": config.get("build_commands", []),
        "editable_paths": config.get("editable_paths", []),
        "build_output_paths": config.get("build_output_paths", []),
        "workspace": workspace,
        "validation_available": bool(config.get("validation_commands")),
    }
    print(json.dumps(payload, indent=2))
    return 0


def _command(
    config: dict[str, Any],
    instance: Path,
    output: Path,
    seed: int,
    budget: float,
    threads: int,
) -> list[str]:
    values = {
        "instance": str(instance),
        "output": str(output),
        "trajectory": str(output.with_suffix(".trajectory.jsonl")),
        "seed": str(seed),
        "budget": str(budget),
        "threads": str(threads),
    }
    command = [part.format_map(values) for part in config["runner"]]
    fixed_parameters = config.get("fixed_solver_parameters")
    if fixed_parameters is not None:
        parameters_path = output.with_suffix(".parameters.json")
        parameters_path.write_text(json.dumps(fixed_parameters, sort_keys=True) + "\n")
        command.extend(["--parameters", str(parameters_path)])
    return command


def _public_verification(
    config: dict[str, Any], instance: Path, output: Path
) -> dict[str, Any]:
    solution_path = output.with_suffix(".solution.json")
    if not solution_path.is_file():
        return {"verification": "no_solution", "verified_valid": False}
    family = config["problem_family"]
    if family == "mip" and instance.suffix.lower() == ".lp":
        known_hashes = {
            item.get("sha256")
            for item in config.get("development_targets", {}).values()
        }
        if _sha256(instance) not in known_hashes:
            return {"verification": "unavailable", "verified_valid": None}
        from pitbench.problem_families.verification import (
            MIPFamily,
            _public_multidimensional_knapsack_data,
            _public_scheduling_processing_times,
        )

        if (
            _public_scheduling_processing_times(instance) is None
            and _public_multidimensional_knapsack_data(instance) is None
        ):
            return {"verification": "unavailable", "verified_valid": None}
        checked = MIPFamily().verify(instance, solution_path)
        return {
            "verification": "independent_public",
            "verified_valid": checked.feasible,
            "verified_objective": _finite_number(checked.objective),
            "verification_detail": checked.detail,
        }
    if family not in {"cvrp", "cp"} or instance.suffix.lower() != ".json":
        return {"verification": "unavailable", "verified_valid": None}
    from pitbench.problem_families.public_verification import (
        check_bin_packing,
        check_cvrp,
        check_graph_coloring,
    )

    try:
        instance_data = json.loads(instance.read_text())
        solution_data = json.loads(solution_path.read_text())
        if family == "cvrp":
            checked = check_cvrp(instance_data, solution_data)
        elif instance_data.get("problem_kind") == "graph_coloring":
            checked = check_graph_coloring(instance_data, solution_data)
        else:
            checked = check_bin_packing(
                solution_data,
                weights=instance_data["weights"],
                capacity=instance_data["capacity"],
            )
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        IndexError,
        json.JSONDecodeError,
    ) as error:
        checked = {"feasible": False, "detail": str(error)}
    return {
        "verification": "independent_public",
        "verified_valid": checked["feasible"],
        "verified_objective": _finite_number(checked.get("objective")),
        "verification_detail": checked.get("detail"),
    }


def _feedback(
    config: dict[str, Any],
    result: dict[str, Any],
    target: object,
    budget: float,
    instance: Path,
    output: Path,
) -> dict[str, Any]:
    reported_objective = _finite_number(result.get("objective"))
    anchor = _finite_number(target)
    reported_valid = result.get("valid") is True
    checked = _public_verification(config, instance, output)
    verified_valid = checked.get("verified_valid")
    objective = checked.get("verified_objective", reported_objective)
    valid = verified_valid if verified_valid is not None else reported_valid
    feedback: dict[str, Any] = {
        "reported_valid": reported_valid,
        "reported_objective": reported_objective,
        "reported_termination": result.get("solver_termination"),
        "reported_cpu_time_sec": _finite_number(result.get("cpu_time_sec")),
        "reported_wall_time_sec": _finite_number(result.get("wall_time_sec")),
        "public_target": anchor,
        **checked,
    }
    if config.get("agent_feedback_mode", "solver_behavior") == "solver_behavior":
        feedback["solver_behavior"] = _solver_behavior(config, result, output, budget)
    protocol = config["performance_protocol"]
    if protocol == "heuristic_fixed_budget":
        feedback["preview_normalized_gap"] = (
            (objective - anchor)
            / abs(anchor)
            * (1 if config["objective_sense"] == "minimize" else -1)
            if valid and objective is not None and anchor not in (None, 0)
            else None
        )
    elif protocol == "exact_verified_solve":
        solved = (
            valid
            and result.get("solver_termination") == "optimal"
            and objective == anchor
            and reported_objective == objective
            if anchor is not None
            else None
        )
        time_basis = config["exact_time_basis"]
        runtime = feedback[
            "reported_cpu_time_sec"
            if time_basis == "cpu_time"
            else "reported_wall_time_sec"
        ]
        feedback["preview_solved"] = solved
        feedback["preview_par2_sec"] = (
            runtime
            if solved and runtime is not None
            else 2 * budget
            if solved is False
            else None
        )
    return feedback


def _run_build_steps(
    config: dict[str, Any], *, dry_run: bool, runs_root: Path | None = None
) -> bool:
    commands = config.get("build_commands", [])
    if not commands:
        print("public build commands are unavailable", file=sys.stderr)
        return False
    log_root = (runs_root or _runs_root(config)) / "build"
    log_root.mkdir(parents=True, exist_ok=True)
    for index, step in enumerate(commands, start=1):
        argv = list(map(str, step["argv"]))
        cwd = _repository() / str(step.get("cwd", "."))
        payload: dict[str, Any] = {
            "build_step": index,
            "command": argv,
            "cwd": str(cwd),
        }
        if dry_run:
            print(json.dumps(payload))
            continue
        environment = os.environ.copy()
        environment.update(
            {str(key): str(value) for key, value in step.get("env", {}).items()}
        )
        try:
            completed = subprocess.run(
                argv,
                cwd=cwd,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=float(step.get("timeout_sec", 1800)),
            )
        except subprocess.TimeoutExpired:
            payload["error"] = "build step timed out"
            print(json.dumps(payload))
            return False
        stdout_path = log_root / f"step-{index}.stdout.log"
        stderr_path = log_root / f"step-{index}.stderr.log"
        stdout_path.write_text(completed.stdout)
        stderr_path.write_text(completed.stderr)
        payload.update(
            returncode=completed.returncode,
            stdout_log=str(stdout_path),
            stderr_log=str(stderr_path),
        )
        print(json.dumps(payload), flush=True)
        if completed.returncode:
            return False
    return True


def _run_one(
    config: dict[str, Any],
    instance: Path,
    seed: int,
    budget: float,
    dry_run: bool,
    label: str,
    patch_sha256: str | None,
    threads: int | None = None,
    artifacts: dict[str, Any] | None = None,
    runs_root: Path | None = None,
) -> bool:
    threads = threads if threads is not None else config["threads"]
    instance_sha256 = _sha256(instance)
    target_info = _target_for(config, instance, instance_sha256)
    artifacts = artifacts if artifacts is not None else _artifact_identity(config)
    run_root = (runs_root or _runs_root(config)) / label
    run_root.mkdir(parents=True, exist_ok=True)
    output = run_root / (
        f"{instance.stem}-{instance_sha256[:12]}.seed-{seed}."
        f"budget-{budget:g}.threads-{threads}.json"
    )
    command = _command(config, instance, output, seed, budget, threads)
    if dry_run:
        _print_development_run(
            {
                "instance": str(instance),
                "instance_id": target_info["id"],
                "instance_sha256": instance_sha256,
                "label": label,
                "seed": seed,
                "budget_sec": budget,
                "threads": threads,
                "command": command,
                "development_parallel_runs": config.get("development_parallel_runs", 1),
            }
        )
        return True
    available, detail = _runner_available(config)
    for old_path in (
        output,
        output.with_suffix(".solution.json"),
        output.with_suffix(".feedback.json"),
        output.with_suffix(".trajectory.jsonl"),
        output.with_suffix(".stdout.log"),
        output.with_suffix(".stderr.log"),
    ):
        old_path.unlink(missing_ok=True)
    started = time.perf_counter()
    timeout = budget + 60
    runner_error: str | None = None
    if not available:
        runner_error = f"runner unavailable: {detail}"
        stdout = ""
        stderr = ""
        returncode = None
    else:
        try:
            completed = subprocess.run(
                command,
                cwd=_path("PITBENCH_REPO", "/workspace/repo"),
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            stdout = completed.stdout
            stderr = completed.stderr
            returncode = completed.returncode
        except subprocess.TimeoutExpired as error:
            runner_error = f"runner timed out after {timeout:g} seconds"
            stdout = error.stdout or ""
            stderr = error.stderr or ""
            returncode = None
        except OSError as error:
            runner_error = f"runner could not start: {error}"
            stdout = ""
            stderr = ""
            returncode = None
    output.with_suffix(".stdout.log").write_text(
        stdout.decode(errors="replace") if isinstance(stdout, bytes) else stdout
    )
    output.with_suffix(".stderr.log").write_text(
        stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr
    )
    payload = {
        "instance": str(instance),
        "instance_id": target_info["id"],
        "instance_sha256": instance_sha256,
        "label": label,
        "seed": seed,
        "budget_sec": budget,
        "threads": threads,
        "development_parallel_runs": config.get("development_parallel_runs", 1),
        "declared_budget": budget in config["budgets_sec"],
        "performance_protocol": config["performance_protocol"],
        "patch_sha256": patch_sha256,
        "artifact_fingerprint": artifacts["fingerprint"],
        "artifact_files": artifacts["files"],
        "artifact_missing": artifacts["missing"],
        "returncode": returncode,
        "wall_time_sec": time.perf_counter() - started,
        "result": str(output),
    }
    if config.get("candidate_experiment") is not None:
        payload["candidate_experiment"] = config["candidate_experiment"]
    if runner_error is not None:
        payload["result_error"] = runner_error
    elif output.is_file():
        try:
            result = json.loads(output.read_text())
            target = target_info.get("target")
            payload.update(_feedback(config, result, target, budget, instance, output))
        except (
            json.JSONDecodeError,
            OSError,
            TypeError,
            ValueError,
            AttributeError,
        ) as error:
            payload["result_error"] = str(error)
    else:
        payload["result_error"] = "solver did not write a result"
    if label == "agent":
        match = _matching_base_feedback(run_root, payload)
        if match is not None:
            base_path, base = match
            payload["paired_base"] = str(base_path)
            payload.update(_paired_feedback(payload, base))
            if (
                patch_sha256 != base.get("patch_sha256")
                and artifacts["fingerprint"] is not None
                and not artifacts["missing"]
                and artifacts["fingerprint"] == base.get("artifact_fingerprint")
            ):
                payload["build_warning"] = (
                    "compiled runner artifacts are byte-identical to Base; "
                    "confirm the changed code affects the executed build"
                )
        else:
            payload["paired_base"] = None
    output.with_suffix(".feedback.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    _print_development_run(payload)
    return (
        returncode == 0
        and "result_error" not in payload
        and not (
            payload.get("verification") == "independent_public"
            and payload.get("verified_valid") is False
        )
    )


def bench_command(args: argparse.Namespace) -> int:
    if args.split != "dev":
        raise SystemExit("only the agent-visible dev split is available")
    config = _config()
    patch = _candidate_patch(_repository()) if not args.dry_run else b""
    if args.label == "base" and patch:
        print("base benchmark requires a clean repository", file=sys.stderr)
        return 2
    patch_sha256 = hashlib.sha256(patch).hexdigest() if not args.dry_run else None
    if not args.dry_run:
        experiment = _candidate_experiment(config, args, patch, patch_sha256)
        if experiment is not None:
            config = {**config, "candidate_experiment": experiment}
    seeds = (
        config["development_seeds"]
        if getattr(args, "all_seeds", False)
        else [args.seed if args.seed is not None else config["development_seeds"][0]]
    )
    budgets = (
        config["budgets_sec"]
        if getattr(args, "all_budgets", False)
        else [args.budget if args.budget is not None else config["primary_budget_sec"]]
    )
    threads = args.threads if args.threads is not None else config["threads"]
    if any(not math.isfinite(budget) or budget <= 0 for budget in budgets):
        raise SystemExit("budget must be a positive finite number")
    if threads < 1:
        raise SystemExit("threads must be positive")
    parallel = getattr(args, "jobs", None)
    parallel = (
        parallel if parallel is not None else config.get("development_parallel_runs", 1)
    )
    if type(parallel) is not int or parallel < 1:
        raise SystemExit("jobs must be a positive integer")
    cpus = config.get("research_cpus")
    if cpus is not None:
        parallel = min(parallel, max(1, math.floor(cpus / threads)))
    config = {**config, "development_parallel_runs": parallel}
    instances = _instances(args.instance, config)
    if not instances:
        print("no visible development instances are mounted", file=sys.stderr)
        return 2
    runs_root = _runs_root(config)
    if args.build and not _run_build_steps(
        config, dry_run=args.dry_run, runs_root=runs_root
    ):
        return 1
    artifacts = _artifact_identity(config)

    def run_one(spec):
        instance, seed, budget = spec
        return _run_one(
            config,
            instance,
            seed,
            budget,
            args.dry_run,
            args.label,
            patch_sha256,
            threads,
            artifacts,
            runs_root,
        )

    grid = (
        (instance, seed, budget)
        for instance in instances
        for seed in seeds
        for budget in budgets
    )
    with ThreadPoolExecutor(max_workers=parallel) as executor:
        succeeded = list(executor.map(run_one, grid))
    return 0 if all(succeeded) else 1


def diagnose_command(args: argparse.Namespace) -> int:
    config = _config()
    budgets = (
        list(config["budgets_sec"])
        if args.all_budgets
        else [args.budget if args.budget is not None else config["primary_budget_sec"]]
    )
    if any(not math.isfinite(budget) or budget <= 0 for budget in budgets):
        raise SystemExit("budget must be a positive finite number")
    run_root = _runs_root(config)
    diagnosis = _development_diagnosis(config, run_root, budgets)
    output = run_root / "diagnosis.json"
    diagnosis["saved_to"] = str(output)
    output.write_text(json.dumps(diagnosis, indent=2) + "\n")
    print(json.dumps(diagnosis, indent=2))
    return 0 if diagnosis["paired_runs"] else 1


def _repository() -> Path:
    return _path("PITBENCH_REPO", "/workspace/repo")


def _workspace_capability(config: dict[str, Any], repository: Path) -> dict[str, Any]:
    editable = tuple(map(str, config.get("editable_paths", [])))
    editable_writable = {
        path: (repository / path).is_dir() and os.access(repository / path, os.W_OK)
        for path in editable
    }
    root_writable = os.access(repository, os.W_OK)
    git_writable = os.access(repository / ".git", os.W_OK)
    effective_uid = os.geteuid()
    return {
        "enforced": (
            effective_uid != 0
            and not root_writable
            and not git_writable
            and bool(editable_writable)
            and all(editable_writable.values())
        ),
        "effective_uid": effective_uid,
        "repository_root_writable": root_writable,
        "git_metadata_writable": git_writable,
        "editable_paths_writable": editable_writable,
    }


def workspace_command(_: argparse.Namespace) -> int:
    result = _workspace_capability(_config(), _repository())
    print(json.dumps(result, indent=2))
    return 0 if result["enforced"] else 1


def _git_paths(repository: Path, *arguments: str) -> list[str]:
    completed = subprocess.run(
        ["git", *arguments, "-z", "--", *_CANDIDATE_PATHS],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    return [
        value.decode(errors="surrogateescape")
        for value in completed.stdout.split(b"\0")
        if value
    ]


def _candidate_patch(repository: Path) -> bytes:
    tracked = subprocess.run(
        [
            "git",
            "diff",
            "--binary",
            "--no-ext-diff",
            "HEAD",
            "--",
            *_CANDIDATE_PATHS,
        ],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    patch = bytearray(tracked.stdout)
    for path in _git_paths(repository, "ls-files", "--others", "--exclude-standard"):
        untracked = subprocess.run(
            [
                "git",
                "diff",
                "--binary",
                "--no-ext-diff",
                "--no-index",
                "--",
                "/dev/null",
                path,
            ],
            cwd=repository,
            check=False,
            capture_output=True,
        )
        if untracked.returncode not in {0, 1}:
            raise subprocess.CalledProcessError(
                untracked.returncode,
                untracked.args,
                output=untracked.stdout,
                stderr=untracked.stderr,
            )
        patch.extend(untracked.stdout)
    return bytes(patch)


def validate_command(_: argparse.Namespace) -> int:
    config = _config()
    repository = _repository()
    commands = config.get("validation_commands") or []
    if not commands:
        print("public validation is unavailable for this task", file=sys.stderr)
        return 2

    patch = _candidate_patch(repository)
    with tempfile.TemporaryDirectory(prefix="pitbench-public-validation-") as root:
        workspace = Path(root) / "workspace"
        cloned = subprocess.run(
            ["git", "clone", "--quiet", "--no-hardlinks", str(repository), workspace],
            check=False,
            capture_output=True,
            text=True,
        )
        if cloned.returncode:
            print(cloned.stderr or cloned.stdout, file=sys.stderr)
            return cloned.returncode
        if patch:
            applied = subprocess.run(
                ["git", "apply", "--whitespace=nowarn", "-"],
                cwd=workspace,
                input=patch,
                check=False,
                capture_output=True,
            )
            if applied.returncode:
                print(applied.stderr.decode(errors="replace"), file=sys.stderr)
                return applied.returncode

        for index, command in enumerate(commands, start=1):
            argv = list(map(str, command["argv"]))
            cwd = workspace / str(command.get("cwd", "."))
            environment = os.environ.copy()
            environment.update(
                {str(key): str(value) for key, value in command.get("env", {}).items()}
            )
            print(
                json.dumps(
                    {"validation_step": index, "command": argv, "cwd": str(cwd)}
                ),
                flush=True,
            )
            try:
                completed = subprocess.run(
                    argv,
                    cwd=cwd,
                    env=environment,
                    check=False,
                    timeout=float(command.get("timeout_sec", 1800)),
                )
            except subprocess.TimeoutExpired:
                print(f"validation step {index} timed out", file=sys.stderr)
                return 124
            if completed.returncode:
                return completed.returncode
    return 0


def _run_options(parser: argparse.ArgumentParser, *, instance_required: bool) -> None:
    parser.add_argument(
        "--instance",
        required=instance_required,
        help="visible instance name/ID or any readable instance file; omit for all visible instances",
    )
    seed_selection = parser.add_mutually_exclusive_group()
    seed_selection.add_argument(
        "--seed", type=int, help="solver seed; defaults to first development seed"
    )
    seed_selection.add_argument(
        "--all-seeds", action="store_true", help="run every declared development seed"
    )
    budget_selection = parser.add_mutually_exclusive_group()
    budget_selection.add_argument(
        "--budget",
        type=float,
        help="positive time budget in seconds; defaults to primary budget",
    )
    budget_selection.add_argument(
        "--all-budgets", action="store_true", help="run every declared budget"
    )
    parser.add_argument(
        "--threads", type=int, help="solver threads; defaults to task setting"
    )
    parser.add_argument("--dry-run", action="store_true")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="pitbench")
    commands = result.add_subparsers(required=True)
    tools = set(_config().get("agent_tools", []))
    if "inspect" in tools:
        inspect_parser = commands.add_parser("inspect")
        inspect_parser.set_defaults(handler=inspect_command)
    if "workspace" in tools:
        workspace = commands.add_parser("workspace")
        workspace.set_defaults(handler=workspace_command)
    if "bench" in tools:
        hypothesis = commands.add_parser(
            "hypothesis",
            help="Register applicability, failure predictions and experiments for this patch",
        )
        hypothesis.add_argument("--plan", type=Path, required=True)
        hypothesis.set_defaults(handler=hypothesis_command)
        bench = commands.add_parser("bench")
        bench.add_argument("--hypothesis-id")
        bench.add_argument("--experiment-id")
        bench.add_argument("--split", default="dev")
        bench.add_argument("--label", choices=("base", "agent"), default="agent")
        bench.add_argument(
            "--build",
            action="store_true",
            help="rebuild the current source before solving",
        )
        bench.add_argument(
            "--jobs",
            type=int,
            help="development workers; defaults to the task preference and fits its CPU allowance",
        )
        _run_options(bench, instance_required=False)
        bench.set_defaults(handler=bench_command)
        diagnose = commands.add_parser("diagnose")
        budget_selection = diagnose.add_mutually_exclusive_group()
        budget_selection.add_argument(
            "--budget",
            type=float,
            help="diagnose one budget; defaults to the primary budget",
        )
        budget_selection.add_argument(
            "--all-budgets",
            action="store_true",
            help="diagnose every declared development budget",
        )
        diagnose.set_defaults(handler=diagnose_command)
    if "validate" in tools:
        validate = commands.add_parser("validate")
        validate.set_defaults(handler=validate_command)
    return result


def main() -> None:
    args = parser().parse_args()
    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()
