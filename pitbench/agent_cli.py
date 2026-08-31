from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_CANDIDATE_PATHS = (
    ".",
    ":(exclude).pitbench",
    ":(exclude).pitbench/**",
)


def _path(environment: str, default: str) -> Path:
    return Path(os.environ.get(environment, default)).resolve()


def _config() -> dict[str, Any]:
    return json.loads(
        _path("PITBENCH_AGENT_CONFIG", "/pitbench/config.json").read_text()
    )


def _instances(selected: str | None) -> list[Path]:
    root = _path("PITBENCH_DEV_INSTANCES", "/pitbench/dev_instances")
    paths = sorted(
        path for path in root.glob("dev_*.*") if path.name != "population.yaml"
    )
    if selected is None:
        return paths
    matches = [path for path in paths if path.stem == selected or path.name == selected]
    if not matches:
        raise SystemExit(f"unknown development instance: {selected}")
    return matches


def _runner_available(config: dict[str, Any]) -> tuple[bool, str]:
    requirement = config.get("runner_requirement")
    if requirement == "pyvrp_import":
        try:
            importlib.import_module("pyvrp")
        except Exception as exc:
            return False, f"pyvrp import failed: {exc}"
        return True, "pyvrp import"
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
        "solver_seeds": config["solver_seeds"],
        "threads": config["threads"],
        "development_instances": [path.name for path in _instances(None)],
        "runner_available": available,
        "runner_detail": detail,
        "editable_paths": config.get("editable_paths", []),
        "workspace": workspace,
        "validation_available": bool(config.get("validation_commands")),
    }
    print(json.dumps(payload, indent=2))
    return 0


def _command(
    config: dict[str, Any], instance: Path, output: Path, seed: int, budget: float
) -> list[str]:
    values = {
        "instance": str(instance),
        "output": str(output),
        "trajectory": str(output.with_suffix(".trajectory.jsonl")),
        "seed": str(seed),
        "budget": str(budget),
        "threads": str(config["threads"]),
    }
    return [part.format_map(values) for part in config["runner"]]


def _run_one(
    config: dict[str, Any], instance: Path, seed: int, budget: float, dry_run: bool
) -> bool:
    run_root = _path("PITBENCH_RUNS", "/pitbench/runs")
    run_root.mkdir(parents=True, exist_ok=True)
    output = run_root / f"{instance.stem}.seed-{seed}.budget-{budget:g}.json"
    command = _command(config, instance, output, seed, budget)
    if dry_run:
        print(json.dumps({"instance": instance.name, "command": command}))
        return True
    available, detail = _runner_available(config)
    if not available:
        print(f"runner unavailable: {detail}", file=sys.stderr)
        return False
    started = time.perf_counter()
    timeout = budget + 60
    try:
        completed = subprocess.run(
            command,
            cwd=_path("PITBENCH_REPO", "/workspace/repo"),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(
            f"runner timed out after {timeout:g} seconds: {instance.name}",
            file=sys.stderr,
        )
        return False
    output.with_suffix(".stdout.log").write_text(completed.stdout)
    output.with_suffix(".stderr.log").write_text(completed.stderr)
    payload = {
        "instance": instance.name,
        "returncode": completed.returncode,
        "wall_time_sec": time.perf_counter() - started,
        "result": str(output),
    }
    print(json.dumps(payload))
    return completed.returncode == 0 and output.is_file()


def bench_command(args: argparse.Namespace) -> int:
    if args.split != "dev":
        raise SystemExit("only the agent-visible dev split is available")
    config = _config()
    seed = args.seed if args.seed is not None else config["solver_seeds"][0]
    budget = args.budget if args.budget is not None else config["budgets_sec"][0]
    succeeded = [
        _run_one(config, instance, seed, budget, args.dry_run)
        for instance in _instances(args.instance)
    ]
    return 0 if all(succeeded) else 1


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
    parser.add_argument("--instance", required=instance_required)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--budget", type=float)
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
        bench = commands.add_parser("bench")
        bench.add_argument("--split", default="dev")
        _run_options(bench, instance_required=False)
        bench.set_defaults(handler=bench_command)
    if "validate" in tools:
        validate = commands.add_parser("validate")
        validate.set_defaults(handler=validate_command)
    return result


def main() -> None:
    args = parser().parse_args()
    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()
