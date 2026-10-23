"""Real-container B1 acceptance probe; uses the production execution boundary."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

from pitbench.evaluator.isolation import ISOLATION_VERSION, execute, protect_output


def main() -> None:
    work = Path("/tmp/b1-work")
    outputs = Path("/tmp/b1-output")
    work.mkdir()
    outputs.mkdir()
    secret = Path(sys.argv[1])
    assert secret.is_file() and secret.read_bytes()
    protect_output(Path("/output"), Path("/output").stat().st_gid)
    old_result = Path("/output/old-result.json")
    old_result.write_text("old-hidden-answer")
    root_output = Path("/tmp/b1-root-group-output")
    root_output.mkdir()
    root_result = root_output / "old-result.json"
    root_result.write_text("root-group-old-answer")
    protect_output(root_output, 0)
    os.environ["GITHUB_TOKEN"] = "b1-controller-only-marker"
    (work / "alias").symlink_to(secret)

    def run(argv, *, build=False, readable=(), timeout=60):
        result = execute(
            SimpleNamespace(argv=argv, env={}, cwd=".", timeout_sec=timeout),
            work,
            readable=readable,
            writable=(work,) if build else (outputs,),
        )
        return result

    checks = []
    for name, path in (
        ("private_path", str(secret)),
        ("traversal", "/tmp/b1-work/../../" + str(secret).lstrip("/")),
        ("symlink", str(work / "alias")),
        ("proc_root", "/proc/self/root" + str(secret)),
        (
            "reference_patch",
            "/opt/pitbench/configs/references/pyvrp_pr1156_source.patch",
        ),
    ):
        script = f"""
from pathlib import Path
try:
    Path({path!r}).read_bytes()
except (PermissionError, FileNotFoundError):
    print('denied')
else:
    raise AssertionError('hidden read allowed')
"""
        result = run([sys.executable, "-c", script])
        assert result.returncode == 0 and result.stdout.strip() == "denied", (
            name,
            result.stderr,
        )
        checks.append(name)
    result = run(
        [
            sys.executable,
            "-c",
            "import os,socket; assert not os.getenv('GITHUB_TOKEN'); socket.socket()",
        ]
    )
    assert result.returncode != 0 and "PermissionError" in result.stderr
    checks.extend(["environment", "network"])
    for path in (secret, old_result, root_result):
        result = run([sys.executable, "-c", f"import os; os.stat({str(path)!r})"])
        assert result.returncode != 0 and "PermissionError" in result.stderr, (
            result.stderr
        )
    checks.extend(
        ["private_metadata", "old_artifact_metadata", "root_group_artifact_metadata"]
    )

    tools = []
    if shutil.which("g++"):
        source = work / "probe.cpp"
        source.write_text(
            '#include <fstream>\n#include <iostream>\nint main(int n,char** v){std::ifstream f(v[1]); if(f) return 42; std::cout<<"denied"; return 0;}\n'
        )
        built = run(["g++", str(source), "-o", str(work / "probe")], build=True)
        assert built.returncode == 0, built.stderr
        result = run([str(work / "probe"), str(secret)])
        assert result.returncode == 0 and result.stdout == "denied", result.stderr
        tools.append("native_cpp_compile_and_denied_read")
    if shutil.which("javac"):
        source = work / "Probe.java"
        source.write_text(
            'import java.nio.file.*; public class Probe { public static void main(String[] a) throws Exception { try { Files.readAllBytes(Path.of(a[0])); throw new AssertionError("hidden read allowed"); } catch (AccessDeniedException e) { System.out.println("denied"); } } }\n'
        )
        built = run(["javac", str(source)], build=True)
        assert built.returncode == 0, built.stderr
        result = run(["java", "-cp", str(work), "Probe", str(secret)])
        assert result.returncode == 0 and result.stdout.strip() == "denied", (
            result.stderr
        )
        tools.append("java_compile_and_denied_read")

    # Run an actual pinned solver on one existing public development instance.
    # Its configured budget, seed and parameters come from the task unchanged.
    solver = None
    if len(sys.argv) > 2:
        from pitbench.problem_families.base import ProblemFamilyRegistry
        from pitbench.repositories.base import (
            BuildKind,
            RepositoryPluginRegistry,
            SolverRunSpec,
        )
        from pitbench.schema.task import PitBenchTask

        task = PitBenchTask.from_yaml(Path("/input/task.yaml"))
        repository = RepositoryPluginRegistry.load(task.repository.plugin)
        workspace = Path("/workspace/repo")
        builds = []
        if not any(workspace.iterdir()):
            workspace = Path("/tmp/b1-solver-workspace")
            shutil.copytree("/input/base", workspace, symlinks=True)
            assert (workspace / ".git").exists(), (
                "source-only judge image requires its pinned source cache"
            )
            for build in repository.build_commands(BuildKind.PERFORMANCE):
                built = execute(build, workspace, writable=(workspace,))
                assert built.returncode == 0, built.stderr or built.stdout
                builds.append({"argv": build.argv, "isolation": built.isolation})
        instance = Path(sys.argv[2])
        result_path = outputs / "solver.json"
        parameters = None
        if task.evaluation.fixed_solver_parameters:
            parameters = Path("/tmp/frozen-parameters.json")
            parameters.write_text(json.dumps(task.evaluation.fixed_solver_parameters))
        seeds = (
            task.evaluation.solver_seeds
            or task.evaluation.seed_robustness.development_seeds
        )
        command = repository.run_command(
            SolverRunSpec(
                instance_path=instance,
                output_path=result_path,
                trajectory_path=outputs / "solver.trajectory.jsonl",
                solver_seed=seeds[0],
                budget_sec=task.evaluation.primary_budget_sec,
                threads=task.evaluation.threads,
                parameters_path=parameters,
            )
        )
        command.argv = [
            "taskset",
            "--cpu-list",
            str(min(os.sched_getaffinity(0))),
            *command.argv,
        ]
        completed = execute(
            command,
            workspace,
            readable=(instance, *((parameters,) if parameters else ())),
            writable=(outputs,),
        )
        assert completed.returncode == 0, completed.stderr
        parsed = repository.parse_output(result_path)
        assert parsed.valid, parsed
        verifier = repository.prepare_independent_verifier(
            Path("/tmp/trusted-verifier")
        )
        verifier = verifier or ProblemFamilyRegistry.load(task.problem_family)
        checked = verifier.verify(instance, result_path.with_suffix(".solution.json"))
        assert checked.feasible and not checked.infrastructure_error, checked
        solver = {
            "task_id": task.task_id,
            "budget_sec": task.evaluation.primary_budget_sec,
            "solver_seed": seeds[0],
            "threads": task.evaluation.threads,
            "independent_verification": checked.model_dump(mode="json"),
            "isolation": completed.isolation,
            "builds": builds,
        }
    print(
        json.dumps(
            {
                "isolation_version": ISOLATION_VERSION,
                "checks": checks,
                "toolchains": tools,
                "solver": solver,
            }
        )
    )


if __name__ == "__main__":
    main()
