from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _choco_assembly(repository: Path) -> Path:
    assemblies = sorted(
        (repository / "solver" / "target").glob(
            "choco-solver-*-jar-with-dependencies.jar"
        )
    )
    if len(assemblies) != 1:
        raise RuntimeError(
            f"expected one Choco assembly, found {[path.name for path in assemblies]}"
        )
    return assemblies[0]


def _ortools_java_artifact(repository: Path, artifact: str) -> Path:
    artifacts = sorted(
        (repository / "build" / "java" / artifact / "target").glob("*.jar")
    )
    artifacts = [
        path
        for path in artifacts
        if "-sources" not in path.name and "-javadoc" not in path.name
    ]
    if len(artifacts) != 1:
        raise RuntimeError(
            f"expected one {artifact} Java artifact, found "
            f"{[path.name for path in artifacts]}"
        )
    return artifacts[0]


def _maven_artifact(*path_parts: str) -> Path:
    artifacts = sorted(
        (Path.home() / ".m2" / "repository").joinpath(*path_parts).glob("*/*.jar")
    )
    if len(artifacts) != 1:
        raise RuntimeError(
            f"expected one Maven artifact {'/'.join(path_parts)}, found "
            f"{[path.name for path in artifacts]}"
        )
    return artifacts[0]


def _ortools_classpath(repository: Path, output: Path) -> str:
    return os.pathsep.join(
        [
            str(output),
            str(_ortools_java_artifact(repository, "ortools-java")),
            str(_ortools_java_artifact(repository, "ortools-linux-x86-64")),
            str(_maven_artifact("com", "google", "protobuf", "protobuf-java")),
            str(_maven_artifact("net", "java", "dev", "jna", "jna")),
            str(_maven_artifact("net", "java", "dev", "jna", "jna-platform")),
        ]
    )


def _adapter_output(repository: Path, solver: str) -> Path:
    if solver == "ortools_cp_sat":
        return repository / "build" / "pitbench-adapter"
    return repository / "solver" / "target" / "pitbench-adapter"


_ORTOOLS_CP_SAT_RUNNER_SOURCE = """\
package org.pitbench.adapters;

import com.google.ortools.Loader;
import com.google.ortools.sat.CpModel;
import com.google.ortools.sat.CpModelProto;
import com.google.ortools.sat.CpSolver;
import com.google.ortools.sat.CpSolverResponse;
import com.google.ortools.sat.CpSolverStatus;
import com.google.ortools.sat.IntVar;
import com.google.ortools.sat.IntervalVar;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public final class OrToolsCpSatRunner {
  private OrToolsCpSatRunner() {}

  private static String jsonLongArray(CpSolverResponse response) {
    StringBuilder result = new StringBuilder("[");
    for (int index = 0; index < response.getSolutionCount(); index++) {
      if (index > 0) {
        result.append(',');
      }
      result.append(response.getSolution(index));
    }
    return result.append(']').toString();
  }

  private static String jsonIntArray(int[] values) {
    StringBuilder result = new StringBuilder("[");
    for (int index = 0; index < values.length; index++) {
      if (index > 0) {
        result.append(',');
      }
      result.append(values[index]);
    }
    return result.append(']').toString();
  }

  private static int[] readJobShopIntegers(Path instancePath) throws IOException {
    List<Integer> values = new ArrayList<>();
    for (String line : Files.readAllLines(instancePath)) {
      String trimmed = line.trim();
      if (trimmed.isEmpty() || trimmed.startsWith("#")) {
        continue;
      }
      try (Scanner scanner = new Scanner(trimmed)) {
        while (scanner.hasNext()) {
          if (!scanner.hasNextInt()) {
            throw new IllegalArgumentException("job-shop instance contains a non-integer token");
          }
          values.add(scanner.nextInt());
        }
      }
    }
    int[] result = new int[values.size()];
    for (int index = 0; index < values.size(); index++) {
      result[index] = values.get(index);
    }
    return result;
  }

  private static void prepareJobShop(String instancePath, String outputPath)
      throws IOException {
    int[] input = readJobShopIntegers(Path.of(instancePath));
    if (input.length < 2 || input[0] <= 0 || input[1] <= 0) {
      throw new IllegalArgumentException("job-shop instance must start with positive job and machine counts");
    }
    int jobCount = input[0];
    int machineCount = input[1];
    int operationCount = jobCount * machineCount;
    if (input.length != 2 + 2 * operationCount) {
      throw new IllegalArgumentException("job-shop instance operation count does not match its header");
    }
    int horizon = 0;
    for (int index = 3; index < input.length; index += 2) {
      if (input[index] <= 0) {
        throw new IllegalArgumentException("job-shop durations must be positive");
      }
      horizon += input[index];
    }

    CpModel model = new CpModel();
    IntVar[][] starts = new IntVar[jobCount][machineCount];
    IntVar[][] ends = new IntVar[jobCount][machineCount];
    @SuppressWarnings("unchecked")
    List<IntervalVar>[] machineIntervals = new List[machineCount];
    for (int machine = 0; machine < machineCount; machine++) {
      machineIntervals[machine] = new ArrayList<>();
    }
    int[] startVariableIndices = new int[operationCount];
    int cursor = 2;
    for (int job = 0; job < jobCount; job++) {
      for (int operation = 0; operation < machineCount; operation++) {
        int machine = input[cursor++];
        int duration = input[cursor++];
        if (machine < 0 || machine >= machineCount) {
          throw new IllegalArgumentException("job-shop machine index outside declared range");
        }
        IntVar start = model.newIntVar(0, horizon, "start_" + job + "_" + operation);
        IntVar end = model.newIntVar(0, horizon, "end_" + job + "_" + operation);
        starts[job][operation] = start;
        ends[job][operation] = end;
        startVariableIndices[job * machineCount + operation] = start.getIndex();
        machineIntervals[machine].add(
            model.newIntervalVar(
                start,
                model.newConstant(duration),
                end,
                "interval_" + job + "_" + operation));
        if (operation > 0) {
          model.addGreaterOrEqual(start, ends[job][operation - 1]);
        }
      }
    }
    for (List<IntervalVar> intervals : machineIntervals) {
      model.addNoOverlap(intervals);
    }
    IntVar makespan = model.newIntVar(0, horizon, "makespan");
    IntVar[] finalOperations = new IntVar[jobCount];
    for (int job = 0; job < jobCount; job++) {
      finalOperations[job] = ends[job][machineCount - 1];
    }
    model.addMaxEquality(makespan, finalOperations);
    model.minimize(makespan);
    Files.write(Path.of(outputPath), model.model().toByteArray());
    System.out.println(
        "{\\\"job_count\\\":" + jobCount
            + ",\\\"machine_count\\\":" + machineCount
            + ",\\\"start_variable_indices\\\":"
            + jsonIntArray(startVariableIndices)
            + "}");
  }

  public static void main(String[] args) throws IOException {
    if (args.length == 3 && args[0].equals("prepare-jssp")) {
      Loader.loadNativeLibraries();
      prepareJobShop(args[1], args[2]);
      return;
    }
    if (args.length != 4) {
      throw new IllegalArgumentException(
          "expected model_proto_path solver_seed budget_sec threads");
    }
    long solverSeed = Long.parseLong(args[1]);
    double budgetSec = Double.parseDouble(args[2]);
    int threads = Integer.parseInt(args[3]);
    if (!Double.isFinite(budgetSec) || budgetSec <= 0 || threads != 8) {
      throw new IllegalArgumentException("parallel CP-SAT requires a positive budget and eight threads");
    }

    Loader.loadNativeLibraries();
    CpModelProto modelProto = CpModelProto.parseFrom(Files.readAllBytes(Path.of(args[0])));
    CpModel model = new CpModel();
    model.getBuilder().mergeFrom(modelProto);

    CpSolver solver = new CpSolver();
    solver.getParameters().setMaxTimeInSeconds(budgetSec);
    solver.getParameters().setNumWorkers(threads);
    solver.getParameters().setRandomSeed(Math.toIntExact(solverSeed));
    CpSolverStatus status = solver.solve(model);
    CpSolverResponse response = solver.response();
    boolean hasSolution = status == CpSolverStatus.OPTIMAL || status == CpSolverStatus.FEASIBLE;
    String objective = hasSolution && Double.isFinite(response.getObjectiveValue())
        ? Double.toString(response.getObjectiveValue())
        : "null";
    String termination = status == CpSolverStatus.OPTIMAL
        ? "optimal"
        : status == CpSolverStatus.FEASIBLE || status == CpSolverStatus.UNKNOWN
            ? "time_limit"
            : "other";
    String solution = hasSolution
        ? "{\\\"values\\\":" + jsonLongArray(response) + "}"
        : "{}";
    System.out.println(
        "{\\\"valid\\\":" + hasSolution
            + ",\\\"has_solution\\\":" + hasSolution
            + ",\\\"objective\\\":" + objective
            + ",\\\"solver_status\\\":\\\"" + status.name() + "\\\""
            + ",\\\"solver_termination\\\":\\\"" + termination + "\\\""
            + ",\\\"solver_runtime_sec\\\":" + solver.wallTime()
            + ",\\\"wall_time_sec\\\":" + solver.wallTime()
            + ",\\\"nodes\\\":" + response.getNumBranches()
            + ",\\\"solution\\\":" + solution
            + "}");
  }
}
"""


def compile_adapter(repository: Path, solver: str) -> None:
    source = Path(__file__).with_name("ExactSolverRunner.java")
    output = _adapter_output(repository, solver)
    output.mkdir(parents=True, exist_ok=True)
    if solver == "ortools_cp_sat":
        source = output / "OrToolsCpSatRunner.java"
        source.write_text(_ORTOOLS_CP_SAT_RUNNER_SOURCE)
        subprocess.run(
            [
                "javac",
                "-cp",
                _ortools_classpath(repository, output),
                "-d",
                str(output),
                str(source),
            ],
            cwd=repository,
            check=True,
        )
        return
    if solver == "ortools_model_build":
        source = output / "ModelBuildRunner.java"
        source.write_text((Path(__file__).with_name("ModelBuildRunner.java")).read_text())
        subprocess.run(
            ["javac", "-cp", _ortools_classpath(repository, output), "-d", str(output), str(source)],
            cwd=repository,
            check=True,
        )
        return
    subprocess.run(
        [
            "javac",
            "-cp",
            str(_choco_assembly(repository)),
            "-d",
            str(output),
            str(source),
        ],
        cwd=repository,
        check=True,
    )


def run_choco(repository: Path) -> None:
    request = json.load(sys.stdin)
    if request.get("threads") != 1:
        raise ValueError("Choco exact runner requires one thread")
    classpath = os.pathsep.join(
        [
            str(_adapter_output(repository, "choco")),
            str(_choco_assembly(repository)),
        ]
    )
    completed = subprocess.run(
        [
            "java",
            "-cp",
            classpath,
            "org.pitbench.adapters.ExactSolverRunner",
            str(request["instance_path"]),
            str(request["solver_seed"]),
            str(request["budget_sec"]),
            str(request["threads"]),
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode:
        raise RuntimeError(
            completed.stderr.strip()
            or f"Choco exact runner exited with {completed.returncode}"
        )
    response = json.loads(completed.stdout)
    print(json.dumps(response, allow_nan=False))


def run_ortools_cp_sat(repository: Path) -> None:
    request = json.load(sys.stdin)
    if request.get("threads") != 8:
        raise ValueError("parallel CP-SAT runner requires eight threads")
    output = _adapter_output(repository, "ortools_cp_sat")
    completed = subprocess.run(
        [
            "java",
            "-cp",
            _ortools_classpath(repository, output),
            "org.pitbench.adapters.OrToolsCpSatRunner",
            str(request["instance_path"]),
            str(request["solver_seed"]),
            str(request["budget_sec"]),
            str(request["threads"]),
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode:
        raise RuntimeError(
            completed.stderr.strip()
            or f"OR-Tools CP-SAT runner exited with {completed.returncode}"
        )
    response = json.loads(completed.stdout)
    print(json.dumps(response, allow_nan=False))


def run_ortools_model_build(repository: Path) -> None:
    request = json.load(sys.stdin)
    output = _adapter_output(repository, "ortools_model_build")
    completed = subprocess.run(
        ["java", "-cp", _ortools_classpath(repository, output), "org.pitbench.adapters.ModelBuildRunner", str(request["instance_path"]), str(request["solver_seed"]), str(request["budget_sec"]), str(request["threads"])],
        cwd=repository, check=False, capture_output=True, text=True,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or "OR-Tools model build runner failed")
    print(json.dumps(json.loads(completed.stdout), allow_nan=False))


def prepare_ortools_cp_sat_job_shop(
    repository: Path,
    instance: Path,
    output: Path,
) -> dict:
    adapter_output = _adapter_output(repository, "ortools_cp_sat")
    completed = subprocess.run(
        [
            "java",
            "-cp",
            _ortools_classpath(repository, adapter_output),
            "org.pitbench.adapters.OrToolsCpSatRunner",
            "prepare-jssp",
            str(instance),
            str(output),
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["compile", "run", "prepare-jssp"],
    )
    parser.add_argument("--solver", choices=["choco", "ortools_cp_sat", "ortools_model_build"], required=True)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--instance", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--budget", type=float)
    args = parser.parse_args()
    repository = args.repository.resolve()
    if args.command == "compile":
        compile_adapter(repository, args.solver)
    elif args.command == "prepare-jssp":
        if (
            args.solver != "ortools_cp_sat"
            or args.instance is None
            or args.output is None
        ):
            raise ValueError(
                "prepare-jssp requires an OR-Tools instance and output path"
            )
        print(
            json.dumps(
                prepare_ortools_cp_sat_job_shop(
                    repository,
                    args.instance.resolve(),
                    args.output.resolve(),
                )
            )
        )
    elif args.solver == "choco":
        run_choco(repository)
    elif args.solver == "ortools_model_build":
        run_ortools_model_build(repository)
    else:
        run_ortools_cp_sat(repository)


if __name__ == "__main__":
    main()
