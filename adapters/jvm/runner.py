from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
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


def _adapter_source_sha256(solver: str) -> str:
    source = Path(__file__).read_bytes()
    if solver == "choco":
        source += Path(__file__).with_name("ExactSolverRunner.java").read_bytes()
    elif solver != "ortools_cp_sat":
        raise ValueError(f"unsupported JVM adapter: {solver}")
    return hashlib.sha256(source).hexdigest()


def _verify_compiled_adapter(repository: Path, solver: str) -> None:
    output = _adapter_output(repository, solver)
    class_name = "ExactSolverRunner" if solver == "choco" else "OrToolsCpSatRunner"
    class_file = output / "org" / "pitbench" / "adapters" / f"{class_name}.class"
    if not class_file.is_file():
        raise RuntimeError(
            f"compiled {solver} adapter is unavailable; rebuild the image"
        )
    fingerprint = output / "adapter-source.sha256"
    if (
        not fingerprint.is_file()
        or fingerprint.read_text().strip() != _adapter_source_sha256(solver)
    ):
        raise RuntimeError(
            f"compiled {solver} adapter differs from source; rebuild the image"
        )


def _repository_commit(repository: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    return commit if completed.returncode == 0 and commit else None


def _solver_identity(repository: Path, solver: str) -> dict:
    if solver == "choco":
        assembly = _choco_assembly(repository)
        match = re.search(
            r"choco-solver-([0-9][^-]+)-jar-with-dependencies", assembly.name
        )
        if match is None:
            raise RuntimeError(f"cannot determine Choco version from {assembly.name}")
        version = match.group(1)
    elif solver == "ortools_cp_sat":
        artifact = _ortools_java_artifact(repository, "ortools-java")
        match = re.search(r"-(\d+\.\d+)(?:\.\d+)?$", artifact.stem)
        if match is None:
            raise RuntimeError(
                f"cannot determine OR-Tools version from {artifact.name}"
            )
        version = match.group(1)
    else:
        raise ValueError(f"unsupported external identity probe: {solver}")
    identity = {"version": version}
    source_commit = _repository_commit(repository)
    if source_commit is not None:
        identity["source_commit"] = source_commit
    return identity


_ORTOOLS_CP_SAT_RUNNER_SOURCE = """\
package org.pitbench.adapters;

import com.google.ortools.Loader;
import com.google.ortools.sat.CpModel;
import com.google.ortools.sat.CpModelProto;
import com.google.ortools.sat.ConstraintProto;
import com.google.ortools.sat.CpSolver;
import com.google.ortools.sat.CpSolverResponse;
import com.google.ortools.sat.CpSolverStatus;
import com.google.ortools.sat.IntVar;
import com.google.ortools.sat.IntervalVar;
import com.google.ortools.sat.SatParameters;
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

  private static boolean booleanField(String input, String name, boolean defaultValue) {
    java.util.regex.Matcher matcher = java.util.regex.Pattern.compile(
        "\\\"" + name + "\\\"\\\\s*:\\\\s*(true|false)").matcher(input);
    return matcher.find() ? Boolean.parseBoolean(matcher.group(1)) : defaultValue;
  }

  private static int integerField(String input, String name, int defaultValue) {
    java.util.regex.Matcher matcher = java.util.regex.Pattern.compile(
        "\\\"" + name + "\\\"\\\\s*:\\\\s*(-?\\\\d+)").matcher(input);
    return matcher.find() ? Integer.parseInt(matcher.group(1)) : defaultValue;
  }

  private static String stringField(String input, String name, String defaultValue) {
    java.util.regex.Matcher matcher = java.util.regex.Pattern.compile(
        "\\\"" + name + "\\\"\\\\s*:\\\\s*\\\"([^\\\"]+)\\\"").matcher(input);
    return matcher.find() ? matcher.group(1) : defaultValue;
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

  private static String buildJobShop(int[] input, String outputPath)
      throws IOException {
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
    return (
        "{\\\"job_count\\\":" + jobCount
            + ",\\\"machine_count\\\":" + machineCount
            + ",\\\"start_variable_indices\\\":"
            + jsonIntArray(startVariableIndices)
            + "}");
  }

  private static void prepareJobShop(String instancePath, String outputPath)
      throws IOException {
    System.out.println(buildJobShop(readJobShopIntegers(Path.of(instancePath)), outputPath));
  }

  private static int[] decodeJobShop(CpModelProto model) {
    java.util.regex.Pattern startName = java.util.regex.Pattern.compile("start_([0-9]+)_([0-9]+)");
    java.util.regex.Pattern intervalName = java.util.regex.Pattern.compile("interval_([0-9]+)_([0-9]+)");
    int jobCount = 0;
    int machineCount = 0;
    for (int index = 0; index < model.getVariablesCount(); index++) {
      java.util.regex.Matcher match = startName.matcher(model.getVariables(index).getName());
      if (match.matches()) {
        jobCount = Math.max(jobCount, Integer.parseInt(match.group(1)) + 1);
      }
    }
    for (int index = 0; index < model.getConstraintsCount(); index++) {
      if (model.getConstraints(index).hasNoOverlap()) {
        machineCount++;
      }
    }
    if (jobCount < 1 || machineCount < 1) {
      throw new IllegalArgumentException("unrecognized job-shop model");
    }
    int[][] machines = new int[jobCount][machineCount];
    int[][] durations = new int[jobCount][machineCount];
    for (int[] row : machines) {
      java.util.Arrays.fill(row, -1);
    }
    int machine = 0;
    for (int index = 0; index < model.getConstraintsCount(); index++) {
      ConstraintProto noOverlap = model.getConstraints(index);
      if (!noOverlap.hasNoOverlap()) {
        continue;
      }
      for (int intervalIndex : noOverlap.getNoOverlap().getIntervalsList()) {
        ConstraintProto interval = model.getConstraints(intervalIndex);
        java.util.regex.Matcher match = intervalName.matcher(interval.getName());
        if (!interval.hasInterval() || !match.matches()) {
          throw new IllegalArgumentException("unrecognized job-shop interval");
        }
        int job = Integer.parseInt(match.group(1));
        int operation = Integer.parseInt(match.group(2));
        if (job >= jobCount || operation >= machineCount
            || machines[job][operation] != -1) {
          throw new IllegalArgumentException("invalid job-shop interval mapping: " + interval.getName());
        }
        com.google.ortools.sat.LinearExpressionProto size = interval.getInterval().getSize();
        long duration;
        if (size.getVarsCount() == 0) {
          duration = size.getOffset();
        } else if (size.getVarsCount() == 1 && size.getCoeffs(0) == 1) {
          com.google.ortools.sat.IntegerVariableProto constant =
              model.getVariables(size.getVars(0));
          if (constant.getDomainCount() != 2 || constant.getDomain(0) != constant.getDomain(1)) {
            throw new IllegalArgumentException("job-shop interval size is not constant");
          }
          duration = constant.getDomain(0) + size.getOffset();
        } else {
          throw new IllegalArgumentException("unsupported job-shop interval size");
        }
        if (duration <= 0 || duration > Integer.MAX_VALUE) {
          throw new IllegalArgumentException("invalid job-shop duration");
        }
        machines[job][operation] = machine;
        durations[job][operation] = (int) duration;
      }
      machine++;
    }
    int[] input = new int[2 + 2 * jobCount * machineCount];
    input[0] = jobCount;
    input[1] = machineCount;
    int cursor = 2;
    for (int job = 0; job < jobCount; job++) {
      for (int operation = 0; operation < machineCount; operation++) {
        if (machines[job][operation] < 0) {
          throw new IllegalArgumentException("job-shop operation is missing");
        }
        input[cursor++] = machines[job][operation];
        input[cursor++] = durations[job][operation];
      }
    }
    return input;
  }

  private static String inspectJobShop(Path path) throws IOException {
    byte[] original = Files.readAllBytes(path);
    CpModelProto model = CpModelProto.parseFrom(original);
    int[] input = decodeJobShop(model);
    Path rebuilt = Files.createTempFile("pitbench-jssp-", ".pb");
    try {
      buildJobShop(input, rebuilt.toString());
      if (!java.util.Arrays.equals(original, Files.readAllBytes(rebuilt))) {
        throw new IllegalArgumentException("job-shop model differs from the pinned builder");
      }
    } finally {
      Files.deleteIfExists(rebuilt);
    }
    int jobCount = input[0];
    int machineCount = input[1];
    int[][] starts = new int[jobCount][machineCount];
    for (int[] row : starts) {
      java.util.Arrays.fill(row, -1);
    }
    java.util.regex.Pattern startName = java.util.regex.Pattern.compile("start_([0-9]+)_([0-9]+)");
    for (int index = 0; index < model.getVariablesCount(); index++) {
      java.util.regex.Matcher match = startName.matcher(model.getVariables(index).getName());
      if (match.matches()) {
        int job = Integer.parseInt(match.group(1));
        int operation = Integer.parseInt(match.group(2));
        if (job >= jobCount || operation >= machineCount || starts[job][operation] != -1) {
          throw new IllegalArgumentException("invalid job-shop start variable mapping");
        }
        starts[job][operation] = index;
      }
    }
    StringBuilder json = new StringBuilder("{\\\"jobs\\\":[");
    int cursor = 2;
    for (int job = 0; job < jobCount; job++) {
      if (job > 0) {
        json.append(',');
      }
      json.append('[');
      for (int operation = 0; operation < machineCount; operation++) {
        if (operation > 0) {
          json.append(',');
        }
        if (starts[job][operation] < 0) {
          throw new IllegalArgumentException("job-shop start variable is missing");
        }
        json.append('[').append(input[cursor++]).append(',')
            .append(input[cursor++]).append(']');
      }
      json.append(']');
    }
    json.append("],\\\"start_variable_indices\\\":[");
    boolean first = true;
    for (int[] row : starts) {
      for (int index : row) {
        if (!first) {
          json.append(',');
        }
        json.append(index);
        first = false;
      }
    }
    return json.append("]}").toString();
  }

  private static void relabelJobShop(Path source, Path output, String mappingText)
      throws IOException {
    inspectJobShop(source);
    int[] input = decodeJobShop(CpModelProto.parseFrom(Files.readAllBytes(source)));
    String[] tokens = mappingText.split(",");
    int jobCount = input[0];
    int machineCount = input[1];
    if (tokens.length != jobCount) {
      throw new IllegalArgumentException("job permutation has the wrong length");
    }
    boolean[] seen = new boolean[jobCount];
    int[] reordered = input.clone();
    for (int newJob = 0; newJob < jobCount; newJob++) {
      int oldJob = Integer.parseInt(tokens[newJob]);
      if (oldJob < 0 || oldJob >= jobCount || seen[oldJob]) {
        throw new IllegalArgumentException("job permutation is not a bijection");
      }
      seen[oldJob] = true;
      System.arraycopy(input, 2 + 2 * machineCount * oldJob,
          reordered, 2 + 2 * machineCount * newJob, 2 * machineCount);
    }
    buildJobShop(reordered, output.toString());
    System.out.println(inspectJobShop(output));
  }

  private static void proveBinPacking(String capacityText, String weightsText,
      String binCountText, String budgetText) {
    int capacity = Integer.parseInt(capacityText);
    int binCount = Integer.parseInt(binCountText);
    double budget = Double.parseDouble(budgetText);
    String[] tokens = weightsText.split(",");
    if (capacity <= 0 || binCount <= 0 || !Double.isFinite(budget) || budget <= 0) {
      throw new IllegalArgumentException("invalid bin-packing proof configuration");
    }
    int[] weights = new int[tokens.length];
    for (int item = 0; item < tokens.length; item++) {
      weights[item] = Integer.parseInt(tokens[item]);
      if (weights[item] <= 0 || weights[item] > capacity) {
        throw new IllegalArgumentException("item weight is outside bin capacity");
      }
    }
    Loader.loadNativeLibraries();
    CpModelProto.Builder proto = CpModelProto.newBuilder();
    int[][] assignments = new int[weights.length][binCount];
    int[] usedBins = new int[binCount];
    for (int bin = 0; bin < binCount; bin++) {
      usedBins[bin] = proto.getVariablesCount();
      proto.addVariablesBuilder().addDomain(0).addDomain(1)
          .setName("used_" + bin);
    }
    for (int item = 0; item < weights.length; item++) {
      for (int bin = 0; bin < binCount; bin++) {
        assignments[item][bin] = proto.getVariablesCount();
        proto.addVariablesBuilder().addDomain(0).addDomain(1)
            .setName("item_" + item + "_bin_" + bin);
      }
      ConstraintProto.Builder once = proto.addConstraintsBuilder();
      for (int bin = 0; bin < binCount; bin++) {
        once.getLinearBuilder().addVars(assignments[item][bin]).addCoeffs(1);
      }
      once.getLinearBuilder().addDomain(1).addDomain(1);
    }
    Integer[] descendingItems = new Integer[weights.length];
    for (int item = 0; item < weights.length; item++) {
      descendingItems[item] = item;
    }
    java.util.Arrays.sort(descendingItems, (left, right) -> {
      int byWeight = Integer.compare(weights[right], weights[left]);
      return byWeight != 0 ? byWeight : Integer.compare(left, right);
    });
    for (int rank = 0; rank < descendingItems.length; rank++) {
      for (int bin = rank + 1; bin < binCount; bin++) {
        proto.addConstraintsBuilder().getLinearBuilder()
            .addVars(assignments[descendingItems[rank]][bin]).addCoeffs(1)
            .addDomain(0).addDomain(0);
      }
    }
    for (int bin = 0; bin < binCount; bin++) {
      ConstraintProto.Builder load = proto.addConstraintsBuilder();
      for (int item = 0; item < weights.length; item++) {
        load.getLinearBuilder().addVars(assignments[item][bin])
            .addCoeffs(weights[item]);
      }
      load.getLinearBuilder().addVars(usedBins[bin]).addCoeffs(-capacity)
          .addDomain(-capacity).addDomain(0);
      proto.getObjectiveBuilder().addVars(usedBins[bin]).addCoeffs(1);
      if (bin > 0) {
        proto.addConstraintsBuilder().getLinearBuilder()
            .addVars(usedBins[bin - 1]).addCoeffs(1)
            .addVars(usedBins[bin]).addCoeffs(-1)
            .addDomain(0).addDomain(1);
      }
    }
    proto.getObjectiveBuilder().setScalingFactor(1.0);
    CpModel model = new CpModel();
    model.getBuilder().mergeFrom(proto.build());
    CpSolver solver = new CpSolver();
    solver.getParameters().setMaxTimeInSeconds(budget);
    solver.getParameters().setNumWorkers(1);
    solver.getParameters().setRandomSeed(0);
    CpSolverStatus status = solver.solve(model);
    CpSolverResponse response = solver.response();
    System.out.println("STATUS " + status.name());
    System.out.println("OBJECTIVE " + response.getObjectiveValue());
    System.out.println("BOUND " + response.getBestObjectiveBound());
    if (status == CpSolverStatus.OPTIMAL || status == CpSolverStatus.FEASIBLE) {
      for (int bin = 0; bin < binCount; bin++) {
        if (response.getSolution(usedBins[bin]) == 0) {
          continue;
        }
        StringBuilder items = new StringBuilder("BIN " + bin);
        for (int item = 0; item < weights.length; item++) {
          if (response.getSolution(assignments[item][bin]) == 1) {
            items.append(' ').append(item);
          }
        }
        System.out.println(items);
      }
    }
  }

  public static void main(String[] args) throws IOException {
    if (args.length == 2 && args[0].equals("inspect-jssp")) {
      Loader.loadNativeLibraries();
      System.out.println(inspectJobShop(Path.of(args[1])));
      return;
    }
    if (args.length == 4 && args[0].equals("relabel-jssp")) {
      Loader.loadNativeLibraries();
      relabelJobShop(Path.of(args[1]), Path.of(args[2]), args[3]);
      return;
    }
    if (args.length == 5 && args[0].equals("prove-bin-packing")) {
      proveBinPacking(args[1], args[2], args[3], args[4]);
      return;
    }
    if (args.length == 3 && args[0].equals("prepare-jssp")) {
      Loader.loadNativeLibraries();
      prepareJobShop(args[1], args[2]);
      return;
    }
    if (args.length < 4 || args.length > 5) {
      throw new IllegalArgumentException(
          "expected model_proto_path solver_seed budget_sec threads [parameters_json]");
    }
    long solverSeed = Long.parseLong(args[1]);
    double budgetSec = Double.parseDouble(args[2]);
    int threads = Integer.parseInt(args[3]);
    if (!Double.isFinite(budgetSec) || budgetSec <= 0 || threads != 1) {
      throw new IllegalArgumentException("CP-SAT requires a positive budget and one solver thread");
    }
    String parameters = args.length == 5 ? args[4] : "{}";
    boolean cpModelPresolve = booleanField(parameters, "cp_model_presolve", true);
    int cpModelProbingLevel = integerField(parameters, "cp_model_probing_level", 2);
    int linearizationLevel = integerField(parameters, "linearization_level", 1);
    int symmetryLevel = integerField(parameters, "symmetry_level", 2);
    String searchBranching = stringField(parameters, "search_branching", "automatic_search");
    SatParameters.SearchBranching branching;
    switch (searchBranching) {
      case "automatic_search":
        branching = SatParameters.SearchBranching.AUTOMATIC_SEARCH;
        break;
      case "lp_search":
        branching = SatParameters.SearchBranching.LP_SEARCH;
        break;
      case "pseudo_cost_search":
        branching = SatParameters.SearchBranching.PSEUDO_COST_SEARCH;
        break;
      case "portfolio_search":
        branching = SatParameters.SearchBranching.PORTFOLIO_SEARCH;
        break;
      default:
        throw new IllegalArgumentException("unsupported search_branching: " + searchBranching);
    }

    Loader.loadNativeLibraries();
    CpModelProto modelProto = CpModelProto.parseFrom(Files.readAllBytes(Path.of(args[0])));
    CpModel model = new CpModel();
    model.getBuilder().mergeFrom(modelProto);

    CpSolver solver = new CpSolver();
    solver.getParameters().setMaxTimeInSeconds(budgetSec);
    solver.getParameters().setNumWorkers(threads);
    solver.getParameters().setRandomSeed(Math.toIntExact(solverSeed));
    solver.getParameters().setCpModelPresolve(cpModelPresolve);
    solver.getParameters().setCpModelProbingLevel(cpModelProbingLevel);
    solver.getParameters().setLinearizationLevel(linearizationLevel);
    solver.getParameters().setSymmetryLevel(symmetryLevel);
    solver.getParameters().setSearchBranching(branching);
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
            + ",\\\"effective_parameters\\\":{"
            + "\\\"cp_model_presolve\\\":" + solver.getParameters().getCpModelPresolve()
            + ",\\\"cp_model_probing_level\\\":" + solver.getParameters().getCpModelProbingLevel()
            + ",\\\"linearization_level\\\":" + solver.getParameters().getLinearizationLevel()
            + ",\\\"symmetry_level\\\":" + solver.getParameters().getSymmetryLevel()
            + ",\\\"search_branching\\\":\\\"" + searchBranching + "\\\"}"
            + ",\\\"solution\\\":" + solution
            + "}");
  }
}
"""


_ORTOOLS_CP_SAT_ASSIGNMENT_VERIFIER_SOURCE = """\
package org.pitbench.adapters;

import com.google.ortools.Loader;
import com.google.ortools.sat.ConstraintProto;
import com.google.ortools.sat.CpModel;
import com.google.ortools.sat.CpModelProto;
import com.google.ortools.sat.CpSolver;
import com.google.ortools.sat.CpSolverResponse;
import com.google.ortools.sat.CpSolverStatus;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class CpSatAssignmentVerifier {
  private CpSatAssignmentVerifier() {}

  private static long[] readValues(Path solutionPath) throws IOException {
    String payload = Files.readString(solutionPath);
    Matcher matcher = Pattern.compile(
        "\\\"values\\\"\\\\s*:\\\\s*\\\\[([^]]*)\\\\]", Pattern.DOTALL).matcher(payload);
    if (!matcher.find()) {
      throw new IllegalArgumentException("solution does not contain values");
    }
    String body = matcher.group(1).trim();
    if (body.isEmpty()) {
      return new long[0];
    }
    String[] tokens = body.split(",", -1);
    long[] values = new long[tokens.length];
    for (int index = 0; index < tokens.length; index++) {
      String token = tokens[index].trim();
      if (!token.matches("-?[0-9]+")) {
        throw new IllegalArgumentException("solution value is not an integer");
      }
      values[index] = Long.parseLong(token);
    }
    return values;
  }

  private static String escape(String value) {
    return value.replace("\\\\", "\\\\\\\\").replace("\\\"", "\\\\\\\"");
  }

  private static void emit(boolean feasible, Double objective, String detail) {
    String objectiveJson = objective == null ? "null" : Double.toString(objective);
    System.out.println(
        "{\\\"feasible\\\":" + feasible
            + ",\\\"objective\\\":" + objectiveJson
            + ",\\\"detail\\\":\\\"" + escape(detail) + "\\\"}");
  }

  public static void main(String[] args) {
    if (args.length != 2) {
      emit(false, null, "expected instance and solution paths");
      return;
    }
    try {
      Loader.loadNativeLibraries();
      CpModelProto modelProto = CpModelProto.parseFrom(Files.readAllBytes(Path.of(args[0])));
      long[] values = readValues(Path.of(args[1]));
      if (values.length != modelProto.getVariablesCount()) {
        emit(false, null, "solution value count does not match model variables");
        return;
      }
      CpModel model = new CpModel();
      model.getBuilder().mergeFrom(modelProto);
      for (int index = 0; index < values.length; index++) {
        ConstraintProto.Builder constraint = model.getBuilder().addConstraintsBuilder();
        constraint.getLinearBuilder()
            .addVars(index)
            .addCoeffs(1)
            .addDomain(values[index])
            .addDomain(values[index]);
      }
      CpSolver solver = new CpSolver();
      solver.getParameters().setNumWorkers(1);
      CpSolverStatus status = solver.solve(model);
      if (status != CpSolverStatus.OPTIMAL && status != CpSolverStatus.FEASIBLE) {
        emit(false, null, "assignment is infeasible in the fixed model");
        return;
      }
      CpSolverResponse response = solver.response();
      emit(true, response.getObjectiveValue(), "independent fixed-proto assignment verification passed");
    } catch (Exception error) {
      emit(false, null, error.getMessage() == null ? error.getClass().getName() : error.getMessage());
    }
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
        (output / "adapter-source.sha256").write_text(
            _adapter_source_sha256(solver) + "\n"
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
    (output / "adapter-source.sha256").write_text(_adapter_source_sha256(solver) + "\n")


def run_choco(repository: Path) -> None:
    request = json.load(sys.stdin)
    if request.get("probe"):
        _verify_compiled_adapter(repository, "choco")
        print(json.dumps(_solver_identity(repository, "choco")))
        return
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
            *(
                [json.dumps(request["parameters"], separators=(",", ":"))]
                if request.get("parameters") is not None
                else []
            ),
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
    if request.get("probe"):
        _verify_compiled_adapter(repository, "ortools_cp_sat")
        print(json.dumps(_solver_identity(repository, "ortools_cp_sat")))
        return
    if request.get("threads") != 1:
        raise ValueError("CP-SAT runner requires one solver thread")
    output = _adapter_output(repository, "ortools_cp_sat")
    native_library_path = repository / "build" / "lib"
    completed = subprocess.run(
        [
            "java",
            f"-Djava.library.path={native_library_path}",
            "-cp",
            _ortools_classpath(repository, output),
            "org.pitbench.adapters.OrToolsCpSatRunner",
            str(request["instance_path"]),
            str(request["solver_seed"]),
            str(request["budget_sec"]),
            str(request["threads"]),
            *(
                [json.dumps(request["parameters"], separators=(",", ":"))]
                if request.get("parameters") is not None
                else []
            ),
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


def prove_bin_packing(repository: Path, instance_path: Path, budget_sec: float) -> dict:
    instance = json.loads(instance_path.read_text())
    weights = instance.get("weights")
    capacity = instance.get("capacity")
    if (
        type(capacity) is not int
        or capacity <= 0
        or not isinstance(weights, list)
        or not weights
        or any(
            type(weight) is not int or not 0 < weight <= capacity for weight in weights
        )
        or budget_sec <= 0
    ):
        raise ValueError("invalid bin-packing proof input")

    loads: list[int] = []
    for item in sorted(range(len(weights)), key=lambda item: (-weights[item], item)):
        for index, load in enumerate(loads):
            if load + weights[item] <= capacity:
                loads[index] += weights[item]
                break
        else:
            loads.append(weights[item])

    adapter_output = _adapter_output(repository, "ortools_cp_sat")
    completed = subprocess.run(
        [
            "java",
            "-cp",
            _ortools_classpath(repository, adapter_output),
            "org.pitbench.adapters.OrToolsCpSatRunner",
            "prove-bin-packing",
            str(capacity),
            ",".join(map(str, weights)),
            str(len(loads)),
            str(budget_sec),
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    fields = {}
    bins = []
    for line in completed.stdout.splitlines():
        key, _, value = line.partition(" ")
        if key == "BIN":
            bins.append([int(item) for item in value.split()[1:]])
        elif key in {"STATUS", "OBJECTIVE", "BOUND", "WALL"}:
            fields[key] = value
    if "STATUS" not in fields:
        raise RuntimeError("bin-packing proof returned no solver status")
    status = fields["STATUS"]
    objective = float(fields["OBJECTIVE"])
    best_bound = float(fields["BOUND"])
    feasible = status in {"OPTIMAL", "FEASIBLE"}
    if feasible:
        items = [item for bin_items in bins for item in bin_items]
        if (
            len(items) != len(weights)
            or set(items) != set(range(len(weights)))
            or any(
                not bin_items or sum(weights[item] for item in bin_items) > capacity
                for bin_items in bins
            )
            or objective != len(bins)
        ):
            raise ValueError("CP-SAT reference packing failed independent verification")
    if status == "OPTIMAL" and best_bound != objective:
        raise ValueError("CP-SAT optimum and proof bound differ")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "instance_sha256": hashlib.sha256(instance_path.read_bytes()).hexdigest(),
        "solver_commit": revision,
        "adapter_source_sha256": _adapter_source_sha256("ortools_cp_sat"),
        "solver_threads": 1,
        "solver_seed": 0,
        "budget_sec": budget_sec,
        "status": status,
        "objective": int(objective) if feasible else None,
        "best_bound": best_bound,
        "bins": bins if feasible else None,
    }


def prepare_job_shop_relabelings(
    repository: Path,
    instances_dir: Path,
    output_dir: Path,
    count: int,
    generation_seed: int,
) -> dict:
    if count < 1:
        raise ValueError("relabeling count must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_output = _adapter_output(repository, "ortools_cp_sat")
    command = [
        "java",
        "-cp",
        _ortools_classpath(repository, adapter_output),
        "org.pitbench.adapters.OrToolsCpSatRunner",
    ]
    generator = random.Random(generation_seed)
    instances = {}
    for original_path in sorted(instances_dir.glob("*.pb")):
        inspected = subprocess.run(
            [*command, "inspect-jssp", str(original_path)],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
        original = json.loads(inspected.stdout)
        job_count = len(original["jobs"])
        identity = tuple(range(job_count))
        seen = {identity}
        variants = []
        while len(variants) < count:
            new_to_original = list(identity)
            generator.shuffle(new_to_original)
            signature = tuple(new_to_original)
            if signature in seen:
                continue
            seen.add(signature)
            transform_id = f"job_relabeling_{len(variants):02d}"
            transformed_path = output_dir / f"{original_path.stem}__{transform_id}.pb"
            completed = subprocess.run(
                [
                    *command,
                    "relabel-jssp",
                    str(original_path),
                    str(transformed_path),
                    ",".join(map(str, new_to_original)),
                ],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            )
            transformed = json.loads(completed.stdout)
            if transformed["jobs"] != [
                original["jobs"][index] for index in new_to_original
            ]:
                raise ValueError("relabeling changed job-shop operations")
            variants.append(
                {
                    "transform_id": transform_id,
                    "new_to_original": new_to_original,
                    "file": transformed_path.name,
                    "sha256": hashlib.sha256(transformed_path.read_bytes()).hexdigest(),
                    "start_variable_indices": transformed["start_variable_indices"],
                }
            )
        instances[original_path.stem] = {
            "original_sha256": hashlib.sha256(original_path.read_bytes()).hexdigest(),
            "jobs": original["jobs"],
            "start_variable_indices": original["start_variable_indices"],
            "variants": variants,
        }
    if not instances:
        raise ValueError("no job-shop model protos found")
    manifest = {
        "kind": "job_relabeling",
        "protocol_version": "0.0.2",
        "generation_seed": generation_seed,
        "relabelings_per_instance": count,
        "solver_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "adapter_source_sha256": _adapter_source_sha256("ortools_cp_sat"),
        "instances": instances,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "compile",
            "run",
            "prepare-jssp",
            "prove-bin-packing",
            "prepare-jssp-relabelings",
        ],
    )
    parser.add_argument("--solver", choices=["choco", "ortools_cp_sat"], required=True)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--instance", type=Path)
    parser.add_argument("--instances-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--budget", type=float)
    parser.add_argument("--count", type=int)
    parser.add_argument("--generation-seed", type=int)
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
    elif args.command == "prove-bin-packing":
        if (
            args.solver != "ortools_cp_sat"
            or args.instance is None
            or args.output is None
            or args.budget is None
        ):
            raise ValueError(
                "prove-bin-packing requires an OR-Tools instance, output, and budget"
            )
        proof = prove_bin_packing(repository, args.instance.resolve(), args.budget)
        args.output.write_text(json.dumps(proof, indent=2) + "\n")
        print(
            json.dumps(
                {key: proof[key] for key in ("status", "objective", "best_bound")}
            )
        )
    elif args.command == "prepare-jssp-relabelings":
        if (
            args.solver != "ortools_cp_sat"
            or args.instances_dir is None
            or args.output is None
            or args.count is None
            or args.generation_seed is None
        ):
            raise ValueError(
                "prepare-jssp-relabelings requires instances, output, count, and seed"
            )
        manifest = prepare_job_shop_relabelings(
            repository,
            args.instances_dir.resolve(),
            args.output.resolve(),
            args.count,
            args.generation_seed,
        )
        print(
            json.dumps({"instances": len(manifest["instances"]), "count": args.count})
        )
    elif args.solver == "choco":
        run_choco(repository)
    else:
        run_ortools_cp_sat(repository)


if __name__ == "__main__":
    main()
