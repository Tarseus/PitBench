"""Solver invocation and output adapters, selected by driver name."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from pitbench.solver_drivers.common import (
    ParameterRejected,
    ScenarioSession,
    SolverTermination,
    append_trajectory,
    failure_reason,
    finite,
    parser,
    process_resources,
    write_result,
    write_solution,
)
from pitbench.solver_drivers.external_runner import execute


class SolverDriver(ABC):
    name: str
    records_trajectory = True
    _drivers: ClassVar[dict[str, type[SolverDriver]]] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        name = getattr(cls, "name", None)
        if name:
            existing = cls._drivers.get(name)
            if existing is not None and existing is not cls:
                raise ValueError(f"duplicate solver driver: {name}")
            cls._drivers[name] = cls

    @staticmethod
    @abstractmethod
    def main(argv: list[str] | None = None) -> None: ...


class PyVRPDriver(SolverDriver):
    """Invoke the pyvrp backend with its existing configuration contract."""

    name = "pyvrp"

    @staticmethod
    def parameters(values: dict):
        """Construct nested solver parameters without mutating default objects."""
        from pyvrp import SolveParams

        defaults = SolveParams()
        groups = {}
        for name, value in values.items():
            group, field = name.split(".", 1)
            groups.setdefault(group, {})[field] = value
        return SolveParams(
            **{
                group: type(getattr(defaults, group))(**fields)
                for group, fields in groups.items()
            }
        )

    @staticmethod
    def parameter_values(params) -> dict:
        """Record the effective public scalar settings, including defaults."""
        return {
            group: {
                name: value
                for name in dir(getattr(params, group))
                if not name.startswith("_")
                and isinstance(
                    value := getattr(getattr(params, group), name),
                    (bool, int, float, str),
                )
            }
            for group in ("ils", "penalty", "neighbourhood", "perturbation")
        }

    @staticmethod
    def _route_visits(route: object) -> list[int]:
        """Return zero-based CVRP node ids across PyVRP route API generations."""

        legacy_visits = getattr(route, "visits", None)
        if callable(legacy_visits):
            return list(map(int, legacy_visits()))

        # PyVRP v0.14 represents routes as scheduled activities. For the
        # single-depot VRPLIB files emitted below, client activity index zero maps
        # to normalized CVRP node one.
        return [
            int(activity.idx) + 1
            for activity in route  # type: ignore[union-attr]
            if activity.is_client()
        ]

    @staticmethod
    def _statistics_rows(stats: object):
        """Yield runtime, feasibility, and incumbent across PyVRP statistics APIs."""

        runtimes = stats.runtimes  # type: ignore[attr-defined]
        legacy_feasible = getattr(stats, "feas_stats", None)
        if legacy_feasible is not None:
            for runtime, datum in zip(runtimes, legacy_feasible, strict=True):
                yield runtime, datum.size > 0, datum.best_cost
            return

        for runtime, datum in zip(runtimes, stats, strict=True):
            yield runtime, datum.best_feas, datum.best_cost

    @staticmethod
    def _vrplib(instance: dict, path: Path) -> None:
        coordinates = instance["coordinates"]
        demands = instance["demands"]
        distance_metric = instance.get("distance_metric", "EUC_2D")
        if distance_metric != "EUC_2D":
            raise ValueError(
                "PyVRP driver only supports normalized instances with EUC_2D "
                f"distance semantics, got {distance_metric!r}"
            )
        lines = [
            f"NAME : {instance.get('name', path.stem)}",
            "TYPE : CVRP",
            f"DIMENSION : {len(coordinates)}",
            f"EDGE_WEIGHT_TYPE : {distance_metric}",
            f"CAPACITY : {instance['capacity']}",
            "NODE_COORD_SECTION",
        ]
        lines.extend(
            f"{index + 1} {coordinate[0]} {coordinate[1]}"
            for index, coordinate in enumerate(coordinates)
        )
        lines.append("DEMAND_SECTION")
        lines.extend(f"{index + 1} {demand}" for index, demand in enumerate(demands))
        lines.extend(["DEPOT_SECTION", "1", "-1", "EOF"])
        path.write_text("\n".join(lines) + "\n")

    @staticmethod
    def main(argv: list[str] | None = None) -> None:
        arguments = parser()
        arguments.add_argument("--parameters", type=Path)
        args = arguments.parse_args(argv)
        started = time.perf_counter()
        instance = json.loads(args.instance.read_text())
        args.trajectory.parent.mkdir(parents=True, exist_ok=True)
        args.trajectory.write_text("")
        try:
            try:
                import tomli  # noqa: F401
            except ModuleNotFoundError:
                import tomllib

                sys.modules["tomli"] = tomllib

            import pyvrp
            from pyvrp.stop import MaxRuntime

            model = getattr(pyvrp, "Model", None)
            options = {}
            if args.parameters is not None:
                if model is None:
                    raise ParameterRejected(
                        "this PyVRP source does not expose the SolveParams API"
                    )
                params = PyVRPDriver.parameters(json.loads(args.parameters.read_text()))
                options["params"] = params
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.with_suffix(".parameters.json").write_text(
                    json.dumps(PyVRPDriver.parameter_values(params), indent=2) + "\n"
                )
            with tempfile.TemporaryDirectory(prefix="pitbench-pyvrp-") as temporary:
                vrp = Path(temporary) / "instance.vrp"
                PyVRPDriver._vrplib(instance, vrp)
                if model is None:
                    from pyvrp.cli import solve as classic_solve

                    result = classic_solve(
                        data_loc=str(vrp),
                        instance_format="vrplib",
                        round_func="round",
                        seed=args.seed,
                        max_runtime=args.budget,
                        max_iterations=None,
                        stats_dir=temporary,
                        sol_dir=None,
                    )
                else:
                    data = pyvrp.read(vrp, round_func="round")
                    solve_options = {
                        "stop": MaxRuntime(args.budget),
                        "seed": args.seed,
                        **options,
                    }
                    if "display" in inspect.signature(model.solve).parameters:
                        solve_options["display"] = False
                    result = model.from_data(data).solve(**solve_options)
                resources = process_resources()
            if not result.best.is_feasible():
                write_result(
                    args.output,
                    started=started,
                    valid=False,
                    has_solution=False,
                    solver_status="Budget stop",
                    solver_termination=SolverTermination.TIME_LIMIT,
                    solver_runtime_sec=result.runtime,
                    iterations=result.num_iterations,
                    **resources,
                )
                return
            routes_method = getattr(result.best, "routes", None)
            solver_routes = (
                routes_method() if callable(routes_method) else result.best.get_routes()
            )
            routes = [PyVRPDriver._route_visits(route) for route in solver_routes]
            objective = float(result.cost())
            write_solution(args.output, {"routes": routes})
            elapsed = 0.0
            incumbent = float("inf")
            for runtime, feasible, best_cost in PyVRPDriver._statistics_rows(
                result.stats
            ):
                elapsed += runtime
                if feasible and best_cost < incumbent:
                    incumbent = float(best_cost)
                    append_trajectory(
                        args.trajectory,
                        {"time_sec": elapsed, "objective": incumbent},
                    )
            if incumbent != objective:
                append_trajectory(
                    args.trajectory,
                    {"time_sec": result.runtime, "objective": objective},
                )
            write_result(
                args.output,
                started=started,
                valid=True,
                has_solution=True,
                solver_status="Budget stop",
                solver_termination=SolverTermination.TIME_LIMIT,
                objective=objective,
                iterations=result.num_iterations,
                solver_runtime_sec=result.runtime,
                **resources,
            )
        except Exception as exc:
            write_result(
                args.output,
                started=started,
                valid=False,
                error=str(exc),
                failure_reason=failure_reason(exc),
            )
            raise


class VroomDriver(SolverDriver):
    """Invoke the vroom backend with its existing configuration contract."""

    name = "vroom"

    @staticmethod
    def _uses_legacy_cli(solver: str) -> bool:
        """Detect old VROOM CLIs where -l selects libosrm, not a time limit."""
        try:
            completed = subprocess.run(
                [solver, "-h"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except subprocess.TimeoutExpired:
            return False
        help_text = completed.stdout + "\n" + completed.stderr
        return "use libosrm rather than osrm-routed" in help_text

    @staticmethod
    def _route_jobs(response: dict) -> list[list[int]]:
        """Read route job IDs from both legacy and current VROOM JSON output."""
        return [
            [
                int(step["id"] if "id" in step else step["job"])
                for step in route.get("steps", [])
                if step.get("type") == "job"
            ]
            for route in response.get("routes", [])
        ]

    @staticmethod
    def _request(instance: dict) -> dict:
        from pitbench.problem_families.public_verification import edge_cost

        coordinates = instance["coordinates"]
        metric = instance.get("distance_metric")
        scale = 1 if metric == "EUC_2D" else 1000
        matrix = [
            [
                round(edge_cost(coordinates, first, second, metric) * scale)
                for second in range(len(coordinates))
            ]
            for first in range(len(coordinates))
        ]
        total = sum(instance["demands"])
        capacity = int(instance["capacity"])
        vehicles = max(1, math.ceil(total / capacity) + 1)
        return {
            "matrix": matrix,
            "jobs": [
                {"id": index, "location_index": index, "amount": [int(demand)]}
                for index, demand in enumerate(instance["demands"])
                if index != int(instance.get("depot", 0))
            ],
            "vehicles": [
                {
                    "id": index,
                    "start_index": int(instance.get("depot", 0)),
                    "end_index": int(instance.get("depot", 0)),
                    "capacity": [capacity],
                }
                for index in range(vehicles)
            ],
        }

    @staticmethod
    def main(argv: list[str] | None = None) -> None:
        arguments = parser(solver=True)
        arguments.add_argument("--parameters", type=Path)
        args = arguments.parse_args(argv)
        started = time.perf_counter()
        try:
            from pitbench.problem_families.verification import CVRPFamily

            legacy_cli = VroomDriver._uses_legacy_cli(args.solver)
            parameters = (
                json.loads(args.parameters.read_text()) if args.parameters else None
            )
            if parameters is not None:
                if set(parameters) != {"exploration_level"}:
                    raise ParameterRejected("invalid VROOM parameter set")
                exploration_level = parameters["exploration_level"]
                if (
                    type(exploration_level) is not int
                    or not 0 <= exploration_level <= 5
                ):
                    raise ParameterRejected(
                        "exploration_level must be an integer in [0, 5]"
                    )
                if legacy_cli:
                    raise ParameterRejected(
                        "this VROOM CLI does not support exploration_level"
                    )
            with tempfile.TemporaryDirectory(prefix="pitbench-vroom-") as temporary:
                request = Path(temporary) / "request.json"
                raw_solution = Path(temporary) / "solution.json"
                instance_data = json.loads(args.instance.read_text())
                request.write_text(json.dumps(VroomDriver._request(instance_data)))
                command = [
                    args.solver,
                    "-i",
                    str(request),
                    "-t",
                    str(args.threads),
                ]
                if legacy_cli:
                    command.extend(["-o", str(raw_solution)])
                else:
                    command.extend(["-l", str(args.budget)])
                    if parameters is not None:
                        command.extend(["-x", str(parameters["exploration_level"])])
                try:
                    completed = subprocess.run(
                        command,
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=args.budget if legacy_cli else args.budget + 30,
                    )
                except subprocess.TimeoutExpired:
                    if not legacy_cli:
                        raise
                    resources = process_resources(child_process=True)
                    if args.trajectory is not None:
                        args.trajectory.parent.mkdir(parents=True, exist_ok=True)
                        args.trajectory.write_text("")
                    write_result(
                        args.output,
                        started=started,
                        valid=False,
                        has_solution=False,
                        solver_status="Time limit reached",
                        solver_termination=SolverTermination.TIME_LIMIT,
                        **resources,
                    )
                    return
                resources = process_resources(child_process=True)
                response_text = (
                    raw_solution.read_text()
                    if legacy_cli and raw_solution.is_file()
                    else completed.stdout
                )
            if completed.returncode:
                raise RuntimeError(completed.stderr.strip() or "vroom failed")
            response = json.loads(response_text)
            if response.get("code", 0) != 0:
                raise RuntimeError(str(response.get("error") or "vroom failed"))
            routes = VroomDriver._route_jobs(response)
            solution_path = write_solution(args.output, {"routes": routes})
            checked = CVRPFamily().verify(args.instance, solution_path)
            if not checked.feasible or checked.objective is None:
                raise ValueError(f"VROOM returned an invalid route: {checked.detail}")
            objective = float(checked.objective)
            solver_internal_objective = float(
                response.get("summary", {}).get("cost", 0)
            )
            solver_termination = (
                SolverTermination.OTHER if legacy_cli else SolverTermination.TIME_LIMIT
            )
            append_trajectory(
                args.trajectory,
                {"time_sec": time.perf_counter() - started, "objective": objective},
            )
            write_result(
                args.output,
                started=started,
                valid=True,
                has_solution=True,
                solver_status=("Completed" if legacy_cli else "Budget stop"),
                solver_termination=solver_termination,
                objective=objective,
                solver_internal_objective=solver_internal_objective,
                **resources,
            )
        except Exception as exc:
            write_result(
                args.output,
                started=started,
                valid=False,
                error=str(exc),
                failure_reason=failure_reason(exc),
            )
            raise


_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _match_number(pattern: str, source: str) -> float | None:
    found = re.search(pattern, source, re.IGNORECASE)
    return float(found.group(1)) if found else None


class HighsDriver(SolverDriver):
    """Invoke the highs backend with its existing configuration contract."""

    name = "highs"

    @staticmethod
    def options_file_content(parameters: dict[str, Any]) -> str:
        """Serialize fixed solver options in HiGHS' portable options-file form."""
        lines = []
        for name, value in sorted(parameters.items()):
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
                raise ValueError(f"invalid HiGHS option name: {name!r}")
            if isinstance(value, bool):
                formatted = str(value).lower()
            elif type(value) in {int, float}:
                formatted = str(value)
            elif isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", value):
                formatted = value
            else:
                raise ValueError(f"invalid HiGHS option value for {name}")
            lines.append(f"{name} = {formatted}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def solve_numeric(
        model,
        original,
        columns,
        directory: Path,
        options: dict,
        result: dict,
        started: float,
    ) -> None:
        """Run and independently verify one preserved or transformed numeric model."""
        import highspy
        import numpy as np

        from pitbench.evaluator.representations import LinearModelRepresentation
        from pitbench.problem_families.verification import NumericModelFamily

        result["error_stage"] = "setup"
        solver = highspy.Highs()
        for key, value in options.items():
            if solver.setOptionValue(key, value) != highspy.HighsStatus.kOk:
                raise ParameterRejected(f"HiGHS rejected {key}={value}")
        if (
            solver.passModel(LinearModelRepresentation.to_highs_lp(model))
            != highspy.HighsStatus.kOk
        ):
            raise ValueError("HiGHS rejected prepared model")
        solver.writeOptions(str(directory / "options.txt"))
        trajectory_path = directory / "trajectory.jsonl"
        trajectory_path.write_text("")

        def observe(event):
            data = event.data_out
            record = {
                "event": int(event.callback_type),
                "time_sec": finite(data.running_time),
                "primal_bound": finite(data.mip_primal_bound),
                "dual_bound": finite(data.mip_dual_bound),
                "solver_gap": finite(data.mip_gap),
                "nodes": int(data.mip_node_count),
                "simplex_iterations": int(data.simplex_iteration_count),
            }
            with trajectory_path.open("a") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")

        solver.cbMipLogging.subscribe(observe)
        solver.cbMipImprovingSolution.subscribe(observe)
        result["effective_parameters"] = {
            key: solver.getOptionValue(key)[1] for key in options
        }
        result["setup_wall_sec"] = time.perf_counter() - started
        solve_started = time.perf_counter()
        cpu_started = time.process_time()
        result["error_stage"] = "solve"
        status = solver.run()
        result.update(process_resources())
        result["solve_wall_sec"] = time.perf_counter() - solve_started
        result["solve_cpu_sec"] = time.process_time() - cpu_started
        info = solver.getInfo()
        model_status = solver.getModelStatus()
        solution = solver.getSolution()
        result.update(
            {
                "execution_status": "returned",
                "call_status": str(status),
                "model_status": solver.modelStatusToString(model_status),
                "solver_runtime_sec": solver.getRunTime(),
                "primal_solution_status": int(info.primal_solution_status),
                "solver_objective": finite(info.objective_function_value)
                if solution.value_valid
                else None,
                "dual_bound": finite(info.mip_dual_bound),
                "solver_gap": finite(info.mip_gap),
                "nodes": int(info.mip_node_count),
                "simplex_iterations": int(info.simplex_iteration_count),
                "optimal_with_configured_tolerances": model_status
                == highspy.HighsModelStatus.kOptimal,
                "time_to_solver_optimal_sec": solver.getRunTime()
                if model_status == highspy.HighsModelStatus.kOptimal
                else None,
                "solution_value_valid": solution.value_valid,
                "termination": (
                    SolverTermination.ERROR.value
                    if status == highspy.HighsStatus.kError
                    else SolverTermination.OPTIMAL.value
                    if model_status == highspy.HighsModelStatus.kOptimal
                    else SolverTermination.TIME_LIMIT.value
                    if model_status == highspy.HighsModelStatus.kTimeLimit
                    else SolverTermination.OTHER.value
                ),
            }
        )
        result["error_stage"] = "verification"
        verification_started = time.perf_counter()
        tolerance = options["mip_feasibility_tolerance"]
        result["verification"] = None
        result["transformed_verification"] = None
        if solution.value_valid:
            values = np.asarray(solution.col_value)
            mapped = LinearModelRepresentation.map_solution(values, columns)
            np.savez_compressed(
                directory / "solution.npz", values=values, original_values=mapped
            )
            solution_path = directory / "solution.json"
            solution_path.write_text(
                json.dumps({"values": values.tolist()}, allow_nan=False) + "\n"
            )
            result["solution_path"] = str(solution_path)
            result["verification"] = NumericModelFamily.check_model(
                original, mapped, tolerance
            )
            result["transformed_verification"] = NumericModelFamily.check_model(
                model, values, tolerance
            )
            result["objective_mapping_difference"] = result["verification"].get(
                "objective", 0
            ) - result["transformed_verification"].get("objective", 0)
            if "objective" in result["verification"]:
                result["objective_reporting_difference"] = (
                    result["verification"]["objective"] - info.objective_function_value
                )
        result["verified_feasible"] = bool(
            (result["verification"] or {}).get("feasible")
        )
        result["verification_wall_sec"] = time.perf_counter() - verification_started
        final_event = {
            "event": "final",
            "time_sec": solver.getRunTime(),
            "primal_bound": result["solver_objective"],
            "dual_bound": result["dual_bound"],
            "solver_gap": result["solver_gap"],
            "nodes": result["nodes"],
        }
        with trajectory_path.open("a") as handle:
            handle.write(json.dumps(final_event, allow_nan=False) + "\n")
        result.pop("error_stage", None)

    @staticmethod
    def _match(pattern: str, text: str) -> float | None:
        return _match_number(pattern, text)

    @staticmethod
    def main(argv: list[str] | None = None) -> None:
        arguments = parser(solver=True)
        arguments.add_argument("--parameters", type=Path)
        args = arguments.parse_args(argv)
        started = time.perf_counter()
        try:
            fixed_parameters = (
                json.loads(args.parameters.read_text())
                if args.parameters is not None
                else None
            )
            if fixed_parameters is not None and (
                not isinstance(fixed_parameters, dict)
                or not fixed_parameters
                or any(
                    not isinstance(name, str)
                    or type(value) not in {str, int, float, bool}
                    or (isinstance(value, float) and not math.isfinite(value))
                    for name, value in fixed_parameters.items()
                )
            ):
                raise ParameterRejected("invalid HiGHS fixed solver parameters")
            if args.parameters is not None:
                args.output.with_suffix(".parameters.json").write_text(
                    json.dumps(fixed_parameters, sort_keys=True, indent=2) + "\n"
                )
            with tempfile.TemporaryDirectory(prefix="pitbench-highs-") as temporary:
                raw_solution = Path(temporary) / "solution.txt"
                command = [
                    args.solver,
                    str(args.instance),
                    f"--time_limit={args.budget}",
                    f"--random_seed={args.seed}",
                    f"--threads={args.threads}",
                    f"--solution_file={raw_solution}",
                ]
                if fixed_parameters is not None:
                    options_file = Path(temporary) / "fixed_options.txt"
                    options_file.write_text(
                        HighsDriver.options_file_content(fixed_parameters)
                    )
                    command.append(f"--options_file={options_file}")
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=args.budget + 60,
                )
                cli_threads_option_fallback = False
                output_text = completed.stdout + completed.stderr
                unsupported_threads = re.search(
                    r"argument was not expected:\s*--threads(?:=|\b)",
                    output_text,
                    re.IGNORECASE,
                ) or re.search(
                    r"Option\s+[^A-Za-z0-9]*threads[^A-Za-z0-9]*\s+does not exist",
                    output_text,
                    re.IGNORECASE,
                )
                if unsupported_threads:
                    cli_threads_option_fallback = True
                    print(completed.stdout, end="", file=sys.stderr)
                    print(completed.stderr, end="", file=sys.stderr)
                    environment = dict(os.environ)
                    for variable in (
                        "OMP_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS",
                    ):
                        environment[variable] = str(args.threads)
                    completed = subprocess.run(
                        [part for part in command if not part.startswith("--threads=")],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=args.budget + 60,
                        env=environment,
                    )
                resources = process_resources(child_process=True)
                raw = (
                    raw_solution.read_text(errors="replace")
                    if raw_solution.exists()
                    else ""
                )
            print(completed.stdout, end="")
            print(completed.stderr, end="", file=sys.stderr)
            log = completed.stdout + "\n" + completed.stderr
            objective = HighsDriver._match(rf"Primal bound\s+({_FLOAT})", log)
            if objective is None:
                objective = HighsDriver._match(rf"(?m)^Objective\s+({_FLOAT})\s*$", raw)
            if objective is None:
                objective = HighsDriver._match(
                    rf"Objective value\s*:\s*({_FLOAT})", log
                )
            dual = HighsDriver._match(rf"Dual bound\s+({_FLOAT})", log)
            nodes = HighsDriver._match(r"Nodes\s+(\d+)", log)
            model_status = re.search(r"Model status\s*:\s*([^\n]+)", log, re.IGNORECASE)
            status = model_status.group(1).strip() if model_status else None
            if status is None and "Solving report" in log:
                report = log.rsplit("Solving report", 1)[-1]
                report_status = re.search(r"(?m)^\s*Status\s+([^\n]+)$", report)
                status = report_status.group(1).strip() if report_status else None
            if status is None:
                solution_status = re.search(r"(?m)^Model status\s*\n([^\n]+)", raw)
                status = solution_status.group(1).strip() if solution_status else None
            reason = None
            if status and "memory limit" in status.lower():
                reason = "out_of_memory"
            elif status and "error" in status.lower():
                reason = "solver_error"
            normalized_status = status.lower() if status is not None else ""
            solver_termination = (
                SolverTermination.OPTIMAL
                if normalized_status == "optimal"
                else SolverTermination.TIME_LIMIT
                if "time limit" in normalized_status
                else SolverTermination.ERROR
                if "error" in normalized_status
                else SolverTermination.OTHER
            )
            if (
                completed.returncode
                and solver_termination != SolverTermination.TIME_LIMIT
            ):
                raise RuntimeError(
                    f"HiGHS process exit {completed.returncode}: {completed.stderr.strip()}"
                )
            # Exact Performance accepts only a solver-optimal result as a
            # candidate solution. A time-limited MIP solve can write a fractional
            # relaxation with a finite primal bound; it is an ordinary unsolved
            # run, not a candidate for Qualification verification.
            has_solution = (
                objective is not None
                and solver_termination == SolverTermination.OPTIMAL
            )
            if has_solution:
                write_solution(
                    args.output,
                    {
                        "raw_solution": raw,
                        "objective": objective,
                        "solver_status": status,
                    },
                )
            append_trajectory(
                args.trajectory,
                {"time_sec": time.perf_counter() - started, "objective": objective},
            )
            write_result(
                args.output,
                started=started,
                valid=has_solution,
                has_solution=has_solution,
                failure_reason=reason,
                objective=objective,
                primal_bound=objective,
                dual_bound=dual,
                nodes=int(nodes) if nodes is not None else None,
                solver_status=status,
                solver_termination=solver_termination,
                cli_threads_option_fallback=cli_threads_option_fallback,
                **resources,
            )
        except Exception as exc:
            if isinstance(exc, subprocess.TimeoutExpired):
                for content, stream in (
                    (exc.stdout, sys.stdout),
                    (exc.stderr, sys.stderr),
                ):
                    if content:
                        print(
                            content.decode(errors="replace")
                            if isinstance(content, bytes)
                            else content,
                            end="",
                            file=stream,
                        )
            write_result(
                args.output,
                started=started,
                valid=False,
                error=str(exc),
                failure_reason=failure_reason(exc),
            )
            raise


class NativeMipCliDriver(SolverDriver):
    """Run a pinned native MIP shell and normalize its solution and progress."""

    name = ""
    dialect: ClassVar[str]

    @classmethod
    def _command(cls, args: argparse.Namespace, solution: Path) -> list[str]:
        if args.threads != 1:
            raise ValueError("native MIP tasks require one solver thread")
        if cls.dialect == "scip":
            return [
                args.solver,
                "-c",
                f"read {args.instance}",
                "-c",
                f"set limits time {args.budget}",
                "-c",
                f"set randomization randomseedshift {args.seed}",
                "-c",
                "optimize",
                "-c",
                "display statistics",
                "-c",
                f"write solution {solution}",
                "-c",
                "quit",
            ]
        return [
            args.solver,
            str(args.instance),
            "-seconds",
            str(args.budget),
            "-threads",
            str(args.threads),
            "-randomSeed",
            str(args.seed),
            "-solve",
            "-solution",
            str(solution),
            "-quit",
        ]

    @classmethod
    def _status(cls, log: str) -> str:
        pattern = (
            r"SCIP Status\s*:\s*([^\n]+)"
            if cls.dialect == "scip"
            else r"Result\s*-\s*([^\n]+)"
        )
        found = re.search(pattern, log)
        if found is None:
            raise ValueError("native MIP solver did not report a result status")
        return found.group(1).strip()

    @classmethod
    def _solution(
        cls, path: Path, names: set[str]
    ) -> tuple[float, dict[str, float]] | None:
        if not path.is_file():
            return None
        lines = path.read_text(errors="replace").splitlines()
        if not lines or "no integer solution" in lines[0].lower():
            return None
        objective_match = re.search(
            rf"objective value\s*:?[ \t]*({_FLOAT})",
            lines[1] if cls.dialect == "scip" and len(lines) > 1 else lines[0],
            re.IGNORECASE,
        )
        if objective_match is None:
            return None
        values = {name: 0.0 for name in names}
        observed: set[str] = set()
        for line in lines[2 if cls.dialect == "scip" else 1 :]:
            fields = line.split()
            if cls.dialect == "scip":
                if len(fields) < 2 or fields[0] not in names:
                    continue
                name, raw_value = fields[:2]
            else:
                if len(fields) < 3 or not fields[0].isdigit() or fields[1] not in names:
                    continue
                name, raw_value = fields[1:3]
            if name in observed:
                raise ValueError(f"duplicate native MIP variable: {name}")
            number = float(raw_value)
            if not math.isfinite(number):
                raise ValueError(f"nonfinite native MIP variable: {name}")
            values[name] = number
            observed.add(name)
        return float(objective_match.group(1)), values

    @classmethod
    def _progress(cls, log: str) -> list[dict]:
        records: list[dict] = []
        if cls.dialect == "scip":
            for line in log.splitlines():
                fields = [part.strip() for part in line.split("|")]
                if len(fields) < 16:
                    continue
                elapsed = re.search(rf"({_FLOAT})s$", fields[0])
                if elapsed is None:
                    continue
                primal = (
                    finite(fields[15]) if re.fullmatch(_FLOAT, fields[15]) else None
                )
                dual = finite(fields[14]) if re.fullmatch(_FLOAT, fields[14]) else None
                if primal is not None or dual is not None:
                    records.append(
                        {
                            "time_sec": float(elapsed.group(1)),
                            "primal_bound": primal,
                            "dual_bound": dual,
                        }
                    )
            return records
        pattern = re.compile(
            rf"Cbc0010I After (\d+) nodes, .*?([+-]?{_FLOAT}) best solution, "
            rf"best possible ([+-]?{_FLOAT}) \(({_FLOAT}) seconds\)"
        )
        for match in pattern.finditer(log):
            records.append(
                {
                    "time_sec": float(match.group(4)),
                    "nodes": int(match.group(1)),
                    "primal_bound": float(match.group(2)),
                    "dual_bound": float(match.group(3)),
                }
            )
        return records

    @classmethod
    def main(cls, argv: list[str] | None = None) -> None:
        from pitbench.problem_families.verification import expected_mip_variable_names

        args = parser(solver=True).parse_args(argv)
        started = time.perf_counter()
        try:
            names = expected_mip_variable_names(args.instance)
            with tempfile.TemporaryDirectory(prefix="pitbench-native-mip-") as root:
                solution_path = Path(root) / "solution.sol"
                completed = subprocess.run(
                    cls._command(args, solution_path),
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=args.budget + 60,
                )
                resources = process_resources(child_process=True)
                solution = cls._solution(solution_path, names)
            print(completed.stdout, end="")
            print(completed.stderr, end="", file=sys.stderr)
            if completed.returncode:
                raise RuntimeError(f"{cls.dialect} process exit {completed.returncode}")
            log = completed.stdout + "\n" + completed.stderr
            status = cls._status(log)
            lower_status = status.lower()
            termination = (
                SolverTermination.OPTIMAL
                if "optimal solution found" in lower_status
                else SolverTermination.TIME_LIMIT
                if "time limit" in lower_status or "maximum time" in lower_status
                else SolverTermination.ERROR
                if "error" in lower_status
                else SolverTermination.OTHER
            )
            if termination == SolverTermination.OPTIMAL and solution is None:
                raise ValueError("solver claimed optimality without a primal solution")
            objective = solution[0] if solution is not None else None
            if termination == SolverTermination.OPTIMAL and solution is not None:
                write_solution(
                    args.output,
                    {
                        "objective": objective,
                        "values": solution[1],
                        "solver_status": status,
                    },
                )
            if cls.dialect == "scip":
                dual = _match_number(rf"Dual Bound\s*:\s*({_FLOAT})", log)
                nodes = _match_number(r"Solving Nodes\s*:\s*(\d+)", log)
            else:
                dual = _match_number(rf"Lower bound\s*:\s*({_FLOAT})", log)
                nodes = _match_number(r"Enumerated nodes\s*:\s*(\d+)", log)
            if dual is None and termination == SolverTermination.OPTIMAL:
                dual = objective
            for record in cls._progress(log):
                append_trajectory(args.trajectory, record)
            append_trajectory(
                args.trajectory,
                {
                    "event": "final",
                    "time_sec": time.perf_counter() - started,
                    "primal_bound": objective,
                    "dual_bound": dual,
                    "nodes": int(nodes) if nodes is not None else None,
                },
            )
            write_result(
                args.output,
                started=started,
                valid=termination == SolverTermination.OPTIMAL,
                has_solution=termination == SolverTermination.OPTIMAL,
                objective=objective,
                primal_bound=objective,
                dual_bound=dual,
                nodes=int(nodes) if nodes is not None else None,
                solver_status=status,
                solver_termination=termination,
                **resources,
            )
        except Exception as exc:
            write_result(
                args.output,
                started=started,
                valid=False,
                error=str(exc),
                failure_reason=failure_reason(exc),
                solver_termination=SolverTermination.ERROR,
            )
            raise


class ScipDriver(NativeMipCliDriver):
    name = "scip"
    dialect = "scip"


class CbcDriver(NativeMipCliDriver):
    name = "cbc"
    dialect = "cbc"


class ChocoDriver(SolverDriver):
    """Invoke the choco backend with its existing configuration contract."""

    name = "choco"

    @staticmethod
    def main(argv: list[str] | None = None) -> None:
        arguments = parser()
        arguments.add_argument("--parameters", type=Path)
        args = arguments.parse_args(argv)
        parameters = (
            json.loads(args.parameters.read_text()) if args.parameters else None
        )
        execute(
            environment_key="PITBENCH_CHOCO_RUNNER",
            instance=args.instance,
            output=args.output,
            trajectory=args.trajectory,
            seed=args.seed,
            budget=args.budget,
            threads=args.threads,
            parameters=parameters,
        )


class GecodeExactDriver(SolverDriver):
    """Invoke the shared native adapter for exact Gecode CP workloads."""

    name = "gecode_exact"

    @staticmethod
    def main(argv: list[str] | None = None) -> None:
        arguments = parser()
        arguments.add_argument("--parameters", type=Path)
        args = arguments.parse_args(argv)
        parameters = (
            json.loads(args.parameters.read_text()) if args.parameters else None
        )
        execute(
            environment_key="PITBENCH_GECODE_RUNNER",
            instance=args.instance,
            output=args.output,
            trajectory=args.trajectory,
            seed=args.seed,
            budget=args.budget,
            threads=args.threads,
            parameters=parameters,
        )


class OrToolsCpSatExactDriver(SolverDriver):
    """Invoke the fixed-proto CP-SAT solving adapter."""

    name = "ortools_cp_sat_exact"

    @staticmethod
    def main(argv: list[str] | None = None) -> None:
        arguments = parser()
        arguments.add_argument("--parameters", type=Path)
        args = arguments.parse_args(argv)
        parameters = (
            json.loads(args.parameters.read_text()) if args.parameters else None
        )
        execute(
            environment_key="PITBENCH_ORTOOLS_CP_SAT_RUNNER",
            instance=args.instance,
            output=args.output,
            trajectory=args.trajectory,
            seed=args.seed,
            budget=args.budget,
            threads=args.threads,
            parameters=parameters,
        )


DRIVERS = SolverDriver._drivers


class HighsScenarioSession(ScenarioSession):
    """Binding-based trajectories and stateful model updates in a fresh worker."""

    @classmethod
    def identity(cls) -> dict:
        import hashlib

        import highspy
        import highspy._core

        solver = highspy.Highs()
        package = Path(highspy.__file__).parent
        python_sources = {
            str(path.relative_to(package)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(package.rglob("*.py"))
        }
        return {
            "plugin": f"{cls.__module__}:{cls.__name__}",
            "repository_plugin": "pitbench.repositories.plugins:HighsRepositoryPlugin",
            "representations": ["row_column_permutation"],
            "version": solver.version(),
            "binding_git_hash": solver.githash(),
            "binding_sha256": hashlib.sha256(
                Path(highspy._core.__file__).read_bytes()
            ).hexdigest(),
            "package_code_sha256": hashlib.sha256(
                json.dumps(python_sources, sort_keys=True).encode()
            ).hexdigest(),
        }

    def __init__(self, *, seed, threads, budget, tolerance, options):
        import highspy

        self.solver = highspy.Highs()
        fixed = {
            "random_seed": seed,
            "threads": threads,
            "time_limit": budget,
            "mip_feasibility_tolerance": tolerance,
        }
        if any(k in fixed and fixed[k] != v for k, v in options.items()):
            raise ValueError(
                "scenario options conflict with the declared experimental grid"
            )
        for key, value in {"output_flag": False, **options, **fixed}.items():
            if self.solver.setOptionValue(key, value) != highspy.HighsStatus.kOk:
                raise ParameterRejected(f"HiGHS rejected {key}={value}")

    def load_model(self, model):
        import highspy

        from pitbench.evaluator.representations import LinearModelRepresentation

        if (
            self.solver.passModel(LinearModelRepresentation.to_highs_lp(model))
            != highspy.HighsStatus.kOk
        ):
            raise ValueError("solver rejected scenario model")

    def update_model(self, previous, model):
        import highspy
        import numpy as np

        from pitbench.solver_drivers.common import ScenarioUnsupported

        if previous["matrix"].shape != model["matrix"].shape or not np.array_equal(
            previous["integrality"], model["integrality"]
        ):
            raise ScenarioUnsupported(
                "in-place scenario updates require fixed dimensions and integrality"
            )
        s = self.solver
        cols = np.arange(len(model["cost"]), dtype=np.int32)
        rows = np.arange(len(model["row_lower"]), dtype=np.int32)
        statuses = [
            s.changeColsCost(len(cols), cols, model["cost"]),
            s.changeColsBounds(len(cols), cols, model["col_lower"], model["col_upper"]),
            s.changeRowsBounds(len(rows), rows, model["row_lower"], model["row_upper"]),
            s.changeObjectiveOffset(model["offset"]),
        ]
        if previous["sense"] != model["sense"]:
            raise ScenarioUnsupported(
                "objective sense changes require another experimental contract"
            )
        delta = (model["matrix"] - previous["matrix"]).tocoo()
        for i, j in zip(delta.row, delta.col, strict=True):
            statuses.append(s.changeCoeff(int(i), int(j), float(model["matrix"][i, j])))
        if any(status != highspy.HighsStatus.kOk for status in statuses):
            raise ValueError("solver rejected a declared model update")

    def solve(self, emit):
        import highspy

        started = time.perf_counter()

        def observe(event):
            values = event.data_out.mip_solution
            if values is not None and len(values) == self.solver.getNumCol():
                emit(time.perf_counter() - started, list(values), None)

        self.solver.cbMipImprovingSolution.subscribe(observe)
        try:
            status = self.solver.run()
        finally:
            self.solver.cbMipImprovingSolution.unsubscribe(observe)
        elapsed = time.perf_counter() - started
        solution = self.solver.getSolution()
        values = list(solution.col_value) if solution.value_valid else None
        multipliers = list(solution.row_dual) if solution.dual_valid else None
        if values is not None or multipliers is not None:
            emit(elapsed, values, multipliers)
        model_status = self.solver.getModelStatus()
        termination = (
            "solver_error"
            if status == highspy.HighsStatus.kError
            else "optimal"
            if model_status == highspy.HighsModelStatus.kOptimal
            else "time_limit"
            if model_status == highspy.HighsModelStatus.kTimeLimit
            else "infeasible"
            if model_status == highspy.HighsModelStatus.kInfeasible
            else "other"
        )
        return {"termination": termination, "values": values, "solve_time_sec": elapsed}

    def reset_search(self):
        self.solver.clearSolver()


def driver_records_trajectory(driver_name: str) -> bool:
    try:
        return DRIVERS[driver_name].records_trajectory
    except KeyError:
        raise ValueError(f"unsupported solver driver: {driver_name}") from None


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    selector = argparse.ArgumentParser(description=__doc__)
    selector.add_argument("driver", choices=tuple(DRIVERS))
    args = selector.parse_args(argv[:1])
    DRIVERS[args.driver].main(argv[1:])


if __name__ == "__main__":
    main()
