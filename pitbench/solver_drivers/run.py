"""Solver invocation and output adapters, selected by driver name."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from pitbench.schema.observation import SolverTermination
from pitbench.solver_drivers.common import (
    ParameterRejected,
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
            from pyvrp import Model, read
            from pyvrp.stop import MaxRuntime

            options = {}
            if args.parameters is not None:
                params = PyVRPDriver.parameters(json.loads(args.parameters.read_text()))
                options["params"] = params
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.with_suffix(".parameters.json").write_text(
                    json.dumps(PyVRPDriver.parameter_values(params), indent=2) + "\n"
                )
            with tempfile.TemporaryDirectory(prefix="pitbench-pyvrp-") as temporary:
                vrp = Path(temporary) / "instance.vrp"
                PyVRPDriver._vrplib(instance, vrp)
                data = read(vrp, round_func="round")
                result = Model.from_data(data).solve(
                    stop=MaxRuntime(args.budget),
                    seed=args.seed,
                    display=False,
                    **options,
                )
                resources = process_resources()
            if not result.best.is_feasible():
                write_result(
                    args.output,
                    started=started,
                    valid=False,
                    has_solution=False,
                    solver_status="Budget stop",
                    solver_runtime_sec=result.runtime,
                    iterations=result.num_iterations,
                    **resources,
                )
                return
            routes = [
                PyVRPDriver._route_visits(route) for route in result.best.routes()
            ]
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
    def _request(instance: dict) -> dict:
        coordinates = instance["coordinates"]
        matrix = [
            [round(math.hypot(a[0] - b[0], a[1] - b[1]) * 1000) for b in coordinates]
            for a in coordinates
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
            with tempfile.TemporaryDirectory(prefix="pitbench-vroom-") as temporary:
                request = Path(temporary) / "request.json"
                request.write_text(
                    json.dumps(
                        VroomDriver._request(json.loads(args.instance.read_text()))
                    )
                )
                command = [
                    args.solver,
                    "-i",
                    str(request),
                    "-t",
                    str(args.threads),
                    "-l",
                    str(args.budget),
                ]
                if parameters is not None:
                    command.extend(["-x", str(parameters["exploration_level"])])
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=args.budget + 30,
                )
            if completed.returncode:
                raise RuntimeError(completed.stderr.strip() or "vroom failed")
            response = json.loads(completed.stdout)
            routes = [
                [step["id"] for step in route["steps"] if step.get("type") == "job"]
                for route in response.get("routes", [])
            ]
            objective = float(response.get("summary", {}).get("cost", 0))
            write_solution(args.output, {"routes": routes})
            append_trajectory(
                args.trajectory,
                {"time_sec": time.perf_counter() - started, "objective": objective},
            )
            write_result(
                args.output,
                started=started,
                valid=True,
                has_solution=True,
                solver_status="Budget stop",
                solver_termination=SolverTermination.TIME_LIMIT,
                objective=objective,
            )
        except Exception as exc:
            write_result(args.output, started=started, valid=False, error=str(exc))
            raise


_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


class HighsDriver(SolverDriver):
    """Invoke the highs backend with its existing configuration contract."""

    name = "highs"

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
        found = re.search(pattern, text, re.IGNORECASE)
        return float(found.group(1)) if found else None

    @staticmethod
    def main(argv: list[str] | None = None) -> None:
        args = parser(solver=True).parse_args(argv)
        started = time.perf_counter()
        try:
            with tempfile.TemporaryDirectory(prefix="pitbench-highs-") as temporary:
                raw_solution = Path(temporary) / "solution.txt"
                completed = subprocess.run(
                    [
                        args.solver,
                        str(args.instance),
                        f"--time_limit={args.budget}",
                        f"--random_seed={args.seed}",
                        f"--threads={args.threads}",
                        f"--solution_file={raw_solution}",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=args.budget + 60,
                )
                resources = process_resources(child_process=True)
                raw = (
                    raw_solution.read_text(errors="replace")
                    if raw_solution.exists()
                    else ""
                )
            print(completed.stdout, end="")
            print(completed.stderr, end="", file=sys.stderr)
            if completed.returncode:
                raise RuntimeError(
                    f"HiGHS process exit {completed.returncode}: {completed.stderr.strip()}"
                )
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
                valid=objective is not None,
                has_solution=has_solution,
                failure_reason=reason,
                objective=objective,
                primal_bound=objective,
                dual_bound=dual,
                nodes=int(nodes) if nodes is not None else None,
                solver_status=status,
                solver_termination=solver_termination,
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


class ChocoDriver(SolverDriver):
    """Invoke the choco backend with its existing configuration contract."""

    name = "choco"

    @staticmethod
    def main(argv: list[str] | None = None) -> None:
        arguments = parser()
        arguments.add_argument("--parameters", type=Path)
        args = arguments.parse_args(argv)
        parameters = json.loads(args.parameters.read_text()) if args.parameters else None
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


class OrToolsCpSatExactDriver(SolverDriver):
    """Invoke the fixed-proto parallel CP-SAT solving adapter."""

    name = "ortools_cp_sat_exact"

    @staticmethod
    def main(argv: list[str] | None = None) -> None:
        arguments = parser(trajectory=False)
        arguments.add_argument("--parameters", type=Path)
        args = arguments.parse_args(argv)
        parameters = json.loads(args.parameters.read_text()) if args.parameters else None
        execute(
            environment_key="PITBENCH_ORTOOLS_CP_SAT_RUNNER",
            instance=args.instance,
            output=args.output,
            trajectory=None,
            seed=args.seed,
            budget=args.budget,
            threads=args.threads,
            parameters=parameters,
        )


DRIVERS = SolverDriver._drivers


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    selector = argparse.ArgumentParser(description=__doc__)
    selector.add_argument("driver", choices=tuple(DRIVERS))
    args = selector.parse_args(argv[:1])
    DRIVERS[args.driver].main(argv[1:])


if __name__ == "__main__":
    main()
