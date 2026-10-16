"""Re-score identical-code judge runs and independent Base repeats.

This reads retained PitBench artifacts. It does not launch a second evaluation
pipeline or substitute synthetic solver observations for real runs.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from pitbench.evaluator.storage import ObservationStore
from pitbench.metrics.performance_report import (
    PerformanceClassification,
    _cluster_mean_ci,
    adjudicate_performance_confirmation,
    compute_exact_run_outcome,
    compute_task_performance_report,
)
from pitbench.metrics.resource_report import compute_resource_reports
from pitbench.schema.observation import (
    CodeState,
    ExpectedRun,
    ExpectedRunGrid,
    RunObservation,
)
from pitbench.schema.task import PitBenchTask

ROOT = Path(__file__).resolve().parents[1]
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _identity(directory: Path) -> dict:
    return json.loads((directory / "judge-checkpoint.json").read_text())["identity"]


def _observations(directory: Path) -> list[RunObservation]:
    return ObservationStore.read(directory / "trials.parquet")


def _expected_grid(directory: Path) -> ExpectedRunGrid:
    return ExpectedRunGrid.model_validate_json(
        (directory / "expected-run-grid.json").read_text()
    )


def _grid_key(run: ExpectedRun | RunObservation) -> tuple:
    return (
        run.task_id,
        run.code_state,
        run.instance_set,
        run.instance_id,
        run.solver_seed,
        run.budget_sec,
        run.condition_id,
        run.equivalence_parent_id,
        run.test_suite,
    )


def _primary_keys(
    observations: list[RunObservation], task: PitBenchTask
) -> set[tuple[str, int, float]]:
    return {
        (item.instance_id, item.solver_seed, item.budget_sec)
        for item in observations
        if item.code_state is CodeState.BASE
        and (item.instance_set_kind or item.instance_set) == "judge_id"
        and item.test_suite is None
        and item.equivalence_parent_id is None
        and item.budget_sec in task.evaluation.budgets_sec
    }


def _score(
    task: PitBenchTask,
    observations: list[RunObservation],
    expected: ExpectedRunGrid,
) -> dict:
    expected_keys = [_grid_key(item) for item in expected.runs]
    observed_keys = [_grid_key(item) for item in observations]
    if (
        len(expected_keys) != len(set(expected_keys))
        or len(observed_keys) != len(set(observed_keys))
        or set(expected_keys) != set(observed_keys)
    ):
        raise ValueError(f"incomplete or duplicate expected grid for {task.task_id}")
    report = compute_task_performance_report(
        observations, task=task, expected_run_grid=expected
    )
    if report is None:
        raise ValueError(f"no Performance report for {task.task_id}")
    primary = report.primary
    result = {
        "classification": report.classification.value,
        "primary_budget_sec": report.primary_budget_sec,
        "observation_count": len(observations),
        "status_counts": dict(
            sorted(Counter(x.status.value for x in observations).items())
        ),
        "paired_diagnostics": _paired_diagnostics(task, observations),
    }
    if hasattr(primary.paired, "mean_gap_reduction"):
        result.update(
            kind="heuristic",
            paired_gap_reduction=primary.paired.mean_gap_reduction,
            paired_gap_ci95=(
                primary.paired.mean_gap_reduction_ci95.model_dump()
                if primary.paired.mean_gap_reduction_ci95
                else None
            ),
        )
    else:
        result.update(
            kind="exact",
            base_solved=primary.base.solved_run_count,
            agent_solved=primary.agent.solved_run_count,
            expected_runs_per_state=primary.base.expected_run_count,
            paired_par2_reduction_sec=primary.paired.mean_penalized_runtime_reduction_sec,
            paired_par2_ci95=(
                primary.paired.mean_penalized_runtime_reduction_ci95.model_dump()
                if primary.paired.mean_penalized_runtime_reduction_ci95
                else None
            ),
        )
        coverage_interval = getattr(
            primary.paired, "mean_verified_solved_coverage_reduction_ci95", None
        )
        result["paired_coverage_ci95"] = (
            coverage_interval.model_dump() if coverage_interval else None
        )
    judge_seeds = {
        item.solver_seed
        for item in observations
        if (item.instance_set_kind or item.instance_set) == "judge_id"
        and item.test_suite is None
        and item.equivalence_parent_id is None
    }
    resources, _ = compute_resource_reports(
        observations,
        primary_budget_sec=task.evaluation.primary_budget_sec,
        expected_seed_count=len(judge_seeds),
        budgets_sec=task.evaluation.budgets_sec,
        expected_instance_counts={item.name: item.size for item in task.instance_sets},
    )
    primary_resources = resources.by_instance_set["judge_id"][
        f"{task.evaluation.primary_budget_sec:g}"
    ]
    result["primary_resource_ratios"] = {
        "memory": primary_resources.memory.model_dump(),
        "cpu": primary_resources.cpu.model_dump(),
    }
    return result


def _paired_diagnostics(task: PitBenchTask, observations: list[RunObservation]) -> dict:
    """Describe noise without changing the task's public Performance metric."""
    paired: dict[tuple[str, int], dict[CodeState, RunObservation]] = defaultdict(dict)
    for item in observations:
        if (
            (item.instance_set_kind or item.instance_set) == "judge_id"
            and item.test_suite is None
            and item.equivalence_parent_id is None
            and item.budget_sec == task.evaluation.primary_budget_sec
        ):
            paired[(item.instance_id, item.solver_seed)][item.code_state] = item
    wall_by_instance: dict[tuple[str, str], list[float]] = defaultdict(list)
    gap_by_instance: dict[tuple[str, str], list[float]] = defaultdict(list)
    coverage_by_instance: dict[tuple[str, str], list[float]] = defaultdict(list)
    changed_coverage_by_instance: dict[str, int] = defaultdict(int)
    changed_objective_by_instance: dict[str, int] = defaultdict(int)
    missing_wall = missing_coverage = 0
    for (instance_id, _), pair in sorted(paired.items()):
        if set(pair) != set(CodeState):
            raise ValueError(f"unpaired primary observation: {instance_id}")
        base, agent = pair[CodeState.BASE], pair[CodeState.AGENT]
        if (
            base.valid
            and agent.valid
            and base.objective is not None
            and agent.objective is not None
            and base.objective != agent.objective
        ):
            changed_objective_by_instance[instance_id] += 1
        if base.wall_time_sec is None or agent.wall_time_sec is None:
            missing_wall += 1
        else:
            wall_by_instance[("judge_id", instance_id)].append(
                base.wall_time_sec - agent.wall_time_sec
            )
        if base.normalized_gap is not None and agent.normalized_gap is not None:
            gap_by_instance[("judge_id", instance_id)].append(
                base.normalized_gap - agent.normalized_gap
            )
        if task.evaluation.performance_protocol.value == "exact_verified_solve":
            base_outcome = compute_exact_run_outcome(
                base, time_basis=task.evaluation.exact_time_basis
            )
            agent_outcome = compute_exact_run_outcome(
                agent, time_basis=task.evaluation.exact_time_basis
            )
            if base_outcome.solved is None or agent_outcome.solved is None:
                missing_coverage += 1
            else:
                delta = int(agent_outcome.solved) - int(base_outcome.solved)
                coverage_by_instance[("judge_id", instance_id)].append(float(delta))
                changed_coverage_by_instance[instance_id] += delta
    wall_ci = _cluster_mean_ci(wall_by_instance, seed=20260824)
    coverage_ci = _cluster_mean_ci(coverage_by_instance, seed=20260824)
    gap_instance_means = [
        statistics.fmean(values) for values in gap_by_instance.values()
    ]
    coverage_instance_means = [
        statistics.fmean(values) for values in coverage_by_instance.values()
    ]
    return {
        "primary_paired_runs": len(paired),
        "primary_instances": len({instance for instance, _ in paired}),
        "mean_wall_time_reduction_sec": (
            statistics.fmean(statistics.fmean(v) for v in wall_by_instance.values())
            if wall_by_instance
            else None
        ),
        "wall_time_ci95": wall_ci.model_dump() if wall_ci else None,
        "missing_wall_pairs": missing_wall,
        "paired_gap_sign_flip_p_two_sided": _sign_flip_p(gap_instance_means),
        "mean_verified_solved_coverage_reduction": (
            statistics.fmean(statistics.fmean(v) for v in coverage_by_instance.values())
            if coverage_by_instance
            else None
        ),
        "coverage_ci95": coverage_ci.model_dump() if coverage_ci else None,
        "coverage_sign_flip_p_two_sided": _sign_flip_p(coverage_instance_means),
        "missing_coverage_pairs": missing_coverage,
        "coverage_changes_by_instance": {
            key: value
            for key, value in sorted(changed_coverage_by_instance.items())
            if value
        },
        "objective_changed_pairs": sum(changed_objective_by_instance.values()),
        "objective_changes_by_instance": dict(
            sorted(changed_objective_by_instance.items())
        ),
    }


def _sign_flip_p(instance_means: list[float]) -> float | None:
    """Exploratory paired-instance label-swap randomization diagnostic."""
    if not instance_means:
        return None
    observed = abs(statistics.fmean(instance_means))
    if observed == 0:
        return 1.0
    rng = random.Random(20261015)
    resamples = 50_000
    exceedances = sum(
        abs(
            statistics.fmean(
                (1 if rng.getrandbits(1) else -1) * value for value in instance_means
            )
        )
        >= observed - 1e-15
        for _ in range(resamples)
    )
    return (exceedances + 1) / (resamples + 1)


def calibrate(manifest: dict) -> dict:
    task_entries = {entry["task_id"]: entry for entry in manifest["task_configs"]}
    task_models = {
        task_id: PitBenchTask.from_yaml(_path(entry["task_config"]))
        for task_id, entry in task_entries.items()
    }
    controls = []
    for task_id, directories in manifest.get("current_noop_runs", {}).items():
        entry = task_entries[task_id]
        task = task_models[task_id]
        for name in directories:
            directory = _path(name)
            identity = _identity(directory)
            if identity["candidate_patch_sha256"] != EMPTY_SHA256:
                raise ValueError(f"not an empty patch: {directory}")
            if identity["task_config_sha256"] != entry["task_config_sha256"]:
                raise ValueError(f"task configuration changed: {directory}")
            if identity["judge_image"] != entry["judge_image"]:
                raise ValueError(f"judge image changed: {directory}")
            if set(identity["code_states"]) != {"base", "agent"}:
                raise ValueError(f"not a same-run Base/Agent control: {directory}")
            resources = manifest["primary_resource_stratum"]
            if (
                identity["cpus"] != resources["judge_cpus"]
                or identity["parallel_runs"] != resources["judge_parallel_runs"]
            ):
                raise ValueError(f"resource stratum changed: {directory}")
            observations = _observations(directory)
            score = _score(task, observations, _expected_grid(directory))
            controls.append(
                {
                    "task_id": task_id,
                    "directory": name,
                    "identity": identity,
                    "harness_identity_recorded": bool(identity.get("harness_sha256")),
                    "false_direction": score["classification"]
                    in {"improved", "regressed"},
                    "public_classification": score["classification"],
                    "performance": score,
                }
            )

    for task_id, repeat_name in manifest.get("confirmation_runs", {}).items():
        matching = [item for item in controls if item["task_id"] == task_id]
        if len(matching) != 1:
            raise ValueError(
                f"confirmation needs exactly one primary control: {task_id}"
            )
        control = matching[0]
        first_dir = _path(control["directory"])
        repeat_dir = _path(repeat_name)
        first_identity = control["identity"]
        repeat_identity = _identity(repeat_dir)
        for field in (
            "candidate_patch_sha256",
            "task_config_sha256",
            "judge_image",
            "harness_sha256",
            "cpus",
            "memory",
            "parallel_runs",
        ):
            if first_identity.get(field) != repeat_identity.get(field):
                raise ValueError(f"confirmation differs in {field}: {task_id}")
        if first_identity["code_states"] != list(
            reversed(repeat_identity["code_states"])
        ):
            raise ValueError(f"confirmation did not reverse order: {task_id}")
        first_grid = _expected_grid(first_dir)
        repeat_grid = _expected_grid(repeat_dir)
        if first_grid != repeat_grid:
            raise ValueError(f"confirmation uses a different grid: {task_id}")
        task = task_models[task_id]
        first_report = compute_task_performance_report(
            _observations(first_dir), task=task, expected_run_grid=first_grid
        )
        repeat_observations = _observations(repeat_dir)
        repeat_report = compute_task_performance_report(
            repeat_observations, task=task, expected_run_grid=repeat_grid
        )
        if first_report is None or repeat_report is None:
            raise ValueError(f"confirmation report missing: {task_id}")
        if first_report.classification not in {
            PerformanceClassification.IMPROVED,
            PerformanceClassification.REGRESSED,
        }:
            raise ValueError(
                f"non-directional control does not need confirmation: {task_id}"
            )
        final = adjudicate_performance_confirmation(
            first_report,
            repeat_report,
            repeat_observation_count=len(repeat_observations),
            repeat_valid_observation_count=sum(
                item.valid for item in repeat_observations
            ),
        )
        control["confirmation"] = {
            "directory": repeat_name,
            "identity": repeat_identity,
            "performance": _score(task, repeat_observations, repeat_grid),
            "final": final.confirmation.model_dump(mode="json"),
        }
        control["public_classification"] = final.classification.value

    baseline_pairs = []
    for task_id, directories in manifest.get("historical_base_groups", {}).items():
        task = task_models[task_id]
        for left_name, right_name in itertools.combinations(directories, 2):
            left, right = _path(left_name), _path(right_name)
            left_identity, right_identity = _identity(left), _identity(right)
            for field in (
                "task_config_sha256",
                "judge_image",
                "cpus",
                "parallel_runs",
            ):
                if left_identity[field] != right_identity[field]:
                    raise ValueError(f"Base repeats differ in {field}: {left}, {right}")
            if (
                left_identity.get("harness_sha256")
                and right_identity.get("harness_sha256")
                and left_identity["harness_sha256"] != right_identity["harness_sha256"]
            ):
                raise ValueError(
                    f"Base repeats use different harness code: {left}, {right}"
                )
            left_base = [
                item
                for item in _observations(left)
                if item.code_state is CodeState.BASE
            ]
            right_base = [
                item
                for item in _observations(right)
                if item.code_state is CodeState.BASE
            ]
            if _primary_keys(left_base, task) != _primary_keys(right_base, task):
                raise ValueError(f"Base primary grids differ: {left}, {right}")
            pseudo_agent = [
                item.model_copy(update={"code_state": CodeState.AGENT})
                for item in right_base
            ]
            score = _score(task, left_base + pseudo_agent, _expected_grid(left))
            baseline_pairs.append(
                {
                    "task_id": task_id,
                    "base_left": left_name,
                    "base_right": right_name,
                    "resource_stratum": {
                        "judge_cpus": left_identity["cpus"],
                        "judge_parallel_runs": left_identity["parallel_runs"],
                    },
                    "harness_identity_verified": bool(
                        left_identity.get("harness_sha256")
                        and right_identity.get("harness_sha256")
                    ),
                    "primary_grid_count": len(_primary_keys(left_base, task)),
                    "false_direction": score["classification"]
                    in {"improved", "regressed"},
                    "performance": score,
                }
            )
    return {
        "method": "docs/evaluation-noise-calibration.md",
        "control_count": len(controls),
        "control_false_direction_count": sum(x["false_direction"] for x in controls),
        "public_control_false_direction_count": sum(
            x["public_classification"] in {"improved", "regressed"} for x in controls
        ),
        "baseline_pair_count": len(baseline_pairs),
        "baseline_pair_false_direction_count": sum(
            x["false_direction"] for x in baseline_pairs
        ),
        "controls": controls,
        "historical_base_pairs": baseline_pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = calibrate(json.loads(args.manifest.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"no-op false directions: {result['control_false_direction_count']}/"
        f"{result['control_count']} raw, "
        f"{result['public_control_false_direction_count']}/"
        f"{result['control_count']} after confirmation; "
        f"independent Base-pair false directions: "
        f"{result['baseline_pair_false_direction_count']}/"
        f"{result['baseline_pair_count']}"
    )


if __name__ == "__main__":
    main()
