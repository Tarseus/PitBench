"""Fixed-panel parameter search; solver-specific collection lives in collection.py."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from threading import Event, Lock
from uuid import uuid4

from pitbench.evaluator.collection import (
    ROOT,
    collection_backend,
    configuration_worker,
    run_collection_jobs,
    run_collection_process,
    write_json,
)


def read_data(path: Path):
    if path.suffix == ".json":
        return json.loads(path.read_text())
    import yaml

    return yaml.safe_load(path.read_text())


def _external_panel(config, seed=0):
    raise RuntimeError("PitBench evaluates complete panels through ask/tell")


def _docker_collector_command(
    image: str,
    command: list[str],
    *,
    cpu: int | None = None,
    environment: dict[str, str] | None = None,
    pull_policy: str = "missing",
    repository_root: Path = ROOT,
    output_root: Path | None = None,
) -> list[str]:
    """Run one collector action inside its pinned solver image."""
    result = [
        "docker",
        "run",
        "--rm",
        f"--pull={pull_policy}",
        "--network",
        "none",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=1g,mode=1777",
        "--env",
        f"PYTHONPATH={repository_root}:/workspace/repo",
        "--volume",
        f"{repository_root}:{repository_root}:rw",
        "--volume",
        f"{repository_root / 'adapters/jvm'}:/opt/pitbench-jvm:ro",
        "--workdir",
        "/workspace/repo",
    ]
    for name, value in sorted((environment or {}).items()):
        result.extend(["--env", f"{name}={value}"])
    if output_root is not None and not output_root.resolve().is_relative_to(
        repository_root.resolve()
    ):
        result.extend(
            ["--volume", f"{output_root.resolve()}:{output_root.resolve()}:rw"]
        )
    if cpu is not None:
        result.extend(["--cpuset-cpus", str(cpu)])
    result.extend([image, "python3", "-m", "scripts.collect_configuration_results"])
    result.extend(command)
    return result


def probe_collector(
    repository_plugin: str,
    collector_image: str | None,
    solver_python: str,
    *,
    environment: dict[str, str],
    repository_root: Path = ROOT,
    pull_policy: str = "missing",
) -> dict:
    """Read the identity of the configured collector without solving an instance."""
    probe = ["probe", "--repository-plugin", repository_plugin]
    command = (
        _docker_collector_command(
            collector_image,
            probe,
            environment=environment,
            pull_policy=pull_policy,
            repository_root=repository_root,
        )
        if collector_image
        else [
            solver_python,
            "-m",
            "scripts.collect_configuration_results",
            *probe,
        ]
    )
    completed = subprocess.run(
        command,
        cwd=repository_root,
        env={**os.environ, **environment},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            completed.stderr.strip()[-500:]
            or f"collector probe exited {completed.returncode}"
        )
    lines = completed.stdout.strip().splitlines()
    if not lines:
        raise ValueError("collector probe returned no identity")
    identity = json.loads(lines[-1])
    if not isinstance(identity, dict) or not isinstance(identity.get("version"), str):
        raise ValueError("collector probe returned invalid identity")
    return identity


def _collector_environment(search: dict) -> dict[str, str]:
    saved = search.get("collector_environment")
    if saved is not None:
        return saved
    from pitbench.repositories.base import RepositoryPluginRegistry

    return dict(
        RepositoryPluginRegistry.load(search["repository_plugin"]).collector_environment
    )


class SmacSearch:
    """One outer trial per lowest-fidelity panel with signed costs."""

    def __init__(self, search: dict, directory: Path):
        import importlib.metadata

        from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer
        from smac import AlgorithmConfigurationFacade, Scenario

        version = importlib.metadata.version("smac")
        if version != search["searcher_version"]:
            raise ValueError(
                f"expected SMAC {search['searcher_version']}, found {version}"
            )
        space = ConfigurationSpace(seed=search["search_seed"])
        for name, domain in search["parameters"].items():
            common = {"name": name, "default": domain["default"]}
            if domain["type"] == "categorical":
                parameter = Categorical(items=domain["choices"], **common)
            elif domain["type"] in ("integer", "real"):
                constructor = Integer if domain["type"] == "integer" else Float
                parameter = constructor(
                    bounds=tuple(domain["bounds"]), log=False, **common
                )
            else:
                raise ValueError(f"unknown parameter type: {domain['type']}")
            space.add(parameter)
        scenario = Scenario(
            space,
            name="search",
            output_directory=directory / "smac",
            seed=search["search_seed"],
            n_trials=search["max_configurations"] + 1,
            deterministic=True,
            adaptive_capping=False,
        )
        # deterministic refers only to reusing the one fixed aggregate panel.
        # The inner solver still runs all explicitly recorded stochastic seeds.
        self.facade = AlgorithmConfigurationFacade(
            scenario,
            _external_panel,
            intensifier=AlgorithmConfigurationFacade.get_intensifier(
                scenario, max_config_calls=1
            ),
            overwrite=False,
            logging_level=40,
        )
        self.metadata = {
            "version": version,
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "configspace": importlib.metadata.version("ConfigSpace"),
            # Categorical forest bounds contain NaN in SMAC's metadata. Keep
            # these sentinels as text; experimental feedback stays finite/null.
            "scenario": json.loads(
                json.dumps(self.facade.scenario.meta), parse_constant=str
            ),
        }

    def ask(self):
        import numpy as np
        from smac.main.exceptions import ConfigurationSpaceExhaustedException

        running = self.facade.runhistory.get_running_trials()
        try:
            info = running[0] if running else self.facade.ask()
        except ConfigurationSpaceExhaustedException as error:
            raise SearchSpaceExhausted from error
        # Pin to the audited SMAC version; save before expensive external runs.
        self.facade._optimizer.save()
        return info, {
            name: value.item() if isinstance(value, np.generic) else value
            for name, value in dict(info.config).items()
        }

    def tell(self, info, degradation: float) -> None:
        from smac.runhistory.dataclasses import TrialValue

        self.facade.tell(info, TrialValue(cost=-degradation))


class SearchSpaceExhausted(Exception):
    """All distinct configurations in a finite configured space were evaluated."""


def configuration_panel(configuration, task, *, repository_root=ROOT):
    from pitbench.schema.task import PitBenchTask

    panels = [
        panel
        for panel in configuration["panels"]
        if PitBenchTask.from_yaml(
            repository_root / panel["task_config"]
        ).release.repository
        == task.release.repository
    ]
    if len(panels) > 1:
        raise ValueError("multiple parameter panels match the task repository")
    return panels[0] if panels else None


def prepare_candidate_configuration(
    request,
    config_path: Path,
    output: Path,
    *,
    cpus: list[int],
    search_seed: int,
    instances_per_panel: int | None = None,
    seeds_per_panel: int | None = None,
) -> dict:
    """Freeze the existing parameter domains against the candidate's native task.

    The input population is the current task's primary hidden panel, not a
    historical collector installation. No hidden feedback is sent to the Agent.
    """
    from adapters.pitbench.adapter import PitBenchAdapter
    from pitbench.evaluator.docker_judge import judge_harness_sha256
    from pitbench.evaluator.judge import JudgePlan, _evaluation_seeds
    from pitbench.evaluator.private_assets import PrivateAssetResolver
    from pitbench.repositories.base import RepositoryPluginRegistry
    from pitbench.schema.task import InstanceSetKind, PitBenchTask

    if any(
        value is not None and value < 1
        for value in (instances_per_panel, seeds_per_panel)
    ):
        raise ValueError("configuration smoke limits must be positive")
    configuration = read_data(config_path)
    context = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in request.evaluator_config.items()
        if not key.startswith("_")
    }
    task_path = Path(context["task_config_path"])
    task = PitBenchTask.from_yaml(task_path)
    patch_hash = hashlib.sha256(request.candidate_patch_path.read_bytes()).hexdigest()
    if patch_hash != request.candidate_patch_sha256:
        raise ValueError("candidate patch changed before configuration evaluation")
    PitBenchAdapter.validate_repository(task, Path(context["base_repository"]))
    identity = {
        "protocol": "native-candidate-configuration-v1",
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "task_sha256": hashlib.sha256(task_path.read_bytes()).hexdigest(),
        "patch_sha256": patch_hash,
        "judge_image": context.get("judge_image"),
        "harness_sha256": judge_harness_sha256(ROOT),
        "cpus": cpus,
        "memory": context.get("judge_memory") or "8g",
        "cpu_quota": context.get("judge_cpus"),
        "parallel_runs": context.get("judge_parallel_runs"),
        "tmpfs_size": context.get("judge_tmpfs_size") or "4g",
        "search_seed": search_seed,
        "instances_per_panel": instances_per_panel,
        "seeds_per_panel": seeds_per_panel,
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "experiment.json"
    if manifest_path.is_file():
        saved = read_data(manifest_path)
        if saved.get("identity") != identity:
            raise ValueError(
                "candidate configuration identity changed; use a new output directory"
            )
        return saved
    panel = configuration_panel(configuration, task)
    repository = RepositoryPluginRegistry.load(task.repository.plugin)
    manifest = {
        "identity": identity,
        "searches": [],
        "config": configuration,
        "cpus": cpus,
        "coverage": "available",
        "candidate_patch_sha256": patch_hash,
    }
    if panel is None or not repository.supports_fixed_solver_parameters:
        manifest.update(
            coverage="unavailable",
            reason="no approved parameter panel or native parameter driver for this repository",
        )
        write_json(manifest_path, manifest)
        return manifest
    resolver = PrivateAssetResolver(Path(context["private_root"]))
    seeds = list(_evaluation_seeds(task, resolver))
    seed_count = (
        len(seeds)
        if panel.get("native_only")
        else panel.get("solver_seed_count", configuration["solver_seed_count"])
    )
    if len(seeds) != seed_count:
        raise ValueError(
            "native seed count differs from the declared configuration protocol"
        )
    plan = JudgePlan.from_instance_set_configs(
        task, resolver, public_root=ROOT, generated_root=output / "inputs"
    )
    cases = [
        case
        for case in plan.cases
        if case.instance_set.kind == InstanceSetKind.JUDGE_ID
    ]
    instance_count = panel.get("instance_count", configuration["instance_count"])
    if len(cases) < instance_count:
        raise ValueError(
            "native hidden panel is smaller than the declared parameter panel"
        )
    instances = [
        {
            "id": f"{case.instance_set.name}/{case.instance_id}",
            "bks": case.anchor,
            "sha256": hashlib.sha256(case.path.read_bytes()).hexdigest(),
        }
        for case in cases[:instance_count]
    ]
    if instances_per_panel is not None:
        instances = instances[:instances_per_panel]
    if seeds_per_panel is not None:
        seeds = seeds[:seeds_per_panel]
    retest = list(seeds) if repository.deterministic else []
    generator = random.SystemRandom()
    while len(retest) < len(seeds):
        seed = generator.randrange(panel.get("seed_min", 0), panel["seed_max"] + 1)
        if seed not in seeds and seed not in retest:
            retest.append(seed)
    context.update(
        candidate_patch=str(request.candidate_patch_path.resolve()),
        build_cache_path=context.get("build_cache_path")
        or str(
            (
                Path(context.get("base_cache_path", ".pitbench/cache/base")).parent
                / "build"
            ).resolve()
            if context.get("use_base_cache")
            else output.parent / "configuration-builds"
        ),
    )
    manifest["native_context"] = context
    for budget in task.evaluation.budgets_sec:
        for state in ("base", "agent"):
            search = {
                **panel,
                "task_id": task.task_id,
                "task_configuration": task.model_dump(mode="json"),
                "id": f"{task.task_id}/budget-{budget:g}/{state}",
                "code_state": state,
                "solver": {
                    "source_commit": task.release.base_commit,
                    "source_tree": task.release.tree_sha,
                    "patch_sha256": patch_hash
                    if state == "agent"
                    else hashlib.sha256(b"").hexdigest(),
                },
                "native_context": context,
                "instances": instances,
                "solver_seeds": seeds,
                "retest_seeds": retest,
                "budget_sec": budget,
                "threads": task.evaluation.threads,
                "objective_sense": task.oracle.objective_sense,
                "deterministic": repository.deterministic,
                "search_seed": search_seed,
                "searcher_version": configuration["searcher_version"],
                "max_configurations": configuration["max_configurations"],
                "watchdog_grace_sec": configuration["watchdog_grace_sec"],
                "fixed_options": {
                    **(task.evaluation.fixed_solver_parameters or {}),
                    **panel.get("fixed_options", {}),
                },
                "search_mode": panel.get("search_mode", "full_enumeration"),
                "fidelity_stage_config": panel.get(
                    "fidelity_stages", configuration.get("fidelity_stages")
                ),
                "feedback_time_basis": "cpu_time",
                "instances_per_panel": instances_per_panel,
                "seeds_per_panel": seeds_per_panel,
            }
            if panel.get("native_only") and search["fidelity_stage_config"]:
                search["fidelity_stage_config"] = [
                    {
                        **stage,
                        "name": "full"
                        if index == len(search["fidelity_stage_config"]) - 1
                        else f"screen_{index + 1}",
                        "solver_seed_count": min(
                            stage["solver_seed_count"], len(seeds)
                        ),
                    }
                    for index, stage in enumerate(search["fidelity_stage_config"])
                ]
            search["fidelity_stages"] = materialize_fidelity_stages(search)
            validate_search(search)
            manifest["searches"].append(search)
    write_json(manifest_path, manifest)
    return manifest


class NativeConfigurationRunner:
    """Reuse the native Judge's builds, verifier and retained parameter cells."""

    def __init__(self, output: Path, cpus: list[int], *, retry_collection_errors=False):
        self.output, self.cpus = output, cpus
        self.retry_collection_errors = retry_collection_errors

    @staticmethod
    def _save_bank(bank, identity, observations):
        from pitbench.evaluator.storage import ObservationStore

        temporary = bank / ("observations-" + uuid4().hex + ".jsonl")
        ObservationStore.write_jsonl(temporary, observations)
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        snapshot = bank / f"observations-{digest}.jsonl"
        temporary.replace(snapshot)
        write_json(
            bank / "identity.json",
            {
                "identity": identity,
                "observations_sha256": digest,
                "snapshot": snapshot.name,
            },
        )
        return snapshot

    def panel(self, search, name, parameters, instances, seeds, fidelity_stage):
        from pitbench.evaluator.docker_judge import DockerJudge, judge_harness_sha256
        from pitbench.evaluator.evaluator import _measurement_hardware_identity
        from pitbench.evaluator.storage import ObservationStore, cache_lock
        from pitbench.schema.observation import CodeState

        context = search["native_context"]
        # Fidelity labels do not change a measurement. All stages share the
        # same source/parameter bank; only missing instance/seed cells run.
        identity = {
            "task": search["task_configuration"],
            "code_state": search["code_state"],
            "evidence_phase": "holdout"
            if fidelity_stage in {"retest_full", "independent_paired"}
            else "search",
            "solver": search["solver"],
            "parameters": {**search["fixed_options"], **parameters},
            "budget": search["budget_sec"],
            "image": context["judge_image"],
            "harness": judge_harness_sha256(ROOT),
            "hardware": _measurement_hardware_identity(self.cpus),
            "cpu_quota": context.get("judge_cpus") or len(self.cpus),
            "memory": context.get("judge_memory") or "8g",
            "tmpfs_size": context.get("judge_tmpfs_size") or "4g",
            "instances": search["instances"],
            "parallel_runs": min(
                int(context.get("judge_parallel_runs") or len(self.cpus)),
                max(
                    1,
                    int(context.get("judge_cpus") or len(self.cpus))
                    // search["threads"],
                ),
                max(1, len(self.cpus) // search["threads"]),
            ),
        }
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        cache_root = (
            Path(context["build_cache_path"]).parent / "configuration-measurements"
        )
        bank = cache_root / key
        bank.mkdir(parents=True, exist_ok=True)
        with cache_lock(bank / "lock"):
            saved = []
            snapshot = None
            if (bank / "identity.json").is_file():
                receipt = read_data(bank / "identity.json")
                snapshot = bank / receipt["snapshot"]
                if (
                    receipt.get("identity") != identity
                    or receipt.get("observations_sha256")
                    != hashlib.sha256(snapshot.read_bytes()).hexdigest()
                ):
                    raise ValueError("retained parameter measurements changed")
                saved = ObservationStore.read_jsonl(snapshot)
            expected = {(item["id"], seed) for item in instances for seed in seeds}
            selected = [
                row
                for row in saved
                if (f"{row.instance_set}/{row.instance_id}", row.solver_seed)
                in expected
                and not (
                    self.retry_collection_errors
                    and row.status.value == "infrastructure_error"
                )
            ]
            if len(selected) != len(expected):
                attempt = bank / ("attempt-" + uuid4().hex)
                attempt.mkdir()
                panel = {
                    "parameters": identity["parameters"],
                    "instances": instances,
                    "seeds": seeds,
                    "budget_sec": search["budget_sec"],
                    "reused_observations": [
                        row.model_dump(mode="json") for row in selected
                    ],
                }
                panel_path = attempt / "panel.json"
                write_json(panel_path, panel)
                judge = DockerJudge(
                    image=context["judge_image"],
                    task_config_path=Path(context["task_config_path"]),
                    base_repository=Path(context["base_repository"]),
                    private_root=Path(context["private_root"]),
                    candidate_patch=Path(context["candidate_patch"]),
                    output_dir=attempt,
                    cpus=identity["cpu_quota"],
                    memory=identity["memory"],
                    tmpfs_size=context.get("judge_tmpfs_size") or "4g",
                    cpuset_cpus=",".join(map(str, self.cpus)),
                    code_states=(CodeState(search["code_state"]),),
                    parallel_runs=identity["parallel_runs"],
                    build_cache_path=Path(context["build_cache_path"]),
                    configuration_panel=panel_path,
                )
                try:
                    selected = judge.run()
                finally:
                    partial = attempt / "judge-observations.jsonl"
                    if partial.is_file():
                        observed = ObservationStore.read_jsonl(partial)
                        merged = {
                            (row.instance_set, row.instance_id, row.solver_seed): row
                            for row in saved
                        }
                        for row in observed:
                            if (
                                f"{row.instance_set}/{row.instance_id}",
                                row.solver_seed,
                            ) not in expected:
                                raise ValueError(
                                    "partial parameter observations are outside the frozen panel"
                                )
                            merged[
                                row.instance_set, row.instance_id, row.solver_seed
                            ] = row
                        self._save_bank(bank, identity, list(merged.values()))
                if len(selected) != len(expected):
                    raise ValueError(
                        "native parameter panel did not return every requested cell"
                    )
                retained = [
                    row
                    for row in saved
                    if (f"{row.instance_set}/{row.instance_id}", row.solver_seed)
                    not in expected
                ]
                snapshot = self._save_bank(bank, identity, [*retained, *selected])
            write_json(
                self.output / search["id"] / name / f"{fidelity_stage}.reuse.json",
                {
                    "measurement_bank": str(bank),
                    "observations": str(snapshot),
                    "requested_cells": len(expected),
                    "requested_cpu_pool": self.cpus,
                    "observations_sha256": hashlib.sha256(
                        snapshot.read_bytes()
                    ).hexdigest(),
                },
            )
        records = [self.feedback_record(row) for row in selected]
        if search["code_state"] == "agent":
            for row in selected:
                if row.status.value == "invalid":
                    digest = hashlib.sha256(row.model_dump_json().encode()).hexdigest()
                    write_json(
                        self.output / "invalid-agent-solutions" / f"{digest}.json",
                        row.model_dump(mode="json"),
                    )
        write_json(
            self.output / search["id"] / name / f"{fidelity_stage}.results.json",
            records,
        )
        return records

    @staticmethod
    def feedback_record(row):
        return {
            "instance_id": f"{row.instance_set}/{row.instance_id}",
            "solver_seed": row.solver_seed,
            "budget_sec": row.budget_sec,
            "execution_status": "collector_error"
            if row.status.value == "infrastructure_error"
            else "returned"
            if row.status.value in {"completed", "no_solution"}
            else row.status.value,
            "termination": row.solver_termination,
            "solution_value_valid": row.valid,
            "solver_runtime_sec": row.cpu_time_sec,
            "verification": {"feasible": row.valid, "objective": row.objective},
            "observation": row.model_dump(mode="json"),
        }


def _write_native_comparisons(manifest, output):
    """Check both sources on the same selected configurations and holdout seeds."""
    from pitbench.metrics.configuration_report import paired_panel

    runner = NativeConfigurationRunner(output, manifest["cpus"])
    comparisons = []
    for budget in sorted({s["budget_sec"] for s in manifest["searches"]}):
        searches = {
            s["code_state"]: s
            for s in manifest["searches"]
            if s["budget_sec"] == budget
        }
        if any(
            not (output / search["id"] / "state.json").is_file()
            for search in searches.values()
        ):
            continue
        states = {
            state: read_data(output / search["id"] / "state.json")
            for state, search in searches.items()
        }
        if any(value.get("status") != "completed" for value in states.values()):
            continue
        defaults = {
            key: value["default"]
            for key, value in searches["base"]["parameters"].items()
        }
        parameters = [defaults]
        for state in ("base", "agent"):
            selected = states[state]["selected"]["parameters"]
            if selected not in parameters:
                parameters.append(selected)
        for index, values in enumerate(parameters):
            records = {
                state: runner.panel(
                    search,
                    f"paired-{index}",
                    values,
                    search["instances"],
                    search["retest_seeds"],
                    "independent_paired",
                )
                for state, search in searches.items()
            }
            search = searches["base"]
            result = paired_panel(
                records["base"],
                records["agent"],
                instances=search["instances"],
                seeds=search["retest_seeds"],
                budget=budget,
                feedback=search["feedback"],
                objective_sense=search["objective_sense"],
            )
            comparisons.append(
                {
                    "budget_sec": budget,
                    "parameters": values,
                    "feedback_time_basis": "cpu_time",
                    "interpretation": "Agent minus Base on identical parameters and independent seeds; negative is improvement",
                    **result,
                }
            )
    write_json(
        output / "candidate-parameter-comparison.json", {"comparisons": comparisons}
    )


def run_candidate_configuration(
    request,
    settings,
    *,
    output: Path | None = None,
    instances_per_panel: int | None = None,
    seeds_per_panel: int | None = None,
) -> Path | None:
    """Run parameter robustness against a captured patch without another Agent."""
    if not settings.enabled:
        return None
    if settings.scope != "candidate":
        raise ValueError(
            "candidate replay requires scope: candidate; use the standalone collector for release-only searches"
        )
    output = output or request.output_dir / settings.output_path
    pool = request.evaluator_config.get("judge_cpuset_cpus")
    cpus = settings.cpus or (
        [
            cpu
            for part in pool.split(",")
            for cpu in range(int(part.split("-")[0]), int(part.split("-")[-1]) + 1)
        ]
        if pool
        else sorted(os.sched_getaffinity(0))
    )
    if (
        not cpus
        or len(set(cpus)) != len(cpus)
        or not set(cpus) <= os.sched_getaffinity(0)
    ):
        raise ValueError("parameter evaluation CPU pool is unavailable")
    config_path = settings.experiment_config
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    manifest = prepare_candidate_configuration(
        request,
        config_path,
        output,
        cpus=cpus,
        search_seed=settings.search_seed,
        instances_per_panel=instances_per_panel,
        seeds_per_panel=seeds_per_panel,
    )
    if manifest["coverage"] != "available":
        from pitbench.metrics.configuration_report import write_configuration_report

        write_configuration_report(output)
        return output
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError(
            "Configuration Robustness requires uv with the configuration extra"
        )
    with (output / "controller.log").open("a") as log:
        subprocess.run(
            [
                uv,
                "run",
                "--extra",
                "configuration",
                "python",
                "-m",
                "scripts.collect_configuration_results",
                "run",
                "--output",
                str(output.resolve()),
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return output


def attach_configuration_report(
    payload: dict, directory: Path, *, output_root: Path
) -> None:
    """Keep the ordinary evaluation envelope linked to the parameter evidence."""
    from pitbench.evaluator.artifacts import artifact_ref

    path = directory / "configuration_report.json"
    report = read_data(path)
    comparisons = report.get("candidate_comparison", {}).get("comparisons", [])
    payload.setdefault("summary", {})["configuration_robustness"] = {
        "coverage": report.get("coverage", "available"),
        "reason": report.get("reason"),
        "invalid_agent_solution_count": report.get("invalid_agent_solution_count", 0),
        "searches": [
            {"id": row["id"], "status": row["status"]} for row in report["searches"]
        ],
        "comparisons": [
            {
                key: value
                for key, value in row.items()
                if key not in {"pairs", "by_instance"}
            }
            for row in comparisons
        ],
    }
    payload.setdefault("artifacts", {})["configuration_robustness_details"] = (
        artifact_ref(
            path, root=output_root, media_type="application/json", private=True
        ).model_dump(mode="json")
    )
    if report.get("invalid_agent_solution_count", 0):
        validity = payload.setdefault("validity", {"checks": []})
        check = {
            "code": "solution",
            "passed": False,
            "detail": "independently invalid candidate solutions under declared parameter settings",
        }
        if check not in validity["checks"]:
            validity["checks"].append(check)
        validity["accepted"] = False


def materialize_fidelity_stages(search: dict) -> list[dict]:
    """Choose and retain nested paired panels from the declared search seed."""
    if search["search_mode"] != "multi_fidelity":
        return []
    stage_configs = search.get("fidelity_stage_config")
    if not isinstance(stage_configs, list) or not stage_configs:
        raise ValueError("multi-fidelity search requires declared fidelity stages")
    instances = list(search["instances"])
    solver_seeds = list(search["solver_seeds"])
    instance_random = random.Random(search["search_seed"])
    seed_random = random.Random(search["search_seed"])
    instance_random.shuffle(instances)
    seed_random.shuffle(solver_seeds)
    stages = []
    previous_instance_count = 0
    previous_seed_count = 0
    previous_configuration_count = None
    names = set()
    for config in stage_configs:
        name = config.get("name")
        instance_count = config.get("instance_count")
        seed_count = config.get("solver_seed_count")
        if isinstance(seed_count, int) and search.get("seeds_per_panel") is not None:
            seed_count = min(seed_count, len(solver_seeds))
        configuration_count = config.get("configuration_count")
        if (
            not isinstance(name, str)
            or not name
            or name in names
            or not isinstance(instance_count, int)
            or not isinstance(seed_count, int)
            or not isinstance(configuration_count, int)
            or not 1 <= seed_count <= len(solver_seeds)
            or configuration_count < 1
            or seed_count < previous_seed_count
            or (
                previous_configuration_count is not None
                and configuration_count > previous_configuration_count
            )
        ):
            raise ValueError("invalid nested fidelity stage declaration")
        if instance_count > len(instances):
            if search.get("instances_per_panel") is None:
                raise ValueError("fidelity stage instance count exceeds the full panel")
            instance_count = len(instances)
        if instance_count < previous_instance_count:
            raise ValueError("fidelity stage instances must be nested")
        names.add(name)
        stages.append(
            {
                "name": name,
                "instances": instances[:instance_count],
                "solver_seeds": solver_seeds[:seed_count],
                "configuration_count": configuration_count,
            }
        )
        previous_instance_count = instance_count
        previous_seed_count = seed_count
        previous_configuration_count = configuration_count
    final_stage = stages[-1]
    if (
        len(final_stage["instances"]) != len(search["instances"])
        or len(final_stage["solver_seeds"]) != len(search["solver_seeds"])
        or stages[0]["configuration_count"] != search["max_configurations"]
    ):
        raise ValueError(
            "fidelity stages must end at the full panel and begin at max configurations"
        )
    return stages


def prepare(
    config_path: Path,
    output: Path,
    *,
    solver_pythons: dict[str, str],
    cpus: list[int],
    search_seed: int,
    instances_per_panel: int | None = None,
    seeds_per_panel: int | None = None,
) -> dict:
    from pitbench.repositories.base import RepositoryPluginRegistry
    from pitbench.schema.task import PitBenchTask

    config = read_data(config_path)
    if (output / "experiment.json").exists():
        raise ValueError("experiment already prepared; run its saved manifest")
    if (
        not cpus
        or len(set(cpus)) != len(cpus)
        or not set(cpus) <= os.sched_getaffinity(0)
    ):
        raise ValueError("cpus must be distinct CPUs available to this process")
    if config["max_configurations"] < 1:
        raise ValueError("max_configurations must be positive")
    if instances_per_panel is not None and instances_per_panel < 1:
        raise ValueError("instances_per_panel must be positive")
    if seeds_per_panel is not None and seeds_per_panel < 1:
        raise ValueError("seeds_per_panel must be positive")
    output.mkdir(parents=True, exist_ok=True)
    searches = []
    for panel in config["panels"]:
        if panel.get("native_only"):
            continue
        task = PitBenchTask.from_yaml(ROOT / panel["task_config"]).model_dump(
            mode="json"
        )
        repository_plugin = task["repository"]["plugin"]
        repository = RepositoryPluginRegistry.load(repository_plugin)
        collector_image = config.get("collector_images", {}).get(panel["collector"])
        python = solver_pythons.get(panel["collector"], sys.executable)
        identity = probe_collector(
            repository_plugin,
            collector_image,
            python,
            environment=repository.collector_environment,
        )
        if identity["version"] != task["release"]["version"]:
            raise ValueError("installed solver version differs from task release")
        if identity.get("binding_git_hash") and not task["release"][
            "base_commit"
        ].startswith(identity["binding_git_hash"]):
            raise ValueError("installed solver revision differs from task release")
        seeds = read_data(ROOT / panel["seed_source"])
        for key in panel["seed_key"].split("."):
            seeds = seeds[key]
        seed_count = panel.get("solver_seed_count", config["solver_seed_count"])
        if len(seeds) != seed_count or len(set(seeds)) != len(seeds):
            raise ValueError("solver seed count or uniqueness differs from protocol")
        upper = panel["seed_max"]
        if any(type(seed) is not int or not 0 <= seed <= upper for seed in seeds):
            raise ValueError("solver seed outside declared domain")
        if seeds_per_panel is not None:
            seeds = seeds[:seeds_per_panel]
        if repository.deterministic:
            retest_seeds = list(seeds)
        else:
            retest_seeds = []
            generator = random.SystemRandom()
            while len(retest_seeds) < len(seeds):
                seed = generator.randrange(upper + 1)
                if seed not in seeds and seed not in retest_seeds:
                    retest_seeds.append(seed)
        from pitbench.instances.generate import prepare_collection_instances

        instances = prepare_collection_instances(
            ROOT / panel["instance_source"],
            output / "inputs" / task["task_id"],
            path_template=panel.get("instance_path_template"),
        )
        instance_count = panel.get("instance_count", config["instance_count"])
        if len(instances) != instance_count or len(
            {item["id"] for item in instances}
        ) != len(instances):
            raise ValueError("instance panel size or uniqueness differs from protocol")
        if instances_per_panel is not None:
            instances = instances[:instances_per_panel]
        if set(panel.get("fixed_options", {})) & set(panel["parameters"]):
            raise ValueError("a searched parameter cannot also be fixed")
        for budget in task["evaluation"]["budgets_sec"]:
            search = {
                **panel,
                "repository_plugin": repository_plugin,
                "collector_environment": dict(repository.collector_environment),
                "task_id": task["task_id"],
                "task_configuration": task,
                "id": f"{task['task_id']}/budget-{budget:g}",
                "solver": identity,
                "deterministic": repository.deterministic,
                "collector_image": collector_image,
                "solver_python": python,
                "instances": instances,
                "solver_seeds": seeds,
                "retest_seeds": retest_seeds,
                "budget_sec": budget,
                "objective_sense": task["oracle"].get("objective_sense"),
                "threads": task["evaluation"]["threads"],
                "search_seed": search_seed,
                "searcher_version": config["searcher_version"],
                "max_configurations": config["max_configurations"],
                "watchdog_grace_sec": config["watchdog_grace_sec"],
                "fixed_options": panel.get("fixed_options", {}),
                "search_mode": panel.get("search_mode", "full_enumeration"),
                "fidelity_stage_config": panel.get(
                    "fidelity_stages", config.get("fidelity_stages")
                ),
                "instances_per_panel": instances_per_panel,
            }
            search["seeds_per_panel"] = seeds_per_panel
            search["fidelity_stages"] = materialize_fidelity_stages(search)
            validate_search(search)
            searches.append(search)
    manifest = {
        "config": config,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "search_seed": search_seed,
        "searches": searches,
        "cpus": cpus,
        "instances_per_panel": instances_per_panel,
        "seeds_per_panel": seeds_per_panel,
        "created_unix": time.time(),
        "environment": {"python": sys.version, "platform": platform.platform()},
    }
    write_json(output / "experiment.json", manifest)
    return manifest


def validate_search(search: dict) -> None:
    if type(search["threads"]) is not int or search["threads"] <= 0:
        raise ValueError("configuration protocol requires positive solver threads")
    if search["feedback"] not in ("normalized_gap", "capped_optimal_time"):
        raise ValueError("unknown configuration feedback")
    search_mode = search.get("search_mode", "full_enumeration")
    if search_mode not in {"full_enumeration", "multi_fidelity"}:
        raise ValueError("unknown configuration search mode")
    if search_mode == "multi_fidelity":
        stages = search.get("fidelity_stages")
        if not isinstance(stages, list) or not stages:
            raise ValueError("multi-fidelity search requires materialized stages")
        if stages[0]["configuration_count"] != search["max_configurations"]:
            raise ValueError("first fidelity stage must match max configurations")
        if {item["id"] for item in stages[-1]["instances"]} != {
            item["id"] for item in search["instances"]
        } or set(stages[-1]["solver_seeds"]) != set(search["solver_seeds"]):
            raise ValueError("final fidelity stage must be the complete search panel")
    seeds, retest = search["solver_seeds"], search["retest_seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or len(retest) != len(seeds):
        raise ValueError("search and retest must have equally sized unique seeds")
    if search.get("deterministic"):
        if retest != seeds:
            raise ValueError("deterministic retest must reuse the fixed seed panel")
    elif len(set(retest)) != len(seeds) or set(seeds) & set(retest):
        raise ValueError("stochastic search and retest seeds must be disjoint")
    if not math.isfinite(search["budget_sec"]) or search["budget_sec"] <= 0:
        raise ValueError("budget must be positive and finite")
    if search["feedback"] == "normalized_gap":
        if search.get("objective_sense") not in {"minimize", "maximize"}:
            raise ValueError("quality feedback requires an objective sense")
        if any(
            not isinstance(item.get("bks"), (int, float))
            or not math.isfinite(item["bks"])
            or item["bks"] == 0
            for item in search["instances"]
        ):
            raise ValueError("quality feedback requires finite nonzero BKS references")
    validate_parameter_domains(search["parameters"])


def validate_parameter_domains(parameters):
    if not isinstance(parameters, dict):
        raise ValueError("parameters must be a mapping")
    for name, domain in parameters.items():
        if domain["type"] == "categorical":
            if domain["default"] not in domain["choices"]:
                raise ValueError(f"default outside parameter domain: {name}")
        elif domain["type"] in ("integer", "real"):
            lower, upper = domain["bounds"]
            if not lower < upper or not lower <= domain["default"] <= upper:
                raise ValueError(f"invalid parameter domain: {name}")
        else:
            raise ValueError(f"unknown parameter type: {domain['type']}")


class ConfigurationRunner:
    def __init__(
        self,
        output: Path,
        cpus: list[int],
        *,
        retry_collection_errors: bool = False,
        reuse_collectors: bool = True,
    ):
        self.output, self.cpus = output, cpus
        self.retry_collection_errors = retry_collection_errors
        self.reuse_collectors = reuse_collectors
        self._collectors: dict[tuple, str] = {}
        self._collector_lock = Lock()
        if not cpus or not set(cpus) <= os.sched_getaffinity(0):
            raise ValueError("requested CPUs are unavailable")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        for name in list(self._collectors.values()):
            subprocess.run(
                ["docker", "rm", "--force", name], capture_output=True, check=False
            )
        self._collectors.clear()

    def _worker_command(self, search, worker_command, cpu, attempt):
        if not search.get("collector_image"):
            return [
                search["solver_python"],
                "-m",
                "scripts.collect_configuration_results",
                *worker_command,
            ]
        command = _docker_collector_command(
            search["collector_image"],
            worker_command,
            cpu=cpu,
            environment=_collector_environment(search),
            output_root=self.output,
        )
        if not self.reuse_collectors:
            return command
        key = (
            search["collector_image"],
            cpu,
            json.dumps(_collector_environment(search), sort_keys=True),
        )
        with self._collector_lock:
            name = self._collectors.get(key)
            if name is None:
                name = "pitbench-configuration-" + uuid4().hex
                image_index = command.index(search["collector_image"])
                create = command[:image_index]
                create[1] = "create"
                create.remove("--rm")
                tmpfs = create.index("--tmpfs")
                del create[tmpfs : tmpfs + 2]
                temporary = self.output.resolve() / "collector-tmp" / name
                temporary.mkdir(parents=True)
                temporary.chmod(0o1777)
                create.extend(
                    [
                        "--name",
                        name,
                        "--volume",
                        f"{temporary}:/tmp:rw",
                        "--entrypoint",
                        "/bin/sh",
                        search["collector_image"],
                        "-c",
                        "sleep infinity",
                    ]
                )
                subprocess.run(create, check=True, capture_output=True, text=True)
                self._collectors[key] = name
                try:
                    subprocess.run(
                        ["docker", "start", name],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                except Exception:
                    self._collectors.pop(key, None)
                    subprocess.run(
                        ["docker", "rm", "--force", name],
                        capture_output=True,
                        check=False,
                    )
                    raise
        temporary = attempt.resolve() / "tmp"
        temporary.mkdir(parents=True, exist_ok=True)
        return [
            "docker",
            "exec",
            "--env",
            f"TMPDIR={temporary}",
            "--env",
            f"TMP={temporary}",
            "--env",
            f"TEMP={temporary}",
            name,
            "python3",
            "-m",
            "scripts.collect_configuration_results",
            *worker_command,
        ]

    def panel(
        self,
        search: dict,
        name: str,
        parameters: dict,
        instances: list[dict],
        seeds: list[int],
        fidelity_stage: str,
    ) -> list[dict]:
        directory = self.output / search["id"] / name
        plan = directory / f"{fidelity_stage}.panel.json"
        if plan.exists():
            jobs = json.loads(plan.read_text())
            if any(job["parameters"] != parameters for job in jobs) or {
                job["solver_seed"] for job in jobs
            } != set(seeds):
                raise ValueError(
                    "saved panel differs from requested parameters or seeds"
                )
        else:
            jobs = [
                {
                    "instance": instance,
                    "instance_id": instance["id"],
                    "solver_seed": seed,
                    "budget_sec": search["budget_sec"],
                    "parameters": parameters,
                    "fixed_options": search["fixed_options"],
                    "threads": search["threads"],
                    "collector": search["collector"],
                    "repository_plugin": search["task_configuration"]["repository"][
                        "plugin"
                    ],
                    "solver": search["solver"],
                    "task_id": search["task_id"],
                    "source_commit": search["task_configuration"]["release"][
                        "base_commit"
                    ],
                    "panel": name,
                    "search_seed": search["search_seed"],
                }
                for instance in instances
                for seed in seeds
            ]
            random.Random(search["search_seed"]).shuffle(jobs)
            write_json(plan, jobs)
        collection_failed = Event()

        def execute(job, cpu):
            run = directory / "runs" / job["instance_id"] / f"seed-{job['solver_seed']}"
            attempts = sorted(run.glob("attempt-*/result.json"))
            if attempts:
                previous = json.loads(attempts[-1].read_text())
                if any(previous.get(key) != value for key, value in job.items()):
                    raise ValueError("saved run identity differs from requested job")
                if not (
                    self.retry_collection_errors
                    and previous.get("execution_status") == "collector_error"
                ):
                    if previous.get("execution_status") == "collector_error":
                        collection_failed.set()
                    return previous
            if collection_failed.is_set():
                return {
                    **job,
                    "execution_status": "not_run",
                    "error": "collection paused after infrastructure failure",
                }
            attempt = run / f"attempt-{len(list(run.glob('attempt-*'))):04d}"
            attempt.mkdir(parents=True)
            write_json(attempt / "job.json", job)
            worker_command = [
                "worker",
                "--job",
                str(attempt / "job.json"),
                "--cpu",
                str(cpu),
            ]
            command = self._worker_command(search, worker_command, cpu, attempt)
            result = run_collection_process(
                command,
                attempt,
                {**job, "cpu": cpu},
                timeout=search["budget_sec"] + search["watchdog_grace_sec"],
            )
            if result.get("execution_status") == "collector_error":
                collection_failed.set()
            if result.get("execution_status") == "watchdog_timeout" and search.get(
                "collector_image"
            ):
                # Terminating docker exec alone does not terminate its process
                # in the container. Discard only this CPU's worker container.
                key = (
                    search["collector_image"],
                    cpu,
                    json.dumps(_collector_environment(search), sort_keys=True),
                )
                name = self._collectors.pop(key, None)
                if name is not None:
                    subprocess.run(
                        ["docker", "rm", "--force", name],
                        capture_output=True,
                        check=False,
                    )
            return result

        records = run_collection_jobs(jobs, self.cpus, execute)
        write_json(directory / f"{fidelity_stage}.results.json", records)
        return records


def _run_full_enumeration_search(
    search: dict,
    directory: Path,
    evaluate,
    *,
    engine_factory=SmacSearch,
    retry_collection_errors: bool = False,
) -> dict:
    from pitbench.metrics.configuration_report import paired_panel

    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "state.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {"status": "running", "candidates": []}
    )
    if state["status"] in (
        "completed",
        "baseline_unavailable",
        "stopped_after_unavailable_feedback",
        "no_candidate",
    ):
        return state
    if state["status"] == "collection_paused" and not retry_collection_errors:
        return state
    defaults = {
        name: domain["default"] for name, domain in search["parameters"].items()
    }
    full_instances = search["instances"]
    full_solver_seeds = search["solver_seeds"]
    kwargs = {
        "budget": search["budget_sec"],
        "feedback": search["feedback"],
        "objective_sense": search.get("objective_sense"),
    }

    def summary(baseline, candidate, instances, seeds):
        return paired_panel(
            baseline, candidate, instances=instances, seeds=seeds, **kwargs
        )

    def save():
        write_json(state_path, state)

    baseline = evaluate(
        "default", defaults, full_instances, full_solver_seeds, "default_full"
    )
    baseline_summary = summary(baseline, baseline, full_instances, full_solver_seeds)
    state["default"] = baseline_summary
    if not baseline_summary["complete"]:
        state["status"] = (
            "collection_paused"
            if "collector_error" in baseline_summary["unavailable"]
            else "baseline_unavailable"
        )
        save()
        return state

    if not state.get("selected"):
        engine = engine_factory(search, directory)
        state["searcher"] = engine.metadata
        save()
        while len(state["candidates"]) < search["max_configurations"] or state.get(
            "pending"
        ):
            try:
                info, parameters = engine.ask()
            except SearchSpaceExhausted:
                break
            if parameters == defaults:
                engine.tell(info, 0.0)
                continue
            previous = next(
                (row for row in state["candidates"] if row["parameters"] == parameters),
                None,
            )
            if previous and previous["summary"]["complete"]:
                engine.tell(info, previous["summary"]["mean_degradation"])
                continue
            name = state.get("pending", {}).get(
                "name", f"candidate-{len(state['candidates']) + 1:04d}"
            )
            if state.get("pending") and state["pending"]["parameters"] != parameters:
                raise ValueError("SMAC pending trial differs from retained panel")
            state["pending"] = {"name": name, "parameters": parameters}
            save()
            records = evaluate(
                name, parameters, full_instances, full_solver_seeds, "full"
            )
            panel = summary(baseline, records, full_instances, full_solver_seeds)
            entry = {"name": name, "parameters": parameters, "summary": panel}
            if not panel["complete"] and "collector_error" in panel["unavailable"]:
                state["status"] = "collection_paused"
                state["pending"]["summary"] = panel
                save()
                return state
            if not previous:
                state["candidates"].append(entry)
            state.pop("pending", None)
            if not panel["complete"]:
                state.update(
                    selected=entry,
                    selection_reason="unavailable_feedback",
                    search_stopped=True,
                )
                save()
                break
            engine.tell(info, panel["mean_degradation"])
            save()
            print(
                f"{search['id']} {name}: degradation={panel['mean_degradation']:.6g}",
                flush=True,
            )
        if not state.get("selected"):
            if not state["candidates"]:
                state["status"] = "no_candidate"
                save()
                return state
            state.update(
                selected=max(
                    state["candidates"],
                    key=lambda row: row["summary"]["mean_degradation"],
                ),
                selection_reason="largest_observed_degradation",
                search_stopped=False,
            )
            save()

    state["status"] = "retesting"
    save()
    reference = evaluate(
        "retest-default",
        defaults,
        full_instances,
        search["retest_seeds"],
        "retest_full",
    )
    candidate = evaluate(
        "retest-selected",
        state["selected"]["parameters"],
        full_instances,
        search["retest_seeds"],
        "retest_full",
    )
    state["retest"] = summary(
        reference, candidate, full_instances, search["retest_seeds"]
    )
    if "collector_error" in state["retest"]["unavailable"]:
        state["status"] = "collection_paused"
    else:
        state["status"] = (
            "stopped_after_unavailable_feedback"
            if state["search_stopped"]
            else "completed"
        )
    save()
    return state


def _rank_fidelity_candidates(
    candidates: list[dict], fidelity_stage: str
) -> list[dict]:
    """Rank complete candidates stably by descending declared-stage degradation."""
    ranked = [
        candidate
        for candidate in candidates
        if candidate.get("fidelity", {}).get(fidelity_stage, {}).get("complete")
    ]
    return sorted(
        ranked,
        key=lambda candidate: candidate["fidelity"][fidelity_stage]["mean_degradation"],
        reverse=True,
    )


def _run_multi_fidelity_search(
    search: dict,
    directory: Path,
    evaluate,
    *,
    engine_factory=SmacSearch,
    retry_collection_errors: bool = False,
) -> dict:
    """Search on the lowest fidelity, then race declared nested panels."""
    from pitbench.metrics.configuration_report import paired_panel

    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "state.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {"status": "running", "candidates": [], "promotions": {}}
    )
    if state["status"] in (
        "completed",
        "baseline_unavailable",
        "stopped_after_unavailable_feedback",
        "no_candidate",
    ):
        return state
    if state["status"] == "collection_paused" and not retry_collection_errors:
        return state
    defaults = {
        name: domain["default"] for name, domain in search["parameters"].items()
    }
    full_instances = search["instances"]
    full_solver_seeds = search["solver_seeds"]
    stages = search["fidelity_stages"]
    feedback_kwargs = {
        "budget": search["budget_sec"],
        "feedback": search["feedback"],
        "objective_sense": search.get("objective_sense"),
    }

    def records_for_panel(records, instances, seeds):
        expected = {(instance["id"], seed) for instance in instances for seed in seeds}
        return [
            record
            for record in records
            if (record["instance_id"], record["solver_seed"]) in expected
        ]

    def summary(baseline, candidate, instances, seeds):
        return paired_panel(
            records_for_panel(baseline, instances, seeds),
            records_for_panel(candidate, instances, seeds),
            instances=instances,
            seeds=seeds,
            **feedback_kwargs,
        )

    def save():
        write_json(state_path, state)

    def stop_for_unavailable(entry, stage_name, panel):
        state.update(
            selected=entry,
            selection_reason="unavailable_feedback",
            search_stopped=True,
            unavailable_fidelity_stage=stage_name,
        )
        if "collector_error" in panel["unavailable"]:
            state["status"] = "collection_paused"
        save()

    baseline = evaluate(
        "default", defaults, full_instances, full_solver_seeds, "default_full"
    )
    baseline_summary = summary(baseline, baseline, full_instances, full_solver_seeds)
    state["default"] = baseline_summary
    if not baseline_summary["complete"]:
        state["status"] = (
            "collection_paused"
            if "collector_error" in baseline_summary["unavailable"]
            else "baseline_unavailable"
        )
        save()
        return state

    first_stage = stages[0]
    pending_stage = (state.get("pending") or {}).get("fidelity_stage")
    if not state.get("selected") and pending_stage in (None, first_stage["name"]):
        engine = engine_factory(search, directory)
        state["searcher"] = engine.metadata
        save()
        while len(state["candidates"]) < first_stage[
            "configuration_count"
        ] or state.get("pending"):
            try:
                info, parameters = engine.ask()
            except SearchSpaceExhausted:
                break
            if parameters == defaults:
                engine.tell(info, 0.0)
                continue
            previous = next(
                (row for row in state["candidates"] if row["parameters"] == parameters),
                None,
            )
            if previous and previous.get("fidelity", {}).get(
                first_stage["name"], {}
            ).get("complete"):
                engine.tell(
                    info, previous["fidelity"][first_stage["name"]]["mean_degradation"]
                )
                continue
            name = state.get("pending", {}).get(
                "name", f"candidate-{len(state['candidates']) + 1:04d}"
            )
            if state.get("pending") and (
                state["pending"]["parameters"] != parameters
                or state["pending"]["fidelity_stage"] != first_stage["name"]
            ):
                raise ValueError(
                    "SMAC pending trial differs from retained fidelity panel"
                )
            state["pending"] = {
                "name": name,
                "parameters": parameters,
                "fidelity_stage": first_stage["name"],
            }
            save()
            records = evaluate(
                name,
                parameters,
                first_stage["instances"],
                first_stage["solver_seeds"],
                first_stage["name"],
            )
            panel = summary(
                baseline,
                records,
                first_stage["instances"],
                first_stage["solver_seeds"],
            )
            entry = previous or {
                "name": name,
                "parameters": parameters,
                "fidelity": {},
            }
            entry["fidelity"][first_stage["name"]] = panel
            if not previous:
                state["candidates"].append(entry)
            state.pop("pending", None)
            if not panel["complete"]:
                stop_for_unavailable(entry, first_stage["name"], panel)
                break
            engine.tell(info, panel["mean_degradation"])
            save()
            print(
                f"{search['id']} {name}/{first_stage['name']}: "
                f"degradation={panel['mean_degradation']:.6g}",
                flush=True,
            )

    if not state.get("selected"):
        for stage_index, stage in enumerate(stages[1:], start=1):
            previous_stage = stages[stage_index - 1]
            promoted_names = state["promotions"].get(stage["name"])
            if promoted_names is None:
                ranked = _rank_fidelity_candidates(
                    state["candidates"], previous_stage["name"]
                )
                promoted_names = [
                    candidate["name"]
                    for candidate in ranked[: stage["configuration_count"]]
                ]
                state["promotions"][stage["name"]] = promoted_names
                save()
            for candidate_name in promoted_names:
                entry = next(
                    candidate
                    for candidate in state["candidates"]
                    if candidate["name"] == candidate_name
                )
                if entry.get("fidelity", {}).get(stage["name"], {}).get("complete"):
                    continue
                state["pending"] = {
                    "name": candidate_name,
                    "parameters": entry["parameters"],
                    "fidelity_stage": stage["name"],
                }
                save()
                records = evaluate(
                    candidate_name,
                    entry["parameters"],
                    stage["instances"],
                    stage["solver_seeds"],
                    stage["name"],
                )
                panel = summary(
                    baseline, records, stage["instances"], stage["solver_seeds"]
                )
                entry["fidelity"][stage["name"]] = panel
                state.pop("pending", None)
                if not panel["complete"]:
                    stop_for_unavailable(entry, stage["name"], panel)
                    break
                save()
                print(
                    f"{search['id']} {candidate_name}/{stage['name']}: "
                    f"degradation={panel['mean_degradation']:.6g}",
                    flush=True,
                )
            if state.get("selected"):
                break

    if not state.get("selected"):
        finalists = _rank_fidelity_candidates(state["candidates"], stages[-1]["name"])
        if not finalists:
            state["status"] = "no_candidate"
            save()
            return state
        state.update(
            selected=finalists[0],
            selection_reason="largest_full_fidelity_degradation",
            search_stopped=False,
        )
        save()

    state["status"] = "retesting"
    save()
    reference = evaluate(
        "retest-default",
        defaults,
        full_instances,
        search["retest_seeds"],
        "retest_full",
    )
    candidate = evaluate(
        "retest-selected",
        state["selected"]["parameters"],
        full_instances,
        search["retest_seeds"],
        "retest_full",
    )
    state["retest"] = summary(
        reference, candidate, full_instances, search["retest_seeds"]
    )
    if "collector_error" in state["retest"]["unavailable"]:
        state["status"] = "collection_paused"
    else:
        state["status"] = (
            "stopped_after_unavailable_feedback"
            if state["search_stopped"]
            else "completed"
        )
    save()
    return state


def run_search(
    search: dict,
    directory: Path,
    evaluate,
    *,
    engine_factory=SmacSearch,
    retry_collection_errors: bool = False,
) -> dict:
    if search.get("search_mode", "full_enumeration") == "multi_fidelity":
        return _run_multi_fidelity_search(
            search,
            directory,
            evaluate,
            engine_factory=engine_factory,
            retry_collection_errors=retry_collection_errors,
        )
    return _run_full_enumeration_search(
        search,
        directory,
        evaluate,
        engine_factory=engine_factory,
        retry_collection_errors=retry_collection_errors,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--config", type=Path, required=True)
    preparation.add_argument("--output", type=Path, required=True)
    preparation.add_argument(
        "--solver-python", action="append", default=[], metavar="COLLECTOR=PYTHON"
    )
    preparation.add_argument("--cpus", type=int, nargs="+", required=True)
    preparation.add_argument("--search-seed", type=int, required=True)
    preparation.add_argument("--instances-per-panel", type=int)
    preparation.add_argument("--seeds-per-panel", type=int)
    run = commands.add_parser("run")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument(
        "--search", action="append", help="Prepared search ID; default: all"
    )
    run.add_argument("--retry-collection-errors", action="store_true")
    report = commands.add_parser("report")
    report.add_argument("--output", type=Path, required=True)
    probe = commands.add_parser("probe")
    probe.add_argument("--repository-plugin", required=True)
    worker = commands.add_parser("worker")
    worker.add_argument("--job", type=Path, required=True)
    worker.add_argument("--cpu", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "probe":
        print(json.dumps(collection_backend(args.repository_plugin).identity()))
        return
    if args.command == "worker":
        configuration_worker(args.job, args.cpu)
        return
    output = args.output.resolve()
    if args.command == "prepare":
        pythons = {
            key: str(Path(value).absolute())
            for key, value in (item.split("=", 1) for item in args.solver_python)
        }
        manifest = prepare(
            args.config.resolve(),
            output,
            solver_pythons=pythons,
            cpus=args.cpus,
            search_seed=args.search_seed,
            instances_per_panel=args.instances_per_panel,
            seeds_per_panel=args.seeds_per_panel,
        )
        print(
            f"Prepared {len(manifest['searches'])} searches; no solver runs executed."
        )
        return
    from pitbench.metrics.configuration_report import write_configuration_report

    if args.command == "run":
        manifest = json.loads((output / "experiment.json").read_text())
        if args.search and not set(args.search) <= {
            item["id"] for item in manifest["searches"]
        }:
            parser.error("unknown prepared search ID")
        with ConfigurationRunner(
            output,
            manifest["cpus"],
            retry_collection_errors=args.retry_collection_errors,
        ) as runner:
            for search in manifest["searches"]:
                if args.search and search["id"] not in args.search:
                    continue
                validate_search(search)
                selected_runner = (
                    NativeConfigurationRunner(
                        output,
                        manifest["cpus"],
                        retry_collection_errors=args.retry_collection_errors,
                    )
                    if "native_context" in search
                    else runner
                )
                run_search(
                    search,
                    output / search["id"],
                    lambda name, params, instances, seeds, stage: selected_runner.panel(
                        search, name, params, instances, seeds, stage
                    ),
                    retry_collection_errors=args.retry_collection_errors,
                )
                write_configuration_report(output)
            if manifest.get("native_context"):
                _write_native_comparisons(manifest, output)
    report = write_configuration_report(output)
    for search in report["searches"]:
        print(
            f"{search['id']}: {search['status']}, {len(search['candidates'])} new configurations"
        )
    print(f"Report: {output / 'configuration_report.json'}")
