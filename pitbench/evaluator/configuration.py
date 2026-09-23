"""Fixed-panel parameter search; solver-specific collection lives in collection.py."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from threading import Event

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
) -> list[str]:
    """Run one collector action inside its pinned solver image."""
    result = [
        "docker",
        "run",
        "--rm",
        "--pull=missing",
        "--network",
        "none",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=1g,mode=1777",
        "--env",
        "PITBENCH_VROOM_SOLVER=/workspace/repo/bin/vroom",
        "--env",
        f"PYTHONPATH={ROOT}:/workspace/repo",
        "--volume",
        f"{ROOT}:{ROOT}:rw",
        "--volume",
        f"{ROOT / 'adapters/jvm'}:/opt/pitbench-jvm:ro",
        "--workdir",
        "/workspace/repo",
    ]
    if cpu is not None:
        result.extend(["--cpuset-cpus", str(cpu)])
    result.extend([image, "python3", "-m", "scripts.collect_configuration_results"])
    result.extend(command)
    return result


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
        raise ValueError("fidelity stages must end at the full panel and begin at max configurations")
    return stages


def prepare(
    config_path: Path,
    output: Path,
    *,
    solver_pythons: dict[str, str],
    cpus: list[int],
    search_seed: int,
    instances_per_panel: int | None = None,
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
    output.mkdir(parents=True, exist_ok=True)
    searches = []
    for panel in config["panels"]:
        task = PitBenchTask.from_yaml(ROOT / panel["task_config"]).model_dump(
            mode="json"
        )
        repository_plugin = task["repository"]["plugin"]
        repository = RepositoryPluginRegistry.load(repository_plugin)
        collector_image = config.get("collector_images", {}).get(panel["collector"])
        python = solver_pythons.get(panel["collector"], sys.executable)
        probe_command = [
            "probe",
            "--repository-plugin",
            repository_plugin,
        ]
        probe = subprocess.run(
            _docker_collector_command(collector_image, probe_command)
            if collector_image
            else [
                python,
                "-m",
                "scripts.collect_configuration_results",
                *probe_command,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        identity = json.loads(probe.stdout.strip().splitlines()[-1])
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
            search["fidelity_stages"] = materialize_fidelity_stages(search)
            validate_search(search)
            searches.append(search)
    manifest = {
        "config": config,
        "searches": searches,
        "cpus": cpus,
        "instances_per_panel": instances_per_panel,
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
        if (
            {item["id"] for item in stages[-1]["instances"]}
            != {item["id"] for item in search["instances"]}
            or set(stages[-1]["solver_seeds"]) != set(search["solver_seeds"])
        ):
            raise ValueError("final fidelity stage must be the complete search panel")
    seeds, retest = search["solver_seeds"], search["retest_seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or len(retest) != len(seeds):
        raise ValueError(
            "search and retest must have equally sized unique seeds"
        )
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
    for name, domain in search["parameters"].items():
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
        self, output: Path, cpus: list[int], *, retry_collection_errors: bool = False
    ):
        self.output, self.cpus = output, cpus
        self.retry_collection_errors = retry_collection_errors
        if not cpus or not set(cpus) <= os.sched_getaffinity(0):
            raise ValueError("requested CPUs are unavailable")

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
            command = (
                _docker_collector_command(
                    search["collector_image"], worker_command, cpu=cpu
                )
                if search.get("collector_image")
                else [
                    search["solver_python"],
                    "-m",
                    "scripts.collect_configuration_results",
                    *worker_command,
                ]
            )
            result = run_collection_process(
                command,
                attempt,
                {**job, "cpu": cpu},
                timeout=search["budget_sec"] + search["watchdog_grace_sec"],
            )
            if result.get("execution_status") == "collector_error":
                collection_failed.set()
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
    baseline_summary = summary(
        baseline, baseline, full_instances, full_solver_seeds
    )
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
            panel = summary(
                baseline, records, full_instances, full_solver_seeds
            )
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
        if candidate.get("fidelity", {})
        .get(fidelity_stage, {})
        .get("complete")
    ]
    return sorted(
        ranked,
        key=lambda candidate: candidate["fidelity"][fidelity_stage][
            "mean_degradation"
        ],
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
    baseline_summary = summary(
        baseline, baseline, full_instances, full_solver_seeds
    )
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
        while len(state["candidates"]) < first_stage["configuration_count"] or state.get(
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
            if previous and previous.get("fidelity", {}).get(first_stage["name"], {}).get(
                "complete"
            ):
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
                raise ValueError("SMAC pending trial differs from retained fidelity panel")
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
        runner = ConfigurationRunner(
            output,
            manifest["cpus"],
            retry_collection_errors=args.retry_collection_errors,
        )
        for search in manifest["searches"]:
            if args.search and search["id"] not in args.search:
                continue
            validate_search(search)
            run_search(
                search,
                output / search["id"],
                lambda name, params, instances, seeds, stage: runner.panel(
                    search, name, params, instances, seeds, stage
                ),
                retry_collection_errors=args.retry_collection_errors,
            )
            write_configuration_report(output)
    report = write_configuration_report(output)
    for search in report["searches"]:
        print(
            f"{search['id']}: {search['status']}, {len(search['candidates'])} new configurations"
        )
    print(f"Report: {output / 'configuration_report.json'}")
