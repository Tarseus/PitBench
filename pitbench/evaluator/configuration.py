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


class SmacSearch:
    """One outer trial per complete panel, with signed costs and no racing."""

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

        running = self.facade.runhistory.get_running_trials()
        info = running[0] if running else self.facade.ask()
        # Pin to the audited SMAC version; save before expensive external runs.
        self.facade._optimizer.save()
        return info, {
            name: value.item() if isinstance(value, np.generic) else value
            for name, value in dict(info.config).items()
        }

    def tell(self, info, degradation: float) -> None:
        from smac.runhistory.dataclasses import TrialValue

        self.facade.tell(info, TrialValue(cost=-degradation))


def prepare(
    config_path: Path,
    output: Path,
    *,
    solver_pythons: dict[str, str],
    cpus: list[int],
    search_seed: int,
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
    output.mkdir(parents=True, exist_ok=True)
    searches = []
    for panel in config["panels"]:
        task = PitBenchTask.from_yaml(ROOT / panel["task_config"]).model_dump(
            mode="json"
        )
        repository_plugin = task["repository"]["plugin"]
        repository = RepositoryPluginRegistry.load(repository_plugin)
        python = solver_pythons[panel["collector"]]
        probe = subprocess.run(
            [
                python,
                "-m",
                "scripts.collect_configuration_results",
                "probe",
                "--repository-plugin",
                repository_plugin,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        identity = json.loads(probe.stdout)
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
            }
            validate_search(search)
            searches.append(search)
    manifest = {
        "config": config,
        "searches": searches,
        "cpus": cpus,
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
        self, search: dict, name: str, parameters: dict, seeds: list[int]
    ) -> list[dict]:
        directory = self.output / search["id"] / name
        plan = directory / "panel.json"
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
                for instance in search["instances"]
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
            result = run_collection_process(
                [
                    search["solver_python"],
                    "-m",
                    "scripts.collect_configuration_results",
                    "worker",
                    "--job",
                    str(attempt / "job.json"),
                    "--cpu",
                    str(cpu),
                ],
                attempt,
                {**job, "cpu": cpu},
                timeout=search["budget_sec"] + search["watchdog_grace_sec"],
            )
            if result.get("execution_status") == "collector_error":
                collection_failed.set()
            return result

        records = run_collection_jobs(jobs, self.cpus, execute)
        write_json(directory / "results.json", records)
        return records


def run_search(
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
    kwargs = {
        "instances": search["instances"],
        "budget": search["budget_sec"],
        "feedback": search["feedback"],
        "objective_sense": search.get("objective_sense"),
    }

    def summary(baseline, candidate, seeds):
        return paired_panel(baseline, candidate, seeds=seeds, **kwargs)

    def save():
        write_json(state_path, state)

    baseline = evaluate("default", defaults, search["solver_seeds"])
    baseline_summary = summary(baseline, baseline, search["solver_seeds"])
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
            info, parameters = engine.ask()
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
            records = evaluate(name, parameters, search["solver_seeds"])
            panel = summary(baseline, records, search["solver_seeds"])
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
    reference = evaluate("retest-default", defaults, search["retest_seeds"])
    candidate = evaluate(
        "retest-selected", state["selected"]["parameters"], search["retest_seeds"]
    )
    state["retest"] = summary(reference, candidate, search["retest_seeds"])
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--config", type=Path, required=True)
    preparation.add_argument("--output", type=Path, required=True)
    preparation.add_argument(
        "--solver-python", action="append", required=True, metavar="COLLECTOR=PYTHON"
    )
    preparation.add_argument("--cpus", type=int, nargs="+", required=True)
    preparation.add_argument("--search-seed", type=int, required=True)
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
                lambda name, params, seeds: runner.panel(search, name, params, seeds),
                retry_collection_errors=args.retry_collection_errors,
            )
            write_configuration_report(output)
    report = write_configuration_report(output)
    for search in report["searches"]:
        print(
            f"{search['id']}: {search['status']}, {len(search['candidates'])} new configurations"
        )
    print(f"Report: {output / 'configuration_report.json'}")
