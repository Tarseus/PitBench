"""Construct evaluator-owned Research Cases and explicit public projections.

This layer reuses existing task metrics, generators and independent verifiers.
It does not certify a new task, score a research report, or enforce compute quotas.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml

from pitbench.instances.generate import (
    materialize_generated_instance_set,
    materialize_instance_set,
    materialize_instance_suite,
)
from pitbench.schema.research import (
    GeneralizationStudyPlan,
    ResearchBudget,
    ResearchCasePlan,
)
from pitbench.schema.task import InstanceSetKind, PitBenchTask


def prepare_targeted_study_views(
    *,
    plan: GeneralizationStudyPlan,
    root: Path,
    destination: Path,
    evaluator_task_configs: dict[str, Path] | None = None,
    development_configs: dict[str, str] | None = None,
    scenario_configs: dict[str, Path] | None = None,
) -> Path:
    """Freeze problem-scoped inputs while sharing each task between Agent conditions."""
    from pitbench.tasks import TaskCatalog

    root, destination = root.resolve(), destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    evaluator_task_configs = evaluator_task_configs or {}
    development_configs = development_configs or {}
    scenario_configs = scenario_configs or {}
    catalog = TaskCatalog(root)
    views = {
        "kind": "problem_scoped_execution_views",
        "status": "development_frozen_evaluation_gated",
        "scientific_evaluation_ready": False,
        "tasks": {},
        "arms": {arm.id: {} for arm in plan.arms},
        "input_files_sha256": {},
    }
    for relative in plan.task_configs:
        original_record = catalog.validate_file(_public_asset(root, relative))
        original = original_record.task
        task_id = original.task_id
        source = evaluator_task_configs.get(task_id, original_record.task_config_path)
        task = catalog.validate_file(source).task.model_copy(deep=True)
        if not study_evaluation_matches(original, task, plan):
            raise ValueError("targeted study changed the frozen release or primary evaluation")
        target = plan.task_targets.get(task_id)
        if target is not None:
            task.problem_variant = target.problem_variant
            if target.formulation_id is not None:
                task.formulation_id = target.formulation_id
            task.instruction = target.instruction
        # Full source validation is retained after adding the task description.
        task = PitBenchTask.model_validate(task.model_dump())
        task_path = destination / "evaluator" / (task_id + ".yaml")
        task_path.parent.mkdir(exist_ok=True)
        task_path.write_text(yaml.safe_dump(task.model_dump(mode="json"), sort_keys=False))
        scenario = scenario_configs.get(task_id)
        if task.metric_scenarios and scenario is None:
            raise ValueError("declared scenarios need their source-bound runtime configuration")
        declared = {p.kind for p in task.metric_scenarios}
        public_panel = next(p for p in task.instance_sets if p.kind == InstanceSetKind.AGENT_DEV)
        metadata = {
            "evaluator_task_config": str(task_path),
            "evaluator_sha256": _sha(task_path),
            "scientific_panels": {
                p.name: {"instances": p.size}
                for p in task.instance_sets
                if p.name not in {q.name for q in original.instance_sets} and not p.scenario_only
            },
            "target_problem": task.problem_variant or task.problem_family.value,
            "primary_panel": "judge_id",
            "cross_problem_performance_role": "separate_transfer_reporting",
            "available_scenario_kinds": sorted(declared),
            "unavailable_scenario_kinds": sorted(set(plan.required_scenario_kinds) - declared),
            "development_original_size": public_panel.size,
        }
        views["input_files_sha256"][str(source.resolve())] = _sha(source)
        if scenario is not None:
            metadata["metric_scenario_config"] = str(scenario.resolve())
            views["input_files_sha256"][str(scenario.resolve())] = _sha(scenario)
        for arm in plan.arms:
            public = task.model_copy(deep=True)
            if task_id in development_configs:
                relative_config = development_configs[task_id]
                development = _public_asset(root, relative_config)
                payload = yaml.safe_load(development.read_text())
                panel = next(p for p in public.instance_sets if p.kind == InstanceSetKind.AGENT_DEV)
                panel.instance_set_config = relative_config
                panel.instance_set_config_sha256 = _sha(development)
                panel.size = len(payload["instances"])
                views["input_files_sha256"][str(development)] = _sha(development)
            public.instruction += "\n\n" + arm.instruction
            public_path = destination / "agent" / arm.id / (task_id + ".yaml")
            public_path.parent.mkdir(parents=True, exist_ok=True)
            public_path.write_text(yaml.safe_dump(public.model_dump(mode="json"), sort_keys=False))
            views["arms"][arm.id][task_id] = {
                "task_config_path": str(task_path),
                "agent_task_config_path": str(public_path),
                "candidate_hypotheses_required": arm.require_candidate_hypotheses,
            }
            metadata["development_diverse_size"] = next(
                p.size for p in public.instance_sets if p.kind == InstanceSetKind.AGENT_DEV
            )
        views["tasks"][task_id] = metadata
    path = destination / "task-views.json"
    _write(path, views)
    return path


def prepare_generalization_study(
    plan: GeneralizationStudyPlan, *, root: Path, destination: Path
) -> dict:
    """Materialize a controlled study design, not an assertion of run readiness."""
    from pitbench.repositories.base import BuildKind, RepositoryPluginRegistry

    root = root.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    manifest_path = _public_asset(root, plan.release_manifest)
    manifest = json.loads(manifest_path.read_text())
    pinned = {row["task_config"]: row for row in manifest["task_matrix"]}
    matrix_path = _public_asset(root, plan.model_matrix)
    models = yaml.safe_load(matrix_path.read_text())["runs"]
    if not models or len({row["label"] for row in models}) != len(models):
        raise ValueError("study model labels must be nonempty and unique")
    populations_path = _public_asset(root, plan.population_plan)
    populations = json.loads(populations_path.read_text())
    budget = resolve_research_budget(root, configured=plan.research_budget)
    references = json.loads((root / "configs/research/budgets.json").read_text())[
        "tracks"
    ]
    if budget.track in references and budget.model_dump(
        exclude_none=True
    ) != ResearchBudget.model_validate(references[budget.track]).model_dump(
        exclude_none=True
    ):
        raise ValueError("study budget differs from the shared named resource profile")
    gaps = []
    for role, split in populations["splits"].items():
        manifest_uri = split.get("manifest")
        if manifest_uri is None or not _public_asset(root, manifest_uri).is_file():
            gaps.append(f"population_manifest_missing:{role}")
    for key in ["headroom_evidence", "upstream_review_evidence", "scoring_plan"]:
        if populations.get(key) is None:
            gaps.append(key + "_missing")
    tasks = []
    for path in plan.task_configs:
        if path not in pinned:
            raise ValueError(
                f"study task is absent from the pinned release manifest: {path}"
            )
        task_path = _public_asset(root, path)
        task = PitBenchTask.from_yaml(task_path)
        selection = manifest["selected_releases"][pinned[path]["repository"]][
            pinned[path]["role"]
        ]
        if (
            task.release.base_commit != selection["commit_sha"]
            or task.release.tree_sha != selection["tree_sha"]
        ):
            raise ValueError("study source differs from the approved release pin")
        plugin = RepositoryPluginRegistry.load(task.repository.plugin)
        commands = plugin.build_commands(BuildKind.VALIDATION)
        command_text = " ".join(" ".join(c.argv) for c in commands)
        suite_declared = any(
            token in command_text
            for token in ["pytest", "ctest", "make test", "make check"]
        ) or any(
            command.argv
            and Path(command.argv[0]).name == "make"
            and any(arg in {"test", "check"} for arg in command.argv[1:])
            for command in commands
        )
        if not suite_declared:
            gaps.append(f"upstream_validation_suite_not_invoked:{task.task_id}")
        exclusions = [
            arg
            for command in commands
            for arg in command.argv
            if arg.startswith("--deselect=") or arg.startswith("--ignore=")
        ]
        if exclusions:
            gaps.append(f"upstream_test_exclusions_need_review:{task.task_id}")
        bindings = populations.get("scenario_bindings", {}).get(task.task_id, {})
        for kind in plan.required_scenario_kinds:
            if not bindings.get(kind):
                gaps.append(f"source_bound_scenario_missing:{task.task_id}:{kind}")
        tasks.append(
            {
                "task_id": task.task_id,
                "task_config": path,
                "task_config_sha256": _sha(task_path),
                "repository": task.release.repository,
                "source_commit": task.release.base_commit,
                "source_tree": task.release.tree_sha,
                "solve_budgets_sec": task.evaluation.budgets_sec,
                "solver_seeds": task.evaluation.solver_seeds,
                "validation_commands": [c.model_dump(mode="json") for c in commands],
                "upstream_suite_command_declared": suite_declared,
                "test_exclusions": exclusions,
            }
        )
    # Evidence files must be replayed by their native validators; mere JSON
    # fields or a ready:true flag cannot authorize this design's execution.
    gaps.append(
        "evaluator_revalidation_of_population_reference_and_scenario_evidence_required"
    )
    jobs = []
    for task, model, arm, attempt in itertools.product(
        tasks, models, plan.arms, range(1, plan.agent_attempts_per_condition + 1)
    ):
        jobs.append(
            {
                "condition_id": f"{plan.study_id}-{task['task_id']}-{model['label']}-{arm.id}-r{attempt}",
                "task_id": task["task_id"],
                "model": model,
                "arm": arm.id,
                "attempt": attempt,
                "research_budget": budget.model_dump(mode="json"),
                "candidate_hypotheses_required": arm.require_candidate_hypotheses,
                "development_view": arm.development_view,
                "status": "preparation_pending",
            }
        )
    destination.mkdir(parents=True)
    for task in tasks:
        for arm in plan.arms:
            # No hidden cells, reference patches, test seeds or headroom answers
            # are included in any arm's Agent-visible projection.
            view = {
                "study_id": plan.study_id,
                "status": "preparation_pending",
                "task_id": task["task_id"],
                "source_commit": task["source_commit"],
                "instruction": arm.instruction,
                "research_budget": budget.model_dump(mode="json"),
                "agent_tools": plan.shared_agent_tools,
                "development_view": arm.development_view,
                "candidate_hypotheses_required": arm.require_candidate_hypotheses,
            }
            _write(
                destination / "agent_views" / task["task_id"] / arm.id / "case.json",
                view,
            )
    receipt = {
        "kind": "generalization_study_design",
        "study_id": plan.study_id,
        "status": "preparation_pending",
        "ready_for_agent_execution": False,
        "claim_scope": plan.claim_scope,
        "inference_scope": plan.inference_scope,
        "model_configurations": len(models),
        "arms": len(plan.arms),
        "source_tasks": len(tasks),
        "agent_attempts_per_condition": plan.agent_attempts_per_condition,
        "planned_agent_runs": len(jobs),
        "maximum_agent_wall_hours": len(jobs) * budget.wall_time_sec / 3600,
        "tasks": tasks,
        "jobs": jobs,
        "readiness_gaps": gaps,
        "plan_sha256": hashlib.sha256(plan.model_dump_json().encode()).hexdigest(),
        "release_manifest_sha256": _sha(manifest_path),
        "population_plan_sha256": _sha(populations_path),
        "model_matrix_sha256": _sha(matrix_path),
        "new_agent_calls": 0,
        "new_solver_runs": 0,
    }
    _write(destination / "study.json", receipt)
    return receipt


def qualify_scenario_runtime(
    *,
    task_path: Path,
    binding_path: Path,
    specification_path: Path,
    destination: Path,
    python: str | None = None,
) -> dict:
    """Exercise the real collector on declared analytic API fixtures, not a study."""
    import copy
    import os
    import sys

    import numpy as np

    from pitbench.evaluator.representations import representation_type
    from pitbench.evaluator.scenarios import prepare, run
    from pitbench.repositories.base import RepositoryPluginRegistry
    from pitbench.schema.scenarios import SCENARIO_METRICS, ScenarioProtocol

    if destination.exists():
        raise FileExistsError(destination)
    specification = yaml.safe_load(specification_path.read_text())
    if specification["claim_scope"] != "implementation_qualification_only":
        raise ValueError(
            "API qualification requires an implementation-only specification"
        )
    task = PitBenchTask.from_yaml(task_path)
    plugin = RepositoryPluginRegistry.load(task.repository.plugin)
    binding = json.loads(binding_path.read_text())
    environment = {
        **os.environ,
        "PITBENCH_SCENARIO_BINDING": str(binding_path.resolve()),
    }
    # Probe in the selected interpreter, with exactly the same environment as collection.
    if python is None:
        executable = Path(binding["source"]) / plugin.driver_python
        python = str(executable) if "/" in plugin.driver_python else sys.executable
    probe = subprocess.run(
        [
            python,
            "-m",
            "scripts.collect_metric_scenarios",
            "probe",
            "--session-plugin",
            plugin.scenario_session_plugin,
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    identity = json.loads(probe.stdout.strip().splitlines()[-1])
    representations = [
        name
        for name in identity["representations"]
        if name in specification["representations"]
    ]
    if len(representations) != 1:
        raise ValueError(
            "qualification specification must select one supported representation"
        )
    representation = representations[0]
    adapter = representation_type(representation)
    cases = copy.deepcopy(specification["representations"][representation]["cases"])
    destination.mkdir(parents=True)
    for case_id, case in cases.items():
        for index, entry in enumerate(case["models"]):
            model = entry["model"]
            if "matrix" in model:
                import scipy.sparse

                model["matrix"] = scipy.sparse.csr_matrix(model["matrix"], dtype=float)
                model["sense"] = -1 if model["sense"] in {-1, "maximize"} else 1
                for key in (
                    "cost",
                    "col_lower",
                    "col_upper",
                    "row_lower",
                    "row_upper",
                    "integrality",
                ):
                    model[key] = np.asarray(
                        model[key], dtype=int if key == "integrality" else float
                    )
            path = destination / f"{case_id}-{index}{adapter.suffix}"
            adapter.write_input(path, model)
            entry.update(path=str(path.resolve()), sha256=_sha(path))
    patch = destination / "empty.patch"
    patch.write_bytes(b"")
    empty_hash = hashlib.sha256(b"").hexdigest()
    if binding["source_patch_sha256"] != empty_hash:
        raise ValueError("baseline API qualification requires an unchanged source tree")
    cpus = sorted(os.sched_getaffinity(0))[: specification["cpu_count"]]
    if len(cpus) != specification["cpu_count"]:
        raise ValueError("the declared qualification CPU pool is unavailable")
    panel = next(
        item for item in task.instance_sets if item.kind == InstanceSetKind.JUDGE_ID
    )
    fixture = task.model_copy(deep=True)
    fixture.task_id = task.task_id + "__api_qualification"
    next(item for item in fixture.instance_sets if item.name == panel.name).size = len(
        cases
    )
    fixture_path = destination / "fixture-task.yaml"
    fixture_path.write_text(
        yaml.safe_dump(fixture.model_dump(mode="json"), sort_keys=False)
    )
    receipt = {
        "kind": "scenario_api_qualification",
        "claim_scope": specification["claim_scope"],
        "task_id": task.task_id,
        "specification_sha256": _sha(specification_path),
        "binding_sha256": identity["binding_sha256"],
        "new_agent_calls": 0,
        "scenarios": {},
    }
    for kind, declaration in specification["scenarios"].items():
        settings = {**specification["default_protocol"], **declaration}
        indices = settings.pop("model_indices")
        protocol = ScenarioProtocol(
            id=kind,
            kind=kind,
            metrics=sorted(SCENARIO_METRICS[kind]),
            representation=representation,
            instance_sets=[panel.name],
            objective_sense=task.oracle.objective_sense,
            target_by_instance={
                key: value["models"][indices[0]]["target"]
                for key, value in cases.items()
            },
            reference_by_instance={
                key: value["models"][indices[0]]["target"]
                for key, value in cases.items()
            },
            loss_scale_by_instance={
                key: value["models"][indices[0]]["loss_scale"]
                for key, value in cases.items()
            },
            reference_solution_by_instance={
                key: value["witness"] for key, value in cases.items()
            },
            qualification_target_by_model_sha256={
                entry["sha256"]: entry["target"]
                for case in cases.values()
                for entry in case["models"]
            },
            model_sequences_sha256={
                panel.name: {
                    key: [value["models"][i]["sha256"] for i in indices]
                    for key, value in cases.items()
                }
            },
            **settings,
        )
        runtime = {
            "python": python,
            "session_plugin": plugin.scenario_session_plugin,
            "source_commit": task.release.base_commit,
            "patch_sha256": empty_hash,
            "expected_binding_sha256": identity["binding_sha256"],
            "options": specification.get("options_by_repository", {})
            .get(task.repository.plugin, {})
            .get(kind, {}),
            "environment": {"PITBENCH_SCENARIO_BINDING": str(binding_path.resolve())},
        }
        config = {
            "task_config": str(fixture_path.resolve()),
            "candidate_patch": str(patch.resolve()),
            "cpus": cpus,
            "watchdog_grace_sec": specification["watchdog_grace_sec"],
            "order": "base_first",
            "runtimes": {"base": runtime, "agent": runtime},
            "protocols": [protocol.model_dump(mode="json")],
            "instances": {
                panel.name: [
                    {"id": key, "models": [value["models"][i]["path"] for i in indices]}
                    for key, value in cases.items()
                ]
            },
        }
        config_path = destination / f"{kind}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        campaign_path = destination / kind
        try:
            prepare(config_path, campaign_path)
            result = run(campaign_path)
            campaign = json.loads((campaign_path / "campaign.json").read_text())
            metric_report = json.loads(
                (campaign_path / "scenario-report.json").read_text()
            )
            quality_failures = [
                value["metric_id"]
                for value in metric_report["results"]
                if value["complete"] and not value["quality_qualified"]
            ]
            unexpected = [
                record
                for record in campaign["runs"]
                if record["status"] not in {"returned", "unsupported"}
            ]
            receipt["scenarios"][kind] = {
                **result,
                "status": "failed"
                if unexpected or result["invalid_runs"] or quality_failures
                else "complete"
                if result["complete"]
                else "partial",
                "unsupported_runs": [
                    record["error"]
                    for record in campaign["runs"]
                    if record["status"] == "unsupported"
                ],
                "errors": [
                    {
                        "key": record["key"],
                        "status": record["status"],
                        "error": record["error"],
                    }
                    for record in unexpected
                ],
                "quality_failures": quality_failures,
            }
        except Exception as error:
            receipt["scenarios"][kind] = {"status": "failed", "error": str(error)}
        _write(destination / "api-qualification.json", receipt)
    return receipt


def construct_generalization_populations(
    *, specification_path: Path, destination: Path
) -> dict:
    destination.mkdir(parents=True, exist_ok=False)
    try:
        return _construct_generalization_populations(
            specification_path=specification_path, destination=destination
        )
    except Exception as error:
        _write(
            destination / "population-construction.json",
            {
                "kind": "generalization_population_construction",
                "status": "failed",
                "specification": str(specification_path),
                "error": f"{type(error).__name__}: {error}",
                "new_agent_calls": 0,
                "new_solver_runs": 0,
            },
        )
        raise


def audit_population_design(
    *, specification_path: Path, population_root: Path, root: Path, destination: Path
) -> dict:
    """Check actual exposure and claimed scale/structure splits before calibration."""
    from collections import defaultdict

    from pitbench.instances.generate import linear_model_features
    from pitbench.problem_families.verification import (
        _public_multidimensional_knapsack_model,
    )

    spec = yaml.safe_load(specification_path.read_text())
    construction = json.loads(
        (population_root / "population-construction.json").read_text()
    )
    if construction["specification_sha256"] != _sha(specification_path):
        raise ValueError("population specification changed after construction")
    conditions = {row["id"]: row for row in spec["conditions"]}
    requirements = spec["audit"]
    inventory = {
        role: json.loads(Path(split["manifest"]).read_text())["instances"]
        for role, split in construction["splits"].items()
    }
    exposed = []
    for entry in inventory[requirements["development_role"]]:
        family_spec = spec["families"][entry["family"]]
        family = family_spec["problem_family"]
        exposed.append(
            {
                "source": "diverse_development",
                "family": entry["family"],
                "problem_family": family,
                "scale_group": family_spec.get("scale_group", family),
                "features": entry["features"],
                "sha256": entry["sha256"],
            }
        )
    with tempfile.TemporaryDirectory(prefix="pitbench-exposure-") as tmp:
        for index, task_relative in enumerate(
            requirements["include_original_development_tasks"]
        ):
            task = PitBenchTask.from_yaml(root / task_relative)
            panel = next(
                s for s in task.instance_sets if s.kind == InstanceSetKind.AGENT_DEV
            )
            paths = materialize_instance_set(
                root / panel.instance_set_config, Path(tmp) / str(index)
            )
            for path in paths:
                if path.suffix == ".json":
                    model = json.loads(path.read_text())
                    features, family = {"customers": len(model["demands"]) - 1}, "cvrp"
                elif path.suffix.lower() == ".mps":
                    features, family = linear_model_features(path), "mip"
                else:
                    model = _public_multidimensional_knapsack_model(path)
                    if model is None:
                        raise ValueError("cannot audit original development model")
                    features, family = (
                        {"variables": len(model[0][0]), "constraints": len(model[0])},
                        "mip",
                    )
                exposed.append(
                    {
                        "source": task_relative,
                        "problem_family": family,
                        "scale_group": family,
                        "features": features,
                        "sha256": _sha(path),
                    }
                )
    maxima = {
        family: max(
            row["features"][feature]
            for row in exposed
            if row.get("scale_group", row["problem_family"]) == family
        )
        for family, feature in requirements["scale_feature_by_problem_family"].items()
    }
    seen_families = {row["family"] for row in exposed if "family" in row}
    seen_hashes = {row["sha256"] for row in exposed}
    cells = defaultdict(list)
    for role, entries in inventory.items():
        if role in {requirements["development_role"], "protocol_calibration"}:
            continue
        for entry in entries:
            cells[(role, entry["condition_id"])].append(entry)
    gaps, measured = [], []
    for (role, condition_id), entries in cells.items():
        condition = conditions[condition_id]
        family_spec = spec["families"][condition["family"]]
        family = family_spec.get("scale_group", family_spec["problem_family"])
        feature = requirements["scale_feature_by_problem_family"][family]
        values = [row["features"][feature] for row in entries]
        claim = condition["claim"]
        if (
            claim in {"scale", "joint"}
            and min(values)
            < maxima[family] * requirements["minimum_scale_extrapolation_ratio"]
        ):
            gaps.append(f"insufficient_scale_extrapolation:{condition_id}")
        if claim == "structure" and max(values) > maxima[family]:
            gaps.append(f"structure_cell_also_changes_scale:{condition_id}")
        if claim in {"structure", "joint"} and condition["family"] in seen_families:
            gaps.append(f"claimed_structure_already_exposed:{condition_id}")
        if any(row["sha256"] in seen_hashes for row in entries):
            gaps.append(f"hidden_content_already_exposed:{condition_id}")
        measured.append(
            {
                "role": role,
                "condition_id": condition_id,
                "claim": claim,
                "family": condition["family"],
                "scale_feature": feature,
                "minimum": min(values),
                "maximum": max(values),
                "development_maximum": maxima[family],
                "instances": len(entries),
                "minimum_extrapolation_ratio": min(values) / maxima[family],
            }
        )
    result = {
        "kind": "generalization_population_audit",
        "status": "failed" if gaps else "passed",
        "specification_sha256": _sha(specification_path),
        "development_maxima": maxima,
        "exposed_development_inputs": exposed,
        "cells": measured,
        "gaps": sorted(set(gaps)),
        "difficulty_qualification": "requires_empirical_calibration",
        "new_solver_runs": 0,
    }
    _write(destination, result)
    return result


def analyze_population_difficulty(
    *,
    population_root: Path,
    result_roots: list[Path],
    protocol_path: Path,
    destination: Path,
) -> dict:
    """Measure difficulty and select budgets from calibration discovery cases only."""
    import math
    from collections import defaultdict

    protocol = yaml.safe_load(protocol_path.read_text())
    construction = json.loads(
        (population_root / "population-construction.json").read_text()
    )
    inventory = [
        {**entry, "role": role}
        for role, split in construction["splits"].items()
        for entry in json.loads(Path(split["manifest"]).read_text())["instances"]
    ]
    instances = {entry["id"]: entry for entry in inventory}
    references, jobs, attempts, annotations = {}, {}, [], {}
    for root in result_roots:
        annotation_file = root / "diagnostic-annotations.json"
        if annotation_file.is_file():
            annotations.update(
                {
                    (str(root), key): value
                    for key, value in json.loads(annotation_file.read_text()).items()
                }
            )
        distribution = json.loads((root / "distribution.json").read_text())
        for job in distribution["original_manifest"]["jobs"]:
            jobs[(str(root), job["id"])] = job
        for identity, ref in json.loads((root / "bks.json").read_text())[
            "references"
        ].items():
            if (
                identity not in instances
                or ref["instance_sha256"] != instances[identity]["sha256"]
            ):
                raise ValueError("difficulty reference belongs to another population")
            if not ref.get("locally_verified"):
                raise ValueError(
                    "difficulty references require local independent verification"
                )
            old = references.get(identity)
            if old is None or (ref["objective"], not ref["proven_optimal"]) < (
                old["objective"],
                not old["proven_optimal"],
            ):
                references[identity] = ref
        for path in sorted((root / "hosts").glob("*/attempts.jsonl")):
            for line in path.read_text().splitlines():
                if line.strip():
                    attempts.append((str(root), json.loads(line)))
    latest = {(root, row["job_id"]): row for root, row in attempts}
    selected = defaultdict(dict)
    spec = yaml.safe_load(Path(protocol["specification"]).read_text())
    for entry in inventory:
        if entry["role"] != "protocol_calibration":
            continue
        family = spec["families"][entry["family"]]["problem_family"]
        for backend, backend_family in protocol["backend_problem_family"].items():
            if family != backend_family:
                continue
            for budget in protocol["budget_grid_sec"]:
                selected[(backend, entry["condition_id"])][(entry["id"], budget)] = {
                    "job_id": None,
                    "instance_id": entry["id"],
                    "budget_sec": budget,
                    "replicate": entry["replicate_index"],
                    "status": "missing",
                    "verified_feasible": False,
                    "target_hit": False,
                    "objective": None,
                    "solver_status": None,
                    "host": None,
                    "error": None,
                }
    for key, job in jobs.items():
        if job.get("purpose") != "calibration":
            continue
        backend = job["backend"]["name"]
        if job["seed"] != protocol["selection_seed_by_backend"].get(backend):
            continue
        if job.get("use_warm_start", True):
            raise ValueError("difficulty calibration must use cold solver starts")
        identity = job["instance"]["id"]
        row = latest.get(key)
        expected = instances[identity]
        if job["instance"]["sha256"] != expected["sha256"]:
            raise ValueError("calibration input differs from the frozen population")
        cell = {
            "job_id": job["id"],
            "instance_id": identity,
            "budget_sec": job["seconds"],
            "replicate": expected["replicate_index"],
            "status": row.get("execution_status") if row else "missing",
            "verified_feasible": bool(row and row.get("verified_feasible")),
            "target_hit": False,
            "objective": row.get("objective") if row else None,
            "solver_status": row.get("solver_status") if row else None,
            "host": row.get("host_id") if row else None,
            "error": row.get("error") if row else None,
        }
        ref = references.get(identity)
        if cell["verified_feasible"] and ref:
            objective = row["objective"]
            gap = (objective - ref["objective"]) / max(1, abs(ref["objective"]))
            cell.update(
                relative_gap_to_reference=gap,
                quality_target_hit=gap <= protocol["routing_target_relative_gap"],
            )
            if expected["generator"]["kind"] in {
                "euclidean_cvrp",
                "heterogeneous_fleet_vrp",
                "multi_depot_vrp",
                "open_vrp",
            }:
                cell.update(
                    relative_gap_to_reference=gap,
                    target_hit=gap <= protocol["routing_target_relative_gap"],
                )
            else:
                cell["target_hit"] = bool(
                    row.get("proven_optimal")
                    and math.isclose(
                        objective,
                        ref["objective"],
                        rel_tol=0,
                        abs_tol=protocol.get("calibration_objective_tolerance", 0),
                    )
                )
        native_status = str(
            (row or {}).get("optimality_basis", {}).get("solver_status", "")
        )
        if key in annotations:
            cell["retained_diagnostic"] = annotations[key]
            native_status = annotations[key]["native_status"]
        cell["native_status"] = native_status
        if (
            ref
            and "infeasib" in native_status.lower()
            and not cell["verified_feasible"]
        ):
            cell["status"] = "solver_status_contradiction"
        selected[(backend, expected["condition_id"])][(identity, job["seconds"])] = cell
    profiles = []
    grid = protocol["budget_grid_sec"]
    for (backend, condition), cells in sorted(selected.items()):
        by_budget = {}
        for budget in grid:
            values = [row for row in cells.values() if row["budget_sec"] == budget]
            split = {}
            for name in ("discovery", "qualification"):
                rows = [
                    row
                    for row in values
                    if row["replicate"] in protocol[name + "_replicates"]
                ]
                split[name] = {
                    "expected": len(rows),
                    "observed": sum(r["status"] != "missing" for r in rows),
                    "feasible": sum(r["verified_feasible"] for r in rows),
                    "target_hits": sum(r["target_hit"] for r in rows),
                    "quality_target_hits": sum(
                        r.get("quality_target_hit", False) for r in rows
                    ),
                    "failures": sum(
                        r["status"] not in {"completed", "missing"} for r in rows
                    ),
                    "missing": sum(r["status"] == "missing" for r in rows),
                }
            by_budget[str(budget)] = split
        chosen, tier = grid[-1], protocol["no_hit_tier"]
        for budget in grid:
            discovery = by_budget[str(budget)]["discovery"]
            if (
                discovery["expected"]
                and discovery["observed"] == discovery["expected"]
                and discovery["target_hits"]
                >= math.ceil(
                    discovery["expected"] * protocol["budget_selection_quantile"]
                )
            ):
                chosen = budget
                tier = protocol["tier_by_first_hit_budget"][str(budget)]
                break
        profiles.append(
            {
                "backend": backend,
                "condition_id": condition,
                "tier": tier,
                "selected_budget_sec": chosen,
                "by_budget": by_budget,
                "qualification_at_selected_budget": by_budget[str(chosen)][
                    "qualification"
                ],
                "runs": list(cells.values()),
            }
        )
    result = {
        "kind": "population_difficulty_calibration",
        "protocol_sha256": _sha(protocol_path),
        "status": "measured"
        if profiles
        and len(references) == len(instances)
        and all(
            row["status"] != "missing"
            for rows in selected.values()
            for row in rows.values()
        )
        else "partial",
        "profiles": profiles,
        "backend_by_task": protocol.get("backend_by_task", {}),
        "population_instances": len(instances),
        "instances_with_verified_reference": len(references),
        "solver_established_optima": sum(
            ref["proven_optimal"] for ref in references.values()
        ),
        "missing_references": sorted(set(instances) - set(references)),
        "retained_attempts": len(attempts),
        "historical_contract_changed": False,
        "selection_scope": "Calibration discovery only; qualification is reported without refitting. No hidden test outcome selects a budget.",
    }
    _write(destination, result)
    return result


def _construct_generalization_populations(
    *, specification_path: Path, destination: Path
) -> dict:
    """Assemble declared distributions through the existing generator suite."""
    import copy

    specification = yaml.safe_load(specification_path.read_text())
    panels, definitions = {}, {}
    for condition in specification["conditions"]:
        for role in condition["roles"]:
            role_spec = specification["roles"][role]
            generator = copy.deepcopy(
                specification["families"][condition["family"]]["generator"]
            )
            generator.update(condition.get("overrides", {}))
            generator["count"] = role_spec["count"]
            generator["randomness"] = {
                key: int.from_bytes(
                    hashlib.sha256(
                        f"{role_spec['seed_namespace']}:{condition['id']}:{key}:{value}".encode()
                    ).digest()[:8],
                    "big",
                )
                for key, value in generator["randomness"].items()
            }
            name = f"{role}__{condition['id']}"
            if name in panels:
                raise ValueError(
                    "population condition IDs must be unique within a role"
                )
            panels[name] = {
                "schema_version": "1.0",
                "visibility": role_spec["visibility"],
                "generator": generator,
            }
            definitions[name] = {
                "role": role,
                "family": condition["family"],
                "condition": condition,
            }
    suite_root = destination / "data"
    suite = materialize_instance_suite({"panels": panels}, suite_root)
    inventories = {role: [] for role in specification["roles"]}
    content_roles = {}
    namespaces = {}
    for name, definition in definitions.items():
        role = definition["role"]
        manifest_path = suite_root / name / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        generator = panels[name]["generator"]
        namespace = hashlib.sha256(
            json.dumps(generator["randomness"], sort_keys=True).encode()
        ).hexdigest()
        if namespace in namespaces:
            raise ValueError("population seed namespace is reused across conditions")
        namespaces[namespace] = name
        for cell in manifest["conditions"]:
            for entry in cell["instances"]:
                path = suite_root / name / entry["path"]
                if _sha(path) != entry["sha256"]:
                    raise ValueError(
                        "generated population content differs from its manifest"
                    )
                content = path.read_bytes()
                if path.suffix == ".json":
                    model = json.loads(content)
                    model.pop("name", None)
                    model.pop("node_ids", None)
                    content = json.dumps(
                        model, sort_keys=True, separators=(",", ":")
                    ).encode()
                semantic_hash = hashlib.sha256(content).hexdigest()
                if semantic_hash in content_roles:
                    raise ValueError(
                        f"duplicate population instance: {name} and {content_roles[semantic_hash]}"
                    )
                content_roles[semantic_hash] = name
                for feature, expected in (
                    definition["condition"].get("expected_features", {}).items()
                ):
                    if entry["features"].get(feature) != expected:
                        raise ValueError(
                            f"requested distribution feature was not realized: {name}:{feature}"
                        )
                for feature, allowed in definition["condition"].get(
                    "expected_feature_values", {}
                ).items():
                    if entry["features"].get(feature) not in allowed:
                        raise ValueError(
                            f"observed feature is outside declared levels: {name}:{feature}"
                        )
                for feature, bounds in definition["condition"].get(
                    "expected_feature_ranges", {}
                ).items():
                    observed = entry["features"].get(feature)
                    if (
                        not isinstance(bounds, list)
                        or len(bounds) != 2
                        or not isinstance(observed, (int, float))
                        or not bounds[0] <= observed <= bounds[1]
                    ):
                        raise ValueError(
                            f"observed feature is outside declared range: {name}:{feature}"
                        )
                inventories[role].append(
                    {
                        "id": f"{name}__{entry['id']}",
                        "path": str(path.resolve()),
                        "sha256": entry["sha256"],
                        "semantic_sha256": semantic_hash,
                        "source_group": f"{namespace}:{entry['replicate_index']}",
                        "family": definition["family"],
                        "condition_id": definition["condition"]["id"],
                        "axes": definition["condition"].get("axes", {}),
                        "features": entry["features"],
                        "generator": generator,
                        "replicate_index": entry["replicate_index"],
                    }
                )
    result = {
        "kind": "generalization_population_construction",
        "status": "generated_not_qualified",
        "specification_sha256": _sha(specification_path),
        "new_agent_calls": 0,
        "new_solver_runs": 0,
        "reference_status": "independent_headroom_qualification_required",
        "splits": {},
    }
    for role, instances in inventories.items():
        path = destination / role / "manifest.json"
        payload = {
            "role": role,
            "visibility": specification["roles"][role]["visibility"],
            "status": "generated_not_qualified",
            "instances": instances,
        }
        _write(path, payload)
        result["splits"][role] = {
            "manifest": str(path.resolve()),
            "sha256": _sha(path),
            "instances": len(instances),
        }
    result["instances"] = sum(len(items) for items in inventories.values())
    result["conditions"] = len(panels)
    result["suite_status"] = suite["status"]
    _write(destination / "population-construction.json", result)
    return result


def prepare_population_references(
    *, population_root: Path, task_path: Path, private_root: Path, destination: Path
) -> dict:
    """Freeze independently checked targets; feasible routing anchors stay nonoptimal."""
    from collections import defaultdict

    from pitbench.evaluator.private_assets import PrivateAssetResolver
    from pitbench.instances.generate import (
        prepare_trusted_optimum_oracle,
    )

    resolver = PrivateAssetResolver(private_root)
    destination = resolver.destination_path(destination)
    private_root = resolver.root
    if not destination.is_relative_to(private_root):
        raise ValueError("evaluator reference answers must remain in private storage")
    destination.mkdir(parents=True, exist_ok=False)
    construction = json.loads(
        (population_root / "population-construction.json").read_text()
    )
    result = {
        "kind": "population_reference_construction",
        "new_agent_calls": 0,
        "new_solver_runs": 0,
        "records": {},
        "panels": {},
        "status": "running",
        "headroom_status": "not_measured",
        "reference_search_required": [],
    }
    try:
        for role, inventory in construction["splits"].items():
            records = json.loads(Path(inventory["manifest"]).read_text())["instances"]
            groups = defaultdict(list)
            for entry in records:
                groups[entry["condition_id"]].append(entry)
            for condition, entries in groups.items():
                entries.sort(key=lambda entry: entry["replicate_index"])
                name = f"{role}__{condition}"
                folder = destination / name
                folder.mkdir()
                generator_kind = entries[0]["generator"]["kind"]
                constructive_mps_kinds = {
                    "multidimensional_knapsack_mps",
                    "random_set_covering_mip",
                    "facility_location_mip",
                    "graph_coloring_mip",
                    "single_machine_weighted_tardiness_mip",
                }
                if generator_kind in constructive_mps_kinds:
                    from pitbench.solver_drivers.reference import solve_reference

                    for entry in entries:
                        input_path = Path(entry["path"])
                        reference, checked = solve_reference(
                            input_path,
                            entry["generator"],
                            entry["replicate_index"],
                            seconds=0.1,
                        )
                        if not checked["feasible"]:
                            raise ValueError(
                                "constructive MPS reference failed source validation "
                                f"for {entry['id']}: {checked.get('detail')}"
                            )
                        solution_path = (
                            folder / f"{entry['replicate_index']:04d}.solution.json"
                        )
                        _write(solution_path, reference.solution)
                        result["records"][entry["id"]] = {
                            "target_kind": "feasible_reference",
                            "objective": checked["objective"],
                            "solution": str(solution_path.resolve()),
                            "solution_sha256": _sha(solution_path),
                            "instance_sha256": entry["sha256"],
                            "proven_optimal": False,
                            "optimality_basis": reference.basis,
                        }
                    result["panels"][name] = {
                        "status": "verified_feasibility_only",
                        "instances": len(entries),
                    }
                    _write(destination / "references.json", result)
                    continue
                analytic_mip_kinds = {
                    "single_machine_scheduling_mip",
                    "set_covering_mip",
                    "binary_clique_cover_mip",
                    "multidimensional_knapsack_mip",
                    "graph_coloring",
                    "balanced_subset_sum",
                    "bin_packing",
                }
                if (
                    Path(entries[0]["path"]).suffix in {".lp", ".mps"}
                    and generator_kind not in analytic_mip_kinds
                ):
                    for entry in entries:
                        result["records"][entry["id"]] = {
                            "target_kind": "reference_search_required",
                            "objective": None,
                            "instance_sha256": entry["sha256"],
                        }
                        result["reference_search_required"].append(entry["id"])
                    result["panels"][name] = {
                        "status": "reference_search_required",
                        "instances": len(entries),
                    }
                    _write(destination / "references.json", result)
                    continue
                if Path(entries[0]["path"]).suffix in {".lp", ".mps"}:
                    config = folder / "generator.yaml"
                    config.write_text(
                        yaml.safe_dump(
                            {
                                "visibility": "judge",
                                "generator": entries[0]["generator"],
                            }
                        )
                    )
                    oracle_path = folder / "oracle.yaml"
                    oracle = prepare_trusted_optimum_oracle(
                        task_config_path=task_path,
                        instance_set_name=name,
                        instance_set_config_path=config,
                        private_root=private_root,
                        output_path=oracle_path,
                        reference_solution_dir=folder / "solutions",
                        materialized_instances=[
                            Path(entry["path"]) for entry in entries
                        ],
                    )
                    for entry, target in zip(entries, oracle["records"], strict=True):
                        if entry["sha256"] != target["instance_sha256"]:
                            raise ValueError(
                                "reference target belongs to another population instance"
                            )
                        result["records"][entry["id"]] = {
                            "target_kind": "trusted_optimum",
                            "objective": target["optimal_objective"],
                            "record": target,
                        }
                    result["panels"][name] = {
                        "status": "verified_optimum",
                        "oracle": str(oracle_path),
                        "sha256": _sha(oracle_path),
                    }
                else:
                    from pitbench.solver_drivers.reference import solve_reference

                    for entry in entries:
                        input_path = Path(entry["path"])
                        reference, checked = solve_reference(
                            input_path,
                            entry["generator"],
                            entry["replicate_index"],
                            seconds=0.1,
                        )
                        solution = reference.solution
                        if not checked["feasible"]:
                            raise ValueError(
                                "feasible reference does not verify on the original model "
                                f"for {entry['id']}: {checked['detail']}"
                            )
                        path = folder / f"{entry['replicate_index']:04d}.solution.json"
                        _write(path, solution)
                        result["records"][entry["id"]] = {
                            "target_kind": "computed_optimum"
                            if reference.proven
                            else "feasible_reference",
                            "objective": checked["objective"],
                            "solution": str(path),
                            "solution_sha256": _sha(path),
                            "instance_sha256": entry["sha256"],
                            "proven_optimal": reference.proven,
                            "optimality_basis": reference.basis,
                        }
                    result["panels"][name] = {
                        "status": "verified_reference_witnesses",
                        "instances": len(entries),
                    }
                _write(destination / "references.json", result)
        result["status"] = (
            "reference_search_pending"
            if result["reference_search_required"]
            else "references_verified_headroom_pending"
        )
        _write(destination / "references.json", result)
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        _write(destination / "references.json", result)
        raise
    return result


def import_population_references(
    *,
    population_root: Path,
    result_roots: list[Path],
    private_root: Path,
    destination: Path,
) -> dict:
    """Import searched BKS/optima with original-model checks and bound provenance."""
    from pitbench.evaluator.private_assets import PrivateAssetResolver
    from pitbench.problem_families.verification import (
        TrustedOptimumFamily,
        TrustedOptimumRecord,
        VerificationResult,
    )
    from pitbench.solver_drivers.reference import verify_reference_file

    resolver = PrivateAssetResolver(private_root)
    destination = resolver.destination_path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    construction = json.loads(
        (population_root / "population-construction.json").read_text()
    )
    inventory = [
        entry
        for split in construction["splits"].values()
        for entry in json.loads(Path(split["manifest"]).read_text())["instances"]
    ]
    best = {}
    for root in result_roots:
        for identity, row in json.loads((root / "bks.json").read_text())[
            "references"
        ].items():
            old = best.get(identity)
            if old is None or (row["objective"], not row["proven_optimal"]) < (
                old["objective"],
                not old["proven_optimal"],
            ):
                best[identity] = row
    result = {
        "kind": "population_reference_construction",
        "status": "running",
        "records": {},
        "missing": [],
        "new_solver_runs": 0,
        "new_agent_calls": 0,
        "optimality_scope": "External solver optimal returns with independently checked original-model witnesses. Search proofs are retained as provenance, not independently replayed.",
    }
    for entry in inventory:
        identity, path = entry["id"], Path(entry["path"])
        row = best.get(identity)
        if row is None:
            result["missing"].append(identity)
            result["records"][identity] = {
                "target_kind": "missing",
                "instance_sha256": entry["sha256"],
            }
            continue
        if (
            not (
                row.get("locally_verified")
                or row.get("verified_feasible")
                or (row.get("verification") or {}).get("feasible")
            )
            or row["instance_sha256"] != entry["sha256"]
            or _sha(path) != entry["sha256"]
        ):
            raise ValueError(
                "imported reference lacks verified matching original input"
            )
        source_solution = Path(row["solution"])
        if _sha(source_solution) != row["solution_sha256"]:
            raise ValueError("reference solution changed after collection")
        checked = VerificationResult.model_validate(
            verify_reference_file(path, source_solution, entry["generator"]["kind"])
        )
        if not checked.feasible or checked.objective != row["objective"]:
            raise ValueError("searched reference fails original-model revalidation")
        solution_target = destination / (identity + ".solution.json")
        shutil.copy2(source_solution, solution_target)
        if path.suffix != ".mps":
            target = {
                "target_kind": "computed_optimum"
                if row["proven_optimal"]
                else "feasible_reference",
                "objective": checked.objective,
                "solution": str(solution_target),
                "solution_sha256": _sha(solution_target),
                "instance_sha256": entry["sha256"],
                "proven_optimal": bool(row["proven_optimal"]),
                "search_provenance": row,
            }
            cp_problem_kinds = {
                "bin_packing": "bin_packing",
                "graph_coloring": "graph_coloring",
                "job_shop_scheduling": "job_shop_scheduling",
                "resource_constrained_project_scheduling": "resource_constrained_project_scheduling",
                "balanced_subset_sum": "balanced_subset_sum",
            }
            problem_kind = cp_problem_kinds.get(entry["generator"]["kind"])
            if problem_kind is not None:
                target["record"] = TrustedOptimumRecord(
                    instance_set=entry["condition_id"],
                    instance_id=identity,
                    instance_sha256=entry["sha256"],
                    target_kind="best_found",
                    problem_kind=problem_kind,
                    objective_sense="minimize",
                    optimal_objective=checked.objective,
                    reference_solution_uri=(
                        "private://"
                        + solution_target.relative_to(resolver.root).as_posix()
                    ),
                    reference_solution_sha256=_sha(solution_target),
                    optimality_basis={
                        "kind": "feasible_search_reference",
                        "search_provenance": row,
                    },
                    generation_provenance={
                        "generator": entry["generator"],
                        "replicate_index": entry["replicate_index"],
                        "source_group": entry["source_group"],
                    },
                    verification_provenance=(
                        "pitbench.problem_families.verification:TrustedOptimumFamily"
                    ),
                ).model_dump(mode="json")
            result["records"][identity] = target
            continue
        basis = row.get("optimality_basis", {})
        proven = bool(
            row.get("proven_optimal") and basis.get("dual_bound") == checked.objective
        )
        target_kind = "trusted_optimum" if proven else "best_found"
        identity_record = {
            key: row.get(key)
            for key in [
                "backend",
                "worker_source_sha256",
                "code_revision",
                "remote_host",
            ]
        }
        proof = {
            "kind": "independent_mip_search" if proven else "feasible_search_reference",
            "solver_identity": identity_record,
            "reported_optimal": bool(row.get("proven_optimal")),
            "reported_primal": checked.objective,
            "reported_dual": basis.get("dual_bound"),
            "reference_result_sha256": hashlib.sha256(
                json.dumps(row, sort_keys=True).encode()
            ).hexdigest(),
        }
        record = TrustedOptimumRecord(
            instance_set=entry["condition_id"],
            instance_id=identity,
            instance_sha256=entry["sha256"],
            target_kind=target_kind,
            problem_kind="linear_mip",
            objective_sense="minimize",
            optimal_objective=checked.objective,
            reference_solution_uri="private://"
            + solution_target.relative_to(resolver.root).as_posix(),
            reference_solution_sha256=_sha(solution_target),
            optimality_basis=proof,
            generation_provenance={
                "generator": entry["generator"],
                "replicate_index": entry["replicate_index"],
                "source_group": entry["source_group"],
            },
            verification_provenance="pitbench.problem_families.verification:TrustedOptimumFamily",
        )
        if TrustedOptimumFamily(
            record, solution_target
        )._check_oracle(path) is not None:
            raise ValueError(
                "imported target failed reference/oracle consistency checks"
            )
        result["records"][identity] = {
            "target_kind": target_kind,
            "objective": checked.objective,
            "record": record.model_dump(mode="json"),
            "search_provenance": row,
        }
    result["status"] = "partial" if result["missing"] else "references_verified"
    _write(destination / "references.json", result)
    return result


def measure_population_headroom(
    *,
    task_path: Path,
    base_repository: Path,
    public_root: Path,
    private_root: Path,
    population_root: Path,
    references_path: Path,
    candidate_patch: Path,
    destination: Path,
    parallel_runs: int = 4,
    condition_ids: list[str] | None = None,
) -> dict:
    """Measure a fixed evaluator reference on disjoint calibration partitions."""
    from collections import Counter, defaultdict

    from pitbench.evaluator.judge import (
        InstanceCase,
        LocalProcessJudge,
        _exact_target_verifier,
    )
    from pitbench.evaluator.private_assets import PrivateAssetResolver
    from pitbench.metrics.performance_report import compute_panel_performance_reports
    from pitbench.problem_families.verification import (
        CVRPFamily,
        TrustedOptimumOracle,
        TrustedOptimumRecord,
    )
    from pitbench.schema.observation import ExpectedRunGrid
    from pitbench.schema.task import InstanceSetSpec

    if destination.exists():
        raise FileExistsError(destination)
    task = PitBenchTask.from_yaml(task_path).model_copy(deep=True)
    references = json.loads(references_path.read_text())
    construction = json.loads(
        (population_root / "population-construction.json").read_text()
    )
    manifest = json.loads(
        Path(construction["splits"]["protocol_calibration"]["manifest"]).read_text()
    )
    exact = task.task_type.value == "exact_solver"
    selected = [
        entry
        for entry in manifest["instances"]
        if (Path(entry["path"]).suffix in {".mps", ".lp"}) == exact
    ]
    groups = defaultdict(list)
    for entry in selected:
        groups[entry["condition_id"]].append(entry)
    if condition_ids is not None:
        if not condition_ids or set(condition_ids) - set(groups):
            raise ValueError(
                "headroom selection contains an unknown or empty condition list"
            )
        groups = {name: groups[name] for name in condition_ids}
    if not groups:
        raise ValueError("no calibration instances match this task")
    if any(
        references["records"].get(entry["id"], {}).get("target_kind", "missing")
        == "missing"
        for entries in groups.values()
        for entry in entries
    ):
        raise ValueError(
            "selected calibration conditions have missing reference targets"
        )
    destination.mkdir(parents=True)
    resolver = PrivateAssetResolver(private_root)
    private_root = resolver.root
    import time

    private_inputs = (
        private_root.resolve()
        / "research/headroom"
        / task.task_id
        / str(time.time_ns())
    )
    private_inputs.mkdir(parents=True, exist_ok=False)
    cases, panels, partitions, exact_records = [], [], {}, []
    for condition, entries in sorted(groups.items()):
        entries.sort(key=lambda entry: entry["replicate_index"])
        name = "judge_id" if entries[0]["axes"].get("axis") == "baseline" else condition
        panel_config = private_inputs / f"{condition}.yaml"
        panel = InstanceSetSpec(
            name=name,
            kind=InstanceSetKind.JUDGE_ID
            if name == "judge_id"
            else InstanceSetKind.JUDGE_SHIFT,
            instance_set_config="private://"
            + panel_config.relative_to(private_root.resolve()).as_posix(),
            size=len(entries),
            performance_protocol=(
                "heuristic_fixed_budget"
                if exact
                and any(
                    references["records"][entry["id"]]["target_kind"] == "best_found"
                    for entry in entries
                )
                else None
            ),
        )
        panels.append(panel)
        panel_rows = []
        for entry in entries:
            reference = references["records"][entry["id"]]
            if _sha(Path(entry["path"])) != entry["sha256"]:
                raise ValueError("calibration input changed after construction")
            input_path = private_inputs / condition / Path(entry["path"]).name
            input_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(entry["path"], input_path)
            if exact:
                record = TrustedOptimumRecord.model_validate(
                    reference["record"]
                ).model_copy(update={"instance_set": name, "instance_id": entry["id"]})
                exact_records.append(record)
                verifier = _exact_target_verifier(
                    record,
                    resolver,
                    performance_protocol=panel.performance_protocol
                    or task.evaluation.performance_protocol,
                )
            else:
                verifier = CVRPFamily()
            panel_rows.append(
                {
                    "id": entry["id"],
                    "uri": "private://"
                    + input_path.relative_to(private_root.resolve()).as_posix(),
                    "optimal_or_bks": reference["objective"],
                }
            )
            cases.append(
                InstanceCase(
                    instance_set=panel,
                    instance_id=entry["id"],
                    path=input_path,
                    anchor=reference["objective"],
                    condition_id=condition,
                    verifier=verifier,
                )
            )
            partitions[entry["id"]] = (
                "discovery"
                if entry["replicate_index"] < len(entries) // 2
                else "qualification"
            )
        panel_config.write_text(
            yaml.safe_dump(
                {
                    "schema_version": "1.0",
                    "visibility": "judge",
                    "instances": panel_rows,
                },
                sort_keys=False,
            )
        )
        panel.instance_set_config_sha256 = _sha(panel_config)
    task.instance_sets = [
        item for item in task.instance_sets if item.kind == InstanceSetKind.AGENT_DEV
    ] + panels
    # This is an evaluator-only per-panel opportunity diagnostic. Formal study
    # retention and representation protocols are not redefined by this task.
    task.evaluation.controlled_transfer = None
    if exact:
        oracle_path = private_inputs / "oracle.yaml"
        oracle_path.write_text(
            yaml.safe_dump(
                TrustedOptimumOracle(
                    task_id=task.task_id, records=exact_records
                ).model_dump(mode="json"),
                sort_keys=False,
            )
        )
        task.oracle.source = (
            "private://" + oracle_path.relative_to(private_root.resolve()).as_posix()
        )
        task.oracle.source_sha256 = _sha(oracle_path)
    task_file = destination / f"{task.task_id}.yaml"
    task_file.write_text(yaml.safe_dump(task.model_dump(mode="json"), sort_keys=False))
    receipt = {
        "kind": "headroom_calibration",
        "task_id": task.task_id,
        "new_agent_calls": 0,
        "candidate_patch_sha256": _sha(candidate_patch),
        "status": "running",
        "interpretation": "Evaluator reference intervention, not an experimental Agent attempt.",
        "partitions": {},
        "source_task_sha256": _sha(task_path),
        "selected_condition_ids": sorted(groups),
    }
    _write(destination / "headroom.json", receipt)
    try:
        judge = LocalProcessJudge(
            task=task,
            base_repository=base_repository,
            public_root=public_root,
            private_root=private_root,
            candidate_patch=candidate_patch,
            output_dir=destination,
            parallel_runs=parallel_runs,
            run_validation_builds=False,
            observations_path=destination / "observations.jsonl",
        )
        observations = judge.run(cases)
        grid = ExpectedRunGrid.model_validate_json(
            (destination / "expected-run-grid.json").read_text()
        )
        receipt["collection_by_panel"] = {
            panel.name: {
                state.value: {
                    "expected_runs": sum(
                        row.instance_set == panel.name and row.code_state == state
                        for row in grid.runs
                    ),
                    "observed_runs": len(rows),
                    "valid_solutions": sum(row.valid for row in rows),
                    "status_counts": dict(Counter(row.status.value for row in rows)),
                    "error_counts": dict(
                        Counter(row.error for row in rows if row.error)
                    ),
                }
                for state in judge.code_states
                for rows in [
                    [
                        row
                        for row in observations
                        if row.instance_set == panel.name and row.code_state == state
                    ]
                ]
            }
            for panel in panels
        }
        for partition in ("discovery", "qualification"):
            rows = [
                row for row in observations if partitions[row.instance_id] == partition
            ]
            expected = grid.model_copy(
                update={
                    "runs": [
                        row
                        for row in grid.runs
                        if partitions[row.instance_id] == partition
                    ]
                }
            )
            reports = compute_panel_performance_reports(
                rows, task=task, expected_run_grid=expected
            )
            receipt["partitions"][partition] = {
                name: report.model_dump(mode="json") for name, report in reports.items()
            }
        receipt["status"] = "measured"
        receipt["opportunity_by_panel"] = {
            panel.name: {
                "discovery": receipt["partitions"]["discovery"]
                .get(panel.name, {})
                .get("classification", "missing"),
                "qualification": receipt["partitions"]["qualification"]
                .get(panel.name, {})
                .get("classification", "missing"),
            }
            for panel in panels
        }
    except Exception as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        _write(destination / "headroom.json", receipt)
        raise
    _write(destination / "headroom.json", receipt)
    return receipt


def materialize_generalization_views(
    *,
    plan: GeneralizationStudyPlan,
    root: Path,
    population_root: Path,
    references_path: Path,
    private_root: Path,
    destination: Path,
    difficulty_path: Path | None = None,
) -> dict:
    """Keep a common evaluator task while varying only public development views."""
    import copy
    from collections import defaultdict

    from pitbench.evaluator.private_assets import PrivateAssetResolver
    from pitbench.problem_families.verification import (
        TrustedOptimumOracle,
        TrustedOptimumRecord,
    )
    from pitbench.schema.task import InstanceSetSpec, PerformanceProtocol
    from pitbench.tasks import TaskCatalog

    root, private_root, destination = (
        root.resolve(),
        private_root.resolve(),
        destination.resolve(),
    )
    private_root = PrivateAssetResolver(private_root).root
    destination.mkdir(parents=True, exist_ok=False)
    construction = json.loads(
        (population_root / "population-construction.json").read_text()
    )
    targets = json.loads(references_path.read_text())["records"]
    inventories = {
        role: json.loads(Path(value["manifest"]).read_text())["instances"]
        for role, value in construction["splits"].items()
    }
    if any(
        targets.get(entry["id"], {}).get("target_kind", "missing") == "missing"
        for entries in inventories.values()
        for entry in entries
    ):
        raise ValueError(
            "study views require reference coverage for every declared input; missing cases cannot be dropped"
        )
    development_roles = [
        role
        for role, split in construction["splits"].items()
        if json.loads(Path(split["manifest"]).read_text())["visibility"] == "agent"
    ]
    if len(development_roles) != 1:
        raise ValueError(
            "study views require one shared diverse-development population"
        )
    difficulty = json.loads(difficulty_path.read_text()) if difficulty_path else None
    if difficulty is not None and difficulty["status"] != "measured":
        raise ValueError("study budgets require complete calibration evidence")
    private_destination = (
        private_root / "research/generalization-views" / destination.name
    )
    private_destination.mkdir(parents=True, exist_ok=False)
    resolver = PrivateAssetResolver(private_root)
    result = {
        "kind": "generalization_task_views",
        "status": "materialized_pending_headroom_and_scenario_protocols",
        "new_agent_calls": 0,
        "tasks": {},
        "arms": {},
    }
    for task_relative in plan.task_configs:
        original = PitBenchTask.from_yaml(root / task_relative)
        task = original.model_copy(deep=True)
        exact = task.task_type.value == "exact_solver"
        target_generators = {
            "multidimensional_knapsack": {"multidimensional_knapsack_mps"},
            "set_covering": {"random_set_covering_mip"},
            "facility_location": {"facility_location_mip"},
            "scheduling": {"single_machine_weighted_tardiness_mip"},
            "single_machine_scheduling": {
                "single_machine_weighted_tardiness_mip"
            },
            "bin_packing": {"bin_packing"},
            "graph_coloring": {"graph_coloring", "graph_coloring_mip"},
            "job_shop_scheduling": {"job_shop_scheduling"},
            "resource_constrained_project_scheduling": {
                "resource_constrained_project_scheduling"
            },
            "balanced_subset_sum": {"balanced_subset_sum"},
            "cvrp": {"euclidean_cvrp"},
            "vrptw": {"euclidean_cvrp"},
            "heterogeneous_fleet_vrp": {"heterogeneous_fleet_vrp"},
            "multi_depot_vrp": {"multi_depot_vrp"},
            "open_vrp": {"open_vrp"},
        }
        expected_generators = target_generators.get(task.problem_variant or "")
        if expected_generators is None:
            raise ValueError(
                f"study task {task.task_id} has no population mapping for "
                f"problem variant {task.problem_variant!r}"
            )
        # The study owns its hidden populations. Keep only the public
        # development panel from the source task before adding v4 panels.
        task.instance_sets = [
            panel
            for panel in task.instance_sets
            if panel.kind == InstanceSetKind.AGENT_DEV or panel.scenario_only
        ]

        def relevant(entry):
            if entry["generator"]["kind"] not in expected_generators:
                return False
            if task.problem_variant in {"cvrp", "vrptw"}:
                return entry["condition_id"].startswith(task.problem_variant + "_")
            if task.problem_family.value == "mip":
                return Path(entry["path"]).suffix in {".lp", ".mps"}
            return exact or Path(entry["path"]).suffix not in {".lp", ".mps"}

        metadata = {}
        oracle_records = []
        if exact:
            oracle_records = list(
                TrustedOptimumOracle.from_yaml(
                    resolver.resolve(task.oracle.source, task.oracle.source_sha256)
                ).records
            )
        for role in inventories:
            if role in development_roles or role == "protocol_calibration":
                continue
            groups = defaultdict(list)
            for entry in inventories[role]:
                if relevant(entry):
                    groups[entry["condition_id"]].append(entry)
            for condition, entries in sorted(groups.items()):
                panel_name = role + "__" + condition
                folder = private_destination / task.task_id / panel_name
                folder.mkdir(parents=True)
                rows = []
                for entry in entries:
                    target = targets[entry["id"]]
                    path = folder / Path(entry["path"]).name
                    shutil.copy2(entry["path"], path)
                    if _sha(path) != entry["sha256"]:
                        raise ValueError("hidden population content changed")
                    rows.append(
                        {
                            "id": entry["id"],
                            "uri": "private://"
                            + path.relative_to(private_root).as_posix(),
                            "optimal_or_bks": target["objective"],
                        }
                    )
                    if exact:
                        oracle_records.append(
                            TrustedOptimumRecord.model_validate(
                                target["record"]
                            ).model_copy(
                                update={
                                    "instance_set": panel_name,
                                    "instance_id": entry["id"],
                                }
                            )
                        )
                config_path = folder / "instances.yaml"
                config_path.write_text(
                    yaml.safe_dump(
                        {
                            "schema_version": "1.0",
                            "visibility": "judge",
                            "instances": rows,
                        },
                        sort_keys=False,
                    )
                )
                panel_kind = (
                    InstanceSetKind.JUDGE_ID
                    if role == "unseen_id"
                    else InstanceSetKind.JUDGE_SHIFT
                )
                panel_protocol = (
                    "heuristic_fixed_budget"
                    if exact
                    and task.evaluation.performance_protocol
                    != PerformanceProtocol.BUDGET_TRUNCATED_QUALITY
                    and any(
                        targets[entry["id"]]["target_kind"] == "best_found"
                        for entry in entries
                    )
                    else None
                )
                task.instance_sets.append(
                    InstanceSetSpec(
                        name=panel_name,
                        kind=panel_kind,
                        instance_set_config="private://"
                        + config_path.relative_to(private_root).as_posix(),
                        instance_set_config_sha256=_sha(config_path),
                        size=len(rows),
                        performance_protocol=panel_protocol,
                    )
                )
                if difficulty is not None:
                    backend = difficulty["backend_by_task"][task.task_id]
                    profile = next(
                        profile
                        for profile in difficulty["profiles"]
                        if profile["backend"] == backend
                        and profile["condition_id"] == condition
                    )
                    selected_budget = profile["selected_budget_sec"]
                    panel = task.instance_sets[-1]
                    panel.budgets_sec = sorted(
                        set([original.evaluation.primary_budget_sec, selected_budget])
                    )
                    panel.primary_budget_sec = selected_budget
                metadata[panel_name] = {
                    "scientific_role": role,
                    "condition_id": condition,
                    "family": entries[0]["family"],
                    "axes": entries[0]["axes"],
                    "instances": len(rows),
                }
        if exact:
            path = private_destination / task.task_id / "oracle.yaml"
            path.write_text(
                yaml.safe_dump(
                    TrustedOptimumOracle(
                        task_id=task.task_id, records=oracle_records
                    ).model_dump(mode="json"),
                    sort_keys=False,
                )
            )
            task.oracle.source = (
                "private://" + path.relative_to(private_root).as_posix()
            )
            task.oracle.source_sha256 = _sha(path)
        evaluator_path = destination / "evaluator" / f"{task.task_id}.yaml"
        evaluator_path.parent.mkdir(parents=True, exist_ok=True)
        evaluator_path.write_text(
            yaml.safe_dump(task.model_dump(mode="json"), sort_keys=False)
        )
        original_dev = next(
            item
            for item in original.instance_sets
            if item.kind == InstanceSetKind.AGENT_DEV
        )
        public_folder = destination / "public-development" / task.task_id
        public_folder.mkdir(parents=True)
        materialize_instance_set(
            root / original_dev.instance_set_config, public_folder / "original"
        )
        index = yaml.safe_load(
            (public_folder / "original/instance_set_config.yaml").read_text()
        )
        public_rows = []
        if plan.comparison_mode != "agent_augmentation":
            for item in index["instances"]:
                data = public_folder / "original" / item["path"]
                solution = public_folder / "original" / item["bks_solution_file"]
                public_rows.append(
                    {
                        "id": item["id"],
                        "instance_file": data.relative_to(public_folder).as_posix(),
                        "instance_file_sha256": _sha(data),
                        "bks": item["bks"],
                        "bks_solution_file": solution.relative_to(public_folder).as_posix(),
                        "bks_solution_file_sha256": _sha(solution),
                        "proven_optimal": item.get("proven_optimal"),
                    }
                )
        for entry in inventories[development_roles[0]]:
            if not relevant(entry):
                continue
            target = targets[entry["id"]]
            folder = public_folder / "diverse"
            folder.mkdir(exist_ok=True)
            data = folder / (entry["id"] + Path(entry["path"]).suffix)
            shutil.copy2(entry["path"], data)
            solution = folder / (entry["id"] + ".solution.json")
            source_solution = (
                resolver.resolve(
                    target["record"]["reference_solution_uri"],
                    target["record"]["reference_solution_sha256"],
                )
                if exact
                else Path(target["solution"])
            )
            shutil.copy2(source_solution, solution)
            public_rows.append(
                {
                    "id": entry["id"],
                    "instance_file": data.relative_to(public_folder).as_posix(),
                    "instance_file_sha256": _sha(data),
                    "bks": target["objective"],
                    "bks_solution_file": solution.relative_to(public_folder).as_posix(),
                    "bks_solution_file_sha256": _sha(solution),
                    "proven_optimal": (
                        target["target_kind"] == "trusted_optimum"
                        if exact
                        else target.get("proven_optimal", False)
                    ),
                }
            )
        public_config = public_folder / "instances.yaml"
        public_config.write_text(
            yaml.safe_dump(
                {
                    "schema_version": "1.0",
                    "visibility": "agent",
                    "instances": public_rows,
                },
                sort_keys=False,
            )
        )
        diverse = copy.deepcopy(original_dev)
        diverse.instance_set_config = public_config.relative_to(root).as_posix()
        diverse.instance_set_config_sha256 = _sha(public_config)
        diverse.size = len(public_rows)
        result["tasks"][task.task_id] = {
            "evaluator_task_config": str(evaluator_path),
            "evaluator_sha256": _sha(evaluator_path),
            "scientific_panels": metadata,
            "original_primary_panels": "retained unchanged; new distributions are reported separately",
            "development_original_size": original_dev.size,
            "development_diverse_size": diverse.size,
        }
        TaskCatalog(root).validate_file(evaluator_path)
        for arm in plan.arms:
            view = task.model_copy(deep=True)
            view.instruction = original.instruction + "\n\n" + arm.instruction
            if plan.comparison_mode == "agent_augmentation" or arm.id != "original":
                view.instance_sets = [
                    diverse if item.kind == InstanceSetKind.AGENT_DEV else item
                    for item in view.instance_sets
                ]
            path = destination / "agent-views" / arm.id / f"{task.task_id}.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                yaml.safe_dump(view.model_dump(mode="json"), sort_keys=False)
            )
            TaskCatalog(root).validate_file(path)
            result["arms"].setdefault(arm.id, {})[task.task_id] = {
                "agent_task_config_path": str(path),
                "task_config_path": str(evaluator_path),
                "candidate_hypotheses_required": arm.require_candidate_hypotheses,
            }
    _write(destination / "task-views.json", result)
    return result


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def balance_generalization_task(
    task: PitBenchTask, *, panel_names: set[str], target_runs: int
) -> tuple[PitBenchTask, int, int]:
    """Balance supplemental panel sizes; count both source states and relabelings."""
    if target_runs < 1:
        raise ValueError("target_runs must be positive")
    from pitbench.evaluator.judge import _development_seeds

    task = task.model_copy(deep=True)
    development_seeds = len(_development_seeds(task))
    seeds = task.evaluation.seed_robustness
    hidden_seeds = seeds.seed_selection.seed_count if seeds else development_seeds
    panels = [p for p in task.instance_sets if p.name in panel_names]
    if len(panels) != len(panel_names) or any(p.scenario_only for p in panels):
        raise ValueError("balancing requires declared non-scenario panels")
    costs = {}
    for panel in task.instance_sets:
        if panel.scenario_only:
            continue
        public = panel.kind == InstanceSetKind.AGENT_DEV
        if (
            public
            and seeds is None
            and task.evaluation.representation_robustness is None
            and task.evaluation.controlled_transfer is None
        ):
            continue
        costs[panel.name] = (
            2
            * (
                panel.solver_seed_count
                if panel.solver_seed_count
                else development_seeds
                if public
                else hidden_seeds
            )
            * len(panel.budgets_sec or task.evaluation.budgets_sec)
        )
    count = sum(p.size * costs.get(p.name, 0) for p in task.instance_sets)
    representation = task.evaluation.representation_robustness
    if representation is not None:
        originals = next(
            p.size for p in task.instance_sets if p.name == representation.instance_set
        )
        count += (
            2
            * originals
            * representation.relabelings_per_instance
            * len(task.evaluation.budgets_sec)
        )
    original_count = count
    while count > target_runs:
        eligible = [p for p in panels if p.size > 1]
        if not eligible:
            raise ValueError("target_runs cannot retain every panel and primary run")
        panel = min(eligible, key=lambda p: (-p.size, p.name))
        panel.size -= 1
        count -= costs[panel.name]
    return task, original_count, count


def prepare_generalization_run_views(
    *,
    views_path: Path,
    private_root: Path,
    destination: Path,
    target_runs: int,
    supplemental_seed_count: int | None = None,
    supplemental_primary_budget_only: bool = False,
) -> Path:
    """Freeze similarly sized evaluation grids independently of solver outcomes."""
    import copy

    from pitbench.evaluator.private_assets import PrivateAssetResolver

    views_path, destination = views_path.resolve(), destination.resolve()
    identity = {"source_views_sha256": _sha(views_path), "target_runs": target_runs}
    if supplemental_seed_count is not None or supplemental_primary_budget_only:
        identity.update(
            supplemental_seed_count=supplemental_seed_count,
            supplemental_primary_budget_only=supplemental_primary_budget_only,
        )
    result_path = destination / "task-views.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result.get("run_policy") != identity:
            raise ValueError("frozen evaluation run policy or source views changed")
        for metadata in result["tasks"].values():
            if (
                _sha(Path(metadata["evaluator_task_config"]))
                != metadata["evaluator_sha256"]
            ):
                raise ValueError("frozen evaluation run view changed")
        return result_path
    result = copy.deepcopy(json.loads(views_path.read_text()))
    resolver = PrivateAssetResolver(private_root)
    namespace = hashlib.sha256(str(destination).encode()).hexdigest()[:12]
    configs = resolver.root / "research/evaluation-panels" / namespace
    destination.mkdir(parents=True, exist_ok=True)
    for task_id, metadata in result["tasks"].items():
        source = PitBenchTask.from_yaml(Path(metadata["evaluator_task_config"]))
        task, before, after = balance_generalization_task(
            source,
            panel_names=set(metadata["scientific_panels"]),
            target_runs=target_runs,
        )
        balanced_runs = after
        if supplemental_seed_count is not None and metadata["scientific_panels"]:
            available_seeds = (
                task.evaluation.seed_robustness.seed_selection.seed_count
                if task.evaluation.seed_robustness
                else len(task.evaluation.solver_seeds)
            )
            if not 0 < supplemental_seed_count <= available_seeds:
                raise ValueError("supplemental seed count exceeds the frozen seed list")
        for panel in task.instance_sets:
            if panel.name not in metadata["scientific_panels"]:
                continue
            if supplemental_seed_count is not None:
                panel.solver_seed_count = supplemental_seed_count
            if supplemental_primary_budget_only:
                if panel.primary_budget_sec is None:
                    raise ValueError(
                        "supplemental panel lacks a calibrated primary budget"
                    )
                panel.budgets_sec = [panel.primary_budget_sec]
        # Count the reduced grid without expanding the already selected panels.
        if supplemental_seed_count is not None or supplemental_primary_budget_only:
            task, _, after = balance_generalization_task(
                task,
                panel_names=set(metadata["scientific_panels"]),
                target_runs=target_runs,
            )
        selected = {}
        for panel, original in zip(
            task.instance_sets, source.instance_sets, strict=True
        ):
            if panel.size == original.size:
                continue
            payload = yaml.safe_load(
                resolver.resolve(
                    original.instance_set_config, original.instance_set_config_sha256
                ).read_text()
            )
            if len(payload.get("instances", [])) != original.size:
                raise ValueError("balancing requires a complete materialized panel")
            # Retain a nested prefix of the frozen input inventory, never rank
            # instances by solver performance or by completed-result availability.
            payload["instances"] = payload["instances"][: panel.size]
            path = configs / task_id / (panel.name + ".yaml")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(payload, sort_keys=False))
            panel.instance_set_config = (
                "private://" + path.relative_to(resolver.root).as_posix()
            )
            panel.instance_set_config_sha256 = _sha(path)
            selected[panel.name] = [row["id"] for row in payload["instances"]]
            metadata["scientific_panels"][panel.name]["instances"] = panel.size
        task_path = destination / "evaluator" / (task_id + ".yaml")
        task_path.parent.mkdir(exist_ok=True)
        task_path.write_text(
            yaml.safe_dump(task.model_dump(mode="json"), sort_keys=False)
        )
        metadata.update(
            evaluator_task_config=str(task_path),
            evaluator_sha256=_sha(task_path),
            run_balance={
                "original_runs": before,
                "balanced_runs": balanced_runs,
                "selected_runs": after,
                "selected_instance_ids": selected,
            },
        )
        changed = {
            p.name: p
            for p in task.instance_sets
            if p.name in metadata["scientific_panels"]
        }
        for arm, tasks in result["arms"].items():
            row = tasks[task_id]
            public = PitBenchTask.from_yaml(Path(row["agent_task_config_path"]))
            public.instance_sets = [
                changed.get(p.name, p) for p in public.instance_sets
            ]
            path = destination / "agent-views" / arm / (task_id + ".yaml")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                yaml.safe_dump(public.model_dump(mode="json"), sort_keys=False)
            )
            row.update(
                task_config_path=str(task_path), agent_task_config_path=str(path)
            )
    result["run_policy"] = identity
    _write(result_path, result)
    return result_path


def prepare_generalization_seed_views(
    *,
    plan: GeneralizationStudyPlan,
    views_path: Path,
    private_root: Path,
    destination: Path,
) -> Path:
    """Freeze a new study's hidden seeds without changing historical task files."""
    import copy
    import random

    from pitbench.evaluator.private_assets import (
        PrivateAssetResolver,
        load_private_seed_robustness_config,
    )
    from pitbench.schema.task import SeedRobustnessConfig, SeedSelectionConfig

    policy = plan.evaluation_seed_policy
    if policy is None and not plan.task_evaluation_seed_policies:
        return views_path
    identity = {
        "source_views_sha256": _sha(views_path),
        "policy": policy.model_dump(mode="json") if policy else None,
        "study_id": plan.study_id,
    }
    if plan.task_evaluation_seed_policies:
        identity["task_policies"] = {
            task_id: value.model_dump(mode="json")
            for task_id, value in plan.task_evaluation_seed_policies.items()
        }
    receipt_path = destination / "seed-policy.json"
    result_path = destination / "task-views.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["identity"] != identity:
            raise ValueError("frozen seed policy or source views changed")
        for filename, expected in receipt["files_sha256"].items():
            if _sha(Path(filename)) != expected:
                raise ValueError("frozen seed view changed")
        return result_path
    source = json.loads(views_path.read_text())
    result = copy.deepcopy(source)
    selected = [Path(path).stem for path in plan.task_configs]
    result["tasks"] = {key: result["tasks"][key] for key in selected}
    result["arms"] = {
        arm.id: {key: result["arms"][arm.id][key] for key in selected}
        for arm in plan.arms
    }
    destination.mkdir(parents=True, exist_ok=True)
    resolver = PrivateAssetResolver(private_root)
    files, counts = {}, {}
    for task_id in selected:
        policy = plan.task_evaluation_seed_policies.get(task_id, plan.evaluation_seed_policy)
        if policy is None:
            continue
        metadata = result["tasks"][task_id]
        task = PitBenchTask.from_yaml(Path(metadata["evaluator_task_config"]))
        current = task.evaluation.seed_robustness
        if current is not None and policy.preserve_existing:
            hidden = load_private_seed_robustness_config(
                resolver, task_id=task_id, public_config=current
            )
            if len(hidden.evaluation_seeds) != policy.seed_count:
                raise ValueError("existing hidden seed count differs from study policy")
            files[str(resolver.resolve(current.evaluation_seeds_file))] = (
                current.evaluation_seeds_file_sha256
            )
        else:
            development = (
                current.development_seeds if current else task.evaluation.solver_seeds
            )
            if not development:
                raise ValueError("study requires declared development seeds")
            selection = SeedSelectionConfig(
                seed_min=policy.seed_min,
                seed_max=policy.seed_max,
                seed_count=policy.seed_count,
                development_seed_count=len(development),
            )
            seed_file = (
                resolver.root
                / "research"
                / plan.study_id
                / "evaluation-seeds"
                / (task_id + ".yaml")
            )
            if not seed_file.exists():
                rng = random.SystemRandom()
                chosen = set()
                while len(chosen) < policy.seed_count:
                    seed = rng.randint(policy.seed_min, policy.seed_max)
                    if seed not in development:
                        chosen.add(seed)
                seed_file.parent.mkdir(parents=True, exist_ok=True)
                seed_file.write_text(
                    yaml.safe_dump(
                        {"task_id": task_id, "evaluation_seeds": sorted(chosen)}
                    )
                )
                seed_file.chmod(0o600)
            current = SeedRobustnessConfig(
                development_seeds=development,
                evaluation_seeds_file="private://"
                + seed_file.relative_to(resolver.root).as_posix(),
                evaluation_seeds_file_sha256=_sha(seed_file),
                seed_selection=selection,
            )
            hidden = load_private_seed_robustness_config(
                resolver, task_id=task_id, public_config=current
            )
            files[str(seed_file)] = _sha(seed_file)
        counts[task_id] = {
            "development": len(current.development_seeds),
            "evaluation": len(hidden.evaluation_seeds),
        }
        task.evaluation.solver_seeds = None
        task.evaluation.seed_robustness = current
        task_path = destination / "evaluator" / (task_id + ".yaml")
        task_path.parent.mkdir(exist_ok=True)
        task_path.write_text(
            yaml.safe_dump(task.model_dump(mode="json"), sort_keys=False)
        )
        metadata.update(
            evaluator_task_config=str(task_path.resolve()),
            evaluator_sha256=_sha(task_path),
        )
        files[str(task_path.resolve())] = _sha(task_path)
        for arm in plan.arms:
            row = result["arms"][arm.id][task_id]
            public = PitBenchTask.from_yaml(Path(row["agent_task_config_path"]))
            public.evaluation = task.evaluation.model_copy(deep=True)
            path = destination / "agent-views" / arm.id / (task_id + ".yaml")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                yaml.safe_dump(public.model_dump(mode="json"), sort_keys=False)
            )
            row.update(
                task_config_path=str(task_path.resolve()),
                agent_task_config_path=str(path.resolve()),
            )
            files[str(path.resolve())] = _sha(path)
    result["seed_policy"] = identity
    _write(result_path, result)
    files[str(result_path.resolve())] = _sha(result_path)
    _write(
        receipt_path, {"identity": identity, "files_sha256": files, "counts": counts}
    )
    return result_path


def study_evaluation_matches(
    original: PitBenchTask, study: PitBenchTask, plan: GeneralizationStudyPlan
) -> bool:
    """Only the explicitly selected seed policy may alter the native contract."""
    if original.release != study.release:
        return False
    if original.evaluation == study.evaluation:
        return True
    policy = plan.task_evaluation_seed_policies.get(original.task_id, plan.evaluation_seed_policy)
    current = study.evaluation.seed_robustness
    if policy is None or current is None or study.evaluation.solver_seeds is not None:
        return False
    if current.seed_selection.seed_count != policy.seed_count:
        return False
    old = original.evaluation.seed_robustness
    if old is not None and policy.preserve_existing and old != current:
        return False
    if current.development_seeds != (
        old.development_seeds if old else original.evaluation.solver_seeds
    ):
        return False
    exclude = {"solver_seeds", "seed_robustness"}
    return original.evaluation.model_dump(
        exclude=exclude
    ) == study.evaluation.model_dump(exclude=exclude)


def freeze_generalization_measurements(
    *,
    views_path: Path,
    population_root: Path,
    references_path: Path,
    specification_path: Path,
    root: Path,
    private_root: Path,
    destination: Path,
) -> dict:
    """Attach fully declared diagnostic models without changing primary panels."""
    import copy

    import numpy as np

    from pitbench.evaluator.private_assets import PrivateAssetResolver
    from pitbench.evaluator.representations import (
        CustomerRepresentation,
        LinearModelRepresentation,
    )
    from pitbench.problem_families.public_verification import check_cvrp
    from pitbench.problem_families.verification import NumericModelFamily
    from pitbench.repositories.base import RepositoryPluginRegistry
    from pitbench.schema.scenarios import SCENARIO_METRICS, ScenarioProtocol
    from pitbench.schema.task import InstanceSetSpec

    source = json.loads(views_path.read_text())
    refs = json.loads(references_path.read_text())["records"]
    spec = yaml.safe_load(specification_path.read_text())
    construction = json.loads(
        (population_root / "population-construction.json").read_text()
    )
    inventories = {
        role: json.loads(Path(row["manifest"]).read_text())["instances"]
        for role, row in construction["splits"].items()
    }
    resolver = PrivateAssetResolver(private_root)
    data_root = resolver.root / "research" / destination.name / "scenario-models"
    data_root.mkdir(parents=True, exist_ok=False)
    destination.mkdir(parents=True, exist_ok=False)
    result = copy.deepcopy(source)
    result.update(
        kind="generalization_execution_views",
        status="development_frozen_evaluation_gated",
        scientific_evaluation_ready=False,
        specification_sha256=_sha(specification_path),
    )
    for task_id, metadata in result["tasks"].items():
        task = PitBenchTask.from_yaml(
            Path(source["tasks"][task_id]["evaluator_task_config"])
        )
        plugin = RepositoryPluginRegistry.load(task.repository.plugin)
        numeric = task.problem_family.value == "mip"
        representation = (
            LinearModelRepresentation if numeric else CustomerRepresentation
        )
        protocols, case_inventory = [], {}
        for name, panel_metadata in metadata["scientific_panels"].items():
            entries = [
                row
                for row in inventories[panel_metadata["scientific_role"]]
                if row["condition_id"] == panel_metadata["condition_id"]
                and row["replicate_index"] in spec["replicate_indices"]
            ]
            if len(entries) != len(spec["replicate_indices"]):
                raise ValueError("diagnostic selection is incomplete")
            panel_name = "scenario__" + name
            parent_panel = next(p for p in task.instance_sets if p.name == name)
            panel_payload = yaml.safe_load(
                resolver.resolve(
                    parent_panel.instance_set_config,
                    parent_panel.instance_set_config_sha256,
                ).read_text()
            )
            wanted_ids = {entry["id"] for entry in entries}
            panel_payload["instances"] = [
                row for row in panel_payload["instances"] if row["id"] in wanted_ids
            ]
            if len(panel_payload["instances"]) != len(entries):
                raise ValueError("diagnostic panel does not match its source inventory")
            panel_config = data_root / "panels" / task_id / (panel_name + ".yaml")
            panel_config.parent.mkdir(parents=True, exist_ok=True)
            panel_config.write_text(yaml.safe_dump(panel_payload, sort_keys=False))
            task.instance_sets.append(
                InstanceSetSpec(
                    name=panel_name,
                    kind="judge_shift",
                    instance_set_config="private://"
                    + panel_config.relative_to(resolver.root).as_posix(),
                    instance_set_config_sha256=_sha(panel_config),
                    size=len(entries),
                    scenario_only=True,
                )
            )
            cases, targets, references, scales, witnesses, model_targets = (
                [],
                {},
                {},
                {},
                {},
                {},
            )
            for entry in entries:
                identity = entry["id"]
                target = refs[identity]
                folder = data_root / entry["sha256"]
                folder.mkdir(exist_ok=True)
                base_path = folder / ("base" + representation.suffix)
                if numeric:
                    original = LinearModelRepresentation.read(Path(entry["path"]))
                    witness_file = resolver.resolve(
                        target["record"]["reference_solution_uri"],
                        target["record"]["reference_solution_sha256"],
                    )
                    values = json.loads(witness_file.read_text())["values"]
                    witness = np.asarray(
                        [values[name] for name in original["col_names"]], dtype=float
                    )
                    rounded = np.where(
                        np.abs(witness - np.rint(witness))
                        <= spec["feasibility_tolerance"],
                        np.rint(witness),
                        witness,
                    )
                    if NumericModelFamily.check_model(
                        original, rounded, spec["feasibility_tolerance"]
                    )["feasible"]:
                        witness = rounded
                    options = []
                    binary = (
                        (original["integrality"] == 1)
                        & (original["col_lower"] == 0)
                        & (original["col_upper"] == 1)
                    )
                    for value in [0, 1]:
                        candidate = witness.copy()
                        candidate[binary] = value
                        if NumericModelFamily.check_model(
                            original, candidate, spec["feasibility_tolerance"]
                        )["feasible"]:
                            options.append(candidate)
                    if not options:
                        options = [witness]
                    extreme_witness = max(
                        options, key=lambda v: float(np.abs(v - witness).sum())
                    )
                    changed = copy.deepcopy(original)
                    changed["cost"] = np.zeros_like(original["cost"])
                    changed["cost"][binary] = np.where(
                        extreme_witness[binary] < 0.5, 1.0, -1.0
                    )
                    extreme_target = NumericModelFamily.check_model(
                        changed, extreme_witness, spec["feasibility_tolerance"]
                    )["objective"]
                    stress = []
                    for factor in spec["row_scales"]:
                        model = copy.deepcopy(original)
                        model["matrix"] = model["matrix"] * factor
                        model["row_lower"] = model["row_lower"] * factor
                        model["row_upper"] = model["row_upper"] * factor
                        if not NumericModelFamily.check_model(
                            model, witness, spec["feasibility_tolerance"]
                        )["feasible"]:
                            raise ValueError(
                                "stress witness is not valid in original stress units"
                            )
                        stress.append(model)
                    witness = witness.tolist()
                else:
                    original = json.loads(Path(entry["path"]).read_text())
                    witness = json.loads(Path(target["solution"]).read_text())
                    stress = []
                    for factor in spec["routing_integer_scales"]:
                        model = copy.deepcopy(original)
                        model["demands"] = [int(v) * factor for v in model["demands"]]
                        model["capacity"] = int(model["capacity"]) * factor
                        if not check_cvrp(model, witness)["feasible"]:
                            raise ValueError("routing integer-range witness is invalid")
                        stress.append(model)
                    changed = stress[0]
                    extreme_target = target["objective"]
                models = [original, changed, *stress]
                files = []
                for index, model in enumerate(models):
                    path = (
                        base_path
                        if index == 0
                        else folder / (f"model-{index}" + representation.suffix)
                    )
                    if not path.exists():
                        representation.write_input(path, model)
                    files.append(str(path.resolve()))
                epsilon = spec["target_absolute_slack"]
                reference = float(target["objective"])
                quality = (
                    reference
                    + epsilon
                    + (
                        spec["routing_relative_target_slack"] * max(1, abs(reference))
                        if not numeric
                        else 0
                    )
                )
                targets[identity] = quality
                references[identity] = reference
                scales[identity] = max(1, abs(reference))
                witnesses[identity] = witness
                model_targets.update({_sha(Path(path)): quality for path in files})
                model_targets[_sha(Path(files[1]))] = float(extreme_target) + epsilon
                by_kind = {kind: [files[0]] for kind in SCENARIO_METRICS}
                by_kind["lifecycle"] = [files[0]] * spec["lifecycle_steps"]
                by_kind["sequence"] = [files[0], files[1], files[0]]
                by_kind["numerical"] = files[2:]
                cases.append(
                    {
                        "id": identity,
                        "models_by_scenario": {
                            kind + "__" + panel_name: paths
                            for kind, paths in by_kind.items()
                        },
                    }
                )
            case_inventory[panel_name] = cases
            parent = next(p for p in task.instance_sets if p.name == name)
            budget = parent.primary_budget_sec or task.evaluation.primary_budget_sec
            for kind, metrics in SCENARIO_METRICS.items():
                settings = copy.deepcopy(spec["default_protocol"])
                if kind == "lifecycle":
                    settings.update(modes=["warm"], warmup_steps=spec["warmup_steps"])
                if kind == "scaling":
                    settings["thread_counts"] = spec["thread_counts"]
                if kind == "sequence":
                    settings["modes"] = ["cold", "warm"]
                if kind == "replay":
                    settings["repeats"] = spec["replay_repeats"]
                pid = kind + "__" + panel_name
                protocol = ScenarioProtocol(
                    id=pid,
                    kind=kind,
                    metrics=sorted(metrics),
                    representation=representation.name,
                    instance_sets=[panel_name],
                    objective_sense=task.oracle.objective_sense,
                    budget_sec=spec["lifecycle_budget_sec"]
                    if kind == "lifecycle"
                    else budget,
                    solver_seeds=spec["solver_seeds"],
                    target_by_instance=targets,
                    reference_by_instance=references,
                    loss_scale_by_instance=scales,
                    reference_solution_by_instance=witnesses,
                    qualification_target_by_model_sha256=model_targets,
                    model_sequences_sha256={
                        panel_name: {
                            case["id"]: [
                                _sha(Path(path))
                                for path in case["models_by_scenario"][pid]
                            ]
                            for case in cases
                        }
                    },
                    memory_sample_interval_sec=spec["memory_sample_interval_sec"],
                    feasibility_tolerance=spec["feasibility_tolerance"],
                    **settings,
                )
                protocols.append(protocol)
        task.metric_scenarios = protocols
        evaluator_path = destination / "evaluator" / (task_id + ".yaml")
        evaluator_path.parent.mkdir(exist_ok=True)
        evaluator_path.write_text(
            yaml.safe_dump(task.model_dump(mode="json"), sort_keys=False)
        )
        metadata.update(
            evaluator_task_config=str(evaluator_path.resolve()),
            evaluator_sha256=_sha(evaluator_path),
            diagnostic_protocols=len(protocols),
            diagnostic_instances=sum(len(v) for v in case_inventory.values()),
        )
        config = {
            "order": "base_first",
            "watchdog_grace_sec": spec["watchdog_grace_sec"],
            "instances": case_inventory,
            "source_build": {
                "source_commit": task.release.base_commit,
                "repository_plugin": task.repository.plugin,
                "session_plugin": plugin.scenario_session_plugin,
                "from_evaluation_repository": True,
            },
            "options_by_scenario": {
                p.id: spec.get("options_by_repository", {})
                .get(task.repository.plugin, {})
                .get(p.kind, {})
                for p in protocols
            },
        }
        config_path = destination / "scenario-configs" / (task_id + ".yaml")
        config_path.parent.mkdir(exist_ok=True)
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        metadata["metric_scenario_config"] = str(config_path.resolve())
        for arm in result["arms"]:
            public = PitBenchTask.from_yaml(
                Path(source["arms"][arm][task_id]["agent_task_config_path"])
            )
            public.metric_scenarios = protocols
            public.instance_sets.extend(
                [p for p in task.instance_sets if p.scenario_only]
            )
            path = destination / "agent-views" / arm / (task_id + ".yaml")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                yaml.safe_dump(public.model_dump(mode="json"), sort_keys=False)
            )
            result["arms"][arm][task_id].update(
                agent_task_config_path=str(path.resolve()),
                task_config_path=str(evaluator_path.resolve()),
            )
    _write(destination / "task-views.json", result)
    return result


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _public_asset(root: Path, value: str) -> Path:
    if value.startswith("private://"):
        raise ValueError("private assets cannot be read as public research inputs")
    candidate = (root / value).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError("public research asset escapes repository root")
    return candidate


def resolve_research_budget(
    root: Path,
    *,
    track: str | None = None,
    configured: ResearchBudget | None = None,
    smoke_test: bool = False,
) -> ResearchBudget:
    """Resolve the selected wall/CPU/memory profile without adding usage caps."""
    catalog = json.loads((root / "configs/research/budgets.json").read_text())
    if catalog.get("schema_version") != "1.0":
        raise ValueError("unsupported research budget catalog")
    if smoke_test:
        selected = resolve_research_budget(root, track=track, configured=configured)
        return selected.model_copy(
            update={
                "track": "custom",
                "wall_time_sec": catalog["smoke_wall_time_sec"],
                "checkpoints_sec": [],
            }
        )
    if track is not None:
        if track not in catalog["tracks"]:
            raise ValueError(f"unknown research track: {track}")
        # Explicit track selection chooses its full profile; do not silently
        # inherit resource overrides from a different track.
        return ResearchBudget.model_validate(catalog["tracks"][track])
    budget = configured or ResearchBudget.model_validate(
        catalog["tracks"][catalog["default_track"]]
    )
    if budget.track != "custom":
        expected = catalog["tracks"].get(budget.track)
        if expected is None or budget.wall_time_sec != expected["wall_time_sec"]:
            raise ValueError("named track wall time differs; use a custom budget")
    return budget


def research_budget_gaps(budget: ResearchBudget | None) -> list[str]:
    if budget is None:
        return ["research_budget_not_declared"]
    return [
        f"research_{key}_not_declared"
        for key in ("cpus", "memory_gib")
        if getattr(budget, key) is None
    ]


def construct_research_case(
    plan: ResearchCasePlan,
    *,
    root: Path,
    destination: Path,
    protocol_path: Path | None = None,
) -> dict:
    """Materialize both guidance settings without exporting the historical answer."""
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        result = _construct_research_case(
            plan, root=root, destination=staging, protocol_path=protocol_path
        )
        if destination.exists():
            raise FileExistsError(destination)
        staging.rename(destination)
        return result
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _construct_research_case(
    plan: ResearchCasePlan,
    *,
    root: Path,
    destination: Path,
    protocol_path: Path | None,
) -> dict:
    protocol_path = protocol_path or root / "configs/research/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("schema_version") != "1.0":
        raise ValueError("unsupported research protocol version")
    protocol_identity = {
        "schema_version": "1.0",
        "sha256": _sha(protocol_path),
        "status": protocol["status"],
    }
    task_path = _public_asset(root, plan.task_config)
    task = PitBenchTask.from_yaml(task_path)
    request_path = _public_asset(root, plan.historical_request)
    bundle_root = _public_asset(root, plan.historical_bundle_root)
    request = json.loads(request_path.read_text())
    if request.get("kind") != "research_case_request":
        raise ValueError("expected an evaluator-owned research construction request")
    record = request["candidate"]
    source = request["git_evidence"]
    if (record["repository"], record["base_commit"]) != (
        task.release.repository,
        task.release.base_commit,
    ):
        raise ValueError("research task and historical source identity differ")
    if source.get("source_status") not in {"cached", "cached_branch_base"}:
        raise ValueError("historical source has not been reconstructed")
    if not source.get("artifact_path") or not source.get("artifact_sha256"):
        raise ValueError("historical source patch is missing")
    patches = {}
    for name, path_key, sha_key in (
        ("solver_source", "artifact_path", "artifact_sha256"),
        (
            "full_integrated",
            "full_integrated_patch_path",
            "full_integrated_patch_sha256",
        ),
    ):
        if path_key not in source:
            continue
        patch = (bundle_root / source[path_key]).resolve()
        if not patch.is_relative_to(bundle_root):
            raise ValueError("historical patch escapes its evidence bundle")
        if _sha(patch) != source.get(sha_key):
            raise ValueError("historical patch hash differs")
        target = destination / "evaluator_only" / "human_reference" / f"{name}.patch"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(patch, target)
        patches[name] = {
            "sha256": _sha(target),
            "path": f"human_reference/{name}.patch",
        }
    development = [
        item for item in task.instance_sets if item.kind == InstanceSetKind.AGENT_DEV
    ]
    hidden = [
        item for item in task.instance_sets if item.kind != InstanceSetKind.AGENT_DEV
    ]
    names = {item.name for item in hidden}
    if set(plan.hidden_view_roles) - names:
        raise ValueError("scientific role references an undeclared hidden panel")
    public_panels = []
    for index, item in enumerate(development):
        config = _public_asset(root, item.instance_set_config)
        if (
            item.instance_set_config_sha256
            and _sha(config) != item.instance_set_config_sha256
        ):
            raise ValueError("frozen development panel hash differs")
        if config.is_relative_to((root / "private").resolve()):
            raise ValueError(
                "development assets must not resolve into the hidden asset root"
            )
        output = (
            destination
            / "evaluator_only"
            / "development_materialization"
            / f"panel_{index:02d}"
        )
        paths = materialize_instance_set(config, output)
        if len(paths) != item.size:
            raise ValueError("development panel size differs from the frozen task")
        public_panels.append(
            {
                "path": f"development/panel_{index:02d}",
                "size": len(paths),
                "manifest_sha256": _sha(output / "instance_set_config.yaml"),
            }
        )
    private_missing = []
    for uri, expected in [
        (task.oracle.source, task.oracle.source_sha256),
        *[
            (item.instance_set_config, item.instance_set_config_sha256)
            for item in hidden
        ],
    ]:
        if uri.startswith("private://"):
            path = _public_asset(root, "private/" + uri.removeprefix("private://"))
            if not path.is_file() or (expected and _sha(path) != expected):
                private_missing.append(uri)
    gaps = research_budget_gaps(plan.research_budget)
    if "full_integrated" not in patches:
        gaps.append("complete_human_patch_missing")
    if private_missing:
        gaps.append("frozen_private_assets_unavailable")
    if not {"id", "structure", "scale"}.issubset(set(plan.hidden_view_roles.values())):
        gaps.append("controlled_structure_and_scale_panels_not_bound")
    # A JSON declaration is not evidence that the host enforces these limits.
    gaps.extend(
        ["research_runtime_not_validated", "research_budget_enforcement_not_validated"]
    )
    covered = set()
    for evidence in plan.guided_context_evidence:
        index = evidence.context_index
        if index >= len(plan.guided_context):
            raise ValueError("context evidence references an undeclared paragraph")
        if (
            hashlib.sha256(plan.guided_context[index].encode()).hexdigest()
            != evidence.context_sha256
        ):
            raise ValueError("reviewed context text hash differs")
        if (
            _sha(_public_asset(root, evidence.snapshot_path))
            != evidence.snapshot_sha256
        ):
            raise ValueError("reviewed context source snapshot hash differs")
        if evidence.available_at > evidence.solution_cutoff:
            raise ValueError("context was not available before the solution cutoff")
        covered.add(index)
    context_ready = bool(plan.guided_context) and covered == set(
        range(len(plan.guided_context))
    )
    modes = {}
    for mode, question in (
        ("open_ended", plan.open_question),
        ("guided", plan.guided_question),
    ):
        mode_gaps = list(gaps)
        if mode == "guided" and not context_ready:
            mode_gaps.append("pre_solution_guided_context_not_verified")
        # Positive projection: no title, PR number, human mechanism, task ID,
        # hidden panel names/configs/seeds, historical instruction or gold patch.
        public = {
            "research_protocol": protocol_identity,
            "schema_version": "1.0",
            "kind": "research_case_agent_view",
            "case_id": plan.case_id,
            "mode": mode,
            "research_question": question,
            "problem_specification": {
                "problem_family": task.problem_family.value,
                "repository": task.repository.clone_url,
                "base_commit": task.release.base_commit,
                "editable_paths": task.repository.editable_paths,
                "mathematical_requirement": "Preserve original-problem feasibility, objective semantics and truthful statuses. Bounds, optimality and infeasibility require corresponding independent evidence.",
            },
            "experimental_environment": {
                "development_panels": public_panels,
                "access": plan.experimental_access,
                "generators": [item.model_dump() for item in plan.generators],
                "research_budget": plan.research_budget.model_dump()
                if plan.research_budget
                else None,
                "development_solver_budgets_sec": task.evaluation.budgets_sec,
                "development_solver_seeds": task.evaluation.seed_robustness.development_seeds
                if task.evaluation.seed_robustness
                else task.evaluation.solver_seeds,
                "development_threads": task.evaluation.threads,
                "bench_interface": "pitbench bench --instance FILE --seed SEED --budget SECONDS",
                "budget_enforcement": "requires_evaluator_owned_runtime",
                "network": "disabled",
                "git_history": "base_only_snapshot_without_remotes",
            },
            "final_evaluation_contract": {
                "performance_protocol": task.evaluation.performance_protocol.value,
                "primary_budget_sec": task.evaluation.primary_budget_sec,
                "verifier": task.evaluation.verifier,
                "scoring_rule": "Existing frozen task metric; no new cross-family loss or composite score.",
                "hidden_feedback": "unavailable_during_research",
            },
            "submission": {
                "required": ["solver_source_patch"],
                "optional_research_track": [
                    "frozen_predictions",
                    "experiment_ledger",
                    "ablation_artifacts",
                ],
                "prediction_commitment": "Evaluator must record patch, prediction and artifact hashes before opening hidden results; Agent-written timestamps do not establish this.",
            },
            "status": "draft",
            "readiness_gaps": [
                gap for gap in mode_gaps if not gap.startswith("frozen_private")
            ],
        }
        if mode == "guided":
            public["problem_context"] = plan.guided_context if context_ready else []
        public_root = destination / "agent" / mode
        shutil.copytree(
            destination / "evaluator_only" / "development_materialization",
            public_root / "development",
        )
        target = public_root / "case.json"
        _write(target, public)
        modes[mode] = {
            "status": "draft",
            "readiness_gaps": mode_gaps,
            "agent_view": f"agent/{mode}/case.json",
            "agent_view_sha256": _sha(target),
        }
    case = {
        "research_protocol": protocol_identity,
        "schema_version": "1.0",
        "kind": "research_case",
        "case_id": plan.case_id,
        "status": "draft",
        "modes": modes,
        "problem_specification": {
            "task_config": plan.task_config,
            "task_config_sha256": _sha(task_path),
        },
        "experimental_environment": plan.model_dump(mode="json"),
        "hidden_scientific_evaluation": {
            "frozen_evaluation": task.evaluation.model_dump(mode="json"),
            "instance_sets": [item.model_dump(mode="json") for item in hidden],
            "scientific_roles": plan.hidden_view_roles,
            "private_assets_missing": private_missing,
            "interpretation": "Instance structure and solver difficulty are distinct. Existing shift labels do not establish a controlled structural or scale intervention.",
        },
        "historical_reference": {
            "candidate": record,
            "patches": patches,
            "request_sha256": _sha(request_path),
            "mechanism": "evaluator_only",
            "gain_status": "not_remeasured_in_research_pilot",
            "agent_visible": False,
        },
        "counting": "Two guidance settings of one source case; not two independent PR tasks.",
        "claim_boundary": "Construction and visibility prototype only. No Agent research, solver qualification, causal mechanism, mathematical correctness or methodological novelty is established.",
    }
    _write(destination / "evaluator_only" / "research_case.json", case)
    return case


def materialize_research_experiment(
    public_case: dict, payload: dict, output_dir: Path
) -> dict:
    """Let a caller choose public generator parameters; reuse the native generator."""
    if public_case.get("kind") != "research_case_agent_view":
        raise ValueError("expected the public case projection")
    environment = public_case["experimental_environment"]
    if environment["access"] != "construct_instances":
        raise ValueError("this experiment arm does not allow constructing instances")
    generator = payload.get("generator", {})
    capabilities = {item["kind"]: item for item in environment["generators"]}
    capability = capabilities.get(generator.get("kind"))
    if capability is None or set(generator) - {"kind", *capability["parameters"]}:
        raise ValueError(
            "generator or parameter is outside the public capability declaration"
        )
    if type(generator.get("count")) is not int or generator["count"] < 1:
        raise ValueError("generated instance count must be positive")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    paths = materialize_generated_instance_set(
        payload, output_dir, expected_visibility="agent"
    )
    record = {
        "kind": "research_experiment_materialization",
        "case_id": public_case["case_id"],
        "generator_payload": payload,
        "instances": [{"path": p.name, "sha256": _sha(p)} for p in paths],
        "solver_runs": 0,
        "budget_enforcement": "not_provided_by_materialization",
    }
    _write(output_dir / "experiment.json", record)
    return record
