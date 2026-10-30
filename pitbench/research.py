"""Construct evaluator-owned Research Cases and explicit public projections.

This layer reuses existing task metrics, generators and independent verifiers.
It does not certify a new task, score a research report, or enforce compute quotas.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import shutil
import tempfile
from pathlib import Path

import yaml

from pitbench.instances.generate import (
    materialize_generated_instance_set,
    materialize_instance_set,
)
from pitbench.schema.research import (
    GeneralizationStudyPlan,
    ResearchBudget,
    ResearchCasePlan,
)
from pitbench.schema.task import InstanceSetKind, PitBenchTask


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
        if split.get("manifest") is None:
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


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
