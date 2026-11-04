"""Universal, non-expert HATS model and opportunity bootstrapping from task configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from global_planner.schema import (
    HATSLevel,
    HATSNode,
    Opportunity,
)


def discover_concrete_nodes_from_repository(
    repository_root: Path,
    declared_paths: list[str],
) -> list[HATSNode]:
    """Scans the repository within declared editable paths and discovers concrete

    code components and mechanisms, binding them into the appropriate Level 1 subsystems.
    """
    discovered: list[HATSNode] = []
    if not repository_root or not repository_root.exists():
        return discovered

    source_extensions = {".c", ".h", ".cpp", ".hpp", ".cc", ".py"}

    for rel_path in declared_paths:
        target_dir = repository_root / rel_path
        if not target_dir.exists():
            continue

        if target_dir.is_file() and target_dir.suffix in source_extensions:
            candidate_files = [target_dir]
        else:
            candidate_files = [
                p for p in target_dir.rglob("*")
                if p.is_file() and p.suffix in source_extensions
            ]

        for file_p in candidate_files:
            rel_file = str(file_p.relative_to(repository_root))
            lower_name = file_p.name.lower()
            stem = file_p.stem

            parent_id = "subsystem.search_engine"
            if any(k in lower_name for k in ("heur", "local", "exchange", "tour", "relocate", "swap")):
                parent_id = "subsystem.heuristics"
            elif any(k in lower_name for k in ("presol", "prep", "bound", "tighten", "clean")):
                parent_id = "subsystem.presolve"
            elif any(k in lower_name for k in ("cons", "cut", "sepa", "prop", "penalty", "valid")):
                parent_id = "subsystem.constraints"
            elif any(k in lower_name for k in ("branch", "nodesel", "tree", "node", "dive", "plung")):
                parent_id = "subsystem.search_engine"

            node_id = f"component.{parent_id.split('.')[-1]}.{stem}"
            discovered.append(
                HATSNode(
                    id=node_id,
                    level=HATSLevel.LEVEL_3_CODE,
                    parent_id=parent_id,
                    purpose=f"Concrete algorithm component in {rel_file}",
                    source_refs=[rel_file],
                    reads=["search_state"],
                    writes=["solution_or_state"],
                    invariants=["original objective semantics"],
                )
            )

    return discovered


def bootstrap_hats_model(
    task_config: dict[str, Any],
    repository_root: Path | None = None,
) -> tuple[str, list[HATSNode], list[Opportunity]]:
    """Generates a solver-agnostic Level 0 goal statement, universal subsystem nodes bound

    to the task's declared editable paths, and autonomous exploration opportunities.
    Actively populates concrete code nodes if repository_root is provided.
    """
    solver_repo = str(task_config.get("repository", {}).get("clone_url") or task_config.get("release", {}).get("repository", "solver"))
    solver_name = solver_repo.rstrip("/").split("/")[-1].replace(".git", "")
    family = str(task_config.get("problem_family", "general")).lower()
    variant = str(task_config.get("problem_variant", family)).lower()
    protocol = str(task_config.get("evaluation", {}).get("performance_protocol", "exact_verified_solve"))
    primary_budget = task_config.get("evaluation", {}).get("primary_budget_sec", 20.0)

    # Resolve declared editable paths from task config
    declared_paths = list(task_config.get("repository", {}).get("editable_paths", ["src"]))
    if not declared_paths:
        declared_paths = ["src"]

    # Level 0 Goal
    if protocol == "heuristic_fixed_budget" or "cvrp" in family or "routing" in family:
        goal_statement = (
            f"Minimize primal objective cost and gap on {variant} within {primary_budget}s budget on {solver_name}"
        )
    else:
        goal_statement = (
            f"Maximize independently verified solved coverage and minimize PAR-2 runtime on {variant} under {primary_budget}s budget on {solver_name}"
        )

    nodes: list[HATSNode] = [
        HATSNode(
            id="goal.primary_performance",
            level=HATSLevel.LEVEL_0_GOAL,
            purpose=goal_statement,
            reads=["solver_status", "objective", "dual_bound", "wall_time_sec"],
            writes=["solution"],
            invariants=["100% solution validity", "original objective semantics"],
        ),
        HATSNode(
            id="subsystem.presolve",
            level=HATSLevel.LEVEL_1_SUBSYSTEM,
            parent_id="goal.primary_performance",
            purpose="Preprocessing, problem reduction, variable fixing, and domain bound tightening",
            source_refs=declared_paths,
            reads=["model_constraints", "variable_bounds"],
            writes=["reduced_model"],
        ),
        HATSNode(
            id="subsystem.search_engine",
            level=HATSLevel.LEVEL_1_SUBSYSTEM,
            parent_id="goal.primary_performance",
            purpose="Global search coordination, tree exploration, branching decisions, or route move neighborhoods",
            source_refs=declared_paths,
            reads=["search_state", "open_nodes_or_routes", "bounds"],
            writes=["current_state", "search_tree"],
        ),
        HATSNode(
            id="subsystem.heuristics",
            level=HATSLevel.LEVEL_1_SUBSYSTEM,
            parent_id="goal.primary_performance",
            purpose="Primal heuristic search, local improvement operators, or feasible solution generation",
            source_refs=declared_paths,
            reads=["current_state", "incumbent", "residual_constraints"],
            writes=["incumbent", "best_solution"],
        ),
        HATSNode(
            id="subsystem.constraints",
            level=HATSLevel.LEVEL_1_SUBSYSTEM,
            parent_id="goal.primary_performance",
            purpose="Constraint propagation, feasibility verification, cut generation, and penalty evaluation",
            source_refs=declared_paths,
            reads=["candidate_solution", "constraints"],
            writes=["violation_status", "penalties_or_cuts"],
        ),
    ]

    # Actively discover concrete code-level nodes from the repository if provided
    if repository_root and Path(repository_root).exists():
        concrete_nodes = discover_concrete_nodes_from_repository(
            Path(repository_root), declared_paths
        )
        nodes.extend(concrete_nodes)

    # Universal autonomous exploration opportunities (NO human expert hints or problem answers!)
    opportunities: list[Opportunity] = [
        Opportunity(
            id="opp_profile_bottlenecks",
            title="Profile computational bottlenecks and search stagnation",
            target_nodes=["subsystem.search_engine", "subsystem.heuristics"],
            source_files=declared_paths,
            proposal="Inspect repository source within declared editable paths, benchmark the baseline on visible development instances, and identify where the solver spends execution time or stalls.",
            expected_effect="Establish calibrated baseline behavior and identify candidate functions/subsystems for targeted optimization.",
            cheapest_discriminating_experiment="Run pitbench bench --label base on visible development instances",
            status="unexplored",
        ),
        Opportunity(
            id="opp_explore_primal_heuristics",
            title="Analyze primal solution generation and feasibility progress",
            target_nodes=["subsystem.heuristics"],
            source_files=declared_paths,
            proposal="Investigate whether search stalls finding initial or improving feasible solutions, and identify relevant heuristic or local search routines in the repository.",
            expected_effect="Formulate an algorithm-state transition hypothesis and design a bounded, semantics-preserving mechanism improvement.",
            cheapest_discriminating_experiment="Run pitbench bench and inspect solution trajectory progress",
            status="unexplored",
        ),
        Opportunity(
            id="opp_explore_search_policy",
            title="Analyze search exploration, branching, or neighborhood evaluation",
            target_nodes=["subsystem.search_engine"],
            source_files=declared_paths,
            proposal="Investigate search-tree exploration efficiency, node selection/branching overhead, or route neighborhood evaluation throughput.",
            expected_effect="Identify search overhead or redundant computations across search iterations.",
            cheapest_discriminating_experiment="Run paired pitbench bench and inspect work metrics (nodes/iterations/propagations)",
            status="unexplored",
        ),
    ]

    return goal_statement, nodes, opportunities
