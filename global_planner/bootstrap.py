"""Bootstrapping initial HATS models and opportunities from task configuration."""

from __future__ import annotations

from typing import Any

from global_planner.schema import (
    HATSLevel,
    HATSNode,
    Opportunity,
)


def bootstrap_hats_model(
    task_config: dict[str, Any],
) -> tuple[str, list[HATSNode], list[Opportunity]]:
    """Generates the goal statement, initial HATS nodes, and opportunity table

    tailored to the task family, solver, and evaluation protocol.
    """
    solver = str(task_config.get("release", {}).get("repository", "solver")).split("/")[-1].lower()
    family = str(task_config.get("problem_family", "general")).lower()
    variant = str(task_config.get("problem_variant", family)).lower()
    primary_budget = task_config.get("evaluation", {}).get("primary_budget_sec", 20.0)

    # Goal statement
    if "cvrp" in family or "vrp" in solver:
        goal_statement = (
            f"Minimize penalized primal tour cost and gap on {variant} within {primary_budget}s budget"
        )
    else:
        goal_statement = (
            f"Maximize verified solved coverage and minimize PAR-2 runtime on {variant} under {primary_budget}s budget"
        )

    nodes: list[HATSNode] = []
    opportunities: list[Opportunity] = []

    # Level 0 Goal Node
    nodes.append(
        HATSNode(
            id="goal.primary_performance",
            level=HATSLevel.LEVEL_0_GOAL,
            purpose=goal_statement,
            reads=["solver_status", "objective", "dual_bound", "wall_time_sec"],
            writes=["solution"],
            invariants=["100% solution validity", "original objective semantics"],
        )
    )

    if "cvrp" in family or "vrp" in solver:
        # Routing / PyVRP Subsystems
        nodes.extend([
            HATSNode(
                id="subsystem.local_search",
                level=HATSLevel.LEVEL_1_SUBSYSTEM,
                parent_id="goal.primary_performance",
                purpose="Iteratively improve routes using 2-opt, relocate, swap and compound neighborhood operators",
                source_refs=["pyvrp/search"],
                reads=["routes", "distance_matrix", "penalties"],
                writes=["routes", "current_cost"],
            ),
            HATSNode(
                id="subsystem.population",
                level=HATSLevel.LEVEL_1_SUBSYSTEM,
                parent_id="goal.primary_performance",
                purpose="Maintain feasible and infeasible sub-populations with diversity bias and survivor selection",
                source_refs=["pyvrp/Population.cpp"],
                reads=["population", "diversity_metrics"],
                writes=["population"],
            ),
            HATSNode(
                id="subsystem.penalty_management",
                level=HATSLevel.LEVEL_1_SUBSYSTEM,
                parent_id="goal.primary_performance",
                purpose="Dynamically update capacity and time-window violation penalties based on feasibility ratio",
                source_refs=["pyvrp/PenaltyManager.cpp"],
                reads=["feasibility_history"],
                writes=["penalty_capacity", "penalty_time_warp"],
            ),
        ])

        opportunities.extend([
            Opportunity(
                id="opp_ls_compound_moves",
                title="Compound inter-route exchanges",
                target_nodes=["subsystem.local_search"],
                source_files=["pyvrp/search"],
                proposal="Identify stalling routes and perform multi-operator compound swaps crossing intermediate infeasible penalties",
                expected_effect="Improve primal gap on large client instances under short budgets",
                cheapest_discriminating_experiment="Run pitbench bench on 500-client dev instance with 5s budget",
            ),
            Opportunity(
                id="opp_horizon_penalty_decay",
                title="Horizon-aware penalty escalation",
                target_nodes=["subsystem.penalty_management"],
                source_files=["pyvrp/PenaltyManager.cpp"],
                proposal="Scale penalty updates according to remaining search budget instead of fixed update intervals",
                expected_effect="Accelerate feasibility convergence early in search without sacrificing late-stage intensification",
                cheapest_discriminating_experiment="Run paired bench comparing 5s vs 10s budgets on diverse instances",
            ),
        ])

    else:
        # MIP / Exact Solvers (SCIP, CBC, HiGHS)
        nodes.extend([
            HATSNode(
                id="subsystem.presolve",
                level=HATSLevel.LEVEL_1_SUBSYSTEM,
                parent_id="goal.primary_performance",
                purpose="Simplify matrix, fix variables, and detect redundant constraints prior to search",
                source_refs=["src/scip/presol_*.c"],
                reads=["model_constraints", "variable_bounds"],
                writes=["reduced_model"],
            ),
            HATSNode(
                id="subsystem.lp_relaxation",
                level=HATSLevel.LEVEL_1_SUBSYSTEM,
                parent_id="goal.primary_performance",
                purpose="Solve continuous LP relaxation at search nodes to compute dual bounds",
                reads=["lp_state"],
                writes=["dual_bound", "fractional_solution"],
            ),
            HATSNode(
                id="subsystem.heuristics",
                level=HATSLevel.LEVEL_1_SUBSYSTEM,
                parent_id="goal.primary_performance",
                purpose="Discover improving primal integer feasible solutions to prune search tree",
                source_refs=["src/scip/heur_*.c"],
                reads=["lp_solution", "incumbent", "constraint_types"],
                writes=["incumbent", "best_solution"],
            ),
            HATSNode(
                id="subsystem.branching_and_nodesel",
                level=HATSLevel.LEVEL_1_SUBSYSTEM,
                parent_id="goal.primary_performance",
                purpose="Select next tree node and fractional variable to branch on",
                source_refs=["src/scip/branch_*.c", "src/scip/nodesel_*.c"],
                reads=["open_nodes", "pseudo_costs", "dual_bounds"],
                writes=["current_node", "tree_depth"],
            ),
        ])

        opportunities.extend([
            Opportunity(
                id="opp_primal_compound_exchange",
                title="Multi-variable primal exchange on tight constraints",
                target_nodes=["subsystem.heuristics"],
                source_files=["src/scip/heur_oneopt.c", "src/scip/heur_simplerounding.c"],
                proposal="Enumerate bounded compound flips (e.g. 6-flip with 3+3 meet-in-the-middle hash matching) to cross intermediate infeasible states",
                expected_effect="Drastically accelerate time to first/best feasible solution on combinatorial 0-1 constraints",
                cheapest_discriminating_experiment="Run pitbench bench on dev instances at primary budget",
            ),
            Opportunity(
                id="opp_nodesel_adaptive_plunge",
                title="Adaptive plunging depth and breadth search",
                target_nodes=["subsystem.branching_and_nodesel"],
                source_files=["src/scip/nodesel_estimate.c"],
                proposal="Dynamically adjust plunging offset and breadth-first depth based on tree depth and dual gap progress",
                expected_effect="Reduce overall branch-and-bound search tree size and PAR-2 runtime",
                cheapest_discriminating_experiment="Run pitbench bench --all-seeds on hard dev instances",
            ),
        ])

    return goal_statement, nodes, opportunities
