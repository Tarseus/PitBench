"""Dependency-free checks shared by the judge and visible development bench."""

from __future__ import annotations

import math


def format_multidimensional_knapsack_lp(
    weights: list[list[int]], capacities: list[int], *, profits: list[int] | None = None
) -> str:
    """Canonical LP representation shared by generation and public checking."""

    if (
        not weights
        or not weights[0]
        or len(weights) != len(capacities)
        or any(len(row) != len(weights[0]) for row in weights)
        or any(value < 0 for row in weights for value in row)
        or any(capacity < 0 for capacity in capacities)
    ):
        raise ValueError("invalid multidimensional knapsack model")
    item_count = len(weights[0])
    if profits is None:
        profits = [sum(row[item] for row in weights) for item in range(item_count)]
    if len(profits) != item_count or any(value < 0 for value in profits):
        raise ValueError("invalid knapsack profits")
    lines = [
        "Minimize",
        " obj: "
        + " ".join(f"- {profit} x{item}" for item, profit in enumerate(profits)),
        "Subject To",
    ]
    for dimension, (row, capacity) in enumerate(zip(weights, capacities, strict=True)):
        lines.append(
            f" c{dimension}: "
            + " + ".join(f"{weight} x{item}" for item, weight in enumerate(row))
            + f" <= {capacity}"
        )
    lines.extend(["Binary", *(f" x{item}" for item in range(item_count)), "End"])
    return "\n".join(lines) + "\n"


def edge_cost(
    coordinates: list[list[float]],
    first: int,
    second: int,
    distance_metric: str | None,
) -> float:
    x1, y1 = coordinates[first]
    x2, y2 = coordinates[second]
    distance = math.hypot(x2 - x1, y2 - y1)
    if distance_metric is None or distance_metric == "EXACT_2D":
        return distance
    if distance_metric == "EUC_2D":
        return float(math.floor(distance + 0.5))
    raise ValueError(f"unsupported CVRP distance metric: {distance_metric}")


def check_cvrp(instance: dict, solution: dict) -> dict:
    if instance.get("problem_kind") in {
        "heterogeneous_fleet_vrp",
        "multi_depot_vrp",
        "open_vrp",
    }:
        return check_routing_variant(instance, solution)
    coordinates = instance["coordinates"]
    demands = instance["demands"]
    capacity = instance["capacity"]
    depot = int(instance.get("depot", 0))
    distance_metric = instance.get("distance_metric")
    expected = set(range(len(coordinates))) - {depot}
    visited: list[int] = []
    objective = 0.0
    windows = instance.get("time_windows")
    services = instance.get("service_times", [0] * len(coordinates))
    if windows is not None and (
        len(windows) != len(coordinates)
        or len(services) != len(coordinates)
        or any(
            len(pair) != 2
            or not all(math.isfinite(value) for value in pair)
            or pair[0] < 0
            or pair[0] > pair[1]
            for pair in windows
        )
        or any(not math.isfinite(value) or value < 0 for value in services)
    ):
        return {"feasible": False, "detail": "invalid time-window model"}
    if (
        "num_vehicles" in instance
        and len(solution["routes"]) > instance["num_vehicles"]
    ):
        return {"feasible": False, "detail": "vehicle limit exceeded"}

    for route in solution["routes"]:
        if any(type(node) is not int or node not in expected for node in route):
            return {"feasible": False, "detail": "invalid customer index"}
        load = sum(demands[node] for node in route)
        if load > capacity:
            return {
                "feasible": False,
                "detail": f"route capacity {load} exceeds {capacity}",
            }
        path = [depot, *route, depot]
        clock = windows[depot][0] if windows is not None else 0
        for first, second in zip(path, path[1:]):
            distance = edge_cost(coordinates, first, second, distance_metric)
            objective += distance
            if windows is not None:
                clock = max(clock + services[first] + distance, windows[second][0])
                if clock > windows[second][1]:
                    return {
                        "feasible": False,
                        "detail": f"time window exceeded at node {second}",
                    }
        visited.extend(route)

    if len(visited) != len(set(visited)):
        return {"feasible": False, "detail": "duplicate customer"}
    if set(visited) != expected:
        return {"feasible": False, "detail": "customer set mismatch"}
    return {
        "feasible": True,
        "objective": objective,
        "detail": "independent VRPTW verification passed"
        if windows is not None
        else "independent CVRP verification passed",
    }


def check_routing_variant(instance: dict, solution: dict) -> dict:
    """Verify heterogeneous, multi-depot, and open-route normalized VRP inputs."""
    variant = instance.get("problem_kind")
    coordinates = instance.get("coordinates")
    demands = instance.get("demands")
    depots = instance.get("depots")
    customers = instance.get("customers")
    vehicle_types = instance.get("vehicle_types")
    routes = solution.get("routes")
    type_ids = solution.get("vehicle_type_ids")
    if (
        variant not in {"heterogeneous_fleet_vrp", "multi_depot_vrp", "open_vrp"}
        or not isinstance(coordinates, list)
        or not isinstance(demands, list)
        or len(demands) != len(coordinates)
        or not isinstance(depots, list)
        or not depots
        or any(type(node) is not int or not 0 <= node < len(coordinates) for node in depots)
        or len(set(depots)) != len(depots)
        or not isinstance(customers, list)
        or any(type(node) is not int or not 0 <= node < len(coordinates) for node in customers)
        or set(customers) & set(depots)
        or not isinstance(vehicle_types, list)
        or not vehicle_types
        or not isinstance(routes, list)
        or not isinstance(type_ids, list)
        or len(routes) != len(type_ids)
    ):
        return {"feasible": False, "detail": "invalid routing-variant input or solution"}
    if set(customers) | set(depots) != set(range(len(coordinates))):
        return {"feasible": False, "detail": "routing nodes are not a depot/client partition"}
    if any(
        len(point) != 2
        or any(not math.isfinite(value) for value in point)
        for point in coordinates
    ) or any(type(value) is not int or value < 0 for value in demands):
        return {"feasible": False, "detail": "invalid routing coordinates or demands"}
    if any(demands[depot] != 0 for depot in depots):
        return {"feasible": False, "detail": "depot demand must be zero"}
    metric = instance.get("distance_metric", "EUC_2D")
    usage = [0] * len(vehicle_types)
    visited, objective = [], 0.0
    for route, type_id in zip(routes, type_ids, strict=True):
        if (
            not isinstance(route, list)
            or not route
            or type(type_id) is not int
            or not 0 <= type_id < len(vehicle_types)
        ):
            return {"feasible": False, "detail": "invalid route or vehicle type"}
        vehicle = vehicle_types[type_id]
        start = vehicle.get("start_depot")
        end = vehicle.get("end_depot")
        capacity, fixed_cost, available = (
            vehicle.get("capacity"),
            vehicle.get("fixed_cost", 0),
            vehicle.get("count"),
        )
        if (
            type(start) is not int
            or start not in depots
            or (end is not None and (type(end) is not int or end not in depots))
            or type(capacity) is not int
            or capacity <= 0
            or type(available) is not int
            or available < 0
            or not math.isfinite(float(fixed_cost))
            or fixed_cost < 0
        ):
            return {"feasible": False, "detail": "invalid vehicle-type definition"}
        usage[type_id] += 1
        if usage[type_id] > available:
            return {"feasible": False, "detail": "vehicle-type availability exceeded"}
        if any(type(node) is not int or node not in customers for node in route):
            return {"feasible": False, "detail": "invalid customer index in route"}
        if sum(demands[node] for node in route) > capacity:
            return {"feasible": False, "detail": "route exceeds its vehicle capacity"}
        objective += float(fixed_cost)
        path = [start, *route] if end is None else [start, *route, end]
        objective += sum(
            edge_cost(coordinates, first, second, metric)
            for first, second in zip(path, path[1:])
        )
        visited.extend(route)
    if len(visited) != len(set(visited)) or set(visited) != set(customers):
        return {"feasible": False, "detail": "client coverage is incomplete or duplicated"}
    return {
        "feasible": True,
        "objective": objective,
        "detail": f"independent {variant} verification passed",
    }


def check_bin_packing(solution: dict, *, weights: list[int], capacity: int) -> dict:
    bins = solution.get("bins")
    if not isinstance(bins, list) or any(
        not isinstance(bin_items, list) or not bin_items for bin_items in bins
    ):
        return {
            "feasible": False,
            "detail": "bin packing solution requires nonempty bins",
        }
    item_indices = [item for bin_items in bins for item in bin_items]
    if any(type(item) is not int for item in item_indices):
        return {"feasible": False, "detail": "bin packing item index is not an integer"}
    if len(item_indices) != len(weights) or set(item_indices) != set(
        range(len(weights))
    ):
        return {
            "feasible": False,
            "detail": "bin packing items must appear exactly once",
        }
    for bin_index, bin_items in enumerate(bins):
        load = sum(weights[item] for item in bin_items)
        if load > capacity:
            return {"feasible": False, "detail": f"bin {bin_index} exceeds capacity"}
    return {
        "feasible": True,
        "objective": float(len(bins)),
        "detail": "exact integer bin packing verification passed",
    }


def check_graph_coloring(instance: dict, solution: dict) -> dict:
    vertex_count = instance.get("vertex_count")
    endpoints = instance.get("edge_endpoints")
    colors = solution.get("colors")
    if (
        type(vertex_count) is not int
        or vertex_count < 2
        or not isinstance(endpoints, list)
        or len(endpoints) % 2
        or not isinstance(colors, list)
        or len(colors) != vertex_count
        or any(
            type(color) is not int or not 0 <= color < vertex_count for color in colors
        )
    ):
        return {"feasible": False, "detail": "invalid graph coloring input or solution"}
    for first, second in zip(endpoints[::2], endpoints[1::2], strict=True):
        if (
            type(first) is not int
            or type(second) is not int
            or not 0 <= first < second < vertex_count
            or colors[first] == colors[second]
        ):
            return {
                "feasible": False,
                "detail": "graph edge is invalid or monochromatic",
            }
    return {
        "feasible": True,
        "objective": float(max(colors) + 1),
        "detail": "independent graph coloring verification passed",
    }


def check_resource_constrained_project_schedule(
    instance: dict, solution: dict
) -> dict:
    """Check non-preemptive RCPSP precedence, renewable resources, and makespan."""
    activities = instance.get("activities")
    arcs = instance.get("precedence_arcs")
    capacities = instance.get("resource_capacities")
    starts = solution.get("start_times")
    if (
        not isinstance(activities, list)
        or not activities
        or not isinstance(arcs, list)
        or not isinstance(capacities, list)
        or not capacities
        or not isinstance(starts, list)
        or len(starts) != len(activities)
        or any(type(value) is not int or value < 0 for value in starts)
    ):
        return {"feasible": False, "detail": "invalid RCPSP instance or start times"}
    resource_count = len(capacities)
    if any(type(value) is not int or value <= 0 for value in capacities):
        return {"feasible": False, "detail": "invalid renewable resource capacity"}
    durations, demands = [], []
    for index, activity in enumerate(activities):
        duration = activity.get("duration")
        usage = activity.get("resource_demands")
        if (
            activity.get("id") != index
            or type(duration) is not int
            or duration <= 0
            or not isinstance(usage, list)
            or len(usage) != resource_count
            or any(type(value) is not int or value < 0 for value in usage)
            or any(value > cap for value, cap in zip(usage, capacities, strict=True))
        ):
            return {"feasible": False, "detail": f"invalid RCPSP activity {index}"}
        durations.append(duration)
        demands.append(usage)
    observed_arcs = set()
    for arc in arcs:
        if (
            not isinstance(arc, list)
            or len(arc) != 2
            or any(type(value) is not int for value in arc)
            or not 0 <= arc[0] < arc[1] < len(activities)
            or tuple(arc) in observed_arcs
        ):
            return {"feasible": False, "detail": "invalid RCPSP precedence arc"}
        observed_arcs.add(tuple(arc))
        predecessor, successor = arc
        if starts[successor] < starts[predecessor] + durations[predecessor]:
            return {
                "feasible": False,
                "detail": f"precedence violation on arc {predecessor}->{successor}",
            }
    if (
        instance.get("activity_count", len(activities)) != len(activities)
        or instance.get("resource_count", resource_count) != resource_count
        or instance.get("durations", durations) != durations
        or instance.get("resource_demands_flat", [x for row in demands for x in row])
        != [x for row in demands for x in row]
        or instance.get("precedence_endpoints", [x for arc in arcs for x in arc])
        != [x for arc in arcs for x in arc]
    ):
        return {
            "feasible": False,
            "detail": "RCPSP flattened solver input differs from canonical activities/arcs",
        }
    makespan = max(
        start + duration
        for start, duration in zip(starts, durations, strict=True)
    )
    for time in range(makespan):
        for resource, capacity in enumerate(capacities):
            used = sum(
                demands[index][resource]
                for index, (start, duration) in enumerate(
                    zip(starts, durations, strict=True)
                )
                if start <= time < start + duration
            )
            if used > capacity:
                return {
                    "feasible": False,
                    "detail": f"renewable resource {resource} exceeds capacity at time {time}",
                }
    return {
        "feasible": True,
        "objective": float(makespan),
        "detail": "independent RCPSP precedence and renewable-resource verification passed",
    }
