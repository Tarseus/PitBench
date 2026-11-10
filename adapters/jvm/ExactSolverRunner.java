package org.pitbench.adapters;

import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.chocosolver.solver.Model;
import org.chocosolver.solver.Solution;
import org.chocosolver.solver.Solver;
import org.chocosolver.solver.constraints.Constraint;
import org.chocosolver.solver.constraints.ternary.PropXplusYeqZ;
import org.chocosolver.solver.search.strategy.Search;
import org.chocosolver.solver.variables.IntVar;
import org.chocosolver.solver.variables.Task;

public final class ExactSolverRunner {
    private ExactSolverRunner() {}

    private static void setSolverSeed(Model model, long solverSeed) {
        try {
            model.getClass().getMethod("setSeed", long.class).invoke(model, solverSeed);
        } catch (NoSuchMethodException unsupported) {
            if (solverSeed != 0) {
                throw new IllegalArgumentException(
                    "this Choco version does not support solver seeds other than zero",
                    unsupported
                );
            }
        } catch (ReflectiveOperationException error) {
            throw new IllegalStateException("failed to configure Choco solver seed", error);
        }
    }

    private static int integerField(String input, String name) {
        Pattern pattern = Pattern.compile("\\\"" + name + "\\\"\\s*:\\s*(\\d+)");
        Matcher matcher = pattern.matcher(input);
        if (!matcher.find()) {
            throw new IllegalArgumentException("missing integer field: " + name);
        }
        return Integer.parseInt(matcher.group(1));
    }

    private static int[] integerArray(String input, String name) {
        Pattern pattern = Pattern.compile(
            "\\\"" + name + "\\\"\\s*:\\s*\\[(.*?)\\]",
            Pattern.DOTALL
        );
        Matcher matcher = pattern.matcher(input);
        if (!matcher.find()) {
            throw new IllegalArgumentException("missing integer array: " + name);
        }
        String body = matcher.group(1).trim();
        if (body.isEmpty()) {
            return new int[0];
        }
        String[] fields = body.split(",");
        int[] values = new int[fields.length];
        for (int index = 0; index < fields.length; index++) {
            values[index] = Integer.parseInt(fields[index].trim());
        }
        return values;
    }

    private static String jsonIntArray(int[] values) {
        StringBuilder result = new StringBuilder("[");
        for (int index = 0; index < values.length; index++) {
            if (index > 0) {
                result.append(',');
            }
            result.append(values[index]);
        }
        return result.append(']').toString();
    }

    private static String stringField(String input, String name, String defaultValue) {
        Pattern pattern = Pattern.compile(
            "\\\"" + name + "\\\"\\s*:\\s*\\\"([^\\\"]+)\\\""
        );
        Matcher matcher = pattern.matcher(input);
        return matcher.find() ? matcher.group(1) : defaultValue;
    }

    private static long processCpuTime() {
        java.lang.management.OperatingSystemMXBean bean =
            ManagementFactory.getOperatingSystemMXBean();
        if (bean instanceof com.sun.management.OperatingSystemMXBean) {
            return ((com.sun.management.OperatingSystemMXBean) bean).getProcessCpuTime();
        }
        return -1L;
    }

    private static String binsJson(
        Solution solution,
        IntVar[] itemBins,
        int[] originalItemIndices,
        int binCount
    ) {
        List<List<Integer>> bins = new ArrayList<>();
        for (int bin = 0; bin < binCount; bin++) {
            bins.add(new ArrayList<>());
        }
        for (int item = 0; item < itemBins.length; item++) {
            bins.get(solution.getIntVal(itemBins[item])).add(originalItemIndices[item]);
        }
        StringBuilder json = new StringBuilder("[");
        for (int bin = 0; bin < bins.size(); bin++) {
            if (bin > 0) {
                json.append(',');
            }
            json.append('[');
            for (int position = 0; position < bins.get(bin).size(); position++) {
                if (position > 0) {
                    json.append(',');
                }
                json.append(bins.get(bin).get(position));
            }
            json.append(']');
        }
        return json.append(']').toString();
    }

    private static String colorsJson(Solution solution, IntVar[] colors) {
        StringBuilder json = new StringBuilder("[");
        for (int vertex = 0; vertex < colors.length; vertex++) {
            if (vertex > 0) {
                json.append(',');
            }
            json.append(solution.getIntVal(colors[vertex]));
        }
        return json.append(']').toString();
    }

    private static String selectedItemsJson(Solution solution, IntVar[] picks) {
        StringBuilder json = new StringBuilder("[");
        boolean first = true;
        for (int item = 0; item < picks.length; item++) {
            if (solution.getIntVal(picks[item]) == 1) {
                if (!first) {
                    json.append(',');
                }
                json.append(item);
                first = false;
            }
        }
        return json.append(']').toString();
    }

    private static void runBalancedSubsetSum(
        String instance,
        long solverSeed,
        double budgetSec,
        String searchStrategy
    ) {
        int[] weights = integerArray(instance, "weights");
        int cardinality = integerField(instance, "cardinality");
        int target = integerField(instance, "target");
        String domainMode = stringField(instance, "domain_mode", "bounded");
        if (weights.length < 2 || cardinality <= 0 || cardinality >= weights.length
            || !(domainMode.equals("bounded") || domainMode.equals("enumerated"))) {
            throw new IllegalArgumentException("invalid balanced subset-sum instance");
        }
        long totalLong = 0;
        for (int weight : weights) {
            if (weight <= 0) {
                throw new IllegalArgumentException("subset-sum weights must be positive");
            }
            totalLong += weight;
        }
        if (totalLong > Integer.MAX_VALUE || target > totalLong) {
            throw new IllegalArgumentException("invalid subset-sum sum or target");
        }
        int total = (int) totalLong;
        boolean bounded = domainMode.equals("bounded");
        Model model = new Model("PitBench balanced subset sum");
        IntVar[] picks = model.boolVarArray("pick", weights.length);
        IntVar[] prefix = new IntVar[weights.length + 1];
        prefix[0] = model.intVar("sum_0", 0);
        for (int item = 0; item < weights.length; item++) {
            IntVar contribution = model.intVar(
                "weight_" + item, new int[] {0, weights[item]}
            );
            model.times(picks[item], weights[item], contribution).post();
            prefix[item + 1] = model.intVar(
                "sum_" + (item + 1), 0, total, bounded
            );
            new Constraint(
                "prefix_sum", new PropXplusYeqZ(prefix[item], contribution, prefix[item + 1])
            ).post();
        }
        model.sum(picks, "=", cardinality).post();
        IntVar difference = model.intVar("difference", -total, total, bounded);
        model.arithm(difference, "+", model.intVar(target), "=", prefix[weights.length]).post();
        IntVar deviation = model.intVar("deviation", 0, total, bounded);
        model.absolute(deviation, difference).post();
        setSolverSeed(model, solverSeed);

        Solver solver = model.getSolver();
        switch (searchStrategy) {
            case "input_order_lb":
                solver.setSearch(Search.inputOrderLBSearch(picks));
                break;
            case "min_dom_lb":
                solver.setSearch(Search.minDomLBSearch(picks));
                break;
            case "dom_over_wdeg":
                solver.setSearch(Search.domOverWDegSearch(picks));
                break;
            case "activity_based":
                solver.setSearch(Search.activityBasedSearch(picks));
                break;
            default:
                throw new IllegalArgumentException("unsupported search_strategy: " + searchStrategy);
        }
        solver.limitTime((long) Math.ceil(budgetSec * 1000.0));
        long cpuStart = processCpuTime();
        Solution solution = solver.findOptimalSolution(deviation, false);
        long cpuEnd = processCpuTime();
        boolean hasSolution = solution != null;
        boolean timedOut = solver.isStopCriterionMet();
        String termination = hasSolution && !timedOut
            ? "optimal" : timedOut ? "time_limit" : "other";
        String solverStatus = hasSolution && !timedOut
            ? "Optimal" : timedOut ? "Time limit" : "No solution";
        String cpuTime = cpuStart >= 0 && cpuEnd >= cpuStart
            ? Double.toString((cpuEnd - cpuStart) / 1_000_000_000.0) : "null";
        String objective = hasSolution
            ? Integer.toString(solution.getIntVal(deviation)) : "null";
        String solutionJson = hasSolution
            ? "{\"selected_items\":" + selectedItemsJson(solution, picks) + "}"
            : "{}";
        System.out.println(
            "{\"valid\":" + hasSolution
                + ",\"has_solution\":" + hasSolution
                + ",\"objective\":" + objective
                + ",\"solver_status\":\"" + solverStatus + "\""
                + ",\"solver_termination\":\"" + termination + "\""
                + ",\"solver_runtime_sec\":" + solver.getTimeCount()
                + ",\"cpu_time_sec\":" + cpuTime
                + ",\"nodes\":" + solver.getNodeCount()
                + ",\"effective_parameters\":{\"search_strategy\":\""
                + searchStrategy + "\"}"
                + ",\"solution\":" + solutionJson + "}"
        );
    }

    private static void runGraphColoring(
        String instance,
        long solverSeed,
        double budgetSec,
        String searchStrategy
    ) {
        int vertexCount = integerField(instance, "vertex_count");
        int[] endpoints = integerArray(instance, "edge_endpoints");
        if (vertexCount < 2 || endpoints.length % 2 != 0) {
            throw new IllegalArgumentException("invalid graph coloring instance");
        }
        Model model = new Model("PitBench graph coloring");
        IntVar[] colors = model.intVarArray("color", vertexCount, 0, vertexCount - 1, false);
        model.arithm(colors[0], "=", 0).post();
        for (int edge = 0; edge < endpoints.length; edge += 2) {
            int first = endpoints[edge];
            int second = endpoints[edge + 1];
            if (first < 0 || first >= second || second >= vertexCount) {
                throw new IllegalArgumentException("invalid graph coloring edge");
            }
            model.arithm(colors[first], "!=", colors[second]).post();
        }
        IntVar maximumColor = model.intVar("maximumColor", 0, vertexCount - 1, true);
        model.max(maximumColor, colors).post();
        setSolverSeed(model, solverSeed);
        Solver solver = model.getSolver();
        switch (searchStrategy) {
            case "input_order_lb":
                solver.setSearch(Search.inputOrderLBSearch(colors));
                break;
            case "min_dom_lb":
                solver.setSearch(Search.minDomLBSearch(colors));
                break;
            case "dom_over_wdeg":
                solver.setSearch(Search.domOverWDegSearch(colors));
                break;
            case "activity_based":
                solver.setSearch(Search.activityBasedSearch(colors));
                break;
            default:
                throw new IllegalArgumentException("unsupported search_strategy: " + searchStrategy);
        }
        solver.limitTime((long) Math.ceil(budgetSec * 1000.0));
        long cpuStart = processCpuTime();
        Solution solution = solver.findOptimalSolution(maximumColor, false);
        long cpuEnd = processCpuTime();
        boolean hasSolution = solution != null;
        boolean timedOut = solver.isStopCriterionMet();
        String termination = hasSolution && !timedOut
            ? "optimal"
            : timedOut ? "time_limit" : "other";
        String solverStatus = hasSolution && !timedOut
            ? "Optimal"
            : timedOut ? "Time limit" : "No solution";
        String cpuTime = cpuStart >= 0 && cpuEnd >= cpuStart
            ? Double.toString((cpuEnd - cpuStart) / 1_000_000_000.0)
            : "null";
        String objective = hasSolution
            ? Integer.toString(solution.getIntVal(maximumColor) + 1)
            : "null";
        String solutionJson = hasSolution
            ? "{\"colors\":" + colorsJson(solution, colors) + "}"
            : "{}";
        System.out.println(
            "{\"valid\":" + hasSolution
                + ",\"has_solution\":" + hasSolution
                + ",\"objective\":" + objective
                + ",\"solver_status\":\"" + solverStatus + "\""
                + ",\"solver_termination\":\"" + termination + "\""
                + ",\"solver_runtime_sec\":" + solver.getTimeCount()
                + ",\"cpu_time_sec\":" + cpuTime
                + ",\"nodes\":" + solver.getNodeCount()
                + ",\"effective_parameters\":{\"search_strategy\":\""
                + searchStrategy + "\"}"
                + ",\"solution\":" + solutionJson + "}"
        );
    }

    private static int[] jobShopIntegers(String instance) {
        String trimmed = instance.trim();
        if (trimmed.isEmpty()) {
            throw new IllegalArgumentException("job-shop instance is empty");
        }
        String[] tokens = trimmed.split("\\s+");
        int[] values = new int[tokens.length];
        for (int index = 0; index < tokens.length; index++) {
            values[index] = Integer.parseInt(tokens[index]);
        }
        return values;
    }

    private static void selectStarts(Solver solver, IntVar[] starts, String strategy) {
        switch (strategy) {
            case "input_order_lb":
                solver.setSearch(Search.inputOrderLBSearch(starts));
                break;
            case "min_dom_lb":
                solver.setSearch(Search.minDomLBSearch(starts));
                break;
            case "dom_over_wdeg":
                solver.setSearch(Search.domOverWDegSearch(starts));
                break;
            case "activity_based":
                solver.setSearch(Search.activityBasedSearch(starts));
                break;
            default:
                throw new IllegalArgumentException("unsupported search_strategy: " + strategy);
        }
    }

    private static void printScheduleResult(
        Solver solver,
        long cpuStart,
        long cpuEnd,
        Solution solution,
        IntVar objective,
        IntVar[] starts,
        String searchStrategy
    ) {
        boolean hasSolution = solution != null;
        boolean timedOut = solver.isStopCriterionMet();
        String termination = hasSolution && !timedOut
            ? "optimal" : timedOut ? "time_limit" : "other";
        String solverStatus = hasSolution && !timedOut
            ? "Optimal" : timedOut ? "Time limit" : "No solution";
        String cpuTime = cpuStart >= 0 && cpuEnd >= cpuStart
            ? Double.toString((cpuEnd - cpuStart) / 1_000_000_000.0) : "null";
        int[] startTimes = new int[starts.length];
        if (hasSolution) {
            for (int index = 0; index < starts.length; index++) {
                startTimes[index] = solution.getIntVal(starts[index]);
            }
        }
        String solutionJson = hasSolution
            ? "{\"start_times\":" + jsonIntArray(startTimes) + "}"
            : "{}";
        String objectiveJson = hasSolution
            ? Integer.toString(solution.getIntVal(objective)) : "null";
        System.out.println(
            "{\"valid\":" + hasSolution
                + ",\"has_solution\":" + hasSolution
                + ",\"objective\":" + objectiveJson
                + ",\"solver_status\":\"" + solverStatus + "\""
                + ",\"solver_termination\":\"" + termination + "\""
                + ",\"solver_runtime_sec\":" + solver.getTimeCount()
                + ",\"cpu_time_sec\":" + cpuTime
                + ",\"nodes\":" + solver.getNodeCount()
                + ",\"effective_parameters\":{\"search_strategy\":\""
                + searchStrategy + "\"}"
                + ",\"solution\":" + solutionJson + "}"
        );
    }

    private static void runJobShop(
        String instance,
        long solverSeed,
        double budgetSec,
        String searchStrategy
    ) {
        int[] input = jobShopIntegers(instance);
        if (input.length < 2 || input[0] <= 0 || input[1] <= 0) {
            throw new IllegalArgumentException("invalid job-shop header");
        }
        int jobCount = input[0];
        int machineCount = input[1];
        int operationCount = jobCount * machineCount;
        if (input.length != 2 + 2 * operationCount) {
            throw new IllegalArgumentException("job-shop operation count does not match header");
        }
        int[][] operationByMachine = new int[machineCount][jobCount];
        int[] machineCounts = new int[machineCount];
        IntVar[] starts = new IntVar[operationCount];
        IntVar[] ends = new IntVar[operationCount];
        int[] machines = new int[operationCount];
        int[] durations = new int[operationCount];
        int horizon = 0;
        int cursor = 2;
        for (int job = 0; job < jobCount; job++) {
            for (int operation = 0; operation < machineCount; operation++) {
                int index = job * machineCount + operation;
                int machine = input[cursor++];
                int duration = input[cursor++];
                if (machine < 0 || machine >= machineCount || duration <= 0) {
                    throw new IllegalArgumentException("invalid job-shop operation");
                }
                machines[index] = machine;
                durations[index] = duration;
                operationByMachine[machine][machineCounts[machine]++] = index;
                horizon += duration;
            }
        }

        Model model = new Model("PitBench job-shop scheduling");
        for (int index = 0; index < operationCount; index++) {
            starts[index] = model.intVar("start_" + index, 0, horizon, true);
            ends[index] = model.intVar("end_" + index, 0, horizon, true);
            model.arithm(ends[index], "=", starts[index], "+", durations[index]).post();
        }
        for (int job = 0; job < jobCount; job++) {
            for (int operation = 1; operation < machineCount; operation++) {
                int previous = job * machineCount + operation - 1;
                int current = job * machineCount + operation;
                model.arithm(starts[current], ">=", ends[previous]).post();
            }
        }
        for (int machine = 0; machine < machineCount; machine++) {
            for (int first = 0; first < machineCounts[machine]; first++) {
                for (int second = first + 1; second < machineCounts[machine]; second++) {
                    int left = operationByMachine[machine][first];
                    int right = operationByMachine[machine][second];
                    model.or(
                        model.arithm(ends[left], "<=", starts[right]),
                        model.arithm(ends[right], "<=", starts[left])
                    ).post();
                }
            }
        }
        IntVar makespan = model.intVar("makespan", 0, horizon, true);
        model.max(makespan, ends).post();
        setSolverSeed(model, solverSeed);
        Solver solver = model.getSolver();
        selectStarts(solver, starts, searchStrategy);
        solver.limitTime((long) Math.ceil(budgetSec * 1000.0));
        long cpuStart = processCpuTime();
        Solution solution = solver.findOptimalSolution(makespan, false);
        long cpuEnd = processCpuTime();
        printScheduleResult(
            solver, cpuStart, cpuEnd, solution, makespan, starts, searchStrategy
        );
    }

    private static void runRcpsp(
        String instance,
        long solverSeed,
        double budgetSec,
        String searchStrategy
    ) {
        int activityCount = integerField(instance, "activity_count");
        int resourceCount = integerField(instance, "resource_count");
        int[] durations = integerArray(instance, "durations");
        int[] demands = integerArray(instance, "resource_demands_flat");
        int[] capacities = integerArray(instance, "resource_capacities");
        int[] precedence = integerArray(instance, "precedence_endpoints");
        if (activityCount < 2 || resourceCount < 1
            || durations.length != activityCount
            || demands.length != activityCount * resourceCount
            || capacities.length != resourceCount || precedence.length % 2 != 0) {
            throw new IllegalArgumentException("invalid flattened RCPSP data");
        }
        int horizon = 0;
        for (int duration : durations) {
            if (duration <= 0) throw new IllegalArgumentException("invalid activity duration");
            horizon += duration;
        }
        Model model = new Model("PitBench renewable-resource project scheduling");
        IntVar[] starts = model.intVarArray("start", activityCount, 0, horizon, true);
        IntVar[] ends = new IntVar[activityCount];
        for (int activity = 0; activity < activityCount; activity++) {
            ends[activity] = model.intVar("end_" + activity, 0, horizon, true);
            model.arithm(ends[activity], "=", starts[activity], "+", durations[activity]).post();
        }
        for (int index = 0; index < precedence.length; index += 2) {
            int predecessor = precedence[index];
            int successor = precedence[index + 1];
            if (predecessor < 0 || predecessor >= activityCount
                || successor < 0 || successor >= activityCount) {
                throw new IllegalArgumentException("RCPSP precedence endpoint outside activity range");
            }
            model.arithm(starts[successor], ">=", ends[predecessor]).post();
        }
        for (int resource = 0; resource < resourceCount; resource++) {
            List<Task> resourceTasks = new ArrayList<>();
            List<IntVar> resourceHeights = new ArrayList<>();
            for (int activity = 0; activity < activityCount; activity++) {
                int demand = demands[activity * resourceCount + resource];
                if (demand < 0 || demand > capacities[resource]) {
                    throw new IllegalArgumentException("RCPSP demand exceeds resource capacity");
                }
                if (demand > 0) {
                    resourceTasks.add(
                        new Task(starts[activity], durations[activity], ends[activity])
                    );
                    resourceHeights.add(model.intVar("d_" + resource + "_" + activity, demand));
                }
            }
            model.cumulative(
                resourceTasks.toArray(new Task[0]),
                resourceHeights.toArray(new IntVar[0]),
                model.intVar("capacity_" + resource, capacities[resource])
            ).post();
        }
        IntVar makespan = model.intVar("makespan", 0, horizon, true);
        model.max(makespan, ends).post();
        setSolverSeed(model, solverSeed);
        Solver solver = model.getSolver();
        selectStarts(solver, starts, searchStrategy);
        solver.limitTime((long) Math.ceil(budgetSec * 1000.0));
        long cpuStart = processCpuTime();
        Solution solution = solver.findOptimalSolution(makespan, false);
        long cpuEnd = processCpuTime();
        printScheduleResult(
            solver, cpuStart, cpuEnd, solution, makespan, starts, searchStrategy
        );
    }

    public static void main(String[] args) throws IOException {
        if (args.length < 4 || args.length > 5) {
            throw new IllegalArgumentException(
                "expected instance_path solver_seed budget_sec threads [parameters_json]"
            );
        }
        Path instancePath = Path.of(args[0]);
        long solverSeed = Long.parseLong(args[1]);
        double budgetSec = Double.parseDouble(args[2]);
        int threads = Integer.parseInt(args[3]);
        if (!Double.isFinite(budgetSec) || budgetSec <= 0 || threads != 1) {
            throw new IllegalArgumentException("invalid budget or thread count");
        }
        String parameters = args.length == 5 ? args[4] : "{}";
        String searchStrategy = stringField(parameters, "search_strategy", "input_order_lb");
        String instance = Files.readString(instancePath);
        if (instancePath.getFileName().toString().endsWith(".txt")) {
            runJobShop(instance, solverSeed, budgetSec, searchStrategy);
            return;
        }
        if (stringField(instance, "problem_kind", "bin_packing").equals("balanced_subset_sum")) {
            runBalancedSubsetSum(instance, solverSeed, budgetSec, searchStrategy);
            return;
        }
        if (stringField(instance, "problem_kind", "bin_packing").equals("resource_constrained_project_scheduling")) {
            runRcpsp(instance, solverSeed, budgetSec, searchStrategy);
            return;
        }
        if (stringField(instance, "problem_kind", "bin_packing").equals("graph_coloring")) {
            runGraphColoring(instance, solverSeed, budgetSec, searchStrategy);
            return;
        }
        int capacity = integerField(instance, "capacity");
        int[] originalWeights = integerArray(instance, "weights");
        if (originalWeights.length == 0) {
            throw new IllegalArgumentException("bin packing instance has no items");
        }
        long totalWeight = 0;
        for (int weight : originalWeights) {
            if (weight <= 0 || weight > capacity) {
                throw new IllegalArgumentException("item weight outside capacity");
            }
            totalWeight += weight;
        }
        int lowerBound = (int) ((totalWeight + capacity - 1) / capacity);
        Integer[] sortedItemOrder = new Integer[originalWeights.length];
        for (int item = 0; item < originalWeights.length; item++) {
            sortedItemOrder[item] = item;
        }
        Arrays.sort(
            sortedItemOrder,
            Comparator.comparingInt((Integer item) -> originalWeights[item])
                .reversed()
                .thenComparingInt(item -> item)
        );
        int[] weights = new int[originalWeights.length];
        int[] originalItemIndices = new int[originalWeights.length];
        for (int item = 0; item < sortedItemOrder.length; item++) {
            originalItemIndices[item] = sortedItemOrder[item];
            weights[item] = originalWeights[originalItemIndices[item]];
        }

        Model model = new Model("PitBench exact bin packing");
        IntVar[] itemBins = model.intVarArray(
            "itemBin",
            weights.length,
            0,
            weights.length - 1,
            false
        );
        IntVar[] binLoads = model.intVarArray(
            "binLoad",
            weights.length,
            0,
            capacity,
            true
        );
        model.binPacking(itemBins, weights, binLoads, 0).post();
        for (int item = 0; item < itemBins.length; item++) {
            model.arithm(itemBins[item], "<=", item).post();
        }
        IntVar maximumBin = model.intVar(
            "maximumBin",
            lowerBound - 1,
            weights.length - 1,
            true
        );
        model.max(maximumBin, itemBins).post();
        setSolverSeed(model, solverSeed);

        Solver solver = model.getSolver();
        switch (searchStrategy) {
            case "input_order_lb":
                solver.setSearch(Search.inputOrderLBSearch(itemBins));
                break;
            case "min_dom_lb":
                solver.setSearch(Search.minDomLBSearch(itemBins));
                break;
            case "dom_over_wdeg":
                solver.setSearch(Search.domOverWDegSearch(itemBins));
                break;
            case "activity_based":
                solver.setSearch(Search.activityBasedSearch(itemBins));
                break;
            default:
                throw new IllegalArgumentException("unsupported search_strategy: " + searchStrategy);
        }
        solver.limitTime((long) Math.ceil(budgetSec * 1000.0));
        long cpuStart = processCpuTime();
        Solution solution = solver.findOptimalSolution(maximumBin, false);
        long cpuEnd = processCpuTime();
        boolean hasSolution = solution != null;
        boolean timedOut = solver.isStopCriterionMet();
        String termination = hasSolution && !timedOut
            ? "optimal"
            : timedOut ? "time_limit" : "other";
        String solverStatus = hasSolution && !timedOut
            ? "Optimal"
            : timedOut ? "Time limit" : "No solution";
        String cpuTime = cpuStart >= 0 && cpuEnd >= cpuStart
            ? Double.toString((cpuEnd - cpuStart) / 1_000_000_000.0)
            : "null";
        String objective = hasSolution
            ? Integer.toString(solution.getIntVal(maximumBin) + 1)
            : "null";
        String solutionJson = hasSolution
            ? "{\"bins\":"
                + binsJson(
                    solution,
                    itemBins,
                    originalItemIndices,
                    solution.getIntVal(maximumBin) + 1
                )
                + "}"
            : "{}";
        System.out.println(
            "{\"valid\":" + hasSolution
                + ",\"has_solution\":" + hasSolution
                + ",\"objective\":" + objective
                + ",\"solver_status\":\"" + solverStatus + "\""
                + ",\"solver_termination\":\"" + termination + "\""
                + ",\"solver_runtime_sec\":" + solver.getTimeCount()
                + ",\"cpu_time_sec\":" + cpuTime
                + ",\"nodes\":" + solver.getNodeCount()
                + ",\"effective_parameters\":{\"search_strategy\":\""
                + searchStrategy + "\"}"
                + ",\"solution\":" + solutionJson + "}"
        );
    }
}
