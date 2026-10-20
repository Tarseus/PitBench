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
        if (stringField(instance, "problem_kind", "bin_packing").equals("balanced_subset_sum")) {
            runBalancedSubsetSum(instance, solverSeed, budgetSec, searchStrategy);
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
