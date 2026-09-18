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
import org.chocosolver.solver.search.strategy.Search;
import org.chocosolver.solver.variables.IntVar;

public final class ExactSolverRunner {
    private ExactSolverRunner() {}

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
        model.setSeed(solverSeed);

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
