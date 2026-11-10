#include <gecode/int.hh>
#include <gecode/search.hh>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace Gecode;

static void enforce_duration(Space& home, IntVar start, IntVar end,
                             int duration) {
  IntVarArgs variables(2);
  variables[0] = end;
  variables[1] = start;
  IntArgs coefficients(2);
  coefficients[0] = 1;
  coefficients[1] = -1;
  linear(home, coefficients, variables, IRT_EQ, duration);
}

class BinPackingModel : public Space {
public:
  IntVarArray loads;
  IntVarArray bins;
  int capacity;
  std::vector<int> weights;

  BinPackingModel(int n_bins, int capacity_, std::vector<int> const& weights_,
                  IntPropLevel level)
      : loads(*this, n_bins, 0, capacity_),
        bins(*this, static_cast<int>(weights_.size()), 0, n_bins - 1),
        capacity(capacity_), weights(weights_) {
    IntArgs sizes(static_cast<int>(weights.size()));
    for (int i = 0; i < static_cast<int>(weights.size()); ++i)
      sizes[i] = weights[i];
    binpacking(*this, loads, bins, sizes, level);
    branch(*this, bins, INT_VAR_SIZE_MIN(), INT_VAL_MIN());
  }

  BinPackingModel(BinPackingModel& other)
      : Space(other), capacity(other.capacity), weights(other.weights) {
    loads.update(*this, other.loads);
    bins.update(*this, other.bins);
  }

  Space* copy() override { return new BinPackingModel(*this); }
};

class BalancedSubsetSumModel : public Space {
public:
  BoolVarArray picks;
  IntVar sum;
  IntVar deviation;
  IntArgs weights;
  int cardinality;
  int target;

  BalancedSubsetSumModel(std::vector<int> const& weights_, int cardinality_, int target_)
      : picks(*this, static_cast<int>(weights_.size()), 0, 1),
        sum(*this, 0, std::accumulate(weights_.begin(), weights_.end(), 0)),
        deviation(*this, 0, std::accumulate(weights_.begin(), weights_.end(), 0)),
        weights(static_cast<int>(weights_.size())), cardinality(cardinality_), target(target_) {
    for (int i = 0; i < static_cast<int>(weights_.size()); ++i) weights[i] = weights_[i];
    IntArgs ones(static_cast<int>(weights_.size()));
    for (int i = 0; i < static_cast<int>(weights_.size()); ++i) ones[i] = 1;
    linear(*this, ones, picks, IRT_EQ, cardinality);
    linear(*this, weights, picks, IRT_EQ, sum);
    IntArgs lower_coeffs(2), upper_coeffs(2);
    IntVarArgs values(2);
    lower_coeffs[0] = 1; lower_coeffs[1] = -1;
    upper_coeffs[0] = 1; upper_coeffs[1] = 1;
    values[0] = sum; values[1] = deviation;
    linear(*this, lower_coeffs, values, IRT_LQ, target);
    linear(*this, upper_coeffs, values, IRT_GQ, target);
    branch(*this, picks, BOOL_VAR_NONE(), BOOL_VAL_MIN());
  }

  BalancedSubsetSumModel(BalancedSubsetSumModel& other)
      : Space(other), weights(other.weights), cardinality(other.cardinality),
        target(other.target) {
    picks.update(*this, other.picks);
    sum.update(*this, other.sum);
    deviation.update(*this, other.deviation);
  }

  Space* copy() override { return new BalancedSubsetSumModel(*this); }

  void constrain(Space const& best) override {
    auto const& solution = static_cast<BalancedSubsetSumModel const&>(best);
    int selected_sum = 0;
    for (int i = 0; i < solution.picks.size(); ++i) {
      if (solution.picks[i].val() == 1) selected_sum += solution.weights[i];
    }
    rel(*this, deviation, IRT_LE,
        std::abs(selected_sum - solution.target) - 1);
  }
};

class GraphColoringModel : public Space {
public:
  IntVarArray colors;
  IntVar maximum_color;
  std::vector<int> endpoints;

  GraphColoringModel(int vertex_count, std::vector<int> const& endpoints_)
      : colors(*this, vertex_count, 0, vertex_count - 1),
        maximum_color(*this, 0, vertex_count - 1), endpoints(endpoints_) {
    rel(*this, colors[0], IRT_EQ, 0);
    for (size_t edge = 0; edge < endpoints.size(); edge += 2) {
      int first = endpoints[edge], second = endpoints[edge + 1];
      if (first < 0 || second < 0 || first >= vertex_count || second >= vertex_count
          || first == second) throw std::invalid_argument("invalid graph coloring edge");
      rel(*this, colors[first], IRT_NQ, colors[second]);
    }
    max(*this, colors, maximum_color);
    branch(*this, colors, INT_VAR_SIZE_MIN(), INT_VAL_MIN());
  }

  GraphColoringModel(GraphColoringModel& other)
      : Space(other), endpoints(other.endpoints) {
    colors.update(*this, other.colors);
    maximum_color.update(*this, other.maximum_color);
  }

  Space* copy() override { return new GraphColoringModel(*this); }

  void constrain(Space const& best) override {
    auto const& solution = static_cast<GraphColoringModel const&>(best);
    int best_maximum_color = 0;
    for (int i = 0; i < solution.colors.size(); ++i) {
      best_maximum_color = std::max(best_maximum_color,
                                    solution.colors[i].val());
    }
    rel(*this, maximum_color, IRT_LE, best_maximum_color - 1);
  }
};

class JobShopModel : public Space {
public:
  IntVarArray starts;
  IntVarArray ends;
  IntVar makespan;
  int job_count;
  int machine_count;
  std::vector<int> machines;
  std::vector<int> durations;

  JobShopModel(int jobs, int machines_count, std::vector<int> const& machines_,
               std::vector<int> const& durations_)
      : starts(*this, jobs * machines_count, 0,
               std::accumulate(durations_.begin(), durations_.end(), 0)),
        ends(*this, jobs * machines_count, 0,
             std::accumulate(durations_.begin(), durations_.end(), 0)),
        makespan(*this, 0, std::accumulate(durations_.begin(), durations_.end(), 0)),
        job_count(jobs), machine_count(machines_count),
        machines(machines_), durations(durations_) {
    int horizon = std::accumulate(durations.begin(), durations.end(), 0);
    std::vector<std::vector<int>> by_machine(machine_count);
    for (int operation = 0; operation < job_count * machine_count; ++operation) {
      if (machines[operation] < 0 || machines[operation] >= machine_count
          || durations[operation] <= 0) throw std::invalid_argument("invalid job-shop operation");
      enforce_duration(*this, starts[operation], ends[operation],
                       durations[operation]);
      by_machine[machines[operation]].push_back(operation);
    }
    for (int job = 0; job < job_count; ++job) {
      for (int operation = 1; operation < machine_count; ++operation) {
        int previous = job * machine_count + operation - 1;
        int current = job * machine_count + operation;
        rel(*this, starts[current], IRT_GQ, ends[previous]);
      }
    }
    for (auto const& operations : by_machine) {
      IntVarArgs machine_starts(static_cast<int>(operations.size()));
      IntArgs machine_durations(static_cast<int>(operations.size()));
      IntArgs usages(static_cast<int>(operations.size()));
      for (int i = 0; i < static_cast<int>(operations.size()); ++i) {
        machine_starts[i] = starts[operations[i]];
        machine_durations[i] = durations[operations[i]];
        usages[i] = 1;
      }
      cumulative(*this, 1, machine_starts, machine_durations, usages, IPL_BND);
    }
    max(*this, ends, makespan);
    branch(*this, starts, INT_VAR_SIZE_MIN(), INT_VAL_MIN());
    (void)horizon;
  }

  JobShopModel(JobShopModel& other)
      : Space(other), job_count(other.job_count),
        machine_count(other.machine_count), machines(other.machines), durations(other.durations) {
    starts.update(*this, other.starts);
    ends.update(*this, other.ends);
    makespan.update(*this, other.makespan);
  }

  Space* copy() override { return new JobShopModel(*this); }

  void constrain(Space const& best) override {
    auto const& solution = static_cast<JobShopModel const&>(best);
    int best_makespan = 0;
    for (int operation = 0; operation < solution.starts.size(); ++operation) {
      best_makespan = std::max(
          best_makespan,
          solution.starts[operation].val() + solution.durations[operation]);
    }
    rel(*this, makespan, IRT_LE, best_makespan - 1);
  }
};

class RcpspModel : public Space {
public:
  IntVarArray starts;
  IntVarArray ends;
  IntVar makespan;
  int activity_count;
  int resource_count;
  std::vector<int> durations;
  std::vector<int> demands;
  std::vector<int> capacities;
  std::vector<int> precedence;

  RcpspModel(int activity_count_, int resource_count_, std::vector<int> const& durations_,
             std::vector<int> const& demands_, std::vector<int> const& capacities_,
             std::vector<int> const& precedence_)
      : starts(*this, activity_count_, 0,
               std::accumulate(durations_.begin(), durations_.end(), 0)),
        ends(*this, activity_count_, 0,
             std::accumulate(durations_.begin(), durations_.end(), 0)),
        makespan(*this, 0, std::accumulate(durations_.begin(), durations_.end(), 0)),
        activity_count(activity_count_), resource_count(resource_count_),
        durations(durations_), demands(demands_), capacities(capacities_), precedence(precedence_) {
    if (activity_count < 2 || resource_count < 1
        || static_cast<int>(durations.size()) != activity_count
        || static_cast<int>(demands.size()) != activity_count * resource_count
        || static_cast<int>(capacities.size()) != resource_count
        || precedence.size() % 2 != 0) throw std::invalid_argument("invalid flattened RCPSP data");
    for (int activity = 0; activity < activity_count; ++activity) {
      if (durations[activity] <= 0) throw std::invalid_argument("invalid activity duration");
      enforce_duration(*this, starts[activity], ends[activity],
                       durations[activity]);
    }
    for (size_t index = 0; index < precedence.size(); index += 2) {
      int predecessor = precedence[index], successor = precedence[index + 1];
      if (predecessor < 0 || predecessor >= activity_count
          || successor < 0 || successor >= activity_count) throw std::invalid_argument("RCPSP precedence outside activity range");
      rel(*this, starts[successor], IRT_GQ, ends[predecessor]);
    }
    for (int resource = 0; resource < resource_count; ++resource) {
      IntVarArgs resource_starts(activity_count);
      IntArgs resource_durations(activity_count), usages(activity_count);
      for (int activity = 0; activity < activity_count; ++activity) {
        int demand = demands[activity * resource_count + resource];
        if (demand < 0 || demand > capacities[resource]) throw std::invalid_argument("RCPSP demand exceeds capacity");
        resource_starts[activity] = starts[activity];
        resource_durations[activity] = durations[activity];
        usages[activity] = demand;
      }
      cumulative(*this, capacities[resource], resource_starts,
                 resource_durations, usages, IPL_BND);
    }
    max(*this, ends, makespan);
    branch(*this, starts, INT_VAR_SIZE_MIN(), INT_VAL_MIN());
  }

  RcpspModel(RcpspModel& other)
      : Space(other), activity_count(other.activity_count),
        resource_count(other.resource_count), durations(other.durations), demands(other.demands),
        capacities(other.capacities), precedence(other.precedence) {
    starts.update(*this, other.starts);
    ends.update(*this, other.ends);
    makespan.update(*this, other.makespan);
  }

  Space* copy() override { return new RcpspModel(*this); }

  void constrain(Space const& best) override {
    auto const& solution = static_cast<RcpspModel const&>(best);
    int best_makespan = 0;
    for (int activity = 0; activity < solution.starts.size(); ++activity) {
      best_makespan = std::max(
          best_makespan,
          solution.starts[activity].val() + solution.durations[activity]);
    }
    rel(*this, makespan, IRT_LE, best_makespan - 1);
  }
};

static std::string json_array(std::vector<int> const& xs) {
  std::string out = "[";
  for (size_t i = 0; i < xs.size(); ++i) {
    if (i) out += ",";
    out += std::to_string(xs[i]);
  }
  return out + "]";
}

struct SearchStats {
  unsigned long long nodes = 0;
  unsigned long long failures = 0;
  double elapsed_ms = 0;
  bool stopped = false;
};

template <class ModelType, class... Args>
static ModelType* optimize(double budget_ms, SearchStats& result, Args&&... args) {
  auto started = std::chrono::steady_clock::now();
  Search::TimeStop stop(budget_ms);
  Search::Options options;
  options.threads = 1;
  options.stop = &stop;
  auto* model = new ModelType(std::forward<Args>(args)...);
  BAB<ModelType> engine(model, options);
  delete model;
  ModelType* best = nullptr;
  while (ModelType* candidate = engine.next()) {
    delete best;
    best = candidate;
  }
  auto statistics = engine.statistics();
  result.nodes = statistics.node;
  result.failures = statistics.fail;
  result.elapsed_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
  result.stopped = engine.stopped();
  return best;
}

static void print_result(bool has_solution, bool optimal, double elapsed_ms,
                         unsigned long long nodes, unsigned long long failures,
                         std::string const& objective, std::string const& solution,
                         std::string const& extra_fields = "") {
  std::string status = optimal ? "Optimal" : has_solution ? "Time limit"
                                                   : "Time limit";
  std::string termination = optimal ? "optimal" : has_solution ? "time_limit"
                                                       : "time_limit";
  std::cout << "{\"valid\":" << (has_solution ? "true" : "false")
            << ",\"has_solution\":" << (has_solution ? "true" : "false")
            << ",\"objective\":" << (has_solution ? objective : "null")
            << ",\"primal_bound\":" << (has_solution ? objective : "null")
            << ",\"dual_bound\":" << (optimal ? objective : "null")
            << ",\"nodes\":" << nodes << ",\"iterations\":" << failures
            << ",\"solver_status\":\"" << status
            << "\",\"solver_termination\":\"" << termination
            << "\",\"solution\":" << (has_solution ? solution : "{}")
            << ",\"effective_parameters\":{}"
            << ",\"trajectory\":[{\"event\":\"final\",\"time_sec\":"
            << elapsed_ms / 1000.0 << ",\"primal_bound\":"
            << (has_solution ? objective : "null") << ",\"dual_bound\":"
            << (optimal ? objective : "null") << ",\"nodes\":" << nodes
            << "}]" << extra_fields << "}\n";
}

static void run_balanced_subset_sum(int argc, char** argv) {
  if (argc < 7) throw std::invalid_argument("invalid balanced subset-sum arguments");
  double budget_ms = std::stod(argv[2]);
  int cardinality = std::stoi(argv[3]);
  int target = std::stoi(argv[4]);
  int count = std::stoi(argv[5]);
  if (count <= 1 || argc != 6 + count || cardinality <= 0 || cardinality >= count
      || budget_ms <= 0) throw std::invalid_argument("invalid balanced subset-sum dimensions");
  std::vector<int> weights(count);
  int total = 0;
  for (int i = 0; i < count; ++i) {
    weights[i] = std::stoi(argv[6 + i]);
    if (weights[i] <= 0) throw std::invalid_argument("subset-sum weights must be positive");
    total += weights[i];
  }
  SearchStats stats;
  auto* solution = optimize<BalancedSubsetSumModel>(
      budget_ms, stats, weights, cardinality, target);
  if (solution != nullptr) {
    std::vector<int> selected;
    int selected_sum = 0;
    for (int i = 0; i < solution->picks.size(); ++i) {
      if (solution->picks[i].val() == 1) {
        selected.push_back(i);
        selected_sum += solution->weights[i];
      }
    }
    int objective = std::abs(selected_sum - solution->target);
    print_result(true, !stats.stopped, stats.elapsed_ms, stats.nodes, stats.failures,
                 std::to_string(objective), "{\"selected_items\":" + json_array(selected) + "}");
    delete solution;
  } else {
    print_result(false, false, stats.elapsed_ms, stats.nodes, stats.failures,
                 "null", "{}");
  }
  (void)total;
}

static void run_graph_coloring(int argc, char** argv) {
  if (argc < 5) throw std::invalid_argument("invalid graph-coloring arguments");
  double budget_ms = std::stod(argv[2]);
  int vertices = std::stoi(argv[3]);
  int edge_count = std::stoi(argv[4]);
  if (vertices < 2 || edge_count < 0 || argc != 5 + 2 * edge_count || budget_ms <= 0)
    throw std::invalid_argument("invalid graph-coloring dimensions");
  std::vector<int> endpoints(2 * edge_count);
  for (int i = 0; i < 2 * edge_count; ++i) endpoints[i] = std::stoi(argv[5 + i]);
  SearchStats stats;
  auto* solution = optimize<GraphColoringModel>(budget_ms, stats, vertices, endpoints);
  if (solution != nullptr) {
    std::vector<int> colors(solution->colors.size());
    int maximum_color = 0;
    for (int i = 0; i < solution->colors.size(); ++i) {
      colors[i] = solution->colors[i].val();
      maximum_color = std::max(maximum_color, colors[i]);
    }
    int objective = maximum_color + 1;
    print_result(true, !stats.stopped, stats.elapsed_ms, stats.nodes, stats.failures,
                 std::to_string(objective), "{\"colors\":" + json_array(colors) + "}");
    delete solution;
  } else {
    print_result(false, false, stats.elapsed_ms, stats.nodes, stats.failures,
                 "null", "{}");
  }
}

static void run_job_shop(int argc, char** argv) {
  if (argc < 5) throw std::invalid_argument("invalid job-shop arguments");
  double budget_ms = std::stod(argv[2]);
  int jobs = std::stoi(argv[3]);
  int machines = std::stoi(argv[4]);
  int operations = jobs * machines;
  if (jobs <= 0 || machines <= 0 || argc != 5 + 2 * operations || budget_ms <= 0)
    throw std::invalid_argument("invalid job-shop dimensions");
  std::vector<int> machine_ids(operations), durations(operations);
  for (int i = 0; i < operations; ++i) {
    machine_ids[i] = std::stoi(argv[5 + 2 * i]);
    durations[i] = std::stoi(argv[6 + 2 * i]);
  }
  SearchStats stats;
  auto* solution = optimize<JobShopModel>(
      budget_ms, stats, jobs, machines, machine_ids, durations);
  if (solution != nullptr) {
    std::vector<int> starts(solution->starts.size());
    int objective = 0;
    for (int i = 0; i < solution->starts.size(); ++i) {
      starts[i] = solution->starts[i].val();
      objective = std::max(objective, starts[i] + solution->durations[i]);
    }
    print_result(true, !stats.stopped, stats.elapsed_ms, stats.nodes, stats.failures,
                 std::to_string(objective), "{\"start_times\":" + json_array(starts) + "}");
    delete solution;
  } else {
    print_result(false, false, stats.elapsed_ms, stats.nodes, stats.failures,
                 "null", "{}");
  }
}

static void run_rcpsp(int argc, char** argv) {
  if (argc < 7) throw std::invalid_argument("invalid RCPSP arguments");
  double budget_ms = std::stod(argv[2]);
  int activities = std::stoi(argv[3]);
  int resources = std::stoi(argv[4]);
  int arc_count = std::stoi(argv[5]);
  int expected = 6 + resources + activities + activities * resources + 2 * arc_count;
  if (activities < 2 || resources < 1 || arc_count < 0 || argc != expected || budget_ms <= 0)
    throw std::invalid_argument("invalid RCPSP dimensions");
  int cursor = 6;
  std::vector<int> capacities(resources), durations(activities), demands(activities * resources), precedence(2 * arc_count);
  for (int& value : capacities) value = std::stoi(argv[cursor++]);
  for (int& value : durations) value = std::stoi(argv[cursor++]);
  for (int& value : demands) value = std::stoi(argv[cursor++]);
  for (int& value : precedence) value = std::stoi(argv[cursor++]);
  SearchStats stats;
  auto* solution = optimize<RcpspModel>(
      budget_ms, stats, activities, resources, durations, demands, capacities, precedence);
  if (solution != nullptr) {
    std::vector<int> starts(solution->starts.size());
    int objective = 0;
    for (int i = 0; i < solution->starts.size(); ++i) {
      starts[i] = solution->starts[i].val();
      objective = std::max(objective, starts[i] + solution->durations[i]);
    }
    print_result(true, !stats.stopped, stats.elapsed_ms, stats.nodes, stats.failures,
                 std::to_string(objective), "{\"start_times\":" + json_array(starts) + "}");
    delete solution;
  } else {
    print_result(false, false, stats.elapsed_ms, stats.nodes, stats.failures,
                 "null", "{}");
  }
}

int main(int argc, char** argv) {
  if (argc >= 2) {
    std::string mode = argv[1];
    if (mode == "balanced_subset_sum") return (run_balanced_subset_sum(argc, argv), 0);
    if (mode == "graph_coloring") return (run_graph_coloring(argc, argv), 0);
    if (mode == "job_shop_scheduling") return (run_job_shop(argc, argv), 0);
    if (mode == "resource_constrained_project_scheduling") return (run_rcpsp(argc, argv), 0);
  }
  if (argc < 5) {
    std::cerr << "usage: gecode_binpack BUDGET_MS LEVEL CAPACITY N WEIGHTS...\n";
    return 2;
  }
  double budget_ms = std::stod(argv[1]);
  std::string level_name = argv[2];
  int capacity = std::stoi(argv[3]);
  int n = std::stoi(argv[4]);
  if (n <= 0 || argc != 5 + n || capacity <= 0 || budget_ms <= 0) return 2;
  std::vector<int> weights(n);
  long long sum = 0;
  for (int i = 0; i < n; ++i) {
    weights[i] = std::stoi(argv[5 + i]);
    if (weights[i] <= 0 || weights[i] > capacity) return 2;
    sum += weights[i];
  }
  IntPropLevel level = static_cast<IntPropLevel>(IPL_BASIC | IPL_ADVANCED);
  if (level_name == "basic") level = IPL_BASIC;
  else if (level_name == "default") level = IPL_DEF;
  else if (level_name == "advanced") level = IPL_ADVANCED;
  else if (level_name != "full") return 2;

  auto start = std::chrono::steady_clock::now();
  double used_ms = 0;
  unsigned long long nodes = 0, failures = 0;
  std::vector<std::vector<int>> solution_bins;
  int lower = std::max(1, static_cast<int>((sum + capacity - 1) / capacity));
  bool optimal = false, timed_out = false;
  for (int k = lower; k <= n; ++k) {
    used_ms = std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - start).count();
    double remaining_ms = budget_ms - used_ms;
    if (remaining_ms <= 0.1) { timed_out = true; break; }
    Search::TimeStop stop(remaining_ms);
    Search::Options options;
    options.threads = 1;
    options.stop = &stop;
    auto* model = new BinPackingModel(k, capacity, weights, level);
    DFS<BinPackingModel> engine(model, options);
    delete model;
    auto* result = engine.next();
    auto stats = engine.statistics();
    nodes += stats.node;
    failures += stats.fail;
    if (result) {
      solution_bins.assign(k, {});
      for (int i = 0; i < n; ++i) {
        solution_bins[result->bins[i].val()].push_back(i);
      }
      delete result;
      solution_bins.erase(std::remove_if(solution_bins.begin(), solution_bins.end(),
                                         [](auto const& b) { return b.empty(); }),
                          solution_bins.end());
      optimal = !engine.stopped();
      if (optimal) break;
      timed_out = true;
      break;
    }
    if (engine.stopped()) { timed_out = true; break; }
  }

  used_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
  if (!solution_bins.empty()) {
    int objective = static_cast<int>(solution_bins.size());
    std::string solver_status = optimal ? "Optimal" : "Time limit";
    std::string termination = optimal ? "optimal" : "time_limit";
    int dual = optimal ? objective : lower;
    std::cout << "{\"valid\":true,\"has_solution\":true,\"objective\":" << objective
              << ",\"primal_bound\":" << objective << ",\"dual_bound\":" << dual
              << ",\"nodes\":" << nodes << ",\"iterations\":" << failures
              << ",\"solver_status\":\"" << solver_status
              << "\",\"solver_termination\":\"" << termination
              << "\",\"solution\":{\"bins\":[";
    for (size_t i = 0; i < solution_bins.size(); ++i) {
      if (i) std::cout << ",";
      std::cout << json_array(solution_bins[i]);
    }
    std::cout << "]},\"trajectory\":[{\"event\":\"final\",\"time_sec\":"
              << used_ms / 1000.0 << ",\"primal_bound\":" << objective
              << ",\"dual_bound\":" << dual << ",\"nodes\":" << nodes << "}]}\n";
    return 0;
  }
  std::string termination = timed_out ? "time_limit" : "infeasible";
  std::cout << "{\"valid\":false,\"has_solution\":false,\"objective\":null,\"nodes\":"
            << nodes << ",\"iterations\":" << failures << ",\"solver_status\":\""
            << (timed_out ? "Time limit" : "Infeasible") << "\",\"solver_termination\":\""
            << termination << "\",\"trajectory\":[{\"event\":\"final\",\"time_sec\":"
            << used_ms / 1000.0 << ",\"nodes\":" << nodes << "}]}\n";
  return 0;
}
