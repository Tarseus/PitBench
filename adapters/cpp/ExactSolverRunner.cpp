#include <gecode/int.hh>
#include <gecode/search.hh>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace Gecode;

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

static std::string json_array(std::vector<int> const& xs) {
  std::string out = "[";
  for (size_t i = 0; i < xs.size(); ++i) {
    if (i) out += ",";
    out += std::to_string(xs[i]);
  }
  return out + "]";
}

int main(int argc, char** argv) {
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
  if (!solution_bins.empty() && optimal) {
    int objective = static_cast<int>(solution_bins.size());
    std::cout << "{\"valid\":true,\"has_solution\":true,\"objective\":" << objective
              << ",\"primal_bound\":" << objective << ",\"dual_bound\":" << objective
              << ",\"nodes\":" << nodes << ",\"iterations\":" << failures
              << ",\"solver_status\":\"Optimal\",\"solver_termination\":\"optimal\",\"solution\":{\"bins\":[";
    for (size_t i = 0; i < solution_bins.size(); ++i) {
      if (i) std::cout << ",";
      std::cout << json_array(solution_bins[i]);
    }
    std::cout << "]},\"trajectory\":[{\"event\":\"final\",\"time_sec\":"
              << used_ms / 1000.0 << ",\"primal_bound\":" << objective
              << ",\"dual_bound\":" << objective << ",\"nodes\":" << nodes << "}]}\n";
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
