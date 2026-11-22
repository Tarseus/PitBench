// Long-lived process adapter for source-built solver APIs.
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "structures/cl_args.h"
#include "structures/vroom/input/input.h"
#include "utils/helpers.h"
#include "utils/input_parser.h"
#include "utils/output_json.h"

int main() {
  std::string line;
  while (std::getline(std::cin, line)) {
    try {
      std::istringstream request(line);
      std::string input_path;
      double budget;
      unsigned threads, exploration;
      if (!(request >> input_path >> budget >> threads >> exploration) ||
          budget <= 0 || threads == 0 || exploration > vroom::MAX_EXPLORATION_LEVEL)
        throw std::invalid_argument("invalid session request");
      std::ifstream file(input_path);
      if (!file) throw std::runtime_error("cannot read scenario input");
      std::ostringstream contents;
      contents << file.rdbuf();
      vroom::io::CLArgs args;
      args.set_exploration_level(exploration);
      vroom::Input input(args.servers, vroom::ROUTER::OSRM, false);
      vroom::io::parse(input, contents.str(), false);
      auto timeout = std::chrono::milliseconds(
          static_cast<std::chrono::milliseconds::rep>(budget * 1000));
      auto solution = input.solve(args.nb_searches, args.depth, threads, timeout);
      vroom::io::write_to_json(solution, "", input.report_distances());
    } catch (const std::exception& error) {
      std::cerr << error.what() << '\n';
      std::cout << "{\"session_error\":true}\n";
    }
    std::cout.flush();
  }
}
