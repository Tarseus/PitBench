// Stateful MIP instrumentation. Compile against the evaluator-built source.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <dlfcn.h>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef PITBENCH_CBC
#include "CbcModel.hpp"
#include "CbcEventHandler.hpp"
#include "Cbc_C_Interface.h"
#include "OsiClpSolverInterface.hpp"
#include "CoinPackedMatrix.hpp"
#include "CoinPackedVector.hpp"
#include "CoinWarmStart.hpp"
#else
#include "scip/scip.h"
#include "scip/scipdefplugins.h"
#include "scip/cons_linear.h"
#endif

#ifndef PITBENCH_SOURCE_COMMIT
#error Source commit must be supplied by the trusted build.
#endif
#ifndef PITBENCH_ADAPTER_SHA256
#error Instrumentation source hash must be supplied by the trusted build.
#endif

using Clock = std::chrono::steady_clock;
using Observer = void (*)(double, int, const double *, double, double);
static thread_local std::string lastError;
struct Session;

#ifdef PITBENCH_CBC
class Events : public CbcEventHandler {
  Session *session_;
public:
  explicit Events(Session *session) : session_(session) {}
  CbcEventHandler *clone() const override { return new Events(*this); }
  CbcAction event(CbcEvent event) override;
};
#else
static void checked(SCIP_RETCODE code) {
  if (code != SCIP_OKAY) throw std::runtime_error("SCIP error " + std::to_string(code));
}
static SCIP_DECL_EVENTEXEC(observeEvent);
static SCIP_DECL_EVENTINIT(initializeEvents);
static SCIP_DECL_EVENTEXIT(finishEvents);
static constexpr SCIP_EVENTTYPE observedEvents =
    SCIP_EVENTTYPE_BESTSOLFOUND | SCIP_EVENTTYPE_LPSOLVED | SCIP_EVENTTYPE_NODEFOCUSED;
#endif

struct Session {
  int seed, threads, columns = 0;
  double budget, tolerance;
  bool reoptimization;
  Observer observer = nullptr;
  Clock::time_point started;
  std::thread::id solveThread;
  std::mutex mutex;
  std::vector<double> finalValues;
#ifdef PITBENCH_CBC
  std::unique_ptr<CbcModel> model;
#else
  SCIP *scip = nullptr;
  std::vector<SCIP_VAR *> variables;
  SCIP_EVENTHDLR *handler = nullptr;
#endif
  Session(int seed, int threads, double budget, double tolerance, bool reoptimization)
      : seed(seed), threads(threads), budget(budget), tolerance(tolerance),
        reoptimization(reoptimization) {
#ifdef PITBENCH_CBC
    if (threads > 1 && !CbcModel::haveMultiThreadSupport())
      throw std::runtime_error("CBC source was built without native thread support");
#else
    checked(SCIPcreate(&scip));
    checked(SCIPincludeDefaultPlugins(scip));
    checked(SCIPsetIntParam(scip, "display/verblevel", 0));
    checked(SCIPsetIntParam(scip, "randomization/randomseedshift", seed));
    checked(SCIPsetRealParam(scip, "numerics/feastol", tolerance));
    checked(SCIPsetIntParam(scip, "parallel/minnthreads", threads));
    checked(SCIPsetIntParam(scip, "parallel/maxnthreads", threads));
    if (reoptimization) checked(SCIPenableReoptimization(scip, TRUE));
    checked(SCIPincludeEventhdlrBasic(scip, &handler, "pitbench-observer",
        "export native incumbents and diagnostic bounds", observeEvent,
        reinterpret_cast<SCIP_EVENTHDLRDATA *>(this)));
    checked(SCIPsetEventhdlrInit(scip, handler, initializeEvents));
    checked(SCIPsetEventhdlrExit(scip, handler, finishEvents));
#endif
  }
  ~Session() {
#ifndef PITBENCH_CBC
    if (scip) {
      for (auto &variable : variables) SCIPreleaseVar(scip, &variable);
      SCIPfree(&scip);
    }
#endif
  }
  void emit(const double *values, double primal, double dual) {
    std::lock_guard<std::mutex> lock(mutex);
    if (observer)
      observer(std::chrono::duration<double>(Clock::now() - started).count(),
               columns, values, primal, dual);
  }
};

#ifdef PITBENCH_CBC
CbcEventHandler::CbcAction Events::event(CbcEvent event) {
  // Heuristic and parallel child models are not global incumbents. In particular,
  // postProcessedSolver() follows parent preprocessing state and resolves an LP;
  // invoking it from a child worker races with the parent's solver state.
  // Parallel search clones can have a null parentModel(); they still must not
  // postprocess the shared original model from a background solver thread.
  if (std::this_thread::get_id() != session_->solveThread ||
      model_->parentModel() != nullptr) return noAction;
  const bool incumbent = event == solution || event == heuristicSolution || event == endSearch;
  const double *values = nullptr;
  if (incumbent && model_->bestSolution()) {
    const OsiSolverInterface *original = model_->postProcessedSolver(1);
    if (original && original->getNumCols() == session_->columns)
      values = original->getColSolution();
    else if (model_->getNumCols() == session_->columns)
      values = model_->bestSolution();
  }
  session_->emit(values,
                 model_->getObjValue(), model_->getBestPossibleObjValue());
  return noAction;
}
#else
static SCIP_DECL_EVENTINIT(initializeEvents) {
  return SCIPcatchEvent(scip, observedEvents, eventhdlr, nullptr, nullptr);
}
static SCIP_DECL_EVENTEXIT(finishEvents) {
  return SCIPdropEvent(scip, observedEvents, eventhdlr, nullptr, -1);
}
static SCIP_DECL_EVENTEXEC(observeEvent) {
  auto *session = reinterpret_cast<Session *>(SCIPeventhdlrGetData(eventhdlr));
  SCIP_SOL *solution = SCIPgetBestSol(scip);
  std::vector<double> values;
  if ((SCIPeventGetType(event) & SCIP_EVENTTYPE_BESTSOLFOUND) && solution) {
    values.resize(session->columns);
    SCIP_RETCODE result = SCIPgetSolVals(scip, solution, session->columns,
                                        session->variables.data(), values.data());
    if (result != SCIP_OKAY) return result;
  }
  session->emit(values.empty() ? nullptr : values.data(),
                SCIPgetPrimalbound(scip), SCIPgetDualbound(scip));
  return SCIP_OKAY;
}
#endif

extern "C" {
const char *pbs_error() { return lastError.c_str(); }
const char *pbs_git_hash() { return PITBENCH_SOURCE_COMMIT; }
const char *pbs_adapter_sha256() { return PITBENCH_ADAPTER_SHA256; }
const char *pbs_version() {
#ifdef PITBENCH_CBC
  return Cbc_getVersion();
#else
  static std::string version = std::to_string(SCIPmajorVersion()) + "." +
      std::to_string(SCIPminorVersion()) + "." + std::to_string(SCIPtechVersion());
  return version.c_str();
#endif
}
const char *pbs_solver_library() {
  static Dl_info info;
#ifdef PITBENCH_CBC
  dladdr(reinterpret_cast<void *>(&Cbc_getVersion), &info);
#else
  dladdr(reinterpret_cast<void *>(&SCIPcreate), &info);
#endif
  return info.dli_fname;
}
void *pbs_create(int seed, int threads, double budget, double tolerance, int reoptimization) {
  try { return new Session(seed, threads, budget, tolerance, reoptimization != 0); }
  catch (const std::exception &error) { lastError = error.what(); return nullptr; }
}
void pbs_destroy(void *opaque) { delete static_cast<Session *>(opaque); }

int pbs_load(void *opaque, int rows, int columns, const int *starts,
             const int *indices, const double *coefficients, const double *cost,
             const double *lower, const double *upper, const int *integer,
             const double *lhs, const double *rhs, int maximize, double offset) {
  auto &session = *static_cast<Session *>(opaque);
  session.columns = columns;
  try {
#ifdef PITBENCH_CBC
    OsiClpSolverInterface solver;
    CoinPackedMatrix matrix(false, 0, 0);
    matrix.setDimensions(0, columns);
    for (int row = 0; row < rows; ++row) {
      CoinPackedVector vector;
      for (int index = starts[row]; index < starts[row + 1]; ++index)
        vector.insert(indices[index], coefficients[index]);
      matrix.appendRow(vector);
    }
    solver.loadProblem(matrix, lower, upper, cost, lhs, rhs);
    solver.setObjSense(maximize ? -1 : 1);
    solver.setDblParam(OsiObjOffset, -offset);
    for (int column = 0; column < columns; ++column)
      if (integer[column]) solver.setInteger(column);
    session.model.reset(new CbcModel(solver));
    CbcMain0(*session.model);
    session.model->saveReferenceSolver();
#else
    checked(SCIPcreateProbBasic(session.scip, "pitbench"));
    checked(SCIPsetObjsense(session.scip, maximize ? SCIP_OBJSENSE_MAXIMIZE : SCIP_OBJSENSE_MINIMIZE));
    checked(SCIPaddOrigObjoffset(session.scip, offset));
    session.variables.resize(columns, nullptr);
    for (int column = 0; column < columns; ++column) {
      std::string name = "x" + std::to_string(column);
      checked(SCIPcreateVarBasic(session.scip, &session.variables[column], name.c_str(),
          lower[column], upper[column], cost[column],
          integer[column] ? (lower[column] == 0 && upper[column] == 1 ?
            SCIP_VARTYPE_BINARY : SCIP_VARTYPE_INTEGER) : SCIP_VARTYPE_CONTINUOUS));
      checked(SCIPaddVar(session.scip, session.variables[column]));
    }
    for (int row = 0; row < rows; ++row) {
      SCIP_CONS *constraint = nullptr;
      std::vector<SCIP_VAR *> vars;
      std::vector<double> values;
      for (int index = starts[row]; index < starts[row + 1]; ++index) {
        vars.push_back(session.variables[indices[index]]);
        values.push_back(coefficients[index]);
      }
      std::string name = "c" + std::to_string(row);
      checked(SCIPcreateConsBasicLinear(session.scip, &constraint, name.c_str(),
          vars.size(), vars.data(), values.data(), lhs[row], rhs[row]));
      checked(SCIPaddCons(session.scip, constraint));
      checked(SCIPreleaseCons(session.scip, &constraint));
    }
#endif
    return 0;
  } catch (const std::exception &error) { lastError = error.what(); return -1; }
}

int pbs_update_objective(void *opaque, const double *cost, int maximize) {
#ifdef PITBENCH_CBC
  auto &session = *static_cast<Session *>(opaque);
  try {
    std::unique_ptr<CoinWarmStart> basis(session.model->solver()->getWarmStart());
    if (!basis) { lastError = "CBC did not retain an LP basis"; return -2; }
    // Retain algorithmic LP state while resetting objective-dependent MIP state.
    // This is a basis warm start, not a claim that CBC reuses its MIP tree.
    session.model->resetToReferenceSolver();
    if (!session.model->solver()->setWarmStart(basis.get())) {
      lastError = "CBC could not map the retained LP basis to the original model";
      return -2;
    }
    session.model->solver()->setObjSense(maximize ? -1 : 1);
    for (int column = 0; column < session.columns; ++column)
      session.model->solver()->setObjCoeff(column, cost[column]);
    session.model->saveReferenceSolver();
    return 0;
  } catch (const std::exception &error) { lastError = error.what(); return -1; }
#else
  auto &session = *static_cast<Session *>(opaque);
  if (!session.reoptimization || session.threads != 1) {
    lastError = "SCIP objective updates require explicitly enabled serial reoptimization";
    return -2;
  }
  try {
    checked(SCIPfreeReoptSolve(session.scip));
    checked(SCIPchgReoptObjective(session.scip,
        maximize ? SCIP_OBJSENSE_MAXIMIZE : SCIP_OBJSENSE_MINIMIZE,
        session.variables.data(), const_cast<double *>(cost), session.columns));
    return 0;
  } catch (const std::exception &error) { lastError = error.what(); return -1; }
#endif
}

int pbs_solve(void *opaque, Observer observer) {
  auto &session = *static_cast<Session *>(opaque);
  session.observer = observer;
  session.started = Clock::now();
  session.solveThread = std::this_thread::get_id();
  session.finalValues.clear();
  try {
#ifdef PITBENCH_CBC
    Events events(&session);
    session.model->passInEventHandler(&events);
    session.model->solver()->setDblParam(OsiPrimalTolerance, session.tolerance);
    std::vector<std::string> strings = {"cbc", "-seconds", std::to_string(session.budget),
      "-timeMode", "elapsed", "-randomSeed", std::to_string(session.seed),
      "-randomCbcSeed", std::to_string(session.seed), "-threads", std::to_string(session.threads),
      "-logLevel", "0", "-solve", "-quit"};
    std::vector<const char *> argv;
    for (auto &value : strings) argv.push_back(value.c_str());
    CbcMain1(argv.size(), argv.data(), *session.model);
    if (auto *values = session.model->bestSolution())
      session.finalValues.assign(values, values + session.columns);
    else if (!session.model->numberIntegers() && session.model->solver()->isProvenOptimal()) {
      const double *values = session.model->solver()->getColSolution();
      session.finalValues.assign(values, values + session.columns);
    }
    int status = (session.model->isProvenOptimal() ||
                 (!session.model->numberIntegers() && session.model->solver()->isProvenOptimal())) ? 0 :
                 session.model->isProvenInfeasible() ? 2 :
                 session.model->isSecondsLimitReached() ? 1 : 3;
#else
    checked(SCIPsetRealParam(session.scip, "limits/time",
                            SCIPgetSolvingTime(session.scip) + session.budget));
    checked(session.threads > 1 ? SCIPsolveConcurrent(session.scip) : SCIPsolve(session.scip));
    if (SCIP_SOL *solution = SCIPgetBestSol(session.scip)) {
      session.finalValues.resize(session.columns);
      checked(SCIPgetSolVals(session.scip, solution, session.columns,
          session.variables.data(), session.finalValues.data()));
    }
    SCIP_STATUS native = SCIPgetStatus(session.scip);
    int status = native == SCIP_STATUS_OPTIMAL ? 0 :
                 native == SCIP_STATUS_TIMELIMIT ? 1 :
                 native == SCIP_STATUS_INFEASIBLE ? 2 : 3;
#endif
    session.observer = nullptr;
    return status;
  } catch (const std::exception &error) {
    session.observer = nullptr;
    lastError = error.what(); return -1;
  }
}
int pbs_solution(void *opaque, double *output) {
  auto &values = static_cast<Session *>(opaque)->finalValues;
  std::copy(values.begin(), values.end(), output);
  return values.size();
}
int pbs_reset(void *opaque) {
  auto &session = *static_cast<Session *>(opaque);
  try {
#ifdef PITBENCH_CBC
    session.model->resetToReferenceSolver();
#else
    checked(SCIPfreeTransform(session.scip));
#endif
    return 0;
  } catch (const std::exception &error) { lastError = error.what(); return -1; }
}
}
