__all__ = ["FixtureJudge", "JudgePlan", "PitBenchEvaluator"]


def __getattr__(name: str):
    if name == "PitBenchEvaluator":
        from pitbench.evaluator.evaluator import PitBenchEvaluator

        return PitBenchEvaluator
    if name in {"FixtureJudge", "JudgePlan"}:
        from pitbench.evaluator.judge import FixtureJudge, JudgePlan

        return {"FixtureJudge": FixtureJudge, "JudgePlan": JudgePlan}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
