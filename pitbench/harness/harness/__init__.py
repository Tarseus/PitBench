__all__ = ["Harness", "BenchmarkResults"]


def __getattr__(name: str):
    if name == "Harness":
        from pitbench.harness.harness.harness import Harness

        return Harness
    if name == "BenchmarkResults":
        from pitbench.harness.harness.models import BenchmarkResults

        return BenchmarkResults
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
