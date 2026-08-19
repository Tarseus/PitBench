__all__ = ["Harness", "BenchmarkResults"]


def __getattr__(name: str):
    if name == "Harness":
        from fceval.harness.harness import Harness

        return Harness
    if name == "BenchmarkResults":
        from fceval.harness.models import BenchmarkResults

        return BenchmarkResults
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
