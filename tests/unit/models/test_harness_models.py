"""Tests for execution-generic harness models."""

from pitbench.harness.harness.models import BenchmarkResults, TrialResults


def test_harness_results_do_not_publish_binary_verdict_aggregates() -> None:
    trial = TrialResults(
        trial_name="trial",
        task_id="task",
        instruction="test instruction",
    )
    results = BenchmarkResults(results=[trial])

    assert "is_resolved" not in trial.model_dump()
    assert "accuracy" not in results.model_dump()
    assert "pass_at_k" not in results.model_dump()
    assert not hasattr(results, "n_resolved")
    assert not hasattr(results, "n_unresolved")
