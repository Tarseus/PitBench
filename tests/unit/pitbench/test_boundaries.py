from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_active_harness_has_no_formulacode_evaluation_ontology() -> None:
    harness = (ROOT / "pitbench/harness/harness/harness.py").read_text().lower()
    models = (ROOT / "pitbench/harness/harness/models.py").read_text().lower()
    forbidden = ("formulacode", "asv", "speedup_percentage", "agent_advantage")
    for token in forbidden:
        assert token not in harness
        assert token not in models


def test_legacy_implementation_is_confined_to_upstream() -> None:
    assert (ROOT / "upstream/fceval/parsers/formulacode_parser.py").is_file()
    assert not (ROOT / "pitbench/harness/parsers/formulacode_parser.py").exists()
    assert not (ROOT / "pitbench/harness/agents/oracle_agent.py").exists()
