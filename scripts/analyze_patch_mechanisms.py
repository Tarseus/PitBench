"""Validate manual patch coding against immutable attempts and summarize it."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = {"implementation", "mixed", "configuration", "none"}


def analyze(attempt_index: Path, coding_path: Path) -> dict:
    attempts = json.loads(attempt_index.read_text())["records"]
    coding = json.loads(coding_path.read_text())
    records = coding["records"]
    if coding.get("schema_version") != 1 or len(records) != len(attempts):
        raise ValueError("coding sheet does not cover the indexed attempts")
    by_identity = {
        (record["model_run"], record["task_id"]): record for record in records
    }
    if len(by_identity) != len(records):
        raise ValueError("duplicate attempt in coding sheet")
    class_counts: Counter[str] = Counter()
    outcome_by_class: dict[str, Counter[str]] = defaultdict(Counter)
    evidence = []
    for attempt in attempts:
        identity = (attempt["model_run"], attempt["task_id"])
        if identity not in by_identity:
            raise ValueError(f"attempt has no manual coding: {identity}")
        record = by_identity[identity]
        path = ROOT / attempt["patch_path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if (
            digest != attempt["patch_sha256"]
            or digest != record["patch_sha256"]
            or attempt["patch_path"] != record["patch_path"]
        ):
            raise ValueError(f"patch identity mismatch: {identity}")
        category = record["source_change_class"]
        if category not in CLASSES:
            raise ValueError(f"unknown source change class: {category}")
        if record["observed_outcome"] != attempt["current_rule_classification"]:
            raise ValueError(f"outcome differs from attempt index: {identity}")
        class_counts[category] += 1
        outcome_by_class[category][record["observed_outcome"]] += 1
        evidence.append(
            {
                "task_id": attempt["task_id"],
                "model_run": attempt["model_run"],
                "patch_sha256": digest,
                "source_change_class": category,
                "coding_confidence": record["coding_confidence"],
                "observed_outcome": record["observed_outcome"],
                "attribution_limit": record["attribution_limit"],
            }
        )
    return {
        "attempts": len(attempts),
        "class_counts": dict(sorted(class_counts.items())),
        "outcome_by_class": {
            category: dict(sorted(counts.items()))
            for category, counts in sorted(outcome_by_class.items())
        },
        "records": evidence,
        "claim_boundary": (
            "Manual patch type is descriptive; outcomes do not prove that an "
            "edit mechanism caused an improvement or regression."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-index", type=Path, required=True)
    parser.add_argument("--coding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.attempt_index, args.coding)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("attempts", "class_counts")}))


if __name__ == "__main__":
    main()
