from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pitbench.schema.observation import RunObservation


class ObservationStore:
    """Canonical columnar storage for raw run observations."""

    @staticmethod
    def write(path: Path, observations: list[RunObservation]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [item.model_dump(mode="json") for item in observations]
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, path, compression="zstd")

    @staticmethod
    def read(path: Path) -> list[RunObservation]:
        table = pq.read_table(path)
        return [RunObservation.model_validate(row) for row in table.to_pylist()]

    @staticmethod
    def write_jsonl(path: Path, observations: list[RunObservation]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            for observation in observations:
                handle.write(observation.model_dump_json())
                handle.write("\n")

    @staticmethod
    def read_jsonl(path: Path) -> list[RunObservation]:
        return [
            RunObservation.model_validate_json(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]

    @staticmethod
    def schema_json(path: Path) -> None:
        path.write_text(json.dumps(RunObservation.model_json_schema(), indent=2))
