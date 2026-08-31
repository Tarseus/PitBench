from __future__ import annotations

from pydantic import BaseModel, Field


class PopulationFeatures(BaseModel):
    population: str
    instance_ids: list[str]
    representation: str
    vectors: list[list[float]] = Field(default_factory=list)
