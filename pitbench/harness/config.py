from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()


class Config:
    """Environment configuration for the execution harness."""

    @staticmethod
    def get_setting(key: str, default: Any = None) -> Any:
        return os.environ.get(key.upper(), default)

    @property
    def aws_region(self):
        return self.get_setting("aws_region", "us-west-2")

    @property
    def s3_bucket_name(self):
        return self.get_setting("s3_bucket_name")

    @property
    def s3_evaluation_snapshots_bucket_name(self):
        return self.get_setting("s3_evaluation_snapshots_bucket_name")


config = Config()
