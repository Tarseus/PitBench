from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()


class Config:
    """Environment-first configuration with optional TOML secrets.

    The execution harness must not import a dashboard framework merely to read
    configuration.
    """

    @staticmethod
    def get_setting(key: str, default: Any = None) -> Any:
        secrets_path = Path(".streamlit/secrets.toml")
        if secrets_path.is_file():
            with secrets_path.open("rb") as handle:
                secrets = tomllib.load(handle)
            if key.lower() in secrets:
                return secrets[key.lower()]
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

    @property
    def s3_recordings_bucket_name(self):
        return self.get_setting("s3_recordings_bucket_name")

    @property
    def db_host(self):
        return self.get_setting("db_host")

    @property
    def db_name(self):
        return self.get_setting("db_name")

    @property
    def db_user(self):
        return self.get_setting("db_user")

    @property
    def db_password(self):
        return self.get_setting("db_password")


config = Config()
