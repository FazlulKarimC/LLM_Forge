"""Run real locking tests in disposable schemas on the configured PostgreSQL.

Only generated llmforge_test_* schemas are created/dropped. Existing tables,
data and migration state are never changed. Credentials are never printed.
"""

import os
import subprocess
import sys
from app.core.config import settings


def main():
    url = os.environ.get("LLMFORGE_TEST_POSTGRES_URL") or settings.DATABASE_URL
    if not url.startswith("postgres"):
        raise SystemExit("A PostgreSQL connection is required")
    env = {**os.environ, "LLMFORGE_TEST_POSTGRES_URL": url}
    raise SystemExit(
        subprocess.call(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_evaluation_postgres.py",
                "-q",
                "--tb=short",
            ],
            env=env,
        )
    )


if __name__ == "__main__":
    main()
