"""Export contracts without starting lifespan or opening database connections."""

import json
import os
from pathlib import Path


def main():
    # Schema export must not initialize an external error-reporting client.
    os.environ["SENTRY_DSN"] = ""
    from app.main import create_application

    destination = Path(__file__).resolve().parents[2] / "contracts" / "openapi.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(
        json.dumps(create_application().openapi(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
