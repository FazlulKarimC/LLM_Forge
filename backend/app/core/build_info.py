"""Explicit non-secret build identity; never inspect arbitrary environment values."""

import json
import re
from pathlib import Path

from app.core.config import settings

BUILD_INFO_PATH = Path(__file__).resolve().parents[1] / "_build_info.json"


def build_identity():
    embedded = {}
    try:
        if BUILD_INFO_PATH.stat().st_size <= 1024:
            value = json.loads(BUILD_INFO_PATH.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                embedded = value
    except (OSError, ValueError):
        pass
    commit = settings.BUILD_COMMIT or embedded.get("commit")
    build_id = settings.BUILD_ID or embedded.get("build_id")
    return {
        "commit": commit if isinstance(commit, str) and re.fullmatch(r"[a-fA-F0-9]{40}", commit) else None,
        "build_id": build_id if isinstance(build_id, str) and re.fullmatch(r"[A-Za-z0-9._-]{1,80}", build_id) else None,
    }
