"""Local storage budgets; no scheduler or additional storage service."""
import hashlib
import json
from app.services.benchmark_execution import BenchmarkBudgetExceeded

MAX_INPUT_BYTES = 8 * 1024
MAX_OUTPUT_BYTES = 8 * 1024
MAX_TRACE_BYTES = 16 * 1024
MAX_CASE_BYTES = 64 * 1024
MAX_ATTEMPT_BYTES = 2 * 1024 * 1024
MAX_PROJECT_BYTES = 100 * 1024 * 1024
MAX_PROJECT_ATTEMPTS = 100


def bounded_record(record):
    record = dict(record)
    metadata = dict(record.get("run_metadata") or {})
    for field, limit in (("prompt", MAX_INPUT_BYTES), ("raw_output", MAX_OUTPUT_BYTES),
                         ("parsed_answer", MAX_OUTPUT_BYTES), ("expected_output", MAX_INPUT_BYTES)):
        value = record.get(field)
        if value is None:
            continue
        encoded = value.encode()
        if len(encoded) > limit:
            record[field] = encoded[:limit].decode("utf-8", errors="ignore")
            metadata.setdefault("retention_truncated", {})[field] = {"original_bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}
    for field in ("agent_trace", "retrieved_chunks"):
        encoded = json.dumps(record.get(field), ensure_ascii=False, default=str).encode()
        if len(encoded) > MAX_TRACE_BYTES:
            record[field] = {"retention_truncated": True, "original_bytes": len(encoded),
                             "sha256": hashlib.sha256(encoded).hexdigest(), "preview": encoded[:8192].decode(errors="ignore")}
    record["run_metadata"] = metadata
    size = len(json.dumps(record, ensure_ascii=False, default=str).encode())
    if size > MAX_CASE_BYTES:
        raise BenchmarkBudgetExceeded("Case storage budget exceeded; completed cases were kept")
    return record, size
