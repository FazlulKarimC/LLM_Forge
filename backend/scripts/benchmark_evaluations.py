"""Provider-free SDK/HTTP/runner benchmark in a disposable LOCAL PostgreSQL schema.

Requires LLMFORGE_TEST_POSTGRES_URL pointing to localhost. Never falls back to
DATABASE_URL. Uses four fixed cases, inline assertions and a saved builtin.
No provider calls, paid judges, hosted writes, or existing project data.
"""

import argparse
import asyncio
from collections import Counter
import json
import logging
import math
import os
from pathlib import Path
import socket
import time
from uuid import uuid4

import httpx
from sqlalchemy import event, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
import uvicorn


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower), 2)


class Timings(logging.Handler):
    def __init__(self):
        super().__init__()
        self.rows = []

    def emit(self, record):
        if record.msg == "evaluation_timing":
            # Only aggregate stage names and durations, never context/payloads.
            self.rows.append((record.stage, record.duration_ms))


async def benchmark(args):
    target = os.environ.get("LLMFORGE_TEST_POSTGRES_URL", "")
    if not target:
        raise SystemExit("Set LLMFORGE_TEST_POSTGRES_URL to a disposable local PostgreSQL database.")
    url = make_url(target)
    if url.get_backend_name() != "postgresql" or url.host not in ("localhost", "127.0.0.1", "::1"):
        raise SystemExit("This benchmark accepts only local PostgreSQL; hosted targets are refused.")
    url = url.set(drivername="postgresql+asyncpg", query={})
    # Isolate global app settings before any app module can load a local .env.
    os.environ.update(DATABASE_URL=url.render_as_string(hide_password=False), INFERENCE_ENGINE="mock", SENTRY_DSN="", REDIS_URL="", QDRANT_API_KEY="", QUEUE_BACKEND_MODE="inline")
    schema = "llmforge_test_" + uuid4().hex
    admin = create_async_engine(url)
    engine = create_async_engine(url, execution_options={"schema_translate_map": {None: schema}})
    counts = Counter()
    event.listen(engine.sync_engine, "before_cursor_execute", lambda *_: counts.update(statements=1))
    event.listen(engine.sync_engine, "commit", lambda *_: counts.update(commits=1))
    server = task = None
    listener = socket.socket()
    timings = Timings()
    try:
        async with admin.begin() as db:
            await db.execute(text(f'CREATE SCHEMA "{schema}"'))
        from app.core import database
        database.engine = engine
        database.async_session_maker = async_sessionmaker(engine, expire_on_commit=False)
        from app.main import create_application
        from app.core.auth import Identity, get_current_user
        from app.core.diagnostics import logger
        from app.models.evaluation_execution import EvaluationCapacity
        async with engine.begin() as db:
            await db.run_sync(database.Base.metadata.create_all)
        async with database.async_session_maker() as db:
            db.add(EvaluationCapacity(id=1))
            await db.commit()
        app = create_application()
        app.dependency_overrides[get_current_user] = lambda: Identity("user_benchmark_" + schema, "Disposable benchmark")
        listener.bind(("127.0.0.1", 0))
        origin = f"http://127.0.0.1:{listener.getsockname()[1]}"
        server = uvicorn.Server(uvicorn.Config(app, log_level="warning", access_log=False))
        task = asyncio.create_task(server.serve(sockets=[listener]))
        while not server.started:
            if task.done():
                await task
                raise RuntimeError("Local benchmark server did not start")
            await asyncio.sleep(0.05)
        logger.addHandler(timings)
        from llmforge import LLMForge
        async with httpx.AsyncClient(base_url=origin, timeout=30) as client:
            workspace = (await client.get("/api/v1/workspaces")).json()
            project_id = workspace["organizations"][0]["projects"][0]["id"]
            headers = {"X-Project-ID": project_id}

            async def post(path, body):
                response = await client.post("/api/v1/" + path, headers=headers, json=body)
                if response.status_code != 201:
                    raise RuntimeError("Disposable benchmark setup failed")
                return response.json()

            prompt = await post("prompt-library", {"name": "Audit echo", "template_text": "{{query}}"})
            dataset = await post("datasets", {"name": "Audit four cases", "cases": [
                {"inputs": {"query": query}, "expected_output": query, "name": f"Case {i}"}
                for i, query in enumerate(("Hello", "Goodbye", "Unicode café", "Literal {braces}"))]})
            evaluator = await post("evaluators", {"name": "Audit exact match", "definition": {
                "kind": "builtin", "assertion": {"kind": "exact_match"}, "outputs": [{"name": "correct"}]}})
            key = await post("project-keys", {"name": "Temporary benchmark", "scopes": ["evaluations:write"]})
            selected = [{"version_id": evaluator["latest"]["id"], "required": True}]

            def run_fixture(saved):
                with LLMForge(base_url=origin + "/api/v1", api_key=key["secret"], timeout=30) as forge:
                    started = time.perf_counter()
                    identity = forge.start_evaluation(prompt["version"]["id"], dataset["revision"]["id"],
                        assertions=[] if saved else [{"kind": "exact_match"}], evaluators=selected if saved else None)
                    result = forge.wait_for_evaluation(identity, timeout=120, poll_interval=0.05)
                    elapsed = (time.perf_counter() - started) * 1000
                    if not result.meets_threshold(1.0):
                        raise RuntimeError("Provider-free fixture did not pass")
                    from datetime import datetime
                    lifecycle = (datetime.fromisoformat(result.run["updated_at"]) - datetime.fromisoformat(result.run["created_at"])).total_seconds() * 1000
                    return elapsed, lifecycle

            report = {"target": "disposable-local-postgresql", "cases": 4, "provider": "mock", "runs_per_fixture": args.runs,
                      "identity": (await client.get("/api/v1/version")).json(), "fixtures": {}}
            for saved in (False, True):
                samples = []
                for _ in range(args.runs + 1):
                    before, first = counts.copy(), len(timings.rows)
                    elapsed, lifecycle = await asyncio.to_thread(run_fixture, saved)
                    stages = Counter()
                    for stage, duration in timings.rows[first:]:
                        stages[stage] += duration
                    samples.append({"sdk_ms": round(elapsed, 2), "lifecycle_ms": round(lifecycle, 2),
                        "statements": counts["statements"] - before["statements"], "commits": counts["commits"] - before["commits"],
                        "stage_ms": {k: round(v, 2) for k, v in stages.items()}})
                warm = samples[1:]
                report["fixtures"]["saved_builtin" if saved else "inline"] = {"first_run": samples[0], "warm_runs": warm,
                    "sdk_p50_ms": percentile([r["sdk_ms"] for r in warm], .5), "sdk_p95_ms": percentile([r["sdk_ms"] for r in warm], .95)}
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Provider-free benchmark completed: {args.runs} warm runs per fixture; report saved.")
        if args.max_p95_ms is not None and any(fixture["sdk_p95_ms"] > args.max_p95_ms for fixture in report["fixtures"].values()):
            raise SystemExit("Measured warm p95 exceeded the requested local regression threshold; report retained.")
    finally:
        if server:
            server.should_exit = True
        if task:
            await task
        listener.close()
        logging.getLogger("app.platform").removeHandler(timings)
        await engine.dispose()
        async with admin.begin() as db:
            assert schema.startswith("llmforge_test_") and len(schema) == 46
            await db.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await admin.dispose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--output", default="artifacts/evaluation-benchmark.json")
    parser.add_argument("--max-p95-ms", type=float, help="Optional threshold established for this local test environment")
    args = parser.parse_args()
    if not 2 <= args.runs <= 50:
        parser.error("Use 2–50 warm runs per fixture")
    if args.max_p95_ms is not None and (not math.isfinite(args.max_p95_ms) or args.max_p95_ms <= 0):
        parser.error("Use a finite positive p95 threshold")
    asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()
