"""Lifecycle-managed runner. Database owns progress; this process owns credentials."""

import asyncio
from contextlib import suppress
from datetime import timedelta
from uuid import uuid4
from sqlalchemy import func, select
from app.core.database import async_session_maker
from app.models.evaluation import EvaluationRun, DatasetRevision, EvaluationResult
from app.models.prompt_version import PromptVersion
from app.models.evaluation_execution import EvaluationExecution
from app.models.prompt import now
from app.repositories import evaluation_execution as repository

from app.core.diagnostics import logger, timed_stage


class EvaluationRunner:
    def __init__(self):
        self.owner = str(uuid4())
        self.credentials = {}
        self.attempts = {}
        self.tasks = {}
        self.wakeup = asyncio.Event()
        self.loop_task = None
        self.last_tick = None

    async def start(self):
        self.wakeup = asyncio.Event()
        self.last_tick = now()
        self.loop_task = asyncio.create_task(self.loop())

    async def stop(self):
        if self.loop_task:
            self.loop_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.loop_task
        for task in self.tasks.values():
            task.cancel()
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)
        self.tasks.clear()
        self.credentials.clear()
        self.attempts.clear()

    async def enqueue(
        self, run_id, project_id, request, evaluator_keys, *, execution_id=None
    ):
        self.credentials[run_id] = (project_id, request, evaluator_keys)
        self.attempts[run_id] = execution_id
        self.wakeup.set()

    async def loop(self):
        while True:
            self.wakeup.clear()
            try:
                await self.tick()
                self.last_tick = now()
            except Exception:
                logger.error("evaluation_runner_tick_failed")
            try:
                await asyncio.wait_for(self.wakeup.wait(), timeout=30)
            except TimeoutError:
                pass

    async def tick(self):
        async with async_session_maker() as db:
            waiting = (
                await db.scalars(
                    select(EvaluationExecution).where(
                        EvaluationExecution.id.in_(
                            [
                                identity
                                for identity in self.attempts.values()
                                if identity
                            ]
                        ),
                        EvaluationExecution.status == "queued",
                    )
                )
            ).all()
            for execution in waiting:
                execution.lease_until = now() + timedelta(seconds=90)
            await db.commit()
            await repository.reconcile(db)
            # Credentials are only reusable in their original live process.
            active_ids = set(
                (
                    await db.scalars(
                        select(EvaluationExecution.id).where(
                            EvaluationExecution.status.in_(("queued", "running"))
                        )
                    )
                ).all()
            )
            for run_id in list(self.credentials):
                if (
                    self.attempts.get(run_id) not in active_ids
                    and run_id not in self.tasks
                ):
                    self.credentials.pop(run_id, None)
                    self.attempts.pop(run_id, None)
            candidates = (
                await db.scalars(
                    select(EvaluationRun)
                    .where(EvaluationRun.status.in_(("queued", "failed")))
                    .order_by(EvaluationRun.created_at)
                )
            ).all()
            for run in candidates:
                if run.id in self.tasks:
                    continue
                if run.id not in self.credentials and not needs_credentials(run.config):
                    execution = await db.scalar(
                        select(EvaluationExecution)
                        .where(EvaluationExecution.run_id == run.id)
                        .order_by(EvaluationExecution.attempt.desc())
                        .limit(1)
                    )
                    if execution is None or execution.status not in (
                        "queued",
                        "interrupted",
                    ):
                        continue  # Legacy runs have no checkpoint recovery contract.
                    if execution.status == "interrupted":
                        identity = await self.recover_provider_free(db, run, execution)
                        if not identity:
                            continue
                    else:
                        identity = execution.id
                    self.credentials[run.id] = (
                        run.project_id,
                        request_from_config(run),
                        {},
                    )
                    self.attempts[run.id] = identity
        for run_id in list(self.credentials):
            if run_id not in self.tasks:
                task = asyncio.create_task(self.execute(run_id))
                self.tasks[run_id] = task
                task.add_done_callback(
                    lambda task, identity=run_id: self.finished(identity, task)
                )

    async def recover_provider_free(self, db, run, execution):
        # Automatic recovery shares project admission with explicit dispatch.
        from app.services.evaluation_runs import lock_project

        await lock_project(db, run.project_id)
        run = await db.scalar(
            select(EvaluationRun)
            .where(EvaluationRun.id == run.id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        execution = await db.scalar(
            select(EvaluationExecution)
            .where(EvaluationExecution.id == execution.id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        active = await db.scalar(
            select(func.count())
            .select_from(EvaluationRun)
            .where(
                EvaluationRun.project_id == run.project_id,
                EvaluationRun.status.in_(("queued", "running")),
            )
        )
        if (
            run.status != "failed"
            or execution.status != "interrupted"
            or active >= 2
            or needs_credentials(run.config)
        ):
            await db.commit()
            return False
        recovered = new_execution(run, execution.attempt + 1, execution.request_id)
        db.add(recovered)
        run.status, run.error = "queued", None
        await db.commit()
        return recovered.id

    def finished(self, run_id, task):
        failure = not task.cancelled() and task.exception()
        if failure:
            logger.error("evaluation_task_failed", extra={"run_id": str(run_id)})
        self.tasks.pop(run_id, None)
        if (
            run_id not in self.credentials
            or task.cancelled()
            or failure
            or task.result()
        ):
            self.wakeup.set()

    def health(self):
        if self.loop_task is None or self.loop_task.done() or self.last_tick is None:
            return "unhealthy"
        return (
            "healthy" if (now() - self.last_tick).total_seconds() < 90 else "unhealthy"
        )

    async def execute(self, run_id):
        accepted = self.credentials.get(run_id)
        expected_id = self.attempts.get(run_id)
        if accepted is None:
            return False
        with timed_stage("claim", run_id=str(run_id)):
            async with async_session_maker() as db:
                execution_id = await repository.claim(
                    db, run_id, self.owner, expected_execution_id=expected_id
                )
        if execution_id is None:
            return self.credentials.get(run_id) is not accepted
        project_id, request, keys = accepted
        from app.core.diagnostics import context

        async with async_session_maker() as db:
            execution = await db.get(EvaluationExecution, execution_id)
            context_token = context.set(
                {
                    "run_id": str(run_id),
                    "project_id": str(project_id),
                    "attempt": execution.attempt,
                    "request_id": execution.request_id,
                }
            )
            created = execution.created_at
            if created.tzinfo is None:
                created = created.replace(tzinfo=now().tzinfo)
            logger.info(
                "evaluation_timing",
                extra={
                    "stage": "queue_wait",
                    "duration_ms": round(
                        max(0, (now() - created).total_seconds() * 1000), 2
                    ),
                    "outcome": "claimed",
                },
            )
        beat = asyncio.create_task(self.beat(execution_id, project_id))
        try:
            async with async_session_maker() as db:
                db.info["project_id"] = project_id
                run = await db.get(EvaluationRun, run_id)
                version = await db.get(PromptVersion, run.prompt_version_id)
                revision = await db.get(DatasetRevision, run.dataset_revision_id)
                saved = None
                if run.source_run_id:
                    saved = [
                        r.data
                        for r in (
                            await db.scalars(
                                select(EvaluationResult)
                                .where(EvaluationResult.run_id == run.source_run_id)
                                .order_by(EvaluationResult.case_index)
                            )
                        ).all()
                    ]
                snapshots = run.config.get("evaluators", [])
                template = {
                    key: getattr(version, key)
                    for key in (
                        "template_text",
                        "template_format",
                        "prompt_type",
                        "messages",
                    )
                }
            from app.services.evaluation_service import execute_evaluation

            await execute_evaluation(
                run_id,
                project_id,
                request,
                template,
                revision.cases,
                snapshots,
                keys,
                saved,
                execution_id=execution_id,
                owner=self.owner,
            )
        finally:
            beat.cancel()
            try:
                await asyncio.gather(beat, return_exceptions=True)
            finally:
                # A superseding resume can supply new keys before this task exits.
                if self.credentials.get(run_id) is accepted:
                    self.credentials.pop(run_id, None)
                    self.attempts.pop(run_id, None)
                context.reset(context_token)
        return True

    async def beat(self, execution_id, project_id):
        while True:
            await asyncio.sleep(repository.HEARTBEAT_SECONDS)
            from app.core.database_transactions import transaction_call

            try:
                alive = await transaction_call(
                    async_session_maker,
                    project_id,
                    lambda db: repository.heartbeat(db, execution_id, self.owner),
                    stage="heartbeat",
                )
                if not alive:
                    return
            except Exception as exc:
                logger.error(
                    "evaluation_heartbeat_failed",
                    extra={
                        "exception_type": type(exc).__name__,
                        "project_id": str(project_id),
                    },
                )
                return


def needs_credentials(config):
    return (
        (
            config.get("source") != "score_only"
            and config.get("provider") not in ("mock", "external")
        )
        or bool(config.get("judge"))
        or any(
            item["definition"]["kind"] == "llm_judge"
            for item in config.get("evaluators", [])
        )
    )


def new_execution(run, attempt=1, request_id=None):
    return EvaluationExecution(
        project_id=run.project_id,
        run_id=run.id,
        attempt=attempt,
        request_id=request_id,
        lease_until=now() + timedelta(seconds=90),
    )


def request_from_config(run, api_key=None, judge_key=None, evaluator_keys=None):
    from app.schemas.evaluation import EvaluationCreate

    config = run.config
    judge = {**config["judge"], "api_key": judge_key} if config.get("judge") else None
    return EvaluationCreate(
        prompt_version_id=run.prompt_version_id,
        dataset_revision_id=run.dataset_revision_id,
        provider="mock"
        if config.get("source") == "score_only"
        else config.get("provider", "mock"),
        model=config.get("model", "demo-model"),
        api_key=api_key,
        temperature=config.get("temperature", 0),
        max_tokens=config.get("max_tokens", 256),
        assertions=config.get("assertions", []),
        judge=judge,
        evaluators=[
            {
                "version_id": item["version_id"],
                "required": item["required"],
                "api_key": (evaluator_keys or {}).get(item["version_id"]),
            }
            for item in config.get("evaluators", [])
        ],
    )


runner = EvaluationRunner()
