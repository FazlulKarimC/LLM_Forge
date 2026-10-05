"""Analytical projections exclude large inputs/outputs unless explicitly opened."""

from sqlalchemy import func, select
from app.models.evaluation import EvaluationResult


def row_projection():
    data = EvaluationResult.data
    return (
        EvaluationResult.case_index,
        data["name"].as_string().label("name"),
        data["passed"].as_boolean().label("passed"),
        data["error"].as_string().label("error"),
        func.substr(data["output"].as_string(), 1, 200).label("output_preview"),
        data["latency_ms"].as_float().label("latency_ms"),
    )


async def case_rows(db, run_id, offset, limit, filter, reference_id=None):
    query = select(*row_projection()).where(EvaluationResult.run_id == run_id)
    data = EvaluationResult.data
    if filter == "passed":
        query = query.where(data["passed"].as_boolean().is_(True))
    elif filter == "failed":
        query = query.where(
            data["passed"].as_boolean().is_(False), data["error"].as_string().is_(None)
        )
    elif filter == "errors":
        query = query.where(data["error"].as_string().is_not(None))
    elif filter in ("improved", "regressed", "unchanged", "comparison_errors"):
        from sqlalchemy.orm import aliased

        reference = aliased(EvaluationResult)
        query = query.join(
            reference,
            (reference.case_index == EvaluationResult.case_index)
            & (reference.run_id == reference_id),
        )
        prior = reference.data
        if filter == "comparison_errors":
            query = query.where(
                data["error"].as_string().is_not(None)
                | prior["error"].as_string().is_not(None)
            )
        else:
            query = query.where(
                data["error"].as_string().is_(None),
                prior["error"].as_string().is_(None),
            )
            if filter == "unchanged":
                query = query.where(
                    data["passed"].as_boolean() == prior["passed"].as_boolean()
                )
            else:
                query = query.where(
                    data["passed"].as_boolean().is_(filter == "improved"),
                    prior["passed"].as_boolean().is_(filter == "regressed"),
                )
    total = await db.scalar(select(func.count()).select_from(query.subquery()))
    rows = (
        (
            await db.execute(
                query.order_by(EvaluationResult.case_index).offset(offset).limit(limit)
            )
        )
        .mappings()
        .all()
    )
    return {
        "items": [dict(row) for row in rows],
        "total": total,
        "offset": offset,
        "limit": limit,
    }
