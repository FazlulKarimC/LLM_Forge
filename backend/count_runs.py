import asyncio
from app.core.database import async_session_maker
from app.models.run import Run
from sqlalchemy import select, func

async def check():
    async with async_session_maker() as db:
        result = await db.execute(select(func.count()).select_from(Run))
        count = result.scalar()
        print(f"Total runs in database: {count}")

asyncio.run(check())
