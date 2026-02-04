import asyncio
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

DSN = "postgresql://user:pass@host:5432/writing_checkpoint_db"

async def main():
    pool = AsyncConnectionPool(
        conninfo=DSN,
        kwargs={"autocommit": True, "row_factory": dict_row},
        open=False,
    )
    await pool.open()
    saver = AsyncPostgresSaver(pool)
    await saver.setup()
    await pool.close()
    print("checkpoint tables ready")

if __name__ == "__main__":
    asyncio.run(main())
