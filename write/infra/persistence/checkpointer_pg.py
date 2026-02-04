from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

@dataclass(frozen=True)
class PgCheckpointSettings:
    dsn: str
    max_pool_size: int = 10

_pool: Optional[AsyncConnectionPool] = None
_saver: Optional[AsyncPostgresSaver] = None

async def init_checkpointer(cfg: PgCheckpointSettings) -> AsyncPostgresSaver:
    global _pool, _saver
    if _saver is not None:
        return _saver

    _pool = AsyncConnectionPool(
        conninfo=cfg.dsn,
        max_size=cfg.max_pool_size,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
        open=False,
    )
    await _pool.open()

    _saver = AsyncPostgresSaver(_pool)
    await _saver.setup()
    return _saver

async def close_checkpointer() -> None:
    global _pool, _saver
    _saver = None
    if _pool is not None:
        await _pool.close()
        _pool = None
