import asyncio
import os

from config import load_config
from infra.persistence.checkpointer_pg import close_checkpointer, ensure_checkpoint_tables

def _resolve_dsn() -> str | None:
    cfg = load_config()
    dsn = os.getenv("CHECKPOINT_DSN") or os.getenv("DATABASE_URL") or cfg.checkpoint_dsn
    if dsn:
        return dsn
    db_name = os.getenv("DB_NAME") or cfg.db_name or "writing_checkpoint_db"
    if cfg.db_host and cfg.db_port and cfg.db_user and cfg.db_password:
        return f"postgresql://{cfg.db_user}:{cfg.db_password}@{cfg.db_host}:{cfg.db_port}/{db_name}"
    return None

async def main():
    """离线初始化脚本：用于手动准备 checkpoint 表结构。"""
    dsn = _resolve_dsn()
    if not dsn:
        raise RuntimeError("Missing checkpoint DSN. Set CHECKPOINT_DSN/DATABASE_URL or config.yaml.")
    try:
        await ensure_checkpoint_tables(dsn)
        print("checkpoint tables ready")
    finally:
        await close_checkpointer()

if __name__ == "__main__":
    asyncio.run(main())
