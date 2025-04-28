# src/gradient/database/session.py

import logging
from pathlib import Path
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional # <--- Add Optional
import datetime # <--- Add datetime

# Import Base and models used by the helper function
from .models import Base, ExperimentRun # <--- Import ExperimentRun
from gradient.core.config import get_setting, PROJECT_ROOT

logger = logging.getLogger(__name__)

# --- Database Configuration ---
DEFAULT_DB_SUBDIR = "database"
DEFAULT_DB_FILENAME = "gradient_data.db"

def get_database_url() -> str:
    db_dir = PROJECT_ROOT / "data" / DEFAULT_DB_SUBDIR
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / DEFAULT_DB_FILENAME
    db_url = f"sqlite+aiosqlite:///{db_path.resolve()}"
    logger.info(f"Using database URL: {db_url}")
    return db_url

DATABASE_URL = get_database_url()

# --- SQLAlchemy Async Engine ---
try:
    async_engine = create_async_engine(
        DATABASE_URL,
        echo=False,
        future=True,
        connect_args={"check_same_thread": False}
    )
except Exception as e:
    logger.critical(f"Failed to create async database engine: {e}", exc_info=True)
    async_engine = None

# --- Async Session Factory ---
AsyncSessionFactory = async_sessionmaker(
    bind=async_engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
    class_=AsyncSession
) if async_engine else None

async def init_db():
    if not async_engine:
        logger.error("Database engine not initialized. Cannot initialize DB.")
        return
    logger.info("Initializing database and creating tables...")
    async with async_engine.begin() as conn:
        try:
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully (if they didn't exist).")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}", exc_info=True)

@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    if AsyncSessionFactory is None:
        logger.error("AsyncSessionFactory not initialized. Cannot get DB session.")
        raise RuntimeError("Database session factory not available.")
    session: AsyncSession = AsyncSessionFactory()
    logger.debug("DB Session created.")
    try:
        yield session
        await session.commit()
        logger.debug("DB Session committed.")
    except Exception as e:
        logger.error(f"Error during DB session: {e}. Rolling back.", exc_info=True)
        await session.rollback()
        raise
    finally:
        await session.close()
        logger.debug("DB Session closed.")


# --- PASTE THE FUNCTION HERE --- vvv
async def update_run_status(run_db_id: int, status: str, error: Optional[str] = None, end_time: Optional[float] = None):
     """Updates the status and end time of an ExperimentRun record."""
     # Note: This function implicitly uses the get_db_session context manager
     try:
         async with get_db_session() as db:
             # Use db.get which is preferred for fetching by primary key
             run = await db.get(ExperimentRun, run_db_id)
             if run:
                 run.status = status
                 if error:
                      # Append error to notes? Or have a dedicated error field?
                      # For now, appending to notes might overwrite existing notes.
                      # Let's just set it directly for simplicity, overwriting old notes.
                      run.user_notes = f"Error: {error}"
                 else:
                      # Clear notes if status is now completed without error? Optional.
                      # run.user_notes = None
                      pass
                 if end_time:
                      run.end_time = datetime.datetime.fromtimestamp(end_time, tz=datetime.timezone.utc)

                 # No need to db.add(run) when modifying an existing tracked object
                 await db.commit() # Commit happens automatically at end of 'with' block now
                 logger.info(f"Updated ExperimentRun ID {run_db_id} status to '{status}'")
             else:
                  logger.warning(f"Could not find ExperimentRun ID {run_db_id} to update status.")
     except Exception as e:
         # Log error but don't rollback here, rollback handled by get_db_session
         logger.error(f"Failed to update DB status for run ID {run_db_id}: {e}", exc_info=True)
# --- PASTE THE FUNCTION HERE --- ^^^